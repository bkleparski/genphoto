#!/usr/bin/env python3
"""GenPhoto — AI photo generation studio (frontend for Stable Diffusion Forge)"""

import base64, collections, hashlib, html, json, mimetypes, os, re, secrets, shutil, sqlite3
import contextlib, threading, time, uuid
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote
import urllib.request, urllib.error

# ── Config ───────────────────────────────────────────────────────────────────
PORT         = int(os.environ.get('GP_PORT', '7862'))
OUTPUTS_DIR  = Path(os.environ.get('GP_OUTPUTS_DIR', '/home/bartek/forge/outputs'))

FORGE_URL    = os.environ.get('GP_FORGE_URL',  'http://localhost:7860').rstrip('/')
COMFY_URL    = os.environ.get('GP_COMFY_URL', 'http://localhost:8189').rstrip('/')
COMFY_OUTPUTS = Path(os.environ.get('GP_COMFY_OUTPUTS', '/home/bartek/comfyui/output'))

# WAN 2.2 T2V-A14B (MoE, GGUF Q3_K_M) — txt2vid
WAN_HIGH_NOISE_UNET = 'Wan2.2-T2V-A14B-HighNoise-Q3_K_M.gguf'
WAN_LOW_NOISE_UNET  = 'Wan2.2-T2V-A14B-LowNoise-Q3_K_M.gguf'
WAN_HIGH_NOISE_LORA = 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors'
WAN_LOW_NOISE_LORA  = 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors'
WAN_CLIP_NAME       = 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
WAN_VAE_NAME        = 'Wan2.1_VAE.safetensors'

KREA2_URL    = os.environ.get('GP_KREA2_URL', 'http://localhost:7870').rstrip('/')
DEEPSEEK_KEY   = os.environ.get('GP_DEEPSEEK_KEY', '')
DEEPSEEK_MODEL = os.environ.get('GP_DEEPSEEK_MODEL', 'deepseek-v4-flash')
AI_PROVIDER    = os.environ.get('GP_AI_PROVIDER', 'deepseek')
OR_KEY         = os.environ.get('GP_OR_KEY', '')
OR_MODEL       = os.environ.get('GP_OR_MODEL', 'nousresearch/hermes-4-405b')
OR_VISION_MODEL = os.environ.get('GP_OR_VISION_MODEL', 'qwen/qwen3-vl-235b-a22b-instruct')
LOCAL_VISION_URL   = os.environ.get('GP_LOCAL_VISION_URL', '').rstrip('/')
LOCAL_VISION_MODEL = os.environ.get('GP_LOCAL_VISION_MODEL', 'qwen2.5vl:32b')
DB_PATH      = Path(os.environ.get('GP_DB_PATH', '/home/bartek/genphoto.db'))
GP_USERNAME  = os.environ.get('GP_USERNAME', 'admin')
GP_PW_HASH   = os.environ.get('GP_PASSWORD_HASH', '')
COOKIE_NAME  = 'gp_sess'
GALLERY_URL  = os.environ.get('GP_GALLERY_URL', 'https://gallery.ebartnet.pl')
PORTAL_URL   = os.environ.get('GP_PORTAL_URL', 'https://images.ebartnet.pl')
PORTAL_KEY   = os.environ.get('GP_PORTAL_KEY', '')

if not GP_PW_HASH:
    raise SystemExit(
        'GP_PASSWORD_HASH not set. '
        'Run: python3 -c "import hashlib; print(hashlib.sha256(b\'pass\').hexdigest())"'
    )

# ── Database ──────────────────────────────────────────────────────────────────
_db_lock = threading.Lock()

@contextlib.contextmanager
def db():
    # UWAGA: sqlite3.Connection jako context manager zarzadza tylko transakcja
    # (commit/rollback), NIE zamyka polaczenia. Bez jawnego close() kazde
    # 'with db() as con' przecieka deskryptor pliku -> po ~1000 zapytaniach
    # proces uderza w limit 'open files' i pada z 'unable to open database file'.
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()

with db() as _c:
    _c.execute('''CREATE TABLE IF NOT EXISTS generations (
        id TEXT PRIMARY KEY, ts INTEGER, description TEXT,
        positive TEXT, negative TEXT, model TEXT,
        sampler TEXT, scheduler TEXT, steps INTEGER, cfg REAL,
        width INTEGER, height INTEGER, seed INTEGER, batch INTEGER,
        preset TEXT, paths TEXT
    )''')
    _c.execute('''CREATE TABLE IF NOT EXISTS edits (
        id TEXT PRIMARY KEY, ts INTEGER,
        description TEXT, positive TEXT, negative TEXT,
        model TEXT, sampler TEXT, scheduler TEXT,
        steps INTEGER, cfg REAL, denoising REAL,
        width INTEGER, height INTEGER, seed INTEGER,
        paths TEXT, source_path TEXT
    )''')
    # Trwałe ustawienia generowania per model — przeżywają restart procesu
    # (w przeciwieństwie do _auto_cache, który jest tylko w pamięci).
    # source: 'manual' (Bartek ręcznie zapisał) | 'civitai' (tag Base Model z Civitai)
    # | 'llm' (zgadnięte przez model językowy, fallback).
    _c.execute('''CREATE TABLE IF NOT EXISTS model_settings (
        model_name TEXT PRIMARY KEY,
        sampler TEXT, scheduler TEXT, steps INTEGER, cfg REAL,
        width INTEGER, height INTEGER, source TEXT, updated_ts INTEGER
    )''')

# ── Sessions ──────────────────────────────────────────────────────────────────
# token -> timestamp wygasniecia. Wczesniej byl to set() bez TTL i bez limitu:
# kazde zalogowanie dokladalo token, ktory zyl az do restartu procesu.
SESSIONS = {}
SESSION_TTL_S = 24 * 3600
_sess_lock = threading.Lock()

def session_new():
    tok = secrets.token_hex(32)
    now = time.time()
    with _sess_lock:
        for t in [t for t, exp in SESSIONS.items() if exp <= now]:
            SESSIONS.pop(t, None)
        SESSIONS[tok] = now + SESSION_TTL_S
    return tok

def session_valid(tok):
    if not tok:
        return False
    with _sess_lock:
        exp = SESSIONS.get(tok)
        if exp is None:
            return False
        if exp <= time.time():
            SESSIONS.pop(tok, None)
            return False
        return True

def session_drop(tok):
    with _sess_lock:
        SESSIONS.pop(tok, None)

# Rate-limit logowania: max LOGIN_MAX_TRIES nieudanych prob z jednego IP
# w oknie LOGIN_WINDOW_S. Bez tego haslo (gole sha256) jest do rozbicia brute-force.
LOGIN_MAX_TRIES = 5
LOGIN_WINDOW_S  = 300
_login_fails = {}            # ip -> [timestamp, ...]
_login_lock  = threading.Lock()

def login_blocked(ip):
    now = time.time()
    with _login_lock:
        hits = [t for t in _login_fails.get(ip, []) if now - t < LOGIN_WINDOW_S]
        _login_fails[ip] = hits
        return len(hits) >= LOGIN_MAX_TRIES

def login_note_fail(ip):
    now = time.time()
    with _login_lock:
        hits = [t for t in _login_fails.get(ip, []) if now - t < LOGIN_WINDOW_S]
        hits.append(now)
        _login_fails[ip] = hits
        # eviction: nie trzymaj wpisow dla IP ktore dawno nie probowaly
        if len(_login_fails) > 500:
            for k in [k for k, v in _login_fails.items() if not v or now - v[-1] > LOGIN_WINDOW_S]:
                _login_fails.pop(k, None)

def login_note_ok(ip):
    with _login_lock:
        _login_fails.pop(ip, None)
_auto_cache = {}  # model_name -> {settings, ts}
AUTO_CACHE_MAX = 300

def auto_cache_put(key, value):
    """Zapis z gornym limitem — bez tego cache rosl z kazdym nowym promptem."""
    if len(_auto_cache) >= AUTO_CACHE_MAX:
        for k in sorted(_auto_cache, key=lambda x: _auto_cache[x].get('ts', 0))[:AUTO_CACHE_MAX // 4]:
            _auto_cache.pop(k, None)
    _auto_cache[key] = value

# ── Jobs ──────────────────────────────────────────────────────────────────────
JOBS = {}
_jlock = threading.Lock()

def job_create():
    jid = uuid.uuid4().hex
    with _jlock:
        JOBS[jid] = {'status': 'queued', 'images': [], 'error': None, 'seed': -1, 't': time.time()}
        if len(JOBS) > 60:
            for k in sorted(JOBS, key=lambda j: JOBS[j]['t'])[:10]:
                del JOBS[k]
    return jid

def job_set(jid, **kw):
    with _jlock:
        if jid in JOBS:
            JOBS[jid].update(kw)

def job_get(jid):
    with _jlock:
        return dict(JOBS.get(jid, {}))

# ── GPU orchestration (RTX 3060 12 GB) ───────────────────────────────────────
# Jeden job GPU na raz + kolejka FIFO (max 1 oczekujący) + pre-flight VRAM.
# Mutex obejmuje pre-flight ORAZ całą generację — atomowo, bez race między
# sprawdzeniem wolnego VRAM a startem joba. Watchdog zwalnia mutex awaryjnie,
# gdy job się zawiesi (threading.Lock — w przeciwieństwie do RLock — może
# zwolnić dowolny wątek; release() na odblokowanym rzuca RuntimeError, łapiemy).
_gpu_lock  = threading.Lock()
_gpu_wlock = threading.Lock()
_gpu_waiters = 0             # liczba jobów oczekujących na mutex
GPU_QUEUE_MAX_WAIT_S = 120   # timeout oczekiwania w kolejce
GPU_DRAIN_MAX_S      = 300   # ile czekac, az zawieszony job odda VRAM
GPU_DRAIN_POLL_S     = 5     # co ile sprawdzac VRAM w trakcie drenazu
GPU_QUEUE_MAX_DEPTH  = 1     # 1 job w toku + 1 w kolejce = "max 2"

# Watchdog per typ joba (s) — powyżej najdłuższego legalnego czasu danego typu.
GPU_WATCHDOG_S = {
    'forge':     900,    # txt2img/img2img/edit/instantid/pose (forge_post timeout 600s + zapas)
    'krea2':     900,    # pętla batch po 1 obrazie
    'zimage':    600,    # 8 kroków turbo + ewentualny reload GGUF
    'wanvideo':  1500,   # wewnętrzny deadline wątku WAN to 900s + ładowanie ~12GB GGUF
    'batch_all': 3600,   # ~20 modeli × (load + generowanie + unload)
    'krea2_batch': 1200, # 4 warianty Krea 2 × (load + 2 zdjęcia)
}

# Minimalny wolny VRAM (MB) przed startem joba.
# "cold" = backend nie działa/nie trzyma modelu; "hot" = backend już jest w VRAM
# (załadowany model nie musi alokować ponownie — wymóg dodatkowej pamięci niższy).
VRAM_NEED_COLD_MB = {'forge': 5000, 'krea2': 4500, 'zimage': 6500, 'wanvideo': 8000}
VRAM_NEED_HOT_MB  = {'forge': 1500, 'krea2': 1500, 'zimage': 2000, 'wanvideo': 3000}

# ── Metryki (ring buffer, endpoint /api/metrics) ─────────────────────────────
_metrics = collections.deque(maxlen=200)
_mlock   = threading.Lock()

def _free_vram_mb():
    import subprocess as _sp
    r = _sp.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
    return int(r.stdout.strip().splitlines()[0].strip())

def metric(event, **kw):
    entry = {'ts': int(time.time()), 'event': event}
    entry.update(kw)
    try:
        entry['vram_free_mb'] = _free_vram_mb()
    except Exception:
        pass
    with _mlock:
        _metrics.append(entry)

def _service_for_backend(backend):
    return {'forge': 'forge', 'krea2': 'krea2',
            'zimage': 'comfyui', 'wanvideo': 'comfyui'}.get(backend)

def _backend_health_url(backend, deep=False):
    if backend == 'forge':
        return FORGE_URL + '/sdapi/v1/sd-models'
    if backend == 'krea2':
        return KREA2_URL + '/health'
    if backend in ('zimage', 'wanvideo'):
        # deep=True → /object_info: po restarcie ComfyUI port otwiera się wcześniej
        # niż kończy się inicjalizacja nodów — bez tego health-check daje false-OK.
        return COMFY_URL + ('/object_info' if deep else '/system_stats')
    return None

def _backend_alive(backend, timeout=4):
    url = _backend_health_url(backend)
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

def _wait_backend_health(backend, timeout_s=120):
    url = _backend_health_url(backend, deep=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': _UA})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

def _ensure_backend_running(backend):
    """Backend nie odpowiada → start serwisu systemd + czekanie na health."""
    if _backend_alive(backend):
        return True
    svc = _service_for_backend(backend)
    if not svc:
        return False
    import subprocess as _sp
    metric('backend_autostart', backend=backend, service=svc)
    try:
        _sp.run(['systemctl', '--user', 'start', svc], timeout=20,
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except Exception:
        metric('backend_autostart_done', backend=backend, ok=False)
        return False
    ok = _wait_backend_health(backend, timeout_s=150)
    metric('backend_autostart_done', backend=backend, ok=ok)
    if ok:
        try:
            # Odśwież cache modeli — auto-start w tle oznaczał, że dropdown
            # na stronie nie miał wpisów tego backendu (np. krea2_oss_turbo).
            _refresh_models_once()
        except Exception:
            pass
    return ok

def _stop_other_backends(backend):
    """Zatrzymaj serwisy niepotrzebne dla tego joba (flaga stop_other_backends)."""
    import subprocess as _sp
    keep = _service_for_backend(backend)
    for svc in ('forge', 'krea2', 'comfyui'):
        if svc == keep:
            continue
        try:
            _sp.run(['systemctl', '--user', 'stop', svc], timeout=15,
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            metric('backend_autostop', service=svc, for_backend=backend)
        except Exception:
            pass

def gpu_preflight(backend):
    """Warunki startu joba. Wywoływane W TRAKCIE trzymania _gpu_lock (atomowo
    ze startem). Zwraca None gdy OK, albo string z komunikatem błędu."""
    hot  = _backend_alive(backend)
    need = (VRAM_NEED_HOT_MB if hot else VRAM_NEED_COLD_MB).get(backend, 4000)
    try:
        free = _free_vram_mb()
    except Exception:
        free = None
    if free is not None and free < need:
        return (f'Za mało wolnego VRAM: {free/1024:.1f} GB wolne, '
                f'wymagane ~{need/1024:.1f} GB. '
                f'Zatrzymaj inne backendy (przycisk ZWOLNIJ) i spróbuj ponownie.')
    if not _ensure_backend_running(backend):
        return f'Backend {backend} nie odpowiada (auto-start serwisu nie powiódł się).'
    return None

def gpu_run(kind, backend, target, params, jid):
    """FIFO + mutex + watchdog dla wszystkich jobów GPU."""
    global _gpu_waiters
    with _gpu_wlock:
        if _gpu_lock.locked() and _gpu_waiters >= GPU_QUEUE_MAX_DEPTH:
            job_set(jid, status='error',
                    error='GPU zajęte, a kolejka pełna (1 job w toku + 1 oczekujący). '
                          'Spróbuj za chwilę.')
            metric('queue_reject', backend=backend)
            return
        _gpu_waiters += 1
    t0 = time.time()
    acquired = _gpu_lock.acquire(timeout=GPU_QUEUE_MAX_WAIT_S)
    with _gpu_wlock:
        _gpu_waiters -= 1
    if not acquired:
        job_set(jid, status='error',
                error=f'Przekroczono czas oczekiwania na GPU ({GPU_QUEUE_MAX_WAIT_S}s).')
        metric('queue_timeout', backend=backend, waited_s=int(time.time() - t0))
        return
    if time.time() - t0 > 1:
        metric('queue_wait', backend=backend, waited_s=int(time.time() - t0))
    # Od tego miejsca trzymamy mutex — pre-flight i job atomowo.
    done_evt  = threading.Event()
    fired_evt = threading.Event()
    # Mutex GPU wolno zwolnic dokladnie raz. Bez tego strazy: watchdog zwalnial
    # mutex, nowy job go przejmowal, a finally starego watku robilo release()
    # na CUDZYM juz mutexie (threading.Lock zwalnia dowolny watek) — i trzeci
    # job wchodzil na zajete GPU.
    released_evt = threading.Event()
    def _release_gpu_once():
        if released_evt.is_set():
            return False
        released_evt.set()
        try:
            _gpu_lock.release()
            return True
        except RuntimeError:
            return False

    watchdog_s = GPU_WATCHDOG_S.get(kind, 900)
    def _watch():
        if done_evt.wait(watchdog_s):
            return
        fired_evt.set()
        job_set(jid, status='error',
                error=f'Watchdog: job przekroczył {watchdog_s}s — zgłoszony jako nieudany.')
        metric('watchdog_fire', backend=backend, kind=kind)
        # Zawieszony watek dalej zyje i trzyma VRAM. Zwolnienie mutexu od razu
        # wpuscilo by drugi job na to samo GPU — czyli OOM. Czekamy, az watek
        # sam skonczy albo VRAM faktycznie wroci.
        need_mb  = VRAM_NEED_COLD_MB.get(backend, 6000)
        deadline = time.time() + GPU_DRAIN_MAX_S
        while time.time() < deadline:
            if done_evt.wait(GPU_DRAIN_POLL_S):
                return          # watek skonczyl — mutex zwolni jego finally
            if _free_vram_mb() >= need_mb:
                metric('watchdog_drain_ok', backend=backend, kind=kind)
                break
        else:
            metric('watchdog_drain_timeout', backend=backend, kind=kind)
        if _release_gpu_once():
            metric('watchdog_release', backend=backend, kind=kind)
    threading.Thread(target=_watch, daemon=True).start()
    try:
        if params.get('stop_other_backends'):
            _stop_other_backends(backend)
        err = gpu_preflight(backend)
        if err:
            job_set(jid, status='error', error=err)
            metric('preflight_fail', backend=backend)
            return
        metric('job_start', backend=backend, kind=kind, model=str(params.get('model', '')))
        tj = time.time()
        target(params, jid)
        if fired_evt.is_set():
            # Wątek odwiesił się PO strzeleniu watchdogu — nie pozwól, aby
            # status 'done' z końca wątku nadpisał awaryjny 'error'.
            job_set(jid, status='error',
                    error=f'Watchdog: job przekroczył {watchdog_s}s — wątek zwolnił się '
                          f'po awaryjnym zwolnieniu mutexa (wynik joba wątpliwy).')
        metric('job_end', backend=backend, kind=kind, duration_s=int(time.time() - tj),
               final_status=job_get(jid).get('status'))
    finally:
        done_evt.set()
        _release_gpu_once()

def gpu_queue_full():
    with _gpu_wlock:
        return _gpu_lock.locked() and _gpu_waiters >= GPU_QUEUE_MAX_DEPTH

# ── Presets ───────────────────────────────────────────────────────────────────
NEG = ('(worst quality:2), (low quality:2), (blurry:1.3), deformed, ugly, extra limbs, '
       'mutated hands, (bad anatomy:1.3), watermark, text, signature, cropped, '
       'painting, artwork, cartoon, anime, 3d render, digital art, illustration, '
       'plastic skin, fake, overexposed, oversaturated, unnatural skin')

PRESETS = [
    {'id':'krea_turbo','name':'Krea 2 Turbo','icon':'&#9889;',
     'model':'krea2_oss_turbo','sampler':'','scheduler':'',
     'steps':6, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'krea_raw',  'name':'Krea 2 Raw',  'icon':'&#127922;',
     'model':'krea2_oss_raw','sampler':'','scheduler':'',
     'steps':28, 'cfg':3.5,'width':1024, 'height':1024, 'batch':1,
     'prefix':'',
     'negative': ''},
    {'id':'krea_moody', 'name':'Krea 2 Moody', 'icon':'&#127917;',
     'model':'krea2_oss_moody','sampler':'','scheduler':'',
     'steps':6, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'krea_krea2gpt', 'name':'Krea 2 GPT', 'icon':'&#129504;',
     'model':'krea2_oss_krea2gpt','sampler':'','scheduler':'',
     'steps':6, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'krea_krealism', 'name':'Krealism', 'icon':'&#128247;',
     'model':'krea2_oss_krealism','sampler':'','scheduler':'',
     'steps':6, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'krea_redcraft', 'name':'RedCraft 3.0', 'icon':'&#128293;',
     'model':'krea2_oss_redcraft','sampler':'','scheduler':'',
     'steps':6, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'zimage_turbo', 'name':'Z-Image Turbo', 'icon':'&#127916;',
     'model':'zimage_turbo','sampler':'','scheduler':'',
     'steps':8, 'cfg':1.0, 'width':832, 'height':1216, 'batch':1,
     'prefix':'',
     'negative': ''},
]

# ── Kategorie i opisy modeli (dla dropdownu — optgroup + tooltip) ─────────────
MODEL_CATEGORIES = ['Fotorealizm', 'Anime / Stylizowane', 'Krea 2', 'Z-Image', 'Inne']

MODEL_INFO = {
    'animagine-xl-3.1': {'cat': 'Anime / Stylizowane', 'desc': 'Anime/ilustracje — żywe kolory, dobre do postaci 2D'},
    'illustriousXL_v01': {'cat': 'Anime / Stylizowane', 'desc': 'Baza anime/ilustracje, mocna stylizacja, dobre do postaci 2D'},
    'BigLove_fp16': {'cat': 'Fotorealizm', 'desc': 'Realistyczne portrety, kobiece sylwetki, miękkie oświetlenie'},
    'CyberRealisticPony_V8': {'cat': 'Fotorealizm', 'desc': 'Baza Pony, realistyczne twarze i ciała, dobra elastyczność promptów'},
    'CyberRealisticXLPlay_V10.0_FP16': {'cat': 'Fotorealizm', 'desc': 'Wariant CyberRealistic pod sceny bardziej stylizowane/NSFW'},
    'epicphotogasm_lastUnicorn': {'cat': 'Fotorealizm', 'desc': 'SD1.5, bardzo realistyczna skóra i portrety, mniejszy VRAM'},
    'epicrealism_pureEvolutionV5': {'cat': 'Fotorealizm', 'desc': 'SD1.5, uniwersalny realizm, dobry balans jakość/szybkość'},
    'epicrealismXL_vxviiCrystalclear': {'cat': 'Fotorealizm', 'desc': 'SDXL, ostry czysty realizm, dobre światło i tekstury skóry'},
    'intorealismUltra_v40': {'cat': 'Fotorealizm', 'desc': 'Najnowsza wersja IntoRealism, wysoka wierność portretów'},
    'intorealismUltra_Asian_v10': {'cat': 'Fotorealizm', 'desc': 'Wariant pod urodę azjatycką, DMD2+Light, wspiera negative prompt'},
    'juggernautXL_ragnarok': {'cat': 'Fotorealizm', 'desc': 'Bardzo wszechstronny — pełne sceny, editorial, nie tylko portrety'},
    'leosamsHelloworldXL_helloworldXL70': {'cat': 'Fotorealizm', 'desc': 'Popularny do portretów, naturalne oświetlenie'},
    'ponyRealism_V23ULTRA': {'cat': 'Fotorealizm', 'desc': 'Baza Pony, realizm + elastyczność promptów w stylu Pony'},
    'pornmaster_proSDXLV7': {'cat': 'Fotorealizm', 'desc': 'NSFW, sceny erotyczne, wysoka szczegółowość'},
    'realisticLustXL_v09': {'cat': 'Fotorealizm', 'desc': 'NSFW, realistyczna skóra i anatomia'},
    'RealVisXL_V5_fp16': {'cat': 'Fotorealizm', 'desc': 'Najostrzejszy detal twarzy spośród modeli SDXL, najlepszy do portretów'},
}

def get_model_info(name):
    info = MODEL_INFO.get(name)
    if info:
        return info
    nl = name.lower()
    if nl.startswith('krea2_'):
        return {'cat': 'Krea 2', 'desc': 'Model Krea 2 (MMDiT + Qwen3-VL)'}
    if nl.startswith('zimage'):
        return {'cat': 'Z-Image', 'desc': 'Z-Image Turbo — szybki, dobre portrety, GGUF przez ComfyUI'}
    return {'cat': 'Inne', 'desc': ''}

def build_model_optgroups(models):
    buckets = {c: [] for c in MODEL_CATEGORIES}
    for m in models:
        name = m.get('model_name', '')
        info = get_model_info(name)
        buckets.get(info['cat'], buckets['Inne']).append((name, info.get('desc', '')))
    out = ''
    for cat in MODEL_CATEGORIES:
        items = buckets[cat]
        if not items:
            continue
        out += f'<optgroup label="{html.escape(cat)}">'
        for name, desc in sorted(items, key=lambda x: x[0].lower()):
            n = html.escape(name)
            d = html.escape(desc)
            out += f'<option value="{n}" title="{d}">{n}</option>'
        out += '</optgroup>'
    return out

# ── Forge API ─────────────────────────────────────────────────────────────────
# Cloudflare (forge-ravnet.ebartnet.pl) blokuje domyślny UA urllib jako bota — udajemy przeglądarkę.
_UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'

def forge_get(path, timeout=10, base_url=None):
    base_url = base_url or FORGE_URL
    try:
        req = urllib.request.Request(f'{base_url}{path}', headers={'User-Agent': _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except:
        return None

def forge_post(path, data, timeout=600, base_url=None):
    base_url = base_url or FORGE_URL
    raw = json.dumps(data).encode()
    req = urllib.request.Request(
        f'{base_url}{path}', data=raw,
        headers={'Content-Type': 'application/json', 'User-Agent': _UA}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

_flux_cache = {}

# Cache list LoRA. Wczesniej kazde zadanie /api/loras robilo pelny glob() po
# katalogu — przy kazdym otwarciu strony i kazdym odswiezeniu listy.
# Kluczem waznosci jest mtime katalogu: dorzucenie/usuniecie pliku je uniewaznia.
_lora_cache = {}          # str(dir) -> (mtime, [nazwy])
_lora_cache_lock = threading.Lock()

def list_lora_names(directory, stem=False):
    key = str(directory)
    try:
        mtime = directory.stat().st_mtime
    except OSError:
        return []
    with _lora_cache_lock:
        hit = _lora_cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        names = sorted(
            (p.stem if stem else p.name)
            for p in directory.glob('*')
            if p.is_file() and p.suffix in ('.safetensors', '.pt', '.ckpt')
            and not p.name.startswith('put_')
        )
    except Exception:
        names = []
    with _lora_cache_lock:
        _lora_cache[key] = (mtime, names)
    return names


def krea_post(path, data, timeout=120):
    """POST to Krea 2 API and return (binary_image_data, headers_dict)"""
    raw = json.dumps(data).encode()
    req = urllib.request.Request(
        f'{KREA2_URL}{path}', data=raw,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)

# Cache modeli Forge — odnawiany w tle co 30s
_models_cache = []
_models_lock  = threading.Lock()
_downloads    = {}   # job_id -> {url,filename,percent,status,error,bytes_done,size,ts}
DOWNLOADS_KEEP    = 40      # ile wpisow trzymac
DOWNLOADS_KEEP_S  = 3600    # jak dlugo trzymac zakonczony wpis (front odpytuje progress)

def downloads_gc():
    """Usun stare zakonczone pobrania. Bez tego _downloads rosl bez konca."""
    now = time.time()
    for k in [k for k, v in list(_downloads.items())
              if v.get('status') in ('done', 'error')
              and now - v.get('ts', now) > DOWNLOADS_KEEP_S]:
        _downloads.pop(k, None)
    if len(_downloads) > DOWNLOADS_KEEP:
        for k in sorted(_downloads, key=lambda j: _downloads[j].get('ts', 0))[:-DOWNLOADS_KEEP]:
            if _downloads[k].get('status') in ('done', 'error'):
                _downloads.pop(k, None)
MODELS_DIR    = Path('/home/bartek/forge/models/Stable-diffusion')
LORAS_DIR     = Path(os.environ.get('GP_LORAS_DIR', '/home/bartek/comfyui/models/loras'))
FORGE_LORAS_DIR = Path('/home/bartek/forge/models/Lora')

def _refresh_models_once(rescan_disk=False):
    # rescan_disk=True wymusza na Forge pełny skan katalogu modeli
    # (/sdapi/v1/refresh-checkpoints) — kosztowne I/O, więc tylko event-driven
    # (upload/download/ręczny refresh/start procesu), NIE w cyklicznym workerze.
    global _models_cache
    data = []
    try:
        if rescan_disk:
            forge_post('/sdapi/v1/refresh-checkpoints', {})
        data = forge_get('/sdapi/v1/sd-models') or []
    except Exception:
        pass
    try:
        krea_health = json.loads(urllib.request.urlopen(f'{KREA2_URL}/health', timeout=3).read())
        for name, exists in (krea_health.get('models') or {}).items():
            if exists:
                data.append({'model_name': f'krea2_{name}', 'title': f'Krea 2 {name}', 'hash': ''})
    except:
        pass
    try:
        urllib.request.urlopen(f'{COMFY_URL}/system_stats', timeout=3).read()
        data.append({'model_name': 'zimage_turbo', 'title': 'Z-Image Turbo', 'hash': ''})
    except:
        pass
    with _models_lock:
        _models_cache = data


# Reguły ustawień per "Base Model" z Civitai — deterministyczne, oparte na
# realnym tagu z metadanych modelu, nie na zgadywaniu z nazwy pliku.
CIVITAI_BASE_MODEL_RULES = [
    (('pony',),                {'sampler': 'DPM++ SDE', 'scheduler': 'Karras', 'steps': 28, 'cfg': 5.5, 'width': 1024, 'height': 1024}),
    (('illustrious',),         {'sampler': 'DPM++ 2M',  'scheduler': 'Karras', 'steps': 30, 'cfg': 7.0, 'width': 1024, 'height': 1024}),
    (('flux',),                {'sampler': 'Euler',     'scheduler': 'Simple', 'steps': 20, 'cfg': 1.0, 'width': 1024, 'height': 1024}),
    (('sdxl', 'xl'),           {'sampler': 'DPM++ SDE', 'scheduler': 'Karras', 'steps': 28, 'cfg': 5.5, 'width': 1024, 'height': 1024}),
    (('sd 1.5', 'sd1.5'),      {'sampler': 'DPM++ SDE', 'scheduler': 'Karras', 'steps': 28, 'cfg': 7.0, 'width': 832,  'height': 1216}),
]

def _civitai_settings_for_base_model(base_model):
    bl = (base_model or '').lower()
    for keys, settings in CIVITAI_BASE_MODEL_RULES:
        if any(k in bl for k in keys):
            return settings
    return None

def _extract_civitai_settings(data):
    """Z odpowiedzi Civitai (model-version) wyciągnij najlepsze dostępne ustawienia:
    najpierw prawdziwe metadane z przykładowych obrazów twórcy (najdokładniejsze),
    z fallbackiem do ogólnej reguły dla danego Base Model."""
    base_model = data.get('baseModel', '')
    rule = _civitai_settings_for_base_model(base_model) or {}

    samplers, schedulers, steps_list, cfg_list = [], [], [], []
    KNOWN_SCHEDULERS = ('Karras', 'Exponential', 'Normal', 'Simple', 'SGM Uniform')
    for img in (data.get('images') or [])[:10]:
        meta = img.get('meta') or {}
        if isinstance(meta.get('steps'), (int, float)):
            steps_list.append(meta['steps'])
        if isinstance(meta.get('cfgScale'), (int, float)):
            cfg_list.append(meta['cfgScale'])
        samp = (meta.get('sampler') or '').strip()
        if samp:
            parts = samp.rsplit(' ', 1)
            if len(parts) == 2 and parts[1] in KNOWN_SCHEDULERS:
                samplers.append(parts[0]); schedulers.append(parts[1])
            else:
                samplers.append(samp)

    def _median(lst):
        if not lst: return None
        s = sorted(lst)
        return s[len(s) // 2]

    def _mode(lst):
        return max(set(lst), key=lst.count) if lst else None

    if not (steps_list or cfg_list) and not rule:
        return None

    settings = {
        'sampler':   _mode(samplers) or rule.get('sampler', 'Euler a'),
        'scheduler': _mode(schedulers) or rule.get('scheduler', 'Karras'),
        'steps':     int(_median(steps_list) or rule.get('steps', 20)),
        'cfg':       float(_median(cfg_list) or rule.get('cfg', 5.5)),
        'width':     rule.get('width', 1024),
        'height':    rule.get('height', 1024),
    }
    return {'settings': settings, 'base_model': base_model}

def _save_model_settings(model_name, settings, source):
    """Zapisz ustawienia trwale — nigdy nie nadpisuj ręcznej (manual) poprawki Bartka."""
    with _db_lock:
        with db() as con:
            row = con.execute('SELECT source FROM model_settings WHERE model_name=?', (model_name,)).fetchone()
            if row and row['source'] == 'manual' and source != 'manual':
                return
            con.execute('''INSERT INTO model_settings (model_name, sampler, scheduler, steps, cfg, width, height, source, updated_ts)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(model_name) DO UPDATE SET
                                sampler=excluded.sampler, scheduler=excluded.scheduler,
                                steps=excluded.steps, cfg=excluded.cfg,
                                width=excluded.width, height=excluded.height,
                                source=excluded.source, updated_ts=excluded.updated_ts''',
                        (model_name, settings['sampler'], settings['scheduler'], settings['steps'],
                         settings['cfg'], settings['width'], settings['height'], source, int(time.time())))

def _get_saved_model_settings(model_name):
    """Trwałe ustawienia dla modelu (manual/civitai/llm), albo None jeśli brak."""
    with db() as con:
        row = con.execute(
            'SELECT sampler, scheduler, steps, cfg, width, height, source FROM model_settings WHERE model_name=?',
            (model_name,)
        ).fetchone()
    if not row:
        return None
    return {
        'settings': {k: row[k] for k in ('sampler', 'scheduler', 'steps', 'cfg', 'width', 'height')},
        'source': row['source'],
    }

def _fetch_civitai_version_json(url_or_version_id):
    """Pobierz JSON model-version z Civitai — po pełnym URL pobierania albo po samym ID wersji."""
    m = re.search(r'/api/download/models/(\d+)', url_or_version_id)
    version_id = m.group(1) if m else url_or_version_id
    req = urllib.request.Request(
        f'https://civitai.com/api/v1/model-versions/{version_id}',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def _fetch_civitai_settings_by_hash(sha256):
    """Znajdź model po SHA256 pliku (dokładne trafienie w konkretną wersję) i wyciągnij ustawienia."""
    req = urllib.request.Request(
        f'https://civitai.com/api/v1/model-versions/by-hash/{sha256}',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return _extract_civitai_settings(data)

def _seed_settings_from_civitai_url(url, model_name):
    """Po pobraniu modelu z civitai(.red) — wyciągnij prawdziwe ustawienia i zapisz je trwale."""
    try:
        data = _fetch_civitai_version_json(url)
        result = _extract_civitai_settings(data)
        if result:
            _save_model_settings(model_name, result['settings'], 'civitai')
    except Exception:
        pass

# Allowlist zrodel pobierania. Bez niej /api/download-model przyjmowal dowolny
# URL (takze http://127.0.0.1:7860/... — SSRF do uslug na loopbacku).
DOWNLOAD_ALLOWED_HOSTS = (
    'civitai.com', 'civitai.red',
    'huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.huggingface.co',
)
DOWNLOAD_MAX_BYTES = 30 * 1024 ** 3   # 30 GB — powyzej najwiekszych sensownych checkpointow
DOWNLOAD_MIN_FREE  = 5 * 1024 ** 3    # zostaw 5 GB zapasu na dysku

def download_url_rejected(url):
    """Zwraca komunikat bledu, jesli URL nie nadaje sie do pobrania. None = OK."""
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
    except Exception:
        return 'nieprawidłowy URL'
    if u.scheme not in ('http', 'https'):
        return 'dozwolone tylko http/https'
    host = (u.hostname or '').lower()
    if not any(host == h or host.endswith('.' + h) for h in DOWNLOAD_ALLOWED_HOSTS):
        return f'niedozwolony host: {host or "?"} (dozwolone: civitai, huggingface)'
    return None

def _download_model_thread(job_id, url, dest_dir=None, refresh=True):
    import time as _t
    dest_dir = dest_dir or MODELS_DIR
    d = _downloads[job_id]
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://civitai.red/'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            cd = resp.headers.get('Content-Disposition', '')
            fname = ''
            for part in cd.split(';'):
                part = part.strip()
                if part.startswith('filename=') or part.startswith('filename*='):
                    fname = part.split('=', 1)[1].strip().strip('"').split("''")[-1]
                    break
            if not fname:
                fname = url.split('?')[0].rstrip('/').split('/')[-1]
            if not (fname.endswith('.safetensors') or fname.endswith('.ckpt') or fname.endswith('.pt')):
                fname += '.safetensors'
            fname = fname.replace('/', '_').replace('\\', '_')
            dest  = dest_dir / fname
            d['filename'] = fname
            d['status']   = 'downloading'
            size = int(resp.headers.get('Content-Length', 0) or 0)
            if size and size > DOWNLOAD_MAX_BYTES:
                raise RuntimeError(f'plik za duzy: {size / 1024**3:.1f} GB '
                                   f'(limit {DOWNLOAD_MAX_BYTES / 1024**3:.0f} GB)')
            free = shutil.disk_usage(dest_dir).free
            if size and free - size < DOWNLOAD_MIN_FREE:
                raise RuntimeError(f'za malo miejsca: wolne {free / 1024**3:.1f} GB, '
                                   f'plik {size / 1024**3:.1f} GB')
            d['size'] = size
            done = 0
            chunk = 1024 * 256
            with open(dest, 'wb') as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf: break
                    f.write(buf)
                    done += len(buf)
                    # serwer moze klamac w Content-Length albo go nie podac
                    if done > DOWNLOAD_MAX_BYTES:
                        raise RuntimeError('przekroczono limit rozmiaru pobierania')
                    d['bytes_done'] = done
                    d['percent'] = int(done * 100 / size) if size else -1
        d['percent'] = 100
        d['status']  = 'done'
        if refresh:
            _refresh_models_once(rescan_disk=True)
            model_name = fname.rsplit('.', 1)[0]
            _seed_settings_from_civitai_url(url, model_name)
    except Exception as e:
        d['status'] = 'error'
        d['error']  = str(e)
        if d.get('filename'):
            p = dest_dir / d['filename']
            if p.exists(): p.unlink()

def _models_cache_worker():
    import time as _time
    while True:
        _time.sleep(300)  # co 5 min wystarczy — skan dysku robią eventy (upload/download)
        try:
            if _gpu_lock.locked():
                continue  # backpressure: nie dokładać I/O na Forge podczas generowania
            _refresh_models_once()
        except Exception:
            pass


# ── InsightFace auto-crop ──────────────────────────────────────────────────
_face_app = None
_face_app_lock = threading.Lock()

def _get_face_app():
    global _face_app
    with _face_app_lock:
        if _face_app is None:
            try:
                import insightface
                app = insightface.app.FaceAnalysis(
                    name='antelopev2',
                    root='/home/bartek/forge/models/insightface',
                    providers=['CPUExecutionProvider']
                )
                app.prepare(ctx_id=0, det_size=(1280, 1280))
                _face_app = app
            except Exception as e:
                _face_app = None
        return _face_app

def _crop_face_with_status(image_b64: str, padding: float = 0.40):
    """Zwraca (face_b64, detected). Jeśli brak twarzy: (oryginał, False)."""
    try:
        import numpy as np, cv2
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        app = _get_face_app()
        if app is None:
            return image_b64, False
        faces = app.get(img)
        if not faces:
            return image_b64, False
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        fw, fh = x2 - x1, y2 - y1
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        cropped = img[cy1:cy2, cx1:cx2]
        ch, cw = cropped.shape[:2]
        if cw < 512 or ch < 512:
            scale = 512 / min(cw, ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buf).decode(), True
    except Exception:
        return image_b64, False

def auto_crop_face(image_b64: str, padding: float = 0.40) -> str:
    face_b64, _ = _crop_face_with_status(image_b64, padding)
    return face_b64

def is_flux_checkpoint(filename):
    if filename in _flux_cache:
        return _flux_cache[filename]
    try:
        with open(filename, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(n))
        result = any('double_blocks' in k for k in header.keys())
    except Exception:
        result = False
    if len(_flux_cache) > 500:
        _flux_cache.clear()
    _flux_cache[filename] = result
    return result

def resolve_model(model_name):
    with _models_lock:
        models = list(_models_cache)
    for m in models:
        if m.get('model_name') == model_name:
            return m['title'], m.get('filename', '')
    return model_name, ''

FLUX_ADDITIONAL_MODULES = [
    '/home/bartek/forge/models/text_encoder/clip_l.safetensors',
    '/home/bartek/forge/models/text_encoder/t5xxl_fp8_e4m3fn.safetensors',
    '/home/bartek/forge/models/VAE/ae.safetensors',
]

# Modele z uszkodzonym/niekompatybilnym wbudowanym VAE (błąd twórcy przy mergu —
# np. przypadkowo wpięty VAE 16-kanałowy zamiast standardowego 4-kanałowego SDXL).
# Wymuszamy zewnętrzny, poprawny plik VAE. Klucz: nazwa modelu bez rozszerzenia, lowercase.
BROKEN_VAE_MODELS = {
    'gonzalomoxlfluxpony_v30unityxldmd': 'sdxl_vae.safetensors',
}

def model_override_settings(title, filename=''):
    name = title.split(' [')[0]
    s = {'sd_model_checkpoint': name, 'sd_vae': '', 'forge_additional_modules': []}
    # Wykrywanie Flux WYŁĄCZNIE po zawartości pliku (klucze double_blocks) — nazwa modelu
    # bywa myląca (np. "gonzalomoXLFluxPony" to zwykły SDXL mimo "flux" w nazwie).
    flux = bool(filename) and is_flux_checkpoint(filename)
    if flux:
        s['forge_additional_modules'] = FLUX_ADDITIONAL_MODULES
    override_vae = BROKEN_VAE_MODELS.get(name.lower().replace('.safetensors', ''))
    if override_vae:
        s['sd_vae'] = override_vae
    return s

def build_adetailer_scripts(params):
    model = params.get('adetailer_model')
    if not model:
        return None
    return {
        'ADetailer': {
            'args': [
                True, False,
                {
                    'ad_model': model,
                    'ad_prompt': params.get('adetailer_prompt', ''),
                    'ad_denoising_strength': float(params.get('adetailer_denoise', 0.4)),
                },
            ]
        }
    }

def _apply_lora_tags(prompt, loras):
    """Dokleja tagi <lora:name:strength> do prompta Forge (A1111 extra-networks syntax)."""
    for l in (loras or []):
        name = (l.get('name') or '').strip()
        if not name:
            continue
        name = name.rsplit('.', 1)[0] if name.lower().endswith(('.safetensors', '.pt', '.ckpt')) else name
        strength = float(l.get('strength', 1.0))
        prompt += f' <lora:{name}:{strength}>'
    return prompt

def forge_generate_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title, fname = resolve_model(params['model'])
        prompt = _apply_lora_tags(params['positive'], params.get('loras'))
        body = {
            'prompt':          prompt,
            'negative_prompt': params['negative'],
            'sampler_name':    params['sampler'],
            'scheduler':       params.get('scheduler', 'Karras'),
            'steps':           int(params['steps']),
            'cfg_scale':       float(params['cfg']),
            'width':           int(params['width']),
            'height':          int(params['height']),
            'seed':            int(params.get('seed', -1)),
            'batch_size':      int(params.get('batch', 1)),
            'n_iter':          1,
            'override_settings': model_override_settings(title, fname),
            'override_settings_restore_afterwards': True,
            'save_images': False,
        }
        adetailer_scripts = build_adetailer_scripts(params)
        if adetailer_scripts:
            body['alwayson_scripts'] = adetailer_scripts
        data = forge_post('/sdapi/v1/txt2img', body)
        imgs_b64 = data.get('images', [])
        info     = json.loads(data.get('info', '{}'))
        seed     = info.get('seed', -1)
        seeds    = info.get('all_seeds', [seed] * len(imgs_b64))

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        paths = []
        for i, b64 in enumerate(imgs_b64):
            s    = seeds[i] if i < len(seeds) else seed
            name = f'gp_{ts}_{i:02d}_s{s}.png'
            (out_dir / name).write_bytes(base64.b64decode(b64))
            paths.append(f'genphoto/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description',''),
                     params['positive'], params['negative'], params['model'],
                     params['sampler'], params.get('scheduler','Karras'),
                     int(params['steps']), float(params['cfg']),
                     int(params['width']), int(params['height']),
                     seed, int(params.get('batch',1)),
                     params.get('preset','custom'), json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def forge_batch_all_models_thread(params, jid):
    """Generuj ten sam prompt sekwencyjnie na wszystkich modelach Forge —
    jeden model na raz (załaduj > wygeneruj > zwolnij VRAM > następny), żeby
    uniknąć OOM na 12 GB VRAM. Ustawienia (sampler/steps/cfg/wymiary) per model
    biorą się z trwałej bazy model_settings (Etap 1), z fallbackiem do wartości
    z formularza dla modeli bez zapisanych ustawień.
    """
    try:
        job_set(jid, status='generating')
        with _models_lock:
            # Pomijamy domyślnie modele architektury Flux (np. flux1-dev-fp8) —
            # znacznie wolniejsze od SDXL (minuty zamiast sekund na obraz),
            # rozwałałyby czas całego przebiegu "wszystkie modele".
            all_models = [
                m for m in _models_cache
                if m.get('filename') and not is_flux_checkpoint(m['filename'])
            ]
        total = len(all_models)
        if total == 0:
            raise RuntimeError('Brak modeli Forge do wygenerowania')

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        gid = params.get('gen_id', uuid.uuid4().hex)
        all_paths = []
        errors = []

        for idx, m in enumerate(all_models):
            model_name = m['model_name']
            title = m['title']
            fname = m.get('filename', '')

            job_set(jid, status='generating',
                    progress={'current': idx + 1, 'total': total, 'model': model_name})

            saved = _get_saved_model_settings(model_name)
            if saved:
                s = saved['settings']
                sampler, scheduler = s['sampler'], s['scheduler']
                steps, cfg = s['steps'], s['cfg']
                width, height = s['width'], s['height']
            else:
                sampler   = params.get('sampler') or 'DPM++ SDE'
                scheduler = params.get('scheduler') or 'Karras'
                steps     = int(params.get('steps', 28))
                cfg       = float(params.get('cfg', 5.5))
                width     = int(params.get('width', 1024))
                height    = int(params.get('height', 1024))

            try:
                batch_prompt = _apply_lora_tags(params['positive'], params.get('loras'))
                body = {
                    'prompt':          batch_prompt,
                    'negative_prompt': params.get('negative', ''),
                    'sampler_name':    sampler,
                    'scheduler':       scheduler,
                    'steps':           steps,
                    'cfg_scale':       cfg,
                    'width':           width,
                    'height':          height,
                    'seed':            int(params.get('seed', -1)),
                    'batch_size':      1,
                    'n_iter':          1,
                    'override_settings': model_override_settings(title, fname),
                    'override_settings_restore_afterwards': False,
                    'save_images': False,
                }
                data = forge_post('/sdapi/v1/txt2img', body, timeout=180)
                imgs_b64 = data.get('images', [])
                info = json.loads(data.get('info', '{}'))
                seed = info.get('seed', -1)

                paths = []
                for i, b64 in enumerate(imgs_b64):
                    name = f'gp_batch_{ts}_{idx:02d}_{model_name}_s{seed}.png'
                    (out_dir / name).write_bytes(base64.b64decode(b64))
                    p = f'genphoto/{today}/{name}'
                    paths.append(p)
                    all_paths.append(p)

                if paths:
                    with _db_lock:
                        with db() as con:
                            con.execute(
                                'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (uuid.uuid4().hex, int(time.time()), params.get('description', ''),
                                 params['positive'], params.get('negative', ''), model_name,
                                 sampler, scheduler, steps, cfg, width, height,
                                 seed, 1, 'batch_all_models', json.dumps(paths))
                            )
            except Exception as model_err:
                errors.append(f'{model_name}: {model_err}')
            finally:
                # Zwolnij VRAM przed załadowaniem kolejnego modelu.
                try:
                    forge_post('/sdapi/v1/unload-checkpoint', {}, timeout=15)
                except Exception:
                    pass

        status = 'done' if all_paths else 'error'
        job_set(jid, status=status, images=all_paths, seed=-1, gen_id=gid,
                error=('; '.join(errors) if errors else None))
    except Exception as e:
        job_set(jid, status='error', error=str(e))

KREA2_BATCH_VARIANTS = ['oss_moody', 'oss_krea2gpt', 'oss_krealism', 'oss_redcraft']

def krea_batch_variants_thread(params, jid):
    """Generuj ten sam prompt sekwencyjnie na wszystkich wariantach Krea 2
    (Moody -> GPT -> Krealism -> RedCraft), po 2 zdjecia na wariant. Krea2 API
    samo zwalnia VRAM poprzedniego checkpointu przy zmianie (patrz get_pipeline
    w krea-2/api.py) — w przeciwienstwie do Forge nie trzeba tu wolac osobnego
    unload miedzy modelami."""
    try:
        job_set(jid, status='generating')
        total = len(KREA2_BATCH_VARIANTS)
        images_per_variant = 2

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        gid = params.get('gen_id', uuid.uuid4().hex)
        base_seed = int(params.get('seed', 0)) or secrets.randbelow(2**31)
        steps = int(params.get('steps') or 6)
        cfg   = float(params.get('cfg') or 0.0)
        width  = min(int(params.get('width', 1024)), 2048)
        height = min(int(params.get('height', 1024)), 2048)

        all_paths = []
        errors = []

        for idx, checkpoint in enumerate(KREA2_BATCH_VARIANTS):
            model_name = f'krea2_{checkpoint}'
            job_set(jid, status='generating',
                    progress={'current': idx + 1, 'total': total, 'model': model_name})
            try:
                paths = []
                first_seed = None
                for i in range(images_per_variant):
                    body = {
                        'prompt': params['positive'],
                        'negative_prompt': params.get('negative', '') or '',
                        'checkpoint': checkpoint,
                        'width': width, 'height': height,
                        'num_images': 1,
                        'seed': base_seed + idx * 100 + i,
                        'steps': steps, 'cfg': cfg,
                    }
                    img_data, headers = krea_post('/generate', body, timeout=600)
                    seed = headers.get('X-Seed', str(body['seed']))
                    if first_seed is None:
                        first_seed = seed
                    name = f'krea_batch_{ts}_{idx:02d}_{checkpoint}_{i:02d}_s{seed}.png'
                    (out_dir / name).write_bytes(img_data)
                    p = f'genphoto/{today}/{name}'
                    paths.append(p)
                    all_paths.append(p)

                if paths:
                    with _db_lock:
                        with db() as con:
                            con.execute(
                                'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (uuid.uuid4().hex, int(time.time()), params.get('description', ''),
                                 params['positive'], params.get('negative', ''), model_name,
                                 '', '', steps, cfg, width, height,
                                 int(first_seed), images_per_variant,
                                 'krea2_batch_variants', json.dumps(paths))
                            )
            except Exception as variant_err:
                errors.append(f'{model_name}: {variant_err}')

        status = 'done' if all_paths else 'error'
        job_set(jid, status=status, images=all_paths, seed=-1, gen_id=gid,
                error=('; '.join(errors) if errors else None))
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def krea_generate_thread(params, jid):
    """Generate using Krea 2 API."""
    try:
        job_set(jid, status='generating')
        
        # Determine checkpoint — model_name z listy zawsze ma postac 'krea2_<checkpoint>'
        # (patrz _refresh_models_once, ktore buduje ta liste wprost z /health Krea2 API,
        # wiec kazdy checkpoint zgloszony przez API automatycznie tu dziala, bez
        # koniecznosci recznego dopisywania kolejnych 'elif' przy nowych modelach).
        model = (params.get('model') or '').strip()
        if model.startswith('krea2_'):
            checkpoint = model[len('krea2_'):]
        else:
            raise RuntimeError(
                f'Nieznany model Krea 2: {model!r}. '
                f'Odśwież stronę (lista modeli mogła być nieaktualna) i wybierz '
                f'jeden z modeli Krea 2 z listy.'
            )

        # Krea 2 parameters — num_images always 1: a batched forward pass
        # doubles VAE-decode activation memory and OOMs on 12 GB VRAM once
        # the quantized model is loaded, so images are generated one at a time.
        body = {
            'prompt': params['positive'],
            'negative_prompt': params.get('negative', '') or '',
            'checkpoint': checkpoint,
            'width': min(int(params['width']), 2048),
            'height': min(int(params['height']), 2048),
            'num_images': 1,
            'seed': int(params.get('seed', 0)),
        }

        # Steps: default 8 for turbo, 52 for raw
        if 'steps' in params and params['steps']:
            body['steps'] = int(params['steps'])

        # CFG: default 0.0 for turbo, 3.5 for raw
        if 'cfg' in params and params['cfg']:
            body['cfg'] = float(params['cfg'])

        today = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        base_seed = body['seed']
        batch = int(params.get('batch', 1))
        paths = []
        first_seed = None

        for i in range(batch):
            body['seed'] = base_seed + i
            # timeout 600s: pierwsze wywołanie po starcie krea2 ładuje pipeline
            # z dysku + kwantyzacja bnb — przy współdzielonym GPU to ~4 min,
            # domyślne 120s zabijało nawet udane generowanie.
            img_data, headers = krea_post('/generate', body, timeout=600)
            seed = headers.get('X-Seed', str(body['seed']))
            if first_seed is None:
                first_seed = seed
            name = f'krea_{ts}_{i:02d}_s{seed}.png'
            (out_dir / name).write_bytes(img_data)
            paths.append(f'genphoto/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description',''),
                     params['positive'], params.get('negative',''), model,
                     params.get('sampler',''), params.get('scheduler',''),
                     int(body.get('steps',8)), float(body.get('cfg',0.0)),
                     int(body['width']), int(body['height']),
                     int(first_seed), batch,
                     params.get('preset','krea'), json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=first_seed, gen_id=gid)

    except Exception as e:
        job_set(jid, status='error', error=str(e))

def forge_img2img_ref_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title, fname = resolve_model(params['model'])
        data  = forge_post('/sdapi/v1/img2img', {
            'init_images':        [params['init_image']],
            'denoising_strength': float(params.get('denoising', 0.65)),
            'mask_blur':          0,
            'inpainting_fill':    0,
            'prompt':             params['positive'],
            'negative_prompt':    params['negative'],
            'sampler_name':       params['sampler'],
            'scheduler':          params.get('scheduler', 'Karras'),
            'steps':              int(params['steps']),
            'cfg_scale':          float(params['cfg']),
            'width':              int(params['width']),
            'height':             int(params['height']),
            'seed':               int(params.get('seed', -1)),
            'batch_size':         int(params.get('batch', 1)),
            'n_iter':             1,
            'override_settings':  model_override_settings(title, fname),
            'override_settings_restore_afterwards': True,
            'save_images':        False,
        })
        imgs_b64 = data.get('images', [])
        info     = json.loads(data.get('info', '{}'))
        seed     = info.get('seed', -1)
        seeds    = info.get('all_seeds', [seed] * len(imgs_b64))

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        paths = []
        for i, b64 in enumerate(imgs_b64):
            s    = seeds[i] if i < len(seeds) else seed
            name = f'gp_i2i_{ts}_{i:02d}_s{s}.png'
            (out_dir / name).write_bytes(base64.b64decode(b64))
            paths.append(f'genphoto/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description', ''),
                     params['positive'], params['negative'], params['model'],
                     params['sampler'], params.get('scheduler', 'Karras'),
                     int(params['steps']), float(params['cfg']),
                     int(params['width']), int(params['height']),
                     seed, int(params.get('batch', 1)),
                     'img2img', json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def forge_edit_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title, fname = resolve_model(params['model'])
        mask  = params.get('mask_b64') or None
        data  = forge_post('/sdapi/v1/img2img', {
            'init_images':        [params['image_b64']],
            'mask':               mask,
            'denoising_strength': float(params.get('denoising', 0.5)),
            'mask_blur':          4,
            'inpainting_fill':    1 if mask else 0,
            'prompt':             params['positive'],
            'negative_prompt':    params['negative'],
            'sampler_name':       params['sampler'],
            'scheduler':          params.get('scheduler', 'Karras'),
            'steps':              int(params['steps']),
            'cfg_scale':          float(params['cfg']),
            'width':              int(params['width']),
            'height':             int(params['height']),
            'seed':               int(params.get('seed', -1)),
            'batch_size':         1,
            'n_iter':             1,
            'override_settings':  model_override_settings(title, fname),
            'override_settings_restore_afterwards': True,
            'save_images':        False,
        })
        imgs_b64 = data.get('images', [])
        info     = json.loads(data.get('info', '{}'))
        seed     = info.get('seed', -1)

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto_edits' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        paths = []
        for i, b64 in enumerate(imgs_b64):
            name = f'gpe_{ts}_{i:02d}_s{seed}.png'
            (out_dir / name).write_bytes(base64.b64decode(b64))
            paths.append(f'genphoto_edits/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO edits VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description', ''),
                     params['positive'], params['negative'], params['model'],
                     params['sampler'], params.get('scheduler', 'Karras'),
                     int(params['steps']), float(params['cfg']),
                     float(params.get('denoising', 0.5)),
                     int(params['width']), int(params['height']),
                     seed, json.dumps(paths),
                     params.get('source_path', ''))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def _find_recent_video(ts_before):
    """Znajdź MP4 wygenerowany przez AnimateDiff po podanym timestampie."""
    search_dirs = [
        OUTPUTS_DIR / 'txt2img-images' / 'AnimateDiff',
        OUTPUTS_DIR / 'img2img-images' / 'AnimateDiff',
        OUTPUTS_DIR / 'AnimateDiff',
    ]
    candidates = []
    for d in search_dirs:
        if d.exists():
            for f in d.rglob('*.mp4'):
                try:
                    if f.stat().st_mtime >= ts_before:
                        candidates.append(f)
                except OSError:
                    pass
    if not candidates:
        for f in OUTPUTS_DIR.rglob('*.mp4'):
            try:
                if f.stat().st_mtime >= ts_before:
                    candidates.append(f)
            except OSError:
                pass
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None

# ── ComfyUI helpers ───────────────────────────────────────────────────────────
def comfy_post(path, data, timeout=30):
    raw = json.dumps(data).encode()
    req = urllib.request.Request(
        f'{COMFY_URL}{path}', data=raw,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def comfy_get(path, timeout=10):
    with urllib.request.urlopen(f'{COMFY_URL}{path}', timeout=timeout) as r:
        return json.loads(r.read())

ZIMAGE_TEMPLATE = json.loads(r'''
{
  "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "z_image_turbo-Q8_0.gguf"}},
  "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "Qwen3-4B-UD-Q5_K_XL.gguf", "type": "lumina2"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
  "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
  "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
  "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
  "7": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["1", 0], "add_noise": "enable", "noise_seed": 0, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 10000, "return_with_leftover_noise": "disable", "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
  "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
  "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "gp_zimage"}}
}
''')

COMFY_UNET_DIR = Path(os.environ.get('GP_COMFY_UNET_DIR', '/home/bartek/comfyui/models/unet'))

def _zimage_unet_name(params):
    """Kwantyzacja GGUF Z-Image: payload 'quant' (np. Q5_K_M — A/B test) >
    env GP_ZIMAGE_QUANT > domyślne Q8_0. Walidacja: plik musi istnieć na dysku."""
    quant = (params.get('quant') or os.environ.get('GP_ZIMAGE_QUANT', 'Q8_0')).strip()
    # tylko bezpieczne znaki — nazwa trafia wprost do ścieżki pliku
    if not re.fullmatch(r'[A-Za-z0-9_]+', quant):
        raise RuntimeError(f'Nieprawidłowa nazwa kwantyzacji: {quant!r}')
    unet = f'z_image_turbo-{quant}.gguf'
    if not (COMFY_UNET_DIR / unet).exists():
        raise RuntimeError(f'Brak pliku {unet} w {COMFY_UNET_DIR} '
                           f'(dostępne: {sorted(p.name for p in COMFY_UNET_DIR.glob("z_image_turbo-*.gguf"))})')
    return unet

def zimage_generate_thread(params, jid):
    """Generate using Z-Image Turbo via local ComfyUI (GGUF, port 8189)."""
    try:
        job_set(jid, status='generating')
        import copy as _copy

        unet_name = _zimage_unet_name(params)
        width  = int(params['width'])
        height = int(params['height'])
        steps  = int(params.get('steps', 8)) or 8
        cfg    = float(params.get('cfg', 1.0)) or 1.0
        batch  = int(params.get('batch', 1))
        seed   = int(params.get('seed', -1))
        if seed < 0:
            seed = secrets.randbelow(2**32)
        ts = int(time.time() * 1000)

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        loras = params.get('loras') or []

        # Jeden obraz na raz — batch>1 w jednym forward passie podwaja pamięć
        # aktywacji przy dekodowaniu VAE i grozi OOM na 12 GB VRAM (patrz Krea 2).
        for i in range(batch):
            graph = _copy.deepcopy(ZIMAGE_TEMPLATE)
            graph['1']['inputs']['unet_name']      = unet_name
            graph['4']['inputs']['text']            = params['positive']
            graph['5']['inputs']['text']             = params.get('negative', '')
            graph['6']['inputs']['width']            = width
            graph['6']['inputs']['height']           = height
            graph['7']['inputs']['noise_seed']       = seed + i
            graph['7']['inputs']['steps']            = steps
            graph['7']['inputs']['cfg']              = cfg
            graph['9']['inputs']['filename_prefix']  = f'gp_zimage_{ts}'

            # Wpinamy LoRA łańcuchowo między UnetLoaderGGUF (node "1") a samplerem —
            # każda kolejna LoRA bierze MODEL z wyjścia poprzedniej.
            model_ref = ['1', 0]
            for li, lora in enumerate(loras):
                node_id = f'lora{li}'
                graph[node_id] = {
                    'class_type': 'LoraLoaderModelOnly',
                    'inputs': {
                        'model': model_ref,
                        'lora_name': lora['name'],
                        'strength_model': float(lora.get('strength', 1.0)),
                    }
                }
                model_ref = [node_id, 0]
            graph['7']['inputs']['model'] = model_ref

            resp = comfy_post('/prompt', {'prompt': graph}, timeout=30)
            prompt_id = resp['prompt_id']

            deadline = time.time() + 300
            history = None
            while time.time() < deadline:
                h = comfy_get(f'/history/{prompt_id}', timeout=15)
                if prompt_id in h and h[prompt_id].get('outputs'):
                    history = h[prompt_id]
                    break
                time.sleep(2)
            if history is None:
                raise RuntimeError('Timeout oczekiwania na wygenerowanie obrazu (Z-Image)')

            images_meta = history['outputs'].get('9', {}).get('images', [])
            if not images_meta:
                raise RuntimeError('ComfyUI nie zwróciło żadnego obrazu (Z-Image)')

            img  = images_meta[0]
            src  = COMFY_OUTPUTS / img.get('subfolder', '') / img['filename']
            name = f'zimage_{ts}_{i:02d}_s{seed+i}.png'
            (out_dir / name).write_bytes(src.read_bytes())
            paths.append(f'genphoto/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description',''),
                     params['positive'], params.get('negative',''),
                     f'z_image_turbo_{unet_name.replace("z_image_turbo-", "").replace(".gguf", "")}',
                     'euler', 'simple', steps, cfg, width, height,
                     seed, batch, params.get('preset','zimage'), json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))


def build_wan_t2v_graph(positive, negative, width, height, length, seed, fps, filename_prefix,
                         extra_lora_high=None, extra_lora_low=None):
    """Graf ComfyUI dla WAN 2.2 T2V-A14B (MoE: high+low noise expert) w wariancie
    lightx2v 4-step (szybki dystylowany sampling — niezbędny na 12GB VRAM / RTX 3060,
    pelny 20-krokowy wariant trwałby wielokrotnie dłużej). Odtworzone z oficjalnego
    szablonu ComfyUI (video_wan2_2_14B_t2v.json), z UNETLoader zamienionym na
    UnetLoaderGGUF (kwantyzacja Q3_K_M zamiast fp8_scaled — mniejszy footprint VRAM).
    extra_lora_high/extra_lora_low: opcjonalne dodatkowe LoRA (np. efekt ruchu z Civitai),
    dolaczane w lancuchu ZA lightx2v — kazdy WAN 2.2 LoRA wymaga oddzielnego pliku dla
    high i low noise expertu."""
    graph = {
        'clip_loader': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': WAN_CLIP_NAME, 'type': 'wan', 'device': 'cpu'}},
        'vae_loader': {'class_type': 'VAELoader', 'inputs': {'vae_name': WAN_VAE_NAME}},
        'unet_high': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': WAN_HIGH_NOISE_UNET}},
        'unet_low':  {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': WAN_LOW_NOISE_UNET}},
        'lora_high': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['unet_high', 0], 'lora_name': WAN_HIGH_NOISE_LORA, 'strength_model': 1.0}},
        'lora_low': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['unet_low', 0], 'lora_name': WAN_LOW_NOISE_LORA, 'strength_model': 1.0}},
    }

    model_high_ref = ['lora_high', 0]
    if extra_lora_high and extra_lora_high.get('name'):
        graph['lora_high_extra'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': model_high_ref, 'lora_name': extra_lora_high['name'],
            'strength_model': float(extra_lora_high.get('strength', 1.0))}}
        model_high_ref = ['lora_high_extra', 0]

    model_low_ref = ['lora_low', 0]
    if extra_lora_low and extra_lora_low.get('name'):
        graph['lora_low_extra'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': model_low_ref, 'lora_name': extra_lora_low['name'],
            'strength_model': float(extra_lora_low.get('strength', 1.0))}}
        model_low_ref = ['lora_low_extra', 0]

    graph.update({
        'samp_high': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': model_high_ref, 'shift': 5.0}},
        'samp_low':  {'class_type': 'ModelSamplingSD3', 'inputs': {'model': model_low_ref, 'shift': 5.0}},
        'latent': {'class_type': 'EmptyHunyuanLatentVideo', 'inputs': {
            'width': width, 'height': height, 'length': length, 'batch_size': 1}},
        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'text': positive, 'clip': ['clip_loader', 0]}},
        'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'text': negative, 'clip': ['clip_loader', 0]}},
        'sampler_high': {'class_type': 'KSamplerAdvanced', 'inputs': {
            'model': ['samp_high', 0], 'add_noise': 'enable', 'noise_seed': seed,
            'steps': 4, 'cfg': 1, 'sampler_name': 'euler', 'scheduler': 'simple',
            'positive': ['pos', 0], 'negative': ['neg', 0], 'latent_image': ['latent', 0],
            'start_at_step': 0, 'end_at_step': 2, 'return_with_leftover_noise': 'enable'}},
        'sampler_low': {'class_type': 'KSamplerAdvanced', 'inputs': {
            'model': ['samp_low', 0], 'add_noise': 'disable', 'noise_seed': 0,
            'steps': 4, 'cfg': 1, 'sampler_name': 'euler', 'scheduler': 'simple',
            'positive': ['pos', 0], 'negative': ['neg', 0], 'latent_image': ['sampler_high', 0],
            'start_at_step': 2, 'end_at_step': 4, 'return_with_leftover_noise': 'disable'}},
        'vae_decode': {'class_type': 'VAEDecode', 'inputs': {
            'samples': ['sampler_low', 0], 'vae': ['vae_loader', 0]}},
        'create_video': {'class_type': 'CreateVideo', 'inputs': {
            'images': ['vae_decode', 0], 'fps': fps}},
        'save_video': {'class_type': 'SaveVideo', 'inputs': {
            'video': ['create_video', 0], 'filename_prefix': filename_prefix,
            'format': 'auto', 'codec': 'auto'}},
    })
    return graph

def wan_video_generate_thread(params, jid):
    """txt2vid przez WAN 2.2 T2V-A14B (ComfyUI, port 8189). Rozdzielczosc/dlugosc
    klatek ograniczone pod 12GB VRAM — patrz build_wan_t2v_graph."""
    try:
        job_set(jid, status='generating')
        width  = int(params.get('width', 640))
        height = int(params.get('height', 640))
        fps    = int(params.get('fps', 16))
        # length musi byc postaci 4n+1 (kompresja czasowa VAE) — dopasuj do najblizszej
        raw_length = int(params.get('length', 81))
        length = max(1, ((raw_length - 1) // 4) * 4 + 1)
        seed   = int(params.get('seed', -1))
        if seed < 0:
            seed = secrets.randbelow(2**32)
        ts = int(time.time() * 1000)

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)

        graph = build_wan_t2v_graph(
            params['positive'], params.get('negative', ''),
            width, height, length, seed, fps,
            filename_prefix=f'genphoto/wan_t2v_{ts}',
            extra_lora_high=params.get('extra_lora_high'),
            extra_lora_low=params.get('extra_lora_low'),
        )

        resp = comfy_post('/prompt', {'prompt': graph}, timeout=30)
        prompt_id = resp['prompt_id']

        # Pierwsze uruchomienie laduje ~12GB modeli z dysku — dajemy dlugi timeout.
        t_start = time.time()
        deadline = t_start + 900
        history = None
        while time.time() < deadline:
            job_set(jid, status='generating', progress={'elapsed': int(time.time() - t_start)})
            h = comfy_get(f'/history/{prompt_id}', timeout=15)
            if prompt_id in h and h[prompt_id].get('outputs'):
                history = h[prompt_id]
                break
            time.sleep(3)
        if history is None:
            raise RuntimeError('Timeout oczekiwania na wygenerowanie wideo (WAN 2.2)')

        # UWAGA: SaveVideo mimo nazwy zwraca wynik pod kluczem 'images', nie 'videos'.
        vids_meta = history['outputs'].get('save_video', {}).get('images', [])
        if not vids_meta:
            raise RuntimeError('ComfyUI nie zwrocilo zadnego wideo (WAN 2.2)')

        vid  = vids_meta[0]
        src  = COMFY_OUTPUTS / vid.get('subfolder', '') / vid['filename']
        name = f'wan_t2v_{ts}_s{seed}.mp4'
        (out_dir / name).write_bytes(src.read_bytes())
        path = f'genphoto/{today}/{name}'

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description', ''),
                     params['positive'], params.get('negative', ''), 'wan2.2_t2v_a14b',
                     'euler', 'simple', 4, 1.0, width, height,
                     seed, 1, 'wanvideo', json.dumps([path]))
                )
        job_set(jid, status='done', images=[path], seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))


# ── Vision Describer ─────────────────────────────────────────────────────────
VISION_SYSTEM_PROMPT = (
    'You are an expert Stable Diffusion prompt engineer. '
    'Analyze the provided image and create an optimized SD img2img prompt '
    'to recreate this person/scene in a specific art style.\n\n'
    'Describe all visible features: hair (color, length, style), eyes (color), '
    'face shape, skin tone, age appearance, distinctive features, clothing details, '
    'pose, expression, lighting, background.\n\n'
    'Output EXACTLY two lines:\n'
    'POSITIVE: [comma-separated SD tags]\n'
    'NEGATIVE: [comma-separated SD tags]\n\n'
    'Do not write anything else — just the two lines.'
)

VISION_STYLE_HINTS = {
    'anime':    'anime style, masterpiece, best quality, very aesthetic, absurdres',
    'photo':    'RAW photo, (photorealistic:1.4), (realistic:1.3), 8k uhd, sharp focus',
    'portrait': 'RAW photo, portrait, photorealistic, professional studio lighting, 8k uhd',
}

INSTANTID_IP_MODEL   = 'ip-adapter-instantid-sdxl [eb2d3ec0]'
INSTANTID_CN_MODEL   = 'instantid-controlnet [c5c25a50]'

def forge_instantid_thread(params, jid):
    try:
        job_set(jid, status='generating')
        face_b64 = auto_crop_face(params['face_image'])
        title, fname = resolve_model(params['model'])
        data = forge_post('/sdapi/v1/txt2img', {
            'prompt':          params['positive'],
            'negative_prompt': params.get('negative', ''),
            'sampler_name':    'DPM++ 2M',
            'scheduler':       'Karras',
            'steps':           int(params.get('steps', 30)),
            'cfg_scale':       float(params.get('cfg', 5.0)),
            'width':           int(params.get('width', 832)),
            'height':          int(params.get('height', 1216)),
            'seed':            int(params.get('seed', -1)),
            'batch_size':      1,
            'n_iter':          1,
            'override_settings': model_override_settings(title, fname),
            'override_settings_restore_afterwards': True,
            'alwayson_scripts': {
                'controlnet': {
                    'args': [
                        {
                            'enabled':      True,
                            'image':        face_b64,
                            'module':       'InsightFace (InstantID)',
                            'model':        INSTANTID_IP_MODEL,
                            'weight':       float(params.get('face_strength', 0.8)),
                            'control_mode': 2,
                            'resize_mode':  1,
                        },
                        {
                            'enabled':      True,
                            'image':        face_b64,
                            'module':       'instant_id_face_keypoints',
                            'model':        INSTANTID_CN_MODEL,
                            'weight':       float(params.get('pose_strength', 0.5)),
                            'control_mode': 2,
                            'resize_mode':  1,
                        },
                    ]
                }
            },
            'save_images': False,
        })
        imgs_b64 = data.get('images', [])
        info     = json.loads(data.get('info', '{}'))
        seed     = info.get('seed', -1)

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        paths = []
        for i, b64 in enumerate(imgs_b64):
            name = f'gp_portrait_{ts}_{i:02d}_s{seed}.png'
            (out_dir / name).write_bytes(base64.b64decode(b64))
            paths.append(f'genphoto/{today}/{name}')

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), '',
                     params['positive'], params.get('negative', ''), params['model'],
                     'DPM++ 2M', 'Karras',
                     int(params.get('steps', 30)), float(params.get('cfg', 5.0)),
                     int(params.get('width', 832)), int(params.get('height', 1216)),
                     seed, 1, 'portrait', json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def forge_pose_thread(params, jid):
    try:
        job_set(jid, status='generating')
        body_b64  = params['body_image']
        face_b64  = auto_crop_face(body_b64)  # auto-crop twarzy z tego samego zdjecia
        title, fname = resolve_model(params['model'])

        cn_args = [
            {   # reference_only — zachowuje sylwetkę/kompozycję
                'enabled':      True,
                'image':        body_b64,
                'module':       'reference_only',
                'model':        'None',
                'weight':       float(params.get('body_strength', 0.65)),
                'control_mode': 0,
                'resize_mode':  1,
            },
        ]
        if face_b64:  # zawsze True po auto_crop_face (fallback = oryginal)
            cn_args += [
                {   # InstantID IP-Adapter — twarz
                    'enabled':      True,
                    'image':        face_b64,
                    'module':       'InsightFace (InstantID)',
                    'model':        INSTANTID_IP_MODEL,
                    'weight':       float(params.get('face_strength', 0.8)),
                    'control_mode': 0,
                    'resize_mode':  1,
                },
                {   # InstantID ControlNet — keypoints twarzy
                    'enabled':      True,
                    'image':        face_b64,
                    'module':       'instant_id_face_keypoints',
                    'model':        INSTANTID_CN_MODEL,
                    'weight':       float(params.get('face_strength', 0.8)) * 0.6,
                    'control_mode': 0,
                    'resize_mode':  1,
                },
            ]

        data = forge_post('/sdapi/v1/txt2img', {
            'prompt':          params['positive'],
            'negative_prompt': params.get('negative', ''),
            'sampler_name':    'DPM++ 2M',
            'scheduler':       'Karras',
            'steps':           int(params.get('steps', 30)),
            'cfg_scale':       float(params.get('cfg', 5.0)),
            'width':           int(params.get('width', 832)),
            'height':          int(params.get('height', 1216)),
            'seed':            int(params.get('seed', -1)),
            'batch_size':      1,
            'n_iter':          1,
            'override_settings': model_override_settings(title, fname),
            'override_settings_restore_afterwards': True,
            'alwayson_scripts': {'controlnet': {'args': cn_args}},
            'save_images': False,
        })
        imgs_b64 = data.get('images', [])
        info     = json.loads(data.get('info', '{}'))
        seed     = info.get('seed', -1)

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        paths = []
        for i, b64 in enumerate(imgs_b64):
            name = f'gp_pose_{ts}_{i:02d}_s{seed}.png'
            (out_dir / name).write_bytes(base64.b64decode(b64))
            paths.append(f'genphoto/{today}/{name}')

        gid = uuid.uuid4().hex
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), '',
                     params['positive'], params.get('negative', ''), params['model'],
                     'DPM++ 2M', 'Karras',
                     int(params.get('steps', 30)), float(params.get('cfg', 5.0)),
                     int(params.get('width', 832)), int(params.get('height', 1216)),
                     seed, 1, 'pose', json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def _vision_call(url, model, image_b64, style_prefix, style, headers_extra=None):
    mime = 'image/jpeg'
    data_url = f'data:{mime};base64,{image_b64}'
    payload = json.dumps({
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': (
                    f'Analyze this image and create a Stable Diffusion img2img prompt '
                    f'to recreate it in {style} style.\n'
                    f'Start POSITIVE with: {style_prefix}\n'
                    f'Then describe all person features, clothing, pose, lighting, background.\n'
                    f'Output exactly:\n'
                    f'POSITIVE: ...\n'
                    f'NEGATIVE: ...'
                )},
                {'type': 'image_url', 'image_url': {'url': data_url}},
            ]
        }],
        'temperature': 0.4,
        'max_tokens': 600,
    }).encode()
    headers = {'Content-Type': 'application/json', 'User-Agent': 'GenPhoto/1.0'}
    if headers_extra:
        headers.update(headers_extra)
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read())
    return (resp['choices'][0]['message'].get('content') or '').strip()

def _parse_vision_text(text):
    pos = neg = ''
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith('POSITIVE:'):
            pos = l[9:].strip().lstrip(':').strip()
        elif l.upper().startswith('NEGATIVE:'):
            neg = l[9:].strip().lstrip(':').strip()
    return pos, neg

def ai_vision_describe(image_b64, style='anime', model=None):
    style_prefix = VISION_STYLE_HINTS.get(style, VISION_STYLE_HINTS['anime'])
    # Jeśli użytkownik wybrał konkretny model w UI, pomijamy lokalny fallback
    # i idziemy prosto na wybrany model przez OpenRouter.
    if not model and LOCAL_VISION_URL:
        try:
            text = _vision_call(
                f'{LOCAL_VISION_URL}/v1/chat/completions',
                LOCAL_VISION_MODEL, image_b64, style_prefix, style
            )
            pos, neg = _parse_vision_text(text)
            if pos:
                return pos, neg
        except Exception:
            pass
    if not OR_KEY:
        raise RuntimeError('GP_OR_KEY not set i GP_LOCAL_VISION_URL nie skonfigurowany')
    text = _vision_call(
        'https://openrouter.ai/api/v1/chat/completions',
        model or OR_VISION_MODEL, image_b64, style_prefix, style,
        headers_extra={
            'Authorization': f'Bearer {OR_KEY}',
            'HTTP-Referer':  'https://ebartnet.pl',
            'X-Title':       'GenPhoto',
        }
    )
    pos, neg = _parse_vision_text(text)
    if not pos:
        raise RuntimeError(f'Vision API returned: {text[:200]}')
    return pos, neg


# ── DeepSeek AI Prompter ──────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    'You are an expert Stable Diffusion prompt engineer specializing in photorealistic images.\n'
    'The user will describe a scene in any language. Your job is to generate an optimized SD prompt.\n\n'
    'Always answer with exactly two lines in this format:\n'
    'POSITIVE: [comma-separated tags]\n'
    'NEGATIVE: [comma-separated tags]\n\n'
    'For POSITIVE always begin with: RAW photo, real person, detailed skin texture, '
    '(photorealistic:1.4), (realistic:1.3), 8k uhd — then add subject details, '
    'lighting (golden hour / dramatic / studio), camera (85mm, f/1.4, bokeh), mood.\n\n'
    'For NEGATIVE always include: (worst quality:2), (low quality:2), (blurry:1.3), deformed, '
    'ugly, extra limbs, mutated hands, (bad anatomy:1.3), watermark, text, painting, '
    'cartoon, anime, 3d render, plastic skin, unnatural skin.\n\n'
    'Do not write anything else — just the two lines starting with POSITIVE: and NEGATIVE:'
)

VIDEO_SYSTEM_PROMPT = (
    'You are an expert AnimateDiff / Stable Diffusion 1.5 prompt engineer for short animated clips.\n'
    'The user describes a motion scene in any language. Generate a prompt optimized for SD1.5 + AnimateDiff.\n\n'
    'Rules:\n'
    '- NEVER use: "real person", "skin texture", "RAW photo", portrait or face-specific tags\n'
    '- Focus on: subject (animal/object/landscape), motion description, environment, mood\n'
    '- Motion tags: smooth motion, fluid animation, cinematic movement, dynamic, flowing\n'
    '- Style tags: photorealistic, highly detailed, natural lighting, 8k uhd\n'
    '- Keep POSITIVE under 20 tags total\n\n'
    'Always answer with exactly two lines:\n'
    'POSITIVE: [comma-separated tags]\n'
    'NEGATIVE: [comma-separated tags]\n\n'
    'For NEGATIVE always include: (worst quality:2), (low quality:2), (blurry:1.3), deformed, '
    'watermark, text, cartoon, anime, 3d render, static image, frozen, jerky, flickering, '
    'morphing faces, mutated.\n\n'
    'Do not write anything else — just the two lines starting with POSITIVE: and NEGATIVE:'
)

EDIT_SYSTEM_PROMPT = (
    'You are an expert Stable Diffusion inpainting / img2img prompt engineer.\n'
    'The user describes a change they want to make to an existing photo (in any language).\n'
    'Your job is to generate a prompt that describes ONLY the modified element, not the whole scene.\n\n'
    'Rules:\n'
    '- POSITIVE: describe what the changed area should look like — be specific about color, texture, material, shape\n'
    '  Begin with photorealistic quality tags, then the specific change\n'
    '  Example: "photorealistic, (red dress:1.4), vibrant red silk fabric, smooth folds, high quality"\n'
    '- NEGATIVE: list what should NOT appear in that area (old element + standard quality negatives)\n'
    '  Example: "blue dress, green dress, (worst quality:2), deformed, blurry, cartoon"\n'
    '- Do NOT describe the background, lighting or other parts of the image — only the changed element\n'
    '- Keep prompts short and focused (max 15 tags each)\n\n'
    'Always answer with exactly two lines:\n'
    'POSITIVE: [comma-separated tags]\n'
    'NEGATIVE: [comma-separated tags]\n\n'
    'Do not write anything else — just the two lines starting with POSITIVE: and NEGATIVE:'
)

def ai_prompt(description, mode='photo'):
    provider = AI_PROVIDER
    
    if provider == 'openrouter':
        if not OR_KEY:
            raise RuntimeError('GP_OR_KEY not set')
        url = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OR_KEY}',
            'HTTP-Referer': 'https://ebartnet.pl',
            'X-Title': 'GenPhoto'
        }
        model = OR_MODEL
    else:
        if not DEEPSEEK_KEY:
            raise RuntimeError('GP_DEEPSEEK_KEY not set')
        url = 'https://api.deepseek.com/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_KEY}',
        }
        model = DEEPSEEK_MODEL
    
    sys_prompt = {'video': VIDEO_SYSTEM_PROMPT, 'edit': EDIT_SYSTEM_PROMPT}.get(mode, SYSTEM_PROMPT)
    raw = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user',   'content': description},
        ],
        'temperature': 0.6,
        'max_tokens': 500,
    }).encode()
    req = urllib.request.Request(url, data=raw, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    msg  = data['choices'][0]['message']
    text = (msg.get('content') or '').strip()
    if not text:
        text = (msg.get('reasoning_content') or '').strip()
    if not text:
        raise RuntimeError(f'AI returned empty response: {list(msg.keys())}')
    pos = neg = ''
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith('POSITIVE:'):
            pos = l[9:].strip().lstrip(':').strip()
        elif l.upper().startswith('NEGATIVE:'):
            neg = l[9:].strip().lstrip(':').strip()
    if not pos and not neg:
        raise RuntimeError(f'Invalid response format: {text[:200]}')
    return pos, neg

# ── HTML ──────────────────────────────────────────────────────────────────────
# Szablony leza w templates/ obok tego pliku. Wczytywane raz, przy starcie —
# wczesniej siedzialy tu jako ~3100 linii literalu w srodku modulu.
TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'

def _load_template(name):
    return (TEMPLATES_DIR / name).read_text(encoding='utf-8')

LOGIN_PAGE = _load_template('login.html')

HTML_TEMPLATE = _load_template('index.html')

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _cookie(self):
        raw = self.headers.get('Cookie', '')
        for c in raw.split(';'):
            c = c.strip()
            if c.startswith(COOKIE_NAME + '='):
                return c[len(COOKIE_NAME)+1:]
        return None

    def _client_ip(self):
        peer = self.client_address[0]
        if peer in ('127.0.0.1', '::1'):
            fwd = (self.headers.get('CF-Connecting-IP')
                   or self.headers.get('X-Forwarded-For', '').split(',')[0].strip())
            if fwd:
                return fwd
        return peer

    def _authed(self):
        return session_valid(self._cookie())

    def _json(self, d, code=200):
        body = json.dumps(d).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body, code=200):
        data = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, loc):
        self.send_response(302)
        self.send_header('Location', loc)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _build_page(self):
        # Fetch models from Forge
        with _models_lock:
            raw_models = [m for m in _models_cache if not m.get('model_name','').lower().startswith('flux')]
        model_opts = build_model_optgroups(raw_models)
        if not model_opts:
            model_opts = '<option value="RealVisXL_V5_fp16">RealVisXL_V5_fp16</option>'

        preset_tabs = ''.join(
            f'<button class="preset-btn" data-id="{p["id"]}" onclick="applyPreset(\'{p["id"]}\')">'
            f'{p["icon"]} {p["name"]}</button>'
            for p in PRESETS
        )
        presets_json = json.dumps([
            {k: p[k] for k in ('id','name','model','sampler','scheduler',
                                'steps','cfg','width','height','batch','prefix','negative')}
            for p in PRESETS
        ])
        # Filtruj modele SD1.5 dla AnimateDiff
        sd15_opts = ''
        for m in raw_models:
            n = html.escape(m.get('model_name',''))
            t = html.escape(m.get('model_name',''))
            nl = n.lower()
            if any(k in nl for k in ('v1-5','v1_5','sd15','sd1.5','pruned')):
                sd15_opts += f'<option value="{t}">{n}</option>'
        if not sd15_opts:
            sd15_opts = '<option value="v1-5-pruned-emaonly">v1-5-pruned-emaonly</option>'

        # Modele dla Portretu: tylko SDXL (bez FLUX i SD1.5)
        portrait_opts = ''
        for m in raw_models:
            n = html.escape(m.get('model_name',''))
            t = html.escape(m.get('model_name',''))
            fn = m.get('filename', '')
            nl = n.lower()
            # pomiń FLUX (double_blocks) i SD1.5
            if any(k in nl for k in ('v1-5','v1_5','sd15','pruned')):
                continue
            if is_flux_checkpoint(fn):
                continue
            portrait_opts += f'<option value="{t}">{n}</option>'
        if not portrait_opts:
            portrait_opts = '<option value="RealVisXL_V5_fp16">RealVisXL_V5_fp16</option>'

        return (HTML_TEMPLATE
                .replace('__PRESET_TABS__', preset_tabs)
                .replace('__MODEL_OPTIONS__', model_opts)
                .replace('__EDIT_MODEL_OPTIONS__', model_opts)
                .replace('__VID_MODEL_OPTIONS__', sd15_opts)
                .replace('__PORTRAIT_MODEL_OPTIONS__', portrait_opts)
                .replace('__PRESETS_JSON__', presets_json)
                .replace('__GALLERY_URL__', GALLERY_URL)
                .replace('__PORTAL_URL__', PORTAL_URL))

    def do_GET(self):
        path = self.path.split('?')[0]

        # Bariera auth na szczycie routingu. Wszystko powyzej niej bylo dostepne
        # publicznie (m.in. /api/model-delete kasujacy pliki i /api/download-model
        # sciagajacy dowolny URL). Jedyny wyjatek to sama strona logowania.
        if path != '/login' and not self._authed():
            self._redirect('/login')
            return

        # Static image serving
        if path.startswith('/img/'):
            rel = unquote(path[5:])
            fpath = (OUTPUTS_DIR / rel).resolve()
            # is_relative_to zamiast startswith na stringu: prefix-match przepuszczal
            # katalogi-siostry o wspolnym poczatku nazwy (np. .../outputs-backup/).
            if fpath.is_relative_to(OUTPUTS_DIR.resolve()) and fpath.is_file():
                mime, _ = mimetypes.guess_type(str(fpath))
                size = fpath.stat().st_size
                self.send_response(200)
                self.send_header('Content-Type', mime or 'image/png')
                self.send_header('Content-Length', size)
                self.send_header('Cache-Control', 'private, max-age=3600')
                self.end_headers()
                # Strumieniowo: read_bytes() ladowal caly plik do RAM, co przy
                # kilkusetmegabajtowym wideo z WAN dawalo skok zuzycia pamieci.
                try:
                    with open(fpath, 'rb') as fh:
                        shutil.copyfileobj(fh, self.wfile, 256 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
            return
        if path == '/api/models/refresh':
            _refresh_models_once(rescan_disk=True)
            self._json({'ok': True})
            return

        if path == '/api/loras':
            self._json(list_lora_names(LORAS_DIR))
            return

        if path == '/api/forge-loras':
            self._json(list_lora_names(FORGE_LORAS_DIR, stem=True))
            return

        if path.startswith('/api/download-progress/'):
            job_id = path[23:]
            d = _downloads.get(job_id)
            if not d: self._json({'ok': False, 'error': 'brak job'}); return
            self._json({'ok': True, **d})
            return

        if path.startswith('/api/model-delete/'):
            fname = unquote(path[18:])
            if '/' in fname or '\\' in fname or not fname:
                self._json({'ok': False, 'error': 'nieprawidłowa nazwa'}); return
            fpath = MODELS_DIR / fname
            if not fpath.exists():
                self._json({'ok': False, 'error': 'plik nie istnieje'}); return
            fpath.unlink()
            _refresh_models_once()
            self._json({'ok': True})
            return

        if path.startswith('/api/auto-model'):
            from urllib.parse import urlparse, parse_qs as pqs
            qs     = pqs(urlparse(self.path).query)
            prompt = qs.get('prompt', [''])[0].strip()
            models = qs.get('models', [''])[0].strip()
            if not prompt or not models:
                self._json({'ok': False, 'error': 'brak prompt lub models'}); return
            if not OR_KEY:
                self._json({'ok': False, 'error': 'GP_OR_KEY not set'}); return
            import time as _t
            cache_key = 'mdl:' + prompt[:80] + '|' + models[:120]
            cached = _auto_cache.get(cache_key)
            if cached and (_t.time() - cached['ts']) < 3600:
                self._json({'ok': True, 'model': cached['model'], 'reason': cached['reason'], 'cached': True}); return
            model_list = [m.strip() for m in models.split(',') if m.strip()]
            llm_prompt = (
                "You are an expert in Stable Diffusion models.\n"
                "Given an image generation prompt and a list of available checkpoint models,\n"
                "choose the single best model for the prompt.\n\n"
                f"Prompt: {prompt[:400]}\n\n"
                "Available models:\n" +
                '\n'.join(f'- {m}' for m in model_list) +
                "\n\nRespond ONLY with valid JSON (no markdown):\n"
                '{"model": "exact_model_name_from_list", "reason": "one sentence why"}'
            )
            req_body = json.dumps({
                'model': OR_MODEL,
                'messages': [{'role': 'user', 'content': llm_prompt}],
                'temperature': 0.2, 'max_tokens': 200
            }).encode()
            req = urllib.request.Request(
                'https://openrouter.ai/api/v1/chat/completions',
                data=req_body,
                headers={'Content-Type': 'application/json',
                         'Authorization': f'Bearer {OR_KEY}',
                         'HTTP-Referer': 'https://genphoto.ebartnet.pl'},
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                raw = data['choices'][0]['message']['content'].strip().strip('`').strip()
                if raw.startswith('json'): raw = raw[4:].strip()
                result = json.loads(raw)
                chosen = result.get('model', '').strip()
                reason = result.get('reason', '')
                if chosen not in model_list:
                    for m in model_list:
                        if chosen.lower() in m.lower() or m.lower() in chosen.lower():
                            chosen = m; break
                auto_cache_put(cache_key, {'model': chosen, 'reason': reason, 'ts': _t.time()})
                self._json({'ok': True, 'model': chosen, 'reason': reason, 'cached': False})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        if path.startswith('/api/civitai-settings'):
            from urllib.parse import urlparse, parse_qs as pqs
            qs    = pqs(urlparse(self.path).query)
            mname = qs.get('model', [''])[0].strip()
            if not mname:
                self._json({'ok': False, 'error': 'brak model'}); return
            with _models_lock:
                m = next((x for x in _models_cache if x.get('model_name') == mname), None)
            sha256 = (m or {}).get('sha256', '')
            if not sha256:
                self._json({'ok': False, 'error': 'Brak hash pliku (Forge jeszcze go nie policzył — wygeneruj raz tym modelem, potem spróbuj ponownie)'}); return
            try:
                result = _fetch_civitai_settings_by_hash(sha256)
                if not result:
                    self._json({'ok': False, 'error': 'Civitai nie zwróciło danych dla tego pliku'}); return
                self._json({'ok': True, 'settings': result['settings'], 'base_model': result['base_model']})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if path.startswith('/api/auto-settings'):
            from urllib.parse import urlparse, parse_qs as pqs
            qs    = pqs(urlparse(self.path).query)
            mname = qs.get('model', [''])[0].strip()
            if not mname:
                self._json({'ok': False, 'error': 'brak model'}); return
            import time as _t
            # Trwałe ustawienia (ręczne, civitai lub wcześniej zgadnięte przez LLM)
            # mają pierwszeństwo — przeżywają restart, w przeciwieństwie do _auto_cache.
            saved = _get_saved_model_settings(mname)
            if saved:
                self._json({'ok': True, 'settings': saved['settings'], 'cached': True, 'source': saved['source']}); return
            cached = _auto_cache.get(mname)
            if cached and (_t.time() - cached['ts']) < 86400 * 7:
                self._json({'ok': True, 'settings': cached['settings'], 'cached': True}); return
            if not OR_KEY:
                self._json({'ok': False, 'error': 'GP_OR_KEY not set'}); return
            prompt = (
                "You are an expert in Stable Diffusion image generation parameters.\n"
                "Given a model filename, return the optimal generation settings.\n\n"
                f"Model filename: \"{mname}\"\n\n"
                "Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
                '{"sampler":"...","scheduler":"...","steps":20,"cfg":1.0,"width":1024,"height":1024,"notes":"..."}\n\n'
                "Key rules:\n"
                "- FLUX/.gguf models: Euler, Simple, steps 20, cfg 1.0, 1024x1024\n"
                "- LCM/Turbo/Lightning/Hyper/Ultra: Euler, Simple, steps 8-12, cfg 1.0-2.0\n"
                "- SDXL/XL/Pony: DPM++ SDE, Karras, steps 28, cfg 5.5, 1024x1024\n"
                "- SD 1.5 realistic: DPM++ SDE, Karras, steps 28, cfg 7.0, 832x1216\n"
                "- SD 1.5 anime: DPM++ 2M, Karras, steps 30, cfg 7.0, 512x768\n"
                "- krea2/krea: Euler, Simple, steps 20, cfg 1.0, 1024x1024\n"
                "Available samplers: Euler a, Euler, DPM++ 2M, DPM++ SDE, DPM++ SDE Karras, DPM++ 2M Karras\n"
                "Available schedulers: Automatic, Karras, Simple, Normal, Exponential"
            )
            req_body = json.dumps({
                'model': OR_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1, 'max_tokens': 1000
            }).encode()
            req = urllib.request.Request(
                'https://openrouter.ai/api/v1/chat/completions',
                data=req_body,
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {OR_KEY}', 'HTTP-Referer': 'https://genphoto.ebartnet.pl'},
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                raw = data['choices'][0]['message']['content'].strip().strip('`').strip()
                if raw.startswith('json'): raw = raw[4:].strip()
                settings = json.loads(raw)
                auto_cache_put(mname, {'settings': settings, 'ts': _t.time()})
                try:
                    _save_model_settings(mname, settings, 'llm')
                except Exception:
                    pass
                self._json({'ok': True, 'settings': settings, 'cached': False})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        # Logout
        if path == '/logout':
            tok = self._cookie()
            if tok: session_drop(tok)
            self.send_response(302)
            self.send_header('Location', '/login')
            self.send_header('Set-Cookie', f'{COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax; Secure')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return

        if path == '/login':
            self._html(LOGIN_PAGE.replace('__ERR__', ''))
            return

        if path == '/':
            self._html(self._build_page())
            return

        if path.startswith('/api/job/'):
            jid = path[9:]
            j = job_get(jid)
            self._json(j if j else {'status':'not_found'})
            return

        if path == '/api/history':
            with _db_lock:
                with db() as con:
                    rows = con.execute(
                        'SELECT * FROM generations ORDER BY ts DESC LIMIT 100'
                    ).fetchall()
            self._json([dict(r) for r in rows])
            return

        if path == '/api/models':
            with _models_lock:
                raw = list(_models_cache)
            self._json([{
                'name': m.get('model_name',''),
                'title': m.get('title',''),
                'is_flux': bool(m.get('filename')) and is_flux_checkpoint(m['filename']),
            } for m in raw])
            return

        if path == '/api/edit-history':
            with _db_lock:
                with db() as con:
                    rows = con.execute(
                        'SELECT * FROM edits ORDER BY ts DESC LIMIT 50'
                    ).fetchall()
            self._json([dict(r) for r in rows])
            return

        if path == '/api/image-meta':
            import urllib.parse as _up
            qs = _up.parse_qs(_up.urlparse(self.path).query)
            img_path = qs.get('path',[''])[0]
            if not img_path:
                self._json({'ok': False, 'error': 'brak parametru path'})
                return
            try:
                from PIL import Image as _Img
                full = Path(img_path) if img_path.startswith('/') else OUTPUTS_DIR / img_path
                with _Img.open(full) as im:
                    params_str = im.info.get('parameters', '')
                result = {'ok': True, 'raw': params_str, 'path': str(full)}
                # Parse SD parameters string
                if params_str:
                    lines = params_str.split('\n')
                    result['positive'] = lines[0] if lines else ''
                    neg = next((l.replace('Negative prompt:','').strip() for l in lines if l.startswith('Negative prompt:')), '')
                    result['negative'] = neg
                    meta_line = next((l for l in lines if 'Steps:' in l), '')
                    for kv in meta_line.split(','):
                        kv = kv.strip()
                        if ': ' in kv:
                            k, v = kv.split(': ', 1)
                            result[k.strip().lower().replace(' ', '_')] = v.strip()
                self._json(result)
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if path == '/api/vram':
            try:
                import subprocess as _sp
                r = _sp.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                parts = [int(x.strip()) for x in r.stdout.strip().split(',')]
                used_mb, free_mb, total_mb = parts
                self._json({
                    'ok':       True,
                    'used_gb':  round(used_mb  / 1024, 2),
                    'free_gb':  round(free_mb  / 1024, 2),
                    'total_gb': round(total_mb / 1024, 2),
                })
            except Exception as e:
                # Log do journalu — wczesniej wyjatek byl zwracany tylko jako
                # ok:false i znikal bez sladu, co uniemozliwialo diagnoze.
                print(f'[genphoto] /api/vram ERROR: {e!r}', flush=True)
                self._json({'ok': False, 'error': str(e)})
            return

        if path == '/api/metrics':
            with _mlock:
                events = list(_metrics)
            self._json({'ok': True, 'count': len(events), 'events': events[-100:]})
            return

        if path == '/api/forge-progress':
            prog = forge_get('/sdapi/v1/progress') or {}
            state = prog.get('state', {}) or {}
            self._json({
                'progress': round(prog.get('progress', 0), 3),
                'eta':      int(prog.get('eta_relative', 0)),
                'textinfo': prog.get('textinfo', '') or '',
                'step':     state.get('sampling_step', 0),
                'steps':    state.get('sampling_steps', 0),
            })
            return



        self.send_error(404)

    def do_DELETE(self):
        if not self._authed():
            self._json({'ok': False, 'error': 'unauthorized'}, 401); return
        path = self.path.split('?')[0]
        if path.startswith('/api/history/'):
            gen_id = path[13:]
            if not gen_id:
                self._json({'ok': False, 'error': 'brak id'}); return
            with _db_lock:
                with db() as con:
                    con.execute('DELETE FROM generations WHERE id=?', (gen_id,))
            self._json({'ok': True})
            return
        if path.startswith('/api/edit-history/'):
            rec_id = path[18:]
            if not rec_id:
                self._json({'ok': False, 'error': 'brak id'}); return
            with _db_lock:
                with db() as con:
                    con.execute('DELETE FROM edits WHERE id=?', (rec_id,))
            self._json({'ok': True})
            return
        self.send_error(404)

    def _handle_model_upload(self):
        if not self._authed():
            self._json({'ok': False, 'error': 'unauthorized'}, 401); return
        ctype = self.headers.get('Content-Type', '')
        m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', ctype)
        if not ctype.startswith('multipart/form-data') or not m:
            self._json({'ok': False, 'error': 'oczekiwano multipart/form-data'}); return
        boundary = (m.group(1) or m.group(2)).encode()
        total = int(self.headers.get('Content-Length', 0))
        rfile = self.rfile
        remaining = [total]

        def read_bytes(n):
            n = min(n, remaining[0])
            if n <= 0: return b''
            chunk = rfile.read(n)
            remaining[0] -= len(chunk)
            return chunk

        def read_line():
            line = b''
            while remaining[0] > 0:
                b = read_bytes(1)
                if not b: break
                line += b
                if line.endswith(b'\r\n'): break
            return line

        delim = b'--' + boundary
        line = read_line()
        if not line.startswith(delim):
            self._json({'ok': False, 'error': 'zly format multipart'}); return
        filename = None
        while True:
            line = read_line()
            if line in (b'\r\n', b''):
                break
            decoded = line.decode('utf-8', errors='replace')
            mm = re.search(r'filename="([^"]*)"', decoded)
            if mm:
                filename = mm.group(1)
        if not filename:
            self._json({'ok': False, 'error': 'brak nazwy pliku'}); return
        filename = os.path.basename(filename)
        if not filename or not filename.lower().endswith(('.safetensors', '.ckpt', '.pt')):
            self._json({'ok': False, 'error': 'nieobslugiwane rozszerzenie (dozwolone: .safetensors, .ckpt, .pt)'}); return
        if (MODELS_DIR / filename).exists():
            self._json({'ok': False, 'error': 'plik ' + filename + ' juz istnieje'}); return

        tmp_path = MODELS_DIR / (filename + '.part')
        closing = b'\r\n' + delim
        tail_len = len(closing) + 4
        buf = b''
        try:
            with open(tmp_path, 'wb') as f:
                while True:
                    chunk = read_bytes(65536)
                    if not chunk and not buf:
                        break
                    buf += chunk
                    idx2 = buf.find(closing)
                    if idx2 != -1:
                        f.write(buf[:idx2])
                        buf = b''
                        break
                    if len(buf) > tail_len:
                        f.write(buf[:-tail_len])
                        buf = buf[-tail_len:]
                    if not chunk:
                        break
            tmp_path.rename(MODELS_DIR / filename)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            self._json({'ok': False, 'error': str(e)}); return

        _refresh_models_once(rescan_disk=True)
        self._json({'ok': True, 'filename': filename})

    def do_POST(self):
        if self.path == '/api/model-upload':
            self._handle_model_upload()
            return
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length).decode('utf-8', errors='replace')

        # Bariera auth przed calym routingiem POST (poza logowaniem).
        # /api/model-upload jest obslugiwany wyzej i ma wlasny check w metodzie.
        if self.path != '/login' and not self._authed():
            self._json({'ok': False, 'error': 'unauthorized'}, 401)
            return

        if self.path == '/api/download-model':
            try:
                data = json.loads(body)
                url = data.get('url', '').strip()
            except:
                self._json({'ok': False, 'error': 'bad JSON'}); return
            bad = download_url_rejected(url) if url else 'nieprawidłowy URL'
            if bad:
                self._json({'ok': False, 'error': bad}); return
            job_id = uuid.uuid4().hex[:12]
            downloads_gc()
            _downloads[job_id] = {'url': url, 'filename': '', 'percent': 0,
                                   'status': 'starting', 'error': '', 'bytes_done': 0,
                                   'size': 0, 'ts': time.time()}
            threading.Thread(target=_download_model_thread, args=(job_id, url), daemon=True).start()
            self._json({'ok': True, 'job_id': job_id})
            return
        if self.path == '/login':
            ip = self._client_ip()
            if login_blocked(ip):
                err = '<div class="err">Za dużo nieudanych prób. Spróbuj za 5 minut.</div>'
                self._html(LOGIN_PAGE.replace('__ERR__', err), 429)
                return
            params   = parse_qs(body)
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0]
            pw_hash  = hashlib.sha256(password.encode()).hexdigest()
            # compare_digest zamiast == : porownanie w stalym czasie
            ok = (secrets.compare_digest(username, GP_USERNAME)
                  and secrets.compare_digest(pw_hash, GP_PW_HASH))
            if ok:
                login_note_ok(ip)
                tok = session_new()
                self.send_response(302)
                self.send_header('Location', '/')
                self.send_header('Set-Cookie',
                                 f'{COOKIE_NAME}={tok}; Path=/; HttpOnly; SameSite=Lax; Secure')
                self.send_header('Content-Length', '0')
                self.end_headers()
            else:
                login_note_fail(ip)
                err = '<div class="err">Błędny login lub hasło</div>'
                self._html(LOGIN_PAGE.replace('__ERR__', err))
            return

        if self.path == '/api/lora-download':
            try:
                data = json.loads(body)
                url = data.get('url', '').strip()
            except:
                self._json({'ok': False, 'error': 'bad JSON'}); return
            bad = download_url_rejected(url) if url else 'nieprawidłowy URL'
            if bad:
                self._json({'ok': False, 'error': bad}); return
            job_id = uuid.uuid4().hex[:12]
            downloads_gc()
            _downloads[job_id] = {'url': url, 'filename': '', 'percent': 0,
                                   'status': 'starting', 'error': '', 'bytes_done': 0,
                                   'size': 0, 'ts': time.time()}
            threading.Thread(target=_download_model_thread,
                              args=(job_id, url), kwargs={'dest_dir': LORAS_DIR, 'refresh': False},
                              daemon=True).start()
            self._json({'ok': True, 'job_id': job_id})
            return

        if self.path == '/api/lora-upload':
            try:
                data = json.loads(body)
                fname = data.get('filename', '').strip()
                b64   = data.get('data', '')
            except:
                self._json({'ok': False, 'error': 'bad JSON'}); return
            if not fname or '/' in fname or '\\' in fname:
                self._json({'ok': False, 'error': 'nieprawidłowa nazwa pliku'}); return
            if not fname.endswith(('.safetensors', '.pt', '.ckpt')):
                self._json({'ok': False, 'error': 'obsługiwane tylko .safetensors/.pt/.ckpt'}); return
            try:
                raw = base64.b64decode(b64)
                (LORAS_DIR / fname).write_bytes(raw)
                self._json({'ok': True})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/auto-settings-save':
            try:
                data = json.loads(body)
                mname = data.get('model', '').strip()
                settings = {
                    'sampler':   data.get('sampler', ''),
                    'scheduler': data.get('scheduler', ''),
                    'steps':     int(data.get('steps', 20)),
                    'cfg':       float(data.get('cfg', 5.0)),
                    'width':     int(data.get('width', 1024)),
                    'height':    int(data.get('height', 1024)),
                }
            except Exception:
                self._json({'ok': False, 'error': 'bad JSON'}); return
            if not mname:
                self._json({'ok': False, 'error': 'brak model'}); return
            _save_model_settings(mname, settings, 'manual')
            self._json({'ok': True})
            return

        if self.path == '/api/vram-free':
            # Restart WSZYSTKICH backendów GPU — poprzednio tylko Forge, a VRAM
            # potrafił trzymać ComfyUI (Z-Image/WAN ~7-9 GB idle).
            try:
                import subprocess as _sp
                for svc in ('forge', 'krea2', 'comfyui'):
                    _sp.Popen(['systemctl', '--user', 'restart', svc],
                              stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                metric('vram_free_restart', services='forge,krea2,comfyui')
                def _health_watch():
                    time.sleep(2)
                    for b in ('forge', 'krea2', 'zimage'):
                        _wait_backend_health(b, timeout_s=150)
                    metric('vram_free_healthy')
                threading.Thread(target=_health_watch, daemon=True).start()
                self._json({'ok': True, 'msg': 'Restart Forge + Krea2 + ComfyUI — pełna gotowość za ~60-90s'})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path.startswith('/api/service/'):
            parts = self.path.split('/')
            valid_services = ('forge', 'krea2', 'comfyui')
            if len(parts) == 5 and parts[3] in valid_services and parts[4] in ('start', 'stop'):
                name, action = parts[3], parts[4]
                try:
                    import subprocess as _sp
                    _sp.run(['systemctl', '--user', action, name], timeout=20, check=True)
                    self._json({'ok': True})
                except Exception as e:
                    self._json({'ok': False, 'error': str(e)})
            else:
                self._json({'ok': False, 'error': 'nieprawidłowy serwis/akcja'})
            return

        if self.path == '/api/generate':
            try:
                params = json.loads(body)
                params['gen_id'] = uuid.uuid4().hex
                backend = params.get('backend', 'forge')
                if backend == 'krea2' and params.get('krea_all_variants'):
                    target, kind = krea_batch_variants_thread, 'krea2_batch'
                elif backend == 'krea2':
                    target, kind = krea_generate_thread, 'krea2'
                elif backend == 'zimage':
                    target, kind = zimage_generate_thread, 'zimage'
                elif backend == 'wanvideo':
                    target, kind = wan_video_generate_thread, 'wanvideo'
                elif backend == 'forge' and params.get('all_models'):
                    target, kind, backend = forge_batch_all_models_thread, 'batch_all', 'forge'
                elif params.get('init_image'):
                    target, kind, backend = forge_img2img_ref_thread, 'forge', 'forge'
                else:
                    target, kind, backend = forge_generate_thread, 'forge', 'forge'
                if gpu_queue_full():
                    metric('queue_reject', backend=backend)
                    self._json({'ok': False,
                                'error': 'GPU zajęte — 1 generacja w toku i 1 w kolejce. '
                                         'Poczekaj na zakończenie.',
                                'retry_after_s': 30}, 503)
                    return
                jid = job_create()
                t = threading.Thread(target=gpu_run,
                                     args=(kind, backend, target, params, jid),
                                     daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/detect-face':
            try:
                data = json.loads(body)
                img_b64 = data.get('image_b64', '')
                if not img_b64:
                    self._json({'ok': False, 'error': 'Brak obrazu'}); return
                face_b64, detected = _crop_face_with_status(img_b64)
                self._json({'ok': True, 'face_b64': face_b64, 'detected': detected})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/face-portrait':
            try:
                params = json.loads(body)
                if not params.get('face_image'):
                    self._json({'ok': False, 'error': 'Brak zdjęcia twarzy'}); return
                if not params.get('positive'):
                    self._json({'ok': False, 'error': 'Brak promptu'}); return
                params['gen_id'] = uuid.uuid4().hex
                jid = job_create()
                t = threading.Thread(target=gpu_run,
                                     args=('forge', 'forge', forge_instantid_thread, params, jid),
                                     daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/pose':
            try:
                params = json.loads(body)
                if not params.get('body_image'):
                    self._json({'ok': False, 'error': 'Brak zdjęcia ciała'}); return
                jid = job_create()
                t = threading.Thread(target=gpu_run,
                                     args=('forge', 'forge', forge_pose_thread, params, jid),
                                     daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/vision-describe':
            try:
                data = json.loads(body)
                image_b64 = data.get('image_b64', '')
                style     = data.get('style', 'anime')
                vmodel    = data.get('model', '').strip() or None
                if not image_b64:
                    self._json({'ok': False, 'error': 'Brak image_b64'}); return
                pos, neg = ai_vision_describe(image_b64, style, model=vmodel)
                self._json({'ok': True, 'positive': pos, 'negative': neg})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/ai-prompt':
            try:
                data = json.loads(body)
                desc = data.get('description', '').strip()
                mode = data.get('mode', 'photo')
                if not desc:
                    self._json({'ok': False, 'error': 'Brak opisu'}); return
                pos, neg = ai_prompt(desc, mode=mode)
                self._json({'ok': True, 'positive': pos, 'negative': neg})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/edit':
            try:
                params = json.loads(body)
                if not params.get('image_b64'):
                    self._json({'ok': False, 'error': 'Brak image_b64'}); return
                if not params.get('positive'):
                    self._json({'ok': False, 'error': 'Brak positive prompt'}); return
                params['gen_id'] = uuid.uuid4().hex
                jid = job_create()
                t = threading.Thread(target=gpu_run,
                                     args=('forge', 'forge', forge_edit_thread, params, jid),
                                     daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/publish_to_portal':
            try:
                data = json.loads(body)
                gen_id   = data.get('gen_id', '').strip()
                path_idx = int(data.get('path_idx', 0))
                is_xxx   = 1 if data.get('is_xxx') else 0
                if not gen_id:
                    self._json({'ok': False, 'error': 'Brak gen_id'}); return
                with _db_lock:
                    with db() as con:
                        row = con.execute('SELECT * FROM generations WHERE id=?', (gen_id,)).fetchone()
                if not row:
                    self._json({'ok': False, 'error': 'Nie znaleziono generacji'}); return
                g = dict(row)
                paths = json.loads(g.get('paths') or '[]')
                if not paths or path_idx >= len(paths):
                    self._json({'ok': False, 'error': 'Brak pliku o tym indeksie'}); return
                file_path = OUTPUTS_DIR / paths[path_idx]
                if not file_path.exists():
                    self._json({'ok': False, 'error': f'Plik nie istnieje: {paths[path_idx]}'}); return

                boundary = uuid.uuid4().hex
                meta_fields = {
                    'ts':        str(g.get('ts', '')),
                    'positive':  g.get('positive', ''),
                    'negative':  g.get('negative', ''),
                    'model':     g.get('model', ''),
                    'sampler':   g.get('sampler', ''),
                    'scheduler': g.get('scheduler', ''),
                    'steps':     str(g.get('steps', '')),
                    'cfg':       str(g.get('cfg', '')),
                    'width':     str(g.get('width', '')),
                    'height':    str(g.get('height', '')),
                    'seed':      str(g.get('seed', '')),
                    'preset':    g.get('preset', ''),
                    'source':    'genphoto',
                    'is_xxx':    str(is_xxx),
                }
                parts = bytearray()
                for k, v in meta_fields.items():
                    parts += (
                        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                    ).encode()
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                parts += (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    f'filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
                ).encode() + file_data + f'\r\n--{boundary}--\r\n'.encode()

                req = urllib.request.Request(
                    PORTAL_URL.rstrip('/') + '/api/publish',
                    data=bytes(parts),
                    headers={
                        'Content-Type': f'multipart/form-data; boundary={boundary}',
                        'X-Api-Key': PORTAL_KEY,
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                self._json({'ok': True, 'url': result.get('url', ''), 'uuid': result.get('uuid', '')})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        self.send_error(404)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'GenPhoto running on http://0.0.0.0:{PORT}')
    import socket as _sock
    class _DualStack(ThreadingHTTPServer):
        address_family = _sock.AF_INET6
        def server_bind(self):
            self.socket.setsockopt(_sock.IPPROTO_IPV6, _sock.IPV6_V6ONLY, 0)
            super().server_bind()
    server = _DualStack(('::', PORT), Handler)
    threading.Thread(target=_models_cache_worker, daemon=True).start()
    _refresh_models_once(rescan_disk=True)  # zaladuj cache przed pierwszym requestem
    server.serve_forever()
