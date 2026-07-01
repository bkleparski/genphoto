#!/usr/bin/env python3
"""GenPhoto — AI photo generation studio (frontend for Stable Diffusion Forge)"""

import base64, hashlib, html, json, mimetypes, os, secrets, shutil, sqlite3
import threading, time, uuid
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
KREA2_URL    = os.environ.get('GP_KREA2_URL', 'http://localhost:7870').rstrip('/')
RAVNET_FORGE_URL = os.environ.get('GP_RAVNET_FORGE_URL', 'https://forge-ravnet.ebartnet.pl').rstrip('/')
DEEPSEEK_KEY   = os.environ.get('GP_DEEPSEEK_KEY', '')
DEEPSEEK_MODEL = os.environ.get('GP_DEEPSEEK_MODEL', 'deepseek-v4-flash')
AI_PROVIDER    = os.environ.get('GP_AI_PROVIDER', 'deepseek')
OR_KEY         = os.environ.get('GP_OR_KEY', '')
OR_MODEL       = os.environ.get('GP_OR_MODEL', 'nousresearch/hermes-4-405b')
OR_VISION_MODEL = os.environ.get('GP_OR_VISION_MODEL', 'qwen/qwen2.5-vl-72b-instruct')
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

def db():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

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

# ── Sessions ──────────────────────────────────────────────────────────────────
SESSIONS = set()
_auto_cache = {}  # model_name -> {settings, ts}

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

# ── Presets ───────────────────────────────────────────────────────────────────
NEG = ('(worst quality:2), (low quality:2), (blurry:1.3), deformed, ugly, extra limbs, '
       'mutated hands, (bad anatomy:1.3), watermark, text, signature, cropped, '
       'painting, artwork, cartoon, anime, 3d render, digital art, illustration, '
       'plastic skin, fake, overexposed, oversaturated, unnatural skin')

PRESETS = [
    {'id':'krea_turbo','name':'Krea 2 Turbo','icon':'&#9889;',
     'model':'krea2_oss_turbo','sampler':'','scheduler':'',
     'steps':8, 'cfg':0.0, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'',
     'negative': ''},
    {'id':'krea_raw',  'name':'Krea 2 Raw',  'icon':'&#127922;',
     'model':'krea2_oss_raw','sampler':'','scheduler':'',
     'steps':28, 'cfg':3.5,'width':1024, 'height':1024, 'batch':1,
     'prefix':'',
     'negative': ''},
]

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
_downloads    = {}   # job_id -> {url,filename,percent,status,error,bytes_done,size}
MODELS_DIR    = Path('/home/bartek/forge/models/Stable-diffusion')

# Cache modeli Ravnet Forge (DGX, zdalny) — odnawiany w tle co 30s
_ravnet_models_cache = []
_ravnet_models_lock  = threading.Lock()

def _refresh_models_once():
    global _models_cache
    forge_post('/sdapi/v1/refresh-checkpoints', {})
    data = forge_get('/sdapi/v1/sd-models') or []
    try:
        krea_health = json.loads(urllib.request.urlopen(f'{KREA2_URL}/health', timeout=3).read())
        if krea_health.get('checkpoints'):
            for name, info in krea_health['checkpoints'].items():
                if info.get('exists'):
                    data.append({'model_name': f'krea2_{name}', 'title': f'Krea 2 {name}', 'hash': ''})
    except:
        pass
    with _models_lock:
        _models_cache = data

def _refresh_ravnet_models_once():
    global _ravnet_models_cache
    data = forge_get('/sdapi/v1/sd-models', timeout=15, base_url=RAVNET_FORGE_URL) or []
    with _ravnet_models_lock:
        _ravnet_models_cache = data


def _download_model_thread(job_id, url):
    import time as _t
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
            dest  = MODELS_DIR / fname
            d['filename'] = fname
            d['status']   = 'downloading'
            size = int(resp.headers.get('Content-Length', 0) or 0)
            d['size'] = size
            done = 0
            chunk = 1024 * 256
            with open(dest, 'wb') as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf: break
                    f.write(buf)
                    done += len(buf)
                    d['bytes_done'] = done
                    d['percent'] = int(done * 100 / size) if size else -1
        d['percent'] = 100
        d['status']  = 'done'
        _refresh_models_once()
    except Exception as e:
        d['status'] = 'error'
        d['error']  = str(e)
        if d.get('filename'):
            p = MODELS_DIR / d['filename']
            if p.exists(): p.unlink()

def _models_cache_worker():
    import time as _time
    while True:
        try:
            _refresh_models_once()
        except Exception:
            pass
        try:
            _refresh_ravnet_models_once()
        except Exception:
            pass
        _time.sleep(30)


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
    _flux_cache[filename] = result
    return result

def resolve_model(model_name):
    with _models_lock:
        models = list(_models_cache)
    for m in models:
        if m.get('model_name') == model_name:
            return m['title'], m.get('filename', '')
    return model_name, ''

def resolve_ravnet_model(model_name):
    with _ravnet_models_lock:
        models = list(_ravnet_models_cache)
    for m in models:
        if m.get('model_name') == model_name:
            return m['title'], m.get('filename', '')
    return model_name, ''

FLUX_ADDITIONAL_MODULES = [
    '/home/bartek/forge/models/text_encoder/clip_l.safetensors',
    '/home/bartek/forge/models/text_encoder/t5xxl_fp8_e4m3fn.safetensors',
    '/home/bartek/forge/models/VAE/ae.safetensors',
]

def model_override_settings(title, filename=''):
    s = {'sd_model_checkpoint': title.split(' [')[0], 'sd_vae': '', 'forge_additional_modules': []}
    flux = 'flux' in title.lower() or (filename and is_flux_checkpoint(filename))
    if flux:
        s['forge_additional_modules'] = FLUX_ADDITIONAL_MODULES
    return s

def forge_generate_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title, fname = resolve_model(params['model'])
        data = forge_post('/sdapi/v1/txt2img', {
            'prompt':          params['positive'],
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

# Ścieżki wewnątrz kontenera ravnet-forge na DGX (bind mounty /forge/models/...)
RAVNET_FLUX1_MODULES = [
    '/forge/models/text_encoder/clip_l.safetensors',
    '/forge/models/text_encoder/t5xxl_fp8_e4m3fn.safetensors',
    '/forge/models/VAE/ae.safetensors',
]
RAVNET_FLUX2_MODULES = [
    '/forge/models/text_encoder/mistral_3_small_flux2_bf16.safetensors',
    '/forge/models/VAE/full_encoder_small_decoder.safetensors',
]

def ravnet_model_override_settings(title):
    s = {'sd_model_checkpoint': title.split(' [')[0], 'sd_vae': '', 'forge_additional_modules': []}
    t = title.lower()
    if 'flux2' in t or 'klein' in t:
        s['forge_additional_modules'] = RAVNET_FLUX2_MODULES
    elif 'flux' in t:
        s['forge_additional_modules'] = RAVNET_FLUX1_MODULES
    return s

def ravnet_forge_generate_thread(params, jid):
    """Generate using remote Forge on Ravnet DGX (forge-ravnet.ebartnet.pl)."""
    try:
        job_set(jid, status='generating')
        title, fname = resolve_ravnet_model(params['model'])
        data = forge_post('/sdapi/v1/txt2img', {
            'prompt':          params['positive'],
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
            'override_settings': ravnet_model_override_settings(title),
            'override_settings_restore_afterwards': True,
            'save_images': False,
        }, timeout=600, base_url=RAVNET_FORGE_URL)
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
            name = f'gp_ravnet_{ts}_{i:02d}_s{s}.png'
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
                     params.get('preset','ravnet_forge'), json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

def krea_generate_thread(params, jid):
    """Generate using Krea 2 API."""
    try:
        job_set(jid, status='generating')
        
        # Determine checkpoint
        model = params.get('model', 'oss_turbo')
        
        # Krea 2 parameters
        body = {
            'prompt': params['positive'],
            'checkpoint': 'oss_turbo' if 'turbo' in model else 'oss_raw',
            'width': min(int(params['width']), 2048),
            'height': min(int(params['height']), 2048),
            'num_images': int(params.get('batch', 1)),
            'seed': int(params.get('seed', 0)),
        }
        
        # Steps: default 8 for turbo, 52 for raw
        if 'steps' in params and params['steps']:
            body['steps'] = int(params['steps'])
        
        # CFG: default 0.0 for turbo, 3.5 for raw
        if 'cfg' in params and params['cfg']:
            body['cfg'] = float(params['cfg'])
        
        # Call Krea 2 API
        img_data, headers = krea_post('/generate', body)
        gen_time = headers.get('X-Generation-Time', '?')
        seed = headers.get('X-Seed', str(body['seed']))
        
        # Save image
        today = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        name = f'krea_{ts}_00_s{seed}.png'
        (out_dir / name).write_bytes(img_data)
        paths = [f'genphoto/{today}/{name}']
        
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
                     int(seed), int(params.get('batch',1)),
                     params.get('preset','krea'), json.dumps(paths))
                )
        job_set(jid, status='done', images=paths, seed=seed, gen_id=gid)
        
        # Also generate on other batch images if requested
        for i in range(1, int(params.get('batch', 1))):
            body['seed'] = int(seed) + i
            img_data2, _ = krea_post('/generate', body)
            name2 = f'krea_{ts}_{i:02d}_s{int(seed)+i}.png'
            (out_dir / name2).write_bytes(img_data2)
            paths.append(f'genphoto/{today}/{name2}')
            
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

def ai_vision_describe(image_b64, style='anime'):
    style_prefix = VISION_STYLE_HINTS.get(style, VISION_STYLE_HINTS['anime'])
    if LOCAL_VISION_URL:
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
        OR_VISION_MODEL, image_b64, style_prefix, style,
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
LOGIN_PAGE = '''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenPhoto — logowanie</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
.box{background:#1e293b;border:1px solid #334155;border-radius:16px;padding:40px 36px;width:340px;box-shadow:0 24px 60px rgba(0,0,0,.5)}
h1{font-size:1.4rem;font-weight:700;margin-bottom:6px;text-align:center}
.sub{color:#64748b;font-size:.85rem;text-align:center;margin-bottom:28px}
label{display:block;font-size:.8rem;color:#94a3b8;margin-bottom:4px}
input{width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:10px 14px;border-radius:8px;font-size:.9rem;outline:none;margin-bottom:14px;transition:border .2s}
input:focus{border-color:#3b82f6}
button{width:100%;background:#3b82f6;color:#fff;border:none;padding:11px;border-radius:8px;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
button:hover{background:#2563eb}
.err{color:#fca5a5;font-size:.82rem;margin-bottom:12px;text-align:center}
</style>
</head>
<body>
<div class="box">
  <h1>&#127912; GenPhoto</h1>
  <div class="sub">AI Photo Generation Studio</div>
  __ERR__
  <form method="post" action="/login">
    <label>Użytkownik</label>
    <input name="username" autocomplete="username" autofocus>
    <label>Hasło</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit">Zaloguj</button>
  </form>
</div>
</body>
</html>'''

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenPhoto</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{overflow-x:hidden}
body{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;min-height:100vh;overflow-x:hidden}
a{color:inherit;text-decoration:none}

/* Header */
header{background:#1e293b;border-bottom:1px solid #334155;padding:0 16px;min-height:52px;display:flex;align-items:center;gap:8px;position:sticky;top:0;z-index:100}
.logo{font-size:1.05rem;font-weight:700;color:#93c5fd;white-space:nowrap;flex-shrink:0}
.hdr-body{display:flex;align-items:center;gap:6px;flex:1;min-width:0}
.preset-tabs{display:flex;gap:5px;overflow-x:auto;scrollbar-width:none;flex-shrink:0}
.preset-tabs::-webkit-scrollbar{display:none}
.preset-btn{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:5px 12px;border-radius:20px;font-size:.78rem;cursor:pointer;white-space:nowrap;transition:all .2s;display:flex;align-items:center;gap:4px}
.preset-btn:hover{border-color:#60a5fa;color:#e2e8f0}
.preset-btn.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;font-weight:600}
.auto-btn{background:#0f2b4d;border:1px solid #2563eb;color:#93c5fd;padding:5px 13px;border-radius:20px;font-size:.78rem;cursor:pointer;white-space:nowrap;transition:all .2s;flex-shrink:0;font-weight:600}
.hdr-gen-btn{flex-shrink:0}
.auto-btn:hover{border-color:#60a5fa;color:#e2e8f0;background:#1e3a5f}
.auto-model-btn{background:#1a0f2e;border-color:#7c3aed;color:#c4b5fd}
.auto-model-btn:hover{background:#2d1b69;border-color:#a78bfa;color:#ede9fe}
.hdr-spacer{flex:1}
.hdr-links{display:flex;gap:6px;align-items:center;flex-shrink:0}
.vram-widget{display:flex;align-items:center;gap:8px;background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;padding:4px 10px;flex-shrink:0}
.vram-canvas{display:block;border-radius:4px}
.vram-info{display:flex;flex-direction:column;align-items:flex-end;line-height:1.2;min-width:70px}
.vram-nums{font-size:.78rem;font-weight:600;font-family:monospace;color:#93c5fd}
.vram-label{font-size:.62rem;color:#475569;white-space:nowrap}
.vram-free-btn{background:#7f1d1d;border:1px solid #991b1b;color:#fca5a5;padding:3px 9px;border-radius:5px;font-size:.72rem;cursor:pointer;white-space:nowrap;transition:all .15s}
.vram-free-btn:hover{background:#991b1b;border-color:#ef4444;color:#fff}
.vram-free-btn:disabled{opacity:.5;cursor:default}
.hdr-btn{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:5px 12px;border-radius:6px;font-size:.78rem;cursor:pointer;transition:all .2s;white-space:nowrap}
.hdr-btn:hover{border-color:#475569;color:#e2e8f0}
.hdr-btn.primary{background:#3b82f6;border-color:#3b82f6;color:#fff;font-weight:600}
.hdr-btn.primary:hover{background:#2563eb}

/* Layout */
main{max-width:1200px;margin:0 auto;padding:24px 20px;display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:780px){main{grid-template-columns:1fr}}

/* Form panel */
.panel{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:22px}
.panel-title{font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px}
textarea,input[type=text],input[type=number],select{width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:9px 13px;border-radius:8px;font-size:.88rem;outline:none;transition:border .2s;font-family:inherit;resize:vertical}
textarea:focus,input:focus,select:focus{border-color:#3b82f6}
select{resize:none;cursor:pointer}
.field{margin-bottom:14px}
.model-row{display:flex;gap:8px;align-items:center}
.model-row select{flex:1;min-width:0}
.field label{display:block;font-size:.75rem;color:#94a3b8;margin-bottom:5px;font-weight:500}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}

/* AI prompt row */
.ai-row{display:flex;gap:8px;margin-bottom:14px}
.ai-row textarea{flex:1;height:70px}
.ai-btn{background:#7c3aed;border:none;color:#fff;padding:8px 14px;border-radius:8px;font-size:.82rem;cursor:pointer;white-space:nowrap;font-weight:600;transition:background .2s;align-self:stretch}
.ai-btn:hover{background:#6d28d9}
.ai-btn:disabled{opacity:.5;cursor:not-allowed}

/* Prompt textarea */
#positive-ta{height:90px;font-size:.82rem;color:#bfdbfe}
#negative-ta{height:60px;font-size:.78rem;color:#fca5a5}

/* Params bar */
#params-bar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;padding:10px 12px;background:#0f172a;border:1px solid #1e3a5f;border-radius:8px}
.pb-item{font-size:.72rem;color:#64748b;display:flex;flex-direction:column;gap:1px}
.pb-item strong{color:#93c5fd;font-size:.8rem;font-weight:600}
.pb-sep{width:1px;background:#1e3a5f;margin:0 4px;align-self:stretch}

/* Advanced toggle */
.adv-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;color:#64748b;font-size:.8rem;margin-bottom:0;user-select:none;padding:8px 0;border-top:1px solid #1e3a5f;margin-top:4px}
.adv-toggle:hover{color:#94a3b8}
.adv-toggle svg{transition:transform .2s}
.adv-toggle.open svg{transform:rotate(180deg)}
#adv-section{display:block;margin-top:14px;padding-top:14px;border-top:1px solid #334155}
.meta-import-box{background:#0c1e35;border:1.5px dashed #1e3a5f;border-radius:10px;padding:10px 14px;margin-bottom:14px;display:flex;align-items:center;gap:10px;cursor:pointer;transition:border-color .2s}
.meta-import-box:hover,.meta-import-box.drag{border-color:#3b82f6;background:#0f2744}
.meta-import-icon{font-size:1.3rem;flex-shrink:0}
.meta-import-label{font-size:.78rem;color:#64748b;flex:1}
.meta-import-label strong{display:block;color:#93c5fd;margin-bottom:2px;font-size:.82rem}
.meta-import-status{font-size:.75rem;color:#22d3ee;white-space:nowrap}
#adv-section.open{display:block}

/* Generate button */
.gen-btn{width:100%;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border:none;color:#fff;padding:13px;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:16px;transition:opacity .2s;letter-spacing:.02em}
.gen-btn:hover{opacity:.9}
.gen-btn:disabled{opacity:.5;cursor:not-allowed}

/* Results panel */
.results-panel{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:22px;min-height:300px;display:flex;flex-direction:column}
#results-placeholder{color:#334155;font-size:.9rem;text-align:center;margin:auto;line-height:2}
#results-placeholder .big{font-size:2.5rem}

/* Progress */
#progress-wrap{display:none;margin-bottom:16px}
.progress-label{font-size:.78rem;color:#64748b;margin-bottom:6px;display:flex;justify-content:space-between}
.progress-bar-bg{background:#0f172a;border-radius:4px;height:6px;overflow:hidden}
.progress-bar-fill{background:linear-gradient(90deg,#3b82f6,#8b5cf6);height:100%;border-radius:4px;transition:width .5s;width:0%}

/* Image grid */
#img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.res-img{width:100%;aspect-ratio:auto;border-radius:8px;cursor:pointer;border:2px solid transparent;transition:border .2s;display:block;object-fit:cover}
.res-img:hover{border-color:#3b82f6}
#seed-info{font-size:.72rem;color:#475569;margin-top:10px}

/* History */
#history-section{grid-column:1/-1;background:#1e293b;border:1px solid #334155;border-radius:14px;padding:22px}
.hist-header{display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}
.hist-header h2{font-size:.9rem;font-weight:600;color:#94a3b8;display:flex;align-items:center;gap:8px}
#hist-body{margin-top:16px;display:flex;flex-direction:column;gap:10px}
.hist-item{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:14px 16px;display:flex;gap:14px;align-items:flex-start;overflow:hidden;min-width:0}
.hist-thumbs{display:flex;gap:4px;flex-shrink:0}
.hist-thumb{width:48px;height:48px;object-fit:cover;border-radius:6px;cursor:pointer;border:1px solid #334155}
.hist-meta{flex:1;min-width:0}
.hist-desc{font-size:.85rem;color:#e2e8f0;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:3px}
.hist-tags{font-size:.72rem;color:#64748b;margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-actions{display:flex;gap:6px;flex-wrap:wrap}
.hist-btn{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:4px 10px;border-radius:6px;font-size:.72rem;cursor:pointer;transition:all .2s}
.hist-btn:hover{border-color:#475569;color:#e2e8f0}
.hist-btn.del{color:#f87171}
.hist-btn.del:hover{border-color:#ef4444;color:#ef4444;background:#1e293b}
.hist-btn.info{color:#60a5fa}
.hist-btn.info:hover{border-color:#3b82f6;color:#93c5fd}
#meta-modal{display:none;position:fixed;inset:0;z-index:9000;background:rgba(0,0,0,.75);backdrop-filter:blur(4px);align-items:center;justify-content:center}
#meta-modal.open{display:flex}
.meta-box{background:#0f172a;border:1px solid #334155;border-radius:14px;padding:24px;max-width:680px;width:95%;max-height:90vh;overflow-y:auto;position:relative}
.meta-box h3{margin:0 0 16px;color:#93c5fd;font-size:1rem;display:flex;align-items:center;gap:8px}
.meta-close{position:absolute;top:14px;right:16px;background:none;border:none;color:#64748b;font-size:1.3rem;cursor:pointer;line-height:1}
.meta-close:hover{color:#f1f5f9}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;margin-bottom:16px}
.meta-field{display:flex;flex-direction:column;gap:2px}
.meta-field.full{grid-column:1/-1}
.meta-label{font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.meta-val{font-size:.8rem;color:#e2e8f0;background:#1e293b;border-radius:6px;padding:5px 8px;word-break:break-all;line-height:1.45;min-height:26px}
.meta-val.prompt{font-size:.77rem;max-height:100px;overflow-y:auto}
.meta-actions{display:flex;gap:8px;margin-top:4px}
.meta-load-btn{flex:1;background:#1e3a5f;border:1px solid #2563eb;color:#93c5fd;padding:7px 14px;border-radius:8px;font-size:.8rem;cursor:pointer;transition:all .2s}
.meta-load-btn:hover{background:#2563eb;color:#fff}
.no-hist{color:#334155;text-align:center;padding:20px;font-size:.85rem}

/* Lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:2000;align-items:center;justify-content:center;flex-direction:row;gap:0}
#lb.open{display:flex}
#lb-main{display:flex;flex-direction:column;align-items:center;flex-shrink:0}
#lb-img{max-width:70vw;max-height:86vh;object-fit:contain;border-radius:8px}
#lb-nav{display:flex;gap:16px;margin-top:14px;align-items:center}
#lb-nav button{background:#1e293b;border:1px solid #475569;color:#e2e8f0;padding:6px 18px;border-radius:8px;cursor:pointer;font-size:.85rem}
#lb-nav button:hover{background:#334155}
#lb-dl{background:#3b82f6;border-color:#3b82f6;color:#fff}
#lb-close{position:absolute;top:14px;right:18px;background:none;border:none;color:#64748b;font-size:1.6rem;cursor:pointer;line-height:1}
#lb-meta{width:300px;min-width:260px;max-width:320px;height:90vh;overflow-y:auto;background:#0f172a;border-left:1px solid #1e293b;padding:16px 14px;display:flex;flex-direction:column;gap:10px;flex-shrink:0}
#lb-meta h4{color:#94a3b8;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;margin:0 0 2px}
.lb-meta-val{font-size:.78rem;color:#e2e8f0;background:#1e293b;border-radius:6px;padding:6px 8px;word-break:break-all;line-height:1.5;white-space:pre-wrap}
.lb-meta-val.prompt{max-height:130px;overflow-y:auto}
.lb-meta-copy{background:none;border:none;color:#64748b;cursor:pointer;font-size:.75rem;padding:2px 4px;border-radius:4px;float:right}
.lb-meta-copy:hover{color:#e2e8f0;background:#1e293b}
.lb-meta-badge{display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:4px;padding:2px 8px;font-size:.75rem}
.lb-meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.lb-meta-kv{background:#1e293b;border-radius:6px;padding:5px 8px}
.lb-meta-kv span:first-child{display:block;font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}
.lb-meta-kv span:last-child{font-size:.78rem;color:#e2e8f0}
#lb-meta-empty{color:#334155;font-size:.8rem;text-align:center;margin-top:20px}

/* Toast */
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(80px);background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:10px 20px;border-radius:8px;font-size:.85rem;z-index:3000;transition:transform .3s;pointer-events:none}
#toast.show{transform:translateX(-50%) translateY(0)}
#toast.ok{border-color:#22c55e;color:#86efac}
#toast.err{border-color:#ef4444;color:#fca5a5}
#toast.info{border-color:#3b82f6;color:#93c5fd}
#model-mgr-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9000;align-items:center;justify-content:center}
#model-mgr-modal.open{display:flex}
.mgr-box{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:22px 24px;width:min(560px,95vw);max-height:80vh;display:flex;flex-direction:column;gap:12px}
.mgr-title{font-size:1rem;font-weight:700;color:#e2e8f0}
.mgr-dl-row{display:flex;gap:8px}
.mgr-dl-row input{flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px 12px;color:#e2e8f0;font-size:.85rem}
.mgr-dl-row input:focus{outline:none;border-color:#3b82f6}
.mgr-dl-btn{background:#1e3a5f;border:1px solid #3b82f6;color:#93c5fd;padding:8px 16px;border-radius:8px;font-size:.83rem;cursor:pointer;white-space:nowrap}
.mgr-dl-btn:hover{background:#2563eb;color:#fff}
.dl-progress{background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;padding:10px 14px;display:none}
.dl-progress.show{display:block}
.dl-bar-bg{background:#1e293b;border-radius:4px;height:8px;margin:6px 0;overflow:hidden}
.dl-bar-fill{background:linear-gradient(90deg,#3b82f6,#06b6d4);height:100%;width:0%;transition:width .4s;border-radius:4px}
.dl-info{font-size:.75rem;color:#64748b}
.mgr-list{overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:6px}
.mgr-item{display:flex;align-items:center;gap:10px;background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;padding:8px 12px}
.mgr-item-name{flex:1;font-size:.82rem;color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mgr-item-size{font-size:.72rem;color:#64748b;flex-shrink:0}
.mgr-del-btn{background:transparent;border:1px solid #7f1d1d;color:#f87171;padding:4px 10px;border-radius:6px;font-size:.72rem;cursor:pointer;flex-shrink:0}
.mgr-del-btn:hover{background:#7f1d1d;color:#fff}
.mgr-drag-handle{color:#334155;cursor:grab;font-size:1rem;flex-shrink:0;user-select:none;padding:0 4px}
.mgr-drag-handle:active{cursor:grabbing}
.mgr-item.dnd-dragging{opacity:.4}
.mgr-item.dnd-over{border-color:#3b82f6;background:#0f2847}
.mgr-manage-btn{background:transparent;border:none;color:#475569;font-size:.72rem;cursor:pointer;padding:4px 8px;margin-top:2px;display:block;width:100%;text-align:left}
.mgr-manage-btn:hover{color:#94a3b8}

/* View tabs */
.view-tabs{display:flex;gap:4px;flex-shrink:0}
.view-tab-btn{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:5px 14px;border-radius:20px;font-size:.8rem;cursor:pointer;transition:all .2s;white-space:nowrap}
.view-tab-btn.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;font-weight:600}
.view-tab-btn:hover{border-color:#60a5fa;color:#e2e8f0}

/* Edit view layout */
.edit-layout{max-width:1200px;margin:0 auto;padding:24px 20px;display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:780px){.edit-layout{grid-template-columns:1fr}}
.edit-canvas-wrap{position:relative;display:block;max-width:100%}
.edit-canvas-wrap img{display:block;max-width:100%;border-radius:8px}
#mask-canvas{position:absolute;top:0;left:0;width:100%;height:100%;border-radius:8px;cursor:crosshair;opacity:.55;pointer-events:auto;touch-action:none;user-select:none}
.upload-area{border:2px dashed #334155;border-radius:12px;padding:40px 20px;text-align:center;color:#64748b;cursor:pointer;transition:border .2s;margin-bottom:14px}
.upload-area:hover,.upload-area.drag{border-color:#3b82f6;color:#93c5fd}
.mask-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.mask-btn{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:5px 11px;border-radius:6px;font-size:.76rem;cursor:pointer;transition:all .2s}
.mask-btn.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd}
.mask-btn:hover{border-color:#475569;color:#e2e8f0}
.denoising-wrap input[type=range]{width:100%;margin:6px 0;accent-color:#3b82f6}
.denoising-labels{display:flex;justify-content:space-between;font-size:.7rem;color:#475569}
.edit-btn{width:100%;background:linear-gradient(135deg,#059669,#3b82f6);border:none;color:#fff;padding:13px;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:16px;transition:opacity .2s;letter-spacing:.02em}
.edit-btn:hover{opacity:.9}
.edit-btn:disabled{opacity:.5;cursor:not-allowed}
.edit-hist-wrap{max-width:1200px;margin:0 auto;padding:0 20px 24px}

/* Edit result wrappers */
.result-wrap{position:relative;display:inline-block;width:100%}
.result-overlay-btn{position:absolute;bottom:5px;left:4px;right:4px;background:rgba(15,23,42,.88);border:1px solid #334155;color:#94a3b8;padding:4px 6px;border-radius:6px;font-size:.7rem;cursor:pointer;text-align:center;opacity:0;transition:opacity .15s;pointer-events:none}
.result-wrap:hover .result-overlay-btn{opacity:1;pointer-events:auto}

/* ── Video view ── */
.vid-preset-btn{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:5px 11px;border-radius:6px;font-size:.76rem;cursor:pointer;transition:all .2s;white-space:nowrap}
.vid-preset-btn.active{background:#3b0764;border-color:#7c3aed;color:#c4b5fd}
.vid-preset-btn:hover{border-color:#475569;color:#e2e8f0}
.vid-btn{width:100%;background:linear-gradient(135deg,#7c3aed,#db2777);border:none;color:#fff;padding:13px;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:16px;transition:opacity .2s;letter-spacing:.02em}
.vid-btn:hover{opacity:.9}
.vid-btn:disabled{opacity:.5;cursor:not-allowed}
#vid-prog-fill{background:linear-gradient(90deg,#7c3aed,#db2777)}
.vid-hist-thumb{width:80px;height:56px;object-fit:cover;border-radius:5px;cursor:pointer;border:1px solid #334155}

/* ── Mobile responsive ── */
@media(max-width:640px){
  /* Linia 1: logo | Generuj | Galeria | logout */
  header{flex-wrap:wrap;height:auto;padding:5px 8px;gap:4px;row-gap:4px;align-items:center}
  .logo{flex-shrink:0;font-size:.88rem;order:1}
  .hdr-gen-btn{order:1;font-size:.75rem;padding:4px 10px}
  .hdr-links{order:1;margin-left:auto;gap:4px}
  .hdr-portal-link{display:none}
  /* Linia 2: presety | auto | edytuj | portret | poza — scroll poziomy */
  .hdr-body{order:2;flex:0 0 100%;border-top:1px solid #1e3a5f;padding-top:5px;margin-top:2px;overflow-x:auto;flex-wrap:nowrap;gap:4px}
  .hdr-spacer{display:none}
  .vram-widget{display:none}
  .view-tab-btn{padding:4px 9px;font-size:.73rem;white-space:nowrap}
  .hdr-gen-btn.view-tab-btn{padding:4px 10px}
  .hdr-btn{padding:4px 7px;font-size:.73rem}
  main{padding:12px 10px;gap:14px}
  .edit-layout{padding:12px 10px;gap:14px}
  .panel{padding:14px 12px}
  .edit-hist-wrap{padding:0 10px 16px}
  .row2{grid-template-columns:1fr 1fr}
  .row3{grid-template-columns:1fr 1fr 1fr}
  #lb{flex-direction:column}
  #lb-img{max-width:96vw;max-height:55vh}
  #lb-meta{width:96vw;max-width:100%;height:auto;max-height:30vh;border-left:none;border-top:1px solid #1e293b}
  #lb-nav{gap:8px;flex-wrap:wrap;justify-content:center}
  #lb-nav button{padding:6px 12px;font-size:.8rem}
  #lb-close{font-size:1.3rem;top:10px;right:12px}
  .mask-toolbar{gap:5px}
  .mask-btn{padding:6px 10px;font-size:.76rem}
  #params-bar{gap:5px;padding:8px 10px}
  .pb-item{font-size:.68rem}
  .pb-item strong{font-size:.74rem}
}
</style>
</head>
<body>

<header>
  <div class="logo">&#127912; GenPhoto</div>
  <button class="view-tab-btn hdr-gen-btn active" id="tab-gen" onclick="switchView('generate');switchBackend('forge');document.getElementById('tab-ravnet').classList.remove('active');this.classList.add('active');">&#127912; Generuj</button>
  <button class="view-tab-btn hdr-gen-btn" id="tab-ravnet" onclick="switchView('generate');switchBackend('ravnet');document.getElementById('tab-gen').classList.remove('active');this.classList.add('active');">&#128640; RAVNET-FORGE</button>
  <div class="hdr-body">
    <div class="preset-tabs" id="preset-tabs">__PRESET_TABS__</div>
    <div class="hdr-spacer"></div>
    <div class="view-tabs">
      <button class="view-tab-btn" id="tab-edit" onclick="switchView('edit')">&#9999;&#65039; Edytuj</button>
      <button class="view-tab-btn" id="tab-portrait" onclick="switchView('portrait')">&#128100; Portret</button>
      <button class="view-tab-btn" id="tab-pose" onclick="switchView('pose')">&#128694; Poza</button>
    </div>
    <div class="vram-widget" id="vram-widget">
      <canvas class="vram-canvas" id="vram-canvas" width="80" height="28"></canvas>
      <div class="vram-info">
        <span class="vram-nums" id="vram-nums">-- / --</span>
        <span class="vram-label">GB zajęte / wolne</span>
      </div>
      <button class="vram-free-btn" id="vram-free-btn" onclick="vramFree()">ZWOLNIJ</button>
    </div>
  </div>
  <div class="hdr-links">
    <a href="__GALLERY_URL__" target="_blank" class="hdr-btn">&#128193; Galeria</a>
    <a href="__PORTAL_URL__" target="_blank" class="hdr-btn hdr-portal-link">&#9889; Portal</a>
    <a href="/logout" class="hdr-btn">&#10155;</a>
  </div>
</header>


<!-- Modal zarządzania modelami -->
<div id="model-mgr-modal">
  <div class="mgr-box">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <div class="mgr-title">&#128194; Zarządzaj modelami</div>
      <button onclick="closeMgrModal()" style="background:none;border:none;color:#64748b;font-size:1.2rem;cursor:pointer">&#10005;</button>
    </div>
    <div>
      <div style="font-size:.75rem;color:#64748b;margin-bottom:6px">Pobierz model z Civitai (wklej link pobierania):</div>
      <div class="mgr-dl-row">
        <input id="mgr-url-input" type="url" placeholder="https://civitai.red/api/download/models/...">
        <button class="mgr-dl-btn" onclick="startDownload()">&#11015; Pobierz</button>
      </div>
      <div class="dl-progress" id="dl-progress">
        <div class="dl-info" id="dl-info">Przygotowywanie...</div>
        <div class="dl-bar-bg"><div class="dl-bar-fill" id="dl-bar-fill"></div></div>
        <div class="dl-info" id="dl-bytes">0 MB / 0 MB</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between">
      <div style="font-size:.75rem;color:#64748b">Zainstalowane modele:</div>
      <button onclick="refreshMgrList()" style="background:none;border:1px solid #334155;color:#94a3b8;padding:3px 10px;border-radius:6px;font-size:.72rem;cursor:pointer">&#8635; Odśwież</button>
    </div>
    <div class="mgr-list" id="mgr-list">
      <div style="color:#475569;font-size:.82rem">Ładowanie...</div>
    </div>
  </div>
</div>
<div id="publish-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;align-items:center;justify-content:center;">
  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:22px 24px;max-width:360px;width:90%;">
    <div style="font-size:16px;font-weight:700;color:#e2e8f0;margin:0 0 8px;">&#9889; Opublikuj zdjęcie</div>
    <div style="color:#94a3b8;font-size:13px;margin:0 0 18px;">Zdjęcie będzie widoczne publicznie na <b>images.ebartnet.pl</b></div>
    <div style="display:flex;flex-direction:column;gap:8px;">
      <button onclick="doPublish(false)" style="padding:10px;background:#6366f1;border:none;border-radius:7px;color:#fff;cursor:pointer;font-size:14px;font-weight:600;">&#128444; Opublikuj normalnie</button>
      <button onclick="doPublish(true)" style="padding:10px;background:#7f1d1d;border:1px solid #991b1b;border-radius:7px;color:#fca5a5;cursor:pointer;font-size:14px;font-weight:600;">&#128286; Opublikuj jako XXX</button>
      <button onclick="document.getElementById('publish-modal').style.display='none'" style="padding:8px;background:transparent;border:1px solid #334155;border-radius:7px;color:#94a3b8;cursor:pointer;font-size:13px;">Anuluj</button>
    </div>
    <input type="hidden" id="publish-gen-id">
    <input type="hidden" id="publish-path-idx">
  </div>
</div>

<div id="view-generate">
<main>

<!-- ── Form ── -->
<div class="panel">
  <div class="panel-title">&#128221; Opis &amp; Prompt</div>

  <div id="params-bar">
    <div class="pb-item"><span>Model</span><strong id="pb-model">—</strong></div>
    <div class="pb-sep"></div>
    <div class="pb-item"><span>Sampler</span><strong id="pb-sampler">—</strong></div>
    <div class="pb-sep"></div>
    <div class="pb-item"><span>Steps</span><strong id="pb-steps">—</strong></div>
    <div class="pb-sep"></div>
    <div class="pb-item"><span>CFG</span><strong id="pb-cfg">—</strong></div>
    <div class="pb-sep"></div>
    <div class="pb-item"><span>Wymiary</span><strong id="pb-size">—</strong></div>
    <div class="pb-sep"></div>
    <div class="pb-item"><span>Batch</span><strong id="pb-batch">—</strong></div>
  </div>

  <div class="meta-import-box" id="meta-import-box"
       onclick="document.getElementById('meta-import-file').click()"
       ondragover="event.preventDefault();this.classList.add('drag')"
       ondragleave="this.classList.remove('drag')"
       ondrop="event.preventDefault();this.classList.remove('drag');metaImportFile(event.dataTransfer.files[0])">
    <span class="meta-import-icon">&#128228;</span>
    <div class="meta-import-label">
      <strong>Wczytaj metadane ze zdjęcia</strong>
      Przeciągnij PNG lub kliknij — automatycznie wypełni prompt i parametry
    </div>
    <span class="meta-import-status" id="meta-import-status"></span>
    <input type="file" id="meta-import-file" accept="image/png,image/jpeg,image/webp" style="display:none"
           onchange="metaImportFile(this.files[0])">
  </div>

  <div class="field">
    <label>Opisz co chcesz wygenerować (po polsku)</label>
    <div class="ai-row">
      <textarea id="desc-ta" placeholder="np. piękna kobieta na plaży o zachodzie słońca, naturalny uśmiech, letnia sukienka..."></textarea>
      <button class="ai-btn" id="ai-btn" onclick="genAiPrompt()">&#10024; AI<br>Prompt</button>
    </div>
  </div>

  <div class="field">
    <label>&#128247; Zdjęcie referencyjne <span style="color:#475569;font-size:.78rem;font-weight:400">(opcjonalne — img2img)</span></label>
    <div id="ref-drop" class="upload-area" style="padding:16px 20px;margin-bottom:0"
         onclick="document.getElementById(\'ref-file-in\').click()"
         ondragover="event.preventDefault();this.classList.add(\'drag\')"
         ondragleave="this.classList.remove(\'drag\')"
         ondrop="event.preventDefault();this.classList.remove(\'drag\');loadRefFile(event.dataTransfer.files[0])">
      <div id="ref-placeholder">&#128247; Kliknij lub przeciągnij zdjęcie referencyjne</div>
      <div id="ref-loaded" style="display:none;text-align:left">
        <img id="ref-thumb" style="max-height:140px;max-width:100%;border-radius:8px;display:block;margin:0 auto 8px">
        <div style="display:flex;gap:8px;align-items:center;justify-content:center">
          <span id="ref-fname" style="font-size:.78rem;color:#94a3b8"></span>
          <button onclick="clearRef(event)" style="background:#334155;border:none;color:#94a3b8;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:.75rem">&#10005; Usuń</button>
        </div>
      </div>
    </div>
    <input type="file" id="ref-file-in" accept="image/*" style="display:none" onchange="loadRefFile(this.files[0])">
    <div id="ref-controls" style="display:none;margin-top:10px;background:#1a2535;border:1px solid #2d3f55;border-radius:10px;padding:14px">
      <div class="row2" style="margin-bottom:10px">
        <div class="field" style="margin-bottom:0">
          <label>Styl generowania</label>
          <select id="ref-style-sel">
            <option value="anime">&#127912; Anime</option>
            <option value="photo">&#128247; Fotorealistyczny</option>
            <option value="portrait">&#128100; Portret</option>
          </select>
        </div>
        <div class="field" style="margin-bottom:0">
          <label>Siła zmiany&#58; <span id="denoising-val">0.65</span></label>
          <input type="range" id="denoising-range" min="0.1" max="1.0" step="0.05" value="0.65"
                 style="width:100%;accent-color:#3b82f6;margin-top:6px"
                 oninput="document.getElementById(\'denoising-val\').textContent=parseFloat(this.value).toFixed(2)">
        </div>
      </div>
      <button onclick="describeRef()" id="ref-desc-btn"
              style="width:100%;background:#0f4c8a;border:none;color:#93c5fd;padding:9px;border-radius:8px;cursor:pointer;font-size:.88rem;font-weight:600">
        &#128269; Opisz zdjęcie i wygeneruj prompt
      </button>
    </div>
  </div>

  <div class="field">
    <label>Positive prompt</label>
    <textarea id="positive-ta" placeholder="masterpiece, best quality..."></textarea>
  </div>

  <div class="adv-toggle open" id="adv-toggle" onclick="toggleAdv()">
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="2,4 7,10 12,4"/></svg>
    Parametry generowania
  </div>

  <div id="adv-section" class="open">
    <div class="field">
      <label>Backend</label>
      <div style="display:flex;gap:8px">
        <button id="backend-forge" class="backend-btn active" onclick="switchBackend('forge')" style="flex:1;padding:6px 12px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:.8rem;transition:all .2s">&#9881; Forge</button>
        <button id="backend-krea2" class="backend-btn" onclick="switchBackend('krea2')" style="flex:1;padding:6px 12px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:.8rem;transition:all .2s">&#9889; Krea 2</button>
        <button id="backend-ravnet" class="backend-btn" onclick="switchBackend('ravnet')" style="flex:1;padding:6px 12px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;font-size:.8rem;transition:all .2s">&#128640; Ravnet</button>
      </div>
    </div>
    <div class="field">
      <label>Model</label>
      <div class="model-row">
        <select id="model-sel" onchange="markCustom();onModelChange(this.value)">__MODEL_OPTIONS__</select>
        <button class="auto-btn" onclick="autoSettings()" title="Dobierz optymalne ustawienia do modelu">&#9881; Auto</button>
        <button class="auto-btn auto-model-btn" onclick="autoModel()" title="AI dobierze najlepszy model do promptu">&#129302; Model</button>
      </div>
      <button class="mgr-manage-btn" onclick="openMgrModal()">&#9881; Zarządzaj modelami / pobierz nowy</button>
    </div>
    <div class="row2">
      <div class="field krea2-hide">
        <label>Sampler</label>
        <select id="sampler-sel" onchange="markCustom()">
          <option>DPM++ SDE</option>
          <option>DPM++ 2M SDE</option>
          <option>DPM++ 3M SDE</option>
          <option>DPM++ 2M</option>
          <option>Euler a</option>
          <option>Euler</option>
        </select>
      </div>
      <div class="field">
        <label>Scheduler</label>
        <select id="sched-sel" onchange="markCustom()">
          <option>Karras</option>
          <option>Exponential</option>
          <option>SGM Uniform</option>
          <option>Simple</option>
          <option>DDIM Uniform</option>
        </select>
      </div>
    </div>
    <div class="row3">
      <div class="field"><label>Steps</label><input type="number" id="steps-in" value="25" min="5" max="80" onchange="markCustom()"></div>
      <div class="field"><label>CFG Scale</label><input type="number" id="cfg-in" value="6" min="1" max="20" step="0.5" onchange="markCustom()"></div>
      <div class="field"><label>Seed</label><input type="number" id="seed-in" value="-1" onchange="markCustom()"></div>
    </div>
    <div class="row3">
      <div class="field" style="grid-column:1/-1">
        <div style="display:flex;gap:8px;align-items:flex-end">
          <div class="field" style="flex:1;margin:0">
            <label>Szerokość</label>
            <input type="number" id="w-in" value="832" step="8" min="256" max="2048" onchange="markCustom()">
          </div>
          <button onclick="swapWH()" title="Zamień szerokość ↔ wysokość"
                  style="flex:0 0 auto;background:#1e3a5f;border:1px solid #3b82f6;color:#93c5fd;border-radius:8px;padding:0 12px;height:38px;cursor:pointer;font-size:1.1rem;margin-bottom:0;transition:background .15s"
                  onmouseover="this.style.background='#1d4ed8'" onmouseout="this.style.background='#1e3a5f'">⇄</button>
          <div class="field" style="flex:1;margin:0">
            <label>Wysokość</label>
            <input type="number" id="h-in" value="1216" step="8" min="256" max="2048" onchange="markCustom()">
          </div>
          <div class="field" style="flex:1;margin:0">
            <label>Batch</label>
            <input type="number" id="batch-in" value="4" min="1" max="8" onchange="markCustom()">
          </div>
        </div>
      </div>
    </div>
    <div class="field">
      <label>Negative prompt</label>
      <textarea id="negative-ta" onchange="markCustom()"></textarea>
    </div>
  </div>

  <button class="gen-btn" id="gen-btn" onclick="startGenerate()">&#127912; GENERUJ</button>
</div>

<!-- ── Results ── -->
<div class="results-panel" id="results-panel">
  <div id="results-placeholder">
    <div class="big">&#127912;</div>
    Wybierz preset i opisz scenerię<br>
    <span style="color:#334155;font-size:.8rem">Wygenerowane zdjęcia pojawią się tutaj</span>
  </div>
  <div id="progress-wrap">
    <div class="progress-label"><span id="prog-label">Generowanie...</span><span id="prog-pct"></span></div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="prog-fill"></div></div>
  </div>
  <div id="img-grid"></div>
  <div id="seed-info"></div>
</div>

<!-- ── History ── -->
<div id="history-section">
  <div class="hist-header" onclick="toggleHist()">
    <h2>&#128337; Historia generowań <span id="hist-count" style="color:#475569;font-weight:400"></span></h2>
    <svg id="hist-arrow" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="2"><polyline points="3,5 8,11 13,5"/></svg>
  </div>
  <div id="hist-body" style="display:none"></div>
</div>

</main>
</div><!-- /view-generate -->

<!-- ── Edit View ── -->
<div id="view-edit" style="display:none">
<div class="edit-layout">

  <!-- Lewa kolumna: upload + canvas -->
  <div class="panel">
    <div class="panel-title">&#128247; Zdjęcie do edycji</div>
    <div id="upload-area" class="upload-area"
         onclick="document.getElementById(\'edit-file-in\').click()"
         ondragover="event.preventDefault();this.classList.add(\'drag\')"
         ondragleave="this.classList.remove(\'drag\')"
         ondrop="handleEditDrop(event)">
      <div style="font-size:2rem">&#128247;</div>
      <div style="margin-top:8px;font-size:.85rem">Kliknij lub przeciągnij zdjęcie</div>
      <div style="font-size:.75rem;color:#475569;margin-top:4px">PNG, JPG, WEBP</div>
    </div>
    <input type="file" id="edit-file-in" accept="image/*" style="display:none" onchange="handleEditFile(this.files[0])">

    <div id="edit-canvas-wrap" class="edit-canvas-wrap" style="display:none">
      <img id="edit-src-img" src="" alt="" style="display:block;max-width:100%;border-radius:8px">
      <canvas id="mask-canvas"></canvas>
    </div>

    <div id="mask-toolbar" class="mask-toolbar" style="display:none">
      <button class="mask-btn active" id="btn-paint" onclick="setEditMode(\'paint\')">&#128396; Maluj</button>
      <button class="mask-btn" id="btn-erase" onclick="setEditMode(\'erase\')">&#9676; Gumka</button>
      <button class="mask-btn" onclick="clearMask()">&#10006; Wyczyść</button>
      <span style="color:#64748b;font-size:.75rem;margin-left:8px">Pędzel:</span>
      <input type="range" id="brush-size" min="5" max="80" value="20" style="width:80px;accent-color:#3b82f6" oninput="document.getElementById(\'brush-sz-lbl\').textContent=this.value+\'px\'">
      <span id="brush-sz-lbl" style="font-size:.75rem;color:#94a3b8;min-width:32px">20px</span>
    </div>
    <div id="mask-hint" style="display:none;font-size:.72rem;color:#475569;margin-top:6px">
      Zamaluj obszar do edycji — lub pozostaw pusty, by edytować całe zdjęcie
    </div>
  </div>

  <!-- Prawa kolumna: prompt + parametry -->
  <div class="panel">
    <div class="panel-title">&#128221; Opis zmiany &amp; Prompt</div>

    <div class="field">
      <label>Opisz co chcesz zmienić (po polsku)</label>
      <div class="ai-row">
        <textarea id="edit-desc-ta" placeholder="np. zmień kolor sukienki na czerwony, dodaj okulary..."></textarea>
        <button class="ai-btn" id="edit-ai-btn" onclick="genEditAiPrompt()">&#10024; AI<br>Prompt</button>
      </div>
    </div>

    <div class="field">
      <label>Positive prompt</label>
      <textarea id="edit-positive-ta" style="height:80px;font-size:.82rem;color:#bfdbfe" placeholder="Co ma być na zdjęciu..."></textarea>
    </div>
    <div class="field">
      <label>Negative prompt</label>
      <textarea id="edit-negative-ta" style="height:50px;font-size:.78rem;color:#fca5a5"></textarea>
    </div>

    <div class="field">
      <label>Skaluj zdjęcie przed wysłaniem do AI</label>
      <select id="scale-sel" onchange="onScaleChange()">
        <option value="0">Oryginał (wolno przy dużych zdjęciach)</option>
        <option value="2048">Max 2048px — wysoka jakość</option>
        <option value="1536" selected>Max 1536px — zalecane</option>
        <option value="1024">Max 1024px — szybkie</option>
        <option value="768">Max 768px — bardzo szybkie</option>
        <option value="512">Max 512px — najszybsze</option>
      </select>
      <div id="scale-info" style="font-size:.7rem;color:#475569;margin-top:4px"></div>
    </div>

    <div class="field denoising-wrap">
      <label>Inwazyjność edycji</label>
      <input type="range" id="denoising-sl" min="0.1" max="1.0" step="0.05" value="0.5" oninput="updateDenoisingLabel()">
      <div class="denoising-labels">
        <span>Subtelna korekta</span>
        <span id="denoising-val" style="color:#93c5fd;font-weight:600">0.50</span>
        <span>Silna przeróbka</span>
      </div>
    </div>

    <div class="adv-toggle" id="edit-adv-toggle" onclick="toggleEditAdv()">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="2,4 7,10 12,4"/></svg>
      Parametry generowania
    </div>
    <div id="edit-adv-section" style="display:none;margin-top:14px;padding-top:14px;border-top:1px solid #334155">
      <div class="field">
        <label>Model</label>
        <select id="edit-model-sel">__EDIT_MODEL_OPTIONS__</select>
      </div>
      <div class="row2">
        <div class="field"><label>Sampler</label>
          <select id="edit-sampler-sel">
            <option>DPM++ SDE</option><option>DPM++ 2M SDE</option><option>Euler a</option><option>Euler</option>
          </select>
        </div>
        <div class="field"><label>Scheduler</label>
          <select id="edit-sched-sel">
            <option>Karras</option><option>Exponential</option><option>Simple</option>
          </select>
        </div>
      </div>
      <div class="row3">
        <div class="field"><label>Steps</label><input type="number" id="edit-steps-in" value="25" min="5" max="80"></div>
        <div class="field"><label>CFG Scale</label><input type="number" id="edit-cfg-in" value="7" min="1" max="20" step="0.5"></div>
        <div class="field"><label>Seed</label><input type="number" id="edit-seed-in" value="-1"></div>
      </div>
      <div class="row2">
        <div class="field"><label>Szerokość</label><input type="number" id="edit-w-in" value="512" step="8" min="256" max="2048"></div>
        <div class="field"><label>Wysokość</label><input type="number" id="edit-h-in" value="512" step="8" min="256" max="2048"></div>
      </div>
    </div>

    <button class="edit-btn" id="edit-btn" onclick="startEdit()">&#9999;&#65039; EDYTUJ ZDJĘCIE</button>

    <div id="edit-progress-wrap" style="display:none;margin-top:16px">
      <div class="progress-label"><span id="edit-prog-label">Edytowanie...</span><span id="edit-prog-pct"></span></div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="edit-prog-fill"></div></div>
    </div>
    <div id="edit-img-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px;margin-top:12px"></div>
    <div id="edit-seed-info" style="font-size:.72rem;color:#475569;margin-top:8px"></div>

    <!-- Porównanie przed/po -->
    <div id="edit-comparison" style="display:none;margin-top:14px;padding-top:14px;border-top:1px solid #1e3a5f">
      <div style="font-size:.72rem;color:#64748b;text-align:center;margin-bottom:8px">&#128247; Porównanie</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div>
          <div style="font-size:.7rem;color:#475569;margin-bottom:3px;text-align:center">Oryginał</div>
          <img id="edit-before-img" style="width:100%;border-radius:6px;display:block" src="" alt="">
        </div>
        <div>
          <div style="font-size:.7rem;color:#475569;margin-bottom:3px;text-align:center">Po edycji</div>
          <img id="edit-after-img" style="width:100%;border-radius:6px;display:block;cursor:pointer" src="" alt="">
          <button id="edit-after-edit-btn" class="hist-btn" style="margin-top:5px;width:100%;text-align:center;font-size:.72rem">&#8635; Edytuj dalej</button>
        </div>
      </div>
    </div>
  </div>

</div><!-- /edit-layout -->

<!-- Historia edycji -->
<div class="edit-hist-wrap">
<div id="edit-history-section" class="panel" style="margin-top:0">
  <div class="hist-header" onclick="toggleEditHist()">
    <h2>&#128337; Historia edycji <span id="edit-hist-count" style="color:#475569;font-weight:400"></span></h2>
    <svg id="edit-hist-arrow" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="2"><polyline points="3,5 8,11 13,5"/></svg>
  </div>
  <div id="edit-hist-body" style="display:none;flex-direction:column;gap:10px"></div>
</div>
</div>

</div><!-- /view-edit -->

<!-- ── Portrait View ── -->
<div id="view-portrait" style="display:none">
<main>
<div class="left-panel">

  <div class="field">
    <label>Zdjęcie twarzy</label>
    <div id="portrait-drop" style="border:2px dashed var(--border);border-radius:10px;padding:20px;text-align:center;cursor:pointer;min-height:120px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px"
      onclick="document.getElementById(\'portrait-file-in\').click()"
      ondragover="event.preventDefault()"
      ondrop="event.preventDefault();loadPortraitFile(event.dataTransfer.files[0])">
      <div style="font-size:2rem">&#128100;</div>
      <div style="font-size:.85rem;color:var(--muted)">Kliknij lub przeciągnij swoje zdjęcie</div>
      <div style="font-size:.75rem;color:var(--muted)">JPG / PNG — twarz wyraźna, frontalnie</div>
    </div>
    <input type="file" id="portrait-file-in" accept="image/*" style="display:none" onchange="loadPortraitFile(this.files[0])">
    <div id="portrait-thumb-wrap" style="display:none;margin-top:10px;position:relative">
      <div style="display:flex;gap:12px;align-items:flex-start">
        <div style="flex:1;text-align:center">
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:4px">Wgrane zdjęcie</div>
          <img id="portrait-thumb" style="max-height:180px;border-radius:8px;border:1px solid var(--border)">
        </div>
        <div id="portrait-face-wrap" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:#22c55e;margin-bottom:4px">&#10003; Wykryta twarz</div>
          <img id="portrait-face-thumb" style="width:110px;height:110px;object-fit:cover;border-radius:8px;border:2px solid #22c55e">
        </div>
        <div id="portrait-face-loading" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:4px">Wykrywam...</div>
          <div style="width:110px;height:110px;border:2px dashed var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.5rem">&#128269;</div>
        </div>
        <div id="portrait-face-notfound" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:#ef4444;margin-bottom:4px">&#10007; Brak twarzy</div>
          <div style="width:110px;height:110px;border:2px dashed #ef4444;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.72rem;color:#ef4444;text-align:center;padding:6px">Wgraj bliższy kadr</div>
        </div>
      </div>
      <button onclick="clearPortrait()" style="position:absolute;top:0;right:0;background:#333;border:none;color:#fff;border-radius:50%;width:22px;height:22px;cursor:pointer;font-size:.75rem">&#x2715;</button>
    </div>
  </div>

  <div class="field">
    <label>Model</label>
    <select id="portrait-model-sel">__PORTRAIT_MODEL_OPTIONS__</select>
  </div>

  <div class="field">
    <label>Opis sceny / co ma być na zdjęciu</label>
    <textarea id="portrait-prompt" rows="4" placeholder="np. elegancki portret biznesowy w garniturze, biuro w tle, profesjonalne oświetlenie, 4k" style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:10px;color:var(--fg);font-size:.9rem;resize:vertical"></textarea>
  </div>

  <div class="field">
    <label>Negative prompt</label>
    <textarea id="portrait-negative" rows="2" placeholder="ugly, deformed, blurry, low quality" style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:10px;color:var(--fg);font-size:.9rem;resize:vertical">ugly, deformed, blurry, low quality, watermark, hair, beard, stubble, facial hair, added hair, different hairstyle</textarea>
  </div>

  <div class="field">
    <label>Siła zachowania twarzy: <span id="face-strength-val">0.80</span></label>
    <input type="range" id="face-strength" min="0.3" max="1.0" step="0.05" value="0.8"
      oninput="document.getElementById(\'face-strength-val\').textContent=parseFloat(this.value).toFixed(2)"
      style="width:100%;accent-color:var(--accent)">
    <div style="display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted)">
      <span>Luźniejsza (0.3)</span><span>Wierna (1.0)</span>
    </div>
  </div>

  <div class="field">
    <label>Siła pozycji twarzy: <span id="pose-strength-val">0.50</span></label>
    <input type="range" id="pose-strength" min="0.1" max="0.8" step="0.05" value="0.5"
      oninput="document.getElementById(\'pose-strength-val\').textContent=parseFloat(this.value).toFixed(2)"
      style="width:100%;accent-color:var(--accent)">
  </div>

  <div class="field" style="grid-column:1/-1">
    <div style="display:flex;gap:8px;align-items:flex-end">
      <div class="field" style="flex:1;margin:0">
        <label>Szerokość</label>
        <input type="number" id="portrait-w" value="832" min="512" max="2048" step="64"
          style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--fg)">
      </div>
      <button onclick="swapPortraitWH()" style="flex:0 0 auto;background:var(--inp);border:1px solid var(--border);border-radius:8px;color:var(--fg);cursor:pointer;height:38px;padding:0 10px;font-size:1rem">&#x21C4;</button>
      <div class="field" style="flex:1;margin:0">
        <label>Wysokość</label>
        <input type="number" id="portrait-h" value="1216" min="512" max="2048" step="64"
          style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--fg)">
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
    <div class="field" style="margin:0">
      <label>Steps</label>
      <input type="number" id="portrait-steps" value="30" min="10" max="60"
        style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--fg)">
    </div>
    <div class="field" style="margin:0">
      <label>CFG</label>
      <input type="number" id="portrait-cfg" value="5.0" min="1" max="15" step="0.5"
        style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--fg)">
    </div>
    <div class="field" style="margin:0">
      <label>Seed</label>
      <input type="number" id="portrait-seed" value="-1"
        style="width:100%;background:var(--inp);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--fg)">
    </div>
  </div>

  <div style="display:flex;gap:8px;margin-top:4px">
    <button onclick="startPortrait()" id="portrait-gen-btn"
      style="flex:1;padding:14px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border:none;border-radius:10px;color:#fff;font-size:1rem;font-weight:700;cursor:pointer;letter-spacing:.05em">
      &#128100; GENERUJ PORTRET
    </button>
  </div>
  <div id="portrait-status" style="margin-top:8px;font-size:.85rem;color:var(--muted);text-align:center"></div>
  <div id="portrait-progress-wrap" style="display:none;margin-top:10px">
    <div class="progress-label"><span id="portrait-prog-label">Generowanie portretu...</span><span id="portrait-prog-pct"></span></div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="portrait-prog-fill"></div></div>
    <div id="portrait-prog-detail" style="font-size:.75rem;color:var(--muted);margin-top:4px;min-height:1em"></div>
  </div>

</div><!-- /left-panel -->

<div class="right-panel" id="portrait-results" style="display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:flex-start;padding:16px">
  <div style="color:var(--muted);font-size:.9rem;margin-top:40px">Wygenerowane portrety pojawią się tutaj</div>
</div>

</main>
</div><!-- /view-portrait -->

<div id="view-pose" style="display:none">
<main>
<div class="left-panel">
  <div style="font-size:.8rem;color:var(--muted);margin-bottom:12px">
    Wgraj zdjęcie swojego ciała — model zachowa Twoją pozę i sylwetkę. Opcjonalnie wgraj też closeup twarzy dla lepszego podobieństwa.
  </div>

  <div style="margin-bottom:16px">
    <div style="font-size:.82rem;color:var(--muted);margin-bottom:6px">&#128694; Zdjęcie — twarz zostanie wykryta automatycznie <span style="color:#ef4444">*</span></div>
    <div id="pose-body-drop" onclick="document.getElementById('pose-body-file').click()"
      ondrop="event.preventDefault();loadPoseBodyFile(event.dataTransfer.files[0])"
      ondragover="event.preventDefault()"
      style="border:2px dashed var(--border);border-radius:8px;padding:16px;text-align:center;cursor:pointer;min-height:110px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;font-size:.8rem;color:var(--muted)">
      &#128694;<br>Kliknij lub upuść swoje zdjęcie<br><span style="font-size:.72rem;opacity:.7">Całe ciało lub portret — twarz wykrywana auto przez InsightFace</span>
    </div>
    <div id="pose-body-thumb-wrap" style="display:none;margin-top:10px;position:relative">
      <div style="display:flex;gap:12px;align-items:flex-start">
        <div style="flex:1;text-align:center">
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:4px">Wgrane zdjęcie</div>
          <img id="pose-body-thumb" style="max-height:180px;border-radius:8px;border:1px solid var(--border)">
        </div>
        <div id="pose-face-wrap" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:#22c55e;margin-bottom:4px">&#10003; Wykryta twarz</div>
          <img id="pose-face-thumb" style="width:110px;height:110px;object-fit:cover;border-radius:8px;border:2px solid #22c55e">
        </div>
        <div id="pose-face-loading" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:4px">Wykrywam...</div>
          <div style="width:110px;height:110px;border:2px dashed var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.5rem">&#128269;</div>
        </div>
        <div id="pose-face-notfound" style="flex:0 0 110px;text-align:center;display:none">
          <div style="font-size:.72rem;color:#ef4444;margin-bottom:4px">&#10007; Brak twarzy</div>
          <div style="width:110px;height:110px;border:2px dashed #ef4444;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.72rem;color:#ef4444;text-align:center;padding:6px">Wgraj bliższy kadr</div>
        </div>
      </div>
      <button onclick="clearPoseBody()" style="position:absolute;top:0;right:0;background:#333;border:none;color:#fff;border-radius:50%;width:22px;height:22px;cursor:pointer;font-size:.75rem">&#x2715;</button>
    </div>
    <input type="file" id="pose-body-file" accept="image/*" style="display:none" onchange="loadPoseBodyFile(this.files[0])">
  </div>

  <div style="margin-bottom:10px">
    <label style="font-size:.82rem;color:var(--muted);display:block;margin-bottom:4px">Model</label>
    <select id="pose-model-sel" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:8px;font-size:.85rem">
      __PORTRAIT_MODEL_OPTIONS__
    </select>
  </div>

  <div style="margin-bottom:10px">
    <label style="font-size:.82rem;color:var(--muted);display:block;margin-bottom:4px">Opis sceny / co ma być na zdjęciu</label>
    <textarea id="pose-prompt" rows="3" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:8px;font-size:.85rem;resize:vertical"
      placeholder="np. biznesowy portret, jasny garnitur, eleganckie biuro..."></textarea>
  </div>

  <div style="margin-bottom:10px">
    <label style="font-size:.82rem;color:var(--muted);display:block;margin-bottom:4px">Negative prompt</label>
    <textarea id="pose-negative" rows="2" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:8px;font-size:.85rem;resize:vertical">ugly, deformed, blurry, low quality, watermark</textarea>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
    <div>
      <label style="font-size:.78rem;color:var(--muted);display:block;margin-bottom:4px">Siła zachowania sylwetki: <span id="pose-body-str-val">0.65</span></label>
      <input type="range" id="pose-body-strength" min="0.1" max="1.0" step="0.05" value="0.65"
        oninput="document.getElementById('pose-body-str-val').textContent=this.value"
        style="width:100%">
    </div>
    <div>
      <label style="font-size:.78rem;color:var(--muted);display:block;margin-bottom:4px">Siła zachowania twarzy: <span id="pose-face-str-val">0.80</span></label>
      <input type="range" id="pose-face-strength" min="0.1" max="1.0" step="0.05" value="0.80"
        oninput="document.getElementById('pose-face-str-val').textContent=this.value"
        style="width:100%">
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;margin-bottom:10px">
    <div><label style="font-size:.78rem;color:var(--muted)">Szerokość</label><input type="number" id="pose-w" value="832" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:6px;font-size:.85rem"></div>
    <button onclick="swapPoseWH()" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.1rem;padding:4px">&#8644;</button>
    <div><label style="font-size:.78rem;color:var(--muted)">Wysokość</label><input type="number" id="pose-h" value="1216" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:6px;font-size:.85rem"></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px">
    <div><label style="font-size:.78rem;color:var(--muted)">Steps</label><input type="number" id="pose-steps" value="30" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:6px;font-size:.85rem"></div>
    <div><label style="font-size:.78rem;color:var(--muted)">CFG</label><input type="number" id="pose-cfg" value="5.0" step="0.5" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:6px;font-size:.85rem"></div>
    <div><label style="font-size:.78rem;color:var(--muted)">Seed</label><input type="number" id="pose-seed" value="-1" style="width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:6px;font-size:.85rem"></div>
  </div>

  <button onclick="startPose()" id="pose-gen-btn"
    style="width:100%;padding:14px;background:linear-gradient(135deg,#0f766e,#0891b2);border:none;border-radius:10px;color:#fff;font-size:1rem;font-weight:700;cursor:pointer;letter-spacing:.05em">
    &#128694; GENERUJ Z POZĄ
  </button>
  <div id="pose-status" style="margin-top:8px;font-size:.85rem;color:var(--muted);text-align:center"></div>
  <div id="pose-progress-wrap" style="display:none;margin-top:10px">
    <div class="progress-label"><span>Generowanie z pozą...</span><span id="pose-prog-pct"></span></div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="pose-prog-fill"></div></div>
    <div id="pose-prog-detail" style="font-size:.75rem;color:var(--muted);margin-top:4px;min-height:1em"></div>
  </div>
</div>

<div class="right-panel" id="pose-results" style="display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:flex-start;padding:16px">
  <div style="color:var(--muted);font-size:.9rem;margin-top:40px">Wygenerowane obrazy pojawią się tutaj</div>
</div>
</main>
</div><!-- /view-pose -->

<!-- ── Video View ── -->

<!-- ── FLUX1 Image ── -->

<!-- ── Lightbox ── -->
<div id="lb" onclick="if(event.target===this)closeLb()">
  <button id="lb-close" onclick="closeLb()">&#10005;</button>
  <div id="lb-main">
    <img id="lb-img" src="" alt="">
    <div id="lb-nav">
      <button onclick="navLb(-1)">&#8592; Poprzednie</button>
      <button id="lb-dl" onclick="dlLb()">&#8595; Pobierz</button>
      <button onclick="editFromLb()" style="background:#059669;border-color:#059669;color:#fff">&#9999;&#65039; Edytuj</button>
      <button onclick="navLb(1)">Następne &#8594;</button>
    </div>
  </div>
  <div id="lb-meta"><span id="lb-meta-empty">Ładowanie metadanych…</span></div>
</div>

<div id="toast"></div>

<script>
var PRESETS = __PRESETS_JSON__;
var GALLERY_URL = '__GALLERY_URL__';
var _refImageB64 = null;
var _curPreset = null;
var _lbImgs = [], _lbIdx = 0;
var _pollTimer = null;
var _histOpen = false;
var _backend = 'forge';
function switchBackend(b) {
  _backend = b;
  document.querySelectorAll('.backend-btn').forEach(function(el) { el.classList.remove('active'); });
  var btn = document.getElementById('backend-' + b);
  if (btn) btn.classList.add('active');
  // Sampler + scheduler hidden in Krea 2 mode
  document.querySelectorAll('.krea2-hide').forEach(function(el) {
    el.style.display = (b === 'krea2') ? 'none' : '';
  });
  if (b === 'ravnet') {
    fetch('/api/ravnet-models').then(function(r){return r.json();}).then(_applyOrderToSelect);
  } else if (b === 'forge') {
    fetch('/api/models').then(function(r){return r.json();}).then(_applyOrderToSelect);
  }
}

/* ── Presets ── */

function applyAutoResult(s, fromCache) {
  if (s.steps)    document.getElementById('steps-in').value = s.steps;
  if (s.cfg !== undefined) document.getElementById('cfg-in').value = s.cfg;
  if (s.width)    document.getElementById('w-in').value   = s.width;
  if (s.height)   document.getElementById('h-in').value   = s.height;
  if (s.sampler)  setVal('sampler-sel', s.sampler);
  if (s.scheduler) setVal('sched-sel',  s.scheduler);
  markCustom();
  updateParamsBar();
  var src = fromCache ? ' (cache)' : ' (AI)';
  var msg = 'Auto' + src + ': ' + (s.sampler||'') + ' · CFG ' + s.cfg + ' · ' + s.steps + ' steps · ' + (s.width||'?') + '\u00d7' + (s.height||'?');
  if (s.notes) msg += ' — ' + s.notes;
  toast(msg, 'ok');
}


/* ── Zarządzaj modelami ── */
var _dlJobId = null, _dlTimer = null;

function openMgrModal() {
  document.getElementById('model-mgr-modal').classList.add('open');
  refreshMgrList();
}
function closeMgrModal() {
  document.getElementById('model-mgr-modal').classList.remove('open');
  if (_dlTimer) { clearInterval(_dlTimer); _dlTimer = null; }
}

/* ── Model order helpers ── */
function _getModelOrder() {
  try { return JSON.parse(localStorage.getItem('model-order') || '[]'); } catch(e) { return []; }
}
function _saveModelOrder(names) {
  localStorage.setItem('model-order', JSON.stringify(names));
}
function _sortByOrder(models) {
  var order = _getModelOrder();
  if (!order.length) return models;
  var map = {};
  models.forEach(function(m){ map[m.model_name || m.title || ''] = m; });
  var sorted = [];
  order.forEach(function(n){ if (map[n]) { sorted.push(map[n]); delete map[n]; } });
  Object.values(map).forEach(function(m){ sorted.push(m); });
  return sorted;
}
function _applyOrderToSelect(models) {
  var sel = document.getElementById('model-sel');
  if (!sel) return;
  var cur = sel.value;
  var sorted = _sortByOrder(models);
  sel.innerHTML = sorted.map(function(m){
    var v = m.model_name || m.title || '';
    return '<option value="' + v + '"' + (v===cur?' selected':'') + '>' + v + '</option>';
  }).join('');
}

/* drag-and-drop state */
var _dndSrc = null;

function _initDnd(list) {
  list.querySelectorAll('.mgr-item').forEach(function(item) {
    item.addEventListener('dragstart', function(e) {
      _dndSrc = item;
      e.dataTransfer.effectAllowed = 'move';
      setTimeout(function(){ item.classList.add('dnd-dragging'); }, 0);
    });
    item.addEventListener('dragend', function() {
      item.classList.remove('dnd-dragging');
      list.querySelectorAll('.mgr-item').forEach(function(i){ i.classList.remove('dnd-over'); });
      /* zapisz nową kolejność */
      var names = [];
      list.querySelectorAll('.mgr-item').forEach(function(i){
        names.push(i.dataset.fname);
      });
      _saveModelOrder(names);
      /* zastosuj do selektu */
      fetch('/api/models').then(function(r){return r.json();}).then(_applyOrderToSelect);
    });
    item.addEventListener('dragover', function(e) {
      e.preventDefault(); e.dataTransfer.dropEffect = 'move';
      list.querySelectorAll('.mgr-item').forEach(function(i){ i.classList.remove('dnd-over'); });
      item.classList.add('dnd-over');
    });
    item.addEventListener('drop', function(e) {
      e.preventDefault();
      if (_dndSrc && _dndSrc !== item) {
        /* wstaw src przed target lub po (zależnie od pozycji kursora) */
        var rect = item.getBoundingClientRect();
        var after = e.clientY > rect.top + rect.height / 2;
        if (after) list.insertBefore(_dndSrc, item.nextSibling);
        else       list.insertBefore(_dndSrc, item);
      }
    });
  });
}

function refreshMgrList() {
  fetch('/api/models/refresh').then(function(){ return fetch('/api/models'); })
    .then(function(r){ return r.json(); })
    .then(function(models) {
      var list = document.getElementById('mgr-list');
      if (!models.length) { list.innerHTML = '<div style="color:#475569;font-size:.82rem">Brak modeli</div>'; return; }
      list.innerHTML = '';
      var sorted = _sortByOrder(models);
      sorted.forEach(function(m) {
        var name = (m.model_name || m.title || '').replace(/\.[^.]+$/, '');
        var fname = (m.filename || m.model_name || '') + '';
        if (!fname.match(/\.(safetensors|ckpt|pt)$/i)) fname += '.safetensors';
        var div = document.createElement('div');
        div.className = 'mgr-item';
        div.draggable = true;
        div.dataset.fname = fname;
        div.innerHTML = '<span class="mgr-drag-handle" title="Przeciągnij aby zmienić kolejność">&#9776;</span>'
          + '<div class="mgr-item-name" title="' + fname + '">' + name + '</div>'
          + '<button class="mgr-del-btn" onclick="deleteModel(' + JSON.stringify(fname) + ',this)">&#128465; Usuń</button>';
        list.appendChild(div);
      });
      _initDnd(list);
      _applyOrderToSelect(models);
    });
}

function deleteModel(fname, btn) {
  if (!confirm('Usunąć model ' + fname + '?')) return;
  btn.disabled = true; btn.textContent = '...';
  fetch('/api/model-delete/' + encodeURIComponent(fname))
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok) { toast('Usunięto: ' + fname, 'ok'); refreshMgrList(); }
      else { toast('Błąd: ' + d.error, 'err'); btn.disabled=false; btn.textContent='🗑 Usu\u0144';; }
    });
}

function startDownload() {
  var url = document.getElementById('mgr-url-input').value.trim();
  if (!url) { toast('Wklej URL pobierania', 'err'); return; }
  var prog = document.getElementById('dl-progress');
  prog.classList.add('show');
  document.getElementById('dl-info').textContent = 'Łączenie...';
  document.getElementById('dl-bar-fill').style.width = '0%';
  document.getElementById('dl-bytes').textContent = '';
  fetch('/api/download-model', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({url: url})})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!d.ok) { toast('Błąd: ' + d.error, 'err'); prog.classList.remove('show'); return; }
      _dlJobId = d.job_id;
      _dlTimer = setInterval(pollDownload, 800);
    })
    .catch(function(e){ toast('Błąd: ' + e, 'err'); prog.classList.remove('show'); });
}

function pollDownload() {
  if (!_dlJobId) return;
  fetch('/api/download-progress/' + _dlJobId)
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!d.ok) return;
      var pct  = d.percent >= 0 ? d.percent : 0;
      var done = d.bytes_done || 0;
      var size = d.size || 0;
      document.getElementById('dl-bar-fill').style.width = (d.percent >= 0 ? pct : 50) + '%';
      document.getElementById('dl-info').textContent = (d.filename || 'Pobieranie...') + (d.percent >= 0 ? '  ' + pct + '%' : '');
      document.getElementById('dl-bytes').textContent = size > 0
        ? (done/1048576).toFixed(1) + ' MB / ' + (size/1048576).toFixed(1) + ' MB'
        : (done/1048576).toFixed(1) + ' MB pobranych';
      if (d.status === 'done') {
        clearInterval(_dlTimer); _dlTimer = null;
        document.getElementById('dl-bar-fill').style.width = '100%';
        document.getElementById('dl-info').textContent = '\u2705 Pobrano: ' + d.filename;
        toast('Pobrano model: ' + d.filename, 'ok');
        document.getElementById('mgr-url-input').value = '';
        refreshMgrList();
      } else if (d.status === 'error') {
        clearInterval(_dlTimer); _dlTimer = null;
        document.getElementById('dl-info').textContent = '\u274C Błąd: ' + d.error;
        toast('Błąd pobierania: ' + d.error, 'err');
      }
    });
}

function autoModel() {
  var sel = document.getElementById('model-sel');
  if (!sel) return;
  var models = Array.from(sel.options).map(function(o){ return o.value; }).filter(Boolean);
  if (!models.length) { toast('Brak modeli na liście', 'err'); return; }

  var promptTxt = (document.getElementById('positive-ta') || {}).value || '';
  var descTxt   = (document.getElementById('desc-ta') || {}).value || '';
  var combined  = (descTxt + ' ' + promptTxt).trim();
  if (!combined) { toast('Wpisz najpierw opis lub prompt', 'err'); return; }

  var cacheKey = 'auto_mdl_' + combined.slice(0, 80) + '|' + models.slice(0, 5).join(',');
  try {
    var cached = localStorage.getItem(cacheKey);
    if (cached) {
      var obj = JSON.parse(cached);
      if (obj && obj.ts && (Date.now() - obj.ts) < 3600000) {
        setVal('model-sel', obj.model);
        onModelChange(obj.model);
        toast('Model (cache): ' + obj.model + ' — ' + (obj.reason || ''), 'ok');
        return;
      }
    }
  } catch(e) {}

  toast('\u23f3 AI dobiera model...', 'info');
  var url = '/api/auto-model?prompt=' + encodeURIComponent(combined.slice(0, 400))
          + '&models=' + encodeURIComponent(models.join(','));
  fetch(url)
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!d.ok) { toast('Błąd AI: ' + (d.error || '?'), 'err'); return; }
      if (d.model) {
        setVal('model-sel', d.model);
        onModelChange(d.model);
        try { localStorage.setItem(cacheKey, JSON.stringify({model: d.model, reason: d.reason, ts: Date.now()})); } catch(e){}
        toast('Model AI: ' + d.model + (d.reason ? ' — ' + d.reason : ''), 'ok');
      } else {
        toast('AI nie wybrało modelu', 'err');
      }
    })
    .catch(function(e){ toast('Błąd: ' + e, 'err'); });
}
function autoSettings() {
  var sel = document.getElementById('model-sel');
  if (!sel) return;
  var modelName = sel.value;
  if (!modelName) { toast('Wybierz model', 'err'); return; }

  var cacheKey = 'auto_s_' + modelName;
  try {
    var cached = localStorage.getItem(cacheKey);
    if (cached) {
      var obj = JSON.parse(cached);
      if (obj && obj.ts && (Date.now() - obj.ts) < 7 * 86400000) {
        applyAutoResult(obj.s, true);
        return;
      }
    }
  } catch(e) {}

  toast('\u23f3 Analizuję model przez AI...', 'info');
  fetch('/api/auto-settings?model=' + encodeURIComponent(modelName))
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!d.ok) { toast('Błąd AI: ' + (d.error||'?'), 'err'); return; }
      try { localStorage.setItem(cacheKey, JSON.stringify({s: d.settings, ts: Date.now()})); } catch(e){}
      applyAutoResult(d.settings, d.cached || false);
    })
    .catch(function(e){ toast('Błąd sieci: ' + e, 'err'); });
}
function applyPreset(id) {
  var p = PRESETS.find(function(x){return x.id===id;});
  if(!p) return;
  _curPreset = id;
  document.querySelectorAll('.preset-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.id===id);
  });
  var sel = document.getElementById('model-sel');
  for(var i=0;i<sel.options.length;i++){
    if(sel.options[i].value===p.model) { sel.selectedIndex=i; break; }
  }
  setVal('sampler-sel', p.sampler);
  setVal('sched-sel',   p.scheduler);
  document.getElementById('steps-in').value = p.steps;
  document.getElementById('cfg-in').value   = p.cfg;
  document.getElementById('w-in').value     = p.width;
  document.getElementById('h-in').value     = p.height;
  document.getElementById('batch-in').value = p.batch;
  document.getElementById('negative-ta').value = p.negative;
  updateParamsBar();
}

function setVal(id, val) {
  var el = document.getElementById(id);
  if(!el) return;
  for(var i=0;i<el.options.length;i++){
    if(el.options[i].value===val||el.options[i].text===val){ el.selectedIndex=i; return; }
  }
}

function markCustom() {
  _curPreset = null;
  document.querySelectorAll('.preset-btn').forEach(function(b){ b.classList.remove('active'); });
  updateParamsBar();
}

function updateParamsBar() {
  var model   = document.getElementById('model-sel');
  var modelTxt = model ? model.options[model.selectedIndex].text : '—';
  document.getElementById('pb-model').textContent   = modelTxt;
  document.getElementById('pb-sampler').textContent = document.getElementById('sampler-sel').value;
  document.getElementById('pb-steps').textContent   = document.getElementById('steps-in').value;
  document.getElementById('pb-cfg').textContent     = document.getElementById('cfg-in').value;
  document.getElementById('pb-size').textContent    = document.getElementById('w-in').value+'×'+document.getElementById('h-in').value;
  document.getElementById('pb-batch').textContent   = document.getElementById('batch-in').value+' szt.';
}

/* ── Advanced toggle ── */
function toggleAdv() {
  var sec = document.getElementById('adv-section');
  var tog = document.getElementById('adv-toggle');
  var open = sec.classList.toggle('open');
  tog.classList.toggle('open', open);
}

/* ── AI Prompt ── */
function genAiPrompt() {
  var desc = document.getElementById('desc-ta').value.trim();
  if(!desc) return toast('Najpierw opisz co chcesz wygenerować', 'err');
  var btn = document.getElementById('ai-btn');
  btn.disabled = true; btn.innerHTML = '&#8987; AI<br>Prompt';
  fetch('/api/ai-prompt', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({description: desc})
  }).then(function(r){return r.json();}).then(function(d){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    if(d.ok) {
      document.getElementById('positive-ta').value = d.positive;
      document.getElementById('negative-ta').value = d.negative;
      toast('Prompt wygenerowany!', 'ok');
    } else toast('Błąd AI: '+d.error, 'err');
  }).catch(function(){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    toast('Błąd połączenia z Ollama', 'err');
  });
}

/* ── Reference image ── */
function loadRefFile(file) {
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    var b64 = e.target.result.split(',')[1];
    _refImageB64 = b64;
    document.getElementById('ref-thumb').src = e.target.result;
    document.getElementById('ref-fname').textContent = file.name;
    document.getElementById('ref-placeholder').style.display = 'none';
    document.getElementById('ref-loaded').style.display = 'block';
    document.getElementById('ref-controls').style.display = 'block';
  };
  reader.readAsDataURL(file);
}
function clearRef(e) {
  e.stopPropagation();
  _refImageB64 = null;
  document.getElementById('ref-thumb').src = '';
  document.getElementById('ref-placeholder').style.display = '';
  document.getElementById('ref-loaded').style.display = 'none';
  document.getElementById('ref-controls').style.display = 'none';
  document.getElementById('ref-file-in').value = '';
}
function describeRef() {
  if (!_refImageB64) return;
  var btn = document.getElementById('ref-desc-btn');
  var style = document.getElementById('ref-style-sel').value;
  btn.textContent = '⏳ Analizuję zdjęcie...';
  btn.disabled = true;
  fetch('/api/vision-describe', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image_b64: _refImageB64, style: style})
  }).then(function(r){return r.json();}).then(function(d){
    if (d.positive) {
      document.getElementById('positive-ta').value = d.positive;
      if (d.negative) document.getElementById('negative-ta').value = d.negative;
      btn.textContent = '✅ Prompt gotowy — kliknij GENERUJ';
      btn.style.background = '#064e3b';
      btn.style.color = '#6ee7b7';
      setTimeout(function(){
        btn.textContent = '🔍 Opisz zdjęcie i wygeneruj prompt';
        btn.style.background = '';
        btn.style.color = '';
        btn.disabled = false;
      }, 3000);
    } else {
      toast('Błąd analizy: ' + (d.error || 'brak odpowiedzi'), 'err');
      btn.textContent = '🔍 Opisz zdjęcie i wygeneruj prompt';
      btn.disabled = false;
    }
  }).catch(function(err){
    toast('Błąd połączenia z API vision', 'err');
    btn.textContent = '🔍 Opisz zdjęcie i wygeneruj prompt';
    btn.disabled = false;
    console.error(err);
  });
}

var MODEL_TRIGGERS = {
  'pony': {
    match: ['pony', 'cyberrealisticpony'],
    positive_prefix: 'score_9, score_8_up, score_7_up, score_6_up, ',
    negative_default: 'score_4, score_3, score_2, score_1, blurry, ugly',
    banner_text: '🏷️ Pony XL: trigger words dodawane auto na początku promptu',
    banner_style: 'background:#1e1a2e;border:1px solid #6d28d9;border-radius:8px;padding:8px 12px;font-size:.8rem;color:#c4b5fd;margin-top:8px'
  }
};

function getModelTriggers(val) {
  var v = val.toLowerCase();
  for (var key in MODEL_TRIGGERS) {
    var t = MODEL_TRIGGERS[key];
    for (var i = 0; i < t.match.length; i++) {
      if (v.indexOf(t.match[i]) !== -1) return t;
    }
  }
  return null;
}

function onModelChange(val) {
  var isFlux = val.toLowerCase().indexOf('flux') !== -1 ||
               val.toLowerCase().indexOf('unstableevolution') !== -1 ||
               val.toLowerCase().indexOf('krea') !== -1;
  var fluxBanner = document.getElementById('flux-banner');
  if (isFlux) {
    if (!fluxBanner) {
      fluxBanner = document.createElement('div');
      fluxBanner.id = 'flux-banner';
      fluxBanner.style.cssText = 'background:#1a2e1a;border:1px solid #166534;border-radius:8px;padding:8px 12px;font-size:.8rem;color:#86efac;margin-top:8px';
      fluxBanner.textContent = '⚡ FLUX: CFG=1.0 · Negative prompt ignorowany · Sampler: Euler';
      document.getElementById('model-sel').closest('.field').appendChild(fluxBanner);
    }
    document.getElementById('cfg-in').value = '1';
    document.getElementById('negative-ta').value = '';
    setVal('sampler-sel', 'Euler');
    setVal('sched-sel', 'Simple');
  } else {
    if (fluxBanner) fluxBanner.remove();
  }

  var triggers = getModelTriggers(val);
  var trigBanner = document.getElementById('trigger-banner');
  if (triggers) {
    if (!trigBanner) {
      trigBanner = document.createElement('div');
      trigBanner.id = 'trigger-banner';
      document.getElementById('model-sel').closest('.field').appendChild(trigBanner);
    }
    trigBanner.style.cssText = triggers.banner_style;
    trigBanner.textContent = triggers.banner_text;
    var negEl = document.getElementById('negative-ta');
    if (!negEl.value.trim()) negEl.value = triggers.negative_default;
  } else {
    if (trigBanner) trigBanner.remove();
  }
}
function swapWH() {
  var w = document.getElementById('w-in');
  var h = document.getElementById('h-in');
  var tmp = w.value; w.value = h.value; h.value = tmp;
  markCustom();
}

/* ── Generate ── */
function startGenerate() {
  var pos = document.getElementById('positive-ta').value.trim();
  if(!pos) return toast('Wpisz positive prompt lub użyj AI Prompt', 'err');
  var p = _curPreset ? PRESETS.find(function(x){return x.id===_curPreset;}) : null;
  var modelVal = document.getElementById('model-sel').value;
  var triggers = getModelTriggers(modelVal);
  var modelPrefix = triggers ? triggers.positive_prefix : '';
  var presetPrefix = p ? p.prefix : '';
  var fullPos = modelPrefix + presetPrefix + pos;
  var negVal = document.getElementById('negative-ta').value.trim();
  if (!negVal && triggers) negVal = triggers.negative_default;
  var params = {
    description: document.getElementById('desc-ta').value.trim(),
    positive:    fullPos,
    negative:    negVal,
    model:       modelVal,
    sampler:     document.getElementById('sampler-sel').value,
    scheduler:   document.getElementById('sched-sel').value,
    steps:       parseInt(document.getElementById('steps-in').value),
    cfg:         parseFloat(document.getElementById('cfg-in').value),
    width:       parseInt(document.getElementById('w-in').value),
    height:      parseInt(document.getElementById('h-in').value),
    seed:        parseInt(document.getElementById('seed-in').value),
    batch:       parseInt(document.getElementById('batch-in').value),
    preset:      _curPreset || 'custom',
    backend:     _backend,
  };
  if (_refImageB64) {
    params.init_image = _refImageB64;
    params.denoising  = parseFloat(document.getElementById('denoising-range').value);
    params.batch      = 1;
  }
  document.getElementById('gen-btn').disabled = true;
  document.getElementById('gen-btn').textContent = 'Generowanie...';
  showProgress(true);
  fetch('/api/generate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(params)
  }).then(function(r){return r.json();}).then(function(d){
    if(d.job_id) pollJob(d.job_id, 0);
    else { resetBtn(); toast('Błąd: '+d.error, 'err'); showProgress(false); }
  }).catch(function(){resetBtn(); toast('Błąd połączenia', 'err'); showProgress(false);});
}

function pollJob(jid, tick) {
  fetch('/api/job/'+jid).then(function(r){return r.json();}).then(function(d){
    if(d.status === 'done') {
      stopForgeProgress();
      document.getElementById('prog-fill').style.width='100%';
      document.getElementById('prog-pct').textContent='';
      setTimeout(function(){showProgress(false); showImages(d.images, d.seed); resetBtn(); loadHistory();}, 400);
    } else if(d.status === 'error') {
      stopForgeProgress();
      showProgress(false); resetBtn(); toast('Błąd: '+d.error, 'err');
    } else {
      _pollTimer = setTimeout(function(){pollJob(jid, tick+1);}, 2000);
    }
  }).catch(function(){
    _pollTimer = setTimeout(function(){pollJob(jid, tick+1);}, 3000);
  });
}

function showProgress(on) {
  document.getElementById('progress-wrap').style.display = on ? 'block' : 'none';
  if(on) {
    document.getElementById('prog-fill').style.width='5%';
    document.getElementById('prog-label').textContent='Generowanie...';
    document.getElementById('prog-pct').textContent='';
    startForgeProgress('prog-fill', 'prog-pct');
  } else {
    stopForgeProgress();
    document.getElementById('prog-pct').textContent='';
  }
}

function resetBtn() {
  var b = document.getElementById('gen-btn');
  b.disabled=false; b.innerHTML='&#127912; GENERUJ';
}

function showImages(paths, seed) {
  document.getElementById('results-placeholder').style.display='none';
  var grid = document.getElementById('img-grid');
  grid.innerHTML='';
  _lbImgs = paths;
  paths.forEach(function(p, i){
    var img=document.createElement('img');
    img.className='res-img'; img.src='/img/'+p; img.loading='lazy';
    img.onclick=function(){openLb(i);};
    grid.appendChild(img);
  });
  document.getElementById('seed-info').textContent = seed>0 ? 'Seed: '+seed : '';
  toast('Wygenerowano '+paths.length+' zdjęć!', 'ok');
}

/* ── Lightbox ── */
function openLb(idx) { _lbIdx=idx; updLb(); document.getElementById('lb').classList.add('open'); document.body.style.overflow='hidden'; }
function closeLb()   { document.getElementById('lb').classList.remove('open'); document.body.style.overflow=''; }
function navLb(d)    { _lbIdx=(_lbIdx+d+_lbImgs.length)%_lbImgs.length; updLb(); }
function dlLb()      { var a=document.createElement('a'); a.href='/img/'+_lbImgs[_lbIdx]; a.download=_lbImgs[_lbIdx].split('/').pop(); a.click(); }
function updLb() {
  var path = _lbImgs[_lbIdx];
  document.getElementById('lb-img').src = '/img/' + path;
  var meta = document.getElementById('lb-meta');
  meta.innerHTML = '<span id="lb-meta-empty">Ładowanie…</span>';
  fetch('/api/image-meta?path=' + encodeURIComponent(path))
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (!d.ok || !d.raw) { meta.innerHTML='<span id="lb-meta-empty">Brak metadanych</span>'; return; }
      var html = '';
      if (d.positive) {
        html += '<div><h4>Prompt <button class="lb-meta-copy" onclick="navigator.clipboard.writeText('+JSON.stringify(d.positive)+')">kopiuj</button></h4>';
        html += '<div class="lb-meta-val prompt">'+escHtml(d.positive)+'</div></div>';
      }
      if (d.negative) {
        html += '<div><h4>Negative prompt</h4>';
        html += '<div class="lb-meta-val prompt">'+escHtml(d.negative)+'</div></div>';
      }
      var modelVal = d.model || '';
      if (modelVal) {
        html += '<div><h4>Model</h4><div><span class="lb-meta-badge">'+escHtml(modelVal)+'</span></div></div>';
      }
      var keys = ['steps','sampler','scheduler','cfg_scale','seed','size'];
      var labels = {steps:'Steps',sampler:'Sampler',scheduler:'Scheduler',cfg_scale:'CFG Scale',seed:'Seed',size:'Size'};
      var gridItems = keys.filter(function(k){ return d[k]; });
      if (gridItems.length) {
        html += '<div><h4>Parametry</h4><div class="lb-meta-grid">';
        gridItems.forEach(function(k){
          html += '<div class="lb-meta-kv"><span>'+labels[k]+'</span><span>'+escHtml(d[k])+'</span></div>';
        });
        html += '</div></div>';
      }
      meta.innerHTML = html || '<span id="lb-meta-empty">Brak metadanych</span>';
    })
    .catch(function(){ meta.innerHTML='<span id="lb-meta-empty">Błąd ładowania</span>'; });
}
function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/* ── History ── */
function toggleHist() {
  _histOpen = !_histOpen;
  var body = document.getElementById('hist-body');
  var arrow = document.getElementById('hist-arrow');
  body.style.display = _histOpen ? 'flex' : 'none';
  arrow.style.transform = _histOpen ? 'rotate(180deg)' : '';
  if(_histOpen) loadHistory();
}

function loadHistory() {
  fetch('/api/history').then(function(r){return r.json();}).then(function(d){
    var body = document.getElementById('hist-body');
    document.getElementById('hist-count').textContent = d.length ? '('+d.length+')' : '';
    if(!_histOpen) return;
    body.innerHTML = '';
    if(!d.length){
      var em = document.createElement('div'); em.className='no-hist'; em.textContent='Brak historii generowań'; body.appendChild(em); return;
    }
    d.forEach(function(g){
      var paths = JSON.parse(g.paths||'[]');
      var item  = document.createElement('div'); item.className='hist-item';

      /* thumbnails */
      var thumbsDiv = document.createElement('div'); thumbsDiv.className='hist-thumbs';
      paths.slice(0,4).forEach(function(p, i){
        var img = document.createElement('img');
        img.className='hist-thumb'; img.loading='lazy';
        img.src='/img/'+encodeURI(p);
        img.onclick=(function(ps,idx){return function(){openHistLb(ps,idx);};})(paths,i);
        thumbsDiv.appendChild(img);
      });

      /* meta */
      var meta = document.createElement('div'); meta.className='hist-meta';

      var desc = document.createElement('div'); desc.className='hist-desc';
      desc.textContent = g.description || (g.positive||'').substring(0,60)+'...';

      var dt   = new Date(g.ts*1000).toLocaleString('pl');
      var tags = document.createElement('div'); tags.className='hist-tags';
      tags.textContent = [g.model, g.sampler, g.width+'x'+g.height, g.batch+' img', dt].join(' · ');

      var actions = document.createElement('div'); actions.className='hist-actions';
      var infoBtn = document.createElement('button'); infoBtn.className='hist-btn info';
      infoBtn.textContent='ℹ Metadane';
      infoBtn.onclick=(function(gen){return function(){openMetaModal(gen);};})(g);
      var btn = document.createElement('button'); btn.className='hist-btn';
      btn.textContent='↺ Powtórz';
      btn.onclick=(function(gen){return function(){repeatGen(gen);};})(g);
      var editBtn = document.createElement('button'); editBtn.className='hist-btn';
      editBtn.textContent='✏ Edytuj';
      editBtn.onclick=(function(ps){return function(){ if(ps.length) openInEditor('/img/'+encodeURI(ps[0])); };})(paths);
      var delBtn = document.createElement('button'); delBtn.className='hist-btn del';
      delBtn.textContent='🗑 Usuń';
      delBtn.onclick=(function(id,el){return function(){deleteHist(id,el);};})(g.id,item);
      var pubBtn = document.createElement('button'); pubBtn.className='hist-btn pub';
      pubBtn.textContent='⚡ Opublikuj';
      pubBtn.onclick=(function(gen){return function(){publishGen(gen);};})(g);
      actions.appendChild(infoBtn); actions.appendChild(btn); actions.appendChild(editBtn); actions.appendChild(delBtn); actions.appendChild(pubBtn);

      meta.appendChild(desc); meta.appendChild(tags); meta.appendChild(actions);
      item.appendChild(thumbsDiv); item.appendChild(meta);
      body.appendChild(item);
    });
  });
}

var PORTAL_URL = '__PORTAL_URL__';

function publishGen(gen) {
  var paths = JSON.parse(gen.paths||'[]');
  if(!paths.length){toast('Brak zdjęcia','err');return;}
  document.getElementById('publish-gen-id').value = gen.id;
  document.getElementById('publish-path-idx').value = '0';
  document.getElementById('publish-modal').style.display='flex';
}

function doPublish(isXxx) {
  var genId = document.getElementById('publish-gen-id').value;
  var pathIdx = parseInt(document.getElementById('publish-path-idx').value)||0;
  document.getElementById('publish-modal').style.display='none';
  toast('Publikowanie…','ok');
  fetch('/api/publish_to_portal',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({gen_id:genId,path_idx:pathIdx,is_xxx:isXxx})
  }).then(function(r){return r.json();}).then(function(d){
    if(d.ok){
      var link='<a href="'+PORTAL_URL+d.url+'" target="_blank" style="color:#a5b4fc">Zobacz →</a>';
      toast('✓ Opublikowano! '+link,'ok');
    } else {
      toast('Błąd: '+d.error,'err');
    }
  }).catch(function(){toast('Błąd sieci','err');});
}

function openHistLb(paths, idx) { _lbImgs=paths; openLb(idx); }

function deleteHist(id, el) {
  el.style.opacity='0.4';
  fetch('/api/history/'+id, {method:'DELETE'})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ el.remove(); loadHistory(); }
      else { el.style.opacity='1'; toast('Błąd usuwania','err'); }
    }).catch(function(){ el.style.opacity='1'; toast('Błąd połączenia','err'); });
}

function deleteEditHist(id, el) {
  el.style.opacity='0.4';
  fetch('/api/edit-history/'+id, {method:'DELETE'})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ el.remove(); loadEditHistory(); }
      else { el.style.opacity='1'; toast('Błąd usuwania','err'); }
    }).catch(function(){ el.style.opacity='1'; toast('Błąd połączenia','err'); });
}

function openMetaModal(g) {
  document.getElementById('md-desc').textContent    = g.description || '—';
  document.getElementById('md-pos').textContent     = g.positive    || '—';
  document.getElementById('md-neg').textContent     = g.negative    || '—';
  document.getElementById('md-model').textContent   = g.model       || '—';
  document.getElementById('md-sampler').textContent = (g.sampler||'') + ' / ' + (g.scheduler||'');
  document.getElementById('md-steps').textContent   = g.steps       || '—';
  document.getElementById('md-cfg').textContent     = g.cfg         || '—';
  document.getElementById('md-size').textContent    = (g.width||'?') + ' × ' + (g.height||'?') + ' px';
  document.getElementById('md-seed').textContent    = g.seed        || '—';
  document.getElementById('md-batch').textContent   = g.batch       || '—';
  document.getElementById('md-date').textContent    = g.ts ? new Date(g.ts*1000).toLocaleString('pl') : '—';
  document.getElementById('md-load-btn').onclick    = function(){ closeMetaModal(); repeatGen(g); };
  document.getElementById('meta-modal').classList.add('open');
  document.body.style.overflow='hidden';
}
function metaImportFile(file) {
  if (!file) return;
  var status = document.getElementById('meta-import-status');
  if (status) status.textContent = 'Wczytuję...';
  var reader = new FileReader();
  reader.onload = function(e) {
    var b64 = e.target.result.split(',')[1];
    fetch('/api/image-meta-upload', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({data: b64})
    })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (!d.ok) { if(status) status.textContent = 'Brak metadanych SD'; return; }
      _fillFormFromMeta(d);
      if(status) status.textContent = '✓ Wczytano';
      setTimeout(function(){ if(status) status.textContent=''; }, 3000);
      toast('Metadane wczytane z ' + file.name, 'ok');
    })
    .catch(function(){ if(status) status.textContent = 'Błąd'; });
  };
  reader.readAsDataURL(file);
}

function _fillFormFromMeta(d) {
  if (d.positive) document.getElementById('positive-ta').value = d.positive;
  if (d.negative) document.getElementById('negative-ta').value = d.negative;
  if (d.model) {
    var sel = document.getElementById('model-sel');
    for(var i=0;i<sel.options.length;i++){
      if(sel.options[i].value===d.model||sel.options[i].text===d.model){sel.selectedIndex=i;break;}
    }
  }
  if (d.sampler)    setVal('sampler-sel', d.sampler);
  if (d.scheduler)  setVal('sched-sel', d.scheduler);
  if (d.steps)      document.getElementById('steps-in').value = d.steps;
  if (d.cfg_scale)  document.getElementById('cfg-in').value   = d.cfg_scale;
  if (d.width)      document.getElementById('w-in').value     = d.width;
  if (d.height)     document.getElementById('h-in').value     = d.height;
  if (d.seed)       document.getElementById('seed-in') && (document.getElementById('seed-in').value = d.seed);
  updateParamsBar && updateParamsBar();
}

/* URL params prefill: genphoto.ebartnet.pl/?positive=...&model=... */
(function(){
  try {
    var p = new URLSearchParams(window.location.search);
    if (p.get('positive') || p.get('model')) {
      var d = {};
      ['positive','negative','model','sampler','scheduler','steps','cfg_scale','width','height','seed'].forEach(function(k){
        if(p.get(k)) d[k] = p.get(k);
      });
      setTimeout(function(){ _fillFormFromMeta(d); toast('Parametry załadowane z galerii','ok'); }, 800);
      window.history.replaceState({}, '', '/');
    }
  } catch(_){}
})();

function closeMetaModal() {
  document.getElementById('meta-modal').classList.remove('open');
  document.body.style.overflow='';
}

function repeatGen(g) {
  document.getElementById('desc-ta').value = g.description || '';
  document.getElementById('positive-ta').value = g.positive;
  document.getElementById('negative-ta').value  = g.negative;
  var sel = document.getElementById('model-sel');
  for(var i=0;i<sel.options.length;i++){ if(sel.options[i].value===g.model){sel.selectedIndex=i;break;} }
  setVal('sampler-sel', g.sampler);
  setVal('sched-sel',   g.scheduler);
  document.getElementById('steps-in').value = g.steps;
  document.getElementById('cfg-in').value   = g.cfg;
  document.getElementById('w-in').value     = g.width;
  document.getElementById('h-in').value     = g.height;
  document.getElementById('batch-in').value = g.batch;
  _curPreset = g.preset;
  document.querySelectorAll('.preset-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.id===g.preset);
  });
  updateParamsBar();
  if(!document.getElementById('adv-section').classList.contains('open')) toggleAdv();
  window.scrollTo({top:0,behavior:'smooth'});
  toast('Parametry załadowane', 'ok');
}

/* ── Forge progress ── */
var _fpTimer = null;
function _translateForgeMsg(msg) {
  if (!msg) return '';
  var m = msg.toLowerCase();
  if (m.includes('insightface') || m.includes('instantid')) return 'Wykrywanie twarzy (InsightFace)...';
  if (m.includes('ipadapter') || m.includes('ip-adapter') || m.includes('ip_adapter') || m.includes('ipadapterpatcher')) return 'Ładowanie IP-Adapter (tożsamość twarzy)...';
  if (m.includes('controlnetpatcher') || m.includes('controlnet')) return 'Ładowanie ControlNet (keypoints twarzy)...';
  if (m.includes('autoencoder') || m.includes('vae')) return 'Dekodowanie obrazu (VAE)...';
  if (m.includes('textencode') || m.includes('text encoder') || m.includes('encoder')) return 'Ładowanie encodera tekstu...';
  if (m.includes('loading model') || m.includes('load model') || m.includes('kmodel') || m.includes('k-model')) return 'Ładowanie modelu SD do GPU...';
  if (m.includes('unload')) return 'Zwalnianie pamięci GPU...';
  if (m.includes('memory')) return 'Zarządzanie pamięcią GPU...';
  return '';
}

function startForgeProgress(fillId, pctId, labelId) {
  clearTimeout(_fpTimer);
  var _phaseIdx = 0;
  var _phases = [
    'Inicjalizacja...', 'Ładowanie modelu SD...',
    'Ładowanie IP-Adapter...', 'Wykrywanie twarzy...',
    'Ładowanie ControlNet...', 'Przygotowanie do samplingу...'
  ];
  (function tick(){
    fetch('/api/forge-progress').then(function(r){return r.json();}).then(function(d){
      var pct = Math.round(d.progress * 100);
      var f = document.getElementById(fillId);
      if (pct > 2 && f) f.style.width = pct + '%';
      var p = pctId ? document.getElementById(pctId) : null;
      if (p && pct > 2) {
        p.textContent = pct + '%' + (d.eta > 0 ? ' · ETA ' + d.eta + 's' : '');
      }
      if (labelId) {
        var lbl = document.getElementById(labelId);
        if (lbl) {
          var msg = _translateForgeMsg(d.textinfo);
          if (!msg && d.step > 0 && d.steps > 0) {
            msg = 'Próbkowanie: krok ' + d.step + ' / ' + d.steps;
          }
          if (!msg && pct < 2) {
            _phaseIdx = (_phaseIdx + 1) % _phases.length;
            msg = _phases[_phaseIdx];
          }
          if (msg) lbl.textContent = msg;
        }
      }
      _fpTimer = setTimeout(tick, 2000);
    }).catch(function(){ _fpTimer = setTimeout(tick, 5000); });
  })();
}
function stopForgeProgress() { clearTimeout(_fpTimer); _fpTimer = null; }

/* ── VRAM widget ── */
(function(){
  var _history = [];
  var _maxPts  = 60;
  var _timer   = null;

  function _drawChart(usedPct) {
    var canvas = document.getElementById('vram-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // tło
    ctx.fillStyle = 'rgba(15,23,42,0.6)';
    ctx.fillRect(0, 0, W, H);

    if (_history.length < 2) return;

    // kolor w zależności od użycia
    function barColor(pct) {
      if (pct >= 0.95) return '#ef4444';
      if (pct >= 0.80) return '#f59e0b';
      return '#22d3ee';
    }

    // linia wykresu
    var pts = _history;
    var step = W / (_maxPts - 1);
    ctx.beginPath();
    for (var i = 0; i < pts.length; i++) {
      var x = i * step;
      var y = H - pts[i] * (H - 2) - 1;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = barColor(usedPct);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // fill pod linią
    ctx.lineTo((pts.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = barColor(usedPct).replace(')', ', 0.15)').replace('rgb', 'rgba');
    ctx.fill();
  }

  function _tick() {
    fetch('/api/vram').then(function(r){ return r.json(); }).then(function(d) {
      if (!d.ok) return;
      var used  = d.used_gb,  free = d.free_gb, total = d.total_gb;
      var pct   = total > 0 ? used / total : 0;

      _history.push(pct);
      if (_history.length > _maxPts) _history.shift();

      _drawChart(pct);

      var nums = document.getElementById('vram-nums');
      if (nums) {
        nums.textContent = used.toFixed(1) + ' / ' + free.toFixed(1);
        nums.style.color = pct >= 0.95 ? '#ef4444' : pct >= 0.80 ? '#f59e0b' : '#93c5fd';
      }
    }).catch(function(){});
    _timer = setTimeout(_tick, 4000);
  }

  window.vramFree = function() {
    if (!confirm('Spowoduje to restart Forge (~30s przerwy). Kontynuować?')) return;
    var btn = document.getElementById('vram-free-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Restart...'; }
    var nums = document.getElementById('vram-nums');
    fetch('/api/vram-free', {method:'POST'}).then(function(r){ return r.json(); }).then(function(d) {
      if (!d.ok) {
        if (btn) { btn.disabled = false; btn.textContent = 'ZWOLNIJ'; }
        toast('Błąd: ' + (d.error || '?'), 'error');
        return;
      }
      toast('Forge restartuje — ~30s przerwy', 'info');
      // polling aż Forge wróci
      var tries = 0;
      (function waitForge(){
        setTimeout(function(){
          tries++;
          fetch('/api/vram').then(function(r){ return r.json(); }).then(function(v){
            if (v.ok) {
              if (btn) { btn.disabled = false; btn.textContent = 'ZWOLNIJ'; }
              toast('Forge gotowy · VRAM wolny: ' + v.free_gb + ' GB', 'success');
            } else if (tries < 20) { waitForge(); }
            else {
              if (btn) { btn.disabled = false; btn.textContent = 'ZWOLNIJ'; }
            }
          }).catch(function(){ if (tries < 20) waitForge(); else { if(btn){btn.disabled=false;btn.textContent='ZWOLNIJ';} } });
        }, 3000);
      })();
    }).catch(function() {
      if (btn) { btn.disabled = false; btn.textContent = 'ZWOLNIJ'; }
    });
  };

  // start po załadowaniu DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _tick);
  } else {
    _tick();
  }
})();

/* ── Toast ── */
function toast(msg,type){ var t=document.getElementById('toast'); t.textContent=msg; t.className='show '+(type||''); clearTimeout(t._t); t._t=setTimeout(function(){t.className='';},3000); }

/* ── Keyboard ── */
document.addEventListener('keydown',function(e){
  if(document.getElementById('lb').classList.contains('open')){
    if(e.key==='Escape') closeLb();
    if(e.key==='ArrowLeft') navLb(-1);
    if(e.key==='ArrowRight') navLb(1);
  }
});

/* ── Edit View ── */
var _editImgB64 = null;
var _editOrigDataUrl = null;
var _editSrcDataUrl = null;
var _editLastResult = null;
var _editMode = 'paint';
var _editDrawing = false;
var _editHistOpen = false;

function switchView(v) {
  ['generate','edit','portrait','pose'].forEach(function(n){
    document.getElementById('view-'+n).style.display = v===n ? '' : 'none';
  });
  document.getElementById('tab-gen').classList.toggle('active',       v==='generate');
  document.getElementById('tab-edit').classList.toggle('active',      v==='edit');
  document.getElementById('tab-portrait').classList.toggle('active',  v==='portrait');
  document.getElementById('tab-pose').classList.toggle('active',      v==='pose');
  if(v==='edit') loadEditHistory();
}

/* ── Portrait / InstantID ── */
var _portraitB64 = null;

function _detectFacePreview(b64, wrapId, loadId, foundId, notfoundId) {
  document.getElementById(wrapId).style.display = 'none';
  document.getElementById(notfoundId).style.display = 'none';
  document.getElementById(loadId).style.display = 'block';
  fetch('/api/detect-face', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image_b64: b64})
  }).then(function(r){return r.json();}).then(function(d){
    document.getElementById(loadId).style.display = 'none';
    if (d.ok && d.detected) {
      document.getElementById(foundId).src = 'data:image/jpeg;base64,'+d.face_b64;
      document.getElementById(wrapId).style.display = 'block';
    } else {
      document.getElementById(notfoundId).style.display = 'block';
    }
  }).catch(function(){
    document.getElementById(loadId).style.display = 'none';
    document.getElementById(notfoundId).style.display = 'block';
  });
}

function loadPortraitFile(file) {
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    _portraitB64 = e.target.result.split(',')[1];
    document.getElementById('portrait-thumb').src = e.target.result;
    document.getElementById('portrait-thumb-wrap').style.display = 'block';
    document.getElementById('portrait-drop').style.display = 'none';
    _detectFacePreview(_portraitB64,
      'portrait-face-wrap', 'portrait-face-loading',
      'portrait-face-thumb', 'portrait-face-notfound');
  };
  reader.readAsDataURL(file);
}

function clearPortrait() {
  _portraitB64 = null;
  document.getElementById('portrait-thumb-wrap').style.display = 'none';
  document.getElementById('portrait-face-wrap').style.display = 'none';
  document.getElementById('portrait-face-loading').style.display = 'none';
  document.getElementById('portrait-face-notfound').style.display = 'none';
  document.getElementById('portrait-drop').style.display = 'flex';
  document.getElementById('portrait-file-in').value = '';
}

function swapPortraitWH() {
  var w = document.getElementById('portrait-w');
  var h = document.getElementById('portrait-h');
  var tmp = w.value; w.value = h.value; h.value = tmp;
}

function startPortrait() {
  if (!_portraitB64) return toast('Wgraj najpierw swoje zdjęcie', 'err');
  var prompt = document.getElementById('portrait-prompt').value.trim();
  if (!prompt) return toast('Wpisz opis sceny', 'err');

  var btn = document.getElementById('portrait-gen-btn');
  var status = document.getElementById('portrait-status');
  btn.disabled = true;
  btn.textContent = '⏳ Generowanie...';
  status.textContent = 'Wykrywanie twarzy i generowanie...';

  var params = {
    face_image:    _portraitB64,
    positive:      prompt,
    negative:      document.getElementById('portrait-negative').value.trim(),
    model:         document.getElementById('portrait-model-sel').value,
    face_strength: parseFloat(document.getElementById('face-strength').value),
    pose_strength: parseFloat(document.getElementById('pose-strength').value),
    width:         parseInt(document.getElementById('portrait-w').value),
    height:        parseInt(document.getElementById('portrait-h').value),
    steps:         parseInt(document.getElementById('portrait-steps').value),
    cfg:           parseFloat(document.getElementById('portrait-cfg').value),
    seed:          parseInt(document.getElementById('portrait-seed').value),
  };

  fetch('/api/face-portrait', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params)})
  .then(function(r){return r.json();})
  .then(function(d){
    if (!d.ok) { toast(d.error||'Błąd', 'err'); btn.disabled=false; btn.textContent='👤 GENERUJ PORTRET'; status.textContent=''; return; }
    startForgeProgress('portrait-prog-fill', 'portrait-prog-pct', 'portrait-prog-detail');
    document.getElementById('portrait-progress-wrap').style.display = 'block';
    pollPortrait(d.job_id, btn, status);
  })
  .catch(function(e){ toast('Błąd połączenia', 'err'); btn.disabled=false; btn.textContent='👤 GENERUJ PORTRET'; status.textContent=''; });
}

function pollPortrait(jid, btn, status) {
  fetch('/api/job/'+jid).then(function(r){return r.json();}).then(function(d){
    if (d.status === 'done') {
      btn.disabled = false;
      stopForgeProgress();
      document.getElementById('portrait-progress-wrap').style.display = 'none';
      btn.textContent = '👤 GENERUJ PORTRET';
      status.textContent = '';
      var results = document.getElementById('portrait-results');
      while (results.firstChild) results.removeChild(results.firstChild);
      (d.images||[]).forEach(function(path){
        var wrap = document.createElement('div');
        wrap.style.cssText = 'position:relative;width:100%';
        var img = document.createElement('img');
        img.src = '/img/'+path;
        img.style.cssText = 'width:100%;border-radius:10px;border:1px solid var(--border);cursor:zoom-in';
        img.onclick = function(){ openLightbox('/img/'+path); };
        var dl = document.createElement('a');
        dl.href = '/img/'+path;
        dl.download = path.split('/').pop();
        dl.style.cssText = 'display:block;margin-top:6px;text-align:center;background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:5px 10px;border-radius:6px;font-size:.8rem;text-decoration:none';
        dl.textContent = '⬇ Pobierz';
        wrap.appendChild(img);
        wrap.appendChild(dl);
        results.appendChild(wrap);
      });
    } else if (d.status === 'error') {
      toast('Błąd: '+d.error, 'err');
      btn.disabled = false;
      stopForgeProgress();
      document.getElementById('portrait-progress-wrap').style.display = 'none';
      btn.textContent = '👤 GENERUJ PORTRET';
      status.textContent = '';
    } else {
      status.textContent = d.status === 'generating' ? 'Generowanie portretu...' : 'Oczekiwanie...';
      setTimeout(function(){ pollPortrait(jid, btn, status); }, 2000);
    }
  });
}

function handleEditFile(file) {
  if(!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    _editOrigDataUrl = e.target.result;
    _editSrcDataUrl  = e.target.result;
    applyScaleAndLoad(e.target.result);
  };
  reader.readAsDataURL(file);
}

function applyScaleAndLoad(origDataUrl) {
  var maxPx = parseInt(document.getElementById('scale-sel').value) || 0;
  if(!maxPx) {
    loadEditImage(origDataUrl);
  } else {
    resizeImageForEdit(origDataUrl, maxPx, loadEditImage);
  }
}

function resizeImageForEdit(dataUrl, maxPx, callback) {
  var img = new Image();
  img.onload = function() {
    var w = img.naturalWidth, h = img.naturalHeight;
    if(w <= maxPx && h <= maxPx) { callback(dataUrl); return; }
    var scale = maxPx / Math.max(w, h);
    var nw = Math.round(w * scale), nh = Math.round(h * scale);
    var c = document.createElement('canvas');
    c.width = nw; c.height = nh;
    c.getContext('2d').drawImage(img, 0, 0, nw, nh);
    callback(c.toDataURL('image/jpeg', 0.93));
  };
  img.src = dataUrl;
}

function onScaleChange() {
  if(!_editOrigDataUrl) return;
  applyScaleAndLoad(_editOrigDataUrl);
}

function updateScaleInfo() {
  var w = parseInt(document.getElementById('edit-w-in').value) || 0;
  var h = parseInt(document.getElementById('edit-h-in').value) || 0;
  var el = document.getElementById('scale-info');
  if(el && w && h) el.textContent = 'Do AI: ' + w + ' × ' + h + 'px';
}

function handleEditDrop(e) {
  e.preventDefault();
  document.getElementById('upload-area').classList.remove('drag');
  var f = e.dataTransfer.files[0];
  if(f && f.type.startsWith('image/')) handleEditFile(f);
}

function loadEditImage(dataUrl) {
  document.getElementById('edit-comparison').style.display='none';
  var img = document.getElementById('edit-src-img');
  img.onload = function() {
    var canvas = document.getElementById('mask-canvas');
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    document.getElementById('edit-w-in').value = img.naturalWidth;
    document.getElementById('edit-h-in').value = img.naturalHeight;
    clearMask();
    initMaskCanvas();
    updateScaleInfo();
  };
  img.src = dataUrl;
  _editImgB64 = dataUrl.indexOf(',') > -1 ? dataUrl.split(',')[1] : dataUrl;
  document.getElementById('upload-area').style.display = 'none';
  document.getElementById('edit-canvas-wrap').style.display = '';
  document.getElementById('mask-toolbar').style.display = '';
  document.getElementById('mask-hint').style.display = '';
}

function initMaskCanvas() {
  var canvas = document.getElementById('mask-canvas');
  var getPos = function(e, isTouch) {
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width  / rect.width;
    var scaleY = canvas.height / rect.height;
    var src = isTouch ? e.touches[0] : e;
    return {x:(src.clientX-rect.left)*scaleX, y:(src.clientY-rect.top)*scaleY};
  };
  var draw = function(pos) {
    var ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, parseInt(document.getElementById('brush-size').value)/2, 0, Math.PI*2);
    ctx.fillStyle = _editMode==='paint' ? 'white' : 'black';
    ctx.fill();
  };
  canvas.onmousedown  = function(e){ _editDrawing=true; draw(getPos(e,false)); e.preventDefault(); };
  canvas.onmousemove  = function(e){ if(_editDrawing) draw(getPos(e,false)); };
  canvas.onmouseup    = function(){ _editDrawing=false; };
  canvas.onmouseleave = function(){ _editDrawing=false; };
  canvas.ontouchstart = function(e){ _editDrawing=true; draw(getPos(e,true)); e.preventDefault(); };
  canvas.ontouchmove  = function(e){ if(_editDrawing) draw(getPos(e,true)); e.preventDefault(); };
  canvas.ontouchend   = function(){ _editDrawing=false; };
}

function clearMask() {
  var canvas = document.getElementById('mask-canvas');
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = 'black';
  ctx.fillRect(0,0,canvas.width,canvas.height);
}

function setEditMode(mode) {
  _editMode = mode;
  document.getElementById('btn-paint').classList.toggle('active', mode==='paint');
  document.getElementById('btn-erase').classList.toggle('active', mode==='erase');
}

function isMaskEmpty() {
  var canvas = document.getElementById('mask-canvas');
  if(!canvas.width) return true;
  var d = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
  for(var i=0;i<d.length;i+=4) if(d[i]>10) return false;
  return true;
}

function getMaskB64() {
  if(isMaskEmpty()) return null;
  return document.getElementById('mask-canvas').toDataURL('image/png').split(',')[1];
}

function updateDenoisingLabel() {
  document.getElementById('denoising-val').textContent =
    parseFloat(document.getElementById('denoising-sl').value).toFixed(2);
}

function toggleEditAdv() {
  var sec = document.getElementById('edit-adv-section');
  var tog = document.getElementById('edit-adv-toggle');
  var open = sec.style.display==='none';
  sec.style.display = open ? '' : 'none';
  tog.classList.toggle('open', open);
}

function genEditAiPrompt() {
  var desc = document.getElementById('edit-desc-ta').value.trim();
  if(!desc) return toast('Najpierw opisz co chcesz zmienić', 'err');
  var btn = document.getElementById('edit-ai-btn');
  btn.disabled=true; btn.innerHTML='&#8987; AI<br>Prompt';
  fetch('/api/ai-prompt', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({description: desc, mode: 'edit'})
  }).then(function(r){return r.json();}).then(function(d){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    if(d.ok) {
      document.getElementById('edit-positive-ta').value = d.positive;
      document.getElementById('edit-negative-ta').value = d.negative;
      toast('Prompt wygenerowany!','ok');
    } else toast('Błąd AI: '+d.error,'err');
  }).catch(function(){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    toast('Błąd połączenia','err');
  });
}

function startEdit() {
  if(!_editImgB64) return toast('Najpierw wczytaj zdjęcie','err');
  var pos = document.getElementById('edit-positive-ta').value.trim();
  if(!pos) return toast('Wpisz positive prompt lub użyj AI Prompt','err');
  var modelSel = document.getElementById('edit-model-sel');
  var params = {
    image_b64:   _editImgB64,
    mask_b64:    getMaskB64(),
    description: document.getElementById('edit-desc-ta').value.trim(),
    positive:    pos,
    negative:    document.getElementById('edit-negative-ta').value.trim(),
    model:       modelSel ? modelSel.value : '',
    sampler:     document.getElementById('edit-sampler-sel').value,
    scheduler:   document.getElementById('edit-sched-sel').value,
    steps:       parseInt(document.getElementById('edit-steps-in').value),
    cfg:         parseFloat(document.getElementById('edit-cfg-in').value),
    denoising:   parseFloat(document.getElementById('denoising-sl').value),
    width:       parseInt(document.getElementById('edit-w-in').value),
    height:      parseInt(document.getElementById('edit-h-in').value),
    seed:        parseInt(document.getElementById('edit-seed-in').value),
  };
  document.getElementById('edit-btn').disabled=true;
  document.getElementById('edit-btn').textContent='Edytowanie...';
  document.getElementById('edit-progress-wrap').style.display='block';
  document.getElementById('edit-prog-fill').style.width='5%';
  document.getElementById('edit-prog-label').textContent='Edytowanie...';
  document.getElementById('edit-prog-pct').textContent='';
  document.getElementById('edit-comparison').style.display='none';
  startForgeProgress('edit-prog-fill', 'edit-prog-pct');
  fetch('/api/edit', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(params)
  }).then(function(r){return r.json();}).then(function(d){
    if(d.job_id) pollEditJob(d.job_id, 0);
    else {
      resetEditBtn();
      document.getElementById('edit-progress-wrap').style.display='none';
      toast('Błąd: '+(d.error||'nieznany'), 'err');
    }
  }).catch(function(){ resetEditBtn(); toast('Błąd połączenia','err'); });
}

function pollEditJob(jid, tick) {
  fetch('/api/job/'+jid).then(function(r){return r.json();}).then(function(d){
    if(d.status==='done') {
      stopForgeProgress();
      document.getElementById('edit-prog-fill').style.width='100%';
      document.getElementById('edit-prog-pct').textContent='';
      setTimeout(function(){
        document.getElementById('edit-progress-wrap').style.display='none';
        showEditImages(d.images, d.seed);
        resetEditBtn();
        loadEditHistory();
      },400);
    } else if(d.status==='error') {
      stopForgeProgress();
      document.getElementById('edit-progress-wrap').style.display='none';
      resetEditBtn(); toast('Błąd: '+d.error,'err');
    } else {
      setTimeout(function(){pollEditJob(jid, tick+1);}, 2000);
    }
  }).catch(function(){ setTimeout(function(){pollEditJob(jid, tick+1);}, 3000); });
}

function showEditImages(paths, seed) {
  _editLastResult = paths.length ? paths[0] : null;

  /* Sekcja porównania przed/po */
  var comp = document.getElementById('edit-comparison');
  if(_editSrcDataUrl && paths.length) {
    document.getElementById('edit-before-img').src = _editSrcDataUrl;
    var afterImg = document.getElementById('edit-after-img');
    afterImg.src = '/img/' + paths[0];
    afterImg.onclick = (function(p0){return function(){_lbImgs=paths;openLb(0);};})(paths[0]);
    var afterBtn = document.getElementById('edit-after-edit-btn');
    afterBtn.onclick = (function(p0){return function(){openInEditor('/img/'+p0);};})(paths[0]);
    comp.style.display = '';
  }

  var grid = document.getElementById('edit-img-grid');
  grid.innerHTML='';
  _lbImgs = paths;
  paths.forEach(function(p,i){
    var wrap=document.createElement('div'); wrap.className='result-wrap';
    var img=document.createElement('img');
    img.className='res-img'; img.src='/img/'+p; img.loading='lazy';
    img.onclick=function(){openLb(i);};
    var btn=document.createElement('button'); btn.className='result-overlay-btn';
    btn.textContent='↺ Edytuj dalej';
    btn.onclick=(function(path){return function(e){e.stopPropagation();openInEditor('/img/'+path);};})(p);
    wrap.appendChild(img); wrap.appendChild(btn);
    grid.appendChild(wrap);
  });
  document.getElementById('edit-seed-info').textContent = seed>0 ? 'Seed: '+seed : '';
  toast('Edycja gotowa!','ok');
}

function resetEditBtn() {
  var b=document.getElementById('edit-btn');
  b.disabled=false; b.innerHTML='&#9999;&#65039; EDYTUJ ZDJĘCIE';
}

function openInEditor(src) {
  var img = new Image();
  img.crossOrigin='anonymous';
  img.onload=function(){
    var c=document.createElement('canvas');
    c.width=img.naturalWidth; c.height=img.naturalHeight;
    c.getContext('2d').drawImage(img,0,0);
    var dataUrl = c.toDataURL('image/jpeg', 0.95);
    _editOrigDataUrl = dataUrl;
    _editSrcDataUrl  = dataUrl;
    switchView('edit');
    applyScaleAndLoad(dataUrl);
  };
  img.onerror=function(){ toast('Nie można załadować zdjęcia','err'); };
  img.src=src;
}

function editFromLb() { closeLb(); openInEditor('/img/'+_lbImgs[_lbIdx]); }

function loadEditHistory() {
  fetch('/api/edit-history').then(function(r){return r.json();}).then(function(d){
    document.getElementById('edit-hist-count').textContent = d.length?'('+d.length+')':'';
    if(!_editHistOpen) return;
    var body=document.getElementById('edit-hist-body');
    body.innerHTML='';
    if(!d.length){
      var em=document.createElement('div'); em.className='no-hist';
      em.textContent='Brak historii edycji'; body.appendChild(em); return;
    }
    d.forEach(function(g){
      var paths=JSON.parse(g.paths||'[]');
      var item=document.createElement('div'); item.className='hist-item';
      var thumbsDiv=document.createElement('div'); thumbsDiv.className='hist-thumbs';
      paths.slice(0,4).forEach(function(p,i){
        var img=document.createElement('img'); img.className='hist-thumb'; img.loading='lazy';
        img.src='/img/'+encodeURI(p);
        img.onclick=(function(ps,idx){return function(){openHistLb(ps,idx);};})(paths,i);
        thumbsDiv.appendChild(img);
      });
      var meta=document.createElement('div'); meta.className='hist-meta';
      var desc=document.createElement('div'); desc.className='hist-desc';
      desc.textContent=g.description||(g.positive||'').substring(0,60)+'...';
      var dt=new Date(g.ts*1000).toLocaleString('pl');
      var tags=document.createElement('div'); tags.className='hist-tags';
      tags.textContent=[g.model,g.sampler,'denoising: '+g.denoising,g.width+'x'+g.height,dt].join(' · ');
      var actions=document.createElement('div'); actions.className='hist-actions';
      var del=document.createElement('button'); del.className='hist-btn del'; del.textContent='✕ Usuń';
      (function(gid,itm){ del.onclick=function(){ deleteEditHist(gid,itm); }; })(g.id,item);
      actions.appendChild(del);
      meta.appendChild(desc); meta.appendChild(tags); meta.appendChild(actions);
      item.appendChild(thumbsDiv); item.appendChild(meta);
      body.appendChild(item);
    });
  });
}

function toggleEditHist() {
  _editHistOpen=!_editHistOpen;
  var body=document.getElementById('edit-hist-body');
  var arrow=document.getElementById('edit-hist-arrow');
  body.style.display=_editHistOpen?'flex':'none';
  arrow.style.transform=_editHistOpen?'rotate(180deg)':'';
  if(_editHistOpen) loadEditHistory();
}


/* ── Poza / Pose View ── */
var _poseBodyB64 = null;

function loadPoseBodyFile(file) {
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    _poseBodyB64 = e.target.result.split(',')[1];
    document.getElementById('pose-body-thumb').src = e.target.result;
    document.getElementById('pose-body-thumb-wrap').style.display = 'block';
    document.getElementById('pose-body-drop').style.display = 'none';
    _detectFacePreview(_poseBodyB64,
      'pose-face-wrap', 'pose-face-loading',
      'pose-face-thumb', 'pose-face-notfound');
  };
  reader.readAsDataURL(file);
}
function clearPoseBody() {
  _poseBodyB64 = null;
  document.getElementById('pose-body-thumb-wrap').style.display = 'none';
  document.getElementById('pose-face-wrap').style.display = 'none';
  document.getElementById('pose-face-loading').style.display = 'none';
  document.getElementById('pose-face-notfound').style.display = 'none';
  document.getElementById('pose-body-drop').style.display = 'flex';
}
function swapPoseWH() {
  var w = document.getElementById('pose-w');
  var h = document.getElementById('pose-h');
  var tmp = w.value; w.value = h.value; h.value = tmp;
}
function startPose() {
  if (!_poseBodyB64) return toast('Wgraj zdjęcie ciała/pozy', 'err');
  var prompt = document.getElementById('pose-prompt').value.trim();
  if (!prompt) return toast('Wpisz opis sceny', 'err');
  var btn = document.getElementById('pose-gen-btn');
  var status = document.getElementById('pose-status');
  btn.disabled = true; btn.textContent = '⏳ Generowanie...';
  status.textContent = 'Przetwarzanie pozy i generowanie...';
  var params = {
    body_image:     _poseBodyB64,
    face_image:     null,  // serwer auto-cropuje z body_image
    positive:       prompt,
    negative:       document.getElementById('pose-negative').value.trim(),
    model:          document.getElementById('pose-model-sel').value,
    body_strength:  parseFloat(document.getElementById('pose-body-strength').value),
    face_strength:  parseFloat(document.getElementById('pose-face-strength').value),
    width:          parseInt(document.getElementById('pose-w').value),
    height:         parseInt(document.getElementById('pose-h').value),
    steps:          parseInt(document.getElementById('pose-steps').value),
    cfg:            parseFloat(document.getElementById('pose-cfg').value),
    seed:           parseInt(document.getElementById('pose-seed').value),
  };
  fetch('/api/pose', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params)})
  .then(function(r){return r.json();})
  .then(function(d){
    if (!d.ok) { toast(d.error||'Błąd', 'err'); btn.disabled=false; btn.textContent='\U0001f6b4 GENERUJ Z POZĄ'; status.textContent=''; return; }
    startForgeProgress('pose-prog-fill', 'pose-prog-pct', 'pose-prog-detail');
    document.getElementById('pose-progress-wrap').style.display = 'block';
    pollPose(d.job_id, btn, status);
  })
  .catch(function(){ toast('Błąd połączenia', 'err'); btn.disabled=false; btn.textContent='\U0001f6b4 GENERUJ Z POZĄ'; status.textContent=''; });
}
function pollPose(jid, btn, status) {
  fetch('/api/job/'+jid).then(function(r){return r.json();}).then(function(d){
    if (d.status === 'done') {
      stopForgeProgress();
      document.getElementById('pose-progress-wrap').style.display = 'none';
      btn.disabled = false; btn.textContent = '\U0001f6b4 GENERUJ Z POZĄ'; status.textContent = '';
      var results = document.getElementById('pose-results');
      while (results.firstChild) results.removeChild(results.firstChild);
      (d.images||[]).forEach(function(path){
        var wrap = document.createElement('div'); wrap.style.cssText = 'position:relative;width:100%';
        var img = document.createElement('img');
        img.src = '/img/'+path;
        img.style.cssText = 'width:100%;border-radius:10px;border:1px solid var(--border);cursor:zoom-in';
        img.onclick = function(){ openLightbox('/img/'+path); };
        var dl = document.createElement('a');
        dl.href = '/img/'+path; dl.download = path.split('/').pop();
        dl.style.cssText = 'display:block;margin-top:6px;text-align:center;background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:5px 10px;border-radius:6px;font-size:.8rem;text-decoration:none';
        dl.textContent = '\u2B07 Pobierz';
        wrap.appendChild(img); wrap.appendChild(dl); results.appendChild(wrap);
      });
    } else if (d.status === 'error') {
      stopForgeProgress();
      document.getElementById('pose-progress-wrap').style.display = 'none';
      toast('Błąd: '+(d.error||'nieznany'), 'err');
      btn.disabled = false; btn.textContent = '\U0001f6b4 GENERUJ Z POZĄ'; status.textContent = '';
    } else {
      status.textContent = 'Generowanie z pozą...';
      setTimeout(function(){ pollPose(jid, btn, status); }, 2000);
    }
  });
}
/* ── Init ── */
applyPreset('portrait');
loadHistory();

/* Auto-detect mobile — większy pędzel i bezpieczna skala */
if('ontouchstart' in window || window.innerWidth < 640) {
  var bsEl = document.getElementById('brush-size');
  if(bsEl) { bsEl.value = 40; document.getElementById('brush-sz-lbl').textContent = '40px'; }
  var scEl = document.getElementById('scale-sel');
  if(scEl) scEl.value = '1024';
}
</script>
</body>
</html>'''

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

    def _authed(self):
        return self._cookie() in SESSIONS

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
        model_opts = ''
        for m in raw_models:
            n = html.escape(m.get('model_name',''))
            t = html.escape(m.get('model_name',''))
            model_opts += f'<option value="{t}">{n}</option>'
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

        # Static image serving
        if path.startswith('/img/'):
            rel = unquote(path[5:])
            fpath = (OUTPUTS_DIR / rel).resolve()
            if str(fpath).startswith(str(OUTPUTS_DIR.resolve())) and fpath.is_file():
                mime, _ = mimetypes.guess_type(str(fpath))
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', mime or 'image/png')
                self.send_header('Content-Length', len(data))
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
            return
        if path == '/api/models/refresh':
            _refresh_models_once()
            self._json({'ok': True})
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
                _auto_cache[cache_key] = {'model': chosen, 'reason': reason, 'ts': _t.time()}
                self._json({'ok': True, 'model': chosen, 'reason': reason, 'cached': False})
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
                _auto_cache[mname] = {'settings': settings, 'ts': _t.time()}
                self._json({'ok': True, 'settings': settings, 'cached': False})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        # Logout
        if path == '/logout':
            tok = self._cookie()
            if tok: SESSIONS.discard(tok)
            self.send_response(302)
            self.send_header('Location', '/login')
            self.send_header('Set-Cookie', f'{COOKIE_NAME}=; Max-Age=0; Path=/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return

        if path == '/login':
            self._html(LOGIN_PAGE.replace('__ERR__', ''))
            return

        if not self._authed():
            self._redirect('/login')
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
                        'SELECT * FROM generations ORDER BY ts DESC LIMIT 50'
                    ).fetchall()
            self._json([dict(r) for r in rows])
            return

        if path == '/api/models':
            with _models_lock:
                raw = list(_models_cache)
            self._json([{'name': m.get('model_name',''), 'title': m.get('title','')} for m in raw])
            return

        if path == '/api/ravnet-models':
            with _ravnet_models_lock:
                raw = list(_ravnet_models_cache)
            self._json([{'name': m.get('model_name',''), 'title': m.get('title','')} for m in raw])
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
                self._json({'ok': False, 'error': str(e)})
            return

        if path == '/api/vram-free':
            try:
                import subprocess as _sp
                _sp.Popen(
                    ['systemctl', '--user', 'restart', 'forge'],
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
                )
                self._json({'ok': True, 'msg': 'Forge restartuje — ~30s przerwy'})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
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

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length).decode('utf-8', errors='replace')

        if self.path == '/api/download-model':
            try:
                data = json.loads(body)
                url = data.get('url', '').strip()
            except:
                self._json({'ok': False, 'error': 'bad JSON'}); return
            if not url or not (url.startswith('http://') or url.startswith('https://')):
                self._json({'ok': False, 'error': 'nieprawidłowy URL'}); return
            job_id = uuid.uuid4().hex[:12]
            _downloads[job_id] = {'url': url, 'filename': '', 'percent': 0,
                                   'status': 'starting', 'error': '', 'bytes_done': 0, 'size': 0}
            threading.Thread(target=_download_model_thread, args=(job_id, url), daemon=True).start()
            self._json({'ok': True, 'job_id': job_id})
            return
        if self.path == '/login':
            params   = parse_qs(body)
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0]
            pw_hash  = hashlib.sha256(password.encode()).hexdigest()
            if username == GP_USERNAME and pw_hash == GP_PW_HASH:
                tok = secrets.token_hex(32)
                SESSIONS.add(tok)
                self.send_response(302)
                self.send_header('Location', '/')
                self.send_header('Set-Cookie', f'{COOKIE_NAME}={tok}; Path=/; HttpOnly; SameSite=Lax')
                self.send_header('Content-Length', '0')
                self.end_headers()
            else:
                err = '<div class="err">Błędny login lub hasło</div>'
                self._html(LOGIN_PAGE.replace('__ERR__', err))
            return

        if not self._authed():
            self._json({'ok': False, 'error': 'unauthorized'}, 401)
            return

        if self.path == '/api/image-meta-upload':
            try:
                import base64 as _b64, io as _io
                payload = json.loads(body)
                data = _b64.b64decode(payload['data'])
                from PIL import Image as _Img
                with _Img.open(_io.BytesIO(data)) as im:
                    params_str = im.info.get('parameters','')
                result = {'ok': bool(params_str), 'raw': params_str}
                if params_str:
                    lines = params_str.split('\n')
                    result['positive'] = lines[0].strip() if lines else ''
                    neg = next((l.replace('Negative prompt:','').strip() for l in lines if l.startswith('Negative prompt:')), '')
                    result['negative'] = neg
                    meta_line = next((l for l in lines if 'Steps:' in l), '')
                    for kv in meta_line.split(','):
                        kv = kv.strip()
                        if ': ' in kv:
                            k, v = kv.split(': ', 1)
                            result[k.strip().lower().replace(' ','_')] = v.strip()
                self._json(result)
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        if self.path == '/api/generate':
            try:
                params = json.loads(body)
                params['gen_id'] = uuid.uuid4().hex
                jid = job_create()
                backend = params.get('backend', 'forge')
                if backend == 'krea2':
                    target = krea_generate_thread
                elif backend == 'ravnet':
                    target = ravnet_forge_generate_thread
                elif params.get('init_image'):
                    target = forge_img2img_ref_thread
                else:
                    target = forge_generate_thread
                t = threading.Thread(target=target, args=(params, jid), daemon=True)
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
                t = threading.Thread(target=forge_instantid_thread, args=(params, jid), daemon=True)
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
                t = threading.Thread(target=forge_pose_thread, args=(params, jid), daemon=True)
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
                if not image_b64:
                    self._json({'ok': False, 'error': 'Brak image_b64'}); return
                pos, neg = ai_vision_describe(image_b64, style)
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
                t = threading.Thread(target=forge_edit_thread, args=(params, jid), daemon=True)
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
    _refresh_models_once()  # zaladuj cache przed pierwszym requestem
    server.serve_forever()
