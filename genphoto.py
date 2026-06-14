#!/usr/bin/env python3
"""GenPhoto — AI photo generation studio (frontend for Stable Diffusion Forge)"""

import base64, hashlib, html, json, mimetypes, os, secrets, shutil, sqlite3
import threading, time, uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote
import urllib.request, urllib.error

# ── Config ───────────────────────────────────────────────────────────────────
PORT         = int(os.environ.get('GP_PORT', '7862'))
OUTPUTS_DIR  = Path(os.environ.get('GP_OUTPUTS_DIR', '/home/bartek/forge/outputs'))
FORGE_URL    = os.environ.get('GP_FORGE_URL',  'http://localhost:7860').rstrip('/')
DEEPSEEK_KEY   = os.environ.get('GP_DEEPSEEK_KEY', '')
DEEPSEEK_MODEL = os.environ.get('GP_DEEPSEEK_MODEL', 'deepseek-v4-flash')
AI_PROVIDER    = os.environ.get('GP_AI_PROVIDER', 'deepseek')
OR_KEY         = os.environ.get('GP_OR_KEY', '')
OR_MODEL       = os.environ.get('GP_OR_MODEL', 'nousresearch/hermes-4-405b')
DB_PATH      = Path(os.environ.get('GP_DB_PATH', '/home/bartek/genphoto.db'))
GP_USERNAME  = os.environ.get('GP_USERNAME', 'admin')
GP_PW_HASH   = os.environ.get('GP_PASSWORD_HASH', '')
COOKIE_NAME  = 'gp_sess'
GALLERY_URL  = os.environ.get('GP_GALLERY_URL', 'https://gallery.ebartnet.pl')

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
    _c.execute('''CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY, ts INTEGER,
        description TEXT, positive TEXT, negative TEXT,
        model TEXT, sampler TEXT, steps INTEGER, cfg REAL,
        width INTEGER, height INTEGER, seed INTEGER,
        frames INTEGER, fps INTEGER, preset TEXT, path TEXT, source_path TEXT
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
    {'id':'portrait',  'name':'Portrait',  'icon':'&#128100;',
     'model':'RealVisXL_V5_fp16',   'sampler':'DPM++ SDE',    'scheduler':'Karras',
     'steps':30, 'cfg':5.5, 'width':832,  'height':1216, 'batch':4,
     'prefix':'RAW photo, real person, detailed skin texture, natural skin pores, (photorealistic:1.4), '
              '(realistic:1.3), 8k uhd, sharp focus, natural lighting, ',
     'negative': NEG},
    {'id':'landscape', 'name':'Krajobraz', 'icon':'&#127748;',
     'model':'juggernautXL_ragnarok','sampler':'DPM++ 2M SDE','scheduler':'Karras',
     'steps':25, 'cfg':6.5, 'width':1216, 'height':832,  'batch':2,
     'prefix':'RAW photo, (photorealistic:1.4), (realistic:1.3), 8k uhd, landscape photography, '
              'dramatic lighting, high detail, sharp focus, ',
     'negative': NEG},
    {'id':'fashion',   'name':'Fashion',   'icon':'&#128247;',
     'model':'RealVisXL_V5_fp16',   'sampler':'DPM++ SDE',    'scheduler':'Karras',
     'steps':30, 'cfg':5.5, 'width':832,  'height':1216, 'batch':2,
     'prefix':'RAW photo, real person, (photorealistic:1.4), (realistic:1.3), 8k uhd, '
              'fashion photography, professional studio lighting, sharp focus, ',
     'negative': NEG},
    {'id':'headshot',  'name':'Headshot',  'icon':'&#127919;',
     'model':'juggernautXL_ragnarok','sampler':'DPM++ SDE',   'scheduler':'Karras',
     'steps':30, 'cfg':5.5, 'width':1024, 'height':1024, 'batch':2,
     'prefix':'RAW photo, real person, detailed skin texture, (photorealistic:1.4), (realistic:1.3), '
              '8k uhd, professional headshot, 85mm lens, f/1.4, shallow depth of field, bokeh, ',
     'negative': NEG},
]

# ── Forge API ─────────────────────────────────────────────────────────────────
def forge_get(path, timeout=10):
    try:
        with urllib.request.urlopen(f'{FORGE_URL}{path}', timeout=timeout) as r:
            return json.loads(r.read())
    except:
        return None

def forge_post(path, data, timeout=600):
    raw = json.dumps(data).encode()
    req = urllib.request.Request(
        f'{FORGE_URL}{path}', data=raw,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def resolve_model(model_name):
    models = forge_get('/sdapi/v1/sd-models') or []
    for m in models:
        if m.get('model_name') == model_name:
            return m['title']
    return model_name

def forge_generate_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title = resolve_model(params['model'])
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
            'override_settings': {'sd_model_checkpoint': title},
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

def forge_edit_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title = resolve_model(params['model'])
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
            'override_settings':  {'sd_model_checkpoint': title},
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

def forge_video_thread(params, jid):
    try:
        job_set(jid, status='generating')
        title  = resolve_model(params['model'])
        frames = int(params.get('frames', 16))
        fps    = int(params.get('fps', 8))
        loop   = params.get('loop', False)

        animatediff_args = {
            'enable':           True,
            'model':            'v3_sd15_mm.ckpt',
            'video_length':     frames,
            'fps':              fps,
            'loop_number':      0,
            'closed_loop':      'R+P' if loop else 'N',
            'batch_size':       frames,
            'stride':           1,
            'overlap':          -1,
            'format':           ['MP4'],
            'interp':           'Off',
            'interp_x':        10,
            'video_source':     None,
            'video_path':       '',
            'latent_power':     1,
            'latent_scale':     32,
            'last_frame':       None,
            'latent_power_last':1,
            'latent_scale_last':32,
            'request_id':       '',
        }
        common = {
            'prompt':           params['positive'],
            'negative_prompt':  params.get('negative', ''),
            'sampler_name':     params.get('sampler', 'DPM++ 2M Karras'),
            'scheduler':        'Karras',
            'steps':            int(params.get('steps', 25)),
            'cfg_scale':        float(params.get('cfg', 6.5)),
            'width':            int(params.get('width', 512)),
            'height':           int(params.get('height', 512)),
            'seed':             int(params.get('seed', -1)),
            'batch_size':       1,
            'n_iter':           1,
            'override_settings': {
                'sd_model_checkpoint': title,
                'sd_vae': 'None',  # wymuś wbudowany VAE SD1.5 — sdxl_vae daje szary obraz
            },
            'override_settings_restore_afterwards': True,
            'save_images':      True,
            'alwayson_scripts': {'animatediff': {'args': [animatediff_args]}},
        }

        # Ze zdjęcia: CLIP interrogate → wzbogać prompt, potem txt2img
        if params.get('image_b64'):
            try:
                clip_resp = forge_post('/sdapi/v1/interrogate', {
                    'image': 'data:image/png;base64,' + params['image_b64'],
                    'model': 'clip',
                }, timeout=60)
                clip_caption = clip_resp.get('caption', '').strip()
                if clip_caption:
                    common['prompt'] = clip_caption + ', ' + common['prompt']
            except Exception:
                pass  # jeśli CLIP zawiedzie, generuj z samego promptu

        ts_before = time.time()
        data = forge_post('/sdapi/v1/txt2img', common, timeout=1200)

        info = json.loads(data.get('info', '{}'))
        seed = info.get('seed', -1)

        mp4_src = _find_recent_video(ts_before)
        if not mp4_src:
            raise RuntimeError('Nie znaleziono pliku MP4. Sprawdź logi Forge i czy AnimateDiff jest włączony.')

        today   = datetime.now().strftime('%Y-%m-%d')
        out_dir = OUTPUTS_DIR / 'genphoto_videos' / today
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = int(time.time() * 1000)
        name = f'gpv_{ts}_s{seed}.mp4'
        dest = out_dir / name
        shutil.copy2(str(mp4_src), str(dest))
        vid_path = f'genphoto_videos/{today}/{name}'

        gid = params.get('gen_id', uuid.uuid4().hex)
        with _db_lock:
            with db() as con:
                con.execute(
                    'INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (gid, int(time.time()), params.get('description', ''),
                     params['positive'], params.get('negative', ''),
                     params['model'], params.get('sampler', 'DPM++ 2M Karras'),
                     int(params.get('steps', 25)), float(params.get('cfg', 6.5)),
                     int(params.get('width', 512)), int(params.get('height', 512)),
                     seed, frames, fps,
                     params.get('preset', 'custom'), vid_path,
                     params.get('source_path', ''))
                )
        job_set(jid, status='done', images=[vid_path], seed=seed, gen_id=gid)
    except Exception as e:
        job_set(jid, status='error', error=str(e))

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
body{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;min-height:100vh}
a{color:inherit;text-decoration:none}

/* Header */
header{background:#1e293b;border-bottom:1px solid #334155;padding:0 20px;height:54px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100}
.logo{font-size:1.05rem;font-weight:700;color:#93c5fd;white-space:nowrap}
.preset-tabs{display:flex;gap:6px;flex:1;overflow-x:auto;scrollbar-width:none}
.preset-tabs::-webkit-scrollbar{display:none}
.preset-btn{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:5px 14px;border-radius:20px;font-size:.8rem;cursor:pointer;white-space:nowrap;transition:all .2s;display:flex;align-items:center;gap:5px}
.preset-btn:hover{border-color:#60a5fa;color:#e2e8f0}
.preset-btn.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;font-weight:600}
.hdr-links{display:flex;gap:8px;align-items:center;flex-shrink:0}
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
.hist-item{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:14px 16px;display:flex;gap:14px;align-items:flex-start}
.hist-thumbs{display:flex;gap:4px;flex-shrink:0}
.hist-thumb{width:48px;height:48px;object-fit:cover;border-radius:6px;cursor:pointer;border:1px solid #334155}
.hist-meta{flex:1;min-width:0}
.hist-desc{font-size:.85rem;color:#e2e8f0;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:3px}
.hist-tags{font-size:.72rem;color:#64748b;margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-actions{display:flex;gap:6px}
.hist-btn{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:4px 10px;border-radius:6px;font-size:.72rem;cursor:pointer;transition:all .2s}
.hist-btn:hover{border-color:#475569;color:#e2e8f0}
.hist-btn.del{color:#f87171}
.hist-btn.del:hover{border-color:#ef4444;color:#ef4444;background:#1e293b}
.no-hist{color:#334155;text-align:center;padding:20px;font-size:.85rem}

/* Lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:2000;align-items:center;justify-content:center;flex-direction:column}
#lb.open{display:flex}
#lb-img{max-width:90vw;max-height:86vh;object-fit:contain;border-radius:8px}
#lb-nav{display:flex;gap:16px;margin-top:14px;align-items:center}
#lb-nav button{background:#1e293b;border:1px solid #475569;color:#e2e8f0;padding:6px 18px;border-radius:8px;cursor:pointer;font-size:.85rem}
#lb-nav button:hover{background:#334155}
#lb-dl{background:#3b82f6;border-color:#3b82f6;color:#fff}
#lb-close{position:absolute;top:14px;right:18px;background:none;border:none;color:#64748b;font-size:1.6rem;cursor:pointer;line-height:1}

/* Toast */
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(80px);background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:10px 20px;border-radius:8px;font-size:.85rem;z-index:3000;transition:transform .3s;pointer-events:none}
#toast.show{transform:translateX(-50%) translateY(0)}
#toast.ok{border-color:#22c55e;color:#86efac}
#toast.err{border-color:#ef4444;color:#fca5a5}

/* View tabs */
.view-tabs{display:flex;gap:4px;margin:0 10px;flex-shrink:0}
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
  header{flex-wrap:wrap;gap:6px;padding:8px 12px}
  .logo{font-size:.9rem;flex-shrink:0}
  .preset-tabs{order:20;width:100%;border-top:1px solid #1e3a5f;margin-top:4px;padding-top:6px}
  .view-tabs{margin:0 2px;flex-shrink:0}
  .view-tab-btn{padding:4px 10px;font-size:.74rem}
  .hdr-links{flex-shrink:0}
  .hdr-btn{padding:4px 9px;font-size:.74rem}
  main{padding:12px 10px;gap:14px}
  .edit-layout{padding:12px 10px;gap:14px}
  .panel{padding:14px 12px}
  .edit-hist-wrap{padding:0 10px 16px}
  .row2{grid-template-columns:1fr 1fr}
  .row3{grid-template-columns:1fr 1fr 1fr}
  #lb-img{max-width:96vw;max-height:78vh}
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
  <div class="preset-tabs" id="preset-tabs">
    __PRESET_TABS__
  </div>
  <div class="view-tabs">
    <button class="view-tab-btn active" id="tab-gen" onclick="switchView(\'generate\')">&#127912; Generuj</button>
    <button class="view-tab-btn" id="tab-edit" onclick="switchView(\'edit\')">&#9999;&#65039; Edytuj</button>
    <button class="view-tab-btn" id="tab-video" onclick="switchView(\'video\')">&#127916; Wideo</button>
  </div>
  <div class="hdr-links">
    <a href="__GALLERY_URL__" target="_blank" class="hdr-btn">&#128193; Galeria</a>
    <a href="/logout" class="hdr-btn">&#10155;</a>
  </div>
</header>

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

  <div class="field">
    <label>Opisz co chcesz wygenerować (po polsku)</label>
    <div class="ai-row">
      <textarea id="desc-ta" placeholder="np. piękna kobieta na plaży o zachodzie słońca, naturalny uśmiech, letnia sukienka..."></textarea>
      <button class="ai-btn" id="ai-btn" onclick="genAiPrompt()">&#10024; AI<br>Prompt</button>
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
      <label>Model</label>
      <select id="model-sel" onchange="markCustom()">__MODEL_OPTIONS__</select>
    </div>
    <div class="row2">
      <div class="field">
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
      <div class="field"><label>Szerokość</label><input type="number" id="w-in" value="832" step="8" min="256" max="2048" onchange="markCustom()"></div>
      <div class="field"><label>Wysokość</label><input type="number" id="h-in" value="1216" step="8" min="256" max="2048" onchange="markCustom()"></div>
      <div class="field"><label>Batch</label><input type="number" id="batch-in" value="4" min="1" max="8" onchange="markCustom()"></div>
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

<!-- ── Video View ── -->
<div id="view-video" style="display:none">
<div class="edit-layout">

  <!-- Lewa kolumna: źródło + presety -->
  <div class="panel">
    <div class="panel-title">&#127916; Źródło wideo</div>

    <div style="display:flex;gap:6px;margin-bottom:14px">
      <button class="mask-btn active" id="vtab-text" onclick="setVideoSrc(\'text\')">&#9999; Z prompta</button>
      <button class="mask-btn" id="vtab-img" onclick="setVideoSrc(\'image\')">&#128247; Ze zdjęcia</button>
    </div>

    <div id="vid-upload-wrap" style="display:none">
      <div id="vid-upload-area" class="upload-area"
           onclick="document.getElementById(\'vid-file-in\').click()"
           ondragover="event.preventDefault();this.classList.add(\'drag\')"
           ondragleave="this.classList.remove(\'drag\')"
           ondrop="handleVidDrop(event)">
        <div style="font-size:2rem">&#128247;</div>
        <div style="font-size:.85rem;margin-top:6px">Kliknij lub przeciągnij zdjęcie</div>
        <div style="font-size:.73rem;color:#475569;margin-top:3px">Pierwsza klatka klipu</div>
      </div>
      <input type="file" id="vid-file-in" accept="image/*" style="display:none" onchange="handleVidFile(this.files[0])">
      <div id="vid-src-preview" style="display:none;margin-top:10px">
        <img id="vid-src-img" style="max-width:100%;border-radius:8px;display:block" src="" alt="">
        <button class="mask-btn" style="margin-top:6px;font-size:.75rem" onclick="clearVidSrc()">&#10005; Zmień zdjęcie</button>
      </div>
      <div class="field" style="margin-top:12px">
        <label>Siła animacji</label>
        <input type="range" id="vid-denoise-sl" min="0.5" max="1.0" step="0.05" value="0.6"
               oninput="document.getElementById(\'vid-denoise-val\').textContent=parseFloat(this.value).toFixed(2)" style="width:100%;accent-color:#7c3aed">
        <div class="denoising-labels">
          <span>Subtelna</span>
          <span id="vid-denoise-val" style="color:#c4b5fd;font-weight:600">0.60</span>
          <span>Dynamiczna</span>
        </div>
      </div>
    </div>

    <div class="panel-title" style="margin-top:14px">&#9889; Preset klipu</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">
      <button class="vid-preset-btn active" id="vpre-flash" onclick="applyVidPreset(\'flash\')">&#9889; Mgnienie 2s</button>
      <button class="vid-preset-btn" id="vpre-short" onclick="applyVidPreset(\'short\')">&#9654; Krótki 3s</button>
      <button class="vid-preset-btn" id="vpre-std" onclick="applyVidPreset(\'std\')">&#9654;&#9654; Standard 4s</button>
      <button class="vid-preset-btn" id="vpre-cine" onclick="applyVidPreset(\'cine\')">&#127916; Kinowy 4s</button>
      <button class="vid-preset-btn" id="vpre-loop" onclick="applyVidPreset(\'loop\')">&#8635; Loop 3s</button>
      <button class="vid-preset-btn" id="vpre-long" onclick="applyVidPreset(\'long\')">&#8987; Długi 6s</button>
    </div>
    <div id="vid-preset-info" style="font-size:.71rem;color:#64748b;padding:7px 10px;background:#0f172a;border-radius:6px;margin-bottom:12px">
      16 klatek × 8fps = 2s · 512×512 · szybkie
    </div>

    <div class="adv-toggle" id="vid-adv-toggle" onclick="toggleVidAdv()">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="2,4 7,10 12,4"/></svg>
      Zaawansowane
    </div>
    <div id="vid-adv-section" style="display:none;margin-top:12px;padding-top:12px;border-top:1px solid #334155">
      <div class="field">
        <label>Model (wymagany SD 1.5)</label>
        <select id="vid-model-sel">__VID_MODEL_OPTIONS__</select>
      </div>
      <div class="row2">
        <div class="field"><label>Klatki</label><input type="number" id="vid-frames-in" value="16" min="8" max="64" step="8"></div>
        <div class="field"><label>FPS</label><input type="number" id="vid-fps-in" value="8" min="4" max="24" step="2"></div>
      </div>
      <div class="row2">
        <div class="field"><label>Sampler</label>
          <select id="vid-sampler-sel">
            <option>DPM++ 2M Karras</option><option>Euler a</option><option>Euler</option><option>DPM++ SDE</option><option>DPM++ 2M</option><option>DDIM</option>
          </select>
        </div>
        <div class="field"><label>Steps</label><input type="number" id="vid-steps-in" value="25" min="10" max="40"></div>
      </div>
      <div class="row3">
        <div class="field"><label>CFG</label><input type="number" id="vid-cfg-in" value="6.5" min="1" max="15" step="0.5"></div>
        <div class="field"><label>Szerokość</label><input type="number" id="vid-w-in" value="512" step="64" min="256" max="768"></div>
        <div class="field"><label>Wysokość</label><input type="number" id="vid-h-in" value="512" step="64" min="256" max="768"></div>
      </div>
      <div class="row2">
        <div class="field"><label>Seed</label><input type="number" id="vid-seed-in" value="-1"></div>
        <div class="field"><label style="display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="vid-loop-cb" style="width:auto;accent-color:#7c3aed">
          Seamless loop
        </label></div>
      </div>
    </div>
  </div>

  <!-- Prawa kolumna: prompt + wynik -->
  <div class="panel">
    <div class="panel-title">&#128221; Prompt &amp; wynik</div>

    <div class="field">
      <label>Opisz animację (po polsku)</label>
      <div class="ai-row">
        <textarea id="vid-desc-ta" placeholder="np. fale oceanu spokojnie uderzają o brzeg, złoty zachód słońca, lekki wiatr..."></textarea>
        <button class="ai-btn" id="vid-ai-btn" onclick="genVidAiPrompt()" style="background:#7c3aed">&#10024; AI<br>Prompt</button>
      </div>
    </div>
    <div class="field">
      <label>Positive prompt</label>
      <textarea id="vid-positive-ta" style="height:80px;font-size:.82rem;color:#bfdbfe"></textarea>
    </div>
    <div class="field">
      <label>Negative prompt</label>
      <textarea id="vid-negative-ta" style="height:44px;font-size:.78rem;color:#fca5a5"></textarea>
    </div>

    <button class="vid-btn" id="vid-btn" onclick="startVideo()">&#127916; GENERUJ WIDEO</button>

    <div id="vid-progress-wrap" style="display:none;margin-top:16px">
      <div class="progress-label"><span id="vid-prog-label">Generowanie wideo...</span><span id="vid-prog-pct"></span></div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="vid-prog-fill"></div></div>
      <div style="font-size:.7rem;color:#64748b;margin-top:4px">Wideo zajmuje zwykle 2–10 minut zależnie od liczby klatek</div>
    </div>

    <div id="vid-result" style="display:none;margin-top:16px">
      <div style="font-size:.72rem;color:#64748b;margin-bottom:6px">&#127916; Wygenerowany klip</div>
      <video id="vid-player" controls autoplay loop playsinline
             style="width:100%;border-radius:8px;background:#000;display:block;max-height:400px"></video>
      <div style="display:flex;gap:6px;margin-top:8px;align-items:center">
        <a id="vid-dl" class="hist-btn" style="flex:1;text-align:center;text-decoration:none" href="#" download>&#11015; Pobierz MP4</a>
        <span id="vid-seed-show" style="font-size:.72rem;color:#475569"></span>
      </div>
    </div>
  </div>

</div>

<div class="edit-hist-wrap">
<div id="vid-hist-section" class="panel" style="margin-top:0">
  <div class="hist-header" onclick="toggleVidHist()">
    <h2>&#128252; Historia wideo <span id="vid-hist-count" style="color:#475569;font-weight:400"></span></h2>
    <svg id="vid-hist-arrow" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="2"><polyline points="3,5 8,11 13,5"/></svg>
  </div>
  <div id="vid-hist-body" style="display:none;flex-direction:column;gap:10px"></div>
</div>
</div>

</div><!-- /view-video -->

<!-- ── Lightbox ── -->
<div id="lb">
  <button id="lb-close" onclick="closeLb()">&#10005;</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-nav">
    <button onclick="navLb(-1)">&#8592; Poprzednie</button>
    <button id="lb-dl" onclick="dlLb()">&#8595; Pobierz</button>
    <button onclick="editFromLb()" style="background:#059669;border-color:#059669;color:#fff">&#9999;&#65039; Edytuj</button>
    <button onclick="navLb(1)">Następne &#8594;</button>
  </div>
</div>

<div id="toast"></div>

<script>
var PRESETS = __PRESETS_JSON__;
var GALLERY_URL = '__GALLERY_URL__';
var _curPreset = null;
var _lbImgs = [], _lbIdx = 0;
var _pollTimer = null;
var _histOpen = false;

/* ── Presets ── */
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

/* ── Generate ── */
function startGenerate() {
  var pos = document.getElementById('positive-ta').value.trim();
  if(!pos) return toast('Wpisz positive prompt lub użyj AI Prompt', 'err');
  var p = _curPreset ? PRESETS.find(function(x){return x.id===_curPreset;}) : null;
  var fullPos = p ? p.prefix + pos : pos;
  var params = {
    description: document.getElementById('desc-ta').value.trim(),
    positive:    fullPos,
    negative:    document.getElementById('negative-ta').value.trim(),
    model:       document.getElementById('model-sel').value,
    sampler:     document.getElementById('sampler-sel').value,
    scheduler:   document.getElementById('sched-sel').value,
    steps:       parseInt(document.getElementById('steps-in').value),
    cfg:         parseFloat(document.getElementById('cfg-in').value),
    width:       parseInt(document.getElementById('w-in').value),
    height:      parseInt(document.getElementById('h-in').value),
    seed:        parseInt(document.getElementById('seed-in').value),
    batch:       parseInt(document.getElementById('batch-in').value),
    preset:      _curPreset || 'custom',
  };
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
function updLb()     { document.getElementById('lb-img').src='/img/'+_lbImgs[_lbIdx]; }
function dlLb()      { var a=document.createElement('a'); a.href='/img/'+_lbImgs[_lbIdx]; a.download=_lbImgs[_lbIdx].split('/').pop(); a.click(); }

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
      var btn = document.createElement('button'); btn.className='hist-btn';
      btn.textContent='↺ Powtórz';
      btn.onclick=(function(gen){return function(){repeatGen(gen);};})(g);
      var editBtn = document.createElement('button'); editBtn.className='hist-btn';
      editBtn.textContent='✏ Edytuj';
      editBtn.onclick=(function(ps){return function(){ if(ps.length) openInEditor('/img/'+encodeURI(ps[0])); };})(paths);
      var delBtn = document.createElement('button'); delBtn.className='hist-btn del';
      delBtn.textContent='🗑 Usuń';
      delBtn.onclick=(function(id,el){return function(){deleteHist(id,el);};})(g.id,item);
      actions.appendChild(btn); actions.appendChild(editBtn); actions.appendChild(delBtn);

      meta.appendChild(desc); meta.appendChild(tags); meta.appendChild(actions);
      item.appendChild(thumbsDiv); item.appendChild(meta);
      body.appendChild(item);
    });
  });
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

function deleteVidHist(id, el) {
  el.style.opacity='0.4';
  fetch('/api/video-history/'+id, {method:'DELETE'})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ el.remove(); loadVidHistory(); }
      else { el.style.opacity='1'; toast('Błąd usuwania','err'); }
    }).catch(function(){ el.style.opacity='1'; toast('Błąd połączenia','err'); });
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
function startForgeProgress(fillId, pctId) {
  clearTimeout(_fpTimer);
  (function tick(){
    fetch('/api/forge-progress').then(function(r){return r.json();}).then(function(d){
      var pct = Math.round(d.progress * 100);
      if(pct > 2) {
        var f = document.getElementById(fillId);
        if(f) f.style.width = pct + '%';
        var p = pctId ? document.getElementById(pctId) : null;
        if(p) { p.textContent = pct + '%' + (d.eta > 0 ? ' · ETA ' + d.eta + 's' : ''); }
      }
      _fpTimer = setTimeout(tick, 2500);
    }).catch(function(){ _fpTimer = setTimeout(tick, 5000); });
  })();
}
function stopForgeProgress() { clearTimeout(_fpTimer); _fpTimer = null; }

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
  ['generate','edit','video'].forEach(function(n){
    document.getElementById('view-'+n).style.display = v===n ? '' : 'none';
  });
  document.getElementById('tab-gen').classList.toggle('active',   v==='generate');
  document.getElementById('tab-edit').classList.toggle('active',  v==='edit');
  document.getElementById('tab-video').classList.toggle('active', v==='video');
  if(v==='edit')  loadEditHistory();
  if(v==='video') loadVidHistory();
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

/* ── Video View ── */
var _vidSrcMode   = 'text';
var _vidImgB64    = null;
var _vidHistOpen  = false;
var _curVidPreset = 'flash';

var VID_PRESETS = {
  flash: {frames:16,fps:8, w:512,h:512,sampler:'DPM++ 2M Karras',steps:25,cfg:6.5,loop:false, label:'16 klatek × 8fps = 2s · 512×512 · szybkie'},
  short: {frames:24,fps:8, w:512,h:512,sampler:'DPM++ 2M Karras',steps:25,cfg:6.5,loop:false, label:'24 klatki × 8fps = 3s · 512×512'},
  std:   {frames:32,fps:8, w:512,h:512,sampler:'DPM++ 2M Karras',steps:28,cfg:6.5,loop:false, label:'32 klatki × 8fps = 4s · 512×512'},
  cine:  {frames:32,fps:8, w:768,h:512,sampler:'DPM++ 2M Karras',steps:30,cfg:6.5,loop:false, label:'32 klatki × 8fps = 4s · 768×512 kinowy (wolniejsze)'},
  loop:  {frames:24,fps:8, w:512,h:512,sampler:'DPM++ 2M Karras',steps:25,cfg:6.5,loop:true,  label:'24 klatki × 8fps = 3s · 512×512 · seamless loop'},
  long:  {frames:48,fps:8, w:512,h:512,sampler:'DPM++ 2M Karras',steps:28,cfg:6.5,loop:false, label:'48 klatek × 8fps = 6s · 512×512 (wolne!)'},
};

function setVideoSrc(mode) {
  _vidSrcMode = mode;
  document.getElementById('vtab-text').classList.toggle('active', mode==='text');
  document.getElementById('vtab-img').classList.toggle('active',  mode==='image');
  document.getElementById('vid-upload-wrap').style.display = mode==='image' ? '' : 'none';
}

function applyVidPreset(key) {
  _curVidPreset = key;
  var p = VID_PRESETS[key];
  if(!p) return;
  document.querySelectorAll('[id^="vpre-"]').forEach(function(b){b.classList.remove('active');});
  document.getElementById('vpre-'+key).classList.add('active');
  document.getElementById('vid-preset-info').textContent = p.label;
  document.getElementById('vid-frames-in').value = p.frames;
  document.getElementById('vid-fps-in').value    = p.fps;
  document.getElementById('vid-w-in').value      = p.w;
  document.getElementById('vid-h-in').value      = p.h;
  setVal('vid-sampler-sel', p.sampler);
  document.getElementById('vid-steps-in').value  = p.steps;
  document.getElementById('vid-cfg-in').value    = p.cfg;
  document.getElementById('vid-loop-cb').checked = !!p.loop;
}

function toggleVidAdv() {
  var sec = document.getElementById('vid-adv-section');
  var tog = document.getElementById('vid-adv-toggle');
  var open = sec.style.display==='none';
  sec.style.display = open ? '' : 'none';
  tog.classList.toggle('open', open);
}

function handleVidFile(file) {
  if(!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    _vidImgB64 = e.target.result.split(',')[1];
    document.getElementById('vid-src-img').src = e.target.result;
    document.getElementById('vid-upload-area').style.display  = 'none';
    document.getElementById('vid-src-preview').style.display  = '';
  };
  reader.readAsDataURL(file);
}

function handleVidDrop(e) {
  e.preventDefault();
  document.getElementById('vid-upload-area').classList.remove('drag');
  var f = e.dataTransfer.files[0];
  if(f && f.type.startsWith('image/')) handleVidFile(f);
}

function clearVidSrc() {
  _vidImgB64 = null;
  document.getElementById('vid-src-img').src = '';
  document.getElementById('vid-upload-area').style.display  = '';
  document.getElementById('vid-src-preview').style.display  = 'none';
}

function genVidAiPrompt() {
  var desc = document.getElementById('vid-desc-ta').value.trim();
  if(!desc) return toast('Opisz animację', 'err');
  var btn = document.getElementById('vid-ai-btn');
  btn.disabled=true; btn.innerHTML='&#8987; AI<br>Prompt';
  fetch('/api/ai-prompt', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({description: desc, mode: 'video'})
  }).then(function(r){return r.json();}).then(function(d){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    if(d.ok) {
      document.getElementById('vid-positive-ta').value = d.positive;
      document.getElementById('vid-negative-ta').value = d.negative;
      toast('Prompt wygenerowany!','ok');
    } else toast('Błąd AI: '+d.error,'err');
  }).catch(function(){
    btn.disabled=false; btn.innerHTML='&#10024; AI<br>Prompt';
    toast('Błąd połączenia','err');
  });
}

function startVideo() {
  var pos = document.getElementById('vid-positive-ta').value.trim();
  if(!pos) return toast('Wpisz prompt lub użyj AI Prompt','err');
  if(_vidSrcMode==='image' && !_vidImgB64) return toast('Wczytaj zdjęcie źródłowe','err');
  var modelSel = document.getElementById('vid-model-sel');
  var params = {
    image_b64:   _vidSrcMode==='image' ? _vidImgB64 : null,
    description: document.getElementById('vid-desc-ta').value.trim(),
    positive:    pos,
    negative:    document.getElementById('vid-negative-ta').value.trim(),
    model:       modelSel ? modelSel.value : 'v1-5-pruned-emaonly',
    sampler:     document.getElementById('vid-sampler-sel').value,
    steps:       parseInt(document.getElementById('vid-steps-in').value),
    cfg:         parseFloat(document.getElementById('vid-cfg-in').value),
    frames:      parseInt(document.getElementById('vid-frames-in').value),
    fps:         parseInt(document.getElementById('vid-fps-in').value),
    width:       parseInt(document.getElementById('vid-w-in').value),
    height:      parseInt(document.getElementById('vid-h-in').value),
    seed:        parseInt(document.getElementById('vid-seed-in').value),
    denoising:   parseFloat(document.getElementById('vid-denoise-sl').value),
    loop:        document.getElementById('vid-loop-cb').checked,
    preset:      _curVidPreset,
  };
  var btn = document.getElementById('vid-btn');
  btn.disabled=true; btn.textContent='Generowanie...';
  document.getElementById('vid-progress-wrap').style.display='block';
  document.getElementById('vid-prog-fill').style.width='5%';
  document.getElementById('vid-prog-pct').textContent='';
  document.getElementById('vid-result').style.display='none';
  startForgeProgress('vid-prog-fill','vid-prog-pct');
  fetch('/api/video', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(params)
  }).then(function(r){return r.json();}).then(function(d){
    if(d.job_id) pollVidJob(d.job_id);
    else {
      resetVidBtn();
      document.getElementById('vid-progress-wrap').style.display='none';
      toast('Błąd: '+(d.error||'nieznany'),'err');
    }
  }).catch(function(){ resetVidBtn(); toast('Błąd połączenia','err'); });
}

function pollVidJob(jid) {
  fetch('/api/job/'+jid).then(function(r){return r.json();}).then(function(d){
    if(d.status==='done') {
      stopForgeProgress();
      document.getElementById('vid-prog-fill').style.width='100%';
      document.getElementById('vid-prog-pct').textContent='';
      setTimeout(function(){
        document.getElementById('vid-progress-wrap').style.display='none';
        showVidResult(d.images[0], d.seed);
        resetVidBtn();
        loadVidHistory();
      }, 400);
    } else if(d.status==='error') {
      stopForgeProgress();
      document.getElementById('vid-progress-wrap').style.display='none';
      resetVidBtn(); toast('Błąd: '+d.error,'err');
    } else {
      setTimeout(function(){pollVidJob(jid);}, 2000);
    }
  }).catch(function(){ setTimeout(function(){pollVidJob(jid);}, 3000); });
}

function showVidResult(path, seed) {
  var src = '/img/'+path;
  var player = document.getElementById('vid-player');
  player.src = src; player.load();
  var dl = document.getElementById('vid-dl');
  dl.href = src; dl.download = path.split('/').pop();
  document.getElementById('vid-seed-show').textContent = seed>0 ? 'Seed: '+seed : '';
  document.getElementById('vid-result').style.display = '';
  toast('Wideo gotowe!','ok');
}

function resetVidBtn() {
  var b = document.getElementById('vid-btn');
  b.disabled=false; b.innerHTML='&#127916; GENERUJ WIDEO';
}

function loadVidHistory() {
  fetch('/api/video-history').then(function(r){return r.json();}).then(function(d){
    document.getElementById('vid-hist-count').textContent = d.length ? '('+d.length+')' : '';
    if(!_vidHistOpen) return;
    var body = document.getElementById('vid-hist-body');
    body.innerHTML='';
    if(!d.length){
      var em=document.createElement('div'); em.className='no-hist';
      em.textContent='Brak historii wideo'; body.appendChild(em); return;
    }
    d.forEach(function(g){
      var item=document.createElement('div'); item.className='hist-item';
      var thumbs=document.createElement('div'); thumbs.className='hist-thumbs';
      var vid=document.createElement('video');
      vid.src='/img/'+g.path; vid.className='vid-hist-thumb';
      vid.muted=true; vid.loop=true; vid.preload='none';
      vid.onmouseenter=function(){this.play();}; vid.onmouseleave=function(){this.pause(); this.currentTime=0;};
      vid.onclick=function(){ showVidResult(g.path, g.seed||0); switchView('video'); };
      thumbs.appendChild(vid);
      var meta=document.createElement('div'); meta.className='hist-meta';
      var desc=document.createElement('div'); desc.className='hist-desc';
      desc.textContent = g.description || (g.positive||'').substring(0,60)+'...';
      var dt=new Date(g.ts*1000).toLocaleString('pl');
      var dur=((g.frames||16)/(g.fps||8)).toFixed(1);
      var tags=document.createElement('div'); tags.className='hist-tags';
      tags.textContent=[g.model,(g.frames||16)+'kl/'+dur+'s',g.width+'x'+g.height,dt].join(' · ');
      var actions=document.createElement('div'); actions.className='hist-actions';
      var dl=document.createElement('a'); dl.className='hist-btn'; dl.textContent='↓ Pobierz';
      dl.href='/img/'+g.path; dl.download=g.path.split('/').pop(); dl.style.textDecoration='none';
      var del=document.createElement('button'); del.className='hist-btn del'; del.textContent='✕ Usuń';
      (function(gid,itm){ del.onclick=function(){ deleteVidHist(gid,itm); }; })(g.id,item);
      actions.appendChild(dl); actions.appendChild(del);
      meta.appendChild(desc); meta.appendChild(tags); meta.appendChild(actions);
      item.appendChild(thumbs); item.appendChild(meta);
      body.appendChild(item);
    });
  });
}

function toggleVidHist() {
  _vidHistOpen=!_vidHistOpen;
  var body=document.getElementById('vid-hist-body');
  var arrow=document.getElementById('vid-hist-arrow');
  body.style.display=_vidHistOpen?'flex':'none';
  arrow.style.transform=_vidHistOpen?'rotate(180deg)':'';
  if(_vidHistOpen) loadVidHistory();
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
        raw_models = forge_get('/sdapi/v1/sd-models') or []
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
        # Filtruj modele SD1.5 dla AnimateDiff (zawierają "v1-5" lub "sd15" w nazwie)
        sd15_opts = ''
        for m in raw_models:
            n = html.escape(m.get('model_name',''))
            t = html.escape(m.get('model_name',''))
            nl = n.lower()
            if any(k in nl for k in ('v1-5','v1_5','sd15','sd1.5','pruned')):
                sd15_opts += f'<option value="{t}">{n}</option>'
        if not sd15_opts:
            sd15_opts = '<option value="v1-5-pruned-emaonly">v1-5-pruned-emaonly</option>'

        return (HTML_TEMPLATE
                .replace('__PRESET_TABS__', preset_tabs)
                .replace('__MODEL_OPTIONS__', model_opts)
                .replace('__EDIT_MODEL_OPTIONS__', model_opts)
                .replace('__VID_MODEL_OPTIONS__', sd15_opts)
                .replace('__PRESETS_JSON__', presets_json)
                .replace('__GALLERY_URL__', GALLERY_URL))

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
            raw = forge_get('/sdapi/v1/sd-models') or []
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

        if path == '/api/forge-progress':
            prog = forge_get('/sdapi/v1/progress') or {}
            self._json({
                'progress': round(prog.get('progress', 0), 3),
                'eta': int(prog.get('eta_relative', 0)),
            })
            return

        if path == '/api/video-history':
            with _db_lock:
                with db() as con:
                    rows = con.execute(
                        'SELECT * FROM videos ORDER BY ts DESC LIMIT 50'
                    ).fetchall()
            self._json([dict(r) for r in rows])
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
        if path.startswith('/api/video-history/'):
            rec_id = path[19:]
            if not rec_id:
                self._json({'ok': False, 'error': 'brak id'}); return
            with _db_lock:
                with db() as con:
                    con.execute('DELETE FROM videos WHERE id=?', (rec_id,))
            self._json({'ok': True})
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length).decode('utf-8', errors='replace')

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

        if self.path == '/api/generate':
            try:
                params = json.loads(body)
                params['gen_id'] = uuid.uuid4().hex
                jid = job_create()
                t = threading.Thread(target=forge_generate_thread, args=(params, jid), daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
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

        if self.path == '/api/video':
            try:
                params = json.loads(body)
                if not params.get('positive'):
                    self._json({'ok': False, 'error': 'Brak positive prompt'}); return
                frames = int(params.get('frames', 16))
                if frames < 8 or frames > 96:
                    self._json({'ok': False, 'error': 'frames musi być 8–96'}); return
                params['gen_id'] = uuid.uuid4().hex
                jid = job_create()
                t = threading.Thread(target=forge_video_thread, args=(params, jid), daemon=True)
                t.start()
                self._json({'ok': True, 'job_id': jid})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return

        self.send_error(404)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'GenPhoto running on http://0.0.0.0:{PORT}')
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server.serve_forever()
