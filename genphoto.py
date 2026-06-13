#!/usr/bin/env python3
"""GenPhoto — AI photo generation studio (frontend for Stable Diffusion Forge)"""

import base64, hashlib, html, json, mimetypes, os, secrets, sqlite3
import threading, time, uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote
import urllib.request, urllib.error

# ── Config ───────────────────────────────────────────────────────────────────
PORT         = int(os.environ.get('GP_PORT', '7862'))
OUTPUTS_DIR  = Path(os.environ.get('GP_OUTPUTS_DIR', '/home/user/stable-diffusion-webui/outputs'))
FORGE_URL    = os.environ.get('GP_FORGE_URL',  'http://localhost:7860').rstrip('/')
DEEPSEEK_KEY   = os.environ.get('GP_DEEPSEEK_KEY', '')
DEEPSEEK_MODEL = os.environ.get('GP_DEEPSEEK_MODEL', 'deepseek-v4-flash')
DB_PATH      = Path(os.environ.get('GP_DB_PATH', '/home/user/genphoto.db'))
GP_USERNAME  = os.environ.get('GP_USERNAME', 'admin')
GP_PW_HASH   = os.environ.get('GP_PASSWORD_HASH', '')
COOKIE_NAME  = 'gp_sess'
GALLERY_URL  = os.environ.get('GP_GALLERY_URL', '')

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

def deepseek_prompt(description):
    if not DEEPSEEK_KEY:
        raise RuntimeError('GP_DEEPSEEK_KEY not set')
    raw = json.dumps({
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': description},
        ],
        'temperature': 0.6,
        'max_tokens': 500,
    }).encode()
    req = urllib.request.Request(
        'https://api.deepseek.com/chat/completions', data=raw,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_KEY}',
        }, method='POST'
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    msg  = data['choices'][0]['message']
    text = (msg.get('content') or '').strip()
    if not text:
        # reasoning models put output in reasoning_content
        text = (msg.get('reasoning_content') or '').strip()
    if not text:
        raise RuntimeError(f'DeepSeek zwrócił pustą odpowiedź: {list(msg.keys())}')
    pos = neg = ''
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith('POSITIVE:'):
            pos = l[9:].strip().lstrip(':').strip()
        elif l.upper().startswith('NEGATIVE:'):
            neg = l[9:].strip().lstrip(':').strip()
    if not pos and not neg:
        # model nie odpowiedział w formacie POSITIVE:/NEGATIVE: — zwróć surowy tekst
        raise RuntimeError(f'Nieprawidłowy format odpowiedzi DeepSeek: {text[:200]}')
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
</style>
</head>
<body>

<header>
  <div class="logo">&#127912; GenPhoto</div>
  <div class="preset-tabs" id="preset-tabs">
    __PRESET_TABS__
  </div>
  <div class="hdr-links">
    <a href="__GALLERY_URL__" target="_blank" class="hdr-btn">&#128193; Galeria</a>
    <a href="/logout" class="hdr-btn">&#10155;</a>
  </div>
</header>

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

<!-- ── Lightbox ── -->
<div id="lb">
  <button id="lb-close" onclick="closeLb()">&#10005;</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-nav">
    <button onclick="navLb(-1)">&#8592; Poprzednie</button>
    <button id="lb-dl" onclick="dlLb()">&#8595; Pobierz</button>
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
    var pct = Math.min(10 + tick*7, 90);
    document.getElementById('prog-fill').style.width = pct+'%';
    if(d.status === 'done') {
      document.getElementById('prog-fill').style.width='100%';
      setTimeout(function(){showProgress(false); showImages(d.images, d.seed); resetBtn(); loadHistory();}, 400);
    } else if(d.status === 'error') {
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
      var delBtn = document.createElement('button'); delBtn.className='hist-btn del';
      delBtn.textContent='🗑 Usuń';
      delBtn.onclick=(function(id,el){return function(){deleteHist(id,el);};})(g.id,item);
      actions.appendChild(btn); actions.appendChild(delBtn);

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

/* ── Init ── */
applyPreset('portrait');
loadHistory();
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
        return (HTML_TEMPLATE
                .replace('__PRESET_TABS__', preset_tabs)
                .replace('__MODEL_OPTIONS__', model_opts)
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
                if not desc:
                    self._json({'ok': False, 'error': 'Brak opisu'}); return
                pos, neg = deepseek_prompt(desc)
                self._json({'ok': True, 'positive': pos, 'negative': neg})
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
