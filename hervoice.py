#!/usr/bin/env python3
"""hervoice · 把语气变成 AI 能读懂的东西

录音 → 声学静音门 → Whisper 转写(自动识别语种) + 个人声学分类器(可选,并行)
     → LLM 综合判情感(你自己的标签体系 + 相对你平时的声学描述 + 你标过的例子)
     → 记日志 → 触发你自己的回调

核心引擎,与任何具体 AI 助手解耦。说了什么、怎么说的,都交给你接的 AI。
个人化是渐进的:没有标注时只靠 LLM;在 /label/ 标注后,例子和声学分类器自动启用。

配置见 .env(复制 .env.example)。隐私默认:音频阅后即焚,KEEP_AUDIO=1 才留存(标注页需要留存)。
"""
import collections
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import acoustic

# ── 配置(全部走环境变量,不硬编码任何密钥)──
_env = os.environ.get
DATA_DIR = Path(_env("HERVOICE_DATA", "./data"))
LOG = DATA_DIR / "voice_logs.jsonl"
LABELS = DATA_DIR / "labels.jsonl"
MODEL_FILE = DATA_DIR / "acoustic_model.json"
CLIPS = DATA_DIR / "clips"
KEEP_AUDIO = _env("KEEP_AUDIO", "0") == "1"   # 默认阅后即焚

GROQ_KEY = _env("GROQ_API_KEY", "")
LLM_KEY = _env("LLM_API_KEY", "")
LLM_BASE = _env("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = _env("LLM_MODEL", "deepseek-flash")
WHISPER_MODEL = _env("WHISPER_MODEL", "whisper-large-v3")
# 留空=自动识别语种(推荐)。锁死成 zh 时,外语会被 Whisper 直接"翻译"成中文。
WHISPER_LANG = _env("WHISPER_LANG", "")
# 识别出的语种不在这里 → 用 WHISPER_FALLBACK_LANG 重转一次(防短句被认成韩语之类)
WHISPER_LANGS = [x.strip().lower() for x in _env("WHISPER_LANGS", "chinese,english").split(",") if x.strip()]
WHISPER_FALLBACK_LANG = _env("WHISPER_FALLBACK_LANG", "zh")
# 专有名词提示:名字、昵称、常用词。措辞很敏感,见 README「转写提示词」
WHISPER_PROMPT = _env("WHISPER_PROMPT", "")
# Whisper 对着静音会编句子,这些是常见的幻觉
HALLUCINATIONS = [x for x in _env("WHISPER_HALLUCINATIONS",
                                  "请不吝点赞|明镜与点点|Thank you for watching|字幕由").split("|") if x]
# 声学静音门:任一命中就当没人声,不调 Whisper
GATE_MIN_DURATION = float(_env("GATE_MIN_DURATION", "0.5"))
GATE_MAX_PAUSE = float(_env("GATE_MAX_PAUSE", "0.75"))
GATE_MIN_PITCH = float(_env("GATE_MIN_PITCH", "100"))

TAGS = [x.strip() for x in _env(
    "HERVOICE_TAGS", "撒娇,嗔怪·假生气,打闹·玩闹,真生气,委屈,难过,累,焦虑,开心,兴奋,想念,表达爱意,害羞,平静,认真说事"
).split(",") if x.strip()]
SPEAKER = _env("HERVOICE_SPEAKER", "说话的人")       # 给 LLM 的称呼,比如你的名字
LISTENER = _env("HERVOICE_LISTENER", "对方")         # 她在对谁说话
FEWSHOT_N = int(_env("FEWSHOT_N", "15"))
ACOUSTIC_SHOW = [x for x in _env("ACOUSTIC_SHOW_TAGS", "难过,委屈,开心,认真说事,真生气").split(",") if x]
ACOUSTIC_THRESHOLD = float(_env("ACOUSTIC_THRESHOLD", "0.6"))

# 相对基线:特征 → (说法, 偏高词, 偏低词)。音量故意不用——它随设备和麦克风距离乱跳
REL_FEATURES = {"pitch_mean_hz": ("音高", "高", "低"), "pitch_var": ("语调起伏", "大", "小"),
                "pause_ratio": ("停顿", "多", "少"), "tempo_strength": ("节奏", "急", "缓")}
REL_MILD, REL_STRONG, REL_MIN_SAMPLES = 0.75, 1.5, 10

WEBHOOK_URL = _env("HERVOICE_WEBHOOK", "")

app = FastAPI()
_cache = {}


def _read_jsonl(path):
    out = []
    if path.exists():
        for l in path.read_text().splitlines():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def _mtime_cached(name, path, build):
    """按文件 mtime 缓存派生数据,文件一变自动重算"""
    mt = path.stat().st_mtime if path.exists() else 0
    hit = _cache.get(name)
    if hit and hit[0] == mt:
        return hit[1]
    data = build(path)
    _cache[name] = (mt, data)
    return data


def _llm(prompt, max_tokens=200):
    body = json.dumps({"model": LLM_MODEL, "max_tokens": max_tokens, "temperature": 0,
                       # DeepSeek 新模型默认开思维链且与正文共用 max_tokens,打分类任务关掉
                       "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{LLM_BASE}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {LLM_KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


# ── 转写 ──
def _groq(wav_path, lang=None):
    args = ["curl", "-s", "-m", "60", "https://api.groq.com/openai/v1/audio/transcriptions",
            "-H", f"Authorization: Bearer {GROQ_KEY}", "-F", f"file=@{wav_path}",
            "-F", f"model={WHISPER_MODEL}", "-F", "response_format=verbose_json", "-F", "temperature=0"]
    if WHISPER_PROMPT:
        args += ["-F", f"prompt={WHISPER_PROMPT}"]
    if lang:
        args += ["-F", f"language={lang}"]
    d = json.loads(subprocess.run(args, capture_output=True, text=True, timeout=70).stdout)
    if "error" in d:
        raise RuntimeError(str(d["error"])[:200])
    return d.get("text", "").strip(), (d.get("language") or "").lower()


def _whisper(wav_path):
    """返回 (text, err)。text="" 表示没听到人话(幻觉/纯标点),调用方按"没听清"处理"""
    if not GROQ_KEY:
        return None, "GROQ_API_KEY not set"
    try:
        text, lang = _groq(wav_path, WHISPER_LANG or None)
        if text and not WHISPER_LANG and WHISPER_LANGS and lang not in WHISPER_LANGS:
            text, lang = _groq(wav_path, WHISPER_FALLBACK_LANG)
    except Exception as e:
        return None, str(e)[:200]
    if any(h in text for h in HALLUCINATIONS) or not re.sub(r"[\W_]", "", text):
        return "", None
    return text, None


# ── 声学 ──
def _acoustic_features(wav_path):
    """轻量声学特征:音高/能量/停顿/节奏 — 静音门和"比平时X"描述用"""
    import librosa
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    dur = len(y) / sr
    if dur < 0.3:
        return {"duration_s": round(dur, 1)}
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    f0v = f0[(f0 > 60) & (f0 < 500)]
    rms = librosa.feature.rms(y=y)[0]
    silent = float(np.mean(rms < np.percentile(rms, 20) * 1.5))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return {
        "duration_s": round(dur, 1),
        "pitch_mean_hz": round(float(np.mean(f0v)), 1) if len(f0v) else 0,
        "pitch_var": round(float(np.std(f0v)), 1) if len(f0v) else 0,
        "energy_mean": round(float(np.mean(rms)), 4),
        "energy_var": round(float(np.std(rms)), 4),
        "pause_ratio": round(silent, 2),
        "tempo_strength": round(float(np.mean(onset)), 2),
    }


def _no_voice(f):
    """静音/误触/只有底噪就别喂 Whisper,它对着静音会编句子"""
    return (f.get("duration_s", 0) < GATE_MIN_DURATION or f.get("pause_ratio", 0) >= GATE_MAX_PAUSE
            or f.get("pitch_mean_hz", 0) < GATE_MIN_PITCH)


def _baseline(path):
    vals = collections.defaultdict(list)
    for e in _read_jsonl(path):
        for k, v in (e.get("features") or {}).items():
            vals[k].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)) or 1.0) for k, v in vals.items() if len(v) >= REL_MIN_SAMPLES}


def _relative(feats):
    """数字特征 → "音高比平时偏高"。绝对阈值对每个人都不一样,只有和自己比才有意义。
    返回 None 表示历史样本还不够,没有基线"""
    base = _mtime_cached("baseline", LOG, _baseline)
    if not base:
        return None
    out = []
    for f, (name, hi, lo) in REL_FEATURES.items():
        if f in feats and f in base:
            z = (feats[f] - base[f][0]) / base[f][1]
            if abs(z) > REL_STRONG:
                out.append(f"{name}比平时明显{hi if z > 0 else lo}")
            elif abs(z) > REL_MILD:
                out.append(f"{name}比平时偏{hi if z > 0 else lo}")
    return "、".join(out)


def _tone(feats):
    """给你的 AI 的"怎么说的"。和平时一样/还没基线就留空,不添噪音"""
    return _relative(feats) or ""


def _acoustic_probs(wav):
    if not MODEL_FILE.exists():
        return None
    try:
        return acoustic.predict(wav, MODEL_FILE)
    except Exception as e:
        print(f"[acoustic] {e}", flush=True)
        return None


# ── 判情感 ──
def _pick_fewshot(_):
    """从你的标注里挑一组覆盖各标签的例子(每个标签尽量 2 条)"""
    lab = {e["audio"]: e for e in _read_jsonl(LABELS)}
    logs = {e["audio"]: e for e in _read_jsonl(LOG) if e.get("audio")}
    cand = [a for a, e in lab.items() if e.get("tags") and not e.get("skip") and not e.get("text_wrong")
            and a in logs and 6 <= len(logs[a].get("text", "")) <= 60]
    need = collections.Counter({t: 2 for t in TAGS})
    pick = []
    for a in sorted(cand, key=lambda a: -sum(need[t] for t in lab[a]["tags"])):
        if len(pick) >= FEWSHOT_N:
            break
        if any(need[t] > 0 for t in lab[a]["tags"]):
            pick.append((logs[a], lab[a]["tags"]))
            for t in lab[a]["tags"]:
                need[t] -= 1
    return pick


def _judge_emotion(text, feats, acou=None):
    shots = _mtime_cached("fewshot", LABELS, _pick_fewshot)
    rel = _relative(feats)
    p = (f"判断{SPEAKER}这段语音当时的情绪。{SPEAKER}是在和{LISTENER}说话。\n"
         f"可选标签(选1-3个,最主要的放第一个): {TAGS}\n"
         "注意: 不要默认选最常见的那个。嘴上骂人可能是打闹/嗔怪也可能真生气;带哭腔、委屈时声音线索比文字更可信。\n")
    if shots:
        p += f"\n{SPEAKER}自己标过的例子(参考{SPEAKER}的习惯):\n" + "\n".join(
            f"- 「{e['text'][:60]}」({_relative(e.get('features') or {}) or '和平时差不多'}) → {tags}"
            for e, tags in shots) + "\n"
    p += (f"\n这段: 「{text}」\n时长{feats.get('duration_s')}秒；声音: "
          f"{(rel or '和平时差不多') if rel is not None else '暂无个人基线'}\n")
    if acou:
        hits = [f"{t}像 {acou[t]:.2f}" for t in ACOUSTIC_SHOW if acou.get(t, 0) >= ACOUSTIC_THRESHOLD]
        p += f"声学模型(专门按{SPEAKER}的声音训练,0-1): {'；'.join(hits) or '无明显信号'}\n"
    p += '只输出JSON: {"tags":["..."],"confidence":0到1,"hint":"一句话描述此刻状态"}'
    raw = _llm(p)
    s, e = raw.find("{"), raw.rfind("}")
    d = json.loads(raw[s:e + 1])
    tags = [t for t in d.get("tags") or [] if t in TAGS][:3]
    if not tags:
        raise ValueError(f"no valid tags: {raw[:100]}")
    return {"emotion": "、".join(tags), "tags": tags, "confidence": d.get("confidence", 0), "hint": d.get("hint", "")}


def _fire_webhook(entry):
    if not WEBHOOK_URL:
        return

    def run():
        try:
            req = urllib.request.Request(WEBHOOK_URL, data=json.dumps(entry, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


@app.on_event("startup")
async def _warm():
    """librosa 懒加载 + pyin 首调 numba 编译要好几秒,启动时先付掉"""
    def warm():
        try:
            import soundfile as sf
            with tempfile.TemporaryDirectory() as td:
                w = Path(td) / "w.wav"
                sf.write(str(w), (np.random.randn(16000) * 0.01).astype(np.float32), 16000)
                _acoustic_features(w)
                acoustic.extract(w)
        except Exception:
            pass
    threading.Thread(target=warm, daemon=True).start()


@app.post("/api/voice/upload")
async def upload(file: UploadFile = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = await file.read()
    clip_name = ""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("in" + (Path(file.filename or "a.webm").suffix or ".webm"))
        src.write_bytes(raw)
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav)],
                       capture_output=True, timeout=60)
        if not wav.exists():
            return JSONResponse({"error": "audio convert failed"}, status_code=400)
        feats = _acoustic_features(wav)
        if _no_voice(feats):
            return JSONResponse({"error": "没听清，再说一次？"}, status_code=422)
        with ThreadPoolExecutor(1) as ex:   # 声学分类器(CPU)和 Whisper(网络)并行
            fut = ex.submit(_acoustic_probs, wav)
            text, err = _whisper(wav)
            acou = fut.result()
        if text is None:
            return JSONResponse({"error": f"whisper failed: {err}"}, status_code=502)
        if not text:
            return JSONResponse({"error": "没听清，再说一次？"}, status_code=422)
        if KEEP_AUDIO:
            try:
                CLIPS.mkdir(parents=True, exist_ok=True)
                clip_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".mp3"
                subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-b:a", "64k",
                                str(CLIPS / clip_name)], capture_output=True, timeout=60)
                if not (CLIPS / clip_name).exists():
                    clip_name = ""
            except Exception:
                clip_name = ""
    try:
        emo = _judge_emotion(text, feats, acou)
    except Exception as e:
        print(f"[judge] {e}", flush=True)
        emo = {"emotion": "未知", "tags": [], "confidence": 0.0, "hint": "情感分析失败"}
    tone = _tone(feats)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "text": text,
             "emotion": emo["emotion"], "tags": emo["tags"], "confidence": emo.get("confidence", 0),
             "hint": emo.get("hint", ""), "tone": tone, "features": feats, "acoustic": acou, "audio": clip_name}
    with LOG.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _fire_webhook(entry)
    return {"text": text, "emotion": entry["emotion"], "tags": entry["tags"],
            "confidence": entry["confidence"], "hint": entry["hint"], "tone": tone}


@app.get("/api/voice/recent")
async def recent(n: int = 10):
    return _read_jsonl(LOG)[-n:]


@app.get("/api/voice/audio/{name}")
async def audio_clip(name: str):
    safe = Path(name).name
    fp = CLIPS / safe
    if not safe.endswith(".mp3") or not fp.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(fp, media_type="audio/mpeg")


# ── 标注页:听自己的语音,标当时真实的情绪(需要 KEEP_AUDIO=1) ──
@app.get("/label/", response_class=HTMLResponse)
async def label_page():
    return (Path(__file__).parent / "label.html").read_text().replace("__TAGS__", json.dumps(TAGS, ensure_ascii=False))


@app.get("/label/api/items")
async def label_items():
    lab = {e["audio"]: e for e in _read_jsonl(LABELS)}
    rows = []
    for e in _read_jsonl(LOG):
        a = e.get("audio")
        if not a or not (CLIPS / a).exists():
            continue
        acou = e.get("acoustic") or {}
        rows.append({"audio": a, "ts": e["ts"], "text": e.get("text", ""),
                     "dur": (e.get("features") or {}).get("duration_s", 0), "label": lab.get(a),
                     "model": {"emotion": e.get("emotion", ""), "hint": e.get("hint", ""),
                               "acoustic": sorted(([t, p] for t, p in acou.items() if p >= ACOUSTIC_THRESHOLD),
                                                  key=lambda x: -x[1])}})
    rows.reverse()  # 新的在前,近的记得清
    return rows


@app.post("/label/api/label")
async def label_save(req: Request):
    d = await req.json()
    a = Path(str(d.get("audio", ""))).name
    if not a.endswith(".mp3") or not (CLIPS / a).exists():
        return JSONResponse({"error": "bad audio"}, status_code=400)
    entry = {"audio": a, "tags": [str(t)[:20] for t in (d.get("tags") or [])][:3],
             "intensity": d.get("intensity"), "text_wrong": bool(d.get("text_wrong")),
             "breathy": bool(d.get("breathy")), "skip": bool(d.get("skip")),
             "note": str(d.get("note") or "")[:300], "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with LABELS.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/label/audio/{name}")
async def label_audio(name: str):
    return await audio_clip(name)


PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,user-scalable=no">
<title>her voice</title><style>
:root{--bg:#faf7f2;--fg:#3a3532;--accent:#c96f5e;--soft:#e8ded2}
@media(prefers-color-scheme:dark){:root{--bg:#1c1a18;--fg:#e8e2da;--accent:#d98873;--soft:#3a342e}}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);
font-family:Georgia,'Songti SC',serif;min-height:100vh;display:flex;flex-direction:column;
align-items:center;justify-content:center;gap:28px;padding:24px}
h1{font-size:1.3rem;font-weight:400;letter-spacing:.15em}
#btn{width:120px;height:120px;border-radius:50%;border:2px solid var(--accent);
background:transparent;color:var(--accent);font-size:1rem;font-family:inherit;
transition:all .2s;touch-action:none;-webkit-user-select:none;user-select:none}
#btn.rec{background:var(--accent);color:var(--bg);transform:scale(1.08);
box-shadow:0 0 0 12px color-mix(in srgb,var(--accent) 18%,transparent)}
#out{max-width:420px;width:100%;display:flex;flex-direction:column;gap:10px}
.card{background:var(--soft);border-radius:14px;padding:14px 16px;font-size:.92rem;line-height:1.55}
.emo{color:var(--accent);font-size:.8rem;letter-spacing:.08em}
#tip{font-size:.78rem;opacity:.55;letter-spacing:.05em}
a{color:var(--accent);font-size:.8rem;opacity:.7;letter-spacing:.1em;text-decoration:none}
</style></head><body>
<h1>her voice 🎙</h1>
<button id=btn>按住说话</button>
<div id=tip>说出来，让它听见你的语气</div>
<div id=out></div>
<a href="/label/">✧ 标注</a>
<script>
const btn=document.getElementById('btn'),out=document.getElementById('out'),tip=document.getElementById('tip');
let mr,chunks=[];
async function start(){
 try{const s=await navigator.mediaDevices.getUserMedia({audio:true});
 chunks=[];mr=new MediaRecorder(s);mr.ondataavailable=e=>chunks.push(e.data);
 mr.onstop=send;mr.start();btn.classList.add('rec');btn.textContent='松开发送';}
 catch(e){tip.textContent='需要麦克风权限';}
}
function stop(){if(mr&&mr.state!=='inactive'){mr.stop();mr.stream.getTracks().forEach(t=>t.stop());}
 btn.classList.remove('rec');btn.textContent='按住说话';}
async function send(){
 const blob=new Blob(chunks,{type:mr.mimeType||'audio/webm'});
 if(blob.size<1000){tip.textContent='太短了，再说一次';return;}
 tip.textContent='分析中…';
 const fd=new FormData();fd.append('file',blob,'voice.webm');
 try{const r=await fetch('/api/voice/upload',{method:'POST',body:fd});
 const d=await r.json();
 if(d.error){tip.textContent=d.error;return;}
 const c=document.createElement('div');c.className='card';
 const em=document.createElement('div');em.className='emo';em.textContent=d.emotion+(d.tone?' · '+d.tone:'')+' · '+(d.hint||'');
 const tx=document.createElement('div');tx.textContent=d.text;c.append(em,tx);
 out.prepend(c);tip.textContent='听到了 ♡';}
 catch(e){tip.textContent='网络出错，再试一次';}
}
btn.addEventListener('pointerdown',e=>{e.preventDefault();start();});
btn.addEventListener('pointerup',stop);btn.addEventListener('pointercancel',stop);
btn.addEventListener('pointerleave',()=>{if(mr&&mr.state==='recording')stop();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
