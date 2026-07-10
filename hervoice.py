#!/usr/bin/env python3
"""hervoice · 把语气变成 AI 能读懂的东西

录音 → Whisper 转写 + librosa 声学特征 → LLM 综合判情感 → 记日志 → 触发你自己的回调

核心引擎，与任何具体 AI 助手解耦。她说了什么、怎么说的，都交给你接的 AI。

配置见 .env（复制 .env.example）。隐私默认：音频阅后即焚，KEEP_AUDIO=1 才留存。
"""
import json
import os
import subprocess
import tempfile
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ── 配置（全部走环境变量，不硬编码任何密钥）──
DATA_DIR = Path(os.environ.get("HERVOICE_DATA", "./data"))
LOG = DATA_DIR / "voice_logs.jsonl"
CLIPS = DATA_DIR / "clips"
KEEP_AUDIO = os.environ.get("KEEP_AUDIO", "0") == "1"   # 默认阅后即焚

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "zh")
# 情感分析出结果后 POST 到这个 URL（你自己的 AI 助手/心跳/通知）。留空则不回调。
WEBHOOK_URL = os.environ.get("HERVOICE_WEBHOOK", "")

EMOTIONS = ["happy", "sad", "angry", "tired", "tender", "excited", "anxious", "neutral"]

app = FastAPI()


def _llm(prompt, max_tokens=200):
    body = json.dumps({"model": LLM_MODEL, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{LLM_BASE}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {LLM_KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _whisper(wav_path):
    if not GROQ_KEY:
        return None, "GROQ_API_KEY not set"
    r = subprocess.run(["curl", "-s", "-m", "60",
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        "-H", f"Authorization: Bearer {GROQ_KEY}",
                        "-F", f"file=@{wav_path}", "-F", f"model={WHISPER_MODEL}",
                        "-F", f"language={WHISPER_LANG}"],
                       capture_output=True, text=True, timeout=70)
    try:
        return json.loads(r.stdout).get("text", "").strip(), None
    except Exception:
        return None, r.stdout[:200]


def _acoustic_features(wav_path):
    """轻量声学特征：音高/能量/停顿/语速感 — 给 LLM 当'怎么说的'线索"""
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
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


def _judge_emotion(text, feats):
    prompt = (
        f"分析一段语音的情感。\n说话内容:「{text}」\n"
        f"声学特征: {json.dumps(feats, ensure_ascii=False)}"
        f"(pitch高+var大=激动; energy低+pause多=低落/疲惫; pitch上扬短句=撒娇可能)\n"
        f"综合'说了什么'和'怎么说的'，从{EMOTIONS}中选1个最贴切的，"
        f'只输出JSON: {{"emotion":"...","confidence":0.0到1.0,"hint":"一句话描述此刻状态"}}'
    )
    raw = _llm(prompt)
    s, e = raw.find("{"), raw.rfind("}")
    return json.loads(raw[s:e + 1])


def _fire_webhook(entry):
    if not WEBHOOK_URL:
        return

    def run():
        try:
            body = json.dumps(entry, ensure_ascii=False).encode()
            req = urllib.request.Request(WEBHOOK_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


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
        text, err = _whisper(wav)
        if text is None:
            return JSONResponse({"error": f"whisper failed: {err}"}, status_code=502)
        feats = _acoustic_features(wav)
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
        emo = _judge_emotion(text, feats)
    except Exception:
        emo = {"emotion": "neutral", "confidence": 0.0, "hint": "emotion analysis failed"}
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "text": text,
             "emotion": emo.get("emotion", "neutral"), "confidence": emo.get("confidence", 0),
             "hint": emo.get("hint", ""), "features": feats, "audio": clip_name}
    with LOG.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _fire_webhook(entry)
    return {"text": text, "emotion": entry["emotion"],
            "confidence": entry["confidence"], "hint": entry["hint"]}


@app.get("/api/voice/recent")
async def recent(n: int = 10):
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines()[-n:]]


@app.get("/api/voice/audio/{name}")
async def audio_clip(name: str):
    from fastapi.responses import FileResponse
    safe = Path(name).name
    fp = CLIPS / safe
    if not safe.endswith(".mp3") or not fp.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(fp, media_type="audio/mpeg")


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
</style></head><body>
<h1>her voice 🎙</h1>
<button id=btn>按住说话</button>
<div id=tip>说出来，让它听见你的语气</div>
<div id=out></div>
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
 if(d.error){tip.textContent='出错了: '+d.error;return;}
 const c=document.createElement('div');c.className='card';
 c.innerHTML='<div class=emo>'+d.emotion+' · '+(d.hint||'')+'</div><div>'+d.text+'</div>';
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
