"""hervoice 个人声学分类器。

librosa 68 维特征(MFCC/pyin 音高/能量/频谱) → 每个标签一个逻辑回归 → 各标签"像不像"的概率。
权重由 train.py 用你自己的标注训练,存成 JSON;这里只做 numpy 推理,不依赖 scikit-learn、不常驻模型。
**训练和推理必须用同一个特征函数、同一个 librosa 版本**,改了特征就要 bump FEATURE_VERSION 并重训。
"""
import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = 1


def extract(path):
    import librosa
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    if len(y) < sr * 0.3:
        y = np.pad(y, (0, int(sr * 0.3) - len(y)))
    out = []
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    d = librosa.feature.delta(mf)
    out += list(mf.mean(1)) + list(mf.std(1)) + list(d.std(1))
    for f in (librosa.feature.spectral_centroid(S=S, sr=sr), librosa.feature.spectral_bandwidth(S=S, sr=sr),
              librosa.feature.spectral_rolloff(S=S, sr=sr), librosa.feature.spectral_flatness(S=S),
              librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)):
        out += [f.mean(), f.std()]
    out += list(librosa.feature.spectral_contrast(S=S, sr=sr).mean(1))
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    db = librosa.amplitude_to_db(rms + 1e-6)
    out += [db.mean(), db.std(), np.percentile(db, 90) - np.percentile(db, 10),
            float(np.mean(rms < np.percentile(rms, 20) * 1.5))]
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=600, sr=sr, frame_length=1024, hop_length=160)
    v = f0[~np.isnan(f0)]
    if len(v) > 5:
        st = 12 * np.log2(v / 100)  # 半音,和设备音量无关
        slope = np.polyfit(np.arange(len(st)), st, 1)[0]
        out += [st.mean(), st.std(), np.percentile(st, 90) - np.percentile(st, 10), slope,
                np.mean(np.abs(np.diff(st))), len(v) / len(f0), st[-max(1, len(st) // 5):].mean() - st.mean()]
    else:
        out += [0, 0, 0, 0, 0, 0, 0]
    out += [len(y) / sr]
    return np.nan_to_num(np.array(out, float))


_model = {}


def predict(path, model_file):
    """返回 {标签: 概率}。模型文件按mtime缓存,重训后自动生效"""
    p = Path(model_file)
    mt = p.stat().st_mtime
    if _model.get("mt") != mt:
        m = json.loads(p.read_text())
        if m["feature_version"] != FEATURE_VERSION:
            raise RuntimeError("acoustic model feature_version mismatch, 要重训")
        _model.update(mt=mt, m=m)
    m = _model["m"]
    x = (extract(path) - np.array(m["mean"])) / np.array(m["scale"])
    z = np.array(m["coef"]) @ x + np.array(m["intercept"])
    return dict(zip(m["tags"], (1 / (1 + np.exp(-z))).round(3).tolist()))
