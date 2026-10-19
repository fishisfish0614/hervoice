#!/usr/bin/env python3
"""hervoice 个人声学分类器训练 + 标注体检报告。

在 /label/ 标注一批之后运行:
    python train.py            # 只出报告
    python train.py --apply    # 确认后才写入模型(旧模型自动备份;新模型变差会拒绝替换)
报告内容:
  ① 线上判的情感 vs 你标的,对得上多少,判错的逐条列出
  ② 标了「转写不对」的条目和备注(调 WHISPER_PROMPT 用)
  ③ 「气音·悄悄话」攒了几条
  ④ 按天分组交叉验证 AUC,新旧模型对比
需要 scikit-learn(pip install -r requirements-train.txt)。至少要几十条标注、每个标签 ≥10 条才有意义。
为什么按天分组:同一晚的录音环境相同,随机切分会让模型"认出那一晚"而不是认出情绪,分数虚高。
"""
import collections
import json
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
import acoustic  # noqa: E402

warnings.filterwarnings("ignore")
DATA = Path(os.environ.get("HERVOICE_DATA", "./data"))
LABELS = DATA / "labels.jsonl"
LOG = DATA / "voice_logs.jsonl"
CLIPS = DATA / "clips"
MODEL = DATA / "acoustic_model.json"
CACHE = DATA / "feats.npz"
MIN_POS = 10          # 一个标签至少10条正例才训
C = 0.05              # 逻辑回归正则(按天分组交叉验证选的)
TOLERANCE = 0.02      # 新模型平均AUC比旧的低不超过这么多才换


def load_jsonl(p, key):
    out = {}
    for l in p.read_text().splitlines():
        try:
            e = json.loads(l)
            out[e[key]] = e
        except Exception:
            pass
    return out


def features(audios):
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    miss = [a for a in audios if a[:-4] not in cache and (CLIPS / a).exists()]
    for i, a in enumerate(miss):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(CLIPS / a), "-ar", "16000", "-ac", "1", str(w)])
            cache[a[:-4]] = acoustic.extract(w)
        print(f"\r  提特征 {i + 1}/{len(miss)}", end="", flush=True)
    if miss:
        print()
        np.savez(CACHE, **cache)
    return cache


def fit(X, Y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    ms = [LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), Y[:, j])
          for j in range(Y.shape[1])]
    return sc, ms


def grouped_cv(X, Y, groups, reps=5):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    P = np.zeros(Y.shape)
    for r in range(reps):
        for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(X, groups=groups):
            sc, ms = fit(X[tr], Y[tr])
            for j, m in enumerate(ms):
                P[te, j] += m.predict_proba(sc.transform(X[te]))[:, 1] / reps
    return {t: round(float(roc_auc_score(Y[:, j], P[:, j])), 3) for j, t in enumerate(TAGS) if 0 < Y[:, j].sum() < len(Y)}


def main(apply):
    global TAGS
    lab = load_jsonl(LABELS, "audio")
    if not LABELS.exists():
        sys.exit(f"还没有标注: {LABELS}")
    logs = {a: e for a, e in load_jsonl(LOG, "audio").items() if a}
    old = json.loads(MODEL.read_text()) if MODEL.exists() else {}
    old_keys = set(old.get("trained_keys", []))
    since = old.get("trained_at", "")
    newly = [a for a, e in lab.items() if e.get("at", "") > since] if since else list(lab)

    print(f"== 标注: 共 {len(lab)} 条, 上次训练({since or '无'})以来新标 {len(newly)} 条; "
          f"语音总数 {len(logs)}, 还没标 {len([a for a in logs if a not in lab])} 条")

    # ① 线上判得对不对(只看新版标签的条目)
    rows = [(a, lab[a]["tags"], logs[a].get("tags") or []) for a in newly
            if a in logs and lab[a]["tags"] and logs[a].get("tags")]
    if rows:
        top = np.mean([s[0] in h for _, h, s in rows])
        anyh = np.mean([bool(set(h) & set(s)) for _, h, s in rows])
        print(f"\n① 判的情感 vs 你标的({len(rows)}条): 首选命中 {top:.0%} | 任一命中 {anyh:.0%}")
        for a, h, s in rows:
            if not set(h) & set(s):
                print(f"   ✗ {a[4:13]} 你:{'、'.join(h)} | 系统:{'、'.join(s)} | 「{logs[a]['text'][:30]}」")

    # ② 转写不对
    wrong = [(a, e) for a, e in lab.items() if e.get("text_wrong") and a in newly]
    print(f"\n② 新标的「转写不对」{len(wrong)} 条:")
    for a, e in wrong:
        print(f"   {a[4:13]} 转写:「{logs.get(a, {}).get('text', '')[:30]}」 备注:{e.get('note', '')}")

    # ③ 气音
    print(f"\n③ 气音·悄悄话 共 {sum(1 for e in lab.values() if e.get('breathy'))} 条(攒到~10条可以做气音描述)")

    # ④ 重训
    keys = [a for a, e in lab.items() if not e.get("skip") and e.get("tags") and (CLIPS / a).exists()]
    F = features(keys)
    keys = [a for a in keys if a[:-4] in F]
    TAGS = [t for t, n in collections.Counter(t for a in keys for t in lab[a]["tags"]).most_common() if n >= MIN_POS]
    X = np.array([F[a[:-4]] for a in keys])
    Y = np.array([[t in lab[a]["tags"] for t in TAGS] for a in keys], int)
    cv = grouped_cv(X, Y, [a[:8] for a in keys])
    old_cv = old.get("cv", {})
    print(f"\n④ 按天分组交叉验证 AUC(n={len(keys)}):")
    for t in TAGS:
        o = old_cv.get(t)
        print(f"   {t:8} {cv.get(t, float('nan')):.2f}" + (f"  (旧 {o:.2f})" if o else "  (新标签)"))
    new_mean = float(np.mean(list(cv.values())))
    old_mean = float(np.mean([old_cv[t] for t in old_cv])) if old_cv else None
    print(f"   平均 {new_mean:.3f}" + (f" vs 旧 {old_mean:.3f}" if old_mean else " (旧模型无记录)"))

    # 旧模型在它没见过的新标注上的表现
    fresh = [a for a in keys if old_keys and a not in old_keys]
    if old and len(fresh) >= 5:
        from sklearn.metrics import roc_auc_score
        ps = [acoustic_feats_predict_cached(old, F[a[:-4]]) for a in fresh]
        print(f"   旧模型在 {len(fresh)} 条新标注上:", " ".join(
            f"{t}{roc_auc_score([t in lab[a]['tags'] for a in fresh], [p.get(t, 0) for p in ps]):.2f}"
            for t in old["tags"] if 0 < sum(t in lab[a]["tags"] for a in fresh) < len(fresh)))

    ok = old_mean is None or new_mean >= old_mean - TOLERANCE
    print(f"\n结论: {'新模型可以换' if ok else '新模型变差了,不换'}" + ("" if apply else "  (报告模式,加 --apply 才写入)"))
    if not (apply and ok):
        return
    sc, ms = fit(X, Y)
    if MODEL.exists():
        MODEL.rename(MODEL.with_name(MODEL.name + ".bak-" + datetime.now().strftime("%Y%m%d%H%M")))
    MODEL.write_text(json.dumps({
        "_doc": "hervoice 个人声学分类器权重(逻辑回归,每标签一行)。由 train.py 生成,别手改。",
        "feature_version": acoustic.FEATURE_VERSION, "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n": len(keys), "tags": TAGS, "cv": cv, "trained_keys": keys,
        "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
        "coef": [m.coef_[0].round(6).tolist() for m in ms],
        "intercept": [round(float(m.intercept_[0]), 6) for m in ms]}, ensure_ascii=False))
    print(f"已写入 {MODEL}(服务按 mtime 自动加载,不用重启)")


def acoustic_feats_predict_cached(m, x):
    z = np.array(m["coef"]) @ ((x - np.array(m["mean"])) / np.array(m["scale"])) + np.array(m["intercept"])
    return dict(zip(m["tags"], 1 / (1 + np.exp(-z))))


TAGS = []
if __name__ == "__main__":
    main("--apply" in sys.argv)
