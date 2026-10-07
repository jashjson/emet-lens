"""Evaluate a detector on the manifest test split.

Reports accuracy, AUC and false-positive rate on real content, split into seen vs unseen fake types
(seen = fake types present in the train split).

  python -m eval.run_eval --detector dummy
  python -m eval.run_eval --detector judge --degrade jpeg50
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from emet_lens.manifest import load_manifest


def _metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": len(y), "acc": float(((p >= thr) == y).mean())}
    out["auc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return out


def evaluate(df, score_fn, thr=0.5):
    """df: manifest rows (with split). score_fn(path)->p_fake. Returns dict of metric groups."""
    train_fake = set(df[(df.split == "train") & (df.label == "fake")]["type"])
    test = df[df.split == "test"].copy()
    test["p"] = [score_fn(p) for p in test.path]
    test["y"] = (test.label == "fake").astype(int)
    real = test[test.y == 0]
    res = {"fpr_real": float((real.p >= thr).mean()) if len(real) else float("nan"), "n_real": len(real)}
    for name, fakes in (("seen", test[(test.y == 1) & test.type.isin(train_fake)]),
                        ("unseen", test[(test.y == 1) & ~test.type.isin(train_fake)])):
        if len(fakes) and len(real):
            m = _metrics(np.r_[np.zeros(len(real)), np.ones(len(fakes))], np.r_[real.p, fakes.p], thr)
            m["tpr_fake"] = float((fakes.p >= thr).mean())
            m["n_fake"] = len(fakes)
        else:
            m = {"n": len(fakes), "n_fake": len(fakes)}
        res[name] = m
    res["by_type"] = {t: float((g.p >= thr).mean()) for t, g in test[test.y == 1].groupby("type")}
    return res


def print_results(name, res):
    print(f"== {name} ==  FPR on real: {res['fpr_real']:.3f} (n_real={res['n_real']})")
    for k in ("seen", "unseen"):
        m = res[k]
        if "auc" in m:
            print(f"  {k:7s} n_fake={m['n_fake']:4d} acc={m['acc']:.3f} auc={m['auc']:.3f} detect_rate={m['tpr_fake']:.3f}")
        else:
            print(f"  {k:7s} n_fake={m['n_fake']} (no data)")
    for t, r in res["by_type"].items():
        print(f"    detected {t}: {r:.3f}")


def make_scorer(name, degrade=None):
    from PIL import Image
    from emet_lens.base import DummyDetector
    from emet_lens.data import load_crop
    from emet_lens.degrade import DEGRADATIONS
    deg = DEGRADATIONS[degrade] if degrade else None
    if name == "dummy":
        d = DummyDetector()
        return lambda p: d.predict(load_crop(p, deg)).score_fake
    if name == "consistency":
        from emet_lens.detectors.consistency import ConsistencyDetector
        d = ConsistencyDetector()
        d.load()
        return lambda p: d.predict(deg(Image.open(p).convert("RGB")) if deg else Image.open(p)).score_fake
    if name in ("clip_probe", "frequency", "ela"):
        from emet_lens.detectors.clip_probe import CLIPProbe
        from emet_lens.detectors.frequency import FrequencyDetector
        from emet_lens.detectors.ela import ELADetector
        d = {"clip_probe": CLIPProbe, "frequency": FrequencyDetector, "ela": ELADetector}[name]()
        if d.needs_training:
            d.load()
        if name == "ela":
            return lambda p: d.predict(deg(Image.open(p).convert("RGB")) if deg else Image.open(p)).score_fake
        return lambda p: d.predict(load_crop(p, deg)).score_fake
    if name == "judge":
        from emet_lens.pipeline import default_detectors, run_image_detectors
        from emet_lens.judge import Judge
        ds, j = default_detectors(), Judge().load()
        def f(p):
            img = Image.open(p).convert("RGB")
            img = deg(img) if deg else img
            sigs, _ = run_image_detectors(img, ds)
            return j.fuse({s.name: s.score_fake for s in sigs})[0]
        return f
    if name == "audio":
        from emet_lens.detectors.audio import AudioDetector, load_audio
        d = AudioDetector()
        return lambda p: d.predict_audio(load_audio(p)).score_fake
    raise ValueError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dummy")
    ap.add_argument("--manifest", default="data/MANIFEST.csv")
    ap.add_argument("--degrade", default=None)
    a = ap.parse_args()
    print_results(f"{a.detector} [{a.degrade or 'clean'}]", evaluate(load_manifest(a.manifest), make_scorer(a.detector, a.degrade)))
