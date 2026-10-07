"""Train CLIP probe, frequency detector and judge on the manifest train split (features cached in cache/).

  python scripts/train_image.py [--augment]
--augment adds JPEG75/JPEG50/downscaled copies of every training image (robustness, Step 6).
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from emet_lens.manifest import load_manifest
from emet_lens.data import cached_features
from emet_lens.degrade import DEGRADATIONS
from emet_lens.detectors.clip_probe import CLIPProbe
from emet_lens.detectors.frequency import FrequencyDetector
from emet_lens.judge import Judge, IMAGE_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/MANIFEST.csv")
ap.add_argument("--augment", action="store_true")
a = ap.parse_args()
os.makedirs("checkpoints", exist_ok=True)
tr = load_manifest(a.manifest).query("split == 'train'")
paths, y, groups = list(tr.path), (tr.label == "fake").astype(int).values, tr.source_id.values
variants = ["clean"] + (["jpeg75", "jpeg50", "down0.5"] if a.augment else [])
clip, freq = CLIPProbe(), FrequencyDetector()
feats = {d.name: [] for d in (clip, freq)}
Y, G = [], []
for v in variants:
    for d in (clip, freq):
        feats[d.name].append(cached_features(d, paths, v, None if v == "clean" else DEGRADATIONS[v]))
    Y.append(y); G.append(groups)
X = {k: np.concatenate(v) for k, v in feats.items()}
Y, G = np.concatenate(Y), np.concatenate(G)
cv = GroupKFold(n_splits=5)
# Out-of-fold scores so the judge is not fit on scores the detectors have memorised.
oof = {}
for d in (clip, freq):
    d.fit(X[d.name], Y)  # final model
    from sklearn.base import clone
    base = LogisticRegression(max_iter=2000, class_weight="balanced") if d is clip else d.clf
    oof[d.name] = cross_val_predict(clone(base), X[d.name], Y, groups=G, cv=cv, method="predict_proba")[:, 1]
J = np.stack([oof["clip_probe"], oof["frequency"]], 1)
Judge(IMAGE_FEATURES).fit(J, Y)
print("trained on", len(Y), "samples; saved checkpoints/")
