"""Train the texture-conditioned noise-consistency detector (TCNC) on the manifest train split.

  python scripts/train_consistency.py
Features come from the FULL image (not the face crop), cached in cache/. Trains on clean + JPEG75 + downscale copies.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from emet_lens.manifest import load_manifest
from emet_lens.data import cached_features
from emet_lens.degrade import DEGRADATIONS
from emet_lens.detectors.consistency import ConsistencyDetector

tr = load_manifest("data/MANIFEST.csv").query("split == 'train'")
paths, y = list(tr.path), (tr.label == "fake").astype(int).values
d = ConsistencyDetector()
X = np.concatenate([cached_features(d, paths, v, None if v == "clean" else DEGRADATIONS[v], loader=lambda p, deg: (
    deg(Image.open(p).convert("RGB")) if deg else Image.open(p).convert("RGB"))) for v in ("clean", "jpeg75", "down0.5")])
os.makedirs("checkpoints", exist_ok=True)
d.fit(X, np.tile(y, 3))
print("trained TCNC on", len(X), "samples; face found in", f"{X[:, 9].mean():.0%}")
