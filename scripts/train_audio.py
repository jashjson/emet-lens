"""Train audio detector from a manifest of real/spoof audio (e.g. ASVspoof 2019 LA subset). Up to 3 windows/file.
Heavy: run on Colab if the Mac is slow; embeddings are cached in cache/."""
import argparse
import hashlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from emet_lens.manifest import load_manifest
from emet_lens.detectors.audio import AudioDetector, load_audio

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/MANIFEST_audio.csv")
ap.add_argument("--max-windows", type=int, default=3)
a = ap.parse_args()
os.makedirs("cache", exist_ok=True)
det = AudioDetector()
tr = load_manifest(a.manifest).query("split == 'train'")
X, y = [], []
for p, lab in zip(tr.path, tr.label):
    f = os.path.join("cache", "aud_" + hashlib.md5(os.path.abspath(p).encode()).hexdigest() + ".npy")
    if os.path.exists(f):
        E = np.load(f)
    else:
        E = np.stack([det.embed(w) for _, w in det.windows(load_audio(p))[: a.max_windows]])
        np.save(f, E)
    X += list(E); y += [int(lab == "fake")] * len(E)
det.fit(np.stack(X), np.array(y))
print("trained audio detector on", len(y), "windows")
