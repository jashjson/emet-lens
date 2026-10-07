"""Build data/MANIFEST.csv from folders; split by source_id.

  python scripts/build_manifest.py --real data/ffhq data/own_real --fake stylegan=data/stylegan diffusion=data/diffusion
source_id = file stem up to '__' (frames <video>__f03.jpg share a source; <personid>__... shares a person).
Types given in --holdout are forced into the test split (unseen fake types).
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from emet_lens.manifest import assign_splits

EXT = {".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3", ".flac"}


def _ok(path):
    if path.lower().endswith((".wav", ".mp3", ".flac")):
        return True
    try:
        from PIL import Image
        Image.open(path).verify()
        return True
    except Exception:
        print("skipping unreadable", path)
        return False


def rows(folder, label, typ):
    for r, _, fs in os.walk(folder):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in EXT and _ok(os.path.join(r, f)):
                stem = os.path.splitext(f)[0].split("__")[0]
                yield dict(path=os.path.join(r, f), label=label, type=typ, source_id=stem)  # no type prefix: a real image and its derived fakes must share a split


ap = argparse.ArgumentParser()
ap.add_argument("--real", nargs="*", default=[])
ap.add_argument("--fake", nargs="*", default=[], help="type=folder")
ap.add_argument("--holdout", nargs="*", default=[], help="fake types used only for testing")
ap.add_argument("--out", default="data/MANIFEST.csv")
a = ap.parse_args()
data = [r for d in a.real for r in rows(d, "real", "real")]
for spec in a.fake:
    t, d = spec.split("=")
    data += list(rows(d, "fake", t))
df = assign_splits(pd.DataFrame(data))
# Held-out fake types are test-only AND derived only from test-split sources (no source overlap with training).
test_sources = set(df[df.split == "test"].source_id)
df = df[~(df.type.isin(a.holdout) & ~df.source_id.isin(test_sources))].copy()
df.loc[df.type.isin(a.holdout), "split"] = "test"
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(df.groupby(["type", "split"]).size())
