"""Extract N frames per video into <out>/<video>__fNN.jpg (so frames share a source_id)."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emet_lens.pipeline import sample_frames, VID_EXT

ap = argparse.ArgumentParser()
ap.add_argument("videos")
ap.add_argument("out")
ap.add_argument("-n", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for f in sorted(os.listdir(a.videos)):
    if os.path.splitext(f)[1].lower() in VID_EXT:
        for k, (_, _, im) in enumerate(sample_frames(os.path.join(a.videos, f), a.n)):
            im.save(os.path.join(a.out, f"{os.path.splitext(f)[0]}__f{k:02d}.jpg"), quality=95)
