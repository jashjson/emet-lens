"""Fetch real in-the-wild faces (FFHQ-256, bitmind/ffhq-256) and StyleGAN fakes (34data/140k-real-fake-faces-fake)
by reading only a few Parquet row groups over HTTP range requests (~100 MB total).

  python scripts/fetch_ffhq_stylegan.py --ffhq-groups 6 --stylegan-groups 3 --out data
"""
import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq
from PIL import Image
from fetch_dff import RangeFile

FFHQ = "https://huggingface.co/datasets/bitmind/ffhq-256/resolve/main/data/train-{:05d}-of-00016.parquet"
SG = "https://huggingface.co/datasets/34data/140k-real-fake-faces-fake/resolve/main/data.parquet"


def dump_group(url, rg, prefix, out_dir, keep):
    pf = pq.ParquetFile(RangeFile(url, tail=2_000_000))
    col = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
    n = 0
    for i, cell in enumerate(col[:keep]):
        dst = os.path.join(out_dir, f"{prefix}{rg:03d}x{i:03d}.jpg")
        if os.path.exists(dst):
            continue
        try:
            Image.open(io.BytesIO(cell["bytes"] if isinstance(cell, dict) else cell)).convert("RGB").save(dst, quality=95)  # uniform container
            n += 1
        except Exception:
            pass
    print(prefix, "row group", rg, "saved", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffhq-groups", type=int, default=6)
    ap.add_argument("--stylegan-groups", type=int, default=3)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    fd, sd = os.path.join(a.out, "ffhq"), os.path.join(a.out, "stylegan")
    os.makedirs(fd, exist_ok=True); os.makedirs(sd, exist_ok=True)
    jobs = []
    for k in range(a.ffhq_groups):  # spread over different shards and positions for variety
        jobs.append((FFHQ.format(k), (k * 7) % 44, "ffhq", fd, 100))
    for k in range(a.stylegan_groups):
        jobs.append((SG, (k * 13) % 40, "sg", sd, 200))
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: dump_group(*j), jobs))
    print("done")
