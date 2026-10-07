"""Fetch a small subset of OpenRL/DeepFakeFace (apache-2.0) via HTTP range requests (no full 5 GB download).
Real = wiki; fakes = text2img, insight (face swap), inpainting. Files are named <personid>__<file>; the person id is the source_id (a real photo and its fakes share it).

  python scripts/fetch_dff.py --identities 200 --per-identity 2 --out data/dff
"""
import argparse
import io
import os
import random
import sys
import zipfile
import requests

BASE = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
IMG = (".jpg", ".jpeg", ".png")


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests; the zip tail (central directory) is cached."""
    def __init__(self, url, tail=16_000_000):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        self.url = r.url  # resolved CDN url
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.tail_start = max(0, self.size - tail)
        self.tail = self._get(self.tail_start, self.size - 1)

    def _get(self, a, b):
        for _ in range(4):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code in (200, 206):
                    return r.content
            except requests.RequestException:
                pass
        raise IOError("range request failed")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        if self.pos >= self.tail_start:
            data = self.tail[self.pos - self.tail_start:end - self.tail_start]
        else:
            data = self._get(self.pos, end - 1)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def listing(z):
    zf = zipfile.ZipFile(RangeFile(BASE + z + ".zip"))
    names = [n for n in zf.namelist() if n.lower().endswith(IMG)]
    return zf, names


def fetch_member(url, info, session):
    """Download one zip member by range request and inflate it (independent of the shared RangeFile)."""
    import struct
    import zlib
    span = 30 + len(info.filename.encode()) + 512 + info.compress_size  # local header + extra field slack
    for _ in range(4):
        try:
            r = session.get(url, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + span - 1}"}, timeout=120)
            if r.status_code in (200, 206):
                break
        except requests.RequestException:
            pass
    else:
        raise IOError("range request failed")
    b = r.content
    n, m = struct.unpack("<HH", b[26:30])
    data = b[30 + n + m: 30 + n + m + info.compress_size]
    return data if info.compress_type == 0 else zlib.decompress(data, -15)


if __name__ == "__main__":
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", type=int, default=200)
    ap.add_argument("--per-identity", type=int, default=1)
    ap.add_argument("--out", default="data/dff")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs("cache", exist_ok=True)
    idx_path = "cache/dff_index.pkl"
    if os.path.exists(idx_path):
        index = pickle.load(open(idx_path, "rb"))
    else:
        index = {}
        for z in ("wiki", "text2img", "insight", "inpainting"):
            rf = RangeFile(BASE + z + ".zip")
            zf = zipfile.ZipFile(rf)
            infos = {i.filename.split("/", 1)[1]: i for i in zf.infolist()
                     if i.filename.lower().endswith(IMG) and "/" in i.filename}
            index[z] = infos
            print(z, len(infos), flush=True)
        pickle.dump(index, open(idx_path, "wb"))
    common = set.intersection(*(set(t) for t in index.values()))
    by_id = {}
    for k in common:
        by_id.setdefault(os.path.basename(k).split("_")[0], []).append(k)  # person id -> source_id
    ids = sorted(by_id)
    random.Random(0).shuffle(ids)
    jobs = []
    for ident in ids[: a.identities]:
        for k in sorted(by_id[ident])[: a.per_identity]:
            for z in index:
                dst = os.path.join(a.out, z, ident + "__" + os.path.basename(k))
                if not os.path.exists(dst):
                    jobs.append((z, k, dst))
    for z in index:
        os.makedirs(os.path.join(a.out, z), exist_ok=True)
    print(len(jobs), "files to fetch", flush=True)
    url = {z: requests.get(BASE + z + ".zip", headers={"Range": "bytes=0-0"}, timeout=60).url for z in index}
    done = [0]

    def work(job):
        z, k, dst = job
        data = fetch_member(url[z], index[z][k], requests.Session())
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        done[0] += 1
        if done[0] % 40 == 0:
            print("fetched", done[0], "files", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    print("done ->", a.out)
