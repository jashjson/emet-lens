import hashlib
import os
import numpy as np
from PIL import Image
from .faces import prep_crop


def load_crop(path, degrade=None):
    img = Image.open(path).convert("RGB")
    if degrade:
        img = degrade(img)
    return prep_crop(img)[0]


def cached_features(det, paths, tag="clean", degrade=None, cache_dir="cache"):
    """Per-image features, cached on disk by (detector, tag, path)."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for p in paths:
        key = hashlib.md5(f"{det.name}|{tag}|{os.path.abspath(p)}|{os.path.getmtime(p)}".encode()).hexdigest()
        f = os.path.join(cache_dir, key + ".npy")
        if os.path.exists(f):
            out.append(np.load(f))
        else:
            v = det.features(load_crop(p, degrade))
            np.save(f, v)
            out.append(v)
    return np.stack(out)
