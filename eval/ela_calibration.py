"""Calibrate ELA on local photos: real = photos (+ recompressed/downscaled copies), edited = spliced copies.
  python -m eval.ela_calibration ~/Downloads/*.jpeg
Prints score distributions; used to check ELA false positives on real, recompressed content."""
import io
import random
import sys
import numpy as np
from PIL import Image, ImageFilter
from emet_lens.degrade import jpeg, downscale
from emet_lens.detectors.ela import ELADetector


def splice(base, donor, rng):
    w, h = base.size
    pw, ph = rng.randint(w // 8, w // 4), rng.randint(h // 8, h // 4)
    dx, dy = rng.randint(0, donor.size[0] - pw), rng.randint(0, donor.size[1] - ph)
    patch = jpeg(donor.crop((dx, dy, dx + pw, dy + ph)), rng.choice([60, 70, 80]))  # different compression history
    x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
    out = base.copy(); out.paste(patch, (x, y))
    return out


def build(paths, seed=0):
    rng = random.Random(seed)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    real, fake = [], []
    for i, im in enumerate(imgs):
        for tag, f in (("orig", lambda x: x), ("q75", lambda x: jpeg(x, 75)), ("down", lambda x: downscale(x, 0.5))):
            real.append((f"{i}:{tag}", f(im)))
            for k in range(3):
                e = splice(im, imgs[(i + 1 + k) % len(imgs)], rng)
                fake.append((f"{i}:{tag}:splice{k}", f(jpeg(e, 90))))  # re-save after edit, as an app would
    return real, fake


def report(det, real, fake):
    r = np.array([det.predict(im).score_fake for _, im in real])
    f = np.array([det.predict(im).score_fake for _, im in fake])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.r_[np.zeros(len(r)), np.ones(len(f))], np.r_[r, f])
    print(f"real n={len(r)} mean={r.mean():.2f} max={r.max():.2f} | edited n={len(f)} mean={f.mean():.2f} min={f.min():.2f} | AUC={auc:.2f}")
    print(f"  at thr 0.6: FPR real={np.mean(r >= .6):.2f} detect edited={np.mean(f >= .6):.2f}")
    return r, f


if __name__ == "__main__":
    real, fake = build(sys.argv[1:])
    report(ELADetector(), real, fake)
