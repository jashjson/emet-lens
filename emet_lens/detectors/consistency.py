"""Texture-conditioned noise consistency (TCNC): does the face carry the same "camera fingerprint" as the rest of the image?

Idea. A real photo is written by one sensor and one pipeline, so fine noise and texture statistics depend on how detailed
the area is, and on nothing else. A swapped, inpainted or re-generated face comes from a different process, so its fine
statistics sit off the curve that the rest of the picture follows. The detector compares the image against itself, so it
needs no knowledge of any generator, and global resolution and compression largely cancel out of the comparison.

Steps. (1) high-pass residual of the picture; (2) six noise/texture statistics per 32 px patch (variance, kurtosis, spatial
correlation in three directions, JPEG-grid strength); (3) per image, fit each statistic as a smooth function of local detail
(least squares, quadratic) so that content differences are removed; (4) compare the face patches with the surrounding patches
in the leftover residuals. A second, face-free score flags outlier patches anywhere (robust fit), and doubles as the heat map.
A small gradient-boosted classifier (depth 3) turns the 10 numbers into a score. Fully synthetic images are internally consistent, so this
detector targets local manipulation; the CLIP probe covers whole-image synthesis.
"""
import cv2
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from ..base import Detector, Signal
from ..faces import crop_face

WIN, STRIDE, MAX_SIDE, K = 32, 16, 512, 6
EPS = 1e-6


def _box(x):
    return cv2.boxFilter(x, -1, (WIN, WIN), normalize=True, borderType=cv2.BORDER_REFLECT)


def patch_stats(gray):
    """gray: HxW float32. Returns (F [gh,gw,K], detail t [gh,gw]) sampled on a WIN/STRIDE grid."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    r = gray - blur
    m2 = np.maximum(_box(r * r), 0)
    m4 = np.maximum(_box(r ** 4), 0)
    ac = []
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        a, b = r[: r.shape[0] - dy, : r.shape[1] - dx], r[dy:, dx:]
        c = np.pad(a * b, ((0, dy), (0, dx)), mode="edge")
        ac.append(_box(c) / (m2 + EPS))
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    detail = np.log(_box(gx * gx + gy * gy) + 1.0)
    # JPEG 8x8 grid strength: gradient across block borders vs inside blocks
    h, w = gray.shape
    dh = np.pad(np.abs(np.diff(gray, axis=1)), ((0, 0), (0, 1)), mode="edge")
    dv = np.pad(np.abs(np.diff(gray, axis=0)), ((0, 1), (0, 0)), mode="edge")
    mb_h = (np.arange(w) % 8 == 7)[None, :].astype(np.float32) * np.ones((h, 1), np.float32)
    mb_v = (np.arange(h) % 8 == 7)[:, None].astype(np.float32) * np.ones((1, w), np.float32)
    blk = np.log((_box(dh * mb_h) / (_box(mb_h) + EPS) + 1) / (_box(dh * (1 - mb_h)) / (_box(1 - mb_h) + EPS) + 1)) \
        + np.log((_box(dv * mb_v) / (_box(mb_v) + EPS) + 1) / (_box(dv * (1 - mb_v)) / (_box(1 - mb_v) + EPS) + 1))
    maps = [np.log(m2 + EPS), np.log(m4 / (m2 * m2 + EPS) + EPS), ac[0], ac[1], ac[2], blk]
    s = slice(WIN // 2, None, STRIDE)
    return np.nan_to_num(np.stack([m[s, s] for m in maps], -1)), detail[s, s]


def _design(t):
    t = (t - t.mean()) / (t.std() + EPS)
    return np.stack([np.ones_like(t), t, t * t], 1)


def _fit_resid(F, t, keep=None):
    """Least-squares fit of every statistic against local detail; returns residuals / robust scale. keep: patches used to fit."""
    X = _design(t)
    k = np.ones(len(F), bool) if keep is None else keep
    beta = np.linalg.lstsq(X[k], F[k], rcond=None)[0]
    R = F - X @ beta
    mad = 1.4826 * np.median(np.abs(R[k] - np.median(R[k], 0)), 0) + EPS
    return R / mad


def _robust_anomaly(F, t):
    keep = np.ones(len(F), bool)
    for _ in range(3):  # drop the worst 20% and refit, so a manipulated region cannot drag the fit towards itself
        Z = _fit_resid(F, t, keep)
        a = np.sqrt((Z ** 2).mean(1))
        keep = a <= np.quantile(a, 0.8)
    return a


class ConsistencyDetector(Detector):
    name = "consistency"
    needs_training = True

    def __init__(self, ckpt="checkpoints/consistency.joblib"):
        self.ckpt = ckpt
        self.clf = None

    def analyse(self, img):
        """Returns (feature vector [10], anomaly grid [gh,gw] or None)."""
        rgb = img.convert("RGB")
        w0, h0 = rgb.size
        sc = min(1.0, MAX_SIDE / max(w0, h0))
        if sc < 1:
            rgb = rgb.resize((max(1, int(w0 * sc)), max(1, int(h0 * sc))), 3)  # LANCZOS-free area-like downscale
        gray = np.asarray(rgb.convert("L"), np.float32)
        if min(gray.shape) < 2 * WIN:
            return np.zeros(10, np.float32), None
        F, t = patch_stats(gray)
        gh, gw, _ = F.shape
        Fp, tp = F.reshape(-1, K), t.reshape(-1)
        anom = _robust_anomaly(Fp, tp)
        vec = np.zeros(10, np.float32)
        vec[7], vec[8] = np.quantile(anom, 0.95), np.sort(anom)[-max(1, len(anom) // 20):].mean()
        _, box = crop_face(rgb, margin=0.0)
        if box is not None:
            ys, xs = (np.arange(gh) * STRIDE + WIN // 2), (np.arange(gw) * STRIDE + WIN // 2)
            inside = ((ys[:, None] >= box[1]) & (ys[:, None] < box[3]) & (xs[None, :] >= box[0]) & (xs[None, :] < box[2])).reshape(-1)
            if inside.sum() >= 6 and (~inside).sum() >= 6:
                Z = _fit_resid(Fp, tp)
                d = Z[inside].mean(0) - Z[~inside].mean(0)
                vec[:6], vec[6], vec[9] = d, float(np.linalg.norm(d)), 1.0
        return vec, anom.reshape(gh, gw)

    def features(self, img):
        return self.analyse(img)[0]

    def fit(self, X, y):
        self.clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                  class_weight="balanced", random_state=0).fit(X, y)
        joblib.dump(self.clf, self.ckpt)

    def load(self):
        if self.clf is None:
            self.clf = joblib.load(self.ckpt)

    def predict(self, img):
        self.load()
        vec, grid = self.analyse(img)
        p = float(self.clf.predict_proba(vec[None])[0, 1])
        heat = None
        if grid is not None:
            heat = np.clip(grid / (np.quantile(grid, 0.5) * 4 + EPS), 0, 1)
        if vec[9] == 0:
            f = "No clear face found, so the face could not be compared with its surroundings."
        elif p > 0.6:
            f = "Noise and texture inside the face do not match the rest of the image, typical of a swapped or re-generated face."
        else:
            f = "Noise and texture are consistent between the face and its surroundings."
        return Signal(self.name, p, f, heatmap=heat, reliable=vec[9] == 1)
