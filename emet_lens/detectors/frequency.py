import numpy as np
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..base import Detector, Signal

NBINS = 32


class FrequencyDetector(Detector):
    """Azimuthally averaged log power spectrum -> logistic regression. Catches upsampling artifacts."""
    name = "frequency"
    needs_training = True

    def __init__(self, ckpt="checkpoints/frequency.joblib"):
        self.ckpt = ckpt
        self.clf = None

    def features(self, img):
        g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
        g = (g - g.mean()) * np.hanning(256)[:, None] * np.hanning(256)[None, :]
        ps = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2 + 1e-8)
        yy, xx = np.indices(ps.shape)
        r = np.hypot(yy - 128, xx - 128)
        bins = np.minimum((r / 128 * NBINS).astype(int), NBINS - 1)
        prof = np.bincount(bins.ravel(), ps.ravel(), NBINS) / np.maximum(np.bincount(bins.ravel(), minlength=NBINS), 1)
        return prof - prof[0]

    def fit(self, X, y):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")).fit(X, y)
        joblib.dump(self.clf, self.ckpt)

    def load(self):
        if self.clf is None:
            self.clf = joblib.load(self.ckpt)

    def predict(self, img):
        self.load()
        p = float(self.clf.predict_proba(self.features(img)[None])[0, 1])
        if p > 0.6:
            f = "High-frequency spectrum shows periodic/upsampling patterns typical of generated images."
        else:
            f = "Frequency spectrum looks like that of camera-captured images."
        return Signal(self.name, p, f)
