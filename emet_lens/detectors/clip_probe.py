import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from ..base import Detector, Signal
from ..device import get_device

CLIP_ID = "openai/clip-vit-base-patch32"


class CLIPProbe(Detector):
    """Frozen CLIP image embeddings + logistic regression."""
    name = "clip_probe"
    needs_training = True

    def __init__(self, ckpt="checkpoints/clip_probe.joblib"):
        self.ckpt = ckpt
        self.clf = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self.device = get_device()
            self._model = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
            self._proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def features_batch(self, imgs):
        self._load_model()
        inp = self._proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt").to(self.device)
        f = self._model.get_image_features(**inp)
        if not torch.is_tensor(f):  # newer transformers return an output object
            f = f.pooler_output
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy()

    def features(self, img):
        return self.features_batch([img])[0]

    def fit(self, X, y):
        self.clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
        joblib.dump(self.clf, self.ckpt)

    def load(self):
        if self.clf is None:
            self.clf = joblib.load(self.ckpt)

    def predict(self, img):
        self.load()
        p = float(self.clf.predict_proba(self.features(img)[None])[0, 1])
        f = ("Overall appearance resembles AI-generated or manipulated images." if p > 0.6
             else "Overall appearance is consistent with real photos.")
        return Signal(self.name, p, f)
