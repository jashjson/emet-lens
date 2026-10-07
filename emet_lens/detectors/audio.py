import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..base import Detector, Signal
from ..device import get_device

W2V_ID = "facebook/wav2vec2-base"
SR = 16000
WIN = 3.0


def load_audio(path, sr=SR):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


class AudioDetector(Detector):
    """Frozen wav2vec2 mean-pooled hidden states (mid layer) + logistic regression, scored in 3 s windows."""
    name = "audio"
    needs_training = True

    def __init__(self, ckpt="checkpoints/audio.joblib", layer=6):
        self.ckpt, self.layer, self.clf, self._model = ckpt, layer, None, None

    def _load_model(self):
        if self._model is None:
            from transformers import Wav2Vec2Model
            self.device = get_device()
            self._model = Wav2Vec2Model.from_pretrained(W2V_ID).to(self.device).eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        self._load_model()
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        x = torch.from_numpy(wav).float()[None].to(self.device)
        hs = self._model(x, output_hidden_states=True).hidden_states[self.layer]
        return torch.cat([hs.mean(1), hs.std(1)], dim=-1)[0].cpu().numpy()

    def windows(self, wav):
        n = int(WIN * SR)
        if len(wav) <= n:
            return [(0, wav)]
        return [(s, wav[s:s + n]) for s in range(0, len(wav) - n // 2, n) if len(wav[s:s + n]) > n // 2]

    def fit(self, X, y):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X, y)
        joblib.dump(self.clf, self.ckpt)

    def load(self):
        if self.clf is None:
            self.clf = joblib.load(self.ckpt)

    def predict_audio(self, wav: np.ndarray, thresh=0.6) -> Signal:
        self.load()
        wins = self.windows(wav)
        scores = [float(self.clf.predict_proba(self.embed(w)[None])[0, 1]) for _, w in wins]
        flagged = [(s / SR, s / SR + len(w) / SR) for (s, w), p in zip(wins, scores) if p > thresh]
        score = float(np.mean(scores))
        if flagged:
            rng = ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in flagged)
            finding = f"Voice sounds synthetic in {rng}."
        else:
            finding = "Voice characteristics are consistent with natural speech."
        sig = Signal(self.name, score, finding, time_ranges=flagged)
        sig.window_scores = scores
        return sig

    def predict_file(self, path):
        return self.predict_audio(load_audio(path))
