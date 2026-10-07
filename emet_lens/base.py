from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class Signal:
    name: str
    score_fake: float          # 0 = real, 1 = fake
    finding: str
    heatmap: Optional[np.ndarray] = None   # HxW float in [0,1]
    time_ranges: list = field(default_factory=list)  # [(start_s, end_s)]
    decisive: bool = False     # self-declared provenance (e.g. AI-generator tag)
    reliable: bool = True      # False: shown as context only, excluded from the verdict

    def to_dict(self):
        d = {"name": self.name, "score_fake": round(float(self.score_fake), 4), "finding": self.finding}
        if not self.reliable:
            d["reliable"] = False
        if self.time_ranges:
            d["time_ranges"] = [[round(a, 2), round(b, 2)] for a, b in self.time_ranges]
        return d


class Detector:
    """Image detectors implement `predict(img)`; audio detectors override `predict_audio`."""
    name = "base"
    needs_training = False

    def predict(self, img: Image.Image) -> Signal:
        raise NotImplementedError

    def features(self, img: Image.Image) -> np.ndarray:
        raise NotImplementedError


class DummyDetector(Detector):
    name = "dummy"

    def predict(self, img):
        s = float(np.asarray(img.convert("L")).std() / 128.0)
        return Signal(self.name, min(1.0, s), "Dummy detector: score from pixel variance.")
