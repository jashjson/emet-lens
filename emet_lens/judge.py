import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

IMAGE_FEATURES = ["clip_probe", "frequency"]  # ELA is context only, not fused


class Judge:
    """Logistic-regression fusion over detector scores (score_fake). Falls back to a plain mean if untrained."""

    def __init__(self, features=IMAGE_FEATURES, ckpt="checkpoints/judge.joblib",
                 disagree=0.6, margin=0.12, min_side=64):
        self.features, self.ckpt = features, ckpt
        self.disagree, self.margin, self.min_side = disagree, margin, min_side
        self.clf = None
        self.shift = 0.0

    def fit(self, X, y, target_fpr=0.10):
        """Fit on out-of-fold detector scores. Classes are balanced (real photos are far more common than
        fakes in practice, so the training prior must not push real images toward 'fake'), and the decision
        point is set so that about `target_fpr` of real training images would be called fake."""
        X, y = np.asarray(X), np.asarray(y)
        self.clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(X, y)
        p_real = self.clf.predict_proba(X[y == 0])[:, 1]
        thr = float(np.quantile(p_real, 1 - target_fpr))
        self.shift = -float(np.log((thr + 1e-6) / (1 - thr + 1e-6)))
        joblib.dump({"clf": self.clf, "shift": self.shift}, self.ckpt)

    def load(self):
        try:
            d = joblib.load(self.ckpt)
            self.clf, self.shift = d["clf"], d["shift"]
        except FileNotFoundError:
            self.clf = None
        return self

    def fuse(self, scores: dict, min_side=None, extra_inconsistency=0.0, meta_shift=0.0, declared_ai=False):
        """scores: {detector: score_fake} of reliable pixel detectors. meta_shift: small logit nudge from file
        metadata. declared_ai: file metadata explicitly declares AI generation. Returns (p_fake, verdict, low_confidence_reason).
        Verdict is always likely_real/likely_fake unless no detector is available at all."""
        avail = {k: v for k, v in scores.items() if k in self.features}
        if not avail:
            if declared_ai:
                return 0.97, "likely_fake", ""
            return 0.5, "inconclusive", "no trained pixel detectors loaded (run scripts/train_image.py)"
        if self.clf is not None and len(avail) == len(self.features):
            p = float(self.clf.predict_proba(np.array([[avail[k] for k in self.features]]))[0, 1])
            meta_shift = meta_shift + self.shift  # recentre so the 0.5 line sits at the target-FPR threshold
        else:
            p = float(np.mean(list(avail.values())))
        p = float(1 / (1 + np.exp(-(np.log((p + 1e-6) / (1 - p + 1e-6)) + meta_shift))))
        if declared_ai:
            return max(p, 0.97), "likely_fake", ""
        spread = max(avail.values()) - min(avail.values())
        reason = ""
        if min_side is not None and min_side < self.min_side:
            reason = f"input is very low resolution ({min_side}px)"
        elif spread > self.disagree:
            reason = "detectors strongly disagree"
        elif extra_inconsistency > 0.3:
            reason = "scores vary strongly between frames"
        elif abs(p - 0.5) < self.margin:
            reason = "score is too close to 50%"
        # Always a call. `reason` explains why confidence is low; it no longer suppresses the verdict.
        verdict = "likely_fake" if p >= 0.5 else "likely_real"
        return p, verdict, reason
