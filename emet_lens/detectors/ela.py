import io
import numpy as np
from PIL import Image, ImageChops
from scipy.ndimage import uniform_filter
from ..base import Detector, Signal
from ..jpeg_info import jpeg_quality

MIN_QUALITY = 88  # below this the file was recompressed; ELA cannot see earlier edits


class ELADetector(Detector):
    """Error Level Analysis: re-save as JPEG, look at where error differs from the rest. No training."""
    name = "ela"

    def __init__(self, quality=90):
        self.quality = quality

    def _ela_map(self, img):
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=self.quality)
        buf.seek(0)
        diff = ImageChops.difference(rgb, Image.open(buf).convert("RGB"))
        return np.asarray(diff, dtype=np.float32).mean(axis=2)

    def predict(self, img, path=None):
        e = self._ela_map(img)
        smooth = uniform_filter(e, size=max(3, min(e.shape) // 24))
        med = np.median(smooth) + 1e-3
        ratio = smooth / med
        heat = np.clip((ratio - 1.0) / 3.0, 0, 1)
        # Score: how concentrated / extreme is the anomalous region relative to the rest.
        p99 = np.percentile(ratio, 99)
        score = float(1 / (1 + np.exp(-(p99 - 5.0) / 0.8)))  # centre above what ordinary photos reach (untuned)
        if score > 0.6:
            ys, xs = np.unravel_index(np.argmax(smooth), smooth.shape)
            where = f"around x={xs / e.shape[1]:.0%}, y={ys / e.shape[0]:.0%} of the image"
            finding = f"Compression error level is inconsistent {where}, which can indicate local editing."
        else:
            finding = "Compression error level is uniform; no sign of local editing."
        sig = Signal(self.name, score, finding, heatmap=heat)
        q = jpeg_quality(path) if path else None
        if path is not None and (q is None or q < MIN_QUALITY or min(img.size) < 256):
            why = (f"it has been recompressed (estimated JPEG quality ~{q:.0f}, typical of messaging apps)" if q is not None
                   else "it is not an original JPEG file" if q is None else "it is small")
            sig.reliable = False
            sig.finding = (f"Not used for the verdict: {why}, which hides editing traces. "
                           "The heatmap only shows where compression error is highest and is not evidence of editing.")
        elif score > 0.6:
            sig.finding = finding + " Treat as a lead, not proof."
        return sig
