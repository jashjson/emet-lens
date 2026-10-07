import numpy as np
from PIL import Image

# IJG standard luminance table (quality 50)
_STD_LUM = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float32)


def jpeg_quality(path):
    """Estimated IJG quality of a JPEG file, or None if not a JPEG."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            t = np.array(list(im.quantization[0]), dtype=np.float32)
    except Exception:
        return None
    # PIL may return the table in zig-zag order; the mean is order independent.
    scale = 100.0 * t.mean() / _STD_LUM.mean()
    return float((200 - scale) / 2 if scale <= 200 else 5000 / scale)
