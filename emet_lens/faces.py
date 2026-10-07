import cv2
import numpy as np
from PIL import Image

_cascade = None


def crop_face(img: Image.Image, margin=0.3, size=224):
    """Return (face_crop, bbox or None). Falls back to center square if no face found."""
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rgb = img.convert("RGB")
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    w, h = rgb.size
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        m = int(max(fw, fh) * margin)
        box = (max(0, x - m), max(0, y - m), min(w, x + fw + m), min(h, y + fh + m))
        found = True
    else:
        s = min(w, h)
        box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        found = False
    crop = rgb.crop(box).resize((size, size), Image.BICUBIC)
    return crop, (box if found else None)


def prep_crop(img: Image.Image, size=224, quality=75):
    """Face crop normalised so pixel detectors cannot key on file properties (original size,
    aspect ratio, JPEG quality): one resample to size x size, then one common JPEG re-encode."""
    import io
    crop, box = crop_face(img, size=size)
    b = io.BytesIO()
    crop.save(b, "JPEG", quality=quality)
    b.seek(0)
    return Image.open(b).convert("RGB"), box
