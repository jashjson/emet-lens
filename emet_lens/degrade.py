import io
import os
import subprocess
import tempfile
from PIL import Image


def jpeg(img: Image.Image, q: int):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def downscale(img: Image.Image, f=0.5):
    w, h = img.size
    return img.convert("RGB").resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def h264(img: Image.Image, crf=30):
    """Intra-frame H.264 round trip through ffmpeg (even dims required)."""
    w, h = img.size
    img = img.convert("RGB").crop((0, 0, w - w % 2, h - h % 2))
    with tempfile.TemporaryDirectory() as d:
        a, b, c = (os.path.join(d, n) for n in ("in.png", "o.mp4", "out.png"))
        img.save(a)
        subprocess.run([_ffmpeg(), "-y", "-i", a, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", b], capture_output=True, check=True)
        subprocess.run([_ffmpeg(), "-y", "-i", b, "-frames:v", "1", c], capture_output=True, check=True)
        return Image.open(c).convert("RGB").copy()


def mp3(in_path: str, out_wav: str, bitrate="32k"):
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "x.mp3")
        subprocess.run([_ffmpeg(), "-y", "-i", in_path, "-b:a", bitrate, m], capture_output=True, check=True)
        subprocess.run([_ffmpeg(), "-y", "-i", m, "-ar", "16000", "-ac", "1", out_wav], capture_output=True, check=True)
    return out_wav


DEGRADATIONS = {
    "clean": lambda im: im,
    "jpeg75": lambda im: jpeg(im, 75),
    "jpeg50": lambda im: jpeg(im, 50),
    "down0.5": lambda im: downscale(im, 0.5),
    "h264_crf30": lambda im: h264(im, 30),
}
