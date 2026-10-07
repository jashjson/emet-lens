"""File metadata as *supporting* evidence only. Pixels decide the verdict; metadata can only
(a) nudge the score slightly or (b) declare provenance (explicit AI-generator markers / C2PA)."""
import re
from PIL import Image, ExifTags
from .base import Signal
from .jpeg_info import jpeg_quality

AI_TOOLS = re.compile(r"stable.?diffusion|midjourney|dall.?e|firefly|imagen|flux|comfyui|automatic1111|novelai|"
                      r"leonardo\.ai|ideogram|runway|sora|gemini|openai|generative", re.I)
AI_SOURCE = b"trainedAlgorithmicMedia"   # IPTC DigitalSourceType for AI-generated media
EDITORS = re.compile(r"photoshop|lightroom|gimp|snapseed|facetune|faceapp|canva|picsart|capcut|affinity", re.I)


def read_metadata(path):
    info = {"format": None, "width": None, "height": None, "camera": None, "software": None,
            "has_exif": False, "ai_marker": None, "c2pa": False, "editor": None, "jpeg_quality": None}
    try:
        with Image.open(path) as im:
            info.update(format=im.format, width=im.size[0], height=im.size[1])
            exif = im.getexif()
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            info["has_exif"] = bool(tags)
            make, model = str(tags.get("Make", "")).strip(), str(tags.get("Model", "")).strip()
            info["camera"] = (make + " " + model).strip() or None
            info["software"] = str(tags["Software"]) if "Software" in tags else None
            texts = [str(v) for v in im.info.values() if isinstance(v, (str, bytes))]  # PNG text chunks etc.
            blob = " ".join(t if isinstance(t, str) else t.decode("latin1", "ignore") for t in texts[:50])
            hay = f"{info['software'] or ''} {blob[:20000]}"
            m = AI_TOOLS.search(hay) or (re.search(r"parameters|prompt|negative prompt|sampler|cfg scale", blob, re.I))
            if m:
                info["ai_marker"] = m.group(0)
    except Exception:
        pass
    try:
        raw = open(path, "rb").read(2_000_000)
        if AI_SOURCE in raw:
            info["ai_marker"] = info["ai_marker"] or "IPTC digital source type: AI-generated"
        info["c2pa"] = b"c2pa" in raw or b"jumbf" in raw
    except Exception:
        pass
    if info["software"] and EDITORS.search(info["software"]):
        info["editor"] = info["software"]
    info["jpeg_quality"] = jpeg_quality(path)
    return info


def metadata_signal(info):
    """Returns (Signal, logit_shift). Shift is small by design: metadata is easy to strip or forge."""
    if info["ai_marker"]:
        s = Signal("metadata", 0.97, f"File metadata declares AI generation ({info['ai_marker']}). "
                   "This is a self-declared marker, so it is strong evidence the image is synthetic.")
        s.decisive = True
        return s, 4.0
    if info["editor"]:
        return Signal("metadata", 0.55, f"Saved by an image editor ({info['editor']}). Editing is common and "
                      "not proof of manipulation."), 0.3
    if info["camera"]:
        return Signal("metadata", 0.35, f"Camera metadata present ({info['camera']}). Consistent with a camera "
                      "capture, but metadata can be copied, so it counts only slightly."), -0.4
    return Signal("metadata", 0.5, "No camera metadata. Messaging apps and social media strip it, so this "
                  "says nothing either way."), 0.0
