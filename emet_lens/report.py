import json
import os
import numpy as np
from PIL import Image


def overlay(img: Image.Image, heat: np.ndarray, path: str, alpha=0.5):
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    h = cv2.resize((heat * 255).astype(np.uint8), (rgb.shape[1], rgb.shape[0]))
    col = cv2.cvtColor(cv2.applyColorMap(h, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out = (rgb * (1 - alpha) + col * alpha).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(out).save(path)
    return path


def _pct(x):
    return f"{x:.0%}"


LIPSYNC_LABEL = {"in_sync": "IN SYNC", "out_of_sync": "OUT OF SYNC", "not_checked": "NOT CHECKED"}


def lipsync_report(signals):
    """Explicit lip-sync answer for videos: status in_sync / out_of_sync / not_checked, or None if not a video with audio."""
    s = next((x for x in signals if x["name"] == "lipsync"), None)
    if s is None or "data" not in s:
        return None
    return {**s["data"], "label": LIPSYNC_LABEL[s["data"]["status"]], "explanation": s["finding"]}


def build_summary(p_fake, verdict, reason, signals, assessment=None, declared_ai=False):
    auth = _pct(1 - p_fake)
    used = [x for x in signals if x["name"] in ("clip_probe", "frequency", "audio") and x.get("reliable", True)]
    if verdict == "inconclusive":  # only when no detector is available
        return f"No verdict: {reason}."
    if declared_ai:
        meta = next(x for x in signals if x["name"] == "metadata")
        out = f"FAKE (authenticity {auth}). {meta['finding']}"
    elif verdict == "likely_real":
        out = (f"REAL (authenticity {auth}). The image content shows no sign of AI generation or face "
               "manipulation. " + " ".join(x["finding"] for x in used))
    else:
        top = max(used, key=lambda x: x["score_fake"])
        out = f"FAKE (authenticity {auth}). Strongest evidence from the image content ({top['name']}): {top['finding']}"
    conf = assessment["confidence"] if assessment else None
    if conf and conf != "none":
        out += f" Confidence: {conf}" + (f" ({reason})." if reason else ".")
    ls = lipsync_report(signals)
    if ls:
        out += f" Lip-sync: {ls['label']}. {ls['explanation']}"
    return out


def build_report(p_fake, verdict, reason, signals, flagged=None, caveats=None, assessment=None, declared_ai=False):
    return {
        "authenticity": round(1 - p_fake, 4),
        "verdict": verdict,
        **({"inconclusive_reason": reason} if reason else {}),
        "summary": build_summary(p_fake, verdict, reason, signals, assessment, declared_ai),
        **({"lipsync": lipsync_report(signals)} if lipsync_report(signals) else {}),
        **({"assessment": assessment} if assessment else {}),
        **({"caveats": caveats} if caveats else {}),
        "signals": signals,
        "flagged": flagged or {"frames": [], "audio_seconds": []},
    }


def dump(report, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
