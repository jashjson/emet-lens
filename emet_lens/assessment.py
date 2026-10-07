import numpy as np

EXPECTED_DETECTORS = ["clip_probe", "frequency"]


def assess(p_fake, verdict, used: dict, skipped: list, meta: dict, face_found: bool, min_side: int, reason: str = ""):
    """Key assessment metrics shown next to the verdict.
    used: {detector: score_fake} actually fused; skipped: [{"name","reason"}]."""
    vals = list(used.values())
    margin = abs(p_fake - 0.5) * 2
    agreement = 1.0 - (max(vals) - min(vals)) if len(vals) > 1 else (1.0 if vals else 0.0)
    coverage = len([k for k in used if k in EXPECTED_DETECTORS]) / len(EXPECTED_DETECTORS)
    quality, qnotes = 1.0, []
    q = meta.get("jpeg_quality")
    if q is not None and q < 80:
        quality *= 0.75; qnotes.append(f"heavily compressed (JPEG ~{q:.0f})")
    if min_side < 256:
        quality *= 0.6; qnotes.append(f"low resolution ({min_side}px)")
    if not face_found:
        quality *= 0.7; qnotes.append("no face found")
    score = margin * agreement * quality * (0.5 + 0.5 * coverage)
    if not [k for k in used if k in EXPECTED_DETECTORS]:
        level = "none"
    else:
        level = "high" if score >= 0.55 else "medium" if score >= 0.3 else "low"
        if reason:
            level = "low"
    return {
        "confidence": level,
        "confidence_score": round(float(score), 3),
        "low_confidence_reason": reason or None,
        "agreement": round(float(agreement), 3),
        "coverage": round(float(coverage), 3),
        "input_quality": "good" if quality >= 0.95 else "degraded" if quality >= 0.6 else "poor",
        "input_quality_notes": qnotes,
        "detectors_used": sorted(used),
        "detectors_skipped": skipped,
        "face_detected": bool(face_found),
        "file": {k: meta.get(k) for k in ("format", "width", "height", "camera", "software", "jpeg_quality", "c2pa")},
    }
