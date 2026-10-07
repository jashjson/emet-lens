import functools
import os
import subprocess
import numpy as np
from PIL import Image
from .base import DummyDetector
from .faces import prep_crop
from .judge import Judge, IMAGE_FEATURES
from .assessment import assess
from .metadata import read_metadata, metadata_signal
from .report import build_report, overlay, dump

# Models are cached locally after the first run; never wait on the network at inference time.
if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUD_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@functools.lru_cache(maxsize=1)
def default_detectors():
    """Loaded once per process (CLIP is ~600 MB); returns a tuple-backed list copy per call."""
    return tuple(_load_default_detectors())


def _load_default_detectors():
    from .detectors.clip_probe import CLIPProbe
    from .detectors.frequency import FrequencyDetector
    from .detectors.ela import ELADetector
    ds = []
    for cls in (CLIPProbe, FrequencyDetector, ELADetector):
        d = cls()
        if d.needs_training:
            try:
                d.load()
            except FileNotFoundError:
                continue  # untrained: skip, judge falls back to the detectors available
        ds.append(d)
    return ds


def run_image_detectors(img, detectors, path=None):
    """Detectors see the face crop (or center crop); ELA runs on the full image so edits aren't resampled away."""
    crop, box = prep_crop(img)
    sigs = []
    for d in detectors:
        sigs.append(d.predict(img, path=path) if d.name == "ela" else d.predict(crop))
    return sigs, box


def analyze_image(path, detectors=None, judge=None, out_dir="out"):
    img = Image.open(path).convert("RGB")
    detectors = default_detectors() if detectors is None else detectors
    judge = Judge().load() if judge is None else judge
    meta = read_metadata(path)
    sigs, box = run_image_detectors(img, detectors, path)
    pixel = {s.name: s.score_fake for s in sigs if s.reliable and s.name != "ela"}
    msig, shift = metadata_signal(meta)
    min_side = min(img.size)
    p, verdict, reason = judge.fuse(pixel, min_side=min_side, meta_shift=shift, declared_ai=msig.decisive)
    skipped, caveats = [], []
    for s in sigs:
        if not s.reliable:
            q = meta["jpeg_quality"]
            skipped.append({"name": s.name, "reason": "file was recompressed" + (f" (JPEG ~{q:.0f})" if q else "")})
            caveats.append("Editing analysis (ELA) is not used for the verdict: this file has been recompressed"
                           + (f" (estimated JPEG quality ~{q:.0f})" if q else "")
                           + ", e.g. by a messaging app, which hides editing traces.")
    for name in ("clip_probe", "frequency"):
        if name not in pixel and not any(s.name == name for s in sigs):
            skipped.append({"name": name, "reason": "not trained"})
    if skipped and any(k["reason"] == "not trained" for k in skipped):
        caveats.append("Pixel detectors are not trained yet, so the verdict cannot rest on the image content.")
    if box is None:
        caveats.append("No face was detected; the center of the image was analysed instead.")
    out = []
    for s in sigs + [msig]:
        d = s.to_dict()
        if s.heatmap is not None:
            d["heatmap"] = overlay(img, s.heatmap, os.path.join(out_dir, f"{s.name}.png"))
        out.append(d)
    assessment = assess(p, verdict, pixel, skipped, meta, box is not None, min_side, reason)
    rep = build_report(p, verdict, reason, out, caveats=caveats, assessment=assessment, declared_ai=msig.decisive)
    dump(rep, os.path.join(out_dir, "report.json"))
    return rep


def lipsync_shift(sig):
    """Logit nudge from the lip-sync signal: mismatch pushes towards fake (max +1.5), good sync towards real (max -0.4)."""
    if sig is None or not sig.reliable:
        return 0.0
    return 1.5 * (2 * sig.score_fake - 1) if sig.score_fake >= 0.5 else -0.4 * (1 - 2 * sig.score_fake)


def sample_frames(path, n=16):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int) if total > 0 else range(n)
    frames = []
    for i in dict.fromkeys(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((int(i), int(i) / fps, Image.fromarray(fr[:, :, ::-1])))
    cap.release()
    return frames


def extract_audio(path, out_wav):
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", out_wav],
                       capture_output=True)
    return r.returncode == 0 and os.path.exists(out_wav)


def analyze_video(path, detectors=None, judge=None, out_dir="out", n_frames=16):
    detectors = default_detectors() if detectors is None else detectors
    judge = Judge().load() if judge is None else judge
    frames = sample_frames(path, n_frames)
    if not frames:
        raise ValueError("could not read any frames")
    per = {}  # detector -> [score per frame]
    heats = {}
    fuse_dets = [d for d in detectors if d.name != "ela"]  # video frames are always re-encoded: ELA is context only
    for _, _, im in frames:
        sigs, _ = run_image_detectors(im, fuse_dets)
        for s in sigs:
            per.setdefault(s.name, []).append(s.score_fake)
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    frame_mean = np.mean([v for v in per.values()], axis=0)
    incons = float(np.mean([np.std(v) for v in per.values()]))
    # Lip-sync is a criterion: a reliable mouth/speech mismatch raises p_fake, good sync lowers it slightly.
    flagged_audio, flagged_sync, caveats, lip = [], [], [], None
    wav = os.path.join(out_dir, "audio.wav")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(wav):
        os.remove(wav)  # never reuse the previous run's audio
    has_audio = extract_audio(path, wav)
    if has_audio:
        from .detectors.lipsync import check_video
        try:
            lip = check_video(path, wav)
        except Exception as e:  # a lip-sync failure must never break the pixel verdict
            from .detectors.lipsync import _unchecked
            lip = _unchecked(f"the check failed ({e})")
            caveats.append(f"Lip-sync check failed: {e}")
    else:
        from .detectors.lipsync import _unchecked
        lip = _unchecked("the video has no audio track")
        caveats.append("No audio track, so lip-sync and voice checks were skipped.")
    shift = lipsync_shift(lip)
    p, verdict, reason = judge.fuse(mean, min_side=min(frames[0][2].size), extra_inconsistency=incons, meta_shift=shift)
    order = np.argsort(-frame_mean)[:3]
    flagged_frames = [{"frame": frames[i][0], "time_s": round(frames[i][1], 2),
                       "score_fake": round(float(frame_mean[i]), 3)} for i in order]
    signals = []
    for k, v in per.items():
        signals.append({"name": k, "score_fake": round(mean[k], 4),
                        "finding": f"Mean over {len(v)} frames; max {max(v):.2f}, frame-to-frame std {np.std(v):.2f}."})
    signals.append({"name": "frame_consistency", "score_fake": round(min(1.0, incons * 2), 4),
                    "finding": f"Average frame-to-frame score variation is {incons:.2f}."})
    top = frames[int(order[0])][2]
    sigs, _ = run_image_detectors(top, [d for d in detectors if d.name == "ela"], None)
    if sigs and sigs[0].heatmap is not None:
        signals[-1]["heatmap"] = overlay(top, sigs[0].heatmap, os.path.join(out_dir, "top_frame_ela.png"))
    if lip is not None:
        signals.append(lip.to_dict())
        flagged_sync = lip.time_ranges
        if lip.reliable and lip.score_fake >= 0.5:
            caveats.append("Possible lip-sync manipulation: mouth movement does not match the speech.")
    if has_audio and os.path.exists("checkpoints/audio.joblib"):
        from .detectors.audio import AudioDetector, load_audio
        a = AudioDetector().predict_audio(load_audio(wav))
        signals.append(a.to_dict())
        flagged_audio = a.time_ranges
    rep = build_report(p, verdict, reason, signals,
                       {"frames": flagged_frames, "audio_seconds": [list(map(float, r)) for r in flagged_audio],
                        "lipsync_seconds": [list(map(float, r)) for r in flagged_sync]}, caveats=caveats)
    dump(rep, os.path.join(out_dir, "report.json"))
    return rep


def analyze_audio(path, out_dir="out"):
    from .detectors.audio import AudioDetector
    sig = AudioDetector().predict_file(path)
    p = sig.score_fake
    ws = getattr(sig, "window_scores", [p])
    reason = "score is too close to 50%" if abs(p - 0.5) < 0.12 else ""
    verdict = "inconclusive" if reason else ("likely_fake" if p >= 0.5 else "likely_real")
    rep = build_report(p, verdict, reason, [sig.to_dict()],
                       {"frames": [], "audio_seconds": [list(map(float, r)) for r in sig.time_ranges]})
    dump(rep, os.path.join(out_dir, "report.json"))
    return rep


def analyze(path, out_dir="out"):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        return analyze_image(path, out_dir=out_dir)
    if ext in VID_EXT:
        return analyze_video(path, out_dir=out_dir)
    if ext in AUD_EXT:
        return analyze_audio(path, out_dir=out_dir)
    raise ValueError(f"unsupported file type: {ext}")
