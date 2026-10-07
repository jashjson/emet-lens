"""Lip-sync consistency: does mouth movement in the video follow the speech energy in the audio?

Training-free heuristic. In genuine talking-head video the mouth-region motion and the audio loudness rise and fall
together (correlation at a small lag). Dubbed audio, voice-cloned speech laid over real video and lip-sync deepfakes
(Wav2Lip style) tend to break that coupling. The score is an uncalibrated mapping of the best correlation within
+-MAX_LAG_S, so it only nudges the judge and is reported as "not reliable" unless there is enough speech and a stable face.
"""
import numpy as np
from ..base import Signal

MAX_SECONDS = 15.0     # analyse the first N seconds of the clip
MAX_LAG_S = 0.25       # tolerated audio/video offset
MIN_SPEECH_FRAC = 0.2  # share of frames with audible audio needed to judge
MIN_FACE_FRAC = 0.7    # share of frames with a detected face needed to judge
WIN_S = 2.0            # window for flagging time ranges
CORR_MID, CORR_SCALE = 0.15, 0.07   # corr 0.15 -> score 0.5; 0.30 -> ~0.1; 0.0 -> ~0.9


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _smooth(x, k=3):
    return np.convolve(x, np.ones(k) / k, mode="same")


def best_lag_corr(mouth, audio, fps, max_lag_s=MAX_LAG_S):
    """Max Pearson correlation of two equal-rate series over lags within +-max_lag_s. Returns (corr, lag_s)."""
    mouth, audio = _z(_smooth(mouth)), _z(_smooth(audio))
    n = min(len(mouth), len(audio))
    mouth, audio = mouth[:n], audio[:n]
    best, best_lag = -1.0, 0
    for lag in range(-int(round(max_lag_s * fps)), int(round(max_lag_s * fps)) + 1):
        a, m = (audio[lag:], mouth[:n - lag]) if lag >= 0 else (audio[:n + lag], mouth[-lag:])
        if len(a) < 8 or a.std() < 1e-9 or m.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, m)[0, 1])
        if c > best:
            best, best_lag = c, lag
    return best, best_lag / fps


def corr_to_score(corr):
    return float(1 / (1 + np.exp((corr - CORR_MID) / CORR_SCALE)))


def _unchecked(why):
    return Signal("lipsync", 0.5, f"Lip-sync not checked: {why}.", reliable=False,
                  data={"status": "not_checked", "reason": why})


def lipsync_signal(mouth, audio_env, fps, face_frac=1.0):
    """mouth / audio_env: per-frame series at `fps`. Returns a Signal; reliable=False when it cannot be judged."""
    n = min(len(mouth), len(audio_env))
    mouth, audio_env = np.asarray(mouth[:n], float), np.asarray(audio_env[:n], float)
    if n < fps * 3:
        return _unchecked("clip is shorter than 3 seconds")
    if face_frac < MIN_FACE_FRAC:
        return _unchecked(f"a face was found in only {face_frac:.0%} of frames")
    speech = audio_env > 0.2 * np.percentile(audio_env, 95) if audio_env.max() > 1e-6 else np.zeros(n, bool)
    if speech.mean() < MIN_SPEECH_FRAC:
        return _unchecked("little or no speech in the audio")
    if mouth.std() < 1e-6:
        return _unchecked("no mouth movement could be measured")
    corr, lag = best_lag_corr(mouth, audio_env, fps)
    score = corr_to_score(corr)
    # windows where the coupling is missing -> time ranges
    w, ranges = int(WIN_S * fps), []
    for s in range(0, n - w + 1, w // 2):
        c, _ = best_lag_corr(mouth[s:s + w], audio_env[s:s + w], fps)
        if speech[s:s + w].mean() > 0.4 and c < CORR_MID - CORR_SCALE:
            a, b = s / fps, (s + w) / fps
            if ranges and a <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], b)
            else:
                ranges.append((a, b))
    if score >= 0.5:
        finding = (f"Mouth movement does not follow the speech (best correlation {corr:.2f}, expected above ~0.3 for "
                   f"genuine talking video). This can indicate dubbed or cloned audio or a lip-sync manipulation.")
    else:
        finding = f"Mouth movement follows the speech (correlation {corr:.2f} at {lag * 1000:+.0f} ms offset)."
    data = {"status": "out_of_sync" if score >= 0.5 else "in_sync", "correlation": round(corr, 3),
            "offset_ms": round(lag * 1000)}
    return Signal("lipsync", score, finding, time_ranges=ranges if score >= 0.5 else [], data=data)


def audio_envelope(wav, sr, fps, n_frames):
    """RMS loudness per video frame (aligned to frame start)."""
    hop = sr / fps
    env = np.array([np.sqrt(np.mean(wav[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12)
                    if int(i * hop) < len(wav) else 0.0 for i in range(n_frames)])
    return env


def mouth_motion(path, max_seconds=MAX_SECONDS):
    """Per-frame motion energy of the mouth region of the main face. Returns (series, fps, face_frac)."""
    import cv2
    from ..faces import _get_cascade
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    grays, boxes = [], []
    cascade = _get_cascade()
    while len(grays) < int(max_seconds * fps):
        ok, fr = cap.read()
        if not ok:
            break
        scale = 480.0 / max(fr.shape[1], 1)
        g = cv2.cvtColor(cv2.resize(fr, None, fx=scale, fy=scale) if scale < 1 else fr, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(g, 1.1, 5, minSize=(48, 48))
        boxes.append(max(found, key=lambda f: f[2] * f[3]) if len(found) else None)
        grays.append(g)
    cap.release()
    got = [b for b in boxes if b is not None]
    if not grays or not got:
        return np.zeros(len(grays)), fps, 0.0
    x, y, w, h = np.median(np.array(got), axis=0).astype(int)   # fixed mouth box: stable against detector jitter
    x0, x1 = x + int(0.2 * w), x + int(0.8 * w)
    y0, y1 = y + int(0.6 * h), y + h
    rois = [cv2.resize(g[y0:y1, x0:x1], (48, 24)).astype(np.float32) for g in grays]
    motion = np.array([0.0] + [float(np.abs(rois[i] - rois[i - 1]).mean()) for i in range(1, len(rois))])
    return motion, fps, len(got) / len(boxes)


def check_video(path, wav_path):
    """Full check for a video file with its extracted mono 16 kHz wav."""
    from .audio import load_audio, SR
    motion, fps, face_frac = mouth_motion(path)
    wav = load_audio(wav_path, SR)
    env = audio_envelope(wav, SR, fps, len(motion))
    return lipsync_signal(motion, env, fps, face_frac)
