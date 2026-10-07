import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw
from emet_lens.base import DummyDetector
from emet_lens.detectors.ela import ELADetector
from emet_lens.detectors.frequency import FrequencyDetector
from emet_lens.detectors.clip_probe import CLIPProbe
from emet_lens.detectors.audio import AudioDetector
from emet_lens.judge import Judge
from emet_lens.manifest import load_manifest, assign_splits
from eval.run_eval import evaluate


def photo(seed, size=128):
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size, 3) * 255).astype(np.uint8))


def smooth(seed, size=128):  # crude "upsampled" image
    return photo(seed, 16).resize((size, size), Image.BICUBIC)


def test_ela_heatmap_and_edit_detected():
    import io
    im = photo(0)
    b = io.BytesIO(); im.save(b, "JPEG", quality=60); b.seek(0)
    im = Image.open(b).convert("RGB")
    ImageDraw.Draw(im).rectangle([40, 40, 90, 90], fill=(200, 30, 30))  # pasted region
    s = ELADetector().predict(im)
    assert s.heatmap.shape == (128, 128) and 0 <= s.score_fake <= 1 and s.finding


def test_frequency_separates_and_predicts(tmp_path):
    d = FrequencyDetector(ckpt=str(tmp_path / "f.joblib"))
    X = np.stack([d.features(photo(i)) for i in range(20)] + [d.features(smooth(i)) for i in range(20)])
    d.fit(X, np.r_[np.zeros(20), np.ones(20)])
    assert d.predict(smooth(99)).score_fake > d.predict(photo(99)).score_fake


def test_clip_probe_predict_path(tmp_path):
    d = CLIPProbe(ckpt=str(tmp_path / "c.joblib"))
    d.features = lambda img: np.asarray(img.convert("L"), dtype=np.float32).mean(axis=0)[:16]  # stub: no download
    d.fit(np.random.rand(30, 16), np.r_[np.zeros(15), np.ones(15)])
    assert 0 <= d.predict(photo(1)).score_fake <= 1


def test_audio_windows_flag(tmp_path):
    d = AudioDetector(ckpt=str(tmp_path / "a.joblib"))
    d.embed = lambda w: np.array([float(w.std()), float(np.abs(w).mean())])  # stub
    quiet = lambda: np.random.randn(16000 * 3).astype(np.float32) * 0.01
    loud = lambda: np.random.randn(16000 * 3).astype(np.float32)
    d.fit(np.stack([d.embed(quiet()) for _ in range(10)] + [d.embed(loud()) for _ in range(10)]), np.r_[np.zeros(10), np.ones(10)])
    s = d.predict_audio(np.concatenate([quiet(), loud(), quiet()]))
    assert s.time_ranges == [(3.0, 6.0)]


def test_judge_inconclusive_rules():
    j = Judge()
    p, v, why = j.fuse({"clip_probe": 0.05, "frequency": 0.95})
    assert v in ("likely_real", "likely_fake") and why == "detectors strongly disagree"  # still a call, flagged low-confidence
    assert j.fuse({"clip_probe": 0.1, "frequency": 0.1})[1] == "likely_real"
    assert j.fuse({"clip_probe": 0.9, "frequency": 0.9})[1] == "likely_fake"
    assert "low resolution" in j.fuse({"clip_probe": 0.1, "frequency": 0.1}, min_side=32)[2]
    assert j.fuse({"ela": 0.05})[1] == "inconclusive"  # no pixel detector at all: setup problem, not a verdict


def test_manifest_split_by_source_and_eval(tmp_path):
    rows = [dict(path=f"x{i}.jpg", label="real" if i % 2 else "fake", type="real" if i % 2 else "gan", source_id=f"s{i // 2}") for i in range(40)]
    df = assign_splits(pd.DataFrame(rows))
    assert (df.groupby("source_id").split.nunique() == 1).all()
    df.to_csv(tmp_path / "m.csv", index=False)
    load_manifest(tmp_path / "m.csv")
    bad = df.copy(); bad.loc[0, "split"] = "test" if bad.loc[0, "split"] == "train" else "train"
    bad.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "b.csv")


def test_eval_dummy_end_to_end(tmp_path):
    rows = []
    for i in range(6):
        for typ, lab, seed in (("real", "real", i), ("gan", "fake", 100 + i), ("diff", "fake", 200 + i)):
            p = tmp_path / f"{typ}{i}.png"; photo(seed).save(p)
            rows.append(dict(path=str(p), label=lab, type=typ, source_id=f"{typ}{i}", split="train" if (i < 3 and typ != "diff") else "test"))
    df = pd.DataFrame(rows)
    d = DummyDetector()
    res = evaluate(df, lambda p: d.predict(Image.open(p)).score_fake)
    assert "seen" in res and "unseen" in res and "fpr_real" in res
    assert res["unseen"]["n_fake"] == 6


def test_ela_unreliable_on_recompressed_file(tmp_path):
    p = tmp_path / "wa.jpg"; photo(3, 300).save(p, "JPEG", quality=70)
    s = ELADetector().predict(Image.open(p).convert("RGB"), path=str(p))
    assert not s.reliable and "Not used for the verdict" in s.finding
    q = tmp_path / "hq.jpg"; photo(3, 300).save(q, "JPEG", quality=95)
    assert ELADetector().predict(Image.open(q).convert("RGB"), path=str(q)).reliable


def _trained_pipeline(tmp_path):
    """Frequency + stubbed CLIP trained on synthetic photo-vs-smooth data."""
    f = FrequencyDetector(ckpt=str(tmp_path / "f.joblib"))
    X = np.stack([f.features(photo(i)) for i in range(20)] + [f.features(smooth(i)) for i in range(20)])
    f.fit(X, np.r_[np.zeros(20), np.ones(20)])
    c = CLIPProbe(ckpt=str(tmp_path / "c.joblib"))
    c.features = lambda img: f.features(img)[:16]
    c.fit(np.stack([c.features(photo(i)) for i in range(20)] + [c.features(smooth(i)) for i in range(20)]), np.r_[np.zeros(20), np.ones(20)])
    return [c, f, ELADetector()]


def test_analyze_image_pixel_verdict_summary_and_assessment(tmp_path):
    from emet_lens.pipeline import analyze_image
    dets = _trained_pipeline(tmp_path)
    real, fake = tmp_path / "r.png", tmp_path / "f.png"
    photo(500, 300).save(real); smooth(500, 300).save(fake)
    r = analyze_image(str(real), dets, Judge(ckpt=str(tmp_path / "j.joblib")), str(tmp_path / "o1"))
    f = analyze_image(str(fake), dets, Judge(ckpt=str(tmp_path / "j.joblib")), str(tmp_path / "o2"))
    assert r["verdict"] == "likely_real" and f["verdict"] == "likely_fake"
    for rep in (r, f):
        assert rep["summary"] and rep["assessment"]["confidence"] in ("high", "medium", "low")
        assert set(rep["assessment"]["detectors_used"]) == {"clip_probe", "frequency"}
        assert any(s["name"] == "metadata" for s in rep["signals"])


def test_metadata_ai_marker_is_decisive_but_absence_is_neutral(tmp_path):
    from PIL import PngImagePlugin
    from emet_lens.metadata import read_metadata, metadata_signal
    from emet_lens.pipeline import analyze_image
    meta = PngImagePlugin.PngInfo(); meta.add_text("parameters", "a portrait, Steps: 20, Sampler: Euler, CFG scale: 7")
    p = tmp_path / "ai.png"; photo(1, 300).save(p, pnginfo=meta)
    sig, shift = metadata_signal(read_metadata(str(p)))
    assert sig.decisive and shift > 0
    rep = analyze_image(str(p), [], Judge(ckpt=str(tmp_path / "j.joblib")), str(tmp_path / "o"))
    assert rep["verdict"] == "likely_fake" and "metadata" in rep["summary"].lower()
    q = tmp_path / "plain.png"; photo(1, 300).save(q)
    sig, shift = metadata_signal(read_metadata(str(q)))
    assert not sig.decisive and shift == 0
    rep = analyze_image(str(q), [], Judge(ckpt=str(tmp_path / "j.joblib")), str(tmp_path / "o"))
    assert rep["verdict"] == "inconclusive"  # no pixel detectors + no declared marker -> no verdict


def _speech_env(n, seed=0):
    rng = np.random.RandomState(seed)
    return np.convolve(np.abs(rng.randn(n)), np.ones(5) / 5, mode="same") ** 2 + 0.01


def test_lipsync_synced_vs_mismatched_series():
    from emet_lens.detectors.lipsync import lipsync_signal
    fps, n = 25, 250
    env = _speech_env(n)
    rng = np.random.RandomState(1)
    synced = lipsync_signal(env * 4 + 0.05 * rng.randn(n), env, fps)           # mouth follows speech, 80 ms lag ignored
    shifted = lipsync_signal(np.roll(env, 2) + 0.05 * rng.randn(n), env, fps)
    other = lipsync_signal(_speech_env(n, seed=9), env, fps)                    # mouth moves to different speech
    assert synced.reliable and synced.score_fake < 0.3 and shifted.score_fake < 0.4
    assert other.reliable and other.score_fake > 0.5 and other.time_ranges and "does not follow" in other.finding
    assert not lipsync_signal(env, env * 0, fps).reliable          # silent audio -> cannot judge
    assert not lipsync_signal(env[:50], env[:50], fps).reliable    # too short
    assert not lipsync_signal(env, env, fps, face_frac=0.2).reliable


def test_lipsync_shift_direction():
    from emet_lens.base import Signal
    from emet_lens.pipeline import lipsync_shift
    assert lipsync_shift(Signal("lipsync", 0.9, "x")) > 1 and lipsync_shift(Signal("lipsync", 0.1, "x")) < 0
    assert lipsync_shift(Signal("lipsync", 0.9, "x", reliable=False)) == 0 and lipsync_shift(None) == 0


def test_lipsync_video_end_to_end(tmp_path):
    """Synthetic talking video (real face photo, mouth area flickers with the audio) with matching vs unrelated audio."""
    import glob, subprocess, cv2, imageio_ffmpeg, soundfile as sf
    from emet_lens.detectors.lipsync import check_video
    faces = sorted(glob.glob("data/ffhq/*.jpg"))
    if not faces:
        pytest.skip("needs a face image in data/ffhq")
    base = cv2.resize(cv2.imread(faces[0]), (256, 256))
    if not len(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
               .detectMultiScale(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(48, 48))):
        pytest.skip("face photo not detected")
    fps, secs, sr = 25, 8, 16000
    env = _speech_env(fps * secs)
    env = env / env.max()
    rng = np.random.RandomState(0)

    def make(audio_env, name):
        vid = str(tmp_path / f"{name}.avi")
        w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), fps, (256, 256))
        for i in range(fps * secs):
            fr = base.astype(np.float32)
            fr[150:240, 60:200] += rng.randn(90, 140, 1) * 0 + 70 * (env[i] - 0.3)   # mouth region follows video-side env
            w.write(np.clip(fr, 0, 255).astype(np.uint8))
        w.release()
        t = np.arange(sr * secs) / sr
        a = np.interp(t, np.arange(len(audio_env)) / fps, audio_env) * np.sin(2 * np.pi * 220 * t)
        wav = str(tmp_path / f"{name}.wav")
        sf.write(wav, a.astype(np.float32), sr)
        return vid, wav

    good = check_video(*make(env, "good"))
    bad = check_video(*make(_speech_env(fps * secs, seed=5) / 1.0, "bad"))
    assert good.reliable and bad.reliable
    assert good.score_fake < bad.score_fake and good.score_fake < 0.5 < bad.score_fake


def test_lipsync_status_in_report():
    from emet_lens.detectors.lipsync import lipsync_signal
    from emet_lens.report import build_report
    fps, n = 25, 250
    env = _speech_env(n)
    for sig, status in [(lipsync_signal(env * 3, env, fps), "in_sync"),
                        (lipsync_signal(_speech_env(n, seed=9), env, fps), "out_of_sync"),
                        (lipsync_signal(env, env * 0, fps), "not_checked")]:
        rep = build_report(0.2, "likely_real", "", [sig.to_dict()])
        assert rep["lipsync"]["status"] == status and "Lip-sync:" in rep["summary"]
    assert "lipsync" not in build_report(0.2, "likely_real", "", [])


def test_consistency_detector_flags_pasted_face_region(tmp_path):
    """Smooth-noise 'camera' texture everywhere vs one region with different noise: the outlier map must light up there."""
    from emet_lens.detectors.consistency import ConsistencyDetector, patch_stats, _robust_anomaly
    rng = np.random.RandomState(0)
    base = np.clip(128 + 40 * np.sin(np.linspace(0, 6, 256))[None, :] + rng.randn(256, 256) * 4, 0, 255).astype(np.float32)
    pasted = base.copy()
    pasted[96:192, 96:192] = np.clip(base[96:192, 96:192] * 0.2 + 100 + rng.randn(96, 96) * 1.0, 0, 255)  # different noise level
    for img, expect_hot in ((base, False), (pasted, True)):
        F, t = patch_stats(img)
        a = _robust_anomaly(F.reshape(-1, F.shape[-1]), t.reshape(-1)).reshape(F.shape[:2])
        inner, outer = a[6:11, 6:11].mean(), np.r_[a[:3].ravel(), a[-3:].ravel()].mean()
        assert (inner > 2 * outer) == expect_hot
    det = ConsistencyDetector(ckpt=str(tmp_path / "c.joblib"))
    X = np.vstack([det.features(Image.fromarray(base.astype(np.uint8)).convert("RGB")) for _ in range(4)])
    det.fit(np.vstack([X, X + 1]), np.array([0, 1] * 4))
    sig = det.predict(Image.fromarray(pasted.astype(np.uint8)).convert("RGB"))
    assert 0 <= sig.score_fake <= 1 and sig.finding and sig.name == "consistency"
    assert sig.heatmap is not None and sig.heatmap.shape[0] > 4   # localisation map is produced
