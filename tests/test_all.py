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
