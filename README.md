# Emet Lens — basic deepfake & digital forensics

Image / video / audio in → authenticity score, verdict (`likely_real | likely_fake | inconclusive`) and evidence
(ELA heatmap, flagged frames, flagged audio time ranges).

Detectors: CLIP probe, frequency check, ELA, wav2vec2 audio, video aggregation; a logistic-regression judge fuses them.

## Setup
```
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```
Device: `mps` if available else `cpu` (never CUDA). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically.

## How the verdict is made
- **Pixels decide.** The verdict comes from the CLIP probe and frequency detector, which look at the (face-cropped) image content.
- **Metadata is supporting evidence only.** Camera EXIF nudges the score slightly; missing metadata is neutral (messaging apps strip it).
  An explicit AI-generator marker (PNG `parameters`, IPTC `trainedAlgorithmicMedia`, generator name in `Software`) is treated as declared provenance.
- **ELA is a visual aid.** It is not fused, and is marked "not used" on recompressed files (JPEG quality < 88).
- Until the pixel detectors are trained the answer is `inconclusive`, never a guess.
- Output includes a summary, confidence (high/medium/low/none), detector agreement, input quality, and skipped checks.

## Workflow
```
python scripts/fetch_dff.py --identities 200 --per-identity 2   # optional: small DeepFakeFace subset via range requests (real=wiki; text2img, insight, inpainting fakes)
python scripts/extract_frames.py data/ffpp_videos data/ffpp_frames -n 8        # video -> frames (same source_id per video)
python scripts/build_manifest.py --real data/ffhq --fake stylegan=data/stylegan ffpp=data/ffpp_frames --holdout ffpp
python scripts/train_image.py [--augment]       # CLIP probe, frequency, judge (features cached in cache/)
python -m eval.run_eval --detector judge        # accuracy, AUC, FPR on real; seen vs unseen fake types
python eval/robustness.py                       # JPEG75/50, downscale, H.264 table (python -m eval.robustness)
python scripts/train_audio.py --manifest data/MANIFEST_audio.csv
python -m emet_lens path/to/file                # JSON report to stdout, heatmaps in out/
python app.py                                   # Gradio demo
pytest
```
`data/MANIFEST.csv` columns: `path,label,type,source_id,split`. Splits are by `source_id` (enforced on load).

## Results
Judge (CLIP probe + frequency), trained with `--augment` on 1,812 faces. Test split is by source (316 real, 471 seen fakes,
124 unseen fakes). **Seen** fake types: text2img, insight (face swap), StyleGAN. **Unseen**: SD inpainting (test-only sources).
Real = DeepFakeFace/wiki portraits + FFHQ. Decision point set for ~10% FPR on training reals.

| input quality | FPR on real | seen AUC | unseen AUC | unseen detect rate |
|---|---|---|---|---|
| clean | 3.8% | 0.896 | 0.844 | 48% |
| JPEG 50 | 8.9% | 0.876 | 0.819 | 56% |
| downscale 0.5 | 10.4% | 0.831 | 0.753 | 43% |

Not augmented (earlier run): FPR 6.6%, detect rate 52-73% by type. Individual detectors alone are weaker
(CLIP probe unseen AUC ~0.80, frequency ~0.63), so the judge mostly leans on CLIP.
H.264 and audio results are not measured yet.

## Limitations
- Roughly half of fakes are missed at the chosen low false-positive setting; AI images from generators other than those trained on may do worse.
- Real training faces are Wikipedia portraits and FFHQ; casual phone photos are out of distribution and can be misjudged. Check any new domain on a labelled set first.
- Haar-cascade face detection; falls back to a center crop when no face is found.
- ELA and frequency cues weaken or vanish under heavy recompression/downscaling; use `--augment` and check the robustness table.
- Detectors generalise poorly to unseen generators; always read the unseen column, not the seen one.
- ELA heuristic score is untuned; the judge learns how much to trust it.
- Lip-sync, blink/pulse and audio-video fusion (stretch) are not implemented.
- Only use your own face/voice or licensed data for self-made fakes.
