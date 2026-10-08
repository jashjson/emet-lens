# Emet Lens — basic deepfake & digital forensics (HNX26PSI10)

Image / video / audio in → authenticity score, a definite verdict (`likely_real | likely_fake`) and evidence
(ELA heatmap, flagged video frames, flagged audio time ranges), as JSON, a Gradio demo, and a web frontend.
It is a prototype: read the **Results** and **Limitations** sections before trusting any verdict.

## Contents
0. [How it works, in plain words](#how-it-works-in-plain-words) · [TCNC](#tcnc-texture-conditioned-noise-consistency) · 1. [Stack](#stack) · 2. [Setup](#setup) · 3. [Running](#running) · 4. [How the verdict is made](#how-the-verdict-is-made)
5. [Detectors](#detectors) · 6. [Output](#output) · 7. [Data](#data) · 8. [Workflow](#workflow)
9. [Results](#results) · 10. [Tests](#tests) · 11. [Project layout](#project-layout) · 12. [Status](#status) · 13. [Limitations](#limitations) · 14. [Ethics](#ethics)

## How it works, in plain words

**The short version.** You give it a picture, a video or a sound clip. It runs a handful of small checks on it. Each check
gives a number from 0 (looks real) to 1 (looks fake) and one sentence saying why. A last step puts those numbers together
and makes the call: REAL or FAKE, with a percentage and a note on how sure it is.

### Step 1: look at the file itself
Before any analysis it reads what the file says about itself: the type, the size, the camera name, and the software that
saved it. If the file openly says "made by an AI image tool", we believe it and call it fake. Most files say nothing, and
that proves nothing, because WhatsApp and most social apps wipe this information. So a missing camera name is ignored.
A real camera name only nudges the score a little towards real, because anyone can copy it onto a fake.

### Step 2: find the face and clean it up
The checks that decide the verdict look at the face. A simple face finder (OpenCV) cuts out the face with some
margin. If it can't find one, it uses the middle of the picture. The crop is then shrunk to 224 x 224 pixels and saved once
as a JPEG at quality 75.

Why the odd re-saving? Early on, the real photos in our data were saved one way and the fakes another way (different
shapes and compression). A model can learn "this was saved like a fake" and score well without learning anything about
fakes. Putting every crop through the same process takes that shortcut away.

### Step 3: the checks
1. **CLIP check (the main one).** CLIP is a model from OpenAI that turns a picture into a list of 512 numbers describing
   what it looks like. We don't change CLIP. We trained a very small classifier (logistic regression) on top of those
   numbers to tell real faces from fake ones. This is the strongest check we have.
2. **Frequency check.** Image generators leave faint repeating patterns that you can't see but that show up when you
   break the picture into its fine and coarse detail (a Fourier transform). We boil that down to 32 numbers and train
   another small classifier on them. On its own it is weak, and it breaks easily when the image is compressed or shrunk.
3. **Error level analysis (ELA).** Save the picture again as a JPEG and see which parts change the most. Parts that were
   pasted in or edited later often change differently from the rest. We draw this as a heat map. It is useful to look at,
   but it gave false alarms on real photos that had been shared and recompressed, so it does **not** affect the verdict.
   It is switched off for low-quality JPEGs, non-JPEG files and tiny images.
4. **Lip-sync check (videos with sound).** In a real video of someone talking, the mouth moves when the sound gets
   louder. We measure how much the mouth area moves in each frame, line that up against the loudness of the audio, and
   take the best match allowing for up to a quarter of a second of offset. High match means in sync. A very low match
   means the sound probably doesn't belong to that mouth (dubbed, cloned voice or a lip-sync fake). It needs a clear face
   in most frames and real speech. Otherwise it says NOT CHECKED and changes nothing. It needs no training, so its
   cut-off points are rough guesses until we test it on real videos.
5. **Audio check (sound).** Another unchanged model (wav2vec2) turns each 3 second piece of audio into numbers, and a
   small classifier decides if it sounds like a synthetic voice. The code is written but we have no audio data yet, so it
   is not trained and does not run.

### Step 4: videos
For a video we take 16 frames spread across it, run the face checks on every frame and average the scores. If the scores
jump around a lot from frame to frame, that lowers our confidence. The three most suspicious frames are listed with their
time. The lip-sync check and the audio check run on the soundtrack.

### Step 5: the judge
The judge is one more small classifier. It takes the CLIP score and the frequency score and learns how much to trust each.
In practice it leans mostly on CLIP. We also tuned it so that only about 1 in 10 real training faces gets called fake, since
accusing a real photo is the worse mistake. The metadata and lip-sync results then push the final number up or down a bit.
A mouth that doesn't match the sound pushes it towards fake. A good match pushes it slightly towards real.

The result is always a clear REAL or FAKE. When we are unsure it still says one of them, but the confidence shows as
"low" and a reason is given: the picture is tiny, the checks disagree, the frames disagree, or the score is close to 50%.
The one exception is when no trained check is installed at all, which means setup is broken, so it says "no verdict".

### How it was trained
- **Data:** 2,723 face images. 1,812 for training (684 real, 1,128 fake) and 911 for testing (316 real, 595 fake).
  Real faces come from Wikipedia portraits (DeepFakeFace) and FFHQ. Fakes come from DeepFakeFace (text-to-image, face
  swap, inpainting) and a Hugging Face set of StyleGAN faces.
- **Splitting:** by person, never by image. If the same person appeared in both train and test, the test would be too easy.
  The loader refuses to run if that happens. Inpainting fakes were kept out of training completely, so we can see how it
  does on a kind of fake it has never met.
- **Training:** compute the CLIP and frequency numbers for every training face, save them, and fit the small classifiers.
  We also train on copies that are compressed or shrunk so the model copes with messy real-world files. The judge is
  trained on scores each classifier produced for faces it did not train on, which keeps the judge from being overconfident.
- **What we did not train:** CLIP and wav2vec2 are used as they are. ELA, metadata and lip-sync have no training.
  Training takes a few minutes on a laptop because only the tiny classifiers are being fitted.

### How good is it, honestly
On the held-out test set, with clean images: about 4 in 100 real faces are wrongly called fake, and about half of the
fakes are caught. For the unseen generator the catch rate is 48%. When images are compressed or shrunk, the false alarm
rate rises to 8 to 10 in 100. These numbers come from a small dataset of mostly portrait photos. Ordinary phone photos, other
AI generators and newer fakes can do worse, so treat the output as a clue, not proof. The tables further down have the details.

### What it can't do yet
- Voice cloning detection (no audio data, so the audio check is untrained).
- A trained lip-sync model. What we have is a simple mouth-movement-versus-loudness match.
- H.264 video compression numbers, and testing on a real video dataset.
- Blink and pulse checks.
- Finding the exact edited area. ELA's heat map is only a hint.

## TCNC: texture-conditioned noise consistency
**Idea.** A real photo comes from one camera and one processing chain. How noisy and grainy a patch is depends on how
detailed that patch is (smooth skin is quiet, hair is busy) and on nothing else. A swapped, inpainted or re-generated face
comes from a different process, so its fine noise sits off that pattern. TCNC checks the image against itself, so it needs no
knowledge of any particular generator, and resolution and compression changes mostly cancel out of the comparison.

**Steps** (`emet_lens/detectors/consistency.py`, about 150 lines, no neural network):
1. Subtract a blurred copy to keep only the fine detail (the "residual").
2. Cut the image into 32 px patches and measure six things per patch: noise strength, how heavy-tailed the noise is,
   how correlated neighbouring pixels are (three directions), and how strong the JPEG 8x8 block grid is.
3. For this image only, work out how each measure normally changes with how detailed the patch is (a quadratic fit).
   Subtract that, so differences caused by content are gone and only the unusual part is left.
4. Compare the face patches with the patches around the face. A second, face-free score drops the worst 20% of patches,
   refits, and marks the outliers. That outlier map is the heat map.
5. Ten numbers go into a small gradient-boosted classifier (depth 3). Trained with `python scripts/train_consistency.py`.

**What was measured** (same held-out split as the other detectors; AUC unless noted):

| detector | FPR on real | face swap (insight) | inpainting (unseen) | text2img | StyleGAN |
|---|---|---|---|---|---|
| TCNC alone, clean | 33% at its default cut-off | 0.75 | 0.62 | 0.65 | 0.63 |
| TCNC alone, JPEG 75 | 31% | 0.77 | 0.65 | 0.68 | 0.49 |
| TCNC alone, downscale 0.5 | 42% | 0.58 | 0.52 | 0.54 | 0.54 |
| CLIP + frequency (current judge inputs) | 6.6% | 0.881 | 0.858 | 0.868 | 0.929 |
| CLIP + frequency + TCNC (one experiment) | 7.6% | 0.900 | 0.867 | 0.878 | 0.932 |

**Honest reading.** On its own it is weak, and shrinking the image kills it. It works best where the idea says it should, on
face swaps, and as a third input it adds about 0.02 AUC there and 0.01 overall. With only 124 to 224 test images per fake type
that gain is inside the noise, so it is **not fused into the verdict**. It is shown as a heat map showing where the
image is inconsistent, which is the part the CLIP and frequency checks cannot give. Fully synthetic images are consistent
with themselves, so by design it does not catch those. To decide whether to fuse it, test it on a bigger set of
face-swap videos (FaceForensics++) and retrain the judge.

- **Python 3.11** venv via `uv`; **PyTorch** on `mps` (Apple GPU) else `cpu`. Never CUDA (`emet_lens/device.py`).
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically.
- **Models:** frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) image features; wav2vec2-base audio features (layer 6, mean/std pooled).
- **Classical ML:** scikit-learn logistic regression (probe, frequency, audio, judge), StandardScaler, GroupKFold.
- **Image/video/audio:** Pillow, OpenCV (`opencv-python-headless<5`, Haar face detection), SciPy, librosa, soundfile,
  `imageio-ffmpeg` (bundled ffmpeg, no system install needed).
- **Data tooling:** pandas, pyarrow, HTTP range requests (custom `RangeFile`) so datasets can be sampled without full downloads.
- **UI:** Gradio 6 (`app.py`) and a FastAPI + static HTML frontend (`frontend/`).
- **Testing:** pytest.

## Setup
```
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python pyarrow fastapi uvicorn python-multipart   # data scripts + web frontend
source .venv/bin/activate
```
`opencv-python-headless` must stay `<5` (OpenCV 5 has no `CascadeClassifier`, which the face detector needs).
Checkpoints, data and feature cache are not in git; train them with the workflow below.

## Running
```
python app.py                         # Gradio demo (http://127.0.0.1:7860)
python -m frontend.server             # web frontend (http://127.0.0.1:8000), PORT=... to change
python -m emet_lens path/to/file      # JSON report to stdout, heatmaps written to out/
pytest                                # unit tests
```
Both UIs load CLIP once at start-up and run offline when the CLIP weights are cached, so wait for the "Running on…" line
before uploading. Only one analysis runs at a time. If a browser shows "Failed to fetch" after a restart, hard-refresh the tab.

## How the verdict is made
- **Pixels decide.** The verdict comes from the CLIP probe and the frequency detector, which look at the face-cropped image content
  (centre crop when no face is found). Video frames go through the same detectors.
- **A judge fuses them.** A logistic regression over the detectors' out-of-fold scores (GroupKFold by source) gives `p_fake`.
  The decision point is shifted so about 10% of training reals would be called fake, to keep the false-positive rate low.
- **Metadata is supporting evidence only.** Camera EXIF nudges the score slightly (towards real); an editor in `Software` nudges it
  slightly towards fake; missing metadata is neutral (messaging apps strip it). An explicit AI-generator marker (PNG `parameters`,
  IPTC `trainedAlgorithmicMedia`, generator name in `Software`, C2PA bytes) is treated as declared provenance and is decisive.
- **ELA is a visual aid.** It is not fused. It is marked "not used" on recompressed files (estimated JPEG quality < 88), non-JPEGs
  and images under 256 px, because it flagged real recompressed photos (42% FPR) when it was fused.
- **Always a call.** The verdict is always REAL or FAKE; uncertainty is reported through *confidence* (high / medium / low), not by
  withholding a verdict. Low-confidence reasons: low resolution, detectors disagree (> 0.6 apart), inconsistent frames (> 0.3),
  or a score close to the decision point (margin < 0.12). The only exception is `inconclusive` when no trained pixel detector is
  loaded, which is a setup problem and not a guess.
- **Shortcut control.** Before any pixel detector sees a face it is resized to 224 and re-encoded at JPEG q75 (`prep_crop`), since
  the real DeepFakeFace images were non-square q≈80 and the fakes 512×512 q≈75, which a model could otherwise learn instead of the fakes.

## Detectors
Every detector returns a `Signal`: `score_fake`, a plain-English `finding`, and a `heatmap` and/or `time_ranges`
(plus `decisive` and `reliable` flags). Code: `emet_lens/detectors/`.

| Detector | What it does | Evidence | Fused? |
|---|---|---|---|
| CLIP probe (`clip_probe.py`) | frozen CLIP ViT-B/32 features → logistic regression | score | yes |
| Frequency (`frequency.py`) | radial log-power-spectrum profile (32 bins) → scaler + LR; looks for generator spectral fingerprints | score | yes |
| ELA / double-JPEG (`ela.py`) | re-saves at q90 and maps error level differences, smoothed; p99 sigmoid score | heatmap | no (visual aid) |
| Audio (`audio.py`) | wav2vec2-base layer-6 mean/std features over 3 s windows → LR | flagged time ranges | audio files and video soundtracks |
| Consistency, TCNC (`consistency.py`) | compares the noise and texture of the face with the rest of the same image | heatmap of odd patches | no (context only, like ELA) |
| Lip-sync (`lipsync.py`) | correlates mouth-region motion (fixed box under the median Haar face) with audio loudness within ±0.25 s; training-free | flagged time ranges (`lipsync_seconds`) | nudges the judge (mismatch up to +1.5 logit towards fake, good sync −0.4); only when reliable |
| Video aggregation (`pipeline.analyze_video`) | samples frames, scores each, aggregates; frame-score inconsistency feeds confidence | flagged frames | yes (ELA excluded) |
| Metadata (`metadata.py`) | format, size, camera, software, EXIF presence, AI markers, C2PA, JPEG quality | finding | small logit shift only |
| Judge (`judge.py`) | LR over `[clip_probe, frequency]` + shifts → `p_fake`, verdict, low-confidence reason | verdict | — |

## Output
`python -m emet_lens <file>` (and the UIs) return a JSON report with:
- `p_fake` / authenticity score, `verdict`, a plain-English **summary**, and caveats;
- per-detector `signals` (score, finding, `reliable`, heatmap path, `time_ranges`);
- an **assessment**: confidence (high/medium/low) and score, low-confidence reason, detector agreement, coverage, input quality
  with notes, detectors used and skipped (with why), face detected, file properties;
- evidence: ELA heatmap overlays (`out/`), flagged frames for video, flagged audio times (`audio_seconds`) and time ranges where the mouth does not follow the speech (`lipsync_seconds`).

The Gradio app shows a verdict card, a metrics grid, and Evidence / Heatmap / Raw JSON tabs.
The web frontend adds pages for the analyzer, sample images, a robustness check, how it works, tools, methodology and a report.

## Data
`data/MANIFEST.csv` columns: `path,label,type,source_id,split`. Splits are by `source_id` (person / source video), never by frame,
and the loader raises an error if a `source_id` appears in more than one split. Held-out fake types come only from test-split sources.
Currently 2,723 rows (~1,812 train faces; test split 316 real, 471 seen fakes, 124 unseen fakes in the headline eval).

| Source | Used as | Notes |
|---|---|---|
| OpenRL/DeepFakeFace (apache-2.0) | real (`wiki`) and fakes `text2img`, `insight` (face swap), `inpainting` | sampled by HTTP range requests on the zips (`scripts/fetch_dff.py`) |
| bitmind/ffhq-256 | real | parquet row groups (`scripts/fetch_ffhq_stylegan.py`) |
| 34data/140k-real-fake-faces-fake | `stylegan` fakes | same script |

- **Seen** fake types (in training): text2img, insight, StyleGAN. **Unseen** (held out, test-only sources): SD inpainting.
- Features are cached to `cache/`; data, checkpoints, cache and logs are git-ignored.
- Audio: no data yet. `scripts/train_audio.py` expects `data/MANIFEST_audio.csv` (e.g. an ASVspoof 2019 LA subset).

## Workflow
```
python scripts/fetch_dff.py --identities 200 --per-identity 2   # small DeepFakeFace subset via range requests
python scripts/fetch_ffhq_stylegan.py                           # FFHQ reals + StyleGAN fakes (parquet row groups)
python scripts/extract_frames.py data/ffpp_videos data/ffpp_frames -n 8   # video -> frames (same source_id per video)
python scripts/build_manifest.py --real data/ffhq --fake stylegan=data/stylegan ffpp=data/ffpp_frames --holdout ffpp
python scripts/train_image.py --augment         # CLIP probe, frequency, judge (features cached in cache/)
python -m eval.run_eval --detector judge        # accuracy, AUC, FPR on real; seen vs unseen fake types
python -m eval.robustness                       # JPEG75/50, downscale, H.264 table
python -m frontend.eval_stats                   # re-run eval for the web Methodology page -> frontend/eval_results.json
python scripts/train_audio.py --manifest data/MANIFEST_audio.csv
python -m eval.ela_calibration                  # ELA splice calibration on local photos
```
`--augment` trains with JPEG/downscale augmentation so scores hold up on compressed input.
Degradations (`emet_lens/degrade.py`): `clean`, `jpeg75`, `jpeg50`, `down0.5`, `h264_crf30`; an MP3 helper exists for audio.

## Results
Held-out test split, judge = CLIP probe + frequency, trained with `--augment`. Real = DeepFakeFace/wiki portraits + FFHQ.
Source: `frontend/eval_results.json`. FPR is on 316 real images; detect rate is the share of fakes called fake at the chosen decision point.

| input quality | FPR on real | seen AUC | seen detect rate | unseen AUC | unseen detect rate |
|---|---|---|---|---|---|
| clean | 3.8% | 0.896 | 55% | 0.844 | 48% |
| JPEG 75 | 8.2% | 0.869 | 63% | 0.792 | 49% |
| JPEG 50 | 8.9% | 0.876 | 63% | 0.819 | 56% |
| downscale 0.5 | 10.4% | 0.831 | 56% | 0.753 | 43% |
| H.264 | not measured | | | | |

Individual detectors on clean input (same split):

| detector | FPR on real | seen AUC | unseen AUC | unseen detect rate |
|---|---|---|---|---|
| CLIP probe | 12.7% | 0.882 | 0.837 | 66% |
| Frequency | 34.2% | 0.663 | 0.585 | 46% |
| Judge (fused) | 3.8% | 0.896 | 0.844 | 48% |

The judge mostly leans on CLIP; frequency alone is weak. Fusion buys a much lower FPR at some cost in detection rate.
Without `--augment` (earlier run): FPR 6.6%, detect rate 52–73% by type. Audio and H.264 results are not measured yet.
On the author's own 9 casual phone photos the model called 6 real and 3 fake, but their true labels are unknown, so that is not an accuracy figure.

## Tests
`pytest` runs 15 fast tests in `tests/test_all.py` (about 3 s). They use synthetic images and stubbed features, so they check
behaviour and wiring, not detection accuracy (that comes from `eval/`).

| Test | Checks |
|---|---|
| `test_ela_heatmap_and_edit_detected` | ELA returns a heatmap; a spliced region scores above a clean image |
| `test_ela_unreliable_on_recompressed_file` | ELA is marked unreliable on low-quality, non-JPEG or tiny files |
| `test_frequency_separates_and_predicts` | frequency detector trains, separates synthetic real/fake, returns score + finding |
| `test_clip_probe_predict_path` | CLIP probe fit/save/load/predict, with CLIP features stubbed |
| `test_audio_windows_flag` | audio split into 3 s windows, time ranges flagged, wav2vec2 embedding stubbed |
| `test_judge_inconclusive_rules` | inconclusive only with no pixel detector; otherwise a definite verdict |
| `test_manifest_split_by_source_and_eval` | split by source, error if a source spans splits |
| `test_eval_dummy_end_to_end` | eval harness runs end to end and reports seen / unseen / FPR |
| `test_analyze_image_pixel_verdict_summary_and_assessment` | pixels drive the verdict; summary and assessment present |
| `test_metadata_ai_marker_is_decisive_but_absence_is_neutral` | AI marker is decisive; missing EXIF is neutral |
| `test_lipsync_synced_vs_mismatched_series` | synced series score real, unrelated series score fake with time ranges; silent / short / no-face input is marked unreliable |
| `test_lipsync_shift_direction` | mismatch pushes towards fake, sync towards real, unreliable does nothing |
| `test_lipsync_video_end_to_end` | synthetic talking video (real face photo, mouth region flickers with the audio): matching audio scores below 0.5, unrelated audio above 0.5 |
| `test_lipsync_status_in_report` | the report carries an explicit IN SYNC / OUT OF SYNC / NOT CHECKED status and the summary mentions it |
| `test_consistency_detector_flags_pasted_face_region` | TCNC's outlier map lights up on a region with different noise and stays quiet on a clean image; it trains and returns a score, finding and heat map |

Not covered: full video analysis with the trained detectors, `degrade.py`, the Gradio app and web frontend, and the real CLIP / wav2vec2 models.

## Project layout
```
emet_lens/   engine: detectors/, judge.py, pipeline.py, metadata.py, assessment.py, report.py, faces.py, degrade.py, manifest.py, data.py, device.py
eval/        run_eval.py (metrics), robustness.py (degradation table), ela_calibration.py
scripts/     fetch_dff.py, fetch_ffhq_stylegan.py, extract_frames.py, build_manifest.py, train_image.py, train_audio.py
frontend/    FastAPI server.py + static HTML pages, eval_stats.py, eval_results.json
app.py       Gradio demo
tests/       test_all.py
data/ checkpoints/ cache/ out/   git-ignored
```
Web frontend API: `/api/health`, `/api/status`, `/api/eval`, `/api/analyze`, `/api/samples`, `/api/analyze-sample/{id}`,
`/api/robustness/{job}`, `/api/heatmap/{job}/{name}`, `/api/media/{job}`. It only calls the engine and never modifies it.

## Status
Done: image pipeline (CLIP probe, frequency, ELA, metadata, judge), definite verdict with confidence and assessment, JSON report,
lip-sync consistency check for video with audio, Gradio demo, web frontend, data fetchers, split-by-source manifest, train/eval/robustness scripts, tests for each detector.

Not done: real-video calibration of the lip-sync threshold, audio detector training and evaluation (code and test exist, no data), H.264 robustness numbers, MP3 audio degradation run,
a test on a real video clip, tests for video / app / degradations, blink / pulse and learned audio-visual sync (SyncNet-style) (stretch goals). Lip-sync is built but only checked on synthetic clips: its thresholds are not calibrated on real talking-head data.

## Limitations
- Lip-sync uses a simple mouth-motion vs loudness correlation. It needs a frontal face in most frames and clear speech, ignores which sound is made (phonemes), and can misfire on head movement, noisy audio, or non-speech sounds. It is a nudge, not proof.
- About half of fakes are missed at the chosen low false-positive setting; generators other than those trained on may do worse.
- Real training faces are Wikipedia portraits and FFHQ; casual phone photos are out of distribution and can be misjudged.
  Check any new domain on a labelled set before relying on it.
- Haar-cascade face detection; falls back to a centre crop when no face is found.
- Frequency and ELA cues weaken or vanish under heavy recompression/downscaling; use `--augment` and read the robustness table.
- Detectors generalise poorly to unseen generators; read the unseen column, not the seen one.
- The ELA score is an untuned heuristic and is not used for the verdict.
- A definite verdict is always given, so check the confidence and the caveats, especially at low confidence.
- Small training set (a few thousand faces); results carry wide uncertainty.

## Ethics
Only use your own face or voice, or licensed data, for self-made fakes. Do not use this tool to harass or target individuals;
a "fake" verdict is a probabilistic signal, not proof.
