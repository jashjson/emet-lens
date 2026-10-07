# Emet Lens rules
- Dev machine is a Mac (M5 Air), no CUDA. Pick `mps` or `cpu` via `emet_lens.device.get_device()`; never hardcode `cuda`.
- Split train/test by source video/person, never by frame.
- Every detector returns score_fake, a plain-English finding, and a heatmap or time range where possible.
- Always report seen vs. unseen fake results and false-positive rate on real content.
- Keep data and checkpoints out of git. Cache features to disk.
- Add a small test for each detector.
- Work one step at a time.
