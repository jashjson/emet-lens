"""Run the engine's own evaluation on the held-out test split and save results for the website.

Read-only with respect to the engine: it only imports eval.run_eval. Run from the repo root:
    nice python -m frontend.eval_stats
Writes frontend/eval_results.json after every step, so the Methodology page can show partial results.
"""
import json
import math
import os
import time

from eval.run_eval import evaluate, make_scorer
from emet_lens.manifest import load_manifest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")

STEPS = [  # (label, detector, degradation)
    ("judge/clean", "judge", None),
    ("clip_probe/clean", "clip_probe", None),
    ("frequency/clean", "frequency", None),
    ("judge/jpeg75", "judge", "jpeg75"),
    ("judge/jpeg50", "judge", "jpeg50"),
    ("judge/down0.5", "judge", "down0.5"),
    ("consistency/clean", "consistency", None),
    ("consistency/jpeg75", "consistency", "jpeg75"),
    ("consistency/down0.5", "consistency", "down0.5"),
]


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def main():
    df = load_manifest()
    test = df[df.split == "test"]
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "threshold": 0.5,
           "n_test": int(len(test)), "n_test_real": int((test.label == "real").sum()),
           "n_test_fake": int((test.label == "fake").sum()),
           "train_fake_types": sorted(set(df[(df.split == "train") & (df.label == "fake")]["type"])),
           "results": {}}
    if os.path.exists(OUT) and "--force" not in os.sys.argv:  # keep finished steps, run only what is missing
        out["results"] = json.load(open(OUT)).get("results", {})
    for label, det, deg in STEPS:
        if label in out["results"]:
            continue
        t0 = time.time()
        print("running", label, flush=True)
        res = evaluate(df, make_scorer(det, deg))
        res["seconds"] = round(time.time() - t0)
        out["results"][label] = clean(res)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
