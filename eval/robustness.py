"""Score-vs-quality table: re-run eval under JPEG 75/50, downscale, H.264. Prints markdown."""
import argparse
from emet_lens.manifest import load_manifest
from emet_lens.degrade import DEGRADATIONS
from eval.run_eval import evaluate, make_scorer

ap = argparse.ArgumentParser()
ap.add_argument("--detectors", nargs="*", default=["clip_probe", "frequency", "ela", "judge"])
ap.add_argument("--manifest", default="data/MANIFEST.csv")
a = ap.parse_args()
df = load_manifest(a.manifest)
print("| detector | degradation | FPR real | seen AUC | unseen AUC | unseen detect rate |\n|---|---|---|---|---|---|")
for d in a.detectors:
    for deg in DEGRADATIONS:
        r = evaluate(df, make_scorer(d, None if deg == "clean" else deg))
        g = lambda k, f: f"{r[k][f]:.3f}" if f in r[k] else "n/a"
        print(f"| {d} | {deg} | {r['fpr_real']:.3f} | {g('seen','auc')} | {g('unseen','auc')} | {g('unseen','tpr_fake')} |")
