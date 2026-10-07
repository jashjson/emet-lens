import html
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import traceback
import gradio as gr
from emet_lens.pipeline import analyze, IMG_EXT, VID_EXT, AUD_EXT

CSS = """
.card{border-radius:14px;padding:20px 24px;border:1px solid var(--border-color-primary)}
.verdict{font-size:2.2rem;font-weight:800;margin:0 0 8px}
.big{font-size:3rem;font-weight:800;line-height:1}
.real{background:#e8f6ee;color:#14532d}.fake{background:#fdecec;color:#7f1d1d}.inc{background:#fff6e0;color:#7a4b00}
.err{background:#fdecec;color:#7f1d1d}
.bar{height:8px;border-radius:4px;background:#e5e7eb;margin:6px 0 4px}
.bar>div{height:100%;border-radius:4px}
.sig{padding:10px 0;border-bottom:1px solid #e5e7eb33}
.muted{opacity:.7;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.cell{padding:6px 0}
"""

STYLE = {"likely_real": ("real", "REAL"), "likely_fake": ("fake", "FAKE"), "inconclusive": ("inc", "NO VERDICT")}


def _color(p):
    return "#16a34a" if p < 0.35 else "#d97706" if p < 0.65 else "#dc2626"


def verdict_card(rep):
    cls, label = STYLE[rep["verdict"]]
    a = rep.get("assessment") or {}
    conf = a.get("confidence", "none")
    chip = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626", "none": "#6b7280"}[conf]
    badge = (f'<span style="background:{chip};color:#fff;border-radius:999px;padding:2px 12px;font-size:.9rem;'
             f'font-weight:600;vertical-align:middle">{conf.title()} confidence</span>') if rep["verdict"] != "inconclusive" else ""
    cav = ""
    if rep.get("caveats"):
        cav = '<ul class="muted">' + "".join(f"<li>{html.escape(c)}</li>" for c in rep["caveats"]) + "</ul>"
    score = "" if rep["verdict"] == "inconclusive" else (
        f'<div class="big">{rep["authenticity"]:.0%}</div><div>estimated authenticity (100% = certainly real)</div>')
    return (f'<div class="card {cls}"><p class="verdict">{label} &nbsp;{badge}</p>{score}'
            f'<p>{html.escape(rep["summary"])}</p>{cav}</div>')


def lipsync_card(rep):
    ls = rep.get("lipsync")
    if not ls:
        return ""
    col = {"in_sync": "#16a34a", "out_of_sync": "#dc2626", "not_checked": "#6b7280"}[ls["status"]]
    detail = f' (correlation {ls["correlation"]:.2f}, offset {ls["offset_ms"]:+d} ms)' if "correlation" in ls else ""
    return (f'<div class="card"><div class="muted">Lip-sync</div><div style="color:{col};font-weight:700;font-size:1.2rem">'
            f'{ls["label"]}{detail}</div><div>{html.escape(ls["explanation"])}</div></div>')


def metrics(rep):
    a = rep.get("assessment")
    if not a:
        return ""
    conf_col = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626", "none": "#6b7280"}[a["confidence"]]
    f = a["file"]
    res = f"{f['width']}×{f['height']}" if f.get("width") else "n/a"
    cells = [
        ("Confidence", f'<span style="color:{conf_col};font-weight:700">{a["confidence"].title()}</span>'),
        ("Detector agreement", f'{a["agreement"]:.0%}' if a["detectors_used"] else "n/a"),
        ("Input quality", a["input_quality"].title()),
        ("Face found", "Yes" if a["face_detected"] else "No"),
        ("Resolution", res),
        ("JPEG quality", f"~{f['jpeg_quality']:.0f}" if f.get("jpeg_quality") else "n/a"),
        ("Camera metadata", html.escape(f["camera"]) if f.get("camera") else "None"),
        ("Checks used", ", ".join(a["detectors_used"]) or "none"),
    ]
    grid = "".join(f'<div class="cell"><div class="muted">{k}</div><div>{v}</div></div>' for k, v in cells)
    skipped = ""
    if a["detectors_skipped"]:
        skipped = '<div class="muted" style="margin-top:8px">Skipped: ' + "; ".join(
            f'{html.escape(x["name"])} ({html.escape(x["reason"])})' for x in a["detectors_skipped"]) + "</div>"
    return f'<div class="card"><div class="grid">{grid}</div>{skipped}</div>'


def evidence(rep):
    rows = []
    for s in rep["signals"]:
        p = s["score_fake"]
        unused = s.get("reliable") is False
        extra = ""
        if s.get("time_ranges"):
            extra = '<div class="muted">Suspicious at: ' + ", ".join(f"{a:.0f}–{b:.0f}s" for a, b in s["time_ranges"]) + "</div>"
        rows.append(f'<div class="sig"><b>{html.escape(s["name"])}</b> <span class="muted">{"not used for verdict" if unused else f"fake score {p:.2f}"}</span>'
                    + ("" if unused else f'<div class="bar"><div style="width:{p * 100:.0f}%;background:{_color(p)}"></div></div>') +
                    f''
                    f'<div>{html.escape(s["finding"])}</div>{extra}</div>')
    fl = rep["flagged"]
    if fl["frames"]:
        rows.append('<div class="sig"><b>Most suspicious frames</b><div class="muted">' +
                    ", ".join(f'#{f["frame"]} ({f["time_s"]}s, {f["score_fake"]:.2f})' for f in fl["frames"]) + "</div></div>")
    return '<div class="card">' + "".join(rows) + "</div>"


def run(path):
    if not path:
        return "", "", None, {}
    try:
        rep = analyze(path)
    except Exception as e:  # show a readable message instead of Gradio's bare "Error"
        traceback.print_exc()
        return f'<div class="card err"><b>Could not analyze this file.</b><div>{html.escape(type(e).__name__)}: {html.escape(str(e))}</div></div>', "", None, {}
    heat = next((s["heatmap"] for s in rep["signals"] if "heatmap" in s), None)
    return verdict_card(rep) + lipsync_card(rep) + metrics(rep), evidence(rep), heat, rep


with gr.Blocks(title="Emet Lens") as demo:
    gr.Markdown("# Emet Lens\nDeepfake & digital-forensics check for images, video and audio. "
                "Results are probabilistic evidence, not proof.")
    with gr.Row():
        with gr.Column(scale=1):
            f = gr.File(label="Upload image, video or audio", type="filepath",
                        file_types=sorted(IMG_EXT | VID_EXT | AUD_EXT))
            btn = gr.Button("Analyze", variant="primary")
            preview = gr.Image(label="Preview", visible=False, interactive=False)
        with gr.Column(scale=2):
            card = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Evidence"):
                    ev = gr.HTML()
                with gr.Tab("Heatmap"):
                    heat = gr.Image(label="Where the editing check looks suspicious", interactive=False)
                with gr.Tab("Raw JSON"):
                    js = gr.JSON()

    def show_preview(path):
        ok = bool(path) and os.path.splitext(path)[1].lower() in IMG_EXT
        return gr.update(value=path if ok else None, visible=ok)

    f.change(show_preview, f, preview)
    # Analyze on upload and on the button; one at a time so two clicks never run the models concurrently.
    for trigger in (f.change, btn.click):
        trigger(run, f, [card, ev, heat, js], show_progress="full", concurrency_limit=1)

if __name__ == "__main__":
    from emet_lens.pipeline import default_detectors
    default_detectors()  # warm up: load CLIP before the first request so it doesn't time out
    demo.launch(css=CSS, theme=gr.themes.Soft())
