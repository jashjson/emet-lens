"""Web frontend for the Emet Lens engine. Run from the repo root: python -m frontend.server  (http://127.0.0.1:8000)

Only calls the engine (emet_lens.pipeline / device / degrade) and reads its data files; never modifies them.
Each upload gets its own temp job dir so runs never touch the engine's default out/ folder.
"""
import csv
import datetime
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)  # the engine uses repo-relative paths (checkpoints/, data/)
JOBS = os.path.join(tempfile.gettempdir(), "emet_lens_jobs")
MAX_BYTES = 500 * 1024 * 1024
_lock = threading.Lock()  # one analysis at a time: models are heavy on a laptop

app = FastAPI(title="Emet Lens")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def _engine():
    from emet_lens import pipeline  # lazy so pages load even if the engine import is slow
    return pipeline


# ---------------------------------------------------------------- pages
def _page(name):
    return lambda: FileResponse(os.path.join(HERE, name))


for route, html in {"/": "index.html", "/analyze": "analyze.html", "/how-it-works": "how-it-works.html",
                    "/tools": "tools.html", "/tools/{slug}": "tools.html", "/methodology": "methodology.html",
                    "/report": "report.html"}.items():
    app.add_api_route(route, _page(html), methods=["GET"], include_in_schema=False)


# ---------------------------------------------------------------- status / eval
def _manifest():
    path = os.path.join(REPO, "data", "MANIFEST.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@app.get("/api/health")
def health():
    p = _engine()
    return {"ok": True, "image": sorted(p.IMG_EXT), "video": sorted(p.VID_EXT), "audio": sorted(p.AUD_EXT)}


@app.get("/api/status")
def status():
    """Live, factual state of the engine: what is trained, what data it saw. Nothing here is invented."""
    from emet_lens.device import get_device
    ck = {n: os.path.exists(os.path.join(REPO, "checkpoints", f"{n}.joblib"))
          for n in ("clip_probe", "frequency", "judge", "audio")}
    rows = _manifest()
    by = {}
    for r in rows:
        d = by.setdefault(r["type"], {"label": r["label"], "train": 0, "test": 0})
        d[r["split"]] = d.get(r["split"], 0) + 1
    return {
        "device": get_device(), "checkpoints": ck,
        "dataset": {"total": len(rows), "types": by,
                    "sources": len({r["source_id"] for r in rows}),
                    "train": sum(1 for r in rows if r["split"] == "train"),
                    "test": sum(1 for r in rows if r["split"] == "test")},
    }


@app.get("/api/eval")
def eval_results():
    path = os.path.join(HERE, "eval_results.json")
    if not os.path.exists(path):
        return {"available": False}
    with open(path) as f:
        return {"available": True, **json.load(f)}


# ---------------------------------------------------------------- analysis
def _run(src, jdir, name):
    """Analyze the file at `src` (already inside jdir as input.<ext>); returns the decorated report."""
    p = _engine()
    ext = os.path.splitext(src)[1].lower()
    with _lock:
        t0 = time.time()
        rep = p.analyze(src, out_dir=os.path.join(jdir, "out"))
        rep["elapsed_s"] = round(time.time() - t0, 1)
    job = os.path.basename(jdir)
    for s in rep.get("signals", []):  # local heatmap paths -> served URLs
        if s.get("heatmap"):
            s["heatmap"] = f"/api/heatmap/{job}/{os.path.basename(s['heatmap'])}"
    from emet_lens.device import get_device
    rep["job"] = job
    rep["media_url"] = f"/api/media/{job}"
    rep["media_kind"] = "image" if ext in p.IMG_EXT else "video" if ext in p.VID_EXT else "audio"
    rep["provenance"] = {"filename": name, "bytes": os.path.getsize(src), "analyzed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                         "device": get_device()}
    return rep


def _new_job():
    job = uuid.uuid4().hex
    jdir = os.path.join(JOBS, job)
    os.makedirs(jdir)
    return jdir


@app.post("/api/analyze")
def analyze_upload(file: UploadFile = File(...)):
    p = _engine()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in p.IMG_EXT | p.VID_EXT | p.AUD_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'.")
    jdir = _new_job()
    src = os.path.join(jdir, "input" + ext)
    size = 0
    with open(src, "wb") as f:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(jdir, ignore_errors=True)
                raise HTTPException(413, "File too large (max 500 MB).")
            f.write(chunk)
    try:
        return _run(src, jdir, file.filename)
    except Exception as e:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- held-out samples (test split only)
def _samples():
    rows = [r for r in _manifest() if r["split"] == "test" and os.path.exists(os.path.join(REPO, r["path"]))
            and os.path.splitext(r["path"])[1].lower() in (".jpg", ".jpeg", ".png", ".webp")]
    rng = random.Random(7)  # stable picks so the gallery doesn't reshuffle on every load
    real = [r for r in rows if r["label"] == "real"]
    by_type = {}
    for r in rows:
        if r["label"] == "fake":
            by_type.setdefault(r["type"], []).append(r)
    picks = rng.sample(real, min(3, len(real)))
    for t in sorted(by_type):
        picks += rng.sample(by_type[t], min(1, len(by_type[t])))
    rng.shuffle(picks)
    return picks


def _sample_by_id(sid):
    for r in _samples():
        if hashlib.md5(r["path"].encode()).hexdigest()[:10] == sid:
            return r
    raise HTTPException(404)


@app.get("/api/samples")
def samples():
    """Held-out test images (never used for training). Ground truth is withheld until analysis."""
    return [{"id": hashlib.md5(r["path"].encode()).hexdigest()[:10]} for r in _samples()]


_thumbs = {}


@app.get("/api/sample/{sid}/thumb")
def sample_thumb(sid: str):
    if sid not in _thumbs:
        from PIL import Image
        r = _sample_by_id(sid)
        im = Image.open(os.path.join(REPO, r["path"])).convert("RGB")
        im.thumbnail((360, 360))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82)
        _thumbs[sid] = b.getvalue()
    return Response(_thumbs[sid], media_type="image/jpeg")


@app.post("/api/analyze-sample/{sid}")
def analyze_sample(sid: str):
    r = _sample_by_id(sid)
    jdir = _new_job()
    src = os.path.join(jdir, "input" + os.path.splitext(r["path"])[1].lower())
    shutil.copyfile(os.path.join(REPO, r["path"]), src)
    try:
        rep = _run(src, jdir, os.path.basename(r["path"]))
    except Exception as e:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    rep["ground_truth"] = {"label": r["label"], "type": r["type"]}
    return rep


# ---------------------------------------------------------------- robustness stress-test (images)
@app.post("/api/robustness/{job}")
def robustness(job: str):
    """Re-score the same image after JPEG recompression / downscaling, to show whether the call is stable."""
    p = _engine()
    src = _input_path(job)
    if os.path.splitext(src)[1].lower() not in p.IMG_EXT:
        raise HTTPException(400, "Stress-test is available for images only.")
    from PIL import Image
    from emet_lens.degrade import jpeg, downscale
    variants = [("JPEG 75", lambda im: jpeg(im, 75)), ("JPEG 50", lambda im: jpeg(im, 50)),
                ("Downscale 50%", lambda im: downscale(im, 0.5))]
    img = Image.open(src).convert("RGB")
    out = []
    with _lock:
        for label, fn in variants:
            vp = os.path.join(JOBS, job, "rob_" + label.replace(" ", "_").replace("%", "") + ".png")
            fn(img).save(vp)
            rep = p.analyze_image(vp, out_dir=os.path.join(JOBS, job, "rob_out"))
            out.append({"label": label, "authenticity": rep["authenticity"], "verdict": rep["verdict"]})
    return out


# ---------------------------------------------------------------- job files
def _input_path(job):
    d = os.path.join(JOBS, job)
    if not job.isalnum() or not os.path.isdir(d):
        raise HTTPException(404)
    name = next((n for n in os.listdir(d) if n.startswith("input.")), None)
    if not name:
        raise HTTPException(404)
    return os.path.join(d, name)


@app.get("/api/heatmap/{job}/{name}")
def heatmap(job: str, name: str):
    if not job.isalnum():
        raise HTTPException(404)
    path = os.path.join(JOBS, job, "out", os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/media/{job}")
def media(job: str):
    return FileResponse(_input_path(job))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
