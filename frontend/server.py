"""Thin web frontend for the Emet Lens engine. Run: python -m frontend.server  (http://127.0.0.1:8000)

Only calls emet_lens.pipeline.analyze(); does not modify the engine. Each upload gets its own temp
job dir so concurrent/other runs never touch the engine's default out/ folder.
"""
import os
import shutil
import tempfile
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(tempfile.gettempdir(), "emet_lens_jobs")
MAX_BYTES = 500 * 1024 * 1024
_lock = threading.Lock()  # one analysis at a time: models are heavy on a laptop

app = FastAPI(title="Emet Lens")


def _engine():
    from emet_lens import pipeline  # lazy so the page loads even if the engine import is slow
    return pipeline


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/health")
def health():
    p = _engine()
    return {"ok": True, "image": sorted(p.IMG_EXT), "video": sorted(p.VID_EXT), "audio": sorted(p.AUD_EXT)}


@app.post("/api/analyze")
def analyze_upload(file: UploadFile = File(...)):
    p = _engine()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in p.IMG_EXT | p.VID_EXT | p.AUD_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'.")
    job = uuid.uuid4().hex
    jdir = os.path.join(JOBS, job)
    os.makedirs(jdir)
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
        with _lock:
            rep = p.analyze(src, out_dir=os.path.join(jdir, "out"))
    except Exception as e:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    for s in rep.get("signals", []):  # local heatmap paths -> served URLs
        if s.get("heatmap"):
            s["heatmap"] = f"/api/heatmap/{job}/{os.path.basename(s['heatmap'])}"
    rep["media_url"] = f"/api/media/{job}"
    rep["media_kind"] = "image" if ext in p.IMG_EXT else "video" if ext in p.VID_EXT else "audio"
    return rep


def _job_file(job, *parts):
    if not job.isalnum():
        raise HTTPException(404)
    path = os.path.join(JOBS, job, *parts)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return path


@app.get("/api/heatmap/{job}/{name}")
def heatmap(job: str, name: str):
    return FileResponse(_job_file(job, "out", os.path.basename(name)))


@app.get("/api/media/{job}")
def media(job: str):
    d = os.path.join(JOBS, job)
    if not job.isalnum() or not os.path.isdir(d):
        raise HTTPException(404)
    name = next((n for n in os.listdir(d) if n.startswith("input.")), None)
    if not name:
        raise HTTPException(404)
    return FileResponse(os.path.join(d, name))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
