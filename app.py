"""audio2notes GUI server.   Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:7860"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import threading
import traceback
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort
from werkzeug.utils import secure_filename

from audio2notes.engine import Session
from audio2notes.instruments import available as available_instruments, DEFAULT as DEFAULT_INSTRUMENT

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "outputs" / "gui"
WEB = ROOT / "web"
WORK.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB uploads

SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()   # heavy jobs run one at a time


def start_job(fn):
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = dict(state="queued", progress=0.0, stage="Waiting…", result=None, error=None)

    def progress(frac, stage):
        job["progress"] = float(max(job["progress"], min(1.0, frac)))
        job["stage"] = stage

    def run():
        with RUN_LOCK:
            job["state"] = "running"
            try:
                job["result"] = fn(progress)
                job["progress"] = 1.0
                job["state"] = "done"
            except Exception as e:  # report to the UI
                traceback.print_exc()
                job["error"] = f"{type(e).__name__}: {e}"
                job["state"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return jid


def session_or_404(sid):
    s = SESSIONS.get(sid)
    if s is None:
        abort(404, "unknown session - load the file again")
    return s


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(WEB / "vendor", name, max_age=86400)


@app.get("/api/instruments")
def instruments():
    return jsonify(dict(instruments=available_instruments(), default=DEFAULT_INSTRUMENT))


@app.get("/api/config")
def config():
    from dataclasses import asdict
    from audio2notes.transcribe import DETAIL_LEVELS, DETAIL_PRESETS, TranscribeParams
    return jsonify(dict(levels=DETAIL_LEVELS, presets=DETAIL_PRESETS, defaults=asdict(TranscribeParams())))


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        abort(400, "no file")
    up = WORK / "uploads"
    up.mkdir(exist_ok=True)
    name = secure_filename(f.filename) or "audio"
    path = up / f"{uuid.uuid4().hex[:6]}_{name}"
    f.save(path)
    s = Session(path, WORK)
    try:
        info = s.load()
    except Exception as e:
        abort(400, f"could not decode audio: {e}")
    info["name"] = f.filename
    SESSIONS[s.id] = s
    return jsonify(info)


@app.post("/api/extract")
def extract():
    d = request.get_json(force=True)
    s = session_or_404(d.get("sid"))
    a, b, model = float(d["start"]), float(d["end"]), d.get("model", "htdemucs_ft")
    return jsonify(job=start_job(lambda p: s.extract(a, b, model, progress=p)))


@app.post("/api/notate")
def notate():
    d = request.get_json(force=True)
    s = session_or_404(d.get("sid"))
    params = d.get("params", {})
    return jsonify(job=start_job(lambda p: s.notate(params, progress=p)))


@app.post("/api/refine")
def refine():
    d = request.get_json(force=True)
    s = session_or_404(d.get("sid"))
    try:
        return jsonify(s.refine(d["start"], d["end"], d.get("params", {})))
    except ValueError as e:
        abort(400, str(e))


@app.post("/api/render_edited")
def render_edited():
    d = request.get_json(force=True)
    s = session_or_404(d.get("sid"))
    notes, inst, tr = d["notes"], d.get("instrument"), d.get("transpose")
    return jsonify(job=start_job(lambda p: s.render_edited(notes, inst, tr, progress=p)))


@app.get("/api/job/<jid>")
def job(jid):
    j = JOBS.get(jid)
    if j is None:
        abort(404)
    return jsonify(j)


@app.get("/media/<sid>/<path:name>")
def media(sid, name):
    s = session_or_404(sid)
    return send_from_directory(s.dir, name, conditional=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"audio2notes GUI -> {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
