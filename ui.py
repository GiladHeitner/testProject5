"""Lightweight local web UI for shorts_bot.py.

Run:
    python3 ui.py
Then open http://127.0.0.1:5005 in your browser.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
HOOK_DIR = OUTPUT_DIR / "hook_image"

app = Flask(__name__)

_run_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "pid": None,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "cmd": None,
    "log": [],
    "proc": None,
}
_log_queue: "queue.Queue[str]" = queue.Queue()
_subscribers: list["queue.Queue[str]"] = []
_subscribers_lock = threading.Lock()


def _broadcast(line: str) -> None:
    with _subscribers_lock:
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait(line)
        except Exception:
            pass


def _stream_process(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip("\n")
        _state["log"].append(line)
        if len(_state["log"]) > 4000:
            del _state["log"][:1000]
        _broadcast(line)
    proc.stdout.close()
    proc.wait()
    _state["running"] = False
    _state["pid"] = None
    _state["exit_code"] = proc.returncode
    _state["finished_at"] = time.time()
    _state["proc"] = None
    _broadcast(f"__END__ exit={proc.returncode}")


def _build_cmd(payload: dict[str, Any]) -> list[str]:
    cmd: list[str] = [sys.executable, str(PROJECT_ROOT / "shorts_bot.py")]

    def add_flag(key: str, flag: str) -> None:
        if bool(payload.get(key)):
            cmd.append(flag)

    def add_val(key: str, flag: str) -> None:
        value = payload.get(key)
        if value is None or value == "":
            return
        cmd.extend([flag, str(value)])

    add_val("words", "--words")
    add_val("topic", "--topic")
    add_val("tts", "--tts")
    add_val("privacy", "--privacy")
    add_val("duration_seconds", "--duration-seconds")
    add_val("speed_ramp_ms", "--speed-ramp-ms")
    add_val("speed_slow", "--speed-slow")
    add_val("speed_fast", "--speed-fast")
    add_val("popup_sfx_volume", "--popup-sfx-volume")
    add_val("popup_sfx_speed", "--popup-sfx-speed")
    add_val("popup_sfx_trim_seconds", "--popup-sfx-trim-seconds")
    add_val("bgm_volume", "--bgm-volume")
    add_val("gameplay_top_crop", "--gameplay-top-crop")
    add_val("script", "--script")

    add_flag("dynamic_speed", "--dynamic-speed")
    add_flag("generate_images", "--generate-images")
    add_flag("images_only", "--images-only")
    add_flag("skip_tts", "--skip-tts")
    add_flag("video_only", "--video-only")
    add_flag("quick_test", "--quick-test")
    add_flag("no_description", "--no-description")
    add_flag("upload", "--upload")
    return cmd


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    with _run_lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Already running."}), 409

        cmd = _build_cmd(payload)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["SHORTS_BOT_INTERACTIVE"] = "1"
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

        _state.update(
            {
                "running": True,
                "pid": proc.pid,
                "started_at": time.time(),
                "finished_at": None,
                "exit_code": None,
                "cmd": " ".join(shlex.quote(c) for c in cmd),
                "log": [],
                "proc": proc,
            }
        )
        threading.Thread(target=_stream_process, args=(proc,), daemon=True).start()
        return jsonify({"ok": True, "pid": proc.pid, "cmd": _state["cmd"]})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    pid = _state.get("pid")
    if not pid:
        return jsonify({"ok": False, "error": "Not running."}), 400
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return jsonify({"ok": True})


@app.route("/api/stdin", methods=["POST"])
def api_stdin():
    proc = _state.get("proc")
    if proc is None or proc.stdin is None or proc.poll() is not None:
        return jsonify({"ok": False, "error": "Not running."}), 400
    body = request.get_json(force=True) or {}
    text = str(body.get("text", ""))
    if not text.endswith("\n"):
        text += "\n"
    try:
        proc.stdin.write(text)
        proc.stdin.flush()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "running": _state["running"],
            "pid": _state["pid"],
            "exit_code": _state["exit_code"],
            "cmd": _state["cmd"],
            "started_at": _state["started_at"],
            "finished_at": _state["finished_at"],
            "log_tail": _state["log"][-50:],
        }
    )


@app.route("/api/stream")
def api_stream() -> Response:
    q: "queue.Queue[str]" = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)

    def gen() -> Iterator[str]:
        try:
            for line in _state["log"][-200:]:
                yield f"data: {json.dumps(line)}\n\n"
            while True:
                try:
                    line = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(line)}\n\n"
        finally:
            with _subscribers_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


def _latest_file(folder: Path, suffixes: set[str]) -> Path | None:
    if not folder.exists():
        return None
    matches = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


@app.route("/api/latest")
def api_latest():
    hook = _latest_file(HOOK_DIR, {".png", ".jpg", ".jpeg", ".webp"})
    video = OUTPUT_DIR / "short.mp4"
    script_file = OUTPUT_DIR / "script.txt"
    metadata_file = OUTPUT_DIR / "metadata.txt"
    return jsonify(
        {
            "hook_image": f"/output/{hook.relative_to(OUTPUT_DIR).as_posix()}" if hook else None,
            "video": "/output/short.mp4" if video.exists() else None,
            "script": script_file.read_text(encoding="utf-8") if script_file.exists() else "",
            "metadata": metadata_file.read_text(encoding="utf-8") if metadata_file.exists() else "",
        }
    )


@app.route("/output/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Shorts Bot · Dashboard</title>
<style>
  :root {
    --bg: #0a0c12;
    --bg-2: #0d1018;
    --panel: #11141c;
    --panel-2: #151926;
    --panel-3: #1a1f2e;
    --border: #242938;
    --border-hi: #313749;
    --text: #e7ecf3;
    --text-dim: #b6bdcd;
    --muted: #7e879b;
    --accent: #7c5cff;
    --accent-2: #22d3ee;
    --green: #34d399;
    --red: #f87171;
    --yellow: #fbbf24;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased; }
  body::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
      radial-gradient(800px 400px at -10% -10%, #7c5cff26, transparent 60%),
      radial-gradient(800px 400px at 110% -10%, #22d3ee1f, transparent 60%);
  }
  a { color: var(--accent-2); text-decoration: none; }

  /* layout */
  .app { position: relative; z-index: 1; display: grid;
    grid-template-columns: 260px 1fr; grid-template-rows: 56px 1fr;
    grid-template-areas: "brand header" "nav main";
    min-height: 100vh; }
  .brand { grid-area: brand; background: var(--panel); border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px;
    padding: 0 18px; }
  .logo { width: 26px; height: 26px; border-radius: 8px;
    background: conic-gradient(from 0deg, #7c5cff, #22d3ee, #34d399, #7c5cff);
    box-shadow: 0 0 20px #7c5cff66; }
  .brand h1 { font-size: 14px; margin: 0; letter-spacing: 0.3px; }
  .brand small { color: var(--muted); display: block; font-size: 11px; margin-top: -1px; }

  .header { grid-area: header; background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 0 22px; }
  .header .left { display: flex; align-items: center; gap: 12px; min-width: 0; }
  .header .right { display: flex; align-items: center; gap: 8px; }
  .cmd { font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted);
    word-break: break-all; max-width: 60vw; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* nav (left sidebar tabs) */
  .nav { grid-area: nav; background: var(--panel); border-right: 1px solid var(--border);
    padding: 14px 10px; display: flex; flex-direction: column; gap: 2px; overflow: auto; }
  .nav button { background: transparent; border: 1px solid transparent; color: var(--text-dim);
    text-align: left; padding: 10px 12px; border-radius: 10px; font: inherit; cursor: pointer;
    display: flex; align-items: center; gap: 10px; }
  .nav button:hover { background: var(--panel-2); color: var(--text); }
  .nav button.active { background: linear-gradient(135deg, #7c5cff14, #22d3ee14);
    border-color: var(--border-hi); color: var(--text); }
  .nav .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); flex: 0 0 auto; }
  .nav button.active .dot { background: var(--accent); box-shadow: 0 0 10px var(--accent); }
  .nav .divider { height: 1px; background: var(--border); margin: 10px 6px; }
  .nav .run-block { padding: 8px 6px; display: flex; flex-direction: column; gap: 8px; margin-top: auto; }

  /* main */
  .main { grid-area: main; padding: 18px 22px 24px;
    display: grid; grid-template-rows: auto auto 1fr; gap: 14px; min-width: 0; }

  /* stats row */
  .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  .steps { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
  .step { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 12px; display: flex; align-items: center; gap: 8px; position: relative; overflow: hidden; }
  .step .sn { width: 22px; height: 22px; border-radius: 50%; background: var(--panel-3);
    color: var(--muted); font-size: 11px; display: flex; align-items: center; justify-content: center; }
  .step .nm { font-size: 12px; color: var(--muted); letter-spacing: 0.2px; }
  .step.done { border-color: #1f3a2e; background: #0e1b15; }
  .step.done .sn { background: var(--green); color: #0a120f; }
  .step.done .nm { color: var(--green); }
  .step.active { border-color: var(--accent); background: linear-gradient(135deg, #7c5cff1f, #22d3ee1f); }
  .step.active .sn { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #0a0c12; }
  .step.active .nm { color: var(--text); }
  .step.active::after {
    content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    animation: slide 1.6s infinite linear;
  }
  @keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

  .progress { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
  .progress .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .progress .bar { position: relative; height: 10px; background: var(--panel-3);
    border-radius: 999px; overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.35); }
  .progress .fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.35s ease, box-shadow 0.4s ease;
    box-shadow: 0 0 8px -1px var(--accent); }
  .progress .fill::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg,
      transparent 0%, rgba(255,255,255,0.0) 30%,
      rgba(255,255,255,0.45) 50%, rgba(255,255,255,0.0) 70%, transparent 100%);
    background-size: 200% 100%;
    animation: shimmer 1.6s infinite linear;
    pointer-events: none;
  }
  .progress .fill.full { box-shadow: 0 0 14px var(--accent); }
  .progress .fill.full::after { animation: none; }
  @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
  .progress .lbl { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
  .progress .val { font: 12px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text); }
  .tier-badge { font-size: 10px; letter-spacing: 0.5px; text-transform: uppercase;
    padding: 2px 8px; border-radius: 999px;
    border: 1px solid rgba(124,92,255,0.4);
    background: rgba(124,92,255,0.12); color: #cfc6ff; }
  .tier-badge.t1 { border-color: #6b7280; background: rgba(120,128,140,0.15); color: #c8cfdb; }
  .tier-badge.t2 { border-color: #4caee5; background: rgba(76,174,229,0.15); color: #cfeaff; }
  .tier-badge.t3 { border-color: #b06bff; background: rgba(176,107,255,0.18); color: #ead8ff; }
  .tier-badge.t4 { border-color: #50e3c2; background: rgba(80,227,194,0.18); color: #cdf8ee; }
  .tier-badge.t5 { border-color: #ffd166; background: rgba(255,209,102,0.18); color: #fff0c8; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; position: relative; overflow: hidden; }
  .stat::before { content: ""; position: absolute; inset: 0; background:
    radial-gradient(200px 80px at 10% 0%, #7c5cff1a, transparent 60%); pointer-events: none; }
  .stat .label { font-size: 11px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
  .stat .value { font-size: 18px; font-weight: 600; margin-top: 4px; color: var(--text); }
  .stat .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* content: preview + log */
  .content { display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; min-height: 0; }

  /* generic card */
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .card > header { padding: 10px 14px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .card > header h3 { margin: 0; font-size: 12px; letter-spacing: 1.2px; color: var(--muted);
    text-transform: uppercase; }
  .card > .body { padding: 14px; flex: 1; min-height: 0; overflow: auto; }

  /* tabs inside preview */
  .tabs { display: flex; gap: 2px; background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 2px; }
  .tabs button { background: transparent; border: none; color: var(--muted); padding: 6px 10px;
    border-radius: 8px; font: inherit; cursor: pointer; font-size: 12px; }
  .tabs button.active { background: var(--panel-3); color: var(--text); }

  .preview-grid { display: grid; gap: 12px; grid-template-columns: 1fr 1fr; }
  .preview-grid .cell { display: flex; flex-direction: column; gap: 8px; }
  .preview-grid img, .preview-grid video { width: 100%; max-height: 380px; border-radius: 12px;
    background: #000; border: 1px solid var(--border); object-fit: contain; }
  .pane-text { white-space: pre-wrap; word-break: break-word; color: var(--text-dim);
    font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
  .placeholder { color: var(--muted); padding: 24px; text-align: center; font-size: 13px;
    background: var(--bg-2); border: 1px dashed var(--border); border-radius: 12px; }

  .log { padding: 12px 14px; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--bg-2); color: #cfd6e3; overflow: auto; flex: 1; white-space: pre-wrap;
    word-break: break-word; }
  .log .ok { color: var(--green); } .log .bad { color: var(--red); }

  /* form views */
  .views > section { display: none; }
  .views > section.active { display: block; }
  .card .body .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 14px; }
  .field { display: flex; flex-direction: column; gap: 6px; padding: 6px 0; }
  .field label { font-size: 12px; color: var(--muted); letter-spacing: 0.3px; }
  .field .hint { font-size: 11px; color: var(--muted); }
  .field .row-inline { display: flex; align-items: center; justify-content: space-between; gap: 10px; }

  input[type="text"], input[type="number"], textarea, select {
    background: var(--bg-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 10px; padding: 9px 11px; width: 100%; font: inherit; }
  textarea { min-height: 90px; resize: vertical; }
  input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px #7c5cff1f; }

  /* toggle switch */
  .switch { position: relative; width: 42px; height: 24px; display: inline-block; flex: 0 0 auto; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: #2a2f3e; border-radius: 24px;
    transition: .18s; cursor: pointer; }
  .slider::before { content: ""; position: absolute; width: 18px; height: 18px; top: 3px; left: 3px;
    background: #e7ecf3; border-radius: 50%; transition: .18s; }
  .switch input:checked + .slider { background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
  .switch input:checked + .slider::before { transform: translateX(18px); }

  .btn { border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    padding: 9px 14px; border-radius: 10px; font: inherit; cursor: pointer; transition: .15s; }
  .btn:hover { border-color: var(--accent); }
  .btn.primary { background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border: none; color: #0a0c12; font-weight: 600; box-shadow: 0 8px 24px -10px var(--accent); }
  .btn.danger { background: transparent; color: var(--red); border-color: #5d3434; }
  .btn.ghost { background: transparent; }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-row { display: flex; gap: 8px; }

  .pill { font-size: 11px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--panel-2); color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }
  .pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: .7; }
  .pill.run { color: var(--green); border-color: #1f3a2e; background: #0e1b15; }
  .pill.err { color: var(--red); border-color: #3a1f1f; background: #1b0e0e; }
  .pill.wait { color: var(--yellow); border-color: #3a2f1f; background: #1b160c; }
  .pill.idle { color: var(--muted); }

  /* prompt bar */
  .prompt { background: linear-gradient(135deg, #7c5cff14, #22d3ee14);
    border: 1px solid var(--accent); border-radius: 14px; padding: 12px 16px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .prompt .left { display: flex; align-items: center; gap: 10px; }

  .flex { display: flex; align-items: center; gap: 8px; }
  .sep { height: 1px; background: var(--border); margin: 14px 0; }

  @media (max-width: 1100px) {
    .app { grid-template-columns: 64px 1fr; }
    .brand h1, .brand small { display: none; }
    .nav button span.label { display: none; }
    .stats { grid-template-columns: 1fr 1fr; }
    .content { grid-template-columns: 1fr; }
    .preview-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="brand">
    <div class="logo"></div>
    <div>
      <h1>Shorts Bot</h1>
      <small>Dashboard</small>
    </div>
  </div>

  <div class="header">
    <div class="left">
      <span class="pill idle" id="statusPill">Idle</span>
      <span class="cmd" id="cmdDisplay">No run yet</span>
    </div>
    <div class="right">
      <button class="btn ghost" id="refreshBtn">Refresh</button>
      <button class="btn danger" id="stopBtn" disabled>Stop</button>
      <button class="btn primary" id="runBtn">Run</button>
    </div>
  </div>

  <nav class="nav" id="nav">
    <button data-view="mode" class="active"><span class="dot"></span><span class="label">Mode</span></button>
    <button data-view="script"><span class="dot"></span><span class="label">Script</span></button>
    <button data-view="audio"><span class="dot"></span><span class="label">Audio</span></button>
    <button data-view="video"><span class="dot"></span><span class="label">Video</span></button>
    <button data-view="upload"><span class="dot"></span><span class="label">Upload</span></button>
    <div class="divider"></div>
    <button data-view="preview"><span class="dot"></span><span class="label">Preview</span></button>
    <button data-view="logs"><span class="dot"></span><span class="label">Logs</span></button>
  </nav>

  <main class="main">
    <!-- Stats -->
    <section class="stats">
      <div class="stat">
        <div class="label">Status</div>
        <div class="value" id="statStatus">Idle</div>
        <div class="sub" id="statStatusSub">Waiting to run</div>
      </div>
      <div class="stat">
        <div class="label">Elapsed</div>
        <div class="value" id="statElapsed">0s</div>
        <div class="sub" id="statElapsedSub">Running time</div>
      </div>
      <div class="stat">
        <div class="label">Video</div>
        <div class="value" id="statVideo">—</div>
        <div class="sub" id="statVideoSub">output/short.mp4</div>
      </div>
      <div class="stat">
        <div class="label">Hook image</div>
        <div class="value" id="statHook">—</div>
        <div class="sub" id="statHookSub">output/hook_image/</div>
      </div>
    </section>

    <!-- Step pipeline -->
    <section class="steps" id="steps">
      <div class="step" data-step="1"><span class="sn">1</span><span class="nm">Script</span></div>
      <div class="step" data-step="2"><span class="sn">2</span><span class="nm">Voiceover</span></div>
      <div class="step" data-step="3"><span class="sn">3</span><span class="nm">Subtitles</span></div>
      <div class="step" data-step="4"><span class="sn">4</span><span class="nm">Popups</span></div>
      <div class="step" data-step="5"><span class="sn">5</span><span class="nm">Render</span></div>
      <div class="step" data-step="6"><span class="sn">6</span><span class="nm">Metadata</span></div>
    </section>

    <!-- Render sub-progress -->
    <section class="progress" id="renderProgress" style="display:none;">
      <div class="row">
        <span class="lbl">Rendering short.mp4</span>
        <span class="val" id="renderStats">waiting…</span>
      </div>
      <div class="bar"><div class="fill" id="renderFill"></div></div>
    </section>

    <!-- Generic step sub-progress (TTS chunks, popup fetches, etc.) -->
    <section class="progress" id="subProgress" style="display:none;">
      <div class="row">
        <span class="lbl">
          <span class="tier-badge t1" id="subTier">starting</span>
          <span id="subLabel">Working…</span>
        </span>
        <span class="val" id="subStats">0%</span>
      </div>
      <div class="bar"><div class="fill" id="subFill"></div></div>
    </section>

    <!-- Prompt bar -->
    <div class="prompt" id="promptBar" style="display:none;">
      <div class="left">
        <span class="pill wait">Waiting</span>
        <span id="promptText">Use this script?</span>
      </div>
      <div class="btn-row">
        <button class="btn primary" id="acceptBtn">Accept</button>
        <button class="btn danger" id="regenBtn">Regenerate</button>
      </div>
    </div>

    <!-- Main content: config card (switches views) + preview/log -->
    <div class="content">
      <article class="card">
        <header>
          <h3 id="configTitle">Configuration</h3>
          <div class="tabs" id="previewTabs" style="display:none;">
            <button class="active" data-tab="preview">Preview</button>
            <button data-tab="script_text">Script</button>
            <button data-tab="metadata">Metadata</button>
          </div>
        </header>
        <div class="body views" id="views">

          <section data-view="mode" class="active">
            <div class="grid-2">
              <div class="field"><div class="row-inline"><label>Images only (fast test)</label>
                <label class="switch"><input type="checkbox" id="images_only"><span class="slider"></span></label></div>
                <div class="hint">Generates only the hook image, skipping TTS/video.</div></div>
              <div class="field"><div class="row-inline"><label>Generate popup images</label>
                <label class="switch"><input type="checkbox" id="generate_images"><span class="slider"></span></label></div>
                <div class="hint">AI-generate popup assets for the video.</div></div>
              <div class="field"><div class="row-inline"><label>Video only (reuse narration)</label>
                <label class="switch"><input type="checkbox" id="video_only"><span class="slider"></span></label></div>
                <div class="hint">Re-render the final video from existing audio/subs.</div></div>
              <div class="field"><div class="row-inline"><label>Skip TTS</label>
                <label class="switch"><input type="checkbox" id="skip_tts"><span class="slider"></span></label></div>
                <div class="hint">Keep previous narration.mp3 as-is.</div></div>
              <div class="field"><div class="row-inline"><label>Quick test (3s)</label>
                <label class="switch"><input type="checkbox" id="quick_test"><span class="slider"></span></label></div>
                <div class="hint">Quick smoke test of the render pipeline.</div></div>
              <div class="field"><div class="row-inline"><label>Dynamic speed ramps</label>
                <label class="switch"><input type="checkbox" id="dynamic_speed" checked><span class="slider"></span></label></div>
                <div class="hint">Slow-mo on highlighted beats, snap back to normal.</div></div>
            </div>
          </section>

          <section data-view="script">
            <div class="grid-2">
              <div class="field"><label>Target words</label>
                <input type="number" id="words" value="100" min="15" max="250">
                <div class="hint">100 ≈ ~30s narrated.</div></div>
              <div class="field"><label>Topic (optional)</label>
                <input type="text" id="topic" placeholder="e.g. people blasting speakerphone in public"></div>
            </div>
            <div class="sep"></div>
            <div class="field"><label>Script override (for images-only)</label>
              <textarea id="script" placeholder="Leave empty to use output/script.txt"></textarea>
              <div class="hint">Only used when running images-only mode.</div></div>
          </section>

          <section data-view="audio">
            <div class="grid-2">
              <div class="field"><label>TTS engine</label>
                <select id="tts"><option value="cloner">cloner (local Adam)</option><option value="openai">openai</option></select></div>
              <div class="field"><label>BGM volume</label>
                <input type="number" id="bgm_volume" step="0.01" value="0.25"></div>
            </div>
            <div class="sep"></div>
            <div class="grid-2">
              <div class="field"><label>Speed slow</label>
                <input type="number" id="speed_slow" step="0.05" value="0.60"></div>
              <div class="field"><label>Speed fast</label>
                <input type="number" id="speed_fast" step="0.05" value="1.15"></div>
              <div class="field"><label>Ramp (ms)</label>
                <input type="number" id="speed_ramp_ms" value="600"></div>
              <div class="field"><label>SFX volume</label>
                <input type="number" id="popup_sfx_volume" step="0.05" value="0.55"></div>
              <div class="field"><label>SFX speed</label>
                <input type="number" id="popup_sfx_speed" step="0.05" value="1.25"></div>
              <div class="field"><label>SFX trim (s)</label>
                <input type="number" id="popup_sfx_trim_seconds" step="0.1" value="1.4"></div>
            </div>
          </section>

          <section data-view="video">
            <div class="grid-2">
              <div class="field"><label>Gameplay top crop (px)</label>
                <input type="number" id="gameplay_top_crop" value="96"></div>
              <div class="field"><label>Duration (s, optional)</label>
                <input type="number" id="duration_seconds" step="0.1" placeholder="full narration"></div>
            </div>
          </section>

          <section data-view="upload">
            <div class="grid-2">
              <div class="field"><div class="row-inline"><label>Upload to YouTube</label>
                <label class="switch"><input type="checkbox" id="upload"><span class="slider"></span></label></div></div>
              <div class="field"><div class="row-inline"><label>No description</label>
                <label class="switch"><input type="checkbox" id="no_description" checked><span class="slider"></span></label></div></div>
              <div class="field"><label>Privacy</label>
                <select id="privacy">
                  <option value="public">public</option>
                  <option value="unlisted">unlisted</option>
                  <option value="private">private</option>
                </select></div>
            </div>
          </section>

          <section data-view="preview">
            <div id="previewPanes">
              <div id="previewPane_preview">
                <div class="preview-grid" id="previewGrid">
                  <div class="placeholder">No output yet.</div>
                </div>
              </div>
              <div id="previewPane_script_text" style="display:none;">
                <pre class="pane-text" id="paneScript">No script yet.</pre>
              </div>
              <div id="previewPane_metadata" style="display:none;">
                <pre class="pane-text" id="paneMetadata">No metadata yet.</pre>
              </div>
            </div>
          </section>

          <section data-view="logs">
            <pre class="log" id="logInline">Open the Logs view once a run has started.</pre>
          </section>
        </div>
      </article>

      <aside class="card">
        <header>
          <h3>Live log</h3>
          <span class="pill" id="logCount">0 lines</span>
        </header>
        <pre class="log" id="log"></pre>
      </aside>
    </div>
  </main>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const runBtn = $("runBtn"), stopBtn = $("stopBtn"), logEl = $("log"), logInline = $("logInline");
  const logCount = $("logCount");
  const statusPill = $("statusPill"), cmdDisplay = $("cmdDisplay");

  // Nav view switching
  const views = document.querySelectorAll("#views > section");
  const navBtns = document.querySelectorAll("#nav button[data-view]");
  const viewTitle = {
    mode: "Mode", script: "Script", audio: "Audio", video: "Video", upload: "Upload",
    preview: "Output preview", logs: "Logs"
  };
  function showView(name) {
    navBtns.forEach(b => b.classList.toggle("active", b.dataset.view === name));
    views.forEach(v => v.classList.toggle("active", v.dataset.view === name));
    $("configTitle").textContent = viewTitle[name] || "Configuration";
    $("previewTabs").style.display = name === "preview" ? "flex" : "none";
  }
  navBtns.forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));

  // Preview sub-tabs
  const pTabs = document.querySelectorAll("#previewTabs button");
  pTabs.forEach(t => t.addEventListener("click", () => {
    pTabs.forEach(x => x.classList.toggle("active", x === t));
    ["preview","script_text","metadata"].forEach(k => {
      $("previewPane_" + k).style.display = (k === t.dataset.tab) ? "block" : "none";
    });
  }));

  // Inputs
  const inputIds = [
    "words","topic","tts","privacy","duration_seconds","speed_ramp_ms","speed_slow","speed_fast",
    "popup_sfx_volume","popup_sfx_speed","popup_sfx_trim_seconds","bgm_volume","gameplay_top_crop",
    "script",
    "dynamic_speed","generate_images","images_only","skip_tts","video_only","quick_test",
    "no_description","upload"
  ];
  function collect() {
    const data = {};
    for (const id of inputIds) {
      const el = $(id); if (!el) continue;
      if (el.type === "checkbox") data[id] = el.checked;
      else data[id] = el.value;
    }
    return data;
  }

  function setStatus(label, kind) {
    statusPill.textContent = label;
    statusPill.className = "pill " + (kind || "idle");
    $("statStatus").textContent = label;
    $("statStatusSub").textContent = kind === "run" ? "Bot is running" : (kind === "err" ? "Last run failed" : "Waiting to run");
  }

  // Step indicators
  const stepEls = () => document.querySelectorAll("#steps .step");
  function resetSteps() {
    stepEls().forEach(el => { el.classList.remove("active", "done"); });
    $("renderProgress").style.display = "none";
    $("renderFill").style.width = "0%";
    $("renderStats").textContent = "waiting…";
    $("subProgress").style.display = "none";
    $("subFill").style.width = "0%";
    $("subFill").classList.remove("full");
    $("subStats").textContent = "0%";
    $("subLabel").textContent = "Working…";
    const tierEl = $("subTier");
    tierEl.classList.remove("t1","t2","t3","t4","t5");
    tierEl.classList.add("t1");
    tierEl.textContent = "starting";
  }
  function setStep(n) {
    stepEls().forEach(el => {
      const k = parseInt(el.dataset.step, 10);
      el.classList.toggle("active", k === n);
      el.classList.toggle("done", k < n);
    });
  }
  function finishAllSteps() {
    stepEls().forEach(el => { el.classList.remove("active"); el.classList.add("done"); });
  }

  // Elapsed timer
  let runStartTs = null, elapsedHandle = null;
  function fmtElapsed(ms) {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60), r = s % 60;
    return m > 0 ? `${m}m ${r}s` : `${s}s`;
  }
  function startElapsed() {
    runStartTs = Date.now();
    clearInterval(elapsedHandle);
    elapsedHandle = setInterval(() => {
      $("statElapsed").textContent = fmtElapsed(Date.now() - runStartTs);
    }, 500);
  }
  function stopElapsed() {
    clearInterval(elapsedHandle); elapsedHandle = null;
    if (runStartTs) $("statElapsedSub").textContent = "Finished in " + fmtElapsed(Date.now() - runStartTs);
  }

  // Prompt bar
  const promptBar = $("promptBar"), promptText = $("promptText");
  function showPrompt(text) { promptText.textContent = text || "Use this script?"; promptBar.style.display = "flex"; }
  function hidePrompt() { promptBar.style.display = "none"; }
  async function sendStdin(text) {
    try {
      await fetch("/api/stdin", { method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({text}) });
    } catch (e) {}
    hidePrompt();
  }
  $("acceptBtn").addEventListener("click", () => sendStdin("Y"));
  $("regenBtn").addEventListener("click", () => sendStdin("N"));

  function appendLog(line) {
    const div = document.createElement("div");
    if (line.startsWith("__END__")) div.className = line.includes("exit=0") ? "ok" : "bad";
    div.textContent = line.replace(/^__END__\s*/, "Finished: ");
    logEl.appendChild(div.cloneNode(true));
    logEl.scrollTop = logEl.scrollHeight;
    logInline.appendChild(div);
    logInline.scrollTop = logInline.scrollHeight;
    logCount.textContent = logEl.children.length + " lines";

    // Prompt bar
    if (/Use this script\? \(Y\/N\)/.test(line)) showPrompt("Use this script?");
    if (/Regenerating/.test(line)) hidePrompt();

    // Sub-progress (e.g. "[sub] [###----] 25% (3/12) Fetching popup image: 'pickles'").
    // Handle BEFORE main step tracker so its (N/M) doesn't overwrite the step.
    if (line.startsWith("[sub] ")) {
      $("subProgress").style.display = "flex";
      const pctM = line.match(/(\d+)%\s*\(/);
      const subM = line.match(/\((\d+)\/(\d+)\)\s+(.+)$/);
      const fillEl = $("subFill");
      if (pctM) {
        const pct = Math.min(100, parseInt(pctM[1], 10));
        fillEl.style.width = pct + "%";
        $("subStats").textContent = pct + "%";
        if (pct >= 100) fillEl.classList.add("full");
        else fillEl.classList.remove("full");
        const tierEl = $("subTier");
        tierEl.classList.remove("t1","t2","t3","t4","t5");
        if (pct < 20)      { tierEl.classList.add("t1"); tierEl.textContent = "starting";    }
        else if (pct < 50) { tierEl.classList.add("t2"); tierEl.textContent = "in progress"; }
        else if (pct < 80) { tierEl.classList.add("t3"); tierEl.textContent = "halfway";     }
        else if (pct < 100){ tierEl.classList.add("t4"); tierEl.textContent = "almost there";}
        else               { tierEl.classList.add("t5"); tierEl.textContent = "done";        }
      }
      if (subM) $("subLabel").textContent = subM[3];
      return;
    }

    // Pipeline step tracker (matches shorts_bot's print_progress output)
    // e.g. "[####----] 33% (2/6) Creating voiceover"
    const stepMatch = line.match(/\((\d+)\/(\d+)\)\s+(.+)$/);
    if (stepMatch) setStep(parseInt(stepMatch[1], 10));

    // Render sub-progress: "[render] frame=... fps=... speed=... time=...s pct=..."
    if (line.startsWith("[render] ")) {
      $("renderProgress").style.display = "flex";
      const m = line.match(/frame=(\S+)\s+fps=(\S+)\s+speed=(\S+)\s+time=(\S+)/);
      const pctM = line.match(/pct=([\d.]+)/);
      if (m) {
        $("renderStats").textContent =
          `frame ${m[1]} · ${m[2]} fps · ${m[3]} · ${m[4]}`;
      }
      if (pctM) $("renderFill").style.width = Math.min(100, parseFloat(pctM[1])) + "%";
    }
    if (/\[step\] end render/.test(line)) {
      $("renderFill").style.width = "100%";
      $("renderStats").textContent = "render complete";
    }
  }

  async function refreshPreview() {
    try {
      const res = await fetch("/api/latest"); const data = await res.json();
      const grid = $("previewGrid"); grid.innerHTML = "";
      if (data.hook_image) {
        const c = document.createElement("div"); c.className = "cell";
        const h = document.createElement("div"); h.className = "hint"; h.textContent = "Hook image";
        const img = document.createElement("img"); img.src = data.hook_image + "?t=" + Date.now();
        c.append(h, img); grid.appendChild(c);
        $("statHook").textContent = "Ready";
        $("statHookSub").textContent = data.hook_image.split("/").pop();
      }
      if (data.video) {
        const c = document.createElement("div"); c.className = "cell";
        const h = document.createElement("div"); h.className = "hint"; h.textContent = "Rendered short";
        const v = document.createElement("video"); v.src = data.video + "?t=" + Date.now(); v.controls = true;
        c.append(h, v); grid.appendChild(c);
        $("statVideo").textContent = "Ready";
        $("statVideoSub").textContent = "short.mp4";
      }
      if (!data.hook_image && !data.video) {
        const p = document.createElement("div"); p.className = "placeholder";
        p.textContent = "No output yet. Run the bot to see preview here."; grid.appendChild(p);
      }
      $("paneScript").textContent = data.script || "No script yet.";
      $("paneMetadata").textContent = data.metadata || "No metadata yet.";
    } catch (e) {}
  }

  runBtn.addEventListener("click", async () => {
    logEl.innerHTML = ""; logInline.innerHTML = ""; logCount.textContent = "0 lines";
    resetSteps();
    setStatus("Starting…", "run");
    const res = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collect())});
    const data = await res.json();
    if (!data.ok) { setStatus("Error", "err"); appendLog("Error: " + data.error); return; }
    cmdDisplay.textContent = data.cmd;
    runBtn.disabled = true; stopBtn.disabled = false;
    setStatus("Running pid " + data.pid, "run");
    $("statElapsedSub").textContent = "Running time";
    startElapsed();
  });
  stopBtn.addEventListener("click", async () => { await fetch("/api/stop", {method: "POST"}); });
  $("refreshBtn").addEventListener("click", refreshPreview);

  const es = new EventSource("/api/stream");
  es.onmessage = (ev) => {
    try {
      const line = JSON.parse(ev.data); appendLog(line);
      if (line.startsWith("__END__")) {
        const ok = line.includes("exit=0");
        setStatus(ok ? "Finished" : "Failed", ok ? "idle" : "err");
        if (ok) finishAllSteps();
        stopElapsed();
        runBtn.disabled = false; stopBtn.disabled = true;
        hidePrompt();
        refreshPreview();
      }
    } catch (e) {}
  };

  (async function init() {
    const res = await fetch("/api/status"); const data = await res.json();
    if (data.running) {
      setStatus("Running pid " + data.pid, "run");
      runBtn.disabled = true; stopBtn.disabled = false;
      cmdDisplay.textContent = data.cmd || "";
    }
    refreshPreview();
  })();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


def main() -> None:
    port = int(os.environ.get("PORT", "5005"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Shorts Bot UI running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
