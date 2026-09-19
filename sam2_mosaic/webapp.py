"""Local web UI: draw bboxes on a video frame, run SAM2 tracking + mosaic,
and watch segmentation happen live (MJPEG stream) before downloading the result.

Usage (from the repository root):
    python -m sam2_mosaic.webapp [--host 0.0.0.0] [--port 8765]

Then open http://<host>:8765 (VS Code Remote usually forwards the port for you).
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request

from .core import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, Box, read_frame, run_pipeline, video_meta

app = Flask(__name__)

JOBS: dict[str, "Job"] = {}


class Job:
    def __init__(self):
        self.status = "running"  # running | done | error
        self.log: list[str] = []
        self.output_path: str | None = None
        self.error: str | None = None
        self.latest_frame: bytes | None = None
        self.frame_version = 0
        self.lock = threading.Lock()

    def push_frame(self, image) -> None:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self.lock:
            self.latest_frame = buf.tobytes()
            self.frame_version += 1


def worker(job: Job, video_path: str, output_path: str, boxes: list[Box], block_size: int, dilate: int, device: str) -> None:
    try:
        for event in run_pipeline(
            video_path=video_path,
            output_path=output_path,
            boxes=boxes,
            block_size=block_size,
            dilate=dilate,
            checkpoint=DEFAULT_CHECKPOINT,
            config=DEFAULT_CONFIG,
            device=device,
        ):
            if event["type"] == "log":
                job.log.append(event["text"])
            elif event["type"] in ("segment_preview", "mosaic_preview"):
                job.push_frame(event["image"])
            elif event["type"] == "done":
                job.output_path = event["output_path"]
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.log.append(f"ERROR: {exc}")


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/meta")
def meta():
    path = request.args.get("path", "")
    try:
        m = video_meta(path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify(m)


@app.route("/frame")
def frame():
    path = request.args.get("path", "")
    idx = int(request.args.get("frame", "0"))
    try:
        img = read_frame(path, idx)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True)
    video_path = data["video_path"]
    output_path = data.get("output_path") or str(Path(video_path).with_name(Path(video_path).stem + "_mosaic.mp4"))
    raw_boxes = data["boxes"]
    if not raw_boxes:
        return jsonify({"error": "no boxes given"}), 400
    boxes = [
        Box(frame_idx=int(b["frame"]), obj_id=i + 1, xyxy=[float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])])
        for i, b in enumerate(raw_boxes)
    ]
    block_size = int(data.get("block_size", 20))
    dilate = int(data.get("dilate", 6))
    device = data.get("device", "cuda:0")

    job_id = uuid.uuid4().hex[:12]
    job = Job()
    JOBS[job_id] = job
    threading.Thread(target=worker, args=(job, video_path, output_path, boxes, block_size, dilate, device), daemon=True).start()
    return jsonify({"job_id": job_id, "output_path": output_path})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job.status,
        "log": job.log[-50:],
        "output_path": job.output_path,
        "error": job.error,
    })


@app.route("/stream/<job_id>")
def stream(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404

    def generate():
        last_version = -1
        while True:
            with job.lock:
                version = job.frame_version
                data = job.latest_frame
                status_ = job.status
            if data is not None and version != last_version:
                last_version = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            elif status_ != "running":
                break
            else:
                time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if job is None or not job.output_path:
        return jsonify({"error": "not ready"}), 404
    p = Path(job.output_path)
    with open(p, "rb") as f:
        data = f.read()
    return Response(data, mimetype="video/mp4", headers={"Content-Disposition": f"attachment; filename={p.name}"})


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAM2 Video Mosaic</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #111; color: #eee; }
  h1 { font-size: 18px; margin: 0 0 16px; }
  .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  fieldset { border: 1px solid #333; border-radius: 8px; margin-bottom: 12px; }
  legend { color: #aaa; font-size: 12px; }
  label { display: block; font-size: 12px; color: #aaa; margin-top: 8px; }
  input, select { width: 100%; box-sizing: border-box; padding: 6px; margin-top: 2px; background: #1c1c1c; color: #eee; border: 1px solid #333; border-radius: 4px; }
  button { padding: 8px 14px; margin-top: 10px; background: #2a6; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
  button.secondary { background: #444; }
  button:disabled { opacity: 0.5; cursor: default; }
  #canvasWrap { position: relative; display: inline-block; border: 1px solid #333; }
  canvas { display: block; max-width: 900px; cursor: crosshair; }
  .panel { width: 320px; flex-shrink: 0; }
  .boxlist div { display: flex; align-items: center; gap: 6px; font-size: 12px; padding: 3px 0; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
  .rm { cursor: pointer; color: #f66; }
  #log { white-space: pre-wrap; font-family: monospace; font-size: 11px; background: #000; padding: 8px; height: 140px; overflow-y: auto; border-radius: 6px; }
  #liveWrap img, #liveWrap video { max-width: 900px; display: block; border: 1px solid #333; }
  .hint { font-size: 11px; color: #888; }
</style>
</head>
<body>
<h1>SAM2 Video Mosaic — draw boxes &amp; watch tracking live</h1>
<div class="row">
  <div>
    <fieldset>
      <legend>1. Load video</legend>
      <label>Video path (on the server)</label>
      <input id="videoPath" placeholder="/path/to/video.mp4">
      <label>Reference frame number (the frame to draw boxes on)</label>
      <input id="frameIdx" type="number" value="0" style="width:100px">
      <button onclick="loadFrame()">Load frame</button>
      <div class="hint" id="metaInfo"></div>
    </fieldset>
    <div id="canvasWrap">
      <canvas id="canvas" width="640" height="360"></canvas>
    </div>
    <p class="hint">Drag on the frame to draw as many boxes as you need. Each box is recorded with the current "Reference frame number".</p>
  </div>

  <div class="panel">
    <fieldset>
      <legend>2. Boxes</legend>
      <div class="boxlist" id="boxList"></div>
      <button class="secondary" onclick="clearBoxes()">Clear all</button>
    </fieldset>

    <fieldset>
      <legend>3. Output settings</legend>
      <label>Output path (leave empty for auto)</label>
      <input id="outputPath" placeholder="auto: &lt;input&gt;_mosaic.mp4">
      <label>Mosaic block size (px)</label>
      <input id="blockSize" type="number" value="20">
      <label>Mask dilation (px)</label>
      <input id="dilate" type="number" value="6">
      <label>GPU</label>
      <select id="device">
        <option value="cuda:0">cuda:0</option>
        <option value="cuda:1">cuda:1</option>
        <option value="cuda:2">cuda:2</option>
        <option value="cuda:3">cuda:3</option>
        <option value="cpu">cpu</option>
      </select>
      <button id="runBtn" onclick="runJob()">Run tracking + mosaic</button>
    </fieldset>
  </div>
</div>

<h2 style="font-size:14px;margin-top:24px;">Live progress</h2>
<div id="liveWrap"><img id="liveImg" style="display:none"></div>
<div id="log"></div>

<script>
let boxes = []; // {frame, x1,y1,x2,y2}
let img = new Image();
let curPath = "", curFrame = 0;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const colors = ['#ff3b30','#3b82f6','#22c55e','#f59e0b','#d946ef','#06b6d4','#eab308','#a855f7'];

async function loadFrame() {
  curPath = document.getElementById('videoPath').value;
  curFrame = parseInt(document.getElementById('frameIdx').value || '0');
  const metaRes = await fetch(`/meta?path=${encodeURIComponent(curPath)}`);
  const meta = await metaRes.json();
  if (meta.error) { alert(meta.error); return; }
  document.getElementById('metaInfo').textContent = `${meta.width}x${meta.height}, ${meta.n_frames} frames, ${meta.fps.toFixed(2)} fps`;
  const url = `/frame?path=${encodeURIComponent(curPath)}&frame=${curFrame}&t=${Date.now()}`;
  img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    redraw();
  };
  img.src = url;
}

function redraw() {
  ctx.drawImage(img, 0, 0);
  boxes.forEach((b, i) => {
    ctx.strokeStyle = colors[i % colors.length];
    ctx.lineWidth = 3;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = colors[i % colors.length];
    ctx.font = '14px monospace';
    ctx.fillText(`#${i+1} f${b.frame}`, b.x1 + 2, b.y1 - 4 < 10 ? b.y1 + 14 : b.y1 - 4);
  });
}

let dragStart = null;
canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  const scale = canvas.width / r.width;
  dragStart = { x: (e.clientX - r.left) * scale, y: (e.clientY - r.top) * scale };
});
canvas.addEventListener('mousemove', e => {
  if (!dragStart) return;
  const r = canvas.getBoundingClientRect();
  const scale = canvas.width / r.width;
  const x = (e.clientX - r.left) * scale, y = (e.clientY - r.top) * scale;
  redraw();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.strokeRect(dragStart.x, dragStart.y, x - dragStart.x, y - dragStart.y);
});
canvas.addEventListener('mouseup', e => {
  if (!dragStart) return;
  const r = canvas.getBoundingClientRect();
  const scale = canvas.width / r.width;
  const x = (e.clientX - r.left) * scale, y = (e.clientY - r.top) * scale;
  const x1 = Math.min(dragStart.x, x), x2 = Math.max(dragStart.x, x);
  const y1 = Math.min(dragStart.y, y), y2 = Math.max(dragStart.y, y);
  dragStart = null;
  if (x2 - x1 < 4 || y2 - y1 < 4) { redraw(); return; }
  boxes.push({ frame: curFrame, x1, y1, x2, y2 });
  renderBoxList();
  redraw();
});

function renderBoxList() {
  const el = document.getElementById('boxList');
  el.innerHTML = '';
  boxes.forEach((b, i) => {
    const div = document.createElement('div');
    div.innerHTML = `<span class="swatch" style="background:${colors[i % colors.length]}"></span>
      #${i+1} frame ${b.frame}: [${b.x1|0},${b.y1|0},${b.x2|0},${b.y2|0}]
      <span class="rm" onclick="removeBox(${i})">✕</span>`;
    el.appendChild(div);
  });
}
function removeBox(i) { boxes.splice(i, 1); renderBoxList(); redraw(); }
function clearBoxes() { boxes = []; renderBoxList(); redraw(); }

let pollTimer = null;
async function runJob() {
  if (boxes.length === 0) { alert('Draw at least one box first'); return; }
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('log').textContent = '';
  const body = {
    video_path: curPath,
    output_path: document.getElementById('outputPath').value || null,
    boxes: boxes,
    block_size: parseInt(document.getElementById('blockSize').value),
    dilate: parseInt(document.getElementById('dilate').value),
    device: document.getElementById('device').value,
  };
  const res = await fetch('/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await res.json();
  if (data.error) { alert(data.error); btn.disabled = false; return; }
  const jobId = data.job_id;
  const liveImg = document.getElementById('liveImg');
  liveImg.style.display = 'block';
  liveImg.src = `/stream/${jobId}?t=${Date.now()}`;

  pollTimer = setInterval(async () => {
    const s = await (await fetch(`/status/${jobId}`)).json();
    document.getElementById('log').textContent = s.log.join('\n');
    document.getElementById('log').scrollTop = 1e9;
    if (s.status === 'done') {
      clearInterval(pollTimer);
      btn.disabled = false;
      const wrap = document.getElementById('liveWrap');
      wrap.innerHTML = `<video controls autoplay src="/download/${jobId}"></video><p class="hint">Saved to: ${s.output_path}</p>`;
    } else if (s.status === 'error') {
      clearInterval(pollTimer);
      btn.disabled = false;
      alert('Error: ' + s.error);
    }
  }, 800);
}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
