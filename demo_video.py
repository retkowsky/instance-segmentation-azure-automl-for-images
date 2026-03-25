"""
Video annotation demo for instance segmentation.

Upload a street-scene video, watch it process frame by frame with a live
dashboard: animated counters, timeline heatmap, frame preview, and progress.
Download the fully annotated video at the end.

Launch:
    python demo_video.py --model onnx_model/train_artifacts/model.onnx \\
                         --labels onnx_model/train_artifacts/labels.json

Then open http://127.0.0.1:5070

Requirements:
    pip install flask onnxruntime opencv-python-headless numpy pillow
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify, Response, send_file
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# ONNX inference
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_COLOR = (231, 76, 60)
COLOR_MAP = {"car": (46, 204, 113), "pedestrian": (52, 152, 219)}


def preprocess(frame_rgb: np.ndarray, th: int = 600, tw: int = 800) -> np.ndarray:
    """Preprocess a numpy RGB frame for Mask R-CNN.

    Args:
        frame_rgb: RGB numpy array (H, W, 3).
        th: Target height.
        tw: Target width.

    Returns:
        Float32 tensor of shape (1, 3, th, tw).
    """
    img = cv2.resize(frame_rgb, (tw, th)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)


def annotate_frame(
    frame_rgb: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    class_names: list[str],
    ih: int = 600,
    iw: int = 800,
    thr: float = 0.5,
    alpha: float = 0.45,
) -> tuple[np.ndarray, list[dict]]:
    """Overlay masks and boxes onto a frame.

    Args:
        frame_rgb: Original RGB frame.
        boxes: Boxes (n, 4).
        labels: Class IDs (n,).
        scores: Scores (n,).
        masks: Masks (n, 1, H, W).
        class_names: Class names.
        ih: Model input height.
        iw: Model input width.
        thr: Score threshold.
        alpha: Mask alpha.

    Returns:
        Tuple of (annotated RGB frame, detection list).
    """
    oh, ow = frame_rgb.shape[:2]
    ov = frame_rgb.copy()
    sx, sy = ow / iw, oh / ih
    dets = []

    for i in range(len(scores)):
        if scores[i] < thr:
            continue
        lid = int(labels[i])
        ln = class_names[lid] if lid < len(class_names) else str(lid)
        sc = float(scores[i])
        c = COLOR_MAP.get(ln.lower(), DEFAULT_COLOR)
        x1, y1 = max(0, int(boxes[i][0] * sx)), max(0, int(boxes[i][1] * sy))
        x2, y2 = min(ow, int(boxes[i][2] * sx)), min(oh, int(boxes[i][3] * sy))

        m = cv2.resize((masks[i, 0] > 0.5).astype(np.uint8), (ow, oh))
        cm = np.zeros_like(ov)
        cm[m > 0] = c
        ov = cv2.addWeighted(ov, 1.0, cm, alpha, 0)
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, 2)

        txt = f"{ln.upper()} {sc:.0%}"
        (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(ov, (x1, y1 - th2 - 10), (x1 + tw2 + 8, y1), c, -1)
        cv2.putText(ov, txt, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        dets.append({"label": ln, "score": round(sc, 4)})

    dets.sort(key=lambda d: d["score"], reverse=True)
    return ov, dets


def frame_to_b64(frame_rgb: np.ndarray, quality: int = 70) -> str:
    """Encode an RGB frame to base64 JPEG data URI.

    Args:
        frame_rgb: RGB numpy array.
        quality: JPEG quality.

    Returns:
        Data URI string.
    """
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ──────────────────────────────────────────────────────────────────────
# In-memory job store
# ──────────────────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Annotation — Instance Segmentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#000;--surface:#0a0c10;--surface2:#12151c;--border:#1a1e28;
  --text:#e8eaef;--text2:#6a7080;--car:#2ecc71;--ped:#3498db;--accent:#2ecc71;
  --danger:#e74c3c;--warn:#f39c12;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh}

/* ═══ HEADER ═══ */
.hdr{display:flex;align-items:center;gap:1rem;padding:.8rem 1.5rem;
  background:var(--surface);border-bottom:1px solid var(--border)}
.hdr .dot{width:10px;height:10px;border-radius:50%;background:var(--danger)}
.hdr .dot.active{background:var(--accent);animation:pulse-live 1s ease infinite}
@keyframes pulse-live{0%,100%{opacity:1}50%{opacity:.5}}
.hdr h1{font-size:1rem;font-weight:700}
.hdr .badge{font-family:var(--mono);font-size:.68rem;padding:.2rem .55rem;
  border-radius:20px;background:rgba(46,204,113,.12);color:var(--accent);
  border:1px solid rgba(46,204,113,.25)}
.hdr .spacer{flex:1}

/* ═══ LAYOUT ═══ */
.container{max-width:1400px;margin:0 auto;padding:1.5rem;display:grid;
  grid-template-columns:1fr 340px;gap:1.5rem}
@media(max-width:1000px){.container{grid-template-columns:1fr}}

/* ═══ PANELS ═══ */
.panel{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1.2rem;overflow:hidden}
.panel+.panel{margin-top:1rem}
.panel h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--text2);margin-bottom:.8rem}

/* ═══ UPLOAD ═══ */
.upload-zone{border:2px dashed var(--border);border-radius:10px;padding:2.5rem 1rem;
  text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
.upload-zone:hover{border-color:var(--accent);background:rgba(46,204,113,.03)}
.upload-zone input{display:none}
.upload-zone .icon{font-size:2.5rem;margin-bottom:.4rem}
.upload-zone p{color:var(--text2);font-size:.88rem}
.upload-zone .fname{color:var(--accent);font-family:var(--mono);font-size:.82rem;margin-top:.4rem;display:none}

/* ═══ CONTROLS ═══ */
.ctrl-row{display:flex;align-items:center;gap:.8rem;margin-top:1rem;flex-wrap:wrap}
.ctrl-row .label{font-size:.78rem;color:var(--text2)}
.ctrl-row input[type=range]{width:90px;height:4px;-webkit-appearance:none;
  background:var(--border);border-radius:2px;outline:none}
.ctrl-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;background:var(--accent);cursor:pointer}
.ctrl-row .val{font-family:var(--mono);font-size:.75rem;color:var(--accent);min-width:28px}
.btn{padding:.55rem 1.1rem;border-radius:8px;border:1px solid var(--border);
  background:var(--surface);color:var(--text);font-family:var(--sans);
  font-size:.82rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.go{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.go:hover{background:#27ae60}
.btn.dl{background:var(--surface2);border-color:var(--accent);color:var(--accent)}

/* ═══ PROGRESS ═══ */
.progress-wrap{margin-top:1rem;display:none}
.progress-bar{height:6px;background:var(--surface2);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--ped));
  border-radius:3px;width:0%;transition:width .15s}
.progress-text{font-family:var(--mono);font-size:.72rem;color:var(--text2);margin-top:.3rem;
  display:flex;justify-content:space-between}

/* ═══ FRAME PREVIEW ═══ */
#frame-preview{width:100%;border-radius:10px;background:var(--surface2);
  min-height:200px;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
#frame-preview img{width:100%;display:block}
#frame-preview .ph{color:var(--text2);font-size:.9rem;padding:3rem}
.frame-badge{position:absolute;top:10px;left:10px;font-family:var(--mono);font-size:.72rem;
  padding:.25rem .5rem;border-radius:6px;background:rgba(0,0,0,.75);color:var(--accent);
  backdrop-filter:blur(4px)}

/* ═══ STATS CARDS ═══ */
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}
.stat{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:.7rem;text-align:center}
.stat .num{font-family:var(--mono);font-size:1.5rem;font-weight:700;transition:all .3s}
.stat .lbl{font-size:.7rem;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-top:.1rem}
.stat.car .num{color:var(--car)}
.stat.ped .num{color:var(--ped)}
.stat.fps .num{color:var(--warn)}
.stat.time .num{color:var(--ped)}

/* ═══ TIMELINE HEATMAP ═══ */
.timeline{margin-top:.3rem}
.timeline canvas{width:100%;height:60px;border-radius:6px;display:block;
  background:var(--surface2);border:1px solid var(--border)}
.timeline .axis{display:flex;justify-content:space-between;font-family:var(--mono);
  font-size:.65rem;color:var(--text2);margin-top:.2rem}
.legend-row{display:flex;gap:1rem;margin-top:.4rem;justify-content:center}
.legend-item{display:flex;align-items:center;gap:.3rem;font-size:.68rem;color:var(--text2)}
.legend-dot{width:8px;height:8px;border-radius:2px}

/* ═══ OUTPUT VIDEO ═══ */
#output-section{display:none}
#output-video{width:100%;border-radius:10px;background:#000}

footer{text-align:center;padding:1.2rem;color:var(--text2);font-size:.72rem}
</style>
</head>
<body>

<div class="hdr">
  <div class="dot" id="live-dot"></div>
  <h1>Video Annotation</h1>
  <div class="badge">ONNX &middot; Mask R-CNN</div>
  <div class="spacer"></div>
  <span id="status-text" style="font-family:var(--mono);font-size:.78rem;color:var(--text2)">Ready</span>
</div>

<div class="container">
  <!-- LEFT -->
  <div>
    <!-- Upload + preview -->
    <div class="panel">
      <h2>Video</h2>
      <div id="frame-preview">
        <div class="ph" id="frame-ph">Upload a video and click <b>Process</b></div>
        <img id="frame-img" style="display:none" alt="frame">
        <div class="frame-badge" id="frame-badge" style="display:none"></div>
      </div>
      <div class="progress-wrap" id="progress-wrap">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <div class="progress-text">
          <span id="prog-frames">0 / 0 frames</span>
          <span id="prog-eta">ETA: —</span>
        </div>
      </div>
    </div>

    <!-- Output video -->
    <div class="panel" id="output-section">
      <h2>Annotated output</h2>
      <video id="output-video" controls></video>
      <div style="margin-top:.8rem;display:flex;gap:.6rem">
        <a class="btn dl" id="dl-btn" download>Download annotated video</a>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div>
    <!-- Upload controls -->
    <div class="panel">
      <h2>Input</h2>
      <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
        <input type="file" id="file-input" accept="video/*">
        <div class="icon">&#127916;</div>
        <p>Drop a video or click to browse</p>
        <div class="fname" id="fname"></div>
      </div>
      <div class="ctrl-row">
        <span class="label">Confidence</span>
        <input type="range" id="conf" min="10" max="95" value="50">
        <span class="val" id="conf-val">0.50</span>
      </div>
      <div class="ctrl-row">
        <span class="label">Process every</span>
        <input type="range" id="skip" min="1" max="10" value="3">
        <span class="val" id="skip-val">3rd frame</span>
      </div>
      <div class="ctrl-row">
        <button class="btn go" id="btn-go" disabled>Process video</button>
      </div>
    </div>

    <!-- Live stats -->
    <div class="panel">
      <h2>Live stats</h2>
      <div class="stats-grid">
        <div class="stat car"><div class="num" id="s-cars">0</div><div class="lbl">Cars</div></div>
        <div class="stat ped"><div class="num" id="s-peds">0</div><div class="lbl">Pedestrians</div></div>
        <div class="stat fps"><div class="num" id="s-fps">—</div><div class="lbl">Frames/sec</div></div>
        <div class="stat time"><div class="num" id="s-time">—</div><div class="lbl">Avg inference</div></div>
        <div class="stat"><div class="num" id="s-total">0</div><div class="lbl">Total dets</div></div>
        <div class="stat"><div class="num" id="s-peak">0</div><div class="lbl">Peak / frame</div></div>
      </div>
    </div>

    <!-- Timeline heatmap -->
    <div class="panel">
      <h2>Detection timeline</h2>
      <div class="timeline">
        <canvas id="heatmap" height="60"></canvas>
        <div class="axis">
          <span>0:00</span>
          <span id="timeline-end">—</span>
        </div>
      </div>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-dot" style="background:var(--car)"></div>Cars</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--ped)"></div>Pedestrians</div>
      </div>
    </div>
  </div>
</div>

<footer>Video Annotation &mdash; Azure AutoML for Images &middot; ONNX Runtime &middot; Mask R-CNN</footer>

<script>
const fileInput=document.getElementById('file-input'),
  fname=document.getElementById('fname'),
  uploadZone=document.getElementById('upload-zone'),
  confInput=document.getElementById('conf'),
  confVal=document.getElementById('conf-val'),
  skipInput=document.getElementById('skip'),
  skipVal=document.getElementById('skip-val'),
  btnGo=document.getElementById('btn-go'),
  frameImg=document.getElementById('frame-img'),
  framePh=document.getElementById('frame-ph'),
  frameBadge=document.getElementById('frame-badge'),
  progWrap=document.getElementById('progress-wrap'),
  progFill=document.getElementById('progress-fill'),
  progFrames=document.getElementById('prog-frames'),
  progEta=document.getElementById('prog-eta'),
  liveDot=document.getElementById('live-dot'),
  statusText=document.getElementById('status-text'),
  outputSection=document.getElementById('output-section'),
  outputVideo=document.getElementById('output-video'),
  dlBtn=document.getElementById('dl-btn'),
  heatmapCanvas=document.getElementById('heatmap'),
  timelineEnd=document.getElementById('timeline-end');

let selectedFile=null;
const heatData=[];

confInput.addEventListener('input',()=>confVal.textContent=(confInput.value/100).toFixed(2));
skipInput.addEventListener('input',()=>skipVal.textContent=skipInput.value+(skipInput.value==='1'?'st':skipInput.value==='2'?'nd':skipInput.value==='3'?'rd':'th')+' frame');

// File handling
['dragenter','dragover'].forEach(e=>uploadZone.addEventListener(e,ev=>{ev.preventDefault();uploadZone.style.borderColor='var(--accent)'}));
['dragleave','drop'].forEach(e=>uploadZone.addEventListener(e,ev=>{ev.preventDefault();uploadZone.style.borderColor=''}));
uploadZone.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){fileInput.files=ev.dataTransfer.files;handleFile(ev.dataTransfer.files[0])}});
fileInput.addEventListener('change',()=>{if(fileInput.files.length)handleFile(fileInput.files[0])});

function handleFile(f){
  selectedFile=f;
  fname.textContent=f.name+' ('+formatBytes(f.size)+')';
  fname.style.display='block';
  btnGo.disabled=false;
}
function formatBytes(b){if(b<1024)return b+'B';if(b<1048576)return(b/1024).toFixed(1)+'KB';return(b/1048576).toFixed(1)+'MB'}
function formatTime(s){const m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+String(ss).padStart(2,'0')}

// Heatmap drawing
function drawHeatmap(){
  const ctx=heatmapCanvas.getContext('2d');
  const W=heatmapCanvas.width=heatmapCanvas.clientWidth*2;
  const H=heatmapCanvas.height=120;
  ctx.clearRect(0,0,W,H);
  if(!heatData.length)return;

  const maxDet=Math.max(1,...heatData.map(d=>d.cars+d.peds));
  const barW=Math.max(1,W/heatData.length);

  heatData.forEach((d,i)=>{
    const x=i*barW;
    // Cars (bottom-up)
    const ch=(d.cars/maxDet)*(H/2);
    ctx.fillStyle='rgba(46,204,113,0.7)';
    ctx.fillRect(x,H-ch,barW-0.5,ch);
    // Peds (stacked on top)
    const ph=(d.peds/maxDet)*(H/2);
    ctx.fillStyle='rgba(52,152,219,0.7)';
    ctx.fillRect(x,H-ch-ph,barW-0.5,ph);
  });

  // Playhead line
  ctx.strokeStyle='rgba(255,255,255,0.3)';
  ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(heatData.length*barW,0);
  ctx.lineTo(heatData.length*barW,H);
  ctx.stroke();
}

// Process video
btnGo.addEventListener('click', async ()=>{
  if(!selectedFile)return;
  btnGo.disabled=true;
  heatData.length=0;
  drawHeatmap();
  liveDot.classList.add('active');
  statusText.textContent='Uploading...';
  progWrap.style.display='block';
  progFill.style.width='0%';
  outputSection.style.display='none';

  // Reset stats
  let totalCars=0,totalPeds=0,totalDets=0,peakFrame=0,infTimes=[];

  const formData=new FormData();
  formData.append('video',selectedFile);
  formData.append('threshold',(confInput.value/100).toFixed(2));
  formData.append('skip',skipInput.value);

  // Upload and get job_id
  let jobId;
  try{
    const resp=await fetch('/upload',{method:'POST',body:formData});
    const data=await resp.json();
    if(data.error){alert(data.error);btnGo.disabled=false;return}
    jobId=data.job_id;
    statusText.textContent='Processing...';
  }catch(e){alert('Upload failed: '+e.message);btnGo.disabled=false;return}

  // Stream results via SSE
  const evtSrc=new EventSource('/stream/'+jobId);
  const t0=performance.now();
  let processedCount=0;

  evtSrc.addEventListener('frame',e=>{
    const d=JSON.parse(e.data);
    processedCount++;

    // Update preview
    frameImg.src=d.preview;
    frameImg.style.display='block';
    framePh.style.display='none';
    frameBadge.textContent='Frame '+d.frame_idx+' / '+d.total_frames;
    frameBadge.style.display='block';

    // Progress
    const pct=(d.frame_idx/d.total_frames*100);
    progFill.style.width=pct+'%';
    progFrames.textContent=d.frame_idx+' / '+d.total_frames+' frames';
    const elapsed=(performance.now()-t0)/1000;
    const fps=processedCount/elapsed;
    const remaining=(d.total_frames-d.frame_idx)/Math.max(fps,0.01);
    progEta.textContent='ETA: '+formatTime(remaining);

    // Accumulate stats
    totalCars+=d.cars;
    totalPeds+=d.peds;
    totalDets+=d.total;
    const frameDets=d.cars+d.peds;
    if(frameDets>peakFrame)peakFrame=frameDets;
    infTimes.push(d.inference_time);

    // Update live stats
    document.getElementById('s-cars').textContent=totalCars;
    document.getElementById('s-peds').textContent=totalPeds;
    document.getElementById('s-total').textContent=totalDets;
    document.getElementById('s-peak').textContent=peakFrame;
    document.getElementById('s-fps').textContent=fps.toFixed(1);
    const avgInf=infTimes.reduce((a,b)=>a+b,0)/infTimes.length;
    document.getElementById('s-time').textContent=avgInf.toFixed(3)+'s';

    // Timeline
    heatData.push({cars:d.cars, peds:d.peds});
    drawHeatmap();
    if(d.video_duration) timelineEnd.textContent=formatTime(d.video_duration);
  });

  evtSrc.addEventListener('done',e=>{
    evtSrc.close();
    const d=JSON.parse(e.data);
    liveDot.classList.remove('active');
    statusText.textContent='Done — '+d.total_frames+' frames processed';
    progFill.style.width='100%';
    progEta.textContent='Complete';
    btnGo.disabled=false;

    // Show output video
    if(d.video_url){
      outputVideo.src=d.video_url;
      dlBtn.href=d.video_url;
      dlBtn.download='annotated_'+selectedFile.name;
      outputSection.style.display='block';
    }
  });

  evtSrc.addEventListener('error_msg',e=>{
    evtSrc.close();
    const d=JSON.parse(e.data);
    alert('Error: '+d.message);
    liveDot.classList.remove('active');
    statusText.textContent='Error';
    btnGo.disabled=false;
  });

  evtSrc.onerror=()=>{
    evtSrc.close();
    liveDot.classList.remove('active');
    statusText.textContent='Connection lost';
    btnGo.disabled=false;
  };
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str) -> Flask:
    """Create the video annotation demo Flask app.

    Args:
        model_path: Path to model.onnx.
        labels_path: Path to labels.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

    with open(labels_path) as f:
        class_names: list[str] = json.load(f)

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    inp_name = session.get_inputs()[0].name

    print(f"Model: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Providers: {session.get_providers()}")

    output_dir = tempfile.mkdtemp(prefix="seg_video_")

    @app.route("/")
    def index():
        """Serve the video demo page."""
        return render_template_string(HTML)

    @app.route("/upload", methods=["POST"])
    def upload():
        """Accept a video upload and prepare a processing job.

        Returns:
            JSON with job_id, total_frames, fps, duration.
        """
        if "video" not in request.files:
            return jsonify({"error": "No video provided"}), 400

        video_file = request.files["video"]
        threshold = float(request.form.get("threshold", "0.5"))
        skip = int(request.form.get("skip", "3"))

        # Save upload
        job_id = str(uuid.uuid4())[:8]
        input_path = os.path.join(output_dir, f"{job_id}_input.mp4")
        video_file.save(input_path)

        # Probe video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()

        jobs[job_id] = {
            "input_path": input_path,
            "threshold": threshold,
            "skip": skip,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
        }

        return jsonify({
            "job_id": job_id,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration": round(duration, 2),
        })

    @app.route("/stream/<job_id>")
    def stream(job_id: str):
        """Stream frame-by-frame results via Server-Sent Events.

        Each processed frame emits an SSE 'frame' event with:
            frame_idx, preview (base64), cars, peds, total, inference_time.

        When complete, emits a 'done' event with video_url.

        Args:
            job_id: The processing job identifier.

        Returns:
            A streaming SSE response.
        """
        if job_id not in jobs:
            def error_gen():
                yield 'event: error_msg\ndata: {"message":"Job not found"}\n\n'
            return Response(error_gen(), content_type="text/event-stream")

        job = jobs[job_id]

        def generate():
            cap = cv2.VideoCapture(job["input_path"])
            if not cap.isOpened():
                yield 'event: error_msg\ndata: {"message":"Cannot open video"}\n\n'
                return

            out_path = os.path.join(output_dir, f"{job_id}_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, job["fps"],
                (job["width"], job["height"]),
            )

            frame_idx = 0
            last_annotated = None
            IH, IW = 600, 800
            thr = job["threshold"]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_idx % job["skip"] == 0 or frame_idx == 1:
                    # Run inference
                    tensor = preprocess(frame_rgb, IH, IW)
                    t0 = time.time()
                    res = session.run(None, {inp_name: tensor})
                    elapsed = time.time() - t0

                    annotated_rgb, dets = annotate_frame(
                        frame_rgb, res[0], res[1], res[2], res[3],
                        class_names, IH, IW, thr=thr,
                    )
                    last_annotated = annotated_rgb

                    cars = sum(1 for d in dets if d["label"] == "car")
                    peds = sum(1 for d in dets if d["label"] == "pedestrian")

                    preview = frame_to_b64(annotated_rgb, quality=60)

                    payload = json.dumps({
                        "frame_idx": frame_idx,
                        "total_frames": job["total_frames"],
                        "preview": preview,
                        "cars": cars,
                        "peds": peds,
                        "total": len(dets),
                        "inference_time": round(elapsed, 4),
                        "video_duration": round(job["duration"], 1),
                    })
                    yield f"event: frame\ndata: {payload}\n\n"

                    # Write annotated frame
                    writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                else:
                    # Write last annotated frame (or original if none yet)
                    if last_annotated is not None:
                        writer.write(cv2.cvtColor(last_annotated, cv2.COLOR_RGB2BGR))
                    else:
                        writer.write(frame)

            cap.release()
            writer.release()

            # Emit done
            done_payload = json.dumps({
                "total_frames": frame_idx,
                "video_url": f"/video/{job_id}",
            })
            yield f"event: done\ndata: {done_payload}\n\n"

        return Response(generate(), content_type="text/event-stream")

    @app.route("/video/<job_id>")
    def serve_video(job_id: str):
        """Serve the annotated output video for playback/download.

        Args:
            job_id: The processing job identifier.

        Returns:
            The MP4 video file.
        """
        out_path = os.path.join(output_dir, f"{job_id}_output.mp4")
        if not os.path.isfile(out_path):
            return jsonify({"error": "Video not found"}), 404
        return send_file(out_path, mimetype="video/mp4", as_attachment=False)

    return app


def main() -> None:
    """Parse arguments and start the video demo server."""
    parser = argparse.ArgumentParser(description="Video annotation demo")
    parser.add_argument("--model", default=os.path.join("onnx_model", "train_artifacts", "model.onnx"))
    parser.add_argument("--labels", default=os.path.join("onnx_model", "train_artifacts", "labels.json"))
    parser.add_argument("--port", type=int, default=5070)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found at '{args.model}'")
        return
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels not found at '{args.labels}'")
        return

    app = create_app(args.model, args.labels)
    print(f"\nVideo demo at http://{args.host}:{args.port}")
    print("Upload a video and watch it process in real time.\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
