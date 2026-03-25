"""
Batch mosaic grid demo for instance segmentation.

Upload 1-50 images at once, process them all through ONNX inference,
and display a responsive mosaic grid of annotated results with:
- Per-image detection counts
- Click-to-zoom lightbox
- Summary stats: total cars/peds, avg inference time
- Class distribution bar chart
- CSV export of all detections

Launch:
    python demo_batch_mosaic.py --model onnx_model/train_artifacts/model.onnx \\
                                --labels onnx_model/train_artifacts/labels.json

Then open http://127.0.0.1:5080

Requirements:
    pip install flask onnxruntime opencv-python-headless numpy pillow
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify, Response
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# ONNX inference
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_COLOR = (231, 76, 60)
COLOR_MAP = {"car": (46, 204, 113), "pedestrian": (52, 152, 219)}


def preprocess(pil_img: Image.Image, th: int = 600, tw: int = 800) -> tuple[np.ndarray, int, int]:
    """Preprocess a PIL image for Mask R-CNN ONNX inference.

    Args:
        pil_img: Input image.
        th: Target height.
        tw: Target width.

    Returns:
        Tuple of (tensor, orig_height, orig_width).
    """
    img = pil_img.convert("RGB")
    ow, oh = img.size
    img = img.resize((tw, th), Image.BILINEAR)
    a = np.array(img, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(a.transpose(2, 0, 1), 0).astype(np.float32), oh, ow


def annotate(
    pil_img: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    class_names: list[str],
    ih: int = 600, iw: int = 800,
    thr: float = 0.5, alpha: float = 0.45,
) -> tuple[Image.Image, list[dict]]:
    """Overlay masks and bounding boxes onto an image.

    Args:
        pil_img: Original image.
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
        Tuple of (annotated PIL image, detection list).
    """
    rgb = np.array(pil_img.convert("RGB"))
    oh, ow = rgb.shape[:2]
    ov = rgb.copy()
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
        cm = np.zeros_like(ov); cm[m > 0] = c
        ov = cv2.addWeighted(ov, 1.0, cm, alpha, 0)
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, 2)
        txt = f"{ln.upper()} {sc:.0%}"
        (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(ov, (x1, y1 - th2 - 8), (x1 + tw2 + 6, y1), c, -1)
        cv2.putText(ov, txt, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        dets.append({"label": ln, "score": round(sc, 4), "bbox": [x1, y1, x2, y2]})

    dets.sort(key=lambda d: d["score"], reverse=True)
    return Image.fromarray(ov), dets


def pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image to a base64 data URI.

    Args:
        img: PIL Image.
        fmt: Image format.
        quality: JPEG quality.

    Returns:
        Data URI string.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch Mosaic — Instance Segmentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#06080c;--surface:#0c0f15;--surface2:#141820;--border:#1e2330;
  --text:#e4e7ef;--text2:#6a7088;--car:#2ecc71;--ped:#3498db;--accent:#2ecc71;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh}

.hdr{display:flex;align-items:center;gap:1rem;padding:.8rem 1.5rem;
  background:var(--surface);border-bottom:1px solid var(--border)}
.hdr h1{font-size:1rem;font-weight:700}
.hdr .badge{font-family:var(--mono);font-size:.68rem;padding:.2rem .55rem;
  border-radius:20px;background:rgba(46,204,113,.12);color:var(--accent);
  border:1px solid rgba(46,204,113,.25)}
.hdr .spacer{flex:1}
.hdr .status{font-family:var(--mono);font-size:.78rem;color:var(--text2)}

.main{max-width:1400px;margin:0 auto;padding:1.5rem}

/* ═══ UPLOAD STRIP ═══ */
.upload-strip{display:flex;align-items:center;gap:1rem;padding:1rem 1.2rem;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;flex-wrap:wrap}
.upload-strip input[type=file]{display:none}
.btn{padding:.5rem 1rem;border-radius:8px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-family:var(--sans);
  font-size:.82rem;font-weight:600;cursor:pointer;transition:all .2s;
  display:inline-flex;align-items:center;gap:.35rem;text-decoration:none}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.go{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.go:hover{background:#27ae60}
.sep{width:1px;height:24px;background:var(--border)}
.ctrl-label{font-size:.78rem;color:var(--text2)}
.slider-w{display:flex;align-items:center;gap:.5rem}
.slider-w input[type=range]{width:90px;height:4px;-webkit-appearance:none;
  background:var(--border);border-radius:2px;outline:none}
.slider-w input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;background:var(--accent);cursor:pointer}
.slider-w .val{font-family:var(--mono);font-size:.75rem;color:var(--accent);min-width:30px}
.file-count{font-family:var(--mono);font-size:.78rem;color:var(--accent)}

/* ═══ PROGRESS ═══ */
.progress-wrap{margin-top:1rem;display:none}
.progress-outer{height:6px;background:var(--surface2);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--ped));width:0%;
  border-radius:3px;transition:width .2s}
.progress-info{display:flex;justify-content:space-between;margin-top:.3rem;
  font-family:var(--mono);font-size:.72rem;color:var(--text2)}

/* ═══ SUMMARY STATS ═══ */
.summary{display:none;margin-top:1.2rem;gap:1rem}
.summary-cards{display:flex;gap:.6rem;flex-wrap:wrap}
.scard{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:.65rem 1rem;display:flex;flex-direction:column;align-items:center;min-width:100px}
.scard .num{font-family:var(--mono);font-size:1.4rem;font-weight:700}
.scard .lbl{font-size:.7rem;color:var(--text2);text-transform:uppercase;letter-spacing:.04em;margin-top:.1rem}
.scard.car .num{color:var(--car)}
.scard.ped .num{color:var(--ped)}
.scard.fps .num{color:#f39c12}
.scard.time .num{color:var(--ped)}

/* ═══ CHART ═══ */
.chart-row{display:flex;gap:1.5rem;margin-top:1rem;flex-wrap:wrap;align-items:flex-end}
.chart-wrap{flex:1;min-width:280px;background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:1rem}
.chart-wrap h3{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:.7rem}
.bar-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem}
.bar-label{font-family:var(--mono);font-size:.72rem;min-width:80px;text-align:right;color:var(--text2)}
.bar-track{flex:1;height:18px;background:var(--surface2);border-radius:4px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:4px;transition:width .5s ease}
.bar-fill.car{background:var(--car)}
.bar-fill.ped{background:var(--ped)}
.bar-val{font-family:var(--mono);font-size:.72rem;min-width:28px;color:var(--text)}
.export-row{display:flex;gap:.5rem;margin-top:1rem}

/* ═══ MOSAIC GRID ═══ */
.mosaic{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
  gap:.8rem;margin-top:1.2rem}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;cursor:pointer;transition:border-color .2s,transform .15s}
.tile:hover{border-color:var(--accent);transform:translateY(-2px)}
.tile img{width:100%;display:block;aspect-ratio:16/10;object-fit:cover}
.tile-info{padding:.5rem .7rem;display:flex;align-items:center;justify-content:space-between}
.tile-name{font-size:.75rem;color:var(--text2);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;max-width:140px}
.tile-badges{display:flex;gap:.3rem}
.tbadge{font-family:var(--mono);font-size:.65rem;padding:.15rem .4rem;border-radius:4px;
  background:var(--surface2);border:1px solid var(--border)}
.tbadge.car{color:var(--car);border-color:rgba(46,204,113,.3)}
.tbadge.ped{color:var(--ped);border-color:rgba(52,152,219,.3)}

/* ═══ LIGHTBOX ═══ */
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:100;
  display:none;align-items:center;justify-content:center;cursor:zoom-out;
  backdrop-filter:blur(4px)}
.lightbox.open{display:flex}
.lightbox img{max-width:92vw;max-height:88vh;border-radius:8px;box-shadow:0 0 60px rgba(0,0,0,.8)}
.lightbox-close{position:absolute;top:16px;right:20px;color:#fff;font-size:1.5rem;
  cursor:pointer;opacity:.6;transition:opacity .2s;background:none;border:none}
.lightbox-close:hover{opacity:1}
.lightbox-info{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);
  font-family:var(--mono);font-size:.82rem;color:var(--text2);
  background:rgba(0,0,0,.7);padding:.4rem 1rem;border-radius:8px;
  backdrop-filter:blur(4px)}

.placeholder{display:flex;align-items:center;justify-content:center;
  min-height:300px;color:var(--text2);font-size:.95rem;margin-top:2rem}

footer{text-align:center;padding:1.2rem;color:var(--text2);font-size:.72rem;margin-top:2rem}
</style>
</head>
<body>

<div class="hdr">
  <h1>Batch Mosaic Grid</h1>
  <div class="badge">ONNX &middot; Mask R-CNN</div>
  <div class="spacer"></div>
  <span class="status" id="status">Ready</span>
</div>

<div class="main">
  <!-- Upload controls -->
  <div class="upload-strip">
    <input type="file" id="file-input" accept="image/*" multiple>
    <button class="btn" onclick="document.getElementById('file-input').click()">Choose images</button>
    <span class="file-count" id="file-count"></span>
    <div class="sep"></div>
    <div class="slider-w">
      <span class="ctrl-label">Confidence</span>
      <input type="range" id="conf" min="10" max="95" value="50">
      <span class="val" id="conf-val">0.50</span>
    </div>
    <div class="sep"></div>
    <button class="btn go" id="btn-go" disabled>Process all</button>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-outer"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="progress-info">
      <span id="prog-text">0 / 0</span>
      <span id="prog-eta">ETA: —</span>
    </div>
  </div>

  <!-- Summary -->
  <div class="summary" id="summary">
    <div class="summary-cards" id="summary-cards"></div>
    <div class="chart-row" id="chart-row"></div>
    <div class="export-row">
      <button class="btn" id="btn-csv">Export detections (.csv)</button>
    </div>
  </div>

  <!-- Mosaic grid -->
  <div class="mosaic" id="mosaic"></div>

  <!-- Placeholder -->
  <div class="placeholder" id="placeholder">
    Select multiple images and click <b>Process all</b> to build the mosaic
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox">
  <button class="lightbox-close" id="lb-close">&times;</button>
  <img id="lb-img" alt="full size">
  <div class="lightbox-info" id="lb-info"></div>
</div>

<footer>Batch Mosaic Grid — Azure AutoML for Images &middot; ONNX Runtime &middot; Mask R-CNN</footer>

<script>
const fileInput=document.getElementById('file-input'),
  fileCount=document.getElementById('file-count'),
  confInput=document.getElementById('conf'),
  confVal=document.getElementById('conf-val'),
  btnGo=document.getElementById('btn-go'),
  progWrap=document.getElementById('progress-wrap'),
  progFill=document.getElementById('progress-fill'),
  progText=document.getElementById('prog-text'),
  progEta=document.getElementById('prog-eta'),
  summary=document.getElementById('summary'),
  summaryCards=document.getElementById('summary-cards'),
  chartRow=document.getElementById('chart-row'),
  mosaic=document.getElementById('mosaic'),
  placeholder=document.getElementById('placeholder'),
  statusEl=document.getElementById('status'),
  lightbox=document.getElementById('lightbox'),
  lbImg=document.getElementById('lb-img'),
  lbInfo=document.getElementById('lb-info'),
  lbClose=document.getElementById('lb-close'),
  btnCsv=document.getElementById('btn-csv');

let selectedFiles=[], allResults=[];

confInput.addEventListener('input',()=>confVal.textContent=(confInput.value/100).toFixed(2));

fileInput.addEventListener('change',()=>{
  selectedFiles=Array.from(fileInput.files);
  fileCount.textContent=selectedFiles.length+' image'+(selectedFiles.length>1?'s':'');
  btnGo.disabled=selectedFiles.length===0;
});

// Lightbox
lightbox.addEventListener('click',e=>{if(e.target===lightbox)lightbox.classList.remove('open')});
lbClose.addEventListener('click',()=>lightbox.classList.remove('open'));
window.addEventListener('keydown',e=>{if(e.key==='Escape')lightbox.classList.remove('open')});

function openLightbox(src,info){
  lbImg.src=src; lbInfo.textContent=info;
  lightbox.classList.add('open');
}

// CSV export
btnCsv.addEventListener('click',()=>{
  if(!allResults.length)return;
  let csv='filename,detection_index,label,score,x_min,y_min,x_max,y_max\n';
  allResults.forEach(r=>{
    r.detections.forEach((d,i)=>{
      csv+=`"${r.filename}",${i+1},${d.label},${d.score},${d.bbox[0]},${d.bbox[1]},${d.bbox[2]},${d.bbox[3]}\n`;
    });
  });
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='detections_batch.csv';
  a.click();
  URL.revokeObjectURL(a.href);
});

// Process all
btnGo.addEventListener('click', async ()=>{
  if(!selectedFiles.length)return;
  btnGo.disabled=true;
  allResults=[];
  mosaic.innerHTML='';
  placeholder.style.display='none';
  progWrap.style.display='block';
  progFill.style.width='0%';
  summary.style.display='none';
  statusEl.textContent='Processing...';

  const t0=performance.now();
  const total=selectedFiles.length;
  let totalCars=0, totalPeds=0, totalDets=0, totalInf=0;
  const perImage=[];

  for(let i=0;i<total;i++){
    const file=selectedFiles[i];
    const formData=new FormData();
    formData.append('image',file);
    formData.append('threshold',(confInput.value/100).toFixed(2));

    try{
      const resp=await fetch('/predict',{method:'POST',body:formData});
      const data=await resp.json();
      if(data.error)continue;

      const cars=data.detections.filter(d=>d.label==='car').length;
      const peds=data.detections.filter(d=>d.label==='pedestrian').length;

      allResults.push({filename:file.name,...data,cars,peds});
      totalCars+=cars; totalPeds+=peds;
      totalDets+=data.detections.length;
      totalInf+=data.inference_time;
      perImage.push({name:file.name,cars,peds,total:data.detections.length,time:data.inference_time});

      // Add tile to mosaic
      const tile=document.createElement('div');
      tile.className='tile';
      const carBadge=cars>0?`<span class="tbadge car">${cars} car${cars>1?'s':''}</span>`:'';
      const pedBadge=peds>0?`<span class="tbadge ped">${peds} ped${peds>1?'s':''}</span>`:'';
      tile.innerHTML=`
        <img src="${data.annotated}" alt="${file.name}">
        <div class="tile-info">
          <span class="tile-name" title="${file.name}">${file.name}</span>
          <div class="tile-badges">${carBadge}${pedBadge}</div>
        </div>`;
      tile.addEventListener('click',()=>{
        openLightbox(data.annotated,`${file.name} — ${cars} car(s), ${peds} ped(s) — ${data.inference_time.toFixed(3)}s`);
      });
      mosaic.appendChild(tile);

    }catch(e){console.error('Error processing',file.name,e)}

    // Progress
    const pct=((i+1)/total*100);
    progFill.style.width=pct+'%';
    progText.textContent=`${i+1} / ${total}`;
    const elapsed=(performance.now()-t0)/1000;
    const fps=(i+1)/elapsed;
    const remaining=(total-i-1)/Math.max(fps,0.01);
    progEta.textContent='ETA: '+Math.ceil(remaining)+'s';
  }

  // Summary
  const avgInf=perImage.length>0?totalInf/perImage.length:0;
  const maxDets=Math.max(1,...perImage.map(p=>p.total));

  summaryCards.innerHTML=`
    <div class="scard car"><div class="num">${totalCars}</div><div class="lbl">Total cars</div></div>
    <div class="scard ped"><div class="num">${totalPeds}</div><div class="lbl">Total pedestrians</div></div>
    <div class="scard"><div class="num">${totalDets}</div><div class="lbl">Total detections</div></div>
    <div class="scard"><div class="num">${perImage.length}</div><div class="lbl">Images processed</div></div>
    <div class="scard time"><div class="num">${avgInf.toFixed(3)}s</div><div class="lbl">Avg inference</div></div>
    <div class="scard fps"><div class="num">${(perImage.length/((performance.now()-t0)/1000)).toFixed(1)}</div><div class="lbl">Images / sec</div></div>
  `;

  // Per-image chart
  let chartHtml='<div class="chart-wrap"><h3>Detections per image</h3>';
  perImage.forEach(p=>{
    const carW=(p.cars/maxDets*100).toFixed(1);
    const pedW=(p.peds/maxDets*100).toFixed(1);
    const shortName=p.name.length>12?p.name.slice(0,10)+'..':p.name;
    chartHtml+=`<div class="bar-row">
      <span class="bar-label">${shortName}</span>
      <div class="bar-track">
        <div class="bar-fill car" style="width:${carW}%;display:inline-block;position:absolute;left:0"></div>
        <div class="bar-fill ped" style="width:${pedW}%;display:inline-block;position:absolute;left:${carW}%"></div>
      </div>
      <span class="bar-val">${p.total}</span>
    </div>`;
  });
  chartHtml+='</div>';

  // Class distribution pie-style summary
  const carPct=totalDets>0?(totalCars/totalDets*100).toFixed(0):0;
  const pedPct=totalDets>0?(totalPeds/totalDets*100).toFixed(0):0;
  const otherPct=totalDets>0?((totalDets-totalCars-totalPeds)/totalDets*100).toFixed(0):0;
  chartHtml+=`<div class="chart-wrap" style="max-width:240px"><h3>Class distribution</h3>
    <div class="bar-row"><span class="bar-label">Cars</span>
      <div class="bar-track"><div class="bar-fill car" style="width:${carPct}%"></div></div>
      <span class="bar-val">${carPct}%</span></div>
    <div class="bar-row"><span class="bar-label">Pedestrians</span>
      <div class="bar-track"><div class="bar-fill ped" style="width:${pedPct}%"></div></div>
      <span class="bar-val">${pedPct}%</span></div>
    ${parseInt(otherPct)>0?`<div class="bar-row"><span class="bar-label">Other</span>
      <div class="bar-track"><div class="bar-fill" style="width:${otherPct}%;background:var(--text2)"></div></div>
      <span class="bar-val">${otherPct}%</span></div>`:''}
  </div>`;

  chartRow.innerHTML=chartHtml;
  summary.style.display='block';

  statusEl.textContent=`Done — ${perImage.length} images, ${totalDets} detections`;
  progEta.textContent='Complete';
  btnGo.disabled=false;
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str) -> Flask:
    """Create the batch mosaic demo Flask app.

    Args:
        model_path: Path to model.onnx.
        labels_path: Path to labels.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB total

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

    @app.route("/")
    def index():
        """Serve the batch mosaic page."""
        return render_template_string(HTML)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Process a single image and return the annotated result.

        Called once per image by the frontend in a sequential loop.

        Expects multipart form:
            - image: the image file.
            - threshold: confidence threshold.

        Returns JSON:
            {
                "annotated": "<base64 data URI>",
                "detections": [...],
                "inference_time": 0.432
            }
        """
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        threshold = float(request.form.get("threshold", "0.5"))

        try:
            pil_img = Image.open(file.stream)
        except Exception as exc:
            return jsonify({"error": f"Invalid image: {exc}"}), 400

        IH, IW = 600, 800
        tensor, oh, ow = preprocess(pil_img, IH, IW)

        t0 = time.time()
        res = session.run(None, {inp_name: tensor})
        elapsed = time.time() - t0

        ann, dets = annotate(pil_img, res[0], res[1], res[2], res[3],
                             class_names, IH, IW, thr=threshold)

        return jsonify({
            "annotated": pil_to_b64(ann),
            "detections": dets,
            "inference_time": round(elapsed, 4),
        })

    return app


def main() -> None:
    """Parse arguments and start the batch mosaic demo server."""
    parser = argparse.ArgumentParser(description="Batch mosaic grid demo")
    parser.add_argument("--model", default=os.path.join("onnx_model", "train_artifacts", "model.onnx"))
    parser.add_argument("--labels", default=os.path.join("onnx_model", "train_artifacts", "labels.json"))
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found at '{args.model}'")
        return
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels not found at '{args.labels}'")
        return

    app = create_app(args.model, args.labels)
    print(f"\nBatch Mosaic demo at http://{args.host}:{args.port}")
    print("Select multiple images and process them all at once.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
