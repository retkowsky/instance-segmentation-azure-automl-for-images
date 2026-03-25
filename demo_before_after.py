"""
Before / After interactive slider demo for instance segmentation.

Upload an image, run ONNX inference, then drag a vertical handle
left-right to reveal the segmentation overlay on top of the original.

Multiple overlay modes: full (masks + boxes + labels), masks only,
boxes only, and silhouette (black background with colored masks).

Launch:
    python demo_before_after.py --model onnx_model/train_artifacts/model.onnx \\
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
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify
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


def build_overlays(
    pil_img: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    class_names: list[str],
    ih: int = 600,
    iw: int = 800,
    thr: float = 0.5,
    alpha: float = 0.5,
) -> dict[str, Image.Image]:
    """Build multiple overlay variants for the slider.

    Produces four versions:
        - full: masks + bounding boxes + labels
        - masks_only: colored masks with no boxes or text
        - boxes_only: bounding boxes + labels, no masks
        - silhouette: black background, detected objects shown with colored masks

    Args:
        pil_img: Original image.
        boxes: Detection boxes (n, 4).
        labels: Class IDs (n,).
        scores: Confidence scores (n,).
        masks: Masks (n, 1, H, W).
        class_names: Ordered class names.
        ih: Model input height.
        iw: Model input width.
        thr: Score threshold.
        alpha: Mask transparency.

    Returns:
        Dict mapping mode names to PIL Images, plus 'detections' list.
    """
    rgb = np.array(pil_img.convert("RGB"))
    oh, ow = rgb.shape[:2]
    sx, sy = ow / iw, oh / ih

    overlay_full = rgb.copy()
    overlay_masks = rgb.copy()
    overlay_boxes = rgb.copy()
    silhouette_bg = np.zeros_like(rgb)
    combined_mask = np.zeros((oh, ow), dtype=np.uint8)

    dets: list[dict] = []

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
        combined_mask = np.maximum(combined_mask, m)
        cm = np.zeros_like(rgb)
        cm[m > 0] = c

        # Full overlay: masks + boxes + labels
        overlay_full = cv2.addWeighted(overlay_full, 1.0, cm, alpha, 0)
        cv2.rectangle(overlay_full, (x1, y1), (x2, y2), c, 2)
        txt = f"{ln.upper()} {sc:.0%}"
        (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(overlay_full, (x1, y1 - th2 - 10), (x1 + tw2 + 8, y1), c, -1)
        cv2.putText(overlay_full, txt, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Masks only
        overlay_masks = cv2.addWeighted(overlay_masks, 1.0, cm, alpha, 0)

        # Boxes only
        cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), c, 2)
        cv2.rectangle(overlay_boxes, (x1, y1 - th2 - 10), (x1 + tw2 + 8, y1), c, -1)
        cv2.putText(overlay_boxes, txt, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Silhouette
        silhouette_bg = cv2.addWeighted(silhouette_bg, 1.0, cm, 0.8, 0)

        dets.append({"label": ln, "score": round(sc, 4), "bbox": [x1, y1, x2, y2]})

    # Silhouette: show original pixels only where mask exists
    sil = np.zeros_like(rgb)
    sil[combined_mask > 0] = rgb[combined_mask > 0]
    # Tint the masked region slightly
    sil = cv2.addWeighted(sil, 1.0, silhouette_bg, 0.4, 0)

    dets.sort(key=lambda d: d["score"], reverse=True)

    return {
        "full": Image.fromarray(overlay_full),
        "masks_only": Image.fromarray(overlay_masks),
        "boxes_only": Image.fromarray(overlay_boxes),
        "silhouette": Image.fromarray(sil),
        "detections": dets,
    }


def pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
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
<title>Before / After — Instance Segmentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#06080c;--surface:#0c0f15;--surface2:#141820;--border:#1e2330;
  --text:#e4e7ef;--text2:#6a7088;--car:#2ecc71;--ped:#3498db;--accent:#2ecc71;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* ═══ HEADER ═══ */
.hdr{display:flex;align-items:center;gap:1rem;padding:.8rem 1.5rem;
  background:var(--surface);border-bottom:1px solid var(--border)}
.hdr h1{font-size:1rem;font-weight:700}
.hdr .badge{font-family:var(--mono);font-size:.68rem;padding:.2rem .55rem;
  border-radius:20px;background:rgba(46,204,113,.12);color:var(--accent);
  border:1px solid rgba(46,204,113,.25)}
.hdr .spacer{flex:1}

/* ═══ LAYOUT ═══ */
.main{max-width:1200px;margin:0 auto;padding:1.5rem}

/* ═══ UPLOAD STRIP ═══ */
.upload-strip{display:flex;align-items:center;gap:1rem;padding:1rem 1.2rem;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;flex-wrap:wrap}
.upload-strip input[type=file]{display:none}
.btn{padding:.5rem 1rem;border-radius:8px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-family:var(--sans);
  font-size:.82rem;font-weight:600;cursor:pointer;transition:all .2s;
  display:inline-flex;align-items:center;gap:.35rem}
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
.fname{font-family:var(--mono);font-size:.78rem;color:var(--accent)}

/* ═══ MODE PILLS ═══ */
.mode-bar{display:flex;gap:.4rem;margin-top:1rem}
.mode-pill{font-family:var(--mono);font-size:.72rem;padding:.35rem .7rem;
  border-radius:20px;border:1px solid var(--border);background:var(--surface);
  color:var(--text2);cursor:pointer;transition:all .2s;user-select:none}
.mode-pill:hover{border-color:var(--text);color:var(--text)}
.mode-pill.active{background:rgba(46,204,113,.12);border-color:var(--accent);color:var(--accent)}

/* ═══ SLIDER CONTAINER ═══ */
.slider-container{position:relative;margin-top:1rem;border-radius:12px;
  overflow:hidden;background:var(--surface);border:1px solid var(--border);
  cursor:col-resize;user-select:none;touch-action:none}
.slider-container img{display:block;width:100%;height:auto;pointer-events:none}
.slider-container .layer{position:absolute;top:0;left:0;width:100%;height:100%}
.slider-container .layer img{width:100%;height:100%;object-fit:cover}
.slider-container .after-layer{clip-path:inset(0 0 0 50%)}

/* Handle */
.slider-handle{position:absolute;top:0;bottom:0;width:3px;background:var(--accent);
  left:50%;transform:translateX(-50%);z-index:10;pointer-events:none;
  box-shadow:0 0 12px rgba(46,204,113,.4)}
.slider-handle::before,.slider-handle::after{content:'';position:absolute;
  left:50%;transform:translateX(-50%);width:0;height:0;
  border-left:6px solid transparent;border-right:6px solid transparent}
.slider-handle::before{top:0;border-top:8px solid var(--accent)}
.slider-handle::after{bottom:0;border-bottom:8px solid var(--accent)}
.handle-knob{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:40px;height:40px;border-radius:50%;background:var(--accent);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 20px rgba(46,204,113,.5);pointer-events:none}
.handle-knob svg{width:20px;height:20px;fill:#000}

/* Labels */
.slider-label{position:absolute;top:12px;padding:.3rem .7rem;
  font-family:var(--mono);font-size:.72rem;border-radius:6px;
  background:rgba(0,0,0,.7);backdrop-filter:blur(6px);z-index:5;
  pointer-events:none}
.slider-label.left{left:12px;color:var(--text2)}
.slider-label.right{right:12px;color:var(--accent)}

/* Placeholder */
.placeholder{display:flex;align-items:center;justify-content:center;
  min-height:400px;color:var(--text2);font-size:.95rem}

/* ═══ STATS STRIP ═══ */
.stats-strip{display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:.6rem .9rem;display:flex;align-items:center;gap:.5rem;
  font-family:var(--mono);font-size:.78rem}
.stat .num{font-weight:700;font-size:1rem}
.stat.car .num{color:var(--car)}
.stat.ped .num{color:var(--ped)}
.stat.time .num{color:#f39c12}

/* ═══ SPINNER ═══ */
.spinner-wrap{position:absolute;inset:0;display:none;align-items:center;
  justify-content:center;background:rgba(6,8,12,.85);z-index:20;border-radius:12px}
.spinner-wrap.active{display:flex}
.spinner{width:40px;height:40px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ═══ TIPS ═══ */
.tip{font-size:.78rem;color:var(--text2);margin-top:.8rem;text-align:center;
  font-family:var(--mono)}

footer{text-align:center;padding:1.2rem;color:var(--text2);font-size:.72rem;margin-top:2rem}
</style>
</head>
<body>

<div class="hdr">
  <h1>Before / After</h1>
  <div class="badge">ONNX &middot; Mask R-CNN &middot; Instance Segmentation</div>
  <div class="spacer"></div>
</div>

<div class="main">
  <!-- Upload controls -->
  <div class="upload-strip">
    <input type="file" id="file-input" accept="image/*">
    <button class="btn" onclick="document.getElementById('file-input').click()">Choose image</button>
    <span class="fname" id="fname"></span>
    <div class="sep"></div>
    <div class="slider-w">
      <span class="ctrl-label">Confidence</span>
      <input type="range" id="conf" min="10" max="95" value="50">
      <span class="val" id="conf-val">0.50</span>
    </div>
    <div class="sep"></div>
    <button class="btn go" id="btn-go" disabled>Run segmentation</button>
  </div>

  <!-- Mode pills -->
  <div class="mode-bar" id="mode-bar" style="display:none">
    <div class="mode-pill active" data-mode="full">Full overlay</div>
    <div class="mode-pill" data-mode="masks_only">Masks only</div>
    <div class="mode-pill" data-mode="boxes_only">Boxes only</div>
    <div class="mode-pill" data-mode="silhouette">Silhouette</div>
  </div>

  <!-- Slider -->
  <div class="slider-container" id="slider" style="display:none">
    <img id="img-before" alt="before">
    <div class="layer after-layer" id="after-layer">
      <img id="img-after" alt="after">
    </div>
    <div class="slider-handle" id="handle">
      <div class="handle-knob">
        <svg viewBox="0 0 24 24"><path d="M8 5l-5 7 5 7M16 5l5 7-5 7"/></svg>
      </div>
    </div>
    <div class="slider-label left">Original</div>
    <div class="slider-label right" id="right-label">Segmented</div>
    <div class="spinner-wrap" id="spinner"><div class="spinner"></div></div>
  </div>

  <!-- Placeholder -->
  <div class="placeholder" id="placeholder">
    Upload an image and click <b>Run segmentation</b> to see the before/after comparison
  </div>

  <!-- Stats -->
  <div class="stats-strip" id="stats" style="display:none"></div>

  <div class="tip" id="tip" style="display:none">Drag the handle left and right to compare</div>
</div>

<footer>Before / After — Azure AutoML for Images &middot; ONNX Runtime &middot; Mask R-CNN</footer>

<script>
const fileInput=document.getElementById('file-input'),
  fnameEl=document.getElementById('fname'),
  confInput=document.getElementById('conf'),
  confVal=document.getElementById('conf-val'),
  btnGo=document.getElementById('btn-go'),
  sliderEl=document.getElementById('slider'),
  imgBefore=document.getElementById('img-before'),
  imgAfter=document.getElementById('img-after'),
  afterLayer=document.getElementById('after-layer'),
  handle=document.getElementById('handle'),
  spinner=document.getElementById('spinner'),
  placeholder=document.getElementById('placeholder'),
  statsEl=document.getElementById('stats'),
  modeBar=document.getElementById('mode-bar'),
  rightLabel=document.getElementById('right-label'),
  tipEl=document.getElementById('tip');

let selectedFile=null, overlays={}, currentMode='full', sliderPos=0.5;

confInput.addEventListener('input',()=>confVal.textContent=(confInput.value/100).toFixed(2));

fileInput.addEventListener('change',()=>{
  if(!fileInput.files.length)return;
  selectedFile=fileInput.files[0];
  fnameEl.textContent=selectedFile.name;
  btnGo.disabled=false;
});

// Mode pills
document.querySelectorAll('.mode-pill').forEach(pill=>{
  pill.addEventListener('click',()=>{
    document.querySelectorAll('.mode-pill').forEach(p=>p.classList.remove('active'));
    pill.classList.add('active');
    currentMode=pill.dataset.mode;
    if(overlays[currentMode]){
      imgAfter.src=overlays[currentMode];
      const labels={full:'Full overlay',masks_only:'Masks only',boxes_only:'Boxes only',silhouette:'Silhouette'};
      rightLabel.textContent=labels[currentMode]||'Segmented';
    }
  });
});

// Run inference
btnGo.addEventListener('click',async()=>{
  if(!selectedFile)return;
  btnGo.disabled=true;
  placeholder.style.display='none';
  sliderEl.style.display='block';
  spinner.classList.add('active');
  modeBar.style.display='none';
  statsEl.style.display='none';
  tipEl.style.display='none';

  // Show original immediately
  const reader=new FileReader();
  reader.onload=e=>{imgBefore.src=e.target.result};
  reader.readAsDataURL(selectedFile);

  const formData=new FormData();
  formData.append('image',selectedFile);
  formData.append('threshold',(confInput.value/100).toFixed(2));

  try{
    const resp=await fetch('/predict',{method:'POST',body:formData});
    const data=await resp.json();
    if(data.error){alert(data.error);return}

    // Store all overlays
    overlays=data.overlays;
    imgAfter.src=overlays[currentMode];

    // Stats
    const dets=data.detections;
    const cars=dets.filter(d=>d.label==='car').length;
    const peds=dets.filter(d=>d.label==='pedestrian').length;
    statsEl.innerHTML=`
      <div class="stat car"><span>Cars</span><span class="num">${cars}</span></div>
      <div class="stat ped"><span>Pedestrians</span><span class="num">${peds}</span></div>
      <div class="stat"><span>Total</span><span class="num">${dets.length}</span></div>
      <div class="stat time"><span>Inference</span><span class="num">${data.inference_time.toFixed(3)}s</span></div>
    `;
    statsEl.style.display='flex';
    modeBar.style.display='flex';
    tipEl.style.display='block';

    // Reset slider to 50%
    setSlider(0.5);

  }catch(e){
    alert('Error: '+e.message);
  }finally{
    spinner.classList.remove('active');
    btnGo.disabled=false;
  }
});

// ═══ SLIDER INTERACTION ═══
function setSlider(pct){
  sliderPos=Math.max(0,Math.min(1,pct));
  afterLayer.style.clipPath=`inset(0 0 0 ${sliderPos*100}%)`;
  handle.style.left=sliderPos*100+'%';
}

let dragging=false;
function getPos(e){
  const rect=sliderEl.getBoundingClientRect();
  const clientX=e.touches?e.touches[0].clientX:e.clientX;
  return(clientX-rect.left)/rect.width;
}

sliderEl.addEventListener('mousedown',e=>{dragging=true;setSlider(getPos(e))});
sliderEl.addEventListener('touchstart',e=>{dragging=true;setSlider(getPos(e))},{passive:true});
window.addEventListener('mousemove',e=>{if(dragging)setSlider(getPos(e))});
window.addEventListener('touchmove',e=>{if(dragging)setSlider(getPos(e))},{passive:true});
window.addEventListener('mouseup',()=>dragging=false);
window.addEventListener('touchend',()=>dragging=false);

// Keyboard: left/right arrows
window.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')setSlider(sliderPos-0.02);
  if(e.key==='ArrowRight')setSlider(sliderPos+0.02);
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str) -> Flask:
    """Create the before/after slider demo Flask app.

    Args:
        model_path: Path to model.onnx.
        labels_path: Path to labels.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

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
        """Serve the before/after demo page."""
        return render_template_string(HTML)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Process an uploaded image and return multiple overlay variants.

        Expects multipart form with:
            - image: the uploaded image file.
            - threshold: confidence threshold (0..1).

        Returns JSON:
            {
                "overlays": {
                    "full": "<base64 data URI>",
                    "masks_only": "<base64 data URI>",
                    "boxes_only": "<base64 data URI>",
                    "silhouette": "<base64 data URI>"
                },
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

        result = build_overlays(
            pil_img, res[0], res[1], res[2], res[3],
            class_names, IH, IW, thr=threshold,
        )

        overlay_b64 = {}
        for mode in ["full", "masks_only", "boxes_only", "silhouette"]:
            overlay_b64[mode] = pil_to_b64(result[mode])

        return jsonify({
            "overlays": overlay_b64,
            "detections": result["detections"],
            "inference_time": round(elapsed, 4),
        })

    return app


def main() -> None:
    """Parse arguments and start the before/after demo server."""
    parser = argparse.ArgumentParser(description="Before/After slider demo")
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
    print(f"\nBefore/After demo at http://{args.host}:{args.port}")
    print("Upload an image and drag the slider.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
