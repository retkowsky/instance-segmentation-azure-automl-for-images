"""
Live webcam / RTSP instance segmentation demo with intrusion alerts.

Captures frames from the browser webcam OR a backend RTSP stream,
runs ONNX Mask R-CNN inference, and displays results with a HUD
overlay. Shows red intrusion alerts when cars or people are detected.

Launch (webcam only):
    python demo_webcam_v2.py --model onnx_model/train_artifacts/model.onnx \\
                             --labels onnx_model/train_artifacts/labels.json

Launch (with default RTSP):
    python demo_webcam_v2.py --model onnx_model/train_artifacts/model.onnx \\
                             --labels onnx_model/train_artifacts/labels.json \\
                             --rtsp "rtsp://your-stream-url"

Then open http://127.0.0.1:5050

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

import threading
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# ONNX inference (same pipeline as your existing app)
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_COLOR: tuple[int, int, int] = (231, 76, 60)
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "car": (46, 204, 113),
    "pedestrian": (52, 152, 219),
}


def preprocess(pil_img: Image.Image, th: int = 600, tw: int = 800) -> tuple[np.ndarray, int, int]:
    """Preprocess a PIL image for Mask R-CNN ONNX inference.

    Args:
        pil_img: Input image.
        th: Target height.
        tw: Target width.

    Returns:
        Tuple of (tensor, original_height, original_width).
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
    ih: int = 600,
    iw: int = 800,
    thr: float = 0.5,
    mthr: float = 0.5,
    alpha: float = 0.45,
    show_masks: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True,
) -> tuple[Image.Image, list[dict]]:
    """Overlay masks and bounding boxes onto the image.

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
        mthr: Mask binarization threshold.
        alpha: Mask transparency.
        show_masks: Whether to draw segmentation masks.
        show_boxes: Whether to draw bounding boxes.
        show_labels: Whether to draw label text.

    Returns:
        Tuple of (annotated PIL image, detection list).
    """
    rgb = np.array(pil_img.convert("RGB"))
    oh, ow = rgb.shape[:2]
    ov = rgb.copy()
    sx, sy = ow / iw, oh / ih
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

        if show_masks:
            m = cv2.resize((masks[i, 0] > mthr).astype(np.uint8), (ow, oh))
            cm = np.zeros_like(ov)
            cm[m > 0] = c
            ov = cv2.addWeighted(ov, 1.0, cm, alpha, 0)

        if show_boxes:
            cv2.rectangle(ov, (x1, y1), (x2, y2), c, 2)

        if show_labels:
            txt = f"{ln.upper()} {sc:.0%}"
            (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(ov, (x1, y1 - th2 - 10), (x1 + tw2 + 8, y1), c, -1)
            cv2.putText(ov, txt, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        dets.append({"label": ln, "score": round(sc, 4), "bbox": [x1, y1, x2, y2]})

    dets.sort(key=lambda d: d["score"], reverse=True)
    return Image.fromarray(ov), dets


def pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 80) -> str:
    """Encode a PIL image to a base64 data URI.

    Args:
        img: PIL Image.
        fmt: Image format.
        quality: JPEG quality (lower = faster transfer).

    Returns:
        Base64-encoded data URI.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────
# RTSP frame grabber (background thread)
# ──────────────────────────────────────────────────────────────────────

class RTSPGrabber:
    """Continuously grabs frames from an RTSP stream in a background thread."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.frame: np.ndarray | None = None
        self.running = False
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Open the stream and start background grabbing.

        Returns:
            True if the stream opened successfully.
        """
        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            print(f"[RTSP] Failed to open: {self.url}")
            return False
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[RTSP] Streaming from: {self.url}")
        return True

    def _loop(self) -> None:
        """Background loop that continuously grabs the latest frame."""
        while self.running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                time.sleep(0.1)
                self._cap.release()
                self._cap = cv2.VideoCapture(self.url)

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame or None."""
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        """Stop grabbing and release resources."""
        self.running = False
        if self._cap:
            self._cap.release()


# ──────────────────────────────────────────────────────────────────────
# HTML template — HUD-style webcam demo
# ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Segmentation — Intrusion Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #000; --surface: #0a0c10; --border: #1a1e28;
  --text: #e8eaef; --text2: #6a7080;
  --car: #2ecc71; --ped: #3498db; --accent: #2ecc71;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);
  min-height:100vh;overflow:hidden}

/* ═══ LAYOUT ═══ */
.app{display:grid;grid-template-rows:auto 1fr auto auto;height:100vh}

/* ═══ HEADER ═══ */
.hdr{display:flex;align-items:center;gap:1rem;padding:.8rem 1.5rem;
  background:var(--surface);border-bottom:1px solid var(--border)}
.hdr .dot{width:10px;height:10px;border-radius:50%;background:#e74c3c;
  animation:pulse 1.5s ease infinite}
.hdr .dot.live{background:var(--accent);animation:pulse-live 1s ease infinite}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
@keyframes pulse-live{0%,100%{opacity:1}50%{opacity:.5}}
.hdr h1{font-size:1rem;font-weight:700;letter-spacing:-.01em}
.hdr .badge{font-family:var(--mono);font-size:.68rem;padding:.2rem .55rem;
  border-radius:20px;background:rgba(46,204,113,.12);color:var(--accent);
  border:1px solid rgba(46,204,113,.25)}
.hdr .spacer{flex:1}
.hdr .fps{font-family:var(--mono);font-size:.82rem;color:var(--accent)}

/* ═══ MAIN VIEWPORT ═══ */
.viewport{position:relative;overflow:hidden;background:#000;display:flex;align-items:center;justify-content:center}
.viewport video{display:none}
.viewport canvas, .viewport img{
  max-width:100%;max-height:100%;object-fit:contain}
#output{z-index:2}
#placeholder{color:var(--text2);font-size:1.1rem;z-index:1;position:absolute}

/* ═══ HUD OVERLAY ═══ */
.hud{position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:5}
.hud-corner{position:absolute;width:40px;height:40px;opacity:.25}
.hud-corner.tl{top:16px;left:16px;border-top:2px solid var(--accent);border-left:2px solid var(--accent)}
.hud-corner.tr{top:16px;right:16px;border-top:2px solid var(--accent);border-right:2px solid var(--accent)}
.hud-corner.bl{bottom:16px;left:16px;border-bottom:2px solid var(--accent);border-left:2px solid var(--accent)}
.hud-corner.br{bottom:16px;right:16px;border-bottom:2px solid var(--accent);border-right:2px solid var(--accent)}

/* ═══ STATS PANEL (overlaid) ═══ */
.stats-overlay{position:absolute;top:20px;right:20px;z-index:10;pointer-events:none;
  display:flex;flex-direction:column;gap:.5rem;min-width:180px}
.stat-pill{font-family:var(--mono);font-size:.78rem;padding:.45rem .75rem;
  border-radius:8px;background:rgba(10,12,16,.85);border:1px solid var(--border);
  backdrop-filter:blur(8px);display:flex;justify-content:space-between;gap:1rem}
.stat-pill .val{font-weight:700}
.stat-pill.car .val{color:var(--car)}
.stat-pill.ped .val{color:var(--ped)}
.stat-pill.fps .val{color:var(--accent)}
.stat-pill.time .val{color:#f39c12}

/* ═══ DETECTION TICKER ═══ */
.ticker{position:absolute;bottom:20px;left:20px;right:20px;z-index:10;
  pointer-events:none;display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center}
.tick{font-family:var(--mono);font-size:.72rem;padding:.3rem .6rem;
  border-radius:6px;background:rgba(10,12,16,.8);border:1px solid var(--border);
  backdrop-filter:blur(6px);animation:tick-in .3s ease}
@keyframes tick-in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.tick.car{border-color:rgba(46,204,113,.4);color:var(--car)}
.tick.ped{border-color:rgba(52,152,219,.4);color:var(--ped)}

/* ═══ CONTROLS BAR ═══ */
.controls{display:flex;align-items:center;gap:1rem;padding:.7rem 1.5rem;
  background:var(--surface);border-top:1px solid var(--border);flex-wrap:wrap}
.btn{padding:.5rem 1.1rem;border-radius:8px;border:1px solid var(--border);
  background:var(--surface);color:var(--text);font-family:var(--sans);
  font-size:.82rem;font-weight:600;cursor:pointer;transition:all .2s;
  display:flex;align-items:center;gap:.4rem}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.active{background:rgba(46,204,113,.12);border-color:var(--accent);color:var(--accent)}
.btn.start{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.start:hover{background:#27ae60}
.btn.stop{background:#e74c3c;color:#fff;border-color:#e74c3c}
.btn.stop:hover{background:#c0392b}
.sep{width:1px;height:24px;background:var(--border)}
.ctrl-label{font-size:.78rem;color:var(--text2)}
.slider-wrap{display:flex;align-items:center;gap:.5rem}
.slider-wrap input[type=range]{width:100px;height:4px;-webkit-appearance:none;
  background:var(--border);border-radius:2px;outline:none}
.slider-wrap input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;background:var(--accent);cursor:pointer}
.slider-wrap .val{font-family:var(--mono);font-size:.75rem;color:var(--accent);min-width:32px}

/* ═══ TOGGLE ═══ */
.toggle{display:flex;align-items:center;gap:.4rem;cursor:pointer;user-select:none}
.toggle input{display:none}
.toggle .track{width:28px;height:16px;border-radius:8px;background:var(--border);
  position:relative;transition:background .2s}
.toggle input:checked+.track{background:var(--accent)}
.toggle .thumb{position:absolute;top:2px;left:2px;width:12px;height:12px;
  border-radius:50%;background:#fff;transition:transform .2s}
.toggle input:checked+.track .thumb{transform:translateX(12px)}
.toggle span{font-size:.78rem;color:var(--text2)}

/* ═══ INTRUSION ALERT ═══ */
.alert-banner{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:50;
  pointer-events:none;display:none;flex-direction:column;align-items:center;gap:.6rem}
.alert-banner.show{display:flex;animation:apulse .6s ease infinite}
@keyframes apulse{0%,100%{opacity:1}50%{opacity:.7}}
.alert-card{font-family:var(--mono);font-size:.9rem;font-weight:700;padding:.7rem 1.5rem;
  border-radius:10px;text-transform:uppercase;letter-spacing:.08em;display:flex;
  align-items:center;gap:.6rem;backdrop-filter:blur(10px);white-space:nowrap}
.alert-card.person{background:rgba(231,76,60,.88);color:#fff;border:2px solid #e74c3c;
  box-shadow:0 0 30px rgba(231,76,60,.5)}
.alert-card.vehicle{background:rgba(243,156,18,.88);color:#000;border:2px solid #f39c12;
  box-shadow:0 0 30px rgba(243,156,18,.5)}
.vp-flash{position:absolute;inset:0;pointer-events:none;z-index:40;
  border:4px solid transparent;transition:border-color .15s}
.vp-flash.danger{border-color:#e74c3c;animation:bflash .5s ease 3}
@keyframes bflash{0%,100%{border-color:#e74c3c}50%{border-color:transparent}}

/* ═══ ALERT LOG ═══ */
.alert-log{background:var(--surface);border-top:1px solid var(--border);
  max-height:110px;overflow-y:auto;display:none}
.alert-log.show{display:block}
.log-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:.35rem 1.5rem;border-bottom:1px solid var(--border)}
.log-hdr span{font-family:var(--mono);font-size:.7rem;color:var(--text2);
  text-transform:uppercase;letter-spacing:.06em}
.log-count{font-family:var(--mono);font-size:.68rem;padding:.1rem .4rem;
  border-radius:8px;background:rgba(231,76,60,.15);color:#e74c3c;
  border:1px solid rgba(231,76,60,.3)}
.log-item{display:flex;align-items:center;gap:.5rem;padding:.3rem 1.5rem;
  border-bottom:1px solid rgba(255,255,255,.03);font-size:.72rem}
.log-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.log-dot.person{background:#e74c3c;box-shadow:0 0 5px #e74c3c}
.log-dot.vehicle{background:#f39c12;box-shadow:0 0 5px #f39c12}
.log-time{font-family:var(--mono);color:var(--text2);min-width:55px;font-size:.65rem}

/* ═══ RTSP ═══ */
.rtsp-input{font-family:var(--mono);font-size:.72rem;padding:.35rem .5rem;
  background:var(--surface);border:1px solid var(--border);border-radius:6px;
  color:var(--text);width:260px;outline:none}
.rtsp-input:focus{border-color:#f39c12}
.rtsp-input::placeholder{color:var(--text2)}
.btn.rtsp-b{background:rgba(243,156,18,.12);border-color:#f39c12;color:#f39c12}
</style>
</head>
<body>
<div class="app">
  <!-- HEADER -->
  <div class="hdr">
    <div class="dot" id="live-dot"></div>
    <h1>Live Instance Segmentation</h1>
    <div class="badge">ONNX &middot; Mask R-CNN</div>
    <div class="spacer"></div>
    <div class="fps" id="fps-hdr">— FPS</div>
  </div>

  <!-- VIEWPORT -->
  <div class="viewport" id="viewport">
    <video id="cam" autoplay playsinline muted></video>
    <canvas id="capture" style="display:none"></canvas>
    <img id="output" alt="annotated frame" style="display:none">
    <div id="placeholder">Click <b>Start camera</b> or connect an <b>RTSP stream</b></div>

    <!-- HUD corners -->
    <div class="hud">
      <div class="hud-corner tl"></div>
      <div class="hud-corner tr"></div>
      <div class="hud-corner bl"></div>
      <div class="hud-corner br"></div>
    </div>

    <!-- Stats overlay -->
    <div class="stats-overlay" id="stats" style="display:none">
      <div class="stat-pill fps"><span>FPS</span><span class="val" id="st-fps">—</span></div>
      <div class="stat-pill time"><span>Inference</span><span class="val" id="st-time">—</span></div>
      <div class="stat-pill car"><span>Cars</span><span class="val" id="st-cars">0</span></div>
      <div class="stat-pill ped"><span>Pedestrians</span><span class="val" id="st-peds">0</span></div>
      <div class="stat-pill"><span>Total</span><span class="val" id="st-total">0</span></div>
      <div class="stat-pill"><span>Frames</span><span class="val" id="st-frames">0</span></div>
    </div>

    <!-- Detection ticker -->
    <div class="ticker" id="ticker"></div>

    <!-- Intrusion alert banner -->
    <div class="alert-banner" id="alert-banner"></div>
    <div class="vp-flash" id="vp-flash"></div>
  </div>

  <!-- ALERT LOG -->
  <div class="alert-log" id="alert-log">
    <div class="log-hdr"><span>Intrusion log</span><span class="log-count" id="log-count">0</span></div>
    <div id="log-items"></div>
  </div>

  <!-- CONTROLS -->
  <div class="controls">
    <button class="btn start" id="btn-start">&#9654; Start camera</button>
    <button class="btn stop" id="btn-stop" style="display:none">&#9724; Stop</button>
    <div class="sep"></div>

    <input class="rtsp-input" id="rtsp-url" placeholder="rtsp://user:pass@ip:port/stream">
    <button class="btn rtsp-b" id="btn-rtsp">&#128249; RTSP</button>
    <button class="btn stop" id="btn-rtsp-stop" style="display:none">&#9724; Disconnect</button>
    <div class="sep"></div>

    <div class="slider-wrap">
      <span class="ctrl-label">Confidence</span>
      <input type="range" id="conf" min="10" max="95" value="50">
      <span class="val" id="conf-val">0.50</span>
    </div>
    <div class="sep"></div>

    <label class="toggle"><input type="checkbox" id="tgl-masks" checked><div class="track"><div class="thumb"></div></div><span>Masks</span></label>
    <label class="toggle"><input type="checkbox" id="tgl-boxes" checked><div class="track"><div class="thumb"></div></div><span>Boxes</span></label>
    <label class="toggle"><input type="checkbox" id="tgl-labels" checked><div class="track"><div class="thumb"></div></div><span>Labels</span></label>
    <label class="toggle"><input type="checkbox" id="tgl-alerts" checked><div class="track"><div class="thumb"></div></div><span>Alerts</span></label>
    <label class="toggle"><input type="checkbox" id="tgl-sound" checked><div class="track"><div class="thumb"></div></div><span>Sound</span></label>
    <div class="sep"></div>

    <div class="slider-wrap">
      <span class="ctrl-label">Capture FPS</span>
      <input type="range" id="target-fps" min="1" max="15" value="5">
      <span class="val" id="fps-val">5</span>
    </div>
  </div>
</div>

<script>
const cam=document.getElementById('cam'),capture=document.getElementById('capture'),output=document.getElementById('output'),
  ph=document.getElementById('placeholder'),btnStart=document.getElementById('btn-start'),btnStop=document.getElementById('btn-stop'),
  btnRtsp=document.getElementById('btn-rtsp'),btnRtspStop=document.getElementById('btn-rtsp-stop'),rtspUrlEl=document.getElementById('rtsp-url'),
  liveDot=document.getElementById('live-dot'),stats=document.getElementById('stats'),ticker=document.getElementById('ticker'),
  confInput=document.getElementById('conf'),confVal=document.getElementById('conf-val'),fpsHdr=document.getElementById('fps-hdr'),
  tgtFps=document.getElementById('target-fps'),fpsVal=document.getElementById('fps-val'),
  tglMasks=document.getElementById('tgl-masks'),tglBoxes=document.getElementById('tgl-boxes'),
  tglLabels=document.getElementById('tgl-labels'),tglAlerts=document.getElementById('tgl-alerts'),tglSound=document.getElementById('tgl-sound'),
  alertBanner=document.getElementById('alert-banner'),vpFlash=document.getElementById('vp-flash'),
  alertLog=document.getElementById('alert-log'),logItems=document.getElementById('log-items'),logCount=document.getElementById('log-count');

let streaming=false,rtspMode=false,intervalId=null,frameCount=0,fpsSmooth=0,lastFpsTime=performance.now(),fpsCounter=0,busy=false,alertCnt=0,lastAlertT=0;

// Audio beep for intrusion alerts
let audioCtx=null;
function beep(){if(!tglSound.checked)return;try{if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  const o=audioCtx.createOscillator(),g=audioCtx.createGain();o.connect(g);g.connect(audioCtx.destination);
  o.frequency.value=880;o.type='square';g.gain.value=.08;o.start();o.stop(audioCtx.currentTime+.12)}catch(e){}}

confInput.addEventListener('input',()=>confVal.textContent=(confInput.value/100).toFixed(2));
tgtFps.addEventListener('input',()=>{fpsVal.textContent=tgtFps.value;
  if(streaming){clearInterval(intervalId);intervalId=setInterval(captureFrame,1000/tgtFps.value)}});
btnStart.addEventListener('click',startCamera);
btnStop.addEventListener('click',stopAll);
btnRtsp.addEventListener('click',startRTSP);
btnRtspStop.addEventListener('click',stopAll);

async function startCamera(){
  try{const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'environment'}});
  cam.srcObject=stream;await cam.play();capture.width=cam.videoWidth;capture.height=cam.videoHeight;
  rtspMode=false;goLive()}catch(e){alert('Camera denied: '+e.message)}}

async function startRTSP(){
  const url=rtspUrlEl.value.trim();if(!url){alert('Enter an RTSP URL');return}
  btnRtsp.disabled=true;btnRtsp.textContent='Connecting...';
  try{const r=await fetch('/rtsp/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
  const d=await r.json();if(d.error){alert(d.error);return}
  rtspMode=true;goLive()}catch(e){alert('RTSP failed: '+e.message)}
  finally{btnRtsp.disabled=false;btnRtsp.innerHTML='&#128249; RTSP'}}

function goLive(){streaming=true;busy=false;frameCount=0;alertCnt=0;logItems.innerHTML='';logCount.textContent='0';
  ph.style.display='none';output.style.display='block';stats.style.display='flex';alertLog.classList.add('show');
  btnStart.style.display='none';btnRtsp.style.display='none';
  if(rtspMode){btnRtspStop.style.display='flex';btnStop.style.display='none'}
  else{btnStop.style.display='flex';btnRtspStop.style.display='none'}
  liveDot.classList.add('live');intervalId=setInterval(captureFrame,1000/tgtFps.value)}

function stopAll(){streaming=false;clearInterval(intervalId);
  if(cam.srcObject)cam.srcObject.getTracks().forEach(t=>t.stop());cam.srcObject=null;
  if(rtspMode)fetch('/rtsp/stop',{method:'POST'}).catch(()=>{});rtspMode=false;
  output.style.display='none';stats.style.display='none';alertBanner.classList.remove('show');
  ph.style.display='block';btnStart.style.display='flex';btnRtsp.style.display='flex';
  btnStop.style.display='none';btnRtspStop.style.display='none';
  liveDot.classList.remove('live');fpsHdr.textContent='— FPS';ticker.innerHTML=''}

async function captureFrame(){
  if(!streaming||busy)return;busy=true;
  let endpoint,body;
  if(rtspMode){endpoint='/predict_rtsp';body=JSON.stringify({threshold:confInput.value/100,
    show_masks:tglMasks.checked,show_boxes:tglBoxes.checked,show_labels:tglLabels.checked})}
  else{const ctx=capture.getContext('2d');ctx.drawImage(cam,0,0);
    const b64=capture.toDataURL('image/jpeg',0.7).split(',')[1];
    endpoint='/predict_frame';body=JSON.stringify({frame:b64,threshold:confInput.value/100,
      show_masks:tglMasks.checked,show_boxes:tglBoxes.checked,show_labels:tglLabels.checked})}

  try{const t0=performance.now();
    const resp=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body});
    const data=await resp.json();if(data.error){console.error(data.error);busy=false;return}
    output.src=data.annotated;output.style.display='block';

    frameCount++;fpsCounter++;const now=performance.now();
    if(now-lastFpsTime>=1000){fpsSmooth=fpsCounter;fpsCounter=0;lastFpsTime=now}
    document.getElementById('st-fps').textContent=fpsSmooth;
    document.getElementById('st-time').textContent=data.inference_time.toFixed(3)+'s';
    document.getElementById('st-cars').textContent=data.cars;
    document.getElementById('st-peds').textContent=data.peds;
    document.getElementById('st-total').textContent=data.total;
    document.getElementById('st-frames').textContent=frameCount;
    fpsHdr.textContent=fpsSmooth+' FPS';

    if(data.detections.length>0){ticker.innerHTML='';
      data.detections.slice(0,8).forEach(d=>{
        const cls=d.label==='car'?'car':d.label==='pedestrian'?'ped':'';
        ticker.innerHTML+=`<div class="tick ${cls}">${d.label.toUpperCase()} ${(d.score*100).toFixed(0)}%</div>`})}

    // ═══ INTRUSION ALERTS ═══
    if(tglAlerts.checked&&data.total>0){const nowMs=Date.now();const alerts=[];
      if(data.peds>0)alerts.push({type:'person',msg:`INTRUSION: ${data.peds} person${data.peds>1?'s':''} detected`});
      if(data.cars>0)alerts.push({type:'vehicle',msg:`INTRUSION: ${data.cars} vehicle${data.cars>1?'s':''} detected`});
      alertBanner.innerHTML='';alerts.forEach(a=>{
        alertBanner.innerHTML+=`<div class="alert-card ${a.type}"><span style="font-size:1.3rem">&#9888;</span>${a.msg}</div>`});
      alertBanner.classList.add('show');
      if(nowMs-lastAlertT>2000){
        vpFlash.classList.remove('danger');void vpFlash.offsetWidth;vpFlash.classList.add('danger');beep();lastAlertT=nowMs;
        const ts=new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
        alerts.forEach(a=>{alertCnt++;const el=document.createElement('div');el.className='log-item';
          el.innerHTML=`<div class="log-dot ${a.type}"></div><span class="log-time">${ts}</span><span>${a.msg}</span>`;
          logItems.prepend(el);if(logItems.children.length>50)logItems.removeChild(logItems.lastChild)});
        logCount.textContent=alertCnt}}
    else{alertBanner.classList.remove('show')}
  }catch(e){console.error('Frame error:',e)}finally{busy=false}}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str, rtsp_url: str | None = None) -> Flask:
    """Create the webcam/RTSP demo Flask application.

    Args:
        model_path: Path to model.onnx.
        labels_path: Path to labels.json.
        rtsp_url: Optional default RTSP URL to connect on startup.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    with open(labels_path) as f:
        class_names: list[str] = json.load(f)

    providers: list[str] = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    inp_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Model: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")
    print(f"Providers: {session.get_providers()}")

    # Shared RTSP grabber
    grabber: dict[str, RTSPGrabber | None] = {"instance": None}
    if rtsp_url:
        g = RTSPGrabber(rtsp_url)
        if g.start():
            grabber["instance"] = g
            print(f"[RTSP] Auto-connected to: {rtsp_url}")

    def _infer(
        pil_img: Image.Image,
        threshold: float,
        show_masks: bool,
        show_boxes: bool,
        show_labels: bool,
    ) -> dict:
        """Run ONNX inference and return JSON-ready results.

        Args:
            pil_img: Input image.
            threshold: Confidence threshold.
            show_masks: Draw masks.
            show_boxes: Draw boxes.
            show_labels: Draw labels.

        Returns:
            Dict with annotated image, detections, counts, timing.
        """
        IH, IW = 600, 800
        tensor, oh, ow = preprocess(pil_img, IH, IW)

        t0 = time.time()
        res = session.run(output_names, {inp_name: tensor})
        elapsed = time.time() - t0

        # Map outputs by name for robustness
        out_map = dict(zip(output_names, res))
        boxes = next((out_map[n] for n in output_names if "box" in n.lower()), res[0])
        lab = next((out_map[n] for n in output_names if "label" in n.lower()), res[1])
        sco = next((out_map[n] for n in output_names if "score" in n.lower()), res[2])
        msk = next((out_map[n] for n in output_names if "mask" in n.lower()), res[3])

        ann, dets = annotate(
            pil_img, boxes, lab, sco, msk,
            class_names, IH, IW, thr=threshold,
            show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels,
        )

        cars = sum(1 for d in dets if d["label"] == "car")
        peds = sum(1 for d in dets if d["label"] == "pedestrian")

        return {
            "annotated": pil_to_b64(ann, quality=75),
            "detections": dets,
            "cars": cars,
            "peds": peds,
            "total": len(dets),
            "inference_time": round(elapsed, 4),
        }

    @app.route("/")
    def index() -> str:
        """Serve the demo page."""
        return render_template_string(HTML)

    @app.route("/predict_frame", methods=["POST"])
    def predict_frame():
        """Process a single webcam frame (base64 JPEG input).

        Expects JSON:
            {
                "frame": "<base64 JPEG>",
                "threshold": 0.5,
                "show_masks": true,
                "show_boxes": true,
                "show_labels": true
            }

        Returns JSON with annotated image, detections, counts, timing.
        """
        try:
            data = request.get_json(force=True)
            img_bytes = base64.b64decode(data.get("frame", ""))
            pil_img = Image.open(io.BytesIO(img_bytes))

            return jsonify(_infer(
                pil_img,
                float(data.get("threshold", 0.5)),
                data.get("show_masks", True),
                data.get("show_boxes", True),
                data.get("show_labels", True),
            ))
        except Exception as exc:
            print(f"[ERROR] /predict_frame:\n{traceback.format_exc()}")
            return jsonify({"error": str(exc)}), 500

    @app.route("/predict_rtsp", methods=["POST"])
    def predict_rtsp():
        """Process the latest frame from the active RTSP stream.

        Expects JSON:
            {
                "threshold": 0.5,
                "show_masks": true,
                "show_boxes": true,
                "show_labels": true
            }

        Returns JSON with annotated image, detections, counts, timing.
        """
        try:
            g = grabber["instance"]
            if g is None or not g.running:
                return jsonify({"error": "No RTSP stream active"}), 400

            frame_rgb = g.get_frame()
            if frame_rgb is None:
                return jsonify({"error": "No frame available yet — stream may still be connecting"}), 400

            pil_img = Image.fromarray(frame_rgb)
            data = request.get_json(force=True)

            return jsonify(_infer(
                pil_img,
                float(data.get("threshold", 0.5)),
                data.get("show_masks", True),
                data.get("show_boxes", True),
                data.get("show_labels", True),
            ))
        except Exception as exc:
            print(f"[ERROR] /predict_rtsp:\n{traceback.format_exc()}")
            return jsonify({"error": str(exc)}), 500

    @app.route("/rtsp/start", methods=["POST"])
    def rtsp_start():
        """Start an RTSP stream grabber.

        Expects JSON: {"url": "rtsp://..."}
        Returns JSON: {"status": "connected", "url": "..."}
        """
        try:
            data = request.get_json(force=True)
            url = data.get("url", "")
            if not url:
                return jsonify({"error": "No URL provided"}), 400

            # Stop any existing grabber
            if grabber["instance"]:
                grabber["instance"].stop()
                grabber["instance"] = None

            g = RTSPGrabber(url)
            if not g.start():
                return jsonify({"error": f"Could not open RTSP stream: {url}"}), 400

            # Wait briefly for the first frame
            time.sleep(1.5)
            if g.get_frame() is None:
                time.sleep(1.5)

            grabber["instance"] = g
            return jsonify({"status": "connected", "url": url})

        except Exception as exc:
            print(f"[ERROR] /rtsp/start:\n{traceback.format_exc()}")
            return jsonify({"error": str(exc)}), 500

    @app.route("/rtsp/stop", methods=["POST"])
    def rtsp_stop():
        """Stop the active RTSP stream grabber."""
        if grabber["instance"]:
            grabber["instance"].stop()
            grabber["instance"] = None
            print("[RTSP] Stream stopped")
        return jsonify({"status": "stopped"})

    @app.route("/health")
    def health():
        """Diagnostic endpoint."""
        rtsp_status = "disconnected"
        if grabber["instance"] and grabber["instance"].running:
            rtsp_status = f"connected to {grabber['instance'].url}"
        return jsonify({
            "status": "ok",
            "classes": class_names,
            "rtsp": rtsp_status,
            "providers": session.get_providers(),
        })

    return app


def main() -> None:
    """Parse arguments and start the webcam/RTSP demo server."""
    parser = argparse.ArgumentParser(description="Live webcam/RTSP segmentation with intrusion alerts")
    parser.add_argument("--model", default=os.path.join("onnx_model", "train_artifacts", "model.onnx"))
    parser.add_argument("--labels", default=os.path.join("onnx_model", "train_artifacts", "labels.json"))
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rtsp", default=None, help="Default RTSP stream URL to connect on startup")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found at '{args.model}'")
        return
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels not found at '{args.labels}'")
        return

    app = create_app(args.model, args.labels, rtsp_url=args.rtsp)
    print(f"\nWebcam/RTSP demo at http://{args.host}:{args.port}")
    print("Open in browser — use webcam or enter an RTSP URL.")
    if args.rtsp:
        print(f"Default RTSP: {args.rtsp}")
    print()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
