"""
Blind spot detection demo for instance segmentation.

Detects vehicles and pedestrians, then computes pairwise proximity
between each pedestrian mask and each vehicle mask. Flags dangerous
situations where a pedestrian is close to or partially hidden by
a vehicle — exactly what a driver might miss.

Features:
    - Image mode: upload a street image, see proximity analysis
    - Video mode: track danger events over time with timeline
    - Danger zones: CRITICAL (< threshold_1), WARNING (< threshold_2), SAFE
    - Proximity lines drawn between close pairs
    - Radar-style danger gauge
    - Pedestrian-level risk cards
    - Heatmap overlay showing danger density

Launch:
    python demo_blind_spot.py --model onnx_model/train_artifacts/model.onnx \\
                              --labels onnx_model/train_artifacts/labels.json

Then open http://127.0.0.1:5075

Requirements:
    pip install flask onnxruntime opencv-python-headless numpy pillow
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import tempfile
import time
import traceback
import uuid

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify, Response
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# ONNX + proximity
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAR_COLOR = (46, 204, 113)
PED_COLOR = (52, 152, 219)
CRIT_COLOR = (231, 76, 60)
WARN_COLOR = (243, 156, 18)
SAFE_COLOR = (46, 204, 113)


def preprocess(frame_rgb: np.ndarray, th: int = 600, tw: int = 800) -> np.ndarray:
    """Preprocess an RGB frame for Mask R-CNN.

    Args:
        frame_rgb: RGB numpy array (H, W, 3).
        th: Target height.
        tw: Target width.

    Returns:
        Float32 tensor (1, 3, th, tw).
    """
    img = cv2.resize(frame_rgb, (tw, th)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)


def detect_all(
    session: ort.InferenceSession,
    inp_name: str,
    output_names: list[str],
    frame_rgb: np.ndarray,
    class_names: list[str],
    ih: int = 600, iw: int = 800,
    thr: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Detect all vehicles and pedestrians in a frame.

    Args:
        session: ONNX session.
        inp_name: Input tensor name.
        output_names: Output tensor names.
        frame_rgb: RGB frame.
        class_names: Ordered class names.
        ih: Model input height.
        iw: Model input width.
        thr: Score threshold.

    Returns:
        Tuple of (vehicles list, pedestrians list). Each item has
        keys: label, score, bbox, mask, center, area.
    """
    oh, ow = frame_rgb.shape[:2]
    tensor = preprocess(frame_rgb, ih, iw)
    res = session.run(output_names, {inp_name: tensor})
    out_map = dict(zip(output_names, res))

    boxes = next((out_map[n] for n in output_names if "box" in n.lower()), res[0])
    labels = next((out_map[n] for n in output_names if "label" in n.lower()), res[1])
    scores = next((out_map[n] for n in output_names if "score" in n.lower()), res[2])
    masks = next((out_map[n] for n in output_names if "mask" in n.lower()), res[3])

    sx, sy = ow / iw, oh / ih
    vehicles, peds = [], []

    for i in range(len(scores)):
        if scores[i] < thr:
            continue
        lid = int(labels[i])
        ln = class_names[lid] if lid < len(class_names) else str(lid)
        sc = float(scores[i])
        x1, y1 = max(0, int(boxes[i][0] * sx)), max(0, int(boxes[i][1] * sy))
        x2, y2 = min(ow, int(boxes[i][2] * sx)), min(oh, int(boxes[i][3] * sy))
        m = cv2.resize((masks[i, 0] > 0.5).astype(np.uint8), (ow, oh))

        obj = {
            "label": ln, "score": round(sc, 4),
            "bbox": [x1, y1, x2, y2], "mask": m,
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "area": int(m.sum()),
        }
        if ln.lower() == "car":
            vehicles.append(obj)
        elif ln.lower() == "pedestrian":
            peds.append(obj)

    return vehicles, peds


def compute_mask_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the minimum pixel distance between two binary masks.

    Uses distance transform for efficiency: computes the distance
    from every pixel in mask_b to the nearest pixel in mask_a.

    Args:
        mask_a: Binary mask (H, W).
        mask_b: Binary mask (H, W).

    Returns:
        Minimum distance in pixels. 0 means masks overlap.
    """
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return 9999.0

    # Check overlap first
    overlap = np.logical_and(mask_a > 0, mask_b > 0)
    if overlap.any():
        return 0.0

    # Distance transform from mask_a boundary
    inv_a = (1 - mask_a).astype(np.uint8)
    dist = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)

    # Min distance at mask_b pixels
    b_coords = np.where(mask_b > 0)
    if len(b_coords[0]) == 0:
        return 9999.0
    return float(dist[b_coords[0], b_coords[1]].min())


def compute_bbox_distance(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Compute edge-to-edge distance between two bounding boxes.

    Args:
        bbox_a: [x1, y1, x2, y2].
        bbox_b: [x1, y1, x2, y2].

    Returns:
        Distance in pixels. 0 if boxes overlap.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return math.sqrt(dx * dx + dy * dy)


def analyze_blind_spots(
    frame_rgb: np.ndarray,
    vehicles: list[dict],
    peds: list[dict],
    crit_dist: float = 50,
    warn_dist: float = 120,
    alpha: float = 0.45,
) -> tuple[np.ndarray, list[dict], dict]:
    """Analyze blind spots and build annotated overlay.

    For each pedestrian, finds the nearest vehicle and classifies
    the pair as CRITICAL, WARNING, or SAFE. Draws proximity lines,
    danger halos, and color-coded overlays.

    Args:
        frame_rgb: Original RGB frame.
        vehicles: Vehicle detections.
        peds: Pedestrian detections.
        crit_dist: Critical distance threshold (pixels).
        warn_dist: Warning distance threshold (pixels).
        alpha: Mask overlay alpha.

    Returns:
        Tuple of (annotated frame, alerts list, summary stats dict).
    """
    oh, ow = frame_rgb.shape[:2]
    ov = frame_rgb.copy()

    # Draw vehicle masks (green)
    for v in vehicles:
        cm = np.zeros_like(ov)
        cm[v["mask"] > 0] = CAR_COLOR
        ov = cv2.addWeighted(ov, 1.0, cm, alpha * 0.4, 0)
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(ov, (x1, y1), (x2, y2), CAR_COLOR, 1)
        cv2.putText(ov, "VEHICLE", (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CAR_COLOR, 1, cv2.LINE_AA)

    alerts: list[dict] = []
    n_crit, n_warn, n_safe = 0, 0, 0

    for pi, ped in enumerate(peds):
        min_dist = 9999.0
        nearest_veh = None
        nearest_vi = -1

        for vi, veh in enumerate(vehicles):
            # Use bbox distance for speed, mask distance for precision on close pairs
            bd = compute_bbox_distance(ped["bbox"], veh["bbox"])
            if bd < warn_dist * 1.5:
                md = compute_mask_distance(ped["mask"], veh["mask"])
                dist = md
            else:
                dist = bd

            if dist < min_dist:
                min_dist = dist
                nearest_veh = veh
                nearest_vi = vi

        # Classify
        if min_dist <= crit_dist:
            level = "CRITICAL"
            color = CRIT_COLOR
            n_crit += 1
        elif min_dist <= warn_dist:
            level = "WARNING"
            color = WARN_COLOR
            n_warn += 1
        else:
            level = "SAFE"
            color = SAFE_COLOR
            n_safe += 1

        # Draw pedestrian mask with danger color
        cm = np.zeros_like(ov)
        cm[ped["mask"] > 0] = color
        ov = cv2.addWeighted(ov, 1.0, cm, alpha * 0.7, 0)

        # Bounding box
        x1, y1, x2, y2 = ped["bbox"]
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, 2)

        # Label with distance
        txt = f"{level} {min_dist:.0f}px"
        (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(ov, (x1, y1 - th2 - 8), (x1 + tw2 + 6, y1), color, -1)
        cv2.putText(ov, txt, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Proximity line to nearest vehicle
        if nearest_veh is not None and min_dist < warn_dist * 1.5:
            pc = ped["center"]
            vc = nearest_veh["center"]
            line_color = color

            # Dashed effect via drawing small segments
            pts = np.linspace(0, 1, 20)
            for j in range(0, len(pts) - 1, 2):
                p1 = (int(pc[0] + (vc[0] - pc[0]) * pts[j]),
                       int(pc[1] + (vc[1] - pc[1]) * pts[j]))
                p2 = (int(pc[0] + (vc[0] - pc[0]) * pts[j + 1]),
                       int(pc[1] + (vc[1] - pc[1]) * pts[j + 1]))
                cv2.line(ov, p1, p2, line_color, 2, cv2.LINE_AA)

            # Distance label at midpoint
            mid = ((pc[0] + vc[0]) // 2, (pc[1] + vc[1]) // 2)
            dist_txt = f"{min_dist:.0f}px"
            cv2.putText(ov, dist_txt, (mid[0] - 15, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ov, dist_txt, (mid[0] - 15, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1, cv2.LINE_AA)

        # Danger halo for critical
        if level == "CRITICAL":
            cx, cy = ped["center"]
            for r in [30, 45, 60]:
                cv2.circle(ov, (cx, cy), r, CRIT_COLOR, 1, cv2.LINE_AA)

        alert = {
            "ped_index": pi,
            "level": level,
            "distance": round(min_dist, 1),
            "ped_center_x": round(ped["center"][0] / ow, 4),
            "ped_center_y": round(ped["center"][1] / oh, 4),
            "nearest_vehicle": nearest_vi,
            "color": f"rgb({color[0]},{color[1]},{color[2]})",
        }
        alerts.append(alert)

    # Overall danger score: 0 (safe) to 100 (critical)
    if not peds:
        danger_score = 0
    else:
        weights = {"CRITICAL": 100, "WARNING": 40, "SAFE": 0}
        danger_score = sum(weights[a["level"]] for a in alerts) / len(alerts)

    summary = {
        "vehicles": len(vehicles),
        "pedestrians": len(peds),
        "critical": n_crit,
        "warning": n_warn,
        "safe": n_safe,
        "danger_score": round(danger_score, 1),
    }

    return ov, alerts, summary


def frame_to_b64(frame_rgb: np.ndarray, quality: int = 75) -> str:
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
jobs: dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind Spot Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#040610;--surface:#0b0e18;--surface2:#121728;--border:#1c2236;
  --text:#e4e7ef;--text2:#5c6380;--car:#2ecc71;--ped:#3498db;
  --crit:#e74c3c;--warn:#f39c12;--safe:#2ecc71;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh}

.hdr{display:flex;align-items:center;gap:.8rem;padding:.7rem 1.5rem;
  background:var(--surface);border-bottom:1px solid var(--border)}
.hdr .icon{width:28px;height:28px;border-radius:6px;background:var(--crit);
  display:grid;place-items:center;font-size:.75rem;font-weight:700;color:#fff}
.hdr h1{font-size:.95rem;font-weight:700}
.hdr .badge{font-family:var(--mono);font-size:.65rem;padding:.18rem .5rem;
  border-radius:20px;background:rgba(231,76,60,.1);color:var(--crit);
  border:1px solid rgba(231,76,60,.2)}
.hdr .spacer{flex:1}
.hdr .status{font-family:var(--mono);font-size:.75rem;color:var(--text2)}

.main{max-width:1400px;margin:0 auto;padding:1.2rem}

/* ═══ TABS ═══ */
.tabs{display:flex;gap:0;margin-bottom:1rem;border-bottom:2px solid var(--border)}
.tab{padding:.6rem 1.2rem;font-size:.8rem;font-weight:600;color:var(--text2);
  cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--crit);border-bottom-color:var(--crit)}
.tab-content{display:none}
.tab-content.active{display:block}

.grid-2{display:grid;grid-template-columns:1fr 340px;gap:1rem}
@media(max-width:1000px){.grid-2{grid-template-columns:1fr}}

.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;overflow:hidden}
.panel+.panel{margin-top:.8rem}
.panel h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:.7rem}

.ctrl-row{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}
.ctrl-row+.ctrl-row{margin-top:.5rem}
.ctrl-row input[type=file]{display:none}
.btn{padding:.45rem .9rem;border-radius:7px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-family:var(--sans);
  font-size:.78rem;font-weight:600;cursor:pointer;transition:all .2s;
  display:inline-flex;align-items:center;gap:.3rem;text-decoration:none}
.btn:hover{border-color:var(--crit);color:var(--crit)}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn.go{background:var(--crit);color:#fff;border-color:var(--crit)}
.btn.go:hover{background:#c0392b}
.sep{width:1px;height:20px;background:var(--border)}
.ctrl-label{font-size:.72rem;color:var(--text2)}
.slider-w{display:flex;align-items:center;gap:.4rem}
.slider-w input[type=range]{width:70px;height:3px;-webkit-appearance:none;background:var(--border);border-radius:2px;outline:none}
.slider-w input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--crit);cursor:pointer}
.slider-w .val{font-family:var(--mono);font-size:.7rem;color:var(--crit);min-width:28px}
.fname{font-family:var(--mono);font-size:.72rem;color:var(--crit)}

.canvas-wrap{position:relative;border-radius:8px;overflow:hidden;background:var(--surface2);min-height:250px}
.canvas-wrap img{width:100%;display:block}
.canvas-wrap .ph{color:var(--text2);font-size:.85rem;padding:3rem;text-align:center}
.spinner-w{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(4,6,16,.85);z-index:10}
.spinner-w.active{display:flex}
.spin-a{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--crit);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ═══ DANGER GAUGE ═══ */
.gauge-wrap{display:flex;flex-direction:column;align-items:center;padding:.8rem 0}
.gauge-ring{position:relative;width:140px;height:140px}
.gauge-ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.gauge-ring .bg{fill:none;stroke:var(--surface2);stroke-width:10}
.gauge-ring .fill{fill:none;stroke-width:10;stroke-linecap:round;transition:stroke-dashoffset .5s,stroke .3s}
.gauge-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge-score{font-family:var(--mono);font-size:1.8rem;font-weight:700}
.gauge-label{font-size:.65rem;color:var(--text2);text-transform:uppercase;letter-spacing:.04em}

/* ═══ STATS ═══ */
.stat-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem}
.sc{background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:.5rem;text-align:center}
.sc .n{font-family:var(--mono);font-size:1.1rem;font-weight:700}
.sc .l{font-size:.58rem;color:var(--text2);text-transform:uppercase;letter-spacing:.04em;margin-top:.05rem}
.sc.crit .n{color:var(--crit)}
.sc.warn .n{color:var(--warn)}
.sc.safe .n{color:var(--safe)}
.sc.car .n{color:var(--car)}
.sc.ped .n{color:var(--ped)}

/* ═══ ALERT LIST ═══ */
.alert-list{max-height:200px;overflow-y:auto}
.alert-item{display:flex;align-items:center;gap:.6rem;padding:.4rem .5rem;
  border-bottom:1px solid var(--border);font-size:.75rem}
.alert-item:last-child{border:none}
.alert-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.alert-dot.crit{background:var(--crit);box-shadow:0 0 8px var(--crit)}
.alert-dot.warn{background:var(--warn);box-shadow:0 0 6px var(--warn)}
.alert-dot.safe{background:var(--safe)}
.alert-level{font-family:var(--mono);font-weight:700;min-width:65px}
.alert-level.crit{color:var(--crit)}
.alert-level.warn{color:var(--warn)}
.alert-level.safe{color:var(--safe)}
.alert-dist{font-family:var(--mono);color:var(--text2)}

/* ═══ CHART ═══ */
.chart-box{position:relative;height:170px}
.chart-box canvas{width:100%!important;height:100%!important}

.pbar-w{margin-top:.5rem;display:none}
.pbar{height:4px;background:var(--surface2);border-radius:2px;overflow:hidden}
.pfill{height:100%;background:linear-gradient(90deg,var(--warn),var(--crit));width:0%;border-radius:2px;transition:width .15s}
.pinfo{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.6rem;color:var(--text2);margin-top:.15rem}

/* ═══ LEGEND ═══ */
.legend{display:flex;gap:1rem;margin-top:.6rem;justify-content:center;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:.3rem;font-size:.68rem;color:var(--text2)}
.legend-dot{width:8px;height:8px;border-radius:50%}

footer{text-align:center;padding:1rem;color:var(--text2);font-size:.65rem;margin-top:1rem}
</style>
</head>
<body>

<div class="hdr">
  <div class="icon">!</div>
  <h1>Blind Spot Detection</h1>
  <div class="badge">Proximity analysis &middot; ONNX &middot; Mask R-CNN</div>
  <div class="spacer"></div>
  <span class="status" id="status">Ready</span>
</div>

<div class="main">
  <div class="tabs">
    <div class="tab active" data-tab="image">Image analysis</div>
    <div class="tab" data-tab="video">Video tracking</div>
  </div>

  <!-- ═══ IMAGE TAB ═══ -->
  <div class="tab-content active" id="tab-image">
    <div class="grid-2">
      <div>
        <div class="panel">
          <h2>Scene analysis</h2>
          <div class="canvas-wrap" id="img-wrap">
            <div class="ph" id="img-ph">Upload a street image to detect blind spots</div>
            <img id="img-result" style="display:none" alt="result">
            <div class="spinner-w" id="img-spinner"><div class="spin-a"></div></div>
          </div>
          <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:var(--crit)"></div> Critical (&lt;50px)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--warn)"></div> Warning (&lt;120px)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--safe)"></div> Safe</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--car)"></div> Vehicle</div>
          </div>
        </div>
      </div>
      <div>
        <div class="panel">
          <h2>Controls</h2>
          <div class="ctrl-row">
            <input type="file" id="img-input" accept="image/*">
            <button class="btn" onclick="document.getElementById('img-input').click()">Choose image</button>
            <span class="fname" id="img-fname"></span>
          </div>
          <div class="ctrl-row">
            <div class="slider-w"><span class="ctrl-label">Confidence</span>
              <input type="range" id="img-conf" min="10" max="95" value="50">
              <span class="val" id="img-conf-val">0.50</span></div>
          </div>
          <div class="ctrl-row">
            <div class="slider-w"><span class="ctrl-label">Critical</span>
              <input type="range" id="crit-dist" min="10" max="100" value="50">
              <span class="val" id="crit-dist-val">50px</span></div>
            <div class="slider-w"><span class="ctrl-label">Warning</span>
              <input type="range" id="warn-dist" min="50" max="250" value="120">
              <span class="val" id="warn-dist-val">120px</span></div>
          </div>
          <button class="btn go" id="btn-img" disabled style="margin-top:.5rem;width:100%">Detect blind spots</button>
        </div>

        <div class="panel">
          <h2>Danger level</h2>
          <div class="gauge-wrap">
            <div class="gauge-ring">
              <svg viewBox="0 0 120 120">
                <circle class="bg" cx="60" cy="60" r="50"/>
                <circle class="fill" id="ig-fill" cx="60" cy="60" r="50"
                  stroke-dasharray="314.16" stroke-dashoffset="314.16"/>
              </svg>
              <div class="gauge-center">
                <div class="gauge-score" id="ig-score">—</div>
                <div class="gauge-label">danger score</div>
              </div>
            </div>
          </div>
          <div class="stat-grid">
            <div class="sc crit"><div class="n" id="si-crit">0</div><div class="l">Critical</div></div>
            <div class="sc warn"><div class="n" id="si-warn">0</div><div class="l">Warning</div></div>
            <div class="sc safe"><div class="n" id="si-safe">0</div><div class="l">Safe</div></div>
            <div class="sc car"><div class="n" id="si-cars">0</div><div class="l">Vehicles</div></div>
            <div class="sc ped"><div class="n" id="si-peds">0</div><div class="l">Pedestrians</div></div>
            <div class="sc"><div class="n" id="si-inf">—</div><div class="l">Inference</div></div>
          </div>
        </div>

        <div class="panel">
          <h2>Alerts</h2>
          <div class="alert-list" id="img-alerts">
            <div style="color:var(--text2);font-size:.75rem;padding:.5rem;text-align:center">Analyze an image to see alerts</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ VIDEO TAB ═══ -->
  <div class="tab-content" id="tab-video">
    <div class="grid-2">
      <div>
        <div class="panel">
          <h2>Video feed</h2>
          <div class="canvas-wrap" id="vid-wrap">
            <div class="ph" id="vid-ph">Upload a street video</div>
            <img id="vid-frame" style="display:none" alt="frame">
            <div class="spinner-w" id="vid-spinner"><div class="spin-a"></div></div>
          </div>
          <div class="pbar-w" id="vid-prog">
            <div class="pbar"><div class="pfill" id="vid-pfill"></div></div>
            <div class="pinfo"><span id="vid-pf">0/0</span><span id="vid-eta">ETA: —</span></div>
          </div>
        </div>
        <div class="panel">
          <h2>Danger score over time</h2>
          <div class="chart-box"><canvas id="chart-danger"></canvas></div>
        </div>
      </div>
      <div>
        <div class="panel">
          <h2>Controls</h2>
          <div class="ctrl-row">
            <input type="file" id="vid-input" accept="video/*">
            <button class="btn" onclick="document.getElementById('vid-input').click()">Choose video</button>
            <span class="fname" id="vid-fname"></span>
          </div>
          <div class="ctrl-row">
            <div class="slider-w"><span class="ctrl-label">Confidence</span>
              <input type="range" id="vid-conf" min="10" max="95" value="50">
              <span class="val" id="vid-conf-val">0.50</span></div>
            <div class="sep"></div>
            <div class="slider-w"><span class="ctrl-label">Every</span>
              <input type="range" id="vid-skip" min="1" max="15" value="5">
              <span class="val" id="vid-skip-val">5th</span></div>
          </div>
          <button class="btn go" id="btn-vid" disabled style="margin-top:.5rem;width:100%">Process video</button>
        </div>

        <div class="panel">
          <h2>Live danger</h2>
          <div class="gauge-wrap">
            <div class="gauge-ring">
              <svg viewBox="0 0 120 120">
                <circle class="bg" cx="60" cy="60" r="50"/>
                <circle class="fill" id="vg-fill" cx="60" cy="60" r="50"
                  stroke-dasharray="314.16" stroke-dashoffset="314.16"/>
              </svg>
              <div class="gauge-center">
                <div class="gauge-score" id="vg-score">—</div>
                <div class="gauge-label">danger score</div>
              </div>
            </div>
          </div>
          <div class="stat-grid">
            <div class="sc crit"><div class="n" id="sv-crit">0</div><div class="l">Critical</div></div>
            <div class="sc warn"><div class="n" id="sv-warn">0</div><div class="l">Warning</div></div>
            <div class="sc safe"><div class="n" id="sv-safe">0</div><div class="l">Safe</div></div>
            <div class="sc crit"><div class="n" id="sv-peak">0</div><div class="l">Peak danger</div></div>
            <div class="sc warn"><div class="n" id="sv-events">0</div><div class="l">Crit events</div></div>
            <div class="sc"><div class="n" id="sv-avg">—</div><div class="l">Avg danger</div></div>
          </div>
        </div>

        <div class="panel">
          <h2>Event log</h2>
          <div class="alert-list" id="vid-log">
            <div style="color:var(--text2);font-size:.75rem;padding:.5rem;text-align:center">Process a video to see events</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>Blind Spot Detection — Azure AutoML for Images &middot; ONNX Runtime &middot; Mask R-CNN</footer>

<script>
const $=id=>document.getElementById(id);
function formatTime(s){const m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+String(ss).padStart(2,'0')}

/* ═══ TABS ═══ */
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
  });
});

function updateGauge(fillId,scoreId,score){
  const circ=314.16,offset=circ-(score/100)*circ;
  const el=$(fillId);
  el.style.strokeDashoffset=offset;
  el.style.stroke=score>70?'var(--crit)':score>30?'var(--warn)':'var(--safe)';
  $(scoreId).textContent=Math.round(score);
  $(scoreId).style.color=score>70?'var(--crit)':score>30?'var(--warn)':'var(--safe)';
}

function renderAlerts(containerId,alerts){
  const el=$(containerId);
  if(!alerts.length){el.innerHTML='<div style="color:var(--text2);font-size:.75rem;padding:.5rem;text-align:center">No pedestrians detected</div>';return}
  el.innerHTML='';
  alerts.forEach((a,i)=>{
    const cls=a.level==='CRITICAL'?'crit':a.level==='WARNING'?'warn':'safe';
    el.innerHTML+=`<div class="alert-item">
      <div class="alert-dot ${cls}"></div>
      <span class="alert-level ${cls}">${a.level}</span>
      <span>Ped #${i+1}</span>
      <span class="alert-dist">${a.distance.toFixed(0)}px from vehicle</span>
    </div>`;
  });
}

/* ═══ IMAGE TAB ═══ */
const imgInput=$('img-input'),imgConf=$('img-conf'),critDist=$('crit-dist'),warnDist=$('warn-dist');
let imgFile=null;

$('img-conf').addEventListener('input',()=>$('img-conf-val').textContent=(imgConf.value/100).toFixed(2));
$('crit-dist').addEventListener('input',()=>$('crit-dist-val').textContent=critDist.value+'px');
$('warn-dist').addEventListener('input',()=>$('warn-dist-val').textContent=warnDist.value+'px');

imgInput.addEventListener('change',()=>{
  if(!imgInput.files.length)return;
  imgFile=imgInput.files[0];
  $('img-fname').textContent=imgFile.name;
  $('btn-img').disabled=false;
  const r=new FileReader();
  r.onload=e=>{$('img-result').src=e.target.result;$('img-result').style.display='block';$('img-ph').style.display='none'};
  r.readAsDataURL(imgFile);
});

$('btn-img').addEventListener('click',async()=>{
  if(!imgFile)return;
  $('btn-img').disabled=true;
  $('img-spinner').classList.add('active');

  const fd=new FormData();
  fd.append('image',imgFile);
  fd.append('threshold',(imgConf.value/100).toFixed(2));
  fd.append('crit_dist',critDist.value);
  fd.append('warn_dist',warnDist.value);

  try{
    const resp=await fetch('/analyze',{method:'POST',body:fd});
    if(!resp.ok){const d=await resp.json().catch(()=>({}));alert(d.error||'Failed');return}
    const data=await resp.json();

    $('img-result').src=data.annotated;$('img-result').style.display='block';

    $('si-crit').textContent=data.summary.critical;
    $('si-warn').textContent=data.summary.warning;
    $('si-safe').textContent=data.summary.safe;
    $('si-cars').textContent=data.summary.vehicles;
    $('si-peds').textContent=data.summary.pedestrians;
    $('si-inf').textContent=data.inference_time.toFixed(3)+'s';
    updateGauge('ig-fill','ig-score',data.summary.danger_score);
    renderAlerts('img-alerts',data.alerts);
  }catch(e){alert(e.message)}
  finally{$('img-spinner').classList.remove('active');$('btn-img').disabled=false}
});

/* ═══ VIDEO TAB ═══ */
let vidFile=null;
$('vid-conf').addEventListener('input',()=>$('vid-conf-val').textContent=($('vid-conf').value/100).toFixed(2));
$('vid-skip').addEventListener('input',()=>{const v=$('vid-skip').value;$('vid-skip-val').textContent=v+(v==='1'?'st':v==='2'?'nd':v==='3'?'rd':'th')});

$('vid-input').addEventListener('change',()=>{
  if(!$('vid-input').files.length)return;
  vidFile=$('vid-input').files[0];
  $('vid-fname').textContent=vidFile.name;
  $('btn-vid').disabled=false;
});

const chartOpts={responsive:true,maintainAspectRatio:false,animation:{duration:0},
  scales:{x:{display:true,grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#5c6380',font:{size:8,family:'JetBrains Mono'}}},
          y:{display:true,beginAtZero:true,max:100,grid:{color:'rgba(255,255,255,.04)'},
            ticks:{color:'#5c6380',font:{size:8,family:'JetBrains Mono'}}}},
  plugins:{legend:{labels:{color:'#8890a8',font:{size:9,family:'DM Sans'},boxWidth:8,padding:6}},
    annotation:false}};

const ctxD=$('chart-danger').getContext('2d');
const chartD=new Chart(ctxD,{type:'line',
  data:{labels:[],datasets:[
    {label:'Danger score',data:[],borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,.1)',
     fill:true,tension:.3,pointRadius:0,borderWidth:1.5},
    {label:'Critical count',data:[],borderColor:'#f39c12',fill:false,tension:.3,pointRadius:0,
     borderWidth:1,borderDash:[4,3]},
  ]},options:chartOpts});

$('btn-vid').addEventListener('click',async()=>{
  if(!vidFile)return;
  $('btn-vid').disabled=true;
  chartD.data.labels=[];chartD.data.datasets.forEach(ds=>ds.data=[]);chartD.update();
  $('vid-ph').style.display='none';$('vid-frame').style.display='none';
  $('vid-prog').style.display='block';$('vid-pfill').style.width='0%';
  $('vid-log').innerHTML='';
  $('status').textContent='Processing...';

  let peakDanger=0,dangerSum=0,processed=0,totalCritEvents=0;

  const fd=new FormData();
  fd.append('video',vidFile);
  fd.append('threshold',($('vid-conf').value/100).toFixed(2));
  fd.append('skip',$('vid-skip').value);

  let jobId;
  try{
    const resp=await fetch('/upload_video',{method:'POST',body:fd});
    if(!resp.ok){alert('Upload failed');$('btn-vid').disabled=false;return}
    jobId=(await resp.json()).job_id;
  }catch(e){alert(e.message);$('btn-vid').disabled=false;return}

  const t0=performance.now();
  const evtSrc=new EventSource('/stream_video/'+jobId);

  evtSrc.addEventListener('frame',e=>{
    const d=JSON.parse(e.data);
    processed++;

    $('vid-frame').src=d.preview;$('vid-frame').style.display='block';
    const pct=d.frame_idx/d.total_frames*100;
    $('vid-pfill').style.width=pct+'%';
    $('vid-pf').textContent=d.frame_idx+'/'+d.total_frames;
    const elapsed=(performance.now()-t0)/1000;
    const fps=processed/elapsed;
    $('vid-eta').textContent='ETA: '+formatTime((d.total_frames-d.frame_idx)/Math.max(fps,.01));

    const ds=d.danger_score;
    dangerSum+=ds;
    if(ds>peakDanger) peakDanger=ds;
    totalCritEvents+=d.critical;

    updateGauge('vg-fill','vg-score',ds);
    $('sv-crit').textContent=d.critical;
    $('sv-warn').textContent=d.warning;
    $('sv-safe').textContent=d.safe;
    $('sv-peak').textContent=Math.round(peakDanger);
    $('sv-events').textContent=totalCritEvents;
    $('sv-avg').textContent=(dangerSum/processed).toFixed(0);

    chartD.data.labels.push(formatTime(d.time));
    chartD.data.datasets[0].data.push(ds);
    chartD.data.datasets[1].data.push(d.critical);
    if(chartD.data.labels.length>300){chartD.data.labels.shift();chartD.data.datasets.forEach(x=>x.data.shift())}
    chartD.update();

    // Log critical events
    if(d.critical>0){
      const logEl=$('vid-log');
      if(logEl.querySelector('.ph'))logEl.innerHTML='';
      const item=document.createElement('div');
      item.className='alert-item';
      item.innerHTML=`<div class="alert-dot crit"></div>
        <span class="alert-level crit">CRITICAL</span>
        <span>${formatTime(d.time)}</span>
        <span class="alert-dist">${d.critical} ped(s) in danger</span>`;
      logEl.prepend(item);
      if(logEl.children.length>30)logEl.removeChild(logEl.lastChild);
    }
  });

  evtSrc.addEventListener('done',()=>{
    evtSrc.close();
    $('status').textContent='Done';$('vid-pfill').style.width='100%';$('vid-eta').textContent='Complete';
    $('btn-vid').disabled=false;
  });
  evtSrc.addEventListener('error_msg',e=>{evtSrc.close();alert(JSON.parse(e.data).message);$('btn-vid').disabled=false});
  evtSrc.onerror=()=>{evtSrc.close();$('btn-vid').disabled=false};
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str) -> Flask:
    """Create the blind spot detection demo Flask app.

    Args:
        model_path: Path to model.onnx.
        labels_path: Path to labels.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    with open(labels_path) as f:
        class_names: list[str] = json.load(f)

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    inp_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Model: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

    output_dir = tempfile.mkdtemp(prefix="blindspot_")

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        """Analyze a single image for blind spots."""
        try:
            if "image" not in request.files:
                return jsonify({"error": "No image provided"}), 400

            file = request.files["image"]
            threshold = float(request.form.get("threshold", "0.5"))
            crit = float(request.form.get("crit_dist", "50"))
            warn = float(request.form.get("warn_dist", "120"))

            pil_img = Image.open(file.stream).convert("RGB")
            frame_rgb = np.array(pil_img)

            t0 = time.time()
            vehicles, peds = detect_all(
                session, inp_name, output_names,
                frame_rgb, class_names, thr=threshold,
            )
            elapsed_detect = time.time() - t0

            annotated, alerts, summary = analyze_blind_spots(
                frame_rgb, vehicles, peds,
                crit_dist=crit, warn_dist=warn,
            )
            total_elapsed = time.time() - t0

            # Remove mask from alerts (not serializable)
            clean_alerts = [{k: v for k, v in a.items() if k != "mask"} for a in alerts]

            return jsonify({
                "annotated": frame_to_b64(annotated),
                "alerts": clean_alerts,
                "summary": summary,
                "inference_time": round(total_elapsed, 4),
            })

        except Exception as exc:
            print(f"[ERROR] /analyze:\n{traceback.format_exc()}")
            return jsonify({"error": str(exc)}), 500

    @app.route("/upload_video", methods=["POST"])
    def upload_video():
        """Accept a video upload."""
        try:
            if "video" not in request.files:
                return jsonify({"error": "No video provided"}), 400

            video_file = request.files["video"]
            threshold = float(request.form.get("threshold", "0.5"))
            skip = int(request.form.get("skip", "5"))

            job_id = str(uuid.uuid4())[:8]
            input_path = os.path.join(output_dir, f"{job_id}_input.mp4")
            video_file.save(input_path)

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return jsonify({"error": "Could not open video"}), 400

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()

            jobs[job_id] = {
                "input_path": input_path,
                "threshold": threshold,
                "skip": skip,
                "total_frames": total_frames,
                "fps": fps,
            }
            return jsonify({"job_id": job_id, "total_frames": total_frames})

        except Exception as exc:
            print(f"[ERROR] /upload_video:\n{traceback.format_exc()}")
            return jsonify({"error": str(exc)}), 500

    @app.route("/stream_video/<job_id>")
    def stream_video(job_id: str):
        """Stream blind spot analysis frame by frame via SSE."""
        if job_id not in jobs:
            def err():
                yield 'event: error_msg\ndata: {"message":"Job not found"}\n\n'
            return Response(err(), content_type="text/event-stream")

        job = jobs[job_id]

        def generate():
            try:
                cap = cv2.VideoCapture(job["input_path"])
                if not cap.isOpened():
                    yield 'event: error_msg\ndata: {"message":"Cannot open video"}\n\n'
                    return

                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    if frame_idx % job["skip"] == 0 or frame_idx == 1:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        t0 = time.time()
                        vehicles, peds = detect_all(
                            session, inp_name, output_names,
                            frame_rgb, class_names, thr=job["threshold"],
                        )

                        annotated, alerts, summary = analyze_blind_spots(
                            frame_rgb, vehicles, peds,
                        )
                        elapsed = time.time() - t0

                        frame_time = frame_idx / job["fps"]

                        payload = json.dumps({
                            "frame_idx": frame_idx,
                            "total_frames": job["total_frames"],
                            "preview": frame_to_b64(annotated, quality=55),
                            "time": round(frame_time, 2),
                            "danger_score": summary["danger_score"],
                            "critical": summary["critical"],
                            "warning": summary["warning"],
                            "safe": summary["safe"],
                            "vehicles": summary["vehicles"],
                            "pedestrians": summary["pedestrians"],
                            "inference_time": round(elapsed, 4),
                        })
                        yield f"event: frame\ndata: {payload}\n\n"

                cap.release()
                yield f'event: done\ndata: {json.dumps({"total_frames": frame_idx})}\n\n'

            except Exception as exc:
                print(f"[ERROR] stream {job_id}:\n{traceback.format_exc()}")
                yield f'event: error_msg\ndata: {json.dumps({"message": str(exc)})}\n\n'

        return Response(generate(), content_type="text/event-stream")

    return app


def main() -> None:
    """Parse arguments and start the blind spot detection demo server."""
    parser = argparse.ArgumentParser(description="Blind spot detection demo")
    parser.add_argument("--model", default=os.path.join("onnx_model", "train_artifacts", "model.onnx"))
    parser.add_argument("--labels", default=os.path.join("onnx_model", "train_artifacts", "labels.json"))
    parser.add_argument("--port", type=int, default=5075)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found at '{args.model}'")
        return
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels not found at '{args.labels}'")
        return

    app = create_app(args.model, args.labels)
    print(f"\nBlind Spot Detection at http://{args.host}:{args.port}")
    print("Upload a street image or video.\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
