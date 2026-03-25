"""
Local web UI for instance segmentation using an ONNX model
from Azure AutoML for Images (Mask R-CNN).

Features:
    - Dark / Light / Midnight / Forest color themes
    - User-selectable colors for each class (car, pedestrian)
    - Color presets (Default, Warm, Neon, Pastel, Electric, Monochrome)
    - Tabbed results: Overview | Bounding Boxes | Polygons
    - Individual crop download (PNG) per detection
    - Bulk ZIP download for all bounding-box or polygon crops
    - Drag-and-drop image upload with adjustable confidence

Launch with:
    python app_instance_segmentation.py

Then open http://127.0.0.1:5000 in your browser.

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
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_COLOR: tuple[int, int, int] = (231, 76, 60)


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_labels(labels_path: str) -> list[str]:
    """Load ordered class labels from a JSON file.

    Args:
        labels_path: Path to labels.json.

    Returns:
        Ordered list of class name strings.
    """
    with open(labels_path, "r") as f:
        labels: list[str] = json.load(f)
    return labels


def load_session(model_path: str) -> ort.InferenceSession:
    """Load an ONNX model into an inference session.

    Args:
        model_path: Path to model.onnx.

    Returns:
        An ONNX Runtime InferenceSession.
    """
    providers: list[str] = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(model_path, providers=providers)


# ──────────────────────────────────────────────────────────────────────
# Inference pipeline
# ──────────────────────────────────────────────────────────────────────

def preprocess(
    pil_image: Image.Image,
    target_height: int = 600,
    target_width: int = 800,
) -> tuple[np.ndarray, int, int]:
    """Preprocess a PIL image for Mask R-CNN inference.

    Args:
        pil_image: Input PIL Image (any mode).
        target_height: Resize height for model input.
        target_width: Resize width for model input.

    Returns:
        Tuple of (img_tensor, orig_height, orig_width).
    """
    img_rgb = pil_image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    img_resized = img_rgb.resize((target_width, target_height), Image.BILINEAR)
    np_img = np.array(img_resized, dtype=np.float32) / 255.0
    np_img = (np_img - IMAGENET_MEAN) / IMAGENET_STD
    np_img = np_img.transpose(2, 0, 1)
    tensor = np.expand_dims(np_img, axis=0).astype(np.float32)
    return tensor, orig_h, orig_w


def run_inference(
    session: ort.InferenceSession,
    img_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the ONNX model on a preprocessed tensor.

    Args:
        session: ONNX Runtime session.
        img_tensor: Preprocessed float32 array (1, 3, H, W).

    Returns:
        Tuple of (boxes, labels, scores, masks).
    """
    input_name: str = session.get_inputs()[0].name
    results = session.run(None, {input_name: img_tensor})
    return results[0], results[1], results[2], results[3]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an (R, G, B) tuple.

    Args:
        hex_color: Color string like '#2ecc71' or '2ecc71'.

    Returns:
        An (R, G, B) tuple with values 0-255.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return DEFAULT_COLOR
    try:
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    except ValueError:
        return DEFAULT_COLOR


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 data URI.

    Args:
        img: PIL Image.
        fmt: Image format (JPEG, PNG).

    Returns:
        Base64-encoded data URI string.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def pil_to_raw_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to raw base64 (no data-URI prefix).

    Args:
        img: PIL Image.
        fmt: Image format.

    Returns:
        Raw base64-encoded string.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────
# Annotation + crop extraction
# ──────────────────────────────────────────────────────────────────────

def annotate_and_extract(
    pil_image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    class_names: list[str],
    input_height: int,
    input_width: int,
    color_map: dict[str, tuple[int, int, int]] | None = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    alpha: float = 0.45,
) -> dict:
    """Annotate image and extract individual bbox / polygon crops.

    Produces:
        - Full annotated overlay image.
        - Per-detection bounding-box crop (rectangular region from original).
        - Per-detection polygon crop (masked region, transparent background).

    Args:
        pil_image: Original image.
        boxes: Detection boxes (n, 4).
        labels: Detection class IDs (n,).
        scores: Detection scores (n,).
        masks: Detection masks (n, 1, H_in, W_in).
        class_names: Ordered class names.
        input_height: Model input height.
        input_width: Model input width.
        color_map: Dict mapping lowercase class names to (R,G,B).
        score_threshold: Min confidence.
        mask_threshold: Binarization threshold for masks.
        alpha: Mask overlay transparency on the full image.

    Returns:
        A dict with keys: annotated_image, detections, bbox_crops, polygon_crops.
        Each crop list contains PIL Images aligned with the detections list.
    """
    if color_map is None:
        color_map = {}

    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]
    overlay = img_rgb.copy()

    scale_x: float = orig_w / input_width
    scale_y: float = orig_h / input_height

    detections: list[dict] = []
    bbox_crops: list[Image.Image] = []
    polygon_crops: list[Image.Image] = []

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        label_id = int(labels[i])
        label_name = class_names[label_id] if label_id < len(class_names) else str(label_id)
        score = float(scores[i])
        color = color_map.get(label_name.lower(), DEFAULT_COLOR)

        x1 = max(0, int(boxes[i][0] * scale_x))
        y1 = max(0, int(boxes[i][1] * scale_y))
        x2 = min(orig_w, int(boxes[i][2] * scale_x))
        y2 = min(orig_h, int(boxes[i][3] * scale_y))

        # ── Binarize & resize mask ────────────────────────────────────
        raw_mask = masks[i, 0]
        binary_mask = (raw_mask > mask_threshold).astype(np.uint8)
        mask_full = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # ── Full-image overlay ────────────────────────────────────────
        colored = np.zeros_like(overlay)
        colored[mask_full > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name.upper()} {score:.0%}"
        (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th_t - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(overlay, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Bounding-box crop (simple rectangle from original) ────────
        bbox_crop = img_rgb[y1:y2, x1:x2]
        if bbox_crop.size > 0:
            bbox_crops.append(Image.fromarray(bbox_crop))
        else:
            bbox_crops.append(Image.new("RGB", (1, 1)))

        # ── Polygon crop (masked region, transparent bg) ──────────────
        # Create RGBA with the mask as alpha channel
        rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_rgb
        rgba[:, :, 3] = mask_full * 255
        # Crop to bounding box of the mask to get a tight region
        mask_coords = np.where(mask_full > 0)
        if len(mask_coords[0]) > 0:
            my1, my2 = mask_coords[0].min(), mask_coords[0].max() + 1
            mx1, mx2 = mask_coords[1].min(), mask_coords[1].max() + 1
            poly_crop = rgba[my1:my2, mx1:mx2]
            polygon_crops.append(Image.fromarray(poly_crop, "RGBA"))
        else:
            polygon_crops.append(Image.new("RGBA", (1, 1)))

        detections.append({
            "label": label_name,
            "score": round(score, 4),
            "bbox": [x1, y1, x2, y2],
        })

    # Sort everything together by score descending
    if detections:
        indices = sorted(range(len(detections)), key=lambda k: detections[k]["score"], reverse=True)
        detections = [detections[j] for j in indices]
        bbox_crops = [bbox_crops[j] for j in indices]
        polygon_crops = [polygon_crops[j] for j in indices]

    return {
        "annotated_image": Image.fromarray(overlay),
        "detections": detections,
        "bbox_crops": bbox_crops,
        "polygon_crops": polygon_crops,
    }


# ──────────────────────────────────────────────────────────────────────
# HTML template
# ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Instance Segmentation — ONNX Local Inference</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  /* ═══ THEMES ═══ */
  :root, [data-theme="dark"] {
    --bg:#0c0e12; --surface:#14171e; --surface2:#1c2029; --border:#282c38;
    --text:#e4e6eb; --text2:#8b8f9a; --danger:#e74c3c; --radius:12px;
    --color-car:#2ecc71; --color-ped:#3498db; --btn-text:#000;
  }
  [data-theme="light"] {
    --bg:#f0f2f5; --surface:#fff; --surface2:#e8eaed; --border:#d1d5db;
    --text:#1a1a2e; --text2:#6b7280; --danger:#dc2626; --btn-text:#fff;
  }
  [data-theme="midnight"] {
    --bg:#0a0a1a; --surface:#111128; --surface2:#1a1a3e; --border:#2d2d5e;
    --text:#d4d4f0; --text2:#7a7aad; --danger:#ff4757; --btn-text:#000;
  }
  [data-theme="forest"] {
    --bg:#0a120e; --surface:#111f18; --surface2:#1a2e22; --border:#2a4a35;
    --text:#d0e8d8; --text2:#7aad8a; --danger:#e74c3c; --btn-text:#000;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);
    min-height:100vh;overflow-x:hidden;transition:background .35s,color .35s}

  /* ═══ HEADER ═══ */
  header{background:var(--surface);border-bottom:1px solid var(--border);
    padding:1.2rem 2rem;display:flex;align-items:center;gap:1rem;transition:all .35s}
  header .logo{width:38px;height:38px;border-radius:10px;
    background:linear-gradient(135deg,var(--color-car),var(--color-ped));
    display:grid;place-items:center;font-size:1.2rem;font-weight:700;color:#000;transition:background .35s}
  header h1{font-size:1.15rem;font-weight:700;letter-spacing:-.02em}
  header .badge{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:.25rem .6rem;
    border-radius:20px;background:var(--surface2);border:1px solid var(--border);color:var(--text2);transition:all .35s}
  header .spacer{flex:1}
  .theme-row{display:flex;align-items:center;gap:.5rem}
  .theme-btn{width:28px;height:28px;border-radius:50%;border:2px solid var(--border);
    cursor:pointer;transition:transform .15s,border-color .2s,box-shadow .2s}
  .theme-btn:hover{transform:scale(1.15)}
  .theme-btn.active{border-color:var(--text);box-shadow:0 0 0 2px var(--text)}
  .theme-btn[data-t="dark"]{background:#0c0e12}
  .theme-btn[data-t="light"]{background:#f0f2f5}
  .theme-btn[data-t="midnight"]{background:linear-gradient(135deg,#0a0a1a,#2d2d5e)}
  .theme-btn[data-t="forest"]{background:linear-gradient(135deg,#0a120e,#2a4a35)}

  /* ═══ LAYOUT ═══ */
  .container{max-width:1480px;margin:0 auto;padding:2rem;
    display:grid;grid-template-columns:400px 1fr;gap:2rem}
  @media(max-width:1000px){.container{grid-template-columns:1fr}}

  /* ═══ PANEL ═══ */
  .panel{background:var(--surface);border:1px solid var(--border);
    border-radius:var(--radius);padding:1.5rem;transition:background .35s,border-color .35s}
  .panel+.panel{margin-top:1.5rem}
  .panel h2{font-size:.82rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:1.1rem}

  /* ═══ UPLOAD ═══ */
  .upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:2rem 1rem;
    text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--color-car);background:rgba(46,204,113,.04)}
  .upload-zone input{display:none}
  .upload-zone .icon{font-size:2.5rem;margin-bottom:.5rem}
  .upload-zone p{color:var(--text2);font-size:.9rem}
  #preview-container{margin-top:1rem;border-radius:var(--radius);overflow:hidden;display:none}
  #preview-container img{width:100%;display:block;border-radius:var(--radius)}

  /* ═══ SLIDER ═══ */
  .slider-group{margin-top:1.2rem}
  .slider-group label{display:flex;justify-content:space-between;font-size:.85rem;color:var(--text2);margin-bottom:.5rem}
  .slider-group label span{color:var(--color-car);font-weight:600}
  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;
    border-radius:3px;background:var(--surface2);outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;
    border-radius:50%;background:var(--color-car);cursor:pointer;border:2px solid var(--bg)}

  /* ═══ COLOR PICKERS ═══ */
  .color-group{margin-top:1.3rem}
  .color-group h3{font-size:.82rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:.8rem}
  .color-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.65rem}
  .color-row .color-label{font-size:.85rem;font-weight:500;min-width:95px}
  .color-swatch-wrapper{position:relative;width:36px;height:36px;flex-shrink:0}
  .color-swatch-wrapper input[type=color]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
  .color-swatch{width:36px;height:36px;border-radius:8px;border:2px solid var(--border);pointer-events:none;transition:border-color .2s,box-shadow .2s}
  .color-swatch-wrapper:hover .color-swatch{border-color:var(--text);box-shadow:0 0 10px rgba(255,255,255,.08)}
  .color-hex{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--text2);
    background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:.3rem .5rem;
    width:80px;text-align:center;outline:none;transition:border-color .2s,color .2s}
  .color-hex:focus{border-color:var(--text);color:var(--text)}
  .preset-row{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:.7rem}
  .preset-chip{font-size:.72rem;padding:.3rem .6rem;border-radius:20px;border:1px solid var(--border);
    background:var(--surface2);color:var(--text2);cursor:pointer;transition:all .2s;
    font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:3px}
  .preset-chip:hover{border-color:var(--text);color:var(--text);transform:translateY(-1px)}
  .preset-chip .dot{display:inline-block;width:8px;height:8px;border-radius:50%;flex-shrink:0}

  /* ═══ BUTTON ═══ */
  .btn-primary{margin-top:1.4rem;width:100%;padding:.85rem;
    background:linear-gradient(135deg,var(--color-car),color-mix(in srgb,var(--color-car) 65%,#000));
    color:var(--btn-text);font-weight:700;font-size:.95rem;border:none;border-radius:var(--radius);
    cursor:pointer;transition:transform .15s,box-shadow .2s,background .35s}
  .btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 24px color-mix(in srgb,var(--color-car) 30%,transparent)}
  .btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}

  /* ═══ RIGHT PANEL ═══ */
  .result-panel{display:flex;flex-direction:column;gap:1.5rem}

  /* ── Tabs ── */
  .tab-bar{display:flex;gap:0;border-bottom:2px solid var(--border);margin-bottom:0}
  .tab-btn{padding:.7rem 1.2rem;font-size:.82rem;font-weight:600;text-transform:uppercase;
    letter-spacing:.05em;color:var(--text2);background:transparent;border:none;cursor:pointer;
    border-bottom:2px solid transparent;margin-bottom:-2px;transition:color .2s,border-color .2s}
  .tab-btn:hover{color:var(--text)}
  .tab-btn.active{color:var(--color-car);border-bottom-color:var(--color-car)}
  .tab-content{display:none;padding-top:1.2rem}
  .tab-content.active{display:block}

  /* ── Result image ── */
  #result-image-container{border-radius:var(--radius);overflow:hidden;background:var(--surface2);
    min-height:300px;display:grid;place-items:center;position:relative;transition:background .35s}
  #result-image-container img{width:100%;display:block}
  #result-image-container .placeholder{color:var(--text2);font-size:.9rem;padding:3rem}

  /* ── Stats bar ── */
  .stats-bar{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem}
  .stat-card{flex:1;min-width:110px;background:var(--surface2);border:1px solid var(--border);
    border-radius:10px;padding:.9rem;text-align:center;transition:all .35s}
  .stat-card .value{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace}
  .stat-card .label{font-size:.72rem;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-top:.15rem}
  .stat-card.cars .value{color:var(--color-car)}
  .stat-card.peds .value{color:var(--color-ped)}
  .stat-card.time .value{color:#f39c12}

  /* ── Detections table ── */
  .det-table{width:100%;border-collapse:collapse;font-size:.85rem}
  .det-table th{text-align:left;padding:.6rem .8rem;color:var(--text2);
    border-bottom:1px solid var(--border);font-weight:500;text-transform:uppercase;letter-spacing:.05em;font-size:.75rem}
  .det-table td{padding:.55rem .8rem;border-bottom:1px solid var(--border)}
  .det-table tr:last-child td{border-bottom:none}
  .conf-bar{height:6px;border-radius:3px;background:var(--surface2);width:100px;display:inline-block;position:relative;vertical-align:middle}
  .conf-bar-fill{height:100%;border-radius:3px;position:absolute;top:0;left:0}

  /* ── Crop grid (bboxes & polygons) ── */
  .crop-toolbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem}
  .crop-toolbar .count{font-size:.85rem;color:var(--text2)}
  .btn-download-all{padding:.5rem 1rem;font-size:.78rem;font-weight:600;
    border-radius:8px;border:1px solid var(--border);background:var(--surface2);
    color:var(--text);cursor:pointer;transition:all .2s;font-family:'JetBrains Mono',monospace}
  .btn-download-all:hover{border-color:var(--color-car);color:var(--color-car)}
  .crop-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}
  .crop-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
    overflow:hidden;transition:border-color .2s,transform .15s}
  .crop-card:hover{border-color:var(--text);transform:translateY(-2px)}
  .crop-card img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;
    background:repeating-conic-gradient(var(--surface) 0% 25%,var(--surface2) 0% 50%) 50%/16px 16px}
  .crop-card .crop-info{padding:.6rem .75rem;display:flex;align-items:center;justify-content:space-between}
  .crop-card .crop-meta{display:flex;flex-direction:column;gap:.15rem}
  .crop-card .crop-label{font-size:.78rem;font-weight:600}
  .crop-card .crop-score{font-size:.72rem;color:var(--text2);font-family:'JetBrains Mono',monospace}
  .crop-card .crop-dl{width:30px;height:30px;border-radius:6px;border:1px solid var(--border);
    background:var(--surface);display:grid;place-items:center;cursor:pointer;
    color:var(--text2);font-size:.85rem;transition:all .2s;flex-shrink:0;text-decoration:none}
  .crop-card .crop-dl:hover{border-color:var(--color-car);color:var(--color-car);background:var(--surface2)}
  .crop-empty{color:var(--text2);font-size:.9rem;padding:2rem;text-align:center}

  /* ── Spinner ── */
  .spinner-overlay{position:absolute;inset:0;background:color-mix(in srgb,var(--bg) 85%,transparent);
    display:none;place-items:center;z-index:10;border-radius:var(--radius)}
  .spinner-overlay.active{display:grid}
  .spinner{width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--color-car);
    border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  footer{text-align:center;padding:1.5rem;color:var(--text2);font-size:.78rem}
</style>
</head>
<body data-theme="dark">

<header>
  <div class="logo">IS</div>
  <h1>Instance Segmentation</h1>
  <div class="badge">ONNX &middot; Mask R-CNN &middot; Local</div>
  <div class="spacer"></div>
  <div class="theme-row">
    <div class="theme-btn active" data-t="dark" title="Dark"></div>
    <div class="theme-btn" data-t="light" title="Light"></div>
    <div class="theme-btn" data-t="midnight" title="Midnight"></div>
    <div class="theme-btn" data-t="forest" title="Forest"></div>
  </div>
</header>

<div class="container">
  <!-- ═══════════ LEFT ═══════════ -->
  <div>
    <div class="panel">
      <h2>Upload Image</h2>
      <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
        <input type="file" id="file-input" accept="image/*">
        <div class="icon">&#128247;</div>
        <p>Drop an image here or click to browse</p>
      </div>
      <div id="preview-container"><img id="preview-img" src="" alt="preview"></div>
      <div class="slider-group">
        <label>Confidence threshold <span id="conf-val">0.50</span></label>
        <input type="range" id="conf-slider" min="0" max="100" value="50">
      </div>
      <button class="btn-primary" id="run-btn" disabled>Run Instance Segmentation</button>
    </div>

    <div class="panel">
      <h2>Class Colors</h2>
      <div class="color-row">
        <span class="color-label">Car</span>
        <div class="color-swatch-wrapper">
          <div class="color-swatch" id="swatch-car" style="background:#2ecc71"></div>
          <input type="color" id="picker-car" value="#2ecc71">
        </div>
        <input type="text" class="color-hex" id="hex-car" value="#2ecc71" maxlength="7" spellcheck="false">
      </div>
      <div class="color-row">
        <span class="color-label">Pedestrian</span>
        <div class="color-swatch-wrapper">
          <div class="color-swatch" id="swatch-ped" style="background:#3498db"></div>
          <input type="color" id="picker-ped" value="#3498db">
        </div>
        <input type="text" class="color-hex" id="hex-ped" value="#3498db" maxlength="7" spellcheck="false">
      </div>
      <div class="preset-row" id="preset-row"></div>
    </div>

    <div class="panel" id="det-panel" hidden>
      <h2>Detections</h2>
      <div style="overflow-x:auto">
        <table class="det-table" id="det-table">
          <thead><tr><th>#</th><th>Label</th><th>Confidence</th><th></th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ═══════════ RIGHT ═══════════ -->
  <div class="result-panel">
    <div class="panel" style="padding-bottom:.5rem">
      <!-- Tab bar -->
      <div class="tab-bar">
        <button class="tab-btn active" data-tab="overview">Overview</button>
        <button class="tab-btn" data-tab="bboxes">Bounding Boxes</button>
        <button class="tab-btn" data-tab="polygons">Polygons</button>
      </div>

      <!-- TAB: Overview -->
      <div class="tab-content active" id="tab-overview">
        <div id="result-image-container">
          <div class="spinner-overlay" id="spinner"><div class="spinner"></div></div>
          <div class="placeholder" id="result-placeholder">Upload an image and click <em>Run</em></div>
          <img id="result-img" src="" alt="result" style="display:none">
        </div>
        <div class="stats-bar" id="stats-bar" hidden>
          <div class="stat-card cars"><div class="value" id="st-cars">0</div><div class="label">Cars</div></div>
          <div class="stat-card peds"><div class="value" id="st-peds">0</div><div class="label">Pedestrians</div></div>
          <div class="stat-card"><div class="value" id="st-total">0</div><div class="label">Total</div></div>
          <div class="stat-card time"><div class="value" id="st-time">&mdash;</div><div class="label">Inference (s)</div></div>
        </div>
      </div>

      <!-- TAB: Bounding Boxes -->
      <div class="tab-content" id="tab-bboxes">
        <div class="crop-toolbar" id="bbox-toolbar" hidden>
          <span class="count" id="bbox-count"></span>
          <button class="btn-download-all" id="dl-all-bbox">&#11015; Download all (.zip)</button>
        </div>
        <div id="bbox-grid" class="crop-grid"></div>
        <div class="crop-empty" id="bbox-empty">Run inference to see individual bounding-box crops</div>
      </div>

      <!-- TAB: Polygons -->
      <div class="tab-content" id="tab-polygons">
        <div class="crop-toolbar" id="poly-toolbar" hidden>
          <span class="count" id="poly-count"></span>
          <button class="btn-download-all" id="dl-all-poly">&#11015; Download all (.zip)</button>
        </div>
        <div id="poly-grid" class="crop-grid"></div>
        <div class="crop-empty" id="poly-empty">Run inference to see individual polygon crops</div>
      </div>
    </div>
  </div>
</div>

<footer>Instance Segmentation &mdash; Azure AutoML for Images &middot; ONNX Runtime &middot; Local inference</footer>

<script>
/* ═══════════════════════════════════════════════════════
   DOM REFS
   ═══════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const fileInput=$('file-input'), previewCont=$('preview-container'), previewImg=$('preview-img'),
  confSlider=$('conf-slider'), confVal=$('conf-val'), runBtn=$('run-btn'),
  resultImg=$('result-img'), placeholder=$('result-placeholder'), spinner=$('spinner'),
  statsBar=$('stats-bar'), detPanel=$('det-panel'),
  detTbody=$('det-table').querySelector('tbody'), uploadZone=$('upload-zone'),
  pickerCar=$('picker-car'), pickerPed=$('picker-ped'),
  swatchCar=$('swatch-car'), swatchPed=$('swatch-ped'),
  hexCar=$('hex-car'), hexPed=$('hex-ped'), presetRow=$('preset-row'),
  bboxGrid=$('bbox-grid'), polyGrid=$('poly-grid'),
  bboxToolbar=$('bbox-toolbar'), polyToolbar=$('poly-toolbar'),
  bboxCount=$('bbox-count'), polyCount=$('poly-count'),
  bboxEmpty=$('bbox-empty'), polyEmpty=$('poly-empty'),
  dlAllBbox=$('dl-all-bbox'), dlAllPoly=$('dl-all-poly');

let selectedFile = null;
let lastBboxB64 = [];   // [{label, score, b64}, ...]
let lastPolyB64 = [];

/* ═══════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════ */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  });
});

/* ═══════════════════════════════════════════════════════
   COLOR PRESETS
   ═══════════════════════════════════════════════════════ */
const COLOR_PRESETS = [
  {name:'Default',car:'#2ecc71',ped:'#3498db'},{name:'Warm',car:'#f39c12',ped:'#e74c3c'},
  {name:'Neon',car:'#00ff88',ped:'#ff00ff'},{name:'Pastel',car:'#a8e6cf',ped:'#dcedc1'},
  {name:'Electric',car:'#00d2ff',ped:'#ff6b6b'},{name:'Monochrome',car:'#ffffff',ped:'#888888'},
];
COLOR_PRESETS.forEach(p => {
  const c = document.createElement('div'); c.className='preset-chip';
  c.innerHTML=`<span class="dot" style="background:${p.car}"></span><span class="dot" style="background:${p.ped}"></span> ${p.name}`;
  c.addEventListener('click',()=>{setCarColor(p.car);setPedColor(p.ped)});
  presetRow.appendChild(c);
});

/* ═══════════════════════════════════════════════════════
   COLOR SYNC
   ═══════════════════════════════════════════════════════ */
function setCarColor(h){pickerCar.value=h;swatchCar.style.background=h;hexCar.value=h;document.documentElement.style.setProperty('--color-car',h)}
function setPedColor(h){pickerPed.value=h;swatchPed.style.background=h;hexPed.value=h;document.documentElement.style.setProperty('--color-ped',h)}
pickerCar.addEventListener('input',()=>setCarColor(pickerCar.value));
pickerPed.addEventListener('input',()=>setPedColor(pickerPed.value));
function syncHex(inp,setter,fb){let v=inp.value.trim();if(!v.startsWith('#'))v='#'+v;/^#[0-9a-fA-F]{6}$/.test(v)?setter(v):inp.value=fb.value}
hexCar.addEventListener('change',()=>syncHex(hexCar,setCarColor,pickerCar));
hexPed.addEventListener('change',()=>syncHex(hexPed,setPedColor,pickerPed));

/* ═══════════════════════════════════════════════════════
   THEME
   ═══════════════════════════════════════════════════════ */
document.querySelectorAll('.theme-btn').forEach(b=>{
  b.addEventListener('click',()=>{document.body.dataset.theme=b.dataset.t;
    document.querySelectorAll('.theme-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active')});
});

/* ═══════════════════════════════════════════════════════
   SLIDER / DRAG-DROP
   ═══════════════════════════════════════════════════════ */
confSlider.addEventListener('input',()=>{confVal.textContent=(confSlider.value/100).toFixed(2)});
['dragenter','dragover'].forEach(e=>uploadZone.addEventListener(e,ev=>{ev.preventDefault();uploadZone.classList.add('drag-over')}));
['dragleave','drop'].forEach(e=>uploadZone.addEventListener(e,ev=>{ev.preventDefault();uploadZone.classList.remove('drag-over')}));
uploadZone.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){fileInput.files=ev.dataTransfer.files;handleFile(ev.dataTransfer.files[0])}});
fileInput.addEventListener('change',()=>{if(fileInput.files.length)handleFile(fileInput.files[0])});
function handleFile(f){selectedFile=f;const r=new FileReader();r.onload=e=>{previewImg.src=e.target.result;previewCont.style.display='block';runBtn.disabled=false};r.readAsDataURL(f)}

/* ═══════════════════════════════════════════════════════
   DOWNLOAD HELPERS
   ═══════════════════════════════════════════════════════ */
function downloadB64(b64DataUri, filename) {
  const a = document.createElement('a');
  a.href = b64DataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

async function downloadAllZip(items, prefix) {
  // items = [{label, score, b64 (raw, no prefix)}, ...]
  // POST to /download_zip
  const resp = await fetch('/download_zip', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({items, prefix})
  });
  if (!resp.ok) { alert('Download failed'); return; }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = prefix + '_crops.zip';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ═══════════════════════════════════════════════════════
   BUILD CROP GRID
   ═══════════════════════════════════════════════════════ */
function buildCropGrid(container, items, prefix) {
  // items = [{label, score, data_uri, raw_b64}, ...]
  container.innerHTML = '';
  items.forEach((it, i) => {
    const idx = i + 1;
    const fname = `${prefix}_${idx}_${it.label}_${(it.score*100).toFixed(0)}pct.png`;
    const clr = it.label==='car' ? pickerCar.value : it.label==='pedestrian' ? pickerPed.value : 'var(--danger)';
    const card = document.createElement('div'); card.className = 'crop-card';
    card.innerHTML = `
      <img src="${it.data_uri}" alt="${it.label} #${idx}">
      <div class="crop-info">
        <div class="crop-meta">
          <span class="crop-label" style="color:${clr}">${it.label.toUpperCase()} #${idx}</span>
          <span class="crop-score">${(it.score*100).toFixed(1)}%</span>
        </div>
        <a class="crop-dl" title="Download PNG">&#11015;</a>
      </div>`;
    card.querySelector('.crop-dl').addEventListener('click', e => {
      e.preventDefault();
      downloadB64(it.data_uri, fname);
    });
    container.appendChild(card);
  });
}

/* ═══════════════════════════════════════════════════════
   RUN INFERENCE
   ═══════════════════════════════════════════════════════ */
runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  spinner.classList.add('active');
  placeholder.style.display = 'none';
  resultImg.style.display = 'none';
  statsBar.hidden = true; detPanel.hidden = true;
  bboxGrid.innerHTML=''; polyGrid.innerHTML='';
  bboxToolbar.hidden=true; polyToolbar.hidden=true;
  bboxEmpty.style.display='block'; polyEmpty.style.display='block';

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('threshold', (confSlider.value/100).toFixed(2));
  formData.append('color_car', pickerCar.value);
  formData.append('color_ped', pickerPed.value);

  try {
    const resp = await fetch('/predict', {method:'POST', body:formData});
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    // ── Overview tab ──
    resultImg.src = data.annotated_image;
    resultImg.style.display = 'block';
    const dets = data.detections;
    const cars = dets.filter(d=>d.label==='car').length;
    const peds = dets.filter(d=>d.label==='pedestrian').length;
    $('st-cars').textContent=cars; $('st-peds').textContent=peds;
    $('st-total').textContent=dets.length; $('st-time').textContent=data.inference_time.toFixed(3);
    statsBar.hidden=false;

    // ── Detections table ──
    const cCar=pickerCar.value, cPed=pickerPed.value;
    detTbody.innerHTML='';
    dets.forEach((d,i)=>{
      const pct=(d.score*100).toFixed(1);
      const color=d.label==='car'?cCar:d.label==='pedestrian'?cPed:'var(--danger)';
      detTbody.innerHTML+=`<tr>
        <td style="color:var(--text2)">${i+1}</td>
        <td><span style="color:${color};font-weight:600">${d.label.toUpperCase()}</span></td>
        <td>${pct}%</td>
        <td><span class="conf-bar"><span class="conf-bar-fill" style="width:${pct}%;background:${color}"></span></span></td>
      </tr>`;
    });
    detPanel.hidden = dets.length===0;

    // ── Bounding Boxes tab ──
    lastBboxB64 = data.bbox_crops.map((b,i) => ({
      label: dets[i].label, score: dets[i].score,
      data_uri: 'data:image/png;base64,' + b,
      raw_b64: b
    }));
    if (lastBboxB64.length) {
      buildCropGrid(bboxGrid, lastBboxB64, 'bbox');
      bboxCount.textContent = `${lastBboxB64.length} crop${lastBboxB64.length>1?'s':''}`;
      bboxToolbar.hidden = false;
      bboxEmpty.style.display = 'none';
    }

    // ── Polygons tab ──
    lastPolyB64 = data.polygon_crops.map((b,i) => ({
      label: dets[i].label, score: dets[i].score,
      data_uri: 'data:image/png;base64,' + b,
      raw_b64: b
    }));
    if (lastPolyB64.length) {
      buildCropGrid(polyGrid, lastPolyB64, 'polygon');
      polyCount.textContent = `${lastPolyB64.length} crop${lastPolyB64.length>1?'s':''}`;
      polyToolbar.hidden = false;
      polyEmpty.style.display = 'none';
    }

  } catch(e) {
    alert('Error: ' + e.message);
  } finally {
    spinner.classList.remove('active');
    runBtn.disabled = false;
  }
});

/* ── Download-all buttons ── */
dlAllBbox.addEventListener('click', () => {
  if (!lastBboxB64.length) return;
  downloadAllZip(lastBboxB64.map((it,i)=>({
    filename: `bbox_${i+1}_${it.label}_${(it.score*100).toFixed(0)}pct.png`,
    b64: it.raw_b64
  })), 'bounding_boxes');
});
dlAllPoly.addEventListener('click', () => {
  if (!lastPolyB64.length) return;
  downloadAllZip(lastPolyB64.map((it,i)=>({
    filename: `polygon_${i+1}_${it.label}_${(it.score*100).toFixed(0)}pct.png`,
    b64: it.raw_b64
  })), 'polygons');
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# Flask application
# ──────────────────────────────────────────────────────────────────────

def create_app(model_path: str, labels_path: str) -> Flask:
    """Create and configure the Flask application.

    Args:
        model_path: Path to the ONNX model file.
        labels_path: Path to the labels.json file.

    Returns:
        A configured Flask app instance.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

    class_names: list[str] = load_labels(labels_path)
    session: ort.InferenceSession = load_session(model_path)

    print(f"Model loaded: {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Providers: {session.get_providers()}")

    @app.route("/")
    def index() -> str:
        """Serve the main UI page."""
        return render_template_string(HTML_TEMPLATE)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Handle an inference request.

        Expects a multipart form with:
            - image: the uploaded image file.
            - threshold: confidence threshold (0..1).
            - color_car: hex color for car class.
            - color_ped: hex color for pedestrian class.

        Returns:
            JSON with annotated_image (data-URI), detections list,
            bbox_crops (list of raw base64 PNGs),
            polygon_crops (list of raw base64 PNGs),
            inference_time (seconds).
        """
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        threshold: float = float(request.form.get("threshold", "0.5"))
        color_car_hex: str = request.form.get("color_car", "#2ecc71")
        color_ped_hex: str = request.form.get("color_ped", "#3498db")

        user_color_map: dict[str, tuple[int, int, int]] = {
            "car": hex_to_rgb(color_car_hex),
            "pedestrian": hex_to_rgb(color_ped_hex),
        }

        try:
            pil_image = Image.open(file.stream)
        except Exception as exc:
            return jsonify({"error": f"Invalid image: {exc}"}), 400

        INPUT_H, INPUT_W = 600, 800
        img_tensor, orig_h, orig_w = preprocess(pil_image, INPUT_H, INPUT_W)

        t0 = time.time()
        boxes, labels, scores, masks = run_inference(session, img_tensor)
        inference_time: float = time.time() - t0

        result = annotate_and_extract(
            pil_image, boxes, labels, scores, masks,
            class_names, INPUT_H, INPUT_W,
            color_map=user_color_map,
            score_threshold=threshold,
        )

        # Encode crops to raw base64
        bbox_b64: list[str] = [pil_to_raw_b64(c) for c in result["bbox_crops"]]
        poly_b64: list[str] = [pil_to_raw_b64(c) for c in result["polygon_crops"]]

        return jsonify({
            "annotated_image": pil_to_base64(result["annotated_image"], "JPEG"),
            "detections": result["detections"],
            "bbox_crops": bbox_b64,
            "polygon_crops": poly_b64,
            "inference_time": round(inference_time, 4),
        })

    @app.route("/download_zip", methods=["POST"])
    def download_zip():
        """Generate a ZIP of crop images from base64 data.

        Expects JSON body:
            - items: list of {filename: str, b64: str}
            - prefix: str used in the ZIP filename.

        Returns:
            A ZIP file as an attachment.
        """
        data = request.get_json(force=True)
        items: list[dict] = data.get("items", [])
        prefix: str = data.get("prefix", "crops")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                fname: str = item.get("filename", "crop.png")
                b64_str: str = item.get("b64", "")
                try:
                    img_bytes = base64.b64decode(b64_str)
                    zf.writestr(fname, img_bytes)
                except Exception:
                    continue
        buf.seek(0)

        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{prefix}_crops.zip",
        )

    return app


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and launch the Flask dev server."""
    parser = argparse.ArgumentParser(
        description="Local ONNX inference web UI for instance segmentation",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.path.join("onnx_model", "train_artifacts", "model.onnx"),
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--labels", type=str,
        default=os.path.join("onnx_model", "train_artifacts", "labels.json"),
        help="Path to labels.json",
    )
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: ONNX model not found at '{args.model}'")
        print("  Download it first using the notebook (Section 3) or Azure ML Studio.")
        print("  Then pass the path with: --model <path_to_model.onnx>")
        return

    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels file not found at '{args.labels}'")
        print("  Download it first using the notebook (Section 3) or Azure ML Studio.")
        print("  Then pass the path with: --labels <path_to_labels.json>")
        return

    app = create_app(args.model, args.labels)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
