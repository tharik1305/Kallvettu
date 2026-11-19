"""
OpenSeadragon Deep-Zoom Inscription Viewer Component for Streamlit (Phase D).

Features:
- Pure client-side OpenSeadragon 4.1.1 viewer rendering via Streamlit Components.
- Pan, deep-zoom, rotation (90° CW / CCW), and fit-to-screen home controls.
- Dynamic Image API 3.0 tileSource loading with automatic offline fallback.
- Live zoom percentage and master pixel dimension badges.
- Side-by-side comparative inspection between raw master plate and restored CLAHE / Squeeze rubbing.
- Epigraphical character annotation overlays color-coded by legibility status:
  CLEAR (gold), WEATHERED (amber), PARTIAL (cyan), ILLEGIBLE (crimson).
- Interactive tooltip cards displaying Unicode label, diplomatic reading, and coordinates.
- Spatial region restoration for shared deep links.
- Robust visual state machine: LOADING, LOADED, PROCESSING, FAILED, IMAGE_NOT_FOUND, TIMEOUT.
- Independent loaders, spinners, and error banners for primary and secondary panels.
- Anti-reentrant pan & zoom synchronization engine.
- SHA-256 image identity and in-memory caching for squeeze rubbings & lightweight previews.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image
import cv2
import numpy as np
import streamlit as st

logger = logging.getLogger("kallvettu.iiif.viewer")

# In-memory LRU caches for ultra-fast reruns
_B64_CACHE: Dict[Tuple[str, int, Optional[int]], str] = {}
_SQUEEZE_CACHE: Dict[Tuple[str, str, float, Tuple[int, int], Optional[int]], Tuple[Image.Image, str]] = {}


def compute_image_sha256(image_input: Any) -> str:
    """Computes a deterministic, ultra-fast SHA-256 identifier for an image input."""
    if image_input is None:
        return "empty"
    try:
        if hasattr(image_input, "_kallvettu_sha256"):
            return image_input._kallvettu_sha256

        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                with open(image_input, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                    return sha
            return hashlib.sha256(image_input.encode("utf-8")).hexdigest()

        elif isinstance(image_input, bytes):
            return hashlib.sha256(image_input).hexdigest()

        elif isinstance(image_input, Image.Image):
            if hasattr(image_input, "filename") and image_input.filename and os.path.isfile(image_input.filename):
                with open(image_input.filename, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
            else:
                raw_bytes = image_input.tobytes()
                header = f"{image_input.size}_{image_input.mode}_".encode("utf-8")
                sample = raw_bytes[:32768] + raw_bytes[-32768:]
                sha = hashlib.sha256(header + sample).hexdigest()
            image_input._kallvettu_sha256 = sha
            return sha

        elif isinstance(image_input, np.ndarray):
            header = f"{image_input.shape}_{image_input.dtype}_".encode("utf-8")
            sample = image_input.tobytes()[:32768] + image_input.tobytes()[-32768:]
            return hashlib.sha256(header + sample).hexdigest()

    except Exception as e:
        logger.debug("SHA-256 computation fallback: %s", e)

    return hashlib.sha256(str(image_input).encode("utf-8")).hexdigest()


def image_to_base64_data_uri(
    image_input: Union[Image.Image, bytes, str, np.ndarray],
    quality: int = 85,
    max_dim: Optional[int] = None,
) -> str:
    """
    Converts a PIL Image, bytes, NumPy array, or file path to a base64 data URI.
    Caches the result by SHA-256, quality, and max_dim for zero-latency Streamlit reruns.
    """
    if image_input is None:
        return ""

    if isinstance(image_input, str) and image_input.startswith("data:image"):
        return image_input

    sha = compute_image_sha256(image_input)
    cache_key = (sha, quality, max_dim)
    if cache_key in _B64_CACHE:
        return _B64_CACHE[cache_key]

    try:
        pil_img: Optional[Image.Image] = None

        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                pil_img = Image.fromarray(image_input, mode="L")
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            if os.path.isfile(image_input):
                pil_img = Image.open(image_input)
            else:
                # Missing file path
                return ""

        if pil_img is not None:
            # Aspect-ratio preserving downsampling for lightweight deep-zoom canvas
            if max_dim and max(pil_img.size) > max_dim:
                ratio = max_dim / float(max(pil_img.size))
                new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64}"
            _B64_CACHE[cache_key] = data_uri
            return data_uri

    except Exception as e:
        logger.warning("Failed to convert image to data URI: %s", e)

    return ""


def get_image_dimensions(image_input: Any) -> Optional[Tuple[int, int]]:
    """Helper to safely determine true pixel dimensions (width, height)."""
    if image_input is None:
        return None
    if isinstance(image_input, Image.Image):
        return image_input.size
    elif isinstance(image_input, np.ndarray):
        return (image_input.shape[1], image_input.shape[0])
    elif isinstance(image_input, str) and os.path.isfile(image_input):
        try:
            with Image.open(image_input) as im:
                return im.size
        except Exception:
            return None
    elif isinstance(image_input, bytes):
        try:
            with Image.open(io.BytesIO(image_input)) as im:
                return im.size
        except Exception:
            return None
    return None


def get_cached_squeeze_rubbing(
    image_input: Any,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
    max_dim: Optional[int] = 1600,
    algo_version: str = "2.1.0-squeeze-relief",
) -> Tuple[Optional[Image.Image], str]:
    """
    Generates and caches an inverted relief squeeze rubbing using CPU-efficient OpenCV ops.
    Cache key: (image SHA-256, algorithm version, clip_limit, tile_grid, max_dim).
    Returns (squeeze_pil, squeeze_data_uri).
    """
    if image_input is None:
        return None, ""

    sha = compute_image_sha256(image_input)
    cache_key = (sha, algo_version, clip_limit, tile_grid, max_dim)
    if cache_key in _SQUEEZE_CACHE:
        return _SQUEEZE_CACHE[cache_key]

    try:
        # Convert to numpy array with minimal memory copies
        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            arr = image_input
        elif isinstance(image_input, str) and os.path.isfile(image_input):
            bgr = cv2.imread(image_input)
            if bgr is None:
                return None, ""
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None, ""
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            return None, ""

        h, w = arr.shape[:2]
        # Lightweight downsampling for preview canvas
        if max_dim and max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            nw, nh = int(w * scale), int(h * scale)
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)

        # Grayscale extraction
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.copy()

        # In-place inverted relief
        cv2.bitwise_not(gray, dst=gray)

        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        squeeze_arr = clahe.apply(gray)

        # Convert back to RGB for OpenSeadragon display
        squeeze_rgb = cv2.cvtColor(squeeze_arr, cv2.COLOR_GRAY2RGB)
        squeeze_pil = Image.fromarray(squeeze_rgb)

        # Encode to optimized base64 JPEG
        squeeze_uri = image_to_base64_data_uri(squeeze_pil, quality=85)

        _SQUEEZE_CACHE[cache_key] = (squeeze_pil, squeeze_uri)
        return squeeze_pil, squeeze_uri

    except Exception as e:
        logger.error("Failed to generate squeeze rubbing: %s", e)
        return None, ""


def render_openseadragon_viewer(
    primary_image: Optional[Union[Image.Image, bytes, str, np.ndarray]] = None,
    comparison_image: Optional[Union[Image.Image, bytes, str, np.ndarray]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    viewer_height: int = 600,
    primary_label: str = "Master High-Resolution Photograph",
    comparison_label: str = "Lithic Squeeze Rubbing (Inverted Relief)",
    tile_source: Optional[Union[str, Dict[str, Any]]] = None,
    comparison_tile_source: Optional[Union[str, Dict[str, Any]]] = None,
    initial_region: Optional[Dict[str, int]] = None,
    image_dimensions: Optional[Tuple[int, int]] = None,
    inscription_metadata: Optional[Dict[str, str]] = None,
    viewer_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Embeds an OpenSeadragon deep-zoom viewer into Streamlit.
    Supports single plate view or side-by-side synchronized comparison.

    Guarantees:
    - Display lightweight preview first, preserving full resolution data.
    - Zero stuck spinners and zero blank panels.
    - Full visual state machine: LOADING, LOADED, PROCESSING, FAILED, IMAGE_NOT_FOUND, TIMEOUT.
    - Robust independent loaders, spinners, and error banners for primary and secondary panels.
    - Anti-reentrant pan and zoom synchronization without triggering OpenCV reprocessing.
    """
    # Accept parameter aliases for backward compatibility
    if primary_image is None and "primary_image_uri" in kwargs:
        primary_image = kwargs["primary_image_uri"]
    if comparison_image is None and "comparison_image_uri" in kwargs:
        comparison_image = kwargs["comparison_image_uri"]
    if "height_px" in kwargs:
        viewer_height = kwargs["height_px"]
    if annotations is None and "glyph_boxes" in kwargs:
        adapted_anns = []
        for gb in kwargs["glyph_boxes"]:
            box = gb.get("box", (0, 0, 0, 0))
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                adapted_anns.append({
                    "bounding_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                    "character_label": gb.get("label", ""),
                    "script_category": gb.get("script", ""),
                    "legibility_status": "CLEAR",
                })
        annotations = adapted_anns

    # Determine master dimensions
    dims = image_dimensions or get_image_dimensions(primary_image)
    dim_text = f"{dims[0]} × {dims[1]} px" if dims else ""

    # Generate lightweight preview URIs (clamped to max_dim 1920 for instant rendering)
    primary_uri = image_to_base64_data_uri(primary_image, quality=85, max_dim=1920) if primary_image else ""
    comp_uri = image_to_base64_data_uri(comparison_image, quality=85, max_dim=1920) if comparison_image else ""
    is_side_by_side = bool(comp_uri or comparison_tile_source)

    ann_json = json.dumps(annotations or [])
    init_region_json = json.dumps(initial_region) if initial_region else "null"
    tile_source_json = json.dumps(tile_source) if tile_source else "null"
    comp_tile_source_json = json.dumps(comparison_tile_source) if comparison_tile_source else "null"

    # Stable component key for Streamlit iframe preservation
    component_key = viewer_id or f"osd_viewer_{compute_image_sha256(primary_image)[:10]}_{is_side_by_side}"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: #111820;
      color: #F1F2F3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    .viewer-container {{
      display: flex;
      gap: 10px;
      width: 100%;
      height: {viewer_height}px;
      background: #171D24;
      border: 1px solid #2D3642;
      border-radius: 8px;
      padding: 8px;
    }}
    .viewer-pane {{
      flex: 1;
      height: 100%;
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: #111820;
      border: 1px solid #2D3642;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }}
    .pane-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 12px;
      background: #202832;
      border-bottom: 1px solid #2D3642;
      font-size: 0.82rem;
      font-weight: 600;
      color: #C5A15A;
      z-index: 10;
      flex-shrink: 0;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .pane-title {{
      font-weight: 600;
      color: #F1F2F3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .meta-badge {{
      background: #171D24;
      border: 1px solid #2D3642;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.72rem;
      color: #94A3B8;
      font-weight: normal;
    }}
    .zoom-badge {{
      background: #1A2836;
      border: 1px solid #38BDF8;
      color: #38BDF8;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
    }}
    .status-pill {{
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 0.70rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      display: inline-flex;
      align-items: center;
      gap: 3px;
      transition: all 0.2s ease;
    }}
    .status-loading {{
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid #38BDF8;
      color: #38BDF8;
    }}
    .status-loaded {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid #10B981;
      color: #10B981;
    }}
    .status-processing {{
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid #F59E0B;
      color: #F59E0B;
    }}
    .status-failed {{
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid #EF4444;
      color: #EF4444;
    }}
    .status-not-found {{
      background: rgba(239, 68, 68, 0.25);
      border: 1px solid #EF4444;
      color: #FCA5A5;
    }}
    .status-timeout {{
      background: rgba(249, 115, 22, 0.2);
      border: 1px solid #F97316;
      color: #FDBA74;
    }}
    .toolbar {{
      display: flex;
      gap: 4px;
      align-items: center;
    }}
    .btn {{
      background: #202832;
      border: 1px solid #2D3642;
      color: #CBD5E1;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 500;
      transition: all 0.15s;
    }}
    .btn:hover {{
      background: #2D3642;
      color: #C5A15A;
      border-color: #C5A15A;
    }}
    .btn-active {{
      background: #3B2E15;
      color: #C5A15A;
      border-color: #C5A15A;
    }}
    .osd-viewport {{
      flex: 1;
      width: 100%;
      height: 100%;
      min-height: 0;
      min-width: 0;
      background: #0D1217;
      position: relative;
    }}
    /* Glyphs Overlay Styling */
    .glyph-overlay {{
      cursor: pointer;
      border-radius: 2px;
      transition: background 0.2s, box-shadow 0.2s;
    }}
    .glyph-clear {{
      border: 2px solid #C5A15A;
      background: rgba(197, 161, 90, 0.22);
    }}
    .glyph-clear:hover {{
      background: rgba(197, 161, 90, 0.45);
      box-shadow: 0 0 8px #C5A15A;
    }}
    .glyph-weathered {{
      border: 2px dashed #F59E0B;
      background: rgba(245, 158, 11, 0.22);
    }}
    .glyph-weathered:hover {{
      background: rgba(245, 158, 11, 0.45);
      box-shadow: 0 0 8px #F59E0B;
    }}
    .glyph-partial {{
      border: 2px dashed #38BDF8;
      background: rgba(56, 189, 248, 0.22);
    }}
    .glyph-partial:hover {{
      background: rgba(56, 189, 248, 0.45);
      box-shadow: 0 0 8px #38BDF8;
    }}
    .glyph-illegible {{
      border: 2px dotted #EF4444;
      background: rgba(239, 68, 68, 0.25);
    }}
    .glyph-illegible:hover {{
      background: rgba(239, 68, 68, 0.50);
      box-shadow: 0 0 8px #EF4444;
    }}
    /* Tooltip Banner */
    .overlay-tooltip {{
      position: absolute;
      bottom: 8px;
      left: 8px;
      background: rgba(23, 29, 36, 0.95);
      border: 1px solid #C5A15A;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 0.75rem;
      color: #F1F2F3;
      z-index: 20;
      pointer-events: none;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    /* Error / Notification Banner */
    .error-banner {{
      position: absolute;
      top: 40px;
      left: 12px;
      right: 12px;
      background: rgba(239, 68, 68, 0.92);
      color: #FFFFFF;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 0.75rem;
      display: none;
      z-index: 30;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    /* Loading Spinner */
    .spinner {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 32px;
      height: 32px;
      border: 3px solid #2D3642;
      border-top: 3px solid #C5A15A;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      z-index: 25;
    }}
    @keyframes spin {{
      0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
      100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <div class="viewer-container">
    <!-- Primary Plate Viewer -->
    <div class="viewer-pane" id="pane-primary">
      <div class="pane-header">
        <div class="header-left">
          <span class="pane-title">📷 {primary_label}</span>
          {'<span class="meta-badge" id="p-dims">📐 ' + dim_text + '</span>' if dim_text else ''}
          <span id="p-status-pill" class="status-pill status-loading">⏳ LOADING</span>
          <span id="p-zoom-level" class="zoom-badge">🔍 100%</span>
        </div>
        <div class="toolbar">
          <button class="btn" id="p-zoom-in" title="Deep Zoom In">+</button>
          <button class="btn" id="p-zoom-out" title="Deep Zoom Out">-</button>
          <button class="btn" id="p-home" title="Fit to Screen / Reset View">⟲</button>
          <button class="btn" id="p-rotate-cw" title="Rotate 90° Clockwise">↻ 90°</button>
          <button class="btn" id="p-rotate-ccw" title="Rotate 90° Counter-Clockwise">↺ 90°</button>
          <button class="btn btn-active" id="p-toggle-ann" title="Toggle Glyph Overlays">🏷️ Glyphs</button>
        </div>
      </div>
      <div id="p-spinner" class="spinner"></div>
      <div id="p-error-banner" class="error-banner"></div>
      <div id="openseadragon-primary" class="osd-viewport"></div>
      <div id="p-tooltip" class="overlay-tooltip"></div>
    </div>

    <!-- Secondary Comparative Viewer -->
    {'<div class="viewer-pane" id="pane-secondary">' if is_side_by_side else ''}
    {'<div class="pane-header"><div class="header-left"><span class="pane-title">🔬 ' + comparison_label + '</span><span id="s-status-pill" class="status-pill status-loading">⏳ LOADING</span></div><div class="toolbar"><button class="btn btn-active" id="s-sync" title="Synchronize Pan & Zoom">🔗 Sync Viewports</button></div></div>' if is_side_by_side else ''}
    {'<div id="s-spinner" class="spinner"></div>' if is_side_by_side else ''}
    {'<div id="s-error-banner" class="error-banner"></div>' if is_side_by_side else ''}
    {'<div id="openseadragon-secondary" class="osd-viewport"></div>' if is_side_by_side else ''}
    {'</div>' if is_side_by_side else ''}
  </div>

  <script>
    const annotations = {ann_json};
    const initialRegion = {init_region_json};
    const primaryFallbackUri = "{primary_uri}";
    const customTileSource = {tile_source_json};
    const isSideBySide = {"true" if is_side_by_side else "false"};
    const compUri = "{comp_uri}";
    const compTileSource = {comp_tile_source_json};

    // Robust Visual State Machine
    function setPanelState(panel, state, message) {{
      const pill = document.getElementById(panel + "-status-pill");
      const spinner = document.getElementById(panel + "-spinner");
      const errBanner = document.getElementById(panel + "-error-banner");

      if (!pill) return;

      pill.className = "status-pill";
      if (state === "LOADING") {{
        pill.classList.add("status-loading");
        pill.innerText = "⏳ LOADING";
        if (spinner) spinner.style.display = "block";
        if (errBanner) errBanner.style.display = "none";
      }} else if (state === "PROCESSING") {{
        pill.classList.add("status-processing");
        pill.innerText = "⚙️ PROCESSING";
        if (spinner) spinner.style.display = "block";
        if (errBanner) errBanner.style.display = "none";
      }} else if (state === "LOADED") {{
        pill.classList.add("status-loaded");
        pill.innerText = "✅ LOADED";
        if (spinner) spinner.style.display = "none";
        if (errBanner) errBanner.style.display = "none";
      }} else if (state === "FAILED") {{
        pill.classList.add("status-failed");
        pill.innerText = "❌ FAILED";
        if (spinner) spinner.style.display = "none";
        if (errBanner) {{
          errBanner.innerText = message || "⚠️ Image failed to render.";
          errBanner.style.display = "block";
        }}
      }} else if (state === "IMAGE_NOT_FOUND") {{
        pill.classList.add("status-not-found");
        pill.innerText = "🚫 NOT FOUND";
        if (spinner) spinner.style.display = "none";
        if (errBanner) {{
          errBanner.innerText = message || "⚠️ Stone epigraph image asset not found.";
          errBanner.style.display = "block";
        }}
      }} else if (state === "TIMEOUT") {{
        pill.classList.add("status-timeout");
        pill.innerText = "⏱️ TIMEOUT";
        if (spinner) spinner.style.display = "none";
        if (errBanner) {{
          errBanner.innerText = message || "⚠️ Image load timed out. Please refresh.";
          errBanner.style.display = "block";
        }}
      }}
    }}

    function resolvePrimaryTileSource() {{
      if (customTileSource && typeof customTileSource === "string") {{
        return customTileSource;
      }} else if (customTileSource && typeof customTileSource === "object") {{
        return customTileSource;
      }}
      if (primaryFallbackUri && primaryFallbackUri.length > 0) {{
        return {{
          type: "image",
          url: primaryFallbackUri
        }};
      }}
      return null;
    }}

    // Check for immediate IMAGE_NOT_FOUND state
    const pSource = resolvePrimaryTileSource();
    if (!pSource) {{
      setPanelState("p", "IMAGE_NOT_FOUND", "⚠️ Master inscription plate is unavailable.");
    }} else {{
      setPanelState("p", "LOADING");
    }}

    let primaryViewer = null;
    if (pSource && typeof OpenSeadragon !== "undefined") {{
      try {{
        primaryViewer = OpenSeadragon({{
          id: "openseadragon-primary",
          prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
          tileSources: pSource,
          showNavigationControl: false,
          maxZoomPixelRatio: 4.0,
          minZoomImageRatio: 0.8,
          visibilityRatio: 0.5,
          constrainDuringPan: true,
          animationTime: 0.35,
          blendTime: 0.1,
          springStiffness: 8.0
        }});

        // Hide spinner and mark LOADED on open
        primaryViewer.addHandler("open", function() {{
          setPanelState("p", "LOADED");

          // If initial spatial region is specified, pan and zoom to it
          if (initialRegion && initialRegion.w > 0 && initialRegion.h > 0) {{
            try {{
              const rect = primaryViewer.viewport.imageToViewportRectangle(
                initialRegion.x, initialRegion.y, initialRegion.w, initialRegion.h
              );
              primaryViewer.viewport.fitBounds(rect, true);
            }} catch(e) {{
              console.error("Failed to fit initial region:", e);
            }}
          }}
        }});

        // Graceful error fallback
        primaryViewer.addHandler("open-failed", function(e) {{
          console.warn("Primary tileSource failed:", e);
          if (customTileSource && primaryFallbackUri && customTileSource !== primaryFallbackUri) {{
            const errBanner = document.getElementById("p-error-banner");
            if (errBanner) {{
              errBanner.innerText = "⚠️ Remote IIIF tile service unavailable. Local image representation restored.";
              errBanner.style.display = "block";
              setTimeout(() => {{ errBanner.style.display = "none"; }}, 4000);
            }}
            try {{
              primaryViewer.open({{ type: "image", url: primaryFallbackUri }});
            }} catch(err) {{
              setPanelState("p", "FAILED", "⚠️ Failed to open local fallback image.");
            }}
          }} else {{
            setPanelState("p", "FAILED", "⚠️ Failed to load primary inscription image.");
          }}
        }});

        primaryViewer.addHandler("tile-load-failed", function(e) {{
          console.warn("Primary tile-load-failed:", e);
        }});

        // Update live zoom badge
        primaryViewer.addHandler("zoom", function() {{
          try {{
            const z = primaryViewer.viewport.getZoom(true);
            const el = document.getElementById("p-zoom-level");
            if (el) {{
              el.innerText = "🔍 " + Math.round(z * 100) + "%";
            }}
          }} catch(e) {{}}
        }});

        // Primary Toolbar Controls
        document.getElementById("p-zoom-in")?.addEventListener("click", () => primaryViewer.viewport.zoomBy(1.25));
        document.getElementById("p-zoom-out")?.addEventListener("click", () => primaryViewer.viewport.zoomBy(0.8));
        document.getElementById("p-home")?.addEventListener("click", () => primaryViewer.viewport.goHome());
        document.getElementById("p-rotate-cw")?.addEventListener("click", () => {{
          const current = primaryViewer.viewport.getRotation();
          primaryViewer.viewport.setRotation((current + 90) % 360);
        }});
        document.getElementById("p-rotate-ccw")?.addEventListener("click", () => {{
          const current = primaryViewer.viewport.getRotation();
          primaryViewer.viewport.setRotation((current - 90 + 360) % 360);
        }});

        // Add and toggle glyph annotations overlays
        let showAnnotations = true;
        const overlayElements = [];
        const tooltip = document.getElementById("p-tooltip");

        primaryViewer.addHandler("open", function() {{
          annotations.forEach((ann, idx) => {{
            if (ann.bounding_box) {{
              const bb = ann.bounding_box;
              const status = (ann.legibility_status || "CLEAR").toLowerCase();
              const elt = document.createElement("div");
              elt.className = "glyph-overlay glyph-" + status;
              elt.id = "glyph-marker-" + idx;

              elt.addEventListener("mouseenter", (ev) => {{
                if (tooltip) {{
                  tooltip.innerHTML = "<strong>" + (ann.character_label || "") + "</strong> (" + (ann.diplomatic_reading || "") + ") &bull; <em>" + (ann.script_category || "") + "</em><br><small>Status: " + (ann.legibility_status || "CLEAR") + " | [" + bb.x + ", " + bb.y + ", " + bb.w + ", " + bb.h + "]</small>";
                  tooltip.style.display = "block";
                }}
              }});
              elt.addEventListener("mouseleave", () => {{
                if (tooltip) tooltip.style.display = "none";
              }});

              try {{
                const rect = primaryViewer.viewport.imageToViewportRectangle(bb.x, bb.y, bb.w, bb.h);
                primaryViewer.addOverlay({{
                  element: elt,
                  location: rect
                }});
                overlayElements.push(elt);
              }} catch(e) {{
                console.error("Failed to add glyph overlay:", e);
              }}
            }}
          }});
        }});

        document.getElementById("p-toggle-ann")?.addEventListener("click", function() {{
          showAnnotations = !showAnnotations;
          this.classList.toggle("btn-active", showAnnotations);
          overlayElements.forEach(el => {{
            el.style.display = showAnnotations ? "block" : "none";
          }});
        }});

      }} catch (err) {{
        console.error("Primary OpenSeadragon initialization error:", err);
        setPanelState("p", "FAILED", "⚠️ Initialization error: " + err.message);
      }}
    }}

    // Secondary Comparative Viewer Engine
    let compViewer = null;
    let isSyncing = true;
    let isSyncLock = false;

    if (isSideBySide && typeof OpenSeadragon !== "undefined") {{
      const compSource = compTileSource || (compUri ? {{ type: "image", url: compUri }} : null);
      if (!compSource) {{
        setPanelState("s", "IMAGE_NOT_FOUND", "⚠️ Squeeze rubbing image unavailable.");
      }} else {{
        setPanelState("s", "LOADING");
        try {{
          compViewer = OpenSeadragon({{
            id: "openseadragon-secondary",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
            tileSources: compSource,
            showNavigationControl: false,
            maxZoomPixelRatio: 4.0,
            minZoomImageRatio: 0.8,
            visibilityRatio: 0.5,
            constrainDuringPan: true,
            animationTime: 0.35,
            blendTime: 0.1,
            springStiffness: 8.0
          }});

          compViewer.addHandler("open", function() {{
            setPanelState("s", "LOADED");
            // Match initial zoom/pan with primary viewer if already ready
            if (primaryViewer && primaryViewer.viewport && isSyncing) {{
              try {{
                compViewer.viewport.panTo(primaryViewer.viewport.getCenter(), true);
                compViewer.viewport.zoomTo(primaryViewer.viewport.getZoom(), null, true);
              }} catch(e) {{}}
            }}
          }});

          compViewer.addHandler("open-failed", function(e) {{
            console.warn("Secondary OpenSeadragon open-failed:", e);
            setPanelState("s", "FAILED", "⚠️ Failed to open squeeze rubbing image.");
          }});

          compViewer.addHandler("tile-load-failed", function(e) {{
            console.warn("Secondary tile-load-failed:", e);
          }});

          // Synchronize Pan & Zoom with mutual anti-recursion lock
          const syncBtn = document.getElementById("s-sync");
          syncBtn?.addEventListener("click", () => {{
            isSyncing = !isSyncing;
            syncBtn.classList.toggle("btn-active", isSyncing);
          }});

          if (primaryViewer) {{
            primaryViewer.addHandler("pan", function() {{
              if (isSyncing && !isSyncLock && compViewer && compViewer.viewport) {{
                isSyncLock = true;
                try {{
                  compViewer.viewport.panTo(primaryViewer.viewport.getCenter(), true);
                }} finally {{
                  setTimeout(() => {{ isSyncLock = false; }}, 20);
                }}
              }}
            }});

            primaryViewer.addHandler("zoom", function() {{
              if (isSyncing && !isSyncLock && compViewer && compViewer.viewport) {{
                isSyncLock = true;
                try {{
                  compViewer.viewport.zoomTo(primaryViewer.viewport.getZoom(), null, true);
                }} finally {{
                  setTimeout(() => {{ isSyncLock = false; }}, 20);
                }}
              }}
            }});

            compViewer.addHandler("pan", function() {{
              if (isSyncing && !isSyncLock && primaryViewer && primaryViewer.viewport) {{
                isSyncLock = true;
                try {{
                  primaryViewer.viewport.panTo(compViewer.viewport.getCenter(), true);
                }} finally {{
                  setTimeout(() => {{ isSyncLock = false; }}, 20);
                }}
              }}
            }});

            compViewer.addHandler("zoom", function() {{
              if (isSyncing && !isSyncLock && primaryViewer && primaryViewer.viewport) {{
                isSyncLock = true;
                try {{
                  primaryViewer.viewport.zoomTo(compViewer.viewport.getZoom(), null, true);
                }} finally {{
                  setTimeout(() => {{ isSyncLock = false; }}, 20);
                }}
              }}
            }});
          }}
        }} catch(err) {{
          console.error("Secondary OpenSeadragon initialization error:", err);
          setPanelState("s", "FAILED", "⚠️ Secondary initialization error: " + err.message);
        }}
      }}
    }}

    // Watchdog Timer: Guarantee panels never stay stuck in LOADING state
    setTimeout(() => {{
      const pPill = document.getElementById("p-status-pill");
      if (pPill && pPill.innerText.includes("LOADING")) {{
        setPanelState("p", "TIMEOUT", "⏱️ Master plate loading timed out.");
      }}
      const sPill = document.getElementById("s-status-pill");
      if (sPill && sPill.innerText.includes("LOADING")) {{
        setPanelState("s", "TIMEOUT", "⏱️ Squeeze rubbing loading timed out.");
      }}
    }}, 7500);
  </script>
</body>
</html>
"""
    st.components.v1.html(html_content, height=viewer_height + 25, scrolling=False)
