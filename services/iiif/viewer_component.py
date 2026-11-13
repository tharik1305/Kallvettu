"""
OpenSeadragon Deep-Zoom Inscription Viewer Component for Streamlit (Phase D).

Features:
- Pure client-side OpenSeadragon 4.1.1 viewer rendering via Streamlit Components.
- Pan, deep-zoom, rotation, and home controls with Kallvettu dark theme.
- Side-by-side comparative inspection between raw master image and restored CLAHE / Squeeze rubbing.
- Character annotation overlay markers with tooltip hover.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import streamlit as st


def image_to_base64_data_uri(image_input: Union[Image.Image, bytes, str]) -> str:
    """Converts a PIL Image, bytes, or file path to a base64 data URI."""
    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    elif isinstance(image_input, bytes):
        b64 = base64.b64encode(image_input).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    elif isinstance(image_input, str):
        if image_input.startswith("data:image"):
            return image_input
        # Check if local file
        import os
        if os.path.isfile(image_input):
            with open(image_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
        return image_input
    return ""


def render_openseadragon_viewer(
    primary_image: Optional[Union[Image.Image, bytes, str]] = None,
    comparison_image: Optional[Union[Image.Image, bytes, str]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    viewer_height: int = 560,
    primary_label: str = "Original Inscription",
    comparison_label: str = "Enhanced / Squeeze Rubbing",
    **kwargs: Any,
) -> None:
    """
    Embeds an OpenSeadragon deep-zoom viewer into Streamlit.
    Supports single plate view or side-by-side synchronized comparison.
    """
    # Accept parameter aliases for robust backward/forward compatibility
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
                })
        annotations = adapted_anns

    primary_uri = image_to_base64_data_uri(primary_image) if primary_image else ""
    comp_uri = image_to_base64_data_uri(comparison_image) if comparison_image else ""
    is_side_by_side = bool(comp_uri)

    ann_json = json.dumps(annotations or [])

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
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
    }}
    .viewer-container {{
      display: flex;
      gap: 12px;
      width: 100%;
      height: {viewer_height}px;
      background: #171D24;
      border: 1px solid #2D3642;
      border-radius: 8px;
      padding: 10px;
    }}
    .viewer-pane {{
      flex: 1;
      height: 100%;
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
    }}
    .osd-viewport {{
      flex: 1;
      width: 100%;
      height: 100%;
      background: #0D1217;
    }}
    .toolbar {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .btn {{
      background: #202832;
      border: 1px solid #2D3642;
      color: #CBD5E1;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.75rem;
    }}
    .btn:hover {{
      background: #2D3642;
      color: #C5A15A;
    }}
    .glyph-overlay {{
      border: 2px solid #C5A15A;
      background: rgba(197, 161, 90, 0.2);
      cursor: pointer;
      transition: all 0.2s;
    }}
    .glyph-overlay:hover {{
      border-color: #4CAF85;
      background: rgba(76, 175, 133, 0.35);
    }}
  </style>
</head>
<body>
  <div class="viewer-container">
    <div class="viewer-pane">
      <div class="pane-header">
        <span>📷 {primary_label}</span>
        <div class="toolbar">
          <button class="btn" id="p-zoom-in">+</button>
          <button class="btn" id="p-zoom-out">-</button>
          <button class="btn" id="p-home">⟲</button>
          <button class="btn" id="p-rotate">↻</button>
        </div>
      </div>
      <div id="openseadragon-primary" class="osd-viewport"></div>
    </div>

    {'<div class="viewer-pane">' if is_side_by_side else ''}
    {'<div class="pane-header"><span>🔬 ' + comparison_label + '</span><div class="toolbar"><button class="btn" id="s-sync">🔗 Sync</button></div></div>' if is_side_by_side else ''}
    {'<div id="openseadragon-secondary" class="osd-viewport"></div>' if is_side_by_side else ''}
    {'</div>' if is_side_by_side else ''}
  </div>

  <script>
    const primaryViewer = OpenSeadragon({{
      id: "openseadragon-primary",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: {{
        type: "image",
        url: "{primary_uri}"
      }},
      showNavigationControl: false,
      maxZoomPixelRatio: 3.0,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      animationTime: 0.3,
      blendTime: 0.1
    }});

    document.getElementById("p-zoom-in").addEventListener("click", () => primaryViewer.viewport.zoomBy(1.2));
    document.getElementById("p-zoom-out").addEventListener("click", () => primaryViewer.viewport.zoomBy(0.8));
    document.getElementById("p-home").addEventListener("click", () => primaryViewer.viewport.goHome());
    document.getElementById("p-rotate").addEventListener("click", () => {{
      const current = primaryViewer.viewport.getRotation();
      primaryViewer.viewport.setRotation((current + 90) % 360);
    }});

    // Add glyph annotations overlays
    const annotations = {ann_json};
    primaryViewer.addHandler("open", function() {{
      const imgWidth = primaryViewer.world.getItemAt(0).getContentSize().x;
      const imgHeight = primaryViewer.world.getItemAt(0).getContentSize().y;

      annotations.forEach(ann => {{
        if (ann.bounding_box) {{
          const bb = ann.bounding_box;
          const elt = document.createElement("div");
          elt.className = "glyph-overlay";
          elt.title = (ann.character_label || "") + " (" + (ann.script_category || "") + ")";

          const rect = primaryViewer.viewport.imageToViewportRectangle(bb.x, bb.y, bb.w, bb.h);
          primaryViewer.addOverlay({{
            element: elt,
            location: rect
          }});
        }}
      }});
    }});

    {'''
    const compViewer = OpenSeadragon({
      id: "openseadragon-secondary",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: {
        type: "image",
        url: "''' + comp_uri + '''"
      },
      showNavigationControl: false,
      maxZoomPixelRatio: 3.0,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.5,
      constrainDuringPan: true
    });

    let isSyncing = true;
    document.getElementById("s-sync")?.addEventListener("click", () => {
      isSyncing = !isSyncing;
      document.getElementById("s-sync").style.color = isSyncing ? "#C5A15A" : "#888";
    });

    primaryViewer.addHandler("pan", function() {
      if (isSyncing && compViewer) {
        compViewer.viewport.panTo(primaryViewer.viewport.getCenter(), true);
      }
    });

    primaryViewer.addHandler("zoom", function() {
      if (isSyncing && compViewer) {
        compViewer.viewport.zoomTo(primaryViewer.viewport.getZoom(), null, true);
      }
    });
    ''' if is_side_by_side else ''}
  </script>
</body>
</html>
"""
    st.components.v1.html(html_content, height=viewer_height + 25, scrolling=False)

