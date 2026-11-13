"""
Tests for Phase D: IIIF Deep-Zoom Inscription Viewer & Coordinates.
"""

import pytest
from services.iiif.manifest_builder import (
    build_iiif_image_info,
    build_iiif_presentation_manifest,
)
from services.iiif.coordinate_mapper import (
    scale_box_to_original,
    scale_box_to_display,
    parse_iiif_region,
)


def test_iiif_image_info():
    info = build_iiif_image_info("plate_001", width=4000, height=3000)
    assert info["@context"] == "http://iiif.io/api/image/3/context.json"
    assert info["width"] == 4000
    assert info["height"] == 3000
    assert info["type"] == "ImageService3"
    assert len(info["tiles"]) >= 1
    assert len(info["sizes"]) >= 1


def test_iiif_presentation_manifest():
    images = [
        {"id": "raw_plate", "width": 4000, "height": 3000, "label": "Original Inscription Photo"},
        {"id": "clahe_plate", "width": 4000, "height": 3000, "label": "CLAHE Contrast Enhanced"},
    ]
    manifest = build_iiif_presentation_manifest(
        inscription_id="KV-INSC-002",
        title_en="Thanjavur Brihadisvara Great Temple Inscription",
        title_ta="தஞ்சைப் பெருவுடையார் கோயில் கற்றளிக் கல்வெட்டு",
        images=images,
        metadata_fields={
            "script": "Medieval Chola Tamil",
            "dynasty": "Chola Dynasty",
            "ruler": "Rajaraja Chola I",
            "period": "11th Century CE",
        },
    )

    assert manifest["@context"] == "http://iiif.io/api/presentation/3/context.json"
    assert manifest["type"] == "Manifest"
    assert "KV-INSC-002" in manifest["id"]
    assert len(manifest["items"]) == 2  # 2 canvases
    assert len(manifest["metadata"]) == 4


def test_coordinate_mapper_scaling():
    # Downsampled display: 1000 x 750 from original 4000 x 3000 (scale factor 4.0)
    disp_box = {"x": 100, "y": 150, "w": 50, "h": 60}
    orig_box = scale_box_to_original(
        display_box=disp_box,
        display_width=1000,
        display_height=750,
        original_width=4000,
        original_height=3000,
    )
    assert orig_box["x"] == 400
    assert orig_box["y"] == 600
    assert orig_box["w"] == 200
    assert orig_box["h"] == 240

    # Reverse scale back to display
    roundtrip_disp = scale_box_to_display(
        original_box=orig_box,
        original_width=4000,
        original_height=3000,
        display_width=1000,
        display_height=750,
    )
    assert roundtrip_disp["x"] == 100
    assert roundtrip_disp["y"] == 150
    assert roundtrip_disp["w"] == 50
    assert roundtrip_disp["h"] == 60


def test_parse_iiif_region():
    w, h = 2000, 1000

    # 1. Full
    assert parse_iiif_region("full", w, h) == (0, 0, 2000, 1000)

    # 2. Square
    sq = parse_iiif_region("square", w, h)
    assert sq == (500, 0, 1000, 1000)

    # 3. Explicit pixels: 100,200,400,300
    assert parse_iiif_region("100,200,400,300", w, h) == (100, 200, 400, 300)

    # 4. Percentage: pct:10,20,50,40
    # 10% of 2000 = 200; 20% of 1000 = 200; 50% of 2000 = 1000; 40% of 1000 = 400
    assert parse_iiif_region("pct:10,20,50,40", w, h) == (200, 200, 1000, 400)


def test_render_openseadragon_viewer_signatures(monkeypatch):
    """Verify render_openseadragon_viewer accepts both standard and keyword alias arguments."""
    from services.iiif.viewer_component import render_openseadragon_viewer
    import streamlit.components.v1 as components

    captured_html = []
    def mock_html(content, height, **kwargs):
        captured_html.append((content, height))

    monkeypatch.setattr(components, "html", mock_html)

    # 1. Standard signature
    render_openseadragon_viewer(
        primary_image="data:image/jpeg;base64,123",
        comparison_image="data:image/jpeg;base64,456",
        viewer_height=580,
    )
    assert len(captured_html) == 1
    assert captured_html[-1][1] == 580 + 25

    # 2. Keyword alias signature (primary_image_uri, comparison_image_uri, height_px, glyph_boxes)
    render_openseadragon_viewer(
        primary_image_uri="data:image/jpeg;base64,123",
        comparison_image_uri="data:image/jpeg;base64,456",
        height_px=600,
        glyph_boxes=[{"box": (10, 20, 30, 40), "label": "CharA"}],
    )
    assert len(captured_html) == 2
    assert captured_html[-1][1] == 600 + 25
    assert "CharA" in captured_html[-1][0]


