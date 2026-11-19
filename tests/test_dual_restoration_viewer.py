"""
Unit & Integration Tests for Kallvettu Digital Restoration Dual Image Viewer & Performance Pipeline.

Validates:
1. Fast SHA-256 image identity & attribute caching.
2. Preview-first image downsampling preserving full-resolution metadata.
3. Squeeze-rubbing caching with compound key (sha256, algo_version, clip_limit, tile_grid, max_dim).
4. Sub-millisecond warm cache retrieval across 1024x768, 1920x1080, and 4000x3000 resolutions.
5. Robust HTML/JavaScript OpenSeadragon dual viewer structure:
   - Primary & secondary viewports (openseadragon-primary, openseadragon-secondary).
   - Independent spinners (#p-spinner, #s-spinner).
   - Independent error banners (#p-error-banner, #s-error-banner).
   - Status pills (#p-status-pill, #s-status-pill).
   - 6 visual states: LOADING, LOADED, PROCESSING, FAILED, IMAGE_NOT_FOUND, TIMEOUT.
   - Anti-reentrant mutual recursion locks for viewport synchronization (isSyncLock).
   - Watchdog timer preventing indefinite loading spinners.
   - Graceful fallback on missing or corrupted image assets.
"""

import io
import time
import pytest
import numpy as np
from PIL import Image

from services.iiif.viewer_component import (
    render_openseadragon_viewer,
    image_to_base64_data_uri,
    get_image_dimensions,
    compute_image_sha256,
    get_cached_squeeze_rubbing,
    _B64_CACHE,
    _SQUEEZE_CACHE,
)


def test_sha256_image_identity():
    """Verify SHA-256 calculation is deterministic, fast, and cached on PIL objects."""
    img = Image.new("RGB", (640, 480), color=(180, 160, 140))
    t0 = time.perf_counter()
    sha1 = compute_image_sha256(img)
    t_first = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sha2 = compute_image_sha256(img)
    t_second = (time.perf_counter() - t0) * 1000

    assert sha1 == sha2
    assert len(sha1) == 64
    assert hasattr(img, "_kallvettu_sha256")
    assert img._kallvettu_sha256 == sha1
    assert t_second < 0.1, f"Cached SHA retrieval took {t_second:.4f} ms (expected < 0.1 ms)"

    # Array input
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    sha_arr = compute_image_sha256(arr)
    assert len(sha_arr) == 64

    # None input
    assert compute_image_sha256(None) == "empty"


def test_image_to_base64_downsampling_and_caching():
    """Verify lightweight preview generation, downsampling, and zero-latency caching."""
    large_img = Image.new("RGB", (3000, 2000), color=(100, 110, 120))
    t0 = time.perf_counter()
    uri_cold = image_to_base64_data_uri(large_img, quality=85, max_dim=1600)
    t_cold = (time.perf_counter() - t0) * 1000

    assert uri_cold.startswith("data:image/jpeg;base64,")
    # Verify downsampled size is significantly smaller than full resolution base64
    assert len(uri_cold) < 500_000, f"Payload {len(uri_cold)} bytes is larger than expected"

    t0 = time.perf_counter()
    uri_warm = image_to_base64_data_uri(large_img, quality=85, max_dim=1600)
    t_warm = (time.perf_counter() - t0) * 1000

    assert uri_cold == uri_warm
    assert t_warm < 0.2, f"Cached Base64 lookup took {t_warm:.4f} ms"

    # Missing file returns empty string gracefully
    assert image_to_base64_data_uri("non_existent_file_path_12345.jpg") == ""
    assert image_to_base64_data_uri(None) == ""


def test_cached_squeeze_rubbing_pipeline():
    """Verify squeeze rubbing generates inverted relief + CLAHE and retrieves from cache."""
    test_img = Image.new("RGB", (800, 600), color=(120, 100, 80))
    
    # Cold execution
    t0 = time.perf_counter()
    sq_pil, sq_uri = get_cached_squeeze_rubbing(
        test_img,
        clip_limit=3.0,
        tile_grid=(8, 8),
        max_dim=1600,
        algo_version="2.1.0-squeeze-relief"
    )
    t_cold = (time.perf_counter() - t0) * 1000

    assert sq_pil is not None
    assert isinstance(sq_pil, Image.Image)
    assert sq_uri.startswith("data:image/jpeg;base64,")
    assert sq_pil.size == (800, 600)

    # Warm execution
    t0 = time.perf_counter()
    sq_pil_warm, sq_uri_warm = get_cached_squeeze_rubbing(
        test_img,
        clip_limit=3.0,
        tile_grid=(8, 8),
        max_dim=1600,
        algo_version="2.1.0-squeeze-relief"
    )
    t_warm = (time.perf_counter() - t0) * 1000

    assert sq_pil_warm is sq_pil
    assert sq_uri_warm == sq_uri
    assert t_warm < 0.1, f"Warm cache retrieval took {t_warm:.4f} ms (expected < 0.1 ms)"

    # Invalid input handling
    nil_pil, nil_uri = get_cached_squeeze_rubbing(None)
    assert nil_pil is None
    assert nil_uri == ""


def test_dual_openseadragon_viewer_component_markup(monkeypatch):
    """Verify OpenSeadragon dual viewer HTML includes all required elements and states."""
    import streamlit.components.v1 as components

    captured_html = []
    def mock_html(content, height, **kwargs):
        captured_html.append((content, height))

    monkeypatch.setattr(components, "html", mock_html)

    img1 = Image.new("RGB", (920, 480), color=(140, 120, 100))
    img2 = Image.new("RGB", (920, 480), color=(50, 50, 50))

    render_openseadragon_viewer(
        primary_image=img1,
        comparison_image=img2,
        viewer_height=580,
        primary_label="Original Mangulam Tamil-Brahmi Inscription",
        comparison_label="Lithic Squeeze Rubbing (Inverted Relief)",
        image_dimensions=(920, 480),
    )

    assert len(captured_html) == 1
    html = captured_html[0][0]

    # Verify containers & viewports
    assert 'id="openseadragon-primary"' in html
    assert 'id="openseadragon-secondary"' in html
    assert 'id="pane-primary"' in html
    assert 'id="pane-secondary"' in html

    # Verify spinners
    assert 'id="p-spinner"' in html
    assert 'id="s-spinner"' in html

    # Verify status pills
    assert 'id="p-status-pill"' in html
    assert 'id="s-status-pill"' in html

    # Verify error banners
    assert 'id="p-error-banner"' in html
    assert 'id="s-error-banner"' in html

    # Verify synchronization lock & controls
    assert 'id="s-sync"' in html
    assert 'isSyncLock' in html
    assert 'compViewer.viewport.panTo' in html

    # Verify 6 visual states supported in state manager
    for state in ["LOADING", "LOADED", "PROCESSING", "FAILED", "IMAGE_NOT_FOUND", "TIMEOUT"]:
        assert state in html

    # Verify watchdog timer
    assert "setTimeout" in html
    assert "7500" in html

    # Verify labels
    assert "Original Mangulam Tamil-Brahmi Inscription" in html
    assert "Lithic Squeeze Rubbing (Inverted Relief)" in html
    assert "920 × 480 px" in html


def test_dual_viewer_graceful_missing_images(monkeypatch):
    """Verify viewer handles missing image paths without throwing exceptions."""
    import streamlit.components.v1 as components

    captured_html = []
    monkeypatch.setattr(components, "html", lambda c, *args, **kw: captured_html.append((c, kw.get("height", 0))))

    render_openseadragon_viewer(
        primary_image="non_existent_file_original.jpg",
        comparison_image="non_existent_file_squeeze.jpg",
        viewer_height=500,
    )

    assert len(captured_html) == 1
    html = captured_html[0][0]
    # IMAGE_NOT_FOUND state logic should be present
    assert "IMAGE_NOT_FOUND" in html
    assert "resolvePrimaryTileSource" in html


def test_benchmark_resolutions_performance():
    """Verify performance metrics across 1024x768, 1920x1080, and 4000x3000."""
    resolutions = [
        (1024, 768),
        (1920, 1080),
        (4000, 3000),
    ]

    for w, h in resolutions:
        # Create test image with noisy texture
        arr = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        # 1. Image decoding / hashing
        t0 = time.perf_counter()
        sha = compute_image_sha256(img)
        t_hash = (time.perf_counter() - t0) * 1000

        # 2. Squeeze rubbing cold
        t0 = time.perf_counter()
        sq_pil, sq_uri = get_cached_squeeze_rubbing(img, max_dim=1600)
        t_squeeze_cold = (time.perf_counter() - t0) * 1000

        # 3. Squeeze rubbing warm
        t0 = time.perf_counter()
        sq_pil_w, sq_uri_w = get_cached_squeeze_rubbing(img, max_dim=1600)
        t_squeeze_warm = (time.perf_counter() - t0) * 1000

        # Assertions
        assert t_squeeze_warm < 0.2, f"Warm cache for {w}x{h} too slow: {t_squeeze_warm:.4f} ms"
        assert len(sq_uri) > 1000
        assert len(sq_uri) < 2_000_000, f"Payload size for {w}x{h} too large: {len(sq_uri)} bytes"
