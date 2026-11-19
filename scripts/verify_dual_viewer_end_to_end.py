import os
import sys
import time
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from services.iiif.viewer_component import (
    render_openseadragon_viewer,
    compute_image_sha256,
    get_cached_squeeze_rubbing,
    image_to_base64_data_uri,
    get_image_dimensions,
)
import streamlit.components.v1 as components

def verify_end_to_end():
    print("================================================================================")
    print("  KALLVETTU DUAL RESTORATION VIEWER END-TO-END VERIFICATION")
    print("================================================================================")

    # 1. Load canonical test image (Mangulam Tamil-Brahmi inscription)
    mangulam_path = os.path.join("assets", "samples", "mangulam.jpg")
    assert os.path.exists(mangulam_path), f"Sample missing: {mangulam_path}"
    
    t0 = time.perf_counter()
    master_pil = Image.open(mangulam_path)
    w, h = master_pil.size
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[✓] 1. Original Image Loaded: {w}x{h} px in {t_load:.2f} ms")

    # 2. SHA-256 Identification
    t0 = time.perf_counter()
    sha = compute_image_sha256(master_pil)
    t_sha = (time.perf_counter() - t0) * 1000
    print(f"[✓] 2. SHA-256 Hash: {sha[:16]}... in {t_sha:.2f} ms")

    # 3. Squeeze Rubbing Generation - Cold vs Warm
    t0 = time.perf_counter()
    sq_pil, sq_uri = get_cached_squeeze_rubbing(master_pil, clip_limit=3.0, tile_grid=(8, 8), max_dim=1600)
    t_sq_cold = (time.perf_counter() - t0) * 1000
    print(f"[✓] 3. Squeeze Rubbing (Cold): Generated in {t_sq_cold:.2f} ms (Payload: {len(sq_uri)/1024:.1f} KB)")

    t0 = time.perf_counter()
    sq_pil_warm, sq_uri_warm = get_cached_squeeze_rubbing(master_pil, clip_limit=3.0, tile_grid=(8, 8), max_dim=1600)
    t_sq_warm = (time.perf_counter() - t0) * 1000
    print(f"[✓] 4. Squeeze Rubbing (Warm Cache): Retrieved in {t_sq_warm:.4f} ms")
    assert sq_pil_warm is sq_pil
    assert sq_uri_warm == sq_uri

    # 4. Mock Streamlit component rendering to inspect generated HTML/JS
    rendered_frames = []
    def capture_html(html, height=0, **kwargs):
        rendered_frames.append((html, height))

    orig_html = components.html
    components.html = capture_html

    try:
        # Test Dual Viewer Rendering (Mangulam Original vs Lithic Squeeze Rubbing)
        t0 = time.perf_counter()
        render_openseadragon_viewer(
            primary_image=master_pil,
            comparison_image=sq_uri,
            viewer_height=580,
            primary_label="Original Mangulam Tamil-Brahmi Inscription",
            comparison_label="Lithic Squeeze Rubbing (Inverted Relief)",
            image_dimensions=(w, h),
        )
        t_render = (time.perf_counter() - t0) * 1000
        print(f"[✓] 5. Dual Viewer HTML/JS Compiled: in {t_render:.2f} ms")

        assert len(rendered_frames) == 1
        html, height = rendered_frames[0]

        # Verify HTML structural integrity
        checks = [
            ("Primary Viewport Container", 'id="openseadragon-primary"'),
            ("Secondary Viewport Container", 'id="openseadragon-secondary"'),
            ("Primary Pane Container", 'id="pane-primary"'),
            ("Secondary Pane Container", 'id="pane-secondary"'),
            ("Primary Loading Spinner", 'id="p-spinner"'),
            ("Secondary Loading Spinner", 'id="s-spinner"'),
            ("Primary Status Pill", 'id="p-status-pill"'),
            ("Secondary Status Pill", 'id="s-status-pill"'),
            ("Primary Error Banner", 'id="p-error-banner"'),
            ("Secondary Error Banner", 'id="s-error-banner"'),
            ("Sync Viewports Button", 'id="s-sync"'),
            ("Anti-recursion Sync Lock", "isSyncLock"),
            ("CompViewer Pan To", "compViewer.viewport.panTo"),
            ("Watchdog Timeout Timer", "7500"),
            ("State: LOADING", "LOADING"),
            ("State: LOADED", "LOADED"),
            ("State: PROCESSING", "PROCESSING"),
            ("State: FAILED", "FAILED"),
            ("State: IMAGE_NOT_FOUND", "IMAGE_NOT_FOUND"),
            ("State: TIMEOUT", "TIMEOUT"),
            ("Label 1: Mangulam Inscription", "Original Mangulam Tamil-Brahmi Inscription"),
            ("Label 2: Lithic Squeeze Rubbing", "Lithic Squeeze Rubbing (Inverted Relief)"),
        ]

        print("\n--- Structural Markup & State Machine Assertions ---")
        for check_name, pattern in checks:
            assert pattern in html, f"Assertion failed: {pattern} not found in viewer HTML!"
            print(f"  [PASS] {check_name}")

        print("\n--- Multi-Resolution Performance Matrix ---")
        resolutions = [
            ("1024 × 768", (1024, 768)),
            ("1920 × 1080", (1920, 1080)),
            ("4000 × 3000", (4000, 3000)),
        ]
        for r_name, (rw, rh) in resolutions:
            test_arr = np.random.randint(60, 200, (rh, rw, 3), dtype=np.uint8)
            t_start = time.perf_counter()
            test_img = Image.fromarray(test_arr)
            t_decode = (time.perf_counter() - t_start) * 1000

            t0 = time.perf_counter()
            sq_p, sq_u = get_cached_squeeze_rubbing(test_img, max_dim=1600)
            t_proc_cold = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            sq_pw, sq_uw = get_cached_squeeze_rubbing(test_img, max_dim=1600)
            t_proc_warm = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            render_openseadragon_viewer(
                primary_image=test_img,
                comparison_image=sq_u,
                viewer_height=580,
            )
            t_init = (time.perf_counter() - t0) * 1000

            total_tti = t_decode + t_proc_cold + t_init
            print(f"  Resolution [{r_name}]:")
            print(f"    - Decoding: {t_decode:.2f} ms")
            print(f"    - Squeeze Proc (Cold): {t_proc_cold:.2f} ms | (Warm): {t_proc_warm:.4f} ms")
            print(f"    - Viewer Compilation: {t_init:.2f} ms")
            print(f"    - Total Time to Interactive (Cold): {total_tti:.2f} ms | (Warm): {(t_proc_warm + t_init):.2f} ms")
            print(f"    - Payload Size: {len(sq_u)/1024:.1f} KB")

    finally:
        components.html = orig_html

    print("\n================================================================================")
    print("  ALL VERIFICATION CHECKS PASSED (100% SUCCESS)")
    print("================================================================================")

if __name__ == "__main__":
    verify_end_to_end()
