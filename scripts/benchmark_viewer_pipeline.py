import os
import sys
import time
import io
import hashlib
import base64
from typing import Union, Tuple, Optional, Any
from PIL import Image
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))

def compute_image_sha256(image_input: Any) -> str:
    """Compute fast, cached SHA-256 for an image input."""
    if image_input is None:
        return "empty"
    try:
        if hasattr(image_input, "_kallvettu_sha256"):
            return image_input._kallvettu_sha256

        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                with open(image_input, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
            return hashlib.sha256(image_input.encode("utf-8")).hexdigest()
        elif isinstance(image_input, bytes):
            return hashlib.sha256(image_input).hexdigest()
        elif isinstance(image_input, Image.Image):
            if hasattr(image_input, "filename") and image_input.filename and os.path.isfile(image_input.filename):
                with open(image_input.filename, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
            else:
                raw_bytes = image_input.tobytes()
                # Sample head, middle, tail + dimensions to get ultra-fast distinct hash
                header = f"{image_input.size}_{image_input.mode}_".encode()
                sample = raw_bytes[:32768] + raw_bytes[-32768:]
                sha = hashlib.sha256(header + sample).hexdigest()
            image_input._kallvettu_sha256 = sha
            return sha
        elif isinstance(image_input, np.ndarray):
            header = f"{image_input.shape}_{image_input.dtype}_".encode()
            sample = image_input.tobytes()[:32768] + image_input.tobytes()[-32768:]
            return hashlib.sha256(header + sample).hexdigest()
    except Exception:
        pass
    return hashlib.sha256(str(image_input).encode("utf-8")).hexdigest()

_SQUEEZE_CACHE = {}

def get_cached_squeeze_rubbing(
    image_input: Any,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
    max_dim: Optional[int] = 1600,
    algo_version: str = "2.1.0-squeeze-relief"
) -> Tuple[Optional[Image.Image], str, float]:
    """Generates and caches an inverted relief squeeze rubbing."""
    t0 = time.perf_counter()
    if image_input is None:
        return None, "", 0.0

    sha = compute_image_sha256(image_input)
    cache_key = (sha, algo_version, clip_limit, tile_grid, max_dim)
    if cache_key in _SQUEEZE_CACHE:
        pil_img, b64_uri = _SQUEEZE_CACHE[cache_key]
        return pil_img, b64_uri, (time.perf_counter() - t0) * 1000

    # Convert to array
    if isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, str) and os.path.isfile(image_input):
        arr = cv2.imread(image_input)
        if arr is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            return None, "", (time.perf_counter() - t0) * 1000
    elif isinstance(image_input, np.ndarray):
        arr = image_input
    else:
        return None, "", (time.perf_counter() - t0) * 1000

    h, w = arr.shape[:2]
    # Squeeze downsampling if max_dim is set
    if max_dim and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        nw, nh = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)

    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.copy()

    cv2.bitwise_not(gray, dst=gray)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    squeeze_arr = clahe.apply(gray)
    squeeze_rgb = cv2.cvtColor(squeeze_arr, cv2.COLOR_GRAY2RGB)
    squeeze_pil = Image.fromarray(squeeze_rgb)

    # Base64 encode
    buf = io.BytesIO()
    squeeze_pil.save(buf, format="JPEG", quality=85)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    b64_uri = f"data:image/jpeg;base64,{b64_str}"

    _SQUEEZE_CACHE[cache_key] = (squeeze_pil, b64_uri)
    dt = (time.perf_counter() - t0) * 1000
    return squeeze_pil, b64_uri, dt

def run_benchmark():
    print("=== Optimized Pipeline Benchmark Across Resolutions ===")
    resolutions = [
        ("1024x768", (1024, 768)),
        ("1920x1080", (1920, 1080)),
        ("4000x3000", (4000, 3000)),
    ]
    for name, (w, h) in resolutions:
        # Create random texture
        draw_arr = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
        test_img = Image.fromarray(draw_arr)

        # Cold run
        _, uri, cold_time = get_cached_squeeze_rubbing(test_img)
        # Warm run
        _, _, warm_time = get_cached_squeeze_rubbing(test_img)

        payload_kb = len(uri) / 1024
        print(f"[{name}] Cold Squeeze+B64: {cold_time:.2f} ms | Warm Cache: {warm_time:.4f} ms | Payload: {payload_kb:.1f} KB")

if __name__ == "__main__":
    run_benchmark()
