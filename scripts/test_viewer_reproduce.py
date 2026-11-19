import os
import sys
import time
import io
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

def reproduce():
    img_path = os.path.join("assets", "samples", "mangulam.jpg")
    print(f"Testing with: {img_path}")
    t0 = time.perf_counter()
    master_pil = Image.open(img_path)
    w, h = master_pil.size
    t_load = (time.perf_counter() - t0) * 1000
    print(f"1. Image retrieval & open: {t_load:.2f} ms ({w}x{h} px)")

    # Squeeze generation
    t0 = time.perf_counter()
    arr_rgb = np.array(master_pil.convert("RGB"))
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    clahe_engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    squeeze_arr = clahe_engine.apply(inverted)
    squeeze_pil = Image.fromarray(cv2.cvtColor(squeeze_arr, cv2.COLOR_GRAY2RGB))
    t_squeeze = (time.perf_counter() - t0) * 1000
    print(f"2. Squeeze-rubbing processing (CLAHE+inverted): {t_squeeze:.2f} ms")

    # Base64 encoding
    t0 = time.perf_counter()
    from services.iiif.viewer_component import image_to_base64_data_uri
    p_uri = image_to_base64_data_uri(master_pil)
    c_uri = image_to_base64_data_uri(squeeze_pil)
    t_b64 = (time.perf_counter() - t0) * 1000
    print(f"3. Base64 conversion time: {t_b64:.2f} ms")
    print(f"   Primary base64 length: {len(p_uri):,} chars ({len(p_uri)/1024/1024:.2f} MB)")
    print(f"   Comp base64 length: {len(c_uri):,} chars ({len(c_uri)/1024/1024:.2f} MB)")

    # Now let's test with higher resolution images (1920x1080 and 4000x3000)
    for res_name, (tw, th) in [("1024x768", (1024, 768)), ("1920x1080", (1920, 1080)), ("4000x3000", (4000, 3000))]:
        test_img = master_pil.resize((tw, th), Image.Resampling.BILINEAR)
        t0 = time.perf_counter()
        arr_rgb = np.array(test_img.convert("RGB"))
        gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        inverted = cv2.bitwise_not(gray)
        clahe_engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        squeeze_arr = clahe_engine.apply(inverted)
        squeeze_pil = Image.fromarray(cv2.cvtColor(squeeze_arr, cv2.COLOR_GRAY2RGB))
        t_sq = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        p_b64 = image_to_base64_data_uri(test_img)
        c_b64 = image_to_base64_data_uri(squeeze_pil)
        t_b = (time.perf_counter() - t0) * 1000
        print(f"Benchmark [{res_name}]: Squeeze={t_sq:.2f} ms | Base64={t_b:.2f} ms | Total Payload Size={(len(p_b64)+len(c_b64))/1024/1024:.2f} MB")

if __name__ == "__main__":
    reproduce()
