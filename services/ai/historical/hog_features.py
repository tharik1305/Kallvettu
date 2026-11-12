"""
HOG (Histogram of Oriented Gradients) Feature Extraction for Historical Epigraphical Glyphs.

Uses OpenCV (cv2) for CPU-efficient image preprocessing and gradient feature computation:
- Validates input dimensions (min 10x10)
- Converts to normalized single-channel grayscale
- Resizes to fixed canonical window (default 64x64) with edge-preserving interpolation
- Applies adaptive contrast (CLAHE, Otsu, or normalized raw)
- Computes OpenCV HOGDescriptor across 9 orientation bins, 8x8 cells, 16x16 blocks
"""

import io
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HOGConfig:
    """Configuration for OpenCV HOGDescriptor and glyph preprocessing."""
    win_size: Tuple[int, int] = (64, 64)       # Fixed normalization window (width, height)
    block_size: Tuple[int, int] = (16, 16)     # Block dimensions in pixels
    block_stride: Tuple[int, int] = (8, 8)     # Block stride in pixels
    cell_size: Tuple[int, int] = (8, 8)        # Cell dimensions in pixels
    nbins: int = 9                             # Gradient histogram bins
    preprocess_mode: str = "clahe"             # 'clahe', 'otsu', 'normalized_raw'
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    @property
    def feature_dim(self) -> int:
        """Calculates expected 1D feature vector length."""
        blocks_x = (self.win_size[0] - self.block_size[0]) // self.block_stride[0] + 1
        blocks_y = (self.win_size[1] - self.block_size[1]) // self.block_stride[1] + 1
        cells_per_block = (self.block_size[0] // self.cell_size[0]) * (
            self.block_size[1] // self.cell_size[1]
        )
        return blocks_x * blocks_y * cells_per_block * self.nbins

    def to_dict(self):
        return asdict(self)


DEFAULT_HOG_CONFIG = HOGConfig()


def prepare_glyph_image(
    image: Union[np.ndarray, Image.Image, bytes, str],
    target_size: Tuple[int, int] = (64, 64),
    preprocess_mode: str = "clahe",
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Validates, grayscales, normalizes, and resizes a candidate character glyph image.
    Returns 8-bit single-channel uint8 numpy array of shape (height, width).
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    # 1. Ingestion into uint8 numpy array
    if isinstance(image, str):
        arr = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to load image from path: {image}")
    elif isinstance(image, (bytes, bytearray)):
        if len(image) == 0:
            raise ValueError("Image byte buffer is empty.")
        nparr = np.frombuffer(image, np.uint8)
        arr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Failed to decode image bytes with OpenCV.")
    elif isinstance(image, Image.Image):
        arr = np.array(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    if arr.size == 0 or len(arr.shape) < 2:
        raise ValueError("Input image has zero size or invalid dimensions.")

    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        raise ValueError(f"Glyph image dimensions ({w}x{h}) are too small (minimum 8x8 pixels).")

    # 2. Grayscale conversion
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            # RGBA / BGRA -> Gray
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY if isinstance(image, str) else cv2.COLOR_RGBA2GRAY)
        elif arr.shape[2] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY if isinstance(image, str) else cv2.COLOR_RGB2GRAY)
        else:
            gray = arr[:, :, 0]
    else:
        gray = arr

    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 3. Canonical sizing
    target_w, target_h = target_size
    if (w, h) != (target_w, target_h):
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA if (w > target_w and h > target_h) else cv2.INTER_LINEAR)

    # 4. Preprocessing enhancement
    mode = preprocess_mode.lower()
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_grid_size)
        gray = clahe.apply(gray)
    elif mode == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "normalized_raw":
        # Min-max stretching to full [0..255] range
        min_v, max_v = int(gray.min()), int(gray.max())
        if max_v > min_v:
            gray = ((gray.astype(float) - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)

    return gray


def extract_hog_features(
    image: Union[np.ndarray, Image.Image, bytes, str],
    config: Optional[HOGConfig] = None,
) -> np.ndarray:
    """
    Extracts a 1D float32 HOG feature vector for a character crop.
    Guarantees fixed feature length matching config.feature_dim (default 1764).
    """
    cfg = config or DEFAULT_HOG_CONFIG
    gray = prepare_glyph_image(
        image,
        target_size=cfg.win_size,
        preprocess_mode=cfg.preprocess_mode,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_grid_size=cfg.clahe_grid_size,
    )

    hog = cv2.HOGDescriptor(
        _winSize=cfg.win_size,
        _blockSize=cfg.block_size,
        _blockStride=cfg.block_stride,
        _cellSize=cfg.cell_size,
        _nbins=cfg.nbins,
    )

    # OpenCV HOG compute expects (H, W) uint8 image of exact size cfg.win_size
    features = hog.compute(gray)
    if features is None:
        raise RuntimeError("cv2.HOGDescriptor.compute returned None.")

    vec = features.flatten().astype(np.float32)
    if len(vec) != cfg.feature_dim:
        raise ValueError(
            f"HOG feature length mismatch: got {len(vec)}, expected {cfg.feature_dim}."
        )

    # L2-unit normalization for robust linear/RBF SVM boundary separation
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm

    return vec

