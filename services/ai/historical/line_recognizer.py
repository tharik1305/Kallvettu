"""
CPU-First Lightweight Historical Text-Line Recognition (Phase 3).

Localizes inscription text lines and proposes candidate glyphs using purely
lightweight OpenCV image analysis (horizontal projection profiles + connected
components) paired with Kallvettu's HOG + SVM glyph classifier.

Scientific & Engineering Constraints:
- Strict CPU-only execution (OpenCV + NumPy + HOG/SVM). Zero deep CNNs or CRNNs.
- Reuses aspect-preserving 1600 px dimension clamping via restoration_engine.
- Preserves original image coordinates for every line crop and character box.
- Marks merged ligatures, heavily overlapping components, low/high-density blobs,
  and out-of-vocabulary rejected glyphs as `[?]` (`is_unresolved=True`) instead
  of guessing or fabricating characters.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from restoration_engine import validate_and_load_image
from services.ai.common.base_provider import BoundingBox
from services.ai.historical.glyph_classifier import (
    GlyphCandidate,
    GlyphPredictionResult,
    HOGSVMGlyphClassifier,
)
from services.research.historical_glyph_dataset import normalize_tamil_unicode


@dataclass
class LineRecognizerConfig:
    """Configuration for projection-profile line splitting and CCA character proposals."""

    max_processing_dim: int = 1600
    invert_polarity: Optional[bool] = None  # None = auto-detect light vs dark stone
    use_clahe: bool = True

    # Line localization parameters
    row_smoothing_kernel: int = 3
    min_line_height_px: int = 8
    min_line_width_px: int = 8
    min_line_ink_ratio: float = 0.015
    line_merge_gap_factor: float = 0.16     # Merge line bands closer than factor * median_h
    line_pad_px: int = 3

    # Connected component & character proposal parameters
    min_char_area_px: int = 12
    min_char_dim_px: int = 4
    merge_vertical_diacritics: bool = True
    diacritic_x_overlap_threshold: float = 0.45

    # Overlap / merged-ligature unresolved heuristics
    max_char_aspect_ratio: float = 2.25     # width / height beyond which ligature/touching is likely
    max_width_to_line_height: float = 2.10  # width / line_height threshold
    heavy_x_overlap_threshold: float = 0.45
    min_ink_density: float = 0.03
    max_ink_density: float = 0.92
    char_crop_pad_px: int = 2

    # Sequence / word reconstruction
    space_gap_width_factor: float = 0.72    # gap > factor * median_char_w inserts space
    top_k_candidates: int = 3


@dataclass
class CharacterProposal:
    """Single localized character proposal inside a text line."""

    char_index: int
    # Pixel box in working image coordinates (x1, y1, x2, y2)
    pixel_box_working: Tuple[int, int, int, int]
    # Pixel box mapped back to original input image coordinates (x1, y1, x2, y2)
    pixel_box_original: Tuple[int, int, int, int]
    # Normalized [0..1] BoundingBox compatible with K-IEFE
    bounding_box: BoundingBox
    character: str
    is_unresolved: bool = False
    is_uncertain: bool = False
    unresolved_reason: str = ""
    confidence: Optional[float] = None      # Only populated when calibrated probabilities exist
    raw_decision_score: float = 0.0
    top_candidates: List[GlyphCandidate] = field(default_factory=list)
    ink_density: float = 0.0
    aspect_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char_index": self.char_index,
            "pixel_box_working": list(self.pixel_box_working),
            "pixel_box_original": list(self.pixel_box_original),
            "bounding_box": self.bounding_box.to_dict(),
            "character": self.character,
            "is_unresolved": self.is_unresolved,
            "is_uncertain": self.is_uncertain,
            "unresolved_reason": self.unresolved_reason,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "raw_decision_score": round(self.raw_decision_score, 4),
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "ink_density": round(self.ink_density, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
        }


@dataclass
class LineSegment:
    """Recognized text line preserving spatial coordinates, crops, and character proposals."""

    line_index: int
    pixel_box_working: Tuple[int, int, int, int]
    pixel_box_original: Tuple[int, int, int, int]
    bounding_box: BoundingBox
    raw_text: str
    characters: List[CharacterProposal] = field(default_factory=list)
    unresolved_regions: List[CharacterProposal] = field(default_factory=list)
    line_crop: Optional[np.ndarray] = field(default=None, repr=False)
    confidence: Optional[float] = None      # Mean calibrated confidence if all resolved chars calibrated
    has_unresolved: bool = False

    def to_dict(self, include_crop: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "line_index": self.line_index,
            "pixel_box_working": list(self.pixel_box_working),
            "pixel_box_original": list(self.pixel_box_original),
            "bounding_box": self.bounding_box.to_dict(),
            "raw_text": self.raw_text,
            "characters": [c.to_dict() for c in self.characters],
            "unresolved_count": len(self.unresolved_regions),
            "has_unresolved": self.has_unresolved,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }
        if include_crop and self.line_crop is not None:
            data["line_crop_shape"] = list(self.line_crop.shape)
        return data


@dataclass
class LineRecognitionResult:
    """Complete output of CPU-based lightweight text-line recognition."""

    text: str
    lines: List[LineSegment] = field(default_factory=list)
    script: str = "Tamil-Brahmi"
    original_dimensions: Tuple[int, int] = (0, 0)  # (width, height)
    working_dimensions: Tuple[int, int] = (0, 0)   # (width, height)
    total_characters: int = 0
    resolved_characters: int = 0
    unresolved_characters: int = 0
    uncertain_characters: int = 0
    confidence: Optional[float] = None             # Strictly None unless calibrated
    latency_seconds: float = 0.0
    polarity_inverted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "script": self.script,
            "original_dimensions": list(self.original_dimensions),
            "working_dimensions": list(self.working_dimensions),
            "line_count": len(self.lines),
            "total_characters": self.total_characters,
            "resolved_characters": self.resolved_characters,
            "unresolved_characters": self.unresolved_characters,
            "uncertain_characters": self.uncertain_characters,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "latency_seconds": round(self.latency_seconds, 4),
            "polarity_inverted": self.polarity_inverted,
            "lines": [line.to_dict() for line in self.lines],
            "metadata": self.metadata,
        }


def _load_working_grayscale(
    image: Union[bytes, Image.Image, np.ndarray, str],
    max_dim: int = 1600,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], float, float]:
    """
    Loads and aspect-clamps an input image without modifying original bytes,
    returning `(gray_working_ndarray, (orig_w, orig_h), (work_w, work_h), scale_x, scale_y)`.
    """
    if isinstance(image, np.ndarray):
        orig_h, orig_w = image.shape[:2]
        if image.ndim == 3:
            if image.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
            else:
                pil_img = Image.fromarray(image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8))
        pristine, working, _ = validate_and_load_image(
            pil_img, min_dimension=12, max_processing_dim=max_dim
        )
    else:
        pristine, working, _ = validate_and_load_image(
            image, min_dimension=12, max_processing_dim=max_dim
        )

    orig_w, orig_h = pristine.size
    work_w, work_h = working.size
    gray = np.array(working.convert("L"), dtype=np.uint8)
    scale_x = orig_w / float(max(1, work_w))
    scale_y = orig_h / float(max(1, work_h))
    return gray, (orig_w, orig_h), (work_w, work_h), scale_x, scale_y


def _binarize_inscription(
    gray: np.ndarray,
    use_clahe: bool = True,
    invert_polarity: Optional[bool] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Binarizes a grayscale stone inscription crop into foreground=255, background=0.
    Auto-detects polarity using outer border pixel luminance if `invert_polarity` is None.
    """
    proc = gray
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = clahe.apply(proc)

    blurred = cv2.GaussianBlur(proc, (3, 3), 0)

    if invert_polarity is None:
        h, w = blurred.shape
        border_vals = np.concatenate([
            blurred[0, :],
            blurred[-1, :],
            blurred[:, 0],
            blurred[:, -1],
        ])
        # Light background (> 127) -> dark glyphs -> invert binary mask
        invert_flag = bool(np.median(border_vals) > 127)
    else:
        invert_flag = bool(invert_polarity)

    thresh_flag = (cv2.THRESH_BINARY_INV if invert_flag else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, binary = cv2.threshold(blurred, 0, 255, thresh_flag)

    # Secondary safety check: if foreground occupies > 65% of image when auto-detecting,
    # background and foreground were likely swapped.
    if invert_polarity is None:
        fg_ratio = float(np.count_nonzero(binary)) / float(max(1, binary.size))
        if fg_ratio > 0.65:
            binary = cv2.bitwise_not(binary)
            invert_flag = not invert_flag

    return binary, invert_flag


class HistoricalLineRecognizer:
    """
    Lightweight CPU-only text-line segmenter and character-sequence recognizer.

    Pipeline:
      1. Aspect-preserving load (<= 1600px) & polarity-aware binarization.
      2. Horizontal projection profile localization + close-band diacritic merging.
      3. Connected-component analysis (CCA) per line crop.
      4. Left-to-right ordering with optional vertical diacritic grouping.
      5. Overlap / merged-ligature detection -> unresolved `[?]` marking.
      6. Per-glyph classification using `HOGSVMGlyphClassifier`.
      7. Coordinate mapping back to original stone photograph pixels & normalized boxes.
    """

    def __init__(
        self,
        classifier: Optional[HOGSVMGlyphClassifier] = None,
        config: Optional[LineRecognizerConfig] = None,
    ):
        self.classifier = classifier
        self.config = config or LineRecognizerConfig()

    def localize_lines(self, binary_fg: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects horizontal text lines from a foreground=255 binary image using
        a smoothed horizontal projection profile.

        Returns list of `(x1, y1, x2, y2)` pixel boxes sorted top-to-bottom.
        """
        h, w = binary_fg.shape
        if h == 0 or w == 0:
            return []

        proj = np.sum(binary_fg > 0, axis=1).astype(np.float32)
        if np.max(proj) <= 0:
            return []

        k = max(1, int(self.config.row_smoothing_kernel))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            proj_smooth = np.convolve(proj, kernel, mode="same")
        else:
            proj_smooth = proj

        # Row is active if it contains meaningful foreground pixels
        row_thresh = max(1.5, 0.03 * float(np.max(proj_smooth)))
        active = proj_smooth >= row_thresh

        raw_bands: List[Tuple[int, int]] = []
        in_band = False
        band_start = 0
        for y in range(h):
            if active[y] and not in_band:
                in_band = True
                band_start = y
            elif not active[y] and in_band:
                in_band = False
                raw_bands.append((band_start, y))
        if in_band:
            raw_bands.append((band_start, h))

        if not raw_bands:
            return []

        # Merge narrowly separated bands (e.g. pulli / upper vowel loops split by valley)
        heights = [y2 - y1 for y1, y2 in raw_bands if (y2 - y1) >= 3]
        median_h = float(np.median(heights)) if heights else float(self.config.min_line_height_px)
        base_max_gap = max(2, int(round(self.config.line_merge_gap_factor * median_h)))

        merged_bands: List[Tuple[int, int]] = [raw_bands[0]]
        for y1, y2 in raw_bands[1:]:
            prev_y1, prev_y2 = merged_bands[-1]
            gap = y1 - prev_y2
            prev_h = prev_y2 - prev_y1
            curr_h = y2 - y1
            taller_h = max(prev_h, curr_h)
            shorter_h = min(prev_h, curr_h)

            # Merge if gap is small OR if one band is a small diacritic/pulli band near a main line
            is_small_gap = gap <= base_max_gap
            is_diacritic_band = (shorter_h <= 0.50 * taller_h) and (gap <= 0.45 * taller_h)

            if is_small_gap or is_diacritic_band:
                merged_bands[-1] = (prev_y1, y2)
            else:
                merged_bands.append((y1, y2))

        line_boxes: List[Tuple[int, int, int, int]] = []
        pad = max(0, int(self.config.line_pad_px))
        for y1, y2 in merged_bands:
            band_h = y2 - y1
            if band_h < self.config.min_line_height_px:
                continue

            band_slice = binary_fg[y1:y2, :]
            col_proj = np.sum(band_slice > 0, axis=0)
            active_cols = np.where(col_proj > 0)[0]
            if active_cols.size == 0:
                continue

            x1 = int(active_cols[0])
            x2 = int(active_cols[-1]) + 1
            if (x2 - x1) < self.config.min_line_width_px:
                continue

            roi = binary_fg[y1:y2, x1:x2]
            ink_ratio = float(np.count_nonzero(roi)) / float(max(1, roi.size))
            if ink_ratio < self.config.min_line_ink_ratio:
                continue

            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            line_boxes.append((px1, py1, px2, py2))

        line_boxes.sort(key=lambda b: (b[1], b[0]))
        return line_boxes

    def propose_character_boxes(
        self,
        line_binary_fg: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Runs Connected-Component Analysis (CCA) on a cropped binary text line and
        returns left-to-right character region proposals with overlap & ligature diagnostics.

        Each item is a dict containing:
          - `box`: (x1, y1, x2, y2) relative to line_binary_fg
          - `ink_density`: float
          - `aspect_ratio`: float
          - `pre_unresolved`: bool
          - `pre_unresolved_reason`: str
        """
        lh, lw = line_binary_fg.shape
        if lh == 0 or lw == 0:
            return []

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            (line_binary_fg > 0).astype(np.uint8),
            connectivity=8,
        )

        raw_boxes: List[Tuple[int, int, int, int, int]] = []
        min_area = max(self.config.min_char_area_px, int(0.004 * lh * lh))
        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            cw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            ch = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if cw < self.config.min_char_dim_px and ch < self.config.min_char_dim_px:
                continue
            if area < min_area:
                continue
            raw_boxes.append((x, y, x + cw, y + ch, area))

        if not raw_boxes:
            return []

        # Sort left-to-right before optional vertical diacritic merging
        raw_boxes.sort(key=lambda b: (b[0], b[1]))

        merged_boxes: List[Tuple[int, int, int, int]] = []
        if self.config.merge_vertical_diacritics:
            for x1, y1, x2, y2, _ in raw_boxes:
                cw = x2 - x1
                merged = False
                for idx in range(len(merged_boxes) - 1, -1, -1):
                    px1, py1, px2, py2 = merged_boxes[idx]
                    pw = px2 - px1
                    overlap_x = min(x2, px2) - max(x1, px1)
                    if overlap_x <= 0:
                        continue
                    overlap_ratio = overlap_x / float(max(1, min(cw, pw)))
                    # Merge only if smaller/stacked vertically and combined width stays bounded
                    comb_w = max(x2, px2) - min(x1, px1)
                    if (
                        overlap_ratio >= self.config.diacritic_x_overlap_threshold
                        and comb_w <= self.config.max_width_to_line_height * max(lh, 1) * 0.9
                    ):
                        merged_boxes[idx] = (
                            min(x1, px1),
                            min(y1, py1),
                            max(x2, px2),
                            max(y2, py2),
                        )
                        merged = True
                        break
                if not merged:
                    merged_boxes.append((x1, y1, x2, y2))
        else:
            merged_boxes = [(b[0], b[1], b[2], b[3]) for b in raw_boxes]

        # Re-sort strictly left-to-right
        merged_boxes.sort(key=lambda b: (b[0], b[1]))

        proposals: List[Dict[str, Any]] = []
        for idx, (x1, y1, x2, y2) in enumerate(merged_boxes):
            cw = max(1, x2 - x1)
            ch = max(1, y2 - y1)
            aspect_ratio = cw / float(ch)
            width_vs_line_h = cw / float(max(1, lh))

            roi = line_binary_fg[y1:y2, x1:x2]
            ink_density = float(np.count_nonzero(roi)) / float(max(1, roi.size))

            reasons: List[str] = []
            if (
                aspect_ratio > self.config.max_char_aspect_ratio
                or width_vs_line_h > self.config.max_width_to_line_height
            ):
                reasons.append("merged_ligature_or_touching_glyphs")
            if ink_density < self.config.min_ink_density:
                reasons.append("low_ink_density_noise")
            elif ink_density > self.config.max_ink_density:
                reasons.append("solid_blob_artifact")

            proposals.append({
                "box": (x1, y1, x2, y2),
                "ink_density": ink_density,
                "aspect_ratio": aspect_ratio,
                "pre_unresolved": len(reasons) > 0,
                "pre_unresolved_reason": "; ".join(reasons),
            })

        # Second pass: detect unresolved heavy horizontal overlap between adjacent distinct boxes
        for i in range(len(proposals) - 1):
            ax1, _, ax2, _ = proposals[i]["box"]
            bx1, _, bx2, _ = proposals[i + 1]["box"]
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            if overlap_x > 0:
                min_w = max(1, min(ax2 - ax1, bx2 - bx1))
                if (overlap_x / float(min_w)) > self.config.heavy_x_overlap_threshold:
                    for j in (i, i + 1):
                        proposals[j]["pre_unresolved"] = True
                        existing = proposals[j]["pre_unresolved_reason"]
                        tag = "heavy_component_overlap"
                        proposals[j]["pre_unresolved_reason"] = (
                            f"{existing}; {tag}" if existing else tag
                        )

        return proposals

    def recognize_lines(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        classifier: Optional[HOGSVMGlyphClassifier] = None,
        script_hint: Optional[str] = None,
    ) -> LineRecognitionResult:
        """
        Executes full CPU-only line segmentation, character proposal extraction,
        HOG+SVM per-glyph recognition, and sequence reconstruction.
        """
        t0 = time.perf_counter()
        active_clf = classifier or self.classifier

        gray_work, (orig_w, orig_h), (work_w, work_h), scale_x, scale_y = _load_working_grayscale(
            image, max_dim=self.config.max_processing_dim
        )
        binary_fg, polarity_inverted = _binarize_inscription(
            gray_work,
            use_clahe=self.config.use_clahe,
            invert_polarity=self.config.invert_polarity,
        )

        line_boxes = self.localize_lines(binary_fg)
        target_script = script_hint or (active_clf.script if active_clf else "Tamil-Brahmi")

        recognized_lines: List[LineSegment] = []
        total_chars = 0
        resolved_chars = 0
        unresolved_chars = 0
        uncertain_chars = 0
        all_calibrated_confs: List[float] = []

        pad_c = max(0, int(self.config.char_crop_pad_px))

        for line_idx, (lx1, ly1, lx2, ly2) in enumerate(line_boxes):
            line_bin = binary_fg[ly1:ly2, lx1:lx2]
            line_gray = gray_work[ly1:ly2, lx1:lx2]
            char_props_raw = self.propose_character_boxes(line_bin)

            char_proposals: List[CharacterProposal] = []
            unresolved_list: List[CharacterProposal] = []
            line_cal_confs: List[float] = []

            for c_idx, prop in enumerate(char_props_raw):
                cx1, cy1, cx2, cy2 = prop["box"]
                gx1, gy1 = lx1 + cx1, ly1 + cy1
                gx2, gy2 = lx1 + cx2, ly1 + cy2

                orig_box = (
                    int(round(gx1 * scale_x)),
                    int(round(gy1 * scale_y)),
                    int(round(gx2 * scale_x)),
                    int(round(gy2 * scale_y)),
                )
                norm_bbox = BoundingBox(
                    ymin=round(gy1 / float(max(1, work_h)), 6),
                    xmin=round(gx1 / float(max(1, work_w)), 6),
                    ymax=round(gy2 / float(max(1, work_h)), 6),
                    xmax=round(gx2 / float(max(1, work_w)), 6),
                )

                # Extract padded glyph crop from grayscale working image
                px1 = max(0, gx1 - pad_c)
                py1 = max(0, gy1 - pad_c)
                px2 = min(work_w, gx2 + pad_c)
                py2 = min(work_h, gy2 + pad_c)
                glyph_crop = gray_work[py1:py2, px1:px2]

                total_chars += 1

                if prop["pre_unresolved"] or active_clf is None or not active_clf.is_fitted:
                    reason = prop["pre_unresolved_reason"] or (
                        "classifier_not_loaded" if (active_clf is None or not active_clf.is_fitted) else ""
                    )
                    cp = CharacterProposal(
                        char_index=c_idx,
                        pixel_box_working=(gx1, gy1, gx2, gy2),
                        pixel_box_original=orig_box,
                        bounding_box=norm_bbox,
                        character="[?]",
                        is_unresolved=True,
                        is_uncertain=True,
                        unresolved_reason=reason,
                        confidence=None,
                        raw_decision_score=0.0,
                        top_candidates=[],
                        ink_density=prop["ink_density"],
                        aspect_ratio=prop["aspect_ratio"],
                    )
                    unresolved_chars += 1
                    uncertain_chars += 1
                    unresolved_list.append(cp)
                    char_proposals.append(cp)
                    continue

                pred: GlyphPredictionResult = active_clf.predict_glyph(
                    glyph_crop, top_k=self.config.top_k_candidates
                )
                if pred.is_rejected:
                    cp = CharacterProposal(
                        char_index=c_idx,
                        pixel_box_working=(gx1, gy1, gx2, gy2),
                        pixel_box_original=orig_box,
                        bounding_box=norm_bbox,
                        character="[?]",
                        is_unresolved=True,
                        is_uncertain=True,
                        unresolved_reason=pred.rejection_reason or "classifier_rejected",
                        confidence=pred.confidence,
                        raw_decision_score=pred.raw_decision_score,
                        top_candidates=pred.top_candidates,
                        ink_density=prop["ink_density"],
                        aspect_ratio=prop["aspect_ratio"],
                    )
                    unresolved_chars += 1
                    uncertain_chars += 1
                    unresolved_list.append(cp)
                else:
                    char_norm = normalize_tamil_unicode(pred.predicted_character)
                    cp = CharacterProposal(
                        char_index=c_idx,
                        pixel_box_working=(gx1, gy1, gx2, gy2),
                        pixel_box_original=orig_box,
                        bounding_box=norm_bbox,
                        character=char_norm,
                        is_unresolved=False,
                        is_uncertain=pred.is_uncertain,
                        unresolved_reason="",
                        confidence=pred.confidence,
                        raw_decision_score=pred.raw_decision_score,
                        top_candidates=pred.top_candidates,
                        ink_density=prop["ink_density"],
                        aspect_ratio=prop["aspect_ratio"],
                    )
                    resolved_chars += 1
                    if pred.is_uncertain:
                        uncertain_chars += 1
                    if pred.confidence is not None:
                        line_cal_confs.append(pred.confidence)
                        all_calibrated_confs.append(pred.confidence)

                char_proposals.append(cp)

            # Reconstruct line string with gap-based word spacing
            line_text = self._reconstruct_line_text(char_proposals)
            line_conf = (
                float(np.mean(line_cal_confs))
                if line_cal_confs and len(line_cal_confs) == len(char_proposals)
                else None
            )

            line_orig_box = (
                int(round(lx1 * scale_x)),
                int(round(ly1 * scale_y)),
                int(round(lx2 * scale_x)),
                int(round(ly2 * scale_y)),
            )
            line_norm_bbox = BoundingBox(
                ymin=round(ly1 / float(max(1, work_h)), 6),
                xmin=round(lx1 / float(max(1, work_w)), 6),
                ymax=round(ly2 / float(max(1, work_h)), 6),
                xmax=round(lx2 / float(max(1, work_w)), 6),
                confidence=line_conf,
                text=line_text,
            )

            recognized_lines.append(
                LineSegment(
                    line_index=line_idx,
                    pixel_box_working=(lx1, ly1, lx2, ly2),
                    pixel_box_original=line_orig_box,
                    bounding_box=line_norm_bbox,
                    raw_text=line_text,
                    characters=char_proposals,
                    unresolved_regions=unresolved_list,
                    line_crop=line_gray,
                    confidence=line_conf,
                    has_unresolved=len(unresolved_list) > 0,
                )
            )

        full_text = "\n".join(seg.raw_text for seg in recognized_lines if seg.raw_text)
        overall_conf = (
            float(np.mean(all_calibrated_confs))
            if all_calibrated_confs and len(all_calibrated_confs) == total_chars
            else None
        )
        latency = time.perf_counter() - t0

        return LineRecognitionResult(
            text=full_text,
            lines=recognized_lines,
            script=target_script,
            original_dimensions=(orig_w, orig_h),
            working_dimensions=(work_w, work_h),
            total_characters=total_chars,
            resolved_characters=resolved_chars,
            unresolved_characters=unresolved_chars,
            uncertain_characters=uncertain_chars,
            confidence=overall_conf,
            latency_seconds=latency,
            polarity_inverted=polarity_inverted,
            metadata={
                "max_processing_dim": self.config.max_processing_dim,
                "classifier_loaded": bool(active_clf and active_clf.is_fitted),
                "classifier_calibrated": bool(active_clf and active_clf.calibrate_probabilities),
                "line_count": len(recognized_lines),
            },
        )

    def _reconstruct_line_text(self, proposals: List[CharacterProposal]) -> str:
        """
        Reconstructs character sequence left-to-right, inserting spaces between words
        when horizontal gap exceeds `space_gap_width_factor * median_char_width`.
        """
        if not proposals:
            return ""
        if len(proposals) == 1:
            return proposals[0].character

        widths = [
            max(1, p.pixel_box_working[2] - p.pixel_box_working[0])
            for p in proposals
        ]
        median_w = float(np.median(widths))
        space_thresh = max(3.0, self.config.space_gap_width_factor * median_w)

        parts: List[str] = [proposals[0].character]
        for idx in range(1, len(proposals)):
            prev_x2 = proposals[idx - 1].pixel_box_working[2]
            curr_x1 = proposals[idx].pixel_box_working[0]
            gap = curr_x1 - prev_x2
            if gap > space_thresh:
                parts.append(" ")
            parts.append(proposals[idx].character)

        return "".join(parts)
