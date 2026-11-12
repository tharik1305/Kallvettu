"""
Tests for CPU-First Lightweight Historical Text-Line Recognition (Phase 3).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from services.ai.historical.hog_features import extract_hog_features
from services.ai.historical.glyph_classifier import HOGSVMGlyphClassifier
from services.ai.historical.line_recognizer import (
    HistoricalLineRecognizer,
    LineRecognizerConfig,
)
from services.ai.historical.model_registry import HistoricalModelRegistry
from services.ai.historical.inference import HistoricalHOGSVMProvider


def _draw_glyph_circle(img: np.ndarray, x: int, y: int, w: int = 28, h: int = 28, color: int = 25):
    cx, cy = x + w // 2, y + h // 2
    r = min(w, h) // 3
    cv2.circle(img, (cx, cy), r, color, 3)


def _draw_glyph_cross(img: np.ndarray, x: int, y: int, w: int = 28, h: int = 28, color: int = 25):
    cv2.line(img, (x + 4, y + h // 2), (x + w - 4, y + h // 2), color, 3)
    cv2.line(img, (x + w // 2, y + 4), (x + w // 2, y + h - 4), color, 3)


def _draw_glyph_triangle(img: np.ndarray, x: int, y: int, w: int = 28, h: int = 28, color: int = 25):
    pts = np.array([
        [x + w // 2, y + 4],
        [x + 4, y + h - 4],
        [x + w - 4, y + h - 4],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)


def _train_synthetic_classifier(calibrate: bool = False) -> HOGSVMGlyphClassifier:
    X_list = []
    y_list = []
    rng = np.random.default_rng(42)

    for _ in range(8):
        for label, drawer in [("க", _draw_glyph_circle), ("ம", _draw_glyph_cross), ("ப", _draw_glyph_triangle)]:
            canvas = np.full((64, 64), 235, dtype=np.uint8)
            jitter_x = int(rng.integers(-2, 3))
            jitter_y = int(rng.integers(-2, 3))
            drawer(canvas, 14 + jitter_x, 14 + jitter_y, 36, 36, color=20)
            X_list.append(extract_hog_features(canvas))
            y_list.append(label)

    clf = HOGSVMGlyphClassifier(
        script="Tamil-Brahmi",
        kernel="linear",
        calibrate_probabilities=calibrate,
        rejection_threshold=-1.0,
    )
    clf.fit(np.vstack(X_list), y_list)
    return clf


def test_multiline_projection_and_word_spacing():
    clf = _train_synthetic_classifier()
    # Create clean 2-line synthetic stone inscription (light background, dark glyphs)
    img = np.full((180, 420), 235, dtype=np.uint8)

    # Line 1: two words separated by wider gap: "கம"   "ப"
    _draw_glyph_circle(img, 30, 28, 30, 30)
    _draw_glyph_cross(img, 70, 28, 30, 30)
    _draw_glyph_triangle(img, 145, 28, 30, 30)

    # Line 2: "பக"
    _draw_glyph_triangle(img, 35, 110, 30, 30)
    _draw_glyph_circle(img, 78, 110, 30, 30)

    recognizer = HistoricalLineRecognizer(classifier=clf)
    result = recognizer.recognize_lines(img)

    assert len(result.lines) == 2
    assert result.total_characters == 5
    assert result.unresolved_characters == 0
    assert result.confidence is None  # Uncalibrated SVM -> must stay None!

    # Check word spacing on line 1
    assert " " in result.lines[0].raw_text
    assert result.lines[0].raw_text.replace(" ", "") == "கமப"
    assert result.lines[1].raw_text == "பக"


def test_dark_estampage_polarity_auto_detection():
    clf = _train_synthetic_classifier()
    # Dark stone/estampage background with light carved strokes
    img = np.full((110, 280), 20, dtype=np.uint8)
    _draw_glyph_circle(img, 30, 35, 32, 32, color=235)
    _draw_glyph_cross(img, 75, 35, 32, 32, color=235)

    recognizer = HistoricalLineRecognizer(classifier=clf)
    result = recognizer.recognize_lines(img)

    assert result.polarity_inverted is False
    assert len(result.lines) == 1
    assert len(result.lines[0].characters) == 2


def test_merged_ligature_marked_unresolved():
    clf = _train_synthetic_classifier()
    img = np.full((120, 320), 235, dtype=np.uint8)

    # Normal character followed by wide merged ligature spanning > 2.25x aspect ratio
    _draw_glyph_circle(img, 30, 35, 30, 30, color=20)
    cv2.rectangle(img, (90, 42), (200, 62), 20, 3)  # Width=110, Height=20 -> AR=5.5

    recognizer = HistoricalLineRecognizer(classifier=clf)
    result = recognizer.recognize_lines(img)

    assert len(result.lines) == 1
    chars = result.lines[0].characters
    assert len(chars) == 2
    assert chars[0].is_unresolved is False
    assert chars[1].is_unresolved is True
    assert chars[1].character == "[?]"
    assert "merged_ligature" in chars[1].unresolved_reason
    assert result.lines[0].has_unresolved is True
    assert "[?]" in result.lines[0].raw_text


def test_vertical_diacritic_merging():
    clf = _train_synthetic_classifier()
    img = np.full((120, 220), 235, dtype=np.uint8)

    # Glyph body + detached dot (pulli) centered directly above it
    _draw_glyph_cross(img, 50, 45, 30, 30, color=20)
    cv2.circle(img, (65, 34), 4, 20, -1)  # Detached pulli above glyph

    recognizer = HistoricalLineRecognizer(
        classifier=clf,
        config=LineRecognizerConfig(merge_vertical_diacritics=True),
    )
    result = recognizer.recognize_lines(img)

    assert len(result.lines) == 1
    assert len(result.lines[0].characters) == 1


def test_original_coordinate_mapping_on_oversized_image():
    clf = _train_synthetic_classifier()
    # Create 2400x1200 image (> 1600 max dim -> scaled by 1.5x to 1600x800)
    img = np.full((1200, 2400), 235, dtype=np.uint8)
    _draw_glyph_circle(img, 300, 300, 120, 120, color=20)

    recognizer = HistoricalLineRecognizer(classifier=clf)
    result = recognizer.recognize_lines(img)

    assert result.original_dimensions == (2400, 1200)
    assert result.working_dimensions == (1600, 800)
    assert len(result.lines) == 1
    orig_x1, orig_y1, orig_x2, orig_y2 = result.lines[0].pixel_box_original
    work_x1, work_y1, work_x2, work_y2 = result.lines[0].pixel_box_working
    # Original box should match ~1.5x working box
    assert orig_x2 > work_x2
    assert abs(orig_x1 - int(round(work_x1 * 1.5))) <= 2
    assert abs(orig_y2 - int(round(work_y2 * 1.5))) <= 2


def test_provider_line_mode_integration(tmp_path):
    clf = _train_synthetic_classifier(calibrate=True)
    registry = HistoricalModelRegistry(models_dir=str(tmp_path))
    registry.register_and_save(clf)

    provider = HistoricalHOGSVMProvider(models_dir=str(tmp_path), default_script="Tamil-Brahmi")
    img = np.full((120, 260), 235, dtype=np.uint8)
    _draw_glyph_circle(img, 30, 35, 30, 30, color=20)
    _draw_glyph_triangle(img, 60, 35, 30, 30, color=20)

    ocr_res = provider.recognize(img, script_hint="Tamil-Brahmi", mode="line")
    assert ocr_res.text == "கப"
    assert len(ocr_res.bounding_boxes) == 1
    assert ocr_res.confidence is not None
    assert ocr_res.metadata["mode"] == "line"
    assert ocr_res.metadata["total_characters"] == 2
