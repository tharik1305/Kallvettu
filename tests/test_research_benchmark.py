"""
Tests for Phase 6: Research Evaluation Suite & CPU Benchmarks.
"""

import os
import numpy as np

from services.evaluation.research_benchmark import (
    ResearchBenchmarkSuite,
    compute_cer,
    compute_wer,
    levenshtein_distance,
    measure_process_memory_mb,
)
from services.ai.historical.glyph_classifier import (
    HOGSVMGlyphClassifier,
    GlyphCandidate,
    GlyphPredictionResult,
)


def test_edit_distance_and_error_rates():
    assert levenshtein_distance("கற்றளி", "கற்றளி") == 0
    assert levenshtein_distance("கற்றளி", "கற்பளி") == 1

    # Exact matches
    assert compute_cer("கோனேரின்மை", "கோனேரின்மை") == 0.0
    assert compute_wer("கோனேரின்மை கொண்டான்", "கோனேரின்மை கொண்டான்") == 0.0

    # 1 substitution in 4 chars = 0.25
    assert compute_cer("நந்த", "நந்த") == 0.0
    assert compute_cer("நந்தா", "நந்த") > 0.0

    # Word error
    assert compute_wer("நந்த குவன்", "நந்த இகுவன்") == 0.5


def test_process_memory_measurement():
    rss = measure_process_memory_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_evaluate_ocr_callable():
    suite = ResearchBenchmarkSuite()
    test_items = [
        ("s1", "dummy1", "கோனேரின்மை"),
        ("s2", "dummy2", "கற்றளி"),
    ]

    # Predictor with 1 exact match and 1 substitution
    def mock_pred(img):
        return "கோனேரின்மை" if img == "dummy1" else "கற்பளி"

    summary = suite.evaluate_ocr_callable(
        engine_id="mock_test",
        engine_name="Mock OCR",
        test_items=test_items,
        predict_fn=mock_pred,
        model_size_mb=4.2,
    )

    assert summary.sample_count == 2
    assert summary.within_25mb_budget is True
    assert summary.exact_match_rate == 0.5
    assert summary.mean_cer > 0.0
    assert summary.mean_latency_ms >= 0.0


class MockClassifier:
    def __init__(self):
        self.script = "Tamil-Brahmi"
        self.model_version = "test-v1"

    def predict_glyph(self, img, top_k=3):
        # returns label passed in img
        if img == "unknown":
            return GlyphPredictionResult(
                predicted_character="[?]",
                is_rejected=True,
                rejection_reason="low_score",
            )
        return GlyphPredictionResult(
            predicted_character=img,
            top_candidates=[
                GlyphCandidate(character=img, score=2.5, is_calibrated=False),
                GlyphCandidate(character="க", score=1.0, is_calibrated=False),
                GlyphCandidate(character="ப", score=0.5, is_calibrated=False),
            ],
            is_rejected=False,
        )


def test_evaluate_glyph_classifier():
    suite = ResearchBenchmarkSuite()
    clf = MockClassifier()
    samples = [
        ("க", "க"),
        ("ப", "ப"),
        ("ப", "ப"),
        ("unknown", "ம"),  # rejected
    ]

    report = suite.evaluate_glyph_classifier(clf, samples, model_size_mb=0.08)
    assert report.sample_count == 4
    assert report.top1_accuracy == 0.75  # 3 of 4 match
    assert report.rejection_rate == 0.25
    assert report.within_25mb_budget is True
    assert len(report.classes) >= 3
    assert len(report.confusion_matrix) == len(report.classes)


def test_export_summaries_and_markdown(tmp_path):
    suite = ResearchBenchmarkSuite()
    summary = suite.evaluate_ocr_callable(
        engine_id="test_engine",
        engine_name="Test Engine",
        test_items=[("1", "x", "கம")],
        predict_fn=lambda _: "கம",
        model_size_mb=1.5,
    )

    j_path = str(tmp_path / "bench.json")
    c_path = str(tmp_path / "bench.csv")
    suite.export_summaries_json([summary], j_path)
    suite.export_summaries_csv([summary], c_path)

    assert os.path.isfile(j_path)
    assert os.path.isfile(c_path)

    md = suite.build_markdown_report([summary])
    assert "Test Engine" in md
    assert "Mean CER" in md

