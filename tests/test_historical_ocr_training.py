"""
Unit Tests for Historical Script Class Inventory & Evaluation Harness (Phase I).
"""

import pytest
from services.ai.historical.class_inventory import (
    ScriptClassInventoryRegistry,
    TAMIL_BRAHMI_CLASSES,
    CHOLA_TAMIL_CLASSES,
    CONFUSABLE_PAIRS,
)
from services.ai.historical.evaluation_harness import (
    HistoricalEvaluationHarness,
    CERResult,
    EvaluationReport,
)


def test_script_class_inventory():
    """Verify inventory retrieval and grapheme definitions."""
    tb_classes = ScriptClassInventoryRegistry.get_classes_for_script("Tamil-Brahmi")
    assert len(tb_classes) >= 20
    assert any(c.canonical_char == "𑀓" for c in tb_classes)

    ct_classes = ScriptClassInventoryRegistry.get_classes_for_script("Chola Tamil")
    assert len(ct_classes) >= 25
    assert any(c.canonical_char == "க" for c in ct_classes)


def test_confusable_pairs_registry():
    """Verify registration of known weathered stone confusions."""
    pairs = ScriptClassInventoryRegistry.get_confusable_pairs("Chola Tamil")
    assert len(pairs) >= 3

    pair_tuples = [(p["glyph_a"], p["glyph_b"]) for p in pairs]
    assert ("ண", "ன") in pair_tuples or ("ர", "ா") in pair_tuples
    assert any("precedes" in p["disambiguation_heuristic"] for p in pairs)


def test_cer_calculation_exact():
    """Verify CER calculation on identical strings."""
    ref = "ஸ்வஸ்திஸ்ரீ கோப்பரகேஸரிபன்மற்க்கு"
    cer_res = HistoricalEvaluationHarness.compute_cer(ref, ref)

    assert cer_res.cer == 0.0
    assert cer_res.substitutions == 0
    assert cer_res.deletions == 0
    assert cer_res.insertions == 0


def test_cer_calculation_with_errors():
    """Verify CER with substitutions, deletions, and insertions."""
    ref = "ஸ்வஸ்திஸ்ரீ கோப்பரகேஸரிபன்மற்க்கு"
    # Replace 'ஸ' with 'ச' (1 sub) and drop final 'கு' (1 del)
    hyp = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரிபன்மற்க்"

    cer_res = HistoricalEvaluationHarness.compute_cer(ref, hyp)
    assert cer_res.cer > 0.0
    assert cer_res.substitutions >= 1
    assert cer_res.reference_length == len(ref)


def test_wer_calculation():
    """Verify Word Error Rate."""
    ref = "நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும்"
    hyp = "நாம் கொடுத்தனவும் நம் அக்கன் குடுத்தனவும்"  # 1 substitution in 5 words = 0.20

    wer = HistoricalEvaluationHarness.compute_wer(ref, hyp)
    assert wer == 0.2


def test_pipeline_evaluation_and_contrast_discrepancy():
    """Verify evaluation report generation and contrast degradation tracking."""
    corpus = [
        ("ஸ்வஸ்திஸ்ரீ கோப்பரகேஸரிபன்மற்க்கு", "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரிபன்மற்க்கு"),
        ("கணிய் நந்தஸிரிகுவன் கள்வெள் அறைய்", "கணிய் நந்தஸிரிகுவன் கல்வெள் அறைய்"),
    ]

    report = HistoricalEvaluationHarness.evaluate_pipeline_on_corpus(corpus, script_name="Chola Tamil")
    assert report.sample_count == 2
    assert report.overall_cer > 0.0
    assert report.grapheme_accuracy > 0.8
    assert "Mild CLAHE (Path B)" in report.preprocessing_comparison
    assert report.preprocessing_comparison["Mild CLAHE (Path B)"]["cer"] < report.preprocessing_comparison["Otsu / Sauvola Binarized"]["cer"]
