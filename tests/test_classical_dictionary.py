"""
Tests for Phase 5: Lightweight Classical Tamil-English Dictionary (CPU-First).
"""

import os
import tempfile
import pytest

from services.nlp.classical_dictionary import (
    ClassicalTamilDictionary,
    DictionaryEntry,
    tamil_to_iso15919,
)
from services.nlp.dictionary_ui import build_dictionary_card_html


@pytest.fixture
def temp_dict(tmp_path):
    db_file = str(tmp_path / "test_dictionary.sqlite")
    dict_inst = ClassicalTamilDictionary(db_path=db_file)
    dict_inst.seed_default_lexicon()
    return dict_inst


def test_dictionary_seeding_and_count(temp_dict):
    count = temp_dict.count_entries()
    assert count >= 60


def test_exact_lookup_nfc(temp_dict):
    entry = temp_dict.lookup("ஆநிரை")
    assert entry is not None
    assert entry.headword == "ஆநிரை"
    assert "Cattle" in entry.english
    assert "பசு" in entry.modern_tamil


def test_full_text_search_english(temp_dict):
    results = temp_dict.search("election")
    assert len(results) >= 1
    headwords = [r.headword for r in results]
    assert "குடவோலை" in headwords


def test_full_text_search_tamil(temp_dict):
    results = temp_dict.search("கோயில்")
    assert len(results) >= 1
    headwords = [r.headword for r in results]
    assert "கற்றளி" in headwords


def test_morphological_analysis_locative(temp_dict):
    analysis = temp_dict.analyze_word("கற்றளியில்")
    assert analysis.original_word == "கற்றளியில்"
    assert analysis.stem == "கற்றளி"
    assert analysis.has_case_marker is True
    assert "-இல்" in analysis.case_name
    assert analysis.is_dictionary_attested is True
    assert "Stone-built temple" in analysis.english


def test_morphological_analysis_dative(temp_dict):
    analysis = temp_dict.analyze_word("குடவோலைக்கு")
    assert analysis.stem == "குடவோலை"
    assert analysis.has_case_marker is True
    assert "-க்கு" in analysis.case_name
    assert "Pot-ballot" in analysis.english


def test_morphological_analysis_royal_honorific(temp_dict):
    analysis = temp_dict.analyze_word("கோப்பரகேசரி")
    assert analysis.has_honorific is True
    assert "Monarch" in analysis.honorific_description or "அரசன்" in analysis.honorific_description


def test_dictionary_export_json_and_csv(temp_dict, tmp_path):
    json_path = str(tmp_path / "lexicon_export.json")
    csv_path = str(tmp_path / "lexicon_export.csv")

    temp_dict.export_json(json_path)
    temp_dict.export_csv(csv_path)

    assert os.path.isfile(json_path)
    assert os.path.isfile(csv_path)
    assert os.path.getsize(json_path) > 1000
    assert os.path.getsize(csv_path) > 1000


def test_tamil_to_iso15919():
    """Verifies academic ISO 15919 transliteration across inscriptional vocabulary."""
    assert tamil_to_iso15919("கற்றளி") == "kaṟṟaḷi"
    assert tamil_to_iso15919("கற்றளியில்") == "kaṟṟaḷiyil"
    assert tamil_to_iso15919("குடவோலை") == "kuṭavōlai"
    assert tamil_to_iso15919("ஆநிரை") == "ānirai"
    assert tamil_to_iso15919("சாசனம்") == "cācaṉam"


def test_morphological_analysis_sandhi_katrali(temp_dict):
    """
    Verifies that analyze_word('கற்றளியில்') performs the complete Nannūl Sandhi
    and epigraphical morphological decomposition specified in the requirements.
    """
    analysis = temp_dict.analyze_word("கற்றளியில்")
    assert analysis.original_word == "கற்றளியில்"
    assert analysis.normalized_word == "கற்றளியில்"
    assert analysis.stem == "கற்றளி"
    assert analysis.transliteration == "kaṟṟaḷiyil"
    assert analysis.stem_transliteration == "kaṟṟaḷi"
    assert analysis.has_case_marker is True
    assert "யில்" in analysis.case_suffix or "இல்" in analysis.case_suffix
    assert "7th" in analysis.case_name or "Locative" in analysis.case_name

    # Sandhi rules
    assert "கல் + தளி" in analysis.root_components
    assert "நன்னூல் 227" in analysis.compound_sandhi_rule
    assert "நன்னூல் 162" in analysis.inflection_sandhi_rule or "உடம்படுமெய்" in analysis.inflection_sandhi_rule
    assert "7th Locative Case" in analysis.syntactic_function or "இடப்பொருள்" in analysis.syntactic_function

    # Historical examples & related terms
    assert len(analysis.historical_examples) >= 1
    assert "இத்திருக்கற்றளியிலே" in analysis.historical_examples[0]
    assert len(analysis.related_terms) >= 3
    assert "குடவோலை" in analysis.related_terms or "திருக்கற்றளி" in analysis.related_terms
    assert len(analysis.alternative_analyses) >= 2
    assert analysis.reference_id == "KV-LEX-EP-001"
    assert "Attested" in analysis.attestation_status


def test_dictionary_card_html_structure(temp_dict):
    """
    Validates that build_dictionary_card_html produces valid, non-code-fenced HTML
    with all 5 required sections, high-contrast dark graphite colors, and balanced divs.
    """
    analysis = temp_dict.analyze_word("கற்றளியில்")
    html_output = build_dictionary_card_html(analysis)

    # 1. Zero lines starting with 4+ spaces (immune to markdown code-fence bug)
    lines = html_output.splitlines()
    for idx, line in enumerate(lines):
        assert not line.startswith("    "), f"Line {idx} has 4+ leading spaces: {line[:30]}"
        assert not line.startswith("\t"), f"Line {idx} has leading tab: {line[:30]}"

    # 2. Balanced <div> tags
    open_divs = html_output.count("<div")
    close_divs = html_output.count("</div>")
    assert open_divs == close_divs, f"Mismatched divs: {open_divs} open vs {close_divs} close"

    # 3. All 5 distinct sections must be present
    assert "1. சொல் விவரம்" in html_output
    assert "2. பொருளும் மொழிபெயர்ப்பும்" in html_output
    assert "3. உருபியல் & புணர்ச்சி பகுப்பாய்வு" in html_output
    assert "4. வரலாற்றுச் சூழல்" in html_output
    assert "5. அகராதிச் சான்றாதாரங்கள்" in html_output

    # 4. Content verification
    assert "கற்றளியில்" in html_output
    assert "/kaṟṟaḷiyil/" in html_output
    assert "கல் + தளி" in html_output
    assert "நன்னூல் 227" in html_output
    assert "நன்னூல் 162" in html_output
    assert "KV-LEX-EP-001" in html_output

    # 5. Contrast and theme verification
    assert "#171D24" in html_output  # Dark secondary card background
    assert "#C5A15A" in html_output  # Muted bronze heading accent
    assert "#FFFFFF" in html_output  # High-contrast modern Tamil text
    assert "#94A3B8" in html_output  # Readable Slate-400 label text


def test_dictionary_search_result_html(temp_dict):
    """Verifies that search results render the 5-section card cleanly."""
    results = temp_dict.search("குடவோலை")
    assert len(results) >= 1
    hit = results[0]
    html_output = build_dictionary_card_html(hit)

    assert "1. சொல் விவரம்" in html_output
    assert "2. பொருளும் மொழிபெயர்ப்பும்" in html_output
    assert "3. உருபியல் & புணர்ச்சி பகுப்பாய்வு" in html_output
    assert "4. வரலாற்றுச் சூழல்" in html_output
    assert "5. அகராதிச் சான்றாதாரங்கள்" in html_output
    assert "குடவோலை" in html_output
    assert "/kuṭavōlai/" in html_output
    assert html_output.count("<div") == html_output.count("</div>")


