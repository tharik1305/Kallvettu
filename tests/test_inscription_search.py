"""
Unit Tests for Epigraphical Inscription Search Engine (Phase H).
"""

import pytest
from services.database.connection import get_db_session
from services.database.repository import EpigraphicalRepository
from services.search.search_engine import EpigraphicalSearchEngine


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_search.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")
    import services.database.connection as conn_mod
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None
    EpigraphicalRepository.init_database()
    EpigraphicalRepository.seed_canonical_corpus()
    yield
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None


def test_unicode_normalization_and_expansion():
    """Verify NFC normalization and domain-specific morphological expansion."""
    raw = "  இராஜராஜ   "
    clean = EpigraphicalSearchEngine.normalize_query(raw)
    assert clean == "இராஜராஜ"

    expanded = EpigraphicalSearchEngine.expand_morphological_variants("இராஜராஜ")
    assert "கோப்பரகேஸரி" in expanded
    assert "ராசராச" in expanded

    kudavolai_expanded = EpigraphicalSearchEngine.expand_morphological_variants("குடவோலை")
    assert "உத்திரமேரூர்" in kudavolai_expanded


def test_keyword_search_rajaraja():
    """Verify search for Rajaraja Chola inscription."""
    results = EpigraphicalSearchEngine.search("இராஜராஜ")
    assert results.total_count >= 1

    top_item = results.items[0]
    assert "KV-INSC-002" in top_item.inscription_id
    assert "Brihadisvara" in top_item.title_en
    assert top_item.relevance_score > 0
    assert len(top_item.matching_snippets) > 0
    assert "<mark>" in top_item.matching_snippets[0]


def test_filter_by_script_and_district():
    """Verify metadata filtering by script category and district."""
    # Filter by Tamil-Brahmi
    tb_results = EpigraphicalSearchEngine.search("", script_category="Tamil-Brahmi")
    assert tb_results.total_count >= 2
    for item in tb_results.items:
        assert "Tamil-Brahmi" in item.script_category

    # Filter by Madurai district
    madurai_results = EpigraphicalSearchEngine.search("", district="Madurai")
    assert madurai_results.total_count >= 1
    assert any("Mangulam" in item.title_en for item in madurai_results.items)


def test_search_facets():
    """Verify faceted aggregation counts across scripts, dynasties, and districts."""
    results = EpigraphicalSearchEngine.search("")
    assert results.total_count == 6
    assert "scripts" in results.facets
    assert "dynasties" in results.facets
    assert "districts" in results.facets

    assert results.facets["scripts"].get("Tamil-Brahmi", 0) >= 2
    assert results.facets["dynasties"].get("Chola Dynasty", 0) >= 2


def test_pagination():
    """Verify pagination slices."""
    p1 = EpigraphicalSearchEngine.search("", page=1, page_size=2)
    assert len(p1.items) == 2
    p2 = EpigraphicalSearchEngine.search("", page=2, page_size=2)
    assert len(p2.items) == 2
    assert p1.items[0].inscription_id != p2.items[0].inscription_id
