"""
Tests for Phase B: Persistent Epigraphical Research Database.
"""

import os
import tempfile
import pytest

from services.database.connection import get_engine, check_database_health
from services.database.models import (
    Inscription,
    TranscriptionTier,
    AnnotationStatus,
    ResearcherRole,
    ReviewAction,
)
from services.database.repository import EpigraphicalRepository
from services.database.schemas import InscriptionSearchFilter


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_research.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")
    # Reset engine
    import services.database.connection as conn_mod
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None
    EpigraphicalRepository.init_database()
    yield
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None


def test_database_health():
    health = check_database_health()
    assert health["status"] == "healthy"
    assert health["dialect"] == "sqlite"


def test_seed_canonical_corpus():
    seeded = EpigraphicalRepository.seed_canonical_corpus()
    assert seeded >= 6
    count = EpigraphicalRepository.count_inscriptions()
    assert count >= 6

    # Verify retrieval
    mangulam = EpigraphicalRepository.get_inscription("KV-INSC-001")
    assert mangulam is not None
    assert "மாங்குளம்" in mangulam.title_ta
    assert len(mangulam.transcriptions) >= 2
    assert len(mangulam.translations) >= 2
    assert len(mangulam.citations) >= 1


def test_inscription_crud():
    data = {
        "id": "KV-TEST-999",
        "title_ta": "சோழர் காலச் சாசனம்",
        "title_en": "Chola Epigraph Record",
        "script_category": "Medieval Chola Tamil",
        "district": "Thanjavur",
        "ruler_name": "Rajaraja I",
    }
    insc = EpigraphicalRepository.create_inscription(data)
    assert insc.id == "KV-TEST-999"

    # Add image asset
    asset = EpigraphicalRepository.add_image_asset(
        inscription_id="KV-TEST-999",
        file_path="samples/chola_kalvettu.jpg",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        width=1600,
        height=1200,
        is_master=True,
    )
    assert asset.id.startswith("IMG-")

    # Add glyph annotation
    ann = EpigraphicalRepository.add_glyph_annotation(
        inscription_id="KV-TEST-999",
        image_asset_id=asset.id,
        character_label="க",
        bounding_box={"x": 50, "y": 100, "w": 40, "h": 45},
        unicode_char="க",
        script_category="Medieval Chola Tamil",
    )
    assert ann.id.startswith("GLYPH-")

    # Fetch annotations
    annotations = EpigraphicalRepository.list_annotations_for_inscription("KV-TEST-999")
    assert len(annotations) == 1
    assert annotations[0].character_label == "க"


def test_researcher_and_review():
    researcher = EpigraphicalRepository.get_or_create_researcher(
        username="epigraphist_dr_raman",
        email="raman@epigraphy.res.in",
        full_name="Dr. K. Raman",
        role=ResearcherRole.REVIEWER,
    )
    assert researcher.id.startswith("USR-")

    # Seed inscription first
    EpigraphicalRepository.seed_canonical_corpus()
    thanjavur = EpigraphicalRepository.get_inscription("KV-INSC-002")
    tr = thanjavur.transcriptions[0]

    # Submit review
    review = EpigraphicalRepository.submit_review(
        transcription_id=tr.id,
        reviewer_id=researcher.id,
        action=ReviewAction.ACCEPT,
        approved_reading=tr.text_content,
        comments="Confirmed by comparing with ASI estampage Vol II, Plate 1.",
    )
    assert review.id.startswith("REV-")
    assert review.action == ReviewAction.ACCEPT


def test_search_inscriptions():
    EpigraphicalRepository.seed_canonical_corpus()

    # Search by site/district
    filters = InscriptionSearchFilter(query="Madurai")
    results, total = EpigraphicalRepository.search_inscriptions(filters)
    assert total >= 1
    assert any(r.id == "KV-INSC-001" for r in results)

    # Search by dynasty
    filters2 = InscriptionSearchFilter(dynasty="Chola")
    results2, total2 = EpigraphicalRepository.search_inscriptions(filters2)
    assert total2 >= 2

