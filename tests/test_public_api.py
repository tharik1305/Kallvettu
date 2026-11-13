"""
Unit & Integration Tests for Kallvettu Public Epigraphical API (Phase K).

Tests route handlers, schema serialization, query filtering, and route registration
in CPU-bound sandboxed execution without requiring local socketpair networking.
"""

import json
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

from services.api.app import (
    app,
    get_health,
    list_inscriptions,
    get_inscription_detail,
    export_epidoc,
    get_iiif_manifest,
    search_inscriptions,
    get_knowledge_graph,
)
from services.database.repository import EpigraphicalRepository


@pytest.fixture(autouse=True)
def setup_api_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_api.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")

    import services.database.connection as conn_mod
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None

    EpigraphicalRepository.init_database()
    EpigraphicalRepository.seed_canonical_corpus()
    yield
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None


def test_api_route_registration():
    """Verify all versioned /api/v1 routes are properly registered on the FastAPI app."""
    paths = [route.path for route in app.routes]
    assert "/api/v1/health" in paths
    assert "/api/v1/inscriptions" in paths
    assert "/api/v1/inscriptions/{inscription_id}" in paths
    assert "/api/v1/inscriptions/{inscription_id}/epidoc" in paths
    assert "/api/v1/inscriptions/{inscription_id}/manifest" in paths
    assert "/api/v1/search" in paths
    assert "/api/v1/knowledge-graph" in paths


def test_api_health_endpoint():
    """Verify /api/v1/health handler returns healthy status and DB metadata."""
    res = get_health()
    assert res["status"] == "healthy"
    assert res["version"] == "3.1.0"
    assert res["database"]["status"] == "healthy"


def test_list_inscriptions_endpoint():
    """Verify listing inscriptions with filtering and pagination."""
    items = list_inscriptions(page=1, page_size=20)
    assert len(items) >= 4

    # Filter by Chola
    chola_items = list_inscriptions(dynasty="Chola", page=1, page_size=20)
    assert len(chola_items) >= 2
    assert all("Chola" in item.dynasty for item in chola_items)

    # Filter by Tamil-Brahmi script
    tb_items = list_inscriptions(script="Tamil-Brahmi", page=1, page_size=20)
    assert len(tb_items) >= 2
    assert all("Tamil-Brahmi" in item.script_category for item in tb_items)


def test_get_inscription_detail_and_404():
    """Verify full detail retrieval and 404 behavior."""
    detail = get_inscription_detail("KV-INSC-002")
    assert detail.id == "KV-INSC-002"
    assert "Brihadisvara" in detail.title_en
    assert len(detail.transcriptions) >= 1

    # 404 test
    with pytest.raises(HTTPException) as exc_info:
        get_inscription_detail("NON_EXISTENT_ID")
    assert exc_info.value.status_code == 404


def test_export_epidoc_xml_and_zip():
    """Verify TEI EpiDoc XML and ZIP bundle generation endpoints."""
    # XML format
    xml_resp = export_epidoc("KV-INSC-002", format="xml")
    assert isinstance(xml_resp, Response)
    assert xml_resp.media_type == "application/xml"
    assert "TEI" in xml_resp.body.decode("utf-8")
    assert "KV-INSC-002" in xml_resp.body.decode("utf-8")

    # ZIP format
    zip_resp = export_epidoc("KV-INSC-002", format="zip")
    assert isinstance(zip_resp, Response)
    assert zip_resp.media_type == "application/zip"
    assert len(zip_resp.body) > 100


def test_iiif_manifest_endpoint():
    """Verify IIIF Presentation 3.0 manifest output."""
    manifest_resp = get_iiif_manifest("KV-INSC-002")
    assert isinstance(manifest_resp, JSONResponse)
    manifest = json.loads(manifest_resp.body.decode("utf-8"))
    assert manifest["type"] == "Manifest"
    assert "http://iiif.io/api/presentation/3/context.json" in manifest["@context"]
    assert len(manifest["items"]) >= 1


def test_search_endpoint():
    """Verify full-text search API with Tamil query expansion."""
    res = search_inscriptions(q="குடவோலை", expand_morphology=True, page=1, page_size=20)
    assert res["total_count"] >= 1
    assert "குடவோலை" in res["query"]
    assert any("KV-INSC-003" in item["inscription_id"] for item in res["items"])


def test_knowledge_graph_endpoint():
    """Verify knowledge graph endpoint."""
    kg_data = get_knowledge_graph()
    assert kg_data["node_count"] >= 15
    assert len(kg_data["nodes"]) == kg_data["node_count"]
