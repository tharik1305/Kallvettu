"""
Kallvettu Public Epigraphical Research API (Phase K).

Open, versioned RESTful API (FastAPI) enabling programmatic access to:
- Inscription catalog & metadata.
- TEI EpiDoc XML & full archival ZIP research bundles.
- IIIF Presentation 3.0 manifests for deep-zoom viewers.
- Epigraphical search engine with Tamil morphological expansion.
- Scholarly Knowledge Graph nodes and verified historical relations.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.database.connection import check_database_health, get_db_session
from services.database.models import (
    Inscription,
    Transcription,
    Translation,
    Citation,
    HistoricalReference,
)
from services.database.repository import EpigraphicalRepository
from services.database.schemas import InscriptionRead, InscriptionSummary
from services.epidoc.exporter import EpiDocExporter
from services.epidoc.bundle import EpiDocArchiveBundler
from services.iiif.manifest_builder import IIIFManifestBuilder
from services.search.search_engine import EpigraphicalSearchEngine
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph

logger = logging.getLogger("kallvettu.api")

app = FastAPI(
    title="Kallvettu Digital Epigraphy API",
    version="3.1.0",
    description="Scholarly API for ancient & medieval Tamil inscriptions, EpiDoc exports, and IIIF manifests.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Open academic access CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """Returns database connection status and API health."""
    db_health = check_database_health()
    return {
        "status": "healthy",
        "service": "Kallvettu Digital Epigraphy API",
        "version": "3.1.0",
        "database": db_health,
    }


@app.get("/api/v1/inscriptions", response_model=List[InscriptionSummary], tags=["Inscriptions"])
def list_inscriptions(
    script: Optional[str] = Query(None, description="Filter by script category (e.g. 'Tamil-Brahmi', 'Chola Tamil')"),
    dynasty: Optional[str] = Query(None, description="Filter by dynasty (e.g. 'Chola', 'Pandya')"),
    district: Optional[str] = Query(None, description="Filter by district (e.g. 'Madurai', 'Thanjavur')"),
    century: Optional[str] = Query(None, description="Filter by century"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Lists inscriptions with pagination and administrative/historical filters."""
    with get_db_session() as s:
        q = s.query(Inscription)
        if script and isinstance(script, str):
            q = q.filter(Inscription.script_category.ilike(f"%{script}%"))
        if dynasty and isinstance(dynasty, str):
            q = q.filter(Inscription.dynasty.ilike(f"%{dynasty}%"))
        if district and isinstance(district, str):
            q = q.filter(Inscription.district.ilike(f"%{district}%"))
        if century and isinstance(century, str):
            q = q.filter(Inscription.century.ilike(f"%{century}%"))

        p = page if isinstance(page, int) else 1
        ps = page_size if isinstance(page_size, int) else 20
        offset = (p - 1) * ps
        records = q.offset(offset).limit(ps).all()
        return [InscriptionSummary.model_validate(r) for r in records]


@app.get("/api/v1/inscriptions/{inscription_id}", response_model=InscriptionRead, tags=["Inscriptions"])
def get_inscription_detail(inscription_id: str):
    """Retrieves full inscription dossier with transcriptions, translations, and citations."""
    insc = EpigraphicalRepository.get_inscription(inscription_id)
    if not insc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inscription '{inscription_id}' not found.",
        )
    return InscriptionRead.model_validate(insc)


@app.get("/api/v1/inscriptions/{inscription_id}/epidoc", tags=["EpiDoc"])
def export_epidoc(
    inscription_id: str,
    format: str = Query("xml", pattern="^(xml|zip)$", description="'xml' for single TEI XML, 'zip' for full research bundle"),
):
    """Exports EpiDoc TEI XML or full ZIP research archive for an inscription."""
    fmt = format if isinstance(format, str) else "xml"
    with get_db_session() as s:
        insc = s.query(Inscription).filter(Inscription.id == inscription_id).first()
        if not insc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inscription '{inscription_id}' not found.",
            )

        diplomatic = ""
        normalized = ""
        for tr in insc.transcriptions:
            if tr.tier.value == "DIPLOMATIC":
                diplomatic = tr.text_content
            elif tr.tier.value == "NORMALIZED":
                normalized = tr.text_content

        english = ""
        modern_ta = ""
        for tl in insc.translations:
            if tl.target_language == "en":
                english = tl.text_content
            elif tl.target_language == "ta":
                modern_ta = tl.text_content

        citation = insc.citations[0].publication_title if insc.citations else ""

        if fmt == "xml":
            xml_content = EpiDocExporter.generate_tei_epidoc_xml(
                inscription_id=insc.id,
                title_en=insc.title_en,
                title_ta=insc.title_ta,
                script_category=insc.script_category,
                historical_period=insc.historical_period or "",
                site_name=insc.site_name or "",
                district=insc.district or "",
                material_type=insc.material_type or "",
                century=insc.century or "",
                ruler_name=insc.ruler_name or "",
                dynasty=insc.dynasty or "",
                diplomatic_text=diplomatic or normalized,
                normalized_text=normalized,
                english_translation=english,
                modern_tamil_translation=modern_ta,
                citation_source=citation,
            )
            return Response(
                content=xml_content,
                media_type="application/xml",
                headers={"Content-Disposition": f'attachment; filename="{insc.id}_epidoc.xml"'},
            )
        else:
            zip_bytes = EpiDocArchiveBundler.create_research_bundle_zip(
                inscription_id=insc.id,
                title_en=insc.title_en,
                title_ta=insc.title_ta,
                script_category=insc.script_category,
                historical_period=insc.historical_period or "",
                site_name=insc.site_name or "",
                district=insc.district or "",
                diplomatic_text=diplomatic or normalized,
                normalized_text=normalized,
                english_translation=english,
                modern_tamil_translation=modern_ta,
                citation=citation,
                description=insc.description or "",
            )
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{insc.id}_research_bundle.zip"'},
            )


@app.get("/api/v1/inscriptions/{inscription_id}/manifest", tags=["IIIF"])
def get_iiif_manifest(inscription_id: str):
    """Generates IIIF Presentation 3.0 manifest for the inscription image."""
    with get_db_session() as s:
        insc = s.query(Inscription).filter(Inscription.id == inscription_id).first()
        if not insc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inscription '{inscription_id}' not found.",
            )

        manifest = IIIFManifestBuilder.build_manifest(
            inscription_id=insc.id,
            label=f"{insc.title_en} ({insc.title_ta})",
            summary=insc.description or insc.title_en,
            image_url=f"/api/v1/images/{insc.id}.jpg",
            metadata_items=[
                {"label": "Script", "value": insc.script_category},
                {"label": "Dynasty", "value": insc.dynasty or "Unknown"},
                {"label": "District", "value": insc.district or "Unknown"},
                {"label": "Period", "value": insc.historical_period or "Unknown"},
            ],
        )
        return JSONResponse(content=manifest, media_type="application/ld+json")


@app.get("/api/v1/search", tags=["Search"])
def search_inscriptions(
    q: str = Query("", description="Keywords in Tamil or English"),
    script: Optional[str] = Query(None),
    dynasty: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    expand_morphology: bool = Query(True),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Full-text and metadata search with classical Tamil morphological variant expansion."""
    query_str = q if isinstance(q, str) else ""
    script_str = script if isinstance(script, str) else None
    dynasty_str = dynasty if isinstance(dynasty, str) else None
    district_str = district if isinstance(district, str) else None
    expand_bool = expand_morphology if isinstance(expand_morphology, bool) else True
    p = page if isinstance(page, int) else 1
    ps = page_size if isinstance(page_size, int) else 20

    res = EpigraphicalSearchEngine.search(
        query=query_str,
        script_category=script_str,
        dynasty=dynasty_str,
        district=district_str,
        expand_morphology=expand_bool,
        page=p,
        page_size=ps,
    )
    return {
        "query": res.query,
        "total_count": res.total_count,
        "expanded_terms": res.expanded_terms,
        "facets": res.facets,
        "items": [
            {
                "inscription_id": item.inscription_id,
                "title_en": item.title_en,
                "title_ta": item.title_ta,
                "script_category": item.script_category,
                "historical_period": item.historical_period,
                "site_name": item.site_name,
                "district": item.district,
                "dynasty": item.dynasty,
                "ruler_name": item.ruler_name,
                "relevance_score": item.relevance_score,
                "matching_snippets": item.matching_snippets,
                "master_reading": item.master_reading,
                "english_translation": item.english_translation,
                "citation": item.citation,
            }
            for item in res.items
        ],
    }


@app.get("/api/v1/knowledge-graph", tags=["Knowledge Graph"])
def get_knowledge_graph(category: Optional[str] = Query(None)):
    """Returns knowledge graph entities and scholarly relations in JSON format."""
    kg = EpigraphicalKnowledgeGraph()
    cat_str = category if isinstance(category, str) else None
    if cat_str and cat_str != "ALL":
        sub = kg.get_subgraph(categories=[cat_str])
        return sub.to_dict()
    return kg.to_dict()
