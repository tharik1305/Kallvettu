"""
Pydantic V2 Schemas for Kallvettu Digital Epigraphy Platform.

Enforces strict input validation, type safety, and serialization for API and DB layers.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from services.database.models import (
    AssetType,
    TranscriptionTier,
    AnnotationStatus,
    ResearcherRole,
    ReviewAction,
    EntityCategory,
)


# --------------------------------------------------------------------------
# Base Schema Configuration
# --------------------------------------------------------------------------

class OrmBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Researcher Schemas
# --------------------------------------------------------------------------

class ResearcherBase(OrmBase):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., max_length=128)
    full_name: str = Field(..., min_length=2, max_length=128)
    institution: Optional[str] = ""
    role: ResearcherRole = ResearcherRole.RESEARCHER


class ResearcherCreate(ResearcherBase):
    pass


class ResearcherResponse(ResearcherBase):
    id: str
    is_active: bool
    created_at: datetime


# --------------------------------------------------------------------------
# ImageAsset Schemas
# --------------------------------------------------------------------------

class ImageAssetBase(OrmBase):
    asset_type: AssetType = AssetType.RAW
    file_path: str
    sha256_hash: str
    width: int = 0
    height: int = 0
    mime_type: str = "image/jpeg"
    is_master: bool = False


class ImageAssetCreate(ImageAssetBase):
    inscription_id: str


class ImageAssetResponse(ImageAssetBase):
    id: str
    inscription_id: str
    created_at: datetime


# --------------------------------------------------------------------------
# GlyphAnnotation Schemas
# --------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class GlyphAnnotationBase(OrmBase):
    character_label: str
    unicode_representation: Optional[str] = ""
    diplomatic_reading: Optional[str] = ""
    script_category: str = "Tamil-Brahmi"
    bounding_box: Dict[str, Any]
    polygon_coords: Optional[List[List[int]]] = None
    legibility_status: str = "CLEAR"
    review_status: AnnotationStatus = AnnotationStatus.UNREVIEWED
    notes: Optional[str] = ""


class GlyphAnnotationCreate(GlyphAnnotationBase):
    inscription_id: str
    image_asset_id: str
    annotator_id: Optional[str] = None


class GlyphAnnotationResponse(GlyphAnnotationBase):
    id: str
    inscription_id: str
    image_asset_id: str
    annotator_id: Optional[str]
    created_at: datetime


# --------------------------------------------------------------------------
# ReadingCandidate & Transcription Schemas
# --------------------------------------------------------------------------

class ReadingCandidateBase(OrmBase):
    provider_name: str
    raw_output: str
    confidence: float = 0.0
    ranking: int = 1


class ReadingCandidateCreate(ReadingCandidateBase):
    transcription_id: str


class ReadingCandidateResponse(ReadingCandidateBase):
    id: str
    transcription_id: str
    created_at: datetime


class TranscriptionBase(OrmBase):
    tier: TranscriptionTier = TranscriptionTier.DIPLOMATIC
    text_content: str
    is_primary: bool = False
    is_ai_generated: bool = False
    provenance: str = "Manual Transcription"


class TranscriptionCreate(TranscriptionBase):
    inscription_id: str
    author_id: Optional[str] = None


class TranscriptionResponse(TranscriptionBase):
    id: str
    inscription_id: str
    author_id: Optional[str]
    created_at: datetime
    reading_candidates: List[ReadingCandidateResponse] = []


# --------------------------------------------------------------------------
# Translation Schemas
# --------------------------------------------------------------------------

class TranslationBase(OrmBase):
    target_language: str = "en"
    translation_type: str = "SCHOLARLY"
    text_content: str
    historical_commentary: Optional[str] = ""
    cultural_context: Optional[str] = ""


class TranslationCreate(TranslationBase):
    inscription_id: str
    translator_id: Optional[str] = None


class TranslationResponse(TranslationBase):
    id: str
    inscription_id: str
    translator_id: Optional[str]
    created_at: datetime


# --------------------------------------------------------------------------
# HistoricalReference & Citation Schemas
# --------------------------------------------------------------------------

class HistoricalReferenceBase(OrmBase):
    ruler_name: Optional[str] = ""
    dynasty: Optional[str] = ""
    regnal_year: Optional[str] = ""
    historical_date: Optional[str] = ""
    historical_event: Optional[str] = ""


class HistoricalReferenceCreate(HistoricalReferenceBase):
    inscription_id: str


class HistoricalReferenceResponse(HistoricalReferenceBase):
    id: str
    inscription_id: str


class CitationBase(OrmBase):
    publication_title: str
    volume: Optional[str] = ""
    number: Optional[str] = ""
    page_numbers: Optional[str] = ""
    year: Optional[int] = None
    external_url: Optional[str] = ""
    doi: Optional[str] = ""


class CitationCreate(CitationBase):
    inscription_id: str


class CitationResponse(CitationBase):
    id: str
    inscription_id: str


# --------------------------------------------------------------------------
# Review Schemas
# --------------------------------------------------------------------------

class ReviewBase(OrmBase):
    action: ReviewAction = ReviewAction.ACCEPT
    previous_reading: Optional[str] = ""
    approved_reading: Optional[str] = ""
    comments: Optional[str] = ""


class ReviewCreate(ReviewBase):
    transcription_id: str
    reviewer_id: str


class ReviewResponse(ReviewBase):
    id: str
    transcription_id: str
    reviewer_id: str
    created_at: datetime


# --------------------------------------------------------------------------
# Knowledge Graph Entity & Relation Schemas
# --------------------------------------------------------------------------

class KnowledgeEntityBase(OrmBase):
    entity_type: EntityCategory
    canonical_name: str
    canonical_name_ta: Optional[str] = ""
    aliases: List[str] = []
    attributes: Dict[str, Any] = {}


class KnowledgeEntityCreate(KnowledgeEntityBase):
    pass


class KnowledgeEntityResponse(KnowledgeEntityBase):
    id: str
    created_at: datetime


class KnowledgeRelationBase(OrmBase):
    relation_type: str
    confidence: float = 1.0
    is_ai_proposed: bool = False
    is_verified: bool = True
    citation_id: Optional[str] = None


class KnowledgeRelationCreate(KnowledgeRelationBase):
    source_id: str
    target_id: str
    source_inscription_id: Optional[str] = None


class KnowledgeRelationResponse(KnowledgeRelationBase):
    id: str
    source_id: str
    target_id: str
    source_inscription_id: Optional[str]
    created_at: datetime


# --------------------------------------------------------------------------
# Inscription Schemas (Aggregate)
# --------------------------------------------------------------------------

class InscriptionBase(OrmBase):
    title_ta: str
    title_en: str
    script_category: str
    historical_period: Optional[str] = ""
    site_name: Optional[str] = ""
    district: Optional[str] = ""
    state: str = "Tamil Nadu"
    material_type: str = "Granite Stone Bed"
    century: Optional[str] = ""
    ruler_name: Optional[str] = ""
    dynasty: Optional[str] = ""
    regnal_year: Optional[str] = ""
    description: Optional[str] = ""


class InscriptionCreate(InscriptionBase):
    id: str


class InscriptionResponse(InscriptionBase):
    id: str
    created_at: datetime
    updated_at: datetime
    image_assets: List[ImageAssetResponse] = []
    transcriptions: List[TranscriptionResponse] = []
    translations: List[TranslationResponse] = []
    historical_references: List[HistoricalReferenceResponse] = []
    citations: List[CitationResponse] = []


class InscriptionSummary(OrmBase):
    id: str
    title_ta: str
    title_en: str
    script_category: str
    historical_period: Optional[str] = ""
    site_name: Optional[str] = ""
    district: Optional[str] = ""
    century: Optional[str] = ""
    dynasty: Optional[str] = ""
    ruler_name: Optional[str] = ""


InscriptionRead = InscriptionResponse


class InscriptionSearchFilter(BaseModel):
    query: Optional[str] = None
    script_category: Optional[str] = None
    period: Optional[str] = None
    site_name: Optional[str] = None
    district: Optional[str] = None
    ruler_name: Optional[str] = None
    dynasty: Optional[str] = None
    century: Optional[str] = None
    limit: int = 50
    offset: int = 0
