"""
SQLAlchemy ORM Data Models for Kallvettu Digital Epigraphy Platform.

Defines the 16 core epigraphical research entities:
1. Inscription
2. ImageAsset
3. Transcription
4. ReadingCandidate
5. GlyphAnnotation
6. Translation
7. HistoricalReference
8. Citation
9. Researcher
10. Review
11. Revision
12. AuditEvent
13. ModelRun
14. DatasetVersion
15. KnowledgeEntity
16. KnowledgeRelation
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import List, Optional

def _utc_now():
    return datetime.now(timezone.utc)

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Enum,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship

from services.database.connection import Base


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class AssetType(str, enum.Enum):
    RAW = "RAW"
    PROCESSED = "PROCESSED"
    SQUEEZE = "SQUEEZE"
    TILE_DZI = "TILE_DZI"
    THUMBNAIL = "THUMBNAIL"


class TranscriptionTier(str, enum.Enum):
    DIPLOMATIC = "DIPLOMATIC"
    NORMALIZED = "NORMALIZED"
    WORD_SEGMENTED = "WORD_SEGMENTED"
    CRITICAL_EDITION = "CRITICAL_EDITION"


class AnnotationStatus(str, enum.Enum):
    UNREVIEWED = "UNREVIEWED"
    ANNOTATED = "ANNOTATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    ILLEGIBLE = "ILLEGIBLE"
    REJECTED = "REJECTED"


class ResearcherRole(str, enum.Enum):
    RESEARCHER = "RESEARCHER"
    REVIEWER = "REVIEWER"
    ADMINISTRATOR = "ADMINISTRATOR"


class ReviewAction(str, enum.Enum):
    ACCEPT = "ACCEPT"
    ADOPT_PROVIDER = "ADOPT_PROVIDER"
    EDIT_SCHOLAR = "EDIT_SCHOLAR"
    MARK_ILLEGIBLE = "MARK_ILLEGIBLE"
    DEFER = "DEFER"
    REJECT = "REJECT"


class EntityCategory(str, enum.Enum):
    RULER = "RULER"
    DYNASTY = "DYNASTY"
    TEMPLE = "TEMPLE"
    LOCATION = "LOCATION"
    DONOR = "DONOR"
    OFFICIAL = "OFFICIAL"
    TAX = "TAX"
    GRANT = "GRANT"
    INSCRIPTION = "INSCRIPTION"


# --------------------------------------------------------------------------
# ORM Models
# --------------------------------------------------------------------------

class Researcher(Base):
    """Authenticated epigraphical researcher or reviewer."""
    __tablename__ = "researchers"

    id = Column(String(64), primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False)
    full_name = Column(String(128), nullable=False)
    institution = Column(String(256), default="")
    role = Column(Enum(ResearcherRole), default=ResearcherRole.RESEARCHER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    transcriptions = relationship("Transcription", back_populates="author")
    reviews = relationship("Review", back_populates="reviewer")


class Inscription(Base):
    """Core epigraphical inscription record."""
    __tablename__ = "inscriptions"

    id = Column(String(64), primary_key=True, index=True)
    title_ta = Column(String(256), nullable=False, index=True)
    title_en = Column(String(256), nullable=False, index=True)
    script_category = Column(String(64), nullable=False, index=True)  # Tamil-Brahmi, Vatteluttu, etc.
    historical_period = Column(String(128), default="", index=True)
    site_name = Column(String(128), default="", index=True)
    district = Column(String(64), default="", index=True)
    state = Column(String(64), default="Tamil Nadu")
    material_type = Column(String(64), default="Granite Stone Bed")
    century = Column(String(64), default="")
    ruler_name = Column(String(128), default="", index=True)
    dynasty = Column(String(64), default="", index=True)
    regnal_year = Column(String(32), default="")
    description = Column(Text, default="")
    created_at = Column(DateTime, default=_utc_now)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    # Relationships
    image_assets = relationship("ImageAsset", back_populates="inscription", cascade="all, delete-orphan")
    transcriptions = relationship("Transcription", back_populates="inscription", cascade="all, delete-orphan")
    translations = relationship("Translation", back_populates="inscription", cascade="all, delete-orphan")
    historical_references = relationship("HistoricalReference", back_populates="inscription", cascade="all, delete-orphan")
    citations = relationship("Citation", back_populates="inscription", cascade="all, delete-orphan")
    glyph_annotations = relationship("GlyphAnnotation", back_populates="inscription", cascade="all, delete-orphan")


class ImageAsset(Base):
    """High-resolution photograph or preprocessed derivative of an inscription."""
    __tablename__ = "image_assets"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_type = Column(Enum(AssetType), default=AssetType.RAW, nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    mime_type = Column(String(64), default="image/jpeg")
    is_master = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    inscription = relationship("Inscription", back_populates="image_assets")
    annotations = relationship("GlyphAnnotation", back_populates="image_asset")


class Transcription(Base):
    """Epigraphical reading at diplomatic, normalized, or segmented tier."""
    __tablename__ = "transcriptions"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    tier = Column(Enum(TranscriptionTier), default=TranscriptionTier.DIPLOMATIC, nullable=False, index=True)
    text_content = Column(Text, nullable=False)
    author_id = Column(String(64), ForeignKey("researchers.id"), nullable=True)
    is_primary = Column(Boolean, default=False)
    is_ai_generated = Column(Boolean, default=False)
    provenance = Column(String(256), default="Manual Transcription")
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    inscription = relationship("Inscription", back_populates="transcriptions")
    author = relationship("Researcher", back_populates="transcriptions")
    reading_candidates = relationship("ReadingCandidate", back_populates="transcription", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="transcription")


class ReadingCandidate(Base):
    """Candidate output generated by an OCR provider or researcher before fusion."""
    __tablename__ = "reading_candidates"

    id = Column(String(64), primary_key=True)
    transcription_id = Column(String(64), ForeignKey("transcriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    provider_name = Column(String(64), nullable=False, index=True)  # PaddleOCR, Gemini, Sarvam, HOG-SVM
    raw_output = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    ranking = Column(Integer, default=1)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    transcription = relationship("Transcription", back_populates="reading_candidates")


class GlyphAnnotation(Base):
    """Bounding box or polygon character annotation on an inscription image."""
    __tablename__ = "glyph_annotations"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    image_asset_id = Column(String(64), ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False, index=True)
    character_label = Column(String(32), nullable=False, index=True)
    unicode_representation = Column(String(32), default="")
    diplomatic_reading = Column(String(32), default="")
    script_category = Column(String(64), default="Tamil-Brahmi")
    bounding_box = Column(JSON, nullable=False)  # {"x": int, "y": int, "w": int, "h": int}
    polygon_coords = Column(JSON, nullable=True)  # [[x1, y1], [x2, y2], ...]
    legibility_status = Column(String(32), default="CLEAR")  # CLEAR, WEATHERED, PARTIAL, ILLEGIBLE
    review_status = Column(Enum(AnnotationStatus), default=AnnotationStatus.UNREVIEWED, index=True)
    annotator_id = Column(String(64), ForeignKey("researchers.id"), nullable=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    inscription = relationship("Inscription", back_populates="glyph_annotations")
    image_asset = relationship("ImageAsset", back_populates="annotations")


class Translation(Base):
    """Scholarly or vernacular translation of an inscription."""
    __tablename__ = "translations"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    target_language = Column(String(16), default="en", index=True)  # en, ta
    translation_type = Column(String(64), default="SCHOLARLY")  # MODERN_PROSE, SCHOLARLY, COMMENTARY
    text_content = Column(Text, nullable=False)
    historical_commentary = Column(Text, default="")
    cultural_context = Column(Text, default="")
    translator_id = Column(String(64), ForeignKey("researchers.id"), nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    inscription = relationship("Inscription", back_populates="translations")


class HistoricalReference(Base):
    """Historical entities mentioned in an inscription (monarch, regnal year, dynasty)."""
    __tablename__ = "historical_references"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    ruler_name = Column(String(128), default="", index=True)
    dynasty = Column(String(64), default="", index=True)
    regnal_year = Column(String(32), default="")
    historical_date = Column(String(64), default="")
    historical_event = Column(Text, default="")

    # Relationships
    inscription = relationship("Inscription", back_populates="historical_references")


class Citation(Base):
    """Bibliographic source reference (Epigraphia Indica, SII, ARE)."""
    __tablename__ = "citations"

    id = Column(String(64), primary_key=True)
    inscription_id = Column(String(64), ForeignKey("inscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    publication_title = Column(String(256), nullable=False)
    volume = Column(String(32), default="")
    number = Column(String(32), default="")
    page_numbers = Column(String(64), default="")
    year = Column(Integer, nullable=True)
    external_url = Column(String(512), default="")
    doi = Column(String(128), default="")

    # Relationships
    inscription = relationship("Inscription", back_populates="citations")


class Review(Base):
    """Peer review and adjudication record by a qualified epigraphist."""
    __tablename__ = "reviews"

    id = Column(String(64), primary_key=True)
    transcription_id = Column(String(64), ForeignKey("transcriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id = Column(String(64), ForeignKey("researchers.id"), nullable=False, index=True)
    action = Column(Enum(ReviewAction), default=ReviewAction.ACCEPT, nullable=False)
    previous_reading = Column(Text, default="")
    approved_reading = Column(Text, default="")
    comments = Column(Text, default="")
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    transcription = relationship("Transcription", back_populates="reviews")
    reviewer = relationship("Researcher", back_populates="reviews")


class Revision(Base):
    """Versioned change record tracking modifications across research objects."""
    __tablename__ = "revisions"

    id = Column(String(64), primary_key=True)
    entity_type = Column(String(64), nullable=False, index=True)  # Inscription, Transcription, etc.
    entity_id = Column(String(64), nullable=False, index=True)
    previous_state = Column(JSON, nullable=True)
    new_state = Column(JSON, nullable=False)
    changed_by = Column(String(64), ForeignKey("researchers.id"), nullable=True)
    timestamp = Column(DateTime, default=_utc_now)


class AuditEvent(Base):
    """Append-only cryptographic event log for data integrity."""
    __tablename__ = "audit_events"

    id = Column(String(64), primary_key=True)
    event_type = Column(String(64), nullable=False, index=True)  # FUSION, ADJUDICATION, EXPORT
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(64), nullable=False)
    user_id = Column(String(64), default="system")
    sha256_signature = Column(String(64), nullable=False)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=_utc_now, index=True)


class ModelRun(Base):
    """Audit log of AI model inference execution."""
    __tablename__ = "model_runs"

    id = Column(String(64), primary_key=True)
    model_name = Column(String(64), nullable=False, index=True)
    model_version = Column(String(32), default="1.0.0")
    input_asset_id = Column(String(64), ForeignKey("image_assets.id"), nullable=True)
    execution_time_ms = Column(Float, default=0.0)
    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utc_now)


class DatasetVersion(Base):
    """Frozen snapshot of epigraphical dataset split."""
    __tablename__ = "dataset_versions"

    id = Column(String(64), primary_key=True)
    version_tag = Column(String(64), unique=True, nullable=False, index=True)
    script_category = Column(String(64), nullable=False)
    manifest_path = Column(String(512), nullable=False)
    sample_count = Column(Integer, default=0)
    is_frozen = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utc_now)


class KnowledgeEntity(Base):
    """Historical entity node in the knowledge graph."""
    __tablename__ = "knowledge_entities"

    id = Column(String(64), primary_key=True)
    entity_type = Column(Enum(EntityCategory), nullable=False, index=True)
    canonical_name = Column(String(128), nullable=False, index=True)
    canonical_name_ta = Column(String(128), default="")
    aliases = Column(JSON, default=list)
    attributes = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    source_relations = relationship("KnowledgeRelation", foreign_keys="KnowledgeRelation.source_id", back_populates="source")
    target_relations = relationship("KnowledgeRelation", foreign_keys="KnowledgeRelation.target_id", back_populates="target")


class KnowledgeRelation(Base):
    """Directed edge in the historical knowledge graph."""
    __tablename__ = "knowledge_relations"

    id = Column(String(64), primary_key=True)
    source_id = Column(String(64), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    target_id = Column(String(64), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    relation_type = Column(String(64), nullable=False, index=True)  # ISSUED, RULED_OVER, GRANTED_TO, DEDICATED_AT
    source_inscription_id = Column(String(64), ForeignKey("inscriptions.id"), nullable=True, index=True)
    confidence = Column(Float, default=1.0)
    is_ai_proposed = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=True)
    citation_id = Column(String(64), ForeignKey("citations.id"), nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    # Relationships
    source = relationship("KnowledgeEntity", foreign_keys=[source_id], back_populates="source_relations")
    target = relationship("KnowledgeEntity", foreign_keys=[target_id], back_populates="target_relations")
