"""
Historical Epigraphical Inscription Search Engine (Phase H).

Provides multi-criteria scholarly search across:
- Inscription titles, site locations, districts, and material types.
- Diplomatic and normalized transcriptions.
- Scholarly English translations and modern Tamil prose.
- Historical rulers, dynasties, and regnal periods.
- Published citations (SII, Epigraphia Indica, Early Tamil Epigraphy).

Features:
- Unicode NFC normalization.
- Epigraphical morphological variant expansion.
- Weighted relevance scoring.
- Highlighted context snippets.
- Faceted aggregation counts (by script, dynasty, district).
"""

from __future__ import annotations

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from sqlalchemy import or_, and_, desc
from sqlalchemy.orm import Session, joinedload

from services.database.connection import get_db_session
from services.database.models import (
    Inscription,
    Transcription,
    Translation,
    HistoricalReference,
    Citation,
    TranscriptionTier,
)

logger = logging.getLogger("kallvettu.search")


@dataclass
class SearchResultItem:
    """Individual search result item with metadata and matched snippets."""
    inscription_id: str
    title_ta: str
    title_en: str
    script_category: str
    historical_period: str
    site_name: str
    district: str
    dynasty: str
    ruler_name: str
    century: str
    material_type: str
    relevance_score: float
    matching_snippets: List[str]
    master_reading: str
    english_translation: str
    citation: str


@dataclass
class SearchResult:
    """Aggregated search result payload."""
    query: str
    total_count: int
    expanded_terms: List[str]
    items: List[SearchResultItem]
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)


class EpigraphicalSearchEngine:
    """Multi-criteria search engine for epigraphical research database."""

    @classmethod
    def normalize_query(cls, text: str) -> str:
        """Normalizes Unicode text to NFC format and strips excess spaces."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFC", str(text).strip())
        return re.sub(r"\s+", " ", norm)

    @classmethod
    def expand_morphological_variants(cls, query: str) -> List[str]:
        """
        Expands epigraphical and classical Tamil root keywords into common
        inflectional, sandhi, and orthographic variants.
        """
        q = cls.normalize_query(query)
        if not q:
            return []

        variants = {q}
        tokens = q.split(" ")

        for tok in tokens:
            variants.add(tok)
            # Common epigraphical root expansions
            if "ராஜ" in tok or "ராச" in tok:
                variants.update(["இராஜராஜ", "ராசராச", "கோப்பரகேஸரி", "இராஜேந்திர"])
            if "பள்ளி" in tok or "படுக்கை" in tok:
                variants.update(["பள்ளிப்படுக்கை", "பாளிய்", "அமணர்", "சமணர்"])
            if "குடவோலை" in tok or "தேர்தல்" in tok:
                variants.update(["குடவோலை", "வாரியம்", "உத்திரமேரூர்", "சபை"])
            if "சேர" in tok or "இரும்பொறை" in tok:
                variants.update(["கோ ஆதன்", "இரும்பொறை", "புகழூர்"])
            if "பாண்டிய" in tok or "நெடுஞ்செழியன்" in tok:
                variants.update(["நெடுஞ்செழியன்", "மாங்குளம்", "வழுதி"])
            if "பல்லவ" in tok or "மகாபலிபுரம்" in tok:
                variants.update(["நரசிம்மவர்மன்", "மாமல்ல", "பல்லவர்"])

        return sorted(list(variants))

    @classmethod
    def _create_snippet(cls, text: str, matched_terms: List[str], max_len: int = 140) -> Optional[str]:
        """Extracts context snippet with highlighted <mark> tags around matches."""
        if not text:
            return None

        lower_text = text.lower()
        earliest_pos = -1
        best_term = ""

        for term in matched_terms:
            pos = lower_text.find(term.lower())
            if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                earliest_pos = pos
                best_term = term

        if earliest_pos == -1:
            return None

        start = max(0, earliest_pos - 40)
        end = min(len(text), earliest_pos + len(best_term) + 60)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        # Highlight term
        pattern = re.compile(re.escape(best_term), re.IGNORECASE)
        snippet = pattern.sub(f"<mark>{best_term}</mark>", snippet)
        return snippet

    @classmethod
    def search(
        cls,
        query: str = "",
        script_category: Optional[str] = None,
        dynasty: Optional[str] = None,
        ruler_name: Optional[str] = None,
        district: Optional[str] = None,
        century: Optional[str] = None,
        material_type: Optional[str] = None,
        tier: Optional[str] = None,
        expand_morphology: bool = True,
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResult:
        """
        Executes multi-criteria search over epigraphical corpus.
        """
        clean_q = cls.normalize_query(query)
        expanded_terms = cls.expand_morphological_variants(clean_q) if (expand_morphology and clean_q) else ([clean_q] if clean_q else [])

        with get_db_session() as s:
            q_builder = s.query(Inscription).options(
                joinedload(Inscription.transcriptions),
                joinedload(Inscription.translations),
                joinedload(Inscription.citations),
                joinedload(Inscription.historical_references),
            )

            # Metadata filters
            if script_category and script_category != "ALL":
                q_builder = q_builder.filter(Inscription.script_category.ilike(f"%{script_category}%"))
            if dynasty and dynasty != "ALL":
                q_builder = q_builder.filter(Inscription.dynasty.ilike(f"%{dynasty}%"))
            if ruler_name:
                q_builder = q_builder.filter(Inscription.ruler_name.ilike(f"%{ruler_name}%"))
            if district and district != "ALL":
                q_builder = q_builder.filter(Inscription.district.ilike(f"%{district}%"))
            if century and century != "ALL":
                q_builder = q_builder.filter(Inscription.century.ilike(f"%{century}%"))
            if material_type and material_type != "ALL":
                q_builder = q_builder.filter(Inscription.material_type.ilike(f"%{material_type}%"))

            all_records = q_builder.all()

            # Rank and score results based on text query matches
            scored_items: List[Tuple[float, SearchResultItem]] = []
            facet_scripts: Dict[str, int] = {}
            facet_dynasties: Dict[str, int] = {}
            facet_districts: Dict[str, int] = {}

            for rec in all_records:
                # Update facets
                if rec.script_category:
                    facet_scripts[rec.script_category] = facet_scripts.get(rec.script_category, 0) + 1
                if rec.dynasty:
                    facet_dynasties[rec.dynasty] = facet_dynasties.get(rec.dynasty, 0) + 1
                if rec.district:
                    facet_districts[rec.district] = facet_districts.get(rec.district, 0) + 1

                score = 0.0
                snippets: List[str] = []

                # Find primary reading
                master_reading = ""
                for tr in rec.transcriptions:
                    if tr.is_primary or not master_reading:
                        master_reading = tr.text_content

                # Find primary translation
                en_translation = ""
                for tl in rec.translations:
                    if tl.target_language == "en" and not en_translation:
                        en_translation = tl.text_content

                # Find citation
                citation_text = ""
                if rec.citations:
                    citation_text = rec.citations[0].publication_title

                if not clean_q:
                    # No text query: base score
                    score = 1.0
                else:
                    # Score text occurrences
                    for term in expanded_terms:
                        term_l = term.lower()

                        # Inscription Titles (Weight: 10.0)
                        if term_l in rec.title_ta.lower() or term_l in rec.title_en.lower():
                            score += 10.0
                            snip = cls._create_snippet(rec.title_en + " - " + rec.title_ta, [term])
                            if snip and snip not in snippets:
                                snippets.append(snip)

                        # Ruler / Dynasty (Weight: 8.0)
                        if term_l in (rec.ruler_name or "").lower() or term_l in (rec.dynasty or "").lower():
                            score += 8.0
                            snip = cls._create_snippet(f"{rec.ruler_name} ({rec.dynasty})", [term])
                            if snip and snip not in snippets:
                                snippets.append(snip)

                        # Transcriptions (Weight: 6.0)
                        for tr in rec.transcriptions:
                            if tier and tr.tier.value != tier:
                                continue
                            if term_l in tr.text_content.lower():
                                score += 6.0
                                snip = cls._create_snippet(tr.text_content, [term])
                                if snip and snip not in snippets:
                                    snippets.append(snip)

                        # Translations & Descriptions (Weight: 4.0)
                        for tl in rec.translations:
                            if term_l in tl.text_content.lower():
                                score += 4.0
                                snip = cls._create_snippet(tl.text_content, [term])
                                if snip and snip not in snippets:
                                    snippets.append(snip)

                        if term_l in (rec.description or "").lower():
                            score += 3.0
                            snip = cls._create_snippet(rec.description, [term])
                            if snip and snip not in snippets:
                                snippets.append(snip)

                        # Citations (Weight: 2.0)
                        for cit in rec.citations:
                            if term_l in cit.publication_title.lower():
                                score += 2.0
                                snip = cls._create_snippet(cit.publication_title, [term])
                                if snip and snip not in snippets:
                                    snippets.append(snip)

                if score > 0.0 or not clean_q:
                    item = SearchResultItem(
                        inscription_id=rec.id,
                        title_ta=rec.title_ta,
                        title_en=rec.title_en,
                        script_category=rec.script_category,
                        historical_period=rec.historical_period or "",
                        site_name=rec.site_name or "",
                        district=rec.district or "",
                        dynasty=rec.dynasty or "",
                        ruler_name=rec.ruler_name or "",
                        century=rec.century or "",
                        material_type=rec.material_type or "",
                        relevance_score=round(score, 2),
                        matching_snippets=snippets[:3],
                        master_reading=master_reading,
                        english_translation=en_translation,
                        citation=citation_text,
                    )
                    scored_items.append((score, item))

            # Sort by score descending
            scored_items.sort(key=lambda x: x[0], reverse=True)
            total_count = len(scored_items)

            # Pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_items = [x[1] for x in scored_items[start_idx:end_idx]]

            return SearchResult(
                query=clean_q,
                total_count=total_count,
                expanded_terms=expanded_terms,
                items=paged_items,
                facets={
                    "scripts": facet_scripts,
                    "dynasties": facet_dynasties,
                    "districts": facet_districts,
                },
            )

