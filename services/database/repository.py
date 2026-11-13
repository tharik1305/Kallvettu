"""
Repository Layer for Epigraphical Research Entities (Phase B).

Provides high-level transactional CRUD operations, search filters,
revision logging, and curated historical inscription seeding.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import or_, and_, desc
from sqlalchemy.orm import Session, joinedload

from services.database.connection import get_engine, get_db_session, Base
from services.database.models import (
    Inscription,
    ImageAsset,
    Transcription,
    ReadingCandidate,
    GlyphAnnotation,
    Translation,
    HistoricalReference,
    Citation,
    Researcher,
    Review,
    Revision,
    AuditEvent,
    ModelRun,
    DatasetVersion,
    KnowledgeEntity,
    KnowledgeRelation,
    AssetType,
    TranscriptionTier,
    AnnotationStatus,
    ResearcherRole,
    ReviewAction,
    EntityCategory,
)
from services.database.schemas import InscriptionSearchFilter

logger = logging.getLogger("kallvettu.repository")


class EpigraphicalRepository:
    """Transactional repository governing epigraphical persistence and research workflows."""

    @classmethod
    def init_database(cls) -> None:
        """Initializes database schema creating all required relational tables."""
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        cls.seed_default_researchers()
        logger.info("Epigraphical research database tables verified/created.")

    @classmethod
    def seed_default_researchers(cls) -> int:
        """Seeds default researchers/scholars into the database if not present."""
        from services.collaboration.auth_manager import DEFAULT_SCHOLAR_PROFILES
        seeded = 0
        with get_db_session() as s:
            for p in DEFAULT_SCHOLAR_PROFILES:
                existing = s.query(Researcher).filter(Researcher.id == p["id"]).first()
                if not existing:
                    r = Researcher(
                        id=p["id"],
                        username=p["username"],
                        email=f"{p['username']}@kallvettu.org",
                        full_name=p["full_name"],
                        role=p["role"],
                        institution=p["institution"],
                    )
                    s.add(r)
                    seeded += 1
            s.commit()
        return seeded

    # ----------------------------------------------------------------------
    # Inscription Operations
    # ----------------------------------------------------------------------

    @classmethod
    def create_inscription(cls, data: Dict[str, Any], session: Optional[Session] = None) -> Inscription:
        """Inserts a new inscription record."""
        def _op(s: Session) -> Inscription:
            insc_id = data.get("id") or f"KV-INSC-{uuid.uuid4().hex[:8].upper()}"
            insc = Inscription(
                id=insc_id,
                title_ta=data.get("title_ta", ""),
                title_en=data.get("title_en", ""),
                script_category=data.get("script_category", "Tamil-Brahmi"),
                historical_period=data.get("historical_period", ""),
                site_name=data.get("site_name", ""),
                district=data.get("district", ""),
                state=data.get("state", "Tamil Nadu"),
                material_type=data.get("material_type", "Granite Stone"),
                century=data.get("century", ""),
                ruler_name=data.get("ruler_name", ""),
                dynasty=data.get("dynasty", ""),
                regnal_year=data.get("regnal_year", ""),
                description=data.get("description", ""),
            )
            s.add(insc)
            s.flush()
            return insc

        if session is not None:
            return _op(session)
        with get_db_session() as s:
            return _op(s)

    @classmethod
    def get_inscription(cls, inscription_id: str, session: Optional[Session] = None) -> Optional[Inscription]:
        """Retrieves an inscription by ID with all eager-loaded assets, transcriptions, and citations."""
        def _op(s: Session) -> Optional[Inscription]:
            return (
                s.query(Inscription)
                .options(
                    joinedload(Inscription.image_assets),
                    joinedload(Inscription.transcriptions).joinedload(Transcription.reading_candidates),
                    joinedload(Inscription.translations),
                    joinedload(Inscription.historical_references),
                    joinedload(Inscription.citations),
                )
                .filter(Inscription.id == inscription_id)
                .first()
            )

        if session is not None:
            return _op(session)
        with get_db_session() as s:
            return _op(s)

    @classmethod
    def list_inscriptions(cls, limit: int = 50, offset: int = 0) -> List[Inscription]:
        """Lists inscriptions with pagination."""
        with get_db_session() as s:
            return s.query(Inscription).order_by(Inscription.id).offset(offset).limit(limit).all()

    @classmethod
    def count_inscriptions(cls) -> int:
        """Counts total inscriptions in database."""
        with get_db_session() as s:
            return s.query(Inscription).count()

    @classmethod
    def search_inscriptions(cls, filters: InscriptionSearchFilter) -> Tuple[List[Inscription], int]:
        """Multi-criteria search over inscriptions with Tamil Unicode and period filtering."""
        with get_db_session() as s:
            query = s.query(Inscription)

            if filters.query:
                term = f"%{filters.query.strip()}%"
                query = query.filter(
                    or_(
                        Inscription.title_ta.ilike(term),
                        Inscription.title_en.ilike(term),
                        Inscription.site_name.ilike(term),
                        Inscription.district.ilike(term),
                        Inscription.ruler_name.ilike(term),
                        Inscription.dynasty.ilike(term),
                        Inscription.description.ilike(term),
                        Inscription.id.ilike(term),
                    )
                )

            if filters.script_category:
                query = query.filter(Inscription.script_category == filters.script_category)
            if filters.district:
                query = query.filter(Inscription.district.ilike(f"%{filters.district}%"))
            if filters.dynasty:
                query = query.filter(Inscription.dynasty.ilike(f"%{filters.dynasty}%"))
            if filters.ruler_name:
                query = query.filter(Inscription.ruler_name.ilike(f"%{filters.ruler_name}%"))

            total = query.count()
            results = query.order_by(Inscription.id).offset(filters.offset).limit(filters.limit).all()
            return results, total

    # ----------------------------------------------------------------------
    # ImageAsset Operations
    # ----------------------------------------------------------------------

    @classmethod
    def add_image_asset(
        cls,
        inscription_id: str,
        file_path: str,
        sha256_hash: str,
        asset_type: AssetType = AssetType.RAW,
        width: int = 0,
        height: int = 0,
        mime_type: str = "image/jpeg",
        is_master: bool = False,
    ) -> ImageAsset:
        """Attaches an image asset to an inscription."""
        with get_db_session() as s:
            asset = ImageAsset(
                id=f"IMG-{uuid.uuid4().hex[:12].upper()}",
                inscription_id=inscription_id,
                asset_type=asset_type,
                file_path=file_path,
                sha256_hash=sha256_hash,
                width=width,
                height=height,
                mime_type=mime_type,
                is_master=is_master,
            )
            s.add(asset)
            s.flush()
            return asset

    # ----------------------------------------------------------------------
    # Transcription Operations
    # ----------------------------------------------------------------------

    @classmethod
    def add_transcription(
        cls,
        inscription_id: str,
        text_content: str,
        tier: TranscriptionTier = TranscriptionTier.DIPLOMATIC,
        author_id: Optional[str] = None,
        is_primary: bool = False,
        is_ai_generated: bool = False,
        provenance: str = "Manual Transcription",
    ) -> Transcription:
        """Stores a diplomatic, normalized, or segmented transcription reading."""
        with get_db_session() as s:
            tr = Transcription(
                id=f"TR-{uuid.uuid4().hex[:12].upper()}",
                inscription_id=inscription_id,
                tier=tier,
                text_content=text_content,
                author_id=author_id,
                is_primary=is_primary,
                is_ai_generated=is_ai_generated,
                provenance=provenance,
            )
            s.add(tr)
            s.flush()
            return tr

    # ----------------------------------------------------------------------
    # Translation Operations
    # ----------------------------------------------------------------------

    @classmethod
    def add_translation(
        cls,
        inscription_id: str,
        text_content: str,
        target_language: str = "en",
        translation_type: str = "SCHOLARLY",
        historical_commentary: str = "",
        cultural_context: str = "",
        translator_id: Optional[str] = None,
    ) -> Translation:
        """Stores modern translation and historical commentary."""
        with get_db_session() as s:
            tl = Translation(
                id=f"TL-{uuid.uuid4().hex[:12].upper()}",
                inscription_id=inscription_id,
                target_language=target_language,
                translation_type=translation_type,
                text_content=text_content,
                historical_commentary=historical_commentary,
                cultural_context=cultural_context,
                translator_id=translator_id,
            )
            s.add(tl)
            s.flush()
            return tl

    # ----------------------------------------------------------------------
    # Glyph Annotation Operations
    # ----------------------------------------------------------------------

    @classmethod
    def add_glyph_annotation(
        cls,
        inscription_id: str,
        image_asset_id: str,
        character_label: str,
        bounding_box: Dict[str, Any],
        polygon_coords: Optional[List[List[int]]] = None,
        unicode_char: str = "",
        diplomatic_reading: str = "",
        script_category: str = "Tamil-Brahmi",
        legibility_status: str = "CLEAR",
        review_status: AnnotationStatus = AnnotationStatus.UNREVIEWED,
        annotator_id: Optional[str] = None,
        notes: str = "",
    ) -> GlyphAnnotation:
        """Saves a character region annotation."""
        with get_db_session() as s:
            ann = GlyphAnnotation(
                id=f"GLYPH-{uuid.uuid4().hex[:12].upper()}",
                inscription_id=inscription_id,
                image_asset_id=image_asset_id,
                character_label=character_label,
                unicode_representation=unicode_char,
                diplomatic_reading=diplomatic_reading,
                script_category=script_category,
                bounding_box=bounding_box,
                polygon_coords=polygon_coords,
                legibility_status=legibility_status,
                review_status=review_status,
                annotator_id=annotator_id,
                notes=notes,
            )
            s.add(ann)
            s.flush()
            return ann

    @classmethod
    def list_annotations_for_inscription(cls, inscription_id: str) -> List[GlyphAnnotation]:
        """Retrieves all character annotations for an inscription."""
        with get_db_session() as s:
            return s.query(GlyphAnnotation).filter(GlyphAnnotation.inscription_id == inscription_id).all()

    # ----------------------------------------------------------------------
    # Researcher & Collaboration Operations
    # ----------------------------------------------------------------------

    @classmethod
    def get_or_create_researcher(
        cls,
        username: str,
        email: str,
        full_name: str,
        role: ResearcherRole = ResearcherRole.RESEARCHER,
        institution: str = "",
    ) -> Researcher:
        """Ensures an authenticated researcher profile exists."""
        with get_db_session() as s:
            user = s.query(Researcher).filter(Researcher.username == username).first()
            if not user:
                user = Researcher(
                    id=f"USR-{uuid.uuid4().hex[:8].upper()}",
                    username=username,
                    email=email,
                    full_name=full_name,
                    role=role,
                    institution=institution,
                )
                s.add(user)
                s.flush()
            return user

    @classmethod
    def submit_review(
        cls,
        transcription_id: str,
        reviewer_id: str,
        action: ReviewAction,
        approved_reading: str,
        previous_reading: str = "",
        comments: str = "",
    ) -> Review:
        """Records an epigraphist's peer review decision."""
        with get_db_session() as s:
            rev = Review(
                id=f"REV-{uuid.uuid4().hex[:12].upper()}",
                transcription_id=transcription_id,
                reviewer_id=reviewer_id,
                action=action,
                previous_reading=previous_reading,
                approved_reading=approved_reading,
                comments=comments,
            )
            s.add(rev)
            # Log append-only audit event
            audit_sig = hashlib.sha256(f"{transcription_id}:{reviewer_id}:{action}:{approved_reading}".encode("utf-8")).hexdigest()
            event = AuditEvent(
                id=f"AUDIT-{uuid.uuid4().hex[:12].upper()}",
                event_type="REVIEW_ADJUDICATION",
                entity_type="Transcription",
                entity_id=transcription_id,
                user_id=reviewer_id,
                sha256_signature=audit_sig,
                details={"action": str(action), "comments": comments},
            )
            s.add(event)
            s.flush()
            return rev

    # ----------------------------------------------------------------------
    # Curated Epigraphical Corpus Seeder
    # ----------------------------------------------------------------------

    @classmethod
    def seed_canonical_corpus(cls) -> int:
        """
        Seeds canonical ancient Tamil inscriptions from Kallvettu knowledge base
        into the persistent database if not already populated.
        """
        cls.init_database()
        canonical_plates = [
            {
                "id": "KV-INSC-001",
                "title_ta": "மாங்குளம் தமிழ்-பிராமி கல்வெட்டு",
                "title_en": "Mangulam Tamil-Brahmi Inscription",
                "script_category": "Tamil-Brahmi",
                "historical_period": "Early Sangam (c. 3rd–2nd c. BCE)",
                "site_name": "Mangulam / Meenakshipuram Cave",
                "district": "Madurai",
                "state": "Tamil Nadu",
                "material_type": "Natural Rock-shelter Cavern",
                "century": "3rd Century BCE",
                "ruler_name": "Pandyan Nedunjeliyan (நெடுஞ்செழியன்)",
                "dynasty": "Early Pandya Dynasty",
                "description": "Oldest securely dated Tamil-Brahmi inscription recording rock beds gifted to Jain monk Kaniyan Nanta-asiriyan.",
                "diplomatic": "கணிய் நந்தஸிரிகுவன் கள்வெள் அறைய் இத்த பாளிய்",
                "normalized": "கணியன் நந்தாசிரியருக்கு நெல்வேலி அறையன் அளித்த பள்ளிப்படுக்கை",
                "english": "The sacred stone bed caused to be made by the chieftain of Nelveli for the revered Jain ascetic Kaniyan Nanta-asiriyan.",
                "modern_ta": "கணியன் நந்தாசிரியருக்கு நெல்வேலி கிராமத்து அறையன் அமைத்துக் கொடுத்த குகைப் பள்ளிப் படுக்கை.",
                "citation": "Mahadevan, Iravatham. Early Tamil Epigraphy, Harvard University Press, 2003, Inscription No. 1.",
            },
            {
                "id": "KV-INSC-002",
                "title_ta": "தஞ்சைப் பெருவுடையார் கோயில் கற்றளிக் கல்வெட்டு",
                "title_en": "Thanjavur Brihadisvara Great Temple Inscription",
                "script_category": "Medieval Chola Tamil",
                "historical_period": "Imperial Chola (c. 1010 CE)",
                "site_name": "Brihadisvara Temple, Thanjavur",
                "district": "Thanjavur",
                "state": "Tamil Nadu",
                "material_type": "Granite Vimana Base Plinth",
                "century": "11th Century CE",
                "ruler_name": "Rajaraja Chola I (முதலாம் இராஜராஜ சோழன்)",
                "dynasty": "Chola Dynasty",
                "regnal_year": "26th Regnal Year",
                "description": "Imperial royal edict ordaining all royal and public donations to be permanently engraved on the granite stone temple walls.",
                "diplomatic": "நாம் குடுத்தனவும் நம் அக்கன் [குடுத்த]னவும் நம் பெண்டுகள் குடுத்தனவும் [மற்றா]ர் குடுத்தனவும் இத்திருக் கற்றளியிலே [கல்லி]லே வெட்டுக...",
                "normalized": "நாம் கொடுத்தனவும் நம் அக்கன் கொடுத்தனவும் நம் பெண்டுகள் கொடுத்தனவும் மற்றார் கொடுத்தனவும் இத்திருக்கற்றளியிலே கல்லிலே வெட்டுக என்று திருவாய் மொழிந்தருள",
                "english": "The gifts endowed by Us, by Our elder sister Kundavai, by Our queens, and by others—let them be chiseled into the stone of this sacred granite temple.",
                "modern_ta": "மன்னர் இராஜராஜ சோழன், தமக்கையார் குந்தவை பிராட்டியார், அரசியர் மற்றும் பொதுமக்கள் அளித்த கொடைகள் அனைத்தும் இந்தத் திருக்கற்றளியின் கல்சுவரில் செதுக்கப்பட வேண்டும் என்று ஆணையிட்டார்.",
                "citation": "South Indian Inscriptions (SII), Vol. II, Part I, No. 1.",
            },
            {
                "id": "KV-INSC-003",
                "title_ta": "உத்திரமேரூர் குடவோலை மக்களாட்சிச் சாசனம்",
                "title_en": "Uttiramerur Democratic Pot-Ballot Edict",
                "script_category": "Medieval Chola Tamil",
                "historical_period": "Early Medieval Chola (c. 921 CE)",
                "site_name": "Vaikunta Perumal Temple, Uttiramerur",
                "district": "Kanchipuram",
                "state": "Tamil Nadu",
                "material_type": "Granite Mandapa Wall",
                "century": "10th Century CE",
                "ruler_name": "Parantaka Chola I (முதலாம் பராந்தக சோழன்)",
                "dynasty": "Chola Dynasty",
                "regnal_year": "14th Regnal Year",
                "description": "World-famous constitutional inscription codifying local assembly qualifications, disqualifications, and the secret pot-ballot election system.",
                "diplomatic": "குடவோலை பறித்துப் பெயரெழுதிப் புகவிடுவோமாகவும் வாரி பெருமக்களாக...",
                "normalized": "குடவோலை பறித்துப் பெயர் எழுதிப் புகவிடுவோமாகவும்; தேர்ந்தெடுக்கப்படும் வாரியப் பெருமக்கள் தூய நிருவாகம் செய்வாராக",
                "english": "Palm leaves inscribed with candidates' names shall be deposited in an urn and drawn by an innocent boy to elect the village assembly councils.",
                "modern_ta": "குடத்தினுள் தகுதியுள்ளோரின் பெயர்களை ஓலைச் சீட்டுகளில் இட்டு சிறுவனைக்கொண்டு எடுத்து வாரியப் பெருமக்களைத் தேர்ந்தெடுக்கும் மக்களாட்சித் தேர்தல் முறை.",
                "citation": "Archaeological Survey of India, Epigraphia Indica, Vol. XXII, pp. 145–150.",
            },
            {
                "id": "KV-INSC-004",
                "title_ta": "புகழூர் சேர மன்னர் தமிழ்-பிராமி கல்வெட்டு",
                "title_en": "Pugalur Chera Lineage Tamil-Brahmi Epigraph",
                "script_category": "Tamil-Brahmi",
                "historical_period": "Sangam Era (c. 2nd c. CE)",
                "site_name": "Pugalur Hill (Arunattarmalai)",
                "district": "Karur",
                "state": "Tamil Nadu",
                "material_type": "Granite Cavern Bed",
                "century": "2nd Century CE",
                "ruler_name": "Chera King Ko Athan Cheral Irumporai",
                "dynasty": "Chera Dynasty",
                "description": "Sangam Chera inscription confirming three generations of Chera kings mentioned in the classical Tamil anthology Patirruppattu.",
                "diplomatic": "கோ ஆதன் செல்லிரும்பொறை மகன் பெருங்கடுங்கோ...",
                "normalized": "கோ ஆதன் சேரல் இரும்பொறை மகன் பெருங்கடுங்கோ அமைத்த அமணர் படுக்கை",
                "english": "The rock shelter made for Jain monks during the reign of the Chera prince Perunkatunko, son of King Athan Cheral Irumporai.",
                "modern_ta": "சேர மன்னன் கோ ஆதன் சேரல் இரும்பொறையின் மகன் பெருங்கடுங்கோ சமண முனிவர்களுக்காக வெட்டிய பாறைப் படுக்கை.",
                "citation": "Epigraphia Indica, Vol. XXXVIII, Inscriptions of Pugalur.",
            },
            {
                "id": "KV-INSC-005",
                "title_ta": "பூலாங்குறிச்சி முற்கால வட்டெழுத்து சாசனம்",
                "title_en": "Pulankurichi Early Vatteluttu Charter",
                "script_category": "Vatteluttu",
                "historical_period": "Transitional Era (c. 5th c. CE)",
                "site_name": "Pulankurichi Rock-cut Hill",
                "district": "Sivaganga",
                "state": "Tamil Nadu",
                "material_type": "Cliff Face Granite Boulder",
                "century": "5th Century CE",
                "ruler_name": "Chendan Kutran (சேந்தன் கூற்றன்)",
                "dynasty": "Kalabhra / Transitional Chieftains",
                "description": "Landmark transitional epigraph bridging Tamil-Brahmi and archaic Vatteluttu with administrative land grant terminology.",
                "diplomatic": "கோச்சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது...",
                "normalized": "கோச்சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது நிலக்கொடை ஆவணம்",
                "english": "In the 27th regnal year of King Chendan Kutran: comprehensive royal charter of village boundaries and irrigation maintenance.",
                "modern_ta": "மன்னன் சேந்தன் கூற்றனின் 27-வது ஆட்சியாண்டில் வழங்கப்பட்ட நிலக்கொடை மற்றும் ஏரி நீர்ப்பாசன நிருவாக ஆவணம்.",
                "citation": "Avanam: Journal of Tamil Nadu Archaeological Society, Vol. 1.",
            },
            {
                "id": "KV-INSC-006",
                "title_ta": "செங்கம் பல்லவர் ஆநிரை நடுகல்",
                "title_en": "Chengam Pallava Cattle-Raid Hero Stone",
                "script_category": "Vatteluttu",
                "historical_period": "Pallava Era (c. 7th c. CE)",
                "site_name": "Chengam Valley",
                "district": "Tiruvannamalai",
                "state": "Tamil Nadu",
                "material_type": "Carved Granite Stele",
                "century": "7th Century CE",
                "ruler_name": "Mahendravarman I / Narasimhavarman",
                "dynasty": "Pallava Dynasty",
                "description": "Commemorative memorial stone depicting a fallen warrior defending the village bovine cattle herd (ஆநிரை மீட்ட பூசல்).",
                "diplomatic": "தொறுவொடு விழந்த பட்டன் நடுகல்...",
                "normalized": "தொறுவோடு விழுந்த பட்டான் நடுகல்; ஆநிரை மீட்ட பூசலில் உயிர் துறந்த வீரனின் நினைவுக்கல்",
                "english": "Hero stone erected for the valiant warrior who laid down his life in battle retrieving the community cattle herd.",
                "modern_ta": "பகைவர்களால் கவரப்பட்ட ஊரின் பசுக்கூட்டங்களை மீட்கும் போரில் வீரமரணம் எய்திய வீரனுக்கு நடப்பட்ட நடுகல்.",
                "citation": "Chengam Nadukarkal, Tamil Nadu State Department of Archaeology, 1972.",
            },
        ]

        seeded_count = 0
        with get_db_session() as s:
            for p in canonical_plates:
                existing = s.query(Inscription).filter(Inscription.id == p["id"]).first()
                if not existing:
                    insc = Inscription(
                        id=p["id"],
                        title_ta=p["title_ta"],
                        title_en=p["title_en"],
                        script_category=p["script_category"],
                        historical_period=p["historical_period"],
                        site_name=p["site_name"],
                        district=p["district"],
                        state=p["state"],
                        material_type=p["material_type"],
                        century=p["century"],
                        ruler_name=p.get("ruler_name", ""),
                        dynasty=p.get("dynasty", ""),
                        regnal_year=p.get("regnal_year", ""),
                        description=p["description"],
                    )
                    s.add(insc)
                    s.flush()

                    # Add transcriptions
                    s.add(Transcription(
                        id=f"TR-DIP-{p['id']}",
                        inscription_id=p["id"],
                        tier=TranscriptionTier.DIPLOMATIC,
                        text_content=p["diplomatic"],
                        is_primary=True,
                        provenance="Curated Epigraphical Catalog",
                    ))
                    s.add(Transcription(
                        id=f"TR-NORM-{p['id']}",
                        inscription_id=p["id"],
                        tier=TranscriptionTier.NORMALIZED,
                        text_content=p["normalized"],
                        is_primary=False,
                        provenance="Standardized Epigraphical Orthography",
                    ))

                    # Add translations
                    s.add(Translation(
                        id=f"TL-EN-{p['id']}",
                        inscription_id=p["id"],
                        target_language="en",
                        translation_type="SCHOLARLY",
                        text_content=p["english"],
                        historical_commentary=p["description"],
                    ))
                    s.add(Translation(
                        id=f"TL-TA-{p['id']}",
                        inscription_id=p["id"],
                        target_language="ta",
                        translation_type="MODERN_PROSE",
                        text_content=p["modern_ta"],
                    ))

                    # Add Citation
                    s.add(Citation(
                        id=f"CIT-{p['id']}",
                        inscription_id=p["id"],
                        publication_title=p["citation"],
                    ))

                    # Add Historical Reference
                    if p.get("ruler_name"):
                        s.add(HistoricalReference(
                            id=f"HR-{p['id']}",
                            inscription_id=p["id"],
                            ruler_name=p["ruler_name"],
                            dynasty=p.get("dynasty", ""),
                            regnal_year=p.get("regnal_year", ""),
                            historical_date=p.get("century", ""),
                        ))

                    seeded_count += 1
            s.commit()
        logger.info("Canonical epigraphical corpus seeding complete. Seeded: %d records.", seeded_count)
        return seeded_count
