"""
SQLite Persistence & Epigraphical Atlas Repository Layer.

Features:
- Versioned SQLite schema (`PRAGMA user_version`) with automatic non-destructive migrations
- Separate storage for Discovery, Collection, Current Repository, and Photograph Capture locations
- Audit history log (`inscription_location_audit`) for provenance changes
- Multi-facet filtering & district/state aggregation queries
- Idempotent seeding from curated Kallvettu historical inscriptions WITHOUT
  inventing coordinates for district-only or regional inscriptions
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from services.geo_atlas.models import (
    GeoInscriptionRecord,
    LocationContext,
    LocationPrecision,
    AtlasFilterCriteria,
)
from services.geo_atlas.geo_loader import normalize_state_name, normalize_district_name
from services.geo_atlas.validation import validate_record_provenance

DEFAULT_ATLAS_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "kallvettu_atlas.db")
)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS atlas_inscriptions (
    inscription_id TEXT PRIMARY KEY,
    title_en TEXT NOT NULL,
    title_ta TEXT DEFAULT '',
    script TEXT DEFAULT '',
    period TEXT DEFAULT '',
    dynasty TEXT DEFAULT '',
    material TEXT DEFAULT 'Granite / Stone',

    country TEXT DEFAULT 'India',
    state TEXT DEFAULT '',
    state_code TEXT DEFAULT '',
    district TEXT DEFAULT '',
    district_code TEXT DEFAULT '',
    historical_district_name TEXT DEFAULT '',
    sub_district_taluk TEXT DEFAULT '',
    village TEXT DEFAULT '',
    archaeological_site TEXT DEFAULT '',

    latitude REAL NULL,
    longitude REAL NULL,
    location_precision TEXT NOT NULL DEFAULT 'unknown',

    discovery_source TEXT DEFAULT '',
    collection_source TEXT DEFAULT '',
    current_repository TEXT DEFAULT '',
    publication_reference TEXT DEFAULT '',
    coordinate_source TEXT DEFAULT '',

    discovery_location_json TEXT DEFAULT '{}',
    collection_location_json TEXT DEFAULT '{}',
    current_repository_location_json TEXT DEFAULT '{}',
    photograph_capture_location_json TEXT DEFAULT '{}',

    transcription_status TEXT DEFAULT 'Reference Curated',
    transcription_preview TEXT DEFAULT '',
    translation_preview TEXT DEFAULT '',
    image_path TEXT DEFAULT '',
    preset_id TEXT DEFAULT '',
    is_curated_preset INTEGER DEFAULT 0,
    spatial_validation_notes TEXT DEFAULT '',
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_atlas_state_district
    ON atlas_inscriptions (state, district);
CREATE INDEX IF NOT EXISTS idx_atlas_script
    ON atlas_inscriptions (script);
CREATE INDEX IF NOT EXISTS idx_atlas_precision
    ON atlas_inscriptions (location_precision);

CREATE TABLE IF NOT EXISTS inscription_location_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inscription_id TEXT NOT NULL,
    action TEXT NOT NULL,
    changed_by TEXT DEFAULT 'Researcher',
    snapshot_json TEXT NOT NULL,
    notes TEXT DEFAULT '',
    recorded_utc TEXT NOT NULL
);
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row: sqlite3.Row) -> GeoInscriptionRecord:
    return GeoInscriptionRecord(
        inscription_id=row["inscription_id"],
        title_en=row["title_en"],
        title_ta=row["title_ta"] or "",
        script=row["script"] or "",
        period=row["period"] or "",
        dynasty=row["dynasty"] or "",
        material=row["material"] or "",
        country=row["country"] or "India",
        state=row["state"] or "",
        state_code=row["state_code"] or "",
        district=row["district"] or "",
        district_code=row["district_code"] or "",
        historical_district_name=row["historical_district_name"] or "",
        sub_district_taluk=row["sub_district_taluk"] or "",
        village=row["village"] or "",
        archaeological_site=row["archaeological_site"] or "",
        latitude=row["latitude"],
        longitude=row["longitude"],
        location_precision=row["location_precision"] or LocationPrecision.UNKNOWN.value,
        discovery_source=row["discovery_source"] or "",
        collection_source=row["collection_source"] or "",
        current_repository=row["current_repository"] or "",
        publication_reference=row["publication_reference"] or "",
        coordinate_source=row["coordinate_source"] or "",
        discovery_location=LocationContext.from_dict(json.loads(row["discovery_location_json"] or "{}")),
        collection_location=LocationContext.from_dict(json.loads(row["collection_location_json"] or "{}")),
        current_repository_location=LocationContext.from_dict(
            json.loads(row["current_repository_location_json"] or "{}")
        ),
        photograph_capture_location=LocationContext.from_dict(
            json.loads(row["photograph_capture_location_json"] or "{}")
        ),
        transcription_status=row["transcription_status"] or "",
        transcription_preview=row["transcription_preview"] or "",
        translation_preview=row["translation_preview"] or "",
        image_path=row["image_path"] or "",
        preset_id=row["preset_id"] or "",
        is_curated_preset=bool(row["is_curated_preset"]),
        spatial_validation_notes=row["spatial_validation_notes"] or "",
        created_utc=row["created_utc"] or "",
        updated_utc=row["updated_utc"] or "",
    )


def build_curated_seed_records() -> List[GeoInscriptionRecord]:
    """
    Constructs initial documented South Indian epigraphical records linked to
    Kallvettu's curated historical presets & sample imagery.

    CRITICAL PROVENANCE RULE:
    - Five specific ASI monuments with verified published coordinates use exact/approximate
      coordinates validated inside their respective GeoJSON districts.
    - 'hero_stone_nadukal' has verified taluk/district (Chengam, Tiruvannamalai) without a
      single fixed monument coordinate, so it is strictly 'district_only' with lat=None, lon=None.
    - 'chola_arulmoli' represents a general Chola heartland temple adhisthana plate with no fixed
      single site/district, so it is strictly 'unknown' with lat=None, lon=None.
    """
    from nlp_translation_engine import KNOWN_INSCRIPTIONS_CATALOG

    preset_lookup = {p["id"]: p for p in KNOWN_INSCRIPTIONS_CATALOG}

    def _p(pid: str) -> Dict[str, Any]:
        return preset_lookup.get(pid, {})

    seeds = [
        GeoInscriptionRecord(
            inscription_id="TB_MGL_001",
            preset_id="mangulam_brahmi",
            is_curated_preset=True,
            title_en=_p("mangulam_brahmi").get("title_en", "Mangulam Tamil-Brahmi Inscription"),
            title_ta=_p("mangulam_brahmi").get("title_ta", "மாங்குளம் தமிழ்ப் பிராமி கல்வெட்டு"),
            script="Tamil-Brahmi",
            period="c. 3rd–2nd Century BCE (Sangam Period)",
            dynasty="Early Pandya (Nedunchezhiyan)",
            material="Natural Granite Rock-Shelter Drip-Ledge",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Madurai",
            district_code="623",
            historical_district_name="Madurai",
            sub_district_taluk="Melur / Madurai East",
            village="Mangulam (Meenakshipuram)",
            archaeological_site="Ovamalai (Kalugumalai) Jain Cavern Shelter",
            latitude=10.01904,
            longitude=78.25367,
            location_precision=LocationPrecision.APPROXIMATE.value,
            discovery_source="Robert Sewell (1882) / V. Venkayya & H. Krishna Sastri (1906–1919)",
            collection_source="ASI Epigraphy Branch & State Dept. of Archaeology Documentation",
            current_repository="In-situ Protected ASI Monument (Kalugumalai Hill, Mangulam)",
            publication_reference="ASI South Indian Inscriptions / Epigraphia Indica / I. Mahadevan Early Tamil Epigraphy #1–6; Mendeley 2spz2js9jp.1",
            coordinate_source="Published ASI monument locality (Ovamalai Hill near Mangulam)",
            discovery_location=LocationContext(
                place_name="Ovamalai Rock Shelter, Mangulam",
                institution_or_site="Cave No. 1 Drip-Ledge",
                state="Tamil Nadu",
                district="Madurai",
                taluk_or_city="Madurai East",
                village="Mangulam",
                latitude=10.01904,
                longitude=78.25367,
                notes="Earliest deciphered Tamil-Brahmi cave inscriptions mentioning Nedunchezhiyan.",
            ),
            collection_location=LocationContext(
                place_name="ASI Southern Circle Field Survey (Estampage Collection)",
                institution_or_site="Archaeological Survey of India, Epigraphy Branch",
                state="Karnataka",
                district="Mysuru",
                taluk_or_city="Mysuru",
                notes="Paper estampages archived in Office of the Chief Epigraphist.",
            ),
            current_repository_location=LocationContext(
                place_name="Mangulam Protected Hill Monument",
                institution_or_site="ASI Chennai Circle Monument Sanctuary",
                state="Tamil Nadu",
                district="Madurai",
                village="Mangulam",
                latitude=10.01904,
                longitude=78.25367,
                notes="Original lithic text remains in-situ on granite brow.",
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (mangulam.jpg)",
                institution_or_site="Ovamalai Hilltop Field Capture / Reference Plate",
                state="Tamil Nadu",
                district="Madurai",
                notes="Digital reference image stored locally in assets/samples/mangulam.jpg.",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("mangulam_brahmi").get("diplomatic_text", ""),
            translation_preview=_p("mangulam_brahmi").get("english_translation", ""),
            image_path="assets/samples/mangulam.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="TB_PGL_001",
            preset_id="pugalur_chera",
            is_curated_preset=True,
            title_en=_p("pugalur_chera").get("title_en", "Pugalur Chera Tamil-Brahmi Inscription"),
            title_ta=_p("pugalur_chera").get("title_ta", "புகழூர் சேரர் தமிழ்ப் பிராமி கல்வெட்டு"),
            script="Tamil-Brahmi",
            period="c. 1st–2nd Century CE (Late Sangam)",
            dynasty="Chera Dynasty (Irumporai Lineage)",
            material="Granite Hillock Cavern Drip-Ledge & Rock Beds",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Karur",
            district_code="613",
            historical_district_name="Tiruchirappalli (Pre-1995) / Karur",
            sub_district_taluk="Manmangalam / Pugalur",
            village="Velayuthampalayam (Pugalur)",
            archaeological_site="Arunattarmalai (Arunattar Hill) Rock Shelters",
            latitude=11.0830,
            longitude=78.0170,
            location_precision=LocationPrecision.APPROXIMATE.value,
            discovery_source="ASI Annual Report on South Indian Epigraphy (1927–28, ARIE)",
            collection_source="ASI Epigraphy Survey / Iravatham Mahadevan Corpus",
            current_repository="In-situ Protected Monument, Tamil Nadu State Dept. of Archaeology / ASI",
            publication_reference="ASI Epigraphia Indica Vol. VIII / ARIE 1927-28 Nos. 201–212; Mahadevan Corpus #56–67",
            coordinate_source="Arunattarmalai / Pugalur locality reference",
            discovery_location=LocationContext(
                place_name="Arunattarmalai Cavern Beds, Pugalur",
                institution_or_site="Arunattar Malai Hillock",
                state="Tamil Nadu",
                district="Karur",
                taluk_or_city="Pugalur",
                village="Velayuthampalayam",
                latitude=11.0830,
                longitude=78.0170,
                notes="Identifies three generations of Chera Irumporai kings corroborated by Patirruppattu.",
            ),
            collection_location=LocationContext(
                place_name="ARIE Estampage Registry",
                institution_or_site="ASI Epigraphy Division",
                state="Tamil Nadu",
                district="Chennai",
                notes="Catalogued during 1927–1928 epigraphical field season.",
            ),
            current_repository_location=LocationContext(
                place_name="Arunattarmalai In-situ Rock Shelter",
                institution_or_site="Protected Archaeological Hillock, Pugalur",
                state="Tamil Nadu",
                district="Karur",
                latitude=11.0830,
                longitude=78.0170,
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (pugalur.jpg)",
                institution_or_site="Pugalur Sample Plate",
                state="Tamil Nadu",
                district="Karur",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("pugalur_chera").get("diplomatic_text", ""),
            translation_preview=_p("pugalur_chera").get("english_translation", ""),
            image_path="assets/samples/pugalur.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="VT_PLK_001",
            preset_id="pulankurichi_vatteluttu",
            is_curated_preset=True,
            title_en=_p("pulankurichi_vatteluttu").get("title_en", "Pulankurichi Early Vatteluttu Inscription"),
            title_ta=_p("pulankurichi_vatteluttu").get("title_ta", "பூலாங்குறிச்சி வட்டெழுத்துக் கல்வெட்டு"),
            script="Vatteluttu",
            period="c. 5th Century CE (442–500 CE)",
            dynasty="Kalabhra / Ko-Chendan Kutran",
            material="Exposed Open Granite Sheet Rock",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Sivaganga",
            district_code="622",
            historical_district_name="Sivagangai / Pasumpon / Ramanathapuram",
            sub_district_taluk="Tiruppathur / Singampunari",
            village="Pulankurichi",
            archaeological_site="Pulankurichi Hillock Rock Sheet Charter",
            latitude=10.2699,
            longitude=78.5880,
            location_precision=LocationPrecision.APPROXIMATE.value,
            discovery_source="Tamil Nadu State Department of Archaeology (1979)",
            collection_source="State Dept. of Archaeology, Chennai (Avanam / Kalvettu Journal)",
            current_repository="In-situ Protected Monument, Tamil Nadu State Dept. of Archaeology",
            publication_reference="TN State Dept. of Archaeology (R. Nagaswamy / Y. Subbarayalu, Avanam)",
            coordinate_source="Pulankurichi village / rock outcrop locality",
            discovery_location=LocationContext(
                place_name="Pulankurichi Rock Face, Singampunari-Tiruppathur belt",
                institution_or_site="Pulankurichi Sheet Rock",
                state="Tamil Nadu",
                district="Sivaganga",
                taluk_or_city="Tiruppathur",
                village="Pulankurichi",
                latitude=10.2699,
                longitude=78.5880,
            ),
            collection_location=LocationContext(
                place_name="Tamil Nadu State Dept. of Archaeology Archive",
                institution_or_site="TNSDA Epigraphy Wing, Egmore",
                state="Tamil Nadu",
                district="Chennai",
            ),
            current_repository_location=LocationContext(
                place_name="Pulankurichi In-situ Granite Outcrop",
                institution_or_site="TNSDA Protected Site, Pulankurichi",
                state="Tamil Nadu",
                district="Sivaganga",
                latitude=10.2699,
                longitude=78.5880,
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (pulankurichi.jpg)",
                institution_or_site="Pulankurichi Sample Plate",
                state="Tamil Nadu",
                district="Sivaganga",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("pulankurichi_vatteluttu").get("diplomatic_text", ""),
            translation_preview=_p("pulankurichi_vatteluttu").get("english_translation", ""),
            image_path="assets/samples/pulankurichi.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="MT_TNJ_001",
            preset_id="thanjavur_rajaraja",
            is_curated_preset=True,
            title_en=_p("thanjavur_rajaraja").get("title_en", "Thanjavur Brihadisvara Temple Rajaraja Chola Inscription"),
            title_ta=_p("thanjavur_rajaraja").get("title_ta", "தஞ்சை பெரிய கோயில் இராஜராஜ சோழன் கல்வெட்டு"),
            script="Medieval Tamil",
            period="1010–1014 CE (26th–29th Regnal Year)",
            dynasty="Imperial Chola (Rajaraja I)",
            material="Cut Granite Adhisthana Moulding Wall",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Thanjavur",
            district_code="620",
            historical_district_name="Tanjore / Thanjavur",
            sub_district_taluk="Thanjavur",
            village="Thanjavur",
            archaeological_site="Brihadisvara Temple (Rajarajesvaram), North Vimana Adhisthana",
            latitude=10.7828,
            longitude=79.1318,
            location_precision=LocationPrecision.VERIFIED.value,
            discovery_source="E. Hultzsch (1891), ASI Epigraphy Survey",
            collection_source="ASI South Indian Inscriptions Vol. II (Great Living Chola Temples)",
            current_repository="Brihadisvara Temple UNESCO World Heritage Monument (In-situ)",
            publication_reference="South Indian Inscriptions (SII) Vol. II, Parts I–V (ASI)",
            coordinate_source="Verified Brihadisvara Temple complex coordinates",
            discovery_location=LocationContext(
                place_name="North Adhisthana Basement Wall, Brihadisvara Temple",
                institution_or_site="Rajarajesvaram Temple",
                state="Tamil Nadu",
                district="Thanjavur",
                taluk_or_city="Thanjavur",
                latitude=10.7828,
                longitude=79.1318,
            ),
            collection_location=LocationContext(
                place_name="South Indian Inscriptions Vol. II Compilation",
                institution_or_site="ASI Epigraphy Archive",
                state="Karnataka",
                district="Mysuru",
            ),
            current_repository_location=LocationContext(
                place_name="Brihadisvara Temple Complex",
                institution_or_site="ASI Trichy Circle / Hindu Religious & Charitable Endowments",
                state="Tamil Nadu",
                district="Thanjavur",
                latitude=10.7828,
                longitude=79.1318,
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (thanjavur.jpg)",
                institution_or_site="Brihadisvara Temple Granite Wall Squeeze",
                state="Tamil Nadu",
                district="Thanjavur",
                latitude=10.7828,
                longitude=79.1318,
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("thanjavur_rajaraja").get("diplomatic_text", ""),
            translation_preview=_p("thanjavur_rajaraja").get("english_translation", ""),
            image_path="assets/samples/thanjavur.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="MT_UTM_001",
            preset_id="uttiramerur_election",
            is_curated_preset=True,
            title_en=_p("uttiramerur_election").get("title_en", "Uttiramerur Kudavolai Democratic Election Inscription"),
            title_ta=_p("uttiramerur_election").get("title_ta", "உத்திரமேரூர் குடவோலை தேர்தல் கல்வெட்டு"),
            script="Medieval Tamil",
            period="919–921 CE (12th & 14th Regnal Years of Parantaka I)",
            dynasty="Imperial Chola (Parantaka I)",
            material="Granite Temple Mandapa Wall",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Kancheepuram",
            district_code="604",
            historical_district_name="Chingleput / Kanchipuram",
            sub_district_taluk="Uttiramerur",
            village="Uttiramerur",
            archaeological_site="Vaikunta Perumal Temple (Sabha Mandapa Wall), Uttiramerur",
            latitude=12.6160,
            longitude=79.7530,
            location_precision=LocationPrecision.APPROXIMATE.value,
            discovery_source="V. Venkayya (1898–1904) / ASI Annual Report 1904–05",
            collection_source="ASI Epigraphy Survey & SII Vol. III / Epigraphia Indica Vol. XXII",
            current_repository="In-situ Protected ASI Monument, Vaikunta Perumal Temple, Uttiramerur",
            publication_reference="ASI Annual Report on Epigraphy 1904–05; Epigraphia Indica Vol. XXII; SII Vol. III",
            coordinate_source="Uttiramerur historic temple precinct coordinates",
            discovery_location=LocationContext(
                place_name="Vaikunta Perumal Temple Mandapa Wall, Uttiramerur",
                institution_or_site="Vaikunta Perumal Temple",
                state="Tamil Nadu",
                district="Kancheepuram",
                taluk_or_city="Uttiramerur",
                village="Uttiramerur",
                latitude=12.6160,
                longitude=79.7530,
            ),
            collection_location=LocationContext(
                place_name="ASI ARIE 1904–05 Corpus",
                institution_or_site="Archaeological Survey of India",
                state="Tamil Nadu",
                district="Chennai",
            ),
            current_repository_location=LocationContext(
                place_name="Vaikunta Perumal Temple, Uttiramerur",
                institution_or_site="ASI Chennai Circle Protected Monument",
                state="Tamil Nadu",
                district="Kancheepuram",
                latitude=12.6160,
                longitude=79.7530,
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (uttiramerur.jpg)",
                institution_or_site="Uttiramerur Sample Plate",
                state="Tamil Nadu",
                district="Kancheepuram",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("uttiramerur_election").get("diplomatic_text", ""),
            translation_preview=_p("uttiramerur_election").get("english_translation", ""),
            image_path="assets/samples/uttiramerur.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="VT_CGM_001",
            preset_id="hero_stone_nadukal",
            is_curated_preset=True,
            title_en=_p("hero_stone_nadukal").get("title_en", "Chengam Hero Stone (Nadukal) Vatteluttu Inscription"),
            title_ta=_p("hero_stone_nadukal").get("title_ta", "செங்கம் நடுகல் வட்டெழுத்து கல்வெட்டு"),
            script="Vatteluttu",
            period="c. 6th–8th Century CE",
            dynasty="Pallava / Regional Chieftains (Vana / Ganga Borderlands)",
            material="Carved Granite Hero Stone Slab (Nadukal)",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="Tiruvannamalai",
            district_code="606",
            historical_district_name="North Arcot (Pre-1989) / Tiruvannamalai",
            sub_district_taluk="Chengam",
            village="",
            archaeological_site="Chengam Region Nadukal Series (Regional Specimen)",
            latitude=None,
            longitude=None,
            location_precision=LocationPrecision.DISTRICT_ONLY.value,
            discovery_source="Tamil Nadu State Dept. of Archaeology (Chengam Nadukarkal Corpus)",
            collection_source="Chengam Hero Stones Exhibition / Government Museum Series",
            current_repository="TNSDA District Archive / Museum Exhibition (Multiple Stones)",
            publication_reference="R. Nagaswamy (Ed.), Chengam Nadukarkal, State Dept. of Archaeology, TN",
            coordinate_source="None — intentional district-only provenance (no single village fixed)",
            discovery_location=LocationContext(
                place_name="Chengam Taluk Belt",
                institution_or_site="Chengam Nadukal Cluster",
                state="Tamil Nadu",
                district="Tiruvannamalai",
                taluk_or_city="Chengam",
                notes="District-level only; exact village plot omitted to avoid fabricating coordinates.",
            ),
            collection_location=LocationContext(
                place_name="TNSDA Dharmapuri / Tiruvannamalai Exploration Unit",
                institution_or_site="Tamil Nadu State Dept. of Archaeology",
                state="Tamil Nadu",
                district="Tiruvannamalai",
            ),
            current_repository_location=LocationContext(
                place_name="TNSDA Archival Repository",
                institution_or_site="State Dept. of Archaeology / Regional Museum",
                state="Tamil Nadu",
                district="Chennai",
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (nadukal.jpg)",
                institution_or_site="Chengam Nadukal Reference Plate",
                state="Tamil Nadu",
                district="Tiruvannamalai",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("hero_stone_nadukal").get("diplomatic_text", ""),
            translation_preview=_p("hero_stone_nadukal").get("english_translation", ""),
            image_path="assets/samples/nadukal.jpg",
        ),
        GeoInscriptionRecord(
            inscription_id="MC_ARL_001",
            preset_id="chola_arulmoli",
            is_curated_preset=True,
            title_en=_p("chola_arulmoli").get("title_en", "Arulmolidevan (Rajaraja Chola I) Epigraph"),
            title_ta=_p("chola_arulmoli").get("title_ta", "அருள்மொழிதேவன் (இராஜராஜ சோழன்) கல்வெட்டு"),
            script="Medieval Tamil",
            period="c. 992 CE (7th Regnal Year)",
            dynasty="Imperial Chola (Arulmolivarman / Rajaraja I)",
            material="Granite Temple Adhisthana Basement Wall",
            country="India",
            state="Tamil Nadu",
            state_code="33",
            district="",
            district_code="",
            historical_district_name="Chola Mandalam (Undisputed Regional Extent)",
            sub_district_taluk="",
            village="",
            archaeological_site="Temple Granite Adhisthana Wall, Chola Heartland",
            latitude=None,
            longitude=None,
            location_precision=LocationPrecision.UNKNOWN.value,
            discovery_source="Chola Epigraphical Corpus Reference Plate",
            collection_source="Kallvettu Curated Teaching Collection",
            current_repository="In-situ Temple Basement (Exact Temple District Unspecified)",
            publication_reference="South Indian Inscriptions / Chola Mandalam Endowment Exemplars",
            coordinate_source="None — intentional regional/unknown provenance",
            discovery_location=LocationContext(
                place_name="Chola Mandalam Temple Basement",
                state="Tamil Nadu",
                notes="Exact district/village not bound to a single point; kept as unknown provenance.",
            ),
            collection_location=LocationContext(
                place_name="Kallvettu Curated Archival Corpus",
                state="Tamil Nadu",
            ),
            current_repository_location=LocationContext(
                place_name="Unassigned Temple Sanctuary (Chola Heartland)",
                state="Tamil Nadu",
            ),
            photograph_capture_location=LocationContext(
                place_name="Kallvettu Archival Sample Plate (chola_kalvettu.jpg)",
                state="Tamil Nadu",
            ),
            transcription_status="Verified Academic Ground Truth",
            transcription_preview=_p("chola_arulmoli").get("diplomatic_text", ""),
            translation_preview=_p("chola_arulmoli").get("english_translation", ""),
            image_path="assets/samples/chola_kalvettu.jpg",
        ),
    ]
    return seeds


class AtlasDatabase:
    """
    SQLite-backed manager for epigraphical geographic records & provenance audit trails.
    """

    def __init__(self, db_path: Optional[str] = None, auto_seed: bool = True):
        self.db_path = db_path or DEFAULT_ATLAS_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()
        if auto_seed and self.count_records() == 0:
            self.seed_default_records()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._session() as conn:
            ver = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.executescript(SCHEMA_SQL)
            if ver < SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()

    def count_records(self) -> int:
        with self._session() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM atlas_inscriptions").fetchone()
            return int(row["cnt"]) if row else 0

    def seed_default_records(self) -> int:
        """Idempotently inserts Kallvettu's curated historical inscriptions."""
        inserted = 0
        for rec in build_curated_seed_records():
            if self.get_inscription(rec.inscription_id) is None:
                self.upsert_inscription(rec, validate_spatial=True, changed_by="system_seed")
                inserted += 1
        return inserted

    def upsert_inscription(
        self,
        record: GeoInscriptionRecord,
        validate_spatial: bool = True,
        changed_by: str = "Researcher",
        strip_disallowed_coords: bool = False,
    ) -> Dict[str, Any]:
        """
        Inserts or updates an epigraphical location record with provenance validation
        and audit trail recording.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if validate_spatial:
            is_valid, errors, warnings = validate_record_provenance(
                record, strip_disallowed_coords=strip_disallowed_coords
            )
            if not is_valid:
                raise ValueError("; ".join(errors))
            if warnings:
                record.spatial_validation_notes = " | ".join(warnings)
        else:
            if record.state:
                record.state = normalize_state_name(record.state)
            if record.state and record.district:
                record.district = normalize_district_name(record.state, record.district)

        now = _now_utc()
        if not record.created_utc:
            record.created_utc = now
        record.updated_utc = now

        existing = self.get_inscription(record.inscription_id)
        action = "UPDATE" if existing else "INSERT"

        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO atlas_inscriptions (
                    inscription_id, title_en, title_ta, script, period, dynasty, material,
                    country, state, state_code, district, district_code,
                    historical_district_name, sub_district_taluk, village, archaeological_site,
                    latitude, longitude, location_precision,
                    discovery_source, collection_source, current_repository,
                    publication_reference, coordinate_source,
                    discovery_location_json, collection_location_json,
                    current_repository_location_json, photograph_capture_location_json,
                    transcription_status, transcription_preview, translation_preview,
                    image_path, preset_id, is_curated_preset,
                    spatial_validation_notes, created_utc, updated_utc
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(inscription_id) DO UPDATE SET
                    title_en = excluded.title_en,
                    title_ta = excluded.title_ta,
                    script = excluded.script,
                    period = excluded.period,
                    dynasty = excluded.dynasty,
                    material = excluded.material,
                    country = excluded.country,
                    state = excluded.state,
                    state_code = excluded.state_code,
                    district = excluded.district,
                    district_code = excluded.district_code,
                    historical_district_name = excluded.historical_district_name,
                    sub_district_taluk = excluded.sub_district_taluk,
                    village = excluded.village,
                    archaeological_site = excluded.archaeological_site,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    location_precision = excluded.location_precision,
                    discovery_source = excluded.discovery_source,
                    collection_source = excluded.collection_source,
                    current_repository = excluded.current_repository,
                    publication_reference = excluded.publication_reference,
                    coordinate_source = excluded.coordinate_source,
                    discovery_location_json = excluded.discovery_location_json,
                    collection_location_json = excluded.collection_location_json,
                    current_repository_location_json = excluded.current_repository_location_json,
                    photograph_capture_location_json = excluded.photograph_capture_location_json,
                    transcription_status = excluded.transcription_status,
                    transcription_preview = excluded.transcription_preview,
                    translation_preview = excluded.translation_preview,
                    image_path = excluded.image_path,
                    preset_id = excluded.preset_id,
                    is_curated_preset = excluded.is_curated_preset,
                    spatial_validation_notes = excluded.spatial_validation_notes,
                    updated_utc = excluded.updated_utc
                """,
                (
                    record.inscription_id.strip(),
                    record.title_en.strip(),
                    record.title_ta,
                    record.script,
                    record.period,
                    record.dynasty,
                    record.material,
                    record.country or "India",
                    record.state,
                    record.state_code,
                    record.district,
                    record.district_code,
                    record.historical_district_name,
                    record.sub_district_taluk,
                    record.village,
                    record.archaeological_site,
                    record.latitude,
                    record.longitude,
                    record.location_precision,
                    record.discovery_source,
                    record.collection_source,
                    record.current_repository,
                    record.publication_reference,
                    record.coordinate_source,
                    json.dumps(record.discovery_location.to_dict(), ensure_ascii=False),
                    json.dumps(record.collection_location.to_dict(), ensure_ascii=False),
                    json.dumps(record.current_repository_location.to_dict(), ensure_ascii=False),
                    json.dumps(record.photograph_capture_location.to_dict(), ensure_ascii=False),
                    record.transcription_status,
                    record.transcription_preview,
                    record.translation_preview,
                    record.image_path,
                    record.preset_id,
                    1 if record.is_curated_preset else 0,
                    record.spatial_validation_notes,
                    record.created_utc,
                    record.updated_utc,
                ),
            )
            conn.execute(
                """
                INSERT INTO inscription_location_audit (
                    inscription_id, action, changed_by, snapshot_json, notes, recorded_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.inscription_id.strip(),
                    action,
                    changed_by,
                    json.dumps(record.to_dict(), ensure_ascii=False),
                    "; ".join(warnings) if warnings else "Validated",
                    now,
                ),
            )
            conn.commit()

        return {"action": action, "warnings": warnings, "inscription_id": record.inscription_id}

    def get_inscription(self, inscription_id: str) -> Optional[GeoInscriptionRecord]:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM atlas_inscriptions WHERE inscription_id = ?",
                (inscription_id,),
            ).fetchone()
            return _row_to_record(row) if row else None

    def delete_inscription(self, inscription_id: str, changed_by: str = "Researcher") -> bool:
        existing = self.get_inscription(inscription_id)
        if not existing:
            return False
        with self._session() as conn:
            conn.execute("DELETE FROM atlas_inscriptions WHERE inscription_id = ?", (inscription_id,))
            conn.execute(
                """
                INSERT INTO inscription_location_audit (
                    inscription_id, action, changed_by, snapshot_json, notes, recorded_utc
                ) VALUES (?, 'DELETE', ?, ?, 'Deleted from Atlas', ?)
                """,
                (inscription_id, changed_by, json.dumps(existing.to_dict(), ensure_ascii=False), _now_utc()),
            )
            conn.commit()
        return True

    def query_inscriptions(self, filters: Optional[AtlasFilterCriteria] = None) -> List[GeoInscriptionRecord]:
        """Queries records matching multi-facet search and filter criteria."""
        f = filters or AtlasFilterCriteria()
        clauses: List[str] = []
        params: List[Any] = []

        if f.state and f.state != "All":
            clauses.append("LOWER(state) = LOWER(?)")
            params.append(normalize_state_name(f.state))
        if f.district and f.district != "All":
            clauses.append("LOWER(district) = LOWER(?)")
            params.append(normalize_district_name(f.state if f.state != "All" else "", f.district))
        if f.script and f.script != "All":
            clauses.append("script = ?")
            params.append(f.script)
        if f.period and f.period != "All":
            clauses.append("period = ?")
            params.append(f.period)
        if f.dynasty and f.dynasty != "All":
            clauses.append("dynasty = ?")
            params.append(f.dynasty)
        if f.material and f.material != "All":
            clauses.append("material = ?")
            params.append(f.material)
        if f.collection_institution and f.collection_institution != "All":
            clauses.append("(current_repository = ? OR collection_source = ?)")
            params.extend([f.collection_institution, f.collection_institution])
        if f.verification_status and f.verification_status != "All":
            clauses.append("location_precision = ?")
            params.append(f.verification_status)

        if f.search_query and f.search_query.strip():
            q = f"%{f.search_query.strip()}%"
            clauses.append(
                "("
                "inscription_id LIKE ? OR title_en LIKE ? OR title_ta LIKE ? OR "
                "state LIKE ? OR district LIKE ? OR historical_district_name LIKE ? OR "
                "sub_district_taluk LIKE ? OR village LIKE ? OR archaeological_site LIKE ? OR "
                "dynasty LIKE ? OR current_repository LIKE ?"
                ")"
            )
            params.extend([q] * 11)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM atlas_inscriptions {where_sql} ORDER BY state, district, inscription_id"
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_district_counts(
        self, filters: Optional[AtlasFilterCriteria] = None
    ) -> Dict[tuple, Dict[str, int]]:
        """
        Aggregates inscription counts per (state, district) INCLUDING district_only
        records without coordinates, for accurate choropleth rendering.
        """
        records = self.query_inscriptions(filters)
        counts: Dict[tuple, Dict[str, int]] = {}
        for rec in records:
            if not rec.state or not rec.district:
                continue
            key = (rec.state, rec.district)
            entry = counts.setdefault(
                key,
                {"total": 0, "pinned": 0, "district_only": 0},
            )
            entry["total"] += 1
            if rec.has_mappable_coordinates():
                entry["pinned"] += 1
            else:
                entry["district_only"] += 1
        return counts

    def get_state_counts(self, filters: Optional[AtlasFilterCriteria] = None) -> Dict[str, Dict[str, int]]:
        """Aggregates inscription counts per State/UT across all filtered records."""
        records = self.query_inscriptions(filters)
        counts: Dict[str, Dict[str, int]] = {}
        for rec in records:
            if not rec.state:
                continue
            entry = counts.setdefault(rec.state, {"total": 0, "pinned": 0, "unpinned": 0})
            entry["total"] += 1
            if rec.has_mappable_coordinates():
                entry["pinned"] += 1
            else:
                entry["unpinned"] += 1
        return counts

    def get_filter_facets(self) -> Dict[str, List[str]]:
        """Returns distinct sorted facet values from the SQLite database."""
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT state, district, script, period, dynasty, material,
                       current_repository, collection_source, location_precision
                FROM atlas_inscriptions
                """
            ).fetchall()

        facets: Dict[str, set] = {
            "states": set(),
            "districts": set(),
            "scripts": set(),
            "periods": set(),
            "dynasties": set(),
            "materials": set(),
            "institutions": set(),
            "verification_statuses": set(),
        }
        for r in rows:
            if r["state"]:
                facets["states"].add(r["state"])
            if r["district"]:
                facets["districts"].add(r["district"])
            if r["script"]:
                facets["scripts"].add(r["script"])
            if r["period"]:
                facets["periods"].add(r["period"])
            if r["dynasty"]:
                facets["dynasties"].add(r["dynasty"])
            if r["material"]:
                facets["materials"].add(r["material"])
            if r["current_repository"]:
                facets["institutions"].add(r["current_repository"])
            if r["location_precision"]:
                facets["verification_statuses"].add(r["location_precision"])

        return {k: sorted(v) for k, v in facets.items()}

    def get_audit_history(self, inscription_id: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Retrieves location provenance change history from SQLite."""
        with self._session() as conn:
            if inscription_id:
                rows = conn.execute(
                    """
                    SELECT audit_id, inscription_id, action, changed_by, notes, recorded_utc
                    FROM inscription_location_audit
                    WHERE inscription_id = ?
                    ORDER BY audit_id DESC LIMIT ?
                    """,
                    (inscription_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT audit_id, inscription_id, action, changed_by, notes, recorded_utc
                    FROM inscription_location_audit
                    ORDER BY audit_id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]
