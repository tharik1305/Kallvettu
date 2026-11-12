"""
Domain Models & Typed Schemas for the India Epigraphical Geographic Atlas.

Maintains strict separation between:
1. Original Discovery Location (findspot / archaeological monument)
2. Collection Location (field survey / archival collection unit)
3. Current Repository Location (museum / temple custodian / ASI circle)
4. Photograph Capture Location (digital squeeze / photographic capture context)

Enforces explicit coordinate provenance tiers so district-level records are
never conflated with exact monument coordinates or fabricated district centroids.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class LocationPrecision(str, Enum):
    """Explicit spatial verification tier for an epigraphical record."""
    VERIFIED = "verified"            # Field/GPS/ASI survey verified monument coordinates
    APPROXIMATE = "approximate"      # Published locality / monument-vicinity coordinates
    DISTRICT_ONLY = "district_only"  # Known district/taluk only; lat/lon intentionally None
    UNKNOWN = "unknown"              # Regional/unspecified provenance; lat/lon intentionally None


VALID_PRECISION_VALUES = {p.value for p in LocationPrecision}


@dataclass
class LocationContext:
    """
    Structured geographic context for a single provenance facet
    (discovery, collection, current repository, or photograph capture).
    """
    place_name: str = ""
    institution_or_site: str = ""
    country: str = "India"
    state: str = ""
    district: str = ""
    taluk_or_city: str = ""
    village: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LocationContext":
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            place_name=str(data.get("place_name") or ""),
            institution_or_site=str(data.get("institution_or_site") or ""),
            country=str(data.get("country") or "India"),
            state=str(data.get("state") or ""),
            district=str(data.get("district") or ""),
            taluk_or_city=str(data.get("taluk_or_city") or ""),
            village=str(data.get("village") or ""),
            latitude=float(data["latitude"]) if data.get("latitude") is not None else None,
            longitude=float(data["longitude"]) if data.get("longitude") is not None else None,
            notes=str(data.get("notes") or ""),
        )


@dataclass
class CoordinateValidationResult:
    """Result of geographic coordinate and administrative boundary validation."""
    is_valid: bool
    within_wgs84_bounds: bool
    within_india_bounds: bool
    point_in_district: Optional[bool] = None
    point_in_state: Optional[bool] = None
    matched_state: str = ""
    matched_state_code: str = ""
    matched_district: str = ""
    matched_district_code: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeoInscriptionRecord:
    """
    Complete epigraphical + geographical record stored in SQLite for Atlas exploration.
    Keeps historical & administrative IDs and all 4 provenance facets separate.
    """
    inscription_id: str
    title_en: str
    title_ta: str = ""
    script: str = "Tamil-Brahmi"
    period: str = ""
    dynasty: str = ""
    material: str = "Granite / Stone"

    # Primary Archaeological Administrative Location (used for Atlas indexing)
    country: str = "India"
    state: str = ""
    state_code: str = ""
    district: str = ""
    district_code: str = ""
    historical_district_name: str = ""
    sub_district_taluk: str = ""
    village: str = ""
    archaeological_site: str = ""

    # Archaeological site coordinates (MUST be None for district_only / unknown)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_precision: str = LocationPrecision.UNKNOWN.value

    # Summary Provenance Strings & References
    discovery_source: str = ""
    collection_source: str = ""
    current_repository: str = ""
    publication_reference: str = ""
    coordinate_source: str = ""

    # Four Strictly Separated Provenance Contexts
    discovery_location: LocationContext = field(default_factory=LocationContext)
    collection_location: LocationContext = field(default_factory=LocationContext)
    current_repository_location: LocationContext = field(default_factory=LocationContext)
    photograph_capture_location: LocationContext = field(default_factory=LocationContext)

    # Epigraphical Content & Workbench Linkage
    transcription_status: str = "Reference Curated"
    transcription_preview: str = ""
    translation_preview: str = ""
    image_path: str = ""
    preset_id: str = ""
    is_curated_preset: bool = False
    spatial_validation_notes: str = ""
    created_utc: str = ""
    updated_utc: str = ""

    def has_mappable_coordinates(self) -> bool:
        """
        True strictly when real site coordinates exist and precision is not
        district_only / unknown. Prevents district centroids from acting as pins.
        """
        if self.latitude is None or self.longitude is None:
            return False
        return self.location_precision in (
            LocationPrecision.VERIFIED.value,
            LocationPrecision.APPROXIMATE.value,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["has_mappable_coordinates"] = self.has_mappable_coordinates()
        return d


@dataclass
class AtlasFilterCriteria:
    """Multi-facet search and filter specification for the Atlas workbench."""
    search_query: str = ""
    state: str = "All"
    district: str = "All"
    script: str = "All"
    period: str = "All"
    dynasty: str = "All"
    material: str = "All"
    collection_institution: str = "All"
    verification_status: str = "All"

