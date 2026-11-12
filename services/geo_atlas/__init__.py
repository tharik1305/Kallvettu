"""
Kallvettu India Epigraphical Geographic Atlas Package.

Provides administrative GeoJSON boundary loading, spatial point-in-district validation,
SQLite provenance records, and MapLibre GL JS / PyDeck interactive mapping.
"""

import importlib
from typing import Any

# Re-export core data models eagerly (lightweight dataclasses, zero I/O)
from services.geo_atlas.models import (
    LocationPrecision,
    LocationContext,
    CoordinateValidationResult,
    GeoInscriptionRecord,
    AtlasFilterCriteria,
)

# Known submodules for dynamic package-level attribute resolution
_SUBMODULES = {
    "geo_loader",
    "models",
    "validation",
    "database",
    "map_builder",
    "ui_atlas",
    "maplibre_component",
}

# Export tables for PEP 562 dynamic lazy-loading.
_LOADER_EXPORTS = {
    "load_geo_metadata",
    "load_all_districts_geojson",
    "load_india_states_geojson",
    "load_state_districts_geojson",
    "get_all_state_names",
    "get_districts_for_state",
    "get_state_admin_info",
    "get_district_feature",
    "get_district_bounds_and_center",
    "find_admin_unit_for_point",
    "normalize_state_name",
    "normalize_district_name",
    "state_name_to_slug",
    "validate_geojson_structure",
    "validate_geojson_feature",
    "clear_geo_cache",
    "GeoAtlasError",
    "GeoDatasetNotFoundError",
    "GeoJSONValidationError",
    "ALL_DISTRICTS_GEOJSON_PATH",
    "STATES_GEOJSON_PATH",
    "PER_STATE_GEO_DIR",
    "GEO_DATA_DIR",
    "GEO_METADATA_PATH",
}

_VALIDATION_EXPORTS = {
    "validate_coordinates",
    "validate_record_provenance",
}

_DATABASE_EXPORTS = {
    "AtlasDatabase",
    "DEFAULT_ATLAS_DB_PATH",
    "build_curated_seed_records",
}

_MAP_BUILDER_EXPORTS = {
    "BASEMAP_STYLES",
    "MAPLIBRE_STYLES",
    "compute_view_state",
    "cluster_pin_records",
    "build_atlas_deck",
    "extract_pydeck_selection",
    "records_to_markers_geojson",
    "build_districts_geojson",
    "build_states_geojson",
    "compute_maplibre_view_state",
}


def __getattr__(name: str) -> Any:
    """
    Dynamically loads symbols and submodules on first access.
    Eliminates circular import deadlocks when importing submodules (like ui_atlas or map_builder)
    and prevents eager initialization of Streamlit custom components.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f"services.geo_atlas.{name}")

    if name in _LOADER_EXPORTS:
        mod = importlib.import_module("services.geo_atlas.geo_loader")
        return getattr(mod, name)

    if name in _VALIDATION_EXPORTS:
        mod = importlib.import_module("services.geo_atlas.validation")
        return getattr(mod, name)

    if name in _DATABASE_EXPORTS:
        mod = importlib.import_module("services.geo_atlas.database")
        return getattr(mod, name)

    if name in _MAP_BUILDER_EXPORTS:
        mod = importlib.import_module("services.geo_atlas.map_builder")
        return getattr(mod, name)

    if name == "maplibre_atlas":
        from services.geo_atlas.maplibre_component import maplibre_atlas
        return maplibre_atlas

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Submodules
    "geo_loader",
    "models",
    "validation",
    "database",
    "map_builder",
    "ui_atlas",
    "maplibre_component",
    # Models
    "LocationPrecision",
    "LocationContext",
    "CoordinateValidationResult",
    "GeoInscriptionRecord",
    "AtlasFilterCriteria",
    # Geo Loader
    "load_geo_metadata",
    "load_all_districts_geojson",
    "load_india_states_geojson",
    "load_state_districts_geojson",
    "get_all_state_names",
    "get_districts_for_state",
    "get_state_admin_info",
    "get_district_feature",
    "get_district_bounds_and_center",
    "find_admin_unit_for_point",
    "normalize_state_name",
    "normalize_district_name",
    "state_name_to_slug",
    "validate_geojson_structure",
    "validate_geojson_feature",
    "clear_geo_cache",
    "GeoAtlasError",
    "GeoDatasetNotFoundError",
    "GeoJSONValidationError",
    "ALL_DISTRICTS_GEOJSON_PATH",
    "STATES_GEOJSON_PATH",
    "PER_STATE_GEO_DIR",
    "GEO_DATA_DIR",
    "GEO_METADATA_PATH",
    # Validation
    "validate_coordinates",
    "validate_record_provenance",
    # Database
    "AtlasDatabase",
    "DEFAULT_ATLAS_DB_PATH",
    "build_curated_seed_records",
    # Map Builder
    "BASEMAP_STYLES",
    "MAPLIBRE_STYLES",
    "compute_view_state",
    "cluster_pin_records",
    "build_atlas_deck",
    "extract_pydeck_selection",
    "records_to_markers_geojson",
    "build_districts_geojson",
    "build_states_geojson",
    "compute_maplibre_view_state",
    # Component
    "maplibre_atlas",
]
