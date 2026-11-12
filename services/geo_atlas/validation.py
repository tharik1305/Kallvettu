"""
Spatial & Geographic Provenance Validator for Kallvettu Atlas Records.

Provides:
- Coordinate range & null-island checks
- India geographic envelope verification
- Point-in-district polygon validation against GeoJSON administrative boundaries
- Strict anti-fabrication guard: ensures district_only / unknown records never
  receive artificial centroid coordinates
"""

import math
from typing import Optional, List, Tuple
from shapely.geometry import shape, Point

from services.geo_atlas.models import (
    CoordinateValidationResult,
    GeoInscriptionRecord,
    LocationPrecision,
    VALID_PRECISION_VALUES,
)
from services.geo_atlas.geo_loader import (
    normalize_state_name,
    normalize_district_name,
    get_state_admin_info,
    get_district_feature,
    find_admin_unit_for_point,
)

# Bounding envelope of India (including islands: Andaman/Nicobar & Lakshadweep)
INDIA_LAT_MIN = 6.4
INDIA_LAT_MAX = 37.6
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.6


def validate_coordinates(
    latitude: Optional[float],
    longitude: Optional[float],
    state: str = "",
    district: str = "",
    check_boundary: bool = True,
) -> CoordinateValidationResult:
    """
    Validates raw latitude/longitude values, India bounds, and (optionally)
    containment within the specified State & District GeoJSON polygon.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if latitude is None or longitude is None:
        return CoordinateValidationResult(
            is_valid=False,
            within_wgs84_bounds=False,
            within_india_bounds=False,
            errors=["Both latitude and longitude must be provided."],
        )

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return CoordinateValidationResult(
            is_valid=False,
            within_wgs84_bounds=False,
            within_india_bounds=False,
            errors=["Latitude and longitude must be valid numeric values."],
        )

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return CoordinateValidationResult(
            is_valid=False,
            within_wgs84_bounds=False,
            within_india_bounds=False,
            errors=["Coordinates cannot be NaN or infinite."],
        )

    within_wgs84 = (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)
    if not within_wgs84:
        errors.append(
            f"Coordinates ({lat:.5f}, {lon:.5f}) are outside valid WGS84 bounds [-90..90, -180..180]."
        )

    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        errors.append("Coordinates (0.0, 0.0) appear to be a null-island placeholder.")

    within_india = (
        INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX
        and INDIA_LON_MIN <= lon <= INDIA_LON_MAX
    )
    if within_wgs84 and not within_india:
        warnings.append(
            f"Coordinates ({lat:.5f}, {lon:.5f}) fall outside India's geographic bounds "
            f"[{INDIA_LAT_MIN}..{INDIA_LAT_MAX} N, {INDIA_LON_MIN}..{INDIA_LON_MAX} E]."
        )

    matched_state = ""
    matched_state_code = ""
    matched_district = ""
    matched_district_code = ""
    point_in_state: Optional[bool] = None
    point_in_district: Optional[bool] = None

    norm_state = normalize_state_name(state)
    norm_district = normalize_district_name(norm_state, district)

    if check_boundary and within_wgs84 and within_india and not errors:
        # Resolve exact containing administrative unit from GeoJSON
        resolved = find_admin_unit_for_point(lat, lon, state_hint=norm_state or None)
        if resolved:
            matched_state = resolved.get("state", "")
            matched_state_code = resolved.get("state_code", "")
            matched_district = resolved.get("district", "")
            matched_district_code = resolved.get("district_code", "")

        if norm_state:
            point_in_state = (matched_state.lower() == norm_state.lower()) if matched_state else False
            if not point_in_state:
                detail = f"detected '{matched_state}'" if matched_state else "outside loaded Indian state polygons"
                warnings.append(
                    f"State mismatch: declared '{norm_state}', but coordinate ({lat:.4f}, {lon:.4f}) is in {detail}."
                )

        if norm_state and norm_district:
            dist_feat = get_district_feature(norm_state, norm_district)
            if dist_feat:
                geom = shape(dist_feat["geometry"])
                pt = Point(lon, lat)
                # Tiny 0.002 deg (~200m) coastal/border tolerance only for containment diagnostic
                inside_exact = bool(geom.covers(pt))
                point_in_district = inside_exact
                if not inside_exact:
                    actual_desc = (
                        f"'{matched_district}, {matched_state}'"
                        if matched_district
                        else "outside the target district boundary"
                    )
                    warnings.append(
                        f"District boundary check: coordinate ({lat:.4f}, {lon:.4f}) does not lie inside "
                        f"'{norm_district}, {norm_state}' ({actual_desc})."
                    )
            else:
                warnings.append(
                    f"Boundary geometry not found for district '{norm_district}' in '{norm_state}'; "
                    "containment check skipped."
                )

    return CoordinateValidationResult(
        is_valid=(len(errors) == 0),
        within_wgs84_bounds=within_wgs84,
        within_india_bounds=within_india,
        point_in_district=point_in_district,
        point_in_state=point_in_state,
        matched_state=matched_state,
        matched_state_code=matched_state_code,
        matched_district=matched_district,
        matched_district_code=matched_district_code,
        warnings=warnings,
        errors=errors,
    )


def validate_record_provenance(
    record: GeoInscriptionRecord,
    strip_disallowed_coords: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates geographic integrity of a GeoInscriptionRecord.

    Returns: (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not record.inscription_id or not record.inscription_id.strip():
        errors.append("inscription_id is required.")
    if not record.title_en or not record.title_en.strip():
        errors.append("title_en is required.")

    if record.location_precision not in VALID_PRECISION_VALUES:
        errors.append(
            f"Invalid location_precision '{record.location_precision}'. "
            f"Allowed: {sorted(VALID_PRECISION_VALUES)}."
        )

    # Normalize canonical state & district while keeping historical district name untouched
    if record.state:
        record.state = normalize_state_name(record.state)
        s_info = get_state_admin_info(record.state)
        if s_info and not record.state_code:
            record.state_code = str(s_info.get("st_code") or "")

    if record.state and record.district:
        canonical_dist = normalize_district_name(record.state, record.district)
        if canonical_dist != record.district and not record.historical_district_name:
            record.historical_district_name = record.district
        record.district = canonical_dist
        dfeat = get_district_feature(record.state, record.district)
        if dfeat and not record.district_code:
            record.district_code = str((dfeat.get("properties") or {}).get("dt_code") or "")

    has_lat = record.latitude is not None
    has_lon = record.longitude is not None

    if has_lat != has_lon:
        errors.append("Latitude and longitude must either both be provided or both be empty (None).")

    if record.location_precision in (LocationPrecision.DISTRICT_ONLY.value, LocationPrecision.UNKNOWN.value):
        if has_lat or has_lon:
            if strip_disallowed_coords:
                record.latitude = None
                record.longitude = None
                warnings.append(
                    f"Removed coordinates for '{record.location_precision}' precision to avoid "
                    "treating non-site coordinates as exact archaeological locations."
                )
            else:
                errors.append(
                    f"Records with location_precision='{record.location_precision}' must NOT have "
                    "exact marker coordinates (latitude/longitude must be None)."
                )
        if record.location_precision == LocationPrecision.DISTRICT_ONLY.value and (
            not record.state or not record.district
        ):
            errors.append("Records with location_precision='district_only' require both state and district.")
    else:
        # verified or approximate
        if not has_lat or not has_lon:
            errors.append(
                f"Records with location_precision='{record.location_precision}' require valid latitude and longitude."
            )
        else:
            cval = validate_coordinates(
                record.latitude,
                record.longitude,
                state=record.state,
                district=record.district,
                check_boundary=True,
            )
            errors.extend(cval.errors)
            warnings.extend(cval.warnings)
            if cval.matched_state_code and not record.state_code:
                record.state_code = cval.matched_state_code
            if cval.matched_district_code and not record.district_code:
                record.district_code = cval.matched_district_code

    return (len(errors) == 0, errors, warnings)

