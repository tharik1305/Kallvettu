"""
GeoJSON Administrative Boundary Loader & Spatial Index for India States/UTs & Districts.

Features:
- Explicit source metadata & attribution via datasets/geo/metadata.json
- Robust multi-source GeoJSON discovery & RFC 7946 schema validation
- Lazy loading & mtime-aware file modification caching (zero stale cache)
- Historical, variant, and Tamil Unicode state & district normalization
- Point-in-polygon containment resolution and camera viewport bounding boxes
- Graceful error recovery: missing or corrupt files never crash the application
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import unicodedata
from shapely.geometry import shape, Point

logger = logging.getLogger(__name__)

# --- Base Directory & File Paths (Pathlib for cross-platform Windows compatibility) ---
BASE_PROJECT_DIR = Path(__file__).resolve().parents[2]
GEO_DATA_DIR = BASE_PROJECT_DIR / "datasets" / "geo"
STATES_GEOJSON_PATH = GEO_DATA_DIR / "india_states.geojson"
ALL_DISTRICTS_GEOJSON_PATH = GEO_DATA_DIR / "india_all_districts.geojson"
PER_STATE_GEO_DIR = GEO_DATA_DIR / "states"
GEO_METADATA_PATH = GEO_DATA_DIR / "metadata.json"

# --- Domain Exception Classes ---
class GeoAtlasError(Exception):
    """Base error for epigraphical geographic atlas operations."""


class GeoDatasetNotFoundError(GeoAtlasError):
    """Raised or logged when an expected geographic dataset is missing."""


class GeoJSONValidationError(GeoAtlasError):
    """Raised or logged when a GeoJSON dataset violates structural requirements."""


# --- Standard GeoJSON Geometry Types ---
VALID_GEOMETRY_TYPES: Set[str] = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
}

# --- State and District Name Aliases (English Variants & Tamil Unicode) ---
STATE_NAME_ALIASES: Dict[str, str] = {
    # English variants
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "uttaranchal": "Uttarakhand",
    "tamilnadu": "Tamil Nadu",
    "tamil nadu": "Tamil Nadu",
    "andaman & nicobar": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "jammu & kashmir": "Jammu and Kashmir",
    "dadra & nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman & diu": "Dadra and Nagar Haveli and Daman and Diu",
    "nct of delhi": "Delhi",
    # Tamil Unicode aliases
    "தமிழ்நாடு": "Tamil Nadu",
    "தமிழ் நாடு": "Tamil Nadu",
    "புதுச்சேரி": "Puducherry",
    "கர்நாடகா": "Karnataka",
    "கருநாடகம்": "Karnataka",
    "கேரளா": "Kerala",
    "கேரளம்": "Kerala",
    "ஆந்திரப் பிரதேசம்": "Andhra Pradesh",
    "ஆந்திரா": "Andhra Pradesh",
    "தெலங்கானா": "Telangana",
    "ஒடிசா": "Odisha",
    "ஒரிசா": "Odisha",
    "மகாராட்டிரம்": "Maharashtra",
    "மகாராஷ்டிரா": "Maharashtra",
    "குஜராத்": "Gujarat",
    "தில்லி": "Delhi",
}

DISTRICT_NAME_ALIASES: Dict[str, str] = {
    # English variants
    "kanchipuram": "Kancheepuram",
    "sivagangai": "Sivaganga",
    "tiruvarur": "Thiruvarur",
    "villupuram": "Viluppuram",
    "tuticorin": "Thoothukkudi",
    "kanniyakumari": "Kanyakumari",
    "tiruvallur": "Thiruvallur",
    "thiruchirappalli": "Tiruchirappalli",
    "trichy": "Tiruchirappalli",
    "mysore": "Mysuru",
    "bangalore": "Bengaluru Urban",
    "bengaluru": "Bengaluru Urban",
    "belgaum": "Belagavi",
    "mangalore": "Dakshina Kannada",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Ernakulam",
    "calicut": "Kozhikode",
    "cuddapah": "Y.S.R.",
    "nellore": "Sri Potti Sriramulu Nellore",
    "madras": "Chennai",
    "tanjore": "Thanjavur",
    "tinnevelly": "Tirunelveli",
    "pondicherry": "Puducherry",
    # Tamil Unicode aliases
    "தஞ்சாவூர்": "Thanjavur",
    "தஞ்சை": "Thanjavur",
    "மதுரை": "Madurai",
    "காஞ்சிபுரம்": "Kancheepuram",
    "திருச்சிராப்பள்ளி": "Tiruchirappalli",
    "திருச்சி": "Tiruchirappalli",
    "சென்னை": "Chennai",
    "திருநெல்வேலி": "Tirunelveli",
    "நெல்லை": "Tirunelveli",
    "சிவகங்கை": "Sivaganga",
    "கன்னியாகுமரி": "Kanyakumari",
    "தூத்துக்குடி": "Thoothukkudi",
    "கோயம்புத்தூர்": "Coimbatore",
    "கோவை": "Coimbatore",
    "திருவள்ளூர்": "Thiruvallur",
    "விழுப்புரம்": "Viluppuram",
    "திருவாரூர்": "Thiruvarur",
    "புதுக்கோட்டை": "Pudukkottai",
    "இராமநாதபுரம்": "Ramanathapuram",
    "திண்டுக்கல்": "Dindigul",
    "சேலம்": "Salem",
    "தருமபுரி": "Dharmapuri",
    "ஈரோடு": "Erode",
    "வேலூர்": "Vellore",
    "பெங்களூரு": "Bengaluru Urban",
    "மைசூர்": "Mysuru",
}


# --- Mtime-Aware In-Memory Caching ---
_MTIME_CACHE: Dict[str, Tuple[float, Any]] = {}


def _read_json_with_mtime_cache(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Reads and parses a JSON file, caching the in-memory object keyed by (absolute_path, st_mtime).
    If the file on disk is modified, st_mtime changes and the file is re-read automatically.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.is_file():
        return None
    try:
        mtime = path_obj.stat().st_mtime
    except OSError:
        return None

    cache_key = str(path_obj)
    if cache_key in _MTIME_CACHE:
        cached_mtime, cached_val = _MTIME_CACHE[cache_key]
        if cached_mtime == mtime:
            return cached_val

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    _MTIME_CACHE[cache_key] = (mtime, data)
    return data


def clear_geo_cache() -> None:
    """Flushes all in-memory GeoJSON and metadata caches."""
    _MTIME_CACHE.clear()


# --- GeoJSON Schema & Feature Validation ---

def _validate_coordinate_pair(pt: Any) -> Tuple[bool, str]:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return False, f"Coordinate must have at least 2 values, got {pt!r}"
    try:
        lon, lat = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return False, f"Non-numeric coordinate values: {pt!r}"
    if math.isnan(lon) or math.isnan(lat):
        return False, f"Coordinate contains NaN: ({lon}, {lat})"
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return False, f"Coordinate out of WGS84 range: lon={lon}, lat={lat}"
    return True, ""


def _validate_coordinate_ranges(coords: Any, geom_type: str) -> Tuple[bool, str]:
    """Validates that coordinate values fall within standard WGS84 ranges in [lon, lat] order."""
    if coords is None:
        return False, "Coordinates cannot be None"
    try:
        if geom_type == "Point":
            return _validate_coordinate_pair(coords)
        elif geom_type in ("MultiPoint", "LineString"):
            if not isinstance(coords, (list, tuple)):
                return False, "Coordinates must be a list"
            for pt in coords:
                ok, err = _validate_coordinate_pair(pt)
                if not ok:
                    return False, err
        elif geom_type in ("Polygon", "MultiLineString"):
            if not isinstance(coords, (list, tuple)):
                return False, "Coordinates must be a list"
            for ring in coords:
                if not isinstance(ring, (list, tuple)):
                    return False, "Ring must be a list of points"
                for pt in ring:
                    ok, err = _validate_coordinate_pair(pt)
                    if not ok:
                        return False, err
        elif geom_type == "MultiPolygon":
            if not isinstance(coords, (list, tuple)):
                return False, "Coordinates must be a list"
            for poly in coords:
                if not isinstance(poly, (list, tuple)):
                    return False, "Polygon must be a list of rings"
                for ring in poly:
                    if not isinstance(ring, (list, tuple)):
                        return False, "Ring must be a list of points"
                    for pt in ring:
                        ok, err = _validate_coordinate_pair(pt)
                        if not ok:
                            return False, err
        return True, ""
    except Exception as exc:
        return False, f"Error validating coordinates: {exc}"


def validate_geojson_feature(
    feat: Any,
    index: int = 0,
    source_name: str = "feature",
) -> Tuple[bool, List[str]]:
    """
    Validates a single GeoJSON Feature object according to RFC 7946.
    Returns (is_valid, list_of_diagnostic_issues).
    """
    issues: List[str] = []
    if not isinstance(feat, dict):
        return False, [f"{source_name}[{index}]: Feature must be a dict, got {type(feat).__name__}."]

    if feat.get("type") != "Feature":
        issues.append(f"{source_name}[{index}]: Expected 'type': 'Feature', got {feat.get('type')!r}.")

    geom = feat.get("geometry")
    if geom is not None:
        if not isinstance(geom, dict):
            return False, [f"{source_name}[{index}]: Geometry must be a dict or None, got {type(geom).__name__}."]
        g_type = geom.get("type")
        if g_type not in VALID_GEOMETRY_TYPES:
            return False, [f"{source_name}[{index}]: Invalid geometry type {g_type!r}."]
        coords = geom.get("coordinates")
        if coords is None and g_type != "GeometryCollection":
            return False, [f"{source_name}[{index}]: Geometry missing 'coordinates'."]
        coord_valid, coord_err = _validate_coordinate_ranges(coords, g_type)
        if not coord_valid:
            issues.append(f"{source_name}[{index}]: {coord_err}")

    props = feat.get("properties")
    if props is not None and not isinstance(props, dict):
        issues.append(f"{source_name}[{index}]: 'properties' must be a dict or None, got {type(props).__name__}.")

    is_valid = (feat.get("type") == "Feature") and (len(issues) == 0)
    return is_valid, issues


def validate_geojson_structure(
    data: Any,
    source_name: str = "GeoJSON",
) -> Tuple[bool, List[str]]:
    """
    Validates GeoJSON FeatureCollection structure according to RFC 7946.
    Returns (is_valid, list_of_diagnostic_issues).
    """
    issues: List[str] = []
    if not isinstance(data, dict):
        return False, [f"{source_name}: Root object must be a JSON object/dict, got {type(data).__name__}."]

    if data.get("type") != "FeatureCollection":
        return False, [f"{source_name}: Expected 'type': 'FeatureCollection', got {data.get('type')!r}."]

    features = data.get("features")
    if not isinstance(features, list):
        return False, [f"{source_name}: 'features' property must be a list, got {type(features).__name__}."]

    valid_features_count = 0
    for idx, feat in enumerate(features):
        is_feat_valid, feat_issues = validate_geojson_feature(feat, idx, source_name)
        if not is_feat_valid:
            issues.extend(feat_issues)
        else:
            valid_features_count += 1

    if len(features) > 0 and valid_features_count == 0:
        return False, issues

    return True, issues


def _is_valid_feature_collection(data: Any) -> bool:
    """Fast check for valid FeatureCollection dict."""
    if not isinstance(data, dict):
        return False
    if data.get("type") != "FeatureCollection":
        return False
    features = data.get("features")
    return isinstance(features, list)


# --- Name Normalization & Slug Generation ---

def state_name_to_slug(state_name: str) -> str:
    """Converts a canonical state/UT name to its lazy-loaded GeoJSON filename slug."""
    norm = normalize_state_name(state_name)
    if not norm:
        return ""
    low = norm.lower()
    if "dadra" in low or "daman" in low:
        return "dnh-and-dd"
    return low.replace("&", "and").replace(" ", "-")


def normalize_state_name(state_name: str) -> str:
    """Normalizes common historical, variant, and Tamil Unicode spellings to canonical state/UT name."""
    if not state_name:
        return ""
    clean = unicodedata.normalize("NFKC", " ".join(state_name.strip().split()))
    low = clean.lower()
    if low in STATE_NAME_ALIASES:
        return STATE_NAME_ALIASES[low]
    if clean in STATE_NAME_ALIASES:
        return STATE_NAME_ALIASES[clean]

    # Case-insensitive match against metadata index
    meta = load_geo_metadata()
    for st in meta.get("states", []):
        st_nm = st.get("st_nm", "")
        if st_nm.lower() == low or unicodedata.normalize("NFKC", st_nm).lower() == low:
            return st_nm
    return clean


def normalize_district_name(state_name: str, district_name: str) -> str:
    """
    Normalizes district spellings (English variants and Tamil Unicode names)
    against aliases and the administrative index for that state.
    """
    if not district_name:
        return ""
    clean = unicodedata.normalize("NFKC", " ".join(district_name.strip().split()))
    low = clean.lower()

    if low in DISTRICT_NAME_ALIASES:
        clean = DISTRICT_NAME_ALIASES[low]
        low = clean.lower()
    elif clean in DISTRICT_NAME_ALIASES:
        clean = DISTRICT_NAME_ALIASES[clean]
        low = clean.lower()

    norm_state = normalize_state_name(state_name)
    if norm_state:
        districts = get_districts_for_state(norm_state)
        for d in districts:
            d_norm = unicodedata.normalize("NFKC", d).lower()
            if d.lower() == low or d_norm == low:
                return d
    return clean


# --- Dataset Loaders (Mtime-cached & Resilient) ---

def load_geo_metadata() -> Dict[str, Any]:
    """Loads datasets/geo/metadata.json containing source attribution & state index."""
    meta_path = Path(GEO_METADATA_PATH)
    if meta_path.is_file():
        try:
            data = _read_json_with_mtime_cache(meta_path)
            if data and isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"Error reading geo metadata {meta_path}: {exc}")
    return {
        "dataset_name": "Unavailable",
        "dataset_version": "unknown",
        "total_states_uts": 0,
        "states": [],
    }


def load_all_districts_geojson() -> Dict[str, Any]:
    """
    Loads or aggregates all Indian district boundary features into a valid GeoJSON FeatureCollection.

    Discovery strategy:
    1. Primary: Composite national file at datasets/geo/india_all_districts.geojson.
    2. Fallback: Aggregation across all per-state GeoJSON files in datasets/geo/states/*.geojson.

    Validates GeoJSON schema, preserves district/state properties & administrative identifiers,
    deduplicates multiple entries, handles missing/malformed files gracefully, and never fabricates features.
    """
    comp_path = Path(ALL_DISTRICTS_GEOJSON_PATH)
    if comp_path.is_file():
        try:
            data = _read_json_with_mtime_cache(comp_path)
            if data and _is_valid_feature_collection(data) and len(data.get("features", [])) > 0:
                return data
            else:
                logger.warning(
                    f"Composite districts file {comp_path} exists but is empty or not a valid FeatureCollection. Falling back to per-state aggregation."
                )
        except Exception as exc:
            logger.warning(
                f"Error reading composite districts GeoJSON from {comp_path}: {exc}. Falling back to per-state aggregation."
            )

    # Fallback: discover and aggregate from per-state datasets in PER_STATE_GEO_DIR
    aggregated_features: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    state_dir = Path(PER_STATE_GEO_DIR)
    if state_dir.is_dir():
        try:
            state_files = sorted(
                p for p in state_dir.glob("*.geojson") if p.is_file()
            )
        except OSError as e:
            logger.error(f"Cannot list per-state GeoJSON directory {state_dir}: {e}")
            state_files = []

        for fpath in state_files:
            try:
                state_data = _read_json_with_mtime_cache(fpath)
                if state_data and _is_valid_feature_collection(state_data):
                    for feat in state_data.get("features", []):
                        if isinstance(feat, dict) and feat.get("geometry"):
                            props = feat.get("properties") or {}
                            d_name = (props.get("district") or props.get("dtname") or props.get("district_name") or "").strip().lower()
                            s_name = (props.get("st_nm") or props.get("stname") or props.get("state_name") or "").strip().lower()
                            dt_code = str(props.get("dt_code") or "").strip()
                            key = (s_name, d_name, dt_code) if (d_name and s_name) else None
                            if key and key in seen_keys:
                                continue
                            if key:
                                seen_keys.add(key)
                            aggregated_features.append(feat)
                else:
                    logger.warning(f"File {fpath.name} is not a valid GeoJSON FeatureCollection, skipping.")
            except Exception as e:
                logger.warning(f"Failed to load or parse state GeoJSON {fpath.name}: {e}, skipping.")

    if aggregated_features:
        logger.info(f"Aggregated {len(aggregated_features)} unique district features from {state_dir}.")
        return {
            "type": "FeatureCollection",
            "features": aggregated_features,
        }

    logger.error(
        f"No valid district GeoJSON datasets found at {comp_path} or in {state_dir}."
    )
    return {"type": "FeatureCollection", "features": []}


def load_india_states_geojson() -> Dict[str, Any]:
    """Loads the national 36 State/UT GeoJSON FeatureCollection."""
    states_path = Path(STATES_GEOJSON_PATH)
    if states_path.is_file():
        try:
            data = _read_json_with_mtime_cache(states_path)
            if data and _is_valid_feature_collection(data):
                return data
        except Exception as exc:
            logger.warning(f"Error reading India states GeoJSON {states_path}: {exc}")
    return {"type": "FeatureCollection", "features": []}


def load_state_districts_geojson(state_name: str) -> Dict[str, Any]:
    """
    Lazily loads district polygons ONLY for the requested state/UT.
    Falls back cleanly to composite or aggregated collection if per-state file is missing.
    """
    norm_state = normalize_state_name(state_name)
    slug = state_name_to_slug(norm_state)
    if not slug:
        return {"type": "FeatureCollection", "features": []}

    per_state_path = Path(PER_STATE_GEO_DIR) / f"{slug}.geojson"
    if per_state_path.is_file():
        try:
            data = _read_json_with_mtime_cache(per_state_path)
            if data and _is_valid_feature_collection(data):
                return data
        except Exception as exc:
            logger.warning(f"Error reading state district GeoJSON {per_state_path}: {exc}")

    # Fallback: filter composite or aggregated collection
    all_gj = load_all_districts_geojson()
    feats = [
        feat for feat in all_gj.get("features", [])
        if normalize_state_name((feat.get("properties") or {}).get("st_nm", "")) == norm_state
    ]
    return {"type": "FeatureCollection", "features": feats}


# Maintain backward compatibility with .cache_clear() calls
load_all_districts_geojson.cache_clear = clear_geo_cache  # type: ignore[attr-defined]
load_india_states_geojson.cache_clear = clear_geo_cache   # type: ignore[attr-defined]
load_state_districts_geojson.cache_clear = clear_geo_cache # type: ignore[attr-defined]
load_geo_metadata.cache_clear = clear_geo_cache           # type: ignore[attr-defined]


# --- Administrative Metadata & Spatial Queries ---

def get_all_state_names() -> List[str]:
    """Returns alphabetically sorted canonical Indian State & UT names."""
    meta = load_geo_metadata()
    names = [s.get("st_nm", "") for s in meta.get("states", []) if s.get("st_nm")]
    if names:
        return sorted(names)
    gj = load_india_states_geojson()
    return sorted({
        (f.get("properties") or {}).get("st_nm", "")
        for f in gj.get("features", [])
        if (f.get("properties") or {}).get("st_nm")
    })


def get_districts_for_state(state_name: str) -> List[str]:
    """Returns sorted canonical district names for a given State/UT."""
    norm_state = normalize_state_name(state_name)
    meta = load_geo_metadata()
    for st in meta.get("states", []):
        if st.get("st_nm") == norm_state and st.get("districts"):
            return sorted(st["districts"])
    gj = load_state_districts_geojson(norm_state)
    dist_names: Set[str] = set()
    for f in gj.get("features", []):
        props = f.get("properties") or {}
        d_val = props.get("district") or props.get("dtname") or props.get("district_name") or ""
        if d_val:
            dist_names.add(d_val.strip())
    return sorted(dist_names)


def get_state_admin_info(state_name: str) -> Optional[Dict[str, Any]]:
    """Returns state metadata including st_code, centroid, bounds, and district count."""
    norm_state = normalize_state_name(state_name)
    meta = load_geo_metadata()
    for st in meta.get("states", []):
        if st.get("st_nm") == norm_state:
            return st
    return None


def get_district_feature(state_name: str, district_name: str) -> Optional[Dict[str, Any]]:
    """Returns the GeoJSON feature dict for (state_name, district_name) if present."""
    norm_state = normalize_state_name(state_name)
    norm_dist = normalize_district_name(norm_state, district_name)
    if not norm_state or not norm_dist:
        return None
    gj = load_state_districts_geojson(norm_state)
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        d_name = props.get("district") or props.get("dtname") or props.get("district_name") or ""
        if d_name.strip().lower() == norm_dist.lower():
            return feat
    return None


def get_district_bounds_and_center(state_name: str, district_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns administrative bounding box and polygon center for CAMERA ZOOMING ONLY.
    NOTE: Must NEVER be used to assign or fabricate coordinates for an inscription.
    """
    feat = get_district_feature(state_name, district_name)
    if not feat or not isinstance(feat, dict):
        return None
    geom_dict = feat.get("geometry")
    if not geom_dict or not isinstance(geom_dict, dict):
        return None
    try:
        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = geom.buffer(0)
        minx, miny, maxx, maxy = geom.bounds
        rep = geom.centroid
        props = feat.get("properties") or {}
        d_code = str(props.get("dt_code") or props.get("district_code") or "")
        s_code = str(props.get("st_code") or props.get("state_code") or "")
        return {
            "bounds": [float(minx), float(miny), float(maxx), float(maxy)],
            "view_center_lat": float(rep.y),
            "view_center_lon": float(rep.x),
            "district_code": d_code,
            "state_code": s_code,
        }
    except Exception as exc:
        logger.warning(f"Could not compute bounds/center for district {district_name!r}: {exc}")
        return None


def _find_containing_district_in_geojson(
    lat: float,
    lon: float,
    geojson_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Tests point-in-polygon containment across features in a GeoJSON dataset."""
    if not geojson_data or not isinstance(geojson_data, dict):
        return None
    features = geojson_data.get("features")
    if not isinstance(features, list):
        return None

    try:
        pt = Point(float(lon), float(lat))
    except (TypeError, ValueError):
        return None

    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom_dict = feat.get("geometry")
        if not geom_dict or not isinstance(geom_dict, dict):
            continue
        try:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.covers(pt):
                props = feat.get("properties") or {}
                s_name = (props.get("st_nm") or props.get("stname") or props.get("state_name") or "").strip()
                s_code = str(props.get("st_code") or props.get("state_code") or "").strip()
                d_name = (props.get("district") or props.get("dtname") or props.get("district_name") or "").strip()
                d_code = str(props.get("dt_code") or props.get("district_code") or "").strip()
                return {
                    "state": s_name,
                    "state_code": s_code,
                    "district": d_name,
                    "district_code": d_code,
                }
        except Exception:
            continue
    return None


def find_admin_unit_for_point(
    lat: float,
    lon: float,
    state_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolves which official India State/UT and District polygon contains (lat, lon).
    Uses bounding-box pre-filter on states before lazily loading candidate state districts.
    """
    if lat is None or lon is None:
        return None
    try:
        f_lat = float(lat)
        f_lon = float(lon)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(f_lat) or not math.isfinite(f_lon):
        return None
    if not (-90.0 <= f_lat <= 90.0 and -180.0 <= f_lon <= 180.0):
        return None

    if state_hint:
        norm_hint = normalize_state_name(state_hint)
        match = _find_containing_district_in_geojson(
            f_lat, f_lon, load_state_districts_geojson(norm_hint)
        )
        if match:
            return match

    pt = Point(f_lon, f_lat)
    meta = load_geo_metadata()
    candidate_states: List[str] = []
    for st in meta.get("states", []):
        bounds = st.get("bounds")
        if bounds and len(bounds) == 4:
            minx, miny, maxx, maxy = bounds
            if minx <= f_lon <= maxx and miny <= f_lat <= maxy:
                candidate_states.append(st["st_nm"])

    for st_nm in candidate_states:
        match = _find_containing_district_in_geojson(
            f_lat, f_lon, load_state_districts_geojson(st_nm)
        )
        if match:
            return match

    # Fallback check against dissolved state polygons in case a point sits near a bbox edge
    states_gj = load_india_states_geojson()
    for sfeat in states_gj.get("features", []):
        if not isinstance(sfeat, dict):
            continue
        s_geom = sfeat.get("geometry")
        if not s_geom or not isinstance(s_geom, dict):
            continue
        try:
            geom = shape(s_geom)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.covers(pt):
                st_nm = (sfeat.get("properties") or {}).get("st_nm", "")
                match = _find_containing_district_in_geojson(
                    f_lat, f_lon, load_state_districts_geojson(st_nm)
                )
                if match:
                    return match
                props = sfeat.get("properties") or {}
                return {
                    "state": (props.get("st_nm") or "").strip(),
                    "state_code": str(props.get("st_code") or "").strip(),
                    "district": "",
                    "district_code": "",
                }
        except Exception:
            continue
    return None


__all__ = [
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
]
