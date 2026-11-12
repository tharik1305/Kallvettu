"""
Interactive GIS Map Layer Builder & OpenFreeMap / MapLibre GL Integration.

Features:
- MapLibre GL JS + OpenFreeMap vector tile integration (Liberty, Bright, Positron, Dark)
- India State/UT boundary GeoJSON layer with count shading
- Lazy-loaded per-state District choropleth GeoJSON layer
- Inscription marker FeatureCollection with MapLibre clustering
- Strict anti-fabrication: Never renders district centroids as fake archaeological markers
- Backward-compatible PyDeck deck builder for offline fallback
- Dark Graphite & Muted Bronze visual hierarchy tailored to Kallvettu's theme
"""

import copy
import math
from typing import Dict, Any, List, Optional, Tuple
import pydeck as pdk

from services.geo_atlas.models import GeoInscriptionRecord, LocationPrecision
try:
    from services.geo_atlas.geo_loader import (
        load_india_states_geojson,
        load_state_districts_geojson,
        load_all_districts_geojson,
        get_state_admin_info,
        get_district_bounds_and_center,
        normalize_state_name,
        normalize_district_name,
    )
except ImportError:
    import importlib
    import sys
    if "services.geo_atlas.geo_loader" in sys.modules:
        importlib.reload(sys.modules["services.geo_atlas.geo_loader"])
    from services.geo_atlas.geo_loader import (
        load_india_states_geojson,
        load_state_districts_geojson,
        load_all_districts_geojson,
        get_state_admin_info,
        get_district_bounds_and_center,
        normalize_state_name,
        normalize_district_name,
    )

# India national overview viewport
INDIA_DEFAULT_VIEW = {"latitude": 21.8, "longitude": 79.2, "zoom": 4.05}

BASEMAP_STYLES: Dict[str, Optional[str]] = {
    "Vector Boundary Canvas (Offline-Safe)": None,
    "Carto Dark Matter (Web Tiles)": "dark",
}

MAPLIBRE_STYLES: Dict[str, str] = {
    "Liberty (Default Vector)": "liberty",
    "Bright (High Contrast)": "bright",
    "Positron (Light Minimal)": "positron",
    "Dark (Graphite Night)": "dark",
}


def _zoom_from_bounds(bounds: List[float], max_zoom: float = 10.5, min_zoom: float = 4.2) -> float:
    """Approximates an appropriate PyDeck WebMercator zoom level from [minx, miny, maxx, maxy]."""
    if not bounds or len(bounds) != 4:
        return 6.2
    dx = max(0.05, abs(bounds[2] - bounds[0]))
    dy = max(0.05, abs(bounds[3] - bounds[1]))
    span = max(dx, dy)
    zoom = math.log2(360.0 / span) - 1.35
    return max(min_zoom, min(max_zoom, round(zoom, 2)))


def compute_view_state(
    selected_state: Optional[str] = None,
    selected_district: Optional[str] = None,
    selected_record: Optional[GeoInscriptionRecord] = None,
    picker_coords: Optional[Tuple[float, float]] = None,
) -> pdk.ViewState:
    """Computes smooth PyDeck ViewState for India, State, District, or Site selection."""
    if picker_coords and picker_coords[0] is not None and picker_coords[1] is not None:
        return pdk.ViewState(
            latitude=float(picker_coords[0]),
            longitude=float(picker_coords[1]),
            zoom=8.8,
            pitch=0,
        )

    if selected_record and selected_record.has_mappable_coordinates():
        return pdk.ViewState(
            latitude=float(selected_record.latitude),
            longitude=float(selected_record.longitude),
            zoom=9.6,
            pitch=0,
        )

    if selected_state and selected_district:
        d_info = get_district_bounds_and_center(selected_state, selected_district)
        if d_info:
            return pdk.ViewState(
                latitude=d_info["view_center_lat"],
                longitude=d_info["view_center_lon"],
                zoom=_zoom_from_bounds(d_info["bounds"], max_zoom=9.5, min_zoom=6.8),
                pitch=0,
            )

    if selected_state:
        s_info = get_state_admin_info(selected_state)
        if s_info:
            c = s_info.get("centroid", {})
            return pdk.ViewState(
                latitude=float(c.get("lat", INDIA_DEFAULT_VIEW["latitude"])),
                longitude=float(c.get("lon", INDIA_DEFAULT_VIEW["longitude"])),
                zoom=_zoom_from_bounds(s_info.get("bounds", []), max_zoom=7.6, min_zoom=5.2),
                pitch=0,
            )

    return pdk.ViewState(**INDIA_DEFAULT_VIEW, pitch=0)


def _choropleth_fill(count: int, max_count: int, is_selected: bool) -> List[int]:
    """Maps inscription count to a dark-theme bronze-amber RGBA choropleth scale."""
    if is_selected:
        return [197, 161, 90, 155]
    if count <= 0:
        return [32, 40, 50, 145]  # #202832 subtle surface fill
    ratio = min(1.0, count / max(1, max_count))
    r = int(88 + 118 * ratio)
    g = int(76 + 88 * ratio)
    b = int(48 + 42 * ratio)
    a = int(145 + 75 * ratio)
    return [r, g, b, a]


def cluster_pin_records(
    records: List[GeoInscriptionRecord],
    cell_deg: float = 0.6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Groups nearby mappable records using a spatial grid strictly over REAL site
    coordinates. District-only and unknown records are excluded before clustering.
    """
    mappable = [r for r in records if r.has_mappable_coordinates()]
    if not mappable:
        return [], []

    if cell_deg <= 0.0:
        singles = [_record_to_pin_dict(r) for r in mappable]
        return singles, []

    buckets: Dict[Tuple[int, int], List[GeoInscriptionRecord]] = {}
    for rec in mappable:
        gx = int(math.floor(float(rec.longitude) / cell_deg))
        gy = int(math.floor(float(rec.latitude) / cell_deg))
        buckets.setdefault((gx, gy), []).append(rec)

    singles: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []

    for bucket_recs in buckets.values():
        if len(bucket_recs) == 1:
            singles.append(_record_to_pin_dict(bucket_recs[0]))
        else:
            avg_lat = sum(float(r.latitude) for r in bucket_recs) / len(bucket_recs)
            avg_lon = sum(float(r.longitude) for r in bucket_recs) / len(bucket_recs)
            titles = ", ".join(r.title_en for r in bucket_recs[:4])
            if len(bucket_recs) > 4:
                titles += f" (+{len(bucket_recs) - 4} more)"
            states = sorted({r.state for r in bucket_recs if r.state})
            districts = sorted({r.district for r in bucket_recs if r.district})
            clusters.append({
                "cluster_count": len(bucket_recs),
                "cluster_label": str(len(bucket_recs)),
                "latitude": round(avg_lat, 5),
                "longitude": round(avg_lon, 5),
                "inscription_ids": [r.inscription_id for r in bucket_recs],
                "state": states[0] if len(states) == 1 else "",
                "district": districts[0] if len(districts) == 1 else "",
                "tooltip_html": (
                    f"<div style='font-family:Inter,sans-serif;'>"
                    f"<b style='color:#C5A15A;'>Archaeological Site Cluster ({len(bucket_recs)} inscriptions)</b><br/>"
                    f"<span>{titles}</span><br/>"
                    f"<i style='color:#A8B0BA;'>Click or zoom into state/district to inspect individual sites</i>"
                    f"</div>"
                ),
            })

    return singles, clusters


def _record_to_pin_dict(rec: GeoInscriptionRecord, is_selected: bool = False) -> Dict[str, Any]:
    """Formats a verified/approximate record into a PyDeck marker payload."""
    is_verified = rec.location_precision == LocationPrecision.VERIFIED.value
    fill_color = [76, 175, 133, 235] if is_verified else [197, 161, 90, 235]
    if is_selected:
        fill_color = [245, 208, 118, 255]
    site_label = rec.archaeological_site or rec.village or rec.district or rec.title_en
    loc_chain = " · ".join([x for x in [rec.archaeological_site, rec.district, rec.state] if x])
    return {
        "inscription_id": rec.inscription_id,
        "title_en": rec.title_en,
        "title_ta": rec.title_ta,
        "script": rec.script,
        "period": rec.period,
        "state": rec.state,
        "district": rec.district,
        "location_precision": rec.location_precision,
        "latitude": float(rec.latitude),
        "longitude": float(rec.longitude),
        "fill_color": fill_color,
        "radius": 14500 if is_selected else 10500,
        "site_label": site_label,
        "tooltip_html": (
            f"<div style='font-family:Inter,sans-serif; max-width:280px;'>"
            f"<b style='color:#C5A15A;'>{rec.title_en}</b><br/>"
            f"<span style='color:#F1F2F3;'>{rec.inscription_id} · {rec.script}</span><br/>"
            f"<span style='color:#A8B0BA;'>{loc_chain}</span><br/>"
            f"<span style='color:#7ee787;'>Precision: {rec.location_precision}</span>"
            f"</div>"
        ),
    }


def _annotate_states_geojson(
    state_counts: Dict[str, Dict[str, int]],
    selected_state: Optional[str],
) -> Dict[str, Any]:
    base = load_india_states_geojson()
    gj = copy.deepcopy(base)
    max_cnt = max([v.get("total", 0) for v in state_counts.values()], default=1)
    norm_sel = normalize_state_name(selected_state) if selected_state else ""

    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        st_nm = props.get("st_nm", "")
        st_code = props.get("st_code", "")
        counts = state_counts.get(st_nm, {"total": 0, "pinned": 0, "unpinned": 0})
        total = counts.get("total", 0)
        pinned = counts.get("pinned", 0)
        is_sel = bool(norm_sel and st_nm == norm_sel)

        fill = _choropleth_fill(total, max_cnt, is_sel)
        if norm_sel and is_sel:
            fill = [20, 26, 33, 55]
        props["inscription_count"] = total
        props["fill_color"] = fill
        props["line_color"] = [197, 161, 90, 235] if is_sel else [110, 126, 145, 165]
        props["line_width"] = 2.5 if is_sel else 1.0
        props["tooltip_html"] = (
            f"<div style='font-family:Inter,sans-serif;'>"
            f"<b style='color:#C5A15A;'>{st_nm}</b> <span style='color:#7E8691;'>(Code {st_code})</span><br/>"
            f"<span>Documented Inscriptions: <b>{total}</b> ({pinned} geo-pinned)</span><br/>"
            f"<i style='color:#A8B0BA;'>Click state to explore district boundaries</i>"
            f"</div>"
        )
    return gj


def _annotate_districts_geojson(
    selected_state: str,
    district_counts: Dict[tuple, Dict[str, int]],
    selected_district: Optional[str],
) -> Dict[str, Any]:
    norm_state = normalize_state_name(selected_state)
    base = load_state_districts_geojson(norm_state)
    gj = copy.deepcopy(base)
    norm_sel_dist = normalize_district_name(norm_state, selected_district) if selected_district else ""

    state_dist_counts = {
        dist: c for (st, dist), c in district_counts.items() if st == norm_state
    }
    max_cnt = max([v.get("total", 0) for v in state_dist_counts.values()], default=1)

    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        dist_nm = (props.get("district") or "").strip()
        dt_code = str(props.get("dt_code") or "")
        counts = state_dist_counts.get(dist_nm, {"total": 0, "pinned": 0, "district_only": 0})
        total = counts.get("total", 0)
        pinned = counts.get("pinned", 0)
        dist_only = counts.get("district_only", 0)
        is_sel = bool(norm_sel_dist and dist_nm.lower() == norm_sel_dist.lower())

        props["st_nm"] = norm_state
        props["inscription_count"] = total
        props["fill_color"] = _choropleth_fill(total, max_cnt, is_sel)
        props["line_color"] = [245, 208, 118, 255] if is_sel else [197, 161, 90, 175]
        props["line_width"] = 3.0 if is_sel else 1.2
        props["tooltip_html"] = (
            f"<div style='font-family:Inter,sans-serif;'>"
            f"<b style='color:#C5A15A;'>{dist_nm} District</b> · {norm_state}<br/>"
            f"<span style='color:#7E8691;'>Admin Code: {dt_code or 'Updated Unit'}</span><br/>"
            f"<span>Total Inscriptions: <b>{total}</b> (Pinned sites: {pinned}, District-only: {dist_only})</span>"
            f"</div>"
        )
    return gj


def build_atlas_deck(
    records: List[GeoInscriptionRecord],
    state_counts: Dict[str, Dict[str, int]],
    district_counts: Dict[tuple, Dict[str, int]],
    selected_state: Optional[str] = None,
    selected_district: Optional[str] = None,
    selected_inscription_id: Optional[str] = None,
    picker_coords: Optional[Tuple[float, float]] = None,
    map_style: Optional[str] = None,
    enable_clustering: bool = True,
) -> pdk.Deck:
    """Constructs the fallback PyDeck Deck object."""
    layers: List[pdk.Layer] = []

    states_gj = _annotate_states_geojson(state_counts, selected_state)
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            id="india_states",
            data=states_gj,
            pickable=True,
            stroked=True,
            filled=True,
            auto_highlight=True,
            highlight_color=[197, 161, 90, 90],
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            get_line_width="properties.line_width",
            line_width_units="pixels",
        )
    )

    if selected_state:
        districts_gj = _annotate_districts_geojson(selected_state, district_counts, selected_district)
        if districts_gj.get("features"):
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    id="state_districts",
                    data=districts_gj,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    auto_highlight=True,
                    highlight_color=[245, 208, 118, 95],
                    get_fill_color="properties.fill_color",
                    get_line_color="properties.line_color",
                    get_line_width="properties.line_width",
                    line_width_units="pixels",
                )
            )

    if selected_district or not enable_clustering:
        cell_deg = 0.0
    elif selected_state:
        cell_deg = 0.22
    else:
        cell_deg = 0.65

    singles, clusters = cluster_pin_records(records, cell_deg=cell_deg)
    for pin in singles:
        if selected_inscription_id and pin["inscription_id"] == selected_inscription_id:
            pin["fill_color"] = [245, 208, 118, 255]
            pin["radius"] = 14500

    if clusters:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="site_clusters",
                data=clusters,
                pickable=True,
                filled=True,
                stroked=True,
                get_position="[longitude, latitude]",
                get_fill_color=[197, 161, 90, 220],
                get_line_color=[241, 242, 243, 240],
                get_radius="18000 + cluster_count * 2500",
                radius_min_pixels=11,
                radius_max_pixels=26,
                line_width_min_pixels=1.5,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                id="site_cluster_labels",
                data=clusters,
                pickable=False,
                get_position="[longitude, latitude]",
                get_text="cluster_label",
                get_color=[17, 24, 32, 255],
                get_size=13,
                get_alignment_baseline="'center'",
            )
        )

    if singles:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="site_markers",
                data=singles,
                pickable=True,
                filled=True,
                stroked=True,
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_line_color=[17, 24, 32, 240],
                get_radius="radius",
                radius_min_pixels=6,
                radius_max_pixels=18,
                line_width_min_pixels=1.5,
            )
        )

    if picker_coords and picker_coords[0] is not None and picker_coords[1] is not None:
        picker_data = [{
            "latitude": float(picker_coords[0]),
            "longitude": float(picker_coords[1]),
            "tooltip_html": (
                f"<div style='font-family:Inter,sans-serif;'>"
                f"<b style='color:#60A5FA;'>Candidate Picked Coordinate</b><br/>"
                f"{picker_coords[0]:.5f}° N, {picker_coords[1]:.5f}° E"
                f"</div>"
            ),
        }]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="picker_preview_pin",
                data=picker_data,
                pickable=True,
                filled=True,
                stroked=True,
                get_position="[longitude, latitude]",
                get_fill_color=[96, 165, 250, 235],
                get_line_color=[241, 242, 243, 255],
                get_radius=12000,
                radius_min_pixels=8,
                radius_max_pixels=16,
                line_width_min_pixels=2,
            )
        )

    selected_rec_obj = next(
        (r for r in records if selected_inscription_id and r.inscription_id == selected_inscription_id),
        None,
    )
    view_state = compute_view_state(
        selected_state=selected_state,
        selected_district=selected_district,
        selected_record=selected_rec_obj,
        picker_coords=picker_coords,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={
            "html": "{tooltip_html}",
            "style": {
                "backgroundColor": "#171D24",
                "color": "#F1F2F3",
                "border": "1px solid #C5A15A",
                "borderRadius": "6px",
                "padding": "8px 10px",
                "fontSize": "12px",
            },
        },
    )


def extract_pydeck_selection(selection_event: Any) -> Dict[str, Any]:
    """Parses Streamlit PyDeck on_select event state into a clean selection payload."""
    result: Dict[str, Any] = {
        "kind": None,
        "state": None,
        "district": None,
        "inscription_id": None,
        "cluster_ids": [],
    }
    if not selection_event:
        return result

    sel = getattr(selection_event, "selection", None)
    if sel is None and isinstance(selection_event, dict):
        sel = selection_event.get("selection")
    if not sel:
        return result

    objects = getattr(sel, "objects", None)
    if objects is None and isinstance(sel, dict):
        objects = sel.get("objects", {})
    if not isinstance(objects, dict):
        return result

    if objects.get("site_markers"):
        obj = objects["site_markers"][0]
        result["kind"] = "inscription"
        result["inscription_id"] = obj.get("inscription_id")
        result["state"] = obj.get("state") or None
        result["district"] = obj.get("district") or None
        return result

    if objects.get("site_clusters"):
        obj = objects["site_clusters"][0]
        result["kind"] = "cluster"
        result["cluster_ids"] = obj.get("inscription_ids", [])
        result["state"] = obj.get("state") or None
        result["district"] = obj.get("district") or None
        return result

    if objects.get("state_districts"):
        raw_obj = objects["state_districts"][0]
        props = raw_obj.get("properties") if isinstance(raw_obj.get("properties"), dict) else raw_obj
        result["kind"] = "district"
        result["state"] = props.get("st_nm") or None
        result["district"] = props.get("district") or None
        return result

    if objects.get("india_states"):
        raw_obj = objects["india_states"][0]
        props = raw_obj.get("properties") if isinstance(raw_obj.get("properties"), dict) else raw_obj
        result["kind"] = "state"
        result["state"] = props.get("st_nm") or None
        return result

    return result


# ==============================================================================
# MAPLIBRE GL JS + OPENFREEMAP GEOJSON BUILDERS & UTILITIES
# ==============================================================================

def records_to_markers_geojson(
    records: List[GeoInscriptionRecord],
    selected_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transforms verified & approximate epigraphical records into a GeoJSON FeatureCollection.
    Strictly excludes district_only and unknown records to prevent coordinate fabrication.
    """
    features: List[Dict[str, Any]] = []
    for r in records:
        if not r.has_mappable_coordinates():
            continue
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r.longitude), float(r.latitude)],
            },
            "properties": {
                "inscription_id": r.inscription_id,
                "title_en": r.title_en,
                "title_ta": r.title_ta or "",
                "script": r.script,
                "period": r.period or "",
                "dynasty": r.dynasty or "",
                "site": r.archaeological_site or r.village or "",
                "district": r.district or "",
                "state": r.state or "",
                "precision": r.location_precision,
                "material": r.material or "",
                "is_selected": bool(selected_id and r.inscription_id == selected_id),
            },
        }
        features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def build_districts_geojson(
    selected_state: Optional[str] = None,
    district_counts: Optional[Dict[str, Any]] = None,
    selected_district: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a GeoJSON FeatureCollection for district boundaries with inscriptional counts.
    Loads per-state GeoJSON lazily when a state is active; loads composite districts otherwise.
    """
    counts = district_counts or {}
    raw_fc = None
    if selected_state:
        raw_fc = load_state_districts_geojson(selected_state)
    if not raw_fc or not raw_fc.get("features"):
        raw_fc = load_all_districts_geojson()

    if not raw_fc or not raw_fc.get("features"):
        return {"type": "FeatureCollection", "features": []}

    features: List[Dict[str, Any]] = []
    sel_dist_norm = (
        normalize_district_name(selected_state or "", selected_district).lower()
        if selected_district
        else ""
    )

    for f in raw_fc["features"]:
        props = dict(f.get("properties", {}))
        d_name = props.get("district") or props.get("dtname") or props.get("district_name") or ""
        s_name = props.get("st_nm") or props.get("stname") or props.get("state_name") or selected_state or ""

        # Extract count whether dictionary is dist_name -> int or (st, dist) -> dict
        c_val = counts.get(d_name, counts.get((s_name, d_name), 0))
        if isinstance(c_val, dict):
            c_int = c_val.get("total", 0)
        else:
            try:
                c_int = int(c_val)
            except Exception:
                c_int = 0

        is_sel = bool(sel_dist_norm and d_name.lower() == sel_dist_norm)

        features.append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": {
                **props,
                "district": d_name,
                "state": s_name,
                "count": c_int,
                "is_selected": is_sel,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def build_states_geojson(
    state_counts: Optional[Dict[str, Any]] = None,
    selected_state: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds a GeoJSON FeatureCollection for India States/UTs with inscription counts."""
    counts = state_counts or {}
    raw_fc = load_india_states_geojson()
    if not raw_fc or not raw_fc.get("features"):
        return {"type": "FeatureCollection", "features": []}

    features: List[Dict[str, Any]] = []
    sel_st_norm = normalize_state_name(selected_state).lower() if selected_state else ""

    for f in raw_fc["features"]:
        props = dict(f.get("properties", {}))
        s_name = props.get("st_nm") or props.get("state") or props.get("name") or ""

        c_val = counts.get(s_name, 0)
        if isinstance(c_val, dict):
            c_int = c_val.get("total", 0)
        else:
            try:
                c_int = int(c_val)
            except Exception:
                c_int = 0

        is_sel = bool(sel_st_norm and s_name.lower() == sel_st_norm)

        features.append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": {
                **props,
                "state": s_name,
                "st_nm": s_name,
                "count": c_int,
                "is_selected": is_sel,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def compute_maplibre_view_state(
    selected_state: Optional[str] = None,
    selected_district: Optional[str] = None,
    selected_record: Optional[GeoInscriptionRecord] = None,
    picker_coords: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Computes camera view state [longitude, latitude] and zoom for MapLibre GL JS.
    Note: MapLibre uses [lng, lat] coordinate order.
    """
    if picker_coords and picker_coords[0] is not None and picker_coords[1] is not None:
        return {
            "center": [float(picker_coords[1]), float(picker_coords[0])],
            "zoom": 8.8,
        }

    if selected_record and selected_record.has_mappable_coordinates():
        return {
            "center": [float(selected_record.longitude), float(selected_record.latitude)],
            "zoom": 10.2,
        }

    if selected_state and selected_district:
        d_info = get_district_bounds_and_center(selected_state, selected_district)
        if d_info:
            return {
                "center": [float(d_info["view_center_lon"]), float(d_info["view_center_lat"])],
                "zoom": 8.2,
            }

    if selected_state:
        s_info = get_state_admin_info(selected_state)
        if s_info:
            c = s_info.get("centroid", {})
            return {
                "center": [float(c.get("lon", 79.2)), float(c.get("lat", 21.8))],
                "zoom": 6.2,
            }

    return {"center": [79.2, 21.8], "zoom": 4.15}
