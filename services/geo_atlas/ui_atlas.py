"""
Streamlit UI Studio for the India Epigraphical Geographic Atlas.

Implements:
- Breadcrumb navigation (India -> State -> District -> Archaeological Site) + Reset Map
- Search & 8-facet epigraphical/geographic filtering
- Interactive PyDeck GIS map with state boundaries, lazy district choropleth,
  marker clustering, legend, basemap toggle, and point-picker preview
- Strict anti-fabrication disclosure for district_only / unknown records
- Rich Inscription Detail panel with thumbnail, 4-pillar provenance, and Workbench launch
- Manual registration & interactive coordinate validation / point-in-district verification
"""

import os
from typing import Callable, Optional, List, Tuple
from PIL import Image
import pandas as pd
import streamlit as st

from ui_components import render_section_header, render_metric_card, render_empty_state
from services.geo_atlas.models import (
    AtlasFilterCriteria,
    GeoInscriptionRecord,
    LocationContext,
    LocationPrecision,
)
try:
    from services.geo_atlas.geo_loader import (
        load_geo_metadata,
        load_all_districts_geojson,
        get_all_state_names,
        get_districts_for_state,
        get_district_bounds_and_center,
        find_admin_unit_for_point,
        normalize_state_name,
        normalize_district_name,
    )
except ImportError:
    import importlib
    import sys
    if "services.geo_atlas.geo_loader" in sys.modules:
        importlib.reload(sys.modules["services.geo_atlas.geo_loader"])
    from services.geo_atlas.geo_loader import (
        load_geo_metadata,
        load_all_districts_geojson,
        get_all_state_names,
        get_districts_for_state,
        get_district_bounds_and_center,
        find_admin_unit_for_point,
        normalize_state_name,
        normalize_district_name,
    )
from services.geo_atlas.validation import validate_coordinates
from services.geo_atlas.database import AtlasDatabase
from services.geo_atlas.map_builder import (
    BASEMAP_STYLES,
    MAPLIBRE_STYLES,
    build_atlas_deck,
    extract_pydeck_selection,
    records_to_markers_geojson,
    build_districts_geojson,
    build_states_geojson,
    compute_maplibre_view_state,
)
from services.geo_atlas.maplibre_component import maplibre_atlas

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@st.cache_resource
def get_atlas_db() -> AtlasDatabase:
    """Returns shared SQLite AtlasDatabase instance."""
    return AtlasDatabase()


@st.cache_data(show_spinner=False)
def cached_geo_metadata():
    return load_geo_metadata()


@st.cache_data(show_spinner=False)
def cached_all_states() -> List[str]:
    return get_all_state_names()


def _resolve_safe_image_path(rel_or_abs_path: str) -> Optional[str]:
    """Safely resolves local inscription thumbnail path inside workspace root."""
    if not rel_or_abs_path:
        return None
    cand = (
        rel_or_abs_path
        if os.path.isabs(rel_or_abs_path)
        else os.path.abspath(os.path.join(PROJECT_ROOT, rel_or_abs_path))
    )
    if not cand.startswith(PROJECT_ROOT):
        return None
    if os.path.isfile(cand):
        return cand
    return None


def _precision_badge_html(precision: str) -> str:
    colors = {
        LocationPrecision.VERIFIED.value: ("#4CAF85", "rgba(76,175,133,0.14)", "✓ Field-Verified Site"),
        LocationPrecision.APPROXIMATE.value: ("#C5A15A", "rgba(197,161,90,0.14)", "≈ Approximate Locality"),
        LocationPrecision.DISTRICT_ONLY.value: ("#D9A441", "rgba(217,164,65,0.14)", "◫ District-Only (No Site Pin)"),
        LocationPrecision.UNKNOWN.value: ("#A8B0BA", "rgba(168,176,186,0.12)", "? Regional / Unspecified"),
    }
    fg, bg, label = colors.get(precision, ("#A8B0BA", "rgba(168,176,186,0.12)", precision))
    return (
        f"<span style='display:inline-block; padding:3px 10px; border-radius:14px; "
        f"font-size:0.76rem; font-weight:600; color:{fg}; background:{bg}; "
        f"border:1px solid {fg}55;'>{label}</span>"
    )


def _init_atlas_session_state():
    defaults = {
        "atlas_selected_state": None,
        "atlas_selected_district": None,
        "atlas_selected_inscription_id": None,
        "atlas_last_pydeck_sig": None,
        "atlas_last_map_event_ts": None,
        "atlas_map_engine": "maplibre",
        "atlas_picker_lat": None,
        "atlas_picker_lon": None,
        "atlas_enable_clustering": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_atlas_view():
    st.session_state.atlas_selected_state = None
    st.session_state.atlas_selected_district = None
    st.session_state.atlas_selected_inscription_id = None
    st.session_state.atlas_last_pydeck_sig = None
    st.session_state.atlas_picker_lat = None
    st.session_state.atlas_picker_lon = None


def _render_breadcrumbs(selected_rec: Optional[GeoInscriptionRecord]):
    st.markdown(
        """
        <style>
        .atlas-crumb-bar {
            background: #171D24;
            border: 1px solid #2D3642;
            border-radius: 8px;
            padding: 8px 14px;
            margin-bottom: 12px;
            font-size: 0.88rem;
            color: #A8B0BA;
        }
        .atlas-crumb-active {
            color: #C5A15A;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sel_state = st.session_state.atlas_selected_state
    sel_dist = st.session_state.atlas_selected_district

    parts = ["🇮🇳 <b>India</b>"]
    if sel_state:
        parts.append(f"<span class='atlas-crumb-active'>{sel_state}</span>")
    if sel_dist:
        parts.append(f"<span class='atlas-crumb-active'>{sel_dist} District</span>")
    if selected_rec:
        site = selected_rec.archaeological_site or selected_rec.title_en
        parts.append(f"<span class='atlas-crumb-active'>{site}</span>")

    c_trail, c_back_state, c_reset = st.columns([3.2, 1.0, 0.9])
    with c_trail:
        st.markdown(f"<div class='atlas-crumb-bar'>{' &nbsp;→&nbsp; '.join(parts)}</div>", unsafe_allow_html=True)
    with c_back_state:
        if sel_dist and st.button("↑ Up to State", width="stretch", key="atlas_up_state_btn"):
            st.session_state.atlas_selected_district = None
            st.session_state.atlas_selected_inscription_id = None
            st.rerun()
        elif sel_state and not sel_dist and st.button("↑ All India", width="stretch", key="atlas_up_india_btn"):
            st.session_state.atlas_selected_state = None
            st.session_state.atlas_selected_inscription_id = None
            st.rerun()
    with c_reset:
        if st.button("🔄 Reset Map", width="stretch", key="atlas_reset_map_btn"):
            _reset_atlas_view()
            st.rerun()


def _render_inscription_detail(
    rec: GeoInscriptionRecord,
    on_open_in_workbench: Optional[Callable[[GeoInscriptionRecord], None]],
):
    st.markdown(
        f"""
        <div style='background:#202832; border:1px solid #2D3642; border-left:4px solid #C5A15A;
                    border-radius:8px; padding:14px 16px; margin-bottom:12px;'>
            <div style='font-size:0.78rem; color:#A8B0BA;'>{rec.inscription_id} · {rec.script}</div>
            <div style='font-size:1.12rem; font-weight:700; color:#F1F2F3; margin:2px 0;'>{rec.title_en}</div>
            <div style='font-size:0.95rem; color:#C5A15A; font-family:"Noto Sans Tamil",sans-serif; margin-bottom:8px;'>{rec.title_ta}</div>
            {_precision_badge_html(rec.location_precision)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if on_open_in_workbench and st.button(
        "🔬 Open Full Research Record in Workbench",
        type="primary",
        width="stretch",
        key=f"atlas_open_wb_{rec.inscription_id}",
    ):
        on_open_in_workbench(rec)
        return

    img_resolved = _resolve_safe_image_path(rec.image_path)
    if img_resolved:
        try:
            thumb = Image.open(img_resolved)
            st.image(thumb, caption=f"{rec.title_en} ({os.path.basename(img_resolved)})", width="stretch")
        except Exception:
            pass

    st.markdown("**Historical & Administrative Summary**")
    coord_text = (
        f"{rec.latitude:.5f}° N, {rec.longitude:.5f}° E"
        if rec.has_mappable_coordinates()
        else "None stored (no fabricated point coordinates)"
    )
    meta_rows = [
        ("Country / State / District", " · ".join([x for x in [rec.country, rec.state, rec.district] if x]) or "Regional"),
        ("Admin Codes", f"State: {rec.state_code or '—'} | District: {rec.district_code or '—'}"),
        ("Historical District", rec.historical_district_name or "—"),
        ("Taluk / Village", " / ".join([x for x in [rec.sub_district_taluk, rec.village] if x]) or "—"),
        ("Archaeological Site", rec.archaeological_site or "—"),
        ("Coordinates", coord_text),
        ("Period / Dynasty", f"{rec.period} · {rec.dynasty}" if rec.dynasty else rec.period),
        ("Material", rec.material or "—"),
        ("Transcription Status", rec.transcription_status or "—"),
    ]
    for label, val in meta_rows:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; gap:10px; "
            f"padding:4px 0; border-bottom:1px solid #252C34; font-size:0.83rem;'>"
            f"<span style='color:#7E8691;'>{label}</span>"
            f"<span style='color:#F1F2F3; text-align:right;'>{val}</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("🧭 Separated 4-Pillar Geographic Provenance", expanded=True):
        pillars = [
            ("1. Original Discovery Location", rec.discovery_location, rec.discovery_source),
            ("2. Collection / Survey Location", rec.collection_location, rec.collection_source),
            ("3. Current Repository / Museum", rec.current_repository_location, rec.current_repository),
            ("4. Photograph Capture Location", rec.photograph_capture_location, ""),
        ]
        for heading, ctx, src in pillars:
            st.markdown(f"**{heading}**")
            loc_str = " · ".join([x for x in [ctx.place_name, ctx.institution_or_site, ctx.district, ctx.state] if x])
            st.caption(loc_str or src or "Not separately documented")
            if ctx.notes:
                st.caption(f"Note: {ctx.notes}")

    if rec.publication_reference:
        st.markdown(f"**📚 Publication / Catalogue Reference:** {rec.publication_reference}")

    if rec.transcription_preview:
        with st.expander("✍️ Diplomatic Transcription & Translation Preview", expanded=False):
            st.markdown(f"<div style='font-family:\"Noto Sans Tamil\",sans-serif; color:#F1F2F3;'>{rec.transcription_preview}</div>", unsafe_allow_html=True)
            if rec.translation_preview:
                st.caption(rec.translation_preview)

    if rec.spatial_validation_notes:
        st.warning(f"Spatial validation note: {rec.spatial_validation_notes}")


def _render_register_location_tab(db: AtlasDatabase):
    st.markdown("##### Register New Inscription Location or Update Existing Provenance")
    st.caption(
        "Keep Discovery, Collection, Current Repository, and Photograph Capture separate. "
        "Select 'district_only' or 'unknown' if exact monument coordinates are unavailable — "
        "Kallvettu never substitutes district centroids as exact site pins."
    )

    all_recs = db.query_inscriptions()
    mode = st.radio(
        "Record Action",
        ["Register New Inscription Record", "Update Existing Inscription Record"],
        horizontal=True,
        key="atlas_reg_mode",
    )

    existing_rec: Optional[GeoInscriptionRecord] = None
    if mode == "Update Existing Inscription Record" and all_recs:
        ids = [r.inscription_id for r in all_recs]
        id_labels = {r.inscription_id: f"{r.inscription_id} — {r.title_en}" for r in all_recs}
        chosen_id = st.selectbox(
            "Select Record to Update",
            ids,
            format_func=lambda i: id_labels.get(i, i),
            key="atlas_edit_target_id",
        )
        existing_rec = db.get_inscription(chosen_id)

    all_states = cached_all_states()

    col_coord_tool1, col_coord_tool2 = st.columns([1.4, 1.0])
    with col_coord_tool1:
        default_state = existing_rec.state if existing_rec and existing_rec.state in all_states else "Tamil Nadu"
        reg_state = st.selectbox(
            "Primary State / UT (Discovery Administrative Unit)",
            [""] + all_states,
            index=(all_states.index(default_state) + 1) if default_state in all_states else 0,
            key="atlas_reg_state_select",
        )
    with col_coord_tool2:
        state_dists = get_districts_for_state(reg_state) if reg_state else []
        default_dist = existing_rec.district if existing_rec and existing_rec.district in state_dists else ""
        reg_district = st.selectbox(
            "Primary District",
            [""] + state_dists,
            index=(state_dists.index(default_dist) + 1) if default_dist in state_dists else 0,
            key="atlas_reg_dist_select",
        )

    # Interactive Coordinate Picker Helper (for point-in-district verification)
    with st.expander("🎯 Interactive Coordinate Entry & Point-in-District Validator", expanded=True):
        prec_options = [p.value for p in LocationPrecision]
        def_prec = existing_rec.location_precision if existing_rec else LocationPrecision.VERIFIED.value
        reg_precision = st.selectbox(
            "Location Precision Tier",
            prec_options,
            index=prec_options.index(def_prec) if def_prec in prec_options else 0,
            help="'district_only' and 'unknown' intentionally clear exact lat/lon so district centroids are never misused.",
            key="atlas_reg_precision",
        )

        pcol1, pcol2, pcol3 = st.columns([1, 1, 1.1])
        def_lat = float(existing_rec.latitude) if existing_rec and existing_rec.latitude is not None else 0.0
        def_lon = float(existing_rec.longitude) if existing_rec and existing_rec.longitude is not None else 0.0
        disabled_coords = reg_precision in (LocationPrecision.DISTRICT_ONLY.value, LocationPrecision.UNKNOWN.value)

        with pcol1:
            lat_input = st.number_input(
                "Latitude (° N)",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.atlas_picker_lat if st.session_state.atlas_picker_lat is not None else def_lat,
                format="%.5f",
                disabled=disabled_coords,
                key="atlas_reg_lat_input",
            )
        with pcol2:
            lon_input = st.number_input(
                "Longitude (° E)",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.atlas_picker_lon if st.session_state.atlas_picker_lon is not None else def_lon,
                format="%.5f",
                disabled=disabled_coords,
                key="atlas_reg_lon_input",
            )
        with pcol3:
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            if st.button(
                "📍 Preview & Validate Point on Map",
                disabled=disabled_coords,
                width="stretch",
                key="atlas_preview_coord_btn",
            ):
                st.session_state.atlas_picker_lat = float(lat_input)
                st.session_state.atlas_picker_lon = float(lon_input)
                if reg_state:
                    st.session_state.atlas_selected_state = reg_state
                if reg_district:
                    st.session_state.atlas_selected_district = reg_district
                st.rerun()

        if disabled_coords:
            st.info(
                f"Precision set to `{reg_precision}` — coordinates will be stored strictly as `NULL` (unpinned). "
                "This inscription will contribute to its District Choropleth count without a fabricated point marker."
            )
        elif lat_input != 0.0 or lon_input != 0.0:
            cval = validate_coordinates(lat_input, lon_input, state=reg_state, district=reg_district)
            if cval.errors:
                for err in cval.errors:
                    st.error(err)
            else:
                detected = f"{cval.matched_district}, {cval.matched_state}" if cval.matched_district else cval.matched_state or "Outside boundary"
                if cval.point_in_district is True:
                    st.success(f"✓ Coordinate validated inside **{detected}** (Admin codes {cval.matched_state_code}/{cval.matched_district_code}).")
                elif cval.warnings:
                    for w in cval.warnings:
                        st.warning(w)
                    st.caption(f"GeoJSON boundary resolver detected: **{detected}**")
                else:
                    st.info(f"GeoJSON boundary resolver detected: **{detected}**")

    with st.form("atlas_inscription_registration_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            reg_id = st.text_input(
                "Inscription ID *",
                value=existing_rec.inscription_id if existing_rec else "",
                disabled=bool(existing_rec),
                placeholder="e.g., TN_KKR_012",
            )
            reg_title_en = st.text_input(
                "English Title *",
                value=existing_rec.title_en if existing_rec else "",
                placeholder="e.g., Jambai Tamil-Brahmi Cavern Inscription",
            )
            reg_title_ta = st.text_input(
                "Tamil Title",
                value=existing_rec.title_ta if existing_rec else "",
            )
        with f2:
            reg_script = st.text_input(
                "Script",
                value=existing_rec.script if existing_rec else "Tamil-Brahmi",
            )
            reg_period = st.text_input(
                "Historical Period",
                value=existing_rec.period if existing_rec else "",
                placeholder="e.g., c. 1st Century CE",
            )
            reg_dynasty = st.text_input(
                "Dynasty / Patron",
                value=existing_rec.dynasty if existing_rec else "",
            )
        with f3:
            reg_material = st.text_input(
                "Inscribed Material",
                value=existing_rec.material if existing_rec else "Granite Stone",
            )
            reg_hist_dist = st.text_input(
                "Historical District / Division Name",
                value=existing_rec.historical_district_name if existing_rec else "",
                placeholder="e.g., South Arcot / Chola Mandalam",
            )
            reg_pub_ref = st.text_input(
                "Publication / Catalogue Reference",
                value=existing_rec.publication_reference if existing_rec else "",
                placeholder="e.g., ASI ARIE / Epigraphia Indica Vol. ...",
            )

        a1, a2, a3 = st.columns(3)
        with a1:
            reg_taluk = st.text_input(
                "Sub-District / Taluk",
                value=existing_rec.sub_district_taluk if existing_rec else "",
            )
        with a2:
            reg_village = st.text_input(
                "Village / Town",
                value=existing_rec.village if existing_rec else "",
            )
        with a3:
            reg_site = st.text_input(
                "Archaeological Site / Monument",
                value=existing_rec.archaeological_site if existing_rec else "",
            )

        st.markdown("###### Separated 4-Pillar Location Provenance")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            disc_place = st.text_input(
                "1. Original Discovery Location",
                value=existing_rec.discovery_location.place_name if existing_rec else "",
                placeholder="e.g., Cavern brow above village tank",
            )
            disc_source = st.text_input(
                "Discovery Source / Explorer",
                value=existing_rec.discovery_source if existing_rec else "",
            )
            coll_place = st.text_input(
                "2. Collection / Survey Provenance",
                value=existing_rec.collection_location.place_name if existing_rec else "",
                placeholder="e.g., State Dept. Survey Estampage Unit",
            )
            coll_source = st.text_input(
                "Collection Source / Agency",
                value=existing_rec.collection_source if existing_rec else "",
            )
        with p_col2:
            repo_name = st.text_input(
                "3. Current Repository / Museum / Custodian",
                value=existing_rec.current_repository if existing_rec else "",
                placeholder="e.g., Government Museum, Egmore / In-situ",
            )
            repo_district = st.text_input(
                "Repository District / City",
                value=existing_rec.current_repository_location.district if existing_rec else "",
            )
            photo_place = st.text_input(
                "4. Photograph Capture Location",
                value=existing_rec.photograph_capture_location.place_name if existing_rec else "",
                placeholder="e.g., On-site field photograph / Archival squeeze",
            )
            coord_source = st.text_input(
                "Coordinate Source / Survey Citation",
                value=existing_rec.coordinate_source if existing_rec else "",
            )

        reg_transcription = st.text_area(
            "Diplomatic Transcription Preview (Optional)",
            value=existing_rec.transcription_preview if existing_rec else "",
            height=70,
        )

        submitted = st.form_submit_button("💾 Save Verified Inscription Location to Database", type="primary")
        if submitted:
            save_lat = None if disabled_coords else float(lat_input)
            save_lon = None if disabled_coords else float(lon_input)
            new_rec = GeoInscriptionRecord(
                inscription_id=reg_id.strip(),
                title_en=reg_title_en.strip(),
                title_ta=reg_title_ta.strip(),
                script=reg_script.strip(),
                period=reg_period.strip(),
                dynasty=reg_dynasty.strip(),
                material=reg_material.strip(),
                country="India",
                state=reg_state,
                district=reg_district,
                historical_district_name=reg_hist_dist.strip(),
                sub_district_taluk=reg_taluk.strip(),
                village=reg_village.strip(),
                archaeological_site=reg_site.strip(),
                latitude=save_lat,
                longitude=save_lon,
                location_precision=reg_precision,
                discovery_source=disc_source.strip(),
                collection_source=coll_source.strip(),
                current_repository=repo_name.strip(),
                publication_reference=reg_pub_ref.strip(),
                coordinate_source=coord_source.strip(),
                discovery_location=LocationContext(
                    place_name=disc_place.strip() or reg_site.strip(),
                    institution_or_site=reg_site.strip(),
                    state=reg_state,
                    district=reg_district,
                    taluk_or_city=reg_taluk.strip(),
                    village=reg_village.strip(),
                    latitude=save_lat,
                    longitude=save_lon,
                ),
                collection_location=LocationContext(
                    place_name=coll_place.strip() or coll_source.strip(),
                    institution_or_site=coll_source.strip(),
                ),
                current_repository_location=LocationContext(
                    place_name=repo_name.strip(),
                    institution_or_site=repo_name.strip(),
                    district=repo_district.strip(),
                ),
                photograph_capture_location=LocationContext(
                    place_name=photo_place.strip(),
                ),
                transcription_status=existing_rec.transcription_status if existing_rec else "Field Record Registered",
                transcription_preview=reg_transcription.strip(),
                translation_preview=existing_rec.translation_preview if existing_rec else "",
                image_path=existing_rec.image_path if existing_rec else "",
                preset_id=existing_rec.preset_id if existing_rec else "",
                is_curated_preset=existing_rec.is_curated_preset if existing_rec else False,
            )
            try:
                res = db.upsert_inscription(new_rec, validate_spatial=True, changed_by="Researcher")
                st.session_state.atlas_selected_inscription_id = new_rec.inscription_id
                if new_rec.state:
                    st.session_state.atlas_selected_state = new_rec.state
                if new_rec.district:
                    st.session_state.atlas_selected_district = new_rec.district
                st.session_state.atlas_picker_lat = None
                st.session_state.atlas_picker_lon = None
                if res.get("warnings"):
                    st.warning("Saved with spatial validation notices: " + " | ".join(res["warnings"]))
                else:
                    st.success(f"✓ {res['action']} completed cleanly for `{new_rec.inscription_id}`.")
                st.rerun()
            except ValueError as err:
                st.error(f"Validation failed: {err}")


def render_geographic_atlas_studio(
    on_open_in_workbench: Optional[Callable[[GeoInscriptionRecord], None]] = None,
):
    """Main entry point for Kallvettu's India Epigraphical Geographic Atlas Studio."""
    _init_atlas_session_state()
    db = get_atlas_db()
    meta = cached_geo_metadata()
    facets = db.get_filter_facets()

    render_section_header(
        title="India Epigraphical Geographic Atlas",
        subtitle="Interactive GIS spatial registry mapping ancient inscriptions across Indian States, Union Territories, and Districts with strict location provenance",
        icon="🗺️",
    )

    # Search & Multi-Facet Filter Bar
    with st.container():
        s_col1, s_col2, s_col3, s_col4 = st.columns([1.8, 1.0, 1.0, 0.9])
        with s_col1:
            search_q = st.text_input(
                "Search Atlas",
                placeholder="Search state, district, village, site, title, dynasty, or ID...",
                key="atlas_search_query",
            )
        with s_col2:
            state_opts = ["All"] + sorted(set(facets["states"]) | set(cached_all_states()))
            cur_st_val = st.session_state.atlas_selected_state or "All"
            if cur_st_val not in state_opts:
                cur_st_val = "All"
            filter_state = st.selectbox(
                "State / UT",
                state_opts,
                index=state_opts.index(cur_st_val),
                key="atlas_filter_state",
            )
            if filter_state != cur_st_val:
                st.session_state.atlas_selected_state = None if filter_state == "All" else filter_state
                st.session_state.atlas_selected_district = None
                st.session_state.atlas_selected_inscription_id = None
                st.rerun()

        with s_col3:
            if st.session_state.atlas_selected_state:
                dist_opts = ["All"] + get_districts_for_state(st.session_state.atlas_selected_state)
            else:
                dist_opts = ["All"] + facets["districts"]
            cur_dist_val = st.session_state.atlas_selected_district or "All"
            if cur_dist_val not in dist_opts:
                cur_dist_val = "All"
            filter_district = st.selectbox(
                "District",
                dist_opts,
                index=dist_opts.index(cur_dist_val),
                key="atlas_filter_district",
            )
            if filter_district != cur_dist_val:
                st.session_state.atlas_selected_district = None if filter_district == "All" else filter_district
                st.session_state.atlas_selected_inscription_id = None
                st.rerun()

        with s_col4:
            basemap_choice = st.selectbox(
                "Map Style (OpenFreeMap)",
                list(MAPLIBRE_STYLES.keys()),
                index=0,
                key="atlas_basemap_choice",
                help="OpenFreeMap vector tile styles hosted without API keys: Liberty (Google Maps alternative), Bright, Positron, Dark.",
            )

        with st.expander("🎛️ Additional Epigraphical & Provenance Filters", expanded=False):
            fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
            with fc1:
                filter_script = st.selectbox("Script", ["All"] + facets["scripts"], key="atlas_f_script")
            with fc2:
                filter_period = st.selectbox("Period", ["All"] + facets["periods"], key="atlas_f_period")
            with fc3:
                filter_dynasty = st.selectbox("Dynasty", ["All"] + facets["dynasties"], key="atlas_f_dynasty")
            with fc4:
                filter_material = st.selectbox("Material", ["All"] + facets["materials"], key="atlas_f_material")
            with fc5:
                filter_inst = st.selectbox("Repository / Inst.", ["All"] + facets["institutions"], key="atlas_f_inst")
            with fc6:
                filter_status = st.selectbox(
                    "Geo Verification",
                    ["All", "verified", "approximate", "district_only", "unknown"],
                    key="atlas_f_status",
                )

    filters = AtlasFilterCriteria(
        search_query=search_q,
        state=st.session_state.atlas_selected_state or "All",
        district=st.session_state.atlas_selected_district or "All",
        script=st.session_state.get("atlas_f_script", "All"),
        period=st.session_state.get("atlas_f_period", "All"),
        dynasty=st.session_state.get("atlas_f_dynasty", "All"),
        material=st.session_state.get("atlas_f_material", "All"),
        collection_institution=st.session_state.get("atlas_f_inst", "All"),
        verification_status=st.session_state.get("atlas_f_status", "All"),
    )

    # Query DB strictly from real records
    filtered_records = db.query_inscriptions(filters)
    state_counts = db.get_state_counts(
        AtlasFilterCriteria(
            search_query=filters.search_query,
            script=filters.script,
            period=filters.period,
            dynasty=filters.dynasty,
            material=filters.material,
            collection_institution=filters.collection_institution,
            verification_status=filters.verification_status,
        )
    )
    district_counts = db.get_district_counts(
        AtlasFilterCriteria(
            search_query=filters.search_query,
            state=filters.state,
            script=filters.script,
            period=filters.period,
            dynasty=filters.dynasty,
            material=filters.material,
            collection_institution=filters.collection_institution,
            verification_status=filters.verification_status,
        )
    )

    pinned_count = sum(1 for r in filtered_records if r.has_mappable_coordinates())
    district_only_cnt = sum(1 for r in filtered_records if not r.has_mappable_coordinates())
    represented_districts = len(district_counts)

    selected_rec = next(
        (r for r in filtered_records if r.inscription_id == st.session_state.atlas_selected_inscription_id),
        None,
    )

    # Top Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Matching Inscriptions", len(filtered_records), "Verified SQLite records")
    with k2:
        render_metric_card("Pinned Site Markers", pinned_count, "Verified & approximate site coords")
    with k3:
        render_metric_card("District-Only / Regional", district_only_cnt, "Aggregate only — no fake centroids")
    with k4:
        render_metric_card("Districts Represented", represented_districts, f"{meta.get('total_states_uts', 36)} Indian States & UTs mapped")

    _render_breadcrumbs(selected_rec)

    tab_explore, tab_register, tab_sources = st.tabs([
        "🗺️ Interactive Epigraphical Map & Inspector",
        "➕ Register / Update Location",
        "📜 Provenance Audit & Boundary Sources",
    ])

    with tab_explore:
        col_map, col_panel = st.columns([1.65, 1.15])

        with col_map:
            picker_coords: Optional[Tuple[float, float]] = None
            if st.session_state.atlas_picker_lat is not None and st.session_state.atlas_picker_lon is not None:
                picker_coords = (st.session_state.atlas_picker_lat, st.session_state.atlas_picker_lon)

            markers_gj = records_to_markers_geojson(
                filtered_records,
                selected_id=st.session_state.atlas_selected_inscription_id,
            )
            districts_gj = build_districts_geojson(
                selected_state=st.session_state.atlas_selected_state,
                district_counts=district_counts,
                selected_district=st.session_state.atlas_selected_district,
            )
            states_gj = build_states_geojson(
                state_counts=state_counts,
                selected_state=st.session_state.atlas_selected_state,
            )
            view_state = compute_maplibre_view_state(
                selected_state=st.session_state.atlas_selected_state,
                selected_district=st.session_state.atlas_selected_district,
                selected_record=selected_rec,
                picker_coords=picker_coords,
            )
            style_key = MAPLIBRE_STYLES.get(basemap_choice, "liberty")

            event = maplibre_atlas(
                markers_geojson=markers_gj,
                districts_geojson=districts_gj,
                states_geojson=states_gj,
                view_state=view_state,
                style=style_key,
                selected_state=st.session_state.atlas_selected_state,
                selected_district=st.session_state.atlas_selected_district,
                selected_inscription_id=st.session_state.atlas_selected_inscription_id,
                height=610,
                key="kallvettu_atlas_maplibre",
            )

            # Handle MapLibre bidirectional events cleanly
            if event and isinstance(event, dict):
                ev_type = event.get("event_type")
                ev_ts = event.get("timestamp")
                if ev_ts and ev_ts != st.session_state.get("atlas_last_map_event_ts"):
                    st.session_state.atlas_last_map_event_ts = ev_ts
                    if ev_type in ("marker_click", "inscription_select"):
                        insc_id = event.get("inscription_id")
                        if insc_id:
                            st.session_state.atlas_selected_inscription_id = insc_id
                            if event.get("state"):
                                st.session_state.atlas_selected_state = normalize_state_name(event["state"])
                            if event.get("district"):
                                st.session_state.atlas_selected_district = normalize_district_name(
                                    st.session_state.atlas_selected_state or "", event["district"]
                                )
                            st.rerun()
                    elif ev_type == "district_select":
                        d_name = event.get("district")
                        s_name = event.get("state")
                        if s_name:
                            st.session_state.atlas_selected_state = normalize_state_name(s_name)
                        if d_name:
                            st.session_state.atlas_selected_district = normalize_district_name(
                                st.session_state.atlas_selected_state or "", d_name
                            )
                        st.session_state.atlas_selected_inscription_id = None
                        st.rerun()
                    elif ev_type == "state_select":
                        s_name = event.get("state")
                        if s_name:
                            st.session_state.atlas_selected_state = normalize_state_name(s_name)
                            st.session_state.atlas_selected_district = None
                            st.session_state.atlas_selected_inscription_id = None
                            st.rerun()
                    elif ev_type in ("map_click", "coordinate_pick"):
                        coords = event.get("coordinates")
                        if coords and len(coords) == 2:
                            st.session_state.atlas_picker_lat = float(coords[0])
                            st.session_state.atlas_picker_lon = float(coords[1])
                            st.rerun()
                    elif ev_type == "reset_view":
                        _reset_atlas_view()
                        st.rerun()

            # Legend & Clustering toggle
            leg_c1, leg_c2 = st.columns([3.2, 1.1])
            with leg_c1:
                st.markdown(
                    """
                    <div style='background:#171D24; border:1px solid #2D3642; border-radius:6px;
                                padding:8px 12px; font-size:0.79rem; color:#A8B0BA; display:flex;
                                flex-wrap:wrap; gap:14px; align-items:center;'>
                        <span><b style='color:#4CAF85;'>●</b> Verified Monument</span>
                        <span><b style='color:#C5A15A;'>●</b> Approximate Locality</span>
                        <span><b style='color:#E0B86A;'>◉</b> Cluster (Zoom/Click)</span>
                        <span><b style='color:#9C7D40;'>■</b> District Choropleth Intensity</span>
                        <span style='color:#7E8691;'>District-only records excluded from point pins</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with leg_c2:
                st.session_state.atlas_enable_clustering = st.checkbox(
                    "Cluster nearby pins",
                    value=st.session_state.atlas_enable_clustering,
                    key="atlas_cluster_chk",
                )

        with col_panel:
            if selected_rec:
                if st.button("← Back to Inscription List", key="atlas_back_list_btn"):
                    st.session_state.atlas_selected_inscription_id = None
                    st.rerun()
                _render_inscription_detail(selected_rec, on_open_in_workbench)
            else:
                if st.session_state.atlas_selected_district:
                    st.markdown(f"#### 📍 {st.session_state.atlas_selected_district} District Roster")
                elif st.session_state.atlas_selected_state:
                    st.markdown(f"#### 🏛️ {st.session_state.atlas_selected_state} — Registered Corpus")
                    # Show district count breakdown for selected state
                    st_dist_rows = [
                        {
                            "District": d,
                            "Total": c["total"],
                            "Pinned": c["pinned"],
                            "District-Only": c["district_only"],
                        }
                        for (s, d), c in sorted(district_counts.items())
                        if s == st.session_state.atlas_selected_state
                    ]
                    if st_dist_rows:
                        st.dataframe(pd.DataFrame(st_dist_rows), width="stretch", hide_index=True)
                else:
                    st.markdown("#### 📜 Documented Archaeological Corpus")
                    st.caption("Click any State on the map, pick a District above, or inspect an inscription below.")

                if not filtered_records:
                    render_empty_state(
                        icon="🔍",
                        title="No Matching Inscriptions",
                        description="Adjust your filters or register a new archaeological site under the Register tab.",
                    )
                else:
                    for rec in filtered_records:
                        with st.container():
                            loc_badge = " · ".join(
                                [x for x in [rec.archaeological_site or rec.village, rec.district, rec.state] if x]
                            ) or "Regional"
                            c_txt, c_act = st.columns([2.6, 1.0])
                            with c_txt:
                                st.markdown(f"**{rec.title_en}**")
                                st.caption(f"{rec.inscription_id} · {rec.script} · {loc_badge}")
                                st.markdown(_precision_badge_html(rec.location_precision), unsafe_allow_html=True)
                            with c_act:
                                if st.button("Inspect", key=f"atlas_inspect_{rec.inscription_id}", width="stretch"):
                                    st.session_state.atlas_selected_inscription_id = rec.inscription_id
                                    if rec.state:
                                        st.session_state.atlas_selected_state = rec.state
                                    if rec.district:
                                        st.session_state.atlas_selected_district = rec.district
                                    st.rerun()
                            st.markdown("<hr style='margin:8px 0; border:0; border-top:1px solid #252C34;'>", unsafe_allow_html=True)

    with tab_register:
        _render_register_location_tab(db)

    with tab_sources:
        st.markdown("##### Geographic Boundary Dataset Attribution & Versioning")
        st.info(
            f"**Dataset:** {meta.get('dataset_name', 'India Administrative Boundaries')}  \n"
            f"**Version:** `{meta.get('dataset_version', 'N/A')}` | "
            f"**States/UTs:** `{meta.get('total_states_uts', 36)}` | "
            f"**Per-State Lazy GeoJSON Files:** `{meta.get('total_district_files', 36)}`  \n"
            f"**Source:** {meta.get('source_repository', 'N/A')}  \n"
            f"**Lineage:** {meta.get('source_lineage', 'N/A')}  \n"
            f"**License & Disclaimer:** {meta.get('license_notice', 'See datasets/geo/SOURCE_README.md')}"
        )

        st.markdown("##### SQLite Provenance Audit Log (Recent Changes)")
        audit_rows = db.get_audit_history(limit=25)
        if audit_rows:
            st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)
        else:
            st.caption("No audit entries recorded yet.")

