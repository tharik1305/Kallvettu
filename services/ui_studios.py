"""
UI Studios Extension for Kallvettu Digital Epigraphy Platform (Phase L).

Implements interactive Streamlit studio renderers for:
1. Historical Inscription Search Studio (EpigraphicalSearchEngine)
2. Historical Knowledge Graph Studio (NetworkX & Vis.js)
3. IIIF Deep-Zoom Viewer Studio (OpenSeadragon)
4. Photometric RTI & Virtual Relighting Studio (RTIEngine)
5. EpiDoc TEI XML Export and Research Archive Generation
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
from typing import Dict, Any, List, Optional
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image

try:
    from services.database.connection import get_db_session
    from services.database.models import Inscription, ImageAsset, GlyphAnnotation, AnnotationStatus
    from services.database.repository import EpigraphicalRepository
    from services.search.search_engine import EpigraphicalSearchEngine, SearchResult
    HAS_DB = True
    _DB_IMPORT_ERR = ""
except ImportError as _e:
    HAS_DB = False
    _DB_IMPORT_ERR = str(_e)

try:
    from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph
    from services.knowledge_graph.graph_renderer import render_knowledge_graph_html
    HAS_GRAPH = True
    _GRAPH_IMPORT_ERR = ""
except ImportError as _e:
    HAS_GRAPH = False
    _GRAPH_IMPORT_ERR = str(_e)

from services.iiif.viewer_component import (
    render_openseadragon_viewer,
    image_to_base64_data_uri,
    get_image_dimensions,
    get_cached_squeeze_rubbing,
)
from services.iiif.manifest_builder import (
    IIIFManifestBuilder,
    resolve_image_asset,
    DEFAULT_IIIF_BASE_URI,
)
from services.iiif.coordinate_mapper import (
    parse_iiif_region,
    format_iiif_region,
    build_iiif_region_url,
    build_deep_link_url,
)
from services.research_tools.rti_engine import RTIEngine
from services.epidoc.exporter import EpiDocExporter
from services.epidoc.bundle import EpiDocArchiveBundler


def _clean_html(raw_html: str) -> str:
    """Strips leading/trailing whitespace per line to avoid markdown code-block bugs."""
    return "\n".join(l.strip() for l in raw_html.strip().splitlines() if l.strip())


# -----------------------------------------------------------------------------
# 1. Epigraphical Search Engine Studio
# -----------------------------------------------------------------------------
def render_epigraphical_search_studio() -> None:
    st.markdown("## 🔍 Historical Inscription Search Engine")
    st.caption("Multi-Criteria Corpus Search with Classical Tamil Morphological Variant Expansion")

    if not HAS_DB:
        st.error(f"⚠️ Search Engine requires database dependencies ({_DB_IMPORT_ERR}).")
        st.info("Please install missing requirements: `pip install -r requirements.txt`")
        return

    # Ensure database is seeded
    with get_db_session() as s:
        if s.query(Inscription).count() == 0:
            EpigraphicalRepository.seed_canonical_corpus()

    # Search query input
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Search Inscriptions (Tamil or English)",
            placeholder="e.g. குடவோலை, இராஜராஜ, மாங்குளம், Nelveli, Brihadisvara...",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔎 Search", width="stretch")

    # Filters
    with st.expander("⚙️ Advanced Epigraphical Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            script_filter = st.selectbox(
                "Script Category",
                ["ALL", "Tamil-Brahmi", "Medieval Chola Tamil", "Vatteluttu", "Pallava Grantha"],
            )
        with c2:
            dynasty_filter = st.selectbox(
                "Dynasty",
                ["ALL", "Chola Dynasty", "Pandya Dynasty", "Chera Dynasty", "Pallava Dynasty"],
            )
        with c3:
            district_filter = st.selectbox(
                "District",
                ["ALL", "Thanjavur", "Madurai", "Kanchipuram", "Karur", "Pudukkottai", "Chengalpattu"],
            )
        with c4:
            expand_morphology = st.checkbox("Expand Morphological Roots", value=True)

    # Perform search
    res: SearchResult = EpigraphicalSearchEngine.search(
        query=search_query,
        script_category=None if script_filter == "ALL" else script_filter,
        dynasty=None if dynasty_filter == "ALL" else dynasty_filter,
        district=None if district_filter == "ALL" else district_filter,
        expand_morphology=expand_morphology,
        page=1,
        page_size=20,
    )

    # Search stats and expanded keywords
    st.markdown("---")
    m1, m2 = st.columns([2, 5])
    with m1:
        st.markdown(f"**Found:** `{res.total_count}` Inscriptions")
    with m2:
        if res.expanded_terms and len(res.expanded_terms) > 1:
            term_pills = " ".join([
                f'<span style="background-color:#202832; border:1px solid #C5A15A; color:#C5A15A; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;">{html.escape(t)}</span>'
                for t in res.expanded_terms[:6]
            ])
            st.markdown(f"**Expanded Roots:** {term_pills}", unsafe_allow_html=True)

    # Display results
    if not res.items:
        st.info("No inscriptions found matching the current search criteria.")
        return

    for item in res.items:
        with st.container():
            card_html = f"""
            <div style="background-color:#1E252E; border:1px solid #2D3748; border-left:4px solid #C5A15A; border-radius:6px; padding:16px; margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:16px; font-weight:bold; color:#FFFFFF;">{html.escape(item.title_ta)} ({html.escape(item.title_en)})</span>
                <span style="background-color:#2D3748; color:#C5A15A; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold;">SCORE: {item.relevance_score}</span>
              </div>
              <div style="font-size:12px; color:#A0AEC0; margin-top:6px;">
                <b>ID:</b> {html.escape(item.inscription_id)} | <b>Script:</b> {html.escape(item.script_category)} | <b>Site:</b> {html.escape(item.site_name)}, {html.escape(item.district)} | <b>Era:</b> {html.escape(item.historical_period)}
              </div>
              {f'<div style="margin-top:8px; font-size:13px; color:#E2E8F0; background:#171D24; padding:8px; border-radius:4px;"><b>Context Excerpt:</b> {" ... ".join(item.matching_snippets)}</div>' if item.matching_snippets else ''}
              <div style="margin-top:8px; font-size:13px; color:#CBD5E0;">
                <b>Master Reading:</b> {html.escape(item.master_reading[:160])}...
              </div>
              <div style="margin-top:4px; font-size:12px; color:#A0AEC0;">
                <b>Citation:</b> {html.escape(item.citation or "Archaeological Survey of India")}
              </div>
            </div>
            """
            st.markdown(_clean_html(card_html), unsafe_allow_html=True)

            b_col1, b_col2, b_col3 = st.columns([2, 2, 2])
            with b_col1:
                if st.button(f"📥 Load into Workspace", key=f"load_{item.inscription_id}"):
                    st.session_state.current_image_name = item.title_en
                    st.session_state.analysis_output = {
                        "id": item.inscription_id,
                        "title_ta": item.title_ta,
                        "title_en": item.title_en,
                        "script_en": item.script_category,
                        "historical_period": item.historical_period,
                        "district": item.district,
                        "site_name": item.site_name,
                        "dynasty_en": item.dynasty,
                        "transcription": item.master_reading,
                        "translation": item.english_translation,
                        "citation": item.citation,
                    }
                    st.success(f"Loaded '{item.title_en}' into workspace!")
            with b_col2:
                # EpiDoc XML Download
                xml_data = EpiDocExporter.generate_tei_epidoc_xml(
                    inscription_id=item.inscription_id,
                    title_en=item.title_en,
                    title_ta=item.title_ta,
                    script_category=item.script_category,
                    historical_period=item.historical_period,
                    site_name=item.site_name,
                    district=item.district,
                    material_type=item.material_type,
                    century=item.century,
                    ruler_name=item.ruler_name,
                    dynasty=item.dynasty,
                    diplomatic_text=item.master_reading,
                    normalized_text=item.master_reading,
                    english_translation=item.english_translation,
                    citation_source=item.citation,
                )
                st.download_button(
                    "📄 Download EpiDoc XML",
                    data=xml_data,
                    file_name=f"{item.inscription_id}_epidoc.xml",
                    mime="application/xml",
                    key=f"epidoc_{item.inscription_id}",
                )
            with b_col3:
                # Research Bundle ZIP Download
                zip_data = EpiDocArchiveBundler.create_research_bundle_zip(
                    inscription_id=item.inscription_id,
                    title_en=item.title_en,
                    title_ta=item.title_ta,
                    script_category=item.script_category,
                    historical_period=item.historical_period,
                    site_name=item.site_name,
                    district=item.district,
                    material_type=item.material_type,
                    century=item.century,
                    ruler_name=item.ruler_name,
                    dynasty=item.dynasty,
                    diplomatic_text=item.master_reading,
                    normalized_text=item.master_reading,
                    english_translation=item.english_translation,
                    citation=item.citation,
                )
                st.download_button(
                    "📦 Full Research Bundle (ZIP)",
                    data=zip_data,
                    file_name=f"{item.inscription_id}_research_bundle.zip",
                    mime="application/zip",
                    key=f"zip_{item.inscription_id}",
                )


# -----------------------------------------------------------------------------
# 2. Historical Knowledge Graph Studio
# -----------------------------------------------------------------------------
def render_knowledge_graph_studio() -> None:
    st.markdown("## 🕸️ Historical Epigraphical Knowledge Graph")
    st.caption("Kings, Dynasties, Sacred Temples & Inscriptional Endowments (NetworkX & Vis.js)")

    if not HAS_GRAPH:
        st.error(f"⚠️ Knowledge Graph requires NetworkX dependencies ({_GRAPH_IMPORT_ERR}).")
        st.info("Please install missing requirements: `pip install -r requirements.txt`")
        return

    kg = EpigraphicalKnowledgeGraph()
    validation = kg.validate()
    if not validation["is_valid"]:
        st.error(f"⚠️ Knowledge Graph Validation Error: {', '.join(validation['errors'])}")
    elif validation.get("warnings"):
        with st.expander("ℹ️ Knowledge Graph Diagnostics", expanded=False):
            for w in validation["warnings"]:
                st.caption(f"• {w}")

    graph_dict = kg.to_dict()

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Entities", graph_dict["node_count"])
    with m2:
        st.metric("Verified Relations", graph_dict["edge_count"])
    with m3:
        st.metric("Imperial Dynasties", 4)
    with m4:
        st.metric("Sacred Inscription Sites", 6)

    # Render interactive Vis.js graph
    graph_html = render_knowledge_graph_html(kg, height_px=650)
    components.html(graph_html, height=660, scrolling=False)

    # Export options
    st.markdown("### 💾 Knowledge Graph Export")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "📥 Download Graph (JSON)",
            data=kg.to_json(),
            file_name="kallvettu_knowledge_graph.json",
            mime="application/json",
            width="stretch",
        )
    with e2:
        st.download_button(
            "📥 Download GraphML (XML)",
            data=kg.to_graphml(),
            file_name="kallvettu_knowledge_graph.graphml",
            mime="application/xml",
            width="stretch",
        )


# -----------------------------------------------------------------------------
# 3. IIIF Deep-Zoom Inscription Viewer Studio
# -----------------------------------------------------------------------------
def render_iiif_studio() -> None:
    st.markdown("## 🔎 IIIF Deep-Zoom Inscription Viewer")
    st.caption("OpenSeadragon 4.1.1 High-Resolution Tiled Inspection, Persistent Annotations & Spatial Sharing")

    if not HAS_DB:
        st.error(f"⚠️ Database dependencies unavailable ({_DB_IMPORT_ERR}).")
        return

    # Seed canonical plates with authentic assets and annotations if unpopulated
    try:
        EpigraphicalRepository.seed_canonical_corpus()
    except Exception as e:
        st.warning(f"Note: Corpus seeder check: {e}")

    # Build inscription list
    options: Dict[str, str] = {}

    # Check if active user image exists in session
    has_active_image = st.session_state.get("current_image") is not None
    if has_active_image:
        options["✨ Active Workspace Inscription (Current Upload/Restoration)"] = "ACTIVE_WORKSPACE"

    with get_db_session() as s:
        inscriptions = s.query(Inscription).all()
        for i in inscriptions:
            options[f"{i.id} — {i.title_en} ({i.script_category})"] = i.id

    if not options:
        st.warning("No inscriptions in catalog. Please seed corpus first.")
        return

    # Handle query params for deep-linking
    qp_insc = st.query_params.get("inscription_id", "")
    default_idx = 0
    if qp_insc:
        for idx, (label, iid) in enumerate(options.items()):
            if iid == qp_insc or qp_insc in label:
                default_idx = idx
                break

    sel_label = st.selectbox(
        "Select Inscription for Deep-Zoom Inspection",
        list(options.keys()),
        index=default_idx,
    )
    insc_id = options[sel_label]

    # Resolve master image, dimensions, metadata, annotations
    master_pil: Optional[Image.Image] = None
    master_asset_id: str = ""
    sha256_hash: str = ""
    script_category: str = "Tamil-Brahmi"
    dynasty_period: str = "Ancient Tamil"
    findspot_district: str = "Tamil Nadu"
    ruler_name: str = ""
    title_en: str = sel_label
    title_ta: str = sel_label
    description: str = ""

    if insc_id == "ACTIVE_WORKSPACE":
        master_pil = st.session_state.current_image
        buf = io.BytesIO()
        master_pil.save(buf, format="JPEG", quality=90)
        sha256_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        master_asset_id = "IMG-ACTIVE-WORKSPACE"
        script_category = "Active Session Script"
        dynasty_period = "Active Epigraphical Workbench"
        findspot_district = "Local Upload / Interactive Session"
        title_en = "Active Workspace Inscription"
        title_ta = "நடப்புப் பணியிடக் கல்வெட்டு"
    else:
        insc = EpigraphicalRepository.get_inscription(insc_id)
        if insc:
            title_en = insc.title_en
            title_ta = insc.title_ta
            script_category = insc.script_category
            dynasty_period = f"{insc.dynasty or 'Unknown'} &bull; {insc.historical_period or insc.century or ''}"
            findspot_district = f"{insc.site_name or ''}, {insc.district or ''}"
            ruler_name = insc.ruler_name or ""
            description = insc.description or ""

            # Locate ImageAsset
            resolved = resolve_image_asset(insc_id)
            if resolved and os.path.exists(resolved["file_path"]):
                master_pil = Image.open(resolved["file_path"])
                master_asset_id = resolved["asset_id"]
                sha256_hash = resolved["sha256_hash"]

    # Fallback if image path not found
    if master_pil is None:
        resolved = resolve_image_asset("KV-INSC-001")
        if resolved and os.path.exists(resolved["file_path"]):
            master_pil = Image.open(resolved["file_path"])
            master_asset_id = resolved["asset_id"]
            sha256_hash = resolved["sha256_hash"]
        else:
            arr = np.ones((480, 920, 3), dtype=np.uint8) * 80
            master_pil = Image.fromarray(arr)
            master_asset_id = "IMG-FALLBACK"
            sha256_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    w, h = master_pil.size

    # Epigraphical Dossier Badges
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"**📜 Script:** {script_category}")
    with b2:
        st.markdown(f"**👑 Dynasty/Era:** {dynasty_period}")
    with b3:
        st.markdown(f"**📍 Findspot:** {findspot_district}")
    with b4:
        st.markdown(f"**📐 Resolution:** {w} × {h} px &bull; `{sha256_hash[:10]}...`")

    # Load Saved Glyph Annotations from Persistent DB
    raw_db_annotations = []
    if insc_id != "ACTIVE_WORKSPACE":
        raw_db_annotations = EpigraphicalRepository.list_annotations_for_inscription(insc_id)

    formatted_annotations = []
    for a in raw_db_annotations:
        formatted_annotations.append({
            "id": a.id,
            "character_label": a.character_label,
            "unicode_representation": a.unicode_representation or a.character_label,
            "diplomatic_reading": a.diplomatic_reading,
            "script_category": a.script_category,
            "bounding_box": a.bounding_box,
            "polygon_coords": a.polygon_coords,
            "legibility_status": a.legibility_status or "CLEAR",
            "review_status": a.review_status.value if hasattr(a.review_status, "value") else str(a.review_status),
            "notes": a.notes or "",
        })

    # Parse initial shared region from URL if present
    qp_region = st.query_params.get("region", "")
    initial_region = None
    if qp_region:
        rx, ry, rw, rh = parse_iiif_region(qp_region, w, h)
        if rw > 0 and rh > 0:
            initial_region = {"x": rx, "y": ry, "w": rw, "h": rh}
            st.info(f"🔍 Restored shared spatial region: `x={rx}, y={ry}, w={rw}, h={rh}`")

    # Viewer display modes
    view_mode = st.radio(
        "Viewer Inspection Mode",
        [
            "Single Master High-Res Plate",
            "Synchronized Side-by-Side Comparison (Master vs Squeeze Rubbing)",
        ],
        horizontal=True,
    )

    if view_mode == "Single Master High-Res Plate":
        render_openseadragon_viewer(
            primary_image=master_pil,
            viewer_height=600,
            primary_label=f"{title_en} — Master Plate",
            annotations=formatted_annotations,
            initial_region=initial_region,
            image_dimensions=(w, h),
        )
    else:
        # Create Inked Squeeze Rubbing / Contrast Plate using cached, CPU-efficient OpenCV pipeline
        squeeze_pil, squeeze_uri = get_cached_squeeze_rubbing(
            master_pil,
            clip_limit=3.0,
            tile_grid=(8, 8),
            max_dim=1600,
            algo_version="2.1.0-squeeze-relief"
        )

        render_openseadragon_viewer(
            primary_image=master_pil,
            comparison_image=squeeze_uri if squeeze_uri else squeeze_pil,
            viewer_height=600,
            primary_label=f"{title_en} — Master Photograph",
            comparison_label="Lithic Squeeze Rubbing (Inverted Relief)",
            annotations=formatted_annotations,
            initial_region=initial_region,
            image_dimensions=(w, h),
        )

    # Sub-panels: Annotations, Region Sharing, Manifest
    tab_ann, tab_share, tab_manifest = st.tabs([
        f"🏷️ Glyph Annotations ({len(formatted_annotations)})",
        "🔗 IIIF Region Sharing & Deep-Links",
        "📜 IIIF Presentation 3.0 Manifest JSON",
    ])

    with tab_ann:
        st.markdown("##### Persistent Epigraphical Character Annotations")
        st.caption("Bounding boxes preserve original master photograph coordinates and survive session restarts.")

        # Metric counts
        clear_cnt = sum(1 for a in formatted_annotations if a["legibility_status"] == "CLEAR")
        weath_cnt = sum(1 for a in formatted_annotations if a["legibility_status"] == "WEATHERED")
        part_cnt = sum(1 for a in formatted_annotations if a["legibility_status"] == "PARTIAL")
        illeg_cnt = sum(1 for a in formatted_annotations if a["legibility_status"] == "ILLEGIBLE")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Glyphs", len(formatted_annotations))
        with c2:
            st.metric("Clear (தெளிவானவை)", clear_cnt)
        with c3:
            st.metric("Weathered / Eroded (தேய்ந்தவை)", weath_cnt)
        with c4:
            st.metric("Partial / Conjectural (பகுதி/ஊகம்)", part_cnt + illeg_cnt)

        # Existing annotations display
        if formatted_annotations:
            with st.expander("📋 View Saved Annotations Catalog", expanded=False):
                for idx, a in enumerate(formatted_annotations):
                    bb = a["bounding_box"]
                    st.markdown(
                        f"**{idx+1}. `{a['character_label']}`** ({a['diplomatic_reading']}) &bull; "
                        f"**Script:** {a['script_category']} &bull; "
                        f"**Status:** `{a['legibility_status']}` &bull; "
                        f"**Region:** `[x={bb.get('x')}, y={bb.get('y')}, w={bb.get('w')}, h={bb.get('h')}]` "
                        f"*(Notes: {a['notes'] or 'None'})*"
                    )

        # Add New Annotation Form
        if insc_id != "ACTIVE_WORKSPACE":
            with st.expander("➕ Annotate New Character Glyph Region", expanded=False):
                with st.form("add_glyph_form"):
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        char_in = st.text_input("Character Label (Tamil Unicode)", value="க")
                    with f2:
                        reading_in = st.text_input("Diplomatic Reading", value="ka")
                    with f3:
                        script_in = st.selectbox(
                            "Script Category",
                            ["Tamil-Brahmi", "Vatteluttu", "Early Chola Tamil", "Medieval Chola Tamil", "Pallava Grantha"],
                            index=0,
                        )
                    with f4:
                        legibility_in = st.selectbox(
                            "Legibility Status",
                            ["CLEAR", "WEATHERED", "PARTIAL", "ILLEGIBLE"],
                            index=0,
                        )

                    st.markdown("**Bounding Box Coordinates (Original Image Pixels):**")
                    coord1, coord2, coord3, coord4 = st.columns(4)
                    with coord1:
                        box_x = st.number_input("X Coordinate", min_value=0, max_value=max(0, w-1), value=min(50, w-1))
                    with coord2:
                        box_y = st.number_input("Y Coordinate", min_value=0, max_value=max(0, h-1), value=min(100, h-1))
                    with coord3:
                        box_w = st.number_input("Width", min_value=1, max_value=max(1, w - box_x), value=min(60, w - box_x))
                    with coord4:
                        box_h = st.number_input("Height", min_value=1, max_value=max(1, h - box_y), value=min(70, h - box_y))

                    notes_in = st.text_area("Scholarly / Palaeographical Notes", value="Epigraphical field observation.")
                    submit_ann = st.form_submit_button("💾 Save Glyph Annotation to Persistent Database", type="primary")

                    if submit_ann:
                        try:
                            EpigraphicalRepository.add_glyph_annotation(
                                inscription_id=insc_id,
                                image_asset_id=master_asset_id,
                                character_label=char_in,
                                unicode_char=char_in,
                                diplomatic_reading=reading_in,
                                script_category=script_in,
                                bounding_box={"x": int(box_x), "y": int(box_y), "w": int(box_w), "h": int(box_h)},
                                legibility_status=legibility_in,
                                notes=notes_in,
                            )
                            st.success(f"✓ Character glyph '{char_in}' saved permanently to database!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Failed to save annotation: {ex}")
        else:
            st.info("💡 To persist annotations permanently, select a registered inscription record from the database.")

    with tab_share:
        st.markdown("##### IIIF Image API 3.0 Spatial Region Sharing")
        st.caption("Generate standard IIIF region URLs and shareable deep-links that restore the viewer view.")

        sh_col1, sh_col2 = st.columns([1, 1])
        with sh_col1:
            st.markdown("**Target Spatial Region (Original High-Res Pixels):**")
            sh_x = st.number_input("Region X", 0, max(0, w-1), min(50, w-1), key="sh_rx")
            sh_y = st.number_input("Region Y", 0, max(0, h-1), min(100, h-1), key="sh_ry")
            sh_w = st.number_input("Region Width", 1, max(1, w - sh_x), min(120, w - sh_x), key="sh_rw")
            sh_h = st.number_input("Region Height", 1, max(1, h - sh_y), min(100, h - sh_y), key="sh_rh")

            # Generate URLs
            iiif_pixel_url = build_iiif_region_url(
                base_uri="https://kallvettu.org/api/v1/iiif/image",
                asset_id=master_asset_id,
                x=sh_x,
                y=sh_y,
                w=sh_w,
                h=sh_h,
                pct=False,
            )
            iiif_pct_url = build_iiif_region_url(
                base_uri="https://kallvettu.org/api/v1/iiif/image",
                asset_id=master_asset_id,
                x=sh_x,
                y=sh_y,
                w=sh_w,
                h=sh_h,
                pct=True,
                image_width=w,
                image_height=h,
            )
            deep_link = build_deep_link_url(
                base_url="http://localhost:8501",
                inscription_id=insc_id,
                x=sh_x,
                y=sh_y,
                w=sh_w,
                h=sh_h,
            )

            st.text_input("IIIF Image API Pixel URL ({x},{y},{w},{h}):", iiif_pixel_url)
            st.text_input("IIIF Image API Percentage URL (pct:x,y,w,h):", iiif_pct_url)
            st.text_input("Kallvettu Deep-Link Share URL:", deep_link)

        with sh_col2:
            st.markdown("**Cropped Region Preview:**")
            try:
                rx, ry, rw, rh = int(sh_x), int(sh_y), int(sh_w), int(sh_h)
                crop_pil = master_pil.crop((rx, ry, rx + rw, ry + rh))
                st.image(crop_pil, caption=f"Original Region Crop ({rw} × {rh} px)", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not crop region preview: {e}")

    with tab_manifest:
        st.markdown("##### Compliant IIIF Presentation API 3.0 Manifest")
        st.caption("Validated against official IIIF 3.0 schema: absolute URLs, canvas dimensions matching master photograph, and ImageService3 binding.")

        manifest = IIIFManifestBuilder.build_manifest_for_inscription(
            inscription_or_id=insc_id if insc_id != "ACTIVE_WORKSPACE" else "KV-INSC-001",
            base_uri=DEFAULT_IIIF_BASE_URI,
        )
        st.json(manifest)

        manifest_str = json.dumps(manifest, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download IIIF Presentation 3.0 Manifest (JSON-LD)",
            data=manifest_str,
            file_name=f"kallvettu_{insc_id}_manifest.json",
            mime="application/ld+json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# 4. Photometric RTI & Virtual Relighting Studio
# -----------------------------------------------------------------------------
def render_rti_studio() -> None:
    st.markdown("## 💡 Photometric RTI & Virtual Relighting Studio")
    st.caption("Simulate Dynamic Raking Light to Expose Weathered Carved Grooves (Pulli, Kombu, Serifs)")

    # Check if active image in session
    if st.session_state.get("current_image") is not None:
        source_pil = st.session_state.current_image
    else:
        # Create simulated carved granite block
        synthetic_stone = np.ones((400, 600, 3), dtype=np.uint8) * 110
        # Carve grooves
        synthetic_stone[180:220, 100:500] = 50
        synthetic_stone[120:280, 280:320] = 50
        source_pil = Image.fromarray(synthetic_stone)

    # Convert to BGR numpy
    source_rgb = np.array(source_pil)
    if len(source_rgb.shape) == 2:
        source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_GRAY2BGR)
    elif source_rgb.shape[2] == 4:
        source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGBA2BGR)
    else:
        source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        azimuth = st.slider("☀️ Virtual Light Azimuth Angle (θ)", min_value=0.0, max_value=360.0, value=45.0, step=5.0)
        elevation = st.slider("📐 Virtual Light Elevation Angle (φ)", min_value=10.0, max_value=85.0, value=35.0, step=5.0)
    with col_ctrl2:
        specular = st.slider("✨ Specular Highlight Gain (Chisel Reflectivity)", min_value=0.0, max_value=1.5, value=0.6, step=0.1)
        diffuse = st.slider("🔆 Diffuse Surface Gain", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    # Compute RTI
    normals = RTIEngine.compute_surface_normals(source_bgr, smooth_sigma=1.2, depth_scale=2.5)
    relit_bgr = RTIEngine.relight_surface(
        source_bgr,
        normals,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        diffuse_gain=diffuse,
        specular_gain=specular,
    )
    relit_rgb = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB)
    normal_map_rgb = RTIEngine.normal_map_to_color(normals)

    # Display comparison
    st.markdown("---")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("**Original Photograph / Squeeze**")
        st.image(source_pil, use_container_width=True)
    with v2:
        st.markdown(f"**Relit Surface (θ={azimuth}°, φ={elevation}°)**")
        st.image(relit_rgb, use_container_width=True)
    with v3:
        st.markdown("**Surface Normal Vector Field (Nx, Ny, Nz)**")
        st.image(normal_map_rgb, use_container_width=True)
