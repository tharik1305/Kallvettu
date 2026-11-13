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

import html
import io
import json
from typing import Dict, Any, List, Optional
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image

try:
    from services.database.connection import get_db_session
    from services.database.models import Inscription
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

from services.iiif.viewer_component import render_openseadragon_viewer, image_to_base64_data_uri
from services.iiif.manifest_builder import IIIFManifestBuilder
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
    components.html(graph_html, height=660)

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
    st.caption("OpenSeadragon 4.1.1 High-Resolution Tiled Image Inspection & Side-by-Side Comparison")

    with get_db_session() as s:
        inscriptions = s.query(Inscription).all()
        options = {f"{i.id} — {i.title_en}": i.id for i in inscriptions}

    if not options:
        st.warning("No inscriptions in catalog. Please seed corpus first.")
        return

    sel_label = st.selectbox("Select Inscription for Deep-Zoom Inspection", list(options.keys()))
    insc_id = options[sel_label]

    # Create dummy high-res image for viewer demo if no local raw image
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 40
    # Draw simulated stone bed texture
    img[200:300, 100:1100] = 70
    img[400:500, 100:1100] = 70
    pil_img = Image.fromarray(img)
    data_uri = image_to_base64_data_uri(pil_img)

    view_mode = st.radio("Viewer Mode", ["Single Master High-Res", "Synchronized Side-by-Side Comparison"], horizontal=True)

    if view_mode == "Single Master High-Res":
        render_openseadragon_viewer(
            primary_image=data_uri,
            viewer_height=600,
            annotations=[
                {"bounding_box": {"x": 120, "y": 210, "w": 80, "h": 80}, "character_label": "ஸ்வ (sva)", "script_category": "Chola Tamil"},
                {"bounding_box": {"x": 220, "y": 210, "w": 80, "h": 80}, "character_label": "ஸ்தி (sti)", "script_category": "Chola Tamil"},
                {"bounding_box": {"x": 320, "y": 210, "w": 80, "h": 80}, "character_label": "ஸ்ரீ (śrī)", "script_category": "Chola Tamil"},
            ],
        )
    else:
        # Comparison mode with inverted squeeze
        squeeze_img = Image.fromarray(255 - img)
        squeeze_uri = image_to_base64_data_uri(squeeze_img)
        render_openseadragon_viewer(
            primary_image=data_uri,
            comparison_image=squeeze_uri,
            viewer_height=600,
        )

    # IIIF Manifest Preview
    with st.expander("📜 IIIF Presentation 3.0 Manifest JSON", expanded=False):
        manifest = IIIFManifestBuilder.build_manifest(
            inscription_id=insc_id,
            label=sel_label,
            summary="High-resolution deep-zoom IIIF epigraphical asset.",
            image_url=f"/iiif/{insc_id}/full/full/0/default.jpg",
        )
        st.json(manifest)


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
