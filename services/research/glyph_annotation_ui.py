"""
Lightweight Streamlit Historical Glyph Dataset & Annotation Studio (Phase 1).

Reuses OpenCV/Pillow image display overlays and restoration_engine enhancements
without heavy dependencies or external annotation servers.
"""

import os
from typing import Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from ui_components import render_section_header, render_metric_card, render_empty_state
from restoration_engine import apply_clahe_pillow
from services.research.historical_glyph_dataset import (
    HistoricalGlyphDatasetManager,
    SUPPORTED_HISTORICAL_SCRIPTS,
    VALID_UNCERTAINTY_STATUSES,
    VALID_APPROVAL_STATUSES,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "assets", "samples")


@st.cache_resource
def get_glyph_dataset_manager() -> HistoricalGlyphDatasetManager:
    return HistoricalGlyphDatasetManager()


def _draw_bbox_preview(pil_img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Draws selection box onto a display copy without altering original file."""
    display = pil_img.copy()
    max_disp = 1600
    scale = 1.0
    if max(display.size) > max_disp:
        scale = max_disp / float(max(display.size))
        display = display.resize(
            (max(1, int(display.width * scale)), max(1, int(display.height * scale))),
            Image.Resampling.BILINEAR,
        )
    arr = np.array(display.convert("RGB"))
    cv2.rectangle(
        arr,
        (int(x1 * scale), int(y1 * scale)),
        (int(x2 * scale), int(y2 * scale)),
        (197, 161, 90),
        thickness=max(2, int(2 * max(1.0, scale))),
    )
    return Image.fromarray(arr)


def render_glyph_annotation_studio():
    """Renders the Phase 1 Historical Glyph Dataset & Annotation Studio."""
    mgr = get_glyph_dataset_manager()
    stats = mgr.get_summary_stats()
    inventories = mgr.load_script_inventories()

    render_section_header(
        title="Historical Glyph Dataset & Annotation Studio",
        subtitle="Curate unmodified lithic plates, crop ancient glyphs, preserve Tamil Unicode & uncertainties, and verify leakage-free splits",
        icon="🏷️",
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Registered Source Plates", stats["total_sources"], "Byte-exact raw/ archive")
    with k2:
        render_metric_card("Annotated Glyphs", stats["total_annotations"], "JSONL Tamil Unicode records")
    with k3:
        render_metric_card("Researcher Approved", stats["approved_annotations"], "Verified training candidates")
    with k4:
        render_metric_card("Unique Glyph Classes", stats["unique_labels"], f"{stats['damaged_or_illegible']} damaged/illegible flagged")

    tab_upload, tab_annotate, tab_review = st.tabs([
        "1. Register Source Inscription",
        "2. Crop & Annotate Glyphs",
        "3. Review, Splits & Export",
    ])

    with tab_upload:
        st.markdown("##### Register Unmodified Inscription Photograph or Archival Plate")
        st.caption("Raw files are hashed (SHA-256) and stored unaltered in `datasets/historical_glyphs/raw/`.")

        src_mode = st.radio(
            "Source Input",
            ["Upload Field Photograph", "Import Existing Kallvettu Sample Plate"],
            horizontal=True,
            key="glyph_src_mode",
        )

        img_bytes: Optional[bytes] = None
        orig_name = ""
        if src_mode == "Upload Field Photograph":
            uploaded = st.file_uploader(
                "Upload lithic inscription image (JPG, PNG, TIFF, WEBP)",
                type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
                key="glyph_raw_uploader",
            )
            if uploaded is not None:
                img_bytes = uploaded.getvalue()
                orig_name = uploaded.name
        else:
            sample_files = sorted(
                f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ) if os.path.isdir(SAMPLES_DIR) else []
            if sample_files:
                chosen_sample = st.selectbox("Select Archival Sample", sample_files, key="glyph_sample_pick")
                orig_name = chosen_sample
                with open(os.path.join(SAMPLES_DIR, chosen_sample), "rb") as sf:
                    img_bytes = sf.read()

        with st.form("glyph_register_source_form"):
            c1, c2 = st.columns(2)
            with c1:
                src_script = st.selectbox("Primary Script", SUPPORTED_HISTORICAL_SCRIPTS, key="glyph_src_script")
                src_prov = st.text_input("Provenance / Monument Citation", placeholder="e.g., Mangulam Cave 1, ASI ARIE")
            with c2:
                src_lic = st.text_input("License", value="CC-BY-4.0 / Research Use")
                src_attr = st.text_input("Photographer / Dataset Attribution", placeholder="e.g., ASI / Field Survey")
            src_notes = st.text_input("Notes (Optional)")
            reg_btn = st.form_submit_button("💾 Register Unmodified Raw Image", type="primary")
            if reg_btn:
                if not img_bytes:
                    st.error("Please upload or choose an inscription image first.")
                else:
                    try:
                        rec = mgr.register_source_image(
                            image_bytes=img_bytes,
                            original_filename=orig_name,
                            script=src_script,
                            provenance=src_prov,
                            license_str=src_lic,
                            attribution=src_attr,
                            notes=src_notes,
                        )
                        st.session_state.glyph_active_source_id = rec.source_id
                        st.success(f"✓ Registered `{rec.source_id}` ({rec.width}×{rec.height}px, SHA-256: `{rec.sha256[:12]}...`).")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")

    with tab_annotate:
        sources = mgr.load_sources()
        if not sources:
            render_empty_state(
                icon="📷",
                title="No Raw Inscriptions Registered Yet",
                description="Register an inscription photograph or sample plate in Tab 1 first.",
            )
        else:
            src_ids = [s.source_id for s in sources]
            src_map = {s.source_id: s for s in sources}
            def_sid = st.session_state.get("glyph_active_source_id", src_ids[0])
            if def_sid not in src_ids:
                def_sid = src_ids[0]

            active_sid = st.selectbox(
                "Active Source Inscription",
                src_ids,
                index=src_ids.index(def_sid),
                format_func=lambda i: f"{i} — {src_map[i].original_filename} ({src_map[i].script}, {src_map[i].width}×{src_map[i].height})",
                key="glyph_ann_source_sel",
            )
            active_src = src_map[active_sid]
            ok_sha, sha_msg = mgr.verify_source_integrity(active_sid)
            if ok_sha:
                st.caption(f"🔒 {sha_msg} Provenance: {active_src.provenance or 'Not specified'} | License: {active_src.license}")
            else:
                st.warning(sha_msg)

            raw_abs = os.path.join(mgr.base_dir, active_src.raw_image_path)
            with Image.open(raw_abs) as raw_im:
                src_pil = raw_im.convert("RGB")
            img_w, img_h = src_pil.size

            col_box, col_form = st.columns([1.45, 1.15])
            with col_box:
                st.markdown("**Select Glyph Bounding Box (Original Pixel Coordinates)**")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    x1 = st.number_input("X Min", 0, max(0, img_w - 2), value=min(40, max(0, img_w // 4)), key=f"gx1_{active_sid}")
                with r2:
                    y1 = st.number_input("Y Min", 0, max(0, img_h - 2), value=min(40, max(0, img_h // 4)), key=f"gy1_{active_sid}")
                with r3:
                    x2 = st.number_input("X Max", 1, img_w, value=min(img_w, int(x1) + max(48, img_w // 8)), key=f"gx2_{active_sid}")
                with r4:
                    y2 = st.number_input("Y Max", 1, img_h, value=min(img_h, int(y1) + max(48, img_h // 6)), key=f"gy2_{active_sid}")

                x1_i, y1_i = int(x1), int(y1)
                x2_i, y2_i = max(x1_i + 4, min(img_w, int(x2))), max(y1_i + 4, min(img_h, int(y2)))

                preview_overlay = _draw_bbox_preview(src_pil, x1_i, y1_i, x2_i, y2_i)
                st.image(preview_overlay, caption=f"Selected Region: [{x1_i}, {y1_i}, {x2_i}, {y2_i}]", width="stretch")

                crop_pil = src_pil.crop((x1_i, y1_i, x2_i, y2_i))
                crop_clahe = apply_clahe_pillow(crop_pil, clip_limit=1.8, blend_strength=0.80)
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.image(crop_pil, caption=f"Original Crop ({crop_pil.width}×{crop_pil.height})", width=180)
                with pc2:
                    st.image(crop_clahe, caption="Mild CLAHE Preview", width=180)

            with col_form:
                st.markdown("**Glyph Unicode Label & Provenance**")
                ann_script = st.selectbox(
                    "Historical Script",
                    SUPPORTED_HISTORICAL_SCRIPTS,
                    index=SUPPORTED_HISTORICAL_SCRIPTS.index(active_src.script) if active_src.script in SUPPORTED_HISTORICAL_SCRIPTS else 0,
                    key="glyph_ann_script",
                )
                script_chars = inventories.get(ann_script, [])
                if script_chars:
                    st.caption("Inventory quick-reference: " + "  ".join(script_chars[:24]))

                with st.form("glyph_save_annotation_form"):
                    char_label = st.text_input("Tamil Character Label (Unicode) *", placeholder="e.g., க, ணி, 𑀓, கோ")
                    alt_raw = st.text_input("Alternative Readings (comma or | separated)", placeholder="e.g., கா | கூ")
                    u_col1, u_col2 = st.columns(2)
                    with u_col1:
                        uncertainty = st.selectbox("Condition / Uncertainty", VALID_UNCERTAINTY_STATUSES, key="glyph_uncert_sel")
                        variant = st.selectbox("Saved Crop Variant", ["raw_crop", "mild_clahe", "grayscale"])
                    with u_col2:
                        is_damaged = st.checkbox("Damaged Glyph", value=False)
                        is_illegible = st.checkbox("Illegible / Unresolved", value=False)

                    a_col1, a_col2 = st.columns(2)
                    with a_col1:
                        annotator = st.text_input("Annotator", value="Epigraphist")
                    with a_col2:
                        approval = st.selectbox("Approval Status", VALID_APPROVAL_STATUSES, index=1)

                    ann_notes = st.text_input("Epigraphical Notes")
                    save_ann = st.form_submit_button("💾 Save Glyph Annotation", type="primary")
                    if save_ann:
                        alts = [s.strip() for s in alt_raw.replace("|", ",").split(",") if s.strip()]
                        try:
                            saved = mgr.annotate_glyph(
                                source_id=active_sid,
                                bbox_pixels=(x1_i, y1_i, x2_i, y2_i),
                                character_label=char_label,
                                script=ann_script,
                                alternative_readings=alts,
                                uncertainty_status=uncertainty,
                                is_damaged=is_damaged,
                                is_illegible=is_illegible,
                                preprocessing_variant=variant,
                                annotator=annotator,
                                approval_status=approval,
                                approved_by=annotator if approval == "approved" else "",
                                notes=ann_notes,
                            )
                            st.success(f"✓ Saved `{saved.annotation_id}` -> `{saved.character_label}` ({saved.crop_path})")
                        except Exception as e:
                            st.error(f"Annotation error: {e}")

    with tab_review:
        anns = mgr.load_annotations()
        st.markdown("##### Saved Glyph Annotations")
        if anns:
            df = pd.DataFrame([
                {
                    "ID": a.annotation_id,
                    "Source": a.source_id,
                    "Script": a.script,
                    "Label": a.character_label,
                    "Alts": "|".join(a.alternative_readings),
                    "Status": a.uncertainty_status,
                    "Damaged": a.is_damaged,
                    "Illegible": a.is_illegible,
                    "Approval": a.approval_status,
                    "Crop": a.crop_path,
                }
                for a in anns
            ])
            st.dataframe(df, width="stretch", hide_index=True)

            r_col1, r_col2, r_col3 = st.columns([1.4, 1.0, 0.8])
            with r_col1:
                pick_ann = st.selectbox("Select Annotation to Review", [a.annotation_id for a in anns], key="glyph_rev_pick")
            with r_col2:
                new_status = st.selectbox("New Approval Status", VALID_APPROVAL_STATUSES, key="glyph_rev_status")
            with r_col3:
                st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
                if st.button("Apply Review", width="stretch", key="glyph_rev_apply"):
                    mgr.update_approval(pick_ann, new_status, approved_by="Researcher")
                    st.rerun()
        else:
            st.info("No character-region annotations recorded yet.")

        st.markdown("---")
        st.markdown("##### Source-Level Dataset Splitting & Data Leakage Verification")
        sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1.2])
        with sc1:
            test_r = st.slider("Test Ratio", 0.1, 0.4, 0.2, 0.05, key="glyph_test_ratio")
        with sc2:
            val_r = st.slider("Val Ratio", 0.0, 0.3, 0.2, 0.05, key="glyph_val_ratio")
        with sc3:
            split_seed = st.number_input("Seed", 0, 99999, 42, key="glyph_split_seed")
        with sc4:
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            if st.button("🔀 Build Source-Level Split", width="stretch", key="glyph_build_splits_btn"):
                manifest = mgr.generate_source_level_splits(test_ratio=test_r, val_ratio=val_r, seed=int(split_seed))
                if manifest.get("leakage_clean"):
                    st.success(f"✓ Leakage-free split created: {manifest['counts']}")
                else:
                    st.error(f"Leakage violations detected: {manifest.get('leakage_violations')}")

        is_clean, violations = mgr.verify_no_data_leakage()
        if is_clean:
            st.caption("✓ Zero cross-split source/synthetic leakage detected.")
        else:
            st.error("Leakage detected: " + "; ".join(violations))

        st.markdown("---")
        st.markdown("##### Export Dataset Metadata & Annotations")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "📥 Export Annotations & Provenance (.JSON)",
                data=mgr.export_annotations_json().encode("utf-8"),
                file_name="historical_glyph_annotations.json",
                mime="application/json",
                width="stretch",
            )
        with ex2:
            st.download_button(
                "📥 Export Annotations Table (.CSV)",
                data=mgr.export_annotations_csv().encode("utf-8-sig"),
                file_name="historical_glyph_annotations.csv",
                mime="text/csv",
                width="stretch",
            )

