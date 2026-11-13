"""
Unit Tests for Advanced Research & Evaluation Tools (Phase J).
"""

import numpy as np
import pytest
from services.research_tools.batch_processor import BatchProcessor
from services.research_tools.rti_engine import RTIEngine
from services.research_tools.comparison_tool import EpigraphicalComparisonTool
from services.research_tools.citation_assistant import ScholarlyCitationAssistant
from services.research_tools.voice_narrator import VoiceNarratorService
from services.research_tools.provenance_viewer import ProvenanceViewer


def test_batch_processor():
    """Verify batch job creation and progress tracking."""
    processor = BatchProcessor(max_workers=2)
    fake_files = ["doc1.png", "doc2.png", "doc3.png"]

    job = processor.create_batch_job(fake_files, script_hint="Chola Tamil")
    assert job.total_count == 3
    assert job.progress_pct == 0.0

    # Test with custom dummy item processor
    def dummy_proc(item):
        return {"processed": True, "path": item.file_path}

    processor._run_job_worker(job, custom_processor=dummy_proc)
    assert job.completed_count == 3
    assert job.failed_count == 0
    assert job.progress_pct == 100.0
    assert job.status == "COMPLETED"


def test_rti_engine_relighting():
    """Verify surface normal generation and photometric relighting."""
    # Synthetic test image (64x64)
    img = np.ones((64, 64, 3), dtype=np.uint8) * 120
    # Carve a horizontal groove
    img[30:34, 10:50] = 50

    normals = RTIEngine.compute_surface_normals(img, smooth_sigma=1.0, depth_scale=2.0)
    assert normals.shape == (64, 64, 3)

    # Unit vector magnitude check
    magnitudes = np.linalg.norm(normals, axis=-1)
    np.testing.assert_allclose(magnitudes, 1.0, atol=1e-3)

    # Relight at 45 deg azimuth, 35 deg elevation
    relit = RTIEngine.relight_surface(img, normals, azimuth_deg=45.0, elevation_deg=35.0)
    assert relit.shape == (64, 64, 3)
    assert relit.dtype == np.uint8

    # Normal map color
    normal_rgb = RTIEngine.normal_map_to_color(normals)
    assert normal_rgb.shape == (64, 64, 3)


def test_comparison_tool():
    """Verify side-by-side comparison, Jaccard similarity, and HTML generation."""
    ref = "ஸ்வஸ்திஸ்ரீ கோப்பரகேஸரிபன்மற்க்கு யாண்டு"
    hyp = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரிபன்மர்க்கு யாண்டு"

    res = EpigraphicalComparisonTool.compare_readings(ref, hyp)
    assert res.similarity_ratio > 0.85
    assert res.jaccard_word_similarity >= 0.5
    assert len(res.diff_operations) > 0
    assert "<div" in res.side_by_side_html
    assert "Reading A" in res.side_by_side_html


def test_citation_assistant():
    """Verify citation formatting across scholarly standards."""
    cits = ScholarlyCitationAssistant.format_citation(
        series="SII",
        volume="2",
        number="1",
        pages="1-14",
        year=1891,
        title="Inscriptions of the Brihadisvara Temple",
        editor="E. Hultzsch",
    )

    assert "South Indian Inscriptions" in cits.epigraphical_standard
    assert "Vol. 2, No. 1 (1891), pp. 1-14" in cits.epigraphical_standard
    assert "@article" in cits.bibtex
    assert "<biblScope unit=\"volume\">2</biblScope>" in cits.tei_epidoc_bibl
    assert "Hultzsch" in cits.chicago


def test_voice_narrator():
    """Verify voice narrator payload and synthetic disclaimer."""
    payload = VoiceNarratorService.generate_narration_payload(
        inscription_id="KV-INSC-002",
        text_content="நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும்",
        language="ta",
    )

    assert payload.target_language == "ta-IN"
    assert payload.provenance_badge == "SYNTHETIC_AI_VOICE"
    assert "synthetic" in payload.synthetic_disclaimer.lower()
    assert "speechSynthesis.speak" in payload.browser_tts_js


def test_provenance_viewer():
    """Verify provenance timeline HTML construction."""
    events = [
        {
            "stage_name": "Optical Capture & Digitization",
            "agent_or_provider": "Tamil Nadu Dept of Archaeology",
            "timestamp": "2024-01-15T10:00:00Z",
            "details": "High-resolution RAW photographic plate capture.",
            "sha256": "abcdef1234567890abcdef1234567890",
        },
        {
            "stage_name": "K-IEFE Evidence Fusion",
            "agent_or_provider": "K-IEFE Arbitration Engine",
            "timestamp": "2024-01-15T10:05:00Z",
            "details": "Multi-engine consensus reached on 98.2% of glyph slots.",
            "sha256": "1234567890abcdef1234567890abcdef",
        },
    ]

    html = ProvenanceViewer.render_provenance_timeline_html("KV-INSC-002", events)
    assert "Epigraphical Provenance Chain: KV-INSC-002" in html
    assert "Optical Capture &amp; Digitization" in html
    assert "K-IEFE Evidence Fusion" in html
    assert "SHA-256" in html
