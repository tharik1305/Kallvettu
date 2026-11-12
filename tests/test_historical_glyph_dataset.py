"""
Unit & Integration Tests for Phase 1 — Lightweight Historical Glyph Dataset.

Verifies:
1. Required directory layout under datasets/historical_glyphs/
2. Original raw inscription image byte-exact preservation & SHA-256 verification
3. Tamil Unicode NFC accuracy & alternative readings / uncertainty / illegible flags
4. Researcher approval tracking & JSON / CSV export integrity
5. Source-inscription-level splitting and synthetic-to-parent leakage detection
"""

import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from PIL import Image

from services.research.historical_glyph_dataset import (
    HistoricalGlyphDatasetManager,
    normalize_tamil_unicode,
)


def _make_sample_png_bytes(width: int = 320, height: int = 180, color=(180, 175, 165)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHistoricalGlyphDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mgr = HistoricalGlyphDatasetManager(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_directory_structure_initialized(self):
        for sub in ("raw", "processed", "annotations", "synthetic", "splits", "metadata"):
            self.assertTrue(os.path.isdir(os.path.join(self.tmpdir.name, sub)))
        self.assertTrue(os.path.isfile(self.mgr.inventories_path))
        inv = self.mgr.load_script_inventories()
        self.assertIn("Tamil-Brahmi", inv)
        self.assertIn("Vatteluttu", inv)

    def test_raw_image_preserved_byte_for_byte(self):
        orig_bytes = _make_sample_png_bytes(400, 220, (120, 118, 110))
        expected_sha = hashlib.sha256(orig_bytes).hexdigest()

        src = self.mgr.register_source_image(
            image_bytes=orig_bytes,
            original_filename="mangulam_plate_1.png",
            script="Tamil-Brahmi",
            provenance="Mangulam Cave 1",
            license_str="CC-BY-4.0",
            attribution="ASI Test Archive",
        )
        self.assertEqual(src.sha256, expected_sha)
        raw_abs = os.path.join(self.tmpdir.name, src.raw_image_path)
        with open(raw_abs, "rb") as f:
            stored_bytes = f.read()
        self.assertEqual(stored_bytes, orig_bytes)

        ok_sha, _ = self.mgr.verify_source_integrity(src.source_id)
        self.assertTrue(ok_sha)

    def test_unicode_annotations_uncertainty_and_exports(self):
        orig_bytes = _make_sample_png_bytes(300, 200, (150, 145, 140))
        src = self.mgr.register_source_image(
            image_bytes=orig_bytes,
            original_filename="pugalur_plate.png",
            script="Tamil-Brahmi",
            provenance="Pugalur Arunattarmalai",
            license_str="ASI Open Research",
        )

        # Decomposed Tamil Unicode input (e.g. NFD-like split vowel/consonant vs NFC)
        label_nfd = "கொ"
        ann1 = self.mgr.annotate_glyph(
            source_id=src.source_id,
            bbox_pixels=(20, 30, 90, 110),
            character_label=label_nfd,
            script="Tamil-Brahmi",
            alternative_readings=["கா", "கி"],
            uncertainty_status="uncertain",
            is_damaged=True,
            preprocessing_variant="mild_clahe",
            annotator="Dr. Epigraphist",
            approval_status="pending",
            notes="Chisel edge chipped near top loop",
        )
        self.assertEqual(ann1.character_label, normalize_tamil_unicode(label_nfd))
        self.assertEqual(ann1.alternative_readings, ["கா", "கி"])
        self.assertTrue(ann1.is_damaged)
        self.assertFalse(ann1.is_illegible)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir.name, ann1.crop_path)))

        # Illegible glyph without label defaults cleanly to [?]
        ann2 = self.mgr.annotate_glyph(
            source_id=src.source_id,
            bbox_pixels=(110, 30, 160, 100),
            character_label="",
            script="Tamil-Brahmi",
            uncertainty_status="illegible",
            is_illegible=True,
            approval_status="approved",
            approved_by="Lead Epigraphist",
        )
        self.assertEqual(ann2.character_label, "[?]")
        self.assertTrue(ann2.is_illegible)

        # Approval transition
        self.assertTrue(self.mgr.update_approval(ann1.annotation_id, "approved", "Senior Reviewer"))
        loaded = self.mgr.load_annotations(approval_status="approved")
        self.assertEqual(len(loaded), 2)

        # JSON & CSV exports preserve Tamil Unicode
        json_out = json.loads(self.mgr.export_annotations_json())
        self.assertEqual(len(json_out["annotations"]), 2)
        self.assertEqual(json_out["annotations"][0]["character_label"], "கொ")

        csv_out = self.mgr.export_annotations_csv()
        rows = list(csv.DictReader(io.StringIO(csv_out)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["character_label"], "கொ")
        self.assertEqual(rows[0]["alternative_readings"], "கா|கி")
        self.assertEqual(rows[0]["provenance"], "Pugalur Arunattarmalai")

    def test_source_level_splits_and_leakage_detection(self):
        sources = []
        for idx, color in enumerate([(100, 100, 100), (120, 120, 120), (140, 140, 140), (160, 160, 160), (180, 180, 180)]):
            s = self.mgr.register_source_image(
                image_bytes=_make_sample_png_bytes(200, 120, color),
                original_filename=f"plate_{idx}.png",
                script="Vatteluttu",
            )
            sources.append(s)
            self.mgr.annotate_glyph(
                source_id=s.source_id,
                bbox_pixels=(10, 10, 60, 60),
                character_label="கோ",
                script="Vatteluttu",
            )

        # Add a synthetic variant linked strictly to sources[0]
        syn = self.mgr.annotate_glyph(
            source_id=sources[0].source_id,
            parent_source_id=sources[0].source_id,
            is_synthetic=True,
            bbox_pixels=(10, 10, 60, 60),
            character_label="கோ",
            script="Vatteluttu",
        )

        manifest = self.mgr.generate_source_level_splits(test_ratio=0.2, val_ratio=0.2, seed=42)
        self.assertTrue(manifest["leakage_clean"])
        self.assertEqual(manifest["leakage_violations"], [])
        self.assertEqual(sum(manifest["counts"].values()), 6)

        # Confirm synthetic child resides in exact same partition as its parent source
        parent_split = next(
            k for k, sids in manifest["source_splits"].items() if sources[0].source_id in sids
        )
        self.assertIn(syn.annotation_id, manifest["annotation_splits"][parent_split])

        # Intentional cross-split leakage injection must be caught
        bad_manifest = json.loads(json.dumps(manifest))
        other_split = "test" if parent_split != "test" else "train"
        bad_manifest["annotation_splits"][parent_split].remove(syn.annotation_id)
        bad_manifest["annotation_splits"][other_split].append(syn.annotation_id)
        ok_bad, violations = self.mgr.verify_no_data_leakage(bad_manifest)
        self.assertFalse(ok_bad)
        self.assertTrue(any("Leakage:" in v for v in violations))


if __name__ == "__main__":
    unittest.main()

