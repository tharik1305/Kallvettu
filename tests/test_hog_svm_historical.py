"""
Unit & Integration Tests for Phase 2 — HOG + SVM Historical Glyph Recognition.

Tests:
1. OpenCV HOG feature extraction, dimension validation, and preprocessing modes
2. Scikit-learn SVM classifier training on CPU with balanced class weights
3. Top-k candidate ranking, uncalibrated score vs calibrated probability separation
4. Unknown character rejection & margin-based uncertainty detection
5. Model persistence and lazy registry loading (ensuring checkpoint <= 25 MB budget)
6. HistoricalHOGSVMProvider integration with Kallvettu's BaseOCRProvider contract
"""

import io
import os
import tempfile
import unittest
import numpy as np
from PIL import Image

from services.research.historical_glyph_dataset import HistoricalGlyphDatasetManager
from services.ai.historical import (
    HOGConfig,
    prepare_glyph_image,
    extract_hog_features,
    HOGSVMGlyphClassifier,
    HistoricalModelRegistry,
    HistoricalHOGSVMProvider,
)
from services.ai.historical.train_glyph_model import train_glyph_model_for_script


def _draw_synthetic_glyph_image(shape_type: str = "cross", size: int = 64) -> Image.Image:
    """Creates a deterministic stone-like synthetic glyph patch for reproducible testing."""
    arr = np.full((size, size), 180, dtype=np.uint8)
    if shape_type == "cross":
        arr[size // 2 - 4 : size // 2 + 4, :] = 40
        arr[:, size // 2 - 4 : size // 2 + 4] = 40
    elif shape_type == "circle":
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
        mask = (dist >= 14) & (dist <= 22)
        arr[mask] = 40
    elif shape_type == "diagonal":
        for i in range(size):
            for t in range(-3, 4):
                if 0 <= i + t < size:
                    arr[i, i + t] = 40
    else:
        # Uniform noise
        arr = np.random.randint(150, 200, size=(size, size), dtype=np.uint8)
    return Image.fromarray(arr)


class TestHOGFeatures(unittest.TestCase):
    def test_feature_dimension_and_modes(self):
        cfg = HOGConfig()
        self.assertEqual(cfg.feature_dim, 1764)

        img = _draw_synthetic_glyph_image("cross", size=64)
        for mode in ("clahe", "otsu", "normalized_raw"):
            cfg_mode = HOGConfig(preprocess_mode=mode)
            feat = extract_hog_features(img, config=cfg_mode)
            self.assertEqual(feat.shape, (1764,))
            self.assertEqual(feat.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(feat)), 1.0, places=4)

    def test_dimension_validation_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            prepare_glyph_image(None)

        # Tiny image < 8x8 rejected
        tiny = np.zeros((6, 6), dtype=np.uint8)
        with self.assertRaises(ValueError):
            prepare_glyph_image(tiny)

        # Non-square input correctly normalized to canonical 64x64
        rect = np.ones((120, 40, 3), dtype=np.uint8) * 128
        norm_img = prepare_glyph_image(rect, target_size=(64, 64))
        self.assertEqual(norm_img.shape, (64, 64))


class TestGlyphClassifier(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _generate_synthetic_dataset(self, samples_per_class: int = 15):
        cfg = HOGConfig()
        classes = ["க", "ணி", "ய"]
        shapes = ["cross", "circle", "diagonal"]
        X_list = []
        y_list = []
        for cls_name, shp in zip(classes, shapes):
            for _ in range(samples_per_class):
                # Add slight variation
                base = _draw_synthetic_glyph_image(shp, size=64)
                arr = np.array(base)
                noise = np.random.randint(-15, 15, size=arr.shape)
                arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
                feat = extract_hog_features(arr, config=cfg)
                X_list.append(feat)
                y_list.append(cls_name)
        return np.array(X_list, dtype=np.float32), y_list

    def test_uncalibrated_mode_decision_scores_and_topk(self):
        X, y = self._generate_synthetic_dataset(samples_per_class=12)
        clf = HOGSVMGlyphClassifier(
            script="Tamil-Brahmi",
            kernel="linear",
            calibrate_probabilities=False,
        )
        meta = clf.fit(X, y)
        self.assertGreater(meta["train_accuracy"], 0.90)

        # Predict a cross glyph (expected 'க')
        test_img = _draw_synthetic_glyph_image("cross", size=64)
        pred = clf.predict_glyph(test_img, top_k=3)

        self.assertEqual(pred.predicted_character, "க")
        self.assertFalse(pred.is_calibrated)
        self.assertIsNone(pred.confidence)  # STRICT REQUIREMENT: No fake prob without calibration
        self.assertIsNone(pred.calibrated_probabilities)
        self.assertIn("க", pred.raw_decision_scores)
        self.assertEqual(len(pred.top_candidates), 3)

        # Raw score of top candidate should be highest
        self.assertGreater(pred.raw_decision_score, 0.0)

    def test_calibrated_probabilities_mode(self):
        X, y = self._generate_synthetic_dataset(samples_per_class=12)
        clf = HOGSVMGlyphClassifier(
            script="Tamil-Brahmi",
            kernel="linear",
            calibrate_probabilities=True,
        )
        clf.fit(X, y)

        test_img = _draw_synthetic_glyph_image("circle", size=64)
        pred = clf.predict_glyph(test_img, top_k=2)

        self.assertEqual(pred.predicted_character, "ணி")
        self.assertTrue(pred.is_calibrated)
        self.assertIsNotNone(pred.confidence)
        self.assertGreater(pred.confidence, 0.50)
        self.assertIsNotNone(pred.calibrated_probabilities)
        sum_probs = sum(pred.calibrated_probabilities.values())
        self.assertAlmostEqual(sum_probs, 1.0, places=3)

    def test_rejection_of_unknown_or_degraded_character(self):
        X, y = self._generate_synthetic_dataset(samples_per_class=10)
        # Set high rejection threshold above max OvR score (~2.23) so uncertain/low-margin sample rejects
        clf = HOGSVMGlyphClassifier(
            script="Tamil-Brahmi",
            kernel="linear",
            rejection_threshold=5.0,
        )
        clf.fit(X, y)

        test_img = _draw_synthetic_glyph_image("cross", size=64)
        pred = clf.predict_glyph(test_img)
        self.assertTrue(pred.is_rejected)
        self.assertEqual(pred.predicted_character, "[?]")
        self.assertIn("rejection threshold", pred.rejection_reason)

        # Also verify unknown noise pattern yields lower score than clean known glyph
        clf_norm = HOGSVMGlyphClassifier(
            script="Tamil-Brahmi",
            kernel="linear",
            rejection_threshold=2.0,
        )
        clf_norm.fit(X, y)
        clean_pred = clf_norm.predict_glyph(test_img)
        noise_pred = clf_norm.predict_glyph(_draw_synthetic_glyph_image("noise", size=64))
        self.assertFalse(clean_pred.is_rejected)
        self.assertLess(noise_pred.raw_decision_score, clean_pred.raw_decision_score)

    def test_model_persistence_and_budget_verification(self):
        X, y = self._generate_synthetic_dataset(samples_per_class=8)
        clf = HOGSVMGlyphClassifier(script="Tamil-Brahmi", kernel="linear")
        clf.fit(X, y)

        model_path = os.path.join(self.tmpdir.name, "tb_model.joblib")
        clf.save(model_path)

        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.assertLess(file_size_mb, 25.0)  # Well within the 25 MB budget (< 0.1 MB actual)

        reloaded = HOGSVMGlyphClassifier.load(model_path)
        self.assertEqual(reloaded.script, "Tamil-Brahmi")
        self.assertEqual(reloaded.classes_, clf.classes_)

        test_img = _draw_synthetic_glyph_image("cross", size=64)
        orig_pred = clf.predict_glyph(test_img)
        reloaded_pred = reloaded.predict_glyph(test_img)
        self.assertEqual(orig_pred.predicted_character, reloaded_pred.predicted_character)


class TestModelRegistryAndProviderIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.reg = HistoricalModelRegistry(models_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_registry_and_provider_lifecycle(self):
        # Empty registry check
        provider = HistoricalHOGSVMProvider(models_dir=self.tmpdir.name)
        self.assertFalse(provider.is_configured())
        ok, msg = provider.check_health()
        self.assertFalse(ok)

        # Train & register minimal model
        cfg = HOGConfig()
        classes = ["க", "த"]
        X = np.vstack([
            extract_hog_features(_draw_synthetic_glyph_image("cross", 64), cfg),
            extract_hog_features(_draw_synthetic_glyph_image("cross", 64), cfg),
            extract_hog_features(_draw_synthetic_glyph_image("diagonal", 64), cfg),
            extract_hog_features(_draw_synthetic_glyph_image("diagonal", 64), cfg),
        ])
        y = ["க", "க", "த", "த"]

        clf = HOGSVMGlyphClassifier(script="Tamil-Brahmi", kernel="linear")
        clf.fit(X, y)
        info = self.reg.register_and_save(clf)

        self.assertTrue(info.within_budget)
        self.assertIn("Tamil-Brahmi", self.reg.list_available_scripts())

        # Provider is now configured
        self.assertTrue(provider.is_configured())
        ok2, msg2 = provider.check_health()
        self.assertTrue(ok2)

        # Run provider recognize
        test_img = _draw_synthetic_glyph_image("cross", 64)
        res = provider.recognize(test_img, script_hint="Tamil-Brahmi")
        self.assertFalse(res.is_cloud)
        self.assertTrue(res.is_historical_script)
        self.assertIn("top_candidates", res.metadata)
        self.assertEqual(res.text, "க")


class TestEndToEndCLITraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ds_dir = os.path.join(self.tmpdir.name, "dataset")
        self.models_dir = os.path.join(self.tmpdir.name, "models")
        self.mgr = HistoricalGlyphDatasetManager(base_dir=self.ds_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_glyph_model_with_dataset(self):
        # Create 4 source plates, each containing multiple glyph classes (cross & circle regions)
        for s_idx in range(4):
            canvas = np.full((140, 140), 180, dtype=np.uint8)
            canvas[10:74, 10:74] = np.array(_draw_synthetic_glyph_image("cross", 64))
            canvas[70:134, 70:134] = np.array(_draw_synthetic_glyph_image("circle", 64))
            buf = io.BytesIO()
            Image.fromarray(canvas).save(buf, format="PNG")
            src = self.mgr.register_source_image(
                image_bytes=buf.getvalue(),
                original_filename=f"plate_{s_idx}.png",
                script="Tamil-Brahmi",
            )
            self.mgr.annotate_glyph(
                source_id=src.source_id,
                bbox_pixels=(10, 10, 74, 74),
                character_label="க",
                script="Tamil-Brahmi",
                approval_status="approved",
            )
            self.mgr.annotate_glyph(
                source_id=src.source_id,
                bbox_pixels=(70, 70, 134, 134),
                character_label="ணி",
                script="Tamil-Brahmi",
                approval_status="approved",
            )

        report = train_glyph_model_for_script(
            script="Tamil-Brahmi",
            dataset_dir=self.ds_dir,
            models_dir=self.models_dir,
            approved_only=True,
            test_ratio=0.25,
            val_ratio=0.25,
            seed=42,
        )

        self.assertEqual(report["script"], "Tamil-Brahmi")
        self.assertTrue(report["within_25mb_budget"])
        self.assertTrue(os.path.isfile(report["checkpoint_path"]))
        self.assertEqual(sorted(report["class_inventory"]), ["க", "ணி"])
        self.assertTrue(report["leakage_clean"])


if __name__ == "__main__":
    unittest.main()

