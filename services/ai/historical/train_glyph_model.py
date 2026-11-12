"""
CPU-First Training Script for Historical HOG + SVM Glyph Recognition (Phase 2).

Usage:
    python -m services.ai.historical.train_glyph_model --script Tamil-Brahmi

Reads curated glyph crops & annotations from HistoricalGlyphDatasetManager,
enforces leakage-free source-level partitioning, extracts OpenCV HOG features,
fits scikit-learn SVM on CPU, reports independent split metrics, and persists
lightweight model checkpoints. Never fabricates training or test samples.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from services.research.historical_glyph_dataset import (
    HistoricalGlyphDatasetManager,
    GlyphAnnotationRecord,
)
from services.ai.historical.hog_features import HOGConfig, extract_hog_features
from services.ai.historical.glyph_classifier import HOGSVMGlyphClassifier
from services.ai.historical.model_registry import HistoricalModelRegistry


def _load_split_samples(
    mgr: HistoricalGlyphDatasetManager,
    annotations: List[GlyphAnnotationRecord],
    hog_config: HOGConfig,
) -> Tuple[np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    for rec in annotations:
        if rec.is_illegible or rec.character_label == "[?]":
            continue
        crop_abs = os.path.join(mgr.base_dir, rec.crop_path)
        if not os.path.isfile(crop_abs):
            continue
        feat = extract_hog_features(crop_abs, config=hog_config)
        features.append(feat)
        labels.append(rec.character_label)
    if not features:
        return np.empty((0, hog_config.feature_dim), dtype=np.float32), []
    return np.vstack(features), labels


def train_glyph_model_for_script(
    script: str = "Tamil-Brahmi",
    dataset_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    kernel: str = "linear",
    c_param: float = 1.0,
    calibrate: bool = False,
    approved_only: bool = False,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end CPU training for a target historical script.
    Returns structured training report including checkpoint path & independent split metrics.
    """
    t0 = time.perf_counter()
    mgr = HistoricalGlyphDatasetManager(base_dir=dataset_dir)
    registry = HistoricalModelRegistry(models_dir=models_dir)

    all_anns = mgr.load_annotations(
        script=script,
        approval_status="approved" if approved_only else None,
    )
    valid_anns = [a for a in all_anns if not a.is_illegible and a.character_label != "[?]"]

    if len(valid_anns) < 4:
        raise ValueError(
            f"Insufficient labeled glyph annotations for script '{script}' "
            f"(found {len(valid_anns)}; annotate at least 4 samples across >=2 classes first)."
        )

    # Generate/load source-level splits to strictly prevent image/crop data leakage
    manifest = mgr.generate_source_level_splits(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    if not manifest.get("leakage_clean", True):
        raise RuntimeError(f"Split leakage detected: {manifest.get('leakage_violations')}")

    ann_by_id = {a.annotation_id: a for a in valid_anns}
    split_ids = manifest.get("annotation_splits", {})

    train_anns = [ann_by_id[aid] for aid in split_ids.get("train", []) if aid in ann_by_id]
    val_anns = [ann_by_id[aid] for aid in split_ids.get("val", []) if aid in ann_by_id]
    test_anns = [ann_by_id[aid] for aid in split_ids.get("test", []) if aid in ann_by_id]

    if not train_anns:
        raise ValueError(f"Training split is empty for script '{script}'. Add more source inscriptions.")

    hog_cfg = HOGConfig()
    X_train, y_train = _load_split_samples(mgr, train_anns, hog_cfg)
    X_val, y_val = _load_split_samples(mgr, val_anns, hog_cfg)
    X_test, y_test = _load_split_samples(mgr, test_anns, hog_cfg)

    clf = HOGSVMGlyphClassifier(
        script=script,
        kernel=kernel,
        c_param=c_param,
        class_weight="balanced",
        calibrate_probabilities=calibrate,
        hog_config=hog_cfg,
    )
    train_meta = clf.fit(
        X_train,
        y_train,
        validation_split=(X_val, y_val) if len(X_val) > 0 else None,
    )

    test_metrics: Dict[str, Any] = {"evaluated": False, "reason": "No held-out test samples available."}
    if len(X_test) > 0:
        y_pred = clf.estimator.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_pred))
        cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        test_metrics = {
            "evaluated": True,
            "test_samples": len(X_test),
            "top1_accuracy": round(test_acc, 4),
            "macro_f1": round(float(cls_report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        }

    saved_info = registry.register_and_save(clf)
    elapsed = time.perf_counter() - t0

    return {
        "script": script,
        "checkpoint_path": saved_info.filepath,
        "checkpoint_size_mb": saved_info.file_size_mb,
        "within_25mb_budget": saved_info.within_budget,
        "class_inventory": saved_info.classes,
        "training_metrics": train_meta,
        "test_metrics": test_metrics,
        "training_duration_seconds": round(elapsed, 4),
        "leakage_clean": manifest["leakage_clean"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train lightweight CPU HOG+SVM historical glyph classifier.")
    parser.add_argument("--script", default="Tamil-Brahmi", help="Target script name (e.g. Tamil-Brahmi, Vatteluttu)")
    parser.add_argument("--dataset-dir", default=None, help="Path to historical_glyphs dataset directory")
    parser.add_argument("--output-dir", default=None, help="Directory to store trained model checkpoint")
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf"], help="SVM kernel")
    parser.add_argument("--c-param", type=float, default=1.0, help="SVM regularization C")
    parser.add_argument("--calibrate", action="store_true", help="Enable Platt sigmoid calibration")
    parser.add_argument("--approved-only", action="store_true", help="Train strictly on researcher-approved glyphs")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Source-level test split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Source-level validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split RNG seed")

    args = parser.parse_args()
    try:
        report = train_glyph_model_for_script(
            script=args.script,
            dataset_dir=args.dataset_dir,
            models_dir=args.output_dir,
            kernel=args.kernel,
            c_param=args.c_param,
            calibrate=args.calibrate,
            approved_only=args.approved_only,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

