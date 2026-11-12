"""
CPU-First HOG + SVM Historical Inscription Glyph Classifier (Phase 2).

Implements scikit-learn Support Vector Machine (SVM) classification over
OpenCV HOG descriptors for individual ancient Tamil glyphs:
- Separate model instances per historical script
- Balanced class weighting for imbalanced historical collections
- Rejection of out-of-vocabulary or severely eroded characters
- Strict separation between raw SVM decision scores and calibrated probabilities
- Top-k candidate predictions with uncertainty flags
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from services.research.historical_glyph_dataset import normalize_tamil_unicode
from services.ai.historical.hog_features import (
    HOGConfig,
    DEFAULT_HOG_CONFIG,
    extract_hog_features,
)


@dataclass
class GlyphCandidate:
    """Alternative candidate prediction with associated score or probability."""
    character: str
    score: float             # Raw decision score OR calibrated probability
    is_calibrated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character": self.character,
            "score": round(self.score, 4),
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class GlyphPredictionResult:
    """Structured classification output for a single historical glyph."""
    predicted_character: str
    top_candidates: List[GlyphCandidate] = field(default_factory=list)
    raw_decision_scores: Dict[str, float] = field(default_factory=dict)
    calibrated_probabilities: Optional[Dict[str, float]] = None
    is_calibrated: bool = False
    confidence: Optional[float] = None     # ONLY set when calibrated; None otherwise
    raw_decision_score: float = 0.0        # Top raw SVM margin / decision score
    is_uncertain: bool = False
    is_rejected: bool = False
    rejection_reason: str = ""
    script: str = "Tamil-Brahmi"
    model_version: str = "1.0.0-hog-svm"
    feature_dimension: int = 1764

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_character": self.predicted_character,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "raw_decision_scores": {k: round(v, 4) for k, v in self.raw_decision_scores.items()},
            "calibrated_probabilities": (
                {k: round(v, 4) for k, v in self.calibrated_probabilities.items()}
                if self.calibrated_probabilities
                else None
            ),
            "is_calibrated": self.is_calibrated,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "raw_decision_score": round(self.raw_decision_score, 4),
            "is_uncertain": self.is_uncertain,
            "is_rejected": self.is_rejected,
            "rejection_reason": self.rejection_reason,
            "script": self.script,
            "model_version": self.model_version,
            "feature_dimension": self.feature_dimension,
        }


class HOGSVMGlyphClassifier:
    """
    Lightweight CPU classifier for individual historical script glyphs.
    Trained strictly on CPU using scikit-learn SVC + OpenCV HOGDescriptor.
    """

    def __init__(
        self,
        script: str = "Tamil-Brahmi",
        kernel: str = "linear",
        c_param: float = 1.0,
        class_weight: str = "balanced",
        calibrate_probabilities: bool = False,
        rejection_threshold: float = -0.35,
        margin_threshold: float = 0.12,
        hog_config: Optional[HOGConfig] = None,
        model_version: str = "1.0.0-hog-svm",
    ):
        self.script = script
        self.kernel = kernel
        self.c_param = c_param
        self.class_weight = class_weight
        self.calibrate_probabilities = calibrate_probabilities
        self.rejection_threshold = rejection_threshold
        self.margin_threshold = margin_threshold
        self.hog_config = hog_config or DEFAULT_HOG_CONFIG
        self.model_version = model_version

        # Core estimator & fitted attributes
        self.estimator: Optional[Any] = None
        self.classes_: List[str] = []
        self.is_fitted: bool = False
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_features: np.ndarray,
        y_labels: List[str],
        validation_split: Optional[Tuple[np.ndarray, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Fits SVM classifier on CPU.
        X_features: (N, feature_dim) float32 array
        y_labels: list of N Tamil Unicode character labels
        """
        if len(X_features) == 0 or len(y_labels) == 0:
            raise ValueError("Training dataset cannot be empty.")
        if len(X_features) != len(y_labels):
            raise ValueError(f"Feature count ({len(X_features)}) != label count ({len(y_labels)}).")

        normalized_y = [normalize_tamil_unicode(y) for y in y_labels]
        unique_classes = sorted(set(normalized_y))
        if len(unique_classes) < 2:
            raise ValueError(
                f"SVM classifier requires at least 2 distinct character classes; found {len(unique_classes)}."
            )

        base_svm = SVC(
            kernel=self.kernel,
            C=self.c_param,
            class_weight=self.class_weight,
            decision_function_shape="ovr",
            random_state=42,
        )

        if self.calibrate_probabilities:
            # Calibrated probabilities via Platt scaling
            cv_splits = min(3, min(normalized_y.count(c) for c in unique_classes))
            if cv_splits >= 2:
                model = CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=cv_splits)
            else:
                base_svm.probability = True
                model = base_svm
        else:
            model = base_svm

        model.fit(X_features, normalized_y)
        self.estimator = model
        self.classes_ = [str(c) for c in model.classes_]
        self.is_fitted = True

        # Validation evaluation if provided
        train_acc = float(np.mean(model.predict(X_features) == np.array(normalized_y)))
        val_acc: Optional[float] = None
        if validation_split is not None:
            X_val, y_val = validation_split
            if len(X_val) > 0:
                y_val_norm = np.array([normalize_tamil_unicode(y) for y in y_val])
                val_acc = float(np.mean(model.predict(X_val) == y_val_norm))

        self.training_metadata = {
            "script": self.script,
            "train_samples": len(X_features),
            "train_classes_count": len(self.classes_),
            "classes": self.classes_,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4) if val_acc is not None else None,
            "kernel": self.kernel,
            "c_param": self.c_param,
            "calibrated": self.calibrate_probabilities,
            "feature_dim": self.hog_config.feature_dim,
            "model_version": self.model_version,
        }
        return self.training_metadata

    def predict_glyph(
        self,
        glyph_image: Union[np.ndarray, Any, bytes, str],
        top_k: int = 3,
    ) -> GlyphPredictionResult:
        """
        Classifies an individual glyph image on CPU and returns structured candidates.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(
                f"Glyph classifier for script '{self.script}' is not fitted or loaded."
            )

        feat = extract_hog_features(glyph_image, self.hog_config).reshape(1, -1)
        classes = self.classes_

        # Raw decision scores (extract directly or average across CalibratedClassifierCV base SVMs)
        scores_vec = np.zeros(len(classes), dtype=float)
        if hasattr(self.estimator, "decision_function"):
            raw_scores = self.estimator.decision_function(feat)
            if raw_scores.ndim == 1:
                if len(classes) == 2:
                    score_val = float(raw_scores[0])
                    scores_vec = np.array([-score_val, score_val])
                else:
                    scores_vec = raw_scores
            else:
                scores_vec = raw_scores[0]
        elif hasattr(self.estimator, "calibrated_classifiers_") and self.estimator.calibrated_classifiers_:
            fold_scores = []
            for cal in self.estimator.calibrated_classifiers_:
                base_est = getattr(cal, "estimator", None) or getattr(cal, "base_estimator", None)
                if base_est is not None and hasattr(base_est, "decision_function"):
                    rs = base_est.decision_function(feat)
                    if rs.ndim == 1 and len(classes) == 2:
                        sv = float(rs[0])
                        fold_scores.append(np.array([-sv, sv]))
                    elif rs.ndim > 1:
                        fold_scores.append(rs[0])
                    else:
                        fold_scores.append(rs)
            if fold_scores:
                scores_vec = np.mean(fold_scores, axis=0)

        raw_score_dict = {cls_name: float(scores_vec[idx]) for idx, cls_name in enumerate(classes)}

        # Calibrated probabilities (strictly None if not calibrated)
        calibrated_dict: Optional[Dict[str, float]] = None
        probs_vec: Optional[np.ndarray] = None
        if hasattr(self.estimator, "predict_proba") and self.calibrate_probabilities:
            try:
                probs_vec = self.estimator.predict_proba(feat)[0]
                calibrated_dict = {cls_name: float(probs_vec[idx]) for idx, cls_name in enumerate(classes)}
            except Exception:
                calibrated_dict = None

        # Sort candidate ranking
        ranking_values = probs_vec if (probs_vec is not None and self.calibrate_probabilities) else scores_vec
        sorted_indices = np.argsort(ranking_values)[::-1]
        top_indices = sorted_indices[:max(1, min(top_k, len(classes)))]

        top_candidates: List[GlyphCandidate] = [
            GlyphCandidate(
                character=classes[idx],
                score=float(ranking_values[idx]),
                is_calibrated=bool(probs_vec is not None and self.calibrate_probabilities),
            )
            for idx in top_indices
        ]

        best_idx = top_indices[0]
        best_char = classes[best_idx]
        best_raw_score = float(scores_vec[best_idx])
        second_score = float(scores_vec[top_indices[1]]) if len(top_indices) > 1 else best_raw_score

        # Uncertainty & Unknown Character Rejection Logic
        is_rejected = False
        is_uncertain = False
        rejection_reason = ""

        # Check raw decision score against rejection floor
        if best_raw_score < self.rejection_threshold:
            is_rejected = True
            rejection_reason = (
                f"Best SVM decision score ({best_raw_score:.3f}) fell below rejection threshold ({self.rejection_threshold:.3f})."
            )

        # Margin ambiguity check
        score_margin = best_raw_score - second_score
        if (len(classes) > 1) and (score_margin < self.margin_threshold):
            is_uncertain = True

        pred_char = "[?]" if is_rejected else best_char

        return GlyphPredictionResult(
            predicted_character=pred_char,
            top_candidates=top_candidates,
            raw_decision_scores=raw_score_dict,
            calibrated_probabilities=calibrated_dict,
            is_calibrated=bool(calibrated_dict is not None),
            confidence=(float(probs_vec[best_idx]) if (probs_vec is not None and self.calibrate_probabilities) else None),
            raw_decision_score=best_raw_score,
            is_uncertain=is_uncertain,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            script=self.script,
            model_version=self.model_version,
            feature_dimension=self.hog_config.feature_dim,
        )

    def save(self, filepath: str) -> None:
        """Persists trained model and configuration to local storage (< 25 MB)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "script": self.script,
            "kernel": self.kernel,
            "c_param": self.c_param,
            "class_weight": self.class_weight,
            "calibrate_probabilities": self.calibrate_probabilities,
            "rejection_threshold": self.rejection_threshold,
            "margin_threshold": self.margin_threshold,
            "hog_config": self.hog_config.to_dict(),
            "model_version": self.model_version,
            "estimator": self.estimator,
            "classes_": self.classes_,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(bundle, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "HOGSVMGlyphClassifier":
        """Loads model bundle from local storage without retraining."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        bundle = joblib.load(filepath)
        hog_cfg = HOGConfig(**bundle.get("hog_config", {}))
        instance = cls(
            script=bundle.get("script", "Tamil-Brahmi"),
            kernel=bundle.get("kernel", "linear"),
            c_param=bundle.get("c_param", 1.0),
            class_weight=bundle.get("class_weight", "balanced"),
            calibrate_probabilities=bundle.get("calibrate_probabilities", False),
            rejection_threshold=bundle.get("rejection_threshold", -0.35),
            margin_threshold=bundle.get("margin_threshold", 0.12),
            hog_config=hog_cfg,
            model_version=bundle.get("model_version", "1.0.0-hog-svm"),
        )
        instance.estimator = bundle.get("estimator")
        instance.classes_ = bundle.get("classes_", [])
        instance.training_metadata = bundle.get("training_metadata", {})
        instance.is_fitted = instance.estimator is not None
        return instance
