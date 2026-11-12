"""
BaseOCRProvider Adapter for Kallvettu's HOG + SVM Historical Glyph Classifier.

Integrates cleanly with ProviderManager and K-IEFE while maintaining:
- Lazy local model loading (zero downloads or training on startup)
- Explicit separation of raw SVM decision scores vs calibrated probabilities
- Honest rejection of unknown/out-of-vocabulary characters as unresolved `[?]`
"""

import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    OCRResult,
    BoundingBox,
)
from services.ai.common.error_handler import ProviderNotConfiguredError
from services.ai.historical.model_registry import HistoricalModelRegistry


class HistoricalHOGSVMProvider(BaseOCRProvider):
    """
    Lightweight CPU-only OCR provider recognizing individual cropped glyphs
    across ancient Tamil scripts using OpenCV HOG + scikit-learn SVM.
    """

    def __init__(self, models_dir: Optional[str] = None, default_script: str = "Tamil-Brahmi"):
        self._registry = HistoricalModelRegistry(models_dir=models_dir)
        self._default_script = default_script

    @property
    def provider_name(self) -> str:
        return "Historical HOG+SVM Glyph Classifier (CPU)"

    @property
    def provider_id(self) -> str:
        return "historical_hog_svm"

    def is_configured(self) -> bool:
        """True if at least one trained historical glyph model exists in local registry."""
        return len(self._registry.list_available_scripts()) > 0

    def check_health(self) -> Tuple[bool, str]:
        scripts = self._registry.list_available_scripts()
        if not scripts:
            return (
                False,
                "No trained HOG+SVM glyph model found locally. Run `python -m services.ai.historical.train_glyph_model` after annotating glyphs.",
            )
        return True, f"HOG+SVM CPU ready for scripts: {', '.join(scripts)}"

    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Classifies a single glyph crop on CPU. Returns top-k candidates, raw decision scores,
        optional calibrated probability, and explicit rejection/uncertainty flags.
        """
        t0 = time.perf_counter()
        target_script = script_hint or self._default_script
        clf = self._registry.get_classifier(target_script)

        if clf is None:
            available = self._registry.list_available_scripts()
            if available and (script_hint is None):
                target_script = available[0]
                clf = self._registry.get_classifier(target_script)

        if clf is None:
            raise ProviderNotConfiguredError(
                f"No HOG+SVM model checkpoint available for script '{target_script}'.",
                provider=self.provider_name,
            )

        top_k = int(kwargs.get("top_k", 3))
        mode = kwargs.get("mode", "auto")  # "glyph", "line", or "auto"

        # Check if caller requested line recognition or provided a full multi-line inscription
        if mode == "line" or kwargs.get("full_image", False):
            from services.ai.historical.line_recognizer import HistoricalLineRecognizer, LineRecognizerConfig
            line_cfg = kwargs.get("line_config") or LineRecognizerConfig(top_k_candidates=top_k)
            recognizer = HistoricalLineRecognizer(classifier=clf, config=line_cfg)
            res = recognizer.recognize_lines(image, script_hint=target_script)
            latency = time.perf_counter() - t0

            boxes = [line.bounding_box for line in res.lines]
            return OCRResult(
                text=res.text,
                provider_name=self.provider_name,
                model_id=f"{clf.model_version}-line",
                confidence=res.confidence,
                latency_seconds=latency,
                bounding_boxes=boxes,
                raw_response=res.to_dict(),
                is_cloud=False,
                is_historical_script=True,
                provenance_notes=(
                    f"HOG+SVM Line Recognizer CPU (script={target_script}, "
                    f"lines={len(res.lines)}, total_chars={res.total_characters}, unresolved={res.unresolved_characters})"
                ),
                metadata={
                    "script": target_script,
                    "mode": "line",
                    "lines": [line.to_dict() for line in res.lines],
                    "total_characters": res.total_characters,
                    "resolved_characters": res.resolved_characters,
                    "unresolved_characters": res.unresolved_characters,
                    "uncertain_characters": res.uncertain_characters,
                },
            )

        pred = clf.predict_glyph(image, top_k=top_k)
        latency = time.perf_counter() - t0

        display_text = "" if pred.is_rejected else pred.predicted_character

        # Bounding box covers full glyph crop when evaluating a cropped glyph
        boxes = [
            BoundingBox(
                ymin=0.0,
                xmin=0.0,
                ymax=1.0,
                xmax=1.0,
                confidence=pred.confidence,
                text=pred.predicted_character,
            )
        ]

        return OCRResult(
            text=display_text,
            provider_name=self.provider_name,
            model_id=pred.model_version,
            confidence=pred.confidence,  # None unless Platt probability calibration performed!
            latency_seconds=latency,
            bounding_boxes=boxes,
            raw_response=pred.to_dict(),
            is_cloud=False,
            is_historical_script=True,
            provenance_notes=(
                f"HOG+SVM CPU inference (script={target_script}, "
                f"calibrated={pred.is_calibrated}, uncertain={pred.is_uncertain}, rejected={pred.is_rejected})"
            ),
            metadata={
                "script": target_script,
                "is_calibrated": pred.is_calibrated,
                "raw_decision_score": pred.raw_decision_score,
                "top_candidates": [c.to_dict() for c in pred.top_candidates],
                "is_uncertain": pred.is_uncertain,
                "is_rejected": pred.is_rejected,
                "rejection_reason": pred.rejection_reason,
                "feature_dimension": pred.feature_dimension,
            },
        )

