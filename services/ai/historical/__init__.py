"""
Kallvettu Lightweight Historical Script Recognition Package (CPU-First).
"""

from services.ai.historical.hog_features import (
    HOGConfig,
    DEFAULT_HOG_CONFIG,
    prepare_glyph_image,
    extract_hog_features,
)
from services.ai.historical.glyph_classifier import (
    GlyphCandidate,
    GlyphPredictionResult,
    HOGSVMGlyphClassifier,
)
from services.ai.historical.model_registry import (
    HistoricalModelRegistry,
    RegisteredModelInfo,
    DEFAULT_MODELS_DIR,
)
from services.ai.historical.inference import HistoricalHOGSVMProvider
from services.ai.historical.line_recognizer import (
    LineRecognizerConfig,
    CharacterProposal,
    LineSegment,
    LineRecognitionResult,
    HistoricalLineRecognizer,
)

__all__ = [
    "HOGConfig",
    "DEFAULT_HOG_CONFIG",
    "prepare_glyph_image",
    "extract_hog_features",
    "GlyphCandidate",
    "GlyphPredictionResult",
    "HOGSVMGlyphClassifier",
    "HistoricalModelRegistry",
    "RegisteredModelInfo",
    "DEFAULT_MODELS_DIR",
    "HistoricalHOGSVMProvider",
    "LineRecognizerConfig",
    "CharacterProposal",
    "LineSegment",
    "LineRecognitionResult",
    "HistoricalLineRecognizer",
]

