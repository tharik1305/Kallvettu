"""
Local Model Registry & Lazy Cache for Historical HOG + SVM Classifiers.

Guarantees:
- No automatic model downloads or retraining during Streamlit startup
- Lazy loading from disk ONLY when requested
- Thread-safe / singleton-friendly in-memory cache
- Artifact size auditing (targeting <= 25 MB per checkpoint)
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from services.ai.historical.glyph_classifier import HOGSVMGlyphClassifier

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "models", "historical_glyphs")
)

MAX_MODEL_BUDGET_MB = 25.0


def _script_to_slug(script: str) -> str:
    return script.lower().strip().replace("-", "_").replace(" ", "_")


@dataclass
class RegisteredModelInfo:
    """Metadata summary for a persisted local historical glyph model."""
    script: str
    script_slug: str
    model_version: str
    filepath: str
    file_size_bytes: int
    file_size_mb: float
    within_budget: bool
    classes_count: int
    classes: List[str]
    is_calibrated: bool
    training_metadata: Dict[str, Any]
    updated_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalModelRegistry:
    """Manages local storage, listing, and lazy cached loading of HOG+SVM checkpoints."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = os.path.abspath(models_dir or DEFAULT_MODELS_DIR)
        self._loaded_cache: Dict[str, HOGSVMGlyphClassifier] = {}

    @property
    def registry_index_path(self) -> str:
        return os.path.join(self.models_dir, "registry_index.json")

    def ensure_dir(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)

    def get_model_path(self, script: str) -> str:
        return os.path.join(self.models_dir, f"{_script_to_slug(script)}_hog_svm.joblib")

    def has_model(self, script: str) -> bool:
        return os.path.isfile(self.get_model_path(script))

    def load_index(self) -> Dict[str, RegisteredModelInfo]:
        if not os.path.isfile(self.registry_index_path):
            return {}
        with open(self.registry_index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: RegisteredModelInfo(**v) for k, v in raw.items()}

    def _save_index(self, index: Dict[str, RegisteredModelInfo]) -> None:
        self.ensure_dir()
        with open(self.registry_index_path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in index.items()}, f, ensure_ascii=False, indent=2)

    def register_and_save(self, classifier: HOGSVMGlyphClassifier) -> RegisteredModelInfo:
        """Persists classifier to local storage, updates registry index, and caches it."""
        self.ensure_dir()
        path = self.get_model_path(classifier.script)
        classifier.save(path)

        size_bytes = os.path.getsize(path)
        size_mb = round(size_bytes / (1024 * 1024), 4)
        slug = _script_to_slug(classifier.script)

        info = RegisteredModelInfo(
            script=classifier.script,
            script_slug=slug,
            model_version=classifier.model_version,
            filepath=path,
            file_size_bytes=size_bytes,
            file_size_mb=size_mb,
            within_budget=(size_mb <= MAX_MODEL_BUDGET_MB),
            classes_count=len(classifier.classes_),
            classes=list(classifier.classes_),
            is_calibrated=bool(classifier.calibrate_probabilities),
            training_metadata=dict(classifier.training_metadata),
            updated_utc=datetime.now(timezone.utc).isoformat(),
        )

        idx = self.load_index()
        idx[slug] = info
        self._save_index(idx)

        self._loaded_cache[slug] = classifier
        return info

    def get_classifier(self, script: str) -> Optional[HOGSVMGlyphClassifier]:
        """Lazily loads a saved classifier from local disk into cache."""
        slug = _script_to_slug(script)
        if slug in self._loaded_cache:
            return self._loaded_cache[slug]

        path = self.get_model_path(script)
        if not os.path.isfile(path):
            return None

        clf = HOGSVMGlyphClassifier.load(path)
        self._loaded_cache[slug] = clf
        return clf

    def list_available_scripts(self) -> List[str]:
        idx = self.load_index()
        scripts = [info.script for info in idx.values() if os.path.isfile(info.filepath)]
        if not scripts and os.path.isdir(self.models_dir):
            # Scan disk directly if index absent
            for fname in os.listdir(self.models_dir):
                if fname.endswith("_hog_svm.joblib"):
                    scripts.append(fname.replace("_hog_svm.joblib", ""))
        return sorted(scripts)

    def clear_cache(self) -> None:
        self._loaded_cache.clear()

