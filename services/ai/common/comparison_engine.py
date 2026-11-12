"""
Multi-Engine Epigraphical Verification & OCR Comparison Engine for Kallvettu.
Executes parallel/multi-provider OCR across:
1. PaddleOCR (Local Offline PP-OCRv5 Tamil)
2. Sarvam Vision (Document AI 3B)
3. Google Gemini Vision (Multimodal AI)

Analyzes character and word-level disagreements via Sequence Alignment (NFC Unicode),
flags disputed segments with dispute types, isolates source image provenance,
and provides researcher-controlled transcription approval without blind majority voting.
"""

import logging
import time
import difflib
import unicodedata
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("kallvettu.ai.comparator")

from services.ai.common.base_provider import (
    BaseOCRProvider,
    OCRResult,
    BoundingBox,
)
from services.ai.common.provider_manager import ProviderManager, get_provider_manager
from services.ai.common.response_validator import clean_text_artifacts, validate_tamil_unicode
from services.ai.common.source_identity import SourceIdentity, create_source_identity, compute_image_sha256
from services.ai.common.error_handler import InvalidInputError


def normalize_tamil_reading(text: Optional[str]) -> str:
    """
    Normalizes inscriptional Tamil Unicode into NFC canonical composition,
    strips non-printable artifacts, zero-width spaces, and normalizes whitespaces.
    """
    if not text:
        return ""
    # NFC Canonical Composition
    normalized = unicodedata.normalize("NFC", str(text))
    # Remove zero-width spaces, joiners, BOM
    normalized = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", normalized)
    # Collapse multiple whitespaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass
class DisputedToken:
    """Represents a word/token where providers proposed competing readings."""
    token_index: int
    readings: Dict[str, str]  # provider_id -> candidate token
    is_disputed: bool
    consensus_candidate: Optional[str] = None
    researcher_correction: Optional[str] = None
    dispute_type: str = "identical"  # 'identical', 'substitution', 'omission', 'word_boundary'
    diff_details: Optional[str] = None
    confidence_by_provider: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def tokens_by_provider(self) -> Dict[str, str]:
        return self.readings

    @property
    def consensus_token(self) -> Optional[str]:
        return self.consensus_candidate

    @property
    def agreement_ratio(self) -> float:
        if not self.readings:
            return 1.0
        from collections import Counter
        counts = Counter(self.readings.values())
        most_common_cnt = counts.most_common(1)[0][1]
        return most_common_cnt / len(self.readings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_index": self.token_index,
            "readings": self.readings,
            "is_disputed": self.is_disputed,
            "consensus_candidate": self.consensus_candidate,
            "researcher_correction": self.researcher_correction,
            "dispute_type": self.dispute_type,
            "diff_details": self.diff_details,
            "confidence_by_provider": self.confidence_by_provider,
        }


@dataclass
class MultiOCRComparisonResult:
    """Unified structured report across all OCR engines."""
    provider_results: Dict[str, OCRResult]  # provider_id -> OCRResult
    disputed_tokens: List[DisputedToken] = field(default_factory=list)
    consensus_ratio: float = 0.0
    character_similarity_ratio: float = 0.0
    total_tokens_evaluated: int = 0
    agreed_tokens_count: int = 0
    disputed_tokens_count: int = 0
    approved_transcription: str = ""
    is_researcher_approved: bool = False
    bounding_boxes_by_provider: Dict[str, List[BoundingBox]] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    comparison_status: str = "success"
    insufficient_reason: Optional[str] = None
    # Enhanced scientific verification attributes
    source_identity: Optional[SourceIdentity] = None
    registered_engines: List[str] = field(default_factory=list)
    configured_engines: List[str] = field(default_factory=list)
    executed_engines: List[str] = field(default_factory=list)
    failed_engines: List[str] = field(default_factory=list)
    agreement_metric_label: str = "Inter-Engine Agreement Rate (Not Absolute Accuracy)"
    agreement_metric_formula: str = "Character & Token Sequence Levenshtein Alignment (NFC Normalized)"

    @property
    def results_by_provider(self) -> Dict[str, OCRResult]:
        return self.provider_results

    @property
    def consensus_text(self) -> str:
        return self.approved_transcription

    @property
    def recommended_text(self) -> str:
        return self.approved_transcription

    @property
    def fastest_provider(self) -> Optional[str]:
        if not self.provider_results:
            return None
        return min(self.provider_results.items(), key=lambda kv: kv[1].latency_seconds)[0]

    @property
    def agreement_percentage(self) -> float:
        return round(self.consensus_ratio * 100.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_results": {k: v.to_dict() for k, v in self.provider_results.items()},
            "disputed_tokens": [t.to_dict() for t in self.disputed_tokens],
            "consensus_ratio": round(self.consensus_ratio, 4),
            "character_similarity_ratio": round(self.character_similarity_ratio, 4),
            "total_tokens_evaluated": self.total_tokens_evaluated,
            "agreed_tokens_count": self.agreed_tokens_count,
            "disputed_tokens_count": self.disputed_tokens_count,
            "approved_transcription": self.approved_transcription,
            "is_researcher_approved": self.is_researcher_approved,
            "bounding_boxes_by_provider": {
                k: [b.to_dict() for b in v] for k, v in self.bounding_boxes_by_provider.items()
            },
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "comparison_status": self.comparison_status,
            "insufficient_reason": self.insufficient_reason,
            "source_identity": self.source_identity.to_dict() if self.source_identity else None,
            "registered_engines": self.registered_engines,
            "configured_engines": self.configured_engines,
            "executed_engines": self.executed_engines,
            "failed_engines": self.failed_engines,
            "agreement_metric_label": self.agreement_metric_label,
            "agreement_metric_formula": self.agreement_metric_formula,
        }


ComparisonResult = MultiOCRComparisonResult


class MultiEngineOCRComparator:
    """
    Executes and cross-verifies lithic OCR across multiple AI providers
    using sequence alignment, cryptographic source identity tracking,
    and character-level dispute resolution.
    """

    TARGET_PROVIDER_IDS = [
        "paddleocr_local",
        "sarvam_vision",
        "gemini_vision",
        "custom_historical_ocr",
        "historical_hog_svm",
    ]

    def __init__(self, provider_manager: Optional[ProviderManager] = None):
        self._manager = provider_manager or get_provider_manager()

    def compare_ocr(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        provider_ids: Optional[List[str]] = None,
        source_identity: Optional[SourceIdentity] = None,
        **kwargs: Any
    ) -> MultiOCRComparisonResult:
        """
        Executes OCR across selected engines, aligns token sequences,
        and identifies disputed characters while tracking engine execution manifest.
        """
        t0 = time.perf_counter()
        targets = provider_ids or self.TARGET_PROVIDER_IDS

        # Establish Source Identity
        active_source = source_identity or create_source_identity(image)

        active_results: Dict[str, OCRResult] = {}
        boxes_by_provider: Dict[str, List[BoundingBox]] = {}

        # Manifest tracking
        registered_provs = getattr(self._manager, "get_ocr_providers", lambda: [])()
        if not registered_provs and hasattr(self._manager, "get_providers_by_type"):
            from services.ai.common.base_provider import ProviderType
            registered_provs = self._manager.get_providers_by_type(ProviderType.OCR)
        registered = [p.provider_id for p in registered_provs] if registered_provs else list(targets)
        configured: List[str] = []
        executed: List[str] = []
        failed: List[str] = []

        configured_ocr = getattr(self._manager, "get_configured_ocr_providers", lambda: [])()
        orig_img = kwargs.get("original_image")
        if configured_ocr and not provider_ids:
            for prov in configured_ocr:
                pid = getattr(prov, "provider_id", getattr(prov, "provider_name", "ocr_provider"))
                pname = getattr(prov, "provider_name", pid)
                configured.append(pid)
                img_for_prov = orig_img if ("sarvam" in str(pid).lower() and orig_img is not None) else image
                call_kwargs = dict(kwargs)
                if pid == "historical_hog_svm" and "mode" not in call_kwargs:
                    call_kwargs["mode"] = "line"
                try:
                    res = prov.recognize(img_for_prov, script_hint=script_hint, **call_kwargs)
                    if isinstance(res, OCRResult) and res.text.strip():
                        active_results[pname] = res
                        executed.append(pid)
                        if res.bounding_boxes:
                            boxes_by_provider[pname] = res.bounding_boxes
                    else:
                        failed.append(pid)
                except Exception as e:
                    failed.append(pid)
                    logger.warning("Comparison: Provider execution skipped: %s", e)
        else:
            for pid in targets:
                prov = self._manager.get_ocr_provider(pid)
                if not prov or not self._manager.is_enabled(pid):
                    continue

                img_for_prov = orig_img if ("sarvam" in str(pid).lower() and orig_img is not None) else image
                call_kwargs = dict(kwargs)
                if pid == "historical_hog_svm" and "mode" not in call_kwargs:
                    call_kwargs["mode"] = "line"
                try:
                    res = prov.recognize(img_for_prov, script_hint=script_hint, **call_kwargs)
                    if isinstance(res, OCRResult) and res.text.strip():
                        active_results[pid] = res
                        executed.append(pid)
                        if res.bounding_boxes:
                            boxes_by_provider[pid] = res.bounding_boxes
                    else:
                        failed.append(pid)
                except Exception as e:
                    failed.append(pid)
                    logger.warning("Comparison: Provider '%s' execution skipped: %s", pid, e)

        # Fallback to local catalog or placeholder if all cloud/local engines failed or unconfigured
        if not active_results:
            cat_prov = self._manager.get_ocr_provider("local_catalog_ocr")
            if cat_prov:
                try:
                    res_cat = cat_prov.recognize(image, **kwargs)
                    active_results["local_catalog_ocr"] = res_cat
                    executed.append("local_catalog_ocr")
                except Exception:
                    failed.append("local_catalog_ocr")

        # Calculate character similarity
        char_sim_ratio = self._calculate_character_similarity(active_results)

        # Perform Sequence Alignment & Disagreement Analysis
        (
            disputed,
            consensus_ratio,
            total_tok,
            agreed_tok,
            disputed_tok,
            default_approved,
            comp_status,
            insufficient_msg,
        ) = self._analyze_disagreements(active_results)

        total_elapsed = time.perf_counter() - t0

        return MultiOCRComparisonResult(
            provider_results=active_results,
            disputed_tokens=disputed,
            consensus_ratio=consensus_ratio,
            character_similarity_ratio=char_sim_ratio,
            total_tokens_evaluated=total_tok,
            agreed_tokens_count=agreed_tok,
            disputed_tokens_count=disputed_tok,
            approved_transcription=default_approved,
            is_researcher_approved=False,
            bounding_boxes_by_provider=boxes_by_provider,
            execution_time_seconds=total_elapsed,
            comparison_status=comp_status,
            insufficient_reason=insufficient_msg,
            source_identity=active_source,
            registered_engines=registered,
            configured_engines=configured,
            executed_engines=executed,
            failed_engines=failed,
        )

    compare_all = compare_ocr

    def _calculate_character_similarity(self, results: Dict[str, OCRResult]) -> float:
        """Calculates pairwise character Levenshtein ratio across valid candidate readings."""
        valid_candidates = {
            pid: normalize_tamil_reading(res.text)
            for pid, res in results.items()
            if res and res.text and res.text.strip() and res.metadata.get("transcription_status") != "description_only"
        }
        if len(valid_candidates) < 2:
            return 1.0 if len(valid_candidates) == 1 else 0.0

        ratios: List[float] = []
        pids = list(valid_candidates.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                s_matcher = difflib.SequenceMatcher(None, valid_candidates[pids[i]], valid_candidates[pids[j]])
                ratios.append(s_matcher.ratio())
        return float(np.mean(ratios)) if ratios else 0.0

    def _analyze_disagreements(
        self,
        results: Dict[str, OCRResult]
    ) -> Tuple[List[DisputedToken], float, int, int, int, str, str, Optional[str]]:
        """
        Cross-checks reading unanimity using Sequence Alignment on NFC-normalized Tamil Unicode.
        Requires at least two non-empty genuine inscription transcriptions to compare.
        Never treats visual descriptions as inscription text.
        """
        if not results:
            return [], 0.0, 0, 0, 0, "", "no_results", "Insufficient OCR output for palaeographical comparison."

        # Filter out empty outputs or visual descriptions
        valid_candidates: Dict[str, OCRResult] = {}
        for pid, res in results.items():
            if not res or not res.text or not res.text.strip():
                continue
            if res.metadata.get("transcription_status") == "description_only":
                continue
            valid_candidates[pid] = res

        # Require at least two non-empty OCR outputs for comparison
        if len(valid_candidates) < 2:
            single_text = ""
            total_tok = 0
            if len(valid_candidates) == 1:
                single_res = list(valid_candidates.values())[0]
                single_text = normalize_tamil_reading(single_res.text)
                total_tok = len(single_text.split())

            return (
                [],
                1.0 if len(valid_candidates) == 1 else 0.0,
                total_tok,
                total_tok,
                0,
                single_text,
                "insufficient_candidates",
                "Insufficient OCR output for palaeographical comparison."
            )

        # Normalize candidate texts with NFC
        norm_candidates: Dict[str, str] = {
            pid: normalize_tamil_reading(res.text)
            for pid, res in valid_candidates.items()
        }

        # Extract token lists
        provider_tokens: Dict[str, List[str]] = {
            pid: [t.strip() for t in text.split() if t.strip()]
            for pid, text in norm_candidates.items()
        }

        token_lengths = [len(t) for t in provider_tokens.values()]
        max_len = max(token_lengths, default=0)
        min_len = min(token_lengths, default=0)

        if max_len == 0:
            return [], 0.0, 0, 0, 0, "", "empty_tokens", "Insufficient OCR output for palaeographical comparison."

        # Case A: Exactly equal token lengths across all engines -> direct token-by-token comparison
        if min_len == max_len and len(provider_tokens) >= 2:
            return self._analyze_positional_tokens(valid_candidates, provider_tokens, max_len)

        # Case B: Differing token counts -> Sequence Alignment (handles token splits, insertions, deletions)
        return self._analyze_sequence_aligned_tokens(valid_candidates, provider_tokens)

    def _analyze_positional_tokens(
        self,
        valid_candidates: Dict[str, OCRResult],
        provider_tokens: Dict[str, List[str]],
        token_count: int
    ) -> Tuple[List[DisputedToken], float, int, int, int, str, str, Optional[str]]:
        """Analyzes disagreements when token sequences map one-to-one across engines."""
        disputed_list: List[DisputedToken] = []
        agreed_count = 0
        disputed_count = 0
        composite_words: List[str] = []

        for idx in range(token_count):
            readings_at_idx: Dict[str, str] = {}
            confs_at_idx: Dict[str, Optional[float]] = {}

            for pid, t_list in provider_tokens.items():
                w = t_list[idx]
                readings_at_idx[pid] = w
                confs_at_idx[pid] = valid_candidates[pid].confidence

            unique_words = set(readings_at_idx.values())
            is_disputed = len(unique_words) > 1

            consensus_candidate = max(set(readings_at_idx.values()), key=list(readings_at_idx.values()).count)

            diff_summary = None
            dispute_type = "identical"
            if is_disputed:
                disputed_count += 1
                dispute_type = "substitution"
                diff_summary = " | ".join([f"{k}: '{v}'" for k, v in readings_at_idx.items()])
            else:
                agreed_count += 1

            disputed_list.append(
                DisputedToken(
                    token_index=idx,
                    readings=readings_at_idx,
                    is_disputed=is_disputed,
                    consensus_candidate=consensus_candidate,
                    researcher_correction=None,
                    dispute_type=dispute_type,
                    diff_details=diff_summary,
                    confidence_by_provider=confs_at_idx,
                )
            )

            if consensus_candidate:
                composite_words.append(consensus_candidate)

        total_evaluated = agreed_count + disputed_count
        ratio = (agreed_count / total_evaluated) if total_evaluated > 0 else 0.0
        default_approved = " ".join(composite_words).strip()

        return (
            disputed_list,
            ratio,
            total_evaluated,
            agreed_count,
            disputed_count,
            default_approved,
            "success",
            None
        )

    def _analyze_sequence_aligned_tokens(
        self,
        valid_candidates: Dict[str, OCRResult],
        provider_tokens: Dict[str, List[str]]
    ) -> Tuple[List[DisputedToken], float, int, int, int, str, str, Optional[str]]:
        """
        Uses SequenceMatcher alignment across uneven token streams to prevent cascading false disputes.
        """
        pids = list(provider_tokens.keys())
        ref_pid = pids[0]
        other_pid = pids[1]

        tokens_ref = provider_tokens[ref_pid]
        tokens_other = provider_tokens[other_pid]

        matcher = difflib.SequenceMatcher(None, tokens_ref, tokens_other)
        opcodes = matcher.get_opcodes()

        disputed_list: List[DisputedToken] = []
        agreed_count = 0
        disputed_count = 0
        composite_words: List[str] = []
        token_idx = 0

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for k in range(i2 - i1):
                    w_ref = tokens_ref[i1 + k]
                    w_oth = tokens_other[j1 + k]
                    readings = {ref_pid: w_ref, other_pid: w_oth}
                    confs = {
                        ref_pid: valid_candidates[ref_pid].confidence,
                        other_pid: valid_candidates[other_pid].confidence,
                    }
                    disputed_list.append(
                        DisputedToken(
                            token_index=token_idx,
                            readings=readings,
                            is_disputed=False,
                            consensus_candidate=w_ref,
                            researcher_correction=None,
                            dispute_type="identical",
                            diff_details=None,
                            confidence_by_provider=confs,
                        )
                    )
                    composite_words.append(w_ref)
                    agreed_count += 1
                    token_idx += 1

            elif tag == "replace":
                span_ref = tokens_ref[i1:i2]
                span_oth = tokens_other[j1:j2]
                if len(span_ref) == len(span_oth):
                    for k in range(len(span_ref)):
                        w_ref = span_ref[k]
                        w_oth = span_oth[k]
                        readings = {ref_pid: w_ref, other_pid: w_oth}
                        confs = {
                            ref_pid: valid_candidates[ref_pid].confidence,
                            other_pid: valid_candidates[other_pid].confidence,
                        }
                        diff_sum = f"{ref_pid}: '{w_ref}' vs {other_pid}: '{w_oth}'"
                        disputed_list.append(
                            DisputedToken(
                                token_index=token_idx,
                                readings=readings,
                                is_disputed=True,
                                consensus_candidate=w_ref,
                                researcher_correction=None,
                                dispute_type="substitution",
                                diff_details=diff_sum,
                                confidence_by_provider=confs,
                            )
                        )
                        composite_words.append(w_ref)
                        disputed_count += 1
                        token_idx += 1
                else:
                    w_ref_joined = " ".join(span_ref)
                    w_oth_joined = " ".join(span_oth)
                    readings = {ref_pid: w_ref_joined, other_pid: w_oth_joined}
                    confs = {
                        ref_pid: valid_candidates[ref_pid].confidence,
                        other_pid: valid_candidates[other_pid].confidence,
                    }
                    diff_sum = f"Word boundary / segmentation: {ref_pid}: '{w_ref_joined}' vs {other_pid}: '{w_oth_joined}'"
                    disputed_list.append(
                        DisputedToken(
                            token_index=token_idx,
                            readings=readings,
                            is_disputed=True,
                            consensus_candidate=w_ref_joined,
                            researcher_correction=None,
                            dispute_type="word_boundary",
                            diff_details=diff_sum,
                            confidence_by_provider=confs,
                        )
                    )
                    composite_words.append(w_ref_joined)
                    disputed_count += 1
                    token_idx += 1

            elif tag == "delete":
                for k in range(i2 - i1):
                    w_ref = tokens_ref[i1 + k]
                    readings = {ref_pid: w_ref, other_pid: ""}
                    confs = {ref_pid: valid_candidates[ref_pid].confidence, other_pid: None}
                    diff_sum = f"Omitted by {other_pid}: '{w_ref}'"
                    disputed_list.append(
                        DisputedToken(
                            token_index=token_idx,
                            readings=readings,
                            is_disputed=True,
                            consensus_candidate=w_ref,
                            researcher_correction=None,
                            dispute_type="omission",
                            diff_details=diff_sum,
                            confidence_by_provider=confs,
                        )
                    )
                    composite_words.append(w_ref)
                    disputed_count += 1
                    token_idx += 1

            elif tag == "insert":
                for k in range(j2 - j1):
                    w_oth = tokens_other[j1 + k]
                    readings = {ref_pid: "", other_pid: w_oth}
                    confs = {ref_pid: None, other_pid: valid_candidates[other_pid].confidence}
                    diff_sum = f"Inserted by {other_pid}: '{w_oth}'"
                    disputed_list.append(
                        DisputedToken(
                            token_index=token_idx,
                            readings=readings,
                            is_disputed=True,
                            consensus_candidate=w_oth,
                            researcher_correction=None,
                            dispute_type="omission",
                            diff_details=diff_sum,
                            confidence_by_provider=confs,
                        )
                    )
                    composite_words.append(w_oth)
                    disputed_count += 1
                    token_idx += 1

        total_evaluated = agreed_count + disputed_count
        ratio = (agreed_count / total_evaluated) if total_evaluated > 0 else 0.0
        default_approved = " ".join(composite_words).strip()

        return (
            disputed_list,
            ratio,
            total_evaluated,
            agreed_count,
            disputed_count,
            default_approved,
            "success",
            None
        )


# Global Singleton Comparator
_GLOBAL_COMPARATOR: Optional[MultiEngineOCRComparator] = None


def get_ocr_comparator() -> MultiEngineOCRComparator:
    """Returns the shared singleton instance of MultiEngineOCRComparator."""
    global _GLOBAL_COMPARATOR
    if _GLOBAL_COMPARATOR is None:
        _GLOBAL_COMPARATOR = MultiEngineOCRComparator()
    return _GLOBAL_COMPARATOR
