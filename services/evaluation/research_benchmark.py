"""
CPU-First Research Evaluation & Benchmark Suite for Kallvettu (Phase 6).

Measures scientific accuracy, uncertainty calibration, and CPU hardware efficiency:
- Sequence OCR metrics: Character Error Rate (CER), Word Error Rate (WER), Character Accuracy, Exact Match.
- Glyph classification metrics: Top-1 / Top-3 Accuracy, Macro Precision / Recall / F1, Confusion Matrix.
- CPU hardware telemetry: Inference latency (mean / p95 ms), peak process RSS memory (MB), model size budget (<= 25 MB).
- Side-by-side model comparison across CPU engines (HOG+SVM, Line Recognizer, PaddleOCR).
- Structured academic export formats: JSON, CSV, and publication-ready Markdown.
"""

from __future__ import annotations

import csv
import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from services.research.historical_glyph_dataset import normalize_tamil_unicode


def measure_process_memory_mb() -> float:
    """
    Returns current process resident set size (RSS) in Megabytes cleanly on Windows/Linux
    without requiring heavyweight external profilers.
    """
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass

    return 0.0


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Standard dynamic-programming edit distance for characters or word tokens."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        s1_i = seq1[i - 1]
        for j in range(1, m + 1):
            cost = 0 if s1_i == seq2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            )
        prev = curr
    return prev[m]


def compute_cer(predicted: str, ground_truth: str, normalize_nfc: bool = True) -> float:
    """Computes Character Error Rate (CER) = edits / max(1, len(reference))."""
    pred = normalize_tamil_unicode(predicted.strip()) if normalize_nfc else predicted.strip()
    ref = normalize_tamil_unicode(ground_truth.strip()) if normalize_nfc else ground_truth.strip()
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    edits = levenshtein_distance(list(pred), list(ref))
    return float(edits) / float(len(ref))


def compute_wer(predicted: str, ground_truth: str, normalize_nfc: bool = True) -> float:
    """Computes Word Error Rate (WER) across whitespace-delimited words."""
    pred = normalize_tamil_unicode(predicted.strip()) if normalize_nfc else predicted.strip()
    ref = normalize_tamil_unicode(ground_truth.strip()) if normalize_nfc else ground_truth.strip()
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    edits = levenshtein_distance(pred_words, ref_words)
    return float(edits) / float(len(ref_words))


@dataclass
class SampleEvaluationRecord:
    """Per-sample OCR evaluation output."""
    sample_id: str
    ground_truth: str
    predicted_text: str
    cer: float
    wer: float
    exact_match: bool
    unresolved_count: int = 0
    latency_ms: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "ground_truth": self.ground_truth,
            "predicted_text": self.predicted_text,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "exact_match": self.exact_match,
            "unresolved_count": self.unresolved_count,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
        }


@dataclass
class EngineBenchmarkSummary:
    """Aggregated OCR & CPU efficiency benchmark across test plates."""
    engine_id: str
    engine_name: str
    sample_count: int
    mean_cer: float
    mean_wer: float
    character_accuracy: float
    exact_match_rate: float
    total_unresolved_markers: int
    mean_latency_ms: float
    p95_latency_ms: float
    rss_memory_before_mb: float
    rss_memory_after_mb: float
    peak_tracemalloc_mb: float
    model_size_mb: Optional[float] = None
    within_25mb_budget: Optional[bool] = None
    samples: List[SampleEvaluationRecord] = field(default_factory=list)
    evaluated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "sample_count": self.sample_count,
            "mean_cer": round(self.mean_cer, 4),
            "mean_wer": round(self.mean_wer, 4),
            "character_accuracy": round(self.character_accuracy, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "total_unresolved_markers": self.total_unresolved_markers,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "rss_memory_before_mb": round(self.rss_memory_before_mb, 2),
            "rss_memory_after_mb": round(self.rss_memory_after_mb, 2),
            "peak_tracemalloc_mb": round(self.peak_tracemalloc_mb, 2),
            "model_size_mb": round(self.model_size_mb, 4) if self.model_size_mb is not None else None,
            "within_25mb_budget": self.within_25mb_budget,
            "evaluated_utc": self.evaluated_utc,
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class GlyphEvaluationReport:
    """Quantitative classification report for individual historical glyph crops."""
    script: str
    model_version: str
    sample_count: int
    top1_accuracy: float
    top3_accuracy: float
    rejection_rate: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    classes: List[str]
    confusion_matrix: List[List[int]]
    mean_latency_ms: float
    model_size_mb: Optional[float] = None
    within_25mb_budget: Optional[bool] = None
    evaluated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchBenchmarkSuite:
    """
    Runs repeatable, strictly CPU-bound benchmark comparisons across
    glyph classifiers, line recognizers, and full OCR providers.
    """

    def evaluate_ocr_callable(
        self,
        engine_id: str,
        engine_name: str,
        test_items: List[Tuple[str, Any, str]],  # [(sample_id, image, ground_truth), ...]
        predict_fn: Callable[[Any], str],
        model_size_mb: Optional[float] = None,
    ) -> EngineBenchmarkSummary:
        """
        Evaluates any OCR callable `(image) -> text` over `(sample_id, image, ground_truth)`
        tuples, capturing CER, WER, latency, unresolved markers, and CPU memory delta.
        """
        rss_before = measure_process_memory_mb()
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        records: List[SampleEvaluationRecord] = []
        latencies: List[float] = []
        cers: List[float] = []
        wers: List[float] = []
        exact_matches = 0
        total_unresolved = 0

        for sample_id, img, gt in test_items:
            gt_norm = normalize_tamil_unicode(gt.strip())
            t0 = time.perf_counter()
            err = ""
            try:
                pred = predict_fn(img) or ""
            except Exception as e:
                pred = ""
                err = str(e)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            pred_norm = normalize_tamil_unicode(pred.strip())
            cer = compute_cer(pred_norm, gt_norm, normalize_nfc=False)
            wer = compute_wer(pred_norm, gt_norm, normalize_nfc=False)
            exact = (pred_norm == gt_norm) and len(gt_norm) > 0
            unresolved_cnt = pred_norm.count("[?]")

            if exact:
                exact_matches += 1
            total_unresolved += unresolved_cnt
            cers.append(cer)
            wers.append(wer)

            records.append(
                SampleEvaluationRecord(
                    sample_id=sample_id,
                    ground_truth=gt_norm,
                    predicted_text=pred_norm,
                    cer=cer,
                    wer=wer,
                    exact_match=exact,
                    unresolved_count=unresolved_cnt,
                    latency_ms=lat_ms,
                    error_message=err,
                )
            )

        _, peak_bytes = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()
        rss_after = measure_process_memory_mb()

        n = max(1, len(records))
        mean_cer = float(np.mean(cers)) if cers else 1.0
        mean_wer = float(np.mean(wers)) if wers else 1.0
        char_acc = max(0.0, 1.0 - mean_cer)

        within_budget = (model_size_mb <= 25.0) if model_size_mb is not None else None

        return EngineBenchmarkSummary(
            engine_id=engine_id,
            engine_name=engine_name,
            sample_count=len(records),
            mean_cer=mean_cer,
            mean_wer=mean_wer,
            character_accuracy=char_acc,
            exact_match_rate=float(exact_matches) / float(n),
            total_unresolved_markers=total_unresolved,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            rss_memory_before_mb=rss_before,
            rss_memory_after_mb=rss_after,
            peak_tracemalloc_mb=float(peak_bytes) / (1024.0 * 1024.0),
            model_size_mb=model_size_mb,
            within_25mb_budget=within_budget,
            samples=records,
        )

    def evaluate_glyph_classifier(
        self,
        classifier: Any,
        test_samples: List[Tuple[Any, str]],  # [(crop_img, ground_truth_label), ...]
        model_size_mb: Optional[float] = None,
    ) -> GlyphEvaluationReport:
        """
        Evaluates a glyph classifier (`predict_glyph(img, top_k=3)`) on labeled crops,
        computing Top-1, Top-3, Precision, Recall, F1, Rejection Rate, and Confusion Matrix.
        """
        y_true: List[str] = []
        y_pred: List[str] = []
        top3_hits = 0
        rejected_count = 0
        latencies: List[float] = []

        for img, label in test_samples:
            gt = normalize_tamil_unicode(label.strip())
            t0 = time.perf_counter()
            res = classifier.predict_glyph(img, top_k=3)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            y_true.append(gt)
            if res.is_rejected:
                rejected_count += 1
                y_pred.append("[?]")
            else:
                pred_c = normalize_tamil_unicode(res.predicted_character)
                y_pred.append(pred_c)

            top3_chars = [normalize_tamil_unicode(c.character) for c in res.top_candidates[:3]]
            if gt in top3_chars:
                top3_hits += 1

        n = max(1, len(test_samples))
        top1_acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p)) / float(n)
        top3_acc = float(top3_hits) / float(n)
        rej_rate = float(rejected_count) / float(n)

        unique_classes = sorted(set(y_true) | set(c for c in y_pred if c != "[?]"))
        if not unique_classes:
            unique_classes = []
            cm = []
            prec = rec = f1 = 0.0
        else:
            all_labels = list(unique_classes) + (["[?]"] if "[?]" in y_pred else [])
            cm_arr = confusion_matrix(y_true, y_pred, labels=all_labels)
            cm = cm_arr.tolist()
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=unique_classes, average="macro", zero_division=0
            )
            prec, rec, f1 = float(p), float(r), float(f)
            unique_classes = all_labels

        within_budget = (model_size_mb <= 25.0) if model_size_mb is not None else None

        return GlyphEvaluationReport(
            script=getattr(classifier, "script", "unknown"),
            model_version=getattr(classifier, "model_version", "unknown"),
            sample_count=len(test_samples),
            top1_accuracy=top1_acc,
            top3_accuracy=top3_acc,
            rejection_rate=rej_rate,
            macro_precision=prec,
            macro_recall=rec,
            macro_f1=f1,
            classes=unique_classes,
            confusion_matrix=cm,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            model_size_mb=model_size_mb,
            within_25mb_budget=within_budget,
        )

    @staticmethod
    def export_summaries_json(summaries: List[EngineBenchmarkSummary], filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in summaries], f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_summaries_csv(summaries: List[EngineBenchmarkSummary], filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        rows = []
        for s in summaries:
            rows.append({
                "engine_id": s.engine_id,
                "engine_name": s.engine_name,
                "sample_count": s.sample_count,
                "mean_cer": round(s.mean_cer, 4),
                "mean_wer": round(s.mean_wer, 4),
                "character_accuracy": round(s.character_accuracy, 4),
                "exact_match_rate": round(s.exact_match_rate, 4),
                "total_unresolved_markers": s.total_unresolved_markers,
                "mean_latency_ms": round(s.mean_latency_ms, 2),
                "p95_latency_ms": round(s.p95_latency_ms, 2),
                "rss_memory_after_mb": round(s.rss_memory_after_mb, 2),
                "peak_tracemalloc_mb": round(s.peak_tracemalloc_mb, 2),
                "model_size_mb": round(s.model_size_mb, 4) if s.model_size_mb is not None else "",
            })
        if not rows:
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def build_markdown_report(
        summaries: List[EngineBenchmarkSummary],
        glyph_reports: Optional[List[GlyphEvaluationReport]] = None,
    ) -> str:
        lines = [
            "# Kallvettu CPU-First Epigraphical Benchmark Report",
            "",
            f"Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## 1. End-to-End OCR & Sequence Comparison (CPU-Only)",
            "",
            "| Engine | Samples | Mean CER | Mean WER | Char Accuracy | Mean Latency (ms) | Peak Traced RAM (MB) | Model Size (MB) |",
            "|--------|---------|----------|----------|---------------|-------------------|----------------------|-----------------|",
        ]
        for s in summaries:
            msize = f"{s.model_size_mb:.3f}" if s.model_size_mb is not None else "N/A"
            lines.append(
                f"| {s.engine_name} (`{s.engine_id}`) | {s.sample_count} | "
                f"{s.mean_cer:.4f} | {s.mean_wer:.4f} | {s.character_accuracy * 100:.1f}% | "
                f"{s.mean_latency_ms:.1f} | {s.peak_tracemalloc_mb:.2f} | {msize} |"
            )

        if glyph_reports:
            lines.extend([
                "",
                "## 2. Historical Glyph Classifier Evaluation",
                "",
                "| Script | Model Version | Samples | Top-1 Acc | Top-3 Acc | Macro F1 | Rejection Rate | Latency (ms) |",
                "|--------|---------------|---------|-----------|-----------|----------|----------------|--------------|",
            ])
            for g in glyph_reports:
                lines.append(
                    f"| {g.script} | `{g.model_version}` | {g.sample_count} | "
                    f"{g.top1_accuracy * 100:.1f}% | {g.top3_accuracy * 100:.1f}% | "
                    f"{g.macro_f1:.4f} | {g.rejection_rate * 100:.1f}% | {g.mean_latency_ms:.2f} |"
                )

        return "\n".join(lines) + "\n"

