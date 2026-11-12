"""
Kallvettu Lightweight Research Evaluation & CPU Benchmark Suite (Phase 6).
"""

from services.evaluation.research_benchmark import (
    SampleEvaluationRecord,
    EngineBenchmarkSummary,
    GlyphEvaluationReport,
    ResearchBenchmarkSuite,
    compute_cer,
    compute_wer,
    levenshtein_distance,
    measure_process_memory_mb,
)

__all__ = [
    "SampleEvaluationRecord",
    "EngineBenchmarkSummary",
    "GlyphEvaluationReport",
    "ResearchBenchmarkSuite",
    "compute_cer",
    "compute_wer",
    "levenshtein_distance",
    "measure_process_memory_mb",
]

