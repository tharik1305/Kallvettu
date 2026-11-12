"""
Kallvettu Intelligent Epigraphical Evidence Fusion Engine (K-IEFE).
Transforms independent OCR engine predictions into a unified evidence-based
recognition, cross-verification, and adjudication platform.

EXPORTS:
- EpigraphicalCrossVerifier, CrossVerificationReport
- ConsensusEntropyCalculator, ConsensusReport, DisagreementSeverity
- EpigraphicalFusionEngine, FusedTranscriptionResult
- VisualEvidenceVerifier, DisputedRegionInspection, VisualVerdict
- HistoricalContextValidator, HistoricalValidationSummary
- AdaptiveRoutingEngine, ImageQualityMetrics, RoutingExecutionReport
- ResearcherReviewManager, AdjudicationItem, AdjudicationDecision
- ActiveLearningManager, VerifiedAnnotationRecord
- FusionAuditLogger, EvidenceFusionAuditRecord
- EpigraphicalFusionEvaluator, CorpusBenchmarkSummary
- fuse_epigraphical_evidence (Master Entrypoint)
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import numpy as np

from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ai.common.source_identity import SourceIdentity, create_source_identity
from services.ocr_fusion.cross_verifier import (
    EpigraphicalCrossVerifier,
    CrossVerificationReport,
    CharacterAlignmentUnit,
    WordAlignmentUnit,
    LineAlignmentUnit,
    normalize_for_comparison,
)
from services.ocr_fusion.consensus import (
    ConsensusEntropyCalculator,
    ConsensusReport,
    PositionEntropyResult,
    DisagreementSeverity,
)
from services.ocr_fusion.fusion_engine import (
    EpigraphicalFusionEngine,
    FusedTranscriptionResult,
    FusedCharacterPosition,
    CandidateAlternative,
)
from services.ocr_fusion.visual_verifier import (
    VisualEvidenceVerifier,
    DisputedRegionInspection,
    VisualVerdict,
    VisualEvidenceHypothesis,
)
from services.ocr_fusion.historical_validator import (
    HistoricalContextValidator,
    HistoricalValidationSummary,
    HistoricalEvidenceMatch,
)
from services.ocr_fusion.adaptive_router import (
    AdaptiveRoutingEngine,
    CPURoutingConfig,
    ImageQualityMetrics,
    RoutingExecutionReport,
    apply_cpu_thread_limits,
)
from services.ocr_fusion.researcher_review import (
    ResearcherReviewManager,
    AdjudicationItem,
    AdjudicationDecision,
    ReviewRevisionEntry,
)
from services.ocr_fusion.active_learning import (
    ActiveLearningManager,
    VerifiedAnnotationRecord,
)
from services.ocr_fusion.audit_trail import (
    FusionAuditLogger,
    EvidenceFusionAuditRecord,
)
from services.ocr_fusion.evaluation import (
    EpigraphicalFusionEvaluator,
    EvaluationSampleResult,
    CorpusBenchmarkSummary,
)


@dataclass
class MasterEvidenceFusionBundle:
    """Master container bundling all K-IEFE cross-verification, consensus, and fusion outputs."""
    execution_id: str
    source_identity: SourceIdentity
    cross_verification_report: CrossVerificationReport
    consensus_report: ConsensusReport
    fused_result: FusedTranscriptionResult
    all_fusion_strategies: Dict[str, FusedTranscriptionResult]
    historical_validation: HistoricalValidationSummary
    visual_inspections: List[DisputedRegionInspection]
    audit_record: EvidenceFusionAuditRecord
    adjudication_items: List[AdjudicationItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "source_identity": self.source_identity.to_dict(),
            "fused_text": self.fused_result.fused_text,
            "conservative_text": self.fused_result.conservative_text,
            "mean_consensus_entropy": round(self.consensus_report.mean_entropy, 4),
            "agreement_ratio": round(self.cross_verification_report.agreement_ratio, 4),
            "strongest_baseline_provider": self.fused_result.strongest_baseline_provider,
            "cross_verification": self.cross_verification_report.to_dict(),
            "consensus": self.consensus_report.to_dict(),
            "fusion": self.fused_result.to_dict(),
            "historical_validation": self.historical_validation.to_dict(),
            "visual_inspections": [v.to_dict() for v in self.visual_inspections],
            "audit_record": self.audit_record.to_dict(),
            "adjudication_items": [it.to_dict() for it in self.adjudication_items],
        }


def fuse_epigraphical_evidence(
    image: Union[Image.Image, np.ndarray, str],
    results: Dict[str, OCRResult],
    script_hint: str = "auto",
    image_quality: str = "good",
    damage_category: str = "weathered",
    source_identity: Optional[SourceIdentity] = None,
    preprocessing_variant: str = "Path A: Original Photograph",
) -> MasterEvidenceFusionBundle:
    """
    Master pipeline executing complete K-IEFE evidence fusion lifecycle:
    1. Cross-Verification (Character, word, and line alignment)
    2. Consensus Entropy distribution calculation
    3. Multi-Strategy Evidence Fusion (ROVER, confidence-weighted, character candidate)
    4. Historical Context & Catalog Cross-referencing
    5. Disputed Region Visual Crop Extraction & Stroke Verification
    6. Adjudication Item Assembly
    7. Cryptographic Audit Dossier Generation
    """
    exec_id = f"KIEFE_{uuid.uuid4().hex[:10].upper()}"

    # Ingest image
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    active_source = source_identity or create_source_identity(pil_img)

    # 1. Cross-Verification
    verifier = EpigraphicalCrossVerifier()
    cross_rep = verifier.verify(results)

    # 2. Consensus Entropy Calculation
    # Compute base weights for consensus
    fusion_eng = EpigraphicalFusionEngine()
    base_weights = {
        pid: fusion_eng.compute_provider_reliability(pid, script_hint, image_quality, damage_category)
        for pid in cross_rep.provider_names
    }
    consensus_rep = ConsensusEntropyCalculator.evaluate_report(
        cross_rep, provider_weights=base_weights
    )

    # 3. Multi-Strategy Evidence Fusion
    strategies = fusion_eng.fuse_all_and_select_best(
        results,
        cross_rep,
        script_hint=script_hint,
        image_quality=image_quality,
        damage_category=damage_category
    )
    rec_fused = strategies["recommended"]

    # 4. Historical Context Validation
    hist_validator = HistoricalContextValidator()
    hist_summary = hist_validator.validate_candidate_text(rec_fused.fused_text, script_hint=script_hint)

    # 5. Visual Evidence Verification for Disputed Positions
    vis_verifier = VisualEvidenceVerifier()
    visual_inspections: List[DisputedRegionInspection] = []
    adjudication_items: List[AdjudicationItem] = []

    # Find regions with high dispute or entropy > 0.5
    disputed_units = [u for u in cross_rep.character_alignments if u.is_disputed]
    for idx, d_unit in enumerate(disputed_units[:5]):  # Focus on top 5 disputed regions
        bbox = d_unit.bounding_box or BoundingBox(
            ymin=max(0.0, 0.1 * idx),
            xmin=0.05,
            ymax=min(1.0, 0.1 * idx + 0.15),
            xmax=0.95,
            confidence=None,
            text=d_unit.candidate_character
        )
        cands = list({c for c in d_unit.normalized_by_provider.values() if c and c != "-"})
        insp = vis_verifier.verify_disputed_region(
            pil_img,
            bbox=bbox,
            candidates=cands,
            region_id=f"region_disp_{idx+1}",
            script_hint=script_hint
        )
        visual_inspections.append(insp)

        # Assemble Adjudication Item
        pos_res = ConsensusEntropyCalculator.calculate_position_entropy(d_unit, provider_weights=base_weights)
        adj_item = AdjudicationItem(
            item_id=f"adj_{idx+1}",
            region_index=d_unit.position_index,
            bounding_box=bbox,
            pixel_crop_coords=insp.pixel_coordinates,
            provider_candidates=d_unit.raw_by_provider,
            provider_confidences=d_unit.confidence_by_provider,
            fused_candidate=d_unit.candidate_character,
            alternative_readings=cands,
            disagreement_reason=f"Status: {d_unit.status}. Distinct candidate readings: {', '.join(cands)}.",
            consensus_entropy=pos_res.entropy,
            visual_hypothesis=insp.hypotheses_by_candidate.get(d_unit.candidate_character, {}).to_dict() if d_unit.candidate_character in insp.hypotheses_by_candidate else None,
            historical_evidence=[m.to_dict() for m in hist_summary.supporting_matches[:2]],
            current_decision=AdjudicationDecision.PENDING.value,
            approved_text=d_unit.candidate_character
        )
        adjudication_items.append(adj_item)

    # 6. Cryptographic Audit Trail
    model_versions = {
        pid: res.model_id for pid, res in results.items() if isinstance(res, OCRResult)
    }
    raw_confs = {
        pid: res.confidence for pid, res in results.items() if isinstance(res, OCRResult)
    }
    audit_rec = FusionAuditLogger.build_audit_record(
        execution_id=exec_id,
        source_identity=active_source,
        preprocessing_variant=preprocessing_variant,
        provider_outputs=cross_rep.raw_texts_by_provider,
        model_versions=model_versions,
        confidences=raw_confs,
        reliabilities=base_weights,
        consensus_entropy=consensus_rep.mean_entropy,
        disagreement_severity="high_dispute" if consensus_rep.high_dispute_positions else "consensus",
        fusion_strategy=rec_fused.strategy_used,
        fused_reading=rec_fused.fused_text,
        conservative_reading=rec_fused.conservative_text,
        visual_hypotheses=[insp.to_dict() for insp in visual_inspections],
        historical_matches=[m.to_dict() for m in hist_summary.supporting_matches],
        competing_sequences=[{"seq": s[0], "score": s[1]} for s in rec_fused.competing_sequences]
    )

    return MasterEvidenceFusionBundle(
        execution_id=exec_id,
        source_identity=active_source,
        cross_verification_report=cross_rep,
        consensus_report=consensus_rep,
        fused_result=rec_fused,
        all_fusion_strategies=strategies,
        historical_validation=hist_summary,
        visual_inspections=visual_inspections,
        audit_record=audit_rec,
        adjudication_items=adjudication_items
    )

