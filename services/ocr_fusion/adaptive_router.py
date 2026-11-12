"""
Adaptive Epigraphical Routing Engine for Kallvettu (K-IEFE, Phase 4 CPU-First).
Orchestrates tiered, strictly SEQUENTIAL recognition execution on CPU based on
image quality assessment, local recognition confidence, script classification,
and prediction caching to eliminate redundant inference and API costs.

ROUTING ARCHITECTURE:
- CPU Thread Governance: Enforces 1-4 thread limits across OpenCV / OpenMP / Paddle.
- Prediction Cache: SHA-256 image digest + script + routing config avoids re-running inference.
- Tier 1a: Archaic Specialist CPU Engine (`historical_hog_svm` line recognizer) for Brahmi/Vatteluttu/Grantha.
- Tier 1b: Primary Local Tamil OCR (`paddleocr_local` PP-OCRv5 CPU).
- Early Exit: Returns immediately when local OCR produces high-confidence non-archaic output.
- Tier 2: Selective Cloud Escalation (`sarvam_vision` / `gemini_vision`), sequentially, opt-out capable.
- Tier 3: Experimental Historical OCR fallback (`custom_historical_ocr`).
- Tier 4: Human Scholar Escalation when all engines fail or produce unresolved glyphs.
"""

import gc
import hashlib
import io
import os
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from PIL import Image

from services.ai.common.base_provider import OCRResult
from services.ai.common.provider_manager import get_provider_manager, ProviderManager
from restoration_engine import rgb_to_gray

logger = logging.getLogger("kallvettu.adaptive_router")

# Estimated API costs per call (USD) for transparency
ESTIMATED_API_COSTS = {
    "paddleocr_local": 0.000,
    "historical_hog_svm": 0.000,
    "custom_historical_ocr": 0.000,
    "local_catalog_ocr": 0.000,
    "sarvam_vision": 0.005,       # ~0.5 cents per parse
    "gemini_vision": 0.002,       # ~0.2 cents per multimodal query
    "sarvam_translate": 0.001,
}


def apply_cpu_thread_limits(num_threads: int = 2) -> int:
    """
    Applies conservative CPU thread bounds (1..4) for standard Windows laptops
    across OpenMP, OpenBLAS, MKL, and OpenCV.
    """
    bounded = max(1, min(int(num_threads), 4))
    val = str(bounded)
    os.environ["OMP_NUM_THREADS"] = val
    os.environ["OPENBLAS_NUM_THREADS"] = val
    os.environ["MKL_NUM_THREADS"] = val
    try:
        cv2.setNumThreads(bounded)
    except Exception:
        pass
    return bounded


@dataclass
class CPURoutingConfig:
    """CPU-first execution governance for AdaptiveRoutingEngine."""
    cpu_threads: int = 2
    max_image_dim: int = 1600
    high_confidence_early_exit: float = 0.85
    enable_cache: bool = True
    cache_max_entries: int = 64
    allow_cloud_escalation: bool = True
    enable_hog_svm_historical: bool = True
    release_models_after_run: bool = False


@dataclass
class ImageQualityMetrics:
    """Quantitative computer vision metrics assessing stone inscription photograph quality."""
    laplacian_variance: float     # Blur measure (higher = sharper chiseled edges; < 80 indicates severe blur)
    michelson_contrast: float     # Contrast in [0.0, 1.0] ((max - min) / (max + min))
    mean_luminance: float         # Exposure level in [0.0, 255.0]
    estimated_noise_floor: float  # High-frequency granite grain variance
    quality_grade: str            # 'pristine', 'good', 'weathered', 'heavily_eroded'
    recommended_enhancement: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laplacian_variance": round(self.laplacian_variance, 2),
            "michelson_contrast": round(self.michelson_contrast, 4),
            "mean_luminance": round(self.mean_luminance, 2),
            "estimated_noise_floor": round(self.estimated_noise_floor, 2),
            "quality_grade": self.quality_grade,
            "recommended_enhancement": self.recommended_enhancement,
        }


@dataclass
class RoutingExecutionReport:
    """Complete diagnostic log of adaptive routing decisions and resource expenditure."""
    primary_engine_used: str
    escalations_triggered: List[str]
    engines_executed: List[str]
    total_latency_seconds: float
    total_estimated_cost_usd: float
    image_quality: ImageQualityMetrics
    requires_human_escalation: bool
    escalation_rationales: List[str]
    results_by_provider: Dict[str, OCRResult]
    cache_hit: bool = False
    early_exit_triggered: bool = False
    cpu_threads_applied: int = 2
    execution_mode: str = "sequential_cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_engine_used": self.primary_engine_used,
            "escalations_triggered": self.escalations_triggered,
            "engines_executed": self.engines_executed,
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 5),
            "image_quality": self.image_quality.to_dict(),
            "requires_human_escalation": self.requires_human_escalation,
            "escalation_rationales": self.escalation_rationales,
            "cache_hit": self.cache_hit,
            "early_exit_triggered": self.early_exit_triggered,
            "cpu_threads_applied": self.cpu_threads_applied,
            "execution_mode": self.execution_mode,
        }


class AdaptiveRoutingEngine:
    """
    Tiered epigraphical dispatch engine routing inscription images sequentially on CPU.
    """

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        config: Optional[CPURoutingConfig] = None,
    ):
        self._manager = provider_manager or get_provider_manager()
        self.config = config or CPURoutingConfig()
        self._prediction_cache: OrderedDict[str, RoutingExecutionReport] = OrderedDict()

    def clear_cache(self) -> None:
        self._prediction_cache.clear()

    @staticmethod
    def _compute_image_cache_key(
        pil_img: Image.Image,
        script_hint: str,
        force_all_engines: bool,
        allow_cloud: bool,
    ) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        return f"{digest}|{script_hint.lower()}|all={force_all_engines}|cloud={allow_cloud}"

    @staticmethod
    def _clamp_image_aspect_preserving(pil_img: Image.Image, max_dim: int = 1600) -> Image.Image:
        w, h = pil_img.size
        if max(w, h) <= max_dim:
            return pil_img
        scale = max_dim / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    def assess_image_quality(self, image: Union[Image.Image, np.ndarray]) -> ImageQualityMetrics:
        """
        Calculates blur, contrast, luminance, and grain metrics on stone photographs.
        """
        if isinstance(image, Image.Image):
            gray = rgb_to_gray(np.array(image.convert("RGB")))
        elif isinstance(image, np.ndarray):
            gray = rgb_to_gray(image) if len(image.shape) == 3 else (image.astype(float) / 255.0)
        else:
            return ImageQualityMetrics(100.0, 0.5, 128.0, 10.0, "good", "None")

        gray_255 = (gray * 255.0).astype(np.uint8)

        # 1. Laplacian variance for blur
        padded = np.pad(gray_255, 1, mode="reflect").astype(float)
        lap = (
            padded[0:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))

        # 2. Michelson Contrast
        min_v = float(np.min(gray_255))
        max_v = float(np.max(gray_255))
        contrast = float((max_v - min_v) / max(1.0, (max_v + min_v)))

        # 3. Mean luminance
        mean_lum = float(np.mean(gray_255))

        # 4. Noise floor estimate
        high_freq = gray_255 - lap
        noise = float(np.std(high_freq))

        # Classify grade
        if lap_var > 220.0 and contrast > 0.65:
            grade = "pristine"
            rec = "Direct Path A (No enhancement required)"
        elif lap_var > 120.0 and contrast > 0.45:
            grade = "good"
            rec = "Mild CLAHE (Path B) recommended"
        elif lap_var > 60.0 or contrast > 0.25:
            grade = "weathered"
            rec = "Bilateral Denoising + Mild CLAHE recommended"
        else:
            grade = "heavily_eroded"
            rec = "Multi-spectral Chisel Groove extraction & visual escalation"

        return ImageQualityMetrics(
            laplacian_variance=lap_var,
            michelson_contrast=contrast,
            mean_luminance=mean_lum,
            estimated_noise_floor=noise,
            quality_grade=grade,
            recommended_enhancement=rec,
        )

    def route_and_execute(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: str = "auto",
        force_all_engines: bool = False,
        **kwargs: Any
    ) -> RoutingExecutionReport:
        """
        Executes sequential, memory-bounded CPU recognition with prediction caching,
        local high-confidence early exit, and selective cloud escalation.
        """
        t0 = time.perf_counter()
        applied_threads = apply_cpu_thread_limits(self.config.cpu_threads)
        allow_cloud = bool(kwargs.pop("allow_cloud_escalation", self.config.allow_cloud_escalation))

        # Safe Ingestion (preserving original; clamping working copy <= max_image_dim)
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        working_img = self._clamp_image_aspect_preserving(pil_img, max_dim=self.config.max_image_dim)

        # Prediction cache lookup
        cache_key = self._compute_image_cache_key(working_img, script_hint, force_all_engines, allow_cloud)
        if self.config.enable_cache and cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            self._prediction_cache.move_to_end(cache_key)
            return RoutingExecutionReport(
                primary_engine_used=cached.primary_engine_used,
                escalations_triggered=list(cached.escalations_triggered),
                engines_executed=list(cached.engines_executed),
                total_latency_seconds=time.perf_counter() - t0,
                total_estimated_cost_usd=0.0,  # Zero cost on cache hit
                image_quality=cached.image_quality,
                requires_human_escalation=cached.requires_human_escalation,
                escalation_rationales=list(cached.escalation_rationales),
                results_by_provider=dict(cached.results_by_provider),
                cache_hit=True,
                early_exit_triggered=cached.early_exit_triggered,
                cpu_threads_applied=applied_threads,
                execution_mode="sequential_cpu_cached",
            )

        # Step 1: Assess image quality
        quality = self.assess_image_quality(working_img)

        executed_engines: List[str] = []
        escalations: List[str] = []
        rationales: List[str] = []
        results: Dict[str, OCRResult] = {}
        total_cost = 0.0
        early_exit = False

        is_archaic_script = any(
            s in script_hint.lower() for s in ["brahmi", "vatteluttu", "grantha"]
        )

        # Tier 1a: Archaic Script Specialist (CPU HOG+SVM Line Recognizer) - runs sequentially first when archaic
        hog_res: Optional[OCRResult] = None
        if is_archaic_script and self.config.enable_hog_svm_historical:
            hog_prov = self._manager.get_ocr_provider("historical_hog_svm")
            if hog_prov and self._manager.is_enabled("historical_hog_svm") and hog_prov.is_configured():
                try:
                    hog_res = hog_prov.recognize(
                        working_img,
                        script_hint=script_hint,
                        mode=kwargs.get("hog_mode", "line"),
                    )
                    if hog_res and hog_res.text.strip():
                        results["historical_hog_svm"] = hog_res
                        executed_engines.append("historical_hog_svm")
                        total_cost += ESTIMATED_API_COSTS["historical_hog_svm"]
                except Exception as e_hog:
                    logger.debug("Tier 1a Historical HOG+SVM skipped: %s", e_hog)

        # Tier 1b: Primary Local OCR (PaddleOCR CPU) - never loaded concurrently with cloud/other heavy engines
        paddle_prov = self._manager.get_ocr_provider("paddleocr_local")
        paddle_res: Optional[OCRResult] = None

        if paddle_prov and self._manager.is_enabled("paddleocr_local"):
            try:
                if hasattr(paddle_prov, "_cpu_threads"):
                    paddle_prov._cpu_threads = applied_threads
                paddle_res = paddle_prov.recognize(working_img, script_hint=script_hint, **kwargs)
                if paddle_res and paddle_res.text.strip():
                    results["paddleocr_local"] = paddle_res
                    executed_engines.append("paddleocr_local")
                    total_cost += ESTIMATED_API_COSTS["paddleocr_local"]
            except Exception as e:
                logger.warning("Tier 1b PaddleOCR execution skipped: %s", e)
            finally:
                if self.config.release_models_after_run and hasattr(paddle_prov, "release_models"):
                    paddle_prov.release_models()
                    gc.collect()

        # Evaluate Early Exit Condition (avoid cloud escalation & further models if local confidence is strong)
        local_best_conf = None
        if paddle_res and paddle_res.confidence is not None and paddle_res.text.strip():
            local_best_conf = float(paddle_res.confidence)
        elif hog_res and hog_res.confidence is not None and hog_res.text.strip():
            local_best_conf = float(hog_res.confidence)

        if (
            not force_all_engines
            and not is_archaic_script
            and local_best_conf is not None
            and local_best_conf >= self.config.high_confidence_early_exit
            and quality.quality_grade in {"pristine", "good"}
        ):
            early_exit = True
            rationales.append(
                f"Early exit triggered: local confidence ({local_best_conf:.2f} >= "
                f"{self.config.high_confidence_early_exit:.2f}) on {quality.quality_grade} image."
            )

        # Decide whether to escalate beyond Tier 1
        should_escalate_cloud = force_all_engines
        requires_human = False

        if not force_all_engines and not early_exit:
            if is_archaic_script:
                should_escalate_cloud = True
                escalations.append("archaic_script_escalation")
                rationales.append(
                    f"Script hint '{script_hint}' is archaic; invoking historical & cross-verification engines."
                )

            if quality.quality_grade in {"weathered", "heavily_eroded"}:
                should_escalate_cloud = True
                escalations.append("weathered_stone_escalation")
                rationales.append(
                    f"Stone surface is {quality.quality_grade} (Laplacian var {quality.laplacian_variance:.1f} < 120)."
                )

            if not paddle_res or not paddle_res.text.strip():
                should_escalate_cloud = True
                escalations.append("empty_local_output_escalation")
                rationales.append("PaddleOCR produced empty or unparseable output on stone.")
            elif paddle_res.confidence is not None and paddle_res.confidence < self.config.high_confidence_early_exit:
                should_escalate_cloud = True
                escalations.append("low_local_confidence_escalation")
                rationales.append(
                    f"PaddleOCR confidence ({paddle_res.confidence:.2f}) below early-exit threshold "
                    f"{self.config.high_confidence_early_exit:.2f}."
                )

        # Tier 2: Selective Sequential Cloud Escalation (Sarvam Vision -> Gemini Vision)
        if should_escalate_cloud and allow_cloud and not early_exit:
            sarvam_prov = self._manager.get_ocr_provider("sarvam_vision")
            if sarvam_prov and self._manager.is_enabled("sarvam_vision") and sarvam_prov.is_configured():
                try:
                    res_s = sarvam_prov.recognize(pil_img, script_hint=script_hint, safe=True, **kwargs)
                    if res_s and res_s.text.strip():
                        results["sarvam_vision"] = res_s
                        executed_engines.append("sarvam_vision")
                        total_cost += ESTIMATED_API_COSTS["sarvam_vision"]
                except Exception as e_s:
                    logger.debug("Sarvam Vision escalation failed: %s", e_s)

            gemini_prov = self._manager.get_ocr_provider("gemini_vision")
            if gemini_prov and self._manager.is_enabled("gemini_vision") and gemini_prov.is_configured():
                try:
                    res_g = gemini_prov.recognize(working_img, script_hint=script_hint, **kwargs)
                    if res_g and res_g.text.strip():
                        results["gemini_vision"] = res_g
                        executed_engines.append("gemini_vision")
                        total_cost += ESTIMATED_API_COSTS["gemini_vision"]
                except Exception as e_g:
                    logger.debug("Gemini Vision escalation failed: %s", e_g)
        elif should_escalate_cloud and not allow_cloud:
            rationales.append("Cloud escalation suppressed by CPU-only / offline policy.")

        # Tier 3: Experimental Historical OCR (if archaic or insufficient candidates)
        if not early_exit and (is_archaic_script or force_all_engines or len(results) < 2):
            hist_prov = self._manager.get_ocr_provider("custom_historical_ocr")
            if hist_prov and self._manager.is_enabled("custom_historical_ocr"):
                try:
                    res_h = hist_prov.recognize(working_img, script_hint=script_hint, **kwargs)
                    if res_h and res_h.text.strip():
                        results["custom_historical_ocr"] = res_h
                        executed_engines.append("custom_historical_ocr")
                        total_cost += ESTIMATED_API_COSTS["custom_historical_ocr"]
                except Exception as e_h:
                    logger.debug("Custom Historical OCR execution skipped: %s", e_h)

            # Also try historical_hog_svm if force_all_engines was requested and wasn't run in Tier 1a
            if force_all_engines and "historical_hog_svm" not in results and self.config.enable_hog_svm_historical:
                hog_prov = self._manager.get_ocr_provider("historical_hog_svm")
                if hog_prov and self._manager.is_enabled("historical_hog_svm") and hog_prov.is_configured():
                    try:
                        res_hog2 = hog_prov.recognize(
                            working_img, script_hint=script_hint, mode=kwargs.get("hog_mode", "line")
                        )
                        if res_hog2 and res_hog2.text.strip():
                            results["historical_hog_svm"] = res_hog2
                            executed_engines.append("historical_hog_svm")
                            total_cost += ESTIMATED_API_COSTS["historical_hog_svm"]
                    except Exception as e_hog2:
                        logger.debug("Historical HOG+SVM skipped: %s", e_hog2)

        # Tier 4: Human Escalation Check
        if not results:
            requires_human = True
            escalations.append("human_scholar_escalation")
            rationales.append(
                "All automated OCR engines failed to recognize character strokes. Direct visual study required."
            )

        elapsed = time.perf_counter() - t0
        report = RoutingExecutionReport(
            primary_engine_used=executed_engines[0] if executed_engines else "none",
            escalations_triggered=escalations,
            engines_executed=executed_engines,
            total_latency_seconds=elapsed,
            total_estimated_cost_usd=total_cost,
            image_quality=quality,
            requires_human_escalation=requires_human,
            escalation_rationales=rationales,
            results_by_provider=results,
            cache_hit=False,
            early_exit_triggered=early_exit,
            cpu_threads_applied=applied_threads,
            execution_mode="sequential_cpu",
        )

        # Store in bounded LRU cache
        if self.config.enable_cache:
            self._prediction_cache[cache_key] = report
            if len(self._prediction_cache) > max(1, self.config.cache_max_entries):
                self._prediction_cache.popitem(last=False)

        return report
