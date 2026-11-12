"""
Tests for Phase 4: CPU-First Sequential Adaptive OCR Routing & K-IEFE Integration.
"""

import numpy as np
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    OCRResult,
    BoundingBox,
)
from services.ai.common.provider_manager import ProviderManager
from services.ocr_fusion.adaptive_router import (
    AdaptiveRoutingEngine,
    CPURoutingConfig,
    apply_cpu_thread_limits,
)


class MockPaddleProvider(BaseOCRProvider):
    def __init__(self, confidence: float = 0.95, text: str = "கோனேரின்மை கொண்டான்"):
        self.ret_confidence = confidence
        self.ret_text = text
        self.call_count = 0
        self.released = False

    @property
    def provider_name(self) -> str:
        return "PaddleOCR Local Engine (PP-OCRv5 Tamil)"

    @property
    def provider_id(self) -> str:
        return "paddleocr_local"

    def is_configured(self) -> bool:
        return True

    def check_health(self):
        return True, "Mock ready"

    def recognize(self, image, script_hint=None, **kwargs):
        self.call_count += 1
        return OCRResult(
            text=self.ret_text,
            provider_name=self.provider_name,
            model_id="mock-paddle-v5",
            confidence=self.ret_confidence,
            latency_seconds=0.01,
            bounding_boxes=[BoundingBox(0.0, 0.0, 1.0, 1.0, confidence=self.ret_confidence, text=self.ret_text)],
            raw_response={},
            is_cloud=False,
        )

    def release_models(self):
        self.released = True


class MockHOGSVMHistoricalProvider(BaseOCRProvider):
    def __init__(self):
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "Historical HOG+SVM Glyph Classifier (CPU)"

    @property
    def provider_id(self) -> str:
        return "historical_hog_svm"

    def is_configured(self) -> bool:
        return True

    def check_health(self):
        return True, "Mock ready"

    def recognize(self, image, script_hint=None, **kwargs):
        self.call_count += 1
        return OCRResult(
            text="நந்த ஸிரிய் குவன்",
            provider_name=self.provider_name,
            model_id="mock-hog-svm-brahmi",
            confidence=None,
            latency_seconds=0.02,
            bounding_boxes=[BoundingBox(0.0, 0.0, 1.0, 1.0, confidence=None, text="நந்த ஸிரிய் குவன்")],
            raw_response={},
            is_cloud=False,
            is_historical_script=True,
        )


class MockSarvamCloudProvider(BaseOCRProvider):
    def __init__(self):
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "Sarvam Vision Document AI"

    @property
    def provider_id(self) -> str:
        return "sarvam_vision"

    def is_configured(self) -> bool:
        return True

    def check_health(self):
        return True, "Mock cloud ready"

    def recognize(self, image, script_hint=None, **kwargs):
        self.call_count += 1
        return OCRResult(
            text="கோனேரின்மை கொண்டான்",
            provider_name=self.provider_name,
            model_id="sarvam-vision-v1",
            confidence=0.92,
            latency_seconds=0.45,
            bounding_boxes=[],
            raw_response={},
            is_cloud=True,
        )


def _create_mock_manager(paddle_conf=0.95, paddle_text="கோனேரின்மை கொண்டான்"):
    pm = ProviderManager()
    paddle = MockPaddleProvider(confidence=paddle_conf, text=paddle_text)
    hog = MockHOGSVMHistoricalProvider()
    sarvam = MockSarvamCloudProvider()

    pm.register_provider(paddle)
    pm.register_provider(hog)
    pm.register_provider(sarvam)
    return pm, paddle, hog, sarvam


def test_cpu_thread_limit_governance():
    assert apply_cpu_thread_limits(2) == 2
    assert apply_cpu_thread_limits(0) == 1
    assert apply_cpu_thread_limits(8) == 4


def test_early_exit_on_high_confidence_local():
    pm, paddle, hog, sarvam = _create_mock_manager(paddle_conf=0.96)
    router = AdaptiveRoutingEngine(
        provider_manager=pm,
        config=CPURoutingConfig(high_confidence_early_exit=0.85, allow_cloud_escalation=True),
    )

    # Sharp synthetic image (pristine grade)
    img = np.zeros((200, 400), dtype=np.uint8)
    img[::4, ::4] = 255

    report = router.route_and_execute(img, script_hint="Tamil")

    assert report.early_exit_triggered is True
    assert paddle.call_count == 1
    assert sarvam.call_count == 0  # Cloud was NOT invoked!
    assert hog.call_count == 0
    assert report.total_estimated_cost_usd == 0.0
    assert "paddleocr_local" in report.results_by_provider


def test_prediction_cache_prevents_duplicate_inference():
    pm, paddle, _, _ = _create_mock_manager(paddle_conf=0.95)
    router = AdaptiveRoutingEngine(
        provider_manager=pm,
        config=CPURoutingConfig(enable_cache=True),
    )

    img = Image.new("RGB", (100, 100), color=(200, 200, 200))

    # Run 1: Cold execution
    rep1 = router.route_and_execute(img, script_hint="Tamil")
    assert rep1.cache_hit is False
    assert paddle.call_count == 1

    # Run 2: Exact same image -> Cache Hit
    rep2 = router.route_and_execute(img, script_hint="Tamil")
    assert rep2.cache_hit is True
    assert paddle.call_count == 1  # Unchanged!
    assert rep2.total_estimated_cost_usd == 0.0


def test_archaic_script_routes_to_historical_hog_svm():
    pm, paddle, hog, sarvam = _create_mock_manager()
    router = AdaptiveRoutingEngine(
        provider_manager=pm,
        config=CPURoutingConfig(allow_cloud_escalation=False),
    )

    img = Image.new("RGB", (150, 150), color=(180, 180, 180))
    report = router.route_and_execute(img, script_hint="Tamil-Brahmi")

    assert hog.call_count == 1
    assert "historical_hog_svm" in report.results_by_provider
    assert "archaic_script_escalation" in report.escalations_triggered


def test_cloud_escalation_suppression_flag():
    # Local Paddle confidence is low (0.40) -> normally triggers cloud
    pm, paddle, hog, sarvam = _create_mock_manager(paddle_conf=0.40)
    router = AdaptiveRoutingEngine(
        provider_manager=pm,
        config=CPURoutingConfig(allow_cloud_escalation=False),
    )

    img = Image.new("RGB", (150, 150), color=(180, 180, 180))
    report = router.route_and_execute(img, script_hint="Tamil")

    assert paddle.call_count == 1
    assert sarvam.call_count == 0  # Suppressed by allow_cloud_escalation=False
    assert report.total_estimated_cost_usd == 0.0


def test_release_models_cleanup():
    pm, paddle, _, _ = _create_mock_manager(paddle_conf=0.92)
    router = AdaptiveRoutingEngine(
        provider_manager=pm,
        config=CPURoutingConfig(release_models_after_run=True),
    )

    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    router.route_and_execute(img, script_hint="Tamil")
    assert paddle.released is True

