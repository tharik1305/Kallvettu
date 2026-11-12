"""
PaddleOCR Local Offline OCR Provider for Kallvettu.
Implements BaseOCRProvider supporting:
- PP-OCRv5 Tamil Recognition (ta_PP-OCRv5_mobile_rec)
- PP-OCRv6 Text Detection (PP-OCRv6_medium_det)
- Complete offline execution without cloud APIs or network access
- Bounding box and inference confidence extraction
- Detection + crop + recognition pipeline for stone inscription photographs
- Accurate runtime state reporting (NOT INSTALLED, MODEL DOWNLOAD REQUIRED, AVAILABLE — CPU, ERROR)
"""

import io
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    ProviderType,
    ProviderStatus,
    OCRResult,
    BoundingBox,
)
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    KallvettuAIError,
)
from services.ai.common.response_validator import (
    clean_text_artifacts,
    clamp_confidence,
    normalize_bounding_box,
    validate_tamil_unicode,
)

logger = logging.getLogger("kallvettu.paddleocr")

# Configure Paddle cache home inside workspace to ensure reliable sandboxed & offline operation
_DEFAULT_PADDLE_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".paddlex_cache")
)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", _DEFAULT_PADDLE_CACHE)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class PaddleOCRProvider(BaseOCRProvider):
    """
    Dedicated Local / Offline OCR Provider based on official PaddleOCR 3.x framework.
    Combines text line detection (PP-OCRv6) with PP-OCRv5 Tamil text line recognition.
    """

    PRIMARY_REC_MODEL = "ta_PP-OCRv5_mobile_rec"
    PRIMARY_DET_MODEL = "PP-OCRv6_medium_det"
    MAX_IMAGE_DIM = 3200
    MIN_IMAGE_DIM = 10

    def __init__(self, use_gpu: bool = False, model_name: Optional[str] = None, cpu_threads: int = 2):
        self._use_gpu = use_gpu
        self._model_name = model_name or self.PRIMARY_REC_MODEL
        self._det_model_name = self.PRIMARY_DET_MODEL
        self._cpu_threads = max(1, min(int(cpu_threads), 4))
        
        # Lazy model instances
        self._det_engine: Optional[Any] = None
        self._rec_engine: Optional[Any] = None
        self._mock_engine: Optional[Any] = None
        
        # Diagnostic runtime info
        self._is_pkg_installed: bool = False
        self._paddle_version: Optional[str] = None
        self._paddleocr_version: Optional[str] = None
        self._cuda_available: bool = False
        self._init_error: Optional[str] = None

        self._inspect_environment()

    def _inspect_environment(self) -> None:
        """
        Inspects package installations and versions lazily via importlib metadata
        (avoiding eager 40+ second C++ DLL loads on application startup).
        """
        import importlib.util
        import importlib.metadata
        import sys

        if importlib.util.find_spec("paddle") is None:
            self._is_pkg_installed = False
            self._init_error = "PaddlePaddle package is not installed."
            return

        if importlib.util.find_spec("paddleocr") is None:
            self._is_pkg_installed = False
            self._init_error = "PaddleOCR package is not installed."
            return

        try:
            try:
                self._paddle_version = importlib.metadata.version("paddlepaddle")
            except Exception:
                self._paddle_version = "installed"

            try:
                self._paddleocr_version = importlib.metadata.version("paddleocr")
            except Exception:
                self._paddleocr_version = "installed"

            self._is_pkg_installed = True
            # Check CUDA only if paddle is already loaded in sys.modules
            if "paddle" in sys.modules:
                try:
                    self._cuda_available = bool(sys.modules["paddle"].device.is_compiled_with_cuda())
                except Exception:
                    self._cuda_available = False
            else:
                self._cuda_available = False
        except Exception as e:
            self._is_pkg_installed = False
            self._init_error = f"PaddleOCR inspection error: {e}"

    @property
    def provider_name(self) -> str:
        return "PaddleOCR Local Engine (PP-OCRv5 Tamil)"

    @property
    def provider_id(self) -> str:
        return "paddleocr_local"

    @property
    def is_package_installed(self) -> bool:
        return self._is_pkg_installed

    @property
    def is_cuda_available(self) -> bool:
        return self._cuda_available

    def _are_model_weights_cached(self) -> bool:
        """Verifies if the required model weights exist in local cache directory."""
        cache_dir = os.environ.get("PADDLE_PDX_CACHE_HOME", _DEFAULT_PADDLE_CACHE)
        rec_dir = os.path.join(cache_dir, "official_models", self._model_name)
        return os.path.isdir(rec_dir) and len(os.listdir(rec_dir)) >= 3

    def get_runtime_status(self) -> Dict[str, Any]:
        """
        Returns precise runtime diagnostic information following 4-state lifecycle:
        - State 1: Package missing -> NOT INSTALLED
        - State 2: Packages installed, weights missing -> MODEL DOWNLOAD REQUIRED
        - State 3: Model initialized / weights ready -> AVAILABLE — CPU (or GPU)
        - State 4: Error -> ERROR
        """
        if not self._is_pkg_installed and self._mock_engine is None:
            return {
                "state": "NOT_INSTALLED",
                "configured": False,
                "enabled": False,
                "status_label": "NOT INSTALLED",
                "message": "PaddlePaddle or PaddleOCR is not installed in the environment.",
                "device": "N/A",
                "paddle_version": None,
                "paddleocr_version": None,
                "cuda_compiled": False,
                "weights_cached": False,
            }

        if self._init_error and self._mock_engine is None:
            return {
                "state": "ERROR",
                "configured": False,
                "enabled": False,
                "status_label": "ERROR",
                "message": self._init_error,
                "device": "N/A",
                "paddle_version": self._paddle_version,
                "paddleocr_version": self._paddleocr_version,
                "cuda_compiled": self._cuda_available,
                "weights_cached": False,
            }

        weights_present = self._are_model_weights_cached() or (self._mock_engine is not None)
        if not weights_present and self._rec_engine is None:
            return {
                "state": "MODEL_DOWNLOAD_REQUIRED",
                "configured": False,
                "enabled": False,
                "status_label": "MODEL DOWNLOAD REQUIRED",
                "message": (
                    f"Model weights for '{self._model_name}' are not yet downloaded. "
                    "Run initial inference or test connection to cache weights locally."
                ),
                "device": "CPU" if not self._cuda_available else "GPU",
                "paddle_version": self._paddle_version,
                "paddleocr_version": self._paddleocr_version,
                "cuda_compiled": self._cuda_available,
                "weights_cached": False,
            }

        device_label = "GPU" if (self._use_gpu and self._cuda_available) else "CPU"
        return {
            "state": "AVAILABLE",
            "configured": True,
            "enabled": True,
            "status_label": f"AVAILABLE — {device_label}",
            "message": f"Local inference ready ({device_label} mode, Model: {self._model_name})",
            "device": device_label,
            "paddle_version": self._paddle_version,
            "paddleocr_version": self._paddleocr_version,
            "cuda_compiled": self._cuda_available,
            "weights_cached": True,
        }

    def is_configured(self) -> bool:
        """Provider is configured when packages are installed and model is ready or cached."""
        status = self.get_runtime_status()
        return status["configured"]

    def check_health(self) -> Tuple[bool, str]:
        status = self.get_runtime_status()
        if status["state"] == "AVAILABLE":
            return True, status["message"]
        elif status["state"] == "MODEL_DOWNLOAD_REQUIRED":
            return False, status["message"]
        elif status["state"] == "NOT_INSTALLED":
            return False, status["message"]
        else:
            return False, status.get("message", "PaddleOCR in error state.")

    def _init_models(self) -> None:
        """Initializes and caches detection and recognition models once for reuse."""
        if self._mock_engine is not None:
            return

        if not self._is_pkg_installed:
            raise ProviderNotConfiguredError(
                "PaddleOCR is not installed. Install via pip install paddlepaddle paddleocr",
                provider=self.provider_name
            )

        try:
            import paddle
            try:
                self._cuda_available = bool(paddle.device.is_compiled_with_cuda())
                paddle.set_num_threads(self._cpu_threads)
            except Exception:
                pass

            from paddleocr import TextDetection, TextRecognition

            if self._det_engine is None:
                logger.info("Initializing PaddleOCR TextDetection engine (threads=%d)...", self._cpu_threads)
                self._det_engine = TextDetection()

            if self._rec_engine is None:
                logger.info("Initializing PaddleOCR TextRecognition engine with %s...", self._model_name)
                self._rec_engine = TextRecognition(model_name=self._model_name)

        except Exception as e:
            self._init_error = str(e)
            raise KallvettuAIError(
                f"Failed to initialize PaddleOCR models: {e}",
                provider=self.provider_name,
                raw_error=e
            ) from e

    def set_mock_engine(self, mock_engine: Any) -> None:
        """Injects mock engine for unit testing without downloading model weights."""
        self._mock_engine = mock_engine
        self._is_pkg_installed = True

    def release_models(self) -> None:
        """Releases cached det/rec models to free RAM during sequential CPU routing."""
        self._det_engine = None
        self._rec_engine = None

    def _validate_and_prepare_image(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str]
    ) -> Tuple[np.ndarray, int, int]:
        """
        Validates input image, rejects invalid or tiny images, clamps oversized
        dimensions to prevent memory exhaustion, and returns BGR numpy array with (h, w).
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        # Load into numpy array
        if isinstance(image, np.ndarray):
            if image.size == 0 or len(image.shape) < 2:
                raise ValueError("Input numpy array is empty or has invalid shape.")
            # Convert RGB or Gray to BGR for OpenCV / PaddleX
            if len(image.shape) == 2:
                img_bgr = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # RGBA -> BGR
                img_rgb = image[:, :, :3]
                img_bgr = img_rgb[:, :, ::-1]
            elif image.shape[2] == 3:
                # Assume incoming RGB array -> BGR
                img_bgr = image[:, :, ::-1].copy()
            else:
                raise ValueError(f"Unsupported numpy array channel count: {image.shape}")

        elif isinstance(image, Image.Image):
            rgb = image.convert("RGB")
            arr = np.array(rgb)
            img_bgr = arr[:, :, ::-1].copy()

        elif isinstance(image, (bytes, bytearray)):
            if len(image) == 0:
                raise ValueError("Image byte buffer is empty.")
            try:
                pil_img = Image.open(io.BytesIO(image)).convert("RGB")
                arr = np.array(pil_img)
                img_bgr = arr[:, :, ::-1].copy()
            except Exception as e:
                raise ValueError(f"Failed to decode image bytes: {e}") from e

        elif isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Image file does not exist: {image}")
            try:
                pil_img = Image.open(image).convert("RGB")
                arr = np.array(pil_img)
                img_bgr = arr[:, :, ::-1].copy()
            except Exception as e:
                raise ValueError(f"Failed to load image file '{image}': {e}") from e
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        h, w = img_bgr.shape[:2]
        if h < self.MIN_IMAGE_DIM or w < self.MIN_IMAGE_DIM:
            raise ValueError(
                f"Image dimensions ({w}x{h}) are too small for OCR (minimum {self.MIN_IMAGE_DIM}x{self.MIN_IMAGE_DIM})."
            )

        # Clamp oversized dimensions maintaining aspect ratio (CWE-400 protection)
        if max(h, w) > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / float(max(h, w))
            new_w = max(int(w * scale), self.MIN_IMAGE_DIM)
            new_h = max(int(h * scale), self.MIN_IMAGE_DIM)
            pil_temp = Image.fromarray(img_bgr[:, :, ::-1])
            pil_resized = pil_temp.resize((new_w, new_h), Image.Resampling.BILINEAR)
            img_bgr = np.array(pil_resized)[:, :, ::-1].copy()
            h, w = img_bgr.shape[:2]

        return img_bgr, h, w

    @staticmethod
    def _sort_polygons_reading_order(polys: List[Any]) -> List[Tuple[int, int, int, int, Any]]:
        """
        Sorts detected text polygons in natural reading order:
        Top-to-bottom, left-to-right, clustering lines using vertical overlap.
        Returns list of (xmin, ymin, xmax, ymax, poly).
        """
        if not polys:
            return []

        parsed = []
        for p in polys:
            pts = np.array(p, dtype=np.float32)
            xmin, ymin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            xmax, ymax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            h = max(ymax - ymin, 1.0)
            y_center = (ymin + ymax) / 2.0
            parsed.append({
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "h": h, "y_center": y_center, "poly": p
            })

        # Sort initially by y_center
        parsed.sort(key=lambda b: b["y_center"])

        # Cluster into lines
        lines: List[List[Dict[str, Any]]] = []
        for box in parsed:
            placed = False
            for line in lines:
                line_avg_h = float(np.mean([b["h"] for b in line]))
                line_avg_yc = float(np.mean([b["y_center"] for b in line]))
                if abs(box["y_center"] - line_avg_yc) < (line_avg_h * 0.5):
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        # Sort lines by average y_center
        lines.sort(key=lambda line: float(np.mean([b["y_center"] for b in line])))

        # Within each line, sort boxes by xmin (left to right)
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda b: b["xmin"])
            for b in line:
                sorted_boxes.append((int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]), b["poly"]))

        return sorted_boxes

    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any
    ) -> OCRResult:
        """
        Executes local detection-and-recognition pipeline on inscription photograph.
        1. Validates and scales image.
        2. Detects text lines using PP-OCRv6 detector.
        3. Crops each detected text line with reading-order sorting.
        4. Transcribes each line using ta_PP-OCRv5_mobile_rec.
        5. Preserves genuine inference confidence and bounding boxes.
        """
        if not self._is_pkg_installed and self._mock_engine is None:
            raise ProviderNotConfiguredError(
                self._init_error or "PaddleOCR is not installed. Install via pip install paddlepaddle paddleocr",
                provider=self.provider_name
            )

        t0 = time.perf_counter()
        img_bgr, h, w = self._validate_and_prepare_image(image)

        # Handle mock engine path for tests
        if self._mock_engine is not None:
            return self._run_mock_recognition(img_bgr, h, w, t0)

        self._init_models()

        extracted_lines: List[str] = []
        bounding_boxes: List[BoundingBox] = []
        confidences: List[float] = []

        try:
            # 1. Text Line Detection
            det_results = self._det_engine.predict(img_bgr)
            polys: List[Any] = []
            for r in det_results:
                dt_p = r.get("dt_polys", []) if isinstance(r, dict) else getattr(r, "dt_polys", [])
                if dt_p is not None:
                    polys.extend(list(dt_p))

            # 2. Sort polygons top-to-bottom, left-to-right
            sorted_boxes = self._sort_polygons_reading_order(polys)
            for xmin, ymin, xmax, ymax, p in sorted_boxes:
                # Clamp to image bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)

                if xmax > xmin and ymax > ymin:
                    crop = img_bgr[ymin:ymax, xmin:xmax]
                    rec_res = self._rec_engine.predict(crop)
                    line_text = ""
                    line_conf: Optional[float] = None

                    for res in rec_res:
                        txt = res.get("rec_text", "") if isinstance(res, dict) else getattr(res, "rec_text", "")
                        score = res.get("rec_score", None) if isinstance(res, dict) else getattr(res, "rec_score", None)
                        if txt and txt.strip():
                            line_text = txt.strip()
                            line_conf = float(score) if score is not None else None

                    if line_text:
                        extracted_lines.append(line_text)
                        if line_conf is not None:
                            confidences.append(line_conf)

                        bounding_boxes.append(BoundingBox(
                            ymin=float(ymin) / h,
                            xmin=float(xmin) / w,
                            ymax=float(ymax) / h,
                            xmax=float(xmax) / w,
                            text=line_text,
                            confidence=clamp_confidence(line_conf) if line_conf is not None else None,
                        ))

            # 3. Fallback: If detector found no lines (e.g. single-word tight crop), run recognizer on full crop
            if not extracted_lines:
                rec_res = self._rec_engine.predict(img_bgr)
                for res in rec_res:
                    txt = res.get("rec_text", "") if isinstance(res, dict) else getattr(res, "rec_text", "")
                    score = res.get("rec_score", None) if isinstance(res, dict) else getattr(res, "rec_score", None)
                    if txt and txt.strip():
                        extracted_lines.append(txt.strip())
                        if score is not None:
                            confidences.append(float(score))

        except Exception as e:
            raise KallvettuAIError(
                f"PaddleOCR local inference failed: {e}",
                provider=self.provider_name,
                raw_error=e
            ) from e

        recognized_text = "\n".join(extracted_lines)
        cleaned_text = clean_text_artifacts(recognized_text)
        
        # Genuine inference confidence only — never fabricate if empty
        mean_conf: Optional[float] = None
        if confidences:
            mean_conf = float(np.mean(confidences))

        latency = time.perf_counter() - t0
        device_label = "GPU" if (self._use_gpu and self._cuda_available) else "CPU"

        return OCRResult(
            text=cleaned_text,
            provider_name=self.provider_name,
            model_id=self._model_name,
            confidence=clamp_confidence(mean_conf) if mean_conf is not None else 0.0,
            latency_seconds=latency,
            bounding_boxes=bounding_boxes,
            raw_response={
                "lines_count": len(extracted_lines),
                "det_model": self._det_model_name,
                "rec_model": self._model_name,
                "device": device_label,
            },
            is_cloud=False,
            is_historical_script=False,
            provenance_notes=f"PaddleOCR Local Inference ({self._model_name} on {device_label})",
            metadata={
                "model": self._model_name,
                "det_model": self._det_model_name,
                "device": device_label,
                "use_gpu_requested": self._use_gpu,
                "cuda_compiled": self._cuda_available,
                "paddle_version": self._paddle_version,
                "paddleocr_version": self._paddleocr_version,
                "lines_detected": len(extracted_lines),
                "has_tamil_unicode": validate_tamil_unicode(cleaned_text),
                "transcription_status": "success" if cleaned_text else "empty",
                "script_limitation_note": (
                    "PP-OCRv5 Tamil is intended for modern Tamil text recognition. "
                    "Ancient Tamil-Brahmi and Vatteluttu recognition has not been established without dedicated validation."
                ),
            }
        )

    def _run_mock_recognition(self, img_bgr: np.ndarray, h: int, w: int, t0: float) -> OCRResult:
        """Helper to execute mocked engine predictions during automated tests."""
        mock = self._mock_engine
        res = mock.predict(img_bgr)
        extracted_lines = []
        confidences = []
        bounding_boxes = []

        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    txt = item.get("rec_text", "")
                    score = item.get("rec_score", 0.9)
                    if txt:
                        extracted_lines.append(txt)
                        confidences.append(score)
                        bounding_boxes.append(BoundingBox(
                            ymin=0.1,
                            xmin=0.1,
                            ymax=0.9,
                            xmax=0.9,
                            confidence=score,
                            text=txt
                        ))
                elif isinstance(item, str):
                    extracted_lines.append(item)
                    confidences.append(0.9)

        cleaned = clean_text_artifacts("\n".join(extracted_lines))
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=cleaned,
            provider_name=self.provider_name,
            model_id=self._model_name,
            confidence=clamp_confidence(mean_conf),
            latency_seconds=time.perf_counter() - t0,
            bounding_boxes=bounding_boxes,
            raw_response={"mock": True},
            is_cloud=False,
            is_historical_script=False,
            provenance_notes="PaddleOCR Mocked Test Inference",
            metadata={
                "model": self._model_name,
                "device": "CPU",
                "has_tamil_unicode": validate_tamil_unicode(cleaned),
                "transcription_status": "success" if cleaned else "empty",
                "script_limitation_note": (
                    "PP-OCRv5 Tamil is intended for modern Tamil text recognition. "
                    "Ancient Tamil-Brahmi and Vatteluttu recognition has not been established without dedicated validation."
                ),
            }
        )
