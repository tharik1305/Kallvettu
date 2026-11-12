"""
Unified Provider Manager for Kallvettu AI Ecosystem.
Orchestrates discovery, configuration, enablement toggles, status tracking,
and multi-engine execution across Sarvam AI, Gemini Vision, and PaddleOCR.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .base_provider import (
    ProviderType,
    ProviderStatus,
    BaseProvider,
    BaseOCRProvider,
    BaseTranslationProvider,
    BaseInterpretationProvider,
    BaseLinguisticProvider,
    OCRResult,
    TranslationResult,
    InterpretationResult,
    LinguisticAnalysisResult,
)
from .error_handler import (
    ProviderNotConfiguredError,
    format_provider_error,
)
from config import CONFIG

logger = logging.getLogger("kallvettu.provider_manager")


class ProviderManager:
    """
    Central coordinator for all AI providers in the Kallvettu platform.
    Ensures that failure or absence of one provider does not disrupt others.
    """

    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._enabled_flags: Dict[str, bool] = {}
        self._default_ocr: Optional[str] = None
        self._default_translation: Optional[str] = None
        self._default_interpretation: Optional[str] = None
        self._default_linguistic: Optional[str] = None

    def register_provider(self, provider: BaseProvider, is_default: bool = False) -> None:
        """Registers a provider instance into the manager."""
        pid = provider.provider_id
        self._providers[pid] = provider

        # Initialize enabled state based on config or default to True
        if pid not in self._enabled_flags:
            self._enabled_flags[pid] = True

        ptype = provider.provider_type
        if is_default or (ptype == ProviderType.OCR and self._default_ocr is None):
            self._default_ocr = pid
        if is_default or (ptype == ProviderType.TRANSLATION and self._default_translation is None):
            self._default_translation = pid
        if is_default or (ptype == ProviderType.INTERPRETATION and self._default_interpretation is None):
            self._default_interpretation = pid
        if is_default or (ptype == ProviderType.LINGUISTIC and self._default_linguistic is None):
            self._default_linguistic = pid

    def unregister_provider(self, provider_id: str) -> None:
        """Removes a provider from the manager."""
        self._providers.pop(provider_id, None)
        self._enabled_flags.pop(provider_id, None)

    def set_enabled(self, provider_id: str, enabled: bool) -> None:
        """Enables or disables a specific provider."""
        if provider_id in self._providers:
            self._enabled_flags[provider_id] = enabled

    def is_enabled(self, provider_id: str) -> bool:
        """Checks if a provider is explicitly enabled by the researcher/configuration."""
        return self._enabled_flags.get(provider_id, False)

    def get_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """Retrieves a provider by its unique identifier."""
        return self._providers.get(provider_id)

    def list_providers(self, provider_type: Optional[ProviderType] = None) -> List[BaseProvider]:
        """Lists registered providers, optionally filtered by capability."""
        if provider_type is None:
            return list(self._providers.values())
        return [p for p in self._providers.values() if p.provider_type == provider_type]

    get_providers_by_type = list_providers

    def get_status(self, provider_id: str) -> ProviderStatus:
        """Evaluates live operational status of a provider."""
        provider = self.get_provider(provider_id)
        if not provider:
            return ProviderStatus.ERROR

        if not self.is_enabled(provider_id):
            return ProviderStatus.UNCONFIGURED

        if not provider.is_configured():
            return ProviderStatus.UNCONFIGURED

        try:
            healthy, _ = provider.check_health()
            return ProviderStatus.AVAILABLE if healthy else ProviderStatus.DEGRADED
        except Exception:
            return ProviderStatus.ERROR

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Returns structured dictionary of all provider statuses for the Settings UI."""
        report = {}
        # Predefined display capabilities according to epigraphical architecture
        capabilities_map = {
            "local_catalog_ocr": "Catalog Matching",
            "gemini_vision": "Multimodal Vision",
            "sarvam_translate": "Translation",
            "sarvam_mayura": "Translation / Transliteration",
            "sarvam_chat_interpretation": "Interpretation",
            "sarvam_vision": "Document OCR",
            "paddleocr_local": "Local OCR",
            "custom_historical_ocr": "Historical Script OCR",
            "historical_hog_svm": "Historical HOG+SVM OCR (CPU)",
        }

        # Fixed display ordering of registered providers
        ordered_pids = [
            "local_catalog_ocr",
            "gemini_vision",
            "sarvam_translate",
            "sarvam_mayura",
            "sarvam_chat_interpretation",
            "sarvam_vision",
            "paddleocr_local",
            "custom_historical_ocr",
            "historical_hog_svm",
        ]

        for pid in ordered_pids:
            if pid not in self._providers:
                continue
            provider = self._providers[pid]
            is_cfg = provider.is_configured()
            is_en = self.is_enabled(pid)
            cap = capabilities_map.get(pid, provider.provider_type.value.upper())

            # Determine truthful runtime status and configuration
            if pid == "local_catalog_ocr":
                cfg_display = "YES"
                en_display = "Active" if is_en else "Disabled"
                status_display = "Available"
            elif pid == "historical_hog_svm":
                cfg_display = "YES" if is_cfg else "NO"
                en_display = "Active" if (is_en and is_cfg) else "Disabled"
                status_display = "Available (CPU)" if is_cfg else "Model not trained"
            elif pid == "paddleocr_local":
                # Detailed PaddleOCR runtime diagnosis
                if hasattr(provider, "get_runtime_status"):
                    p_stat = provider.get_runtime_status()
                    cfg_state = p_stat.get("state")
                    if cfg_state == "NOT_INSTALLED":
                        cfg_display = "NO"
                        en_display = "Disabled"
                        status_display = "NOT INSTALLED"
                    elif cfg_state == "MODEL_DOWNLOAD_REQUIRED":
                        cfg_display = "PARTIAL"
                        en_display = "Disabled"
                        status_display = "MODEL DOWNLOAD REQUIRED"
                    elif cfg_state == "AVAILABLE":
                        cfg_display = "YES"
                        en_display = "Active" if is_en else "Disabled"
                        status_display = p_stat.get("status_label", "AVAILABLE — CPU")
                    else:
                        cfg_display = "NO"
                        en_display = "Disabled"
                        status_display = "ERROR"
                else:
                    cfg_display = "YES" if is_cfg else "NO"
                    en_display = "Active" if (is_en and is_cfg) else "Disabled"
                    status_display = "Available" if is_cfg else "NOT INSTALLED"
            else:
                # Cloud API-dependent providers (Gemini, Sarvam)
                cfg_display = "YES" if is_cfg else "NO"
                en_display = "Active" if (is_en and is_cfg) else "Disabled"
                status_display = "Connected" if is_cfg else "Missing key"

            report[pid] = {
                "id": pid,
                "name": provider.provider_name,
                "capability": cap,
                "type": provider.provider_type.value,
                "configured": cfg_display,
                "enabled": en_display,
                "status": status_display,
            }
        return report

    # --------------------------------------------------------------------------
    # Specialized Provider Accessors
    # --------------------------------------------------------------------------

    def get_ocr_providers(self) -> List[BaseOCRProvider]:
        """Returns all registered OCR providers."""
        return [
            p for p in self.get_providers_by_type(ProviderType.OCR)
            if isinstance(p, BaseOCRProvider)
        ]

    def get_ocr_provider(self, provider_id: Optional[str] = None) -> Optional[BaseOCRProvider]:
        """Returns the requested or default OCR provider."""
        pid = provider_id or self._default_ocr
        if pid and pid in self._providers and self.is_enabled(pid):
            prov = self._providers[pid]
            if isinstance(prov, BaseOCRProvider):
                return prov
        return None

    def get_configured_ocr_providers(self) -> List[BaseOCRProvider]:
        """Returns all configured and enabled OCR providers."""
        return [
            p for p in self.get_providers_by_type(ProviderType.OCR)
            if isinstance(p, BaseOCRProvider) and p.is_configured() and self.is_enabled(p.provider_id)
        ]

    def get_translation_provider(self, provider_id: Optional[str] = None) -> Optional[BaseTranslationProvider]:
        """Returns the requested or default Translation provider."""
        pid = provider_id or self._default_translation
        if pid and pid in self._providers and self.is_enabled(pid):
            prov = self._providers[pid]
            if isinstance(prov, BaseTranslationProvider):
                return prov
        return None

    def get_interpretation_provider(self, provider_id: Optional[str] = None) -> Optional[BaseInterpretationProvider]:
        """Returns the requested or default Interpretation provider."""
        pid = provider_id or self._default_interpretation
        if pid and pid in self._providers and self.is_enabled(pid):
            prov = self._providers[pid]
            if isinstance(prov, BaseInterpretationProvider):
                return prov
        return None

    def get_linguistic_provider(self, provider_id: Optional[str] = None) -> Optional[BaseLinguisticProvider]:
        """Returns the requested or default Linguistic provider."""
        pid = provider_id or self._default_linguistic
        if pid and pid in self._providers and self.is_enabled(pid):
            prov = self._providers[pid]
            if isinstance(prov, BaseLinguisticProvider):
                return prov
        return None

    # --------------------------------------------------------------------------
    # Multi-Engine OCR Parallel/Sequential Executor
    # --------------------------------------------------------------------------

    def execute_multi_ocr(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        provider_ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Union[OCRResult, str]]:
        """
        Executes OCR across selected or all active OCR providers.
        Returns a dictionary mapping provider_id -> OCRResult (or error string on failure).
        """
        target_ids = provider_ids or [
            p.provider_id for p in self.list_providers(ProviderType.OCR) if self.is_enabled(p.provider_id)
        ]

        results: Dict[str, Union[OCRResult, str]] = {}
        for pid in target_ids:
            prov = self.get_ocr_provider(pid)
            if not prov:
                results[pid] = f"Provider '{pid}' is not available or disabled."
                continue

            try:
                t0 = time.perf_counter()
                res = prov.recognize(image, script_hint=script_hint, **kwargs)
                res.latency_seconds = time.perf_counter() - t0
                results[pid] = res
            except Exception as e:
                err_msg = format_provider_error(e, prov.provider_name)
                logger.warning("OCR execution error for provider '%s': %s", pid, err_msg)
                results[pid] = err_msg

        return results


# Global Singleton Provider Manager
_GLOBAL_PROVIDER_MANAGER: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Returns the shared singleton instance of ProviderManager, auto-registering standard providers."""
    global _GLOBAL_PROVIDER_MANAGER
    if _GLOBAL_PROVIDER_MANAGER is None:
        mgr = ProviderManager()
        # Register standard offline & cloud providers with graceful fallbacks
        try:
            from services.ai.offline.offline_ocr_provider import LocalCatalogOCRProvider
            mgr.register_provider(LocalCatalogOCRProvider(), is_default=True)
        except Exception as e:
            logger.debug("LocalCatalogOCRProvider registration skipped: %s", e)

        try:
            from services.ai.gemini.vision_provider import GeminiVisionProvider
            mgr.register_provider(GeminiVisionProvider())
        except Exception as e:
            logger.debug("GeminiVisionProvider registration skipped: %s", e)

        try:
            from services.ai.sarvam.translation_provider import SarvamTranslateProvider
            mgr.register_provider(SarvamTranslateProvider(), is_default=True)
        except Exception as e:
            logger.debug("SarvamTranslateProvider registration skipped: %s", e)

        try:
            from services.ai.sarvam.mayura_provider import MayuraTranslationProvider
            mgr.register_provider(MayuraTranslationProvider())
        except Exception as e:
            logger.debug("MayuraTranslationProvider registration skipped: %s", e)

        try:
            from services.ai.sarvam.chat_provider import SarvamChatInterpretationProvider
            mgr.register_provider(SarvamChatInterpretationProvider(), is_default=True)
        except Exception as e:
            logger.debug("SarvamChatInterpretationProvider registration skipped: %s", e)

        try:
            from services.ai.sarvam.vision_provider import SarvamVisionProvider
            mgr.register_provider(SarvamVisionProvider())
        except Exception as e:
            logger.debug("SarvamVisionProvider registration skipped: %s", e)

        try:
            from services.ai.offline.paddleocr_provider import PaddleOCRProvider
            mgr.register_provider(PaddleOCRProvider())
        except Exception as e:
            logger.debug("PaddleOCRProvider registration skipped: %s", e)

        try:
            from services.ai.offline.historical_ocr_provider import CustomHistoricalOCRProvider
            mgr.register_provider(CustomHistoricalOCRProvider())
        except Exception as e:
            logger.debug("CustomHistoricalOCRProvider registration skipped: %s", e)

        try:
            from services.ai.historical.inference import HistoricalHOGSVMProvider
            mgr.register_provider(HistoricalHOGSVMProvider())
        except Exception as e:
            logger.debug("HistoricalHOGSVMProvider registration skipped: %s", e)

        _GLOBAL_PROVIDER_MANAGER = mgr
    return _GLOBAL_PROVIDER_MANAGER

