"""
Synthetic Scholarly Voice Narration Engine (Phase J).

Provides phonetic epigraphical reading and historical narration:
- Bilingual narration: Classical Tamil / Modern Tamil and Scholarly English commentary.
- Browser Web Speech API synthesizer script generator.
- Strictly adheres to Rule 9: Always tags outputs as SYNTHETIC_AI_VOICE with scholarly disclaimers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class NarrationPayload:
    inscription_id: str
    target_language: str
    speech_text: str
    provenance_badge: str
    synthetic_disclaimer: str
    browser_tts_js: str


class VoiceNarratorService:
    """Manages synthetic narration for historical inscriptions."""

    DISCLAIMER = (
        "Notice: This audio is a synthetic computer-generated historical reconstruction "
        "(Kallvettu Phonetic Reconstructor). It does not represent an authentic ancient human recording."
    )

    @classmethod
    def generate_narration_payload(
        cls,
        inscription_id: str,
        text_content: str,
        language: str = "ta",
        reading_type: str = "Normalized Epigraphical Tamil",
        pitch: float = 1.0,
        rate: float = 0.85,
    ) -> NarrationPayload:
        """
        Creates a narration payload and browser Web Speech API runner.
        """
        clean_text = text_content.strip()
        lang_code = "ta-IN" if language.lower() in ("ta", "tamil") else "en-US"

        # Web Speech API JS snippet
        safe_text = json.dumps(clean_text)
        tts_js = f"""
        (function() {{
          if (!('speechSynthesis' in window)) {{
            console.warn('Web Speech API not supported in this browser.');
            return;
          }}
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance({safe_text});
          utterance.lang = '{lang_code}';
          utterance.rate = {rate};
          utterance.pitch = {pitch};
          window.speechSynthesis.speak(utterance);
        }})();
        """

        return NarrationPayload(
            inscription_id=inscription_id,
            target_language=lang_code,
            speech_text=clean_text,
            provenance_badge="SYNTHETIC_AI_VOICE",
            synthetic_disclaimer=cls.DISCLAIMER,
            browser_tts_js=tts_js.strip(),
        )

