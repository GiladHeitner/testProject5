"""Language settings for narration, subtitles, and TTS."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    whisper_code: str
    tts_language: str
    subtitle_font: str
    rtl: bool
    uppercase_ratio: float


_EN = LanguageConfig(
    code="en",
    whisper_code="en",
    tts_language="English",
    subtitle_font="Gibson",
    rtl=False,
    uppercase_ratio=0.15,
)

_AR = LanguageConfig(
    code="ar",
    whisper_code="ar",
    tts_language="Arabic",
    subtitle_font="DejaVu Sans",
    rtl=True,
    uppercase_ratio=0.0,
)


def is_arabic_text(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))


def resolve_language(value: str | None = None) -> LanguageConfig:
    raw = (value or os.environ.get("SHORTS_BOT_LANGUAGE") or "en").strip().lower()
    if raw in {"ar", "arabic", "arb"}:
        return _AR
    return _EN
