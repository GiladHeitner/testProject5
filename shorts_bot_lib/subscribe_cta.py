"""Subscribe-button GIF overlay timed to the script's closing CTA line."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .runner import ffprobe_duration_seconds
from .types import PopupImage

DEFAULT_GIF = Path("assets/youtubebutton.gif")
DEFAULT_CTA_WIDTH = int(os.environ.get("SUBSCRIBE_CTA_WIDTH", "920"))
DEFAULT_CTA_Y = int(os.environ.get("SUBSCRIBE_CTA_Y", "780"))
DEFAULT_CHROMA_KEY = os.environ.get("SUBSCRIBE_CTA_CHROMA", "0x00FF00")
_SUBSCRIBE_PHRASES = (
    "subscribe before i get banned",
    "subscribe before",
    "subscribe",
)
_CTA_RESERVED_WORDS = frozenset({"subscribe", "banned"})


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def exclude_subscribe_from_keywords(keywords: List[dict]) -> List[dict]:
    """Drop keyword picks handled by the subscribe CTA overlay."""
    out: List[dict] = []
    for entry in keywords:
        phrase = _normalize(str(entry.get("phrase") or ""))
        if not phrase:
            continue
        tokens = set(phrase.split())
        if tokens & _CTA_RESERVED_WORDS or any(w in phrase for w in _CTA_RESERVED_WORDS):
            continue
        out.append(entry)
    return out


def find_subscribe_phrase_start(
    word_segments: List[dict],
    narration_duration: float,
) -> Optional[float]:
    """Return start time (seconds) when the subscribe CTA line begins."""
    from .keyword_popups import _find_phrase_window

    if not word_segments or narration_duration <= 0:
        return None
    tail_window = max(8.0, narration_duration * 0.25)
    for phrase in _SUBSCRIBE_PHRASES:
        win = _find_phrase_window(phrase, word_segments)
        if win is None:
            continue
        start, _end = win
        if phrase == "subscribe" and start < narration_duration - tail_window:
            continue
        return start
    return None


def build_subscribe_cta_popup(
    *,
    word_segments: List[dict],
    narration_duration: float,
    gif_path: Path,
    popup_width: int = DEFAULT_CTA_WIDTH,
    popup_y: int = DEFAULT_CTA_Y,
) -> Optional[PopupImage]:
    """Build a visual-only subscribe GIF popup aligned to speech."""
    path = Path(gif_path)
    if not path.is_file():
        print(f"Subscribe CTA: missing GIF at {path}")
        return None
    start = find_subscribe_phrase_start(word_segments, narration_duration)
    if start is None:
        print("Subscribe CTA: could not time subscribe phrase in narration.")
        return None
    try:
        gif_duration = ffprobe_duration_seconds(path)
    except Exception as exc:
        print(f"Subscribe CTA: could not read GIF duration: {exc}")
        return None
    end = min(start + max(0.2, gif_duration), narration_duration - 0.05)
    if end <= start:
        return None
    x = (1080 - popup_width) // 2
    print(f"Subscribe CTA: {path.name} @ {start:.2f}-{end:.2f}s")
    return PopupImage(
        path=path.resolve(),
        start_sec=start,
        end_sec=end,
        x=x,
        y=popup_y,
        width=popup_width,
        preserve_aspect=True,
        use_fade=True,
        play_sfx=False,
        sfx_path=None,
        chroma_key=DEFAULT_CHROMA_KEY or None,
    )


def filter_popups_for_subscribe_cta(
    popups: List[PopupImage],
    cta_start: float,
    cta_end: float,
    *,
    protect: PopupImage | None = None,
) -> List[PopupImage]:
    """Remove popups that overlap the subscribe CTA window."""
    kept: List[PopupImage] = []
    for popup in popups:
        if protect is not None and popup is protect:
            kept.append(popup)
            continue
        if popup.start_sec < cta_end and popup.end_sec > cta_start:
            print(
                f"Subscribe CTA: dropping overlapping popup "
                f"{popup.path.name} ({popup.start_sec:.2f}-{popup.end_sec:.2f}s)"
            )
            continue
        kept.append(popup)
    return kept


def apply_subscribe_cta(
    popups: List[PopupImage],
    *,
    word_segments: List[dict],
    narration_duration: float,
    gif_path: Path,
    protect: PopupImage | None = None,
) -> List[PopupImage]:
    """Filter overlapping popups and append subscribe GIF overlay last."""
    cta = build_subscribe_cta_popup(
        word_segments=word_segments,
        narration_duration=narration_duration,
        gif_path=gif_path,
    )
    if cta is None:
        return popups
    filtered = filter_popups_for_subscribe_cta(
        popups,
        cta.start_sec,
        cta.end_sec,
        protect=protect,
    )
    filtered.append(cta)
    return filtered
