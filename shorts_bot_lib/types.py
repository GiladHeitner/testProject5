"""Shared dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PopupImage:
    path: Path
    start_sec: float
    end_sec: float
    x: int
    y: int
    width: int
    play_sfx: bool = False
    use_fade: bool = True
    is_emoji: bool = False
    sfx_path: Path | None = None
    preserve_aspect: bool = False
    chroma_key: str | None = None  # e.g. "0x00FF00" for green-screen removal


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
