"""Emoji selection for captions and Apple-style emoji PNG download."""

from __future__ import annotations

import random
from pathlib import Path

import requests


def pick_emoji_for_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["secret", "code", "language", "hide"]):
        return "\U0001F928"
    if any(k in t for k in ["teacher", "school", "class"]):
        return "\U0001F610"
    if any(k in t for k in ["crush", "love", "girl", "boy"]):
        return "\U0001F64F"
    if any(k in t for k in ["caught", "freeze", "panic", "scared"]):
        return "\U0001F62D"
    if any(k in t for k in ["funny", "laugh", "joke"]):
        return "\U0001F610"
    return random.choice(["\U0001F64F", "\U0001F610", "\U0001F928", "\U0001F62D"])


def pick_non_repeating_emoji(text: str, used: set[str], last_emoji: str | None) -> str:
    pool = ["\U0001F64F", "\U0001F610", "\U0001F928", "\U0001F62D"]
    preferred = pick_emoji_for_text(text)
    if preferred not in used and preferred != last_emoji:
        return preferred

    candidates = [e for e in pool if e not in used and e != last_emoji]
    if candidates:
        return random.choice(candidates)

    candidates = [e for e in pool if e != last_emoji]
    if candidates:
        return random.choice(candidates)
    return preferred


def emoji_codepoint_path(emoji: str) -> str:
    codepoints = []
    for ch in emoji:
        cp = ord(ch)
        if cp == 0xFE0F:
            continue
        codepoints.append(f"{cp:x}")
    return "-".join(codepoints)


def _emoji_codepoint_path_with_fe0f(emoji: str) -> str:
    codepoints = []
    for ch in emoji:
        cp = ord(ch)
        codepoints.append(f"{cp:x}")
    return "-".join(codepoints)


def download_twemoji_png(emoji: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    code = emoji_codepoint_path(emoji)
    out_file = out_dir / f"{code}-ios.png"
    if out_file.exists():
        return out_file
    base = "https://cdn.jsdelivr.net/gh/iamcal/emoji-data@master/img-apple-64"
    urls = [f"{base}/{code}.png"]
    # Some emoji (notably ♥️/❤️) require FE0F in the filename.
    if "\ufe0f" in emoji:
        code2 = _emoji_codepoint_path_with_fe0f(emoji)
        if code2 != code:
            urls.append(f"{base}/{code2}.png")
    last_status = None
    for apple_url in urls:
        response = requests.get(apple_url, timeout=60)
        last_status = response.status_code
        if response.status_code == 200:
            out_file.write_bytes(response.content)
            return out_file
    raise RuntimeError(f"Failed to download emoji image for {emoji} (last={last_status}): {urls[-1]}")
