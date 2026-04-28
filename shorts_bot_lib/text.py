"""Text manipulation helpers: script cleanup, time formatters, captions."""

from __future__ import annotations

import math
import random
import re
from typing import List


def normalize_word_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def strip_script_markup(script_text: str) -> str:
    cleaned = (
        script_text.replace("--", " ")
        .replace("*", "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def script_words_for_alignment(script_text: str) -> List[str]:
    cleaned = strip_script_markup(script_text).replace('"', " ")
    return [word for word in cleaned.split() if word]


def strip_wrapping_quotes(text: str) -> str:
    cleaned = (text or "").strip()
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
    ]
    changed = True
    while changed and cleaned:
        changed = False
        for left, right in quote_pairs:
            if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
                changed = True
    return cleaned


def ass_ts(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    centiseconds = int(round((total_seconds - math.floor(total_seconds)) * 100))
    if centiseconds == 100:
        centiseconds = 0
        seconds += 1
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def to_srt_time(total_seconds: float) -> str:
    total_ms = int(round(total_seconds * 1000))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def to_ass_time(total_seconds: float) -> str:
    total_cs = int(round(total_seconds * 100))
    hours = total_cs // 360000
    rem = total_cs % 360000
    minutes = rem // 6000
    rem = rem % 6000
    seconds = rem // 100
    cs = rem % 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def sanitize_ass_text(text: str) -> str:
    return text.replace("\\", "").replace("{", "").replace("}", "")


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def maybe_upper(token: str, ratio: float) -> str:
    ratio = float(ratio or 0.0)
    if ratio <= 0:
        return token
    if ratio >= 1:
        return token.upper()
    if not any(ch.isalnum() for ch in token):
        return token
    return token.upper() if random.random() < ratio else token


def format_bold_srt(text: str) -> str:
    parts = text.split("*")
    for i in range(1, len(parts), 2):
        parts[i] = f"<b>{parts[i]}</b>"
    return "".join(parts)


def format_bold_ass(text: str) -> str:
    parts = text.split("*")
    for i in range(1, len(parts), 2):
        parts[i] = f"{{\\b1}}{parts[i]}{{\\b0}}"
    return "".join(parts)


def random_caps_text(text: str, probability: float = 0.3) -> str:
    rng = random.Random(text)
    words = text.split(" ")
    result = []
    for word in words:
        clean = "".join(c for c in word if c.isalpha())
        if clean and len(clean) >= 3 and rng.random() < probability:
            result.append(word.upper())
        else:
            result.append(word)
    return " ".join(result)


def format_caption_multiline(
    text: str,
    max_words_per_line: int = 2,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> str:
    """Shorts captions: max 2 lines, ~2 words per line, hard char cap."""
    raw_words = text.split()
    words: List[str] = []
    for w in raw_words:
        if len(w) <= max_chars_per_line:
            words.append(w)
        else:
            for i in range(0, len(w), max_chars_per_line):
                words.append(w[i : i + max_chars_per_line])
    if not words:
        return text

    lines: List[str] = []
    i = 0
    while i < len(words) and len(lines) < max_lines:
        line_words = words[i : i + max_words_per_line]
        line = " ".join(line_words)[:max_chars_per_line].rstrip()
        lines.append(line)
        i += max_words_per_line

    if i < len(words):
        last = lines[-1]
        if len(last) >= max_chars_per_line - 1:
            last = last[: max_chars_per_line - 1].rstrip()
        lines[-1] = (last + "\u2026").rstrip()

    return "\n".join(lines)


def estimate_line_width_px(line_text: str, font_size: int = 100) -> int:
    width = 0.0
    for ch in line_text:
        if ch == " ":
            width += 0.35
        elif ch in "ilI|.,'`!:":
            width += 0.40
        elif ch in "mwMW@#%&":
            width += 0.90
        elif ch.isupper():
            width += 0.75
        else:
            width += 0.60
    return int(width * font_size)


def text_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stop = {
        "the", "and", "that", "with", "this", "from", "your", "just",
        "were", "have", "what", "when", "they", "them", "then", "into",
        "over", "about", "there", "would", "could",
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def get_highlight_timestamps(script: str, words):
    """Return [(start_ms, end_ms)] for phrases wrapped in `--double hyphens--`."""
    phrases = [
        phrase.strip()
        for phrase in re.findall(r"--([^-][\s\S]*?[^-])--", script)
        if phrase.strip()
    ]
    if not phrases:
        return []

    normalized_words = [
        {
            "text": normalize_word_token(str(word.get("text", ""))),
            "start": float(word["start"]) * 1000.0,
            "end": float(word["end"]) * 1000.0,
        }
        for word in words
        if normalize_word_token(str(word.get("text", "")))
    ]

    highlights = []
    search_from = 0
    for phrase in phrases:
        phrase_words = [normalize_word_token(part) for part in phrase.split()]
        phrase_words = [part for part in phrase_words if part]
        if not phrase_words:
            continue

        for idx in range(search_from, len(normalized_words)):
            if normalized_words[idx]["text"] != phrase_words[0]:
                continue
            end_idx = idx
            match_ok = True
            for phrase_word in phrase_words[1:]:
                found = False
                for probe in range(end_idx + 1, min(end_idx + 8, len(normalized_words))):
                    if normalized_words[probe]["text"] == phrase_word:
                        end_idx = probe
                        found = True
                        break
                if not found:
                    match_ok = False
                    break
            if match_ok:
                highlights.append(
                    (normalized_words[idx]["start"], normalized_words[end_idx]["end"])
                )
                search_from = end_idx + 1
                break
    return highlights
