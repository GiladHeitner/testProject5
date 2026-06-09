"""Text manipulation helpers: script cleanup, time formatters, captions."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import List


def normalize_word_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


CURRENT_STORY_YEAR = 2026


def modernize_source_years(text: str, *, current_year: int = CURRENT_STORY_YEAR) -> str:
    """Rewrite stale year refs in Reddit source so scripts feel current."""
    if not text:
        return text
    out = text
    for old in range(2010, current_year):
        out = re.sub(rf"\b{old}\b", str(current_year), out)
    return out


def strip_speed_ramp_hyphens(script_text: str) -> str:
    """Remove --double-hyphen-- markers; keep paragraph breaks."""
    cleaned = re.sub(r"--([^-][^-]*?)--", r"\1", script_text)
    cleaned = re.sub(r"--+", "", cleaned)
    return cleaned.strip()


_PARALINGUISTIC_TAG_RE = re.compile(r"\[([^\]]+)\]", re.IGNORECASE)

# Maps script tags → Qwen CustomVoice/VoiceDesign instruct (not spoken words).
_TAG_INSTRUCT: dict[str, str] = {
    "sigh": "Speak with a frustrated sigh in your tone.",
    "breath": "Brief inhale, then speak naturally.",
    "slow breath": "Slow breath, then speak with tension.",
    "pause": "Brief dramatic pause, then continue.",
    "scoff": "Speak with dismissive scoffing energy.",
}


def strip_paralinguistic_tags(script_text: str) -> str:
    """Remove [sigh]-style director tags from display/subtitle text."""
    cleaned = _PARALINGUISTIC_TAG_RE.sub("", script_text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _tag_instruct(tag: str) -> str:
    key = (tag or "").strip().lower()
    if key in _TAG_INSTRUCT:
        return _TAG_INSTRUCT[key]
    for pattern, instruct in _TAG_INSTRUCT.items():
        if pattern in key or key in pattern:
            return instruct
    return ""


def split_script_for_qwen_delivery(script_text: str) -> list[tuple[str, str]]:
    """Split script at [tags]. Returns [(delivery_instruct, spoken_text), ...]."""
    text = script_text or ""
    if not _PARALINGUISTIC_TAG_RE.search(text):
        spoken = strip_paralinguistic_tags(strip_script_markup(text))
        return [("", spoken)] if spoken else [("", text.strip())]

    segments: list[tuple[str, str]] = []
    pending_instruct = ""
    pos = 0
    for match in _PARALINGUISTIC_TAG_RE.finditer(text):
        before = text[pos : match.start()].strip()
        if before:
            segments.append((pending_instruct, before))
            pending_instruct = ""
        tag_instruct = _tag_instruct(match.group(1))
        if tag_instruct:
            pending_instruct = tag_instruct
        pos = match.end()
    tail = text[pos:].strip()
    if tail:
        segments.append((pending_instruct, tail))
    if not segments:
        spoken = strip_paralinguistic_tags(strip_script_markup(text))
        return [("", spoken)] if spoken else [("", "")]
    return segments


def strip_script_markup(script_text: str) -> str:
    cleaned = strip_speed_ramp_hyphens(script_text)
    cleaned = (
        cleaned.replace("*", "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_CONTRACTIONS = {
    "I'M": "I'm",
    "I'VE": "I've",
    "I'LL": "I'll",
    "I'D": "I'd",
    "YOU'RE": "You're",
    "YOU'VE": "You've",
    "YOU'LL": "You'll",
    "WE'RE": "We're",
    "THEY'RE": "They're",
    "CAN'T": "can't",
    "WON'T": "won't",
    "DON'T": "don't",
    "DIDN'T": "didn't",
    "ISN'T": "isn't",
    "AREN'T": "aren't",
}


def _split_word_punctuation(word: str) -> tuple[str, str]:
    m = re.match(r"^([^\w]*)([\w']+)([^\w]*)$", word, flags=re.UNICODE)
    if not m:
        return word, ""
    return m.group(2), f"{m.group(1)}{m.group(3)}"


def normalize_word_for_tts(word: str) -> str:
    """Make one token safe for TTS (avoid spelling ALL CAPS letter-by-letter)."""
    core, punct = _split_word_punctuation(word)
    if not core:
        return word
    upper = core.upper()
    if upper in _CONTRACTIONS:
        return _CONTRACTIONS[upper] + punct
    if core == "A":
        return "a" + punct
    letters = [c for c in core if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return core.lower() + punct
    if core.isupper() and len(core) >= 2 and core.isalpha():
        return core.lower() + punct
    return word


def normalize_script_for_tts(script_text: str) -> str:
    """Normal case + contractions for voice cloning/TTS (keeps script file unchanged)."""
    cleaned = strip_script_markup(script_text)
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        words = [normalize_word_for_tts(w) for w in line.split()]
        lines.append(" ".join(words))
    return "\n".join(lines) if len(lines) > 1 else (lines[0] if lines else "")


# Qwen English TTS misreads many Arabic loanwords; respell without hyphens (hyphens add gaps).
_SLANG_TTS_SPELLINGS: tuple[tuple[str, str], ...] = (
    (r"\bharam\b", "huhraam"),
    (r"\bhalal\b", "huhlal"),
    (r"\bwallah\b", "walah"),
    (r"\bwallahi\b", "walahi"),
    (r"\byallah\b", "yallah"),
    (r"\byalla\b", "yalla"),
    (r"\bhabibi\b", "huhbeebee"),
    (r"\bhabibti\b", "huhbeebtee"),
    (r"\binshallah\b", "inshalluh"),
    (r"\binsallah\b", "inshalluh"),
    (r"\bmashallah\b", "mashalluh"),
    (r"\bmasallah\b", "mashalluh"),
    (r"\bdeen\b", "deen"),
    (r"\bdua\b", "dooah"),
    (r"\bsalah\b", "salaah"),
    (r"\bsalat\b", "salaat"),
    (r"\biftar\b", "iftaar"),
    (r"\bsuhoor\b", "sohoor"),
    (r"\bsuhur\b", "sohoor"),
    (r"\bameen\b", "ahmeen"),
    (r"\bsubhanallah\b", "subhannallah"),
    (r"\bastaghfirullah\b", "istaghfirullah"),
    (r"\bummah\b", "oommah"),
    (r"\bquran\b", "kooraan"),
    (r"\bhijab\b", "heejaab"),
    (r"\bjannah\b", "jannuh"),
)


# Words that make Qwen insert laughs — 1:1 swaps only (word-count must match for alignment).
_TTS_VOCAL_TRIGGER_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\blaughter\b", "mockery"),
    (r"\blaughing\b", "mocking"),
    (r"\blaughed\b", "mocked"),
    (r"\blaughs\b", "mocks"),
    (r"\blaugh\b", "mock"),
    (r"\bchuckling\b", "snickering"),
    (r"\bchuckled\b", "snickered"),
    (r"\bchuckle\b", "snicker"),
    (r"\bgiggling\b", "snickering"),
    (r"\bgiggled\b", "snickered"),
    (r"\bgiggle\b", "snicker"),
    (r"\bhaha+\b", ""),
    (r"\blol\b", ""),
    (r"\blmao\b", ""),
    (r"\bfunny\b", "absurd"),
    (r"\bjoke\b", "prank"),
    (r"\bjoking\b", "kidding"),
)


def neutralize_vocal_triggers_for_tts(script: str) -> str:
    out = script or ""
    for pattern, replacement in _TTS_VOCAL_TRIGGER_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def respell_muslim_slang_for_tts(script: str) -> str:
    out = script or ""
    for pattern, replacement in _SLANG_TTS_SPELLINGS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def format_omar_script(script: str) -> str:
    return respell_muslim_slang_for_tts(neutralize_vocal_triggers_for_tts(script))


def prepare_script_for_qwen_tts(script_text: str) -> str:
    """One flat line; strip punctuation that makes Qwen insert long pauses."""
    cleaned = strip_paralinguistic_tags(script_text)
    normalized = normalize_script_for_tts(cleaned)
    flat = re.sub(r"\s+", " ", normalized).strip()
    flat = flat.replace("—", " ").replace("–", " ")
    flat = re.sub(r'["""]', "", flat)
    flat = re.sub(r"[.!?…]+", " ", flat)
    flat = re.sub(r"[,;:]+", " ", flat)
    flat = re.sub(r"\s+", " ", flat).strip()
    return format_omar_script(flat)


def load_alignment_script(display_script: str, tts_script_path: Path | None = None) -> str:
    """Text that was (or would be) spoken — use for whisper/subtitle alignment."""
    if tts_script_path and tts_script_path.is_file():
        saved = tts_script_path.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    return prepare_script_for_qwen_tts(display_script)


def script_words_for_alignment(script_text: str) -> List[str]:
    cleaned = strip_paralinguistic_tags(strip_script_markup(script_text)).replace('"', " ")
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
