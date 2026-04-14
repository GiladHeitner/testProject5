import argparse
import difflib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))

try:
    from bing_image_downloader import downloader as bing_downloader
except Exception:
    bing_downloader = None


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


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


def respeed(segment: AudioSegment, speed_factor: float, output_frame_rate: int) -> AudioSegment:
    new_sample_rate = max(1000, int(segment.frame_rate * speed_factor))
    warped = segment._spawn(segment.raw_data, overrides={"frame_rate": new_sample_rate})
    return warped.set_frame_rate(output_frame_rate)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def get_dynamic_speed_factor(
    t_ms: float,
    highlights: List[Tuple[float, float]],
    ramp_ms: int,
    slow_factor: float,
    fast_factor: float,
) -> float:
    if not highlights:
        return fast_factor
    if ramp_ms <= 0:
        for start_ms, end_ms in highlights:
            if start_ms <= t_ms <= end_ms:
                return slow_factor
        return fast_factor

    best = fast_factor
    for start_ms, end_ms in highlights:
        if start_ms <= t_ms <= end_ms:
            best = min(best, slow_factor)
        elif start_ms - ramp_ms <= t_ms < start_ms:
            u = 1.0 - ((start_ms - t_ms) / ramp_ms)
            f = fast_factor + (slow_factor - fast_factor) * _smoothstep(u)
            best = min(best, f)
        elif end_ms < t_ms <= end_ms + ramp_ms:
            u = (t_ms - end_ms) / ramp_ms
            f = slow_factor + (fast_factor - slow_factor) * _smoothstep(u)
            best = min(best, f)
    return best


def smart_speed_ramp(
    input_path: Path,
    output_path: Path,
    interesting_segments: List[Tuple[float, float]],
    ramp_ms: int = 600,
    slow_factor: float = 0.60,
    fast_factor: float = 1.15,
    step_ms: int = 40,
    bitrate: str = "320k",
) -> None:
    audio = AudioSegment.from_file(input_path)
    padding = AudioSegment.silent(duration=200)
    audio = padding + audio
    frame_rate = audio.frame_rate
    total_ms = len(audio)
    out = AudioSegment.empty()
    pos = 0
    step_ms = max(5, int(step_ms))

    while pos < total_ms:
        end = min(pos + step_ms, total_ms)
        chunk = audio[pos:end]
        mid_time_ms = (pos + end) / 2.0
        speed = get_dynamic_speed_factor(
            mid_time_ms,
            interesting_segments,
            ramp_ms,
            slow_factor,
            fast_factor,
        )
        out += respeed(chunk, speed, frame_rate)
        pos = end

    out.export(output_path, format="mp3", bitrate=bitrate)


def _normalize_word_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def strip_script_markup(script_text: str) -> str:
    cleaned = (
        script_text.replace("--", " ")
        .replace("*", "")
        .replace("“", '"')
        .replace("”", '"')
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def script_words_for_alignment(script_text: str) -> List[str]:
    cleaned = strip_script_markup(script_text).replace('"', " ")
    return [word for word in cleaned.split() if word]


def get_highlight_timestamps(script: str, words: List[dict]) -> List[Tuple[float, float]]:
    phrases = [
        phrase.strip()
        for phrase in re.findall(r'--([^-][\s\S]*?[^-])--', script)
        if phrase.strip()
    ]
    if not phrases:
        return []

    normalized_words = [
        {
            "text": _normalize_word_token(str(word.get("text", ""))),
            "start": float(word["start"]) * 1000.0,
            "end": float(word["end"]) * 1000.0,
        }
        for word in words
        if _normalize_word_token(str(word.get("text", "")))
    ]

    highlights: List[Tuple[float, float]] = []
    search_from = 0
    for phrase in phrases:
        phrase_words = [_normalize_word_token(part) for part in phrase.split()]
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


def run(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def transcribe_words(
    audio_path: Path,
    model_name: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
) -> List[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing `faster-whisper`. Install it with `pip install faster-whisper`."
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
    )

    out: List[Word] = []
    for seg in segments:
        if not getattr(seg, "words", None):
            continue
        for word in seg.words:
            text = (word.word or "").strip()
            if not text:
                continue
            out.append(
                Word(
                    text=text,
                    start=float(word.start),
                    end=float(word.end),
                )
            )
    if not out:
        raise RuntimeError("No words produced from transcription.")
    return out


def _ass_ts(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    centiseconds = int(round((total_seconds - math.floor(total_seconds)) * 100))
    if centiseconds == 100:
        centiseconds = 0
        seconds += 1
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _sanitize_ass_text(text: str) -> str:
    return text.replace("\\", "").replace("{", "").replace("}", "")


def _maybe_upper(token: str, ratio: float) -> str:
    ratio = float(ratio or 0.0)
    if ratio <= 0:
        return token
    if ratio >= 1:
        return token.upper()
    if not any(ch.isalnum() for ch in token):
        return token
    return token.upper() if random.random() < ratio else token


def write_karaoke_block_ass(
    words: Sequence[Word],
    out_ass: Path,
    *,
    words_per_block: int = 3,
    uppercase_ratio: float = 0.15,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    y: int = 520,
    font: str = "Gibson",
    font_size: int = 100,
) -> None:
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    x = play_res_x // 2
    words_per_block = max(2, int(words_per_block))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,7,0,8,180,180,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    override = (
        rf"{{\an8\pos({x},{y})"
        r"\bord7\shad0\3c&H000000&\4c&H000000&"
        r"\1c&HFFFFFF&}"
    )

    lines: List[str] = [header]
    idx = 0
    while idx < len(words):
        chunk = words[idx : idx + words_per_block]
        if not chunk:
            break

        chunk_tokens = [
            _maybe_upper(_sanitize_ass_text(word.text.strip()), uppercase_ratio)
            for word in chunk
        ]
        for current_idx, word in enumerate(chunk):
            start = _ass_ts(word.start)
            end = _ass_ts(max(word.end, word.start + 0.05))
            parts: List[str] = []
            for token_idx, token in enumerate(chunk_tokens):
                if token_idx == current_idx:
                    parts.append(r"{\1c&H00FFFF&}" + token + r"{\1c&HFFFFFF&}")
                else:
                    parts.append(token)
            lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{override}{' '.join(parts)}\n"
            )

        idx += words_per_block

    out_ass.write_text("".join(lines), encoding="utf-8")


def ffmpeg_has_subtitles_filter() -> bool:
    result = subprocess.run(
        "ffmpeg -hide_banner -filters",
        shell=True,
        capture_output=True,
        text=True,
    )
    return " subtitles " in result.stdout


def ffprobe_duration_seconds(file_path: Path) -> float:
    cmd = (
        f"ffprobe -v error -show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(file_path))}"
    )
    output = run(cmd)
    return float(output)


def to_srt_time(total_seconds: float) -> str:
    total_ms = int(round(total_seconds * 1000))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def format_bold_srt(text: str) -> str:
    parts = text.split('*')
    for i in range(1, len(parts), 2):
        parts[i] = f"<b>{parts[i]}</b>"
    return "".join(parts)

def write_srt_from_segments(segments: List[dict], out_path: Path) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        text = str(seg.get("raw_text") or seg["text"]).strip()
        text = format_bold_srt(text)
        lines.append(str(idx))
        lines.append(f"{to_srt_time(start_s)} --> {to_srt_time(end_s)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def to_ass_time(total_seconds: float) -> str:
    total_cs = int(round(total_seconds * 100))
    hours = total_cs // 360000
    rem = total_cs % 360000
    minutes = rem // 6000
    rem = rem % 6000
    seconds = rem // 100
    cs = rem % 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def format_caption_multiline(
    text: str,
    max_words_per_line: int = 2,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> str:
    """
    Shorts captions:
    - max 2 lines
    - ~2 words per line
    - hard char cap so text never falls off screen
    """
    raw_words = text.split()
    words: List[str] = []
    # Split long tokens so a single word never overflows.
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
        lines[-1] = (last + "…").rstrip()

    return "\n".join(lines)


def estimate_line_width_px(line_text: str, font_size: int = 100) -> int:
    # Tune estimate for bold, uppercase-heavy caption styling.
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


def split_caption_chunks(segments: List[dict], words_per_chunk: int = 1) -> List[dict]:
    """
    Turn uneven Whisper segments into steady Shorts caption beats.
    """
    chunks: List[dict] = []
    for seg in segments:
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        raw_text = str(seg.get("raw_text") or seg.get("text") or "").strip()
        words = raw_text.split()
        if not words:
            continue
        if end_s <= start_s:
            end_s = start_s + 0.25
        pieces = [
            " ".join(words[i : i + words_per_chunk]).strip()
            for i in range(0, len(words), words_per_chunk)
            if words[i : i + words_per_chunk]
        ]
        dur = max(0.2, end_s - start_s)
        chunk_duration = dur / len(pieces)
        for idx, piece in enumerate(pieces):
            c_start = start_s + (idx * chunk_duration)
            c_end = end_s if idx == len(pieces) - 1 else (c_start + chunk_duration)
            emoji = pick_emoji_for_text(piece)
            caption_text = format_caption_multiline(
                piece, max_words_per_line=1, max_chars_per_line=16, max_lines=2
            )
            lines = [ln for ln in caption_text.split("\n") if ln.strip()]
            last_line = lines[-1] if lines else piece
            chunks.append(
                {
                    "start": c_start,
                    "end": c_end,
                    "raw_text": piece,
                    "caption_text": caption_text,
                    "emoji": emoji,
                    "line_count": max(1, len(lines)),
                    "last_line_chars": len(last_line),
                    "last_line_text": last_line,
                }
            )
    return chunks


def format_bold_ass(text: str) -> str:
    parts = text.split('*')
    for i in range(1, len(parts), 2):
        parts[i] = f"{{\\b1}}{parts[i]}{{\\b0}}"
    return "".join(parts)

def random_caps_text(text: str, probability: float = 0.3) -> str:
    rng = random.Random(text)
    words = text.split(' ')
    result = []
    for word in words:
        clean = ''.join(c for c in word if c.isalpha())
        if clean and len(clean) >= 3 and rng.random() < probability:
            result.append(word.upper())
        else:
            result.append(word)
    return ' '.join(result)


def write_ass_from_segments(segments: List[dict], out_path: Path) -> None:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 1",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Default,Gibson,100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,7,0,8,180,180,520,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    lines = header[:]
    for seg in segments:
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        raw_text = str(seg.get("raw_text") or seg.get("text") or "").strip()
        if not raw_text:
            continue
        if end_s <= start_s:
            end_s = start_s + 0.25
        text = escape_ass_text(
            format_caption_multiline(
                random_caps_text(raw_text), max_words_per_line=1, max_chars_per_line=16, max_lines=2
            )
        )
        text = format_bold_ass(text)
        lines.append(
            f"Dialogue: 0,{to_ass_time(start_s)},{to_ass_time(end_s)},Default,,0,0,0,,{text}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def read_srt_segments(srt_path: Path) -> List[dict]:
    raw = srt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    segments: List[dict] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_idx]:
            continue
        t_start, t_end = [part.strip() for part in lines[time_line_idx].split("-->")]
        def parse_srt_time(t: str) -> float:
            hh, mm, rest = t.split(":")
            ss, ms = rest.split(",")
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
        start_s = parse_srt_time(t_start)
        end_s = parse_srt_time(t_end)
        text_lines = lines[time_line_idx + 1 :]
        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": start_s, "end": end_s, "text": text})
    return segments


def remap_script_to_reference_timing(
    script_text: str, reference_segments: List[dict], max_words_per_segment: int = 5
) -> List[dict]:
    words = script_text.strip().split()
    if not words:
        return []
    if not reference_segments:
        return []

    # Preserve original subtitle rhythm by mirroring per-segment word density.
    ref_counts = [max(1, len(str(seg.get("text", "")).split())) for seg in reference_segments]
    out: List[dict] = []
    cursor = 0
    for i, ref in enumerate(reference_segments):
        start = float(ref["start"])
        end = float(ref["end"])
        if end <= start:
            end = start + 0.2

        if cursor >= len(words):
            break

        remaining_segments = len(reference_segments) - i
        remaining_words = len(words) - cursor
        remaining_ref = sum(ref_counts[i:])
        if i == len(reference_segments) - 1:
            take = remaining_words
        else:
            ideal = (ref_counts[i] / max(1, remaining_ref)) * remaining_words
            take = max(1, int(round(ideal)))
            # Keep at least one word for each remaining segment.
            take = min(take, max(1, remaining_words - (remaining_segments - 1)))
        # Soft cap to reduce giant blocks, but allow overflow for last segment.
        if i != len(reference_segments) - 1:
            take = min(take, max_words_per_segment + 2)
        chunk = words[cursor : cursor + take]
        cursor += take

        out.append({"start": start, "end": end, "text": " ".join(chunk)})

    if cursor < len(words) and out:
        out[-1]["text"] = str(out[-1]["text"]) + " " + " ".join(words[cursor:])
    return out


def pick_emoji_for_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["secret", "code", "language", "hide"]):
        return "🤨"
    if any(k in t for k in ["teacher", "school", "class"]):
        return "😐"
    if any(k in t for k in ["crush", "love", "girl", "boy"]):
        return "🙏"
    if any(k in t for k in ["caught", "freeze", "panic", "scared"]):
        return "😭"
    if any(k in t for k in ["funny", "laugh", "joke"]):
        return "😐"
    return random.choice(["🙏", "😐", "🤨", "😭"])


def pick_non_repeating_emoji(text: str, used: set[str], last_emoji: str | None) -> str:
    pool = ["🙏", "😐", "🤨", "😭"]
    preferred = pick_emoji_for_text(text)
    if preferred not in used and preferred != last_emoji:
        return preferred

    candidates = [e for e in pool if e not in used and e != last_emoji]
    if candidates:
        return random.choice(candidates)

    # If we exhaust the pool, avoid immediate repeats.
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


def download_twemoji_png(emoji: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    code = emoji_codepoint_path(emoji)
    out_file = out_dir / f"{code}-ios.png"
    if out_file.exists():
        return out_file
    apple_url = (
        f"https://cdn.jsdelivr.net/gh/iamcal/emoji-data@master/img-apple-64/{code}.png"
    )
    response = requests.get(apple_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download emoji image for {emoji}: {apple_url}")
    out_file.write_bytes(response.content)
    return out_file


def pick_random_file(folder: Path, extensions: List[str]) -> Path:
    items = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in set(extensions)
    ]
    if not items:
        raise FileNotFoundError(f"No matching files found in: {folder}")
    return random.choice(items)


def print_progress(step: int, total: int, label: str) -> None:
    width = 24
    filled = int(width * step / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * step / total)
    print(f"[{bar}] {percent:3d}% ({step}/{total}) {label}")


def generate_script(client: OpenAI, target_words: int, topic: str = "") -> str:
    low = max(55, int(target_words * 0.85))
    high = int(target_words * 1.15)
    topic_line = topic.strip() or "a relatable personal story about a social situation"
    prompt = f"""
Write a YouTube Shorts storytime script about: {topic_line} Cater the story to a middle school audience.

Use the EXACT same writing style, pacing, tone, and length as this example script:

---
My crush kissed me. --No, I'm not kidding--. Okay, quick recap. Last time she hugged me after our walk and said, "See you tomorrow, --singer boy.--" That was 3 days ago, and we haven't stopped talking since. She texted me till midnight, walking home together, sharing songs. Somehow, she's already part of my everyday. First off, she posted a video of me singing on her story with a tiny red heart --right over my face.-- Yeah, my face. My friends went feral in the group chat. Yesterday at lunch, we were sitting together when two idiots from my class came up trying to be funny. One goes, "Hey, your girlfriend got room for one more." Before I could even answer, she smiles and goes, "Maybe ask your boyfriend first. He looks like the jealous type." --Bro, they froze.-- One muttered something and they both left. I was just sitting there laughing and half shocked. She looked at me and said, "What? I was saving you." Later that day, she texted, "You want to come over and study?" Of course, I said yes. We ended up doing way more laughing than studying. Sitting close, sharing snacks, making dumb jokes like always. Somehow, we ended up --half cuddled on the couch.-- The room quiet except for her playlist, softly playing. Then, I dropped my pen. I leaned down to grab it, and when I sat back up, --she was already looking at me.-- For a second, neither of us said anything. Then, she leaned in --and kissed me.-- Quick, soft. Then, she pulled back, met my eyes, smiled, and kissed me again. --Slower this time.-- It was honestly incredible. Like nothing else mattered for a second. I --don't even remember-- how I got home that night.
---

STYLE RULES (match these exactly):
- have an antagonist and make the viewers want me an underdog to win
- End the video by saying subscribe before I get banned!
- Must rehook the person throughout the video
- The video is going to be posted on shorts so it auto loops so make sure the last line loops into the first line without repeating the first line it should sound unfinished unless you hear the first line again
- DYNAMIC SPEED RAMPS: You MUST wrap 6 to 8 crucial action beats, plot twists, or heavy punchlines in double hyphens to trigger a slow-motion audio effect (e.g., "--my iPad went flying--"). 
- FOCUS ON IMPACT: Do NOT wrap descriptive fluff or narrator asides (like "slow motion, like a movie"). Only wrap the actual event or the most shocking part of the sentence.
- RAMP LENGTH: Only wrap short phrases of 2 to 5 words. Never wrap an entire sentence.
- RAMP SPACING: NEVER put hyphenated phrases back-to-back. You must space them out evenly throughout the script so the audio has time to return to normal speed between drops.
- End the video by saying subscribe before I get --banned--!
- Hook must have a high chance of being used in the title
- First-person, past tense, told like you're talking to a best friend
- Fast-paced. Short punchy sentences. No filler.
- Use quoted dialogue to bring scenes to life
- Build tension and emotion beat by beat
- End on a high — a moment that makes the viewer feel something
- TARGET WORD COUNT: 85–115 words
- Output plain dialogue only. No stage directions, no emojis, no section labels.
- Research the topic to write authentically and specifically

Write ONE complete script now.
"""
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.7,
    )
    return resp.output_text.strip()


TITLE_PROMPT = """Create a viral YouTube Shorts TITLE for this story.

Rules:
- 55–80 characters
- curiosity-driven
- use 1–2 emojis like 😭🙏
- include #shorts and 1–2 relevant hashtags

Output ONLY the title, nothing else."""

DESCRIPTION_PROMPT = """Create a viral YouTube Shorts DESCRIPTION for this story.

Rules:
- 1 short line summarizing the story
- conversational tone
- encourage engagement
- include relevant hashtags
- end with a copyright credit section exactly like this:

Gameplay Credit: Dope Gameplays
Roblox Parkour Gameplay No Copyright | Roblox Gameplay No Copyright | 33
https://www.youtube.com/shorts/8Vo-3dhM7lM
Licensed under Creative Commons Attribution.

Output ONLY the description, nothing else."""


def strip_wrapping_quotes(text: str) -> str:
    cleaned = (text or "").strip()
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]
    changed = True
    while changed and cleaned:
        changed = False
        for left, right in quote_pairs:
            if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
                changed = True
    return cleaned


def generate_metadata(client: OpenAI, script: str, include_description: bool = True) -> Tuple[str, str]:
    def _call(prompt: str) -> str:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{prompt}\n\nStory:\n{script}"}],
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

    title = strip_wrapping_quotes(_call(TITLE_PROMPT))
    description = _call(DESCRIPTION_PROMPT) if include_description else ""

    if not title:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook or "Crazy Story You Won't Believe"
    if include_description and not description:
        description = "Subscribe for more storytime shorts!\n#shorts #storytime"
    return title, description


def _resolve_adam_cloner_script(project_root: Path, configured_script: str) -> Path:
    if configured_script:
        script_path = Path(configured_script).expanduser().resolve()
    else:
        candidates = [
            project_root / "VoiceCloner" / "run_clone.sh",
            project_root / "AdamVoice" / "VoiceCloner" / "run_clone.sh",
            project_root / "adamvoice" / "VoiceCloner" / "run_clone.sh",
        ]
        script_path = next((p for p in candidates if p.exists()), Path(""))
    if not script_path or not script_path.exists():
        raise RuntimeError(
            "Adam cloner script not found. Expected `VoiceCloner/run_clone.sh`."
        )
    return script_path


def get_whisper_word_timestamps(client: OpenAI, audio_path: Path, script_text: str = "") -> List[dict]:
    """Get exact word-level timestamps from Whisper for subtitle sync."""
    whisper_prompt = build_whisper_prompt(script_text) if script_text else ""
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=whisper_prompt or None,
        )
    words = []
    if hasattr(transcript, "words") and transcript.words:
        for w in transcript.words:
            if isinstance(w, dict):
                words.append({"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
            else:
                words.append({"text": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    elif isinstance(transcript, dict) and "words" in transcript:
        for w in transcript["words"]:
            words.append({"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words


def generate_voiceover_from_cloner_script(
    script_text: str, out_audio_path: Path, project_root: Path, cloner_script: str
) -> None:
    run_clone_path = _resolve_adam_cloner_script(project_root, cloner_script)
    tmp_wav = out_audio_path.with_suffix(".adam_tmp.wav")
    env = os.environ.copy()
    env["TEXT"] = strip_script_markup(script_text)
    env["OUTPUT"] = str(tmp_wav)
    env["USE_BATCH"] = "false"
    result = subprocess.run(
        ["bash", str(run_clone_path)],
        cwd=str(run_clone_path.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Adam run_clone.sh failed.\n"
            f"stdout:\n{result.stdout.strip()}\n\nstderr:\n{result.stderr.strip()}"
        )
    transformed = tmp_wav.with_name(tmp_wav.stem + "_sp.wav")
    source_audio = transformed if transformed.exists() else tmp_wav
    if not source_audio.exists():
        raise RuntimeError("Adam cloner did not produce output audio.")
    run(
        f"ffmpeg -y -i {shlex.quote(str(source_audio))} "
        f"-c:a libmp3lame -q:a 2 {shlex.quote(str(out_audio_path))}"
    )
    if tmp_wav.exists():
        tmp_wav.unlink()
    if transformed.exists():
        transformed.unlink()


def generate_voiceover_openai_tts(client: OpenAI, script_text: str, out_audio_path: Path) -> None:
    text = strip_script_markup(script_text)
    if not text:
        raise RuntimeError("Empty script text; cannot generate voiceover.")
    models_to_try = ["gpt-4o-mini-tts", "tts-1"]
    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            audio = client.audio.speech.create(
                model=model,
                voice="alloy",
                input=text,
                format="mp3",
            )
            out_audio_path.write_bytes(audio.read())
            return
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"OpenAI TTS failed: {last_exc}")


def build_whisper_prompt(script_text: str) -> str:
    cleaned = strip_script_markup(script_text)
    words = cleaned.split()
    return " ".join(words[:244])


def build_caption_chunks_from_word_timestamps(
    script_text: str,
    word_timestamps: List[dict],
    words_per_chunk: int = 1,
) -> List[dict]:
    clean_script = script_words_for_alignment(script_text)
    if not clean_script or not word_timestamps:
        return []

    def norm(token: str) -> str:
        return re.sub(r"[^a-z0-9]", "", token.lower())

    script_norm = [norm(word) for word in clean_script]
    whisper_norm = [norm(str(word.get("text", ""))) for word in word_timestamps]
    matcher = difflib.SequenceMatcher(None, script_norm, whisper_norm)
    aligned_words: List[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aligned_words.append(
                    {
                        "text": clean_script[i],
                        "start": float(word_timestamps[j]["start"]),
                        "end": float(word_timestamps[j]["end"]),
                    }
                )
            continue

        block_script = clean_script[i1:i2]
        block_whisper = word_timestamps[j1:j2]
        if not block_script:
            continue

        if block_whisper:
            t_start = float(block_whisper[0]["start"])
            t_end = float(block_whisper[-1]["end"])
        else:
            t_start = float(aligned_words[-1]["end"]) if aligned_words else 0.0
            t_end = t_start + (0.25 * len(block_script))

        duration = max(0.1, t_end - t_start)
        time_per_word = duration / len(block_script)
        for idx, word in enumerate(block_script):
            aligned_words.append(
                {
                    "text": word,
                    "start": t_start + (idx * time_per_word),
                    "end": t_start + ((idx + 1) * time_per_word),
                }
            )

    chunks: List[dict] = []
    for i in range(0, len(aligned_words), words_per_chunk):
        chunk = aligned_words[i : i + words_per_chunk]
        if not chunk:
            continue
        start_s = float(chunk[0]["start"])
        end_s = float(chunk[-1]["end"])
        if end_s <= start_s:
            end_s = start_s + 0.2
        raw_text = " ".join(str(word["text"]) for word in chunk).strip()
        if raw_text:
            chunks.append({"start": start_s, "end": end_s, "raw_text": raw_text})
    return chunks


def transcribe_audio_to_srt(
    client: OpenAI | None,
    audio_path: Path,
    out_srt_path: Path,
    script_text: str = "",
    reference_segments: List[dict] | None = None,
) -> Tuple[List[dict], List[dict]]:
    del client, script_text, reference_segments

    words = transcribe_words(audio_path)
    word_segments = [
        {
            "start": float(word.start),
            "end": float(max(word.end, word.start + 0.05)),
            "raw_text": word.text,
            "text": word.text,
        }
        for word in words
        if word.text.strip()
    ]
    if not word_segments:
        raise RuntimeError("No transcription segments returned; cannot build subtitles.")

    if out_srt_path.suffix.lower() == ".ass":
        write_karaoke_block_ass(words, out_srt_path)
        write_srt_from_segments(word_segments, out_srt_path.with_suffix(".srt"))
    else:
        write_srt_from_segments(word_segments, out_srt_path)
    return [], word_segments


def maybe_generate_images(
    client: OpenAI, script: str, images_dir: Path, count: int, generate_images: bool
) -> None:
    if not generate_images:
        return
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_seed = (
        "Generate visual prompts for still images that match moments in this story. "
        "Return a JSON array of short prompts only, no explanations."
    )
    prompt_resp = client.responses.create(
        model="gpt-4.1-mini",
        input=f"{prompt_seed}\n\nStory:\n{script}\n\nNeed exactly {count} prompts.",
    )
    raw = prompt_resp.output_text.strip()
    try:
        prompts = json.loads(raw)
    except json.JSONDecodeError:
        prompts = [f"School hallway dramatic scene, cinematic, no text"] * count
    if not isinstance(prompts, list):
        prompts = [f"School classroom cinematic still, no text"] * count
    prompts = prompts[:count]
    while len(prompts) < count:
        prompts.append("School drama scene, cinematic still image, no text")

    for i, prompt in enumerate(prompts, start=1):
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"{prompt}. Vertical composition, high contrast, no text overlays.",
            size="1024x1536",
        )
        image_b64 = img.data[0].b64_json
        image_bytes = __import__("base64").b64decode(image_b64)
        out_path = images_dir / f"generated_{i}.png"
        out_path.write_bytes(image_bytes)


def choose_popup_images(
    images_dir: Path,
    video_duration: float,
    count: int = 3,
    popup_duration: float = 1.8,
) -> List[PopupImage]:
    candidates = [
        p
        for p in images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not candidates:
        return []

    chosen_files = random.sample(candidates, k=min(count, len(candidates)))
    popups = []
    for img in chosen_files:
        start = random.uniform(2.0, max(2.1, video_duration - 4.5))
        end = min(video_duration - 0.2, start + max(0.4, float(popup_duration)))
        width = 700
        x = (1080 - width) // 2
        y = 730
        popups.append(PopupImage(path=img, start_sec=start, end_sec=end, x=x, y=y, width=width))
    return sorted(popups, key=lambda p: p.start_sec)


def text_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stop = {
        "the",
        "and",
        "that",
        "with",
        "this",
        "from",
        "your",
        "just",
        "were",
        "have",
        "what",
        "when",
        "they",
        "them",
        "then",
        "into",
        "over",
        "about",
        "there",
        "would",
        "could",
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def maybe_download_story_images(
    story_images_dir: Path,
    story_text: str,
    client: OpenAI | None = None,
    min_count: int = 18,
) -> None:
    return


def fixed_image_times(
    video_duration: float, interval_seconds: float = 2.5, first_offset_seconds: float = 1.2
) -> List[float]:
    times: List[float] = []
    t = max(0.2, float(first_offset_seconds))
    interval = max(0.5, float(interval_seconds))
    while t < max(0.8, video_duration - 0.2):
        times.append(t)
        t += interval
    return times


def choose_story_related_popups(
    story_images_dir: Path,
    story_text: str,
    video_duration: float,
    subtitle_segments: List[dict] | None = None,
    planned_times: List[float] | None = None,
    min_gap: float = 3.0,
    max_gap: float = 6.0,
    popup_duration: float = 1.8,
) -> List[PopupImage]:
    candidates = [
        p
        for p in story_images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not candidates:
        return []

    story_keys = text_keywords(story_text)
    scored: List[tuple[int, float, Path]] = []
    for p in candidates:
        name_blob = (
            f"{p.parent.name} {p.stem}"
            .replace("_", " ")
            .replace("-", " ")
        )
        file_keys = text_keywords(name_blob)
        overlap = len(story_keys.intersection(file_keys))
        tie_break = random.random()
        scored.append((overlap, tie_break, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked_images = [item[2] for item in scored]
    if not ranked_images:
        return []

    del subtitle_segments

    # Use fixed planned beats so popup timing is independent from subtitle timing.
    popups: List[PopupImage] = []
    beat_times = planned_times[:] if planned_times else []
    if not beat_times:
        t = 0.8
        while t < max(1.0, video_duration - 0.3):
            beat_times.append(t)
            t += random.uniform(min_gap, max_gap)

    for idx, beat in enumerate(beat_times):
        start = max(0.0, float(beat))
        end = min(video_duration - 0.05, start + max(1.0, float(popup_duration)))
        if end <= start:
            continue
        width = 700
        x = (1080 - width) // 2
        y = 860
        img = ranked_images[idx % len(ranked_images)]
        popups.append(
            PopupImage(
                path=img,
                start_sec=start,
                end_sec=end,
                x=x,
                y=y,
                width=width,
                play_sfx=True,
            )
        )
    return sorted(popups, key=lambda p: p.start_sec)


def build_emoji_overlays(subtitle_segments: List[dict], emoji_dir: Path) -> List[PopupImage]:
    overlays = []
    used_emojis: set[str] = set()
    last_emoji: str | None = None
    font_size = 100
    line_step = int(font_size * 1.05)
    top_margin = 520
    caption_center_x = 540
    emoji_size = 105
    padding = 25
    screen_padding = 56
    safety_padding = 90
    # Match ASS caption chunking exactly so emoji placement stays aligned.
    for chunk in split_caption_chunks(subtitle_segments, words_per_chunk=1):
        emoji = pick_non_repeating_emoji(str(chunk["raw_text"]), used_emojis, last_emoji)
        used_emojis.add(emoji)
        last_emoji = emoji
        img_path = download_twemoji_png(emoji, emoji_dir)
        display_text = format_caption_multiline(
            random_caps_text(str(chunk.get("raw_text", "")).strip()),
            max_words_per_line=1,
            max_chars_per_line=16,
            max_lines=2,
        )
        display_lines = [ln for ln in display_text.split("\n") if ln.strip()]
        line_count = max(1, len(display_lines))
        last_line_text = display_lines[-1] if display_lines else str(chunk.get("raw_text", "")).strip()
        last_line_w = estimate_line_width_px(last_line_text, font_size=font_size)
        line_left_x = int(caption_center_x - (last_line_w / 2.0))
        # Rely on explicit padding instead of pulling into the final word.
        right_edge = line_left_x + last_line_w
        trailing_char = last_line_text[-1] if last_line_text else ""
        punct_extra = int(font_size * 0.12) if trailing_char in ".!?,;:" else 0
        x_inline = right_edge + padding + punct_extra
        max_x = 1080 - emoji_size - screen_padding
        overflow = x_inline > (max_x - safety_padding)
        # Keep emoji on line 2 when caption has exactly two lines.
        force_under = line_count >= 3

        if overflow or force_under:
            # If inline placement exceeds bounds, move emoji to a new line.
            x_pos = caption_center_x - (emoji_size // 2)
            y_pos = top_margin + (line_count * line_step) + 6
        else:
            x_pos = max(screen_padding, x_inline)
            # Align with the current last subtitle line.
            y_pos = top_margin + ((line_count - 1) * line_step) - 2
        overlays.append(
            PopupImage(
                path=img_path,
                start_sec=float(chunk["start"]),
                end_sec=float(chunk["end"]),
                x=x_pos,
                y=y_pos,
                width=emoji_size,
                use_fade=False,
                is_emoji=True,
            )
        )
    return overlays


def build_filter_complex(
    subtitle_path: Path,
    start_time: float,
    total_duration: float,
    popups: List[PopupImage],
    burn_subtitles: bool,
    source_top_crop: int = 96,
) -> str:
    chains = []
    top_crop = max(0, int(source_top_crop))
    crop_prefix = f"crop=in_w:in_h-{top_crop}:0:{top_crop}," if top_crop > 0 else ""
    chains.append(
        "[0:v]"
        f"{crop_prefix}"
        f"trim=start={start_time:.3f}:duration={total_duration * 2.0:.3f},"
        "setpts=PTS-STARTPTS,"
        "setpts=0.5*PTS,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "fps=60"
        "[v0]"
    )

    for i, popup in enumerate(popups, start=0):
        popup_len = max(0.2, popup.end_sec - popup.start_sec)
        fade_dur = min(0.25, popup_len * 0.35)
        fade_out_start = max(popup.start_sec, popup.end_sec - fade_dur)
        chains.append(
            (
                f"[{i + 2}:v]"
                + (
                    f"scale={popup.width}:-1,"
                    if popup.is_emoji
                    else f"scale={popup.width}:{popup.width}:force_original_aspect_ratio=increase,crop={popup.width}:{popup.width},"
                )
                + 
                f"format=rgba,"
                f"fade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f}:alpha=1"
                f"[img{i}]"
            )
            if popup.use_fade
            else (
                f"[{i + 2}:v]"
                + (
                    f"scale={popup.width}:-1,"
                    if popup.is_emoji
                    else f"scale={popup.width}:{popup.width}:force_original_aspect_ratio=increase,crop={popup.width}:{popup.width},"
                )
                +
                f"format=rgba"
                f"[img{i}]"
            )
        )

    current = "v0"
    for i, popup in enumerate(popups, start=0):
        next_label = f"v{i + 1}"
        chains.append(
            f"[{current}][img{i}]overlay="
            f"x={popup.x}:y={popup.y}:"
            f"enable='between(t,{popup.start_sec:.3f},{popup.end_sec:.3f})'"
            f"[{next_label}]"
        )
        current = next_label

    if burn_subtitles:
        escaped_srt = (
            str(subtitle_path)
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
        )
        chains.append(f"[{current}]subtitles=filename={escaped_srt}[vout]")
    else:
        chains.append(f"[{current}]null[vout]")
    return ";".join(chains)


def build_popup_sfx_audio_chain(
    popups: List[PopupImage],
    sfx_input_index: int,
    sfx_trim_seconds: float = 1.4,
    sfx_speed: float = 1.25,
    sfx_volume: float = 0.2,
) -> str:
    sfx_events = [p for p in popups if p.play_sfx]
    if not sfx_events:
        return ""

    chains = ["[1:a]aresample=44100,volume=1.0[abase]"]
    for i, popup in enumerate(sfx_events):
        delay_ms = max(0, int(popup.start_sec * 1000))
        trim_s = max(0.15, float(sfx_trim_seconds))
        speed = min(2.0, max(0.5, float(sfx_speed)))
        vol = max(0.0, float(sfx_volume))
        chains.append(
            f"[{sfx_input_index}:a]"
            f"atrim=0:{trim_s:.2f},asetpts=N/SR/TB,atempo={speed:.2f},volume={vol:.2f},adelay={delay_ms}|{delay_ms}"
            f"[boom{i}]"
        )
    mix_inputs = "[abase]" + "".join(f"[boom{i}]" for i in range(len(sfx_events)))
    chains.append(
        f"{mix_inputs}amix=inputs={1 + len(sfx_events)}:duration=first:normalize=0:dropout_transition=0[aout]"
    )
    return ";".join(chains)


def compose_video(
    gameplay_path: Path,
    narration_path: Path,
    srt_path: Path,
    popup_images: List[PopupImage],
    out_video_path: Path,
    duration_seconds: float | None = None,
    burn_subtitles: bool = True,
    popup_sfx_path: Path | None = None,
    popup_sfx_trim_seconds: float = 1.4,
    popup_sfx_speed: float = 1.25,
    popup_sfx_volume: float = 0.18,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.08,
    source_top_crop: int = 96,
) -> float:
    narration_duration = ffprobe_duration_seconds(narration_path)
    target_duration = duration_seconds if duration_seconds is not None else narration_duration
    gameplay_duration = ffprobe_duration_seconds(gameplay_path)
    required_source_duration = target_duration * 2.0
    unique_mux_token = str(time.time_ns())

    if gameplay_duration <= required_source_duration + 0.5:
        start_time = 0.0
    else:
        start_time = random.uniform(0.0, gameplay_duration - required_source_duration - 0.3)

    filter_complex = build_filter_complex(
        subtitle_path=srt_path,
        start_time=start_time,
        total_duration=target_duration,
        popups=popup_images,
        burn_subtitles=burn_subtitles,
        source_top_crop=source_top_crop,
    )

    input_parts = [
        f"-i {shlex.quote(str(gameplay_path))}",
        f"-i {shlex.quote(str(narration_path))}",
    ]
    for popup in popup_images:
        input_parts.append(f"-loop 1 -t {target_duration:.3f} -i {shlex.quote(str(popup.path))}")
    popup_sfx_input_index = -1
    if popup_sfx_path is not None and popup_sfx_path.exists() and any(
        p.play_sfx for p in popup_images
    ):
        input_parts.append(f"-i {shlex.quote(str(popup_sfx_path))}")
        popup_sfx_input_index = len(input_parts) - 1
    bgm_input_index = -1
    if bgm_path is not None and bgm_path.exists():
        input_parts.append(f"-stream_loop -1 -i {shlex.quote(str(bgm_path))}")
        bgm_input_index = len(input_parts) - 1
    subtitle_input_index = len(input_parts)
    if not burn_subtitles and srt_path.exists():
        input_parts.append(f"-i {shlex.quote(str(srt_path))}")

    current_audio_label = "1:a"
    audio_map = "-map 1:a "
    if popup_sfx_input_index >= 0:
        audio_chain = build_popup_sfx_audio_chain(
            popup_images,
            popup_sfx_input_index,
            sfx_trim_seconds=popup_sfx_trim_seconds,
            sfx_speed=popup_sfx_speed,
            sfx_volume=popup_sfx_volume,
        )
        if audio_chain:
            filter_complex = f"{filter_complex};{audio_chain}"
            current_audio_label = "aout"
            audio_map = "-map [aout] "
    if bgm_input_index >= 0:
        bgm_vol = max(0.0, float(bgm_volume))
        bgm_chain = (
            f"[{current_audio_label}]aresample=44100,volume=1.0[abase2];"
            f"[{bgm_input_index}:a]silenceremove=start_periods=1:start_duration=0.03:start_threshold=-45dB,atrim=0:{target_duration:.3f},asetpts=N/SR/TB,volume={bgm_vol:.3f}[bgm];"
            f"[abase2][bgm]amix=inputs=2:duration=first:normalize=0:dropout_transition=0[aoutmix]"
        )
        filter_complex = f"{filter_complex};{bgm_chain}"
        audio_map = "-map [aoutmix] "

    cmd = (
        "ffmpeg -y "
        + " ".join(input_parts)
        + " "
        + f"-filter_complex {shlex.quote(filter_complex)} "
        + f"-map [vout] {audio_map}"
        + "-map_metadata -1 -map_chapters -1 "
        + "-c:v libx264 -preset medium -crf 20 -r 60 "
        + "-c:a aac -b:a 160k "
        + f"-metadata comment={shlex.quote(unique_mux_token)} "
        + "-movflags +faststart "
        + "-shortest "
        + shlex.quote(str(out_video_path))
    )
    if not burn_subtitles and srt_path.exists():
        cmd = (
            "ffmpeg -y "
            + " ".join(input_parts)
            + " "
            + f"-filter_complex {shlex.quote(filter_complex)} "
            + f"-map [vout] {audio_map}-map {subtitle_input_index}:0 "
            + "-map_metadata -1 -map_chapters -1 "
            + "-c:v libx264 -preset medium -crf 20 -r 60 "
            + "-c:a aac -b:a 160k "
            + "-c:s mov_text "
            + "-metadata:s:s:0 language=eng "
            + f"-metadata comment={shlex.quote(unique_mux_token)} "
            + "-movflags +faststart "
            + "-shortest "
            + shlex.quote(str(out_video_path))
        )
    run(cmd)
    return start_time


def get_youtube_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
    ]
    client_secret_file = Path(os.environ.get("YOUTUBE_CLIENT_SECRET_FILE", "client_secret.json"))
    token_file = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "youtube_token.json"))

    if not client_secret_file.exists():
        raise RuntimeError(
            f"Missing YouTube OAuth client file: {client_secret_file}. "
            "Download a Desktop app OAuth client JSON from Google Cloud and save it there."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes=scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.exceptions import RefreshError
            try:
                creds.refresh(Request())
            except RefreshError:
                print("YouTube token expired or revoked. Re-authenticating...")
                token_file.unlink(missing_ok=True)
                flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
                creds = flow.run_local_server(port=0, open_browser=True)
        else:
            print("Opening browser to link your YouTube channel...")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
            creds = flow.run_local_server(port=0, open_browser=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


def upload_to_youtube(video_file: Path, title: str, description: str, tags: List[str], privacy: str) -> str:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    creds = get_youtube_credentials()

    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22",
        },
        "status": {"privacyStatus": privacy},
    }
    media = MediaFileUpload(str(video_file), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()

    video_id = response["id"]
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            status_response = youtube.videos().list(
                part="status,processingDetails",
                id=video_id,
            ).execute()
        except HttpError as exc:
            if getattr(exc, "resp", None) is not None and exc.resp.status == 403:
                print("Upload completed, but processing status check was skipped due to YouTube scope limits.")
                break
            raise
        items = status_response.get("items", [])
        if not items:
            break
        item = items[0]
        processing_status = (
            item.get("processingDetails", {}).get("processingStatus", "").lower()
        )
        upload_status = item.get("status", {}).get("uploadStatus", "").lower()
        if upload_status == "processed" or processing_status == "succeeded":
            break
        if processing_status == "failed":
            raise RuntimeError(f"YouTube processing failed for uploaded video {video_id}.")
        time.sleep(5)
    return f"https://www.youtube.com/watch?v={video_id}"


PINNED_COMMENTS = [
    "Should we make a Part 2? 👀",
    "Which part hit different for you? Drop it below 👇",
    "Who else has been through this?? 😭",
    "Tell me I'm not the only one 💀",
    "Tag someone who needs to see this 👀",
    "Part 2 if this gets 500 likes? 🤔",
    "What would YOU have done in this situation? 👇",
    "This actually happened btw 😭",
]


def post_pinned_comment(youtube, video_id: str, script: str) -> None:
    comment_text = random.choice(PINNED_COMMENTS)
    try:
        response = youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {"textOriginal": comment_text}
                    },
                }
            },
        ).execute()
        comment_id = response["snippet"]["topLevelComment"]["id"]
        youtube.comments().setModerationStatus(
            id=comment_id,
            moderationStatus="published",
        ).execute()
        print(f"Pinned comment: {comment_text}")
    except Exception as exc:
        print(f"Could not post pinned comment: {exc}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate and upload YouTube Shorts automatically.")
    parser.add_argument("--words", type=int, default=70, help="Approx script word count (70 ≈ 18–24s narrated)")
    parser.add_argument(
        "--topic",
        default="",
        help="Optional rant topic, e.g. people blasting speakerphone in public",
    )
    parser.add_argument(
        "--adam-cloner-script",
        default="",
        help="Optional path to Adam voice cloner run script (defaults to VoiceCloner/run_clone.sh)",
    )
    parser.add_argument(
        "--tts",
        default="cloner",
        choices=["cloner", "openai"],
        help="Narration engine: cloner (local Adam voice) or openai (cloud-friendly).",
    )
    parser.add_argument(
        "--dynamic-speed",
        action="store_true",
        help='Apply speed ramps around phrases inside double quotes or --double hyphens--, then re-transcribe for final subtitle sync.',
    )
    parser.add_argument(
        "--speed-ramp-ms",
        type=int,
        default=600,
        help="How long the speed ramps take to glide in/out, in milliseconds.",
    )
    parser.add_argument(
        "--speed-slow",
        type=float,
        default=0.60,
        help="Slow speed factor during highlighted words (default: 0.60).",
    )
    parser.add_argument(
        "--speed-fast",
        type=float,
        default=1.15,
        help="Base speed factor outside highlights (default: 1.15).",
    )
    parser.add_argument("--upload", action="store_true", help="Upload the output video to YouTube")
    parser.add_argument("--privacy", default="public", choices=["private", "unlisted", "public"], help="public = Shorts feed; private = no impressions")
    parser.add_argument("--no-description", action="store_true", help="Upload with an empty YouTube description")
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate popup images using OpenAI image model",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional final video length in seconds (default: full narration length)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation and reuse existing output narration audio",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Render video only from existing narration/subtitles without new script or voice",
    )
    parser.add_argument(
        "--popup-sfx",
        default="assets/vine-boom.mp3",
        help="Sound effect played when story-related popup images appear",
    )
    parser.add_argument(
        "--popup-sfx-volume",
        type=float,
        default=0.18,
        help="Vine boom volume multiplier (default: 0.18)",
    )
    parser.add_argument(
        "--popup-sfx-speed",
        type=float,
        default=1.25,
        help="Vine boom playback speed (default: 1.25)",
    )
    parser.add_argument(
        "--popup-sfx-trim-seconds",
        type=float,
        default=1.4,
        help="How much of vine boom to keep in seconds (default: 1.4)",
    )
    parser.add_argument(
        "--bgm-path",
        default="assets/Chopin - Nocturne op.9 No.2.mp3",
        help="Optional quiet background music file",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.08,
        help="Background music volume multiplier (default: 0.08)",
    )
    parser.add_argument(
        "--gameplay-top-crop",
        type=int,
        default=96,
        help="Crop this many pixels from the top of the gameplay source before vertical reframing",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a 3-second quick test for styling and video pipeline",
    )
    args = parser.parse_args()

    if args.quick_test:
        args.words = 15
        args.duration_seconds = 3.0
        print("--- QUICK TEST MODE ENABLED (3 seconds) ---")

    if args.video_only and args.generate_images:
        raise RuntimeError("--video-only cannot be combined with --generate-images.")

    needs_openai = not args.video_only or args.generate_images

    api_key = os.environ.get("OPENAI_API_KEY")
    if needs_openai and not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    project_root = Path.cwd()
    gameplay_dir = project_root / "assets" / "gameplay"
    images_dir = project_root / "assets" / "popups"
    story_images_dir = project_root / "assets" / "story_images"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    gameplay_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    story_images_dir.mkdir(parents=True, exist_ok=True)

    if args.upload:
        get_youtube_credentials()

    gameplay_file = pick_random_file(gameplay_dir, ["mp4", "mov", "mkv", "webm"])
    client = OpenAI(api_key=api_key) if api_key else None

    total_steps = 6 + (1 if args.upload else 0)
    step = 1
    start_ts = time.time()

    script_file = output_dir / "script.txt"
    if args.video_only:
        print_progress(step, total_steps, "Reusing existing script")
        if script_file.exists():
            script = script_file.read_text(encoding="utf-8").strip()
        else:
            script = "Crazy School Story You Won't Believe"
    else:
        while True:
            print_progress(step, total_steps, "Generating story script")
            script = generate_script(
                client,
                args.words,
                topic=args.topic,
            )  # type: ignore[arg-type]
            clean_script = script.replace('*', '')
            print("\n--- Generated Script ---")
            print(clean_script)
            print("------------------------")
            if sys.stdin.isatty():
                yn = input("Use this script? (Y/N) ").strip().upper()
                if yn in ("Y", "YES", ""):
                    break
                print("Regenerating...\n")
            else:
                break
        script_file.write_text(script, encoding="utf-8")

    step += 1
    narration_file = output_dir / "narration.mp3"
    narration_reference_segments: List[dict] = []
    if args.video_only or args.skip_tts:
        print_progress(step, total_steps, "Reusing existing narration")
        if not narration_file.exists():
            raw_narration_file = output_dir / "narration_raw.mp3"
            if raw_narration_file.exists():
                narration_file = raw_narration_file
            else:
                raise RuntimeError("Missing output narration file for reuse mode.")
    else:
        print_progress(step, total_steps, "Creating voiceover")
        raw_narration_file = output_dir / "narration_raw.mp3"
        if args.tts == "openai":
            if client is None:
                raise RuntimeError("OpenAI client missing; cannot use --tts openai.")
            generate_voiceover_openai_tts(client, script, raw_narration_file)
        else:
            generate_voiceover_from_cloner_script(
                script_text=script,
                out_audio_path=raw_narration_file,
                project_root=project_root,
                cloner_script=args.adam_cloner_script,
            )
        if args.dynamic_speed and re.search(r'--([^-][\s\S]*?[^-])--', script):
            if client is None:
                raise RuntimeError("OpenAI client missing; cannot use --dynamic-speed.")
            print("Applying dynamic speed ramps from --hyphen-wrapped-- phrases...")
            raw_words = get_whisper_word_timestamps(client, raw_narration_file, script)
            highlights = get_highlight_timestamps(script, raw_words)
            if highlights:
                smart_speed_ramp(
                    input_path=raw_narration_file,
                    output_path=narration_file,
                    interesting_segments=highlights,
                    ramp_ms=args.speed_ramp_ms,
                    slow_factor=args.speed_slow,
                    fast_factor=args.speed_fast,
                )
            else:
                print("No matching --hyphen-wrapped-- timestamps found; using raw narration.")
                shutil.copyfile(raw_narration_file, narration_file)
        else:
            narration_file = raw_narration_file
        if client is not None:
            narration_reference_segments = get_whisper_word_timestamps(client, narration_file, script)

    step += 1
    subtitle_file = output_dir / "subtitles.ass"
    emoji_events_file = output_dir / "emoji_events.json"
    subtitle_segments: List[dict] = []
    if args.video_only:
        print_progress(step, total_steps, "Reusing existing subtitles")
        old_srt_file = output_dir / "subtitles.srt"
        # In video-only mode, prefer Whisper from narration for tight sync.
        if client is not None:
            _, subtitle_segments = transcribe_audio_to_srt(
                client,
                narration_file,
                subtitle_file,
                script_text=script,
                reference_segments=narration_reference_segments or None,
            )  # type: ignore[arg-type]
        elif old_srt_file.exists():
            timed_segments = read_srt_segments(old_srt_file)
            subtitle_segments = timed_segments
            write_ass_from_segments(subtitle_segments, subtitle_file)
        elif not subtitle_file.exists():
            raise RuntimeError("Missing output/subtitles.ass for --video-only mode.")
    else:
        print_progress(step, total_steps, "Building subtitles")
        generated_emoji_events, subtitle_segments = transcribe_audio_to_srt(
            client,
            narration_file,
            subtitle_file,
            script_text=script,
            reference_segments=narration_reference_segments or None,
        )  # type: ignore[arg-type]
        emoji_events_file.write_text(json.dumps(generated_emoji_events), encoding="utf-8")

    step += 1
    print_progress(step, total_steps, "Preparing popup images")
    if args.generate_images:
        maybe_generate_images(
            client=client,  # type: ignore[arg-type]
            script=script,
            images_dir=images_dir,
            count=3,
            generate_images=True,
        )

    narration_duration = ffprobe_duration_seconds(narration_file)
    story_text_for_matching = script
    if subtitle_segments:
        story_text_for_matching += " " + " ".join(
            str(s.get("text") or s.get("raw_text") or "") for s in subtitle_segments
        )
    # Force cadence: 2 images every 5 seconds => one every 2.5 seconds.
    planned_image_times: List[float] = fixed_image_times(
        narration_duration, interval_seconds=2.5
    )
    maybe_download_story_images(
        story_images_dir, story_text_for_matching, client=client, min_count=18
    )
    popups = choose_story_related_popups(
        story_images_dir,
        story_text_for_matching,
        narration_duration,
        subtitle_segments=subtitle_segments,
        planned_times=planned_image_times,
        min_gap=2.0,
        max_gap=6.0,
    )
    if not popups:
        popups = choose_popup_images(images_dir, narration_duration, count=3)
    burn_subtitles = ffmpeg_has_subtitles_filter()
    if not burn_subtitles:
        print("Subtitle burn-in unavailable in this ffmpeg build; exporting without burned subtitles.")

    step += 1
    print_progress(step, total_steps, "Rendering short video")
    output_video = output_dir / "short.mp4"
    selected_start = compose_video(
        gameplay_path=gameplay_file,
        narration_path=narration_file,
        srt_path=subtitle_file,
        popup_images=popups,
        out_video_path=output_video,
        duration_seconds=args.duration_seconds,
        burn_subtitles=burn_subtitles,
        popup_sfx_path=Path(args.popup_sfx) if args.popup_sfx else None,
        popup_sfx_trim_seconds=args.popup_sfx_trim_seconds,
        popup_sfx_speed=args.popup_sfx_speed,
        popup_sfx_volume=args.popup_sfx_volume,
        bgm_path=Path(args.bgm_path) if args.bgm_path else None,
        bgm_volume=args.bgm_volume,
        source_top_crop=args.gameplay_top_crop,
    )

    step += 1
    print_progress(step, total_steps, "Generating metadata")
    if client is not None:
        title, description = generate_metadata(client, script, include_description=not args.no_description)
    else:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook
        title = title or "Crazy School Story You Won't Believe"
        description = "" if args.no_description else (
            "Subscribe for more storytime shorts!\n"
            "#shorts #storytime #schoolstory"
        )
    # Ensure #Shorts in both (API uploads often need this for feed classification)
    if "#shorts" not in title.lower():
        title = f"{title} #Shorts"
    if description and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"
    tags = ["shorts", "storytime", "school story", "crazy story", "viral short"]

    metadata_file = output_dir / "metadata.txt"
    metadata_file.write_text(
        f"Title:\n{title}\n\nDescription:\n{description}\n\nTags: {', '.join(tags)}",
        encoding="utf-8",
    )
    print("\n--- YouTube Shorts Metadata ---")
    print(f"Title:\n{title}\n")
    print(f"Description:\n{description if description else '(empty)'}")

    print("\nDone.")
    print(f"Total run time: {time.time() - start_ts:.1f}s")
    print(f"Gameplay source: {gameplay_file.name}")
    print(f"Random start time: {selected_start:.2f}s")
    if args.duration_seconds is not None:
        print(f"Forced output duration: {args.duration_seconds:.2f}s")
    print(f"Output video: {output_video}")
    print(f"Script text: {script_file}")
    print(f"Subtitles: {subtitle_file}")
    if not burn_subtitles:
        print("Subtitles added as selectable subtitle track (not burned into pixels).")

    if args.upload:
        step += 1
        print_progress(step, total_steps, "Uploading to YouTube")
        video_url = upload_to_youtube(
            video_file=output_video,
            title=title,
            description=description,
            tags=tags,
            privacy=args.privacy,
        )
        print(f"Uploaded: {video_url}")
        print("Note: Shorts can take 1–5 min to process before appearing in the feed.")
        from googleapiclient.discovery import build as _build
        _yt = _build("youtube", "v3", credentials=get_youtube_credentials())
        video_id = video_url.split("v=")[-1]
        post_pinned_comment(_yt, video_id, script)
    else:
        print("Upload skipped. Run with --upload to publish.")


if __name__ == "__main__":
    main()
