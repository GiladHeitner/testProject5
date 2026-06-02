"""Subtitle file generation: ASS / SRT writers and caption chunking."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .emoji import pick_emoji_for_text
from .text import (
    ass_ts,
    escape_ass_text,
    format_bold_ass,
    format_bold_srt,
    format_caption_multiline,
    maybe_upper,
    random_caps_text,
    sanitize_ass_text,
    to_ass_time,
    to_srt_time,
)
from .types import Word


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
    rtl: bool = False,
) -> None:
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    x = play_res_x // 2
    words_per_block = max(2, int(words_per_block))

    # Classic \\k karaoke keeps earlier words highlighted; we draw full caption on a base
    # layer (all white) and per-word overlay lines (same text, only active word yellow).
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

    rtl_tag = r"{\rtl1}" if rtl else ""
    override = rf"{{\an8\pos({x},{y})\bord7\shad0\3c&H000000&\4c&H000000&}}{rtl_tag}"
    yellow = "&H0000FFFF&"
    white = "&H00FFFFFF&"

    lines: List[str] = [header]
    idx = 0
    while idx < len(words):
        chunk = words[idx : idx + words_per_block]
        if not chunk:
            break

        chunk_tokens = [
            maybe_upper(sanitize_ass_text(word.text.strip()), uppercase_ratio)
            for word in chunk
        ]
        line_start_s = float(chunk[0].start)
        line_end_s = float(max(chunk[-1].end, chunk[-1].start + 0.05))
        start = ass_ts(line_start_s)
        end = ass_ts(line_end_s)

        # Layer 0: full line white for the whole chunk duration.
        plain_text = " ".join(
            rf"{{\1c{white}}}{escape_ass_text(t)}" for t in chunk_tokens
        )
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{override}{plain_text}\n"
        )

        # Layer 1: during each word's window, redraw full line with only that word yellow.
        for wi, w in enumerate(chunk):
            ws = float(w.start)
            we = float(max(w.end, w.start + 0.05))
            colored = []
            for wj, tok in enumerate(chunk_tokens):
                et = escape_ass_text(tok)
                col = yellow if wj == wi else white
                colored.append(rf"{{\1c{col}}}{et}")
            line_colored = " ".join(colored)
            lines.append(
                f"Dialogue: 1,{ass_ts(ws)},{ass_ts(we)},Default,,0,0,0,,{override}{line_colored}\n"
            )

        idx += words_per_block

    out_ass.write_text("".join(lines), encoding="utf-8")


def write_srt_from_segments(segments: List[dict], out_path: Path) -> None:
    lines: List[str] = []
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


def split_caption_chunks(segments: List[dict], words_per_chunk: int = 1) -> List[dict]:
    """Turn uneven Whisper segments into steady Shorts caption beats."""
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


def remap_script_to_reference_timing(
    script_text: str, reference_segments: List[dict], max_words_per_segment: int = 5
) -> List[dict]:
    words = script_text.strip().split()
    if not words or not reference_segments:
        return []

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
            take = min(take, max(1, remaining_words - (remaining_segments - 1)))
        if i != len(reference_segments) - 1:
            take = min(take, max_words_per_segment + 2)
        chunk = words[cursor : cursor + take]
        cursor += take

        out.append({"start": start, "end": end, "text": " ".join(chunk)})

    if cursor < len(words) and out:
        out[-1]["text"] = str(out[-1]["text"]) + " " + " ".join(words[cursor:])
    return out
