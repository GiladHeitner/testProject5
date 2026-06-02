"""Transcription helpers: faster-whisper words + OpenAI Whisper API alignment."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

from .subtitles import write_karaoke_block_ass, write_srt_from_segments
from .text import script_words_for_alignment, strip_script_markup
from .types import Word


def transcribe_words(
    audio_path: Path,
    model_name: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> List[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing `faster-whisper`. Install it with `pip install faster-whisper`."
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    transcribe_kwargs = {
        "word_timestamps": True,
        "vad_filter": True,
    }
    if language:
        transcribe_kwargs["language"] = language
    segments, _info = model.transcribe(str(audio_path), **transcribe_kwargs)

    out: List[Word] = []
    for seg in segments:
        if not getattr(seg, "words", None):
            continue
        for word in seg.words:
            text = (word.word or "").strip()
            if not text:
                continue
            out.append(Word(text=text, start=float(word.start), end=float(word.end)))
    if not out:
        raise RuntimeError("No words produced from transcription.")
    return out


def build_whisper_prompt(script_text: str) -> str:
    cleaned = strip_script_markup(script_text)
    words = cleaned.split()
    return " ".join(words[:244])


def get_whisper_word_timestamps(client: OpenAI, audio_path: Path, script_text: str = "") -> List[dict]:
    """Get exact word-level timestamps from the OpenAI Whisper API."""
    whisper_prompt = build_whisper_prompt(script_text) if script_text else ""
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=whisper_prompt or None,
        )
    words: List[dict] = []
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


def build_caption_chunks_from_word_timestamps(
    script_text: str,
    word_timestamps: List[dict],
    words_per_chunk: int = 1,
) -> List[dict]:
    clean_script = script_words_for_alignment(script_text)
    if not clean_script or not word_timestamps:
        return []

    def norm(token: str) -> str:
        return re.sub(r"[^\w]", "", token, flags=re.UNICODE).lower()

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


def _words_from_script_alignment(
    script_text: str, audio_path: Path, *, language: str | None = None
) -> List[Word]:
    """Script words with timings from faster-whisper (fixes IDIOT -> ID I on screen)."""
    whisper_words = transcribe_words(audio_path, language=language)
    ts = [
        {
            "text": w.text,
            "start": float(w.start),
            "end": float(max(w.end, w.start + 0.05)),
        }
        for w in whisper_words
    ]
    chunks = build_caption_chunks_from_word_timestamps(
        script_text, ts, words_per_chunk=1
    )
    out: List[Word] = []
    for ch in chunks:
        text = str(ch.get("raw_text", "")).strip()
        if not text:
            continue
        out.append(
            Word(
                text=text,
                start=float(ch["start"]),
                end=float(ch["end"]),
            )
        )
    return out


def transcribe_audio_to_srt(
    client: OpenAI | None,
    audio_path: Path,
    out_srt_path: Path,
    script_text: str = "",
    reference_segments: List[dict] | None = None,
    *,
    language: str | None = None,
    subtitle_font: str = "Gibson",
    subtitle_rtl: bool = False,
    subtitle_uppercase_ratio: float = 0.15,
) -> Tuple[List[dict], List[dict]]:
    """Transcribe narration and write subtitle files."""
    del client, reference_segments

    if script_text.strip():
        words = _words_from_script_alignment(script_text, audio_path, language=language)
    else:
        words = transcribe_words(audio_path, language=language)

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
        write_karaoke_block_ass(
            words,
            out_srt_path,
            font=subtitle_font,
            rtl=subtitle_rtl,
            uppercase_ratio=subtitle_uppercase_ratio,
        )
        write_srt_from_segments(word_segments, out_srt_path.with_suffix(".srt"))
    else:
        write_srt_from_segments(word_segments, out_srt_path)
    return [], word_segments
