"""Narration generation: Adam voice cloner script + OpenAI TTS."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from .runner import print_sub_progress, run
from .text import strip_script_markup


def _split_for_tts(text: str, max_chars: int = 220) -> list[str]:
    """Split a script into ~sentence-sized chunks for streaming TTS progress."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current = f"{current} {s}"
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks or [text]


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
    chunks = _split_for_tts(text)
    total = len(chunks)
    print_sub_progress(0, total, f"Generating voiceover (0/{total} chunks)")

    models_to_try = ["gpt-4o-mini-tts", "tts-1"]

    def _synth_chunk(chunk_text: str) -> bytes:
        last_exc: Exception | None = None
        for model in models_to_try:
            try:
                audio = client.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=chunk_text,
                    format="mp3",
                )
                return audio.read()
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"OpenAI TTS failed: {last_exc}")

    if total == 1:
        out_audio_path.write_bytes(_synth_chunk(chunks[0]))
        print_sub_progress(1, 1, "Voiceover done")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        chunk_paths: list[Path] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_path = tmp_root / f"chunk_{idx:03d}.mp3"
            chunk_path.write_bytes(_synth_chunk(chunk))
            chunk_paths.append(chunk_path)
            print_sub_progress(idx, total, f"Generating voiceover ({idx}/{total} chunks)")

        list_file = tmp_root / "list.txt"
        list_file.write_text(
            "\n".join(f"file {shlex.quote(str(p))}" for p in chunk_paths),
            encoding="utf-8",
        )
        run(
            f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(list_file))} "
            f"-c copy {shlex.quote(str(out_audio_path))}"
        )
    print_sub_progress(total, total, "Voiceover stitched")
