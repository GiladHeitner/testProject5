"""Narration generation: local voice cloner (Omar reference) + OpenAI TTS."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from .runner import print_sub_progress, run
from .text import (
    normalize_script_for_tts,
    prepare_script_for_qwen_tts,
    strip_paralinguistic_tags,
    strip_script_markup,
)
from .qwen_voice import load_qwen_voice_config, qwen_voice_env


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


def cloner_reference_audio_path(project_root: Path) -> Path:
    """Reference clip for clone mode (REF_AUDIO env or qwen_voice.json)."""
    ref = os.environ.get("REF_AUDIO", "").strip()
    if not ref:
        cfg = load_qwen_voice_config(project_root / "assets" / "qwen_voice.json")
        ref = cfg.ref_audio
    ref_path = Path(ref).expanduser()
    if not ref_path.is_absolute():
        for candidate in (
            project_root / "VoiceCloner" / ref,
            project_root / ref.lstrip("./"),
        ):
            if candidate.is_file():
                return candidate.resolve()
        return (project_root / "VoiceCloner" / ref).resolve()
    return ref_path.resolve()


def resolve_tts_engine(requested: str, project_root: Path) -> str:
    """Pick cloner vs OpenAI (OpenAI only when explicitly requested or ref missing in clone mode)."""
    engine = (requested or "cloner").strip().lower()
    if engine == "openai":
        return "openai"
    cfg = load_qwen_voice_config(project_root / "assets" / "qwen_voice.json")
    if cfg.mode in ("custom", "design"):
        return "cloner"
    ref = cloner_reference_audio_path(project_root)
    if ref.is_file():
        return "cloner"
    print(
        f"[tts] Reference audio missing ({ref.name}) — using OpenAI TTS.",
        flush=True,
    )
    return "openai"


def _resolve_cloner_script(project_root: Path, configured_script: str) -> Path:
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
            "Voice cloner script not found. Expected `VoiceCloner/run_clone.sh`."
        )
    return script_path


def generate_voiceover_from_cloner_script(
    script_text: str, out_audio_path: Path, project_root: Path, cloner_script: str
) -> None:
    run_clone_path = _resolve_cloner_script(project_root, cloner_script)
    voice_cfg = load_qwen_voice_config(project_root / "assets" / "qwen_voice.json")
    tmp_wav = out_audio_path.with_suffix(".adam_tmp.wav")
    tts_text = prepare_script_for_qwen_tts(script_text)
    tts_script_path = out_audio_path.parent / "tts_script.txt"
    tts_script_path.write_text(tts_text + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["TEXT"] = tts_text
    env["OUTPUT"] = str(tmp_wav)
    env["USE_BATCH"] = "false"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(qwen_voice_env(voice_cfg, project_root))
    print(
        f"[tts] Qwen mode={voice_cfg.mode} speaker={voice_cfg.speaker} model={voice_cfg.model}",
        flush=True,
    )

    # Heuristic phase markers — match common substrings emitted by huggingface
    # / TTS pipelines so the UI's sub-progress bar moves while the model runs.
    phases: list[tuple[str, int, str]] = [
        ("loading", 1, "Loading voice model"),
        ("downloading", 1, "Downloading model files"),
        ("cloning", 2, "Cloning reference voice"),
        ("synthesi", 3, "Synthesizing speech"),
        ("generating", 3, "Synthesizing speech"),
        ("saving", 4, "Saving audio"),
        ("post", 4, "Post-processing audio"),
    ]
    total_phases = 5  # last 5/5 is the ffmpeg encode below
    print_sub_progress(0, total_phases, "Starting voice cloner")

    proc = subprocess.Popen(
        ["bash", str(run_clone_path)],
        cwd=str(run_clone_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    last_phase: int = 0
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        captured.append(line)
        print(f"[cloner] {line}", flush=True)
        low = line.lower()
        for needle, phase_idx, label in phases:
            if needle in low and phase_idx > last_phase:
                last_phase = phase_idx
                print_sub_progress(phase_idx, total_phases, label)
                break
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            "Voice cloner run_clone.sh failed.\n"
            + "\n".join(captured[-20:])
        )

    transformed = tmp_wav.with_name(tmp_wav.stem + "_sp.wav")
    source_audio = transformed if transformed.exists() else tmp_wav
    if not source_audio.exists():
        raise RuntimeError("Voice cloner did not produce output audio.")
    print_sub_progress(4, total_phases, "Encoding mp3")
    run(
        f"ffmpeg -y -i {shlex.quote(str(source_audio))} "
        f"-c:a libmp3lame -q:a 2 {shlex.quote(str(out_audio_path))}"
    )
    if tmp_wav.exists():
        tmp_wav.unlink()
    if transformed.exists():
        transformed.unlink()
    print_sub_progress(total_phases, total_phases, "Voiceover ready")


def generate_voiceover_openai_tts(client: OpenAI, script_text: str, out_audio_path: Path) -> None:
    text = strip_paralinguistic_tags(normalize_script_for_tts(script_text))
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
