"""Narration generation: Adam voice cloner script + OpenAI TTS."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from openai import OpenAI

from .runner import run
from .text import strip_script_markup


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
