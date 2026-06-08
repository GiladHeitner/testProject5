"""Qwen3-TTS voice settings (CustomVoice / VoiceDesign / clone)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QwenVoiceConfig:
    mode: str = "custom"  # custom | design | clone
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    speaker: str = "Ryan"
    language: str = "English"
    instruct: str = ""
    design_instruct: str = ""
    ref_audio: str = "../assets/grove_3.m4a"
    speed: float = 1.12
    pitch: float = 0.0


def load_qwen_voice_config(path: Path | None = None) -> QwenVoiceConfig:
    cfg_path = path or Path(
        os.environ.get("QWEN_VOICE_CONFIG", "assets/qwen_voice.json")
    )
    if not cfg_path.is_file():
        return QwenVoiceConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return QwenVoiceConfig(
        mode=str(data.get("mode", "custom")).strip().lower() or "custom",
        model=str(data.get("model", QwenVoiceConfig.model)).strip(),
        speaker=str(data.get("speaker", "Ryan")).strip() or "Ryan",
        language=str(data.get("language", "English")).strip() or "English",
        instruct=str(data.get("instruct", "")).strip(),
        design_instruct=str(data.get("design_instruct", "")).strip(),
        ref_audio=str(data.get("ref_audio", "../assets/grove_3.m4a")).strip(),
        speed=float(data.get("speed", 1.12)),
        pitch=float(data.get("pitch", 0.0)),
    )


def qwen_voice_env(config: QwenVoiceConfig, project_root: Path) -> dict[str, str]:
    """Env vars for VoiceCloner/run_clone.sh."""
    mode = config.mode.strip().lower()
    instruct = config.instruct
    if mode == "design" and config.design_instruct:
        instruct = config.design_instruct
    ref = config.ref_audio
    if ref and not Path(ref).is_absolute():
        ref_path = (project_root / "VoiceCloner" / ref).resolve()
        if not ref_path.is_file():
            alt = (project_root / ref.lstrip("./")).resolve()
            ref = str(alt if alt.is_file() else ref_path)
        else:
            ref = str(ref_path)
    return {
        "QWEN_VOICE_MODE": mode,
        "MODEL": config.model,
        "SPEAKER": config.speaker,
        "LANGUAGE": config.language,
        "VOICE_INSTRUCT": instruct,
        "REF_AUDIO": ref,
        "SPEED": str(config.speed),
        "PITCH": str(config.pitch),
    }
