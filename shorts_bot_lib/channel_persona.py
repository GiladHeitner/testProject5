"""Channel host persona config for consistent script voice."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ChannelPersona:
    name: str
    age: int
    gender: str
    identity: str
    rant_style: str
    slang: List[str] = field(default_factory=list)
    speech_rules: List[str] = field(default_factory=list)


def default_persona_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    env_path = os.environ.get("CHANNEL_PERSONA_FILE", "").strip()
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else root / p
    return root / "assets" / "channel_persona.json"


def load_channel_persona(path: Path | None = None, *, project_root: Path | None = None) -> ChannelPersona:
    persona_path = path or default_persona_path(project_root)
    if not persona_path.is_file():
        raise FileNotFoundError(f"Channel persona file not found: {persona_path}")
    raw = json.loads(persona_path.read_text(encoding="utf-8"))
    if raw.get("home_city"):
        print(
            f"Channel persona: ignoring deprecated home_city field in {persona_path.name}"
        )
    name = str(raw.get("name") or "Omar").strip()
    age = int(raw.get("age") or 17)
    gender = str(raw.get("gender") or "male").strip().lower()
    identity = str(raw.get("identity") or "Muslim Arab teen").strip()
    rant_style = str(raw.get("rant_style") or "heated, sarcastic").strip()
    slang = [str(s).strip() for s in (raw.get("slang") or []) if str(s).strip()]
    speech_rules = [
        str(r).strip() for r in (raw.get("speech_rules") or []) if str(r).strip()
    ]
    if age < 13 or age > 19:
        raise ValueError(f"Channel persona age must be 13-19, got {age}")
    if not name:
        raise ValueError("Channel persona name is required")
    return ChannelPersona(
        name=name,
        age=age,
        gender=gender,
        identity=identity,
        rant_style=rant_style,
        slang=slang,
        speech_rules=speech_rules,
    )


def format_persona_block(persona: ChannelPersona) -> str:
    slang_line = ", ".join(persona.slang) if persona.slang else "wallah, yallah, inshallah"
    rules = "\n".join(f"- {rule}" for rule in persona.speech_rules)
    return (
        f"Name: {persona.name} (same person every video — do not introduce by name unless natural)\n"
        f"Age: {persona.age}\n"
        f"Gender: {persona.gender}\n"
        f"Identity: {persona.identity}\n"
        f"Voice: {persona.rant_style}\n"
        f"Slang to use naturally (1-3 per script): {slang_line}\n"
        f"Rules:\n{rules}"
    )


def persona_summary(persona: ChannelPersona) -> str:
    return f"{persona.name}, {persona.age}, {persona.gender}"
