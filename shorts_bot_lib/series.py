"""Recurring-character "Omar" series mechanic (Part 1 / Part 2).

Why: the channel's biggest weakness is that nobody returns (<0.1% regular
viewers). Series are the strongest sub-conversion lever — when a viewer
watches Part 1 they have a concrete reason to subscribe so they don't miss
Part 2. See CHANNEL_IMPROVEMENT_PLAN.md.

How it works across runs:
- With probability SERIES_PART1_PCT, a run becomes a **Part 1**: the script
  stops on a real cliffhanger and the CTA promises Part 2. The story context
  is saved to .github/series_state.json.
- The very next run sees the pending state and becomes **Part 2**: it
  continues that exact story, resolves the cliffhanger, then clears the state.
- Everything else is a normal standalone short.

This module is pure and side-effect-light (only reads/writes the state JSON),
so it is unit-testable without any network/LLM calls — see tests/test_series.py.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_SERIES_STATE = Path(".github/series_state.json")

# Roles a single run can take.
ROLE_STANDALONE = "standalone"
ROLE_PART1 = "part1"
ROLE_PART2 = "part2"


@dataclass
class PendingSeries:
    """A Part 1 awaiting its Part 2 continuation."""

    part: int          # the part number to produce NEXT (always 2 for now)
    title: str         # Part 1's title (for recap/continuity)
    topic: str         # original source topic/text, so Part 2 continues it
    summary: str       # what happened in Part 1 (trimmed script)
    cliffhanger: str   # the open loop Part 2 must resolve


# --- env knobs ----------------------------------------------------------------
def series_enabled() -> bool:
    return os.environ.get("SERIES_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def part1_pct() -> float:
    try:
        return max(0.0, min(100.0, float(os.environ.get("SERIES_PART1_PCT", "30"))))
    except ValueError:
        return 30.0


def state_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return path
    return Path(os.environ.get("SERIES_STATE_FILE", str(DEFAULT_SERIES_STATE)))


# --- state persistence --------------------------------------------------------
def load_pending(path: Optional[Path] = None) -> Optional[PendingSeries]:
    p = state_path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return PendingSeries(
            part=int(raw["part"]),
            title=str(raw.get("title", "")),
            topic=str(raw.get("topic", "")),
            summary=str(raw.get("summary", "")),
            cliffhanger=str(raw.get("cliffhanger", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


def save_pending(pending: PendingSeries, path: Optional[Path] = None) -> None:
    p = state_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(pending), ensure_ascii=False, indent=2), encoding="utf-8")


def clear_pending(path: Optional[Path] = None) -> None:
    p = state_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    except OSError:
        pass


# --- decision + prompt building ----------------------------------------------
def decide_role(
    pending: Optional[PendingSeries],
    *,
    rng: Optional[random.Random] = None,
    enabled: Optional[bool] = None,
    pct: Optional[float] = None,
) -> str:
    """Pick this run's role.

    A pending Part 2 always wins (we must resolve an open cliffhanger before
    starting a new one). Otherwise roll for Part 1 vs standalone.
    """
    is_enabled = series_enabled() if enabled is None else enabled
    if not is_enabled:
        return ROLE_STANDALONE
    if pending is not None:
        return ROLE_PART2
    threshold = part1_pct() if pct is None else pct
    roll = (rng or random).random() * 100.0
    return ROLE_PART1 if roll < threshold else ROLE_STANDALONE


def script_directive(role: str, pending: Optional[PendingSeries]) -> str:
    """Extra instructions injected into the script prompt for this role."""
    if role == ROLE_PART1:
        return (
            "\n\nSERIES — THIS IS PART 1 OF 2:\n"
            "- Tell only the FIRST half of the story and stop at the most "
            "suspenseful moment.\n"
            "- Do NOT resolve it. End on a genuine cliffhanger that makes the "
            "viewer desperate to know what happens.\n"
            "- The closing line must promise the next part, e.g. "
            "\"Part 2 tomorrow, subscribe so you don't miss it.\"\n"
        )
    if role == ROLE_PART2 and pending is not None:
        return (
            "\n\nSERIES — THIS IS PART 2, THE FINALE. Continue this exact story.\n"
            f"PART 1 TITLE: {pending.title}\n"
            f"WHAT HAPPENED IN PART 1: {pending.summary}\n"
            f"CLIFFHANGER TO RESOLVE: {pending.cliffhanger}\n"
            "- Open with ONE quick line reminding the viewer where Part 1 left "
            "off.\n"
            "- Then continue and RESOLVE the story — pay off the cliffhanger.\n"
            "- Closing line: \"Subscribe so tomorrow's story finds you.\"\n"
        )
    return ""


def title_suffix(role: str) -> str:
    return {ROLE_PART1: " (Part 1)", ROLE_PART2: " (Part 2)"}.get(role, "")


_CTA_MARKERS = ("subscribe", "part 2", "part two", "follow so", "don't miss", "dont miss")


def _extract_cliffhanger(script: str) -> str:
    """Best-effort: the last real story sentence, before any CTA/promo line."""
    text = " ".join((script or "").split())
    lowered = text.lower()
    # Cut at the EARLIEST CTA/promo marker ("Part 2 tomorrow", "subscribe", ...).
    cuts = [lowered.find(m) for m in _CTA_MARKERS if lowered.find(m) > 0]
    if cuts:
        text = text[: min(cuts)].strip(" .,!?")
    # Last sentence-ish chunk of the remaining story.
    for sep in (". ", "! ", "? "):
        idx = text.rfind(sep)
        if idx > 0:
            return text[idx + len(sep):].strip()
    return text[-200:].strip()


def build_pending_from_part1(*, title: str, topic: str, script: str) -> PendingSeries:
    """Capture a freshly generated Part 1 as the pending Part 2."""
    summary = " ".join((script or "").split())[:600]
    return PendingSeries(
        part=2,
        title=(title or "").strip(),
        topic=(topic or "").strip(),
        summary=summary,
        cliffhanger=_extract_cliffhanger(script),
    )
