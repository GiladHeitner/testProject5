"""Process / shell helpers used by the rest of the package."""

from __future__ import annotations

import random
import shlex
import subprocess
import time
from pathlib import Path
from typing import List


def run(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def run_ffmpeg_with_progress(command: str, total_duration: float | None = None) -> None:
    """Run ffmpeg streaming `-progress` output so the UI can show sub-progress."""
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    pending: dict[str, str] = {}
    last_print = 0.0
    tail_errors: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip("\n")
        if not line:
            continue
        if "=" in line and not line.startswith("["):
            key, _, value = line.partition("=")
            pending[key.strip()] = value.strip()
            if key.strip() == "progress":
                out_time_us = pending.get("out_time_us") or pending.get("out_time_ms") or "0"
                try:
                    out_sec = int(out_time_us) / 1_000_000.0
                except ValueError:
                    out_sec = 0.0
                fps = pending.get("fps", "0")
                speed = pending.get("speed", "0x")
                frame = pending.get("frame", "0")
                pct = ""
                if total_duration and total_duration > 0:
                    pct = f" pct={min(100.0, out_sec / total_duration * 100.0):.1f}"
                now = time.time()
                if now - last_print >= 0.5 or pending.get("progress") == "end":
                    last_print = now
                    print(
                        f"[render] frame={frame} fps={fps} speed={speed} "
                        f"time={out_sec:.2f}s{pct}",
                        flush=True,
                    )
                pending.clear()
        else:
            tail_errors.append(line)
            if len(tail_errors) > 60:
                tail_errors.pop(0)

    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(tail_errors[-40:])
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {command}\n{tail}"
        )


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


def print_progress(step: int, total: int, label: str) -> None:
    width = 24
    filled = int(width * step / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * step / total)
    print(f"[{bar}] {percent:3d}% ({step}/{total}) {label}")


def pick_random_file(folder: Path, extensions: List[str]) -> Path:
    items = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in set(extensions)
    ]
    if not items:
        raise FileNotFoundError(f"No matching files found in: {folder}")
    return random.choice(items)
