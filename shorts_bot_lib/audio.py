"""Audio processing: speed ramps, peak/RMS normalization."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment


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
    """Snap from slow → fast at end of highlight (cut), ramp into slow."""
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


def _two_pass_loudnorm(src: Path, dst: Path) -> bool:
    """Peak-normalize each SFX so no clip ends up quieter than the rest."""
    measure_cmd = (
        f"ffmpeg -hide_banner -nostats -i {shlex.quote(str(src))} "
        f"-af volumedetect -f null -"
    )
    try:
        result = subprocess.run(
            measure_cmd, shell=True, capture_output=True, text=True, check=False
        )
    except Exception as exc:
        print(f"Peak measure failed for {src.name}: {exc}")
        return False
    stderr = result.stderr or ""
    peak_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr)
    mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr)
    if not peak_match:
        print(f"Could not parse peak for {src.name}")
        return False

    peak_db = float(peak_match.group(1))
    mean_db = float(mean_match.group(1)) if mean_match else -25.0
    peak_boost_db = max(0.0, -1.0 - peak_db)
    target_rms_db = -14.0
    projected_rms = mean_db + peak_boost_db
    extra_rms_boost = max(0.0, target_rms_db - projected_rms)
    extra_rms_boost = min(extra_rms_boost, 8.0)
    total_boost_db = peak_boost_db + extra_rms_boost

    filter_chain = f"volume={total_boost_db:.2f}dB,alimiter=limit=0.97:level=false"
    apply_cmd = (
        f"ffmpeg -y -hide_banner -i {shlex.quote(str(src))} "
        f"-af {shlex.quote(filter_chain)} -ar 48000 -c:a libmp3lame -q:a 2 "
        f"{shlex.quote(str(dst))}"
    )
    try:
        subprocess.run(apply_cmd, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Normalize apply failed for {src.name}: {exc.stderr[:200] if exc.stderr else exc}")
        return False
    return True


def ensure_normalized_sounds(src_dir: Path, dst_dir: Path) -> Path:
    if not src_dir.exists():
        return src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg"}
    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in audio_exts:
            continue
        dst = dst_dir / (src.stem + ".mp3")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        print(f"Normalizing {src.name} ...")
        if not _two_pass_loudnorm(src, dst):
            shutil.copyfile(src, dst)
    return dst_dir
