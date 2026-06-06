"""ffmpeg filter graph + final video render + popup SFX chain."""

from __future__ import annotations

import random
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Literal

from .runner import ffprobe_duration_seconds, run_ffmpeg_with_progress
from .types import PopupImage

GameplayFamily = Literal["roblox", "minecraft", "other"]

_GAMEPLAY_EXTENSIONS = ("mp4", "mov", "mkv", "webm")

# Gameplay files that should play at real-time speed (no 2x PTS ramp).
_REALTIME_GAMEPLAY_FILENAMES = frozenset(
    {
        "2026-05-16 14-56-45.mov",
    }
)
_DEFAULT_GAMEPLAY_SPEED = 2.0

GAMEPLAY_CREDIT_BLOCKS: dict[GameplayFamily, str] = {
    "roblox": (
        "Gameplay Credit: Dope Gameplays\n"
        "Roblox Parkour Gameplay No Copyright | Roblox Gameplay No Copyright | 33\n"
        "https://www.youtube.com/shorts/8Vo-3dhM7lM\n"
        "Licensed under Creative Commons Attribution."
    ),
    "minecraft": (
        "Gameplay Credit: Minecraft Parkour Gameplay\n"
        "Minecraft Gameplay No Copyright\n"
        "Licensed under Creative Commons Attribution."
    ),
    "other": (
        "Gameplay Credit: Background Gameplay\n"
        "Licensed under Creative Commons Attribution."
    ),
}


def classify_gameplay_family(gameplay_path: Path) -> GameplayFamily:
    name = gameplay_path.name.lower()
    if "roblox" in name:
        return "roblox"
    if "minecraft" in name:
        return "minecraft"
    return "other"


def gameplay_credit_block(gameplay_path: Path) -> str:
    return GAMEPLAY_CREDIT_BLOCKS[classify_gameplay_family(gameplay_path)]


def list_gameplay_files(folder: Path) -> List[Path]:
    ext_set = set(_GAMEPLAY_EXTENSIONS)
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in ext_set
    )


def pick_random_gameplay(folder: Path) -> Path:
    """Pick Roblox or Minecraft at random, then a file from that family."""
    items = list_gameplay_files(folder)
    if not items:
        raise FileNotFoundError(f"No matching files found in: {folder}")

    by_family: dict[GameplayFamily, List[Path]] = {}
    for path in items:
        by_family.setdefault(classify_gameplay_family(path), []).append(path)

    preferred = {
        family: paths
        for family, paths in by_family.items()
        if family in ("roblox", "minecraft")
    }
    pool = preferred or by_family
    family = random.choice(list(pool.keys()))
    return random.choice(pool[family])


def gameplay_speed_factor(gameplay_path: Path) -> float:
    """Return source consumption multiplier: 2.0 = 2x fast-forward, 1.0 = real-time."""
    name = gameplay_path.name
    if name in _REALTIME_GAMEPLAY_FILENAMES:
        return 1.0
    if "minecraft" in name.lower():
        return 1.0
    return _DEFAULT_GAMEPLAY_SPEED


def detect_content_crop(
    video_path: Path,
    start_time: float = 0.0,
    source_top_crop: int = 0,
) -> str | None:
    """Detect and remove letterboxing (OBS window padding) via cropdetect."""
    top_crop = max(0, int(source_top_crop))
    pre = f"crop=in_w:in_h-{top_crop}:0:{top_crop}," if top_crop > 0 else ""
    vf = f"{pre}cropdetect=24:16:0"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        f"{max(0.0, float(start_time)):.3f}",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        "45",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    stderr = proc.stderr or ""
    last: tuple[str, str, str, str] | None = None
    for line in stderr.splitlines():
        if "crop=" not in line:
            continue
        match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
        if match:
            last = match.groups()
    if last is None:
        return None
    w, h, x, y = (int(v) for v in last)
    return f"crop={w}:{h}:{x}:{y},"


def pick_sfx_for_popups(popups: List[PopupImage], sounds_dir: Path) -> None:
    from shorts_bot_lib.audio import POPUP_SFX_NAMES

    if not sounds_dir.exists():
        return
    all_sounds = [
        sounds_dir / (Path(name).stem + ".mp3")
        for name in POPUP_SFX_NAMES
        if (sounds_dir / (Path(name).stem + ".mp3")).exists()
    ]
    if not all_sounds:
        return
    last_sfx: Path | None = None
    for popup in popups:
        if not popup.play_sfx:
            continue
        if popup.sfx_path is not None:
            last_sfx = popup.sfx_path
            continue
        choices = [s for s in all_sounds if s != last_sfx] or all_sounds
        popup.sfx_path = random.choice(choices)
        last_sfx = popup.sfx_path


def _popup_sfx_gain(path: Path | None, base_volume: float) -> float:
    """Per-SFX multiplier — clicks and discord chimes sit under narration."""
    vol = max(0.0, float(base_volume))
    if path is None:
        return vol
    name = path.name.lower()
    if "mouse-click" in name:
        return vol * 0.45
    if "discord" in name:
        return vol * 0.35
    return vol


def build_popup_sfx_audio_chain(
    popups: List[PopupImage],
    sfx_input_indices: dict,
    sfx_trim_seconds: float = 1.4,
    sfx_speed: float = 1.25,
    sfx_volume: float = 0.15,
    narration_label: str = "narr",
) -> str:
    sfx_events = [p for p in popups if p.play_sfx and p.sfx_path is not None]
    if not sfx_events:
        return ""

    chains = [f"[{narration_label}]aresample=44100[abase]"]
    for i, popup in enumerate(sfx_events):
        sfx_idx = sfx_input_indices.get(str(popup.sfx_path))
        if sfx_idx is None:
            continue
        delay_ms = max(0, int(popup.start_sec * 1000))
        trim_s = max(0.15, float(sfx_trim_seconds))
        speed = min(2.0, max(0.5, float(sfx_speed)))
        vol = _popup_sfx_gain(popup.sfx_path, sfx_volume)
        # Some SFX files (notably mouse clicks) include leading silence which
        # makes them sound "late" vs the visual beat. Strip that before delay.
        pre = ""
        if popup.sfx_path is not None and "mouse-click" in popup.sfx_path.name.lower():
            pre = "silenceremove=start_periods=1:start_duration=0.005:start_threshold=-50dB,"
        chains.append(
            f"[{sfx_idx}:a]"
            f"{pre}atrim=0:{trim_s:.2f},asetpts=N/SR/TB,"
            f"atempo={speed:.2f},volume={vol:.2f},"
            f"adelay={delay_ms}|{delay_ms}"
            f"[boom{i}]"
        )
    mix_inputs = "[abase]" + "".join(f"[boom{i}]" for i in range(len(sfx_events)))
    chains.append(
        f"{mix_inputs}amix=inputs={1 + len(sfx_events)}:duration=first:normalize=0:dropout_transition=0[aout]"
    )
    return ";".join(chains)


def build_filter_complex(
    subtitle_path: Path,
    start_time: float,
    total_duration: float,
    popups: List[PopupImage],
    burn_subtitles: bool,
    hook_video_input_index: int | None = None,
    hook_video_duration: float = 1.5,
    hook_text: str | None = None,
    source_top_crop: int = 96,
    content_crop: str | None = None,
    gameplay_speed: float = _DEFAULT_GAMEPLAY_SPEED,
) -> str:
    chains = []
    top_crop = max(0, int(source_top_crop))
    crop_prefix = f"crop=in_w:in_h-{top_crop}:0:{top_crop}," if top_crop > 0 else ""
    letterbox_crop = content_crop or ""
    speed = max(1.0, float(gameplay_speed))
    source_span = total_duration * speed
    pts_chain = "setpts=PTS-STARTPTS,"
    if speed > 1.0:
        pts_chain += f"setpts={(1.0 / speed):.6f}*PTS,"
    # Center-crop a 9:16 portrait window, then scale to the Shorts canvas.
    chains.append(
        "[0:v]"
        f"{crop_prefix}"
        f"{letterbox_crop}"
        "crop=in_h*9/16:in_h:(iw-in_h*9/16)/2:0,"
        f"trim=start={start_time:.3f}:duration={source_span:.3f},"
        f"{pts_chain}"
        "scale=1080:1920:flags=lanczos,"
        "setsar=1,"
        "fps=60"
        "[v0]"
    )

    # Optional hook intro clip overlay (short video) at the beginning.
    # This is intentionally a small square (PiP), not full-screen.
    if hook_video_input_index is not None:
        dur = max(0.4, float(hook_video_duration))
        chains.append(
            f"[{hook_video_input_index}:v]"
            f"trim=duration={dur:.3f},setpts=PTS-STARTPTS,"
            "scale=700:700:force_original_aspect_ratio=increase,"
            "crop=700:700,"
            "fps=60"
            "[vh]"
        )
        if hook_text:
            safe_txt = (
                str(hook_text)
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )
            chains.append(
                "[vh]"
                "drawbox=x=0:y=0:w=700:h=150:color=black@0.55:t=fill,"
                f"drawtext=text='{safe_txt}':"
                "fontcolor=white:fontsize=58:borderw=6:bordercolor=black:"
                "x=(w-text_w)/2:y=42"
                "[vht]"
            )
            hook_src = "vht"
        else:
            hook_src = "vh"
        hook_x = (1080 - 700) // 2
        hook_y = 860
        chains.append(
            f"[v0][{hook_src}]overlay="
            f"x={hook_x}:y={hook_y}:enable='between(t,0,{dur:.3f})'"
            "[v0h]"
        )
        base_video = "v0h"
    else:
        base_video = "v0"

    for i, popup in enumerate(popups, start=0):
        popup_len = max(0.2, popup.end_sec - popup.start_sec)
        fade_dur = min(0.25, popup_len * 0.35)
        # Reddit card feature disabled; preserve aspect ratio still works via
        # PopupImage.preserve_aspect / is_emoji. The dedicated reddit_card.png
        # branch is left commented out for an easy re-enable.
        # is_reddit_card = popup.path.name.lower() == "reddit_card.png"
        scale_part = (
            f"scale={popup.width}:-1,"
            if (popup.is_emoji or popup.preserve_aspect)
            else f"scale={popup.width}:{popup.width}:force_original_aspect_ratio=increase,crop={popup.width}:{popup.width},"
        )
        is_gif = popup.path.suffix.lower() == ".gif"
        pts_prefix = ""
        if is_gif:
            gif_speed = max(1.0, popup.playback_speed)
            if gif_speed > 1.001:
                pts_prefix = (
                    f"setpts='(PTS-STARTPTS)/{gif_speed:.6f}+{popup.start_sec:.3f}/TB',"
                )
            else:
                pts_prefix = f"setpts=PTS-STARTPTS+{popup.start_sec:.3f}/TB,"
            fade_st = max(popup.start_sec, popup.end_sec - fade_dur)
        else:
            fade_st = max(popup.start_sec, popup.end_sec - fade_dur)
        filters_after_scale = "format=rgba"
        if popup.chroma_key:
            filters_after_scale += f",colorkey={popup.chroma_key}:0.32:0.08"
        if popup.use_fade:
            filters_after_scale += (
                f",fade=t=out:st={fade_st:.3f}:d={fade_dur:.3f}:alpha=1"
            )
        chains.append(f"[{i + 2}:v]{pts_prefix}{scale_part}{filters_after_scale}[img{i}]")

    current = base_video
    for i, popup in enumerate(popups, start=0):
        next_label = f"v{i + 1}"
        overlay_opts = (
            f"x={popup.x}:y={popup.y}:"
            f"enable='between(t,{popup.start_sec:.3f},{popup.end_sec:.3f})'"
        )
        if popup.path.suffix.lower() == ".gif":
            overlay_opts += ":repeatlast=0"
        chains.append(
            f"[{current}][img{i}]overlay={overlay_opts}[{next_label}]"
        )
        current = next_label

    if burn_subtitles:
        escaped_srt = (
            str(subtitle_path)
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
        )
        chains.append(f"[{current}]subtitles=filename={escaped_srt},setsar=1[vout]")
    else:
        chains.append(f"[{current}]setsar=1[vout]")
    return ";".join(chains)


def compose_video(
    gameplay_path: Path,
    narration_path: Path,
    srt_path: Path,
    popup_images: List[PopupImage],
    out_video_path: Path,
    hook_video_path: Path | None = None,
    hook_video_duration: float = 1.5,
    hook_text: str | None = None,
    duration_seconds: float | None = None,
    burn_subtitles: bool = True,
    popup_sfx_path: Path | None = None,
    popup_sfx_trim_seconds: float = 1.4,
    popup_sfx_speed: float = 1.25,
    popup_sfx_volume: float = 0.15,
    narration_volume: float = 2.7,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.08,
    source_top_crop: int = 96,
) -> float:
    narration_duration = ffprobe_duration_seconds(narration_path)
    target_duration = duration_seconds if duration_seconds is not None else narration_duration
    gameplay_duration = ffprobe_duration_seconds(gameplay_path)
    speed_factor = gameplay_speed_factor(gameplay_path)
    required_source_duration = target_duration * speed_factor
    unique_mux_token = str(time.time_ns())
    if speed_factor <= 1.0:
        print(f"Gameplay real-time speed: {gameplay_path.name}")
    else:
        print(f"Gameplay {speed_factor:.0f}x speed: {gameplay_path.name}")

    if gameplay_duration <= required_source_duration + 0.5:
        start_time = 0.0
    else:
        start_time = random.uniform(0.0, gameplay_duration - required_source_duration - 0.3)

    input_parts = [
        f"-i {shlex.quote(str(gameplay_path))}",
        f"-i {shlex.quote(str(narration_path))}",
    ]
    hook_video_input_index: int | None = None
    if hook_video_path is not None and hook_video_path.exists():
        input_parts.append(f"-i {shlex.quote(str(hook_video_path))}")
        hook_video_input_index = len(input_parts) - 1

    content_crop = detect_content_crop(
        gameplay_path, start_time=start_time, source_top_crop=source_top_crop
    )
    if content_crop:
        print(f"Gameplay letterbox crop: {content_crop.rstrip(',')}")
    filter_complex = build_filter_complex(
        subtitle_path=srt_path,
        start_time=start_time,
        total_duration=target_duration,
        popups=popup_images,
        burn_subtitles=burn_subtitles,
        hook_video_input_index=hook_video_input_index,
        hook_video_duration=hook_video_duration,
        hook_text=hook_text,
        source_top_crop=source_top_crop,
        content_crop=content_crop,
        gameplay_speed=speed_factor,
    )
    for popup in popup_images:
        if popup.path.suffix.lower() == ".gif":
            source_len = max(
                0.5,
                (popup.end_sec - popup.start_sec) * max(1.0, popup.playback_speed),
            )
            input_parts.append(
                f"-t {source_len:.3f} -i {shlex.quote(str(popup.path))}"
            )
        else:
            input_parts.append(
                f"-loop 1 -t {target_duration:.3f} -i {shlex.quote(str(popup.path))}"
            )
    sfx_input_indices: dict = {}
    unique_sfx_paths = []
    seen_sfx = set()
    for popup in popup_images:
        if popup.play_sfx and popup.sfx_path is not None and popup.sfx_path.exists():
            key = str(popup.sfx_path)
            if key not in seen_sfx:
                seen_sfx.add(key)
                unique_sfx_paths.append(popup.sfx_path)
    for sfx_path in unique_sfx_paths:
        input_parts.append(f"-i {shlex.quote(str(sfx_path))}")
        sfx_input_indices[str(sfx_path)] = len(input_parts) - 1
    bgm_input_index = -1
    if bgm_path is not None and bgm_path.exists():
        input_parts.append(f"-stream_loop -1 -i {shlex.quote(str(bgm_path))}")
        bgm_input_index = len(input_parts) - 1
    subtitle_input_index = len(input_parts)
    if not burn_subtitles and srt_path.exists():
        input_parts.append(f"-i {shlex.quote(str(srt_path))}")

    narr_vol = max(0.5, float(narration_volume))
    audio_prefix = f"[1:a]aresample=44100,volume={narr_vol:.2f}[narr]"
    filter_complex = f"{audio_prefix};{filter_complex}"

    current_audio_label = "narr"
    audio_map = "-map [narr] "
    if sfx_input_indices:
        audio_chain = build_popup_sfx_audio_chain(
            popup_images,
            sfx_input_indices,
            sfx_trim_seconds=popup_sfx_trim_seconds,
            sfx_speed=popup_sfx_speed,
            sfx_volume=popup_sfx_volume,
            narration_label=current_audio_label,
        )
        if audio_chain:
            filter_complex = f"{filter_complex};{audio_chain}"
            current_audio_label = "aout"
            audio_map = "-map [aout] "
    if bgm_input_index >= 0:
        bgm_vol = max(0.0, float(bgm_volume))
        bgm_chain = (
            f"[{current_audio_label}]aresample=44100[abase2];"
            f"[{bgm_input_index}:a]silenceremove=start_periods=1:start_duration=0.03:start_threshold=-45dB,atrim=0:{target_duration:.3f},asetpts=N/SR/TB,volume={bgm_vol:.3f}[bgm];"
            f"[abase2][bgm]amix=inputs=2:duration=first:normalize=0:dropout_transition=0[aoutmix]"
        )
        filter_complex = f"{filter_complex};{bgm_chain}"
        audio_map = "-map [aoutmix] "

    # Single-threaded filter graph keeps macOS pthread/swscale from exhausting.
    thread_flags = "-filter_threads 1 -filter_complex_threads 1 -threads 2 "
    progress_flags = "-progress pipe:1 -nostats "
    cmd = (
        "ffmpeg -y "
        + thread_flags
        + progress_flags
        + " ".join(input_parts)
        + " "
        + f"-filter_complex {shlex.quote(filter_complex)} "
        + f"-map [vout] {audio_map}"
        + "-map_metadata -1 -map_chapters -1 "
        + "-c:v libx264 -preset medium -crf 20 -r 60 "
        + "-c:a aac -b:a 160k "
        + f"-metadata comment={shlex.quote(unique_mux_token)} "
        + "-movflags +faststart "
        + f"-t {target_duration:.3f} "
        + shlex.quote(str(out_video_path))
    )
    if not burn_subtitles and srt_path.exists():
        cmd = (
            "ffmpeg -y "
            + thread_flags
            + progress_flags
            + " ".join(input_parts)
            + " "
            + f"-filter_complex {shlex.quote(filter_complex)} "
            + f"-map [vout] {audio_map}-map {subtitle_input_index}:0 "
            + "-map_metadata -1 -map_chapters -1 "
            + "-c:v libx264 -preset medium -crf 20 -r 60 "
            + "-c:a aac -b:a 160k "
            + "-c:s mov_text "
            + "-metadata:s:s:0 language=eng "
            + f"-metadata comment={shlex.quote(unique_mux_token)} "
            + "-movflags +faststart "
            + f"-t {target_duration:.3f} "
            + shlex.quote(str(out_video_path))
        )
    print(f"[step] begin render duration={target_duration:.2f}s", flush=True)
    run_ffmpeg_with_progress(cmd, total_duration=target_duration)
    print("[step] end render", flush=True)
    return start_time
