"""ffmpeg filter graph + final video render + popup SFX chain."""

from __future__ import annotations

import random
import shlex
import time
from pathlib import Path
from typing import List

from .runner import ffprobe_duration_seconds, run_ffmpeg_with_progress
from .types import PopupImage


def pick_sfx_for_popups(popups: List[PopupImage], sounds_dir: Path) -> None:
    if not sounds_dir.exists():
        return
    blocked_keywords = {"fahhh", "taco", "bell"}
    all_sounds = sorted(
        p for p in sounds_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg"}
        and not any(kw in p.name.lower() for kw in blocked_keywords)
    )
    if not all_sounds:
        return
    last_sfx: Path | None = None
    for popup in popups:
        if not popup.play_sfx:
            continue
        choices = [s for s in all_sounds if s != last_sfx] or all_sounds
        popup.sfx_path = random.choice(choices)
        last_sfx = popup.sfx_path


def build_popup_sfx_audio_chain(
    popups: List[PopupImage],
    sfx_input_indices: dict,
    sfx_trim_seconds: float = 1.4,
    sfx_speed: float = 1.25,
    sfx_volume: float = 0.2,
) -> str:
    sfx_events = [p for p in popups if p.play_sfx and p.sfx_path is not None]
    if not sfx_events:
        return ""

    chains = ["[1:a]aresample=44100,volume=1.0[abase]"]
    for i, popup in enumerate(sfx_events):
        sfx_idx = sfx_input_indices.get(str(popup.sfx_path))
        if sfx_idx is None:
            continue
        delay_ms = max(0, int(popup.start_sec * 1000))
        trim_s = max(0.15, float(sfx_trim_seconds))
        speed = min(2.0, max(0.5, float(sfx_speed)))
        vol = max(0.0, float(sfx_volume))
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
) -> str:
    chains = []
    top_crop = max(0, int(source_top_crop))
    crop_prefix = f"crop=in_w:in_h-{top_crop}:0:{top_crop}," if top_crop > 0 else ""
    chains.append(
        "[0:v]"
        f"{crop_prefix}"
        f"trim=start={start_time:.3f}:duration={total_duration * 4.0:.3f},"
        "setpts=PTS-STARTPTS,"
        "setpts=0.25*PTS,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
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

    overlay_specs: List[tuple[str, str]] = []
    for i, popup in enumerate(popups, start=0):
        popup_len = max(0.2, popup.end_sec - popup.start_sec)
        fade_dur = min(0.25, popup_len * 0.35)
        fade_out_start = max(popup.start_sec, popup.end_sec - fade_dur)
        is_reddit_card = popup.path.name.lower() == "reddit_card.png"

        if is_reddit_card:
            # Slowly grow the reddit card while keeping its center fixed.
            try:
                from PIL import Image as _Image
                with _Image.open(popup.path) as _im:
                    iw_in, ih_in = _im.size
            except Exception:
                iw_in, ih_in = popup.width, popup.width
            initial_h = popup.width * ih_in / max(1, iw_in)
            cx = popup.x + popup.width / 2.0
            cy = popup.y + initial_h / 2.0
            grow_rate = 0.04
            start = popup.start_sec
            growth = f"(1+{grow_rate}*max(0\\,t-{start:.3f}))"
            scale_part = (
                f"scale=eval=frame:"
                f"w='{popup.width}*{growth}':"
                f"h='{int(round(initial_h))}*{growth}',"
            )
            overlay_x = f"({cx:.2f}-w/2)"
            overlay_y = f"({cy:.2f}-h/2)"
        else:
            scale_part = (
                f"scale={popup.width}:-1,"
                if (popup.is_emoji or popup.preserve_aspect)
                else f"scale={popup.width}:{popup.width}:force_original_aspect_ratio=increase,crop={popup.width}:{popup.width},"
            )
            overlay_x = f"{popup.x}"
            overlay_y = f"{popup.y}"
        overlay_specs.append((overlay_x, overlay_y))
        chains.append(
            (
                f"[{i + 2}:v]"
                + scale_part
                + f"format=rgba,"
                  f"fade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f}:alpha=1"
                  f"[img{i}]"
            )
            if popup.use_fade
            else (
                f"[{i + 2}:v]"
                + scale_part
                + f"format=rgba"
                  f"[img{i}]"
            )
        )

    current = base_video
    for i, popup in enumerate(popups, start=0):
        next_label = f"v{i + 1}"
        ox, oy = overlay_specs[i]
        chains.append(
            f"[{current}][img{i}]overlay="
            f"x='{ox}':y='{oy}':"
            f"enable='between(t,{popup.start_sec:.3f},{popup.end_sec:.3f})'"
            f"[{next_label}]"
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
        chains.append(f"[{current}]subtitles=filename={escaped_srt}[vout]")
    else:
        chains.append(f"[{current}]null[vout]")
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
    popup_sfx_volume: float = 0.55,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.08,
    source_top_crop: int = 96,
) -> float:
    narration_duration = ffprobe_duration_seconds(narration_path)
    target_duration = duration_seconds if duration_seconds is not None else narration_duration
    gameplay_duration = ffprobe_duration_seconds(gameplay_path)
    required_source_duration = target_duration * 4.0
    unique_mux_token = str(time.time_ns())

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
    )
    for popup in popup_images:
        if popup.path.suffix.lower() == ".gif":
            input_parts.append(
                f"-ignore_loop 0 -t {target_duration:.3f} -i {shlex.quote(str(popup.path))}"
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
    if not unique_sfx_paths and popup_sfx_path is not None and popup_sfx_path.exists() and any(
        p.play_sfx for p in popup_images
    ):
        unique_sfx_paths.append(popup_sfx_path)
        for popup in popup_images:
            if popup.play_sfx and popup.sfx_path is None:
                popup.sfx_path = popup_sfx_path
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

    current_audio_label = "1:a"
    audio_map = "-map 1:a "
    if sfx_input_indices:
        audio_chain = build_popup_sfx_audio_chain(
            popup_images,
            sfx_input_indices,
            sfx_trim_seconds=popup_sfx_trim_seconds,
            sfx_speed=popup_sfx_speed,
            sfx_volume=popup_sfx_volume,
        )
        if audio_chain:
            filter_complex = f"{filter_complex};{audio_chain}"
            current_audio_label = "aout"
            audio_map = "-map [aout] "
    if bgm_input_index >= 0:
        bgm_vol = max(0.0, float(bgm_volume))
        bgm_chain = (
            f"[{current_audio_label}]aresample=44100,volume=1.0[abase2];"
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
        + "-shortest "
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
            + "-shortest "
            + shlex.quote(str(out_video_path))
        )
    print(f"[step] begin render duration={target_duration:.2f}s", flush=True)
    run_ffmpeg_with_progress(cmd, total_duration=target_duration)
    print("[step] end render", flush=True)
    return start_time
