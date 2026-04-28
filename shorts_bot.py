"""CLI entry point for the Shorts pipeline.

The actual implementation lives in `shorts_bot_lib/`. This file is a
thin orchestrator: parse args, wire pipeline stages together, and
print progress.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from shorts_bot_lib.audio import smart_speed_ramp, ensure_normalized_sounds
from shorts_bot_lib.images import (
    choose_popup_images,
    choose_story_related_popups,
    fetch_unsplash_hook_image,
    fixed_image_times,
    generate_hook_image,
    maybe_download_story_images,
    maybe_generate_images,
)
from shorts_bot_lib.runner import (
    ffmpeg_has_subtitles_filter,
    ffprobe_duration_seconds,
    pick_random_file,
    print_progress,
)
from shorts_bot_lib.script_ai import (
    generate_metadata,
    generate_script,
    summarize_script_for_image,
)
from shorts_bot_lib.subtitles import read_srt_segments, write_ass_from_segments
from shorts_bot_lib.text import get_highlight_timestamps
from shorts_bot_lib.transcribe import (
    get_whisper_word_timestamps,
    transcribe_audio_to_srt,
)
from shorts_bot_lib.types import PopupImage
from shorts_bot_lib.video import compose_video, pick_sfx_for_popups
from shorts_bot_lib.voiceover import (
    generate_voiceover_from_cloner_script,
    generate_voiceover_openai_tts,
)
from shorts_bot_lib.youtube_api import (
    get_youtube_credentials,
    post_pinned_comment,
    upload_to_youtube,
)

# Bump file-descriptor limit so ffmpeg with many overlay inputs has headroom.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, _hard))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and upload YouTube Shorts automatically.")
    parser.add_argument("--words", type=int, default=70, help="Approx script word count (70 \u2248 18\u201324s narrated)")
    parser.add_argument(
        "--topic",
        default="",
        help="Optional rant topic, e.g. people blasting speakerphone in public",
    )
    parser.add_argument(
        "--adam-cloner-script",
        default="",
        help="Optional path to Adam voice cloner run script (defaults to VoiceCloner/run_clone.sh)",
    )
    parser.add_argument(
        "--tts",
        default="cloner",
        choices=["cloner", "openai"],
        help="Narration engine: cloner (local Adam voice) or openai (cloud-friendly).",
    )
    parser.add_argument(
        "--dynamic-speed",
        action="store_true",
        help="Apply speed ramps around --double hyphenated-- phrases.",
    )
    parser.add_argument("--speed-ramp-ms", type=int, default=600)
    parser.add_argument("--speed-slow", type=float, default=0.60)
    parser.add_argument("--speed-fast", type=float, default=1.15)
    parser.add_argument("--upload", action="store_true", help="Upload the output video to YouTube")
    parser.add_argument(
        "--privacy",
        default="public",
        choices=["private", "unlisted", "public"],
        help="public = Shorts feed; private = no impressions",
    )
    parser.add_argument("--no-description", action="store_true", help="Upload with an empty YouTube description")
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate popup images using OpenAI image model",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only generate the hook image, then exit (no TTS/subtitles/video).",
    )
    parser.add_argument(
        "--script",
        default="",
        help="Optional script text to use for --images-only (otherwise uses output/script.txt).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional final video length in seconds (default: full narration length)",
    )
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--popup-sfx", default="assets/sounds/vine-boom.mp3")
    parser.add_argument("--popup-sfx-volume", type=float, default=0.55)
    parser.add_argument("--popup-sfx-speed", type=float, default=1.25)
    parser.add_argument("--popup-sfx-trim-seconds", type=float, default=1.4)
    parser.add_argument("--bgm-path", default="assets/Chopin - Nocturne op.9 No.2.mp3")
    parser.add_argument("--bgm-volume", type=float, default=0.08)
    parser.add_argument("--gameplay-top-crop", type=int, default=96)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a 3-second quick test for styling and video pipeline",
    )
    return parser


def _confirm_script_interactive(script: str) -> bool:
    """Return True if the user accepted the script (Y/empty), False to regenerate."""
    interactive = (
        sys.stdin.isatty()
        or os.environ.get("SHORTS_BOT_INTERACTIVE") == "1"
    )
    if not interactive:
        return True
    print("Use this script? (Y/N) ", flush=True)
    try:
        yn = input().strip().upper()
    except EOFError:
        yn = "Y"
    if yn in ("Y", "YES", ""):
        return True
    print("Regenerating...\n")
    return False


def _run_images_only(
    args: argparse.Namespace,
    client: OpenAI | None,
    gemini_key: str | None,
    output_dir: Path,
) -> None:
    script_file = output_dir / "script.txt"
    script = (args.script or "").strip()
    if not script and script_file.exists():
        script = script_file.read_text(encoding="utf-8").strip()
    if not script:
        raise RuntimeError(
            "No script found for --images-only. Provide --script or ensure output/script.txt exists."
        )
    print("Generating hook image...")
    hook_image_path = generate_hook_image(
        openai_client=client,
        gemini_api_key=gemini_key,
        script=script,
        out_dir=output_dir / "hook_image",
    )
    if hook_image_path is None or not hook_image_path.exists():
        hook_query = summarize_script_for_image(client, script)
        print(f"Falling back to Unsplash search: {hook_query!r}")
        hook_image_path = fetch_unsplash_hook_image(hook_query, output_dir / "hook_image")
    if hook_image_path is not None and hook_image_path.exists():
        print(f"Hook image saved: {hook_image_path}")
    print("Done (images-only).")


def main() -> None:
    load_dotenv()
    args = _build_arg_parser().parse_args()

    if args.images_only and not args.generate_images:
        args.generate_images = True

    if args.quick_test:
        args.words = 15
        args.duration_seconds = 3.0
        print("--- QUICK TEST MODE ENABLED (3 seconds) ---")

    if args.video_only and args.generate_images:
        raise RuntimeError("--video-only cannot be combined with --generate-images.")

    needs_openai = not args.video_only or args.generate_images

    api_key = os.environ.get("OPENAI_API_KEY")
    if needs_openai and not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    project_root = Path.cwd()
    gameplay_dir = project_root / "assets" / "gameplay"
    images_dir = project_root / "assets" / "popups"
    story_images_dir = project_root / "assets" / "story_images"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    gameplay_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    story_images_dir.mkdir(parents=True, exist_ok=True)

    if args.upload:
        get_youtube_credentials()

    client = OpenAI(api_key=api_key) if api_key else None

    if args.images_only:
        _run_images_only(args, client, gemini_key, output_dir)
        return

    gameplay_file = pick_random_file(gameplay_dir, ["mp4", "mov", "mkv", "webm"])

    total_steps = 6 + (1 if args.upload else 0)
    step = 1
    start_ts = time.time()

    # Step 1: script
    script_file = output_dir / "script.txt"
    if args.video_only:
        print_progress(step, total_steps, "Reusing existing script")
        if script_file.exists():
            script = script_file.read_text(encoding="utf-8").strip()
        else:
            script = "Crazy School Story You Won't Believe"
    else:
        while True:
            print_progress(step, total_steps, "Generating story script")
            script = generate_script(client, args.words, topic=args.topic)  # type: ignore[arg-type]
            clean_script = script.replace("*", "")
            print("\n--- Generated Script ---")
            print(clean_script)
            print("------------------------")
            if _confirm_script_interactive(script):
                break
        script_file.write_text(script, encoding="utf-8")

    # Step 2: voiceover
    step += 1
    narration_file = output_dir / "narration.mp3"
    narration_reference_segments: List[dict] = []
    if args.video_only or args.skip_tts:
        print_progress(step, total_steps, "Reusing existing narration")
        if not narration_file.exists():
            raw_narration_file = output_dir / "narration_raw.mp3"
            if raw_narration_file.exists():
                narration_file = raw_narration_file
            else:
                raise RuntimeError("Missing output narration file for reuse mode.")
    else:
        print_progress(step, total_steps, "Creating voiceover")
        raw_narration_file = output_dir / "narration_raw.mp3"
        if args.tts == "openai":
            if client is None:
                raise RuntimeError("OpenAI client missing; cannot use --tts openai.")
            generate_voiceover_openai_tts(client, script, raw_narration_file)
        else:
            generate_voiceover_from_cloner_script(
                script_text=script,
                out_audio_path=raw_narration_file,
                project_root=project_root,
                cloner_script=args.adam_cloner_script,
            )
        if args.dynamic_speed and re.search(r"--([^-][\s\S]*?[^-])--", script):
            if client is None:
                raise RuntimeError("OpenAI client missing; cannot use --dynamic-speed.")
            print("Applying dynamic speed ramps from --hyphen-wrapped-- phrases...")
            raw_words = get_whisper_word_timestamps(client, raw_narration_file, script)
            highlights = get_highlight_timestamps(script, raw_words)
            if highlights:
                smart_speed_ramp(
                    input_path=raw_narration_file,
                    output_path=narration_file,
                    interesting_segments=highlights,
                    ramp_ms=args.speed_ramp_ms,
                    slow_factor=args.speed_slow,
                    fast_factor=args.speed_fast,
                )
            else:
                print("No matching --hyphen-wrapped-- timestamps found; using raw narration.")
                shutil.copyfile(raw_narration_file, narration_file)
        else:
            narration_file = raw_narration_file
        if client is not None:
            narration_reference_segments = get_whisper_word_timestamps(client, narration_file, script)

    # Step 3: subtitles
    step += 1
    subtitle_file = output_dir / "subtitles.ass"
    emoji_events_file = output_dir / "emoji_events.json"
    subtitle_segments: List[dict] = []
    if args.video_only:
        print_progress(step, total_steps, "Reusing existing subtitles")
        old_srt_file = output_dir / "subtitles.srt"
        if client is not None:
            _, subtitle_segments = transcribe_audio_to_srt(
                client,
                narration_file,
                subtitle_file,
                script_text=script,
                reference_segments=narration_reference_segments or None,
            )  # type: ignore[arg-type]
        elif old_srt_file.exists():
            subtitle_segments = read_srt_segments(old_srt_file)
            write_ass_from_segments(subtitle_segments, subtitle_file)
        elif not subtitle_file.exists():
            raise RuntimeError("Missing output/subtitles.ass for --video-only mode.")
    else:
        print_progress(step, total_steps, "Building subtitles")
        generated_emoji_events, subtitle_segments = transcribe_audio_to_srt(
            client,
            narration_file,
            subtitle_file,
            script_text=script,
            reference_segments=narration_reference_segments or None,
        )  # type: ignore[arg-type]
        emoji_events_file.write_text(json.dumps(generated_emoji_events), encoding="utf-8")

    # Step 4: popup images
    step += 1
    print_progress(step, total_steps, "Preparing popup images")
    if args.generate_images:
        maybe_generate_images(
            client=client,  # type: ignore[arg-type]
            script=script,
            images_dir=images_dir,
            count=3,
            generate_images=True,
        )

    narration_duration = ffprobe_duration_seconds(narration_file)
    story_text_for_matching = script
    if subtitle_segments:
        story_text_for_matching += " " + " ".join(
            str(s.get("text") or s.get("raw_text") or "") for s in subtitle_segments
        )
    planned_image_times: List[float] = fixed_image_times(narration_duration, interval_seconds=2.5)
    maybe_download_story_images(
        story_images_dir, story_text_for_matching, client=client, min_count=18
    )
    popups = choose_story_related_popups(
        story_images_dir,
        story_text_for_matching,
        narration_duration,
        subtitle_segments=subtitle_segments,
        planned_times=planned_image_times,
        min_gap=2.0,
        max_gap=6.0,
    )
    if not popups:
        popups = choose_popup_images(images_dir, narration_duration, count=3)

    hook_image_path = generate_hook_image(
        openai_client=client,
        gemini_api_key=gemini_key,
        script=script,
        out_dir=output_dir / "hook_image",
    )
    if hook_image_path is None or not hook_image_path.exists():
        hook_query = summarize_script_for_image(client, script)
        print(f"Falling back to Unsplash search: {hook_query!r}")
        hook_image_path = fetch_unsplash_hook_image(hook_query, output_dir / "hook_image")
    if hook_image_path is None or not hook_image_path.exists():
        fallback_dirs = [story_images_dir, images_dir]
        fallback_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        fallback_candidates = []
        for folder in fallback_dirs:
            if folder.exists():
                fallback_candidates.extend(
                    p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in fallback_exts
                )
        if fallback_candidates:
            hook_image_path = random.choice(fallback_candidates)
            print(f"Using local fallback hook image: {hook_image_path.name}")
        else:
            hook_image_path = None
    if hook_image_path is not None and hook_image_path.exists():
        hook_duration = 1.5
        hook_width = 700
        hook_x = (1080 - hook_width) // 2
        hook_y = 860
        hook_popup = PopupImage(
            path=hook_image_path,
            start_sec=0.0,
            end_sec=min(narration_duration - 0.05, hook_duration),
            x=hook_x,
            y=hook_y,
            width=hook_width,
            play_sfx=False,
            use_fade=True,
        )
        popups = [hook_popup] + [p for p in popups if p.start_sec >= hook_popup.end_sec]

    normalized_sounds_dir = ensure_normalized_sounds(
        project_root / "assets" / "sounds",
        project_root / "assets" / "sounds_normalized",
    )
    pick_sfx_for_popups(popups, normalized_sounds_dir)

    burn_subtitles = ffmpeg_has_subtitles_filter()
    if not burn_subtitles:
        print("Subtitle burn-in unavailable in this ffmpeg build; exporting without burned subtitles.")

    # Step 5: render
    step += 1
    print_progress(step, total_steps, "Rendering short video")
    output_video = output_dir / "short.mp4"
    selected_start = compose_video(
        gameplay_path=gameplay_file,
        narration_path=narration_file,
        srt_path=subtitle_file,
        popup_images=popups,
        out_video_path=output_video,
        duration_seconds=args.duration_seconds,
        burn_subtitles=burn_subtitles,
        popup_sfx_path=Path(args.popup_sfx) if args.popup_sfx else None,
        popup_sfx_trim_seconds=args.popup_sfx_trim_seconds,
        popup_sfx_speed=args.popup_sfx_speed,
        popup_sfx_volume=args.popup_sfx_volume,
        bgm_path=Path(args.bgm_path) if args.bgm_path else None,
        bgm_volume=args.bgm_volume,
        source_top_crop=args.gameplay_top_crop,
    )

    # Step 6: metadata
    step += 1
    print_progress(step, total_steps, "Generating metadata")
    if client is not None:
        title, description = generate_metadata(
            client, script, include_description=not args.no_description
        )
    else:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook
        title = title or "Crazy School Story You Won't Believe"
        description = "" if args.no_description else (
            "Subscribe for more storytime shorts!\n#shorts #storytime #schoolstory"
        )
    if "#shorts" not in title.lower():
        title = f"{title} #Shorts"
    if description and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"
    tags = ["shorts", "storytime", "school story", "crazy story", "viral short"]

    metadata_file = output_dir / "metadata.txt"
    metadata_file.write_text(
        f"Title:\n{title}\n\nDescription:\n{description}\n\nTags: {', '.join(tags)}",
        encoding="utf-8",
    )
    print("\n--- YouTube Shorts Metadata ---")
    print(f"Title:\n{title}\n")
    print(f"Description:\n{description if description else '(empty)'}")

    print("\nDone.")
    print(f"Total run time: {time.time() - start_ts:.1f}s")
    print(f"Gameplay source: {gameplay_file.name}")
    print(f"Random start time: {selected_start:.2f}s")
    if args.duration_seconds is not None:
        print(f"Forced output duration: {args.duration_seconds:.2f}s")
    print(f"Output video: {output_video}")
    print(f"Script text: {script_file}")
    print(f"Subtitles: {subtitle_file}")
    if not burn_subtitles:
        print("Subtitles added as selectable subtitle track (not burned into pixels).")

    # Step 7 (optional): upload
    if args.upload:
        step += 1
        print_progress(step, total_steps, "Uploading to YouTube")
        video_url = upload_to_youtube(
            video_file=output_video,
            title=title,
            description=description,
            tags=tags,
            privacy=args.privacy,
            thumbnail_file=hook_image_path if hook_image_path and hook_image_path.exists() else None,
        )
        print(f"Uploaded: {video_url}")
        print("Note: Shorts can take 1\u20135 min to process before appearing in the feed.")
        from googleapiclient.discovery import build as _build
        _yt = _build("youtube", "v3", credentials=get_youtube_credentials())
        video_id = video_url.split("v=")[-1]
        post_pinned_comment(_yt, video_id, script)
    else:
        print("Upload skipped. Run with --upload to publish.")


if __name__ == "__main__":
    main()
