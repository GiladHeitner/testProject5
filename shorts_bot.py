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
    fixed_image_times,
    maybe_download_story_images,
    maybe_generate_images,
)
from shorts_bot_lib.runner import (
    ffmpeg_has_subtitles_filter,
    ffprobe_duration_seconds,
    pick_random_file,
    print_progress,
)
from shorts_bot_lib.keyword_popups import build_keyword_popups
from shorts_bot_lib.scene_assets import build_scene_popups
from shorts_bot_lib.script_ai import (
    generate_metadata,
    generate_script,
)
from shorts_bot_lib.subtitles import read_srt_segments, write_ass_from_segments
from shorts_bot_lib.text import get_highlight_timestamps, text_keywords
from shorts_bot_lib.transcribe import (
    get_whisper_word_timestamps,
    transcribe_audio_to_srt,
)
# Reddit card feature disabled; keep import commented for easy re-enable.
# from shorts_bot_lib.reddit_card import RedditCardSpec, render_reddit_card_png
from shorts_bot_lib.types import PopupImage


class _SkipFirstSentencePopup(Exception):
    """Sentinel used to bail out of the first-sentence guarantee when disabled."""
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
    parser.add_argument(
        "--no-scene-assets",
        action="store_true",
        help="Disable scene-aware Pexels/Unsplash popups and use the local "
             "reaction-image folders instead.",
    )
    parser.add_argument(
        "--no-keyword-popups",
        action="store_true",
        help="Disable LLM-picked keyword popups timed to spoken phrases.",
    )
    parser.add_argument(
        "--no-fallback-popups",
        action="store_true",
        help="Disable the local reaction-image fallback popups when scene "
             "assets and keyword popups produce nothing.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip every generation step and only upload the existing "
             "output/short.mp4 to YouTube using output/script.txt for "
             "metadata. Implies --upload.",
    )
    parser.add_argument(
        "--no-reddit-card",
        action="store_true",
        help="Skip the first-sentence guarantee popup. The Reddit hook card "
             "feature itself is disabled; this flag is kept for backward "
             "compatibility and now controls only the guaranteed first-sentence popup.",
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


def _run_upload_only(
    args: argparse.Namespace,
    client: OpenAI | None,
    output_dir: Path,
) -> None:
    """Upload an already-rendered output/short.mp4 to YouTube."""
    output_video = output_dir / "short.mp4"
    if not output_video.exists():
        raise RuntimeError(
            f"Cannot upload: {output_video} not found. Run a normal generation first."
        )
    script_file = output_dir / "script.txt"
    script = script_file.read_text(encoding="utf-8").strip() if script_file.exists() else ""

    print_progress(1, 2, "Generating metadata")
    title = ""
    description = ""
    if client is not None and script:
        title, description = generate_metadata(
            client, script, include_description=not args.no_description
        )
    if not title:
        hook = script.split(".")[0].strip() if script else ""
        title = (hook[:82] + "...") if len(hook) > 85 else (hook or "Crazy Story You Won't Believe")
    if "#shorts" not in title.lower():
        title = f"{title} #Shorts"
    if description and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"
    tags = ["shorts", "storytime", "school story", "crazy story", "viral short"]
    print(f"Title: {title}")
    print(f"Description: {description if description else '(empty)'}")

    print_progress(2, 2, "Uploading to YouTube")
    video_url = upload_to_youtube(
        video_file=output_video,
        title=title,
        description=description,
        tags=tags,
        privacy=args.privacy,
        thumbnail_file=None,
    )
    print(f"Uploaded: {video_url}")
    print("Note: Shorts can take 1\u20135 min to process before appearing in the feed.")
    if script:
        try:
            from googleapiclient.discovery import build as _build
            _yt = _build("youtube", "v3", credentials=get_youtube_credentials())
            video_id = video_url.split("v=")[-1]
            post_pinned_comment(_yt, video_id, script)
        except Exception as exc:
            print(f"Could not post pinned comment: {exc}")


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
    # Reddit card rendering disabled. To re-enable, uncomment the block below
    # and re-add the `reddit_card` import at the top of this file.
    # hook_text = script.split(".")[0].strip()
    # if not hook_text:
    #     hook_text = script.strip()
    # if not hook_text:
    #     raise RuntimeError("Script is empty; cannot render reddit card.")
    # out_path = render_reddit_card_png(
    #     title=hook_text,
    #     out_path=output_dir / "reddit_card" / "reddit_card.png",
    #     spec=RedditCardSpec(),
    # )
    # print(f"Reddit card saved: {out_path}")
    print("Done (images-only).")


def main() -> None:
    load_dotenv()
    args = _build_arg_parser().parse_args()

    def _norm_words(txt: str) -> list[str]:
        return [w for w in re.sub(r"[^a-z0-9 ]+", " ", txt.lower()).split() if w]

    def _find_spoken_window(phrase: str, word_segs: List[dict]) -> tuple[float, float] | None:
        """Return (start,end) when `phrase` is spoken using whisper word-level segments.

        Strategy:
        1. Try an exact contiguous match of the normalized phrase.
        2. If that fails, find the first occurrence of the title's first word
           and the first occurrence of the title's last word that comes after
           it -- this handles punctuation / minor transcription differences.
        3. Otherwise fall back to the first occurrence of the first word.
        """
        target = _norm_words(phrase)
        if not target or not word_segs:
            return None
        norm_words: list[tuple[str, float, float]] = []
        for seg in word_segs:
            w = str(seg.get("text") or seg.get("raw_text") or "").strip()
            nw = _norm_words(w)
            if not nw:
                continue
            try:
                s = float(seg["start"])
                e = float(seg["end"])
            except Exception:
                continue
            norm_words.append((nw[0], s, e))
        if not norm_words:
            return None

        window = len(target)
        if window <= len(norm_words):
            target_str = " ".join(target)
            for i in range(0, len(norm_words) - window + 1):
                chunk = " ".join(norm_words[i + j][0] for j in range(window))
                if chunk == target_str:
                    return norm_words[i][1], norm_words[i + window - 1][2]

        # Fuzzy match: first(first_word) -> first(last_word) after it.
        head = target[0]
        tail = target[-1]
        first_idx: int | None = None
        for i, (w, _s, _e) in enumerate(norm_words):
            if w == head:
                first_idx = i
                break
        if first_idx is not None and head != tail:
            for j in range(first_idx + 1, len(norm_words)):
                if norm_words[j][0] == tail:
                    return norm_words[first_idx][1], norm_words[j][2]

        # Fallback: first token only.
        if first_idx is not None:
            return norm_words[first_idx][1], norm_words[first_idx][2]
        return None

    if args.images_only and not args.generate_images:
        args.generate_images = True

    if args.quick_test:
        args.words = 15
        args.duration_seconds = 3.0
        print("--- QUICK TEST MODE ENABLED (3 seconds) ---")

    if args.video_only and args.generate_images:
        raise RuntimeError("--video-only cannot be combined with --generate-images.")

    if args.upload_only:
        args.upload = True

    needs_openai = (not args.video_only or args.generate_images) and not args.upload_only

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

    if args.upload_only:
        _run_upload_only(args, client, output_dir)
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

    popups: List[PopupImage] = []
    if not args.no_keyword_popups and client is not None and subtitle_segments:
        try:
            keyword_popups, keyword_mapping = build_keyword_popups(
                client=client,
                script_text=script,
                word_segments=subtitle_segments,
                narration_duration=narration_duration,
                out_dir=output_dir / "scene_assets",
            )
            popups = keyword_popups
            if keyword_mapping:
                (output_dir / "keyword_map.json").write_text(
                    json.dumps(keyword_mapping, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            print(f"Keyword popup pipeline failed: {exc}\nFalling back to scene assets.")

    if not popups and not args.no_scene_assets and client is not None:
        try:
            scene_popups, scene_mapping = build_scene_popups(
                client=client,
                script_text=script,
                narration_duration=narration_duration,
                out_dir=output_dir / "scene_assets",
            )
            popups = scene_popups
            (output_dir / "scene_map.json").write_text(
                json.dumps(scene_mapping, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Scene-asset pipeline failed: {exc}\nFalling back to local images.")

    if not popups and not args.no_fallback_popups:
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
            min_gap=0.0,
            max_gap=6.0,
        )
    if not popups and not args.no_fallback_popups:
        popups = choose_popup_images(images_dir, narration_duration, count=3)

    # Reddit-style hook card is disabled. Original implementation is preserved
    # below in a commented-out block in case we want to bring it back. The new
    # behavior only guarantees that one normal popup overlaps the spoken first
    # sentence; it does not strip other popups or rewrite subtitles.
    #
    # try:
    #     ...render_reddit_card_png(...)...
    #     popups.insert(0, PopupImage(path=card_path, ..., preserve_aspect=True))
    #     # drop/trim popups during card window, rebuild karaoke .ass, etc.
    # except Exception as exc:
    #     print(f"Reddit hook card generation failed: {exc}")

    # Guarantee at least one normal popup overlaps the spoken first sentence.
    if args.no_reddit_card:
        print("First-sentence popup guarantee disabled (--no-reddit-card).")
    try:
        if args.no_reddit_card:
            raise _SkipFirstSentencePopup()
        normalized_script = script.replace("\r\n", "\n").replace("\r", "\n")
        first_block = normalized_script.split("\n\n", 1)[0].strip()
        if not first_block or first_block == normalized_script.strip():
            first_block = re.split(r"(?<=[.!?])\s+", first_block, maxsplit=1)[0].strip()
        if len(first_block.split()) > 20:
            first_block = re.split(r"(?<=[.!?])\s+", first_block, maxsplit=1)[0].strip()
        first_sentence = first_block or script.split(".")[0].strip()

        if first_sentence:
            srt_word_segments: List[dict] = []
            srt_path = output_dir / "subtitles.srt"
            if srt_path.exists():
                try:
                    srt_word_segments = read_srt_segments(srt_path)
                except Exception as exc:
                    print(f"Could not read {srt_path}: {exc}")
            win = _find_spoken_window(first_sentence, srt_word_segments) if srt_word_segments else None
            if win is None and narration_reference_segments:
                win = _find_spoken_window(first_sentence, narration_reference_segments)
            if win is None and subtitle_segments:
                win = _find_spoken_window(first_sentence, subtitle_segments)

            tail = max(0.0, narration_duration - 0.05)
            if win is not None:
                t0 = max(0.0, float(win[0]))
                t1 = min(tail, float(win[1]))
            else:
                n_words = max(1, len(first_sentence.split()))
                est_dur = n_words * 0.42 + 0.5
                t0 = 0.0
                t1 = min(tail, est_dur)
            if t1 - t0 < 0.2:
                t1 = min(tail, t0 + 0.5)

            already_covered = any(p.start_sec < t1 and p.end_sec > t0 for p in popups)
            if already_covered:
                print(
                    f"First-sentence popup already covered by existing popup "
                    f"({t0:.2f}s -> {t1:.2f}s)."
                )
            else:
                hook_keys = text_keywords(first_sentence)
                exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
                chosen: Path | None = None
                story_candidates = [
                    p for p in story_images_dir.glob("*")
                    if p.is_file() and p.suffix.lower() in exts
                ]
                if story_candidates:
                    scored: list[tuple[int, float, Path]] = []
                    for p in story_candidates:
                        name_blob = f"{p.parent.name} {p.stem}".replace("_", " ").replace("-", " ")
                        overlap = len(hook_keys.intersection(text_keywords(name_blob)))
                        scored.append((overlap, random.random(), p))
                    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    chosen = scored[0][2]
                if chosen is None:
                    fallback_files = [
                        p for p in images_dir.glob("*")
                        if p.is_file() and p.suffix.lower() in exts
                    ]
                    if fallback_files:
                        chosen = random.choice(fallback_files)

                if chosen is None:
                    print("First-sentence guarantee: no local popup assets available; skipping.")
                else:
                    target_dur = 1.6
                    window_dur = max(0.0, t1 - t0)
                    if window_dur <= target_dur:
                        start_sec = t0
                        end_sec = t1
                    else:
                        center = 0.5 * (t0 + t1)
                        start_sec = max(t0, center - target_dur / 2)
                        end_sec = min(t1, start_sec + target_dur)
                    width = 700
                    popups.append(
                        PopupImage(
                            path=chosen,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            x=(1080 - width) // 2,
                            y=860,
                            width=width,
                            play_sfx=True,
                            use_fade=True,
                        )
                    )
                    print(
                        f"First-sentence popup guaranteed: {chosen.name} "
                        f"({start_sec:.2f}s -> {end_sec:.2f}s)"
                    )
        popups.sort(key=lambda p: p.start_sec)
    except _SkipFirstSentencePopup:
        pass
    except Exception as exc:
        print(f"First-sentence popup guarantee failed: {exc}")

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
            thumbnail_file=None,
        )
        print(f"Uploaded: {video_url}")
        print("Note: Shorts can take 1\u20135 min to process before appearing in the feed.")
        from googleapiclient.discovery import build as _build
        _yt = _build("youtube", "v3", credentials=get_youtube_credentials())
        video_id = video_url.split("v=")[-1]
        post_pinned_comment(_yt, video_id, script)
    else:
        print("Upload skipped. Run with --upload to publish.")

    # Cleanup intermediate image/video assets after the short is rendered (and uploaded).
    for sub in ("scene_assets", "hook_video"):
        target = output_dir / sub
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


if __name__ == "__main__":
    main()
