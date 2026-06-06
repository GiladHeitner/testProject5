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
    print_progress,
)
from shorts_bot_lib.video import (
    classify_gameplay_family,
    gameplay_credit_block,
    pick_random_gameplay,
)
from shorts_bot_lib.channel_persona import (
    load_channel_persona,
    persona_summary,
)
from shorts_bot_lib.keyword_popups import build_keyword_popups
from shorts_bot_lib.subscribe_cta import apply_subscribe_cta
from shorts_bot_lib.scene_assets import Scene, _fetch_scene_image, build_scene_popups
from shorts_bot_lib.script_ai import (
    MUSLIM_SHORT_TAGS,
    generate_metadata,
    generate_script,
)
from shorts_bot_lib.subtitles import read_srt_segments, write_ass_from_segments
from shorts_bot_lib.text import get_highlight_timestamps, strip_speed_ramp_hyphens
from shorts_bot_lib.transcribe import (
    get_whisper_word_timestamps,
    transcribe_audio_to_srt,
)
# Reddit card feature disabled; keep import commented for easy re-enable.
# from shorts_bot_lib.reddit_card import RedditCardSpec, render_reddit_card_png
from shorts_bot_lib.types import PopupImage
from shorts_bot_lib.video import compose_video, pick_sfx_for_popups
from shorts_bot_lib.voiceover import (
    generate_voiceover_from_cloner_script,
    generate_voiceover_openai_tts,
)
from shorts_bot_lib.youtube_api import (
    append_upload_registry,
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
        help="Optional topic text (short prompt or full Reddit post body).",
    )
    parser.add_argument(
        "--topic-file",
        default="",
        metavar="PATH",
        help="Read topic from a file (used for long Reddit posts in CI).",
    )
    parser.add_argument(
        "--reddit-topic",
        action="store_true",
        help="Fetch a top weekly text post from Reddit (PRAW) and use it as --topic.",
    )
    parser.add_argument(
        "--adam-cloner-script",
        default="",
        help="Optional path to voice cloner run script (defaults to VoiceCloner/run_clone.sh)",
    )
    parser.add_argument(
        "--tts",
        default="cloner",
        choices=["cloner", "openai"],
        help="Narration engine: cloner (local Omar voice) or openai (cloud-friendly).",
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
        help="Use this narration text instead of generating a script (full pipeline or --images-only).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional final video length in seconds (default: full narration length)",
    )
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--popup-sfx", default="assets/sounds/mouse-click-sound.mp3")
    parser.add_argument("--popup-sfx-volume", type=float, default=0.15)
    parser.add_argument("--popup-sfx-speed", type=float, default=1.25)
    parser.add_argument("--popup-sfx-trim-seconds", type=float, default=1.4)
    parser.add_argument(
        "--narration-volume",
        type=float,
        default=2.7,
        help="Voice-over gain in the final mix (default 2.7).",
    )
    parser.add_argument(
        "--no-popup-sfx",
        action="store_true",
        help="Disable random popup sound effects. The opening popup still "
             "plays assets/discord-notification.mp3 when present.",
    )
    parser.add_argument("--bgm-path", default="assets/BackgroundMusic.mp3")
    parser.add_argument("--bgm-volume", type=float, default=0.08)
    parser.add_argument(
        "--gameplay-path",
        default=None,
        help="Gameplay video file (default: random pick from assets/gameplay).",
    )
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
        "--no-subscribe-cta",
        action="store_true",
        help="Disable the subscribe-button GIF overlay at the end of the script.",
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
        "--persona-file",
        default="",
        metavar="PATH",
        help="Channel host persona JSON (default: assets/channel_persona.json).",
    )
    parser.add_argument(
        "--no-reddit-card",
        action="store_true",
        help="Skip the opening hook popup at t=0 (and its Discord chime). "
             "Kept for backward compatibility as --no-reddit-card.",
    )
    return parser


def _confirm_script_interactive(script: str, *, auto_accept: bool = False) -> bool:
    """Return True if the user accepted the script (Y/empty), False to regenerate."""
    if auto_accept:
        return True
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS"):
        return True
    interactive = (
        sys.stdin.isatty()
        and os.environ.get("SHORTS_BOT_INTERACTIVE") == "1"
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


_OPENING_POPUP_DURATION_SEC = 1.0


def _first_sentence_from_script(script: str) -> str:
    normalized = script.replace("\r\n", "\n").replace("\r", "\n")
    first_block = normalized.split("\n\n", 1)[0].strip()
    if not first_block or first_block == normalized.strip():
        first_block = re.split(r"(?<=[.!?])\s+", first_block, maxsplit=1)[0].strip()
    if len(first_block.split()) > 20:
        first_block = re.split(r"(?<=[.!?])\s+", first_block, maxsplit=1)[0].strip()
    return first_block or script.split(".")[0].strip()


_HOOK_STOCK_QUERY_SYSTEM = (
    "You write 2-4 word stock-photo search queries for Muslim/Arab teen storytime "
    "YouTube Shorts hook images (hijab, mosque, airport security, family dinner, "
    "school hallway, yearbook, etc.). Use visual nouns and adjectives only — no "
    "proper names, no verbs, no quotes."
)


def _hook_stock_search_query(phrase: str, client: OpenAI | None) -> str:
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("KEYWORD_LLM_MODEL", "gpt-4o-mini"),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _HOOK_STOCK_QUERY_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            'Return JSON: {"query": "<2-4 word stock photo query>"}\n\n'
                            f"Hook narration:\n{phrase.strip()}"
                        ),
                    },
                ],
                temperature=0.4,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            query = re.sub(r"[^a-zA-Z0-9 ]", " ", str(data.get("query") or "")).strip()
            if query:
                return query
        except Exception as exc:
            print(f"Hook stock query LLM failed: {exc}")
    tokens = [
        w for w in re.sub(r"[^a-z0-9 ]+", " ", phrase.lower()).split() if len(w) > 2
    ]
    stop = {
        "the", "and", "that", "with", "this", "from", "your", "just", "were",
        "have", "what", "when", "they", "them", "then", "into", "over", "about",
        "there", "would", "could", "hate", "like", "people", "who",
    }
    visual = [t for t in tokens if t not in stop][:4]
    return " ".join(visual) if visual else "dramatic scene"


def _fetch_hook_stock_image(
    phrase: str,
    client: OpenAI | None,
    output_dir: Path,
) -> Path | None:
    """Fetch a Pexels/Unsplash stock photo for the hook (not local reaction images)."""
    if client is None:
        print("Opening popup: OPENAI_API_KEY required for stock image search.")
        return None
    pexels_key = os.environ.get("PEXELS_API_KEY")
    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not pexels_key and not unsplash_key:
        print("Opening popup: set PEXELS_API_KEY or UNSPLASH_ACCESS_KEY.")
        return None
    query = _hook_stock_search_query(phrase, client)
    print(f"Fetching hook stock image (Pexels/Unsplash): {query!r}")
    scene = Scene(
        index=0,
        text=phrase,
        query=query,
        word_count=max(1, len(query.split())),
    )
    path, _source = _fetch_scene_image(
        scene=scene,
        out_dir=output_dir / "hook_popup",
        openai_client=client,
        pexels_key=pexels_key,
        unsplash_key=unsplash_key,
        gemini_key=os.environ.get("GEMINI_API_KEY"),
    )
    return path


def _ensure_opening_popup_at_start(
    popups: List[PopupImage],
    *,
    script: str,
    client: OpenAI | None,
    narration_duration: float,
    output_dir: Path,
) -> PopupImage | None:
    """Guarantee a popup at t=0 (opening hook). Caller assigns Discord SFX to it."""
    phrase = _first_sentence_from_script(script)
    if not phrase:
        return None
    chosen = _fetch_hook_stock_image(phrase, client, output_dir)
    if chosen is None:
        print("Opening popup: stock image fetch failed; skipping.")
        return None

    start_sec = 0.0
    end_sec = min(max(0.0, narration_duration - 0.05), _OPENING_POPUP_DURATION_SEC)
    width = 700

    opening = PopupImage(
        path=chosen,
        start_sec=start_sec,
        end_sec=end_sec,
        x=(1080 - width) // 2,
        y=860,
        width=width,
        play_sfx=True,
        use_fade=True,
    )
    popups.append(opening)
    popups.sort(key=lambda p: p.start_sec)
    print(
        f"Opening popup at video start: {chosen.name} "
        f"(0.00s -> {end_sec:.2f}s)"
    )
    return opening


def _drop_hook_overlapping_popups(
    popups: List[PopupImage],
    opening: PopupImage,
    *,
    gap_sec: float = 0.2,
) -> List[PopupImage]:
    """Drop keyword popups that would overlap the opening hook window."""
    hook_cutoff = float(opening.end_sec) + gap_sec
    kept: List[PopupImage] = []
    for popup in popups:
        if popup is opening:
            kept.append(popup)
            continue
        if popup.start_sec < hook_cutoff:
            print(
                f"Dropping keyword popup overlapping hook: "
                f"{popup.path.name} @ {popup.start_sec:.2f}s"
            )
            continue
        kept.append(popup)
    return kept


def _resolve_persona_path(args: argparse.Namespace, project_root: Path) -> Path | None:
    raw = (getattr(args, "persona_file", None) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _load_channel_persona(args: argparse.Namespace, project_root: Path):
    return load_channel_persona(_resolve_persona_path(args, project_root), project_root=project_root)


def _save_persona_snapshot(persona, output_dir: Path) -> None:
    from dataclasses import asdict

    snapshot = output_dir / "persona.json"
    snapshot.write_text(json.dumps(asdict(persona), indent=2) + "\n", encoding="utf-8")


def _save_gameplay_snapshot(gameplay_file: Path, output_dir: Path) -> None:
    snapshot = output_dir / "gameplay.json"
    snapshot.write_text(
        json.dumps(
            {
                "file": gameplay_file.name,
                "family": classify_gameplay_family(gameplay_file),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_gameplay_credit(output_dir: Path, project_root: Path) -> str:
    snapshot = output_dir / "gameplay.json"
    if snapshot.exists():
        try:
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            name = str(data.get("file") or "").strip()
            if name:
                return gameplay_credit_block(project_root / "assets" / "gameplay" / name)
        except (json.JSONDecodeError, OSError):
            pass
    return ""


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
    script = strip_speed_ramp_hyphens(
        script_file.read_text(encoding="utf-8").strip() if script_file.exists() else ""
    )
    project_root = Path.cwd()
    channel_persona = _load_channel_persona(args, project_root)
    gameplay_credit = _load_gameplay_credit(output_dir, project_root)

    print_progress(1, 2, "Generating metadata")
    title = ""
    description = ""
    if client is not None and script:
        title, description = generate_metadata(
            client,
            script,
            include_description=not args.no_description,
            persona=channel_persona,
            gameplay_credit=gameplay_credit,
        )
    if not title:
        hook = script.split(".")[0].strip() if script else ""
        title = (hook[:82] + "...") if len(hook) > 85 else (hook or "Crazy Story You Won't Believe")
    if "#shorts" not in title.lower():
        title = f"{title} #Shorts"
    if description and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"
    tags = list(MUSLIM_SHORT_TAGS)
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
            post_pinned_comment(_yt, video_id, script, client=client)
            append_upload_registry(video_id, title=title, script=script)
        except Exception as exc:
            print(f"Could not post pinned comment: {exc}")


def _run_images_only(
    args: argparse.Namespace,
    client: OpenAI | None,
    gemini_key: str | None,
    output_dir: Path,
) -> None:
    script_file = output_dir / "script.txt"
    script = strip_speed_ramp_hyphens((args.script or "").strip())
    if not script and script_file.exists():
        script = strip_speed_ramp_hyphens(script_file.read_text(encoding="utf-8").strip())
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

    if args.topic_file:
        topic_path = Path(args.topic_file)
        if not topic_path.is_absolute():
            topic_path = project_root / topic_path
        if not topic_path.exists():
            raise FileNotFoundError(f"Topic file not found: {topic_path}")
        args.topic = topic_path.read_text(encoding="utf-8").strip()
    if args.reddit_topic:
        from shorts_bot_lib.reddit_topics import fetch_topic_for_pipeline, mark_post_used

        used_reddit = project_root / ".github" / "used_reddit.txt"
        topic_text, post_id = fetch_topic_for_pipeline(used_file=used_reddit)
        args.topic = topic_text
        mark_post_used(post_id, used_reddit)
        (project_root / ".github").mkdir(parents=True, exist_ok=True)
        (project_root / ".github" / "reddit_post_id.txt").write_text(
            post_id, encoding="utf-8"
        )
        (project_root / ".github" / "reddit_topic.txt").write_text(
            topic_text, encoding="utf-8"
        )

    if args.topic.strip() and not args.reddit_topic:
        from shorts_bot_lib.reddit_topics import matches_host_persona, matches_muslim_arab_niche

        if not matches_muslim_arab_niche(args.topic):
            print(
                "[topic] Warning: topic may not match Muslim/Arab niche — "
                "script will still follow channel prompts.",
                file=sys.stderr,
            )
        if not matches_host_persona(args.topic):
            print(
                "[topic] Warning: topic may not fit the channel host persona — "
                "consider a male-host-friendly story.",
                file=sys.stderr,
            )

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

    if args.gameplay_path:
        gameplay_file = Path(args.gameplay_path)
        if not gameplay_file.is_absolute():
            gameplay_file = project_root / gameplay_file
        if not gameplay_file.exists():
            raise FileNotFoundError(f"Gameplay file not found: {gameplay_file}")
    else:
        gameplay_file = pick_random_gameplay(gameplay_dir)
    _save_gameplay_snapshot(gameplay_file, output_dir)

    total_steps = 6 + (1 if args.upload else 0)
    step = 1
    start_ts = time.time()
    channel_persona = _load_channel_persona(args, project_root)
    print(f"Channel host: {persona_summary(channel_persona)}")

    # Step 1: script
    script_file = output_dir / "script.txt"
    provided_script = strip_speed_ramp_hyphens((args.script or "").strip())
    if args.video_only:
        print_progress(step, total_steps, "Reusing existing script")
        if script_file.exists():
            script = strip_speed_ramp_hyphens(script_file.read_text(encoding="utf-8").strip())
        else:
            script = "Muslim Teen Story You Won't Believe"
    elif provided_script:
        print_progress(step, total_steps, "Using custom script")
        script = provided_script
        print("\n--- Custom Script ---")
        print(script.replace("*", ""))
        print("---------------------")
    else:
        while True:
            print_progress(step, total_steps, "Generating story script")
            script = generate_script(
                client,
                args.words,
                topic=args.topic,
                persona=channel_persona,
            )  # type: ignore[arg-type]
            clean_script = script.replace("*", "")
            print("\n--- Generated Script ---")
            print(clean_script)
            print("------------------------")
            if _confirm_script_interactive(
                script,
                auto_accept=bool(args.reddit_topic or args.topic_file),
            ):
                break
    script = strip_speed_ramp_hyphens(script)
    script_file.write_text(script, encoding="utf-8")
    _save_persona_snapshot(channel_persona, output_dir)

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
        if client is not None:
            narration_reference_segments = get_whisper_word_timestamps(
                client, narration_file, script
            )
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
            keyword_word_segments = subtitle_segments
            if narration_reference_segments:
                keyword_word_segments = [
                    {
                        "text": str(s.get("text") or "").strip(),
                        "raw_text": str(s.get("text") or "").strip(),
                        "start": float(s["start"]),
                        "end": float(max(float(s["end"]), float(s["start"]) + 0.05)),
                    }
                    for s in narration_reference_segments
                    if str(s.get("text") or "").strip()
                ]
            keyword_popups, keyword_mapping = build_keyword_popups(
                client=client,
                script_text=script,
                word_segments=keyword_word_segments,
                narration_duration=narration_duration,
                out_dir=output_dir / "scene_assets",
                hook_end_sec=_OPENING_POPUP_DURATION_SEC,
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

    if not args.no_subscribe_cta and subtitle_segments:
        cta_word_segments: List[dict] = subtitle_segments
        if narration_reference_segments:
            cta_word_segments = [
                {
                    "text": str(s.get("text") or "").strip(),
                    "raw_text": str(s.get("text") or "").strip(),
                    "start": float(s["start"]),
                    "end": float(max(float(s["end"]), float(s["start"]) + 0.05)),
                }
                for s in narration_reference_segments
                if str(s.get("text") or "").strip()
            ]
        subscribe_gif = project_root / "assets" / "youtubebutton.gif"
        if subscribe_gif.is_file():
            popups = apply_subscribe_cta(
                popups,
                word_segments=cta_word_segments,
                narration_duration=narration_duration,
                gif_path=subscribe_gif,
            )
        else:
            print(f"Subscribe CTA: GIF not found at {subscribe_gif}")

    # Opening hook popup at t=0; Discord notification SFX is wired to it below.
    opening_popup: PopupImage | None = None
    try:
        opening_popup = _ensure_opening_popup_at_start(
            popups,
            script=script,
            client=client,
            narration_duration=narration_duration,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Opening popup failed: {exc}")

    if opening_popup is not None:
        opening_popup.start_sec = 0.0
        popups = _drop_hook_overlapping_popups(popups, opening_popup)

    normalized_sounds_dir = ensure_normalized_sounds(
        project_root / "assets" / "sounds",
        project_root / "assets" / "sounds_normalized",
    )
    subscribe_popup = next(
        (p for p in reversed(popups) if p.path.name.lower() == "youtubebutton.gif"),
        None,
    )
    click_sfx = normalized_sounds_dir / "mouse-click-sound.mp3"
    if subscribe_popup is not None and click_sfx.is_file():
        subscribe_popup.play_sfx = True
        subscribe_popup.sfx_path = click_sfx.resolve()
        print(f"Click SFX on subscribe CTA: {click_sfx.name}")
    if args.no_popup_sfx:
        for popup in popups:
            if popup is opening_popup or popup is subscribe_popup:
                continue
            popup.play_sfx = False
            popup.sfx_path = None
    discord_sfx = project_root / "assets" / "discord-notification.mp3"
    discord_target = opening_popup
    if discord_target is None and popups:
        discord_target = min(popups, key=lambda p: (p.start_sec, p.end_sec))
    if discord_target is not None and discord_sfx.exists():
        discord_target.start_sec = 0.0
        discord_target.play_sfx = True
        discord_target.sfx_path = discord_sfx.resolve()
        print(f"Discord SFX on opening popup: {discord_target.path.name}")
    if not args.no_popup_sfx:
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
        duration_seconds=args.duration_seconds or narration_duration,
        burn_subtitles=burn_subtitles,
        popup_sfx_path=Path(args.popup_sfx) if args.popup_sfx else None,
        popup_sfx_trim_seconds=args.popup_sfx_trim_seconds,
        popup_sfx_speed=args.popup_sfx_speed,
        popup_sfx_volume=args.popup_sfx_volume,
        narration_volume=args.narration_volume,
        bgm_path=Path(args.bgm_path) if args.bgm_path else None,
        bgm_volume=args.bgm_volume,
        source_top_crop=args.gameplay_top_crop,
    )

    # Step 6: metadata
    step += 1
    print_progress(step, total_steps, "Generating metadata")
    if client is not None:
        title, description = generate_metadata(
            client,
            script,
            include_description=not args.no_description,
            persona=channel_persona,
            gameplay_credit=gameplay_credit_block(gameplay_file),
        )
    else:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook
        title = title or "Muslim Teen Story You Won't Believe 😭 #shorts #muslim"
        description = "" if args.no_description else (
            "Subscribe for more storytime shorts!\n#shorts #storytime #schoolstory"
        )
        if description:
            description = f"{description}\n\n{gameplay_credit_block(gameplay_file)}"
    if "#shorts" not in title.lower():
        title = f"{title} #Shorts"
    if description and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"
    tags = list(MUSLIM_SHORT_TAGS)

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
    print(f"Gameplay source: {gameplay_file.name} ({classify_gameplay_family(gameplay_file)})")
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
        post_pinned_comment(_yt, video_id, script, client=client)
        append_upload_registry(video_id, title=title, script=script)
    else:
        print("Upload skipped. Run with --upload to publish.")

    # Cleanup intermediate image/video assets after the short is rendered (and uploaded).
    for sub in ("scene_assets", "hook_video"):
        target = output_dir / sub
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


if __name__ == "__main__":
    main()
