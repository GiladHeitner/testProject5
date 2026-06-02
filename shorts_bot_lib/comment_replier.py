"""Conservative auto-reply to YouTube comments on bot-uploaded Shorts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .script_ai import generate_comment_reply
from .youtube_api import (
    VideoComment,
    build_youtube_client,
    get_own_channel_id,
    list_video_comments,
    load_upload_registry,
    post_comment_reply,
)

DEFAULT_REPLY_LOG = Path(".github/comment_reply_log.jsonl")
DEFAULT_REPLIED_IDS = Path(".github/replied_comments.txt")

_SPAM_PATTERNS = re.compile(
    r"(^|\s)(first|first!|who'?s watching|sub 4 sub|check my channel|http://|https://)(\s|$)",
    re.IGNORECASE,
)
_EMOJI_ONLY = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]+$"
)
_HATE_PATTERNS = re.compile(
    r"\b(towelhead|sand\s*nigger|terrorist\s*lover|go\s*back\s*to)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReplyCandidate:
    comment: VideoComment
    video_title: str
    script: str
    score: float
    min_age_minutes: float


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_replied_comment_ids(
    log_path: Path | None = None,
    ids_path: Path | None = None,
) -> set[str]:
    ids: set[str] = set()
    log = log_path or Path(os.environ.get("COMMENT_REPLY_LOG", str(DEFAULT_REPLY_LOG)))
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cid = str(row.get("comment_id") or "").strip()
                if cid:
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    txt = ids_path or Path(os.environ.get("REPLIED_COMMENTS_FILE", str(DEFAULT_REPLIED_IDS)))
    if txt.is_file():
        for line in txt.read_text(encoding="utf-8").splitlines():
            cid = line.strip().split("#", 1)[0].strip()
            if cid:
                ids.add(cid)
    return ids


def count_replies_last_hours(hours: float, log_path: Path | None = None) -> int:
    log = log_path or Path(os.environ.get("COMMENT_REPLY_LOG", str(DEFAULT_REPLY_LOG)))
    if not log.is_file():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    count = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(
                str(row.get("replied_at") or "").replace("Z", "+00:00")
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if ts >= cutoff:
            count += 1
    return count


def author_ids_replied_last_hours(
    hours: float, log_path: Path | None = None
) -> set[str]:
    log = log_path or Path(os.environ.get("COMMENT_REPLY_LOG", str(DEFAULT_REPLY_LOG)))
    if not log.is_file():
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    authors: set[str] = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(
                str(row.get("replied_at") or "").replace("Z", "+00:00")
            )
            author = str(row.get("author_channel_id") or "").strip()
        except (json.JSONDecodeError, ValueError):
            continue
        if ts >= cutoff and author:
            authors.add(author)
    return authors


def append_reply_log(
    comment_id: str,
    video_id: str,
    *,
    reply_text: str,
    author_channel_id: str = "",
    log_path: Path | None = None,
    ids_path: Path | None = None,
) -> None:
    log = log_path or Path(os.environ.get("COMMENT_REPLY_LOG", str(DEFAULT_REPLY_LOG)))
    ids = ids_path or Path(os.environ.get("REPLIED_COMMENTS_FILE", str(DEFAULT_REPLIED_IDS)))
    log.parent.mkdir(parents=True, exist_ok=True)
    ids.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "comment_id": comment_id,
        "video_id": video_id,
        "author_channel_id": author_channel_id,
        "reply_text": reply_text,
        "replied_at": datetime.now(timezone.utc).isoformat(),
    }
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with ids.open("a", encoding="utf-8") as fh:
        fh.write(f"{comment_id}\n")


def _is_spam_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if not re.search(r"[a-zA-Z]", t):
        return True
    if _EMOJI_ONLY.match(t):
        return True
    if _SPAM_PATTERNS.search(t):
        return True
    if _HATE_PATTERNS.search(t):
        return True
    if re.match(r"^https?://\S+$", t, re.IGNORECASE):
        return True
    return False


def _score_comment(text: str) -> float:
    t = text.lower()
    score = 0.0
    if "?" in text:
        score += 5.0
    if any(w in t for w in ("you", "your", "why", "how", "what", "muslim", "hijab", "arab")):
        score += 3.0
    if len(text) >= 30:
        score += 2.0
    if len(text) >= 80:
        score += 1.0
    return score


def collect_candidates(
    youtube,
    own_channel_id: str,
    registry: list[dict[str, Any]],
    replied_ids: set[str],
    authors_recent: set[str],
    *,
    min_age_min: float,
    max_age_min: float,
    random_skip_rate: float,
) -> list[ReplyCandidate]:
    now = datetime.now(timezone.utc)
    candidates: list[ReplyCandidate] = []

    for row in registry:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        title = str(row.get("title") or "")
        script = str(row.get("script") or "")
        try:
            comments = list_video_comments(youtube, video_id)
        except Exception as exc:
            print(f"[reply] skip comments for {video_id}: {exc}", file=sys.stderr)
            continue

        for comment in comments:
            if comment.comment_id in replied_ids:
                continue
            if comment.author_channel_id == own_channel_id:
                continue
            if comment.author_channel_id in authors_recent:
                continue
            if _is_spam_text(comment.text):
                continue

            age_min = (now - comment.published_at).total_seconds() / 60.0
            if age_min < min_age_min:
                continue
            if age_min > max_age_min:
                continue
            if random.random() < random_skip_rate:
                continue

            per_comment_min = random.uniform(min_age_min, max(min_age_min, min_age_min + 135))
            if age_min < per_comment_min:
                continue

            candidates.append(
                ReplyCandidate(
                    comment=comment,
                    video_title=title,
                    script=script,
                    score=_score_comment(comment.text),
                    min_age_minutes=per_comment_min,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def run_comment_replier(
    *,
    dry_run: bool = False,
    max_replies: int | None = None,
    max_video_age_hours: float | None = None,
    registry_path: Path | None = None,
) -> int:
    max_per_run = max_replies if max_replies is not None else _env_int("COMMENT_REPLY_MAX_PER_RUN", 2)
    max_per_day = _env_int("COMMENT_REPLY_MAX_PER_DAY", 15)
    max_age_h = max_video_age_hours if max_video_age_hours is not None else _env_float(
        "COMMENT_REPLY_MAX_VIDEO_AGE_HOURS", 168.0
    )
    min_age = _env_float("COMMENT_REPLY_MIN_AGE_MINUTES", 45.0)
    max_comment_age = _env_float("COMMENT_REPLY_MAX_COMMENT_AGE_HOURS", 72.0) * 60.0
    random_skip = _env_float("COMMENT_REPLY_RANDOM_SKIP_RATE", 0.20)
    sleep_min = _env_float("COMMENT_REPLY_SLEEP_MIN_SECONDS", 120.0)
    sleep_max = _env_float("COMMENT_REPLY_SLEEP_MAX_SECONDS", 360.0)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not dry_run:
        raise RuntimeError("Missing OPENAI_API_KEY for comment replies.")

    client = OpenAI(api_key=api_key) if api_key else None
    youtube = build_youtube_client()
    own_channel_id = get_own_channel_id(youtube)

    if count_replies_last_hours(24.0) >= max_per_day:
        print(f"[reply] Daily cap reached ({max_per_day}/24h). Exiting.")
        return 0

    registry = load_upload_registry(registry_path, max_age_hours=max_age_h)
    if not registry:
        print("[reply] No videos in upload registry for the age window.", file=sys.stderr)
        return 0

    replied_ids = load_replied_comment_ids()
    authors_recent = author_ids_replied_last_hours(24.0)

    candidates = collect_candidates(
        youtube,
        own_channel_id,
        registry,
        replied_ids,
        authors_recent,
        min_age_min=min_age,
        max_age_min=max_comment_age,
        random_skip_rate=random_skip,
    )
    if not candidates:
        print("[reply] No eligible comments to reply to.")
        return 0

    posted = 0
    used_videos: set[str] = set()

    for cand in candidates:
        if posted >= max_per_run:
            break
        if count_replies_last_hours(24.0) >= max_per_day:
            break
        if cand.comment.video_id in used_videos:
            continue

        reply_text: str | None = None
        if client is not None:
            try:
                reply_text = generate_comment_reply(
                    client,
                    script=cand.script,
                    video_title=cand.video_title,
                    comment_text=cand.comment.text,
                    author_name=cand.comment.author_display_name,
                )
            except Exception as exc:
                print(f"[reply] LLM failed for {cand.comment.comment_id}: {exc}")
                continue
        if not reply_text:
            print(f"[reply] SKIP (model/filter): {cand.comment.text[:60]!r}...")
            continue

        print(
            f"[reply] {'[dry-run] ' if dry_run else ''}video={cand.comment.video_id} "
            f"score={cand.score:.1f}\n"
            f"  comment: {cand.comment.text[:120]!r}\n"
            f"  reply: {reply_text!r}"
        )

        if dry_run:
            posted += 1
            used_videos.add(cand.comment.video_id)
            continue

        try:
            post_comment_reply(youtube, cand.comment.comment_id, reply_text)
            append_reply_log(
                cand.comment.comment_id,
                cand.comment.video_id,
                reply_text=reply_text,
                author_channel_id=cand.comment.author_channel_id,
            )
            replied_ids.add(cand.comment.comment_id)
            used_videos.add(cand.comment.video_id)
            posted += 1
        except Exception as exc:
            print(f"[reply] Post failed: {exc}", file=sys.stderr)
            continue

        if posted < max_per_run and not dry_run:
            delay = random.uniform(sleep_min, max(sleep_min, sleep_max))
            print(f"[reply] Sleeping {delay:.0f}s before next reply...")
            time.sleep(delay)

    print(f"[reply] Done. {'Would post' if dry_run else 'Posted'} {posted} reply/replies.")
    return posted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reply to eligible YouTube comments on bot-uploaded Shorts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find comments and generate replies without posting.",
    )
    parser.add_argument(
        "--max-replies",
        type=int,
        default=None,
        help="Max replies this run (default: env COMMENT_REPLY_MAX_PER_RUN or 2).",
    )
    parser.add_argument(
        "--max-video-age-hours",
        type=float,
        default=None,
        help="Only consider uploads in registry newer than this (default 168h).",
    )
    parser.add_argument(
        "--registry",
        default="",
        help="Path to upload_registry.jsonl (default .github/upload_registry.jsonl).",
    )
    args = parser.parse_args(argv)
    dry_run = args.dry_run or os.environ.get("COMMENT_REPLY_DRY_RUN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    registry = Path(args.registry) if args.registry else None
    run_comment_replier(
        dry_run=dry_run,
        max_replies=args.max_replies,
        max_video_age_hours=args.max_video_age_hours,
        registry_path=registry,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
