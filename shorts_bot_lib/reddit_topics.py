"""Fetch top weekly text posts from Reddit for Shorts topic input.

Uses PRAW when REDDIT_CLIENT_ID/SECRET are set; otherwise public .json
endpoints (no Reddit app required).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

_OAUTH_TOKEN: str | None = None

DEFAULT_SUBREDDITS = (
    "teenagers",
    "MuslimLounge",
    "ABCDesis",
    "islam",
)

# Posts must match at least one keyword (title + body) for this channel niche.
_MUSLIM_ARAB_KEYWORDS = re.compile(
    r"\b("
    r"muslim|moslem|islam|islamophob|anti[\s-]?muslim|anti[\s-]?arab|"
    r"hijab|niqab|burqa|hijabi|ramadan|eid|mosque|masjid|quran|koran|"
    r"arab|arabs|arabic|middle eastern|palestin|ummah|allah|halal|"
    r"against arabs?|against muslims?|muslim hate|arab hate"
    r")\b",
    re.IGNORECASE,
)


def matches_muslim_arab_niche(text: str) -> bool:
    return bool(_MUSLIM_ARAB_KEYWORDS.search(text or ""))


@dataclass(frozen=True)
class SubredditSource:
    """How to list posts from one subreddit."""

    name: str
    kind: str = "top"  # top | hot | search
    time_filter: str = "week"
    search_query: str = ""
    required_flair: str = ""
    limit: int = 25


DEFAULT_SOURCES: tuple[SubredditSource, ...] = (
    SubredditSource(
        "teenagers",
        kind="search",
        search_query="muslim OR hijab OR arab OR islamophobia",
        time_filter="year",
        limit=60,
    ),
    SubredditSource(
        "teenagers",
        kind="search",
        search_query='"against arabs" OR "anti muslim" OR "muslim hate"',
        time_filter="year",
        limit=60,
    ),
    SubredditSource(
        "teenagers",
        kind="search",
        search_query="ramadan OR mosque OR quran OR wearing hijab",
        time_filter="year",
        limit=50,
    ),
    SubredditSource("MuslimLounge", kind="hot", limit=50),
    SubredditSource("ABCDesis", kind="top", time_filter="month", limit=50),
    SubredditSource("islam", kind="top", time_filter="week", limit=40),
)

MIN_SELFTEXT_CHARS = 80
MIN_TITLE_CHARS = 100  # AskReddit etc. often have empty body but a long title
MAX_TOPIC_CHARS = 12_000
# Separates full Reddit posts in topics.txt (title + body for script generation)
TOPICS_ENTRY_SEP = "\n---\n"
MIN_TOPIC_ENTRY_CHARS = 80


@dataclass(frozen=True)
class RedditPost:
    post_id: str
    subreddit: str
    title: str
    selftext: str
    permalink: str
    score: int

    @property
    def topic_text(self) -> str:
        """Full post body for --topic / script generation."""
        title = self.title.strip()
        body = self.selftext.strip()
        if self.subreddit == "topics":
            text = title
        else:
            lines = [
                f"Reddit post from r/{self.subreddit}:",
                "",
                title,
                "",
                body,
            ]
            text = "\n".join(lines).strip()
        if len(text) > MAX_TOPIC_CHARS:
            text = text[: MAX_TOPIC_CHARS - 3].rstrip() + "..."
        return text


def topic_entry_id(text: str) -> str:
    return f"topic:{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def load_topic_entries(topics_path: Path) -> list[str]:
    """Load topics from topics.txt (--- separated full posts, or legacy one per line)."""
    raw = topics_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if TOPICS_ENTRY_SEP in raw:
        parts = raw.split(TOPICS_ENTRY_SEP)
    else:
        parts = raw.splitlines()
    return [
        p.strip()
        for p in parts
        if p.strip() and not p.strip().startswith("#")
    ]


def _topic_preview(text: str, max_len: int = 80) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Reddit post from r/"):
            continue
        if len(line) > max_len:
            return line[: max_len - 1] + "…"
        return line
    compact = text.replace("\n", " ")[:max_len]
    return compact + ("…" if len(text) > max_len else "")


def _load_used_ids(used_file: Path) -> set[str]:
    if not used_file.exists():
        return set()
    return {
        line.strip()
        for line in used_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _user_agent() -> str:
    return os.environ.get(
        "REDDIT_USER_AGENT",
        "VideoBots:shorts-topic-picker:v1.0 (by /u/GiladHeitner; "
        "https://github.com/GiladHeitner/YoutubeUploader)",
    ).strip()


def _request_headers(*, bearer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
    }
    if bearer:
        headers["Authorization"] = f"bearer {bearer}"
    return headers


def _oauth_token() -> str | None:
    """Application-only OAuth token (works from CI when API secrets are set)."""
    global _OAUTH_TOKEN
    if _OAUTH_TOKEN:
        return _OAUTH_TOKEN
    if not _has_praw_credentials():
        return None
    cid = os.environ["REDDIT_CLIENT_ID"].strip()
    secret = os.environ["REDDIT_CLIENT_SECRET"].strip()
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": _user_agent(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception as exc:
        print(f"[reddit] OAuth token failed: {exc}", file=sys.stderr)
        return None
    token = (payload.get("access_token") or "").strip()
    if token:
        _OAUTH_TOKEN = token
        print("[reddit] Using OAuth API (client credentials).", file=sys.stderr)
    return token or None


def _http_json(url: str, headers: dict[str, str], *, retries: int = 2) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (403, 429, 503) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.0)
                continue
            raise
    raise last_err  # type: ignore[misc]


def _is_ci() -> bool:
    return os.environ.get("CI") == "true" or bool(os.environ.get("GITHUB_ACTIONS"))


def _use_topics_file_directly() -> bool:
    """Skip Reddit HTTP in CI without API keys (datacenter IPs always get 403)."""
    if os.environ.get("REDDIT_NO_TOPICS_FALLBACK") == "1":
        return False
    if os.environ.get("REDDIT_FORCE_PUBLIC") == "1":
        return False
    if _has_praw_credentials():
        return False
    return _is_ci()


def _should_use_topics_fallback() -> bool:
    if os.environ.get("REDDIT_NO_TOPICS_FALLBACK") == "1":
        return False
    if os.environ.get("REDDIT_TOPICS_FALLBACK") == "1":
        return True
    if _is_ci():
        return True
    return Path(os.environ.get("TOPICS_FILE", "topics.txt")).is_file()


def _filter_niche_posts(posts: list[RedditPost]) -> list[RedditPost]:
    return [p for p in posts if matches_muslim_arab_niche(p.topic_text)]


def _filter_niche_topic_entries(topics: list[str]) -> list[str]:
    return [t for t in topics if matches_muslim_arab_niche(t)]


def pick_topics_file_fallback(
    used_file: Path | None = None,
    topics_file: Path | None = None,
) -> RedditPost:
    """Fallback when Reddit blocks datacenter IPs (e.g. GitHub Actions)."""
    topics_path = topics_file or Path(os.environ.get("TOPICS_FILE", "topics.txt"))
    if not topics_path.is_file():
        raise RuntimeError(f"Topics fallback file not found: {topics_path}")
    used_path = used_file or Path(".github/used_topics.txt")
    used_ids = _load_used_ids(used_path)
    topics = load_topic_entries(topics_path)
    if not topics:
        raise RuntimeError(f"No topics in {topics_path}")
    niche_topics = _filter_niche_topic_entries(topics)
    if niche_topics:
        topics = niche_topics
    else:
        print(
            "[reddit] No Muslim/Arab keyword matches in topics.txt; using full file.",
            file=sys.stderr,
        )
    fresh = [t for t in topics if topic_entry_id(t) not in used_ids]
    pool = fresh if fresh else topics
    chosen = random.choice(pool)
    post_id = topic_entry_id(chosen)
    print(f"[reddit] topics.txt: {_topic_preview(chosen)!r}", file=sys.stderr)
    return RedditPost(
        post_id=post_id,
        subreddit="topics",
        title=chosen,
        selftext="",
        permalink="",
        score=0,
    )


def _has_praw_credentials() -> bool:
    return bool(
        os.environ.get("REDDIT_CLIENT_ID", "").strip()
        and os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    )


def _reddit_client():
    try:
        import praw
    except ImportError as exc:
        raise RuntimeError(
            "Missing `praw`. Install with: pip install praw"
        ) from exc

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"].strip(),
        client_secret=os.environ["REDDIT_CLIENT_SECRET"].strip(),
        user_agent=_user_agent(),
    )


def _flair_matches(data: dict[str, Any], required: str) -> bool:
    if not required:
        return True
    flair = (data.get("link_flair_text") or "").strip()
    return flair.lower() == required.strip().lower()


def _post_from_listing(
    data: dict[str, Any],
    *,
    required_flair: str = "",
) -> RedditPost | None:
    if data.get("stickied"):
        return None
    if not _flair_matches(data, required_flair):
        return None
    selftext = (data.get("selftext") or "").strip()
    title = (data.get("title") or "").strip()
    if len(selftext) < MIN_SELFTEXT_CHARS and len(title) < MIN_TITLE_CHARS:
        return None
    if data.get("over_18"):
        return None
    if not title:
        return None
    permalink = data.get("permalink") or ""
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"
    sub = data.get("subreddit") or data.get("subreddit_name_prefixed", "").removeprefix("r/")
    return RedditPost(
        post_id=str(data.get("id") or ""),
        subreddit=str(sub),
        title=title,
        selftext=selftext,
        permalink=permalink,
        score=int(data.get("score") or 0),
    )


def _resolve_sources(subreddits: list[str] | None) -> list[SubredditSource]:
    if subreddits is None:
        return list(DEFAULT_SOURCES)
    by_name = {s.name.lower(): s for s in DEFAULT_SOURCES}
    return [
        by_name.get(name.lower(), SubredditSource(name))
        for name in subreddits
    ]


def _listing_path(source: SubredditSource, *, json_suffix: bool = True) -> str:
    name = source.name
    limit = source.limit
    ext = ".json" if json_suffix else ""
    if source.kind == "search":
        q = urllib.parse.quote(source.search_query)
        return (
            f"/r/{name}/search{ext}"
            f"?q={q}&restrict_sr=1&sort=top&t={source.time_filter}"
            f"&limit={limit}&raw_json=1"
        )
    if source.kind == "hot":
        return f"/r/{name}/hot{ext}?limit={limit}&raw_json=1"
    return (
        f"/r/{name}/top{ext}?t={source.time_filter}&limit={limit}&raw_json=1"
    )


def _listing_urls(source: SubredditSource) -> list[str]:
    path = _listing_path(source, json_suffix=True)
    return [
        f"https://oauth.reddit.com{path.replace('.json', '')}",
        f"https://www.reddit.com{path}",
        f"https://old.reddit.com{path}",
    ]


def _fetch_listing_payload(source: SubredditSource) -> dict[str, Any] | None:
    token = _oauth_token()
    if token:
        oauth_url = f"https://oauth.reddit.com{_listing_path(source, json_suffix=False)}"
        try:
            return _http_json(oauth_url, _request_headers(bearer=token))
        except Exception as exc:
            print(
                f"[reddit] OAuth skip r/{source.name}: {exc}",
                file=sys.stderr,
            )
    for url in _listing_urls(source)[1:]:
        try:
            return _http_json(url, _request_headers())
        except urllib.error.HTTPError as exc:
            print(
                f"[reddit] skip r/{source.name} ({url.split('/')[2]}): HTTP {exc.code}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[reddit] skip r/{source.name}: {exc}", file=sys.stderr)
    return None


def _submission_dict(submission) -> dict[str, Any]:
    return {
        "id": submission.id,
        "subreddit": submission.subreddit.display_name,
        "title": submission.title,
        "selftext": submission.selftext,
        "permalink": submission.permalink,
        "score": submission.score,
        "stickied": submission.stickied,
        "over_18": submission.over_18,
        "link_flair_text": getattr(submission, "link_flair_text", None),
    }


def _iter_submission_list(reddit, source: SubredditSource):
    sub = reddit.subreddit(source.name)
    if source.kind == "search":
        return sub.search(
            source.search_query,
            sort="top",
            time_filter=source.time_filter,
            limit=source.limit,
        )
    if source.kind == "hot":
        return sub.hot(limit=source.limit)
    return sub.top(time_filter=source.time_filter, limit=source.limit)


def _iter_candidates_public(sources: list[SubredditSource]) -> Iterator[RedditPost]:
    for source in sources:
        payload = _fetch_listing_payload(source)
        if payload is None:
            continue
        children = payload.get("data", {}).get("children", [])
        if not children and source.kind == "top":
            hot = SubredditSource(source.name, kind="hot", limit=source.limit)
            payload = _fetch_listing_payload(hot)
            children = (payload or {}).get("data", {}).get("children", [])
            if children:
                print(f"[reddit] r/{source.name}: top empty, using hot", file=sys.stderr)
        for child in children:
            data = child.get("data") if isinstance(child, dict) else None
            if not isinstance(data, dict):
                continue
            post = _post_from_listing(
                data, required_flair=source.required_flair
            )
            if post and post.post_id:
                yield post


def _iter_candidates(reddit, sources: list[SubredditSource]) -> Iterator[RedditPost]:
    for source in sources:
        try:
            for submission in _iter_submission_list(reddit, source):
                post = _post_from_listing(
                    _submission_dict(submission),
                    required_flair=source.required_flair,
                )
                if post:
                    yield post
        except Exception as exc:
            print(f"[reddit] skip r/{source.name}: {exc}", file=sys.stderr)


def pick_reddit_post(
    used_file: Path | None = None,
    subreddits: list[str] | None = None,
) -> RedditPost:
    """Return one unused top-weekly text post from the configured subreddits."""
    used_path = used_file or Path(".github/used_reddit.txt")
    used_ids = _load_used_ids(used_path)
    sources = _resolve_sources(
        list(subreddits) if subreddits else None
    )
    random.shuffle(sources)

    if _use_topics_file_directly():
        print(
            "[reddit] CI without Reddit API keys — using topics.txt "
            "(Reddit blocks GitHub Actions; set REDDIT_CLIENT_ID/SECRET to enable Reddit).",
            file=sys.stderr,
        )
        return pick_topics_file_fallback()

    if _has_praw_credentials():
        candidates = list(_iter_candidates(_reddit_client(), sources))
    else:
        print(
            "[reddit] No API credentials — using public JSON (local/dev only).",
            file=sys.stderr,
        )
        candidates = list(_iter_candidates_public(sources))
    if not candidates:
        if _should_use_topics_fallback():
            print(
                "[reddit] Reddit unavailable — falling back to topics.txt.",
                file=sys.stderr,
            )
            return pick_topics_file_fallback()
        raise RuntimeError(
            "No suitable Reddit text posts found. Check subreddit names, API secrets, "
            "or add topics.txt for CI fallback."
        )

    niche = _filter_niche_posts(candidates)
    if niche:
        candidates = niche
        print(
            f"[reddit] {len(candidates)} Muslim/Arab niche post(s) after keyword filter.",
            file=sys.stderr,
        )
    else:
        print(
            "[reddit] No Muslim/Arab keyword matches in Reddit results; using all candidates.",
            file=sys.stderr,
        )

    random.shuffle(candidates)
    fresh = [p for p in candidates if p.post_id not in used_ids]
    pool = fresh if fresh else candidates
    pool.sort(key=lambda p: p.score, reverse=True)
    # Pick from top-scored among unused (or all if everything was used).
    top_n = min(8, len(pool))
    chosen = random.choice(pool[:top_n])
    print(
        f"Reddit topic: r/{chosen.subreddit} | score={chosen.score} | "
        f"{chosen.title[:72]!r}"
    )
    print(f"Permalink: {chosen.permalink}")
    return chosen


def collect_reddit_candidates(
    subreddits: list[str] | None = None,
    *,
    per_source_limit: int = 50,
) -> list[RedditPost]:
    """Gather posts from all configured sources (for local sync → topics.txt)."""
    sources = _resolve_sources(list(subreddits) if subreddits else None)
    sources = [replace(s, limit=per_source_limit) for s in sources]
    random.shuffle(sources)

    if _has_praw_credentials():
        print("[reddit] Fetching via PRAW/API…", file=sys.stderr)
        candidates = list(_iter_candidates(_reddit_client(), sources))
    else:
        print("[reddit] Fetching via public JSON (run this on your Mac, not CI)…", file=sys.stderr)
        candidates = list(_iter_candidates_public(sources))

    seen: set[str] = set()
    unique: list[RedditPost] = []
    for post in candidates:
        if post.post_id and post.post_id not in seen:
            seen.add(post.post_id)
            unique.append(post)
    unique.sort(key=lambda p: p.score, reverse=True)
    return unique


def sync_topics_file(
    topics_file: Path | None = None,
    *,
    limit: int = 25,
    merge: bool = True,
    subreddits: list[str] | None = None,
    min_entry_chars: int = MIN_TOPIC_ENTRY_CHARS,
) -> int:
    """Scrape Reddit locally; write full posts (title + body) into topics.txt for CI."""
    path = topics_file or Path(os.environ.get("TOPICS_FILE", "topics.txt"))
    posts = collect_reddit_candidates(subreddits)
    if not posts:
        raise RuntimeError(
            "No Reddit posts fetched. Run on your Mac (not GitHub Actions) or add API keys."
        )

    new_entries: list[str] = []
    for post in posts:
        text = post.topic_text
        if len(text) >= min_entry_chars and matches_muslim_arab_niche(text):
            new_entries.append(text)
        if len(new_entries) >= limit:
            break

    existing: list[str] = []
    if merge and path.is_file():
        existing = load_topic_entries(path)

    seen_ids = {topic_entry_id(t) for t in existing}
    added = 0
    combined = list(existing)
    for entry in new_entries:
        eid = topic_entry_id(entry)
        if eid in seen_ids:
            continue
        combined.append(entry)
        seen_ids.add(eid)
        added += 1

    path.write_text(TOPICS_ENTRY_SEP.join(combined) + "\n", encoding="utf-8")
    print(
        f"topics.txt: {len(combined)} full posts, {added} new from Reddit → {path}",
        file=sys.stderr,
    )
    for entry in new_entries[:5]:
        print(f"  • {_topic_preview(entry)}", file=sys.stderr)
    if len(new_entries) > 5:
        print(f"  … and {len(new_entries) - 5} more", file=sys.stderr)
    return added


def fetch_topic_for_pipeline(
    used_file: Path | None = None,
    subreddits: list[str] | None = None,
) -> tuple[str, str]:
    """Return (topic_text, post_id) for the chosen Reddit submission."""
    post = pick_reddit_post(used_file=used_file, subreddits=subreddits)
    return post.topic_text, post.post_id


def mark_post_used(post_id: str, used_file: Path | None = None) -> None:
    if post_id.startswith("topic:"):
        used_file = Path(".github/used_topics.txt")
    else:
        used_file = used_file or Path(".github/used_reddit.txt")
    used_file.parent.mkdir(parents=True, exist_ok=True)
    if used_file.exists():
        existing = {
            ln.strip()
            for ln in used_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        if post_id in existing:
            return
    with used_file.open("a", encoding="utf-8") as fh:
        fh.write(f"{post_id}\n")


def _cli_sync_topics(args: argparse.Namespace) -> int:
    subs = None
    if args.subreddits:
        subs = [s.strip() for s in re.split(r"[, ]+", args.subreddits) if s.strip()]
    sync_topics_file(
        Path(args.out),
        limit=args.limit,
        merge=not args.replace,
        subreddits=subs,
    )
    return 0


def _cli_pick(args: argparse.Namespace) -> int:
    used_file = Path(args.used)
    out_file = Path(args.out)
    id_file = Path(args.id_out) if args.id_out else None

    subs = None
    if args.subreddits:
        subs = [s.strip() for s in re.split(r"[, ]+", args.subreddits) if s.strip()]

    post = pick_reddit_post(used_file=used_file, subreddits=subs)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(post.topic_text, encoding="utf-8")
    if id_file:
        id_file.write_text(post.post_id, encoding="utf-8")
    if args.mark_used:
        mark_post_used(post.post_id, used_file)
    print(f"Wrote topic ({len(post.topic_text)} chars) -> {out_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick a Reddit post for Shorts topic input.")
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("pick", help="Fetch one top-weekly post and write topic text to a file.")
    pick.add_argument(
        "--used",
        default=".github/used_reddit.txt",
        help="File tracking used Reddit post IDs (one per line).",
    )
    pick.add_argument(
        "--out",
        default=".github/reddit_topic.txt",
        help="Output file for full topic text (title + body).",
    )
    pick.add_argument(
        "--id-out",
        default=".github/reddit_post_id.txt",
        help="Output file for the chosen post ID.",
    )
    pick.add_argument(
        "--subreddits",
        default="",
        help=(
            "Comma-separated subreddit names "
            "(default: teenagers,MuslimLounge,ABCDesis,islam)."
        ),
    )
    pick.add_argument(
        "--mark-used",
        action="store_true",
        help="Append post ID to --used immediately after picking.",
    )
    pick.set_defaults(func=_cli_pick)

    sync = sub.add_parser(
        "sync-topics",
        help="Scrape full Reddit posts into topics.txt (commit & push for CI).",
    )
    sync.add_argument(
        "--out",
        default="topics.txt",
        help="Output topics file (default: topics.txt).",
    )
    sync.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max new topics to add from Reddit this run.",
    )
    sync.add_argument(
        "--replace",
        action="store_true",
        help="Replace topics.txt instead of merging with existing lines.",
    )
    sync.add_argument(
        "--subreddits",
        default="",
        help="Comma-separated subreddits (default: Muslim/Arab niche sources).",
    )
    sync.set_defaults(func=_cli_sync_topics)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
