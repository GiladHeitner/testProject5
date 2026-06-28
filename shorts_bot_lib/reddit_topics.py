"""Fetch top weekly text posts from Reddit for Shorts topic input.

Uses PRAW when REDDIT_CLIENT_ID/SECRET are set; otherwise public .json
endpoints (no Reddit app required). Harvest can also use pullpush.io (Reddit archive).
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
    "hijabis",
    "muslim",
    "arabs",
    "ArabWorld",
)

# Prefer topics that match high-performing angles on this channel.
_PRIORITY_NICHE_KEYWORDS = re.compile(
    r"islamophob|against arabs?|terrorist|muslim hate|arab hate|"
    r"ramadan|picture day|group chat|yearbook|mosque|airport|tsa|halal|"
    r"arranged|cousin|convert|hate crime|iftar|suhoor|fasting|eid|"
    r"diaspora|desi|prayer|imam|profiling|jummah|"
    r"bomb joke|racism against|being muslim|being arab",
    re.IGNORECASE,
)

# Meme titles, copypasta, awareness spam — not storytime.
_STORYTIME_REJECT_RE = re.compile(
    r"deep down we all know|repost it please|spread awareness|"
    r"concentration camps in china|step 1:|step 2:|"
    r"\bpepe\b|minecraft|terraria|gamer moment|"
    r"ternion all powerful|gifted ternion|"
    r"jedi and heres a tutorial|halal love\.\s*$|"
    r"^any muslims\?\s*$|just seeing if there are other muslims|"
    r"greatest nation on earth|happy 4th of july|proud americans|"
    r"the weeknd|marawi 2\.0|^my 0pinion$|^my opinion$",
    re.IGNORECASE | re.MULTILINE,
)

# r/teenagers keyword hits include meme posts; require a real story beat.
_TEEN_STORY_ANGLE_RE = re.compile(
    r"teacher|school|parents?|friend|coworker|boss|manager|"
    r"class|principal|called me|group chat|airport|security|"
    r"fasting|ramadan|mosque|neighbor|convert|marry|cousin|"
    r"bomb joke|racist|discriminat|islamophob|halal|iftar|"
    r"crush|girl likes|friend zone",
    re.IGNORECASE,
)

_HARVEST_PREFERRED_SUBREDDITS = frozenset(
    {"muslim", "islam", "muslimlounge", "arabs", "abcdesis", "arabworld"}
)
MIN_HARVEST_SELFTEXT_CHARS = 120

# Stories centered on female-only experiences (hijab, prom dress, etc.).
_FEMALE_PROTAGONIST_RE = re.compile(
    r"\b("
    r"hijab|hijabi|niqab|burqa|abaya|take off my hijab|my hijab|"
    r"muslim girl|prom dress|buying dresses|look oppressed|"
    r"she wears hijab|headscarf.*distract|dress code violation.*hijab"
    r")\b",
    re.IGNORECASE,
)


def topic_priority_score(text: str) -> int:
    score = 1 if matches_muslim_arab_niche(text) else 0
    if _PRIORITY_NICHE_KEYWORDS.search(text or ""):
        score += 10
    return score


def _rank_topics_by_priority(topics: list[str]) -> list[str]:
    return sorted(topics, key=topic_priority_score, reverse=True)


# --- Topic-diversity cooldown -------------------------------------------------
# The priority score boosts "arranged"/islamophobia heavily, which made the
# channel publish 10+ near-identical "arranged marriage" shorts. A coarse theme
# bucket + a short cooldown spreads picks across themes so consecutive uploads
# don't repeat. (Repetition is also what YouTube's "inauthentic content" policy
# targets — see CHANNEL_IMPROVEMENT_PLAN.md.)
_THEME_PATTERNS = [
    ("marriage", re.compile(
        r"\b(arrang\w*|marry|marriage|married|wedding|nikah|engaged|engagement|"
        r"proposal|suitor|rishta|spouse|husband|wife)\b", re.IGNORECASE)),
    ("islamophobia", re.compile(
        r"\b(islamophob\w*|racis\w*|hate crime|terrorist|bomb joke|profil\w*|"
        r"slur|discriminat\w*|anti[\s-]?muslim|anti[\s-]?arab|muslim hate|arab hate)\b",
        re.IGNORECASE)),
    ("ramadan", re.compile(r"\b(ramadan|fasting|fast|iftar|suhoor|eid|sawm)\b", re.IGNORECASE)),
    ("religion", re.compile(
        r"\b(prayer|salah|mosque|masjid|quran|koran|jummah|imam|hijab|niqab|"
        r"abaya|convert|revert|halal|haram)\b", re.IGNORECASE)),
    ("dating", re.compile(
        r"\b(crush|dating|girlfriend|boyfriend|texting|ex|flirt\w*|hookup|"
        r"relationship)\b", re.IGNORECASE)),
    ("school", re.compile(
        r"\b(teacher|school|class|classroom|principal|suspend\w*|detention|"
        r"exam|test|picture day|yearbook|homework|grade)\b", re.IGNORECASE)),
    ("family", re.compile(
        r"\b(parents?|mom|mum|dad|cousin|sibling|brother|sister|uncle|aunt|"
        r"family|household|relatives?)\b", re.IGNORECASE)),
]


def classify_topic_theme(text: str) -> str:
    """Bucket a topic into a coarse theme for the diversity cooldown."""
    t = text or ""
    for name, pat in _THEME_PATTERNS:
        if pat.search(t):
            return name
    return "other"


def _theme_cooldown_window() -> int:
    """How many recent uploads' themes to avoid repeating (env: TOPIC_THEME_COOLDOWN)."""
    try:
        return max(0, int(os.environ.get("TOPIC_THEME_COOLDOWN", "4")))
    except ValueError:
        return 4


def _theme_file_for(used_path: Path) -> Path:
    return used_path.parent / "used_themes.txt"


def _load_recent_themes(theme_path: Path, window: int) -> list[str]:
    if window <= 0:
        return []
    try:
        lines = [ln.strip() for ln in theme_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []
    return lines[-window:]


def mark_theme_used(theme: str, theme_path: Path, *, keep: int = 200) -> None:
    """Append the chosen theme so future picks can cool it down."""
    theme = (theme or "other").strip() or "other"
    try:
        existing = [ln.strip() for ln in theme_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        existing = []
    existing.append(theme)
    existing = existing[-keep:]
    theme_path.parent.mkdir(parents=True, exist_ok=True)
    theme_path.write_text("\n".join(existing) + "\n", encoding="utf-8")


def _apply_theme_cooldown(pool: list, used_path: Path, theme_of) -> list:
    """Drop pool items whose theme was used in the last N picks; keep order.

    `theme_of` maps a pool item to its theme string. Falls back to the full
    pool if every candidate matches a recent theme (so we never fail to pick).
    """
    window = _theme_cooldown_window()
    if window <= 0:
        return pool
    recent = set(_load_recent_themes(_theme_file_for(used_path), window))
    if not recent:
        return pool
    fresh = [item for item in pool if theme_of(item) not in recent]
    if fresh:
        return fresh
    print(
        f"[reddit] every candidate matches a recent theme {sorted(recent)}; "
        f"ignoring theme cooldown this round.",
        file=sys.stderr,
    )
    return pool

# Posts must match at least one keyword (title + body) for this channel niche.
_MUSLIM_ARAB_KEYWORDS = re.compile(
    r"\b("
    r"muslim|moslem|islam|islamophob|anti[\s-]?muslim|anti[\s-]?arab|"
    r"hijab|niqab|burqa|hijabi|ramadan|eid|mosque|masjid|quran|koran|"
    r"arab|arabs|arabic|middle eastern|palestin|ummah|allah|halal|"
    r"against arabs?|against muslims?|muslim hate|arab hate|"
    r"desi|diaspora|iftar|suhoor|fasting|salah|prayer|imam|abaya|"
    r"pakistani|lebanese|syrian|moroccan|egyptian|turkish|turk|"
    r"cousin|arranged|terrorist|islamophob|revert|honor kill|"
    r"yearbook.*hijab|hijab.*yearbook"
    r")\b",
    re.IGNORECASE,
)


def matches_muslim_arab_niche(text: str) -> bool:
    return bool(_MUSLIM_ARAB_KEYWORDS.search(text or ""))


def host_persona_gender() -> str:
    try:
        from .channel_persona import load_channel_persona

        return load_channel_persona().gender.strip().lower() or "male"
    except Exception:
        return os.environ.get("HOST_PERSONA_GENDER", "male").strip().lower() or "male"


def matches_host_persona(text: str, gender: str | None = None) -> bool:
    """True if the story fits the channel host (e.g. male Omar, not hijab-first posts)."""
    g = (gender or host_persona_gender()).strip().lower()
    if g == "male" and _FEMALE_PROTAGONIST_RE.search(text or ""):
        return False
    return True


def _topic_body(text: str) -> str:
    lines = (text or "").splitlines()
    if len(lines) > 3 and lines[0].startswith("Reddit post from r/"):
        return "\n".join(lines[3:]).strip()
    return text.strip()


def _topic_subreddit(text: str) -> str:
    m = re.match(r"Reddit post from r/([^:]+):", (text or "").splitlines()[0] if text else "")
    return (m.group(1) if m else "").strip().lower()


def is_quality_storytime_post(post: RedditPost) -> bool:
    """Real narrative posts for Omar — not memes or one-line Muslim mentions."""
    text = post.topic_text
    if not matches_muslim_arab_niche(text) or not matches_host_persona(text):
        return False
    if _STORYTIME_REJECT_RE.search(text):
        return False
    body = post.selftext.strip() or _topic_body(text)
    if len(body) < MIN_HARVEST_SELFTEXT_CHARS:
        return False
    sub = post.subreddit.strip().lower()
    if sub == "teenagers" and not _TEEN_STORY_ANGLE_RE.search(text):
        return False
    if topic_priority_score(text) >= 11:
        return True
    if sub == "teenagers" and _TEEN_STORY_ANGLE_RE.search(text) and len(body) >= 200:
        return True
    if sub in _HARVEST_PREFERRED_SUBREDDITS and len(body) >= 200:
        return True
    return False


def is_quality_storytime_entry(text: str) -> bool:
    if not matches_muslim_arab_niche(text) or not matches_host_persona(text):
        return False
    if _STORYTIME_REJECT_RE.search(text):
        return False
    body = _topic_body(text)
    if len(body) < MIN_HARVEST_SELFTEXT_CHARS:
        return False
    sub = _topic_subreddit(text)
    if sub == "teenagers" and not _TEEN_STORY_ANGLE_RE.search(text):
        return False
    if topic_priority_score(text) >= 11:
        return True
    if sub == "teenagers" and len(body) >= 200:
        return True
    if sub in _HARVEST_PREFERRED_SUBREDDITS and len(body) >= 200:
        return True
    return False


def _filter_host_persona_topics(topics: list[str]) -> list[str]:
    return [t for t in topics if matches_host_persona(t)]


def _filter_host_persona_posts(posts: list[RedditPost]) -> list[RedditPost]:
    return [p for p in posts if matches_host_persona(p.topic_text)]


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
    SubredditSource("hijabis", kind="hot", limit=40),
    SubredditSource("muslim", kind="hot", limit=40),
    SubredditSource("arabs", kind="hot", limit=40),
    SubredditSource("ArabWorld", kind="hot", limit=30),
    SubredditSource(
        "teenagers",
        kind="search",
        search_query="islamophobic OR racial profiling OR airport security",
        time_filter="year",
        limit=50,
    ),
    SubredditSource(
        "teenagers",
        kind="search",
        search_query="halal OR eid OR fasting at school",
        time_filter="year",
        limit=40,
    ),
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


def _http_timeout() -> float:
    return float(os.environ.get("REDDIT_HTTP_TIMEOUT", "60"))


def _fetch_delay() -> float:
    return float(os.environ.get("REDDIT_FETCH_DELAY", "6.0"))


def _http_json(url: str, headers: dict[str, str], *, retries: int = 2) -> dict[str, Any]:
    last_err: Exception | None = None
    timeout = _http_timeout()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (403, 429, 503) and attempt < retries:
                time.sleep(_fetch_delay() * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(_fetch_delay())
                continue
            raise
    raise last_err  # type: ignore[misc]


PULLPUSH_SEARCH = "https://api.pullpush.io/reddit/search/submission/"

# (subreddit, query) — empty query = top-scored posts in that subreddit.
PULLPUSH_HARVEST_BATCHES: tuple[tuple[str, str], ...] = (
    ("muslim", ""),
    ("MuslimLounge", ""),
    ("islam", ""),
    ("arabs", ""),
    ("ABCDesis", ""),
    ("ArabWorld", ""),
    ("MuslimLounge", "islamophobia"),
    ("MuslimLounge", "school"),
    ("MuslimLounge", "work"),
    ("muslim", "convert"),
    ("muslim", "ramadan"),
    ("islam", "convert"),
    ("islam", "revert"),
    ("arabs", "discrimination"),
    ("ABCDesis", "parents"),
    ("ABCDesis", "cousin"),
    ("ABCDesis", "arranged"),
    ("teenagers", "islamophobia"),
    ("teenagers", "ramadan"),
    ("teenagers", "arab"),
    ("teenagers", "halal"),
    ("teenagers", "mosque"),
    ("teenagers", "profiling"),
    ("teenagers", "convert"),
)


def _fetch_pullpush_posts(
    *,
    subreddit: str,
    query: str,
    size: int = 100,
) -> list[dict[str, Any]]:
    params: dict[str, str | int] = {
        "subreddit": subreddit,
        "size": min(100, max(10, size)),
        "sort": "desc",
        "sort_type": "score",
    }
    if query.strip():
        params["q"] = query
    params_encoded = urllib.parse.urlencode(params)
    url = f"{PULLPUSH_SEARCH}?{params_encoded}"
    try:
        payload = _http_json(url, _request_headers(), retries=3)
    except Exception as exc:
        print(
            f"[reddit] pullpush skip r/{subreddit} q={query!r}: {exc}",
            file=sys.stderr,
        )
        return []
    finally:
        time.sleep(_fetch_delay())
    rows = payload.get("data") if isinstance(payload, dict) else None
    result = rows if isinstance(rows, list) else []
    if result or not query.strip() or " " not in query.strip():
        return result
    # Multi-word queries often return nothing; try each keyword.
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for word in query.split():
        for row in _fetch_pullpush_posts(subreddit=subreddit, query=word, size=size):
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                merged.append(row)
        time.sleep(_fetch_delay() * 0.5)
    return merged


def _post_from_pullpush(data: dict[str, Any]) -> RedditPost | None:
    if data.get("stickied") or data.get("over_18"):
        return None
    selftext = (data.get("selftext") or "").strip()
    title = (data.get("title") or "").strip()
    if len(selftext) < MIN_SELFTEXT_CHARS and len(title) < MIN_TITLE_CHARS:
        return None
    if not title:
        return None
    permalink = (data.get("permalink") or "").strip()
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"
    sub = (data.get("subreddit") or "").strip() or "unknown"
    post_id = str(data.get("id") or "")
    if not post_id:
        return None
    return RedditPost(
        post_id=post_id,
        subreddit=sub,
        title=title,
        selftext=selftext,
        permalink=permalink,
        score=int(data.get("score") or 0),
    )


def _iter_candidates_pullpush(
    queries: list[tuple[str, str]],
    *,
    per_query: int = 100,
) -> Iterator[RedditPost]:
    seen: set[str] = set()
    for i, (subreddit, query) in enumerate(queries):
        if i > 0:
            time.sleep(_fetch_delay())
        for row in _fetch_pullpush_posts(
            subreddit=subreddit, query=query, size=per_query
        ):
            if not isinstance(row, dict):
                continue
            post = _post_from_pullpush(row)
            if post and post.post_id and post.post_id not in seen:
                seen.add(post.post_id)
                yield post


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
    if not niche_topics:
        raise RuntimeError(
            f"No Muslim/Arab topics in {topics_path}. "
            "Run sync_reddit_topics.sh locally or add niche posts to topics.txt."
        )
    host_topics = _filter_host_persona_topics(niche_topics)
    host_topics = [t for t in host_topics if is_quality_storytime_entry(t)]
    if not host_topics:
        raise RuntimeError(
            f"No topics in {topics_path} match the channel host persona "
            f"({host_persona_gender()}). Add male-host-friendly stories."
        )
    topics = _rank_topics_by_priority(host_topics)
    fresh = [t for t in topics if topic_entry_id(t) not in used_ids]
    pool = fresh if fresh else topics
    pool = _apply_theme_cooldown(pool, used_path, classify_topic_theme)
    # Weighted pick: top-scored topics are more likely.
    top = pool[: min(8, len(pool))]
    weights = [max(1, topic_priority_score(t)) for t in top]
    chosen = random.choices(top, weights=weights, k=1)[0]
    chosen_theme = classify_topic_theme(chosen)
    mark_theme_used(chosen_theme, _theme_file_for(used_path))
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
    for i, source in enumerate(sources):
        if i > 0:
            time.sleep(float(os.environ.get("REDDIT_FETCH_DELAY", "2.0")))
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
    if not niche:
        candidates = []
    else:
        candidates = _filter_host_persona_posts(niche)
        candidates = [p for p in candidates if is_quality_storytime_post(p)]
        if candidates:
            print(
                f"[reddit] {len(candidates)} post(s) after niche + host persona filter.",
                file=sys.stderr,
            )
        else:
            print(
                "[reddit] Niche posts found but none match channel host persona.",
                file=sys.stderr,
            )
            candidates = []
    if not candidates:
        if _should_use_topics_fallback():
            print(
                "[reddit] No host-persona matches on Reddit — falling back to topics.txt.",
                file=sys.stderr,
            )
            return pick_topics_file_fallback(used_file=used_path)
        raise RuntimeError(
            "No Reddit posts match Muslim/Arab niche and channel host persona. "
            "Add topics.txt, set REDDIT_TOPICS_FALLBACK=1, or broaden subreddit sources."
        )

    random.shuffle(candidates)
    fresh = [p for p in candidates if p.post_id not in used_ids]
    pool = fresh if fresh else candidates
    pool.sort(key=lambda p: (topic_priority_score(p.topic_text), p.score), reverse=True)
    pool = _apply_theme_cooldown(pool, used_path, lambda p: classify_topic_theme(p.topic_text))
    top_n = min(8, len(pool))
    top = pool[:top_n]
    weights = [max(1, topic_priority_score(p.topic_text)) for p in top]
    chosen = random.choices(top, weights=weights, k=1)[0]
    chosen_theme = classify_topic_theme(chosen.topic_text)
    mark_theme_used(chosen_theme, _theme_file_for(used_path))
    print(
        f"Reddit topic: r/{chosen.subreddit} | score={chosen.score} | "
        f"theme={chosen_theme} | {chosen.title[:72]!r}"
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
        if not candidates:
            print(
                "[reddit] Public JSON blocked — using pullpush.io (Reddit archive)…",
                file=sys.stderr,
            )
            candidates = list(
                _iter_candidates_pullpush(
                    list(PULLPUSH_HARVEST_BATCHES), per_query=per_source_limit
                )
            )

    seen: set[str] = set()
    unique: list[RedditPost] = []
    for post in candidates:
        if post.post_id and post.post_id not in seen:
            seen.add(post.post_id)
            unique.append(post)
    unique.sort(key=lambda p: p.score, reverse=True)
    return [p for p in unique if is_quality_storytime_post(p)]


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
        if (
            len(text) >= min_entry_chars
            and is_quality_storytime_entry(text)
        ):
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


def harvest_persona_topics(
    *,
    limit: int = 100,
    topics_file: Path | None = None,
    merge: bool = False,
    per_source_limit: int = 100,
    batch: int = 0,
    source: str = "auto",
) -> int:
    """Scrape Reddit for high-scoring posts matching niche + channel host persona."""
    path = topics_file or Path(os.environ.get("TOPICS_FILE", "topics.txt"))
    all_sources = (
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="muslim OR islamophobia OR arab OR middle eastern",
            time_filter="all",
            limit=100,
        ),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="ramadan OR fasting OR halal OR iftar",
            time_filter="all",
            limit=100,
        ),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="terrorist OR profiling OR airport OR TSA",
            time_filter="all",
            limit=100,
        ),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="mosque OR quran OR jummah OR prayer",
            time_filter="year",
            limit=80,
        ),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query='"against arabs" OR "muslim hate" OR islamophob',
            time_filter="all",
            limit=80,
        ),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="cousin marriage OR arranged OR parents want",
            time_filter="year",
            limit=60,
        ),
        SubredditSource("MuslimLounge", kind="top", time_filter="year", limit=100),
        SubredditSource("MuslimLounge", kind="hot", limit=80),
        SubredditSource("ABCDesis", kind="top", time_filter="year", limit=100),
        SubredditSource("ABCDesis", kind="hot", limit=80),
        SubredditSource("islam", kind="top", time_filter="year", limit=80),
        SubredditSource("muslim", kind="top", time_filter="year", limit=80),
        SubredditSource("muslim", kind="hot", limit=60),
        SubredditSource("arabs", kind="top", time_filter="year", limit=80),
        SubredditSource("arabs", kind="hot", limit=60),
        SubredditSource("ArabWorld", kind="hot", limit=50),
        SubredditSource(
            "teenagers",
            kind="search",
            search_query="convert OR reverted OR revert",
            time_filter="year",
            limit=50,
        ),
    )
    batch_size = max(1, int(os.environ.get("REDDIT_HARVEST_BATCH_SIZE", "2")))
    start = max(0, batch) * batch_size
    end = start + batch_size
    pullpush_queries = list(PULLPUSH_HARVEST_BATCHES[start:end])
    if start >= len(all_sources) and not pullpush_queries:
        raise RuntimeError(
            f"Batch {batch} out of range (max batch "
            f"{max((len(all_sources) - 1) // batch_size, (len(PULLPUSH_HARVEST_BATCHES) - 1) // batch_size)})."
        )
    sources = [
        replace(s, limit=per_source_limit)
        for s in all_sources[start:min(end, len(all_sources))]
    ]
    if sources:
        print(
            f"[reddit] Harvest batch {batch}: sources {start + 1}-"
            f"{min(end, len(all_sources))}/{len(all_sources)}",
            file=sys.stderr,
        )
    elif pullpush_queries:
        print(
            f"[reddit] Harvest batch {batch}: pullpush queries "
            f"{start + 1}-{end}/{len(PULLPUSH_HARVEST_BATCHES)}",
            file=sys.stderr,
        )

    candidates: list[RedditPost] = []
    src = (source or "auto").strip().lower()
    if src == "pullpush" and pullpush_queries:
        batch_desc = ", ".join(
            f"r/{s}" + (f" ({q})" if q else " top")
            for s, q in pullpush_queries
        )
        print(
            f"[reddit] Harvest batch {batch}: pullpush {batch_desc}",
            file=sys.stderr,
        )
        candidates = list(
            _iter_candidates_pullpush(pullpush_queries, per_query=per_source_limit)
        )
    elif src in ("auto", "reddit") and sources:
        if _has_praw_credentials():
            print("[reddit] Harvesting via PRAW/API…", file=sys.stderr)
            candidates = list(_iter_candidates(_reddit_client(), sources))
        else:
            print("[reddit] Harvesting via public JSON…", file=sys.stderr)
            candidates = list(_iter_candidates_public(sources))
        if not candidates and pullpush_queries:
            print(
                "[reddit] Public JSON blocked — pullpush.io (Reddit archive)…",
                file=sys.stderr,
            )
            candidates = list(
                _iter_candidates_pullpush(pullpush_queries, per_query=per_source_limit)
            )
    elif src in ("auto", "pullpush") and pullpush_queries:
        candidates = list(
            _iter_candidates_pullpush(pullpush_queries, per_query=per_source_limit)
        )

    if not candidates:
        raise RuntimeError(
            "No Reddit posts fetched this batch. "
            "Try --batch N+1, --source pullpush, or set REDDIT_CLIENT_ID/SECRET."
        )

    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    filtered: list[RedditPost] = []
    for post in candidates:
        if not post.post_id or post.post_id in seen_ids:
            continue
        title_key = re.sub(r"[^a-z0-9]+", " ", post.title.lower()).strip()
        if title_key in seen_titles:
            continue
        text = post.topic_text
        if len(text) < MIN_TOPIC_ENTRY_CHARS:
            continue
        if not is_quality_storytime_post(post):
            continue
        seen_ids.add(post.post_id)
        seen_titles.add(title_key)
        filtered.append(post)

    filtered.sort(key=lambda p: (p.score, topic_priority_score(p.topic_text)), reverse=True)

    existing: list[str] = []
    existing_ids: set[str] = set()
    existing_titles: set[str] = set()
    if merge and path.is_file():
        existing = load_topic_entries(path)
        existing_ids = {topic_entry_id(t) for t in existing}
        for t in existing:
            for line in t.splitlines():
                line = line.strip()
                if line and not line.startswith("Reddit post from r/"):
                    existing_titles.add(
                        re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
                    )
                    break

    chosen: list[RedditPost] = []
    for post in filtered:
        if len(chosen) >= limit:
            break
        eid = topic_entry_id(post.topic_text)
        title_key = re.sub(r"[^a-z0-9]+", " ", post.title.lower()).strip()
        if eid in existing_ids or title_key in existing_titles:
            continue
        chosen.append(post)

    if not chosen:
        raise RuntimeError(
            "No new persona-matching Reddit posts found this batch. "
            "Try --batch N+1, set REDDIT_CLIENT_ID/SECRET, or run on your Mac."
        )
    if len(chosen) < limit:
        print(
            f"[reddit] Warning: only {len(chosen)} new persona topics "
            f"(wanted {limit}).",
            file=sys.stderr,
        )

    entries = [post.topic_text for post in chosen]
    combined = list(existing)
    for entry in entries:
        combined.append(entry)

    path.write_text(TOPICS_ENTRY_SEP.join(combined) + "\n", encoding="utf-8")
    print(
        f"topics.txt: {len(combined)} posts (harvested {len(chosen)} this run) → {path}",
        file=sys.stderr,
    )
    for post in chosen[:8]:
        print(
            f"  • score={post.score} r/{post.subreddit}: {_topic_preview(post.topic_text)!r}",
            file=sys.stderr,
        )
    if len(chosen) > 8:
        print(f"  … and {len(chosen) - 8} more", file=sys.stderr)
    return len(chosen)


def prune_topics_file(topics_file: Path | None = None) -> int:
    """Drop meme/spam/female-persona topics from topics.txt."""
    path = topics_file or Path(os.environ.get("TOPICS_FILE", "topics.txt"))
    entries = load_topic_entries(path)
    kept = [e for e in entries if is_quality_storytime_entry(e)]
    removed = len(entries) - len(kept)
    path.write_text(
        (TOPICS_ENTRY_SEP.join(kept) + "\n") if kept else "",
        encoding="utf-8",
    )
    print(
        f"topics.txt: kept {len(kept)}, removed {removed} → {path}",
        file=sys.stderr,
    )
    return removed


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


def _cli_prune_topics(args: argparse.Namespace) -> int:
    prune_topics_file(Path(args.out))
    return 0


def _cli_harvest_topics(args: argparse.Namespace) -> int:
    harvest_persona_topics(
        limit=args.limit,
        topics_file=Path(args.out),
        merge=args.merge,
        per_source_limit=args.per_source,
        batch=args.batch,
        source=args.source,
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

    harvest = sub.add_parser(
        "harvest-topics",
        help="Scrape top Reddit posts into topics.txt (niche + host persona filter).",
    )
    harvest.add_argument(
        "--out",
        default="topics.txt",
        help="Output topics file (default: topics.txt).",
    )
    harvest.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max topics to write (sorted by Reddit score).",
    )
    harvest.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing topics.txt instead of replacing.",
    )
    harvest.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Source batch index (4 sources per batch). Run 0,1,2… for 20 at a time.",
    )
    harvest.add_argument(
        "--source",
        choices=("auto", "reddit", "pullpush"),
        default="auto",
        help="auto=Reddit then pullpush archive; pullpush=Reddit archive only.",
    )
    harvest.add_argument(
        "--per-source",
        type=int,
        default=100,
        help="Max posts to fetch per subreddit/search source.",
    )
    harvest.set_defaults(func=_cli_harvest_topics)

    prune = sub.add_parser(
        "prune-topics",
        help="Remove low-quality / wrong-persona entries from topics.txt.",
    )
    prune.add_argument(
        "--out",
        default="topics.txt",
        help="Topics file to clean (default: topics.txt).",
    )
    prune.set_defaults(func=_cli_prune_topics)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
