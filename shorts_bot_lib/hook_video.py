"""Hook intro video: fetch a short relevant clip for the first seconds."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI

from .image_judge import judge_image


PEXELS_VIDEO_SEARCH = "https://api.pexels.com/videos/search"


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (slug[:40] or "hook").strip("_")


def build_hook_video_queries(
    client: OpenAI | None, script: str, base_query: str | None = None
) -> List[str]:
    """Produce 4-6 ranked stock-video search queries for the hook clip.

    The queries must depict the OVERALL TOPIC of the video (not just the
    first sentence) — e.g. hijab conflict at school, airport profiling, Ramadan
    fasting, family dinner, mosque, yearbook photo line.
    """
    fallback: List[str] = []
    if base_query:
        fallback.append(base_query.strip())
    fallback.extend(["teen hijab school", "airport security line", "family dinner table"])
    fallback = [q for q in dict.fromkeys(fallback) if q]

    if client is None:
        return fallback[:5]

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("HOOK_VIDEO_QUERY_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate stock-video search queries for the FIRST 1.5 seconds "
                        "of a Muslim/Arab teen storytime YouTube Short. Queries must "
                        "produce a CINEMATIC, eye-catching clip for the overall story "
                        "(hijab at school, mosque, airport line, family kitchen, Ramadan, "
                        "yearbook, etc. — NOT just the first sentence). Use 2-4 visual "
                        "words per query. Prefer concrete motion (walking hallway, "
                        "security scanner, dinner table). No proper names."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON: {\"queries\": [\"...\", ...]} with 5 queries ranked "
                        "from MOST visually fitting to fallback-generic. They should "
                        "visually evoke the TOPIC/THEME of the entire script below.\n\n"
                        f"SCRIPT:\n{script}\n"
                    ),
                },
            ],
            temperature=0.5,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
        cleaned: List[str] = []
        for q in queries:
            q = re.sub(r"[^a-zA-Z0-9 ]", " ", q).strip()
            q = re.sub(r"\s+", " ", q)
            if q and q.lower() not in {c.lower() for c in cleaned}:
                cleaned.append(q)
        if cleaned:
            return cleaned[:6]
    except Exception as exc:
        print(f"[hook-video] query LLM failed: {exc}")
    return fallback[:5]


def _pexels_search(api_key: str, query: str, per_page: int = 12) -> list[dict]:
    try:
        r = requests.get(
            PEXELS_VIDEO_SEARCH,
            headers={"Authorization": api_key},
            params={
                "query": query,
                "per_page": per_page,
                "orientation": "portrait",
                "size": "large",
            },
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[pexels-video] search '{query}' failed ({r.status_code}): {r.text[:160]}")
            return []
        return r.json().get("videos") or []
    except Exception as exc:
        print(f"[pexels-video] search '{query}' error: {exc}")
        return []


def _best_pexels_file(video: dict) -> tuple[str | None, int]:
    """Return (url, score) for the best file in a Pexels video record."""
    best_url: str | None = None
    best_score = -1
    for f in video.get("video_files") or []:
        link = f.get("link")
        w = int(f.get("width") or 0)
        h = int(f.get("height") or 0)
        if not link or w <= 0 or h <= 0:
            continue
        portraitish = 1 if h >= w else 0
        # Cap rewarded resolution so we don't always grab giant 4K files.
        capped = min(w * h, 1080 * 1920)
        score = portraitish * 10_000_000 + capped
        if score > best_score:
            best_score = score
            best_url = link
    return best_url, best_score


def fetch_pexels_hook_video(query: str, out_dir: Path) -> Path | None:
    """Backwards-compatible single-query fetch."""
    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not api_key:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = _pexels_search(api_key, query)
    if not videos:
        return None
    best_url: str | None = None
    best_score = -1
    for v in videos:
        url, score = _best_pexels_file(v)
        # Slight preference for short, punchy clips (3-15s).
        dur = float(v.get("duration") or 0.0)
        if 3.0 <= dur <= 15.0:
            score += 250_000
        if score > best_score:
            best_score = score
            best_url = url
    if not best_url:
        return None
    out_path = out_dir / f"hook_{_safe_slug(query)}_{int(time.time())}.mp4"
    try:
        vr = requests.get(best_url, timeout=60)
        if vr.status_code != 200:
            print(f"[pexels-video] download failed ({vr.status_code})")
            return None
        out_path.write_bytes(vr.content)
        return out_path
    except Exception as exc:
        print(f"[pexels-video] download error: {exc}")
        return None


def fetch_best_hook_video(
    client: OpenAI | None,
    script: str,
    out_dir: Path,
    base_query: str | None = None,
) -> Path | None:
    """Try several relevant queries, judge candidates, return the best clip.

    Strategy:
    1. Build a ranked query list from the script.
    2. For each query, pull top Pexels candidates, download, judge first frame
       for "no text" + relevance to the script hook. Pick the best scoring one.
    """
    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not api_key:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = build_hook_video_queries(client, script, base_query)
    if not queries:
        return None
    print(f"[hook-video] queries: {queries}")

    expected = (
        "A cinematic vertical stock clip whose first frame visually represents "
        "the OVERALL TOPIC / THEME / setting of the script (not just the first "
        "sentence). NO on-screen text, subtitles, logos, watermarks, or "
        "talking-head presenters. Eye-catching, in-motion, well-lit."
    )

    best_path: Path | None = None
    best_score = -1
    tried_urls: set[str] = set()

    for query in queries:
        videos = _pexels_search(api_key, query, per_page=8)
        if not videos:
            continue
        # Per query, evaluate up to 3 candidates; stop early on a clear winner.
        ranked: list[tuple[int, str, dict]] = []
        for v in videos:
            url, score = _best_pexels_file(v)
            if not url or url in tried_urls:
                continue
            dur = float(v.get("duration") or 0.0)
            if 3.0 <= dur <= 15.0:
                score += 250_000
            ranked.append((score, url, v))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, url, _v in ranked[:3]:
            tried_urls.add(url)
            tmp_path = out_dir / f"hook_cand_{_safe_slug(query)}_{int(time.time()*1000)}.mp4"
            try:
                vr = requests.get(url, timeout=60)
                if vr.status_code != 200:
                    continue
                tmp_path.write_bytes(vr.content)
            except Exception as exc:
                print(f"[hook-video] download error: {exc}")
                continue

            verdict_score = _judge_candidate(client, tmp_path, expected)
            if verdict_score < 0:
                tmp_path.unlink(missing_ok=True)
                continue
            print(f"[hook-video] candidate '{query}' scored {verdict_score}")
            if verdict_score > best_score:
                if best_path is not None:
                    best_path.unlink(missing_ok=True)
                best_path = tmp_path
                best_score = verdict_score
                if best_score >= 9:
                    break
            else:
                tmp_path.unlink(missing_ok=True)
        if best_score >= 9:
            break

    if best_path is None:
        # Last ditch: just take the first query's best clip without judging.
        return fetch_pexels_hook_video(queries[0], out_dir)
    final_path = out_dir / f"hook_{_safe_slug(queries[0])}_{int(time.time())}.mp4"
    try:
        best_path.rename(final_path)
        return final_path
    except Exception:
        return best_path


def _judge_candidate(client: OpenAI | None, video_path: Path, expected: str) -> int:
    """Extract first frame and run image judge; return score (or -1 if rejected hard)."""
    if client is None:
        return 5  # neutral pass when no judge
    if not video_path.exists():
        return -1
    frame_path = video_path.with_suffix(".frame0.png")
    try:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                "select=eq(n\\,0),scale=540:-1",
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return -1
    if not frame_path.exists():
        return -1
    verdict = judge_image(
        client=client,
        image_path=frame_path,
        expected=expected,
        min_score=6,
    )
    try:
        frame_path.unlink(missing_ok=True)
    except Exception:
        pass
    if not verdict.ok:
        print(f"[hook-video] reject (score={verdict.score}): {verdict.reason}")
        return -1
    return verdict.score


def generate_brainrot_text(client: OpenAI | None, script: str) -> str:
    """Return a short, bold overlay text."""
    first = (script.split(".")[0] or script).strip()
    fallback = (" ".join(first.split()[:4]) or "NO WAY").upper()
    if client is None:
        return fallback[:18]
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("HOOK_TEXT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Create a 1-4 word clickbait/brainrot overlay text for a Shorts hook. "
                        "ALL CAPS. No punctuation. No emojis. Must relate to the hook.\n\n"
                        f"HOOK:\n{first}\n"
                    ),
                }
            ],
            temperature=0.8,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text).strip().upper()
        text = re.sub(r"\s+", " ", text)
        return (text or fallback)[:18]
    except Exception:
        return fallback[:18]


def hook_video_has_no_text(client: OpenAI | None, video_path: Path) -> bool:
    """Extract a frame and reject videos that contain on-screen text/logos."""
    if client is None:
        return True
    if not video_path.exists():
        return False
    frame_path = video_path.with_suffix(".frame0.png")
    try:
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0),scale=540:-1",
            "-frames:v",
            "1",
            str(frame_path),
        ]
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return True
    if not frame_path.exists():
        return True
    verdict = judge_image(
        client=client,
        image_path=frame_path,
        expected="A clean stock clip frame with NO on-screen text, subtitles, logos, or watermarks.",
        min_score=8,
    )
    try:
        frame_path.unlink(missing_ok=True)
    except Exception:
        pass
    if not verdict.ok:
        print(f"[hook-video] rejected (score={verdict.score}): {verdict.reason}")
    return verdict.ok

