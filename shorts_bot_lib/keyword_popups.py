"""LLM-picked keyword/phrase popups timed to when the words are spoken.

Unlike `scene_assets`, which produces continuous-coverage scene images,
this module asks the LLM to pick the most VISUAL punchy phrases in the
script (named things, actions, objects, places) and lands a tightly
timed popup right when each phrase is narrated.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from .scene_assets import Scene, _fetch_scene_image, _with_retries
from .types import PopupImage


KEYWORD_LLM_MODEL = os.environ.get("KEYWORD_LLM_MODEL", "gpt-4o-mini")

# When the next keyword is spoken soon after this one, shorten this popup's
# end (never its start) so both can appear; never shorter than this many seconds.
_MIN_KEYWORD_VISIBLE_SEC = 0.28


_KEYWORD_SYSTEM = (
    "You pick the most VISUAL, punchy phrases from a YouTube Shorts narration "
    "script — words that benefit from being illustrated with a single still "
    "image. Each pick must be a verbatim 1-4 word substring of the script, "
    "and ideally a concrete noun, named entity, action, place, or object "
    "(e.g. 'burning Moscow', 'cannon fire', 'snowy steppe', 'crown', "
    "'duel', 'flag'). Skip filler, conjunctions, and abstract words. "
    "For each pick, also write a 2-4 word stock-photo / image search "
    "query (visual nouns + adjectives only, no proper names, no verbs)."
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _build_keyword_user_prompt(script_text: str, target_count: int) -> str:
    return (
        "Return ONLY valid JSON in this exact shape:\n"
        '{"keywords": ['
        '{"phrase": "<verbatim 1-4 word phrase from the script>", '
        '"query": "<2-4 word stock-photo query>"}, ...'
        "]}\n\n"
        f"Pick about {target_count} keywords. They MUST appear verbatim "
        "in the script. Spread them evenly through the script.\n\n"
        f"SCRIPT:\n{script_text.strip()}"
    )


def extract_keywords(
    client: OpenAI, script_text: str, target_count: int
) -> List[dict]:
    user_prompt = _build_keyword_user_prompt(script_text, target_count)

    def _call() -> str:
        resp = client.chat.completions.create(
            model=KEYWORD_LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _KEYWORD_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    raw = _with_retries(_call, label="keyword-llm")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON for keywords:\n{raw}") from exc

    items: List[dict] = []
    seen: set[str] = set()
    norm_script = _normalize(script_text)
    for entry in data.get("keywords") or []:
        phrase = str(entry.get("phrase") or "").strip()
        query = str(entry.get("query") or "").strip()
        if not phrase or not query:
            continue
        norm = _normalize(phrase)
        if not norm or norm in seen:
            continue
        if norm not in norm_script:
            # Skip phrases the LLM hallucinated.
            continue
        seen.add(norm)
        items.append({"phrase": phrase, "query": re.sub(r"[^a-zA-Z0-9 ]", " ", query).strip()})
    return items


def _find_phrase_window(
    phrase: str, word_segments: List[dict]
) -> Optional[tuple[float, float]]:
    """Find when `phrase` is spoken using a sliding window over whisper words."""
    if not word_segments:
        return None
    target_tokens = _normalize(phrase).split()
    if not target_tokens:
        return None
    window = len(target_tokens)
    norm_words: List[tuple[str, float, float]] = []
    for seg in word_segments:
        word = str(seg.get("text") or seg.get("raw_text") or "").strip()
        norm = _normalize(word)
        if not norm:
            continue
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        norm_words.append((norm, start, end))

    target_str = " ".join(target_tokens)
    for i in range(0, len(norm_words) - window + 1):
        chunk_tokens = [norm_words[i + j][0] for j in range(window)]
        if " ".join(chunk_tokens) == target_str:
            return norm_words[i][1], norm_words[i + window - 1][2]

    # Fallback: match just the first/most-distinctive token.
    head = target_tokens[0]
    for w, s, e in norm_words:
        if w == head:
            return s, e
    return None


def build_keyword_popups(
    *,
    client: OpenAI,
    script_text: str,
    word_segments: List[dict],
    narration_duration: float,
    out_dir: Path,
    popup_width: int = 700,
    popup_y: int = 860,
    popup_play_sfx: bool = True,
    target_count: int | None = None,
    popup_duration: float = 1.6,
    min_gap: float = 0.0,
) -> tuple[List[PopupImage], List[dict]]:
    """Pick keywords and produce timed popups when each phrase is spoken."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pexels_key = os.environ.get("PEXELS_API_KEY")
    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if target_count is None:
        # ~0.6 keywords per second (≈6 popups for a 10s clip), capped by 75% of seconds.
        cap = max(6, int(round(narration_duration * 0.75)))
        target_count = max(6, min(cap, int(round(narration_duration * 0.6))))
    print(f"Picking ~{target_count} keyword popups for {narration_duration:.1f}s narration...")

    keywords = extract_keywords(client, script_text, target_count)
    if not keywords:
        print("Keyword extraction returned nothing; no keyword popups.")
        return [], []
    print(f"Got {len(keywords)} keyword(s); aligning + fetching images...")

    # Align phrases to timestamps, drop ones we can't time.
    timed: List[tuple[dict, float, float]] = []
    for kw in keywords:
        win = _find_phrase_window(kw["phrase"], word_segments)
        if win is None:
            print(f"  skip (untimed): {kw['phrase']!r}")
            continue
        start, end = win
        # Center a fixed-duration popup around the phrase.
        center = 0.5 * (start + end)
        half = popup_duration / 2.0
        s = max(0.0, center - half)
        e = min(narration_duration - 0.05, center + half)
        if e <= s:
            continue
        timed.append((kw, s, e))

    if not timed:
        return [], []

    # Sort by start; trim each popup's end only when the next would start too soon
    # (keeps each start tied to the spoken phrase; drops a keyword only if it
    # cannot show at all after trimming).
    timed.sort(key=lambda t: t[1])
    spaced: List[tuple[dict, float, float]] = []
    n = len(timed)
    tail = narration_duration - 0.05
    for i in range(n):
        kw, s, e = timed[i]
        if i + 1 < n:
            s_next = timed[i + 1][1]
            max_end = min(s_next - min_gap, tail)
        else:
            max_end = tail
        e = min(e, max_end)
        if e <= s:
            continue
        if e - s < _MIN_KEYWORD_VISIBLE_SEC:
            stretched = min(s + _MIN_KEYWORD_VISIBLE_SEC, max_end)
            if stretched <= s:
                continue
            e = stretched
        spaced.append((kw, s, e))

    popups: List[PopupImage] = []
    mapping: List[dict] = []
    width = popup_width
    x = (1080 - width) // 2

    for idx, (kw, s, e) in enumerate(spaced, start=1):
        scene = Scene(index=idx, text=kw["phrase"], query=kw["query"], word_count=1)
        print(f"[kw {idx:02d}] {kw['phrase']!r} -> query={kw['query']!r} ({s:.2f}-{e:.2f}s)")
        path = _fetch_scene_image(
            scene=scene,
            out_dir=out_dir,
            openai_client=client,
            pexels_key=pexels_key,
            unsplash_key=unsplash_key,
            gemini_key=gemini_key,
        )
        if path is None:
            mapping.append(
                {
                    "phrase": kw["phrase"],
                    "query": kw["query"],
                    "start": s,
                    "end": e,
                    "image_path": None,
                    "source": None,
                }
            )
            continue

        if path.name.endswith("_generated.png"):
            source = "gemini"
        elif path.name.endswith("_generated.jpg"):
            source = "openai"
        else:
            source = "stock"
        popups.append(
            PopupImage(
                path=path,
                start_sec=s,
                end_sec=e,
                x=x,
                y=popup_y,
                width=width,
                play_sfx=popup_play_sfx,
                use_fade=True,
            )
        )
        mapping.append(
            {
                "phrase": kw["phrase"],
                "query": kw["query"],
                "start": s,
                "end": e,
                "image_path": str(path),
                "source": source,
            }
        )
    return popups, mapping
