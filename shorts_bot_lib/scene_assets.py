"""Scene-aware popup images for the Shorts pipeline.

Splits the narration script into scenes via an LLM, fetches a portrait
stock photo for each scene (Pexels primary, Unsplash secondary), and
falls back to OpenAI image generation when stock returns nothing.

Each scene is materialised as a `PopupImage` whose `start_sec` /
`end_sec` are derived from the narration timing, so popups land on
the part of the audio that actually describes them.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI

from .images import (
    _gemini_client,
    _gemini_extract_inline_image_bytes,
    _gemini_image_models,
)
from .image_judge import judge_image
from .types import PopupImage


PEXELS_BASE = "https://api.pexels.com/v1/search"
UNSPLASH_BASE = "https://api.unsplash.com/search/photos"

LLM_MODEL = "gpt-4o-mini"
IMAGE_GEN_MODEL = "gpt-image-1"

REQUEST_TIMEOUT = 30


# --------------------------------------------------------------------------- #
# Retry helpers
# --------------------------------------------------------------------------- #

class RateLimited(Exception):
    """Raised when an upstream returns 429 even after retries."""


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _with_retries(fn, *, attempts: int = 5, base_delay: float = 1.5, label: str = "request"):
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            transient = _is_rate_limit_error(exc) or isinstance(exc, requests.RequestException)
            if not transient or attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(
                f"[{label}] attempt {attempt} hit transient error, "
                f"sleeping {delay:.1f}s ({exc})"
            )
            time.sleep(delay)
    raise RateLimited(f"{label} exhausted retries") from last_exc


# --------------------------------------------------------------------------- #
# Scene extraction (LLM)
# --------------------------------------------------------------------------- #

@dataclass
class Scene:
    index: int
    text: str
    query: str
    word_count: int

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.query.lower()).strip("_")
        return slug[:48] or f"scene_{self.index:02d}"


_SCENE_SYSTEM = (
    "You split a narration script into visual scenes for a YouTube Short. "
    "Each scene gets a short stock-photo search query (2-5 words, nouns + "
    "visual adjectives only, no proper names, no quotes, no punctuation, no "
    "verbs). Create a NEW scene for every meaningful visual change, action, "
    "subject, or location shift. Err on the side of MORE scenes, not fewer, "
    "so the final video has constant fresh imagery."
)


_SCENE_USER = """Return ONLY valid JSON in this exact shape:

{{
  "scenes": [
    {{"text": "<verbatim consecutive text from the script>", "query": "<2-5 word visual query>"}}
  ]
}}

Rules:
- Concatenated `text` fields, in order, must reproduce the script verbatim
  (you may merge consecutive sentences but never reorder or paraphrase).
- Aim for ~{target_scenes} scenes total (between {min_scenes} and {max_scenes}).
  Roughly one scene every 2 seconds of narration.

SCRIPT:
{script}"""


def extract_scenes(
    client: OpenAI, script_text: str, narration_duration: float | None = None
) -> List[Scene]:
    if narration_duration and narration_duration > 0:
        target_scenes = max(6, int(round(narration_duration / 2.0)))
    else:
        target_scenes = 12
    min_scenes = max(4, target_scenes - 2)
    max_scenes = target_scenes + 3
    user_prompt = _SCENE_USER.format(
        script=script_text.strip(),
        target_scenes=target_scenes,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    def _call():
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SCENE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    raw = _with_retries(_call, label="openai-llm")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON for scene extraction:\n{raw}") from exc

    raw_scenes = data.get("scenes") or []
    scenes: List[Scene] = []
    for i, item in enumerate(raw_scenes, start=1):
        text = str(item.get("text") or "").strip()
        query = str(item.get("query") or "").strip()
        query = re.sub(r"[^a-zA-Z0-9 ]", " ", query).strip()
        if not text or not query:
            continue
        word_count = max(1, len(text.split()))
        scenes.append(Scene(index=i, text=text, query=query, word_count=word_count))
    if not scenes:
        raise RuntimeError("LLM returned no usable scenes.")
    return scenes


# --------------------------------------------------------------------------- #
# Image providers
# --------------------------------------------------------------------------- #

def _search_pexels(api_key: str, query: str) -> Optional[str]:
    def _call():
        r = requests.get(
            PEXELS_BASE,
            headers={"Authorization": api_key},
            params={
                "query": query,
                "per_page": 5,
                "orientation": "portrait",
                "size": "large",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 429:
            raise requests.HTTPError("429 rate limited", response=r)
        r.raise_for_status()
        return r.json()

    try:
        data = _with_retries(_call, label="pexels")
    except Exception as exc:
        print(f"[pexels] search failed for {query!r}: {exc}")
        return None

    photos = data.get("photos") or []
    if not photos:
        return None
    best = photos[0]
    return best.get("src", {}).get("portrait") or best.get("src", {}).get("large")


def _search_unsplash(access_key: str, query: str) -> Optional[str]:
    def _call():
        r = requests.get(
            UNSPLASH_BASE,
            headers={"Authorization": f"Client-ID {access_key}"},
            params={
                "query": query,
                "per_page": 5,
                "orientation": "portrait",
                "content_filter": "high",
                "order_by": "relevant",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 429:
            raise requests.HTTPError("429 rate limited", response=r)
        r.raise_for_status()
        return r.json()

    try:
        data = _with_retries(_call, label="unsplash")
    except Exception as exc:
        print(f"[unsplash] search failed for {query!r}: {exc}")
        return None

    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("urls", {}).get("regular")


def _download_image(url: str, dst: Path) -> bool:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:
        print(f"[download] failed for {url}: {exc}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(r.content)
    return True


def _generate_gemini_image(api_key: str, query: str, dst: Path) -> bool:
    """Generate a scene image with Gemini, trying the model fallback chain."""
    prompt = (
        f"{query}. Cinematic, photorealistic stock-photo style, "
        f"vertical 9:16 composition, soft natural lighting, no text, "
        f"no logos, no watermark."
    )
    try:
        client = _gemini_client(api_key)
    except Exception as exc:
        print(f"[gemini-img] client unavailable: {exc}")
        return False
    try:
        from google.genai import types  # type: ignore[import-not-found]
    except Exception:
        types = None
    config = (
        types.GenerateContentConfig(response_modalities=["IMAGE"])
        if types is not None
        else None
    )

    last_exc: Optional[Exception] = None
    for model in _gemini_image_models():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            data = _gemini_extract_inline_image_bytes(response)
            if data:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                print(f"[gemini-img] {model} -> {dst.name}")
                return True
            print(f"[gemini-img] {model} returned no inline image data.")
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                print(f"[gemini-img] {model} unavailable, trying next...")
                continue
            print(f"[gemini-img] {model} failed: {exc}")
            continue
    if last_exc is not None:
        print(f"[gemini-img] all models failed. Last error: {last_exc}")
    return False


def _generate_openai_image(client: OpenAI, query: str, dst: Path) -> bool:
    prompt = (
        f"{query}. Cinematic, photorealistic stock-photo style, "
        f"vertical 9:16 composition, soft natural lighting, no text, "
        f"no logos, no watermark."
    )

    def _call():
        return client.images.generate(
            model=IMAGE_GEN_MODEL,
            prompt=prompt,
            size="1024x1536",
            n=1,
        )

    try:
        response = _with_retries(_call, label="openai-image")
    except Exception as exc:
        print(f"[openai-img] image generation failed: {exc}")
        return False

    items = getattr(response, "data", None) or []
    if not items:
        return False
    entry = items[0]
    b64 = getattr(entry, "b64_json", None)
    url = getattr(entry, "url", None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if b64:
        dst.write_bytes(base64.b64decode(b64))
        return True
    if url:
        return _download_image(url, dst)
    return False


def _fetch_scene_image(
    scene: Scene,
    out_dir: Path,
    openai_client: OpenAI,
    pexels_key: Optional[str],
    unsplash_key: Optional[str],
    gemini_key: Optional[str],
) -> Optional[Path]:
    base_name = f"scene_{scene.index:02d}_{scene.slug}"
    target_jpg = out_dir / f"{base_name}.jpg"
    target_gemini = out_dir / f"{base_name}_generated.png"
    target_openai = out_dir / f"{base_name}_generated.jpg"
    expected = (
        f"A sharp, relevant portrait-friendly scene image for: {scene.query}. "
        "No watermark, no overlaid text."
    )

    if pexels_key:
        url = _search_pexels(pexels_key, scene.query)
        if url and _download_image(url, target_jpg):
            print(f"   pexels    -> {target_jpg.name}")
            verdict = judge_image(client=openai_client, image_path=target_jpg, expected=expected)
            if verdict.ok:
                return target_jpg
            print(f"   judge REJECT stock (score={verdict.score}): {verdict.reason}")
            if gemini_key and _generate_gemini_image(gemini_key, scene.query, target_gemini):
                return target_gemini

    if unsplash_key:
        url = _search_unsplash(unsplash_key, scene.query)
        if url and _download_image(url, target_jpg):
            print(f"   unsplash  -> {target_jpg.name}")
            verdict = judge_image(client=openai_client, image_path=target_jpg, expected=expected)
            if verdict.ok:
                return target_jpg
            print(f"   judge REJECT stock (score={verdict.score}): {verdict.reason}")
            if gemini_key and _generate_gemini_image(gemini_key, scene.query, target_gemini):
                return target_gemini

    if gemini_key and _generate_gemini_image(gemini_key, scene.query, target_gemini):
        verdict = judge_image(client=openai_client, image_path=target_gemini, expected=expected)
        if verdict.ok:
            return target_gemini
        print(f"   judge REJECT gemini (score={verdict.score}): {verdict.reason}")

    if _generate_openai_image(openai_client, scene.query, target_openai):
        print(f"   openai    -> {target_openai.name}")
        verdict = judge_image(client=openai_client, image_path=target_openai, expected=expected)
        if verdict.ok:
            return target_openai
        print(f"   judge REJECT openai (score={verdict.score}): {verdict.reason}")
        if gemini_key and _generate_gemini_image(gemini_key, scene.query, target_gemini):
            verdict2 = judge_image(client=openai_client, image_path=target_gemini, expected=expected)
            if verdict2.ok:
                return target_gemini
            print(f"   judge REJECT gemini replacement (score={verdict2.score}): {verdict2.reason}")
        return target_openai

    print(f"   FAILED to obtain image for scene {scene.index}")
    return None


# --------------------------------------------------------------------------- #
# Timing: place scenes inside the narration
# --------------------------------------------------------------------------- #

def _scene_time_windows(
    scenes: List[Scene], narration_duration: float
) -> List[tuple[float, float]]:
    """Distribute scenes across the narration in proportion to their word count."""
    duration = max(0.5, float(narration_duration))
    total_words = sum(s.word_count for s in scenes)
    if total_words <= 0:
        per = duration / max(1, len(scenes))
        return [(i * per, (i + 1) * per) for i in range(len(scenes))]

    windows: List[tuple[float, float]] = []
    cursor = 0.0
    for scene in scenes:
        share = (scene.word_count / total_words) * duration
        start = cursor
        end = min(duration, cursor + share)
        windows.append((start, end))
        cursor = end
    if windows:
        last_start, _last_end = windows[-1]
        windows[-1] = (last_start, duration)
    return windows


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_scene_popups(
    *,
    client: OpenAI,
    script_text: str,
    narration_duration: float,
    out_dir: Path,
    popup_width: int = 700,
    popup_y: int = 860,
    popup_play_sfx: bool = True,
    popup_lead_in: float = 0.05,
    popup_tail_buffer: float = 0.05,
) -> tuple[List[PopupImage], List[dict]]:
    """Generate scene-matched stock/AI images and return timed popups.

    Returns
    -------
    popups : list of PopupImage timed across the narration.
    mapping : JSON-serialisable list of `{scene, text, query, image_path, source}`
              suitable for writing alongside the video for editor reuse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pexels_key = os.environ.get("PEXELS_API_KEY")
    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not pexels_key and not unsplash_key:
        print(
            "WARN: no PEXELS_API_KEY or UNSPLASH_ACCESS_KEY set "
            "\u2014 every scene will fall through to AI generation."
        )

    print("Extracting scenes for popup assets via LLM...")
    scenes = extract_scenes(client, script_text, narration_duration=narration_duration)
    print(f"Got {len(scenes)} scene(s) for {narration_duration:.1f}s narration; fetching images...")

    windows = _scene_time_windows(scenes, narration_duration)

    popups: List[PopupImage] = []
    mapping: List[dict] = []
    width = popup_width
    x = (1080 - width) // 2

    for scene, (start, end) in zip(scenes, windows):
        print(f"[{scene.index:02d}] query={scene.query!r}")
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
                    "scene": scene.index,
                    "text": scene.text,
                    "query": scene.query,
                    "image_path": None,
                    "source": None,
                    "start": start,
                    "end": end,
                }
            )
            continue

        if path.name.endswith("_generated.png"):
            source = "gemini"
        elif path.name.endswith("_generated.jpg"):
            source = "openai"
        else:
            source = "stock"
        clipped_start = max(0.0, float(start) + popup_lead_in)
        clipped_end = max(clipped_start + 0.6, float(end) - popup_tail_buffer)
        popups.append(
            PopupImage(
                path=path,
                start_sec=clipped_start,
                end_sec=clipped_end,
                x=x,
                y=popup_y,
                width=width,
                play_sfx=popup_play_sfx,
                use_fade=True,
            )
        )
        mapping.append(
            {
                "scene": scene.index,
                "text": scene.text,
                "query": scene.query,
                "image_path": str(path),
                "source": source,
                "start": clipped_start,
                "end": clipped_end,
            }
        )

    return popups, mapping
