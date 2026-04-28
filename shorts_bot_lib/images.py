"""Image-related pipeline pieces: hook image, popups, story-related image picker."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import List

import requests
from openai import OpenAI

from .emoji import download_twemoji_png, pick_non_repeating_emoji
from .script_ai import build_dalle_prompt
from .subtitles import split_caption_chunks
from .text import format_caption_multiline, random_caps_text, text_keywords
from .types import PopupImage


UNSPLASH_ACCESS_KEY = os.environ.get(
    "UNSPLASH_ACCESS_KEY",
    "3U8WmU73GKXwREspDQOKbf4e4mLGT4df4Bfp337klWQ",
)


# ---------------------------------------------------------------------------
# Gemini hook image (with OpenAI fallback)
# ---------------------------------------------------------------------------


def _gemini_client(api_key: str):
    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Missing `google-genai`. Install it with `pip install -U google-genai`."
        ) from exc
    return genai.Client(api_key=api_key)


def _gemini_image_models() -> list[str]:
    env_model = os.environ.get("GEMINI_IMAGE_MODEL", "").strip()
    fallbacks = [
        "gemini-3-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-preview-image-generation",
    ]
    if env_model:
        return [env_model] + [m for m in fallbacks if m != env_model]
    return fallbacks


def _gemini_extract_inline_image_bytes(response) -> bytes | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if data:
            return bytes(data)
    return None


def _try_gemini_hook_image(
    gemini_api_key: str, prompt: str, out_dir: Path
) -> Path | None:
    try:
        client = _gemini_client(gemini_api_key)
    except Exception as exc:
        print(f"Gemini client unavailable: {exc}")
        return None
    try:
        from google.genai import types  # type: ignore[import-not-found]
    except Exception:
        types = None
    config = (
        types.GenerateContentConfig(response_modalities=["IMAGE"])
        if types is not None
        else None
    )
    last_exc: Exception | None = None
    for model in _gemini_image_models():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            data = _gemini_extract_inline_image_bytes(response)
            if data:
                out_path = out_dir / f"hook_gemini_{int(time.time())}.png"
                out_path.write_bytes(data)
                print(f"Hook image generated with Gemini model: {model}")
                return out_path
            print(f"Gemini model {model} returned no inline image data.")
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                print(f"Gemini model {model} unavailable, trying next...")
                continue
            print(f"Gemini model {model} failed: {exc}")
            continue
    if last_exc is not None:
        print(f"All Gemini image models failed. Last error: {last_exc}")
    return None


def _try_openai_hook_image(
    openai_client: OpenAI, prompt: str, out_dir: Path
) -> Path | None:
    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1536",
            quality="medium",
            n=1,
        )
    except Exception as exc:
        print(f"OpenAI hook image generation error: {exc}")
        return None
    data = getattr(response, "data", None) or []
    if not data:
        print("OpenAI image model returned no data.")
        return None
    entry = data[0]
    out_path = out_dir / f"hook_openai_{int(time.time())}.png"
    b64 = getattr(entry, "b64_json", None)
    image_url = getattr(entry, "url", None)
    if b64:
        import base64
        out_path.write_bytes(base64.b64decode(b64))
    elif image_url:
        image_response = requests.get(image_url, timeout=60)
        if image_response.status_code != 200:
            print(f"OpenAI image download failed: {image_response.status_code}")
            return None
        out_path.write_bytes(image_response.content)
    else:
        print("OpenAI image model returned neither b64_json nor url.")
        return None
    print("Hook image generated with OpenAI gpt-image-1 (medium).")
    return out_path


def generate_hook_image(
    openai_client: OpenAI | None,
    gemini_api_key: str | None,
    script: str,
    out_dir: Path,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_dalle_prompt(openai_client, script)
    print(f"Hook image prompt: {prompt[:200]}")
    if gemini_api_key:
        result = _try_gemini_hook_image(gemini_api_key, prompt, out_dir)
        if result is not None:
            return result
        print("Falling back to OpenAI gpt-image-1...")
    if openai_client is not None:
        return _try_openai_hook_image(openai_client, prompt, out_dir)
    return None


# ---------------------------------------------------------------------------
# Unsplash fallback
# ---------------------------------------------------------------------------


def _unsplash_search(query: str) -> list | None:
    if not UNSPLASH_ACCESS_KEY:
        return None
    try:
        response = requests.get(
            "https://api.unsplash.com/search/photos",
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            params={
                "query": query,
                "per_page": 15,
                "orientation": "portrait",
                "content_filter": "high",
                "order_by": "relevant",
            },
            timeout=30,
        )
        if response.status_code != 200:
            print(f"Unsplash search failed ({response.status_code}): {response.text[:200]}")
            return None
        return (response.json().get("results") or []) or []
    except Exception as exc:
        print(f"Unsplash fetch error: {exc}")
        return None


def fetch_unsplash_hook_image(query: str, out_dir: Path) -> Path | None:
    if not UNSPLASH_ACCESS_KEY:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    candidates = [query]
    words = [w for w in query.split() if w]
    if len(words) >= 2:
        candidates.append(" ".join(words[:2]))
    if words:
        candidates.append(words[0])
    candidates.append("mood")

    results = []
    final_query = query
    for attempt in candidates:
        if not attempt or attempt in tried:
            continue
        tried.append(attempt)
        attempt_results = _unsplash_search(attempt)
        if attempt_results:
            results = attempt_results
            final_query = attempt
            break
    if not results:
        print(f"Unsplash returned no images for any query (tried: {tried}).")
        return None

    top_relevant = results[:8]
    best = max(top_relevant, key=lambda r: int(r.get("likes") or 0))
    image_url = best.get("urls", {}).get("regular")
    if not image_url:
        return None
    try:
        image_response = requests.get(image_url, timeout=60)
        if image_response.status_code != 200:
            return None
    except Exception as exc:
        print(f"Unsplash download error: {exc}")
        return None
    safe_query = re.sub(r"[^a-zA-Z0-9]+", "_", final_query)[:40] or "hook"
    out_path = out_dir / f"hook_{safe_query}.jpg"
    out_path.write_bytes(image_response.content)
    return out_path


# ---------------------------------------------------------------------------
# Popup image generation + selection
# ---------------------------------------------------------------------------


def maybe_generate_images(
    client: OpenAI, script: str, images_dir: Path, count: int, generate_images: bool
) -> None:
    if not generate_images:
        return
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_seed = (
        "Generate visual prompts for still images that match moments in this story. "
        "Return a JSON array of short prompts only, no explanations."
    )
    prompt_resp = client.responses.create(
        model="gpt-4.1-mini",
        input=f"{prompt_seed}\n\nStory:\n{script}\n\nNeed exactly {count} prompts.",
    )
    raw = prompt_resp.output_text.strip()
    try:
        prompts = json.loads(raw)
    except json.JSONDecodeError:
        prompts = ["School hallway dramatic scene, cinematic, no text"] * count
    if not isinstance(prompts, list):
        prompts = ["School classroom cinematic still, no text"] * count
    prompts = prompts[:count]
    while len(prompts) < count:
        prompts.append("School drama scene, cinematic still image, no text")

    for i, prompt in enumerate(prompts, start=1):
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"{prompt}. Vertical composition, high contrast, no text overlays.",
            size="1024x1536",
        )
        image_b64 = img.data[0].b64_json
        image_bytes = __import__("base64").b64decode(image_b64)
        out_path = images_dir / f"generated_{i}.png"
        out_path.write_bytes(image_bytes)


def choose_popup_images(
    images_dir: Path,
    video_duration: float,
    count: int = 3,
    popup_duration: float = 1.8,
) -> List[PopupImage]:
    candidates = [
        p
        for p in images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    if not candidates:
        return []

    chosen_files = random.sample(candidates, k=min(count, len(candidates)))
    popups = []
    for img in chosen_files:
        start = random.uniform(2.0, max(2.1, video_duration - 4.5))
        end = min(video_duration - 0.2, start + max(0.4, float(popup_duration)))
        width = 700
        x = (1080 - width) // 2
        y = 730
        popups.append(PopupImage(path=img, start_sec=start, end_sec=end, x=x, y=y, width=width))
    return sorted(popups, key=lambda p: p.start_sec)


def maybe_download_story_images(
    story_images_dir: Path,
    story_text: str,
    client: OpenAI | None = None,
    min_count: int = 18,
) -> None:
    return


def fixed_image_times(
    video_duration: float, interval_seconds: float = 2.5, first_offset_seconds: float = 1.2
) -> List[float]:
    times: List[float] = []
    t = max(0.2, float(first_offset_seconds))
    interval = max(0.5, float(interval_seconds))
    while t < max(0.8, video_duration - 0.2):
        times.append(t)
        t += interval
    return times


def choose_story_related_popups(
    story_images_dir: Path,
    story_text: str,
    video_duration: float,
    subtitle_segments: List[dict] | None = None,
    planned_times: List[float] | None = None,
    min_gap: float = 3.0,
    max_gap: float = 6.0,
    popup_duration: float = 1.8,
) -> List[PopupImage]:
    candidates = [
        p
        for p in story_images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    if not candidates:
        return []

    story_keys = text_keywords(story_text)
    scored: List[tuple[int, float, Path]] = []
    for p in candidates:
        name_blob = (
            f"{p.parent.name} {p.stem}"
            .replace("_", " ")
            .replace("-", " ")
        )
        file_keys = text_keywords(name_blob)
        overlap = len(story_keys.intersection(file_keys))
        tie_break = random.random()
        scored.append((overlap, tie_break, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked_images = [item[2] for item in scored]
    if not ranked_images:
        return []

    del subtitle_segments

    popups: List[PopupImage] = []
    beat_times = planned_times[:] if planned_times else []
    if not beat_times:
        t = 0.8
        while t < max(1.0, video_duration - 0.3):
            beat_times.append(t)
            t += random.uniform(min_gap, max_gap)

    for idx, beat in enumerate(beat_times):
        start = max(0.0, float(beat))
        end = min(video_duration - 0.05, start + max(1.0, float(popup_duration)))
        if end <= start:
            continue
        width = 700
        x = (1080 - width) // 2
        y = 860
        img = ranked_images[idx % len(ranked_images)]
        popups.append(
            PopupImage(
                path=img,
                start_sec=start,
                end_sec=end,
                x=x,
                y=y,
                width=width,
                play_sfx=True,
            )
        )
    return sorted(popups, key=lambda p: p.start_sec)


def build_emoji_overlays(subtitle_segments: List[dict], emoji_dir: Path) -> List[PopupImage]:
    overlays: List[PopupImage] = []
    used_emojis: set[str] = set()
    last_emoji: str | None = None
    font_size = 100
    line_step = int(font_size * 1.05)
    top_margin = 520
    caption_center_x = 540
    emoji_size = 105
    emoji_gap = 18

    # Cap emoji beats so the ffmpeg filter graph stays cheap.
    max_beats = 4

    words_per_block = 3
    blocks = split_caption_chunks(subtitle_segments, words_per_chunk=words_per_block)
    blocks = [b for b in blocks if str(b.get("raw_text", "")).strip()]
    if not blocks:
        return overlays

    if len(blocks) > max_beats:
        step = len(blocks) / max_beats
        selected = [blocks[int(i * step)] for i in range(max_beats)]
    else:
        selected = blocks

    for block in selected:
        block_text = str(block.get("raw_text", "")).strip()

        primary = pick_non_repeating_emoji(block_text, used_emojis, last_emoji)
        used_emojis.add(primary)
        secondary = pick_non_repeating_emoji(block_text, used_emojis, primary)
        used_emojis.add(secondary)
        last_emoji = secondary

        display_text = format_caption_multiline(
            random_caps_text(block_text),
            max_words_per_line=1,
            max_chars_per_line=16,
            max_lines=3,
        )
        line_count = max(1, len([ln for ln in display_text.split("\n") if ln.strip()]))

        total_w = emoji_size * 2 + emoji_gap
        x_left = caption_center_x - (total_w // 2)
        y_pos = top_margin + (line_count * line_step) + 12

        start_sec = float(block["start"])
        end_sec = float(block["end"])

        overlays.append(
            PopupImage(
                path=download_twemoji_png(primary, emoji_dir),
                start_sec=start_sec,
                end_sec=end_sec,
                x=x_left,
                y=y_pos,
                width=emoji_size,
                use_fade=False,
                is_emoji=True,
            )
        )
        overlays.append(
            PopupImage(
                path=download_twemoji_png(secondary, emoji_dir),
                start_sec=start_sec,
                end_sec=end_sec,
                x=x_left + emoji_size + emoji_gap,
                y=y_pos,
                width=emoji_size,
                use_fade=False,
                is_emoji=True,
            )
        )
    return overlays
