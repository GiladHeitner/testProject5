"""Lightweight image quality judge using a multimodal LLM.

Used to automatically reject low-quality / off-topic / watermarked images
and trigger regeneration.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


@dataclass(frozen=True)
class ImageJudgeResult:
    ok: bool
    score: int
    reason: str


def _as_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".") or "png"
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/png")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def judge_image(
    *,
    client: OpenAI | None,
    image_path: Path,
    expected: str,
    min_score: int | None = None,
) -> ImageJudgeResult:
    """Return whether the image is usable for the given expectation.

    If no client is provided or IMAGE_JUDGE=0, the image is treated as OK.
    """
    if os.environ.get("IMAGE_JUDGE", "1").strip() in {"0", "false", "False"}:
        return ImageJudgeResult(ok=True, score=10, reason="judge disabled")
    if client is None:
        return ImageJudgeResult(ok=True, score=10, reason="no judge client")
    if not image_path.exists():
        return ImageJudgeResult(ok=False, score=0, reason="missing image file")

    # Cheap sanity checks before calling the model.
    try:
        size = image_path.stat().st_size
    except Exception:
        size = 0
    if size < 15_000:
        return ImageJudgeResult(ok=False, score=1, reason="file too small / likely invalid")

    threshold = (
        int(os.environ.get("IMAGE_JUDGE_MIN_SCORE", "7"))
        if min_score is None
        else int(min_score)
    )

    system = (
        "You are a strict image quality reviewer for a YouTube Shorts pipeline. "
        "You must reject images that are blurry, low-res, irrelevant, have "
        "watermarks/overlaid text, or look like an obvious AI glitch. "
        "Also reject images that are not vertical-friendly (subject too tiny)."
    )
    user_text = (
        "Return ONLY valid JSON:\n"
        '{ "score": 0-10, "ok": true/false, "reason": "<short>" }\n\n'
        f"Expectation: {expected}\n"
        f"Minimum acceptable score: {threshold}\n"
        "If you see any watermark/logo/text overlay, set ok=false.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("IMAGE_JUDGE_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": _as_data_url(image_path)}},
                    ],
                },
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        # If the judge fails (vision unavailable), don't block the pipeline.
        return ImageJudgeResult(ok=True, score=10, reason=f"judge error ignored: {exc}")

    try:
        import json

        data = json.loads(raw)
        score = int(data.get("score") or 0)
        ok = bool(data.get("ok"))
        reason = str(data.get("reason") or "").strip() or "no reason"
    except Exception:
        # If it didn't follow spec, treat as pass to avoid breaking runs.
        return ImageJudgeResult(ok=True, score=10, reason="judge parse failure ignored")

    if score < threshold:
        ok = False
    return ImageJudgeResult(ok=ok, score=score, reason=reason)

