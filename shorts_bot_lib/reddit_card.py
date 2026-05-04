"""Render a Reddit-style hook popup card as a transparent PNG.

Used for a "reddit post title" style intro overlay.
Run standalone for quick iteration:
  python -m shorts_bot_lib.reddit_card --text "My mom k*lled my daughter" --out output/reddit_card_test.png
"""

from __future__ import annotations

import argparse
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .emoji import download_twemoji_png


@dataclass(frozen=True)
class RedditCardSpec:
    width: int = 980
    height: int = 540
    padding: int = 42
    radius: int = 44
    scale: float = 1.18
    handle: str = "@ChaosStories"
    verified: bool = True
    awards: str = "🐙 🦀 👻 🧿 ❤️ 🚀"
    awards_dir: Path | None = None
    awards_size: int = 56
    awards_spacing: int = 54
    likes_text: str = "99+"
    comments_text: str = "99+"
    emoji_dir: Path = Path("assets/reddit_card/emoji")
    reddit_icon_path: Path = Path("assets/reddit_card/icons/reddit_icon.png")
    verified_badge_path: Path = Path("assets/reddit_card/icons/verified.png")
    comment_icon_path: Path = Path("assets/reddit_card/icons/comment.png")
    heart_icon_path: Path = Path("assets/reddit_card/icons/heart.png")
    share_icon_path: Path = Path("assets/reddit_card/icons/share.png")
    footer_icon_size: int = 40
    heart_icon_size: int = 46
    share_icon_size: int = 54


def _load_font(size: int, *, bold: bool = False):
    # Pillow is an optional dependency at import time; keep local imports.
    from PIL import ImageFont  # type: ignore[import-not-found]

    # Try common fonts; fall back to default bitmap font.
    candidates: list[str] = []
    if bold:
        candidates += [
            "assets/fonts/Inter-Bold.ttf",
            "assets/fonts/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    else:
        candidates += [
            "assets/fonts/Inter-Regular.ttf",
            "assets/fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _wrap_to_width(draw, text: str, font, max_width: int) -> list[str]:
    # Prefer a greedy wrap with measurement.
    words = (text or "").strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        nxt = (" ".join(cur + [w])).strip()
        if cur and _text_width(draw, nxt, font) > max_width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _load_verified_badge_rgba(path: Path):
    from PIL import Image  # type: ignore[import-not-found]

    return Image.open(path).convert("RGBA")


def _load_twemoji_rgba(emoji: str, *, out_dir: Path):
    from PIL import Image  # type: ignore[import-not-found]

    p = download_twemoji_png(emoji, out_dir)
    return Image.open(p).convert("RGBA")

def _load_award_images(awards_dir: Path) -> list[Path]:
    if not awards_dir.exists() or not awards_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for p in sorted(awards_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        # Prevent accidentally picking up the verified badge as an "award".
        if "verified" in p.name.lower() or "check" in p.name.lower():
            continue
        # Exclude pyramid awards.
        if "pyramid" in p.name.lower():
            continue
        if p.name.lower() in {"images.png", "5378429-middle.png"}:
            continue
        files.append(p)
    # Preferred rarity order (rarest -> commonest).
    preferred = [
        "lizard.png",
        "eagle.png",
        "snoo_crown.png",
        "multi_gem.png",
        "mithril_512.png",
        "abovemithril.png",
        "gold_256.png",
        "silver_256.png",
    ]
    by_name = {p.name.lower(): p for p in files}
    ordered = [by_name[n] for n in preferred if n in by_name]
    # Append any extras not in the known set, in stable order.
    used = {p.name.lower() for p in ordered}
    ordered += [p for p in files if p.name.lower() not in used]
    return ordered


def _paste_centered(base_img, overlay_rgba, cx: int, cy: int, size: int) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    ol = overlay_rgba.copy()
    ol = ol.resize((size, size), Image.LANCZOS)
    x0 = int(cx - size // 2)
    y0 = int(cy - size // 2)
    base_img.alpha_composite(ol, dest=(x0, y0))

def _mirror_horiz(rgba_img):
    from PIL import Image  # type: ignore[import-not-found]

    return rgba_img.transpose(Image.FLIP_LEFT_RIGHT)

def _key_out_white_bg(rgba_img, *, threshold: int = 248, softness: int = 12):
    """Make near-white pixels transparent (for PNGs with baked white boxes)."""
    from PIL import Image  # type: ignore[import-not-found]

    img = rgba_img.convert("RGBA")
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            # Distance from white (0 == pure white)
            d = 255 - min(r, g, b)
            if d <= softness:
                # Soft fade: the closer to white, the more transparent.
                t = max(0, min(255, int((d / max(1, softness)) * 255)))
                px[x, y] = (r, g, b, int(a * (t / 255.0)))
            elif r >= threshold and g >= threshold and b >= threshold:
                px[x, y] = (r, g, b, 0)
    return img


def render_reddit_card_png(
    *,
    title: str,
    out_path: Path,
    spec: RedditCardSpec = RedditCardSpec(),
    seed: Optional[int] = None,
) -> Path:
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    if seed is not None:
        random.seed(int(seed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (spec.width, spec.height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Shadow
    shadow = Image.new("RGBA", (spec.width, spec.height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    card_x0 = 0
    card_y0 = 0
    card_x1 = spec.width
    card_y1 = spec.height
    shadow_offset = 10
    sd.rounded_rectangle(
        (card_x0 + shadow_offset, card_y0 + shadow_offset, card_x1 + shadow_offset, card_y1 + shadow_offset),
        radius=spec.radius,
        fill=(0, 0, 0, 70),
    )
    try:
        from PIL import ImageFilter  # type: ignore[import-not-found]
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    except Exception:
        pass
    img.alpha_composite(shadow)

    # Card
    border = (235, 235, 235, 255)
    d.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=spec.radius, fill=(255, 255, 255, 255), outline=border, width=3)

    s = float(spec.scale) if getattr(spec, "scale", 1.0) else 1.0
    pad = int(round(spec.padding * s))
    x = pad
    y = pad

    font_handle = _load_font(int(round(42 * s)), bold=True)
    font_title = _load_font(int(round(66 * s)), bold=True)
    font_meta = _load_font(int(round(36 * s)), bold=False)

    # Avatar circle
    avatar_r = int(round(52 * s))
    avatar_cx = x + avatar_r
    avatar_cy = y + avatar_r
    orange = (255, 69, 0, 255)
    d.ellipse((avatar_cx - avatar_r, avatar_cy - avatar_r, avatar_cx + avatar_r, avatar_cy + avatar_r), fill=orange)
    # Reddit Snoo icon inside the orange circle (transparent PNG).
    try:
        if spec.reddit_icon_path.exists():
            snoo = Image.open(spec.reddit_icon_path).convert("RGBA")
            _paste_centered(img, snoo, avatar_cx, avatar_cy, size=int(round(78 * s)))
    except Exception:
        pass

    # Handle text
    handle_x = x + avatar_r * 2 + int(round(26 * s))
    handle_y = y + int(round(10 * s))
    d.text((handle_x, handle_y), spec.handle, font=font_handle, fill=(20, 20, 20, 255))
    if spec.verified:
        handle_w = _text_width(d, spec.handle, font_handle)
        badge_size = int(round(40 * s))
        badge_cx = handle_x + handle_w + int(round(16 * s)) + badge_size // 2
        badge_cy = handle_y + int(round(28 * s))
        try:
            if spec.verified_badge_path.exists():
                vb = _load_verified_badge_rgba(spec.verified_badge_path)
                _paste_centered(img, vb, badge_cx, badge_cy, size=badge_size)
        except Exception:
            pass

    # Awards row: left-aligned with the handle (@...), compact spacing.
    awards_y = handle_y + int(round(60 * s))
    right_pad = pad
    available_w = max(0, (spec.width - right_pad) - handle_x)
    size = int(round(spec.awards_size * s))
    spacing = int(round(spec.awards_spacing * s))
    used_awards = False
    if spec.awards_dir is not None:
        try:
            from PIL import Image  # type: ignore[import-not-found]

            award_paths = _load_award_images(spec.awards_dir)[:12]
            if award_paths:
                cy = awards_y + size // 2
                cx = handle_x + size // 2
                for p in award_paths:
                    if (cx + size // 2) - handle_x > available_w:
                        break
                    a = Image.open(p).convert("RGBA")
                    a = _key_out_white_bg(a)
                    _paste_centered(img, a, cx, cy, size=size)
                    cx += spacing
                used_awards = True
        except Exception:
            used_awards = False
    if not used_awards:
        awards_text = (spec.awards or "").strip()
        if awards_text:
            toks = awards_text.split()
            cy = awards_y + size // 2
            cx = handle_x + size // 2
            for tok in toks:
                try:
                    eimg = _load_twemoji_rgba(tok, out_dir=spec.emoji_dir)
                    if (cx + size // 2) - handle_x > available_w:
                        break
                    _paste_centered(img, eimg, cx, cy, size=size)
                except Exception:
                    # Fallback to monochrome text if emoji download fails.
                    if (cx + size // 2) - handle_x > available_w:
                        break
                    d.text((cx, awards_y), tok, font=font_meta, fill=(20, 20, 20, 255), anchor="ma")
                cx += spacing

    # Title (wrapped). Shrink font progressively so it fits the card without
    # ellipsizing or overflowing into the footer.
    title_y = y + int(round(148 * s))
    max_text_w = spec.width - pad * 2
    footer_y_for_title = spec.height - pad - int(round(60 * s))
    available_h = max(40, footer_y_for_title - title_y - int(round(16 * s)))

    base_size = int(round(66 * s))
    min_size = max(28, int(round(34 * s)))
    font_used = font_title
    lines = _wrap_to_width(d, title.strip(), font_used, max_text_w)

    cur_size = base_size
    while cur_size > min_size:
        font_used = _load_font(cur_size, bold=True)
        lines = _wrap_to_width(d, title.strip(), font_used, max_text_w)
        line_h = int(font_used.size * 1.12)
        total_h = line_h * len(lines)
        too_wide = any(_text_width(d, ln, font_used) > max_text_w for ln in lines)
        if total_h <= available_h and not too_wide:
            break
        cur_size -= 4
    else:
        font_used = _load_font(min_size, bold=True)
        lines = _wrap_to_width(d, title.strip(), font_used, max_text_w)
        # Last-resort: trim to as many lines as fit.
        max_lines = max(1, available_h // int(font_used.size * 1.12))
        if len(lines) > max_lines:
            lines = lines[:max_lines]

    ty = title_y
    for line in lines:
        d.text((x, ty), line, font=font_used, fill=(10, 10, 10, 255))
        ty += int(font_used.size * 1.12)

    # Footer icons + counts (Twemoji icons look closer than line art).
    footer_y = spec.height - pad - int(round(60 * s))
    icon_color = (140, 140, 140, 255)
    icon_cy = footer_y + int(round(26 * s))
    icon_sz = int(round(spec.footer_icon_size * s))
    heart_sz = int(round(spec.heart_icon_size * s))
    try:
        if spec.heart_icon_path.exists():
            heart = Image.open(spec.heart_icon_path).convert("RGBA")
            heart = _key_out_white_bg(heart)
            _paste_centered(img, heart, x + int(round(20 * s)), icon_cy, size=heart_sz)
        else:
            heart = _load_twemoji_rgba("❤️", out_dir=spec.emoji_dir)
            _paste_centered(img, heart, x + int(round(20 * s)), icon_cy, size=heart_sz)
    except Exception:
        pass
    d.text((x + int(round(20 * s)) + icon_sz // 2 + int(round(18 * s)), footer_y), spec.likes_text, font=font_meta, fill=icon_color)
    try:
        if spec.comment_icon_path.exists():
            chat = Image.open(spec.comment_icon_path).convert("RGBA")
            chat = _mirror_horiz(chat)
            _paste_centered(img, chat, x + int(round(210 * s)), icon_cy, size=icon_sz)
        else:
            chat = _load_twemoji_rgba("💬", out_dir=spec.emoji_dir)
            _paste_centered(img, chat, x + int(round(210 * s)), icon_cy, size=icon_sz)
    except Exception:
        pass
    d.text((x + int(round(210 * s)) + icon_sz // 2 + int(round(18 * s)), footer_y), spec.comments_text, font=font_meta, fill=icon_color)

    # Share icon on the bottom-right (no count).
    try:
        if spec.share_icon_path.exists():
            share = Image.open(spec.share_icon_path).convert("RGBA")
            share = _key_out_white_bg(share)
            share_sz = int(round(spec.share_icon_size * s))
            share_label = "share"
            gap = int(round(12 * s))
            label_w = _text_width(d, share_label, font_meta)
            # Place so icon + gap + label fit within the right padding.
            share_right = spec.width - pad
            label_x = share_right - label_w
            share_cx = label_x - gap - (share_sz // 2)
            _paste_centered(img, share, share_cx, icon_cy, size=share_sz)
            d.text((label_x, footer_y), share_label, font=font_meta, fill=icon_color)
    except Exception:
        pass

    img.save(out_path)
    return out_path


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a Reddit-style hook popup card PNG.")
    p.add_argument("--text", required=True, help="Main title text to render on the card.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--handle", default="@ChaosStories")
    p.add_argument("--no-verified", action="store_true")
    p.add_argument("--awards-dir", default="", help="Optional directory of award PNGs to show.")
    p.add_argument("--width", type=int, default=980)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    spec = RedditCardSpec(
        width=int(args.width),
        height=int(args.height),
        handle=str(args.handle),
        verified=not bool(args.no_verified),
        awards_dir=Path(args.awards_dir) if str(args.awards_dir).strip() else None,
    )
    out_path = render_reddit_card_png(
        title=str(args.text),
        out_path=Path(str(args.out)),
        spec=spec,
        seed=args.seed,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

