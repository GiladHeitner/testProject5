"""Tiered online image fetch: Pixabay (graphics) → Met Museum → Wikimedia.

No persistent cache between runs; URLs are resolved per request and bytes are
written only when downloading for the video render pipeline.

Set IMAGE_PIPELINE_DISABLED=1 to skip this module entirely.
Pixabay L1 requires PIXABAY_API_KEY (L2 Met and L3 Wikimedia need no key).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import requests

KeywordKind = Literal["graphic", "historical", "place", "generic"]
SourceKind = Literal["pixabay", "met", "wikimedia"]

REQUEST_TIMEOUT = 30
PIXABAY_API = "https://pixabay.com/api/"
MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MET_ISLAMIC_ART_DEPT = 14
DEFAULT_CONTACT = "videobots@localhost"

_RASTER_URL_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif")
_NON_RASTER_URL_SUFFIXES = (".svg", ".pdf", ".djvu", ".tif", ".tiff")


def _is_raster_image_url(url: str) -> bool:
    path = (url or "").split("?", 1)[0].lower()
    if any(path.endswith(ext) for ext in _NON_RASTER_URL_SUFFIXES):
        return False
    if any(path.endswith(ext) for ext in _RASTER_URL_SUFFIXES):
        return True
    # Wikimedia thumb URLs often omit extensions; allow those.
    return "/thumb/" in path or "upload.wikimedia.org" in path


def is_raster_image_file(path: Path) -> bool:
    try:
        return _is_raster_image_bytes(path.read_bytes())
    except OSError:
        return False


def _is_raster_image_bytes(data: bytes) -> bool:
    if not data or len(data) < 12:
        return False
    head = data[:32].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<svg") or b"<svg" in head[:256].lower():
        return False
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def _user_agent() -> str:
    custom = os.environ.get("HTTP_USER_AGENT", "").strip()
    if custom:
        return custom
    contact = os.environ.get("WIKIMEDIA_USER_AGENT_CONTACT", DEFAULT_CONTACT).strip() or DEFAULT_CONTACT
    return f"VideoBots/1.0 (YouTube Shorts image pipeline; {contact}) Python/requests"


def _request_headers(url: str = "") -> dict[str, str]:
    headers = {"User-Agent": _user_agent()}
    host = url.lower()
    if "wikimedia.org" in host or "wikipedia.org" in host:
        headers["Referer"] = "https://en.wikipedia.org/"
        headers["Accept"] = "image/webp,image/apng,image/*,*/*;q=0.8"
    return headers

_GRAPHIC_WORDS = frozenset(
    {
        "border",
        "icon",
        "graphic",
        "graphics",
        "pattern",
        "lantern",
        "mosaic",
        "ornament",
        "calligraphy",
        "vector",
        "illustration",
        "emblem",
        "motif",
        "tile",
        "arabesque",
    }
)
_HISTORICAL_WORDS = frozenset(
    {
        "empire",
        "manuscript",
        "caliph",
        "ancient",
        "art",
        "dynasty",
        "sultan",
        "ottoman",
        "crusade",
        "historical",
        "history",
        "museum",
        "artifact",
        "relic",
        "scroll",
        "civilization",
        "medieval",
        "umayyad",
        "abbasid",
        "mamluk",
    }
)
_PLACE_PHRASES = frozenset(
    {
        "mecca",
        "makkah",
        "medina",
        "madinah",
        "alhambra",
        "blue mosque",
        "hagia sophia",
        "istanbul",
        "cordoba",
        "cordóba",
        "dome of the rock",
        "jerusalem",
        "kaaba",
        "kaabah",
        "petra",
        "cairo",
        "damascus",
        "baghdad",
        "samarkand",
        "fez",
        "marrakech",
        "granada",
        "masjid",
        "mosque",
    }
)

_TIER_ORDER: dict[KeywordKind, list[SourceKind]] = {
    "graphic": ["pixabay", "met", "wikimedia"],
    "historical": ["met", "wikimedia", "pixabay"],
    "place": ["wikimedia", "met", "pixabay"],
    "generic": ["met", "wikimedia", "pixabay"],
}


@dataclass(frozen=True)
class ImageFetchResult:
    url: str
    source: SourceKind
    content_type: str | None = None


def _pipeline_disabled() -> bool:
    return os.environ.get("IMAGE_PIPELINE_DISABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _tokenize(query: str) -> set[str]:
    return {t for t in _normalize_query(query).split() if t}


def classify_keyword(query: str) -> KeywordKind:
    norm = _normalize_query(query)
    tokens = _tokenize(query)
    if tokens & _GRAPHIC_WORDS or any(w in norm for w in _GRAPHIC_WORDS):
        return "graphic"
    if tokens & _HISTORICAL_WORDS or any(w in norm for w in _HISTORICAL_WORDS):
        return "historical"
    if any(phrase in norm for phrase in _PLACE_PHRASES):
        return "place"
    if tokens & {"mosque", "minaret", "minarets", "dome", "palace", "fortress"}:
        return "place"
    return "generic"


def _with_retries(fn, *, attempts: int = 3, base_delay: float = 1.0, label: str = "request"):
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            print(f"[{label}] retry {attempt} in {delay:.1f}s ({exc})", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"{label} failed") from last_exc


def _fetch_pixabay(query: str, api_key: str) -> ImageFetchResult | None:
    for image_type in ("vector", "illustration"):
        try:
            response = requests.get(
                PIXABAY_API,
                params={
                    "key": api_key,
                    "q": query,
                    "image_type": image_type,
                    "per_page": 5,
                    "safesearch": "true",
                    "orientation": "vertical",
                },
                timeout=REQUEST_TIMEOUT,
                headers=_request_headers(PIXABAY_API),
            )
            response.raise_for_status()
            hits = response.json().get("hits") or []
        except Exception as exc:
            print(f"[pixabay] search failed ({image_type}): {exc}")
            continue
        for hit in hits:
            url = hit.get("largeImageURL") or hit.get("webformatURL")
            if url:
                return ImageFetchResult(url=str(url), source="pixabay")
    return None


def _met_search_ids(query: str, *, department_id: int | None) -> list[int]:
    params: dict[str, str | int | bool] = {"q": query, "hasImages": True}
    if department_id is not None:
        params["departmentId"] = department_id
    try:
        response = requests.get(
            MET_SEARCH, params=params, timeout=REQUEST_TIMEOUT, headers=_request_headers(MET_SEARCH)
        )
        response.raise_for_status()
        ids = response.json().get("objectIDs") or []
        return [int(i) for i in ids if i is not None]
    except Exception as exc:
        print(f"[met] search failed: {exc}")
        return []


def _met_object_url(object_id: int) -> ImageFetchResult | None:
    try:
        response = requests.get(
            f"{MET_OBJECT}/{object_id}",
            timeout=REQUEST_TIMEOUT,
            headers=_request_headers(MET_OBJECT),
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"[met] object {object_id}: {exc}")
        return None
    if not data.get("isPublicDomain"):
        return None
    url = (data.get("primaryImageSmall") or data.get("primaryImage") or "").strip()
    if not url:
        return None
    return ImageFetchResult(url=url, source="met")


def _fetch_met(query: str, *, islamic_dept: bool) -> ImageFetchResult | None:
    id_lists: list[list[int]] = []
    if islamic_dept:
        ids = _met_search_ids(query, department_id=MET_ISLAMIC_ART_DEPT)
        if ids:
            id_lists.append(ids)
    ids = _met_search_ids(query, department_id=None)
    if ids:
        id_lists.append(ids)
    seen: set[int] = set()
    candidates: list[int] = []
    for id_list in id_lists:
        for oid in id_list:
            if oid not in seen:
                seen.add(oid)
                candidates.append(oid)
    if not candidates:
        return None
    random.shuffle(candidates)
    for object_id in candidates[:8]:
        result = _met_object_url(object_id)
        if result is not None:
            return result
    return None


def _wiki_page_image(titles: list[str]) -> ImageFetchResult | None:
    if not titles:
        return None
    try:
        response = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "prop": "pageimages",
                "piprop": "original",
                "titles": "|".join(titles[:3]),
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
            headers=_request_headers(WIKI_API),
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages") or {}
    except Exception as exc:
        print(f"[wikimedia] pageimages failed: {exc}")
        return None
    for page in pages.values():
        if page.get("missing") or page.get("invalid"):
            continue
        original = (page.get("original") or {})
        url = (original.get("source") or "").strip()
        if url and _is_raster_image_url(url):
            return ImageFetchResult(url=url, source="wikimedia")
    return None


def _wiki_opensearch_titles(query: str) -> list[str]:
    try:
        response = requests.get(
            WIKI_API,
            params={
                "action": "opensearch",
                "search": query,
                "limit": 3,
                "namespace": 0,
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
            headers=_request_headers(WIKI_API),
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) >= 2:
            return [str(t) for t in data[1] if t]
    except Exception as exc:
        print(f"[wikimedia] opensearch failed: {exc}")
    return []


def _wiki_generator_search(query: str) -> ImageFetchResult | None:
    try:
        response = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "pageimages",
                "piprop": "original",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
            headers=_request_headers(WIKI_API),
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages") or {}
    except Exception as exc:
        print(f"[wikimedia] generator search failed: {exc}")
        return None
    for page in pages.values():
        original = (page.get("original") or {})
        url = (original.get("source") or "").strip()
        if url and _is_raster_image_url(url):
            return ImageFetchResult(url=url, source="wikimedia")
    return None


def _fetch_wikimedia(query: str) -> ImageFetchResult | None:
    titles = _wiki_opensearch_titles(query)
    result = _wiki_page_image(titles)
    if result is not None:
        return result
    return _wiki_generator_search(query)


def _fetch_from_source(source: SourceKind, query: str) -> ImageFetchResult | None:
    if source == "pixabay":
        api_key = os.environ.get("PIXABAY_API_KEY", "").strip()
        if not api_key:
            return None
        return _fetch_pixabay(query, api_key)
    if source == "met":
        kind = classify_keyword(query)
        islamic = kind in ("historical", "graphic", "generic")
        return _fetch_met(query, islamic_dept=islamic)
    if source == "wikimedia":
        return _fetch_wikimedia(query)
    return None


def fetch_image_url(query: str, *, prefer: str | None = None) -> ImageFetchResult | None:
    """Resolve an image URL using tiered providers (no disk write)."""
    if _pipeline_disabled():
        return None
    q = (query or "").strip()
    if not q:
        return None
    kind = classify_keyword(q)
    order: list[SourceKind] = list(_TIER_ORDER[kind])
    if prefer in ("pixabay", "met", "wikimedia"):
        pref = prefer  # type: ignore[assignment]
        order = [pref] + [s for s in order if s != pref]
    for source in order:
        try:
            result = _fetch_from_source(source, q)
        except Exception as exc:
            print(f"[image_pipeline] {source} error: {exc}")
            result = None
        if result is not None:
            print(f"[image_pipeline] {source} -> {result.url[:80]}...")
            return result
    return None


def download_image(result: ImageFetchResult, dst: Path, timeout: int = REQUEST_TIMEOUT) -> bool:
    try:
        response = requests.get(
            result.url,
            timeout=timeout,
            headers=_request_headers(result.url),
            allow_redirects=True,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"[image_pipeline] download failed: {exc}")
        return False
    content_type = (response.headers.get("Content-Type") or "").lower()
    if not content_type.startswith("image/"):
        print(f"[image_pipeline] download rejected non-image content-type: {content_type or 'unknown'}")
        return False
    if "svg" in content_type or not _is_raster_image_url(result.url):
        print(f"[image_pipeline] download rejected non-raster image: {result.url[:80]}...")
        return False
    if not _is_raster_image_bytes(response.content):
        print(f"[image_pipeline] download rejected invalid raster bytes from {result.url[:80]}...")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(response.content)
    return True


def fetch_image_to_path(query: str, dst: Path) -> tuple[Path | None, str | None]:
    """Fetch URL via tiered APIs and write to dst. Returns (path, source)."""
    result = fetch_image_url(query)
    if result is None:
        return None, None
    if download_image(result, dst):
        return dst, result.source
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test tiered image fetch.")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--out", default="", help="Download to this path (optional)")
    parser.add_argument("--url-only", action="store_true", help="Print URL only")
    args = parser.parse_args(argv)
    result = fetch_image_url(args.query)
    if result is None:
        print("No image found.")
        return 1
    print(f"source={result.source} url={result.url}")
    if args.url_only:
        return 0
    if args.out:
        if download_image(result, Path(args.out)):
            print(f"Wrote {args.out}")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
