"""YouTube OAuth, video upload, custom thumbnail, and pinned-comment helpers."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from openai import OpenAI

from .script_ai import generate_pinned_comment

DEFAULT_UPLOAD_REGISTRY = Path(".github/upload_registry.jsonl")
SCRIPT_EXCERPT_CHARS = 800


_PINNED_COMMENT_FALLBACKS = [
    "Which part hit different for you? Drop it below \U0001F447",
    "Who else has been through this?? \U0001F62D",
    "Tell me I'm not the only one \U0001F480",
    "Tag someone who needs to see this \U0001F440",
    "Part 2 if this gets 500 likes? \U0001F914",
    "What would YOU have done in this situation? \U0001F447",
    "This actually happened btw \U0001F62D",
]


def get_youtube_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
    ]
    client_secret_file = Path(os.environ.get("YOUTUBE_CLIENT_SECRET_FILE", "client_secret.json"))
    token_file = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "youtube_token.json"))

    if not client_secret_file.exists():
        raise RuntimeError(
            f"Missing YouTube OAuth client file: {client_secret_file}. "
            "Download a Desktop app OAuth client JSON from Google Cloud and save it there."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes=scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.exceptions import RefreshError
            try:
                creds.refresh(Request())
            except RefreshError:
                print("YouTube token expired or revoked. Re-authenticating...")
                token_file.unlink(missing_ok=True)
                flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
                creds = flow.run_local_server(port=0, open_browser=True)
        else:
            print("Opening browser to link your YouTube channel...")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
            creds = flow.run_local_server(port=0, open_browser=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


def upload_to_youtube(
    video_file: Path,
    title: str,
    description: str,
    tags: List[str],
    privacy: str,
    thumbnail_file: Path | None = None,
) -> str:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    creds = get_youtube_credentials()

    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22",
        },
        "status": {"privacyStatus": privacy},
    }
    media = MediaFileUpload(str(video_file), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()

    video_id = response["id"]

    if thumbnail_file is not None and thumbnail_file.exists():
        try:
            mime = "image/png" if thumbnail_file.suffix.lower() == ".png" else "image/jpeg"
            thumb_media = MediaFileUpload(str(thumbnail_file), mimetype=mime, resumable=False)
            youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            print(f"Custom thumbnail set from {thumbnail_file.name}")
        except HttpError as exc:
            print(f"Custom thumbnail upload skipped: {exc}")
        except Exception as exc:
            print(f"Custom thumbnail upload error: {exc}")

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            status_response = youtube.videos().list(
                part="status,processingDetails",
                id=video_id,
            ).execute()
        except HttpError as exc:
            if getattr(exc, "resp", None) is not None and exc.resp.status == 403:
                print("Upload completed, but processing status check was skipped due to YouTube scope limits.")
                break
            raise
        items = status_response.get("items", [])
        if not items:
            break
        item = items[0]
        processing_status = (
            item.get("processingDetails", {}).get("processingStatus", "").lower()
        )
        upload_status = item.get("status", {}).get("uploadStatus", "").lower()
        if upload_status == "processed" or processing_status == "succeeded":
            break
        if processing_status == "failed":
            raise RuntimeError(f"YouTube processing failed for uploaded video {video_id}.")
        time.sleep(5)
    return f"https://www.youtube.com/watch?v={video_id}"


def post_pinned_comment(
    youtube,
    video_id: str,
    script: str,
    *,
    client: OpenAI | None = None,
    lead_line: str = "",
) -> None:
    comment_text: str | None = None
    if client is not None and script.strip():
        try:
            comment_text = generate_pinned_comment(client, script)
        except Exception as exc:
            print(f"Pinned comment LLM failed, using fallback: {exc}")
    if not comment_text:
        comment_text = random.choice(_PINNED_COMMENT_FALLBACKS)
    # Series Part 2: lead with a link back to Part 1 so viewers binge both.
    if lead_line.strip():
        comment_text = f"{lead_line.strip()}\n\n{comment_text}"
    try:
        response = youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {"textOriginal": comment_text}
                    },
                }
            },
        ).execute()
        comment_id = response["snippet"]["topLevelComment"]["id"]
        youtube.comments().setModerationStatus(
            id=comment_id,
            moderationStatus="published",
        ).execute()
        print(f"Pinned comment: {comment_text}")
    except Exception as exc:
        print(f"Could not post pinned comment: {exc}")


@dataclass(frozen=True)
class VideoComment:
    comment_id: str
    video_id: str
    author_channel_id: str
    author_display_name: str
    text: str
    published_at: datetime


def append_upload_registry(
    video_id: str,
    *,
    title: str,
    script: str,
    title_variant: str = "",
    registry_path: Path | None = None,
) -> None:
    """Append one upload record for the comment replier (JSONL).

    title_variant records which A/B title formula produced this upload
    ("curiosity" or "legacy") so performance can be compared later.
    """
    path = registry_path or Path(
        os.environ.get("UPLOAD_REGISTRY_FILE", str(DEFAULT_UPLOAD_REGISTRY))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    excerpt = (script or "").strip().replace("\n", " ")[:SCRIPT_EXCERPT_CHARS]
    row = {
        "video_id": video_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "title": (title or "").strip(),
        "title_variant": (title_variant or "").strip(),
        "script": excerpt,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_upload_registry(
    registry_path: Path | None = None,
    *,
    max_age_hours: float = 168.0,
) -> list[dict[str, Any]]:
    path = registry_path or Path(
        os.environ.get("UPLOAD_REGISTRY_FILE", str(DEFAULT_UPLOAD_REGISTRY))
    )
    if not path.is_file():
        return []
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = str(row.get("video_id") or "").strip()
        if not vid or vid in seen_videos:
            continue
        uploaded_raw = str(row.get("uploaded_at") or "")
        try:
            uploaded_at = datetime.fromisoformat(uploaded_raw.replace("Z", "+00:00"))
        except ValueError:
            uploaded_at = now
        age_h = (now - uploaded_at).total_seconds() / 3600.0
        if age_h > max_age_hours:
            continue
        seen_videos.add(vid)
        out.append(row)
    return out


def build_youtube_client():
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=get_youtube_credentials())


def get_own_channel_id(youtube) -> str:
    response = youtube.channels().list(part="id", mine=True).execute()
    items = response.get("items") or []
    if not items:
        raise RuntimeError("Could not resolve authenticated YouTube channel id.")
    return str(items[0]["id"])


def _parse_youtube_datetime(value: str) -> datetime:
    cleaned = (value or "").strip().replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def list_video_comments(
    youtube,
    video_id: str,
    *,
    max_results: int = 100,
) -> list[VideoComment]:
    """Top-level comments on a video (newest pages up to max_results)."""
    comments: list[VideoComment] = []
    page_token: str | None = None
    while len(comments) < max_results:
        batch = min(100, max_results - len(comments))
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=batch,
                order="time",
                pageToken=page_token,
                textFormat="plainText",
            )
            .execute()
        )
        for item in response.get("items") or []:
            snippet = item.get("snippet") or {}
            top = snippet.get("topLevelComment") or {}
            top_snip = top.get("snippet") or {}
            comment_id = str(top.get("id") or "").strip()
            if not comment_id:
                continue
            comments.append(
                VideoComment(
                    comment_id=comment_id,
                    video_id=video_id,
                    author_channel_id=str(top_snip.get("authorChannelId") or ""),
                    author_display_name=str(top_snip.get("authorDisplayName") or ""),
                    text=str(top_snip.get("textDisplay") or top_snip.get("textOriginal") or "").strip(),
                    published_at=_parse_youtube_datetime(str(top_snip.get("publishedAt") or "")),
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return comments


def post_comment_reply(youtube, parent_comment_id: str, text: str) -> str:
    """Reply to a top-level comment; returns new reply comment id."""
    response = (
        youtube.comments()
        .insert(
            part="snippet",
            body={
                "snippet": {
                    "parentId": parent_comment_id,
                    "textOriginal": text[:240],
                }
            },
        )
        .execute()
    )
    return str(response.get("id") or "")
