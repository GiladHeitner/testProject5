"""YouTube OAuth, video upload, custom thumbnail, and pinned-comment helpers."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import List

from openai import OpenAI

from .script_ai import generate_pinned_comment


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
) -> None:
    comment_text: str | None = None
    if client is not None and script.strip():
        try:
            comment_text = generate_pinned_comment(client, script)
        except Exception as exc:
            print(f"Pinned comment LLM failed, using fallback: {exc}")
    if not comment_text:
        comment_text = random.choice(_PINNED_COMMENT_FALLBACKS)
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
