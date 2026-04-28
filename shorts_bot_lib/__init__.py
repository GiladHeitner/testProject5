"""Modular implementation of the Shorts Bot pipeline.

The original `shorts_bot.py` was a single ~2,600-line file. The same
functions now live in dedicated modules grouped by responsibility:

- `types`        – dataclasses (PopupImage, Word)
- `runner`       – subprocess helpers, progress bar, ffmpeg meta
- `text`         – text/format helpers (strip markup, captions, ASS escape)
- `emoji`        – emoji picker + Apple twemoji PNG downloader
- `audio`        – speed ramps, loudnorm, normalized sounds folder
- `subtitles`    – ASS / SRT writers and caption chunking
- `transcribe`   – Whisper / faster-whisper transcription + alignment
- `script_ai`    – LLM script + metadata + image-prompt generation
- `voiceover`    – TTS engines (Adam cloner + OpenAI TTS)
- `images`       – hook image (Gemini → OpenAI), Unsplash, popup chooser, emoji overlays
- `video`        – ffmpeg filter graph + final render
- `youtube_api`  – YouTube OAuth, upload, pinned comment

`shorts_bot.py` keeps the CLI entry point; it imports from this package
and orchestrates the pipeline.
"""
