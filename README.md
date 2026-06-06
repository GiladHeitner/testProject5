# YouTube Shorts Story Bot (MVP)

This bot generates Muslim/Arab teen storytime Shorts from Reddit or `topics.txt`, adds subtitles, popup images, and can upload to YouTube.

## What it does

- Generates a script with AI
- Generates voiceover audio with ElevenLabs
- Auto-creates subtitles (SRT) with Whisper
- Picks random Minecraft or Roblox gameplay from `assets/gameplay`
- Starts from a random timestamp each run
- Adds random popup images from `assets/popups`
- Exports vertical short video to `output/short.mp4`
- Optionally uploads to YouTube

## Requirements

- Python 3.10+
- `ffmpeg` + `ffprobe` installed and available in PATH

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create env file:

   ```bash
   cp .env.example .env
   ```

   Required keys:
   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - Optional: `ELEVENLABS_VOICE_ID`
   - Optional popup images: `PEXELS_API_KEY`, `UNSPLASH_ACCESS_KEY`, `PIXABAY_API_KEY` (vectors/graphics)
   - Set `IMAGE_PIPELINE_DISABLED=1` to skip Pixabay/Met/Wikimedia and use stock APIs only

3. Put gameplay clips in:

   - `assets/gameplay/`

4. (Optional) Put engagement images in:

   - `assets/popups/`

5. For YouTube upload, download OAuth client secret JSON from Google Cloud and place it at `client_secret.json` (or set path in `.env`).

## Run

Generate short only:

```bash
python shorts_bot.py
```

High-pitch + fast narration (young audience style):

```bash
python shorts_bot.py --speech-speed 1.45 --pitch-factor 1.35
```

Generate + auto-generate popup images:

```bash
python shorts_bot.py --generate-images
```

Generate and upload to YouTube:

```bash
python shorts_bot.py --upload --privacy private
```

## Notes

- First YouTube upload run opens browser OAuth flow.
- Output files:
  - `output/script.txt`
  - `output/narration.mp3`
  - `output/subtitles.srt`
  - `output/short.mp4`
