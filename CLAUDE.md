# CLAUDE.md — VideoBots / @HalaalRants

Project context and working state for Claude sessions. **Keep the "Current status" section updated as work progresses.**

## What this repo is
Automated YouTube **Shorts** factory for the channel **@HalaalRants** (faceless AI storytime; persona "Omar", a 17-year-old Muslim Arab teen). Pipeline: pick a Reddit/topic source → adapt to a first-person rant script (`script_ai.py`) → synthetic voiceover (Qwen TTS / ElevenLabs Adam clone) → render over gameplay with captions/popups (`video.py`, `keyword_popups.py`) → upload + pinned comment (`youtube_api.py`).

- Entry point: `shorts_bot.py` (CLI) / `run.sh`. UI: `ui.py`. Library: `shorts_bot_lib/`.
- CI: `.github/workflows/upload.yml` generates + uploads. **Triggered by an external cron website** via `repository_dispatch`/`workflow_dispatch` — the GitHub `schedule:` crons were REMOVED 2026-06-28 (they double-fired on top of the external cron). Do not re-add a `schedule:` block.
- State files in `.github/`: `used_reddit.txt`, `upload_registry.jsonl` (now includes `title_variant`), etc.

## The core problem (diagnosed 2026-06-28 from live Studio analytics)
The channel does NOT have a reach problem — it has a **loyalty/conversion problem**.
- 295K views/28d, 60.6% stay-to-watch, 85–98% retention on hits → top-of-funnel works.
- BUT **<0.1% regular viewers**, **~0.12% sub conversion**, 516 subs on 233 videos.
- Competitor **Studonomy**: 26.4K subs on 63 videos (~190× our per-video sub efficiency) using **curiosity-gap, third-person, no-emoji titles** that withhold the outcome.
- Root causes: (1) no reason to return (faceless, identical voice, disposable one-offs); (2) titles spoil the story + niche-gate the audience.

Full diagnosis + roadmap: **`CHANNEL_IMPROVEMENT_PLAN.md`** (read it first).

## Current status (update this section regularly)
_Last updated: 2026-06-28_

**Shipped & committed:**
- ✅ Removed GitHub `schedule:` crons from `upload.yml` (committed + pushed to main: `a4b9249..ecab486`).

**Shipped, UNCOMMITTED in working tree (awaiting user go-ahead to commit):**
- ✅ **P0 — Title rewrite:** new curiosity-gap `TITLE_PROMPT` in `script_ai.py` (no emoji/hashtags, withholds payoff, 30–50 chars front-loaded, universal-conflict framing, Studonomy-style few-shots). Removed forced `#Shorts` title append + cleaned fallback titles in `shorts_bot.py`.
- ✅ **A/B harness:** `pick_title_variant()` sends ~20% of uploads through legacy `TITLE_PROMPT_LEGACY`; env `TITLE_AB_LEGACY_PCT` (default 20). Variant recorded as `title_variant` in `upload_registry.jsonl`.
- ✅ **P1 (partial) — CTA text rewrite:** script CTA "subscribe before I get banned" → "Subscribe so tomorrow's story finds you" (`script_ai.py`); `_SUBSCRIBE_PHRASES` updated in `subscribe_cta.py` so the GIF overlay still times correctly.
- ✅ **P2 — Topic-diversity cooldown:** `classify_topic_theme()` + `_apply_theme_cooldown()` in `reddit_topics.py` (both reddit + topics.txt paths); history in `.github/used_themes.txt` (added to `upload.yml` commit step); env `TOPIC_THEME_COOLDOWN` (default 4). Tested — excludes recently-used themes. Kills the 10× "arranged marriage" run.
- ✅ **Cadence — confirmed already at ~1/day**, no change needed (see §3 status of plan). cron-job.org "YoutubeUploader" fires 1×/day @ 8 AM; the 2 GitHub schedules that stacked on top were already removed.

**Not yet done / next up (need decisions or local verification):**
- ❌ **CTA timing — on-screen TEXT CTA for muted viewers: NOT NEEDED (decided 2026-06-28).** Subtitles transcribe the FULL narration (Whisper, `transcribe.py`) with no subscribe-line stripping, so the closing "Subscribe so tomorrow's story finds you" line is already captioned + burned in. Muted viewers see the CTA as text already. The research's muted-CTA concern applies to audio-only-CTA channels, not us. (The ~20s mid-video verbal CTA is also redundant for 18–28s shorts.)
- ⏳ **P1 — series / recurring-character mechanic** (biggest sub-conversion lever per §5.5): real feature needing design (how to generate + link Part 1/Part 2, consistent "Omar" branding). Needs user direction.
- ⏳ **P3** — reach-broadening universal hooks + a few searchable evergreen titles (94.8% Shorts-feed dependent today).
- ⏳ Measure A/B after ~2 weeks, then set `TITLE_AB_LEGACY_PCT=0`.
- 📌 **None of the working-tree pipeline edits are committed yet** — awaiting user's go-ahead.

**Research done (2026-06-28):** Deep-research workflow `wrmwhzc38` FAILED (synthesis-schema error); replaced with manual web-search pass. Findings in `CHANNEL_IMPROVEMENT_PLAN.md` §5. Key: (1) sub-conversion is now a 2026 *distribution* signal, so fixing it also lifts views; (2) YouTube's July-2025 "inauthentic content" policy targets repetition/low-human-value (monetization risk for us) — not AI use itself; (3) >10 Shorts/week dilutes reach; (4) titles truncate ~40 chars, hashtags-in-title hurt CTR (already addressed in P0); (5) series + mid-video/muted CTAs push conversion from ~0.12% toward 1–2%.

## Conventions / gotchas
- Voice: the view-driving voice is the **ElevenLabs Adam clone** (not synthetic Ryan) — see `memory/voice-config-adam-clone.md`.
- Scripts: present tense, no profanity, no place names, no stage directions, Muslim teen slang 1–3×, end on a cliffhanger. Never use "astaghfir*" (TTS can't pronounce).
- Titles now live WITHOUT hashtags; hashtags belong in the description only.
