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

**Shipped & committed/pushed to main (2026-06-28):**
- ✅ **P0 — Title rewrite:** curiosity-gap `TITLE_PROMPT` (no emoji/hashtags, withholds payoff, 30–50 chars, universal-conflict framing). Removed forced `#Shorts` append + cleaned fallback titles.
- ✅ **A/B harness:** `pick_title_variant()` — ~20% legacy (`TITLE_PROMPT_LEGACY`), env `TITLE_AB_LEGACY_PCT`; arm recorded as `title_variant` in `upload_registry.jsonl`.
- ✅ **CTA text rewrite:** → "Subscribe so tomorrow's story finds you" (`script_ai.py` + `_SUBSCRIBE_PHRASES`).
- ✅ **Topic-diversity cooldown:** `classify_topic_theme()` + `_apply_theme_cooldown()` (both selectors); `.github/used_themes.txt`; env `TOPIC_THEME_COOLDOWN`. Tested.
- ✅ **Recurring-Omar Part 1 / Part 2 series** — `shorts_bot_lib/series.py` (unit-tested, `tests/test_series.py`, 7 passing). ~30% (`SERIES_PART1_PCT`) become cliffhanger Part 1; next run resolves as Part 2. State in `.github/series_state.json`. Titles get `(Part 1)/(Part 2)`. **Part 2 auto-pins a "Watch Part 1" link** (stores part1 video id). Workflow has a `series_part1_pct` dispatch input to force a Part 1.
- ✅ **Stronger 3-second hook:** `SCRIPT_PROMPT` opens mid-conflict, no slow setup.
- ✅ **Length bump:** `run.sh` WORDS 100→115 (~30–38s sweet spot). WATCH retention % — revert toward 100 if it drops.
- ✅ **Cadence ~1/day confirmed** (cron-job.org "YoutubeUploader" 8 AM; GitHub schedules removed earlier).
- ✅ **First series Part 1 uploaded** 2026-06-28: youtube.com/watch?v=mcBsUYjEGNM (rolled the legacy A/B title arm). Series state now pending → next run = Part 2.

**Confirmed already working (no change needed):**
- ✅ Comment auto-replies: `comment-reply.yml` runs 3×/day, healthy (verified successful runs).
- ❌ Muted-viewer text CTA: NOT NEEDED — captions transcribe the full narration incl. the subscribe line (already burned in).

**Next up:**
- ⏳ **#3 posting-time optimization** — blocked: Studio analytics "when viewers online" page too flaky to load. Low impact for Shorts. Revisit or set an evening cron slot on cron-job.org.
- ⏳ Measure A/B after ~2 weeks → set `TITLE_AB_LEGACY_PCT=0`. Watch length-bump retention.
- ⏳ Optional: tighten niche filter (a body-image topic slipped through), consistent visual/intro branding, searchable evergreen titles.

**Research done (2026-06-28):** Deep-research workflow `wrmwhzc38` FAILED (synthesis-schema error); replaced with manual web-search pass. Findings in `CHANNEL_IMPROVEMENT_PLAN.md` §5. Key: (1) sub-conversion is now a 2026 *distribution* signal, so fixing it also lifts views; (2) YouTube's July-2025 "inauthentic content" policy targets repetition/low-human-value (monetization risk for us) — not AI use itself; (3) >10 Shorts/week dilutes reach; (4) titles truncate ~40 chars, hashtags-in-title hurt CTR (already addressed in P0); (5) series + mid-video/muted CTAs push conversion from ~0.12% toward 1–2%.

## Conventions / gotchas
- Voice: the view-driving voice is the **ElevenLabs Adam clone** (not synthetic Ryan) — see `memory/voice-config-adam-clone.md`.
- Scripts: present tense, no profanity, no place names, no stage directions, Muslim teen slang 1–3×, end on a cliffhanger. Never use "astaghfir*" (TTS can't pronounce).
- Titles now live WITHOUT hashtags; hashtags belong in the description only.
