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
_Last updated: 2026-07-01_

**2026-07-01 live-Studio + registry audit (key corrections):**
- ❗ **Actual cadence through Jun 28 was 4–5 uploads/day** (registry timestamps ~01:15/06:30/15:14/20:15 UTC — the external cron was firing 4–5×/day). Dropped to 1/day from Jun 29. The Jun-28 "cadence ~1/day confirmed" note was wrong at the time. Verify cron-job.org has exactly ONE enabled job.
- ❗ Content flood Jun 22–28 was ~8 near-identical arranged-marriage videos + **out-of-season Ramadan scripts** (Ramadan 2026 ended Mar 19; bot posted "I'm fasting right now" content in late June). Daily views fell ~37K/day (early June) → ~5K/day.
- ❗ The ED/body-image story ("Why Do I Feel Fat at 103 Pounds?!", mcBsUYjEGNM) became series Part 1 + Part 2 (Jun 28/30) — uploaded before the safety filter shipped. Brand/policy risk; user to decide on unlisting.
- ✅ New pipeline confirmed live in production as of the Jul 1 upload: curiosity title variant, no emoji/hashtags, new CTA ("tomorrow's story…"). Old CTA "Subscribe before I get banned!" ran on every video through Jun 28.
- 📊 28-day (Jun 3–30): 252.6K views (+28%), 816.9h watch (flat), +202 subs (−21%), 522 total.
- 🏆 All-time top Shorts = **specific incident + injustice/identity conflict, school-set**: "Why All the Muslim Hate?" 25K/318 comments; "Secret Language" 22K; "Teacher Forces Hijab Removal" 10K; "Cops Called for Eid Prayers" 9.6K; "Teacher Calls Quran a Comic Book" 9K. Generic musing-style scripts underperform.
- ℹ️ Channel identity: **the name stays "HalalRants"** (user's explicit 2026-07-01 decision). What the user wants back is the **"Muslim Rants"-era CONTENT style** (the Apr–early-June hits): specific incident + injustice/identity-conflict stories, school/public settings, "the hate we get" discussion topics. NOT a rename. (Note: both allowed name changes for the 14-day window were burned on 2026-07-01 by a rename+revert misunderstanding — no name changes possible until ~2026-07-15.)
- 🎣 **Hook fix shipped 2026-07-01:** 21/25 recent uploads opened with rhetorical questions; root cause was `ADAPTATION_BRIEF_PROMPT` asking for a "reddit-title style question" hook. Now: brief + script prompt demand declarative just-happened statement hooks, and `generate_script` regenerates (≤3 attempts) when `hook_is_weak()` flags a question/vague opener. Part 2 continuations exempt.

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

**Shipped 2026-06-28 (quality batch, from 3-agent code audit):**
- ✅ **Scripts:** ED/body-image safety filter (`_STORYTIME_REJECT_RE` + SAFE CONTENT rule + new fallback image); 4 rotating exemplars (was 1 hardcoded → sameness); anti-cliché ban list + mid-video re-hook; script temp 0.5→0.85.
- ✅ **Theme-cooldown bug fix:** `classify_topic_theme` now scores all buckets (was first-match → marriage/islamophobia swallowed others).
- ✅ **Render (verified via isolated harness, frames inspected):** sidechain music ducking; `loudnorm` -14 LUFS master; popups cropped 4:5 portrait (was square → cut off faces); **gameplay swap** — background cuts to a new spot every ~6s (env `GAMEPLAY_SWAP_SECONDS`, 0 disables) via split/trim/concat in `build_filter_complex`.
- Render verification harness: `scratchpad/render_harness.py` (synthetic narration, no TTS/OpenAI) — reuse for future `video.py` changes.

**Known deferred items from the audit (not yet done):** wire the dead `hook_video.py` PiP intro back in (biggest visual fix, needs render verify); popup entrance "pop" animation; caption scale-pop + drop random caps; LLM story-quality judge for topic pick; wire unused emoji-caption layer. TTS 1.7× speed is memory-protected (Adam clone) — verify production path before touching.

**Next up:**
- ⏳ **#3 posting-time optimization** — blocked: Studio analytics "when viewers online" page too flaky to load. Low impact for Shorts. Revisit or set an evening cron slot on cron-job.org.
- ⏳ Measure A/B after ~2 weeks → set `TITLE_AB_LEGACY_PCT=0`. Watch length-bump retention.
- ⏳ Optional: tighten niche filter (a body-image topic slipped through), consistent visual/intro branding, searchable evergreen titles.

**Research done (2026-06-28):** Deep-research workflow `wrmwhzc38` FAILED (synthesis-schema error); replaced with manual web-search pass. Findings in `CHANNEL_IMPROVEMENT_PLAN.md` §5. Key: (1) sub-conversion is now a 2026 *distribution* signal, so fixing it also lifts views; (2) YouTube's July-2025 "inauthentic content" policy targets repetition/low-human-value (monetization risk for us) — not AI use itself; (3) >10 Shorts/week dilutes reach; (4) titles truncate ~40 chars, hashtags-in-title hurt CTR (already addressed in P0); (5) series + mid-video/muted CTAs push conversion from ~0.12% toward 1–2%.

## Conventions / gotchas
- Voice: the view-driving voice is the **ElevenLabs Adam clone** (not synthetic Ryan) — see `memory/voice-config-adam-clone.md`.
- Scripts: present tense, no profanity, no place names, no stage directions, Muslim teen slang 1–3×, end on a cliffhanger. Never use "astaghfir*" (TTS can't pronounce).
- Titles now live WITHOUT hashtags; hashtags belong in the description only.
