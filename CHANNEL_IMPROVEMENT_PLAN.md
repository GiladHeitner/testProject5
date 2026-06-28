# @HalaalRants — Pipeline Improvement Plan

_Drafted 2026-06-28. Based on live YouTube Studio analytics + code audit of `shorts_bot` pipeline._
_The "External research" section is filled in when the deep-research workflow completes._

---

## 1. The diagnosis (what the data actually says)

The channel is **NOT failing on reach** — it is failing on **loyalty / subscriber conversion**.

| Metric (last 28 days) | Value | Read |
|---|---|---|
| Views | 295,441 (+85%) | Algorithm IS pushing us |
| Unique monthly viewers | 144.7K | Big top-of-funnel |
| Stayed to watch | **60.6%** | Hooks work |
| Avg % viewed (top vids) | 85–98% | Retention is strong |
| Discovery: Shorts feed | **94.8%** | 100% algo-dependent, no search moat |
| **Regular viewers** | **<0.1%** | ⚠️ Nobody comes back |
| **Subscriber conversion** | **~0.12%** (≈1 / 900 views) | ⚠️ Renting views, not building audience |
| Total subs / videos | 516 / 233 (≈2.2 subs/video) | — |

**Competitor benchmark — Studonomy (@Studonomy):** same faceless-storytime-over-gameplay format.
- 26.4K subs on **63 videos** (≈419 subs/video — **~190× our per-video efficiency**).
- Recent shorts: 25K–375K views each.
- Titles: _"Why Would She Push Her Brother Into Traffic?"_, _"Everyone Hated Her Until They Learned Why…"_ — third-person, curiosity-gap, **no emoji, no hashtags, answer withheld.**

**Two root causes:**
1. **No reason to return** — faceless, identical synthetic voice, every video a disposable one-off, no series/arc/recurring character.
2. **Titles spoil the story AND niche-gate the audience** — first-person, emoji-spammed, hashtag-stuffed, premise given away. This caps both reach and the curiosity that drives a follow.

---

## 2. Where the code is actively causing this

| File / location | Problem | Evidence |
|---|---|---|
| `shorts_bot_lib/script_ai.py:84-100` (`TITLE_PROMPT`) | Instructs **the exact failing pattern**: `use 1–2 emojis like 😭🙏`, `include #shorts and 1–2 niche hashtags`, `Title MUST echo that hook` (spoils story). | Our titles all look like _"17 & Pressured to Marry?! 😭 #shorts"_ |
| `shorts_bot.py:485-486`, `1039-1040` | Force-appends `#Shorts` to every title. | Hashtag clutter in title text |
| `script_ai.py:72`, `:58` + `SCRIPT_PROMPT` | CTA is hardcoded `"subscribe before I get banned!"` — generic, gives no reason to return. | Weak conversion despite high retention |
| `script_ai.py:25,44` + `channel_persona.py` | Persona is locked first-person Muslim-Arab-teen; titles inherit the niche gate. | Audience ceiling; non-Muslim scrollers don't click |
| `reddit_topics.py` (no diversity/cooldown logic found) | Only exact-post dedup (`used_reddit.txt`); **no thematic spread or cooldown**. | 10+ near-identical "arranged marriage" shorts |
| `run.sh` / posting cadence | 5–8 uploads/day floods our own tiny base; algo picks 1–2 winners, buries rest (13/19/78-view flops). | Recent decline from 9–18K hits to 1–2.5K |

---

## 3. The plan (prioritized by leverage)

### P0 — Title rewrite (highest ROI, lowest effort, free) — ✅ SHIPPED 2026-06-28
**Goal:** curiosity-gap, third-person-or-ambiguous, answer withheld, no emoji/hashtag spam.

- [x] Rewrote `TITLE_PROMPT` in `script_ai.py:84`:
  - "Title MUST echo that hook" → **"WITHHOLD THE PAYOFF — open a curiosity gap, never reveal the outcome."**
  - Removed the `use 1–2 emojis` rule (now **NO emojis**).
  - Removed `include #shorts and niche hashtags` from the title — hashtags stay in the **description** only.
  - Added 5 Studonomy-style few-shot examples adapted to the niche + a universal-conflict rule (non-Muslim scrollers should still click).
  - Tightened length 55–80 → **40–70 chars**.
- [x] Removed the forced `#Shorts` title append (`shorts_bot.py` both metadata blocks); cleaned the emoji/hashtag-laden fallback titles in `shorts_bot.py` and `script_ai.py`.
- [x] **A/B discipline — SHIPPED 2026-06-28.** `pick_title_variant()` in `script_ai.py` sends ~20% of uploads through the legacy emoji/hashtag prompt (`TITLE_PROMPT_LEGACY`) and the rest through the curiosity-gap prompt. Share controlled by env `TITLE_AB_LEGACY_PCT` (default 20; set 0 to ship 100% new). The chosen arm is recorded per upload as `title_variant` in `.github/upload_registry.jsonl` (via `append_upload_registry`) so CTR/conversion can be compared. **Measure after ~2 weeks, then set `TITLE_AB_LEGACY_PCT=0`.**

### P1 — Give viewers a reason to return (attacks the <0.1% number)
- [x] **Stronger, specific CTA — SHIPPED 2026-06-28.** Replaced "subscribe before I get banned" with a return-promise CTA: _"Subscribe so tomorrow's story finds you."_ Updated `SCRIPT_PROMPT_TEMPLATE` (example + rule) in `script_ai.py` and `_SUBSCRIBE_PHRASES` in `subscribe_cta.py` so the GIF overlay still times to the new line (legacy phrase still detected as fallback).
- [x] **Series / open loops — SHIPPED 2026-06-28.** New `shorts_bot_lib/series.py`: ~`SERIES_PART1_PCT`% (default 30) of automated runs become a **Part 1** ending on a cliffhanger with a "Part 2 tomorrow" CTA; the next run continues and resolves it as **Part 2**, then clears state. State persists in `.github/series_state.json` (committed by the workflow). Wired into `main()` (role decision → topic continuation → script directive → title `(Part 1)/(Part 2)` suffix → state advance on upload). Unit-tested: `tests/test_series.py` (7 tests, runs with or without pytest). Env: `SERIES_ENABLED`, `SERIES_PART1_PCT`.
- [ ] **Recurring identity (partial).** "Omar" is already the consistent named host (`channel_persona.py`); the series mechanic adds continuity. Still TODO: a consistent visual signature / intro tag for instant recognition.

### P2 — Topic diversity + cadence (stop self-cannibalizing)
- [x] **Thematic cooldown — SHIPPED 2026-06-28.** `classify_topic_theme()` buckets each topic (marriage / islamophobia / ramadan / religion / dating / school / family / other); `_apply_theme_cooldown()` drops candidates whose theme appears in the last N picks (env `TOPIC_THEME_COOLDOWN`, default 4) in BOTH `pick_reddit_post` and `pick_topics_file_fallback`. History persisted in `.github/used_themes.txt` (added to the workflow commit step). Verified: marriage+islamophobia get excluded when recently used. This directly kills the 10× "arranged marriage" run.
- [x] **Cadence — already handled, no change needed.** Reality check: `run.sh` = 1 video/run (no loop). Real triggers were cron-job.org "YoutubeUploader" (1×/day @ 8 AM) **+ the 2 GitHub `schedule:` crons removed earlier today** (committed `a4b9249..ecab486`). Net is now **~1 upload/day** — already inside the ideal 1–2/day range. (Earlier "5–8/day" was an overcount of YouTube's "1 day ago" buckets.) Could add a 2nd daily cron slot later, but do NOT exceed 2/day (>10/week dilutes reach — §5.3).
- [ ] Double down on proven winners (Muslim-hate, hijab-at-school, Ramadan-at-school all hit 9–18K) and retire arranged-marriage repetition. _(Now enforced automatically by the cooldown; could further bias `topic_priority_score` toward proven themes.)_

### P3 — Broaden reach without abandoning the niche
- [ ] Frame hooks so a non-Muslim scroller still clicks (universal-conflict angle in the title, niche payoff in the video). Studonomy's universal hooks are why they hit 375K.
- [ ] Build a search moat: currently 94.8% Shorts-feed — add a few evergreen, searchable titles to reduce 100% algo dependence.

---

## 4. Suggested implementation order
1. **Title rewrite + drop `#Shorts` append** (P0) — one prompt edit, ship today, measure CTR/conversion in 1–2 weeks.
2. **CTA rewrite** (P1) — pairs naturally with the title change.
3. **Theme cooldown + cadence cut** (P2).
4. **Recurring-character / series mechanic** (P1) — bigger lift, do after P0/P1 prove out.
5. **Reach-broadening hooks** (P3).

Measure success by the **regular-viewer %** and **subs-per-1K-views**, not raw views (we already win on views).

---

## 5. External research findings (2025–2026 sources)
> _Deep-research harness run (`wrmwhzc38`) FAILED on a synthesis-schema error after 103 agents — no cited report produced. Findings below come from a focused manual web-search pass (June 2026)._

### 5.1 YouTube's "inauthentic / mass-produced content" policy (effective July 15, 2025)
- YouTube renamed "repetitious content" → **"inauthentic content"** in the YPP policy. It explicitly disallows **"content that is only slightly different from video to video,"** template-based output, and anything **"made with a template with little to no variation"** or **"easily replicable at scale."** Synthetic voiceovers churned at high volume are called out. ([TechCrunch](https://techcrunch.com/2025/07/09/youtube-prepares-crackdown-on-mass-produced-and-repetitive-videos-as-concern-over-ai-slop-grows/), [Social Media Today](https://www.socialmediatoday.com/news/youtube-clarifies-monetization-update-inauthentic-repeated-content/752892/))
- **Important nuance:** this is primarily a **monetization-eligibility** policy, not a confirmed distribution throttle. AI-assisted content **remains eligible** if it has **unique human-added value** and is thoughtfully edited. ([Fliki](https://fliki.ai/blog/youtube-monetization-policy-2025), [Social Media Today](https://www.socialmediatoday.com/news/youtube-clarifies-monetization-update-inauthentic-repeated-content/752892/))
- **Verdict for us:** Our 10× near-identical "arranged marriage" shorts + synthetic voice at 5–8/day is squarely the pattern this targets — a real monetization risk and a "low human value" signal. **Confirms P2 (topic diversity) and the cadence cut.** The fix is variation + human-added value per video, _not_ abandoning AI.

### 5.2 Shorts algorithm 2026 — why high views ≠ subscribers
- **Subscriber conversion is now an explicit ranking signal:** Shorts that convert even a small % of viewers to subs get **boosted future distribution.** So our <0.1% regular / ~0.12% conversion isn't just a vanity problem — it is **actively capping our reach.** Fixing conversion should lift views too. ([Socialync](https://www.socialync.io/blog/youtube-shorts-algorithm-2026), [vidIQ](https://vidiq.com/blog/post/youtube-shorts-algorithm/))
- **Watch-time replaced swipe-rate** as the dominant signal. Push-wider retention threshold ≈ **65% for sub-30s** Shorts, **50% for 30–60s.** ([Socialync](https://www.socialync.io/blog/youtube-shorts-algorithm-2026), [Miraflow](https://miraflow.ai/blog/youtube-shorts-algorithm-update-january-2026))
- **Views are inflated:** since Mar 31, 2025 every play/replay counts as a view; only **Engaged Views** count for YPP. (Maps exactly to our 295K views vs 168K engaged views.) ([Epidemic Sound](https://www.epidemicsound.com/blog/youtube-shorts-algorithm/), [vidIQ](https://vidiq.com/blog/post/youtube-shorts-algorithm/))
- **Subscriber boost has declined** — having subs no longer guarantees a strong cold-start seed; each Short is re-tested on its own merits. ([Socialync](https://www.socialync.io/blog/youtube-shorts-algorithm-2026))
- **Length:** 30–45s is the 2026 sweet spot; **sub-15s collapsed** in reach (can't clear the absolute watch-time bar even at 100%). Our hits run 0:18–0:28 — short but OK; nudging toward 30–40s while holding retention may raise absolute watch time. ([Miraflow](https://miraflow.ai/blog/how-youtube-algorithm-decides-who-sees-your-shorts-2026))

### 5.3 Posting frequency
- Channels posting **>10 Shorts/week see per-Short viewership decline** as the algorithm spreads distribution across too many uploads. **3–4 high-quality/week beats daily mediocre.** 2–3/day is the upper bound *if* quality holds. We're at **35–56/week** — far over. **Confirms the cadence cut to 1–2/day.** ([BigMotion](https://www.bigmotion.ai/blog/how-many-youtube-shorts-should-i-post-a-day), [JoinBrands](https://joinbrands.com/blog/youtube-shorts-best-practices/))

### 5.4 Titles & hooks
- **First <3 seconds decides stay-vs-swipe**; the goal is an **information gap / curiosity loop** that compels "what happens next." Optimizing the first ~7s correlated with large distribution lifts. ([Praper Media](https://prapermedia.com/blog/make-viral-youtube-shorts/), [Miraflow](https://miraflow.ai/blog/youtube-shorts-best-practices-2026-complete-guide))
- **Title length:** keep **30–40 chars (4–6 words)** — that's the feed-truncation limit; front-load the hook. A trailing "…" can raise clicks. **→ Applied:** tightened `TITLE_PROMPT` to 30–50 chars, front-loaded. ([Miraflow templates](https://miraflow.ai/blog/youtube-shorts-titles-descriptions-2026-templates), [ytzolo](https://ytzolo.com/blog/youtube-video-title-length-best-practices-2026/))
- **Hashtags in the title kill CTR** — max 1–2, at the end, or none. **→ Confirms** our P0 move of stripping `#shorts`/niche tags from titles into the description. ([HashtagTools](https://hashtagtools.io/blog/youtube-shorts-hashtags-title-vs-description-2026))

### 5.5 Converting Shorts viewers → subscribers (the core fix)
- **Benchmark:** 1M Shorts views ≈ 500–5,000 subs (**0.05–0.5%** is normal). Our ~0.12% is mid-range — _normal but not optimized_. Tactics that push it to **1–2%**: **cliffhanger format, recurring series, specific CTAs, end-card/playlist retention.** ([Fluxnote](https://fluxnote.io/guides/youtube-shorts-to-subscribers-strategy-2026))
- **Series are the single biggest conversion lever:** "when a viewer watches part one of a series, they immediately have a reason to subscribe so they don't miss the next part." Run 3–4 recurring series with consistent naming. **→ Confirms P1 (series / recurring identity).** ([Conbersa](https://www.conbersa.ai/learn/how-to-grow-on-youtube-shorts-in-2026), [Fluxnote](https://fluxnote.io/guides/youtube-shorts-to-subscribers-strategy-2026))
- **CTA mechanics we're under-using:** add a **verbal CTA around the 20s mark** (not only the end — many swipe before the final frame) AND **on-screen text CTA in the final 3s** for the 30–40% who watch muted. Make it **specific**, not generic. **→ New action: move/duplicate our subscribe CTA earlier + add muted-viewer on-screen text.** ([Fluxnote](https://fluxnote.io/guides/youtube-shorts-to-subscribers-strategy-2026), [Subscribr](https://subscribr.ai/p/convert-shorts-viewers-to-subscribers))
- **Cohesive branding** (consistent hook patterns, clustered topics, recognizable style) makes subscribing "a natural decision." ([Conbersa](https://www.conbersa.ai/learn/how-to-grow-on-youtube-shorts-in-2026))

### 5.6 How the research reconciles with the plan
Everything above **confirms the original ranking.** Net adjustments:
1. **Reframe the goal:** subscriber conversion is now a *distribution* lever, not just vanity — the whole plan also raises views. (Strengthens urgency of P1.)
2. **Title length:** tightened to 30–50 chars (done) — feed truncates ~40.
3. **New P1 sub-task — CTA timing:** add a ~20s mid-video verbal CTA + a final-3s on-screen **text** CTA for muted viewers (currently we only have one end-of-video GIF). See `subscribe_cta.py` / `keyword_popups.py`.
4. **Cadence:** research backs ≤1–2/day; >10/week measurably dilutes per-Short reach.
5. **Don't fear AI per se:** the policy risk is *repetition / low human value*, not AI use — so P2 (variation) matters more than going faceless→faced for policy reasons (though a face still helps conversion).
