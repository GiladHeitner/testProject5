"""LLM-driven script, metadata, and image-prompt generation."""

from __future__ import annotations

import os
import random
import re
from typing import TYPE_CHECKING, Tuple

from openai import OpenAI

from .text import (
    CURRENT_STORY_YEAR,
    modernize_source_years,
    strip_paralinguistic_tags,
    strip_script_markup,
    strip_speed_ramp_hyphens,
    strip_wrapping_quotes,
)

if TYPE_CHECKING:
    from .channel_persona import ChannelPersona


ADAPTATION_BRIEF_PROMPT = """You are planning a YouTube Shorts rant script.

TODAY IS {current_year}. The host is a 17-year-old Muslim Arab teen in {current_year}. Retell the source as happening now — never 2023 or earlier.

SOURCE STORY (Reddit post or topic):
{topic_line}

CHANNEL HOST (same person every video — retell SOURCE as their own experience):
{persona_block}

Write a short adaptation brief (plain text, not JSON):
1. How this story maps to the host (what happened, emotional beat)
2. Three to five concrete details to weave in (emotions, people, actions — no city/state/country names)
3. Hook angle in the host's voice (reddit-title style question or shock line)

Keep it under 120 words."""


SCRIPT_PROMPT_TEMPLATE = """
This is the source material for a YouTube Shorts story (often a full Reddit post). Retell it in first person as the CHANNEL HOST below — this is their own story, venting to camera. Teen audience; keep it raw and authentic.

TIMELINE: Today is {current_year}. Omar is 17. Everything happens in {current_year}. Never say 2023, 2024, or 2025 — if the source mentions old years, retell as happening now.

CHANNEL HOST (same person every video):
{persona_block}

ADAPTATION NOTES:
{adaptation_brief}

SOURCE:
{topic_line}

Example script style (match this energy — male Muslim teen, no swearing, Muslim slang):
Why does my teacher think Ramadan fasting is just skipping meals?

Wallah I'm sitting in class and my teacher keeps going on about how it's not healthy to skip meals like I'm on some diet trend. I tell her it's Ramadan and she says I need to focus on my studies, not starve myself. My friends exchange looks because they know how important this is to me. I feel my stomach twist, not from hunger but from frustration... The principal calls me in and says we can discuss accommodations, but I know that means eating alone in the office while everyone else is at lunch. I say no. He tells me I'm being difficult. The whole class is watching. I grab my stuff and walk out before they can call my parents. Half the school is going to think I'm just skipping lunch for fun. Subscribe so tomorrow's story finds you.

STYLE RULES (match these exactly):
- NO PROFANITY: Never use swear words. Express anger through tone and slang instead.
- SLANG: Use Muslim/Arab teen slang (wallah, yallah, habibi, inshallah, etc.) like a real diaspora kid — 1-3 times per script, natural. Never use "astaghfir" or "astaghfirullah" — the voiceover cannot pronounce them.
- NO STAGE DIRECTIONS: Do not include bracketed tags like [sigh] / [pause] / [breath] or any other actions.
- NO LAUGH WORDS: Never write laugh, laughter, funny, chuckle, giggle, haha, or lol — TTS will actually laugh. Say "mocking", "teasing", or "making fun of me" instead.
- NO GEOGRAPHY: Never name cities, states, or countries. Say "my school" or "at home", not place names. Strip place names from the source if needed.
- Match the source's intensity; rant posts should sound like real angry teens, not a cleaned-up school essay
- Write in normal sentence case (not ALL CAPS). Use "I'm" not "I'M". TTS reads ALL CAPS letter-by-letter (IDIOT sounds like I-D-I-O-T).
- Use aggressive punctuation (!, ..., ?) at emotional peaks to force faster, angrier delivery
- Hook must be like a reddit post title after hook start a new paragraph
- The FIRST sentence after the hook must be the same idea as the post title (viewer should recognize the title instantly)
- Lean into conflict: discrimination, islamophobia, school rules, family pressure, Ramadan, diaspora — not generic teen drama
- End with a subscribe line that gives a REASON TO RETURN, not a generic plea. It must contain the word "subscribe" and promise the next story, e.g. "Subscribe so tomorrow's story finds you" or "Subscribe, I post one of these every day." Do NOT use "subscribe before I get banned".
- Must rehook the person throughout the video
- NEVER use double hyphens (--word--) or em dashes. Use commas or periods instead.
- EVERYTHING IS IN THE PRESENT TENSE
- Output spoken dialogue only. No stage directions, no bracketed actions, no emojis, no section labels.
- Dont drag out the end of the story by giving a lesson
- End ON A CLIFFHANGER

Write ONE complete script now.
"""


TITLE_PROMPT = """Create a viral YouTube Shorts TITLE for this story.

The video hook (first spoken line) is:
{hook}

CHANNEL HOST (the story is told by this creator, but the TITLE itself does NOT have to be first-person):
{persona_block}

GOAL: open a CURIOSITY GAP. The title must make a scroller NEED to watch to find out what happened. Tease the conflict; never reveal the outcome or the lesson.

Rules:
- WITHHOLD THE PAYOFF. Hint at something shocking/unfair/surprising WITHOUT saying how it ends.
  Bad (spoils it): "17 & Pressured to Marry, Here's How I Said No"
  Good (gap):       "My Parents Sat Me Down and Said I'm Getting Married Next Month"
- 30–50 characters, ~4–7 words. Shorts-feed titles truncate near 40 chars, so FRONT-LOAD the hook in the first 30. Punchy.
- Plain, conversational, specific. First-person ("My teacher...", "They made me...") OR
  ambiguous third-person is fine — whichever is more clickable.
- A scroller who is NOT Muslim should still want to click. Lead with the UNIVERSAL conflict
  (teacher, cops, family, betrayal, being singled out); the niche payoff lives inside the video.
- NO emojis. NO hashtags. NO "#shorts". Output the bare title text only.
- No swearing; no city/country names; no ALL-CAPS words.

Examples of the curiosity-gap style to match:
- My Teacher Made Me Take This Off in Front of the Whole Class
- They Called the Cops on Us for Being "Too Loud"
- My School Suspended Me and Wouldn't Tell Me Why
- Everyone Went Quiet When I Walked In, Then I Found Out Why
- My Family Found Out Who I've Been Texting

Output ONLY the title, nothing else."""


# Old emoji/hashtag formula, kept only as the A/B control arm (see pick_title_variant).
TITLE_PROMPT_LEGACY = """Create a viral YouTube Shorts TITLE for this story.

The video hook (first spoken line) is:
{hook}

CHANNEL HOST (title should sound like this creator posted it):
{persona_block}

Rules:
- Title MUST echo that hook / core conflict (question, shock, or "they made me...")
- 55–80 characters
- curiosity-driven
- use 1–2 emojis like 😭🙏
- include #shorts and 1–2 niche hashtags (#muslim #arab #islam #storytime — pick what fits)
- No swearing; no city names

Output ONLY the title, nothing else."""


DESCRIPTION_PROMPT = """Create a viral YouTube Shorts DESCRIPTION for this Muslim/Arab teen story Short.

CHANNEL HOST (write as this creator):
{persona_block}

Rules:
- 2-3 short lines summarizing the story
- conversational tone matching the host
- ask one engagement question
- include hashtags: #shorts #storytime #muslim #arab #islam #teenstory (and 2-3 more if relevant)
- No swearing; no city names
- Do NOT include gameplay credits (added automatically after generation)

Output ONLY the description, nothing else."""


PINNED_COMMENT_PROMPT = """Write ONE YouTube pinned comment for this Muslim/Arab teen story Short.

CHANNEL HOST (write as this creator):
{persona_block}

Rules:
- 1-2 short sentences max
- Ask a specific question about THIS exact story (school, family, racism, Ramadan, etc.)
- Casual teen voice; Muslim slang ok; at most one emoji
- Good vibe: "Wallah has a teacher ever done this to you?" or "Would you walk out too?"
- Do NOT say "tag a friend" or generic engagement bait unrelated to the story
- No hashtags; no swearing

Output ONLY the comment text."""


MUSLIM_SHORT_TAGS = (
    "shorts",
    "storytime",
    "muslim",
    "arab",
    "islam",
    "hijab",
    "ramadan",
    "muslim teen",
    "arab teen",
    "islamophobia",
    "diaspora",
    "hijabi",
    "reddit story",
)


def _resolve_persona(persona: ChannelPersona | None) -> ChannelPersona:
    if persona is not None:
        return persona
    from .channel_persona import load_channel_persona

    return load_channel_persona()


def _persona_block(persona: ChannelPersona | None) -> str:
    from .channel_persona import format_persona_block

    return format_persona_block(_resolve_persona(persona))


def _generate_adaptation_brief(
    client: OpenAI,
    *,
    topic_line: str,
    persona: ChannelPersona | None,
) -> str:
    prompt = ADAPTATION_BRIEF_PROMPT.format(
        topic_line=topic_line,
        persona_block=_persona_block(persona),
        current_year=CURRENT_STORY_YEAR,
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or "Retell the source as the host's own rant in present tense."
    except Exception as exc:
        print(f"Adaptation brief failed ({exc}); continuing without brief.")
        return "Retell the source as the host's own rant in present tense."


def generate_script(
    client: OpenAI,
    target_words: int,
    topic: str = "",
    *,
    persona: ChannelPersona | None = None,
) -> str:
    topic_line = modernize_source_years(topic.strip()) or (
        "a Muslim or Arab teen dealing with Ramadan fasting at school, "
        "islamophobia, family pressure, or diaspora identity"
    )
    brief = _generate_adaptation_brief(client, topic_line=topic_line, persona=persona)
    prompt = SCRIPT_PROMPT_TEMPLATE.format(
        persona_block=_persona_block(persona),
        adaptation_brief=brief,
        topic_line=topic_line,
        current_year=CURRENT_STORY_YEAR,
    )
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.5,
    )
    return strip_speed_ramp_hyphens(resp.output_text.strip())


def generate_pinned_comment(
    client: OpenAI,
    script: str,
    *,
    persona: ChannelPersona | None = None,
) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{PINNED_COMMENT_PROMPT.format(persona_block=_persona_block(persona))}"
                    f"\n\nSTORY:\n{script.strip()}"
                ),
            },
        ],
        temperature=0.8,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Pinned comment generation returned empty text.")
    return text


COMMENT_REPLY_PROMPT = """You are the channel host of a Muslim/Arab teen storytime YouTube Short. Write ONE reply to a viewer comment.

CHANNEL HOST:
{persona_block}

Rules:
- 1-2 short sentences max, casual creator voice (yeah, honestly, ngl ok sometimes)
- Muslim slang ok; no swearing
- Reference the video story when relevant — not generic "thanks for watching"
- Answer questions directly; be warm but real
- No hashtags, no "subscribe", no "as a creator", no em dashes, no city names
- If the comment is toxic, bait, spam, or you cannot reply naturally, output exactly: SKIP
- Otherwise output ONLY the reply text (no quotes, no labels)

VIDEO TITLE:
{video_title}

VIDEO STORY (context):
{script}

VIEWER COMMENT:
{comment}
"""


def generate_comment_reply(
    client: OpenAI,
    *,
    script: str,
    video_title: str,
    comment_text: str,
    author_name: str = "",
    persona: ChannelPersona | None = None,
) -> str | None:
    """Return reply text, or None if the model chose SKIP."""
    user = COMMENT_REPLY_PROMPT.format(
        persona_block=_persona_block(persona),
        video_title=(video_title or "Storytime short").strip(),
        script=(script or "").strip()[:1200],
        comment=(comment_text or "").strip(),
    )
    if author_name.strip():
        user += f"\n\nCommenter display name: {author_name.strip()}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user}],
        temperature=0.9,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw or raw.upper() == "SKIP" or raw.upper().startswith("SKIP"):
        return None
    if len(raw) > 240:
        raw = raw[:237].rstrip() + "..."
    return raw


def pick_title_variant() -> str:
    """A/B split for title style.

    Returns "legacy" (old emoji/hashtag formula) for a fraction of uploads so we
    keep a clean control arm to measure the curiosity-gap rewrite against; returns
    "curiosity" otherwise. Control the legacy share with TITLE_AB_LEGACY_PCT
    (default 20). Set it to 0 once the new formula is proven to ship 100% new.
    """
    try:
        pct = float(os.environ.get("TITLE_AB_LEGACY_PCT", "20"))
    except ValueError:
        pct = 20.0
    pct = max(0.0, min(100.0, pct))
    return "legacy" if random.random() * 100.0 < pct else "curiosity"


def generate_metadata(
    client: OpenAI,
    script: str,
    include_description: bool = True,
    *,
    persona: ChannelPersona | None = None,
    gameplay_credit: str = "",
    title_variant: str = "curiosity",
) -> Tuple[str, str]:
    hook = extract_hook_text(script)
    block = _persona_block(persona)

    def _call(prompt: str) -> str:
        spoken_script = strip_paralinguistic_tags(script)
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{prompt}\n\nStory:\n{spoken_script}"}],
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

    title_prompt = TITLE_PROMPT_LEGACY if title_variant == "legacy" else TITLE_PROMPT
    title = strip_wrapping_quotes(
        _call(title_prompt.format(hook=hook, persona_block=block))
    )
    description = (
        _call(DESCRIPTION_PROMPT.format(persona_block=block))
        if include_description
        else ""
    )

    if not title:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook or "I Wasn't Ready for What Happened Next"
    if include_description and not description:
        description = (
            "Muslim & Arab teen stories every day.\n"
            "#shorts #storytime #muslim #arab #islam #teenstory"
        )
    if include_description and description and gameplay_credit.strip():
        description = f"{description.rstrip()}\n\n{gameplay_credit.strip()}"
    return title, description


def extract_hook_text(script: str) -> str:
    cleaned = strip_paralinguistic_tags(strip_script_markup(script))
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    hook_sentences = [s for s in sentences if s.strip()][:2]
    hook = " ".join(hook_sentences).strip()
    return hook or cleaned[:160]


def summarize_script_for_image(client: OpenAI | None, script: str) -> str:
    hook = extract_hook_text(script)
    fallback = " ".join(hook.split()[:4]) or "story"
    if client is None:
        return fallback
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Turn this video HOOK into a 2-4 word photo search query that visually "
                        "represents the hook's subject/setting. Nouns only. No punctuation. "
                        "No people names.\n\nHOOK:\n"
                        f"{hook}"
                    ),
                }
            ],
            temperature=0.3,
        )
        query = (resp.choices[0].message.content or "").strip().strip('"').strip()
        query = re.sub(r"[^a-zA-Z0-9 ]", "", query)
        return query or fallback
    except Exception:
        return fallback


def build_dalle_prompt(client: OpenAI | None, script: str) -> str:
    cleaned = strip_script_markup(script)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    first_sentence = sentences[0] if sentences else cleaned[:200]
    fallback = (
        "A hyper-saturated, surreal illustration of a teen boy in a school hallway "
        "whose eyes are literally popping out in shock at a cafeteria lunch line. "
        "A glowing sign reads \"NOT A DIET\" in neon red. Explosive composition. "
        "Vertical 9:16."
    )
    if client is None:
        return fallback
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite Shorts thumbnail artist. Your goal is maximum "
                        "visual shock in 0.5 seconds. You create high-energy, over-the-top, "
                        "vibrant imagery, NOT realistic cinematic shots."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create a surreal, explosively vibrant visual that DIRECTLY depicts "
                        "the literal subject, setting, and action of the FIRST SENTENCE below. "
                        "The image must visually match that exact moment so a viewer instantly "
                        "understands what the video is about in under a second. Use the rest of "
                        "the script only as secondary context.\n\n"
                        "MANDATORY RULES:\n"
                        "1. The scene MUST depict the people, objects, and setting stated in the "
                        "first sentence (not a generic school scene, unless the hook is about school).\n"
                        "2. Pick a 1-to-3 word curiosity-spike phrase drawn from the first sentence "
                        "(e.g., \"SECRET CODE\", \"BUSTED!\", \"BIG MISTAKE\", \"DO NOT READ\").\n"
                        "3. This text MUST appear on an object that fits the first sentence "
                        "(phone, note, chalkboard, sign, jersey, screen, etc.).\n"
                        "4. The text must GLOW with intense, neon energy (like \"radioactive green\", "
                        "\"electric blue\", or \"hot pink\") and must be clearly readable.\n"
                        "5. The subject's face must be hyper-exaggerated—eyes literally popping, "
                        "mouth hanging open, extreme cartoony panic.\n"
                        "6. Use an \"Explosive Composition\" where fitting elements are flying "
                        "around the subject.\n"
                        "7. Specify \"Hyper-saturated colors\" and \"Illustrative, high-energy style\".\n"
                        "8. Vertical 9:16 composition.\n\n"
                        f"FIRST SENTENCE (primary focus):\n{first_sentence}\n\n"
                        f"FULL SCRIPT (context only):\n{cleaned}"
                    ),
                },
            ],
            temperature=0.9,
        )
        prompt = (resp.choices[0].message.content or "").strip().strip('"').strip()
        if not prompt:
            return fallback
        if "9:16" not in prompt:
            prompt = f"{prompt} Vertical 9:16 composition."
        return prompt
    except Exception as exc:
        print(f"Error engineering vibrant prompt: {exc}")
        return fallback
