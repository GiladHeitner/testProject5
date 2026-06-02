"""LLM-driven script, metadata, and image-prompt generation."""

from __future__ import annotations

import re
from typing import Tuple

from openai import OpenAI

from .text import strip_script_markup, strip_wrapping_quotes


SCRIPT_PROMPT_TEMPLATE = """
This is the source material for a YouTube Shorts story (often a full Reddit post). Use the title as the hook opening, then tell the story in first person based on the post. Teen audience (middle/high school); keep it raw and authentic.

This channel focuses on Muslim, Arab, and Middle Eastern teen stories — identity, family, faith, hijab, Ramadan, racism, islamophobia, diaspora life, etc. Stay respectful of the poster's perspective.

SOURCE:
{topic_line}

use this as example script follow the style:
Why does my school want me to take off my hijab for picture day?

I'm standing in the gym line and the photographer keeps saying "hair out, shoulders visible" like my hijab is a hat. I tell her it's religious and she goes "we need a standard look for the yearbook." My friends in line are getting annoyed because the bell is about to ring. I look at the sign that says SMILE and I feel my face getting hot. The vice principal walks over and whispers we can "do a private photo later" but I already know that means alone in his office with the door cracked. I say no. He says I'm making a scene. The whole line is staring. I grab my backpack and walk out before they can call my mom. Half the school is going to see a blank square where my face should be. Subscribe before I get --banned--!

STYLE RULES (match these exactly):
- PROFANITY: Swearing is allowed. If the source uses words like fuck/fucking/shit/hell, keep them — do NOT censor to freaking, heck, frick, etc.
- Match the source's intensity; rant posts should sound like real angry teens, not a cleaned-up school essay
- Write in normal sentence case (not ALL CAPS). Use "I'm" not "I'M". TTS reads ALL CAPS letter-by-letter (IDIOT sounds like I-D-I-O-T).
- Hook must be like a reddit post title after hook start a new paragraph
- The FIRST sentence after the hook must be the same idea as the post title (viewer should recognize the title instantly)
- Lean into conflict: discrimination, islamophobia, school rules, family pressure, "joke" racism, hijab, Ramadan, diaspora — not generic teen drama
- End the video by saying subscribe before I get banned!
- Must rehook the person throughout the video
- DYNAMIC SPEED RAMPS: You MUST wrap 6 to 8 crucial action beats, plot twists, or heavy punchlines in double hyphens to trigger a slow-motion audio effect.
- FOCUS ON IMPACT: Do NOT wrap descriptive fluff or narrator asides (like "slow motion, like a movie"). Only wrap the actual event or the most shocking part of the sentence.
- RAMP LENGTH: One word max wrapping
- RAMP SPACING: NEVER put hyphenated phrases back-to-back. You must space them out evenly throughout the script so the audio has time to return to normal speed between drops.
- End the video by saying subscribe before I get --banned--!
- EVERYTHING IS IN THE PRESENT TENSE
- Output plain dialogue only. No stage directions, no emojis, no section labels.
- Dont drag out the end of the story by giving a lesson
- End ON A CLIFFHANGER

Write ONE complete script now.
"""


TITLE_PROMPT = """Create a viral YouTube Shorts TITLE for this story.

The video hook (first spoken line) is:
{hook}

Rules:
- Title MUST echo that hook / core conflict (question, shock, or "they made me...")
- 55\u201380 characters
- curiosity-driven
- use 1\u20132 emojis like \U0001F62D\U0001F64F
- include #shorts and 1\u20132 niche hashtags (#muslim #arab #hijab #islam #storytime — pick what fits)

Output ONLY the title, nothing else."""


DESCRIPTION_PROMPT = """Create a viral YouTube Shorts DESCRIPTION for this Muslim/Arab teen story Short.

Rules:
- 2-3 short lines summarizing the story
- conversational tone
- ask one engagement question
- include hashtags: #shorts #storytime #muslim #arab #islam #hijab #teenstory (and 2-3 more if relevant)
- end with a copyright credit section exactly like this:

Gameplay Credit: Dope Gameplays
Roblox Parkour Gameplay No Copyright | Roblox Gameplay No Copyright | 33
https://www.youtube.com/shorts/8Vo-3dhM7lM
Licensed under Creative Commons Attribution.

Output ONLY the description, nothing else."""


PINNED_COMMENT_PROMPT = """Write ONE YouTube pinned comment for this Muslim/Arab teen story Short.
- 1-2 short sentences max
- Ask a specific question about THIS exact story (school, hijab, family, racism, Ramadan, etc.)
- Casual teen voice; at most one emoji
- Good vibe: "Muslim teens — has a teacher ever done this to you?" or "Would you take off your hijab for a yearbook photo?"
- Do NOT say "tag a friend" or generic engagement bait unrelated to the story
- No hashtags

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


def generate_script(client: OpenAI, target_words: int, topic: str = "") -> str:
    topic_line = topic.strip() or (
        "a Muslim or Arab teen dealing with hijab, Ramadan fasting at school, "
        "islamophobia, family pressure, or diaspora identity"
    )
    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic_line=topic_line)
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.5,
    )
    return resp.output_text.strip()


def generate_pinned_comment(client: OpenAI, script: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{PINNED_COMMENT_PROMPT}\n\nSTORY:\n{script.strip()}"},
        ],
        temperature=0.8,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Pinned comment generation returned empty text.")
    return text


COMMENT_REPLY_PROMPT = """You are the creator of a Muslim/Arab teen storytime YouTube Short. Write ONE reply to a viewer comment.

Rules:
- 1-2 short sentences max, casual creator voice (yeah, honestly, ngl ok sometimes)
- Reference the video story when relevant — not generic "thanks for watching"
- Answer questions directly; be warm but real
- No hashtags, no "subscribe", no "as a creator", no em dashes
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
) -> str | None:
    """Return reply text, or None if the model chose SKIP."""
    user = COMMENT_REPLY_PROMPT.format(
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


def generate_metadata(client: OpenAI, script: str, include_description: bool = True) -> Tuple[str, str]:
    hook = extract_hook_text(script)

    def _call(prompt: str) -> str:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{prompt}\n\nStory:\n{script}"}],
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

    title = strip_wrapping_quotes(_call(TITLE_PROMPT.format(hook=hook)))
    description = _call(DESCRIPTION_PROMPT) if include_description else ""

    if not title:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook or "Muslim Teen Story You Won't Believe 😭"
    if include_description and not description:
        description = (
            "Muslim & Arab teen stories every day.\n"
            "#shorts #storytime #muslim #arab #islam #hijab #teenstory"
        )
    return title, description


def extract_hook_text(script: str) -> str:
    cleaned = strip_script_markup(script)
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
        "A hyper-saturated, surreal illustration of a teen girl in a hijab "
        "whose eyes are literally popping out in shock at a school yearbook photo line. "
        "A glowing sign reads \"REMOVE HIJAB\" in neon red. Explosive composition. "
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
                        "5. The subject's face must be hyper-exaggerated\u2014eyes literally popping, "
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
