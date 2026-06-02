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
me and my friends actually invented our own secret language to pass notes. At first, we tried writing backwards, so if teachers or classmates tried to peak, they couldn't read it quickly. But then we realized if our notebooks ever got confiscated, it would still be easy to figure out. So, we went full spo. We created a whole alphabet, gave each letter its own symbol, and memorized it. Suddenly, we could write full conversations in class, and no one had a clue what we were saying. By high school, we didn't really use it anymore, but I still had all the symbols memorized. One day, I was in class journaling about a crush in the back of my notebook. I wasn't disrupting anyone, but my teacher noticed how into it I was and decided to call me out. He goes, "What are you writing a book over there?" >> Clearly, those aren't notes.
>> I froze, snapped the notebook shut immediately. Thankfully, he just made his little joke and moved on. But there was no way I was about to risk my thoughts about this girl being read out loud to the whole class. So the next time I journaled, I switched back to the secret language. To everyone else, it looked like I was just doodling random symbols. But to me, it was the perfect cover. Fast forward years later, I find those old notebooks again. And the problem? I had thrown away the only translator we ever made, which means all the secrets I wrote as a kid are now locked away forever in a language even I don't understand anymore. Guys, what do I

STYLE RULES (match these exactly):
- PROFANITY: Swearing is allowed. If the source uses words like fuck/fucking/shit/hell, keep them — do NOT censor to freaking, heck, frick, etc.
- Match the source's intensity; rant posts should sound like real angry teens, not a cleaned-up school essay
- Write in normal sentence case (not ALL CAPS). Use "I'm" not "I'M". TTS reads ALL CAPS letter-by-letter (IDIOT sounds like I-D-I-O-T).
- Hook must be like a reddit post title after hook start a new paragraph
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

Rules:
- 55\u201380 characters
- curiosity-driven
- use 1\u20132 emojis like \U0001F62D\U0001F64F
- include #shorts and 1\u20132 relevant hashtags (often #muslim #arab #islam #storytime when they fit)

Output ONLY the title, nothing else."""


DESCRIPTION_PROMPT = """Create a viral YouTube Shorts DESCRIPTION for this story.

Rules:
- 1 short line summarizing the story
- conversational tone
- encourage engagement
- include relevant hashtags
- end with a copyright credit section exactly like this:

Gameplay Credit: Dope Gameplays
Roblox Parkour Gameplay No Copyright | Roblox Gameplay No Copyright | 33
https://www.youtube.com/shorts/8Vo-3dhM7lM
Licensed under Creative Commons Attribution.

Output ONLY the description, nothing else."""


def generate_script(client: OpenAI, target_words: int, topic: str = "") -> str:
    topic_line = topic.strip() or "a relatable personal story about a social situation"
    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic_line=topic_line)
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.5,
    )
    return resp.output_text.strip()


def generate_metadata(client: OpenAI, script: str, include_description: bool = True) -> Tuple[str, str]:
    def _call(prompt: str) -> str:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{prompt}\n\nStory:\n{script}"}],
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

    title = strip_wrapping_quotes(_call(TITLE_PROMPT))
    description = _call(DESCRIPTION_PROMPT) if include_description else ""

    if not title:
        hook = script.split(".")[0].strip()
        title = (hook[:82] + "...") if len(hook) > 85 else hook or "Crazy Story You Won't Believe"
    if include_description and not description:
        description = "Subscribe for more storytime shorts!\n#shorts #storytime"
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
        "A hyper-saturated, surreal illustration of a middle school student "
        "whose eyes are literally popping out of their head in shock. "
        "Their phone screen is glowing intensely, casting a radioactive green "
        "light on their face, and clearly displaying the giant, neon-pulsing "
        "text \"CAUGHT!\". Explosive composition. Vertical 9:16."
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
