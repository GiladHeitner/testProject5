"""Tests for the recurring-Omar Part 1 / Part 2 series mechanic.

Pure logic, no network/LLM. Run either way:
    python -m pytest tests/test_series.py -q
    python tests/test_series.py            # plain-python fallback (no pytest needed)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shorts_bot_lib import series  # noqa: E402


def test_decide_role_disabled():
    assert series.decide_role(None, enabled=False) == series.ROLE_STANDALONE
    # Even with a pending part, disabled means standalone.
    pending = series.PendingSeries(2, "t", "topic", "summary", "cliff")
    assert series.decide_role(pending, enabled=False) == series.ROLE_STANDALONE


def test_pending_always_becomes_part2():
    pending = series.PendingSeries(2, "t", "topic", "summary", "cliff")
    # pct=0 would never roll Part 1, but a pending part still forces Part 2.
    assert series.decide_role(pending, enabled=True, pct=0) == series.ROLE_PART2


def test_part1_probability_bounds():
    rng = random.Random(1234)
    # pct=100 always Part 1; pct=0 never.
    assert series.decide_role(None, enabled=True, pct=100, rng=rng) == series.ROLE_PART1
    assert series.decide_role(None, enabled=True, pct=0, rng=rng) == series.ROLE_STANDALONE


def test_part1_roughly_matches_pct():
    rng = random.Random(42)
    n = 4000
    hits = sum(
        series.decide_role(None, enabled=True, pct=30, rng=rng) == series.ROLE_PART1
        for _ in range(n)
    )
    frac = hits / n
    assert 0.25 < frac < 0.35, f"expected ~0.30, got {frac:.3f}"


def test_directives_and_suffix():
    assert "PART 1 OF 2" in series.script_directive(series.ROLE_PART1, None)
    assert "Part 2 tomorrow" in series.script_directive(series.ROLE_PART1, None)
    pending = series.PendingSeries(2, "My Teacher Did This", "topic", "she froze", "will I be expelled")
    d2 = series.script_directive(series.ROLE_PART2, pending)
    assert "FINALE" in d2 and "will I be expelled" in d2 and "she froze" in d2
    assert series.title_suffix(series.ROLE_PART1) == " (Part 1)"
    assert series.title_suffix(series.ROLE_PART2) == " (Part 2)"
    assert series.title_suffix(series.ROLE_STANDALONE) == ""
    # A standalone run gets no directive.
    assert series.script_directive(series.ROLE_STANDALONE, None) == ""


def test_state_roundtrip_and_cliffhanger(tmp_path=None):
    import tempfile

    d = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    state = d / "series_state.json"

    # Nothing pending initially.
    assert series.load_pending(state) is None

    script = (
        "My teacher pulls me aside in front of everyone. She says my essay "
        "is too political and I have to rewrite it tonight. Then the principal "
        "walks in and says we need to talk about something else. "
        "Part 2 tomorrow, subscribe so you don't miss it."
    )
    pending = series.build_pending_from_part1(
        title="My Teacher Pulled Me Aside (Part 1)", topic="source story",
        script=script, video_id="abc123",
    )
    # Cliffhanger should be the last story beat, not the CTA.
    assert "subscribe" not in pending.cliffhanger.lower()
    assert "principal" in pending.cliffhanger.lower()

    series.save_pending(pending, state)
    loaded = series.load_pending(state)
    assert loaded is not None and loaded.part == 2
    assert loaded.title == "My Teacher Pulled Me Aside (Part 1)"
    assert loaded.topic == "source story"
    # Part 1 video id roundtrips and yields a Shorts URL for the Part 2 pin.
    assert loaded.part1_video_id == "abc123"
    assert series.part1_shorts_url(loaded) == "https://www.youtube.com/shorts/abc123"
    assert series.part1_shorts_url(None) == ""

    # A pending state forces the next run to Part 2...
    assert series.decide_role(loaded, enabled=True, pct=0) == series.ROLE_PART2
    # ...and clearing it returns to standalone eligibility.
    series.clear_pending(state)
    assert series.load_pending(state) is None


def test_full_two_run_cycle():
    """Simulate two consecutive automated runs: Part 1 then Part 2."""
    import tempfile

    state = Path(tempfile.mkdtemp()) / "series_state.json"

    # Run 1: no pending, force Part 1.
    role1 = series.decide_role(series.load_pending(state), enabled=True, pct=100)
    assert role1 == series.ROLE_PART1
    part1_script = "Crazy thing happened at school today and then everything stopped. " \
                   "Part 2 tomorrow, subscribe so you don't miss it."
    series.save_pending(
        series.build_pending_from_part1(title="Day at School (Part 1)", topic="t", script=part1_script),
        state,
    )

    # Run 2: pending exists -> must be Part 2 even at pct=0.
    pending2 = series.load_pending(state)
    role2 = series.decide_role(pending2, enabled=True, pct=0)
    assert role2 == series.ROLE_PART2
    assert pending2.topic == "t"
    series.clear_pending(state)

    # Run 3: state cleared -> no longer forced into Part 2.
    assert series.decide_role(series.load_pending(state), enabled=True, pct=0) == series.ROLE_STANDALONE


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc!r}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
