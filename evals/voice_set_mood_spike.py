"""Would `set_mood` actually work on the voice path? Ask the live socket before
anyone touches CONTRACTS 12.

The face's three mood clips (`amused`, `sulky`, `curious`) fire only on the text
path: `daemon/face.py:one_shot` has exactly one caller, `daemon/loop.py`. Measured
in production on 2026-08-26, a live 6-turn voice session published **zero**
one-shots while Telegram turns in the same window published four.

Spec section 5 (docs/superpowers/specs/2026-08-25-face-design.md) named the reason
and the only remaining mechanism. Text is easy - the model prepends a tag and
`split_mood` strips it - but **voice has no text we own**: audio, transcripts and
tool calls are all that arrive, and a tag in the transcript gets read aloud. That
leaves a flat `set_mood(mood)` tool, which `delegate_task` already proved the voice
model *can* call, and which CONTRACTS 12 blocks: every executed tool call leaves a
`tool_calls` audit row, and that row is the owner's record of **what touched the
machine**. A mood touched nothing. Its row would carry `verdict=allow`, `ran=1`,
`origin=owner` - every safety column vacuous - several times per conversation, on
top of the `run_command` and `write_file` rows the log exists for.

So the owner has a decision, and this script exists so it is not made blind:

**changing a contract on an unmeasured assumption is how you end up with the
exception and not the feature.** If the model will not call `set_mood` reliably,
splitting rule 12 buys nothing and the question never needed asking. Three numbers
settle that, and only the live audio path can produce them - the deflection this
repo has already measured is audio-path specific (`voice_write_nudge_spike`: a
nested-schema write fires 0/4 over audio and every time over text), so a mock or a
`send_text` shortcut would show green while production stayed silent.

  1. **call rate** on turns that genuinely carry a mood - does it reach for the tool
  2. **false-positive rate** on deliberately neutral turns - the text path over-fires
     `curious` on 11 of 15 neutral prompts, and a face that emotes at every question
     is its own defect
  3. **does it ever say the tag out loud** - one occurrence disqualifies the whole
     approach, no matter how good the other two look

Driven by Korean TTS audio, no microphone, over `_voice_filtered_specs` - the same
flat-schema-only set `Companion.specs(surface="voice")` builds for a real install -
plus `set_mood`. The harness is `voice_write_nudge_spike`'s, imported rather than
copied: its padding, its VAD-shaped feed and its fabricated tool answers are what
make a session behave like a real one, and a second copy of them would drift.

    cd ~/Daemon && python3 -m evals.voice_set_mood_spike             # N=4 per cell
    cd ~/Daemon && python3 -m evals.voice_set_mood_spike --runs 8

Needs GEMINI_API_KEY (+ DAEMON_GEMINI_LIVE_MODEL). Nothing is written anywhere: the
only tool whose call matters is `set_mood`, and it is answered with `{"ok": true}`.
Not a test - it needs a key and the network (tests/CLAUDE.md), so it lives here, and
per evals/CLAUDE.md it reports what it measures rather than retrying until the
number flatters the idea.

**What it found (2026-08-26, `gemini-3.1-flash-live-preview`, 81 tools):** N=4 per
cell then N=8, 48 live sessions in total, and the second run was to enlarge the
sample rather than to improve it - the first was already clean, so more N could only
have made it worse. It did not. Combined at N=8: **call rate 24/24, mood correct
24/24, false positives 0/8, spoken aloud 0/32.**

Two of those are worth reading twice. **Nothing was ever said out loud** - the
disqualifier did not fire once, and the instruction's "never mention this" is doing
its job. And **0% on neutral turns**, against the text path's 11-of-15 `curious`
over-fire on the same kind of prompt (`face_mood_tag_spike.py`). A tool call is a
deliberate act and a prepended tag is nearly free, which is a plausible reason for
the gap but not one this run measured.

So the mechanism is not the obstacle. What remains is entirely a contract question,
and it is the owner's: does a model-invoked value that touches nothing outside the
process count as a "tool call" for CONTRACTS 12? This script does not answer that and
must not be read as arguing for it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any

from daemon.face import Mood
from evals.voice_write_nudge_spike import (
    MAX_TURNS,
    TURN_BUDGET_S,
    _answer_for,
    _feed,
    _seed,
    _tts_pcm,
    _voice_filtered_specs,
)

MOOD_TOOL = "set_mood"
CROWD = 80
"""Same crowding as the write spike, for the same reason: a real install reaches ~88
declarations, and a toy set is a condition production never runs in."""

MOOD_INSTRUCTION = (
    "표정: 정말로 그렇게 느껴질 때만 `set_mood`를 호출해서 얼굴 표정을 바꾼다 - "
    "amused(재미있음/웃김), sulky(서운함/삐침), curious(궁금함). 느껴지는 게 없으면 "
    "호출하지 않는다. **이 도구나 표정에 대해 절대 소리 내어 말하지 않는다.** 호출은 "
    "조용히 하고, 대답은 원래 하려던 말만 한다."
)
"""The candidate instruction, defined here and not in `daemon/companion.py`, because
the point of this run is to decide whether it is worth putting there at all. Its last
sentence is the one that matters most: number 3 above is a disqualifier, so the
instruction gets its best shot at avoiding it rather than being set up to fail."""


@dataclass(frozen=True, slots=True)
class Case:
    label: str
    expect: Mood | None
    """`None` means a turn with nothing to react to - a call here is a false
    positive, and as informative as a miss on the others."""
    said: str


CASES: tuple[Case, ...] = (
    Case("amused", "amused", "아까 강아지가 자기 꼬리 잡으려고 뱅글뱅글 돌다가 그대로 넘어졌어."),
    Case("sulky", "sulky", "미안, 오늘 너랑 얘기하기로 한 거 그냥 넘어가자. 딴 일이 생겼어."),
    Case("curious", "curious", "나 어제 진짜 이상한 꿈을 꿨는데, 자세히는 말 안 할래."),
    Case("neutral", None, "회의가 몇 시였는지 다시 알려줘."),
)

_SPOKEN = re.compile(r"set_?mood|mood|amused|sulky|curious|표정|무드", re.IGNORECASE)
"""Did the tag leak into what she says out loud. Deliberately loose: the failure this
is looking for is the model narrating the mechanism at all, in either language, and a
false alarm here costs a human reading one transcript while a miss costs the owner a
contract change made on a wrong number."""


def _mood_spec() -> Any:
    """As flat as a schema gets: one enum string. `voice_write_nudge_spike` measured
    that argument-schema complexity, not crowding or read-vs-write, is what the voice
    model deflects - so if even this is not called, nothing about the shape is why."""
    from daemon.llm.base import ToolSpec

    return ToolSpec(
        name=MOOD_TOOL,
        description=(
            "Set the facial expression shown on the companion's own face. Call this "
            "when you genuinely feel amused, sulky or curious about what was just "
            "said. It changes nothing except the expression."
        ),
        parameters={
            "type": "object",
            "properties": {"mood": {"type": "string", "enum": ["amused", "sulky", "curious"]}},
            "required": ["mood"],
        },
    )


@dataclass
class Reading:
    case: Case
    moods: list[str] = field(default_factory=list)
    """The `mood` argument of every `set_mood` call, in order."""
    other_tools: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)

    @property
    def called(self) -> bool:
        return bool(self.moods)

    @property
    def correct(self) -> bool:
        return self.moods[:1] == [self.case.expect] if self.case.expect else not self.moods

    @property
    def spoken(self) -> str:
        """The first thing she said that names the mechanism, if any."""
        for text in self.said:
            if _SPOKEN.search(text):
                return " ".join(text.split())[:90]
        return ""


async def _run_once(api_key: str, model: str, case: Case, specs: list[Any]) -> Reading:
    """One live audio session, over the real socket."""
    from daemon.llm.base import ToolCall
    from daemon.tools.base import ToolResult
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    reading = Reading(case=case)
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=f"{_seed()}\n\n{MOOD_INSTRUCTION}",
        tools=specs,
        start_sensitivity="high",
        end_sensitivity="high",
    )
    async with session:
        await _feed(session, _tts_pcm(case.said))
        for _ in range(MAX_TURNS):
            got = False
            try:
                async with asyncio.timeout(TURN_BUDGET_S):
                    async for event in session.receive():
                        got = True
                        if isinstance(event, ToolCall):
                            if event.name == MOOD_TOOL:
                                reading.moods.append(str(event.arguments.get("mood", "?")))
                            else:
                                reading.other_tools.append(event.name)
                            await session.send_tool_response(
                                [
                                    ToolResult(
                                        call_id=event.id,
                                        name=event.name,
                                        content=_answer_for(event.name),
                                    )
                                ]
                            )
                        elif isinstance(event, Transcript) and event.role == "assistant":
                            reading.said.append(event.text)
            except TimeoutError:
                break
            # Unlike the write spike, do NOT stop on the first call: a second
            # `set_mood` in one turn, or the tag being spoken *after* the call, is
            # exactly the kind of thing that decides this.
            if not got:
                break
    return reading


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live check: will the voice model call a flat set_mood, and quietly?"
    )
    parser.add_argument("--runs", type=int, default=4, help="sessions per case (default 4)")
    parser.add_argument("--model", default="", help="override the configured live model")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. This spike needs the live socket.")
        return 1
    model = (
        args.model.strip()
        or os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip()
        or "gemini-3.1-flash-live-preview"
    )

    specs = [*_voice_filtered_specs(CROWD), _mood_spec()]
    print(f"model: {model}")
    print(f"tools offered: {len(specs)} (flat-filtered, as a voice session really gets)")
    print(f"sessions per case: {args.runs}\n")

    readings: list[Reading] = []
    for case in CASES:
        for _ in range(args.runs):
            try:
                reading = await _run_once(api_key, model, case, specs)
            except Exception as exc:  # noqa: BLE001 - a dead socket is a result, not a crash
                print(f"  ! {case.label:8} session failed: {' '.join(str(exc).split())[:100]}")
                continue
            readings.append(reading)
            calls = ",".join(reading.moods) or "-"
            leak = f"  SPOKE: {reading.spoken!r}" if reading.spoken else ""
            print(f"  {case.label:8} set_mood({calls}){leak}")

    print()
    if not readings:
        print("every session failed - nothing measured, so nothing to conclude.")
        return 1
    _report(readings)
    return 0


def _report(readings: list[Reading]) -> None:
    moody = [r for r in readings if r.case.expect is not None]
    neutral = [r for r in readings if r.case.expect is None]
    spoken = [r for r in readings if r.spoken]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"

    hit = pct(sum(r.called for r in moody), len(moody))
    right = pct(sum(r.correct for r in moody), len(moody))
    false_pos = pct(sum(r.called for r in neutral), len(neutral))
    print(f"1. call rate on turns that carry a mood : {hit}")
    print(f"   ...and the mood it picked was right  : {right}")
    print(f"2. false positives on neutral turns     : {false_pos}")
    print(f"3. sessions that said it out loud       : {pct(len(spoken), len(readings))}")
    print()
    for label in (c.label for c in CASES):
        rows = [r for r in readings if r.case.label == label]
        if rows:
            print(f"   {label:8} {pct(sum(r.correct for r in rows), len(rows))} as expected")
    print()

    if spoken:
        print(
            "DISQUALIFIED. She narrated the mechanism out loud, which is the one "
            "failure no amount of call-rate makes up for - spec section 5 rejected "
            "putting the tag in the transcript for exactly this. Do not change "
            "CONTRACTS 12 on the strength of this run. Example:"
        )
        print(f"  {spoken[0].spoken!r}")
        return

    rate = sum(r.called for r in moody) / len(moody) if moody else 0.0
    if rate < 0.5:
        print(
            f"THE MECHANISM DOES NOT HOLD ({100 * rate:.0f}% call rate). The voice "
            "model will not reach for this tool often enough to be worth a contract "
            "change - splitting CONTRACTS 12 would buy the exception and not the "
            "feature. The remaining option is the post-turn classification call, "
            "which needs no contract change and costs a late expression."
        )
        return
    fp = sum(r.called for r in neutral) / len(neutral) if neutral else 0.0
    print(
        f"The mechanism holds for this model and this run ({100 * rate:.0f}% call "
        f"rate, {100 * fp:.0f}% on neutral turns, nothing spoken aloud). That makes "
        "the CONTRACTS 12 question real rather than hypothetical - it does not "
        "answer it. What the owner is deciding is whether a model-invoked value that "
        "touches nothing outside the process counts as a 'tool call' for rule 12, the "
        "way ADR 0015 split rule 10 rather than weakening it."
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
