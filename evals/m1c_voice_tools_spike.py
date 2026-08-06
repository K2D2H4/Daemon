"""Does a voice tool call cost the answer it was called from? Ask the socket.

`daemon/voice/gemini_live.py` can now declare tools, read a `toolCall` and send a
`toolResponse`. Four things about that cannot be settled against a fake socket,
and each one is a different kind of wrong:

  1. **Does a native-audio session accept `tools` at all,** and does it still
     return audio once it has them. The docs say native audio and function calling
     compose; a socket closed 1007 would say otherwise, and it would say it by
     ending voice mode rather than by failing a setting.
  2. **Is `behavior: NON_BLOCKING` accepted on this model.** The docs say
     asynchronous function calling is *not* supported on Gemini 3.1 Flash Live,
     which is the model this repo runs (`m0_voice_spike.RECOMMENDED_MODEL`). If
     that is current, `scheduling` is unreachable here and question 4 is moot. The
     brief that asked for this work named `gemini-live-2.5-flash-preview` instead -
     an id `m0_voice_spike` already recorded as shut down - so the two facts have
     to be re-read off the wire rather than off either document.
  3. **Does `toolResponse` interrupt generation.** This is the one that matters,
     and it matters because the same shape has already cost this product a
     conversation. `send_context` is `clientContent`, which the Live API documents
     as interrupting "any current model generation": seeding one recall mid-answer
     produced **2.2 s of audio against 46.7 s with recall off and 38.8 s with it
     deferred to the turn boundary**, and `serverContent.interrupted` came back
     90 ms later (docs/PLAN.md, `daemon/voice/base.py:Interrupted`). A
     `toolResponse` is a *different* top-level client message and nothing documents
     it as interrupting. "Different message, therefore safe" is exactly the kind of
     inference this file exists to replace.
  4. **What each `scheduling` value actually does** - `INTERRUPT`, `WHEN_IDLE`,
     `SILENT` - measured rather than read. `INTERRUPT` is deliberately not the
     default in `gemini_live.py`; if it turns out to be harmless, that is a change
     to make on evidence.

Run it once the key is in `.env`:

    python3 -m evals.m1c_voice_tools_spike

It sends text and reads the reply, so it needs no microphone. The tool it declares
is a fake clock that touches nothing. Nothing is written to the repo, and the key
is only ever read from the environment.

**What it found** lives where the code that acts on it is - the
`--- tool calling ---` comment in `daemon/voice/gemini_live.py` - and in
`evals/CLAUDE.md`. Short version, 2026-08-05: 3 is answered and is the good answer,
1 holds, and 2/4 came back worse than a rejection would have been. The questions
above are left as questions because they are what to re-ask when the model changes.

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md); that is why this lives in `evals/`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

# The same `.env` reader `m0_voice_spike` uses, imported rather than copied: two
# parsers for one file is two places for "why is my key not picked up" to live.
from evals.m0_voice_spike import RECOMMENDED_MODEL, _load_env

PROMPT = (
    "지금 시각을 알려줘. 반드시 get_time 도구를 써서 확인하고, "
    "확인하는 동안에도 말을 멈추지 말고 한두 문장 더 얘기해줘."
)
"""Two demands on purpose. The first forces a tool call; the second asks the model
to keep talking while it waits, which is the only state in which question 3 has an
answer - a response that lands in a silence cannot interrupt anything."""

TOOL_ANSWER = "2026-08-05T14:03:00+09:00"

TURN_BUDGET_SECONDS = 60.0
"""One turn's ceiling. `receive()` ends at the turn boundary, but a session that
stops answering would otherwise sit here until the server's own idle abort - which
m0 measured arriving as 1008 "The operation was aborted", i.e. minutes."""

MAX_TURNS = 3
"""A blocking tool call may end the turn at the call and deliver the answer in the
next `receive()`. Whether it does is itself unknown, so the spike reads a few
turns rather than assuming which one carries the audio.

It only reaches for another turn when the current one left the question open - a
call was answered and nothing played after it. Reading speculatively would park on
`receive()` for the whole `TURN_BUDGET_SECONDS`, four times over, on a socket
billed by the minute."""


def _spec() -> Any:
    from daemon.llm.base import ToolSpec

    return ToolSpec(
        name="get_time",
        description="The current time on the owner's machine, as an ISO-8601 string.",
        parameters={"type": "object", "properties": {}},
    )


class Reading:
    """What one configuration did. Deliberately flat: this is printed, not consumed.

    Times are recorded relative to the moment the `toolResponse` was written, because
    every question above is really "what happened after we answered".
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.error: str | None = None
        self.key_leaked = False
        self.setup_accepted = False
        self.calls: list[tuple[str, str]] = []
        self.audio_before = 0
        self.audio_after = 0
        self.interrupted_after: float | None = None
        self.interrupts = 0
        self.transcripts: list[tuple[str, str]] = []
        self.answered_at: float | None = None
        self.turns = 0
        self.ended: str | None = None

    @property
    def seconds_before(self) -> float:
        return _seconds(self.audio_before)

    @property
    def seconds_after(self) -> float:
        return _seconds(self.audio_after)


def _seconds(pcm_bytes: int) -> float:
    """Output is 24 kHz 16-bit mono, so two bytes a sample. Seconds rather than
    bytes because seconds are what the owner experienced: "2.2s of audio" is the
    number that made the recall defect legible."""
    from daemon.voice.gemini_live import OUTPUT_SAMPLE_RATE

    return round(pcm_bytes / (OUTPUT_SAMPLE_RATE * 2), 2)


async def _measure(
    api_key: str, model: str, label: str, *, behavior: str = "", scheduling: str = ""
) -> Reading:
    from daemon.llm.base import ToolCall
    from daemon.tools.base import ToolResult
    from daemon.voice.base import Interrupted, Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    reading = Reading(label)
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        tools=[_spec()],
        tool_behavior=behavior,
        tool_scheduling=scheduling,
    )
    try:
        async with session:
            reading.setup_accepted = True
            await session.send_text(PROMPT)
            for _ in range(MAX_TURNS):
                got_anything = False
                async with asyncio.timeout(TURN_BUDGET_SECONDS):
                    async for event in session.receive():
                        got_anything = True
                        if isinstance(event, bytes):
                            if reading.answered_at is None:
                                reading.audio_before += len(event)
                            else:
                                reading.audio_after += len(event)
                        elif isinstance(event, Transcript):
                            reading.transcripts.append((event.role, event.text))
                        elif isinstance(event, Interrupted):
                            reading.interrupts += 1
                            if (
                                reading.answered_at is not None
                                and reading.interrupted_after is None
                            ):
                                reading.interrupted_after = round(
                                    time.perf_counter() - reading.answered_at, 3
                                )
                        elif isinstance(event, ToolCall):
                            reading.calls.append((event.id, event.name))
                            # Answered from inside the loop, on purpose: that is
                            # where PR-2b will answer it, and answering after the
                            # turn boundary would measure a different thing.
                            await session.send_tool_response(
                                [
                                    ToolResult(
                                        call_id=event.id,
                                        name=event.name,
                                        content=TOOL_ANSWER,
                                    )
                                ]
                            )
                            reading.answered_at = time.perf_counter()
                if not got_anything:
                    break
                reading.turns += 1
                if reading.answered_at is None or reading.audio_after:
                    # Either the model never called the tool, or it kept talking
                    # after being answered. Both are answers; another `receive()`
                    # would only wait out the budget.
                    break
            reading.ended = session.ended
    except TimeoutError:
        reading.error = f"no turn boundary within {TURN_BUDGET_SECONDS:.0f}s"
    except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
        reading.error = f"{type(exc).__name__}: {exc}"
        reading.key_leaked = api_key in str(exc) or api_key in repr(exc)
    return reading


def _report(reading: Reading) -> None:
    print(f"  {reading.label}")
    if reading.error:
        print(f"    FAILED: {reading.error}")
        print(f"    key present in the error text: {reading.key_leaked}  <- must be False")
        if not reading.setup_accepted:
            print("    setup was refused, so this configuration is unusable on this model")
        return
    print(f"    tool calls received: {reading.calls or 'NONE'}")
    if not reading.calls:
        print("    the model never asked for the tool - the frame is declared and unexercised")
    print(
        f"    audio before the response: {reading.seconds_before}s"
        f"  after: {reading.seconds_after}s"
    )
    print(
        f"    interrupted: {reading.interrupts}x"
        + (
            f", first {reading.interrupted_after}s after the response"
            if reading.interrupted_after is not None
            else " after the response: no"
        )
    )
    print(f"    turns read: {reading.turns}, session ended: {reading.ended!r}")
    for role, text in reading.transcripts:
        print(f"    [{role}] {text!r}")
    if reading.calls and reading.seconds_after == 0.0 and reading.seconds_before > 0:
        print(
            "    ^ nothing played after the answer. That is the shape recall had "
            "(2.2s vs 46.7s) - suspect the response killed the turn."
        )


async def main() -> int:
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Put it in .env and run this again.")
        return 1

    configured = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip()
    model = configured or RECOMMENDED_MODEL
    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"model: {model}{'' if configured else '  (falling back to the recommended id)'}\n")

    print("1+3. a blocking call - the default, and what gemini_live.py sends today:")
    blocking = await _measure(api_key, model, "behavior=<absent> scheduling=<absent>")
    _report(blocking)
    print()

    if blocking.error and not blocking.setup_accepted:
        print(
            "Setup was refused with tools declared at all, so questions 2 and 4 have "
            "no ground to stand on. Fix this first."
        )
        return 1

    print("2+4. NON_BLOCKING, one session per scheduling value:")
    from daemon.voice.gemini_live import TOOL_SCHEDULING

    for scheduling in TOOL_SCHEDULING:
        reading = await _measure(
            api_key,
            model,
            f"behavior=NON_BLOCKING scheduling={scheduling}",
            behavior="NON_BLOCKING",
            scheduling=scheduling,
        )
        _report(reading)
        print()

    print(
        "Put the numbers in the `--- tool calling ---` comment in "
        "daemon/voice/gemini_live.py, next to the recall ones. A default changes "
        "only on the strength of what is printed above."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
