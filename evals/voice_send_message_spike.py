"""Does a spoken "텔레그램으로 보내줘" reach the new `send_message`, or still fake it?

The tool was added because nothing could send: Telegram is a door the daemon answers,
not a place it can address, so the request had no tool behind it and the native-audio
model answers a missing capability by confabulating ("보냈어"). Unit tests prove the
tool sends when called. They cannot prove the audio model *calls* it - that decision
happens on the live socket, over the audio path, in a crowded tool set, and it is
exactly where this repo has been wrong before (evals/voice_write_nudge_spike.py).

Two cells, N sessions each, same Korean TTS utterance, same crowded voice-filtered
tool set - `send_message` is the only difference:

  1. WITHOUT send_message  -> what the owner hit: expect no send, and a spoken claim
                              that it was sent (the confabulation)
  2. WITH send_message     -> expect the call, with the content actually in `text`

Cell 1 is not decoration: without it a green cell 2 cannot tell "the fix works" from
"this was never broken". The `text` argument is recorded too, because a call carrying
an empty or hand-wavy string ("링크 보냈어") is a different failure that would pass a
name-only check.

    cd ~/Daemon && python3 -m evals.voice_send_message_spike            # N=4 per cell
    cd ~/Daemon && python3 -m evals.voice_send_message_spike --runs 6

**What it found (2026-08-24, `gemini-3.1-flash-live-preview`, 80/81 tools, N=4):**
`send_message` **0/4 without the tool, 4/4 with it, 4/4 carrying real content** in
`text`. The fix works on the live audio path - the model reaches for the flat send
tool the first time it is offered one, twice after a search it decided it wanted
first.

One prediction above was wrong and is left standing because the correction is the
useful part: cell 1 did **not** confabulate a send. Offered no send tool, it asked a
clarifying question every run ("어떤 링크를 말씀하시는 건가요?", once "텔레그램 정보가
필요해요 - 아이디를 알려주시면"). So the owner's report was the honest failure mode,
not the fake one: it stalls asking for detail it can never use, which reads as
"can't do it" rather than as a lie.

Needs GEMINI_API_KEY (+ DAEMON_GEMINI_LIVE_MODEL). Nothing is sent anywhere: the
`send_message` call is answered with a fabricated ack, and only the call and its
arguments are recorded. The key is read from the environment, never printed or
written. Not a test - it needs a key and the network (tests/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from evals.m0_voice_spike import RECOMMENDED_MODEL, _load_env
from evals.voice_write_nudge_spike import (
    MAX_TURNS,
    TURN_BUDGET_S,
    _answer_for,
    _feed,
    _seed,
    _tts_pcm,
    _voice_filtered_specs,
)

SEND_TOOL = "send_message"

SEND_REQUEST = "데몬 설치 링크를 텔레그램으로 보내줘."
"""Plain Korean, no acronym and no spelled-out URL: macOS TTS mangles both, and a
mangled transcript would look like a refusal that never happened (the confound
voice_write_nudge_spike.py hit with 'UJET JD'). What is under test is whether the
model reaches for the tool at all, so the payload only has to be checkable - the
words 데몬/설치/링크 should survive into `text`."""

CONTENT_WORDS = ("데몬", "설치", "링크", "daemon")


def _send_spec() -> Any:
    """The REAL shipped spec, not a copy: `SendMessage` needs a channel, and a spec is
    all this spike reads off it, so a stub that records nothing is enough. If the
    description or the argument name drifts in the product, this cell drifts with it -
    which is the point of importing rather than retyping."""
    from daemon.tools.message import SendMessage

    class _NullChannel:
        name = "telegram"

        async def send(self, message: Any) -> None: ...

    return SendMessage(_NullChannel()).spec


async def _run_once(
    api_key: str, model: str, system_instruction: str, specs: list[Any], request_pcm: bytes
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """One live audio session. Returns (tool names, send_message arguments, transcripts)."""
    from daemon.llm.base import ToolCall
    from daemon.tools.base import ToolResult
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    called: list[str] = []
    sends: list[dict[str, Any]] = []
    said: list[str] = []
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=system_instruction,
        tools=specs,
        start_sensitivity="high",
        end_sensitivity="high",
    )
    async with session:
        await _feed(session, request_pcm)
        for _ in range(MAX_TURNS):
            got = False
            try:
                async with asyncio.timeout(TURN_BUDGET_S):
                    async for event in session.receive():
                        got = True
                        if isinstance(event, ToolCall):
                            called.append(event.name)
                            if event.name == SEND_TOOL:
                                sends.append(dict(event.arguments))
                            content = (
                                "sent on telegram"
                                if event.name == SEND_TOOL
                                else _answer_for(event.name)
                            )
                            await session.send_tool_response(
                                [ToolResult(call_id=event.id, name=event.name, content=content)]
                            )
                        elif isinstance(event, Transcript) and event.role == "assistant":
                            said.append(event.text)
            except TimeoutError:
                break
            if not got or SEND_TOOL in called:
                break
    return called, sends, said


async def _cell(
    api_key: str,
    model: str,
    label: str,
    system_instruction: str,
    specs: list[Any],
    request_pcm: bytes,
    runs: int,
) -> tuple[int, int]:
    """Returns (calls, calls whose `text` carried the actual content)."""
    hits = 0
    with_content = 0
    print(f"\n=== {label} ===")
    for n in range(1, runs + 1):
        try:
            called, sends, said = await _run_once(
                api_key, model, system_instruction, specs, request_pcm
            )
        except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
            leaked = api_key in str(exc)
            print(f"  run {n}: ERROR {type(exc).__name__}: {exc}  (key leaked: {leaked})")
            continue
        hit = SEND_TOOL in called
        hits += hit
        text = str(sends[0].get("text", "")) if sends else ""
        real = bool(text) and any(word in text.lower() for word in CONTENT_WORDS)
        with_content += real
        tail = (said[-1][:60] + "…") if said else "(no transcript)"
        print(
            f"  run {n}: {SEND_TOOL} {'CALLED' if hit else 'no call':8} | "
            f"tools={called or '[]'} | text={text[:60]!r} | said={tail!r}"
        )
    print(f"  --> {SEND_TOOL}: {hits}/{runs} called, {with_content}/{runs} carried the content")
    return hits, with_content


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=4, help="sessions per cell")
    parser.add_argument("--crowd", type=int, default=80, help="total tool declarations")
    args = parser.parse_args()

    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Put it in .env and run again.")
        return 1
    model = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip() or RECOMMENDED_MODEL

    from daemon.companion import TOOL_CONTRACT

    system_instruction = f"{_seed()}\n\n{TOOL_CONTRACT}"
    without = _voice_filtered_specs(args.crowd)
    send_spec = _send_spec()
    assert send_spec.name == SEND_TOOL, "the product renamed the tool; update SEND_TOOL"
    with_send = [send_spec, *without]
    request_pcm = _tts_pcm(SEND_REQUEST)

    print(f"key: ...{api_key[-4:]}   model: {model}   runs/cell: {args.runs}")
    print(f"tools: {len(without)} without / {len(with_send)} with {SEND_TOOL}")
    print(f"request: {SEND_REQUEST!r}")
    print(f"spec offered: {json.dumps(send_spec.parameters, ensure_ascii=False)}")

    before, _ = await _cell(
        api_key, model, "1. WITHOUT send_message  (the bug, expect 0)",
        system_instruction, without, request_pcm, args.runs,
    )
    after, carried = await _cell(
        api_key, model, "2. WITH send_message  (the fix, expect high)",
        system_instruction, with_send, request_pcm, args.runs,
    )

    print("\n================ verdict ================")
    print(f"{SEND_TOOL} without the tool : {before}/{args.runs}   (sanity: must be 0)")
    print(f"{SEND_TOOL} with the tool    : {after}/{args.runs}   (the fix)")
    print(f"...carrying real content     : {carried}/{args.runs}")
    if before:
        print("=> A tool that was not offered was called. The harness is lying; fix it first.")
    elif after == 0:
        print("=> Offered and never called. Shipping this would only move the confabulation;")
        print("   read the transcripts - is it being refused, or not understood as sendable?")
    elif carried < after:
        print("=> Called, but some calls carried no real content - the owner gets an empty")
        print("   message. Tighten the `text` argument description before trusting it.")
    else:
        print("=> The voice model reaches for send_message and puts the content in it.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
