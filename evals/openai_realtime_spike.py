"""The OpenAI Realtime live spike. The only part of Phase B1 a real key can answer.

`daemon/voice/openai_realtime.py` was built from the Realtime API reference (both
its GA and older beta event-name spellings) and unit-tested against a fake
socket. Two things cannot be settled that way:

  1. **Which event spelling the model actually sends.** GA renamed several
     events (`response.output_audio.delta` instead of `response.audio.delta`,
     and similarly for the transcript events); the decoder accepts both
     spellings so it works either way, but only a live run says which one this
     account's model actually emits. This script prints every raw `type`
     string it sees on the wire, in order, alongside the decoder's own output.
  2. **Whether 1007/1008 close codes are really permanent here.** Ported from
     gemini_live.py's classification as the closest analogue, not yet measured
     against OpenAI's own socket (see the `_PERMANENT_CLOSE_CODES` comment in
     openai_realtime.py). This spike does not manufacture a bad-key run - that
     is cheap to check separately - but a real run's own close code, if the
     socket drops, is worth reading against that constant.

Run it once the key is in `.env`:

    python3 -m evals.openai_realtime_spike

It sends text and reads the reply, so it needs no microphone. Nothing is
written to the repo, and the key is only ever read from the environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

PROMPT = "Say a one-sentence hello."

VOICE_NAME = "alloy"

SYSTEM_INSTRUCTION = "You are a terse assistant in a one-off live diagnostic. Be brief."


def _load_env() -> None:
    """Read .env the same way pydantic-settings would, without importing it, so
    this script also works before the package is installed."""
    from pathlib import Path

    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class _SniffingSocket:
    """Wraps the real websocket connection to also record each raw message's
    `type` field, in arrival order, without changing anything `receive()` sees.

    `OpenAIRealtimeSession.receive()` only yields decoded items - audio bytes,
    `Transcript`, `Interrupted`, `ToolCall` - never the server's own event name,
    so pinning the GA/beta spelling needs a second look at the same messages.
    Going through the session's own `connect` injection point (already there
    for the unit tests' fake socket) rather than reaching into its private
    `_decode` keeps this spike outside the seam it is measuring.
    """

    def __init__(self, inner: Any, raw_types: list[str]) -> None:
        self._inner = inner
        self._raw_types = raw_types

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        async for raw in self._inner:
            try:
                msg = json.loads(raw)
            except ValueError:
                msg = None
            if isinstance(msg, dict):
                event_type = msg.get("type")
                if isinstance(event_type, str):
                    self._raw_types.append(event_type)
            yield raw


async def _run(api_key: str, model: str) -> int:
    import websockets

    from daemon.llm.base import ToolCall
    from daemon.voice.base import Interrupted, Transcript
    from daemon.voice.openai_realtime import OpenAIRealtimeSession

    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"model: {model}")
    print(f"voice: {VOICE_NAME}\n")

    raw_types: list[str] = []

    async def _connect(url: str, **kwargs: Any) -> Any:
        ws = await websockets.connect(url, **kwargs)
        return _SniffingSocket(ws, raw_types)

    session = OpenAIRealtimeSession(
        api_key=api_key,
        model=model,
        system_instruction=SYSTEM_INSTRUCTION,
        voice_name=VOICE_NAME,
        connect=_connect,
    )

    audio_bytes = 0
    first_audio: float | None = None
    transcripts: list[tuple[str, bool, str]] = []
    saw_interrupted = False
    tool_calls: list[tuple[str, dict[str, Any]]] = []

    try:
        async with session:
            started = time.perf_counter()
            await session.send_text(PROMPT)
            async for item in session.receive():
                if isinstance(item, (bytes, bytearray)):
                    audio_bytes += len(item)
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    print(f"  audio delta: {len(item)} bytes")
                elif isinstance(item, Transcript):
                    transcripts.append((item.role, item.final, item.text))
                    print(f"  transcript [{item.role}] final={item.final}: {item.text!r}")
                elif isinstance(item, Interrupted):
                    saw_interrupted = True
                    print("  Interrupted")
                elif isinstance(item, ToolCall):
                    tool_calls.append((item.name, item.arguments))
                    print(f"  ToolCall: {item.name}({item.arguments!r})")
                else:
                    print(f"  unrecognised item: {item!r}")
    except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(f"  key present in the error text: {api_key in str(exc)}  <- must be False")
        return 1

    time_to_first_audio_ms = round((first_audio - started) * 1000) if first_audio else None

    print("\nraw server event types seen, in order:")
    for event_type in raw_types:
        print(f"  {event_type}")
    if not raw_types:
        print("  NONE - nothing arrived before the turn ended")

    print(f"\ntime to first audio: {time_to_first_audio_ms} ms")
    print(f"total audio bytes: {audio_bytes}")
    print(f"transcripts ({len(transcripts)}):")
    for role, final, text in transcripts:
        print(f"  [{role}] final={final} {text!r}")
    if not transcripts:
        print("  NONE - memory and persona evolution do not work in voice mode without these")
    print(f"interrupted seen: {saw_interrupted}")
    print(f"tool calls: {tool_calls}")

    if not audio_bytes:
        print("\n  WARNING: no audio - the text turn may not have triggered generation")

    return 0


def main() -> int:
    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("DAEMON_OPENAI_REALTIME_MODEL", "").strip()

    missing = [
        name
        for name, value in (("OPENAI_API_KEY", api_key), ("DAEMON_OPENAI_REALTIME_MODEL", model))
        if not value
    ]
    if missing:
        print(
            f"Missing: {', '.join(missing)}. Set them in .env or the environment, e.g.:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  DAEMON_OPENAI_REALTIME_MODEL=gpt-realtime\n"
            "then run this again.",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_run(api_key, model))


if __name__ == "__main__":
    sys.exit(main())
