"""Does *our* websocket reach Vertex, or only the SDK's?

`docs/design/vertex-live-transport.md` measured the Vertex endpoint with
google-genai. That proved the endpoint, not this repo's client: the URI, the
bearer-token header and the project-qualified model path are hand-built in
`daemon/voice/vertex.py` and `daemon/voice/gemini_live.py`, and every one of them
fails at the handshake rather than at startup. Three things cannot be settled
against a fake socket:

  1. **The URI and the model path together.** A wrong region is an HTTP 404 on the
     handshake; a bare model id is a 1008 close naming a model nobody configured.
     Both look like "voice is broken" from the outside.
  2. **The bearer token.** `x-goog-api-key` is what this file's sibling sends;
     Vertex refuses it and wants `Authorization`. The token also expires, which is
     why the session asks for headers per connect attempt - visible here only as a
     session that opens twice.
  3. **Whether the transcripts still arrive.** They are not optional: without them
     memory and persona evolution get nothing, and `_setup_message` sends both
     transcription fields unconditionally. Vertex accepting the *fields* does not
     mean it fills them for Korean audio.

  4. **Whether the answer survives the loop that reads it.** Added after the first
     version of this file shipped a transport that then said nothing on the owner's
     own machine. That version opened a bare session - no tools, one text turn - and
     passed, because the defect was not in the session at all: first audio arrives
     2.5-4.4s away here against ~1.7s on the API-key endpoint, and
     `voice/conversation.py` used to read two turn boundaries in that gap as a
     finished session. A spike that does not run the loop cannot see that, so this
     one runs `VoiceConversation` with a microphone that hears nothing, declares
     tools, answers the calls, and reports how many boundaries preceded the answer.

Run it with credentials for a project that has Vertex AI enabled:

    DAEMON_VERTEX_PROJECT=<project> python3 -m evals.vertex_live_spike

No microphone: the fake audio device below hears silence and counts what it is
handed. Nothing is written to the repo.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evals.m0_voice_spike import _load_env

MODEL = "gemini-live-2.5-flash-native-audio"
"""The reason this transport exists: the API-key endpoint closes 1008 for it."""

PROMPT = "안녕? 오늘 좀 피곤한데 짧게 한마디만 해줘."


class _SilentAudio:
    """A microphone that hears nothing and a speaker that counts.

    `AudioIO` (daemon/voice/base.py) exists so a conversation can be driven with no
    hardware, which is what lets this spike run the real loop.
    """

    sample_rate = 16_000
    playback_sample_rate = 24_000

    def __init__(self) -> None:
        self.played = 0
        self.first_at: float | None = None

    async def record(self) -> Any:
        frame = bytes(640)  # 20 ms of silence at 16 kHz
        while True:
            await asyncio.sleep(0.02)
            yield frame

    async def play(self, chunk: bytes) -> None:
        if self.first_at is None:
            self.first_at = time.perf_counter()
        self.played += len(chunk)

    async def stop_playback(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _tools() -> tuple[Any, ...]:
    """A handful of flat declarations, the shape a voice session is offered.

    Not the owner's real registry - a spike may not need their data dir - but not
    zero either, which is what the first version of this file sent and why it
    passed while the product did not: the model answers a declared session by
    calling a tool first, and the answer arrives a turn or two later.
    """
    from daemon.llm.base import ToolSpec

    return tuple(
        ToolSpec(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "무엇을 할지"}},
                "required": [] if name == "system_state" else ["query"],
            },
        )
        for name, description in (
            ("system_state", "사용자가 지금 자리에 있는지, 화면이 켜져 있는지 확인한다."),
            ("read_file", "파일 내용을 읽는다."),
            ("notify", "사용자에게 알림을 보낸다."),
        )
    )


async def _conversation_turn(project: str, location: str, credentials: str) -> dict[str, Any]:
    """One opening, through the product's own loop, on the Vertex transport."""
    from daemon.companion import Companion
    from daemon.voice import vertex
    from daemon.voice.conversation import VoiceConversation
    from daemon.voice.gemini_live import GeminiLiveSession

    result: dict[str, Any] = {}
    url = vertex.ws_url(location)
    model = vertex.model_path(project, location, MODEL)
    result["url"], result["model"] = url, model
    session = GeminiLiveSession(
        api_key="",
        model=model,
        url=url,
        auth=vertex.auth_headers(credentials),
        system_instruction=(
            "너는 사용자의 개인 비서야. 반말로, 차분하고 낮은 톤으로 짧게 말해."
        ),
        voice_name="Despina",
        tools=_tools(),
    )
    audio = _SilentAudio()
    with tempfile.TemporaryDirectory() as scratch:
        # A scratch data dir, so a spike never writes into the owner's memory.
        companion = Companion(_NullMemory(), data_dir=Path(scratch))
        conversation = VoiceConversation(
            session, audio, companion, opening_text=PROMPT, barge_in=False,
            idle_timeout=12.0,
        )
        original = conversation._one_turn
        boundaries = {"run": 0, "worst": 0}

        async def counted(session_arg: Any, budget: Any) -> bool:
            produced = await original(session_arg, budget)
            if produced:
                boundaries["run"] = 0
            elif audio.played == 0:
                boundaries["run"] += 1
                boundaries["worst"] = max(boundaries["worst"], boundaries["run"])
            return produced

        conversation._one_turn = counted  # type: ignore[method-assign]
        started = time.perf_counter()
        try:
            await conversation.run()
        except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
    result["played_bytes"] = audio.played
    result["first_audio_ms"] = (
        round((audio.first_at - started) * 1000) if audio.first_at else None
    )
    result["boundaries_before_audio"] = boundaries["worst"]
    result["ended"] = conversation.ended
    return result


class _NullMemory:
    """The narrowest `MemoryWriter` a conversation will accept: it records nothing.

    A spike has no business writing to the owner's log, and recall is not what is
    being measured.
    """

    async def record(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def window(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


async def _one_turn(project: str, location: str, credentials: str) -> dict[str, Any]:
    from daemon.voice import vertex
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    result: dict[str, Any] = {"transcripts": [], "audio_bytes": 0, "fetches": 0}
    provider = vertex.auth_headers(credentials)

    def counted() -> dict[str, str]:
        result["fetches"] += 1
        return provider()

    url = vertex.ws_url(location)
    model = vertex.model_path(project, location, MODEL)
    result["url"] = url
    result["model"] = model
    session = GeminiLiveSession(
        api_key="",  # deliberately absent: this endpoint does not take one
        model=model,
        url=url,
        auth=counted,
        voice_name="Despina",
    )
    started = time.perf_counter()
    try:
        async with session:
            result["setup_seconds"] = round(time.perf_counter() - started, 2)
            await session.send_text(PROMPT)
            async for event in session.receive():
                if isinstance(event, Transcript):
                    result["transcripts"].append((event.role, event.text, event.final))
                else:
                    result["audio_bytes"] += len(event)
    except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


async def main() -> int:
    _load_env()
    project = os.environ.get("DAEMON_VERTEX_PROJECT", "").strip()
    location = os.environ.get("DAEMON_VERTEX_LOCATION", "us-central1").strip()
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not project:
        print("DAEMON_VERTEX_PROJECT is not set. Put it in .env and run this again.")
        return 1
    print(f"project: {project}  region: {location}")
    print(f"credentials: {credentials or 'Application Default Credentials'}\n")

    result = await _one_turn(project, location, credentials)
    print(f"uri:   {result['url']}")
    print(f"model: {result['model']}")
    if "error" in result:
        print(f"\n  FAILED: {result['error']}")
        print("  A 404 here is the region; a 1008 naming a model is the path; a 401 "
              "or 403 is the credential.")
        return 1
    print(f"\n  setup (handshake + setupComplete): {result['setup_seconds']}s")
    print(f"  audio: {result['audio_bytes']} bytes of PCM")
    print(f"  credential fetches: {result['fetches']}  <- one per connect attempt")
    print(f"  transcripts ({len(result['transcripts'])}):")
    for role, text, final in result["transcripts"]:
        print(f"    [{role}] final={final} {text!r}")
    if not result["audio_bytes"]:
        print("    no audio: the turn completed without speech - see m0_voice_spike's "
              "docstring 4, the same failure mode on the other endpoint")
    if not result["transcripts"]:
        print("    NONE - memory and persona evolution get nothing from voice mode "
              "without these, whatever the latency is")

    print("\nthe product's own loop, same transport, microphone hearing nothing:")
    loop_result = await _conversation_turn(project, location, credentials)
    if "error" in loop_result:
        print(f"  FAILED: {loop_result['error']}")
        return 1
    first = loop_result["first_audio_ms"]
    print(f"  first audio: {first} ms   played: {loop_result['played_bytes']} bytes")
    print(f"  turn boundaries before that audio: {loop_result['boundaries_before_audio']}"
          "   <- two used to end the session here")
    print(f"  ended: {loop_result['ended']!r}")
    if not loop_result["played_bytes"]:
        print("  NOTHING WAS SPOKEN. The session may be fine and the loop still wrong: "
              "that is exactly how the Vertex transport shipped mute "
              "(daemon/voice/conversation.py, ANSWER_PATIENCE_SECONDS).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
