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

Run it with credentials for a project that has Vertex AI enabled:

    DAEMON_VERTEX_PROJECT=<project> python3 -m evals.vertex_live_spike

It sends text and reads the reply, so it needs no microphone. Latency is
deliberately not reported here - a text turn skips the server's end-of-speech
wait, which is most of what an owner feels, and the numbers that matter were taken
through the audio path (see the design doc). Nothing is written to the repo.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from evals.m0_voice_spike import _load_env

MODEL = "gemini-live-2.5-flash-native-audio"
"""The reason this transport exists: the API-key endpoint closes 1008 for it."""

PROMPT = "안녕? 오늘 좀 피곤한데 짧게 한마디만 해줘."


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
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
