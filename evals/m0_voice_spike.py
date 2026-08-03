"""M0: the voice spike. The only part of M1b that a real key can answer.

Everything in `daemon/voice/` was built from the protocol reference and the
google-genai SDK, and unit-tested against a fake socket. Six things could not be
settled that way, and each one fails at a different moment:

  1. **The model id.** `config.py` deliberately has no default. Two ids the docs
     described - `gemini-live-2.5-flash-preview`, `gemini-2.0-flash-live-001` -
     are already shut down, and `gemini-2.5-flash-native-audio-preview-12-2025`
     is deprecated. A guessed id fails on the first voice turn, not at startup.
  2. **Header auth.** We send `x-goog-api-key` as a header rather than `?key=`,
     so the key stays out of every error that quotes the URI. That is
     SDK-verified, not documented. If the server insists on the query param,
     `_redact` becomes the only thing keeping the key out of logs.
  3. **A wrong key must be permanent.** Close code 1008-for-bad-key is inference
     from SDK source and community reports. If it is classified as transient, a
     revoked key means retrying forever instead of dying loudly.
  4. **A text turn must actually generate.** `realtimeInput.text` is the
     proactive path. Turn end is "derived from user activity" for the realtime
     stream, and whether a bare text message starts generation or needs
     `activityEnd` is not stated anywhere.
  5. **Korean transcript deltas.** Gemini streams transcription as increments
     with no partial/final flag; we accumulate and join with "". Whether that is
     right for Korean needs an eyeball on real output.
  6. **Latency.** PLAN 6.5 bets on hosted native audio *because* a local cascade
     cannot hold the paralinguistics. That bet is worth checking against a number.

Run it once the key is in `.env`:

    python3 -m evals.m0_voice_spike

It sends text and reads the reply, so it needs no microphone. Nothing is written
to the repo, and the key is only ever read from the environment.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

RECOMMENDED_MODEL = "gemini-3.1-flash-live-preview"
"""Google's own skill names this one as current. The spike tries the configured
id first and falls back to this, reporting which worked."""

PROMPT = "안녕? 오늘 좀 피곤한데 짧게 한마디만 해줘."


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


async def _try_model(api_key: str, model: str) -> dict[str, Any]:
    """Open a session, send text, read until the turn ends. Report what happened."""
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    result: dict[str, Any] = {"model": model, "transcripts": [], "audio_bytes": 0}
    started = time.perf_counter()
    session = GeminiLiveSession(api_key=api_key, model=model)
    try:
        async with session:
            result["setup_seconds"] = round(time.perf_counter() - started, 2)
            sent = time.perf_counter()
            await session.send_text(PROMPT)
            first_audio: float | None = None
            async for event in session.receive():
                if isinstance(event, Transcript):
                    result["transcripts"].append((event.role, event.text, event.final))
                else:
                    result["audio_bytes"] += len(event)
                    if first_audio is None:
                        first_audio = time.perf_counter()
            result["time_to_first_audio_ms"] = (
                round((first_audio - sent) * 1000) if first_audio else None
            )
            result["turn_seconds"] = round(time.perf_counter() - sent, 2)
    except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["key_leaked"] = api_key in str(exc) or api_key in repr(exc)
    return result


async def _wrong_key_is_permanent(model: str) -> str:
    """A revoked key must kill the session, not be retried until the heat death."""
    from daemon.voice.gemini_live import GeminiLiveError, GeminiLiveSession

    session = GeminiLiveSession(api_key="AIzaSy-deliberately-invalid", model=model)
    try:
        async with session:
            return "FAIL: an invalid key was accepted"
    except GeminiLiveError as exc:
        # No `.status`: GeminiLiveError does not carry one, and reading it raised
        # an AttributeError from inside this handler - so the one check that was
        # supposed to report a verdict could only ever report a crash.
        verdict = "ok" if exc.permanent else "FAIL: classified as transient, will retry forever"
        return f"{verdict} ({exc})"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: not normalised to GeminiLiveError - {type(exc).__name__}: {exc}"


def _report(result: dict[str, Any]) -> None:
    if "error" in result:
        print(f"  FAILED: {result['error']}")
        print(f"  key present in the error text: {result.get('key_leaked')}  <- must be False")
        return
    print(f"  setup (handshake + setupComplete): {result['setup_seconds']}s")
    print(f"  time to first audio: {result['time_to_first_audio_ms']} ms")
    print(f"  whole turn: {result['turn_seconds']}s, {result['audio_bytes']} bytes of PCM")
    print(f"  transcripts ({len(result['transcripts'])}):")
    for role, text, final in result["transcripts"]:
        print(f"    [{role}] final={final} {text!r}")
    if not result["transcripts"]:
        print("    NONE - memory and persona evolution do not work in voice mode without these")
    if not result["audio_bytes"]:
        print("    no audio: a bare text turn did not trigger generation (see docstring 4)")


async def main() -> int:
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Put it in .env and run this again.")
        return 1

    configured = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip()
    candidates = [m for m in (configured, RECOMMENDED_MODEL) if m]
    seen: set[str] = set()
    candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"candidate models: {', '.join(candidates)}\n")

    working: str | None = None
    for model in candidates:
        print(f"model {model}:")
        result = await _try_model(api_key, model)
        _report(result)
        print()
        if "error" not in result:
            working = model
            break

    if working is None:
        print("No candidate model worked. Check the current Live API model list.")
        return 1

    print(f"invalid-key handling on {working}:")
    print(f"  {await _wrong_key_is_permanent(working)}\n")
    print(f"Set DAEMON_GEMINI_LIVE_MODEL={working} in .env.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
