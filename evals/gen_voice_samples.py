"""Generate the admin's voice-preview clips for both providers. Manual; hits the live API.

    GEMINI_API_KEY=... DAEMON_GEMINI_LIVE_MODEL=... \\
    OPENAI_API_KEY=... DAEMON_OPENAI_REALTIME_MODEL=... \\
    python -m evals.gen_voice_samples

Writes daemon/admin/static/voice-samples/<provider>/<voice>.mp3 - one per
GEMINI_LIVE_VOICES under gemini/ and one per OPENAI_REALTIME_VOICES under openai/. The
two passes are independent: each is skipped (with a printed reason) when its own
key/model env is absent, so a Gemini-only or OpenAI-only owner still regenerates that
provider's clips. Needs ffmpeg on PATH for PCM->MP3. Not a test: the suite never uses a
key or network (tests/CLAUDE.md), so this lives in evals/, which may import product code
and does.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

from daemon.config import GEMINI_LIVE_VOICES, OPENAI_REALTIME_VOICES
from daemon.voice.gemini_live import GeminiLiveSession
from daemon.voice.openai_realtime import OpenAIRealtimeSession

PHRASE = "Hi, I'm Daemon. This is what I sound like."
# send_text is a prompt, not verbatim TTS, so instruct the model to read the line.
READ_VERBATIM = "Repeat the user's message back word for word, and say nothing else."
OUT = Path(__file__).resolve().parents[1] / "daemon" / "admin" / "static" / "voice-samples"
PLAYBACK_RATE = 24_000  # Both providers return 24 kHz mono 16-bit PCM.


async def _capture(api_key: str, model: str, voice: str) -> bytes:
    pcm = bytearray()
    async with GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=READ_VERBATIM,
        voice_name=voice,
    ) as session:
        await session.send_text(PHRASE)
        async for item in session.receive():
            if isinstance(item, bytes):
                pcm += item
    return bytes(pcm)


async def _capture_openai(api_key: str, model: str, voice: str) -> bytes:
    pcm = bytearray()
    async with OpenAIRealtimeSession(
        api_key=api_key,
        model=model,
        system_instruction=READ_VERBATIM,
        voice_name=voice,
    ) as session:
        await session.send_text(PHRASE)
        async for item in session.receive():
            if isinstance(item, bytes):
                pcm += item
    return bytes(pcm)


def _to_mp3(pcm: bytes, dest: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav = Path(handle.name)
    try:
        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(PLAYBACK_RATE)
            writer.writeframes(pcm)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-b:a", "64k", str(dest)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        wav.unlink(missing_ok=True)


async def _run_pass(
    provider: str,
    voices: frozenset[str],
    capture: Callable[[str, str, str], Awaitable[bytes]],
    key: str,
    model: str,
) -> int:
    if not key or not model:
        print(f"skip {provider}: set its API key + realtime model", file=sys.stderr)
        return 0
    dest_dir = OUT / provider
    dest_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for voice in sorted(voices):
        try:
            pcm = await capture(key, model, voice)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1
            print(f"! {provider}/{voice}: {exc}", file=sys.stderr)
            continue
        if not pcm:
            failures += 1
            print(f"! {provider}/{voice}: no audio returned", file=sys.stderr)
            continue
        _to_mp3(pcm, dest_dir / f"{voice}.mp3")
        print(f"OK {provider}/{voice} ({len(pcm)} bytes pcm)")
    return failures


async def main() -> int:
    gemini_failures = await _run_pass(
        "gemini",
        GEMINI_LIVE_VOICES,
        _capture,
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("DAEMON_GEMINI_LIVE_MODEL", ""),
    )
    openai_failures = await _run_pass(
        "openai",
        OPENAI_REALTIME_VOICES,
        _capture_openai,
        os.environ.get("OPENAI_API_KEY", ""),
        os.environ.get("DAEMON_OPENAI_REALTIME_MODEL", ""),
    )
    return gemini_failures + openai_failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
