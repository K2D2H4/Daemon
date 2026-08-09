"""Generate the admin's Gemini Live voice-preview clips. Manual; hits the live API.

    GEMINI_API_KEY=... DAEMON_GEMINI_LIVE_MODEL=... python -m evals.gen_voice_samples

Writes daemon/admin/static/voice-samples/<Voice>.mp3, one per GEMINI_LIVE_VOICES.
Needs ffmpeg on PATH for PCM->MP3. Not a test: the suite never uses a key or network
(tests/CLAUDE.md), so this lives in evals/, which may import product code and does.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from daemon.config import GEMINI_LIVE_VOICES
from daemon.voice.gemini_live import GeminiLiveSession

PHRASE = "Hi, I'm Daemon. This is what I sound like."
# send_text is a prompt, not verbatim TTS, so instruct the model to read the line.
READ_VERBATIM = "Repeat the user's message back word for word, and say nothing else."
OUT = Path(__file__).resolve().parents[1] / "daemon" / "admin" / "static" / "voice-samples"
PLAYBACK_RATE = 24_000  # Gemini Live returns 24 kHz mono 16-bit PCM.


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


async def main() -> int:
    key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "")
    if not key or not model:
        print("set GEMINI_API_KEY and DAEMON_GEMINI_LIVE_MODEL", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for voice in sorted(GEMINI_LIVE_VOICES):
        try:
            pcm = await _capture(key, model, voice)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1
            print(f"! {voice}: {exc}", file=sys.stderr)
            continue
        if not pcm:
            failures += 1
            print(f"! {voice}: no audio returned", file=sys.stderr)
            continue
        _to_mp3(pcm, OUT / f"{voice}.mp3")
        print(f"OK {voice} ({len(pcm)} bytes pcm)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
