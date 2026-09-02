"""Does the daemon answer the person and ignore the room? With a real microphone.

The question `tests/` cannot ask and a fake `AudioIO` cannot either. What the owner
reported on 2026-09-02 - "혼자 말하고 혼자 대답하고 난리도 아니야" - was the daemon
answering room sound and its own leaked voice as though they were the owner. Nothing
in the repo could see it: the session was fine, the loop was fine, and the defect
lived in what the microphone carried.

So this drives the product with the room in the loop. A person is played by `say`
through a chosen output device, the daemon answers through the system default, the
microphone is the product's own `VoiceProcessingAudio`, and every turn the server
transcribes is scored against the utterance windows:

    answered        a *fresh burst* of audio after the utterance ended - not the
                    monologue that was already playing, which is what "talking over
                    the person" looks like from the outside
    spurious        a user turn the server heard when nobody was speaking: the room,
                    or the tail of the reply just played

Run it with the resident stopped (it holds the microphone) and a live key or Vertex
credentials in `.env`:

    python3 -m evals.voice_room_spike --sessions 2 --person-device 126

**Read the numbers knowing what the rig cannot do.** Measured while building it: the
same TTS through the laptop's own speakers read 72% speech frames at the microphone
early in a run and 0-2% an hour later, 27 dB quieter, because macOS voice processing
converges on the machine's own output - which is exactly its job. A synthetic person
is therefore a *fading* stand-in for a real one, and a session that hears nothing may
be the rig rather than the product. When the numbers go quiet, re-measure the input
(`--measure-only`) before believing anything about the daemon.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import subprocess
import sys
import time
from typing import Any

import numpy as np

from evals.m0_voice_spike import _load_env

VOICE = "Flo (Korean (South Korea))"
"""A macOS Korean voice. `say -v '?'` lists them; the name must match exactly."""
UTTERANCES = (
    "오늘 기분 어때? 나는 좀 피곤하네.",
    "아무 말이나 좀 길게 해 줄래? 요즘 관심 있는 주제로.",
    "고마워. 그럼 내일 아침에 뭐부터 하면 좋을까?",
)
QUIET_AFTER_ANSWER = 6.0
"""Seconds of silence held after each answer, which is when the daemon answering the
room or its own tail shows up."""
OPENING = "네, 여기 있어요."


async def _measure_input(person_device: str) -> dict[str, float]:
    """What does the product's microphone make of a voice in the room right now?

    The rig's own health check, and the first thing to run when a session hears
    nothing: `speech` here is what the local VAD - and therefore the speech gate -
    will see.
    """
    from daemon.voice.apple_audio import VoiceProcessingAudio
    from daemon.voice.vad import FRAME_BYTES, SileroVad

    audio = VoiceProcessingAudio()
    vad = SileroVad()
    frames: list[tuple[float, float]] = []
    carry = bytearray()

    async def listen() -> None:
        async for block in audio.record():
            carry.extend(block)
            while len(carry) >= FRAME_BYTES:
                frame = bytes(carry[:FRAME_BYTES])
                del carry[:FRAME_BYTES]
                samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
                frames.append((float(np.sqrt((samples**2).mean())), vad.probability(frame)))

    task = asyncio.create_task(listen())
    try:
        await asyncio.sleep(1.5)
        floor = list(frames)
        await asyncio.to_thread(
            subprocess.run,
            ["say", "-v", VOICE, "-a", person_device, UTTERANCES[0]],
            check=False,
        )
        await asyncio.sleep(0.4)
        spoken = frames[len(floor) :]
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        with contextlib.suppress(BaseException):
            await audio.close()

    def db(value: float) -> float:
        return 20 * math.log10(max(value, 1e-6))

    if not spoken:
        return {"speech": 0.0, "peak_dbfs": -120.0, "floor_dbfs": -120.0}
    return {
        "speech": sum(1 for _, p in spoken if p >= 0.5) / len(spoken),
        "peak_dbfs": db(max(r for r, _ in spoken)),
        "floor_dbfs": db(max((r for r, _ in floor), default=1e-6)),
    }


async def _session(settings: Any, person_device: str, gate: bool) -> dict[str, Any]:
    """One conversation, driven by a spoken person, scored."""
    from daemon.app import WS_URL, _build_tools
    from daemon.companion import Companion
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter
    from daemon.voice import vertex
    from daemon.voice.apple_audio import VoiceProcessingAudio
    from daemon.voice.conversation import VoiceConversation
    from daemon.voice.gemini_live import GeminiLiveSession
    from daemon.voice.vad import SileroVad

    store = Store.open(settings.data_dir / "daemon.sqlite3")
    runner, bridge, _ = await _build_tools(settings, store, mode="allowlist")
    companion = Companion(
        FileMemoryWriter(settings.data_dir, store), data_dir=settings.data_dir, tools=runner
    )
    # The session the app would build, assembled from the same settings. Not
    # imported from `run_voice`: that function owns a whole wake-to-report lifecycle
    # and a spike may not have a wake gate, a face or a channel.
    url, model, auth = WS_URL, settings.gemini_live_model, None
    if settings.gemini_live_transport == "vertex":
        url = vertex.ws_url(settings.vertex_location)
        model = vertex.model_path(
            settings.vertex_project, settings.vertex_location, settings.gemini_live_model
        )
        auth = vertex.auth_headers(settings.vertex_credentials_path)
    session = GeminiLiveSession(
        api_key=settings.gemini_api_key,
        model=model,
        url=url,
        auth=auth,
        system_instruction=await companion.persona(),
        tools=companion.specs(origin="owner", surface="voice"),
        start_sensitivity=settings.voice_start_sensitivity,
        end_sensitivity=settings.voice_end_sensitivity,
        prefix_padding_ms=settings.voice_prefix_padding_ms,
        silence_duration_ms=settings.voice_silence_duration_ms,
        voice_name=settings.gemini_live_voice,
    )
    audio = VoiceProcessingAudio()
    conversation = VoiceConversation(
        session,
        audio,
        companion,
        opening_text=OPENING,
        barge_in=settings.voice_barge_in,
        idle_timeout=30.0,
        vad=SileroVad() if gate else None,
    )

    started = time.perf_counter()

    def now() -> float:
        return time.perf_counter() - started

    idle: list[float] = []
    played: list[float] = []
    heard: list[tuple[float, str]] = []
    windows: list[tuple[float, float, str]] = []

    ran_dry = conversation._speaker_ran_dry

    def note_dry() -> None:
        ran_dry()
        idle.append(now())

    conversation._speaker_ran_dry = note_dry  # type: ignore[method-assign]
    play = audio.play

    async def note_play(chunk: bytes) -> None:
        played.append(now())
        await play(chunk)

    audio.play = note_play  # type: ignore[method-assign]
    on_transcript = conversation._on_transcript

    async def note_transcript(session_arg: Any, transcript: Any) -> None:
        if transcript.final and transcript.role == "user":
            heard.append((now(), transcript.text.strip()))
        await on_transcript(session_arg, transcript)

    conversation._on_transcript = note_transcript  # type: ignore[method-assign]

    async def wait_idle(after: float, give_up_after: float) -> float | None:
        deadline = now() + give_up_after
        while now() < deadline:
            fresh = [t for t in idle if t > after]
            if fresh:
                return fresh[-1]
            await asyncio.sleep(0.05)
        return None

    async def person() -> None:
        if await wait_idle(0.0, 40.0) is None:
            return
        await asyncio.sleep(0.8)
        for text in UTTERANCES:
            begun = now()
            await asyncio.to_thread(
                subprocess.run, ["say", "-v", VOICE, "-a", person_device, text], check=False
            )
            windows.append((begun, now(), text))
            if await wait_idle(now(), 20.0) is None:
                await asyncio.sleep(2.0)
                continue
            # Hold the silence until the daemon has stopped for real, which is when
            # it answering the room shows up as a turn nobody asked for.
            settled = now()
            while now() - settled < QUIET_AFTER_ANSWER:
                last = idle[-1]
                await asyncio.sleep(0.2)
                if idle[-1] != last:
                    settled = now()

    driver = asyncio.create_task(person())
    error: str | None = None
    try:
        await asyncio.wait_for(conversation.run(), timeout=180)
    except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
        error = f"{type(exc).__name__}: {exc}"
    finally:
        driver.cancel()
        with contextlib.suppress(BaseException):
            await driver
        with contextlib.suppress(BaseException):
            await audio.close()
        if bridge is not None:
            with contextlib.suppress(BaseException):
                await bridge.close()

    latencies: list[int | None] = []
    for _, ended, _ in windows:
        first = None
        for i, at in enumerate(played):
            # A *fresh* burst: audio already flowing when the person finished is the
            # daemon talking over them, not an answer to them.
            if at > ended and (i == 0 or at - played[i - 1] > 0.5):
                first = at
                break
        latencies.append(None if first is None else round((first - ended) * 1000))

    def inside(at: float) -> bool:
        return any(a - 1.0 <= at <= b + 6.5 for a, b, _ in windows)

    spurious = [text for at, text in heard if not inside(at)]
    return {
        "gate": gate,
        "utterances": len(windows),
        "answered": sum(1 for ms in latencies if ms is not None),
        "latencies_ms": latencies,
        "heard": [text for _, text in heard],
        "spurious": spurious,
        "turns": conversation.turns,
        "interruptions": conversation.interruptions,
        "ended": conversation.ended,
        "error": error,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument(
        "--person-device",
        default="126",
        help="`say -a '?'` id for the speaker the person comes out of. It must NOT be "
        "the system default output - that one is the echo canceller's reference and "
        "the microphone hears nothing of it (measured: 0%% speech frames).",
    )
    parser.add_argument("--no-gate", action="store_true", help="without the speech gate")
    parser.add_argument("--measure-only", action="store_true", help="just check the input")
    args = parser.parse_args()

    _load_env()
    from daemon.config import Settings

    settings = Settings()
    if not settings.voice_enabled:
        print("DAEMON_VOICE_ENABLED is false; nothing to drive.")
        return 1

    print(f"transport {settings.gemini_live_transport} · model {settings.gemini_live_model} "
          f"· voice {settings.gemini_live_voice} · barge-in {settings.voice_barge_in} "
          f"· server silence {settings.voice_silence_duration_ms}")
    health = await _measure_input(args.person_device)
    print(f"the rig's own input: a person reads {100 * health['speech']:.0f}% speech frames, "
          f"peak {health['peak_dbfs']:.1f} dBFS over a {health['floor_dbfs']:.1f} dBFS floor")
    if health["speech"] < 0.2:
        print("  TOO QUIET TO DRIVE ANYTHING. macOS voice processing has converged on this "
              "machine's own speakers, or the device is wedged. Nothing below would be "
              "about the daemon - see the module docstring.")
        if not args.measure_only:
            return 1
    if args.measure_only:
        return 0

    reports = []
    for n in range(1, args.sessions + 1):
        report = await _session(settings, args.person_device, not args.no_gate)
        reports.append(report)
        print(f"\nsession {n}: answered {report['answered']}/{report['utterances']} "
              f"latencies {report['latencies_ms']} ms")
        print(f"  the server heard: {report['heard']}")
        if report["spurious"]:
            print(f"  NOBODY SAID THESE: {report['spurious']}  <- the room, or our own tail")
        if report["error"]:
            print(f"  error: {report['error']}")
        await asyncio.sleep(2.0)

    said = sum(r["utterances"] for r in reports)
    print(f"\nanswered {sum(r['answered'] for r in reports)}/{said} · "
          f"turns nobody asked for {sum(len(r['spurious']) for r in reports)} · "
          f"interruptions {sum(r['interruptions'] for r in reports)}")
    return 0 if said and all(r["answered"] == r["utterances"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
