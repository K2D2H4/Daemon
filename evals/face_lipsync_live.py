"""Drive the assembled daemon's lip-sync path and measure what comes off the socket.

By hand, on a machine that has the weights. **Never in CI** - it loads 1.7GB of MLX
UNet and reads a 1GB frame store.

    python3 -m evals.face_lipsync_live --data-dir /path/to/data --wav /path/to/24k.wav

What it is for: `tests/test_face_lipsync_wiring.py` proves the assembly's decisions
against fakes, and `tests/test_face_routes.py` proves the transport against a fake
source. Neither can answer the only question that matters here - **does a real mouth
come out of a real socket at 24fps, and does it track the sound.** The pass mark for
this feature is a person looking at it, so this ends by writing an mp4 that person can
look at.

## What is real and what is not

Real: `create_app` and its lifespan, `_build_lipsync`, the MLX engine, the prepared
`idle2` cache, `PcmRing`, `FrameClock`, `Renderer`, `Slot`, uvicorn on a real port,
`/face/manifest` and `/face/frames` over a real HTTP connection, and `SpeechClock` -
the same object `daemon/voice/conversation.py` builds, fed with `loop.time()` stamps
and pumped at 25Hz exactly as `_face_pump` does.

Not real: Gemini Live and PortAudio. The audio comes from a wav file instead of a
socket, and nothing is played out of a speaker. That seam is deliberate and it is the
only one - `SpeechClock.fed` is where the voice path hands audio over, so everything
downstream of it is the product. What this therefore cannot prove is that a *spoken*
conversation reaches this point; for that, run the resident with the switch on and
talk to it.

## The four phases

1. **one utterance** - frames per second at the socket, inter-frame gaps, payload size.
2. **a turn boundary** - silence, then a second utterance. `PcmRing.feed` re-anchors on
   the discontinuity and `FrameClock` restarts its count; the question is whether
   frames resume rather than stall or replay.
3. **barge-in** - `daemon/voice/conversation.py:_barge_in` replaces `SpeechClock`
   wholesale, which resets `_until` to 0 and makes the next chunk's audible time jump
   *backwards*. The ring is supposed to notice that and drop the cancelled turn's
   audio; if it does not, the mouth keeps mouthing a sentence nobody is hearing.
4. **resident memory**, sampled before the weights load, after each phase and after an
   idle stretch - because `release_lipsync_memory` between utterances is spec section
   7's requirement and the only way to see it working is to watch the number.

Every phase feeds audio *ahead* of playback by `--rate` (default 2x), because a hosted
model generates faster than the speaker drains and the ring holds future audio on
purpose - `FrameClock`'s batch-fill wait is the whole reason that lookahead exists.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import statistics
import subprocess
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

CHUNK_MS = 100.0
"""One handover from the voice session, in ms of audio. Gemini Live's chunks vary;
100ms is inside the range and is what `SpeechClock` sees per call."""

PUMP_HZ = 25.0
"""`daemon/voice/conversation.py:_face_pump`'s own rate, so `speaking` rises and falls
here exactly as it does in a real conversation."""


def _mlx_mb() -> tuple[float, float]:
    """MLX's own (active, cached) allocation in MB.

    RSS alone cannot answer whether `release_lipsync_memory` does anything: the
    process also holds 1.6GB of weights and a 1GB memory-mapped frame store, and
    freeing a buffer back to MLX's allocator need not shrink either. This is the
    number spec section 7's requirement is actually about.
    """
    import mlx.core as mx

    return mx.get_active_memory() / 2**20, mx.get_cache_memory() / 2**20


def _rss_mb(pid: int) -> float:
    """Resident set size in MB, from `ps`. Not `resource.getrusage`, which reports the
    high-water mark and so cannot show memory being handed back."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    try:
        return int(out.stdout.strip()) / 1024.0
    except ValueError:
        return float("nan")


def read_wav(path: Path) -> tuple[bytes, int]:
    """16-bit mono PCM and its rate. Refuses anything else rather than resampling:
    the ring is built at the voice path's 24kHz and `resample_to_whisper` will not
    take another rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: need 16-bit mono, got {handle.getnchannels()}ch "
                f"{handle.getsampwidth() * 8}-bit"
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


class FrameReader:
    """Everything `/face/frames` sent, with the time each one arrived.

    The arrival time is the measurement. A generator that produced 24 frames a second
    into a socket nobody drained would look identical from the renderer's side, which
    is why this is read over a real connection rather than out of the `Slot`.
    """

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.arrivals: list[float] = []
        self.keepalives = 0
        self.closed_at: float | None = None
        self.status: int | None = None

    async def run(self, base_url: str) -> None:
        loop = asyncio.get_running_loop()
        timeout = httpx.Timeout(None)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            async with client.stream("GET", "/face/frames") as response:
                self.status = response.status_code
                if response.status_code != 200:
                    self.closed_at = loop.time()
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        self.frames.append(base64.b64decode(line[5:].strip()))
                        self.arrivals.append(loop.time())
                    elif line.startswith(":"):
                        self.keepalives += 1
        self.closed_at = loop.time()


async def speak(
    clock, pcm: bytes, *, rate: int, ahead: float, chunk_ms: float = CHUNK_MS
) -> tuple[float, float]:
    """Hand `pcm` to `clock` the way a live session does, and report the window it
    will be audible in.

    `at` is `loop.time()` and the chunks go out `ahead`x faster than they play, which
    is what makes `SpeechClock._until` run into the future - the state the ring's
    lookahead exists for. Returns `(first_audible, last_audible)`.
    """
    loop = asyncio.get_running_loop()
    step = int(rate * chunk_ms / 1000.0) * 2
    first = loop.time()
    for offset in range(0, len(pcm), step):
        clock.fed(pcm[offset : offset + step], loop.time())
        await asyncio.sleep(chunk_ms / 1000.0 / ahead)
    return first, first + len(pcm) / (rate * 2)


async def pump_forever(clock) -> None:
    """`_face_pump`, verbatim in effect: the falling edge and the level, at 25Hz."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(1.0 / PUMP_HZ)
        clock.pump(loop.time())


def report_phase(name: str, reader: FrameReader, mark: int, began: float) -> dict:
    """Frames per second at the socket, and the gaps that average hides."""
    frames = reader.frames[mark:]
    arrivals = reader.arrivals[mark:]
    if not arrivals:
        print(f"  {name}: NO FRAMES")
        return {"name": name, "frames": 0}
    span = arrivals[-1] - began
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
    sizes = [len(frame) for frame in frames]
    stats = {
        "name": name,
        "frames": len(frames),
        "seconds": round(span, 2),
        "fps": round(len(frames) / span, 2) if span > 0 else 0.0,
        "first_frame_after": round(arrivals[0] - began, 3),
        "gap_ms_median": round(statistics.median(gaps) * 1000, 1) if gaps else None,
        "gap_ms_p90": (
            round(sorted(gaps)[int(len(gaps) * 0.9)] * 1000, 1) if len(gaps) > 4 else None
        ),
        "gap_ms_max": round(max(gaps) * 1000, 1) if gaps else None,
        "kb_median": round(statistics.median(sizes) / 1024, 1),
    }
    print(
        f"  {name}: {stats['frames']} frames in {stats['seconds']}s = "
        f"{stats['fps']} fps | first at +{stats['first_frame_after']}s | "
        f"gap med {stats['gap_ms_median']}ms p90 {stats['gap_ms_p90']}ms "
        f"max {stats['gap_ms_max']}ms | {stats['kb_median']}KB/frame"
    )
    return stats


def write_video(
    out: Path,
    frames: list[bytes],
    arrivals: list[float],
    began: float,
    *,
    cache_dir: Path,
    box: tuple[int, int, int, int],
    fps: float,
    wav: Path | None,
) -> None:
    """The deliverable: what the page would have shown, as one mp4.

    Built on a wall-clock timeline rather than one frame per received frame, and that
    is the whole point of it being honest. The page runs the driving clip at 1.0x from
    the start of the turn and lays whatever crop last arrived over it, so a frame the
    renderer skipped shows as the previous crop held one frame longer - not as the
    video running slow. Writing one output frame per received frame would instead
    stretch a 20fps stream into 24fps of playback and make a mouth that drops frames
    look like a mouth that is late.

    So: output frame *t* is driving frame *t* with the newest crop that had arrived by
    `began + t/fps`. The audio muxed underneath starts at `began` too, which is what
    makes "does the mouth match the sound" a question this file can be wrong about.
    """
    import cv2

    driving = np.load(cache_dir / "frames.npy", mmap_mode="r")
    height, width = driving.shape[1:3]
    x1, y1, x2, y2 = box
    silent = out.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    total = int((arrivals[-1] - began) * fps) + 1
    nxt, crop = 0, None
    for index in range(total):
        at = began + index / fps
        while nxt < len(arrivals) and arrivals[nxt] <= at:
            crop = cv2.imdecode(np.frombuffer(frames[nxt], dtype=np.uint8), cv2.IMREAD_COLOR)
            nxt += 1
        frame = np.array(driving[index % driving.shape[0]])
        if crop is not None:
            frame[y1:y2, x1:x2] = crop
        writer.write(frame)
    writer.release()
    if wav is None:
        silent.rename(out)
        print(f"  wrote {out} ({len(frames)} frames, no audio track)")
        return
    # Muxed rather than left silent: the judgement being asked for is whether the
    # mouth matches the sound, and a silent video cannot be wrong about that.
    done = subprocess.run(
        ["ffmpeg", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
         "-c:a", "aac", "-shortest", str(out)],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        silent.rename(out)
        print(f"  wrote {out} (ffmpeg mux failed, video only)")
        return
    silent.unlink()
    print(f"  wrote {out} ({len(frames)} frames at {fps}fps, audio muxed)")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True, help="16-bit mono 24kHz")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--rate", type=float, default=2.0, help="x realtime to feed at")
    parser.add_argument("--seconds", type=float, default=8.0, help="audio per utterance")
    parser.add_argument("--out", type=Path, default=Path("/tmp/face_lipsync_live"))
    parser.add_argument("--idle", type=float, default=6.0, help="idle stretch to watch")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="after the phases, keep the server up and keep speaking for N seconds so "
        "a browser can watch /face. The pass mark is a person looking at it, and this "
        "is the only mode that lets them",
    )
    args = parser.parse_args()

    import uvicorn

    from daemon.app import create_app
    from daemon.config import Settings
    from daemon.face import SpeechClock

    args.out.mkdir(parents=True, exist_ok=True)
    pcm, rate = read_wav(args.wav)
    pcm = pcm[: int(rate * 2 * args.seconds)]
    pid = os.getpid()
    before_mb = _rss_mb(pid)

    settings = Settings(
        _env_file=None,
        provider="ollama",
        data_dir=args.data_dir,
        face_lipsync_enabled=True,
        # Everything that would open a socket or a microphone, off. What is under test
        # is the render path, and a wake gate fighting the owner's own resident for
        # CoreAudio would only add a failure that has nothing to do with it.
        voice_enabled=False,
        wake_enabled=False,
        proactive_enabled=False,
        port=args.port,
    )
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    )
    serving = asyncio.create_task(server.serve())
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(600):
        await asyncio.sleep(0.1)
        if server.started:
            break
    else:
        raise SystemExit("uvicorn never started")
    loaded_mb = _rss_mb(pid)

    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        manifest = (await client.get("/face/manifest")).json()
    print(f"\n/face/manifest lipsync = {json.dumps(manifest['lipsync'])}")
    if manifest["lipsync"] is False:
        body = (await httpx.AsyncClient(base_url=base).get("/face/frames")).text
        print(f"/face/frames says: {body.strip()}")
        raise SystemExit(
            "lip-sync did not assemble, so there is nothing to measure. This is a "
            "'could not check' result, not a pass."
        )
    box = tuple(manifest["lipsync"]["box"])
    clip_dir = Path(args.data_dir) / "face" / "lipsync" / manifest["lipsync"]["clip"]
    fps = float(json.loads((clip_dir / "boxes.json").read_text())["fps"])

    face = app.state.face
    sink = app.state.face_pcm_sink
    reader = FrameReader()
    reading = asyncio.create_task(reader.run(base))
    await asyncio.sleep(0.3)

    results = []
    loop = asyncio.get_running_loop()
    clock = SpeechClock(face, sample_rate=rate, bytes_per_frame=2, pcm_sink=sink)
    pumping = asyncio.create_task(pump_forever(clock))

    print("\nphase 1 - one utterance")
    mark, began = len(reader.frames), loop.time()
    await speak(clock, pcm, rate=rate, ahead=args.rate)
    # Wait out the audio that was handed over ahead of playback: `speaking` stays true
    # until the last chunk is heard, and those are exactly the frames a run that
    # stopped at the last `fed()` would throw away.
    await asyncio.sleep(args.seconds / args.rate + 0.5)
    results.append(report_phase("phase 1", reader, mark, began))
    phase1_frames = list(reader.frames[mark:])
    phase1_arrivals = list(reader.arrivals[mark:])
    phase1_began = began
    spoke_mb = _rss_mb(pid)

    print(f"\nidle {args.idle}s - the face should stop and hand memory back")
    idle_mark = len(reader.frames)
    spoke_active, spoke_cache = _mlx_mb()
    await asyncio.sleep(args.idle)
    print(
        f"  frames while idle: {len(reader.frames) - idle_mark} "
        f"(expect 0) | activity = {face.state.activity}"
    )
    idle_active, idle_cache = _mlx_mb()
    print(
        f"  MLX MB: active {spoke_active:.0f} -> {idle_active:.0f}, "
        f"cached {spoke_cache:.0f} -> {idle_cache:.0f}"
    )
    idle_mb = _rss_mb(pid)

    print("\nphase 2 - a second utterance after a gap (the turn boundary)")
    mark, began = len(reader.frames), loop.time()
    await speak(clock, pcm, rate=rate, ahead=args.rate)
    await asyncio.sleep(args.seconds / args.rate + 0.5)
    results.append(report_phase("phase 2", reader, mark, began))

    print("\nphase 3 - barge-in: a fresh SpeechClock mid-utterance")
    mark, began = len(reader.frames), loop.time()
    speaking = asyncio.create_task(speak(clock, pcm, rate=rate, ahead=args.rate))
    await asyncio.sleep(min(1.5, args.seconds / args.rate / 2))
    speaking.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await speaking
    # What `_barge_in` does: the clock is replaced, so `_until` is 0 again and the next
    # chunk's audible time jumps *backwards*. `PcmRing.feed` has to notice.
    pumping.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pumping
    clock = SpeechClock(face, sample_rate=rate, bytes_per_frame=2, pcm_sink=sink)
    pumping = asyncio.create_task(pump_forever(clock))
    await speak(clock, pcm, rate=rate, ahead=args.rate)
    await asyncio.sleep(args.seconds / args.rate + 0.5)
    results.append(report_phase("phase 3", reader, mark, began))

    if args.watch:
        # Nothing about this is measurement: it holds the whole assembled thing open,
        # speaking on a loop with a gap between utterances, so the face page can be
        # opened and looked at. The gap matters - the switch's whole justification is
        # that the mouth has to be judged against the v1 clips, and the falling edge
        # is where the two are side by side.
        print(f"\nwatch: open http://127.0.0.1:{args.port}/face - speaking for {args.watch}s")
        until = loop.time() + args.watch
        while loop.time() < until:
            await speak(clock, pcm, rate=rate, ahead=args.rate)
            await asyncio.sleep(args.seconds / args.rate + 2.0)
        print("watch: done")

    pumping.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pumping
    reading.cancel()
    with contextlib.suppress(asyncio.CancelledError, httpx.ReadError):
        await reading

    print(
        f"\nresident MB: {before_mb:.0f} before the switch -> {loaded_mb:.0f} loaded -> "
        f"{spoke_mb:.0f} after speaking -> {idle_mb:.0f} after {args.idle}s idle"
    )
    print(f"keepalive comments on the frame stream: {reader.keepalives}")

    if phase1_frames:
        print("\nthe thing to look at:")
        write_video(
            args.out / "phase1.mp4",
            phase1_frames,
            phase1_arrivals,
            phase1_began,
            cache_dir=clip_dir,
            box=box,
            fps=fps,
            wav=args.wav,
        )
        strip = args.out / "strip.jpg"
        _write_strip(strip, phase1_frames, fps=fps)
        print(f"  wrote {strip}")

    (args.out / "run.json").write_text(
        json.dumps(
            {
                "phases": results,
                "rss_mb": {
                    "before": before_mb,
                    "loaded": loaded_mb,
                    "spoke": spoke_mb,
                    "idle": idle_mb,
                },
                "mlx_mb": {
                    "active_after_speaking": round(spoke_active, 1),
                    "cached_after_speaking": round(spoke_cache, 1),
                    "active_after_idle": round(idle_active, 1),
                    "cached_after_idle": round(idle_cache, 1),
                },
                "keepalives": reader.keepalives,
                "box": list(box),
                "fps": fps,
                "feed_rate_x_realtime": args.rate,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    server.should_exit = True
    with contextlib.suppress(Exception):
        await asyncio.wait_for(serving, 10.0)
    return 0


def _write_strip(path: Path, frames: list[bytes], *, fps: float) -> None:
    """Eight crops a third of a second apart, side by side.

    A strip answers a different question from the video: whether the mouth *shapes*
    differ frame to frame. A blurred mouth that vibrates and a mouth that articulates
    look similar in motion and nothing alike in a row.
    """
    import cv2

    step = max(1, int(fps / 3))
    picked = [frames[i] for i in range(0, len(frames), step)][:8]
    crops = [
        cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        for payload in picked
    ]
    if not crops:
        return
    cv2.imwrite(str(path), np.hstack(crops), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
