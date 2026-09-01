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
`idle2` cache, `PcmRing`, `FrameClock`, `ClipClock`, `Renderer`, `Slot`, uvicorn on a
real port, `/face/manifest` and `/face/frames` over a real HTTP connection, and
`SpeechClock` - the same object `daemon/voice/conversation.py` builds, fed with
`loop.time()` stamps and pumped at 25Hz exactly as `_face_pump` does.

The one thing this stands in for is the browser. `Playhead` reproduces the page's own
playhead arithmetic against the same `/face/manifest` anchor, so "is the mouth on the
frame the page is showing" is answerable without one - and `--watch` is still there
because a person looking at a real window is the pass mark and this is not.

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
                # multipart/x-mixed-replace, parsed by the declared Content-Length
                # rather than by scanning for the next boundary: a JPEG is binary and
                # may contain the boundary bytes, and a scanner would truncate exactly
                # the frames this is meant to measure.
                buf = b""
                async for chunk in response.aiter_bytes():
                    buf += chunk
                    while True:
                        head_end = buf.find(b"\r\n\r\n")
                        if head_end < 0:
                            break
                        head = buf[:head_end]
                        length = None
                        for hline in head.split(b"\r\n"):
                            if hline.lower().startswith(b"content-length"):
                                length = int(hline.split(b":")[1])
                        if length is None:
                            if b"--\r\n" in head or head.endswith(b"--"):
                                buf = b""
                            break
                        body_at = head_end + 4
                        if len(buf) < body_at + length:
                            break
                        self.frames.append(buf[body_at : body_at + length])
                        self.arrivals.append(loop.time())
                        buf = buf[body_at + length :]
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
        # Every gap, not just the summary: the failure this feature keeps having is
        # bimodal - a pair arriving together and then a hole - and a median alone
        # cannot tell that from an even cadence at the same average rate.
        "gaps_ms": [round(gap * 1000, 1) for gap in gaps],
    }
    print(
        f"  {name}: {stats['frames']} frames in {stats['seconds']}s = "
        f"{stats['fps']} fps | first at +{stats['first_frame_after']}s | "
        f"gap med {stats['gap_ms_median']}ms p90 {stats['gap_ms_p90']}ms "
        f"max {stats['gap_ms_max']}ms | {stats['kb_median']}KB/frame"
    )
    return stats


Box = tuple[int, int, int, int]


def _tell_tale(driving: np.ndarray, crop_boxes: list[Box]) -> Box:
    """A 48x48 patch that identifies which driving frame a received JPEG was built on.

    Outside every crop box, so the composite leaves it byte-identical to the driving
    clip, and the highest-variance window that qualifies - a patch of still background
    would match all 193 frames equally well and JPEG noise would pick the winner.
    """
    n, height, width = driving.shape[:3]
    sample = np.asarray(driving[:: max(1, n // 24)], dtype=np.float32)
    moving = sample.std(axis=0).mean(axis=2)
    for xs, ys, xe, ye in crop_boxes:                 # never read the mouth region
        moving[max(0, ys - 8) : ye + 8, max(0, xs - 8) : xe + 8] = -1.0
    size, best, at = 48, -1.0, (0, 0)
    for y in range(0, height - size, 24):
        for x in range(0, width - size, 24):
            block = moving[y : y + size, x : x + size]
            if block.min() < 0:
                continue
            if block.mean() > best:
                best, at = float(block.mean()), (x, y)
    return at[0], at[1], at[0] + size, at[1] + size


class Playhead:
    """Where `daemon/static/face.html` has its driving clip, at any wall-clock moment.

    The page's own arithmetic, reproduced here on purpose rather than shared: this
    file measures the page, so a helper imported from the page's own source would only
    prove the renderer agrees with itself. `/face/manifest` hands over the clip's
    position once and the page anchors it to its own clock; so does this, at the same
    instant, over the same loopback connection.

    Anchored ONCE, at construction, and never re-read - which is what makes a drift
    between the two clocks show up here as a growing lag instead of being quietly
    corrected away.
    """

    def __init__(self, *, position: float, at: float, fps: float, frames: int) -> None:
        self._epoch = at - position
        self._fps = fps
        self._period = frames / fps

    def index(self, at: float) -> float:
        """Frames since the anchor, unwrapped and fractional - the fraction is the
        point, because a lag of half a frame is a real answer and rounding it away
        would report a precision this cannot have."""
        return (at - self._epoch) * self._fps

    def position(self, at: float) -> float:
        return (at - self._epoch) % self._period


async def read_playhead(base_url: str, *, fps: float, frames: int) -> Playhead:
    """Anchor a `Playhead` off `/face/manifest`, exactly as the page's `boot()` does -
    stamping the arrival, not the request."""
    loop = asyncio.get_running_loop()
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        manifest = (await client.get("/face/manifest")).json()
    at = loop.time()
    return Playhead(
        position=manifest["lipsync"]["position"], at=at, fps=fps, frames=frames
    )


def report_alignment(
    frames: list[bytes],
    arrivals: list[float],
    *,
    page: Playhead,
    cache_dir: Path,
    fps: float,
) -> dict:
    """Which driving frame each JPEG carries, against the one the page is showing.

    The page runs the driving clip at 1.0x and **never rewinds it** - speech begins on
    the frame that is already up - and lays the rendered frame over it, so at the 180ms
    crossfade in `daemon/static/face.html` the two are dissolved into each other. If
    the renderer is N frames off the playhead, that dissolve is between two poses N
    frames apart in the avatar's own motion, and N is what this measures.

    That "never rewinds" is why this takes a `Playhead` rather than the turn's start.
    It used to model the page as frame 0 at the moment `speaking` arrived, which was
    true of the page it was written against and is the exact assumption the owner saw
    fail: idle rotated between three clips and the driving clip was rewound at every
    turn, so speech began by swapping the head. Both are gone, and the page is now
    wherever the daemon's clip clock says (`render.py:ClipClock`) - so this has to ask
    the same clock, the same way the page does, or it would be measuring the renderer
    against a page that no longer exists.

    Two numbers in one, and which one this is depends on `render.py:DISPLAY_LEAD`.
    With no lead the driving index and the audio index are the same number, so this is
    both the pose offset and the audio-to-mouth lag. With the lead applied this is the
    *residual* pose offset - the thing the crossfade dissolves across, and it should
    read near zero - and the audio lag is this plus `DISPLAY_LEAD`.

    The frame is identified rather than trusted: `_tell_tale` finds a patch the
    composite never touches, and the driving frame whose patch is closest is the one
    this JPEG was built on.

    **Half a frame is this number's noise floor, and one run cannot tell you otherwise.**
    Three runs back to back on the same build, same machine, same wav: phase 1 read
    +0.37, +0.61, +0.41 frames and phase 2 read -0.42, -0.41, -0.71. Nothing about the
    clocks changed between them. What moves is the pipeline's own latency - this is the
    arrival time at the socket against the playhead, so it is `DISPLAY_LEAD` (a
    constant 6) against however long that particular turn actually took, and the first
    utterance of a run is reliably slower than the second. Read a single reading inside
    +-0.7 frames as "the two clocks agree", and go looking only if a run leaves that
    band or if phase 2 walks steadily further from phase 1 across a long session -
    that second one would be drift, which is the failure this anchoring can actually
    have.
    """
    import cv2

    driving = np.load(cache_dir / "frames.npy", mmap_mode="r")
    meta = json.loads((cache_dir / "boxes.json").read_text())
    x1, y1, x2, y2 = _tell_tale(driving, [tuple(b) for b in meta["crop_boxes"]])
    book = np.asarray(driving[:, y1:y2, x1:x2], dtype=np.float32).reshape(
        driving.shape[0], -1
    )
    lags = []
    for payload, at in zip(frames, arrivals, strict=True):
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        patch = image[y1:y2, x1:x2].astype(np.float32).reshape(-1)
        index = int(np.abs(book - patch).mean(axis=1).argmin())
        # Unwrap the clip's cycle: the page's playhead runs on, but the renderer's
        # index is taken modulo the clip length.
        elapsed = page.index(at)
        cycles = round((elapsed - index) / driving.shape[0])
        lags.append(elapsed - (index + cycles * driving.shape[0]))
    stats = {
        "tell_tale_box": [x1, y1, x2, y2],
        "lag_frames_median": round(statistics.median(lags), 2),
        "lag_ms_median": round(statistics.median(lags) / fps * 1000, 1),
        "lag_frames_first": round(lags[0], 2),
        "lag_frames_last": round(lags[-1], 2),
    }
    print(
        f"  pose behind the page's playhead: median {stats['lag_frames_median']} frames "
        f"({stats['lag_ms_median']}ms) | first {stats['lag_frames_first']} | "
        f"last {stats['lag_frames_last']}"
    )
    return stats


IDLE_MARGIN = 3.0
"""Seconds of idle `write_video` puts on either side of the utterance.

Not padding. The thing being judged is the HANDOVER - "nothing about the body moves at
the instant speech begins" - and a video that starts at the first word has cut away the
half of it that shows whether that is true. Three seconds is long enough for an eye to
have settled on the pose before it has to notice whether the pose changed.
"""


def write_video(
    out: Path,
    frames: list[bytes],
    arrivals: list[float],
    began: float,
    *,
    page: Playhead,
    cache_dir: Path,
    fps: float,
    wav: Path | None,
    margin: float = IDLE_MARGIN,
) -> None:
    """The deliverable: what the page would have shown, as one mp4 - idle, speech, idle.

    Built on a wall-clock timeline rather than one frame per received frame, and that
    is the whole point of it being honest. The page runs the driving clip at 1.0x and
    lays whatever frame last arrived over it, so a frame the renderer skipped shows as
    the previous one held a frame longer - not as the video running slow. Writing one
    output frame per received frame would instead stretch a 20fps stream into 24fps of
    playback and make a mouth that drops frames look like a mouth that is late.

    So: output frame *t* is the newest RENDERED frame that had arrived by that instant,
    and outside the utterance it is the driving clip at `Playhead.index` - the same
    free-running playhead the page has, which is what puts the two handovers in this
    video at the frame they really happen on.

    **The handovers are hard cuts here and the page dissolves them over 180ms.** That
    is deliberate and it is the stricter test: a dissolve hides a pose mismatch of a
    few frames, and a cut cannot. If the body moves at either boundary of this video,
    it moves.

    The rendered frame is used whole. An earlier version pasted a crop onto the driving
    frame here, which mirrored a transport that no longer exists - and reproducing that
    paste in the harness would hide the very seam it was written to reveal.
    """
    import cv2

    driving = np.load(cache_dir / "frames.npy", mmap_mode="r")
    height, width = driving.shape[1:3]
    silent = out.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    start = min(began, arrivals[0]) - margin
    total = int((arrivals[-1] + margin - start) * fps) + 1
    nxt, rendered = 0, None
    for index in range(total):
        at = start + index / fps
        while nxt < len(arrivals) and arrivals[nxt] <= at:
            rendered = cv2.imdecode(
                np.frombuffer(frames[nxt], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            nxt += 1
        # Past the last arrival the overlay comes off and the clip is what is left -
        # the second handover, and the one where a mismatch is easiest to see because
        # the mouth stops moving at the same instant.
        clip = rendered is None or at > arrivals[-1]
        writer.write(
            np.array(driving[int(page.index(at)) % driving.shape[0]])
            if clip
            else rendered
        )
    writer.release()
    if wav is None:
        silent.rename(out)
        print(f"  wrote {out} ({len(frames)} frames, no audio track)")
        return
    # Muxed rather than left silent: the judgement being asked for is whether the
    # mouth matches the sound, and a silent video cannot be wrong about that. Delayed
    # by the lead-in, or the sound would start over the idle stretch the lead-in exists
    # to show.
    delay = int(max(0.0, began - start) * 1000)
    done = subprocess.run(
        ["ffmpeg", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
         "-af", f"adelay={delay}|{delay}", "-c:a", "aac", "-shortest", str(out)],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        silent.rename(out)
        print(f"  wrote {out} (ffmpeg mux failed, video only)")
        return
    silent.unlink()
    print(
        f"  wrote {out} ({len(frames)} frames at {fps}fps, audio muxed, "
        f"{margin}s of idle either side)"
    )


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
    # No box any more: a frame is the whole composited image. Assert the manifest does
    # not offer one, because a box reappearing would mean the crop transport is back.
    assert "box" not in manifest["lipsync"], manifest["lipsync"]
    clip_dir = Path(args.data_dir) / "face" / "lipsync" / manifest["lipsync"]["clip"]
    fps = float(json.loads((clip_dir / "boxes.json").read_text())["fps"])

    face = app.state.face
    sink = app.state.face_pcm_sink
    # Anchored the way the page's boot() anchors, and before anything speaks: from here
    # on this harness knows where the page's <video> is without ever asking again, which
    # is what makes a drift between the two clocks visible instead of self-correcting.
    page = await read_playhead(base, fps=fps, frames=len(json.loads(
        (clip_dir / "boxes.json").read_text())["boxes"]))
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
    phase2_frames = list(reader.frames[mark:])
    phase2_arrivals = list(reader.arrivals[mark:])

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

    alignment = {}
    if phase1_frames:
        # Both utterances, and phase 2 is the one that matters most now. The page does
        # not rewind, so a second turn after an idle gap begins wherever the clip has
        # run to - which is the case the old "frame 0 at the turn start" model could
        # not even express. If phase 2 reads worse than phase 1, the two clocks are
        # drifting apart across the gap between them, and that is the failure mode
        # this whole anchoring scheme has.
        print("\nthe overlay against the clip the page is playing under it:")
        print("  phase 1:")
        alignment = report_alignment(
            phase1_frames, phase1_arrivals, page=page, cache_dir=clip_dir, fps=fps
        )
        if phase2_frames:
            print("  phase 2 (a turn boundary, mid-clip):")
            alignment = {
                "phase1": alignment,
                "phase2": report_alignment(
                    phase2_frames, phase2_arrivals, page=page, cache_dir=clip_dir,
                    fps=fps,
                ),
            }

    if phase1_frames:
        print("\nthe thing to look at:")
        write_video(
            args.out / "phase1.mp4",
            phase1_frames,
            phase1_arrivals,
            phase1_began,
            page=page,
            cache_dir=clip_dir,
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
                "alignment": alignment,
                "keepalives": reader.keepalives,
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
