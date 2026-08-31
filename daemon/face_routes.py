"""The face's five endpoints: a page, a stream, clip bytes, transitions, and lip-sync frames.

Read-only, side-effect free, and it carries no conversation. What goes out is an
activity name and a float - never text the daemon said or heard. That is the same
line `daemon/admin/routes.py` draws for the same reason: with no authentication on
127.0.0.1 there is no way to prove owner origin, so the paths that would need it do
not exist here.

Clip names are an allowlist, not a path. `CLIPS` is the whole vocabulary and a name
outside it is a 404 before it is ever joined to a directory (CONTRACTS 13's habit:
the shape is a constant, the value travels as data).

The lip-sync frames are injected, not imported. `daemon/app.py` builds the renderer
and puts something `LipsyncFrames`-shaped on `app.state.face_frames`; this module
names no part of `daemon/face_lipsync` and imports nothing from it (CONTRACTS 4),
exactly as it takes the `FaceBus` from `app.state.face` rather than building one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from daemon.face import Event, FaceBus, OneShot

router = APIRouter()

PAGE = Path(__file__).parent / "static" / "face.html"

CLIPS: tuple[str, ...] = (
    # Order is preload order, and it is not arbitrary: idle is what a cold page shows
    # first, and speaking is what it needs next and soonest.
    "idle1",
    "idle2",
    "idle3",
    "speaking_soft",
    "speaking_loud",
    "listening",
    "thinking",
    "working",
    "amused",
    "sulky",
    "curious",
    "flourish_arms",
)

_STEMS = frozenset(CLIPS)

KEEPALIVE_SECONDS = 20.0
"""A comment line on an idle stream. Nothing here needs it to stay correct - the
browser reconnects on its own - but a proxy or a sleeping laptop dropping a silent
socket looks exactly like a frozen face, and a colon every 20s is cheaper than
explaining that."""


class LipsyncFrames(Protocol):
    """The lip-sync renderer as `/face` sees it, and nothing more.

    Four members, because four is what the page needs once it stops compositing AND
    stops rewinding. `daemon/app.py` builds whatever satisfies this; nothing here knows
    that a model, a driving-clip cache or a PCM ring exist, which is what keeps this
    file free of `daemon.face_lipsync` (CONTRACTS 4) and what lets the tests drive the
    route with a dozen-line fake.

    **A frame is the WHOLE composited frame, and two earlier versions of this contract
    sent less.** The first sent each frame's crop box, the second their union. Both
    rested on a real measurement - the crop is 3.4x cheaper, 55KB against 180KB - that
    does not apply where this runs: the page is served over loopback to the same
    machine, where 47 Mbit/s costs nothing. And both produced the defect the crop was
    supposed to avoid. A JPEG has no alpha, so laying one over the page's own `<video>`
    has a hard edge however small it is, and the video is never on the same frame as
    the render, so the margin lands on mismatched pixels: a bright rectangle across
    the head, which is what the owner saw. Measured, the model only rewrites 289-314 x
    230-274px of a 1080x1620 frame - so the union was carrying seven times the pixels
    the model touches, all of it head and chest, to create a seam that need not exist.

    The spike's own 1:1 comparison has no seam for one reason: the server composited
    into the frame, and a gaussian-blurred mask leaves no boundary. This transport
    carries that composite whole and the page displays it.
    """

    failed: bool
    """Latched, never cleared. The renderer gives up once and logs once rather than
    retrying at 24Hz, so this is a one-way door and the transport's job is to make it
    visible: the response closes and the page goes back to the pre-rendered clips."""

    clip: str
    """The driving clip these frames were rendered from, by the same stem
    `/face/clips/{name}` serves - `idle2` for the shipped avatar. The page no longer
    composites, so it does not need this to place anything; it needs it to know which
    clip to fall back to when the frames stop, and to report what is on screen."""

    def get(self) -> bytes | None:
        """The newest whole-frame JPEG, or None before the first one exists.

        Latest wins: this returns whatever is there now and never a backlog. A
        transport that queued would show a mouth lagging the sound by however far
        behind the reader is, which is worse than dropping movement. Repeating the
        same bytes between polls is normal rather than a fault: the producer paces one
        frame in here per 41.67ms and this side polls eight times as often.
        """
        ...

    def position(self) -> float:
        """Where the driving clip's playhead is now, in seconds from its start.

        The page plays that clip on a loop and never rewinds it, so speech begins on
        the frame that is already up rather than on frame 0 - which is the whole
        reason this member exists. The renderer composites onto the frame IT believes
        is on screen, so the page has to be showing that frame, and it can only know
        which one that is by being told once and running from there.

        Seconds, not a frame index, because a `<video>`'s `currentTime` is seconds and
        the page must not be made to know the clip's frame rate to use this.
        """
        ...



BOUNDARY = b"daemonface"
"""multipart part separator. Fixed rather than random: it appears in the response's own
Content-Type, so a reader tailing `curl` can grep for it."""

FRAME_POLL_SECONDS = 0.005
"""How often the open stream looks for a new frame.

The slot is a lock-and-overwrite with no way to wait on it, so this side polls - and a
poll needs a floor. An eighth of the 41.67ms frame budget adds at most 5ms to a frame's
age and costs 200 wakeups a second while a stream is open; no floor at all would spin
this loop as fast as the event loop allows, which is the shape that once turned a long
poll into 16,000 requests a second elsewhere in this project. The sleep is on every
turn of the loop, sent frame or not, so the floor holds even while a turn is speaking.
"""


def _lipsync(app: Any) -> LipsyncFrames | None:
    """The injected frame source, or None with the reason left to the caller.

    Absent covers three different installs - switch off, switch on but nothing wired
    (no weights, no driving-clip cache), and a renderer that has since failed - and
    they collapse here because the page does the same thing in all three: play the
    clips. `_lipsync_unavailable` is what tells a human which one it was.
    """
    if not getattr(app.state.settings, "face_lipsync_enabled", False):
        return None
    source: LipsyncFrames | None = getattr(app.state, "face_frames", None)
    if source is None or source.failed:
        return None
    return source


def _lipsync_unavailable(app: Any) -> str:
    """Why there are no frames, in a sentence, naming the switch by its env key.

    A bare 503 sends someone reading their own logs hunting for a renderer that was
    never meant to be running. Three causes, three sentences, and the switch is quoted
    the way it is written in `.env` so the fix is the line you already read.
    """
    if not getattr(app.state.settings, "face_lipsync_enabled", False):
        return "lip-sync is off: set DAEMON_FACE_LIPSYNC_ENABLED=true to render a mouth."
    source: LipsyncFrames | None = getattr(app.state, "face_frames", None)
    if source is None:
        return (
            "lip-sync is on but no renderer is loaded - the weights or the "
            "driving-clip cache under <data_dir>/face/lipsync/ are missing."
        )
    return (
        "the lip-sync renderer failed and has latched off; the face is playing "
        "pre-rendered clips. See the daemon's log for the exception."
    )


def lipsync_manifest(app: Any) -> dict[str, Any] | bool:
    """What `/face/manifest` says about lip-sync: False, or the clip and where it is.

    False is the switch, the missing renderer and the failed renderer at once - see
    `_lipsync`.

    `position` is the page's one and only anchor onto the daemon's clip clock, so it is
    read here, per request, on the same `loop.time()` the renderer is stepping against
    - never cached. A value a second old would put the page a second, 24 frames, away
    from the pose every rendered frame is drawn into.
    """
    source = _lipsync(app)
    if source is None:
        return False
    # No box: the page no longer positions anything, and reporting one it cannot use
    # would be the only remaining trace of the crop transport.
    return {"clip": source.clip, "position": source.position()}


def face_dir(settings: Any) -> Path:
    """`<data_dir>/face`. Never `~/.daemon`: the data dir is `DAEMON_DATA_DIR`,
    default `./data`, and an installed daemon's is wherever it runs from."""
    return Path(settings.data_dir) / "face"


def available_clips(settings: Any) -> tuple[str, ...]:
    """Which clips actually exist on disk, in the order of CLIPS."""
    d = face_dir(settings)
    return tuple(name for name in CLIPS if (d / f"{name}.mp4").is_file())


def _payload(event: Event) -> dict[str, Any]:
    """Serialize a FaceState or OneShot event to its SSE JSON form."""
    if isinstance(event, OneShot):
        return {"kind": "shot", "clip": event.clip}
    return {"kind": "state", "activity": event.activity, "level": round(event.level, 3)}


@router.get("/face", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    """Serve the face page.

    `Cache-Control: no-store`, matching `/face/stream`'s own header - a browser
    caching this shell means a shipped fix needs a hard refresh to actually
    take effect, which is exactly what cost the owner and the coordinator time
    chasing this follow-up. `/face/clips/{name}` and `/face/transitions` keep
    whatever caching FastAPI/Starlette's `FileResponse` already gives them by
    default (no header added here): both are content that changes rarely (a
    new clip, a rebuilt table) rather than something that changes underneath a
    page already open, so a stale cache there costs far less than a stale page
    shell does, and touching them wasn't part of what broke.
    """
    return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@router.get("/face/clips/{name}")
async def clip(name: str, request: Request) -> Response:
    """Serve a clip by name. Rejects unknown names by allowlist before ever
    joining to a path - CONTRACTS 13."""
    if name not in _STEMS:
        return Response(status_code=404)
    path = face_dir(request.app.state.settings) / f"{name}.mp4"
    if not path.is_file():
        return Response(status_code=404)
    return FileResponse(path, media_type="video/mp4")


@router.get("/face/manifest")
async def manifest(request: Request) -> dict[str, Any]:
    """Which clips actually exist, so the page can degrade instead of 404-ing - and
    whether lip-sync is running, so it does not have to guess from a failed request."""
    return {
        "clips": list(available_clips(request.app.state.settings)),
        "lipsync": lipsync_manifest(request.app),
    }


@router.get("/face/transitions")
async def transitions(request: Request) -> Response:
    """Serve `daemon face-transitions`' pose-match table if it has been built,
    404 otherwise - same posture as `/face/clips/{name}`: check, then serve,
    never guess. Task 9 rule 4: the page's own fallback to `currentTime = 0`
    depends on a fresh install (no table yet) getting a clean 404 here, not an
    error - the table is derived from the owner's own clips, so it is never
    checked into the repo.
    """
    path = face_dir(request.app.state.settings) / "transitions.json"
    if not path.is_file():
        return Response(status_code=404)
    return FileResponse(path, media_type="application/json")


@router.get("/face/stream")
async def stream(request: Request) -> StreamingResponse:
    """Server-sent events: the current state (snapshot on open), then all future
    events. Coalesces state changes but queues one-shots. Emits a keepalive comment
    every N seconds so sleeping clients don't close the connection."""
    bus: FaceBus = request.app.state.face
    keepalive = getattr(request.app.state, 'keepalive_seconds', KEEPALIVE_SECONDS)

    async def events() -> AsyncIterator[bytes]:
        agen = bus.subscribe()
        # One `__anext__` that OUTLIVES a keepalive tick, waited on rather than
        # raced. `asyncio.wait_for(agen.__anext__(), ...)` *cancels* the pending
        # call on timeout, and the bus generator is suspended inside
        # `await sub.wake.wait()` (daemon/face.py), so that cancellation runs its
        # `finally` - unsubscribing this client and ending the generator. The
        # next `__anext__` then raised StopAsyncIteration and the stream simply
        # stopped after the first quiet 20 seconds. `EventSource` reconnects, so
        # the face survived it, but a one-shot published in the gap is gone for
        # good: the reconnect's snapshot re-sends state and nothing else.
        pending: asyncio.Future[Event] = asyncio.ensure_future(agen.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({pending}, timeout=keepalive)
                if not done:
                    yield b":\n\n"
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                pending = asyncio.ensure_future(agen.__anext__())
                yield f"data: {json.dumps(_payload(event))}\n\n".encode()
        finally:
            pending.cancel()
            # Awaited, not merely cancelled. Delivering the cancellation is what
            # runs the bus generator's own `finally` and discards this subscriber,
            # and until it lands the generator still counts as running - which
            # makes the `aclose()` below raise `RuntimeError: aclose():
            # asynchronous generator is already running` instead (measured on the
            # client-disconnect path). Once it has landed the generator is
            # finished and `aclose()` is the no-op that says so.
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
            await agen.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/face/frames")
async def frames(request: Request) -> Response:
    """The lip-synced face: one whole composited JPEG per video frame, 24 a second.

    **multipart/x-mixed-replace, which is what 2026-08-26-face-lipsync-design.md
    approved all along.** An earlier version of this route shipped SSE instead, and
    that was the right call for the page it was written against - one that composited
    a crop over its own `<video>` and therefore needed a per-frame arrival time and an
    end-of-stream event, neither of which multipart into an `<img>` provides. The page
    no longer composites, so that requirement is gone, and what is left is the cost:
    SSE has to base64 the payload, which is +33% on the wire (180KB becomes 241KB) and
    hands the main thread a quarter-megabyte JS string 24 times a second to decode.
    An `<img>` fed multipart decodes natively, off that thread, from the raw bytes.

    What the page gives up is knowing, from this route, that a frame has arrived.
    It does not need to: it already subscribes to `/face/stream` for activity, and a
    renderer that latches `failed` ends this response, which fires `error` on the
    `<img>`. That is the fallback signal.

    Latest-wins, never a queue. `daemon/face_lipsync/ring.py:Slot` holds one frame and
    `get()` returns whatever is there; the identity comparison skips the polls between
    two frames, which is most of them - `daemon/app.py:_lipsync_loop` puts one frame in
    there per 41.67ms and this side looks eight times as often. Polling rather than
    waiting on a condition keeps this side free of the renderer's threading (`Slot`
    takes a lock, and this never holds that lock across an await).
    """
    source = _lipsync(request.app)
    if source is None:
        # 503 rather than 404: the route exists, and which of the three reasons it is
        # unavailable belongs in the body where a person reading `curl` output sees it.
        return PlainTextResponse(
            f"face: {_lipsync_unavailable(request.app)}\n", status_code=503
        )
    if source.failed:
        return PlainTextResponse(
            "face: the lip-sync renderer has failed; the face is back on its clips\n",
            status_code=503,
        )

    async def parts() -> AsyncIterator[bytes]:
        last: bytes | None = None
        while not source.failed:
            frame = source.get()
            if frame is not None and frame is not last:
                last = frame
                yield (
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
            await asyncio.sleep(FRAME_POLL_SECONDS)
        # Closing the multipart stream is the signal: the <img> fires `error` and the
        # page reverts to clip playback. No keepalive - unlike SSE there is nothing to
        # keep alive, and a comment frame here would be a malformed part.
        yield b"--" + BOUNDARY + b"--\r\n"

    return StreamingResponse(
        parts(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
