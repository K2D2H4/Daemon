"""The face's four endpoints: a page, a stream, clip bytes, and lip-sync frames.

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
import base64
import json
from collections.abc import AsyncIterator
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
    """The lip-sync renderer as `/face/frames` sees it, and nothing more.

    Four members, because four is what a transport needs. `daemon/app.py` builds
    whatever satisfies this; nothing here knows that a model, a driving-clip cache or
    a PCM ring exist, which is what keeps this file free of `daemon.face_lipsync`
    (CONTRACTS 4) and what lets the tests drive the route with a dozen-line fake.

    **Deliberately no frame counter and no "changed since" token**, even though either
    would make a per-frame-fetch transport possible. `daemon/face_lipsync/ring.py:Slot`
    is `put` and `get`, and nothing else; a protocol asking for more than the object it
    describes actually offers would be satisfiable only by an adapter invented to
    satisfy it. That absence is a real constraint, and it is what settles the transport
    in `frames()` below.
    """

    failed: bool
    """Latched, never cleared. The renderer gives up once and logs once rather than
    retrying at 24Hz, so this is a one-way door and the transport's job is to make it
    visible: the response closes and the page goes back to the pre-rendered clips."""

    clip: str
    """The driving clip these crops belong on, by the same stem `/face/clips/{name}`
    serves - `idle2` for the shipped avatar. The page has to lay the crop over *this*
    clip, not over the v1 speaking clip, or the head under the mouth is a different
    head."""

    box: tuple[int, int, int, int]
    """Where the crop sits in the driving clip's own pixels, `(x1, y1, x2, y2)` - the
    same corner convention as the renderer's own boxes, so the wiring passes a crop box
    straight through instead of converting it into a width and a height on the way past.

    **One box for the whole clip, and the per-frame ones really do differ.** MuseTalk
    derives the blend region from each frame's own face box, so it breathes: measured
    over `idle1.mp4`'s 193 frames it ranges 572 to 608 px square
    (`evals/face_lipsync_prepare.py`). This is therefore their *union*, not
    `crop_boxes[0]`. A rectangle that moved frame to frame would be a rectangle the
    page has to re-place frame to frame, which reads as a wobble around the jaw - the
    artefact 2026-08-25-face-design.md spent nine attempts chasing. The union costs a
    few hundred pixels of JPEG and buys a still seam.

    Two things the wiring owns, written here because this is the contract they have to
    meet and neither is true of `daemon/face_lipsync` today:

    - the union has to be padded to a constant size, since a JPEG whose *pixel*
      dimensions changed per frame would resize the page's overlay per frame;
    - `render.py:Renderer._render` encodes `out`, which is the whole composited
      1080x1620 frame. To satisfy `get()` below it must encode `out[y1:y2, x1:x2]` at
      this box instead. That file is outside this task's scope, so it is named rather
      than changed."""

    def get(self) -> bytes | None:
        """The newest JPEG **of `box` only**, or None before the first one exists.

        Latest wins: this returns whatever is there now and never a backlog. A
        transport that queued would show a mouth that lags the sound by however far
        behind the reader is, which is worse than dropping movement.

        Crop and not the whole frame, because the page already has the driving clip
        decoded and the other 87% of those pixels are pixels it can draw itself.
        Re-measured here on frame 40 of the shipped `idle2.mp4` (1080x1620, 24fps),
        JPEG q85, against 2026-08-25-face-design.md's own 174KB/52KB and 2.43/0.46ms:

            whole frame  1080x1620   2.70ms   180KB   35.5 Mbit/s
            crop box       590x590   0.54ms    55KB   10.8 Mbit/s

        The encode is on the CPU side, so the 2.2ms it saves is not taken out of the
        2.5% of headroom the GPU budget has left - but it is still 2.2ms, and the bytes
        are 3.3x."""
        ...


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
    """What `/face/manifest` says about lip-sync: False, or where to put the crops.

    False is the switch, the missing renderer and the failed renderer at once - see
    `_lipsync`. The geometry rides here rather than on the frames themselves because
    `<img>` cannot read a multipart part's headers; it is static anyway (`box`), and
    the page re-asks after every spoken turn, which is what makes the switch a live
    toggle instead of a restart.
    """
    source = _lipsync(app)
    if source is None:
        return False
    return {"clip": source.clip, "box": list(source.box)}


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
    """Serve the face page."""
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


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


@router.get("/face/stream")
async def stream(request: Request) -> StreamingResponse:
    """Server-sent events: the current state (snapshot on open), then all future
    events. Coalesces state changes but queues one-shots. Emits a keepalive comment
    every N seconds so sleeping clients don't close the connection."""
    bus: FaceBus = request.app.state.face
    keepalive = getattr(request.app.state, 'keepalive_seconds', KEEPALIVE_SECONDS)

    async def events() -> AsyncIterator[bytes]:
        agen = bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(agen.__anext__(), keepalive)
                except TimeoutError:
                    yield b":\n\n"
                    continue
                except StopAsyncIteration:
                    return
                yield f"data: {json.dumps(_payload(event))}\n\n".encode()
        finally:
            await agen.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/face/frames")
async def frames(request: Request) -> Response:
    """The lip-synced mouth: one crop-box JPEG per video frame, 24 a second, over SSE.

    **This is not the transport 2026-08-26-face-lipsync-design.md approved**, and the
    departure is deliberate rather than an oversight. That table says "서버가 완성
    프레임, MJPEG" - whole frames, multipart. Both halves changed, for different
    reasons, and each is named here so nobody has to reverse-engineer which:

    - *whole frames -> the crop box.* The design deferred crops because a patch needs
      browser->daemon feedback, which §2 deliberately removed. It is worth the deferral
      no longer: 180KB against 55KB per frame, 35.5 against 10.8 Mbit/s (see
      `LipsyncFrames.get`). The design's own render-loop budget in §3 already quotes
      the 0.46ms *crop* encode, so this is the direction that makes that document
      internally consistent, not the one that breaks it. What it costs is named under
      "the seam" below.
    - *multipart -> SSE.* Argued next.

    **Why SSE.** The deciding constraint is not bandwidth - this is loopback - it is
    that the renderer can latch `failed` mid-utterance, after which it publishes
    nothing, and the page has to notice and go back to the clips rather than hold a
    frozen mouth over a speaking face. Noticing needs two signals in the page: a
    per-frame arrival time, and an end-of-stream. `EventSource` gives both for free
    (`onmessage`, then `onerror` with `readyState === CLOSED` once the retry lands on
    the 503 below), it is the idiom `/face/stream` and `face.html` already use, and the
    page still just assigns `img.src`, so the browser keeps doing the decode.

    It pays base64, and that is the honest cost: 55KB becomes 73KB, 10.8 becomes
    14.4 Mbit/s, and `b64encode` costs 0.039ms per frame (measured, same frame as
    `LipsyncFrames.get`). Against the approved full-frame MJPEG's 35.5 Mbit/s it is
    still 2.5x less traffic, so the crop pays for the framing twice over.

    **What was rejected.**

    - *`multipart/x-mixed-replace` into `<img>`* - fewer bytes and the browser frames
      it, but no portable per-frame or end-of-stream event: Chrome fires `load` once
      for the whole stream and a clean end fires nothing at all. That is exactly the
      signal the `failed` path needs, so buying 17KB a frame on loopback would cost
      the failure mode. Parsing multipart out of `fetch()` by hand instead would get
      the signal back and put a framing parser on the main thread to do it.
    - *One fetch per frame* - the tidiest client and the only one whose latest-wins is
      in the protocol rather than in this generator. It needs to ask "anything newer
      than what I hold?", and `LipsyncFrames` cannot answer: `Slot` is `put`/`get`
      with no counter (see the protocol's note), so a poll either re-serves the 55KB
      it already has 24 times a second or the protocol grows a field the object it
      describes does not have. Note the cost that is *not* a reason: `daemon log`
      already filters successful `uvicorn.access` GET lines (`cli.py`'s `_LOG_NOISE`),
      so 24 requests a second would not bury the log the way an earlier draft of this
      comment claimed - the log *file* still grows, which is a smaller objection.
    - *A websocket* - nothing here is bidirectional, and it would be a dependency for
      a stream that flows one way.
    - *Folding frames into `/face/stream`* - one channel, no second connection. But
      that stream carries activity and level, exists switch or no switch, and must not
      have a state change queued behind a 73KB frame. Different lifetime, different
      payload size, different channel.

    **The seam, and its one real limitation.** The page holds the driving clip under
    the crop, so if its playhead is not on the frame the renderer drew, the pose inside
    the box does not match the pose outside it. `FrameClock` restarts at frame 0 on
    every re-anchor and `render.py` indexes `% len(cache.boxes)`, so the convention is:
    the page restarts the driving clip at the start of each spoken turn and both run at
    1.0x off their own clocks. That is why `face.html` must not modulate the driving
    clip's `playbackRate` the way the v1 mouth does. It does *not* cure drift over a
    long turn - `<video>` at 1.0x and the render loop's wall clock are not the same
    clock - and curing it needs the driving frame index in the payload, which needs the
    source to expose one. Not added speculatively; named so the fix is obvious if the
    seam turns out to be visible on a real window, which is the only place it can be.

    Latest-wins survives a slow reader: the generator re-reads the slot each turn, so a
    page that falls behind loses the frames in between rather than accumulating them,
    and the renderer never blocks on this side (it overwrites the slot under its own
    lock, and this never holds that lock across an await).
    """
    source = _lipsync(request.app)
    if source is None:
        # 503, not 404: the route exists and will serve frames the moment the renderer
        # does. The body says which of the three reasons it is - a bare status here
        # would be the `HTTP 409` that cost someone hours in `channels/telegram.py`.
        # It is also load-bearing for the page: a non-200 makes `EventSource` fail the
        # connection permanently instead of retrying, which is how a latched `failed`
        # becomes a definite answer rather than a reconnect loop.
        return PlainTextResponse(
            f"face: {_lipsync_unavailable(request.app)}\n", status_code=503
        )

    # Same override `/face/stream` above takes, so a test can watch the keepalive path
    # without waiting 20s for it. Worth correcting the neighbouring comment while
    # relying on it: `httpx.ASGITransport` (0.28.1) does not merely hold the first
    # chunk, it has no streaming path at all and runs the ASGI app to completion, so a
    # short keepalive is not what unblocks such a test - ending the response is
    # (`tests/test_face_routes.py:_transcript`).
    keepalive = getattr(request.app.state, "keepalive_seconds", KEEPALIVE_SECONDS)

    async def events() -> AsyncIterator[bytes]:
        last: bytes | None = None
        quiet = 0.0
        # No `is_disconnected()` check: `StreamingResponse` already runs
        # `listen_for_disconnect` beside this generator and cancels it, which is what
        # `/face/stream` above relies on too. Peeking at `receive` from in here would
        # put a second consumer on the one channel that call is already blocked on.
        while not source.failed:
            frame = source.get()
            if frame is not None and frame is not last:
                # Identity, not equality: `Slot.get` hands back the same object until
                # the renderer overwrites it, and `last` holds a reference so the
                # address cannot be recycled underneath. Comparing 55KB instead would
                # also call two consecutive identical mouths "unchanged", which they
                # are not - a still mouth is a frame the page should still receive.
                last = frame
                quiet = 0.0
                yield b"data: " + base64.b64encode(frame) + b"\n\n"
            else:
                # A gap of a tick or two is the normal state, not a fault: a step
                # covers `BATCH` frames and cannot start until the last of them has
                # its audio, so `FrameClock.due` answers None on some ticks by design
                # (daemon/face_lipsync/render.py). The page's staleness threshold, not
                # this loop, is what has to tolerate that.
                quiet += FRAME_POLL_SECONDS
                if quiet >= keepalive:
                    # A comment line, so it never reaches the page's `onmessage` and
                    # cannot be mistaken for a frame. Same reason `/face/stream` has
                    # one, and more pressing here: silence is this stream's resting
                    # state, so a proxy or a sleeping laptop dropping the socket would
                    # otherwise leave a face that looks fine and never moves again.
                    quiet = 0.0
                    yield b":\n\n"
            await asyncio.sleep(FRAME_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
