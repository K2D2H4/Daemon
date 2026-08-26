"""The face's three endpoints: a page, a stream, and clip bytes.

Read-only, side-effect free, and it carries no conversation. What goes out is an
activity name and a float - never text the daemon said or heard. That is the same
line `daemon/admin/routes.py` draws for the same reason: with no authentication on
127.0.0.1 there is no way to prove owner origin, so the paths that would need it do
not exist here.

Clip names are an allowlist, not a path. `CLIPS` is the whole vocabulary and a name
outside it is a 404 before it is ever joined to a directory (CONTRACTS 13's habit:
the shape is a constant, the value travels as data).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

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
    """Which clips actually exist, so the page can degrade instead of 404-ing."""
    return {"clips": list(available_clips(request.app.state.settings))}


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
                done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_SECONDS)
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
