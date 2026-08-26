"""The face routes: a page, a stream, clip bytes, and lip-sync frames.

The stream is the interesting one. It carries three things on one channel and its
first line must be a snapshot rather than a change, because a reconnect in the middle
of a reply otherwise shows an idle face over a speaking daemon.

The frames are the other interesting one, and everything worth asserting about them is
what the page is told when there is nothing to send: switch off, nothing wired, a
renderer that has failed, and a renderer that simply has not drawn yet are four
different sentences and only one of them is an error the owner should care about. All
four have to leave the page able to fall back to the clips, so each test below ends by
checking the signal the page actually keys off.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daemon.config import Settings
from daemon.face import FaceBus
from daemon.face_routes import BOUNDARY, CLIPS, available_clips, face_dir, router


def _settings(tmp_path, **kw):
    """`provider="ollama"` needs no key, and `_env_file=None` keeps the worktree's own
    `.env` out - the same base `tests/test_admin.py:_settings` uses. A bare
    `Settings(data_dir=...)` raises ConfigError: a hosted task is routed with no
    provider set."""
    kw.setdefault("provider", "ollama")
    return Settings(_env_file=None, data_dir=tmp_path, **kw)


@pytest.fixture
def app(tmp_path):
    api = FastAPI()
    api.state.settings = _settings(tmp_path)
    api.state.face = FaceBus()
    api.include_router(router)
    return api


def _clip(settings, name: str) -> None:
    d = face_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")


def test_face_dir_is_under_the_data_dir_not_the_home_dir(tmp_path):
    assert face_dir(_settings(tmp_path)) == tmp_path / "face"


def test_the_page_renders_with_no_assets_at_all(app):
    r = TestClient(app).get("/face")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_available_clips_reports_only_what_exists(app):
    settings = app.state.settings
    assert available_clips(settings) == ()
    _clip(settings, "idle1")
    _clip(settings, "not_a_face_clip")
    assert available_clips(settings) == ("idle1",)


def test_clip_bytes_are_served_and_unknown_names_are_refused(app):
    _clip(app.state.settings, "idle1")
    client = TestClient(app)
    assert client.get("/face/clips/idle1").status_code == 200
    assert client.get("/face/clips/missing").status_code == 404
    # Not an allowlisted stem: refused before it ever becomes a path.
    assert client.get("/face/clips/..%2F..%2Fdaemon.sqlite3").status_code == 404


def test_manifest_lists_available_clips(app):
    settings = app.state.settings
    r = TestClient(app).get("/face/manifest")
    assert r.status_code == 200
    assert r.json() == {"clips": [], "lipsync": False}
    _clip(settings, "idle1")
    _clip(settings, "speaking_soft")
    r = TestClient(app).get("/face/manifest")
    assert r.json() == {"clips": ["idle1", "speaking_soft"], "lipsync": False}


async def test_the_stream_opens_with_a_snapshot(app):
    """Stream must open with a snapshot, not wait for the first event.

    Override keepalive to be fast: httpx.ASGITransport buffers the first response
    chunk until a subsequent chunk arrives (the keepalive), so production's 20s
    default becomes a test stall. Tests must set this small to exercise the
    snapshot path without blocking.
    """
    app.state.face.set_activity("thinking")
    app.state.keepalive_seconds = 0.1
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://face") as client:
        async with client.stream("GET", "/face/stream") as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    assert json.loads(line[5:]) == {
                        "kind": "state",
                        "activity": "thinking",
                        "level": 0.0,
                    }
                    break


async def test_the_stream_carries_a_one_shot(app):
    """Published one-shots appear on the stream as separate events.

    Override keepalive to be fast: see test_the_stream_opens_with_a_snapshot.
    """
    app.state.keepalive_seconds = 0.1
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://face") as client:
        async with client.stream("GET", "/face/stream") as r:
            assert r.status_code == 200
            lines = r.aiter_lines()
            # Read the snapshot.
            async for line in lines:
                if line.startswith("data:"):
                    break
            # Publish a one-shot in the middle of the stream.
            app.state.face.one_shot("amused")
            # Read the one-shot event.
            async for line in lines:
                if line.startswith("data:"):
                    assert json.loads(line[5:]) == {"kind": "shot", "clip": "amused"}
                    break


def test_clips_lists_every_stem_the_page_expects(app):
    assert set(CLIPS) == {
        "idle1",
        "idle2",
        "idle3",
        "listening",
        "thinking",
        "working",
        "speaking_soft",
        "speaking_loud",
        "amused",
        "sulky",
        "curious",
        "flourish_arms",
    }


# --- lip-sync frames -------------------------------------------------------


class FakeFrames:
    """A `LipsyncFrames` with no model behind it - and a real `Slot`'s semantics.

    `get()` returns the newest frame and only the newest, which is the whole point of
    the slot the renderer writes into: a reader that falls behind loses movement rather
    than accumulating a backlog that arrives late. Tests that need two frames on the
    wire therefore have to publish them over time, the way the render loop does, not
    stack them up first.

    Nothing here imports `daemon.face_lipsync`, on purpose: the route names no part of
    that package (CONTRACTS 4), so its tests must not either, or they would prove the
    route works against the one object the layering says it must not depend on.
    """

    def __init__(self, *, clip: str = "idle2") -> None:
        self.failed = False
        self.clip = clip
        self._frame: bytes | None = None

    def put(self, frame: bytes) -> None:
        self._frame = frame

    def get(self) -> bytes | None:
        return self._frame


# Bytes that are not valid UTF-8 and never will be. A frame is binary on a channel
# that is defined as text, so "did the transport mangle it" is the question worth
# asking, and a printable payload cannot ask it.
FRAME = b"\xff\xd8\xff\xe0" + bytes(range(256)) + b"\xff\xd9"


def _with_lipsync(app, source=None, *, enabled: bool = True):
    """Flip the switch, inject the source, and shorten the keepalive.

    Settings are rebuilt rather than mutated so the test sees exactly the object a
    real boot would build from `.env`. The keepalive is shortened for the reason every
    stream test in this file does it: `httpx.ASGITransport` holds the first chunk until
    a second one exists, so production's 20s would be a stall rather than a test.
    """
    app.state.settings = _settings(
        app.state.settings.data_dir, face_lipsync_enabled=enabled
    )
    app.state.keepalive_seconds = 0.05
    if source is not None:
        app.state.face_frames = source
    return app




async def _transcript(app, drive):
    """Everything `/face/frames` put on the wire, as lines, with `drive` running beside it.

    `drive` MUST end by latching `source.failed`, and that is not a convenience: this
    version of `httpx.ASGITransport` runs the ASGI app **to completion** and only then
    hands back a response (it has no streaming path at all), so a request against a
    generator that never returns does not merely buffer - it never opens. Two things
    follow, and both are the reason this helper exists rather than `client.stream`:

    - the driver task has to be created BEFORE the request, because the `async with`
      body of a streamed request is not reached until the app is already done;
    - `wait_for` turns "the response never ended" into a failing test instead of a
      suite that hangs, which is exactly the property the latched-`failed` path needs.

    What this cannot show is that a frame reached the page *promptly*; the transcript
    proves order and content, not latency. That is what running the real thing is for.

    Returns the raw body, not lines: the response is multipart with JPEG payloads, so
    splitting it on newlines would cut through the images themselves.
    """
    task = asyncio.create_task(drive())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://face") as c:
            r = await asyncio.wait_for(c.get("/face/frames"), 5.0)
    finally:
        task.cancel()
    assert r.status_code == 200
    assert "multipart/x-mixed-replace" in r.headers["content-type"]
    return r.content


def _frames_in(body):
    """The JPEGs the page would have received, taken back out of the multipart parts.

    Parsed by the declared Content-Length rather than by searching for the next
    boundary: JPEG is binary and may contain the boundary bytes, and a parser that
    scanned for them would truncate exactly the frames it was meant to prove intact.
    """
    out, sep = [], b"--" + BOUNDARY + b"\r\n"
    for part in body.split(sep)[1:]:
        head, _, rest = part.partition(b"\r\n\r\n")
        length = next(
            (
                int(line.split(b":")[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length")
            ),
            None,
        )
        assert length is not None, f"part with no Content-Length: {head!r}"
        assert b"image/jpeg" in head.lower(), f"part is not a JPEG: {head!r}"
        out.append(rest[:length])
    return out


def test_frames_say_the_switch_is_off_rather_than_just_failing(app):
    """The default install. 503 is right, and so is saying which of three reasons.

    The status is load-bearing beyond diagnostics: a non-200 is what makes the page's
    `EventSource` fail the connection for good instead of retrying it forever.
    """
    r = TestClient(app).get("/face/frames")
    assert r.status_code == 503
    # The env key, verbatim, because that is the line the reader has to change.
    assert "DAEMON_FACE_LIPSYNC_ENABLED" in r.text
    # And the page is told the same thing without having to make a failing request.
    assert TestClient(app).get("/face/manifest").json()["lipsync"] is False


def test_frames_distinguish_a_missing_renderer_from_a_closed_switch(app):
    """Switch on, nothing wired: the owner has not fetched the weights yet.

    This is the install that would otherwise look broken - the switch says yes and
    there is no mouth - so the 503 has to name the cache it is missing rather than
    repeat the switch back at someone who has already set it.
    """
    _with_lipsync(app)
    r = TestClient(app).get("/face/frames")
    assert r.status_code == 503
    assert "no renderer" in r.text
    assert "DAEMON_FACE_LIPSYNC_ENABLED" not in r.text
    assert TestClient(app).get("/face/manifest").json()["lipsync"] is False


def test_a_latched_failure_stops_both_surfaces_advertising_a_mouth(app):
    """`Renderer.failed` latches, so the page must be able to go back to the clips.

    The healthy assertion first is the point: the same source advertises a mouth until
    it fails and refuses afterwards, so this is the latch being observed rather than a
    route that was never working in the first place.
    """
    source = FakeFrames()
    source.put(FRAME)
    _with_lipsync(app, source)
    assert TestClient(app).get("/face/manifest").json()["lipsync"] == {
        "clip": "idle2",
    }

    source.failed = True
    r = TestClient(app).get("/face/frames")
    assert r.status_code == 503
    assert "failed" in r.text
    # No stale mouth on the manifest either, so a page booting after the failure
    # never opens the stream at all and plays clips from the first frame.
    assert TestClient(app).get("/face/manifest").json()["lipsync"] is False


def test_the_manifest_names_the_driving_clip_and_no_geometry(app):
    """This used to carry a box for the page to place a crop with. The frames are whole
    composited images now, so there is nothing to place - and a box the page cannot use
    would be the last trace of the transport that drew a rectangle across the head."""
    _with_lipsync(app, FakeFrames(clip="idle3"))
    body = TestClient(app).get("/face/manifest").json()
    assert body["lipsync"] == {"clip": "idle3"}


async def test_a_quiet_renderer_keeps_the_stream_open_and_sends_no_frame(app):
    """Idle is not an error, and neither is an alternate tick.

    A model step covers two frames and cannot start until the second one's audio has
    arrived, so the source answers "nothing new" on some ticks by design; between turns
    it answers that for minutes, and the response has to stay open across it without
    inventing a picture. The keepalive this test used to assert is gone with SSE: a
    multipart stream has nothing to keep alive, and a comment frame would be a
    malformed part. What proves liveness now is that the response is still open when
    the driver ends it - which `_transcript` establishes by getting a 200 at all.
    """
    source = FakeFrames()          # never fed: no frame has ever existed

    async def drive():
        await asyncio.sleep(0.25)  # ~50 poll turns at FRAME_POLL_SECONDS
        source.failed = True

    _with_lipsync(app, source)
    body = await _transcript(app, drive)

    assert not _frames_in(body), "a quiet renderer must not produce a frame"
    assert body.endswith(b"--" + BOUNDARY + b"--\r\n"), (
        "the stream must close cleanly so the page's <img> sees the end"
    )


async def test_the_frames_reach_the_page_byte_for_byte_and_in_order(app):
    """The JPEGs the renderer published, base64'd, in the order it published them.

    Fed from a task rather than pre-loaded because that is how they really arrive - the
    slot holds one frame, so stacking three up first would only ever deliver the third.
    Three frames spread over time also mean the transcript could not have come from a
    single read of the slot.
    """
    source = FakeFrames()

    async def drive():
        for n in range(3):
            await asyncio.sleep(0.02)
            source.put(FRAME + bytes([n]))
        await asyncio.sleep(0.05)
        source.failed = True

    _with_lipsync(app, source)
    lines = await _transcript(app, drive)

    assert _frames_in(lines) == [FRAME + bytes([n]) for n in range(3)]


async def test_the_same_frame_is_never_sent_twice(app):
    """The slot is latest-wins, so an unchanged slot means "nothing new", not "send it
    again". A transport that re-sent would spend 55KB a poll on a still mouth."""
    source = FakeFrames()
    source.put(FRAME)

    async def drive():
        # ~40 poll turns over an unchanged slot, then a second, distinct frame, so the
        # assertion is about de-duplication and not about the loop having stalled.
        await asyncio.sleep(0.2)
        source.put(FRAME + b"next")
        await asyncio.sleep(0.05)
        source.failed = True

    _with_lipsync(app, source)
    lines = await _transcript(app, drive)

    assert _frames_in(lines) == [FRAME, FRAME + b"next"]


async def test_a_failure_mid_stream_ends_the_response(app):
    """The one signal the page gets that the mouth is not coming back.

    Every test above leans on this, and `_transcript` fails on a hang rather than on a
    wrong value, which is the failure that matters: a renderer latching `failed` under
    an open stream would otherwise leave an `<img>` holding a frozen mouth over a
    speaking face, with nothing else for the page to learn it from - the manifest is
    only read at boot. Here it is asserted on its own so the reason is named once.
    """
    source = FakeFrames()
    source.put(FRAME)

    async def drive():
        await asyncio.sleep(0.05)
        source.failed = True

    _with_lipsync(app, source)
    lines = await _transcript(app, drive)

    assert _frames_in(lines) == [FRAME], "the frame it already had should still go out"
