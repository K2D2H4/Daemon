"""The face routes: a page, a stream, and clip bytes.

The stream is the interesting one. It carries three things on one channel and its
first line must be a snapshot rather than a change, because a reconnect in the middle
of a reply otherwise shows an idle face over a speaking daemon.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daemon import face_routes
from daemon.config import Settings
from daemon.face import FaceBus
from daemon.face_routes import CLIPS, available_clips, face_dir, router


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


def test_the_page_is_never_cached(app):
    # A browser caching the page shell means a shipped fix needs a hard
    # refresh to take effect - measured cost, not a hypothetical one.
    r = TestClient(app).get("/face")
    assert r.headers.get("cache-control") == "no-store"


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
    assert r.json() == {"clips": []}
    _clip(settings, "idle1")
    _clip(settings, "speaking_soft")
    r = TestClient(app).get("/face/manifest")
    assert r.json() == {"clips": ["idle1", "speaking_soft"]}


# --- the stream, read as the server actually writes it -----------------------

_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0"},
    "http_version": "1.1",
    "method": "GET",
    "headers": [],
    "scheme": "http",
    "path": "/face/stream",
    "raw_path": b"/face/stream",
    "query_string": b"",
    "root_path": "",
    "server": ("face", 80),
    "client": ("127.0.0.1", 1),
}


class _Stream:
    """One open `/face/stream`, chunk by chunk, in the order the server sent them."""

    def __init__(self) -> None:
        self.start: dict = {}
        self.chunks: asyncio.Queue[bytes] = asyncio.Queue()

    @property
    def status(self) -> int:
        return self.start["status"]

    @property
    def content_type(self) -> str:
        return dict(self.start["headers"])[b"content-type"].decode()

    async def payload(self, within: float = 1.0) -> dict:
        """The next `data:` event, decoded - or fail the test rather than hang
        the suite. Timing out here is the assertion: an event that never arrives
        is exactly the failure these tests are about."""
        async with asyncio.timeout(within):
            while True:
                for line in (await self.chunks.get()).decode().splitlines():
                    if line.startswith("data:"):
                        return json.loads(line[5:])

    async def comment(self, within: float = 1.0) -> bytes:
        """The next keepalive comment, skipping any events on the way."""
        async with asyncio.timeout(within):
            while True:
                chunk = await self.chunks.get()
                if chunk.startswith(b":"):
                    return chunk


@asynccontextmanager
async def open_stream(app):
    """Drive `GET /face/stream` as ASGI, exposing each chunk as it is sent.

    Deliberately not `httpx.ASGITransport`, which these tests used to use: that
    transport awaits the *whole* application call before it returns a response
    (httpx 0.28's `handle_async_request` ends `await self.app(...)` with
    `assert response_complete.is_set()`, then hands back the joined body), so it
    cannot see a chunk until the stream finishes - and on a stream that never
    finishes it never returns at all. Both of those mattered here: the old tests
    could only run because the keepalive bug was ending the stream after one
    tick, and the one-shot they published was published after the response had
    already completed, so it could never have appeared no matter what was
    asserted about it.
    """
    stream = _Stream()
    disconnected = asyncio.Event()

    async def receive() -> dict:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            stream.start = message
        elif message["type"] == "http.response.body" and message.get("body"):
            stream.chunks.put_nowait(message["body"])

    task = asyncio.create_task(app(_SCOPE, receive, send))
    try:
        yield stream
    finally:
        # The window closing, which is how a face stream really ends - not
        # `task.cancel()`. Closing one whose `__anext__` was still in flight
        # raised `RuntimeError: aclose(): asynchronous generator is already
        # running` until the route learned to await that cancellation before
        # closing, and nothing about `_subs` can see that: the subscriber is
        # discarded on the way out either way.
        disconnected.set()
        try:
            async with asyncio.timeout(1.0):
                (outcome,) = await asyncio.gather(task, return_exceptions=True)
        except TimeoutError:
            task.cancel()
            raise AssertionError("the stream did not end when the client went away") from None
        assert outcome is None, outcome


async def test_the_stream_opens_with_a_snapshot(app):
    """Stream must open with a snapshot, not wait for the first event: without
    one, a reconnect during a reply leaves the face on idle over a speaking
    daemon."""
    app.state.face.set_activity("thinking")
    async with open_stream(app) as stream:
        snapshot = await stream.payload()
        assert stream.status == 200
        assert "text/event-stream" in stream.content_type
        assert snapshot == {"kind": "state", "activity": "thinking", "level": 0.0}


async def test_the_stream_carries_a_one_shot(app):
    """A one-shot published while the stream is open arrives on it as its own
    event, after the snapshot."""
    async with open_stream(app) as stream:
        assert await stream.payload() == {"kind": "state", "activity": "idle", "level": 0.0}
        app.state.face.one_shot("amused")
        assert await stream.payload() == {"kind": "shot", "clip": "amused"}


async def test_a_keepalive_does_not_end_the_stream(app, monkeypatch):
    """The keepalive must not cost the stream the events it exists to keep
    carrying.

    `asyncio.wait_for(agen.__anext__(), ...)` cancelled the pending call on
    every tick, which unsubscribed the client inside the bus and left the route
    reading a closed generator - so the stream died after the first quiet
    interval. The browser reconnects, which is why the face survived it, but a
    one-shot published in the gap is lost outright: a reconnect's snapshot
    re-sends state and nothing else.

    `KEEPALIVE_SECONDS` is patched rather than read from a test-only hook on
    `app.state` (which the route used to carry): the production constant is what
    the route reads, and 20 seconds of real waiting is not a test.
    """
    monkeypatch.setattr(face_routes, "KEEPALIVE_SECONDS", 0.01)
    async with open_stream(app) as stream:
        await stream.payload()                    # the snapshot
        assert await stream.comment() == b":\n\n"
        await asyncio.sleep(0.1)                  # ~10 more ticks
        app.state.face.one_shot("amused")
        assert await stream.payload() == {"kind": "shot", "clip": "amused"}
        app.state.face.set_activity("speaking")
        assert await stream.payload() == {
            "kind": "state",
            "activity": "speaking",
            "level": 0.0,
        }


async def test_a_disconnected_client_is_unsubscribed(app):
    """The bus must not accumulate a subscriber per closed face window: the
    route holds a live `__anext__` now, and only cancelling it runs the bus
    generator's own `finally`."""
    bus = app.state.face
    async with open_stream(app) as stream:
        await stream.payload()
        assert len(bus._subs) == 1
    await asyncio.sleep(0)
    assert len(bus._subs) == 0


def test_transitions_are_served_when_present_and_404_otherwise(app):
    """Task 9 rule 4: no table on disk must be a clean 404, not an error - the
    page's fallback to `currentTime = 0` depends on that, and the table is
    derived from the owner's own clips so it never ships in the repo."""
    client = TestClient(app)
    assert client.get("/face/transitions").status_code == 404

    settings = app.state.settings
    d = face_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    (d / "transitions.json").write_text('{"version": 1, "match": {}}', encoding="utf-8")

    r = client.get("/face/transitions")
    assert r.status_code == 200
    assert r.json() == {"version": 1, "match": {}}


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
