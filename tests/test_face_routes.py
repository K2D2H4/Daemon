"""The face routes: a page, a stream, and clip bytes.

The stream is the interesting one. It carries three things on one channel and its
first line must be a snapshot rather than a change, because a reconnect in the middle
of a reply otherwise shows an idle face over a speaking daemon.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
