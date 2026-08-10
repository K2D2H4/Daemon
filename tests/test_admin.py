"""M5 admin web, Phase 1 — acceptance via FastAPI TestClient.

The four properties that matter (docs/design/2026-08-07-m5-admin-web-design.md
decision 2, and the M5 test gate):

  a. chat-test produces a reply while it touches memory/recall ZERO times and
     offers the model no tools (it is the "chat version of /health").
  b. a PATCH whose candidate fails validation returns 400 and writes nothing.
  c. a PATCH whose candidate validates writes exactly the changed key to `.env`,
     leaving every other line alone.
  d. GET settings never lets a raw secret off the machine.

The app is assembled without its lifespan on purpose: these endpoints read
`app.state` handles, so the test sets those handles directly rather than booting
the real Telegram channel, the scheduler and Ollama - none of which this feature
touches. Fakes come from `conftest.py` (`fake_provider`); no parallel ones.

The admin router is loopback-only and refuses cross-site writes (routes.py's
`_loopback_only` guard), so every client here binds to a loopback `base_url` - a
real browser on `127.0.0.1` sends exactly that Host, and `TestClient`'s default
`testserver` would be rejected as a DNS-rebinding attempt, which is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from daemon.admin.restart import is_supervised
from daemon.app import create_app
from daemon.config import Route, Settings
from daemon.llm.gateway import LLMGateway
from daemon.tasks import Task

# The admin guard rejects any Host that is not loopback (defeats DNS-rebinding).
# `TestClient` defaults to `Host: testserver`; a real browser hitting the admin on
# 127.0.0.1 sends this, so it is what the tests must send too.
LOOPBACK = "http://127.0.0.1"


def _settings(tmp_path: Path, **kw: object) -> Settings:
    """A valid offline configuration, isolated from the developer's own `.env`.

    `offline` needs no key and no hosted provider, so it is the cheapest base a
    validation test can start from - and `_env_file=None` keeps the worktree's own
    `.env` out of it (the same reason `conftest` strips the environment)."""
    return Settings(_env_file=None, preset="offline", data_dir=tmp_path, **kw)


def _with_gateway(app, provider) -> None:
    """Wire a fake-backed gateway the way the lifespan would, minus the network."""
    app.state.gateway = LLMGateway(
        {provider.name: provider}, {Task.CHAT_TEXT: Route(provider.name, "model")}
    )


class SpyMemory:
    """Counts every write path. chat-test must reach none of them."""

    def __init__(self) -> None:
        self.calls = 0

    async def record(self, *a: object, **k: object) -> None:
        self.calls += 1

    async def recent(self, *a: object, **k: object):
        self.calls += 1
        return []

    async def seen(self, *a: object, **k: object) -> bool:
        self.calls += 1
        return False


class SpyRecall:
    """Counts search and index. chat-test must embed and search nothing."""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, *a: object, **k: object):
        self.calls += 1
        return []

    async def index(self, *a: object, **k: object) -> None:
        self.calls += 1


# --- a. chat-test is side-effect free ----------------------------------------


def test_chat_test_replies_without_touching_memory_or_offering_tools(
    tmp_path: Path, fake_provider
) -> None:
    fake_provider.reply = "안녕하세요, 무엇을 도와드릴까요?"
    app = create_app(_settings(tmp_path))
    _with_gateway(app, fake_provider)
    memory, recall = SpyMemory(), SpyRecall()
    app.state.memory = memory
    app.state.recall = recall

    client = TestClient(app, base_url=LOOPBACK)
    resp = client.post("/admin/api/chat-test", json={"text": "테스트 메시지"})

    assert resp.status_code == 200
    assert resp.json()["reply"] == "안녕하세요, 무엇을 도와드릴까요?"
    # The whole point of decision 2: no write, no embed, no search, no tool.
    assert memory.calls == 0, "chat-test wrote to memory"
    assert recall.calls == 0, "chat-test searched or embedded"
    assert fake_provider.offered_tools[-1] == (), "chat-test offered the model tools"


def test_chat_test_rejects_an_empty_body(tmp_path: Path, fake_provider) -> None:
    app = create_app(_settings(tmp_path))
    _with_gateway(app, fake_provider)
    client = TestClient(app, base_url=LOOPBACK)
    assert client.post("/admin/api/chat-test", json={"text": "   "}).status_code == 400


def test_chat_test_surfaces_a_provider_failure_as_502(tmp_path: Path) -> None:
    """The chat test exists to catch an unreachable or misconfigured provider - a
    missing model, ollama down, a bad key. That failure must read as a clean
    message, not the 500 traceback an uncaught ProviderError produced."""
    from daemon.llm.base import ProviderError

    class Broken:
        name = "ollama"

        async def complete(self, *a: object, **k: object):
            raise ProviderError("ollama rejected the request: model not found")

        async def health(self) -> bool:
            return False

    app = create_app(_settings(tmp_path))
    _with_gateway(app, Broken())
    client = TestClient(app, base_url=LOOPBACK)
    resp = client.post("/admin/api/chat-test", json={"text": "ping"})
    assert resp.status_code == 502
    assert "model not found" in resp.json()["detail"]


# --- b. invalid PATCH writes nothing -----------------------------------------


def test_patch_with_an_invalid_value_is_400_and_leaves_env_untouched(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"preset": "does-not-exist"})

    assert resp.status_code == 400
    assert "preset" in resp.json()["detail"].lower()
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"


# --- c. valid PATCH writes exactly the changed key ---------------------------


def test_patch_with_a_valid_value_writes_only_that_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"recall_limit": 10})

    assert resp.status_code == 200
    body = resp.json()
    assert body["restart_required"] is True
    assert isinstance(body["supervised"], bool)

    text = env.read_text(encoding="utf-8")
    assert "DAEMON_RECALL_LIMIT=10" in text
    assert "DAEMON_PRESET=offline" in text, "an unrelated line was lost"


# --- d. secrets never leave the machine --------------------------------------


def test_get_settings_masks_secrets(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, anthropic_api_key="sk-ant-SUPERSECRET-value"))
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.get("/admin/api/settings")
    assert resp.status_code == 200
    assert "SUPERSECRET" not in resp.text, "a raw secret was returned"

    editable = resp.json()["editable"]
    assert editable["anthropic_api_key"] == "set"
    assert editable["openai_api_key"] is None


# --- health, shell, restart --------------------------------------------------


def test_admin_health_matches_the_health_endpoint(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    admin = client.get("/admin/api/health")
    plain = client.get("/health")

    assert admin.status_code == 200
    assert admin.json() == plain.json()
    assert admin.json()["preset"] == "offline"


def test_shell_page_renders_offline(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    resp = client.get("/admin/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # Self-contained: no CDN font or stylesheet request (offline requirement).
    assert "fonts.googleapis.com" not in resp.text
    assert "Daemon" in resp.text


def test_restart_refuses_when_not_supervised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(_settings(tmp_path))
    # Force not-supervised rather than trusting the ambient environment: a CI
    # runner under systemd sets INVOCATION_ID, so `is_supervised()` is True there
    # and this test would see a 200 (it did). Pin it, like the supervised sibling.
    monkeypatch.setattr("daemon.admin.routes.is_supervised", lambda *a, **k: False)
    called = {"exit": False}
    monkeypatch.setattr(
        "daemon.admin.routes.schedule_exit", lambda *a, **k: called.__setitem__("exit", True)
    )
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.post("/admin/api/restart")
    assert resp.status_code == 409
    assert resp.json()["supervised"] is False
    assert called["exit"] is False, "a non-supervised restart tried to exit anyway"


def test_supervised_restart_schedules_a_graceful_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When supervised, the restart schedules a graceful exit on the *running* loop
    (restart.py `schedule_exit`, finding #7). The actual SIGTERM is stubbed so the
    test process survives; what matters is the endpoint reaches it without error."""
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr("daemon.admin.routes.is_supervised", lambda *a, **k: True)
    # The scheduled signal would kill the test runner; replace it with a no-op.
    monkeypatch.setattr("daemon.admin.restart._raise_sigterm", lambda: None)

    with TestClient(app, base_url=LOOPBACK) as client:
        resp = client.post("/admin/api/restart")
    assert resp.status_code == 200
    assert resp.json() == {"restarted": True, "supervised": True}


def test_is_supervised_reads_the_supervisor_markers() -> None:
    assert is_supervised({}) is False
    assert is_supervised({"XPC_SERVICE_NAME": "ai.daemon.default"}) is True
    assert is_supervised({"INVOCATION_ID": "abc123"}) is True


# --- the loopback/CSRF guard (finding #1) ------------------------------------
# The admin has no auth by design (loopback = owner). A malicious page the owner
# visits can still `fetch()` 127.0.0.1, and DNS-rebinding makes it same-origin, so
# a router-level guard rejects a non-loopback Host and cross-site writes.


def test_a_non_loopback_host_is_refused(tmp_path: Path) -> None:
    """DNS-rebinding lands as a request whose Host is the attacker's name resolving
    to 127.0.0.1. Screening the Host hostname defeats it before the body is read."""
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url="http://attacker.example")

    # Even a safe GET is refused when the Host is not a loopback name.
    resp = client.get("/admin/api/health")
    assert resp.status_code == 403
    assert "loopback" in resp.json()["detail"].lower()


def test_a_cross_site_origin_on_a_write_is_refused(tmp_path: Path) -> None:
    """A simple cross-site POST/PATCH (no preflight) must not land its side effect.
    The Host is loopback here, so this isolates the Origin check."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"recall_limit": 10},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403
    # And the side effect never happened.
    assert env.read_text(encoding="utf-8") == "DAEMON_PRESET=offline\n"


def test_a_cross_site_fetch_metadata_header_is_refused(tmp_path: Path) -> None:
    """`Sec-Fetch-Site: cross-site` is the browser telling us the request came from
    another origin; a write carrying it is refused."""
    app = create_app(_settings(tmp_path))
    app.state.env_path = tmp_path / ".env"
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"recall_limit": 10},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert resp.status_code == 403


def test_a_same_origin_write_still_works(tmp_path: Path) -> None:
    """The guard must not break the page it protects: a same-origin write, the
    browser's own fetch from the admin tab, still lands."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"recall_limit": 10},
        headers={"Origin": LOOPBACK, "Sec-Fetch-Site": "same-origin"},
    )
    assert resp.status_code == 200
    assert "DAEMON_RECALL_LIMIT=10" in env.read_text(encoding="utf-8")


# --- malformed request bodies (finding #5) -----------------------------------


def test_a_non_json_body_is_a_400_not_a_500(tmp_path: Path, fake_provider) -> None:
    """A body that is not valid JSON must be a clean 400, never a 500 out of an
    unguarded `await request.json()`."""
    app = create_app(_settings(tmp_path))
    _with_gateway(app, fake_provider)
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.post(
        "/admin/api/chat-test",
        content=b"this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "json" in resp.json()["detail"].lower()


# --- .env newline injection (finding #2) -------------------------------------
# `.env` is line-oriented: a value carrying a newline would write an extra KEY=value
# line, escaping the EDITABLE allowlist and the validate-before-write check (e.g.
# smuggling DAEMON_HOST=0.0.0.0 to bind the admin to every interface on next boot).


def test_a_newline_in_a_secret_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"anthropic_api_key": "sk-x\nDAEMON_HOST=0.0.0.0"},
    )
    assert resp.status_code == 400
    text = env.read_text(encoding="utf-8")
    assert text == original, "an injected line reached .env"
    assert "DAEMON_HOST" not in text


def test_patch_sets_the_gemini_live_voice(tmp_path: Path) -> None:
    """PATCH writes `.env` and reports `restart_required` - it does not hot-reload
    `app.state.settings` (that object is fixed at boot, by design). So the real
    contract is proven by simulating the restart: build a fresh `Settings` from the
    written `.env` and confirm *that* process surfaces the value."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Kore"})
    assert resp.status_code == 200
    assert resp.json()["restart_required"] is True
    assert "DAEMON_GEMINI_LIVE_VOICE=Kore" in env.read_text(encoding="utf-8")

    restarted = create_app(Settings(_env_file=str(env), preset="offline", data_dir=tmp_path))
    got = TestClient(restarted, base_url=LOOPBACK).get("/admin/api/settings").json()
    assert got["editable"]["gemini_live_voice"] == "Kore"
    assert "Kore" in got["options"]["gemini_live_voices"]
    assert got["options"]["gemini_live_voices"][0] == "", (
        "empty (server default) must be offered first"
    )


def test_patch_rejects_an_unknown_gemini_live_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Nope"})
    assert resp.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected voice still wrote"


def test_a_newline_in_a_route_override_is_refused(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"route_overrides": {"chat_text": "ollama\nDAEMON_HOST=0.0.0.0"}},
    )
    assert resp.status_code == 400
    # Pinned to the newline guard specifically: without it the value would still be
    # a 400 (an unknown provider), so the message is what proves the newline was the
    # reason and the guard fired before Settings validation ever saw it.
    assert "newline" in resp.json()["detail"].lower()
    assert env.read_text(encoding="utf-8") == original


def test_voice_sample_serves_a_present_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    samples.mkdir()
    (samples / "Kore.mp3").write_bytes(b"ID3-fake-mp3-bytes")
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    resp = client.get("/admin/api/voice-sample/Kore")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"ID3-fake-mp3-bytes"


def test_voice_sample_404_for_missing_or_unknown_and_never_reads_a_bad_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    samples.mkdir()
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    # Known voice, but no file generated yet -> 404, not 500.
    assert client.get("/admin/api/voice-sample/Kore").status_code == 404
    # A name outside the allowlist -> 404, and the allowlist check runs before any
    # filesystem touch, so a traversal attempt never resolves a path.
    assert client.get("/admin/api/voice-sample/Nope").status_code == 404
    assert client.get("/admin/api/voice-sample/..%2f..%2fetc%2fpasswd").status_code == 404


# --- the switches and ids that used to be hand-edit-only ----------------------
#
# The Overview reports a proactivity budget, a wake gate and a screen capability;
# until this milestone none of their switches were editable here, so the page
# described a daemon it could not configure. These prove the round-trip for the
# kinds that are not plain strings.


def test_patch_sets_a_switch_the_overview_reports_on(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings", json={"proactive_enabled": True, "voice_barge_in": False}
    )
    assert resp.status_code == 200, resp.text
    written = env.read_text(encoding="utf-8")
    assert "DAEMON_PROACTIVE_ENABLED=true" in written
    assert "DAEMON_VOICE_BARGE_IN=false" in written

    restarted = Settings(_env_file=str(env), preset="offline", data_dir=tmp_path)
    assert restarted.proactive_enabled is True
    assert restarted.voice_barge_in is False, (
        "the switch main added in v0.1.27 must survive the round-trip, or the page "
        "silently reverts a value the owner set by hand"
    )


def test_wake_aliases_round_trip_as_the_comma_form_a_person_types(tmp_path: Path) -> None:
    """Korean, because that is what the recognizer returns for this owner and the
    JSON-array form a naive round-trip produces is not what `.env` holds."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"wake_aliases": "헤이 대문,루씨"})
    assert resp.status_code == 200, resp.text
    assert "DAEMON_WAKE_ALIASES=헤이 대문,루씨" in env.read_text(encoding="utf-8")

    restarted = Settings(_env_file=str(env), preset="offline", data_dir=tmp_path)
    assert restarted.wake_aliases == ("헤이 대문", "루씨")

    shown = TestClient(
        create_app(restarted), base_url=LOOPBACK
    ).get("/admin/api/settings").json()
    assert shown["editable"]["wake_aliases"] == "헤이 대문,루씨", (
        "reported as the comma form, not as a JSON array - the field is a text input"
    )


def test_the_telegram_token_is_never_reported_back(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\nTELEGRAM_BOT_TOKEN=123:secret\n", encoding="utf-8")
    settings = Settings(_env_file=str(env), preset="offline", data_dir=tmp_path)
    client = TestClient(create_app(settings), base_url=LOOPBACK)

    body = client.get("/admin/api/settings").json()

    assert body["editable"]["telegram_bot_token"] == "set"
    assert "123:secret" not in json.dumps(body), "the token left the machine"


def test_an_incoherent_switch_combination_is_refused_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The wake gate exists only to open a voice session, so `Settings` refuses it
    without voice. The admin must surface that as a 400 rather than write a config
    the next boot would die on - the whole reason a patch is validated by
    constructing a candidate before a byte is written."""
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"wake_enabled": True})

    assert resp.status_code == 400
    assert "DAEMON_VOICE_ENABLED" in resp.json()["detail"], (
        "the error has to name the setting that would fix it"
    )
    assert env.read_text(encoding="utf-8") == original, "a refused patch reached .env"


def test_a_saved_value_is_reported_as_pending_until_the_restart(tmp_path: Path) -> None:
    """The running daemon does not hot-reload, so after a save `app.state.settings`
    still holds the old value. Reporting only that made a reload after a successful
    save look exactly like a save that was lost (measured by doing it). `pending`
    says what `.env` holds *beside* what the process is running - never instead of
    it, because the admin's one job is not to lie about the daemon's actual state."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\nDAEMON_RECALL_LIMIT=6\n", encoding="utf-8")
    app = create_app(Settings(_env_file=str(env), preset="offline", data_dir=tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    assert client.get("/admin/api/settings").json()["pending"] == {}, "nothing saved yet"

    assert client.patch("/admin/api/settings", json={"recall_limit": 17}).status_code == 200

    body = client.get("/admin/api/settings").json()
    assert body["pending"] == {"recall_limit": 17}, "the saved value has to be visible"
    assert body["editable"]["recall_limit"] == 6, (
        "and the running value must still be reported as the running value"
    )


def test_an_unreadable_env_reports_no_pending_rather_than_failing(tmp_path: Path) -> None:
    """`pending` is a convenience over the settings page's real job. A `.env` that
    cannot be built into a candidate must cost the reader that convenience, never
    the page."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=nonsense-preset\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.get("/admin/api/settings")

    assert resp.status_code == 200
    assert resp.json()["pending"] == {}
