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
