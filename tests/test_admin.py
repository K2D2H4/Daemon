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

    client = TestClient(app)
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
    client = TestClient(app)
    assert client.post("/admin/api/chat-test", json={"text": "   "}).status_code == 400


# --- b. invalid PATCH writes nothing -----------------------------------------


def test_patch_with_an_invalid_value_is_400_and_leaves_env_untouched(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app)

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
    client = TestClient(app)

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
    client = TestClient(app)

    resp = client.get("/admin/api/settings")
    assert resp.status_code == 200
    assert "SUPERSECRET" not in resp.text, "a raw secret was returned"

    editable = resp.json()["editable"]
    assert editable["anthropic_api_key"] == "set"
    assert editable["openai_api_key"] is None


# --- health, shell, restart --------------------------------------------------


def test_admin_health_matches_the_health_endpoint(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    admin = client.get("/admin/api/health")
    plain = client.get("/health")

    assert admin.status_code == 200
    assert admin.json() == plain.json()
    assert admin.json()["preset"] == "offline"


def test_shell_page_renders_offline(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)
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
    # No launchd/systemd markers -> not supervised. Assert we never try to exit.
    called = {"exit": False}
    monkeypatch.setattr(
        "daemon.admin.routes.schedule_exit", lambda *a, **k: called.__setitem__("exit", True)
    )
    client = TestClient(app)

    resp = client.post("/admin/api/restart")
    assert resp.status_code == 409
    assert resp.json()["supervised"] is False
    assert called["exit"] is False, "a non-supervised restart tried to exit anyway"


def test_is_supervised_reads_the_supervisor_markers() -> None:
    assert is_supervised({}) is False
    assert is_supervised({"XPC_SERVICE_NAME": "ai.daemon.default"}) is True
    assert is_supervised({"INVOCATION_ID": "abc123"}) is True
