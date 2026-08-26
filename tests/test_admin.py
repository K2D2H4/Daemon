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
from daemon.admin.settings_io import current_settings_payload
from daemon.app import create_app
from daemon.config import Route, Settings
from daemon.llm.gateway import LLMGateway
from daemon.tasks import Task

# The admin guard rejects any Host that is not loopback (defeats DNS-rebinding).
# `TestClient` defaults to `Host: testserver`; a real browser hitting the admin on
# 127.0.0.1 sends this, so it is what the tests must send too.
LOOPBACK = "http://127.0.0.1"


def _settings(tmp_path: Path, **kw: object) -> Settings:
    """A valid all-local configuration, isolated from the developer's own `.env`.

    `provider="ollama"` needs no key and no hosted provider, so it is the cheapest
    base a validation test can start from - and `_env_file=None` keeps the
    worktree's own `.env` out of it (the same reason `conftest` strips the
    environment)."""
    kw.setdefault("provider", "ollama")
    return Settings(_env_file=None, data_dir=tmp_path, **kw)


def _served_index(tmp_path: Path) -> str:
    """The shell HTML string, as `test_shell_page_renders_offline` reads it - the
    structural tests below slice a named JS function's body out of this text."""
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    return client.get("/admin/").text


@pytest.fixture(autouse=True)
def _no_live_model_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /settings now probes the hosted three for live model ids
    (routes.py `_live_model_lists`) - a real `httpx.get` to Anthropic/OpenAI/Gemini
    for any provider whose key is set. Several tests in this module set a fake key
    (e.g. `gemini_api_key="k"`) for reasons unrelated to this feature, and the
    suite may never touch the network (tests/CLAUDE.md) - so stub all three probes
    here by default. The live-model-list tests below override individual probes
    with their own `monkeypatch.setattr`."""
    from daemon.admin import routes
    from daemon.setup import Verdict

    def fake(*_a: object, **_k: object) -> Verdict:
        return Verdict(ok=True, detail="stub")

    monkeypatch.setattr(routes, "check_anthropic", fake)
    monkeypatch.setattr(routes, "check_openai", fake)
    monkeypatch.setattr(routes, "check_gemini", fake)


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
    original = "DAEMON_PROVIDER=ollama\n"
    env.write_text(original, encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"provider": "does-not-exist"})

    assert resp.status_code == 400
    assert "provider" in resp.json()["detail"].lower()
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"


# --- c. valid PATCH writes exactly the changed key ---------------------------


def test_patch_with_a_valid_value_writes_only_that_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")

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
    assert "DAEMON_PROVIDER=ollama" in text, "an unrelated line was lost"


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


# --- e. the openai_compatible endpoint is editable like its sibling models ---


def test_patch_sets_the_compatible_endpoint_and_model(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={
            "provider": "openai_compatible",
            "openai_compatible_base_url": "https://api.deepseek.com/v1",
            "openai_compatible_model": "deepseek-chat",
            "openai_compatible_api_key": "sk-compat-key",
        },
    )

    assert resp.status_code == 200, resp.text
    written = env.read_text(encoding="utf-8")
    assert "DAEMON_PROVIDER=openai_compatible" in written
    assert "DAEMON_OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1" in written
    assert "DAEMON_OPENAI_COMPATIBLE_MODEL=deepseek-chat" in written


# --- f. the settings payload speaks provider, not preset (docs/adr/0014) -----


def test_settings_offers_the_provider_axis_not_a_preset(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    client = TestClient(app, base_url=LOOPBACK)

    got = client.get("/admin/api/settings").json()

    assert "preset" not in got["editable"] and "hosted_provider" not in got["editable"]
    assert got["editable"]["provider"] == "gemini"
    assert got["editable"]["proactive_judge_local"] is True
    assert got["options"]["providers"][0] == "", (
        "empty (no provider chosen yet) must be offered first"
    )
    assert "ollama" in got["options"]["providers"]
    assert got["options"]["model_suggestions"]["gemini"]  # non-empty


def test_patch_sets_ollama_as_the_provider_and_its_model(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings", json={"provider": "ollama", "ollama_model": "qwen3:14b"}
    )

    assert resp.status_code == 200, resp.text
    text = env.read_text(encoding="utf-8")
    assert "DAEMON_PROVIDER=ollama" in text and "DAEMON_OLLAMA_MODEL=qwen3:14b" in text
    assert "DAEMON_PRESET" not in text


def test_the_off_provider_note_appears_only_for_out_of_band_routing(tmp_path: Path) -> None:
    plain = current_settings_payload(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    assert plain["editable"]["off_provider_note"] is None

    routed = current_settings_payload(_settings(
        tmp_path,
        provider="gemini",
        gemini_model="g",
        gemini_api_key="k",
        anthropic_model="c",
        anthropic_api_key="k",
        route_overrides={"reflection": "anthropic"},
    ))
    assert routed["editable"]["off_provider_note"] is not None
    assert "anthropic" in routed["editable"]["off_provider_note"]


def test_the_off_provider_note_ignores_voices_own_provider(tmp_path: Path) -> None:
    """Voice is its own axis (ADR 0012): `voice_provider` is expected to differ
    from the chat `provider`, so CHAT_VOICE resolving to a different provider is
    not the out-of-band routing D9 exists to surface."""
    settings = _settings(
        tmp_path,
        provider="ollama",
        voice_enabled=True,
        voice_provider="gemini",
        gemini_live_model="live-model",
        gemini_api_key="k",
    )
    assert settings.routing[Task.CHAT_VOICE] == "gemini"
    payload = current_settings_payload(settings)
    assert payload["editable"]["off_provider_note"] is None


def test_the_off_provider_note_also_covers_the_fallback_provider(tmp_path: Path) -> None:
    """D9 names both `route_overrides` and `fallback_provider` as the two
    hand-edit-only ways work can land off-`provider` - `fallback_provider` is a
    separate attribute `settings.routing` does not fold in (it is only consulted
    via `fallback_route()`/`routing_table()`), so the note has to check it too or
    it silently misses a genuine off-provider config."""
    settings = _settings(
        tmp_path,
        provider="gemini",
        gemini_model="g",
        gemini_api_key="k",
        anthropic_model="c",
        anthropic_api_key="k",
        fallback_provider="anthropic",
    )
    payload = current_settings_payload(settings)
    assert payload["editable"]["off_provider_note"] is not None
    assert "anthropic" in payload["editable"]["off_provider_note"]


def test_patch_rejects_an_endpoint_carrying_the_full_path(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PROVIDER=ollama\n"
    env.write_text(original, encoding="utf-8")

    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={"openai_compatible_base_url": "https://api.deepseek.com/v1/chat/completions"},
    )

    assert resp.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"


def test_get_settings_masks_the_compatible_key(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, openai_compatible_api_key="sk-compat-SUPERSECRET"))
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.get("/admin/api/settings")
    assert resp.status_code == 200
    assert "SUPERSECRET" not in resp.text

    assert resp.json()["editable"]["openai_compatible_api_key"] == "set"


def test_settings_offer_openai_compatible_as_a_provider(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    payload = client.get("/admin/api/settings").json()

    assert "openai_compatible" in payload["options"]["providers"]


def test_the_compatible_fields_survive_a_restart_with_the_key_still_masked(
    tmp_path: Path,
) -> None:
    """PATCH writes `.env` and reports `restart_required` - it does not hot-reload
    `app.state.settings` (that object is fixed at boot, by design; see
    `test_patch_sets_the_gemini_live_voice` above). The page's own reload cannot
    re-run the process, so the real contract is proven the same way: build a fresh
    `Settings` from the written `.env` and confirm *that* process's GET shows the
    endpoint and model back, and the key as `"set"` - never in plaintext."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch(
        "/admin/api/settings",
        json={
            "provider": "openai_compatible",
            "openai_compatible_base_url": "https://api.deepseek.com/v1",
            "openai_compatible_model": "deepseek-chat",
            "openai_compatible_api_key": "sk-compat-SUPERSECRET",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["restart_required"] is True

    restarted = create_app(Settings(_env_file=str(env), data_dir=tmp_path))
    got = TestClient(restarted, base_url=LOOPBACK).get("/admin/api/settings")
    assert "SUPERSECRET" not in got.text

    editable = got.json()["editable"]
    assert editable["provider"] == "openai_compatible"
    assert editable["openai_compatible_base_url"] == "https://api.deepseek.com/v1"
    assert editable["openai_compatible_model"] == "deepseek-chat"
    assert editable["openai_compatible_api_key"] == "set"


# --- health, shell, restart --------------------------------------------------


def test_admin_health_matches_the_health_endpoint(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    admin = client.get("/admin/api/health")
    plain = client.get("/health")

    assert admin.status_code == 200
    assert admin.json() == plain.json()
    assert admin.json()["provider"] == "ollama"


def test_shell_page_renders_offline(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    resp = client.get("/admin/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # Self-contained: no CDN font or stylesheet request (offline requirement).
    assert "fonts.googleapis.com" not in resp.text
    assert "Daemon" in resp.text


def test_voice_sensitivities_are_wired_through_the_provider_aware_block(
    tmp_path: Path,
) -> None:
    """voice_start/end_sensitivity are Gemini-only (daemon/voice/openai_realtime.py
    never reads them, and app.py passes them only in the GeminiLiveSession branch).
    The shell has no JS harness, so this cannot drive a real provider switch - but it
    can assert the source is structured so a switch would work: the two sensitivity
    fields must be emitted from inside `voiceField()`, the function the `voice_provider`
    `change` listener re-renders, rather than unconditionally in `renderSettings()`.
    Reverting to the old unconditional emission (both fieldStr calls sitting in
    renderSettings, outside voiceField) would fail this test.
    """
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    html = client.get("/admin/").text

    start = html.index("function voiceField(")
    end = html.index("\nfunction ", start + 1)
    voice_field_fn = html[start:end]
    outside = html[:start] + html[end:]

    for name in ("voice_start_sensitivity", "voice_end_sensitivity"):
        assert name in voice_field_fn, f"{name} must render from inside voiceField()"
        assert name not in outside, f"{name} must not also render unconditionally"


def test_voice_model_ids_render_from_the_provider_aware_block_not_unconditionally(
    tmp_path: Path,
) -> None:
    """gemini_live_model and openai_realtime_model are each read by only one
    voice_provider (config.py) - the old flat MODEL card showed both regardless of
    which provider was selected, the exact "shown but not in force" problem the
    provider-first redesign exists to remove. Both must render from inside
    `voiceField()` - the function the `voice_provider` `change` listener
    re-renders - never unconditionally from `renderSettings()`."""
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    html = client.get("/admin/").text

    start = html.index("function voiceField(")
    end = html.index("\nfunction ", start + 1)
    voice_field_fn = html[start:end]
    outside = html[:start] + html[end:]

    for name in ("gemini_live_model", "openai_realtime_model"):
        assert name in voice_field_fn, f"{name} must render from inside voiceField()"
        assert name not in outside, f"{name} must not also render unconditionally"


# --- g. the MODEL card asks provider first, then that provider's model (D5/D7/D9) --


def test_model_fields_render_from_brainfield_not_rendersettings(tmp_path: Path) -> None:
    """Every model-shaped field must come from inside `brainField()` - the function
    the provider `<select>`'s change handler re-renders - not unconditionally from
    `renderSettings()`, which cannot react to a provider switch. The old flat
    eight-box card (`fieldStr('preset', ...)` / `fieldStr('hosted_provider', ...)`)
    must be gone entirely, not just supplemented."""
    html = _served_index(tmp_path)
    start = html.index("function brainField(")
    body = html[start : html.index("\nfunction ", start + 1)]
    for name in (
        "provider",
        "proactive_judge_local",
        "ollama_model",
        "openai_compatible_base_url",
    ):
        assert name in body, f"{name} must render from inside brainField()"
    assert "fieldStr('preset'" not in html and "fieldStr('hosted_provider'" not in html


def test_the_chat_model_field_prefers_the_live_list(tmp_path: Path) -> None:
    """D5/D6: the hosted-three model field prefers the live probe result
    (`options.model_lists`), falling back to the static `options.model_suggestions`
    only when the probe found nothing."""
    html = _served_index(tmp_path)
    assert "model_lists" in html and "model_suggestions" in html


def test_the_chat_model_field_is_a_click_open_select_not_a_datalist(tmp_path: Path) -> None:
    """A <datalist> does not open on click - it reads as empty/broken (reported). With
    a list to offer, fieldModel renders a real <select> (opens on click, like the
    wizard) whose last option swaps to free text, so a not-yet-listed id stays typable.
    No <datalist>/`list=` input remains for a model field."""
    html = _served_index(tmp_path)
    # fieldModel delegates the has-a-list case to modelSelect, which builds a real
    # <select> with the free-text escape option.
    assert "modelSelect(" in html
    ms = html.index("function modelSelect(")
    body = html[ms : html.index("\nfunction ", ms + 1)]
    assert "<select" in body and "MODEL_CUSTOM" in body  # the free-text escape option
    assert "MODEL_CUSTOM='__custom__'" in html.replace(" ", "")  # sentinel is defined
    # An empty running value must keep an empty option selected, or the <select> would
    # default to its first option and phantom-select a model the user never chose when
    # they switch to a not-yet-configured provider. The head option must carry selected.
    assert 'value="" selected' in body
    # No <datalist> element / `list=` input remains anywhere on the page (the old,
    # click-broken form). Match the element form so the prose comment explaining the
    # removal does not trip the assertion.
    assert "<datalist id" not in html and 'list="dl-' not in html


def test_collect_patch_emits_provider_from_the_brain_select(tmp_path: Path) -> None:
    """The provider select carries `data-brain`, not `data-f`, so the generic
    `[data-f]` loop in `collectPatch()` skips it - a provider switch is a structural
    change (which model field exists, whether the toggle shows) that brainField's
    own re-render handles, not something the loop's per-field logic understands.
    `collectPatch()` must still read it explicitly and emit `patch.provider` when it
    differs from the value the form was drawn with."""
    html = _served_index(tmp_path)
    start = html.index("function collectPatch(")
    body = html[start : html.index("\nfunction ", start + 1)]
    assert 'data-brain="provider"' in body
    assert "patch.provider" in body


def test_status_meta_reads_provider_not_preset(tmp_path: Path) -> None:
    """/health carries `provider`, not `preset` (Task 3 reshaped the payload) - the
    top bar's own rendering must read the field that actually exists."""
    html = _served_index(tmp_path)
    assert "h.provider" in html
    assert "h.preset" not in html


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


def test_supervised_restart_releases_the_face_stream_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit this endpoint promises did not happen while a face page was open:
    `/face/stream` is a response uvicorn cannot close, so the process sat in
    `Waiting for connections to close` and launchd never revived it (measured, 6 of
    8 restarts - daemon/MEASURED.md). So the bus is closed before the signal, and
    the endpoint has to actually reach it - not just not error."""
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr("daemon.admin.routes.is_supervised", lambda *a, **k: True)
    monkeypatch.setattr("daemon.admin.restart._raise_sigterm", lambda: None)
    closed: list[bool] = []
    monkeypatch.setattr(app.state.face, "close", lambda: closed.append(True))

    with TestClient(app, base_url=LOOPBACK) as client:
        assert client.post("/admin/api/restart").status_code == 200

    assert closed == [True], "the restart never released the face stream"


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
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
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
    assert env.read_text(encoding="utf-8") == "DAEMON_PROVIDER=ollama\n"


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
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
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
    original = "DAEMON_PROVIDER=ollama\n"
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
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Kore"})
    assert resp.status_code == 200
    assert resp.json()["restart_required"] is True
    assert "DAEMON_GEMINI_LIVE_VOICE=Kore" in env.read_text(encoding="utf-8")

    restarted = create_app(Settings(_env_file=str(env), data_dir=tmp_path))
    got = TestClient(restarted, base_url=LOOPBACK).get("/admin/api/settings").json()
    assert got["editable"]["gemini_live_voice"] == "Kore"
    assert "Kore" in got["options"]["gemini_live_voices"]
    assert got["options"]["gemini_live_voices"][0] == "", (
        "empty (server default) must be offered first"
    )


def test_patch_sets_the_openai_voice_provider_and_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={
        "voice_provider": "openai",
        "openai_realtime_voice": "alloy",
        "openai_realtime_model": "gpt-realtime",
    })
    assert resp.status_code == 200
    text = env.read_text(encoding="utf-8")
    assert "DAEMON_VOICE_PROVIDER=openai" in text
    assert "DAEMON_OPENAI_REALTIME_VOICE=alloy" in text
    assert "DAEMON_OPENAI_REALTIME_MODEL=gpt-realtime" in text

    got = client.get("/admin/api/settings").json()
    voice_providers = got["options"]["voice_providers"]
    assert "openai" in voice_providers and "gemini" in voice_providers
    assert got["options"]["openai_realtime_voices"][0] == ""  # server-default offered first
    assert "alloy" in got["options"]["openai_realtime_voices"]


def test_patch_rejects_an_unknown_voice_provider_and_openai_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PROVIDER=ollama\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    bad_provider = client.patch("/admin/api/settings", json={"voice_provider": "anthropic"})
    assert bad_provider.status_code == 400
    bad_voice = client.patch("/admin/api/settings", json={"openai_realtime_voice": "nope"})
    assert bad_voice.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"


def test_patch_rejects_an_unknown_gemini_live_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PROVIDER=ollama\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Nope"})
    assert resp.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected voice still wrote"


def test_a_newline_in_a_route_override_is_refused(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PROVIDER=ollama\n"
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


def test_voice_sample_serves_both_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    (samples / "gemini").mkdir(parents=True)
    (samples / "openai").mkdir(parents=True)
    (samples / "gemini" / "Kore.mp3").write_bytes(b"ID3-gemini")
    (samples / "openai" / "alloy.mp3").write_bytes(b"ID3-openai")
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    g = client.get("/admin/api/voice-sample/gemini/Kore")   # Gemini no-regression
    assert g.status_code == 200 and g.headers["content-type"] == "audio/mpeg"
    assert g.content == b"ID3-gemini"
    o = client.get("/admin/api/voice-sample/openai/alloy")
    assert o.status_code == 200 and o.content == b"ID3-openai"


def test_voice_sample_404_for_unknown_provider_voice_or_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    (samples / "gemini").mkdir(parents=True)
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    assert client.get("/admin/api/voice-sample/gemini/Kore").status_code == 404   # known, no file
    assert client.get("/admin/api/voice-sample/gemini/Nope").status_code == 404   # unknown voice
    assert client.get("/admin/api/voice-sample/openai/Kore").status_code == 404   # wrong provider
    assert client.get("/admin/api/voice-sample/nope/Kore").status_code == 404     # unknown provider
    assert client.get("/admin/api/voice-sample/gemini/..%2f..%2fetc%2fpasswd").status_code == 404


# --- the switches and ids that used to be hand-edit-only ----------------------
#
# The Overview reports a proactivity budget, a wake gate and a screen capability;
# until this milestone none of their switches were editable here, so the page
# described a daemon it could not configure. These prove the round-trip for the
# kinds that are not plain strings.


def test_patch_sets_a_switch_the_overview_reports_on(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
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

    restarted = Settings(_env_file=str(env), data_dir=tmp_path)
    assert restarted.proactive_enabled is True
    assert restarted.voice_barge_in is False, (
        "the switch main added in v0.1.27 must survive the round-trip, or the page "
        "silently reverts a value the owner set by hand"
    )


def test_patch_sets_proactive_judge_local(tmp_path: Path) -> None:
    """`proactive_judge_local` is a BOOL_FIELD like the switches above, not a
    STR_FIELD - only a GET-default test covered it before this; this proves the
    PATCH direction round-trips too."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"proactive_judge_local": False})
    assert resp.status_code == 200, resp.text
    assert "DAEMON_PROACTIVE_JUDGE_LOCAL=false" in env.read_text(encoding="utf-8")

    restarted = Settings(_env_file=str(env), data_dir=tmp_path)
    assert restarted.proactive_judge_local is False

    got = TestClient(create_app(restarted), base_url=LOOPBACK).get("/admin/api/settings").json()
    assert got["editable"]["proactive_judge_local"] is False


def test_wake_aliases_round_trip_as_the_comma_form_a_person_types(tmp_path: Path) -> None:
    """Korean, because that is what the recognizer returns for this owner and the
    JSON-array form a naive round-trip produces is not what `.env` holds."""
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"wake_aliases": "헤이 대문,루씨"})
    assert resp.status_code == 200, resp.text
    assert "DAEMON_WAKE_ALIASES=헤이 대문,루씨" in env.read_text(encoding="utf-8")

    restarted = Settings(_env_file=str(env), data_dir=tmp_path)
    assert restarted.wake_aliases == ("헤이 대문", "루씨")

    shown = TestClient(
        create_app(restarted), base_url=LOOPBACK
    ).get("/admin/api/settings").json()
    assert shown["editable"]["wake_aliases"] == "헤이 대문,루씨", (
        "reported as the comma form, not as a JSON array - the field is a text input"
    )


def test_the_telegram_token_is_never_reported_back(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PROVIDER=ollama\nTELEGRAM_BOT_TOKEN=123:secret\n", encoding="utf-8")
    settings = Settings(_env_file=str(env), data_dir=tmp_path)
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
    original = "DAEMON_PROVIDER=ollama\n"
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
    env.write_text("DAEMON_PROVIDER=ollama\nDAEMON_RECALL_LIMIT=6\n", encoding="utf-8")
    app = create_app(Settings(_env_file=str(env), data_dir=tmp_path))
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
    env.write_text("DAEMON_PROVIDER=nonsense-provider\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.get("/admin/api/settings")

    assert resp.status_code == 200
    assert resp.json()["pending"] == {}


# --- g. live model lists probe off the event loop, never the network in tests -


async def test_live_model_lists_fall_back_to_empty_when_a_probe_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from daemon.admin import routes

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(routes, "check_gemini", boom)
    lists = await routes._live_model_lists(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    assert lists.get("gemini") == [], "failure must fall back to empty, never raise"


def test_chat_only_drops_specialties_and_keeps_chat_ids() -> None:
    """The blocklist itself, without the probe scaffolding - so a marker regression
    fails here plainly. Includes the two deliberate drops of chat-capable ids
    (vision/audio) so their exclusion is explicit, not accidental."""
    from daemon.admin.routes import _chat_only

    assert _chat_only(
        (
            "gemini-3.6-flash", "gpt-5.2", "claude-opus-5",  # mainstream chat - kept
            "gemini-3-pro-image", "dall-e-3", "text-embedding-3-large",  # dropped
            "whisper-1", "gpt-4o-realtime-preview", "gemini-robotics-er-2",  # dropped
            "gpt-4-vision-preview", "gpt-4o-audio-preview",  # deliberate drops
        )
    ) == ["gemini-3.6-flash", "gpt-5.2", "claude-opus-5"]
    # An empty probe stays empty; order is preserved.
    assert _chat_only(()) == []
    assert _chat_only(("b-flash", "a-flash")) == ["b-flash", "a-flash"]


async def test_live_model_lists_keep_only_chat_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The account's `/models` returns image/tts/robotics/embedding specialties
    alongside the chat families; the admin datalist keeps only the chat ones so the
    dropdown is not a wall of every capability the key can touch."""
    from daemon.admin import routes
    from daemon.setup import Verdict

    mixed = (
        "gemini-3.6-flash", "gemini-3.1-pro-preview",  # chat - kept
        "gemini-3-pro-image", "gemini-3.1-flash-tts-preview",  # image/tts - dropped
        "gemini-robotics-er-2-preview", "deep-research-pro-preview",  # dropped
        "lyria-3-pro-preview", "nano-banana-pro-preview",  # dropped
    )

    def fake(*_a: object, **_k: object) -> Verdict:
        return Verdict(ok=True, detail="ok", models={"DAEMON_GEMINI_MODEL": mixed})

    monkeypatch.setattr(routes, "check_gemini", fake)
    lists = await routes._live_model_lists(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    assert lists["gemini"] == ["gemini-3.6-flash", "gemini-3.1-pro-preview"]


async def test_live_model_lists_skip_providers_without_a_key(tmp_path: Path) -> None:
    from daemon.admin import routes

    # provider=gemini needs its own key to construct at all (config.py); the point
    # here is anthropic, which has none, so it must be skipped rather than probed.
    lists = await routes._live_model_lists(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    assert "anthropic" not in lists, "no key means not probed, not an empty list"


def test_get_settings_includes_live_model_lists_for_configured_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wiring in `get_settings`, not just `_live_model_lists` in isolation:
    a provider with a key gets its live ids under `options.model_lists`, and one
    without a key is simply absent - through the real endpoint, monkeypatched
    module attribute and all (a regression test for a dict built once at import
    time silently ignoring the patch and hitting the real network instead)."""
    from daemon.admin import routes
    from daemon.setup import Verdict

    monkeypatch.setattr(
        routes,
        "check_gemini",
        lambda key: Verdict(
            ok=True, detail="ok", models={"DAEMON_GEMINI_MODEL": ("gemini-2.5-flash",)}
        ),
    )
    app = create_app(
        _settings(tmp_path, provider="gemini", gemini_model="g", gemini_api_key="k")
    )
    client = TestClient(app, base_url=LOOPBACK)

    got = client.get("/admin/api/settings").json()

    assert got["options"]["model_lists"]["gemini"] == ["gemini-2.5-flash"]
    assert "anthropic" not in got["options"]["model_lists"]
