"""M5 admin web, Phase 2 + 2b — MCP install routes and the OAuth bridge.

What matters here (docs/design/2026-08-07-m5-admin-web-design.md, "JSON API" and
"OAuth (Phase 2b)"), and the CONTRACTS 12 rule that MCP lives behind its own
switch:

  a. every MCP route is a 409 while DAEMON_MCP_ENABLED is off.
  b. the catalog is listed with the fields the UI needs and no command/url.
  c. connect persists the server block *before* it connects, so a failed connect
     leaves it configured-but-not-connected rather than vanishing.
  d. a key server's secret is written to `.env` under the entry's key_env and then
     passed to the connect explicitly (a fresh `.env` is not in os.environ).
  e. the OAuth flow coordinates across two requests: start yields an authorize URL
     and invokes connect_server with `auth=`; the callback resolves the suspended
     connect and the tokens land 0600.

No network: the provider and the bridge are faked at the network edge exactly as
the model/embedder/telegram are elsewhere. The fake bridge *drives* the OAuth
handlers the way `streamablehttp_client(auth=...)` would, so what is under test is
the coordination this module owns, not Notion's token endpoint.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from daemon.admin import mcp_oauth
from daemon.app import create_app
from daemon.config import Settings
from daemon.mcp_catalog import lookup
from daemon.tools.mcp import ServerConfig, load_config, save_server, server_config_from_catalog


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(_env_file=None, preset="offline", data_dir=tmp_path, **kw)


class FakeBridge:
    """A stand-in for `McpBridge` at the network edge.

    Records how connect/disconnect were called, and for an OAuth connect drives the
    provider's redirect/callback handlers and persists a token - the coordination a
    real `streamablehttp_client(auth=...)` performs, minus the socket.
    """

    def __init__(self) -> None:
        self.failures: dict[str, str] = {}
        self.connected: list[tuple[ServerConfig, str | None, Any]] = []
        self.disconnected: list[str] = []
        self.fail_connect = False

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.connected.append((config, secret, auth))
        if self.fail_connect:
            self.failures[config.name] = "the server refused the connection"
            raise RuntimeError("the server refused the connection")
        if auth is not None:
            # What the SDK's streamable-HTTP client does when it hits a 401: hand the
            # authorize URL to the redirect handler, wait for the code+state on the
            # callback, then exchange and store the token.
            await auth.redirect_handler(
                "https://auth.example/authorize?state=STATE-XYZ&client_id=abc"
            )
            code, _state = await auth.callback_handler()
            from mcp.shared.auth import OAuthToken

            await auth.storage.set_tokens(
                OAuthToken(access_token=f"tok-{code}", token_type="Bearer")
            )
        return 3

    async def disconnect_server(self, name: str) -> None:
        self.disconnected.append(name)

    async def aclose(self) -> None:  # the lifespan teardown calls this
        return None


def _fake_build_provider(
    server_url: str,
    redirect_uri: str,
    storage: Any,
    redirect_handler: Any,
    callback_handler: Any,
) -> SimpleNamespace:
    """The provider the fake bridge drives - it just carries the handlers and the
    token storage the coordination needs, no SDK and no network."""
    return SimpleNamespace(
        server_url=server_url,
        redirect_uri=redirect_uri,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


@pytest.fixture(autouse=True)
def _clear_flows() -> Any:
    """The pending-flow registry is module-level; keep tests independent of order."""
    mcp_oauth._FLOWS.clear()
    yield
    mcp_oauth._FLOWS.clear()


def _enabled_app(tmp_path: Path, bridge: FakeBridge | None = None) -> Any:
    """An app with MCP on, a fake bridge on `app.state.mcp`, and `.env` pointed at
    the temp dir - assembled without the lifespan the way the Phase 1 tests do."""
    app = create_app(_settings(tmp_path, mcp_enabled=True))
    app.state.mcp = bridge if bridge is not None else FakeBridge()
    app.state.env_path = tmp_path / ".env"
    return app


# --- a. every route is gated on the switch -----------------------------------


def test_mcp_routes_are_409_when_disabled(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))  # mcp_enabled defaults False
    app.state.env_path = tmp_path / ".env"
    client = TestClient(app)

    assert client.get("/admin/api/mcp/catalog").status_code == 409
    assert client.get("/admin/api/mcp/servers").status_code == 409
    assert client.post("/admin/api/mcp/connect", json={"name": "fetch"}).status_code == 409
    assert client.delete("/admin/api/mcp/servers/fetch").status_code == 409
    assert client.post("/admin/api/mcp/oauth/start", json={"name": "notion"}).status_code == 409
    # And the 409 names the setting that turns it on, not a bare code.
    assert "DAEMON_MCP_ENABLED" in client.get("/admin/api/mcp/catalog").json()["detail"]


# --- b. the catalog is listed for the UI, without command/url ----------------


def test_catalog_lists_entries_without_leaking_commands(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path))
    body = client.get("/admin/api/mcp/catalog").json()
    by_name = {c["name"]: c for c in body["catalog"]}

    assert by_name["fetch"]["auth"] == "none"
    assert by_name["fetch"]["needs_key"] is False
    assert by_name["tavily"]["auth"] == "key"
    assert by_name["tavily"]["needs_key"] is True
    assert by_name["notion"]["auth"] == "oauth"
    # 2b has not confirmed a live flow, so the one-click stays off (reality gate).
    assert by_name["notion"]["oauth_verified"] is False
    # The command a server runs is a code constant, never sent to the browser.
    for card in body["catalog"]:
        assert "command" not in card and "url" not in card and "args" not in card


# --- c/d. connect: persist first, key handling -------------------------------


def test_connect_keyless_saves_then_connects(tmp_path: Path) -> None:
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge))

    resp = client.post("/admin/api/mcp/connect", json={"name": "fetch"})
    assert resp.status_code == 200
    assert resp.json()["connected"] is True

    # Persisted to mcp.json, and connected with no secret.
    assert "fetch" in {c.name for c in load_config(tmp_path).servers}
    config, secret, auth = bridge.connected[0]
    assert config.name == "fetch" and secret is None and auth is None


def test_connect_key_server_writes_env_then_passes_secret(tmp_path: Path) -> None:
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge))

    resp = client.post(
        "/admin/api/mcp/connect", json={"name": "tavily", "secret": "tvly-abc123"}
    )
    assert resp.status_code == 200

    # The value is in .env under the catalog's key_env, and mcp.json holds only the
    # name (server_config_from_catalog carries key_env, never the value).
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "TAVILY_API_KEY=tvly-abc123" in env_text
    saved = {c.name: c for c in load_config(tmp_path).servers}["tavily"]
    assert saved.key_env == "TAVILY_API_KEY"
    # Passed explicitly: a fresh .env is not in this process's environment.
    _config, secret, _auth = bridge.connected[0]
    assert secret == "tvly-abc123"


def test_connect_key_server_needs_a_secret(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path))
    resp = client.post("/admin/api/mcp/connect", json={"name": "tavily"})
    assert resp.status_code == 400
    # Nothing written when the secret is missing.
    assert not (tmp_path / "mcp.json").exists()


def test_connect_rejects_unknown_and_oauth_names(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path))
    assert client.post("/admin/api/mcp/connect", json={"name": "nope"}).status_code == 400
    # OAuth goes through oauth/start, not here.
    resp = client.post("/admin/api/mcp/connect", json={"name": "notion"})
    assert resp.status_code == 400
    assert "oauth" in resp.json()["detail"].lower()


def test_a_failed_connect_leaves_the_entry_persisted(tmp_path: Path) -> None:
    bridge = FakeBridge()
    bridge.fail_connect = True
    client = TestClient(_enabled_app(tmp_path, bridge))

    resp = client.post("/admin/api/mcp/connect", json={"name": "fetch"})
    assert resp.status_code == 502
    assert resp.json()["connected"] is False
    # Persisted before the connect, so it shows as configured-but-not-connected.
    assert "fetch" in {c.name for c in load_config(tmp_path).servers}


# --- servers listing ----------------------------------------------------------


def test_servers_reports_connection_state(tmp_path: Path) -> None:
    save_server(tmp_path, server_config_from_catalog(lookup("fetch")))
    save_server(tmp_path, server_config_from_catalog(lookup("tavily")))
    bridge = FakeBridge()
    bridge.failures["tavily"] = "bad key"
    client = TestClient(_enabled_app(tmp_path, bridge))

    servers = {s["name"]: s for s in client.get("/admin/api/mcp/servers").json()["servers"]}
    assert servers["fetch"]["connected"] is True
    assert servers["fetch"]["reason"] is None
    assert servers["tavily"]["connected"] is False
    assert servers["tavily"]["reason"] == "bad key"


# --- delete -------------------------------------------------------------------


def test_delete_removes_and_disconnects_idempotently(tmp_path: Path) -> None:
    save_server(tmp_path, server_config_from_catalog(lookup("fetch")))
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge))

    first = client.delete("/admin/api/mcp/servers/fetch")
    assert first.status_code == 200 and first.json()["removed"] is True
    assert "fetch" not in {c.name for c in load_config(tmp_path).servers}
    assert bridge.disconnected == ["fetch"]

    # Idempotent: removing it again is a no-op, still 200.
    second = client.delete("/admin/api/mcp/servers/fetch")
    assert second.status_code == 200 and second.json()["removed"] is False


# --- e. the OAuth bridge, driven directly (the coordination this module owns) --


async def test_oauth_flow_start_then_callback_persists_token_0600(tmp_path: Path) -> None:
    entry = lookup("notion")
    bridge = FakeBridge()

    authorize_url = await mcp_oauth.start_oauth_flow(
        bridge,
        tmp_path,
        entry,
        redirect_uri="http://127.0.0.1:8787/admin/api/mcp/oauth/callback",
        build_provider=_fake_build_provider,
    )
    # start returns the URL the browser must open, and connect was invoked with the
    # provider as `auth=` (the 2b seam), not as a bearer secret.
    assert "state=STATE-XYZ" in authorize_url
    _config, secret, auth = bridge.connected[0]
    assert secret is None and auth is not None

    name = await mcp_oauth.complete_oauth_flow("the-code", "STATE-XYZ")
    assert name == "notion"

    token_file = tmp_path / "mcp_tokens" / "notion.json"
    assert token_file.exists()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600, "token store is not 0600"
    stored = json.loads(token_file.read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "tok-the-code"
    # The completed flow is cleaned out of the registry.
    assert "STATE-XYZ" not in mcp_oauth._FLOWS


async def test_complete_oauth_flow_rejects_unknown_state() -> None:
    with pytest.raises(mcp_oauth.OAuthError):
        await mcp_oauth.complete_oauth_flow("code", "not-a-real-state")


async def test_start_oauth_flow_surfaces_a_connect_failure(tmp_path: Path) -> None:
    bridge = FakeBridge()
    bridge.fail_connect = True
    with pytest.raises(mcp_oauth.OAuthError):
        await mcp_oauth.start_oauth_flow(
            bridge,
            tmp_path,
            lookup("notion"),
            redirect_uri="http://127.0.0.1:8787/admin/api/mcp/oauth/callback",
            build_provider=_fake_build_provider,
        )


# --- the OAuth routes over HTTP (persistent loop for the cross-request flow) ---


def test_oauth_routes_bridge_two_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_oauth, "_build_provider", _fake_build_provider)
    app = create_app(_settings(tmp_path, mcp_enabled=True))
    app.state.env_path = tmp_path / ".env"

    # `with` keeps one event loop for the client's lifetime, so the connect task the
    # start request suspends survives until the callback request resolves it.
    with TestClient(app) as client:
        app.state.mcp = FakeBridge()

        start = client.post("/admin/api/mcp/oauth/start", json={"name": "notion"})
        assert start.status_code == 200
        assert "state=STATE-XYZ" in start.json()["authorize_url"]

        done = client.get("/admin/api/mcp/oauth/callback?code=xyz&state=STATE-XYZ")
        assert done.status_code == 200
        assert done.headers["content-type"].startswith("text/html")
        assert "connected" in done.text.lower()

    assert (tmp_path / "mcp_tokens" / "notion.json").exists()


def test_oauth_start_rejects_a_non_oauth_name(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path))
    resp = client.post("/admin/api/mcp/oauth/start", json={"name": "fetch"})
    assert resp.status_code == 400
    assert "not an OAuth server" in resp.json()["detail"]


def test_oauth_callback_reports_unknown_state_in_html(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path))
    resp = client.get("/admin/api/mcp/oauth/callback?code=x&state=stale")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "failed" in resp.text.lower()
