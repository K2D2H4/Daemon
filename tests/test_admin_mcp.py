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

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from daemon.admin import mcp_oauth
from daemon.app import create_app
from daemon.config import Settings
from daemon.mcp_catalog import lookup
from daemon.tools.mcp import ServerConfig, load_config, save_server, server_config_from_catalog

# The admin router is loopback-only (routes.py `_loopback_only`); a real browser on
# 127.0.0.1 sends this Host, and `TestClient`'s default `testserver` is rejected.
LOOPBACK = "http://127.0.0.1"

OAUTH_REDIRECT = "http://127.0.0.1:8787/admin/api/mcp/oauth/callback"


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
        self.live: set[str] = set()
        """Names with a live session, the way the real bridge tracks `_sessions`.
        `/servers` reads `is_connected`, so a test states liveness here rather than
        inferring it from `failures`."""

    def is_connected(self, name: str) -> bool:
        return name in self.live

    def connected_names(self) -> tuple[str, ...]:
        return tuple(self.live)

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.connected.append((config, secret, auth))
        if self.fail_connect:
            self.failures[config.name] = "the server refused the connection"
            raise RuntimeError("the server refused the connection")
        self.live.add(config.name)
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
    app = create_app(_settings(tmp_path, mcp_enabled=False))  # MCP defaults on; turn it off
    app.state.env_path = tmp_path / ".env"
    client = TestClient(app, base_url=LOOPBACK)

    assert client.get("/admin/api/mcp/catalog").status_code == 409
    assert client.get("/admin/api/mcp/servers").status_code == 409
    assert client.post("/admin/api/mcp/connect", json={"name": "fetch"}).status_code == 409
    assert client.delete("/admin/api/mcp/servers/fetch").status_code == 409
    assert client.post("/admin/api/mcp/oauth/start", json={"name": "notion"}).status_code == 409
    # And the 409 names the setting that turns it on, not a bare code.
    assert "DAEMON_MCP_ENABLED" in client.get("/admin/api/mcp/catalog").json()["detail"]


# --- b. the catalog is listed for the UI, without command/url ----------------


def test_catalog_lists_entries_without_leaking_commands(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path), base_url=LOOPBACK)
    body = client.get("/admin/api/mcp/catalog").json()
    by_name = {c["name"]: c for c in body["catalog"]}

    assert by_name["fetch"]["auth"] == "none"
    assert by_name["fetch"]["needs_key"] is False
    assert by_name["tavily"]["auth"] == "key"
    assert by_name["tavily"]["needs_key"] is True
    assert by_name["notion"]["auth"] == "oauth"
    # notion's live DCR + localhost flow is confirmed (a token persists under
    # mcp_tokens/), so the one-click is on. Unconfirmed OAuth servers stay off.
    assert by_name["notion"]["oauth_verified"] is True
    # The command a server runs is a code constant, never sent to the browser.
    for card in body["catalog"]:
        assert "command" not in card and "url" not in card and "args" not in card


# --- c/d. connect: persist first, key handling -------------------------------


def test_connect_keyless_saves_then_connects(tmp_path: Path) -> None:
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

    resp = client.post("/admin/api/mcp/connect", json={"name": "fetch"})
    assert resp.status_code == 200
    assert resp.json()["connected"] is True

    # Persisted to mcp.json, and connected with no secret.
    assert "fetch" in {c.name for c in load_config(tmp_path).servers}
    config, secret, auth = bridge.connected[0]
    assert config.name == "fetch" and secret is None and auth is None


def test_connect_key_server_writes_env_then_passes_secret(tmp_path: Path) -> None:
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

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
    client = TestClient(_enabled_app(tmp_path), base_url=LOOPBACK)
    resp = client.post("/admin/api/mcp/connect", json={"name": "tavily"})
    assert resp.status_code == 400
    # Nothing written when the secret is missing.
    assert not (tmp_path / "mcp.json").exists()


def test_connect_rejects_unknown_and_oauth_names(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path), base_url=LOOPBACK)
    assert client.post("/admin/api/mcp/connect", json={"name": "nope"}).status_code == 400
    # OAuth goes through oauth/start, not here.
    resp = client.post("/admin/api/mcp/connect", json={"name": "notion"})
    assert resp.status_code == 400
    assert "oauth" in resp.json()["detail"].lower()


def test_a_failed_connect_leaves_the_entry_persisted(tmp_path: Path) -> None:
    bridge = FakeBridge()
    bridge.fail_connect = True
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

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
    bridge.live.add("fetch")  # fetch has a live session; tavily does not
    bridge.failures["tavily"] = "bad key"
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

    servers = {s["name"]: s for s in client.get("/admin/api/mcp/servers").json()["servers"]}
    assert servers["fetch"]["connected"] is True
    assert servers["fetch"]["reason"] is None
    assert servers["tavily"]["connected"] is False
    assert servers["tavily"]["reason"] == "bad key"


def test_servers_configured_but_never_connected_is_not_green(tmp_path: Path) -> None:
    """Bug 3: a server persisted in mcp.json that no connect ever succeeded for is in
    neither `live` nor `failures`. "not in failures" showed it falsely connected; the
    live-session check reports it honestly as not connected."""
    save_server(tmp_path, server_config_from_catalog(lookup("notion")))
    bridge = FakeBridge()  # nothing live, nothing failed
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

    servers = {s["name"]: s for s in client.get("/admin/api/mcp/servers").json()["servers"]}
    assert servers["notion"]["connected"] is False
    assert servers["notion"]["reason"] == "not connected"


# --- delete -------------------------------------------------------------------


def test_delete_removes_and_disconnects_idempotently(tmp_path: Path) -> None:
    save_server(tmp_path, server_config_from_catalog(lookup("fetch")))
    bridge = FakeBridge()
    client = TestClient(_enabled_app(tmp_path, bridge), base_url=LOOPBACK)

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

    result = await mcp_oauth.start_oauth_flow(
        bridge,
        tmp_path,
        entry,
        redirect_uri="http://127.0.0.1:8787/admin/api/mcp/oauth/callback",
        build_provider=_fake_build_provider,
    )
    # start returns the URL the browser must open, and connect was invoked with the
    # provider as `auth=` (the 2b seam), not as a bearer secret.
    assert result.connected is False
    assert result.authorize_url is not None and "state=STATE-XYZ" in result.authorize_url
    _config, secret, auth = bridge.connected[0]
    assert secret is None and auth is not None
    # Bug 1: the server is persisted to mcp.json before the connect, carrying
    # auth="oauth" so a restart knows to reconnect it through its stored token.
    saved = {c.name: c for c in load_config(tmp_path).servers}
    assert "notion" in saved and saved["notion"].auth == "oauth"

    name = await mcp_oauth.complete_oauth_flow("the-code", "STATE-XYZ")
    assert name == "notion"

    token_file = tmp_path / "mcp_tokens" / "notion.json"
    assert token_file.exists()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600, "token store is not 0600"
    stored = json.loads(token_file.read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "tok-the-code"
    # The completed flow is cleaned out of the registry.
    assert "STATE-XYZ" not in mcp_oauth._FLOWS


class _StoredTokenBridge(FakeBridge):
    """The second-connect path (bug 2): the provider already has a stored token, so
    the SDK connects without ever hitting a 401. `redirect_handler` never fires,
    `authorize_url` stays empty, and the connect simply finishes."""

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.connected.append((config, secret, auth))
        self.live.add(config.name)
        # No redirect: a valid stored token means the SDK connects straight through.
        return 3


async def test_oauth_stored_token_is_success_not_a_false_error(tmp_path: Path) -> None:
    """Bug 2: after the first auth, a second connect uses the stored token and returns
    without a redirect. `start_oauth_flow` must report that as connected, not raise
    the old "no authorization URL was produced"."""
    bridge = _StoredTokenBridge()
    result = await mcp_oauth.start_oauth_flow(
        bridge, tmp_path, lookup("notion"),
        redirect_uri=OAUTH_REDIRECT, build_provider=_fake_build_provider,
    )
    assert result.connected is True
    assert result.authorize_url is None
    # Persisted, and no dead flow or task left behind (no redirect ever registered).
    assert "notion" in {c.name for c in load_config(tmp_path).servers}
    assert mcp_oauth._FLOWS == {}


def test_oauth_start_route_reports_a_stored_token_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route turns a stored-token connect into {authorize_url: null,
    connected: true}, so the frontend refreshes instead of redirecting."""
    monkeypatch.setattr(mcp_oauth, "_build_provider", _fake_build_provider)
    app = create_app(_settings(tmp_path, mcp_enabled=True))
    app.state.env_path = tmp_path / ".env"
    with TestClient(app, base_url=LOOPBACK) as client:
        app.state.mcp = _StoredTokenBridge()
        start = client.post("/admin/api/mcp/oauth/start", json={"name": "notion"})
        assert start.status_code == 200
        body = start.json()
        assert body["authorize_url"] is None
        assert body["connected"] is True


async def test_reconnect_provider_handlers_raise_rather_than_block(tmp_path: Path) -> None:
    """The startup provider has no browser behind it, so both handlers must raise a
    clean 'reauthorize' message rather than blocking - a boot with a dead token fails
    fast rather than hanging on a redirect nobody can follow."""
    captured: dict[str, Any] = {}

    def build(
        url: str, redirect_uri: str, storage: Any, redirect_handler: Any, callback_handler: Any
    ) -> SimpleNamespace:
        captured["redirect"] = redirect_handler
        captured["callback"] = callback_handler
        return SimpleNamespace()

    config = server_config_from_catalog(lookup("notion"))
    mcp_oauth.build_reconnect_provider(
        tmp_path, config, redirect_uri=OAUTH_REDIRECT, build_provider=build
    )
    with pytest.raises(mcp_oauth.OAuthError):
        await captured["redirect"]("https://auth.example/authorize")
    with pytest.raises(mcp_oauth.OAuthError):
        await captured["callback"]()


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
    with TestClient(app, base_url=LOOPBACK) as client:
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
    client = TestClient(_enabled_app(tmp_path), base_url=LOOPBACK)
    resp = client.post("/admin/api/mcp/oauth/start", json={"name": "fetch"})
    assert resp.status_code == 400
    assert "not an OAuth server" in resp.json()["detail"]


def test_oauth_callback_reports_unknown_state_in_html(tmp_path: Path) -> None:
    client = TestClient(_enabled_app(tmp_path), base_url=LOOPBACK)
    resp = client.get("/admin/api/mcp/oauth/callback?code=x&state=stale")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "failed" in resp.text.lower()


# --- the OAuth flow does not leak tasks or registry entries (finding #4) -------
# `callback_handler` awaits the redirect with a timeout, and every error/return path
# in `start` cancels the connect task and evicts the state, so an abandoned flow, a
# stateless authorize URL, and a burst of starts all clean up after themselves.


class _StatelessBridge(FakeBridge):
    """Drives the OAuth handlers with an authorize URL that carries no `state`, the
    one the callback could never be matched to - so `start` must treat it as dead."""

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.connected.append((config, secret, auth))
        if auth is not None:
            await auth.redirect_handler("https://auth.example/authorize?client_id=abc")
            await auth.callback_handler()  # suspends until start cancels the task
        return 3


class _CountingBridge(FakeBridge):
    """A distinct `state` per flow, so several pending flows coexist in `_FLOWS`
    rather than colliding on one key."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.connected.append((config, secret, auth))
        if auth is not None:
            self._n += 1
            await auth.redirect_handler(f"https://auth.example/authorize?state=S{self._n}")
            await auth.callback_handler()
        return 3


async def test_an_abandoned_oauth_flow_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner opens the authorize URL, then closes the tab and never returns. The
    suspended connect must give up rather than hold its task and `_FLOWS` entry."""
    monkeypatch.setattr(mcp_oauth, "CALLBACK_TIMEOUT", 0.05)
    bridge = FakeBridge()

    await mcp_oauth.start_oauth_flow(
        bridge, tmp_path, lookup("notion"),
        redirect_uri=OAUTH_REDIRECT, build_provider=_fake_build_provider,
    )
    flow = mcp_oauth._FLOWS["STATE-XYZ"]
    task = flow.task
    assert task is not None

    await asyncio.sleep(0.2)  # past CALLBACK_TIMEOUT

    assert "STATE-XYZ" not in mcp_oauth._FLOWS, "an abandoned flow leaked its registry entry"
    assert task.done(), "an abandoned flow leaked its connect task"


async def test_a_stateless_authorize_url_cancels_the_connect_task(tmp_path: Path) -> None:
    """An authorize URL with no `state` is a flow whose callback can never be matched.
    `start` raises, and it must first cancel the task it left parked in the callback."""
    bridge = _StatelessBridge()
    with pytest.raises(mcp_oauth.OAuthError):
        await mcp_oauth.start_oauth_flow(
            bridge, tmp_path, lookup("notion"),
            redirect_uri=OAUTH_REDIRECT, build_provider=_fake_build_provider,
        )

    await asyncio.sleep(0.05)  # let the cancellation propagate
    lingering = [
        t for t in asyncio.all_tasks()
        if t.get_name() == "mcp-oauth-notion" and not t.done()
    ]
    assert not lingering, "the no-state error stranded the connect task"
    assert mcp_oauth._FLOWS == {}, "a dead flow was left in the registry"


async def test_pending_flows_do_not_grow_without_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of abandoned flows must not pile up in `_FLOWS` forever - the timeout
    reaps every one of them."""
    monkeypatch.setattr(mcp_oauth, "CALLBACK_TIMEOUT", 0.05)
    bridge = _CountingBridge()

    for _ in range(5):
        await mcp_oauth.start_oauth_flow(
            bridge, tmp_path, lookup("notion"),
            redirect_uri=OAUTH_REDIRECT, build_provider=_fake_build_provider,
        )
    # Each abandoned flow (its callback never arrives) reaps itself when
    # CALLBACK_TIMEOUT fires. Don't assert the intermediate "all 5 live" count: with
    # a 50ms timeout a slow runner reaps the early flows before the last one starts
    # (that raced on CI). Assert only the invariant the name promises - none linger -
    # polling generously so the reaper cannot lose the race.
    for _ in range(200):
        if not mcp_oauth._FLOWS:
            break
        await asyncio.sleep(0.02)
    assert mcp_oauth._FLOWS == {}, "abandoned flows were never reaped"


# --- the connect/disconnect lock serialises mcp.json writers (finding #6) ------


class _OrderBridge(FakeBridge):
    """Records the interleaving of its connect bodies. With the route-level lock the
    two persist+connect regions run one after the other; without it they overlap at
    the await and the `mcp.json` read-modify-write can lose an update."""

    def __init__(self) -> None:
        super().__init__()
        self.order: list[str] = []

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        self.order.append(f"start:{config.name}")
        await asyncio.sleep(0.02)
        self.order.append(f"end:{config.name}")
        self.connected.append((config, secret, auth))
        return 3


async def test_concurrent_connects_are_serialized(tmp_path: Path) -> None:
    bridge = _OrderBridge()
    app = _enabled_app(tmp_path, bridge)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=LOOPBACK) as client:
        await asyncio.gather(
            client.post("/admin/api/mcp/connect", json={"name": "fetch"}),
            client.post("/admin/api/mcp/connect", json={"name": "time"}),
        )

    assert bridge.order == ["start:fetch", "end:fetch", "start:time", "end:time"], (
        "the two persist+connect regions interleaved"
    )
    # Both landed in mcp.json - the serialised read-modify-write lost neither.
    assert {c.name for c in load_config(tmp_path).servers} == {"fetch", "time"}
