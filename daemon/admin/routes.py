"""The admin JSON API and its server-rendered shell.

Endpoints (docs/design/2026-08-07-m5-admin-web-design.md, "JSON API"):

    GET   /admin/                 the shell page (static, self-contained, offline)
    GET   /admin/api/health       the same body as /health
    POST  /admin/api/chat-test    {text} -> {reply}; side-effect free
    GET   /admin/api/settings     editable settings, secrets masked
    PATCH /admin/api/settings     validate -> write .env -> {restart_required, supervised}
    POST  /admin/api/restart      graceful exit, only when supervised

    --- Phase 2, all behind DAEMON_MCP_ENABLED (409 with guidance when off) ---
    GET    /admin/api/mcp/catalog          the trusted catalog (no commands/urls)
    GET    /admin/api/mcp/servers          configured servers + connection state
    POST   /admin/api/mcp/connect          {name, secret?} -> persist then connect
    DELETE /admin/api/mcp/servers/{name}   unregister + disconnect (idempotent)
    POST   /admin/api/mcp/oauth/start      {name} -> {authorize_url}
    GET    /admin/api/mcp/oauth/callback   code+state -> finish connect, persist token

The chat test is the load-bearing one. It calls the gateway directly with
`Task.CHAT_TEXT` and **no tools**, records nothing, embeds nothing, and never
claims owner origin (decision 2). That is not a shortcut - with no auth (decision
1) the web cannot honestly prove owner origin, and a tool path reachable from
127.0.0.1 is exactly the CONTRACTS-10 hole the origin gate exists to close. So
the path simply does not exist here.

The MCP routes stay honest about the same boundary a different way: they install
and connect servers, but every tool a server offers is registered `guarded` by the
engine (daemon/tools/mcp.py) and still passes the origin gate, so nothing the web
adds here can run on a turn that is not the owner's (CONTRACTS 10, 12).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from daemon.admin.mcp_oauth import OAuthError, complete_oauth_flow, start_oauth_flow
from daemon.admin.restart import is_supervised, schedule_exit
from daemon.admin.settings_io import (
    PatchError,
    apply_patch,
    current_settings_payload,
    write_env_secret,
)
from daemon.app import health_payload
from daemon.llm.base import Message, ProviderError
from daemon.mcp_catalog import CATALOG, lookup
from daemon.tasks import Task
from daemon.tools.mcp import (
    load_config,
    remove_server,
    save_server,
    server_config_from_catalog,
)

SHELL = Path(__file__).parent / "static" / "index.html"

CHAT_TEST_TIMEOUT = 60.0
"""Ceiling on the chat-test round-trip. A hung provider must not hold the admin
request open forever - past this the answer is a 504 that says so, not a socket that
never closes (finding #7)."""

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
"""The only Host/Origin hostnames the admin serves. Anything else is either a
DNS-rebinding attempt or a genuinely remote caller, and the admin is loopback-only
by design (docs/design decision 1)."""

_UNSAFE_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})
_SAFE_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


def _hostname(authority: str) -> str:
    """The bare host of a `Host`/`Origin` authority, port and IPv6 brackets removed.

    `127.0.0.1:8787` -> `127.0.0.1`; `[::1]:8787` -> `::1`; `localhost` -> `localhost`.
    """
    authority = authority.strip()
    if not authority:
        return ""
    if authority.startswith("["):  # `[::1]` or `[::1]:8787`
        return authority[1:].split("]", 1)[0]
    return authority.rsplit(":", 1)[0] if ":" in authority else authority


async def _loopback_only(request: Request) -> None:
    """Reject CSRF and DNS-rebinding on the no-auth loopback admin (finding #1).

    The admin deliberately has no auth: local is owner (docs/design decision 1). That
    makes two browser attacks reachable, and this router-level guard closes both:

      * DNS-rebinding - a page whose domain the attacker has re-pointed at 127.0.0.1
        can talk to us same-origin. The defence is the `Host` header: the browser
        still sends the attacker's name, and we serve only loopback hostnames.
      * cross-site CSRF - a *simple* request (text/plain body, no custom header) skips
        the CORS preflight, so its side effect lands unseen. For unsafe methods we
        refuse a request the browser labels cross-site (`Sec-Fetch-Site`) or whose
        `Origin` is not loopback. The admin's own same-origin fetches carry neither
        signal against them, so the served page keeps working.
    """
    if _hostname(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail=(
                "the admin is loopback-only; this request's Host is not a loopback "
                "address (a DNS-rebinding attempt looks exactly like this)."
            ),
        )
    if request.method not in _UNSAFE_METHODS:
        return
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site not in _SAFE_FETCH_SITES:
        raise HTTPException(
            status_code=403,
            detail=(
                f"cross-site request refused (Sec-Fetch-Site: {fetch_site}); the "
                "admin accepts writes only from its own page."
            ),
        )
    origin = request.headers.get("origin")
    if origin and _hostname(urlsplit(origin).netloc) not in LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="cross-origin request refused; the Origin is not a loopback address.",
        )


router = APIRouter(prefix="/admin", dependencies=[Depends(_loopback_only)])


@router.get("/", response_class=HTMLResponse)
async def shell() -> HTMLResponse:
    """The one page. Read from disk each request rather than cached in memory:
    it is one small file, and a running daemon should serve an edited admin
    without a restart of its own."""
    return HTMLResponse(SHELL.read_text(encoding="utf-8"))


@router.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    return health_payload(request.app.state, request.app.state.settings)


@router.post("/api/chat-test")
async def chat_test(request: Request) -> JSONResponse:
    """A routed provider round-trip and nothing else - the chat version of /health.

    No persona, no recall, no history, no tools, no write. Just: is the model this
    preset routes `chat_text` to reachable, and what does it say back? Everything
    that would make it a conversation is deliberately absent (see module docstring).
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"detail": "body must be valid JSON"}, status_code=400)
    text = str(body.get("text", "")).strip() if isinstance(body, dict) else ""
    if not text:
        return JSONResponse({"detail": "text is required"}, status_code=400)

    gateway = request.app.state.gateway
    try:
        completion = await asyncio.wait_for(
            gateway.complete(Task.CHAT_TEXT, [Message(role="user", content=text)]),
            timeout=CHAT_TEST_TIMEOUT,
        )
    except TimeoutError:
        # A hung provider would otherwise hold this request open forever (finding #7).
        return JSONResponse(
            {"detail": "the model did not respond in time"}, status_code=504
        )
    except ProviderError as exc:
        # An unreachable or misconfigured provider is exactly what this endpoint
        # exists to surface (a missing model, a bad key, ollama down). Return the
        # message as a 502 so the UI shows "model not found" rather than a 500
        # traceback - the diagnostic is the whole point of a chat health check.
        return JSONResponse({"detail": str(exc)}, status_code=502)
    return JSONResponse({"reply": completion.text})


@router.get("/api/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    payload = current_settings_payload(request.app.state.settings)
    payload["supervised"] = is_supervised()
    return payload


@router.patch("/api/settings")
async def patch_settings(request: Request) -> JSONResponse:
    """Validate current-plus-patch, and only then write the changed keys to `.env`.

    A rejected patch is a 400 whose body names the problem and whose side effect is
    nothing at all - the write is downstream of a successful `Settings`
    construction (daemon/admin/settings_io.py)."""
    try:
        patch = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"detail": "body must be valid JSON"}, status_code=400)
    if not isinstance(patch, dict):
        return JSONResponse({"detail": "body must be an object"}, status_code=400)

    try:
        result = apply_patch(patch, request.app.state.env_path)
    except PatchError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return JSONResponse(
        {"restart_required": bool(result.changed), "supervised": is_supervised()}
    )


@router.post("/api/restart")
async def restart() -> JSONResponse:
    """Exit gracefully so the supervisor revives us on the new config - but only
    if a supervisor exists. Otherwise say so plainly rather than killing a process
    nothing will bring back (decision 3)."""
    if not is_supervised():
        return JSONResponse(
            {
                "restarted": False,
                "supervised": False,
                "detail": (
                    "Not running under launchd or systemd, so exiting would just stop "
                    "the daemon. Install it as a service first: `daemon install`."
                ),
            },
            status_code=409,
        )
    schedule_exit()
    return JSONResponse({"restarted": True, "supervised": True})


# --- MCP (Phase 2), all behind DAEMON_MCP_ENABLED ----------------------------

MCP_OFF = (
    "MCP is off. Enable DAEMON_MCP_ENABLED on the settings page and restart the "
    "daemon, then this page can install servers."
)


def _mcp_off() -> JSONResponse:
    """The shared 409 for every MCP route when the switch is off.

    A 409 rather than a 404 because the routes exist and the feature is real - it is
    a conflict with the current configuration, and the body says which setting turns
    it on. The engine reads `mcp.json` only when this switch is set (config.py), so
    connecting while it is off would write a server nothing would ever load."""
    return JSONResponse({"detail": MCP_OFF}, status_code=409)


@router.get("/api/mcp/catalog")
async def mcp_catalog(request: Request) -> JSONResponse:
    """The trusted catalog, structured fields only.

    Deliberately not the command or the url: the front-end needs a name, a
    description, and enough about auth to draw the right button (a password field, a
    "Connect with OAuth" button, or a plain Connect). The command a server runs is a
    code constant (CONTRACTS 13) and nothing the browser needs to see."""
    if not request.app.state.settings.mcp_enabled:
        return _mcp_off()
    return JSONResponse(
        {
            "catalog": [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "auth": entry.auth,
                    "oauth_verified": entry.oauth_verified,
                    "needs_key": entry.key_env is not None,
                }
                for entry in CATALOG
            ]
        }
    )


@router.get("/api/mcp/servers")
async def mcp_servers(request: Request) -> JSONResponse:
    """The configured servers and whether each is actually connected.

    "Connected" is the bridge's *live session* for that name, not "absent from
    `failures`" - a persisted server that no connect ever succeeded for is in neither
    dict, and the old "not in failures" test showed it falsely green (bug 3). The
    failure reason still rides along for the not-connected ones, because a silently
    absent tool is indistinguishable from a model that chose not to use it."""
    if not request.app.state.settings.mcp_enabled:
        return _mcp_off()
    data_dir = request.app.state.settings.data_dir
    bridge = request.app.state.mcp
    failures: dict[str, str] = dict(getattr(bridge, "failures", {}) or {})
    servers = []
    for config in load_config(data_dir).servers:
        connected = bridge is not None and bridge.is_connected(config.name)
        servers.append(
            {
                "name": config.name,
                "remote": config.is_remote,
                "connected": connected,
                "reason": (
                    None
                    if connected
                    else failures.get(config.name)
                    or (
                        "not connected"
                        if bridge is not None
                        else "the MCP bridge is not running"
                    )
                ),
            }
        )
    return JSONResponse({"servers": servers})


@router.post("/api/mcp/connect")
async def mcp_connect(request: Request) -> JSONResponse:
    """Install and connect a catalog server. Persist first, then connect.

    Only a catalog `name` crosses from the client (CONTRACTS 13); the command, url
    and key-var name all come from the trusted entry. OAuth servers are refused here
    - they go through `oauth/start`. A key server needs its secret, which is written
    to `.env` under the entry's `key_env` and then passed to `connect_server`
    explicitly, because a freshly-written `.env` is not yet in this process's
    environment. `save_server` runs before the connect, so a connect that then fails
    leaves the entry showing as configured-but-not-connected rather than vanishing.
    """
    if not request.app.state.settings.mcp_enabled:
        return _mcp_off()
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"detail": "body must be valid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "body must be an object"}, status_code=400)
    name = str(body.get("name", "")).strip()
    entry = lookup(name)
    if entry is None:
        return JSONResponse(
            {"detail": f"no catalog server named {name!r}"}, status_code=400
        )
    if entry.auth == "oauth":
        return JSONResponse(
            {"detail": f"{name} uses OAuth; start it through /admin/api/mcp/oauth/start"},
            status_code=400,
        )

    bridge = request.app.state.mcp
    if bridge is None:
        return JSONResponse(
            {"detail": "the MCP bridge is not running"}, status_code=409
        )
    settings = request.app.state.settings

    secret: str | None = None
    if entry.auth == "key":
        raw = body.get("secret")
        secret = str(raw).strip() if raw is not None else ""
        if not secret:
            return JSONResponse(
                {"detail": f"{name} needs an API key"}, status_code=400
            )
        assert entry.key_env  # a key server always names its env var (catalog invariant)
        try:
            write_env_secret(request.app.state.env_path, entry.key_env, secret)
        except PatchError as exc:
            # A key carrying a newline would inject an `.env` line (finding #2).
            return JSONResponse({"detail": str(exc)}, status_code=400)

    config = server_config_from_catalog(entry)
    # Persist-then-connect, serialised against a concurrent disconnect (finding #6):
    # two tabs, or a double-click, would otherwise interleave the `mcp.json`
    # read-modify-write with `remove_server` and lose one update.
    async with request.app.state.mcp_persist_lock:
        # Persist before connecting (see docstring). The entry survives a failed connect.
        save_server(settings.data_dir, config)
        try:
            landed = await bridge.connect_server(config, secret=secret)
        except Exception as exc:  # noqa: BLE001 - the server, not us; reported to the client
            return JSONResponse(
                {
                    "detail": f"{name} was saved but did not connect: {exc}",
                    "connected": False,
                },
                status_code=502,
            )
    return JSONResponse({"name": name, "connected": True, "tools": landed})


@router.delete("/api/mcp/servers/{name}")
async def mcp_disconnect(request: Request, name: str) -> JSONResponse:
    """Remove a server from `mcp.json` and close its session. Idempotent - removing
    one that was never there, or disconnecting one that never connected, is a no-op,
    so a half-happened removal is safe to retry."""
    if not request.app.state.settings.mcp_enabled:
        return _mcp_off()
    # The same lock the connect route takes (finding #6), so persist+disconnect cannot
    # interleave with a concurrent persist+connect and lose an `mcp.json` update.
    async with request.app.state.mcp_persist_lock:
        removed = remove_server(request.app.state.settings.data_dir, name)
        bridge = request.app.state.mcp
        if bridge is not None:
            await bridge.disconnect_server(name)
    return JSONResponse({"removed": removed})


@router.post("/api/mcp/oauth/start")
async def mcp_oauth_start(request: Request) -> JSONResponse:
    """Begin an OAuth connect and return the URL the browser must open.

    The redirect comes back to this same loopback server - which is the whole reason
    OAuth is possible here and was not from the CLI (docs/design). The connect
    suspends until `oauth/callback` resolves it; this returns as soon as the
    authorize URL exists."""
    if not request.app.state.settings.mcp_enabled:
        return _mcp_off()
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"detail": "body must be valid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "body must be an object"}, status_code=400)
    name = str(body.get("name", "")).strip()
    entry = lookup(name)
    if entry is None:
        return JSONResponse(
            {"detail": f"no catalog server named {name!r}"}, status_code=400
        )
    if entry.auth != "oauth":
        return JSONResponse(
            {"detail": f"{name} is not an OAuth server; use /admin/api/mcp/connect"},
            status_code=400,
        )

    bridge = request.app.state.mcp
    if bridge is None:
        return JSONResponse(
            {"detail": "the MCP bridge is not running"}, status_code=409
        )
    settings = request.app.state.settings
    redirect_uri = (
        f"http://{settings.host}:{settings.port}/admin/api/mcp/oauth/callback"
    )
    # `start_oauth_flow` writes the server to mcp.json before connecting, so it takes
    # the same lock the key connect/disconnect routes do (finding #6) - two starts, or
    # a start racing a disconnect, must not interleave the read-modify-write.
    try:
        async with request.app.state.mcp_persist_lock:
            result = await start_oauth_flow(
                bridge, settings.data_dir, entry, redirect_uri=redirect_uri
            )
    except OAuthError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)
    # authorize_url null + connected true means the provider used a stored token and
    # connected without a browser step; the frontend refreshes instead of redirecting.
    return JSONResponse(
        {"authorize_url": result.authorize_url, "connected": result.connected}
    )


CALLBACK_PAGE = """<!doctype html><meta charset="utf-8">
<title>Daemon — {title}</title>
<style>body{{font-family:ui-monospace,Menlo,monospace;background:#120F18;color:#ECE7F5;
display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}}
.box{{max-width:32rem;padding:2rem;text-align:center}}
h1{{color:#A78BFA;font-size:1rem;letter-spacing:.05em}}p{{color:#8E85A3}}</style>
<div class="box"><h1>{heading}</h1><p>{message}</p></div>"""


@router.get("/api/mcp/oauth/callback", response_class=HTMLResponse)
async def mcp_oauth_callback(
    request: Request, code: str = "", state: str = ""
) -> HTMLResponse:
    """Where the OAuth provider sends the browser back. Resolve the pending flow so
    its suspended connect finishes and the tokens land 0600, then show a small page
    telling the owner to return to the admin tab. This is a browser redirect target,
    so it answers in HTML rather than JSON, and always 200 - the browser is a person,
    not the admin's fetch()."""
    if not request.app.state.settings.mcp_enabled:
        return HTMLResponse(
            CALLBACK_PAGE.format(
                title="MCP off",
                heading="MCP is off",
                message="Enable DAEMON_MCP_ENABLED and try again.",
            )
        )
    if not code or not state:
        return HTMLResponse(
            CALLBACK_PAGE.format(
                title="authorization failed",
                heading="Authorization failed",
                message="The provider returned no code. Nothing was connected.",
            )
        )
    try:
        name = await complete_oauth_flow(code, state)
    except OAuthError as exc:
        return HTMLResponse(
            CALLBACK_PAGE.format(
                title="authorization failed",
                heading="Authorization failed",
                message=str(exc),
            )
        )
    return HTMLResponse(
        CALLBACK_PAGE.format(
            title="connected",
            heading=f"{name} connected",
            message="You can close this tab and return to the Daemon admin.",
        )
    )
