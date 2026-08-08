"""OAuth for MCP servers, bridged across two HTTP requests.

docs/design/2026-08-07-m5-admin-web-design.md, "OAuth (Phase 2b)". The SDK's
`OAuthClientProvider` drives an *inline* flow: on a 401 it discovers the server's
metadata, dynamically registers a client (RFC 7591), builds an authorize URL and
hands it to a `redirect_handler`, then suspends inside a `callback_handler`
awaiting the redirect back with `code` + `state`. That shape assumes one process
that can both open a browser and receive the callback - which a CLI or a chat
frontend cannot honestly do, and is exactly why earlier grillings deferred OAuth.
A loopback web admin *can*: the callback is one more route on the same server.

So this module bridges that one inline flow across two HTTP requests. `start` kicks
off `bridge.connect_server(config, auth=provider)` as a background task; the
provider suspends at `callback_handler`, `start` returns the authorize URL the
`redirect_handler` produced, and the browser follows it. The provider's redirect
carries `code`+`state` back to the `callback` route, which resolves the future the
suspended `callback_handler` is awaiting; the connect then completes and the SDK
persists the tokens through `FileTokenStorage`. Pending flows are keyed by the
OAuth `state` - the one value that survives the round-trip through the provider and
comes back on the redirect - so the callback can find the flow it belongs to.

Tokens are at-rest 0600 under the data dir, never in a service unit file - the same
rule `daemon/service.py` and the bearer-key path follow. The SDK handles refresh on
its own once the storage exists.

The `mcp` SDK is imported lazily (inside `_build_provider` and `FileTokenStorage`'s
methods) so importing this module - which the admin router does at process start -
never requires the optional `mcp` extra; a flow is only ever *started* when MCP is
enabled and a catalog entry is `auth="oauth"`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from daemon.fs import write_private_replace
from daemon.mcp_catalog import CatalogEntry
from daemon.tools.mcp import server_config_from_catalog

logger = logging.getLogger(__name__)

TOKENS_DIRNAME = "mcp_tokens"
"""Under the data dir, one JSON file per server, 0600. Beside `mcp.json` rather
than inside it because `mcp.json` is the shareable, secret-free config and these
files are the secret."""

START_TIMEOUT = 30.0
"""How long `start` waits for the provider to produce an authorize URL. Discovery
plus dynamic registration is a couple of round-trips; past this the server is not
answering and returning is better than a request that hangs open."""


class OAuthError(RuntimeError):
    """A flow that could not start or complete. Its message is safe to return to
    the client - it names the failure, never a token."""


class FileTokenStorage:
    """A `mcp.client.auth.TokenStorage` persisting tokens and the registered client
    at 0600 under the data dir.

    Not declared as inheriting `TokenStorage`: it is a `Protocol`, so structural
    typing is enough and the class definition then needs no import of the optional
    `mcp` extra. The token and client-info models *are* imported, but lazily, inside
    the methods the SDK calls - reached only once a flow is actually running.
    """

    def __init__(self, data_dir: Path, name: str) -> None:
        self._path = Path(data_dir) / TOKENS_DIRNAME / f"{name}.json"

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        # One atomic 0600 write for both halves, the same durable writer the rest of
        # the data dir uses. `set_tokens` and `set_client_info` are called at
        # different points in the flow, so each merges into what is already there.
        write_private_replace(
            self._path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )

    async def get_tokens(self) -> Any:
        from mcp.shared.auth import OAuthToken

        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: Any) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write(data)

    async def get_client_info(self) -> Any:
        from mcp.shared.auth import OAuthClientInformationFull

        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: Any) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)


@dataclass
class PendingFlow:
    """One suspended connect, waiting for its redirect to come back.

    `future` is what the provider's `callback_handler` awaits; the callback route
    resolves it with `(code, state)`. `ready` is set once the authorize URL exists
    (or the connect failed before producing one), which is what `start` waits on.
    """

    name: str
    ready: asyncio.Event
    future: asyncio.Future[tuple[str, str | None]]
    authorize_url: str = ""
    error: Exception | None = None
    task: asyncio.Task[None] | None = None


_FLOWS: dict[str, PendingFlow] = {}
"""Pending flows keyed by OAuth `state`. Module-level because the two requests
that make up one flow (`start` and `callback`) do not share anything else, and the
`state` is the only value that survives the round-trip through the provider."""


def _state_of(url: str) -> str | None:
    """The `state` query parameter of an authorize URL, or None. That is the key
    the callback will arrive with, so it is how the flow is registered."""
    values = parse_qs(urlsplit(url).query).get("state")
    return values[0] if values else None


def _build_provider(
    server_url: str,
    redirect_uri: str,
    storage: FileTokenStorage,
    redirect_handler: Any,
    callback_handler: Any,
) -> Any:
    """The real `OAuthClientProvider`. A module-level function so a test can replace
    it (monkeypatch) with a fake that captures the handlers, keeping the network at
    the edge exactly as the model/embedder/telegram fakes do - the coordination
    around it is what these routes get wrong, not the SDK's own token exchange."""
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    metadata = OAuthClientMetadata(
        client_name="Daemon admin",
        redirect_uris=[redirect_uri],  # type: ignore[list-item]  # pydantic coerces str -> AnyUrl
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


async def start_oauth_flow(
    bridge: Any,
    data_dir: Path,
    entry: CatalogEntry,
    *,
    redirect_uri: str,
    build_provider: Any = None,
) -> str:
    """Begin an OAuth connect and return the URL the browser must visit.

    Kicks `bridge.connect_server(config, auth=provider)` off as a background task;
    it suspends inside the provider's `callback_handler` until `complete_oauth_flow`
    resolves the future. Returns as soon as the `redirect_handler` has produced the
    authorize URL, or raises `OAuthError` if the connect fails before it does (a
    server that is unreachable, or refuses dynamic registration).

    `build_provider` defaults to the module's real one; the parameter, and the
    `or _build_provider` fallthrough (a live global lookup, so a monkeypatch on the
    module attribute wins), are the seam a test drives without a network.
    """
    build = build_provider or _build_provider
    config = server_config_from_catalog(entry)
    loop = asyncio.get_running_loop()
    flow = PendingFlow(name=entry.name, ready=asyncio.Event(), future=loop.create_future())
    storage = FileTokenStorage(data_dir, entry.name)
    registered_state: str | None = None

    async def redirect_handler(url: str) -> None:
        nonlocal registered_state
        flow.authorize_url = url
        state = _state_of(url)
        if state is None:
            # A provider that produced no state is one whose callback we could never
            # match, so this is a dead flow rather than one to register.
            flow.error = OAuthError("the authorization URL carried no state parameter")
        else:
            registered_state = state
            _FLOWS[state] = flow
        flow.ready.set()

    async def callback_handler() -> tuple[str, str | None]:
        return await flow.future

    provider = build(config.url, redirect_uri, storage, redirect_handler, callback_handler)

    async def connect() -> None:
        try:
            await bridge.connect_server(config, auth=provider)
        except Exception as exc:  # noqa: BLE001 - surfaced to whichever request is waiting
            flow.error = flow.error or exc
            logger.error("MCP OAuth connect for %r failed: %s", entry.name, exc)
        finally:
            # `start` may still be waiting (a failure before the redirect); release
            # it. And drop the flow from the registry now the connect is over, so a
            # completed or failed flow's state can never be resolved twice.
            if not flow.ready.is_set():
                flow.ready.set()
            if registered_state is not None:
                _FLOWS.pop(registered_state, None)

    flow.task = asyncio.create_task(connect(), name=f"mcp-oauth-{entry.name}")

    try:
        await asyncio.wait_for(flow.ready.wait(), timeout=START_TIMEOUT)
    except TimeoutError:
        flow.task.cancel()
        raise OAuthError(
            f"{entry.name}: timed out waiting for the authorization URL"
        ) from None

    if flow.error is not None:
        raise OAuthError(str(flow.error)) from flow.error
    if not flow.authorize_url:
        raise OAuthError(f"{entry.name}: no authorization URL was produced")
    return flow.authorize_url


async def complete_oauth_flow(code: str, state: str) -> str:
    """Resolve the pending flow for `state` and wait for its connect to finish.

    Returns the server name on success. Awaits the suspended connect so that by the
    time the callback route answers, the SDK has exchanged the code and persisted the
    tokens - the page can then honestly say the server is connected. Raises
    `OAuthError` for an unknown/expired state or a connect that then failed.
    """
    flow = _FLOWS.get(state)
    if flow is None:
        raise OAuthError(
            "no authorization is pending for this request; it may have expired or "
            "already completed"
        )
    if not flow.future.done():
        flow.future.set_result((code, state))
    if flow.task is not None:
        # connect() swallows its own exception into flow.error, so awaiting it here
        # cannot raise; it just lets the token exchange and 0600 persist complete.
        await flow.task
    if flow.error is not None:
        raise OAuthError(str(flow.error)) from flow.error
    return flow.name
