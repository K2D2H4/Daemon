"""The curated MCP catalog - a trusted code constant, never built from data.

CONTRACTS 13: code is never built from data. An MCP server is one more way to reach
the machine, so *what* the admin flow may install cannot come from a registry search
or any string a page wrote. It comes from here: a Python constant the owner's own
build shipped. The only thing that travels from outside is a `name` used to *look
up* an entry - the command, the URL, and the name of the env var holding a secret
are all fixed here and never assembled from user input.

Each entry is enough to build a `daemon.tools.mcp.ServerConfig` (see
`server_config_from_catalog`) without deciding anything at connect time:

  * `kind="uvx"` -> a stdio server run as `command` + `args` (a uv tool run).
  * `kind="url"` -> a streamable-HTTP server reached at `url`.
  * `key_env`    -> the NAME of the environment variable that holds this server's
                    secret. The value lives in `.env` (0600); mcp.json stores only
                    the name, so the config file stays shareable.
  * `auth`       -> how the secret is presented: `none`, an API `key` (a bearer
                    header for url servers, one env var for stdio), or `oauth`.
  * `oauth_verified` -> True only for a server we have actually confirmed supports
                    dynamic client registration (RFC 7591) and a localhost redirect.
                    False until 2b earns it - a premature True would one-click-expose
                    a flow that then fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["uvx", "url"]
Auth = Literal["none", "key", "oauth"]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: str
    kind: Kind
    description: str
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    key_env: str | None = None
    auth: Auth = "none"
    oauth_verified: bool = False


CATALOG: tuple[CatalogEntry, ...] = (
    # --- keyless uvx servers (the reference servers, run via `uvx`) -----------
    CatalogEntry(
        name="fetch",
        kind="uvx",
        description="Fetch a URL and return its content as markdown.",
        command="uvx",
        args=("mcp-server-fetch",),
        auth="none",
    ),
    CatalogEntry(
        name="time",
        kind="uvx",
        description="Current time and timezone conversion.",
        command="uvx",
        args=("mcp-server-time",),
        auth="none",
    ),
    CatalogEntry(
        name="git",
        kind="uvx",
        description="Read and search a local git repository.",
        command="uvx",
        args=("mcp-server-git",),
        auth="none",
    ),
    CatalogEntry(
        name="sqlite",
        kind="uvx",
        description="Query a local SQLite database.",
        command="uvx",
        args=("mcp-server-sqlite",),
        auth="none",
    ),
    # --- an API-key (bearer) server -------------------------------------------
    CatalogEntry(
        name="tavily",
        kind="url",
        description="Tavily web search, authenticated with an API key.",
        url="https://mcp.tavily.com/mcp/",
        key_env="TAVILY_API_KEY",
        auth="key",
    ),
    # --- an OAuth server (2b installs it; 2a only lists it) --------------------
    CatalogEntry(
        name="notion",
        kind="url",
        description="Notion workspace access over OAuth.",
        url="https://mcp.notion.com/mcp",
        auth="oauth",
        # Not yet confirmed against a live dynamic-registration + localhost-redirect
        # flow. 2b flips this once it has.
        oauth_verified=False,
    ),
)


def lookup(name: str) -> CatalogEntry | None:
    """The catalog entry with this name, or None. The one thing a caller may pass
    from outside - a name to look up, never a command to run."""
    for entry in CATALOG:
        if entry.name == name:
            return entry
    return None
