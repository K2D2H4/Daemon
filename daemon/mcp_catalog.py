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
  * `pin_mcp`     -> whether to launch a uvx server with the daemon's own `mcp`
                    pinned in (`--with mcp==<version>`). True for the reference
                    servers, which under-constrain `mcp` and die on a resolved-too-new
                    one. False for a server that brings its own resolved stack - a
                    `fastmcp`-based server (workspace, slack) needs a newer `mcp` than
                    the daemon pins, and the pin would make it fail to resolve.
  * `env_passthrough` -> NAMES of environment variables a stdio server may read from
                    Daemon's own environment. The stdio child sees an allowlist, not
                    `os.environ` (see `_stdio_env`), so a server that runs its own
                    OAuth (workspace reads GOOGLE_OAUTH_CLIENT_ID/SECRET) would
                    otherwise get nothing. Names only - a code constant, never a value;
                    the secret still lives in the environment, never in `mcp.json`.
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
    pin_mcp: bool = True
    env_passthrough: tuple[str, ...] = ()


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
    # --- GitHub's own remote server, authenticated with a PAT -----------------
    # GitHub's hosted MCP has no dynamic client registration (a DCR attempt 422s),
    # so the OAuth button cannot serve it; a Personal Access Token presented as a
    # bearer is the supported path. `key`, not `oauth`, is the honest mapping.
    CatalogEntry(
        name="github",
        kind="url",
        description="GitHub repos, issues and pull requests, via a Personal Access Token.",
        url="https://api.githubcopilot.com/mcp/",
        key_env="GITHUB_MCP_PAT",
        auth="key",
    ),
    # --- a fastmcp stdio server that carries its own Google OAuth --------------
    # `workspace-mcp` reaches Gmail/Calendar/Drive; it runs the Google OAuth flow
    # itself (the daemon presents no secret, so `auth="none"`). It reads
    # GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET from the environment - the
    # stdio child sees only an allowlist (see `_stdio_env`), so those two names are
    # declared in `env_passthrough`. The owner exports both, creates a Google Cloud
    # OAuth client, and completes a one-time browser consent before CONNECT succeeds.
    # `pin_mcp=False`: it needs a newer `mcp` (via fastmcp) than the daemon pins.
    # Permissions are pinned to the owner's decision: Gmail up to `send` (read,
    # organise, draft, send) but NOT `full`, so a compromised turn cannot permanently
    # delete mail (`--help` confirms the cumulative order ...drafts<send<full, and
    # the space-separated `SERVICE:LEVEL ...` form); Calendar `full` for creating
    # events; Drive `readonly`, its stated use.
    CatalogEntry(
        name="google",
        kind="uvx",
        description=(
            "Gmail (read/send, no permanent delete), Calendar (read/write) and Drive "
            "(read). Needs a Google Cloud OAuth client exported as "
            "GOOGLE_OAUTH_CLIENT_ID/SECRET and a one-time browser consent."
        ),
        command="uvx",
        args=("workspace-mcp", "--permissions", "gmail:send", "calendar:full", "drive:readonly"),
        auth="none",
        pin_mcp=False,
        env_passthrough=("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"),
    ),
    # --- a fastmcp stdio server, authenticated with a Slack bot token ----------
    # Slack's hosted server (mcp.slack.com) has no dynamic client registration
    # either, so this stdio server takes a bot token (xoxb-...) as an env secret.
    # `pin_mcp=False` for the same fastmcp reason as `google`.
    CatalogEntry(
        name="slack",
        kind="uvx",
        description="Slack channels and messages, via a bot token.",
        command="uvx",
        args=("--from", "workos-slack-mcp-server", "slack-mcp-server"),
        key_env="SLACK_BOT_TOKEN",
        auth="key",
        pin_mcp=False,
    ),
    # --- an OAuth server (verified against the live DCR + localhost flow) ------
    CatalogEntry(
        name="notion",
        kind="url",
        description="Notion workspace access over OAuth.",
        url="https://mcp.notion.com/mcp",
        auth="oauth",
        # Confirmed against a live dynamic-registration + localhost-redirect flow:
        # the admin OAuth button completes and persists a token under mcp_tokens/.
        # This is the server that earns the "verified" promise (the rest stay False).
        oauth_verified=True,
    ),
)


def lookup(name: str) -> CatalogEntry | None:
    """The catalog entry with this name, or None. The one thing a caller may pass
    from outside - a name to look up, never a command to run."""
    for entry in CATALOG:
        if entry.name == name:
            return entry
    return None
