"""The curated MCP catalog: a code constant, not data (CONTRACTS 13).

The catalog is the only thing the admin "add a server" flow trusts. It is a Python
constant so no untrusted string ever decides what command runs or where a request
goes - the values travel as structured fields, exactly as the origin-gate design
requires. These tests pin the shape the admin routes and the connect flow read.
"""

from __future__ import annotations

from importlib.metadata import version

from daemon.mcp_catalog import CATALOG, CatalogEntry, lookup
from daemon.tools.mcp import server_config_from_catalog


def test_lookup_finds_an_entry_by_name() -> None:
    entry = lookup("fetch")
    assert entry is not None and entry.name == "fetch"


def test_a_uvx_server_is_launched_against_the_daemons_own_mcp() -> None:
    """A bare `uvx <server>` resolves an `mcp` that can be too new for the reference
    server, which then dies at import. The catalog config pins the daemon's own mcp
    so the server actually starts - the pin rides as a plain argv prefix."""
    entry = lookup("time")
    assert entry is not None and entry.kind == "uvx"
    config = server_config_from_catalog(entry)
    assert config.args[:2] == ("--with", f"mcp=={version('mcp')}")
    assert "mcp-server-time" in config.args  # the tool itself still runs


def test_a_fastmcp_server_opts_out_of_the_mcp_pin() -> None:
    """A fastmcp-based server (google/slack) resolves a newer `mcp` than the daemon
    pins; forcing `--with mcp==<daemon version>` would make uvx fail to resolve rather
    than start. `pin_mcp=False` keeps the pin off so it launches as `uvx <server>`."""
    entry = lookup("google")
    assert entry is not None and entry.kind == "uvx" and entry.pin_mcp is False
    config = server_config_from_catalog(entry)
    assert "--with" not in config.args
    assert config.args[0] == "workspace-mcp"


def test_google_declares_the_oauth_env_it_reads() -> None:
    """`google` runs its own Google OAuth and reads GOOGLE_OAUTH_CLIENT_ID/SECRET; the
    stdio child sees only an allowlist, so the entry must name them in
    `env_passthrough` or the server can never receive its credentials. The mapping
    carries the names through to the ServerConfig the bridge starts."""
    entry = lookup("google")
    assert entry is not None
    assert entry.env_passthrough == ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET")
    config = server_config_from_catalog(entry)
    assert config.env_passthrough == ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET")


def test_google_runs_single_user_so_a_spawned_launch_reuses_the_cached_token() -> None:
    """The owner is the only account: `--single-user` makes the server load the one
    cached credential without session mapping, so a stdio launch the daemon spawns
    reuses a prior consent instead of trying (and failing) to authenticate headless."""
    entry = lookup("google")
    assert entry is not None
    assert "--single-user" in entry.args


def test_a_url_server_gets_no_uvx_pin() -> None:
    entry = lookup("tavily")
    assert entry is not None and entry.kind == "url"
    config = server_config_from_catalog(entry)
    assert "--with" not in config.args


def test_lookup_of_an_unknown_name_is_none() -> None:
    assert lookup("definitely-not-in-the-catalog") is None


def test_all_three_auth_kinds_are_represented() -> None:
    """The admin UI has to render each; a catalog with only keyless entries would
    leave the key and OAuth paths untested against anything real."""
    kinds = {entry.auth for entry in CATALOG}
    assert {"none", "key", "oauth"} <= kinds


def test_entry_names_are_unique() -> None:
    names = [entry.name for entry in CATALOG]
    assert len(names) == len(set(names))


def test_a_keyless_entry_names_no_env_var() -> None:
    for entry in CATALOG:
        if entry.auth == "none":
            assert entry.key_env is None, f"{entry.name} is keyless but names an env var"


def test_a_key_entry_names_the_env_var_that_holds_its_secret() -> None:
    """The secret lives in .env under this name; mcp.json only ever stores the name.
    A key server that named no variable would have nowhere to read the secret from."""
    key_entries = [entry for entry in CATALOG if entry.auth == "key"]
    assert key_entries, "expected at least one key/bearer server"
    for entry in key_entries:
        assert entry.key_env, f"{entry.name} is a key server but names no env var"


def test_oauth_entries_are_verified_only_when_confirmed() -> None:
    """`oauth_verified` is a promise about a specific server's dynamic client
    registration and localhost redirect, earned by actually completing the flow.
    `notion` was confirmed live (the admin button persists a token under
    mcp_tokens/). Any OAuth server we have NOT confirmed must stay False - a True we
    did not earn would one-click-expose a flow that then fails."""
    confirmed = {"notion"}
    oauth_entries = [entry for entry in CATALOG if entry.auth == "oauth"]
    assert oauth_entries, "expected at least one OAuth server"
    for entry in oauth_entries:
        expected = entry.name in confirmed
        assert entry.oauth_verified is expected, (
            f"{entry.name}: oauth_verified should be {expected}"
        )


def test_a_uvx_entry_carries_a_command_and_a_url_entry_a_url() -> None:
    for entry in CATALOG:
        if entry.kind == "uvx":
            assert entry.command and not entry.url, f"{entry.name} is uvx but has no command"
        else:
            assert entry.url and not entry.command, f"{entry.name} is url but has no url"


def test_a_catalog_entry_is_immutable() -> None:
    entry = CATALOG[0]
    import dataclasses

    assert dataclasses.is_dataclass(entry)
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.name = "mutated"  # type: ignore[misc]


def test_a_catalog_entry_is_a_CatalogEntry() -> None:
    assert all(isinstance(entry, CatalogEntry) for entry in CATALOG)
