"""The curated MCP catalog: a code constant, not data (CONTRACTS 13).

The catalog is the only thing the admin "add a server" flow trusts. It is a Python
constant so no untrusted string ever decides what command runs or where a request
goes - the values travel as structured fields, exactly as the origin-gate design
requires. These tests pin the shape the admin routes and the connect flow read.
"""

from __future__ import annotations

from daemon.mcp_catalog import CATALOG, CatalogEntry, lookup


def test_lookup_finds_an_entry_by_name() -> None:
    entry = lookup("fetch")
    assert entry is not None and entry.name == "fetch"


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


def test_oauth_entries_stay_unverified_until_actually_confirmed() -> None:
    """`oauth_verified` is a promise we made about a specific server's dynamic
    client registration and localhost redirect. Until 2b confirms one, it is False -
    a True we did not earn would one-click-expose a flow that then fails."""
    oauth_entries = [entry for entry in CATALOG if entry.auth == "oauth"]
    assert oauth_entries, "expected at least one OAuth server"
    for entry in oauth_entries:
        assert entry.oauth_verified is False


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
