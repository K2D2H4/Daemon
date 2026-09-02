"""Which Google accounts the settings page may suggest (docs/adr/0021).

Every case here is about failing *quietly and correctly*. This module reads
another project's directory to offer a convenience the google MCP server cannot
provide, so the one property that has to hold is that nothing it finds - or fails
to find - can break the settings page or reach the proactive path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daemon.admin.google_accounts import (
    DEFAULT_DIR,
    DIR_ENV,
    authenticated_accounts,
    credentials_dir,
)


def _store(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "credentials"
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_it_lists_the_accounts_by_credential_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live install's shape: one file per authenticated account, named for it,
    plus `oauth_states.json` which is not an account and must not be offered as
    one."""
    monkeypatch.setenv(
        DIR_ENV[0],
        str(_store(tmp_path, "owner@gmail.com.json", "work@example.co.kr.json",
                   "oauth_states.json")),
    )

    assert authenticated_accounts() == ["owner@gmail.com", "work@example.co.kr"]


def test_a_missing_directory_is_no_suggestions_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary state on an install with no google server. It has to render as
    a plain text field, not as a broken page."""
    monkeypatch.setenv(DIR_ENV[0], str(tmp_path / "nope"))

    assert authenticated_accounts() == []


def test_a_file_that_is_not_an_address_is_not_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matched by what to keep, not by filtering out the names known today - the
    same discipline `agenda._EVENT_RE` uses, so a file this version has never seen
    is ignored rather than suggested into a settings field."""
    monkeypatch.setenv(
        DIR_ENV[0],
        str(_store(tmp_path, "README.json", "notes.txt", "cache.json", "a@b.co.json")),
    )

    assert authenticated_accounts() == ["a@b.co"]


def test_a_path_shaped_filename_cannot_be_suggested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name is written into a settings field the owner may then save to `.env`.
    Nothing path-shaped gets that far, even though a real credential store would
    never contain one."""
    directory = _store(tmp_path, "a@b.co.json")
    nested = directory / "sub"
    nested.mkdir()
    (nested / "evil@x.co.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv(DIR_ENV[0], str(directory))

    assert authenticated_accounts() == ["a@b.co"], "it must not walk into subdirectories"


def test_the_override_env_vars_are_honoured_in_the_servers_own_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`workspace-mcp` reads both names, and an owner who moved the store must get
    the same picker as one who did not - otherwise an empty list is
    indistinguishable from "nothing is authenticated"."""
    first = _store(tmp_path / "one", "first@x.co.json")
    second = _store(tmp_path / "two", "second@x.co.json")

    monkeypatch.setenv(DIR_ENV[1], str(second))
    assert authenticated_accounts() == ["second@x.co"]

    monkeypatch.setenv(DIR_ENV[0], str(first))
    assert authenticated_accounts() == ["first@x.co"], f"{DIR_ENV[0]} must win"


def test_it_falls_back_to_the_servers_default_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in DIR_ENV:
        monkeypatch.delenv(name, raising=False)

    assert credentials_dir() == DEFAULT_DIR


def test_it_opens_no_credential_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names only. The files hold OAuth tokens, and this runs on a page with no
    auth in front of it (admin decision 1), so a token that is never read cannot
    leak from here."""
    directory = _store(tmp_path, "a@b.co.json")
    (directory / "a@b.co.json").chmod(0o000)
    monkeypatch.setenv(DIR_ENV[0], str(directory))

    try:
        assert authenticated_accounts() == ["a@b.co"], (
            "an unreadable file was still listed, which proves the name is all "
            "this reads"
        )
    finally:
        (directory / "a@b.co.json").chmod(0o600)
