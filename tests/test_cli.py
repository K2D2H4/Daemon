"""The `daemon` command: dispatch, and whether doctor actually finds problems.

Every test chdirs into tmp_path, so no developer `.env` leaks in and no real
service is touched. Nothing here reaches the network: the Ollama probe is a seam.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from daemon import cli
from daemon.config import Settings
from daemon.fs import DIR_MODE
from daemon.service import ServiceAction, ServiceStatus


@pytest.fixture(autouse=True)
def _sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAEMON_PRESET", "offline")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture
def reachable_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(settings: Settings) -> tuple[bool, str]:
        return True, settings.ollama_base_url

    monkeypatch.setattr(cli, "probe_ollama", probe)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir(mode=DIR_MODE)
    return path


class FakeService:
    """Stands in for the real Service; records which verb the CLI asked for."""

    def __init__(self, unit_path: Path) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._unit_path = unit_path

    def install(self, *, force: bool = False) -> ServiceAction:
        self.calls.append(("install", force))
        return ServiceAction(label="ai.daemon.default", unit_path=self._unit_path, applied=True)

    def uninstall(self) -> ServiceAction:
        self.calls.append(("uninstall", None))
        return ServiceAction(label="ai.daemon.default", unit_path=self._unit_path, applied=True)

    def status(self) -> ServiceStatus:
        self.calls.append(("status", None))
        return ServiceStatus(
            label="ai.daemon.default",
            unit_path=self._unit_path,
            installed=True,
            loaded=True,
            running=True,
        )


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeService:
    fake = FakeService(tmp_path / "ai.daemon.default.plist")
    monkeypatch.setattr(cli, "service_for", lambda settings: fake)
    return fake


# --- dispatch ----------------------------------------------------------------


def test_no_arguments_still_runs_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    # M1a's console script started the server with no arguments; plists in the
    # wild depend on that staying true.
    served: list[Settings] = []
    monkeypatch.setattr(cli, "_serve", lambda settings: served.append(settings) or 0)

    assert cli.main([]) == 0
    assert served[0].preset == "offline"


def test_run_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_serve", lambda settings: 0)

    assert cli.main(["run"]) == 0


def test_install_calls_install(service: FakeService) -> None:
    assert cli.main(["install"]) == 0
    assert service.calls == [("install", False)]


def test_install_force_is_passed_through(service: FakeService) -> None:
    cli.main(["install", "--force"])

    assert service.calls == [("install", True)]


def test_uninstall_calls_uninstall(service: FakeService) -> None:
    assert cli.main(["uninstall"]) == 0
    assert service.calls == [("uninstall", None)]


def test_status_calls_status(service: FakeService, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status"]) == 0
    assert service.calls == [("status", None)]
    assert "running:   True" in capsys.readouterr().out


def test_status_of_a_dead_service_is_a_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Dead(FakeService):
        def status(self) -> ServiceStatus:
            return ServiceStatus(
                label="ai.daemon.default",
                unit_path=tmp_path / "x.plist",
                installed=True,
                loaded=True,
                running=False,
            )

    monkeypatch.setattr(cli, "service_for", lambda settings: Dead(tmp_path / "x.plist"))

    assert cli.main(["status"]) == 1


def test_reindex_reports_what_it_repaired(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_reindex", lambda settings: 3)

    assert cli.main(["reindex"]) == 0
    assert "reindexed 3 message(s)" in capsys.readouterr().out


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["dance"])

    assert exit_info.value.code == 2


def test_a_broken_config_stops_a_command_that_needs_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DAEMON_PRESET", "cheap")

    assert cli.main(["status"]) == 2
    assert "unknown DAEMON_PRESET" in capsys.readouterr().err


# --- doctor ------------------------------------------------------------------


def test_doctor_is_happy_with_a_sound_install(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "[ok] config" in out
    assert "[ok] data dir" in out
    assert "[ok] ollama" in out
    assert "FAIL" not in out


def test_doctor_explains_a_config_it_cannot_even_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The one command that has to work on a broken configuration - anything else
    # would leave the user with a traceback and no idea which field is wrong.
    monkeypatch.setenv("DAEMON_PRESET", "cheap")

    assert cli.main(["doctor"]) == 1
    assert "[FAIL] config: unknown DAEMON_PRESET" in capsys.readouterr().out


def test_doctor_notices_a_missing_data_dir(
    reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 1
    assert "[FAIL] data dir" in capsys.readouterr().out


def test_doctor_notices_a_world_readable_data_dir(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir.chmod(0o755)

    assert cli.main(["doctor"]) == 1
    assert "readable by other local accounts" in capsys.readouterr().out


def test_doctor_notices_that_ollama_is_down(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def probe(settings: Settings) -> tuple[bool, str]:
        return False, settings.ollama_base_url

    monkeypatch.setattr(cli, "probe_ollama", probe)

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] ollama" in out
    # Embeddings are local in every preset, so this breaks recall even for a
    # fully hosted setup. The message has to say so.
    assert "bge-m3" in out


def test_doctor_refuses_a_database_from_a_newer_build(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from daemon.app import DB_FILENAME

    conn = sqlite3.connect(data_dir / DB_FILENAME)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_version VALUES (99, '2026-08-03T00:00:00Z')")
    conn.commit()
    conn.close()

    assert cli.main(["doctor"]) == 1
    assert "is v99" in capsys.readouterr().out


def test_doctor_accepts_a_data_dir_with_no_database_yet(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # A fresh install has no sqlite file, and that is not a fault: the markdown is
    # the source of truth and the mirror is built on first run.
    assert cli.main(["doctor"]) == 0
    assert "no database yet" in capsys.readouterr().out
