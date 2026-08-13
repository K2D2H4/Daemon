"""The `daemon` command: dispatch, and whether doctor actually finds problems.

Every test chdirs into tmp_path, so no developer `.env` leaks in and no real
service is touched. Nothing here reaches the network: the Ollama probe is a seam,
and so is `daemon.app.build_reflection` (which would otherwise build a provider).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import FakeProvider

from daemon import app as daemon_app
from daemon import cli, macapp
from daemon.config import ConfigError, Route, Settings
from daemon.fs import DIR_MODE
from daemon.llm.gateway import LLMGateway
from daemon.memory.base import LoggedMessage
from daemon.memory.curated import CuratedMemory
from daemon.memory.entities import EntityNotes
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter
from daemon.persona.evolve import PersonaEvolution
from daemon.reflection import Reflection, artifact_path
from daemon.service import RunResult, Service, ServiceAction, ServiceStatus
from daemon.tasks import Task

DAY = "2026-08-03"
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
"""Midday UTC on purpose: `catch_up` compares against the *local* day, so a
timestamp near midnight would make the backlog days below "today" in some
timezones and not others."""

REPLY = json.dumps(
    {
        "facts": [{"body": "연희동에 산다", "importance": 8, "key": "home"}],
        "entities": [
            {
                "name": "지현",
                "kind": "person",
                "note": "연희동 카페에서 만났다",
                "links": ["연희동"],
            }
        ],
        "observations": [{"body": "아침에는 짧은 메시지가 낫다", "confidence": 0.7}],
    },
    ensure_ascii=False,
)


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
    # `_uninstall` builds the service via `cli.service_for`; `_install` goes through
    # the shared, .app-aware `daemon.macapp.build_resident_service`. Patch both so the
    # generic dispatch tests never construct a real Service.
    monkeypatch.setattr(cli, "service_for", lambda settings: fake)
    monkeypatch.setattr("daemon.macapp.build_resident_service", lambda settings: fake)
    # Pin the platform so the test is deterministic on every dev machine and never
    # takes the real `.app`/launchctl/mic path - and, for uninstall, never rmtrees a
    # developer's real ~/Applications/Daemon.app (tests/CLAUDE.md: no test may touch
    # a microphone, and none may delete real user files).
    monkeypatch.setattr(cli.sys, "platform", "linux")
    return fake


def _reflection_for(data_dir: Path, store: Store, provider: FakeProvider) -> Reflection:
    return Reflection(
        data_dir,
        store,
        LLMGateway({provider.name: provider}, {Task.REFLECTION: Route(provider.name, "m")}),
    )


def _logged_day(data_dir: Path, day: str) -> None:
    """One day of conversation, in the log and in the mirror.

    Both halves are needed and for different readers: the log file is what makes a
    day *pending*, the mirror row is what reflection actually reads.
    """
    log_dir = data_dir / "memory" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{day}.md").write_text(f"# {day}\n", encoding="utf-8")
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        store.insert_message(
            LoggedMessage(
                ts=NOW,
                role="user",
                content="연희동으로 이사했어",
                origin="owner",
                session_kind="interactive",
                modality="text",
                channel="telegram",
                sender_id="42",
            ),
            log_file=f"memory/log/{day}.md",
        )
    finally:
        store.close()


def _reflected_day(data_dir: Path, day: str = DAY) -> None:
    """A day the real pass has already been over, so doctor has something to report.

    The pass is the real one with a fake model behind it. Building the mirror by
    hand here would let doctor print a graph no write path can actually produce.
    """
    _logged_day(data_dir, day)
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        asyncio.run(_reflection_for(data_dir, store, FakeProvider(REPLY)).run(day))
    finally:
        store.close()


@pytest.fixture
def reflection_seam(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeProvider]:
    """Replace the one thing in `daemon reflect` that would reach a provider.

    `daemon.app.build_reflection` is the seam the scheduler and the CLI share, so
    patching it leaves the pass, the store, the markdown and the exit code real -
    only the model is a fake.
    """

    def install(reply: str = REPLY, *, fail: bool = False) -> FakeProvider:
        provider = FakeProvider(reply, fail=fail)

        async def build(settings: Settings) -> tuple[Reflection, Any]:
            store = Store.open(settings.data_dir / daemon_app.DB_FILENAME)

            async def close() -> None:
                store.close()

            return _reflection_for(settings.data_dir, store, provider), close

        monkeypatch.setattr(daemon_app, "build_reflection", build)
        return provider

    return install


def _add_observations(data_dir: Path, n: int, *, day: str = "2026-08-01") -> None:
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        for i in range(n):
            store.insert_observation(body=f"관찰 {i}", observed_from=day, now=NOW)
    finally:
        store.close()


@pytest.fixture
def persona_seam(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeProvider]:
    """Replace the one thing in `daemon persona evolve` that would reach a
    provider - same seam shape as `reflection_seam`, for the same reason:
    `daemon.app.build_persona_evolution` is what the Monday job and the CLI
    share, so patching it leaves the pass, the store and the exit code real.
    """

    def install(reply: str = '{"rules": []}', *, fail: bool = False, **kwargs: Any) -> FakeProvider:
        provider = FakeProvider(reply, fail=fail)

        async def build(settings: Settings) -> tuple[PersonaEvolution, Any]:
            store = Store.open(settings.data_dir / daemon_app.DB_FILENAME)

            async def close() -> None:
                store.close()

            gateway = LLMGateway(
                {provider.name: provider}, {Task.PERSONA_RULE: Route(provider.name, "m")}
            )
            evolution = PersonaEvolution(
                settings.data_dir, store, gateway, min_observations=1, **kwargs
            )
            return evolution, close

        monkeypatch.setattr(daemon_app, "build_persona_evolution", build)
        return provider

    return install


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


def test_version_flag_prints_the_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`daemon --version` is the first line of any bug report, so it must work on a
    machine where the install is broken. It reads the one place the version is
    declared - `daemon.__version__` - which is the same number a source checkout and
    an installed wheel both carry, so it needs no network and no metadata lookup."""
    from daemon import __version__

    assert cli.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_version_is_a_release_number() -> None:
    """Guards the bump: an empty or malformed `__version__` would make `--version`
    print nothing useful and would tag a release nobody can pin."""
    import re

    from daemon import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_update_installs_the_latest_release(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`daemon update` re-installs through uv exactly what the one-liner installs,
    so the two cannot drift. The network and the subprocess are seams; what this
    pins is that the resolved ref reaches the install command as a tarball (not a
    git ref - a bare machine has no git)."""
    ran: list[list[str]] = []
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(cli, "_latest_ref", lambda: "v9.9.9")
    monkeypatch.setattr(cli, "_run", lambda cmd: ran.append(cmd) or 0)
    # This test pins the install command, not the restart that follows it; the
    # restart has its own tests and must not reach a real service here.
    monkeypatch.setattr(cli, "_restart_after_update", lambda: None)
    monkeypatch.delenv("DAEMON_VERSION", raising=False)

    assert cli.main(["update"]) == 0
    assert ran, "the install command never ran"
    cmd = " ".join(ran[0])
    assert "uv" in ran[0] and "tool" in ran[0] and "install" in ran[0]
    assert "--force" in ran[0]
    assert "git+" not in cmd
    # The `[mcp]` extra rides a PEP 508 direct reference - `name[extra] @ url` - as a
    # single requirement, because uv rejects extras on the positional alongside
    # `--from`. The extra must ride along or an update silently drops MCP (defaults
    # on); install.sh uses the same spec, so the two do not drift.
    assert "daemon-ai[mcp] @ " in cmd
    assert "archive/v9.9.9.tar.gz" in cmd
    assert "--from" not in ran[0], "extras + --from is the combination uv rejects"


class _FakeService:
    """Records whether `restart` was called; reports a fixed status."""

    def __init__(self, *, installed: bool, running: bool = True) -> None:
        self._status = SimpleNamespace(installed=installed, running=running)
        self.restarted = False

    def status(self) -> Any:
        return self._status

    def restart(self) -> Any:
        self.restarted = True
        return SimpleNamespace(applied=True)


def test_update_restarts_the_resident_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the whole command: after the reinstall, the running resident is
    still on the old code (uv replaced the binary, but the supervisor holds the old
    one until it re-execs). So a successful update of an installed service restarts
    it - the user does not have to."""
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(cli, "_latest_ref", lambda: "v9.9.9")
    monkeypatch.setattr(cli, "_run", lambda cmd: 0)
    monkeypatch.setattr(cli, "Settings", lambda: object())
    service = _FakeService(installed=True, running=True)
    monkeypatch.setattr(cli, "service_for", lambda settings: service)
    monkeypatch.delenv("DAEMON_VERSION", raising=False)

    assert cli.main(["update"]) == 0
    assert service.restarted, "an installed service was not restarted after update"
    assert "restarted the resident service" in capsys.readouterr().out


def test_update_without_an_installed_service_does_not_restart(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `daemon run` in a terminal is a foreground process this command cannot
    reach, and a machine with no service has nothing to restart - so update says so
    rather than shelling out to a job that is not there."""
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(cli, "_latest_ref", lambda: "v9.9.9")
    monkeypatch.setattr(cli, "_run", lambda cmd: 0)
    monkeypatch.setattr(cli, "Settings", lambda: object())
    service = _FakeService(installed=False)
    monkeypatch.setattr(cli, "service_for", lambda settings: service)
    monkeypatch.delenv("DAEMON_VERSION", raising=False)

    assert cli.main(["update"]) == 0
    assert not service.restarted
    assert "nothing to restart" in capsys.readouterr().out


def test_update_skips_restart_when_config_will_not_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """You may be updating precisely because a version is broken, so the reinstall
    must still count as done even when the config cannot load - update degrades to
    telling the user to restart by hand, and never touches the service."""
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(cli, "_latest_ref", lambda: "v9.9.9")
    monkeypatch.setattr(cli, "_run", lambda cmd: 0)

    def _raise() -> Any:
        raise cli.ConfigError("bad configuration")

    monkeypatch.setattr(cli, "Settings", _raise)
    monkeypatch.setattr(
        cli, "service_for", lambda settings: pytest.fail("service must not be built")
    )
    monkeypatch.delenv("DAEMON_VERSION", raising=False)

    assert cli.main(["update"]) == 0
    assert "restart the service yourself" in capsys.readouterr().out


def test_update_honours_the_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """`DAEMON_VERSION=<ref>` pins the update, and skips the latest-release lookup -
    the same override install.sh honours."""
    ran: list[list[str]] = []
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(
        cli, "_latest_ref", lambda: pytest.fail("must not resolve latest when pinned")
    )
    monkeypatch.setattr(cli, "_run", lambda cmd: ran.append(cmd) or 0)
    monkeypatch.setattr(cli, "_restart_after_update", lambda: None)
    monkeypatch.setenv("DAEMON_VERSION", "v0.1.0")

    assert cli.main(["update"]) == 0
    assert "archive/v0.1.0.tar.gz" in " ".join(ran[0])


def test_update_without_uv_explains_and_does_not_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source/pip install has no uv; update must say so and point elsewhere, not
    shell out to a binary that is not there."""
    ran: list[list[str]] = []
    monkeypatch.setattr(cli, "_uv_present", lambda: False)
    monkeypatch.setattr(cli, "_run", lambda cmd: ran.append(cmd) or 0)

    assert cli.main(["update"]) == 1
    assert not ran, "update shelled out despite uv being absent"
    assert "uv" in capsys.readouterr().out.lower()


def test_update_reports_a_failed_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_uv_present", lambda: True)
    monkeypatch.setattr(cli, "_latest_ref", lambda: "main")
    monkeypatch.setattr(cli, "_run", lambda cmd: 1)
    monkeypatch.delenv("DAEMON_VERSION", raising=False)

    assert cli.main(["update"]) == 1


# --- .env secrets reach os.environ so MCP keys survive a restart -------------


def test_load_env_secrets_puts_dotenv_values_into_the_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An MCP key lives in .env under its variable name; the engine reads it back
    from os.environ at connect time, so a restart must load .env into the environ or
    a url server loses its bearer and fails with a taskgroup error."""
    (tmp_path / ".env").write_text("TAVILY_API_KEY=tvly-secret\n", encoding="utf-8")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    cli._load_env_secrets()  # the _sandbox fixture already chdir'd into tmp_path

    assert os.environ["TAVILY_API_KEY"] == "tvly-secret"


def test_load_env_secrets_does_not_override_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """setdefault, not overwrite: a value the shell already exported wins, matching
    how Settings resolves precedence (env over file)."""
    (tmp_path / ".env").write_text("TAVILY_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("TAVILY_API_KEY", "from-shell")

    cli._load_env_secrets()

    assert os.environ["TAVILY_API_KEY"] == "from-shell"


def test_latest_ref_falls_back_to_main_when_there_is_no_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the first release is cut, GitHub's releases/latest 404s (or the
    network is down). The installer treats that as 'install main'; so does update."""
    import httpx

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise httpx.HTTPError("no network")

    monkeypatch.setattr(httpx, "get", boom)
    assert cli._latest_ref() == "main"


def test_run_warns_when_the_shell_overrides_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`daemon run` is the command that suffers for it, and doctor is the command
    nobody runs first.

    An install lost hours to a repeating Telegram 409 because `~/.zshrc` exported
    TELEGRAM_BOT_TOKEN for a different tool. The environment outranks `.env`, so the
    file named the right bot and was never consulted, and the only place that said so
    was a check you had to already suspect something to run.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DAEMON_PRESET=offline\n"
        "DAEMON_OLLAMA_MODEL=gemma3:4b\n"
        "TELEGRAM_BOT_TOKEN=1111111111:AAH-in-the-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "2222222222:AAH-in-the-shell")
    monkeypatch.setattr(cli, "_serve", lambda settings: 0)

    with caplog.at_level(logging.WARNING):
        assert cli.main(["run"]) == 0

    assert "overrides .env" in caplog.text
    assert "2222222222" in caplog.text and "1111111111" in caplog.text
    # The id half is the bot's user id and public; the secret half is neither.
    assert "AAH-in-the-shell" not in caplog.text


def test_run_is_quiet_when_the_environment_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A warning on every start would be a warning nobody reads."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DAEMON_PRESET=offline\nDAEMON_OLLAMA_MODEL=gemma3:4b\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "_serve", lambda settings: 0)

    with caplog.at_level(logging.WARNING):
        assert cli.main(["run"]) == 0

    assert "overrides" not in caplog.text


def test_update_install_command_forces_managed_python_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS: the framework Python.app breaks the headless mic grant, so `daemon
    update` must force a uv-managed (non-.app) CPython - see the helper's docstring."""
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    cmd = cli._update_install_command("https://github.com/x/archive/v1.tar.gz")
    assert "--python-preference" in cmd
    assert cmd[cmd.index("--python-preference") + 1] == "only-managed"
    assert cmd[-1] == "daemon-ai[mcp] @ https://github.com/x/archive/v1.tar.gz"


def test_update_install_command_leaves_python_selection_alone_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "linux")
    cmd = cli._update_install_command("https://github.com/x/archive/v1.tar.gz")
    assert "--python-preference" not in cmd and "only-managed" not in cmd
    assert cmd[-1] == "daemon-ai[mcp] @ https://github.com/x/archive/v1.tar.gz"


def test_install_calls_install(service: FakeService) -> None:
    assert cli.main(["install"]) == 0
    assert service.calls == [("install", False)]


def test_install_force_is_passed_through(service: FakeService) -> None:
    cli.main(["install", "--force"])

    assert service.calls == [("install", True)]


def test_uninstall_calls_uninstall(service: FakeService) -> None:
    assert cli.main(["uninstall"]) == 0
    assert service.calls == [("uninstall", None)]


def test_macos_program_puts_launcher_first_then_daemon_argv() -> None:
    program = macapp.macos_program(
        Path("/Users/x/Applications/Daemon.app/Contents/MacOS/launcher"),
        ("/Users/x/.local/bin/daemon", "run"),
    )
    assert program == (
        "/Users/x/Applications/Daemon.app/Contents/MacOS/launcher",
        "/Users/x/.local/bin/daemon",
        "run",
    )


def test_macos_install_writes_launcher_first_in_the_plist(tmp_path: Path) -> None:
    """The load-bearing wiring: launchd must exec Daemon.app's launcher (argv[0]),
    which then execs the daemon path with `run` - in that order, or a real machine
    starts the wrong thing (or nothing). Proven on a real plist on disk, not a mock
    of launchctl/codesign/open (tests/CLAUDE.md: a test that passes for the wrong
    reason is worse than none).
    """
    launcher = tmp_path / "Applications" / "Daemon.app" / "Contents" / "MacOS" / "launcher"
    daemon_path = "/x/.local/bin/daemon"
    program = macapp.macos_program(launcher, (daemon_path, "run"))

    def fake_runner(command: object) -> RunResult:
        return RunResult(0)

    svc = Service(
        label="default",
        working_dir=tmp_path,
        log_dir=tmp_path,
        program=program,
        home=tmp_path,
        platform="darwin",
        runner=fake_runner,
    )
    svc.install()

    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{svc.label}.plist"
    written = plist_path.read_text(encoding="utf-8")

    launcher_index = written.index(str(launcher))
    daemon_index = written.index(daemon_path)
    run_index = written.index("<string>run</string>")
    assert launcher_index < daemon_index < run_index


def test_grant_open_argv_keeps_only_the_console_script_path() -> None:
    """default_program()'s console-script shape: (daemon, "run")."""
    argv = macapp.grant_open_argv(Path("/A/Daemon.app"), ("/x/daemon", "run"))

    assert argv[-3:] == ["--args", "/x/daemon", "request-mic"]
    assert "run" not in argv


def test_grant_open_argv_keeps_the_module_prefix_for_a_checkout() -> None:
    """default_program()'s checkout-fallback shape: (python, "-m", "daemon.cli",
    "run") - the bug this covers: the old code took only daemon_argv[0] (the
    python executable) and dropped `-m daemon.cli`, so the launcher execed
    `python request-mic`, which fails silently and never pops the prompt.
    """
    argv = macapp.grant_open_argv(
        Path("/A/Daemon.app"), ("/usr/bin/python3", "-m", "daemon.cli", "run")
    )

    assert argv[-5:] == ["--args", "/usr/bin/python3", "-m", "daemon.cli", "request-mic"]
    assert "run" not in argv


def test_install_reports_a_codesign_failure_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """build_bundle raises a bare RuntimeError on a codesign failure. It is a
    RuntimeError subclass of ServiceError's own base, so `except ServiceError` in
    main() would NOT catch it - it must be handled inside `_install` itself, before
    any Service is constructed (no launchctl/codesign/open is reached).
    """
    monkeypatch.setattr(cli.sys, "platform", "darwin")

    def fail(app_path: Path) -> Path:
        raise RuntimeError("codesign failed: errSecInternalComponent")

    monkeypatch.setattr("daemon.macapp.build_bundle", fail)

    settings = Settings()
    assert cli._install(settings, force=False) == 1
    assert "daemon:" in capsys.readouterr().err


def test_install_reports_missing_codesign_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FileNotFoundError (an OSError) is what a subprocess call raises when
    `codesign` is not on PATH at all - an Xcode-less acceptance Mac."""
    monkeypatch.setattr(cli.sys, "platform", "darwin")

    def fail(app_path: Path) -> Path:
        raise FileNotFoundError("codesign")

    monkeypatch.setattr("daemon.macapp.build_bundle", fail)

    settings = Settings()
    assert cli._install(settings, force=False) == 1
    assert "daemon:" in capsys.readouterr().err


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


def test_request_mic_reports_status_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import daemon.cli as cli

    monkeypatch.setattr(
        "daemon.voice.mic_access.request_microphone_access", lambda **_: "authorized"
    )
    assert cli.main(["request-mic"]) == 0
    assert "authorized" in capsys.readouterr().out

    monkeypatch.setattr(
        "daemon.voice.mic_access.request_microphone_access", lambda **_: "denied"
    )
    assert cli.main(["request-mic"]) == 1


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


def test_doctor_names_the_service_behind_a_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    reachable_ollama: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`openai_compatible` is one word for many services, so the config line would
    otherwise not say which one is answering - only the address does, and this is
    the same `vendor_label` reverse lookup the wizard uses.
    """
    monkeypatch.setenv("DAEMON_PRESET", "balanced")
    monkeypatch.setenv("DAEMON_HOSTED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DAEMON_OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DAEMON_OPENAI_COMPATIBLE_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-x")

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "[ok] config" in out
    assert "endpoint=DeepSeek" in out
    assert "FAIL" not in out


def test_doctor_names_no_endpoint_for_a_named_hosted_provider(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    reachable_ollama: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anthropic, OpenAI and Gemini already say who they are; the `endpoint=`
    suffix exists only for the one provider name that does not."""
    monkeypatch.setenv("DAEMON_PRESET", "balanced")
    monkeypatch.setenv("DAEMON_HOSTED_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "endpoint=" not in out


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


# --- doctor: what reflection built -------------------------------------------


def test_doctor_reports_the_memory_reflection_built(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The M2 gate is a graph worth reading, and this is where it is readable
    without opening sqlite."""
    _reflected_day(data_dir)

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    # 지현 was noted, 연희동 came in as a link - both are entities, one is a fact.
    assert "[ok] memory: 1 curated fact(s), 2 entity(ies), 1 observation(s)" in out
    assert "지현 (1) -> 연희동" in out


def test_doctor_reports_a_reflection_backlog_and_what_to_run(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reflection loop that has never run leaves no trace anywhere else, so an
    untouched month has to be visible here or it is invisible everywhere."""
    _logged_day(data_dir, "2026-08-01")
    _logged_day(data_dir, "2026-08-02")

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "2 day(s) not reflected on yet (oldest 2026-08-01) - run `daemon reflect`" in out


def test_doctor_does_not_call_today_a_backlog(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Today's log has no reflection artifact and never will until tomorrow -
    `Reflection.catch_up` drops the day still being written to. Counting it made
    doctor say "1 day(s) not reflected on yet - run `daemon reflect`" every day of
    the product's life, and the command it named answered "nothing to reflect on".
    """
    from daemon import clock
    from daemon.memory import log

    _logged_day(data_dir, log.local_date(clock.now()))

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "not reflected on yet" not in out


def test_doctor_says_nothing_recorded_yet_on_a_fresh_install(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 0
    assert "[ok] memory: nothing recorded yet" in capsys.readouterr().out


def test_doctor_survives_a_database_it_cannot_open(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Letting the memory check raise turned the whole of doctor into a traceback -
    the command whose entire job is to explain what is wrong. The schema check
    above it carries the real message."""
    (data_dir / daemon_app.DB_FILENAME).write_bytes(b"this is not a database")

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[ok] memory: not readable" in out
    assert "[FAIL] schema" in out


# --- doctor: what persona evolution built (M4) --------------------------------


def test_doctor_says_nothing_recorded_yet_for_persona_on_a_fresh_install(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 0
    assert "[ok] persona: nothing recorded yet" in capsys.readouterr().out


def test_doctor_reports_why_the_next_persona_run_would_skip(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty rule set and a working weekly pass look the same from outside,
    so doctor has to say *why* the next run would do nothing - not just that
    nothing has happened yet."""
    Store.open(data_dir / daemon_app.DB_FILENAME).close()  # create the mirror, no rows

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "[ok] persona: 0 active rule(s), 0 unconsumed observation(s)" in out
    assert "next run would skip: not enough observations (0<5)" in out


def test_doctor_reports_active_persona_rules_and_unconsumed_observations(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_observations(data_dir, 6)
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        store.insert_persona_rule(body="아침엔 인사만 짧게 한다", created_at=NOW, evidence=[])
    finally:
        store.close()

    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "[ok] persona: 1 active rule(s), 6 unconsumed observation(s)" in out
    assert "next run would skip" not in out, "there is room and evidence; it should not skip"


def test_doctor_reports_a_learned_file_diverged_from_an_empty_mirror(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash between `add()`'s markdown write and its mirror commit: the
    database exists, but the row for this bullet was never committed.
    `LearnedRules.add` now refuses to write in that state rather than
    silently dropping the rule, so this has to be visible somewhere an
    operator will actually look."""
    from daemon.persona.rules import render as render_learned

    Store.open(data_dir / daemon_app.DB_FILENAME).close()  # create the mirror, no rows
    (data_dir / "persona").mkdir(exist_ok=True)
    (data_dir / "persona" / "learned.md").write_text(
        render_learned(["손으로 있던 규칙"]), encoding="utf-8"
    )

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] persona" in out
    assert "learned.md has 1 rule(s) the mirror does not know about" in out
    assert "daemon reindex" in out


def test_doctor_reports_a_learned_file_diverged_with_no_database_at_all(
    data_dir: Path, reachable_ollama: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The literal reproduction: `rm daemon.sqlite3` (non-negotiable 1 calls
    that legitimate) leaves no database file at all, not just an empty one.
    `_persona_check` used to shortcut straight to "nothing recorded yet"
    whenever the file was simply absent, which hid this exact state - the one
    the bug report actually describes."""
    from daemon.persona.rules import render as render_learned

    (data_dir / "persona").mkdir(exist_ok=True)
    (data_dir / "persona" / "learned.md").write_text(
        render_learned(["규칙 0", "규칙 1"]), encoding="utf-8"
    )
    assert not (data_dir / daemon_app.DB_FILENAME).exists()

    assert cli.main(["doctor"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] persona" in out
    assert "learned.md has 2 rule(s)" in out
    assert "no database yet" in out
    assert "daemon reindex" in out


# --- reflect -----------------------------------------------------------------


def test_reflect_with_no_date_catches_up_and_reports_each_day(
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _logged_day(data_dir, "2026-08-01")
    _logged_day(data_dir, "2026-08-02")
    reflection_seam()
    monkeypatch.setattr("daemon.reflection.clock_now", lambda: NOW)

    assert cli.main(["reflect"]) == 0

    out = capsys.readouterr().out
    counts = "(1 message(s) -> 1 fact(s), 1 entity(ies), 1 observation(s))"
    assert f"2026-08-01: written {counts}" in out
    assert f"2026-08-02: written {counts}" in out


def test_reflect_with_a_date_runs_exactly_that_day(
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _logged_day(data_dir, "2026-08-01")
    _logged_day(data_dir, DAY)
    reflection_seam()

    assert cli.main(["reflect", "--date", DAY]) == 0

    assert "2026-08-01" not in capsys.readouterr().out
    assert not artifact_path(data_dir, "2026-08-01").exists()


def test_reflect_force_redoes_a_day_that_is_already_done(
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reflected_day(data_dir)
    reflection_seam()

    assert cli.main(["reflect", "--date", DAY]) == 0
    assert f"{DAY}: skipped" in capsys.readouterr().out

    assert cli.main(["reflect", "--date", DAY, "--force"]) == 0
    assert f"{DAY}: written" in capsys.readouterr().out


def test_reflect_with_nothing_to_do_says_so(
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No log at all is the first-run case, and it is not a fault."""
    reflection_seam()

    assert cli.main(["reflect"]) == 0
    assert "nothing to reflect on" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reply", "fail", "status"),
    [("죄송해요, 잘 모르겠어요", False, "unparseable"), ("", True, "unavailable")],
)
def test_reflect_exits_nonzero_when_a_day_did_not_land(
    reply: str,
    fail: bool,
    status: str,
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pass that wrote nothing must not look like success to the shell: this is
    the only way a cron entry or an operator finds out reflection is not running."""
    _logged_day(data_dir, DAY)
    reflection_seam(reply, fail=fail)

    assert cli.main(["reflect", "--date", DAY]) == 1
    assert f"{DAY}: {status}" in capsys.readouterr().out


def test_reflect_prints_the_problems_a_written_day_still_hit(
    data_dir: Path,
    reflection_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partially applied pass is the failure mode reflection is arranged against,
    so what it dropped has to reach the terminal."""
    _logged_day(data_dir, DAY)
    reflection_seam(json.dumps({"facts": "이건 배열이 아님", "observations": [{"body": "관찰"}]}))

    assert cli.main(["reflect", "--date", DAY]) == 0
    assert "  ! facts was not a list" in capsys.readouterr().out


def test_reflect_passes_date_and_force_through(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    seen: dict[str, Any] = {}

    async def fake(settings: Settings, *, date: str | None, force: bool) -> int:
        seen.update(date=date, force=force)
        return 0

    monkeypatch.setattr(cli, "_reflect", fake)

    assert cli.main(["reflect", "--date", DAY, "--force"]) == 0
    assert seen == {"date": DAY, "force": True}


# --- persona (M4) --------------------------------------------------------------


def test_persona_with_no_rules_says_so(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["persona"]) == 0
    out = capsys.readouterr().out
    assert "no active persona rules yet." in out
    assert "last rule created: never" in out
    assert "last diary: none yet" in out


def test_persona_reports_the_most_recent_diary(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    diary_dir = data_dir / "persona" / "diary"
    diary_dir.mkdir(parents=True)
    (diary_dir / "2026-07-27.md").write_text("older week", encoding="utf-8")
    (diary_dir / "2026-08-03.md").write_text("latest week", encoding="utf-8")

    assert cli.main(["persona"]) == 0
    assert "last diary: 2026-08-03" in capsys.readouterr().out


def test_persona_lists_active_rules(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        store.insert_persona_rule(
            body="아침엔 인사만 짧게 한다", created_at=NOW, evidence=[1, 2]
        )
    finally:
        store.close()

    assert cli.main(["persona"]) == 0
    out = capsys.readouterr().out
    assert "아침엔 인사만 짧게 한다" in out
    assert "2 observation(s)" in out
    assert "last rule created: never" not in out


def test_persona_forget_retires_a_rule(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        rule_id = store.insert_persona_rule(body="규칙", created_at=NOW, evidence=[])
    finally:
        store.close()

    assert cli.main(["persona", "forget", str(rule_id), "--why", "싫다"]) == 0
    assert f"retired rule {rule_id}: 싫다" in capsys.readouterr().out

    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        assert store.active_persona_rules() == []
    finally:
        store.close()


def test_persona_forget_an_unknown_id_is_a_usage_error(data_dir: Path) -> None:
    assert cli.main(["persona", "forget", "999", "--why", "no such rule"]) == 2


def test_persona_evolve_runs_the_real_pass_and_reports_it(
    data_dir: Path,
    persona_seam: Callable[..., FakeProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_observations(data_dir, 3)
    reply = json.dumps(
        {"rules": [{"body": "아침엔 인사만 짧게 한다", "evidence": [1, 2], "key": None}]}
    )
    persona_seam(reply)

    assert cli.main(["persona", "evolve"]) == 0

    out = capsys.readouterr().out
    assert "1 added" in out

    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        assert [r["body"] for r in store.active_persona_rules()] == ["아침엔 인사만 짧게 한다"]
    finally:
        store.close()


def test_persona_evolve_passes_force_through(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class FakeEvolution:
        async def run(self, *, force: bool = False) -> Any:
            seen["force"] = force

            @dataclass
            class Result:
                date: str = "2026-08-03"
                observations_read: int = 0
                proposed: int = 0
                added: int = 0
                retired: int = 0
                skipped: str = ""
                problems: tuple[str, ...] = ()

            return Result()

    async def build(settings: Settings) -> tuple[Any, Any]:
        async def close() -> None: ...

        return FakeEvolution(), close

    monkeypatch.setattr(daemon_app, "build_persona_evolution", build)

    assert cli.main(["persona", "evolve", "--force"]) == 0
    assert seen == {"force": True}


# --- reindex -----------------------------------------------------------------


async def test_reindex_rebuilds_all_three_markdown_tiers(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-negotiable 1 says throwing the database away must lose nothing. A
    rebuild that only restored messages silently dropped every curated fact and
    entity note reflection had concluded, which is the same data loss with a
    passing test suite in front of it.
    """
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    await FileMemoryWriter(data_dir, store).record(
        LoggedMessage(
            ts=NOW,
            role="user",
            content="연희동으로 이사했어",
            origin="owner",
            session_kind="interactive",
            modality="text",
            channel="telegram",
            sender_id="42",
        )
    )
    await CuratedMemory(data_dir, store).add("연희동에 산다", importance=8, supersession_key="home")
    await EntityNotes(data_dir, store).note(
        "지현", "연희동 카페에서 만났다", kind="person", links=("연희동",), date=DAY
    )
    store.close()
    for suffix in ("", "-wal", "-shm"):
        (data_dir / f"{daemon_app.DB_FILENAME}{suffix}").unlink(missing_ok=True)

    assert cli.main(["reindex"]) == 0
    assert "reindexed 1 message(s)" in capsys.readouterr().out

    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        assert [row["content"] for row in store.recent()] == ["연희동으로 이사했어"]
        assert [row["body"] for row in store.active_entries()] == ["연희동에 산다"]
        graph = {name: linked for name, _mentions, linked in EntityNotes(data_dir, store).graph()}
        assert graph["지현"] == ["연희동"]
    finally:
        store.close()


def test_reindex_restores_persona_rules_from_learned_md(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fourth tier `_reindex` rebuilds, additively: a mirror wiped clean
    (`rm daemon.sqlite3`, a legitimate state per non-negotiable 1) comes back
    with every rule `learned.md` still names, and a second run adds nothing
    more."""
    from daemon.persona.rules import render as render_learned

    Store.open(data_dir / daemon_app.DB_FILENAME).close()
    persona_dir = data_dir / "persona"
    persona_dir.mkdir(exist_ok=True)
    bodies = [f"규칙 {i}" for i in range(5)]
    (persona_dir / "learned.md").write_text(render_learned(bodies), encoding="utf-8")

    assert cli.main(["reindex"]) == 0
    capsys.readouterr()

    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        assert sorted(row["body"] for row in store.active_persona_rules()) == sorted(bodies)
    finally:
        store.close()

    assert cli.main(["reindex"]) == 0
    store = Store.open(data_dir / daemon_app.DB_FILENAME)
    try:
        assert len(store.active_persona_rules()) == 5, "a second reindex must not duplicate rows"
    finally:
        store.close()


# --- `daemon help` -----------------------------------------------------------


def test_help_prints_exactly_what_dash_dash_help_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of the subcommand: one document, reachable two ways. If
    these ever differ, someone has written a second command list by hand."""
    assert cli.main(["help"]) == 0
    printed = capsys.readouterr().out
    assert printed == cli.build_parser().format_help()


def test_every_command_is_filed_under_a_group() -> None:
    """The grouped epilog is the only listing there is, so a command missing from
    it is a command nobody can discover."""
    parser = cli.build_parser()
    grouped = {name for entries in parser.command_groups.values() for name, _ in entries}
    assert grouped == set(parser.command_parsers)


def test_groups_are_the_declared_ones() -> None:
    parser = cli.build_parser()
    assert set(parser.command_groups) <= set(cli._GROUP_ORDER)


def test_help_lists_each_command_once(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["help"])
    printed = capsys.readouterr().out
    for name in cli.build_parser().command_parsers:
        listed = [
            line for line in printed.splitlines() if line.startswith(f"  {name} ")
        ]
        assert len(listed) == 1, f"{name} appears {len(listed)} times"


def test_help_for_one_command_prints_that_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["help", "log"]) == 0
    printed = capsys.readouterr().out
    assert "usage: daemon log" in printed
    assert "--raw" in printed


def test_help_for_an_unknown_command_names_the_real_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["help", "lgo"]) == cli.USAGE
    err = capsys.readouterr().err
    assert "no such command: lgo" in err
    assert "doctor" in err


def test_help_works_without_a_usable_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Being told what the commands are is most needed when nothing else works."""

    def broken() -> Settings:
        raise AssertionError("help must not construct Settings")

    monkeypatch.setattr(cli, "Settings", broken)
    assert cli.main(["help"]) == 0
    assert "every day:" in capsys.readouterr().out


# --- `daemon log` ------------------------------------------------------------

NOISE = (
    '2026-08-11 11:03:49,334 INFO uvicorn.access 127.0.0.1:54813 - '
    '"GET /admin/api/health HTTP/1.1" 200'
)
POLL = (
    "2026-08-11 11:03:01,433 INFO httpx HTTP Request: POST "
    'https://api.telegram.org/bot<token>/getUpdates "HTTP/1.1 200 OK"'
)
SIGNAL = "2026-08-11 11:01:27,688 INFO daemon.app proactive: silent - 0 generated"
WRITE = (
    "2026-08-11 11:03:49,371 INFO uvicorn.access 127.0.0.1:54813 - "
    '"PATCH /admin/api/settings HTTP/1.1" 200'
)
"""A successful admin *write*. The first filter dropped every successful admin
request, which swallowed the settings change that had just rewritten `.env`."""


@pytest.fixture
def log_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point `daemon log` at a log this test owns, never a developer's real one."""
    path = tmp_path / "ai.daemon.default.err.log"
    monkeypatch.setattr(
        cli, "service_for", lambda settings: SimpleNamespace(err_log=path)
    )
    return path


def test_log_drops_the_polling_and_keeps_the_rest(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file.write_text("\n".join([NOISE, SIGNAL, POLL, NOISE]) + "\n", encoding="utf-8")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_raw_keeps_the_polling(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file.write_text("\n".join([NOISE, SIGNAL, POLL]) + "\n", encoding="utf-8")

    assert cli.main(["log", "--raw"]) == 0
    assert capsys.readouterr().out.splitlines() == [NOISE, SIGNAL, POLL]


def test_log_counts_the_lines_it_shows_not_the_lines_it_reads(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Over half a real log is polling. Filtering *after* taking the last N made
    `daemon log` print an empty screen on an idle daemon, which reads as broken."""
    log_file.write_text(
        "\n".join([SIGNAL, *([NOISE] * 200), SIGNAL]) + "\n", encoding="utf-8"
    )

    assert cli.main(["log", "-n", "2"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL, SIGNAL]


def test_log_keeps_a_failed_poll(log_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A 409 from getUpdates is the line that explains two bots sharing a token -
    exactly the failure this filter must never hide."""
    conflict = POLL.replace("200 OK", "409 Conflict")
    server_error = NOISE.replace("HTTP/1.1\" 200", "HTTP/1.1\" 500")
    log_file.write_text("\n".join([conflict, server_error]) + "\n", encoding="utf-8")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [conflict, server_error]


def test_log_without_a_file_says_so(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["log"]) == cli.PROBLEM
    assert "no log at" in capsys.readouterr().err


def test_tail_reads_back_across_block_boundaries(tmp_path: Path) -> None:
    """The backward reader stitches blocks together; a line split across two of
    them must not come back mangled or doubled."""
    path = tmp_path / "big.log"
    lines = [f"{n:05d} " + "x" * 200 for n in range(400)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert cli._tail_lines(path, 3, cli._keeper(raw=True)) == lines[-3:]
    assert cli._tail_lines(path, 400, cli._keeper(raw=True)) == lines
    assert cli._tail_lines(path, 900, cli._keeper(raw=True)) == lines


def test_tail_of_an_empty_log_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.log"
    path.write_text("", encoding="utf-8")
    assert cli._tail_lines(path, 10, cli._keeper(raw=True)) == []


def test_follow_yields_lines_as_they_are_appended(tmp_path: Path) -> None:
    """The follow path is a generator precisely so it can be tested without a
    thread and without waiting on a loop that never ends."""
    path = tmp_path / "live.log"
    path.write_text("already here\n", encoding="utf-8")

    stream = cli._stream_lines(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("first\nsecond\n")
    try:
        assert next(stream) == "first"
        assert next(stream) == "second"
    finally:
        stream.close()


def _unusable(reason: str) -> type[Settings]:
    """A `Settings` that refuses to build, but is still the class.

    Replacing the name with a bare function would also remove `model_fields`,
    which the fallback path reads to find the defaults - the test would then fail
    for a reason the product never hits.
    """

    class Unusable(Settings):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ConfigError(reason)

    return Unusable


def test_log_keeps_what_the_admin_console_changed(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reads repeat unprompted; a write is always something that happened. Only
    the polling GETs are noise, and only when they succeeded."""
    log_file.write_text("\n".join([NOISE, WRITE, NOISE]) + "\n", encoding="utf-8")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [WRITE]


def test_log_reads_the_default_path_when_the_configuration_will_not_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A configuration the process cannot load is one of the ways the resident
    fails, and the log is where that failure is written down - so `daemon log` must
    not be gated on the same validation that is refusing to pass."""
    monkeypatch.chdir(tmp_path)
    # Cleared so this really exercises the model's own default (`./data`) rather
    # than the value the sandbox fixture exports.
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")

    monkeypatch.setattr(cli, "Settings", _unusable("preset 'offline' routes no voice"))

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_honours_a_configured_data_dir_without_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fallback still reads DAEMON_DATA_DIR - it falls back to the model's own
    defaults, not to a second copy of them."""
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "elsewhere" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "elsewhere"))

    monkeypatch.setattr(cli, "Settings", _unusable("nope"))

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


OAUTH = (
    "2026-08-11 11:03:49,371 INFO uvicorn.access 127.0.0.1:54813 - "
    '"GET /admin/api/mcp/oauth/callback?code=abc&state=xyz HTTP/1.1" 200'
)
"""A GET that is a write: the OAuth redirect has to be a GET, and completing it
lands MCP tokens on disk. The only line that says a connection finished."""


def test_log_keeps_the_oauth_callback(
    log_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The method is a proxy for read-vs-write, not the thing itself. Dropping
    every successful GET would hide the one admin GET that changes state."""
    log_file.write_text("\n".join([NOISE, OAUTH, NOISE]) + "\n", encoding="utf-8")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [OAUTH]


def test_log_survives_a_field_that_will_not_coerce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd number in `.env` raises pydantic's ValidationError, not ConfigError -
    it comes from field parsing, before the model validator that raises ConfigError
    runs at all. Catching only the latter left the ordinary broken `.env` printing a
    traceback from the command that exists to survive one."""
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DAEMON_PORT", "not-a-number")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_survives_a_service_label_that_is_not_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`Settings` rejects the label and so does `Service`, from a different
    exception type - so the fallback re-raised as a traceback the command it was
    added to protect. The label is the filename, so an unusable one never named a
    log; the default is the name a resident would have run under."""
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DAEMON_SERVICE_LABEL", "bad label!")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_honours_a_lowercase_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`Settings` is `case_sensitive=False`, so a working build honours
    `daemon_data_dir`. A stricter lookup in the fallback would send the two paths to
    different directories for the same environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    logs = tmp_path / "mydata" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    monkeypatch.setenv("daemon_data_dir", str(tmp_path / "mydata"))
    monkeypatch.setenv("DAEMON_PORT", "not-a-number")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_survives_an_env_file_the_encoding_of_which_is_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `.env` holding Korean, saved once by an editor as CP949, is not valid UTF-8.
    `Settings` raises UnicodeDecodeError while *reading the file* - a ValueError, so
    neither the ConfigError nor the OSError guard saw it, and the command that exists
    to survive a broken configuration printed a traceback for a plausible one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    (tmp_path / ".env").write_bytes("DAEMON_PERSONA_SEED=연희동에 산다\n".encode("cp949"))

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_log_survives_a_data_dir_that_is_not_a_usable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An embedded NUL makes `Path.resolve` raise ValueError, not ServiceError, so the
    label guard did not cover it and the retry reused the same bad directory. Falls
    all the way back to the model's defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "ai.daemon.default.err.log").write_text(SIGNAL + "\n", encoding="utf-8")
    (tmp_path / ".env").write_bytes(b"DAEMON_DATA_DIR=/tmp/bad\x00dir\n")

    assert cli.main(["log"]) == 0
    assert capsys.readouterr().out.splitlines() == [SIGNAL]


def test_the_last_of_two_cases_wins_like_settings_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`case_sensitive=False` case-folds into a dict, so the later name is the one
    that counts. Returning the first match sent the fallback to a directory the
    resident would not have used."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "first"))
    monkeypatch.setenv("daemon_data_dir", str(tmp_path / "second"))

    assert Settings(preset="offline").data_dir == tmp_path / "second", "premise"
    assert cli._env_setting("DAEMON_DATA_DIR", "data_dir") == str(tmp_path / "second")


def test_doctor_reports_a_field_that_will_not_coerce(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command whose whole job is explaining a configuration that will not load
    must not be the one that raises. ConfigError is only what the model validator
    produces; a field that will not coerce raises pydantic's ValidationError first."""
    monkeypatch.setenv("DAEMON_PORT", "not-a-number")

    assert cli.main(["doctor"]) == cli.PROBLEM
    printed = capsys.readouterr().out
    assert "[FAIL] config:" in printed
    assert "DAEMON_PORT" in printed


def test_doctor_survives_an_env_file_the_encoding_of_which_is_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The third of the three reasons `_doctor`'s catch is wide, and the one the
    commit that widened it named while leaving it untested: `Settings` raises
    UnicodeDecodeError while *reading* a `.env` that an editor saved as CP949, so
    neither ConfigError nor ValidationError describes it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_bytes("DAEMON_PERSONA_SEED=연희동에 산다\n".encode("cp949"))

    assert cli.main(["doctor"]) == cli.PROBLEM
    assert "[FAIL] config:" in capsys.readouterr().out


def test_update_that_cannot_load_the_config_still_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`daemon update` runs before the Settings gate on purpose, so its restart step
    meets every `.env` there is. A traceback on the last line of a reinstall that
    already succeeded is the one outcome its docstring rules out."""
    monkeypatch.setenv("DAEMON_PORT", "not-a-number")

    cli._restart_after_update()

    assert "the code is updated" in capsys.readouterr().out
