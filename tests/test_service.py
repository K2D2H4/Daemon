"""OS residency: the unit file, its permissions, and the launchctl/systemctl calls.

No subprocess is ever spawned and nothing is written outside tmp_path: the runner
is injected and `home` is a parameter, because a test suite that installs a real
LaunchAgent on the developer's machine is a test suite nobody runs twice.
"""

from __future__ import annotations

import stat
from collections.abc import Sequence
from pathlib import Path

import pytest

from daemon import service as service_module
from daemon.fs import DIR_MODE, FILE_MODE
from daemon.service import RunResult, Service, ServiceError

PROGRAM = ("/opt/venv/bin/daemon", "run")

BOOTSTRAP_EIO = (
    "Bootstrap failed: 5: Input/output error\n"
    "Try re-running the command as root for richer errors."
)
"""What real launchd says when it will not bootstrap - measured, not invented. It
is the same text for a job that is already loaded and for a path that does not
exist, so no substring of it can tell the two apart."""

NO_SUCH_SERVICE = 'Could not find service "ai.daemon.default" in domain for user gui: 501'
"""And what it says (exit 113) when the job really is not loaded."""


class FakeRunner:
    """Records every command. Answers 0 unless told otherwise for a substring."""

    def __init__(self, failures: dict[str, RunResult] | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._failures = failures or {}

    def __call__(self, command: Sequence[str]) -> RunResult:
        self.commands.append(tuple(command))
        joined = " ".join(command)
        for needle, result in self._failures.items():
            if needle in joined:
                return result
        return RunResult(0)


def make_service(
    tmp_path: Path,
    *,
    platform: str = "darwin",
    label: str = "default",
    runner: FakeRunner | None = None,
) -> Service:
    working = tmp_path / "project"
    working.mkdir(exist_ok=True)
    return Service(
        label=label,
        working_dir=working,
        log_dir=tmp_path / "data" / "logs",
        program=PROGRAM,
        home=tmp_path / "home",
        platform=platform,
        uid=501,
        runner=runner or FakeRunner(),
    )


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- what the unit file may and may not contain ------------------------------


def test_the_plist_carries_a_working_directory_and_no_secrets(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    # The credentials the process needs are in .env, in the working directory.
    (service.working_dir / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-secret\nTELEGRAM_BOT_TOKEN=123:secret-token\n"
    )

    plist = service.render()

    assert str(service.working_dir) in plist
    assert "sk-ant-secret" not in plist
    assert "secret-token" not in plist
    # A plist is echoed back by `launchctl print` and lands in every backup of
    # ~/Library, so it must not learn how to carry a key at all.
    assert "EnvironmentVariables" not in plist


def test_the_plist_names_the_label_the_program_and_the_logs(tmp_path: Path) -> None:
    service = make_service(tmp_path, label="second")

    plist = service.render()

    assert "<string>ai.daemon.second</string>" in plist
    assert "<string>/opt/venv/bin/daemon</string>" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert str(service.out_log) in plist
    assert str(service.err_log) in plist


def test_log_paths_are_absolute_even_when_the_data_dir_is_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DAEMON_DATA_DIR defaults to ./data. systemd refuses a relative `append:`
    # path and launchd resolves one against the wrong directory, so a relative
    # log path means logs nobody can find - or a unit that will not start.
    monkeypatch.chdir(tmp_path)
    service = Service(
        working_dir=tmp_path,
        log_dir=Path("data/logs"),
        program=PROGRAM,
        home=tmp_path / "home",
        platform="linux",
        uid=501,
        runner=FakeRunner(),
    )

    assert service.out_log.is_absolute()
    assert f"append:{tmp_path / 'data/logs'}" in service.render()


def test_the_linux_unit_is_a_user_unit(tmp_path: Path) -> None:
    service = make_service(tmp_path, platform="linux")

    unit = service.render()

    assert service.unit_path == tmp_path / "home/.config/systemd/user/ai.daemon.default.service"
    assert "ExecStart=/opt/venv/bin/daemon run" in unit
    assert "Restart=always" in unit
    assert "WantedBy=default.target" in unit
    assert "User=" not in unit  # a system unit would need one; this is not one


# --- permissions -------------------------------------------------------------


def test_install_writes_an_owner_only_plist_and_owner_only_logs(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    action = service.install()

    assert action.applied
    assert mode_of(service.unit_path) == FILE_MODE
    assert mode_of(service.log_dir) == DIR_MODE
    # launchd would create these with the ambient umask, so they are created here
    # first: a traceback in the error log can quote what the user said.
    assert mode_of(service.out_log) == FILE_MODE
    assert mode_of(service.err_log) == FILE_MODE


def test_install_tightens_a_plist_that_was_already_world_readable(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.unit_path.parent.mkdir(parents=True)
    service.unit_path.write_text("<plist>old</plist>")
    service.unit_path.chmod(0o644)

    service.install(force=True)

    assert mode_of(service.unit_path) == FILE_MODE


def test_install_does_not_re_permission_the_directory_it_writes_into(tmp_path: Path) -> None:
    """`~/Library/LaunchAgents` is not ours. It is shared with every other agent the
    user has installed, it normally stands at 0755, and `fs.secure_dir` would take
    it to 0700 - the same overreach its own docstring records `daemon install`
    committing once already, when it walked out to $HOME.

    The plist inside is ours and stays 0600; the directory holding it is not.
    """
    service = make_service(tmp_path)
    service.unit_path.parent.mkdir(parents=True)
    service.unit_path.parent.chmod(0o755)

    service.install()

    assert mode_of(service.unit_path.parent) == 0o755
    assert mode_of(service.unit_path) == FILE_MODE


# --- the unit file is replaced, never truncated ------------------------------
#
# A plist launchd cannot parse is not loaded, and a job that is not loaded is a
# daemon that quietly never comes back after the next login. So the window where
# this file is empty or half-written has to not exist, which is `fs.py`'s atomic
# replace rather than an O_TRUNC write.


def test_replacing_the_unit_file_swaps_it_instead_of_truncating_it(tmp_path: Path) -> None:
    """Rename, not rewrite: the inode under the path changes, so no reader - and no
    login-time launchd scan - can observe the file mid-write."""
    service = make_service(tmp_path)
    service.install()
    service.unit_path.write_text("<plist>hand edited</plist>")
    before = service.unit_path.stat().st_ino

    service.install(force=True)

    assert service.unit_path.read_text() == service.render()
    assert service.unit_path.stat().st_ino != before
    # And nothing left beside it. A stray dotfile in ~/Library/LaunchAgents is
    # somebody else's confusing morning.
    assert [path.name for path in service.unit_path.parent.iterdir()] == [
        service.unit_path.name
    ]


def test_a_write_that_fails_leaves_the_installed_unit_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this ordering exists for. A truncating write that dies partway
    leaves a plist launchd refuses, and the previous working one is already gone -
    strictly worse than not having written at all."""
    service = make_service(tmp_path)
    service.install()
    service.unit_path.write_text("<plist>hand edited</plist>")

    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(service_module, "write_private_replace", boom, raising=False)

    with pytest.raises(OSError):
        service.install(force=True)

    assert service.unit_path.read_text() == "<plist>hand edited</plist>"


# --- launchctl / systemctl ---------------------------------------------------


def test_install_bootstraps_the_agent(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)

    service.install()

    assert runner.commands == [
        ("launchctl", "bootstrap", "gui/501", str(service.unit_path)),
    ]


def test_install_on_linux_reloads_and_enables_the_unit(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, platform="linux", runner=runner)

    action = service.install()

    assert runner.commands == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", "--now", "ai.daemon.default.service"),
    ]
    # Without lingering the unit dies at logout, which is the one thing residency
    # exists to prevent (docs/PLAN.md 3.1).
    assert any("linger" in note for note in action.notes)


def test_a_failing_launchctl_becomes_a_ServiceError(tmp_path: Path) -> None:
    runner = FakeRunner(
        {
            "bootstrap": RunResult(5, stderr=BOOTSTRAP_EIO),
            "print": RunResult(113, stderr=NO_SUCH_SERVICE),
        }
    )
    service = make_service(tmp_path, runner=runner)

    with pytest.raises(ServiceError, match="Input/output error"):
        service.install()


def test_an_already_loaded_job_is_not_a_failure(tmp_path: Path) -> None:
    """Measured against real launchd (Darwin 25.5, uid 501): bootstrapping a job
    launchd already has exits **5 with `Input/output error`** - byte for byte the
    same generic EIO it gives for a plist path that does not exist. This test used
    to assert `17: service already loaded`, a string launchctl never produces, so
    it stayed green while `daemon install` raised on its own second run.

    There is no message to match, which is the whole point: ask launchd whether the
    job is there instead of reading its prose.
    """
    runner = FakeRunner({"bootstrap": RunResult(5, stderr=BOOTSTRAP_EIO)})
    service = make_service(tmp_path, runner=runner)

    service.install()  # must not raise

    assert ("launchctl", "print", "gui/501/ai.daemon.default") in runner.commands


# --- an existing install -----------------------------------------------------


def test_reinstalling_the_same_definition_only_reloads_it(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)
    service.install()
    written = service.unit_path.read_text()
    runner.commands.clear()

    action = service.install()

    assert not action.applied
    assert action.changes == ()
    assert service.unit_path.read_text() == written
    assert runner.commands == [("launchctl", "bootstrap", "gui/501", str(service.unit_path))]


def test_a_changed_definition_is_reported_instead_of_overwritten(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)
    service.install()
    service.unit_path.write_text(service.render().replace("<true/>", "<false/>"))
    runner.commands.clear()

    action = service.install()

    assert not action.applied
    assert any(line.startswith("-") and "false" in line for line in action.changes)
    assert any(line.startswith("+") and "true" in line for line in action.changes)
    assert "<false/>" in service.unit_path.read_text()  # untouched
    assert runner.commands == []  # nothing was loaded or unloaded either


def test_force_boots_the_old_job_out_before_replacing_it(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)
    service.install()
    service.unit_path.write_text("<plist>hand edited</plist>")
    runner.commands.clear()

    action = service.install(force=True)

    assert action.applied
    assert service.unit_path.read_text() == service.render()
    assert runner.commands == [
        ("launchctl", "bootout", "gui/501/ai.daemon.default"),
        ("launchctl", "bootstrap", "gui/501", str(service.unit_path)),
    ]


# --- uninstall ---------------------------------------------------------------


def test_uninstall_boots_out_removes_the_unit_and_keeps_the_logs(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)
    service.install()
    service.out_log.write_text("a log line\n")
    runner.commands.clear()

    action = service.uninstall()

    assert action.applied
    assert not service.unit_path.exists()
    assert service.out_log.read_text() == "a log line\n"
    assert runner.commands == [("launchctl", "bootout", "gui/501/ai.daemon.default")]


def test_uninstalling_what_was_never_installed_does_nothing(tmp_path: Path) -> None:
    runner = FakeRunner()

    action = make_service(tmp_path, runner=runner).uninstall()

    assert not action.applied
    assert runner.commands == []


# --- status ------------------------------------------------------------------


def test_status_of_a_running_agent(tmp_path: Path) -> None:
    runner = FakeRunner(
        {"print": RunResult(0, stdout="ai.daemon.default = {\n\tstate = running\n")}
    )
    service = make_service(tmp_path, runner=runner)
    service.install()

    status = service.status()

    assert (status.installed, status.loaded, status.running) == (True, True, True)


def test_status_when_nothing_is_installed(tmp_path: Path) -> None:
    runner = FakeRunner({"print": RunResult(113, stderr="Could not find service")})

    status = make_service(tmp_path, runner=runner).status()

    assert (status.installed, status.loaded, status.running) == (False, False, False)
    assert "Could not find service" in status.detail


def test_status_on_linux_reads_systemctl(tmp_path: Path) -> None:
    runner = FakeRunner(
        {
            "is-enabled": RunResult(0, stdout="enabled\n"),
            "is-active": RunResult(0, stdout="active\n"),
        }
    )
    service = make_service(tmp_path, platform="linux", runner=runner)
    service.install()

    status = service.status()

    assert (status.loaded, status.running) == (True, True)


# --- restart -----------------------------------------------------------------


def test_restart_kickstarts_the_agent_on_macos(tmp_path: Path) -> None:
    """`daemon update` reinstalls the code in place; the running job keeps the old
    code until it re-execs, so restart kickstarts it - `-k` kills the current
    instance and starts a fresh one from the same plist."""
    runner = FakeRunner()
    service = make_service(tmp_path, runner=runner)

    action = service.restart()

    assert runner.commands == [
        ("launchctl", "kickstart", "-k", "gui/501/ai.daemon.default"),
    ]
    assert action.applied


def test_restart_on_linux_uses_systemctl_restart(tmp_path: Path) -> None:
    runner = FakeRunner()
    service = make_service(tmp_path, platform="linux", runner=runner)

    service.restart()

    assert runner.commands == [
        ("systemctl", "--user", "restart", "ai.daemon.default.service"),
    ]


def test_a_failing_restart_becomes_a_ServiceError(tmp_path: Path) -> None:
    """A job that is not loaded fails here loudly - the caller (`daemon update`)
    checks `status().installed` first, but a lie from launchd must not pass as a
    restart that happened."""
    runner = FakeRunner({"kickstart": RunResult(3, stderr="No such process")})
    service = make_service(tmp_path, runner=runner)

    with pytest.raises(ServiceError, match="kickstart"):
        service.restart()


# --- refusals ----------------------------------------------------------------


def test_windows_is_refused_with_a_reason(tmp_path: Path) -> None:
    service = make_service(tmp_path, platform="win32")

    for call in (service.install, service.uninstall, service.status, service.restart):
        with pytest.raises(ServiceError, match="not supported on 'win32'"):
            call()


def test_a_label_that_would_escape_the_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ServiceError, match="not a usable service label"):
        make_service(tmp_path, label="../../etc/cron.d/evil")
