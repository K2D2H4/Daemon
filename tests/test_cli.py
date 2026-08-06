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
from typing import Any

import pytest
from conftest import FakeProvider

from daemon import app as daemon_app
from daemon import cli
from daemon.config import Route, Settings
from daemon.fs import DIR_MODE
from daemon.llm.gateway import LLMGateway
from daemon.memory.base import LoggedMessage
from daemon.memory.curated import CuratedMemory
from daemon.memory.entities import EntityNotes
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter
from daemon.persona.evolve import PersonaEvolution
from daemon.reflection import Reflection, artifact_path
from daemon.service import ServiceAction, ServiceStatus
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
    monkeypatch.setattr(cli, "service_for", lambda settings: fake)
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
