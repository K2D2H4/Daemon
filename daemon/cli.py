"""The `daemon` command.

`run` is the resident process; everything else is an operator command that has to
work on a machine where the service is misconfigured, so each one prints what it
found instead of raising a traceback.

Exit codes: 0 fine, 1 something is wrong with the install, 2 the command or the
configuration could not even be read.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from daemon.config import OLLAMA, ConfigError, Settings
from daemon.fs import DIR_MODE
from daemon.service import Service, ServiceAction, ServiceError, ServiceStatus

OK = 0
PROBLEM = 1
USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daemon",
        description="Self-hosted AI companion. `daemon run` with no arguments is the default.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="run the daemon in the foreground (what the service supervises)")
    install = sub.add_parser("install", help="install the OS service so it survives a reboot")
    install.add_argument(
        "--force",
        action="store_true",
        help="replace an existing unit file after showing what changes",
    )
    sub.add_parser("uninstall", help="stop the OS service and remove its unit file")
    sub.add_parser("status", help="is the service installed and running")
    sub.add_parser("doctor", help="check configuration, Ollama, data dir and schema")
    sub.add_parser("reindex", help="rebuild the sqlite mirror from the markdown log")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        # `daemon` alone kept running the server through all of M1a. Breaking that
        # to make a point about explicitness would only break people's plists.
        args_list = ["run"]

    args = build_parser().parse_args(args_list)
    command: str = args.command

    if command == "doctor":
        # Doctor is the one command that must survive a configuration it cannot
        # load - explaining the breakage is its whole job.
        return _doctor()

    try:
        settings = Settings()
    except ConfigError as exc:
        print(f"daemon: bad configuration: {exc}", file=sys.stderr)
        print("daemon: run `daemon doctor` for the full picture.", file=sys.stderr)
        return USAGE

    if command == "run":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        return _serve(settings)
    if command == "reindex":
        inserted = _reindex(settings)
        print(f"reindexed {inserted} message(s) the mirror was missing")
        return OK

    try:
        if command == "install":
            return _print_action(service_for(settings).install(force=args.force))
        if command == "uninstall":
            return _print_action(service_for(settings).uninstall())
        if command == "status":
            return _print_status(service_for(settings).status())
    except ServiceError as exc:
        print(f"daemon: {exc}", file=sys.stderr)
        return PROBLEM

    raise AssertionError(f"unhandled command {command!r}")  # pragma: no cover


# --- seams the tests replace -------------------------------------------------


def service_for(settings: Settings) -> Service:
    """Build the service definition from settings.

    The working directory is where `.env` lives - the current directory, which is
    the directory the user is standing in when they install. The unit file carries
    that path and nothing else, so the secrets stay in one file.
    """
    return Service(
        label=settings.service_label,
        working_dir=Path.cwd(),
        log_dir=settings.data_dir / "logs",
    )


def _serve(settings: Settings) -> int:
    import uvicorn

    from daemon.app import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
    return OK


def _reindex(settings: Settings) -> int:
    from daemon.app import DB_FILENAME
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        return reindex(settings.data_dir, store)
    finally:
        store.close()


async def probe_ollama(settings: Settings) -> tuple[bool, str]:
    """Is Ollama reachable? Every preset routes embeddings there, so recall is
    dead without it even under a fully hosted setup."""
    from daemon.llm.providers.ollama import OllamaProvider

    provider = OllamaProvider(settings.ollama_base_url)
    try:
        reachable = await provider.health()
    finally:
        await provider.aclose()
    return reachable, settings.ollama_base_url


# --- doctor ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _doctor() -> int:
    try:
        settings = Settings()
    except ConfigError as exc:
        checks = [Check("config", False, str(exc))]
    else:
        table = ", ".join(
            f"{task.value}->{route.provider}" for task, route in settings.routing_table().items()
        )
        checks = [
            Check(
                "config",
                True,
                f"preset={settings.preset} voice={settings.voice_enabled} [{table}]",
            ),
            _data_dir_check(settings),
            _schema_check(settings),
            *_ollama_checks(settings),
        ]

    failed = 0
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        failed += not check.ok
    if failed:
        # Flushed first, or the summary lands above the checks it summarises when
        # stdout is a pipe and stderr is not.
        sys.stdout.flush()
        print(f"\n{failed} check(s) failed.", file=sys.stderr)
    return PROBLEM if failed else OK


def _data_dir_check(settings: Settings) -> Check:
    path = settings.data_dir
    if not path.exists():
        return Check(
            "data dir",
            False,
            f"{path} does not exist; it holds the markdown source of truth "
            "(it is created on first run, so this only matters if you expected history)",
        )
    mode = path.stat().st_mode & 0o777
    if mode != DIR_MODE:
        return Check(
            "data dir",
            False,
            f"{path} is mode {mode:04o}, expected {DIR_MODE:04o} - verbatim "
            "conversations are readable by other local accounts",
        )
    return Check("data dir", True, f"{path} exists, mode {mode:04o}")


def _schema_check(settings: Settings) -> Check:
    from daemon.app import DB_FILENAME
    from daemon.memory.store import SCHEMA_VERSION, Store

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check(
            "schema", True, f"no database yet at {path}; it is built on first run from the markdown"
        )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return Check("schema", False, f"cannot open {path}: {exc}")
    try:
        found = Store(conn).schema_version()
    except sqlite3.DatabaseError as exc:
        return Check("schema", False, f"{path} is not a readable database: {exc}")
    finally:
        conn.close()

    if found is None:
        return Check("schema", False, f"{path} has no schema version recorded")
    if found > SCHEMA_VERSION:
        return Check(
            "schema",
            False,
            f"{path} is v{found}, this build understands v{SCHEMA_VERSION}; it will "
            "refuse to open. Downgrade Daemon or move the file aside - the markdown "
            "under memory/ can rebuild it",
        )
    if found < SCHEMA_VERSION:
        return Check("schema", True, f"v{found}, migrates to v{SCHEMA_VERSION} on next start")
    return Check("schema", True, f"v{found}")


def _ollama_checks(settings: Settings) -> list[Check]:
    providers = {route.provider for route in settings.routing_table().values()}
    if OLLAMA not in providers:
        return []
    reachable, url = asyncio.run(probe_ollama(settings))
    if not reachable:
        return [
            Check(
                "ollama",
                False,
                f"not reachable at {url}; local routing and recall embeddings "
                f"({settings.embed_model}) cannot work",
            )
        ]
    return [Check("ollama", True, f"reachable at {url}, embed model {settings.embed_model}")]


# --- printing ----------------------------------------------------------------


def _print_action(action: ServiceAction) -> int:
    if action.changes:
        print(f"changes to {action.unit_path}:")
        for line in action.changes:
            print(f"  {line}")
    for note in action.notes:
        print(note)
    for command in action.commands:
        print(f"ran: {' '.join(command)}")
    if action.applied:
        print(f"{action.label} installed at {action.unit_path}")
        return OK
    # Not applied is not always an error, but "nothing happened" should never
    # look like success to a script.
    return OK if not action.changes else PROBLEM


def _print_status(status: ServiceStatus) -> int:
    print(f"label:     {status.label}")
    print(f"unit file: {status.unit_path} ({'present' if status.installed else 'missing'})")
    print(f"loaded:    {status.loaded}")
    print(f"running:   {status.running}")
    if status.detail:
        print(f"detail:    {status.detail}")
    for note in status.notes:
        print(note)
    return OK if status.installed and status.running else PROBLEM


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
