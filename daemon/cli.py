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
from typing import Any

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
    setup = sub.add_parser(
        "setup", help="first-run onboarding: pick a preset, verify keys, write .env"
    )
    setup.add_argument(
        "--check",
        action="store_true",
        help="report what is missing and exit; asks nothing, contacts nobody",
    )
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
    reflect = sub.add_parser(
        "reflect", help="consolidate a day of conversation into memory and observations"
    )
    reflect.add_argument(
        "--date",
        help="a local YYYY-MM-DD. Omitted: every unreflected day except today, "
        "oldest first - today is still being written to.",
    )
    reflect.add_argument(
        "--force", action="store_true", help="redo a day that already has an artifact"
    )
    proactive = sub.add_parser(
        "proactive",
        help="run one proactivity round now and print its verdicts (dry by default)",
    )
    proactive.add_argument(
        "--speak",
        action="store_true",
        help="actually decide what to say and deliver it. Without this the round "
        "stops at the gate and costs no model call.",
    )

    sub.add_parser("voice", help="hold one spoken conversation at this machine")

    pairing = sub.add_parser("pairing", help="see and approve who may talk to Daemon")
    pairing_sub = pairing.add_subparsers(dest="pairing_command", required=True)
    pairing_sub.add_parser("list", help="pending pairing requests and their codes")
    approve = pairing_sub.add_parser("approve", help="approve a pairing code")
    approve.add_argument("code", help="the 8-character code the bot replied with")

    tools = sub.add_parser("tools", help="what Daemon may do to this machine, and what it did")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    tools_sub.add_parser("list", help="the tools that are loaded, and the policy in force")
    tools_sub.add_parser("log", help="recent tool calls, including the refused ones")
    tools_sub.add_parser("pending", help="approvals waiting on an answer")
    forget = tools_sub.add_parser("forget", help="drop a standing approval granted with 'always'")
    forget.add_argument("pattern", help="the command pattern to stop trusting, e.g. 'git status'")
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
    if command == "setup":
        # Same reason, more so: setup exists for the machine that has no usable
        # configuration yet, so it must not require one to start.
        return _setup(check_only=args.check)

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
    if command == "proactive":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        return asyncio.run(_proactive(settings, speak=args.speak))
    if command == "reflect":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        return asyncio.run(_reflect(settings, date=args.date, force=args.force))
    if command == "voice":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        from daemon.app import run_voice

        return asyncio.run(run_voice(settings))
    if command == "pairing":
        return _pairing(settings, args)
    if command == "tools":
        return _tools(settings, args)

    try:
        if command == "install":
            return _print_action(service_for(settings).install(force=args.force))
        if command == "uninstall":
            return _print_action(service_for(settings).uninstall(), verb="removed")
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


def _setup(*, check_only: bool) -> int:
    from daemon.setup import run

    return run(check_only=check_only)


def _serve(settings: Settings) -> int:
    import uvicorn

    from daemon.app import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
    return OK


def _pairing(settings: Settings, args: Any) -> int:
    """See and approve pairing requests.

    This is the other half of onboarding: the bot hands a stranger a code, and the
    only place that code can be approved is a terminal on the machine that holds
    the data. Nothing about approval is reachable from the channel itself, which
    is what keeps a stranger from approving themselves.
    """
    from daemon.app import DB_FILENAME
    from daemon.channels.pairing import Pairing, PairingError
    from daemon.channels.telegram import TelegramChannel
    from daemon.memory.store import Store

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        pairing = Pairing(store, TelegramChannel.name)
        if args.pairing_command == "list":
            requests = pairing.pending()
            if not requests:
                print("no pending requests. Message your bot and one appears here.")
                return OK
            for request in requests:
                print(
                    f"  {request.code}  from id={request.sender_id}  "
                    f"expires {request.expires_at:%H:%M}"
                )
            print(f"\napprove one with: daemon pairing approve {requests[0].code}")
            return OK

        try:
            approval = pairing.approve(args.code)
        except PairingError as exc:
            print(f"daemon: {exc}", file=sys.stderr)
            return USAGE
        who = "owner" if approval.is_owner else "guest"
        print(f"approved id={approval.sender_id} as {who}. They can talk to Daemon now.")
        return OK
    finally:
        store.close()


def _tools(settings: Settings, args: Any) -> int:
    """What Daemon may do to this machine, and what it has done.

    The `log` subcommand is the one that earns this command's existence. Tool calls
    are audited into sqlite rather than into the markdown log (schema.sql), so
    without a way to read them back the audit trail is real but unreachable, which
    is the same as not having one.
    """
    from daemon.app import DB_FILENAME
    from daemon.memory.store import Store
    from daemon.tools.builtin import builtin_tools

    if args.tools_command == "list":
        print(f"tools:      {'on' if settings.tools_enabled else 'off (DAEMON_TOOLS_ENABLED)'}")
        print(f"mode:       {settings.tools_mode}")
        print(f"roots:      {', '.join(settings.tools_roots) or '(none)'}")
        print(f"allowlist:  {', '.join(settings.tools_allowlist) or '(none)'}")
        browser = f"on ({settings.browser_app})" if settings.browser_enabled else "off"
        print(f"browser:    {browser}")
        print(f"mcp:        {'on' if settings.mcp_enabled else 'off'}")
        print()
        try:
            available = builtin_tools(
                roots=settings.tools_roots,
                timeout_secs=settings.tools_timeout_secs,
                max_output=settings.tools_max_output,
            )
            if settings.browser_enabled:
                # Listed here too, or this command answers "what may it do" with
                # three of the answers missing.
                from daemon.tools.browser import browser_tools

                available += browser_tools(
                    app=settings.browser_app,
                    timeout_secs=settings.tools_timeout_secs,
                    max_output=settings.tools_max_output,
                )
        except Exception as exc:
            print(f"daemon: the built-in tools cannot be built: {exc}", file=sys.stderr)
            return PROBLEM
        for tool in available:
            gate = "asks first" if tool.risk == "guarded" else "runs freely"
            print(f"  {tool.spec.name:<14} {gate}")
        # MCP tools are not listed: finding out what a server offers means starting
        # it, and an operator command should not spawn subprocesses to print a list.
        if settings.mcp_enabled:
            print("\n  plus whatever mcp.json's servers offer, once the daemon is running")
        return OK

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        if args.tools_command == "log":
            rows = store.recent_tool_calls(30)
            if not rows:
                print("nothing yet. No tool has been asked for on this install.")
                return OK
            for row in rows:
                ran = "ran" if row["ran"] else row["verdict"]
                mark = "" if row["ok"] in (1, None) else " FAILED"
                print(f"  {row['ts']}  {ran:<6}{mark}  {row['preview']}")
                if not row["ran"]:
                    print(f"                                    {row['reason']}")
            return OK

        if args.tools_command == "pending":
            from daemon import clock

            rows = store.pending_tool_approvals(now=clock.now())
            if not rows:
                print("no approvals are waiting.")
                return OK
            for row in rows:
                print(f"  {row['code']}  {row['preview']}  (lapses {row['expires_at']})")
            print("\nApprovals are answered in the conversation, not here: reply /approve <code>.")
            return OK

        removed = store.remove_tool_allowlist_entry(args.pattern)
        if not removed:
            standing = store.all_tool_allowlist()
            print(f"daemon: nothing standing matches {args.pattern!r}", file=sys.stderr)
            if standing:
                print("granted:", file=sys.stderr)
                for row in standing:
                    print(f"  {row['pattern']}  ({row['tool']})", file=sys.stderr)
            return USAGE
        print(f"forgotten. {args.pattern!r} will be asked about again.")
        return OK
    finally:
        store.close()


def _reindex(settings: Settings) -> int:
    from daemon.app import DB_FILENAME
    from daemon.memory.curated import rebuild as rebuild_curated
    from daemon.memory.entities import rebuild as rebuild_entities
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        inserted = reindex(settings.data_dir, store)
        # The other two markdown tiers are mirrors too, and a rebuild that only
        # restored messages would silently drop every curated fact and entity note
        # the reflection pass had concluded - which is the thing non-negotiable 1
        # exists to make impossible.
        rebuild_curated(settings.data_dir, store)
        rebuild_entities(settings.data_dir, store)
        return inserted
    finally:
        store.close()


async def _proactive(settings: Settings, *, speak: bool = False) -> int:
    """One tick, printed. This is how the deterministic half gets checked.

    Without `--speak` it cannot speak at all: the tick is assembled with no judge
    and no delivery, so there is no gateway, no channel and no speaker to reach.
    That is the order PLAN 6.4 asks for - a gate whose verdicts nobody has read is
    not a gate anyone should trust with a speaker - and it is also the cheapest way
    to see why the daemon has been quiet, since it costs no model call.

    What it prints either way: the reading the probes took, every candidate that was
    due, and which rule allowed or blocked each one.
    """
    from daemon.app import build_proactive_tick

    tick, closing = await build_proactive_tick(settings, speak=speak)
    try:
        result = await tick.run()
    finally:
        await closing()

    reading = result.reading
    idle = "unknown" if reading.idle_seconds is None else f"{reading.idle_seconds:.0f}s"
    audio = {True: "busy", False: "free", None: "unknown"}[reading.audio_busy]
    print(
        f"presence: idle {idle} · app {reading.foreground_app or 'unknown'} · audio {audio}"
    )
    for reason in reading.unknown:
        print(f"  ! {reason}")

    if result.disabled:
        print("\nproactivity is off (DAEMON_PROACTIVE_ENABLED=true to turn it on).")
        print("Nothing was generated, so switching it on later starts from today.")
        return OK

    print(f"\ngenerated {result.generated} new candidate(s), expired {result.expired}")
    if not result.considered:
        print("nothing is due. That is the default and usually the right answer.")
        return OK

    for item in result.considered:
        mark = "SPEAK" if item.verdict.allowed else "  -  "
        where = f" -> {item.verdict.delivery}" if item.verdict.allowed else ""
        print(f"\n[{mark}] {item.candidate.kind}{where}")
        print(f"        why : {item.verdict.why}")
        print(f"        said: {item.candidate.reason}")
    blocked = result.blocked_by
    if blocked:
        print("\nblocked by: " + " · ".join(f"{rule} x{n}" for rule, n in sorted(blocked.items())))
    if result.declined:
        print(f"\n{result.declined} allowed, and there was nothing worth saying.")
        print("That is the default answer and usually the right one.")
    if result.spoke:
        said = next(item for item in result.considered if item.delivered)
        print(f"\nspoke via {said.delivered.route}: {said.utterance.text}")
        print(f"  label it with the buttons on the message (id {said.delivered.utterance_id})")
    elif result.allowed and not speak:
        print(f"\n{len(result.allowed)} would have been spoken. Add --speak to let it.")
    return OK


async def _reflect(settings: Settings, *, date: str | None, force: bool) -> int:
    """Run the reflection pass now. This is also how the pass is verified at all:
    the scheduler runs it at 04:00 and nobody is awake to read the log."""
    from daemon.app import build_reflection

    reflection, closing = await build_reflection(settings)
    try:
        results = (
            [await reflection.run(date, force=force)]
            if date
            else await reflection.catch_up()
        )
    finally:
        await closing()

    if not results:
        print("nothing to reflect on: no day has a log without a reflection already.")
        return OK
    for result in results:
        print(
            f"{result.date}: {result.status}"
            + (f" - {result.detail}" if result.detail else "")
            + (
                f" ({result.messages_read} message(s) -> {result.facts} fact(s), "
                f"{result.entities} entity(ies), {result.observations} observation(s))"
                if result.status == "written"
                else ""
            )
        )
        for problem in result.problems:
            print(f"  ! {problem}")
    return OK if all(result.ok for result in results) else PROBLEM


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
            _memory_check(settings),
            _proactivity_check(settings),
            _tools_check(settings),
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


def _tools_check(settings: Settings) -> Check:
    """What Daemon may do to this machine, said out loud.

    Here because tools are on by default: a capability nobody was asked about and
    which is reported nowhere is the silent state this project keeps being bitten by
    (CLAUDE.md, "report state, do not assume it"). `daemon setup` does not ask about
    tools, so this line and the startup log are the only places the answer appears.
    """
    if not settings.tools_enabled:
        return Check("tools", True, "off (DAEMON_TOOLS_ENABLED=true to turn it on)")

    extras = []
    if settings.browser_enabled:
        extras.append(f"browser={settings.browser_app}")
    if settings.mcp_enabled:
        extras.append("mcp=on")
    allowed = ", ".join(settings.tools_allowlist) or "nothing"
    detail = (
        f"on mode={settings.tools_mode} roots={', '.join(settings.tools_roots)} "
        f"runs-without-asking={allowed}"
    )
    if extras:
        detail += " " + " ".join(extras)
    # `full` is the one setting where nothing but the origin gate is left, so it is
    # reported as a failed check rather than as a detail someone might skim past.
    if settings.tools_mode == "full":
        return Check("tools", False, detail + " - `full` asks about nothing")
    return Check("tools", True, detail)


def _proactivity_check(settings: Settings) -> Check:
    """Whether it is on, and what the labels say.

    The label counts are the M3 gate itself - *"오답률·방해도가 허용 범위. 스토커도
    죽은봇도 아니다"* is not answerable from a log line, and docs/PLAN.md 8.1 says
    the label clock cannot be compressed: precision needs dozens of real presses
    over weeks. Printing the tally here is what makes the weeks visible.
    """
    from daemon.app import DB_FILENAME
    from daemon.memory.store import Store

    if not settings.proactive_enabled:
        return Check("proactivity", True, "off (DAEMON_PROACTIVE_ENABLED=true to turn it on)")

    # A missing seed is a *blocker*, not a cosmetic gap: the judge declines every
    # candidate without one rather than speaking as a generic assistant, so
    # proactivity would look switched on and never say a word. Reported by the
    # agent that wrote the judge, which could not reach this file.
    seed = settings.data_dir / "persona" / "seed.md"
    if not seed.exists() or not seed.read_text(encoding="utf-8").strip():
        return Check(
            "proactivity",
            False,
            f"on, but {seed} is empty or missing. Every candidate will be declined "
            "rather than spoken in a generic voice - run `daemon setup` to write a "
            "persona seed.",
        )

    speaker = "speaker on" if settings.proactive_speaker_enabled else "telegram only"
    quiet = settings.proactive_quiet_hours or "no quiet window"
    detail = (
        f"on, {speaker} · budget {settings.proactive_daily_budget}/day "
        f"({settings.proactive_open_loop_budget} open_loop) · quiet {quiet}"
    )

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check("proactivity", True, f"{detail}\n         nothing spoken yet")
    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - doctor reports state, it does not die
        return Check("proactivity", True, f"{detail}\n         labels unreadable ({exc})")
    try:
        counts = store.label_counts()
        total = sum(n for verdict, n in counts.items() if verdict != "responded")
        if not total:
            return Check("proactivity", True, f"{detail}\n         nothing spoken yet")
        good, bad = counts.get("good", 0), counts.get("bad", 0)
        judged = good + bad
        line = (
            f"{total} spoken · {good} good, {bad} bad, "
            f"{counts.get('unlabeled', 0)} unlabeled · {counts.get('responded', 0)} replied to"
        )
        if judged:
            line += f" · precision {good / judged:.0%}"
        else:
            # Said plainly: an unlabelled history is not a good result, it is no
            # result, and the tuning PLAN 6.2 defers to labels cannot start.
            line += " · no labels yet, so precision is unknown"
        return Check("proactivity", True, f"{detail}\n         {line}")
    finally:
        store.close()


def _memory_check(settings: Settings) -> Check:
    """What reflection has actually built.

    Reported rather than logged because an empty graph and a working one look
    identical from the outside, and the M2 gate is "an entity graph I did not fix
    by hand is worth reading" - so it has to be readable without opening sqlite.
    The backlog is in here for the same reason: a reflection loop that has run
    zero times leaves no trace anywhere else.
    """
    from daemon.app import DB_FILENAME
    from daemon.memory.entities import EntityNotes
    from daemon.memory.store import Store
    from daemon.reflection import pending_days

    # The backlog comes off the filesystem, so it is known even when the mirror is
    # not. That ordering is the fix for a real hole: with the sqlite file deleted -
    # which non-negotiable 1 calls a legitimate state - this reported "nothing
    # recorded yet" and hid a month of unreflected log behind it.
    backlog = pending_days(settings.data_dir)
    behind = (
        f"{len(backlog)} day(s) not reflected on yet (oldest {backlog[0]})"
        " - run `daemon reflect`"
        if backlog
        else ""
    )

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check("memory", True, behind or "nothing recorded yet")

    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - doctor reports state, it does not die
        # A database from a newer build refuses to open, and the schema check above
        # already says so with the version numbers. Letting this one raise turned
        # the whole of `daemon doctor` into a traceback - the command whose entire
        # job is to explain what is wrong.
        return Check("memory", True, f"not readable ({exc.__class__.__name__}); see schema above")

    try:
        notes = EntityNotes(settings.data_dir, store)
        graph = notes.graph(limit=10_000)
        detail = (
            f"{store.count_entries()} curated fact(s), {len(graph)} entity(ies), "
            f"{store.count_observations()} observation(s)"
        )
        if behind:
            detail += f"; {behind}"
        for name, mentions, linked in graph[:5]:
            arrow = f" -> {', '.join(linked)}" if linked else ""
            detail += f"\n         {name} ({mentions}){arrow}"
        return Check("memory", True, detail)
    finally:
        store.close()


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


def _print_action(action: ServiceAction, *, verb: str = "installed") -> int:
    if action.changes:
        print(f"changes to {action.unit_path}:")
        for line in action.changes:
            print(f"  {line}")
    for note in action.notes:
        print(note)
    for command in action.commands:
        print(f"ran: {' '.join(command)}")
    if action.applied:
        # `applied` only says something happened. Printing "installed" for every
        # outcome meant a successful uninstall reported "ai.daemon.default
        # installed at ..." - the action was right and the sentence was wrong,
        # which is worse than a crash because nothing looks broken.
        print(f"{action.label} {verb}: {action.unit_path}")
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
