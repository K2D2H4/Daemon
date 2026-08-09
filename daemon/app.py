"""Single-process entrypoint: FastAPI + in-process APScheduler.

CONTRACTS.md 9: one process. No Celery, no Redis, no separate worker - there is
exactly one user, so a distributed queue would be cost without a reason.

This module is the only place allowed to break the layering rule and import
concrete providers, channels and memory writers. Everything else talks to
protocols. The imports are function-local to keep that exception visible and
to keep `import daemon.app` cheap.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon import __version__
from daemon.channels.base import Channel
from daemon.companion import TOOL_CONTRACT, Companion, ResolveId
from daemon.config import ANTHROPIC, ENV_FILE, GEMINI, OLLAMA, OPENAI, ConfigError, Settings
from daemon.llm.base import Provider
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop
from daemon.memory.base import MemoryWriter, Recall
from daemon.persona.evolve import PersonaEvolution
from daemon.proactivity.tick import ProactiveTick
from daemon.reflection import Reflection
from daemon.tasks import Task

if TYPE_CHECKING:  # the wake gate, used only in the signatures below
    from daemon.voice.base import AudioIO, SpeechRecognizer
    from daemon.voice.wake import WakeGate

WakeRound = Callable[[], Awaitable[None]]
"""One round of listen-until-called-then-converse.

Defined at runtime rather than under TYPE_CHECKING because `create_app` takes one
as a real argument, and a name that only exists for a type checker is a name that
fails at import for anyone reading the signature literally."""

logger = logging.getLogger(__name__)

WAKE_RETRY_SECONDS = 5.0
"""Floor between wake rounds after a failure.

Same reason the Telegram poll has one (daemon/channels/telegram.py): a round
that fails instantly - no microphone, a revoked permission - would otherwise
retry as fast as the process can manage."""

WAKE_REARM_SETTLE_SECONDS = 1.0
"""Pause after a spoken conversation before the next wake round opens a fresh
microphone stream.

The conversation ran through the macOS Voice-Processing I/O unit
(daemon/voice/apple_audio.py), which releases the shared input device
*asynchronously* when the engine stops. A fresh PortAudio capture opened inside that
window has come up dead - all-zero blocks, no error - leaving the gate `running` and
permanently deaf until a restart (measured live: a session worked, the next wake word
was never heard). Letting CoreAudio finish the teardown first is the cheap half of the
fix; the wake gate's own dead-stream detection (daemon/voice/wake.py) is the half that
recovers if this is ever not enough. Pure wait - it can cost a second of not-yet-
listening, never the listening itself."""

OK = 0
PROBLEM = 1
"""Shell exit codes, matching cli.py - `daemon voice` is a command."""

DB_FILENAME = "daemon.sqlite3"
"""Lives inside the data dir. Deleting it must never lose user data - the
markdown log is the original (CONTRACTS.md non-negotiable 1)."""

PROACTIVE_TICK_MINUTES = 5
"""docs/PLAN.md 6.1's tick. Deterministic work only, unless a candidate is due and
passes the gate - so the cost of the interval is a few sqlite reads and three
subprocess probes, not a model call."""

REFLECT_HOUR = 4
"""Local hour for the nightly pass. Late enough that the day is over, early enough
that the morning's first message already sees what it concluded."""

PERSONA_DAY = "mon"
PERSONA_HOUR = 5
"""Weekly persona-evolution pass: Monday, 05:00 local - one hour after reflection
so a week's worth of observations has already had its last night's reflection
land before evolution reads them."""


def create_app(
    settings: Settings | None = None,
    *,
    channel: Channel | None = None,
    memory: MemoryWriter | None = None,
    recall: Recall | None = None,
    wake: WakeRound | None = None,
) -> FastAPI:
    """Assemble the process. `channel`/`memory`/`recall`/`wake` are injection points
    for tests; normally all four are built from settings during startup.

    `wake` is one round of "listen until called, then hold a conversation" - see
    `_wake_round`. Injected rather than assembled in tests because the real one
    opens a microphone and then a billed session, and neither belongs in a test.
    """
    resolved = settings or Settings()
    app = FastAPI(title="Daemon", version=__version__, lifespan=_lifespan)
    app.state.settings = resolved
    app.state.channel = channel
    app.state.memory = memory
    app.state.recall = recall
    app.state.recall_status = "injected" if recall is not None else "not started"
    app.state.loop_task = None
    app.state.reflection_boot_task = None
    app.state.wake_round = wake
    app.state.wake_task = None
    app.state.wake_status = "off"
    app.state.tools = None
    app.state.tools_status = "not started"
    app.state.mcp = None
    # Serialises the admin's persist-then-(dis)connect MCP routes so two of them
    # cannot interleave the `mcp.json` read-modify-write and lose an update
    # (daemon/admin/routes.py, finding #6). Constructed here, not in the lifespan,
    # because the routes run under `TestClient` without one; it binds to the running
    # loop on first use.
    app.state.mcp_persist_lock = asyncio.Lock()
    # Where a settings patch is written and where `.env` is read from - the same
    # file Settings loaded (config.ENV_FILE, cwd-relative), so the admin edits the
    # one source of truth rather than a second copy. A test overrides it.
    app.state.env_path = Path(ENV_FILE)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return health_payload(app.state, resolved)

    # The operator-facing control plane: health, a side-effect-free chat test, and
    # validated settings edits (docs/design/2026-08-07-m5-admin-web-design.md). It
    # talks only to `app.state` handles and protocols; importing a concrete
    # provider/channel stays this file's exception alone (CONTRACTS 4), so the
    # import is here and function-local like every other one in this module.
    from daemon.admin.routes import router as admin_router

    app.include_router(admin_router)

    return app


def health_payload(state: Any, settings: Settings) -> dict[str, Any]:
    """The `/health` body, built from `app.state`.

    A module-level function rather than a closure so the admin's own health
    endpoint returns byte-for-byte the same thing - "the chat version of /health"
    is only honest if the health it shows is the same health (docs/design)."""
    task = getattr(state, "loop_task", None)
    return {
        "status": "ok",
        "preset": settings.preset,
        "voice_enabled": settings.voice_enabled,
        "routing": {
            task_key.value: route.provider
            for task_key, route in settings.routing_table().items()
        },
        "conversation_loop": "running" if task is not None and not task.done() else "stopped",
        # Recall can be absent while the rest of the process is healthy (the
        # embedder is down, the module is mid-rewrite). Saying so here is the
        # difference between a degraded daemon and one that quietly forgets.
        "recall": _recall_health(state),
        # Same reason, and the failure is even quieter: a wake gate that died
        # leaves a daemon that answers Telegram normally and has simply stopped
        # hearing the room, with nothing anywhere saying so.
        "wake_gate": _wake_health(state),
        # macOS: a wake gate can be "running" while the mic is denied, which
        # reads as a quiet room. Naming the grant here is the difference between
        # a diagnosable and an invisible failure (spec §6).
        "mic": _mic_health(),
        # Same reasoning: an MCP server that failed to start leaves the model
        # with fewer tools and nothing else different, which is exactly the kind
        # of quiet degradation this endpoint exists to name.
        "tools": _tools_health(state),
    }


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    providers = _build_providers(settings)
    gateway = LLMGateway(
        providers, settings.routing_table(), fallback=settings.fallback_route()
    )
    app.state.gateway = gateway

    scheduler = AsyncIOScheduler(timezone="UTC")
    # One lock shared by the two catch-up crons and the boot task below. Each cron's
    # `max_instances=1` only stops it overlapping *itself*; it says nothing about the
    # boot task, which fires at startup and can land on top of a ~04:00 cron with a
    # backlog. Two `run(date)` for one day would double-write the append-only
    # reflection artifact and double-insert its observations (append-only, no dedup),
    # corrupting the M4 log clock (docs/PLAN.md 8.1). Whoever acquires it second finds
    # the day's artifact/diary already written and skips.
    catchup_lock = asyncio.Lock()
    # Local time, not UTC: "overnight" is a fact about the person asleep next to
    # the machine, and a UTC 04:00 lands mid-afternoon in KST. The 5-minute
    # proactivity tick (M3) lands here too.
    scheduler.add_job(
        _reflect_tick,
        "cron",
        hour=REFLECT_HOUR,
        minute=0,
        timezone=None,
        args=[settings, catchup_lock],
        id="reflection",
        # A pass that overruns until the next night's must not stack up, and a
        # machine that was asleep at 04:00 should still reflect when it wakes -
        # `catch_up` covers every missed day anyway, so one late run is enough.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
    )
    scheduler.add_job(
        _persona_tick,
        "cron",
        day_of_week=PERSONA_DAY,
        hour=PERSONA_HOUR,
        minute=0,
        timezone=None,
        args=[settings, catchup_lock],
        id="persona-evolution",
        # Same guards as reflection: an overrunning pass must not stack, and a
        # machine asleep on Monday morning should still evolve when it wakes -
        # `PersonaEvolution.run`'s own diary-file gate covers a late run, so one
        # catch-up is enough and there is no `catch_up` loop here to misfire.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
    )
    if settings.proactive_enabled:
        # Registered only when the user asked for it. A job that wakes every five
        # minutes to decide against speaking is cheap but not free, and its absence
        # is a clearer statement of "off" than a disabled job that still fires.
        scheduler.add_job(
            _proactive_tick,
            "interval",
            minutes=PROACTIVE_TICK_MINUTES,
            args=[settings],
            id="proactivity",
            max_instances=1,
            coalesce=True,
            # A laptop that was asleep must not fire the ticks it missed: the
            # candidates are still there and the gate would say the same thing, so
            # the only effect would be several rounds of judging in one second.
            misfire_grace_time=60,
        )
    scheduler.start()
    app.state.scheduler = scheduler

    channel = app.state.channel
    memory = app.state.memory
    recall: Recall | None = app.state.recall
    resolve_id: ResolveId | None = None
    close_io: Callable[[], None] | None = None
    embedder: Any = None
    tools: Any = None
    if channel is None or memory is None:
        try:
            io = _build_io(settings)
        except Exception as exc:
            # Loud, not silent: /health will report the loop as stopped.
            logger.error("conversation loop not started: %s", exc)
            channel = memory = None
            app.state.recall_status = "not started"
        else:
            channel, memory = io.channel, io.memory
            recall, resolve_id, close_io = io.recall, io.resolve_id, io.close
            embedder = io.embedder
            app.state.recall = recall
            app.state.recall_status = io.recall_status
            tools, app.state.mcp, app.state.tools_status = await _build_tools(
                settings, io.store
            )
            app.state.tools = tools

    # Set regardless of the channel: memory, recall and the tool layer are
    # channel-independent capabilities, and tying their `app.state` handles to a
    # working channel made `/health` under-report them - and left the admin surface
    # (M5) without the memory it is entitled to - in a tokenless boot. The
    # conversation loop, and only the loop, needs the channel; that guard stays
    # below.
    app.state.channel = channel
    app.state.memory = memory
    if channel is not None and memory is not None:
        loop = ConversationLoop(
            channel,
            gateway,
            Companion(
                memory,
                data_dir=settings.data_dir,
                recall=recall,
                recall_limit=settings.recall_limit,
                resolve_id=resolve_id,
                tools=tools,
            ),
            max_tool_rounds=settings.tools_max_rounds,
        )
        task = asyncio.create_task(loop.run(), name="conversation-loop")
        # run() only guards individual turns; anything raised by the channel's own
        # listen() - a revoked bot token surfaces as TelegramFatal - ends the task.
        # Without this the process stays alive and healthy-looking with no inbound
        # path and not one line in the log, because app.state holds the reference
        # so even asyncio's "never retrieved" warning never fires.
        task.add_done_callback(_report_loop_death)
        app.state.loop_task = task

    if recall is not None:
        # Backfill after the loop is already serving, and in the background: a
        # rebuilt sqlite file gives every message a new id and drops `embeddings`
        # by cascade, so without this the vector lane stays empty for all history
        # while /health still says recall is ready. Measured on the golden set,
        # that silent state is a 50% ceiling for Korean rather than the hybrid
        # number - a regression where nothing fails. Not awaited, because a cold
        # embedder must not delay the log clock (docs/PLAN.md 8.1).
        app.state.backfill_task = asyncio.create_task(
            _backfill(recall), name="recall-backfill"
        )

    if memory is not None:
        # Boot-time catch-up for the reflection and persona passes. Their crons are
        # local-time jobs (04:00, and Monday 05:00) that a machine powered off
        # overnight sleeps straight through, so without this the log clock
        # (docs/PLAN.md 8.1) never advances for that user. Not awaited, and for the
        # same reason as the backfill just above: a cold-boot backlog is up to 14
        # sequential model calls, and awaiting it before the `yield` would block
        # uvicorn's "startup complete" and /health. Gated on the writer because
        # catch-up reads the mirror `_build_io` has just rebuilt from markdown - if
        # that failed (`memory is None`) there is nothing to read.
        app.state.reflection_boot_task = asyncio.create_task(
            _boot_catchup(settings, catchup_lock), name="reflection-boot-catchup"
        )

    # The always-on gate. This is what makes "it hears me when I call it" a property
    # of the resident process rather than of a command somebody remembered to run -
    # `daemon wake test` proves the gate works, and this is what uses it.
    wake_round: WakeRound | None = app.state.wake_round
    if wake_round is not None:
        # Injected: a test drives the resident behaviour without a microphone.
        app.state.wake_task = asyncio.create_task(_rounds(wake_round), name="wake-gate")
    elif settings.wake_enabled:
        try:
            await _claim_microphone(settings)
            # Built once here so an unavailable recognizer or a missing model is
            # reported at startup instead of five seconds into a retry loop that
            # looks like a quiet room.
            recognizer = build_wake_recognizer()
            if not recognizer.available:
                raise RuntimeError(
                    "no on-device speech recognizer, so nothing could hear a wake word; "
                    "install the voice extra (pip install -e '.[voice]') or turn "
                    "DAEMON_WAKE_ENABLED off"
                )
        except Exception as exc:
            # Not fatal: Telegram still works and the daemon is still worth running.
            # Loud, and /health says `unavailable` rather than `off`, because the two
            # mean different things to whoever is wondering why it stopped answering.
            logger.error("wake gate not started: %s", exc)
            app.state.wake_status = "unavailable"
        else:
            app.state.wake_task = asyncio.create_task(
                _wake_forever(settings), name="wake-gate"
            )

    try:
        yield
    finally:
        # `suppress(CancelledError)` alone was not enough. A task that had
        # *already* finished with some other exception re-raises it on await, and
        # that escaped the finally block - skipping the channel close, the sqlite
        # close, the scheduler shutdown and every provider aclose below it. A
        # revoked bot token was enough to leak the lot on every restart.
        for name in ("backfill_task", "loop_task", "wake_task", "reflection_boot_task"):
            pending = getattr(app.state, name, None)
            if pending is None:
                continue
            pending.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await pending
        if channel is not None:
            with suppress(Exception):
                await channel.close()
        mcp = getattr(app.state, "mcp", None)
        if mcp is not None:
            # Before the sqlite close and the scheduler shutdown, because these are
            # child processes: a stdio MCP server left running is one more orphan
            # per restart.
            with suppress(Exception):
                await mcp.aclose()
        if close_io is not None:
            with suppress(Exception):
                close_io()
        scheduler.shutdown(wait=False)
        closeables: list[Any] = [*providers.values()]
        if tools is not None:
            # `fetch_page` owns an HTTP client; the runner walks the registry for it.
            closeables.append(tools)
        if embedder is not None:
            # The embedder holds its own HTTP client, and it is not in `providers`
            # because embeddings are routed separately (llm/base.py).
            closeables.append(embedder)
        for closeable in closeables:
            closer = getattr(closeable, "aclose", None)
            if closer is not None:
                with suppress(Exception):
                    await closer()


def _build_providers(settings: Settings) -> dict[str, Provider]:
    """Only the providers the routing table actually names get built."""
    from daemon.llm.providers.anthropic import AnthropicProvider
    from daemon.llm.providers.gemini import GeminiProvider
    from daemon.llm.providers.ollama import OllamaProvider
    from daemon.llm.providers.openai import OpenAIProvider

    wanted = {route.provider for route in settings.routing_table().values()}
    fallback = settings.fallback_route()
    if fallback is not None:
        wanted.add(fallback.provider)

    providers: dict[str, Provider] = {}
    for name in sorted(wanted):
        if name == OLLAMA:
            providers[name] = OllamaProvider(settings.ollama_base_url)
        elif name == ANTHROPIC:
            providers[name] = AnthropicProvider(settings.anthropic_api_key)
        elif name == OPENAI:
            providers[name] = OpenAIProvider(settings.openai_api_key)
        elif name == GEMINI:
            providers[name] = GeminiProvider(
                settings.gemini_api_key, thinking_level=settings.gemini_thinking_level
            )
        else:
            raise ConfigError(
                f"routing names provider {name!r}, which has no implementation yet "
                f"(M1a ships {OLLAMA} and {ANTHROPIC})"
            )
    return providers


@dataclass(frozen=True, slots=True)
class _IO:
    channel: Channel | None
    memory: MemoryWriter
    recall: Recall | None
    recall_status: str
    resolve_id: ResolveId | None
    close: Callable[[], None]
    embedder: Any = None
    """Held only so the lifespan can close its HTTP client on shutdown."""
    store: Any = None
    """The open sqlite handle. Carried out because the tool layer needs it for the
    approval and audit tables, and building those inside `_build_io` would mean
    starting MCP subprocesses from a synchronous function."""


def _build_channel(settings: Settings, store: Any) -> Channel:
    """The one place the concrete channel is constructed.

    Shared by the conversation loop and the proactivity tick so the two cannot end
    up with different allowlists or different pairing policies - "who may talk to
    Daemon" and "who Daemon speaks to unprompted" are the same person, and two
    construction sites is how they would quietly stop being.
    """
    from daemon.channels.pairing import Pairing
    from daemon.channels.telegram import TelegramChannel

    # Pairing by default: on a first run the env allowlist is empty, and in
    # `allowlist` mode that refuses to start - correct as a policy, useless as an
    # onboarding step. The owner's id is captured from their first message instead
    # of transcribed by hand.
    pairing = (
        Pairing(store, TelegramChannel.name)
        if settings.telegram_dm_policy == "pairing"
        else None
    )
    return TelegramChannel(
        settings.telegram_bot_token,
        settings.telegram_allowed_user_ids,
        cursor=store,
        dm_policy=settings.telegram_dm_policy,
        pairing=pairing,
        # Without this a 👍 press is received, authorised, and then dropped with an
        # error - so the label clock (docs/PLAN.md 8.1) reads as "the owner never
        # labels anything" and M3's own gate has nothing to measure.
        labels=store,
    )


def _build_io(settings: Settings) -> _IO:
    """Wire the concrete channel, memory writer and recall, plus their teardown.

    TelegramChannel raises on an empty token or an empty allowlist, so those
    checks are deliberately not repeated here.
    """
    from daemon.fs import harden_existing
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter

    # Installs created before permissions were pinned still hold world-readable
    # conversation logs, and new writes alone would not fix the old files.
    harden_existing(settings.data_dir)

    # Built before the channel: the channel takes the cursor from it, so a
    # restart does not re-receive and re-answer what it already handled.
    store = Store.open(settings.data_dir / DB_FILENAME)
    # Everything after the open is guarded so a failure between here and the return
    # closes the sqlite handle rather than leaking it (finding #7): `reindex` can
    # raise on a corrupt mirror, and the process now keeps running degraded rather
    # than crashing, so a leaked handle would accumulate across restarts.
    try:
        # Repairs a mirror that fell behind its markdown - a failed mirror write, a
        # crash between the two writes, or a deleted database. Without this the
        # markdown being the source of truth is a claim nothing acts on.
        reindex(settings.data_dir, store)
        # Pairing by default: on a first run the env allowlist is empty, and in
        # `allowlist` mode that refuses to start - correct as a policy, useless as an
        # onboarding step. The owner's id is captured from their first message
        # instead of transcribed by hand.
        # A missing bot token must not cost the daemon its memory, tools and admin
        # surface: those are channel-independent, and the admin web (M5) is explicitly
        # a local-only, tokenless-deployment surface. So a channel that will not build
        # degrades to "no inbound conversation path" rather than taking the whole
        # process down - the same tolerance `build_proactive_tick` already applies to
        # its own `_build_channel`. `_lifespan` starts the conversation loop only when
        # the channel is present, so None here simply means no Telegram loop.
        try:
            channel: Channel | None = _build_channel(settings, store)
        except Exception as exc:
            logger.error(
                "channel not started; continuing with memory, tools and admin "
                "(no inbound conversation path): %s",
                exc,
            )
            channel = None
        recall, recall_status, embedder = _build_recall(settings, store)
        writer = FileMemoryWriter(settings.data_dir, store)
        return _IO(
            channel=channel,
            memory=writer,
            recall=recall,
            recall_status=recall_status,
            resolve_id=_id_resolver(writer),
            close=store.close,
            embedder=embedder,
            store=store,
        )
    except Exception:
        store.close()
        raise


BACKFILL_CHUNK = 500
"""Rows per backfill call. Small enough to yield often, and the loop below keeps
going until nothing is left."""


def _report_loop_death(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.critical(
            "conversation loop died; the daemon is running but deaf", exc_info=exc
        )


async def _backfill(recall: Recall) -> None:
    """Embed history the vector lane is missing, to exhaustion.

    One call was not enough. It stopped at its default 500 rows and never ran
    again - no retry, no periodic job - so a rebuilt sqlite file over a year of
    history left the great majority of messages with no vector while /health
    still reported recall ready. That is the same invisible Korean ceiling the
    protocol change was meant to prevent, just further along.

    Never fatal: recall degrades to keyword-only, which is worse than the full
    answer and far better than a dead conversation loop.
    """
    total = 0
    try:
        while True:
            landed = await recall.backfill(BACKFILL_CHUNK)
            total += landed
            if landed < BACKFILL_CHUNK:
                break
            # Let the conversation loop breathe between batches; this runs in the
            # background precisely so a long history does not delay serving.
            await asyncio.sleep(0)
    except Exception:
        logger.exception(
            "recall backfill stopped after %d message(s); the vector lane stays partial",
            total,
        )
        return
    if total:
        logger.info("recall backfill embedded %d message(s)", total)


def _build_recall(settings: Settings, store: Any) -> tuple[Recall | None, str, Any]:
    """Assemble Lane 1 recall. Returns (recall, status, embedder).

    Imported here rather than at module scope because a missing or broken recall
    stack must not stop the process from booting: without this, an embedder that
    cannot reach Ollama would cost the user their conversation loop as well as
    their memory, and the log clock (docs/PLAN.md 8.1) is the thing that cannot
    be caught up later.

    The embedder comes back so the lifespan can close its HTTP client; recall
    itself owns no connection.
    """
    try:
        from daemon.llm.embedders.ollama import OllamaEmbedder
        from daemon.memory.recall import MemoryRecall
    except ImportError as exc:
        logger.warning("recall unavailable, continuing without it: %s", exc)
        return None, f"unavailable: {exc}", None

    try:
        embedder = OllamaEmbedder(settings.ollama_base_url, settings.embed_model)
        recall = MemoryRecall(
            store, embedder, half_life_days=settings.recall_half_life_days
        )
        return recall, "ready", embedder
    except Exception as exc:
        logger.warning("recall could not be built, continuing without it: %s", exc)
        return None, f"unavailable: {exc}", None


async def build_proactive_tick(
    settings: Settings, *, speak: bool = False
) -> tuple[ProactiveTick, Callable[[], Awaitable[None]]]:
    """A tick and the coroutine that releases what it holds.

    `speak=False` assembles only the deterministic half: no gateway, no channel, no
    speaker, and therefore no possibility of an utterance. That is not a debugging
    convenience - PLAN 6.4 asks for the gate to be trustworthy *before* anything is
    wired to a speaker, and a mode where speaking is structurally impossible is a
    stronger statement of that than a flag checked at the end.

    The speaker is built only when the user asked for it *and* the platform can do
    it. Everything else degrades to Telegram, which is the safe direction.
    """
    from daemon.fs import harden_existing
    from daemon.memory.store import Store
    from daemon.proactivity.presence import MachinePresence

    harden_existing(settings.data_dir)
    store = Store.open(settings.data_dir / DB_FILENAME)
    closers: list[Callable[[], Awaitable[None]]] = []

    judge = None
    delivery = None
    if speak:
        from daemon.memory.writer import FileMemoryWriter
        from daemon.proactivity.delivery import ProactiveDelivery
        from daemon.proactivity.judge import Judge

        providers = _build_providers(settings)
        gateway = LLMGateway(
            providers, settings.routing_table(), fallback=settings.fallback_route()
        )
        judge = Judge(gateway, data_dir=settings.data_dir)

        channel = None
        try:
            channel = _build_channel(settings, store)
        except Exception as exc:  # noqa: BLE001 - a missing token must not stop the tick
            # Loud, and then Telegram simply is not a route. The gate still runs and
            # the local speaker may still work, which is more than nothing.
            logger.error("proactive: no channel, so nothing can be delivered there: %s", exc)

        speaker = None
        if settings.proactive_speaker_enabled:
            from daemon.proactivity.speaker import LocalSpeaker

            speaker = LocalSpeaker()
            closers.append(speaker.aclose)

        delivery = ProactiveDelivery(
            store,
            FileMemoryWriter(settings.data_dir, store),
            channel=channel,
            speaker=speaker,
        )
        if channel is not None:
            closers.append(channel.close)
        for provider in providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                closers.append(closer)

    async def close() -> None:
        store.close()
        for closer in closers:
            with suppress(Exception):
                await closer()

    return (
        ProactiveTick(
            store, settings, MachinePresence(), judge=judge, delivery=delivery
        ),
        close,
    )


async def build_reflection(settings: Settings) -> tuple[Reflection, Callable[[], Awaitable[None]]]:
    """A `Reflection` and the coroutine that releases what it holds.

    Assembled here because this is the only file allowed to import concrete
    providers and writers, and `daemon reflect` needs the same object the
    scheduler runs. Returning the closer rather than a context manager keeps it
    usable from both a CLI command and a scheduled job without one of them
    pretending to be the other.
    """
    from daemon.fs import harden_existing
    from daemon.memory.store import Store

    harden_existing(settings.data_dir)
    store = Store.open(settings.data_dir / DB_FILENAME)
    providers = _build_providers(settings)
    gateway = LLMGateway(providers, settings.routing_table(), fallback=settings.fallback_route())

    async def close() -> None:
        store.close()
        for provider in providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                with suppress(Exception):
                    await closer()

    return Reflection(settings.data_dir, store, gateway), close


async def build_persona_evolution(
    settings: Settings,
) -> tuple[PersonaEvolution, Callable[[], Awaitable[None]]]:
    """A `PersonaEvolution` and the coroutine that releases what it holds.

    Follows `build_reflection` exactly, for the same reason: this is the only
    file allowed to import concrete providers, and `daemon persona evolve` needs
    the same object the Monday scheduler job runs - a scheduled pass nobody can
    run by hand is a pass nobody can verify.
    """
    from daemon.fs import harden_existing
    from daemon.memory.store import Store

    harden_existing(settings.data_dir)
    store = Store.open(settings.data_dir / DB_FILENAME)
    providers = _build_providers(settings)
    gateway = LLMGateway(providers, settings.routing_table(), fallback=settings.fallback_route())

    async def close() -> None:
        store.close()
        for provider in providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                with suppress(Exception):
                    await closer()

    return (
        PersonaEvolution(
            settings.data_dir,
            store,
            gateway,
            max_active=settings.persona_max_active_rules,
            max_new=settings.persona_max_new_per_cycle,
            min_observations=settings.persona_min_observations,
        ),
        close,
    )


async def _proactive_tick(settings: Settings) -> None:
    """The five-minute round. Catches everything, for the same reason the reflection
    tick does: a job that raises inside APScheduler is logged once and then the
    schedule carries on, which reads as a working loop that has silently decided
    nothing for a month.

    Logged at INFO even when nothing happened, because "it stayed silent" is the
    output people need to be able to check.
    """
    try:
        tick, close = await build_proactive_tick(settings, speak=True)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("proactive tick could not start: %s", exc)
        return
    try:
        result = await tick.run()
    except Exception as exc:  # noqa: BLE001
        logger.error("proactive tick failed: %s", exc)
        return
    finally:
        with suppress(Exception):
            await close()

    spoken = next((item for item in result.considered if item.delivered), None)
    if spoken is not None and spoken.utterance is not None:
        logger.info(
            "proactive: spoke (%s via %s): %s",
            spoken.candidate.kind,
            spoken.delivered.route if spoken.delivered else "?",
            spoken.utterance.text,
        )
        return
    logger.info(
        "proactive: silent - %d generated, %d considered, %d declined, blocked %s",
        result.generated,
        len(result.considered),
        result.declined,
        result.blocked_by or "nothing",
    )


async def _reflect_tick(settings: Settings, lock: asyncio.Lock | None = None) -> None:
    """The scheduled pass. Catches everything: a job that raises inside
    APScheduler is logged once and then the schedule carries on, which reads as a
    working reflection loop that has silently done nothing for a month.

    `lock`, when passed, serialises the actual `catch_up` against the boot task
    (`_boot_catchup`) running the same pass: both walk the unreflected days, and two
    `run(date)` for one day double-write its append-only artifact and observations.
    A lock-less call (`lock=None`) still works via `nullcontext`."""
    try:
        reflection, close = await build_reflection(settings)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("reflection tick could not start: %s", exc)
        return
    try:
        async with lock if lock is not None else nullcontext():
            results = await reflection.catch_up()
    except Exception as exc:  # noqa: BLE001
        logger.error("reflection tick failed: %s", exc)
        return
    finally:
        with suppress(Exception):
            await close()
    for result in results:
        logger.info(
            "reflection %s: %s (%d message(s) -> %d fact(s), %d entity(ies), %d observation(s))%s",
            result.date,
            result.status,
            result.messages_read,
            result.facts,
            result.entities,
            result.observations,
            f" problems={result.problems}" if result.problems else "",
        )


async def _persona_tick(settings: Settings, lock: asyncio.Lock | None = None) -> None:
    """The weekly persona-evolution pass. Catches everything, same reason as
    reflection and the proactive tick: a job that raises inside APScheduler is
    logged once and then the schedule carries on, which reads as a working
    weekly pass that has silently done nothing for months.

    Logged at INFO even when the pass was skipped, because "not enough
    observations yet" and "already ran this week" both have to be visible
    without opening sqlite - the same reasoning as the reflection tick's log
    line.

    `lock`, when passed, serialises the actual `run` against the boot task the
    same way the reflection tick does: two `run()` in one week would both write
    the week's diary and re-consume observations. A lock-less call (`lock=None`)
    still works via `nullcontext`.
    """
    try:
        evolution, close = await build_persona_evolution(settings)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("persona evolution tick could not start: %s", exc)
        return
    try:
        async with lock if lock is not None else nullcontext():
            result = await evolution.run()
    except Exception as exc:  # noqa: BLE001
        logger.error("persona evolution tick failed: %s", exc)
        return
    finally:
        with suppress(Exception):
            await close()

    logger.info(
        "persona evolve %s: %s (%d observation(s) read -> %d proposed, %d added, "
        "%d retired)%s",
        result.date,
        result.skipped or "ran",
        result.observations_read,
        result.proposed,
        result.added,
        result.retired,
        f" problems={result.problems}" if result.problems else "",
    )


async def _boot_catchup(settings: Settings, lock: asyncio.Lock) -> None:
    """Run the reflection and persona passes once at startup.

    The 04:00 reflection cron and the Monday 05:00 persona cron are local-time
    APScheduler jobs, so a user who powers the machine off overnight is never on
    when they fire and the M4 "two weeks of observations" log clock (docs/PLAN.md
    8.1) never advances. This complements those crons rather than replacing them: it
    calls the very same ticks once at boot, so the passes still run for a machine
    that is only ever on during the day.

    Reflection first, then persona - the crons' own order (04:00 then 05:00) - so
    persona reads the observations this catch-up has just written. Neither pass needs
    a guard for "was it already run": reflection's per-day artifact and persona's
    weekly diary already make each a no-op when nothing is pending, at zero model
    cost, and `lock` covers the one race a boot near 04:00 introduces.

    Both ticks already catch and log their own failures; this wrapper is
    defence-in-depth, so nothing can escape as an unretrieved exception on a
    background task.
    """
    try:
        await _reflect_tick(settings, lock)
        await _persona_tick(settings, lock)
    except Exception:
        logger.exception("boot catch-up failed")


async def run_voice(settings: Settings, *, opening_audio: bytes = b"") -> int:
    """One spoken conversation at this machine, then exit.

    Assembled here rather than inside the daemon's own loop because voice is a
    thing a person starts, not a thing that happens to them: the session is
    billed per minute, so holding one open on the chance of being spoken to is
    pure cost (docs/PLAN.md 6.5). Proactive speech at the machine is the local
    speaker's job and belongs to M3.

    Returns a shell exit code, because the caller is a CLI command.
    """
    from daemon.fs import harden_existing
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter
    from daemon.voice.gemini_live import GeminiLiveError, GeminiLiveSession

    if not settings.voice_enabled:
        logger.error("voice is off; set DAEMON_VOICE_ENABLED=true (see `daemon setup`)")
        return PROBLEM
    # route_for raises with the specific reason - no voice route in this preset,
    # voice disabled, no live model id - which is more use than anything this
    # function could say about it.
    route = settings.route_for(Task.CHAT_VOICE)

    harden_existing(settings.data_dir)
    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        reindex(settings.data_dir, store)
        writer = FileMemoryWriter(settings.data_dir, store)
        recall, _status, embedder = _build_recall(settings, store)
        # The controller for the live-share start/stop tools (Task 2.3). Built
        # only when screen sharing is on at all - `None` otherwise, which is what
        # keeps those two tools off `_build_tools`'s registry entirely. Built here
        # rather than inside `_voice_attempts` because the same instance has to
        # survive a reconnect: the tools registered below hold a reference to it,
        # and a fresh controller per attempt would leave them pointing at a stale
        # one.
        screen_share = None
        if settings.screen_enabled:
            # Guarded like the screen-tool block in `_build_tools`: screen sharing
            # needs Pillow (daemon/voice/screen_share.py imports it at module scope).
            # A missing Pillow must lose only the feature, not crash the whole voice
            # session on the wake word - the failure this caught on the owner's Mac.
            try:
                from daemon.voice.screen_share import ScreenShareController

                screen_share = ScreenShareController()
            except ImportError as exc:
                logger.warning("voice screen sharing off (missing dependency): %s", exc)
        # Tools follow the owner's configured mode - `full` for this install, so a
        # spoken turn runs guarded tools the same as the text path does. A microphone
        # has no relay path, so a spoken turn is the owner's own words and the origin
        # gate is the real boundary; pinning `allowlist` here silently refused every
        # guarded call the owner made by voice, `open_path` among them. The one mode a
        # spoken turn cannot honour is `ask`: it has nowhere to surface an approval, so
        # `ask` would pile up rows that lapse unanswered - the silent degradation this
        # repo calls the dangerous failure - and so degrades to `allowlist` here.
        # The allowlist and standing grants are the same table the text path edits,
        # so voice reads the surface text writes to; it just never adds to it. Off
        # entirely when `DAEMON_TOOLS_ENABLED` is false, exactly like text.
        voice_mode = "allowlist" if settings.tools_mode == "ask" else settings.tools_mode
        tools, mcp_bridge, _tools_status = await _build_tools(
            settings, store, mode=voice_mode, screen_share=screen_share
        )
        companion = Companion(
            writer,
            data_dir=settings.data_dir,
            recall=recall,
            recall_limit=settings.recall_limit,
            # The half the voice path never had. Without a resolver `record` cannot
            # learn the row id, so nothing said out loud was embedded until the next
            # restart's backfill got to it - the vector lane silently missing exactly
            # the words the owner was most likely to ask about later.
            resolve_id=_id_resolver(writer),
            tools=tools,
        )
        # Seed and learned rules both, same as the text path, and through the same
        # `Companion.persona` -> `load_persona`: a conversation surface is a
        # conversation surface, and M4's learned half reaches every one of them
        # except the proactive judge, which stays seed-only on purpose
        # (daemon/proactivity/judge.py).
        seed = await companion.persona()
        # Owner, always: a microphone has no relay path, so a spoken turn is the
        # owner's own words (daemon/voice/conversation.py `_record`), and the origin
        # gate offers tools only to it. Empty when tools are off, which leaves the
        # session declaring none and so never yielding a tool call.
        tool_specs = companion.specs(origin="owner")
        # The tool contract rides with the persona in the system instruction, so the
        # endpoint getting tools inherits the rules the text path already has instead
        # of being written without them - which is exactly how voice came to have no
        # index() call (daemon/companion.py, TOOL_CONTRACT). Only when there is a tool
        # to use: 200 tokens of rules about a capability the model lacks buys nothing.
        instruction_parts = [
            block for block in (seed, TOOL_CONTRACT if tool_specs else "") if block
        ]
        system_instruction = "\n\n".join(instruction_parts) or None
        audio = build_voice_audio()
        # Before the handshake, so the acknowledgement is as close to the wake word
        # as it can be. Nothing is feeding the session yet, so the cue cannot be
        # heard as the owner interrupting.
        await play_ready_cue(audio)

        def new_session() -> Any:
            """A fresh session per attempt. Reconnecting means starting clean: the
            old one carries a half-flushed transcript, a partial-transcript queue
            nobody will read again, and a log filter holding the API key."""
            return GeminiLiveSession(
                api_key=settings.gemini_api_key,
                model=route.model,
                # The persona, plus the tool contract when tools are on offer. Without
                # the persona the model answers as a generic assistant, which is the
                # one voice PLAN 5 says this product must not have.
                system_instruction=system_instruction,
                # What the model may reach this turn. Declared in setup, which is what
                # lets `receive()` ever yield a tool call - a session offered none
                # cannot be asked for one (daemon/voice/base.py).
                tools=tool_specs,
                # Empty and None pass straight through as "leave it to the server",
                # so an unconfigured install sends no `realtimeInputConfig` at all -
                # the behaviour every session had before these settings existed.
                start_sensitivity=settings.voice_start_sensitivity,
                end_sensitivity=settings.voice_end_sensitivity,
                prefix_padding_ms=settings.voice_prefix_padding_ms,
                silence_duration_ms=settings.voice_silence_duration_ms,
                # Empty passes straight through as "leave it to the server", exactly
                # as before this setting existed (daemon/voice/gemini_live.py).
                voice_name=settings.gemini_live_voice,
            )

        screen_pump_factory = None
        if screen_share is not None:
            from daemon.tools.screen import capture_display
            from daemon.voice.screen_share import ScreenSharePump

            def screen_pump_factory(session: Any) -> ScreenSharePump:
                """A fresh pump bound to `session`, built once it is live.
                `_voice_attempts` calls this again on every reconnect, so a
                dropped session's pump is never reused against a new one."""

                async def _capture() -> bytes:
                    jpeg, _w, _h = await capture_display(long_edge=settings.screen_frame_px)
                    return jpeg

                return ScreenSharePump(
                    session=session,
                    capture=_capture,
                    fps=settings.screen_fps,
                    dedup_threshold=settings.screen_dedup_threshold,
                    keepalive_secs=settings.screen_keepalive_secs,
                )

        try:
            return await _voice_attempts(
                new_session,
                audio,
                companion,
                GeminiLiveError,
                opening_audio=opening_audio,
                screen_share=screen_share,
                screen_pump_factory=screen_pump_factory,
            )
        finally:
            with suppress(Exception):
                await audio.close()
            # Before the sqlite close below, because an MCP server is a child process:
            # one left running is an orphan per `daemon voice` run, the same reason the
            # lifespan closes the bridge ahead of the store.
            if mcp_bridge is not None:
                with suppress(Exception):
                    await mcp_bridge.aclose()
            if tools is not None:
                # The runner walks its registry for anything holding a client -
                # `fetch_page`'s, if the browser group is on - the same as the lifespan.
                with suppress(Exception):
                    await tools.aclose()
            if embedder is not None:
                closer = getattr(embedder, "aclose", None)
                if closer is not None:
                    with suppress(Exception):
                        await closer()
    finally:
        store.close()


READY_CUE_HZ = (784.0, 1046.5)
"""Two short notes, G5 then C6, rising. A rising pair reads as "go ahead" where a
falling one reads as "finished", and neither is a word - so it cannot be mistaken for
the daemon speaking or transcribed as one."""

READY_CUE_MS = 90
READY_CUE_GAIN = 0.18
"""Short and quiet on purpose. The cue answers "may I speak now?", and the honest
answer is that until it existed there was none: the wake gate released the microphone
and about a second passed with nothing to say the session was live, so the owner
guessed - and guessing early is how an utterance lands in the handover and is lost."""


def ready_cue(sample_rate: int) -> bytes:
    """A short rising two-note cue as 16-bit mono PCM at `sample_rate`.

    Synthesised rather than shipped as a file: it is 90 ms of arithmetic, a wav in
    the package would need finding at runtime from a LaunchAgent whose working
    directory is not the repo, and this way it is correct at whatever rate the
    device wants.
    """
    import numpy as np

    per_note = int(sample_rate * (READY_CUE_MS / 1000.0) / len(READY_CUE_HZ))
    notes = []
    for hz in READY_CUE_HZ:
        t = np.arange(per_note) / float(sample_rate)
        tone = np.sin(2.0 * np.pi * hz * t)
        # Raised-cosine envelope. A bare sine starts and stops on a discontinuity,
        # which is an audible click and exactly the kind of noise a VAD notices.
        envelope = np.sin(np.pi * np.arange(per_note) / max(per_note - 1, 1))
        notes.append(tone * envelope * READY_CUE_GAIN)
    wave = np.concatenate(notes) if notes else np.zeros(0)
    return np.clip(wave * 32768.0, -32768, 32767).astype("<i2").tobytes()


async def play_ready_cue(audio: AudioIO) -> None:
    """Tell the owner the microphone is theirs.

    Played before the session is even opened, which is the whole point: the sooner it
    lands the less of the handover a person talks into. Never raises - a missing cue
    is a worse conversation, and an exception here would be no conversation.
    """
    try:
        await audio.play(ready_cue(audio.playback_sample_rate))
    except Exception:
        logger.debug("voice: could not play the ready cue", exc_info=True)


VOICE_RECONNECT_ATTEMPTS = 3
"""How many times a dropped conversation is picked back up.

Bounded, and bounded here rather than inside the session, because the two failures
need different answers: a socket that will not *open* should fail fast and let the
caller fall back to text, while a socket that opened, worked, and was then cut has a
conversation in progress that the user is standing in the middle of. Three, because
the thing being ridden out is a transport hiccup or a server-side session limit, and
anything that survives three attempts and 6s is not weather. Sessions bill per
minute, so this is deliberately not a reconnect-forever loop."""

VOICE_RECONNECT_BACKOFF_SECONDS = 2.0
"""Flat, not exponential. The gap the user is standing in is silence, and doubling
it turns a recoverable hiccup into an abandoned conversation."""


async def _voice_attempts(
    new_session: Callable[[], Any],
    audio: AudioIO,
    companion: Companion,
    session_error: type[Exception],
    opening_audio: bytes = b"",
    screen_share: Any = None,
    screen_pump_factory: Callable[[Any], Any] | None = None,
) -> int:
    """Hold a conversation, and pick it back up if the session is cut.

    Until this existed a dropped socket simply ended `daemon voice`: mid-sentence,
    with a shell exit code, and nothing said about whether it was the network or the
    key. The two cases that are worth resuming are a transient close - 1011, or the
    1008 that is really an idle timeout - and a `goAway`, which is the server saying
    it is about to end the session rather than the turn and which used to read
    exactly like a conversation that had finished.

    What is *not* resumed: a permanent failure (a bad key, a wrong model id, a
    malformed setup), because retrying leaves the process alive, healthy-looking and
    mute; and an ordinary idle timeout, because that is the conversation being over.

    `screen_share` and `screen_pump_factory`, if given, are passed to every
    `VoiceConversation` this loop builds - the same controller across every
    attempt (its tools hold a reference to it), but a fresh pump built from
    `screen_pump_factory(session)` for each attempt's own session. A share left
    running by a dropped session must not survive into the next attempt's, and
    `VoiceConversation.run`'s teardown is what guarantees that per session.
    """
    from daemon.voice.conversation import VoiceConversation

    # Carried across attempts until something actually answers it. Dropping it on
    # every reconnect would reopen the exact failure the opening exists to fix: an
    # attempt cut down during the handshake never gets the utterance in front of a
    # model, and the owner is back to saying the wake phrase twice.
    pending_opening = opening_audio
    for attempt in range(1, VOICE_RECONNECT_ATTEMPTS + 1):
        session = new_session()
        conversation = VoiceConversation(
            session,
            audio,
            companion,
            opening_audio=pending_opening,
            screen_share=screen_share,
            screen_pump_factory=screen_pump_factory,
        )
        failure: Exception | None = None
        try:
            await conversation.run()
        except session_error as exc:
            failure = exc
        finally:
            # Reported on every exit path including the failing ones, because a
            # session that cut itself off mid-answer is a session that *ran*. This
            # is the only place these numbers surface: `interruptions` was counted
            # for three milestones and printed nowhere, so a self-interruption on
            # every single turn - a full answer generated and 0.0s of it played -
            # looked like nothing at all from outside.
            logger.info(
                "voice session%s: %s",
                f" (attempt {attempt})" if attempt > 1 else "",
                conversation.stats.describe(),
            )
        if conversation.ended:
            logger.info("voice session ended: %s", conversation.ended)
        if conversation.stats.played_seconds > 0:
            # Something was said back, so the opening utterance has been answered.
            # Re-sending it on a later attempt would have the daemon answer the same
            # question twice.
            pending_opening = b""
        elif pending_opening:
            logger.info("voice: the opening utterance was not answered; carrying it over")

        if failure is None and not getattr(session, "going_away", False):
            return OK
        if failure is not None and getattr(failure, "permanent", False):
            logger.error("voice session failed and retrying cannot help: %s", failure)
            return PROBLEM
        if attempt == VOICE_RECONNECT_ATTEMPTS:
            if failure is not None:
                logger.error(
                    "voice session failed %d times, giving up: %s", attempt, failure
                )
                return PROBLEM
            # A `goAway` on the last attempt is the server ending things, not a
            # fault. The conversation is over; saying so is not an error.
            logger.info("voice: the server ended the session and the retries are spent")
            return OK
        logger.warning(
            "voice: %s; reconnecting in %.0fs (attempt %d of %d)",
            failure or "the server announced it was ending the session",
            VOICE_RECONNECT_BACKOFF_SECONDS,
            attempt + 1,
            VOICE_RECONNECT_ATTEMPTS,
        )
        await asyncio.sleep(VOICE_RECONNECT_BACKOFF_SECONDS)
    return PROBLEM  # pragma: no cover - the loop returns on every path


def _wake_health(state: Any) -> str:
    """What the wake gate is actually doing, not whether it was switched on.

    Four answers, because they need four different actions and `recall`'s history
    is the argument for spelling them out: `"ready"` there was once set when the
    object was built, and three unrelated failures then looked identical from
    outside. Here: `off` is nobody asked for it, `running` is listening,
    `unavailable` is asked-for-but-nothing-to-listen-with (no microphone, no
    on-device recognizer, no model), and `stopped` is it was listening and is not
    any more - the one that used to be invisible.
    """
    task = getattr(state, "wake_task", None)
    if task is None:
        return getattr(state, "wake_status", "off")
    if not task.done():
        return "running"
    return "stopped"


def _mic_health() -> str:
    """The microphone TCC decision, read (never prompted) at request time. `n/a`
    off macOS, where there is no TCC gate."""
    if sys.platform != "darwin":
        return "n/a"
    from daemon.voice.mic_access import microphone_authorization_status

    return microphone_authorization_status()


async def _wake_round(settings: Settings) -> None:
    """Listen until called, hold one spoken conversation, release the microphone.

    One round, because the caller loops: keeping this a single round is what makes
    the gate stop listening while the conversation runs. Two recording streams do
    coexist on a Mac - measured, both opened and both delivered audio - so this is
    not about the device. It is that a gate listening through a conversation hears
    the conversation: the owner's half is speech, it starts with whatever they say
    next, and the alias is one ordinary Korean word (`데몬` arrives as `질문`), so
    the gate would re-fire on the daemon's own conversation.

    Cost shape, which is the reason the gate exists at all: the VAD and the
    recognizer are local and free, and only the `run_voice` call below opens the
    per-minute session (docs/PLAN.md 6.5).
    """
    gate, close_gate = await build_wake_gate(settings)
    fired = None
    try:
        async for event in gate.listen():
            fired = event
            break
    finally:
        # Before the conversation, not after: this is what hands the microphone over
        # and stops the gate hearing what happens next.
        with suppress(Exception):
            await close_gate()
    if fired is None:
        # The stream ended without a wake word - a closed device, or a test's
        # scripted audio running out. Not an error, but not a reason to spin either;
        # the caller's own guard handles the pacing.
        return
    logger.info("wake: heard %r matching %r; opening a voice session", fired.heard, fired.matched)
    # The segment that fired the gate goes with it. Without this the session opens
    # deaf to the question it was opened for: the gate consumed "루시 뭐 해", matched
    # on the alias, and the owner had to say "뭐 해" again into a microphone that had
    # just changed hands.
    await run_voice(settings, opening_audio=fired.pcm)
    # Let the conversation's Voice-Processing unit finish releasing the microphone
    # before the next round opens a fresh capture on it - see WAKE_REARM_SETTLE_SECONDS.
    await asyncio.sleep(WAKE_REARM_SETTLE_SECONDS)


async def _rounds(round_: WakeRound) -> None:
    """Drive an injected round forever, with the same guards as the real loop.

    Exists so `create_app(wake=...)` exercises the *resident* behaviour - repeats,
    survives a failing round - rather than a single call a test could have made
    itself. Sharing the guards with `_wake_forever` matters: a test that proves the
    loop recovers has proved it about the loop the product runs.
    """
    while True:
        try:
            await round_()
        except asyncio.CancelledError:
            # Explicit, though `except Exception` below could not catch it anyway:
            # CancelledError is a BaseException. Kept because the clause is what tells
            # a reader that shutdown is not one of the failures this loop absorbs, and
            # because daemon/voice/conversation.py states it the same way. A mutation
            # removing it changes nothing, which is the honest description of it.
            raise
        except Exception:
            logger.exception("wake: injected round failed; continuing")
            await asyncio.sleep(WAKE_RETRY_SECONDS)
        else:
            await asyncio.sleep(0)


async def _wake_forever(settings: Settings) -> None:
    """Rounds, until cancelled.

    Every failure is caught and the loop continues. A wake gate that dies is the
    failure this repo has shipped three times in a different costume: the process
    stays alive, `/health` says `ok`, Telegram still answers, and the machine has
    simply gone deaf with nothing saying why. `/health`'s `wake_gate` is the other
    half of that guarantee, and it reads `stopped` only if even this loop is gone.

    The floor exists for the same reason the Telegram poll has one: a round that
    fails immediately - no microphone, a revoked device permission - would otherwise
    spin as fast as the process can retry it.
    """
    while True:
        try:
            await _wake_round(settings)
        except asyncio.CancelledError:
            raise  # BaseException, so the clause below would not have caught it
        except Exception:
            logger.exception("wake: round failed; listening again shortly")
            await asyncio.sleep(WAKE_RETRY_SECONDS)
        else:
            await asyncio.sleep(0)  # yield, so a stream that ends instantly cannot pin the loop


async def _claim_microphone(settings: Settings) -> None:
    """macOS: claim the mic grant under the .app identity before PortAudio opens a
    stream that would otherwise return silence (spec D1). Headless-and-granted this
    is instant; ungranted it is the harmless no-op that only `daemon request-mic`
    (foreground, under Daemon.app) can turn into a prompt. Off the event loop
    because the runloop pump is blocking.
    """
    if sys.platform != "darwin":
        return
    from daemon.voice.mic_access import request_microphone_access

    status = await asyncio.to_thread(request_microphone_access, timeout=2.0)
    if status != "authorized":
        logger.warning(
            "wake gate: microphone not granted (%s); run `daemon install` and click "
            "Allow so it can hear the wake word",
            status,
        )


# --- the wake gate ------------------------------------------------------------
# Three assembly points for the always-on gate, all here for the same reason as
# everything else in this module: nothing outside it may import an implementation
# (docs/CONTRACTS.md 4). The commands that use them are in `daemon/wake_cli.py`,
# which talks only to the protocols in `daemon/voice/base.py`.


def build_wake_recognizer() -> SpeechRecognizer:
    """The on-device recognizer, for `daemon wake calibrate`.

    Nothing about it is configurable: it is whatever this OS can do on-device, and
    whether that is anything at all is `SpeechRecognizer.available` - which the
    command asks before it records, because a missing locale and a failed
    transcription are indistinguishable after the fact.
    """
    from daemon.voice.apple_speech import AppleSpeechRecognizer

    return AppleSpeechRecognizer()


def build_wake_audio() -> AudioIO:
    """The microphone, for one calibration take.

    Deliberately *not* the echo-cancelling path, even on macOS. Voice processing
    applies its own gain control, and `DAEMON_WAKE_ALIASES` is a measurement of
    what the recognizer returns for this speaker through this capture path - so
    calibrating through one path and listening through another would quietly
    invalidate the strings the gate matches on.
    """
    from daemon.voice.audio import SoundDeviceAudio

    return SoundDeviceAudio()


def build_voice_audio() -> AudioIO:
    """Microphone and speaker for one conversation, echo-cancelled where possible.

    The conversation is the only place that needs cancellation, and it needs it
    badly: it keeps the microphone open while the model talks, because that is the
    only way a barge-in can be noticed (daemon/voice/conversation.py). Measured on
    this project's target machine, Silero at 0.5 over 10 s of Korean TTS out of the
    speaker - 80.1% of microphone frames read as speech through PortAudio and 0.0%
    through macOS voice processing, while speech from a source the canceller does
    not know about still reads 81.6%. So the echo goes and the user does not.

    The wake gate keeps PortAudio (see `build_wake_audio`), and so does every
    platform without an Apple canceller: PortAudio has no path to enable one, so on
    Linux this is the honest best available rather than a fallback.
    """
    if sys.platform != "darwin":
        from daemon.voice.audio import SoundDeviceAudio

        return SoundDeviceAudio()
    from daemon.voice.apple_audio import VoiceProcessingAudio

    # No automatic fall back to PortAudio if this cannot start. Falling back would
    # mean the daemon interrupting itself while looking healthy, which is the
    # failure class this repo treats as the dangerous one; a readable error naming
    # the engine is worth more than a working-looking session that cuts itself off.
    return VoiceProcessingAudio()


async def build_wake_gate(
    settings: Settings,
) -> tuple[WakeGate, Callable[[], Awaitable[None]]]:
    """The always-on gate and the coroutine that releases the microphone.

    A gate rather than a bare stream of events, because `WakeGate.counters` is the
    other half of the answer: a gate whose recognizer is unavailable, or whose every
    segment is too short, hears nothing forever and looks exactly like a quiet house
    (daemon/voice/wake.py). `daemon wake test` prints those counters, which is what
    makes "nothing fired" a diagnosis instead of a shrug.

    Every knob comes from `.env` rather than from this function's opinion.
    `DAEMON_WAKE_ALIASES` is the one that has to: the on-device recognizer never
    returns a coined name, so the strings that work are the ones measured on the
    owner's own voice by `daemon wake calibrate` (daemon/wake_cli.py).
    """
    from daemon.voice.apple_speech import AppleSpeechRecognizer
    from daemon.voice.audio import SoundDeviceAudio
    from daemon.voice.vad import SileroVad
    from daemon.voice.wake import WakeGate

    audio = SoundDeviceAudio()
    gate = WakeGate(
        audio,
        SileroVad(),
        AppleSpeechRecognizer(),
        settings.wake_aliases,
        threshold=settings.wake_vad_threshold,
        hangover_ms=settings.wake_hangover_ms,
        pre_roll_ms=settings.wake_pre_roll_ms,
        min_speech_ms=settings.wake_min_speech_ms,
        max_segment_ms=settings.wake_max_segment_ms,
        cooldown_seconds=settings.wake_cooldown_seconds,
    )

    async def close() -> None:
        with suppress(Exception):
            await audio.close()

    return gate, close


async def _build_tools(
    settings: Settings, store: Any, *, mode: str | None = None, screen_share: Any = None
) -> tuple[Any, Any, str]:
    """Assemble the tool layer. Returns (runner, mcp bridge, status).

    Nothing here is fatal, on the same principle as `_build_recall`: a broken tool
    configuration should cost the user their tools, not their conversation. The
    difference from recall is that this one is off unless asked for, so "not
    configured" is the ordinary answer rather than a degradation.

    `mode` overrides `DAEMON_TOOLS_MODE`, and only `run_voice` uses it - to degrade
    `ask` to `allowlist`, because a spoken turn has nowhere to ask for approval. `ask`
    there would mint approval rows that lapse unanswered while nobody is watching,
    which is the silent degradation this project treats as the dangerous failure.
    Every other mode passes straight through, so `full` reaches voice exactly as the
    owner configured it. The registry, the allowlist and the standing grants are the
    same as the text path either way - the approval surface is shared, text is its
    editor, and voice only reads it.

    `screen_share`, likewise, is only ever passed by `run_voice` - a
    `ScreenShareController` for the live-share start/stop tools to toggle. `None`
    (the default, and always what `create_app`'s text path passes) means those two
    tools are never registered at all, so the text loop can never offer them.
    """
    if not settings.tools_enabled:
        return None, None, "off (DAEMON_TOOLS_ENABLED)"

    try:
        from daemon.tools.base import Registry
        from daemon.tools.builtin import builtin_tools
        from daemon.tools.policy import ToolPolicy
        from daemon.tools.runner import ToolRunner
    except ImportError as exc:
        logger.error("tool layer unavailable, continuing without it: %s", exc)
        return None, None, f"unavailable: {exc}"

    # The same presence the proactivity tick reads, so `system_state` cannot drift
    # from the gate's own view of whether the owner is here (docs/PLAN.md 6.1).
    from daemon.proactivity.presence import MachinePresence

    registry = Registry()
    try:
        for tool in builtin_tools(
            roots=settings.tools_roots,
            timeout_secs=settings.tools_timeout_secs,
            max_output=settings.tools_max_output,
            presence=MachinePresence(),
        ):
            registry.register(tool)
    except Exception as exc:
        # An unusable DAEMON_TOOLS_ROOTS lands here, and the honest response is no
        # tools at all rather than tools with no idea where they may look.
        logger.error("built-in tools could not be built, continuing without them: %s", exc)
        return None, None, f"unavailable: {exc}"

    if settings.browser_enabled:
        # Guarded like the built-ins above: a name collision or an import failure
        # here would otherwise raise out of startup and take the whole daemon with
        # it, when losing three tools is the proportionate outcome.
        try:
            from daemon.tools.browser import browser_tools

            for tool in browser_tools(
                app=settings.browser_app,
                timeout_secs=settings.tools_timeout_secs,
                max_output=settings.tools_max_output,
            ):
                registry.register(tool)
            logger.info("browser tools on, reading %s", settings.browser_app)
        except Exception as exc:
            logger.error("browser tools could not be built, continuing without: %s", exc)

    if settings.screen_enabled:
        # Guarded like the browser block above: a failure here loses only the
        # screen tool, not startup.
        try:
            from daemon.tools.screen import screen_tools

            for tool in screen_tools(
                max_px=settings.screen_max_px,
                timeout_secs=settings.tools_timeout_secs,
            ):
                registry.register(tool)
            logger.info("screen tools on")
        except Exception as exc:
            logger.error("screen tools could not be built, continuing without: %s", exc)

        if screen_share is not None:
            # The live-share start/stop tools. Only `run_voice` ever passes
            # `screen_share` - the pump they toggle needs a live `VoiceSession`,
            # which the text path never has - so this is voice-only by
            # construction, not by a mode check. Guarded like the block above:
            # losing these two tools should not cost the rest of the tool layer.
            try:
                from daemon.tools.screen import screen_share_tools

                for tool in screen_share_tools(screen_share):
                    registry.register(tool)
                logger.info("screen-share tools on")
            except Exception as exc:
                logger.error(
                    "screen-share tools could not be built, continuing without: %s", exc
                )

    bridge: Any = None
    if settings.mcp_enabled:
        # The oauth-provider factory lives here because this is the one file allowed
        # to import `daemon/admin` (CONTRACTS 4); `tools/mcp.py` must not. It builds a
        # non-interactive provider (redirect/callback handlers raise, never block) so
        # a persisted oauth server reconnects with its stored token at startup and
        # fails gracefully - not a browser hang - when the token is gone.
        from daemon.admin.mcp_oauth import build_reconnect_provider
        from daemon.tools.mcp import McpBridge, ServerConfig, load_config

        redirect_uri = (
            f"http://{settings.host}:{settings.port}/admin/api/mcp/oauth/callback"
        )

        def oauth_provider_factory(config: ServerConfig) -> Any:
            return build_reconnect_provider(
                settings.data_dir, config, redirect_uri=redirect_uri
            )

        bridge = McpBridge(
            load_config(settings.data_dir),
            oauth_provider_factory=oauth_provider_factory,
        )
        try:
            landed = await bridge.start(registry)
        except Exception:
            logger.exception("MCP startup failed; continuing with the built-in tools only")
            landed = 0
        logger.info("MCP contributed %d tool(s)", landed)

    effective_mode = mode or settings.tools_mode
    policy = ToolPolicy(
        store,
        mode=effective_mode,  # type: ignore[arg-type]
        allowlist=settings.tools_allowlist,
        enabled=True,
    )
    runner = ToolRunner(registry, policy, store)
    logger.info(
        "tool layer ready: %d tool(s), mode=%s", len(registry), effective_mode
    )
    browser = f", browser={settings.browser_app}" if settings.browser_enabled else ""
    screen = ", screen=on" if settings.screen_enabled else ""
    return (
        runner,
        bridge,
        f"ready, {len(registry)} tools, mode={effective_mode}{browser}{screen}",
    )


def _tools_health(state: Any) -> str:
    """What the tool layer is actually offering, and what failed to load.

    The failure list is the point. An MCP server that did not start leaves a model
    with fewer tools and no error anywhere the user will see, which reads as "the
    daemon decided not to use it" rather than as the misconfiguration it is.
    """
    status = str(getattr(state, "tools_status", "not started"))
    bridge = getattr(state, "mcp", None)
    failures = getattr(bridge, "failures", None) if bridge is not None else None
    if failures:
        broken = "; ".join(f"{name}: {why}" for name, why in sorted(failures.items()))
        return f"{status}; mcp failed: {broken}"
    return status


def _recall_health(state: Any) -> str:
    """What recall is actually doing, not whether it was constructed.

    `"ready"` used to be set once, when the object was built - before anything had
    asked the embedder a question. Three unrelated failures then looked identical
    from outside: no embedder reachable, a dimension mismatch after a model swap,
    an unfinished backfill. Each one caps Korean recall at the keyword-only
    ceiling (measured: 50% where the hybrid reaches 93%) while raising nothing and
    failing nothing, on a process that stays up for days and whose logs nobody
    reads. The vector count is included because it separates "no embedder" from
    "embedder fine, backfill still working".
    """
    recall = getattr(state, "recall", None)
    if recall is None:
        return state.recall_status
    status = getattr(recall, "vector_lane_status", None)
    if status is None:
        return state.recall_status
    lane = status()
    if lane != "ok":
        return f"degraded: {lane}"
    vectors = recall.vector_count()
    if vectors == 0:
        # Distinguishable from a broken lane on purpose: nothing is wrong, there
        # is simply nothing indexed yet, and it resolves itself as messages
        # arrive or when backfill runs.
        return "ready, nothing indexed yet"
    return f"ready, {vectors} vectors"


def _id_resolver(writer: Any) -> ResolveId:
    """The id of the row `record()` just wrote.

    `MemoryWriter.record()` is frozen and returns nothing, so the loop gets this
    closure rather than a widened protocol - but it now reads the id the writer
    kept from `insert_message`, instead of guessing at "the newest row".
    """

    def resolve(_text: str) -> int | None:
        return getattr(writer, "last_inserted_id", None)

    return resolve


def main() -> None:
    """Entry point for the `daemon` console script (pyproject [project.scripts]).

    Delegates to the CLI so `daemon` keeps starting the server while `daemon
    install`, `daemon doctor` and the rest exist alongside it.
    """
    from daemon.cli import main as cli_main

    raise SystemExit(cli_main())
