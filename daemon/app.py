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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon.channels.base import Channel
from daemon.config import ANTHROPIC, GEMINI, OLLAMA, OPENAI, ConfigError, Settings
from daemon.llm.base import Provider
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop, ResolveId
from daemon.memory.base import MemoryWriter, Recall
from daemon.persona.evolve import PersonaEvolution
from daemon.persona.loader import load_persona
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
    app = FastAPI(title="Daemon", version="0.0.1", lifespan=_lifespan)
    app.state.settings = resolved
    app.state.channel = channel
    app.state.memory = memory
    app.state.recall = recall
    app.state.recall_status = "injected" if recall is not None else "not started"
    app.state.loop_task = None
    app.state.wake_round = wake
    app.state.wake_task = None
    app.state.wake_status = "off"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        task = app.state.loop_task
        return {
            "status": "ok",
            "preset": resolved.preset,
            "voice_enabled": resolved.voice_enabled,
            "routing": {
                task_key.value: route.provider
                for task_key, route in resolved.routing_table().items()
            },
            "conversation_loop": "running" if task is not None and not task.done() else "stopped",
            # Recall can be absent while the rest of the process is healthy (the
            # embedder is down, the module is mid-rewrite). Saying so here is the
            # difference between a degraded daemon and one that quietly forgets.
            "recall": _recall_health(app.state),
            # Same reason, and the failure is even quieter: a wake gate that died
            # leaves a daemon that answers Telegram normally and has simply stopped
            # hearing the room, with nothing anywhere saying so.
            "wake_gate": _wake_health(app.state),
        }

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    providers = _build_providers(settings)
    gateway = LLMGateway(
        providers, settings.routing_table(), fallback=settings.fallback_route()
    )
    app.state.gateway = gateway

    scheduler = AsyncIOScheduler(timezone="UTC")
    # Local time, not UTC: "overnight" is a fact about the person asleep next to
    # the machine, and a UTC 04:00 lands mid-afternoon in KST. The 5-minute
    # proactivity tick (M3) lands here too.
    scheduler.add_job(
        _reflect_tick,
        "cron",
        hour=REFLECT_HOUR,
        minute=0,
        timezone=None,
        args=[settings],
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
        args=[settings],
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

    if channel is not None and memory is not None:
        app.state.channel = channel
        app.state.memory = memory
        loop = ConversationLoop(
            channel,
            gateway,
            memory,
            data_dir=settings.data_dir,
            recall=recall,
            recall_limit=settings.recall_limit,
            resolve_id=resolve_id,
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

    # The always-on gate. This is what makes "it hears me when I call it" a property
    # of the resident process rather than of a command somebody remembered to run -
    # `daemon wake test` proves the gate works, and this is what uses it.
    wake_round: WakeRound | None = app.state.wake_round
    if wake_round is not None:
        # Injected: a test drives the resident behaviour without a microphone.
        app.state.wake_task = asyncio.create_task(_rounds(wake_round), name="wake-gate")
    elif settings.wake_enabled:
        try:
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
        for name in ("backfill_task", "loop_task", "wake_task"):
            pending = getattr(app.state, name, None)
            if pending is None:
                continue
            pending.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await pending
        if channel is not None:
            with suppress(Exception):
                await channel.close()
        if close_io is not None:
            with suppress(Exception):
                close_io()
        scheduler.shutdown(wait=False)
        closeables: list[Any] = [*providers.values()]
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
            providers[name] = GeminiProvider(settings.gemini_api_key)
        else:
            raise ConfigError(
                f"routing names provider {name!r}, which has no implementation yet "
                f"(M1a ships {OLLAMA} and {ANTHROPIC})"
            )
    return providers


@dataclass(frozen=True, slots=True)
class _IO:
    channel: Channel
    memory: MemoryWriter
    recall: Recall | None
    recall_status: str
    resolve_id: ResolveId | None
    close: Callable[[], None]
    embedder: Any = None
    """Held only so the lifespan can close its HTTP client on shutdown."""


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
    # Repairs a mirror that fell behind its markdown - a failed mirror write, a
    # crash between the two writes, or a deleted database. Without this the
    # markdown being the source of truth is a claim nothing acts on.
    reindex(settings.data_dir, store)
    # Pairing by default: on a first run the env allowlist is empty, and in
    # `allowlist` mode that refuses to start - correct as a policy, useless as an
    # onboarding step. The owner's id is captured from their first message
    # instead of transcribed by hand.
    channel = _build_channel(settings, store)
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
    )


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


async def _reflect_tick(settings: Settings) -> None:
    """The scheduled pass. Catches everything: a job that raises inside
    APScheduler is logged once and then the schedule carries on, which reads as a
    working reflection loop that has silently done nothing for a month."""
    try:
        reflection, close = await build_reflection(settings)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("reflection tick could not start: %s", exc)
        return
    try:
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


async def _persona_tick(settings: Settings) -> None:
    """The weekly persona-evolution pass. Catches everything, same reason as
    reflection and the proactive tick: a job that raises inside APScheduler is
    logged once and then the schedule carries on, which reads as a working
    weekly pass that has silently done nothing for months.

    Logged at INFO even when the pass was skipped, because "not enough
    observations yet" and "already ran this week" both have to be visible
    without opening sqlite - the same reasoning as the reflection tick's log
    line.
    """
    try:
        evolution, close = await build_persona_evolution(settings)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("persona evolution tick could not start: %s", exc)
        return
    try:
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


async def run_voice(settings: Settings) -> int:
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
    from daemon.voice.audio import SoundDeviceAudio
    from daemon.voice.conversation import VoiceConversation
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
        memory = FileMemoryWriter(settings.data_dir, store)
        recall, _status, embedder = _build_recall(settings, store)
        # Seed and learned rules both, same as the text path (daemon/loop.py):
        # a conversation surface is a conversation surface, and M4's learned
        # half is meant to reach every one of them except the proactive judge,
        # which stays seed-only on purpose (daemon/proactivity/judge.py).
        seed = await load_persona(settings.data_dir)
        audio = SoundDeviceAudio()
        session = GeminiLiveSession(
            api_key=settings.gemini_api_key,
            model=route.model,
            # Without a persona the model answers as a generic assistant, which
            # is the one voice PLAN 5 says this product must not have.
            system_instruction=seed or None,
        )
        conversation = VoiceConversation(
            session, audio, memory, recall=recall, recall_limit=settings.recall_limit
        )
        try:
            await conversation.run()
        except GeminiLiveError as exc:
            logger.error("voice session failed: %s", exc)
            return PROBLEM
        finally:
            with suppress(Exception):
                await audio.close()
            if embedder is not None:
                closer = getattr(embedder, "aclose", None)
                if closer is not None:
                    with suppress(Exception):
                        await closer()
        if conversation.ended:
            logger.info("voice session ended: %s", conversation.ended)
        return OK
    finally:
        store.close()


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
    await run_voice(settings)


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
    """The microphone, for one calibration take."""
    from daemon.voice.audio import SoundDeviceAudio

    return SoundDeviceAudio()


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
