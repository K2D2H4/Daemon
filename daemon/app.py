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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon.channels.base import Channel
from daemon.config import ANTHROPIC, GEMINI, OLLAMA, OPENAI, ConfigError, Settings
from daemon.llm.base import Provider
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop, ResolveId
from daemon.memory.base import MemoryWriter, Recall
from daemon.tasks import Task

logger = logging.getLogger(__name__)

OK = 0
PROBLEM = 1
"""Shell exit codes, matching cli.py - `daemon voice` is a command."""

DB_FILENAME = "daemon.sqlite3"
"""Lives inside the data dir. Deleting it must never lose user data - the
markdown log is the original (CONTRACTS.md non-negotiable 1)."""


def create_app(
    settings: Settings | None = None,
    *,
    channel: Channel | None = None,
    memory: MemoryWriter | None = None,
    recall: Recall | None = None,
) -> FastAPI:
    """Assemble the process. `channel`/`memory`/`recall` are injection points for
    tests; normally all three are built from settings during startup."""
    resolved = settings or Settings()
    app = FastAPI(title="Daemon", version="0.0.1", lifespan=_lifespan)
    app.state.settings = resolved
    app.state.channel = channel
    app.state.memory = memory
    app.state.recall = recall
    app.state.recall_status = "injected" if recall is not None else "not started"
    app.state.loop_task = None

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

    # M1a registers no jobs. The seam exists now so the reflection loop (M2) and
    # the 5-minute proactivity tick (M3) have somewhere to land.
    scheduler = AsyncIOScheduler(timezone="UTC")
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

    try:
        yield
    finally:
        # `suppress(CancelledError)` alone was not enough. A task that had
        # *already* finished with some other exception re-raises it on await, and
        # that escaped the finally block - skipping the channel close, the sqlite
        # close, the scheduler shutdown and every provider aclose below it. A
        # revoked bot token was enough to leak the lot on every restart.
        for name in ("backfill_task", "loop_task"):
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


def _build_io(settings: Settings) -> _IO:
    """Wire the concrete channel, memory writer and recall, plus their teardown.

    TelegramChannel raises on an empty token or an empty allowlist, so those
    checks are deliberately not repeated here.
    """
    from daemon.channels.pairing import Pairing
    from daemon.channels.telegram import TelegramChannel
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
    pairing = (
        Pairing(store, TelegramChannel.name)
        if settings.telegram_dm_policy == "pairing"
        else None
    )
    channel = TelegramChannel(
        settings.telegram_bot_token,
        settings.telegram_allowed_user_ids,
        cursor=store,
        dm_policy=settings.telegram_dm_policy,
        pairing=pairing,
    )
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
        seed = _read_seed(settings.data_dir)
        audio = SoundDeviceAudio()
        session = GeminiLiveSession(
            api_key=settings.gemini_api_key,
            model=route.model,
            # The persona is the seed, same as the text path. Without it the model
            # answers as a generic assistant, which is the one voice PLAN 5 says
            # this product must not have.
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


def _read_seed(data_dir: Path) -> str:
    """The human-owned half of the persona. Never written by us (PLAN 5.1)."""
    path = Path(data_dir) / "persona" / "seed.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


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
