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
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon.channels.base import Channel
from daemon.config import ANTHROPIC, OLLAMA, ConfigError, Settings
from daemon.llm.base import Provider
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop, ResolveId
from daemon.memory.base import MemoryWriter, Recall

logger = logging.getLogger(__name__)

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
            "recall": app.state.recall_status,
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
        app.state.loop_task = asyncio.create_task(loop.run(), name="conversation-loop")

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
        backfill = getattr(app.state, "backfill_task", None)
        if backfill is not None:
            backfill.cancel()
            with suppress(asyncio.CancelledError):
                await backfill
        task = app.state.loop_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
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
    from daemon.llm.providers.ollama import OllamaProvider

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
    channel = TelegramChannel(
        settings.telegram_bot_token,
        settings.telegram_allowed_user_ids,
        cursor=store,
    )
    recall, recall_status, embedder = _build_recall(settings, store)
    return _IO(
        channel=channel,
        memory=FileMemoryWriter(settings.data_dir, store),
        recall=recall,
        recall_status=recall_status,
        resolve_id=_id_resolver(store),
        close=store.close,
        embedder=embedder,
    )


async def _backfill(recall: Recall) -> None:
    """Embed history the vector lane is missing. Never fatal: recall degrades to
    keyword-only, which is worse than the full answer and far better than a dead
    conversation loop."""
    try:
        landed = await recall.backfill()
        if landed:
            logger.info("recall backfill embedded %d message(s)", landed)
    except Exception:
        logger.exception("recall backfill failed; the vector lane stays partial")


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


def _id_resolver(store: Any) -> ResolveId:
    """Read back the id of the message that was just recorded.

    `MemoryWriter.record()` is frozen and returns nothing, and the id lives in the
    mirror, so the loop gets this small closure instead of a widened protocol. The
    text is compared before the id is handed over: `recent(1)` is "newest by
    timestamp", and a channel-supplied timestamp that runs behind our own clock
    would otherwise point at the previous turn's row, silently filing one
    message's vector under another message's id.
    """

    def resolve(text: str) -> int | None:
        rows = store.recent(1)
        if not rows:
            return None
        row = rows[-1]
        return int(row["id"]) if row["content"] == text.strip() else None

    return resolve


def main() -> None:
    """Entry point for the `daemon` console script (pyproject [project.scripts]).

    Delegates to the CLI so `daemon` keeps starting the server while `daemon
    install`, `daemon doctor` and the rest exist alongside it.
    """
    from daemon.cli import main as cli_main

    raise SystemExit(cli_main())
