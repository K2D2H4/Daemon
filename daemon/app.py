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
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon.channels.base import Channel
from daemon.config import ANTHROPIC, OLLAMA, ConfigError, Settings
from daemon.llm.base import Provider
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop
from daemon.memory.base import MemoryWriter

logger = logging.getLogger(__name__)

DB_FILENAME = "daemon.sqlite3"
"""Lives inside the data dir. Deleting it must never lose user data - the
markdown log is the original (CONTRACTS.md non-negotiable 1)."""


def create_app(
    settings: Settings | None = None,
    *,
    channel: Channel | None = None,
    memory: MemoryWriter | None = None,
) -> FastAPI:
    """Assemble the process. `channel`/`memory` are injection points for tests;
    normally both are built from settings during startup."""
    resolved = settings or Settings()
    app = FastAPI(title="Daemon", version="0.0.1", lifespan=_lifespan)
    app.state.settings = resolved
    app.state.channel = channel
    app.state.memory = memory
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
    close_io: Callable[[], None] | None = None
    if channel is None or memory is None:
        try:
            channel, memory, close_io = _build_io(settings)
        except Exception as exc:
            # Loud, not silent: /health will report the loop as stopped.
            logger.error("conversation loop not started: %s", exc)
            channel = memory = None

    if channel is not None and memory is not None:
        app.state.channel = channel
        app.state.memory = memory
        loop = ConversationLoop(channel, gateway, memory, data_dir=settings.data_dir)
        app.state.loop_task = asyncio.create_task(loop.run(), name="conversation-loop")

    try:
        yield
    finally:
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
        for provider in providers.values():
            closer = getattr(provider, "aclose", None)
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


def _build_io(settings: Settings) -> tuple[Channel, MemoryWriter, Callable[[], None]]:
    """Wire the concrete channel and memory writer, plus their teardown.

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
    return channel, FileMemoryWriter(settings.data_dir, store), store.close


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
