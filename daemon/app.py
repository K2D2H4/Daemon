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
import json
import logging
import random
import secrets
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from daemon import __version__, mic_floor
from daemon.channels.base import Channel
from daemon.clock import now_iso
from daemon.companion import TOOL_CONTRACT, Companion, ResolveId
from daemon.config import (
    ANTHROPIC,
    ENV_FILE,
    GEMINI,
    OLLAMA,
    OPENAI,
    OPENAI_COMPATIBLE,
    ConfigError,
    Settings,
)
from daemon.face import MOOD_TOOL, MOOD_VOICE_INSTRUCTION
from daemon.face_clips import ClipQueue, wanted
from daemon.llm.base import Provider, ToolSpec
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop
from daemon.memory.base import MemoryWriter, Recall
from daemon.ollama_process import LocalOllama
from daemon.persona.evolve import EvolutionResult, PersonaEvolution
from daemon.proactivity.tick import ProactiveTick
from daemon.reflection import Reflection, Result
from daemon.tasks import Task

if TYPE_CHECKING:  # the wake gate, used only in the signatures below
    from daemon.face import FaceBus
    from daemon.face_lipsync import Cache
    from daemon.face_lipsync.render import Driver, FrameClock, Renderer
    from daemon.face_lipsync.ring import PcmRing, Slot
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

VIDEO_CAPABLE_VOICE_PROVIDERS = frozenset({"gemini"})
"""Voice providers whose session actually accepts video frames via `send_frame`.

`OpenAIRealtimeSession.send_frame` (daemon/voice/openai_realtime.py) is a
deliberate no-op - OpenAI Realtime has no realtime video input channel - so
building a `ScreenShareController` and its start/stop tools for it would let the
model tell the owner "I'm watching your screen now" while every frame is
silently dropped. ADR 0009 (docs/adr/0009-images-in-the-message-contract.md)
is explicit that a capability the code cannot deliver must say so, not attach a
frame and hope. Gated on this set, not on the current session's `route.provider`
string inline, so the question reads as "can this provider take video" rather
than a bare provider-name check."""

PROACTIVE_TICK_MINUTES = 5
"""docs/PLAN.md 6.1's tick. Deterministic work only, unless a candidate is due and
passes the gate - so the cost of the interval is a few sqlite reads and three
subprocess probes, not a model call."""

# A `PROACTIVE_TOOLS_BUILD_TIMEOUT` used to live here, wrapping this module's own
# `_build_tools` call in `asyncio.wait_for` so a wedged MCP server could not stop
# the scheduler (`_proactive_tick` is registered `max_instances=1`, so a tick that
# never returns is every later tick silently skipped). The PR #113 review showed it
# made the failure worse, not better: `_build_tools` constructs the bridge *inside*
# the awaited coroutine, and `McpBridge._bring_up` opens each server in a detached
# task, so cancelling the caller left every already-connected stdio child running
# with nothing holding a reference that could ever close it. The ceiling now sits
# where the task owning the transport can actually be cancelled -
# `_ServerLink.open` in `daemon/tools/mcp.py`, under `STARTUP_TIMEOUT` - which also
# bounds the lifespan's own build and every other `_build_tools` caller, not just
# this one.

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
    local_ollama: LocalOllama | None = None,
) -> FastAPI:
    """Assemble the process. `channel`/`memory`/`recall`/`wake` are injection points
    for tests; normally all four are built from settings during startup.

    `wake` is one round of "listen until called, then hold a conversation" - see
    `_wake_round`. Injected rather than assembled in tests because the real one
    opens a microphone and then a billed session, and neither belongs in a test.

    `local_ollama` is never built here - `None` means no start task, exactly
    today's behaviour, so every existing test that does not pass one keeps not
    touching the network. Only `daemon run` (`daemon/cli.py`) constructs the real
    one; a test that built its own here would probe localhost and could spawn a
    real `ollama serve`.
    """
    resolved = settings or Settings()
    app = FastAPI(title="Daemon", version=__version__, lifespan=_lifespan)
    app.state.settings = resolved
    app.state.channel = channel
    app.state.memory = memory
    app.state.recall = recall
    app.state.local_ollama = local_ollama
    app.state.recall_status = "injected" if recall is not None else "not started"
    app.state.loop_task = None
    app.state.reflection_boot_task = None
    app.state.wake_round = wake
    app.state.wake_task = None
    app.state.wake_status = "off"
    app.state.tools = None
    app.state.tools_status = "not started"
    app.state.voice_runtime = None
    app.state.wake_gate = None
    app.state.mcp = None
    # The lip-sync renderer and the sink that feeds it, both absent until the
    # lifespan builds them and both absent for good on an install that never turns
    # the switch on. Declared here rather than left to `getattr` so that the two
    # readers - `face_routes._lipsync` and `_wake_round` - see the same "off" whether
    # or not a lifespan ever ran (tests build the app without one).
    app.state.face_frames = None
    app.state.face_pcm_sink = None
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
    # One bus for the process. Publishing to it with nobody subscribed costs a
    # comparison, so a text-only install that never opens /face pays nothing.
    from daemon.face import FaceBus

    app.state.face = FaceBus()

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

    # The face's own endpoints: a page, an SSE stream off the bus above, and clip
    # bytes. Read-only and side-effect free (daemon/face_routes.py), so mounting it
    # unconditionally costs nothing on an install that never opens it.
    from daemon.face_routes import router as face_router

    app.include_router(face_router)

    return app


def health_payload(state: Any, settings: Settings) -> dict[str, Any]:
    """The `/health` body, built from `app.state`.

    A module-level function rather than a closure so the admin's own health
    endpoint returns byte-for-byte the same thing - "the chat version of /health"
    is only honest if the health it shows is the same health (docs/design)."""
    task = getattr(state, "loop_task", None)
    return {
        "status": "ok",
        "provider": settings.provider,
        "proactive_judge_local": settings.proactive_judge_local,
        "voice_enabled": settings.voice_enabled,
        # When this process came up, so a reader can tell "quiet for a week" from
        # "restarted a minute ago and has not had time to do anything yet". Absent
        # rather than faked when the lifespan never ran (tests build the app
        # directly), because a zero uptime would be a lie the admin would print.
        "started_at": getattr(state, "started_at", None),
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
    app.state.started_at = now_iso()
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
    # Exposed so the admin's "run now" buttons take the very same lock. Without
    # this they would be a third writer beside the cron and the boot task, and the
    # comment above says what two `run(date)` for one day costs.
    app.state.catchup_lock = catchup_lock
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
        #
        # `_get_lifespan_bridge` is a closure, not `app.state.mcp` itself:
        # `scheduler.add_job`'s `args` are captured now, before the rest of this
        # function has necessarily built `app.state.mcp` (it is set further down,
        # only once `io` succeeds) - so the job needs something that reads the
        # attribute fresh on every fire, not its value at registration time. Every
        # fire after that reuses whatever live bridge the lifespan is currently
        # holding instead of `_proactive_tick` connecting and tearing down every
        # configured MCP server itself, every five minutes (whole-branch review;
        # see `build_proactive_tick`'s `bridge` parameter).
        def _get_lifespan_bridge() -> Any:
            return getattr(app.state, "mcp", None)

        scheduler.add_job(
            _proactive_tick,
            "interval",
            minutes=PROACTIVE_TICK_MINUTES,
            args=[settings, _get_lifespan_bridge],
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
    local_ollama: LocalOllama | None = app.state.local_ollama
    resolve_id: ResolveId | None = None
    close_io: Callable[[], None] | None = None
    embedder: Any = None
    tools: Any = None
    store: Any = None
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
            store = io.store
            app.state.recall = recall
            app.state.recall_status = io.recall_status
            # Created before `_build_voice_runtime` regardless of whether that call
            # happens below: the delegation worker (assembled further down, once
            # `channel`/`memory` are confirmed) waits on this same event, and it
            # must exist even when voice/wake is off so that assembly does not
            # depend on this `if`.
            app.state.delegate_wake = asyncio.Event()
            tools, app.state.mcp, app.state.tools_status = await _build_tools(
                settings, io.store, face=app.state.face
            )
            app.state.tools = tools
            if settings.voice_enabled and settings.wake_enabled:
                # The wake path's boot-once services (VoiceRuntime): without this,
                # every wake word rebuilt the tool layer and reconnected every MCP
                # server before the daemon could speak - ~4s of the owner's "why
                # does it take six seconds to answer", paid per call.
                try:
                    app.state.voice_runtime = await _build_voice_runtime(
                        settings,
                        io.store,
                        memory,
                        recall,
                        delegate_wake=app.state.delegate_wake,
                        channel=channel,
                        face=app.state.face,
                    )
                except Exception as exc:
                    # The wake round falls back to building its own per call -
                    # slower, never deaf.
                    logger.error(
                        "voice runtime not prebuilt (wake rounds build their own): %s", exc
                    )

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
            face=app.state.face,
        )
        task = asyncio.create_task(loop.run(), name="conversation-loop")
        # run() only guards individual turns; anything raised by the channel's own
        # listen() - a revoked bot token surfaces as TelegramFatal - ends the task.
        # Without this the process stays alive and healthy-looking with no inbound
        # path and not one line in the log, because app.state holds the reference
        # so even asyncio's "never retrieved" warning never fires.
        task.add_done_callback(_report_loop_death)
        app.state.loop_task = task

        if store is not None:
            # The other half of `delegate_task` (daemon/tools/delegate.py): a voice
            # turn only queues a row, and this is what actually runs it - through
            # the same text ConversationLoop the Telegram path uses, so a
            # nested-schema tool the voice model could not call is called here
            # where it can be - and reports the result back by presence. `store`
            # is only set when `_build_io` ran above; the fake channel/memory a
            # couple of tests inject carry no store, and delegation is simply not
            # wired for them - the same degrade-not-crash shape as `tools`.
            from daemon.delegation import DelegationWorker, build_run_request, deliver_result
            from daemon.proactivity.presence import MachinePresence
            from daemon.proactivity.speaker import LocalSpeaker

            def _delegate_companion_factory() -> Companion:
                return Companion(
                    memory,
                    data_dir=settings.data_dir,
                    recall=recall,
                    recall_limit=settings.recall_limit,
                    resolve_id=resolve_id,
                    tools=tools,
                )

            run_request = build_run_request(
                gateway=gateway, companion_factory=_delegate_companion_factory
            )
            presence = MachinePresence()
            speaker = LocalSpeaker()

            async def deliver(text: str, task_row: Any) -> None:
                await deliver_result(
                    text,
                    presence=presence,
                    speaker=speaker,
                    channel=channel,
                    recipient_id=None,
                )

            worker = DelegationWorker(
                store, run_request, deliver, wake=app.state.delegate_wake
            )

            # Boot recovery: a restart mid-task leaves a queued row nobody will ever
            # finish - the worker only claims what it can run, and a task claimed by
            # a process that is gone stays "running" forever otherwise. Reported,
            # not resumed: resuming would re-run a request whose side effects (a
            # created Notion page) may have already landed.
            for row in store.pending_tasks():
                await deliver(
                    f"아까 부탁한 '{row['request'][:40]}' 작업을 다 못 끝내고 재시작됐어. "
                    "다시 시켜줘.",
                    row,
                )
                store.mark_task_failed(row["id"], "interrupted by restart")

            app.state.delegation_task = asyncio.create_task(
                worker.run(), name="delegation-worker"
            )

    app.state.ollama_task = None
    if local_ollama is not None:
        # Not gated on `recall`: `provider=ollama` routes chat here too, and
        # `recall` is None whenever `_build_recall` failed - exactly the case where
        # hanging the start off the backfill would leave the daemon with no local
        # model at all. Not awaited, for the same log-clock reason as the backfill
        # below. Only `daemon run` passes one; a test passing none gets no start
        # task, because a test that spawns a server is a broken test.
        app.state.ollama_task = asyncio.create_task(
            local_ollama.ensure_running(), name="ollama-start"
        )
        # Defense in depth alongside `_probe`'s broad `except`: `ensure_running`
        # promises never to raise, but if that promise is ever broken again, this
        # is what stands between the exception and vanishing the same way a dead
        # conversation loop used to - `app.state` holding the reference is what
        # keeps asyncio's own "never retrieved" warning from firing either.
        app.state.ollama_task.add_done_callback(_report_ollama_start_death)

    if recall is not None:
        # Backfill after the loop is already serving, and in the background: a
        # rebuilt sqlite file gives every message a new id and drops `embeddings`
        # by cascade, so without this the vector lane stays empty for all history
        # while /health still says recall is ready. Measured on the golden set,
        # that silent state is a 50% ceiling for Korean rather than the hybrid
        # number - a regression where nothing fails. Not awaited, because a cold
        # embedder must not delay the log clock (docs/PLAN.md 8.1) - and that is
        # also why `_backfill` waits for Ollama inside itself rather than here.
        app.state.backfill_task = asyncio.create_task(
            _backfill(recall, app.state.ollama_task), name="recall-backfill"
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

    # The face's second mouth, if it is switched on and assembled. Built before the
    # wake gate below because that is what opens a spoken conversation, and the sink
    # has to be on `app.state` before the first one does. Synchronous on purpose: it
    # realises 1.62GB of weights (measured 693ms), and paying that at startup is
    # honest where paying it on the first spoken word would land inside the reply the
    # owner is waiting for. Independent of the channel and the writer - the face is
    # served whether or not Telegram built.
    lipsync = _build_lipsync(settings, app.state.face)
    if lipsync is not None:
        app.state.face_frames = lipsync.frames
        app.state.face_pcm_sink = lipsync.sink
        app.state.lipsync_task = asyncio.create_task(lipsync.run(), name="face-lipsync")

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
                _wake_forever(
                    settings,
                    shared=getattr(app.state, "voice_runtime", None),
                    state=app.state,
                ),
                name="wake-gate",
            )

    try:
        yield
    finally:
        # `suppress(CancelledError)` alone was not enough. A task that had
        # *already* finished with some other exception re-raises it on await, and
        # that escaped the finally block - skipping the channel close, the sqlite
        # close, the scheduler shutdown and every provider aclose below it. A
        # revoked bot token was enough to leak the lot on every restart.
        for name in (
            "backfill_task",
            "ollama_task",
            "loop_task",
            "wake_task",
            "reflection_boot_task",
            "delegation_task",
            "lipsync_task",
        ):
            pending = getattr(app.state, name, None)
            if pending is None:
                continue
            pending.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await pending
        if channel is not None:
            with suppress(Exception):
                await channel.close()
        local = getattr(app.state, "local_ollama", None)
        if local is not None:
            # A child process, like the stdio MCP servers below: an Ollama this
            # daemon started and did not stop is one more orphan per restart. One
            # it did *not* start is somebody else's and stays running.
            with suppress(Exception):
                await local.aclose()
        mcp = getattr(app.state, "mcp", None)
        if mcp is not None:
            # Before the sqlite close and the scheduler shutdown, because these are
            # child processes: a stdio MCP server left running is one more orphan
            # per restart.
            with suppress(Exception):
                await mcp.aclose()
        voice_runtime = getattr(app.state, "voice_runtime", None)
        if voice_runtime is not None and voice_runtime.mcp is not None:
            # The voice half's bridge, for the same orphan reason as the text one.
            with suppress(Exception):
                await voice_runtime.mcp.aclose()
        if close_io is not None:
            with suppress(Exception):
                close_io()
        scheduler.shutdown(wait=False)
        closeables: list[Any] = [*providers.values()]
        if tools is not None:
            # `fetch_page` owns an HTTP client; the runner walks the registry for it.
            closeables.append(tools)
        if voice_runtime is not None and voice_runtime.tools is not None:
            closeables.append(voice_runtime.tools)
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
    from daemon.llm.providers.openai_compatible import OpenAICompatibleProvider

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
        elif name == OPENAI_COMPATIBLE:
            providers[name] = OpenAICompatibleProvider(
                settings.openai_compatible_api_key,
                settings.openai_compatible_base_url,
            )
        else:
            raise ConfigError(
                f"routing names provider {name!r}, which has no implementation yet "
                f"(this build ships {OLLAMA}, {ANTHROPIC}, {OPENAI}, {GEMINI} and "
                f"{OPENAI_COMPATIBLE})"
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


def _report_ollama_start_death(task: asyncio.Task[bool]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("ollama start task died; recall stays keyword-only", exc_info=exc)


async def _backfill(recall: Recall, ollama_ready: Awaitable[bool] | None = None) -> None:
    """Embed history the vector lane is missing, to exhaustion.

    One call was not enough. It stopped at its default 500 rows and never ran
    again - no retry, no periodic job - so a rebuilt sqlite file over a year of
    history left the great majority of messages with no vector while /health
    still reported recall ready. That is the same invisible Korean ceiling the
    protocol change was meant to prevent, just further along.

    The wait for Ollama lives here rather than in `lifespan` on purpose: a cold
    embedder must not delay the log clock (docs/PLAN.md 8.1), and awaiting
    readiness before the `yield` would block uvicorn's "startup complete" and
    /health for as long as a cold start takes. Measured 2026-08-26, skipping the
    wait entirely is what logged `backfill stopped after 0 message(s)` and left 49
    messages unembedded until an unrelated restart.

    Never fatal: recall degrades to keyword-only, which is worse than the full
    answer and far better than a dead conversation loop.
    """
    total = 0
    try:
        if ollama_ready is not None and not await ollama_ready:
            logger.info(
                "recall backfill skipped: no embedder answered. Recall stays keyword-only "
                "and the next restart tries again"
            )
            return
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


@contextmanager
def open_store(settings: Settings) -> Iterator[Any]:
    """A short-lived `Store` handle, closed on the way out.

    Exists so the admin's read-only endpoints can query the mirror without
    importing an implementation themselves - this module is the only one allowed
    to (docs/CONTRACTS.md, the layering rule), and every other assembly point here
    opens the store the same way.

    One connection per request rather than a shared one: sqlite3 connections are
    not safe to share across threads, and the admin's callers are one browser
    polling every fifteen seconds. `Store.open` is not free - it replays the whole
    of `schema.sql` and hardens three files - so this is a real cost, just a small
    one against that traffic: measured end to end, the endpoints that use it answer
    in 1.3-2.6ms against a 753-message database, under what `/health` takes. Reach
    for a cached handle if something noisier than the admin ever wants one.
    """
    from daemon.memory.store import Store

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        yield store
    finally:
        store.close()


async def build_proactive_tick(
    settings: Settings,
    *,
    speak: bool = False,
    bridge: Any = None,
    wake_loop: bool = False,
) -> tuple[ProactiveTick, Callable[[], Awaitable[None]]]:
    """A tick and the coroutine that releases what it holds.

    `speak=False` assembles only the deterministic half: no gateway, no channel, no
    speaker, and therefore no possibility of an utterance. That is not a debugging
    convenience - PLAN 6.4 asks for the gate to be trustworthy *before* anything is
    wired to a speaker, and a mode where speaking is structurally impossible is a
    stronger statement of that than a flag checked at the end.

    The speaker is built only when the user asked for it *and* the platform can do
    it. Everything else degrades to Telegram, which is the safe direction.

    `bridge`, when given, is a live MCP bridge this call does **not** own - the
    resident's `app.state.mcp`, connected once by `_lifespan` and closed only at
    shutdown. `_proactive_tick` passes it on every scheduled fire so a tick reuses
    the connections already up instead of connecting and tearing every configured
    server down 288 times a day (whole-branch review: `_build_tools` unconditionally
    called `bridge.start(registry)`, a stdio child process per server, then
    `bridge.aclose()` at the end of the very same tick). `None` - the default, and
    what `daemon proactive` passes, since the CLI runs with no lifespan and no
    `app.state` to reuse at all - falls back to building, and owning, and later
    closing, a bridge of its own. A wedged MCP server costs this tick rather than
    the scheduler, but the ceiling is not here: it is in `_ServerLink.open`
    (`daemon/tools/mcp.py`), per server, where the task that owns the transport
    can be cancelled. So this call is bounded by roughly the server count times
    `STARTUP_TIMEOUT` rather than by any one number - on a multi-server install it
    can outrun `PROACTIVE_TICK_MINUTES`, which `coalesce=True` absorbs by skipping
    a round. What `max_instances=1` needs is that the job always *returns*, and
    that is what moving the timeout down a level bought.
    """
    from daemon.fs import harden_existing
    from daemon.memory.store import Store
    from daemon.proactivity.presence import MachinePresence

    harden_existing(settings.data_dir)
    store = Store.open(settings.data_dir / DB_FILENAME)
    closers: list[Callable[[], Awaitable[None]]] = []

    # Built regardless of `speak`: `daemon proactive` (no speaking, structurally)
    # is how a human sees whether type E produces anything worth saying before it
    # is ever allowed to say it.
    recall, _status, embedder = _build_recall(settings, store)
    if embedder is not None:
        closer = getattr(embedder, "aclose", None)
        if closer is not None:
            closers.append(closer)

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

        # The `topic` candidate's one search (ADR 0015) goes through this bridge,
        # never through `tools/policy.py:decide` - it calls `MCPBridge.call`
        # directly, so `tools_mode`'s own handling of `off` (ToolPolicy refusing
        # every guarded tool) never sees this call and cannot stop it by itself.
        # Rule 10 forbids the *model* choosing and running a tool on a non-owner
        # turn; a fixed, code-issued, read-only search is the thing ADR 0015 says
        # is not that - so `tools_enabled=false` would already be a defensible
        # place to stop.
        #
        # This wiring goes one step further and also withholds the bridge when
        # `tools_mode == "off"`, even though nothing in ADR 0015 requires it.
        # Reasoning: `off` is the setting an owner reaches for to mean "nothing
        # this daemon does reaches outside this conversation without me asking",
        # and proactivity is the one channel where that promise matters most - an
        # utterance that was never asked for, arriving in Telegram or out of the
        # laptop speaker in a voice the owner trusts. Honouring the letter of the
        # ADR (only `tools_enabled` gates it) while ignoring the mode the owner
        # actually set because this path is technically outside the policy it
        # governs is exactly the kind of capability-nobody-was-asked-about state
        # CONTRACTS rule 12 exists to prevent. `full`, `ask` and `allowlist` all
        # leave tool use switched on in spirit, so those three still get the
        # bridge; only `off` (and the master switch) withhold it.
        #
        # With no bridge, `Judge` drops every `topic` candidate and the other
        # four generators are unaffected - the same degrade path a missing or
        # unconfigured MCP server already takes.
        tick_bridge = None
        if settings.tools_enabled and settings.tools_mode != "off":
            if bridge is not None:
                # Reused, owned by the caller - never added to `closers`. Closing
                # a bridge this tick did not build would tear down every MCP
                # server the rest of the running app depends on the moment this
                # one tick ends, orphaning `app.state.mcp` for everything else
                # that reaches for it until the next full restart.
                tick_bridge = bridge
            else:
                # No `wait_for` around this: a server that hangs on connect is
                # bounded inside `_ServerLink.open`, in the task that owns the
                # transport and can therefore be cancelled cleanly. Wrapping the
                # call here instead orphaned the children it had already started
                # (see the note on the removed `PROACTIVE_TOOLS_BUILD_TIMEOUT`).
                tools_runner, tick_bridge, _tools_status = await _build_tools(
                    settings, store
                )
                if tools_runner is not None:
                    closers.append(tools_runner.aclose)
                if tick_bridge is not None:
                    # A stdio MCP server is a child process - one left running is
                    # an orphan per tick, the same reason the app lifespan closes
                    # its own bridge ahead of the store (see `_lifespan` above).
                    # Only reached on this, the "this tick built its own" branch -
                    # a reused bridge is never closed here (see above).
                    closers.append(tick_bridge.aclose)
        judge = Judge(gateway, data_dir=settings.data_dir, bridge=tick_bridge)

        channel = None
        try:
            channel = _build_channel(settings, store)
        except Exception as exc:  # noqa: BLE001 - a missing token must not stop the tick
            # Loud, and then Telegram simply is not a route. The gate still runs and
            # the local speaker may still work, which is more than nothing.
            logger.error("proactive: no channel, so nothing can be delivered there: %s", exc)

        speaker = None
        if settings.voice_enabled:
            from daemon.proactivity.speaker import LocalSpeaker

            speaker = LocalSpeaker()
            closers.append(speaker.aclose)

        # The floor, only where a wake round could answer it. `wake_loop` is passed
        # by `_proactive_tick` and by nothing else, because it is a fact about *this
        # process* that no setting can stand in for: `daemon proactive --speak` sets
        # `speak=True` too (`daemon/cli.py`) and has no wake loop at all, and
        # `settings.wake_enabled` stays true on a resident whose wake task was never
        # created - no on-device recognizer, a microphone grant this build cannot
        # use. An earlier version of this comment claimed `speak` told those apart.
        # It does not (PR #115 review), and the cost of believing it was a ten-second
        # stall on every line from a command a person is watching run.
        ask_for_the_floor = None
        if wake_loop and speak and settings.voice_enabled and settings.wake_enabled:
            ask_for_the_floor = mic_floor.request

        delivery = ProactiveDelivery(
            store,
            FileMemoryWriter(settings.data_dir, store),
            ask_for_the_floor=ask_for_the_floor,
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
            store, settings, MachinePresence(), judge=judge, delivery=delivery, recall=recall
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


# --- the face's second mouth ------------------------------------------------
#
# Everything from here to `_build_lipsync` is assembly, which is why it is in this
# file and not in `daemon/face_lipsync/` (CONTRACTS 4: that package imports nothing
# from `daemon/` and nothing outside this module imports it). What it assembles is
# described in docs/superpowers/specs/2026-08-26-face-lipsync-design.md.

LIPSYNC_FIRST_CLIP = "idle2"
"""Which prepared clip the face starts on, before the first boundary moves it.

`idle2` when it is prepared, else whichever sorts first. It is the best case for a
driving clip among the idles - movement 4.77 against `idle1`'s 0.46, which is why it was
the one clip the single-clip build drove - and starting there means the first thing a
page sees is the clip every earlier judgement was made against. Nothing else depends on
it: `face_clips.wanted` moves the face at the first clip boundary if the activity says
otherwise."""

LIPSYNC_ARTEFACTS = ("frames.npy", "masks.npz", "boxes.json", "latents.safetensors")
"""What `evals/face_lipsync_prepare.py` writes, and all four are required.

A directory under `<data_dir>/face/lipsync/` is a driveable clip when it has all four
AND `<data_dir>/face/<name>.mp4` exists. Both halves matter. `models/` and a spike's
output directory have none of them; `idle2-raw-backup/` has three - it is the box
backup, kept so a re-smooth can be undone without the prep venv - and would otherwise
become a clip named after a backup. The mp4 requirement keeps the driveable set equal to
the servable set: `face_routes.py` serves `/face/clips/{name}` off that file, and it is
what the page falls back to when the renderer latches `failed`.

Ten clips are prepared today. There is no constant naming them here on purpose: which
caches exist is the owner's data, and rule 4 is omission - a clip that was never
prepared is absent, never interpolated or substituted."""

LIPSYNC_WHISPER_REPO = "mlx-community/whisper-tiny-mlx"
"""The audio encoder, by HuggingFace repo id rather than a path.

`mlx_whisper` resolves and caches it itself (~150MB), so unlike the UNet and TAESD
this one is not something the owner lays out by hand - which is also why a missing
cache here shows up as a download on first boot rather than as the "no weights"
degradation below."""

LIPSYNC_CATCHUP_LIMIT = 64
"""How many frame indices one pass may discard to stay level with the audio.

`FrameClock.due` hands back one frame per call and deliberately never skips - it says
so, and its reason is that jumping forward after a hitch is worse than a moment of
catching up. That is right about a hitch and wrong about a **structural** deficit, and
the first build had one: with the composite and the JPEGs in sequence behind the model
step, a two-frame step cost 86.6ms against the 83.3ms two frames are worth, so the
loop sustained 20.4fps where the clock granted 24 and consuming one grant per tick
drifted the mouth 34 frames - 1.4 seconds - over 9 seconds of speech. Not a mouth
slightly behind: a different sentence.

That deficit is gone (`render.py:Renderer` splits the model half from the CPU half and
`_lipsync_loop` runs them on two threads), so this now absorbs a *hitch* rather than a
structural shortfall and is normally a no-op - the newest grant is the next one in
sequence. What it still buys is the thing a hitch would otherwise cost permanently:
the mouth stays at the audio front, and the rendered driving-frame index keeps
tracking wall-clock elapsed the way the page's own `<video>` playhead does - measured
at the socket, the overlay now sits a median **0.14 frames** behind that playhead
where it used to sit 5.81 behind it (`render.py:DISPLAY_LEAD` closed the rest).

The limit is a bound on one pass's work, not a policy: `due` returns None as soon as it
reaches audio that has not arrived, so the drain ends on its own. 64 is two and a half
seconds of frames - enough to absorb a long stall, small enough that a wild `origin`
cannot spin here."""

LIPSYNC_RING_SECONDS = 60.0
"""How much spoken audio the PCM ring holds, and it is sized by *utterance* length
rather than by the 2.2s the model reads.

`PcmRing.window` needs `audio.CONTEXT_MS` (2.0s) of lead-in plus the 200ms window, and
a ring shorter than that truncates the context silently - the features degrade and
nothing raises. That is the floor. This is 27x the floor because of a second
interaction that binds much harder: once the ring is full it drops samples on every
feed, `PcmRing.origin` creeps forward as it does, and `FrameClock.due` treats 0.2s of
cumulative movement as a new turn and restarts its frame count at 0. Mid-utterance
that would point the frame index at the oldest audio still held and leave the mouth a
whole ring-length behind the sound for the rest of the turn. So the ring has to
outlast a single spoken reply, not a single window.

The cost is a `np.concatenate` of the whole ring per chunk: 60s of 24kHz mono int16 is
2.9MB, which is ~0.3ms of memcpy against audio chunks that arrive at most a few dozen
times a second. Cheap enough that buying a wide margin here is better than tuning it.

Flagged rather than fixed: the frame counter restarting on a ring that is merely full
is a defect in `FrameClock`, which this task may not edit (see the report)."""


@dataclass(frozen=True, slots=True)
class Lipsync:
    """The assembled lip-sync path: what the transport reads, what the voice path
    feeds, and the loop that turns one into the other."""

    frames: Any
    """Satisfies `face_routes.LipsyncFrames`. Goes on `app.state.face_frames`."""

    sink: Callable[[bytes, float], None]
    """`(chunk, audible_at)`, handed to `VoiceConversation(pcm_sink=...)` and called
    from `SpeechClock.fed`. Thread-safe by being a `deque.append` and nothing else -
    see `_build_lipsync`."""

    run: Callable[[], Awaitable[None]]
    """The render loop, as a coroutine the lifespan turns into one task."""


@dataclass
class _Driving:
    """What the two halves of the render loop agree is on screen right now.

    Mutable and shared on purpose. `publish` reads it every frame interval to index the
    clip and to find that clip's encoded stills; the model half writes it when
    `ClipQueue` says a boundary has been reached. Both run on the event loop, so the
    swap is never seen half-done - and it is written in ONE place (the model half),
    because `Renderer.switch` mutates renderer state that the executor threads read.
    """

    driver: Driver
    frames: list[bytes]
    """This clip's own stills, encoded once on first use - see `_encode_for`."""


class _LipsyncFrames:
    """`face_routes.LipsyncFrames` over a `Renderer`, its `Slot` and its `ClipClock`.

    Four members and no more, which is the protocol's whole point. It exists because
    the three objects that hold those members are deliberately separate: `Slot` is
    `put`/`get` with no idea what wrote it, `Renderer` knows whether it has given up,
    and `ClipClock` knows only where the driving clip is. Joining them is assembly, so
    it happens here.
    """

    def __init__(self, *, renderer: Renderer, slot: Slot, driving: _Driving) -> None:
        self._renderer = renderer
        self._slot = slot
        self._driving = driving
        """The live holder, not a copy of its name: the clip changes while this object
        is on `app.state.face_frames`, and a page told the wrong one falls back to a
        `<video>` of a clip the frames are not of."""
        # No box. Every JPEG is the whole composited frame now, so there is no
        # rectangle for the page to place - and reading `renderer.frame_box` here after
        # it was removed would have been an AttributeError on the first real build,
        # which no test caught because they all inject a fake renderer.

    @property
    def failed(self) -> bool:
        """Read through rather than copied: the renderer latches this mid-utterance
        and the transport has to see it on the next poll, not at the next restart."""
        return self._renderer.failed

    def get(self) -> bytes | None:
        return self._slot.get()

    def position(self) -> float:
        """Where the driving clip is now, in seconds, for the page to seek to.

        Read at the moment the manifest is built rather than cached, and the route is
        `async`, so `loop.time()` here is the same clock the renderer stepped against
        a millisecond ago. Over loopback the page stamps its own arrival ~2ms later,
        which is a twentieth of a frame.
        """
        return self._driving.driver.clip.position(asyncio.get_running_loop().time())

    @property
    def clip(self) -> str:
        """Which clip the frames are currently of. Read through, never cached."""
        return self._driving.driver.name


def _prepared_clips(root: Path, face_dir: Path) -> list[str]:
    """Which clips under `root` are driveable, sorted so a log line is stable.

    Both conditions, for the reasons `LIPSYNC_ARTEFACTS` gives. Nothing is loaded here -
    this only decides what is worth loading, so a directory that fails is never opened.
    """
    found = []
    for entry in sorted(root.iterdir() if root.is_dir() else []):
        if not entry.is_dir():
            continue
        if not all((entry / name).is_file() for name in LIPSYNC_ARTEFACTS):
            continue
        if not (face_dir / f"{entry.name}.mp4").is_file():
            continue
        found.append(entry.name)
    return found


def _load_lipsync_cache(directory: Path) -> tuple[Cache, float]:
    """One prepared driving clip off disk, plus the fps it was shot at.

    Written by `evals/face_lipsync_prepare.py`, by hand, once per clip. Two things
    here are its constraints rather than choices:

    `frames.npy` is memory-mapped. It is 1.01GB for `idle2`'s 193 frames of
    1080x1620, and the renderer touches one frame per model step - paging it in on
    demand is the difference between 1GB of resident memory and a few MB.

    `masks.npz` is a pile of differently-shaped arrays, not a stack, because
    MuseTalk derives each frame's blend region from that frame's own face box (608
    to 726px square across this clip). `Cache.masks` is annotated `np.ndarray` and
    gets a list; `composite` indexes it per frame and never treats it as one array.
    That annotation is in a file this task may not edit, so it is named here rather
    than corrected.
    """
    import numpy as np

    from daemon.face_lipsync import Cache

    meta = json.loads((directory / "boxes.json").read_text(encoding="utf-8"))
    frames = np.load(directory / "frames.npy", mmap_mode="r")
    # Read it once, here, so the render path never waits on a disk. Measured: the
    # per-step CPU tail (two composites, two JPEG encodes, and two 5.2MB frame reads)
    # is 11ms on frames the page cache already holds and 16-32ms on frames it does
    # not, and that variance lands straight on a 41.67ms budget. Worse, it compounds -
    # a slower step drops more frames, which scatters the next reads further apart,
    # which slows the step again: a 20fps mouth measured 12fps once the access pattern
    # stopped being sequential. One 1GB sequential read at startup costs a few hundred
    # milliseconds and takes the disk out of the loop for good.
    #
    # `sum` because it is one pass nothing can optimise away, `uint64` so numpy does
    # not promote per element. The array stays memory-mapped: this warms the page
    # cache, it does not copy a gigabyte onto the heap.
    frames.sum(dtype=np.uint64)
    with np.load(directory / "masks.npz") as archive:
        # Sorted, not `archive.files` order: the keys are the frame index zero-padded
        # to six digits, and a mask off by one frame is a paste off by one face.
        masks = [archive[name] for name in sorted(archive.files)]
    cache = Cache(
        frames=frames,
        boxes=[tuple(box) for box in meta["boxes"]],
        crop_boxes=[tuple(box) for box in meta["crop_boxes"]],
        masks=masks,
    )
    return cache, float(meta["fps"])


def _queue_pcm(queued: deque[tuple[bytes, float]], chunk: bytes, audible_at: float) -> None:
    """`SpeechClock`'s `pcm_sink`: one chunk and the moment it becomes audible.

    A named two-argument function rather than the `deque.append` this started as -
    `append` takes exactly one argument, so the sink raised `TypeError` on the very
    first chunk of the very first spoken turn and the mouth never moved. Nothing in
    the unit suite could see it: the sink was only ever compared by identity, and the
    render loop's tests put tuples in the queue themselves. `partial(_queue_pcm, q)`
    is callable the way the clock calls it, and `tests/test_face_lipsync_wiring.py`
    now drives a real `SpeechClock` into it.
    """
    queued.append((chunk, audible_at))


def _build_lipsync(settings: Settings, face: FaceBus) -> Lipsync | None:
    """Assemble the lip-sync path, or `None` and one log line saying why not.

    Three ways this returns `None` and all three are ordinary installs, not faults:
    the switch is off (the default), the weights are not laid out, or no clip cache
    has been prepared. Each leaves the face playing the pre-rendered clips of v1 -
    which are not a fallback, they are the other half of the face - and
    `face_routes._lipsync_unavailable` is what tells a human which one it was.

    A *failure* is different from an absence and is handled elsewhere: the renderer
    latches `failed`, logs once, and the loop below stops. Never a traceback per
    frame at 24Hz.
    """
    if not settings.face_lipsync_enabled:
        return None
    root = Path(settings.data_dir) / "face" / "lipsync"
    models = root / "models"
    needed = {
        "the MLX UNet": models / "unet.safetensors",
        "its config": models / "musetalk.json",
        "the TAESD decoder": models / "taesd.safetensors",
    }
    missing = {what: path for what, path in needed.items() if not path.exists()}
    if missing:
        # One line, naming the first thing that is absent and the directory it belongs
        # in. Not a warning per file: an install that has fetched none of this would
        # print three, and the owner's question is "where do I put them", once.
        what, path = next(iter(missing.items()))
        logger.info(
            "face: lip-sync is on but not assembled - %s is missing (%s). The face "
            "will play its pre-rendered clips. Weights go under %s; a driving clip's "
            "cache is built by `python3 -m evals.face_lipsync_prepare`.",
            what,
            path,
            models,
        )
        return None
    # `root.parent` rather than importing `face_routes.face_dir`: it is the same
    # `<data_dir>/face`, and this module already computed it one line above.
    prepared = _prepared_clips(root, root.parent)
    if not prepared:
        # Separate from the weights above because the answer is a different command.
        # Named as its own absence rather than folded into the seven-file list this
        # used to print: with the weights in place and no cache, the owner has one
        # thing left to run and it is not a download.
        logger.info(
            "face: lip-sync is on and the weights are in place, but no clip under %s "
            "has a prepared cache with a matching mp4. The face will play its "
            "pre-rendered clips. A cache is built by `python3 -m "
            "evals.face_lipsync_prepare <clip>.mp4 --out %s/<clip>`.",
            root,
            root,
        )
        return None

    started = time.perf_counter()
    try:
        # Inside the try, and every one of them: `mlx` has no Linux wheel and `cv2`
        # comes from an optional extra, so an install with the switch on and the files
        # in place but the `face` extra missing has to lose the mouth - not the
        # process. Above the try these were an ImportError out of the lifespan, which
        # is a daemon that will not start because of a face.
        from daemon.face_lipsync.audio import CONTEXT_MS
        from daemon.face_lipsync.engine import load as load_engine
        from daemon.face_lipsync.render import (
            ClipClock,
            Driver,
            FrameClock,
            Renderer,
            encode_clip,
        )
        from daemon.face_lipsync.ring import PcmRing, Slot
        from daemon.voice.audio import OUTPUT_SAMPLE_RATE

        caches = {name: _load_lipsync_cache(root / name) for name in prepared}
        rates = {fps for _, fps in caches.values()}
        assert len(rates) == 1, (
            f"the prepared clips were shot at different rates ({sorted(rates)}), and "
            "one FrameClock paces them all - reprepare them at one fps"
        )
        fps = rates.pop()
        engine = load_engine(
            unet_weights=models / "unet.safetensors",
            unet_config_json=models / "musetalk.json",
            taesd_weights=models / "taesd.safetensors",
            whisper_repo=LIPSYNC_WHISPER_REPO,
            # One engine, one latent table per clip. The UNet, TAESD and whisper
            # weights are 1.6GB and say nothing about which clip they are drawing; the
            # latents are 1.25-3.02 MiB measured over these ten, so holding them all
            # is ~25MB and a switch is a dict lookup rather than a reload.
            latents={
                name: root / name / "latents.safetensors" for name in prepared
            },
            # The rate the ring holds and therefore the rate the engine has to
            # resample from - the voice path's playback rate, not whisper's 16kHz.
            # `resample_to_whisper` refuses anything else rather than stretching the
            # mouth by 1.5x in silence.
            sample_rate=OUTPUT_SAMPLE_RATE,
        )
        assert LIPSYNC_RING_SECONDS * 1000.0 >= CONTEXT_MS + 200.0, (
            "the ring is shorter than the window plus its lead-in, which truncates "
            "the context silently - see LIPSYNC_RING_SECONDS"
        )
        ring = PcmRing(
            sample_rate=OUTPUT_SAMPLE_RATE, width=2, seconds=LIPSYNC_RING_SECONDS
        )
        slot = Slot()
        # The page's playhead, as a function of `loop.time()`. Anchored here, once, so
        # the clip's position is a small readable number rather than a remainder of
        # seconds-since-boot; any fixed instant would define the same clock. There is a
        # running loop - the lifespan is what calls this - and it has to be that loop's
        # clock, because that is the one the audio is stamped with.
        first = LIPSYNC_FIRST_CLIP if LIPSYNC_FIRST_CLIP in caches else prepared[0]
        cache, _ = caches[first]
        driver = Driver(
            name=first,
            cache=cache,
            clip=ClipClock(
                fps=fps,
                frames=len(cache.boxes),
                epoch=asyncio.get_running_loop().time(),
            ),
        )
        renderer = Renderer(engine=engine, driver=driver, ring=ring)
        # Encoded per clip on first use, not all ten at boot. The bytes never change
        # once made - a clip is fixed - but ~35MB and ~470ms per 193-frame clip is a
        # bill this process should only pay for clips a session actually reaches.
        driving = _Driving(driver=driver, frames=encode_clip(cache))
        # How long each clip runs, which is the only thing `ClipQueue` needs to know
        # where a boundary is. Read off the caches rather than tabulated: the
        # durations are the owner's footage, not a constant of this design.
        lengths = {
            name: len(clip_cache.boxes) / clip_fps
            for name, (clip_cache, clip_fps) in caches.items()
        }
        clock = FrameClock(fps=fps)
        # One model step on THIS thread before the render loop ever uses another one,
        # and it is a requirement rather than a warm-up. **MLX aborts the process** -
        # `libc++abi: terminating due to uncaught exception of type
        # std::runtime_error: There is no Stream(gpu, 0) in current thread`, not a
        # Python exception anything could catch - if the first UNet/TAESD step of the
        # process runs on a thread other than the one that loaded the weights.
        # Measured 2026-08-26 on mlx 0.32.2: reproducible on every run, and neither
        # `mx.default_stream(mx.gpu)`, a trivial `mx.eval`, nor a whisper-encoder pass
        # on the main thread is enough - only a full `mouths` call is. After it, any
        # thread works indefinitely.
        #
        # `engine.mouths` and not `renderer.render`, deliberately: a render would
        # publish a silence-conditioned frame into the slot and hold its batch partner
        # for the first tick of the first real utterance. This leaves both empty. The
        # window comes from the ring while it is still empty, so it is the right
        # length and all zeros without inventing a shape here.
        silent = ring.window(frame_index=0, fps=fps, origin=0.0, context_ms=CONTEXT_MS)
        engine.mouths([silent, silent], [0, 0], clip=first)
    except Exception:
        # Loud and once. A broken cache, a mismapped weight file or a missing extra
        # must cost the mouth and nothing else: the voice session, the text loop and
        # the pre-rendered clips all still work, and this process has to keep running
        # for them.
        logger.exception("face: lip-sync failed to load; the face will play its clips")
        return None
    # The sink queues; the render loop drains. It is called from `SpeechClock.fed` on
    # the event loop, while `PcmRing.window` is read inside the render thread below,
    # and the ring has no lock: `feed` rebinds `_samples` and then advances `_start`,
    # so a reader landing between those two statements gets new samples against a
    # stale origin - a window off by one chunk, which is a mouth off by one chunk.
    # Queueing here instead means the ring is only ever touched by the loop that owns
    # the render, and only while no render is in flight. `deque.append`/`popleft` are
    # atomic, so the handover needs no lock of its own.
    queued: deque[tuple[bytes, float]] = deque()
    height, width = cache.frames[0].shape[:2]
    logger.info(
        "face: lip-sync ready - %d clips (%s), starting on %s, %.3ffps, %dx%d, "
        "loaded in %.0fms",
        len(prepared),
        ", ".join(prepared),
        first,
        fps,
        width,
        height,
        (time.perf_counter() - started) * 1000.0,
    )

    async def run() -> None:
        await _lipsync_loop(
            face,
            renderer,
            clock,
            ring,
            queued,
            slot,
            driving,
            caches,
            lengths,
            fps=fps,
        )

    return Lipsync(
        frames=_LipsyncFrames(renderer=renderer, slot=slot, driving=driving),
        sink=partial(_queue_pcm, queued),
        run=run,
    )


async def _lipsync_loop(
    face: FaceBus,
    renderer: Renderer,
    clock: FrameClock,
    ring: PcmRing,
    queued: deque[tuple[bytes, float]],
    slot: Slot,
    driving: _Driving,
    caches: dict[str, tuple[Cache, float]],
    lengths: dict[str, float],
    *,
    fps: float,
) -> None:
    """One frame into the `Slot` per frame interval while the daemon is speaking.

    In-process, not a subprocess: CONTRACTS 9 allows the former explicitly and the
    model has to read audio this process is holding.

    **Two halves, because one cannot be both.** Producing a frame takes longer than
    displaying one - a model step covers `BATCH` frames and costs ~73ms against the
    83.3ms two frames are worth - so a single loop that both stepped and published
    could only publish between steps, in pairs. That is what the first build did, and
    at the socket it measured 20.1fps arriving as pairs at 10Hz (gap median 81.6ms
    where a frame is 41.67ms): a head moving smoothly over a mouth that stutters,
    which is what the owner read as the picture getting worse the moment it spoke. So
    `_step` produces into `ready` and `_publish` drains it on the clock. Nothing
    serial fixes this - measured, releasing the held frame a whole interval later
    instead made the cycle 117ms and dropped the socket to 14.2fps.

    **The model step runs in a thread, the decisions do not.** ~73ms of UNet and
    TAESD run inline would hold the event loop for most of every 83ms - starving the
    voice websocket, the audio pump and the activity stream, which is the difference
    between a lip-synced face and a stuttering conversation. Draining `queued` and
    asking `FrameClock` stay here, on the loop, because `loop.time()` is the clock the
    audio is stamped with and because it keeps the ring single-threaded (see the
    sink's comment in `_build_lipsync`).

    **Two single-worker executors, not `asyncio.to_thread`.** Two workers because the
    CPU tail has to overlap the next model step or the ceiling is 42ms a frame instead
    of 36.6 (`render.py:Renderer`), and *single*-worker each because `Renderer.encode`
    reuses one frame buffer and MLX is only safe off the loading thread once
    `_build_lipsync`'s warm-up step has run - one thread of our own makes that a fact
    about this loop rather than about whichever pool worker happened to be free. Not
    the shared default executor either: a 73ms step queued in it delays whatever else
    the process offloads there (starlette runs sync route handlers and file responses
    through it).

    **`loop.time()` for `now`, `origin` and the publish grid.**
    `daemon/voice/conversation.py` stamps every chunk with it deliberately (its
    `_playback_until` note), and `daemon.clock.now()` here would type-check, run, and
    put the mouth an arbitrary offset from the sound.

    **Only while `speaking`.** The measured throughput has 13% of headroom on a real
    duty cycle and none on a continuous run, and the idle windows are also where
    `release_lipsync_memory` keeps resident memory flat - which the spec's section 7
    calls a requirement rather than an optimisation, because without it the spike's
    drift came back.

    **Catches at the top level, once.** A background task that raises is logged by
    nobody and leaves a schedule reading as healthy forever - the failure this
    project's own brief names. Both `Renderer` halves already swallow their own
    failure into `failed`; this is for everything else, and the publisher carries the
    same guard because a task nobody awaits is exactly the shape that goes unnoticed.
    """
    from daemon.face_lipsync.render import (
        BATCH,
        RELEASE_FRAMES,
        ClipClock,
        Driver,
        encode_clip,
    )

    loop = asyncio.get_running_loop()
    interval = 1.0 / fps
    ready: asyncio.Queue[bytes] = asyncio.Queue()
    turn = 0
    available = frozenset(caches)
    clip_queue = ClipQueue(
        current=driving.driver.name,
        ends_at=loop.time()
        + lengths[driving.driver.name]
        - driving.driver.clip.position(loop.time()),
        lengths=lengths,
    )
    """When the clip on screen next reaches its own end, on `loop.time()`'s clock.

    The remainder, not the length: the clock was anchored in `_build_lipsync` and this
    loop starts a moment later, so the first boundary is what is LEFT of the clip.
    Writing it as `position() + length` types and runs and is wrong by however long this
    process has been up - `position` is a place inside the clip and `at` is seconds since
    boot, so `due()` found every boundary already past and rolled its own `while` forward
    a million times on the first tick. The pacing tests caught it as a dropped frame."""
    encoded: dict[str, list[bytes]] = {driving.driver.name: driving.frames}
    """Each clip's stills, encoded on first use and kept - the bytes never change."""
    shots: deque[str] = deque()
    """One-shots the bus published, waiting for a boundary. A deque and not a slot
    because two moods inside one clip is ordinary, and `ClipQueue` decides which
    survives rather than this hand-off."""

    async def watch_shots() -> None:
        """Put every one-shot the bus publishes in front of `clip_queue`.

        An activity is readable as state (`face.state.activity`); a one-shot is not - it
        is an event, and nothing else in this loop would ever see one. Its own task
        because `subscribe()` parks on a wake, and awaiting it inline would stop the
        render between moods.
        """
        async for event in face.subscribe():
            clip = getattr(event, "clip", None)
            if clip is not None:
                shots.append(clip)
    """Bumped on every falling edge. A pair whose encode outlived its own utterance is
    dropped rather than published into the next one - the last turn's mouth finishing
    somebody else's sentence, at exactly the moment the page fades the face in."""

    def collect(at: int, encoded: asyncio.Future[list[bytes]]) -> None:
        """The CPU half's output, back on the event loop, in pair order.

        A callback and not an await, because the whole point of the second worker is
        that this runs *during* the next model step: awaiting it before starting that
        step would serialise the two again, and awaiting it after would put every
        pair's JPEGs a step - 73ms - further behind the sound for nothing.
        """
        if at != turn or encoded.cancelled():
            return
        for frame in encoded.result():
            ready.put_nowait(frame)

    async def publish() -> None:
        """One frame into the slot per interval, and never two inside one.

        `Slot` is latest-wins and never queues, so two puts closer together than the
        transport's 5ms poll means the first is simply never seen: measured on the
        build where a step published its own pair, **85 frames published, 47 seen at
        the socket, 38 lost**, and the mouth advanced two frames at a time. The queue
        is here rather than in `Slot` for the reason `Slot`'s docstring gives - a
        transport that queued would show a mouth lagging the sound by however far
        behind the reader is - and this side is the only place that knows the frame
        rate the queue should come out at.

        `target` is a grid, not `now + interval`: a put lands on it, so the average
        gap is exactly one interval and the sleep's own overshoot does not accumulate
        into 23fps. The floor is what stops a frame that arrived late from following
        the one before it too closely.
        """
        target = loop.time()
        released = 0
        """How far into the falling edge's ramp we are. Reset by every speaking pass,
        so a barge-in mid-ramp starts the next one over."""
        spoke = False
        """Whether there has been anything to release yet. Without it the first ticks
        of a fresh process would each pay an executor hop to be told there is no held
        mouth."""
        try:
            while True:
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                frame: bytes | None = None
                if face.state.activity == "speaking":
                    spoke = True
                    released = 0
                    # Checked here and not only at the falling edge: this half can be
                    # holding a frame it took out of the queue before the turn ended,
                    # and a frame whose sound has already finished playing is not a
                    # frame anyone wants to see. An empty queue mid-sentence publishes
                    # nothing rather than falling through to the clip below - the slot
                    # holds its last frame, which is a mouth one frame stale instead of
                    # a mouth that shut mid-word.
                    try:
                        frame = ready.get_nowait()
                    except asyncio.QueueEmpty:
                        frame = None
                else:
                    index = driving.driver.clip.index(loop.time())
                    if spoke and released < RELEASE_FRAMES:
                        # The utterance just ended. Speech stops and the frame after it
                        # is the clip untouched, so a generated mouth is replaced by a
                        # real one between two frames - measured as a 5.97px step in the
                        # mouth region against a 2.46px median during speech, and seen
                        # as the mouth "갑자기 확 닫히는" snap. `release` ramps the paste
                        # to nothing instead, which takes the step to 2.26px; the
                        # dissolve is the closing motion, because a closed mouth is what
                        # shows through. One frame per tick, and on the same executor as
                        # `encode` because they share the renderer's frame buffer.
                        released += 1
                        frame = await loop.run_in_executor(
                            cpu, partial(renderer.release, index=index, step=released)
                        )
                    if frame is None:
                        # Idle goes out on this same grid, as the clip's own frame
                        # encoded once at load. The page used to play the clip in a
                        # `<video>` here and Chrome's decode of it disagrees with its
                        # decode of our JPEGs - measured R +3.0, G +2.1, B +1.2 - so the
                        # whole picture shifted darker and off-hue the moment speech
                        # started. One decoder, no shift. See `render.encode_clip`.
                        frame = driving.frames[index % len(driving.frames)]
                if frame is not None:
                    slot.put(frame)
                target = max(target + interval, loop.time() + interval / 2)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("face: the lip-sync publisher failed")

    try:
        with (
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="face-lipsync"
            ) as model,
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="face-lipsync-cpu"
            ) as cpu,
        ):
            pacer = asyncio.create_task(publish(), name="face-lipsync-publish")
            watcher = asyncio.create_task(watch_shots(), name="face-lipsync-shots")

            async def switch_to(stem: str) -> None:
                """Move the render onto `stem`, from its frame 0.

                Called only from this half. `Renderer.switch` mutates state the executor
                threads read and `publish` is its own task, so doing it here - between
                awaits, with nothing of ours in flight - is what makes the swap atomic
                as far as either half can see.
                """
                clip_cache, clip_fps = caches[stem]
                stills = encoded.get(stem)
                if stills is None:
                    # On the CPU executor: ~470ms for a 193-frame clip, and this loop
                    # shares the event loop with the voice websocket. The outgoing clip
                    # keeps publishing meanwhile, because `driving` is not swapped until
                    # the stills exist.
                    stills = await loop.run_in_executor(cpu, encode_clip, clip_cache)
                    encoded[stem] = stills
                new = Driver(
                    name=stem,
                    cache=clip_cache,
                    clip=ClipClock(
                        fps=clip_fps,
                        frames=len(clip_cache.boxes),
                        epoch=loop.time(),
                    ),
                )
                renderer.switch(new)
                driving.driver = new
                driving.frames = stills
                # Frames rendered for the outgoing clip must not be published over the
                # incoming one - the same reason a turn boundary drains this.
                while not ready.empty():
                    ready.get_nowait()
            try:
                speaking = False
                sent: int | None = None
                wait = interval
                while not renderer.failed:
                    await asyncio.sleep(wait)
                    # Every path below sets its own next wait. An absolute tick grid
                    # is what this replaced: a step overruns one interval, so the grid
                    # is permanently behind and every later sleep collapses to zero.
                    while queued:
                        chunk, audible_at = queued.popleft()
                        ring.feed(chunk, audible_at)
                    # Clip policy, every iteration and not only while speaking:
                    # `ClipQueue.due` is also what notices the current clip looping, and
                    # a queue that missed a loop would apply the next want wherever it
                    # arrived - the mid-clip cut ADR 0020 exists to avoid.
                    while shots:
                        clip_queue.want(shots.popleft(), one_shot=True)
                    clip_queue.want(
                        wanted(
                            face.state.activity,
                            pending_shot=clip_queue.pending_shot,
                            current=clip_queue.current,
                            available=available,
                            pick=random.choice,
                        )
                    )
                    boundary = clip_queue.due(at=loop.time())
                    if boundary is not None:
                        await switch_to(boundary)
                    if face.state.activity != "speaking":
                        if speaking:
                            speaking = False
                            sent = None
                            turn += 1
                            while not ready.empty():
                                ready.get_nowait()
                            release_lipsync_memory()
                        wait = interval
                        continue
                    speaking = True
                    if ready.qsize() >= BATCH:
                        # Backpressure, and the only thing that bounds the queue: a
                        # whole pair still unpublished means the publisher is the slow
                        # half, and another step would buy latency rather than motion.
                        wait = interval / 4
                        continue
                    # Read `origin` once and pass the same value to the clock and the
                    # renderer: it moves, and asking twice would let the frame the
                    # clock allowed be windowed against a different one.
                    origin = ring.origin
                    # The newest frame whose audio is already here, not the next one in
                    # sequence - see LIPSYNC_CATCHUP_LIMIT. `None` is normal rather
                    # than a fault: a step covers `BATCH` frames and cannot start until
                    # the last of them has its audio, so the first passes of a turn are
                    # a wait by design.
                    found = None
                    for _ in range(LIPSYNC_CATCHUP_LIMIT):
                        due = clock.due(now=loop.time(), origin=origin)
                        if due is None:
                            break
                        found = due
                    if found is None:
                        # A quarter of a frame, not a whole one: the next grant is less
                        # than an interval away and sleeping a full one overshoots it.
                        wait = interval / 4
                        continue
                    if sent is not None and found < sent:
                        # The count went backwards, so the clock re-anchored under us -
                        # a new turn inside one speaking stretch, or a barge-in. The
                        # contiguity below is measured from the new count, not the old
                        # one, or the loop would wait for a grant that will never come.
                        sent = None
                    if sent is not None and found < sent + BATCH:
                        # A pair has to start where the last one ended. `due` hands out
                        # one grant per call and a step covers `BATCH` of them, so this
                        # half is now *faster* than the clock it is paced by - 27.3fps
                        # of capacity against 24fps of grants - and a pass that gained
                        # only one grant has nothing new to render. Stepping anyway
                        # would render `found + 1` a second time and publish it twice:
                        # a mouth that pauses for a frame three times a second, which
                        # is a stutter rather than a frame rate. The first build could
                        # not reach this - it ran at 20fps, behind the grants, always
                        # skipping forward.
                        wait = interval / 4
                        continue
                    sent = found
                    # Awaited, and that is what keeps the ring lock-free: the drain
                    # above is the only writer and it cannot run while this is in
                    # flight. `Renderer.step` reads the ring inside the thread.
                    step = await loop.run_in_executor(
                        model,
                        partial(
                            renderer.step, frame_index=found, origin=origin, fps=fps
                        ),
                    )
                    if step is None:
                        continue        # latched; the `while` above ends the loop
                    encoding = loop.run_in_executor(cpu, renderer.encode, step)
                    encoding.add_done_callback(partial(collect, turn))
                    # Straight on: the next step's 73ms is the window this pair's
                    # 11ms of compositing and JPEG has to finish inside.
                    wait = 0.0
            finally:
                pacer.cancel()
                watcher.cancel()
    except asyncio.CancelledError:
        # Shutdown, not a failure. Explicit because the clause below would not catch
        # it anyway and a reader should not have to know that.
        raise
    except Exception:
        logger.exception("face: the lip-sync render loop failed")
    logger.warning(
        "face: the lip-sync render loop has stopped; the face is back on its clips"
    )


def release_lipsync_memory() -> None:
    """Hand MLX's buffer cache back between utterances.

    Two lines in their own function because they are the only `mlx` in this module and
    `mlx` has no Linux wheel: the render loop above has to be exercisable in CI, and a
    function-scoped import inside that loop would make it importable but not runnable.
    Spec section 7 calls this a requirement rather than an optimisation - running MLX
    without it is what let the spike's cache grow until frame drift reached +41.2% and
    the dev machine stalled once.
    """
    import mlx.core as mx

    mx.clear_cache()


async def _proactive_tick(
    settings: Settings, get_bridge: Callable[[], Any] | None = None
) -> None:
    """The five-minute round. Catches everything, for the same reason the reflection
    tick does: a job that raises inside APScheduler is logged once and then the
    schedule carries on, which reads as a working loop that has silently decided
    nothing for a month.

    Logged at INFO even when nothing happened, because "it stayed silent" is the
    output people need to be able to check.

    `get_bridge`, when given, is called fresh on every fire to read whatever the
    lifespan's `app.state.mcp` currently is - a callable rather than the bridge
    itself, because `scheduler.add_job` captures its `args` once at registration
    time, before `_lifespan` has necessarily finished building `app.state.mcp`
    (see `_lifespan`'s own registration call). `None` (the default, and what every
    test constructing this function directly gets) means `build_proactive_tick`
    builds its own bridge, exactly as it always has.
    """
    bridge = get_bridge() if get_bridge is not None else None
    try:
        # `wake_loop=True` because this is the resident, the only process that runs
        # one. Not conditioned on the task being *alive*: a resident whose wake task
        # died or was never created answers nothing, `mic_floor.request` reports
        # `no-listener` after its take timeout, and `ProactiveDelivery._say` uses the
        # speaker directly - correct, and self-correcting, at the cost of one ten-
        # second wait per line in a state `daemon doctor` already reports.
        tick, close = await build_proactive_tick(
            settings, speak=True, bridge=bridge, wake_loop=True
        )
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


async def run_reflection_now(
    settings: Settings, lock: asyncio.Lock | None
) -> list[Result]:
    """Reflect on every unreflected day, and *raise* if it could not.

    The opposite contract from `_reflect_tick`, on purpose. A scheduled job that
    raises inside APScheduler stops being scheduled, so the tick swallows. A
    button press has a person waiting for the answer, and swallowing there would
    report success for a pass that never reached the model.

    `lock` is `app.state.catchup_lock`: the same one the cron and the boot task
    take, because this is a third writer of the same append-only artifact.
    """
    reflection_pass, close = await build_reflection(settings)
    try:
        async with lock if lock is not None else nullcontext():
            return await reflection_pass.catch_up()
    finally:
        with suppress(Exception):
            await close()


async def run_persona_evolution_now(
    settings: Settings, lock: asyncio.Lock | None, *, force: bool = False
) -> EvolutionResult:
    """Run the weekly pass now, and raise if it could not. Same split as
    `run_reflection_now`, same reason.

    `lock` is `app.state.catchup_lock` here too: two `run()` in one week would
    both write the week's diary and re-consume observations.
    """
    evolution, close = await build_persona_evolution(settings)
    try:
        async with lock if lock is not None else nullcontext():
            return await evolution.run(force=force)
    finally:
        with suppress(Exception):
            await close()


async def _reflect_tick(settings: Settings, lock: asyncio.Lock | None = None) -> None:
    """The scheduled pass. Catches everything: a job that raises inside
    APScheduler is logged once and then the schedule carries on, which reads as a
    working reflection loop that has silently done nothing for a month.

    The work itself is `run_reflection_now`, which raises - the admin's button
    needs the failure. This wrapper is the swallowing half.
    """
    try:
        results = await run_reflection_now(settings, lock)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("reflection tick failed: %s", exc)
        return
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
    reflection and the proactive tick.

    Logged at INFO even when the pass was skipped, because "not enough
    observations yet" and "already ran this week" both have to be visible
    without opening sqlite.
    """
    try:
        result = await run_persona_evolution_now(settings, lock)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("persona evolution tick failed: %s", exc)
        return

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


@dataclass
class VoiceRuntime:
    """What a wake-opened conversation reuses instead of rebuilding.

    Built once at resident startup and handed to every wake round. Before this
    existed each wake word rebuilt the lot - reopened sqlite, reindexed, and
    reconnected every MCP server - which put ~4s of setup between "벨라" and the
    daemon being able to say anything, on top of the session handshake. Boot-once
    services are how every voice stack does it (LiveKit/Pipecat keep pipelines warm
    for the process lifetime); the billed Gemini session stays per-conversation.

    Ownership stays with the lifespan: `run_voice` treats everything here as
    borrowed and closes none of it, and the lifespan's shutdown closes the MCP
    bridge and the runner exactly as it does the text path's.
    """

    store: Any
    writer: Any
    recall: Any
    tools: Any
    mcp: Any
    screen_share: Any


async def _build_voice_runtime(
    settings: Settings,
    store: Any,
    writer: Any,
    recall: Any,
    *,
    delegate_wake: asyncio.Event | None = None,
    channel: Any = None,
    face: FaceBus | None = None,
) -> VoiceRuntime:
    """The voice half of the tool layer, built once at startup.

    A separate runner from the text path's on purpose: voice degrades `ask` to
    `allowlist` (a spoken turn has nowhere to surface an approval) and carries the
    live-share start/stop tools bound to a `ScreenShareController`, neither of which
    the text registry has. The price is a second set of MCP connections held open -
    paid once at boot instead of on every single wake word.
    """
    screen_share = None
    if settings.screen_enabled:
        route = settings.route_for(Task.CHAT_VOICE)
        if route.provider not in VIDEO_CAPABLE_VOICE_PROVIDERS:
            # See VIDEO_CAPABLE_VOICE_PROVIDERS: this provider's session drops
            # every frame, so building the controller here would only let the
            # model promise a screen it will never see.
            logger.warning(
                "live screen share unavailable on voice provider %r (send_frame "
                "is a no-op there); live-share tools will not be offered",
                route.provider,
            )
        else:
            # Guarded like run_voice's own block: a missing Pillow must cost the
            # feature, not the resident.
            try:
                from daemon.voice.screen_share import ScreenShareController

                screen_share = ScreenShareController()
            except ImportError as exc:
                logger.warning("voice screen sharing off (missing dependency): %s", exc)
    voice_mode = "allowlist" if settings.tools_mode == "ask" else settings.tools_mode
    tools, mcp, _status = await _build_tools(
        settings,
        store,
        mode=voice_mode,
        screen_share=screen_share,
        delegate_wake=delegate_wake,
        channel=channel,
        face=face,
    )
    return VoiceRuntime(
        store=store, writer=writer, recall=recall, tools=tools, mcp=mcp, screen_share=screen_share
    )


def _mood_declaration() -> ToolSpec:
    """`set_mood` as the model is told about it. **Declared, never registered.**

    A `ToolSpec` is how a model is offered anything at all, so this rides the
    function-calling channel - that is transport, not a claim that it is a tool. What
    makes it not one is that `daemon/tools/` has no entry for it and
    `daemon/voice/conversation.py` answers it before `ToolRunner` is reached, which is
    what CONTRACTS 12's exemption rests on (docs/adr/0018).

    Flat on purpose - one enum string. `evals/voice_write_nudge_spike.py` measured that
    nested argument schemas are what the voice model fakes rather than calls, and
    `evals/voice_set_mood_spike.py` measured this shape at 24/24 over the live socket.
    """
    return ToolSpec(
        name=MOOD_TOOL,
        description=(
            "Set the facial expression shown on the companion's own face. Call this "
            "when you genuinely feel amused, sulky or curious about what was just "
            "said. It changes nothing except the expression."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mood": {"type": "string", "enum": ["amused", "sulky", "curious"]}
            },
            "required": ["mood"],
        },
    )


async def run_voice(
    settings: Settings,
    *,
    opening_audio: bytes = b"",
    opening_text: str = "",
    shared: VoiceRuntime | None = None,
    face: FaceBus | None = None,
    on_spoke: Callable[[], None] | None = None,
    opening_already_logged: bool = False,
    pcm_sink: Callable[[bytes, float], None] | None = None,
) -> int:
    """One spoken conversation at this machine, then exit.

    Assembled here rather than inside the daemon's own loop because voice is a
    thing a person starts, not a thing that happens to them: the session is
    billed per minute, so holding one open on the chance of being spoken to is
    pure cost (docs/PLAN.md 6.5).

    Proactive speech at the machine is one of the things that starts one, though,
    and it is the caller `on_spoke` and `opening_already_logged` exist for -
    `_speak_unprompted` is the only one that passes either. `on_spoke` fires on the
    first chunk of audio the session produces - from `VoiceConversation._on_audio`,
    the one place that knows, and *not* at the end of the attempt, which is far too
    late for the caller waiting on it - and is what decides whether the line still
    needs `/usr/bin/say`; `opening_already_logged` says the opening is
    already in the conversation log, so the turn that speaks it must not be written
    down a second time (`VoiceConversation._skip_opening_record`).

    Returns a shell exit code, because the caller is a CLI command.
    """
    from daemon.fs import harden_existing
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter
    from daemon.voice.gemini_live import GeminiLiveError, GeminiLiveSession
    from daemon.voice.openai_realtime import OpenAIRealtimeError, OpenAIRealtimeSession

    if not settings.voice_enabled:
        logger.error("voice is off; set DAEMON_VOICE_ENABLED=true (see `daemon setup`)")
        return PROBLEM
    # `Settings._check` has already applied `voice_session_problems()` at load
    # time, so there is deliberately no second application here. It briefly lived
    # in this function, while `voice_enabled` meant both "a hosted session may
    # run" and "a proactive line may leave the local speaker" and the `offline`
    # preset could satisfy only the second - checking at load then stopped
    # `Settings` from loading at all, which stops the daemon. ADR 0012 removed
    # that premise by making voice its own axis, which put the checks back where
    # they belong, and a repeat here would only be a branch no configuration can
    # reach.
    #
    # route_for raises with the specific reason - voice disabled, no live model
    # id - which is more use than anything this function could say about it.
    route = settings.route_for(Task.CHAT_VOICE)

    harden_existing(settings.data_dir)
    # `shared` is the resident's boot-once voice runtime (VoiceRuntime): store,
    # recall, tool layer and MCP connections built at startup and reused across
    # wake rounds. Rebuilding them here cost every single wake word ~4s of MCP
    # reconnect plus a reindex before the daemon could say anything - the owner's
    # "why does it take six seconds to answer". `daemon voice` (the CLI, one
    # conversation per process) passes nothing and keeps building - and closing -
    # its own, which is what `owns` guards.
    owns = shared is None
    store = Store.open(settings.data_dir / DB_FILENAME) if owns else shared.store
    try:
        if owns:
            reindex(settings.data_dir, store)
            writer = FileMemoryWriter(settings.data_dir, store)
            recall, _status, embedder = _build_recall(settings, store)
        else:
            writer, recall, embedder = shared.writer, shared.recall, None
        if owns:
            # The controller for the live-share start/stop tools (Task 2.3). Built
            # only when screen sharing is on at all - `None` otherwise, which is what
            # keeps those two tools off `_build_tools`'s registry entirely. Built here
            # rather than inside `_voice_attempts` because the same instance has to
            # survive a reconnect: the tools registered below hold a reference to it,
            # and a fresh controller per attempt would leave them pointing at a stale
            # one.
            screen_share = None
            if settings.screen_enabled and route.provider not in VIDEO_CAPABLE_VOICE_PROVIDERS:
                # See VIDEO_CAPABLE_VOICE_PROVIDERS: this provider's session drops
                # every frame it is handed (OpenAIRealtimeSession.send_frame is a
                # no-op), so building the controller - and registering its
                # start/stop tools below - would let the model tell the owner
                # it is watching a screen it will never see. ADR 0009 requires
                # the code say so instead.
                logger.warning(
                    "live screen share unavailable on voice provider %r (send_frame "
                    "is a no-op there); live-share tools will not be offered",
                    route.provider,
                )
            elif settings.screen_enabled:
                # Guarded like the screen-tool block in `_build_tools`: screen sharing
                # needs Pillow (daemon/voice/screen_share.py imports it at module
                # scope). A missing Pillow must lose only the feature, not crash the
                # whole voice session on the wake word - the failure this caught on
                # the owner's Mac.
                try:
                    from daemon.voice.screen_share import ScreenShareController

                    screen_share = ScreenShareController()
                except ImportError as exc:
                    logger.warning("voice screen sharing off (missing dependency): %s", exc)
            # Tools follow the owner's configured mode - `full` for this install, so a
            # spoken turn runs guarded tools the same as the text path does. A
            # microphone has no relay path, so a spoken turn is the owner's own words
            # and the origin gate is the real boundary; pinning `allowlist` here
            # silently refused every guarded call the owner made by voice, `open_path`
            # among them. The one mode a spoken turn cannot honour is `ask`: it has
            # nowhere to surface an approval, so `ask` would pile up rows that lapse
            # unanswered - the silent degradation this repo calls the dangerous
            # failure - and so degrades to `allowlist` here. The allowlist and
            # standing grants are the same table the text path edits, so voice reads
            # the surface text writes to; it just never adds to it. Off entirely when
            # `DAEMON_TOOLS_ENABLED` is false, exactly like text.
            voice_mode = "allowlist" if settings.tools_mode == "ask" else settings.tools_mode
            tools, mcp_bridge, _tools_status = await _build_tools(
                settings, store, mode=voice_mode, screen_share=screen_share, face=face
            )
        else:
            screen_share = shared.screen_share
            tools, mcp_bridge = shared.tools, None
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
        # conversation surface. The proactive judge reaches the same two files too,
        # just not through this path - it builds its own persona block directly
        # (daemon/proactivity/judge.py, `Judge._persona`), because a background
        # tick has no `Companion` to ask.
        seed = await companion.persona()
        # Owner, always: a microphone has no relay path, so a spoken turn is the
        # owner's own words (daemon/voice/conversation.py `_record`), and the origin
        # gate offers tools only to it. Empty when tools are off, which leaves the
        # session declaring none and so never yielding a tool call.
        tool_specs = companion.specs(origin="owner", surface="voice")
        # `set_mood` is declared here rather than by `Companion.specs`, and only with a
        # face attached, for two separate reasons. It is not a tool - it never reaches
        # `ToolRunner` and leaves no audit row (docs/adr/0018, CONTRACTS 12) - so it has
        # no business in the list a registry builds. And declaring it without a face
        # would offer the model a switch wired to nothing. This module is the one
        # allowed to assemble (CONTRACTS 4), and it is the only place that has both.
        mood_specs = [_mood_declaration()] if face is not None else []
        tool_specs = (*tool_specs, *mood_specs)
        # The tool contract rides with the persona in the system instruction, so the
        # endpoint getting tools inherits the rules the text path already has instead
        # of being written without them - which is exactly how voice came to have no
        # index() call (daemon/companion.py, TOOL_CONTRACT). Only when there is a tool
        # to use: 200 tokens of rules about a capability the model lacks buys nothing.
        #
        # `MOOD_VOICE_INSTRUCTION` only when the switch is actually on offer, same
        # rule: an instruction about a tool the model does not have is pure tax, and
        # its "never say this out loud" sentence is load-bearing (0/32 spoken aloud,
        # `evals/voice_set_mood_spike.py`) rather than decorative.
        instruction_parts = [
            block
            for block in (
                seed,
                TOOL_CONTRACT if tool_specs else "",
                MOOD_VOICE_INSTRUCTION if mood_specs else "",
            )
            if block
        ]
        system_instruction = "\n\n".join(instruction_parts) or None
        audio = build_voice_audio()
        # Before the handshake, so the acknowledgement is as close to the wake word
        # as it can be. Nothing is feeding the session yet, so the cue cannot be
        # heard as the owner interrupting.
        #
        # Not when the daemon is the one about to talk. The cue means "the
        # microphone is yours", and `on_spoke` is set by exactly one caller:
        # `_speak_unprompted`, which opens a session to *say* something unprompted.
        # Playing it there tells the owner to go ahead and then talks over him 1.3 s
        # later, which is the opposite of what it means. Under `origin/main` the
        # order was line-then-cue - `/usr/bin/say` spoke first and this ran
        # afterwards - so routing the line through the session inverted it; caught
        # in review of PR #126, whose whole subject is how being spoken to first
        # sounds. The cue comes back on the owner's next turn, in the same session,
        # because the session stays open to listen.
        if on_spoke is None:
            await play_ready_cue(audio)

        def new_session() -> Any:
            """A fresh session per attempt. Reconnecting means starting clean: the
            old one carries a half-flushed transcript, a partial-transcript queue
            nobody will read again, and a log filter holding the API key."""
            if route.provider == "openai":
                return OpenAIRealtimeSession(
                    api_key=settings.openai_api_key,
                    model=route.model,
                    system_instruction=system_instruction,
                    tools=tool_specs,
                    voice_name=settings.openai_realtime_voice,
                )
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
                (GeminiLiveError, OpenAIRealtimeError),
                opening_audio=opening_audio,
                opening_text=opening_text,
                screen_share=screen_share,
                screen_pump_factory=screen_pump_factory,
                barge_in=settings.voice_barge_in,
                face=face,
                on_spoke=on_spoke,
                opening_already_logged=opening_already_logged,
                # Where the lip-sync ring is fed, when this process has one. It rides
                # all the way down to `SpeechClock`, which is the one place that
                # already computes when a chunk becomes *audible* rather than when it
                # was queued - a second tap would have to repeat that arithmetic, and
                # the copy that drifts is the one that makes the mouth look dubbed.
                pcm_sink=pcm_sink,
            )
        finally:
            with suppress(Exception):
                await audio.close()
            # Everything below is owned teardown: a shared runtime's store, tools
            # and MCP connections belong to the resident's lifespan and must
            # outlive this one conversation - closing them here would take the
            # next wake round's tools with it.
            # Before the sqlite close below, because an MCP server is a child process:
            # one left running is an orphan per `daemon voice` run, the same reason the
            # lifespan closes the bridge ahead of the store.
            if mcp_bridge is not None:
                with suppress(Exception):
                    await mcp_bridge.aclose()
            if owns and tools is not None:
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
        if owns:
            store.close()


READY_CUE_HZ = (784.0, 1046.5)
"""Two short notes, G5 then C6, rising. A rising pair reads as "go ahead" where a
falling one reads as "finished", and neither is a word - so it cannot be mistaken for
the daemon speaking or transcribed as one."""

READY_CUE_MS = 180
READY_CUE_GAIN = 0.35
"""Short, and loud enough to be unmistakable. The cue answers "may I speak now?",
and the honest answer is that until it existed there was none: the wake gate released
the microphone and about a second passed with nothing to say the session was live, so
the owner guessed - and guessing early is how an utterance lands in the handover and
is lost.

Raised from 90 ms / 0.18 after the owner reported the failure this was supposed to
prevent, in its worst form. Below about a second of silence a person assumes the wake
word did not register and **says it again, louder** - and on this provider each repeat
lands in the already-open session as a fresh turn that *interrupts the answer being
generated*, which the server then discards and regenerates from the top (documented:
Live API cancels and discards an interrupted generation; google-gemini/cookbook#1197
reproduces the resulting restart loop on this exact model). The observed symptom was
the daemon restarting the same sentence three times. So the acknowledgement has to be
heard the first time: an ack nobody notices is the root of a cascade, not a nicety.
Still well under the "startle" bound, still not a word, and still played through the
echo-cancelled engine so it cannot be heard as speech by the far end."""


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
    session_error: type[Exception] | tuple[type[Exception], ...],
    opening_audio: bytes = b"",
    opening_text: str = "",
    screen_share: Any = None,
    screen_pump_factory: Callable[[Any], Any] | None = None,
    barge_in: bool = True,
    face: FaceBus | None = None,
    on_spoke: Callable[[], None] | None = None,
    opening_already_logged: bool = False,
    pcm_sink: Callable[[bytes, float], None] | None = None,
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
    # Carried and cleared exactly like `pending_opening` above, and for a sharper
    # reason: with `CALLED_BY_NAME` a repeat was a redundant "네?", while with
    # `SPEAK_VERBATIM` it is a second literal delivery of the proactive line, in the
    # same voice, to an owner who has already answered the first one.
    pending_text = opening_text
    reported = False

    def note_first_audio() -> None:
        """Tell the caller the daemon has started talking, once per call.

        Handed to every attempt's conversation, which fires it on the first chunk of
        audio it receives - the instant the answer begins, which is the only instant
        the one caller can use. `_speak_unprompted` is blocked inside
        `mic_floor.request` under `REPLY_CEILING_SECONDS` while a conversation runs
        with no total cap of its own, so a report at the end of the attempt is a
        report after the deadline: the row would record `route='telegram'`,
        `modality='text'` for a line the owner heard in her voice and answered aloud.
        PR #126 shipped it in this function's `finally` and had exactly that.

        What the earlier placement was right about is kept for free: a failure that
        is not a `session_error` leaves `_voice_attempts` entirely, and firing at the
        first chunk has already reported by the time anything can raise.

        `reported` rather than clearing `on_spoke`, so the guard reads the same on a
        reconnect: a second attempt that plays audio must not report twice, because
        the caller uses this to decide whether the line still needs `/usr/bin/say`
        and it may already have answered a future on the strength of the first one.
        """
        nonlocal reported
        if on_spoke is None or reported:
            return
        reported = True
        # Deliberately not "the room heard it": `_on_audio` counts the chunk one
        # line before `AudioIO.play` is awaited with it, so a dead output device
        # reports here too. That is the safe direction - a false positive costs a
        # line nobody heard, a false negative says the same sentence twice.
        on_spoke()

    for attempt in range(1, VOICE_RECONNECT_ATTEMPTS + 1):
        session = new_session()
        conversation = VoiceConversation(
            session,
            audio,
            companion,
            opening_audio=pending_opening,
            opening_text=pending_text,
            # Only while the opening is still pending. Once it has been said, a
            # later attempt's first assistant turn is an answer to the owner and
            # belongs in the log like any other voice turn.
            opening_already_logged=opening_already_logged and bool(pending_text),
            on_first_audio=None if on_spoke is None else note_first_audio,
            screen_share=screen_share,
            screen_pump_factory=screen_pump_factory,
            barge_in=barge_in,
            face=face,
            # Handed to every attempt, like `face`. A reconnect builds a fresh
            # conversation and therefore a fresh `SpeechClock`; the ring behind this
            # sink is the process's one ring, and it re-anchors on the discontinuity
            # the new turn's timestamps make (daemon/face_lipsync/ring.py).
            pcm_sink=pcm_sink,
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
            # question twice - or, for a proactive opening, say it twice.
            pending_opening = b""
            pending_text = ""
        elif pending_opening or pending_text:
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
        gate = getattr(state, "wake_gate", None)
        if gate is None:
            # Between rounds - usually because a conversation holds the microphone.
            return "running"
        # The live counters, so "is it actually hearing?" is answerable from
        # outside: frames at zero with the gate up is a dead capture stream, not a
        # quiet room (the failure that used to be invisible until someone spoke to
        # a deaf machine).
        c = gate.counters
        return f"running, {c.frames_seen} frames, {c.transcribed} transcribed, {c.fired} fired"
    return "stopped"


def _mic_health() -> str:
    """The microphone TCC decision, read (never prompted) at request time. `n/a`
    off macOS, where there is no TCC gate."""
    if sys.platform != "darwin":
        return "n/a"
    from daemon.voice.mic_access import microphone_authorization_status

    return microphone_authorization_status()


def _only_the_wake_word(fired: Any) -> bool:
    """Did the owner call the name and stop, with no question attached?

    Compared in the wake gate's own normalised form, so spacing, punctuation and
    Unicode form cannot make "벨라" look different from the alias it matched.
    """
    from daemon.voice.wake import normalise

    return normalise(fired.heard) == normalise(fired.matched)


async def _wake_round(
    settings: Settings, shared: VoiceRuntime | None = None, state: Any = None
) -> None:
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
    if state is not None:
        # The live gate's counters, for /health: `frames_seen` is what separates "a
        # quiet room" from "a dead capture stream" from outside the process - the
        # question nobody could answer the night the resident went deaf while
        # `running` (see WakeGate's dead-stream handling).
        state.wake_gate = gate
    fired = None
    # Held rather than iterated anonymously, because the handover depends on *when*
    # this generator is finalised. `async for gate.listen()` leaves that to the
    # garbage collector: breaking out drops the last reference, CPython schedules the
    # `aclose()` on a later loop turn, and `record`'s `finally` - the one that hands
    # the microphone to its release thread - had not run yet by the time the session
    # below built a second CoreAudio client on the same device. That race deadlocked
    # the resident (see `close_gate` and daemon/voice/audio.py:wait_for_input_release).
    listening = gate.listen()
    released = True
    try:
        async for event in listening:
            fired = event
            break
    finally:
        if state is not None:
            state.wake_gate = None
        # Before the conversation, not after: this is what hands the microphone over
        # and stops the gate hearing what happens next. `aclose()` first, so the
        # release is under way and `close_gate` has something to wait for.
        with suppress(Exception):
            await listening.aclose()
        try:
            released = await close_gate()
        except Exception:
            # Fail closed, and this is the one place in the round that does. Every
            # other guard here starts from "carry on" because losing a round is worse
            # than the failure it is guarding; this one is the opposite, because what
            # a swallowed error costs is not a round but the guarantee - `released`
            # would stay True, the handover would race exactly as it used to, and
            # nothing anywhere would say so. A daemon that opens no session is noticed
            # on the first wake word; a daemon that deadlocks once a day is not.
            logger.exception("wake: releasing the microphone failed; not opening a session")
            released = False
    if not released:
        # The microphone never came back. Opening a session now means building a
        # VoiceProcessing engine on a device that is already wedged, which is what
        # turned a stuck stop into a daemon that never heard another word - so the
        # round is lost instead, loudly, and the caller comes back around.
        if fired is not None:
            # Said here rather than left to `close_gate`'s device error, because the
            # gate has already logged `wake: heard ...` and a reader would otherwise
            # see a matched wake word followed by silence with nothing joining them.
            logger.error(
                "wake: heard %r but dropping it - the microphone never came back",
                fired.heard,
            )
        # `mic_floor.take` hands over a debt as well as a line: the taker owes the
        # future exactly one answer, and a line nobody takes sits out its own timeout
        # instead of falling back to Telegram now, while it is still worth saying.
        taken = mic_floor.take()
        if taken is not None:
            logger.error("wake: not speaking a waiting line - the microphone is wedged")
            mic_floor.answer(taken[1], False)
        # Paced on the way out for the reason WAKE_RETRY_SECONDS exists: the next
        # round opens a fresh capture on the same wedged device and parks another
        # thread on it, and an unpaced return would do that as fast as the process
        # can manage.
        await asyncio.sleep(WAKE_RETRY_SECONDS)
        return
    if fired is None:
        # The stream ended without a wake word - a closed device, a test's scripted
        # audio running out, the gate's own dead-stream watchdog asking to be
        # rebuilt, or proactivity asking for the microphone. Not an error, but not a
        # reason to spin either; the caller's own guard handles the pacing.
        taken = mic_floor.take()
        if taken is not None:
            await _speak_unprompted(settings, taken, shared)
        return
    logger.info("wake: heard %r matching %r; opening a voice session", fired.heard, fired.matched)
    # The segment that fired the gate goes with it - but only when it carries more
    # than the name. Without it the session opens deaf to the question it was opened
    # for: the gate consumed "루시 뭐 해", matched on the alias, and the owner had to
    # say "뭐 해" again into a microphone that had just changed hands.
    #
    # When the owner *only* called the name, the segment is the name alone, and
    # handing over its audio gives the session's own transcriber a syllable to
    # misread as a first utterance. Measured: "벨라" arrived at the model as "별로"
    # - an ordinary Korean word meaning "not really" - so the daemon opened by
    # answering something the owner had not said. The local recognizer has already
    # decided what this segment is, so the *words* are settled and only the audio
    # is ambiguous: the name goes over as text instead, where there is nothing left
    # to mishear. Sending nothing at all was tried in between and is worse - the
    # owner called, got silence, waited ten seconds and called again.
    from daemon.voice.conversation import CALLED_BY_NAME

    name_only = _only_the_wake_word(fired)
    await run_voice(
        settings,
        opening_audio=b"" if name_only else fired.pcm,
        # What being called *means*, not what it sounded like: the recognizer's
        # rendering is routinely not the name (see CALLED_BY_NAME), and handing that
        # over made the daemon puzzle over a word nobody said.
        opening_text=CALLED_BY_NAME if name_only else "",
        shared=shared,
        # `state` is `app.state` when this round is the resident's own (the
        # injected-round tests pass none) - the same bus the text loop publishes
        # to, so the face reflects a spoken turn exactly as it does a typed one.
        face=getattr(state, "face", None),
        # And the same state's lip-sync sink, `None` unless `_lifespan` assembled a
        # renderer. Read here rather than captured when the wake loop started, so
        # that the switch is a property of the process rather than of the moment the
        # gate came up.
        pcm_sink=getattr(state, "face_pcm_sink", None),
    )
    # Let the conversation's Voice-Processing unit finish releasing the microphone
    # before the next round opens a fresh capture on it - see WAKE_REARM_SETTLE_SECONDS.
    await asyncio.sleep(WAKE_REARM_SETTLE_SECONDS)


async def _speak_unprompted(
    settings: Settings, taken: tuple[str, asyncio.Future[bool]], shared: VoiceRuntime | None
) -> None:
    """Say a proactive line at this machine, then listen for the answer.

    Reached only from `_wake_round`, after its `finally` has closed the gate - so
    the microphone is already free, released by the one sequence in this process
    that is allowed to release it. Nothing here opens or closes a capture stream.

    **The session says it, in her own voice.** A session was always going to open
    here - the daemon has just spoken first, and the owner is likeliest to answer
    in the next few seconds, the one moment a gate rebuild would miss - so letting
    it deliver the line costs nothing and ends the thing the owner reported on
    2026-08-27: a proactive line arriving in `/usr/bin/say`'s system voice while
    every answer he had ever heard came from the one he picked.

    PR #115 built it the other way round and argued that `opening_text` is a prompt
    the model *answers*, so the line would come out as a paraphrase and the sentence
    the judge length-capped and refused for URLs would not be the sentence the room
    heard. That argument was never measured, and it was wrong:
    `evals/proactive_verbatim_spike.py` (8 live sessions per cell) puts a plain
    instruction at **exact 0/8** and `SPEAK_VERBATIM` at **8/8**, and the same again
    after the nonce fence went in - 0/16 against 16/16 over two runs. The 0/8 is not
    paraphrase either: the model says the line and then adds a question of its own,
    which `SPEAK_VERBATIM` forbids by name. What that measurement does *not* cover -
    the far harder prompt production actually sends around it - is written out in
    `SPEAK_VERBATIM`'s own docstring, and nobody has measured that.

    Note what is *not* at risk, and was overstated in that PR: the search titles
    never reach a voice session at all (they exist only in the judge's prompt), so
    the model cannot speak a pointer it has never seen. What was really at risk is
    that `proactive_utterances.text` - the row the owner's label attaches to - stops
    being what was said, which is why the wording is measured and pinned.

    `/usr/bin/say` stays as the fallback, for an install with voice off and for a
    session that never played anything - guarded on whether audio was handed to the
    player (`on_spoke`, and see what it can and cannot tell in `_voice_attempts`)
    rather than on whether an exception was raised.

    **The session knows what was said because it said it**, which is what makes the
    ordering here different from PR #115's. There, `deliver` ran `_say` -> `_send`
    -> `_log` and the log row raced this function's websocket connect for a place in
    `continuity_block`. Now the line is this session's own first turn, so nothing
    has to be carried into its context at all - and `_log` no longer even tries to
    get there first: `_say` does not return until `note_spoke` answers the future,
    which happens once the line is playing. The row lands during the conversation
    rather than before it, and the only thing that depended on the old order - the
    session knowing what it had just said - is now structural.

    That row is also why the session is opened with `opening_already_logged=True`.
    `_log` writes the sentence as `session_kind="proactive"`; the transcript of the
    turn that speaks it would be a second copy, filed as `voice`, and
    `daemon/memory/store.py`'s three M3 readers select exactly
    `session_kind IN ('interactive', 'voice')` so that the daemon's own speech does
    not reset the 12-hour silence clock it is measured against.

    `IDLE_TIMEOUT_SECONDS` bounds the cost at 30 seconds of silence, which is what
    makes this affordable on a per-minute session that may well go unanswered.
    """
    from daemon.voice.conversation import speak_verbatim

    text, future = taken
    spoke = False

    def note_spoke() -> None:
        """The room has heard the line. Answer the caller now, not at the end."""
        nonlocal spoke
        spoke = True
        # Here rather than after the conversation, because the conversation has no
        # total cap: `VoiceConversation._receive` reschedules its idle budget per
        # audio item, so a real exchange plus the closing 30 s of silence runs past
        # `mic_floor.REPLY_CEILING_SECONDS`. Waiting would have `request` log a
        # broken contract and return `not-spoken`, and the row would then record
        # `route='telegram'`, `modality='text'` for a line the owner heard in her
        # voice and answered aloud - the same verdict-and-outcome mismatch the
        # ceiling exists to close. Answering now also lets `deliver` finish while
        # she is still talking, so the Telegram copy and its label buttons arrive in
        # seconds rather than minutes, and the proactive tick stops being blocked
        # for the length of a conversation under `max_instances=1`.
        mic_floor.answer(future, True)

    try:
        try:
            await run_voice(
                settings,
                # Fenced under a fresh nonce, not `.format`ted here: this is the
                # judge's reply, and it is about to enter a model that holds this
                # install's tools. See `speak_verbatim`.
                opening_text=speak_verbatim(text, secrets.token_hex(4)),
                shared=shared,
                on_spoke=note_spoke,
                # `deliver` already wrote this sentence down as `proactive`. The
                # turn that says it must not be written down again as `voice`, or
                # the daemon speaking resets its own silence clock
                # (daemon/memory/store.py).
                opening_already_logged=True,
            )
        except Exception:
            # Never raises into the wake loop. A round is the microphone, so a
            # failed line costs the utterance its voice and nothing else.
            logger.exception("wake: the session opened to speak first failed")

        if not spoke:
            # Nothing reached the room - no voice configured, a socket that never
            # opened, a session that generated an answer and interrupted itself
            # before playing any of it. `/usr/bin/say` is the fallback rather than
            # the first choice because it is a different engine in a different voice
            # from every answer the owner has ever heard, which is the whole reason
            # this path changed. Guarded on `spoke` and not on the exception: a
            # session that played audio and *then* failed has already said the line,
            # and saying it again is worse than not saying it at all.
            try:
                from daemon.proactivity.speaker import LocalSpeaker

                speaker = LocalSpeaker()
                try:
                    spoke = await speaker.say(text)
                finally:
                    await speaker.aclose()
            except Exception:
                logger.exception("wake: could not speak a proactive line")
    finally:
        # Owed on *every* path, which is why this is a `finally` and not a statement
        # after an `except Exception` that cannot catch a `CancelledError` (PR #115;
        # `daemon/mic_floor.py` says so in three places and this is the code those
        # sentences are about). A no-op when `note_spoke` already answered it -
        # `answer` checks. `request` is sitting on this future, and dropping it
        # turns its fallback into a two-and-a-half-minute stall.
        mic_floor.answer(future, spoke)

    # Same handover the wake path takes, for the same reason: let the session's
    # Voice-Processing unit finish releasing the microphone before the next round
    # opens a fresh capture on it. See WAKE_REARM_SETTLE_SECONDS. Unconditional
    # because a session is now opened on every path through this function, even the
    # ones where it never gets as far as the device.
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


async def _wake_forever(
    settings: Settings, shared: VoiceRuntime | None = None, state: Any = None
) -> None:
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
            await _wake_round(settings, shared, state)
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
) -> tuple[WakeGate, Callable[[], Awaitable[bool]]]:
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

    async def close() -> bool:
        """Release the device and wait for it to actually be gone. True if it is.

        Two halves, and the second is the one the handover needs. `audio.close()`
        is the speaker; the microphone is let go by `record`'s `finally`, which hands
        the stream to a detached thread and returns - so without the wait this
        returns while `Pa_StopStream` is still inside CoreAudio. Anything that then
        opens a second client on the same device races it, and the two deadlock
        (daemon/voice/audio.py:wait_for_input_release). The caller must have closed
        the gate's stream before calling this, or there is nothing to wait on yet.
        """
        with suppress(Exception):
            await audio.close()
        # Not suppressed, unlike the speaker close above: this one's answer is the
        # whole point of the call, and an error swallowed here reads as "released"
        # (see `_wake_round`, which fails closed on it).
        released = await audio.wait_for_input_release()
        if not released:
            logger.error(
                "wake: the microphone did not come back after the gate let it go; "
                "the capture device is wedged inside CoreAudio and nothing in this "
                "process can free it - a restart is the only fix"
            )
        return released

    return gate, close


async def _build_tools(
    settings: Settings,
    store: Any,
    *,
    mode: str | None = None,
    screen_share: Any = None,
    delegate_wake: asyncio.Event | None = None,
    channel: Any = None,
    face: FaceBus | None = None,
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

    `delegate_wake`, likewise, is only ever passed by `_build_voice_runtime` - the
    resident's boot-once voice tool layer. `None` (the default, and what the text
    path and `run_voice`'s own call pass) means `delegate_task` is never
    registered: it is a voice-only tool, since it exists for work the native-audio
    model cannot do itself.

    `channel` is the same voice-only story, for `send_message`: on the text path the
    reply already reaches the channel, so a send tool there would only produce a
    second copy of it. `None` - the text path, a standalone `run_voice`, or a
    channel that failed to build - means the tool is not registered at all, because
    a tool that cannot deliver is worse than a missing one: the audio model reports
    the send either way.
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

    if delegate_wake is not None:
        # Voice-only: a spoken turn has no direct path to a nested-schema tool
        # (evals/voice_write_nudge_spike.py), so this is the one flat-schema tool
        # that hands such work to the background worker in `daemon/delegation.py`.
        from daemon.tools.delegate import DelegateTask

        registry.register(
            DelegateTask(
                enqueue=lambda request: store.enqueue_task(
                    request=request, origin="owner", channel="voice", sender_id=None
                ),
                notify=delegate_wake.set,
            )
        )

    if channel is not None:
        # Voice-only, same as `delegate_task` above: this is how a spoken turn puts
        # a link or a name in writing where the owner can keep it.
        from daemon.tools.message import SendMessage

        registry.register(
            SendMessage(
                channel,
                # The paired owner, read per call rather than captured at boot: an
                # install can be paired after the daemon starts, and a tool holding
                # the boot-time answer would stay unable to address anyone.
                recipient=lambda: store.owner_id(channel.name),
            )
        )

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
    runner = ToolRunner(registry, policy, store, face=face)
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
