"""`daemon/app.py`'s lip-sync assembly: the switch, the loop, and the two seams.

`daemon/face_lipsync/*` is tested against a fake engine and, by hand, against
MuseTalk's own pipeline. None of that says the daemon *runs* it, and the shape of
every wiring defect this repo has shipped is exactly that gap - built, tested,
unreachable. So what is asserted here is only what assembly owns:

  * the three ways lip-sync is absent, all of which must leave the face playing
    its pre-rendered clips rather than crash a process;
  * the render loop's decisions - which clock it reads, when it renders at all,
    when it stops - none of which are visible from inside `Renderer`;
  * that the PCM sink reaches `VoiceConversation`, because a ring nothing feeds
    renders a mouth conditioned on silence and never says so.

**No model and no weights.** `mlx` has no Linux wheel and CI is ubuntu, so
anything here that touched the engine would either skip in CI or fail there. The
loop is driven with fakes for the same reason `daemon/face_lipsync` has an engine
protocol at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from daemon import app as app_module
from daemon.app import _build_lipsync, _lipsync_loop, _LipsyncFrames, create_app
from daemon.config import Settings
from daemon.face import FaceBus

FPS = 24.0

CACHE_FILES = (
    "models/unet.safetensors",
    "models/musetalk.json",
    "models/taesd.safetensors",
    "idle2/latents.safetensors",
    "idle2/frames.npy",
    "idle2/masks.npz",
    "idle2/boxes.json",
)
"""Everything `_build_lipsync` requires before it will try to load anything. Listed
here rather than derived so that a file being dropped from the requirement is a test
change somebody has to make deliberately."""


def _settings(tmp_path: Path, **kw) -> Settings:
    """`provider="ollama"` needs no key and `_env_file=None` keeps the worktree's own
    `.env` out - the same base `tests/test_face_routes.py` uses."""
    kw.setdefault("provider", "ollama")
    return Settings(_env_file=None, data_dir=tmp_path, **kw)


def _lay_out(tmp_path: Path, *, skip: str = "") -> Path:
    """Files with the right names and the wrong contents.

    Enough to get past the presence check and no further, which is the only thing
    these degradation tests need: a cache that loads would need a gigabyte of real
    frames and an engine that ran would need weights CI must never hold.
    """
    root = tmp_path / "face" / "lipsync"
    for name in CACHE_FILES:
        if name == skip:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really a model")
    return root


# --- the three absences ------------------------------------------------------


def test_the_default_install_builds_nothing_and_says_nothing(tmp_path, caplog):
    """Off is the default, and an install that never turns it on must pay nothing -
    no import of a 1.7GB weight loader, and no log line about a feature nobody asked
    for."""
    with caplog.at_level(logging.DEBUG, logger="daemon.app"):
        assert _build_lipsync(_settings(tmp_path), FaceBus()) is None
    assert not caplog.records


def test_the_switch_on_with_nothing_fetched_degrades_in_one_line(tmp_path, caplog):
    """The install that would otherwise look broken: the switch says yes and there is
    no mouth. One line, naming a path and the command that produces the rest - not
    seven lines for seven absent files, and not a traceback."""
    settings = _settings(tmp_path, face_lipsync_enabled=True)
    with caplog.at_level(logging.INFO, logger="daemon.app"):
        assert _build_lipsync(settings, FaceBus()) is None
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "unet.safetensors" in message
    assert "face_lipsync_prepare" in message, "the reader has to be told what builds a cache"


def test_a_missing_clip_cache_is_not_the_same_as_missing_weights(tmp_path, caplog):
    """Weights fetched, no clip prepared. Still a degradation, and the line has to
    name the file that is actually absent - the owner who has already downloaded
    1.7GB should not be told to download it again."""
    _lay_out(tmp_path, skip="idle2/frames.npy")
    settings = _settings(tmp_path, face_lipsync_enabled=True)
    with caplog.at_level(logging.INFO, logger="daemon.app"):
        assert _build_lipsync(settings, FaceBus()) is None
    assert "frames.npy" in caplog.records[0].getMessage()


def test_a_broken_cache_costs_the_mouth_and_not_the_process(tmp_path, caplog):
    """Every file present and none of them loadable - a truncated download, a
    half-written cache, or the `face` extra missing so `mlx` will not import.

    All of it has to land here: one logged exception and `None`. Above the try these
    imports were an ImportError out of the lifespan, which is a daemon that will not
    start because of a face.
    """
    _lay_out(tmp_path)
    settings = _settings(tmp_path, face_lipsync_enabled=True)
    with caplog.at_level(logging.INFO, logger="daemon.app"):
        assert _build_lipsync(settings, FaceBus()) is None
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert caplog.records[0].exc_info is not None, "the traceback is the point of once"


def test_the_assembled_app_starts_with_no_mouth_and_no_sink(tmp_path):
    """`create_app` declares both as absent rather than leaving them to `getattr`.

    Two readers depend on it - the transport and `_wake_round` - and a test that
    builds the app without a lifespan is the case that used to see neither.
    """
    app = create_app(_settings(tmp_path))
    assert app.state.face_frames is None
    assert app.state.face_pcm_sink is None


# --- the render loop --------------------------------------------------------


class FakeRenderer:
    """Records what it was asked to render. `failed` is settable, like the real
    latch."""

    def __init__(self) -> None:
        self.failed = False
        self.calls: list[tuple[int, float, float]] = []
        self.holding = False
        """Alternates, like the real one: a model step leaves the pair's second frame
        held, and the next call publishes it and does no work. The loop paces off this
        rather than off how long the call took, so a fake that always reported False
        would let the loop release both frames of a pair inside one tick again."""

    def render(self, *, frame_index: int, origin: float, fps: float) -> None:
        self.calls.append((frame_index, origin, fps))
        self.holding = not self.holding


class FakeClock:
    """`FrameClock` with the arithmetic replaced by a script, so a test can say
    "this tick has a frame and that one does not" without simulating audio."""

    def __init__(self, answers: list[int | None]) -> None:
        self.answers = answers
        self.asked: list[tuple[float, float]] = []

    def due(self, *, now: float, origin: float) -> int | None:
        self.asked.append((now, origin))
        return self.answers.pop(0) if self.answers else None


class FakeRing:
    """Records feeds and reports a fixed origin. The real ring's arithmetic is
    `tests/test_face_lipsync_ring.py`'s subject, not this file's."""

    def __init__(self, origin: float = 100.0) -> None:
        self.origin = origin
        self.fed: list[tuple[bytes, float]] = []

    def feed(self, chunk: bytes, audible_at: float) -> None:
        self.fed.append((chunk, audible_at))


async def _run_loop(face, renderer, clock, ring, queued, *, stop_after: float):
    """Drive `_lipsync_loop` for `stop_after` seconds, then cancel it."""
    task = asyncio.create_task(
        _lipsync_loop(face, renderer, clock, ring, queued, fps=FPS)
    )
    await asyncio.sleep(stop_after)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_an_idle_face_never_reaches_the_model(monkeypatch):
    """The duty cycle is the throughput budget. Measured, the engine holds 24fps on a
    real conversation and does not on a continuous run, so a loop that rendered while
    nothing was being said would be spending the headroom the speaking half needs -
    and pinning the GPU of a machine that is doing nothing.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    renderer, clock, ring = FakeRenderer(), FakeClock([0] * 20), FakeRing()
    await _run_loop(FaceBus(), renderer, clock, ring, deque(), stop_after=0.3)
    assert not renderer.calls
    assert not clock.asked, "an idle tick must not even ask which frame is due"


async def test_the_loop_renders_the_newest_frame_whose_audio_has_arrived(monkeypatch):
    """One render per tick, at the audio front - not the next index in sequence.

    `FrameClock.due` never skips, on purpose, and against a hitch that is right. This
    build has a structural deficit instead: a two-frame step measures 85.9ms in situ
    against the 83.3ms two frames are worth, so consuming one grant per tick drifted
    the mouth 1.4s behind over 9s of speech (measured). So the loop drains the grants
    it is behind on and renders the newest - here 0 and 1 are dropped and 2 is drawn.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([0, 1, 2, None, 7, None]), FakeRing()
    await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.3)
    # Each model step is followed one interval later by the release of the pair's
    # second frame, at the same index - so a step at 2 and a step at 7 read as
    # [2, 2, 7, 7]. The releases cost no model work; they are what makes the socket
    # see one frame per interval instead of two 10ms apart.
    assert [call[0] for call in renderer.calls] == [2, 2, 7, 7]
    assert all(call[1] == ring.origin for call in renderer.calls)
    assert all(call[2] == FPS for call in renderer.calls)


async def test_a_tick_whose_audio_has_not_arrived_renders_nothing(monkeypatch):
    """The batch-fill wait, which is most of the ticks at the start of a turn: a step
    covers two frames and cannot begin until the second one's audio is here. Rendering
    anyway would not fail - `PcmRing.window` zero-fills - it would silently condition
    the mouth on silence."""
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([None, None, 4]), FakeRing()
    await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.2)
    # Two calls, not one, and the second is not a second step: a model step leaves the
    # pair's second frame held, and the loop releases it one interval later at the same
    # index. Asserting a single call here is what let both frames of a pair go out
    # 10ms apart.
    assert [call[0] for call in renderer.calls] == [4, 4]


async def test_the_clock_is_asked_in_the_event_loop_s_own_time(monkeypatch):
    """`now` and `origin` have to come from `loop.time()`.

    `daemon/voice/conversation.py` stamps every chunk with it deliberately, and
    `daemon.clock.now()` here type-checks, runs, and puts the mouth an arbitrary
    offset from the sound. A monotonic clock reads as seconds-since-boot and a wall
    clock as seconds-since-1970, so the two are told apart by nine orders of
    magnitude - this asserts against the loop's own value rather than against a
    magnitude, but that gap is why the mistake is silent.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([0]), FakeRing()
    await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.1)
    loop_now = asyncio.get_running_loop().time()
    asked_now, asked_origin = clock.asked[0]
    assert abs(asked_now - loop_now) < 1.0
    assert asked_origin == ring.origin


def test_the_sink_is_callable_the_way_the_speech_clock_calls_it():
    """The whole feature, on one argument list, and the unit suite could not see it.

    `SpeechClock.fed` calls its sink with `(chunk, audible_at)`. This started life as
    `deque.append`, which takes exactly one argument, so the first chunk of the first
    spoken turn raised `TypeError` inside the audio path and the mouth never moved
    once - found by running it, not by any test here. Every assertion about the sink
    was about *identity*: which object got passed where, never whether it could be
    called. So this drives the real clock into the real sink, and the real clock is
    what makes it a contract rather than a signature I chose twice.

    `audible_at` is the second half of it. The clock's whole reason for owning this
    seam is that it knows when a chunk will be *heard*, not when it was queued - so
    the value that arrives has to be that instant, and for the first chunk of a turn
    that is the arrival time itself.
    """
    from daemon.app import _queue_pcm
    from daemon.face import SpeechClock

    queued: deque[tuple[bytes, float]] = deque()
    clock = SpeechClock(
        FaceBus(),
        sample_rate=24_000,
        bytes_per_frame=2,
        pcm_sink=partial(_queue_pcm, queued),
    )
    # 4800 bytes = 2400 samples = 100ms, the size a live session hands over.
    clock.fed(b"\x01\x02" * 2400, 500.0)
    clock.fed(b"\x03\x04" * 2400, 500.01)
    assert [at for _chunk, at in queued] == [500.0, 500.1], (
        "the second chunk is queued behind the first, so it is audible 100ms later - "
        "not at the moment it arrived"
    )
    assert [len(chunk) for chunk, _at in queued] == [4800, 4800]


async def test_queued_audio_reaches_the_ring_on_the_loop_that_renders(monkeypatch):
    """The sink queues and the loop drains, which is what keeps the ring
    single-threaded.

    `PcmRing` has no lock: `feed` rebinds its sample array and *then* advances its
    origin, so a `window()` landing between those two statements reads new samples
    against a stale origin - a window off by one chunk, which is a mouth off by one
    chunk. The render runs in a thread, so the fix is that only this loop ever
    touches the ring, and only while no render is in flight.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    renderer, clock, ring = FakeRenderer(), FakeClock([]), FakeRing()
    queued: deque[tuple[bytes, float]] = deque([(b"\x01\x02", 1.0), (b"\x03\x04", 2.0)])
    # Idle on purpose: audio arrives before `SpeechClock.pump` has published
    # `speaking`, so a loop that only drained while speaking would hold the first
    # chunk of every single turn.
    await _run_loop(FaceBus(), renderer, clock, ring, queued, stop_after=0.15)
    assert ring.fed == [(b"\x01\x02", 1.0), (b"\x03\x04", 2.0)]
    assert not queued


async def test_the_end_of_speech_hands_back_the_gpu_cache(monkeypatch):
    """Spec section 7 calls this a requirement, not an optimisation: without it the
    spike's MLX cache grew until frame drift hit +41.2%. Once per falling edge, not
    once per idle tick - a clear on every tick of a quiet night is work for nothing.
    """
    released = []
    monkeypatch.setattr(
        app_module, "release_lipsync_memory", lambda: released.append(True)
    )
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([0]), FakeRing()
    task = asyncio.create_task(
        _lipsync_loop(face, renderer, clock, ring, deque(), fps=FPS)
    )
    await asyncio.sleep(0.1)
    assert not released, "nothing to release while it is still speaking"
    face.set_activity("idle")
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == [True], "once on the falling edge, not once a tick"


async def test_a_latched_failure_ends_the_loop(monkeypatch, caplog):
    """`Renderer.failed` is a one-way door: it has already logged its exception once,
    and a loop that kept ticking would ask a dead engine 24 times a second forever."""
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([0] * 10), FakeRing()
    renderer.failed = True
    with caplog.at_level(logging.WARNING, logger="daemon.app"):
        await asyncio.wait_for(
            _lipsync_loop(face, renderer, clock, ring, deque(), fps=FPS), 1.0
        )
    assert not renderer.calls
    assert "back on its clips" in caplog.text


async def test_a_raising_tick_is_logged_rather_than_silently_orphaned(monkeypatch):
    """The failure this project's own brief names: a background task that raises is
    logged by nobody and the schedule reads as healthy forever. `Renderer.render`
    swallows an engine failure into `failed`; everything else in a tick lands here.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)

    class Exploding(FakeRing):
        def feed(self, chunk: bytes, audible_at: float) -> None:
            raise ValueError("malformed chunk")

    renderer, clock = FakeRenderer(), FakeClock([])
    queued = deque([(b"\x01\x02", 1.0)])
    # No `pytest.raises`: a loop that let this out would take the task with it, and
    # `wait_for` returning is the assertion.
    await asyncio.wait_for(
        _lipsync_loop(FaceBus(), renderer, clock, Exploding(), queued, fps=FPS), 1.0
    )


# --- the two seams ----------------------------------------------------------


def test_the_frames_source_is_exactly_what_the_transport_asks_for():
    """`face_routes.LipsyncFrames` is four members, and they live on two objects that
    are deliberately kept apart - `Slot` is `put`/`get` and knows nothing about
    geometry, `Renderer` knows the geometry and nothing about who reads it. Joining
    them is assembly, so this is the join.
    """
    from daemon.face_lipsync.ring import Slot

    renderer, slot = FakeRenderer(), Slot()
    frames = _LipsyncFrames(renderer=renderer, slot=slot, clip="idle2")
    assert frames.clip == "idle2"
    assert not hasattr(frames, "box"), (
        "a box here would mean the page is placing a crop again"
    )
    assert frames.get() is None
    slot.put(b"jpeg")
    assert frames.get() == b"jpeg"
    assert frames.failed is False
    renderer.failed = True
    assert frames.failed is True, "the latch has to be read through, not copied at boot"


def test_the_cache_loader_reads_what_the_prepare_tool_writes(tmp_path):
    """The one contract between an offline tool and the runtime, at a size CI can
    hold: three frames instead of 193.

    The masks are the part worth asserting. `masks.npz` is a pile of
    differently-shaped arrays because MuseTalk derives each frame's blend region from
    that frame's own face box (608 to 726px square across `idle2`), so a loader that
    stacked them would raise and a loader that sorted them wrongly would paste last
    frame's jaw onto this frame's face.
    """
    from daemon.app import _load_lipsync_cache

    out = tmp_path / "idle2"
    out.mkdir()
    np.save(out / "frames.npy", np.zeros((3, 8, 6, 3), dtype=np.uint8))
    np.savez(
        out / "masks.npz",
        **{f"{i:06d}": np.full((4 + i, 4 + i), i, dtype=np.uint8) for i in range(3)},
    )
    (out / "boxes.json").write_text(
        json.dumps(
            {
                "fps": 24.0,
                "size": [6, 8],
                "boxes": [[1, 1, 3, 3]] * 3,
                "crop_boxes": [[0, 0, 4 + i, 4 + i] for i in range(3)],
            }
        ),
        encoding="utf-8",
    )

    cache, fps = _load_lipsync_cache(out)
    assert fps == 24.0
    assert cache.frames.shape == (3, 8, 6, 3)
    assert cache.boxes == [(1, 1, 3, 3)] * 3
    assert cache.crop_boxes == [(0, 0, 4, 4), (0, 0, 5, 5), (0, 0, 6, 6)]
    assert [mask.shape for mask in cache.masks] == [(4, 4), (5, 5), (6, 6)]
    assert [int(mask[0, 0]) for mask in cache.masks] == [0, 1, 2], "in frame order"


async def test_a_wake_round_hands_the_process_s_own_sink_to_the_voice_call(monkeypatch):
    """The first link of the same chain, and it had no test at all.

    `_wake_round` is what the resident's gate calls, and it is where `app.state` meets
    `run_voice`. Nothing under `tests/` drove it before this, so a sink dropped here
    would have left every assertion green and every spoken turn mouthless - the same
    defect as the `deque.append` one, one signature earlier.

    Both handles are read off `state` at call time rather than captured when the gate
    came up, so the switch is a property of the process and not of the moment the wake
    loop started.
    """
    from types import SimpleNamespace

    from daemon import app as module

    captured: dict[str, object] = {}

    async def fake_run_voice(settings, **kwargs):
        captured.update(kwargs)
        return 0

    class FakeGate:
        async def listen(self):
            yield SimpleNamespace(heard="\ubca8\ub77c", matched="\ubca8\ub77c", pcm=b"\x00\x01")

    async def fake_build(settings):
        async def close() -> None:
            return None

        return FakeGate(), close

    monkeypatch.setattr(module, "run_voice", fake_run_voice)
    monkeypatch.setattr(module, "build_wake_gate", fake_build)
    monkeypatch.setattr(module, "WAKE_REARM_SETTLE_SECONDS", 0.0)

    bus = FaceBus()

    def sink(chunk: bytes, at: float) -> None:
        return None

    state = SimpleNamespace(face=bus, face_pcm_sink=sink, wake_gate=None)
    await module._wake_round(_settings(Path("/tmp")), state=state)

    assert captured["pcm_sink"] is sink
    assert captured["face"] is bus


async def test_the_voice_path_carries_the_sink_to_the_conversation(monkeypatch):
    """The seam that makes the whole feature work, and the one nothing else can see.

    `SpeechClock` is the only place that computes when a chunk becomes *audible*
    rather than when it was queued, so the ring has to be fed from there or the mouth
    is dubbed. Between `app.state` and that clock sit two function signatures, and a
    sink dropped at either of them leaves a renderer that runs, publishes frames
    conditioned on silence, and reports nothing wrong.
    """
    from daemon.app import _voice_attempts

    captured: dict[str, object] = {}

    class FakeConversation:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)
            self.ended = "test"
            self.stats = type(
                "Stats", (), {"played_seconds": 1.0, "describe": lambda self: "ok"}
            )()

        async def run(self) -> None:
            return None

    import daemon.voice.conversation as conversation_module

    monkeypatch.setattr(conversation_module, "VoiceConversation", FakeConversation)

    def sink(chunk: bytes, at: float) -> None:
        return None

    face = FaceBus()
    code = await _voice_attempts(
        lambda: object(), object(), object(), RuntimeError, face=face, pcm_sink=sink
    )
    assert code == 0
    assert captured["pcm_sink"] is sink
    assert captured["face"] is face
