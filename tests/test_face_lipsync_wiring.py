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
import time
from collections import deque
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from daemon import app as app_module
from daemon.app import _build_lipsync, _lipsync_loop, _LipsyncFrames, create_app
from daemon.config import Settings
from daemon.face import FaceBus
from daemon.face_lipsync.render import RELEASE_FRAMES, ClipClock

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
    # And the mp4 the cache was prepared from. `_prepared_clips` requires both, so that
    # the set lip-sync can drive equals the set `/face/clips/{name}` can serve - the
    # page falls back to that file when the renderer latches `failed`. Without it a
    # cache directory is skipped rather than loaded, which is a different degradation
    # from the one each of these tests is about.
    (tmp_path / "face" / "idle2.mp4").write_bytes(b"not really a clip")
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
    """Weights fetched, no clip prepared. Still a degradation, and the line has to send
    the owner who already downloaded 1.7GB to a command rather than back to a download.

    It no longer names one absent file, and that is the multi-clip change rather than a
    weakening: with ten possible caches under one directory there is no single missing
    path to name, so the line says none of them is complete and gives the command that
    completes one. The distinction this test exists for is intact - a reader can still
    tell this from the weights line, which is what `skip` used to prove by filename.
    """
    _lay_out(tmp_path, skip="idle2/frames.npy")
    settings = _settings(tmp_path, face_lipsync_enabled=True)
    with caplog.at_level(logging.INFO, logger="daemon.app"):
        assert _build_lipsync(settings, FaceBus()) is None
    message = caplog.records[0].getMessage()
    assert "no clip" in message, "the absence has to be the clips, not the weights"
    assert "unet.safetensors" not in message, "the weights are present; do not send him back"
    assert "face_lipsync_prepare" in message, "and it has to name what builds one"


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
    """Two halves, like the real one, and it publishes nothing.

    `step` runs on a worker thread and returns an opaque token; `encode` turns that
    token into two identifiable byte strings. What a test then asserts on is which of
    those reached the `Slot` and when - the pacing that used to be inside `Renderer`
    is the loop's now, so it has to be visible from out here.
    """

    def switch(self, driver) -> None:
        """Recorded when the test prepared more than one clip, and a failure otherwise.

        With one prepared clip the policy has nothing to switch to, so a call there means
        the loop moved a clip the test never asked it to - worth failing on rather than
        absorbing.
        """
        if not self.allow_switch:
            raise AssertionError(
                f"the loop switched to {driver.name!r} with one clip prepared"
            )
        self.switched.append(driver.name)

    def __init__(self, *, step_seconds: float = 0.0, allow_switch: bool = False) -> None:
        self.allow_switch = allow_switch
        self.switched: list[str] = []
        self.failed = False
        self.calls: list[tuple[int, float, float]] = []
        self.encoded: list[int] = []
        self.in_flight = False
        """True while `step` is on the worker thread. `FakeRing` asserts against it:
        the real ring has no lock, and the loop's whole reason for awaiting the step
        is that it must never feed one while a step is reading it."""
        self._step_seconds = step_seconds
        """Blocking time per step, for the tests about overlap. A real one is ~73ms."""
        self.released: list[tuple[int, int]] = []
        """Every `release` call, so a test can see the ramp's length and its indices."""
        self.releases = 0
        """How many ramp frames this fake will answer with before returning `None`."""
        self.label = "a"
        """Stamped into every encoded frame. A test that flips this between two turns
        can tell which turn a frame in the slot came from, which frame indices cannot
        do - the real clock restarts its count at every turn boundary."""

    def step(self, *, frame_index: int, origin: float, fps: float) -> int | None:
        self.in_flight = True
        try:
            if self._step_seconds:
                time.sleep(self._step_seconds)
            self.calls.append((frame_index, origin, fps))
            return frame_index
        finally:
            self.in_flight = False

    def encode(self, step: int) -> list[bytes]:
        self.encoded.append(step)
        return [f"{self.label}{step}-0".encode(), f"{self.label}{step}-1".encode()]

    def release(self, *, index: int, step: int) -> bytes | None:
        """The falling edge's ramp. `None` past `self.releases`, which is how the real
        one answers once its held mouth is spent - the loop then falls through to the
        clip."""
        self.released.append((index, step))
        if step > self.releases:
            return None
        return f"{self.label}r{step}".encode()


class FakeClock:
    """`FrameClock` with the arithmetic replaced by a script, so a test can say
    "this pass has a frame and that one does not" without simulating audio."""

    def __init__(self, answers: list[int | None]) -> None:
        self.answers = answers
        self.asked: list[tuple[float, float]] = []

    def due(self, *, now: float, origin: float) -> int | None:
        self.asked.append((now, origin))
        return self.answers.pop(0) if self.answers else None


class PacedClock:
    """`FrameClock`'s pacing without its windowing: the frame whose time has come.

    `FakeClock`'s scripted list cannot express "and keep going", which is exactly what
    the throughput tests need - the real clock grants at `fps` off the playback
    timeline and that rate is what the loop is paced by.
    """

    def __init__(self, *, fps: float) -> None:
        self._fps = fps
        self._frame = 0
        self._began: float | None = None

    def due(self, *, now: float, origin: float) -> int | None:
        if self._began is None:
            self._began = now
        if (now - self._began) * self._fps < self._frame:
            return None
        frame = self._frame
        self._frame += 1
        return frame


class FakeRing:
    """Records feeds and reports a fixed origin. The real ring's arithmetic is
    `tests/test_face_lipsync_ring.py`'s subject, not this file's."""

    def __init__(self, origin: float = 100.0, renderer: FakeRenderer | None = None) -> None:
        self.origin = origin
        self.fed: list[tuple[bytes, float]] = []
        self._renderer = renderer
        """Given one, every feed asserts no step is in flight - see FakeRenderer."""

    def feed(self, chunk: bytes, audible_at: float) -> None:
        if self._renderer is not None and self._renderer.in_flight:
            raise AssertionError(
                "the ring was fed while a model step was reading it - PcmRing has no "
                "lock, and `feed` rebinds its samples before it advances its origin"
            )
        self.fed.append((chunk, audible_at))


class RecordingSlot:
    """`Slot` plus the moment of each put, which is what these tests measure.

    The real `Slot` is latest-wins and never queues, so a put that lands inside the
    same tick as the one before it is a frame nothing ever sees. Timing it is the only
    way to assert that from out here.
    """

    def __init__(self) -> None:
        self.puts: list[bytes] = []
        self.at: list[float] = []

    def put(self, frame: bytes) -> None:
        self.puts.append(frame)
        self.at.append(asyncio.get_running_loop().time())

    @property
    def gaps(self) -> list[float]:
        return [b - a for a, b in zip(self.at, self.at[1:], strict=False)]


IDLE_FRAMES = [f"idle-{i}".encode() for i in range(6)]
"""Stand-ins for the clip's own frames, distinguishable from a rendered one so a test
can tell which half of the publisher produced what."""


def _driving(frames=None):
    """The single-clip world these tests were written in, in the shape the loop now takes.

    Every test in this file is about the *pacing* of the two halves - which frame reached
    the slot and when - not about which clip is up, so one prepared clip is the right
    harness. `_Driving`/`caches`/`lengths` is what `_lipsync_loop` reads now, in place of
    the bare `ClipClock` and frame list it used to take, and with a single entry the
    clip policy has nothing to switch to: `wanted` answers with the clip already up and
    `ClipQueue` drops that want, so these tests see the same loop they always did.
    """
    from types import SimpleNamespace

    from daemon.app import _Driving
    from daemon.face_lipsync.render import Driver

    frames = IDLE_FRAMES if frames is None else frames
    cache = SimpleNamespace(boxes=list(range(len(frames))), frames=list(frames))
    driver = Driver(
        name="idle2",
        cache=cache,
        clip=ClipClock(fps=FPS, frames=len(frames), epoch=0.0),
    )
    return (
        _Driving(driver=driver, frames=list(frames)),
        {"idle2": (cache, FPS)},
        {"idle2": len(frames) / FPS},
    )


def _driving_two():
    """Two prepared clips, which is the world every other test in this file is not.

    They are all single-clip, and that is exactly how a name collision reached a live
    run: `prefetch` reads `stem in encoded or stem in pending_stills`, and with one clip
    the first half short-circuits true so the second is never evaluated. It was bound to
    a `Future` by the publish loop's own local of the same name, and `in` on a Future
    raises `RuntimeError: await wasn't used with future` - which took the render loop
    down on the first spoken turn with the whole suite green.

    Short clips (4 frames, 1/6th of a second) so a boundary arrives inside a test.
    """
    from types import SimpleNamespace

    from daemon.app import _Driving
    from daemon.face_lipsync.render import Driver

    def one(name):
        frames = [f"{name}-{i}".encode() for i in range(4)]
        cache = SimpleNamespace(boxes=list(range(4)), frames=list(frames))
        return cache, frames

    cache2, frames2 = one("idle2")
    cache1, _frames1 = one("idle1")
    cache_mood, _frames_mood = one("amused")
    driver = Driver(
        name="idle2", cache=cache2, clip=ClipClock(fps=FPS, frames=4, epoch=0.0)
    )
    return (
        _Driving(driver=driver, frames=frames2),
        {"idle2": (cache2, FPS), "idle1": (cache1, FPS), "amused": (cache_mood, FPS)},
        {"idle2": 4 / FPS, "idle1": 4 / FPS, "amused": 4 / FPS},
    )


async def test_a_mood_queued_while_speaking_reaches_the_boundary(monkeypatch):
    """Speaking AND a clip whose stills do not exist yet - the one combination that
    reached a live run, and the one no other test here can make.

    Every other test prepares a single clip, so `prefetch`'s
    `stem in encoded or stem in pending_stills` short-circuits on the first half and the
    second is never evaluated. Reaching it needs a clip that is wanted but not yet
    encoded, which during speech only a mood one-shot produces - `wanted("speaking")`
    answers with the clip already up. And the failure needed the model half to have run
    first, because that is what bound a `Future` to the name this dict used to share:
    `in` on a Future raises `RuntimeError: await wasn't used with future`, which took the
    render loop down on the first spoken turn with the whole suite green.
    """
    import daemon.face_lipsync.render as render_module

    monkeypatch.setattr(
        render_module, "encode_clip", lambda cache: list(cache.frames)
    )
    face = FaceBus()
    face.set_activity("speaking")
    renderer = FakeRenderer(allow_switch=True)
    slot = RecordingSlot()
    task = asyncio.create_task(
        _lipsync_loop(
            face, renderer, PacedClock(fps=FPS), FakeRing(), deque(), slot,
            *_driving_two(), fps=FPS,
        )
    )
    await asyncio.sleep(0.12)          # let the model half run and rebind its own local
    assert renderer.encoded, "the model half never ran, so the collision cannot appear"
    face.one_shot("amused")            # a clip with a cache and no stills yet
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not renderer.failed, "the loop latched failed rather than switching"
    assert "amused" in renderer.switched, (
        f"the mood never reached the renderer - switched to {renderer.switched}"
    )


async def _run_loop(face, renderer, clock, ring, queued, *, stop_after: float, slot=None):
    """Drive `_lipsync_loop` for `stop_after` seconds, then cancel it."""
    slot = RecordingSlot() if slot is None else slot
    task = asyncio.create_task(
        _lipsync_loop(
            face, renderer, clock, ring, queued, slot, *_driving(), fps=FPS
        )
    )
    await asyncio.sleep(stop_after)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return slot


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


async def test_the_loop_steps_the_newest_frame_whose_audio_has_arrived(monkeypatch):
    """One step per pair, at the audio front - not the next index in sequence.

    `FrameClock.due` never skips, on purpose, and against a hitch that is right. A
    hitch still has to be caught up on though, or the mouth stays behind for the rest
    of the utterance: measured on the build whose step overran its budget, consuming
    one grant per tick drifted the mouth 1.4s behind over 9s of speech. So the loop
    drains the grants it is behind on and steps at the newest - here 0 and 1 are
    dropped and 2 is drawn.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([0, 1, 2, None, 7, None]), FakeRing()
    slot = await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.3)
    # One call per pair now, not two: the pair's second frame is a queued JPEG the
    # publisher releases an interval later, not a second trip through the renderer.
    assert [call[0] for call in renderer.calls] == [2, 7]
    assert all(call[1] == ring.origin for call in renderer.calls)
    assert all(call[2] == FPS for call in renderer.calls)
    # And both frames of each pair reach the slot, in order.
    assert slot.puts == [b"a2-0", b"a2-1", b"a7-0", b"a7-1"]


async def test_a_pass_whose_audio_has_not_arrived_renders_nothing(monkeypatch):
    """The batch-fill wait, which is most of the passes at the start of a turn: a step
    covers two frames and cannot begin until the second one's audio is here. Stepping
    anyway would not fail - `PcmRing.window` zero-fills - it would silently condition
    the mouth on silence."""
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, clock, ring = FakeRenderer(), FakeClock([None, None, 4]), FakeRing()
    slot = await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.2)
    assert [call[0] for call in renderer.calls] == [4]
    assert slot.puts == [b"a4-0", b"a4-1"], "the pair still goes out, one frame apart"


async def test_the_pair_goes_out_one_frame_apart_and_never_two_in_one_tick(monkeypatch):
    """The defect that made 20fps read as 10Hz, asserted at the slot.

    `Slot` is latest-wins, so two puts closer together than the transport's 5ms poll
    means the first is never seen - measured on the build where a step published its
    own pair, 85 published and 38 of them lost. The owner read it exactly as it looks:
    a head moving smoothly over a mouth that stutters. One frame per interval is the
    whole point of a frame rate.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, ring = FakeRenderer(), FakeRing()
    slot = await _run_loop(
        face, renderer, PacedClock(fps=FPS), ring, deque(), stop_after=0.6
    )
    assert len(slot.puts) >= 8, f"only {len(slot.puts)} frames in 0.6s"
    interval = 1.0 / FPS
    assert min(slot.gaps) > interval / 2, (
        f"two frames landed {min(slot.gaps) * 1000:.1f}ms apart - closer than half a "
        f"frame is a frame nothing will ever see: {[round(g * 1000, 1) for g in slot.gaps]}"
    )
    assert len(set(slot.puts)) == len(slot.puts), "a frame was published twice"


async def test_the_next_model_step_runs_while_the_pair_is_still_going_out(monkeypatch):
    """The throughput fix, and the only test here that can see it.

    A step is ~73ms of model work for two frames worth 83.3ms, so a loop that had to
    finish publishing before it could step again spent the publishing time idle and
    the socket saw 20.1fps. Here the step blocks for 60ms - longer than a frame - and
    the assertion is that frames keep landing at the frame rate anyway, which is only
    possible if the publisher runs while the step does.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer = FakeRenderer(step_seconds=0.060)
    slot = await _run_loop(
        face, renderer, PacedClock(fps=FPS), FakeRing(), deque(), stop_after=0.6
    )
    interval = 1.0 / FPS
    # 0.6s is 14 frames. Serial, a 60ms step per pair caps this at 20 - and every
    # frame would arrive in a pair, so the gaps would alternate ~0 and ~60ms.
    assert len(slot.puts) >= 11, f"only {len(slot.puts)} frames in 0.6s"
    assert max(slot.gaps) < interval * 2, (
        "a whole step went by with nothing published: "
        f"{[round(g * 1000, 1) for g in slot.gaps]}"
    )


async def test_the_ring_is_never_fed_while_a_model_step_is_reading_it(monkeypatch):
    """`PcmRing` has no lock, and this is why it does not need one.

    `feed` rebinds the sample array and *then* advances the origin, so a `window()`
    landing between those two statements reads new samples against a stale origin - a
    window off by one chunk, which is a mouth off by one chunk. The step runs on a
    thread and reads the ring there, so the guarantee is that the loop awaits it: the
    drain cannot run while a step is in flight. `FakeRing` raises if it ever does, and
    the loop logs rather than propagates, so the assertion is that nothing was logged.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer = FakeRenderer(step_seconds=0.050)
    ring = FakeRing(renderer=renderer)
    queued: deque[tuple[bytes, float]] = deque()

    async def feeding() -> None:
        """Audio arriving mid-step, which is when it always arrives."""
        while True:
            await asyncio.sleep(0.01)
            queued.append((b"\x01\x02", 1.0))

    pump = asyncio.create_task(feeding())
    try:
        await _run_loop(
            face, renderer, PacedClock(fps=FPS), ring, queued, stop_after=0.5
        )
    finally:
        pump.cancel()
    assert len(ring.fed) > 10, "the fixture never fed anything"


async def test_the_falling_edge_drops_a_pair_the_next_turn_must_not_show(monkeypatch):
    """A frame from the utterance that just ended, published into the first interval
    of the next one, is the mouth finishing somebody else's sentence - at exactly the
    moment the page is fading the face in. The CPU half can still be encoding when
    the turn ends, so the queue is emptied and the in-flight pair is disowned.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, ring = FakeRenderer(), FakeRing()
    slot = RecordingSlot()
    # Twice the frame rate, so production runs into the loop's backpressure limit and
    # a whole pair is always waiting when the turn ends. At the real 24fps of grants
    # the queue hovers at one frame and the leak would be intermittent.
    task = asyncio.create_task(
        _lipsync_loop(
            face, renderer, PacedClock(fps=FPS * 2), ring, deque(), slot,
            *_driving(), fps=FPS
        )
    )
    await asyncio.sleep(0.15)
    face.set_activity("idle")
    renderer.label = "b"               # everything encoded from here belongs to turn 2
    await asyncio.sleep(0.15)
    published = list(slot.puts)
    assert published, "the first turn published nothing at all"
    face.set_activity("speaking")
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fresh = slot.puts[len(published):]
    assert fresh, "the second turn published nothing at all"
    assert all(frame.startswith(b"b") for frame in fresh), (
        f"the new turn opened with the last turn's frames: {fresh}"
    )


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
        _lipsync_loop(
            face, renderer, clock, ring, deque(), RecordingSlot(),
            *_driving(), fps=FPS,
        )
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
            _lipsync_loop(
                face, renderer, clock, ring, deque(), RecordingSlot(),
                *_driving(), fps=FPS,
            ),
            1.0,
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
        _lipsync_loop(
            FaceBus(), renderer, clock, Exploding(), queued, RecordingSlot(),
            *_driving(), fps=FPS
        ),
        1.0,
    )


# --- the two seams ----------------------------------------------------------


async def test_the_frames_source_is_exactly_what_the_transport_asks_for():
    """`face_routes.LipsyncFrames` is four members, and they live on three objects that
    are deliberately kept apart - `Slot` is `put`/`get` and knows nothing about who
    reads it, `Renderer` knows whether it has given up, and `ClipClock` knows only
    where the driving clip is. Joining them is assembly, so this is the join.

    `async` for `position` alone: it reads `loop.time()`, so a sync test would call it
    with no running loop and never find out whether the assembly wired the clock in at
    all - which is the failure this file exists for.
    """
    from daemon.face_lipsync.render import ClipClock
    from daemon.face_lipsync.ring import Slot

    renderer, slot = FakeRenderer(), Slot()
    epoch = asyncio.get_running_loop().time()
    driving, _caches, _lengths = _driving()
    driving.driver = replace(
        driving.driver, clip=ClipClock(fps=24.0, frames=193, epoch=epoch - 4.0)
    )
    frames = _LipsyncFrames(renderer=renderer, slot=slot, driving=driving)
    assert frames.clip == "idle2"
    assert not hasattr(frames, "box"), (
        "a box here would mean the page is placing a crop again"
    )
    # Four seconds after the epoch, on the clock this is actually being asked on -
    # a `position` wired to a different clock, or to none, cannot come out here.
    assert 4.0 <= frames.position() < 4.2
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
        async def close() -> bool:
            # `-> bool`, not `None`: `_wake_round` drops the round when the release
            # reports failure rather than opening a session on a wedged device, so a
            # fake that returns None reads as "the microphone never came back" and
            # `run_voice` is never reached. Main's own fakes use the same shape
            # (tests/test_wake.py).
            return True

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


# --- idle goes out on the same stream, for a colour reason ----------------------


async def test_an_idle_face_still_publishes_the_clips_own_frames(monkeypatch):
    """Nothing is rendered while idle, but something is still sent.

    The page used to play the driving clip in a `<video>` at idle and show this canvas
    only while speaking. Chrome's decode of the untagged mp4 disagrees with its decode
    of our JPEGs - measured R +3.0, G +2.1, B +1.2, against a JPEG path faithful to our
    bytes within 0.2 - so the entire picture, background included, shifted darker and
    off-hue the instant speech began. One decoder is the only fix that does not depend
    on guessing how a browser will read an untagged file.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("idle")
    renderer, clock, ring = FakeRenderer(), FakeClock([0] * 20), FakeRing()
    slot = await _run_loop(face, renderer, clock, ring, deque(), stop_after=0.3)

    assert not renderer.calls, "an idle face must still never reach the model"
    assert slot.puts, "and must still be publishing - that is the whole point"
    assert all(p in IDLE_FRAMES for p in slot.puts), (
        f"idle must publish the clip's own frames, got {slot.puts[:3]}"
    )
    assert len(set(slot.puts)) > 1, (
        "and it must advance through the clip rather than repeating one frame"
    )


async def test_speaking_with_nothing_ready_publishes_nothing_rather_than_the_clip(
    monkeypatch,
):
    """The failure the branch order exists to prevent: falling through to the clip's
    own frame mid-sentence would shut the mouth for a frame in the middle of a word.
    Holding the slot's last frame is a mouth one frame stale, which is not the same
    thing at all."""
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer, ring = FakeRenderer(), FakeRing()
    # A clock that never grants: the queue stays empty for the whole run.
    slot = await _run_loop(face, renderer, FakeClock([]), ring, deque(), stop_after=0.25)

    assert not any(p in IDLE_FRAMES for p in slot.puts), (
        f"a speaking face must never be handed a clip frame, got {slot.puts[:3]}"
    )


async def _run_edges(face, renderer, clock, ring, *, speak: float, then_idle: float):
    """Speak, fall silent, and keep publishing - the two edges in one run."""
    slot = RecordingSlot()
    task = asyncio.create_task(
        _lipsync_loop(
            face, renderer, clock, ring, deque(), slot, *_driving(), fps=FPS
        )
    )
    await asyncio.sleep(speak)
    face.set_activity("idle")
    await asyncio.sleep(then_idle)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return slot


async def test_the_falling_edge_ramps_the_mouth_out_before_the_clip_takes_over(
    monkeypatch,
):
    """The end of an utterance is a dissolve, not a cut.

    Speech stops and the next frame used to be the clip untouched - a generated mouth
    replaced by a real one between two frames, measured as a 5.97px step in the mouth
    region against a 2.46px median during speech, and seen as the mouth "갑자기 확
    닫히는" snap. It is structural: `evals/face_lipsync_idle_spike.py` measured all 88
    conditioning windows and not one renders this avatar's resting mouth closed, so the
    last generated frame is always parted where the clip's own is sealed.
    """
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer = FakeRenderer()
    renderer.releases = RELEASE_FRAMES
    clock, ring = FakeClock([0, 1, 2, 3] + [None] * 60), FakeRing()
    slot = await _run_edges(
        face, renderer, clock, ring, speak=0.2, then_idle=RELEASE_FRAMES / FPS + 0.25
    )

    ramp = [p.decode() for p in slot.puts if p.startswith(b"ar")]
    assert ramp == [f"ar{i}" for i in range(1, RELEASE_FRAMES + 1)], ramp
    last_ramp = max(i for i, p in enumerate(slot.puts) if p.startswith(b"ar"))
    first_clip = next(i for i, p in enumerate(slot.puts) if p in IDLE_FRAMES)
    assert last_ramp < first_clip, (
        f"the clip took over at {first_clip}, before the ramp ended at {last_ramp}"
    )
    # The index handed over is the page's own playhead, so it advances rather than
    # counting up from wherever the utterance stopped.
    handed = [index for index, _ in renderer.released]
    assert handed == sorted(handed), handed


async def test_a_barge_in_mid_ramp_starts_the_next_ramp_over(monkeypatch):
    """A ramp that kept its place would hand the next utterance a half-spent one, so
    the second falling edge would cut where the first dissolved."""
    monkeypatch.setattr(app_module, "release_lipsync_memory", lambda: None)
    face = FaceBus()
    face.set_activity("speaking")
    renderer = FakeRenderer()
    renderer.releases = RELEASE_FRAMES
    clock, ring = FakeClock([0, 1, 2, 3] + [None] * 60), FakeRing()
    slot = RecordingSlot()
    task = asyncio.create_task(
        _lipsync_loop(
            face, renderer, clock, ring, deque(), slot, *_driving(), fps=FPS
        )
    )
    await asyncio.sleep(0.15)
    face.set_activity("idle")
    await asyncio.sleep(3 / FPS)          # part-way into the ramp
    face.set_activity("speaking")         # barge-in
    await asyncio.sleep(0.1)
    face.set_activity("idle")
    await asyncio.sleep(RELEASE_FRAMES / FPS + 0.25)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    steps = [step for _, step in renderer.released]
    assert steps.count(1) == 2, f"each falling edge must start the ramp at 1, got {steps}"
    assert max(steps) <= RELEASE_FRAMES, steps

