"""The face bus and the mood tag stripper.

The bus is a fan-out with two deliberate asymmetries, and both are tested here
because both are the point: `level` is a latest-value slot (a late mouth position
is worse than a skipped one) while one-shots are queued (a dropped laugh is a
missing expression). And with nobody subscribed, publishing must be free.
"""

import array
import asyncio

import pytest

from daemon.face import FaceBus, FaceState, OneShot, SpeechClock, split_mood


async def _drain(agen, n):
    """The next `n` events, or fail the test rather than hang the suite."""
    out = []
    async with asyncio.timeout(1.0):
        for _ in range(n):
            out.append(await agen.__anext__())
    return out


def test_publishing_with_no_subscribers_changes_only_the_snapshot():
    bus = FaceBus()
    bus.set_activity("speaking")
    bus.set_level(0.5)
    bus.one_shot("amused")
    assert bus.state == FaceState(activity="speaking", level=0.5)


async def test_first_event_is_a_snapshot_not_a_change():
    bus = FaceBus()
    bus.set_activity("thinking")
    agen = bus.subscribe()
    (first,) = await _drain(agen, 1)
    assert first == FaceState(activity="thinking", level=0.0)
    await agen.aclose()


async def test_level_is_coalesced_not_queued():
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)  # snapshot
    for v in (0.1, 0.2, 0.3):
        bus.set_level(v)
    (only,) = await _drain(agen, 1)
    assert only.level == 0.3
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agen.__anext__(), 0.05)
    await agen.aclose()


async def test_one_shots_are_queued_and_arrive_before_state():
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)  # snapshot
    bus.one_shot("amused")
    bus.set_activity("speaking")
    first, second = await _drain(agen, 2)
    assert first == OneShot(clip="amused")
    assert second.activity == "speaking"
    await agen.aclose()


async def test_a_repeated_activity_is_not_republished():
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)
    bus.set_activity("idle")  # already idle
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agen.__anext__(), 0.05)
    await agen.aclose()


async def test_two_subscribers_both_get_events():
    bus = FaceBus()
    a, b = bus.subscribe(), bus.subscribe()
    await _drain(a, 1)
    await _drain(b, 1)
    bus.set_activity("working")
    (ea,) = await _drain(a, 1)
    (eb,) = await _drain(b, 1)
    assert ea.activity == eb.activity == "working"
    await a.aclose()
    await b.aclose()


def test_level_is_clamped():
    bus = FaceBus()
    bus.set_level(-1.0)
    assert bus.state.level == 0.0
    bus.set_level(9.0)
    assert bus.state.level == 1.0


class RecordingBus(FaceBus):
    """Captures every publish.

    Exists because the bus *coalesces* state on purpose: a subscriber that is slow -
    or a transition faster than one event loop turn - correctly sees only the latest
    activity. That is right for a display and useless for asserting a sequence, so
    tests that care about the order record the publishes instead of racing the stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.activities: list[str] = []
        self.shots: list[str] = []

    def set_activity(self, activity):  # type: ignore[override]
        before = self.state.activity
        super().set_activity(activity)
        if activity != before:
            self.activities.append(activity)

    def one_shot(self, clip):  # type: ignore[override]
        super().one_shot(clip)
        self.shots.append(clip)


def test_a_recording_bus_sees_transitions_a_subscriber_may_coalesce_away():
    bus = RecordingBus()
    bus.set_activity("thinking")
    bus.set_activity("speaking")
    bus.set_activity("idle")
    assert bus.activities == ["thinking", "speaking", "idle"]


# --- SpeechClock: speaking and level on the playback clock -------------------

RATE, WIDTH = 24_000, 2          # what the voice path plays: 24kHz mono 16-bit


def _pcm(seconds: float, amplitude: int = 16384) -> bytes:
    n = int(RATE * seconds)
    return array.array("h", [amplitude] * n).tobytes()


def test_speaking_holds_for_exactly_the_audio_that_was_fed():
    bus = RecordingBus()
    clock = SpeechClock(bus, sample_rate=RATE, bytes_per_frame=WIDTH)
    clock.fed(_pcm(0.5), at=100.0)
    clock.pump(100.0)
    assert bus.state.activity == "speaking"
    clock.pump(100.49)
    assert bus.state.activity == "speaking"
    clock.pump(100.51)
    assert bus.state.activity == "idle"


def test_a_second_chunk_extends_speaking_rather_than_restarting_it():
    bus = RecordingBus()
    clock = SpeechClock(bus, sample_rate=RATE, bytes_per_frame=WIDTH)
    clock.fed(_pcm(0.5), at=100.0)
    clock.fed(_pcm(0.5), at=100.1)   # arrives early - it queues behind, it does not overlap
    clock.pump(100.9)
    assert bus.state.activity == "speaking"
    clock.pump(101.01)
    assert bus.state.activity == "idle"
    # One transition in and one out. A queue that momentarily empties mid-utterance
    # must not flicker the face.
    assert bus.activities == ["speaking", "idle"]


def test_level_is_published_for_when_the_chunk_is_audible_not_when_it_arrived():
    bus = RecordingBus()
    clock = SpeechClock(bus, sample_rate=RATE, bytes_per_frame=WIDTH)
    clock.fed(_pcm(0.5, amplitude=0), at=100.0)        # silence first
    clock.fed(_pcm(0.5, amplitude=32000), at=100.0)    # loud, queued behind it
    clock.pump(100.2)
    quiet = bus.state.level
    clock.pump(100.7)
    loud = bus.state.level
    assert quiet < 0.05, "the loud chunk is still queued; the mouth must not open yet"
    assert loud > 0.5, "by now the loud chunk is audible"


def test_level_returns_to_zero_once_playback_is_over():
    bus = RecordingBus()
    clock = SpeechClock(bus, sample_rate=RATE, bytes_per_frame=WIDTH)
    clock.fed(_pcm(0.2, amplitude=32000), at=100.0)
    clock.pump(100.1)
    assert bus.state.level > 0.5
    clock.pump(100.5)
    assert bus.state.level == 0.0


@pytest.mark.parametrize(
    "raw,text,mood",
    [
        ("[mood:amused] 그래서 웃었어", "그래서 웃었어", "amused"),
        ("  [mood:sulky]\n삐졌어", "삐졌어", "sulky"),
        ("[MOOD:Curious] 궁금해", "궁금해", "curious"),
        ("웃었어", "웃었어", None),
        ("태그는 [mood:amused] 형식이야", "태그는 [mood:amused] 형식이야", None),
        ("[mood:furious] 화났어", "[mood:furious] 화났어", None),
    ],
)
def test_split_mood(raw, text, mood):
    assert split_mood(raw) == (text, mood)


# --- the lip-sync PCM sink ---------------------------------------------------
#
# The lip-sync ring is addressed by when audio is HEARD, and this class already does
# that arithmetic for `speaking`. Feeding the ring anywhere else would mean doing it
# twice, which is how a mouth ends up dubbed.


def test_the_sink_receives_every_chunk_with_the_moment_it_becomes_audible():
    got: list[tuple[int, float]] = []
    clock = SpeechClock(
        FaceBus(),
        sample_rate=RATE,
        bytes_per_frame=WIDTH,
        pcm_sink=lambda chunk, at: got.append((len(chunk), at)),
    )
    one = b"\x00\x01" * (RATE // 2)          # 0.5s
    clock.fed(one, at=10.0)
    clock.fed(one, at=10.0)                  # queued while the first is still playing
    assert [n for n, _ in got] == [len(one), len(one)]
    assert got[0][1] == 10.0
    assert got[1][1] == 10.5, (
        "the second chunk is heard when the first finishes, not when it was queued"
    )


def test_the_sink_gets_the_chunks_start_not_its_end():
    """`starts`, not `_until`. Handing over the end stamps every chunk one chunk
    late, and the whole mouth runs behind the sound by however long a chunk is -
    which is exactly the dubbing this class exists to prevent."""
    got: list[float] = []
    clock = SpeechClock(
        FaceBus(),
        sample_rate=RATE,
        bytes_per_frame=WIDTH,
        pcm_sink=lambda chunk, at: got.append(at),
    )
    clock.fed(b"\x00\x01" * RATE, at=4.0)    # 1.0s of audio
    assert got == [4.0], "a chunk queued into silence is audible immediately"


def test_empty_chunks_reach_neither_the_clock_nor_the_sink():
    got: list[float] = []
    clock = SpeechClock(
        FaceBus(), sample_rate=RATE, bytes_per_frame=WIDTH,
        pcm_sink=lambda chunk, at: got.append(at),
    )
    clock.fed(b"", at=1.0)
    assert got == []


def test_no_sink_is_the_default_and_changes_nothing():
    """Lip-sync is off by default, and the class must behave exactly as before."""
    bus = FaceBus()
    clock = SpeechClock(bus, sample_rate=RATE, bytes_per_frame=WIDTH)
    clock.fed(b"\x00\x01" * RATE, at=0.0)
    clock.pump(at=0.5)
    assert bus.state.activity == "speaking"
