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


async def test_one_shots_are_queued_and_arrive_after_state():
    """A one-shot is still queued (a dropped laugh is a missing expression), but
    it is delivered *after* the state it shares a wake with, not before.

    The two are published in one synchronous block by `ConversationLoop._speak`,
    so they always arrive together and only this order can separate them. Sent
    shot-first the page starts the mood and the `speaking` right behind it cuts
    it - spec 3.2 lets `speaking` cut a one-shot - and the expression was on
    screen for about 0ms. Spec 3.6's original ordering was written for voice,
    where the audio really does arrive after the tag; the text path has no audio
    for the mouth to lag, and 3.6 now says so.
    """
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)  # snapshot
    bus.set_activity("speaking")
    bus.one_shot("amused")
    first, second = await _drain(agen, 2)
    assert first.activity == "speaking"
    assert second == OneShot(clip="amused")
    await agen.aclose()


async def test_the_order_within_a_wake_does_not_depend_on_publish_order():
    """Same wake, published the other way round: still state, then the shot.

    Both land in the subscriber's mailbox before it runs, so `one_shot` before
    `set_activity` and `set_activity` before `one_shot` are the same event as far
    as the bus is concerned - and the delivery order has to be the bus's own
    decision rather than a side effect of where its generator was suspended.
    """
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)  # snapshot
    bus.one_shot("amused")
    bus.set_activity("speaking")
    first, second = await _drain(agen, 2)
    assert first.activity == "speaking"
    assert second == OneShot(clip="amused")
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


async def test_an_unchanged_level_is_not_republished():
    """Design spec §2: the socket stays open and the traffic is zero when nothing
    is happening. `SpeechClock.pump` runs at 25Hz for the whole of a voice
    conversation and hands `set_level` the same 0.0 on every tick once the
    speaker is empty - forty identical events a second down every open stream
    unless the bus stops them here, the way it already stops a repeated
    activity."""
    bus = FaceBus()
    agen = bus.subscribe()
    await _drain(agen, 1)  # snapshot, level 0.0
    # A real change gets through first, so this is a filter and not a mute.
    bus.set_level(0.4)
    (changed,) = await _drain(agen, 1)
    assert changed.level == 0.4
    for _ in range(5):
        bus.set_level(0.4)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agen.__anext__(), 0.05)
    await agen.aclose()


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
