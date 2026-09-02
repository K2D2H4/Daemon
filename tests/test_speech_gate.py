"""The speech gate: only a person talking reaches the model.

Measured 2026-09-02 on the owner's Mac (evals notes in daemon/MEASURED.md): the
echo-cancelled microphone carries the room at about -57 dBFS all the time, leaks
the daemon's own voice in bursts of at most three consecutive VAD frames, and hears
a person at 72% speech frames with sustained runs. Vertex 2.5 native audio answers
the first two as if the owner had spoken; 3.1 ignores them. The gate turns
everything that is not a sustained run of speech into digital silence, which is
the input shape 2.5 was measured to answer fastest and most reliably on.
"""

from __future__ import annotations

from daemon.voice.speech_gate import SpeechGate
from daemon.voice.vad import FRAME_BYTES

ROOM = b"\x01\x00" * (FRAME_BYTES // 2)
"""A frame with content the VAD does not call speech - the room."""
SPEECH = b"\x10\x00" * (FRAME_BYTES // 2)
"""A frame the fake VAD calls speech."""
ZEROS = bytes(FRAME_BYTES)


class FakeVad:
    """Speech is whatever starts with 0x10. Stateful only in that it counts resets."""

    sample_rate = 16_000
    frame_samples = FRAME_BYTES // 2

    def __init__(self) -> None:
        self.resets = 0
        self.seen: list[bytes] = []

    def probability(self, frame: bytes) -> float:
        assert len(frame) == FRAME_BYTES, len(frame)
        self.seen.append(frame)
        return 0.9 if frame[0] == 0x10 else 0.1

    def reset(self) -> None:
        self.resets += 1


def gate(**kwargs) -> SpeechGate:
    defaults = dict(open_after_frames=4, pre_roll_frames=6, hangover_frames=10,
                    density_open_frames=1000)  # density off unless a test asks for it
    defaults.update(kwargs)
    return SpeechGate(FakeVad(), **defaults)


def feed_all(g: SpeechGate, frames: list[bytes]) -> list[bytes]:
    out: list[bytes] = []
    for frame in frames:
        sent = g.feed(frame)
        if sent:
            out.append(sent)
    return out


def split(sent: list[bytes]) -> list[bytes]:
    joined = b"".join(sent)
    return [joined[i:i + FRAME_BYTES] for i in range(0, len(joined), FRAME_BYTES)]


def test_the_room_alone_becomes_digital_silence_of_the_same_length() -> None:
    g = gate()
    frames = split(feed_all(g, [ROOM] * 20))
    assert len(frames) == 20
    assert all(f == ZEROS for f in frames)


def test_a_leak_shorter_than_the_opening_run_never_gets_through() -> None:
    """The echo canceller's worst measured leak was eight consecutive frames."""
    g = gate(open_after_frames=12)
    frames = split(feed_all(g, [ROOM] * 5 + [SPEECH] * 8 + [ROOM] * 5))
    assert len(frames) == 18
    assert all(f == ZEROS for f in frames)


def test_the_shipped_numbers_clear_the_measured_leak() -> None:
    """A regression guard on the fit, not a restatement of it: the leak measured on
    the owner's machine was 8 consecutive frames and 8 in its densest 25, and the
    defaults must not open on either."""
    from daemon.voice.speech_gate import SpeechGate

    g = SpeechGate(FakeVad())
    leak = [ROOM] * 30 + [SPEECH] * 8 + [ROOM] * 30
    assert all(f == ZEROS for f in split(feed_all(g, leak)))
    assert g.open is False


def test_speech_the_vad_scores_unevenly_opens_on_density() -> None:
    """A quiet voice gives no long run. 18 of 25 opens it; the leak's 8 does not."""
    g = gate(open_after_frames=12, density_open_frames=18, pre_roll_frames=0)
    uneven = ([SPEECH] * 3 + [ROOM]) * 6  # 18 speech in 24 frames, never 12 in a row
    feed_all(g, uneven)
    assert g.open is True


def test_sustained_speech_opens_with_the_head_kept_by_the_pre_roll() -> None:
    g = gate(open_after_frames=4, pre_roll_frames=6)
    marked = [bytes([0x01, i]) * (FRAME_BYTES // 2) for i in range(8)]  # 8 distinct room frames
    frames = split(feed_all(g, marked + [SPEECH] * 4))
    # 8 room frames and the first 3 speech frames went out as zeros while the gate
    # was shut ...
    assert frames[:11] == [ZEROS] * 11
    # ... then, on the fourth speech frame, the pre-roll flushes: the two room
    # frames just before speech began plus the four speech frames, verbatim. The
    # stream carries that stretch twice - once as silence, once as recorded - which
    # the server tolerates and a lost syllable would not be.
    assert frames[11:] == marked[6:8] + [SPEECH] * 4


def test_the_gate_stays_open_across_a_pause_shorter_than_the_hangover() -> None:
    g = gate(open_after_frames=2, pre_roll_frames=0, hangover_frames=5)
    frames = split(feed_all(g, [SPEECH] * 2 + [ROOM] * 4 + [SPEECH] * 2))
    # the first speech frame went out as silence (gate still shut), the second
    # opened it and flushed both; from there everything crossed as recorded - the
    # pause included
    assert frames == [ZEROS] + [SPEECH] * 2 + [ROOM] * 4 + [SPEECH] * 2


def test_the_gate_closes_after_the_hangover_and_goes_back_to_silence() -> None:
    g = gate(open_after_frames=2, pre_roll_frames=0, hangover_frames=3)
    frames = split(feed_all(g, [SPEECH] * 2 + [ROOM] * 6))
    assert frames[:3] == [ZEROS] + [SPEECH] * 2
    # three room frames of hangover cross as recorded, the rest are zeros
    assert frames[3:6] == [ROOM] * 3
    assert frames[6:] == [ZEROS] * 3


def test_blocks_are_reframed_and_the_remainder_carried() -> None:
    g = gate()
    first = g.feed(ROOM[:700])
    assert first == b""  # not a whole frame yet
    second = g.feed(ROOM[700:] + ROOM[:100])
    assert second == ZEROS  # exactly one frame completed
    third = g.feed(ROOM[100:])
    assert third == ZEROS


def test_reset_forgets_the_pre_roll_and_resets_the_vad() -> None:
    """Called whenever the half-duplex gate drops audio: room sound from before the
    daemon spoke must not flush as the head of what comes after."""
    g = gate(open_after_frames=2, pre_roll_frames=4)
    marked = [bytes([0x01, i]) * (FRAME_BYTES // 2) for i in range(4)]
    feed_all(g, marked)
    g.reset()
    frames = split(feed_all(g, [SPEECH] * 2))
    # the first speech frame went out as zeros (gate still shut); the second opened
    # it and flushed a pre-roll holding only the two speech frames - no room frame
    # from before the reset
    assert frames == [ZEROS, SPEECH, SPEECH]
    assert not any(f in marked for f in frames)
    assert g.vad.resets == 1


def test_a_second_reset_with_nothing_pending_is_free() -> None:
    g = gate()
    g.reset()
    assert g.vad.resets == 0  # a fresh gate has nothing to forget
    g.feed(ROOM)
    g.reset()
    g.reset()
    assert g.vad.resets == 1  # nothing to forget the second time


def test_open_reports_the_state_for_the_caller() -> None:
    g = gate(open_after_frames=2, pre_roll_frames=0, hangover_frames=2)
    assert g.open is False
    feed_all(g, [SPEECH] * 2)
    assert g.open is True
    feed_all(g, [ROOM] * 3)
    assert g.open is False
