"""The audio ring and the frame slot.

Both encode a rule from the design. The ring clamps rather than indexes negatively,
because a turn's first frames legitimately ask for audio from before the turn began.
The slot overwrites rather than queues, for the same reason `level` does on the bus:
a window that stopped consuming should resume at the current mouth, not replay a
backlog of stale ones.
"""

import numpy as np

from daemon.face_lipsync.ring import PcmRing, Slot


def _silence(ms: int, rate: int = 24_000) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def test_a_window_before_the_start_is_clamped_not_negative():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_silence(500), audible_at=0.0)
    got = ring.window(frame_index=0, fps=24.0, origin=0.0)
    assert len(got) > 0
    assert not np.isnan(got).any()


def test_the_window_grows_no_longer_than_ten_indices():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_silence(2000), audible_at=0.0)
    got = ring.window(frame_index=48, fps=24.0, origin=0.0)
    assert len(got) == int(24_000 * 0.200)


def test_the_slot_keeps_only_the_latest_frame():
    slot = Slot()
    slot.put(b"first")
    slot.put(b"second")
    assert slot.get() == b"second"
    assert slot.get() == b"second"


def test_an_empty_slot_reports_nothing_rather_than_blocking():
    assert Slot().get() is None
