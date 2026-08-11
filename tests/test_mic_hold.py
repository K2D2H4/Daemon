"""The process-wide microphone-hold counter.

The daemon's own wake listener holds the microphone whenever
DAEMON_WAKE_ENABLED is on. Without this counter the CoreAudio probe reports the
input device busy forever, the gate reads that as "on a call", and the local
speaker becomes unreachable - so switching voice *on* is what switches the voice
route *off*. See docs/superpowers/specs/2026-08-11-proactivity-humanization-design.md
"""

from daemon import mic_hold


def test_not_held_by_default() -> None:
    assert mic_hold.held() is False


def test_hold_marks_it_held() -> None:
    with mic_hold.hold():
        assert mic_hold.held() is True
    assert mic_hold.held() is False


def test_nested_holds_release_in_order() -> None:
    """Two streams may overlap - a wake listener and a voice session. The inner
    one exiting must not tell the gate the microphone is free."""
    with mic_hold.hold():
        with mic_hold.hold():
            assert mic_hold.held() is True
        assert mic_hold.held() is True
    assert mic_hold.held() is False


def test_release_survives_an_exception() -> None:
    try:
        with mic_hold.hold():
            raise RuntimeError("stream died")
    except RuntimeError:
        pass
    assert mic_hold.held() is False
