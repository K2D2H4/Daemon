"""macOS mic TCC access, tested with injected fake frameworks — no AVFoundation,
no prompt, no microphone (tests/CLAUDE.md rule).
"""

from __future__ import annotations

from daemon.voice.mic_access import (
    Frameworks,
    microphone_authorization_status,
    request_microphone_access,
)


class FakeCaptureDevice:
    def __init__(self, status: int, granted: bool | None = None) -> None:
        self._status = status
        self._granted = granted
        self.requested = False

    def authorizationStatusForMediaType_(self, _media) -> int:
        return self._status

    def requestAccessForMediaType_completionHandler_(self, _media, handler) -> None:
        self.requested = True
        # The real API delivers asynchronously; the fake fires synchronously, which
        # is enough for the pump loop to see "done" on its first check.
        if self._granted is not None:
            handler(self._granted)


class FakeAV:
    AVMediaTypeAudio = "audio"

    def __init__(self, device: FakeCaptureDevice) -> None:
        self.AVCaptureDevice = device


class FakeDate:
    @staticmethod
    def dateWithTimeIntervalSinceNow_(_secs):
        return object()


class FakeLoop:
    def runMode_beforeDate_(self, _mode, _date) -> None:  # never needs to pump
        raise AssertionError("pumped the runloop though the handler already fired")


class FakeRunLoop:
    @staticmethod
    def currentRunLoop() -> FakeLoop:
        return FakeLoop()


class FakeFoundation:
    NSDate = FakeDate
    NSRunLoop = FakeRunLoop


def _fw(status: int, granted: bool | None = None) -> tuple[Frameworks, FakeCaptureDevice]:
    device = FakeCaptureDevice(status, granted)
    return Frameworks(av=FakeAV(device), foundation=FakeFoundation()), device


def test_status_maps_the_avauthorization_ints() -> None:
    status_map = {
        0: "not_determined",
        1: "restricted",
        2: "denied",
        3: "authorized",
    }
    for code, expected in status_map.items():
        fw, _ = _fw(code)
        assert microphone_authorization_status(frameworks=fw) == expected


def test_status_never_raises_on_absent_frameworks() -> None:
    # frameworks=None on a machine without AVFoundation → the real import fails and
    # is caught. Simulate by passing a Frameworks whose av lacks the method.
    class Broken:
        pass

    fw = Frameworks(av=Broken(), foundation=Broken())
    assert microphone_authorization_status(frameworks=fw) == "unavailable"


def test_request_returns_authorized_when_already_authorized_without_prompting() -> None:
    fw, device = _fw(3)
    assert request_microphone_access(frameworks=fw) == "authorized"
    assert device.requested is False, "must not re-prompt when already decided"


def test_request_returns_denied_when_already_denied_without_prompting() -> None:
    fw, device = _fw(2)
    assert request_microphone_access(frameworks=fw) == "denied"
    assert device.requested is False


def test_request_prompts_when_not_determined_and_grant_is_given() -> None:
    fw, device = _fw(0, granted=True)
    assert request_microphone_access(frameworks=fw) == "authorized"
    assert device.requested is True


def test_request_prompts_when_not_determined_and_grant_is_refused() -> None:
    fw, device = _fw(0, granted=False)
    assert request_microphone_access(frameworks=fw) == "denied"
    assert device.requested is True


# Coverage tests for pump loop and timeout paths (Task 4 review finding 2)


class DeferredCaptureDevice(FakeCaptureDevice):
    """Stores the handler instead of firing it synchronously."""

    def requestAccessForMediaType_completionHandler_(self, _media, handler) -> None:
        self.requested = True
        self._handler = handler  # do NOT call yet


class PumpingLoop:
    """Fires the stored handler on the first runMode_beforeDate_ call."""

    def __init__(self, device: DeferredCaptureDevice) -> None:
        self._device = device

    def runMode_beforeDate_(self, _mode, _date) -> None:
        self._device._handler(True)  # callback arrives during the pump


class PumpingFoundation:
    """Foundation mock that returns a PumpingLoop."""

    def __init__(self, loop: PumpingLoop) -> None:
        self._loop = loop
        self.NSDate = FakeDate

    @property
    def NSRunLoop(self) -> type:
        class RunLoop:
            @staticmethod
            def currentRunLoop():
                return self._loop

        return RunLoop


def test_request_pumps_and_returns_result_when_handler_fires_during_pump() -> None:
    """Exercises the pump loop body. Handler is deferred until pump calls it."""
    device = DeferredCaptureDevice(0)  # not_determined
    pumping_loop = PumpingLoop(device)
    fw = Frameworks(
        av=FakeAV(device),
        foundation=PumpingFoundation(pumping_loop),
    )
    assert request_microphone_access(frameworks=fw) == "authorized"
    assert device.requested is True


class NoopLoop:
    """Runloop that never needs to pump (handler never fires, just passes)."""

    def runMode_beforeDate_(self, _mode, _date) -> None:
        pass  # harmless no-op


class NoopFoundation:
    """Foundation mock that returns a NoopLoop."""

    NSDate = FakeDate

    class NSRunLoop:
        @staticmethod
        def currentRunLoop() -> NoopLoop:
            return NoopLoop()


def test_request_times_out_and_returns_not_determined() -> None:
    """Handler never fires, tiny timeout. Guarantees the pump loop cannot block."""
    device = DeferredCaptureDevice(0)  # not_determined
    fw = Frameworks(
        av=FakeAV(device),
        foundation=NoopFoundation(),
    )
    assert request_microphone_access(frameworks=fw, timeout=0.05) == "not_determined"
    assert device.requested is True


def test_request_unmapped_status_returns_unavailable_without_prompting() -> None:
    """Unknown/unmapped status code 4 must return unavailable, not prompt."""
    fw, device = _fw(4)  # status code 4 is unmapped (no such AVAuthorizationStatus)
    assert request_microphone_access(frameworks=fw) == "unavailable"
    assert device.requested is False, "must not prompt for unmapped status"


def test_request_broken_frameworks_never_raises() -> None:
    """request_microphone_access with missing/broken frameworks → unavailable."""

    class Broken:
        pass

    fw = Frameworks(av=Broken(), foundation=Broken())
    assert request_microphone_access(frameworks=fw) == "unavailable"


def test_request_restricted_returns_restricted_without_prompting() -> None:
    """Status 1 (restricted) is already decided; must not prompt."""
    fw, device = _fw(1)  # restricted
    assert request_microphone_access(frameworks=fw) == "restricted"
    assert device.requested is False
