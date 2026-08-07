import platform

import pytest

from daemon.tools import screen
from daemon.tools.base import ToolError


def test_screencapture_argv_full_display_omits_cursor():
    argv = screen.SCREENCAPTURE_ARGS("/tmp/x.jpg", None)
    assert argv[0] == "screencapture"
    assert "-C" not in argv           # cursor never captured — Global Constraints
    assert "-x" in argv               # silent, no shutter sound
    assert argv[-1] == "/tmp/x.jpg"


def test_screencapture_argv_window_uses_l_flag():
    argv = screen.SCREENCAPTURE_ARGS("/tmp/x.jpg", 42)
    assert "-l" in argv and "42" in argv
    assert "-C" not in argv          # cursor never captured, window branch too


def test_sips_resize_caps_long_edge():
    argv = screen.SIPS_RESIZE_ARGS("/tmp/x.jpg", 1536)
    assert argv[:2] == ["sips", "-Z"]
    assert "1536" in argv
    assert "jpeg" in argv             # force jpeg output


@pytest.mark.skipif(platform.system() == "Darwin", reason="guard is for non-mac")
async def test_capture_refuses_off_darwin():
    with pytest.raises(ToolError, match="macOS"):
        await screen.capture_display(long_edge=1536)


class _FakeProcess:
    """Just enough of `asyncio.subprocess.Process` for `_run` to drive."""

    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self):
        return b"", self._stderr

    def kill(self) -> None:  # pragma: no cover - not exercised by this test
        pass

    async def wait(self) -> None:  # pragma: no cover - not exercised by this test
        pass


async def test_run_keeps_tcc_hint_but_preserves_real_stderr(monkeypatch):
    """Regression for task 0.1 review Finding 1: a non-zero `screencapture` exit
    used to be laundered into the TCC hint with the real reason thrown away. A
    transient failure ("could not create image from display", measured on a real
    Mac - see task-0.1-report.md) must still name itself, even though the TCC
    hint stays the primary message.
    """

    async def fake_exec(*argv, **kwargs):
        return _FakeProcess(1, stderr=b"could not create image from display\n")

    monkeypatch.setattr(screen.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(ToolError) as excinfo:
        await screen._run(
            "screencapture", ["screencapture", "-x"], timeout_secs=5, on_failure=screen.TCC_HINT
        )
    message = str(excinfo.value)
    assert screen.TCC_HINT in message
    assert "could not create image from display" in message


async def test_run_tcc_hint_alone_when_screencapture_says_nothing(monkeypatch):
    async def fake_exec(*argv, **kwargs):
        return _FakeProcess(1, stderr=b"")

    monkeypatch.setattr(screen.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(ToolError, match=f"^{screen.TCC_HINT}$"):
        await screen._run(
            "screencapture", ["screencapture", "-x"], timeout_secs=5, on_failure=screen.TCC_HINT
        )
