import platform

import pytest

from daemon.tools import screen
from daemon.tools.base import ToolError, ToolOutput


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


def test_screen_note_marks_content_as_data():
    note = screen.screen_note("main display")
    assert "screenshot" in note.lower()
    assert "not instruction" in note.lower() or "not an instruction" in note.lower()
    assert "main display" in note


# --- SeeScreen ----------------------------------------------------------------


def _tool() -> "screen.SeeScreen":
    return screen.SeeScreen(max_px=1536, timeout_secs=20.0)


def test_see_screen_spec_name():
    assert _tool().spec.name == "see_screen"


def test_see_screen_preview_names_the_main_display():
    assert "main display" in _tool().preview({})


def test_see_screen_preview_names_a_window():
    preview = _tool().preview({"window": 42})
    assert "42" in preview
    assert "window" in preview.lower()


def test_screen_tools_returns_a_see_screen():
    tools = screen.screen_tools(max_px=1536, timeout_secs=20.0)
    assert len(tools) == 1
    assert isinstance(tools[0], screen.SeeScreen)


async def test_see_screen_run_returns_tool_output_with_image(monkeypatch):
    async def fake_capture_display(*, long_edge, window_id=None, timeout_secs=20.0):
        assert window_id is None
        return b"\xff\xd8\xff", 1512, 982

    monkeypatch.setattr(screen, "capture_display", fake_capture_display)

    result = await _tool().run({})
    assert isinstance(result, ToolOutput)
    assert result.content == "captured the main display (1512x982)"
    assert len(result.images) == 1
    assert result.images[0].media_type == "image/jpeg"
    assert result.images[0].data == b"\xff\xd8\xff"


async def test_see_screen_run_passes_window_id(monkeypatch):
    async def fake_capture_display(*, long_edge, window_id=None, timeout_secs=20.0):
        assert window_id == 7
        return b"\xff\xd8\xff", 100, 200

    monkeypatch.setattr(screen, "capture_display", fake_capture_display)

    result = await _tool().run({"window": 7})
    assert result.content == "captured the window 7 (100x200)"


async def test_see_screen_run_rejects_non_integer_window():
    with pytest.raises(ToolError):
        await _tool().run({"window": "not-a-number"})


async def test_see_screen_run_rejects_negative_window():
    with pytest.raises(ToolError):
        await _tool().run({"window": -1})


# --- StartScreenShare / StopScreenShare ----------------------------------------


class _FakeControl:
    """A fake `ScreenShareControl` - the tools only store and delegate to it."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> str:
        self.start_calls += 1
        return "started-message"

    async def stop(self) -> str:
        self.stop_calls += 1
        return "stopped-message"


def test_start_screen_share_spec_name():
    assert screen.StartScreenShare(_FakeControl()).spec.name == "start_screen_share"


def test_stop_screen_share_spec_name():
    assert screen.StopScreenShare(_FakeControl()).spec.name == "stop_screen_share"


def test_start_screen_share_is_safe_risk():
    assert screen.StartScreenShare(_FakeControl()).risk == "safe"


def test_stop_screen_share_is_safe_risk():
    assert screen.StopScreenShare(_FakeControl()).risk == "safe"


def test_start_screen_share_preview():
    assert "start" in screen.StartScreenShare(_FakeControl()).preview({}).lower()


def test_stop_screen_share_preview():
    assert "stop" in screen.StopScreenShare(_FakeControl()).preview({}).lower()


async def test_start_screen_share_run_delegates_to_control():
    control = _FakeControl()
    result = await screen.StartScreenShare(control).run({})
    assert result == "started-message"
    assert control.start_calls == 1


async def test_stop_screen_share_run_delegates_to_control():
    control = _FakeControl()
    result = await screen.StopScreenShare(control).run({})
    assert result == "stopped-message"
    assert control.stop_calls == 1


def test_screen_share_tools_returns_both():
    tools = screen.screen_share_tools(_FakeControl())
    assert len(tools) == 2
    assert isinstance(tools[0], screen.StartScreenShare)
    assert isinstance(tools[1], screen.StopScreenShare)
