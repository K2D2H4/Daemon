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


def test_sips_resize_caps_long_edge():
    argv = screen.SIPS_RESIZE_ARGS("/tmp/x.jpg", 1536)
    assert argv[:2] == ["sips", "-Z"]
    assert "1536" in argv
    assert "jpeg" in argv             # force jpeg output


@pytest.mark.skipif(platform.system() == "Darwin", reason="guard is for non-mac")
async def test_capture_refuses_off_darwin():
    with pytest.raises(ToolError, match="macOS"):
        await screen.capture_display(long_edge=1536)
