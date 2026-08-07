"""A screenshot of the owner's screen, as JPEG bytes.

macOS only, and deliberately narrow: this module is the capture core, nothing
more. It shells out to `screencapture` and `sips` - both ship with macOS, so no
new dependency (Pillow or otherwise) is needed to get pixels off the display and
into a small JPEG. What is *not* here: a tool wrapper, a policy switch, or the
untrusted-data framing a frame needs once it reaches the model - those belong to
the layers built on top of this one.

**The cursor is never captured.** `-C` is the flag that would draw it into the
frame, and it must never appear in `SCREENCAPTURE_ARGS` - a stray mouse position
is a private detail (where the owner is pointing, what they are about to click)
that this capture has no business carrying. Asserted by the tests, not just
promised here.

**Residual risk, stated rather than assumed away**: a full-display screenshot
sees whatever is on screen - a password manager, a DM, a document - with no
redaction. Nothing in this file limits *when* a capture happens; that judgment
belongs to whatever calls `capture_display`, the same way `run_command` trusts
its caller to gate what commands reach it.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import tempfile
from pathlib import Path

from daemon.tools.base import ToolError

logger = logging.getLogger(__name__)

SCREENCAPTURE_ARGS = lambda path, window_id: (  # noqa: E731
    ["screencapture", "-x", "-l", str(window_id), "-t", "jpg", path]
    if window_id is not None
    else ["screencapture", "-x", "-t", "jpg", path]
)
"""`-x`: silent, no shutter sound - this fires without the owner clicking a
button. `-l <id>` captures one window instead of the whole display. `-C`, which
would draw the cursor into the frame, must never appear here."""

SIPS_RESIZE_ARGS = lambda path, long_edge: (  # noqa: E731
    ["sips", "-Z", str(long_edge), "-s", "format", "jpeg", path]
)
"""`-Z <n>` scales so the *longer* edge is at most `n`, preserving aspect ratio -
exactly what a frame budget needs. `-s format jpeg` forces JPEG even though
`screencapture -t jpg` already asked for it, because `sips` is what actually
writes the file this function reads back."""

TCC_HINT = (
    "macOS is blocking me from recording the screen. The owner can allow it in "
    "System Settings > Privacy & Security > Screen Recording, then restart me."
)
"""macOS TCC denial for Screen Recording does not raise an error `screencapture`
can report - it either exits non-zero or writes a zero-byte file, silently. Both
are mapped to this sentence, the same way `browser.py`'s `_explain` turns a
`-1743` AppleScript failure into an owner-facing instruction. Permission denial
is the most likely cause of a non-zero exit and `screencapture`'s stderr is often
unhelpful on its own, so this stays the *primary* message - but unlike a bare
guess, `_run` appends whatever `screencapture` actually said (Finding 1, task
0.1 review): a transient failure unrelated to TCC - `could not create image from
display`, measured on this machine - must not be laundered into a permission
story with no trace of what really happened."""


async def _run(
    name: str, argv: list[str], *, timeout_secs: float, on_failure: str | None = None
) -> str:
    """Run a subprocess to completion and return its stdout.

    `argv[0]` is a resolved absolute path (see `shutil.which` below); `name` is
    the short program name to speak in an error, so a failure reads "sips
    failed" rather than naming its `/usr/bin` location. `on_failure`, if given,
    is the *primary* message for a non-zero exit - used only by `screencapture`,
    whose failure exit is how a Screen Recording (TCC) denial usually shows up.
    A timeout is never remapped: that is a real timeout either way.

    Either way, the real stderr is never thrown away: it is logged at `warning`
    with the return code, and - mirroring `browser.py`'s `_explain`, which picks
    the last non-empty stderr line as `detail` - appended to `on_failure` when
    there is anything to append, so a non-permission failure still names itself.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_secs)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ToolError(f"{name} did not finish in time") from None

    if process.returncode != 0:
        lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = lines[-1] if lines else ""
        logger.warning("%s exited %d: %s", name, process.returncode, detail or "(no stderr)")
        if on_failure is not None:
            raise ToolError(f"{on_failure} ({name} said: {detail})" if detail else on_failure)
        raise ToolError(f"{name} failed: {detail or 'unknown error'}")
    return stdout.decode("utf-8", errors="replace")


def _parse_dimensions(raw: str) -> tuple[int, int]:
    """Parse `sips -g pixelWidth -g pixelHeight` output.

    Its output is two lines shaped like `  pixelWidth: 1536`, not anything
    structured - there is no `--json` to ask for instead.
    """
    width: int | None = None
    height: int | None = None
    for line in raw.splitlines():
        key, _, value = line.strip().partition(":")
        if key == "pixelWidth":
            width = int(value.strip())
        elif key == "pixelHeight":
            height = int(value.strip())
    if width is None or height is None:
        raise ToolError("sips did not report the image's dimensions")
    return width, height


async def capture_display(
    *, long_edge: int, window_id: int | None = None, timeout_secs: float = 20.0
) -> tuple[bytes, int, int]:
    """Capture the display (or one window) as a JPEG, downscaled to `long_edge`.

    Returns `(jpeg_bytes, width, height)`. Raises `ToolError` on non-Darwin,
    missing tools, a Screen Recording permission denial, or a timeout.
    """
    if platform.system() != "Darwin":
        raise ToolError("I can only capture the screen on macOS")
    screencapture = shutil.which("screencapture")
    sips = shutil.which("sips")
    if screencapture is None or sips is None:
        raise ToolError("screencapture or sips is not available, so I cannot capture the screen")

    with tempfile.TemporaryDirectory() as scratch:
        path = str(Path(scratch) / "capture.jpg")

        # A Screen Recording (TCC) denial has no error of its own - it shows up as
        # `screencapture` exiting non-zero ("could not create image from display",
        # measured on this machine) or, just as often, exiting 0 having written
        # nothing. Both signals get the same owner-facing sentence.
        capture_argv = SCREENCAPTURE_ARGS(path, window_id)
        capture_argv[0] = screencapture
        await _run("screencapture", capture_argv, timeout_secs=timeout_secs, on_failure=TCC_HINT)

        output = Path(path)
        size = await asyncio.to_thread(lambda: output.stat().st_size if output.exists() else 0)
        if size == 0:
            raise ToolError(TCC_HINT)

        resize_argv = SIPS_RESIZE_ARGS(path, long_edge)
        resize_argv[0] = sips
        await _run("sips", resize_argv, timeout_secs=timeout_secs)

        dims_argv = [sips, "-g", "pixelWidth", "-g", "pixelHeight", path]
        dims_raw = await _run("sips", dims_argv, timeout_secs=timeout_secs)
        width, height = _parse_dimensions(dims_raw)

        jpeg_bytes = await asyncio.to_thread(output.read_bytes)

    return jpeg_bytes, width, height
