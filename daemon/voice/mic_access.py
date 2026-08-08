"""macOS microphone (TCC) access: the one prompt, and the status read.

In the Apple-only corner next to apple_speech.py, guarded the same way: AVFoundation
is pyobjc, exists only on macOS, and a text-only or Linux install must still import
the package. Everything is caught — an absent or broken framework reads as
"unavailable", never an exception into a caller that has to outlive it.

Why this exists (spec §1, §4.3): a launchd-spawned bare Python cannot obtain the
mic grant (no TCC prompt is possible headless). The daemon is wrapped in a thin
Daemon.app whose code-signing identity *is* the grant. PortAudio's HAL access never
pops the prompt — it just returns silence when ungranted — so the prompt has to come
from an explicit AVCaptureDevice.requestAccess call. That call is a real prompt under
the .app foreground (`daemon request-mic`) and a cached no-op headless.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# AVAuthorizationStatus (AVFoundation): the integer the OS returns.
_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
_PUMP_SLICE_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class Frameworks:
    """Injected in tests so no OS media service is touched (mirrors apple_speech)."""

    av: Any  # AVFoundation
    foundation: Any  # Foundation


def _load(frameworks: Frameworks | None) -> Frameworks:
    if frameworks is not None:
        return frameworks
    import AVFoundation
    import Foundation

    return Frameworks(av=AVFoundation, foundation=Foundation)


def microphone_authorization_status(*, frameworks: Frameworks | None = None) -> str:
    """The current TCC decision, asking nothing of the user. Never prompts."""
    try:
        fw = _load(frameworks)
        status = fw.av.AVCaptureDevice.authorizationStatusForMediaType_(fw.av.AVMediaTypeAudio)
        return _STATUS.get(int(status), "unavailable")
    except Exception:
        logger.debug("mic access: no AVFoundation here", exc_info=True)
        return "unavailable"


def request_microphone_access(
    *, timeout: float = 12.0, frameworks: Frameworks | None = None
) -> str:
    """Claim the mic grant. Prompts only when the decision is still open and a GUI
    session can show it; otherwise returns the cached decision immediately.

    Pumps the runloop in short slices until the completion handler fires or the
    deadline passes — the handler is what the prompt (or the cached decision)
    resolves to. Headless-and-ungranted it returns "not_determined" without hanging
    past `timeout`.
    """
    try:
        fw = _load(frameworks)
        av = fw.av
        current = _STATUS.get(
            int(av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeAudio))
        )
        # Already decided (authorized/denied/restricted): don't pump for nothing.
        if current in ("authorized", "denied", "restricted"):
            return current

        done: dict[str, bool] = {}
        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            av.AVMediaTypeAudio, lambda granted: done.__setitem__("g", bool(granted))
        )
        loop = fw.foundation.NSRunLoop.currentRunLoop()
        deadline = time.monotonic() + timeout
        while "g" not in done and time.monotonic() < deadline:
            loop.runMode_beforeDate_(
                "kCFRunLoopDefaultMode",
                fw.foundation.NSDate.dateWithTimeIntervalSinceNow_(_PUMP_SLICE_SECONDS),
            )
        if "g" not in done:
            return "not_determined"
        return "authorized" if done["g"] else "denied"
    except Exception:
        logger.debug("mic access: request failed", exc_info=True)
        return "unavailable"
