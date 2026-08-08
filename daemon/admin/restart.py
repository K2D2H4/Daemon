"""Self-restart: how the admin makes a settings change take effect.

docs/design decision 3: settings apply through a restart, not a runtime hot
reload. The daemon writes `.env`, then exits gracefully, and the supervisor that
`daemon/service.py` installed - launchd's `KeepAlive` on macOS, systemd's
`Restart=always` on Linux - brings it straight back up on the new config.

So the admin only ever exits when something will revive it. `is_supervised`
answers that from the one signal that is about *this* running process rather than
about a unit file on disk: launchd and systemd each stamp a marker into the
environment of a process they started. Asking `launchctl print` for our own pid
would be a subprocess on the request path for a question the environment already
answers, and it would say "a job is installed" rather than "I am that job".
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Mapping

logger = logging.getLogger(__name__)

EXIT_DELAY_SECONDS = 0.25
"""Long enough for the HTTP response to flush before the signal lands. The
restart endpoint has already returned by the time this fires, so the browser
gets its "applying…" acknowledgement rather than a dropped connection."""


def is_supervised(environ: Mapping[str, str] | None = None) -> bool:
    """Is this process running under launchd or a systemd user unit?

    - launchd exports `XPC_SERVICE_NAME` naming the job it started (our label is
      `ai.daemon.<label>`; see daemon/service.py).
    - systemd sets `INVOCATION_ID` (and `JOURNAL_STREAM`) for a unit it started.

    A false negative is the safe direction: it turns the restart button into
    "install as a service first" instead of exiting a process nothing will revive.
    """
    env = os.environ if environ is None else environ
    xpc = env.get("XPC_SERVICE_NAME", "")
    if xpc and xpc != "0" and not xpc.startswith("com.apple."):
        return True
    return bool(env.get("INVOCATION_ID") or env.get("JOURNAL_STREAM"))


def schedule_exit() -> None:
    """Ask the running server to shut down gracefully, shortly.

    A SIGTERM rather than `sys.exit`: uvicorn installs a handler that runs the
    FastAPI lifespan teardown - closing the channel, the sqlite handle and the MCP
    subprocesses - so the revived process starts from a clean shutdown rather than
    a half-closed one. Scheduled on the loop so the endpoint's response is sent
    first; only reached when `is_supervised` said something will bring us back.
    """
    loop = asyncio.get_event_loop()
    loop.call_later(EXIT_DELAY_SECONDS, _raise_sigterm)
    logger.info("admin: graceful restart requested; exiting in %.2fs", EXIT_DELAY_SECONDS)


def _raise_sigterm() -> None:
    signal.raise_signal(signal.SIGTERM)
