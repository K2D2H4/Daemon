"""Whether *this process* is holding the microphone.

`presence.py` needs to subtract our own hold from the CoreAudio input probe, and
it cannot ask `daemon/voice/audio.py` to find out: importing the voice layer into
presence means a text-only install without PortAudio cannot read presence at all.
`daemon/config.py` duplicates the wake defaults for the same reason, and states
it - the cost of the copy is one line, the cost of the import is that the module
stops loading.

So the audio layer *tells* this module, and presence *asks* it. Neither imports
the other.

A counter rather than a flag: a wake listener and a voice session can hold the
device at the same time, and a flag would report the microphone free the moment
the shorter one ended. Not thread-safe by design - everything that touches it
runs on the one event loop (CONTRACTS non-negotiable 9), and a lock here would
be a claim about concurrency this process does not have.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_holds = 0


def held() -> bool:
    """Whether this process currently holds the microphone."""
    return _holds > 0


@contextmanager
def hold() -> Iterator[None]:
    """Mark the microphone as held for the duration of the block.

    Reentrant, and the release is in a `finally` so a stream that dies mid-read
    does not leave the daemon permanently convinced it is on a call - which would
    silence the speaker route for the rest of the process's life.
    """
    global _holds
    _holds += 1
    try:
        yield
    finally:
        _holds -= 1
