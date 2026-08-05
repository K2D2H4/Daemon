"""The persona, assembled from the files that own their halves.

`seed.md` is human-owned and `learned.md` is the daemon's own (docs/CONTRACTS.md 5);
this is the one place they are joined, so M4 - which is what makes the second file
exist - changes this file and nothing that reads it. Today only the seed exists and
this is a read; the point is where the read lives, not how much it does yet.

Re-read on every call rather than cached at startup. `seed.md` is the file the owner
edits to change how the daemon talks to them, and that edit has to land on the next
turn without a restart (docs/PLAN.md 5.1). Caching it is the obvious optimisation
and it passes the entire suite bar one test, which is why that test exists.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SEED = "seed.md"
"""Human-owned. **Code must never write to it** (docs/CONTRACTS.md 5) - that
asymmetry is the anchor that stops an evolving personality collapsing into
agreement, so this module reads and offers no way to write."""


async def load(data_dir: Path) -> str:
    """The persona as prompt text, or an empty string if there is none.

    Never raises. A persona that cannot be read costs the daemon its voice for a
    turn; raising here would cost the user the turn itself, and the log clock is the
    thing that cannot be caught up later (docs/PLAN.md 8.1).
    """
    return await asyncio.to_thread(_read, Path(data_dir) / "persona" / SEED)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # The ordinary state before onboarding, and not worth a line in the log.
        return ""
    except OSError:
        logger.exception("could not read %s", path)
        return ""
