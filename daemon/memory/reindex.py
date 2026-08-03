"""Rebuild the sqlite mirror from the markdown.

docs/CONTRACTS.md non-negotiable 1 says deleting the database must never lose
user data. Until this module existed that was an aspiration: nothing read the
markdown back, so a mirror write that failed after the markdown succeeded left a
turn that the model would never see again, with no way to notice and no way to
repair. Now the claim is executable.

Two situations, one mechanism:

  * A mirror write failed, or the process died between the two writes. The
    markdown has more records for a day than the mirror has rows.
  * The database was thrown away entirely. Every day is short by everything.

The log is append-only and ordered, so "short by" always means a missing tail:
compare counts per file and insert the records past the ones already there. That
is idempotent without needing a uniqueness key over the content - which would be
wrong anyway, since the same message legitimately repeats ("ㅇㅇ" twice in one
second is not a duplicate).

What cannot be recovered: provenance. The markdown deliberately carries none of
it (it lives in columns so a model cannot forge it in prose), so rebuilt rows get
defaults and are flagged `reindexed = 1`. Reflection has to be able to tell an
inferred origin from an observed one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from daemon.memory import log
from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store

logger = logging.getLogger(__name__)

REBUILT_CHANNEL = "unknown"
"""Rebuilt rows cannot say which channel they arrived on."""


def reindex(data_dir: Path, store: Store) -> int:
    """Insert every markdown record the mirror is missing. Returns the count."""
    log_dir = data_dir / log.LOG_SUBDIR
    if not log_dir.exists():
        return 0

    inserted = 0
    for path in sorted(log_dir.glob("*.md")):
        records = log.read(path)
        if not records:
            continue
        relative = path.relative_to(data_dir).as_posix()
        mirrored = store.count_for_log_file(relative)
        for record in records[mirrored:]:
            store.insert_message(
                LoggedMessage(
                    ts=record.ts,
                    role=record.role,
                    content=record.content,
                    # Inferred, not observed: the user's own turns are the user's,
                    # and anything the assistant said came from the agent. Both
                    # are marked reindexed so nothing downstream mistakes these
                    # for facts recorded at the time.
                    origin="owner" if record.role == "user" else "agent",
                    session_kind="interactive",
                    modality="text",
                    channel=REBUILT_CHANNEL,
                ),
                log_file=relative,
                reindexed=True,
            )
            inserted += 1
    if inserted:
        logger.warning(
            "reindexed %d message(s) the mirror was missing; provenance for those "
            "rows is inferred, not observed",
            inserted,
        )
    return inserted
