"""The `MemoryWriter` implementation: markdown first, sqlite second.

That order is the contract, not an optimisation (docs/CONTRACTS.md
non-negotiable 1):

  * markdown write fails  -> nothing is mirrored, the error propagates. A row
    pointing at a record that does not exist would be worse than a lost turn.
  * sqlite mirror fails   -> the error still propagates, but the markdown stays.
    The user's words survive; the index is rebuildable from them.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from daemon.memory import log
from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store


class FileMemoryWriter:
    """Records conversation to `{data_dir}/memory/log/` with a sqlite mirror."""

    def __init__(self, data_dir: Path, store: Store) -> None:
        self._data_dir = data_dir
        self._store = store

    async def record(self, message: LoggedMessage) -> None:
        # Normalised once, here, so the markdown and its mirror hold byte-identical
        # text: the blank line between records means the log format cannot carry
        # surrounding whitespace, and a mirror that disagrees with the original is
        # a mirror nobody can verify.
        message = replace(message, content=message.content.strip())
        log_file = await log.append(self._data_dir, message)
        # Kept so the caller can index this exact row. Reading it back out of
        # the mirror instead needed "newest by timestamp", and user rows carry a
        # channel timestamp while assistant rows carry our own clock - so a user
        # who sent a second message while the model was still thinking pointed the
        # lookup at the previous reply, and that utterance was never embedded.
        # Observed happening inside a *passing* test.
        self.last_inserted_id = self._store.insert_message(
            message, log_file=log_file, external_id=message.external_id
        )

    async def seen(self, channel: str, external_id: str) -> bool:
        return self._store.seen_external(channel, external_id)

    async def recent(self, limit: int = 20) -> list[LoggedMessage]:
        return [_from_row(row) for row in self._store.recent(limit)]


def _from_row(row: sqlite3.Row) -> LoggedMessage:
    return LoggedMessage(
        ts=log.from_iso(row["ts"]),
        role=row["role"],
        content=row["content"],
        origin=row["origin"],
        session_kind=row["session_kind"],
        modality=row["modality"],
        channel=row["channel"],
        sender_id=row["sender_id"],
    )
