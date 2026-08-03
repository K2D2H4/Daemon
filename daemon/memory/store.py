"""SQLite access.

This database is a **rebuildable index**, never an original: markdown under
`memory/` is the source of truth (docs/CONTRACTS.md non-negotiable 1). The rule
that follows from that, and the one to keep in mind when extending this module:
nothing may live here that cannot be reconstructed from the markdown, except the
provenance columns, which exist precisely because a model must not be able to
write them in prose (non-negotiable 3).

M1a touches `messages` only. The other tables in schema.sql belong to later
milestones and are created up front so those milestones need no migration.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from daemon.memory.base import LoggedMessage
from daemon.memory.log import utc_iso

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1

_INSERT_MESSAGE = """
INSERT INTO messages
    (ts, role, content, origin, session_kind, modality, channel, sender_id, log_file)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Store:
    """Owns one sqlite connection.

    Why the methods are synchronous even though every caller is async: each one
    is a single indexed insert or a LIMIT-ed index scan against a local file -
    tens of microseconds, less than the cost of an executor round trip - and a
    sqlite3 connection is bound to the thread that created it unless we take on
    cross-thread coordination we do not need at one user's message rate. When a
    genuinely slow statement appears later (an FTS rebuild, a full re-index from
    the markdown), that one call gets `asyncio.to_thread`; the class does not
    need to become async for it.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path: Path) -> Store:
        """Connect, creating and migrating the file if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(sqlite3.connect(path))
        store.apply_schema()
        return store

    def close(self) -> None:
        self.conn.close()

    # --- schema -------------------------------------------------------------

    def apply_schema(self) -> None:
        """Idempotent: schema.sql is all `IF NOT EXISTS`, so this also serves as
        the rebuild path after the sqlite file has been thrown away."""
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        # WAL is durable in the file header, but foreign_keys is per-connection,
        # so the PRAGMA inside schema.sql only covers the connection that ran it.
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.schema_version() is None:
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utc_iso(datetime.now(UTC))),
            )
            self.conn.commit()

    def schema_version(self) -> int | None:
        row = self.conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        return None if row["version"] is None else int(row["version"])

    # --- messages -----------------------------------------------------------

    def insert_message(self, message: LoggedMessage, *, log_file: str) -> int:
        """Mirror one already-written markdown record. `log_file` points back at
        the original so a row can always be traced to the file it came from."""
        cursor = self.conn.execute(
            _INSERT_MESSAGE,
            (
                utc_iso(message.ts),
                message.role,
                message.content,
                message.origin,
                message.session_kind,
                message.modality,
                message.channel,
                message.sender_id,
                log_file,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        """The last `limit` messages, oldest first - i.e. in reading order, ready
        to become a context window. `id` breaks ties because several messages can
        share a whole-second timestamp."""
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY ts DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        rows.reverse()
        return rows
