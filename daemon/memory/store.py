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

from daemon.fs import secure_dir, secure_file
from daemon.memory.base import LoggedMessage
from daemon.memory.log import utc_iso

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 2

_INSERT_MESSAGE = """
INSERT INTO messages
    (ts, role, content, origin, session_kind, modality, channel, sender_id, log_file,
     external_id, reindexed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        secure_dir(path.parent)
        store = cls(sqlite3.connect(path))
        store.apply_schema()
        # sqlite creates the db, -wal and -shm itself, so they have to be
        # tightened after the fact. -wal holds recent rows in the clear, so
        # leaving it readable would defeat locking down the db alone.
        for suffix in ("", "-wal", "-shm"):
            secure_file(path.with_name(path.name + suffix))
        return store

    def close(self) -> None:
        self.conn.close()

    # --- schema -------------------------------------------------------------

    def apply_schema(self) -> None:
        """Idempotent: schema.sql is all `IF NOT EXISTS`, so this also serves as
        the rebuild path after the sqlite file has been thrown away.

        `IF NOT EXISTS` is also why an existing file needs explicit migration: a
        table that is already there is skipped, so a column added later would
        never appear. And a file written by a *newer* version has to be refused
        rather than opened optimistically - recording the version without ever
        comparing it would let new code write into an old shape and believe it
        was current.
        """
        found = self.schema_version()
        if found is not None and found > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema is v{found}, this build understands v{SCHEMA_VERSION}. "
                "Refusing to open it: markdown under memory/ is the source of truth, "
                "so a newer file is safer to leave alone than to write into."
            )
        # Migrate *before* running schema.sql, not after: schema.sql creates an
        # index over external_id, and on a pre-existing table that column does
        # not exist yet, so the whole script would fail before the ALTER ran.
        if found is not None and found < SCHEMA_VERSION:
            self._migrate(found)
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        # WAL is durable in the file header, but foreign_keys is per-connection,
        # so the PRAGMA inside schema.sql only covers the connection that ran it.
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Default 5s: with a second writer on the file (a backup tool, the sqlite
        # CLI, a stray instance) a single insert would block the whole event loop
        # for that long, stalling the inbound poll with it.
        self.conn.execute("PRAGMA busy_timeout = 300")
        if found is None:
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utc_iso(datetime.now(UTC))),
            )
            self.conn.commit()

    def _migrate(self, found: int) -> None:
        """Bring an older file up to SCHEMA_VERSION, additively."""
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(messages)")}
        if found < 2:
            for column, ddl in (
                ("external_id", "ALTER TABLE messages ADD COLUMN external_id TEXT"),
                (
                    "reindexed",
                    "ALTER TABLE messages ADD COLUMN reindexed INTEGER NOT NULL DEFAULT 0",
                ),
            ):
                if column not in existing:
                    self.conn.execute(ddl)
            # The index over external_id is left to schema.sql, which runs next
            # now that the column exists.
        self.conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utc_iso(datetime.now(UTC))),
        )
        self.conn.commit()

    # --- channel cursor -----------------------------------------------------

    def load_cursor(self, channel: str) -> int | None:
        row = self.conn.execute(
            "SELECT offset_at FROM channel_cursor WHERE channel = ?", (channel,)
        ).fetchone()
        return None if row is None else int(row["offset_at"])

    def save_cursor(self, channel: str, offset: int) -> None:
        self.conn.execute(
            "INSERT INTO channel_cursor (channel, offset_at, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (channel) DO UPDATE SET offset_at = excluded.offset_at, "
            "updated_at = excluded.updated_at",
            (channel, offset, utc_iso(datetime.now(UTC))),
        )
        self.conn.commit()

    def seen_external(self, channel: str, external_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM messages WHERE channel = ? AND external_id = ? LIMIT 1",
            (channel, external_id),
        ).fetchone()
        return row is not None

    def schema_version(self) -> int | None:
        """None for a database that has no version recorded yet - including a
        brand new file, where the table itself does not exist so the version has
        to be read before schema.sql has had a chance to create it."""
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if exists is None:
            return None
        row = self.conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        return None if row["version"] is None else int(row["version"])

    # --- messages -----------------------------------------------------------

    def insert_message(
        self,
        message: LoggedMessage,
        *,
        log_file: str,
        external_id: str | None = None,
        reindexed: bool = False,
    ) -> int:
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
                external_id,
                1 if reindexed else 0,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def count_for_log_file(self, log_file: str) -> int:
        """How many rows this markdown file has been mirrored into. The log is
        append-only, so the mirror being short always means a missing tail."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE log_file = ?", (log_file,)
        ).fetchone()
        return int(row["n"])

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        """The last `limit` messages, oldest first - i.e. in reading order, ready
        to become a context window. `id` breaks ties because several messages can
        share a whole-second timestamp."""
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY ts DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        rows.reverse()
        return rows
