"""SQLite access.

This database is a **rebuildable index**, never an original: markdown under
`memory/` is the source of truth (docs/CONTRACTS.md non-negotiable 1). The rule
that follows from that, and the one to keep in mind when extending this module:
nothing may live here that cannot be reconstructed from the markdown, except the
provenance columns, which exist precisely because a model must not be able to
write them in prose (non-negotiable 3).

M1a touches `messages` only; M1b adds `embeddings` and the FTS5 read path. The
other tables in schema.sql belong to later milestones and are created up front so
those milestones need no migration.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from daemon.fs import secure_dir, secure_file
from daemon.memory.base import LoggedMessage
from daemon.memory.log import utc_iso

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 4

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
        # v3 and v4 add only new tables, which schema.sql creates on its own.
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

    # --- channel pairing ----------------------------------------------------
    # The one thing in this file that is NOT rebuildable from the markdown: these
    # rows *are* the allowlist (schema.sql). Losing them costs a pairing round,
    # not user data, which is why the file is still an index in spirit.
    #
    # Every timestamp here is written with `utc_iso` and every comparison happens
    # against a value written the same way. That is load-bearing: mixing it with
    # a millisecond form would break the lexicographic ordering the expiry DELETE
    # relies on, since '.' sorts before 'Z'.

    def is_allowed(self, channel: str, sender_id: str) -> bool:
        """Whether this sender has been approved. Numeric id only - the caller
        must never pass a username or display name (see channels/base.py)."""
        row = self.conn.execute(
            "SELECT 1 FROM channel_pairing "
            "WHERE channel = ? AND sender_id = ? AND state = 'approved' LIMIT 1",
            (channel, sender_id),
        ).fetchone()
        return row is not None

    def has_owner(self, channel: str) -> bool:
        """Whether first-run onboarding is over for this channel.

        Distinguishes "nobody has ever been approved" from "an owner exists and
        this is a guest request", which is what lets the pending cap be generous
        exactly once - see channels/pairing.BOOTSTRAP_MAX_PENDING.
        """
        row = self.conn.execute(
            "SELECT 1 FROM channel_pairing WHERE channel = ? AND is_owner = 1 LIMIT 1",
            (channel,),
        ).fetchone()
        return row is not None

    def create_pairing(
        self,
        channel: str,
        sender_id: str,
        code: str,
        *,
        created_at: datetime,
        expires_at: datetime,
    ) -> bool:
        """Record a pending request. False when the code collided with a live one
        (or the sender already has a row), so the caller can retry with a new code
        instead of an IntegrityError escaping into the inbound poll loop."""
        try:
            self.conn.execute(
                "INSERT INTO channel_pairing "
                "(channel, sender_id, code, state, created_at, expires_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?)",
                (channel, sender_id, code, utc_iso(created_at), utc_iso(expires_at)),
            )
        except sqlite3.IntegrityError:
            self.conn.rollback()
            return False
        self.conn.commit()
        return True

    def pairing_by_code(self, channel: str, code: str) -> sqlite3.Row | None:
        """The pending row holding this code. Scoped by channel, and approved rows
        have their code cleared, so a spent code matches nothing."""
        return self.conn.execute(
            "SELECT * FROM channel_pairing WHERE channel = ? AND code = ?",
            (channel, code),
        ).fetchone()

    def pending_pairings(self, channel: str) -> list[sqlite3.Row]:
        """Requests awaiting approval, oldest first. Includes rows whose deadline
        has passed - call `expire_pairings` first."""
        return self.conn.execute(
            "SELECT * FROM channel_pairing WHERE channel = ? AND state = 'pending' "
            "ORDER BY created_at, sender_id",
            (channel,),
        ).fetchall()

    def approve_pairing(
        self, channel: str, sender_id: str, *, approved_at: datetime
    ) -> bool | None:
        """Approve a pending sender permanently. Returns whether this approval
        bootstrapped the owner, or None if there was nothing pending to approve.

        The owner check and the write are one transaction: two approvals racing on
        a "is there an owner yet" read would otherwise both see none and mint two
        owners. `code` is cleared so it cannot be replayed.
        """
        with self.conn:  # commit on success, roll back on error
            owned = self.conn.execute(
                "SELECT 1 FROM channel_pairing WHERE channel = ? AND is_owner = 1 LIMIT 1",
                (channel,),
            ).fetchone()
            is_owner = owned is None
            cursor = self.conn.execute(
                "UPDATE channel_pairing SET state = 'approved', code = NULL, expires_at = NULL, "
                "approved_at = ?, is_owner = ? WHERE channel = ? AND sender_id = ? "
                "AND state = 'pending'",
                (utc_iso(approved_at), 1 if is_owner else 0, channel, sender_id),
            )
        return is_owner if cursor.rowcount else None

    def expire_pairings(self, channel: str, *, now: datetime) -> int:
        """Delete pending requests whose deadline has passed, returning how many.

        Deleting rather than flagging: `state` admits only 'pending'/'approved'
        (schema.sql is frozen), and a row that is gone is what lets the caller
        read "a row exists" as "this sender has already been sent a code".
        """
        cursor = self.conn.execute(
            "DELETE FROM channel_pairing WHERE channel = ? AND state = 'pending' "
            "AND expires_at IS NOT NULL AND expires_at <= ?",
            (channel, utc_iso(now)),
        )
        self.conn.commit()
        return cursor.rowcount

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

    def messages_by_ids(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        """Rows for a set of ids, keyed by id. The vector lane produces ids and
        needs the text and timestamp back; returning a mapping keeps the caller
        from re-deriving the order it already knows."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def mark_recalled(self, ids: Sequence[int]) -> None:
        """Flag rows that recall put in front of the model.

        docs/PLAN.md 4.2 hygiene rule 2: reflection must not re-extract what was
        recalled, or its own injected context becomes new evidence and the loop
        amplifies itself. Note that this UPDATE fires `messages_au`, which rewrites
        the row's FTS entry with identical content - correct, just not free.
        """
        if not ids:
            return
        self.conn.executemany(
            "UPDATE messages SET recalled = 1 WHERE id = ? AND recalled = 0",
            [(int(i),) for i in ids],
        )
        self.conn.commit()

    # --- keyword search -----------------------------------------------------

    def search_fts(self, match_query: str, limit: int) -> list[tuple[sqlite3.Row, float]]:
        """FTS5 hits with their bm25 rank, best first.

        `match_query` must already be valid FTS5 *query syntax*, not raw user
        text - FTS5 parses bound values as syntax, so quoting is the caller's job
        (see `recall.fts_query`). A malformed query is an OperationalError, which
        is caught here rather than allowed to kill a conversation turn: keyword
        recall going quiet is recoverable, a raised exception mid-turn is not.
        """
        try:
            rows = self.conn.execute(
                "SELECT m.*, bm25(messages_fts) AS rank FROM messages_fts "
                "JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.warning("recall: sqlite rejected FTS query %r", match_query)
            return []
        return [(row, float(row["rank"])) for row in rows]

    # --- embeddings ---------------------------------------------------------

    def upsert_embedding(self, message_id: int, model: str, vector: Sequence[float]) -> None:
        """Store one vector as a raw float32 BLOB.

        Normalisation belongs to the embedder (llm/base.py), not here; this method
        stores what it is given. `load_embeddings` re-normalises on the way out,
        which is where the dot-product-as-cosine invariant is actually relied on.
        """
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"embedding for message {message_id} has shape {array.shape}")
        self.conn.execute(
            "INSERT INTO embeddings (message_id, model, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (message_id) DO UPDATE SET model = excluded.model, "
            "dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at",
            (
                int(message_id),
                model,
                int(array.size),
                array.tobytes(),
                utc_iso(datetime.now(UTC)),
            ),
        )
        self.conn.commit()

    def load_embeddings(self, model: str) -> tuple[list[int], np.ndarray]:
        """Every vector for one model as (message ids, (N, dim) matrix).

        Rows are L2-normalised here so recall can score with a plain dot product.
        Rows whose width disagrees with the newest row are dropped: re-tagging a
        model without renaming it would otherwise mix two vector spaces into one
        matrix, and `frombuffer` would fail on the reshape rather than degrade.
        """
        newest = self.conn.execute(
            "SELECT dim FROM embeddings WHERE model = ? ORDER BY message_id DESC LIMIT 1",
            (model,),
        ).fetchone()
        if newest is None:
            return [], np.zeros((0, 0), dtype=np.float32)

        dim = int(newest["dim"])
        rows = self.conn.execute(
            "SELECT message_id, vector FROM embeddings WHERE model = ? AND dim = ? "
            "ORDER BY message_id",
            (model, dim),
        ).fetchall()

        ids = [int(row["message_id"]) for row in rows]
        matrix = np.zeros((len(ids), dim), dtype=np.float32)
        for index, row in enumerate(rows):
            matrix[index] = np.frombuffer(row["vector"], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return ids, matrix

    def messages_without_embedding(self, model: str, limit: int) -> list[sqlite3.Row]:
        """Messages this model has not embedded yet, oldest first. The backfill
        path after a re-index or a model change."""
        return self.conn.execute(
            "SELECT m.id, m.content FROM messages m "
            "LEFT JOIN embeddings e ON e.message_id = m.id AND e.model = ? "
            # Newest first. Recency decay gives recent messages the highest
            # scores, so a backfill that stops early must have covered those
            # rather than the oldest history.
            "WHERE e.message_id IS NULL ORDER BY m.id DESC LIMIT ?",
            (model, limit),
        ).fetchall()

    def delete_embeddings(self, model: str) -> int:
        """Drop one model's vectors. Safe by construction - the vectors are an
        index, re-derivable from the markdown - and the way a model change is
        rolled out without mixing vector spaces."""
        cursor = self.conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        self.conn.commit()
        return cursor.rowcount
