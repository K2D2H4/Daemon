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

import json
import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from daemon.fs import secure_dir, secure_file
from daemon.memory.base import LoggedMessage
from daemon.memory.log import from_iso, local_date, utc_iso

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 7

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
        # v3 to v7 add only new tables, which schema.sql creates on its own.
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

    # --- tool use -----------------------------------------------------------
    # Not rebuildable from the markdown either (schema.sql): an audit trail of what
    # the machine did, and the standing approvals that let it. Same timestamp
    # discipline as pairing above - every value written with `utc_iso`, so the
    # expiry comparison stays a lexicographic one.

    def record_tool_call(
        self,
        *,
        tool: str,
        arguments: str,
        preview: str,
        verdict: str,
        mode: str,
        reason: str,
        origin: str,
        channel: str,
        sender_id: str | None,
        ran: bool = False,
        ok: bool | None = None,
        output_excerpt: str | None = None,
        elapsed_ms: int | None = None,
        now: datetime | None = None,
    ) -> int:
        """Append one audit row, returning its id.

        Written for refused calls as well as executed ones. A denial that leaves no
        trace is the same as no policy at all when someone later asks what happened.
        """
        cursor = self.conn.execute(
            "INSERT INTO tool_calls (ts, tool, arguments, preview, verdict, mode, reason, "
            "origin, channel, sender_id, ran, ok, output_excerpt, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                utc_iso(now or datetime.now(UTC)),
                tool,
                arguments,
                preview,
                verdict,
                mode,
                reason,
                origin,
                channel,
                sender_id,
                1 if ran else 0,
                None if ok is None else (1 if ok else 0),
                output_excerpt,
                elapsed_ms,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def recent_tool_calls(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def tool_calls_for_day(self, date: str) -> list[sqlite3.Row]:
        """One local day's tool calls, oldest first - reflection's second input.

        `tool_calls` carries no `log_file`, only a UTC `ts`, so the local day has
        to be recovered rather than looked up. A `BETWEEN` over `ts` would be
        wrong by up to a day in either direction: 23:00Z is already tomorrow
        morning in Seoul. So the SQL window is deliberately loose - a day either
        side, which covers every real offset - and `local_date` decides
        membership, the same function that named the log file this day's messages
        went into.

        Refused and failed calls come back too. What to do with them is the
        caller's question: they carry no output to remember but they are part of
        what the day looked like.
        """
        day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
        rows = self.conn.execute(
            "SELECT * FROM tool_calls WHERE ts >= ? AND ts < ? ORDER BY ts ASC, id ASC",
            (utc_iso(day - timedelta(days=1)), utc_iso(day + timedelta(days=2))),
        ).fetchall()
        return [row for row in rows if local_date(from_iso(row["ts"])) == date]

    def create_tool_approval(
        self,
        *,
        code: str,
        channel: str,
        sender_id: str,
        tool: str,
        arguments: str,
        fingerprint: str,
        preview: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> bool:
        """Record a pending approval. False when the code collided with a live one,
        so the caller retries with a fresh code rather than letting an
        IntegrityError escape into the conversation loop (as `create_pairing` does).
        """
        try:
            self.conn.execute(
                "INSERT INTO tool_approvals (code, channel, sender_id, tool, arguments, "
                "fingerprint, preview, state, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    code,
                    channel,
                    sender_id,
                    tool,
                    arguments,
                    fingerprint,
                    preview,
                    utc_iso(created_at),
                    utc_iso(expires_at),
                ),
            )
        except sqlite3.IntegrityError:
            self.conn.rollback()
            return False
        self.conn.commit()
        return True

    def count_pending_tool_approvals(self, *, now: datetime) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM tool_approvals WHERE state = 'pending' "
            "AND expires_at > ?",
            (utc_iso(now),),
        ).fetchone()
        return int(row["n"])

    def tool_approval(self, code: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM tool_approvals WHERE code = ?", (code,)
        ).fetchone()

    def pending_tool_approvals(self, *, now: datetime) -> list[sqlite3.Row]:
        """Live pending approvals, oldest first."""
        return self.conn.execute(
            "SELECT * FROM tool_approvals WHERE state = 'pending' AND expires_at > ? "
            "ORDER BY created_at, code",
            (utc_iso(now),),
        ).fetchall()

    def spend_tool_approval(
        self, code: str, *, sender_id: str, denied: bool, now: datetime
    ) -> sqlite3.Row | None:
        """Claim a pending, unexpired approval, returning the row that was claimed.

        Two statements in one transaction, and the `state = 'pending'` in the SELECT
        is what makes it single-use: once the UPDATE has landed, a second attempt
        selects nothing. (An earlier version of this docstring claimed "one statement",
        which was simply wrong about its own mechanism.) `sender_id` is part of the
        WHERE clause because a guest who was allowlisted for conversation must not be
        able to spend an approval addressed to the owner.
        """
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM tool_approvals WHERE code = ? AND sender_id = ? "
                "AND state = 'pending' AND expires_at > ?",
                (code, sender_id, utc_iso(now)),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE tool_approvals SET state = ?, decided_at = ? WHERE code = ?",
                ("denied" if denied else "spent", utc_iso(now), code),
            )
        return row

    def expire_tool_approvals(self, *, now: datetime) -> int:
        """Delete pending approvals past their deadline, returning how many."""
        cursor = self.conn.execute(
            "DELETE FROM tool_approvals WHERE state = 'pending' AND expires_at <= ?",
            (utc_iso(now),),
        )
        self.conn.commit()
        return cursor.rowcount

    def add_tool_allowlist_entry(self, tool: str, pattern: str, *, now: datetime) -> None:
        """Grant a standing approval. Idempotent - re-granting the same pattern is
        what a user does when they have forgotten they already did."""
        self.conn.execute(
            "INSERT INTO tool_allowlist (tool, pattern, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT (tool, pattern) DO NOTHING",
            (tool, pattern, utc_iso(now)),
        )
        self.conn.commit()

    def tool_allowlist(self, tool: str) -> list[str]:
        return [
            str(row["pattern"])
            for row in self.conn.execute(
                "SELECT pattern FROM tool_allowlist WHERE tool = ? ORDER BY pattern", (tool,)
            )
        ]

    def all_tool_allowlist(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM tool_allowlist ORDER BY tool, pattern"
        ).fetchall()

    def remove_tool_allowlist_entry(self, pattern: str) -> int:
        """Revoke a standing approval, returning how many rows went.

        By pattern across every tool rather than by (tool, pattern): the user typing
        this has seen `git status` in a list and should not also have to know which
        tool it belongs to.
        """
        cursor = self.conn.execute(
            "DELETE FROM tool_allowlist WHERE pattern = ?", (" ".join(pattern.split()),)
        )
        self.conn.commit()
        return cursor.rowcount

    # Per-tool grants, the other standing axis. `'*'` rather than an import of
    # `daemon.tools.policy.ANY_CHANNEL`: memory is foundation and the tool layer
    # sits above it, so importing upwards to share one character would invert the
    # layering - the same trade `config.py` takes with the voice sensitivities.

    def add_tool_grant(self, tool: str, *, channel: str = "*", now: datetime) -> None:
        """Grant a whole tool. Idempotent, for the reason
        `add_tool_allowlist_entry` is."""
        self.conn.execute(
            "INSERT INTO tool_grants (tool, channel, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT (tool, channel) DO NOTHING",
            (tool, channel, utc_iso(now)),
        )
        self.conn.commit()

    def tool_grants(self, tool: str) -> list[str]:
        """Channels this tool is granted on, `'*'` meaning all of them.

        Read on every decision rather than cached: the row may be written by
        something else - a web admin, another process - into a database this one
        already has open, and a grant that needed a restart would be a setting
        that lies.
        """
        return [
            str(row["channel"])
            for row in self.conn.execute(
                "SELECT channel FROM tool_grants WHERE tool = ? ORDER BY channel", (tool,)
            )
        ]

    def all_tool_grants(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM tool_grants ORDER BY tool, channel"
        ).fetchall()

    def remove_tool_grant(self, tool: str) -> int:
        """Revoke a tool everywhere, returning how many rows went.

        By tool and not by (tool, channel), for the reason
        `remove_tool_allowlist_entry` takes the whole pattern: someone revoking a
        grant wants the tool to stop, and leaving one channel's row behind would
        be a revocation that did not revoke.
        """
        cursor = self.conn.execute("DELETE FROM tool_grants WHERE tool = ?", (tool,))
        self.conn.commit()
        return cursor.rowcount

    # --- delegated tasks - async work queue ----------------------------------

    def enqueue_task(
        self,
        *,
        request: str,
        origin: str,
        channel: str,
        sender_id: str | None,
        now: datetime | None = None,
    ) -> int:
        """Insert a queued task, returning its id."""
        cursor = self.conn.execute(
            "INSERT INTO delegated_tasks (request, status, created_ts, origin, channel, sender_id) "
            "VALUES (?, 'queued', ?, ?, ?, ?)",
            (request, utc_iso(now or datetime.now(UTC)), origin, channel, sender_id),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def claim_next_queued(self, *, now: datetime | None = None) -> sqlite3.Row | None:
        """Flip the oldest queued row to running and return it, atomically."""
        row = self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE delegated_tasks SET status='running' WHERE id=?", (row["id"],)
        )
        self.conn.commit()
        return self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE id=?", (row["id"],)
        ).fetchone()

    def mark_task_done(
        self, task_id: int, result: str, *, now: datetime | None = None
    ) -> None:
        """Mark a task as done with its result."""
        self.conn.execute(
            "UPDATE delegated_tasks SET status='done', result=?, finished_ts=? WHERE id=?",
            (result, utc_iso(now or datetime.now(UTC)), task_id),
        )
        self.conn.commit()

    def mark_task_failed(
        self, task_id: int, error: str, *, now: datetime | None = None
    ) -> None:
        """Mark a task as failed with its error message."""
        self.conn.execute(
            "UPDATE delegated_tasks SET status='failed', error=?, finished_ts=? WHERE id=?",
            (error, utc_iso(now or datetime.now(UTC)), task_id),
        )
        self.conn.commit()

    def pending_tasks(self) -> list[sqlite3.Row]:
        """Every row still queued or running, oldest first."""
        return self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE status IN ('queued', 'running') ORDER BY id"
        ).fetchall()

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

    def count_embeddings(self, model: str) -> int:
        """How many vectors this model has. A COUNT, not a load: the point of the
        number is observability, and reading it off the in-memory cache made it
        lag behind reality in the one direction that hides progress."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM embeddings WHERE model = ?", (model,)
        ).fetchone()
        return int(row["n"])

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

        **Nothing reads this flag any more** (2026-08-05). It existed for
        docs/PLAN.md 4.2 hygiene rule 2, which excluded flagged rows from
        `messages_for_day` permanently; that exclusion cost 29 of 38 messages on a
        real day and blocked no loop, so it is gone and the reasoning is written
        out in `messages_for_day`. The write stays because "recall has surfaced
        this row" is a true and cheap thing to record, and because deleting a
        column from a frozen schema is a bigger decision than this was.

        **Do not reinstate a reflection filter on it** without reading that
        docstring first: the input it removes is exactly what persona evolution
        runs on. Note also that this UPDATE fires `messages_au`, which rewrites the
        row's FTS entry with identical content - correct, just not free.
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

    # --- M2: the curated tier -----------------------------------------------

    def insert_entry(
        self,
        *,
        body: str,
        importance: int,
        trigger_phrases: Sequence[str],
        origin: str,
        session_kind: str,
        modality: str,
        now: datetime,
        supersession_key: str | None = None,
        supersedes: int | None = None,
        source_file: str | None = None,
        source_anchor: str | None = None,
        commit: bool = True,
    ) -> int:
        """Add one curated fact and retire whatever it supersedes, in ONE
        transaction.

        Atomic because the unique index on `supersession_key` only permits one
        active row per key: as two commits there is a window with no active row for
        that key, and if the second then fails the fact is gone from the mirror
        entirely. All of it or none of it.

        The order inside is forced by that same index - retire, insert, then point
        the retired row at its replacement. Inserting first raises
        `UNIQUE constraint failed` before anything has been retired, which is the
        index doing its job.

        `commit=False` leaves the transaction open so the caller can write the
        markdown *before* the mirror is durable, which is what non-negotiable 1
        requires. This connection then sees its own uncommitted rows, so the
        caller can render the file from the post-insert state without
        reimplementing the ordering. The caller owns the commit and the rollback -
        see `memory.curated.CuratedMemory.add`.

        `supersedes` names a row directly, which is how reflection retires a fact
        the model chose to update (ADR 0010). It is *additional* to the key match,
        not an alternative: a fact can both replace row 4 and carry a key row 7
        holds, and retiring only one of them would hit the unique index. Both
        retire into the new row. Already-retired ids are the caller's to reject -
        this only ever retires what is active, so a settled `superseded_by`
        pointer cannot be moved.
        """
        stamp = utc_iso(now)
        try:
            # The retire set is *read* below and acted on immediately after, so the
            # read has to be under the write lock. sqlite's legacy transaction
            # handling opens the implicit BEGIN on the first write, which left that
            # SELECT in autocommit - a check-then-act across connections, and
            # `daemon reflect` during the 04:00 pass is two connections. Measured
            # lock hold for one supersession is 0.5-2.7ms against the 300ms
            # busy_timeout set in `apply_schema`, so taking it earlier costs
            # nothing a second writer would notice.
            if not self.conn.in_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            previous = [
                int(row["id"])
                for row in self.conn.execute(
                    "SELECT id FROM memory_entries WHERE status = 'active' "
                    "AND (id = ? OR (supersession_key IS NOT NULL AND supersession_key = ?))",
                    (supersedes, supersession_key),
                ).fetchall()
            ]
            for entry_id in previous:
                self.conn.execute(
                    "UPDATE memory_entries SET status = 'retired', updated_at = ? WHERE id = ?",
                    (stamp, entry_id),
                )

            cursor = self.conn.execute(
                "INSERT INTO memory_entries "
                "(body, importance, trigger_phrases, origin, session_kind, modality,"
                " created_at, updated_at, supersession_key, source_file, source_anchor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    body,
                    importance,
                    json.dumps(list(trigger_phrases), ensure_ascii=False),
                    origin,
                    session_kind,
                    modality,
                    stamp,
                    stamp,
                    supersession_key,
                    source_file,
                    source_anchor,
                ),
            )
            new_id = int(cursor.lastrowid or 0)
            for entry_id in previous:
                self.conn.execute(
                    "UPDATE memory_entries SET superseded_by = ? WHERE id = ?",
                    (new_id, entry_id),
                )
        except Exception:
            # Discards the retire too. Without this a failed insert would leave
            # the old fact retired and nothing active for its key.
            self.conn.rollback()
            raise
        if commit:
            self.conn.commit()
        return new_id

    def active_entries(self, limit: int = 50) -> list[sqlite3.Row]:
        """The curated tier, most important first, then most recent.

        Ordered by importance rather than recency because this tier is *always*
        injected under a budget (docs/PLAN.md 4.1): when the budget truncates, it
        must drop the least important fact, not the oldest one.
        """
        return self.conn.execute(
            "SELECT * FROM memory_entries WHERE status = 'active' "
            "ORDER BY importance DESC, updated_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def count_entries(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM memory_entries WHERE status = 'active'"
        ).fetchone()
        return int(row["n"])

    # --- M2: entity notes ---------------------------------------------------

    def upsert_entity(self, *, name: str, kind: str | None, file: str, now: datetime) -> int:
        """Create or touch one entity, returning its id and counting the mention.

        `kind` is only ever filled in, never overwritten with NULL: a later pass
        that mentions someone without classifying them must not erase what an
        earlier pass worked out.
        """
        stamp = utc_iso(now)
        with self.conn:
            self.conn.execute(
                "INSERT INTO entities (name, kind, file, created_at, updated_at, mention_count) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT (name) DO UPDATE SET "
                "  kind = COALESCE(excluded.kind, entities.kind), "
                "  file = excluded.file, "
                "  updated_at = excluded.updated_at, "
                "  mention_count = entities.mention_count + 1",
                (name, kind, file, stamp, stamp),
            )
            row = self.conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
        return int(row["id"])

    def set_mention_count(self, entity_id: int, count: int) -> None:
        """Overwrite the count rather than increment it. Only the rebuild path uses
        this: the count is implied by how many dated sections the note has, so a
        rebuild that incremented would double it on every run."""
        self.conn.execute(
            "UPDATE entities SET mention_count = ? WHERE id = ?", (count, entity_id)
        )
        self.conn.commit()

    def entity_by_name(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()

    def entities(self, limit: int = 500) -> list[sqlite3.Row]:
        """Most-mentioned first - the reading order for a graph nobody curated."""
        return self.conn.execute(
            "SELECT * FROM entities ORDER BY mention_count DESC, name ASC LIMIT ?",
            (limit,),
        ).fetchall()

    def link_entities(self, src_id: int, dst_id: int) -> None:
        """Record that two entities appeared together. Undirected in meaning, so
        both directions are stored - a graph read from either end shows the edge.
        Self-links are dropped rather than rejected: the extractor naming the same
        entity twice in one note is not an error worth failing a reflection over.
        """
        if src_id == dst_id:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO entity_links (src_id, dst_id) VALUES (?, ?)",
                [(src_id, dst_id), (dst_id, src_id)],
            )

    def links_for(self, entity_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT e.* FROM entity_links l JOIN entities e ON e.id = l.dst_id "
            "WHERE l.src_id = ? ORDER BY e.name",
            (entity_id,),
        ).fetchall()

    # --- M2: observations (append-only) -------------------------------------

    def insert_observation(
        self,
        *,
        body: str,
        observed_from: str,
        now: datetime,
        modality: str = "text",
        origin: str = "agent",
        confidence: float = 0.5,
    ) -> int:
        """Append one observation about how to deal with this person.

        There is deliberately no update or delete for this table. It is the log
        clock (docs/PLAN.md 8.1): persona evolution needs weeks of accumulated
        real observations before it can be judged, and an observation that can be
        rewritten later is not evidence of anything.
        """
        cursor = self.conn.execute(
            "INSERT INTO observations "
            "(body, created_at, observed_from, modality, origin, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body, utc_iso(now), observed_from, modality, origin, confidence),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def unconsumed_observations(self, limit: int = 200) -> list[sqlite3.Row]:
        """Observations no persona rule has used yet - M4's input, oldest first."""
        return self.conn.execute(
            "SELECT * FROM observations WHERE consumed_by IS NULL "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()

    def count_observations(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()
        return int(row["n"])

    # --- M4: persona rules ---------------------------------------------------
    # Body lives in persona/learned.md (docs/CONTRACTS.md non-negotiable 5);
    # these columns are the metadata a model must not be able to forge in prose
    # (non-negotiable 3). Unlike memory_entries there is no unique index on
    # supersession_key - persona.rules.LearnedRules is what keeps one active row
    # per key, by resolving a batch before it ever reaches these methods.

    def insert_persona_rule(
        self,
        *,
        body: str,
        created_at: datetime,
        evidence: Sequence[int],
        supersession_key: str | None = None,
    ) -> int:
        """Append one persona rule to the mirror. Self-commits: the caller
        (`persona.rules.LearnedRules.add`) writes and fsyncs `learned.md`
        *before* calling this, so nothing here needs to defer a commit the way
        `insert_entry` does for the curated tier."""
        cursor = self.conn.execute(
            "INSERT INTO persona_rules (body, created_at, evidence, supersession_key) "
            "VALUES (?, ?, ?, ?)",
            (body, utc_iso(created_at), json.dumps(list(evidence)), supersession_key),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def active_persona_rules(self) -> list[sqlite3.Row]:
        """Active rules, oldest first - reads as an accumulated history."""
        return self.conn.execute(
            "SELECT * FROM persona_rules WHERE status = 'active' ORDER BY created_at ASC, id ASC"
        ).fetchall()

    def retire_persona_rule(self, rule_id: int, *, when: datetime, why: str) -> bool:
        """Retire one rule - a human's `daemon persona forget`, or one batch
        auto-superseding an older rule that shares its key. Returns whether a
        matching active row existed; never raises."""
        cursor = self.conn.execute(
            "UPDATE persona_rules SET status = 'retired', retired_at = ?, retired_why = ? "
            "WHERE id = ? AND status = 'active'",
            (utc_iso(when), why, rule_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def count_active_persona_rules(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM persona_rules WHERE status = 'active'"
        ).fetchone()
        return int(row["n"])

    def persona_rules_created_since(self, ts: str) -> int:
        """How many rules (any status) were created at or after `ts` - the
        weekly cap's raw input for `daemon doctor` / `daemon persona`."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM persona_rules WHERE created_at >= ?", (ts,)
        ).fetchone()
        return int(row["n"])

    def consume_observations(self, ids: Sequence[int], rule_id: int) -> None:
        """Mark observations consumed by this rule. Only `consumed_by IS NULL`
        rows are touched, so an observation is consumed exactly once - the one
        field on an append-only table (docs/CONTRACTS.md non-negotiable 6) that
        may be set later, and only the once."""
        if not ids:
            return
        self.conn.executemany(
            "UPDATE observations SET consumed_by = ? WHERE id = ? AND consumed_by IS NULL",
            [(rule_id, int(i)) for i in ids],
        )
        self.conn.commit()

    def last_persona_rule_created_at(self) -> str | None:
        """When the most recent persona rule (any status) was created, for
        `daemon doctor` / `daemon persona` - not evolve.py's own idempotency
        check, which is the diary file's existence."""
        row = self.conn.execute("SELECT MAX(created_at) AS ts FROM persona_rules").fetchone()
        return row["ts"]

    # --- M3: proactive candidates -------------------------------------------

    def insert_candidate(
        self,
        *,
        kind: str,
        reason: str,
        payload: str,
        now: datetime,
        due_at: datetime | None = None,
        expires_at: datetime | None = None,
        fire_budget: int = 1,
        cooldown_secs: int = 86_400,
    ) -> int:
        """Mirror one candidate. Payload arrives already JSON-encoded so this layer
        does not have to know what any generator puts in it."""
        cursor = self.conn.execute(
            "INSERT INTO proactive_candidates "
            "(kind, reason, payload, created_at, due_at, expires_at, fire_budget, cooldown_secs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                reason,
                payload,
                utc_iso(now),
                None if due_at is None else utc_iso(due_at),
                None if expires_at is None else utc_iso(expires_at),
                fire_budget,
                cooldown_secs,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    _LIVE = (
        # `fired` is live too, as long as the candidate's own budget is not spent.
        # Leaving it out meant a candidate allowed to fire twice was silently
        # retired after the first - the state machine in schema.sql distinguishes
        # `fired` from `done` precisely so that can be told apart, and a query that
        # stops at 'armed' throws the distinction away.
        "state IN ('pending', 'armed', 'fired') "
        "AND fire_count < fire_budget "
        "AND (expires_at IS NULL OR expires_at > :now)"
    )

    def live_candidates(self, *, now: datetime) -> list[sqlite3.Row]:
        """Candidates still in play.

        **Not the deduplication check**, which an earlier version of this docstring
        claimed. It excludes `done`, `cancelled` and `expired`, and rows whose
        budget is spent - so an open loop that already fired and was marked `done`
        has no live row, is regenerated on the next tick, and the daemon asks about
        the same presentation again. That is the stalker failure, arriving through
        the one method that looked like it prevented it. Use
        `existing_dedup_keys`, which reads *every* state.
        """
        return self.conn.execute(
            f"SELECT * FROM proactive_candidates WHERE {self._LIVE} "
            "ORDER BY due_at IS NULL, due_at ASC, id ASC",
            {"now": utc_iso(now)},
        ).fetchall()

    def due_candidates(self, *, now: datetime) -> list[sqlite3.Row]:
        """Live candidates whose moment has arrived - what the gate is offered.

        Two separate cooldowns exist and this is the per-candidate one
        (`cooldown_secs`): "do not raise *this* again for a day". The global gap
        between any two utterances is the gate's, from settings. Conflating them
        would let five different candidates fire in five minutes, each within its
        own cooldown.
        """
        return self.conn.execute(
            f"SELECT * FROM proactive_candidates WHERE {self._LIVE} "
            "AND (due_at IS NULL OR due_at <= :now) "
            "AND (last_fired_at IS NULL OR strftime('%Y-%m-%dT%H:%M:%SZ', last_fired_at, "
            "     '+' || cooldown_secs || ' seconds') <= :now) "
            "ORDER BY due_at IS NULL, due_at ASC, id ASC",
            {"now": utc_iso(now)},
        ).fetchall()

    def set_candidate_state(self, candidate_id: int, state: str) -> None:
        self.conn.execute(
            "UPDATE proactive_candidates SET state = ? WHERE id = ?", (state, candidate_id)
        )
        self.conn.commit()

    def push_candidate_due(self, candidate_id: int, *, due_at: datetime) -> None:
        """Move a candidate's `due_at` forward without touching `state` or
        `fire_count`.

        For a judge that declined: `due_candidates` reads `due_at` and not the
        clock the judge ran on, so leaving it alone means the very next tick
        offers the same candidate again - one hosted-model call per tick,
        forever, for a decision that already had its answer. This is the rest
        `daemon/proactivity/tick.py` gives a declined candidate; state and
        `fire_count` stay as they were because nothing fired.
        """
        self.conn.execute(
            "UPDATE proactive_candidates SET due_at = ? WHERE id = ?",
            (utc_iso(due_at), candidate_id),
        )
        self.conn.commit()

    def mark_candidate_fired(self, candidate_id: int, *, now: datetime) -> None:
        """One more firing against this candidate's own budget.

        `state` becomes `done` once the budget is spent, so a candidate that may
        fire twice is not silently retired after the first.
        """
        stamp = utc_iso(now)
        with self.conn:
            self.conn.execute(
                "UPDATE proactive_candidates "
                "SET fire_count = fire_count + 1, last_fired_at = ?, "
                "    state = CASE WHEN fire_count + 1 >= fire_budget THEN 'done' ELSE 'fired' END "
                "WHERE id = ?",
                (stamp, candidate_id),
            )

    def expire_candidates(self, *, now: datetime) -> int:
        """Retire what ran out of time. Returns how many.

        Separate from `live_candidates` filtering it out: a row left `pending`
        forever is indistinguishable from one still waiting, and the state machine
        in schema.sql is the thing a human reads to understand why nothing spoke.
        """
        cursor = self.conn.execute(
            "UPDATE proactive_candidates SET state = 'expired' "
            "WHERE state IN ('pending', 'armed') AND expires_at IS NOT NULL AND expires_at <= ?",
            (utc_iso(now),),
        )
        self.conn.commit()
        return cursor.rowcount

    def existing_dedup_keys(self, keys: Sequence[str]) -> set[str]:
        """Which of these dedup keys the candidate table has *ever* held.

        Every state, on purpose - see `live_candidates`. A key is spent the moment
        a candidate carrying it exists, whatever became of that candidate: fired,
        cancelled by hand, or expired unspoken. "It has been quiet too long" is one
        observation, and a version that retries is an alarm clock.

        The key lives in `payload` rather than a column only because
        `daemon/memory/schema.sql` is frozen; a column with a unique index is the
        better shape whenever v5 opens.
        """
        if not keys:
            return set()
        placeholders = ",".join("?" * len(keys))
        rows = self.conn.execute(
            "SELECT json_extract(payload, '$.dedup') AS dedup FROM proactive_candidates "
            f"WHERE json_extract(payload, '$.dedup') IN ({placeholders})",
            tuple(keys),
        ).fetchall()
        return {row["dedup"] for row in rows}

    # --- M3: what the generators read ---------------------------------------
    # `session_kind IN ('interactive', 'voice')` in all three is load-bearing, not
    # hygiene: without it one proactive utterance resets the silence clock, and
    # speaking becomes its own excuse to stop noticing the silence.
    #
    # Bounds are rendered with `utc_iso`, never `to_iso`. `messages.ts` is written
    # at second precision, and as strings "...T12:00:00Z" > "...T12:00:00.000Z"
    # because 'Z' > '.', so a millisecond bound silently drops a row sitting
    # exactly on it.

    def last_conversation_at(self) -> datetime | None:
        """When the user and the daemon last actually talked."""
        row = self.conn.execute(
            "SELECT MAX(ts) AS ts FROM messages WHERE session_kind IN ('interactive', 'voice')"
        ).fetchone()
        return None if row["ts"] is None else from_iso(row["ts"])

    def conversation_between(self, start: datetime, end: datetime) -> list[sqlite3.Row]:
        """Conversation in a window, inclusive at both ends, in reading order."""
        return self.conn.execute(
            "SELECT * FROM messages WHERE session_kind IN ('interactive', 'voice') "
            "AND ts >= ? AND ts <= ? ORDER BY ts ASC, id ASC",
            (utc_iso(start), utc_iso(end)),
        ).fetchall()

    def conversation_times(self, since: datetime) -> list[datetime]:
        """Just the timestamps. `pattern_time` reads two months of them to learn
        which hours this person talks in, and wants none of the text."""
        rows = self.conn.execute(
            "SELECT ts FROM messages WHERE session_kind IN ('interactive', 'voice') "
            "AND ts >= ? ORDER BY ts ASC",
            (utc_iso(since),),
        ).fetchall()
        return [from_iso(row["ts"]) for row in rows]

    # --- M3: what was actually said -----------------------------------------

    def insert_utterance(
        self,
        *,
        utterance_id: str,
        candidate_id: int | None,
        kind: str,
        text: str,
        route: str,
        gate_snapshot: str,
        now: datetime,
    ) -> None:
        """Record one proactive utterance. `utterance_id` is a uuid the label
        button echoes back, so it is chosen by the caller rather than by sqlite."""
        self.conn.execute(
            "INSERT INTO proactive_utterances "
            "(id, candidate_id, kind, text, spoken_at, route, gate_snapshot) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (utterance_id, candidate_id, kind, text, utc_iso(now), route, gate_snapshot),
        )
        self.conn.commit()

    def set_utterance_route(self, utterance_id: str, route: str) -> None:
        """Correct the route to what delivery actually achieved.

        The row is written *before* sending so the id on the label button resolves
        the moment it is pressed, which means the route stored at that point is the
        intended one. A speaker that failed while Telegram succeeded turns `both`
        into `telegram`, and the snapshot has to say what happened rather than what
        was planned.
        """
        self.conn.execute(
            "UPDATE proactive_utterances SET route = ? WHERE id = ?", (route, utterance_id)
        )
        self.conn.commit()

    def delete_utterance(self, utterance_id: str) -> None:
        """Remove a row for an utterance that was never delivered.

        Deliberate, and the one place this table is written destructively -
        non-negotiable 6 makes `observations` append-only and says nothing about
        this one, because these rows are a record of *what was said*. An utterance
        that reached nobody was not said, and leaving the row would spend the day's
        budget on silence and put an unlabelable message in the precision numbers.
        """
        self.conn.execute("DELETE FROM proactive_utterances WHERE id = ?", (utterance_id,))
        self.conn.commit()

    def last_utterance_at(self) -> datetime | None:
        """When it last spoke first, of any kind - the cooldown's input."""
        row = self.conn.execute(
            "SELECT spoken_at FROM proactive_utterances ORDER BY spoken_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else from_iso(row["spoken_at"])

    def utterances_since(self, *, since: datetime) -> list[sqlite3.Row]:
        """Everything spoken since `since`, newest first.

        The budget is per *local* day while `spoken_at` is UTC, so the caller
        passes the local day's start converted to UTC rather than this method
        trying to know which day it is - the same reason `messages_for_day`
        filters on `log_file`.
        """
        return self.conn.execute(
            "SELECT * FROM proactive_utterances WHERE spoken_at >= ? ORDER BY spoken_at DESC",
            (utc_iso(since),),
        ).fetchall()

    def label_utterance(self, utterance_id: str, label: str, *, now: datetime) -> bool:
        """Record the user's verdict. Returns whether the row existed.

        This is the label clock (docs/PLAN.md 8.1): tuning proactivity needs weeks
        of real "that was good" / "that was annoying", and no amount of thinking
        substitutes for it.
        """
        cursor = self.conn.execute(
            "UPDATE proactive_utterances SET label = ?, labeled_at = ? WHERE id = ?",
            (label, utc_iso(now), utterance_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def mark_responded(self, utterance_id: str) -> None:
        """The user replied to it, which is a label nobody had to press."""
        self.conn.execute(
            "UPDATE proactive_utterances SET responded = 1 WHERE id = ?", (utterance_id,)
        )
        self.conn.commit()

    def label_counts(self) -> dict[str, int]:
        """`good` / `bad` / `unlabeled` / `responded`. What `daemon doctor` prints,
        because "is it a stalker or a dead bot" is the M3 gate and it is not
        answerable from a log line."""
        rows = self.conn.execute(
            "SELECT COALESCE(label, 'unlabeled') AS verdict, COUNT(*) AS n "
            "FROM proactive_utterances GROUP BY verdict"
        ).fetchall()
        counts = {row["verdict"]: int(row["n"]) for row in rows}
        responded = self.conn.execute(
            "SELECT COUNT(*) AS n FROM proactive_utterances WHERE responded = 1"
        ).fetchone()
        counts["responded"] = int(responded["n"])
        return counts

    def recent_bad_labels(self, *, since: datetime) -> list[tuple[str, datetime]]:
        """(kind, labeled_at) for every 👎 at or after `since`, newest first.

        The gate counts them itself, for the same reason it counts utterances
        itself (`utterances_since`): the day boundary is local and `labeled_at`
        is UTC, so which rows belong to today is the caller's question.
        """
        rows = self.conn.execute(
            "SELECT kind, labeled_at FROM proactive_utterances "
            "WHERE label = 'bad' AND labeled_at >= ? ORDER BY labeled_at DESC",
            (utc_iso(since),),
        ).fetchall()
        return [(row["kind"], from_iso(row["labeled_at"])) for row in rows]

    # --- M5: telemetry the admin reads back ----------------------------------

    def record_proactive_round(
        self,
        *,
        generated: int,
        expired: int,
        considered: int,
        spoke: int,
        declined: int,
        blocked_by: str,
        now: datetime,
    ) -> int:
        """Append one round to the audit, returning its id.

        Written for the rounds that did nothing too. Those are the majority and
        they are the point: "288 rounds, none of them spoke" and "the loop stopped
        running" are the same picture without them.
        """
        cursor = self.conn.execute(
            "INSERT INTO proactive_rounds "
            "(ts, generated, expired, considered, spoke, declined, blocked_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (utc_iso(now), generated, expired, considered, spoke, declined, blocked_by),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def proactive_rounds_since(self, *, since: datetime) -> list[sqlite3.Row]:
        """Every round since `since`, oldest first - the timeline's reading order."""
        return self.conn.execute(
            "SELECT * FROM proactive_rounds WHERE ts >= ? ORDER BY ts",
            (utc_iso(since),),
        ).fetchall()

    def record_reflection_run(
        self,
        *,
        date: str,
        status: str,
        messages_read: int,
        facts: int,
        entities: int,
        observations: int,
        detail: str,
        now: datetime,
    ) -> int:
        """Append one pass to the audit, returning its id.

        Includes the passes that failed, which the artifact file cannot: it is not
        written when the model is unreachable, so a broken night and a night with
        nothing to say look identical on disk.
        """
        cursor = self.conn.execute(
            "INSERT INTO reflection_runs "
            "(ts, date, status, messages_read, facts, entities, observations, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                utc_iso(now),
                date,
                status,
                messages_read,
                facts,
                entities,
                observations,
                detail,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def reflection_runs_since(self, *, since: datetime) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM reflection_runs WHERE ts >= ? ORDER BY ts",
            (utc_iso(since),),
        ).fetchall()

    def recent_reflection_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM reflection_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # --- M2: what reflection reads ------------------------------------------

    def messages_for_day(self, date: str) -> list[sqlite3.Row]:
        """One local day's conversation, in reading order.

        `log_file` is the filter rather than a range over `ts`, because the log is
        split on the *local* day while timestamps are UTC - a KST day legitimately
        holds records whose UTC date is the day before, so a BETWEEN on `ts` would
        silently reflect on a nine-hour-shifted window.

        Rule 1 of docs/PLAN.md 4.2 is here: proactive and reflection sessions are
        excluded, because their content is the daemon's own speech and letting
        that become evidence is the self-amplifying loop the rule blocks.

        **Rule 2 used to be here too and is not any more** (2026-08-05). It
        excluded `recalled = 1` - rows recall had put in front of the model - and
        the exclusion was permanent. Two things measured on one real day of use:

          * It removed 29 of 38 messages, and the 29 were the persona-relevant
            ones ("너무 말이 길이 조금 짧게 대답해줄래", "답장이 왜케 오래걸려")
            while the 9 survivors were wake-word noise ("루시아", "에이데몬").
            Reflection produced 0 facts, 0 entities, 0 observations - it was not
            judging badly, it was handed nothing to judge.
          * It never blocked the loop it named. Recall injects its hits as a
            *system block*; `loop.py` records only the user turn and the reply, so
            injected context is never a row in the first place. The path that does
            exist - the reply restating what was recalled - produces a fresh row
            carrying no flag, which this filter never saw.

        So the cost was the whole of M4's input and the benefit was nothing. The
        column and `mark_recalled` are left alone: the flag still records what
        recall has surfaced, and nothing now reads it (see its docstring).
        """
        return self.conn.execute(
            "SELECT * FROM messages WHERE log_file = ? "
            "AND session_kind IN ('interactive', 'voice') "
            "ORDER BY ts ASC, id ASC",
            (f"memory/log/{date}.md",),
        ).fetchall()
