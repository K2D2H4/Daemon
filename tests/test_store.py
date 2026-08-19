"""The sqlite mirror. Nothing here is allowed to be the only copy of anything."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from daemon.memory.base import LoggedMessage
from daemon.memory.store import SCHEMA_VERSION, Store


def message(
    content: str,
    *,
    ts: datetime | None = None,
    role: Literal["user", "assistant"] = "user",
) -> LoggedMessage:
    return LoggedMessage(
        ts=ts or datetime(2026, 8, 3, 7, 14, 0, tzinfo=UTC),
        role=role,
        content=content,
        origin="owner" if role == "user" else "agent",
        session_kind="interactive",
        modality="text",
        channel="telegram",
        sender_id="42",
    )


def test_open_applies_schema_and_records_its_version(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "sub" / "daemon.sqlite3")
    try:
        assert store.schema_version() == SCHEMA_VERSION
        assert store.recent() == []
    finally:
        store.close()


def test_apply_schema_is_idempotent(tmp_path: Path) -> None:
    """Same call is the rebuild path, so running it twice must not double up."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        store.insert_message(message("안녕"), log_file="memory/log/2026-08-03.md")
        store.apply_schema()

        assert store.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        assert len(store.recent()) == 1
    finally:
        store.close()


def test_insert_keeps_provenance_in_columns(db: sqlite3.Connection) -> None:
    store = Store(db)
    store.insert_message(
        replace(message("음성으로 말했어"), modality="voice", session_kind="voice"),
        log_file="memory/log/2026-08-03.md",
    )

    (row,) = store.recent()
    assert row["origin"] == "owner"
    assert row["session_kind"] == "voice"
    assert row["modality"] == "voice"
    assert row["channel"] == "telegram"
    assert row["sender_id"] == "42"
    assert row["log_file"] == "memory/log/2026-08-03.md"
    assert row["recalled"] == 0


def test_recent_is_oldest_first(db: sqlite3.Connection) -> None:
    store = Store(db)
    for minute in (3, 1, 2):  # inserted out of order on purpose
        store.insert_message(
            message(f"{minute}분", ts=datetime(2026, 8, 3, 7, minute, tzinfo=UTC)),
            log_file="memory/log/2026-08-03.md",
        )

    assert [row["content"] for row in store.recent()] == ["1분", "2분", "3분"]


def test_recent_limit_keeps_the_newest(db: sqlite3.Connection) -> None:
    store = Store(db)
    for minute in range(1, 6):
        store.insert_message(
            message(f"{minute}분", ts=datetime(2026, 8, 3, 7, minute, tzinfo=UTC)),
            log_file="memory/log/2026-08-03.md",
        )

    assert [row["content"] for row in store.recent(limit=2)] == ["4분", "5분"]


def test_messages_sharing_a_second_keep_insertion_order(db: sqlite3.Connection) -> None:
    """Timestamps are second-resolution, so a question and its answer can collide."""
    store = Store(db)
    store.insert_message(message("질문"), log_file="f.md")
    store.insert_message(message("대답", role="assistant"), log_file="f.md")

    assert [row["content"] for row in store.recent()] == ["질문", "대답"]


def test_korean_full_text_search(db: sqlite3.Connection) -> None:
    """FTS5 with unicode61 has no Korean morphology, so it matches whitespace
    tokens and prefixes. That is what M1b recall will be built on, so verify the
    behaviour we actually get rather than assume word-level matching."""
    store = Store(db)
    store.insert_message(message("오늘 저녁에 김치찌개 먹었어"), log_file="f.md")
    store.insert_message(message("어제는 파스타 만들었어", role="assistant"), log_file="f.md")

    def search(query: str) -> list[str]:
        rows = db.execute(
            "SELECT m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid"
            " WHERE messages_fts MATCH ? ORDER BY m.id",
            (query,),
        ).fetchall()
        return [row["content"] for row in rows]

    assert search("김치찌개") == ["오늘 저녁에 김치찌개 먹었어"]
    assert search("파스타") == ["어제는 파스타 만들었어"]
    assert search("김치*") == ["오늘 저녁에 김치찌개 먹었어"]
    assert search("먹었어 OR 만들었어") == [
        "오늘 저녁에 김치찌개 먹었어",
        "어제는 파스타 만들었어",
    ]
    # A substring that is not a token boundary does not match - a known limit of
    # unicode61 on Korean, to be solved with embeddings in M1b, not here.
    assert search("김치찌") == []


def test_bad_provenance_is_rejected_by_the_schema(db: sqlite3.Connection) -> None:
    store = Store(db)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_message(replace(message("x"), origin="whoever"), log_file="f.md")


def test_enqueue_then_claim_marks_running_and_returns_it(db: sqlite3.Connection) -> None:
    store = Store(db)
    tid = store.enqueue_task(
        request="노션에 페이지 만들어줘",
        origin="owner",
        channel="voice",
        sender_id=None,
    )
    assert tid > 0
    row = store.claim_next_queued()
    assert row is not None
    assert row["id"] == tid
    assert row["request"] == "노션에 페이지 만들어줘"
    assert row["status"] == "running"
    # Only one queued row existed, so the next claim finds nothing.
    assert store.claim_next_queued() is None


def test_mark_done_records_the_result(db: sqlite3.Connection) -> None:
    store = Store(db)
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()
    store.mark_task_done(tid, "만들었어요")
    (row,) = [r for r in store.pending_tasks()] or [None]
    assert row is None  # a done task is not pending
    done = store.conn.execute(
        "SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)
    ).fetchone()
    assert done["status"] == "done"
    assert done["result"] == "만들었어요"


def test_mark_failed_records_the_error(db: sqlite3.Connection) -> None:
    store = Store(db)
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()
    store.mark_task_failed(tid, "notion 400")
    row = store.conn.execute(
        "SELECT status, error FROM delegated_tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "notion 400"


def test_pending_reports_queued_and_running_left_by_a_restart(
    db: sqlite3.Connection,
) -> None:
    store = Store(db)
    a = store.enqueue_task(request="a", origin="owner", channel="voice", sender_id=None)
    b = store.enqueue_task(request="b", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()  # a -> running
    pending = store.pending_tasks()
    assert [r["id"] for r in pending] == [a, b]
    assert {r["status"] for r in pending} == {"running", "queued"}


# --- tool calls, by the local day ------------------------------------------


def _tool_call(store: Store, tool: str, ts: datetime, *, ok: bool = True) -> None:
    store.record_tool_call(
        tool=tool,
        arguments="{}",
        preview=f"{tool} ...",
        verdict="allow",
        mode="full",
        reason="",
        origin="owner",
        channel="telegram",
        sender_id="1",
        ran=True,
        ok=ok,
        output_excerpt="발표는 목요일",
        now=ts,
    )


def test_a_tool_call_belongs_to_the_local_day_not_the_utc_one(
    db: sqlite3.Connection, seoul: None
) -> None:
    """`tool_calls` carries only a UTC `ts`, and reflection reads days that were
    split locally. 23:00Z is already tomorrow morning in Seoul, so a BETWEEN over
    `ts` would hand that call to the wrong night - the same nine-hour shift
    `messages_for_day` refuses by filtering on `log_file` instead.
    """
    store = Store(db)
    _tool_call(store, "read_page", datetime(2026, 8, 3, 23, 0, tzinfo=UTC))
    _tool_call(store, "fetch_page", datetime(2026, 8, 3, 1, 0, tzinfo=UTC))

    third = [row["tool"] for row in store.tool_calls_for_day("2026-08-03")]
    fourth = [row["tool"] for row in store.tool_calls_for_day("2026-08-04")]

    assert third == ["fetch_page"]
    assert fourth == ["read_page"]
