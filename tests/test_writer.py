"""Markdown-first, sqlite-second. These tests exist to pin the *order*, because
the order is the whole reason losing the database is survivable."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from daemon.memory import log
from daemon.memory.base import LoggedMessage, MemoryWriter
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter

LOG_FILE = Path("memory") / "log" / "2026-08-03.md"


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


def _delete_database(path: Path) -> None:
    for leftover in path.parent.glob(f"{path.name}*"):  # the -wal and -shm files too
        leftover.unlink()


@pytest.fixture
def writer(data_dir: Path, db: sqlite3.Connection) -> FileMemoryWriter:
    return FileMemoryWriter(data_dir, Store(db))


def test_satisfies_the_protocol(writer: FileMemoryWriter) -> None:
    assert isinstance(writer, MemoryWriter)


async def test_record_lands_in_both_places(
    writer: FileMemoryWriter, data_dir: Path, db: sqlite3.Connection
) -> None:
    await writer.record(message("오늘 이사 준비했어"))
    await writer.record(message("힘들었겠다", role="assistant"))

    text = (data_dir / LOG_FILE).read_text(encoding="utf-8")
    assert "오늘 이사 준비했어" in text
    assert "힘들었겠다" in text

    rows = db.execute("SELECT content, log_file FROM messages ORDER BY id").fetchall()
    assert [row["content"] for row in rows] == ["오늘 이사 준비했어", "힘들었겠다"]
    assert rows[0]["log_file"] == LOG_FILE.as_posix()


async def test_recent_is_oldest_first(writer: FileMemoryWriter) -> None:
    for minute in (1, 2, 3):
        await writer.record(message(f"{minute}분", ts=datetime(2026, 8, 3, 7, minute, tzinfo=UTC)))

    recalled = await writer.recent()
    assert [m.content for m in recalled] == ["1분", "2분", "3분"]
    assert recalled[0].ts == datetime(2026, 8, 3, 7, 1, tzinfo=UTC)
    assert recalled[0].origin == "owner"
    assert recalled[0].sender_id == "42"


async def test_provenance_never_reaches_the_markdown(
    writer: FileMemoryWriter, data_dir: Path
) -> None:
    """Columns only. If any of this appeared in prose, a model could write it."""
    await writer.record(message("비밀 얘기"))

    text = (data_dir / LOG_FILE).read_text(encoding="utf-8")
    for forgeable in ("owner", "interactive", "telegram", "42", "importance"):
        assert forgeable not in text


async def test_a_new_local_day_splits_the_file(writer: FileMemoryWriter, data_dir: Path) -> None:
    await writer.record(message("첫날", ts=datetime(2026, 8, 3, 7, 0, tzinfo=UTC)))
    await writer.record(message("다음날", ts=datetime(2026, 8, 4, 7, 0, tzinfo=UTC)))

    log_dir = data_dir / "memory" / "log"
    assert sorted(p.name for p in log_dir.iterdir()) == ["2026-08-03.md", "2026-08-04.md"]


async def test_failed_markdown_write_mirrors_nothing(
    writer: FileMemoryWriter, data_dir: Path, db: sqlite3.Connection
) -> None:
    """A row pointing at a record that was never written is worse than a lost
    turn, so the markdown write has to succeed first."""
    (data_dir / LOG_FILE).mkdir(parents=True)  # the log file's path is now a directory

    with pytest.raises(OSError):
        await writer.record(message("이건 못 써"))

    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


async def test_failed_mirror_keeps_the_markdown(
    writer: FileMemoryWriter, data_dir: Path, db: sqlite3.Connection
) -> None:
    db.close()

    with pytest.raises(sqlite3.ProgrammingError):
        await writer.record(message("데이터베이스가 죽어도 이 말은 남아야 해"))

    text = (data_dir / LOG_FILE).read_text(encoding="utf-8")
    assert log.parse(text)[0].content == "데이터베이스가 죽어도 이 말은 남아야 해"


async def test_markdown_survives_deleting_the_database(data_dir: Path) -> None:
    """The source-of-truth check: throw the index away, rebuild the schema, and
    the conversation is still there in full and still machine-readable."""
    db_path = data_dir / "daemon.sqlite3"
    store = Store.open(db_path)
    writer = FileMemoryWriter(data_dir, store)
    await writer.record(message("나 요즘 러닝 시작했어"))
    await writer.record(message("좋네, 몇 킬로 뛰어?", role="assistant"))
    store.close()

    _delete_database(db_path)

    rebuilt = Store.open(db_path)
    try:
        assert rebuilt.recent() == []  # the mirror is empty, as expected
        records = log.parse((data_dir / LOG_FILE).read_text(encoding="utf-8"))
        assert [(r.role, r.content) for r in records] == [
            ("user", "나 요즘 러닝 시작했어"),
            ("assistant", "좋네, 몇 킬로 뛰어?"),
        ]
    finally:
        rebuilt.close()


async def test_concurrent_records_lose_nothing(
    writer: FileMemoryWriter, data_dir: Path, db: sqlite3.Connection
) -> None:
    bodies = [f"동시에 온 {i}번" for i in range(20)]
    await asyncio.gather(*(writer.record(message(b)) for b in bodies))

    records = log.parse((data_dir / LOG_FILE).read_text(encoding="utf-8"))
    assert sorted(r.content for r in records) == sorted(bodies)
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == len(bodies)


async def test_markdown_and_mirror_hold_identical_text(
    writer: FileMemoryWriter, data_dir: Path, db: sqlite3.Connection
) -> None:
    """Surrounding whitespace cannot survive the log format, so it is normalised
    before either write - a mirror that disagrees with the original is unverifiable."""
    await writer.record(message("  앞뒤에 공백이 있었어\n"))

    (record,) = log.parse((data_dir / LOG_FILE).read_text(encoding="utf-8"))
    stored = db.execute("SELECT content FROM messages").fetchone()["content"]
    assert record.content == stored == "앞뒤에 공백이 있었어"
