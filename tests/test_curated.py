"""The curated tier: `memory/core.md`, its mirror, and the write order."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daemon.memory import curated
from daemon.memory.store import Store

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)


@pytest.fixture
def tier(data_dir: Path, db: Any) -> curated.CuratedMemory:
    return curated.CuratedMemory(data_dir, Store(db))


# --- the file ---------------------------------------------------------------


async def test_a_fact_lands_in_the_markdown_and_the_mirror(
    tier: curated.CuratedMemory, data_dir: Path
) -> None:
    await tier.add("김치찌개를 좋아한다", importance=7, now=NOW)

    assert curated.read(data_dir) == ["김치찌개를 좋아한다"]
    assert tier.entries() == ["김치찌개를 좋아한다"]


async def test_the_file_holds_every_active_fact_not_only_the_injected_ones(
    tier: curated.CuratedMemory, data_dir: Path
) -> None:
    """The file is the source of truth; the budget is an injection limit.

    Rendering the whole file from the top `MAX_INJECTED` rows conflated the two, so
    the 51st active fact was dropped from `core.md` on the next write while staying
    active in the mirror. `rebuild` only adds what it finds in the markdown, so
    `daemon reindex` could not notice, recall still served the row, and deleting the
    database - the documented recovery - lost it for good.
    """
    for index in range(curated.MAX_INJECTED + 2):
        tier._store.insert_entry(
            body=f"사실 {index:02d}",
            importance=5,
            trigger_phrases=(),
            origin="agent",
            session_kind="reflection",
            modality="text",
            now=NOW,
        )

    await tier.add("마지막 사실", importance=5, now=NOW)

    bodies = curated.read(data_dir)
    assert len(bodies) == curated.MAX_INJECTED + 3
    # The lowest-ranked row is the one truncation used to eat.
    assert "사실 00" in bodies
    assert len(tier.entries()) == curated.MAX_INJECTED, "the injection budget still holds"


async def test_the_file_is_owner_only(tier: curated.CuratedMemory, data_dir: Path) -> None:
    """It is a description of a person. 0644 hands it to every local account."""
    await tier.add("연희동에 산다", now=NOW)
    assert curated.core_path(data_dir).stat().st_mode & 0o777 == 0o600


async def test_a_superseding_fact_replaces_it_in_the_file_too(
    tier: curated.CuratedMemory, data_dir: Path
) -> None:
    """The file is a rewrite of the active set, so a retired fact must disappear
    from it - otherwise a rebuild would resurrect what was superseded."""
    await tier.add("여자친구가 있다", supersession_key="relationship", now=NOW)
    await tier.add("여자친구가 없다", supersession_key="relationship", now=NOW)

    assert curated.read(data_dir) == ["여자친구가 없다"]


async def test_the_file_is_ordered_by_importance(
    tier: curated.CuratedMemory, data_dir: Path
) -> None:
    await tier.add("사소한 것", importance=2, now=NOW)
    await tier.add("중요한 것", importance=9, now=NOW)

    assert curated.read(data_dir) == ["중요한 것", "사소한 것"]


async def test_a_multi_line_fact_is_folded_to_one_line(
    tier: curated.CuratedMemory, data_dir: Path
) -> None:
    """One bullet per entry is the format, so a newline would parse back as two."""
    await tier.add("첫 줄\n두 번째 줄", now=NOW)
    assert curated.read(data_dir) == ["첫 줄 두 번째 줄"]


async def test_an_empty_fact_is_refused(tier: curated.CuratedMemory) -> None:
    with pytest.raises(ValueError):
        await tier.add("   \n  ", now=NOW)


# --- the write order --------------------------------------------------------


async def test_a_failed_markdown_write_leaves_the_mirror_untouched(
    tier: curated.CuratedMemory, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-negotiable 1: markdown first. If the file cannot be written, the mirror
    must not have moved either - a row pointing at a record that does not exist is
    the failure the ordering exists to prevent.
    """
    await tier.add("먼저 있던 사실", supersession_key="k", now=NOW)

    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(curated, "write_private_replace", boom)

    with pytest.raises(OSError):
        await tier.add("대체하려던 사실", supersession_key="k", now=NOW)

    # The old fact is still active, and still the only one.
    assert tier.entries() == ["먼저 있던 사실"]
    assert curated.read(data_dir) == ["먼저 있던 사실"]


async def test_the_markdown_is_written_before_the_mirror_commits(
    tier: curated.CuratedMemory, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserts the *order*, not just the outcome: when the file is being written,
    a second connection must not yet be able to see the row. That is what makes
    the markdown the durable original rather than merely the first call."""
    seen_by_other_connection: list[int] = []
    real = curated.write_private_replace

    def observe(path: Path, text: str) -> None:
        other = sqlite3.connect(data_dir / "daemon.sqlite3")
        try:
            row = other.execute("SELECT COUNT(*) FROM memory_entries").fetchone()
            seen_by_other_connection.append(int(row[0]))
        finally:
            other.close()
        real(path, text)

    monkeypatch.setattr(curated, "write_private_replace", observe)
    await tier.add("아직 커밋 안 된 사실", now=NOW)

    assert seen_by_other_connection == [0]
    assert tier.entries() == ["아직 커밋 안 된 사실"]


async def test_a_failure_before_the_file_write_leaves_nothing_pending(
    tier: curated.CuratedMemory, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror write is already open when the file is rendered, so a failure
    *there* has to roll back too.

    Found by review. Rendering sat outside the guard, so the retire-and-insert
    stayed open and uncommitted - and the next commit on this connection, which
    `reflection.Reflection.run` always makes to record the run, turned it durable
    with no markdown behind it. That is the one direction non-negotiable 1 calls
    unrecoverable, and the pass reported the fact as not recorded.
    """
    await tier.add("먼저 있던 사실", supersession_key="k", now=NOW)

    def boom(*_: object, **__: object) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(curated, "render", boom)

    with pytest.raises(sqlite3.OperationalError):
        await tier.add("대체하려던 사실", supersession_key="k", now=NOW)

    assert not tier._store.conn.in_transaction, "the transaction must not be left open"
    # A fresh connection: the retire must not be waiting for someone else's commit.
    other = sqlite3.connect(data_dir / "daemon.sqlite3")
    try:
        rows = other.execute(
            "SELECT body FROM memory_entries WHERE status = 'active'"
        ).fetchall()
    finally:
        other.close()
    assert [row[0] for row in rows] == ["먼저 있던 사실"]
    assert curated.read(data_dir) == ["먼저 있던 사실"]


# --- rebuild ----------------------------------------------------------------


async def test_the_mirror_rebuilds_from_the_file(
    tier: curated.CuratedMemory, data_dir: Path, db: Any
) -> None:
    await tier.add("고양이를 키운다", importance=8, now=NOW)
    await tier.add("연희동에 산다", importance=4, now=NOW)

    store = Store(db)
    store.conn.execute("DELETE FROM memory_entries")
    store.conn.commit()
    assert store.count_entries() == 0

    assert curated.rebuild(data_dir, store) == 2
    assert sorted(row["body"] for row in store.active_entries()) == [
        "고양이를 키운다",
        "연희동에 산다",
    ]


async def test_a_rebuilt_entry_says_it_was_rebuilt(
    tier: curated.CuratedMemory, data_dir: Path, db: Any
) -> None:
    """Provenance cannot come back from the markdown, so it must not be faked.
    `origin='system'` is how reflection tells a fact it concluded from one a
    rebuild guessed at."""
    await tier.add("고양이를 키운다", importance=8, now=NOW)
    store = Store(db)
    store.conn.execute("DELETE FROM memory_entries")
    store.conn.commit()

    curated.rebuild(data_dir, store)

    row = store.active_entries()[0]
    assert row["origin"] == "system"
    assert row["importance"] == 5  # not the 8 that reflection chose


async def test_rebuilding_twice_adds_nothing(
    tier: curated.CuratedMemory, data_dir: Path, db: Any
) -> None:
    await tier.add("고양이를 키운다", now=NOW)
    store = Store(db)
    assert curated.rebuild(data_dir, store) == 0  # already active
    assert store.count_entries() == 1


def test_rebuilding_with_no_file_is_zero_not_an_error(data_dir: Path, db: Any) -> None:
    assert curated.rebuild(data_dir, Store(db)) == 0


# --- parsing ----------------------------------------------------------------


def test_a_hand_mangled_file_degrades_to_fewer_entries(data_dir: Path) -> None:
    """The source of truth must never raise on read: an exception here would fail
    every future reflection, permanently."""
    text = "# core\n\n산문이 섞여 있고\n- 진짜 항목\n## 사람이 넣은 제목\n- 또 하나\n"
    assert curated.parse(text) == ["진짜 항목", "또 하나"]


def test_the_header_is_not_mistaken_for_an_entry(data_dir: Path) -> None:
    assert curated.parse(curated.render([])) == []
