"""Regression tests for the findings of the reliability audit.

The theme is that the markdown log is the source of truth, so anything that can
lose it, tear it, duplicate into it, or silently let its mirror drift is worse
than an outage.
"""

from __future__ import annotations

import asyncio
import fcntl
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from daemon.channels.telegram import (
    TelegramChannel,
    TelegramFatal,
)
from daemon.memory import log
from daemon.memory.base import LoggedMessage
from daemon.memory.reindex import reindex
from daemon.memory.store import SCHEMA_VERSION, Store
from daemon.memory.writer import FileMemoryWriter

TOKEN = "123456:AAHfake-token-value"
OWNER = 4242


def _msg(content: str, second: int = 0, role: str = "user") -> LoggedMessage:
    return LoggedMessage(
        ts=datetime(2026, 8, 3, 7, 0, second, tzinfo=UTC),
        role=role,  # type: ignore[arg-type]
        content=content,
        origin="owner" if role == "user" else "agent",
        session_kind="interactive",
        modality="text",
        channel="telegram",
    )


# --- CRITICAL: the source of truth must be at least as durable as its mirror --


async def test_the_markdown_append_is_fsynced(tmp_path: Path, monkeypatch) -> None:
    """sqlite commits with synchronous=FULL, so without an fsync here the
    markdown is *less* durable than the index built from it: a power cut inside
    the page-cache window leaves a row whose record is missing from the log."""
    synced: list[int] = []
    real_fsync = log.os.fsync

    def spy(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(log.os, "fsync", spy)
    await log.append(tmp_path, _msg("durable?"))
    assert synced, "the markdown append returned without fsyncing"


def test_sqlite_still_commits_with_full_synchronous(tmp_path: Path) -> None:
    """Pins the asymmetry the fix above is balancing. If this ever drops to
    NORMAL, revisit whether the markdown fsync is still the cheaper side."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        assert store.conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        store.close()


async def test_the_append_takes_an_exclusive_file_lock(tmp_path: Path, monkeypatch) -> None:
    """The asyncio lock only covers this process. Two instances - overlapping
    systemd restarts, a stray copy, a backfill script - would both write the date
    header, and the extra heading is absorbed into the previous record's body."""
    modes: list[int] = []
    real_flock = fcntl.flock

    def spy(fd: int, operation: int) -> None:
        modes.append(operation)
        real_flock(fd, operation)

    monkeypatch.setattr(log.fcntl, "flock", spy)
    await log.append(tmp_path, _msg("locked?"))
    assert fcntl.LOCK_EX in modes


# --- HIGH: a torn tail must not cost the whole day --------------------------


def test_a_torn_utf8_tail_costs_one_record_not_the_day(tmp_path: Path) -> None:
    """A write cut mid-record can split a UTF-8 sequence. Strict decoding then
    raises for the entire file, so one lost tail would take a day of Korean
    conversation with it."""
    path = tmp_path / "2026-08-03.md"
    text = (
        "# 2026-08-03\n\n"
        "## 2026-08-03T07:00:00Z user\n첫 번째 기록입니다\n\n"
        "## 2026-08-03T07:00:05Z assistant\n두 번째 기록입니다\n"
    )
    raw = text.encode("utf-8")

    # Only some cut points land inside a multi-byte sequence; those are the ones
    # that would have taken the whole file down with them.
    undecodable = 0
    for cut in range(1, 12):
        path.write_bytes(raw[:-cut])
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            undecodable += 1
        records = log.read(path)  # must not raise, whatever the cut
        assert records, "the earlier records did not survive the torn tail"
        assert records[0].content == "첫 번째 기록입니다"
    assert undecodable, "no cut point actually tore a UTF-8 sequence; test is not exercising it"


# --- HIGH: a long outage must not become a dead daemon ----------------------


async def test_backoff_survives_more_than_a_thousand_failures(monkeypatch) -> None:
    """min() bounded the delay but 2 ** (failures - 1) kept growing, so about 17
    hours of downtime overflowed float and threw OverflowError from inside the
    retry handler - killing the inbound path exactly when the network returned."""
    attempts = 0

    def always_502(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(502, text="bad gateway")

    slept: list[float] = []

    async def no_wait(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) > 1100:
            raise asyncio.CancelledError

    monkeypatch.setattr("daemon.channels.telegram._sleep", no_wait)
    client = httpx.AsyncClient(transport=httpx.MockTransport(always_502))
    channel = TelegramChannel(TOKEN, [OWNER], client=client)
    try:
        with pytest.raises(asyncio.CancelledError):
            async for _ in channel.listen():
                pass
    finally:
        await channel.close()

    assert len(slept) > 1024, "the loop died before the overflow point"
    assert max(slept) <= 60.0


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_a_permanent_rejection_kills_the_loop_instead_of_hiding(status: int) -> None:
    """A revoked token retried forever leaves the process alive, healthy-looking
    and permanently deaf - the worst failure for something meant to be
    listening."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json={"ok": False}))
    )
    channel = TelegramChannel(TOKEN, [OWNER], client=client)
    try:
        with pytest.raises(TelegramFatal):
            async for _ in channel.listen():
                pass
    finally:
        await channel.close()


# --- HIGH: a restart must not duplicate into an append-only log --------------


def _update(update_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {"from": {"id": OWNER}, "date": 1785744000, "text": text},
    }


async def test_the_offset_survives_a_restart(tmp_path: Path) -> None:
    """Telegram only confirms an update on the *next* getUpdates, so a restart in
    that window re-delivers it. Nothing else can reconcile the duplicate: the
    markdown is append-only."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        batch = [_update(100, "안녕"), _update(101, "잘 있었어?")]
        served = 0

        def serve(request: httpx.Request) -> httpx.Response:
            nonlocal served
            served += 1
            return httpx.Response(200, json={"ok": True, "result": batch if served == 1 else []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
        first = TelegramChannel(TOKEN, [OWNER], client=client, cursor=store)
        seen = []
        async for inbound in first.listen():
            seen.append(inbound.external_id)
            if len(seen) == 2:
                break
        await first.close()
        assert seen == ["100", "101"]
        # 101, not 102: the cursor advances only after the consumer comes back
        # from the yield, and breaking out of the loop means the second message
        # was never confirmed as handled. That is the intended direction of the
        # trade - at most one message is re-delivered, and the dedup check
        # catches it - because losing what the user said is the worse half.
        assert store.load_cursor("telegram") == 101

        # A fresh channel, as after a restart: it resumes rather than rewinding
        # to the start of the batch.
        client2 = httpx.AsyncClient(transport=httpx.MockTransport(serve))
        second = TelegramChannel(TOKEN, [OWNER], client=client2, cursor=store)
        assert second._offset == 101
        await second.close()
    finally:
        store.close()


async def test_a_redelivered_message_is_not_recorded_twice(tmp_path: Path) -> None:
    """Belt and braces for the one-message window the cursor cannot close: the
    crash between handling and confirming."""
    from daemon.channels.base import InboundMessage
    from daemon.llm.base import Completion
    from daemon.loop import ConversationLoop

    store = Store.open(tmp_path / "daemon.sqlite3")
    writer = FileMemoryWriter(tmp_path, store)

    class Gateway:
        calls = 0

        async def complete(self, task, messages, **kw):  # noqa: ANN001, ANN003
            Gateway.calls += 1
            return Completion(text="응", model="fake")

    class Channel:
        name = "telegram"

        async def send(self, message) -> None:  # noqa: ANN001
            pass

        def listen(self):  # pragma: no cover
            raise NotImplementedError

        async def close(self) -> None: ...

    loop = ConversationLoop(Channel(), Gateway(), writer, data_dir=tmp_path)
    inbound = InboundMessage(
        text="같은 메시지",
        sender_id=str(OWNER),
        received_at=datetime.now(UTC),
        channel="telegram",
        external_id="777",
    )
    try:
        await loop.handle(inbound)
        await loop.handle(inbound)  # the redelivery
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE external_id = '777'"
        ).fetchone()
        assert rows["n"] == 1
        assert Gateway.calls == 1, "the model was called again for a message already answered"
        assert len(log.read(next((tmp_path / "memory" / "log").glob("*.md")))) == 2
    finally:
        store.close()


# --- MEDIUM: the mirror must be rebuildable, not just described as such -------


async def test_a_deleted_database_is_rebuilt_from_the_markdown(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "daemon.sqlite3")
    writer = FileMemoryWriter(tmp_path, store)
    await writer.record(_msg("김치찌개 먹었어", 0))
    await writer.record(_msg("맛있었어?", 1, role="assistant"))
    store.close()

    (tmp_path / "daemon.sqlite3").unlink()
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"daemon.sqlite3{suffix}").unlink(missing_ok=True)

    rebuilt = Store.open(tmp_path / "daemon.sqlite3")
    try:
        assert rebuilt.recent() == []  # nothing yet - the repair is explicit
        assert reindex(tmp_path, rebuilt) == 2
        rows = rebuilt.recent()
        assert [r["content"] for r in rows] == ["김치찌개 먹었어", "맛있었어?"]
        assert all(r["reindexed"] == 1 for r in rows), "inferred provenance was not flagged"
        assert reindex(tmp_path, rebuilt) == 0, "a second pass duplicated rows"
    finally:
        rebuilt.close()


async def test_a_mirror_that_fell_behind_is_repaired(tmp_path: Path) -> None:
    """The realistic case: the markdown write succeeded and the mirror write did
    not, so one turn would otherwise be invisible to the model forever."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        writer = FileMemoryWriter(tmp_path, store)
        await writer.record(_msg("첫 줄", 0))
        # Simulate the failed mirror write: markdown only.
        await log.append(tmp_path, _msg("미러가 놓친 줄", 5))
        assert len(store.recent()) == 1

        assert reindex(tmp_path, store) == 1
        assert [r["content"] for r in store.recent()] == ["첫 줄", "미러가 놓친 줄"]
    finally:
        store.close()


# --- MEDIUM/LOW: schema, locks, empty sends ---------------------------------


def test_a_newer_database_is_refused_rather_than_written_into(tmp_path: Path) -> None:
    path = tmp_path / "daemon.sqlite3"
    store = Store.open(path)
    store.conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, '2026-08-03T00:00:00Z')",
        (SCHEMA_VERSION + 5,),
    )
    store.conn.commit()
    store.close()

    with pytest.raises(RuntimeError, match="Refusing to open"):
        Store.open(path)


def test_an_older_database_is_migrated_additively(tmp_path: Path) -> None:
    """A v1 file has no external_id column, and CREATE TABLE IF NOT EXISTS would
    skip the table entirely rather than add it."""
    path = tmp_path / "daemon.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL) STRICT;
        INSERT INTO schema_version VALUES (1, '2026-08-01T00:00:00Z');
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, ts TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, origin TEXT NOT NULL, session_kind TEXT NOT NULL,
            modality TEXT NOT NULL, channel TEXT NOT NULL, sender_id TEXT,
            log_file TEXT NOT NULL, recalled INTEGER NOT NULL DEFAULT 0
        ) STRICT;
        """
    )
    conn.commit()
    conn.close()

    store = Store.open(path)
    try:
        columns = {row["name"] for row in store.conn.execute("PRAGMA table_info(messages)")}
        assert {"external_id", "reindexed"} <= columns
        assert store.schema_version() == SCHEMA_VERSION
    finally:
        store.close()


def test_the_log_lock_is_not_bound_to_a_dead_event_loop(tmp_path: Path) -> None:
    """A module-level asyncio.Lock outlives the loop it bound to, and reusing it
    raises - then stays latched, so every later append to that path fails."""

    async def once() -> None:
        await log.append(tmp_path, _msg("같은 경로"))

    asyncio.run(once())
    asyncio.run(once())  # a second loop, same path

    assert len(log.read(next((tmp_path / "memory" / "log").glob("*.md")))) == 2


async def test_an_empty_completion_is_not_sent(tmp_path: Path) -> None:
    """Telegram rejects empty text with a 400, which would turn a harmless empty
    turn into a failed one after the empty record was already logged."""
    calls: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (calls.append(r), httpx.Response(200, json={"ok": True, "result": {}}))[1]
        )
    )
    channel = TelegramChannel(TOKEN, [OWNER], client=client)
    try:
        from daemon.channels.base import OutboundMessage

        await channel.send(OutboundMessage(text="   \n  "))
        assert calls == []
    finally:
        await channel.close()


async def test_one_unreachable_recipient_does_not_silence_the_others() -> None:
    import json

    delivered: list[int] = []

    def serve(request: httpx.Request) -> httpx.Response:
        chat_id = json.loads(request.content)["chat_id"]
        if chat_id == OWNER:
            return httpx.Response(403, json={"ok": False, "description": "blocked"})
        delivered.append(chat_id)
        return httpx.Response(200, json={"ok": True, "result": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    channel = TelegramChannel(TOKEN, [OWNER, 777], client=client)
    try:
        from daemon.channels.base import OutboundMessage

        await channel.send(OutboundMessage(text="발표 어떻게 됐어?"))
        assert delivered == [777]
    finally:
        await channel.close()
