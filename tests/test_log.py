"""The markdown log is the original, so these tests are about one question:
can everything written be read back?"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from daemon.memory import log
from daemon.memory.base import LoggedMessage


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


@contextmanager
def timezone(name: str) -> Iterator[None]:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_record_roundtrips_korean_multiline() -> None:
    body = "오늘 저녁에 김치찌개 먹었어.\n\n근데 좀 짰어 — 다음엔 덜 넣을게."
    (record,) = log.parse(log.render(message(body)))

    assert record.content == body
    assert record.role == "user"
    assert record.ts == datetime(2026, 8, 3, 7, 14, 0, tzinfo=UTC)


def test_body_that_looks_like_a_heading_roundtrips() -> None:
    """A user can paste a log fragment into a message. It must not become a
    record boundary."""
    body = "이렇게 쓰면 되나?\n## 2026-08-03T07:14:00Z assistant\n아니 이건 내가 쓴 거야"
    rendered = log.render(message(body))

    records = log.parse(rendered)
    assert len(records) == 1
    assert records[0].content == body


def test_escaping_survives_repeated_roundtrips() -> None:
    body = "\\## 2026-08-03T07:14:00Z user"
    for _ in range(3):
        (record,) = log.parse(log.render(message(body)))
        assert record.content == body


def test_parse_tolerates_hand_edited_lines() -> None:
    """A human may open the file in Obsidian and add a note. Fewer records is an
    acceptable outcome; an exception is not."""
    text = "# 2026-08-03\n\n사람이 적어둔 메모\n\n## 2026-08-03T07:14:00Z user\n안녕\n"
    (record,) = log.parse(text)
    assert record.content == "안녕"


async def test_append_writes_date_header_once(data_dir: Path) -> None:
    await log.append(data_dir, message("첫 줄"))
    await log.append(data_dir, message("둘째 줄", role="assistant"))

    text = (data_dir / "memory" / "log" / "2026-08-03.md").read_text(encoding="utf-8")
    assert text.startswith("# 2026-08-03\n")
    assert text.count("# 2026-08-03\n") == 1
    assert [r.content for r in log.parse(text)] == ["첫 줄", "둘째 줄"]


async def test_append_returns_path_relative_to_data_dir(data_dir: Path) -> None:
    assert await log.append(data_dir, message("안녕")) == "memory/log/2026-08-03.md"


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs a POSIX tzset")
async def test_a_new_day_starts_a_new_file(data_dir: Path) -> None:
    late = message("어제 얘기", ts=datetime(2026, 8, 3, 23, 59, tzinfo=UTC))
    early = message("오늘 얘기", ts=datetime(2026, 8, 4, 0, 1, tzinfo=UTC))

    with timezone("UTC"):
        await log.append(data_dir, late)
        await log.append(data_dir, early)

    log_dir = data_dir / "memory" / "log"
    assert sorted(p.name for p in log_dir.iterdir()) == ["2026-08-03.md", "2026-08-04.md"]


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs a POSIX tzset")
async def test_file_splits_on_local_day_but_timestamps_stay_utc(data_dir: Path) -> None:
    """17:00Z on the 2nd is already the 3rd in Seoul. The file follows the user's
    day; the timestamp inside stays UTC, which is why it is written in full."""
    ts = datetime(2026, 8, 2, 17, 0, 0, tzinfo=UTC)

    with timezone("Asia/Seoul"):
        assert await log.append(data_dir, message("자기 전에", ts=ts)) == "memory/log/2026-08-03.md"

    text = (data_dir / "memory" / "log" / "2026-08-03.md").read_text(encoding="utf-8")
    assert "## 2026-08-02T17:00:00Z user" in text
    assert log.parse(text)[0].ts == ts


async def test_concurrent_appends_lose_nothing(data_dir: Path) -> None:
    bodies = [f"{i}번째 메시지" for i in range(40)]
    await asyncio.gather(*(log.append(data_dir, message(b)) for b in bodies))

    text = (data_dir / "memory" / "log" / "2026-08-03.md").read_text(encoding="utf-8")
    assert text.count("# 2026-08-03\n") == 1
    assert sorted(r.content for r in log.parse(text)) == sorted(bodies)


def test_naive_timestamps_are_read_as_utc() -> None:
    assert log.utc_iso(datetime(2026, 8, 3, 7, 14, 0)) == "2026-08-03T07:14:00Z"
    assert log.from_iso("2026-08-03T07:14:00Z") == datetime(2026, 8, 3, 7, 14, 0, tzinfo=UTC)
