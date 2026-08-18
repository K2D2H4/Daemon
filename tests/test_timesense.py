"""How the daemon speaks about time.

Every string here is Korean and every one is asserted whole: these functions exist
because the model cannot do date arithmetic, so a rendering that is subtly wrong is
worse than none - it would be believed. Time is pinned to Asia/Seoul (`seoul`,
autouse) because every boundary these functions draw is a *human* one, and `now` is
always passed in rather than read from the clock.
"""

from __future__ import annotations

import time as time_module
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from daemon import timesense

NOW = datetime(2026, 8, 18, 1, 0, 0, tzinfo=UTC)      # 2026-08-18 10:00 KST, Tuesday
FRIDAY = datetime(2026, 8, 14, 7, 32, 0, tzinfo=UTC)  # 2026-08-14 16:32 KST, Friday


@pytest.fixture(autouse=True)
def seoul(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the machine's local zone. The product is Korean; so is its calendar."""
    monkeypatch.setenv("TZ", "Asia/Seoul")
    time_module.tzset()
    yield
    monkeypatch.undo()
    time_module.tzset()


def test_now_block_names_the_date_weekday_and_part_of_day() -> None:
    assert timesense.now_block(NOW) == (
        "[현재 시각] 지금은 2026년 8월 18일 화요일 오전 10시입니다. 평일 오전입니다."
    )


def test_now_block_says_weekend_on_a_saturday() -> None:
    saturday = datetime(2026, 8, 15, 5, 0, 0, tzinfo=UTC)  # 14:00 KST, Saturday
    assert timesense.now_block(saturday) == (
        "[현재 시각] 지금은 2026년 8월 15일 토요일 오후 2시입니다. 주말 오후입니다."
    )


@pytest.mark.parametrize(
    ("utc_hour", "expected"),
    [
        (18, "평일 새벽입니다."),   # 03:00 KST, 2026-08-18
        (22, "평일 아침입니다."),   # 07:00 KST, 2026-08-18
        (1, "평일 오전입니다."),    # 10:00 KST, 2026-08-17
        (4, "평일 점심입니다."),    # 13:00 KST, 2026-08-17
        (7, "평일 오후입니다."),    # 16:00 KST, 2026-08-17
        (11, "평일 저녁입니다."),   # 20:00 KST, 2026-08-17
        (14, "평일 밤입니다."),     # 23:00 KST, 2026-08-17
    ],
)
def test_day_parts_at_each_bucket(utc_hour: int, expected: str) -> None:
    """The buckets are what make "이 시간까지 안 자고 계시네요" possible, so each is
    pinned at an hour inside it. 2026-08-17 is a Monday and the two late-UTC cases
    land on Tuesday the 18th, so every case is 평일."""
    moment = datetime(2026, 8, 17, utc_hour, 0, 0, tzinfo=UTC)
    assert timesense.now_block(moment).endswith(expected)


def test_relative_names_last_week_with_the_gap() -> None:
    assert timesense.relative(FRIDAY, NOW) == "지난주 금요일 오후 4시 32분 (4일 전)"


def test_relative_can_drop_the_gap_for_use_mid_sentence() -> None:
    assert timesense.relative(FRIDAY, NOW, with_gap=False) == "지난주 금요일 오후 4시 32분"


def test_relative_uses_the_self_pinning_words_for_the_nearest_days() -> None:
    """오늘/어제/그저께 already say which day, so no "(N일 전)" is appended."""
    assert timesense.relative(datetime(2026, 8, 18, 0, 5, tzinfo=UTC), NOW) == "오늘 오전 9시 5분"
    assert timesense.relative(datetime(2026, 8, 17, 5, 0, tzinfo=UTC), NOW) == "어제 오후 2시"
    assert timesense.relative(datetime(2026, 8, 16, 5, 0, tzinfo=UTC), NOW) == "그저께 오후 2시"


def test_relative_separates_this_week_from_last_week_on_the_same_weekday() -> None:
    """The reason absolute dates stay in the string: two hits can both be 금요일."""
    this_week = datetime(2026, 8, 21, 1, 0, tzinfo=UTC)   # Fri 10:00 KST, same ISO week
    assert timesense.relative(this_week, NOW) == "이번주 금요일 오전 10시 (3일 후)"
    older = datetime(2026, 8, 7, 7, 32, tzinfo=UTC)       # Fri 16:32 KST, two weeks back
    assert timesense.relative(older, NOW) == "8월 7일 금요일 오후 4시 32분 (11일 전)"


def test_relative_reads_forwards_too() -> None:
    """`commitments` renders a due time that has not arrived yet."""
    tonight = datetime(2026, 8, 18, 11, 0, tzinfo=UTC)    # 20:00 KST today
    assert timesense.relative(tonight, NOW) == "오늘 오후 8시"
    tomorrow = datetime(2026, 8, 19, 11, 0, tzinfo=UTC)
    assert timesense.relative(tomorrow, NOW) == "내일 오후 8시"


def test_relative_crosses_the_year_boundary_without_claiming_last_week() -> None:
    """`isocalendar` weeks renumber at the year edge; the Monday-difference does not."""
    now = datetime(2027, 1, 5, 1, 0, tzinfo=UTC)          # Tue 10:00 KST
    last_week = datetime(2026, 12, 29, 1, 0, tzinfo=UTC)  # Tue 10:00 KST, one week back
    assert timesense.relative(last_week, now) == "지난주 화요일 오전 10시 (7일 전)"
