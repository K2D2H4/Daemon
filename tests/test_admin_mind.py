"""M5 admin, the Memory and Persona tabs — what she knows and what she learned.

Every other tab answers "what did she do". These two answer "what does she know,
and how has she worked out to deal with me". The properties worth testing are
about what stays *visible* and what stays *unwritable*:

  a. a day that has a reflection artifact but no `reflection_runs` row is still
     listed — the table arrived in M5 and the files predate it.
  b. retired facts and rules are separated from active ones, not dropped.
  c. the body caps drop bodies, never list entries, and say so.
  d. `forget` on a diverged `learned.md` refuses *with the reason*.
  e. the two "run now" endpoints take the same lock the crons take.
  f. no route writes `persona/seed.md`.

Loopback `base_url` for the same reason as `test_admin.py`: the router refuses any
Host that is not loopback, which is what defeats DNS-rebinding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from daemon.memory.store import Store


def _dt(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path):
    store = Store.open(tmp_path / "daemon.sqlite3")
    yield store
    store.close()


def _fact(store: Store, body: str, *, importance: int, key: str | None = None) -> int:
    return store.insert_entry(
        body=body,
        importance=importance,
        trigger_phrases=[],
        origin="agent",
        session_kind="reflection",
        modality="text",
        now=_dt(19),
        supersession_key=key,
    )


def test_recent_entries_includes_retired_and_active_entries_does_not(store: Store) -> None:
    keep = _fact(store, "활성 사실", importance=9)
    old = _fact(store, "낡은 사실", importance=5, key="k")
    # A new fact on the same key retires `old`.
    new = _fact(store, "새 사실", importance=6, key="k")

    active_ids = {int(r["id"]) for r in store.active_entries(50)}
    recent = store.recent_entries(50)
    recent_ids = {int(r["id"]) for r in recent}

    assert old not in active_ids
    assert old in recent_ids
    assert keep in recent_ids
    # Retired rows carry the status the view splits on.
    by_id = {int(r["id"]): r for r in recent}
    assert by_id[old]["status"] == "retired"
    assert by_id[keep]["status"] == "active"
    # Most important first (importances 9, 6, 5) - the docstring's promise, not
    # just insertion or id order.
    assert [int(r["id"]) for r in recent] == [keep, new, old]


def test_recent_observations_includes_consumed_ones(store: Store) -> None:
    pending = store.insert_observation(
        body="아직 안 쓰인 관찰", observed_from="2026-08-20/2026-08-20",
        confidence=0.7, now=_dt(20),
    )
    used = store.insert_observation(
        body="규칙이 먹은 관찰", observed_from="2026-08-19/2026-08-19",
        confidence=0.8, now=_dt(19),
    )
    rule = store.insert_persona_rule(
        body="규칙", created_at=_dt(19), evidence=[used], supersession_key=None
    )
    store.consume_observations([used], rule)

    unconsumed = {int(r["id"]) for r in store.unconsumed_observations()}
    rows = store.recent_observations(50)
    ids = {int(r["id"]) for r in rows}

    assert unconsumed == {pending}
    assert ids == {pending, used}
    by_id = {int(r["id"]): r for r in rows}
    assert by_id[used]["consumed_by"] == rule
    assert by_id[pending]["consumed_by"] is None
    # Newest first: the view reads top-down as "most recent thing she noticed".
    assert [int(r["id"]) for r in rows] == [pending, used]


def test_retired_persona_rules_carries_when_and_why(store: Store) -> None:
    kept = store.insert_persona_rule(
        body="살아있는 규칙", created_at=_dt(9), evidence=[], supersession_key=None
    )
    # Inserted (and so id-ordered) before `gone`, but retired *later* below.
    retired_earlier = store.insert_persona_rule(
        body="먼저 은퇴할 규칙", created_at=_dt(10), evidence=[], supersession_key=None
    )
    gone = store.insert_persona_rule(
        body="은퇴할 규칙", created_at=_dt(11), evidence=[], supersession_key=None
    )
    # `retired_earlier` has the lower id but the earlier `retired_at`; `gone` has
    # the higher id but the later `retired_at`. So the correct `retired_at DESC`
    # order - [gone, retired_earlier] - is the *reverse* of sqlite's natural
    # rowid scan order. That makes a missing or reversed ORDER BY visible: either
    # one would return [retired_earlier, gone] instead.
    assert store.retire_persona_rule(retired_earlier, when=_dt(20), why="중복")
    assert store.retire_persona_rule(gone, when=_dt(24), why="사용자가 지우라고 했다")

    rows = store.retired_persona_rules(50)
    assert [int(r["id"]) for r in rows] == [gone, retired_earlier]
    assert rows[0]["retired_why"] == "사용자가 지우라고 했다"
    assert rows[0]["retired_at"] == "2026-08-24T12:00:00Z"
    assert {int(r["id"]) for r in store.active_persona_rules()} == {kept}
