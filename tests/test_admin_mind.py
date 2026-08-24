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

import daemon.admin.mind as mind
from daemon.admin.mind import memory_payload
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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_a_day_with_an_artifact_and_no_row_is_still_listed(
    tmp_path: Path, store: Store
) -> None:
    """reflection_runs arrived in M5; the artifacts predate it. Measured on the
    real install: 5 rows, 9 artifacts. Keyed on the table, four days vanish."""
    _write(tmp_path / "memory" / "reflections" / "2026-08-14.md", "# 2026-08-14 성찰\n")
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "# 2026-08-19 성찰\n")
    store.record_reflection_run(
        now=_dt(19, 19), date="2026-08-19", status="written",
        messages_read=72, facts=1, entities=2, observations=2, detail="",
    )   # `now=`, not `ts=` - verified against store.py:1471

    payload = memory_payload(store, tmp_path)
    dates = [r["date"] for r in payload["reflections"]]

    assert dates == ["2026-08-19", "2026-08-14"]        # newest first
    assert payload["reflections_total"] == 2
    assert payload["reflections_recorded"] == 1
    recorded, only_file = payload["reflections"]
    assert recorded["status"] == "written"
    assert recorded["messages_read"] == 72
    assert only_file["status"] is None                   # artifact only
    assert only_file["body"] == "# 2026-08-14 성찰\n"


def test_b_retired_facts_are_kept_and_counted_apart(tmp_path: Path, store: Store) -> None:
    _fact(store, "활성", importance=9, key="k")
    _fact(store, "대체", importance=8, key="k")          # retires the first

    payload = memory_payload(store, tmp_path)

    assert payload["facts_active"] == 1
    assert payload["facts_retired"] == 1
    assert {f["status"] for f in payload["facts"]} == {"active", "retired"}
    # importance DESC: the active one leads.
    assert payload["facts"][0]["body"] == "활성"


def test_c_the_body_cap_drops_bodies_not_list_entries(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mind, "MAX_REFLECTION_BODIES", 2)
    for day in (10, 11, 13, 14):
        _write(tmp_path / "memory" / "reflections" / f"2026-08-{day}.md", f"day {day}\n")

    payload = memory_payload(store, tmp_path)

    assert [r["date"] for r in payload["reflections"]] == [
        "2026-08-14", "2026-08-13", "2026-08-11", "2026-08-10"
    ]
    assert payload["reflections_bodies_truncated"] is True
    assert payload["reflections"][0]["body"] == "day 14\n"
    assert payload["reflections"][1]["body"] == "day 13\n"
    assert payload["reflections"][2]["body"] is None     # listed, no body
    assert payload["reflections"][2]["file"] == "memory/reflections/2026-08-11.md"


def test_c_the_byte_budget_also_only_drops_bodies(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mind, "MAX_BODY_BYTES", 40)
    for day in (10, 11, 13):
        _write(tmp_path / "memory" / "reflections" / f"2026-08-{day}.md", "x" * 30)

    payload = memory_payload(store, tmp_path)

    assert len(payload["reflections"]) == 3
    assert payload["reflections_bodies_truncated"] is True
    assert sum(1 for r in payload["reflections"] if r["body"] is not None) == 1


def test_entities_carry_their_note_and_links(tmp_path: Path, store: Store) -> None:
    _write(tmp_path / "memory" / "entities" / "UJET.cx.md", "# UJET.cx\n\n회사.\n")
    _write(tmp_path / "memory" / "entities" / "Schubert Chin.md", "# Schubert Chin\n")
    a = store.upsert_entity(
        name="UJET.cx", kind="company", file="memory/entities/UJET.cx.md", now=_dt(19)
    )
    b = store.upsert_entity(
        name="Schubert Chin", kind="person",
        file="memory/entities/Schubert Chin.md", now=_dt(19),
    )
    store.set_mention_count(a, 3)
    store.set_mention_count(b, 2)
    store.link_entities(a, b)

    payload = memory_payload(store, tmp_path)
    first = payload["entities"][0]

    assert first["name"] == "UJET.cx"                    # mention_count DESC
    assert first["kind"] == "company"
    assert first["mentions"] == 3
    assert first["links"] == ["Schubert Chin"]
    assert first["body"] == "# UJET.cx\n\n회사.\n"
    assert payload["entities_total"] == 2


def test_a_missing_note_file_is_a_null_body_not_a_crash(
    tmp_path: Path, store: Store
) -> None:
    """The real install has an entities row ('벨라') whose note file is absent."""
    store.upsert_entity(
        name="벨라", kind=None, file="memory/entities/벨라.md", now=_dt(19)
    )

    payload = memory_payload(store, tmp_path)

    assert payload["entities"][0]["name"] == "벨라"
    assert payload["entities"][0]["body"] is None


def test_a_second_reflection_run_for_the_same_date_wins(
    tmp_path: Path, store: Store
) -> None:
    """`recent_reflection_runs` is id DESC (store.py:1513); `memory_payload`
    reverses it before folding into a dict so the *last-inserted* (newest) row
    for a date overwrites the first, rather than the other way around."""
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "# 2026-08-19\n")
    store.record_reflection_run(
        now=_dt(19, 10), date="2026-08-19", status="written",
        messages_read=10, facts=1, entities=1, observations=1, detail="first pass",
    )
    store.record_reflection_run(
        now=_dt(19, 20), date="2026-08-19", status="skipped",
        messages_read=99, facts=9, entities=9, observations=9, detail="second pass",
    )

    payload = memory_payload(store, tmp_path)

    assert len(payload["reflections"]) == 1
    row = payload["reflections"][0]
    assert row["status"] == "skipped"
    assert row["messages_read"] == 99
    assert row["facts"] == 9
    assert row["entities"] == 9
    assert row["observations"] == 9
    assert row["detail"] == "second pass"


def test_the_body_budget_is_shared_and_entities_go_first(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`BodyBudget`'s own docstring says one budget is shared across every
    section. Give the entity note and the reflection artifact bodies that
    cannot both fit under one small `MAX_BODY_BYTES`, and confirm exactly one
    survives - the entity's, because `memory_payload` spends the shared budget
    on entities before it reaches reflections."""
    monkeypatch.setattr(mind, "MAX_BODY_BYTES", 20)
    _write(tmp_path / "memory" / "entities" / "E.md", "E" * 15)
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "R" * 15)
    store.upsert_entity(name="E", kind=None, file="memory/entities/E.md", now=_dt(19))

    payload = memory_payload(store, tmp_path)

    assert payload["entities"][0]["body"] == "E" * 15
    assert payload["entities_bodies_truncated"] is False
    assert payload["reflections"][0]["body"] is None
    assert payload["reflections_bodies_truncated"] is True
