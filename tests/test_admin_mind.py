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

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import daemon.admin.mind as mind
from daemon.admin.mind import memory_payload, persona_payload
from daemon.app import create_app
from daemon.config import Settings
from daemon.memory.store import Store

# `TestClient` defaults to `Host: testserver`; the router refuses any Host that
# is not loopback (`test_admin.py`'s reasoning applies here unchanged).
LOOPBACK = "http://127.0.0.1"


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
    # Inserted (and so id-ordered) before `pending`, but with an *earlier*
    # `created_at` below - so the correct `created_at DESC` order, [pending,
    # used], is the reverse of sqlite's natural rowid scan order. That makes a
    # missing or reversed ORDER BY visible: either one would return
    # [used, pending] instead.
    used = store.insert_observation(
        body="규칙이 먹은 관찰", observed_from="2026-08-19/2026-08-19",
        confidence=0.8, now=_dt(19),
    )
    pending = store.insert_observation(
        body="아직 안 쓰인 관찰", observed_from="2026-08-20/2026-08-20",
        confidence=0.7, now=_dt(20),
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


def test_fact_counts_and_list_truncated_read_the_table_not_the_window(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`facts_active`/`facts_retired` used to sum over the MAX_FACTS-capped
    window (`fact_rows`), so a corpus past the cap under-reported both while
    the list looked complete. COUNT(*) fixes the totals; `facts_list_truncated`
    is the honest flag once the fetched rows fall short of them."""
    monkeypatch.setattr(mind, "MAX_FACTS", 2)
    _fact(store, "하나", importance=5)
    _fact(store, "둘", importance=4)
    _fact(store, "셋", importance=3)

    payload = memory_payload(store, tmp_path)

    assert len(payload["facts"]) == 2                 # window, still capped
    assert payload["facts_active"] == 3                # table truth
    assert payload["facts_retired"] == 0
    assert payload["facts_list_truncated"] is True


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
    assert payload["entities_list_truncated"] is False


def test_entities_total_and_list_truncated_read_the_table_not_the_window(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same bug as facts, on `entities_total`: it used to be `len(entities)`,
    the MAX_ENTITIES-capped list, not a table count."""
    monkeypatch.setattr(mind, "MAX_ENTITIES", 1)
    store.upsert_entity(name="A", kind=None, file="memory/entities/A.md", now=_dt(19))
    store.upsert_entity(name="B", kind=None, file="memory/entities/B.md", now=_dt(19))

    payload = memory_payload(store, tmp_path)

    assert len(payload["entities"]) == 1
    assert payload["entities_total"] == 2
    assert payload["entities_list_truncated"] is True


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
    """`reflection_runs_by_dates` orders by id ASC and folds rows into a dict
    by date, so the *last-inserted* (newest) row for a date overwrites the
    first, rather than the other way around."""
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


def test_reflection_runs_by_dates_keeps_the_newest_row_and_ignores_others(
    store: Store,
) -> None:
    """Direct `Store` coverage for the method `memory_payload` now resolves
    reflection status through, mirroring `observations_by_ids`: a date with two
    passes keeps the later row, an unrequested date is absent even though its
    row exists, and a requested date with no row is simply missing (not a
    KeyError)."""
    store.record_reflection_run(
        now=_dt(19, 10), date="2026-08-19", status="written",
        messages_read=1, facts=1, entities=1, observations=1, detail="first",
    )
    store.record_reflection_run(
        now=_dt(19, 20), date="2026-08-19", status="skipped",
        messages_read=2, facts=2, entities=2, observations=2, detail="second",
    )
    store.record_reflection_run(
        now=_dt(1), date="2026-08-01", status="written",
        messages_read=3, facts=3, entities=3, observations=3, detail="not requested",
    )

    got = store.reflection_runs_by_dates(["2026-08-19", "2026-08-14"])

    assert set(got) == {"2026-08-19"}
    assert got["2026-08-19"]["detail"] == "second"


def test_reflection_runs_by_dates_of_an_empty_list_is_empty(store: Store) -> None:
    assert store.reflection_runs_by_dates([]) == {}


def test_a_reflection_row_resolves_without_the_capped_lookup(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this fixes: `recent_reflection_runs(N)` is `ORDER BY id DESC
    LIMIT N` - a fixed row-count window that quietly drops old rows once a
    table outgrows it, so an old day's real status renders as `artifact
    only`. `memory_payload` must resolve by date instead, and must not touch
    `recent_reflection_runs` to do it - breaking that call here proves it."""
    monkeypatch.setattr(
        Store,
        "recent_reflection_runs",
        lambda self, *a, **k: (_ for _ in ()).throw(
            AssertionError("memory_payload must not call recent_reflection_runs")
        ),
    )
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "# 2026-08-19\n")
    store.record_reflection_run(
        now=_dt(19), date="2026-08-19", status="written",
        messages_read=5, facts=1, entities=1, observations=1, detail="",
    )

    payload = memory_payload(store, tmp_path)

    assert payload["reflections"][0]["status"] == "written"
    assert payload["reflections_recorded"] == 1


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


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(_env_file=None, provider="ollama", data_dir=tmp_path, **kw)


def test_the_anchor_reads_the_caps_and_both_files(tmp_path: Path, store: Store) -> None:
    _write(tmp_path / "persona" / "seed.md", "# seed\n\n- 너는 벨라다.\n")
    _write(tmp_path / "persona" / "learned.md", "# learned\n\n- 규칙 하나.\n")
    store.insert_persona_rule(
        body="규칙 하나.", created_at=_dt(9), evidence=[], supersession_key=None
    )
    store.insert_observation(
        body="아직 안 쓰인 관찰", observed_from="2026-08-20/2026-08-20",
        confidence=0.7, now=_dt(20),
    )

    settings = _settings(
        tmp_path,
        persona_max_active_rules=12,
        persona_max_new_per_cycle=4,
        persona_min_observations=9,
    )
    payload = persona_payload(store, tmp_path, settings)
    anchor = payload["anchor"]

    assert anchor["active"] == 1
    # Distinct, non-default values, so the test cannot pass against a
    # hardcoded 20/3/5 - it must actually be reading `settings`.
    assert anchor["max_active"] == 12
    assert anchor["max_new_per_cycle"] == 4
    assert anchor["min_observations"] == 9
    assert anchor["unconsumed"] == 1
    assert anchor["last_rule_at"] == "2026-08-09T12:00:00Z"
    assert anchor["seed"]["lines"] == 3
    assert anchor["seed"]["file"] == "persona/seed.md"
    assert "너는 벨라다" in anchor["seed"]["text"]
    assert anchor["learned"]["lines"] == 3


def test_anchor_unconsumed_reads_the_table_not_the_window(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`anchor.unconsumed` used to sum over the MAX_OBSERVATIONS-capped window -
    the same bug class as `observations_total`/`observations_consumed`, and it
    sits right next to `observations_list_truncated`, which already admits the
    cap is in effect. `observations_total - observations_consumed` is table
    truth regardless of the window."""
    monkeypatch.setattr(mind, "MAX_OBSERVATIONS", 1)
    store.insert_observation(
        body="첫", observed_from="2026-08-01/2026-08-01", confidence=0.5, now=_dt(1),
    )
    store.insert_observation(
        body="둘째", observed_from="2026-08-02/2026-08-02", confidence=0.5, now=_dt(2),
    )
    store.insert_observation(  # newest - the one row the window keeps
        body="셋째", observed_from="2026-08-20/2026-08-20", confidence=0.5, now=_dt(20),
    )

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert len(payload["observations"]) == 1          # window, still capped
    assert payload["observations_list_truncated"] is True
    assert payload["anchor"]["unconsumed"] == 3        # table truth, not 1


def test_a_rule_carries_its_evidence_as_sentences(tmp_path: Path, store: Store) -> None:
    """`evidence` is a list of observation ids. A screen showing '3 observations'
    and not which three is the blindness this tab exists to fix."""
    first = store.insert_observation(
        body="솔직하게 인정하는 소통을 선호한다.",
        observed_from="2026-08-06/2026-08-06", confidence=0.85, now=_dt(6),
    )
    second = store.insert_observation(
        body="오답을 꼼꼼하게 검증한다.",
        observed_from="2026-08-07/2026-08-07", confidence=0.8, now=_dt(7),
    )
    rule = store.insert_persona_rule(
        body="변명 없이 인정하라.", created_at=_dt(9),
        evidence=[first, second], supersession_key=None,
    )
    store.consume_observations([first, second], rule)

    payload = persona_payload(store, tmp_path, _settings(tmp_path))
    got = payload["rules"][0]

    assert got["id"] == rule
    assert got["status"] == "active"
    assert [e["id"] for e in got["evidence"]] == [first, second]
    assert got["evidence"][0]["body"] == "솔직하게 인정하는 소통을 선호한다."
    assert got["evidence"][0]["confidence"] == 0.85
    assert payload["observations_total"] == 2
    assert payload["observations_consumed"] == 2


def test_b_a_retired_rule_stays_with_its_reason(tmp_path: Path, store: Store) -> None:
    """learned.md is rewritten whole, so a vanished rule leaves no trace there."""
    gone = store.insert_persona_rule(
        body="틀린 규칙", created_at=_dt(10), evidence=[], supersession_key=None
    )
    store.retire_persona_rule(gone, when=_dt(24), why="사용자가 아니라고 했다")

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert payload["rules_active"] == 0
    assert payload["rules_retired"] == 1
    retired = [r for r in payload["rules"] if r["status"] == "retired"][0]
    assert retired["retired_why"] == "사용자가 아니라고 했다"
    assert retired["retired_at"] == "2026-08-24T12:00:00Z"


def test_an_evidence_id_with_no_observation_row_is_skipped(
    tmp_path: Path, store: Store
) -> None:
    """`evidence` is model-supplied json. A stale id must not 500 the tab."""
    store.insert_persona_rule(
        body="근거가 사라진 규칙", created_at=_dt(9),
        evidence=[4242], supersession_key=None,
    )

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert payload["rules"][0]["evidence"] == []


def test_evidence_cited_distinguishes_no_evidence_from_lost_evidence(
    tmp_path: Path, store: Store
) -> None:
    """`rebuild()` (persona/rules.py) restores a rule with `evidence='[]'` when
    learned.md never recorded any - that must not read the same as a rule
    whose cited ids all lost their observation row. `evidence_cited` is the
    count the rule claims, before resolution; `evidence` is empty in both
    cases, so the front end needs `evidence_cited` to tell them apart."""
    never_cited = store.insert_persona_rule(
        body="learned.md never recorded evidence for this",
        created_at=_dt(9), evidence=[], supersession_key=None,
    )
    lost_its_row = store.insert_persona_rule(
        body="근거가 사라진 규칙", created_at=_dt(9),
        evidence=[4242], supersession_key=None,
    )

    payload = persona_payload(store, tmp_path, _settings(tmp_path))
    by_id = {r["id"]: r for r in payload["rules"]}

    assert by_id[never_cited]["evidence"] == []
    assert by_id[never_cited]["evidence_cited"] == 0
    assert by_id[lost_its_row]["evidence"] == []
    assert by_id[lost_its_row]["evidence_cited"] == 1


def test_evidence_rejects_a_bool_masquerading_as_an_id(
    tmp_path: Path, store: Store
) -> None:
    """`int(True)` is `1`. A rule whose evidence json is `[true]` - which
    `insert_persona_rule` will happily store, since `bool` is an `int`
    subclass - must not resolve to observation id 1. This module treats model
    output as hostile."""
    obs_id = store.insert_observation(
        body="첫 번째 관찰", observed_from="2026-08-01/2026-08-01",
        confidence=0.5, now=_dt(1),
    )
    assert obs_id == 1  # the id `int(True)` would coerce to, in a fresh store
    rule = store.insert_persona_rule(
        body="불리언 근거로 만든 규칙", created_at=_dt(9),
        evidence=[True], supersession_key=None,
    )

    payload = persona_payload(store, tmp_path, _settings(tmp_path))
    got = next(r for r in payload["rules"] if r["id"] == rule)

    assert got["evidence"] == []
    assert got["evidence_cited"] == 0


def test_rule_evidence_resolves_past_the_capped_observation_window(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`observations` (the list this payload also returns) is capped at
    MAX_OBSERVATIONS. A rule's evidence used to be looked up in that same
    capped list, so an old id that fell out of the window rendered as if the
    row had been deleted - the worst of the silent-ceiling bugs, since it made
    an affirmatively false claim about a row still in the table. Evidence must
    resolve against the table directly, by id."""
    monkeypatch.setattr(mind, "MAX_OBSERVATIONS", 1)
    old = store.insert_observation(
        body="오래된 관찰", observed_from="2026-08-01/2026-08-01",
        confidence=0.6, now=_dt(1),
    )
    store.insert_observation(  # newer - the one row the cap keeps
        body="최근 관찰", observed_from="2026-08-20/2026-08-20",
        confidence=0.6, now=_dt(20),
    )
    rule = store.insert_persona_rule(
        body="오래된 근거로 만든 규칙", created_at=_dt(9),
        evidence=[old], supersession_key=None,
    )

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert len(payload["observations"]) == 1          # `old` fell out of the window
    got = payload["rules"][0]
    assert got["id"] == rule
    assert [e["id"] for e in got["evidence"]] == [old]
    assert got["evidence"][0]["body"] == "오래된 관찰"


def test_observation_counts_and_list_truncated_read_the_table_not_the_window(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mind, "MAX_OBSERVATIONS", 1)
    first = store.insert_observation(
        body="첫", observed_from="2026-08-06/2026-08-06", confidence=0.5, now=_dt(6),
    )
    store.insert_observation(
        body="둘째", observed_from="2026-08-07/2026-08-07", confidence=0.5, now=_dt(7),
    )
    rule = store.insert_persona_rule(
        body="규칙", created_at=_dt(9), evidence=[first], supersession_key=None
    )
    store.consume_observations([first], rule)

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert len(payload["observations"]) == 1          # window, still capped
    assert payload["observations_total"] == 2          # table truth
    assert payload["observations_consumed"] == 1
    assert payload["observations_list_truncated"] is True


def test_rules_list_truncated_when_retired_rules_exceed_the_cap(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mind, "MAX_RULES", 1)
    first = store.insert_persona_rule(
        body="첫 규칙", created_at=_dt(9), evidence=[], supersession_key=None
    )
    second = store.insert_persona_rule(
        body="둘째 규칙", created_at=_dt(10), evidence=[], supersession_key=None
    )
    store.retire_persona_rule(first, when=_dt(11), why="이유1")
    store.retire_persona_rule(second, when=_dt(12), why="이유2")

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert payload["rules_retired"] == 1               # capped display
    assert payload["rules_list_truncated"] is True


def test_c_the_diary_cap_drops_bodies_not_entries(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mind, "MAX_DIARY_BODIES", 1)
    _write(tmp_path / "persona" / "diary" / "2026-08-10.md", "10\n")
    _write(tmp_path / "persona" / "diary" / "2026-08-24.md", "24\n")

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert [d["date"] for d in payload["diaries"]] == ["2026-08-24", "2026-08-10"]
    assert payload["diaries"][0]["body"] == "24\n"
    assert payload["diaries"][1]["body"] is None
    assert payload["diaries_bodies_truncated"] is True


def test_missing_seed_and_learned_are_null_not_a_crash(
    tmp_path: Path, store: Store
) -> None:
    """A fresh install before the first evolution has neither file."""
    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert payload["anchor"]["seed"]["text"] is None
    # A missing file has no lines to report either - `0` would claim an empty
    # file that never existed.
    assert payload["anchor"]["seed"]["lines"] is None
    assert payload["anchor"]["seed"]["truncated"] is False   # absent, not cut
    assert payload["anchor"]["learned"]["text"] is None


def test_the_persona_body_budget_is_shared_across_diary_seed_and_learned(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`persona_payload` threads one `BodyBudget` through the whole payload
    (`BodyBudget`'s own docstring). It is spent in the order the code actually
    reads: the diary bodies first (`_bodies` runs before the return dict is
    built), then `seed.md`, then `learned.md`. Size the three bodies so the
    diary and the seed both fit but nothing is left for `learned.md`, and
    confirm the starvation lands there rather than each file getting its own
    budget. Also proves `learned.md`'s `lines`/`truncated` tell the starved
    case (a file that exists but lost to the budget) apart from a missing
    file: `lines` is `None`, not the fabricated `0` a file that was never read
    used to report, and `truncated` is `True` only for the file that was cut."""
    monkeypatch.setattr(mind, "MAX_BODY_BYTES", 25)
    _write(tmp_path / "persona" / "diary" / "2026-08-10.md", "D" * 10)
    _write(tmp_path / "persona" / "seed.md", "S" * 10)
    _write(tmp_path / "persona" / "learned.md", "L" * 10)

    payload = persona_payload(store, tmp_path, _settings(tmp_path))

    assert payload["diaries"][0]["body"] == "D" * 10
    assert payload["anchor"]["seed"]["text"] == "S" * 10
    assert payload["anchor"]["seed"]["truncated"] is False
    assert payload["anchor"]["learned"]["text"] is None
    assert payload["anchor"]["learned"]["lines"] is None
    assert payload["anchor"]["learned"]["truncated"] is True


def test_the_read_endpoints_serve_the_payloads_over_loopback(tmp_path: Path) -> None:
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "# 성찰\n")
    _write(tmp_path / "persona" / "seed.md", "# seed\n")
    # 7, not the field's own default of 20 - a non-default value here proves the
    # route threads `request.app.state.settings` through to `persona_payload`,
    # rather than a fresh `Settings()` that would coincidentally also read 20.
    app = create_app(_settings(tmp_path, persona_max_active_rules=7))
    with TestClient(app, base_url=LOOPBACK) as client:
        memory = client.get("/admin/api/memory")
        persona = client.get("/admin/api/persona")

    assert memory.status_code == 200
    assert [r["date"] for r in memory.json()["reflections"]] == ["2026-08-19"]
    assert persona.status_code == 200
    assert persona.json()["anchor"]["max_active"] == 7


def test_f_no_route_writes_the_seed() -> None:
    """CONTRACTS non-negotiable 5. Asserted on the router, not on a code review:
    a later hand could add a PATCH and every other test would stay green."""
    from daemon.admin import routes

    seed_routes = [
        (route.path, sorted(getattr(route, "methods", set())))
        for route in routes.router.routes
        if "seed" in route.path
    ]
    assert seed_routes == []

    persona_writes = [
        (route.path, sorted(getattr(route, "methods", set())))
        for route in routes.router.routes
        if route.path.startswith("/admin/api/persona")
        and set(getattr(route, "methods", set())) - {"GET", "HEAD"}
    ]
    # Only `forget` and `evolve` write anything under /persona, and neither
    # touches seed.md - see Task 6.
    assert {path for path, _ in persona_writes} <= {
        "/admin/api/persona/forget", "/admin/api/persona/evolve"
    }


def test_e_the_app_exposes_the_catchup_lock(tmp_path: Path) -> None:
    """Without this the two run-now endpoints cannot take the lock the crons take,
    and a click during the 04:00 cron double-writes the append-only artifact
    (daemon/app.py:232-237). A type check alone is not enough - a fresh, unshared
    `asyncio.Lock()` would pass `isinstance` while serialising nothing against the
    crons, so this asserts identity against the exact lock object each scheduled
    job was handed."""
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK):
        lock = app.state.catchup_lock
        assert isinstance(lock, asyncio.Lock)
        reflect_job = app.state.scheduler.get_job("reflection")
        persona_job = app.state.scheduler.get_job("persona-evolution")
        assert reflect_job.args[1] is lock
        assert persona_job.args[1] is lock


@pytest.mark.asyncio
async def test_run_reflection_now_raises_where_the_tick_swallows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The tick logs and returns None so APScheduler carries on. The endpoint
    needs the failure, or the browser reports success for a pass that never ran."""
    from daemon import app as app_mod

    async def boom(settings):
        raise RuntimeError("no provider")

    monkeypatch.setattr(app_mod, "build_reflection", boom)

    with pytest.raises(RuntimeError, match="no provider"):
        await app_mod.run_reflection_now(_settings(tmp_path), None)

    # The tick still swallows it - that contract does not change. It must not
    # raise, and it must say so at ERROR rather than going silent.
    with caplog.at_level("ERROR"):
        await app_mod._reflect_tick(_settings(tmp_path), None)
    assert "reflection tick failed" in caplog.text


@pytest.mark.asyncio
async def test_run_reflection_now_holds_the_lock_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daemon import app as app_mod

    held = asyncio.Event()
    release = asyncio.Event()
    closed = False

    class SlowReflection:
        async def catch_up(self):
            held.set()
            await release.wait()
            return []

    async def build(settings):
        async def close() -> None:
            nonlocal closed
            closed = True

        return SlowReflection(), close

    monkeypatch.setattr(app_mod, "build_reflection", build)
    lock = asyncio.Lock()
    task = asyncio.create_task(app_mod.run_reflection_now(_settings(tmp_path), lock))
    await held.wait()

    assert lock.locked()
    # The gateway/store is still open while the work runs - `close()` fires only
    # once the pass is done, in the `finally`.
    assert not closed
    release.set()
    assert await task == []
    assert not lock.locked()
    assert closed


@pytest.mark.asyncio
async def test_run_reflection_now_closes_the_gateway_even_when_the_work_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`close()` sits in a `finally` precisely so a pass that blows up mid-way
    does not leak the provider connection. A deleted `finally` still passes
    every other test in this file, so this is the one that catches it."""
    from daemon import app as app_mod

    closed = False

    class BoomingReflection:
        async def catch_up(self):
            raise RuntimeError("model unavailable")

    async def build(settings):
        async def close() -> None:
            nonlocal closed
            closed = True

        return BoomingReflection(), close

    monkeypatch.setattr(app_mod, "build_reflection", build)

    with pytest.raises(RuntimeError, match="model unavailable"):
        await app_mod.run_reflection_now(_settings(tmp_path), None)

    assert closed


@pytest.mark.asyncio
async def test_run_persona_evolution_now_raises_where_the_tick_swallows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Mirrors the reflection case on the other 'run now' endpoint: a raising
    APScheduler job silently un-schedules the weekly persona pass, which is
    exactly what `_persona_tick`'s swallow exists to prevent."""
    from daemon import app as app_mod

    async def boom(settings):
        raise RuntimeError("no provider")

    monkeypatch.setattr(app_mod, "build_persona_evolution", boom)

    with pytest.raises(RuntimeError, match="no provider"):
        await app_mod.run_persona_evolution_now(_settings(tmp_path), None)

    # The tick still swallows it - that contract does not change for persona
    # either.
    with caplog.at_level("ERROR"):
        await app_mod._persona_tick(_settings(tmp_path), None)
    assert "persona evolution tick failed" in caplog.text


@pytest.mark.asyncio
async def test_run_persona_evolution_now_holds_the_lock_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same proof as the reflection helper, for the second 'run now' endpoint -
    the file's own header claims both take the same lock, but only reflection
    was ever tested. Also exercises `force=`, which `run_reflection_now` has no
    equivalent of."""
    from daemon import app as app_mod
    from daemon.persona.evolve import EvolutionResult

    held = asyncio.Event()
    release = asyncio.Event()
    closed = False
    seen_force: bool | None = None

    class SlowEvolution:
        async def run(self, *, now=None, force: bool = False) -> EvolutionResult:
            nonlocal seen_force
            seen_force = force
            held.set()
            await release.wait()
            return EvolutionResult(
                date="2026-08-24",
                observations_read=0,
                proposed=0,
                added=0,
                retired=0,
                skipped="",
                problems=(),
            )

    async def build(settings):
        async def close() -> None:
            nonlocal closed
            closed = True

        return SlowEvolution(), close

    monkeypatch.setattr(app_mod, "build_persona_evolution", build)
    lock = asyncio.Lock()
    task = asyncio.create_task(
        app_mod.run_persona_evolution_now(_settings(tmp_path), lock, force=True)
    )
    await held.wait()

    assert lock.locked()
    assert not closed
    release.set()
    result = await task
    assert result.date == "2026-08-24"
    assert not lock.locked()
    assert closed
    assert seen_force is True


def test_d_forget_refuses_a_diverged_file_with_the_reason(tmp_path: Path) -> None:
    """The point of LearnedFileDiverged (daemon/cli.py:1237-1239): the rewrite is
    computed from the mirror, so forgetting one rule on a diverged file would take
    every orphaned bullet with it. A generic 500 reads as a broken button."""
    from daemon.app import DB_FILENAME

    store = Store.open(tmp_path / DB_FILENAME)
    rule = store.insert_persona_rule(
        body="미러가 아는 규칙", created_at=_dt(9), evidence=[], supersession_key=None
    )
    store.close()
    _write(
        tmp_path / "persona" / "learned.md",
        "# learned\n\n- 미러가 아는 규칙\n- 손으로 적은 줄\n",
    )

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/forget", json={"id": rule, "why": "지워"})
        persona = client.get("/admin/api/persona").json()

    assert r.status_code == 409
    assert "손으로 적은 줄" in r.json()["detail"]
    # And nothing was written: the mirror row is still active and the file untouched.
    active_ids = {rule["id"] for rule in persona["rules"] if rule["status"] == "active"}
    assert rule in active_ids
    text = (tmp_path / "persona" / "learned.md").read_text(encoding="utf-8")
    assert "손으로 적은 줄" in text
    assert "미러가 아는 규칙" in text


def test_forget_retires_the_rule_and_rewrites_the_file(tmp_path: Path) -> None:
    from daemon.app import DB_FILENAME

    store = Store.open(tmp_path / DB_FILENAME)
    keep = store.insert_persona_rule(
        body="남을 규칙", created_at=_dt(9), evidence=[], supersession_key=None
    )
    gone = store.insert_persona_rule(
        body="지울 규칙", created_at=_dt(10), evidence=[], supersession_key=None
    )
    store.close()
    _write(tmp_path / "persona" / "learned.md", "# learned\n\n- 남을 규칙\n- 지울 규칙\n")

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post(
            "/admin/api/persona/forget", json={"id": gone, "why": "사용자가 아니라고 했다"}
        )
        assert r.status_code == 200, r.text
        payload = client.get("/admin/api/persona").json()

    text = (tmp_path / "persona" / "learned.md").read_text(encoding="utf-8")
    assert "지울 규칙" not in text
    assert "남을 규칙" in text
    assert payload["rules_active"] == 1
    assert payload["rules_retired"] == 1
    retired = [rule for rule in payload["rules"] if rule["status"] == "retired"][0]
    assert retired["id"] == gone
    assert retired["retired_why"] == "사용자가 아니라고 했다"
    assert keep in {rule["id"] for rule in payload["rules"] if rule["status"] == "active"}


def test_forget_rejects_an_empty_why_and_an_unknown_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        blank = client.post("/admin/api/persona/forget", json={"id": 1, "why": "   "})
        missing = client.post("/admin/api/persona/forget", json={"id": 4242, "why": "왜"})

    assert blank.status_code == 400
    assert missing.status_code == 404


def test_forget_rejects_a_bool_or_float_id(tmp_path: Path) -> None:
    """`int(True)` is 1 and `int(3.9)` is 3 - either would coerce into retiring
    a rule the caller never named. `bool` is an `int` subclass, so the guard
    must check for it explicitly."""
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        as_bool = client.post("/admin/api/persona/forget", json={"id": True, "why": "왜"})
        as_float = client.post("/admin/api/persona/forget", json={"id": 3.9, "why": "왜"})
        as_string = client.post("/admin/api/persona/forget", json={"id": "1", "why": "왜"})

    assert as_bool.status_code == 400
    assert as_float.status_code == 400
    assert as_string.status_code == 400


def test_e_the_run_now_endpoints_take_the_shared_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both buttons must serialise against the cron and the boot task."""
    from daemon.admin import routes as routes_mod

    seen: list[object] = []

    async def fake_reflect(settings, lock):
        seen.append(lock)
        return []

    async def fake_evolve(settings, lock, *, force=False):
        seen.append(lock)

        class R:
            date, skipped = "2026-08-24", None
            observations_read = proposed = added = retired = 0
            problems: list[str] = []

        return R()

    monkeypatch.setattr(routes_mod, "run_reflection_now", fake_reflect)
    monkeypatch.setattr(routes_mod, "run_persona_evolution_now", fake_evolve)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        assert client.post("/admin/api/reflect", json={}).status_code == 200
        assert client.post("/admin/api/persona/evolve", json={}).status_code == 200
        assert seen == [app.state.catchup_lock, app.state.catchup_lock]


def test_reflect_reports_the_pass_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daemon.admin import routes as routes_mod

    class Result:
        date, status = "2026-08-24", "written"
        messages_read, facts, entities, observations = 41, 2, 1, 1
        problems: list[str] = []

    async def fake_reflect(settings, lock):
        return [Result()]

    monkeypatch.setattr(routes_mod, "run_reflection_now", fake_reflect)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/reflect", json={})

    assert r.json()["results"] == [
        {
            "date": "2026-08-24", "status": "written", "messages_read": 41,
            "facts": 2, "entities": 1, "observations": 1, "problems": [],
        }
    ]


def test_a_failed_pass_is_a_502_not_a_silent_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daemon.admin import routes as routes_mod

    async def boom(settings, lock):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(routes_mod, "run_reflection_now", boom)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/reflect", json={})

    assert r.status_code == 502
    assert "provider unreachable" in r.json()["detail"]


def test_forget_rejects_a_malformed_body(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/forget", content=b"not json")

    assert r.status_code == 400
    assert r.json()["detail"] == "body must be valid JSON"


def test_forget_rejects_a_non_object_body(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/forget", json=[1, 2, 3])

    assert r.status_code == 400
    assert r.json()["detail"] == "body must be an object"


def test_evolve_rejects_a_malformed_body(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/evolve", content=b"not json")

    assert r.status_code == 400
    assert r.json()["detail"] == "body must be valid JSON"


def test_evolve_rejects_a_non_object_body(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/evolve", json=[1, 2, 3])

    assert r.status_code == 400
    assert r.json()["detail"] == "body must be an object"


def test_forget_holds_the_catchup_lock_while_it_retires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`LearnedRules.retire` snapshots the active rows, then awaits a file read
    and a threaded write - and `diverged_bodies` only catches a file-side orphan
    (a bullet the mirror does not know), never the reverse. So a weekly `add()`
    that finishes inside that window would have its new bullets silently
    dropped by this rewrite, with nothing to detect it. This proves the retire
    call actually runs with `catchup_lock` held, the same lock
    reflect_now/persona_evolve take."""
    from daemon.admin import routes as routes_mod
    from daemon.app import DB_FILENAME

    store = Store.open(tmp_path / DB_FILENAME)
    rule = store.insert_persona_rule(
        body="규칙", created_at=_dt(9), evidence=[], supersession_key=None
    )
    store.close()
    _write(tmp_path / "persona" / "learned.md", "# learned\n\n- 규칙\n")

    app = create_app(_settings(tmp_path))
    seen_locked: list[bool] = []
    real_retire = routes_mod.LearnedRules.retire

    async def spying_retire(self, rule_id, *, why, now=None):
        # Records whether the lock is held *during* the call, then defers to
        # the real implementation so the request still succeeds normally.
        seen_locked.append(app.state.catchup_lock.locked())
        return await real_retire(self, rule_id, why=why, now=now)

    monkeypatch.setattr(routes_mod.LearnedRules, "retire", spying_retire)

    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/forget", json={"id": rule, "why": "이유"})

    assert r.status_code == 200, r.text
    assert seen_locked == [True]


def test_evolve_reports_a_failed_pass_as_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symmetric to `test_a_failed_pass_is_a_502_not_a_silent_ok` on the other
    'run now' route - a raising `run_persona_evolution_now` must not read as a
    silent success either."""
    from daemon.admin import routes as routes_mod

    async def boom(settings, lock, *, force=False):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(routes_mod, "run_persona_evolution_now", boom)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.post("/admin/api/persona/evolve", json={})

    assert r.status_code == 502
    assert "model unavailable" in r.json()["detail"]


def test_the_run_now_routes_serialise_every_field_of_the_real_dataclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other test in this file feeds a fake result object carrying only
    the fields the route happens to read - a field rename on the real `Result`
    (daemon.reflection) or `EvolutionResult` (daemon.persona.evolve) would pass
    every one of them and only surface at runtime. This constructs the actual
    dataclasses and pins the route's serialisation to them."""
    from daemon.admin import routes as routes_mod
    from daemon.persona.evolve import EvolutionResult
    from daemon.reflection import Result

    real_result = Result(
        date="2026-08-24", status="written", messages_read=41,
        facts=2, entities=1, observations=1, tool_facts=0,
        detail="", problems=["헤아릴 수 없음"],
    )
    real_evolution = EvolutionResult(
        date="2026-08-24", observations_read=9, proposed=3, added=2,
        retired=1, skipped="", problems=("증거 부족",),
    )

    async def fake_reflect(settings, lock):
        return [real_result]

    async def fake_evolve(settings, lock, *, force=False):
        return real_evolution

    monkeypatch.setattr(routes_mod, "run_reflection_now", fake_reflect)
    monkeypatch.setattr(routes_mod, "run_persona_evolution_now", fake_evolve)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        reflect = client.post("/admin/api/reflect", json={})
        evolve = client.post("/admin/api/persona/evolve", json={})

    assert reflect.json()["results"] == [
        {
            "date": "2026-08-24", "status": "written", "messages_read": 41,
            "facts": 2, "entities": 1, "observations": 1,
            "problems": ["헤아릴 수 없음"],
        }
    ]
    assert evolve.json() == {
        "date": "2026-08-24", "skipped": "", "observations_read": 9,
        "proposed": 3, "added": 2, "retired": 1, "problems": ["증거 부족"],
    }
