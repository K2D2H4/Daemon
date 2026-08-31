# 어드민 Memory · Persona 탭 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 어드민 콘솔에 `Memory` · `Persona` 탭을 추가해, 지금 `daemon persona`와 파일 직접 열기로만 볼 수 있는 큐레이션 사실·엔티티·성찰 원문·관찰·학습 규칙·진화 다이어리를 브라우저에서 읽고, 규칙 은퇴 / 성찰 지금 돌리기 / 진화 지금 돌리기 세 동작을 할 수 있게 한다.

**Architecture:** `daemon/admin/activity.py`와 같은 형태 — 새 모듈 `daemon/admin/mind.py`에 순수 페이로드 함수 두 개(모델 호출 없음, 쓰기 없음), 라우트는 `open_store`로 감싸 호출만. 마크다운 본문은 페이로드에 인라인해서 경로 파라미터가 파일시스템에 닿지 않게 한다. 쓰기 세 개는 이미 검증된 `LearnedRules.retire` / `Reflection.catch_up` / `PersonaEvolution.run`을 부르고, 뒤 두 개는 크론과 같은 `catchup_lock`을 잡는다.

**Tech Stack:** Python 3.13, FastAPI, sqlite3(STRICT), pytest / pytest-asyncio, 바닐라 JS (빌드 스텝 없음, CDN 금지).

**Spec:** [docs/design/2026-08-24-admin-mind-design.md](../../design/2026-08-24-admin-mind-design.md)

## Global Constraints

- **CONTRACTS 5**: `data/persona/seed.md`는 사람 소유. 코드가 쓰는 경로를 만들지 않는다. 이 계획에서 `seed.md`는 **읽기만**.
- **CONTRACTS 6**: `observations`는 append-only. UPDATE/DELETE 없음. `consumed_by`만 앞으로 움직인다. 이 계획의 관찰 관련 코드는 전부 SELECT.
- **CONTRACTS 10**: 새 라우트는 전부 기존 `router`(`daemon/admin/routes.py:154`, `dependencies=[Depends(_loopback_only)]`) 아래에 붙인다. 새 가드를 만들지 않는다.
- **레이어링**: `daemon/admin/*`는 구현체를 직접 import하지 않는다. `daemon/app.py`가 노출하는 것만 쓴다(`routes.py:64`가 이미 `from daemon.app import health_payload, open_store`).
- **마크다운→HTML 변환 금지.** 모든 텍스트는 `esc()`(`daemon/admin/static/index.html:537`)를 통과해 `<pre>`/텍스트로. 파서 추가 금지.
- **CDN 금지, 빌드 스텝 금지.** `index.html` 한 파일에 바닐라 JS로.
- **어드민 UI 문자열은 영어.** 제목·라벨·버튼은 영어(Silkscreen에 한글 글리프가 없다). **사용자 데이터(사실·규칙·관찰 본문)는 원문 그대로** — `--mono`(DM Mono) 폴백으로 렌더된다.
- **본문 인라인 상한**: 성찰 14일, 다이어리 8개, 엔티티 60개, 세 종류 합계 65536바이트. 목록은 언제나 전부. 초과 시 그 섹션에 `bodies_truncated: true`.
- 매 태스크 끝에서 `python3 -m pytest -q` 와 `python3 -m ruff check .` 둘 다 그린이어야 커밋한다.

---

### Task 1: `Store` 읽기 메서드 3개

기존 `active_entries` / `unconsumed_observations` / `active_persona_rules`는 **활성만** 준다. 폐기된 사실, 소비된 관찰, 은퇴한 규칙을 보여주려면 세 개가 더 필요하다.

**Files:**
- Modify: `daemon/memory/store.py` (`active_entries` 뒤, `count_entries` 앞 / `count_observations` 뒤 / `count_active_persona_rules` 뒤)
- Test: `tests/test_admin_mind.py` (신규)

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces:
  - `Store.recent_entries(self, limit: int = 100) -> list[sqlite3.Row]`
  - `Store.recent_observations(self, limit: int = 200) -> list[sqlite3.Row]`
  - `Store.retired_persona_rules(self, limit: int = 50) -> list[sqlite3.Row]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_admin_mind.py` 를 새로 만든다:

```python
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
    _fact(store, "새 사실", importance=6, key="k")

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
    gone = store.insert_persona_rule(
        body="은퇴할 규칙", created_at=_dt(10), evidence=[], supersession_key=None
    )
    assert store.retire_persona_rule(gone, when=_dt(24), why="사용자가 지우라고 했다")

    rows = store.retired_persona_rules(50)
    assert [int(r["id"]) for r in rows] == [gone]
    assert rows[0]["retired_why"] == "사용자가 지우라고 했다"
    assert rows[0]["retired_at"] == "2026-08-24T12:00:00Z"
    assert {int(r["id"]) for r in store.active_persona_rules()} == {kept}
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'recent_entries'` (세 테스트 모두)

먼저 `insert_entry` / `insert_observation` / `insert_persona_rule` 의 실제 시그니처를 확인하고 위 헬퍼를 맞춘다:

```bash
sed -n '820,860p' daemon/memory/store.py
sed -n '1009,1035p' daemon/memory/store.py
sed -n '1054,1074p' daemon/memory/store.py
```

키워드 이름이 다르면 **테스트를 실제 시그니처에 맞춘다**(구현을 테스트에 맞추지 않는다).

- [ ] **Step 3: 세 메서드를 구현한다**

`daemon/memory/store.py`, `active_entries` 바로 뒤:

```python
    def recent_entries(self, limit: int = 100) -> list[sqlite3.Row]:
        """Every curated fact, retired ones included, most important first.

        `active_entries` is what gets injected and deliberately hides the retired
        rows. This is the admin's read: a fact that was superseded is the evidence
        that supersession works, and dropping it makes a wrong fact look like it
        was never there.
        """
        return self.conn.execute(
            "SELECT * FROM memory_entries "
            "ORDER BY importance DESC, updated_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
```

`count_observations` 바로 뒤:

```python
    def recent_observations(self, limit: int = 200) -> list[sqlite3.Row]:
        """Every observation, newest first, with whichever rule consumed it.

        The opposite order from `unconsumed_observations`: that one feeds M4 and
        wants the oldest evidence first, this one is read top-down as "the most
        recent thing she noticed about me".
        """
        return self.conn.execute(
            "SELECT * FROM observations ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
```

`count_active_persona_rules` 바로 뒤:

```python
    def retired_persona_rules(self, limit: int = 50) -> list[sqlite3.Row]:
        """Retired rules, most recently retired first - with `retired_why`.

        `learned.md` is rewritten whole on every change, so a rule that vanished
        leaves no trace in the file. This is the only place "what did she stop
        believing, and why" can be read.
        """
        return self.conn.execute(
            "SELECT * FROM persona_rules WHERE status = 'retired' "
            "ORDER BY retired_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q && python3 -m ruff check .`
Expected: 3 passed, ruff clean

- [ ] **Step 5: 커밋**

```bash
git add daemon/memory/store.py tests/test_admin_mind.py
git commit -m "store: three reads the admin needs - retired facts, consumed observations, retired rules"
```

---

### Task 2: `mind.memory_payload`

**Files:**
- Create: `daemon/admin/mind.py`
- Test: `tests/test_admin_mind.py` (추가)

**Interfaces:**
- Consumes: `Store.recent_entries`, `Store.entities`, `Store.links_for`, `Store.recent_reflection_runs` (Task 1 + 기존)
- Produces:
  - `daemon.admin.mind.memory_payload(store: Store, data_dir: Path) -> dict[str, Any]`
  - 상수 `MAX_REFLECTION_BODIES = 14`, `MAX_DIARY_BODIES = 8`, `MAX_ENTITY_BODIES = 60`, `MAX_BODY_BYTES = 65536`
  - 내부 헬퍼 `_read_bodies(paths: list[tuple[str, Path]], *, limit: int, budget: BodyBudget) -> tuple[dict[str, str], bool]`
  - `class BodyBudget` — 세 섹션이 하나의 64KB 예산을 나눠 쓴다

페이로드 모양 (뒤 태스크가 이 키를 그대로 읽는다):

```python
{
  "facts": [{"id": 2, "body": "...", "importance": 9, "triggers": ["개발자"],
             "key": "user_job", "status": "active", "updated_at": "2026-08-19T…Z"}],
  "facts_active": 11, "facts_retired": 1,
  "entities": [{"name": "김대현", "kind": "person", "mentions": 5,
                "links": ["UJET.cx"], "body": "...", "file": "memory/entities/김대현.md"}],
  "entities_total": 11, "entities_bodies_truncated": False,
  "reflections": [{"date": "2026-08-19", "status": "written", "ts": "…Z",
                   "messages_read": 72, "facts": 1, "entities": 2, "observations": 2,
                   "body": "...", "file": "memory/reflections/2026-08-19.md"}],
  "reflections_total": 9, "reflections_recorded": 5,
  "reflections_bodies_truncated": False,
  "pending_days": ["2026-08-24"],
}
```

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_admin_mind.py` 에 추가 (import에 `from daemon.admin.mind import memory_payload` 와 `import daemon.admin.mind as mind` 추가):

```python
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
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemon.admin.mind'`

`record_reflection_run` / `upsert_entity` / `set_mention_count` 의 실제 시그니처를 확인해 테스트를 맞춘다:

```bash
grep -n "def record_reflection_run" -A 14 daemon/memory/store.py
grep -n "def recent_reflection_runs" -A 8 daemon/memory/store.py
```

- [ ] **Step 3: `daemon/admin/mind.py` 를 만든다**

```python
"""What the daemon knows and what it worked out - the Memory and Persona tabs.

Two read-only payloads over five tables (`memory_entries`, `entities`,
`entity_links`, `observations`, `persona_rules`, `reflection_runs`) and the
markdown those tables mirror. Same rules as `daemon/admin/activity.py`: no
writes, no model calls, no side effects.

## Bodies are inlined, and that is a security decision

The markdown is returned *inside* the payload rather than through a
`/api/memory/reflection/{date}` endpoint. Measured on the real install the whole
corpus is 11.6 KB (entities 1.7, reflections 7.7, diaries 2.2), so the saving from
lazy-loading would be nothing - and the endpoint that would do it takes a date or
an entity name from the URL and joins it onto a path. There is no such endpoint
here, so there is no traversal bug to get wrong.

It does grow: roughly 800 bytes a day of reflections. `MAX_BODY_BYTES` and the
three count caps bound it, and they bound **bodies only** - the list is always
complete. A capped list would silently shorten history, which is the failure this
whole tab exists to fix.

## The reflection list is keyed on the files, not on `reflection_runs`

`reflection_runs` arrived with M5; the artifacts predate it. Measured on the real
install: 5 rows against 9 artifacts. Keyed on the table, four days of history
disappear from the only screen that shows it. So the day is the file, and the row
is the extra detail a day may or may not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daemon.config import Settings
from daemon.memory.store import Store

MAX_FACTS = 200
MAX_ENTITIES = 500
MAX_OBSERVATIONS = 300
MAX_RULES = 100
MAX_REFLECTION_ROWS = 400
"""List ceilings, so a hand-edited limit cannot make the admin read an unbounded
table into memory. Far above the real corpus (12 / 11 / 9 / 3) on purpose: these
are a backstop, not the body caps below."""

MAX_REFLECTION_BODIES = 14
MAX_DIARY_BODIES = 8
MAX_ENTITY_BODIES = 60
MAX_BODY_BYTES = 64 * 1024
"""Body caps. Read the module docstring first: these drop *bodies*, never list
entries, and whichever section they bite reports `bodies_truncated`."""


class BodyBudget:
    """One byte budget shared by every section of one payload.

    Per-section budgets would let a long entity corpus and a long reflection
    corpus each stay under its own ceiling while the response is twice the size
    anyone signed off on.
    """

    def __init__(self, total: int) -> None:
        self.left = total

    def take(self, text: str) -> str | None:
        cost = len(text.encode("utf-8"))
        if cost > self.left:
            return None
        self.left -= cost
        return text


def _read(path: Path) -> str | None:
    """A note's text, or None when the file is not there.

    None rather than an exception: the real install has an `entities` row
    ('벨라') whose note file is absent, and a missing note must render as a
    missing note rather than 500 the whole tab.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _bodies(
    items: list[tuple[str, Path]], *, limit: int, budget: BodyBudget
) -> tuple[dict[str, str], bool]:
    """Bodies for the first `limit` items that fit the budget, plus whether
    anything was left without one. `items` is already in display order."""
    bodies: dict[str, str] = {}
    truncated = False
    for index, (key, path) in enumerate(items):
        if index >= limit:
            truncated = True
            continue
        text = _read(path)
        if text is None:
            continue
        kept = budget.take(text)
        if kept is None:
            truncated = True
            continue
        bodies[key] = kept
    return bodies, truncated


def _triggers(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _rel(data_dir: Path, path: Path) -> str:
    """The path as the markdown contract names it, for the 'no body' case. Relative
    so the response never carries the owner's home directory."""
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.name


# --- Memory -----------------------------------------------------------------


def memory_payload(store: Store, data_dir: Path) -> dict[str, Any]:
    """Curated facts, entity notes and the reflection history."""
    from daemon import reflection as reflection_mod

    budget = BodyBudget(MAX_BODY_BYTES)

    fact_rows = store.recent_entries(MAX_FACTS)
    facts = [
        {
            "id": int(row["id"]),
            "body": row["body"],
            "importance": int(row["importance"]),
            "triggers": _triggers(row["trigger_phrases"]),
            "key": row["supersession_key"],
            "status": row["status"],
            "updated_at": row["updated_at"],
        }
        for row in fact_rows
    ]

    entity_rows = store.entities(MAX_ENTITIES)
    entity_paths = [
        (row["name"], data_dir / row["file"]) for row in entity_rows
    ]
    entity_bodies, entities_cut = _bodies(
        entity_paths, limit=MAX_ENTITY_BODIES, budget=budget
    )
    entities = [
        {
            "name": row["name"],
            "kind": row["kind"],
            "mentions": int(row["mention_count"]),
            "links": [link["name"] for link in store.links_for(int(row["id"]))],
            "body": entity_bodies.get(row["name"]),
            "file": row["file"],
        }
        for row in entity_rows
    ]

    # The files are the axis - see the module docstring.
    reflect_dir = data_dir / reflection_mod.REFLECTIONS_SUBDIR
    days = sorted(
        (path.stem for path in reflect_dir.glob("*.md")) if reflect_dir.exists() else (),
        reverse=True,
    )
    # `recent_reflection_runs` is id DESC (store.py:1513); reversed makes it id
    # ASC so a day with two passes keeps the *newest* row, not the first.
    runs = {
        row["date"]: row
        for row in reversed(store.recent_reflection_runs(MAX_REFLECTION_ROWS))
    }
    reflect_paths = [
        (day, reflection_mod.artifact_path(data_dir, day)) for day in days
    ]
    reflect_bodies, reflect_cut = _bodies(
        reflect_paths, limit=MAX_REFLECTION_BODIES, budget=budget
    )
    reflections = []
    for day, path in reflect_paths:
        row = runs.get(day)
        reflections.append(
            {
                "date": day,
                "status": row["status"] if row is not None else None,
                "ts": row["ts"] if row is not None else None,
                "messages_read": int(row["messages_read"]) if row is not None else None,
                "facts": int(row["facts"]) if row is not None else None,
                "entities": int(row["entities"]) if row is not None else None,
                "observations": int(row["observations"]) if row is not None else None,
                "detail": row["detail"] if row is not None else "",
                "body": reflect_bodies.get(day),
                "file": _rel(data_dir, path),
            }
        )

    return {
        "facts": facts,
        "facts_active": sum(1 for f in facts if f["status"] == "active"),
        "facts_retired": sum(1 for f in facts if f["status"] == "retired"),
        "entities": entities,
        "entities_total": len(entities),
        "entities_bodies_truncated": entities_cut,
        "reflections": reflections,
        "reflections_total": len(reflections),
        "reflections_recorded": sum(
            1 for r in reflections if r["status"] is not None
        ),
        "reflections_bodies_truncated": reflect_cut,
        "pending_days": reflection_mod.pending_days(data_dir),
    }
```

`from daemon import reflection as reflection_mod` 이 함수 안에 있는 이유: `daemon/reflection.py`는 임포트 시 `numpy`를 끌고 오는 `daemon.memory.store`를 넘어 LLM 게이트웨이까지 닿는다. 모듈 최상단에 두면 어드민 임포트가 무거워진다. 최상단으로 올려도 테스트가 통과하면 올려도 된다 — 그때는 이 문단을 지운다.

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q && python3 -m ruff check .`
Expected: 9 passed, ruff clean

- [ ] **Step 5: 커밋**

```bash
git add daemon/admin/mind.py tests/test_admin_mind.py
git commit -m "admin: the memory payload - facts, entity notes, and the reflection history keyed on its files"
```

---

### Task 3: `mind.persona_payload`

**Files:**
- Modify: `daemon/admin/mind.py`
- Test: `tests/test_admin_mind.py` (추가)

**Interfaces:**
- Consumes: Task 2의 `BodyBudget`, `_bodies`, `_read`, `_rel`; Task 1의 `Store.recent_observations`, `Store.retired_persona_rules`
- Produces: `daemon.admin.mind.persona_payload(store: Store, data_dir: Path, settings: Settings) -> dict[str, Any]`

```python
{
  "anchor": {"active": 3, "max_active": 20, "max_new_per_cycle": 3,
             "min_observations": 5, "unconsumed": 0,
             "last_rule_at": "2026-08-23T20:00:00Z",
             "seed": {"text": "...", "lines": 26, "file": "persona/seed.md"},
             "learned": {"text": "...", "lines": 3, "file": "persona/learned.md"}},
  "rules": [{"id": 1, "body": "...", "created_at": "…Z", "status": "active",
             "retired_at": None, "retired_why": None,
             "evidence": [{"id": 1, "body": "...", "confidence": 0.85}]}],
  "rules_active": 3, "rules_retired": 0,
  "observations": [{"id": 9, "body": "...", "confidence": 0.8,
                    "consumed_by": 2, "observed_from": "…", "created_at": "…Z"}],
  "observations_total": 9, "observations_consumed": 9,
  "diaries": [{"date": "2026-08-24", "body": "...", "file": "persona/diary/2026-08-24.md"}],
  "diaries_total": 2, "diaries_bodies_truncated": False,
}
```

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
from daemon.admin.mind import memory_payload, persona_payload
from daemon.config import Settings


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

    payload = persona_payload(store, tmp_path, _settings(tmp_path))
    anchor = payload["anchor"]

    assert anchor["active"] == 1
    assert anchor["max_active"] == 20            # Settings default
    assert anchor["max_new_per_cycle"] == 3
    assert anchor["min_observations"] == 5
    assert anchor["unconsumed"] == 1
    assert anchor["last_rule_at"] == "2026-08-09T12:00:00Z"
    assert anchor["seed"]["lines"] == 3
    assert anchor["seed"]["file"] == "persona/seed.md"
    assert "너는 벨라다" in anchor["seed"]["text"]
    assert anchor["learned"]["lines"] == 3


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
    assert payload["anchor"]["seed"]["lines"] == 0
    assert payload["anchor"]["learned"]["text"] is None
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q`
Expected: FAIL — `ImportError: cannot import name 'persona_payload'`

- [ ] **Step 3: `persona_payload` 를 구현한다**

`daemon/admin/mind.py` 끝에 추가:

```python
# --- Persona ----------------------------------------------------------------


def _file_view(data_dir: Path, rel: str, budget: BodyBudget) -> dict[str, Any]:
    text = _read(data_dir / rel)
    kept = None if text is None else budget.take(text)
    return {
        "text": kept,
        "lines": 0 if kept is None else len(kept.splitlines()),
        "file": rel,
    }


def _evidence(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed if isinstance(item, int)]


def persona_payload(store: Store, data_dir: Path, settings: Settings) -> dict[str, Any]:
    """The anchor readout, the learned rules with their evidence, every
    observation, and the evolution diaries.

    The anchor leads with numbers rather than with the two files because the
    anchor's claim is not "seed.md is untouched", it is "change is slow"
    (docs/PLAN.md 5.1) - and that is only legible as a rate. The files follow
    because the ownership claim needs the evidence to be actually there.
    """
    from daemon.persona.evolve import DIARY_SUBDIR

    budget = BodyBudget(MAX_BODY_BYTES)

    observation_rows = store.recent_observations(MAX_OBSERVATIONS)
    observations = [
        {
            "id": int(row["id"]),
            "body": row["body"],
            "confidence": float(row["confidence"]),
            "consumed_by": None if row["consumed_by"] is None else int(row["consumed_by"]),
            "observed_from": row["observed_from"],
            "created_at": row["created_at"],
        }
        for row in observation_rows
    ]
    by_id = {obs["id"]: obs for obs in observations}

    def one_rule(row: Any, status: str) -> dict[str, Any]:
        evidence = [
            {
                "id": by_id[oid]["id"],
                "body": by_id[oid]["body"],
                "confidence": by_id[oid]["confidence"],
            }
            # A stale id is skipped, not raised on: `evidence` is model-supplied
            # json and an observation it names may predate a rebuild.
            for oid in _evidence(row["evidence"])
            if oid in by_id
        ]
        return {
            "id": int(row["id"]),
            "body": row["body"],
            "created_at": row["created_at"],
            "status": status,
            "retired_at": row["retired_at"],
            "retired_why": row["retired_why"],
            "evidence": evidence,
        }

    active = [one_rule(row, "active") for row in store.active_persona_rules()]
    retired = [one_rule(row, "retired") for row in store.retired_persona_rules(MAX_RULES)]

    diary_dir = data_dir / DIARY_SUBDIR
    diary_days = sorted(
        (path.stem for path in diary_dir.glob("*.md")) if diary_dir.exists() else (),
        reverse=True,
    )
    diary_paths = [(day, diary_dir / f"{day}.md") for day in diary_days]
    diary_bodies, diaries_cut = _bodies(
        diary_paths, limit=MAX_DIARY_BODIES, budget=budget
    )
    diaries = [
        {
            "date": day,
            "body": diary_bodies.get(day),
            "file": _rel(data_dir, path),
        }
        for day, path in diary_paths
    ]

    return {
        "anchor": {
            "active": store.count_active_persona_rules(),
            "max_active": settings.persona_max_active_rules,
            "max_new_per_cycle": settings.persona_max_new_per_cycle,
            "min_observations": settings.persona_min_observations,
            "unconsumed": sum(1 for obs in observations if obs["consumed_by"] is None),
            "last_rule_at": store.last_persona_rule_created_at(),
            # Read, never written - docs/CONTRACTS.md non-negotiable 5.
            "seed": _file_view(data_dir, "persona/seed.md", budget),
            "learned": _file_view(data_dir, "persona/learned.md", budget),
        },
        "rules": active + retired,
        "rules_active": len(active),
        "rules_retired": len(retired),
        "observations": observations,
        "observations_total": len(observations),
        "observations_consumed": sum(
            1 for obs in observations if obs["consumed_by"] is not None
        ),
        "diaries": diaries,
        "diaries_total": len(diaries),
        "diaries_bodies_truncated": diaries_cut,
    }
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q && python3 -m ruff check .`
Expected: 15 passed, ruff clean

- [ ] **Step 5: 커밋**

```bash
git add daemon/admin/mind.py tests/test_admin_mind.py
git commit -m "admin: the persona payload - the anchor as a rate, and every rule with its evidence"
```

---

### Task 4: 읽기 엔드포인트 두 개

**Files:**
- Modify: `daemon/admin/routes.py` (`tools_log` 뒤, `# --- MCP (Phase 2)` 주석 앞)
- Test: `tests/test_admin_mind.py` (추가)

**Interfaces:**
- Consumes: `mind.memory_payload`, `mind.persona_payload`
- Produces: `GET /admin/api/memory`, `GET /admin/api/persona`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
from fastapi.testclient import TestClient

from daemon.app import create_app

LOOPBACK = "http://127.0.0.1"


def test_the_read_endpoints_serve_the_payloads_over_loopback(tmp_path: Path) -> None:
    _write(tmp_path / "memory" / "reflections" / "2026-08-19.md", "# 성찰\n")
    _write(tmp_path / "persona" / "seed.md", "# seed\n")
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        memory = client.get("/admin/api/memory")
        persona = client.get("/admin/api/persona")

    assert memory.status_code == 200
    assert [r["date"] for r in memory.json()["reflections"]] == ["2026-08-19"]
    assert persona.status_code == 200
    assert persona.json()["anchor"]["max_active"] == 20


def test_f_no_route_writes_the_seed(tmp_path: Path) -> None:
    """CONTRACTS non-negotiable 5. Asserted on the router, not on a code review:
    a later hand could add a PATCH and every other test would stay green."""
    from daemon.admin import routes

    seed_routes = [
        (route.path, sorted(route.methods))
        for route in routes.router.routes
        if "seed" in route.path
    ]
    assert seed_routes == []

    persona_writes = [
        (route.path, sorted(route.methods))
        for route in routes.router.routes
        if route.path.startswith("/admin/api/persona")
        and set(route.methods) - {"GET", "HEAD"}
    ]
    # Only `forget` and `evolve` write anything under /persona, and neither
    # touches seed.md - see Task 6.
    assert {path for path, _ in persona_writes} <= {
        "/admin/api/persona/forget", "/admin/api/persona/evolve"
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -k endpoints -q`
Expected: FAIL — 404

- [ ] **Step 3: 라우트를 붙인다**

`daemon/admin/routes.py`, import 블록에 추가:

```python
from daemon.admin.mind import memory_payload, persona_payload
```

`tools_log` 함수 뒤:

```python
# --- Memory and Persona: what she knows, and what she worked out -------------
# The read half. Not on the 15-second poll the browser runs for health/activity:
# these change once a day and carry ~12 KB of markdown, so `index.html` loads them
# on tab entry, on an explicit refresh, and after a write.


@router.get("/api/memory")
async def memory(request: Request) -> JSONResponse:
    """Curated facts, entity notes, and the reflection history with its artifacts."""
    settings = request.app.state.settings
    with open_store(settings) as store:
        payload = memory_payload(store, settings.data_dir)
    return JSONResponse(payload)


@router.get("/api/persona")
async def persona(request: Request) -> JSONResponse:
    """The anchor readout, learned rules with their evidence, observations, diaries."""
    settings = request.app.state.settings
    with open_store(settings) as store:
        payload = persona_payload(store, settings.data_dir, settings)
    return JSONResponse(payload)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q && python3 -m ruff check .`
Expected: 17 passed, ruff clean

- [ ] **Step 5: 커밋**

```bash
git add daemon/admin/routes.py tests/test_admin_mind.py
git commit -m "admin: two read endpoints for what she knows and what she learned"
```

---

### Task 5: `app.state.catchup_lock` + 결과를 반환하는 헬퍼 둘

스펙의 「발견」 섹션. `catchup_lock`은 `_start` 안의 지역 변수여서 라우트가 잡을 수 없고, 락 없이 `Reflect now`를 누르면 04:00 크론과 겹쳐 append-only 성찰 아티팩트를 이중 기록하고 관찰을 중복 삽입한다(`daemon/app.py:232-237` 주석). 그리고 `_reflect_tick` / `_persona_tick`은 예외를 삼키고 `None`을 반환해 브라우저에 결과를 줄 수 없다.

**Files:**
- Modify: `daemon/app.py` (`catchup_lock = asyncio.Lock()` 직후 / `_reflect_tick` / `_persona_tick`)
- Test: `tests/test_admin_mind.py` (추가)

**Interfaces:**
- Consumes: 기존 `build_reflection`, `build_persona_evolution`
- Produces:
  - `daemon.app.run_reflection_now(settings: Settings, lock: asyncio.Lock | None) -> list[Result]`
  - `daemon.app.run_persona_evolution_now(settings: Settings, lock: asyncio.Lock | None, *, force: bool = False) -> EvolutionResult`
  - `app.state.catchup_lock: asyncio.Lock`
  - 둘 다 실패 시 예외를 **올린다** (기존 틱과 정반대). 틱은 그 예외를 잡아 로그로 찍는 래퍼가 된다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
import asyncio


def test_e_the_app_exposes_the_catchup_lock(tmp_path: Path) -> None:
    """Without this the two run-now endpoints cannot take the lock the crons take,
    and a click during the 04:00 cron double-writes the append-only artifact
    (daemon/app.py:232-237)."""
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK):
        assert isinstance(app.state.catchup_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_run_reflection_now_raises_where_the_tick_swallows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tick logs and returns None so APScheduler carries on. The endpoint
    needs the failure, or the browser reports success for a pass that never ran."""
    from daemon import app as app_mod

    async def boom(settings):
        raise RuntimeError("no provider")

    monkeypatch.setattr(app_mod, "build_reflection", boom)

    with pytest.raises(RuntimeError, match="no provider"):
        await app_mod.run_reflection_now(_settings(tmp_path), None)

    # The tick still swallows it - that contract does not change.
    await app_mod._reflect_tick(_settings(tmp_path), None)


@pytest.mark.asyncio
async def test_run_reflection_now_holds_the_lock_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daemon import app as app_mod

    held = asyncio.Event()
    release = asyncio.Event()

    class SlowReflection:
        async def catch_up(self):
            held.set()
            await release.wait()
            return []

    async def build(settings):
        async def close() -> None:
            return None

        return SlowReflection(), close

    monkeypatch.setattr(app_mod, "build_reflection", build)
    lock = asyncio.Lock()
    task = asyncio.create_task(app_mod.run_reflection_now(_settings(tmp_path), lock))
    await held.wait()

    assert lock.locked()
    release.set()
    assert await task == []
    assert not lock.locked()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -k "catchup or run_reflection" -q`
Expected: FAIL — `AttributeError: 'State' object has no attribute 'catchup_lock'`, `module 'daemon.app' has no attribute 'run_reflection_now'`

- [ ] **Step 3: 한 줄 + 헬퍼 둘을 넣고 틱을 래퍼로 만든다**

`daemon/app.py`, `catchup_lock = asyncio.Lock()` (line 238) 바로 뒤:

```python
    # Exposed so the admin's "run now" buttons take the very same lock. Without
    # this they would be a third writer beside the cron and the boot task, and the
    # comment above says what two `run(date)` for one day costs.
    app.state.catchup_lock = catchup_lock
```

`_reflect_tick` 을 둘로 쪼갠다 — 기존 본문에서 락·빌드·실행만 떼어내 새 함수로:

```python
async def run_reflection_now(
    settings: Settings, lock: asyncio.Lock | None
) -> list[Result]:
    """Reflect on every unreflected day, and *raise* if it could not.

    The opposite contract from `_reflect_tick`, on purpose. A scheduled job that
    raises inside APScheduler stops being scheduled, so the tick swallows. A
    button press has a person waiting for the answer, and swallowing there would
    report success for a pass that never reached the model.

    `lock` is `app.state.catchup_lock`: the same one the cron and the boot task
    take, because this is a third writer of the same append-only artifact.
    """
    reflection_pass, close = await build_reflection(settings)
    try:
        async with lock if lock is not None else nullcontext():
            return await reflection_pass.catch_up()
    finally:
        with suppress(Exception):
            await close()


async def run_persona_evolution_now(
    settings: Settings, lock: asyncio.Lock | None, *, force: bool = False
) -> EvolutionResult:
    """Run the weekly pass now, and raise if it could not. Same split as
    `run_reflection_now`, same reason."""
    evolution, close = await build_persona_evolution(settings)
    try:
        async with lock if lock is not None else nullcontext():
            return await evolution.run(force=force)
    finally:
        with suppress(Exception):
            await close()
```

그리고 두 틱은 이 헬퍼를 감싸는 것으로 줄인다 — 로그 문구는 **한 글자도 바꾸지 않는다**(`tests/test_cli.py` 등이 읽을 수 있다):

```python
async def _reflect_tick(settings: Settings, lock: asyncio.Lock | None = None) -> None:
    """The scheduled pass. Catches everything: a job that raises inside
    APScheduler is logged once and then the schedule carries on, which reads as a
    working reflection loop that has silently done nothing for a month.

    The work itself is `run_reflection_now`, which raises - the admin's button
    needs the failure. This wrapper is the swallowing half.
    """
    try:
        results = await run_reflection_now(settings, lock)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("reflection tick failed: %s", exc)
        return
    for result in results:
        logger.info(
            "reflection %s: %s (%d message(s) -> %d fact(s), %d entity(ies), %d observation(s))%s",
            result.date,
            result.status,
            result.messages_read,
            result.facts,
            result.entities,
            result.observations,
            f" problems={result.problems}" if result.problems else "",
        )


async def _persona_tick(settings: Settings, lock: asyncio.Lock | None = None) -> None:
    """The weekly persona-evolution pass. Catches everything, same reason as
    reflection and the proactive tick.

    Logged at INFO even when the pass was skipped, because "not enough
    observations yet" and "already ran this week" both have to be visible
    without opening sqlite.
    """
    try:
        result = await run_persona_evolution_now(settings, lock)
    except Exception as exc:  # noqa: BLE001 - the tick must survive a bad config
        logger.error("persona evolution tick failed: %s", exc)
        return

    logger.info(
        "persona evolve %s: %s (%d observation(s) read -> %d proposed, %d added, "
        "%d retired)%s",
        result.date,
        result.skipped or "ran",
        result.observations_read,
        result.proposed,
        result.added,
        result.retired,
        f" problems={result.problems}" if result.problems else "",
    )
```

**로그 문구가 하나 줄어든다 — 의도된 것이다.** 기존 틱은 빌드 실패와 실행 실패를 다른 문구로 로그했다(`"... could not start: %s"` 대 `"... failed: %s"`). 위 래퍼는 둘을 `"... failed: %s"` 하나로 합친다. 이 계획을 쓰기 전에 확인했다:

```
$ grep -rn "could not start" tests/
tests/test_setup.py:2599    # (주석)
tests/test_presence.py:268,283,291,568    # lsappinfo, 무관
```

성찰·페르소나 틱의 그 문구를 읽는 테스트는 **없다**. 구분을 유지하려면 build 단계를 틱에 남기고 헬퍼를 별도로 두어야 하는데 그건 lock·close 처리 10여 줄의 중복이고, 합친 문구도 `%s`에 예외 본문이 실려 설정 오류와 패스 실패가 실제로는 구분된다. 그 트레이드로 합친다 — **로그 문구를 바꾸는 것이므로 커밋 메시지에 적는다.**

**임포트 두 개를 고쳐야 한다.** 확인 결과 `daemon/app.py`는 `from daemon.reflection import Reflection`(:46)과 `from daemon.persona.evolve import PersonaEvolution`(:44)만 갖고 있고, 위 시그니처가 쓰는 `Result` / `EvolutionResult`는 없다:

```python
from daemon.persona.evolve import EvolutionResult, PersonaEvolution   # :44
from daemon.reflection import Reflection, Result                      # :46
```

그리고 헬퍼 시그니처의 `list[reflection.Result]` 를 `list[Result]` 로 쓴다(모듈 별칭을 새로 만들지 않는다).

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py tests/test_cli.py tests/test_admin.py -q && python3 -m ruff check .`
Expected: all passed, ruff clean

전체도 돌린다 — `app.py`는 모두가 임포트한다:

Run: `python3 -m pytest -q`
Expected: 실패 0

- [ ] **Step 5: 커밋**

```bash
git add daemon/app.py tests/test_admin_mind.py
git commit -m "app: the catch-up lock the crons share is reachable, and a run that raises for a caller who is waiting

The two tick log lines lose their build/run distinction: 'could not start' and
'failed' merge into 'failed', whose %s still carries which it was. No test read
the dropped string (checked: only test_setup/test_presence match, both unrelated)."
```

---

### Task 6: 쓰기 엔드포인트 세 개

**Files:**
- Modify: `daemon/admin/routes.py`
- Test: `tests/test_admin_mind.py` (추가)

**Interfaces:**
- Consumes: Task 5의 `run_reflection_now` / `run_persona_evolution_now` / `app.state.catchup_lock`; 기존 `LearnedRules`, `LearnedFileDiverged`
- Produces:
  - `POST /admin/api/persona/forget` — body `{"id": int, "why": str}` → `{"retired": true, "id": 2}`
  - `POST /admin/api/reflect` — body `{}` → `{"results": [{"date","status","messages_read","facts","entities","observations","problems"}]}`
  - `POST /admin/api/persona/evolve` — body `{"force": bool}` → `{"date","skipped","observations_read","proposed","added","retired","problems"}`
  - 오류: 빈 `why` → 400 / 없는 id → 404 / `LearnedFileDiverged` → 409 with `detail` = 예외 문구

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
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

    assert r.status_code == 409
    assert "손으로 적은 줄" in r.json()["detail"]
    # And nothing was written: the rule is still active and the file untouched.
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


@pytest.mark.asyncio
async def test_e_the_run_now_endpoints_take_the_shared_lock(
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
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -k "forget or run_now or reflect or failed_pass" -q`
Expected: FAIL — 404

- [ ] **Step 3: 라우트 세 개를 붙인다**

`daemon/admin/routes.py` import에 추가:

```python
from daemon.app import (
    health_payload,
    open_store,
    run_persona_evolution_now,
    run_reflection_now,
)
from daemon.persona.rules import LearnedFileDiverged, LearnedRules
```

읽기 라우트 뒤:

```python
# The write half: three handles, each calling a function the CLI already calls.
# Nothing here adds a rule or edits one - `persona/learned.md` is AI-owned
# (docs/CONTRACTS.md non-negotiable 5) and retiring is the one thing a human was
# ever able to ask for (`daemon persona forget`). No route writes `seed.md`.


@router.post("/api/persona/forget")
async def persona_forget(request: Request) -> JSONResponse:
    """Retire one learned rule. The browser's `daemon persona forget`."""
    body = await request.json()
    try:
        rule_id = int(body.get("id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="id must be a rule id") from None
    why = str(body.get("why") or "").strip()
    if not why:
        # Required for the same reason the CLI requires it: a rule that vanished
        # with no reason is indistinguishable from one that vanished by accident.
        raise HTTPException(status_code=400, detail="why is required")

    settings = request.app.state.settings
    with open_store(settings) as store:
        try:
            retired = await LearnedRules(settings.data_dir, store).retire(rule_id, why=why)
        except LearnedFileDiverged as diverged:
            # 409, and the exception's own words. This is a refusal with a fix -
            # the file holds bullets the mirror does not know, and rewriting it
            # would drop them (daemon/cli.py:1237-1239). A generic failure here
            # reads as a broken button.
            raise HTTPException(status_code=409, detail=str(diverged)) from None
    if not retired:
        raise HTTPException(status_code=404, detail=f"no active rule with id {rule_id}")
    return JSONResponse({"retired": True, "id": rule_id})


@router.post("/api/reflect")
async def reflect_now(request: Request) -> JSONResponse:
    """Run the reflection pass over every unreflected day, now.

    Takes `app.state.catchup_lock` - the same lock the 04:00 cron and the boot
    catch-up take. Two `run(date)` for one day double-write its append-only
    artifact and its observations (daemon/app.py:232-237).
    """
    settings = request.app.state.settings
    try:
        results = await run_reflection_now(settings, request.app.state.catchup_lock)
    except Exception as exc:  # noqa: BLE001 - a person is waiting for the answer
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return JSONResponse(
        {
            "results": [
                {
                    "date": r.date,
                    "status": r.status,
                    "messages_read": r.messages_read,
                    "facts": r.facts,
                    "entities": r.entities,
                    "observations": r.observations,
                    "problems": list(r.problems),
                }
                for r in results
            ]
        }
    )


@router.post("/api/persona/evolve")
async def persona_evolve(request: Request) -> JSONResponse:
    """Run the weekly persona pass now - the only way to see it without waiting
    for Monday 05:00. Same lock, same reason as `reflect_now`."""
    body = await request.json()
    force = bool(body.get("force"))
    settings = request.app.state.settings
    try:
        result = await run_persona_evolution_now(
            settings, request.app.state.catchup_lock, force=force
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return JSONResponse(
        {
            "date": result.date,
            "skipped": result.skipped,
            "observations_read": result.observations_read,
            "proposed": result.proposed,
            "added": result.added,
            "retired": result.retired,
            "problems": list(result.problems),
        }
    )
```

`HTTPException`의 `detail`이 `api()`의 `b.detail`로 그대로 오는지 확인한다 (`index.html:536`) — FastAPI는 `{"detail": ...}`로 직렬화하므로 맞다.

- [ ] **Step 4: 통과를 확인한다**

Run: `python3 -m pytest tests/test_admin_mind.py -q && python3 -m ruff check .`
Expected: 24 passed, ruff clean

- [ ] **Step 5: 커밋**

```bash
git add daemon/admin/routes.py tests/test_admin_mind.py
git commit -m "admin: three handles - forget a rule, reflect now, evolve now, all under the shared lock"
```

---

### Task 7: `index.html` — nav 2개 + section 2개 + 렌더

**Files:**
- Modify: `daemon/admin/static/index.html`

**Interfaces:**
- Consumes: `GET /admin/api/memory`, `GET /admin/api/persona`, 세 POST; 기존 `api()`(:536), `esc()`(:537), `stamp()`, `NAV`(:555-559), `go()`(:563)
- Produces: nav 항목 `memory` / `persona`, section `#view-memory` / `#view-persona`, 함수 `loadMemory()` / `loadPersona()`

- [ ] **Step 1: nav 항목과 NAV 표에 등록한다**

`daemon/admin/static/index.html:400` (`mcp` nav 뒤), Chat test 앞:

```html
    <button class="nav-item" data-nav="memory">Memory<span class="count" id="c-memory"></span></button>
    <button class="nav-item" data-nav="persona">Persona<span class="count" id="c-persona"></span></button>
```

`NAV` 표(:559)를 고친다:

```javascript
  mcp:{view:'mcp'},memory:{view:'memory'},persona:{view:'persona'},
  chat:{view:'chat'},settings:{view:'settings'}};
```

`go()`(:571) `mcp` 줄 뒤:

```javascript
  if(spec.view==='memory')loadMemory();
  if(spec.view==='persona')loadPersona();
```

**폴링에 넣지 않는다.** `refresh()`(:1431)는 건드리지 않는다.

- [ ] **Step 2: 두 섹션 마크업을 넣는다**

`<section id="view-mcp">` 닫는 태그 뒤, `<section id="view-chat">` 앞:

```html
  <section id="view-memory">
    <div><span class="lab">MEMORY</span>
      <p class="cap" id="memory-summary" style="margin:8px 0 0">—</p></div>
    <div class="card">
      <div class="card-head"><span class="card-title">FACTS</span>
        <span class="right cap" id="fact-count"></span></div>
      <div id="fact-list"><p class="empty">Loading…</p></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">ENTITIES</span>
        <span class="right cap" id="entity-count"></span></div>
      <div id="entity-list"><p class="empty">Loading…</p></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">REFLECTION</span>
        <span class="right"><button class="btn" id="reflect-btn">Reflect now</button></span></div>
      <div id="reflect-list"><p class="empty">Loading…</p></div>
    </div>
    <p class="note" id="memory-note"></p>
  </section>

  <section id="view-persona">
    <div><span class="lab">PERSONA</span>
      <p class="cap" style="margin:8px 0 0;max-width:640px">What she has worked out about dealing with you. She writes <code>learned.md</code>; <code>seed.md</code> is yours and code never writes it.</p></div>
    <div class="card pad">
      <div class="card-head"><span class="card-title">ANCHOR</span></div>
      <div id="anchor-gauge" class="kv"></div>
      <div id="anchor-files"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">LEARNED RULES</span>
        <span class="right cap" id="rule-count"></span></div>
      <div id="rule-list"><p class="empty">Loading…</p></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">OBSERVATIONS</span>
        <span class="right cap" id="obs-count"></span></div>
      <div id="obs-list"><p class="empty">Loading…</p></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">EVOLUTION</span>
        <span class="right"><button class="btn" id="evolve-btn">Evolve now</button></span></div>
      <div id="diary-list"><p class="empty">Loading…</p></div>
    </div>
    <p class="note" id="persona-note"></p>
  </section>
```

`.kv` / `.pad` / `.card-head` / `.card-title` / `.empty` / `.note` 는 이미 있는 클래스다. 없으면 `grep -n '\.kv{' daemon/admin/static/index.html` 로 확인하고 가장 가까운 것으로 바꾼다. **새 CSS 토큰을 만들지 않는다.**

- [ ] **Step 3: 렌더와 로드를 넣는다**

`loadToolLog()` 뒤:

```javascript
// --- memory & persona: what she knows, and what she learned ------------------
// Not on the 15s poll: these change once a day and carry ~12KB of markdown.
// Bodies arrive inlined, so `disclose` never fetches - it only unhides.
function disclose(label,body,file){
  if(body==null)return `<p class="cap muted">${esc(file)} — body not loaded (over the cap)</p>`;
  return `<details><summary class="cap">${esc(label)}</summary><pre class="well">${esc(body)}</pre></details>`;
}
function factRow(f){
  return `<div class="trow${f.status==='retired'?' muted':''}">
    <span class="chip k-${f.status==='retired'?'refused':'reflect'}">${f.importance}</span>
    <span class="x">${esc(f.body)}</span>
    <span class="m">${esc(f.key||'')}${f.triggers.length?' · '+esc(f.triggers.join(', ')):''}</span>
    <span class="t">${stamp(f.updated_at)}</span></div>`;
}
function entityRow(e){
  const head=`${e.name}${e.kind?' · '+e.kind:''} · ${e.mentions} mention(s)`
    +(e.links.length?' · '+e.links.join(', '):'');
  return `<div class="trow col">${disclose(head,e.body,e.file)}</div>`;
}
function reflectRow(r){
  const counts=r.status==null?'artifact only'
    :`${r.status} · ${r.messages_read} msg → ${r.facts} fact(s), ${r.entities} entity(ies), ${r.observations} obs`;
  return `<div class="trow col">${disclose(`${r.date}  ${counts}`,r.body,r.file)}</div>`;
}
async function loadMemory(){
  $('#memory-note').textContent='';
  try{const m=await api('/admin/api/memory');
    $('#memory-summary').textContent=
      `${m.facts_active} fact(s) always injected · ${m.entities_total} entity note(s) · `
      +`${m.reflections_total} day(s) reflected`
      +(m.pending_days.length?` · ${m.pending_days.length} day(s) waiting`:'')
      +' · loads when you open this view';
    $('#fact-count').textContent=`${m.facts_active} active · ${m.facts_retired} retired`;
    $('#fact-list').innerHTML=m.facts.length?m.facts.map(factRow).join('')
      :'<p class="empty">Nothing curated yet. Facts land here after a reflection pass.</p>';
    $('#entity-count').textContent=`${m.entities_total}`
      +(m.entities_bodies_truncated?' · some notes not loaded':'');
    $('#entity-list').innerHTML=m.entities.length?m.entities.map(entityRow).join('')
      :'<p class="empty">No entity notes yet.</p>';
    $('#reflect-list').innerHTML=m.reflections.length?m.reflections.map(reflectRow).join('')
      :'<p class="empty">No reflection has run yet.</p>';
    $('#c-memory').textContent=m.facts_active||'';
  }catch(e){$('#fact-list').innerHTML=`<p class="empty">${esc(e.message)}</p>`;}
}
function ruleRow(r){
  const ev=r.evidence.length
    ? `<pre class="well">${r.evidence.map(o=>`[${o.confidence.toFixed(2)}] ${o.body}`).map(esc).join('\n')}</pre>`
    : '<p class="cap muted">No observation row is left for this rule’s evidence.</p>';
  const head=r.status==='retired'
    ? `retired ${stamp(r.retired_at)} — ${r.retired_why||''}`
    : `${stamp(r.created_at)} · ${r.evidence.length} observation(s)`;
  return `<div class="trow col${r.status==='retired'?' muted':''}">
    <div class="row"><span class="x">${esc(r.body)}</span>
      ${r.status==='active'?`<button class="btn" data-forget="${r.id}">Forget</button>`:''}</div>
    <details><summary class="cap">${esc(head)}</summary>${ev}</details></div>`;
}
function obsRow(o){
  return `<div class="trow"><span class="chip k-reflect">${o.confidence.toFixed(2)}</span>
    <span class="x">${esc(o.body)}</span>
    <span class="m">${o.consumed_by==null?'pending':'→ rule '+o.consumed_by}</span>
    <span class="t">${stamp(o.created_at)}</span></div>`;
}
async function loadPersona(){
  $('#persona-note').textContent='';
  try{const p=await api('/admin/api/persona');
    const a=p.anchor;
    $('#anchor-gauge').innerHTML=[
      ['ACTIVE',`${a.active}/${a.max_active}`],
      ['MAX NEW / WEEK',String(a.max_new_per_cycle)],
      ['MIN OBSERVATIONS',String(a.min_observations)],
      ['UNCONSUMED',String(a.unconsumed)],
      ['LAST RULE',a.last_rule_at?stamp(a.last_rule_at):'never'],
    ].map(([k,v])=>`<span class="k">${k}</span><span class="v">${esc(v)}</span>`).join('');
    $('#anchor-files').innerHTML=
      disclose(`seed.md · ${a.seed.lines} line(s) — yours, code never writes this`,
               a.seed.text,a.seed.file)
     +disclose(`learned.md · ${a.learned.lines} line(s) — hers`,
               a.learned.text,a.learned.file);
    $('#rule-count').textContent=`${p.rules_active} active · ${p.rules_retired} retired`;
    $('#rule-list').innerHTML=p.rules.length?p.rules.map(ruleRow).join('')
      :'<p class="empty">She has not concluded anything yet. Rules need '
       +`${a.min_observations} observation(s) and a weekly pass.</p>`;
    $('#obs-count').textContent=
      `${p.observations_total} · ${p.observations_consumed} consumed`;
    $('#obs-list').innerHTML=p.observations.length?p.observations.map(obsRow).join('')
      :'<p class="empty">No observations yet.</p>';
    $('#diary-list').innerHTML=p.diaries.length
      ?p.diaries.map(d=>`<div class="trow col">${disclose(d.date,d.body,d.file)}</div>`).join('')
      :'<p class="empty">No evolution pass has written a diary yet.</p>';
    $('#c-persona').textContent=p.rules_active||'';
    $$('[data-forget]').forEach(b=>b.addEventListener('click',()=>forgetRule(+b.dataset.forget)));
  }catch(e){$('#rule-list').innerHTML=`<p class="empty">${esc(e.message)}</p>`;}
}
async function forgetRule(id){
  const why=prompt('Why should she stop believing this? (recorded with the rule)');
  if(why===null)return;
  if(!why.trim()){$('#persona-note').textContent='A reason is required.';return;}
  $('#persona-note').textContent='Forgetting…';
  try{await api('/admin/api/persona/forget',{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify({id,why})});
    $('#persona-note').textContent=`Rule ${id} retired.`;
    await loadPersona();
  }catch(e){$('#persona-note').textContent=e.message;}
}
async function runPass(btn,note,path,done){
  const b=$(btn);b.disabled=true;$(note).textContent='Running… this makes a model call.';
  try{const r=await api(path,{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify({})});
    $(note).textContent=done(r);
  }catch(e){$(note).textContent=e.message;}
  finally{b.disabled=false;}
}
$('#reflect-btn').addEventListener('click',()=>runPass(
  '#reflect-btn','#memory-note','/admin/api/reflect',
  r=>r.results.length
    ?r.results.map(x=>`${x.date}: ${x.status} (${x.messages_read} msg → ${x.facts} fact(s), `
       +`${x.entities} entity(ies), ${x.observations} obs)`).join(' · ')
    :'Nothing to reflect on: every day with a log already has a reflection.')
  .then(loadMemory));
$('#evolve-btn').addEventListener('click',()=>runPass(
  '#evolve-btn','#persona-note','/admin/api/persona/evolve',
  r=>`${r.date}: ${r.skipped||'ran'} (${r.observations_read} observation(s) read → `
     +`${r.proposed} proposed, ${r.added} added, ${r.retired} retired)`)
  .then(loadPersona));
```

`$`(:534) · `$$`(:535) · `stamp(iso)`(:909)는 이미 있다 — 확인했으니 다시 찾지 않는다.

`.well` 은 **없다**(확인: 0건). `.trow`·`.col`·`.kv`·`.pad`·`.card-head`·`.muted`·`.chip` 은 있다. 그래서 CSS에 딱 두 줄을 추가한다 (기존 토큰만 사용, 새 토큰 금지):

```css
.trow.col{flex-direction:column;align-items:flex-start;gap:6px}
pre.well{background:var(--well);border:1px solid var(--line);border-radius:6px;
  padding:10px;margin:8px 0 0;overflow-x:auto;white-space:pre-wrap;
  font-family:var(--mono);font-size:12px;color:var(--tb)}
```

- [ ] **Step 4: 라우터가 새 HTML을 그대로 서빙하는지 확인한다**

Run: `python3 -m pytest tests/test_admin.py tests/test_admin_mind.py -q`
Expected: passed — `shell()`은 디스크에서 매 요청 읽으므로 마크업 변경에 테스트가 필요 없다. 문법만 확인:

```bash
python3 - <<'PY'
import re,pathlib
t=pathlib.Path("daemon/admin/static/index.html").read_text()
for sid in ("view-memory","view-persona"):
    assert f'id="{sid}"' in t, sid
for nav in ('data-nav="memory"','data-nav="persona"'):
    assert nav in t, nav
assert "loadMemory" in t and "loadPersona" in t
# no markdown parser crept in, and no CDN
assert "marked" not in t and "cdn." not in t
print("ok")
PY
```

- [ ] **Step 5: 커밋**

```bash
git add daemon/admin/static/index.html
git commit -m "admin: two tabs for what she knows and what she learned"
```

---

### Task 8: 문서 · 도달성 · 실물 QA

**Files:**
- Modify: `daemon/CLAUDE.md`, `docs/ARCHITECTURE.md`, `tests/CLAUDE.md` (해당 표에 새 파일 한 줄씩)
- Test: `python3 scripts/check_docs.py`, 실제 브라우저

- [ ] **Step 1: 새 파일을 문서 표에 한 줄씩 넣는다**

먼저 `activity.py` 가 어디에 적혀 있는지 찾아 같은 자리에 `mind.py` 를 넣는다:

```bash
grep -rn "admin/activity.py\|activity.py" daemon/CLAUDE.md docs/ARCHITECTURE.md tests/CLAUDE.md
```

`test_admin_mind.py` 도 `test_admin_activity.py` 옆에 넣는다.

- [ ] **Step 2: 문서 검사와 전체 스위트**

```bash
python3 scripts/check_docs.py && python3 -m pytest -q && python3 -m ruff check .
```
Expected: `ok: N documented path(s) all exist`, 실패 0, ruff clean

- [ ] **Step 3: 커밋**

```bash
git add daemon/CLAUDE.md docs/ARCHITECTURE.md tests/CLAUDE.md
git commit -m "docs: the two new admin tabs and their module"
```

- [ ] **Step 4: 실 데이터로 브라우저에서 확인한다 — 이 스텝은 생략 불가**

그린 유닛 테스트는 증거가 아니다(메모리 `verify-by-running-real`, `qa-drive-the-live-ux`, `partial-success-means-check-harness`). 실 데이터 사본으로 워크트리 빌드를 띄운다. **`cp` 로 sqlite 파일만 복사하면 WAL을 잃는다** — `.backup` 을 쓴다(메모리 `live-qa-a-worktree-build`):

```bash
QA=/private/tmp/claude-501/qa-mind && rm -rf $QA && mkdir -p $QA
cp -R ~/Daemon/data/memory ~/Daemon/data/persona $QA/
sqlite3 ~/Daemon/data/daemon.sqlite3 ".backup '$QA/daemon.sqlite3'"
```

워크트리 루트에서 (cwd가 PYTHONPATH를 가리므로 워크트리 코드가 실행된다):

```bash
DAEMON_DATA_DIR=/private/tmp/claude-501/qa-mind DAEMON_PORT=8799 python3 -m daemon.cli serve
```

브라우저로 `http://127.0.0.1:8799/admin#memory` 와 `#persona` 를 열어 **눈으로** 확인한다:

1. Memory 탭 — 사실 11 active / 1 retired, 엔티티 11개, **성찰 9일**(5개만 카운트가 붙고 나머지는 `artifact only`)
2. 성찰 `▸` 를 펴면 한글 원문이 `<pre>` 로 깨지지 않고 나온다 (DM Mono 폴백)
3. Persona 탭 — ANCHOR 가 `3/20 · 3 · 5 · 0`, `seed.md` 26줄 / `learned.md` 3줄이 접혀 있다
4. 규칙 3개, 각각 `▸` 를 펴면 근거 관찰 3개가 문장으로 나온다
5. 관찰 9개, 전부 `→ rule N`
6. 다이어리 2개(08-24, 08-10) 원문이 나온다
7. **`Forget` 을 실제로 누른다** — 이유를 넣고, `learned.md` 에서 그 줄이 사라지고 retired 로 옮겨가는 것을 확인한 뒤 `git diff` 없는 QA 사본이므로 그대로 둔다
8. **`Reflect now` 를 누른다** — 오늘(08-24) 로그가 아직 성찰되지 않았으므로 실제 모델 호출이 나가고 결과 숫자가 note 에 찍힌다
9. 개발자 콘솔에 에러 0

그리고 **실 데이터에 손대지 않았음**을 확인한다:

```bash
git -C ~/Daemon status --short   # 데이터 디렉터리는 repo 밖이지만 습관으로 확인
sqlite3 "file:$HOME/Daemon/data/daemon.sqlite3?mode=ro" \
  "select count(*) from persona_rules where status='active'"   # 여전히 3
```

- [ ] **Step 5: 실물 QA 결과를 정직하게 보고한다**

9개 항목 각각에 대해 무엇을 보았는지 쓴다. 통과하지 못한 항목이 있으면 그 항목을 적고 고친다 — 부분 통과를 통과로 보고하지 않는다.

---

## Self-Review

**1. 스펙 커버리지**

| 스펙 항목 | 태스크 |
|---|---|
| 결정 1 — 탭 두 개 | 7 |
| 결정 2 — 손잡이 3개, 사실 폐기 제외 | 6 (폐기 라우트 없음) |
| 결정 3 — 본문 인라인, 경로 파라미터 없음 | 2, 3, 4 |
| 결정 4 — 15초 폴링 제외 | 7 Step 1 (`refresh()` 미변경) |
| 결정 5 — 성찰 목록은 파일 축 | 2 (`test_a_…still_listed`) |
| 결정 6 — 마크다운→HTML 변환 없음 | 7 (`esc()` + `<pre>`, Step 4가 파서 부재를 검사) |
| 발견 — `catchup_lock` | 5, 6 |
| Store 읽기 3개 | 1 |
| Memory 화면 | 7 |
| Persona 화면 · 앵커 순서 | 3, 7 |
| Forget 흐름 · divergence | 6 |
| CONTRACTS 5 (seed 읽기만) | 4 (`test_f_no_route_writes_the_seed`) |
| CONTRACTS 6 (관찰 append-only) | 1, 3 (SELECT만) |
| CONTRACTS 10 (루프백) | 4, 6 (기존 `router` 사용) |
| 개인정보 | 8 Step 4 (실물 확인 시 주소가 렌더되는 것을 확인) |
| 테스트 a~f | 2(a,c) 3(b,c) 4(f) 5(e) 6(d,e) |
| 실물 브라우저 QA | 8 |

빈 칸 없음.

**2. 플레이스홀더 검사** — "TBD"/"적절히 처리"/"위의 것 테스트" 없음. 모든 코드 스텝에 실제 코드가 있다. Task 5 Step 3은 조건 분기("문구를 읽는 테스트가 있으면")를 담지만, 두 갈래 모두 무엇을 할지 명시했다.

**3. 타입 일관성**

- `memory_payload(store, data_dir)` — Task 2 정의, Task 4 사용. 일치.
- `persona_payload(store, data_dir, settings)` — Task 3 정의, Task 4 사용. 일치.
- `run_reflection_now(settings, lock)` / `run_persona_evolution_now(settings, lock, *, force=False)` — Task 5 정의, Task 6 사용·monkeypatch. 일치.
- `bodies_truncated` 접두사: `entities_bodies_truncated` / `reflections_bodies_truncated` / `diaries_bodies_truncated` — Task 2·3 정의, Task 7 에서 `entities_bodies_truncated` 만 화면에 쓴다(나머지는 `disclose` 가 `body==null` 로 알아서 표시). 키 이름 일치.
- `MAX_REFLECTION_BODIES` / `MAX_DIARY_BODIES` / `MAX_ENTITY_BODIES` / `MAX_BODY_BYTES` — Task 2 정의, Task 2·3 테스트가 `monkeypatch.setattr(mind, …)` 로 같은 이름을 쓴다. 일치. (스펙 초안의 `MAX_REFLECTIONS`/`MAX_DIARIES`/`MAX_ENTITIES`/`MAX_BODY_BYTES` 에서 이름을 바꿨다 — `MAX_ENTITIES`가 목록 상한과 본문 상한 두 뜻으로 쓰여 충돌했다. 목록 상한은 `MAX_ENTITIES`, 본문 상한은 `MAX_ENTITY_BODIES`.)
- `_bodies(items, *, limit, budget)` — 스펙은 `_read_bodies`로 적었다. 구현 이름은 `_bodies`. Task 2·3 에서 일관.
- `MAX_REFLECTION_ROWS` — Task 2 에서 새로 도입(스펙에 없음). `recent_reflection_runs` 의 행 상한이 사실 목록 상한(`MAX_FACTS`)을 재사용하고 있던 것을 분리했다.

**4. 실제 시그니처 대조** (계획을 쓴 뒤 소스에서 확인, 5건 수정)

| 확인한 것 | 결과 |
|---|---|
| `record_reflection_run` | `now=` 다 (`ts=` 아님) — Task 2 테스트 수정 |
| `recent_reflection_runs` | `id DESC` — `reversed`로 최신 우선이 되는 것 확인, 주석 추가 |
| `insert_entry` / `insert_observation` / `insert_persona_rule` | 계획의 호출과 일치 |
| `create_app(settings)` | 위치 인자 — 일치 |
| `DB_FILENAME = "daemon.sqlite3"` | 일치 |
| `$`(:534) `$$`(:535) `stamp`(:909) | 존재 — 확인 스텝 삭제 |
| `.well` CSS | **없음** — 추가 필요 |
| `Result` / `EvolutionResult` | `app.py`에 **미임포트** — Task 5에 임포트 수정 추가 |
| `"could not start"` 를 읽는 테스트 | **없음** — 문구 병합 확정 |
