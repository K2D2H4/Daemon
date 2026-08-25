# Graded persona learning — implementation plan

> **Outcome:** Task 1 shipped. Tasks 2-6 (the dating mechanism) were built, measured and
> reverted — three runs found no effect and the one significant run did not replicate.
> `daemon/MEASURED.md` carries the result. Task 8 was never run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a single remark from becoming a permanent personality change, without making the daemon ignore what it has genuinely learned.

**Architecture:** Two independent changes. (A) `daemon/reflection.py`'s prompt stops letting manner instructions into `facts`, so they land in `observations` and inherit M4's existing rate limits. (B) the learned block the model sees carries each rule's formation date and observation count, assembled at prompt time from the sqlite columns — never written to `learned.md`, because docs/CONTRACTS.md non-negotiable 3 forbids provenance in prose.

**Tech Stack:** Python 3.13, pytest, sqlite3, Gemini via `LLMGateway`.

**Spec:** [docs/superpowers/specs/2026-08-24-graded-persona-learning-design.md](../specs/2026-08-24-graded-persona-learning-design.md)

## Global Constraints

- **Provenance is columns, never prose** (docs/CONTRACTS.md non-negotiable 3). `learned.md` keeps carrying rule bodies and nothing else. No task may write a date, an evidence count, an id or a status into that file.
- **Markdown is the source of truth**, the mirror is derived (non-negotiable 1). No task changes the write order in `LearnedRules.add`.
- **Only `daemon/app.py` constructs implementations** (non-negotiable 4 / layering). `daemon/persona/loader.py` must not import `Store`.
- **Degrade, never crash.** Every new path must fall back to today's behaviour — plain unannotated bodies — when the mirror is missing, diverged, or unreadable. A persona that fails to assemble costs the turn.
- No test may touch the network, a key, a microphone or a speaker (tests/CLAUDE.md). Live measurement goes in `evals/`, run by hand.
- Assert behaviour in Korean where the product is Korean.
- Every module you touch has a docstring that carries the *reason*, not the *what*. Match that; record the measurement, not the intention.

---

### Task 1: Reflection's `facts` may not carry manner

**Files:**
- Modify: `daemon/reflection.py:112-133` (the `SYSTEM` prompt)
- Test: `tests/test_reflection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks import. This task is self-contained.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reflection.py`:

```python
def test_the_facts_bucket_refuses_manner_and_names_where_it_goes() -> None:
    """The leak this whole plan exists for. On 2026-08-19 the owner said once that
    he was tired of being asked `무슨 재미난 얘기 있어요?`, and reflection wrote

        - 사용자가 AI 비서에게 반복적인 질문(...)을 자제하고 담백하게 대화해 줄 것을 요청함

    into `core.md` as a *fact*, beside his dog's name and his birthday. `core.md`
    is injected whole on every turn, has no cap, no decay and no retraction, so
    one remark became a standing order.

    The prompt already had the right bucket - `observations` is "대화 내용이 아니라
    대화 방식에 대한 것" and feeds M4's rated path (weekly, >=5 observations, <=3
    new, <=20 active, `daemon persona forget`). The model simply filed it in the
    wrong one. So the boundary is stated, and stated as a prohibition on `facts`
    rather than only as a description of `observations`: the description was
    already there and lost.
    """
    from daemon.reflection import SYSTEM

    facts_rule = SYSTEM.split("- entities:")[0]
    assert "말투" in facts_rule and "observations" in facts_rule, (
        "the facts bucket must name manner and say where it goes instead"
    )
    for banned in ("자제", "요청", "선호"):
        assert banned in facts_rule, (
            f"{banned!r} is the shape the misfiled line took; the prompt has to "
            "name the kind, not hope the model generalises"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reflection.py -k manner -v`
Expected: FAIL — `AssertionError: the facts bucket must name manner and say where it goes instead`

- [ ] **Step 3: Write the minimal implementation**

In `daemon/reflection.py`, replace the `- facts:` bullet (currently lines 114-122) with the same text plus a boundary clause. Keep every existing sentence — they carry measured behaviour (`updates`, the merge rule, `triggers`) and none of it is what leaked:

```python
- facts: 앞으로 계속 기억할 가치가 있는 사실. 그날의 잡담은 넣지 않는다.
  **이 사람의 삶과 세계에 대한 것만 넣는다** (이름, 가족, 사는 곳, 일, 일정,
  가진 것). 나를 어떻게 대해 달라는 말 - 말투, 호칭, 태도, 무엇을 하지 말라거나
  더 해 달라는 요청·선호·자제 요구 - 는 사실이 아니다. 그런 것은 facts 가 아니라
  observations 에 넣는다. 한 문장이 '~해 달라고 요청함', '~를 선호함',
  '~를 자제해 달라고 함' 으로 끝난다면 그건 observations 다.
  importance 는 1~10. key 는 나중에 바뀔 수 있는 사실에만 넣는다
  (예: 사는 곳, 직장, 관계). 같은 key 는 이전 사실을 대체한다.
  updates 는 이미 기억하고 있는 사실을 고쳐 쓰는 경우에만, 그 번호를 적는다.
  겹치는 내용을 새로 추가하지 말고 updates 로 대체한다. 새로운 사실이면 null.
  단, 기존 사실에만 있는 내용이 새 문장에서 빠지면 안 된다 - 그럴 때는
  둘을 합친 문장을 쓰거나, 대체하지 말고 새 사실로 추가한다.
  triggers 는 이 사실을 떠올려야 할 때 대화에 나올 만한 단어 2~4개.
  조사 없이 짧게 (예: "이사", "연희동").
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_reflection.py -v`
Expected: PASS, and no other test in the file breaks. If one asserts the exact `SYSTEM` string, update it — it is pinning the prompt, not a behaviour.

- [ ] **Step 5: Commit**

```bash
git add daemon/reflection.py tests/test_reflection.py
git commit -m "reflection: a request about manner is not a fact about a life"
```

---

### Task 2: The annotated bullet, as a pure function

**Files:**
- Modify: `daemon/persona/loader.py`
- Test: `tests/test_persona_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `daemon.persona.loader.rule_line(body: str, *, formed: str, observations: int) -> str` — used by Task 4.

`formed` is a `YYYY-MM-DD` date string, already sliced from `created_at` by the caller. This function does no clock reading and no I/O, so it is testable without fixtures.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_persona_loader.py`:

```python
def test_a_rule_line_carries_its_date_and_how_often_it_was_seen() -> None:
    """A rule with no date is read at full weight forever, which is what made one
    remark on 2026-08-19 still be governing the daemon five days later. The date
    is absolute rather than relative because the model is already told
    `[현재 시각]` on every turn and can do the subtraction, while a stored "어제"
    is a lie by the following week."""
    from daemon.persona.loader import rule_line

    assert rule_line("변명을 싫어한다", formed="2026-08-09", observations=3) == (
        "2026-08-09 (관찰 3건) 변명을 싫어한다"
    )
    assert rule_line("짧은 답을 선호했다", formed="2026-08-23", observations=1) == (
        "2026-08-23 (관찰 1건) 짧은 답을 선호했다"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_persona_loader.py -k rule_line -v`
Expected: FAIL — `ImportError: cannot import name 'rule_line'`

- [ ] **Step 3: Write the minimal implementation**

Add to `daemon/persona/loader.py`, below `rule_bodies`:

```python
def rule_line(body: str, *, formed: str, observations: int) -> str:
    """One learned rule as the model should read it: what was noticed, when, and
    how often.

    The date makes the difference between a tendency and a standing order. An
    undated rule is read at full weight forever - which is how one remark on
    2026-08-19 was still governing the daemon five days later - while a dated one
    can be weighed against `[현재 시각]`, which every prompt already carries.

    Absolute, never relative: "어제" written down once is wrong by the following
    week, and this string is rebuilt on every turn precisely so it never has to be
    stored.
    """
    return f"{formed} (관찰 {observations}건) {body}"
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_persona_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add daemon/persona/loader.py tests/test_persona_loader.py
git commit -m "persona: a rule the model can weigh carries its date and its count"
```

---

### Task 3: Read the annotations off the mirror

**Files:**
- Modify: `daemon/persona/rules.py`
- Test: `tests/test_persona_rules.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `LearnedRules.annotations() -> dict[str, tuple[str, int]]` — maps a rule body to `(formed_date, observation_count)`. Task 5 passes this into `load_persona`.

Joining on the body is not a shortcut: `diverged_bodies` already treats the body as the key between file and mirror, so this uses the same join the divergence gate does. A body the mirror does not know about simply gets no annotation, which is the degrade path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_persona_rules.py`:

`tests/test_persona_rules.py` already defines `store` and `learned` fixtures and
already imports `json`, `UTC`, `datetime`, `Any` and `Store`. Use them — the `db`
fixture is a raw `sqlite3.Connection`, not a `Store`, so `LearnedRules(data_dir,
db)` would fail on the wrong thing.

```python
async def test_annotations_map_each_body_to_its_date_and_evidence_count(
    learned: LearnedRules,
) -> None:
    """What the prompt needs, read from the columns rather than from the file -
    docs/CONTRACTS.md non-negotiable 3, and `rules.py`'s own docstring names the
    threat: a model that could write its own `created_at` could backdate a rule to
    look established. This design weights rules by date, so it would have both the
    means and the motive."""
    await learned.add(
        [
            Proposal(body="변명을 싫어한다", evidence=(1, 2, 3)),
            Proposal(body="짧은 답을 선호했다", evidence=(4,)),
        ],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )

    assert learned.annotations() == {
        "변명을 싫어한다": ("2026-08-09", 3),
        "짧은 답을 선호했다": ("2026-08-09", 1),
    }


async def test_annotations_survive_a_row_with_unreadable_evidence(
    learned: LearnedRules, store: Store
) -> None:
    """`evidence` is a JSON column with a CHECK, but a hand-repaired database or a
    future migration can still put something unexpected there. A rule that cannot
    report its count must still reach the prompt with its date - losing the whole
    persona over one malformed column is the failure this repo calls silent
    degradation, and it is worse than a rule that reads as seen once."""
    await learned.add(
        [Proposal(body="변명을 싫어한다", evidence=(1, 2))],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )
    store.conn.execute("""UPDATE persona_rules SET evidence = '"not-a-list"'""")
    store.conn.commit()

    assert learned.annotations() == {"변명을 싫어한다": ("2026-08-09", 1)}
```

Add `from datetime import UTC, datetime` and `from typing import Any` to the file's imports if they are not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_persona_rules.py -k annotations -v`
Expected: FAIL — `AttributeError: 'LearnedRules' object has no attribute 'annotations'`

- [ ] **Step 3: Write the minimal implementation**

Add to `LearnedRules` in `daemon/persona/rules.py`, next to `active()`:

```python
    def annotations(self) -> dict[str, tuple[str, int]]:
        """Each active rule's body mapped to `(formed date, observation count)`.

        The half of a rule that `learned.md` deliberately does not carry (see the
        module docstring) and that the prompt needs anyway. Assembled here, on
        every turn, from the columns - never written to the file, because a model
        that could write its own `created_at` could backdate a rule to look
        established, and a prompt that weights rules by date hands it a reason to.

        Keyed by body because that is already the join between file and mirror:
        `diverged_bodies` compares the same two sides the same way. A body with no
        row gets no annotation and is rendered plain, which is what today's prompt
        does for every rule.
        """
        out: dict[str, tuple[str, int]] = {}
        for row in self.active():
            try:
                evidence = json.loads(row["evidence"])
                count = len(evidence) if isinstance(evidence, list) else 1
            except (TypeError, ValueError):
                # A rule that cannot report its count still has a date worth
                # having. Counting it as one understates it, which is the safe
                # direction: this block exists to make old single remarks weigh
                # less, never to inflate one.
                count = 1
            out[row["body"]] = (str(row["created_at"])[:10], max(count, 1))
        return out
```

Add `import json` to the module's imports.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_persona_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add daemon/persona/rules.py tests/test_persona_rules.py
git commit -m "persona: the mirror can report what the file must not carry"
```

---

### Task 4: `load_persona` renders the annotated block

**Files:**
- Modify: `daemon/persona/loader.py`
- Test: `tests/test_persona_loader.py`

**Interfaces:**
- Consumes: `rule_line` (Task 2).
- Produces: `load_persona(data_dir: Path, *, annotations: Mapping[str, tuple[str, int]] | None = None) -> str`. Task 5 calls it with the mapping from Task 3.

Keyword-only and defaulted, so every existing caller and test keeps working unchanged and gets exactly today's output.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_persona_loader.py`:

```python
async def test_the_learned_block_is_dated_when_the_mirror_can_say_so(
    data_dir: Path,
) -> None:
    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "seed.md").write_text("나는 벨라다.\n", encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n- 짧은 답을 선호했다\n", encoding="utf-8"
    )

    block = await load_persona(
        data_dir,
        annotations={
            "변명을 싫어한다": ("2026-08-09", 3),
            "짧은 답을 선호했다": ("2026-08-23", 1),
        },
    )

    assert "- 2026-08-09 (관찰 3건) 변명을 싫어한다" in block
    assert "- 2026-08-23 (관찰 1건) 짧은 답을 선호했다" in block
    # The seed is still the anchor and still verbatim, above the learned half.
    assert block.index("나는 벨라다.") < block.index("2026-08-09")


async def test_an_unannotated_rule_still_reaches_the_prompt(data_dir: Path) -> None:
    """The degrade path, and it is the common one: no mirror, a diverged mirror, or
    a body the rows do not know about. Today's behaviour - the plain bullet - is
    the fallback, because a rule dropped for want of a date is a personality
    quietly losing a piece of itself."""
    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n", encoding="utf-8"
    )

    assert "- 변명을 싫어한다" in await load_persona(data_dir)
    assert "- 변명을 싫어한다" in await load_persona(data_dir, annotations={})


async def test_the_learned_header_tells_the_model_to_weigh_not_obey(
    data_dir: Path,
) -> None:
    """`LEARNED_PREFIX` used to introduce these as things worked out about the
    owner - a flat claim, which is how a five-day-old single remark kept full
    force. It now says they are dated observations and that older, thinner ones
    count for less; without that sentence the dates are decoration."""
    from daemon.persona.loader import LEARNED_PREFIX

    assert "날짜" in LEARNED_PREFIX
    assert "관찰" in LEARNED_PREFIX
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_persona_loader.py -k "dated or unannotated or weigh" -v`
Expected: FAIL — `TypeError: load_persona() got an unexpected keyword argument 'annotations'`

- [ ] **Step 3: Write the minimal implementation**

In `daemon/persona/loader.py`, add `from collections.abc import Mapping` to the imports, replace `LEARNED_PREFIX`, and give `load_persona` the new keyword:

```python
LEARNED_PREFIX = (
    "What I've worked out about dealing with you specifically, from our own "
    "conversations (not who I am - that part never changes). 각 줄 앞의 날짜는 "
    "그렇게 느낀 때이고, 괄호 안은 그렇게 본 관찰의 수다. 오래됐거나 관찰이 "
    "한두 번뿐인 것은 그때 그랬다는 뜻이지 늘 그렇다는 뜻이 아니니, 지금 "
    "시각과 견주어 무게를 달아서 읽는다. 규칙이 아니라 관찰이다:"
)
```

```python
async def load_persona(
    data_dir: Path,
    *,
    annotations: Mapping[str, tuple[str, int]] | None = None,
) -> str:
    seed = await read_file(seed_path(data_dir))
    learned = await read_file(learned_path(data_dir))

    parts = []
    if seed:
        parts.append(seed)
    bodies = rule_bodies(learned)
    if bodies:
        found = annotations or {}
        lines = []
        for body in bodies:
            dated = found.get(body)
            # Plain when the mirror cannot say - see the caller's degrade path.
            lines.append(
                rule_line(body, formed=dated[0], observations=dated[1])
                if dated
                else body
            )
        rules = "\n".join(f"- {line}" for line in lines)
        parts.append(f"{LEARNED_PREFIX}\n{rules}")
    return "\n\n".join(parts)
```

Extend the existing `load_persona` docstring with a paragraph naming why the annotation arrives as an argument rather than being read here: this module reads files and must not import `Store` (layering, non-negotiable 4), and the provenance it renders may not live in the file it reads (non-negotiable 3).

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_persona_loader.py tests/test_companion.py -v`
Expected: PASS. `test_companion.py` is in scope because it asserts on the persona block's position among `context()`'s blocks.

- [ ] **Step 5: Commit**

```bash
git add daemon/persona/loader.py tests/test_persona_loader.py
git commit -m "persona: the learned half is observations to weigh, not rules to obey"
```

---

### Task 5: Wire the mirror into the prompt path

**Files:**
- Modify: `daemon/companion.py:210-226` (constructor), `daemon/companion.py:246-255` (`persona`)
- Modify: `daemon/app.py:360`, `daemon/app.py:393`, `daemon/app.py:1299` (the three `Companion(...)` sites)
- Test: `tests/test_companion.py`

**Interfaces:**
- Consumes: `LearnedRules.annotations()` (Task 3), `load_persona(..., annotations=...)` (Task 4).
- Produces: `Companion(..., rules: LearnedRules | None = None)`.

`Companion` takes the `LearnedRules` object rather than a `Store` so it depends on the narrow thing it uses. `app.py` already holds both.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_companion.py`:

`test_companion.py` has no `store` fixture of its own and `db` is a raw
`sqlite3.Connection`, so wrap it: `Store(db)`.

```python
async def test_the_persona_block_is_dated_when_rules_are_wired(
    db: Any, data_dir: Path
) -> None:
    from datetime import UTC, datetime

    from daemon.memory.store import Store
    from daemon.persona.rules import LearnedRules, Proposal

    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "seed.md").write_text("나는 벨라다.\n", encoding="utf-8")
    rules = LearnedRules(data_dir, Store(db))
    await rules.add(
        [Proposal(body="변명을 싫어한다", evidence=(1, 2, 3))],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )
    companion = Companion(FakeMemory(), data_dir=data_dir, rules=rules)

    assert "2026-08-09 (관찰 3건) 변명을 싫어한다" in await companion.persona()


async def test_a_companion_with_no_rules_wired_still_has_a_persona(
    data_dir: Path,
) -> None:
    """The two fake channel/memory setups in `app.py` carry no store, and neither
    do most tests. Voice and text must both keep working there - the same
    degrade-not-crash shape `tools` and delegation already have."""
    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "seed.md").write_text("나는 벨라다.\n", encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n", encoding="utf-8"
    )

    block = await Companion(FakeMemory(), data_dir=data_dir).persona()

    assert "나는 벨라다." in block
    assert "- 변명을 싫어한다" in block


async def test_an_unreadable_mirror_costs_the_dates_not_the_persona(
    data_dir: Path,
) -> None:
    """Reading the mirror is one more thing that can fail on a turn that must not
    fail. A raise here would take the seed down with it."""

    class Broken:
        def annotations(self) -> dict[str, tuple[str, int]]:
            raise sqlite3.OperationalError("no such table: persona_rules")

    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "seed.md").write_text("나는 벨라다.\n", encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n", encoding="utf-8"
    )

    block = await Companion(FakeMemory(), data_dir=data_dir, rules=Broken()).persona()

    assert "나는 벨라다." in block
    assert "- 변명을 싫어한다" in block
```

Add `import sqlite3` and `from typing import Any` to the test file's imports if absent.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_companion.py -k "dated or no_rules_wired or unreadable_mirror" -v`
Expected: FAIL — `TypeError: Companion.__init__() got an unexpected keyword argument 'rules'`

- [ ] **Step 3: Write the minimal implementation**

In `daemon/companion.py`, add the parameter to `__init__` (after `tools`):

```python
        rules: LearnedRulesLike | None = None,
```
```python
        self._rules = rules
```

Define the narrow protocol above the class so `companion.py` does not import `LearnedRules` itself (it would pull `Store` in behind it):

```python
class LearnedRulesLike(Protocol):
    """Just the half of `daemon/persona/rules.LearnedRules` the prompt needs.

    A protocol rather than the class so this module keeps out of the store's
    import graph, and so a test can hand in something that raises.
    """

    def annotations(self) -> dict[str, tuple[str, int]]: ...
```

Add `from typing import Protocol` to the imports. Then replace `persona()`:

```python
    async def persona(self) -> str:
        """Who the daemon is, as prompt text. Empty if there is no persona yet.

        Re-read per call, which is a promise the product makes: `seed.md` is the
        file the owner edits to change how they are spoken to, and the edit lands on
        the next turn (docs/PLAN.md 5.1). Assembly - the human-owned seed plus M4's
        accumulated learned rules - is `daemon/persona/loader.py`'s job, so this is
        one call and stays one call.

        The dates and observation counts come from the mirror rather than from
        `learned.md`, which carries bodies alone on purpose (non-negotiable 3). A
        mirror that cannot be read costs the annotation and nothing else: the
        rules still go in, undated, exactly as they did before this existed.
        """
        annotations = None
        if self._rules is not None:
            try:
                annotations = self._rules.annotations()
            except Exception:
                logger.exception("companion: could not date the learned rules")
        return await persona.load_persona(self._data_dir, annotations=annotations)
```

In `daemon/app.py`, all three `Companion(...)` sites gain `rules=...`. At each site, build it only where a `Store` exists — the two fake-injection paths have none:

```python
                rules=LearnedRules(settings.data_dir, store) if store is not None else None,
```

Import `LearnedRules` at the top of `app.py` alongside the other persona imports.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest -v`
Expected: PASS, whole suite. `tests/test_reachable.py` checks that a built thing is constructed by the assembled app; `LearnedRules` is already constructed by `evolve.py`, so no `PENDING_*` entry changes.

- [ ] **Step 5: Commit**

```bash
git add daemon/companion.py daemon/app.py tests/test_companion.py
git commit -m "persona: the prompt path can read what the file must not carry"
```

---

### Task 6: Evolve stops writing conclusions

**Files:**
- Modify: `daemon/persona/evolve.py:77-92` (the `SYSTEM` prompt)
- Test: `tests/test_persona_evolve.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks import.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_persona_evolve.py`:

```python
def test_the_rule_prompt_asks_for_an_observation_not_a_standing_demand() -> None:
    """The prompt used to say `body 는 이 사람에 대한 사실처럼 짧게 한 문장으로 적는다`,
    and a fact carries no strength, so it is read at full weight forever. What that
    produced after five days of terse terminal QA was

        용건 위주의 빠른 응답과 즉각적이고 담백한 피드백을 요구한다

    - a standing demand governing a midnight voice chat as much as a debugging
    session. Naming the unwanted form is the point: `CALLED_BY_NAME` already
    measured that omitting a move is not enough, the move has to be forbidden.
    """
    from daemon.persona.evolve import SYSTEM

    assert "사실처럼" not in SYSTEM
    assert "요구한다" in SYSTEM, "the standing-demand form has to be named to be banned"
    assert "관찰" in SYSTEM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_persona_evolve.py -k standing_demand -v`
Expected: FAIL — `AssertionError: assert '사실처럼' not in SYSTEM`

- [ ] **Step 3: Write the minimal implementation**

In `daemon/persona/evolve.py`, replace the `body` sentence inside the `- rules:` bullet. Keep the rest of the bullet — `evidence` and `key` are load-bearing and unrelated:

```python
- rules: 여러 관찰을 관통하는 패턴이 보일 때만 규칙을 만든다. 관찰 하나만 보고
  규칙을 만들지 않는다. body 는 그 무렵 이 사람이 어떠했는지에 대한 **관찰**을
  짧게 한 문장으로 적는다. 늘 그렇다는 단정이나 상시 요구로 쓰지 않는다 -
  '~를 요구한다', '~를 중시한다' 처럼 언제나 참인 성질처럼 적지 말고,
  '~한 편이었다', '~를 선호했다' 처럼 그때 그렇게 보였다고 적는다.
  날짜와 관찰 수는 시스템이 붙이므로 body 에 쓰지 않는다.
  evidence 는 이 규칙의 근거가 된 관찰의 id 목록(정수) - 입력에
  실제로 있던 id만 쓴다. key 는 이 규칙이 나중에 바뀔 수 있는 것이면 넣는다
  (예: "greeting_style"). 같은 key 는 이전 규칙을 대체한다.
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_persona_evolve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add daemon/persona/evolve.py tests/test_persona_evolve.py
git commit -m "persona: a week of terse QA is an observation, not a standing demand"
```

---

### Task 7: Measure both arms against the live model

**Files:**
- Create: `evals/graded_persona_spike.py`
- Modify: `evals/CLAUDE.md` (one row in the table)
- Modify: `daemon/MEASURED.md` (the result, after running it)

**Interfaces:**
- Consumes: the prompts from Tasks 1, 4 and 6.
- Produces: numbers. Nothing imports this.

This is an eval, not a test: it needs a real key and never runs in CI (tests/CLAUDE.md). It follows `evals/m0_voice_spike.py`'s shape — read `.env` by hand, print a table, write nothing to the repo.

- [ ] **Step 1: Write the spike**

Follow `evals/m0_voice_spike.py`'s shape: read `.env` by hand, print a table,
write nothing to the repo. Create `evals/graded_persona_spike.py`:

```python
"""Did dating a learned rule make the daemon calmer, or just deafer?

Three arms, 30 trials each, old prompt against new, identical inputs.

Arm 1 and arm 2 are one A/B run twice with different probes, and arm 2 is the
one that can fail this change: a run where the stale correction stops dominating
AND the real preference stops being honoured is a regression reported as a
success. Arm 3 measures (A) over reflection rather than conversation.
"""
from __future__ import annotations

import asyncio, json, os, pathlib, sys
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daemon.llm.base import Message
from daemon.persona.loader import LEARNED_PREFIX, rule_line

SEED = "나는 벨라다. 재치 있고 장난기 있다. 다정함은 툭툭 던지듯 챙기는 것으로 드러낸다."

STALE = ("담백하게, 용건 위주로 이야기해 달라고 했다", "2026-08-19", 1)
REAL = ("문제가 생겼을 때 변명하는 것을 싫어한다", "2026-08-09", 3)

OLD_PREFIX = (
    "What I've worked out about dealing with you specifically, from our own "
    "conversations (not who I am - that part never changes):"
)

TODAY = "[현재 시각] 지금은 2026년 08월 24일 월요일 저녁 8시다."


def persona(*, dated: bool) -> str:
    """The two arms differ only in whether the rules carry their provenance."""
    lines = []
    for body, formed, seen in (REAL, STALE):
        lines.append(
            f"- {rule_line(body, formed=formed, observations=seen)}" if dated else f"- {body}"
        )
    prefix = LEARNED_PREFIX if dated else OLD_PREFIX
    return SEED + "\n\n" + prefix + "\n" + "\n".join(lines)


# Probes with room for the behaviour to appear. A three-word greeting has none -
# daemon/MEASURED.md records a 4/20-vs-2/20 tie that was a fact about the probe.
WARM = "그냥 목소리 듣고 싶어서 불렀어. 요즘 어떻게 지내?"
BROKEN = "방금 시킨 거 왜 안 됐어? 결과가 하나도 안 나왔는데."

JUDGE = """아래는 AI 동반자가 사용자에게 한 답변이다. 질문 하나에만 답한다.
{question}
yes 또는 no 한 단어만 출력한다."""

Q_TERSE = "이 답변이 용건만 처리하는 사무적인 말투인가? 농담이나 자기 이야기가 없는가?"
Q_OWNS = "이 답변이 변명 없이 문제를 인정하고 바로 확인하겠다고 하는가?"


async def reply(gateway, persona_text: str, probe: str) -> str:
    out = await gateway.complete(
        "chat",
        [
            Message(role="system", content=persona_text),
            Message(role="system", content=TODAY),
            Message(role="user", content=probe),
        ],
    )
    return out.text.strip()


async def judge(gateway, text: str, question: str) -> bool:
    out = await gateway.complete(
        "chat",
        [
            Message(role="system", content=JUDGE.format(question=question)),
            Message(role="user", content=text),
        ],
    )
    return out.text.strip().lower().startswith("yes")


async def arm(gateway, label: str, *, dated: bool, probe: str, question: str, n: int) -> int:
    hits, block = 0, persona(dated=dated)
    for i in range(n):
        text = await reply(gateway, block, probe)
        verdict = await judge(gateway, text, question)
        hits += verdict
        # Printed beside the reply so every label can be audited by hand:
        # MEASURED.md records a parse that mislabelled 7 of 60 records.
        print(f"   {label:<12} {i+1:>2}: {'YES' if verdict else 'no '}  {text[:110]}", flush=True)
    print(f"\n== {label}: {hits}/{n}\n", flush=True)
    return hits


def fisher(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-tailed p that the two arms came from the same distribution."""
    total, k = n_a + n_b, a + b
    return sum(
        comb(n_a, i) * comb(n_b, k - i) / comb(total, k)
        for i in range(0, min(a, k) + 1)
        if 0 <= k - i <= n_b
    )


async def main() -> None:
    from daemon.config import Settings
    from daemon.llm.gateway import build_gateway

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    gateway = build_gateway(Settings())

    print("ARM 1 - does the stale one-off stop dominating? (want: down)\n")
    a1 = await arm(gateway, "undated", dated=False, probe=WARM, question=Q_TERSE, n=n)
    b1 = await arm(gateway, "dated", dated=True, probe=WARM, question=Q_TERSE, n=n)

    print("ARM 2 - is the real preference still honoured? (want: UNCHANGED)\n")
    a2 = await arm(gateway, "undated", dated=False, probe=BROKEN, question=Q_OWNS, n=n)
    b2 = await arm(gateway, "dated", dated=True, probe=BROKEN, question=Q_OWNS, n=n)

    print(f"ARM 1 stale dominates : undated {a1}/{n} -> dated {b1}/{n}  p={fisher(b1,n,a1,n):.5f}")
    print(f"ARM 2 real honoured   : undated {a2}/{n} -> dated {b2}/{n}  p={fisher(a2,n,b2,n):.5f}")
    if b2 < a2 * 0.8:
        print("\nARM 2 FELL. This traded learning away for calmness - do not ship.")


asyncio.run(main())
```

Arm 3 measures (A) and runs over reflection rather than conversation. Add it to
the same file: read the real `data/memory/log/2026-08-19.md`, send it to
`reflection.SYSTEM` (old text and new) 30 times each, parse the JSON with
`daemon.reflection.extract_json`, and count how often a sentence matching
`담백|자제|요청함|선호` lands in `facts` versus `observations`. Report the same
way — counts, a p-value, and every verdict printed beside its sentence.

Confirm `build_gateway`'s actual name and signature against `daemon/llm/gateway.py`
before running; if the assembled `Settings` needs more than `.env`, mirror what
`evals/openai_compatible_loop_spike.py` does.

- [ ] **Step 2: Run it**

```bash
python3 -m evals.graded_persona_spike 30
```

Expected: arm 1 down with p < 0.05, arm 2 unchanged, arm 3 up with p < 0.05.

- [ ] **Step 3: Audit the labels by hand**

Re-parse the output and confirm every verdict matches its reply text. A mismatch means the measurement is wrong, not the code.

- [ ] **Step 4: Write the result down**

Add the table to `daemon/MEASURED.md` — including any arm that did not move, and what it would have taken to see it if it did not. Add the spike's row to `evals/CLAUDE.md`'s table.

- [ ] **Step 5: Commit**

```bash
git add evals/graded_persona_spike.py evals/CLAUDE.md daemon/MEASURED.md
git commit -m "evals: what dating a learned rule actually changed, both arms"
```

- [ ] **Step 6: If arm 2 fell, stop**

Do not ship. Report to the owner that the change traded learning away for calmness, and that (C) — strength and decay, deferred in the spec — is the next thing to try instead of tuning this prompt further.

---

### Task 8: Retire what is already written (owner's data — confirm first)

**Files:** none in the repo. This task touches `~/Daemon/data/`.

**Interfaces:** none.

Fixing the prompts does not remove the sentences already in place. Every step below changes the owner's own memory, so **ask before running any of them** and show the exact line first.

- [ ] **Step 1: Retire the learned rule**

```bash
daemon persona forget 2 --why "one debugging week, not how he wants to be talked to"
```

- [ ] **Step 2: Retire the curated fact**

`memory_entries` id 12 (`사용자가 AI 비서에게 … 담백하게 대화해 줄 것을 요청함`) and the matching line in `data/memory/core.md`. **There is no CLI for this** — the schema has `status='retired'` and nothing exposes it, so it is a manual `UPDATE` plus a file edit. Note in the handover that the nightly pass rewrites `core.md` whole and the evidence is still in the log, so the line can return; Task 1 is what stops that.

- [ ] **Step 3: Pin the form of address in the seed**

`창조주님` comes from `core.md:10`, which is a true fact and should stay. The address belongs in the one file nothing overwrites. Offer the owner this line for `seed.md` and let them make the edit:

```
- How I address the user: 존댓말 (가끔 편하게 풀어도 됨). 호칭은 '대현님'.
  '창조주', '개발자님' 같은 호칭은 쓰지 않는다.
```

- [ ] **Step 4: Report the gap**

`daemon persona forget` exists for learned rules and has no counterpart for the always-injected curated tier. Recorded in the spec's "Cleanup" section; raise it as its own piece of work rather than smuggling a CLI into this plan.
