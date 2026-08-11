"""The daily reflection pass - M2.

Reads one day of conversation and writes three things:

  * **curated facts** into `memory/core.md` and `memory_entries` (always injected)
  * **entity notes** into `memory/entities/*.md` (searched, wiki-linked)
  * **observations** into `observations` (append-only; M4's evidence)

plus the artifact a human reads to check the other three:
`memory/reflections/YYYY-MM-DD.md`. That file is also the idempotence marker -
if it exists, the day has been reflected on. The marker is a file rather than a
column because `memory/schema.sql` is frozen, and because the state it records
belongs to the markdown side of the contract anyway.

## Everything here treats the model's output as hostile

Not because the model is adversarial but because the difference does not matter:
its output names files, sets a recall multiplier, and can *retire* facts the user
stated. So `_clean` clamps every number, `entities.safe_name` refuses anything
path-shaped, keys are normalised to a narrow charset, and an unparseable reply
writes nothing at all rather than writing what it managed. A half-applied
reflection is worse than a skipped one: the day is marked done either way.

## Order

The artifact is written first, then each fact and note writes its own markdown
before its own mirror row (docs/CONTRACTS.md non-negotiable 1). So a crash
anywhere leaves markdown that `daemon reindex` can mirror, never a row pointing at
prose that does not exist.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import open_private_append, secure_dir
from daemon.llm.base import Message, ProviderError
from daemon.llm.gateway import LLMGateway
from daemon.memory import entities as entity_notes
from daemon.memory import log
from daemon.memory.curated import MAX_INJECTED, CuratedMemory
from daemon.memory.entities import EntityNotes, UnsafeName
from daemon.memory.store import Store
from daemon.tasks import Task

logger = logging.getLogger(__name__)

REFLECTIONS_SUBDIR = Path("memory") / "reflections"

MAX_FACTS = 8
MAX_ENTITIES = 12
MAX_OBSERVATIONS = 6
"""Per-day caps. A pass that produced fifty facts would drown the always-injected
tier in one night, and the cap is cheaper than trusting the prompt to self-limit."""

_KEY_RE = re.compile(r"[^a-z0-9_]+")

SYSTEM = """너는 하루치 대화를 정리하는 역할이다. 아래 규칙을 지켜 JSON만 출력한다.

- facts: 앞으로 계속 기억할 가치가 있는 사실. 그날의 잡담은 넣지 않는다.
  importance 는 1~10. key 는 나중에 바뀔 수 있는 사실에만 넣는다
  (예: 사는 곳, 직장, 관계). 같은 key 는 이전 사실을 대체한다.
  updates 는 이미 기억하고 있는 사실을 고쳐 쓰는 경우에만, 그 번호를 적는다.
  겹치는 내용을 새로 추가하지 말고 updates 로 대체한다. 새로운 사실이면 null.
  단, 기존 사실에만 있는 내용이 새 문장에서 빠지면 안 된다 - 그럴 때는
  둘을 합친 문장을 쓰거나, 대체하지 말고 새 사실로 추가한다.
  triggers 는 이 사실을 떠올려야 할 때 대화에 나올 만한 단어 2~4개.
  조사 없이 짧게 (예: "이사", "연희동").
- entities: 사람 / 장소 / 프로젝트 / 주제. note 는 그 대상에 대해 알게 된 것
  한두 문장. links 는 함께 언급된 다른 대상의 이름.
- observations: 이 사람을 어떻게 대하면 좋은지에 대한 관찰.
  대화 내용이 아니라 대화 방식에 대한 것이다. confidence 는 0~1.

확실하지 않으면 넣지 않는다. 빈 배열도 정답이다. 설명이나 인사말 없이 JSON만.

{"facts": [{"body": "...", "importance": 5, "key": null, "updates": null,
            "triggers": ["..."]}],
 "entities": [{"name": "...", "kind": "person", "note": "...", "links": []}],
 "observations": [{"body": "...", "confidence": 0.5}]}"""

KNOWN_HEADER = "[이미 기억하고 있는 것 - 고쳐 쓸 사실은 이 번호를 updates 에 적는다]"


@dataclass(frozen=True, slots=True)
class Fact:
    body: str
    importance: int = 5
    key: str | None = None
    updates: int | None = None
    """The id of a curated fact this one replaces (ADR 0010). Unverified here -
    `Reflection._apply` checks it against the store, because a number the model
    chose is a claim about a row, not a row."""
    triggers: tuple[str, ...] = ()
    """Words that should pull this fact forward even when the importance budget
    would have dropped it. Recall matches them as substrings against the query -
    see `memory.recall.MemoryRecall._curated_tier`."""


@dataclass(frozen=True, slots=True)
class EntityDraft:
    name: str
    note: str
    kind: str | None = None
    links: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Observation:
    body: str
    confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class Conclusion:
    """What the model proposed, already clamped and filtered."""

    facts: tuple[Fact, ...] = ()
    entities: tuple[EntityDraft, ...] = ()
    observations: tuple[Observation, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.facts or self.entities or self.observations)


@dataclass(frozen=True, slots=True)
class Result:
    """What the pass did. Every field is reported rather than logged, because a
    reflection that silently did nothing looks exactly like one that worked."""

    date: str
    status: str
    """`written` · `skipped` (already done) · `empty` (nothing to read) ·
    `nothing` (looked, nothing eligible, marked done) · `unparseable` ·
    `unavailable` (the model could not be reached)."""
    messages_read: int = 0
    facts: int = 0
    entities: int = 0
    observations: int = 0
    detail: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"written", "skipped", "empty", "nothing"}


def artifact_path(data_dir: Path, date: str) -> Path:
    return data_dir / REFLECTIONS_SUBDIR / f"{date}.md"


# --- parsing ----------------------------------------------------------------


def extract_json(text: str) -> dict[str, object] | None:
    """The first JSON object in a model reply, or None.

    Tolerant on purpose. A local 4B model wraps JSON in ```json fences, prefixes
    it with "물론이죠!", and sometimes appends a second object. Being strict here
    would turn a formatting habit into a lost day of reflection, and the day is
    marked done either way.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced[1]] if fenced else []
    start = text.find("{")
    if start != -1:
        candidates.append(text[start : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean(raw: dict[str, object]) -> tuple[Conclusion, list[str]]:
    """Turn the model's object into a `Conclusion`, dropping what cannot be used.

    Returns the problems alongside it rather than raising: one malformed entry in
    a list of eight must not cost the other seven, but it must also not be
    invisible.
    """
    problems: list[str] = []

    facts: list[Fact] = []
    for item in _items(raw, "facts", problems)[:MAX_FACTS]:
        body = _text(item, "body")
        if not body:
            problems.append("a fact with no body")
            continue
        facts.append(
            Fact(
                body=body,
                importance=_int(item, "importance", 5, 1, 10),
                key=_key(item),
                updates=_updates(item, problems),
                triggers=_triggers(item, problems),
            )
        )
    facts = _one_per_key(facts, problems)

    drafts = []
    for item in _items(raw, "entities", problems)[:MAX_ENTITIES]:
        name, note = _text(item, "name"), _text(item, "note")
        if not name or not note:
            problems.append("an entity with no name or no note")
            continue
        try:
            name = entity_notes.safe_name(name)
        except UnsafeName as exc:
            problems.append(f"unusable entity name: {exc}")
            continue
        drafts.append(
            EntityDraft(
                name=name,
                note=note,
                kind=_text(item, "kind") or None,
                links=_links(item, problems),
            )
        )

    observations = []
    for item in _items(raw, "observations", problems)[:MAX_OBSERVATIONS]:
        body = _text(item, "body")
        if not body:
            problems.append("an observation with no body")
            continue
        observations.append(
            Observation(body=body, confidence=_float(item, "confidence", 0.5, 0.0, 1.0))
        )

    return Conclusion(tuple(facts), tuple(drafts), tuple(observations)), problems


def _one_per_key(facts: list[Fact], problems: list[str]) -> list[Fact]:
    """At most one fact per supersession key, keeping the most important.

    Found by running the pass for real. A local model answered "이주 위치: 연희동"
    (importance 8) and "이전 주소: 망원동" (importance 3) both keyed `location`. Applied
    in order, the second retired the first, so `core.md` ended up holding the
    *less* important half of a fact the artifact claimed in full - a silent
    inversion, and one no unit test with a hand-written reply would have produced.

    A key means "one active fact for this", so resolving it here rather than
    letting the mirror's unique index arbitrate makes the choice explicit and the
    artifact honest: it is rendered from what this returns.
    """
    chosen: dict[str, Fact] = {}
    for fact in facts:
        if fact.key is None:
            continue
        previous = chosen.get(fact.key)
        if previous is None:
            chosen[fact.key] = fact
            continue
        problems.append(f"two facts keyed {fact.key!r}; kept the more important one")
        if fact.importance > previous.importance:
            chosen[fact.key] = fact
    # The model's own ordering is preserved for everything that did not collide.
    return [
        fact for fact in facts if fact.key is None or chosen[fact.key] is fact
    ]


def _items(raw: dict[str, object], key: str, problems: list[str]) -> list[dict[str, object]]:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append(f"{key} was not a list")
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        else:
            problems.append(f"an entry in {key} was not an object")
    return out


def _text(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    return " ".join(value.split()) if isinstance(value, str) else ""


def _int(item: dict[str, object], key: str, default: int, low: int, high: int) -> int:
    """Clamped, never rejected. `importance` multiplies the recall score, so an
    out-of-range value would let one night's conclusion outrank everything."""
    value = item.get(key, default)
    try:
        return max(low, min(high, int(float(value))))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float(item: dict[str, object], key: str, default: float, low: float, high: float) -> float:
    value = item.get(key, default)
    try:
        return max(low, min(high, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _key(item: dict[str, object]) -> str | None:
    """A supersession key, narrowed to `[a-z0-9_]`.

    The model chooses this and it *retires* an existing fact, so the charset is a
    deliberate bottleneck: it keeps a key recognisable and stable across days
    (`home`, `job`) and stops one from carrying punctuation or whitespace that
    would make two spellings of the same key look like two different facts.
    """
    raw = _text(item, "key").lower()
    if not raw:
        return None
    narrowed = _KEY_RE.sub("_", raw).strip("_")
    return narrowed[:40] or None


def _updates(item: dict[str, object], problems: list[str]) -> int | None:
    """The id of the fact being replaced, or None.

    Only the shape is checked here; whether the row exists and is still active is
    `_apply`'s job, which is the half that needs the store. A value that is present
    but not a positive integer is reported rather than dropped silently: it means
    the model tried to retire something and the pass declined, which is exactly the
    kind of near-miss that should be visible before it becomes a habit.
    """
    value = item.get("updates")
    if value is None or isinstance(value, bool):
        return None
    try:
        target = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        problems.append(f"updates was not an id: {value!r}")
        return None
    if target <= 0:
        problems.append(f"updates was not an id: {value!r}")
        return None
    return target


MAX_TRIGGERS = 4
MAX_TRIGGER_CHARS = 24


def _triggers(item: dict[str, object], problems: list[str]) -> tuple[str, ...]:
    """Trigger phrases, bounded and deduplicated.

    Bounded because recall matches every phrase against every query on the voice
    latency path, and because a "phrase" that is really a sentence matches nothing:
    the match is a substring test, so the longer the phrase the narrower it gets,
    until a fact with a paragraph-long trigger can never be pulled forward at all.

    Without this the column had no producer. Recall implemented the matching and
    `Store.insert_entry` accepted the value, so the feature was complete, tested,
    and reachable by nothing - which is the defect shape `tests/test_reachable.py`
    exists for, one layer below what that file can see.
    """
    value = item.get("triggers")
    if value is None:
        return ()
    if not isinstance(value, list):
        problems.append("triggers was not a list")
        return ()
    out = []
    for phrase in value:
        if not isinstance(phrase, str):
            continue
        cleaned = " ".join(phrase.split())
        if cleaned and len(cleaned) <= MAX_TRIGGER_CHARS:
            out.append(cleaned)
    return tuple(dict.fromkeys(out))[:MAX_TRIGGERS]


def _links(item: dict[str, object], problems: list[str]) -> tuple[str, ...]:
    value = item.get("links")
    if not isinstance(value, list):
        return ()
    out = []
    for name in value:
        if not isinstance(name, str):
            continue
        try:
            out.append(entity_notes.safe_name(name))
        except UnsafeName as exc:
            problems.append(f"unusable link: {exc}")
    return tuple(dict.fromkeys(out))


# --- the artifact -----------------------------------------------------------


def render_artifact(date: str, conclusion: Conclusion, *, messages_read: int) -> str:
    lines = [
        f"# {date} 성찰",
        "",
        f"대화 {messages_read}건을 읽고 정리했다. 이 파일은 사람이 검토용으로 읽는 것이고,",
        "여기 있는 내용은 `memory/core.md` · `memory/entities/` 에 반영되어 있다.",
        "",
    ]
    if conclusion.facts:
        lines += ["## 기억할 사실", ""]
        for fact in conclusion.facts:
            suffix = f" (key: {fact.key})" if fact.key else ""
            if fact.updates is not None:
                suffix += f" · updates: {fact.updates}"
            if fact.triggers:
                suffix += f" · triggers: {', '.join(fact.triggers)}"
            lines.append(f"- [{fact.importance}] {fact.body}{suffix}")
        lines.append("")
    if conclusion.entities:
        lines += ["## 엔티티", ""]
        for draft in conclusion.entities:
            kind = f" ({draft.kind})" if draft.kind else ""
            links = "".join(f" [[{name}]]" for name in draft.links)
            lines.append(f"- **{draft.name}**{kind}: {draft.note}{links}")
        lines.append("")
    if conclusion.observations:
        lines += ["## 관찰 (대하는 방식)", ""]
        for observation in conclusion.observations:
            lines.append(f"- [{observation.confidence:.2f}] {observation.body}")
        lines.append("")
    if not conclusion:
        lines += ["정리할 만한 것이 없었다.", ""]
    return "\n".join(lines)


def _write_artifact(path: Path, text: str) -> None:
    secure_dir(path.parent)
    with open_private_append(path) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


# --- the pass ---------------------------------------------------------------


class Reflection:
    """One night's consolidation. Constructed per run; holds no state between them."""

    def __init__(
        self,
        data_dir: Path,
        store: Store,
        gateway: LLMGateway,
        *,
        curated: CuratedMemory | None = None,
        notes: EntityNotes | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._store = store
        self._gateway = gateway
        self._curated = curated or CuratedMemory(data_dir, store)
        self._notes = notes or EntityNotes(data_dir, store)

    async def run(self, date: str, *, force: bool = False) -> Result:
        """Reflect on one local day, and record that it ran.

        Safe to call twice - the second call is a no-op unless `force`, because the
        artifact is the marker. The audit row is not that marker and must not be
        read as one: it is written for every outcome - including `unavailable`,
        which writes no artifact at all, and including a pass that raised, which
        records `failed` on its way out rather than vanishing. A night the model
        could not be reached and a night with nothing to say are indistinguishable
        on disk, and that difference is exactly what someone asking "did last night
        run?" means.
        """
        try:
            result = await self._reflect(date, force=force)
        except Exception as exc:
            # A pass that *broke* - an unwritable artifact, a sqlite error, anything
            # the gateway raises that is not ProviderError - is the one outcome this
            # row exists to make visible, and it was the one outcome that left none:
            # the exception used to travel straight past the record below.
            self._store.record_reflection_run(
                date=date,
                status="failed",
                messages_read=0,
                facts=0,
                entities=0,
                observations=0,
                detail=f"{type(exc).__name__}: {exc}",
                now=clock_now(),
            )
            raise
        self._store.record_reflection_run(
            date=result.date,
            status=result.status,
            messages_read=result.messages_read,
            facts=result.facts,
            entities=result.entities,
            observations=result.observations,
            detail=result.detail,
            now=clock_now(),
        )
        return result

    async def _reflect(self, date: str, *, force: bool = False) -> Result:
        path = artifact_path(self._data_dir, date)
        if path.exists() and not force:
            return Result(date=date, status="skipped", detail=f"{path.name} already exists")

        rows = self._store.messages_for_day(date)
        if not rows:
            return self._nothing_to_read(date, path)

        try:
            completion = await self._gateway.complete(
                Task.REFLECTION,
                [
                    Message(role="system", content=SYSTEM),
                    Message(role="user", content=self._known_facts() + _transcript(rows)),
                ],
            )
        except ProviderError as exc:
            # Not an error state to recover from here: without a model there is
            # nothing to write, and the day stays unmarked so tomorrow retries it.
            logger.warning("reflection: model unavailable for %s (%s)", date, exc)
            return Result(date=date, status="unavailable", messages_read=len(rows), detail=str(exc))

        raw = extract_json(completion.text)
        if raw is None:
            logger.warning("reflection: %s produced no JSON for %s", completion.model, date)
            return Result(
                date=date,
                status="unparseable",
                messages_read=len(rows),
                detail=f"{completion.model} did not return a JSON object",
            )

        conclusion, problems = _clean(raw)
        conclusion = self._resolve_updates(conclusion, problems)
        # The artifact goes down first: it is the markdown record of what this pass
        # concluded, and everything below it is a mirror of that.
        _write_artifact(path, render_artifact(date, conclusion, messages_read=len(rows)))

        applied = await self._apply(conclusion, date, problems)
        return Result(
            date=date,
            status="written",
            messages_read=len(rows),
            problems=problems,
            **applied,
        )

    def _resolve_updates(self, conclusion: Conclusion, problems: list[str]) -> Conclusion:
        """Strip every `updates` that does not name a fact this pass can retire.

        Before the artifact is rendered, not during `_apply`, for the reason
        `_one_per_key` gives: the artifact is rendered from the conclusion, so a
        claim settled afterwards is a claim the file gets wrong. An id the model
        invented would otherwise be written down as a retirement that never
        happened.

        The active set shrinks as it goes, so two facts in one night claiming the
        same row cannot both be honoured - the second is left as an addition rather
        than silently retiring nothing, which is the same shape as two facts
        colliding on one key.
        """
        active = {int(row["id"]) for row in self._store.active_entries(MAX_INJECTED)}
        facts = []
        for fact in conclusion.facts:
            if fact.updates is None:
                facts.append(fact)
                continue
            if fact.updates not in active:
                problems.append(f"updates named {fact.updates}, which is not an active fact")
                facts.append(replace(fact, updates=None))
                continue
            active.discard(fact.updates)
            facts.append(fact)
        return replace(conclusion, facts=tuple(facts))

    def _known_facts(self) -> str:
        """The curated tier, numbered, ahead of the transcript.

        Without this the pass was write-only with respect to memory: supersession
        fired only when the model independently reinvented the same `key` string on
        two different nights, which is not a mechanism (ADR 0010). Bounded by the
        injection budget for the same reason that tier is - it is the set the
        daemon actually carries, so it is the set worth reconciling against.

        Empty when nothing is known yet, header included, rather than an empty
        section the model has to interpret.
        """
        known = self._store.active_entries(MAX_INJECTED)
        if not known:
            return ""
        lines = [KNOWN_HEADER, *(f"{row['id']}: {row['body']}" for row in known), "", ""]
        return "\n".join(lines)

    def _nothing_to_read(self, date: str, path: Path) -> Result:
        """A day with no eligible messages: either not mirrored yet, or genuinely
        nothing to reflect on. The two need different answers.

        Reported by running it: a day whose log existed but yielded nothing stayed
        in `pending_days` forever, so `daemon doctor` nagged about a day that
        `daemon reflect` could never clear. Marking every such day done is just as
        wrong - a mirror that has not caught up yet would be skipped permanently,
        and that is the case non-negotiable 1 calls legitimate.

        So the mirror decides. Rows exist for this log file but none are eligible
        (all of it was the daemon's own speech, or already recalled) -> we looked,
        there was nothing, mark it. No rows at all -> `daemon reindex` has work to
        do first, leave it pending.
        """
        if self._store.count_for_log_file(f"memory/log/{date}.md") == 0:
            return Result(
                date=date,
                status="empty",
                detail="not mirrored yet - run `daemon reindex`",
            )
        if date == log.local_date(clock_now()):
            # Today is still being written to; marking it done loses the evening.
            return Result(date=date, status="empty", detail="today is not finished")
        _write_artifact(path, render_artifact(date, Conclusion(), messages_read=0))
        return Result(date=date, status="nothing", detail="nothing worth recording")

    async def _apply(
        self, conclusion: Conclusion, date: str, problems: list[str]
    ) -> dict[str, int]:
        facts = entities = observations = 0

        for fact in conclusion.facts:
            try:
                await self._curated.add(
                    fact.body,
                    importance=fact.importance,
                    supersession_key=fact.key,
                    supersedes=fact.updates,
                    trigger_phrases=fact.triggers,
                    session_kind="reflection",
                )
                facts += 1
            except (OSError, sqlite3.Error, ValueError) as exc:
                # One bad fact must not cost the rest of the night's work, but it
                # must be reported - a partially applied pass that claims success
                # is the failure mode this whole file is arranged against.
                problems.append(f"could not record fact: {exc}")

        for draft in conclusion.entities:
            try:
                await self._notes.note(
                    draft.name,
                    draft.note,
                    kind=draft.kind,
                    links=draft.links,
                    # The day being reflected on, not the day of the run: a
                    # catch-up over months of history would otherwise stamp every
                    # section with today.
                    date=date,
                )
                entities += 1
            except (OSError, sqlite3.Error, UnsafeName) as exc:
                problems.append(f"could not write note for {draft.name}: {exc}")

        for observation in conclusion.observations:
            try:
                self._store.insert_observation(
                    body=observation.body,
                    observed_from=f"{date}/{date}",
                    now=clock_now(),
                    confidence=observation.confidence,
                )
                observations += 1
            except sqlite3.Error as exc:
                problems.append(f"could not record observation: {exc}")

        return {"facts": facts, "entities": entities, "observations": observations}

    async def catch_up(self, *, limit: int = 14, now: datetime | None = None) -> list[Result]:
        """Reflect on every unreflected day the log has, oldest first.

        The log clock (docs/PLAN.md 8.1) is why this exists: observations need
        weeks of accumulation before persona evolution can be judged, and the log
        already goes back further than reflection does. Bounded so a first run over
        months of history does not become one unbounded batch of model calls.
        """
        today = log.local_date(now or clock_now())
        # Today is dropped *before* the cap, not after: slicing first meant a run
        # where today was pending processed limit-1 days and silently fell one
        # behind every time.
        backlog = [date for date in self.pending_days() if date != today]
        return [await self.run(date) for date in backlog[:limit]]

    def pending_days(self) -> list[str]:
        """Days with a log file and no reflection artifact, oldest first."""
        return pending_days(self._data_dir)


def pending_days(data_dir: Path) -> list[str]:
    """Days with a log file and no reflection artifact, oldest first.

    Module-level so `daemon doctor` can report the backlog without building a
    gateway and a store it would not otherwise need: "how many days has reflection
    not caught up on" is exactly the kind of state that looks healthy from the
    outside while nothing has run for a month.
    """
    log_dir = data_dir / log.LOG_SUBDIR
    if not log_dir.exists():
        return []
    days = sorted(path.stem for path in log_dir.glob("*.md"))
    return [day for day in days if not artifact_path(data_dir, day).exists()]


def _transcript(rows: list[sqlite3.Row]) -> str:
    """The day as plain text. Roles are labelled because "who said this" is the
    whole basis for an observation about how to treat someone."""
    lines = []
    for row in rows:
        who = "나" if row["role"] == "user" else "너"
        lines.append(f"{who}: {row['content']}")
    return "\n".join(lines)
