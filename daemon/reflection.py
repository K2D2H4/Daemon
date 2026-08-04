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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import open_private_append, secure_dir
from daemon.llm.base import Message, ProviderError
from daemon.llm.gateway import LLMGateway
from daemon.memory import entities as entity_notes
from daemon.memory import log
from daemon.memory.curated import CuratedMemory
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
- entities: 사람 / 장소 / 프로젝트 / 주제. note 는 그 대상에 대해 알게 된 것
  한두 문장. links 는 함께 언급된 다른 대상의 이름.
- observations: 이 사람을 어떻게 대하면 좋은지에 대한 관찰.
  대화 내용이 아니라 대화 방식에 대한 것이다. confidence 는 0~1.

확실하지 않으면 넣지 않는다. 빈 배열도 정답이다. 설명이나 인사말 없이 JSON만.

{"facts": [{"body": "...", "importance": 5, "key": null}],
 "entities": [{"name": "...", "kind": "person", "note": "...", "links": []}],
 "observations": [{"body": "...", "confidence": 0.5}]}"""


@dataclass(frozen=True, slots=True)
class Fact:
    body: str
    importance: int = 5
    key: str | None = None


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
    `unparseable` · `unavailable` (the model could not be reached)."""
    messages_read: int = 0
    facts: int = 0
    entities: int = 0
    observations: int = 0
    detail: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"written", "skipped", "empty"}


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
            Fact(body=body, importance=_int(item, "importance", 5, 1, 10), key=_key(item))
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
        """Reflect on one local day. Safe to call twice - the second call is a
        no-op unless `force`, because the artifact is the marker."""
        path = artifact_path(self._data_dir, date)
        if path.exists() and not force:
            return Result(date=date, status="skipped", detail=f"{path.name} already exists")

        rows = self._store.messages_for_day(date)
        if not rows:
            return Result(date=date, status="empty", detail="no messages to reflect on")

        try:
            completion = await self._gateway.complete(
                Task.REFLECTION,
                [
                    Message(role="system", content=SYSTEM),
                    Message(role="user", content=_transcript(rows)),
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
        today = (now or clock_now()).astimezone().strftime("%Y-%m-%d")
        results = []
        for date in self.pending_days(now=now)[:limit]:
            if date == today:
                # Today is still being written to. Reflecting on a partial day
                # would mark it done and lose the evening.
                continue
            results.append(await self.run(date))
        return results

    def pending_days(self, *, now: datetime | None = None) -> list[str]:
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
