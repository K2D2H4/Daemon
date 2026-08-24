"""The weekly persona-evolution pass - M4.

Turns accumulated `observations` (append-only, written by `daemon/reflection.py`)
into `persona_rules`, mirrored into `persona/learned.md`
(`daemon/persona/rules.py`). One model call at most, and often none at all: the
first four gates in `run()` are deterministic and cost nothing, because a
scheduler tick happens whether or not there is anything to evolve, and this
pass is meant to be a bounded weekly cost, not a per-tick one.

## Everything here treats the model's output as hostile

Same stance as `daemon/reflection.py`, for the same reason: this output can
retire a rule, invent a supersession key, or claim evidence that was never
given to it. So `_clean` clamps every body, narrows every key to a fixed
charset, and drops any claimed evidence id that was not actually in the
unconsumed set handed to the prompt. An unparseable reply writes nothing at
all - no diary, no rule - rather than writing what it managed; the next
attempt (a later day this week, or `daemon persona evolve --force`) gets a
clean try instead of a half-applied one.

## The diary is the idempotency marker

Same shape as `memory/reflections/YYYY-MM-DD.md`: if this week's diary file
exists, the week is done, because `daemon/memory/schema.sql` is frozen and "did
this week evolve" is state that belongs on the markdown side of the contract
(docs/CONTRACTS.md non-negotiable 1). The file is dated by the Monday of the
local week containing `now`, not by the day `run()` happens to execute on - so
a scheduled Monday run and a `daemon persona evolve` run by hand on Wednesday
agree on which week they are talking about, and the check is a single
existence test rather than a date-range scan.

Only a *complete* pass writes the diary. A gate skipping before the model is
ever called, a model that could not be reached, and a reply that did not parse
all leave no diary - each is retryable later in the same week, because nothing
was concluded.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import open_private_append, secure_dir
from daemon.llm.base import Message, ProviderError
from daemon.llm.gateway import LLMGateway
from daemon.memory.store import Store
from daemon.persona.loader import learned_path, read_file, rule_bodies, seed_path
from daemon.persona.rules import (
    MAX_BODY_CHARS,
    LearnedFileDiverged,
    LearnedRules,
    Proposal,
    diverged_bodies,
    resolve_supersessions,
)
from daemon.reflection import extract_json
from daemon.tasks import Task

logger = logging.getLogger(__name__)

OBSERVATION_BUDGET = 60
"""Max observations put in the prompt. Not the same as `min_observations` (the
gate): this bounds the *ceiling*, so a long-neglected backlog does not grow the
prompt without limit."""

DIARY_SUBDIR = Path("persona") / "diary"

_KEY_RE = re.compile(r"[^a-z0-9_]+")
MAX_KEY_CHARS = 40

SYSTEM = """너는 이 사람을 어떻게 대하면 좋은지에 대한 관찰을 규칙으로 정리하는 역할이다.
아래 규칙을 지켜 JSON만 출력한다.

입력은 이 순서로 온다: 내 정체성(고정, 절대 바꿀 수 없다), 지금 이미 활성인 규칙들
(이미 반영되어 있으니 같은 내용을 다시 만들지 않는다), 아직 규칙으로 정리되지 않은
관찰들(각 줄 앞에 (id=번호)가 붙어 있다).

- rules: 여러 관찰을 관통하는 패턴이 보일 때만 규칙을 만든다. 관찰 하나만 보고
  규칙을 만들지 않는다. body 는 그 무렵 이 사람이 어떠했는지에 대한 **관찰**을
  짧게 한 문장으로 적는다. 늘 그렇다는 단정이나 상시 요구로 쓰지 않는다 -
  '~를 요구한다', '~를 중시한다' 처럼 언제나 참인 성질처럼 적지 말고,
  '~한 편이었다', '~를 선호했다' 처럼 그때 그렇게 보였다고 적는다.
  날짜와 관찰 수는 시스템이 붙이므로 body 에 쓰지 않는다.
  evidence 는 이 규칙의 근거가 된 관찰의 id 목록(정수) - 입력에
  실제로 있던 id만 쓴다. key 는 이 규칙이 나중에 바뀔 수 있는 것이면 넣는다
  (예: "greeting_style"). 같은 key 는 이전 규칙을 대체한다.

확신이 없으면 넣지 않는다. 빈 배열도 정답이다. 설명이나 인사말 없이 JSON만 출력한다.

{"rules": [{"body": "...", "evidence": [1, 2], "key": null}]}"""


@dataclass(frozen=True)
class EvolutionResult:
    """What one call to `run()` did. Every field is reported rather than
    logged, for the same reason `reflection.Result` is: a pass that silently
    did nothing looks exactly like one that worked."""

    date: str
    """The Monday of the week this pass belongs to, `YYYY-MM-DD`."""
    observations_read: int
    proposed: int
    added: int
    retired: int
    skipped: str
    """Non-empty only for the three deterministic gates in `run()` - empty
    means the pass actually ran (docs/design/2026-08-05-m4-persona-design.md)."""
    problems: tuple[str, ...]


class PersonaEvolution:
    """One weekly pass. Constructed per run; holds no state between them."""

    def __init__(
        self,
        data_dir: Path,
        store: Store,
        gateway: LLMGateway,
        *,
        max_active: int = 20,
        max_new: int = 3,
        min_observations: int = 5,
        rules: LearnedRules | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._store = store
        self._gateway = gateway
        self._max_active = max_active
        self._max_new = max_new
        self._min_observations = min_observations
        self._rules = rules or LearnedRules(data_dir, store)

    def diary_path(self, date: str) -> Path:
        return self._data_dir / DIARY_SUBDIR / f"{date}.md"

    async def run(self, *, now: datetime | None = None, force: bool = False) -> EvolutionResult:
        moment = now or clock_now()
        week = _week_start(moment)
        path = self.diary_path(week)

        # --- gate 1: already run this week? zero model calls -----------------
        if path.exists() and not force:
            return EvolutionResult(
                date=week,
                observations_read=0,
                proposed=0,
                added=0,
                retired=0,
                skipped="already run this week",
                problems=(),
            )

        # --- gate 2: enough observations to justify a call? ------------------
        unconsumed = self._store.unconsumed_observations(limit=OBSERVATION_BUDGET)
        if len(unconsumed) < self._min_observations:
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=0,
                added=0,
                retired=0,
                skipped=f"not enough observations ({len(unconsumed)}<{self._min_observations})",
                problems=(),
            )

        # --- gate 3: room in the active-rule budget? --------------------------
        active_rows = self._store.active_persona_rules()
        active_count = len(active_rows)
        if active_count >= self._max_active:
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=0,
                added=0,
                retired=0,
                skipped=f"rule budget full ({active_count}/{self._max_active})",
                problems=(),
            )

        # --- gate 4: has learned.md diverged from the mirror? zero model calls -
        # Checked before the model is ever called - not only as a backstop
        # around `self._rules.add` below - because a reply proposing zero new
        # rules would otherwise skip past `add()` entirely (see the `if
        # winners` guard) and write a diary anyway, marking a diverged week
        # done. `daemon reindex` is additive-only and does not delete or
        # update, so it is always safe to run and this is always the fix.
        learned_text = await read_file(learned_path(self._data_dir))
        orphaned = diverged_bodies(
            rule_bodies(learned_text), (row["body"] for row in active_rows)
        )
        if orphaned:
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=0,
                added=0,
                retired=0,
                # Named in `skipped`, not only in `problems`: this is a gate, and
                # with it empty the CLI's headline read "ran (8 read -> 0
                # proposed)" - indistinguishable from a pass that reached the
                # model and concluded there was nothing to add.
                skipped=f"learned.md has {len(orphaned)} rule(s) the mirror does not know about",
                problems=(
                    f"learned.md has {len(orphaned)} rule(s) the mirror does not know "
                    "about - run `daemon reindex` to repair, then this week can run again",
                ),
            )

        # --- the one model call ------------------------------------------------
        seed = await read_file(seed_path(self._data_dir))
        active_bodies = [row["body"] for row in active_rows]
        prompt = _context(seed, active_bodies, unconsumed)

        try:
            completion = await self._gateway.complete(
                Task.PERSONA_RULE,
                [Message(role="system", content=SYSTEM), Message(role="user", content=prompt)],
            )
        except ProviderError as exc:
            logger.warning("persona evolve: model unavailable (%s)", exc)
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=0,
                added=0,
                retired=0,
                skipped="",
                problems=(f"model unavailable: {exc}",),
            )

        raw = extract_json(completion.text)
        if raw is None:
            logger.warning("persona evolve: %s produced no JSON", completion.model)
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=0,
                added=0,
                retired=0,
                skipped="",
                problems=(f"{completion.model} did not return a JSON object",),
            )

        unconsumed_ids = {int(row["id"]) for row in unconsumed}
        proposals, problems = _clean(raw, unconsumed_ids)
        proposed = len(proposals)

        # Rate limit: at most `max_new` this cycle, and never past `max_active`.
        # Gate 3 already guarantees active_count < max_active, so room >= 1.
        room = max(0, self._max_active - active_count)
        capped = proposals[: min(self._max_new, room)]
        if len(proposals) > len(capped):
            problems.append(
                f"{len(proposals) - len(capped)} proposal(s) dropped: over this "
                f"cycle's cap (max_new={self._max_new}, room={room})"
            )

        existing_bodies = {row["body"] for row in active_rows}
        deduped = []
        for proposal in capped:
            if proposal.body in existing_bodies:
                problems.append(f"proposal duplicates an existing active rule: {proposal.body!r}")
                continue
            deduped.append(proposal)

        # Resolved here too (not only inside LearnedRules.add) so the discard is
        # visible in *this* result, and so `winners` lines up 1:1 with the ids
        # `add()` returns.
        winners, discarded = resolve_supersessions(deduped)
        for proposal in discarded:
            problems.append(
                f"two proposals keyed {proposal.supersession_key!r} in one batch; "
                "kept the earlier one"
            )

        # A model can cite the same observation id from two different
        # proposals in one reply; only the first should keep it as evidence.
        winners, evidence_problems = _claim_evidence_once(winners)
        problems.extend(evidence_problems)

        try:
            added_ids = await self._rules.add(winners, now=moment) if winners else []
        except LearnedFileDiverged as exc:
            # Gate 4 already checks this before the model is ever called, so
            # reaching this is a race rather than the normal path - but
            # `add()` is the one method that can raise it, and a pass that
            # cannot write must not look like a pass that decided there was
            # nothing to add. No diary either way: the week has to retry
            # after `daemon reindex`, not be marked done.
            problems.append(str(exc))
            return EvolutionResult(
                date=week,
                observations_read=len(unconsumed),
                proposed=proposed,
                added=0,
                retired=0,
                skipped="",
                problems=tuple(problems),
            )
        active_after = self._store.count_active_persona_rules()
        retired = active_count + len(added_ids) - active_after

        observations_by_id = {int(row["id"]): row["body"] for row in unconsumed}
        added_detail = [
            (
                proposal.body,
                tuple(observations_by_id.get(i, f"#{i}") for i in proposal.evidence),
            )
            for proposal in winners
        ]
        retired_keys = sorted(
            {p.supersession_key for p in winners if p.supersession_key}
            & {row["supersession_key"] for row in active_rows if row["supersession_key"]}
        )

        diary = render_diary(
            week, added=added_detail, retired_keys=retired_keys, problems=tuple(problems)
        )
        _write_diary(path, diary)

        return EvolutionResult(
            date=week,
            observations_read=len(unconsumed),
            proposed=proposed,
            added=len(added_ids),
            retired=retired,
            skipped="",
            problems=tuple(problems),
        )


# --- the prompt --------------------------------------------------------------


def _context(seed: str, active_bodies: list[str], observations: list[sqlite3.Row]) -> str:
    lines = ["# 정체성 (고정, 앵커)", seed or "(비어 있음)", "", "# 지금 활성인 규칙"]
    lines += [f"- {body}" for body in active_bodies] if active_bodies else ["(아직 없음)"]
    lines += ["", "# 아직 규칙으로 정리되지 않은 관찰"]
    lines += [f"- (id={row['id']}) {row['body']}" for row in observations]
    return "\n".join(lines)


# --- parsing -------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _narrow_key(raw: object) -> str | None:
    """A supersession key, narrowed to `[a-z0-9_]`, max `MAX_KEY_CHARS` -
    same reasoning as `reflection._key`: the model chooses this and it
    *retires* an existing rule, so it must not carry punctuation or whitespace
    that would make two spellings of the same key look like two different
    ones."""
    if not isinstance(raw, str):
        return None
    text = _one_line(raw).lower()
    if not text:
        return None
    narrowed = _KEY_RE.sub("_", text).strip("_")
    return narrowed[:MAX_KEY_CHARS] or None


def _clean(raw: dict[str, object], unconsumed_ids: set[int]) -> tuple[list[Proposal], list[str]]:
    """Turn the model's object into proposals, dropping what cannot be used.

    Returns the problems alongside rather than raising - one malformed entry
    must not cost the rest of the batch, but it must not be invisible either.
    """
    problems: list[str] = []
    value = raw.get("rules")
    if value is None:
        return [], problems
    if not isinstance(value, list):
        problems.append("rules was not a list")
        return [], problems

    proposals: list[Proposal] = []
    for item in value:
        if not isinstance(item, dict):
            problems.append("an entry in rules was not an object")
            continue

        body_raw = item.get("body")
        body = _one_line(body_raw)[:MAX_BODY_CHARS] if isinstance(body_raw, str) else ""
        if not body:
            problems.append("a rule with no body")
            continue

        evidence_raw = item.get("evidence")
        evidence: tuple[int, ...] = ()
        if isinstance(evidence_raw, list):
            ids = []
            fabricated = []
            for entry in evidence_raw:
                try:
                    as_int = int(entry)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if as_int in unconsumed_ids:
                    ids.append(as_int)
                else:
                    fabricated.append(as_int)
            evidence = tuple(dict.fromkeys(ids))
            if fabricated:
                # The model cited an id that was never in the prompt - either
                # made up, or already consumed by an earlier rule. Dropped
                # either way, but silently would hide a model inventing its
                # own evidence.
                problems.append(
                    f"evidence id(s) {fabricated} for {body!r} were not offered in "
                    "the prompt; dropped"
                )
        elif evidence_raw is not None:
            problems.append("evidence was not a list")

        proposals.append(
            Proposal(body=body, evidence=evidence, supersession_key=_narrow_key(item.get("key")))
        )
    return proposals, problems


def _claim_evidence_once(proposals: list[Proposal]) -> tuple[list[Proposal], list[str]]:
    """Within one batch, an observation id may back only the first proposal
    that cites it.

    `Store.consume_observations`'s `consumed_by IS NULL` guard already means
    only the first `add()` call for a given id actually claims it - but every
    proposal's own `evidence` column would otherwise still list the id
    regardless, so a later proposal's rule would claim an observation that
    `daemon persona`'s "N observation(s)" count, and the mirror's own
    `consumed_by`, both say belongs to an earlier one. The first proposal to
    cite an id keeps it; later ones lose it from their own evidence, and the
    drop is reported.
    """
    problems: list[str] = []
    claimed: set[int] = set()
    result: list[Proposal] = []
    for proposal in proposals:
        kept = []
        dropped = []
        for observation_id in proposal.evidence:
            if observation_id in claimed:
                dropped.append(observation_id)
                continue
            claimed.add(observation_id)
            kept.append(observation_id)
        if dropped:
            problems.append(
                f"observation id(s) {dropped} already claimed by an earlier proposal "
                f"in this batch; dropped from {proposal.body!r}"
            )
            proposal = Proposal(
                body=proposal.body,
                evidence=tuple(kept),
                supersession_key=proposal.supersession_key,
            )
        result.append(proposal)
    return result, problems


# --- the diary -----------------------------------------------------------


def render_diary(
    week: str,
    *,
    added: list[tuple[str, tuple[str, ...]]],
    retired_keys: list[str],
    problems: tuple[str, ...],
) -> str:
    lines = [f"# {week} 페르소나 진화", ""]
    if added:
        lines += ["## 추가된 규칙", ""]
        for body, evidence in added:
            lines.append(f"- {body}")
            for observation in evidence:
                lines.append(f"  - 근거: {observation}")
        lines.append("")
    else:
        lines += ["이번 주에 추가된 규칙은 없다.", ""]
    if retired_keys:
        lines += ["## 은퇴된 규칙", ""]
        for key in retired_keys:
            lines.append(f"- key: {key}")
        lines.append("")
    if problems:
        lines += ["## 문제", ""]
        for problem in problems:
            lines.append(f"- {problem}")
        lines.append("")
    return "\n".join(lines)


def _write_diary(path: Path, text: str) -> None:
    secure_dir(path.parent)
    with open_private_append(path) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


# --- the week ---------------------------------------------------------------


def _week_start(moment: datetime) -> str:
    """The Monday of the local week containing `moment`, as `YYYY-MM-DD`.

    One diary file per week regardless of which day `run()` actually executes
    on - a scheduled Monday job and a `daemon persona evolve` run by hand on a
    Wednesday to check the pass must agree on which week's marker they are
    reading, or the idempotency check in `run()` would not be a simple
    existence test.
    """
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    local = aware.astimezone()
    monday = local.date() - timedelta(days=local.weekday())
    return monday.isoformat()
