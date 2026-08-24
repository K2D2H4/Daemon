"""`persona/learned.md` and its mirror in `persona_rules`. The only write path
for either.

Same shape as `daemon/memory/curated.py` and for the same reason
(docs/CONTRACTS.md non-negotiable 1): markdown is rewritten whole and
durably before the mirror changes, so a crash between the two leaves a
markdown file with nothing pointing at a mirror row that does not exist -
never the other way around.

## What the file deliberately does not carry

Only rule bodies. Not `created_at`, not `evidence`, not `supersession_key`, not
`status` - those live in sqlite columns because a model must not be able to
write them in prose (non-negotiable 3). A model that could write its own
`created_at` could backdate a rule to look established; one that could write
`evidence` could claim observations that never happened; one that could write
`status` could un-retire something a human asked to forget.

## The batch-collision defect this guards against

`persona_rules` has no unique index on `supersession_key` - unlike
`memory_entries`, which schema.sql gives one. Applying a batch of proposals one
at a time would let a second proposal sharing a key silently retire the first
one this same call just inserted, which is exactly the defect docs/PLAN.md
8.2.1 recorded for the curated tier ("이주 위치: 연희동" / "이전 주소: 망원동",
both keyed `location`, applied in order, the important half lost). So
`resolve_supersessions` runs before any write happens and keeps the first
proposal per key, deterministically - not "whichever the model happened to list
last".
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import write_private_replace
from daemon.memory.store import Store
from daemon.persona.loader import learned_path, read_file, rule_bodies

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 200

HEADER = """# learned

What I have worked out about dealing with you specifically - not what you
said, but how you like to be dealt with. I write this file and rewrite it
whole, at most a few times a week; anything added here by hand does not
survive the next rewrite. You cannot edit a line, but you can ask me to drop
one: `daemon persona forget <id> --why "..."`. `persona/seed.md` is the file I
never touch.
"""


@dataclass(frozen=True, slots=True)
class Proposal:
    body: str
    evidence: tuple[int, ...]
    """Observation ids this rule was derived from - the diary's citation and
    what gets marked consumed once the rule lands."""
    supersession_key: str | None = None


def render(bodies: list[str]) -> str:
    """The whole file. One rule = one bullet; multi-line bodies are folded to
    one line, same reasoning as `memory.curated.render` - a body containing a
    newline would parse back as two rules, and a rule is a sentence, not a
    document. Also re-clamps to `MAX_BODY_CHARS`: this is the one function
    every write goes through, so it is the actual enforcement point regardless
    of whether a caller already truncated."""
    lines = [HEADER]
    for body in bodies:
        folded = _one_line(body)[:MAX_BODY_CHARS]
        if folded:
            lines.append(f"- {folded}")
    return "\n".join(lines) + "\n"


def _one_line(body: str) -> str:
    return " ".join(body.split())


def resolve_supersessions(
    proposals: list[Proposal],
) -> tuple[list[Proposal], list[Proposal]]:
    """Split one batch into (kept, discarded) wherever two proposals share a
    supersession key.

    Must run before anything is written - see the module docstring. The one
    kept is the first to appear in `proposals`, not the last: a deterministic
    tie-break instead of "whichever got applied last", which is the failure
    mode being avoided.
    """
    kept: list[Proposal] = []
    discarded: list[Proposal] = []
    seen: set[str] = set()
    for proposal in proposals:
        key = proposal.supersession_key
        if key is not None and key in seen:
            discarded.append(proposal)
            continue
        if key is not None:
            seen.add(key)
        kept.append(proposal)
    return kept, discarded


class LearnedFileDiverged(RuntimeError):
    """`learned.md` holds one or more rule bodies with no active row in the
    mirror.

    Two ways this happens, both legitimate states rather than corruption: the
    sqlite file was deleted (docs/CONTRACTS.md non-negotiable 1 requires that
    to lose no user data, and `daemon reindex` is the documented recovery), or
    the process crashed between `add()`'s markdown write and its mirror
    commit. Either way, `LearnedRules.add()` computes the *whole* new file
    from the mirror's active rows - so writing here would silently render a
    file without the orphaned bullet, which is exactly the data loss this
    guards against. Nothing is written when this is raised: no markdown, no
    mirror row, no observation consumed. `daemon reindex` repairs it
    (additively - see `rebuild` below) and the caller can then retry.
    """

    def __init__(self, orphaned_bodies: list[str]) -> None:
        self.orphaned_bodies = list(orphaned_bodies)
        preview = "; ".join(self.orphaned_bodies[:3])
        more = f" (+{len(self.orphaned_bodies) - 3} more)" if len(self.orphaned_bodies) > 3 else ""
        super().__init__(
            f"learned.md has {len(self.orphaned_bodies)} rule(s) the mirror does not "
            f"know about: {preview}{more} - run `daemon reindex` first"
        )


def diverged_bodies(file_bodies: list[str], active_bodies: Iterable[str]) -> list[str]:
    """Bodies in `learned.md` that no active mirror row accounts for.

    Shared by `LearnedRules.add` (which refuses to write when this is
    non-empty) and `daemon doctor` / `daemon persona`'s reporting, so what
    counts as diverged is defined in exactly one place.
    """
    known = set(active_bodies)
    return [body for body in file_bodies if body not in known]


class LearnedRules:
    """Reads and writes `persona/learned.md` and its mirror. Owns the write
    order."""

    def __init__(self, data_dir: Path, store: Store) -> None:
        self._data_dir = data_dir
        self._store = store

    def active(self) -> list[sqlite3.Row]:
        """The active rules, oldest first - what gets injected and what
        `daemon persona` shows."""
        return self._store.active_persona_rules()

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

    async def add(
        self, proposals: list[Proposal], *, now: datetime | None = None
    ) -> list[int]:
        """Add one batch of rules, returning the new mirror ids in the same
        order as the (possibly narrowed) input.

        Order is markdown, then mirror, then observation consumption - the
        same three steps `docs/design/2026-08-05-m4-persona-design.md`
        specifies and for the reason `memory.curated.CuratedMemory.add`
        documents in full: unlike that method, `persona_rules` carries no
        unique index to make a single insert atomic with its own retire, so
        the whole batch's *file content* is computed in Python first (current
        active bodies, minus whatever this batch supersedes, plus the new
        ones) and written durably before any row in the mirror changes. A
        crash before the mirror writes leaves a markdown file describing rules
        the mirror does not have yet - recoverable by re-running the pass,
        since the diary marker was not written either. A crash after would
        leave a mirror row with nothing in the file, which non-negotiable 1
        forbids.

        `resolve_supersessions` runs here too (not only in the caller) so this
        method is safe to call directly with a colliding batch, not only
        through `daemon/persona/evolve.py`'s already-narrowed input.

        Raises `LearnedFileDiverged` - and writes nothing at all - if
        `learned.md` already holds a bullet the mirror has no active row for.
        """
        if not proposals:
            return []
        moment = now or clock_now()

        winners, _ = resolve_supersessions(proposals)
        winners = [
            Proposal(
                body=_one_line(p.body)[:MAX_BODY_CHARS],
                evidence=p.evidence,
                supersession_key=p.supersession_key,
            )
            for p in winners
        ]
        winners = [p for p in winners if p.body]
        if not winners:
            return []

        active_rows = self._store.active_persona_rules()

        # Refuse rather than rewrite if the file already disagrees with the
        # mirror - the file content below is computed *from the mirror*, so
        # writing it now would silently drop every bullet the mirror does not
        # know about (docs/CONTRACTS.md non-negotiable 1). See
        # `LearnedFileDiverged`.
        current_text = await read_file(learned_path(self._data_dir))
        orphaned = diverged_bodies(
            rule_bodies(current_text), (row["body"] for row in active_rows)
        )
        if orphaned:
            raise LearnedFileDiverged(orphaned)

        active_id_by_key = {
            row["supersession_key"]: int(row["id"])
            for row in active_rows
            if row["supersession_key"]
        }
        keys_touched = {p.supersession_key for p in winners if p.supersession_key}
        retiring = {key: active_id_by_key[key] for key in keys_touched if key in active_id_by_key}

        kept_bodies = [
            row["body"] for row in active_rows if int(row["id"]) not in retiring.values()
        ]
        text = render(kept_bodies + [p.body for p in winners])

        # Durable before anything below changes - docs/CONTRACTS.md non-negotiable 1.
        await asyncio.to_thread(write_private_replace, learned_path(self._data_dir), text)

        for key, old_id in retiring.items():
            self._store.retire_persona_rule(
                old_id, when=moment, why=f"superseded by a new rule keyed {key!r}"
            )

        new_ids: list[int] = []
        for proposal in winners:
            new_id = self._store.insert_persona_rule(
                body=proposal.body,
                created_at=moment,
                evidence=proposal.evidence,
                supersession_key=proposal.supersession_key,
            )
            new_ids.append(new_id)
            self._store.consume_observations(proposal.evidence, new_id)
        return new_ids

    async def retire(self, rule_id: int, *, why: str, now: datetime | None = None) -> bool:
        """A human's deletion request. False if `rule_id` does not exist or is
        already retired; raises `LearnedFileDiverged` and nothing else.

        Rewrites `learned.md` without this rule's body first, same ordering as
        `add`, then flips the mirror row. Does **not** touch the observations
        this rule consumed - `consumed_by` only ever moves forward
        (docs/CONTRACTS.md non-negotiable 6), and reverting it would let next
        week's pass revive the very rule a human just asked to forget, from
        the same evidence. Respecting a delete request means the evidence
        stays spent, not that it becomes available again.

        The divergence check is here for the same reason it is in `add`, and it
        was missing here first: this rewrite is also computed from the mirror, so
        forgetting one rule on a diverged file would take every orphaned bullet
        with it - a deletion request costing the user rules they never named.
        """
        active_rows = self._store.active_persona_rules()
        target = next((row for row in active_rows if int(row["id"]) == rule_id), None)
        if target is None:
            return False

        current_text = await read_file(learned_path(self._data_dir))
        orphaned = diverged_bodies(
            rule_bodies(current_text), (row["body"] for row in active_rows)
        )
        if orphaned:
            raise LearnedFileDiverged(orphaned)

        moment = now or clock_now()
        remaining = [row["body"] for row in active_rows if int(row["id"]) != rule_id]
        text = render(remaining)
        await asyncio.to_thread(write_private_replace, learned_path(self._data_dir), text)

        return self._store.retire_persona_rule(rule_id, when=moment, why=why)


# --- rebuild ------------------------------------------------------------


def rebuild(data_dir: Path, store: Store) -> int:
    """Restore `persona_rules` from `learned.md`, returning how many landed.

    Called by `daemon reindex`. Same shape as `memory.curated.rebuild` and
    `memory.entities.rebuild`, with one difference that is the whole point of
    this function: it is **additive only**. A body already represented by an
    active row is left alone; a body the mirror has no active row for gets a
    new row with `evidence='[]'` and no supersession key - the same "rebuilt
    rows carry defaults, not what evolve.py concluded" trade the other two
    tiers make. What it never does is update or delete an existing row:
    `observations.consumed_by` is a foreign key onto `persona_rules(id)`, so
    deleting a row here would either fail outright or orphan real evidence a
    previous week's pass recorded. Running this twice restores nothing the
    second time, because by then every file body already has a matching
    active row.
    """
    path = learned_path(data_dir)
    if not path.exists():
        return 0
    bodies = rule_bodies(path.read_text(encoding="utf-8"))
    if not bodies:
        return 0

    active = {row["body"] for row in store.active_persona_rules()}
    moment = clock_now()
    restored = 0
    for body in bodies:
        if body in active:
            continue
        store.insert_persona_rule(body=body, created_at=moment, evidence=())
        restored += 1
    if restored:
        logger.warning(
            "restored %d persona rule(s) from learned.md; their evidence and "
            "supersession key are defaults, not what evolve.py chose - a rule "
            "that used to cite specific observations no longer does",
            restored,
        )
    return restored
