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
