"""Persona assembly: `persona/seed.md` (human-owned) + `persona/learned.md` (AI-owned).

This is the read half of docs/PLAN.md 5.1's file-ownership split. The write halves
live elsewhere on purpose: `seed.md` has no writer in this codebase at all
(docs/CONTRACTS.md non-negotiable 5), and `learned.md` is written only by
`daemon/persona/rules.py`. This module never writes either file.

Read on every turn, not once at startup - this is `daemon/loop.py`'s current
`_read_seed` behaviour, kept exactly: `seed.md` is edited by hand, and an edit
has to take effect on the very next message with no restart. `learned.md`
changes weekly (or on a `daemon persona forget`), so the same freshness applies
there for the same reason.

Errors are swallowed to an empty string rather than raised: a conversation turn
must not die because a persona file has a permission problem, and a missing
seed is exactly what a fresh install looks like before `daemon setup` runs.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_FILE = Path("persona") / "seed.md"
LEARNED_FILE = Path("persona") / "learned.md"

LEARNED_PREFIX = (
    "What I've worked out about dealing with you specifically, from our own "
    "conversations (not who I am - that part never changes):"
)
"""Marks the learned block as separate from the seed. The split in
docs/CONTRACTS.md non-negotiable 5 only works as an anchor if the model can tell
which part is the fixed anchor and which part is what it accumulated - otherwise
a learned rule reads with the same authority as the identity above it."""


_BULLET_RE = re.compile(r"^- (?P<body>.+)$")


def seed_path(data_dir: Path) -> Path:
    return data_dir / SEED_FILE


def learned_path(data_dir: Path) -> Path:
    return data_dir / LEARNED_FILE


async def read_file(path: Path) -> str:
    """A persona file's text, or "" if it is absent or unreadable.

    Shared by `load_persona` (seed + learned) and `daemon/persona/evolve.py`
    (seed alone, for the weekly prompt), so the one policy - missing is normal,
    unreadable is logged, neither raises - lives in one place.
    """
    try:
        return (await asyncio.to_thread(path.read_text, encoding="utf-8")).strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError):
        # UnicodeDecodeError is a ValueError, not an OSError, so it would escape a
        # bare `except OSError` and raise straight through `load_persona` - which the
        # module docstring above promises never happens. It is a live case here, not
        # a hypothetical: `seed.md` is hand-edited by the owner, and a Korean owner's
        # editor saving it in CP949/EUC-KR produces exactly this. A persona that
        # cannot be decoded is treated the same as one that cannot be read.
        logger.exception("persona: could not read %s", path)
        return ""


def rule_bodies(text: str) -> list[str]:
    """The rule bullets in `learned.md`, without the file's header.

    The two persona files are not symmetrical here. `seed.md` goes into the
    prompt whole because every line of it was written to *be* prompt - "How I
    talk", "I do not simply agree". `learned.md` opens with a notice addressed
    to the person reading the file: who owns it, that hand edits do not survive
    a rewrite, and how to ask for a rule to be dropped. Injected verbatim - as
    it was when this was first measured - that put `daemon persona forget <id>`
    in the model's system prompt and repeated `LEARNED_PREFIX`'s own sentence
    straight back at it.

    The file is AI-owned and `rules.render` writes one bullet per rule, so the
    bullets are the whole content and anything else in the file is furniture.
    """
    bodies = []
    for line in text.splitlines():
        match = _BULLET_RE.match(line.strip())
        if match:
            bodies.append(match.group("body").strip())
    return bodies


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


async def load_persona(data_dir: Path) -> str:
    """The persona system message: seed verbatim, then the learned rules under a
    header that marks them as separate from the anchor.

    Seed goes in unmodified - it is the anchor, and whatever the user wrote
    stays exactly as written. Learned rules get `LEARNED_PREFIX` so the model
    can distinguish "who I am" (fixed) from "what I've learned about dealing
    with this person" (accumulated, and the whole point of M4).

    Empty when both files are empty or absent, so the caller adds no system
    message at all rather than an empty one - matching what `_read_seed` did
    before this replaced it.
    """
    seed = await read_file(seed_path(data_dir))
    learned = await read_file(learned_path(data_dir))

    parts = []
    if seed:
        parts.append(seed)
    bodies = rule_bodies(learned)
    if bodies:
        rules = "\n".join(f"- {body}" for body in bodies)
        parts.append(f"{LEARNED_PREFIX}\n{rules}")
    return "\n\n".join(parts)
