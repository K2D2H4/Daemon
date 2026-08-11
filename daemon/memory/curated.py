"""The curated tier: `memory/core.md` and its mirror in `memory_entries`.

Layer 2 of the five in docs/PLAN.md 4.1 - small, and **always injected**, which is
what separates it from the episodic log. The log is large and searched; this is
tiny and unconditional. Mixing them blows the context window or makes recall
meaningless, so the boundary is the point.

Markdown first, then the mirror, same as `log.append` and for the same reason
(docs/CONTRACTS.md non-negotiable 1). The difference is that this file is
*rewritten whole* rather than appended to, so the write has to be atomic and
durable - see `fs.write_private_replace`.

## What the file deliberately does not carry

Only the bodies. Not `importance`, not `supersession_key`, not
`trigger_phrases`. Those live in sqlite columns because a model must not be able
to write them in prose (non-negotiable 3), and the two that matter are worth
spelling out:

  * `importance` multiplies the recall score. In the file, a model could inflate
    its own conclusions until they crowd out everything else.
  * `supersession_key` *retires* an existing fact. In the file, a model could
    silently delete something the user said by claiming the same key.

The cost is that a rebuild from markdown restores bodies with defaults and
`origin='system'` - the same trade the message log already makes, and the reason
rebuilt rows are distinguishable at all. Since reflection started producing trigger
phrases that trade has a behavioural price: a fact that used to surface on the word
"이사" stops doing so after a `daemon reindex`, and only the next reflection pass over
that day puts it back. Recorded rather than fixed, because the alternative is
letting a model write its own recall multipliers into prose.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import write_private_replace
from daemon.memory.store import Store

logger = logging.getLogger(__name__)

CORE_FILE = Path("memory") / "core.md"

HEADER = """# core

What I know about you that is worth always remembering. Written by the nightly
reflection pass and rewritten whole, so notes added here by hand do not survive -
the file this reads from is `memory/log/`, and the one you own is
`persona/seed.md`.
"""

_BULLET_RE = re.compile(r"^- (?P<body>.+)$")

MAX_INJECTED = 50
"""Injection budget, in entries. The tier is unconditional, so this is the only
thing standing between it and the context window."""

ALL_ACTIVE = 10_000
"""What the *file* holds: every active fact, which is not the same set as what gets
injected. Rendering `core.md` from `MAX_INJECTED` conflated the two, and since the
file is written as a whole replace, the 51st fact was dropped from the source of
truth while staying active in the mirror - unnoticeable to `rebuild`, which only
adds what the markdown already has. A ceiling rather than no limit, so a runaway
mirror cannot render an unbounded file; the same number `rebuild` reads with."""


def core_path(data_dir: Path) -> Path:
    return data_dir / CORE_FILE


# --- rendering / parsing ----------------------------------------------------


def render(bodies: list[str]) -> str:
    """The whole file. Multi-line bodies are folded to one line: the format is one
    bullet per entry, so a body containing a newline would parse back as two
    entries - and a curated fact is a sentence, not a document."""
    lines = [HEADER]
    for body in bodies:
        lines.append(f"- {_one_line(body)}")
    return "\n".join(lines) + "\n"


def parse(text: str) -> list[str]:
    """Bodies, in file order. Anything that is not a bullet is ignored rather than
    fatal, for the same reason `log.parse` is forgiving: this is the source of
    truth, so a hand-mangled file must degrade to fewer entries, never to an
    exception that fails every future reflection."""
    bodies = []
    for line in text.split("\n"):
        match = _BULLET_RE.match(line.rstrip())
        if match:
            bodies.append(match["body"].strip())
    return bodies


def read(data_dir: Path) -> list[str]:
    path = core_path(data_dir)
    if not path.exists():
        return []
    return parse(path.read_bytes().decode("utf-8", errors="replace"))


def _one_line(body: str) -> str:
    return " ".join(body.split())


# --- writing ----------------------------------------------------------------


class CuratedMemory:
    """Reads and writes the curated tier. Owns the write order."""

    def __init__(self, data_dir: Path, store: Store) -> None:
        self._data_dir = data_dir
        self._store = store

    def entries(self, limit: int = MAX_INJECTED) -> list[str]:
        """The bodies to inject, most important first."""
        return [row["body"] for row in self._store.active_entries(limit)]

    async def add(
        self,
        body: str,
        *,
        importance: int = 5,
        trigger_phrases: tuple[str, ...] = (),
        supersession_key: str | None = None,
        supersedes: int | None = None,
        origin: str = "agent",
        session_kind: str = "reflection",
        modality: str = "text",
        now: datetime | None = None,
    ) -> int:
        """Record one curated fact: markdown first, then the mirror. Returns the
        mirror's row id.

        The ordering here is the contract (docs/CONTRACTS.md non-negotiable 1) and
        it needs three steps rather than two, because the file is a rewrite of the
        whole active set and only the mirror knows what this fact *retires*:

          1. run the retire-and-insert without committing. This connection now
             sees its own uncommitted rows, so the file can be rendered from the
             post-insert ordering instead of that ordering being reimplemented
             here and drifting.
          2. write and fsync the markdown.
          3. commit the mirror.

        A failure at 2 rolls step 1 back, so neither side moved. A failure at 3
        leaves the fact in the file and not in the mirror, which is the direction
        the contract prefers: `rebuild()` puts it back, whereas the reverse is
        unrecoverable.
        """
        body = _one_line(body)
        if not body:
            raise ValueError("a curated entry cannot be empty")

        entry_id = self._store.insert_entry(
            body=body,
            importance=importance,
            trigger_phrases=trigger_phrases,
            origin=origin,
            session_kind=session_kind,
            modality=modality,
            now=now or clock_now(),
            supersession_key=supersession_key,
            supersedes=supersedes,
            commit=False,
        )
        try:
            # Rendered on this thread because a sqlite connection belongs to the
            # thread that made it (see store.Store); only the write, which fsyncs
            # twice, is worth handing to the executor.
            #
            # Inside the guard, not before it: this read runs on the connection
            # that is already mid-transaction, so a failure here leaves the
            # retire-and-insert open. `reflection.Reflection.run` commits this same
            # connection to record the run, which would make that durable with no
            # markdown behind it - the unrecoverable direction.
            text = render(self.entries(ALL_ACTIVE))
            await asyncio.to_thread(write_private_replace, core_path(self._data_dir), text)
        except BaseException:
            # BaseException because the await above is a cancellation point, and a
            # daemon shutting down mid-pass must not leave the transaction open
            # either - `app.py` cancels the reflection catch-up task on every
            # lifespan exit.
            self._store.conn.rollback()
            raise
        self._store.conn.commit()
        return entry_id


# --- rebuild ----------------------------------------------------------------


def rebuild(data_dir: Path, store: Store) -> int:
    """Restore `memory_entries` from `core.md`, returning how many landed.

    Called by `daemon reindex`. Only fills a gap: an entry whose body is already
    active is left alone, so running it twice does nothing the second time. There
    is no content key to dedupe on beyond the body itself, which is correct here
    in a way it would not be for messages - two identical curated facts *are* one
    fact, whereas the same message twice is two messages.

    Provenance cannot come back (see the module docstring), so restored rows are
    `origin='system'` with default importance. That is deliberately visible:
    reflection can tell a fact it concluded from one that a rebuild guessed at.
    """
    bodies = read(data_dir)
    if not bodies:
        return 0
    active = {row["body"] for row in store.active_entries(limit=10_000)}
    moment = clock_now()
    restored = 0
    for body in bodies:
        if body in active:
            continue
        store.insert_entry(
            body=body,
            importance=5,
            trigger_phrases=(),
            origin="system",
            session_kind="reflection",
            modality="text",
            now=moment,
        )
        restored += 1
    if restored:
        logger.warning(
            "restored %d curated entry(ies) from core.md; their importance, "
            "supersession keys and trigger phrases are defaults, not what "
            "reflection chose - a trigger that used to pull a fact forward is gone",
            restored,
        )
    return restored
