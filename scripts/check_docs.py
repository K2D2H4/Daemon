#!/usr/bin/env python3
"""Verify that every path our documentation points at actually exists.

Written because README pointed at `persona/seed.md` and `memory/log/` while the
real files live under `data/`, and line 111 of the same file got it right - so the
document disagreed with itself, and anyone following it, person or agent, looked
in a directory that was never there. A stale reference is worse than a missing
one: a missing one makes you ask, a wrong one makes you confident.

Run it directly, or let CI run it:

    python3 scripts/check_docs.py

Exits non-zero on the first broken reference, listing all of them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = (
    "README.md",
    "docs/PLAN.md",
    "docs/CONTRACTS.md",
    "docs/ARCHITECTURE.md",
    "CLAUDE.md",
)
"""Checked in full. Anything else is prose we do not treat as navigation.

ARCHITECTURE.md was missing from this list for its whole existence, which is the
funnier version of the bug this script exists for: the document whose entire job is
to say where things live was the one document nobody verified.
"""

EXTRA_GLOBS = ("**/CLAUDE.md", "**/AGENTS.md", "docs/adr/*.md")

SKIP_DIRS = {".venv", "venv", ".git", "node_modules", ".claude"}
"""Directories the recursive globs must not descend into.

`.claude` is the one that bit. Agent sessions create git worktrees under
`.claude/worktrees/`, each a full copy of the repo, so `**/CLAUDE.md` started
matching a second and third set of module docs: the count went from 131 paths to
209 with nothing in the working tree changed. Two ways that is wrong - the number
stops meaning anything, and a stale doc in an abandoned worktree can fail this
check for the main tree, where the reference it names is fine.

A checker whose result depends on whether someone happens to have a worktree open
is the thing this script exists to prevent, one level up."""

BACKTICKED = re.compile(r"`([^`\n]+)`")

# A backticked span is only a path claim if it looks like one. Commands, flags,
# code fragments and prose all end up in backticks too, and treating those as
# paths produces noise that gets the whole check switched off.
PATHY = re.compile(r"^[\w./-]+$")
SUFFIXES = {
    ".py", ".md", ".sql", ".toml", ".yaml", ".yml", ".json", ".txt", ".example", ".sh",
}

CITED = {
    # Other projects, named in docs/PLAN.md so the design can say what it borrowed
    # and from where. Listed explicitly rather than guessed at: a checker that
    # tries to infer "is this ours?" produces noise, and a noisy checker is a
    # checker somebody turns off. Adding a line here is a deliberate act.
    "run_agent.py", "cron/scheduler.py", "tools/registry.py", "jobs.json",
    "plugins/memory/memory_provider.py",              # Hermes Agent
    "SOUL.md", "MEMORY.md", "USER.md",                # OpenClaw
    "backend/app/services/llm_port.py",               # ReadyTalk-Onpremis
    "models/",                                        # Gemini model id prefix
    "seed.md", "learned.md",                          # discussed by bare name
}

PLACEHOLDERS = ("YYYY-MM-DD", "{", "<", "*", "…")
"""`memory/log/YYYY-MM-DD.md` is a shape, not a file. The directory it lives in
is still checked, because that is the part someone would `cd` into."""

RUNTIME = ("data/", ".env")
"""Exists only after the daemon runs, or only on this machine.

Deliberately *not* including bare `persona/` or `memory/`: those are the exact
mistake this script exists for. The real files are `data/persona/seed.md` and
`data/memory/log/`, and exempting the un-prefixed form would have made the check
pass on the defect that motivated it - which is how a check ends up being
decoration."""


def _claims(text: str) -> set[str]:
    found = set()
    for raw in BACKTICKED.findall(text):
        token = raw.strip()
        if not PATHY.match(token):
            continue
        if token.startswith("/"):
            continue  # `/health`, `/newbot`: an endpoint or a chat command
        if token in CITED:
            continue
        if "/" not in token and Path(token).suffix not in SUFFIXES:
            continue  # a bare word like `pytest` is not a path
        found.add(token)
    return found


def _candidates(token: str, doc: Path) -> list[Path] | None:
    """Where the token could legitimately point, or None if uncheckable.

    Resolved against the document's own directory *and* the repo root, because a
    module's CLAUDE.md writing `app.py` means the one beside it - which is how a
    person reads it too. Checking from the root alone reported 32 correct
    references as broken, and a check that cries wolf is a check someone turns
    off.
    """
    if any(mark in token for mark in PLACEHOLDERS):
        parent = token.rsplit("/", 1)[0] if "/" in token else ""
        return [ROOT / parent] if parent else None
    return [doc.parent / token, ROOT / token]


def targets() -> list[Path]:
    """Every document this check reads, deduplicated and ordered.

    Its own function so the scope can be asserted on directly. Inlined in `main` it
    could only be tested by a copy of the same loop, and a test that reimplements
    the filter it is checking passes whatever the filter does - which is how the
    `.venv`-only version survived a mutation check.
    """
    found: list[Path] = []
    for name in DOCS:
        path = ROOT / name
        if path.exists():
            found.append(path)
    for pattern in EXTRA_GLOBS:
        # Relative to ROOT, not absolute: when ROOT is itself a worktree the
        # absolute parts contain `.claude` for *every* match, so the skip matched
        # everything and these globs contributed nothing at all. See SKIP_DIRS.
        found.extend(
            p
            for p in ROOT.glob(pattern)
            if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
        )
    return sorted(set(found))


def main() -> int:
    broken: list[tuple[str, str]] = []
    checked = 0
    for doc in targets():
        text = doc.read_text(encoding="utf-8")
        for token in sorted(_claims(text)):
            candidates = _candidates(token, doc)
            if candidates is None:
                continue
            checked += 1
            if any(candidate.exists() for candidate in candidates):
                continue
            # Runtime paths are allowed to be absent, but only if the document
            # says where they come from - a bare `persona/seed.md` is the exact
            # mistake this script exists for.
            if token.startswith(RUNTIME):
                continue
            # `persona/seed.md` is the case this script was written for: it reads
            # as ours, it is not, and the real file is under data/. So a token
            # whose first segment is not a real top-level entry is reported even
            # when a same-named runtime directory exists somewhere below.
            broken.append((str(doc.relative_to(ROOT)), token))

    if broken:
        print(f"{len(broken)} broken reference(s) out of {checked} checked:\n")
        for doc, token in broken:
            print(f"  {doc}: {token}")
        print(
            "\nA path in backticks is a promise. Fix the reference, or prefix it with "
            "the directory it actually lives in (runtime paths under data/ are exempt)."
        )
        return 1

    print(f"ok: {checked} documented path(s) all exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
