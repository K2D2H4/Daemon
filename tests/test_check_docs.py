"""The documentation checker's own scope.

`scripts/` holds checks rather than tests and imports no product code, which is why
nothing here had ever tested one. That is fine until a check quietly changes what it
looks at: `**/CLAUDE.md` began matching git worktrees under `.claude/`, each a full
copy of the repo, and the reported count went from 131 paths to 209 with nothing in
the working tree changed.

Two failures in one. The number stops meaning anything, and a stale doc in an
abandoned worktree can fail the check for the main tree - where the reference it
names is perfectly fine. A gate whose verdict depends on whether someone happens to
have a worktree open is the thing this checker exists to prevent, one level up.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check_docs() -> object:
    """Loaded by path: `scripts/` is not a package, and deliberately not importable
    as one - nothing there may import `daemon`, and this keeps the arrow one-way."""
    spec = importlib.util.spec_from_file_location(
        "_check_docs_under_test", ROOT / "scripts" / "check_docs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scanned(module: object) -> list[Path]:
    """The checker's own scope, asked for rather than reconstructed.

    An earlier version of this helper rebuilt the glob-and-filter loop, and a
    mutation reverting the filter to `.venv`-only went undetected: the test was
    exercising its own copy. `targets()` exists so this line can be one call.
    """
    return module.targets()  # type: ignore[attr-defined]


def test_a_worktree_copy_of_the_repo_is_not_scanned_twice(check_docs: object) -> None:
    """Agent sessions create these under `.claude/worktrees/`, so this is the live
    case rather than a hypothetical - it was already true when it was found.

    Relative to the checker's own root, because a copy is only a copy if it is
    *below* the root. Asserting on the absolute parts made this fail whenever the
    suite ran from a worktree, whatever the checker did - and it read as worktree
    noise rather than as the finding it was sitting next to.
    """
    files = scanned(check_docs)

    assert files, "the checker found nothing to scan, so this proves nothing"
    root = check_docs.ROOT  # type: ignore[attr-defined]
    duplicated = [p for p in files if "worktrees" in p.relative_to(root).parts]
    assert duplicated == [], f"scanning worktree copies: {duplicated}"


def test_the_directories_that_hold_repo_copies_are_all_skipped(check_docs: object) -> None:
    """`.venv` was the only exclusion, and it was not the only such directory."""
    skip = check_docs.SKIP_DIRS  # type: ignore[attr-defined]

    assert {".claude", ".venv", ".git", "node_modules"} <= skip


def test_the_module_docs_are_still_scanned(check_docs: object) -> None:
    """The narrowing must not have thrown out what the checker is for. A skip list
    that also skipped `daemon/CLAUDE.md` would pass the tests above and check
    nothing."""
    relative = {p.relative_to(check_docs.ROOT).as_posix() for p in scanned(check_docs)}  # type: ignore[attr-defined]

    assert {
        "CLAUDE.md",
        "daemon/CLAUDE.md",
        "tests/CLAUDE.md",
        "evals/CLAUDE.md",
        "scripts/CLAUDE.md",
        "docs/CONTRACTS.md",
        "docs/ARCHITECTURE.md",
    } <= relative


def test_the_module_docs_are_scanned_when_the_repo_lives_in_a_worktree(
    check_docs: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The skip must be judged relative to the root, not against the absolute path.

    `.claude` is in every glob result's absolute parts once the root is *itself*
    under `.claude/worktrees/`, so the filter matched everything and `EXTRA_GLOBS`
    returned nothing: `check_docs.py` printed `ok` having read five files and no
    module doc, no ADR. That is the normal case for an agent session in this repo,
    so the half of the check nobody ran was the half that covers the docs an agent
    actually navigates by.

    Built on a fake root rather than asserting on the real one, because the real
    one's location is the variable under test - a test that only fails when the
    suite happens to run from a worktree is the same defect one level up.
    """
    root = tmp_path / ".claude" / "worktrees" / "some-branch"
    for relative in (
        "CLAUDE.md",
        "daemon/CLAUDE.md",
        "docs/adr/0001-a-decision.md",
        ".claude/worktrees/a-copy/daemon/CLAUDE.md",  # a nested copy: still skipped
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr(check_docs, "ROOT", root)

    relative = {p.relative_to(root).as_posix() for p in scanned(check_docs)}

    assert {"CLAUDE.md", "daemon/CLAUDE.md", "docs/adr/0001-a-decision.md"} <= relative
    assert ".claude/worktrees/a-copy/daemon/CLAUDE.md" not in relative


def test_the_checker_passes_on_this_repo(check_docs: object) -> None:
    """The check CI runs, run here too, so a broken reference fails the suite rather
    than waiting for the workflow."""
    assert check_docs.main() == 0  # type: ignore[attr-defined]
