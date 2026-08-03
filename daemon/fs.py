"""Filesystem permissions for private data.

Everything Daemon writes under the data dir is the user's most private material:
verbatim conversations, what the AI has concluded about them, an evolving model
of their personality. Default permissions are not acceptable here — with a
typical umask of 022 the log lands world-readable, so any other local account
on the host can read it.

Modes are pinned explicitly rather than left to umask, because umask can only
remove bits and a permissive umask would leave these files exposed.
"""

from __future__ import annotations

import os
from pathlib import Path

DIR_MODE = 0o700
FILE_MODE = 0o600


def secure_dir(path: Path, *, stop_at: Path | None = None) -> None:
    """Create `path` owner-only, tightening only what this call is responsible for.

    `Path.mkdir(mode=...)` applies the mode only to the leaf, so intermediates it
    creates would keep the default and have to be tightened too.

    The walk is bounded on purpose. An earlier version climbed every parent until
    a chmod failed, which was harmless while the data dir was relative (`./data`
    stopped at the repo root) and became a real side effect once log paths were
    absolutised: `daemon install` would have walked out to $HOME and set it to
    0700, cutting off ~/Public and friends and stripping setgid or sticky bits
    from directories that have nothing to do with us. Nothing outside our own
    data directory is ours to re-permission.

    `stop_at` is the highest directory this call may touch; it defaults to the
    directories this call actually created.
    """
    created: list[Path] = []
    if not path.exists():
        for candidate in (path, *path.parents):
            if candidate.exists():
                break
            created.append(candidate)
    path.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)

    targets = [path, *created]
    if stop_at is not None:
        # Everything from `path` up to and including `stop_at`.
        for parent in path.parents:
            targets.append(parent)
            if parent == stop_at:
                break
    for target in dict.fromkeys(targets):
        try:
            if target.stat().st_mode & 0o777 != DIR_MODE:
                os.chmod(target, DIR_MODE)
        except (OSError, PermissionError):
            continue


def open_private_append(path: Path):
    """Append-mode handle whose file, if newly created, is owner-only.

    `os.open` sets the mode at creation, so there is no window where the file
    exists world-readable — a create-then-chmod would have one.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_MODE)
    return os.fdopen(fd, "a", encoding="utf-8")


def secure_file(path: Path) -> None:
    """Tighten an existing file. For paths a library created for us (sqlite)."""
    try:
        if path.exists() and path.stat().st_mode & 0o777 != FILE_MODE:
            os.chmod(path, FILE_MODE)
    except (OSError, PermissionError):
        pass


def harden_existing(data_dir: Path) -> None:
    """One-shot migration for installs created before permissions were pinned.

    Cheap enough to run at every startup: a few hundred stat calls at the scale
    of one person's conversation history.
    """
    if not data_dir.exists():
        return
    for path in data_dir.rglob("*"):
        try:
            if path.is_dir():
                if path.stat().st_mode & 0o777 != DIR_MODE:
                    os.chmod(path, DIR_MODE)
            elif path.stat().st_mode & 0o777 != FILE_MODE:
                os.chmod(path, FILE_MODE)
        except (OSError, PermissionError):
            continue
