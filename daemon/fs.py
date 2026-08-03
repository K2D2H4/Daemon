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


def secure_dir(path: Path) -> None:
    """Create `path` (and parents) owner-only.

    `Path.mkdir(mode=...)` applies the mode only to the leaf, so intermediate
    directories would keep the default. Walk the chain and pin each one.
    """
    path.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    for parent in (path, *path.parents):
        try:
            if parent.stat().st_mode & 0o777 != DIR_MODE:
                os.chmod(parent, DIR_MODE)
        except (OSError, PermissionError):
            # Reached a directory we do not own (a shared mount, $HOME on a
            # managed box). Stop rather than fail the write.
            break
        if parent.name in ("", os.sep):
            break


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
