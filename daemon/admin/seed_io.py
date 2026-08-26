"""Read and replace `persona/seed.md` for the admin's editor — the owner's own
keystrokes, and nothing else.

docs/CONTRACTS.md non-negotiable 5 used to say code must never write this file,
and `daemon/admin/routes.py` used to have a comment saying no route did.
docs/adr/0019 narrowed that: the anchor's guarantee is that **no model output
ever reaches the seed**, which is a claim about where the bytes came from, not
about whether a `write()` exists. `daemon/setup.py` already wrote the file from
three answers the owner typed; this module is the same thing after first run.

So there is exactly one entry point, it takes a string that arrived on an admin
request, and it never composes, edits or completes that string. Anything with a
model on the other end of it belongs in `daemon/persona/rules.py`, which writes
the *other* file.

Three refusals carry the weight, and each is a trap in the code this sits on:

  * **A stale hash is refused.** `seed.md` is meant to be hand-edited - that is
    what "human-owned" bought us - so a browser tab left open is holding text
    that may no longer be on disk. The hash is over the file's *bytes*, so a
    change that only altered the encoding counts as a change too.
  * **A file that cannot be decoded cannot be overwritten.** `loader.read_file`
    swallows a `UnicodeDecodeError` into `""`, so a CP949-saved seed reads as
    empty everywhere else in the daemon. Here it raises, and the write path
    refuses it for free: a caller that never got a hash cannot produce one.
  * **A blank seed is refused.** Not because empty is invalid - a fresh install
    has no seed at all - but because an empty *save* is what every truncated or
    failed read looks like, and `daemon doctor` calls an empty seed a
    proactivity blocker (`daemon/cli.py`). The editor reads through
    `read_seed`, never through `/admin/api/persona`, whose seed body shares a
    64 KB budget with the diaries and legitimately comes back `None`.

Sync, like `daemon/admin/settings_io.py` beside it and `daemon/admin/mind.py`
next to that: the admin's own small local file I/O is called straight from its
async handlers throughout this package, and inventing a second pattern for one
route would be the more confusing choice.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from daemon.fs import write_private_replace
from daemon.persona.loader import SEED_FILE, seed_path

BACKUP_SUFFIX = ".bak"
"""One slot, not a history. `seed.md` had no writer at all before this and so no
undo either; the point is to survive a mis-save, not to become a version store
inside a directory the owner browses in Obsidian."""

MAX_SEED_BYTES = 64 * 1024
"""Every byte of the seed is re-sent to the model on every turn (loader.py reads
it per call), so an accidental paste of a whole document is a permanent cost on
every message rather than a one-off mistake. Generous - the wizard's seed is
under a kilobyte - and named, so the refusal can say what the limit is."""


class SeedRejected(ValueError):
    """The text itself is not something we will write. 400."""


class SeedConflict(ValueError):
    """The file on disk is not the one the caller was editing. 409."""


class SeedUnreadable(ValueError):
    """The file exists and we cannot decode it. 409, and the editor stays shut."""


@dataclass(frozen=True)
class SeedView:
    text: str
    sha256: str
    exists: bool
    file: str


@dataclass(frozen=True)
class SeedSaved:
    sha256: str
    lines: int
    backup: str | None


def _current(path: Path) -> tuple[bytes | None, str]:
    """The bytes on disk and their hash, or `(None, "")` when there is no file.

    The empty hash is what a caller editing a not-yet-existing seed sends back,
    so "no file" has to be representable rather than an error.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, ""
    except OSError as exc:
        raise SeedUnreadable(f"{SEED_FILE} cannot be read: {exc}") from exc
    return raw, hashlib.sha256(raw).hexdigest()


def _normalise(text: str) -> str:
    """What a textarea posts, as a markdown file: LF endings, and a final newline.

    Without the ending conversion a CRLF submission would be written verbatim, go
    into the prompt verbatim, and hash differently from the text the page is
    showing - so the next save from the same page would conflict with itself.

    It stops there. An earlier version also `rstrip("\\n")`-ed, which collapsed
    whatever blank lines the owner had left between sections - an edit they did
    not make, to the one file this contract exists to keep exactly as written.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    return body if body.endswith("\n") else body + "\n"


def read_seed(data_dir: Path) -> SeedView:
    """The seed as text, with the hash a later `write_seed` must present.

    Unlike `/admin/api/persona`'s copy this is unbudgeted and never truncated -
    it is the only read an editor may load from.
    """
    path = seed_path(data_dir)
    raw, digest = _current(path)
    if raw is None:
        return SeedView(text="", sha256="", exists=False, file=str(SEED_FILE))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Named, not swallowed. `loader.read_file` turns this into `""` so a
        # conversation survives it; an editor doing the same would show an empty
        # box over a file full of text.
        raise SeedUnreadable(
            f"{SEED_FILE} is not valid UTF-8 ({exc.reason}), so it cannot be edited "
            "here without risking its contents. Re-save it as UTF-8 first."
        ) from exc
    return SeedView(text=text, sha256=digest, exists=True, file=str(SEED_FILE))


def write_seed(data_dir: Path, text: str, *, expected_sha256: str) -> SeedSaved:
    """Replace the seed with `text`, if the file is still the one at
    `expected_sha256`. Writes nothing on any refusal.

    `expected_sha256` is required rather than optional: an absent hash meaning
    "there was no file" is the one default that overwrites, and it is exactly
    what a caller who forgot the field would send.
    """
    path = seed_path(data_dir)
    body = _normalise(text)
    if not body.strip():
        raise SeedRejected(
            "an empty seed is not saved: the daemon would speak as a stock "
            "assistant and every proactive candidate would be declined"
        )
    size = len(body.encode("utf-8"))
    if size > MAX_SEED_BYTES:
        raise SeedRejected(
            f"the seed is {size} bytes; the limit is {MAX_SEED_BYTES}, because "
            "every byte of it is sent to the model on every turn"
        )

    raw, digest = _current(path)
    if digest != expected_sha256:
        raise SeedConflict(
            f"{SEED_FILE} changed on disk since it was loaded, so it was left alone. "
            "Reload the file, and re-apply the edit on top of what is there."
        )

    # The backup is staged first and moved into place last, so the slot holds the
    # content of the last *successful* save and nothing else. Neither simpler
    # order gets that: writing the backup outright would let a seed write that
    # then failed (no space, a read-only dir) consume the owner's only undo for a
    # save that never happened, and writing the seed first would leave a bad save
    # - the case the backup exists for - with nothing behind it if the backup
    # write is the one that fails.
    #
    # `raw` decodes: the hash matched, so these are the bytes `read_seed`
    # already decoded to hand the caller that hash.
    staged = None
    if raw is not None:
        staged = path.with_name(path.name + BACKUP_SUFFIX + ".new")
        write_private_replace(staged, raw.decode("utf-8"))
    try:
        write_private_replace(path, body)
    except BaseException:
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise

    backup: str | None = None
    if staged is not None:
        os.replace(staged, path.with_name(path.name + BACKUP_SUFFIX))
        backup = str(SEED_FILE) + BACKUP_SUFFIX
    return SeedSaved(
        sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        lines=len(body.splitlines()),
        backup=backup,
    )
