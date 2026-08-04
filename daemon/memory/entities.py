"""Entity notes: `memory/entities/{name}.md` and their mirror in `entities`.

The other half of layer 3 in docs/PLAN.md 4.1. The log says what was said; these
say what is known about a person, a place, a project. Wiki-linked with `[[name]]`
so Obsidian and Logseq render the graph for free - that is the whole reason the
link syntax is what it is, and why links live inline in the prose rather than in a
separate section: a note that is appended to can keep its links, whereas a
trailing "related" block would have to be rewritten on every pass.

Notes are **appended**, never rewritten. One dated section per reflection pass, so
the file reads as a history and a torn write costs the last section rather than the
note. Same shape, and the same fsync, as `log.append`.

## Names become filenames, so names are checked

The name comes out of a model. Left alone, `../../persona/seed.md` would be a path
this module writes to - which is non-negotiable 5 with extra steps. `safe_name`
rejects anything that is not a plain file name, and `note_path` then verifies the
resolved path is still inside the entities directory. Two checks rather than one
because the first is a blocklist and the second is a boundary, and only the second
is exhaustive.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from daemon.clock import now as clock_now
from daemon.fs import open_private_append, secure_dir
from daemon.memory import log
from daemon.memory.store import Store

logger = logging.getLogger(__name__)

ENTITIES_SUBDIR = Path("memory") / "entities"

LINK_RE = re.compile(r"\[\[([^\[\]|]+?)\]\]")
"""A wiki link. `|` is excluded so an Obsidian alias (`[[name|shown]]`) does not
parse the display text as an entity name."""

_SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})$")

MAX_NAME_BYTES = 160
"""Filesystem name limits are per-byte (255 on ext4/APFS) and Korean is three
bytes a character, so a character count would not bound the filename. 160 leaves
room for the `.md` and for a name to stay readable."""

_FORBIDDEN = {".", ".."}


class UnsafeName(ValueError):
    """The model produced something that must not become a path."""


def safe_name(name: str) -> str:
    """The entity name, or raise. Never returns something with a path in it.

    Unicode is normalised to NFC first: on macOS the filesystem stores NFD, so
    `지현` typed one way and produced by a model the other way are different byte
    strings and would become two notes about one person.
    """
    cleaned = unicodedata.normalize("NFC", name).strip()
    if not cleaned:
        raise UnsafeName("empty entity name")
    if cleaned in _FORBIDDEN:
        raise UnsafeName(f"reserved name {cleaned!r}")
    if "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        raise UnsafeName(f"path separator in entity name {cleaned!r}")
    if any(unicodedata.category(char) == "Cc" for char in cleaned):
        raise UnsafeName("control character in entity name")
    if cleaned.startswith("."):
        # A dotfile would be invisible in the very tool this exists to feed.
        raise UnsafeName(f"entity name may not start with a dot: {cleaned!r}")
    if len(cleaned.encode("utf-8")) > MAX_NAME_BYTES:
        raise UnsafeName(f"entity name too long ({cleaned[:20]}...)")
    return cleaned


def entities_dir(data_dir: Path) -> Path:
    return data_dir / ENTITIES_SUBDIR


def note_path(data_dir: Path, name: str) -> Path:
    """Where this entity's note lives. Raises `UnsafeName` if that is anywhere
    other than directly inside the entities directory.

    The boundary check is not redundant with `safe_name`: that one enumerates what
    is known to be dangerous, this one states what is allowed. A symlink, a name
    with a Unicode form the blocklist did not anticipate, or a future edit to the
    blocklist all land here.
    """
    directory = entities_dir(data_dir)
    candidate = directory / f"{safe_name(name)}.md"
    resolved_dir = directory.resolve()
    if candidate.resolve().parent != resolved_dir:
        raise UnsafeName(f"entity note would land outside {resolved_dir}")
    return candidate


# --- rendering / parsing ----------------------------------------------------


def render_header(name: str, kind: str | None) -> str:
    """The first lines of a new note. `kind` is prose here *and* a column; the
    column is what anything reads, the prose is for the human browsing the graph.
    """
    what = f"\n{kind}\n" if kind else ""
    return f"# {name}\n{what}"


def render_section(date: str, body: str, links: tuple[str, ...] = ()) -> str:
    """One dated section, newline-terminated.

    Links are appended to the body rather than woven into it because this text
    comes from a model: rewriting its sentences to inject `[[...]]` would mean
    guessing where a name appears, and guessing wrong reads as a typo in the
    user's own memory.
    """
    trailing = ""
    mentioned = [f"[[{name}]]" for name in links if f"[[{name}]]" not in body]
    if mentioned:
        trailing = "\n\n" + " · ".join(mentioned)
    return f"## {date}\n{body.strip()}{trailing}\n"


def links_in(text: str) -> list[str]:
    """Entity names this text links to, in order, deduplicated."""
    return list(dict.fromkeys(match.strip() for match in LINK_RE.findall(text)))


def sections_in(text: str) -> list[str]:
    """The dates of each section, which is how many times this note was written
    to - the only reconstruction of `mention_count` the markdown allows."""
    return [match[1] for line in text.split("\n") if (match := _SECTION_RE.match(line.rstrip()))]


# --- writing ----------------------------------------------------------------


class EntityNotes:
    """Appends to entity notes and mirrors them. Owns the write order."""

    def __init__(self, data_dir: Path, store: Store) -> None:
        self._data_dir = data_dir
        self._store = store

    async def note(
        self,
        name: str,
        body: str,
        *,
        kind: str | None = None,
        links: tuple[str, ...] = (),
        date: str | None = None,
        now: datetime | None = None,
    ) -> int:
        """Append one observation about `name`, returning the entity's row id.

        `date` is the day the note is *about*, defaulting to today. Reflection
        passes the day it read, which matters as soon as anything catches up on
        history: without it, a first run over three months of log stamps every
        section with the date of the run and the note stops reading as a history.

        Markdown first, then the mirror (non-negotiable 1). A failure after the
        append leaves a note the mirror has not counted, which `rebuild` repairs;
        the reverse would leave a row pointing at a section that does not exist.

        Links are recorded in both directions in the mirror and written inline in
        the markdown, but linked entities do **not** get notes of their own here.
        Creating a note for every name mentioned would fill the graph with empty
        stubs; a name earns a note when reflection has something to say about it.
        Such a stub row's `file` therefore names where its note *would* live, which
        is deterministic from the name - so nothing has to be updated if it later
        earns one.
        """
        moment = now or clock_now()
        path = note_path(self._data_dir, name)
        canonical = safe_name(name)
        section_date = date or log.local_date(moment)
        is_new = not path.exists()
        header = render_header(canonical, kind) if is_new else ""
        section = render_section(section_date, body, links)

        await asyncio.to_thread(_append_blocking, path, header, section)

        entity_id = self._store.upsert_entity(
            name=canonical,
            kind=kind,
            file=path.relative_to(self._data_dir).as_posix(),
            now=moment,
        )
        for other in links:
            try:
                other_name = safe_name(other)
            except UnsafeName as exc:
                logger.warning("entities: dropping link (%s)", exc)
                continue
            if other_name == canonical:
                continue
            other_id = self._store.upsert_entity(
                name=other_name,
                kind=None,
                file=note_path(self._data_dir, other_name)
                .relative_to(self._data_dir)
                .as_posix(),
                now=moment,
            )
            self._store.link_entities(entity_id, other_id)
        return entity_id

    def graph(self, limit: int = 500) -> list[tuple[str, int, list[str]]]:
        """`(name, mentions, linked names)`, most mentioned first. What `daemon
        doctor` prints and what makes the M2 gate checkable by reading it."""
        out = []
        for row in self._store.entities(limit):
            linked = [other["name"] for other in self._store.links_for(int(row["id"]))]
            out.append((row["name"], int(row["mention_count"]), linked))
        return out


def _append_blocking(path: Path, header: str, section: str) -> None:
    secure_dir(path.parent)
    with open_private_append(path) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        prefix = header if header and path.stat().st_size == 0 else ""
        handle.write(f"{prefix}\n{section}")
        # The note is the source of truth and the mirror fsyncs on every commit,
        # so skipping this would make the original less durable than its index.
        handle.flush()
        os.fsync(handle.fileno())


# --- rebuild ----------------------------------------------------------------


def rebuild(data_dir: Path, store: Store) -> int:
    """Restore `entities` and `entity_links` from the notes. Returns entities seen.

    Idempotent in the sense that matters: `mention_count` is recomputed from the
    number of dated sections rather than incremented, so running this twice does
    not double it. That is also the only reconstruction available - the count is
    not written in the file, it is implied by how many times the file was written.
    """
    directory = entities_dir(data_dir)
    if not directory.exists():
        return 0

    seen = 0
    for path in sorted(directory.glob("*.md")):
        text = path.read_bytes().decode("utf-8", errors="replace")
        try:
            name = safe_name(path.stem)
        except UnsafeName as exc:
            logger.warning("entities: skipping note (%s)", exc)
            continue
        mentions = max(1, len(sections_in(text)))
        relative = path.relative_to(data_dir).as_posix()
        entity_id = store.upsert_entity(name=name, kind=None, file=relative, now=clock_now())
        store.set_mention_count(entity_id, mentions)
        seen += 1

        for other in links_in(text):
            try:
                other_name = safe_name(other)
            except UnsafeName:
                continue
            if other_name == name:
                continue
            other_id = store.upsert_entity(
                name=other_name,
                kind=None,
                file=(ENTITIES_SUBDIR / f"{other_name}.md").as_posix(),
                now=clock_now(),
            )
            store.link_entities(entity_id, other_id)
    return seen
