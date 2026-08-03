"""Markdown conversation log - the SOURCE OF TRUTH.

One file per *local* calendar day, `memory/log/YYYY-MM-DD.md`:

    # 2026-08-03

    ## 2026-08-02T17:00:00Z user
    오늘 저녁에 김치찌개 먹었어.

    ## 2026-08-02T17:00:09Z assistant
    맛있었어?

Why this shape, and why it re-parses (docs/CONTRACTS.md non-negotiable 1 makes
this file the original, so a later reflection pass has to be able to read it
back exactly):

  * One `## ` heading per record, carrying the two fields a human reader needs
    to follow a conversation: when, and who spoke. Body follows verbatim on the
    next lines. Records are append-only, so a whole day reads in one sequential
    scan.
  * The heading carries the **full** ISO-8601 UTC timestamp, not just the time
    of day. The file is split on the *local* day while timestamps are UTC, so a
    KST file legitimately contains records whose UTC date is the day before -
    a bare `17:00:00Z` would need the parser to know the writer's UTC offset to
    resolve. Writing the date makes a record self-contained and reparsable on
    any machine, in any timezone, and matches the ISO-8601-UTC-with-Z
    convention in schema.sql.
  * `RECORD_RE` anchors the entire line, so the only text that can be mistaken
    for a heading is that exact line. Bodies are escaped on write with a
    leading backslash - standard markdown escaping, so Obsidian still renders a
    literal `##` - and unescaped on read. Round-tripping is therefore lossless
    for arbitrary user text, including text that looks like a log heading.
  * Plain markdown only: no YAML front matter, no HTML comments. In particular
    provenance (`origin` / `session_kind` / `modality` / `sender_id`)
    deliberately does **not** appear here. It lives in sqlite columns so a
    model cannot forge it in prose (docs/CONTRACTS.md non-negotiable 3).

Trailing whitespace on a body is not preserved: the blank line between records
is structure, so the parser trims it. Callers normalise before writing (see
writer.py) to keep the markdown and the sqlite mirror byte-identical.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from daemon.fs import open_private_append, secure_dir
from daemon.memory.base import LoggedMessage

logger = logging.getLogger(__name__)

LOG_SUBDIR = Path("memory") / "log"

_TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
_ROLE = r"user|assistant"

RECORD_RE = re.compile(rf"^## ({_TS}) ({_ROLE})$")
"""A record heading. Anchored both ends on purpose - see the module docstring."""

_ESCAPED_RE = re.compile(rf"^(\\+)(## {_TS} (?:{_ROLE}))$")


@dataclass(frozen=True, slots=True)
class LogRecord:
    """What the markdown alone can tell us. Provenance is not in here by design."""

    ts: datetime
    role: Literal["user", "assistant"]
    content: str


# --- timestamps -------------------------------------------------------------
# The single helper the contract asks for (docs/CONTRACTS.md 8). It lives here
# rather than in store.py because markdown defines the format and sqlite mirrors
# it, so the dependency points from the mirror to the original, never back.


def utc_iso(ts: datetime) -> str:
    """ISO-8601 UTC with `Z`. A naive datetime is read as already-UTC."""
    return _as_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(value: str) -> datetime:
    """Inverse of `utc_iso`, returning a UTC-aware datetime."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


# --- paths ------------------------------------------------------------------


def local_date(ts: datetime) -> str:
    """The day this message belongs to, from the user's point of view.

    Days are split locally because "what happened on the 3rd" is a human
    question; the timestamps inside stay UTC.
    """
    return _as_utc(ts).astimezone().strftime("%Y-%m-%d")


def log_path(data_dir: Path, ts: datetime) -> Path:
    return data_dir / LOG_SUBDIR / f"{local_date(ts)}.md"


# --- rendering / parsing ----------------------------------------------------


def render(message: LoggedMessage) -> str:
    """One record, newline-terminated."""
    return f"## {utc_iso(message.ts)} {message.role}\n{_escape(message.content)}\n"


def parse(text: str) -> list[LogRecord]:
    """Read a day's log back. Lines outside any record (the date header, stray
    notes a human added) are ignored rather than fatal - a hand-edited log
    should degrade to fewer records, never to an exception."""
    records: list[LogRecord] = []
    heading: re.Match[str] | None = None
    body: list[str] = []

    for line in text.split("\n"):
        match = RECORD_RE.match(line)
        if match:
            if heading is not None:
                _push(records, heading, body)
            heading, body = match, []
        elif heading is not None:
            body.append(line)

    if heading is not None:
        _push(records, heading, body)
    return records


def _push(records: list[LogRecord], heading: re.Match[str], body: list[str]) -> None:
    """Append the record, or drop it if the heading is not actually parseable.

    RECORD_RE is looser than strptime - it accepts `2026-99-99T99:99:99Z` as a
    heading shape - and parse() promises that a hand-edited or sync-mangled log
    yields fewer records rather than an exception. Something else can write
    these files: the user, Obsidian conflict copies, a future importer. The
    markdown is the source of truth, so a crash here has nothing to fall back
    on: reflection would fail on that day forever.
    """
    try:
        records.append(_record(heading, body))
    except ValueError:
        # Heading only, never the body - the body is private conversation.
        logger.warning("log: skipping record with unparseable timestamp %r", heading[1])


def _record(heading: re.Match[str], body: list[str]) -> LogRecord:
    while body and not body[-1].strip():
        body.pop()  # the blank line separating records is structure, not content
    role: Literal["user", "assistant"] = "user" if heading[2] == "user" else "assistant"
    return LogRecord(ts=from_iso(heading[1]), role=role, content=_unescape("\n".join(body)))


def _escape(body: str) -> str:
    return "\n".join(
        "\\" + line if RECORD_RE.match(line) or _ESCAPED_RE.match(line) else line
        for line in body.split("\n")
    )


def _unescape(body: str) -> str:
    lines = []
    for line in body.split("\n"):
        match = _ESCAPED_RE.match(line)
        lines.append(match[1][1:] + match[2] if match else line)
    return "\n".join(lines)


# --- appending --------------------------------------------------------------

_locks: dict[Path, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    """One lock per log file. Single process, single event loop, so a plain dict
    is enough: there is no await between the lookup and the insert."""
    lock = _locks.get(path)
    if lock is None:
        lock = _locks.setdefault(path, asyncio.Lock())
    return lock


async def append(data_dir: Path, message: LoggedMessage) -> str:
    """Append one record and return the log file's path relative to `data_dir`,
    which is what the sqlite mirror stores to point back at the original."""
    path = log_path(data_dir, message.ts)
    async with _lock_for(path):
        await asyncio.to_thread(_append_blocking, path, render(message), local_date(message.ts))
    return path.relative_to(data_dir).as_posix()


def _append_blocking(path: Path, record: str, date_header: str) -> None:
    # Owner-only: these are the user's verbatim private conversations, and the
    # default of 0o644 under a 022 umask hands them to every local account.
    secure_dir(path.parent)
    header = f"# {date_header}\n" if not path.exists() else ""
    with open_private_append(path) as fh:
        fh.write(f"{header}\n{record}")
