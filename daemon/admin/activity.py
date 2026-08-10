"""What the daemon decided, shaped for the admin's Activity, Proactive and Tools views.

Three read-only payloads over four tables the process already writes:
`proactive_rounds`, `proactive_utterances`, `tool_calls`, `reflection_runs`. No
writes, no model calls, no side effects - these endpoints exist to *show* what
happened, and an endpoint that changed anything while being polled every fifteen
seconds would be a worse bug than the blindness it was added to fix.

## Which rounds become log rows

A round that considered nothing is not a row. There are 288 of them a day (a tick
every five minutes) and every one of them would push a real event off the page:
the Overview shows seven rows, so a day of empty rounds would render an activity
log that says the daemon did nothing, on a day it ran tools and spoke. They stay
visible where they cost nothing and mean something - as marks on the timeline,
where "quiet all afternoon" is a shape rather than 96 identical lines.

So: a round is a row when it considered at least one candidate. Everything else
about the day is still counted (`rounds`, `blocked_by`) from every round.

## Timestamps

Rows carry the UTC instant, verbatim from the column. The browser renders local
time - the same split as the markdown log (`log.local_date` for the filename, UTC
inside the record), and the reason is the same: a stored local time is wrong the
moment the machine travels.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from daemon.clock import now as clock_now
from daemon.config import Settings
from daemon.memory.log import utc_iso
from daemon.memory.store import Store
from daemon.proactivity.gate import local_day_start, parse_quiet_hours

KINDS = ("proact", "tool", "reflect", "refused")
"""The chips a row can carry. `refused` is a tool call the policy denied - it is
its own kind rather than a `tool` with a bad verdict because "what was it stopped
from doing" is the question that gets asked on its own."""

DEFAULT_LIMIT = 60
MAX_LIMIT = 500
"""A ceiling, so a hand-typed `?limit=100000` cannot make the admin read the whole
audit into memory."""

def _int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def clamp_limit(raw: Any) -> int:
    return max(1, min(MAX_LIMIT, _int(raw, DEFAULT_LIMIT)))


def _blocked_by(raw: str) -> dict[str, int]:
    """A round's `blocked_by` column as counts.

    Tolerant for the same reason `tick.row_candidate` is: the CHECK proves the text
    is JSON, not that it is an object of numbers, and one malformed row must not
    cost the whole view.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): _int(v, 0) for k, v in parsed.items()}


def _dominant_rule(counts: dict[str, int]) -> str | None:
    """The rule that blocked the most candidates this round, ties broken by name."""
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


# --- rows -------------------------------------------------------------------


def _round_rows(store: Store, since: datetime) -> list[dict[str, Any]]:
    """Rounds that considered something. See the module docstring for why the
    empty ones are left out."""
    rows = []
    for row in store.proactive_rounds_since(since=since):
        considered = int(row["considered"])
        if considered == 0:
            continue
        blocked = _blocked_by(row["blocked_by"])
        rule = _dominant_rule(blocked)
        if int(row["spoke"]):
            # The utterance itself is a row of its own, with the text. This one
            # would repeat it without saying anything the other does not.
            continue
        if rule is not None:
            verdict, detail = "blocked", rule
        elif int(row["declined"]):
            # The gate allowed it and the judge had nothing worth saying. The
            # healthy case, and the one worth telling apart from a block.
            verdict, detail = "silent", "no cost"
        else:
            verdict, detail = "silent", ""
        rows.append(
            {
                "ts": row["ts"],
                "kind": "proact",
                "text": f"{considered} candidate(s) considered",
                "verdict": verdict,
                "rule": detail,
            }
        )
    return rows


def _utterance_rows(store: Store, since: datetime) -> list[dict[str, Any]]:
    return [
        {
            "ts": row["spoken_at"],
            "kind": "proact",
            "text": row["text"],
            "verdict": "spoke",
            "rule": row["route"],
        }
        for row in store.utterances_since(since=since)
    ]


VERDICT_DETAIL_CHARS = 60
"""Ceiling on the text beside a verdict. `output_excerpt` holds what the tool told
the *model* - a paragraph of instructions, in one measured case - and a log row is
one line. The full text is still in the audit; this is the column, not the record."""


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    if len(text) <= VERDICT_DETAIL_CHARS:
        return text
    return text[: VERDICT_DETAIL_CHARS - 1] + "\u2026"


def _tool_rows(store: Store, limit: int) -> list[dict[str, Any]]:
    """Tool calls, refused ones included - a denial that leaves no trace is the
    same as no policy at all (the reason `record_tool_call` writes them)."""
    rows = []
    for row in store.recent_tool_calls(limit):
        denied = row["verdict"] == "deny"
        ran, ok = int(row["ran"]), row["ok"]
        if denied:
            verdict, rule = "refused", _clip(row["reason"] or "policy")
        elif not ran:
            verdict, rule = "waiting", _clip(row["reason"] or "approval")
        elif ok:
            verdict, rule = "ok", ""
        else:
            verdict, rule = "failed", _clip(row["output_excerpt"] or "")
        rows.append(
            {
                "ts": row["ts"],
                "kind": "refused" if denied else "tool",
                "text": row["preview"] or row["tool"],
                "verdict": verdict,
                "rule": rule,
                "duration_ms": row["elapsed_ms"],
            }
        )
    return rows


def _reflection_rows(store: Store, limit: int) -> list[dict[str, Any]]:
    rows = []
    for row in store.recent_reflection_runs(limit):
        status = row["status"]
        parts = []
        if int(row["facts"]):
            parts.append(f"{row['facts']} fact(s)")
        if int(row["entities"]):
            parts.append(f"{row['entities']} entity note(s)")
        if int(row["observations"]):
            parts.append(f"{row['observations']} observation(s)")
        summary = ", ".join(parts) if parts else _clip(row["detail"])
        rows.append(
            {
                "ts": row["ts"],
                "kind": "reflect",
                "text": f"reflection {row['date']}" + (f" - {summary}" if summary else ""),
                # `written` is the good outcome; the rest are reported as-is so a
                # night the model was unreachable reads as such instead of blank.
                "verdict": "ok" if status == "written" else status,
                "rule": "",
            }
        )
    return rows


def activity_payload(
    store: Store, *, kind: str = "all", limit: int = DEFAULT_LIMIT
) -> dict[str, Any]:
    """The merged log, newest first.

    `kind` filters to one chip; anything unrecognised is treated as `all` rather
    than as an error, because the filter arrives from a URL fragment a user can
    edit and an unusable admin is a worse answer than an unfiltered one.
    """
    wanted = kind if kind in KINDS else "all"
    since = clock_now() - timedelta(days=7)

    rows: list[dict[str, Any]] = []
    if wanted in ("all", "proact"):
        rows += _round_rows(store, since)
        rows += _utterance_rows(store, since)
    if wanted in ("all", "tool", "refused"):
        rows += _tool_rows(store, limit if wanted != "all" else MAX_LIMIT)
    if wanted in ("all", "reflect"):
        rows += _reflection_rows(store, limit)

    if wanted in KINDS:
        rows = [row for row in rows if row["kind"] == wanted]
    rows.sort(key=lambda row: row["ts"], reverse=True)
    return {"items": rows[:limit], "kind": wanted}


# --- today ------------------------------------------------------------------


def _quiet_window(settings: Settings) -> dict[str, Any] | None:
    """Quiet hours as fractions of the day, so the timeline can draw the band
    without re-parsing `HH:MM-HH:MM` in the browser. `None` when the window was
    emptied on purpose or is malformed - the gate warns about the latter, and the
    admin's job here is not to be the second place that decides what a typo means.
    """
    raw = settings.proactive_quiet_hours.strip()
    if not raw:
        return None
    try:
        start, end = parse_quiet_hours(raw)
    except ValueError:
        return None
    return {
        "label": raw,
        "start": (start.hour * 60 + start.minute) / 1440,
        "end": (end.hour * 60 + end.minute) / 1440,
    }


def _marks(store: Store, since: datetime) -> list[dict[str, str]]:
    """Every mark on the local day's timeline as `{ts, kind}`.

    Two different things, deliberately given two different shapes by the browser:

      * `round` - the tick ran here and had nothing to do. Every round emits one,
        including the ones that considered a candidate, because what this answers
        is "was the loop alive at 3pm", and a gap in the run is the answer that
        matters. Drawn as a continuous strip: at a tick every five minutes there
        are 288 a day, which as individual squares overlap into a smear (measured:
        3.9px of track per round, at 7px a square).
      * everything else - something actually happened. Drawn as squares above the
        strip, where they have room to be seen.

    The browser places each mark by its local hour, so this stays a flat list
    rather than pre-bucketed percentages: the daemon's UTC and the reader's
    timezone are allowed to disagree, and the axis is a local day.
    """
    marks: list[dict[str, str]] = []
    for row in store.proactive_rounds_since(since=since):
        marks.append({"ts": row["ts"], "kind": "round"})
        # A round the gate refused is an event, not just a heartbeat - it is the
        # one place "it wanted to speak and a rule stopped it" is visible at a
        # glance. Without this it was indistinguishable from an idle round.
        if not int(row["spoke"]) and _blocked_by(row["blocked_by"]):
            marks.append({"ts": row["ts"], "kind": "blocked"})
    for row in store.utterances_since(since=since):
        marks.append({"ts": row["spoken_at"], "kind": "spoke"})
    for row in store.recent_tool_calls(MAX_LIMIT):
        if row["ts"] < utc_iso(since):
            continue
        marks.append({"ts": row["ts"], "kind": "refused" if row["verdict"] == "deny" else "tool"})
    marks.sort(key=lambda mark: mark["ts"])
    return marks


def today_payload(store: Store, settings: Settings) -> dict[str, Any]:
    """The Overview's timeline and budget card: what the day looked like, and what
    the daemon is still allowed to do in it."""
    now = clock_now()
    day_start = local_day_start(now)

    rounds = store.proactive_rounds_since(since=day_start)
    blocked: dict[str, int] = {}
    for row in rounds:
        for rule, count in _blocked_by(row["blocked_by"]).items():
            blocked[rule] = blocked.get(rule, 0) + count

    spoken = store.utterances_since(since=day_start)
    tool_calls = [
        row
        for row in store.recent_tool_calls(MAX_LIMIT)
        if row["ts"] >= utc_iso(day_start)
    ]
    reflections = [
        {"ts": row["ts"], "date": row["date"], "status": row["status"]}
        for row in store.reflection_runs_since(since=day_start)
    ]

    last_spoke = store.last_utterance_at()
    cooldown = settings.proactive_cooldown_minutes
    next_allowed = None
    if last_spoke is not None and cooldown > 0:
        candidate = last_spoke + timedelta(minutes=cooldown)
        if candidate > now:
            next_allowed = utc_iso(candidate)

    return {
        "rounds": len(rounds),
        "spoke": len(spoken),
        "declined": sum(int(row["declined"]) for row in rounds),
        "considered": sum(int(row["considered"]) for row in rounds),
        "blocked_by": blocked,
        "tool_calls": len(tool_calls),
        "refusals": sum(1 for row in tool_calls if row["verdict"] == "deny"),
        "reflections": reflections,
        "budget": {"used": len(spoken), "total": settings.proactive_daily_budget},
        "next_allowed_at": next_allowed,
        "quiet_hours": _quiet_window(settings),
        "cooldown_minutes": cooldown,
        "enabled": settings.proactive_enabled,
        "marks": _marks(store, day_start),
        "now": utc_iso(now),
    }


# --- tools ------------------------------------------------------------------


def tool_log_payload(store: Store, *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The Tools view: what ran, what was refused, and under which mode.

    `arguments` is summarised rather than returned whole. The column holds whatever
    the model passed - a file's contents on a `write_file`, a page of text - and the
    admin renders one line per row.
    """
    items = []
    for row in store.recent_tool_calls(limit):
        items.append(
            {
                "ts": row["ts"],
                "tool": row["tool"],
                "preview": row["preview"],
                "arguments": _argument_summary(row["arguments"]),
                "verdict": row["verdict"],
                "mode": row["mode"],
                "reason": row["reason"],
                "origin": row["origin"],
                "channel": row["channel"],
                "ran": bool(row["ran"]),
                "ok": None if row["ok"] is None else bool(row["ok"]),
                "duration_ms": row["elapsed_ms"],
            }
        )
    return {"items": items}


ARGUMENT_SUMMARY_CHARS = 120


def _argument_summary(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    text = ", ".join(f"{key}={_short(value)}" for key, value in parsed.items())
    return text[:ARGUMENT_SUMMARY_CHARS]


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 40 else text[:39] + "…"
