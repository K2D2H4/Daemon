"""Does the calendar generator say anything the daemon could not already say?

docs/adr/0021 moves ADR 0015's boundary once more: deterministic code now reads
the owner's calendar in stage 1, before the gate. That ADR names its own reversal
test and this is it:

    "If arm B does not clear 20/39 lines that name the event and state the time
    correctly against arm A's <= 2/39 - or if a single URL reaches an utterance,
    or a single spoken time disagrees with the clock - the generator has not
    earned the placement and this ADR is reverted."

## The two arms

Same event, same clock, same persona, same everything - the only difference is
whether the fenced title block (`agenda.render`) sits in the prompt.

  arm A - the judge sees `Candidate.reason` alone: "25분 뒤에 유저의 캘린더에
          적힌 일정이 하나 시작된다." A real line for it to try, not a straw man,
          because after ADR 0016 an elapsed-time-only reason is a reason the
          prompt says to answer.
  arm B - the reason plus the title, composed by `judge.compose_reason` - the
          same function `Judge.decide` calls, never a hand-rebuilt copy of that
          layout. `proactive_topic_spike`'s docstring records what a hand-copied
          copy cost the first time: it measured a message shape the daemon had
          already stopped sending.

Arm A is what this daemon would say today with a calendar candidate and no
material, which is the honest counterfactual - "would the line read the same
either way", the question 0015 asked and 0021 inherits.

## Per reply

Four yes/no questions, computed on the line the model actually wrote, before
`has_url`'s production veto would blank it - blanking first would make the leak
question trivially "no" for every reply that was ever going to leak, which is
the one thing worth counting on a kind whose raw material is 100% URLs.

  1. did it decline (empty `say`, invalid JSON, or over `MAX_CHARS`)?
  2. **names_event** - does the line carry the event, not just the clock?
     (`_names_event`, a heuristic. Not ground truth - hand-audit it.)
  3. **wrong_time** - does a number of minutes in the line disagree with the
     minutes the code put in the reason? The failure mode unique to this kind,
     and the reason `agenda.render` keeps the timestamp out of the block.
  4. would `has_url` refuse it?

Every reply is printed beside its verdicts so the labels can be hand-audited -
`daemon/MEASURED.md` records a run where a parse mislabelled 7 of 60 records and
nearly carried a wrong conclusion, and 0015's own `_carries_concrete_fact` is
named in that ADR as a heuristic that would pass a contentless opener.

## The three measurement defects this project has already paid for

Inherited wholesale from `evals/proactive_topic_spike.py`, whose docstring
explains each one against the `daemon/MEASURED.md` entry that produced it, and
whose implementations of the last two are imported here rather than copied:

1. **Arms interleaved trial by trial** - never a block of A then a block of B.
2. **Nothing pooled into a shared threshold** - the four questions are yes/no per
   reply and the comparison is a direct 2x2 Fisher exact on the arms' own counts.
3. **A tie is a statement about power, not about the mechanism** - `_verdict_line`
   reports "no power to detect the effect at n" and never "no effect".

## Data, and the honest limit on it

Real events, real persona, real MCP server, real hosted model - nothing here is a
fixture. But **the forward calendar is empty**: measured 2026-09-01, the owner has
0 events in the next 14 days and 13 in the last 180. A generator that only fires
before an event therefore has nothing to fire on today, and a live `daemon
proactive` at this moment correctly produces zero candidates.

So the population is the owner's real **past** events, replayed with the clock
wound back: for each event, `now = start - CALENDAR_LEAD_MINUTES`, and
`agenda.fetch` is called against the real server for that real historical window.
The MCP call, the reply, the parse and the fence are all the production path; only
`now` is supplied. That is a replay, not a simulation, and it is stated plainly
here rather than reported as if it were a live tick.

`--repeats` exists because the model is not deterministic and `daemon/MEASURED.md`
records what one batch is worth: a single run per arm looked convincing at 0 -> 2
and twenty runs put the same arm at 0.85 with 9 zeros. The default is 3, giving
39 trials per arm over 13 events, and per-event counts are printed alongside the
pooled ones so a single unusual event cannot carry the result on its own.

## Running it

    python3 -u -m evals.proactive_calendar_spike

Needs the owner's real `.env` (read from the fixed real path, never this
worktree's), `DAEMON_CALENDAR_EMAIL` set in it, the `google` MCP server reachable
with a valid Google consent, and `PROACTIVE_JUDGE` routed hosted rather than to
`ollama` - PLAN 6.2.1 already measured the local model on this prompt family, and
a silent fallback would remeasure that and call it evidence about the calendar.

Nothing here runs in CI, and evals/CLAUDE.md governs how a run is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from evals.proactive_topic_spike import (
    REAL_DATA_DIR,
    _load_real_env,
    _verdict_line,
)

DEFAULT_LOOKBACK_DAYS = 180
POPULATION_MAX = 100
"""`max_results` for the population sweep only - never for a production window.
See `_past_events`."""

DEFAULT_REPEATS = 3
"""Three passes over every event. See the module docstring's note on what one
batch is worth; 13 events x 3 is the 39 the ADR's reversal clause names."""


# --- the event population ------------------------------------------------------


async def _past_events(bridge, email: str, now: datetime, days: int) -> list:
    """The owner's real events from the last `days`, parsed by the production
    parser (`agenda.parse_events`) off a real server reply.

    Past rather than future because the forward calendar is empty - see the module
    docstring. The *query* is this file's, not production's: `agenda.fetch` caps
    `max_results` at `agenda.MAX_EVENTS` (5), which is right for a 30-minute
    production window and far too small for a 180-day population sweep. So the
    call is issued here with a larger cap and the reply is handed to the same
    parser a live tick uses - the population is the eval's business, the parsing
    is not.

    `agenda.parse_events`'s own filtering therefore applies unchanged: all-day
    events and unparseable stamps are dropped exactly as on a live tick, so the
    population is the set of events this generator could actually have spoken
    about.
    """
    from daemon.proactivity import agenda
    from daemon.tools.base import ToolError

    try:
        raw = await bridge.call(
            agenda.SERVER,
            agenda.TOOL,
            {
                "user_google_email": email,
                "time_min": (now - timedelta(days=days)).isoformat().replace("+00:00", "Z"),
                "time_max": now.isoformat().replace("+00:00", "Z"),
                "max_results": POPULATION_MAX,
            },
        )
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return []
    return sorted(agenda.parse_events(raw), key=lambda e: e.starts_at)


# --- the two arms --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmReply:
    """One arm's outcome for one trial - the raw line, and the four verdicts
    computed on it before `has_url`'s production veto would blank it."""

    text: str
    declined: bool
    names_event: bool
    wrong_time: bool
    would_leak_url: bool
    why_not: str = ""


_MINUTES_RE = re.compile(r"(\d+)\s*분")
"""Any "N분" in the line. The only number this kind is allowed to state, and the
one the code already decided - so a mismatch is the model inventing a time."""

_STOPWORDS = {"the", "and", "with", "for", "a", "an", "of", "to", "in", "on"}


def _title_tokens(title: str) -> list[str]:
    """Words from the title worth looking for in a reply.

    Latin runs of 3+ and Hangul runs of 2+, minus English function words. The
    owner's real titles are `Interview with UJET`, `BEN home assessment`,
    `Mistral | Applied AI - Hiring Manager`, `회의` - so the distinctive token is
    almost always a Latin brand name, and `with`/`for` would otherwise match a
    line that named nothing.
    """
    words = re.findall(r"[A-Za-z]{3,}|[가-힣]{2,}", title)
    return [w for w in words if w.casefold() not in _STOPWORDS]


def _names_event(line: str, title: str) -> bool:
    """Whether the line carries the *event*, not just the clock.

    Best-effort, not ground truth - the run's hand audit is why this is a
    heuristic rather than the verdict. It is deliberately stricter than
    `proactive_topic_spike._carries_concrete_fact`, which counts any run of two
    Latin letters and would therefore score `UJET 면접 25분 남았네` and a bare
    `25분 뒤에 뭐 있지 않아?` the same way: here a line counts only if it repeats
    a distinctive token of the title. A digit alone never counts, because in arm A
    the reason hands the model a digit and nothing else.
    """
    if not line:
        return False
    folded = line.casefold()
    return any(token.casefold() in folded for token in _title_tokens(title))


def _wrong_time(line: str, minutes: int) -> bool:
    """Whether a minute count in the line disagrees with the one code computed.

    A line with no "N분" in it is not wrong, only silent about the time - the
    reason is allowed to be answered without restating it. A line that states a
    *different* number has invented one, which is the failure `agenda.render`
    keeps the raw timestamp out of the block to prevent, and which the ADR's
    reversal clause counts at zero tolerance.
    """
    stated = [int(m) for m in _MINUTES_RE.findall(line)]
    return any(value != minutes for value in stated)


async def _judge_call(gateway, messages, title: str, minutes: int) -> ArmReply:
    """One model call, parsed exactly as `Judge.decide` parses its own -
    `judge._read_reply` for the JSON and length contract, `judge.has_url` for the
    production veto - except the veto is *reported* rather than applied, so a
    reply that would have been blanked still gets all four verdicts computed on
    what the model actually wrote.
    """
    from daemon.llm.base import ProviderError
    from daemon.proactivity import judge as judge_module
    from daemon.tasks import Task

    try:
        completion = await gateway.complete(
            Task.PROACTIVE_JUDGE, messages, max_output_tokens=judge_module.MAX_OUTPUT_TOKENS
        )
    except ProviderError as exc:
        return ArmReply(
            text="",
            declined=True,
            names_event=False,
            wrong_time=False,
            would_leak_url=False,
            why_not=f"model unavailable: {exc}",
        )

    utterance = judge_module._read_reply(
        completion.text,
        model=completion.model,
        stop_reason=completion.meta.get("stop_reason", ""),
    )
    line = utterance.text if utterance else ""
    return ArmReply(
        text=line,
        declined=not utterance,
        names_event=_names_event(line, title),
        wrong_time=_wrong_time(line, minutes),
        would_leak_url=bool(line) and judge_module.has_url(line),
        why_not=utterance.why_not,
    )


@dataclass(frozen=True, slots=True)
class Trial:
    title: str
    minutes: int
    reason: str
    arm_a: ArmReply
    arm_b: ArmReply


async def _run_trial(gateway, persona: str, candidate, title: str, minutes: int) -> Trial:
    """One event, both arms, back to back - never a whole arm before the other."""
    from daemon.llm.base import Message
    from daemon.proactivity import agenda
    from daemon.proactivity import judge as judge_module

    system_msgs = [
        Message(role="system", content=persona),
        Message(role="system", content=judge_module.SYSTEM),
    ]
    block = agenda.render(title, secrets.token_hex(4))

    arm_a_messages = [
        *system_msgs,
        Message(role="user", content=judge_module._reason_block(candidate)),
    ]
    arm_b_messages = [
        *system_msgs,
        Message(role="user", content=judge_module.compose_reason(candidate, block)),
    ]

    arm_a = await _judge_call(gateway, arm_a_messages, title, minutes)
    arm_b = await _judge_call(gateway, arm_b_messages, title, minutes)
    return Trial(
        title=title, minutes=minutes, reason=candidate.reason, arm_a=arm_a, arm_b=arm_b
    )


# --- reporting -----------------------------------------------------------------


def _print_trial(i: int, trial: Trial) -> None:
    print(f"[trial {i}] title={trial.title!r} minutes={trial.minutes}")
    print(f"  reason: {trial.reason}")
    for label, reply in (("A (reason only)", trial.arm_a), ("B (reason+title)", trial.arm_b)):
        verdicts = (
            f"declined={reply.declined} names_event={reply.names_event} "
            f"wrong_time={reply.wrong_time} would_leak_url={reply.would_leak_url}"
        )
        if reply.declined:
            print(f"  {label}: (declined) {verdicts}  why_not={reply.why_not!r}")
        else:
            print(f"  {label}: {reply.text!r}  {verdicts}")
    print()


def _report(trials: list[Trial]) -> None:
    n = len(trials)
    print(f"=== {n} paired trials (arm A vs arm B, interleaved trial by trial) ===\n")
    for i, trial in enumerate(trials, 1):
        _print_trial(i, trial)

    def counts(pick) -> tuple[int, int, int, int]:
        return (
            sum(1 for t in trials if pick(t.arm_a)),
            n,
            sum(1 for t in trials if pick(t.arm_b)),
            n,
        )

    print("=== per event (arm B only - does one unusual event carry the result?) ===")
    by_title: dict[str, list[Trial]] = {}
    for trial in trials:
        by_title.setdefault(trial.title, []).append(trial)
    for title, group in by_title.items():
        named = sum(1 for t in group if t.arm_b.names_event)
        print(f"  {named}/{len(group)}  {title!r}")

    print("\n=== comparison (direct, arm A vs arm B - nothing pooled) ===")
    print(_verdict_line("declined      ", *counts(lambda r: r.declined)))
    print(_verdict_line("names_event   ", *counts(lambda r: r.names_event)))
    print(_verdict_line("wrong_time    ", *counts(lambda r: r.wrong_time)))
    print(_verdict_line("would_leak_url", *counts(lambda r: r.would_leak_url)))

    a_named = sum(1 for t in trials if t.arm_a.names_event)
    b_named = sum(1 for t in trials if t.arm_b.names_event)
    leaks = sum(1 for t in trials for r in (t.arm_a, t.arm_b) if r.would_leak_url)
    wrong = sum(1 for t in trials for r in (t.arm_a, t.arm_b) if r.wrong_time)
    print("\n=== ADR 0021's reversal clause, applied ===")
    print(f"  arm B names_event   {b_named}/{n}   (needs >= 20/39 at the stated n)")
    print(f"  arm A names_event   {a_named}/{n}   (needs <= 2/39)")
    print(f"  url leaks           {leaks}        (needs 0)")
    print(f"  wrong times         {wrong}        (needs 0)")
    print(
        "\n  These counts are the input to the call, not the call. Hand-audit every\n"
        "  line above before writing a verdict into daemon/MEASURED.md - "
        "`_names_event`\n  is a heuristic and this file does not pretend otherwise."
    )


# --- assembly ------------------------------------------------------------------


async def _build_bridge(data_dir: Path):
    """A real `McpBridge` connected to only the `google` entry of
    `data/mcp.json`. Read-only on `data_dir` (`load_config` reads `mcp.json` and
    nothing else); the other configured servers are filtered out rather than
    started and left idle, exactly as `proactive_topic_spike` filters to `tavily`.
    """
    from daemon.proactivity import agenda
    from daemon.tools.base import Registry
    from daemon.tools.mcp import McpBridge, load_config

    configs = [c for c in load_config(data_dir).servers if c.name == agenda.SERVER]
    if not configs:
        raise RuntimeError(f"no {agenda.SERVER!r} server in {data_dir / 'mcp.json'}")
    bridge = McpBridge(configs)
    await bridge.start(Registry())
    return bridge


async def _persona_text(data_dir: Path) -> str:
    from daemon.persona.loader import load_persona, read_file, seed_path

    if not (await read_file(seed_path(data_dir))).strip():
        return ""
    return await load_persona(data_dir)


async def _main(repeats: int, days: int) -> int:
    _load_real_env()

    from daemon.clock import now as clock_now
    from daemon.config import Settings
    from daemon.proactivity.candidates import CALENDAR_LEAD_MINUTES, calendar_candidates
    from evals.proactive_judge import RoutedToLocalModel, build_judge_gateway

    settings = Settings(_env_file=None)  # os.environ only - never this worktree's .env
    if not settings.calendar_email:
        print("error: DAEMON_CALENDAR_EMAIL is not set in the real .env", file=sys.stderr)
        return 1
    try:
        gateway, _recorder = build_judge_gateway(settings)
    except RoutedToLocalModel as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    persona = await _persona_text(REAL_DATA_DIR)
    if not persona:
        print(f"error: no persona seed under {REAL_DATA_DIR}", file=sys.stderr)
        return 1

    bridge = await _build_bridge(REAL_DATA_DIR)
    try:
        events = await _past_events(bridge, settings.calendar_email, clock_now(), days)
        if not events:
            print(f"no events in the last {days} days; nothing to replay.", file=sys.stderr)
            return 1
        print(f"replaying {len(events)} real event(s) x {repeats} pass(es)\n")

        class _NoKeys:
            """No candidate has ever been raised, so nothing is deduplicated away.
            The real reader would suppress a replayed event whose key is already
            spent, which for a replay is an artefact of the replay rather than a
            fact about the generator."""

            def existing_dedup_keys(self, keys):
                return set()

        trials: list[Trial] = []
        for _pass in range(repeats):
            for event in events:
                moment = event.starts_at - timedelta(minutes=CALENDAR_LEAD_MINUTES)
                # The production generator, against the real server, for the real
                # historical window - only `now` is supplied.
                found, note = await calendar_candidates(
                    bridge, _NoKeys(), settings.calendar_email, now=moment
                )
                if note:
                    print(f"  skipped {event.title!r}: {note}", file=sys.stderr)
                    continue
                if not found:
                    print(
                        f"  skipped {event.title!r}: the replayed window produced "
                        "no candidate",
                        file=sys.stderr,
                    )
                    continue
                candidate = found[0]
                title = str(candidate.payload["title"])
                trials.append(
                    await _run_trial(
                        gateway, persona, candidate, title, CALENDAR_LEAD_MINUTES
                    )
                )
        if not trials:
            print("no trials ran.", file=sys.stderr)
            return 1
        _report(trials)
    finally:
        await bridge.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    args = parser.parse_args()
    return asyncio.run(_main(args.repeats, args.days))


if __name__ == "__main__":
    raise SystemExit(main())
