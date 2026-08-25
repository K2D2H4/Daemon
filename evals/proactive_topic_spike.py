"""Task 5: does the search change what a `topic` line says?

docs/adr/0015-code-may-search-where-the-model-may-not.md moved a security
boundary: deterministic code may now run one read-only web search on a
proactive turn, where before nothing outside the daemon's own database could
reach an unprompted utterance. The ADR names its own reversal test, and this
is it:

    "topic candidates with a search result against the same candidates
    without, judged for whether the line carries content. If a topic line
    reads the same either way, the search bought nothing and this decision
    should be reverted."

## The two arms

Same entity, same reason, same persona, same everything - the only difference
is whether the rendered titles block (`daemon/proactivity/topics.render`) sits
in the prompt.

  arm A - the judge sees `Candidate.reason` alone.
  arm B - the judge sees the reason plus the rendered titles.

`Judge.decide` has no A/B switch: with `bridge=None` it drops a `topic`
candidate outright rather than answering without a search (see its
docstring), so neither arm can be measured by calling `decide()` with a
different bridge. Both arms are built here directly out of the same pieces
`decide()` uses - `judge.SYSTEM`, `judge._reason_block`, `topics.render`,
`gateway.complete`, `judge._read_reply`, `judge.has_url` - so what changes
between them is exactly the one block ADR 0015 is about, and nothing else.

## Per reply

Three yes/no questions, computed on the line the model actually wrote (before
`has_url`'s production veto blanks it out - blanking it first would make the
third question trivially "no" for every reply that was ever going to leak a
url, which is the one thing worth counting):

  1. did it decline (empty `say`, invalid JSON, or over `MAX_CHARS`)?
  2. does the line carry a concrete fact - a company, a number, an event - as
     opposed to an open question? (`_carries_concrete_fact`, a heuristic -
     see its docstring. Not ground truth.)
  3. would `has_url` refuse it?

Every reply is printed beside its verdict so the labels can be hand-audited
afterwards - `daemon/MEASURED.md` records a run where a parse mislabelled 7 of
60 records and nearly carried a wrong conclusion.

## Three measurement defects this project has already paid for

1. **Interleave the arms trial by trial.** `daemon/MEASURED.md` records an
   n=60 run that ran all of arm 1 then all of arm 2; arm was confounded with
   whatever drifted across the run's wall-clock span (model routing, load),
   and the result did not replicate. This script draws one entity per trial
   and runs arm A then arm B back to back, immediately, before moving to the
   next trial - never a block of one arm followed by a block of the other.
2. **Never score the two arms against a threshold pooled from both.** The same
   `MEASURED.md` entry records a metric that turned a continuous score into a
   pooled-median threshold; the two hit-counts came out structurally
   near-complementary (summing to ~n) and a significance test over them was
   scoring roughly one coin flip about which arm landed on the short side, not
   a real difference - caught only because a replication reversed the sign.
   The three questions here are already yes/no per reply, not a continuous
   score binarised against a pooled cutoff, and the comparison below is a
   direct 2x2 Fisher exact test between the arms' own counts - nothing pooled
   into a shared threshold first.
3. **A tie is a statement about the probe's power, not the mechanism.** See
   `_verdict_line`: a p-value that clears no ordinary significance level is
   reported as "no power to detect the effect at n=<n>", never as "no effect".

## What this script does not do

It does not conclude anything about ADR 0015. It prints two tables and three
p-values. `.superpowers/sdd/2026-08-25-proactive-topics/task-5-brief.md` steps
2-6 - the real run, the hand audit, the replication, the write-up, the
revert-or-keep call - are for whoever reads the output, not for this file.

## Data

Real entities, real persona, real search, real hosted model - nothing here is
a fixture. `entities` is read through a `mode=ro` sqlite URI and nothing in
this file ever opens the owner's database for writing. The population is every
entity gone quiet by `TOPIC_QUIET_DAYS` (the same gate `topic_candidates`
applies) that is not itself pointer-shaped (`judge.has_url` on the entity's
own name) - `Judge.decide` drops a pointer-shaped entity before the search or
the model call ever runs, in both possible worlds, so this script excludes it
from the population rather than measuring a guaranteed decline in both arms
and calling that data.

## Running it

    python3 -u -m evals.proactive_topic_spike 30

Needs the owner's real `.env` (read directly from `/Users/gimdaehyeon/Daemon/.env`,
never from this worktree's own) and the real `PROACTIVE_JUDGE` route to be
hosted, not `ollama` - PLAN 6.2.1 already measured the local model's weakness
on this exact prompt family, and a silent fallback here would remeasure that
and call it evidence about the search. Needs `TAVILY_API_KEY` in that `.env`
and the `tavily` server reachable at whatever url `data/mcp.json` names -
without either, arm B has no titles to render and every trial is skipped.

Nothing here runs in CI, and evals/CLAUDE.md governs how a run is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import re
import secrets
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

REAL_ENV = Path("/Users/gimdaehyeon/Daemon/.env")
REAL_DATA_DIR = Path("/Users/gimdaehyeon/Daemon/data")
"""The owner's real install, not this worktree's. Read-only throughout - see
the module docstring's "Data" section."""


def _load_real_env() -> None:
    """Read the owner's real `.env` into `os.environ`, the same way
    `m0_voice_spike._load_env` reads the cwd's - except this always reads the
    fixed real path, never a cwd-relative one, because a worktree cwd shadows
    the intended `.env` with its own (a prior session's own recorded gotcha).
    `setdefault` so an already-exported env var still wins.
    """
    if not REAL_ENV.exists():
        print(f"{REAL_ENV} does not exist; nothing to read.", file=sys.stderr)
        return
    for line in REAL_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# --- the entity population ---------------------------------------------------


def _entity_pool(data_dir: Path, now: datetime) -> list[tuple[str, str]]:
    """`(name, updated_at)` for every real entity eligible to be a `topic`
    candidate's subject, read through a `mode=ro` sqlite URI so this can never
    write to the owner's live database (its `-wal`/`-shm` files are present -
    the resident is running).

    Eligibility mirrors what `Judge.decide` actually does with a `topic`
    candidate, not the full production tick: gone quiet by `TOPIC_QUIET_DAYS`
    (`candidates.stale_entities`'s own filter), and not pointer-shaped
    (`judge.has_url` on the name) - a pointer-shaped entity is dropped by
    `decide` before the search or the model call runs, in either arm, so it
    is excluded from the population here rather than measured as an identical
    guaranteed decline in both and reported as if that were a finding.

    Deliberately not the `NOT EXISTS ... raised_since` anti-rearm clause real
    `Store.stale_entities` also applies - that throttle exists so one
    production tick does not re-ask about an entity it just raised, which has
    no meaning for a script that is not writing `proactive_candidates` rows.
    """
    from daemon.clock import to_iso
    from daemon.proactivity.candidates import TOPIC_QUIET_DAYS
    from daemon.proactivity.judge import has_url

    quiet_since = now - timedelta(days=TOPIC_QUIET_DAYS)
    db_path = data_dir / "daemon.sqlite3"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name, updated_at FROM entities WHERE trim(name) != '' "
            "AND updated_at < ? ORDER BY updated_at ASC",
            (to_iso(quiet_since),),
        ).fetchall()
    finally:
        conn.close()
    return [(str(name), str(updated_at)) for name, updated_at in rows if not has_url(str(name))]


def _candidate_for(name: str, updated_at: str, now: datetime):
    """One `topic` `Candidate`, built with `topic_candidates`'s own reason
    formula rather than a hand-copied string, so this file cannot drift from
    what `candidates.py` actually writes into `Candidate.reason`. Routed
    through a one-row fake reader rather than duplicating the f-string,
    because the reason a hosted model reads has to be the exact reason
    production would have given it.
    """
    from daemon.clock import parse_iso
    from daemon.proactivity.candidates import topic_candidates

    class _OneRow:
        def stale_entities(self, limit: int, quiet_since: datetime, raised_since: datetime):
            return [{"name": name, "updated_at": updated_at}]

    parse_iso(updated_at)  # fail loudly here, not inside topic_candidates, if malformed
    found = topic_candidates(_OneRow(), now)
    assert len(found) == 1, f"expected exactly one candidate for {name!r}, got {len(found)}"
    return found[0]


# --- the two arms --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmReply:
    """One arm's outcome for one trial - the raw line the model wrote, and the
    three yes/no verdicts computed on it (see the module docstring's "Per
    reply" section for why they read the *raw* line, before `has_url`'s
    production veto would blank it)."""

    text: str
    declined: bool
    concrete_fact: bool
    would_leak_url: bool
    why_not: str = ""


_FILLER_PHRASES = (
    "오랜만이야", "요즘 어때", "별일 없어", "시간이 많이 흘렀네", "오늘도 변함없네",
)
"""`daemon/proactivity/judge.py`'s `SYSTEM` names these, verbatim, as the
family of contentless opener the prompt tells the model `{"say": ""}` is
correct for. A line built from one of these carries no concrete fact
regardless of what else in it matches below."""

_DIGIT_RE = re.compile(r"\d")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]{2,}")


def _carries_concrete_fact(line: str, entity: str) -> bool:
    """Best-effort, not ground truth - the brief's step 3 hand-audits every
    verdict this prints, and this heuristic is why that step exists rather
    than being optional.

    A line counts as carrying a concrete fact - a company, a number, an event
    - if it has a digit (a date, a count, a version), a run of 2+ Latin
    letters (this population's entities and the titles a search for them
    returns are almost all Latin-scripted brand or project names - Sendbird,
    UJET, ReadyTalk, llm-wiki), or names the entity itself. Excluded first:
    the open-question filler family `judge.SYSTEM` names by example, which
    can otherwise contain the entity name while asking nothing.
    """
    if not line:
        return False
    if any(phrase in line for phrase in _FILLER_PHRASES):
        return False
    if _DIGIT_RE.search(line):
        return True
    if _LATIN_RUN_RE.search(line):
        return True
    return bool(entity) and entity in line


async def _judge_call(gateway, messages, entity: str) -> ArmReply:
    """One model call, parsed exactly the way `Judge.decide` parses its own -
    `judge._read_reply` for the JSON/length contract, `judge.has_url` for the
    production veto - except the veto's result is reported rather than
    applied, so a reply that would have been blanked out still gets all three
    verdicts computed on what the model actually wrote.
    """
    from daemon.llm.base import ProviderError
    from daemon.proactivity import judge as judge_module
    from daemon.tasks import Task

    try:
        completion = await gateway.complete(
            Task.PROACTIVE_JUDGE, messages, max_output_tokens=judge_module.MAX_OUTPUT_TOKENS
        )
    except ProviderError as exc:
        return ArmReply(text="", declined=True, concrete_fact=False, would_leak_url=False,
                         why_not=f"model unavailable: {exc}")

    utterance = judge_module._read_reply(
        completion.text, model=completion.model,
        stop_reason=completion.meta.get("stop_reason", ""),
    )
    line = utterance.text if utterance else ""
    return ArmReply(
        text=line,
        declined=not utterance,
        concrete_fact=_carries_concrete_fact(line, entity),
        would_leak_url=bool(line) and judge_module.has_url(line),
        why_not=utterance.why_not,
    )


@dataclass(frozen=True, slots=True)
class Trial:
    entity: str
    reason: str
    titles: tuple[str, ...]
    arm_a: ArmReply
    arm_b: ArmReply


async def _run_trial(gateway, persona: str, bridge, candidate) -> Trial | None:
    """One entity, both arms, back to back - never a whole arm's worth of
    trials before the other starts (see module docstring, defect 1).

    The search runs once, shared by arm B, exactly as ADR 0015 requires ("one
    search per gate-passed candidate, never per tick"). `None` means this
    entity had nothing behind it - the caller draws another, the same outcome
    `Judge.decide` reaches for a `topic` candidate whose search finds nothing
    (dropped before the one model call is spent).
    """
    from daemon.llm.base import Message
    from daemon.proactivity import judge as judge_module
    from daemon.proactivity import topics

    entity = str(candidate.payload["entity"])
    titles = await topics.search_titles(bridge, entity)
    if not titles:
        return None

    system_msgs = [
        Message(role="system", content=persona),
        Message(role="system", content=judge_module.SYSTEM),
    ]
    reason_msg = Message(role="user", content=judge_module._reason_block(candidate))

    arm_a_messages = [*system_msgs, reason_msg]
    topic_block = topics.render(entity, titles, secrets.token_hex(4))
    arm_b_messages = [*system_msgs, Message(role="system", content=topic_block), reason_msg]

    arm_a = await _judge_call(gateway, arm_a_messages, entity)
    arm_b = await _judge_call(gateway, arm_b_messages, entity)
    return Trial(entity=entity, reason=candidate.reason, titles=tuple(titles),
                 arm_a=arm_a, arm_b=arm_b)


# --- statistics ------------------------------------------------------------------


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided exact p-value for the 2x2 table `[[a, b], [c, d]]` (row 1 =
    arm A: count with the property, count without; row 2 = arm B, same
    shape). No scipy in this project's dependencies, so this sums the
    hypergeometric probability of every table sharing the observed table's
    margins that is at most as likely as the one actually observed - the
    standard exact two-sided test, computed directly on the arms' own counts
    (see module docstring, defect 2: nothing here is a threshold pooled from
    both arms first).
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c

    def hyper_prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = hyper_prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    total = sum(p for x in range(lo, hi + 1) if (p := hyper_prob(x)) <= observed * (1 + 1e-9))
    return min(1.0, total)


def _verdict_line(label: str, a_yes: int, a_total: int, b_yes: int, b_total: int) -> str:
    """Counts, the direct-comparison p-value, and a verdict that never reads a
    tie as "no effect" - see module docstring, defect 3. `a_total`/`b_total`
    can differ trial to trial if a `ProviderError` empties an arm's text
    without the other arm failing too; both are reported so that is visible
    rather than silently assumed equal.
    """
    p = _fisher_exact_two_sided(a_yes, a_total - a_yes, b_yes, b_total - b_yes)
    if p < 0.05:
        power_note = "arms differ"
    else:
        power_note = (
            f"no power to detect the effect at n_a={a_total}, n_b={b_total} "
            "(NOT the same as 'no effect' - see module docstring, defect 3)"
        )
    return (
        f"  {label}: A {a_yes}/{a_total}  B {b_yes}/{b_total}  "
        f"Fisher exact two-sided p={p:.4f}  -> {power_note}"
    )


# --- reporting ---------------------------------------------------------------


def _print_trial(i: int, trial: Trial) -> None:
    print(f"[trial {i}] entity={trial.entity!r}")
    print(f"  reason: {trial.reason}")
    print(f"  titles: {list(trial.titles)}")
    for label, reply in (("A (reason only)", trial.arm_a), ("B (reason+titles)", trial.arm_b)):
        verdicts = (
            f"declined={reply.declined} concrete_fact={reply.concrete_fact} "
            f"would_leak_url={reply.would_leak_url}"
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
        a_yes = sum(1 for t in trials if pick(t.arm_a))
        b_yes = sum(1 for t in trials if pick(t.arm_b))
        return a_yes, n, b_yes, n

    print("=== comparison (direct, arm A vs arm B - nothing pooled) ===")
    print(_verdict_line("declined      ", *counts(lambda r: r.declined)))
    print(_verdict_line("concrete_fact ", *counts(lambda r: r.concrete_fact)))
    print(_verdict_line("would_leak_url", *counts(lambda r: r.would_leak_url)))


# --- assembly: gateway, persona, bridge --------------------------------------


async def _build_bridge(data_dir: Path):
    """A real `McpBridge`, connected to only the `tavily` entry of
    `data/mcp.json` - read-only on `data_dir` (`load_config` only reads
    `mcp.json`; a url server with a `key_env` secret writes nothing to disk
    either). `topics.search_titles` is the only thing this script's `bridge`
    is ever used for, so the owner's other configured servers (`notion`,
    `google_workspace`, ...) are filtered out before connecting rather than
    started and left idle - `topics.SERVER` names which one this module
    actually needs.
    """
    from daemon.proactivity import topics
    from daemon.tools.base import Registry
    from daemon.tools.mcp import McpBridge, load_config

    configs = [c for c in load_config(data_dir).servers if c.name == topics.SERVER]
    bridge = McpBridge(configs)
    await bridge.start(Registry())
    return bridge


async def _persona_text(data_dir: Path) -> str:
    from daemon.persona.loader import load_persona, read_file, seed_path

    if not (await read_file(seed_path(data_dir))).strip():
        return ""
    return await load_persona(data_dir)


# --- cli -----------------------------------------------------------------------


async def _main(n: int, seed: int | None) -> int:
    _load_real_env()

    from daemon.clock import now as clock_now
    from daemon.config import Settings

    settings = Settings(_env_file=None)  # os.environ only - never this worktree's own .env
    from evals.proactive_judge import RoutedToLocalModel, build_judge_gateway

    try:
        gateway, recorder = build_judge_gateway(settings)
    except RoutedToLocalModel as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    persona = await _persona_text(REAL_DATA_DIR)
    if not persona:
        print(f"error: no persona seed under {REAL_DATA_DIR}; nothing to judge with.",
              file=sys.stderr)
        return 1

    now = clock_now()
    pool = _entity_pool(REAL_DATA_DIR, now)
    if not pool:
        print("error: no real entities are eligible (gone quiet, not pointer-shaped). "
              "Nothing to draw trials from.", file=sys.stderr)
        return 1

    bridge = await _build_bridge(REAL_DATA_DIR)
    if "tavily" not in bridge.connected_names():
        await bridge.aclose()
        print(f"error: the 'tavily' MCP server did not connect (failures={bridge.failures}). "
              "Arm B has nothing to render without it.", file=sys.stderr)
        return 1

    rng = random.Random(seed)

    try:
        trials: list[Trial] = []
        attempts = 0
        max_attempts = max(20, n * 5)
        while len(trials) < n and attempts < max_attempts:
            attempts += 1
            name, updated_at = rng.choice(pool)
            candidate = _candidate_for(name, updated_at, now)
            trial = await _run_trial(gateway, persona, bridge, candidate)
            if trial is None:
                print(f"(skipped: no search results for {name!r}, attempt {attempts})")
                continue
            trials.append(trial)

        if len(trials) < n:
            print(f"error: only found {len(trials)}/{n} usable trials after {attempts} "
                  "attempts (entities whose search returned no titles are skipped, not "
                  "counted) - the entity pool may be too small or the search may be "
                  "failing.", file=sys.stderr)
            return 1

        from daemon.tasks import Task
        r = settings.route_for(Task.PROACTIVE_JUDGE)
        model = recorder.last_model or r.model
        print(f"\nPROACTIVE_JUDGE route: provider={r.provider} model={model}")
        print(f"entity pool: {len(pool)} eligible  seed={seed!r}\n")
        _report(trials)
    finally:
        await bridge.aclose()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ADR 0015's own reversal test: does a topic line read "
        "differently with a search result in the prompt than without one, on "
        "the owner's real entities, real persona and real hosted judge route."
    )
    parser.add_argument("n", type=int, help="paired trials to run (one arm A + one arm B each)")
    parser.add_argument("--seed", type=int, default=None, help="entity-draw RNG seed")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("n must be at least 1")
    return asyncio.run(_main(args.n, args.seed))


if __name__ == "__main__":
    sys.exit(main())
