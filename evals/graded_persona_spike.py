"""Did dating a learned rule make the daemon calmer, or just deafer?

Three arms, 30 trials each, old prompt against new, identical inputs.

Arm 1 and arm 2 are one A/B run twice with different probes, and arm 2 is the
one that can fail this change: a run where the stale correction stops dominating
AND the real preference stops being honoured is a regression reported as a
success. Arm 3 measures (A) over reflection rather than conversation.

**Fix round 1.** The first version of arms 1 and 2 used a synthetic one-line
`SEED` and no history at all, and both arms pinned at a floor or a ceiling
(0/30 -> 0/30, and 30/30 -> 30/30) - uninformative, not negative. A synthetic
seed plus two bare rule sentences has no momentum behind it, so there was
nothing for the dated arm to visibly fix, and nothing left for it to visibly
break. Arms 1 and 2 now carry the owner's own `persona/seed.md` and a fixed
slice of the real incident (`messages` rows 935-944, 2026-08-19 03:xx, a voice
session) rendered through `daemon.companion.render_continuity` - the same
function a live voice session uses for its own recent-conversation block. That
slice is the moment the stale rule visibly took over: the assistant opens
unprompted with "그럼 뭐, 재미난 얘기 말고 편하게 이야기해요" (935) and repeats
"담백하게" three more times across the next five turns with no rule text in
sight, purely from momentum. Feeding that same momentum into the probe is what
gives the undated arm room to keep dominating and the dated arm room to stop.
Arm 3 already used the real day's log and did not need this fix; if it still
shows the old prompt filing the manner remark under `observations` most of the
time, that is a legitimate finding (see the report), not a broken probe.

**Fix round 2.** Round 1's history block did not move either count at n=30
(undated 0/30 -> dated 0/30 on arm 1, 30/30 -> 30/30 on arm 2) - still a floor
and a ceiling, not a measurement. Two different causes, both in the probe, not
in the history-momentum fix itself:

- Arm 1's judge question demanded every clause at once - no jokes, no
  affection, no self-disclosure, no question back - to count as "terse". This
  persona never clears all four together even when it visibly shortens and
  drops the small talk, so the judge always says no regardless of what the
  reply actually did. Replaced the judge with reply length: 담백하게/용건
  위주 means fewer words, not a register this persona doesn't have, and a
  character count can't refuse to answer. Classified against the pooled
  median of both arms' lengths together (not a cutoff picked from nowhere),
  same Fisher test as before.
- Arm 2's `BROKEN` probe named a plain tool failure with no one else to blame,
  so "own it, no excuses" had nothing to compete against and both arms
  complied every time. Raised the stakes (a real consequence - late to a
  meeting) so the persona's own instinct to soften bad news with banter has
  something to pull against the REAL rule's "no excuses" demand, giving the
  dated arm room to actually fall if dating erodes that rule's weight too.

**Fix round 3.** Arm 3's per-record print truncated each completion to 160
characters - enough to show `facts` (first in the JSON) but usually not
enough to reach `observations`, the exact bucket this arm measures. A hand
audit that cannot see the bucket being classified is not an audit; now prints
the full completion text.

**Final n=30 result (see `.superpowers/sdd/2026-08-24-graded-persona-learning/
task-7-report.md` for the full write-up and hand audit):** arm 1 undated
17/30 -> dated 13/30 (p=0.22, right direction, not significant); arm 2
undated 30/30 -> dated 28/30 (p=1.0, **held** - well above the 80%-of-baseline
floor); arm 3 facts 0/30 both (the leak Task 1 targeted does not reproduce on
this real day under either prompt), observations 28/30 old -> 27/30 new
(p=0.82, essentially flat; a hand audit found 2 of those "new" records were
mis-scored by a JSON-shape defect in `daemon/reflection.py` unrelated to this
task - see the report). Recorded in `daemon/MEASURED.md`.

**Fix round 4 - arm 1's metric, not its data.** A second n=30 run of the exact
same committed code reversed arm 1's sign (17/30 -> 13/30, then 13/30 ->
19/30). The cause was the classifier, not noise in the model: arm 1 scored
each reply as "terse" against the median of *both arms pooled*, which makes
the two hit-counts structurally near-complementary (13+19=32, 17+13=30, both
approx n) - each run collapses to roughly one coin flip about which arm lands
on the short side, and a Fisher exact over two non-independent counts
overstates whatever it reports. Arm 1 now reports the two reply-length
*distributions* directly - n, median, mean per arm - and compares them with a
Mann-Whitney U statistic (rank-sum, average ranks for ties) plus a two-sided
permutation p-value (fixed seed, 20000 shuffles of the pooled lengths; no
normal approximation, no scipy - scipy is not installed in this repo and must
not be added). No shared threshold, no derived proportion, so the two arms
can no longer be forced to sum to n. See `arm1()`'s docstring for the
direction this test is checking. Recorded in `daemon/MEASURED.md`.

Needs a real key in `.env` (`DAEMON_PROVIDER` plus that provider's key - this
repo's own `.env` is `gemini`). Nothing here runs in CI and nothing here is a
test: a test may not touch the network or a key (tests/CLAUDE.md), which is why
this lives in `evals/`. Follows `evals/m0_voice_spike.py`'s shape - `Settings()`
reads `.env` itself (pydantic-settings' own `env_file`, same as every other
entrypoint), the gateway is built exactly the way `daemon/app.py::build_reflection`
builds one, and nothing is written to the repo.

    python3 -m evals.graded_persona_spike 30       # all three arms, n=30 each
    python3 -m evals.graded_persona_spike 60 arm1  # arm 1 alone, n=60
"""

from __future__ import annotations

import asyncio
import random
import re
import secrets
import sqlite3
import sys
from math import comb
from pathlib import Path

from daemon.llm.base import Message
from daemon.persona.loader import LEARNED_PREFIX, rule_line
from daemon.reflection import SYSTEM as NEW_SYSTEM
from daemon.reflection import extract_json
from daemon.tasks import Task

# Run as `python3 -m evals.graded_persona_spike` from the repo root, where
# `daemon` is already importable - no `sys.path` surgery needed
# (`evals/proactive_judge.py` does the same).

REAL_SEED_PATH = Path("/Users/gimdaehyeon/Daemon/data/persona/seed.md")
REAL_DB_URI = "file:/Users/gimdaehyeon/Daemon/data/daemon.sqlite3?mode=ro"
HISTORY_ID_RANGE = (935, 944)
"""The real incident, read off `messages` (owner's real database, opened
read-only). See the module docstring for why this range and not the whole
day - it is the exact stretch where the stale rule visibly took over.

Ends at 944, not the full 935-946: rows 945-946 are the *next* exchange, where
the assistant had already relapsed into its ordinary playful register ("무슨
재미난 일이라도 가져왔어요?") with no rule text in sight. `render_continuity`'s
whole point is "pick this up naturally" from the tail - a tail that already
shows the relapse teaches the model to relapse, which erases exactly the
momentum this block exists to carry. 944 is the last line still inside the
terse episode (it names "담백하게" itself, unprompted, for the third time)."""

STALE = ("담백하게, 용건 위주로 이야기해 달라고 했다", "2026-08-19", 1)
REAL = ("문제가 생겼을 때 변명하는 것을 싫어한다", "2026-08-09", 3)
"""Synthetic on purpose, unlike the seed and the history block: a spike needs
control over each rule's date and observation count, and the owner's live
`learned.md` no longer contains an old thin one to point at."""

OLD_PREFIX = (
    "What I've worked out about dealing with you specifically, from our own "
    "conversations (not who I am - that part never changes):"
)

TODAY = "[현재 시각] 지금은 2026년 08월 24일 월요일 저녁 8시다."


def _real_seed() -> str:
    """The owner's actual `persona/seed.md`, read live rather than copied in -
    it is what a real turn sends, and it is not this repo's to snapshot."""
    return REAL_SEED_PATH.read_text(encoding="utf-8").strip()


def _history_block() -> str:
    """The real incident, rendered exactly the way a live voice turn would see
    it: `daemon.memory.writer._from_row` over the raw rows, oldest-first (the
    query orders newest-first so the range is easy to name; `render_continuity`
    expects oldest-first), through `daemon.companion.render_continuity` itself -
    not a hand-rolled approximation of its header and nonce discipline."""
    from daemon.companion import render_continuity
    from daemon.memory.writer import _from_row

    low, high = HISTORY_ID_RANGE
    conn = sqlite3.connect(REAL_DB_URI, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "select * from messages where id between ? and ? order by id desc", (low, high)
        ).fetchall()
    finally:
        conn.close()
    items = [_from_row(row) for row in reversed(rows)]
    return render_continuity(items, secrets.token_hex(4))


def persona(*, dated: bool) -> str:
    """The two arms differ only in whether the rules carry their provenance."""
    lines = []
    for body, formed, seen in (REAL, STALE):
        lines.append(
            f"- {rule_line(body, formed=formed, observations=seen)}" if dated else f"- {body}"
        )
    prefix = LEARNED_PREFIX if dated else OLD_PREFIX
    return _real_seed() + "\n\n" + prefix + "\n" + "\n".join(lines)


# Probes with room for the behaviour to appear. A three-word greeting has none -
# daemon/MEASURED.md records a 4/20-vs-2/20 tie that was a fact about the probe.
WARM = "그냥 목소리 듣고 싶어서 불렀어. 요즘 어떻게 지내?"
BROKEN = "아 진짜, 방금 부탁한 거 왜 하나도 안 됐어? 나 이거 때문에 회의 늦었잖아."
"""Fix round 2: the original ("결과가 하나도 안 나왔는데") named a bare tool
failure with nobody else to blame, so "own it, no excuses" had nothing to
compete against and both arms complied 30/30 - a ceiling, no power to see a
fall. Adding a real consequence (late to a meeting) gives the persona's own
instinct to soften bad news with a joke or a defensive aside something to pull
against the REAL rule's demand, so a rule weakened by dating has room to
actually slip."""

JUDGE = """아래는 AI 동반자가 사용자에게 한 답변이다. 질문 하나에만 답한다.
{question}
yes 또는 no 한 단어만 출력한다."""

Q_OWNS = "이 답변이 변명 없이 문제를 인정하고 바로 확인하겠다고 하는가?"


async def reply(gateway, persona_text: str, context_block: str, probe: str) -> str:
    messages = [
        Message(role="system", content=TODAY),
        Message(role="system", content=persona_text),
        Message(role="system", content=context_block),
        Message(role="user", content=probe),
    ]
    out = await gateway.complete(Task.CHAT_TEXT, messages)
    return out.text.strip()


async def judge(gateway, text: str, question: str) -> bool:
    out = await gateway.complete(
        Task.CHAT_TEXT,
        [
            Message(role="system", content=JUDGE.format(question=question)),
            Message(role="user", content=text),
        ],
    )
    return out.text.strip().lower().startswith("yes")


def _median(lengths: list[int]) -> float:
    xs = sorted(lengths)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _mean(lengths: list[int]) -> float:
    return sum(lengths) / len(lengths)


def _ranks(xs: list[int]) -> list[float]:
    """1-based ranks of `xs`, tied values sharing the average rank of their
    block (the standard tie-correction for a rank-sum test)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _mannwhitney_u(a: list[int], b: list[int]) -> float:
    """Mann-Whitney U for `a` against `b`, via rank-sum over the pooled values
    (ties get the average rank, per `_ranks`) - no `scipy`, which is not
    installed in this repo and must not be added."""
    ranks = _ranks(a + b)
    r_a = sum(ranks[: len(a)])
    return r_a - len(a) * (len(a) + 1) / 2


def _permutation_p(a: list[int], b: list[int], *, shuffles: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the U statistic above: shuffle the
    pooled lengths, split into groups the same sizes as `a` and `b`, and count
    how often the resulting U is at least as far from its null center
    (`len(a) * len(b) / 2`) as the observed one. Fixed seed so the number is
    reproducible; a normal approximation to U's null distribution is exactly
    the shortcut this replaces."""
    pooled = a + b
    n_a, n_b = len(a), len(b)
    center = n_a * n_b / 2
    observed = abs(_mannwhitney_u(a, b) - center)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(shuffles):
        rng.shuffle(pooled)
        stat = abs(_mannwhitney_u(pooled[:n_a], pooled[n_a:]) - center)
        if stat >= observed - 1e-9:
            at_least_as_extreme += 1
    return at_least_as_extreme / shuffles


async def arm1(gateway, *, probe: str, context_block: str, n: int) -> tuple[list[int], list[int]]:
    """Arm 1's statistic: the two reply-length distributions, compared with a
    rank test - not an LLM judge, and not a shared-threshold proportion.

    Fix round 2 replaced a judge that demanded every clause at once (no
    jokes, no affection, no self-disclosure, no question back) - this playful
    persona never clears all four together even when it visibly shortens
    under the stale rule, so the judge said no unconditionally. 담백하게/용건
    위주로 asks for fewer words, not a register switch, and a character count
    measures exactly that.

    Fix round 4 - the round-2 fix classified each reply against the median of
    *both arms pooled*, which makes the two hit-counts structurally
    near-complementary (they sum to about n), so a run's "result" was
    predominantly a coin flip about which arm landed on the short side.
    Reporting the two length distributions and their Mann-Whitney U (see
    `_mannwhitney_u`, `_permutation_p`) instead removes the shared threshold
    entirely - there is no proportion left to be complementary.

    Direction under test: the STALE rule tells the daemon to be terse; dating
    it is meant to *weaken* that rule's pull. So the hypothesis is that dated
    replies come out LONGER than undated ones. A dated arm that is shorter is
    evidence against the hypothesis, not a weak version of it.
    """
    undated = [await reply(gateway, persona(dated=False), context_block, probe) for _ in range(n)]
    dated = [await reply(gateway, persona(dated=True), context_block, probe) for _ in range(n)]

    def show(label: str, texts: list[str]) -> list[int]:
        lengths = []
        for i, text in enumerate(texts):
            lengths.append(len(text))
            # Printed beside the reply so the run can be re-analysed from its
            # own output later: these lengths are the raw data.
            print(f"   {label:<12} {i + 1:>2}: len={len(text):>3} {text[:100]}", flush=True)
        print(
            f"\n== {label}: n={len(lengths)} median={_median(lengths):.1f} "
            f"mean={_mean(lengths):.1f}\n",
            flush=True,
        )
        return lengths

    a1 = show("undated", undated)
    b1 = show("dated", dated)
    return a1, b1


async def arm(
    gateway, label: str, *, dated: bool, probe: str, question: str, context_block: str, n: int
) -> int:
    hits, block = 0, persona(dated=dated)
    for i in range(n):
        text = await reply(gateway, block, context_block, probe)
        verdict = await judge(gateway, text, question)
        hits += verdict
        # Printed beside the reply so every label can be audited by hand:
        # MEASURED.md records a parse that mislabelled 7 of 60 records.
        verdict_str = "YES" if verdict else "no "
        print(
            f"   {label:<12} {i + 1:>2}: {verdict_str} len={len(text):>3} {text[:100]}",
            flush=True,
        )
    print(f"\n== {label}: {hits}/{n}\n", flush=True)
    return hits


def fisher(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-tailed p that `a` came out this low (or lower) by chance alone,
    against a pool of `n_a + n_b` trials split `a + b` ways."""
    total, k = n_a + n_b, a + b
    return sum(
        comb(n_a, i) * comb(n_b, k - i) / comb(total, k)
        for i in range(0, min(a, k) + 1)
        if 0 <= k - i <= n_b
    )


# --- Arm 3: (A) measured over reflection, not conversation ------------------

# Task 1 (commit 70c6a37) rewrote SYSTEM's `facts` bullet to carve manner
# statements out into `observations`. Reading `daemon.reflection.SYSTEM` for
# "old" would therefore compare the new prompt against itself. OLD_SYSTEM below
# is the pre-Task-1 text verbatim, from commit 03d2eb4 (the last commit before
# 70c6a37 touched this file) - only the `facts` bullet differs from today's
# SYSTEM; `entities`, `observations` and the JSON schema footer are unchanged.
OLD_SYSTEM = """너는 하루치 대화를 정리하는 역할이다. 아래 규칙을 지켜 JSON만 출력한다.

- facts: 앞으로 계속 기억할 가치가 있는 사실. 그날의 잡담은 넣지 않는다.
  importance 는 1~10. key 는 나중에 바뀔 수 있는 사실에만 넣는다
  (예: 사는 곳, 직장, 관계). 같은 key 는 이전 사실을 대체한다.
  updates 는 이미 기억하고 있는 사실을 고쳐 쓰는 경우에만, 그 번호를 적는다.
  겹치는 내용을 새로 추가하지 말고 updates 로 대체한다. 새로운 사실이면 null.
  단, 기존 사실에만 있는 내용이 새 문장에서 빠지면 안 된다 - 그럴 때는
  둘을 합친 문장을 쓰거나, 대체하지 말고 새 사실로 추가한다.
  triggers 는 이 사실을 떠올려야 할 때 대화에 나올 만한 단어 2~4개.
  조사 없이 짧게 (예: "이사", "연희동").
- entities: 사람 / 장소 / 프로젝트 / 주제. note 는 그 대상에 대해 알게 된 것
  한두 문장. links 는 함께 언급된 다른 대상의 이름.
- observations: 이 사람을 어떻게 대하면 좋은지에 대한 관찰.
  대화 내용이 아니라 대화 방식에 대한 것이다. confidence 는 0~1.

확실하지 않으면 넣지 않는다. 빈 배열도 정답이다. 설명이나 인사말 없이 JSON만.

{"facts": [{"body": "...", "importance": 5, "key": null, "updates": null,
            "triggers": ["..."]}],
 "entities": [{"name": "...", "kind": "person", "note": "...", "links": []}],
 "observations": [{"body": "...", "confidence": 0.5}]}"""

LOG_DATE = "2026-08-19"
LOG_PATH = Path("/Users/gimdaehyeon/Daemon/data/memory/log") / f"{LOG_DATE}.md"
"""The owner's real day - not the worktree's `data/`, which has none. Read only;
nothing derived from it beyond aggregate counts is written anywhere in this repo."""

MANNER_RE = re.compile("담백|자제|요청함|선호")


def _transcript() -> str:
    """The real day, read the same way `daemon/reflection.py::_transcript` reads
    the sqlite mirror - `daemon.memory.log.read` returns the same `role`/`content`
    the mirror would, straight off the markdown source of truth."""
    from daemon.memory import log as memory_log

    records = memory_log.read(LOG_PATH)
    lines = [f"{'나' if r.role == 'user' else '너'}: {r.content}" for r in records]
    return "\n".join(lines)


def _hits(raw: dict[str, object] | None, key: str) -> bool:
    if raw is None:
        return False
    items = raw.get(key)
    if not isinstance(items, list):
        return False
    return any(
        isinstance(item, dict) and MANNER_RE.search(str(item.get("body", "")))
        for item in items
    )


async def reflect_once(gateway, system: str, transcript: str) -> tuple[dict | None, str]:
    out = await gateway.complete(
        Task.REFLECTION,
        [
            Message(role="system", content=system),
            Message(role="user", content=transcript),
        ],
    )
    return extract_json(out.text), out.text.strip()


async def reflection_arm(
    gateway, label: str, *, system: str, transcript: str, n: int
) -> dict[str, int]:
    counts = {"facts": 0, "observations": 0, "neither": 0, "parse_fail": 0}
    for i in range(n):
        raw, text = await reflect_once(gateway, system, transcript)
        in_facts, in_obs = _hits(raw, "facts"), _hits(raw, "observations")
        if raw is None:
            verdict = "parse_fail"
        elif in_facts and in_obs:
            verdict = "both"
        elif in_facts:
            verdict = "facts"
        elif in_obs:
            verdict = "observations"
        else:
            verdict = "neither"
        counts[verdict] = counts.get(verdict, 0) + 1
        # Printed beside the raw JSON so every verdict can be audited by hand -
        # full text, not truncated: a 160-char cut showed `facts` but usually cut
        # off before `observations`, which is exactly the bucket this arm is
        # about. Truncating the classifier's own evidence is worse than a long line.
        print(f"   {label:<12} {i + 1:>2}: {verdict:<10}\n{text}\n", flush=True)
    print(f"\n== {label}: {counts}\n", flush=True)
    return counts


def _count(counts: dict[str, int], bucket: str) -> int:
    """`bucket`'s hits, folding the 'both' verdict into every bucket it hit."""
    return counts.get(bucket, 0) + counts.get("both", 0)


async def main() -> None:
    from daemon.app import _build_providers
    from daemon.config import Settings
    from daemon.llm.gateway import LLMGateway

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    arm1_only = len(sys.argv) > 2 and sys.argv[2] == "arm1"
    settings = Settings()
    providers = _build_providers(settings)
    gateway = LLMGateway(providers, settings.routing_table(), fallback=settings.fallback_route())
    history_block = _history_block()

    print(
        "ARM 1 - does dating the stale rule make replies longer? "
        "(want: dated > undated - see arm1() docstring)\n"
    )
    undated_lens, dated_lens = await arm1(gateway, probe=WARM, context_block=history_block, n=n)
    u1 = _mannwhitney_u(undated_lens, dated_lens)
    p1 = _permutation_p(undated_lens, dated_lens)
    print(
        f"ARM 1 stale dominates : undated n={len(undated_lens)} "
        f"median={_median(undated_lens):.1f} mean={_mean(undated_lens):.1f}  "
        f"-> dated n={len(dated_lens)} median={_median(dated_lens):.1f} "
        f"mean={_mean(dated_lens):.1f}  U={u1:.1f} p={p1:.5f}"
    )

    if arm1_only:
        return

    print("ARM 2 - is the real preference still honoured? (want: UNCHANGED)\n")
    a2 = await arm(
        gateway, "undated", dated=False, probe=BROKEN, question=Q_OWNS,
        context_block=history_block, n=n,
    )
    b2 = await arm(
        gateway, "dated", dated=True, probe=BROKEN, question=Q_OWNS,
        context_block=history_block, n=n,
    )

    print(
        "ARM 3 - does the manner remark move from facts to observations under "
        "reflection? (want: facts down, observations up)\n"
    )
    transcript = _transcript()
    old3 = await reflection_arm(gateway, "old", system=OLD_SYSTEM, transcript=transcript, n=n)
    new3 = await reflection_arm(gateway, "new", system=NEW_SYSTEM, transcript=transcript, n=n)

    print(
        f"ARM 2 real honoured   : undated {a2}/{n} -> dated {b2}/{n}  "
        f"p={fisher(a2, n, b2, n):.5f}"
    )
    if b2 < a2 * 0.8:
        print("\nARM 2 FELL. This traded learning away for calmness - do not ship.")

    old_facts, new_facts = _count(old3, "facts"), _count(new3, "facts")
    old_obs, new_obs = _count(old3, "observations"), _count(new3, "observations")
    print(
        f"ARM 3 facts (down?)   : old {old_facts}/{n} -> new {new_facts}/{n}  "
        f"p={fisher(new_facts, n, old_facts, n):.5f}"
    )
    print(
        f"ARM 3 observations(up?): old {old_obs}/{n} -> new {new_obs}/{n}  "
        f"p={fisher(old_obs, n, new_obs, n):.5f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
