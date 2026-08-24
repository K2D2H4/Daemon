# Graded persona learning — design

- Date: 2026-08-24
- Status: approved for planning
- Branch of origin: `claude/bella-natural-speech-0e4522`

## Problem

The owner's complaint was not that his daemon learns. Learning is the product
(docs/PLAN.md §5). It was that the learning is **too extreme to read as human**:

> 사실 이런 자연스러운 학습 및 변화가 의도한거긴한데 너무 극단적인 느낌이라
> 사람같은 느낌이 안드는게 문제인거 같아

On 2026-08-19 he said once, in the middle of a voice call, that he was tired of
being asked `무슨 재미난 얘기 있어요?`. Five days later the daemon addresses him as
`창조주님`, apologises in the register of a servant (`실망시켜 드려 죄송합니다.
앞으로는 더 꼼꼼히 보도록 할게요`), and volunteers nothing. The seed — playful,
teasing, `툭툭 던지듯` — was restored byte-for-byte on 08-19 and is not the cause.

A person told "stop asking me that" stops asking. They do not become a different
person, and they do not still be that person a week later.

## Evidence (measured, not argued)

### The change did not come through the path built to rate-limit it

M4 already has a change-rate policy, and it is a good one:

| | `persona/learned.md` (M4 weekly evolution) | `memory/core.md` (nightly reflection) |
|---|---|---|
| cadence | weekly | **nightly** |
| evidence required | ≥5 unconsumed observations; prompt says *관찰 하나만 보고 규칙을 만들지 않는다* | **none — one day** |
| new per pass | ≤3 (`max_new`) | **≤8 facts** (`MAX_FACTS`) |
| total cap | 20 (`max_active`) | **none** |
| retraction | `daemon persona forget <id>` | **none exists** |
| always in the prompt | yes | **yes** |

The sentence that changed her personality is in the **right-hand column**:

```
data/memory/core.md:18
- 사용자가 AI 비서에게 반복적인 질문('재미있는 일 없었냐' 등)을 자제하고
  담백하게 대화해 줄 것을 요청함
```

`core.md` is injected whole on every turn. One remark became a standing order,
through the path with no evidence threshold, no cap, no decay and no retraction.

### And it was misfiled, not mis-learned

`daemon/reflection.py`'s prompt already has the right bucket:

> `observations`: 이 사람을 어떻게 대하면 좋은지에 대한 관찰.
> **대화 내용이 아니라 대화 방식에 대한 것이다.**

`observations` is what feeds M4's rated path. The model put a manner instruction
in `facts` instead, and thereby routed a behavioural correction **around** every
gate M4 has. Set beside its neighbours the misfiling is plain:

```
- 사용자는 두 살 된 수컷 빠삐용 강아지 'Kiwi(키위)'를 키우고 있다.
- 사용자의 생일은 2월 24일이다.
- 사용자가 AI 비서에게 … 담백하게 대화해 줄 것을 요청함     ← not the same kind
```

`core.md:10` (`사용자는 개발자(AI/LLM 비서 벨라의 창조주/개발자)이다.`,
importance 9) is a genuine fact, but she reads it as a form of address. It is the
source of `창조주님` and is treated separately in §"Cleanup" below.

### Even the rated path states its conclusions as absolutes

`daemon/persona/evolve.py`'s prompt instructs:

> body 는 이 사람에 대한 **사실처럼** 짧게 한 문장으로 적는다

A fact carries no strength and no date, so it is read at full weight forever. What
that produced, one day after five days of terse terminal QA:

```
[2] 잡담이나 의미 없는 반복 테스트, 할루시네이션에 거부감을 느끼며,
    용건 위주의 빠른 응답과 즉각적이고 담백한 피드백을 요구한다.
    created 2026-08-23T20:00:00Z - 3 observation(s)
```

Three observations, all from one debugging week, rendered as a standing demand
that governs a midnight voice chat as much as a terminal session.

## Approach

Two changes, in order. A third is deliberately deferred.

### (A) Seal the leak: `facts` may not carry manner

`reflection.py`'s prompt draws the boundary explicitly — `facts` are about the
person's life and world; anything about tone, manner, address, or what to stop or
start saying goes to `observations`. Nothing else changes: `observations` already
flows to M4, so the correction inherits the weekly cadence, the two-observation
floor, the cap of 20, and `daemon persona forget`.

This alone would have prevented what happened.

### (B) Dated observations instead of undated conclusions

Two edits, no schema change.

**The prompt.** Drop `사실처럼`. A rule body becomes an observation of a
tendency, and the standing-demand form (`~를 요구한다`) is ruled out by name — the
same lesson as `CALLED_BY_NAME`, where omitting the unwanted move was not enough
and it had to be forbidden explicitly.

**The rendering.** Each bullet in `learned.md` carries the date it was formed and
how many observations stand behind it:

```
- 2026-08-09 (관찰 3건) 시스템 오류를 인정할 때 변명을 싫어한다
- 2026-08-23 (관찰 3건) 용건 위주의 빠른 응답을 선호했다
```

**Absolute dates, never relative ones.** A relative phrase written into a file is
a lie by the following week; a date is not, and `[현재 시각]` is already in every
prompt, so the model computes recency itself. Markdown stays the source of truth,
`persona_rules` needs no new column, and `loader.py` keeps reading the file alone.

`LEARNED_PREFIX` is rewritten to match: these are dated observations to be weighed
by recency and repetition, not rules to be obeyed.

### (C) Deferred: strength and decay

Rules carrying a weight that decays unless reinforced, retiring themselves below a
floor. It is the most human of the three and the most expensive — a schema change
and a decay pass. (B) may deliver most of the effect for none of the cost, and
until (B) is measured there is no way to know. The open question that governs (C)
— how long a single remark should keep mattering — is left unanswered on purpose.

### Hazard the implementer will hit

Annotating the bullets **silently stops the weekly pass**, and the error it
produces sends you the wrong way. `evolve.py`'s gate 4 compares the file against
the mirror verbatim:

```python
orphaned = diverged_bodies(rule_bodies(learned_text), (row["body"] for row in active_rows))
```

A bullet rendered as `- 2026-08-09 (관찰 3건) …` matches no `persona_rules.body`,
so **every** rule reads as orphaned, the pass returns
`skipped="learned.md has N rule(s) the mirror does not know about"` — and its
advice, `run daemon reindex`, does not repair this and never will. Persona
evolution would be dead with a plausible explanation on the console, which is this
repo's signature defect.

So the annotated line and the stored body must be separable, and the two readers
want different halves:

| reader | wants |
|---|---|
| gate 4 / `diverged_bodies` | the clean body, to compare against the mirror |
| `load_persona` → the prompt | the annotated line, dates and counts included |

One parser, two accessors — not one function whose result is right for whichever
caller was written first. `tests/test_persona_rules.py` and `tests/test_persona_evolve.py` must cover the round-trip:
render → read back → the body still equals what the mirror holds, and the weekly
pass still runs against a freshly rendered file.

## Measurement

**"Less extreme" must not become "ignores what it learned".** That is the failure
this design can most easily cause, so both arms are measured and shipping requires
both to hold. The owner's 08-19 → 08-24 log is the material; old prompt and new
prompt see identical observations, 30 trials per arm.

| what is measured | required direction |
|---|---|
| a stale single-remark correction (`담백하게`, 08-19) dominating the reply | **down** |
| a repeatedly-observed real preference (dislikes excuses when something breaks) still honoured | **unchanged** |

A run where the first falls and the second falls with it is a regression reported
as a success, and is the specific outcome to refuse.

For (A), the measurement is over reflection rather than conversation: feed the
08-19 day and count how often the manner remark lands in `facts` versus
`observations`, old prompt against new.

Both are LLM behaviour changes, so a tie is a result about power, not about the
mechanism (`daemon/MEASURED.md`: a 4/20-vs-2/20 tie on a probe with no room for
the behaviour was the wrong conclusion, not a null one). Pick a probe with room,
and report a p-value.

## Cleanup (owner's data — confirm before running)

Fixing the prompts does not remove what is already written:

1. `daemon persona forget 2 --why "..."` — the learned rule. CLI exists.
2. `memory_entries` id 12 (`담백하게 … 요청함`) and the matching `core.md` line.
   **No CLI exists for this.** The schema has `status='retired'`, nothing exposes
   it, so today this is a manual edit — and `core.md` is rewritten whole by the
   nightly pass, so the line can return while the evidence sits in the log.
3. `창조주님`: `core.md:10` is a true fact and should stay. The form of address
   belongs in the one file nothing overwrites, so `seed.md`'s
   `How I address the user` line pins it — the owner's edit to make, not the
   daemon's.

**Item 2 names a real gap: a curated fact cannot be retracted.** `persona forget`
exists for learned rules and has no counterpart for the always-injected tier. Out
of scope here, recorded so it is a line someone chose to leave open.

## Out of scope

- `delegate_task` has never been called in the whole history (`daemon/MEASURED.md`).
- The verbal-tic detector is calibrated for Korean only (PR #104 review).
- A `daemon memory forget <id>` CLI (§Cleanup item 2).
