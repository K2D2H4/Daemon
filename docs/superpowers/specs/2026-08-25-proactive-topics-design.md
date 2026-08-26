# Proactive topics — design

- Date: 2026-08-25
- Status: approved for planning
- Branch of origin: `claude/graded-persona-learning`

## Problem

Proactivity has never once spoken.

```
572   proactive_judge LLM calls, all time
  0   messages with session_kind='proactive'
```

The owner's report was `아직도 먼저 말거는건 제대로 동작을 안하는듯`, then
`판정기가 너무 빡쎈거 같아`. The second half is wrong, and the repo already
measured why (docs/PLAN.md §8.2): the judge answers `open_loop` reasons **26/26**
and `silence` reasons **20/20** with "nothing to say". It is not strict. It is
starved.

Run the real generators over the live database, 7 days, 75 owner utterances:

| generator | fires |
|---|---|
| `open_loop` — a day marker **and** an event word in one owner sentence | **3/75**, all on 08-18/08-19 |
| `emotional` — an emotion word | **0/75** |
| `silence`, `pattern_time` | reason contains only elapsed hours; correctly declined |
| `association` | surfaces the owner's own command history; judged 20/20 as worthless |

The candidate table agrees: the newest `open_loop` is 2026-08-19, six days
stale. One generator works, and it fires only when the owner happens to mention a
dated future event in chat. He talks to this daemon in imperatives about tools.

**Loosening the judge cannot fix this.** A `silence` reason contains
`마지막 대화가 30시간 전이다` and nothing else; the only line derivable from it is
`오랜만이야, 요즘 어때?` — which is the same empty opener the owner asked to have
removed at the start of this same conversation. The fix is material, not a looser
gate.

## What the owner asked for

> 좀 주도적으로 대화를 여러 토픽에대해 대화를 먼저 꺼냈으면 좋겠어
> 웹서치도 좀 활용하고 해서 내가 평소에 관심이 있는 주제에 대해
> 하루에 3~4번은 말걸면 좋을듯

**The original design already wanted this.** `daemon/config.py` on the kind
budgets:

> the cheap kind to generate (`open_loop`) eats the budget on equal terms and
> **turns a companion into a reminder app**, and the Her feeling comes from the
> kinds with **no business to transact**. So the two businessless kinds get the
> most room.

The businessless kinds were given the most room and then never built. This spec
builds them.

## The contract change

docs/CONTRACTS.md non-negotiable 10 currently reads *"No tool runs on a turn
whose origin is not `owner`"*, and its rationale is that recall replays arbitrary
old text, so `"look at this message"` must not become a way to hand a stranger a
shell.

That rule conflates two things. **Only the second one changes:**

| | before | after |
|---|---|---|
| the **model** choosing and running a tool on a non-owner turn | forbidden | **still forbidden** — the judge stays `tools_offered=0`, asserted by test |
| **deterministic code** calling one read-only search on the proactive path | forbidden | **allowed** |

The model does not decide whether to search, and does not decide what to search
for. Code issues the query, and the query is an entity name read out of
`entities.name`. Nobody is handed a shell.

This is written down rather than done quietly, as
[docs/CONTRACTS.md](../../CONTRACTS.md) requires of a frozen-contract change, and
it gets an ADR: four ADRs in this repo were later overturned by measurement, so
why this boundary moved has to survive the person who moved it.

### What defends the new surface

Untrusted web text reaching an unprompted line that gets spoken aloud is the
worst surface this product has. Six defences, and the fourth is the one that
matters most:

1. **The query is first-party only** — `entities.name`, never web text, never
   model output.
2. **Results reduce to titles**, at most 3, each capped at 80 characters. Page
   bodies never enter.
3. **Fenced under a nonce** and marked as reference material that is never an
   instruction — the same discipline `render_recall` already uses.
4. **A URL in the utterance is a decline.** This kills the vector that actually
   matters: the daemon's trusted voice telling the owner to visit somewhere. The
   judge's output is already capped at `MAX_CHARS` and already declined unless it
   is `{"say": ...}`; this is one more filter on the same choke point.
5. **The judge is offered no tools**, unchanged, with a test that fails if that
   ever becomes true.
6. **One search per gate-passed candidate**, never per tick. Non-negotiable 7's
   cost shape — deterministic generation, deterministic gate, then exactly one
   expensive step — is preserved, and the search rides in the same slot the one
   LLM call already occupies.

If the search fails, errors, or is disabled, the `topic` candidate is dropped and
the other three generators are unaffected.

## The four generators

| kind | material | search |
|---|---|---|
| `topic` | `entities.name` + `entities.updated_at` | **yes — the only one** |
| `calendar` | event times from the Google MCP server | reads the Google server — the same kind of external call under a different name; see the plan's Scope |
| `weather` | a condition code, only when it changes the day (rain, snow, cold) | no |
| `diary` | last night's reflection | no |

`topic` is the primary engine, not a fallback. It picks the entity whose
`updated_at` is oldest — the thing gone quiet longest — so variety falls out of
the ordering and needs no separate rule. The owner's 11 entities today are
`Sendbird`, `UJET`, `UJET.cx`, `Emil Kowalski`, `llm-wiki`, `Kiwi`, `Scale Out`,
`ReadyTalk`, `Daemon`, `Schubert Chin`, `김대현`.

**A `topic` candidate with no search result is dropped, not sent.** This is where
the empty opener would come back: four topic candidates a day with nothing behind
them is `Sendbird 요즘 어때요?` four times, which is `재미난 얘기 있어요?` wearing a
different noun. Having something to say is the admission ticket.

## Frequency

The owner asked for 3-4 a day. **The budget was never the constraint** — it is
already 8/day with a 30-minute cooldown, against 0 actual utterances.

| | now | after |
|---|---|---|
| `proactive_daily_budget` | 8 | **5** |
| `proactive_cooldown_minutes` | 30 | **90** |
| quiet hours | 23:00–09:00 | unchanged |

**No per-kind quotas are added.** An earlier draft allocated `topic` 4,
`calendar` 2, `weather` 1, `diary` 1; the owner rejected it as artificial and he
is right — a person does not hold a daily quota of weather remarks. `config.py`
already says these are *"ceilings, not allocations, and the total is what
binds"*. Whichever kind actually has material that day wins the slot: an
interview day is calendar-heavy, a quiet day is topics.

The real regulator already exists and is the owner's: a 👎 rests that kind for 6
hours, two in 24 hours rests it longer, three in a day stops everything. Tuning
comes from labels, not from numbers chosen in advance — which is also why the
labels clock in PLAN §8.1 has never started: it cannot, at zero utterances.

The kind names stay. They are not decoration: rest and labels are keyed by kind,
so a 👎 has to be able to mean "not weather" rather than "not proactivity".

## Measurement

Shipping this without measuring it would repeat what this branch just spent a day
learning.

| what | required |
|---|---|
| does it speak at all | utterances/day > 0, sustained over a week |
| does it repeat itself | no entity twice within the rotation window; the verbal-tic detector's own rule — a phrase the daemon repeats and the owner never uses — applied to proactive lines |
| does the owner want them | 👍/👎 ratio; this is the first data the labels clock has ever had |
| does the search change anything | `topic` candidates with a search result vs the same candidates without, judged for whether the line carries content |

The last row is the one that decides whether the contract change earned itself. If
a `topic` line reads the same with and without the search, the search bought
nothing and the boundary should move back.

## Out of scope

- Loosening the judge. It answers 26/26 on real material; the problem was never
  there.
- Widening the emotion or event lexicons. Measured 2026-08-18: 30 extra emotion
  words found 3 hits in 579 lines. This owner does not talk that way to this
  daemon, and no lexicon fixes that.
- `association` (type E). Already measured as surfacing command history.
- `daemon memory forget <id>` — still missing, still the reason the owner's
  `core.md` line cannot be retracted, still tracked separately.
