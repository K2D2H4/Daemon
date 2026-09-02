# 0021 — The calendar read happens before the gate, not after it

**Status:** accepted · 2026-09-01 · extends [0015](0015-code-may-search-where-the-model-may-not.md)

## Context

[ADR 0015](0015-code-may-search-where-the-model-may-not.md) split
[CONTRACTS](../CONTRACTS.md) non-negotiable 10 so that deterministic code — never
the model — may issue **one read-only search on the proactive path, for a
candidate that already passed the gate.** That placement was not incidental. It is
what keeps non-negotiable 7's cost shape intact: deterministic generation,
deterministic gate, then exactly one expensive step, only for what survived.

0015 then measured what the surface it opened actually bought, and the honest
number it tells anyone re-reading it to start from is **two of six**: the web
search pays off on entity names the web knows as the owner knows them, and buys
nothing on names that collide with something famous. The live database agrees.
Nine proactive utterances exist, all time; seven are `topic`, and read as a set
they are the same sentence with a different noun in it:

    요즘 키위 작업은 잘 돼가고 있어요?
    요즘 데몬 작업은 잘 진행돼가고 있어요?
    ReadyTalk 소식 안 본 지 좀 됐네요. 요즘은 어떻게 돼가요?
    슈베르트 진 소식은 여전히 조용한가 봐요. 별일 없이 잘 지내죠?   ← the one 👎

The machine works. The material is thin, and it is thin because
`entities.name` against the open web is a query about *what a word means*, not
about *what happened to this person*.

The owner's calendar is the opposite kind of material: first-party, dated, and
true. Measured against the live `google` MCP server on 2026-09-01, the primary
calendar holds 7 events in the last 30 days and 13 in 180 — `Interview with
UJET`, `BEN home assessment`, `Mistral | Applied AI - Hiring Manager`. A daemon
that cannot say *"UJET 면접 30분 남았어"* is missing the one thing it is best
placed to say.

## The problem placement causes

A `topic` candidate can be *generated* with no network at all: `stale_entities`
is a SQL query, and the search only supplies what to say about a subject the
database already knew was worth raising. A calendar candidate has no such
subject. **Whether there is anything to say at all is the thing only the calendar
knows.** Reading it after the gate means stage 1 must emit a candidate blind —
one per tick, 288 a day, on the chance that an event exists — and the gate, which
is arithmetic on rows and settings, has nothing to filter them by. That is not a
gate doing its job; it is 288 rows a day and a model call for each one that
survives the cooldown.

## Decision

**For the `calendar` kind, the one read-only MCP call moves from stage 3 to
stage 1.** Everything else 0015 fixed stays fixed:

| | 0015 (`topic`) | here (`calendar`) |
|---|---|---|
| who decides to call | deterministic code | **unchanged** |
| who chooses the arguments | code, from a first-party constant/column | **unchanged** |
| a result becoming the next call's argument | never | **never** |
| tools offered to the model | zero | **zero** |
| the output URL refusal (`judge.has_url`) | load-bearing defence | **unchanged, still load-bearing** |
| *when* the call happens | after the gate, per gate-passed candidate | **before the gate, once per tick** |

The arguments are `user_google_email` (the owner's own setting,
`DAEMON_CALENDAR_EMAIL` — never discovered by calling `list_calendars` first,
which would be exactly the result-becomes-argument shape 0015 forbids), plus
`time_min`/`time_max` computed from the clock and `max_results`, all constants
this module writes.

## What this costs, stated rather than discovered later

**288 read-only calls a day** to a local stdio child, against a Google API whose
quota is six orders of magnitude larger. The window is narrow on purpose
(`now` to `now + CALENDAR_LEAD_MINUTES`, `max_results=5`), so the reply is small
and usually the string `No events found`.

Non-negotiable 7 counts **LLM** calls, and this changes none of them: stage 1 and
stage 2 still make zero, stage 3 is still exactly one, still only for a candidate
the gate passed. So the rule is not violated. But the sentence people *read* rule
7 as saying — "nothing expensive happens before the gate" — is now less true than
it was, and pretending otherwise is how a cost shape drifts. It is written here so
the next person weighing a third network generator starts from a number.

**It also inverts which stage is idempotent.** A `topic` candidate is re-searched
every time it is rested and comes due again (0015's own round-4 correction). A
`calendar` candidate is read once, at generation, and carries its material in
`payload` from there on — so stage 3 makes **no** MCP call for this kind at all.
On the retry axis this is strictly cheaper than `topic`, not more expensive.

## What would overturn this

Written before the code, so it can fail:

The spec's paired replay (`evals/proactive_calendar_spike.py`) runs the owner's
real past events at `now = start − CALENDAR_LEAD_MINUTES`, arm A without the
generator and arm B with it, interleaved trial by trial, hand-audited. **If arm B
does not clear 20/39 lines that name the event and state the time correctly
against arm A's ≤ 2/39 — or if a single URL reaches an utterance, or a single
spoken time disagrees with the clock — the generator has not earned the placement
and this ADR is reverted.** Reverting is one generator, one kind, and one
migration; nothing about `topic` moves.

The weaker failure worth naming separately: if the *rate* is the problem rather
than the content — if 288 calls a day turns out to cost battery, wake a sleeping
laptop, or trip an API limit nobody predicted — the fix is a poll interval, not a
move back to stage 3, because stage 3 cannot answer the question stage 1 is
asking.

**Run 2026-09-01 (n=39 paired, `gemini-3.6-flash`, the owner's 13 real events x3):
not overturned, and not narrowly.**

| | arm A (reason only) | arm B (reason + fenced title) |
|---|---|---|
| names the event (scored) | 0/39 | 32/39 (p < 0.0001, Fisher exact) |
| names the event (hand-audited) | 0/39 | **36/39** |
| declined | 3/39 | 0/39 |
| wrong time | 0/39 | 0/39 |
| url leaked | 0/39 | 0/39 |

The hand audit raised the count rather than lowering it: `_names_event` wants a
literal title token and the model translates (`The Monkey Forest Ubud` → `몽키
포레스트 우붓`, `FE Coding` → `FE 코딩 테스트`), so four scored misses do name their
event. All three genuine misses are the `Sr.Lead` event the shipped path drops
before the model call — the spike builds its messages directly rather than through
`Judge.decide`, so it measured an event production never speaks about. On the
production-reachable 12 events: **36/36**.

Read arm A honestly before quoting the gap: it is not a straw man. After ADR 0016
an elapsed-time-only reason is one the prompt tells the model to answer, and arm A
answered 36 of 39 times — with `30분 뒤에 일정 있던데, 잊지 않았지?`. A perfectly
good sentence that tells the owner nothing they did not already know. That is
precisely the difference this ADR was built to buy, and it is the same difference
ADR 0015 bought on two entities out of six.

## Two things the first live run found, both recorded rather than smoothed over

**Google's `timeMax` is exclusive and this design's window is not.** Replaying all
13 of the owner's real events at exactly `start - CALENDAR_LEAD_MINUTES` produced
**13 "no candidate"** against a calendar that plainly had them, because a window
ending at exactly 13:00:00 does not contain a 13:00 event (verified directly: the
same window ending 13:00:01 returns it). `agenda.fetch` now adds one second to the
bound it sends, so its documented inclusive `[start, end]` is true and
`calendar_candidates`'s `starts_at <= horizon` filter has a reachable upper branch.
In production the miss was nearly harmless - `now` is whatever the clock says and
the next tick catches the event at 25 minutes out - which is exactly why it would
have gone unnoticed. After the fix: **13/13**.

**One of the owner's 13 real titles is permanently unspeakable, and it stays that
way.** `Sr.Lead Engineer- Seoul - DAEHYUN KIM and Gabriela Guerrero` satisfies
`judge._BARE_DOMAIN_RE` - a word, a dot, two or more TLD-shaped letters - exactly as
`Node.js` and `report.docx` do, which that regex's docstring already names as
accepted false positives. `Judge.decide` therefore drops it before the model call.

The cost lands harder here than it does for `topic` and that difference is the
part worth writing down. ADR 0015's stated remedy for a refused *entity* is that
the owner renames the entity note. An owner cannot rename someone else's meeting
invitation, and `Sr.`/`Dr.`/`Mr.`/`vs.` before a word is an ordinary way to title
a meeting - so this is a recurring loss at roughly **1 in 13 on real data**, not a
freak one. It is accepted anyway: `has_url` is the single defence bounding what
leaves this daemon's mouth, five rounds of narrowing it produced no safe
exemption, and a calendar-specific weakening would be a second, weaker copy of the
check for the one kind whose raw material is 100% urls. The lever, if this ever
becomes intolerable, is the owner-typed allowlist ADR 0015 already names - never a
rule derived from data an attacker can also write.

## The cost of the placement that only review found

Stage 1 runs on a timer with nobody watching, and that changes what a *failure*
costs. `get_events` for an address the server holds no credential for opens
Google's consent page in the owner's browser - so at 288 ticks a day, a lapsed
consent is a browser window every five minutes until somebody notices. A stage-3
call could not have done that: it only runs for a gate-passed candidate, behind a
cooldown, a handful of times a day at most.

So the placement carries one obligation stage 3 would not have: **a per-tick call
whose failure has a side effect needs a latch, not just a log line.**
`agenda._CONSENT_PENDING` is that latch, process-lifetime because every remedy
ends in a restart. Anyone adding a third generator that reads something on every
tick inherits this obligation, and it is the part of this ADR most likely to be
skipped.

## What this does not touch

`topic` keeps 0015's placement exactly. This ADR adds a second shape beside it; it
does not migrate the first one, and it is not a licence to move any future
generator's network call forward by default. A generator whose *subject* the
database already knows belongs after the gate, where 0015 put it. Only a
generator that cannot know whether it has a subject at all has the argument made
here.
