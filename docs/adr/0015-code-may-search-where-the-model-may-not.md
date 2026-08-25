# 0015 — Deterministic code may search on a proactive turn; the model still may not

**Status:** accepted · 2026-08-25

## Context

[docs/CONTRACTS.md](../CONTRACTS.md) non-negotiable 10 reads *"No tool runs on a
turn whose origin is not `owner`. Not in any mode, not with any setting, `full`
included."* Its stated rationale is that recall replays arbitrary old text into
every prompt, so `"look at this message"` must not become a way to hand a
stranger a shell.

Proactivity has never spoken. **572 judge calls, all time, 0 utterances.** The
owner read that as an over-strict judge; [PLAN.md](../PLAN.md) §8.2 had already
measured the opposite — the judge answers `open_loop` reasons **26/26** and
`silence` reasons **20/20** with "nothing to say". Running the generators over
the live database, 7 days, 75 owner utterances: `open_loop` fires **3 times**
(all on two days), `emotional` **0**. One generator works, and it needs the owner
to mention a dated future event in chat. He talks to this daemon in imperatives
about tools.

The material is the constraint, and the material the owner actually wants
conversation about — what is happening with the companies he is interviewing at,
the designer whose work he follows, his own project — lives on the web, not in
the daemon's tables.

## Decision

**Split non-negotiable 10 rather than weaken it.**

| | before | after |
|---|---|---|
| the **model** choosing and running a tool on a non-owner turn | forbidden | **still forbidden**, unchanged |
| **deterministic code** issuing one read-only search on the proactive path | forbidden | **allowed** |

The judge stays `tools_offered=0`, with a test that fails if that ever stops
being true. The model does not decide whether to search and does not decide what
to search for: code issues the query, and the query is an entity name read out of
`entities.name`.

## Why this is not the thing rule 10 forbids

Rule 10's threat is a model, holding tools, reading attacker-controlled text and
choosing to act on it. Here there is no choosing: the search is a fixed call with
a first-party argument, made before the model is invoked at all. What the model
receives is text, and text is what it already receives from recall.

That is the honest framing — **this widens what untrusted text can reach an
unprompted utterance**, and unprompted matters. A prompted answer is one the
owner asked for and is reading with that in mind; an unprompted line arrives in
their Telegram or out of their laptop speaker, in a voice they trust, when they
were not expecting it. `daemon/proactivity/judge.py` already refuses even *the owner's own words* in
a reason for exactly this reason.

So the surface is defended at the choke point that already exists — the judge's
output — rather than by trusting the fence around the input:

1. Query is `entities.name` only, chosen by code with no model call in
   between - never a value derived from a search result or from the judge's
   own reply. (Corrected 2026-08-25, round 4 of the task that built this
   surface: `entities.name` is not itself free of model influence -
   `daemon/reflection.py` writes it from a model reading the day's conversation
   log -
   so this defence is about the query being chosen deterministically at call
   time, not about the string being guaranteed harmless. Defence 4 (the URL
   refusal on the output) and `Judge.decide`'s early drop of a pointer-shaped
   entity are what carry that weight instead.)
2. Results reduce to **titles**, at most 3, each capped at 80 characters.
3. Fenced under a nonce, marked reference material and never an instruction.
4. **A URL in the utterance is a decline.** The vector worth fearing is not
   exfiltration — proactive delivery goes to the paired owner or the local
   speaker — it is this daemon's trusted voice telling its owner where to go.
5. One search per gate-passed candidate, never per tick, so non-negotiable 7's
   cost shape holds: deterministic generation, deterministic gate, then exactly
   one expensive step.

Defence 4 is the load-bearing one. The others reduce what gets in; only that one
bounds what gets out, and this repo has already watched a fence lose — the
`render_continuity` header says *"do not imitate the style of these lines"* and
the model imitated them anyway, measurably, until the phrases were named.

## Consequences

- `daemon/proactivity/candidates.py` stops being purely deterministic-from-the-database.
  Its module docstring's claim that every reason is built from *"its own lexicons,
  clock times and dates"* becomes true of three generators and false of the fourth.
- A failed or disabled search drops the `topic` candidate and leaves the other
  three generators working. Proactivity degrades to today's behaviour, not to an
  error.
- The daily budget drops 8 → 5 and the cooldown rises 30 → 90 minutes. Not
  because the budget was ever binding — at 0 utterances nothing was binding — but
  because a generator that can always find material needs a real ceiling where
  one that fires three times a week did not.

## What would overturn this

The spec requires measuring whether the search changes the utterance: `topic`
candidates with a search result against the same candidates without, judged for
whether the line carries content. **If a topic line reads the same either way,
the search bought nothing and this decision should be reverted** — the boundary
goes back and the three offline generators stay.

Four ADRs in this file were overturned by measurement. This one names its own
test in advance so it can be the fifth without an argument.

**A separate note, added after five rounds of hardening `has_url` (2026-08-25):**
a `topic` candidate whose own entity name reads as a pointer (`daemon/proactivity/judge.py:has_url`
returns true for it) is dropped before the search or the model call runs — it is
never spoken, not even as its own gate-passed subject. Three rounds tried to carve
a safe exemption for exactly this case and none survived review: nothing in the
data this module can see tells a legitimate domain-shaped entity (`UJET.cx`) apart
from an attacker-chosen one (`evil.com`) — not shape, not `created_at`, not
`mention_count` (both reachable by the same reflection pass that could plant an
attacker's text), and turn-level provenance does not exist in
`daemon/memory/schema.sql`. This is accepted as a real, permanent cost, not a bug
to be revisited with more mechanism: an owner who wants `UJET.cx` speakable again
has to rename the entity note to something that does not read as a domain, which
is their call to make, not code's to infer. If that remedy stops being enough —
if the owner routinely has domain-shaped entities worth naming and renaming each
one is real friction — the next lever is an **owner-typed allowlist** (a config
list of entity names the owner has explicitly approved as safe to speak), never
a rule this module derives on its own from data an attacker can also write to.
