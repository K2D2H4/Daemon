# 0008 — Proactivity is three stages, and exactly one model call

**Status:** accepted · 2026-08-04

## Context

"It speaks first, on its own judgement" is one of this product's two differentiators,
and the obvious implementation is one prompt: hand the model the recent conversation
and the clock, ask whether to say something, deliver whatever comes back.
[PLAN.md](../PLAN.md) §6.2 predicted that this fails in a specific way — *asked "should
I speak?" as an open question, an LLM answers yes almost every time* — but that was an
inference, and the cheap shape keeps looking reasonable to whoever reads the code next.

**Measured, against local gemma3:4b, one reason per call, 15 calls per prompt:**

| prompt | declines |
|---|---|
| declining explicitly permitted and called correct | **0 of 15** |
| silence stated as the default, speaking the exception, two conditions | **3 of 15** |

The first prompt answered even `특별한 일은 없다.` ("nothing in particular is
happening") with `별일 없어.`, and drifted out of the persona's 반말 into polite forms
while doing it. The second declined on exactly the contentless reasons. So the
prediction survives narrowing the question, and a model asked to judge its own
permission to interrupt will grant it.

The second half of the context is that the model cannot answer the question anyway. It
does not know the user is in a meeting, how many times it already spoke today, or
whether the audio device is in use.

## Decision

Three stages, separate objects, and the boundaries are the design:

| | | model calls |
|---|---|---|
| `daemon/proactivity/candidates.py` | reasons it might be worth speaking | 0 |
| `daemon/proactivity/gate.py` | is now safe, and where may it go | 0 |
| `daemon/proactivity/judge.py` | what to say — or nothing | **1**, only after the gate |
| `daemon/proactivity/delivery.py` | get it there, record it, attach the label | 0 |

Written into [CONTRACTS.md](../CONTRACTS.md) as non-negotiable 7 so it cannot be
softened quietly: **silence is the default**, stages 1 and 2 are deterministic, and the
one call happens only for a candidate that already passed the gate. The model is asked
what to say about a specific reason at a moment already judged safe — the one question
it is good at — and declining is a first-class answer, carried as a falsy `Utterance`
rather than an exception.

The gate's two decisions are also kept apart, because [PLAN.md](../PLAN.md) §6.4's
asymmetry is not symmetric: quiet hours, the cooldown and the budgets *block*, while
anything bearing only on interruption — an unreadable probe, a meeting app in front, an
audio device in use — costs the **speaker** and sends the same words to Telegram.

## Consequences

Deterministic stage 1 means candidate generation is string matching over Korean, which
is harder than a prompt and is the reason `daemon/proactivity/candidates.py` carries
lexicons of surface forms. Deterministic stage 2 means every threshold is a number somebody has to choose,
and the day it is wrong nothing speaks. Both are the intended trade: a wrong number is
diagnosable from `proactive_utterances.gate_snapshot`, and a model that always says yes
is not.

It also makes the loop cheap enough to leave on — 288 ticks a day cost sqlite reads and
three subprocess probes, and `Task.PROACTIVE_JUDGE` stays local in every preset but
`quality`. And it made `daemon proactive` possible: the CLI assembles the deterministic
half *only* — no gateway, no channel, no speaker — so the gate could be read by a human
before anything was wired to a speaker, which is the order §6.4 asks for.

What it does not fix: a 4B model still fills a contentless reason with an empty
pleasantry when the reason carries only elapsed time. That is a budget question, so it
belongs to the gate and to label-driven tuning, not to a better prompt.

## What would change our mind

A measurement showing a model declining most of the time when asked the open question —
plausible for a much larger one. It would move stage 2's *thresholds* into the prompt at
best, not the stage: the gate exists as much because the model cannot see the machine as
because it will not say no.
