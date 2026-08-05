# 0006 — Reachability and acceptance are required gates

**Status:** accepted · 2026-08-03

## Context

A milestone shipped with 470 passing unit tests and failed on first contact three
separate ways: `daemon run` refused to start, the bot never answered, and voice was
reported complete while no code path could reach it. Later the same shape recurred:
`openai` and `gemini` were nameable in a route and unbuildable in the app.

The pattern is always identical — **contract satisfied, unit-tested, unreachable.**
Deep unit coverage cannot see it, and thirty seconds of using the product can.

## Decision

Two more gates, required by [CONTRACTS.md](../CONTRACTS.md), in CI:

- **`tests/test_reachable.py`** — every `Task` has a caller, every nameable
  provider is buildable, every protocol implementation is constructed by something.
  Anything not built yet is declared PENDING **with the milestone that owns it**, so
  a gap becomes a line someone chose to write. It checks **both directions**: a
  stale PENDING fails too.
- **`tests/test_acceptance.py`** — assemble the app the way the entrypoint does and
  drive a real conversation, with fakes only at the network edge.

And a habit, not a file: when the suite passes, **run the product.**

## Consequences

Writing them immediately found four defects unit tests could not: the inbound poll
had no floor and spun at ~16,000 requests/second on a transport that returned
immediately; `uninstall` printed "installed"; a voice route demanded an env var
voice never reads; and an unknown provider raised `AttributeError` from inside
pydantic instead of naming the missing variable.

Reachability then failed twice more *by working* — when providers landed, and when
voice was wired — each time saying which declaration had gone stale.

Cost: PENDING lists need maintaining, and a check that produces noise is a check
somebody turns off. `scripts/check_docs.py` needed an explicit citation allowlist
for exactly that reason.

## What would change our mind

Nothing. This is the cheapest defect class we have and it had a 100% escape rate
before these existed.
