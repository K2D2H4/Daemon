## What this changes, and why

<!-- The why matters more than the what; the diff already says the what. -->

## Gates

- [ ] `python3 -m pytest`
- [ ] `python3 -m ruff check .`
- [ ] `python3 scripts/check_docs.py`
- [ ] **I ran the product**, not only the suite.

That last one is not ceremony. Every defect this project has shipped passed the
suite first: `daemon run` refusing to start, a bot that never answered, and voice
reported complete with nothing constructing a session.

## If you built something new

- [ ] Something constructs it — or `tests/test_reachable.py` declares it PENDING
      with the milestone that owns it.
- [ ] A new `Task` is routed in all three presets and has a caller.
- [ ] A new provider is in `HOSTED_PROVIDERS`, built in `daemon/app.py`, and
      offered by `daemon setup`.

## If you touched a contract

`daemon/tasks.py`, `daemon/memory/schema.sql` and the protocol files are frozen —
changing one is allowed, doing it silently is not. Say so here, and check
[docs/adr/](../docs/adr/) first: several of those shapes were corrected by
measurement rather than by reading docs.

## If you changed recall

- [ ] Quoted all three embedder modes, not just the best one, and regenerated
      `evals/agent-results.json` so the score carries its conditions.
