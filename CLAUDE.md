# Daemon — orientation

A self-hosted AI companion: one resident process, markdown as the source of
truth, a personality that evolves, and a loop that decides when to speak first.

**Read [docs/CONTRACTS.md](docs/CONTRACTS.md) before writing code** — short and
binding: breaking one of its rules loses user data, leaks a secret, or launders
untrusted text into the personality. It also holds the layering rule (only
`daemon/app.py` imports an implementation) and the non-negotiables everything
here used to summarise. Rationale: [docs/PLAN.md](docs/PLAN.md) (Korean).
Decisions and the measurements that overturned some: [docs/adr/](docs/adr/).
Layout and runtime data flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Where things are

| | |
|---|---|
| [daemon/](daemon/CLAUDE.md) | the process: entrypoint, config, channels, memory, llm, voice, persona, tools. Recipes in [RECIPES.md](daemon/RECIPES.md), hard-won measurements in [MEASURED.md](daemon/MEASURED.md) |
| [tests/](tests/CLAUDE.md) | the suite, plus reachability and acceptance gates |
| [evals/](evals/CLAUDE.md) | recall golden set, the live-API voice spike, and how to report a run honestly |
| `docs/` | PLAN (design), CONTRACTS (rules), ARCHITECTURE (layout), adr/ (decisions) |
| [scripts/](scripts/CLAUDE.md) | repo checks that run in CI and import no product code |
| `site/` | the landing page — one self-contained file, deployed to GitHub Pages. Everything in here is published; put working files in `docs/design/` instead |

## Commands

```bash
python3 -m pytest                  # the whole suite
python3 -m ruff check .            # lint
python3 scripts/check_docs.py      # documented paths exist
python3 -m evals.golden_set --json # recall quality, and log the run's conditions
daemon setup                       # onboarding; daemon doctor to inspect
```

## Two changes that span the repo

**Starting a milestone.** Its gate is in `docs/PLAN.md` §8.2, written as a thing you
can do rather than build. Close the matching `PENDING_TASKS` / `PENDING_CLASSES`
entries in `tests/test_reachable.py` as you wire each piece up.

**Changing a frozen contract** — `daemon/tasks.py`, `daemon/memory/schema.sql`, or a
protocol file. Allowed; doing it quietly is not. Say so, and read
[docs/adr/](docs/adr/) first: four were corrected by measurement.
