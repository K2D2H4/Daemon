# tests/ — three kinds, and why

## Owns

Every gate the project has. Unit coverage alone shipped a milestone that failed on first contact three ways:
`daemon run` refused to start, the bot never answered, and voice was reported
complete while nothing could reach it. So there are two more gates, and
`docs/CONTRACTS.md` requires them.

| | |
|---|---|
| **unit** — most files here | one module, fakes at its edges |
| **`test_reachable.py`** | is every built thing reachable from the assembled app? |
| **`test_acceptance.py`** | does the user's journey work, end to end? |

```bash
python3 -m pytest                              # all of it, ~6 s
python3 -m pytest tests/test_reachable.py tests/test_acceptance.py   # the gates
python3 -m pytest -k korean -q                 # the cases that break first
python3 -m pytest tests/test_loop.py -x        # one module, stop on first failure
```

## test_reachable.py

Every `Task` needs a caller. Every nameable provider must be buildable. Every
protocol implementation needs something that constructs it. Anything genuinely not
built yet is declared in `PENDING_TASKS` / `PENDING_CLASSES` **with the milestone
that owns it** — a gap becomes a line someone chose to write.

It checks **both directions**: a stale PENDING fails too. When voice was wired
this file failed, saying so, and closing those entries was the fix. That is the
file working, not the file being annoying.

## test_acceptance.py

Assembles the app the way the entrypoint does and drives a real conversation:
first run with no allowlist can start, the lifespan actually runs the loop, a
stranger gets a code and the owner is approved, and a turn reaches the reply, the
markdown, the mirror, the vector — then comes back through recall. Fakes stop at
the network edge, because the defects live between.

## Common changes

**Testing a new module.** One file per module, `conftest.py`'s `db` / `data_dir` /
`fake_provider` fixtures, and at least one Korean case if the module touches text.

**You built something new.** `tests/test_reachable.py` will fail until the app
constructs it or a PENDING entry names the milestone that owns it. That failure is
the point; do not silence it by deleting the assertion.

**A time-dependent test.** Pin the timestamp *and* the expected value — the suite's
convention is `2026-08-03`. **Gotcha:** pinning only the input breaks the day the
product's own clock disagrees. One test asserted a single log file while the loop
stamped its reply from the live clock, so it passed only on the day it was written
and failed a day later.

## Depends on

[daemon/](../daemon/CLAUDE.md) — all of it, which is the point — and `conftest.py`.
Never the network, a key, a microphone or a speaker. The gates these files enforce
are required by [CONTRACTS.md](../docs/CONTRACTS.md), and why they exist at all is
[ADR 0006](../docs/adr/0006-reachability-and-acceptance-gates.md).

## Rules

- **No test may touch the network, an API key, a microphone or a speaker.** One
  that needs any of those is broken. Use `conftest.py`'s `db`, `data_dir` and
  `fake_provider`; do not invent parallel fixtures.
- **Assert behaviour, and in Korean where the product is Korean** — CJK width,
  FTS5 tokenisation and transcript joining have all broken on Korean specifically.
- **Cover the failure path, not only the happy one.** Most findings in this repo
  were degradations that raised nothing: recall silently keyword-only, a dead
  loop, a queue behind a dead speaker.
- **A test that passes for the wrong reason is worse than none.** Two here were
  tautologies (a fake returning the string the assertion looked for) and one
  pinned a bug as the spec. If a test would still pass with the fix reverted, it
  is not a test — mutate it and check.
- Nothing here may hang. Anything driving a poll loop must be bounded; the first
  acceptance file blocked the whole suite because a channel polls forever by
  design.
