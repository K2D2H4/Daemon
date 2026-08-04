# scripts/ — repo checks

## Owns

Checks that run in CI and are not tests. Nothing here imports `daemon`, so a
broken product does not stop the checks from telling you what is broken.

| | |
|---|---|
| `scripts/check_docs.py` | every path written in backticks in a `.md` must exist |

## Common changes

**Adding a check.** One file, stdlib only, exit non-zero on failure, and print
what to *do* rather than what is wrong. Then add a named step to
`.github/workflows/ci.yml` — a check nothing runs is a comment.

**A check fires on something correct.** Fix the resolution rule, not the message.
`check_docs.py` resolves a token against the document's own directory *and* the
repo root, because `daemon/CLAUDE.md` writing `daemon/loop.py`
means the file beside it, and that is how a person reads it too. Add to `CITED`
only for paths in *another* project that this repo quotes on purpose.

```bash
python3 scripts/check_docs.py            # every documented path exists
git config core.hooksPath .githooks      # run that check before each commit
```

## Depends on

Nothing in [daemon/](../daemon/CLAUDE.md). Used by `.github/workflows/ci.yml` and
the optional `.githooks/pre-commit`.

## Why this exists at all

**Note:** documentation that names a file that does not exist is worse than
documentation that says nothing — an agent follows the wrong path confidently.
This check found 9 such references in one pass, in files a human had just
reviewed.
