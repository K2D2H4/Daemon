#!/usr/bin/env python3
"""Verify that the numbers the landing page quotes are the product's real defaults.

`site/index.html` tells a visitor that quiet hours are 23:00-09:00, that the
cooldown is 90 minutes, that the daily budget is three and that `open_loop` gets
one of them - and then invites them to run `daemon proactive` and see the same
rules on their own machine. Every one of those is a `Field(default=...)` in
`daemon/config.py`, so changing a default silently turns the page into a lie
about software the reader is about to install.

That is not hypothetical. The page shipped with a proactivity log whose cooldown
arithmetic did not hold, a foreground app rendered as a *block* when it is only a
*route*, and a count of considered utterances that came from nowhere. Prose does
not have a type checker; this is the closest thing.

Like `check_docs.py`, this imports nothing from `daemon` - it reads config.py as
text. A broken product should still be able to tell you the page is wrong.

    python3 scripts/check_landing_claims.py

Exits non-zero listing every claim that no longer matches, and what to edit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
README = ROOT / "README.md"
CONFIG = ROOT / "daemon" / "config.py"

# alias in config.py -> (what the page must contain, human name)
# The expected string is built from the default we read, so these stay correct
# when a default changes and the page is updated to match.
CLAIMS: tuple[tuple[str, str, str], ...] = (
    ("DAEMON_PROACTIVE_QUIET_HOURS", "{}", "quiet hours window"),
    ("DAEMON_PROACTIVE_COOLDOWN_MINUTES", "needs {}m", "cooldown, in the log's reason string"),
    ("DAEMON_PROACTIVE_DAILY_BUDGET", "daily budget of {}", "daily budget, in the caption"),
    ("DAEMON_PROACTIVE_OPEN_LOOP_BUDGET", "{} of {} already spoken", "open_loop cap, in the log"),
)

# Claims that are about a boolean default rather than a number. If proactivity
# ever ships on by default, the page's "ships off by default" copy must change.
BOOL_CLAIMS: tuple[tuple[str, str, str], ...] = (
    ("DAEMON_PROACTIVE_ENABLED", "False", "Speaking first ships off by default"),
    ("DAEMON_VOICE_ENABLED", "False", "the switch that lets it speak out loud ships off too"),
)


# Prose spells small numbers out - "a daily budget of three" reads better than
# "of 3" in a caption, and both are the same claim. Accept either form rather
# than forcing the page to write like a config file.
WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def read_defaults(text: str) -> dict[str, str]:
    """Map every alias in config.py to its literal default.

    Handles both one-line Fields and the wrapped form black produces, so a
    reformat does not silently drop a claim from the check.
    """
    defaults: dict[str, str] = {}
    for m in re.finditer(
        r"Field\(\s*default=([^,\s)]+)\s*,\s*alias=\"([A-Z_]+)\"", text, re.S
    ):
        defaults[m.group(2)] = m.group(1).strip("\"'")
    return defaults


def check_env_table(text: str, defaults: dict[str, str], where: str) -> list[str]:
    """Any DAEMON_* alias a document names must show its real default on that line.

    README.md documents the proactivity settings as a table. Rather than matching
    prose, the rule is positional and so cannot rot: name the setting, state its
    default beside it. A line that names an alias and then contradicts the code is
    the failure this catches.

    Fenced blocks are skipped. They hold pasted command output and shell lines -
    `DAEMON_PROACTIVE_ENABLED=true to turn it on` is the product telling you how to
    switch it on, not a document claiming that is the default.
    """
    problems = []
    fenced = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for alias in re.findall(r"DAEMON_[A-Z_]+", line):
            value = defaults.get(alias)
            if value is None:
                continue
            # `false` in a document is the honest rendering of Python's False
            wanted = {value, value.lower()}
            if not any(w in line for w in wanted):
                problems.append(
                    f"{where}:{i} names {alias} but does not show its default "
                    f"({value}) on the same line: {line.strip()[:70]!r}"
                )
    return problems


def main() -> int:
    missing = [p for p in (PAGE, README, CONFIG) if not p.exists()]
    if missing:
        for p in missing:
            print(f"cannot check: {p.relative_to(ROOT)} does not exist", file=sys.stderr)
        return 1

    page = PAGE.read_text(encoding="utf-8")
    defaults = read_defaults(CONFIG.read_text(encoding="utf-8"))

    problems: list[str] = []

    for alias, template, label in CLAIMS:
        if alias not in defaults:
            problems.append(
                f"{alias} is no longer a Field(default=..., alias=...) in "
                f"daemon/config.py - update CLAIMS in this script to match its new shape"
            )
            continue
        value = defaults[alias]
        n = template.count("{}")
        forms = [template.format(*([value] * n))]
        if value in WORDS:
            forms.append(template.format(*([WORDS[value]] * n)))
        if not any(f in page for f in forms):
            wanted = " or ".join(repr(f) for f in forms)
            problems.append(
                f"{label}: daemon/config.py says {alias}={value}, so site/index.html "
                f"should contain {wanted} - it contains neither. Edit the page to "
                f"match the default, or the default to match the page."
            )

    for alias, expected_literal, quoted in BOOL_CLAIMS:
        if defaults.get(alias) != expected_literal:
            problems.append(
                f"{alias} default is now {defaults.get(alias)!r}, but site/index.html "
                f"still tells the reader {quoted!r}. Rewrite that copy."
            )

    # The one number on the page that is arithmetic rather than a default: the
    # log claims a five-minute loop, and 24h/5min is where 288 would come from.
    # It was quoted once as fact with ten rows on screen; refuse to let it back.
    if "288" in page:
        problems.append(
            "site/index.html contains '288'. It previously claimed the loop "
            "'considered speaking 288 times' above a log of ten rows, which is "
            "not in the copy deck. State a number the reader can count, or none."
        )

    problems += check_env_table(README.read_text(encoding="utf-8"), defaults, "README.md")

    if problems:
        print("documentation disagrees with the product:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}\n", file=sys.stderr)
        return 1

    checked = len(CLAIMS) + len(BOOL_CLAIMS)
    print(
        f"ok: {checked} landing-page claim(s) and every DAEMON_* default named in "
        f"README.md match daemon/config.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
