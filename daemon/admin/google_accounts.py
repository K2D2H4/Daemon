"""Which Google accounts the `google` MCP server has already authenticated.

A **suggestion source for the settings page and nothing else.** The proactive
path never calls this: `daemon/proactivity/agenda.py` reads
`settings.calendar_email`, a value the owner confirmed, and it stays that way
whatever this module answers. That separation is the whole reason this is a
distinct file rather than a helper inside `settings_io.py` - a bad or empty
answer here costs a dropdown entry, never a wrong calendar read.

## Why the filesystem, and why that is not the layering break it looks like

`workspace-mcp` exposes no way to ask "who are you authenticated as". Measured
against the live server, 2026-09-01: all 27 of its tools require
`user_google_email`, except `start_google_auth` - whose body is
`if not user_google_email: raise ValueError("user_google_email must be
provided.")` and whose only default is a `USER_GOOGLE_EMAIL` environment variable
you would have to already know. Even its OAuth flow is "consent as *this*
address", not "pick an address", and it returns the verified identity to its own
`localhost:8000` callback page rather than to the MCP client. So there is no API
answer to this question, at any price.

What there is: the server writes one credential file per authenticated account,
named for the account. Reading those *names* - never the contents, which are
tokens - is the only discovery this daemon can do, and it is read-only, local,
and confined to a UI hint.

The honest cost is coupling to another project's on-disk layout. It is bounded
three ways: the path is overridable the same way `workspace-mcp` overrides it,
every failure returns `[]`, and `[]` degrades the settings field from a picker to
a plain text box that has always worked. The alternative that avoids the coupling
is making the owner type an address they cannot check, which is the friction the
picker exists to remove.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DIR_ENV = ("GOOGLE_MCP_CREDENTIALS_DIR", "WORKSPACE_MCP_CREDENTIALS_DIR")
"""The two names `workspace-mcp` itself reads, in its own order of preference.

Honoured rather than assumed away: an owner who has moved the credential store
gets the same picker as one who has not, and hardcoding the default would show
them an empty list with no way to tell that from "nothing is authenticated"."""

DEFAULT_DIR = Path.home() / ".google_workspace_mcp" / "credentials"

_EMAIL_FILE_RE = re.compile(r"^(?P<email>[^@/\\]+@[^@/\\]+\.[A-Za-z]{2,})\.json$")
"""A credential file named for the account it holds.

Anchored and email-shaped on purpose. The directory also holds
`oauth_states.json` (observed on the live install), and anything that is not an
address is not an account - so this matches what to *keep* rather than filtering
out the names known today, the same discipline `agenda._EVENT_RE` uses on the
calendar reply. `/` and `\\` are excluded from both halves so a crafted filename
cannot suggest something path-shaped into a settings field.
"""


def credentials_dir() -> Path:
    """Where `workspace-mcp` keeps its credential files."""
    for name in DIR_ENV:
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw).expanduser()
    return DEFAULT_DIR


def authenticated_accounts() -> list[str]:
    """Addresses the google server has a stored credential for, sorted.

    Never raises and never returns a partial failure as an error: a missing
    directory, an unreadable one, and one holding nothing account-shaped are all
    "no suggestions", which the settings page renders as an ordinary text field.
    Logged at debug, not warning - on an install with no google server this is
    the normal state, not a problem anybody needs telling about.

    Returns names only. This function opens no file.
    """
    directory = credentials_dir()
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        logger.debug("google accounts: %s is not readable (%s)", directory, exc)
        return []
    found = {
        match.group("email")
        for entry in entries
        if (match := _EMAIL_FILE_RE.match(entry.name)) is not None
    }
    return sorted(found)
