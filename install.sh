#!/usr/bin/env bash
#
# Daemon installer. One command, on a machine with nothing on it:
#
#   curl -fsSL https://raw.githubusercontent.com/K2D2H4/Daemon/main/install.sh | bash
#
# It bootstraps uv (its one dependency), then lets uv provision Python 3.13 and
# install the `daemon` command into an isolated tool environment. It deliberately
# stops there: it does NOT run onboarding or register a background service. Those
# are `daemon setup` and `daemon install` - the first wants a real terminal to ask
# you for a Telegram token and API keys, and neither should happen inside a pipe
# you cannot see. Re-running this script upgrades in place.
#
# Overrides (env vars):
#   DAEMON_VERSION   git ref to install - a tag like v0.1.0, a branch, or a sha.
#                    Default: the latest published GitHub release, or `main` if no
#                    release exists yet.

set -euo pipefail

REPO="K2D2H4/Daemon"
PACKAGE="daemon-ai"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required to run this installer."

# 1. uv - the only thing the installer itself has to put on the machine.
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (it provisions Python 3.13 and isolates the install)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv's installer drops the binary in ~/.local/bin; make this shell see it now.
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 \
  || die "uv is installed but not on PATH. Open a new shell and run this again."

# 2. Decide which ref to install. An unset DAEMON_VERSION means "latest release";
#    before the first release is cut, that 404s and we fall back to main.
ref="${DAEMON_VERSION:-}"
if [ -z "${ref}" ]; then
  ref="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
          | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' \
          | head -1 | sed 's/.*"\([^"]*\)"$/\1/')" || true
  if [ -z "${ref}" ]; then
    warn "No published release found; installing the development head (main)."
    ref="main"
  fi
fi

# 3. Install from a source tarball, not git+, so the machine needs no git - a
#    fresh Mac has none until the Command Line Tools are installed, and a bare
#    Linux box may not either. GitHub's /archive/<ref>.tar.gz resolves a tag, a
#    branch or a sha alike. uv fetches it, builds it (hatchling), and drops the
#    `daemon` command in. --python 3.13 pulls a managed CPython if the host has
#    none; --force makes a re-run an upgrade rather than a no-op.
say "Installing ${PACKAGE} @ ${ref} ..."
# The `[mcp]` extra is included because MCP defaults on: without it the admin's MCP
# tab shows but every connect fails "No module named 'mcp'". `daemon update` installs
# the same spec, so an update does not drop it. (The extra pins mcp<2 - the v2 client
# surface is incompatible - so this stays on the working 1.x.)
uv tool install --force --python 3.13 \
  --from "https://github.com/${REPO}/archive/${ref}.tar.gz" "${PACKAGE}[mcp]"

# Put uv's tool bin dir on PATH - for the version check below, and, persistently,
# for the user's future shells.
uv tool update-shell >/dev/null 2>&1 || true
export PATH="${HOME}/.local/bin:${PATH}"

# 4. Confirm, then point at onboarding rather than piping into it.
if command -v daemon >/dev/null 2>&1; then
  say "Installed: $(daemon --version)"
else
  warn "Installed, but 'daemon' is not on your PATH in this shell yet."
  warn "Open a new terminal, or add this line to your shell profile:"
  warn '  export PATH="$HOME/.local/bin:$PATH"'
fi

cat <<'NEXT'

Next:
  daemon setup      pick a preset, add a Telegram token, verify keys
  daemon install    keep it running after you close the terminal, and after a reboot

NEXT
