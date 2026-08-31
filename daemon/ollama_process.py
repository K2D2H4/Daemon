"""The local Ollama the daemon starts when nothing else has.

`Task.EMBED` is always `ollama` whatever `DAEMON_PROVIDER` says, so recall's
vector lane depends on a server that - measured 2026-08-26 - nothing on a fresh
macOS install is responsible for starting. The daemon survives a reboot on the
LaunchAgent's `RunAtLoad`; its embedder had no equivalent, and the result was
fifteen hours of keyword-only Korean recall with nothing failing.

Four gates, and two of them exist in order *not* to act:

  * The URL must name this machine. `DAEMON_OLLAMA_BASE_URL` may point at another
    host, and spawning a local server for it would start an empty second Ollama
    while the real one carried on unused.
  * It must not already answer. Somebody running Ollama under `brew services` or
    Ollama.app owns that process; this one neither duplicates nor stops it.
  * The binary must be findable - see `find_binary` for why `which` is not enough.
  * It must answer within `READY_TIMEOUT_SECONDS`.

Failing any gate is not an error. The daemon boots, serves, and answers with
keyword-only recall, exactly as it did for those fifteen hours.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

READY_TIMEOUT_SECONDS = 30.0
"""How long to wait for a spawned Ollama to answer. Generous on purpose: a cold
start loads no model, but a machine still finishing login is slow at everything."""

POLL_SECONDS = 0.5

BINARY_FALLBACKS = (
    "/opt/homebrew/bin/ollama",
    "/usr/local/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)
"""Where to look when PATH does not say.

Measured 2026-08-26: the resident launchd job runs with
`PATH=/usr/bin:/bin:/usr/sbin:/sbin` - `_render_plist` omits
`EnvironmentVariables` deliberately (daemon/service.py:226) and
`launchctl getenv PATH` is unset. Homebrew's bin is on neither, so
`shutil.which("ollama")` succeeds in a terminal and finds nothing in the service
this code actually runs in."""

LOCAL_HOSTNAMES = frozenset({"localhost"})


def is_local(base_url: str) -> bool:
    """Does this URL name the machine we are running on?

    A URL with no hostname is not local. `urlparse` reads a scheme-less
    `192.168.1.50:11434` as scheme `192.168.1.50` with no host at all, so counting
    an absent hostname as loopback opened this gate for exactly the remote address
    it exists to refuse - measured, both `192.168.1.50:11434` and
    `ollama.internal:11434` answered True, and the daemon would have started an
    empty local server while the real remote one carried on unused.
    """
    host = urlparse(base_url).hostname
    if host is None:
        return False
    if host in LOCAL_HOSTNAMES:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def find_binary() -> str | None:
    """`ollama` on PATH, or in a known install location. See BINARY_FALLBACKS."""
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in BINARY_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    return None


async def _probe(base_url: str) -> bool:
    """Does Ollama answer? Its own client, not a provider's: this runs before the
    gateway exists and must not depend on it (CONTRACTS 4)."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
    except Exception as exc:
        # Broad on purpose: httpx.InvalidURL (e.g. a non-numeric port from a
        # typo'd .env) is an Exception, not an httpx.HTTPError, and
        # ollama_base_url has no validator (config.py) - a malformed URL must
        # close this gate like any other, not raise past ensure_running's
        # "never raises" promise.
        #
        # Logged rather than discarded: swallowing it silently made a typo'd
        # DAEMON_OLLAMA_BASE_URL report "ollama did not answer within 30s", so the
        # owner went looking at a healthy Ollama and never learned their .env was
        # unparseable. debug, not warning - a closed gate is already reported by
        # `ensure_running`, and every failed poll of a cold start comes through
        # here too.
        logger.debug("ollama probe at %s failed: %s", base_url, exc)
        return False
    return response.status_code == 200


async def _spawn(binary: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        binary,
        "serve",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


class LocalOllama:
    """One child Ollama. Cheap to construct; nothing runs until `ensure_running`."""

    def __init__(
        self,
        base_url: str,
        *,
        find: Callable[[], str | None] = find_binary,
        probe: Callable[[str], Awaitable[bool]] | None = None,
        spawn: Callable[[str], Awaitable[asyncio.subprocess.Process]] | None = None,
    ) -> None:
        self._base_url = base_url
        self._find = find
        self._probe = probe or _probe
        self._spawn = spawn or _spawn
        self._process: asyncio.subprocess.Process | None = None

    @property
    def started_by_us(self) -> bool:
        return self._process is not None

    async def ensure_running(self) -> bool:
        """True when Ollama answers at the end of this call.

        Never raises. A gate that closes is a log line and a keyword-only recall,
        not a dead daemon."""
        if not is_local(self._base_url):
            logger.info(
                "ollama at %s is not on this machine; leaving it alone", self._base_url
            )
            return await self._probe(self._base_url)
        if await self._probe(self._base_url):
            logger.info("ollama already running at %s", self._base_url)
            return True
        binary = self._find()
        if binary is None:
            logger.warning(
                "ollama is not installed where this process can see it; recall stays "
                "keyword-only. Install it from https://ollama.com, or start it yourself"
            )
            return False
        try:
            self._process = await self._spawn(binary)
        except OSError as exc:
            logger.warning("could not start %s serve (%s); recall stays keyword-only", binary, exc)
            return False
        logger.info("started %s serve (pid %s)", binary, self._process.pid)
        if await self._wait_until_ready():
            return True
        # Two different failures, and naming the wrong one sends the owner to the
        # wrong place: a child that exited at once is a broken start (a port already
        # bound, a half-installed binary), not a slow one, and reporting it as a 30s
        # timeout hides both the cause and the exit code - the only trace there is,
        # since stdout and stderr go to DEVNULL.
        exit_code = self._exit_code()
        if exit_code is not None:
            logger.warning(
                "the ollama we started exited with %s before answering; recall stays "
                "keyword-only",
                exit_code,
            )
            return False
        logger.warning(
            "ollama did not answer within %.0fs of being started; recall stays "
            "keyword-only until it does",
            READY_TIMEOUT_SECONDS,
        )
        return False

    async def _wait_until_ready(self) -> bool:
        """Poll until it answers, the child dies, or the budget runs out.

        Watching `returncode` is what makes a broken start diagnosable. `_spawn`
        raises nothing when the binary launches and exits immediately - a port
        already bound by an Ollama.app still coming up at login, a half-installed
        binary - and `stdout`/`stderr` go to DEVNULL, so without this the only
        report was "did not answer within 30s" thirty seconds later, naming a
        timeout where the truth was an instant exit with a code.
        """
        deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            if await self._probe(self._base_url):
                return True
            if self._exit_code() is not None:
                return False
            await asyncio.sleep(POLL_SECONDS)
        return False

    def _exit_code(self) -> int | None:
        """The code the child we spawned exited with, or None while it is alive."""
        return None if self._process is None else self._process.returncode

    async def aclose(self) -> None:
        """Stop only what this object started.

        An Ollama that was already up belongs to whoever started it and outlives
        this daemon. Same orphan reasoning as the stdio MCP servers in
        `app.py`'s lifespan `finally`: without this, every restart leaves one more."""
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            async with asyncio.timeout(5.0):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
