"""Is every built thing actually reachable from the assembled product?

This exists because of a repeated failure that 470 unit tests could not see. Each
time, a component was written, satisfied its frozen protocol, shipped with its own
passing tests - and nothing in the running daemon ever constructed it. Reported as
done, and broken the moment a person tried it:

  * `daemon run` refused to start, because pairing was implemented and tested but
    no config selected it and no assembly passed it.
  * The voice layer was called complete while nothing instantiated a session.
  * `openai` and `gemini` were nameable in a route and unbuildable in the app, so
    "provider-agnostic" was true of the config surface and false of the product.
  * chat_voice had no caller at all.

The pattern is always the same: contract satisfied, unit-tested, unreachable. So
reachability is asserted here, and anything genuinely not built yet has to be
declared PENDING with the milestone that owns it. That declaration is the point -
a gap becomes a line someone chose to write rather than something nobody noticed.

The check runs in both directions. A pending item that turns out to be reachable
also fails, because a stale PENDING is how this file would quietly stop working.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from daemon.config import PROVIDER_KEY_ENV
from daemon.tasks import Task

DAEMON = pathlib.Path(__file__).resolve().parents[1] / "daemon"

PENDING_PROVIDERS: dict[str, str] = {
    # Empty, and that is the point: every provider a route may name is now one the
    # app can build. openai and gemini were listed here, and this file failing when
    # they landed is the both-directions property doing its job.
}

PENDING_TASKS = {
    Task.RECALL_ESCALATION: "M1b+ - Lane 2 is specified, Lane 1 ships first",
}

PENDING_CLASSES: dict[str, str] = {
    # Empty. `LocalSpeaker` was the last entry and closing it is what M3b is: the
    # thing that speaks at the machine now exists and `app.build_proactive_tick`
    # constructs it.
    #
    # And it stays empty through the tool-grant work, which is worth writing down
    # because that work *is* unreachable and this dict cannot say so: it is not a
    # class. A grant is rows plus methods on `Store`, which `app.py` already
    # constructs - so `_constructed` finds it and reports the whole thing wired. The
    # `tool_grant` entries in `PENDING_WIRING` below are the both-directions
    # declaration instead. Same blind spot `WakeGate`'s comment describes: this file
    # asks whether something *calls* a name, never with what.
    #
    # The voice tool frame was the other half of that, until PR-2b. `send_tool_response`
    # was declared in `PENDING_WIRING`, not here, because `GeminiLiveSession(tools=...)`
    # is an argument to a class `app.py` already builds - invisible to `_constructed`
    # exactly like a grant. That entry is closed now: `daemon/voice/conversation.py`
    # routes a spoken tool call through `Companion.run_tools` and answers it with
    # `send_tool_response`, and `daemon/app.py`'s `run_voice` offers the session the
    # owner's tool specs, pinned to `allowlist`.
}

PENDING_WIRING = {
    # `send_tool_response` was here through PR-2a and is gone now: PR-2b wired it,
    # `daemon/voice/conversation.py` calls it, and `test_declared_wiring_gaps_are_still_gaps`
    # failing on it was the reminder to delete the entry. That is the file working.
    #
    # The three write/read surfaces of the grant table, none of them called from
    # daemon/ yet. `tool_grants` (the read the policy makes) is deliberately absent:
    # it *is* wired, through `policy.py:_granted`, which is the point of the both-
    # directions check. All three below are built and tested (test_tools.py) and
    # reachable from nothing, which is exactly the state this dict exists to name -
    # tracking only the write path would have been the same inconsistency, one layer
    # down, that PENDING_CLASSES' comment warns about.
    "add_tool_grant": (
        "M1c+ / PR-2b - the surface an owner grants a whole tool through. M1c",
        ("store.py",),
    ),
    "remove_tool_grant": (
        "M1c+ / PR-2b - revocation, the `daemon tools forget` of grants. M1c",
        ("store.py",),
    ),
    "all_tool_grants": (
        # This is the one the reviewer was right to single out. CONTRACTS rule 12:
        # a capability reported nowhere is the silent state this project keeps being
        # bitten by, and `_tools_check` in cli.py reads config only - so once
        # something writes a grant, a granted tool runs without asking while
        # `daemon doctor` still says runs-without-asking=nothing. `all_tool_grants`
        # is *how* doctor would show them, so leaving it untracked is how PR-2b could
        # wire `add_tool_grant` and forget visibility with every test still green.
        # Tracked here so wiring the write path and wiring the view are two conscious
        # acts, not one that silently drops the other.
        "M1c+ / PR-2b - the read `daemon doctor` needs so a grant is visible (rule 12). M1c",
        ("store.py",),
    ),
}
"""Built, tested, and called by nothing - declared the way PENDING_CLASSES declares
an unbuilt class, because these are functions and that dict only holds types.

The value is (owning milestone, the files allowed to mention it): the module that
declares it and the module that implements it. Any *other* file under `daemon/`
mentioning the name means it has been wired, and the test below then fails - which
is the reminder to delete the entry rather than leave a closed gap declared open.

Not a "load-bearing seams only" list. The file's whole subject is built-tested-
unreachable, and picking which unreachable symbols to track by how important they
feel is how the untracked one turns out to be the one that mattered - here, the
rule-12 visibility read."""

WIRED_CLASSES = (
    # What both endpoints do their work through. Named here because a capability
    # layer nothing constructs is the same defect as an unreachable provider, one
    # level up: `daemon/loop.py` and `daemon/voice/conversation.py` would take one
    # as an argument, their tests would pass against a hand-made instance, and the
    # assembled daemon would carry none.
    "Companion",
    "TelegramChannel",
    "FileMemoryWriter",
    "MemoryRecall",
    "OllamaEmbedder",
    "Pairing",
    # Closed when `daemon voice` landed. Both were declared pending while the
    # voice layer was complete, tested and unreachable - which is the whole
    # reason this file exists, and this line is what that gap closing looks like.
    "GeminiLiveSession",
    "SoundDeviceAudio",
    "VoiceConversation",
    # The echo-cancelling AudioIO, built by `app.build_voice_audio` on macOS.
    # `SoundDeviceAudio` stays above it rather than being replaced: it is still what
    # the wake gate and every non-Apple platform record through.
    "VoiceProcessingAudio",
    # M2. `Reflection` is the one that matters: the pass is only reachable because
    # `app.build_reflection` constructs it for both the 04:00 job and
    # `daemon reflect`, and a scheduled job nobody can run by hand is a job nobody
    # can verify.
    "Reflection",
    "CuratedMemory",
    "EntityNotes",
    # M3a, the deterministic half. Reachable because `daemon proactive` runs the
    # same tick the 05-minute job will - a scheduled job nobody can run by hand is
    # a job nobody can verify, and that is doubly true of one whose whole purpose
    # is to decide *not* to do something.
    "ProactiveTick",
    "Gate",
    "MachinePresence",
    # M3b: the one model call, the voice at the machine, and what records both.
    "Judge",
    "LocalSpeaker",
    "ProactiveDelivery",
    # M4: the weekly pass, and its one write path for persona/learned.md.
    # `app.build_persona_evolution` constructs the pass for both the Monday job
    # and `daemon persona evolve` - a scheduled pass nobody can run by hand is a
    # pass nobody can verify, the same reason `Reflection` is wired the same way.
    "PersonaEvolution",
    "LearnedRules",
    # The wake gate, built by `app.wake_events` - which is also what
    # `daemon wake test` drives. Being *named* here is not enough on its own: this
    # file only asks whether something calls `WakeGate(...)`, and it was called with
    # arguments the class did not have for as long as it took two agents to agree,
    # which no assertion here can see. `test_wake.py` drives `app.wake_events`
    # itself for that reason.
    "WakeGate",

    # M1c, the tool layer. Written, tested and unreachable is exactly the defect
    # this file exists for, and a policy nothing constructs is the worst version of
    # it: the tests would pass, the tools would still work, and nothing would be
    # gated.
    "ToolPolicy",
    "ToolRunner",
    "McpBridge",
    "Registry",
)


def _sources() -> list[tuple[pathlib.Path, str]]:
    return [(p, p.read_text(encoding="utf-8")) for p in DAEMON.rglob("*.py")]


def _defining(name: str) -> list[pathlib.Path]:
    """Files under daemon/ that define `class name`."""
    pattern = rf"^class {re.escape(name)}\b"
    return [path for path, text in _sources() if re.search(pattern, text, re.M)]


def _constructed(name: str) -> pathlib.Path | None:
    """A file under daemon/ that calls `name(...)` without defining it."""
    for path, text in _sources():
        if f"class {name}" in text:
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", text):
            return path
    return None


def _task_callers(task: Task) -> list[pathlib.Path]:
    """Files that route work to this Task. tasks.py declares them and config.py
    tabulates them; neither is a caller."""
    return [
        path
        for path, text in _sources()
        if path.name not in ("tasks.py", "config.py") and f"Task.{task.name}" in text
    ]


# --- providers ---------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(PROVIDER_KEY_ENV))
def test_every_nameable_provider_can_be_built(provider: str) -> None:
    """A provider a route may name must be a provider the app can construct.
    Anything else means a configuration that validates and then dies."""
    module = DAEMON / "llm" / "providers" / f"{provider}.py"
    app = (DAEMON / "app.py").read_text(encoding="utf-8")
    reachable = module.exists() and (provider.upper() in app or f'"{provider}"' in app)

    if provider in PENDING_PROVIDERS:
        assert not reachable, (
            f"{provider} is now reachable - remove it from PENDING_PROVIDERS. "
            "A stale entry here is how this check stops working."
        )
        return
    assert reachable, (
        f"{provider} is nameable in a route but the app cannot build it. "
        f"Either implement daemon/llm/providers/{provider}.py and assemble it, or "
        f"declare it in PENDING_PROVIDERS with the milestone that owns it."
    )


def test_a_route_to_an_unbuildable_provider_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not mid-conversation. The whole point of validating configuration eagerly.

    There is no real unbuildable provider left, so one is invented: the guarantee
    is that *any* future name added to the config surface without an
    implementation dies at startup, and that has to stay tested even while the
    gap it was written for is closed.
    """
    from daemon import config as config_module
    from daemon.app import _build_providers
    from daemon.config import ConfigError, Settings

    monkeypatch.setitem(config_module.PROVIDER_KEY_ENV, "imaginary", None)

    # A nameable provider with no model field is a configuration mistake, and it
    # has to read as one rather than as an AttributeError out of pydantic.
    with pytest.raises(ConfigError, match="no model is set"):
        Settings(
            _env_file=None,
            DAEMON_PRESET="offline",
            DAEMON_OLLAMA_MODEL="gemma3:4b",
            DAEMON_DATA_DIR="/tmp/daemon-reachability",
            DAEMON_ROUTE_OVERRIDES={"reflection": "imaginary"},
        )

    # And with a model, it gets as far as assembly and dies there - at startup,
    # not mid-conversation.
    monkeypatch.setattr(
        config_module.Settings, "imaginary_model", "some-model", raising=False
    )
    settings = Settings(
        _env_file=None,
        DAEMON_PRESET="offline",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR="/tmp/daemon-reachability",
        DAEMON_ROUTE_OVERRIDES={"reflection": "imaginary"},
    )
    with pytest.raises(ConfigError, match="no implementation"):
        _build_providers(settings)


# --- tasks -------------------------------------------------------------------


@pytest.mark.parametrize("task", list(Task))
def test_every_task_has_a_caller(task: Task) -> None:
    """A Task nobody routes work to is a table entry pretending to be a feature."""
    callers = _task_callers(task)

    if task in PENDING_TASKS:
        assert not callers, (
            f"{task.value} now has a caller ({callers[0].name}) - remove it from "
            "PENDING_TASKS."
        )
        return
    assert callers, (
        f"nothing in daemon/ routes work to {task.value}. Either call it or declare "
        f"it in PENDING_TASKS with the milestone that owns it."
    )


# --- protocol implementations ------------------------------------------------


@pytest.mark.parametrize("name", WIRED_CLASSES)
def test_wired_implementations_are_constructed_somewhere(name: str) -> None:
    assert _constructed(name) is not None, (
        f"{name} is implemented and tested but nothing under daemon/ builds it. "
        "That is the exact shape of every wiring defect this file exists for."
    )


@pytest.mark.parametrize("name", sorted(PENDING_CLASSES))
def test_pending_implementations_are_still_pending(name: str) -> None:
    """The other direction. When voice finally gets wired this fails, which is the
    reminder to move it into WIRED_CLASSES rather than leave the gap declared."""
    where = _constructed(name)
    assert where is None, (
        f"{name} is now constructed in {where.name} - move it from PENDING_CLASSES "
        "to WIRED_CLASSES."
    )


def test_the_pending_lists_do_not_overlap_the_wired_list() -> None:
    assert not set(PENDING_CLASSES) & set(WIRED_CLASSES)


@pytest.mark.parametrize("name", sorted(PENDING_WIRING))
def test_declared_wiring_gaps_are_still_gaps(name: str) -> None:
    """The other direction for a *function* nothing calls yet.

    `PENDING_CLASSES` cannot hold these - they are not classes - and the class
    checks above cannot see them either, because the classes that own them
    (`GeminiLiveSession`, `Store`) are constructed already. So a seam can be built,
    unit-tested and never called while every assertion in this file passes, which is
    the one failure mode the whole file exists to prevent.

    When PR-2b calls one of these, this fails and the entry goes. That is the file
    working.
    """
    reason, allowed = PENDING_WIRING[name]
    callers = sorted(
        path.name for path, text in _sources() if name in text and path.name not in allowed
    )
    assert not callers, (
        f"{name} is now used in {callers} - remove it from PENDING_WIRING. {reason}"
    )


def test_every_declared_wiring_gap_names_a_real_seam() -> None:
    """A gap declared against a name nothing defines is a gap that cannot close,
    and it would sit here reading as work someone still owes."""
    for name, (_reason, allowed) in PENDING_WIRING.items():
        defining = {path.name for path, text in _sources() if f"def {name}" in text}
        assert defining, f"nothing under daemon/ defines {name}"
        assert defining <= set(allowed), (
            f"{name} is defined in {sorted(defining - set(allowed))}, which "
            "PENDING_WIRING does not list as allowed to mention it"
        )


# --- tools -------------------------------------------------------------------


@pytest.mark.parametrize("module", ["builtin", "browser", "screen"])
def test_every_tool_class_is_in_its_factory(module: str, tmp_path: pathlib.Path) -> None:
    """A tool class its factory does not return is invisible to the model, and its
    own unit tests pass regardless.

    This is the same defect as the provider list, one layer down: writing the thing
    and wiring the thing are separate acts, and only the second one ships.
    """
    import importlib

    from daemon.tools.base import Tool

    loaded = importlib.import_module(f"daemon.tools.{module}")
    if module == "builtin":
        factory, kwargs = loaded.builtin_tools, {"roots": [tmp_path]}
    elif module == "screen":
        factory, kwargs = loaded.screen_tools, {"max_px": 1536, "timeout_secs": 20.0}
    else:
        factory, kwargs = loaded.browser_tools, {}

    defined = {
        name
        for name, value in vars(loaded).items()
        # Concrete tools only: `PathScope` is not a tool, and `Tool` itself is the
        # protocol. `isinstance` against a runtime_checkable protocol only checks
        # methods, so the risk attribute is what separates them.
        if isinstance(value, type) and hasattr(value, "spec") and hasattr(value, "risk")
    }
    wired = {type(tool).__name__ for tool in factory(**kwargs)}
    assert defined == wired, (
        f"{', '.join(sorted(defined - wired))} is defined in {module}.py but not "
        f"returned by its factory, so nothing registers it and no model can call it."
    )
    assert all(isinstance(tool, Tool) for tool in factory(**kwargs))


def test_no_two_tools_answer_to_the_same_name(tmp_path: pathlib.Path) -> None:
    """`Registry.register` refuses a collision, so a clash between the built-ins and
    the browser group would surface as a crash at startup rather than here."""
    from daemon.tools.browser import browser_tools
    from daemon.tools.builtin import builtin_tools
    from daemon.tools.screen import screen_tools

    names = (
        [t.spec.name for t in builtin_tools(roots=[tmp_path])]
        + [t.spec.name for t in browser_tools()]
        + [t.spec.name for t in screen_tools(max_px=1536, timeout_secs=20.0)]
    )
    assert len(names) == len(set(names)), f"duplicate tool name in {names}"


def test_the_tool_layer_is_reachable_from_settings() -> None:
    """The config surface and the assembly have to agree, which is the failure
    `openai`/`gemini` shipped with once already: nameable and unbuildable."""
    app = (DAEMON / "app.py").read_text(encoding="utf-8")
    assert "tools_enabled" in app
    assert "tools_max_rounds" in app, "the round cap is configurable and must be passed on"
    assert "mcp_enabled" in app
    assert "browser_enabled" in app, "a setting nothing reads is a setting that lies"
    assert "browser_app" in app


def test_every_pending_entry_names_its_milestone() -> None:
    """A gap without an owner is just a gap."""
    wiring = {name: reason for name, (reason, _allowed) in PENDING_WIRING.items()}
    for label, reason in (*PENDING_TASKS.items(), *PENDING_CLASSES.items(), *wiring.items()):
        assert re.search(r"M\d", reason), f"{label} is pending with no milestone: {reason!r}"


# --- configuration a person can actually write --------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4242", ("4242",)),
        ("4242,4243", ("4242", "4243")),
        ("4242, 4243", ("4242", "4243")),
        ('["4242"]', ("4242",)),  # still accepted, for anyone who wrote one
        ("", ()),
    ],
)
def test_the_allowlist_accepts_what_a_person_would_type(
    raw: str, expected: tuple[str, ...]
) -> None:
    """.env.example documents a comma-separated list, and only JSON worked:
    pydantic-settings decodes a complex field before field validators run, so
    `4242` raised a ValidationError and `4242,4243` raised a SettingsError. The
    consequence was not cosmetic - `dm_policy=allowlist` could not be configured
    at all, and the splitting validator was dead code that looked live.

    This belongs in the reachability file because it is the same defect class:
    a configuration surface that documents something the product cannot accept.
    """
    from daemon.config import Settings

    settings = Settings(
        _env_file=None,
        DAEMON_PRESET="offline",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR="/tmp/daemon-allowlist",
        TELEGRAM_BOT_TOKEN="fake",
        TELEGRAM_ALLOWED_USER_IDS=raw,
    )
    assert settings.telegram_allowed_user_ids == expected


def test_every_documented_env_key_is_a_real_setting() -> None:
    """.env.example is the only instruction most people will read. A key that
    exists there and nowhere in the code is a documented no-op."""
    import re as _re

    from daemon.config import Settings

    example = (DAEMON.parent / ".env.example").read_text(encoding="utf-8")
    documented = set(_re.findall(r"^([A-Z][A-Z0-9_]+)=", example, _re.MULTILINE))
    aliases = {
        field.alias
        for field in Settings.model_fields.values()
        if field.alias is not None
    }
    unknown = documented - aliases
    assert not unknown, f".env.example documents keys nothing reads: {sorted(unknown)}"


def test_voice_tool_mode_is_not_a_setting() -> None:
    """Voice is pinned to `allowlist` in `run_voice`, in code, and deliberately not
    configurable. A `DAEMON_TOOLS_VOICE_MODE` that could be turned back to `ask`
    would let a spoken turn mint approval rows that lapse unanswered - nobody is
    watching a voice turn for a code - which is the silent degradation this project
    calls a dangerous failure. The one approval surface, shared with text, is the
    only management point there should be. This fails the moment such a knob is added.
    """
    from daemon.config import Settings

    aliases = {
        field.alias for field in Settings.model_fields.values() if field.alias is not None
    }
    offenders = sorted(alias for alias in aliases if "VOICE" in alias and "MODE" in alias)
    assert not offenders, (
        f"a voice tool-mode knob exists ({offenders}); voice must stay pinned to "
        "allowlist in code, not be configurable back to a mode that can park a call"
    )


# --- the check that was itself unreachable -----------------------------------


def test_no_name_this_file_reasons_about_is_defined_twice() -> None:
    """A duplicated class name makes the checks above ambiguous.

    Found by an agent that had just added `LocalSpeaker` to
    `daemon/proactivity/speaker.py` while a dead one of the same name sat in
    `daemon/voice/audio.py`. `_constructed` skips any file that *defines* the name
    and returns the first that *calls* it, so with two definitions it can only say
    "something builds a LocalSpeaker" - never which. A dead implementation with a
    hundred lines of tests was being reported as wired.

    Scoped to the names this file actually reasons about rather than every class
    under `daemon/`. The broader rule would fail today on `Verdict`
    (`daemon/setup.py` and `daemon/proactivity/base.py`), and that pair is a
    readability question, not a hole in a gate - widening a check until it needs an
    allowlist is how a check stops being run.
    """
    watched = set(WIRED_CLASSES) | set(PENDING_CLASSES)
    duplicates = {
        name: sorted(path.name for path in _defining(name))
        for name in sorted(watched)
        if len(_defining(name)) > 1
    }
    assert not duplicates, (
        "these names are checked above and defined in more than one module under "
        f"daemon/, so the checks cannot tell which one they found: {duplicates}"
    )


# --- the suite's own greenness must not depend on the shell -------------------


def test_the_environment_is_isolated_from_the_developers_own() -> None:
    """`Settings(_env_file=None, ...)` does not stop pydantic-settings reading the
    *process* environment, only the file. So a shell that had sourced `.env` - which
    is how you run the product - leaked real values into tests that had explicitly
    chosen their own, and `DAEMON_VOICE_ENABLED=true` made three allowlist cases
    above fail with a ConfigError about voice.

    Same defect class as the rest of this file, one level up: a gate whose result
    depends on something nobody declared. `conftest.CANARY` is what makes this
    falsifiable in CI, where the environment is clean and the assertion would
    otherwise be vacuous.
    """
    import os

    from tests.conftest import CANARY, CONFIG_PREFIXES

    leaked = sorted(k for k in os.environ if k.startswith(CONFIG_PREFIXES))
    assert leaked == [], f"real configuration is visible to tests: {leaked}"
    assert CANARY not in os.environ
