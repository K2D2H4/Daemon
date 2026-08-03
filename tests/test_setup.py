"""`daemon setup`: the conversation, the verification, and the file it writes.

Nothing here touches the network, a real key, a browser, or a developer's `.env`.
The probes are either injected fakes or the real functions with `httpx.get`
replaced by a canned response - the second kind matters, because the message a
user sees when Google refuses a Standard key is a promise this suite should hold
us to, not a string a fake made up.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from daemon import cli, setup
from daemon.setup import Checks, OllamaState, Verdict


@pytest.fixture(autouse=True)
def _sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # `_finish` validates the file it wrote through Settings, which also reads the
    # environment - so a developer's exported key must not decide the outcome.
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


GOOD_KEY = "sk-ant-api03-REALKEY9999"
GOOD_TOKEN = "8012345678:AAH-realtokenABCD"


def working_checks() -> Checks:
    """Every provider says yes. Records nothing; use `Recorder` for that."""
    return Checks(
        anthropic=lambda key, model: Verdict(True, "key works"),
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ("gemma3:4b",)),
    )


@dataclass
class Recorder:
    """Counts probe calls, so a test can assert what setup did *not* do."""

    anthropic: list[str] = field(default_factory=list)
    gemini: list[str] = field(default_factory=list)
    telegram: list[str] = field(default_factory=list)
    ollama: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)

    def checks(self, *, gemini_verdict: Verdict | None = None) -> Checks:
        def anthropic(key: str, model: str) -> Verdict:
            self.anthropic.append(key)
            return Verdict(True, "key works")

        def gemini(key: str) -> Verdict:
            self.gemini.append(key)
            return gemini_verdict or Verdict(True, "key works")

        def telegram(token: str) -> Verdict:
            self.telegram.append(token)
            return Verdict(True, "connected to @test_bot")

        def ollama(url: str) -> OllamaState:
            self.ollama.append(url)
            return OllamaState(True, f"reachable at {url} (v0.5.0)", ("gemma3:4b", "bge-m3"))

        return Checks(anthropic=anthropic, gemini=gemini, telegram=telegram, ollama=ollama)


@dataclass(frozen=True, slots=True)
class Run:
    code: int
    out: str
    env_path: Path

    @property
    def written(self) -> str:
        return self.env_path.read_text(encoding="utf-8") if self.env_path.exists() else ""


def drive(
    tmp_path: Path,
    answers: Sequence[str],
    *,
    existing: str | None = None,
    checks: Checks | None = None,
    opener: Callable[[str], object] | None = None,
    stdin: io.TextIOBase | None = None,
) -> Run:
    """Run the whole wizard against `answers`, one per prompt."""
    env_path = tmp_path / ".env"
    if existing is not None:
        env_path.write_text(existing, encoding="utf-8")
    out = io.StringIO()
    code = setup.run(
        env_path=env_path,
        stdin=stdin if stdin is not None else io.StringIO("".join(f"{a}\n" for a in answers)),
        stdout=out,
        checks=checks if checks is not None else working_checks(),
        opener=opener if opener is not None else (lambda url: True),
    )
    return Run(code, out.getvalue(), env_path)


# --- what each preset asks for -----------------------------------------------


def test_the_preset_menu_says_why_offline_has_no_voice(tmp_path: Path) -> None:
    # docs/PLAN.md 7 rests on the offline preset being real. If the menu does not
    # say that voice is the thing being traded away, the user cannot make the
    # choice the privacy promise is built on.
    result = drive(tmp_path, ["1", "gemma3:4b", GOOD_TOKEN, "y"])

    assert "Voice is not available" in result.out
    assert "privacy promise" in result.out


def test_offline_asks_for_no_hosted_key_at_all(tmp_path: Path) -> None:
    recorder = Recorder()
    result = drive(
        tmp_path,
        ["1", "gemma3:4b", GOOD_TOKEN, "y"],
        checks=recorder.checks(),
        opener=recorder.opened.append,
    )

    assert result.code == 0
    assert "ANTHROPIC_API_KEY" not in result.out
    assert "GEMINI_API_KEY" not in result.out
    assert recorder.anthropic == []
    assert recorder.gemini == []
    assert "ANTHROPIC_API_KEY" not in result.written
    # The only page it offered to open was BotFather's.
    assert recorder.opened == [setup.BOTFATHER_URL]


def test_balanced_asks_for_anthropic_but_not_for_voice(tmp_path: Path) -> None:
    recorder = Recorder()
    result = drive(
        tmp_path,
        ["2", "n", "gemma3:4b", GOOD_KEY, GOOD_TOKEN, "y"],
        checks=recorder.checks(),
    )

    assert result.code == 0
    assert recorder.anthropic == [GOOD_KEY]
    # Voice off means the hosted voice key is not required (docs/PLAN.md 6.5).
    assert recorder.gemini == []
    assert "DAEMON_VOICE_ENABLED=false" in result.written


def test_voice_asks_for_gemini_and_writes_both_model_ids(tmp_path: Path) -> None:
    recorder = Recorder()
    result = drive(
        tmp_path,
        ["2", "y", "gemma3:4b", GOOD_KEY, "AIzaGEMINIKEY", "", GOOD_TOKEN, "y"],
        checks=recorder.checks(),
    )

    assert result.code == 0
    assert recorder.gemini == ["AIzaGEMINIKEY"]
    # The Live id is the one voice uses; the text id is required by Settings all
    # the same, and the wizard fills it rather than asking twice.
    assert f"DAEMON_GEMINI_LIVE_MODEL={setup.DEFAULT_GEMINI_LIVE_MODEL}" in result.written
    assert f"DAEMON_GEMINI_MODEL={setup.DEFAULT_GEMINI_MODEL}" in result.written


def test_quality_does_not_make_ollama_a_condition_of_finishing(tmp_path: Path) -> None:
    # Nothing in `quality` generates locally, so a wizard that blocked on Ollama
    # would be standing between the user and their first message for a reason that
    # only affects recall quality.
    recorder = Recorder()
    result = drive(
        tmp_path, ["3", "n", GOOD_KEY, GOOD_TOKEN, "y"], checks=recorder.checks()
    )

    assert result.code == 0
    assert recorder.ollama == []
    assert "DAEMON_OLLAMA_MODEL" not in result.out
    # It still says embeddings are local, because they are, in every preset.
    assert "embeddings are local in every preset" in result.out


def test_missing_ollama_models_are_printed_as_commands_not_run(tmp_path: Path) -> None:
    checks = Checks(
        anthropic=lambda key, model: Verdict(True, "key works"),
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        # Ollama is up but empty: the interesting case, and the common one.
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ()),
    )
    result = drive(tmp_path, ["1", "gemma3:4b", GOOD_TOKEN, "y"], checks=checks)

    assert result.code == 0
    assert "ollama pull gemma3:4b" in result.out
    assert "ollama pull bge-m3" in result.out
    assert "they are large" in result.out


def test_unreachable_ollama_is_a_warning_not_a_dead_end(tmp_path: Path) -> None:
    checks = Checks(
        anthropic=lambda key, model: Verdict(True, "key works"),
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(False, f"not reachable at {url}"),
    )
    result = drive(tmp_path, ["1", "gemma3:4b", GOOD_TOKEN, "y"], checks=checks)

    assert result.code == 0
    assert "not reachable" in result.out
    assert "https://ollama.com" in result.out
    assert result.env_path.exists()


# --- verification happens before anything is written -------------------------


def test_a_rejected_key_is_re_asked_and_the_bad_one_is_never_written(
    tmp_path: Path,
) -> None:
    # The whole point of the module: a bad key is a sentence here, not a broken
    # conversation turn hours later.
    seen: list[str] = []

    def anthropic(key: str, model: str) -> Verdict:
        seen.append(key)
        if key == GOOD_KEY:
            return Verdict(True, "key works")
        return Verdict(False, "Anthropic rejected the key (HTTP 401).", hint="Copy it again.")

    checks = Checks(
        anthropic=anthropic,
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, "reachable", ("gemma3:4b", "bge-m3")),
    )
    result = drive(
        tmp_path,
        ["2", "n", "gemma3:4b", "sk-ant-TYPO", GOOD_KEY, GOOD_TOKEN, "y"],
        checks=checks,
    )

    assert result.code == 0
    assert seen == ["sk-ant-TYPO", GOOD_KEY]
    assert "Anthropic rejected the key" in result.out
    assert "Try again." in result.out
    assert "sk-ant-TYPO" not in result.written
    assert f"ANTHROPIC_API_KEY={GOOD_KEY}" in result.written


def test_the_key_is_checked_against_the_configured_model(tmp_path: Path) -> None:
    # Checking the default model on an install that overrides it would verify a
    # model this daemon never asks for.
    seen: list[str] = []

    def anthropic(key: str, model: str) -> Verdict:
        seen.append(model)
        return Verdict(True, "key works")

    checks = Checks(
        anthropic=anthropic,
        gemini=lambda key: Verdict(True, "ok"),
        telegram=lambda token: Verdict(True, "ok"),
        ollama=lambda url: OllamaState(True, "reachable", ("gemma3:4b", "bge-m3")),
    )
    existing = "DAEMON_PRESET=balanced\nDAEMON_ANTHROPIC_MODEL=claude-opus-4-1\n"
    drive(tmp_path, ["n", "gemma3:4b", GOOD_KEY, GOOD_TOKEN, "y"], existing=existing, checks=checks)

    assert seen == ["claude-opus-4-1"]


def test_giving_up_on_a_key_writes_nothing(tmp_path: Path) -> None:
    checks = Checks(
        anthropic=lambda key, model: Verdict(False, "Anthropic rejected the key."),
        gemini=lambda key: Verdict(True, "ok"),
        telegram=lambda token: Verdict(True, "ok"),
        ollama=lambda url: OllamaState(True, "reachable", ("gemma3:4b", "bge-m3")),
    )
    answers = ["2", "n", "gemma3:4b", "bad1", "bad2", "bad3", GOOD_TOKEN, "y"]
    result = drive(tmp_path, answers, checks=checks)

    assert result.code == 1
    assert not result.env_path.exists()


def test_a_verdict_hint_is_shown_so_the_user_can_act_on_it(tmp_path: Path) -> None:
    recorder = Recorder()
    verdict = Verdict(
        False, "Google refused the key (HTTP 403).", hint=setup.GEMINI_STANDARD_KEY_HINT
    )
    result = drive(
        tmp_path,
        ["3", "y", GOOD_KEY, "AIza-STANDARD-KEY", "AIza-AUTH-KEY", "", GOOD_TOKEN, "y"],
        checks=Recorder().checks(gemini_verdict=verdict),
        opener=recorder.opened.append,
    )

    assert "Standard" in result.out
    assert "September 2026" in result.out
    assert setup.AI_STUDIO_URL in result.out
    # Three tries, all refused by this fake, so nothing is written.
    assert result.code == 1
    assert not result.env_path.exists()


# --- the real probes, without a network --------------------------------------


def canned(status: int, body: dict[str, object] | None = None) -> Callable[..., httpx.Response]:
    def get(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(status, json=body or {}, request=httpx.Request("GET", url))

    return get


def test_a_403_from_gemini_is_reported_as_a_standard_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Google's API already refuses unrestricted Standard keys and refuses the rest
    # in September 2026. "403" alone would send the user hunting; this must name
    # the cause and the fix.
    monkeypatch.setattr(setup.httpx, "get", canned(403))

    verdict = setup.check_gemini("AIza-anything")

    assert not verdict.ok
    assert "Standard" in verdict.hint
    assert "September 2026" in verdict.hint
    assert setup.AI_STUDIO_URL in verdict.hint


def test_a_400_from_gemini_is_reported_as_an_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.httpx, "get", canned(400))

    verdict = setup.check_gemini("nonsense")

    assert not verdict.ok
    assert "not valid" in verdict.detail
    assert "Standard" not in verdict.hint


def test_the_gemini_key_is_not_put_in_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The documented form is `?key=`, which would leave the secret in every proxy
    # log between here and Google.
    seen: dict[str, object] = {}

    def get(url: str, **kwargs: object) -> httpx.Response:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    monkeypatch.setattr(setup.httpx, "get", get)
    setup.check_gemini("AIza-SECRET")

    assert "AIza-SECRET" not in str(seen["url"])
    assert seen["headers"] == {"x-goog-api-key": "AIza-SECRET"}


def test_anthropic_flags_a_model_id_that_is_not_on_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup.httpx, "get", canned(200, {"data": [{"id": "claude-haiku-4"}]})
    )

    verdict = setup.check_anthropic(GOOD_KEY, "claude-sonnet-5")

    assert verdict.ok
    assert "not in your model list" in verdict.detail
    assert "DAEMON_ANTHROPIC_MODEL" in verdict.hint


def test_anthropic_reports_a_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.httpx, "get", canned(401))

    verdict = setup.check_anthropic("sk-ant-wrong", "claude-sonnet-5")

    assert not verdict.ok
    assert "rejected the key" in verdict.detail
    assert setup.ANTHROPIC_KEYS_URL in verdict.hint


def test_telegram_reports_the_bot_it_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup.httpx, "get", canned(200, {"ok": True, "result": {"username": "my_daemon_bot"}})
    )

    verdict = setup.check_telegram(GOOD_TOKEN)

    assert verdict.ok
    assert "@my_daemon_bot" in verdict.detail


def test_a_telegram_transport_error_does_not_leak_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # httpx puts the request URL in its error messages, and for the Bot API the
    # token *is* in the URL.
    def get(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {url}")

    monkeypatch.setattr(setup.httpx, "get", get)
    verdict = setup.check_telegram(GOOD_TOKEN)

    assert not verdict.ok
    assert GOOD_TOKEN not in verdict.detail
    assert "<token>" in verdict.detail


def test_ollama_probe_reports_the_installed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, **kwargs: object) -> httpx.Response:
        body: dict[str, object] = (
            {"version": "0.5.0"}
            if url.endswith("/api/version")
            else {"models": [{"name": "bge-m3:latest"}]}
        )
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(setup.httpx, "get", get)
    state = setup.check_ollama("http://127.0.0.1:11434/")

    assert state.reachable
    assert "v0.5.0" in state.detail
    assert state.models == ("bge-m3:latest",)
    # A bare name means the `latest` tag; `ollama list` prints the tag.
    assert setup._installed(state.models, "bge-m3")


# --- the file -----------------------------------------------------------------


EXISTING = """# my own notes
DAEMON_PRESET=offline
DAEMON_DATA_DIR=/somewhere/private
DAEMON_PORT=9999

# a key setup knows nothing about
DAEMON_RECALL_LIMIT=12
"""


def test_an_existing_env_is_merged_not_replaced(tmp_path: Path) -> None:
    result = drive(tmp_path, ["gemma3:4b", GOOD_TOKEN, "y"], existing=EXISTING)

    assert result.code == 0
    written = result.written
    # Nothing of theirs is gone: comments, ordering, and keys we never heard of.
    assert "# my own notes" in written
    assert "DAEMON_DATA_DIR=/somewhere/private" in written
    assert "DAEMON_PORT=9999" in written
    assert "DAEMON_RECALL_LIMIT=12" in written
    assert "# a key setup knows nothing about" in written
    assert f"TELEGRAM_BOT_TOKEN={GOOD_TOKEN}" in written


def test_a_value_already_in_the_file_is_not_asked_for_again(tmp_path: Path) -> None:
    recorder = Recorder()
    existing = f"DAEMON_PRESET=balanced\nANTHROPIC_API_KEY={GOOD_KEY}\n"
    result = drive(
        tmp_path, ["n", "gemma3:4b", GOOD_TOKEN, "y"], existing=existing, checks=recorder.checks()
    )

    assert result.code == 0
    assert "already in .env, keeping it" in result.out
    assert recorder.anthropic == []  # not re-verified, not re-asked
    assert f"ANTHROPIC_API_KEY={GOOD_KEY}" in result.written


def test_an_in_place_update_does_not_duplicate_the_key(tmp_path: Path) -> None:
    merged = setup.merge_env(
        "DAEMON_PRESET=offline\n# comment\nTELEGRAM_BOT_TOKEN=old\n",
        {"TELEGRAM_BOT_TOKEN": "new", "ANTHROPIC_API_KEY": "added"},
    )

    assert merged.count("TELEGRAM_BOT_TOKEN") == 1
    assert "TELEGRAM_BOT_TOKEN=new" in merged
    assert "ANTHROPIC_API_KEY=added" in merged
    assert "# comment" in merged


def test_the_written_file_is_owner_only(tmp_path: Path) -> None:
    result = drive(tmp_path, ["1", "gemma3:4b", GOOD_TOKEN, "y"])

    assert result.env_path.stat().st_mode & 0o777 == 0o600


def test_a_world_readable_env_is_tightened_when_it_is_touched(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    env_path.chmod(0o644)

    result = drive(tmp_path, ["gemma3:4b", GOOD_TOKEN, "y"], existing="DAEMON_PRESET=offline\n")

    assert result.code == 0
    assert result.env_path.stat().st_mode & 0o777 == 0o600


def test_nothing_left_to_do_is_not_a_rewrite(tmp_path: Path) -> None:
    existing = (
        f"DAEMON_PRESET=offline\nDAEMON_OLLAMA_MODEL=gemma3:4b\n"
        f"TELEGRAM_BOT_TOKEN={GOOD_TOKEN}\n"
    )
    result = drive(tmp_path, [], existing=existing)

    assert result.code == 0
    assert "already configured" in result.out
    assert result.written == existing


def test_declining_the_write_leaves_the_file_alone(tmp_path: Path) -> None:
    result = drive(tmp_path, ["gemma3:4b", GOOD_TOKEN, "n"], existing=EXISTING)

    assert result.code == 1
    assert "Nothing was written." in result.out
    assert result.written == EXISTING


# --- stopping halfway ---------------------------------------------------------


def test_end_of_input_leaves_an_existing_file_untouched(tmp_path: Path) -> None:
    result = drive(tmp_path, ["gemma3:4b"], existing=EXISTING)  # runs out at the token

    assert result.code == 1
    assert "was not touched" in result.out
    assert result.written == EXISTING


def test_end_of_input_creates_no_file(tmp_path: Path) -> None:
    result = drive(tmp_path, ["2"])

    assert result.code == 1
    assert not result.env_path.exists()


def test_ctrl_c_leaves_an_existing_file_untouched(tmp_path: Path) -> None:
    class Interrupting(io.StringIO):
        def readline(self, *args: object, **kwargs: object) -> str:
            raise KeyboardInterrupt

    result = drive(tmp_path, [], existing=EXISTING, stdin=Interrupting())

    assert result.code == 1
    assert result.written == EXISTING


def test_a_file_that_still_fails_validation_is_explained_not_traced(tmp_path: Path) -> None:
    # A hand-edited `.env` can be invalid in ways the wizard never asks about.
    # Startup would reject it; setup has to say so in words.
    existing = "DAEMON_PRESET=offline\nDAEMON_RECALL_LIMIT=0\n"
    result = drive(tmp_path, ["gemma3:4b", GOOD_TOKEN, "y"], existing=existing)

    assert result.code == 1
    assert "not usable yet" in result.out
    assert "DAEMON_RECALL_LIMIT" in result.out


# --- secrets stay out of the transcript --------------------------------------


def test_no_secret_is_ever_echoed_back(tmp_path: Path) -> None:
    result = drive(
        tmp_path,
        ["2", "y", GOOD_KEY, "AIzaGEMINIKEY", "", GOOD_TOKEN, "y"],
        existing="DAEMON_OLLAMA_MODEL=gemma3:4b\n",
    )

    assert result.code == 0
    for secret in (GOOD_KEY, GOOD_TOKEN, "AIzaGEMINIKEY"):
        assert secret not in result.out, f"{secret} appeared in the transcript"
        assert secret in result.written  # it did get saved, just never printed
    # The last four characters are shown, which is how you recognise what you
    # pasted without the transcript becoming a credential.
    assert f"...{GOOD_KEY[-4:]}" in result.out
    assert f"...{GOOD_TOKEN[-4:]}" in result.out


def test_a_replaced_secret_is_masked_in_the_change_list(tmp_path: Path) -> None:
    existing = "DAEMON_PRESET=offline\nTELEGRAM_BOT_TOKEN=1111:OLDTOKENZZZZ\n"
    result = drive(
        tmp_path, ["gemma3:4b", "y"], existing=existing
    )  # token already set, so only the model is asked

    assert result.code == 0
    assert "1111:OLDTOKENZZZZ" not in result.out


def test_mask_never_shows_a_short_value(tmp_path: Path) -> None:
    assert setup.mask("abcd") == "(set)"
    assert setup.mask("") == "(empty)"
    assert setup.mask("0123456789") == "...6789"


# --- pairing is not this wizard's business -----------------------------------


def test_no_numeric_telegram_id_is_ever_requested(tmp_path: Path) -> None:
    result = drive(tmp_path, ["1", "gemma3:4b", GOOD_TOKEN, "y"])

    assert result.code == 0
    # No prompt for an id, and no trace of the @userinfobot procedure this command
    # replaced. The only mention of a numeric id is the promise not to ask for one.
    assert "TELEGRAM_ALLOWED_USER_IDS" not in result.out
    assert "userinfobot" not in result.out
    assert "numeric user id here" in result.out
    # Instead it says how pairing works, which is the channel's job to implement.
    assert "pairing code" in result.out
    assert "TELEGRAM_ALLOWED_USER_IDS" not in result.written


def test_an_existing_allowlist_suppresses_the_pairing_note(tmp_path: Path) -> None:
    existing = "DAEMON_PRESET=offline\nTELEGRAM_ALLOWED_USER_IDS=4242\n"
    result = drive(tmp_path, ["gemma3:4b", GOOD_TOKEN, "y"], existing=existing)

    assert "pairing code" not in result.out


def test_the_telegram_token_can_be_skipped(tmp_path: Path) -> None:
    result = drive(tmp_path, ["1", "gemma3:4b", "", "y"])

    assert result.code == 0
    assert "Skipped" in result.out
    assert "TELEGRAM_BOT_TOKEN" not in result.written


# --- --check ------------------------------------------------------------------


def test_check_reports_what_is_missing_and_fails(tmp_path: Path) -> None:
    out = io.StringIO()

    code = setup.run(check_only=True, env_path=tmp_path / ".env", stdout=out)

    assert code == 1
    assert "missing" in out.getvalue()
    assert "ANTHROPIC_API_KEY" in out.getvalue()
    assert "TELEGRAM_BOT_TOKEN" in out.getvalue()


def test_check_asks_nothing_and_opens_nothing(tmp_path: Path) -> None:
    # It has to be usable from CI and from documentation, so it must not block on
    # a terminal that is not there.
    opened: list[str] = []
    empty = io.StringIO("")
    out = io.StringIO()

    code = setup.run(
        check_only=True,
        env_path=tmp_path / ".env",
        stdin=empty,
        stdout=out,
        opener=opened.append,
    )

    assert code == 1
    assert opened == []
    assert ":" in out.getvalue()  # it printed a report rather than a prompt
    assert "?" not in out.getvalue()


def test_check_is_satisfied_by_a_complete_offline_install(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"DAEMON_PRESET=offline\nTELEGRAM_BOT_TOKEN={GOOD_TOKEN}\n", encoding="utf-8"
    )
    out = io.StringIO()

    code = setup.run(check_only=True, env_path=tmp_path / ".env", stdout=out)

    assert code == 0
    assert "Nothing missing" in out.getvalue()


def test_check_does_not_fail_over_a_key_that_has_a_working_default(tmp_path: Path) -> None:
    # DAEMON_OLLAMA_MODEL is offered by the wizard but has a built-in default, so
    # an install without it still starts - the exit code must not claim otherwise.
    (tmp_path / ".env").write_text(
        f"DAEMON_PRESET=offline\nTELEGRAM_BOT_TOKEN={GOOD_TOKEN}\n", encoding="utf-8"
    )
    out = io.StringIO()

    assert setup.run(check_only=True, env_path=tmp_path / ".env", stdout=out) == 0
    assert "[offered] DAEMON_OLLAMA_MODEL" in out.getvalue()


def test_check_masks_the_keys_it_found(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"DAEMON_PRESET=offline\nTELEGRAM_BOT_TOKEN={GOOD_TOKEN}\n", encoding="utf-8"
    )
    out = io.StringIO()

    setup.run(check_only=True, env_path=tmp_path / ".env", stdout=out)

    assert GOOD_TOKEN not in out.getvalue()
    assert f"...{GOOD_TOKEN[-4:]}" in out.getvalue()


def test_check_rejects_a_preset_it_does_not_know(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DAEMON_PRESET=cheap\n", encoding="utf-8")
    out = io.StringIO()

    assert setup.run(check_only=True, env_path=tmp_path / ".env", stdout=out) == 1
    assert "not a preset" in out.getvalue()


# --- wiring -------------------------------------------------------------------


def test_the_cli_exposes_setup_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # chdir'd into tmp_path by the sandbox fixture, so this reads a `.env` that
    # does not exist rather than the developer's.
    assert cli.main(["setup", "--check"]) == 1
    assert "missing" in capsys.readouterr().out


def test_setup_does_not_need_a_loadable_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every other command exits 2 on a broken config. Setup is the one that has to
    # run *because* the config is broken.
    monkeypatch.setenv("DAEMON_PRESET", "cheap")

    assert cli.main(["setup", "--check"]) == 1
    assert "bad configuration" not in capsys.readouterr().err


def test_needs_come_from_the_preset_table(tmp_path: Path) -> None:
    # One assertion that the routing table in config is what drives the questions,
    # rather than a second copy of it living here.
    offline = {need.key for need in setup.needs_for({"DAEMON_PRESET": "offline"})}
    balanced = {need.key for need in setup.needs_for({"DAEMON_PRESET": "balanced"})}
    voice = {
        need.key
        for need in setup.needs_for(
            {"DAEMON_PRESET": "balanced", "DAEMON_VOICE_ENABLED": "true"}
        )
    }

    assert "ANTHROPIC_API_KEY" not in offline
    assert "ANTHROPIC_API_KEY" in balanced
    assert "GEMINI_API_KEY" not in balanced
    assert "GEMINI_API_KEY" in voice
