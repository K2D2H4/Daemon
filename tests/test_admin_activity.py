"""M5 admin, the decision log — the three read-only endpoints and what feeds them.

The feature exists because the daemon was already writing down every gate verdict,
tool call and reflection pass and showing none of them. So the properties worth
testing are about what stays *visible*, not about rendering:

  a. a tick records a round even when it did nothing, and `daemon proactive`'s
     hand-run tick records one too.
  b. a reflection pass that could not reach the model is recorded - the artifact
     file cannot say so, because a failed pass writes no artifact.
  c. empty rounds stay out of the log (288 a day would bury the real events) and
     stay in the timeline, where they are the whole point.
  d. a refused tool call is its own kind, reachable by its own filter.

Loopback `base_url` for the same reason as `test_admin.py`: the router refuses any
Host that is not loopback, which is what defeats DNS-rebinding.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from daemon.admin.activity import activity_payload, today_payload
from daemon.app import create_app
from daemon.clock import now as clock_now
from daemon.config import Settings
from daemon.memory.store import Store
from daemon.proactivity.base import Reading
from daemon.proactivity.tick import ProactiveTick

LOOPBACK = "http://127.0.0.1"


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(_env_file=None, preset="offline", data_dir=tmp_path, **kw)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store.open(tmp_path / "daemon.sqlite3")
    yield store
    store.close()


class StillPresence:
    """A `Presence` that reads the same every time - the tick needs one, and what
    it says does not matter to anything under test here."""

    async def read(self) -> Reading:
        return Reading(at=clock_now())


# --- a. every round is recorded ---------------------------------------------


@pytest.mark.asyncio
async def test_a_tick_records_a_round_even_when_it_did_nothing(
    tmp_path: Path, store: Store
) -> None:
    settings = _settings(tmp_path, proactive_enabled=True)
    tick = ProactiveTick(store, settings, StillPresence())

    await tick.run()

    rounds = store.proactive_rounds_since(since=clock_now() - timedelta(minutes=1))
    assert len(rounds) == 1, (
        "a round that considered nothing left no trace, which is exactly the state "
        "that is indistinguishable from a loop that stopped running"
    )
    assert rounds[0]["considered"] == 0
    assert rounds[0]["spoke"] == 0


@pytest.mark.asyncio
async def test_a_disabled_proactivity_records_nothing(tmp_path: Path, store: Store) -> None:
    """The scheduler does not register the job when proactivity is off, so a row
    here would be the CLI's alone and would read as a round the daemon ran."""
    settings = _settings(tmp_path, proactive_enabled=False)

    await ProactiveTick(store, settings, StillPresence()).run()

    assert store.proactive_rounds_since(since=clock_now() - timedelta(minutes=1)) == []


# --- b. a reflection that failed is still recorded ---------------------------


def test_b_a_failed_pass_is_recorded_where_the_artifact_cannot_say_so(
    tmp_path: Path, store: Store
) -> None:
    store.record_reflection_run(
        date="2026-08-09",
        status="unavailable",
        messages_read=0,
        facts=0,
        entities=0,
        observations=0,
        detail="ollama refused the connection",
        now=clock_now(),
    )

    rows = activity_payload(store, kind="reflect")["items"]

    assert len(rows) == 1
    assert rows[0]["verdict"] == "unavailable", (
        "a night the model was unreachable must not read as a night with nothing "
        "to say - on disk the two are identical, which is why the row exists"
    )
    assert "ollama refused" in rows[0]["text"]


# --- c. empty rounds: out of the log, into the timeline ----------------------


def test_c_empty_rounds_stay_out_of_the_log_and_in_the_timeline(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin "now" to midday, not the live clock: the rounds span 55 minutes back and
    # `today_payload` counts only those on the current *local* day (`local_day_start`),
    # so a live clock within an hour of local midnight drops the older rounds into
    # yesterday and the count comes up short. CI caught this at 00:16 UTC (5 != 13).
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("daemon.admin.activity.clock_now", lambda: now)
    for minutes in range(0, 60, 5):
        store.record_proactive_round(
            generated=0,
            expired=0,
            considered=0,
            spoke=0,
            declined=0,
            blocked_by="{}",
            now=now - timedelta(minutes=minutes),
        )
    store.record_proactive_round(
        generated=1,
        expired=0,
        considered=1,
        spoke=0,
        declined=0,
        blocked_by=json.dumps({"cooldown": 1}),
        now=now - timedelta(minutes=2),
    )

    rows = activity_payload(store, kind="proact")["items"]
    assert len(rows) == 1, "twelve empty rounds would have buried the one real event"
    assert rows[0]["verdict"] == "blocked"
    assert rows[0]["rule"] == "cooldown", "a block that does not name its rule is not a readout"

    today = today_payload(store, _settings(tmp_path))
    assert today["rounds"] == 13, "the day's count includes the rounds the log leaves out"
    assert today["blocked_by"] == {"cooldown": 1}
    covered = [mark for mark in today["marks"] if mark["kind"] == "round"]
    assert len(covered) == 13, (
        "every round is a segment of the timeline's coverage strip - a gap there is "
        "the only way 'the daemon was not running at 3pm' is visible"
    )
    blocked = [mark for mark in today["marks"] if mark["kind"] == "blocked"]
    assert len(blocked) == 1, "a round the gate refused is an event, not just a heartbeat"


# --- d. refusals are their own kind -----------------------------------------


def test_d_a_refused_call_is_its_own_kind(tmp_path: Path, store: Store) -> None:
    now = clock_now()
    store.record_tool_call(
        tool="read_file",
        arguments='{"path": "/etc/hosts"}',
        preview="read_file /etc/hosts",
        verdict="allow",
        mode="full",
        reason="",
        origin="owner",
        channel="telegram",
        sender_id="1",
        ran=True,
        ok=True,
        now=now,
    )
    store.record_tool_call(
        tool="run_command",
        arguments='{"command": "rm -rf /"}',
        preview="run_command rm -rf /",
        verdict="deny",
        mode="full",
        reason="not on the allowlist",
        origin="untrusted",
        channel="telegram",
        sender_id="1",
        now=now,
    )

    refused = activity_payload(store, kind="refused")["items"]
    assert [row["text"] for row in refused] == ["run_command rm -rf /"]
    assert refused[0]["rule"] == "not on the allowlist", (
        "a refusal that does not say what stopped it cannot be acted on"
    )
    # `tool` admits refusals too: "what did it try to run" includes the call a rule
    # stopped, and that is the most interesting one on the page.
    tools = activity_payload(store, kind="tool")["items"]
    assert {row["text"] for row in tools} == {
        "read_file /etc/hosts",
        "run_command rm -rf /",
    }


def test_d_a_filter_reaches_past_the_page_limit(tmp_path: Path, store: Store) -> None:
    """The `kind` filter is applied before `limit` truncates. Applied after, a page
    of sixty allowed calls hid the refusal underneath them and `?kind=refused` came
    back empty while the row was in the table."""
    now = clock_now()
    store.record_tool_call(
        tool="write_file",
        arguments="{}",
        preview="write_file /etc/hosts",
        verdict="deny",
        mode="full",
        reason="outside the allowed roots",
        origin="owner",
        channel="cli",
        sender_id=None,
        now=now - timedelta(hours=2),
    )
    for i in range(60):
        store.record_tool_call(
            tool="read_file",
            arguments="{}",
            preview=f"read_file {i}",
            verdict="allow",
            mode="full",
            reason="",
            origin="owner",
            channel="cli",
            sender_id=None,
            ran=True,
            ok=True,
            now=now,
        )

    refused = activity_payload(store, kind="refused", limit=60)["items"]

    assert [row["text"] for row in refused] == ["write_file /etc/hosts"]


# --- the endpoints themselves ------------------------------------------------


def test_endpoints_serve_the_payloads_over_loopback(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    client = TestClient(app, base_url=LOOPBACK)

    for path in ("/admin/api/activity", "/admin/api/proactive/today", "/admin/api/tools/log"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.text}"

    today = client.get("/admin/api/proactive/today").json()
    assert today["budget"]["total"] == settings.proactive_daily_budget
    assert today["cooldown_minutes"] == settings.proactive_cooldown_minutes


def test_limit_is_clamped(tmp_path: Path) -> None:
    """A hand-typed `?limit=` must not make the admin read the whole audit."""
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    assert client.get("/admin/api/activity?limit=100000").status_code == 200
    assert client.get("/admin/api/tools/log?limit=-5").status_code == 200
