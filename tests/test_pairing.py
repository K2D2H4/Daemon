"""Pairing policy and its storage. No network, no clock: `FakeClock` is the seam.

The tests that matter are the limits - one notice per code lifetime, three
pending requests per channel, an hour of validity, and an owner bootstrap that
happens exactly once. Each of them is the only thing standing between "a stranger
found the bot" and "a stranger is talking to someone's companion".
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from daemon.channels.pairing import (
    CODE_ALPHABET,
    CODE_LENGTH,
    CODE_TTL,
    MAX_PENDING,
    Pairing,
    PairingError,
    PairingStore,
    generate_code,
)
from daemon.memory.store import Store

CHANNEL = "telegram"
OWNER = "4242"
STRANGER = "9999"


class FakeClock:
    """A clock the test drives. Pairing's whole behaviour is time-dependent, and
    sleeping through an hour is not a test."""

    def __init__(self) -> None:
        self.moment = datetime(2026, 8, 3, 7, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, **delta: float) -> None:
        self.moment += timedelta(**delta)


@pytest.fixture
def store(db: sqlite3.Connection) -> Store:
    return Store(db)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pairing(store: Store, clock: FakeClock) -> Pairing:
    return Pairing(store, CHANNEL, now=clock)


def code_of(pairing: Pairing, sender_id: str) -> str:
    (request,) = [r for r in pairing.pending() if r.sender_id == sender_id]
    return request.code


# --- the code itself --------------------------------------------------------


def test_code_alphabet_leaves_out_the_glyphs_people_misread() -> None:
    assert set("0O1I").isdisjoint(CODE_ALPHABET)
    assert len(set(CODE_ALPHABET)) == len(CODE_ALPHABET)


def test_generated_codes_are_eight_upper_case_symbols_and_not_repeated() -> None:
    codes = {generate_code() for _ in range(200)}
    assert len(codes) == 200  # 32**8: a collision here means it is not random
    for code in codes:
        assert len(code) == CODE_LENGTH
        assert set(code) <= set(CODE_ALPHABET)


def test_store_satisfies_the_pairing_store_protocol(store: Store) -> None:
    assert isinstance(store, PairingStore)


# --- screening --------------------------------------------------------------


def test_unknown_sender_is_denied_and_handed_a_code(pairing: Pairing) -> None:
    decision = pairing.screen(STRANGER)

    assert decision.allowed is False
    assert decision.notice is not None
    (request,) = pairing.pending()
    assert request.sender_id == STRANGER
    assert request.code in decision.notice
    assert request.expires_at == request.created_at + CODE_TTL


def test_the_notice_quotes_nothing_the_sender_wrote(pairing: Pairing) -> None:
    """The message body is dropped, not echoed: it is untrusted text and the
    reply is the one thing a stranger can make the bot emit."""
    notice = pairing.screen(STRANGER).notice
    assert notice is not None
    assert "{" not in notice  # the template was actually filled in


def test_a_second_message_within_the_hour_earns_no_second_notice(pairing: Pairing) -> None:
    first = pairing.screen(STRANGER)
    assert first.notice is not None

    for _ in range(5):
        again = pairing.screen(STRANGER)
        assert again.allowed is False
        assert again.notice is None  # otherwise the bot is an outbound generator

    (request,) = pairing.pending()
    assert request.code in first.notice  # and the code did not rotate underneath them


def test_a_fresh_code_is_issued_once_the_old_one_expires(
    pairing: Pairing, clock: FakeClock
) -> None:
    pairing.screen(STRANGER)
    first = code_of(pairing, STRANGER)
    clock.advance(hours=1, minutes=1)

    assert pairing.pending() == []  # swept, so the slot is free again
    second = pairing.screen(STRANGER)
    assert second.notice is not None
    assert first not in second.notice


def test_only_max_pending_requests_are_ever_in_flight(pairing: Pairing) -> None:
    """Once an owner exists, the tight cap is what stops a code being guessed by
    volume - a stranger being ignored then costs nothing."""
    # Bootstrap an owner first, so the guest-phase cap is the one under test.
    pairing.screen("4242")
    pairing.approve(next(r.code for r in pairing.pending() if r.sender_id == "4242"))

    notices = [pairing.screen(str(9000 + n)).notice for n in range(MAX_PENDING + 2)]

    assert [n is not None for n in notices] == [True] * MAX_PENDING + [False, False]
    assert len(pairing.pending()) == MAX_PENDING


def test_first_run_does_not_let_strangers_crowd_out_the_owner(pairing: Pairing) -> None:
    """With a flat cap of 3, three strangers who found the bot first held every
    slot for an hour and the real owner's first message got no code - at the one
    moment the product cannot afford to be unusable, and with no way back except
    pasting a numeric id by hand."""
    for n in range(MAX_PENDING + 3):
        pairing.screen(str(9000 + n))

    # The owner arrives last and still gets a code.
    decision = pairing.screen("4242")
    assert decision.notice is not None

    # And the tight cap takes over the moment they are approved.
    code = next(r.code for r in pairing.pending() if r.sender_id == "4242")
    pairing.approve(code)
    assert pairing.screen("9999").notice is None


def test_the_pending_cap_is_per_channel(store: Store, clock: FakeClock) -> None:
    telegram = Pairing(store, "telegram", now=clock)
    other = Pairing(store, "signal", now=clock)
    for n in range(MAX_PENDING):
        telegram.screen(str(9000 + n))

    assert other.screen(STRANGER).notice is not None
    assert len(telegram.pending()) == MAX_PENDING
    assert len(other.pending()) == 1


def test_pending_is_listed_oldest_first(pairing: Pairing, clock: FakeClock) -> None:
    for n in range(3):
        pairing.screen(str(9000 + n))
        clock.advance(seconds=1)

    assert [r.sender_id for r in pairing.pending()] == ["9000", "9001", "9002"]


# --- approval ---------------------------------------------------------------


def test_approved_sender_is_allowed_and_never_paired_again(
    pairing: Pairing, store: Store
) -> None:
    pairing.screen(STRANGER)
    approval = pairing.approve(code_of(pairing, STRANGER))

    assert approval.sender_id == STRANGER
    assert store.is_allowed(CHANNEL, STRANGER) is True
    assert pairing.pending() == []  # the request is spent, not still waiting
    assert pairing.screen(STRANGER).allowed is True
    assert pairing.screen(STRANGER).notice is None


def test_only_the_first_approval_bootstraps_the_owner(pairing: Pairing) -> None:
    """Otherwise pairing twice is a privilege escalation: approve, approve again,
    and the second sender quietly becomes the owner."""
    pairing.screen(OWNER)
    pairing.screen(STRANGER)

    first = pairing.approve(code_of(pairing, OWNER))
    second = pairing.approve(code_of(pairing, STRANGER))

    assert first.is_owner is True
    assert second.is_owner is False


def test_owner_stays_the_owner_after_being_re_paired(pairing: Pairing, store: Store) -> None:
    pairing.screen(OWNER)
    pairing.approve(code_of(pairing, OWNER))
    # An approved sender cannot even get a new code, so there is nothing to
    # re-approve - which is the property being pinned down.
    assert pairing.screen(OWNER).notice is None

    owners = store.conn.execute(
        "SELECT sender_id FROM channel_pairing WHERE channel = ? AND is_owner = 1", (CHANNEL,)
    ).fetchall()
    assert [row["sender_id"] for row in owners] == [OWNER]


def test_approving_an_expired_code_fails(pairing: Pairing, clock: FakeClock) -> None:
    pairing.screen(STRANGER)
    code = code_of(pairing, STRANGER)
    clock.advance(hours=1, seconds=1)

    with pytest.raises(PairingError, match="expired"):
        pairing.approve(code)
    assert pairing.pending() == []


def test_a_code_expires_exactly_at_its_deadline(pairing: Pairing, clock: FakeClock) -> None:
    pairing.screen(STRANGER)
    code = code_of(pairing, STRANGER)
    clock.advance(seconds=CODE_TTL.total_seconds())

    with pytest.raises(PairingError, match="expired"):
        pairing.approve(code)


def test_an_unknown_code_fails(pairing: Pairing) -> None:
    with pytest.raises(PairingError, match="no pairing request"):
        pairing.approve("ABCDEFGH")


def test_a_spent_code_cannot_be_replayed(pairing: Pairing) -> None:
    pairing.screen(STRANGER)
    code = code_of(pairing, STRANGER)
    pairing.approve(code)

    with pytest.raises(PairingError, match="no pairing request"):
        pairing.approve(code)


def test_a_code_from_another_channel_approves_nobody(store: Store, clock: FakeClock) -> None:
    telegram = Pairing(store, "telegram", now=clock)
    other = Pairing(store, "signal", now=clock)
    telegram.screen(STRANGER)
    code = code_of(telegram, STRANGER)

    with pytest.raises(PairingError, match="no pairing request"):
        other.approve(code)
    assert store.is_allowed("signal", STRANGER) is False
    assert store.is_allowed("telegram", STRANGER) is False


def test_a_lower_case_paste_still_approves(pairing: Pairing) -> None:
    pairing.screen(STRANGER)
    code = code_of(pairing, STRANGER)

    assert pairing.approve(f"  {code.lower()}\n").sender_id == STRANGER


def test_approval_does_not_widen_to_other_senders(pairing: Pairing, store: Store) -> None:
    pairing.screen(OWNER)
    pairing.screen(STRANGER)
    pairing.approve(code_of(pairing, OWNER))

    assert store.is_allowed(CHANNEL, STRANGER) is False


# --- storage edges ----------------------------------------------------------


def test_a_colliding_code_is_refused_instead_of_raising(store: Store, clock: FakeClock) -> None:
    """`screen` retries on False. An IntegrityError instead would surface inside
    the inbound poll loop and end it."""
    now = clock()
    assert store.create_pairing(CHANNEL, OWNER, "ABCDEFGH", created_at=now, expires_at=now) is True
    assert (
        store.create_pairing(CHANNEL, STRANGER, "ABCDEFGH", created_at=now, expires_at=now)
        is False
    )
    assert [r["sender_id"] for r in store.pending_pairings(CHANNEL)] == [OWNER]


def test_approving_nothing_reports_nothing(store: Store, clock: FakeClock) -> None:
    assert store.approve_pairing(CHANNEL, STRANGER, approved_at=clock()) is None


def test_expiry_leaves_approved_rows_alone(
    pairing: Pairing, store: Store, clock: FakeClock
) -> None:
    pairing.screen(OWNER)
    pairing.approve(code_of(pairing, OWNER))
    clock.advance(days=30)

    assert store.expire_pairings(CHANNEL, now=clock()) == 0
    assert store.is_allowed(CHANNEL, OWNER) is True


def test_pairing_rows_survive_a_reopen(tmp_path_factory: pytest.TempPathFactory) -> None:
    """These rows are the allowlist, not a rebuildable index: a restart that
    forgot them would lock the owner out of their own daemon."""
    path = tmp_path_factory.mktemp("pairing") / "daemon.sqlite3"
    clock = FakeClock()
    first = Store.open(path)
    Pairing(first, CHANNEL, now=clock).screen(OWNER)
    pairing = Pairing(first, CHANNEL, now=clock)
    pairing.approve(code_of(pairing, OWNER))
    first.close()

    second = Store.open(path)
    try:
        assert second.is_allowed(CHANNEL, OWNER) is True
    finally:
        second.close()
