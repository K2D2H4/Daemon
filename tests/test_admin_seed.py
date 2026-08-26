"""The admin's seed editor — `daemon/admin/seed_io.py` and its two routes.

This is the write half of docs/CONTRACTS.md non-negotiable 5 as amended by
docs/adr/0019: the owner's own keystrokes may reach `persona/seed.md` through the
admin form, and nothing else may. So the tests here are mostly about *refusing*
to write, and each refusal is a trap that exists in the code this feature had to
be built on top of:

  * `/admin/api/persona`'s `anchor.seed.text` is `None` whenever the shared 64 KB
    body budget was spent on diaries first (`daemon/admin/mind.py:_file_view`).
    An editor loading from that payload would save an empty seed over a real one,
    so the editor has its own unbudgeted GET and that is asserted here.
  * `daemon/persona/loader.py:read_file` swallows a `UnicodeDecodeError` into
    `""`. A CP949-saved seed therefore *reads* as empty. Saving over it must be
    impossible, not merely unlikely.
  * The file had no writer at all before this, so there was no undo. One backup
    slot is part of the write, not a nicety.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from daemon.admin.seed_io import (
    MAX_SEED_BYTES,
    SeedConflict,
    SeedRejected,
    SeedUnreadable,
    read_seed,
    write_seed,
)
from daemon.app import create_app
from daemon.config import Settings

# The admin router refuses any Host that is not loopback (test_admin.py's
# reasoning applies here unchanged).
LOOPBACK = "http://127.0.0.1"

SEED = "# 나\n\n말투는 짧게.\n"


def _settings(tmp_path: Path, **kw: object) -> Settings:
    kw.setdefault("provider", "ollama")
    return Settings(_env_file=None, data_dir=tmp_path, **kw)


def _seed_file(data_dir: Path) -> Path:
    return data_dir / "persona" / "seed.md"


def _write_seed_file(data_dir: Path, text: str) -> Path:
    path = _seed_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- read -------------------------------------------------------------------


def test_read_reports_the_text_and_a_hash_of_the_bytes_on_disk(tmp_path: Path) -> None:
    """The hash is over the file's bytes, not over the decoded text, because what
    the save has to detect is "the file changed" - including a change that only
    altered its encoding."""
    _write_seed_file(tmp_path, SEED)

    view = read_seed(tmp_path)

    assert view.exists is True
    assert view.text == SEED
    assert view.sha256 == _sha(SEED)
    assert view.file == "persona/seed.md"


def test_read_does_not_make_a_missing_seed_look_like_an_empty_one(tmp_path: Path) -> None:
    view = read_seed(tmp_path)

    assert view.exists is False
    assert view.text == ""
    assert view.sha256 == ""


def test_read_refuses_a_seed_it_cannot_decode(tmp_path: Path) -> None:
    """`loader.read_file` turns this file into `""` and the daemon runs on with no
    persona. The editor must not repeat that: an unreadable seed is the one case
    where showing an empty box would invite the owner to destroy it."""
    _seed_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _seed_file(tmp_path).write_bytes("# 나\n말투는 짧게.\n".encode("cp949"))

    with pytest.raises(SeedUnreadable):
        read_seed(tmp_path)


# --- write ------------------------------------------------------------------


def test_write_replaces_the_file_when_the_hash_still_matches(tmp_path: Path) -> None:
    _write_seed_file(tmp_path, SEED)

    saved = write_seed(tmp_path, "# 나\n\n말투는 길게.\n", expected_sha256=_sha(SEED))

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == "# 나\n\n말투는 길게.\n"
    assert saved.sha256 == _sha("# 나\n\n말투는 길게.\n")
    assert saved.lines == 3


def test_write_creates_the_seed_when_there_is_none(tmp_path: Path) -> None:
    """A fresh install that skipped `daemon setup`'s persona step has no file, and
    the empty hash is what `read_seed` handed the page for that case."""
    saved = write_seed(tmp_path, SEED, expected_sha256="")

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED
    assert saved.backup is None
    assert _seed_file(tmp_path).stat().st_mode & 0o777 == 0o600


def test_write_refuses_a_stale_hash_and_leaves_the_file_alone(tmp_path: Path) -> None:
    """The owner edits `seed.md` by hand in Obsidian - that is the whole point of
    the file - so a page left open for an hour is holding stale text."""
    _write_seed_file(tmp_path, SEED)
    _write_seed_file(tmp_path, "# 나\n\n손으로 고친 줄\n")

    with pytest.raises(SeedConflict):
        write_seed(tmp_path, "웹에서 고친 줄\n", expected_sha256=_sha(SEED))

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == "# 나\n\n손으로 고친 줄\n"


def test_write_cannot_clobber_a_seed_that_could_not_be_read(tmp_path: Path) -> None:
    """The other half of `test_read_refuses_a_seed_it_cannot_decode`: a caller who
    never got a hash cannot produce one, so the undecodable file is safe by the
    same mechanism as the stale one rather than by a second special case."""
    raw = "# 나\n말투는 짧게.\n".encode("cp949")
    _seed_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _seed_file(tmp_path).write_bytes(raw)

    with pytest.raises(SeedConflict):
        write_seed(tmp_path, SEED, expected_sha256="")

    assert _seed_file(tmp_path).read_bytes() == raw


def test_write_keeps_the_previous_text_in_one_backup_slot(tmp_path: Path) -> None:
    """`seed.md` had no writer before this, so it had no undo either."""
    _write_seed_file(tmp_path, SEED)

    saved = write_seed(tmp_path, "덮어쓴 내용\n", expected_sha256=_sha(SEED))

    assert saved.backup == "persona/seed.md.bak"
    assert (tmp_path / "persona" / "seed.md.bak").read_text(encoding="utf-8") == SEED


def test_a_failed_write_does_not_spend_the_one_backup_slot(tmp_path: Path) -> None:
    """The backup is only meaningful as "the content the last *successful* save
    replaced". Written before the seed, a write that then failed would leave the
    slot holding the same text as the file - consuming the owner's only undo for
    a save that never happened."""
    _write_seed_file(tmp_path, SEED)
    (tmp_path / "persona" / "seed.md.bak").write_text("훨씬 예전 내용\n", encoding="utf-8")

    from daemon.admin import seed_io

    real = seed_io.write_private_replace

    def fail_on_the_seed(path: Path, content: str, **kw: object) -> None:
        if path.name == "seed.md":
            raise OSError("no space left on device")
        real(path, content, **kw)  # type: ignore[arg-type]

    seed_io.write_private_replace = fail_on_the_seed  # type: ignore[assignment]
    try:
        with pytest.raises(OSError):
            write_seed(tmp_path, "새 내용\n", expected_sha256=_sha(SEED))
    finally:
        seed_io.write_private_replace = real  # type: ignore[assignment]

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED
    assert (tmp_path / "persona" / "seed.md.bak").read_text(encoding="utf-8") == "훨씬 예전 내용\n"


def test_write_keeps_the_blank_lines_the_owner_left_at_the_end(tmp_path: Path) -> None:
    """Normalisation guarantees a *final* newline; it does not get to reformat
    human-owned text. Collapsing the spacing someone left between sections is an
    edit they did not make to the one file this contract keeps as written."""
    saved = write_seed(tmp_path, "# 나\n\n짧게 말한다.\n\n\n", expected_sha256="")

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == "# 나\n\n짧게 말한다.\n\n\n"
    assert saved.sha256 == _sha("# 나\n\n짧게 말한다.\n\n\n")


def test_write_refuses_a_blank_seed(tmp_path: Path) -> None:
    """An empty seed is not a neutral state: the daemon speaks as a stock
    assistant and `daemon doctor` calls it a proactivity blocker. It is also what
    a truncated read would produce, so refusing it is the second lock on that."""
    _write_seed_file(tmp_path, SEED)

    with pytest.raises(SeedRejected):
        write_seed(tmp_path, "   \n\n", expected_sha256=_sha(SEED))

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_write_refuses_more_than_the_cap(tmp_path: Path) -> None:
    """Every byte here is re-sent to the model on every single turn."""
    _write_seed_file(tmp_path, SEED)

    with pytest.raises(SeedRejected):
        write_seed(tmp_path, "가" * MAX_SEED_BYTES, expected_sha256=_sha(SEED))

    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_write_normalises_the_line_endings_a_textarea_submits(tmp_path: Path) -> None:
    """A browser can post CRLF. The seed goes into the prompt verbatim, and a
    file that differs from what the page shows makes the next hash mismatch."""
    saved = write_seed(tmp_path, "# 나\r\n\r\n말투는 짧게.", expected_sha256="")

    text = _seed_file(tmp_path).read_text(encoding="utf-8")
    assert text == "# 나\n\n말투는 짧게.\n"
    assert saved.sha256 == _sha(text)


# --- routes -----------------------------------------------------------------


def test_the_editor_reads_the_whole_seed_even_when_the_persona_payload_truncates_it(
    tmp_path: Path,
) -> None:
    """The trap this endpoint exists for. `/admin/api/persona` shares one 64 KB
    budget across diaries, seed and learned, and the seed is last in line - so it
    comes back `text: None`. Saving that would empty the file."""
    _write_seed_file(tmp_path, SEED)
    # `persona/diary`, the directory `evolve.DIARY_SUBDIR` names. Eight bodies
    # (`MAX_DIARY_BODIES`) that each *fit* and together spend the budget exactly:
    # `BodyBudget.take` drops a body too big for what is left without charging
    # for it, so one enormous diary would starve nothing.
    for day in range(1, 9):
        diary = tmp_path / "persona" / "diary" / f"2026-08-{day:02d}.md"
        diary.parent.mkdir(parents=True, exist_ok=True)
        diary.write_text("D" * (64 * 1024 // 8), encoding="utf-8")

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        budgeted = client.get("/admin/api/persona").json()["anchor"]["seed"]
        editable = client.get("/admin/api/persona/seed")

    assert budgeted["text"] is None and budgeted["truncated"] is True
    assert editable.status_code == 200
    assert editable.json()["text"] == SEED
    assert editable.json()["sha256"] == _sha(SEED)


def test_the_editor_refuses_to_open_a_seed_it_cannot_decode(tmp_path: Path) -> None:
    _seed_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _seed_file(tmp_path).write_bytes("말투는 짧게.\n".encode("cp949"))

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.get("/admin/api/persona/seed")

    assert r.status_code == 409
    assert "seed.md" in r.json()["detail"]


def test_saving_the_seed_takes_effect_on_the_next_turn_with_no_restart(
    tmp_path: Path,
) -> None:
    """`load_persona` re-reads the file every turn on purpose (loader.py), and that
    promise is the reason the editor is worth having at all."""
    import asyncio

    from daemon.persona.loader import load_persona

    _write_seed_file(tmp_path, SEED)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.put(
            "/admin/api/persona/seed",
            json={"text": "# 나\n\n반말로 말한다.\n", "sha256": _sha(SEED)},
        )

    assert r.status_code == 200, r.text
    assert r.json()["lines"] == 3
    assert "반말로 말한다." in asyncio.run(load_persona(tmp_path))


def test_saving_a_stale_seed_is_a_409_that_names_the_conflict(tmp_path: Path) -> None:
    _write_seed_file(tmp_path, SEED)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.put(
            "/admin/api/persona/seed",
            json={"text": "웹에서 고친 줄\n", "sha256": _sha("전혀 다른 내용")},
        )

    assert r.status_code == 409
    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_saving_a_blank_seed_is_a_400(tmp_path: Path) -> None:
    _write_seed_file(tmp_path, SEED)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.put("/admin/api/persona/seed", json={"text": "  ", "sha256": _sha(SEED)})

    assert r.status_code == 400
    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_the_save_route_requires_a_hash_rather_than_defaulting_to_overwrite(
    tmp_path: Path,
) -> None:
    """A missing `sha256` must not read as "no file was there" - that is exactly
    the request a hand-rolled client would send, and it would overwrite."""
    _write_seed_file(tmp_path, SEED)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        r = client.put("/admin/api/persona/seed", json={"text": "덮어쓰기\n"})

    assert r.status_code == 400
    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_a_cross_site_put_to_the_seed_is_refused(tmp_path: Path) -> None:
    """PUT is the first unsafe verb this router serves, and every other guard test
    drives PATCH /api/settings - so nothing proved the screen covers this one. A
    page on any other origin can `fetch()` 127.0.0.1 with no preflight, and the
    thing on the other end here is the personality anchor."""
    _write_seed_file(tmp_path, SEED)

    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        cross = client.put(
            "/admin/api/persona/seed",
            json={"text": "공격자가 쓴 인격\n", "sha256": _sha(SEED)},
            headers={"Origin": "http://evil.example"},
        )
        metadata = client.put(
            "/admin/api/persona/seed",
            json={"text": "공격자가 쓴 인격\n", "sha256": _sha(SEED)},
            headers={"Sec-Fetch-Site": "cross-site"},
        )

    assert cross.status_code == 403
    assert metadata.status_code == 403
    assert _seed_file(tmp_path).read_text(encoding="utf-8") == SEED


def test_the_reload_button_asks_before_discarding_an_edit(tmp_path: Path) -> None:
    """`go()` already refuses to re-read over an unsaved edit; the button was the
    one path that skipped that guard, and there is no draft and no undo behind it.
    Asserted on the shipped page, the way the other index.html structure tests are."""
    app = create_app(_settings(tmp_path))
    with TestClient(app, base_url=LOOPBACK) as client:
        html = client.get("/admin/").text

    start = html.index("async function loadSeed(")
    body = html[start : html.index("\nasync function ", start + 1)]
    assert "seedEdited()" in body and "confirm(" in body


def test_the_only_seed_write_route_is_the_owners_own_form() -> None:
    """The replacement for `test_f_no_route_writes_the_seed`, and the same job:
    docs/adr/0019 narrowed non-negotiable 5 from "no route writes seed.md" to
    "exactly one does, and it writes only what the owner typed". A later hand
    adding a second one - a POST that took text from a model, say - fails here
    while every other test stays green."""
    from daemon.admin import routes

    seed_writes = sorted(
        (route.path, method)
        for route in routes.router.routes
        if "seed" in route.path
        for method in sorted(set(getattr(route, "methods", set())) - {"GET", "HEAD"})
    )
    assert seed_writes == [("/admin/api/persona/seed", "PUT")]
