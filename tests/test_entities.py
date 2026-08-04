"""Entity notes: the graph, the wiki links, and the fact that a model names files."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daemon.memory import entities
from daemon.memory.store import Store

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)


@pytest.fixture
def notes(data_dir: Path, db: Any) -> entities.EntityNotes:
    return entities.EntityNotes(data_dir, Store(db))


# --- a model chooses these filenames ----------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "../../persona/seed.md",
        "../seed",
        "memory/log/2026-08-03",
        "a\\b",
        "..",
        ".",
        ".hidden",
        "",
        "   ",
        "nul\x00byte",
        "bell\x07",
        "가" * 200,
    ],
)
def test_a_name_that_is_really_a_path_is_refused(name: str) -> None:
    """The name comes out of a model. `data/persona/seed.md` is human-owned and code
    must never write to it (non-negotiable 5), so this is that contract's front
    door."""
    with pytest.raises(entities.UnsafeName):
        entities.safe_name(name)


def test_the_boundary_is_checked_as_well_as_the_blocklist(data_dir: Path) -> None:
    """`safe_name` enumerates what is dangerous; `note_path` states what is
    allowed. Only the second is exhaustive, so it has to be reachable."""
    assert entities.note_path(data_dir, "지현").parent == entities.entities_dir(data_dir)


async def test_an_unsafe_name_never_reaches_the_filesystem(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    seed = data_dir / "persona" / "seed.md"
    seed.write_text("사람이 쓴 것", encoding="utf-8")

    with pytest.raises(entities.UnsafeName):
        await notes.note("../../persona/seed", "덮어쓰기 시도", now=NOW)

    assert seed.read_text(encoding="utf-8") == "사람이 쓴 것"


def test_a_name_is_normalised_so_one_person_is_one_note() -> None:
    """macOS stores NFD. The same Korean name typed one way and produced by a model
    the other way is a different byte string, and would become two notes."""
    decomposed = unicodedata.normalize("NFD", "지현")
    assert decomposed != "지현"  # guard: the fixture is actually testing something
    assert entities.safe_name(decomposed) == "지현"


# --- the note ---------------------------------------------------------------


async def test_a_note_is_created_with_a_header_and_a_dated_section(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    await notes.note("지현", "연희동 카페에서 만났다고 했다", kind="person", now=NOW)

    text = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert text.startswith("# 지현")
    assert "person" in text
    assert "## 2026-08-03" in text
    assert "연희동 카페에서 만났다고 했다" in text


async def test_a_second_pass_appends_rather_than_rewrites(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    await notes.note("지현", "처음 들은 이야기", now=NOW)
    await notes.note("지현", "나중에 들은 이야기", now=NOW)

    text = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert "처음 들은 이야기" in text
    assert "나중에 들은 이야기" in text
    assert text.count("# 지현") == 1


async def test_the_note_is_owner_only(notes: entities.EntityNotes, data_dir: Path) -> None:
    await notes.note("지현", "사적인 내용", now=NOW)
    mode = entities.note_path(data_dir, "지현").stat().st_mode & 0o777
    assert mode == 0o600


async def test_links_are_written_inline_so_obsidian_renders_the_graph(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    await notes.note("지현", "카페에서 만났다", links=("연희동",), now=NOW)

    text = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert "[[연희동]]" in text
    assert entities.links_in(text) == ["연희동"]


async def test_a_link_already_in_the_body_is_not_repeated(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    await notes.note("지현", "[[연희동]] 카페에서 만났다", links=("연희동",), now=NOW)

    text = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert text.count("[[연희동]]") == 1


def test_an_obsidian_alias_does_not_become_an_entity() -> None:
    """`[[name|shown]]` links to `name`; parsing the display text would invent an
    entity nobody mentioned."""
    assert entities.links_in("[[연희동|우리 동네]] 에 갔다") == []


# --- the mirror -------------------------------------------------------------


async def test_the_graph_counts_mentions_and_links(notes: entities.EntityNotes) -> None:
    await notes.note("지현", "처음", kind="person", links=("연희동",), now=NOW)
    await notes.note("지현", "두 번째", now=NOW)

    graph = dict((name, (count, linked)) for name, count, linked in notes.graph())
    assert graph["지현"][0] == 2
    assert graph["지현"][1] == ["연희동"]
    # The link reads from the other end too, without 연희동 having earned a note.
    assert graph["연희동"][1] == ["지현"]


async def test_a_linked_entity_gets_a_row_but_no_note(
    notes: entities.EntityNotes, data_dir: Path
) -> None:
    """Creating a note for every mentioned name would fill the graph with stubs."""
    await notes.note("지현", "카페에서 만났다", links=("연희동",), now=NOW)

    assert not entities.note_path(data_dir, "연희동").exists()
    assert any(name == "연희동" for name, _, _ in notes.graph())


async def test_a_kind_learned_once_survives_a_later_mention(
    notes: entities.EntityNotes, db: Any
) -> None:
    await notes.note("지현", "처음", kind="person", now=NOW)
    await notes.note("지현", "두 번째", now=NOW)

    row = Store(db).entity_by_name("지현")
    assert row is not None
    assert row["kind"] == "person"


async def test_an_unsafe_link_is_dropped_not_fatal(
    notes: entities.EntityNotes, db: Any
) -> None:
    """One bad name out of a model must not lose the whole reflection pass."""
    await notes.note("지현", "카페에서 만났다", links=("연희동", "../../etc/passwd"), now=NOW)

    names = {name for name, _, _ in notes.graph()}
    assert names == {"지현", "연희동"}


async def test_a_self_link_is_dropped(notes: entities.EntityNotes) -> None:
    await notes.note("지현", "혼잣말", links=("지현",), now=NOW)
    graph = dict((name, linked) for name, _, linked in notes.graph())
    assert graph["지현"] == []


# --- write order and rebuild ------------------------------------------------


async def test_a_failed_note_write_leaves_the_mirror_untouched(
    notes: entities.EntityNotes, monkeypatch: pytest.MonkeyPatch, db: Any
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(entities, "_append_blocking", boom)

    with pytest.raises(OSError):
        await notes.note("지현", "기록 실패", now=NOW)

    assert Store(db).entity_by_name("지현") is None


async def test_the_mirror_rebuilds_from_the_notes(
    notes: entities.EntityNotes, data_dir: Path, db: Any
) -> None:
    await notes.note("지현", "처음", kind="person", links=("연희동",), now=NOW)
    await notes.note("지현", "두 번째", now=NOW)

    store = Store(db)
    store.conn.executescript("DELETE FROM entity_links; DELETE FROM entities;")
    store.conn.commit()

    assert entities.rebuild(data_dir, store) == 1  # one note file on disk

    row = store.entity_by_name("지현")
    assert row is not None
    assert row["mention_count"] == 2  # two dated sections
    assert [other["name"] for other in store.links_for(int(row["id"]))] == ["연희동"]


async def test_rebuilding_twice_does_not_double_the_count(
    notes: entities.EntityNotes, data_dir: Path, db: Any
) -> None:
    """The count is implied by the sections, so a rebuild that incremented would
    inflate it on every run."""
    await notes.note("지현", "처음", now=NOW)
    await notes.note("지현", "두 번째", now=NOW)

    store = Store(db)
    entities.rebuild(data_dir, store)
    entities.rebuild(data_dir, store)

    row = store.entity_by_name("지현")
    assert row is not None
    assert row["mention_count"] == 2


def test_rebuilding_with_no_notes_is_zero(data_dir: Path, db: Any) -> None:
    assert entities.rebuild(data_dir, Store(db)) == 0


def test_a_note_with_a_torn_tail_still_reads(data_dir: Path, db: Any) -> None:
    """The note is the source of truth, so a write cut mid-record must not make the
    whole file unreadable - the same reason `log.read` decodes with replacement."""
    path = entities.entities_dir(data_dir) / "지현.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = "# 지현\n\n## 2026-08-03\n연희동에서 \xed\xa0"
    path.write_bytes(torn.encode("utf-8", "surrogatepass"))

    assert entities.rebuild(data_dir, Store(db)) == 1
