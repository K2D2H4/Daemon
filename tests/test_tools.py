"""The tools themselves, the registry, and the runner that audits them.

Nothing here reaches outside `tmp_path`, and the commands that do run are ones any
POSIX machine has. The runner tests use the real `Store` rather than a stub,
because "every executed call leaves an audit row" is a claim about the two of them
together.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from daemon import clock
from daemon.llm.base import ImageBlock, ToolCall, ToolSpec
from daemon.memory.store import Store
from daemon.tools.base import Registry, ToolError, ToolOutput, canonical_arguments
from daemon.tools.builtin import (
    NOTIFY_SCRIPT,
    ListDir,
    Notify,
    OpenPath,
    PathScope,
    ReadFile,
    RunCommand,
    SystemState,
    WriteFile,
    builtin_tools,
)
from daemon.tools.policy import ANY_CHANNEL, Command, ToolPolicy
from daemon.tools.runner import ToolRunner, TurnContext

OWNER = "5502877373"
CONTEXT = TurnContext(origin="owner", channel="telegram", sender_id=OWNER)


@pytest.fixture
def scope(tmp_path: Path) -> PathScope:
    return PathScope([tmp_path])


@pytest.fixture
def store(db: sqlite3.Connection) -> Store:
    return Store(db)


def script(tmp_path: Path, body: str) -> str:
    """A command that runs `body` as Python, without needing shell quoting.

    `python -c "print('x')"` cannot be used: `run_command` splits with `shlex`, so
    the quotes are consumed and the program sees a bare `print(x)`. A file has no
    quoting problem to have.
    """
    path = tmp_path / "probe.py"
    path.write_text(body, encoding="utf-8")
    return f"{sys.executable} {path}"


# --- path scoping -----------------------------------------------------------


async def test_a_path_outside_every_root_is_refused(scope: PathScope) -> None:
    with pytest.raises(ToolError) as caught:
        scope.resolve("/etc/passwd")
    assert "outside" in str(caught.value)


async def test_dot_dot_cannot_climb_out(scope: PathScope, tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        scope.resolve(str(tmp_path / ".." / ".." / "etc" / "passwd"))


async def test_a_symlink_out_of_the_root_is_refused(scope: PathScope, tmp_path: Path) -> None:
    """The reason resolution happens before the check rather than after it: a link
    that lives in an allowed place pointing somewhere else is the whole trick."""
    (tmp_path / "escape").symlink_to("/etc")
    with pytest.raises(ToolError) as caught:
        scope.resolve(str(tmp_path / "escape" / "passwd"))
    assert "outside" in str(caught.value)


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "id_rsa", "id_ed25519", "server.pem", "key.key"]
)
async def test_credential_files_are_refused_inside_an_allowed_root(
    scope: PathScope, tmp_path: Path, name: str
) -> None:
    """`.env` is this daemon's own API keys. A companion that can be talked into
    reading them out has handed over the owner's billing."""
    (tmp_path / name).write_text("ANTHROPIC_API_KEY=sk-live-secret")
    with pytest.raises(ToolError) as caught:
        scope.resolve(str(tmp_path / name))
    assert "credentials" in str(caught.value)


@pytest.mark.parametrize("directory", [".ssh", ".aws", ".gnupg", "Keychains"])
async def test_credential_directories_are_refused_at_any_depth(
    scope: PathScope, tmp_path: Path, directory: str
) -> None:
    target = tmp_path / "deep" / directory / "nested" / "thing"
    target.parent.mkdir(parents=True)
    target.write_text("secret")
    with pytest.raises(ToolError) as caught:
        scope.resolve(str(target))
    assert "off limits" in str(caught.value)


async def test_the_sqlite_mirror_is_not_readable_as_a_file(
    scope: PathScope, tmp_path: Path
) -> None:
    (tmp_path / "daemon.sqlite3").write_bytes(b"SQLite format 3")
    with pytest.raises(ToolError):
        scope.resolve(str(tmp_path / "daemon.sqlite3"))


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
async def test_a_path_that_is_not_a_path_is_refused(scope: PathScope, bad: Any) -> None:
    with pytest.raises(ToolError):
        scope.resolve(bad)


async def test_a_scope_with_no_usable_root_refuses_to_exist() -> None:
    """Better than a scope that silently allows nothing and reads as broken tools."""
    with pytest.raises(ValueError):
        PathScope([])


# --- file tools -------------------------------------------------------------


async def test_list_dir_marks_directories(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hi")
    (tmp_path / "projects").mkdir()
    out = await ListDir(scope).run({"path": str(tmp_path)})
    assert "notes.md" in out
    assert "projects/" in out


async def test_list_dir_says_so_when_empty(scope: PathScope, tmp_path: Path) -> None:
    assert "empty" in await ListDir(scope).run({"path": str(tmp_path)})


async def test_list_dir_caps_a_huge_directory(scope: PathScope, tmp_path: Path) -> None:
    for index in range(30):
        (tmp_path / f"f{index:03d}").write_text("x")
    out = await ListDir(scope, max_entries=10).run({"path": str(tmp_path)})
    assert "and 20 more" in out


async def test_list_dir_on_a_file_explains_itself(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hi")
    with pytest.raises(ToolError) as caught:
        await ListDir(scope).run({"path": str(tmp_path / "notes.md")})
    assert "not a directory" in str(caught.value)


async def test_read_file_returns_the_text(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("어제 얘기했던 그거", encoding="utf-8")
    assert "어제" in await ReadFile(scope).run({"path": str(tmp_path / "notes.md")})


async def test_read_file_truncates_and_says_how_much_it_dropped(
    scope: PathScope, tmp_path: Path
) -> None:
    (tmp_path / "big.txt").write_text("x" * 5000)
    out = await ReadFile(scope, max_output=1000).run({"path": str(tmp_path / "big.txt")})
    assert len(out) < 1200
    assert "4000 more characters" in out


async def test_read_file_survives_a_bad_byte(scope: PathScope, tmp_path: Path) -> None:
    """A log with one broken byte in it is still worth reading."""
    (tmp_path / "log").write_bytes(b"fine \xff\xfe still fine")
    assert "still fine" in await ReadFile(scope).run({"path": str(tmp_path / "log")})


async def test_read_file_that_is_missing_explains_itself(
    scope: PathScope, tmp_path: Path
) -> None:
    with pytest.raises(ToolError) as caught:
        await ReadFile(scope).run({"path": str(tmp_path / "nope.md")})
    assert "does not exist" in str(caught.value)


# --- read_file: documents, not just text ------------------------------------
#
# The measured bug: `read_file` on a PDF decoded its bytes as UTF-8 and handed
# back 32% replacement characters, which the model turned into a confabulated
# "internal error". These build the four common document formats by hand - no
# library, no network, no fixture files - and assert the *content* comes out.


def _make_pdf(text: str) -> bytes:
    """A minimal one-page PDF whose text object is Flate-compressed, exactly like
    the real resume that surfaced the bug - so the raw bytes do NOT contain the
    text and the test can only pass through genuine extraction. Offsets are
    computed so the xref is valid. ASCII only: a simple Helvetica font cannot
    encode CJK, so Korean coverage lives in the Office formats below."""
    import zlib

    content = zlib.compress(f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1"))
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length %d/Filter/FlateDecode>>stream\n%s\nendstream" % (len(content), content),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)


def _zip_bytes(members: dict[str, str]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    # DEFLATED, not the default STORED: a stored member leaves its text as
    # plaintext inside the archive, and a raw UTF-8 decode would then "find" it
    # without any real extraction - a test that passes for the wrong reason.
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buf.getvalue()


def _make_docx(paragraphs: list[str]) -> bytes:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    return _zip_bytes({"word/document.xml": doc})


def _make_pptx(slides: list[str]) -> bytes:
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    members = {}
    for index, text in enumerate(slides, start=1):
        pns = "http://schemas.openxmlformats.org/presentationml/2006/main"
        members[f"ppt/slides/slide{index}.xml"] = (
            f'<?xml version="1.0"?><p:sld xmlns:p="{pns}" xmlns:a="{ns}">'
            f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>"
        )
    return _zip_bytes(members)


def _make_xlsx(strings: list[str]) -> bytes:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    cells = "".join(
        f'<row><c t="s"><v>{i}</v></c></row>' for i in range(len(strings))
    )
    return _zip_bytes(
        {
            "xl/sharedStrings.xml": f'<sst xmlns="{ns}">{shared}</sst>',
            "xl/worksheets/sheet1.xml": (
                f'<worksheet xmlns="{ns}"><sheetData>{cells}</sheetData></worksheet>'
            ),
        }
    )


async def test_read_file_extracts_text_from_a_pdf(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "resume.pdf").write_bytes(_make_pdf("Daehyun Kim Senior Engineer"))
    out = await ReadFile(scope).run({"path": str(tmp_path / "resume.pdf")})
    assert "Daehyun Kim Senior Engineer" in out


async def test_read_file_says_when_a_pdf_has_no_extractable_text(
    scope: PathScope, tmp_path: Path
) -> None:
    """A scanned resume is images, not text. Better to say so than return empty
    and let the model claim it read something."""
    (tmp_path / "scan.pdf").write_bytes(_make_pdf(""))
    out = await ReadFile(scope).run({"path": str(tmp_path / "scan.pdf")})
    assert "no" in out.lower() and "text" in out.lower()


async def test_read_file_extracts_korean_from_a_docx(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "note.docx").write_bytes(_make_docx(["대현 님 이력서", "시니어 엔지니어"]))
    out = await ReadFile(scope).run({"path": str(tmp_path / "note.docx")})
    assert "대현 님 이력서" in out
    assert "시니어 엔지니어" in out


async def test_read_file_extracts_text_from_a_pptx(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "deck.pptx").write_bytes(_make_pptx(["첫 슬라이드", "두 번째 슬라이드"]))
    out = await ReadFile(scope).run({"path": str(tmp_path / "deck.pptx")})
    assert "첫 슬라이드" in out
    assert "두 번째 슬라이드" in out


async def test_read_file_extracts_text_from_an_xlsx(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "sheet.xlsx").write_bytes(_make_xlsx(["매출", "이름", "김대현"]))
    out = await ReadFile(scope).run({"path": str(tmp_path / "sheet.xlsx")})
    assert "매출" in out
    assert "김대현" in out


async def test_read_file_refuses_an_opaque_binary_honestly(
    scope: PathScope, tmp_path: Path
) -> None:
    """The heart of the bug: don't hand back binary garbage as if it were text.
    A PNG (has a NUL byte) is not readable, and read_file must say that plainly."""
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40)
    with pytest.raises(ToolError) as caught:
        await ReadFile(scope).run({"path": str(tmp_path / "photo.png")})
    message = str(caught.value).lower()
    assert "text" in message  # it explains it is not text, not a fake error


async def test_read_file_refuses_a_document_too_large_on_disk(
    scope: PathScope, tmp_path: Path
) -> None:
    """The document path must honour the same input-size ceiling READ_MAX_BYTES
    gives the text path - a measured guard (a huge file blew RSS past 600 MB).
    Extraction runs before any bounded read, so the ceiling has to be checked up
    front from the file size on disk."""
    (tmp_path / "huge.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 500)
    reader = ReadFile(scope, max_document_bytes=200)
    with pytest.raises(ToolError) as caught:
        await reader.run({"path": str(tmp_path / "huge.pdf")})
    assert "large" in str(caught.value).lower()


async def test_read_file_refuses_a_decompression_bomb_document(
    scope: PathScope, tmp_path: Path
) -> None:
    """An OOXML file is tiny on disk but can expand to gigabytes - the realistic
    hazard when the owner reads a document a third party sent. A member that
    decompresses past the ceiling must be refused, not fully expanded into memory."""
    (tmp_path / "bomb.docx").write_bytes(_make_docx(["A" * 3_000_000]))
    assert (tmp_path / "bomb.docx").stat().st_size < 100_000  # tiny on disk
    reader = ReadFile(scope, max_uncompressed=1_000_000)
    with pytest.raises(ToolError) as caught:
        await reader.run({"path": str(tmp_path / "bomb.docx")})
    message = str(caught.value).lower()
    assert "large" in message or "expand" in message


async def test_write_file_creates_then_overwrites(scope: PathScope, tmp_path: Path) -> None:
    write = WriteFile(scope)
    target = str(tmp_path / "deep" / "notes.md")
    assert "created" in await write.run({"path": target, "content": "first"})
    assert "overwrote" in await write.run({"path": target, "content": "second"})
    assert (tmp_path / "deep" / "notes.md").read_text() == "second"


async def test_write_file_needs_a_string(scope: PathScope, tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        await WriteFile(scope).run({"path": str(tmp_path / "x"), "content": {"not": "text"}})


async def test_write_file_previews_the_size_not_the_content(
    scope: PathScope, tmp_path: Path
) -> None:
    """The preview goes into an approval message; pasting a whole file into it
    would bury the code the owner has to answer with."""
    preview = WriteFile(scope).preview({"path": "/tmp/x", "content": "y" * 900})
    assert "900 characters" in preview
    assert "yyy" not in preview


# --- run_command ------------------------------------------------------------


async def test_run_command_returns_output(scope: PathScope) -> None:
    out = await RunCommand(scope).run({"command": f"{sys.executable} -c print(41+1)"})
    assert out.strip() == "42"


async def test_run_command_reports_a_non_zero_exit_without_raising(scope: PathScope) -> None:
    """A failing command is information. Raising would hide the output the model
    needs in order to act on it."""
    out = await RunCommand(scope).run({"command": f"{sys.executable} -c exit(3)"})
    assert "exited 3" in out


async def test_run_command_kills_what_overruns(scope: PathScope) -> None:
    with pytest.raises(ToolError) as caught:
        await RunCommand(scope, timeout_secs=0.2).run({"command": "sleep 5"})
    assert "stopped" in str(caught.value)


async def test_run_command_truncates_a_flood(scope: PathScope, tmp_path: Path) -> None:
    out = await RunCommand(scope, max_output=500).run(
        {"command": script(tmp_path, "print('x' * 20000)")}
    )
    assert "more characters" in out
    assert len(out) < 700


async def test_run_command_says_when_a_program_is_missing(scope: PathScope) -> None:
    with pytest.raises(ToolError) as caught:
        await RunCommand(scope).run({"command": "definitely-not-installed-anywhere"})
    assert "not installed" in str(caught.value)


async def test_run_command_runs_in_the_requested_directory(
    scope: PathScope, tmp_path: Path
) -> None:
    (tmp_path / "here").mkdir()
    out = await RunCommand(scope).run(
        {"command": f"{sys.executable} -c print(1)", "cwd": str(tmp_path / "here")}
    )
    assert out.strip() == "1"


async def test_run_command_refuses_a_cwd_outside_the_scope(scope: PathScope) -> None:
    with pytest.raises(ToolError) as caught:
        await RunCommand(scope).run({"command": "sleep 0", "cwd": "/etc"})
    assert "outside" in str(caught.value)


async def test_run_command_defaults_to_the_first_root(scope: PathScope, tmp_path: Path) -> None:
    """Asked where it is rather than compared against the fixture, so this fails if
    the default changes to the daemon's own working directory - which is where the
    `.env` lives."""
    out = await RunCommand(scope).run(
        {"command": script(tmp_path, "import os; print(os.getcwd())")}
    )
    assert out.strip() == str(scope.roots[0])


async def test_run_command_gets_no_stdin(scope: PathScope, tmp_path: Path) -> None:
    """Otherwise a command that reads stdin waits for the timeout instead of
    finishing - and the daemon has no terminal to type into."""
    out = await RunCommand(scope, timeout_secs=5).run(
        {"command": script(tmp_path, "import sys; print(len(sys.stdin.read()))")}
    )
    assert out.strip() == "0"


# --- open_path --------------------------------------------------------------


async def test_open_path_passes_a_url_through_unscoped(scope: PathScope) -> None:
    """A URL has no place on the filesystem to check; it is approval-checked
    instead, which is why the tool is guarded rather than safe."""
    argv = OpenPath(scope).argv({"target": "https://example.com/a"})
    assert argv[-1] == "https://example.com/a"


async def test_open_path_scopes_a_local_path(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hi")
    argv = OpenPath(scope).argv({"target": str(tmp_path / "notes.md")})
    assert argv[-1] == str((tmp_path / "notes.md").resolve())
    with pytest.raises(ToolError):
        OpenPath(scope).argv({"target": "/etc/passwd"})


@pytest.mark.parametrize("target", ["file:///etc/passwd", "javascript:alert(1)", "ftp://x/y"])
async def test_open_path_only_opens_http(scope: PathScope, target: str) -> None:
    with pytest.raises(ToolError) as caught:
        OpenPath(scope).argv({"target": target})
    assert "not opened" in str(caught.value)


# --- notify -----------------------------------------------------------------


async def test_notify_never_builds_a_script_out_of_its_arguments() -> None:
    """The injection this design exists to prevent: with the obvious
    `-e 'display notification "TEXT"'`, a title carrying a double quote closes the
    string and everything after it is AppleScript running as the owner.

    So the script is a constant, and the text arrives as argv.
    """
    hostile = 'x" \n do shell script "rm -rf ~" \n display notification "'
    for line in NOTIFY_SCRIPT:
        assert "rm -rf" not in line
        assert hostile not in line
        # No interpolation anywhere in the script: it only ever reads `argv`.
        assert "{" not in line and "%" not in line
    assert any("item 1 of argv" in line for line in NOTIFY_SCRIPT)
    assert any("item 2 of argv" in line for line in NOTIFY_SCRIPT)


async def test_notify_needs_something_to_say() -> None:
    with pytest.raises(ToolError):
        await Notify().run({"title": "  ", "body": ""})


# --- registry ---------------------------------------------------------------


async def test_the_registry_keeps_the_order_it_was_given(tmp_path: Path) -> None:
    """Stable schema order between turns, so prefix caching on the hosted providers
    is not defeated by a set's iteration order."""
    registry = Registry()
    for tool in builtin_tools(roots=[tmp_path]):
        registry.register(tool)
    assert registry.names() == (
        "list_dir",
        "read_file",
        "system_state",
        "notify",
        "write_file",
        "run_command",
        "open_path",
    )
    assert [spec.name for spec in registry.specs()] == list(registry.names())


async def test_the_registry_refuses_a_duplicate_name(tmp_path: Path) -> None:
    """Two tools under one name means one is unreachable, and which one would
    depend on import order. An MCP server colliding with a built-in is the
    realistic way to get here."""
    registry = Registry()
    registry.register(ListDir(PathScope([tmp_path])))
    with pytest.raises(ValueError):
        registry.register(ListDir(PathScope([tmp_path])))


async def test_every_builtin_declares_a_usable_schema(tmp_path: Path) -> None:
    for tool in builtin_tools(roots=[tmp_path]):
        assert tool.spec.name and tool.spec.description
        assert tool.spec.parameters["type"] == "object"
        assert tool.risk in ("safe", "guarded")
        # Every guarded tool must be previewable, because the preview is what the
        # owner is shown when asked to approve it.
        assert tool.preview({})


async def test_canonical_arguments_survives_something_unserialisable() -> None:
    """A bad call still has to be recorded, so this must not raise."""
    assert "object" in canonical_arguments({"x": object()})


# --- the runner -------------------------------------------------------------


def runner(store: Store, tmp_path: Path, **kw: Any) -> ToolRunner:
    registry = Registry()
    for tool in builtin_tools(roots=[tmp_path]):
        registry.register(tool)
    return ToolRunner(registry, ToolPolicy(store, **kw), store)


async def test_an_allowed_call_runs_and_leaves_an_audit_row(
    store: Store, tmp_path: Path
) -> None:
    (tmp_path / "notes.md").write_text("hello")
    outcome = await runner(store, tmp_path, mode="ask").execute(
        [ToolCall(id="1", name="read_file", arguments={"path": str(tmp_path / "notes.md")})],
        CONTEXT,
    )
    assert outcome.results[0].ok
    assert "hello" in outcome.results[0].content
    (row,) = store.recent_tool_calls()
    assert row["tool"] == "read_file"
    assert row["ran"] == 1 and row["ok"] == 1
    assert row["verdict"] == "allow"
    assert row["origin"] == "owner"


async def test_a_refused_call_also_leaves_an_audit_row(store: Store, tmp_path: Path) -> None:
    """A denial that leaves no trace is the same as no policy at all when someone
    later asks what happened."""
    outcome = await runner(store, tmp_path, mode="ask").execute(
        [ToolCall(id="1", name="read_file", arguments={"path": "/etc/passwd"})],
        TurnContext(origin="untrusted", channel="telegram", sender_id=OWNER),
    )
    assert not outcome.results[0].ok
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "deny" and row["ran"] == 0
    assert row["origin"] == "untrusted"


async def test_an_ask_parks_the_call_and_runs_nothing(store: Store, tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    outcome = await runner(store, tmp_path, mode="ask").execute(
        [ToolCall(id="1", name="write_file", arguments={"path": str(target), "content": "x"})],
        CONTEXT,
    )
    assert not target.exists()
    assert outcome.approvals and outcome.approvals[0].code
    assert "waiting" in outcome.results[0].content
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "ask" and row["ran"] == 0


async def test_an_approved_call_then_runs(store: Store, tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    tools = runner(store, tmp_path, mode="ask")
    outcome = await tools.execute(
        [ToolCall(id="1", name="write_file", arguments={"path": str(target), "content": "x"})],
        CONTEXT,
    )
    claimed = tools.claim(
        Command(outcome.approvals[0].code, denied=False, always=False), sender_id=OWNER
    )
    assert claimed is not None
    result = await tools.resume(claimed, CONTEXT)
    assert result.ok
    assert target.read_text() == "x"


async def test_a_drifted_approval_is_refused_at_execution_time(
    store: Store, tmp_path: Path
) -> None:
    """The gap between minting a code and spending it is a gap the arguments could
    change across, and the binding is the only thing that makes an approval an
    approval."""
    tools = runner(store, tmp_path, mode="ask")
    outcome = await tools.execute(
        [
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": str(tmp_path / "a"), "content": "x"},
            )
        ],
        CONTEXT,
    )
    claimed = tools.claim(
        Command(outcome.approvals[0].code, denied=False, always=False), sender_id=OWNER
    )
    assert claimed is not None
    drifted = type(claimed)(
        tool=claimed.tool,
        arguments={"path": str(tmp_path / "b"), "content": "somethingelse"},
        preview=claimed.preview,
        denied=False,
        fingerprint=claimed.fingerprint,
    )
    result = await tools.resume(drifted, CONTEXT)
    assert not result.ok
    assert "does not match" in result.content
    assert not (tmp_path / "b").exists()


async def test_an_unknown_tool_is_answered_not_raised(store: Store, tmp_path: Path) -> None:
    """The name came from the model, so this is a hallucination it can recover from
    once it is told what does exist."""
    outcome = await runner(store, tmp_path, mode="ask").execute(
        [ToolCall(id="1", name="delete_everything", arguments={})], CONTEXT
    )
    assert not outcome.results[0].ok
    assert "read_file" in outcome.results[0].content


async def test_a_tool_that_raises_unexpectedly_does_not_end_the_round(
    store: Store, tmp_path: Path
) -> None:
    class Exploding:
        risk = "safe"
        spec = ToolSpec(name="boom", description="raises", parameters={"type": "object"})

        def preview(self, arguments: Any) -> str:
            return "boom"

        async def run(self, arguments: Any) -> str:
            raise RuntimeError("upstream fell over")

    registry = Registry()
    registry.register(Exploding())
    registry.register(ListDir(PathScope([tmp_path])))
    tools = ToolRunner(registry, ToolPolicy(store, mode="ask"), store)

    outcome = await tools.execute(
        [
            ToolCall(id="1", name="boom", arguments={}),
            ToolCall(id="2", name="list_dir", arguments={"path": str(tmp_path)}),
        ],
        CONTEXT,
    )
    assert not outcome.results[0].ok
    assert "upstream fell over" in outcome.results[0].content
    assert outcome.results[1].ok, "one broken tool must not take the round down"


async def test_a_tool_returning_tool_output_carries_its_images(
    store: Store, tmp_path: Path
) -> None:
    """The bridge Task 1.7 adds: a tool may return `ToolOutput` instead of a bare
    `str`, and the runner must carry its images into the `ToolResult` the loop
    turns into a framed user message (Task 1.6)."""

    class SeesSomething:
        risk = "safe"
        spec = ToolSpec(name="sees_something", description="looks", parameters={"type": "object"})

        def preview(self, arguments: Any) -> str:
            return "look"

        async def run(self, arguments: Any) -> ToolOutput:
            return ToolOutput("txt", (ImageBlock(b"..", "image/jpeg"),))

    registry = Registry()
    registry.register(SeesSomething())
    tools = ToolRunner(registry, ToolPolicy(store, mode="full"), store)

    outcome = await tools.execute(
        [ToolCall(id="1", name="sees_something", arguments={})], CONTEXT
    )
    result = outcome.results[0]
    assert result.content == "txt"
    assert result.images == (ImageBlock(b"..", "image/jpeg"),)


async def test_a_tool_returning_a_plain_string_carries_no_images(
    store: Store, tmp_path: Path
) -> None:
    """Every existing tool returns a bare `str`; the bridge must not force images
    onto them."""
    (tmp_path / "notes.md").write_text("hello")
    outcome = await runner(store, tmp_path, mode="full").execute(
        [ToolCall(id="1", name="read_file", arguments={"path": str(tmp_path / "notes.md")})],
        CONTEXT,
    )
    assert outcome.results[0].images == ()


async def test_calls_run_in_the_order_they_were_asked_for(store: Store, tmp_path: Path) -> None:
    """A model that writes a file and then reads it back means those two in that
    order, and there is no latency budget here worth reordering for."""
    target = tmp_path / "notes.md"
    tools = runner(store, tmp_path, mode="full")
    outcome = await tools.execute(
        [
            ToolCall(id="1", name="write_file", arguments={"path": str(target), "content": "one"}),
            ToolCall(id="2", name="read_file", arguments={"path": str(target)}),
        ],
        CONTEXT,
    )
    assert outcome.results[1].content == "one"


async def test_every_executed_call_leaves_an_audit_row(
    store: Store, tmp_path: Path
) -> None:
    """The audit row - not a line in the reply - is how the owner sees a tool they
    did not expect. It is the ground truth `daemon tools log` reads back, written for
    every executed call and independent of whatever the model's prose chose to say."""
    (tmp_path / "notes.md").write_text("hi")
    outcome = await runner(store, tmp_path, mode="full").execute(
        [ToolCall(id="1", name="read_file", arguments={"path": str(tmp_path / "notes.md")})],
        CONTEXT,
    )
    assert outcome.results[0].ok
    (row,) = store.recent_tool_calls()
    assert row["tool"] == "read_file" and row["ran"] == 1
    assert row["preview"] == f"read {tmp_path / 'notes.md'}"


async def test_the_audit_row_keeps_an_excerpt_not_the_flood(
    store: Store, tmp_path: Path
) -> None:
    (tmp_path / "big.txt").write_text("y" * 9000)
    await runner(store, tmp_path, mode="full").execute(
        [ToolCall(id="1", name="read_file", arguments={"path": str(tmp_path / "big.txt")})],
        CONTEXT,
    )
    (row,) = store.recent_tool_calls()
    assert 0 < len(row["output_excerpt"]) <= 500


async def test_a_preview_that_raises_does_not_take_the_approval_down(
    store: Store, tmp_path: Path
) -> None:
    class BadPreview:
        risk = "guarded"
        spec = ToolSpec(name="odd", description="x", parameters={"type": "object"})

        def preview(self, arguments: Any) -> str:
            raise RuntimeError("preview is broken")

        async def run(self, arguments: Any) -> str:
            return "ran"

    registry = Registry()
    registry.register(BadPreview())
    tools = ToolRunner(registry, ToolPolicy(store, mode="ask"), store)
    outcome = await tools.execute([ToolCall(id="1", name="odd", arguments={})], CONTEXT)
    assert outcome.approvals, "the owner still has to be asked"


async def test_standing_grants_can_be_revoked(store: Store) -> None:
    from daemon import clock

    store.add_tool_allowlist_entry("run_command", "git status", now=clock.now())
    assert store.tool_allowlist("run_command") == ["git status"]
    assert store.remove_tool_allowlist_entry("git status") == 1
    assert store.tool_allowlist("run_command") == []
    assert store.remove_tool_allowlist_entry("git status") == 0


async def test_granting_the_same_pattern_twice_is_not_an_error(store: Store) -> None:
    """Which is what a user does when they have forgotten they already did."""
    from daemon import clock

    store.add_tool_allowlist_entry("run_command", "date", now=clock.now())
    store.add_tool_allowlist_entry("run_command", "date", now=clock.now())
    assert store.tool_allowlist("run_command") == ["date"]


async def test_the_environment_is_not_stripped_for_a_command(
    store: Store, tmp_path: Path
) -> None:
    """A command with no PATH and no HOME fails in ways nobody can debug from a
    chat window."""
    out = await RunCommand(PathScope([tmp_path])).run(
        {"command": script(tmp_path, "import os; print(bool(os.environ.get('PATH')))")}
    )
    assert out.strip() == "True"
    assert os.environ.get("PATH")


# --- resource bounds (regressions) -------------------------------------------


async def test_command_output_is_bounded_while_it_runs(
    scope: PathScope, tmp_path: Path
) -> None:
    """Measured before the fix: 200 MB of stdout grew RSS by 651 MB, because
    `communicate()` kept all of it and `_truncate` ran afterwards. One `cat` of the
    wrong file was enough to take the process down.

    Asserted as behaviour, since RSS is not something a unit test should measure: the
    command still completes and the result is small.
    """
    from daemon.tools.builtin import OUTPUT_READ_CAP

    body = "import sys\nchunk='x'*(1024*1024)\nfor _ in range(8): sys.stdout.write(chunk)\n"
    out = await RunCommand(scope, max_output=500, timeout_secs=60).run(
        {"command": script(tmp_path, body)}
    )
    assert "more characters" in out
    assert len(out) < 700
    assert OUTPUT_READ_CAP < 8 * 1024 * 1024, "the cap has to bite before 8 MB"


async def test_a_command_that_outfloods_the_cap_still_exits_cleanly(
    scope: PathScope, tmp_path: Path
) -> None:
    """Draining past the cap rather than stopping matters: a pipe nobody reads fills
    up and the child blocks on write forever, so the timeout would fire instead of
    the command finishing."""
    body = (
        "import sys\n"
        "chunk='y'*(1024*1024)\n"
        "for _ in range(4): sys.stdout.write(chunk)\n"
        "sys.stderr.write('done')\n"
    )
    out = await RunCommand(scope, max_output=200, timeout_secs=30).run(
        {"command": script(tmp_path, body)}
    )
    assert "exited" not in out, f"the command did not finish cleanly: {out[:120]!r}"


async def test_read_file_is_bounded_too(scope: PathScope, tmp_path: Path) -> None:
    from daemon.tools.builtin import READ_MAX_BYTES

    (tmp_path / "huge").write_text("z" * (READ_MAX_BYTES * 2))
    out = await ReadFile(scope, max_output=300).run({"path": str(tmp_path / "huge")})
    assert len(out) < 500


async def test_notify_separates_its_arguments_from_its_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `--`, a title of `--help` is read as an option to notify-send rather
    than as the text to show. Confirmed by probe before the fix."""
    seen: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        def kill(self) -> None: ...

        async def wait(self) -> None: ...

    async def spawn(*argv: str, **kw: Any) -> Process:
        seen.append(argv)
        return Process()

    monkeypatch.setattr("daemon.tools.builtin.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "daemon.tools.builtin.shutil.which", lambda _n: "/usr/bin/notify-send"
    )

    await Notify().run({"title": "--help", "body": "-x"})
    argv = list(seen[0])
    assert "--" in argv
    assert argv.index("--") < argv.index("--help"), "the title precedes the separator"


async def test_notify_on_macos_also_separates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        def kill(self) -> None: ...

        async def wait(self) -> None: ...

    async def spawn(*argv: str, **kw: Any) -> Process:
        seen.append(argv)
        return Process()

    monkeypatch.setattr("daemon.tools.builtin.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    monkeypatch.setattr("daemon.tools.builtin.shutil.which", lambda _n: "/usr/bin/osascript")

    await Notify().run({"title": "-e", "body": "x"})
    argv = list(seen[0])
    # The `-e` in the title must land after `--`, never be read as another script line.
    assert argv.index("--") < argv.index("-e", argv.index("--")) or argv[-2] == "-e"
    assert argv[-2:] == ["-e", "x"]


# --- paths coverage found nothing exercising ---------------------------------
# Collected after measuring: these are branches that ship and that no test reached.
# The parsing ones matter most - a bad regex here does not fail, it quietly reports
# the wrong number, and docs/PLAN.md 6.1's gate is going to read these.


def fake_spawn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    code: int = 0,
    hang: bool = False,
    raises: OSError | None = None,
) -> list[tuple[str, ...]]:
    seen: list[tuple[str, ...]] = []

    class Process:
        returncode = code
        killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            if hang:
                await asyncio.sleep(30)
            return stdout, stderr

        def kill(self) -> None:
            Process.killed = True

        async def wait(self) -> None: ...

        @property
        def stdout(self) -> Any:
            class Reader:
                _done = False

                async def read(self, _n: int) -> bytes:
                    if Reader._done:
                        return b""
                    Reader._done = True
                    return stdout

            return Reader()

    async def spawn(*argv: str, **kw: Any) -> Process:
        seen.append(argv)
        if raises is not None:
            raise raises
        return Process()

    monkeypatch.setattr("daemon.tools.builtin.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("daemon.tools.builtin.shutil.which", lambda n, **kw: f"/usr/bin/{n}")
    return seen


import asyncio  # noqa: E402  (used by the helper above)

# --- open_path actually executing --------------------------------------------


async def test_open_path_opens_a_url(
    scope: PathScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OpenPath.run` had no test at all - it was asserted only through `argv()`, so
    the half that spawns a process was shipping unexercised."""
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    seen = fake_spawn(monkeypatch)
    out = await OpenPath(scope).run({"target": "https://example.com/a"})
    assert "opened https://example.com/a" in out
    assert seen[0][0].endswith("open")


async def test_open_path_reports_what_the_opener_said(
    scope: PathScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    fake_spawn(monkeypatch, stderr=b"No application knows how to open that", code=1)
    with pytest.raises(ToolError) as caught:
        await OpenPath(scope).run({"target": "https://example.com/a"})
    assert "No application knows" in str(caught.value)


async def test_open_path_that_hangs_is_killed(
    scope: PathScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    fake_spawn(monkeypatch, hang=True)
    with pytest.raises(ToolError) as caught:
        await OpenPath(scope, timeout_secs=0.05).run({"target": "https://example.com/a"})
    assert "did not finish in time" in str(caught.value)


async def test_open_path_needs_an_opener(
    scope: PathScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Linux")
    monkeypatch.setattr("daemon.tools.builtin.shutil.which", lambda _n, **kw: None)
    with pytest.raises(ToolError) as caught:
        await OpenPath(scope).run({"target": "https://example.com/a"})
    assert "not available" in str(caught.value)


@pytest.mark.parametrize("target", ["", "   ", None, 42])
async def test_open_path_needs_a_target(scope: PathScope, target: Any) -> None:
    with pytest.raises(ToolError):
        OpenPath(scope).argv({"target": target})


# --- system_state, which now delegates --------------------------------------
# The probes themselves are `daemon/proactivity/presence.py`'s and are tested in
# tests/test_presence.py. What is left here is the rendering and the delegation:
# that the three-way answers survive, that a failed probe is said rather than
# dropped, and that no presence at all is admitted instead of guessed.


class FakePresence:
    """A `Presence` that returns whatever a test hands it."""

    def __init__(self, reading: Any) -> None:
        self._reading = reading
        self.reads = 0

    async def read(self) -> Any:
        self.reads += 1
        return self._reading


def reading(**kw: Any) -> Any:
    from daemon import clock
    from daemon.proactivity.base import Reading

    return Reading(at=kw.pop("at", clock.now()), **kw)


async def test_system_state_reports_what_the_presence_measured() -> None:
    presence = FakePresence(
        reading(idle_seconds=41.0, foreground_app="Warp", audio_busy=False)
    )
    out = await SystemState(presence).run({})
    assert "input idle: 41s" in out
    assert "frontmost app: Warp" in out
    assert "audio in use: no" in out
    assert presence.reads == 1, "it must ask the presence, not probe on its own"


async def test_at_the_keyboard_keeps_its_three_way_answer() -> None:
    """`at_keyboard` is None when it cannot be known, and None is not "away" -
    flattening it to a boolean is the mistake `Reading` documents against."""
    assert "at the keyboard: yes" in await SystemState(
        FakePresence(reading(idle_seconds=2.0))
    ).run({})
    assert "at the keyboard: no" in await SystemState(
        FakePresence(reading(idle_seconds=9999.0))
    ).run({})
    unknown = await SystemState(FakePresence(reading())).run({})
    assert "input idle: unknown" in unknown
    assert "at the keyboard:" not in unknown, "no idle reading means no verdict at all"


async def test_a_probe_that_failed_is_said_not_dropped() -> None:
    """Omitting it reads as "nothing to report", which is the silent degradation
    this project keeps being bitten by."""
    out = await SystemState(
        FakePresence(reading(unknown=("audio_busy: CoreAudio did not answer",)))
    ).run({})
    assert "could not read: audio_busy: CoreAudio did not answer" in out


async def test_no_presence_is_admitted_rather_than_guessed() -> None:
    """`builtin_tools()` is callable without one - the CLI lists tools without
    starting anything - and the honest answer then is that it does not know."""
    out = await SystemState().run({})
    assert "unavailable in this configuration" in out
    assert "input idle" not in out


async def test_the_local_clock_is_always_reported() -> None:
    """The question is often "is it the middle of their night", and that needs no
    probe - so it survives even when every probe fails."""
    out = await SystemState(FakePresence(reading(unknown=("everything",)))).run({})
    assert "local time:" in out and "utc:" in out and "platform:" in out


async def test_the_factory_passes_the_presence_through(tmp_path: Path) -> None:
    presence = FakePresence(reading(idle_seconds=1.0))
    tools = builtin_tools(roots=[tmp_path], presence=presence)
    state = next(t for t in tools if t.spec.name == "system_state")
    await state.run({})
    assert presence.reads == 1


# --- notify failure paths ----------------------------------------------------


async def test_notify_reports_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    fake_spawn(monkeypatch, stderr=b"not authorised", code=1)
    with pytest.raises(ToolError) as caught:
        await Notify().run({"title": "hi", "body": "there"})
    assert "not authorised" in str(caught.value)


async def test_notify_that_hangs_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daemon.tools.builtin.platform.system", lambda: "Darwin")
    fake_spawn(monkeypatch, hang=True)
    with pytest.raises(ToolError) as caught:
        await Notify(timeout_secs=0.05).run({"title": "hi", "body": "there"})
    assert "did not go out in time" in str(caught.value)


async def test_notify_without_a_notifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daemon.tools.builtin.shutil.which", lambda _n, **kw: None)
    with pytest.raises(ToolError) as caught:
        await Notify().run({"title": "hi", "body": "there"})
    assert "cannot show notifications" in str(caught.value)


# --- filesystem failure paths ------------------------------------------------


async def test_an_unreadable_directory_explains_itself(
    scope: PathScope, tmp_path: Path
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(ToolError) as caught:
            await ListDir(scope).run({"path": str(locked)})
        assert "could not be listed" in str(caught.value)
    finally:
        locked.chmod(0o755)


async def test_an_unwritable_place_explains_itself(scope: PathScope, tmp_path: Path) -> None:
    locked = tmp_path / "ro"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(ToolError) as caught:
            await WriteFile(scope).run({"path": str(locked / "x.md"), "content": "hi"})
        assert "could not be written" in str(caught.value)
    finally:
        locked.chmod(0o755)


async def test_reading_a_directory_points_at_list_dir(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    with pytest.raises(ToolError) as caught:
        await ReadFile(scope).run({"path": str(tmp_path / "adir")})
    assert "use list_dir" in str(caught.value)


async def test_a_symlink_loop_fails_cleanly_rather_than_hanging(
    scope: PathScope, tmp_path: Path
) -> None:
    """Not at resolve time - non-strict `resolve()` on 3.13 hands the path back
    unchanged - but at open time, as an ordinary ToolError rather than a hang or an
    escaped OSError."""
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    scope.resolve(str(tmp_path / "a"))  # must not raise
    with pytest.raises(ToolError) as caught:
        await ReadFile(scope).run({"path": str(tmp_path / "a")})
    assert "could not be read" in str(caught.value) or "does not exist" in str(caught.value)


def test_an_unresolvable_root_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A null byte raises ValueError, not OSError, so it escaped the constructor and
    took the usable roots down with it. Found by measuring coverage."""
    expected = Path(str(tmp_path)).resolve()
    with caplog.at_level("WARNING"):
        scope = PathScope(["\x00not-a-path", tmp_path])
    assert scope.roots == (expected,), "one bad root cost every good one"
    assert any("cannot be resolved" in r.message for r in caplog.records)


async def test_a_cwd_that_is_a_file_is_refused(scope: PathScope, tmp_path: Path) -> None:
    (tmp_path / "afile").write_text("x")
    with pytest.raises(ToolError) as caught:
        await RunCommand(scope).run({"command": "date", "cwd": str(tmp_path / "afile")})
    assert "not a directory" in str(caught.value)


async def test_a_command_of_only_quotes_names_no_program(scope: PathScope) -> None:
    with pytest.raises(ToolError) as caught:
        RunCommand(scope).argv({"command": '""'})
    assert "name a program" in str(caught.value)


async def test_a_tool_whose_close_fails_does_not_break_shutdown(
    store: Store, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown runs alongside whatever else the lifespan is unwinding, so one
    tool's teardown must not take the rest with it."""

    class Leaky:
        risk = "safe"
        spec = ToolSpec(name="leaky", description="x", parameters={"type": "object"})

        def preview(self, arguments: Any) -> str:
            return "leaky"

        async def run(self, arguments: Any) -> str:
            return "ok"

        async def aclose(self) -> None:
            raise RuntimeError("the socket was already gone")

    registry = Registry()
    registry.register(Leaky())
    runner = ToolRunner(registry, ToolPolicy(store, mode="ask"), store)
    with caplog.at_level("ERROR"):
        await runner.aclose()
    assert any("closing tool" in r.message for r in caplog.records)


async def test_a_call_needing_approval_that_cannot_be_minted_is_refused(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing to ask must not look like permission."""
    tools = runner(store, tmp_path, mode="ask")
    monkeypatch.setattr(
        "daemon.tools.policy.ToolPolicy.request", lambda *a, **kw: None
    )
    outcome = await tools.execute(
        [
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": str(tmp_path / "x"), "content": "y"},
            )
        ],
        CONTEXT,
    )
    assert not outcome.results[0].ok
    assert "could not ask" in outcome.results[0].content
    assert not outcome.approvals
    assert not (tmp_path / "x").exists()


async def test_resuming_a_tool_that_no_longer_exists_is_refused(
    store: Store, tmp_path: Path
) -> None:
    """A tool can vanish between the approval and the answer - an MCP server that
    died, or a setting turned off on restart."""
    from daemon.tools.policy import Claimed

    tools = runner(store, tmp_path, mode="ask")
    result = await tools.resume(
        Claimed(tool="gone_away", arguments={}, preview="p", denied=False, fingerprint="f"),
        CONTEXT,
    )
    assert not result.ok
    assert "no longer available" in result.content


# --- per-tool grants ---------------------------------------------------------
# `tool_allowlist` is an argv pattern; `tool_grants` is a whole tool. The second
# exists because the first cannot describe a tool with no argv, and `mode=allowlist`
# therefore refused `write_file` and every MCP tool in every configuration.


async def test_a_grant_round_trips_through_the_real_store(store: Store) -> None:
    store.add_tool_grant("write_file", now=clock.now())
    assert store.tool_grants("write_file") == [ANY_CHANNEL]
    assert store.tool_grants("run_command") == []


async def test_granting_the_same_tool_twice_is_one_row(store: Store) -> None:
    """Re-granting is what a person does when they have forgotten they already
    did, same as `add_tool_allowlist_entry`."""
    store.add_tool_grant("write_file", now=clock.now())
    store.add_tool_grant("write_file", now=clock.now())
    assert store.tool_grants("write_file") == [ANY_CHANNEL]


async def test_a_channel_scoped_grant_is_a_separate_row(store: Store) -> None:
    store.add_tool_grant("write_file", channel="telegram", now=clock.now())
    store.add_tool_grant("write_file", now=clock.now())
    assert store.tool_grants("write_file") == [ANY_CHANNEL, "telegram"]


async def test_forgetting_a_grant_leaves_the_others_alone(store: Store) -> None:
    store.add_tool_grant("write_file", channel="telegram", now=clock.now())
    store.add_tool_grant("open_path", now=clock.now())
    assert store.remove_tool_grant("write_file") == 1
    assert store.tool_grants("write_file") == []
    assert store.tool_grants("open_path") == [ANY_CHANNEL]


async def test_forgetting_a_grant_nobody_made_says_nothing_went(store: Store) -> None:
    assert store.remove_tool_grant("write_file") == 0


async def test_all_grants_are_listable_for_an_operator(store: Store) -> None:
    """`daemon tools forget` prints what is standing when it cannot match, and a
    grant nobody can see is a grant nobody can revoke."""
    store.add_tool_grant("write_file", now=clock.now())
    rows = store.all_tool_grants()
    assert [(r["tool"], r["channel"]) for r in rows] == [("write_file", ANY_CHANNEL)]


async def test_a_granted_write_runs_without_asking_and_is_audited(
    store: Store, tmp_path: Path
) -> None:
    """End to end through the runner, in the mode that refused it before: the file
    is written, nothing is parked for approval, and the audit row says so.

    A Korean filename because the preview and the audit excerpt both carry it, and
    CJK is where this repo's text handling has broken before.
    """
    target = tmp_path / "메모.md"
    store.add_tool_grant("write_file", now=clock.now())
    tools = runner(store, tmp_path, mode="allowlist")
    outcome = await tools.execute(
        [
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": str(target), "content": "안녕"},
            )
        ],
        CONTEXT,
    )
    assert outcome.results[0].ok, outcome.results[0].content
    assert target.read_text(encoding="utf-8") == "안녕"
    assert not outcome.approvals
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "allow" and row["ran"] == 1
    assert "grant" in row["reason"]
    assert row["channel"] == "telegram"
    assert "메모.md" in row["preview"]


async def test_an_ungranted_write_is_still_refused_in_allowlist_mode(
    store: Store, tmp_path: Path
) -> None:
    """The other direction, so the grant above cannot be passing for a reason that
    would let everything through."""
    target = tmp_path / "메모.md"
    outcome = await runner(store, tmp_path, mode="allowlist").execute(
        [ToolCall(id="1", name="write_file", arguments={"path": str(target), "content": "안녕"})],
        CONTEXT,
    )
    assert not outcome.results[0].ok
    assert not target.exists()


async def test_a_grant_is_read_per_call_not_cached_at_construction(
    store: Store, tmp_path: Path
) -> None:
    """The web admin writes a row into a database the running daemon already has
    open. A policy that read the grants once at startup would need a restart to
    honour one, which is the shape of a setting that lies."""
    tools = runner(store, tmp_path, mode="allowlist")
    call = ToolCall(
        id="1", name="write_file", arguments={"path": str(tmp_path / "a.md"), "content": "x"}
    )
    assert not (await tools.execute([call], CONTEXT)).results[0].ok
    store.add_tool_grant("write_file", now=clock.now())
    assert (await tools.execute([call], CONTEXT)).results[0].ok


async def test_the_channel_the_runner_reports_is_the_one_the_grant_is_matched_on(
    store: Store, tmp_path: Path
) -> None:
    """The audit column and the decision have to agree, or a grant scoped to one
    channel would be honoured on another and the row would say the wrong thing."""
    store.add_tool_grant("write_file", channel="telegram", now=clock.now())
    tools = runner(store, tmp_path, mode="allowlist")
    call = ToolCall(
        id="1", name="write_file", arguments={"path": str(tmp_path / "a.md"), "content": "x"}
    )
    assert (await tools.execute([call], CONTEXT)).results[0].ok
    elsewhere = TurnContext(origin="owner", channel="voice", sender_id=OWNER)
    assert not (await tools.execute([call], elsewhere)).results[0].ok
