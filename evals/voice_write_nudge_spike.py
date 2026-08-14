"""Does the voice model fake a *write* it never performs, and does naming writes in
the tool contract stop it? Ask the live socket, over the audio path.

The audit told a sharp story (daemon voice channel, all-time): over voice the model
runs reads (search, fetch, list) and the two side-effecting tools the TOOL_CONTRACT
*names* - `open_path` (23 ok) and `run_command` - but has **never once** called a
write it does not name: no `notion-create-pages`, no `write_file`. Asked to create a
Notion subpage it says "만들었어요" with no tool call. The measured deflection this
repo already documented (daemon/companion.py, the "you can act on this machine"
comment) was `open_path`; the nudge that fixed it names only "open, launch, run or
show". The hypothesis: the same native-audio deflection still bites *writes the nudge
does not name*, and naming them fixes it the same way.

This settles it on the wire, because it cannot be settled anywhere else: the
deflection is audio-path + crowded-tool-set specific (text calls the tool every time,
a 4-tool set never reproduced it - see the memory), so a mock, a unit test, or a
`send_text` spike would all show green while production fakes the write.

**Design - the crowding is held fixed, only the nudge varies.** One crowded tool set
(~80 declarations, like a real install) carrying BOTH `open_path` (a nudge-named
write, the control) and `notion__notion-create-pages` (an un-named write, the
subject). Three cells, N sessions each, driven by Korean TTS audio - no microphone:

  1. baseline nudge, "크롬 열어줘"         -> open_path call rate   (control: high == harness works)
  2. baseline nudge, "하위 페이지 만들어줘"  -> create call rate      (the bug: expect ~0)
  3. patched nudge,  "하위 페이지 만들어줘"  -> create call rate      (the fix: expect up)

If 1 is high and 2 is ~0 in the *same* tool set, crowding is not the difference - the
nudge naming is. If 3 lifts 2, the patch is the fix, measured not argued. The patched
nudge is defined here, not in the source: measure first, then move the winning text
into daemon/companion.py.

    cd ~/Daemon && python3 -m evals.voice_write_nudge_spike            # N=5 per cell
    cd ~/Daemon && python3 -m evals.voice_write_nudge_spike --runs 8

Needs GEMINI_API_KEY (+ DAEMON_GEMINI_LIVE_MODEL). Nothing is written to Notion or the
repo: create/search tool calls are answered with fabricated results so the model can
proceed, and only the *names* of the calls it makes are recorded. The key is read from
the environment and never printed or written. Not a test - it needs a key and the
network (tests/CLAUDE.md), so it lives here.

**What it found (2026-08-14, `gemini-3.1-flash-live-preview`, 80 tools, N=4):**
`open_path` (flat) 4/4, `notion-create-pages` (nested) 0/4 with a fabricated error
spoken and `tools=[]`, the same nested tool + a "name your writes" contract nudge
0/4 (nudge inert), `create_note` (flat `{title, content}`) 4/4. **The wall is
argument-schema complexity** - not tool crowding (opens fire in the same 80-tool
set), not the contract wording, not read-vs-write (a flat create fires). The fix
this measured is not a prompt: it is to keep nested-schema tools off the voice
session and route their work through a flat delegate tool to the text path. Design:
`docs/superpowers/specs/2026-08-14-async-delegation-design.md`.

**Shipped-fix verification (2026-08-14, N=3):** cell 5 applies the real
`is_flat_schema` + `surface="voice"` filter (nested tools withheld, `delegate_task`
offered) to the same create request. `delegate_task` fires **3/3** where the nested
create faked 0/3 - the voice model reliably reaches for the flat escape hatch. The
fix works on the live audio path, measured, not inferred.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from evals.m0_voice_spike import RECOMMENDED_MODEL, _load_env

INPUT_RATE = 16_000
CHUNK_BYTES = 960
"""30 ms of 16 kHz 16-bit mono - the block size daemon/voice/audio.py sends."""
LEAD_SILENCE_S = 0.3
TRAIL_SILENCE_S = 1.5
"""The zero-PCM the server VAD needs around the utterance or the turn never triggers
and an engaged session reads as a false refusal (the memory's hard-won catch)."""

TURN_BUDGET_S = 30.0
MAX_TURNS = 3
"""A write often takes two turns - search for the parent, then create under it - so
the spike reads a few turns rather than assuming the create rides the first one."""

TTS_VOICE = "Yuna"

OPEN_REQUEST = "크롬 브라우저 좀 열어줘."
# All-Korean, no English acronym: an earlier version said "UJET JD" and macOS TTS
# rendered it "유적지 D", so the model searched the wrong title and honestly asked
# for the right one - a transcription confound, not the deflection under test. This
# create needs no parent lookup, so a create call is the single expected action and
# "저장했어" with no tool call is unambiguously the confabulation we are hunting.
CREATE_REQUEST = "노션에 '내일 준비물'이라는 새 페이지를 만들어서 저장해 줘."

CREATE_TOOL = "notion__notion-create-pages"
FLAT_CREATE_TOOL = "create_note"
OPEN_TOOL = "open_path"
DELEGATE_TOOL = "delegate_task"


def _tool_specs(crowd: int, *, flat: bool = False) -> list[Any]:
    """A realistically crowded declaration list carrying the control and the subject.

    The two under test are real in shape; the rest are plausible read-only padding
    whose only job is to crowd the list to the size a real install reaches (the
    owner's is 88). Names look like the real MCP servers so nothing about the set
    reads as a toy the model would treat differently.
    """
    from daemon.llm.base import ToolSpec

    specs = [
        ToolSpec(
            name=OPEN_TOOL,
            description=(
                "Open a file, folder, application or website on the owner's computer "
                "(a browser, Finder, a PDF, a URL)."
            ),
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        ),
        ToolSpec(
            name=CREATE_TOOL,
            description=(
                "Create one or more Notion pages under a parent page. Use this to add "
                "a new page or subpage."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "parent": {
                        "type": "object",
                        "properties": {"page_id": {"type": "string"}},
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "properties": {"type": "object"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["pages"],
            },
        ),
        ToolSpec(
            name="notion__notion-search",
            description="Search the owner's Notion workspace for pages by title or text.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]
    # Plausible read-only padding across the real server namespaces.
    families = [
        ("notion__notion", ["fetch", "list-private-pages", "list-users", "get-comments",
                            "list-databases", "query-database", "get-page-property"]),
        ("google", ["search_gmail", "list_events", "list_calendars", "get_message",
                    "list_drive_files", "get_file", "list_labels", "get_contact"]),
        ("git", ["status", "log", "diff", "show", "blame", "branch_list", "grep"]),
        ("tavily", ["tavily_search", "tavily_extract", "tavily_map", "tavily_crawl"]),
    ]
    if flat:
        # The same create intent behind a flat, `open_path`-shaped schema - two string
        # args, no nesting. If this fires where the nested create does not, the schema
        # is the wall, not the verb.
        specs[1] = ToolSpec(
            name=FLAT_CREATE_TOOL,
            description="Create a new note or page with a title and text body.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        )
    i = 0
    while len(specs) < crowd:
        family, leaves = families[i % len(families)]
        leaf = leaves[(i // len(families)) % len(leaves)]
        name = f"{family}-{leaf}-{i}"
        specs.append(
            ToolSpec(
                name=name,
                description=f"Read-only {family} operation ({leaf}).",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        )
        i += 1
    return specs[:crowd]


def _seed() -> str:
    """The real persona seed if it is here, so the system instruction has production
    shape; a short stand-in otherwise. Held identical across both cells either way -
    the variable under test is the tool contract, not the persona."""
    try:
        from daemon.config import Settings

        path = Settings().data_dir / "persona" / "seed.md"
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001 - a stand-in seed is fine, this is not the variable
        pass
    return "너는 대현 님의 개인 비서 '벨라'야. 한국어로, 친근하고 간결하게 말해."


def _patched_contract(baseline: str) -> str:
    """The baseline nudge plus one bullet that names writes the way the existing one
    names opens. Inserted right after the open/run bullet so it reads as its sibling;
    appended if that anchor ever moves, so the spike never silently tests the baseline
    twice."""
    write_bullet = (
        "\n- The same is true of creating things, not only opening them. When the "
        "owner asks you to create, write, save, add, update or put something somewhere "
        "- a note, a file, a Notion page or subpage, a calendar event - call the tool "
        "that does it and actually do it. Never say you created, saved or added "
        "something without having called a tool for it."
    )
    anchor = "never tell them you are unable to open things \\\nhere."
    if anchor in baseline:
        return baseline.replace(anchor, anchor + write_bullet, 1)
    anchor2 = "never tell them you are unable to open things here."
    if anchor2 in baseline:
        return baseline.replace(anchor2, anchor2 + write_bullet, 1)
    return baseline + write_bullet


def _tts_pcm(text: str) -> bytes:
    """Korean TTS as 16 kHz mono 16-bit little-endian PCM, via `say`. The utterance
    the socket hears, so the whole path (VAD, transcription, tool decision) is the
    real one and not a `send_text` shortcut the bug does not survive."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav = Path(handle.name)
    try:
        subprocess.run(
            ["say", "-v", TTS_VOICE, "--file-format=WAVE",
             "--data-format=LEI16@16000", "-o", str(wav), text],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with wave.open(str(wav), "rb") as reader:
            return reader.readframes(reader.getnframes())
    finally:
        wav.unlink(missing_ok=True)


async def _feed(session: Any, pcm: bytes) -> None:
    """Lead silence, the utterance in 30 ms chunks, trail silence - the shape the
    server VAD needs to mark a turn (memory: without the padding the turn never fires
    and the empty result reads as a refusal that never happened)."""
    lead = b"\x00" * (int(INPUT_RATE * LEAD_SILENCE_S) * 2)
    trail = b"\x00" * (int(INPUT_RATE * TRAIL_SILENCE_S) * 2)
    stream = lead + pcm + trail
    for start in range(0, len(stream), CHUNK_BYTES):
        await session.send_audio(stream[start : start + CHUNK_BYTES])


def _answer_for(name: str) -> str:
    """A fabricated but plausible result so the model can proceed past a read to the
    write we are watching for. A search returns the real-looking UJET JD page so a
    create has a parent to aim at; everything else gets a benign ok."""
    if "search" in name or "fetch" in name or "list" in name:
        return (
            '{"results":[{"id":"3bbae2b1-7432-806f-93c9-d84be1d934c3",'
            '"title":"UJET JD","url":"https://app.notion.com/p/'
            '3bbae2b17432806f93c9d84be1d934c3"}]}'
        )
    return '{"ok": true}'


async def _run_once(api_key: str, model: str, system_instruction: str,
                    specs: list[Any], request_pcm: bytes) -> tuple[list[str], list[str]]:
    """One live audio session. Returns (tool names called, assistant transcripts)."""
    from daemon.llm.base import ToolCall
    from daemon.tools.base import ToolResult
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    called: list[str] = []
    said: list[str] = []
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=system_instruction,
        tools=specs,
        # No voice_name: TTS_VOICE is the macOS `say` voice that generates the *input*
        # audio, not a Gemini Live *output* voice (a mismatch the server closes 1007).
        # What it answers in is irrelevant here - only which tools it calls is.
        start_sensitivity="high",
        end_sensitivity="high",
    )
    async with session:
        await _feed(session, request_pcm)
        for _ in range(MAX_TURNS):
            got = False
            try:
                async with asyncio.timeout(TURN_BUDGET_S):
                    async for event in session.receive():
                        got = True
                        if isinstance(event, ToolCall):
                            called.append(event.name)
                            await session.send_tool_response(
                                [ToolResult(call_id=event.id, name=event.name,
                                            content=_answer_for(event.name))]
                            )
                        elif isinstance(event, Transcript) and event.role == "assistant":
                            said.append(event.text)
            except TimeoutError:
                break
            if not got or CREATE_TOOL in called or OPEN_TOOL in called or DELEGATE_TOOL in called:
                break
    return called, said


def _voice_filtered_specs(crowd: int) -> list[Any]:
    """The shipped fix, applied to the crowded set: what `Companion.specs(surface=
    "voice")` actually offers - every flat-schema tool plus `delegate_task`, with the
    nested tools (notion-create-pages and friends) withheld. Uses the REAL
    `is_flat_schema` from the product, so this cell exercises the code that ships, not
    a hand-rolled copy of its rule."""
    from daemon.llm.base import ToolSpec
    from daemon.tools.schema import is_flat_schema

    specs = _tool_specs(crowd)  # includes the nested CREATE_TOOL the fix withholds
    delegate = ToolSpec(
        name=DELEGATE_TOOL,
        description=(
            "Hand a task off to be done in the background and reported back when "
            "finished. Use this for anything you cannot do in one direct step - "
            "creating or editing a Notion page, multi-step work. Pass the owner's "
            "request in plain language; do not try to do it yourself first."
        ),
        parameters={
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        },
    )
    kept = [s for s in specs if s.name == DELEGATE_TOOL or is_flat_schema(s.parameters)]
    return [delegate, *kept]


async def _cell(api_key: str, model: str, label: str, system_instruction: str,
                specs: list[Any], request_pcm: bytes, target: str, runs: int) -> int:
    hits = 0
    print(f"\n=== {label} ===")
    for n in range(1, runs + 1):
        try:
            called, said = await _run_once(api_key, model, system_instruction, specs, request_pcm)
        except Exception as exc:  # noqa: BLE001 - a spike reports rather than raises
            leaked = api_key in str(exc)
            print(f"  run {n}: ERROR {type(exc).__name__}: {exc}  (key leaked: {leaked})")
            continue
        hit = target in called
        hits += hit
        mark = "CALLED" if hit else "no call"
        tail = (said[-1][:70] + "…") if said else "(no transcript)"
        print(f"  run {n}: {target} {mark:8} | tools={called or '[]'} | said={tail!r}")
    print(f"  --> {target}: {hits}/{runs} calls")
    return hits


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="sessions per cell")
    parser.add_argument("--crowd", type=int, default=80, help="total tool declarations")
    args = parser.parse_args()

    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Put it in .env and run again.")
        return 1
    model = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip() or RECOMMENDED_MODEL

    from daemon.companion import TOOL_CONTRACT

    seed = _seed()
    baseline = f"{seed}\n\n{TOOL_CONTRACT}"
    patched = f"{seed}\n\n{_patched_contract(TOOL_CONTRACT)}"
    if patched == baseline:
        print("PATCHED == BASELINE - the write bullet did not attach; fix _patched_contract.")
        return 1
    specs = _tool_specs(args.crowd)
    open_pcm = _tts_pcm(OPEN_REQUEST)
    create_pcm = _tts_pcm(CREATE_REQUEST)

    print(f"key: ...{api_key[-4:]}   model: {model}   tools: {len(specs)}   runs/cell: {args.runs}")
    print(f"control request: {OPEN_REQUEST!r}")
    print(f"subject request: {CREATE_REQUEST!r}")

    ctrl = await _cell(api_key, model, "1. baseline nudge + open  (CONTROL, expect high)",
                       baseline, specs, open_pcm, OPEN_TOOL, args.runs)
    bug = await _cell(api_key, model, "2. baseline nudge + create  (THE BUG, expect ~0)",
                      baseline, specs, create_pcm, CREATE_TOOL, args.runs)
    fix = await _cell(api_key, model, "3. patched nudge + create  (nudge FIX, expect up)",
                      patched, specs, create_pcm, CREATE_TOOL, args.runs)
    flat_specs = _tool_specs(args.crowd, flat=True)
    flat = await _cell(api_key, model, "4. baseline nudge + FLAT-schema create  (schema test)",
                       baseline, flat_specs, create_pcm, FLAT_CREATE_TOOL, args.runs)
    # The shipped fix: voice is offered the surface="voice" set (flat tools + delegate,
    # nested withheld) and asked to create. It cannot create directly, so it should
    # reach for delegate_task - the flat escape hatch it CAN call.
    voice_specs = _voice_filtered_specs(args.crowd)
    delegated = await _cell(
        api_key, model, "5. SHIPPED FIX: voice-filtered set + create  (expect delegate)",
        baseline, voice_specs, create_pcm, DELEGATE_TOOL, args.runs)

    print("\n================ verdict ================")
    print(f"open_path (flat schema),      baseline : {ctrl}/{args.runs}   (control)")
    print(f"notion-create (nested schema), baseline : {bug}/{args.runs}   (the failure)")
    print(f"notion-create (nested schema), patched  : {fix}/{args.runs}   (nudge fix)")
    print(f"create_note (flat schema),     baseline : {flat}/{args.runs}   (schema test)")
    print(f"delegate_task (shipped fix),   baseline : {delegated}/{args.runs}   (the fix)")
    if ctrl <= bug:
        print("=> Control did NOT beat the failure cell - suspect the harness (audio never")
        print("   engaged?), not the hypothesis. Read the transcripts before trusting the rest.")
    elif fix > bug and flat <= bug:
        print("=> The NUDGE is the lever: naming writes lifted creates, flat schema did not.")
        print("   Move the write bullet from _patched_contract into companion.py.")
    elif flat > bug and fix <= bug:
        print("=> The SCHEMA is the wall: a flat create fires where the nested one is faked,")
        print("   and the nudge did nothing. Fix = give voice a flat-schema create tool.")
    elif flat > bug and fix > bug:
        print("=> Both help. Prefer the smaller change; note schema mattered more/less by margin.")
    else:
        print("=> Neither lifted creates. It is deeper than nudge-or-schema; do not ship either.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
