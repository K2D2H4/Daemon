# 0019 — The seed is authored by the owner, not unreachable by code

**Status:** accepted · 2026-08-26 · refines [0003](0003-file-ownership-is-the-anchor.md) · measured

## Context

[ADR 0003](0003-file-ownership-is-the-anchor.md) made the persona anchor a
filesystem fact rather than a rule the model must obey, and stated it as: **the
anchor is not a rule, it is the fact that the AI cannot reach `seed.md`.**
CONTRACTS non-negotiable 5 encoded that as "code must never write to it", and
`daemon/admin/routes.py` carried a comment saying no route did.

The owner asked for the seed to be editable in the admin console. That is a
direct collision with the sentence above, so it is worth being exact about which
part of 0003 was load-bearing.

Two things were already true before this change:

1. **Code did write the seed.** `daemon/setup.py:_seed_persona` composes it from
   three answers the owner types during `daemon setup`, and refuses to touch an
   existing file. Nobody considered that a violation, which tells us the rule was
   never really about the existence of a `write()`.
2. **The file was not unreachable by the model either.** Measured here rather
   than inferred: `DAEMON_TOOLS_ROOTS` defaults to `~`, and
   `daemon/tools/builtin.py:PathScope` resolves `~/…/data/persona/seed.md`
   cleanly — while refusing `.env` by name from `DENIED_GLOBS`. With tools at
   their default `full` mode, `write_file` on an owner turn already reached the
   seed. So 0003's strongest sentence was, in the shipped default configuration,
   not literally true.

What has *actually* held all along, and what the anchor's value comes from, is
narrower: **nothing the daemon produces has ever been composed into `seed.md`.**
No reflection pass, no weekly persona pass, no model reply. `daemon/persona/rules.py`
writes the other file, and `daemon/persona/evolve.py` reads the seed to write proposals
into `learned.md`.

## Decision

**Non-negotiable 5 is restated as a claim about authorship, not about
reachability.** The seed is the owner's text. Exactly two paths write it, both
carrying only characters a person typed:

```
daemon/setup.py        creates it from the wizard's answers; never overwrites
daemon/admin/seed_io.py  replaces it with the text posted by the admin form
```

There is no third, and nothing that thinks may become one. A path that took the
seed's content from a model — a "help me write my persona" button, an evolution
pass that edits the anchor, a tool the model can call — is still forbidden, and
that is the rule the ADR exists to carry.

[ADR 0018](0018-a-declared-expression-is-not-a-tool-call.md) landed the same day
and the two restatements now sit beside each other in CONTRACTS, so it is worth
saying that they do not stack. 0018 carved out what counts as a *tool call* for
rule 12's audit row: a value the model declares which touches nothing outside
this process. Nothing reachable through that carve-out can write the seed —
writing a file is the definition of touching something outside the process — and
"the model declared it rather than calling a tool" is not a route past rule 5
either, because rule 5 is about where the bytes came from, and a declared value
came from the model.

`PUT /admin/api/persona/seed` is the whole surface. Three refusals in
`daemon/admin/seed_io.py` do the work, and each one closes a trap that was
already in the code the editor had to sit on:

1. **Stale writes are refused (409), keyed on a sha256 of the file's bytes.**
   Hand-editing the seed is the point of it being human-owned, so the browser is
   not the only writer and a page left open is holding text that may be gone.
   The hash is required on the request, not optional: "no hash" meaning "there
   was no file" is the one default that silently overwrites.
2. **An undecodable seed cannot be opened or overwritten.**
   `daemon/persona/loader.py:read_file` swallows a `UnicodeDecodeError` into `""` so a
   conversation survives it — which means a CP949-saved seed reads as *empty*
   everywhere else in the daemon. An editor repeating that would show an empty
   box over a full file. `read_seed` raises instead, and the write path refuses
   the same file for free: a caller who never got a hash cannot produce one.
3. **A blank save is refused.** Not because empty is invalid — a fresh install
   has no seed — but because an empty save is what every failed or truncated read
   looks like, and `daemon doctor` calls an empty seed a proactivity blocker.

The editor also does not read the seed from `GET /admin/api/persona`, though that
payload contains it. That copy shares one 64 KB body budget with the evolution
diaries and legitimately arrives as `text: null` once they fill it
(`daemon/admin/mind.py:_file_view`) — correct for a read-only disclosure, and a
way to delete a personality from an editor. `GET /admin/api/persona/seed` is
unbudgeted and is the only read a save may be based on.

One backup slot (`data/persona/seed.md.bak`) is written before each replace. The file
had no writer at all before this and therefore no undo either.

## Consequences

The loopback admin gains no new class of privilege: it can already write `.env`,
including API keys, and restart the process. What it gains is the one edit the
owner previously had to leave the console to make — and `load_persona` re-reads
the file every turn, so the edit lands on the next message with no restart.

0003's write-conflict answer is weakened slightly and deliberately: the file a
person edits in Obsidian now has a second human writer in the browser. That is
what the sha256 check is for, and a conflict is reported to the owner with their
own text still on screen rather than resolved by guessing.

The measurement above leaves a real gap that this ADR does not close: on an owner
turn with tools at `full`, `write_file` can still write the seed, because
`seed.md` is in neither `DENIED_NAMES` nor `DENIED_GLOBS`. Naming it there would
be the change that makes 0003's original sentence true for the first time. It is
recorded here rather than done, because it narrows a tool the owner deliberately
runs wide open ([CONTRACTS.md](../CONTRACTS.md) non-negotiable 12) and is their
call, not this task's.

## What would change our mind

If a seed written through the browser turns out to drift the way `learned.md`
would — the owner accepting suggestions rather than writing their own text —
then the problem is not the write path but what sits in front of it, and the
answer is to keep the write path and refuse to put a model behind the button.

Nothing would justify a path that composes the seed from model output. That is
the anchor.
