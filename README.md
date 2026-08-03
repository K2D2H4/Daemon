# Daemon

> An open-source, self-hosted AI companion that you shape — and that reshapes
> itself around you.

Daemon is a process that lives on your machine. You give it a starting
personality; it builds its own understanding of you as you talk, lets that
understanding change how it behaves, and decides for itself when to speak first.

Two things here are not available anywhere else, open or closed:

**Its personality evolves.** You write the seed — name, temperament, how it
talks. From then on it watches how you actually respond and accumulates rules
about dealing with *you specifically*: shorter messages in the morning land
better, this person wants to be heard rather than advised. Those rules feed back
into who it is. Every other companion has a personality file a human edits.

**It speaks first, on its own judgement.** Not a reminder you scheduled, and not
a canned line when you open the app. A background loop asks whether there is a
reason to say something right now, whether now is a bad moment, and whether it
has already talked too much today — and stays silent unless all three answers
line up. Silence is the default.

Everything else — the memory graph, the reflection pass, recall scoring — is
table stakes we borrow rather than claim. Design notes, including what we took
from whom and why: [docs/PLAN.md](docs/PLAN.md) (Korean).

---

## Status

Early. Public because the design decisions are worth arguing with, not because
it is finished.

| | |
|---|---|
| **M1a** ✅ | Text conversation over Telegram, logged to markdown |
| **M1b** 🔨 | Recall, voice, OS residency, recall evaluation |
| M2 | Reflection, entity notes, observation capture |
| M3 | Proactivity — it speaks first |
| M4 | Persona evolution |

## How it is put together

- **One process.** `uvicorn` plus an in-process scheduler. No Celery, no Redis,
  no Postgres — there is exactly one user, so a distributed queue would be cost
  without a reason. It installs as a LaunchAgent or a systemd user unit, because
  something that speaks first has to outlive your terminal.
- **Markdown is the source of truth.** Your conversations live in
  `memory/log/YYYY-MM-DD.md`, entity notes in `memory/entities/`, the personality
  in `persona/`. Plain files, wiki-linked, openable in Obsidian or Logseq or
  `cat`. SQLite holds metadata and search indexes and is rebuildable from the
  markdown — delete it and run `daemon reindex`.
- **Provenance lives in columns, not prose.** Where a memory came from, how
  important it is, when it was recorded: none of it is written in text a model
  could forge or mangle.
- **File ownership is split, deliberately.** `persona/seed.md` is yours and the
  code never writes to it. `persona/learned.md` is the AI's and you only read it
  or ask for a rule to be dropped. That asymmetry is what stops an evolving
  personality from drifting into whatever agrees with you most.
- **Vector search is brute-force numpy over float32 blobs**, not a SQLite
  extension: 0.18 ms per query over 10k messages, and it works on Python builds
  that cannot load extensions at all — which is many of them.

## Your data, and where it goes

Plainly, because the honest version is more useful than a slogan:

- Memory and personality are **files on your machine**. Always. Nothing is
  synced anywhere.
- **Text mode with a local model is fully offline.** The `offline` preset routes
  every task to Ollama and has no voice route at all, which is what makes that
  sentence true rather than aspirational.
- **Voice sends audio to whichever provider you chose**, with your own API key.
  Turn voice off and it does not. We use hosted native-audio models on purpose:
  a local speech-to-text → model → text-to-speech cascade throws away the tone
  and timing that make a voice sound human, and the open-weight native-audio
  models cannot generate audio on Apple Silicon today.
- **Telegram means your messages cross Telegram's servers.** It is the first
  channel because it is the only one that actually reaches your phone for free.
- When Daemon speaks first while you are at your desk, it comes out of your
  **local speaker** — that path touches no network at all.

Hosted models are bring-your-own-key throughout, with a hard spend limit.

## Try it

Requires Python 3.13.

```bash
git clone https://github.com/K2D2H4/Daemon.git && cd Daemon
pip install -e ".[dev]"
cp .env.example .env
```

Fill in three things in `.env`:

- `TELEGRAM_BOT_TOKEN` — from [@BotFather](https://t.me/BotFather), `/newbot`
- `TELEGRAM_ALLOWED_USER_IDS` — your **numeric** id, from
  [@userinfobot](https://t.me/userinfobot). Anyone not on this list is ignored,
  and an empty list refuses to start rather than defaulting to open.
- One model key — `ANTHROPIC_API_KEY`, or set `DAEMON_PRESET=offline` and run
  [Ollama](https://ollama.com) instead.

Then:

```bash
python3 -c "from daemon.app import main; main()"
```

Message your bot. The exchange lands in `data/memory/log/`.

## Contributing

Read [docs/CONTRACTS.md](docs/CONTRACTS.md) first — it is short, and it is
binding. The nine non-negotiables in it are not style preferences; each one
exists because breaking it loses user data, leaks a secret, or launders
untrusted text into the personality.

Tests never touch the network and never need an API key. `python3 -m pytest`
and `python3 -m ruff check .` both pass, or the change is not done.

## License

MIT.
