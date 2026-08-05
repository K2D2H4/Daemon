-- Daemon storage contract.
--
-- Markdown files under memory/ and persona/ are the SOURCE OF TRUTH.
-- This database holds (a) metadata the model must not be able to forge in prose
-- and (b) regenerable search indexes. Deleting this file must never lose user
-- data: it is rebuilt from the markdown. See docs/PLAN.md 4.2.
--
-- Verified against SQLite 3.45.3 (Python 3.13 stdlib): STRICT, FTS5, json1 all OK.
--
-- Conventions
--   * Timestamps are TEXT, ISO-8601 UTC with 'Z' ('2026-08-03T07:14:00Z').
--   * Booleans are INTEGER 0/1.
--   * Every table that the model can cause writes to carries provenance columns.
--   * Milestone markers below say when a table starts being used; create them
--     all up front so later milestones never need a migration that rewrites
--     earlier rows.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL
) STRICT;

-- ---------------------------------------------------------------------------
-- M1a: conversation log mirror
-- ---------------------------------------------------------------------------
-- The markdown file memory/log/YYYY-MM-DD.md is the original. This table is a
-- mirror that exists so FTS5 recall (M1b) and reflection (M2) can query
-- efficiently. It is rebuildable from the markdown.

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY,
    ts            TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content       TEXT    NOT NULL,

    -- provenance (docs/PLAN.md 4.2) -----------------------------------------
    origin        TEXT    NOT NULL CHECK (origin IN ('owner', 'agent', 'untrusted', 'system')),
    session_kind  TEXT    NOT NULL CHECK (session_kind IN ('interactive', 'voice', 'proactive', 'reflection')),
    modality      TEXT    NOT NULL CHECK (modality IN ('text', 'voice')),

    channel       TEXT    NOT NULL,
    sender_id     TEXT,
    log_file      TEXT    NOT NULL,   -- relative path of the markdown original
    recalled      INTEGER NOT NULL DEFAULT 0,
                  -- 1 = this row was surfaced by recall. Still written, no longer
                  -- read (2026-08-05): it used to exclude the row from reflection
                  -- permanently, which removed 29 of 38 messages on a real day and
                  -- prevented no loop - recall's hits are injected as a system
                  -- block and are never rows. See Store.messages_for_day.

    -- The channel's own id for this message (a Telegram update_id). Telegram
    -- only confirms an update on the *next* getUpdates, so a restart in that
    -- window re-delivers it; without a dedup key the append-only markdown would
    -- gain a duplicate on an ordinary restart, with nothing to reconcile it by.
    external_id   TEXT,

    -- 1 = provenance was inferred by rebuilding from the markdown, not observed
    -- when the message arrived. The markdown deliberately carries no provenance
    -- (it lives in columns so a model cannot forge it), so a rebuild has to
    -- guess, and reflection must be able to tell a guess from a fact.
    reindexed     INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (ts);
CREATE INDEX IF NOT EXISTS idx_messages_kind ON messages (session_kind, ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_external
    ON messages (channel, external_id) WHERE external_id IS NOT NULL;

-- Where each channel's inbound stream got to. Written only after a message has
-- been fully handled: saving on receipt would trade duplicates for silent loss,
-- and losing what the user said is the worse failure.
CREATE TABLE IF NOT EXISTS channel_cursor (
    channel     TEXT    PRIMARY KEY,
    offset_at   INTEGER NOT NULL,
    updated_at  TEXT    NOT NULL
) STRICT;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5 (
    content,
    content = 'messages',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
END;

-- ---------------------------------------------------------------------------
-- Onboarding: channel pairing
-- ---------------------------------------------------------------------------
-- How the owner's channel id gets known without anyone copying a numeric id by
-- hand. An unknown sender's first DM is dropped and answered with a short code;
-- the owner approves that code from the terminal, and the id is captured from
-- the message rather than typed.
--
-- The shape follows OpenClaw's, which has the security details already worked
-- out: 8 uppercase characters with the ambiguous ones (0O1I) left out, roughly an
-- hour of validity, one code per sender per hour, and at most a few pending
-- requests at a time so the code cannot be guessed by volume. Approval is
-- per-sender and grants nothing else.
--
-- Unlike everything else in this file these rows are NOT rebuildable from the
-- markdown: they are the allowlist. Losing them means pairing again, which is
-- recoverable, so the file stays a rebuildable index in spirit.

CREATE TABLE IF NOT EXISTS channel_pairing (
    channel     TEXT    NOT NULL,
    sender_id   TEXT    NOT NULL,
    code        TEXT,                -- NULL once approved; the code is spent
    state       TEXT    NOT NULL CHECK (state IN ('pending', 'approved')),
    created_at  TEXT    NOT NULL,
    expires_at  TEXT,                -- pending only
    approved_at TEXT,
    -- The first approval bootstraps the owner. Recorded so it can only happen
    -- once: later approvals add a guest, they do not hand over ownership.
    is_owner    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (channel, sender_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_pairing_pending ON channel_pairing (channel, state, expires_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pairing_code
    ON channel_pairing (channel, code) WHERE code IS NOT NULL;

-- ---------------------------------------------------------------------------
-- M1b: vector index for recall
-- ---------------------------------------------------------------------------
-- Vectors are stored as raw float32 BLOBs and searched by brute force in numpy,
-- deliberately *not* through a sqlite extension. Measured on this project's
-- target machine: 0.18 ms per query over 10k messages, 1.07 ms over 50k - far
-- inside the voice latency budget at one person's scale. An extension would buy
-- nothing here and cost portability: this very Python build ships with
-- enable_load_extension disabled, so sqlite-vec cannot load at all, and the same
-- failure would hit anyone self-hosting on such a build.
--
-- Regenerable like every other index: drop the rows and re-embed from the
-- markdown. `model` and `dim` are recorded so a model change invalidates only
-- its own rows instead of silently mixing vector spaces.
--
-- Measured afterwards, and worth knowing before optimising the wrong thing: the
-- vector lane is the cheap half. At 10k messages it costs 0.22 ms against the
-- FTS5 lane's 1.9 ms, and at 50k it is 1.29 ms against 9.1 ms - a six-term OR of
-- common tokens is simply more work than one matmul. Both stay far inside the
-- voice budget. What actually dominates Lane 1 is neither: the embedder round
-- trip for the query is ~117 ms, almost all of it fixed overhead rather than
-- inference (see docs/PLAN.md 4.3.1).

CREATE TABLE IF NOT EXISTS embeddings (
    message_id  INTEGER PRIMARY KEY REFERENCES messages (id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,   -- float32, L2-normalised
    created_at  TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings (model);

-- ---------------------------------------------------------------------------
-- M1b/M2: curated memory tier
-- ---------------------------------------------------------------------------
-- Small, always injected at session start, gated writes. Body lives in
-- memory/core.md; this table holds the metadata used for recall scoring.
-- Recall score = hybrid similarity x exp recency decay (30d half-life) x importance.

CREATE TABLE IF NOT EXISTS memory_entries (
    id                INTEGER PRIMARY KEY,
    body              TEXT    NOT NULL,
    importance        INTEGER NOT NULL DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    trigger_phrases   TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(trigger_phrases)),

    origin            TEXT    NOT NULL CHECK (origin IN ('owner', 'agent', 'untrusted', 'system')),
    session_kind      TEXT    NOT NULL CHECK (session_kind IN ('interactive', 'voice', 'proactive', 'reflection')),
    modality          TEXT    NOT NULL DEFAULT 'text' CHECK (modality IN ('text', 'voice')),
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,

    -- New facts with the same key retire the old row instead of piling up a
    -- contradiction ("has a girlfriend" / "does not"). docs/PLAN.md 4.4.
    supersession_key  TEXT,
    status            TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    superseded_by     INTEGER REFERENCES memory_entries (id),

    source_file       TEXT,
    source_anchor     TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_memory_active ON memory_entries (status, updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_supersession
    ON memory_entries (supersession_key) WHERE status = 'active' AND supersession_key IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5 (
    body,
    content = 'memory_entries',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_entries BEGIN
    INSERT INTO memory_fts (rowid, body) VALUES (new.id, new.body);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_entries BEGIN
    INSERT INTO memory_fts (memory_fts, rowid, body) VALUES ('delete', old.id, old.body);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory_entries BEGIN
    INSERT INTO memory_fts (memory_fts, rowid, body) VALUES ('delete', old.id, old.body);
    INSERT INTO memory_fts (rowid, body) VALUES (new.id, new.body);
END;

-- ---------------------------------------------------------------------------
-- M2: entity notes (episodic tier metadata)
-- ---------------------------------------------------------------------------
-- Body lives in memory/entities/{name}.md, written only by the AI, wiki-linked
-- with [[name]] so Obsidian/Logseq can render the graph for free.

CREATE TABLE IF NOT EXISTS entities (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL UNIQUE,
    kind         TEXT,                -- person / place / project / topic / ...
    file         TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS entity_links (
    src_id  INTEGER NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    dst_id  INTEGER NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    PRIMARY KEY (src_id, dst_id)
) STRICT;

-- ---------------------------------------------------------------------------
-- M2: persona observations (append-only) - THE LOG CLOCK
-- ---------------------------------------------------------------------------
-- Lit early on purpose: persona evolution needs ~2 weeks of accumulated real
-- observations before it can be judged, and that wall-clock time cannot be
-- compressed. Extraction is retroactive over messages, so the earlier the log
-- exists the earlier this can be backfilled. docs/PLAN.md 8.1.
-- Never UPDATE or DELETE rows here.

CREATE TABLE IF NOT EXISTS observations (
    id            INTEGER PRIMARY KEY,
    body          TEXT    NOT NULL,   -- "shorter messages in the morning land better"
    created_at    TEXT    NOT NULL,
    observed_from TEXT    NOT NULL,   -- ISO date range of the source messages
    modality      TEXT    NOT NULL DEFAULT 'text' CHECK (modality IN ('text', 'voice')),
    origin        TEXT    NOT NULL DEFAULT 'agent' CHECK (origin IN ('owner', 'agent', 'untrusted', 'system')),
    confidence    REAL    NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    consumed_by   INTEGER REFERENCES persona_rules (id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_observations_unconsumed ON observations (consumed_by, created_at);

-- ---------------------------------------------------------------------------
-- M4: persona rules
-- ---------------------------------------------------------------------------
-- Body is mirrored into persona/learned.md (AI-owned). persona/seed.md is
-- human-owned and never represented here - that asymmetry IS the anchor.
-- docs/PLAN.md 5.1.
-- Rate limits live in config: max active rules, max added per cycle.

CREATE TABLE IF NOT EXISTS persona_rules (
    id           INTEGER PRIMARY KEY,
    body         TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    retired_at   TEXT,
    retired_why  TEXT,
    supersession_key TEXT,
    evidence     TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(evidence))
                 -- observation ids this rule was derived from, for the diff diary
) STRICT;

CREATE INDEX IF NOT EXISTS idx_persona_active ON persona_rules (status, created_at);

-- ---------------------------------------------------------------------------
-- M3: proactivity - THE LABEL CLOCK
-- ---------------------------------------------------------------------------
-- Candidates are generated deterministically with ZERO model calls, then pass a
-- deterministic gate, and only then does one LLM call decide whether to speak.
-- Silence is the default. docs/PLAN.md 6.1.

CREATE TABLE IF NOT EXISTS proactive_candidates (
    id            INTEGER PRIMARY KEY,
    kind          TEXT    NOT NULL CHECK (kind IN (
                      'open_loop',      -- A: unfinished context, due
                      'emotional',      -- B: emotional follow-up
                      'silence',        -- C: quiet too long
                      'pattern_time',   -- D: usual talking hour, nothing today
                      'association'     -- E: old memory strongly linked to recent context
                  )),
    reason        TEXT    NOT NULL,     -- human-readable, goes into the LLM prompt
    payload       TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(payload)),
    created_at    TEXT    NOT NULL,
    due_at        TEXT,
    expires_at    TEXT,

    state         TEXT    NOT NULL DEFAULT 'pending' CHECK (state IN (
                      'pending', 'armed', 'fired', 'done', 'cancelled', 'expired'
                  )),
    fire_count    INTEGER NOT NULL DEFAULT 0,
    fire_budget   INTEGER NOT NULL DEFAULT 1,
    cooldown_secs INTEGER NOT NULL DEFAULT 86400,
    last_fired_at TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_candidates_due ON proactive_candidates (state, due_at);

CREATE TABLE IF NOT EXISTS proactive_utterances (
    id             TEXT    PRIMARY KEY,   -- uuid; echoed back by the label button
    candidate_id   INTEGER REFERENCES proactive_candidates (id),
    kind           TEXT    NOT NULL,
    text           TEXT    NOT NULL,
    spoken_at      TEXT    NOT NULL,
    route          TEXT    NOT NULL CHECK (route IN ('local_speaker', 'telegram', 'both')),

    -- deterministic gate readings at decision time, so a bad call can be
    -- diagnosed later instead of guessed at
    gate_snapshot  TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(gate_snapshot)),

    label          TEXT    CHECK (label IN ('good', 'bad')),
    labeled_at     TEXT,
    responded      INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX IF NOT EXISTS idx_utterances_label ON proactive_utterances (spoken_at, label);
