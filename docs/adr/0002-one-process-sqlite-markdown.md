# 0002 — One process, SQLite, markdown as source of truth

**Status:** accepted · 2026-08-03

## Context

The first stack was FastAPI + Celery workers + Postgres/pgvector + SSE. The
product is a companion resident on one person's laptop, 24 hours a day.

## Decision

One process: uvicorn plus an in-process scheduler. SQLite, not Postgres. Markdown
files as the source of truth with SQLite as a rebuildable index. WebSocket, not
SSE.

Celery's reason to exist is distributing work across machines, and there is one
user. Postgres' durability advantage does not apply to an index that can be
regenerated. And both references converged here independently: Hermes and OpenClaw
each run a single resident process over SQLite plus markdown, supervised by
LaunchAgent or a systemd user unit.

Order matters and is not negotiable: **the markdown is written first and fsynced,
then the mirror.** sqlite commits with `synchronous=FULL`, so without the fsync the
source of truth was *less* durable than its own index — a power cut left a row
whose record did not exist.

## Consequences

Deleting the database is recoverable (`daemon reindex`); losing the markdown is
not. What a rebuild cannot restore is provenance, since the markdown deliberately
does not carry it — so rebuilt rows are flagged `reindexed = 1`.

A later revision: the original argument included "Docker Desktop costs 2–4 GB
resident." That argument is void — with voice models resident we cost more. The
decision stands on distribution simplicity, file ownership and OS service
integration instead.

## What would change our mind

Multi-user, or a second machine sharing one memory. Neither is on the roadmap;
both would make the whole design different, not just this part.
