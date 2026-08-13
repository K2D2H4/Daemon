# Decisions

One file per decision that would be expensive to rediscover. Not a log of
everything — [PLAN.md](../PLAN.md) carries the design and the git history carries
the rest. These are the ones where the *reasoning* is the valuable part, and where
someone could reasonably undo them by accident.

A recurring shape here is worth naming up front: **several of these record a
measurement overturning something we had inferred from documentation or SDK
source.** Where a doc and a socket disagreed, the socket won. Any ADR below that
says "measured" was checked against the real thing, and the number is in the file.

| | decision | status |
|---|---|---|
| [0001](0001-borrow-the-memory-design.md) | The memory layer is borrowed, not invented | accepted |
| [0002](0002-one-process-sqlite-markdown.md) | One process, SQLite, markdown as source of truth | accepted |
| [0003](0003-file-ownership-is-the-anchor.md) | File ownership split is the persona anchor | accepted |
| [0004](0004-hosted-native-audio-for-voice.md) | Voice is hosted native audio | accepted, argument revised |
| [0005](0005-vectors-belong-in-m1b.md) | The vector index moved into M1b | accepted |
| [0006](0006-reachability-and-acceptance-gates.md) | Reachability and acceptance are required gates | accepted |
| [0007](0007-no-default-hosted-provider.md) | The hosted provider has no default | accepted |
| [0008](0008-three-stages-one-model-call.md) | Proactivity is three stages, one model call | accepted |
| [0009](0009-images-in-the-message-contract.md) | Images in the Message contract | accepted |
| [0010](0010-supersession-needs-an-id-not-a-name.md) | Supersession keys off an id, not a name | accepted, measured |
| [0011](0011-the-file-holds-more-than-it-injects.md) | core.md holds every active fact, not the injected ones | accepted, measured |
| [0012](0012-split-the-presence-signals.md) | Split the presence signals | accepted, measured |

## Format

Context · Decision · Consequences, and a line saying what would change our mind.
Short. A decision record nobody reads is the same as no decision record.
