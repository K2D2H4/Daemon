"""Task taxonomy for LLM routing.

Every LLM call in Daemon belongs to exactly one Task. The gateway routes per
Task, not per global setting - see docs/PLAN.md 3.2.

Do not add a task without also adding it to `Settings.routing` in config.
"""

from enum import StrEnum


class Task(StrEnum):
    """Routing key for every LLM call."""

    CHAT_TEXT = "chat_text"
    """Text conversation turn. User picks local or hosted."""

    CHAT_VOICE = "chat_voice"
    """Voice conversation turn. Hosted native-audio only - the audio model IS
    the brain, so local is not an option here."""

    RECALL_ESCALATION = "recall_escalation"
    """Lane 2 recall. Rare: only when recall intent is detected and Lane 1 was
    weak. Lane 1 itself makes ZERO model calls and therefore has no Task."""

    PROACTIVE_JUDGE = "proactive_judge"
    """'Is it worth speaking now?' - one call, only for candidates that already
    passed the deterministic gate. Runs on a 5-minute tick, so it accumulates
    cost; defaults to local."""

    REFLECTION = "reflection"
    """Daily consolidation: entity extraction, memory merge, observation
    extraction. Quality here propagates to the whole graph."""

    PERSONA_RULE = "persona_rule"
    """Weekly: turn accumulated observations into persona rules."""

    EMBED = "embed"
    """Text -> vector, for recall. Runs on every message and on every recall
    query, so it sits on the voice latency path and stays local regardless of provider:
    a network round trip per turn would cost more than all of Lane 1."""
