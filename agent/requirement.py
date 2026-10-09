"""A requirement defines what the agent needs before it can act."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    """The agent should act when P(H | evidence) >= confidence.

    threshold:  the condition under investigation (true state >= threshold)
    confidence: how confident the agent must be before acting
    """
    threshold: float
    confidence: float
