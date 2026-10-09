"""Stopping logic: should the agent act or keep gathering evidence?"""

import numpy as np
from scipy import stats

from agent.evidence import EvidenceStream
from agent.requirement import Requirement


def compute_confidence(stream: EvidenceStream,
                       threshold: float) -> float | None:
    """Estimate P(true state >= threshold | accumulated evidence).

    Uses a t-distribution to account for uncertainty in both
    the mean and variance estimates from finite observations.
    """
    if stream.n < 3:
        return None

    mean = stream.running_mean
    se = stream.standard_error
    t_stat = (mean - threshold) / se
    return float(1 - stats.t.cdf(-t_stat, df=stream.n - 1))


def should_stop(stream: EvidenceStream,
                requirement: Requirement) -> bool:
    """Returns True when the accumulated evidence meets the requirement."""
    p = compute_confidence(stream, requirement.threshold)
    if p is None:
        return False
    return p >= requirement.confidence


def run_until_stop(stream: EvidenceStream,
                   requirement: Requirement,
                   max_steps: int = 500) -> int:
    """Observe step by step until the evidence meets the requirement.

    Returns the number of observations needed to act,
    or max_steps if the requirement was never met.
    """
    for _ in range(max_steps):
        stream.observe()
        if stream.n >= 3 and should_stop(stream, requirement):
            return stream.n
    return max_steps
