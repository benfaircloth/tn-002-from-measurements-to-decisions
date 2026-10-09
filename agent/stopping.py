"""Stopping logic: should the agent act or keep gathering evidence?"""

import math

from agent.evidence import EvidenceStream
from agent.requirement import Requirement


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_confidence(stream: EvidenceStream, threshold: float,
                       prior_mean: float = 0.5,
                       prior_var: float = 1.0) -> float:
    """P(true state >= threshold | accumulated evidence).

    Conjugate normal update. The agent starts with a prior belief
    about the true state and revises it as observations arrive.
    """
    if stream.n == 0:
        posterior_mean = prior_mean
        posterior_var = prior_var
    else:
        noise_var = stream.noise_std ** 2
        prior_precision = 1.0 / prior_var
        obs_precision = 1.0 / noise_var
        posterior_precision = prior_precision + stream.n * obs_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (
            prior_precision * prior_mean
            + obs_precision * sum(stream.observations)
        )

    posterior_std = math.sqrt(posterior_var)
    z = (threshold - posterior_mean) / posterior_std
    return 1.0 - _normal_cdf(z)


def should_stop(stream: EvidenceStream,
                requirement: Requirement) -> bool:
    """Returns True when the accumulated evidence meets the requirement."""
    if stream.n == 0:
        return False
    return compute_confidence(stream, requirement.threshold) >= requirement.confidence


def run_until_stop(stream: EvidenceStream,
                   requirement: Requirement,
                   max_steps: int = 500) -> int:
    """Observe step by step until the evidence meets the requirement.

    Returns the number of observations needed to act,
    or max_steps if the requirement was never met.
    """
    for _ in range(max_steps):
        stream.observe()
        if should_stop(stream, requirement):
            return stream.n
    return max_steps
