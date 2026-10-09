"""An evidence stream that an agent can sample from step by step."""

import numpy as np


class EvidenceStream:
    """Produces observations drawn from a true underlying distribution.

    The agent does not know the true parameters. It only sees
    the observations it has collected so far.
    """

    def __init__(self, true_mean: float, true_std: float,
                 seed: int = 42):
        self.true_mean = true_mean
        self.true_std = true_std
        self._rng = np.random.default_rng(seed)
        self.observations: list[float] = []

    def observe(self) -> float:
        value = self._rng.normal(self.true_mean, self.true_std)
        self.observations.append(value)
        return value

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def running_mean(self) -> float | None:
        if not self.observations:
            return None
        return float(np.mean(self.observations))

    @property
    def running_std(self) -> float | None:
        if len(self.observations) < 2:
            return None
        return float(np.std(self.observations, ddof=1))

    @property
    def standard_error(self) -> float | None:
        if self.running_std is None:
            return None
        return self.running_std / np.sqrt(self.n)
