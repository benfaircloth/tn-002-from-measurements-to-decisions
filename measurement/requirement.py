from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Requirement:
    minimum_quality: float
    required_probability: float


def meets_requirement(samples: np.ndarray, requirement: Requirement) -> bool:
    observed_probability = float(
        np.mean(samples >= requirement.minimum_quality)
    )
    return observed_probability >= requirement.required_probability
