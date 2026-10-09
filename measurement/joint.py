import numpy as np


def joint_probability(quality: np.ndarray, latency: np.ndarray,
                      quality_threshold: float,
                      latency_threshold: float) -> float:
    meets_requirements = (
        (quality >= quality_threshold)
        & (latency <= latency_threshold)
    )
    return float(meets_requirements.mean())
