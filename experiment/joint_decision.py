"""Exercise part 2: add latency as a second uncertain quantity
and evaluate a joint requirement."""

import numpy as np

from measurement.simulation import simulate_scores
from measurement.joint import joint_probability


def main():
    rng = np.random.default_rng(42)
    n = 100_000

    quality_a = simulate_scores(0.84, 0.08, n)
    quality_b = simulate_scores(0.81, 0.03, n)

    latency_a = rng.normal(1.2, 0.4, n)
    latency_b = rng.normal(0.8, 0.15, n)

    quality_threshold = 0.75
    latency_threshold = 1.5

    p_a = joint_probability(quality_a, latency_a,
                            quality_threshold, latency_threshold)
    p_b = joint_probability(quality_b, latency_b,
                            quality_threshold, latency_threshold)

    print("Joint requirement: quality >= 0.75 AND latency <= 1.5s")
    print(f"Model A: {p_a:.3f}")
    print(f"Model B: {p_b:.3f}")
    print()

    quality_threshold = 0.85
    latency_threshold = 1.0

    p_a = joint_probability(quality_a, latency_a,
                            quality_threshold, latency_threshold)
    p_b = joint_probability(quality_b, latency_b,
                            quality_threshold, latency_threshold)

    print("Joint requirement: quality >= 0.85 AND latency <= 1.0s")
    print(f"Model A: {p_a:.3f}")
    print(f"Model B: {p_b:.3f}")


if __name__ == "__main__":
    main()
