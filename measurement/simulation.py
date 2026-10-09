import numpy as np


def simulate_scores(mean: float, std: float, n: int = 100_000,
                    seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


def probability_above(samples: np.ndarray, threshold: float) -> float:
    return float(np.mean(samples >= threshold))


if __name__ == "__main__":
    model_a = simulate_scores(0.84, 0.08)
    model_b = simulate_scores(0.81, 0.03)

    threshold = 0.75

    p_a = probability_above(model_a, threshold)
    p_b = probability_above(model_b, threshold)

    print(f"Model A: {p_a:.3f}")
    print(f"Model B: {p_b:.3f}")
