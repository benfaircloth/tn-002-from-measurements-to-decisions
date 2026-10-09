"""Exercise part 1: sweep the quality threshold from 0.70 to 0.90
and observe where the preferred model changes."""

import numpy as np

from measurement.simulation import simulate_scores, probability_above


def main():
    model_a = simulate_scores(0.84, 0.08)
    model_b = simulate_scores(0.81, 0.03)

    thresholds = np.arange(0.70, 0.91, 0.01)

    print(f"{'threshold':>10}  {'P(A >= τ)':>10}  {'P(B >= τ)':>10}  {'preferred':>10}")
    print("-" * 48)

    for tau in thresholds:
        p_a = probability_above(model_a, tau)
        p_b = probability_above(model_b, tau)

        preferred = "A" if p_a > p_b else "B" if p_b > p_a else "tie"
        print(f"{tau:>10.2f}  {p_a:>10.3f}  {p_b:>10.3f}  {preferred:>10}")


if __name__ == "__main__":
    main()
