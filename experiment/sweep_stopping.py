"""Sweep the confidence requirement and observe what it costs.

Same evidence source, same hypothesis, same threshold.
The only variable is how confident the agent must be before acting.
Reports the tradeoff: decision reliability vs. evidence-gathering cost."""

import numpy as np

from agent.evidence import EvidenceStream
from agent.requirement import Requirement
from agent.stopping import run_until_stop


def main():
    true_mean = 0.80
    noise_std = 0.12
    threshold = 0.75
    n_trials = 200

    confidence_levels = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    print(f"Investigating: is the true state >= {threshold}?")
    print(f"Evidence source: true mean={true_mean}, noise std={noise_std}")
    print(f"Trials per confidence level: {n_trials}")
    print()
    print(f"{'confidence':>12}  {'median steps':>13}  {'min':>5}  {'max':>5}  {'never stopped':>15}")
    print("-" * 60)

    for conf in confidence_levels:
        req = Requirement(threshold=threshold, confidence=conf)
        steps = []

        for trial in range(n_trials):
            stream = EvidenceStream(true_mean, noise_std,
                                    seed=1000 + trial)
            n = run_until_stop(stream, req, max_steps=500)
            steps.append(n)

        arr = np.array(steps)
        never = int(np.sum(arr >= 500))
        finished = arr[arr < 500]

        if len(finished) > 0:
            median = int(np.median(finished))
            print(f"{conf:>12.2f}  {median:>13}  "
                  f"{int(finished.min()):>5}  {int(arr.max()):>5}  "
                  f"{never:>15}")
        else:
            print(f"{conf:>12.2f}  {'---':>13}  "
                  f"{int(arr.min()):>5}  {int(arr.max()):>5}  "
                  f"{never:>15}")


if __name__ == "__main__":
    main()
