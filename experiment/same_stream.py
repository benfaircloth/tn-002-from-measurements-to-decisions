"""Two requirements evaluate the same evidence stream.

One agent acts early. The other keeps gathering.
The evidence did not change. The requirement did."""

from agent.evidence import EvidenceStream
from agent.requirement import Requirement
from agent.stopping import compute_confidence


def main():
    stream = EvidenceStream(true_mean=0.80, true_std=0.12, seed=42)

    threshold = 0.75

    decisive = Requirement(threshold=threshold, confidence=0.80)
    cautious = Requirement(threshold=threshold, confidence=0.95)

    decisive_stopped = None
    cautious_stopped = None

    print(f"Investigating: is the true state >= {threshold}?")
    print(f"Evidence source: true mean=0.80, true std=0.12")
    print()
    print(f"{'step':>5}  {'mean':>7}  {'se':>7}  "
          f"{'P(H|E)':>8}  {'event'}")
    print("-" * 52)

    for step in range(1, 201):
        stream.observe()

        if stream.n < 3:
            continue

        p = compute_confidence(stream, threshold)

        event = ""
        if decisive_stopped is None and p >= decisive.confidence:
            decisive_stopped = step
            event = "ACT (confidence >= 0.80)"
        if cautious_stopped is None and p >= cautious.confidence:
            cautious_stopped = step
            event = "ACT (confidence >= 0.95)"

        if event or step <= 10 or step % 20 == 0:
            print(f"{step:>5}  {stream.running_mean:>7.4f}  "
                  f"{stream.standard_error:>7.4f}  "
                  f"{p:>8.3f}  {event}")

        if decisive_stopped and cautious_stopped:
            break

    print()
    print(f"Decisive agent (confidence >= 0.80) stopped at step {decisive_stopped}")
    print(f"Cautious agent (confidence >= 0.95) stopped at step {cautious_stopped or 'never (200 steps)'}")
    print()
    print("Same evidence. Same hypothesis. Different requirement.")
    print("The requirement decided when the agent could act.")


if __name__ == "__main__":
    main()
