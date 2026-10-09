# TN-002: From Measurements to Decisions

Companion code for [Teaching Note 002](https://benfaircloth.com/teaching/tn-002-from-measurements-to-decisions/).

An agent accumulates evidence for a proposition and decides when it has enough to act. There is no stopping point in the evidence alone. Stopping happens when the evidence meets a requirement.

## Structure

```
agent/
  evidence.py       Evidence stream with running statistics
  requirement.py    Requirement dataclass (threshold, confidence)
  stopping.py       Stopping logic: P(H | evidence) >= confidence

experiment/
  same_stream.py    Two confidence requirements on the same evidence
  sweep_stopping.py Sweep confidence and observe the cost

tests/
  test_evidence.py  Evidence accumulation and convergence
  test_stopping.py  Stopping boundary and ordering tests
```

## Run the tests

```
pytest
```

## Run the experiments

```
python -m experiment.same_stream
python -m experiment.sweep_stopping
```

`same_stream` shows two agents investigating the same hypothesis from identical evidence. The decisive agent (confidence >= 0.80) acts early. The cautious agent (confidence >= 0.95) keeps gathering. The evidence did not change. The requirement decided when to act.

`sweep_stopping` varies the confidence requirement across many trials and reports how many observations the agent needed before acting. As the requirement increases, the agent needs substantially more evidence. The tradeoff between decision reliability and evidence-gathering cost becomes visible.

## The point

An agent that can observe its environment still needs something external to decide when its observations are sufficient. The evidence tells the agent what it has seen so far. The requirement tells it when that is enough. Those are different questions, and collapsing them produces an agent that either acts too early or never acts at all.

## License

MIT

## Author

Ben Faircloth
