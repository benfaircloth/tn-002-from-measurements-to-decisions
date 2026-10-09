# TN-002: From Measurements to Decisions

Companion code for [Teaching Note 002](https://benfaircloth.com/teaching/tn-002-from-measurements-to-decisions/).

Monte Carlo simulation for reasoning about model selection under uncertainty, requirements, and joint constraints.

## Structure

```
measurement/
  simulation.py     Score sampling and threshold probability
  requirement.py    Requirement dataclass and acceptance rule
  joint.py          Joint probability across multiple dimensions

experiment/
  sweep_threshold.py   Sweep quality threshold from 0.70 to 0.90
  joint_decision.py    Quality + latency joint requirement

tests/
  test_requirement.py  Acceptance rule boundary tests
  test_simulation.py   Sampling and probability tests
```

## Run the tests

```
pytest
```

## Run the experiments

```
python -m experiment.sweep_threshold
python -m experiment.joint_decision
```

The threshold sweep shows where the preferred model reverses as the requirement changes. The joint decision adds latency as a second uncertain dimension and evaluates both constraints simultaneously.

## The point

The value is not the simulation. The value is keeping measurement, uncertainty, requirement, and decision separate so each can be inspected independently.

A score tells us what happened. A requirement tells us what is enough. A decision asks whether the evidence meets the requirement. Simulation is useful when several uncertain conditions interact, but it does not remove the judgment behind the requirement.

## License

MIT

## Author

Ben Faircloth
