import numpy as np

from measurement.simulation import simulate_scores, probability_above


def test_simulate_returns_correct_count():
    samples = simulate_scores(0.80, 0.05, n=1000)
    assert len(samples) == 1000


def test_simulate_is_deterministic():
    a = simulate_scores(0.80, 0.05, seed=99)
    b = simulate_scores(0.80, 0.05, seed=99)
    np.testing.assert_array_equal(a, b)


def test_probability_above_all():
    samples = np.array([0.90, 0.91, 0.92])
    assert probability_above(samples, 0.80) == 1.0


def test_probability_above_none():
    samples = np.array([0.50, 0.55, 0.60])
    assert probability_above(samples, 0.80) == 0.0


def test_probability_above_partial():
    samples = np.array([0.70, 0.80, 0.90])
    p = probability_above(samples, 0.75)
    assert abs(p - 2 / 3) < 1e-9
