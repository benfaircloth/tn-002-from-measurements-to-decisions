import numpy as np

from agent.evidence import EvidenceStream


def test_observe_accumulates():
    stream = EvidenceStream(0.80, 0.05)
    stream.observe()
    stream.observe()
    assert stream.n == 2


def test_deterministic_with_seed():
    a = EvidenceStream(0.80, 0.05, seed=7)
    b = EvidenceStream(0.80, 0.05, seed=7)
    assert a.observe() == b.observe()
    assert a.observe() == b.observe()


def test_running_mean_converges():
    stream = EvidenceStream(0.80, 0.01, seed=1)
    for _ in range(10_000):
        stream.observe()
    assert abs(stream.running_mean - 0.80) < 0.01


def test_standard_error_shrinks():
    stream = EvidenceStream(0.80, 0.05, seed=1)
    for _ in range(10):
        stream.observe()
    se_10 = stream.standard_error
    for _ in range(90):
        stream.observe()
    se_100 = stream.standard_error
    assert se_100 < se_10
