from agent.evidence import EvidenceStream
from agent.requirement import Requirement
from agent.stopping import should_stop, run_until_stop


def test_no_evidence():
    stream = EvidenceStream(0.90, 0.01)
    req = Requirement(threshold=0.50, confidence=0.90)
    assert should_stop(stream, req) is False


def test_strong_evidence_stops():
    stream = EvidenceStream(0.90, 0.01, seed=1)
    req = Requirement(threshold=0.70, confidence=0.90)
    for _ in range(50):
        stream.observe()
    assert should_stop(stream, req) is True


def test_weak_evidence_does_not_stop():
    stream = EvidenceStream(0.76, 0.10, seed=1)
    req = Requirement(threshold=0.85, confidence=0.95)
    for _ in range(20):
        stream.observe()
    assert should_stop(stream, req) is False


def test_decisive_stops_before_cautious():
    decisive = Requirement(threshold=0.75, confidence=0.80)
    cautious = Requirement(threshold=0.75, confidence=0.95)

    stream_d = EvidenceStream(0.82, 0.06, seed=42)
    stream_c = EvidenceStream(0.82, 0.06, seed=42)

    n_decisive = run_until_stop(stream_d, decisive)
    n_cautious = run_until_stop(stream_c, cautious)

    assert n_decisive < n_cautious


def test_requirement_is_immutable():
    req = Requirement(threshold=0.75, confidence=0.95)
    try:
        req.threshold = 0.80
        assert False, "should not allow mutation"
    except AttributeError:
        pass
