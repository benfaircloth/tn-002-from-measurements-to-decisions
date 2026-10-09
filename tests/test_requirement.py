import numpy as np

from measurement.requirement import Requirement, meets_requirement


def test_clearly_above():
    samples = np.array([0.90, 0.92, 0.88, 0.91, 0.89])
    req = Requirement(minimum_quality=0.75, required_probability=0.95)
    assert meets_requirement(samples, req) is True


def test_clearly_below():
    samples = np.array([0.60, 0.65, 0.58, 0.62, 0.61])
    req = Requirement(minimum_quality=0.75, required_probability=0.95)
    assert meets_requirement(samples, req) is False


def test_probability_threshold_matters():
    samples = np.array([0.80, 0.80, 0.80, 0.70, 0.70])
    req_lenient = Requirement(minimum_quality=0.75, required_probability=0.50)
    req_strict = Requirement(minimum_quality=0.75, required_probability=0.80)
    assert meets_requirement(samples, req_lenient) is True
    assert meets_requirement(samples, req_strict) is False


def test_requirement_is_immutable():
    req = Requirement(minimum_quality=0.75, required_probability=0.95)
    try:
        req.minimum_quality = 0.80
        assert False, "should not allow mutation"
    except AttributeError:
        pass
