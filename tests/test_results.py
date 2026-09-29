from __future__ import annotations

import math

import pytest

from mcengine.results import GreeksResult, PricingResult


def _mc(price: float = 10.0, se: float = 0.1) -> PricingResult:
    return PricingResult(price, se, price - 0.196, price + 0.196, 1000, 1, "mc-plain", 0.01)


def test_error_in_se_and_contains() -> None:
    res = _mc().with_reference(10.2)
    assert res.error() == pytest.approx(-0.2)
    assert res.error_in_se() == pytest.approx(-2.0)
    assert res.contains(level=0.99)
    assert not res.contains(level=0.95)
    assert res.is_stochastic
    assert "mc-plain" in str(res)


def test_zero_standard_error() -> None:
    res = _mc(se=0.0)
    assert res.error_in_se(10.0) == 0.0
    assert res.error_in_se(9.0) == math.inf


def test_deterministic_result() -> None:
    res = PricingResult(10.45, None, None, None, None, None, "bs-analytic", 0.0)
    assert not res.is_stochastic
    assert str(res) == "bs-analytic: 10.450000"
    with pytest.raises(ValueError, match="standard error"):
        res.error_in_se(10.0)
    with pytest.raises(ValueError, match="confidence"):
        res.confidence_interval()
    with pytest.raises(ValueError, match="reference"):
        res.error()


def test_frozen() -> None:
    res = _mc()
    with pytest.raises(AttributeError):
        res.price = 1.0  # type: ignore[misc]


def test_greeks_as_dict() -> None:
    g = GreeksResult(0.5, 0.02, None, None, None, "x", 0.0)
    assert g.as_dict() == {"delta": 0.5, "gamma": 0.02}
