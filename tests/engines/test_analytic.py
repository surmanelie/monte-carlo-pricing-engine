from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mcengine.engines.analytic import bs_digital_price, bs_price, price_analytic
from mcengine.models.gbm import GBM
from mcengine.products.base import Product
from mcengine.products.european import DigitalOption, EuropeanOption

spot = st.floats(min_value=20.0, max_value=300.0)
strike = st.floats(min_value=20.0, max_value=300.0)
maturity = st.floats(min_value=0.05, max_value=5.0)
rate = st.floats(min_value=-0.02, max_value=0.15)
vol = st.floats(min_value=0.05, max_value=1.0)
div = st.floats(min_value=0.0, max_value=0.08)


def test_textbook_values() -> None:
    """Reference values S0=K=100, r=5 %, sigma=20 %, T=1 (Hull 2018)."""
    assert bs_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, abs=5e-5)
    assert bs_price(100, 100, 1.0, 0.05, 0.2, option_type="put") == pytest.approx(5.5735, abs=5e-5)


@settings(max_examples=200, deadline=None)
@given(s=spot, k=strike, t=maturity, r=rate, v=vol, q=div)
def test_put_call_parity(s: float, k: float, t: float, r: float, v: float, q: float) -> None:
    call = bs_price(s, k, t, r, v, q, "call")
    put = bs_price(s, k, t, r, v, q, "put")
    assert call - put == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-9 * s)


@settings(max_examples=200, deadline=None)
@given(s=spot, k=strike, t=maturity, r=st.floats(0.0, 0.15), v=vol)
def test_call_price_bounds(s: float, k: float, t: float, r: float, v: float) -> None:
    """max(S0 - K e^{-rT}, 0) <= C <= S0 (no dividends)."""
    call = float(bs_price(s, k, t, r, v))
    assert max(s - k * math.exp(-r * t), 0.0) - 1e-9 * s <= call <= s * (1 + 1e-12)


@settings(max_examples=200, deadline=None)
@given(s=spot, k=strike, t=maturity, r=rate, v=vol, bump=st.floats(1e-3, 0.5))
def test_call_monotone_in_spot_and_vol(
    s: float, k: float, t: float, r: float, v: float, bump: float
) -> None:
    base = float(bs_price(s, k, t, r, v))
    assume(base > 1e-8 * s)  # avoid comparisons at the underflow floor
    tol = 1e-12 * s  # floating-point noise when the time value is negligible
    assert float(bs_price(s * (1 + bump), k, t, r, v)) >= base - tol
    assert float(bs_price(s, k, t, r, v * (1 + bump))) >= base - tol


@settings(max_examples=100, deadline=None)
@given(s=spot, k=strike, t=maturity, r=rate, v=vol, q=div)
def test_digital_parity(s: float, k: float, t: float, r: float, v: float, q: float) -> None:
    """Digital call + digital put = discount factor."""
    total = bs_digital_price(s, k, t, r, v, q, "call") + bs_digital_price(s, k, t, r, v, q, "put")
    assert total == pytest.approx(math.exp(-r * t), rel=1e-12)


def test_digital_is_minus_strike_derivative_of_call() -> None:
    h = 1e-4
    dcall = (bs_price(100, 100 + h, 1, 0.05, 0.2) - bs_price(100, 100 - h, 1, 0.05, 0.2)) / (2 * h)
    assert bs_digital_price(100, 100, 1, 0.05, 0.2) == pytest.approx(-dcall, rel=1e-7)


def test_vectorised_and_scalar_outputs() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(100.0, strikes, 1.0, 0.05, 0.2)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)
    assert isinstance(bs_price(100.0, 100.0, 1.0, 0.05, 0.2), float)


@pytest.mark.parametrize("bad", [{"s0": -1.0}, {"strike": 0.0}, {"maturity": 0.0}, {"sigma": 0.0}])
def test_invalid_inputs(bad: dict[str, float]) -> None:
    args = {"s0": 100.0, "strike": 100.0, "maturity": 1.0, "r": 0.05, "sigma": 0.2} | bad
    with pytest.raises(ValueError, match="must be finite and > 0"):
        bs_price(**args)
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 1, 0.05, 0.2, option_type="chooser")


def test_price_analytic_dispatch(bs_model: GBM) -> None:
    res = price_analytic(bs_model, EuropeanOption(100.0, 1.0))
    assert res.method == "bs-analytic"
    assert res.std_error is None
    assert res.price == pytest.approx(10.450583572185565, rel=1e-12)
    dig = price_analytic(bs_model, DigitalOption(100.0, 1.0, payout=3.0))
    assert dig.price == pytest.approx(3.0 * float(bs_digital_price(100, 100, 1, 0.05, 0.2)))

    with pytest.raises(ValueError, match="no closed form"):
        price_analytic(bs_model, _Unsupported())


class _Unsupported(Product):
    maturity = 1.0
    is_path_dependent = False
    name = "unsupported"

    def monitoring_times(self) -> np.ndarray:
        return np.array([1.0])

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return paths[:, -1]
