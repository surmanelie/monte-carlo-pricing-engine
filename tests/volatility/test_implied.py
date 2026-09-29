"""Implied volatility: round trips, bounds, vectorisation, Brent fallback."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcengine.engines.analytic import bs_price
from mcengine.volatility.implied import implied_vol, price_bounds


@settings(max_examples=300, deadline=None)
@given(
    k=st.floats(60.0, 160.0),
    t=st.floats(0.05, 5.0),
    r=st.floats(-0.01, 0.1),
    q=st.floats(0.0, 0.05),
    vol=st.floats(0.05, 1.5),
    kind=st.sampled_from(["call", "put"]),
)
def test_round_trip(k: float, t: float, r: float, q: float, vol: float, kind: str) -> None:
    price = float(bs_price(100.0, k, t, r, vol, q, kind))
    low, _ = price_bounds(100.0, k, t, r, q, kind)
    vega_like = (
        100.0 * math.sqrt(t) * math.exp(-0.5 * (math.log(100 / k) / (vol * math.sqrt(t))) ** 2)
    )
    if price - float(low) < 1e-9 or vega_like < 1e-4:
        return  # time value numerically indistinguishable from zero: vol not identifiable
    iv = implied_vol(price, 100.0, k, t, r, q, kind)
    assert iv == pytest.approx(vol, abs=1e-6)


def test_vectorised_over_strikes_and_maturities() -> None:
    strikes = np.linspace(70.0, 140.0, 15)[:, None]
    maturities = np.array([0.25, 1.0, 3.0])[None, :]
    vols = 0.2 + 0.1 * (strikes / 100.0 - 1.0) ** 2 + 0.0 * maturities
    prices = bs_price(100.0, strikes, maturities, 0.02, vols)
    iv = implied_vol(prices, 100.0, strikes, maturities, 0.02)
    assert isinstance(iv, np.ndarray)
    assert iv.shape == (15, 3)
    np.testing.assert_allclose(iv, vols, atol=1e-8)


def test_arbitrage_bounds() -> None:
    iv = implied_vol([0.0, 100.0, 200.0], 100.0, 100.0, 1.0, 0.05)
    assert np.isnan(iv).all()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(200.0, 100.0, 100.0, 1.0, 0.05, strict=True)
    with pytest.raises(ValueError, match="> 0"):
        implied_vol(5.0, 100.0, -1.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="option_type"):
        implied_vol(5.0, 100.0, 100.0, 1.0, 0.05, option_type="straddle")


def test_brent_fallback_for_extreme_quotes() -> None:
    """Very high vol and deep OTM strikes push Newton out of its safe range."""
    price = float(bs_price(100.0, 400.0, 2.0, 0.0, 3.0))
    assert implied_vol(price, 100.0, 400.0, 2.0, 0.0) == pytest.approx(3.0, abs=1e-6)
    # far out of the money with a short maturity: the price is at the machine-precision
    # floor, so the volatility is not identifiable and nan is returned
    tiny = float(bs_price(100.0, 50.0, 0.1, 0.0, 0.2, option_type="put"))
    assert math.isnan(implied_vol(tiny, 100.0, 50.0, 0.1, 0.0, option_type="put"))  # type: ignore[arg-type]


def test_newton_budget_exhausted_falls_back_to_brent() -> None:
    price = float(bs_price(100.0, 120.0, 1.0, 0.01, 0.45))
    assert implied_vol(price, 100.0, 120.0, 1.0, 0.01, max_iter=1) == pytest.approx(0.45, abs=1e-9)


def test_brent_returns_nan_when_not_bracketed() -> None:
    from mcengine.volatility.implied import _brent

    assert math.isnan(_brent(99.99999, 100.0, 100.0, 1.0, 0.0, 0.0, "call", 1e-12))
    assert _brent(10.0, 100.0, 100.0, 1.0, 0.0, 0.0, "call", 1e-12) == pytest.approx(
        0.2506, abs=1e-3
    )
