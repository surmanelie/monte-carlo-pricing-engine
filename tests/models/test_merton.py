"""Merton jump-diffusion: exact simulation, series formula, Fourier cross-checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import poisson

from mcengine.engines.analytic import bs_price, merton_price, price_analytic
from mcengine.engines.fourier import carr_madan_prices, gil_pelaez_price
from mcengine.engines.monte_carlo import price_mc
from mcengine.models.merton import Merton, poisson_inverse
from mcengine.products.european import EuropeanOption
from tests.conftest import TOL_SE

MODEL = Merton(100.0, 0.05, 0.2, 1.0, -0.1, 0.15)


def test_poisson_inverse_matches_scipy() -> None:
    u = ndtr(np.random.default_rng(0).standard_normal(10_000))
    np.testing.assert_array_equal(poisson_inverse(u, 0.7), poisson.ppf(u, 0.7).astype(int))
    assert poisson_inverse(u, 0.0).sum() == 0


def test_terminal_mean_is_forward() -> None:
    s_t = MODEL.simulate_paths(np.array([0.0, 2.0]), 200_000, seed=1)[:, -1]
    se = s_t.std() / math.sqrt(s_t.size)
    assert abs(s_t.mean() - MODEL.forward(2.0)) < TOL_SE * se


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
def test_series_matches_fourier(kind: str, strike: float) -> None:
    series = merton_price(100.0, strike, 1.0, 0.05, 0.2, 1.0, -0.1, 0.15, 0.0, kind)
    assert series == pytest.approx(gil_pelaez_price(MODEL, strike, 1.0, kind), abs=1e-9)
    cm = carr_madan_prices(MODEL, [strike], 1.0, kind)[0]
    assert series == pytest.approx(cm, abs=1e-7)


def test_no_jumps_is_black_scholes() -> None:
    no_jumps = merton_price(100.0, 95.0, 1.0, 0.05, 0.2, 0.0, -0.1, 0.15)
    assert no_jumps == pytest.approx(float(bs_price(100, 95, 1, 0.05, 0.2)), rel=1e-14)


@pytest.mark.parametrize("method", ["plain", "antithetic", "cv"])
def test_mc_matches_series(method: str) -> None:
    ref = price_analytic(MODEL, EuropeanOption(100.0, 1.0))
    assert ref.method == "merton-series"
    res = price_mc(MODEL, EuropeanOption(100.0, 1.0), n_paths=100_000, method=method, seed=9)
    assert res.std_error is not None
    assert abs(res.price - ref.price) < TOL_SE * res.std_error


def test_multi_step_paths_have_same_terminal_law() -> None:
    res = price_mc(MODEL, EuropeanOption(100.0, 1.0), n_paths=100_000, n_steps=12, seed=4)
    ref = merton_price(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, -0.1, 0.15)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error


def test_invalid() -> None:
    with pytest.raises(ValueError, match="lam"):
        Merton(100.0, 0.05, 0.2, -1.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="tol"):
        merton_price(100, 100, 1, 0.05, 0.2, 1.0, 0.0, 0.1, tol=0.0)
    with pytest.raises(ValueError, match="lam and delta_j"):
        merton_price(100, 100, 1, 0.05, 0.2, 1.0, 0.0, -0.1)
