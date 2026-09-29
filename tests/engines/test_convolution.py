"""Recursive-convolution reference for discrete arithmetic Asian options."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcengine.engines.analytic import bs_price
from mcengine.engines.convolution import (
    asian_arithmetic_price,
    log_sum_density,
    price_asian_convolution,
)
from mcengine.engines.monte_carlo import price_mc
from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption
from tests.conftest import TOL_SE


def test_single_fixing_equals_black_scholes(bs_model: GBM) -> None:
    for kind in ("call", "put"):
        price = asian_arithmetic_price(bs_model, AsianOption(100.0, 1.0, 1, kind))
        ref = float(bs_price(100, 100, 1, 0.05, 0.2, option_type=kind))
        assert price == pytest.approx(ref, abs=1e-6)


def test_density_integrates_to_one_with_exact_mean(bs_model: GBM) -> None:
    times = AsianOption(100.0, 1.0, 12).monitoring_times()
    x, f = log_sum_density(bs_model, times)
    assert np.trapezoid(f, x) == pytest.approx(1.0, abs=1e-12)
    exact_mean = np.exp(0.05 * times).sum()
    assert np.trapezoid(np.exp(x) * f, x) == pytest.approx(exact_mean, rel=1e-6)


@pytest.mark.parametrize("n_fixings", [12, 52])
def test_grid_convergence(bs_model: GBM, n_fixings: int) -> None:
    product = AsianOption(100.0, 1.0, n_fixings)
    coarse = asian_arithmetic_price(bs_model, product, 2**14)
    fine = asian_arithmetic_price(bs_model, product, 2**15)
    assert fine == pytest.approx(coarse, rel=1e-5)


def test_put_call_parity(bs_model: GBM) -> None:
    call = asian_arithmetic_price(bs_model, AsianOption(100.0, 1.0, 12, "call"))
    put = asian_arithmetic_price(bs_model, AsianOption(100.0, 1.0, 12, "put"))
    times = np.arange(1, 13) / 12
    forward_avg = 100.0 * np.exp(0.05 * times).mean()
    assert call - put == pytest.approx(math.exp(-0.05) * (forward_avg - 100.0), abs=2e-6)


def test_agrees_with_control_variate_mc(bs_model: GBM) -> None:
    product = AsianOption(100.0, 1.0, 12)
    ref = price_asian_convolution(bs_model, product)
    assert ref.method == "recursive-convolution"
    res = price_mc(bs_model, product, n_paths=100_000, method="cv", seed=77)
    assert res.std_error is not None
    assert abs(res.price - ref.price) < TOL_SE * res.std_error
    assert res.diagnostics["vr_factor"] > 100.0


def test_invalid_inputs(bs_model: GBM) -> None:
    with pytest.raises(ValueError, match="arithmetic"):
        asian_arithmetic_price(bs_model, AsianOption(100.0, 1.0, 12, average="geometric"))
    with pytest.raises(ValueError, match="fixing_times"):
        log_sum_density(bs_model, np.array([0.5, 0.25]))
    with pytest.raises(ValueError, match="n_grid"):
        log_sum_density(bs_model, np.array([1.0]), n_grid=8)
