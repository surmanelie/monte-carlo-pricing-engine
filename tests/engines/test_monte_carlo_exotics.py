"""Monte Carlo on path-dependent products: Asian control variate and barrier estimators."""

from __future__ import annotations

import pytest

from mcengine.engines.analytic import barrier_price, price_analytic
from mcengine.engines.monte_carlo import default_control, price_mc
from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BARRIER_TYPES, BarrierOption
from mcengine.products.european import EuropeanOption
from tests.conftest import TOL_SE


def test_default_controls(bs_model: GBM) -> None:
    assert default_control(bs_model, AsianOption(100.0, 1.0, 12)).name == "geometric-asian"
    assert default_control(bs_model, EuropeanOption(100.0, 1.0)).name == "discounted-terminal"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_asian_control_variate_reduces_variance(bs_model: GBM, option_type: str) -> None:
    product = AsianOption(100.0, 1.0, 12, option_type)
    plain = price_mc(bs_model, product, n_paths=50_000, seed=4)
    cv = price_mc(bs_model, product, n_paths=50_000, method="cv", seed=4)
    assert plain.std_error is not None
    assert cv.std_error is not None
    assert cv.std_error < plain.std_error / 10.0
    assert cv.diagnostics["vr_factor"] > 100.0


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bridge_estimator_matches_continuous_formula(
    bs_model: GBM, barrier_type: str, option_type: str
) -> None:
    barrier = 90.0 if barrier_type.startswith("down") else 115.0
    product = BarrierOption(
        100.0, 1.0, barrier, barrier_type, option_type, 25, "bridge", bs_model.sigma
    )
    ref = price_analytic(bs_model, product)
    assert ref.method == "reiner-rubinstein"
    res = price_mc(bs_model, product, n_paths=100_000, seed=12)
    assert res.std_error is not None
    assert abs(res.price - ref.price) < TOL_SE * res.std_error


def test_discretisation_bias_shrinks_and_corrections_help(bs_model: GBM) -> None:
    cont = barrier_price(100, 100, 95, 1, 0.05, 0.2)
    errors = {}
    for correction in ("none", "bgk"):
        sigma = None if correction == "none" else 0.2
        for m in (8, 128):
            product = BarrierOption(100.0, 1.0, 95.0, "down-and-out", "call", m, correction, sigma)
            errors[correction, m] = (
                price_mc(bs_model, product, n_paths=100_000, seed=5).price - cont
            )
    assert errors["none", 8] > errors["none", 128] > 0.0  # discrete knock-out is worth more
    assert abs(errors["bgk", 8]) < errors["none", 8] / 3.0
