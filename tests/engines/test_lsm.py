"""Longstaff-Schwartz: validation against the Bermudan tree and policy properties."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.engines.analytic import bs_price
from mcengine.engines.lsm import basis_functions, fit_lsm, price_lsm
from mcengine.engines.monte_carlo import price_mc
from mcengine.engines.tree import price_tree
from mcengine.models.gbm import GBM
from mcengine.products.american import AmericanOption
from mcengine.products.european import EuropeanOption
from tests.conftest import TOL_SE

LS_MODEL = GBM(36.0, 0.06, 0.2)
LS_PUT = AmericanOption(40.0, 1.0, "put", 50)


@pytest.mark.parametrize("basis", ["laguerre", "monomial"])
def test_lsm_matches_bermudan_tree(basis: str) -> None:
    """Longstaff & Schwartz (2001), Table 1, first row: S0=36, sigma=0.2, T=1."""
    ref = price_tree(LS_MODEL, LS_PUT, n_steps=2500).price
    res = price_lsm(LS_MODEL, LS_PUT, n_paths=50_000, basis=basis, seed=21)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error
    assert res.method == "lsm"
    assert res.diagnostics["n_train"] == 50_000


def test_american_call_without_dividends_equals_european() -> None:
    model = GBM(40.0, 0.06, 0.2)
    res = price_lsm(model, AmericanOption(40.0, 1.0, "call", 50), n_paths=50_000, seed=3)
    ref = float(bs_price(40.0, 40.0, 1.0, 0.06, 0.2))
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error


def test_chunked_pricing_pass_is_invariant() -> None:
    a = price_lsm(LS_MODEL, LS_PUT, n_paths=6000, n_train=5000, seed=5, chunk_size=6000)
    b = price_lsm(LS_MODEL, LS_PUT, n_paths=6000, n_train=5000, seed=5, chunk_size=1000)
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_exercise_boundary_is_below_strike_and_increasing() -> None:
    fit = fit_lsm(LS_MODEL, LS_PUT, n_paths=50_000, seed=8)
    boundary = fit.exercise_boundary(LS_PUT)
    assert boundary.shape == (50,)
    assert np.all(boundary[1:-1] < 40.0)  # the first date may have no exercise region
    assert boundary[45] > boundary[5]
    call = AmericanOption(40.0, 1.0, "call", 10)
    call_fit = fit_lsm(GBM(40.0, 0.06, 0.2, q=0.1), call, n_paths=20_000, seed=8)
    assert np.nanmin(call_fit.exercise_boundary(call)) >= 40.0


def test_multiple_steps_per_date_and_degenerate_dates() -> None:
    far_otm = AmericanOption(5.0, 1.0, "put", 5)
    res = price_lsm(LS_MODEL, far_otm, n_paths=2000, steps_per_date=3, seed=1)
    assert res.price == 0.0
    assert res.n_steps == 15


def test_dates_without_enough_itm_paths_never_exercise() -> None:
    """Deep out-of-the-money early on: no regression, so no early exercise there."""
    model = GBM(36.0, 0.06, 0.2, q=0.05)
    call = AmericanOption(50.0, 1.0, "call", 10)
    fit = fit_lsm(model, call, n_paths=200, seed=2)
    assert not fit.fitted[0]
    assert np.isinf(fit.continuation(0, np.array([60.0, 70.0]))).all()
    assert np.isnan(fit.exercise_boundary(call)[0])
    assert np.isnan(fit.support[0]).all()


def test_basis_functions() -> None:
    x = np.array([0.5, 1.0])
    lag = basis_functions(x, "laguerre", 3)
    assert lag.shape == (2, 4)
    np.testing.assert_allclose(lag[:, 1], np.exp(-x / 2))
    np.testing.assert_allclose(lag[:, 2], np.exp(-x / 2) * (1 - x))
    mono = basis_functions(x, "monomial", 2)
    np.testing.assert_allclose(mono, [[1, 0.5, 0.25], [1, 1, 1]])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"basis": "chebyshev"}, "basis"), ({"degree": 0}, "degree"), ({"n_paths": 1}, "n_paths")],
)
def test_invalid(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        price_lsm(LS_MODEL, LS_PUT, **kwargs)  # type: ignore[arg-type]


def test_product_type_checks(bs_model: GBM) -> None:
    with pytest.raises(ValueError, match="AmericanOption"):
        price_lsm(bs_model, EuropeanOption(100.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="price_lsm"):
        price_mc(bs_model, AmericanOption(100.0, 1.0))
