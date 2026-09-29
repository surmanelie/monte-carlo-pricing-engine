from __future__ import annotations

import numpy as np
import pytest

from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption


def test_payoffs_on_known_path() -> None:
    times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 110.0, 130.0]])
    arith = AsianOption(100.0, 1.0, n_fixings=2)
    geo = arith.with_average("geometric")
    assert arith.payoff(paths, times)[0] == pytest.approx(20.0)
    assert geo.payoff(paths, times)[0] == pytest.approx(np.sqrt(110.0 * 130.0) - 100.0)
    put = AsianOption(130.0, 1.0, n_fixings=2, option_type="put")
    assert put.payoff(paths, times)[0] == pytest.approx(10.0)
    assert arith.is_path_dependent
    assert "arith" in arith.label
    assert "m=2" in arith.label


def test_geometric_below_arithmetic_average(bs_model: GBM) -> None:
    """AM-GM inequality path by path."""
    product = AsianOption(100.0, 1.0, n_fixings=12)
    times = np.concatenate(([0.0], product.monitoring_times()))
    paths = bs_model.simulate_paths(times, 1000, seed=0)
    arith = product.average_price(paths, times)
    geo = product.with_average("geometric").average_price(paths, times)
    assert np.all(geo <= arith + 1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fixings": 0},
        {"average": "harmonic"},
        {"option_type": "binary"},
        {"strike": -1.0},
    ],
)
def test_invalid(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {"strike": 100.0, "maturity": 1.0} | kwargs
    with pytest.raises(ValueError, match=r"must|unsupported"):
        AsianOption(**args)  # type: ignore[arg-type]
