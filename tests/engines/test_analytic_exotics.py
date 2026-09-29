"""Kemna-Vorst geometric Asian and Reiner-Rubinstein barrier formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcengine.engines.analytic import (
    barrier_price,
    barrier_price_discrete_bgk,
    bs_price,
    geometric_asian_price,
    price_analytic,
)
from mcengine.engines.monte_carlo import price_mc
from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BarrierOption
from tests.conftest import TOL_SE


def test_geometric_asian_single_fixing_is_black_scholes() -> None:
    for kind in ("call", "put"):
        kv = geometric_asian_price(100, 95, [0.7], 0.03, 0.3, 0.01, kind)
        assert kv == pytest.approx(float(bs_price(100, 95, 0.7, 0.03, 0.3, 0.01, kind)), rel=1e-12)


def test_geometric_asian_parity() -> None:
    """C - P = e^{-rT}(E[G] - K) with E[G] = exp(mu_G + sigma_G^2 / 2)."""
    t = np.arange(1, 13) / 12
    call = geometric_asian_price(100, 100, t, 0.05, 0.2, 0.0, "call")
    put = geometric_asian_price(100, 100, t, 0.05, 0.2, 0.0, "put")
    mu = math.log(100) + (0.05 - 0.02) * t.mean()
    var = 0.04 * np.minimum.outer(t, t).mean()
    assert call - put == pytest.approx(math.exp(-0.05) * (math.exp(mu + var / 2) - 100), rel=1e-12)


def test_geometric_asian_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="fixing_times"):
        geometric_asian_price(100, 100, [0.5, 0.5], 0.05, 0.2)
    with pytest.raises(ValueError, match="must be finite"):
        geometric_asian_price(100, 100, [1.0], 0.05, 0.0)


@pytest.mark.parametrize("method", ["plain", "antithetic"])
def test_geometric_asian_mc_matches_kemna_vorst(bs_model: GBM, method: str) -> None:
    product = AsianOption(100.0, 1.0, 12, average="geometric")
    ref = price_analytic(bs_model, product)
    assert ref.method == "kemna-vorst"
    res = price_mc(bs_model, product, n_paths=100_000, method=method, seed=31)
    assert res.std_error is not None
    assert abs(res.price - ref.price) < TOL_SE * res.std_error


barrier_params = {
    "s": st.floats(80.0, 120.0),
    "k": st.floats(60.0, 140.0),
    "t": st.floats(0.1, 3.0),
    "r": st.floats(0.0, 0.1),
    "v": st.floats(0.1, 0.6),
    "q": st.floats(0.0, 0.05),
    "gap": st.floats(0.02, 0.4),
}


@settings(max_examples=150, deadline=None)
@given(**barrier_params, down=st.booleans(), kind=st.sampled_from(["call", "put"]))
def test_barrier_in_out_parity(
    s: float, k: float, t: float, r: float, v: float, q: float, gap: float, down: bool, kind: str
) -> None:
    h = s * (1 - gap) if down else s * (1 + gap)
    prefix = "down" if down else "up"
    knock_in = barrier_price(s, k, h, t, r, v, q, f"{prefix}-and-in", kind)
    knock_out = barrier_price(s, k, h, t, r, v, q, f"{prefix}-and-out", kind)
    vanilla = float(bs_price(s, k, t, r, v, q, kind))
    assert knock_in >= 0.0
    assert knock_out >= 0.0
    assert knock_in + knock_out == pytest.approx(vanilla, abs=1e-9 * s)


def test_barrier_limits() -> None:
    vanilla = float(bs_price(100, 100, 1, 0.05, 0.2))
    assert barrier_price(100, 100, 1e-6, 1, 0.05, 0.2) == pytest.approx(vanilla, rel=1e-12)
    assert barrier_price(100, 100, 1e6, 1, 0.05, 0.2, 0.0, "up-and-in") == pytest.approx(
        0.0, abs=1e-12
    )
    # an up-and-out call whose strike is above the barrier can never pay
    assert barrier_price(100, 130, 120, 1, 0.05, 0.2, 0.0, "up-and-out") == 0.0
    # a down-and-out put whose strike is below the barrier can never pay
    assert barrier_price(100, 80, 90, 1, 0.05, 0.2, 0.0, "down-and-out", "put") == 0.0


def test_barrier_invalid_spot() -> None:
    with pytest.raises(ValueError, match="live side"):
        barrier_price(100, 100, 105, 1, 0.05, 0.2, 0.0, "down-and-out")
    with pytest.raises(ValueError, match="live side"):
        barrier_price(100, 100, 95, 1, 0.05, 0.2, 0.0, "up-and-in")


def test_bgk_discrete_approximation_is_between_raw_monitoring_limits() -> None:
    """Discrete knock-out is worth more than continuous; BGK sits above RR and tends to it."""
    cont = barrier_price(100, 100, 95, 1, 0.05, 0.2)
    approx = [barrier_price_discrete_bgk(100, 100, 95, 1, 0.05, 0.2, m) for m in (4, 16, 64, 256)]
    assert all(a > cont for a in approx)
    assert np.all(np.diff(approx) < 0)


def test_bgk_approximates_discrete_monitoring_mc(bs_model: GBM) -> None:
    product = BarrierOption(100.0, 1.0, 95.0, "down-and-out", "call", n_monitoring=50)
    res = price_mc(bs_model, product, n_paths=200_000, method="cv", seed=8)
    approx = barrier_price_discrete_bgk(100, 100, 95, 1, 0.05, 0.2, 50)
    assert res.std_error is not None
    # BGK has an o(sqrt(dt)) error, so allow a small absolute slack on top of 4 SE.
    assert abs(res.price - approx) < TOL_SE * res.std_error + 0.01
