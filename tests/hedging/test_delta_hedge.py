"""Discrete delta hedging: error scaling, misspecification, transaction costs."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.hedging.delta_hedge import fit_std_slope, hedge_pnl, hedging_experiment
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston

FREQS = [4, 12, 42, 126]


def test_gbm_hedging_error_scales_like_inverse_sqrt_n(bs_model: GBM) -> None:
    results = hedging_experiment(
        bs_model, 100.0, 1.0, FREQS, hedge_sigma=0.2, n_paths=20_000, seed=1
    )
    slope = fit_std_slope(results)
    assert -0.6 < slope < -0.4
    for res in results:
        assert abs(res.mean) < 4.0 * res.std_error  # fair premium: no systematic P&L
    assert results[-1].relative_std() < results[0].relative_std()


def test_misspecified_heston_hedge_does_not_vanish() -> None:
    model = Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.5, -0.7)
    results = hedging_experiment(model, 100.0, 1.0, FREQS, hedge_sigma=0.2, n_paths=10_000, seed=2)
    stds = [r.std for r in results]
    assert stds[-1] > 0.5 * stds[1]  # volatility risk: the error plateaus
    assert fit_std_slope(results) > -0.4


def test_transaction_costs_reduce_mean_pnl_more_when_rebalancing_often(bs_model: GBM) -> None:
    free = hedging_experiment(bs_model, 100.0, 1.0, [4, 126], hedge_sigma=0.2, n_paths=5000, seed=3)
    costly = hedging_experiment(
        bs_model, 100.0, 1.0, [4, 126], hedge_sigma=0.2, cost_rate=0.002, n_paths=5000, seed=3
    )
    drag = [c.mean - f.mean for c, f in zip(costly, free, strict=True)]
    assert drag[0] < 0.0
    assert drag[1] < drag[0]
    assert costly[1].cost_rate == 0.002


def test_put_and_dividends() -> None:
    model = GBM(100.0, 0.03, 0.25, q=0.02)
    res = hedging_experiment(
        model, 95.0, 0.5, [126], hedge_sigma=0.25, option_type="put", n_paths=20_000, seed=4
    )[0]
    assert abs(res.mean) < 4.0 * res.std_error
    assert res.premium > 0.0


def test_single_rebalance_is_static_hedge(bs_model: GBM) -> None:
    times = np.array([0.0, 1.0])
    paths = np.array([[100.0, 120.0], [100.0, 80.0]])
    res = hedge_pnl(paths, times, 100.0, 0.0, 0.0, 0.2, 1)
    assert res.pnl.shape == (2,)
    assert res.n_rebalance == 1


def test_invalid(bs_model: GBM) -> None:
    times = np.linspace(0.0, 1.0, 11)
    paths = np.full((2, 11), 100.0)
    with pytest.raises(ValueError, match="multiple"):
        hedge_pnl(paths, times, 100.0, 0.0, 0.0, 0.2, 3)
    with pytest.raises(ValueError, match="hedge_sigma"):
        hedge_pnl(paths, times, 100.0, 0.0, 0.0, 0.0, 5)
    with pytest.raises(ValueError, match="cost_rate"):
        hedge_pnl(paths, times, 100.0, 0.0, 0.0, 0.2, 5, cost_rate=-0.1)
