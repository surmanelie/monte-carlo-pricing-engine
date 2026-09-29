from __future__ import annotations

import numpy as np
import pytest

from mcengine.models.gbm import GBM
from mcengine.products.barrier import BARRIER_TYPES, CORRECTIONS, BarrierOption
from mcengine.products.european import EuropeanOption


def _product(
    barrier_type: str, correction: str = "none", option_type: str = "call"
) -> BarrierOption:
    barrier = 90.0 if barrier_type.startswith("down") else 115.0
    sigma = None if correction == "none" else 0.2
    return BarrierOption(100.0, 1.0, barrier, barrier_type, option_type, 12, correction, sigma)


@pytest.mark.parametrize("correction", CORRECTIONS)
@pytest.mark.parametrize("barrier_type", ["down-and-out", "up-and-out"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_in_out_parity_path_by_path(
    bs_model: GBM, correction: str, barrier_type: str, option_type: str
) -> None:
    out = _product(barrier_type, correction, option_type)
    knock_in = out.parity_partner
    assert knock_in.is_knock_in
    assert knock_in.parity_partner == out
    times = np.concatenate(([0.0], out.monitoring_times()))
    paths = bs_model.simulate_paths(times, 5000, seed=1)
    vanilla = EuropeanOption(100.0, 1.0, option_type).payoff(paths, times)
    total = out.payoff(paths, times) + knock_in.payoff(paths, times)
    np.testing.assert_allclose(total, vanilla, rtol=0, atol=1e-12)


def test_discrete_knock_out_indicator() -> None:
    times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 89.0, 120.0], [100.0, 95.0, 120.0]])
    product = BarrierOption(100.0, 1.0, 90.0, "down-and-out", "call", n_monitoring=2)
    np.testing.assert_array_equal(product.payoff(paths, times), [0.0, 20.0])
    up = BarrierOption(100.0, 1.0, 110.0, "up-and-in", "call", n_monitoring=2)
    np.testing.assert_array_equal(up.payoff(paths, times), [20.0, 20.0])


def test_bridge_survival_is_below_discrete_survival(bs_model: GBM) -> None:
    """Continuous monitoring knocks out more often than discrete monitoring."""
    raw, bridge = _product("down-and-out"), _product("down-and-out", "bridge")
    times = np.concatenate(([0.0], raw.monitoring_times()))
    paths = bs_model.simulate_paths(times, 5000, seed=2)
    s_raw, s_bridge = raw.survival(paths, times), bridge.survival(paths, times)
    assert np.all(s_bridge <= s_raw + 1e-15)
    assert np.all((s_bridge >= 0.0) & (s_bridge <= 1.0))
    assert s_bridge.mean() < s_raw.mean()


def test_bgk_shift_moves_barrier_towards_spot(bs_model: GBM) -> None:
    times = np.concatenate(([0.0], _product("up-and-out").monitoring_times()))
    paths = bs_model.simulate_paths(times, 5000, seed=3)
    for bt in ("down-and-out", "up-and-out"):
        raw, bgk = _product(bt), _product(bt, "bgk")
        assert bgk.survival(paths, times).mean() < raw.survival(paths, times).mean()


def test_spot_on_wrong_side_raises() -> None:
    times = np.array([0.0, 1.0])
    paths = np.array([[85.0, 100.0]])
    with pytest.raises(ValueError, match="strictly above"):
        _product("down-and-out").payoff(paths, times)
    with pytest.raises(ValueError, match="strictly below"):
        BarrierOption(100.0, 1.0, 80.0, "up-and-in", n_monitoring=1).check_spot(85.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"barrier": 0.0},
        {"barrier_type": "double-knock-out"},
        {"correction": "magic"},
        {"correction": "bgk"},
        {"correction": "bridge", "correction_sigma": -0.1},
        {"n_monitoring": 0},
    ],
)
def test_invalid(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {"strike": 100.0, "maturity": 1.0, "barrier": 90.0} | kwargs
    with pytest.raises(ValueError, match=r"must|unsupported|requires"):
        BarrierOption(**args)  # type: ignore[arg-type]


def test_labels_and_flags() -> None:
    for bt in BARRIER_TYPES:
        product = _product(bt)
        assert bt in product.label
        assert product.is_down == bt.startswith("down")
