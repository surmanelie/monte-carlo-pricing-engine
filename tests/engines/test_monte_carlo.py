"""Monte Carlo engine: convergence to Black-Scholes, variance reduction, reproducibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcengine.engines.analytic import bs_digital_price, bs_price
from mcengine.engines.monte_carlo import (
    ControlVariate,
    auto_chunk_size,
    build_time_grid,
    chunk_sizes,
    price_mc,
    simulate_discounted_payoffs,
)
from mcengine.models.gbm import GBM
from mcengine.products.european import DigitalOption, EuropeanOption
from tests.conftest import TOL_SE

N = 200_000


@pytest.mark.parametrize("method", ["plain", "antithetic", "cv"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_converges_to_black_scholes(bs_model: GBM, method: str, option_type: str) -> None:
    product = EuropeanOption(100.0, 1.0, option_type)
    ref = float(bs_price(100, 100, 1.0, 0.05, 0.2, option_type=option_type))
    res = price_mc(bs_model, product, n_paths=N, method=method, seed=2024)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error
    assert res.ci_low is not None
    assert res.ci_high is not None
    assert res.ci_low < res.price < res.ci_high
    assert res.n_paths == N
    assert res.n_steps == 1


@pytest.mark.parametrize("method", ["plain", "antithetic", "cv"])
def test_digital_converges(bs_model: GBM, method: str) -> None:
    ref = float(bs_digital_price(100, 105, 1.0, 0.05, 0.2))
    res = price_mc(bs_model, DigitalOption(105.0, 1.0), n_paths=N, method=method, seed=7)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error


def test_mc_put_call_parity(bs_model: GBM) -> None:
    """Same seed -> same paths, so parity holds up to the (tiny) SE of S_T's mean."""
    call = price_mc(bs_model, EuropeanOption(100.0, 1.0, "call"), n_paths=N, seed=3)
    put = price_mc(bs_model, EuropeanOption(100.0, 1.0, "put"), n_paths=N, seed=3)
    s_t = simulate_discounted_payoffs(bs_model, EuropeanOption(1e-12, 1.0), N, seed=3)
    assert call.price - put.price == pytest.approx(s_t.mean() - 100.0 * math.exp(-0.05), abs=1e-9)


@pytest.mark.parametrize("method", ["antithetic", "cv"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_variance_reduction_lowers_standard_error(
    bs_model: GBM, method: str, option_type: str
) -> None:
    product = EuropeanOption(100.0, 1.0, option_type)
    plain = price_mc(bs_model, product, n_paths=N, method="plain", seed=11)
    reduced = price_mc(bs_model, product, n_paths=N, method=method, seed=11)
    assert plain.std_error is not None
    assert reduced.std_error is not None
    assert reduced.std_error < plain.std_error


def test_cv_diagnostics(bs_model: GBM, atm_call: EuropeanOption) -> None:
    res = price_mc(bs_model, atm_call, n_paths=N, method="cv", seed=5)
    assert 0.0 < res.diagnostics["beta"] < 1.0
    assert res.diagnostics["vr_factor"] > 1.0
    assert 0.0 < res.diagnostics["corr"] <= 1.0


@pytest.mark.parametrize("method", ["plain", "antithetic", "cv"])
def test_chunked_equals_single_chunk(bs_model: GBM, atm_call: EuropeanOption, method: str) -> None:
    single = price_mc(bs_model, atm_call, n_paths=10_000, method=method, seed=9, chunk_size=10_000)
    chunked = price_mc(bs_model, atm_call, n_paths=10_000, method=method, seed=9, chunk_size=998)
    assert chunked.price == pytest.approx(single.price, rel=1e-12)
    assert chunked.std_error == pytest.approx(single.std_error, rel=1e-9)


def test_seed_reproducibility_and_rng_injection(bs_model: GBM, atm_call: EuropeanOption) -> None:
    a = price_mc(bs_model, atm_call, n_paths=1000, seed=1)
    b = price_mc(bs_model, atm_call, n_paths=1000, rng=np.random.default_rng(1))
    assert a.price == b.price
    c = price_mc(bs_model, atm_call, n_paths=1000, seed=2)
    assert c.price != a.price


def test_multi_step_grid_gives_same_distribution(bs_model: GBM, atm_call: EuropeanOption) -> None:
    res = price_mc(bs_model, atm_call, n_paths=N, n_steps=12, seed=4)
    assert res.n_steps == 12
    assert res.std_error is not None
    assert abs(res.price - float(bs_price(100, 100, 1, 0.05, 0.2))) < TOL_SE * res.std_error


def test_custom_control_variate(bs_model: GBM, atm_call: EuropeanOption) -> None:
    ref = float(bs_price(100, 100, 1, 0.05, 0.2))
    # Control: the discounted put itself (known BS mean) -> near-perfect via parity.
    put_ref = float(bs_price(100, 100, 1, 0.05, 0.2, option_type="put"))
    put = EuropeanOption(100.0, 1.0, "put")
    control = ControlVariate("put", put_ref, lambda p, t: math.exp(-0.05) * put.payoff(p, t))
    res = price_mc(bs_model, atm_call, n_paths=20_000, method="cv", seed=1, control=control)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error


def test_degenerate_control_has_zero_beta(bs_model: GBM, atm_call: EuropeanOption) -> None:
    control = ControlVariate("const", 1.0, lambda p, t: np.ones(p.shape[0]))
    res = price_mc(bs_model, atm_call, n_paths=1000, method="cv", seed=1, control=control)
    assert res.diagnostics["beta"] == 0.0
    assert res.diagnostics["corr"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_paths": 1}, "n_paths"),
        ({"method": "magic"}, "method"),
        ({"method": "antithetic", "n_paths": 1001}, "even"),
        ({"method": "antithetic", "chunk_size": 101}, "even"),
        ({"chunk_size": 1}, "chunk_size"),
        ({"n_steps": 0}, "n_steps"),
    ],
)
def test_invalid_arguments(
    bs_model: GBM, atm_call: EuropeanOption, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        price_mc(bs_model, atm_call, **kwargs)  # type: ignore[arg-type]


def test_time_grid(bs_model: GBM, atm_call: EuropeanOption) -> None:
    np.testing.assert_allclose(build_time_grid(bs_model, atm_call), [0.0, 1.0])
    assert build_time_grid(bs_model, atm_call, 4).size == 5
    two_dates = _TwoDates()
    with pytest.raises(ValueError, match="monitoring"):
        build_time_grid(bs_model, two_dates, 4)
    np.testing.assert_allclose(build_time_grid(bs_model, two_dates), [0.0, 0.3, 1.0])
    with pytest.raises(ValueError, match="n_steps"):
        build_time_grid(_DiscretisedGBM(100.0, 0.05, 0.2), atm_call)


class _TwoDates(EuropeanOption):
    def __init__(self) -> None:
        super().__init__(100.0, 1.0)

    def monitoring_times(self) -> np.ndarray:
        return np.array([0.3, 1.0])


class _DiscretisedGBM(GBM):
    exact_simulation = False


def test_chunk_helpers() -> None:
    assert list(chunk_sizes(10, 4)) == [4, 4, 2]
    size = auto_chunk_size(253, 1)
    assert size % 2 == 0
    assert 2 <= size <= 2**16
