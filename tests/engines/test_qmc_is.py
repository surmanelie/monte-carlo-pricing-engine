"""Randomised QMC and importance sampling in the Monte Carlo engine."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.engines.analytic import bs_digital_price, bs_price, price_analytic
from mcengine.engines.convolution import asian_arithmetic_price
from mcengine.engines.fourier import gil_pelaez_price
from mcengine.engines.monte_carlo import (
    default_is_shift,
    price_mc,
    qmc_path_count,
    reference_vol,
)
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BarrierOption
from mcengine.products.european import DigitalOption, EuropeanOption
from tests.conftest import TOL_SE
from tests.engines.test_analytic import _Unsupported

N_QMC = 16 * 2**12


def _within(res_price: float, se: float | None, ref: float) -> None:
    assert se is not None
    assert abs(res_price - ref) < TOL_SE * se


def test_qmc_european_is_far_more_accurate(bs_model: GBM, atm_call: EuropeanOption) -> None:
    ref = float(bs_price(100, 100, 1, 0.05, 0.2))
    qmc = price_mc(bs_model, atm_call, n_paths=N_QMC, method="qmc", seed=1)
    plain = price_mc(bs_model, atm_call, n_paths=N_QMC, method="plain", seed=1)
    _within(qmc.price, qmc.std_error, ref)
    assert qmc.method == "qmc-sobol-bb"
    assert qmc.diagnostics["n_scrambles"] == 16
    assert qmc.std_error is not None
    assert plain.std_error is not None
    assert qmc.std_error < plain.std_error / 20.0


def test_qmc_asian_bridge_beats_standard_construction(bs_model: GBM) -> None:
    product = AsianOption(100.0, 1.0, 16)
    ref = asian_arithmetic_price(bs_model, product)
    bb = price_mc(bs_model, product, n_paths=N_QMC, method="qmc", seed=2)
    std = price_mc(bs_model, product, n_paths=N_QMC, method="qmc", seed=2, bridge=False)
    _within(bb.price, bb.std_error, ref)
    _within(std.price, std.std_error, ref)
    assert std.method == "qmc-sobol"
    assert bb.std_error is not None
    assert std.std_error is not None
    assert bb.std_error < std.std_error


def test_qmc_barrier_heston_and_merton(bs_model: GBM) -> None:
    barrier = BarrierOption(100.0, 1.0, 90.0, "down-and-out", "call", 32, "bridge", 0.2)
    res = price_mc(bs_model, barrier, n_paths=N_QMC, method="qmc", seed=3)
    _within(res.price, res.std_error, price_analytic(bs_model, barrier).price)
    heston = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7)
    res = price_mc(
        heston, EuropeanOption(100.0, 1.0), n_paths=N_QMC, n_steps=16, method="qmc", seed=4
    )
    _within(res.price, res.std_error, gil_pelaez_price(heston, 100.0, 1.0))
    merton = Merton(100.0, 0.05, 0.2, 1.0, -0.1, 0.15)
    res = price_mc(merton, EuropeanOption(100.0, 1.0), n_paths=N_QMC, method="qmc", seed=5)
    _within(res.price, res.std_error, price_analytic(merton, EuropeanOption(100.0, 1.0)).price)


def test_qmc_chunk_invariance(bs_model: GBM, atm_call: EuropeanOption) -> None:
    a = price_mc(bs_model, atm_call, n_paths=4 * 1024, method="qmc", n_scrambles=4, seed=6)
    b = price_mc(
        bs_model, atm_call, n_paths=4 * 1024, method="qmc", n_scrambles=4, seed=6, chunk_size=300
    )
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_qmc_validation(bs_model: GBM, atm_call: EuropeanOption) -> None:
    with pytest.raises(ValueError, match="2\\^m"):
        price_mc(bs_model, atm_call, n_paths=1000, method="qmc")
    with pytest.raises(ValueError, match="n_scrambles"):
        price_mc(bs_model, atm_call, n_paths=1024, method="qmc", n_scrambles=1)
    assert qmc_path_count(100_000) == 16 * 4096
    assert qmc_path_count(10, 16) == 16


@pytest.mark.parametrize("strike", [140.0, 180.0])
def test_importance_sampling_deep_otm_call(bs_model: GBM, strike: float) -> None:
    product = EuropeanOption(strike, 1.0)
    ref = float(bs_price(100, strike, 1, 0.05, 0.2))
    res = price_mc(bs_model, product, n_paths=50_000, method="is", seed=7)
    plain = price_mc(bs_model, product, n_paths=50_000, method="plain", seed=7)
    _within(res.price, res.std_error, ref)
    assert res.method == "mc-is"
    assert res.diagnostics["vr_factor"] > 10.0
    assert res.std_error is not None
    assert plain.std_error is not None
    assert res.std_error < plain.std_error / 3.0


def test_importance_sampling_digital_and_explicit_shift(bs_model: GBM) -> None:
    digital = DigitalOption(150.0, 1.0)
    ref = float(bs_digital_price(100, 150, 1, 0.05, 0.2))
    res = price_mc(bs_model, digital, n_paths=50_000, method="is", seed=8)
    _within(res.price, res.std_error, ref)
    zero_shift = price_mc(bs_model, digital, n_paths=50_000, method="is", is_shift=0.0, seed=8)
    plain = price_mc(bs_model, digital, n_paths=50_000, method="plain", seed=8)
    assert zero_shift.price == pytest.approx(plain.price, rel=1e-12)  # weight 1: plain MC


def test_importance_sampling_multi_step_and_other_models(bs_model: GBM) -> None:
    res = price_mc(
        bs_model, EuropeanOption(150.0, 1.0), n_paths=50_000, n_steps=8, method="is", seed=9
    )
    _within(res.price, res.std_error, float(bs_price(100, 150, 1, 0.05, 0.2)))
    heston = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7)
    res = price_mc(
        heston, EuropeanOption(150.0, 1.0), n_paths=50_000, n_steps=16, method="is", seed=10
    )
    _within(res.price, res.std_error, gil_pelaez_price(heston, 150.0, 1.0))
    merton = Merton(100.0, 0.05, 0.2, 1.0, -0.1, 0.15)
    res = price_mc(merton, EuropeanOption(160.0, 1.0), n_paths=50_000, method="is", seed=11)
    ref = price_analytic(merton, EuropeanOption(160.0, 1.0)).price
    _within(res.price, res.std_error, ref)


def test_default_shift_and_reference_vol(bs_model: GBM) -> None:
    mu = default_is_shift(bs_model, EuropeanOption(100.0 * np.exp(0.03), 1.0))
    assert mu == pytest.approx(0.0, abs=1e-12)  # strike at the median of S_T
    assert reference_vol(bs_model) == 0.2
    with pytest.raises(ValueError, match="importance-sampling shift"):
        default_is_shift(bs_model, _Unsupported())

    class Custom(GBM):
        pass

    assert reference_vol(Custom(100.0, 0.0, 0.3)) == 0.3
    from tests.engines.test_fourier import Plain

    with pytest.raises(ValueError, match="reference volatility"):
        reference_vol(Plain())
