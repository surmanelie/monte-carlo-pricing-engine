"""Heston model: QE and Euler schemes, characteristic function, parameter checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcengine.engines.analytic import bs_price
from mcengine.engines.fourier import gil_pelaez_price
from mcengine.engines.monte_carlo import price_mc
from mcengine.models.heston import Heston
from mcengine.products.european import EuropeanOption
from tests.conftest import TOL_SE

STANDARD = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7)
#: Feller-violating stress case (2 kappa theta / xi^2 = 0.04).
STRESS = Heston(100.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9)


@pytest.mark.parametrize("scheme", ["qe", "euler"])
def test_variance_mean_reversion(scheme: str) -> None:
    """E[V_t] = theta + (v0 - theta) e^{-kappa t} (exact for QE, which matches moments)."""
    model = Heston(100.0, 0.03, 0.09, 1.5, 0.04, 0.4, -0.5, scheme=scheme)
    times = np.linspace(0.0, 2.0, 81)
    z = np.random.default_rng(0).standard_normal((100_000, 80, 2))
    _, var = model.simulate_with_variance(times, z)
    expected = 0.04 + 0.05 * math.exp(-3.0)
    se = var[:, -1].std() / math.sqrt(var.shape[0])
    assert abs(var[:, -1].mean() - expected) < TOL_SE * se
    if scheme == "qe":
        assert np.all(var >= 0.0)


def test_qe_discounted_price_is_a_martingale() -> None:
    times = np.linspace(0.0, 1.0, 21)
    s_t = STANDARD.simulate_paths(times, 200_000, seed=1)[:, -1]
    se = s_t.std() / math.sqrt(s_t.size)
    assert abs(s_t.mean() - STANDARD.forward(1.0)) < TOL_SE * se


@pytest.mark.parametrize(("strike", "n_steps"), [(90.0, 20), (100.0, 20), (110.0, 50)])
def test_qe_prices_match_fourier(strike: float, n_steps: int) -> None:
    ref = gil_pelaez_price(STANDARD, strike, 1.0)
    res = price_mc(STANDARD, EuropeanOption(strike, 1.0), n_paths=100_000, n_steps=n_steps, seed=5)
    assert res.std_error is not None
    assert abs(res.price - ref) < TOL_SE * res.std_error


def test_qe_beats_euler_in_the_stress_case() -> None:
    ref = gil_pelaez_price(STRESS, 100.0, 10.0)
    product = EuropeanOption(100.0, 10.0)
    qe = price_mc(STRESS, product, n_paths=50_000, n_steps=40, seed=2)
    euler = price_mc(STRESS.with_scheme("euler"), product, n_paths=50_000, n_steps=40, seed=2)
    assert qe.std_error is not None
    assert euler.std_error is not None
    assert abs(qe.price - ref) < TOL_SE * qe.std_error
    assert abs(euler.price - ref) > 10.0 * euler.std_error  # large discretisation bias


def test_martingale_correction_flag_and_exponential_branch() -> None:
    """Large xi forces the exponential branch; without the correction prices still run."""
    model = Heston(100.0, 0.0, 0.01, 0.5, 0.04, 1.5, -0.9, martingale_correction=False)
    paths = model.simulate_paths(np.linspace(0.0, 1.0, 5), 1000, seed=3)
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0.0)


def test_char_func_properties() -> None:
    t = 1.5
    assert STANDARD.char_func(np.array([0.0]), t)[0] == pytest.approx(1.0)
    assert STANDARD.char_func(np.array([-1j]), t)[0].real == pytest.approx(STANDARD.forward(t))
    assert STANDARD.has_char_func


def test_small_vol_of_vol_asymptotics() -> None:
    """With v0 = theta and xi -> 0, Heston -> GBM(sqrt(theta)); the gap is O(xi rho) + O(xi^2)."""
    ref = float(bs_price(100.0, 105.0, 1.0, 0.03, 0.2))

    def gap(xi: float, rho: float) -> float:
        return gil_pelaez_price(Heston(100.0, 0.03, 0.04, 2.0, 0.04, xi, rho), 105.0, 1.0) - ref

    assert abs(gap(1e-3, 0.0)) < 1e-5
    assert gap(1e-2, 0.0) / gap(1e-3, 0.0) == pytest.approx(100.0, rel=0.05)  # quadratic
    assert gap(1e-2, -0.5) / gap(1e-3, -0.5) == pytest.approx(10.0, rel=0.05)  # linear


def test_feller_and_scheme_switch() -> None:
    assert STANDARD.feller_ratio == pytest.approx(0.64)
    euler = STANDARD.with_scheme("euler")
    assert euler.scheme == "euler"
    assert euler.kappa == STANDARD.kappa


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v0": -0.01},
        {"kappa": 0.0},
        {"theta": -1.0},
        {"xi": 0.0},
        {"rho": -1.5},
        {"scheme": "milstein"},
        {"psi_c": 3.0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "s0": 100.0,
        "r": 0.03,
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.5,
        "rho": -0.7,
    }
    with pytest.raises(ValueError, match=r"must|unsupported"):
        Heston(**(base | kwargs))  # type: ignore[arg-type]


def test_requires_explicit_steps() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        price_mc(STANDARD, EuropeanOption(100.0, 1.0), n_paths=100)
