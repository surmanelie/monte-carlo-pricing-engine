"""Numba kernels reproduce the NumPy implementation under the same random numbers.

The kernels are always tested through their pure-Python source (``py_func``), which also
counts for coverage; when Numba is installed the compiled versions are compared too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcengine._numba import HAS_NUMBA, heston_qe_kernel, lsm_policy_kernel, require_backend
from mcengine.engines.lsm import fit_lsm, price_lsm
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.products.american import AmericanOption

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
HESTONS = [
    Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7),
    Heston(100.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9),  # exponential branch often active
    Heston(100.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9, martingale_correction=False),
]


def _py(kernel: object) -> object:
    return getattr(kernel, "py_func", kernel)


@pytest.mark.parametrize("model", HESTONS)
def test_qe_kernel_matches_numpy(model: Heston) -> None:
    times = np.linspace(0.0, 2.0, 9)
    z = np.random.default_rng(0).standard_normal((300, 8, 2))
    s_ref, v_ref = model.simulate_with_variance(times, z)
    log_s, var = _py(heston_qe_kernel)(  # type: ignore[operator]
        math.log(model.s0), model.v0, np.diff(times), z[:, :, 0].copy(), z[:, :, 1].copy(),
        model.kappa, model.theta, model.xi, model.rho, model.r - model.q, model.psi_c,
        model.martingale_correction,
    )  # fmt: skip
    np.testing.assert_allclose(np.exp(log_s), s_ref, rtol=1e-12)
    np.testing.assert_allclose(var, v_ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("basis", ["laguerre", "monomial"])
def test_lsm_kernel_matches_numpy(basis: str) -> None:
    model = GBM(36.0, 0.06, 0.2)
    product = AmericanOption(40.0, 1.0, "put", 10)
    fit = fit_lsm(model, product, n_paths=5000, basis=basis, degree=3, seed=1)
    times = np.linspace(0.0, 1.0, 11)
    paths = model.simulate_paths(times, 2000, seed=2)
    spots = paths[:, 1:]
    kernel_value = _py(lsm_policy_kernel)(  # type: ignore[operator]
        spots, product.exercise_value(spots), np.exp(-0.06 * fit.exercise_times),
        fit.coefficients, fit.fitted, fit.strike, basis == "laguerre",
    )  # fmt: skip
    from mcengine.engines.lsm import _apply_policy

    ref = _apply_policy(model, product, fit, paths, np.arange(1, 11))
    np.testing.assert_allclose(kernel_value, ref, rtol=1e-12)


@needs_numba
@pytest.mark.parametrize("model", HESTONS[:2])
def test_compiled_heston_backend_is_identical(model: Heston) -> None:
    fast = Heston(
        model.s0, model.r, model.v0, model.kappa, model.theta, model.xi, model.rho,
        backend="numba",
    )  # fmt: skip
    times = np.linspace(0.0, 1.0, 13)
    z = np.random.default_rng(3).standard_normal((1000, 12, 2))
    np.testing.assert_allclose(
        fast.paths_from_normals(times, z), model.paths_from_normals(times, z), rtol=1e-12
    )
    assert fast.with_scheme("euler").backend == "numpy"


@needs_numba
def test_compiled_lsm_backend_is_identical() -> None:
    model = GBM(36.0, 0.06, 0.2)
    product = AmericanOption(40.0, 1.0, "put", 50)
    a = price_lsm(model, product, n_paths=20_000, seed=4)
    b = price_lsm(model, product, n_paths=20_000, seed=4, backend="numba")
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_backend_validation() -> None:
    with pytest.raises(ValueError, match="backend"):
        require_backend("cuda")
    with pytest.raises(ValueError, match=r"QE scheme|\[fast\]"):
        Heston(100.0, 0.0, 0.04, 1.0, 0.04, 0.5, -0.5, scheme="euler", backend="numba")
    assert require_backend("numpy") == "numpy"
