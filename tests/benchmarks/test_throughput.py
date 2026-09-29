"""pytest-benchmark suite (skipped by default; run with ``pytest --benchmark-only``)."""

from __future__ import annotations

from typing import Any

import pytest

from mcengine._numba import HAS_NUMBA
from mcengine.engines.lsm import price_lsm
from mcengine.engines.monte_carlo import price_mc
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.european import EuropeanOption

CALL = EuropeanOption(100.0, 1.0)

#: Excluded from the default run by ``-m "not slow"``; the benchmark script passes ``-m ""``.
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("method", ["plain", "antithetic", "cv", "is"])
def test_gbm_european(benchmark: Any, method: str) -> None:
    benchmark(price_mc, GBM(100.0, 0.05, 0.2), CALL, n_paths=200_000, method=method, seed=1)


def test_gbm_asian_qmc(benchmark: Any) -> None:
    product = AsianOption(100.0, 1.0, 12)
    benchmark(price_mc, GBM(100.0, 0.05, 0.2), product, n_paths=16 * 2**12, method="qmc", seed=1)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_heston_qe(benchmark: Any, backend: str) -> None:
    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("numba not installed")
    model = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7, backend=backend)
    price_mc(model, CALL, n_paths=64, n_steps=4, seed=0)  # compile outside the timing
    benchmark(price_mc, model, CALL, n_paths=50_000, n_steps=252, seed=1)


def test_merton(benchmark: Any) -> None:
    model = Merton(100.0, 0.05, 0.2, 1.0, -0.1, 0.15)
    benchmark(price_mc, model, CALL, n_paths=50_000, n_steps=252, seed=1)


def test_lsm(benchmark: Any) -> None:
    put = AmericanOption(40.0, 1.0, "put", 50)
    benchmark(price_lsm, GBM(36.0, 0.06, 0.2), put, n_paths=50_000, seed=1)
