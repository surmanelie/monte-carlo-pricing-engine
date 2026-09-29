"""Fourier pricing: Gil-Pelaez (adaptive and Gauss-Legendre) and Carr-Madan FFT."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.engines.analytic import bs_price
from mcengine.engines.fourier import (
    carr_madan_prices,
    fourier_prices,
    gil_pelaez_price,
    price_fourier,
)
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption
from mcengine.products.european import EuropeanOption
from tests.models.test_heston import STANDARD, STRESS

STRIKES = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
GBM_Q = GBM(100.0, 0.05, 0.2, q=0.01)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_all_methods_reproduce_black_scholes(kind: str) -> None:
    ref = np.asarray(bs_price(100.0, STRIKES, 1.0, 0.05, 0.2, 0.01, kind))
    gp = np.array([gil_pelaez_price(GBM_Q, k, 1.0, kind) for k in STRIKES])
    np.testing.assert_allclose(gp, ref, atol=1e-10)
    np.testing.assert_allclose(fourier_prices(GBM_Q, STRIKES, 1.0, kind), ref, atol=1e-10)
    np.testing.assert_allclose(carr_madan_prices(GBM_Q, STRIKES, 1.0, kind), ref, atol=1e-7)


@pytest.mark.parametrize("maturity", [0.1, 1.0, 5.0])
def test_heston_methods_agree(maturity: float) -> None:
    gp = np.array([gil_pelaez_price(STANDARD, k, maturity) for k in STRIKES])
    np.testing.assert_allclose(fourier_prices(STANDARD, STRIKES, maturity), gp, atol=1e-9)
    np.testing.assert_allclose(carr_madan_prices(STANDARD, STRIKES, maturity), gp, atol=1e-7)


def test_heston_stress_case_and_damping() -> None:
    gp = gil_pelaez_price(STRESS, 100.0, 10.0)
    for alpha in (0.75, 1.0, 1.5, 2.0):
        assert carr_madan_prices(STRESS, [100.0], 10.0, alpha=alpha)[0] == pytest.approx(
            gp, abs=1e-6
        )


def test_heston_put_call_parity() -> None:
    call = gil_pelaez_price(STANDARD, 105.0, 1.0, "call")
    put = gil_pelaez_price(STANDARD, 105.0, 1.0, "put")
    assert call - put == pytest.approx(100.0 - 105.0 * np.exp(-0.03), abs=1e-10)


def test_price_fourier_wrapper() -> None:
    product = EuropeanOption(100.0, 1.0)
    gp = price_fourier(STANDARD, product)
    cm = price_fourier(STANDARD, product, "carr-madan")
    assert gp.method == "fourier-gp"
    assert cm.method == "fourier-cm"
    assert gp.price == pytest.approx(cm.price, abs=1e-7)
    with pytest.raises(ValueError, match="European"):
        price_fourier(STANDARD, AsianOption(100.0, 1.0))
    with pytest.raises(ValueError, match="method"):
        price_fourier(STANDARD, product, "cos")


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="strikes"):
        fourier_prices(STANDARD, [-1.0], 1.0)
    with pytest.raises(ValueError, match="strikes"):
        carr_madan_prices(STANDARD, [0.0], 1.0)
    with pytest.raises(ValueError, match="n_fft"):
        carr_madan_prices(STANDARD, [100.0], 1.0, n_fft=1000)
    with pytest.raises(ValueError, match="grid"):
        carr_madan_prices(STANDARD, [1e-14], 1.0)


class Plain(Model):
    """A model without a characteristic function."""

    s0, r, q = 100.0, 0.0, 0.0
    n_factors, exact_simulation, name = 1, True, "plain"

    def paths_from_normals(self, times: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(z[:, :, 0])


def test_model_without_char_func() -> None:
    assert not Plain().has_char_func
    with pytest.raises(ValueError, match="characteristic"):
        gil_pelaez_price(Plain(), 100.0, 1.0)
    with pytest.raises(NotImplementedError):
        Plain().char_func(np.array([0.0]), 1.0)


def test_truncation_is_capped() -> None:
    from mcengine.engines.fourier import _truncation

    assert _truncation(GBM(100.0, 0.0, 0.01), 0.01) == 2.0**12
    assert _truncation(STANDARD, 1.0) < 2.0**12
