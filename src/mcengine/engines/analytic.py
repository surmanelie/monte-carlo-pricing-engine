r"""Closed-form reference prices.

Black-Scholes (1973) with continuous dividend yield :math:`q`:

.. math::

    d_{1,2} = \frac{\ln(S_0/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad
    C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2),\qquad
    P = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1).

A cash-or-nothing digital call pays :math:`e^{-rT}N(d_2)` per unit payout
(Reiner & Rubinstein, 1991b).

All functions are vectorised: array inputs broadcast, scalar inputs return ``float``.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from mcengine._typing import FloatArray
from mcengine._validation import require_choice
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.products.base import OPTION_TYPES, Product
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.results import PricingResult


def as_output(x: FloatArray) -> float | FloatArray:
    """Return a Python ``float`` for 0-d arrays, otherwise the array itself."""
    arr = np.asarray(x, dtype=np.float64)
    return float(arr) if arr.ndim == 0 else arr


def _check_inputs(*arrays: FloatArray, names: tuple[str, ...]) -> None:
    for name, arr in zip(names, arrays, strict=True):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be finite and > 0")


def bs_d1_d2(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Black-Scholes :math:`d_1` and :math:`d_2`."""
    s, k, t, rr, vol, qq = (
        np.asarray(a, dtype=np.float64) for a in (s0, strike, maturity, r, sigma, q)
    )
    _check_inputs(s, k, t, vol, names=("s0", "strike", "maturity", "sigma"))
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (rr - qq + 0.5 * vol**2) * t) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def bs_price(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> float | FloatArray:
    """Black-Scholes price of a European call or put.

    Examples
    --------
    >>> round(bs_price(100, 100, 1.0, 0.05, 0.2), 4)
    10.4506
    >>> round(bs_price(100, 100, 1.0, 0.05, 0.2, option_type="put"), 4)
    5.5735
    """
    require_choice("option_type", option_type, OPTION_TYPES)
    d1, d2 = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    s, k, t, rr, qq = (np.asarray(a, dtype=np.float64) for a in (s0, strike, maturity, r, q))
    fwd_s = s * np.exp(-qq * t)
    disc_k = k * np.exp(-rr * t)
    if option_type == "call":
        price = fwd_s * ndtr(d1) - disc_k * ndtr(d2)
    else:
        price = disc_k * ndtr(-d2) - fwd_s * ndtr(-d1)
    return as_output(price)


def bs_digital_price(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    payout: float = 1.0,
) -> float | FloatArray:
    """Cash-or-nothing digital price ``payout * exp(-rT) N(+/- d2)``."""
    require_choice("option_type", option_type, OPTION_TYPES)
    _, d2 = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    disc = np.exp(-np.asarray(r, dtype=np.float64) * np.asarray(maturity, dtype=np.float64))
    prob = ndtr(d2) if option_type == "call" else ndtr(-d2)
    return as_output(payout * disc * prob)


def deterministic_result(price: float, method: str, started: float) -> PricingResult:
    """Wrap a deterministic price into a :class:`PricingResult`."""
    return PricingResult(
        price=float(price),
        std_error=None,
        ci_low=None,
        ci_high=None,
        n_paths=None,
        n_steps=None,
        method=method,
        elapsed_s=time.perf_counter() - started,
    )


def price_analytic(model: Model, product: Product) -> PricingResult:
    """Closed-form price for a supported ``(model, product)`` pair.

    Raises
    ------
    ValueError
        If no closed form is implemented for the combination.
    """
    started = time.perf_counter()
    if isinstance(model, GBM):
        m = model
        if isinstance(product, EuropeanOption):
            k, t, kind = product.strike, product.maturity, product.option_type
            price = bs_price(m.s0, k, t, m.r, m.sigma, m.q, kind)
            return deterministic_result(float(price), "bs-analytic", started)
        if isinstance(product, DigitalOption):
            k, t, kind = product.strike, product.maturity, product.option_type
            price = bs_digital_price(m.s0, k, t, m.r, m.sigma, m.q, kind, product.payout)
            return deterministic_result(float(price), "bs-analytic", started)
    raise ValueError(
        f"no closed form for model {type(model).__name__} and product {type(product).__name__}"
    )
