r"""Black-Scholes Greeks in closed form (Hull, 2018, Ch. 19).

With :math:`\varphi` the standard normal density and :math:`d_{1,2}` as in
:mod:`mcengine.engines.analytic`:

.. math::

    \Delta_C = e^{-qT}N(d_1),\quad \Delta_P = \Delta_C - e^{-qT},\quad
    \Gamma = \frac{e^{-qT}\varphi(d_1)}{S_0\sigma\sqrt T},\quad
    \mathcal V = S_0e^{-qT}\varphi(d_1)\sqrt T,

.. math::

    \Theta_C = -\frac{S_0e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} - rKe^{-rT}N(d_2)
    + qS_0e^{-qT}N(d_1),\qquad \rho_C = KTe^{-rT}N(d_2).

For the cash-or-nothing digital call (payout 1):
:math:`\Delta = e^{-rT}\varphi(d_2)/(S_0\sigma\sqrt T)`,
:math:`\Gamma = -e^{-rT}\varphi(d_2)d_1/(S_0^2\sigma^2T)`,
:math:`\mathcal V = -e^{-rT}\varphi(d_2)d_1/\sigma`.

Units: vega and rho per unit (not per 1 %) change, theta per year (calendar time
derivative :math:`\partial V/\partial t = -\partial V/\partial T`).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from mcengine._typing import FloatArray
from mcengine._validation import require_choice
from mcengine.engines.analytic import as_output, bs_d1_d2
from mcengine.products.base import OPTION_TYPES
from mcengine.results import GreeksResult


def _phi(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi), dtype=np.float64)


def bs_delta(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> float | FloatArray:
    """Black-Scholes delta (vectorised)."""
    require_choice("option_type", option_type, OPTION_TYPES)
    d1, _ = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    carry = np.exp(-np.asarray(q, dtype=np.float64) * np.asarray(maturity, dtype=np.float64))
    delta = carry * ndtr(d1) if option_type == "call" else carry * (ndtr(d1) - 1.0)
    return as_output(delta)


def bs_gamma(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> float | FloatArray:
    """Black-Scholes gamma (identical for calls and puts)."""
    d1, _ = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    s, t, vol, qq = (np.asarray(a, dtype=np.float64) for a in (s0, maturity, sigma, q))
    return as_output(np.exp(-qq * t) * _phi(d1) / (s * vol * np.sqrt(t)))


def bs_vega(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> float | FloatArray:
    """Black-Scholes vega per unit volatility (identical for calls and puts)."""
    d1, _ = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    s, t, qq = (np.asarray(a, dtype=np.float64) for a in (s0, maturity, q))
    return as_output(s * np.exp(-qq * t) * _phi(d1) * np.sqrt(t))


def bs_greeks(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> GreeksResult:
    """All five Black-Scholes Greeks of a European call or put."""
    started = time.perf_counter()
    require_choice("option_type", option_type, OPTION_TYPES)
    d1a, d2a = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    d1, d2 = float(d1a), float(d2a)
    sqrt_t = np.sqrt(maturity)
    carry, disc = np.exp(-q * maturity), np.exp(-r * maturity)
    pdf = float(_phi(np.asarray(d1)))
    gamma = carry * pdf / (s0 * sigma * sqrt_t)
    vega = s0 * carry * pdf * sqrt_t
    decay = -s0 * carry * pdf * sigma / (2.0 * sqrt_t)
    if option_type == "call":
        delta = carry * ndtr(d1)
        theta = decay - r * strike * disc * ndtr(d2) + q * s0 * carry * ndtr(d1)
        rho = strike * maturity * disc * ndtr(d2)
    else:
        delta = carry * (ndtr(d1) - 1.0)
        theta = decay + r * strike * disc * ndtr(-d2) - q * s0 * carry * ndtr(-d1)
        rho = -strike * maturity * disc * ndtr(-d2)
    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        method="bs-analytic",
        elapsed_s=time.perf_counter() - started,
    )


def digital_greeks(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    payout: float = 1.0,
) -> GreeksResult:
    """Delta, gamma and vega of a cash-or-nothing digital option."""
    started = time.perf_counter()
    require_choice("option_type", option_type, OPTION_TYPES)
    d1a, d2a = bs_d1_d2(s0, strike, maturity, r, sigma, q)
    d1, d2 = float(d1a), float(d2a)
    sign = 1.0 if option_type == "call" else -1.0
    base = sign * payout * np.exp(-r * maturity) * float(_phi(np.asarray(d2)))
    return GreeksResult(
        delta=float(base / (s0 * sigma * np.sqrt(maturity))),
        gamma=float(-base * d1 / (s0**2 * sigma**2 * maturity)),
        vega=float(-base * d1 / sigma),
        theta=None,
        rho=None,
        method="bs-analytic",
        elapsed_s=time.perf_counter() - started,
    )
