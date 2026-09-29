r"""Closed-form reference prices.

Black-Scholes (1973) with continuous dividend yield :math:`q`:

.. math::

    d_{1,2} = \frac{\ln(S_0/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad
    C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2),\qquad
    P = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1).

A cash-or-nothing digital call pays :math:`e^{-rT}N(d_2)` per unit payout
(Reiner & Rubinstein, 1991b).

Discrete geometric Asian (Kemna & Vorst, 1990, discrete-monitoring version): with
fixing dates :math:`t_1<\dots<t_m`, :math:`\ln G` is normal with

.. math::

    \mu_G = \ln S_0 + (r - q - \tfrac12\sigma^2)\,\bar t,\qquad
    \sigma_G^2 = \frac{\sigma^2}{m^2}\sum_{i,j}\min(t_i, t_j),

so the call is :math:`e^{-rT}\big(e^{\mu_G + \sigma_G^2/2}N(d_1) - KN(d_2)\big)` with
:math:`d_1 = (\mu_G - \ln K + \sigma_G^2)/\sigma_G`, :math:`d_2 = d_1 - \sigma_G`.

Continuously monitored single barriers (Reiner & Rubinstein, 1991a), in the notation
of Haug (2007, Section 4.17.1) with zero rebate: the eight contracts are combinations of
the terms :math:`A, B, C, D` implemented in :func:`barrier_price`.

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
from mcengine.models.merton import Merton
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BARRIER_TYPES, BGK_BETA, BarrierOption
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


def geometric_asian_price(
    s0: float,
    strike: float,
    fixing_times: ArrayLike,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Discrete geometric-average Asian option (Kemna & Vorst, 1990).

    ``fixing_times`` are the averaging dates; the payment date is the last of them.
    """
    require_choice("option_type", option_type, OPTION_TYPES)
    t = np.asarray(fixing_times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("fixing_times must be positive and strictly increasing")
    _check_inputs(np.asarray([s0, strike, sigma]), names=("s0, strike and sigma",))
    maturity = float(t[-1])
    mu = np.log(s0) + (r - q - 0.5 * sigma**2) * float(t.mean())
    var = sigma**2 * float(np.minimum.outer(t, t).mean())
    vol = np.sqrt(var)
    d1 = (mu - np.log(strike) + var) / vol
    d2 = d1 - vol
    fwd = np.exp(mu + 0.5 * var)
    disc = np.exp(-r * maturity)
    if option_type == "call":
        return float(disc * (fwd * ndtr(d1) - strike * ndtr(d2)))
    return float(disc * (strike * ndtr(-d2) - fwd * ndtr(-d1)))


def barrier_price(
    s0: float,
    strike: float,
    barrier: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    barrier_type: str = "down-and-out",
    option_type: str = "call",
) -> float:
    """Continuously monitored single-barrier option, zero rebate (Reiner & Rubinstein, 1991).

    Raises
    ------
    ValueError
        For invalid inputs or a spot already on the knocked side of the barrier.
    """
    require_choice("barrier_type", barrier_type, BARRIER_TYPES)
    require_choice("option_type", option_type, OPTION_TYPES)
    _check_inputs(
        np.asarray([s0, strike, barrier, maturity, sigma]),
        names=("s0, strike, barrier, maturity and sigma",),
    )
    down = barrier_type.startswith("down")
    if (down and s0 <= barrier) or (not down and s0 >= barrier):
        raise ValueError(f"spot {s0:g} is not on the live side of the {barrier_type} barrier")
    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if down else -1.0
    b = r - q
    vol_t = sigma * np.sqrt(maturity)
    mu = (b - 0.5 * sigma**2) / sigma**2
    shift = (1.0 + mu) * vol_t
    x1 = np.log(s0 / strike) / vol_t + shift
    x2 = np.log(s0 / barrier) / vol_t + shift
    y1 = np.log(barrier**2 / (s0 * strike)) / vol_t + shift
    y2 = np.log(barrier / s0) / vol_t + shift
    carry = s0 * np.exp((b - r) * maturity)
    disc_k = strike * np.exp(-r * maturity)
    up_pow = (barrier / s0) ** (2.0 * (mu + 1.0))
    k_pow = (barrier / s0) ** (2.0 * mu)

    def vanilla_term(x: float) -> float:
        return float(phi * carry * ndtr(phi * x) - phi * disc_k * ndtr(phi * x - phi * vol_t))

    def reflected_term(y: float) -> float:
        return float(
            phi * carry * up_pow * ndtr(eta * y)
            - phi * disc_k * k_pow * ndtr(eta * y - eta * vol_t)
        )

    a_t, b_t = vanilla_term(x1), vanilla_term(x2)
    c_t, d_t = reflected_term(y1), reflected_term(y2)
    above = strike >= barrier
    table = {
        ("down-and-in", "call"): c_t if above else a_t - b_t + d_t,
        ("up-and-in", "call"): a_t if above else b_t - c_t + d_t,
        ("down-and-in", "put"): b_t - c_t + d_t if above else a_t,
        ("up-and-in", "put"): a_t - b_t + d_t if above else c_t,
        ("down-and-out", "call"): a_t - c_t if above else b_t - d_t,
        ("up-and-out", "call"): 0.0 if above else a_t - b_t + c_t - d_t,
        ("down-and-out", "put"): a_t - b_t + c_t - d_t if above else 0.0,
        ("up-and-out", "put"): b_t - d_t if above else a_t - c_t,
    }
    return max(table[(barrier_type, option_type)], 0.0)


def barrier_price_discrete_bgk(
    s0: float,
    strike: float,
    barrier: float,
    maturity: float,
    r: float,
    sigma: float,
    n_monitoring: int,
    q: float = 0.0,
    barrier_type: str = "down-and-out",
    option_type: str = "call",
) -> float:
    """Broadie-Glasserman-Kou (1997) approximation of a *discretely* monitored barrier.

    The continuous formula is evaluated with the barrier shifted away from the spot by
    ``exp(beta sigma sqrt(T / m))``.
    """
    shift = BGK_BETA * sigma * np.sqrt(maturity / n_monitoring)
    shifted = float(barrier * np.exp(-shift if barrier_type.startswith("down") else shift))
    return barrier_price(s0, strike, shifted, maturity, r, sigma, q, barrier_type, option_type)


def merton_price(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta_j: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-12,
) -> float:
    r"""Merton (1976) jump-diffusion price as a Poisson mixture of Black-Scholes prices.

    .. math::

        V = \sum_{n\ge0} e^{-\lambda'T}\frac{(\lambda'T)^n}{n!}\,
        \mathrm{BS}\big(S_0, K, T, r_n, \sigma_n, q\big),\qquad
        \lambda' = \lambda(1 + \bar k),\quad
        \sigma_n^2 = \sigma^2 + \frac{n\delta^2}{T},\quad
        r_n = r - \lambda\bar k + \frac{n\ln(1 + \bar k)}{T}.

    The series is truncated once the remaining Poisson mass is below ``tol``; since every
    term is bounded by ``S0`` (calls) or ``K`` (puts), the truncation error is at most
    ``tol * max(S0, K)``.
    """
    require_choice("option_type", option_type, OPTION_TYPES)
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must lie in (0, 1), got {tol}")
    _check_inputs(np.asarray([s0, strike, maturity, sigma]), names=("s0, strike, maturity, sigma",))
    if lam < 0.0 or delta_j < 0.0:
        raise ValueError("lam and delta_j must be >= 0")
    kbar = np.exp(mu_j + 0.5 * delta_j**2) - 1.0
    lam_t = lam * (1.0 + kbar) * maturity
    total, mass, n = 0.0, 0.0, 0
    weight = np.exp(-lam_t)
    while True:
        sigma_n = np.sqrt(sigma**2 + n * delta_j**2 / maturity)
        r_n = r - lam * kbar + n * np.log1p(kbar) / maturity
        total += weight * float(bs_price(s0, strike, maturity, r_n, sigma_n, q, option_type))
        mass += weight
        if 1.0 - mass < tol or lam_t == 0.0:
            break
        n += 1
        weight *= lam_t / n
        if n > 10_000:  # pragma: no cover - guards against pathological inputs
            raise ValueError("Merton series did not converge")
    return float(total)


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
        if isinstance(product, AsianOption) and product.average == "geometric":
            times = product.monitoring_times()
            kind = product.option_type
            price = geometric_asian_price(m.s0, product.strike, times, m.r, m.sigma, m.q, kind)
            return deterministic_result(price, "kemna-vorst", started)
        if isinstance(product, BarrierOption):
            price = barrier_price(
                m.s0,
                product.strike,
                product.barrier,
                product.maturity,
                m.r,
                m.sigma,
                m.q,
                product.barrier_type,
                product.option_type,
            )
            return deterministic_result(price, "reiner-rubinstein", started)
    if isinstance(model, Merton) and isinstance(product, EuropeanOption):
        m2 = model
        price = merton_price(
            m2.s0,
            product.strike,
            product.maturity,
            m2.r,
            m2.sigma,
            m2.lam,
            m2.mu_j,
            m2.delta_j,
            m2.q,
            product.option_type,
        )
        return deterministic_result(price, "merton-series", started)
    raise ValueError(
        f"no closed form for model {type(model).__name__} and product {type(product).__name__}"
    )
