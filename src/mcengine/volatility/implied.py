r"""Black-Scholes implied volatility, vectorised over strikes and maturities.

For each price the solver:

1. checks the no-arbitrage bounds
   :math:`\max(S_0e^{-qT} - Ke^{-rT}, 0) < C < S_0e^{-qT}` (calls) or
   :math:`\max(Ke^{-rT} - S_0e^{-qT}, 0) < P < Ke^{-rT}` (puts); prices outside return
   ``nan`` (or raise with ``strict=True``);
2. runs Newton's method
   :math:`\sigma \leftarrow \sigma - (\mathrm{BS}(\sigma) - V)/\mathcal V(\sigma)`,
   vectorised over all entries, from the Manaster & Koehler (1982) initial guess
   (Brenner & Subrahmanyam's ATM approximation at the money);
3. solves every entry where Newton did not converge (tiny vega far from the money) with
   Brent's method on :math:`[10^{-6}, 10]`.

References
----------
Manaster, S., & Koehler, G. (1982). The calculation of implied variances from the
Black-Scholes model: a note. *Journal of Finance*, 37(1), 227-230.
Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*, Ch. 4.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from mcengine._typing import FloatArray
from mcengine._validation import require_choice
from mcengine.engines.analytic import as_output, bs_price
from mcengine.greeks.analytic import bs_vega
from mcengine.products.base import OPTION_TYPES

_SIGMA_MIN, _SIGMA_MAX = 1e-6, 10.0


def price_bounds(
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> tuple[FloatArray, FloatArray]:
    """Model-free lower and upper bounds of a European option price."""
    s, k, t, rr, qq = (np.asarray(a, dtype=np.float64) for a in (s0, strike, maturity, r, q))
    fwd_s, disc_k = s * np.exp(-qq * t), k * np.exp(-rr * t)
    if option_type == "call":
        return np.maximum(fwd_s - disc_k, 0.0), np.asarray(fwd_s + 0.0 * disc_k)
    return np.maximum(disc_k - fwd_s, 0.0), np.asarray(disc_k + 0.0 * fwd_s)


def implied_vol(
    price: ArrayLike,
    s0: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    tol: float = 1e-12,
    max_iter: int = 60,
    strict: bool = False,
) -> float | FloatArray:
    """Implied volatility of European option prices (vectorised).

    Returns ``nan`` for prices violating the no-arbitrage bounds unless ``strict=True``,
    in which case a :class:`ValueError` is raised. Convergence is declared when the
    price error is below ``tol * max(1, price)``; the implied volatility is then accurate
    to about that error divided by vega, so far from the money (tiny vega) it is only
    as precise as the input price allows.

    Examples
    --------
    >>> round(implied_vol(10.450583572185565, 100, 100, 1.0, 0.05), 10)
    0.2
    """
    require_choice("option_type", option_type, OPTION_TYPES)
    arrays = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (price, s0, strike, maturity, r, q))
    )
    v, s, k, t, rr, qq = (a.ravel().copy() for a in arrays)
    shape = arrays[0].shape
    if np.any(s <= 0.0) or np.any(k <= 0.0) or np.any(t <= 0.0):
        raise ValueError("s0, strike and maturity must be > 0")
    low, high = price_bounds(s, k, t, rr, qq, option_type)
    feasible = (v > low + 1e-14 * s) & (v < high)
    if strict and not np.all(feasible):
        raise ValueError("price outside the no-arbitrage bounds")
    sigma = np.full(v.shape, np.nan)
    idx = np.flatnonzero(feasible)
    if idx.size:
        sigma[idx] = _solve(
            v[idx], s[idx], k[idx], t[idx], rr[idx], qq[idx], option_type, tol, max_iter
        )
    return as_output(sigma.reshape(shape))


def _solve(
    v: FloatArray,
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    option_type: str,
    tol: float,
    max_iter: int,
) -> FloatArray:
    # Manaster-Koehler guess: the vol at which the inflection point sits at the strike.
    log_moneyness = np.log(s / k) + (r - q) * t
    sigma = np.clip(np.sqrt(2.0 * np.abs(log_moneyness) / t), 0.05, 2.0)
    sigma = np.where(np.abs(log_moneyness) < 1e-8, np.sqrt(2.0 * np.pi / t) * v / s, sigma)
    sigma = np.clip(sigma, 0.01, 3.0)
    converged = np.zeros(v.shape, dtype=bool)
    for _ in range(max_iter):
        active = ~converged
        if not np.any(active):
            break
        a = np.flatnonzero(active)
        diff = np.asarray(bs_price(s[a], k[a], t[a], r[a], sigma[a], q[a], option_type)) - v[a]
        vega = np.asarray(bs_vega(s[a], k[a], t[a], r[a], sigma[a], q[a]))
        done = np.abs(diff) < tol * np.maximum(1.0, v[a])
        converged[a[done]] = True
        step = np.where(vega > 1e-12 * s[a], diff / np.maximum(vega, 1e-300), np.inf)
        new = sigma[a] - step
        bad = ~np.isfinite(new) | (new <= _SIGMA_MIN) | (new >= _SIGMA_MAX)
        sigma[a] = np.where(done, sigma[a], np.where(bad, np.nan, new))
        converged[a[bad & ~done]] = True  # hand over to Brent below
    failed = np.flatnonzero(~np.isfinite(sigma) | ~converged)
    for i in failed:
        sigma[i] = _brent(v[i], s[i], k[i], t[i], r[i], q[i], option_type, tol)
    return sigma


def _brent(
    v: float, s: float, k: float, t: float, r: float, q: float, option_type: str, tol: float
) -> float:
    def f(sig: float) -> float:
        return float(bs_price(s, k, t, r, sig, q, option_type)) - v

    lo, hi = f(_SIGMA_MIN), f(_SIGMA_MAX)
    if lo > 0.0 or hi < 0.0:
        return math.nan
    return float(
        brentq(f, _SIGMA_MIN, _SIGMA_MAX, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
    )
