r"""Cox-Ross-Rubinstein binomial tree for European, American and Bermudan options.

With :math:`\Delta t = T/N`, :math:`u = e^{\sigma\sqrt{\Delta t}}`, :math:`d = 1/u` and
risk-neutral probability :math:`p = (e^{(r-q)\Delta t} - d)/(u - d)`, values are rolled
back as

.. math:: V_{i,j} = \max\Big(h(S_{i,j}),\ e^{-r\Delta t}\big(pV_{i+1,j+1} + (1-p)V_{i+1,j}\big)\Big)

at exercise dates (only the continuation value otherwise). The backward loop runs over
time steps and is vectorised over nodes.

Accuracy options (Broadie & Detemple, 1996):

* ``smoothing=True`` (BBS) replaces the continuation value at the last step before
  maturity by the Black-Scholes price, which removes the odd/even oscillation;
* ``richardson=True`` (BBSR) returns :math:`2V(N) - V(N/2)`, cancelling the leading
  :math:`O(1/N)` error term.

References
----------
Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: a simplified approach.
*Journal of Financial Economics*, 7(3), 229-263.
Broadie, M., & Detemple, J. (1996). American option valuation: new bounds,
approximations, and a comparison of existing methods. *Review of Financial Studies*,
9(4), 1211-1250.
"""

from __future__ import annotations

import math
import time

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import (
    require_choice,
    require_int_at_least,
    require_non_negative,
    require_positive,
)
from mcengine.engines.analytic import bs_price, deterministic_result
from mcengine.models.gbm import GBM
from mcengine.products.american import AmericanOption
from mcengine.products.base import OPTION_TYPES, Product, vanilla_payoff
from mcengine.products.european import EuropeanOption
from mcengine.results import PricingResult

EXERCISE_STYLES: tuple[str, ...] = ("european", "american", "bermudan")


def _exercise_steps(
    n_steps: int, maturity: float, style: str, exercise_times: FloatArray | None
) -> np.ndarray:
    """Boolean mask over steps ``0..n_steps`` where early exercise is allowed."""
    allowed = np.zeros(n_steps + 1, dtype=bool)
    if style == "american":
        allowed[:] = True
    elif style == "bermudan":
        if exercise_times is None:
            raise ValueError("bermudan exercise requires exercise_times")
        steps = np.asarray(exercise_times, dtype=np.float64) / maturity * n_steps
        idx = np.rint(steps).astype(int)
        if np.any(np.abs(steps - idx) > 1e-8) or np.any(idx < 1) or np.any(idx > n_steps):
            raise ValueError("n_steps must place every exercise date on the tree grid")
        allowed[idx] = True
    return allowed


def crr_price(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    n_steps: int = 1000,
    style: str = "american",
    exercise_times: FloatArray | None = None,
    smoothing: bool = False,
    richardson: bool = False,
) -> float:
    """CRR binomial price.

    Parameters
    ----------
    s0, strike, maturity, r, sigma, q
        Market and contract data (``sigma > 0``, ``T > 0``).
    option_type
        ``"call"`` or ``"put"``.
    n_steps
        Number of time steps ``N`` (even when ``richardson=True``).
    style
        ``"european"``, ``"american"`` (exercise at every node) or ``"bermudan"``
        (exercise only at ``exercise_times``, which must fall on the grid).
    smoothing
        Black-Scholes value at the penultimate step (BBS).
    richardson
        Two-point Richardson extrapolation ``2 V(N) - V(N/2)`` (BBSR with smoothing).
    """
    require_positive("s0", s0)
    require_positive("strike", strike)
    require_positive("maturity", maturity)
    require_positive("sigma", sigma)
    require_non_negative("q", q)
    require_choice("option_type", option_type, OPTION_TYPES)
    require_choice("style", style, EXERCISE_STYLES)
    require_int_at_least("n_steps", n_steps, 2 if richardson else 1)
    if richardson:
        if n_steps % 2:
            raise ValueError("richardson extrapolation needs an even n_steps")
        args = (s0, strike, maturity, r, sigma, q, option_type)
        fine = crr_price(*args, n_steps, style, exercise_times, smoothing)
        coarse = crr_price(*args, n_steps // 2, style, exercise_times, smoothing)
        return 2.0 * fine - coarse
    values, _ = _roll_back(
        s0, strike, maturity, r, sigma, q, option_type, n_steps, style, exercise_times, smoothing
    )
    return values


def _roll_back(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    n_steps: int,
    style: str,
    exercise_times: FloatArray | None,
    smoothing: bool,
    track_boundary: bool = False,
) -> tuple[float, FloatArray]:
    dt = maturity / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1); increase n_steps")
    disc = math.exp(-r * dt)
    allowed = _exercise_steps(n_steps, maturity, style, exercise_times)
    log_u = math.log(u)
    boundary = np.full(n_steps + 1, np.nan)

    def spots(i: int) -> FloatArray:
        return np.asarray(s0 * np.exp((2.0 * np.arange(i + 1) - i) * log_u), dtype=np.float64)

    if smoothing and n_steps >= 1:
        s_last = spots(n_steps - 1)
        values = np.asarray(
            bs_price(s_last, strike, dt, r, sigma, q, option_type), dtype=np.float64
        ).reshape(-1)
        start = n_steps - 1
    else:
        values = vanilla_payoff(spots(n_steps), strike, option_type)
        start = n_steps
    if start == n_steps - 1 and allowed[start]:
        values = np.maximum(values, vanilla_payoff(spots(start), strike, option_type))
    for i in range(start - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if allowed[i]:
            s_i = spots(i)
            exercise = vanilla_payoff(s_i, strike, option_type)
            if track_boundary:
                stop = (exercise >= values) & (exercise > 0.0)
                if np.any(stop):
                    boundary[i] = s_i[stop].max() if option_type == "put" else s_i[stop].min()
            values = np.maximum(values, exercise)
    boundary[n_steps] = strike  # at expiry exercise is optimal whenever in the money
    return float(values[0]), boundary


def crr_exercise_boundary(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    n_steps: int = 2000,
    style: str = "american",
    exercise_times: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Early-exercise boundary from the CRR tree (American, or Bermudan on given dates).

    Returns ``(times, critical_prices)``; for a put the critical price at step ``i`` is the
    highest node where immediate exercise is optimal. It is ``nan`` at steps where
    exercise is not allowed and at early steps whose nodes do not reach the exercise
    region.
    """
    require_choice("style", style, ("american", "bermudan"))
    _, boundary = _roll_back(
        s0, strike, maturity, r, sigma, q, option_type, n_steps, style, exercise_times, False, True
    )
    return np.linspace(0.0, maturity, n_steps + 1), boundary


def price_tree(
    model: GBM,
    product: Product,
    n_steps: int = 5000,
    smoothing: bool = True,
    richardson: bool = True,
) -> PricingResult:
    """CRR price of a European option or of a Bermudan :class:`AmericanOption`.

    The American product's exercise dates are enforced exactly (Bermudan tree); ``n_steps``
    must therefore be a multiple of ``product.n_exercise`` (and of ``2 n_exercise`` with
    Richardson extrapolation).
    """
    started = time.perf_counter()
    if isinstance(product, AmericanOption):
        style, ex_times, kind = "bermudan", product.monitoring_times(), product.option_type
    elif isinstance(product, EuropeanOption):
        style, ex_times, kind = "european", None, product.option_type
    else:
        raise ValueError(f"the tree engine does not price {type(product).__name__}")
    price = crr_price(
        model.s0,
        product.strike,
        product.maturity,
        model.r,
        model.sigma,
        model.q,
        kind,
        n_steps,
        style,
        ex_times,
        smoothing,
        richardson,
    )
    method = "crr-bbsr" if smoothing and richardson else "crr"
    result = deterministic_result(price, method, started)
    return PricingResult(
        price=result.price,
        std_error=None,
        ci_low=None,
        ci_high=None,
        n_paths=None,
        n_steps=n_steps,
        method=method,
        elapsed_s=result.elapsed_s,
    )
