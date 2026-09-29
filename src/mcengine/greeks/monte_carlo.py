r"""Monte Carlo Greeks under GBM: bump-and-revalue, pathwise and likelihood-ratio estimators.

Write :math:`S_T = S_0\exp\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt T Z\big)` and
:math:`Y = e^{-rT}f(S_T)`. Every estimator is an average of i.i.d. per-path quantities,
so each Greek comes with a standard error (Glasserman, 2003, Ch. 7).

**Bump and revalue with common random numbers** (``method="bump"``): central differences
:math:`(Y(S_0 + h) - Y(S_0 - h))/2h`, :math:`(Y(S_0 + h) - 2Y(S_0) + Y(S_0 - h))/h^2`
and :math:`(Y(\sigma + h_\sigma) - Y(\sigma - h_\sigma))/2h_\sigma`, all evaluated on the
*same* normals :math:`Z`; bias :math:`O(h^2)`. For discontinuous payoffs the gamma
estimator's variance grows like :math:`h^{-3}`.

**Pathwise derivative** (``method="pathwise"``): differentiate :math:`Y` along the path,

.. math::

    \Delta^{PW} = e^{-rT}f'(S_T)\frac{S_T}{S_0},\qquad
    \mathcal V^{PW} = e^{-rT}f'(S_T)\,\frac{S_T}{\sigma}
    \Big(\ln\frac{S_T}{S_0} - (r - q + \tfrac12\sigma^2)T\Big).

It requires a payoff that is Lipschitz in :math:`S_T`. For vanillas :math:`f'` jumps at
the strike, so gamma uses the mixed pathwise-likelihood-ratio estimator
:math:`\Gamma = \pm e^{-rT}K\mathbf 1\{\cdot\}Z/(S_0^2\sigma\sqrt T)` (Glasserman 2003,
Section 7.3.3). For the digital, :math:`f' = 0` almost everywhere: the pathwise delta,
gamma and vega are identically zero - **the method fails** because the derivative of the
expectation is carried by the discontinuity, which pathwise differentiation cannot see.

**Likelihood ratio** (``method="lr"``): differentiate the density of :math:`S_T` instead
of the payoff, :math:`\partial_\theta E[Y] = E[Y\,\partial_\theta\ln p_\theta(S_T)]`, with

.. math::

    w_\Delta = \frac{Z}{S_0\sigma\sqrt T},\quad
    w_\Gamma = \frac{Z^2 - 1}{S_0^2\sigma^2T} - \frac{Z}{S_0^2\sigma\sqrt T},\quad
    w_{\mathcal V} = \frac{Z^2 - 1}{\sigma} - Z\sqrt T.

It needs no smoothness of the payoff, so it works for the digital, at the price of a
higher variance for smooth payoffs.

References
----------
Broadie, M., & Glasserman, P. (1996). Estimating security price derivatives using
simulation. *Management Science*, 42(2), 269-285.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Chapter 7.
"""

from __future__ import annotations

import math
import time

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import require_choice, require_int_at_least, require_positive
from mcengine.models.gbm import GBM
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.random.generators import make_rng
from mcengine.results import GreeksResult

GREEK_METHODS: tuple[str, ...] = ("bump", "pathwise", "lr")
GreekProduct = EuropeanOption | DigitalOption


def _terminal(model: GBM, maturity: float, z: FloatArray, s0: float, sigma: float) -> FloatArray:
    drift = (model.r - model.q - 0.5 * sigma**2) * maturity
    return np.asarray(s0 * np.exp(drift + sigma * math.sqrt(maturity) * z), dtype=np.float64)


def _payoff(product: GreekProduct, s_t: FloatArray) -> FloatArray:
    times = np.array([0.0, product.maturity])
    paths = np.column_stack((np.full_like(s_t, np.nan), s_t))
    return product.payoff(paths, times)


def greek_samples(
    model: GBM,
    product: GreekProduct,
    z: FloatArray,
    method: str,
    bump_spot: float = 0.01,
    bump_vol: float = 0.01,
) -> dict[str, FloatArray]:
    """Per-path samples of delta, gamma and vega for standard normals ``z``.

    ``bump_spot`` and ``bump_vol`` are relative bump sizes (``h = bump * S0``,
    ``h_sigma = bump * sigma``) for ``method="bump"``.
    """
    require_choice("method", method, GREEK_METHODS)
    if not isinstance(product, EuropeanOption | DigitalOption):
        raise ValueError("Monte Carlo Greeks support European and digital options")
    s0, sigma, t = model.s0, model.sigma, product.maturity
    disc = math.exp(-model.r * t)
    s_t = _terminal(model, t, z, s0, sigma)
    if method == "bump":
        require_positive("bump_spot", bump_spot)
        require_positive("bump_vol", bump_vol)
        h, hv = bump_spot * s0, bump_vol * sigma
        y0 = disc * _payoff(product, s_t)
        up = disc * _payoff(product, _terminal(model, t, z, s0 + h, sigma))
        dn = disc * _payoff(product, _terminal(model, t, z, s0 - h, sigma))
        v_up = disc * _payoff(product, _terminal(model, t, z, s0, sigma + hv))
        v_dn = disc * _payoff(product, _terminal(model, t, z, s0, sigma - hv))
        return {
            "delta": (up - dn) / (2.0 * h),
            "gamma": (up - 2.0 * y0 + dn) / h**2,
            "vega": (v_up - v_dn) / (2.0 * hv),
        }
    if method == "pathwise":
        if isinstance(product, DigitalOption):
            zeros = np.zeros_like(s_t)
            return {"delta": zeros, "gamma": zeros.copy(), "vega": zeros.copy()}
        sign = 1.0 if product.option_type == "call" else -1.0
        itm = (s_t > product.strike) if sign > 0 else (s_t < product.strike)
        slope = sign * disc * itm  # e^{-rT} f'(S_T)
        log_ret = np.log(s_t / s0) - (model.r - model.q + 0.5 * sigma**2) * t
        return {
            "delta": slope * s_t / s0,
            "gamma": sign * disc * itm * product.strike * z / (s0**2 * sigma * math.sqrt(t)),
            "vega": slope * s_t * log_ret / sigma,
        }
    y = disc * _payoff(product, s_t)
    sqrt_t = math.sqrt(t)
    return {
        "delta": y * z / (s0 * sigma * sqrt_t),
        "gamma": y * ((z**2 - 1.0) / (s0**2 * sigma**2 * t) - z / (s0**2 * sigma * sqrt_t)),
        "vega": y * ((z**2 - 1.0) / sigma - z * sqrt_t),
    }


def mc_greeks(
    model: GBM,
    product: GreekProduct,
    *,
    method: str = "lr",
    n_paths: int = 100_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    bump_spot: float = 0.01,
    bump_vol: float = 0.01,
) -> GreeksResult:
    """Delta, gamma and vega by ``bump``, ``pathwise`` or likelihood-ratio (``lr``) estimators.

    Standard errors are the per-path sample standard deviations over ``sqrt(n_paths)``.
    """
    started = time.perf_counter()
    if not isinstance(model, GBM):
        raise ValueError("Monte Carlo Greeks are implemented for the GBM model")
    require_int_at_least("n_paths", n_paths, 2)
    z = make_rng(seed, rng).standard_normal(n_paths)
    samples = greek_samples(model, product, z, method, bump_spot, bump_vol)
    means = {k: float(v.mean()) for k, v in samples.items()}
    ses = {k: float(v.std(ddof=1) / math.sqrt(n_paths)) for k, v in samples.items()}
    return GreeksResult(
        delta=means["delta"],
        gamma=means["gamma"],
        vega=means["vega"],
        theta=None,
        rho=None,
        method=f"mc-{method}",
        elapsed_s=time.perf_counter() - started,
        std_errors=ses,
        n_paths=n_paths,
    )
