r"""Discrete arithmetic Asian options under GBM by recursive density convolution.

There is no closed form for the arithmetic average, so the Monte Carlo estimators are
validated against this independent deterministic method (Carverhill & Clewlow, 1990;
Benhamou, 2002). Write :math:`R_k = S_{t_k}/S_{t_{k-1}}` (independent log-normal
ratios) and define backwards

.. math:: Z_m = R_m,\qquad Z_k = R_k\,(1 + Z_{k+1}),

so that :math:`\sum_{i=1}^m S_{t_i} = S_0 Z_1`. On a uniform grid in
:math:`x = \ln Z`, one step maps the density :math:`f_{k+1}` of
:math:`X_{k+1} = \ln Z_{k+1}` to the density of :math:`Y = \ln(1 + e^{X_{k+1}})`,

.. math:: g(y) = f_{k+1}\big(\ln(e^y - 1)\big)\,\frac{1}{1 - e^{-y}},\quad y > 0,

and then convolves it with the Gaussian density of :math:`\ln R_k` (FFT). The price is
:math:`e^{-rT}\int (S_0 e^x/m - K)^+ f_1(x)\,dx` (trapezoidal rule). The error is
:math:`O(\Delta x^2)` from linear interpolation and quadrature, so a Richardson
extrapolation :math:`(4P_{n} - P_{n/2})/3` over two grids (default :math:`n = 2^{15}`)
removes the leading term; the tests check agreement to better than
:math:`10^{-5}` relative across grids.

References
----------
Carverhill, A., & Clewlow, L. (1990). Flexible convolution. *Risk*, 3(4), 25-29.
Benhamou, E. (2002). Fast Fourier transform for discrete Asian options.
*Journal of Computational Finance*, 6(1), 49-68.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import fftconvolve

from mcengine._typing import FloatArray
from mcengine._validation import require_int_at_least
from mcengine.engines.analytic import deterministic_result
from mcengine.models.gbm import GBM
from mcengine.products.asian import AsianOption
from mcengine.results import PricingResult


def _gaussian_kernel(dx: float, mean: float, sd: float, max_half: int) -> FloatArray:
    half = min(max_half, int(np.ceil((abs(mean) + 12.0 * sd) / dx)) + 1)
    offsets = np.arange(-half, half + 1) * dx
    kernel = np.exp(-0.5 * ((offsets - mean) / sd) ** 2)
    return np.asarray(kernel / (kernel.sum() * dx), dtype=np.float64)


def log_sum_density(
    model: GBM, fixing_times: FloatArray, n_grid: int = 2**15
) -> tuple[FloatArray, FloatArray]:
    """Grid ``x`` and density of ``ln(sum_i S_{t_i} / S_0)`` on that grid."""
    require_int_at_least("n_grid", n_grid, 256)
    t = np.asarray(fixing_times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("fixing_times must be positive and strictly increasing")
    dts = np.diff(np.concatenate(([0.0], t)))
    nu = model.r - model.q - 0.5 * model.sigma**2
    spread = 10.0 * model.sigma * np.sqrt(t[-1]) + abs(nu) * t[-1]
    lo, hi = -spread - 1.0, np.log(t.size) + spread + 1.0
    x = np.linspace(lo, hi, n_grid)
    dx = float(x[1] - x[0])

    def kernel(k: int) -> FloatArray:
        return _gaussian_kernel(dx, nu * dts[k], model.sigma * np.sqrt(dts[k]), n_grid // 2 - 1)

    # X_m = ln R_m: Gaussian density on the grid.
    sd_m = model.sigma * np.sqrt(dts[-1])
    density = np.exp(-0.5 * ((x - nu * dts[-1]) / sd_m) ** 2) / (sd_m * np.sqrt(2.0 * np.pi))
    positive = x > 0.0
    y = x[positive]
    x_of_y = y + np.log(-np.expm1(-y))  # ln(e^y - 1), stable for small and large y
    jacobian = -1.0 / np.expm1(-y)  # dx/dy = 1 / (1 - e^{-y})
    for k in range(t.size - 2, -1, -1):
        g = np.zeros_like(x)
        g[positive] = np.interp(x_of_y, x, density, left=0.0, right=0.0) * jacobian
        density = np.maximum(fftconvolve(g, kernel(k), mode="same") * dx, 0.0)
    density /= np.trapezoid(density, x)
    return x, np.asarray(density, dtype=np.float64)


def asian_arithmetic_price(
    model: GBM, product: AsianOption, n_grid: int = 2**15, richardson: bool = True
) -> float:
    """Price of a discrete arithmetic-average Asian option by recursive convolution.

    With ``richardson=True`` the result is ``(4 P(n_grid) - P(n_grid / 2)) / 3``.
    """
    if product.average != "arithmetic":
        raise ValueError("recursive convolution prices arithmetic averages only")
    if richardson:
        fine = asian_arithmetic_price(model, product, n_grid, richardson=False)
        coarse = asian_arithmetic_price(model, product, n_grid // 2, richardson=False)
        return (4.0 * fine - coarse) / 3.0
    fixings = product.monitoring_times()
    x, density = log_sum_density(model, fixings, n_grid)
    average = model.s0 * np.exp(x) / fixings.size
    if product.option_type == "call":
        payoff = np.maximum(average - product.strike, 0.0)
    else:
        payoff = np.maximum(product.strike - average, 0.0)
    return float(np.exp(-model.r * product.maturity) * np.trapezoid(payoff * density, x))


def price_asian_convolution(model: GBM, product: AsianOption, n_grid: int = 2**15) -> PricingResult:
    """:func:`asian_arithmetic_price` wrapped in a :class:`PricingResult`."""
    started = time.perf_counter()
    price = asian_arithmetic_price(model, product, n_grid)
    return deterministic_result(price, "recursive-convolution", started)
