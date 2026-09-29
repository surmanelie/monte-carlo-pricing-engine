r"""Fourier pricing of European options from the characteristic function of ``ln S_T``.

**Gil-Pelaez inversion.** With :math:`k = \ln K` and :math:`\varphi` the characteristic
function of :math:`\ln S_T`, the two probabilities of the Heston (1993) formula combine
into one integral (using :math:`\varphi(-i) = S_0e^{(r-q)T}`):

.. math::

    C = \tfrac12\big(S_0e^{-qT} - Ke^{-rT}\big) + \frac{e^{-rT}}{\pi}\int_0^\infty
    \mathrm{Re}\Big[\frac{e^{-iuk}\big(\varphi(u - i) - K\varphi(u)\big)}{iu}\Big]\,du.

``method="quad"`` integrates adaptively (QUADPACK, reference accuracy); ``"gauss"`` uses a
fixed Gauss-Legendre rule on a truncated domain, vectorised over strikes (fast, for
calibration). Puts follow from put-call parity.

**Carr-Madan FFT.** The damped call :math:`e^{\alpha k}C(k)` has Fourier transform

.. math::

    \psi(v) = \frac{e^{-rT}\varphi(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v},

inverted on a log-strike grid with the FFT and Simpson weights; prices at arbitrary
strikes come from cubic-spline interpolation. The damping :math:`\alpha` is configurable.

References
----------
Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38(3-4), 481-482.
Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform.
*Journal of Computational Finance*, 2(4), 61-73.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from mcengine._typing import ComplexArray, FloatArray
from mcengine._validation import require_choice, require_int_at_least, require_positive
from mcengine.engines.analytic import deterministic_result
from mcengine.models.base import Model
from mcengine.products.base import OPTION_TYPES, Product
from mcengine.products.european import EuropeanOption
from mcengine.results import PricingResult

FOURIER_METHODS: tuple[str, ...] = ("gil-pelaez", "carr-madan")
_GL_CACHE: dict[int, tuple[FloatArray, FloatArray]] = {}


def _require_char_func(model: Model) -> None:
    if not model.has_char_func:
        raise ValueError(f"model {type(model).__name__} has no characteristic function")


def _parity(
    calls: FloatArray, model: Model, strikes: FloatArray, maturity: float, option_type: str
) -> FloatArray:
    if option_type == "call":
        return calls
    forward_s = model.s0 * math.exp(-model.q * maturity)
    return np.asarray(calls - forward_s + strikes * math.exp(-model.r * maturity))


def _gil_pelaez_integrand(
    model: Model, maturity: float, log_k: float, strike: float
) -> Callable[[float], float]:
    def integrand(u: float) -> float:
        uu = np.array([u, u - 1j])
        phi = model.char_func(uu, maturity)
        val = np.exp(-1j * u * log_k) * (phi[1] - strike * phi[0]) / (1j * u)
        return float(val.real)

    return integrand


def gil_pelaez_price(
    model: Model,
    strike: float,
    maturity: float,
    option_type: str = "call",
) -> float:
    """Reference price by adaptive Gil-Pelaez integration (``scipy.integrate.quad``)."""
    _require_char_func(model)
    require_positive("strike", strike)
    require_positive("maturity", maturity)
    require_choice("option_type", option_type, OPTION_TYPES)
    integral, _ = quad(
        _gil_pelaez_integrand(model, maturity, math.log(strike), strike),
        0.0,
        np.inf,
        limit=1000,
        epsabs=1e-13,
        epsrel=1e-11,
    )
    disc = math.exp(-model.r * maturity)
    call = (
        0.5 * (model.s0 * math.exp(-model.q * maturity) - strike * disc) + disc / math.pi * integral
    )
    return float(_parity(np.array([call]), model, np.array([strike]), maturity, option_type)[0])


def _gauss_legendre(n: int) -> tuple[FloatArray, FloatArray]:
    if n not in _GL_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = (np.asarray(nodes), np.asarray(weights))
    return _GL_CACHE[n]


def _truncation(model: Model, maturity: float, tol: float = 1e-14) -> float:
    """Smallest power of two ``u`` with ``|phi(u)| < tol`` (at most 2^12)."""
    for power in range(3, 13):
        u = 2.0**power
        if abs(model.char_func(np.array([u]), maturity)[0]) < tol:
            return u
    return 2.0**12


def fourier_prices(
    model: Model,
    strikes: ArrayLike,
    maturity: float,
    option_type: str = "call",
    n_nodes: int = 400,
) -> FloatArray:
    """Gil-Pelaez prices for many strikes with a fixed Gauss-Legendre rule (vectorised).

    The domain is truncated where :math:`|\\varphi(u)| < 10^{-14}`; ``n_nodes`` nodes
    are used on each of the two halves of the truncated interval.
    """
    _require_char_func(model)
    require_positive("maturity", maturity)
    require_choice("option_type", option_type, OPTION_TYPES)
    require_int_at_least("n_nodes", n_nodes, 16)
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if np.any(k <= 0.0):
        raise ValueError("strikes must be > 0")
    u_max = _truncation(model, maturity)
    x, w = _gauss_legendre(n_nodes)
    # split [0, u_max] into [0, u_max/8] and [u_max/8, u_max]: the integrand varies fastest near 0
    edges = [(0.0, u_max / 8.0), (u_max / 8.0, u_max)]
    u = np.concatenate([0.5 * (b - a) * x + 0.5 * (b + a) for a, b in edges])
    wu = np.concatenate([0.5 * (b - a) * w for a, b in edges])
    phi = model.char_func(u.astype(np.complex128), maturity)
    phi_shift = model.char_func(np.asarray(u - 1j, dtype=np.complex128), maturity)
    kernel: ComplexArray = np.exp(-1j * np.outer(np.log(k), u))
    integrand = (kernel * (phi_shift[None, :] - k[:, None] * phi[None, :]) / (1j * u)).real
    disc = math.exp(-model.r * maturity)
    calls = 0.5 * (model.s0 * math.exp(-model.q * maturity) - k * disc) + disc / math.pi * (
        integrand @ wu
    )
    return _parity(calls, model, k, maturity, option_type)


def carr_madan_prices(
    model: Model,
    strikes: ArrayLike,
    maturity: float,
    option_type: str = "call",
    alpha: float = 1.5,
    n_fft: int = 2**14,
    eta: float = 0.1,
) -> FloatArray:
    """Carr-Madan FFT prices, interpolated at ``strikes`` by a cubic spline.

    Parameters
    ----------
    alpha
        Damping factor (> 0); ``alpha + 1`` must keep ``E[S_T^{alpha+1}]`` finite.
    n_fft, eta
        FFT size (power of two) and spacing of the integration grid; the log-strike
        spacing is ``2 pi / (n_fft eta)``.
    """
    _require_char_func(model)
    require_positive("alpha", alpha)
    require_positive("eta", eta)
    require_choice("option_type", option_type, OPTION_TYPES)
    if n_fft < 16 or n_fft & (n_fft - 1):
        raise ValueError(f"n_fft must be a power of two >= 16, got {n_fft}")
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if np.any(k <= 0.0):
        raise ValueError("strikes must be > 0")
    lam = 2.0 * math.pi / (n_fft * eta)
    b = 0.5 * n_fft * lam
    centre = math.log(model.s0)
    v = eta * np.arange(n_fft)
    disc = math.exp(-model.r * maturity)
    psi = (
        disc
        * model.char_func(np.asarray(v - (alpha + 1.0) * 1j, dtype=np.complex128), maturity)
        / (alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v)
    )
    simpson = (3.0 + (-1.0) ** (np.arange(n_fft) + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    x = np.exp(-1j * v * (centre - b)) * psi * eta * simpson
    log_strikes = centre - b + lam * np.arange(n_fft)
    calls_grid = np.exp(-alpha * log_strikes) / math.pi * np.fft.fft(x).real
    log_k = np.log(k)
    if np.any(log_k < log_strikes[0]) or np.any(log_k > log_strikes[-1]):
        raise ValueError("strike outside the FFT log-strike grid; decrease eta")
    calls = CubicSpline(log_strikes, calls_grid)(log_k)
    return _parity(np.asarray(calls, dtype=np.float64), model, k, maturity, option_type)


def price_fourier(
    model: Model,
    product: Product,
    method: str = "gil-pelaez",
    alpha: float = 1.5,
    n_fft: int = 2**14,
    eta: float = 0.1,
) -> PricingResult:
    """Fourier price of a European option (``gil-pelaez`` or ``carr-madan``)."""
    started = time.perf_counter()
    require_choice("method", method, FOURIER_METHODS)
    if not isinstance(product, EuropeanOption):
        raise ValueError("Fourier pricing supports European options only")
    if method == "gil-pelaez":
        price = gil_pelaez_price(model, product.strike, product.maturity, product.option_type)
        return deterministic_result(price, "fourier-gp", started)
    prices = carr_madan_prices(
        model, [product.strike], product.maturity, product.option_type, alpha, n_fft, eta
    )
    return deterministic_result(float(prices[0]), "fourier-cm", started)
