"""Optional Numba kernels (``pip install monte-carlo-pricing-engine[fast]``).

The kernels loop over paths and time steps in compiled code. Random numbers are always
drawn by NumPy *before* calling a kernel, and the arithmetic mirrors the vectorised NumPy
implementation line by line, so both backends return the same numbers for the same seed
(up to floating-point rounding, checked to ``1e-12`` relative in the tests).

Without Numba the decorator is a no-op: the kernels remain importable (and are exercised
as plain Python in the tests), but the ``"numba"`` backend refuses to run.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])
BACKENDS: tuple[str, ...] = ("numpy", "numba")

try:  # pragma: no cover - depends on the environment
    _numba: Any = importlib.import_module("numba")
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    _numba = None
    HAS_NUMBA = False

#: ``numba.prange`` (parallel loop over independent paths) or the builtin ``range``.
prange = cast(Callable[[int], range], _numba.prange if HAS_NUMBA else range)


def jit(func: F) -> F:
    """``numba.njit(cache=True, parallel=True)`` if Numba is installed, identity otherwise.

    ``parallel=True`` only distributes independent paths over threads (``prange``); the
    per-path arithmetic, and hence the result, is unchanged.
    """
    if HAS_NUMBA:  # pragma: no cover - depends on the environment
        return cast(F, _numba.njit(cache=True, parallel=True)(func))
    return func  # pragma: no cover - depends on the environment


def require_backend(backend: str) -> str:
    """Validate a backend name and check that Numba is available if requested."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}; expected one of {BACKENDS}")
    if backend == "numba" and not HAS_NUMBA:  # pragma: no cover - depends on the environment
        raise ValueError("backend 'numba' requires the [fast] extra: pip install numba")
    return backend


@jit
def heston_qe_kernel(
    log_s0: float,
    v0: float,
    dts: np.ndarray,
    zv: np.ndarray,
    zs: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    carry: float,
    psi_c: float,
    correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Andersen QE scheme path by path (same arithmetic as ``Heston._qe_step``)."""
    n, m = zv.shape
    log_s = np.empty((n, m + 1))
    var = np.empty((n, m + 1))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # per-step constants (same formulas as the NumPy implementation)
    ekd = np.empty(m)
    c1 = np.empty(m)
    c2 = np.empty(m)
    k0 = np.empty(m)
    k1 = np.empty(m)
    k2 = np.empty(m)
    k3 = np.empty(m)
    for k in range(m):
        dt = dts[k]
        ekd[k] = math.exp(-kappa * dt)
        c1[k] = xi**2 * ekd[k] * (1.0 - ekd[k]) / kappa
        c2[k] = theta * xi**2 * (1.0 - ekd[k]) ** 2 / (2.0 * kappa)
        k0[k] = -rho * kappa * theta * dt / xi
        k1[k] = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2[k] = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3[k] = 0.5 * dt * (1.0 - rho**2)
    for i in prange(n):
        log_s[i, 0] = log_s0
        var[i, 0] = v0
        for k in range(m):
            v = var[i, k]
            big_a = k2[k] + 0.5 * k3[k]
            mean = theta + (v - theta) * ekd[k]
            s2 = v * c1[k] + c2[k]
            psi = s2 / mean**2
            if psi <= psi_c:
                inv = 2.0 / psi
                b2 = inv - 1.0 + math.sqrt(inv * (inv - 1.0))
                a = mean / (1.0 + b2)
                v_next = a * (math.sqrt(b2) + zv[i, k]) ** 2
                two_aa = 2.0 * big_a * a
                if correction and two_aa < 1.0:
                    shift = (
                        -big_a * b2 * a / (1.0 - two_aa)
                        + 0.5 * math.log1p(-two_aa)
                        - 0.5 * k3[k] * v
                    )
                else:
                    shift = k0[k] + k1[k] * v
            else:
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / mean
                one_minus_u = 0.5 * math.erfc(zv[i, k] * inv_sqrt2)
                if one_minus_u >= 1.0 - p:
                    v_next = 0.0
                else:
                    v_next = math.log((1.0 - p) / max(one_minus_u, 1e-300)) / beta
                if correction and big_a < beta:
                    shift = -math.log(p + beta * (1.0 - p) / (beta - big_a)) - 0.5 * k3[k] * v
                else:
                    shift = k0[k] + k1[k] * v
            diffusion = math.sqrt(max(k3[k] * v + k3[k] * v_next, 0.0))
            log_s[i, k + 1] = (
                log_s[i, k] + carry * dts[k] + shift + k2[k] * v_next + diffusion * zs[i, k]
            )
            var[i, k + 1] = v_next
    return log_s, var


@jit
def lsm_policy_kernel(
    spots: np.ndarray,
    exercise: np.ndarray,
    discounts: np.ndarray,
    coefficients: np.ndarray,
    fitted: np.ndarray,
    strike: float,
    laguerre: bool,
) -> np.ndarray:
    """Discounted cash flow of a fitted LSM policy, path by path.

    ``spots`` and ``exercise`` have shape ``(n, M)`` (exercise dates in columns),
    ``discounts[k] = exp(-r t_k)``. The basis is the constant plus weighted Laguerre
    polynomials (three-term recurrence) or monomials of ``x = S / K``.
    """
    n, n_dates = spots.shape
    degree = coefficients.shape[1] - 1
    value = np.zeros(n)
    for i in prange(n):
        for k in range(n_dates):
            ex = exercise[i, k]
            if ex <= 0.0:
                continue
            if k == n_dates - 1:
                value[i] = discounts[k] * ex
                break
            if not fitted[k]:
                continue
            x = spots[i, k] / strike
            cont = coefficients[k, 0]
            if laguerre:
                weight = math.exp(-0.5 * x)
                l_prev, l_curr = 0.0, 1.0
                for j in range(degree):
                    cont += coefficients[k, j + 1] * weight * l_curr
                    l_next = ((2.0 * j + 1.0 - x) * l_curr - j * l_prev) / (j + 1.0)
                    l_prev, l_curr = l_curr, l_next
            else:
                power = 1.0
                for j in range(degree):
                    power *= x
                    cont += coefficients[k, j + 1] * power
            if ex >= cont:
                value[i] = discounts[k] * ex
                break
    return value
