r"""Merton (1976) jump-diffusion.

.. math::

    \frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sigma\,dW_t + (J - 1)\,dN_t,
    \qquad \ln J \sim N(\mu_J, \delta^2),\quad \bar k = e^{\mu_J + \delta^2/2} - 1,

with :math:`N` a Poisson process of intensity :math:`\lambda`. The compensator
:math:`\lambda\bar k` makes :math:`e^{-(r-q)t}S_t` a martingale.

**Exact simulation.** Over a step :math:`\Delta`, the number of jumps is
:math:`N_\Delta \sim \mathrm{Poisson}(\lambda\Delta)` and, conditionally on
:math:`N_\Delta = n`, the sum of the log-jumps is exactly :math:`N(n\mu_J, n\delta^2)`:

.. math::

    \ln\frac{S_{t+\Delta}}{S_t} = (r - q - \lambda\bar k - \tfrac12\sigma^2)\Delta
    + \sigma\sqrt\Delta\,Z_1 + N_\Delta\mu_J + \delta\sqrt{N_\Delta}\,Z_3,

where :math:`N_\Delta = F^{-1}_{\mathrm{Poi}}(\Phi(Z_2))` by inversion. Three independent
normals per step therefore drive the scheme, so it plugs into every sampling method of
the engine (antithetic, quasi-Monte Carlo, importance sampling).

References
----------
Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous.
*Journal of Financial Economics*, 3(1-2), 125-144.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.5.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import ndtr
from scipy.stats import poisson

from mcengine._typing import ComplexArray, FloatArray, IntArray
from mcengine._validation import (
    require_finite,
    require_non_negative,
    require_positive,
)
from mcengine.models.base import Model


def poisson_inverse(u: FloatArray, mean: float, tail: float = 1e-15) -> IntArray:
    """Inverse Poisson CDF by table look-up (exact up to a tail mass ``tail``)."""
    if mean == 0.0:
        return np.zeros(np.shape(u), dtype=np.int64)
    k_max = int(poisson.isf(tail, mean)) + 1
    cdf = poisson.cdf(np.arange(k_max + 1), mean)
    return np.minimum(np.searchsorted(cdf, u, side="left"), k_max).astype(np.int64)


@dataclass(frozen=True)
class Merton(Model):
    """Merton jump-diffusion.

    Parameters
    ----------
    s0
        Spot (> 0).
    r, q
        Rate and dividend yield (per year).
    sigma
        Diffusion volatility (> 0).
    lam
        Jump intensity :math:`\\lambda` (jumps per year, >= 0).
    mu_j, delta_j
        Mean and standard deviation of the log jump size.
    """

    s0: float
    r: float
    sigma: float
    lam: float
    mu_j: float
    delta_j: float
    q: float = 0.0

    n_factors: ClassVar[int] = 3
    exact_simulation: ClassVar[bool] = True
    name: ClassVar[str] = "merton"

    def __post_init__(self) -> None:
        require_positive("s0", self.s0)
        require_finite("r", self.r)
        require_non_negative("q", self.q)
        require_positive("sigma", self.sigma)
        require_non_negative("lam", self.lam)
        require_finite("mu_j", self.mu_j)
        require_non_negative("delta_j", self.delta_j)

    @property
    def kbar(self) -> float:
        r"""Mean relative jump size :math:`\bar k = E[J] - 1`."""
        return math.exp(self.mu_j + 0.5 * self.delta_j**2) - 1.0

    def paths_from_normals(self, times: FloatArray, z: FloatArray) -> FloatArray:
        """Exact paths (normals: diffusion, jump count, jump size along the last axis)."""
        t = self._check_normals(times, z)
        dt = np.diff(t)
        drift = (self.r - self.q - self.lam * self.kbar - 0.5 * self.sigma**2) * dt
        increments = drift + self.sigma * np.sqrt(dt) * z[:, :, 0]
        uniforms = ndtr(z[:, :, 1])
        for k, step in enumerate(dt):
            counts = poisson_inverse(uniforms[:, k], self.lam * float(step))
            increments[:, k] += counts * self.mu_j + self.delta_j * np.sqrt(counts) * z[:, k, 2]
        log_paths = np.zeros((z.shape[0], t.size))
        np.cumsum(increments, axis=1, out=log_paths[:, 1:])
        return np.asarray(self.s0 * np.exp(log_paths), dtype=np.float64)

    def char_func(self, u: ComplexArray | FloatArray, maturity: float) -> ComplexArray:
        r"""Characteristic function of :math:`\ln S_T`:

        .. math::

            \exp\!\Big(iu\big(\ln S_0 + (r - q - \lambda\bar k - \tfrac12\sigma^2)T\big)
            - \tfrac12\sigma^2u^2T + \lambda T\big(e^{iu\mu_J - \delta^2u^2/2} - 1\big)\Big).
        """
        uu = np.asarray(u, dtype=np.complex128)
        t = maturity
        drift = (
            math.log(self.s0) + (self.r - self.q - self.lam * self.kbar - 0.5 * self.sigma**2) * t
        )
        jumps = self.lam * t * (np.exp(1j * uu * self.mu_j - 0.5 * self.delta_j**2 * uu**2) - 1.0)
        return np.asarray(
            np.exp(1j * uu * drift - 0.5 * self.sigma**2 * uu**2 * t + jumps), dtype=np.complex128
        )
