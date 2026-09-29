r"""Black-Scholes geometric Brownian motion.

Under the risk-neutral measure

.. math:: dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t,

and the exact solution on any grid is

.. math::

    S_{t_{k+1}} = S_{t_k}\exp\!\Big(\big(r - q - \tfrac12\sigma^2\big)\Delta t_k
    + \sigma\sqrt{\Delta t_k}\,Z_k\Big),\qquad Z_k \sim N(0, 1)\ \text{i.i.d.},

implemented as a cumulative sum of log-increments (no discretisation bias).

References
----------
Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
*Journal of Political Economy*, 81(3), 637-654.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mcengine._typing import ComplexArray, FloatArray
from mcengine._validation import require_finite, require_non_negative, require_positive
from mcengine.models.base import Model


@dataclass(frozen=True)
class GBM(Model):
    """Geometric Brownian motion.

    Parameters
    ----------
    s0
        Spot price (> 0).
    r
        Continuously compounded risk-free rate (per year).
    sigma
        Volatility (per square-root year, > 0).
    q
        Continuous dividend yield (per year, default 0).
    """

    s0: float
    r: float
    sigma: float
    q: float = 0.0

    n_factors: ClassVar[int] = 1
    exact_simulation: ClassVar[bool] = True
    name: ClassVar[str] = "gbm"

    def __post_init__(self) -> None:
        require_positive("s0", self.s0)
        require_finite("r", self.r)
        require_positive("sigma", self.sigma)
        require_non_negative("q", self.q)

    def paths_from_normals(self, times: FloatArray, z: FloatArray) -> FloatArray:
        """Exact GBM paths; see the module docstring for the scheme."""
        t = self._check_normals(times, z)
        dt = np.diff(t)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        log_increments = drift + self.sigma * np.sqrt(dt) * z[:, :, 0]
        log_paths = np.empty((z.shape[0], t.size))
        log_paths[:, 0] = 0.0
        np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
        return np.asarray(self.s0 * np.exp(log_paths), dtype=np.float64)

    def char_func(self, u: ComplexArray | FloatArray, maturity: float) -> ComplexArray:
        r"""Characteristic function of :math:`\ln S_T`, a normal with mean
        :math:`\ln S_0 + (r - q - \sigma^2/2)T` and variance :math:`\sigma^2 T`."""
        uu = np.asarray(u, dtype=np.complex128)
        mean = np.log(self.s0) + (self.r - self.q - 0.5 * self.sigma**2) * maturity
        return np.asarray(
            np.exp(1j * uu * mean - 0.5 * self.sigma**2 * maturity * uu**2), dtype=np.complex128
        )
