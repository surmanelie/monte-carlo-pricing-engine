r"""Heston (1993) stochastic-volatility model.

.. math::

    dS_t = (r - q)S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t,\qquad
    dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW^V_t,\qquad
    d\langle W^S, W^V\rangle_t = \rho\,dt.

Two discretisations are provided (``scheme``):

``"qe"`` - Andersen (2008) quadratic-exponential scheme. Given :math:`V_t`, the next
variance is moment-matched to the exact non-central chi-square law with mean
:math:`m = \theta + (V_t - \theta)e^{-\kappa\Delta}` and variance
:math:`s^2 = V_t\xi^2e^{-\kappa\Delta}(1 - e^{-\kappa\Delta})/\kappa
+ \theta\xi^2(1 - e^{-\kappa\Delta})^2/(2\kappa)`; with :math:`\psi = s^2/m^2`:

* :math:`\psi \le \psi_c`: :math:`V_{t+\Delta} = a(b + Z_V)^2` with
  :math:`b^2 = 2/\psi - 1 + \sqrt{2/\psi}\sqrt{2/\psi - 1}`, :math:`a = m/(1 + b^2)`;
* :math:`\psi > \psi_c`: :math:`V_{t+\Delta} = \beta^{-1}\ln\frac{1-p}{1-U}` if
  :math:`U > p` else 0, with :math:`p = (\psi - 1)/(\psi + 1)`, :math:`\beta = (1 - p)/m`,
  :math:`U = \Phi(Z_V)`.

The log-price uses the central discretisation (:math:`\gamma_1 = \gamma_2 = \tfrac12`)

.. math::

    \ln S_{t+\Delta} = \ln S_t + (r - q)\Delta + K_0^* + K_1V_t + K_2V_{t+\Delta}
    + \sqrt{K_3V_t + K_4V_{t+\Delta}}\,Z,

with Andersen's martingale correction :math:`K_0^*`, which makes
:math:`e^{-(r-q)\Delta}S_{t+\Delta}` an exact martingale of the discrete scheme.

``"euler"`` - full-truncation Euler (Lord, Koekkoek & van Dijk, 2010), with
:math:`V^+ = \max(V, 0)` in drift and diffusion and a log-Euler step for the price.

The characteristic function uses the "little Heston trap" formulation of Albrecher,
Mayer, Schoutens & Tistaert (2007), which is continuous in the complex logarithm.

References
----------
Heston, S. L. (1993). A closed-form solution for options with stochastic volatility.
*Review of Financial Studies*, 6(2), 327-343.
Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility
model. *Journal of Computational Finance*, 11(3), 1-42.
Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). The little Heston trap.
*Wilmott Magazine*, January, 83-92.
Lord, R., Koekkoek, R., & van Dijk, D. (2010). A comparison of biased simulation schemes
for stochastic volatility models. *Quantitative Finance*, 10(2), 177-194.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import ndtr

from mcengine._typing import ComplexArray, FloatArray
from mcengine._validation import (
    require_choice,
    require_finite,
    require_in_range,
    require_non_negative,
    require_positive,
)
from mcengine.models.base import Model

SCHEMES: tuple[str, ...] = ("qe", "euler")


@dataclass(frozen=True)
class Heston(Model):
    """Heston stochastic-volatility model.

    Parameters
    ----------
    s0
        Spot price (> 0).
    r, q
        Risk-free rate and dividend yield (per year).
    v0
        Initial variance (>= 0).
    kappa
        Mean-reversion speed (> 0, per year).
    theta
        Long-run variance (> 0).
    xi
        Volatility of variance (> 0).
    rho
        Correlation between price and variance shocks, in [-1, 1].
    scheme
        ``"qe"`` (Andersen 2008) or ``"euler"`` (full truncation).
    psi_c
        QE switching threshold (Andersen recommends 1.5).
    martingale_correction
        Apply Andersen's :math:`K_0^*` correction in the QE scheme.
    """

    s0: float
    r: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    q: float = 0.0
    scheme: str = "qe"
    psi_c: float = 1.5
    martingale_correction: bool = True

    n_factors: ClassVar[int] = 2
    exact_simulation: ClassVar[bool] = False
    name: ClassVar[str] = "heston"

    def __post_init__(self) -> None:
        require_positive("s0", self.s0)
        require_finite("r", self.r)
        require_non_negative("q", self.q)
        require_non_negative("v0", self.v0)
        require_positive("kappa", self.kappa)
        require_positive("theta", self.theta)
        require_positive("xi", self.xi)
        require_in_range("rho", self.rho, -1.0, 1.0)
        require_choice("scheme", self.scheme, SCHEMES)
        require_in_range("psi_c", self.psi_c, 1.0, 2.0)

    @property
    def feller_ratio(self) -> float:
        r""":math:`2\kappa\theta/\xi^2`; the variance stays strictly positive iff it is >= 1."""
        return 2.0 * self.kappa * self.theta / self.xi**2

    def with_scheme(self, scheme: str) -> Heston:
        """Same parameters, other discretisation scheme."""
        return Heston(
            self.s0, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, self.q,
            scheme, self.psi_c, self.martingale_correction,
        )  # fmt: skip

    def paths_from_normals(self, times: FloatArray, z: FloatArray) -> FloatArray:
        """Price paths; ``z[..., 0]`` drives the variance, ``z[..., 1]`` the price."""
        return self.simulate_with_variance(times, z)[0]

    def simulate_with_variance(
        self, times: FloatArray, z: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Price and variance paths, both of shape ``(n, len(times))``."""
        t = self._check_normals(times, z)
        n = z.shape[0]
        log_s = np.empty((n, t.size))
        var = np.empty((n, t.size))
        log_s[:, 0] = math.log(self.s0)
        var[:, 0] = self.v0
        step = self._qe_step if self.scheme == "qe" else self._euler_step
        for k, dt in enumerate(np.diff(t)):
            log_s[:, k + 1], var[:, k + 1] = step(
                log_s[:, k], var[:, k], float(dt), z[:, k, 0], z[:, k, 1]
            )
        return np.exp(log_s), var

    def _qe_step(
        self, log_s: FloatArray, v: FloatArray, dt: float, zv: FloatArray, zs: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        ekd = math.exp(-kappa * dt)
        m = theta + (v - theta) * ekd
        s2 = v * (xi**2 * ekd * (1.0 - ekd) / kappa) + theta * xi**2 * (1.0 - ekd) ** 2 / (
            2.0 * kappa
        )
        psi = s2 / m**2
        k0 = -rho * kappa * theta * dt / xi
        k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = k4 = 0.5 * dt * (1.0 - rho**2)
        big_a = k2 + 0.5 * k4
        v_next = np.empty_like(v)
        shift = np.empty_like(v)  # K0* + K1 V (or K0 + K1 V without the correction)
        quad = psi <= self.psi_c
        n_quad = int(np.count_nonzero(quad))
        if n_quad:
            # quadratic branch: V' = a (b + Zv)^2, b^2 = 2/psi - 1 + sqrt(2/psi (2/psi - 1))
            iq: slice | np.ndarray = slice(None) if n_quad == v.size else np.flatnonzero(quad)
            inv = 2.0 / psi[iq]
            b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
            a = m[iq] / (1.0 + b2)
            v_next[iq] = a * (np.sqrt(b2) + zv[iq]) ** 2
            if self.martingale_correction:
                two_aa = 2.0 * big_a * a
                ok = two_aa < 1.0
                safe = np.where(ok, two_aa, 0.0)
                corrected = (
                    -big_a * b2 * a / (1.0 - safe) + 0.5 * np.log1p(-safe) - 0.5 * k3 * v[iq]
                )
                shift[iq] = np.where(ok, corrected, k0 + k1 * v[iq])
        if n_quad < v.size:
            # exponential branch: point mass p at zero, exponential tail with rate beta
            ie: slice | np.ndarray = slice(None) if n_quad == 0 else np.flatnonzero(~quad)
            psi_e = psi[ie]
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - p) / m[ie]
            one_minus_u = ndtr(-zv[ie])  # 1 - Phi(zv) without cancellation
            tail = np.log((1.0 - p) / np.maximum(one_minus_u, 1e-300)) / beta
            v_next[ie] = np.where(one_minus_u >= 1.0 - p, 0.0, tail)
            if self.martingale_correction:
                ok = big_a < beta
                gap = np.where(ok, beta - big_a, 1.0)
                corrected = -np.log(p + beta * (1.0 - p) / gap) - 0.5 * k3 * v[ie]
                shift[ie] = np.where(ok, corrected, k0 + k1 * v[ie])
        if not self.martingale_correction:
            shift = k0 + k1 * v
        diffusion = np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0))
        log_next = log_s + (self.r - self.q) * dt + shift + k2 * v_next + diffusion * zs
        return log_next, v_next

    def _euler_step(
        self, log_s: FloatArray, v: FloatArray, dt: float, zv: FloatArray, zs: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        v_plus = np.maximum(v, 0.0)
        sq = np.sqrt(v_plus * dt)
        z_price = self.rho * zv + math.sqrt(1.0 - self.rho**2) * zs
        log_next = log_s + (self.r - self.q - 0.5 * v_plus) * dt + sq * z_price
        v_next = v + self.kappa * (self.theta - v_plus) * dt + self.xi * sq * zv
        return log_next, v_next

    def char_func(self, u: ComplexArray | FloatArray, maturity: float) -> ComplexArray:
        r"""Characteristic function of :math:`\ln S_T` (little-trap form).

        .. math::

            \varphi(u) = \exp\!\Big(iu(\ln S_0 + (r - q)T)
            + \frac{\kappa\theta}{\xi^2}\big[(\kappa - \rho\xi iu - d)T
            - 2\ln\tfrac{1 - ge^{-dT}}{1 - g}\big]
            + \frac{v_0}{\xi^2}(\kappa - \rho\xi iu - d)\frac{1 - e^{-dT}}{1 - ge^{-dT}}\Big),

        :math:`d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)}`,
        :math:`g = (\kappa - \rho\xi iu - d)/(\kappa - \rho\xi iu + d)`.
        """
        uu = np.asarray(u, dtype=np.complex128)
        kappa, theta, xi, rho, t = self.kappa, self.theta, self.xi, self.rho, maturity
        iu = 1j * uu
        beta = kappa - rho * xi * iu
        d = np.sqrt(beta**2 + xi**2 * (iu + uu**2))
        g = (beta - d) / (beta + d)
        edt = np.exp(-d * t)
        c_term = (kappa * theta / xi**2) * (
            (beta - d) * t - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
        )
        d_term = (beta - d) / xi**2 * (1.0 - edt) / (1.0 - g * edt)
        drift = iu * (math.log(self.s0) + (self.r - self.q) * t)
        return np.asarray(np.exp(drift + c_term + d_term * self.v0), dtype=np.complex128)
