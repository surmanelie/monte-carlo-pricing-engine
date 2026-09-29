r"""Randomised quasi-Monte Carlo: scrambled Sobol points and Brownian-bridge construction.

**Scrambled Sobol.** Points come from :class:`scipy.stats.qmc.Sobol` with Owen-type
scrambling, always drawn in powers of two so that the net balance properties hold. They
are mapped to normals by the inverse CDF.

**Brownian bridge.** A :math:`d`-dimensional low-discrepancy point is most uniform in its
first coordinates, so these are assigned to the path features that explain most of the
variance: first :math:`W_T`, then :math:`W_{T/2}` given its endpoints, then the quarter
points, and so on (Moskowitz & Caflisch, 1996; Glasserman 2003, Section 3.1.2). For
:math:`t_l < t_j < t_r`,

.. math::

    W_{t_j} = \frac{(t_r - t_j)W_{t_l} + (t_j - t_l)W_{t_r}}{t_r - t_l}
    + \sqrt{\frac{(t_j - t_l)(t_r - t_j)}{t_r - t_l}}\;Z_j.

The resulting increments :math:`(W_{t_{k+1}} - W_{t_k})/\sqrt{\Delta t_k}` are again
i.i.d. :math:`N(0, 1)` (the map is a change of basis of the Gaussian vector), so the
bridge can feed any model of the engine unchanged.

**Randomisation.** ``R`` independent scramblings give ``R`` i.i.d. unbiased estimates;
their sample standard deviation divided by :math:`\sqrt R` is an honest standard error
(Owen, 1997; L'Ecuyer & Lemieux, 2002).

References
----------
Sobol', I. M. (1967). On the distribution of points in a cube and the approximate
evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics*, 7(4).
Owen, A. B. (1997). Scrambled net variance for integrals of smooth functions.
*Annals of Statistics*, 25(4), 1541-1562.
Moskowitz, B., & Caflisch, R. E. (1996). Smoothness and dimension reduction in
quasi-Monte Carlo methods. *Mathematical and Computer Modelling*, 23(8-9), 37-54.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from mcengine._typing import FloatArray, IntArray

#: Largest dimension supported by SciPy's Sobol direction numbers.
MAX_SOBOL_DIM = 21201
_EPS = 1e-16


def is_power_of_two(n: int) -> bool:
    """Whether ``n`` is a positive power of two."""
    return n > 0 and n & (n - 1) == 0


@dataclass(frozen=True)
class BridgePlan:
    """Construction order of a Brownian bridge on a time grid ``t_0 = 0 < ... < t_m``.

    Step ``j`` of the construction fills grid index ``target[j]`` from the already known
    indices ``left[j]`` and ``right[j]`` (``right = -1`` for the terminal point, which is
    built from ``W_0 = 0`` alone).
    """

    target: IntArray
    left: IntArray
    right: IntArray
    w_left: FloatArray
    w_right: FloatArray
    scale: FloatArray


def bridge_plan(times: FloatArray) -> BridgePlan:
    """Bisection order for the grid ``times`` (``times[0] = 0``)."""
    t = np.asarray(times, dtype=np.float64)
    m = t.size - 1
    if m < 1 or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("bridge grid must start at 0 and be strictly increasing")
    target, left, right = [m], [0], [-1]
    w_left, w_right, scale = [0.0], [0.0], [float(np.sqrt(t[m]))]
    queue = [(0, m)]
    while queue:
        lo, hi = queue.pop(0)
        if hi - lo < 2:
            continue
        mid = (lo + hi) // 2
        span = t[hi] - t[lo]
        target.append(mid)
        left.append(lo)
        right.append(hi)
        w_left.append(float((t[hi] - t[mid]) / span))
        w_right.append(float((t[mid] - t[lo]) / span))
        scale.append(float(np.sqrt((t[mid] - t[lo]) * (t[hi] - t[mid]) / span)))
        queue.extend([(lo, mid), (mid, hi)])
    return BridgePlan(
        np.asarray(target, dtype=np.int64),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(w_left),
        np.asarray(w_right),
        np.asarray(scale),
    )


def bridge_increments(
    z: FloatArray, times: FloatArray, plan: BridgePlan | None = None
) -> FloatArray:
    """Standardised Brownian increments built from ``z`` in bridge order.

    ``z`` has shape ``(n, m)``; column ``j`` drives construction step ``j`` of the plan.
    Returns ``(W_{t_{k+1}} - W_{t_k}) / sqrt(dt_k)`` with shape ``(n, m)``.
    """
    t = np.asarray(times, dtype=np.float64)
    p = bridge_plan(t) if plan is None else plan
    n, m = z.shape
    if m != t.size - 1:
        raise ValueError(f"expected {t.size - 1} normals per path, got {m}")
    w = np.zeros((n, m + 1))
    for j in range(m):
        k, lo, hi = p.target[j], p.left[j], p.right[j]
        base = p.w_left[j] * w[:, lo] + (p.w_right[j] * w[:, hi] if hi >= 0 else 0.0)
        w[:, k] = base + p.scale[j] * z[:, j]
    return np.asarray(np.diff(w, axis=1) / np.sqrt(np.diff(t)), dtype=np.float64)


class SobolNormals:
    """Stream of scrambled-Sobol normals shaped ``(n, n_steps, n_factors)``.

    Sobol coordinate ``j * n_factors + f`` drives factor ``f`` at bridge step ``j``, so the
    best-distributed coordinates go to the coarse structure of every factor. With
    ``bridge=False`` the coordinates are used in time order (standard construction).
    """

    def __init__(
        self,
        times: FloatArray,
        n_factors: int,
        rng: np.random.Generator,
        bridge: bool = True,
    ) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.n_steps = self.times.size - 1
        self.n_factors = n_factors
        self.dim = self.n_steps * n_factors
        if self.dim > MAX_SOBOL_DIM:
            raise ValueError(f"Sobol dimension {self.dim} exceeds {MAX_SOBOL_DIM}")
        self.bridge = bridge
        self._plan = bridge_plan(self.times) if bridge else None
        self._engine = qmc.Sobol(d=self.dim, scramble=True, seed=rng)

    def draw(self, n: int) -> FloatArray:
        """Next ``n`` points (``n`` must be a power of two) as normals."""
        if not is_power_of_two(n):
            raise ValueError(f"Sobol draws must be powers of two, got {n}")
        u = self._engine.random(n)
        z: FloatArray = np.asarray(ndtri(np.clip(u, _EPS, 1.0 - _EPS)), dtype=np.float64).reshape(
            n, self.n_steps, self.n_factors
        )
        if self._plan is None:
            return z
        out = np.empty_like(z)
        for f in range(self.n_factors):
            out[:, :, f] = bridge_increments(z[:, :, f], self.times, self._plan)
        return out
