"""Common interface of the asset-price models.

A model maps a tensor of independent standard normals ``z`` with shape
``(n_paths, n_steps, n_factors)`` to price paths on a time grid. Separating the
Gaussian inputs from the dynamics lets one model implementation serve every sampling
scheme of the Monte Carlo engine (pseudo-random, antithetic ``-z``, scrambled Sobol
with Brownian bridge, importance-sampling drift shifts): the engine only changes how
``z`` is produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from mcengine._typing import ComplexArray, FloatArray
from mcengine.random.generators import make_rng, standard_normals


def validate_time_grid(times: FloatArray) -> FloatArray:
    """Return ``times`` as a float array after checking ``t0 = 0 < t1 < ... < tn``."""
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time grid must be one-dimensional with at least two points")
    if t[0] != 0.0:
        raise ValueError(f"time grid must start at 0, got {t[0]}")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("time grid must be finite and strictly increasing")
    return t


class Model(ABC):
    """Abstract risk-neutral model of one asset with constant rate ``r`` and yield ``q``."""

    s0: float
    r: float
    q: float
    #: Number of independent standard normals consumed per path and time step.
    n_factors: ClassVar[int]
    #: Index of the factor that drives the price (shifted by importance sampling).
    price_factor: ClassVar[int] = 0
    #: Whether paths are exact on any grid (no time-discretisation bias).
    exact_simulation: ClassVar[bool]
    #: Short name used in tables and the CLI.
    name: ClassVar[str]

    @abstractmethod
    def paths_from_normals(self, times: FloatArray, z: FloatArray) -> FloatArray:
        """Map normals ``z`` of shape ``(n, len(times) - 1, n_factors)`` to price paths.

        Returns an array of shape ``(n, len(times))`` whose first column is ``s0``.
        """

    def char_func(self, u: ComplexArray | FloatArray, maturity: float) -> ComplexArray:
        """Characteristic function of ``ln S_T``: ``E[exp(i u ln S_T)]``."""
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form characteristic function"
        )

    @property
    def has_char_func(self) -> bool:
        """Whether :meth:`char_func` is implemented."""
        return type(self).char_func is not Model.char_func

    def forward(self, maturity: float) -> float:
        """Risk-neutral forward ``S0 exp((r - q) T)``."""
        return float(self.s0 * np.exp((self.r - self.q) * maturity))

    def simulate_paths(
        self,
        times: FloatArray,
        n_paths: int,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """Simulate ``n_paths`` price paths on ``times`` (first point must be 0)."""
        t = validate_time_grid(times)
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        gen = make_rng(seed, rng)
        z = standard_normals(gen, (n_paths, t.size - 1, self.n_factors))
        return self.paths_from_normals(t, z)

    def _check_normals(self, times: FloatArray, z: FloatArray) -> FloatArray:
        t = validate_time_grid(times)
        if z.ndim != 3 or z.shape[1] != t.size - 1 or z.shape[2] != self.n_factors:
            raise ValueError(
                f"normals must have shape (n, {t.size - 1}, {self.n_factors}), got {z.shape}"
            )
        return t
