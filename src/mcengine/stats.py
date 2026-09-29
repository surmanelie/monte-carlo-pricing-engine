r"""Streaming moments and confidence intervals.

Monte Carlo runs are processed in chunks so that memory stays constant. Each chunk's
mean and centred co-moment matrix are computed with a two-pass formula and merged into
the running totals with the pairwise update of Chan, Golub & LeVeque (1979):

.. math::

    n = n_a + n_b,\qquad \delta = \bar x_b - \bar x_a,\qquad
    \bar x = \bar x_a + \delta\,\frac{n_b}{n},\qquad
    C = C_a + C_b + \delta\delta^\top \frac{n_a n_b}{n}.

This is the batched generalisation of Welford's (1962) one-pass update and is
numerically stable for any chunk size.

References
----------
Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). *Updating formulae and a pairwise
algorithm for computing sample variances*. Stanford CS report STAN-CS-79-773.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from mcengine._typing import FloatArray


def z_value(level: float) -> float:
    """Two-sided standard-normal quantile for a confidence ``level`` in (0, 1)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must lie in (0, 1), got {level!r}")
    return float(ndtri(0.5 + 0.5 * level))


def confidence_interval(mean: float, std_error: float, level: float = 0.95) -> tuple[float, float]:
    """Normal-approximation confidence interval ``mean +/- z * std_error``."""
    if std_error < 0.0 or not math.isfinite(std_error):
        raise ValueError(f"std_error must be finite and >= 0, got {std_error!r}")
    half = z_value(level) * std_error
    return mean - half, mean + half


class StreamingMoments:
    """Streaming mean and covariance of ``dim``-dimensional samples.

    Parameters
    ----------
    dim
        Number of jointly observed quantities (1 for a plain estimator, 2 for an
        estimator with one control variate).

    Examples
    --------
    >>> acc = StreamingMoments()
    >>> acc.update(np.array([1.0, 2.0])); acc.update(np.array([3.0, 4.0]))
    >>> float(acc.mean[0]), float(acc.variance[0])
    (2.5, 1.6666666666666667)
    """

    def __init__(self, dim: int = 1) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.count = 0
        self._mean: FloatArray = np.zeros(dim)
        self._comoment: FloatArray = np.zeros((dim, dim))

    def update(self, samples: ArrayLike) -> None:
        """Add a batch of samples with shape ``(n,)`` (if ``dim == 1``) or ``(n, dim)``."""
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim == 1 and self.dim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected samples of shape (n, {self.dim}), got {x.shape}")
        n_b = x.shape[0]
        if n_b == 0:
            return
        mean_b = x.mean(axis=0)
        centred = x - mean_b
        self._merge(n_b, mean_b, centred.T @ centred)

    def merge(self, other: StreamingMoments) -> None:
        """Merge another accumulator (e.g. from a parallel worker) into this one."""
        if other.dim != self.dim:
            raise ValueError("cannot merge accumulators of different dimension")
        if other.count:
            self._merge(other.count, other._mean.copy(), other._comoment.copy())

    def _merge(self, n_b: int, mean_b: FloatArray, comoment_b: FloatArray) -> None:
        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / n)
        self._comoment = self._comoment + comoment_b + np.outer(delta, delta) * (n_a * n_b / n)
        self.count = n

    @property
    def mean(self) -> FloatArray:
        """Sample mean vector."""
        if self.count == 0:
            raise ValueError("no samples accumulated")
        return self._mean.copy()

    @property
    def covariance(self) -> FloatArray:
        """Unbiased sample covariance matrix (``ddof = 1``)."""
        if self.count < 2:
            raise ValueError("at least two samples are needed for a covariance")
        return self._comoment / (self.count - 1)

    @property
    def variance(self) -> FloatArray:
        """Unbiased sample variances (diagonal of :attr:`covariance`)."""
        return np.diag(self.covariance).copy()

    def std_error(self, index: int = 0) -> float:
        """Standard error of the mean of component ``index``."""
        return math.sqrt(max(float(self.variance[index]), 0.0) / self.count)
