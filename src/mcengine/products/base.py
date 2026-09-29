"""Common interface of the payoffs.

A product declares the dates it needs to observe (``monitoring_times``) and computes
undiscounted payoffs from simulated paths. The Monte Carlo engine builds a time grid
that contains every monitoring date, simulates, and discounts ``payoff(paths, times)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np

from mcengine._typing import FloatArray, IntArray

OptionType = Literal["call", "put"]
OPTION_TYPES: tuple[str, ...] = ("call", "put")


def time_indices(times: FloatArray, targets: FloatArray, *, atol: float = 1e-10) -> IntArray:
    """Indices ``i`` such that ``times[i] == targets`` (within ``atol``); raise if absent."""
    t = np.asarray(times, dtype=np.float64)
    idx = np.clip(np.searchsorted(t, targets - atol), 0, t.size - 1)
    if not np.allclose(t[idx], targets, rtol=0.0, atol=atol):
        raise ValueError("simulation grid does not contain every monitoring date of the product")
    return idx.astype(np.int64)


class Product(ABC):
    """Abstract payoff on a single underlying."""

    maturity: float
    #: Whether the payoff depends on more than the terminal price.
    is_path_dependent: ClassVar[bool]
    #: Short name used in tables and the CLI.
    name: ClassVar[str]

    @abstractmethod
    def monitoring_times(self) -> FloatArray:
        """Strictly increasing observation dates in ``(0, maturity]``, ending at maturity."""

    @abstractmethod
    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Undiscounted payoff for paths of shape ``(n, len(times))`` on grid ``times``."""

    @property
    def label(self) -> str:
        """Human-readable description."""
        return self.name


def uniform_monitoring(maturity: float, n_dates: int) -> FloatArray:
    """Equally spaced dates ``T i / m`` for ``i = 1..m``."""
    if n_dates < 1:
        raise ValueError(f"number of monitoring dates must be >= 1, got {n_dates}")
    return np.asarray(maturity * np.arange(1, n_dates + 1) / n_dates, dtype=np.float64)


def vanilla_payoff(spot: FloatArray, strike: float, option_type: str) -> FloatArray:
    """``max(S - K, 0)`` for calls, ``max(K - S, 0)`` for puts."""
    if option_type == "call":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)
