"""Immutable result containers returned by every pricing and Greeks engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from mcengine.stats import z_value


@dataclass(frozen=True)
class PricingResult:
    """Price estimate with its statistical diagnostics.

    Attributes
    ----------
    price
        Estimated (or exact) present value.
    std_error
        Standard error of a Monte Carlo estimate; ``None`` for deterministic engines.
    ci_low, ci_high
        Bounds of the 95 % confidence interval (``None`` for deterministic engines).
    n_paths, n_steps
        Number of simulated paths and time steps (``None`` if not applicable).
    method
        Short identifier, e.g. ``"mc-plain"``, ``"mc-cv"``, ``"qmc-sobol-bb"``,
        ``"lsm"``, ``"bs-analytic"``, ``"fourier-cm"``, ``"crr"``.
    elapsed_s
        Wall-clock time in seconds.
    reference
        Optional independent reference price attached for validation.
    diagnostics
        Method-specific numbers, e.g. the control-variate coefficient ``beta`` or the
        variance-reduction factor ``vr_factor``.
    """

    price: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    n_paths: int | None
    n_steps: int | None
    method: str
    elapsed_s: float
    reference: float | None = None
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_stochastic(self) -> bool:
        """Whether the result carries a Monte Carlo standard error."""
        return self.std_error is not None

    def with_reference(self, reference: float) -> PricingResult:
        """Return a copy with ``reference`` attached."""
        return replace(self, reference=float(reference))

    def error(self, reference: float | None = None) -> float:
        """Signed error ``price - reference``."""
        ref = self._resolve_reference(reference)
        return self.price - ref

    def error_in_se(self, reference: float | None = None) -> float:
        """Signed error expressed in standard errors."""
        if self.std_error is None:
            raise ValueError("deterministic result has no standard error")
        err = self.error(reference)
        if self.std_error == 0.0:
            return 0.0 if err == 0.0 else math.copysign(math.inf, err)
        return err / self.std_error

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Normal confidence interval at ``level``."""
        if self.std_error is None:
            raise ValueError("deterministic result has no confidence interval")
        half = z_value(level) * self.std_error
        return self.price - half, self.price + half

    def contains(self, reference: float | None = None, level: float = 0.99) -> bool:
        """Whether ``reference`` lies inside the ``level`` confidence interval."""
        low, high = self.confidence_interval(level)
        return low <= self._resolve_reference(reference) <= high

    def _resolve_reference(self, reference: float | None) -> float:
        ref = self.reference if reference is None else reference
        if ref is None:
            raise ValueError("no reference price available")
        return float(ref)

    def __str__(self) -> str:
        if self.std_error is None:
            return f"{self.method}: {self.price:.6f}"
        return (
            f"{self.method}: {self.price:.6f} +/- {self.std_error:.6f} "
            f"(95% CI [{self.ci_low:.6f}, {self.ci_high:.6f}], n={self.n_paths})"
        )


@dataclass(frozen=True)
class GreeksResult:
    """Option sensitivities with optional Monte Carlo standard errors.

    Greeks that an estimator cannot produce are ``None``. Units: ``vega`` and ``rho``
    per unit change of volatility / rate (not per 1 %), ``theta`` per year.
    """

    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None
    method: str
    elapsed_s: float
    std_errors: Mapping[str, float] = field(default_factory=dict)
    n_paths: int | None = None

    def as_dict(self) -> dict[str, float]:
        """Available Greeks as a ``{name: value}`` mapping."""
        values = {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }
        return {k: v for k, v in values.items() if v is not None}
