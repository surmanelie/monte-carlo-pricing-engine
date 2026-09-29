r"""Single-barrier knock-in / knock-out options with discrete monitoring.

The barrier :math:`H` is checked on ``n_monitoring`` equally spaced dates. Three
payoff estimators are available (``correction``):

``"none"``
    raw discrete monitoring: the option knocks out (in) if some monitored price is
    beyond :math:`H`. Converges to the continuous price with an :math:`O(\sqrt{\Delta t})`
    bias.
``"bgk"``
    Broadie-Glasserman-Kou (1997) continuity correction: a discretely monitored option
    with barrier :math:`H` is priced like a continuously monitored one with barrier
    :math:`He^{\pm\beta\sigma\sqrt{\Delta t}}`, :math:`\beta = -\zeta(1/2)/\sqrt{2\pi}
    \approx 0.5826`. To estimate the *continuous* price from a discrete simulation, the
    monitored barrier is moved towards the spot: :math:`He^{+\beta\sigma\sqrt{\Delta t}}`
    for down barriers, :math:`He^{-\beta\sigma\sqrt{\Delta t}}` for up barriers.
``"bridge"``
    Brownian-bridge estimator of *continuous* monitoring: conditionally on the simulated
    grid, the log-price between two nodes is a Brownian bridge, whose probability of
    not crossing :math:`H` is

    .. math:: 1 - \exp\!\Big(-\frac{2\ln(S_i/H)\ln(S_{i+1}/H)}{\sigma^2\Delta t}\Big)

    (Glasserman 2003, Section 6.4). The knock-out payoff is the vanilla payoff times the
    product of these survival probabilities; it is unbiased for the continuously
    monitored price under GBM.

``correction_sigma`` is the (GBM) volatility used by the ``bgk`` and ``bridge``
corrections. Knock-in payoffs are ``vanilla - knock-out`` path by path, so in-out
parity holds exactly for every estimator.

References
----------
Broadie, M., Glasserman, P., & Kou, S. (1997). A continuity correction for discrete
barrier options. *Mathematical Finance*, 7(4), 325-349.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import require_choice, require_int_at_least, require_positive
from mcengine.products.base import (
    OPTION_TYPES,
    Product,
    time_indices,
    uniform_monitoring,
    vanilla_payoff,
)

BARRIER_TYPES: tuple[str, ...] = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")
CORRECTIONS: tuple[str, ...] = ("none", "bgk", "bridge")
#: Broadie-Glasserman-Kou constant ``-zeta(1/2) / sqrt(2 pi)``.
BGK_BETA = 0.5825971579390106


@dataclass(frozen=True)
class BarrierOption(Product):
    """Single-barrier European option.

    Parameters
    ----------
    strike
        Strike ``K > 0``.
    maturity
        Expiry ``T > 0``.
    barrier
        Barrier level ``H > 0``; must lie below the spot for down barriers and above it
        for up barriers (checked when paths are priced).
    barrier_type
        One of ``down-and-out``, ``down-and-in``, ``up-and-out``, ``up-and-in``.
    option_type
        ``"call"`` or ``"put"``.
    n_monitoring
        Number of equally spaced monitoring dates.
    correction
        ``"none"``, ``"bgk"`` or ``"bridge"`` (see module docstring).
    correction_sigma
        Volatility used by the corrections (required unless ``correction="none"``).
    """

    strike: float
    maturity: float
    barrier: float
    barrier_type: str = "down-and-out"
    option_type: str = "call"
    n_monitoring: int = 252
    correction: str = "none"
    correction_sigma: float | None = None

    is_path_dependent: ClassVar[bool] = True
    name: ClassVar[str] = "barrier"

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_positive("maturity", self.maturity)
        require_positive("barrier", self.barrier)
        require_choice("barrier_type", self.barrier_type, BARRIER_TYPES)
        require_choice("option_type", self.option_type, OPTION_TYPES)
        require_int_at_least("n_monitoring", self.n_monitoring, 1)
        require_choice("correction", self.correction, CORRECTIONS)
        if self.correction != "none":
            if self.correction_sigma is None:
                raise ValueError(f"correction {self.correction!r} requires correction_sigma")
            require_positive("correction_sigma", self.correction_sigma)

    @property
    def is_down(self) -> bool:
        """Whether the barrier is below the spot."""
        return self.barrier_type.startswith("down")

    @property
    def is_knock_in(self) -> bool:
        """Whether the option is activated (rather than extinguished) by the barrier."""
        return self.barrier_type.endswith("in")

    @property
    def parity_partner(self) -> BarrierOption:
        """The knock-out (knock-in) twin with otherwise identical terms."""
        swapped = self.barrier_type.replace("-in", "-tmp").replace("-out", "-in")
        return replace(self, barrier_type=swapped.replace("-tmp", "-out"))

    def monitoring_times(self) -> FloatArray:
        return uniform_monitoring(self.maturity, self.n_monitoring)

    def check_spot(self, s0: float) -> None:
        """Raise if the spot is not strictly on the live side of the barrier."""
        if (self.is_down and s0 <= self.barrier) or (not self.is_down and s0 >= self.barrier):
            side = "above" if self.is_down else "below"
            raise ValueError(
                f"{self.barrier_type} barrier {self.barrier:g} requires a spot strictly {side} "
                f"it, got {s0:g}"
            )

    def survival(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Probability-weighted indicator that the barrier has *not* been hit."""
        self.check_spot(float(paths[0, 0]))
        if self.correction == "bridge":
            return self._bridge_survival(paths, times)
        barrier = self.barrier
        if self.correction == "bgk":
            assert self.correction_sigma is not None
            dt = self.maturity / self.n_monitoring
            shift = BGK_BETA * self.correction_sigma * math.sqrt(dt)
            barrier *= math.exp(shift if self.is_down else -shift)
        monitored = paths[:, time_indices(times, self.monitoring_times())]
        alive = (
            np.all(monitored > barrier, axis=1)
            if self.is_down
            else np.all(monitored < barrier, axis=1)
        )
        return alive.astype(np.float64)

    def _bridge_survival(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        assert self.correction_sigma is not None
        log_dist = np.log(paths / self.barrier)
        if not self.is_down:
            log_dist = -log_dist
        # log_dist > 0 on the live side. Survive a step with 1 - exp(-2 a b / (s^2 dt)).
        dt = np.diff(times)
        a, b = log_dist[:, :-1], log_dist[:, 1:]
        live = (a > 0.0) & (b > 0.0)
        exponent = -2.0 * np.where(live, a * b, 0.0) / (self.correction_sigma**2 * dt)
        step_survival = np.where(live, -np.expm1(exponent), 0.0)
        return np.asarray(np.prod(step_survival, axis=1), dtype=np.float64)

    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        spot = paths[:, time_indices(times, np.array([self.maturity]))[0]]
        vanilla = vanilla_payoff(spot, self.strike, self.option_type)
        alive = self.survival(paths, times)
        return vanilla * (1.0 - alive) if self.is_knock_in else vanilla * alive

    @property
    def label(self) -> str:
        return (
            f"{self.barrier_type} {self.option_type} K={self.strike:g} H={self.barrier:g} "
            f"T={self.maturity:g}"
        )
