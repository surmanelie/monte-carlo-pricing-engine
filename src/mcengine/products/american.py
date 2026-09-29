r"""American and Bermudan options on a grid of exercise dates.

The holder may exercise at any date of ``exercise_times`` (equally spaced,
``t_i = iT/M``) and receives :math:`(S_t - K)^+` (call) or :math:`(K - S_t)^+` (put).
As :math:`M \to \infty` the Bermudan price converges to the American price. Such
options are priced by :mod:`mcengine.engines.lsm` (Monte Carlo) and
:mod:`mcengine.engines.tree` (binomial tree); :func:`mcengine.engines.monte_carlo.price_mc`
rejects them because averaging a fixed payoff cannot capture optimal stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class AmericanOption(Product):
    """Bermudan-style American option with ``n_exercise`` equally spaced exercise dates.

    Parameters
    ----------
    strike
        Strike ``K > 0``.
    maturity
        Expiry ``T > 0``.
    option_type
        ``"call"`` or ``"put"``.
    n_exercise
        Number of exercise dates ``M`` in ``(0, T]``; Longstaff & Schwartz (2001) use
        50 per year.
    """

    strike: float
    maturity: float
    option_type: str = "put"
    n_exercise: int = 50

    is_path_dependent: ClassVar[bool] = True
    name: ClassVar[str] = "american"

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_positive("maturity", self.maturity)
        require_choice("option_type", self.option_type, OPTION_TYPES)
        require_int_at_least("n_exercise", self.n_exercise, 1)

    def monitoring_times(self) -> FloatArray:
        return uniform_monitoring(self.maturity, self.n_exercise)

    def exercise_value(self, spot: FloatArray) -> FloatArray:
        """Immediate exercise value ``(S - K)^+`` or ``(K - S)^+``."""
        return vanilla_payoff(np.asarray(spot, dtype=np.float64), self.strike, self.option_type)

    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Exercise value at maturity (the payoff if the option is held to expiry)."""
        return self.exercise_value(paths[:, time_indices(times, np.array([self.maturity]))[0]])

    @property
    def label(self) -> str:
        return (
            f"American {self.option_type} K={self.strike:g} T={self.maturity:g} "
            f"({self.n_exercise} dates)"
        )
