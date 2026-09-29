r"""Discretely monitored Asian (average-price) options.

With fixing dates :math:`t_i = iT/m,\ i = 1..m`:

* arithmetic average :math:`A = \frac1m\sum_{i=1}^m S_{t_i}`,
* geometric average :math:`G = \big(\prod_{i=1}^m S_{t_i}\big)^{1/m}`,

and payoff :math:`(A - K)^+` (call) or :math:`(K - A)^+` (put), likewise for :math:`G`.
The geometric option has a closed form under GBM (Kemna & Vorst, 1990) and is the
classic control variate for the arithmetic one.

References
----------
Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on
average asset values. *Journal of Banking & Finance*, 14(1), 113-129.
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

AVERAGE_TYPES: tuple[str, ...] = ("arithmetic", "geometric")


@dataclass(frozen=True)
class AsianOption(Product):
    """Fixed-strike Asian option on the discrete average of ``n_fixings`` prices.

    Parameters
    ----------
    strike
        Strike ``K > 0``.
    maturity
        Expiry ``T > 0`` in years.
    n_fixings
        Number of equally spaced averaging dates ``m >= 1`` (``t_i = iT/m``).
    option_type
        ``"call"`` or ``"put"``.
    average
        ``"arithmetic"`` or ``"geometric"``.
    """

    strike: float
    maturity: float
    n_fixings: int = 12
    option_type: str = "call"
    average: str = "arithmetic"

    is_path_dependent: ClassVar[bool] = True
    name: ClassVar[str] = "asian"

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_positive("maturity", self.maturity)
        require_int_at_least("n_fixings", self.n_fixings, 1)
        require_choice("option_type", self.option_type, OPTION_TYPES)
        require_choice("average", self.average, AVERAGE_TYPES)

    def monitoring_times(self) -> FloatArray:
        return uniform_monitoring(self.maturity, self.n_fixings)

    def average_price(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Arithmetic or geometric average over the fixing dates."""
        fixings = paths[:, time_indices(times, self.monitoring_times())]
        if self.average == "arithmetic":
            return np.asarray(fixings.mean(axis=1), dtype=np.float64)
        return np.asarray(np.exp(np.log(fixings).mean(axis=1)), dtype=np.float64)

    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        return vanilla_payoff(self.average_price(paths, times), self.strike, self.option_type)

    def with_average(self, average: str) -> AsianOption:
        """Same contract with another averaging rule."""
        return AsianOption(self.strike, self.maturity, self.n_fixings, self.option_type, average)

    @property
    def label(self) -> str:
        return (
            f"Asian {self.average[:5]}. {self.option_type} K={self.strike:g} "
            f"T={self.maturity:g} m={self.n_fixings}"
        )
