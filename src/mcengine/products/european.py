r"""European vanilla and cash-or-nothing digital options.

Payoffs at maturity :math:`T`:

* call :math:`(S_T - K)^+`, put :math:`(K - S_T)^+` (Black & Scholes, 1973);
* digital call :math:`Q\,\mathbf 1\{S_T > K\}`, digital put :math:`Q\,\mathbf 1\{S_T < K\}`
  (cash-or-nothing, payout :math:`Q`; Reiner & Rubinstein, 1991b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import require_choice, require_positive
from mcengine.products.base import OPTION_TYPES, Product, time_indices, vanilla_payoff


@dataclass(frozen=True)
class EuropeanOption(Product):
    """European call or put.

    Parameters
    ----------
    strike
        Strike price ``K > 0``.
    maturity
        Time to expiry ``T > 0`` in years.
    option_type
        ``"call"`` or ``"put"``.
    """

    strike: float
    maturity: float
    option_type: str = "call"

    is_path_dependent: ClassVar[bool] = False
    name: ClassVar[str] = "european"

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_positive("maturity", self.maturity)
        require_choice("option_type", self.option_type, OPTION_TYPES)

    def monitoring_times(self) -> FloatArray:
        return np.array([self.maturity])

    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        spot = paths[:, time_indices(times, np.array([self.maturity]))[0]]
        return vanilla_payoff(spot, self.strike, self.option_type)

    @property
    def label(self) -> str:
        return f"European {self.option_type} K={self.strike:g} T={self.maturity:g}"


@dataclass(frozen=True)
class DigitalOption(Product):
    """Cash-or-nothing digital option paying ``payout`` if it finishes in the money."""

    strike: float
    maturity: float
    option_type: str = "call"
    payout: float = 1.0

    is_path_dependent: ClassVar[bool] = False
    name: ClassVar[str] = "digital"

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_positive("maturity", self.maturity)
        require_positive("payout", self.payout)
        require_choice("option_type", self.option_type, OPTION_TYPES)

    def monitoring_times(self) -> FloatArray:
        return np.array([self.maturity])

    def payoff(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        spot = paths[:, time_indices(times, np.array([self.maturity]))[0]]
        itm = spot > self.strike if self.option_type == "call" else spot < self.strike
        return np.where(itm, self.payout, 0.0)

    @property
    def label(self) -> str:
        return f"Digital {self.option_type} K={self.strike:g} T={self.maturity:g}"
