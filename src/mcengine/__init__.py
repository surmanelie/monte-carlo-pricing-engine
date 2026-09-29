"""mcengine: Monte Carlo derivatives pricing and hedging engine.

Every Monte Carlo estimator in the package is validated against an independent
reference (closed form, Fourier inversion, binomial tree); see ``mcengine validate``.
"""

from __future__ import annotations

from mcengine.engines.analytic import bs_digital_price, bs_price, price_analytic
from mcengine.engines.monte_carlo import price_mc
from mcengine.greeks.analytic import bs_greeks
from mcengine.models.gbm import GBM
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.results import GreeksResult, PricingResult

__version__ = "0.3.0"

__all__ = [
    "GBM",
    "DigitalOption",
    "EuropeanOption",
    "GreeksResult",
    "PricingResult",
    "__version__",
    "bs_digital_price",
    "bs_greeks",
    "bs_price",
    "price_analytic",
    "price_mc",
]
