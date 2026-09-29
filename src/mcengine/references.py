"""Independent reference prices for any supported ``(model, product)`` pair.

Used by the CLI and the dashboard to put every Monte Carlo price next to a price computed
by a different method. Returns ``None`` when no independent method is implemented (e.g.
barrier options under Heston).
"""

from __future__ import annotations

from dataclasses import dataclass

from mcengine.engines.analytic import (
    barrier_price,
    barrier_price_discrete_bgk,
    price_analytic,
)
from mcengine.engines.convolution import asian_arithmetic_price
from mcengine.engines.fourier import gil_pelaez_price
from mcengine.engines.tree import price_tree
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BarrierOption
from mcengine.products.base import Product
from mcengine.products.european import DigitalOption, EuropeanOption


@dataclass(frozen=True)
class Reference:
    """A reference value, the method that produced it, and whether it is exact."""

    value: float
    method: str
    exact: bool = True


def reference_price(model: Model, product: Product) -> Reference | None:
    """Best available independent reference for ``product`` under ``model``."""
    if isinstance(model, GBM):
        return _gbm_reference(model, product)
    if isinstance(model, Heston | Merton) and isinstance(product, EuropeanOption):
        if isinstance(model, Merton):
            return Reference(price_analytic(model, product).price, "Merton series")
        value = gil_pelaez_price(model, product.strike, product.maturity, product.option_type)
        return Reference(value, "Gil-Pelaez Fourier")
    return None


def _gbm_reference(model: GBM, product: Product) -> Reference | None:
    if isinstance(product, EuropeanOption | DigitalOption):
        return Reference(price_analytic(model, product).price, "Black-Scholes")
    if isinstance(product, AsianOption):
        if product.average == "geometric":
            return Reference(price_analytic(model, product).price, "Kemna-Vorst")
        return Reference(asian_arithmetic_price(model, product), "Recursive convolution")
    if isinstance(product, BarrierOption):
        args = (model.s0, product.strike, product.barrier, product.maturity, model.r, model.sigma)
        kinds = (product.barrier_type, product.option_type)
        if product.correction == "bridge":
            return Reference(
                barrier_price(*args, model.q, *kinds), "Reiner-Rubinstein (continuous)"
            )
        value = barrier_price_discrete_bgk(*args, product.n_monitoring, model.q, *kinds)
        return Reference(value, "BGK approximation (discrete)", exact=False)
    if isinstance(product, AmericanOption):
        # >= 5000 steps, a multiple of 2 x n_exercise so that Richardson's N/2 tree also
        # places every exercise date on its grid
        per_date = max(2, -(-5000 // product.n_exercise))
        steps = product.n_exercise * (per_date + per_date % 2)
        return Reference(price_tree(model, product, n_steps=steps).price, f"CRR tree (N={steps})")
    return None
