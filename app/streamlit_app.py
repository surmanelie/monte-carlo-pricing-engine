"""Interactive dashboard for mcengine.

Run locally with ``streamlit run app/streamlit_app.py`` (after ``pip install -e ".[app]"``).
Every Monte Carlo price is shown next to an independent reference whenever one exists.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mcengine import __version__
from mcengine.cli import MethodName, ModelName, ProductName, build_model, build_product
from mcengine.engines.fourier import fourier_prices
from mcengine.engines.lsm import price_lsm
from mcengine.engines.monte_carlo import build_time_grid, price_mc, qmc_path_count
from mcengine.hedging.delta_hedge import hedging_experiment
from mcengine.models.base import Model
from mcengine.products.american import AmericanOption
from mcengine.products.base import Product
from mcengine.references import reference_price
from mcengine.reporting.style import INK, PALETTE
from mcengine.results import PricingResult
from mcengine.volatility.implied import implied_vol

st.set_page_config(page_title="mcengine dashboard", layout="wide")

# ----------------------------------------------------------------------------- inputs
sb = st.sidebar
sb.title("mcengine")
sb.caption(f"Monte Carlo pricing engine v{__version__}")
model_name = ModelName(sb.selectbox("Model", [m.value for m in ModelName]))
s0 = sb.number_input("Spot S0", 1.0, 1e4, 100.0)
r = sb.number_input("Rate r", -0.05, 0.3, 0.05, step=0.005, format="%.3f")
q = sb.number_input("Dividend yield q", 0.0, 0.3, 0.0, step=0.005, format="%.3f")
sigma, v0, kappa, theta, xi, rho, lam, mu_j, delta_j = (
    0.2,
    0.04,
    2.0,
    0.04,
    0.5,
    -0.7,
    1.0,
    -0.1,
    0.15,
)
if model_name in (ModelName.gbm, ModelName.merton):
    sigma = sb.slider("Volatility sigma", 0.05, 1.0, 0.2, 0.01)
if model_name is ModelName.heston:
    v0 = sb.slider("v0", 0.001, 0.25, 0.04, 0.001, format="%.3f")
    kappa = sb.slider("kappa", 0.1, 10.0, 2.0, 0.1)
    theta = sb.slider("theta", 0.001, 0.25, 0.04, 0.001, format="%.3f")
    xi = sb.slider("xi (vol of vol)", 0.05, 2.0, 0.5, 0.05)
    rho = sb.slider("rho", -0.99, 0.99, -0.7, 0.01)
if model_name is ModelName.merton:
    lam = sb.slider("Jump intensity lambda", 0.0, 5.0, 1.0, 0.1)
    mu_j = sb.slider("Mean log-jump", -0.5, 0.3, -0.1, 0.01)
    delta_j = sb.slider("Log-jump volatility", 0.0, 0.5, 0.15, 0.01)

product_name = ProductName(sb.selectbox("Product", [p.value for p in ProductName]))
strike = sb.number_input("Strike K", 1.0, 1e4, 100.0)
maturity = sb.number_input("Maturity T (years)", 0.05, 10.0, 1.0, step=0.25)
barrier_level, fixings, monitoring, correction, exercise_dates = None, 12, 252, "none", 50
if product_name is ProductName.barrier:
    kind = sb.selectbox(
        "Barrier type",
        [
            f"{b}-{o}"
            for b in ("down-and-out", "down-and-in", "up-and-out", "up-and-in")
            for o in ("call", "put")
        ],
    )
    default_level = 0.9 * s0 if kind.startswith("down") else 1.2 * s0
    barrier_level = sb.number_input("Barrier B", 1.0, 1e4, float(default_level))
    monitoring = sb.slider("Monitoring dates", 4, 504, 252)
    choices = ["none", "bgk", "bridge"] if model_name is ModelName.gbm else ["none"]
    correction = sb.selectbox("Estimator", choices, help="bgk / bridge need the GBM volatility")
else:
    kind = sb.selectbox("Option type", ["call", "put"])
if product_name is ProductName.asian:
    fixings = sb.slider("Averaging dates", 1, 252, 12)
if product_name is ProductName.american:
    exercise_dates = sb.slider("Exercise dates", 2, 252, 50)
    methods = [MethodName.lsm.value]
else:
    methods = [m.value for m in MethodName if m is not MethodName.lsm]
method = MethodName(sb.selectbox("Method", methods))
n_paths = int(sb.select_slider("Paths", [2**k for k in range(10, 21)], 2**16))
steps = sb.slider("Time steps (Heston)", 4, 504, 100) if model_name is ModelName.heston else None
seed = int(sb.number_input("Seed", 0, 2**31 - 1, 42))

model_args = (model_name, s0, r, q, sigma, v0, kappa, theta, xi, rho, lam, mu_j, delta_j)
product_args = (
    product_name,
    kind,
    strike,
    maturity,
    barrier_level,
    fixings,
    monitoring,
    exercise_dates,
    correction,
    sigma,
)


# ----------------------------------------------------------------------------- engine
def _objects() -> tuple[Model, Product]:
    return build_model(*model_args), build_product(*product_args)


@st.cache_data(show_spinner=False)
def run_pricing(
    model_args: tuple[Any, ...],
    product_args: tuple[Any, ...],
    method: str,
    n: int,
    n_steps: int | None,
    seed: int,
) -> PricingResult:
    """Price with the selected estimator (cached on all inputs)."""
    model, product = build_model(*model_args), build_product(*product_args)
    if isinstance(product, AmericanOption):
        return price_lsm(model, product, n_paths=n, seed=seed)
    if method == "qmc":
        n = qmc_path_count(n)
    return price_mc(model, product, n_paths=n, method=method, n_steps=n_steps, seed=seed)


@st.cache_data(show_spinner=False)
def run_reference(
    model_args: tuple[Any, ...], product_args: tuple[Any, ...]
) -> tuple[float, str] | None:
    """Independent reference price, if any."""
    ref = reference_price(build_model(*model_args), build_product(*product_args))
    return None if ref is None else (ref.value, ref.method + ("" if ref.exact else ", approximate"))


try:
    model, product = _objects()
    with st.spinner("Simulating..."):
        result = run_pricing(model_args, product_args, method.value, n_paths, steps, seed)
        reference = run_reference(model_args, product_args)
except ValueError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()

# ----------------------------------------------------------------------------- headline
st.title(product.label)
st.caption(
    f"{type(model).__name__} model, estimator `{result.method}`, "
    f"{result.n_paths:,} paths, {result.n_steps} time steps, {result.elapsed_s:.2f} s"
)
cols = st.columns(4)
se = result.std_error or 0.0
cols[0].metric("Monte Carlo price", f"{result.price:.4f}", help="Discounted payoff average")
cols[1].metric("95 % confidence interval", f"± {1.96 * se:.4f}")
if reference is None:
    cols[2].metric("Reference", "n/a", help="No independent method for this model/product")
else:
    cols[2].metric("Reference", f"{reference[0]:.4f}", help=reference[1])
    err_se = (result.price - reference[0]) / se if se > 0 else 0.0
    cols[3].metric(
        "Error vs reference", f"{err_se:+.2f} SE", help="Inside the 99 % CI when |error| < 2.58 SE"
    )
    st.caption(f"Reference: {reference[1]}")
if result.diagnostics:
    st.caption(" · ".join(f"{k} = {v:.4g}" for k, v in result.diagnostics.items()))

tab_conv, tab_paths, tab_smile, tab_hedge = st.tabs(
    ["Convergence", "Sample paths", "Implied-volatility smile", "Hedging P&L"]
)


def _layout(fig: go.Figure, x: str, y: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x,
        yaxis_title=y,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=420,
    )
    return fig


with tab_conv:
    counts = [n_paths // 2**k for k in range(6, -1, -1) if n_paths // 2**k >= 1024]
    runs = [
        run_pricing(model_args, product_args, method.value, n, steps, seed + i)
        for i, n in enumerate(counts)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r_.n_paths for r_ in runs],
            y=[r_.price for r_ in runs],
            mode="markers+lines",
            name=result.method,
            line={"color": PALETTE[0]},
            error_y={"type": "data", "array": [1.96 * (r_.std_error or 0) for r_ in runs]},
        )
    )
    if reference is not None:
        fig.add_hline(
            y=reference[0],
            line_dash="dash",
            line_color=INK,
            annotation_text=f"reference ({reference[1]})",
        )
    fig.update_xaxes(type="log")
    st.plotly_chart(_layout(fig, "Number of paths", "Price (95 % CI)"), use_container_width=True)

with tab_paths:
    times = build_time_grid(model, product, steps if not model.exact_simulation else None)
    if times.size < 30:
        times = np.linspace(0.0, maturity, 101)
    paths = model.simulate_paths(times, 40, seed=seed)
    fig = go.Figure()
    for i in range(paths.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=paths[i],
                mode="lines",
                showlegend=False,
                line={"width": 1, "color": PALETTE[i % 3]},
                opacity=0.6,
            )
        )
    fig.add_hline(y=strike, line_dash="dot", line_color=INK, annotation_text="strike")
    if barrier_level is not None:
        fig.add_hline(
            y=barrier_level, line_dash="dash", line_color=PALETTE[7], annotation_text="barrier"
        )
    st.plotly_chart(_layout(fig, "Time (years)", "Price"), use_container_width=True)

with tab_smile:
    if not model.has_char_func:  # pragma: no cover - every shipped model has one
        st.info("This model has no characteristic function.")
    else:
        moneyness = np.linspace(0.7, 1.3, 41)
        fig = go.Figure()
        all_vols = []
        for i, t in enumerate((0.25, 0.5, 1.0, 2.0)):
            prices = fourier_prices(model, s0 * moneyness, t)
            vols = np.asarray(implied_vol(prices, s0, s0 * moneyness, t, r, q))
            all_vols.append(vols)
            fig.add_trace(
                go.Scatter(x=moneyness, y=vols, name=f"T = {t:g}", line={"color": PALETTE[i]})
            )
        lo, hi = np.nanmin(all_vols), np.nanmax(all_vols)
        fig.update_yaxes(range=[lo - 0.02, hi + 0.02], tickformat=".3f")
        st.plotly_chart(
            _layout(fig, "Moneyness K / S0", "Black-Scholes implied volatility"),
            use_container_width=True,
        )
        st.caption("Computed by Fourier inversion (Gil-Pelaez) and implied-volatility inversion.")

with tab_hedge:
    st.write(
        "Short call sold at the Black-Scholes price and delta-hedged with the "
        "Black-Scholes delta at the ATM implied volatility of the model."
    )
    n_reb = st.select_slider("Rebalancing dates N", [1, 2, 4, 12, 21, 42, 126, 252], 42)
    cost = st.slider("Proportional transaction cost (bp)", 0, 50, 0) / 1e4
    atm_price = float(fourier_prices(model, [s0], maturity)[0])
    hedge_vol = float(implied_vol(atm_price, s0, s0, maturity, r, q))

    @st.cache_data(show_spinner=False)
    def hedge(model_args: tuple[Any, ...], n: int, c: float, vol: float) -> np.ndarray:
        """Hedging P&L sample (cached)."""
        res = hedging_experiment(
            build_model(*model_args),
            s0,
            maturity,
            [n],
            hedge_sigma=vol,
            cost_rate=c,
            n_paths=20_000,
            seed=seed,
        )
        return res[0].pnl

    pnl = hedge(model_args, int(n_reb), cost, hedge_vol)
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=80, marker_color=PALETTE[0]))
    st.plotly_chart(_layout(fig, "Discounted P&L per option", "Count"), use_container_width=True)
    st.caption(
        f"mean {pnl.mean():+.4f}, std {pnl.std(ddof=1):.4f} "
        f"(premium {atm_price:.4f}, hedge vol {hedge_vol:.4f}, 20,000 paths)"
    )
    if math.isnan(hedge_vol):  # pragma: no cover
        st.warning("Implied volatility undefined for these inputs.")
