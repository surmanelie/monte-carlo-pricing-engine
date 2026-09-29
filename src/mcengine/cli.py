"""Command-line interface: ``mcengine price | validate | figures | benchmark | calibrate``.

Examples
--------
.. code-block:: console

    mcengine price --model gbm --product european --type call --K 100 --n 100000
    mcengine price --model heston --product barrier --type down-and-out-call --K 100 --B 90 \\
        --n 1000000 --method qmc
    mcengine validate --quick
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mcengine import __version__
from mcengine.engines.lsm import price_lsm
from mcengine.engines.monte_carlo import price_mc, qmc_path_count
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BARRIER_TYPES, BarrierOption
from mcengine.products.base import Product
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.references import reference_price
from mcengine.results import PricingResult

app = typer.Typer(
    name="mcengine",
    help="Monte Carlo derivatives pricing engine: price, validate, reproduce figures.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


class ModelName(StrEnum):
    """Supported models."""

    gbm = "gbm"
    heston = "heston"
    merton = "merton"


class ProductName(StrEnum):
    """Supported products."""

    european = "european"
    digital = "digital"
    asian = "asian"
    barrier = "barrier"
    american = "american"


class MethodName(StrEnum):
    """Supported estimators."""

    plain = "plain"
    antithetic = "antithetic"
    cv = "cv"
    is_ = "is"
    qmc = "qmc"
    lsm = "lsm"


def build_model(
    model: ModelName,
    s0: float,
    r: float,
    q: float,
    sigma: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float,
    mu_j: float,
    delta_j: float,
    scheme: str = "qe",
    backend: str = "numpy",
) -> Model:
    """Instantiate the requested model from CLI options."""
    if model is ModelName.gbm:
        return GBM(s0, r, sigma, q)
    if model is ModelName.heston:
        return Heston(s0, r, v0, kappa, theta, xi, rho, q, scheme=scheme, backend=backend)
    return Merton(s0, r, sigma, lam, mu_j, delta_j, q)


def build_product(
    product: ProductName,
    option_type: str,
    strike: float,
    maturity: float,
    barrier: float | None,
    fixings: int,
    monitoring: int,
    exercise_dates: int,
    correction: str,
    sigma: float | None,
) -> Product:
    """Instantiate the requested product; barrier types look like ``down-and-out-call``."""
    if product is ProductName.european:
        return EuropeanOption(strike, maturity, option_type)
    if product is ProductName.digital:
        return DigitalOption(strike, maturity, option_type)
    if product is ProductName.asian:
        return AsianOption(strike, maturity, fixings, option_type)
    if product is ProductName.american:
        return AmericanOption(strike, maturity, option_type, exercise_dates)
    barrier_type, _, kind = option_type.rpartition("-")
    if barrier_type not in BARRIER_TYPES or kind not in ("call", "put"):
        raise typer.BadParameter(
            f"barrier --type must look like 'down-and-out-call', got {option_type!r}"
        )
    if barrier is None:
        raise typer.BadParameter("barrier options need --B")
    corr_sigma = sigma if correction != "none" else None
    return BarrierOption(
        strike, maturity, barrier, barrier_type, kind, monitoring, correction, corr_sigma
    )


def result_table(result: PricingResult, title: str) -> Table:
    """Rich table with price, standard error, confidence interval and reference."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Quantity")
    table.add_column("Value", justify="right")
    table.add_row("Method", result.method)
    table.add_row("Price", f"{result.price:.6f}")
    if result.std_error is not None:
        low, high = result.confidence_interval(0.95)
        table.add_row("Standard error", f"{result.std_error:.6f}")
        table.add_row("95 % CI", f"[{low:.6f}, {high:.6f}]")
    table.add_row("Paths", f"{result.n_paths:,}" if result.n_paths else "-")
    table.add_row("Time steps", str(result.n_steps) if result.n_steps else "-")
    for key, value in result.diagnostics.items():
        shown = f"{int(value):,}" if float(value).is_integer() else f"{value:.4g}"
        table.add_row(key, shown)
    table.add_row("Elapsed", f"{result.elapsed_s:.3f} s")
    return table


@app.command()
def price(
    model: Annotated[ModelName, typer.Option(help="Asset model.")] = ModelName.gbm,
    product: Annotated[ProductName, typer.Option(help="Payoff family.")] = ProductName.european,
    option_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="call / put; for barriers e.g. down-and-out-call, up-and-in-put.",
        ),
    ] = "call",
    strike: Annotated[float, typer.Option("--K", help="Strike.")] = 100.0,
    barrier: Annotated[float | None, typer.Option("--B", help="Barrier level.")] = None,
    maturity: Annotated[float, typer.Option("--T", help="Maturity in years.")] = 1.0,
    s0: Annotated[float, typer.Option("--S0", help="Spot price.")] = 100.0,
    r: Annotated[float, typer.Option("--r", help="Risk-free rate.")] = 0.05,
    q: Annotated[float, typer.Option("--q", help="Dividend yield.")] = 0.0,
    sigma: Annotated[float, typer.Option(help="GBM / Merton diffusion volatility.")] = 0.2,
    v0: Annotated[float, typer.Option(help="Heston initial variance.")] = 0.04,
    kappa: Annotated[float, typer.Option(help="Heston mean reversion.")] = 2.0,
    theta: Annotated[float, typer.Option(help="Heston long-run variance.")] = 0.04,
    xi: Annotated[float, typer.Option(help="Heston vol of variance.")] = 0.5,
    rho: Annotated[float, typer.Option(help="Heston correlation.")] = -0.7,
    lam: Annotated[float, typer.Option(help="Merton jump intensity.")] = 1.0,
    mu_j: Annotated[float, typer.Option(help="Merton mean log-jump.")] = -0.1,
    delta_j: Annotated[float, typer.Option(help="Merton log-jump volatility.")] = 0.15,
    n: Annotated[int, typer.Option("--n", help="Number of paths.")] = 100_000,
    method: Annotated[MethodName, typer.Option(help="Estimator.")] = MethodName.plain,
    steps: Annotated[
        int | None, typer.Option(help="Time steps (default: monitoring dates / 100 for Heston).")
    ] = None,
    fixings: Annotated[int, typer.Option(help="Asian averaging dates.")] = 12,
    monitoring: Annotated[int, typer.Option(help="Barrier monitoring dates.")] = 252,
    exercise_dates: Annotated[int, typer.Option(help="American exercise dates.")] = 50,
    correction: Annotated[
        str, typer.Option(help="Barrier estimator: none, bgk or bridge (GBM volatility).")
    ] = "none",
    scheme: Annotated[str, typer.Option(help="Heston scheme: qe or euler.")] = "qe",
    backend: Annotated[str, typer.Option(help="numpy or numba (Heston QE, LSM).")] = "numpy",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Price one option and compare it with an independent reference."""
    try:
        mdl = build_model(
            model, s0, r, q, sigma, v0, kappa, theta, xi, rho, lam, mu_j, delta_j, scheme,
            backend if model is ModelName.heston else "numpy",
        )  # fmt: skip
        prod = build_product(
            product, option_type, strike, maturity, barrier, fixings, monitoring,
            exercise_dates, correction, sigma,
        )  # fmt: skip
        if isinstance(prod, AmericanOption) or method is MethodName.lsm:
            if not isinstance(prod, AmericanOption):
                raise typer.BadParameter("--method lsm prices American options only")
            result = price_lsm(mdl, prod, n_paths=n, seed=seed, backend=backend)
        else:
            n_steps = steps
            if n_steps is None and not mdl.exact_simulation:
                n_steps = max(prod.monitoring_times().size, round(100 * maturity))
            n_paths = n
            if method is MethodName.qmc:
                lower = qmc_path_count(n)
                n_paths = lower if n - lower <= 2 * lower - n else 2 * lower
            result = price_mc(
                mdl, prod, n_paths=n_paths, method=method.value, n_steps=n_steps, seed=seed
            )
        ref = reference_price(mdl, prod)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    table = result_table(result, f"{prod.label} under {type(mdl).__name__}")
    if ref is None:
        table.add_row("Reference", "no independent method for this model/product")
    else:
        table.add_row("Reference", f"{ref.value:.6f} ({ref.method})")
        if result.std_error:
            table.add_row("Error (SE)", f"{result.error_in_se(ref.value):+.2f}")
            inside = result.contains(ref.value, 0.99)
            note = "" if ref.exact else " (approximate reference)"
            table.add_row("Inside 99 % CI", ("yes" if inside else "no") + note)
    console.print(table)


@app.command()
def validate(
    quick: Annotated[bool, typer.Option(help="10x fewer paths (smoke run).")] = False,
    filter_: Annotated[str, typer.Option("--filter", help="Only matching case ids.")] = "",
    out: Annotated[Path, typer.Option(help="Output directory for results.")] = Path("results"),
    coverage: Annotated[bool, typer.Option(help="Also run the CI coverage study.")] = True,
) -> None:
    """Run the validation table (every MC estimator vs an independent reference)."""
    from mcengine.reporting.validation import (
        all_cases,
        coverage_cases,
        coverage_markdown,
        coverage_study,
        run_validation,
        write_outputs,
    )

    cases = [c for c in all_cases() if filter_ in c.case_id]
    rows = run_validation(
        cases,
        scale=0.1 if quick else 1.0,
        progress=lambda row: console.print(
            f"{'[green]PASS[/green]' if row.passed else '[red]FAIL[/red]'} {row.case_id} "
            f"{row.error_se:+.2f} SE"
        ),
    )
    table = Table(title="Validation", header_style="bold")
    for col in ("Model", "Product", "Method", "Price", "Reference", "Error (SE)", "Pass"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            row.model, row.product, row.method, f"{row.price:.4f}", f"{row.reference:.4f}",
            f"{row.error_se:+.2f}", "yes" if row.passed else "NO",
        )  # fmt: skip
    console.print(table)
    if not quick and not filter_:
        write_outputs(rows, out)
        if coverage:
            (out / "coverage.md").write_text(
                coverage_markdown(coverage_study(coverage_cases())), encoding="utf-8"
            )
        console.print(f"wrote {out / 'validation.md'}")
    if not all(row.passed for row in rows):
        raise typer.Exit(code=1)


@app.command()
def figures(
    quick: Annotated[bool, typer.Option(help="Small experiments (smoke run).")] = False,
    only: Annotated[list[str] | None, typer.Option(help="Generate only these figures.")] = None,
    out: Annotated[Path, typer.Option(help="Figure directory.")] = Path("figures"),
    results: Annotated[Path, typer.Option(help="Directory for measured numbers.")] = Path(
        "results"
    ),
) -> None:
    """Regenerate every figure (and the numbers quoted in the docs)."""
    from mcengine.reporting.figures import make_all

    make_all(
        out,
        results_dir=None if quick else results,
        quick=quick,
        only=only,
        progress=lambda name: console.print(f"figure: {name}"),
    )


@app.command()
def benchmark(
    quick: Annotated[bool, typer.Option(help="Small runs (smoke test).")] = False,
    out: Annotated[Path, typer.Option(help="Figure directory.")] = Path("figures"),
    results: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
) -> None:
    """Throughput, error-vs-time and efficiency benchmarks."""
    from mcengine.reporting.benchmark import (
        EfficiencyRow,
        ThroughputRow,
        efficiency_markdown,
        run_all,
        throughput_markdown,
    )

    report = run_all(out, None if quick else results, quick)
    tp = [ThroughputRow(**r) for r in report["throughput"]]  # type: ignore[attr-defined]
    ef = [EfficiencyRow(**r) for r in report["efficiency"]]  # type: ignore[attr-defined]
    console.print(throughput_markdown(tp))
    console.print(efficiency_markdown(ef))


@app.command()
def calibrate(
    csv: Annotated[Path, typer.Option(help="Option-chain CSV (see the documented schema).")],
    spot: Annotated[float, typer.Option(help="Spot price of the underlying.")],
    rate: Annotated[float, typer.Option(help="Continuously compounded rate.")] = 0.0,
    dividend: Annotated[float, typer.Option(help="Dividend yield.")] = 0.0,
    starts: Annotated[int, typer.Option(help="Number of optimiser starting points.")] = 4,
) -> None:
    """Calibrate Heston to a user-provided option chain (optional; never downloads data)."""
    from mcengine.calibration.heston_calibration import calibrate_heston, load_option_chain_csv

    try:
        surface = load_option_chain_csv(csv, spot, rate, dividend)
    except (OSError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    fit = calibrate_heston(surface, n_starts=starts)
    table = Table(title=f"Heston calibration ({surface.size} quotes)", header_style="bold")
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    for name, value in fit.params.items():
        table.add_row(name, f"{value:.6f}")
    table.add_row("RMSE (vol)", f"{fit.rmse_vol:.6f}")
    table.add_row("Max |error| (vol)", f"{fit.max_abs_error_vol:.6f}")
    table.add_row("Feller ratio 2 kappa theta / xi^2", f"{fit.feller_ratio:.4f}")
    table.add_row("Converged", str(fit.success))
    console.print(table)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
