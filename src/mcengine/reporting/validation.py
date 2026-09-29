"""Validation registry: every Monte Carlo estimator against an independent reference.

Each :class:`ValidationCase` runs one Monte Carlo pricing with a seed derived from its
identifier (CRC-32, fixed once and never tuned) and compares it with a reference
computed by a different method (closed form, Fourier inversion, binomial tree).

Two criteria are reported for every row:

* ``in_ci99``: the reference lies inside the 99 % confidence interval;
* ``passed``: ``|MC - reference| < 4 SE`` (the project-wide statistical tolerance).

With many rows, about 1 % of them are expected outside the 99 % interval by chance
even for a perfectly unbiased estimator; seeds are never re-drawn to hide this. The
calibration of the standard errors themselves is checked separately by
:func:`coverage_study`, which measures the empirical coverage of the confidence
intervals over hundreds of independent seeds.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mcengine._numba import HAS_NUMBA
from mcengine.engines.analytic import (
    barrier_price,
    bs_digital_price,
    bs_price,
    geometric_asian_price,
    merton_price,
)
from mcengine.engines.convolution import asian_arithmetic_price
from mcengine.engines.fourier import gil_pelaez_price
from mcengine.engines.lsm import price_lsm
from mcengine.engines.monte_carlo import price_mc, qmc_path_count
from mcengine.engines.tree import price_tree
from mcengine.greeks.analytic import bs_greeks, digital_greeks
from mcengine.greeks.monte_carlo import mc_greeks
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BARRIER_TYPES, BarrierOption
from mcengine.products.base import Product
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.results import PricingResult
from mcengine.stats import z_value

#: Confidence level reported in the ``in_ci99`` column.
CI_LEVEL = 0.99
#: Pass/fail tolerance in standard errors.
PASS_TOL_SE = 4.0


def case_seed(case_id: str) -> int:
    """Deterministic seed derived from the case identifier."""
    return zlib.crc32(case_id.encode("utf-8"))


@dataclass(frozen=True)
class ValidationCase:
    """One validation experiment.

    ``run(seed, scale)`` returns the Monte Carlo result (``scale`` multiplies the path
    count, e.g. ``0.1`` for quick smoke runs); ``reference()`` returns the reference
    value and a label of the method that produced it.
    """

    case_id: str
    model: str
    product: str
    run: Callable[[int, float], PricingResult]
    reference: Callable[[], tuple[float, str]]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationRow:
    """Outcome of one :class:`ValidationCase`."""

    case_id: str
    model: str
    product: str
    method: str
    n_paths: int
    n_steps: int
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    reference: float
    reference_method: str
    error_se: float
    in_ci99: bool
    passed: bool
    elapsed_s: float
    diagnostics: dict[str, float]


def run_case(case: ValidationCase, scale: float = 1.0) -> ValidationRow:
    """Execute ``case`` and compare it with its reference."""
    result = case.run(case_seed(case.case_id), scale)
    ref, ref_method = case.reference()
    if result.std_error is None or result.n_paths is None:
        raise ValueError(f"case {case.case_id} did not return a Monte Carlo result")
    low, high = result.confidence_interval(0.95)
    return ValidationRow(
        case_id=case.case_id,
        model=case.model,
        product=case.product,
        method=result.method,
        n_paths=result.n_paths,
        n_steps=result.n_steps or 0,
        price=result.price,
        std_error=result.std_error,
        ci_low=low,
        ci_high=high,
        reference=ref,
        reference_method=ref_method,
        error_se=result.error_in_se(ref),
        in_ci99=result.contains(ref, CI_LEVEL),
        passed=abs(result.error_in_se(ref)) < PASS_TOL_SE,
        elapsed_s=result.elapsed_s,
        diagnostics=dict(result.diagnostics),
    )


def run_validation(
    cases: Iterable[ValidationCase],
    scale: float = 1.0,
    progress: Callable[[ValidationRow], None] | None = None,
) -> list[ValidationRow]:
    """Run every case; ``progress`` is called after each row."""
    rows = []
    for case in cases:
        row = run_case(case, scale)
        rows.append(row)
        if progress is not None:
            progress(row)
    return rows


def to_markdown(rows: Sequence[ValidationRow]) -> str:
    """Render rows as the Markdown validation table."""
    header = (
        "| Model | Product | Method | Paths | Steps | MC price | 95 % CI | Reference "
        "| Ref. method | Error (SE) | In 99 % CI | Pass |\n"
        "|---|---|---|--:|--:|--:|---|--:|---|--:|:-:|:-:|\n"
    )
    lines = [
        f"| {r.model} | {r.product} | `{r.method}` | {r.n_paths:,} | {r.n_steps} | "
        f"{r.price:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.reference:.4f} | "
        f"{r.reference_method} | {r.error_se:+.2f} | {'yes' if r.in_ci99 else 'no'} | "
        f"{'✅' if r.passed else '❌'} |"
        for r in rows
    ]
    n_pass = sum(r.passed for r in rows)
    n_in = sum(r.in_ci99 for r in rows)
    footer = (
        f"\n\n{n_pass}/{len(rows)} rows pass (|error| < {PASS_TOL_SE:g} SE); "
        f"{n_in}/{len(rows)} references lie inside the {CI_LEVEL:.0%} confidence interval "
        f"(about {0.01 * len(rows):.1f} misses expected by chance for unbiased estimators)."
    )
    return header + "\n".join(lines) + footer + "\n"


def write_outputs(rows: Sequence[ValidationRow], directory: Path) -> tuple[Path, Path]:
    """Write ``validation.md`` and ``validation.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    md = directory / "validation.md"
    js = directory / "validation.json"
    md.write_text(to_markdown(rows), encoding="utf-8")
    js.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    return md, js


@dataclass(frozen=True)
class CoverageRow:
    """Empirical coverage of the confidence intervals of one estimator."""

    case_id: str
    model: str
    product: str
    method: str
    n_paths: int
    n_seeds: int
    coverage95: float
    coverage99: float
    mean_error_se: float
    std_error_se: float


def coverage_study(
    cases: Iterable[ValidationCase], n_seeds: int = 1000, scale: float = 0.2
) -> list[CoverageRow]:
    """Re-run each case with ``n_seeds`` independent seeds and measure CI coverage.

    If the standard errors are honest and the estimator unbiased, the errors in SE
    units are approximately standard normal: coverage close to 95 % / 99 %, mean
    close to 0 and standard deviation close to 1.
    """
    rows = []
    for case in cases:
        ref, _ = case.reference()
        base = case_seed("coverage-" + case.case_id)
        errs, n_paths, method = [], 0, ""
        for i in range(n_seeds):
            res = case.run(base + i, scale)
            errs.append(res.error_in_se(ref))
            n_paths, method = res.n_paths or 0, res.method
        e = np.asarray(errs)
        rows.append(
            CoverageRow(
                case_id=case.case_id,
                model=case.model,
                product=case.product,
                method=method,
                n_paths=n_paths,
                n_seeds=n_seeds,
                coverage95=float(np.mean(np.abs(e) <= z_value(0.95))),
                coverage99=float(np.mean(np.abs(e) <= z_value(0.99))),
                mean_error_se=float(e.mean()),
                std_error_se=float(e.std(ddof=1)),
            )
        )
    return rows


def coverage_markdown(rows: Sequence[CoverageRow]) -> str:
    """Render the coverage study as Markdown."""
    header = (
        "| Model | Product | Method | Paths | Seeds | Coverage 95 % CI | Coverage 99 % CI "
        "| Mean error (SE) | Std error (SE) |\n|---|---|---|--:|--:|--:|--:|--:|--:|\n"
    )
    lines = [
        f"| {r.model} | {r.product} | `{r.method}` | {r.n_paths:,} | {r.n_seeds} | "
        f"{r.coverage95:.1%} | {r.coverage99:.1%} | {r.mean_error_se:+.3f} | "
        f"{r.std_error_se:.3f} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_BS = GBM(s0=100.0, r=0.05, sigma=0.2)
_N_EURO = 100_000


def _paths(n: int, scale: float, multiple: int = 2) -> int:
    scaled = max(multiple * 8, round(n * scale))
    return scaled - scaled % multiple


def mc_case(
    case_id: str,
    model_label: str,
    model: Model,
    product: Product,
    method: str,
    n_paths: int,
    reference: float | Callable[[], float],
    reference_label: str,
    tags: tuple[str, ...] = (),
    **mc_kwargs: Any,
) -> ValidationCase:
    """Build a case that runs :func:`price_mc` against a fixed or lazily computed reference.

    For ``method="qmc"`` the scaled path count is rounded to a valid ``16 x 2^m``.
    """

    def run(seed: int, scale: float) -> PricingResult:
        n = _paths(n_paths, scale)
        if method == "qmc":
            n = qmc_path_count(n)
        return price_mc(model, product, n_paths=n, method=method, seed=seed, **mc_kwargs)

    def ref() -> tuple[float, str]:
        value = reference() if callable(reference) else reference
        return float(value), reference_label

    return ValidationCase(case_id, model_label, product.label, run, ref, tags)


def _european_cases() -> list[ValidationCase]:
    m = _BS
    cases: list[ValidationCase] = []
    for kind, strike in (("call", 100.0), ("call", 120.0), ("put", 100.0)):
        product = EuropeanOption(strike, 1.0, kind)
        ref = float(bs_price(m.s0, strike, 1.0, m.r, m.sigma, m.q, kind))
        for method in ("plain", "antithetic", "cv"):
            cid = f"gbm-euro-{kind}-{strike:g}-{method}"
            cases.append(mc_case(cid, "GBM", m, product, method, _N_EURO, ref, "Black-Scholes"))
    digital = DigitalOption(100.0, 1.0, "call")
    dref = float(bs_digital_price(m.s0, 100.0, 1.0, m.r, m.sigma))
    for method in ("plain", "cv"):
        cid = f"gbm-digital-call-{method}"
        cases.append(mc_case(cid, "GBM", m, digital, method, _N_EURO, dref, "e^{-rT} N(d2)"))
    return cases


def _asian_cases() -> list[ValidationCase]:
    m = _BS
    cases: list[ValidationCase] = []
    geo = AsianOption(100.0, 1.0, 12, "call", "geometric")
    kv = geometric_asian_price(m.s0, 100.0, geo.monitoring_times(), m.r, m.sigma, m.q)
    for method in ("plain", "antithetic"):
        cid = f"gbm-asian-geo-call-12-{method}"
        cases.append(mc_case(cid, "GBM", m, geo, method, _N_EURO, kv, "Kemna-Vorst"))
    specs = [
        ("call", 12, ("plain", "antithetic", "cv")),
        ("put", 12, ("cv",)),
        ("call", 52, ("cv",)),
    ]
    for kind, n_fix, methods in specs:
        product = AsianOption(100.0, 1.0, n_fix, kind, "arithmetic")

        def reference(p: AsianOption = product) -> float:
            return asian_arithmetic_price(m, p)

        for method in methods:
            cid = f"gbm-asian-arith-{kind}-{n_fix}-{method}"
            cases.append(
                mc_case(cid, "GBM", m, product, method, _N_EURO, reference, "Recursive convolution")
            )
    return cases


def _barrier_cases() -> list[ValidationCase]:
    m = _BS
    cases: list[ValidationCase] = []
    for barrier_type in BARRIER_TYPES:
        barrier = 90.0 if barrier_type.startswith("down") else 120.0
        for kind in ("call", "put"):
            product = BarrierOption(100.0, 1.0, barrier, barrier_type, kind, 50, "bridge", m.sigma)
            ref = barrier_price(m.s0, 100.0, barrier, 1.0, m.r, m.sigma, m.q, barrier_type, kind)
            cid = f"gbm-barrier-{barrier_type}-{kind}-bridge"
            cases.append(
                mc_case(cid, "GBM", m, product, "plain", _N_EURO, ref, "Reiner-Rubinstein")
            )
    return cases


def lsm_case(
    case_id: str,
    model_label: str,
    model: GBM,
    product: AmericanOption,
    n_paths: int,
    reference: Callable[[], float],
    reference_label: str,
) -> ValidationCase:
    """Longstaff-Schwartz case (Laguerre degree 3, independent pricing paths)."""

    def run(seed: int, scale: float) -> PricingResult:
        n = _paths(n_paths, scale)
        return price_lsm(model, product, n_paths=n, seed=seed)

    return ValidationCase(
        case_id,
        model_label,
        product.label,
        run,
        lambda: (float(reference()), reference_label),
    )


def _american_cases() -> list[ValidationCase]:
    """Longstaff & Schwartz (2001), Table 1 grid, against a Bermudan CRR tree."""
    cases: list[ValidationCase] = []
    for s0 in (36.0, 40.0, 44.0):
        for sigma in (0.2, 0.4):
            for maturity in (1.0, 2.0):
                model = GBM(s0, 0.06, sigma)
                product = AmericanOption(40.0, maturity, "put", round(50 * maturity))
                n_steps = round(5000 * maturity)

                def reference(
                    m: GBM = model, p: AmericanOption = product, n: int = n_steps
                ) -> float:
                    return price_tree(m, p, n_steps=n).price

                cid = f"gbm-american-put-s{s0:g}-v{sigma:g}-t{maturity:g}-lsm"
                label = f"GBM S0={s0:g} σ={sigma:g}"
                ref_label = f"CRR Bermudan tree (N={n_steps}, BBS-Richardson)"
                cases.append(lsm_case(cid, label, model, product, _N_EURO, reference, ref_label))
    call_model = GBM(40.0, 0.06, 0.2)
    call = AmericanOption(40.0, 1.0, "call", 50)

    def call_reference() -> float:
        return float(bs_price(40.0, 40.0, 1.0, 0.06, 0.2))

    cases.append(
        lsm_case(
            "gbm-american-call-no-dividend-lsm",
            "GBM S0=40 σ=0.2",
            call_model,
            call,
            _N_EURO,
            call_reference,
            "Black-Scholes (no early exercise)",
        )
    )
    return cases


#: Heston parameters used throughout the validation (Feller ratio 0.64).
HESTON = Heston(s0=100.0, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
#: Merton parameters used throughout the validation.
MERTON = Merton(s0=100.0, r=0.05, sigma=0.2, lam=1.0, mu_j=-0.1, delta_j=0.15)


def _heston_cases() -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    specs = [
        ("qe", "call", 90.0, "plain", 50),
        ("qe", "call", 100.0, "plain", 50),
        ("qe", "call", 110.0, "plain", 50),
        ("qe", "call", 100.0, "antithetic", 50),
        ("qe", "put", 100.0, "cv", 50),
        ("euler", "call", 100.0, "plain", 200),
    ]
    for scheme, kind, strike, method, n_steps in specs:
        model = HESTON.with_scheme(scheme)
        product = EuropeanOption(strike, 1.0, kind)

        def reference(k: float = strike, o: str = kind) -> float:
            return gil_pelaez_price(HESTON, k, 1.0, o)

        label = f"Heston {'QE' if scheme == 'qe' else 'Euler FT'} (Δt = 1/{n_steps})"
        cid = f"heston-{scheme}-{kind}-{strike:g}-{method}"
        cases.append(
            mc_case(
                cid,
                label,
                model,
                product,
                method,
                _N_EURO,
                reference,
                "Gil-Pelaez (little trap)",
                n_steps=n_steps,
            )
        )
    return cases


def _merton_cases() -> list[ValidationCase]:
    m = MERTON
    cases: list[ValidationCase] = []
    specs = [
        ("call", 80.0, "plain"),
        ("call", 100.0, "plain"),
        ("call", 120.0, "plain"),
        ("call", 100.0, "cv"),
        ("put", 100.0, "antithetic"),
    ]
    for kind, strike, method in specs:
        product = EuropeanOption(strike, 1.0, kind)
        ref = merton_price(m.s0, strike, 1.0, m.r, m.sigma, m.lam, m.mu_j, m.delta_j, m.q, kind)
        cid = f"merton-{kind}-{strike:g}-{method}"
        cases.append(mc_case(cid, "Merton", m, product, method, _N_EURO, ref, "Merton series"))
    return cases


def greek_case(
    case_id: str,
    product: EuropeanOption | DigitalOption,
    greek: str,
    method: str,
    reference: float,
) -> ValidationCase:
    """Monte Carlo Greek (as a :class:`PricingResult`) against its closed form."""

    def run(seed: int, scale: float) -> PricingResult:
        n = _paths(_N_EURO, scale)
        res = mc_greeks(_BS, product, method=method, n_paths=n, seed=seed)
        value, se = float(getattr(res, greek)), res.std_errors[greek]
        half = z_value(0.95) * se
        return PricingResult(
            value, se, value - half, value + half, n, 1, f"mc-{method}", res.elapsed_s
        )

    label = f"{product.label}: {greek}"
    return ValidationCase(case_id, "GBM", label, run, lambda: (reference, f"Black-Scholes {greek}"))


def _greek_cases() -> list[ValidationCase]:
    m = _BS
    call = EuropeanOption(100.0, 1.0, "call")
    digital = DigitalOption(100.0, 1.0, "call")
    call_ref = bs_greeks(m.s0, 100.0, 1.0, m.r, m.sigma).as_dict()
    dig_ref = digital_greeks(m.s0, 100.0, 1.0, m.r, m.sigma).as_dict()
    cases: list[ValidationCase] = []
    for greek in ("delta", "gamma", "vega"):
        for method in ("bump", "pathwise", "lr"):
            cid = f"gbm-greek-call-{greek}-{method}"
            cases.append(greek_case(cid, call, greek, method, call_ref[greek]))
    for greek, method in (("delta", "bump"), ("delta", "lr"), ("gamma", "lr"), ("vega", "lr")):
        cid = f"gbm-greek-digital-{greek}-{method}"
        cases.append(greek_case(cid, digital, greek, method, dig_ref[greek]))
    return cases


#: Path count of the quasi-Monte Carlo rows: 16 scramblings of 2^13 Sobol points.
_N_QMC = 16 * 2**13


def _qmc_is_cases() -> list[ValidationCase]:
    m = _BS
    cases: list[ValidationCase] = []
    call = EuropeanOption(100.0, 1.0)
    bs_call = float(bs_price(m.s0, 100.0, 1.0, m.r, m.sigma))
    cases.append(
        mc_case("gbm-euro-call-100-qmc", "GBM", m, call, "qmc", _N_QMC, bs_call, "Black-Scholes")
    )
    asian = AsianOption(100.0, 1.0, 12)
    cases.append(
        mc_case(
            "gbm-asian-arith-call-12-qmc",
            "GBM",
            m,
            asian,
            "qmc",
            _N_QMC,
            lambda: asian_arithmetic_price(m, asian),
            "Recursive convolution",
        )
    )
    barrier = BarrierOption(100.0, 1.0, 90.0, "down-and-out", "call", 50, "bridge", m.sigma)
    ref = barrier_price(m.s0, 100.0, 90.0, 1.0, m.r, m.sigma)
    cases.append(
        mc_case(
            "gbm-barrier-down-and-out-call-bridge-qmc",
            "GBM",
            m,
            barrier,
            "qmc",
            _N_QMC,
            ref,
            "Reiner-Rubinstein",
        )
    )
    for strike in (140.0, 180.0):
        product = EuropeanOption(strike, 1.0)
        ref = float(bs_price(m.s0, strike, 1.0, m.r, m.sigma))
        cases.append(
            mc_case(
                f"gbm-euro-call-{strike:g}-is",
                "GBM",
                m,
                product,
                "is",
                _N_EURO,
                ref,
                "Black-Scholes",
            )
        )
    digital = DigitalOption(150.0, 1.0)
    ref = float(bs_digital_price(m.s0, 150.0, 1.0, m.r, m.sigma))
    cases.append(
        mc_case("gbm-digital-call-150-is", "GBM", m, digital, "is", _N_EURO, ref, "e^{-rT} N(d2)")
    )
    heston_call = EuropeanOption(100.0, 1.0)

    def heston_ref() -> float:
        return gil_pelaez_price(HESTON, 100.0, 1.0)

    cases.append(
        mc_case(
            "heston-qe-call-100-qmc",
            "Heston QE (Δt = 1/50)",
            HESTON,
            heston_call,
            "qmc",
            _N_QMC,
            heston_ref,
            "Gil-Pelaez (little trap)",
            n_steps=50,
        )
    )
    if HAS_NUMBA:  # pragma: no branch - depends on the environment
        fast = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7, backend="numba")
        cases.append(
            mc_case(
                "heston-qe-call-100-plain-numba",
                "Heston QE, Numba (Δt = 1/50)",
                fast,
                heston_call,
                "plain",
                _N_EURO,
                heston_ref,
                "Gil-Pelaez (little trap)",
                n_steps=50,
            )
        )
    merton_call = EuropeanOption(100.0, 1.0)
    mm = MERTON
    ref = merton_price(mm.s0, 100.0, 1.0, mm.r, mm.sigma, mm.lam, mm.mu_j, mm.delta_j)
    cases.append(
        mc_case(
            "merton-call-100-qmc", "Merton", mm, merton_call, "qmc", _N_QMC, ref, "Merton series"
        )
    )
    return cases


def all_cases() -> list[ValidationCase]:
    """The full validation registry, in table order."""
    return [
        *_european_cases(),
        *_asian_cases(),
        *_barrier_cases(),
        *_american_cases(),
        *_heston_cases(),
        *_merton_cases(),
        *_greek_cases(),
        *_qmc_is_cases(),
    ]


#: Case identifiers used in the confidence-interval coverage study.
COVERAGE_CASE_IDS: tuple[str, ...] = (
    "gbm-euro-call-100-plain",
    "gbm-euro-call-100-antithetic",
    "gbm-euro-call-100-cv",
    "gbm-euro-call-120-cv",
    "gbm-digital-call-plain",
    "gbm-asian-arith-call-12-cv",
    "gbm-barrier-down-and-out-call-bridge",
    "merton-call-100-plain",
    "gbm-greek-call-gamma-pathwise",
    "gbm-greek-digital-gamma-lr",
    "gbm-euro-call-140-is",
)


def coverage_cases() -> list[ValidationCase]:
    """Cases of :data:`COVERAGE_CASE_IDS`, taken from the registry."""
    by_id = {c.case_id: c for c in all_cases()}
    return [by_id[i] for i in COVERAGE_CASE_IDS if i in by_id]
