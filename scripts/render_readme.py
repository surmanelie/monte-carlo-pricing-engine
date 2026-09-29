"""Fill the generated blocks of README.md and the docs summary tables from real results.

Reads results/validation.json, results/figures.json, results/benchmark.json and
results/coverage.json (``pytest --cov --cov-report=json:results/coverage.json``), runs
the quick-start commands to capture their actual output, and rewrites every
``<!-- BEGIN:name --> ... <!-- END:name -->`` block of README.md. It also writes
results/highlights.md and results/calibration.md, which the documentation includes.

Usage: ``python scripts/render_readme.py``
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> Any:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def row(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    return next(r for r in rows if r["case_id"] == case_id)


def highlights(val: list[dict[str, Any]], fig: dict[str, Any], bench: dict[str, Any]) -> str:
    euro = row(val, "gbm-euro-call-100-plain")
    rel = abs(euro["price"] - euro["reference"]) / euro["reference"] * 100
    asian = row(val, "gbm-asian-arith-call-12-cv")
    is180 = row(val, "gbm-euro-call-180-is")
    american = [r for r in val if r["case_id"].startswith("gbm-american-put")]
    worst_am = max(abs(r["error_se"]) for r in american)
    ls = row(val, "gbm-american-put-s36-v0.2-t1-lsm")
    qmc_eff = next(
        r
        for r in bench["efficiency"]
        if r["method"] == "qmc-sobol-bb" and r["problem"].startswith("European")
    )
    hedge = fig["hedging"]
    bias = fig["heston_scheme_bias"]
    calib = fig["calibration"]
    n_pass = sum(r["passed"] for r in val)
    n_in = sum(r["in_ci99"] for r in val)
    slopes = fig["error_vs_n"]
    lines = [
        f"- **European call:** with 100,000 simulated paths, the Monte Carlo price is "
        f"{euro['price']:.4f}, within {rel:.2f}% of the Black-Scholes price "
        f"{euro['reference']:.4f} ({euro['error_se']:+.2f} standard errors).",
        f"- **Validation:** {n_pass}/{len(val)} Monte Carlo estimators agree with an "
        f"independent reference within 4 SE; {n_in}/{len(val)} references lie inside the 99 % "
        "confidence interval. RMSE decays with fitted slopes "
        f"{slopes['slope_plain']:.3f} (plain), {slopes['slope_antithetic']:.3f} "
        f"(antithetic), {slopes['slope_cv']:.3f} (control variate) vs the theoretical -0.5.",
        f"- **Variance reduction:** the geometric control variate divides the variance of the "
        f"arithmetic Asian by {asian['diagnostics']['vr_factor']:,.0f}; importance sampling "
        f"divides that of a K = 180 call by {is180['diagnostics']['vr_factor']:,.0f}; "
        f"randomised QMC with a Brownian bridge divides that of a K = 140 call by "
        f"{qmc_eff['variance_reduction']:,.0f}.",
        f"- **American put (Longstaff-Schwartz):** {ls['price']:.4f} ± {ls['std_error']:.4f} vs "
        f"{ls['reference']:.4f} for the Bermudan CRR tree (S0 = 36, sigma = 0.2, T = 1); over "
        f"the 12 cases of Longstaff & Schwartz's Table 1 the largest error is "
        f"{worst_am:.2f} SE.",
        f"- **Heston:** in a Feller-violating case (ratio {bias['feller_ratio']:.2f}), "
        f"full-truncation Euler is still biased by {bias['euler_bias_dt0.03125']:+.3f} at "
        f"dt = 1/32, while Andersen's QE is at {bias['qe_bias_dt0.03125']:+.3f} "
        f"(SE {bias['qe_se_dt0.03125']:.3f}).",
        f"- **Delta hedging:** the std of the hedging error scales like N^{hedge['gbm_slope']:.2f} "
        f"under GBM, but only like N^{hedge['heston_slope']:.2f} when a Heston world is hedged "
        "with Black-Scholes deltas.",
        f"- **Calibration:** Heston fitted to a noisy synthetic surface ({calib['n_quotes']:.0f} "
        f"quotes, {calib['noise_vol'] * 1e4:.0f} bp noise) with an implied-vol RMSE of "
        f"{calib['rmse_vol'] * 1e4:.1f} bp in {calib['elapsed_s']:.1f} s; rho recovered as "
        f"{calib['fit_rho']:.3f} (true {calib['true_rho']:.3f}).",
    ]
    return "\n".join(lines) + "\n"


def calibration_table(fig: dict[str, Any]) -> str:
    calib = fig["calibration"]
    lines = ["| Parameter | True | Calibrated |", "|---|--:|--:|"]
    for name in ("v0", "kappa", "theta", "xi", "rho"):
        lines.append(f"| {name} | {calib['true_' + name]:.4f} | {calib['fit_' + name]:.4f} |")
    lines.append(
        f"\nImplied-vol RMSE {calib['rmse_vol'] * 1e4:.2f} bp, max |error| "
        f"{calib['max_abs_error_vol'] * 1e4:.2f} bp (noise {calib['noise_vol'] * 1e4:.0f} bp); "
        f"Feller ratio of the fit {calib['feller_ratio_fit']:.3f}."
    )
    return "\n".join(lines) + "\n"


def cli_output(args: list[str]) -> str:
    from typer.testing import CliRunner

    from mcengine.cli import app

    result = CliRunner().invoke(app, args, terminal_width=90)
    if result.exit_code != 0:
        raise RuntimeError(result.output)
    return result.output.rstrip()


PYTHON_EXAMPLE = """\
from mcengine import GBM, EuropeanOption, price_mc, price_analytic

model = GBM(s0=100.0, r=0.05, sigma=0.2)
call = EuropeanOption(strike=100.0, maturity=1.0)
for method in ("plain", "antithetic", "cv", "qmc"):
    n = 16 * 2**13 if method == "qmc" else 100_000
    print(price_mc(model, call, n_paths=n, method=method, seed=42))
print(price_analytic(model, call))
"""


def python_output() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(PYTHON_EXAMPLE, {})  # trusted, repository-local example
    return buffer.getvalue().rstrip()


def benchmark_table(bench: dict[str, Any]) -> str:
    from mcengine.reporting.benchmark import (
        EfficiencyRow,
        ThroughputRow,
        efficiency_markdown,
        throughput_markdown,
    )

    machine = bench["machine"]
    head = (
        f"Measured on {machine['processor']} ({machine['cpu_count']} logical CPUs), "
        f"{machine['platform']}, Python {machine['python']}, NumPy {machine['numpy']}, "
        f"Numba {machine['numba']}.\n\n"
    )
    tp = throughput_markdown([ThroughputRow(**r) for r in bench["throughput"]])
    ef = efficiency_markdown([EfficiencyRow(**r) for r in bench["efficiency"]])
    return head + tp + "\n" + ef


def replace_block(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if not pattern.search(text):
        raise KeyError(f"README has no block {name!r}")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content.rstrip()}\n{m.group(3)}", text)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    val, fig, bench = load("validation.json"), load("figures.json"), load("benchmark.json")
    cov = load("coverage.json")["totals"]["percent_covered"]
    hl = highlights(val, fig, bench)
    (RESULTS / "highlights.md").write_text(hl, encoding="utf-8")
    (RESULTS / "calibration.md").write_text(calibration_table(fig), encoding="utf-8")
    colour = "brightgreen" if cov >= 90 else "orange"
    badge = (
        f"[![coverage](https://img.shields.io/badge/coverage-{cov:.0f}%25-{colour})]"
        "(https://github.com/surmanelie/monte-carlo-pricing-engine/actions/workflows/ci.yml)"
    )
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    blocks = {
        "coverage-badge": badge,
        "highlights": hl,
        "validation": (RESULTS / "validation.md").read_text(encoding="utf-8"),
        "coverage-study": (RESULTS / "coverage.md").read_text(encoding="utf-8"),
        "benchmark": benchmark_table(bench),
        "cli-output": "```console\n$ mcengine price --model gbm --product european --type call "
        "--K 100 --n 100000\n"
        + cli_output(
            [
                "price",
                "--model",
                "gbm",
                "--product",
                "european",
                "--type",
                "call",
                "--K",
                "100",
                "--n",
                "100000",
            ]
        )
        + "\n```",
        "python-output": f"```python\n{PYTHON_EXAMPLE}```\n\nOutput:\n\n```text\n"
        f"{python_output()}\n```",
        "calibration": calibration_table(fig),
    }
    for name, content in blocks.items():
        readme = replace_block(readme, name, content)
    readme_path.write_text(readme, encoding="utf-8")
    print(hl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
