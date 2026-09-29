"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcengine import __version__
from mcengine.cli import app
from mcengine.models.heston import Heston
from mcengine.products.barrier import BarrierOption
from mcengine.references import reference_price

runner = CliRunner()


def _run(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_version() -> None:
    assert __version__ in _run("version")


def test_price_gbm_european_reports_reference() -> None:
    out = _run("price", "--n", "20000", "--seed", "1")
    assert "Black-Scholes" in out
    assert "Error (SE)" in out
    assert "Inside 99 % CI" in out


@pytest.mark.parametrize(
    "args",
    [
        [
            "--product",
            "barrier",
            "--type",
            "down-and-out-call",
            "--B",
            "90",
            "--correction",
            "bridge",
            "--monitoring",
            "50",
        ],
        ["--product", "barrier", "--type", "up-and-in-put", "--B", "120", "--monitoring", "12"],
        ["--product", "asian", "--method", "cv", "--fixings", "12"],
        ["--product", "digital", "--type", "put", "--method", "antithetic"],
        ["--product", "european", "--K", "150", "--method", "is"],
        ["--product", "european", "--method", "qmc"],
        ["--model", "merton", "--method", "cv"],
        ["--model", "heston", "--steps", "10"],
        [
            "--model",
            "heston",
            "--scheme",
            "euler",
            "--product",
            "barrier",
            "--type",
            "down-and-out-call",
            "--B",
            "80",
            "--monitoring",
            "10",
        ],
        [
            "--product",
            "american",
            "--type",
            "put",
            "--S0",
            "36",
            "--K",
            "40",
            "--r",
            "0.06",
            "--exercise-dates",
            "10",
        ],
    ],
)
def test_price_variants(args: list[str]) -> None:
    out = _run("price", "--n", "8192", *args)
    assert "Price" in out
    assert "Reference" in out


def test_price_errors() -> None:
    bad_type = runner.invoke(
        app, ["price", "--product", "barrier", "--type", "sideways", "--B", "90"]
    )
    assert bad_type.exit_code != 0
    no_barrier = runner.invoke(
        app, ["price", "--product", "barrier", "--type", "down-and-out-call"]
    )
    assert no_barrier.exit_code != 0
    lsm_european = runner.invoke(app, ["price", "--method", "lsm"])
    assert lsm_european.exit_code != 0
    invalid = runner.invoke(app, ["price", "--sigma", "-0.2"])
    assert invalid.exit_code == 2
    assert "sigma" in invalid.output


def test_validate_quick_subset(tmp_path: Path) -> None:
    out = _run("validate", "--quick", "--filter", "gbm-euro-call-100", "--out", str(tmp_path))
    assert "PASS" in out


def test_validate_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcengine.reporting.validation as validation

    subset = [c for c in validation.all_cases() if c.case_id == "gbm-euro-call-100-cv"]
    monkeypatch.setattr(validation, "all_cases", lambda: subset)
    monkeypatch.setattr(validation, "coverage_cases", lambda: subset)
    original = validation.coverage_study
    monkeypatch.setattr(validation, "coverage_study", lambda cases: original(cases, 5, 0.01))
    _run("validate", "--out", str(tmp_path))
    assert (tmp_path / "validation.md").exists()
    assert (tmp_path / "coverage.md").exists()


def test_validate_failure_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcengine.reporting.validation as validation

    case = next(c for c in validation.all_cases() if c.case_id == "gbm-euro-call-100-plain")
    wrong = validation.ValidationCase(case.case_id, case.model, case.product, case.run,
                                      lambda: (0.0, "wrong"))  # fmt: skip
    monkeypatch.setattr(validation, "all_cases", lambda: [wrong])
    result = runner.invoke(app, ["validate", "--quick"])
    assert result.exit_code == 1


def test_figures_quick(tmp_path: Path) -> None:
    _run("figures", "--quick", "--only", "convergence", "--out", str(tmp_path))
    assert (tmp_path / "convergence.png").exists()


def test_benchmark_quick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcengine.reporting.benchmark as bench

    def fake_run_all(out: Path, results: Path | None, quick: bool) -> dict[str, object]:
        return {"throughput": [], "efficiency": []}

    monkeypatch.setattr(bench, "run_all", fake_run_all)
    assert "Paths / s" in _run("benchmark", "--quick", "--out", str(tmp_path))


def test_calibrate_from_csv(tmp_path: Path) -> None:
    from mcengine.calibration.heston_calibration import synthetic_surface

    surface = synthetic_surface(
        Heston(100.0, 0.01, 0.04, 1.5, 0.05, 0.6, -0.7), maturities=(0.5, 1.0)
    )
    lines = ["maturity,strike,implied_vol"] + [
        f"{t},{k},{v}"
        for t, k, v in zip(surface.maturities, surface.strikes, surface.vols, strict=True)
    ]
    path = tmp_path / "chain.csv"
    path.write_text("\n".join(lines) + "\n")
    out = _run("calibrate", "--csv", str(path), "--spot", "100", "--rate", "0.01", "--starts", "1")
    assert "rho" in out
    assert "Feller" in out
    missing = runner.invoke(
        app, ["calibrate", "--csv", str(tmp_path / "nope.csv"), "--spot", "100"]
    )
    assert missing.exit_code == 2


def test_reference_price_coverage() -> None:
    heston = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7)
    assert reference_price(heston, BarrierOption(100.0, 1.0, 90.0)) is None
