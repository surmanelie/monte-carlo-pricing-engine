from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcengine.reporting.validation import (
    ValidationCase,
    all_cases,
    case_seed,
    coverage_cases,
    coverage_markdown,
    coverage_study,
    run_case,
    run_validation,
    to_markdown,
    write_outputs,
)
from mcengine.results import PricingResult


def test_registry_ids_are_unique_and_seeds_stable() -> None:
    ids = [c.case_id for c in all_cases()]
    assert len(ids) == len(set(ids))
    assert case_seed("abc") == case_seed("abc") != case_seed("abd")


def test_quick_validation_run(tmp_path: Path) -> None:
    cases = all_cases()[:3]
    seen: list[str] = []
    rows = run_validation(cases, scale=0.05, progress=lambda r: seen.append(r.case_id))
    assert seen == [c.case_id for c in cases]
    for row in rows:
        assert row.ci_low < row.price < row.ci_high
        assert row.passed == (abs(row.error_se) < 4.0)
    table = to_markdown(rows)
    assert table.count("\n| GBM") == 3
    assert "rows pass" in table
    md, js = write_outputs(rows, tmp_path)
    assert md.read_text(encoding="utf-8") == table
    assert len(json.loads(js.read_text(encoding="utf-8"))) == 3


def test_coverage_study_small() -> None:
    rows = coverage_study(coverage_cases()[:1], n_seeds=20, scale=0.01)
    assert rows[0].n_seeds == 20
    assert 0.0 <= rows[0].coverage95 <= rows[0].coverage99 <= 1.0
    assert "Coverage 95 %" in coverage_markdown(rows)


def test_deterministic_case_is_rejected() -> None:
    def run(seed: int, scale: float) -> PricingResult:
        return PricingResult(1.0, None, None, None, None, None, "bs-analytic", 0.0)

    case = ValidationCase("x", "GBM", "p", run, lambda: (1.0, "ref"))
    with pytest.raises(ValueError, match="Monte Carlo"):
        run_case(case)


@pytest.mark.parametrize(
    "case_id", ["gbm-american-put-s36-v0.2-t1-lsm", "gbm-asian-arith-call-12-cv"]
)
def test_selected_cases_pass_at_small_scale(case_id: str) -> None:
    case = next(c for c in all_cases() if c.case_id == case_id)
    row = run_case(case, scale=0.1)
    assert row.passed
