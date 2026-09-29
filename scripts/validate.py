"""Run the full validation table and write results/validation.{md,json}.

Usage: ``python scripts/validate.py [--quick] [--filter SUBSTRING]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcengine.reporting.validation import (
    all_cases,
    coverage_cases,
    coverage_markdown,
    coverage_study,
    run_validation,
    to_markdown,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="10x fewer paths (smoke run)")
    parser.add_argument("--filter", default="", help="only cases whose id contains this")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-coverage", action="store_true", help="skip the CI coverage study")
    args = parser.parse_args(argv)

    cases = [c for c in all_cases() if args.filter in c.case_id]
    rows = run_validation(
        cases,
        scale=0.1 if args.quick else 1.0,
        progress=lambda r: print(
            f"{'PASS' if r.passed else 'FAIL'} {r.case_id:<45} {r.error_se:+6.2f} SE "
            f"({r.elapsed_s:.1f}s)",
            file=sys.stderr,
        ),
    )
    print(to_markdown(rows))
    if not args.quick and not args.filter:
        write_outputs(rows, args.out)
        if not args.skip_coverage:
            print("coverage study ...", file=sys.stderr)
            coverage = coverage_markdown(coverage_study(coverage_cases()))
            print(coverage)
            (args.out / "coverage.md").write_text(coverage, encoding="utf-8")
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
