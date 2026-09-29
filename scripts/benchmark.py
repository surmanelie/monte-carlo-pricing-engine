"""Throughput, error-vs-time and efficiency benchmarks.

Writes results/benchmark.{md,json} and figures/error_vs_time.png. With ``--pytest`` it
also runs the pytest-benchmark suite in tests/benchmarks and stores its JSON report.

Usage: ``python scripts/benchmark.py [--quick] [--pytest]``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mcengine.reporting.benchmark import efficiency_markdown, run_all, throughput_markdown

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small runs (smoke test)")
    parser.add_argument("--pytest", action="store_true", help="also run pytest-benchmark")
    args = parser.parse_args(argv)
    report = run_all(ROOT / "figures", None if args.quick else ROOT / "results", args.quick)
    from mcengine.reporting.benchmark import EfficiencyRow, ThroughputRow

    print(throughput_markdown([ThroughputRow(**r) for r in report["throughput"]]))  # type: ignore[attr-defined]
    print(efficiency_markdown([EfficiencyRow(**r) for r in report["efficiency"]]))  # type: ignore[attr-defined]
    if args.pytest:
        out = ROOT / "results" / "pytest_benchmark.json"
        cmd = [
            sys.executable, "-m", "pytest", str(ROOT / "tests" / "benchmarks"), "--benchmark-only",
            "-p", "no:cacheprovider", f"--benchmark-json={out}", "-m", "",
        ]  # fmt: skip
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
