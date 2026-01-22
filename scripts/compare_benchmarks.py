#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_total_real_time(path: Path) -> float:
    data = json.loads(path.read_text())
    total = 0.0
    for entry in data.get("benchmarks", []):
        # Skip aggregate rows (mean/median/stddev) to avoid double counting.
        if "aggregate_name" in entry:
            continue
        if "real_time" in entry:
            total += float(entry["real_time"])
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Google Benchmark JSON totals and enforce regression threshold."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=10.0,
        help="Fail if current total real_time is more than this percent slower.",
    )
    args = parser.parse_args()

    baseline_total = load_total_real_time(args.baseline)
    current_total = load_total_real_time(args.current)

    if baseline_total <= 0:
        print(f"Baseline total is non-positive ({baseline_total}); cannot compare.")
        return 2

    delta = current_total - baseline_total
    delta_pct = (delta / baseline_total) * 100.0

    print(f"Baseline total real_time: {baseline_total:.4f}")
    print(f"Current total real_time:  {current_total:.4f}")
    print(f"Delta: {delta:.4f} ({delta_pct:.2f}%)")

    if delta_pct > args.max_regression_pct:
        print(f"Regression detected: {delta_pct:.2f}% > {args.max_regression_pct:.2f}%")
        return 1

    print("Benchmark regression within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
