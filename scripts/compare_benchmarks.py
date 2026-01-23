#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_benchmark_real_times(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    results: dict[str, float] = {}
    for entry in data.get("benchmarks", []):
        # Skip aggregate rows (mean/median/stddev) to avoid double counting.
        if "aggregate_name" in entry:
            continue
        name = entry.get("name")
        real_time = entry.get("real_time")
        if not name or real_time is None:
            continue
        results[name] = float(real_time)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Google Benchmark JSON per-test times and enforce regression threshold."
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

    baseline_times = load_benchmark_real_times(args.baseline)
    current_times = load_benchmark_real_times(args.current)

    if not baseline_times:
        print("Baseline benchmark set is empty; cannot compare.")
        return 2

    regressions: list[tuple[str, float, float, float]] = []
    missing_in_current: list[str] = []
    missing_in_baseline: list[str] = []

    for name, baseline_time in baseline_times.items():
        current_time = current_times.get(name)
        if current_time is None:
            missing_in_current.append(name)
            continue
        if baseline_time <= 0:
            continue
        delta = current_time - baseline_time
        delta_pct = (delta / baseline_time) * 100.0
        if delta_pct > args.max_regression_pct:
            regressions.append((name, baseline_time, current_time, delta_pct))

    for name in current_times.keys():
        if name not in baseline_times:
            missing_in_baseline.append(name)

    if missing_in_current:
        print(f"Missing in current run ({len(missing_in_current)}):")
        for name in sorted(missing_in_current):
            print(f"  - {name}")

    if missing_in_baseline:
        print(f"New in current run ({len(missing_in_baseline)}):")
        for name in sorted(missing_in_baseline):
            print(f"  - {name}")

    if regressions:
        print("Benchmark regressions detected:")
        for name, baseline_time, current_time, delta_pct in sorted(
            regressions, key=lambda item: item[3], reverse=True
        ):
            print(
                f"  - {name}: {baseline_time:.4f} -> {current_time:.4f} "
                f"({delta_pct:.2f}%)"
            )
        return 1

    print("Benchmark regressions within threshold for all matched tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
