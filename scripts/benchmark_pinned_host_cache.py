#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

"""Build and compare the same public-API benchmark against two existing Ninja builds."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess


def cache_values(build):
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def build_benchmark(build, output, source):
    cache = cache_values(build)
    commands = subprocess.check_output(
        [
            cache["CMAKE_MAKE_PROGRAM"],
            "-C",
            str(build),
            "-t",
            "commands",
            "cucascade_tests",
        ],
        text=True,
    ).splitlines()
    compile_line = next(
        line
        for line in commands
        if " -c " in line
        and "/test/memory/test_small_pinned_host_memory_resource.cpp" in line
    )
    parts = shlex.split(compile_line)
    flags = []
    position = 0
    while position < len(parts):
        part = parts[position]
        if part in ("-isystem", "-I", "-D"):
            flags.extend(parts[position : position + 2])
            position += 2
            continue
        if part.startswith(("-I", "-D")):
            flags.append(part)
        position += 1
    obj = output.with_suffix(".o")
    compile_command = [
        cache["CMAKE_CXX_COMPILER"],
        *flags,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    subprocess.run(compile_command, cwd=build, check=True)
    link = shlex.split(commands[-1])
    if link[:2] == [":", "&&"]:
        link = link[2:]
    if link[-2:] == ["&&", ":"]:
        link = link[:-2]
    result = [link[0], str(obj)]
    position = 1
    while position < len(link):
        part = link[position]
        if part == "-o":
            result.extend(["-o", str(output)])
            position += 2
            continue
        if (
            not part.endswith(".o")
            and not part.startswith("-Wl,--dependency-file=")
            and not Path(part).name.startswith("libCatch2")
        ):
            result.append(part)
        position += 1
    subprocess.run(result, cwd=build, check=True)
    source_root = Path(cache["CMAKE_HOME_DIRECTORY"])
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    )
    return {
        "revision": revision,
        "status": status,
        "build": str(build),
        "compile_command": compile_command,
        "link_command": result,
        "configuration": {
            key: cache.get(key)
            for key in (
                "CMAKE_BUILD_TYPE",
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_FLAGS",
                "CMAKE_CXX_FLAGS_RELEASE",
                "CMAKE_CUDA_COMPILER",
                "CMAKE_CUDA_FLAGS",
                "CMAKE_CUDA_FLAGS_RELEASE",
                "CMAKE_CUDA_ARCHITECTURES",
                "CUCASCADE_BUILD_CUDF",
                "CUCASCADE_BUILD_IO",
                "CUCASCADE_BUILD_BENCHMARKS",
                "CUCASCADE_BUILD_SHARED_LIBS",
                "CUCASCADE_BUILD_STATIC_LIBS",
                "CUCASCADE_WARNINGS_AS_ERRORS",
            )
        },
    }


def distribution(values):
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--hit-iterations", type=int, default=500000)
    parser.add_argument("--full-iterations", type=int, default=3000)
    parser.add_argument("--half-iterations", type=int, default=40)
    args = parser.parse_args()
    if args.pairs < 5:
        parser.error("At least five paired trials are required")
    if min(args.hit_iterations, args.full_iterations, args.half_iterations) < 1:
        parser.error("Iteration counts must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.jsonl").exists():
        parser.error("Output already contains raw.jsonl; use a fresh directory")
    source = (
        Path(__file__).resolve().parents[1]
        / "benchmark/benchmark_pinned_host_cache.cpp"
    )
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "benchmark_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pairs": args.pairs,
    }
    executables = {}
    for name, build in (
        ("baseline", args.baseline_build),
        ("candidate", args.candidate_build),
    ):
        executables[name] = output / name
        metadata[name] = build_benchmark(build.resolve(), executables[name], source)
    baseline_config = metadata["baseline"]["configuration"]
    candidate_config = metadata["candidate"]["configuration"]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if baseline_config != candidate_config:
        raise RuntimeError(
            "Build configurations differ; inspect metadata and use matching builds"
        )
    cases = (
        ("concurrent_hits", "hits", 4, args.hit_iterations, 128),
        ("concurrent_full_cap", "full", 4, args.full_iterations, 128),
        ("deterministic_full_cap", "full", 1, args.full_iterations, 128),
        ("half_cap_rounding", "half", 1, args.half_iterations, 4),
    )
    rows = []
    with (output / "raw.jsonl").open("w") as raw:
        for case, mode, threads, iterations, warmup in cases:
            for pair in range(args.pairs):
                order = (
                    ("baseline", "candidate")
                    if pair % 2 == 0
                    else ("candidate", "baseline")
                )
                for name in order:
                    command = [
                        str(executables[name]),
                        mode,
                        str(threads),
                        str(iterations),
                        str(warmup),
                    ]
                    result = subprocess.run(
                        command, text=True, capture_output=True, check=True
                    )
                    row = json.loads(result.stdout)
                    row.update(
                        case=case, pair=pair + 1, revision=name, order=list(order)
                    )
                    rows.append(row)
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()
                    print(
                        f"{case} {pair + 1} {name}: {row['operations_per_second']:.0f} ops/s",
                        flush=True,
                    )
    summary = {}
    for case, *_ in cases:
        measured = {
            name: [r for r in rows if r["case"] == case and r["revision"] == name]
            for name in executables
        }
        distributions = {
            name: distribution([r["operations_per_second"] for r in records])
            for name, records in measured.items()
        }
        ratios = [
            candidate["operations_per_second"] / baseline["operations_per_second"]
            for baseline, candidate in zip(measured["baseline"], measured["candidate"])
        ]
        ratio = (
            distributions["candidate"]["median"] / distributions["baseline"]["median"]
        )
        summary[case] = {
            **distributions,
            "median_throughput_ratio": ratio,
            "paired_ratios": distribution(ratios),
            "requires_regression_investigation": ratio < 0.95,
            "counters": {
                name: {
                    key: [r[key] for r in records]
                    for key in ("hits", "misses", "evictions")
                }
                for name, records in measured.items()
            },
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
