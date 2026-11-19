#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the multi-device Catch2 cases, requiring complete coverage without skips."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


TEST_BINARIES = (
    "cucascade_tests",
    "cucascade_cudf_tests",
    "cucascade_topology_discovery_tests",
)


def check_report(path: Path) -> int:
    root = ET.parse(path).getroot()
    if root.tag != "Catch2TestRun" or not root.findall("TestCase"):
        raise ValueError("report contains no Catch2 test cases")
    if root.find(".//Skip") is not None:
        raise ValueError("a selected test or section was skipped")
    cases = root.findall("TestCase")
    for case in cases:
        result = case.find("OverallResult")
        if (
            result is None
            or result.get("success") != "true"
            or result.get("skips") != "0"
        ):
            raise ValueError(f"incomplete result for {case.get('name')}")
    for tag in ("OverallResults", "OverallResultsCases"):
        totals = root.find(tag)
        if totals is None:
            raise ValueError(f"report has no {tag}")
        counts = {
            key: int(totals.attrib[key])
            for key in ("successes", "failures", "expectedFailures", "skips")
        }
        if counts["successes"] <= 0 or any(
            counts[key] != 0 for key in ("failures", "expectedFailures", "skips")
        ):
            raise ValueError(f"incomplete coverage: {tag}={counts}")
    count = int(root.find("OverallResultsCases").attrib["successes"])
    if count != len(cases):
        raise ValueError("case count does not match the report's successful cases")
    return count


@contextmanager
def defer_interruptions():
    pending = []
    handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def defer(signum, frame):
        if handlers[signum] != signal.SIG_IGN:
            pending.append((signum, frame))

    try:
        for signum in handlers:
            signal.signal(signum, defer)
        yield pending
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        for signum, frame in pending:
            handler = handlers[signum]
            if callable(handler):
                handler(signum, frame)
            elif handler == signal.SIG_DFL:
                raise SystemExit(128 + signum)


def run_tests(binary: Path, output_dir: Path, timeout: int) -> bool:
    report = output_dir / f"{binary.name}.xml"
    log = output_dir / f"{binary.name}.log"
    try:
        with log.open("w") as output:
            report.unlink(missing_ok=True)
            with defer_interruptions() as pending:
                process = None
                exception_in_flight = False
                try:
                    process = subprocess.Popen(
                        [
                            str(binary),
                            "[multi-device]",
                            "--reporter",
                            "xml",
                            "--warn",
                            "NoAssertions",
                            "--out",
                            str(report),
                        ],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    deadline = time.monotonic() + timeout
                    while True:
                        if pending:
                            raise ValueError("interrupted")
                        # Keep the group leader unreaped so killpg cannot target a reused PID.
                        if (
                            os.waitid(
                                os.P_PID,
                                process.pid,
                                os.WEXITED | os.WNOHANG | os.WNOWAIT,
                            )
                            is not None
                        ):
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ValueError(f"timed out after {timeout}s")
                        time.sleep(min(remaining, 0.1))
                except BaseException:
                    exception_in_flight = True
                    raise
                finally:
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            result = process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if not exception_in_flight:
                                raise ValueError(
                                    "test did not exit within 5s of SIGKILL"
                                ) from None
                            print(
                                f"{binary.name}: test did not exit within 5s of SIGKILL",
                                file=sys.stderr,
                            )
        if result != 0:
            raise ValueError(f"test process exited with status {result}")
        count = check_report(report)
    except (OSError, ET.ParseError, KeyError, ValueError) as error:
        print(f"FAIL {binary.name}: {error}; diagnostics: {output_dir}", flush=True)
        return False
    print(f"PASS {binary.name}: {count} cases; diagnostics: {output_dir}", flush=True)
    return True


def handle_termination(signum: int, _frame) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build/release"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout per executable, in seconds"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output_dir = (args.output_dir or args.build_dir / "multi-gpu-results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_handler = signal.signal(signal.SIGTERM, handle_termination)
    previous_child_handler = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        results = [
            run_tests(
                (args.build_dir / "test" / name).resolve(), output_dir, args.timeout
            )
            for name in TEST_BINARIES
        ]
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGCHLD, previous_child_handler)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
