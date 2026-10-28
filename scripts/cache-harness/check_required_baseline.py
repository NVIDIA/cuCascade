#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the known red baseline. Success here NEVER grants feature qualification."""
import argparse
import json
from pathlib import Path
import sys
from required_features import evaluate, REQUIRED


def check(result):
    expected = {name: "UNSUPPORTED" for name in REQUIRED}
    expected.update(
        {
            "root-full-cold": "NOT_TESTED",
            "root-footer-cold": "NOT_TESTED",
            "list-etag": "PASS",
        }
    )
    actual = {r["id"]: r["status"] for r in result["rows"]}
    ok = (
        result["status"] == "FAIL"
        and actual == expected
        and len(actual) == len(result["rows"])
        and len(result["controls"]) >= 31
        and all(r["status"] == "PASS" for r in result["controls"])
    )
    return dict(
        baseline_check="PASS" if ok else "FAIL",
        qualification=result["status"],
        meaning="baseline capability failure is expected, not qualified",
        unexpected={k: v for k, v in actual.items() if expected.get(k) != v},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet", type=Path)
    args = p.parse_args()
    try:
        result = check(evaluate(args.packet))
    except Exception as exc:
        result = dict(baseline_check="FAIL", error=str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result["baseline_check"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
