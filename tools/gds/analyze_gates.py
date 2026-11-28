#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze Phase 7 CSVs without treating compat/simulator throughput as GDS proof.

qualify.sh manifest.tsv supplies matrix coordinates and command exit status.
Standalone CSVs can instead include step,mount,fs_class,device,direction columns.
Optional direct-evidence.json maps CSV relative paths to an operator-reviewed
{qualified: true, reason: ..., artifacts: [...]} record. Raw nvidia-fs snapshots
alone never qualify a run (AM-11: no attribution/parser is assumed).
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics


def num(row, key, default=None):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else default
    except (KeyError, ValueError, TypeError):
        return default


def csv_rows(path):
    lines = [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    # P0/microbench may print a preamble. Find the real header, not a diagnostic.
    for index, line in enumerate(lines):
        if "," in line and any(
            k in line.split(",")
            for k in ("gib_s", "gib_per_s", "elapsed_s", "version", "heads")
        ):
            return list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
    return []


def load_rows(root):
    root = Path(root)
    manifest = root / "manifest.tsv"
    entries = (
        list(csv.DictReader(io.StringIO(manifest.read_text()), delimiter="\t"))
        if manifest.exists()
        else []
    )
    evidence_path = root / "direct-evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    rows, consumed = [], set()
    for entry in entries:
        path = root / entry["path"]
        # Engine writes its actual data to --csv, not to the captured stdout log.
        if entry["step"] in ("B-3", "B-4", "B-5", "B-6", "B-8", "B-9", "G-7"):
            path = path.with_suffix(".csv")
        data = []
        if entry["step"] == "B-2" and path.exists():
            match = re.search(
                r"Throughput\s*:\s*([\d.]+)\s*(GiB|GB|MiB|MB)/s", path.read_text(), re.I
            )
            if match:
                factors = {
                    "gib": 1,
                    "gb": 10**9 / 2**30,
                    "mib": 1 / 1024,
                    "mb": 10**6 / 2**30,
                }
                data = [{"gib_s": str(float(match[1]) * factors[match[2].lower()])}]
        elif path.exists():
            data = csv_rows(path)
        if entry["step"] == "G-7":
            data = []
            for sample in root.glob(path.name + ".soak.csv"):
                for index, row in enumerate(csv_rows(sample)):
                    rows.append(
                        {**entry, **row, "source": sample.name, "input_row": index + 2}
                    )
                consumed.add(sample)
            for epoch in root.glob(path.name + ".epoch-*.csv"):
                consumed.add(
                    epoch
                )  # Loaded below as epoch evidence, never throughput cells.
        for index, row in enumerate(data):
            rows.append(
                {
                    **entry,
                    **row,
                    "source": str(path.relative_to(root)),
                    "input_row": index + 2,
                }
            )
        if not data and entry["step"] not in (
            "G-7",
            "metadata",
            "topology",
            "prepare",
            "analyze",
        ):
            rows.append({**entry, "source": entry["path"], "missing_data": True})
        if entry["step"] == "G-7" and entry.get("exit_code") != "0":
            rows.append({**entry, "source": entry["path"], "missing_data": True})
        consumed.add(path)
        consumed.add(root / entry["path"])
    for path in sorted(root.glob("*.csv")):
        if path in consumed or ".epoch-" in path.name:
            continue
        for index, row in enumerate(csv_rows(path)):
            rows.append({**row, "source": path.name, "input_row": index + 2})
    # Attach epoch summaries and successful-exit markers to every sample, including
    # standalone soaks. A truncated/crashed standalone run cannot qualify.
    for row in rows:
        if "op_pool_live" in row:
            base = row["source"].removesuffix(".soak.csv")
            epoch = root / (base + ".epoch-" + row.get("epoch", "0") + ".csv")
            data = csv_rows(epoch) if epoch.exists() else []
            row["epoch_summary"] = data[0] if len(data) == 1 else {}
            row["complete"] = (root / (base + ".ok")).exists()
            row["verify"] = row["epoch_summary"].get("verify", "0")
            row["workload"] = row["epoch_summary"].get("workload", "unspecified")
            for key in ("threads", "op_bytes", "engine"):
                if key in row["epoch_summary"]:
                    row.setdefault(key, row["epoch_summary"][key])
    requirements = root / "expected.json"
    expected = json.loads(requirements.read_text()) if requirements.exists() else {}
    failure_log = root / "failures.log"
    failures = failure_log.read_text().strip() if failure_log.exists() else ""
    for row in rows:
        row["expected"] = expected
        row["runner_failures"] = failures
    for row in rows:
        row.setdefault(
            "step",
            (
                "G-7"
                if "op_pool_live" in row
                else (
                    "B-10"
                    if "version" in row
                    else (
                        "B-0"
                        if row.get("kind") == "result"
                        else (
                            "B-1"
                            if row.get("kind") == "micro"
                            else {
                                "sustained": "B-3",
                                "restore": "B-4",
                                "single": "B-5",
                                "spill": "B-6",
                            }.get(row.get("workload"), "unknown")
                        )
                    )
                )
            ),
        )
        if row["step"] == "unknown":
            raise ValueError(f"Unclassified CSV: {row['source']}")
        row.setdefault("mount", "unspecified")
        row.setdefault("device", "0")
        row.setdefault(
            "direction", "write" if row.get("workload") == "spill" else "read"
        )
        row.setdefault("exit_code", "0")
        row.setdefault(
            "access_pattern",
            (
                row.get("variant", "").removeprefix("gdsio-")
                if row["step"] == "B-2" and row.get("variant", "").startswith("gdsio-")
                else "sequential" if row.get("workload") == "single" else "random"
            ),
        )
        if "gib_per_s" in row:
            row["gib_s"] = row["gib_per_s"]
        if "op_bytes" in row:
            row["op_mib"] = str((num(row, "op_bytes", 0)) / 2**20)
        proof = evidence.get(row["source"], {})
        row["direct_qualified"] = bool(
            proof.get("qualified")
            and proof.get("reason")
            and proof.get("artifacts")
            and row.get("engine") != "simulated"
            and row.get("admission") not in ("test_only_compat", "simulated")
            and num(row, "compat_only", 0) != 1
            and num(row, "forced_compat", 0) != 1
            and row.get("fs_class") != "compat"
        )
        row["direct_evidence"] = proof
    return rows


def engine_fraction(row):
    values = [
        num(row, k) for k in ("engine_logical_bytes", "legacy_bytes", "sync_bytes")
    ]
    if None in values or min(values) < 0 or sum(values) <= 0:
        return None
    return values[0] / sum(values)


def routed(row):
    fraction = engine_fraction(row)
    return fraction is not None and fraction >= 0.99 and num(row, "legacy_bytes") == 0


def analyze(rows, warmup_s=60, soak_min_s=3600, tolerance=0.01):
    gates, performance = [], []

    def gate(name, status, reason, inputs):
        gates.append(
            {"gate": name, "status": status, "reason": reason, "input_rows": inputs}
        )

    def group(data, keys):
        result = {}
        for row in data:
            result.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
        return result

    def good(row):
        return (
            row.get("exit_code", "0") == "0"
            and not row.get("missing_data")
            and (
                num(row, "verify") == 1
                or row["step"] in ("B-1", "B-2", "metadata", "topology")
            )
            and num(row, "data_mismatches", 0) == 0
            and num(row, "bytes_done_mismatches", 0) == 0
        )

    def label(row):
        return f"{row['mount']}/{row.get('workload', row.get('variant', ''))}/device={row['device']}"

    baselines = [
        r for r in rows if r["step"] == "B-2" and good(r) and num(r, "gib_s", 0) > 0
    ]
    for row in [
        r for r in rows if r["step"] in ("B-3", "B-4", "B-5", "B-6", "B-8", "B-9")
    ]:
        # Match actual effective geometry when reported; requested values are not proof.
        qd = num(row, "Q_eff_max", 0) or num(row, "qd")
        op = (num(row, "S_eff_max", 0) / 2**20) or num(row, "op_mib")
        matches = [
            b
            for b in baselines
            if all(b.get(k) == row.get(k) for k in ("mount", "device", "direction"))
            and b.get("access_pattern") == row.get("access_pattern")
            and num(b, "qd") == qd
            and num(b, "op_mib") == op
        ]
        base = statistics.mean(num(b, "gib_s") for b in matches) if matches else None
        rate = num(row, "gib_s")
        ratio = rate / base if rate is not None and base else None
        performance.append(
            {
                "mount": row["mount"],
                "workload": row.get("workload", row["step"]),
                "variant": row.get("variant", row.get("engine", "")),
                "gib_s": rate,
                "baseline_gib_s": base,
                "ratio": ratio,
                "source": row["source"],
                "matched_rows": matches,
            }
        )
        if row["step"] not in ("B-8", "B-9") and row.get("engine") == "async":
            status = (
                "FAIL"
                if not good(row)
                else (
                    "N/A"
                    if ratio is None
                    or not routed(row)
                    or num(row, "seconds", 0) < 30
                    or num(row, "repetitions", 0) < 3
                    or (row.get("direction") == "write" and num(row, "flush", 0) != 1)
                    or not all(r["direct_qualified"] for r in [row] + matches)
                    else "PASS" if ratio >= 0.8 else "FAIL"
                )
            )
            if good(row) and not routed(row) and num(row, "policy_demotions", 0) > 0:
                status = (
                    "FAIL"  # Engine-only case demoted mid-run, not missing evidence.
                )
            gate(
                "G-2",
                status,
                f"{label(row)} engine_fraction={engine_fraction(row)} (routed legacy/sync if <0.99 or legacy_bytes>0); ratio={ratio}; requires >=0.8 and reviewed direct-path evidence; exact baseline QD={qd}, MiB={op}",
                [row] + matches,
            )

    b0 = [r for r in rows if r["step"] == "B-0" and r.get("mode", "async") == "async"]
    for key, cases in group(b0, ("mount", "device", "registration", "op_mib")).items():
        one = [
            r for r in cases if num(r, "qd") == 1 and good(r) and num(r, "gib_s", 0) > 0
        ]
        many = [
            r
            for r in cases
            if num(r, "qd") == 32 and good(r) and num(r, "gib_s", 0) > 0
        ]
        ratio = (
            statistics.mean(num(r, "gib_s") for r in many)
            / statistics.mean(num(r, "gib_s") for r in one)
            if one and many
            else None
        )
        status = (
            "N/A"
            if ratio is None or not all(r["direct_qualified"] for r in one + many)
            else "PASS" if ratio >= 4 else "FAIL"
        )
        if any(not good(r) for r in cases):
            status = "FAIL"
        gate(
            "G-1",
            status,
            f"{key}: QD32/QD1={ratio}; requires >=4 and direct evidence (GPFS blocking gate)",
            cases,
        )

    engines = [
        r
        for r in rows
        if r["step"] in ("B-1", "B-3", "B-4", "B-5", "B-6")
        and r.get("engine") in ("async", "simulated")
    ]
    for row in engines:
        effective = num(row, "Q_eff_max") or num(row, "Q_eff_min")
        duration = num(row, "backlog_ns", 0)
        mean = num(row, "backlog_mean_qd")
        ratio = mean / effective if mean is not None and effective else None
        gap_ratio = num(row, "refill_gap_ns", 0) / duration if duration else None
        status = (
            "INFO"
            if not row.get("direct_qualified") or row.get("engine") == "simulated"
            else (
                "N/A"
                if ratio is None or not duration or not routed(row)
                else "PASS" if ratio >= 0.9 and gap_ratio <= 0.1 else "FAIL"
            )
        )
        if not good(row) and status != "INFO":
            status = "FAIL"
        gate(
            "G-3",
            status,
            f"{label(row)} time-integral backlog mean/Q_eff={ratio}; requires >=0.9; refill_gap/backlog={gap_ratio} (<=0.1); backlog_ns={duration}; engine_fraction={engine_fraction(row)}",
            [row],
        )
        queries = num(row, "queries_per_op")
        status = (
            "N/A"
            if queries is None or not num(row, "completed_ops", 0) or not routed(row)
            else "PASS" if queries <= 70 else "FAIL"
        )
        if not good(row):
            status = "FAIL"
        gate(
            "G-5a",
            status,
            f"{label(row)} queries_per_op={queries} (ceiling 70); residence_p99_us={row.get('residence_p99_us', 'not reported')}; AM-15 subcriterion only; not NFR-3/NFR-4 qualification",
            [row],
        )

    for row in [r for r in rows if r["step"] == "B-0"]:
        verified = (
            num(row, "data_mismatches") is not None
            and num(row, "bytes_done_mismatches") is not None
        )
        status = (
            "FAIL"
            if not good(row) or num(row, "bytes_done_mismatches", 0)
            else (
                "PASS"
                if verified
                and (row.get("registration") != "arena" or row.get("direct_qualified"))
                else "N/A"
            )
        )
        gate(
            "AM-9",
            status,
            f"{label(row)} registration={row.get('registration')}: content mismatches={row.get('data_mismatches', 'not reported')}; full P0 verification required; this does not qualify shared registration for production",
            [row],
        )

    b10 = [r for r in rows if r["step"] == "B-10"]
    for key, cases in group(
        b10,
        (
            "mount",
            "device",
            "workload",
            "reactors",
            "task_bytes",
            "file_bytes",
            "drop_cache",
            "arm",
            "writer_threads",
            "inode_mode",
        ),
    ).items():
        pairs = group(cases, ("iteration",))
        ratios = []
        complete = True
        for pair in pairs.values():
            old = [
                r
                for r in pair
                if r.get("version") == "old" and good(r) and num(r, "gib_s", 0) > 0
            ]
            new = [
                r
                for r in pair
                if r.get("version") == "new" and good(r) and num(r, "gib_s", 0) > 0
            ]
            if len(old) != 1 or len(new) != 1:
                complete = False
            else:
                ratios.append(num(new[0], "gib_s") / num(old[0], "gib_s"))
        ratio = statistics.mean(ratios) if ratios else None
        expected_pairs = cases[0].get("expected", {}).get("b10_repetitions", 5)
        complete = complete and len(ratios) >= expected_pairs
        # Conservative two-sided Student-t 95% interval (df>30 uses df=30).
        critical = [
            0,
            12.706,
            4.303,
            3.182,
            2.776,
            2.571,
            2.447,
            2.365,
            2.306,
            2.262,
            2.228,
            2.201,
            2.179,
            2.160,
            2.145,
            2.131,
            2.120,
            2.110,
            2.101,
            2.093,
            2.086,
            2.080,
            2.074,
            2.069,
            2.064,
            2.060,
            2.056,
            2.052,
            2.048,
            2.045,
            2.042,
        ]
        margin = (
            (
                critical[min(len(ratios) - 1, 30)]
                * statistics.stdev(ratios)
                / math.sqrt(len(ratios))
            )
            if len(ratios) >= 2
            else None
        )
        lower = ratio - margin if margin is not None else None
        status = (
            "FAIL"
            if any(not good(r) or r.get("runner_failures") for r in cases)
            else (
                "N/A"
                if not complete or ratio is None
                else "PASS" if lower is not None and lower >= 0.95 else "FAIL"
            )
        )
        gate(
            "G-4",
            status,
            f"{key}: paired mean new/old={ratio}; 95% CI lower={lower}; requires lower>=0.95; pairs={len(ratios)}/{expected_pairs}; complete={complete}",
            cases,
        )

    samples = [r for r in rows if r["step"] == "G-7"]
    for key, cases in group(samples, ("mount", "device", "variant", "source")).items():
        stable = [
            r
            for r in cases
            if num(r, "elapsed_s", -1) >= warmup_s
            and r.get("phase", "quiescent") == "quiescent"
        ]
        fields = (
            "arena_bytes",
            "registered_bytes",
            "unregistered_bytes",
            "pinned_host_bytes",
            "op_pool_live",
            "lane_count",
            "thread_count",
            "fd_count",
            "device_used_bytes",
        )
        failures, missing = [], []
        for field in fields:
            values = [num(r, field) for r in stable]
            if not values or None in values:
                missing.append(field)
                continue
            allowed = (
                0 if field != "pinned_host_bytes" else max(1, min(values)) * tolerance
            )
            if max(values) - min(values) > allowed:
                failures.append(
                    f"{field} range {min(values)}..{max(values)} exceeds {allowed}"
                )
        errors = any(
            not good(r)
            or num(r, "quarantines", 0)
            or num(r, "late_copy_failures", 0)
            or (r in stable and num(r, "op_pool_live", 0) != 0)
            for r in cases
        )
        duration = max((num(r, "elapsed_s", 0) for r in cases), default=0)
        no_progress = any(num(r, "completed_ops", 0) == 0 for r in stable)
        for previous, current in zip(stable, stable[1:]):
            if previous.get("epoch") == current.get("epoch") and num(
                current, "completed_ops", 0
            ) <= num(previous, "completed_ops", 0):
                no_progress = True
        epochs = {r.get("epoch", "0"): r.get("epoch_summary", {}) for r in stable}
        routing_bad = any(not routed(epoch) for epoch in epochs.values())
        complete = bool(cases) and all(r.get("complete") for r in cases)
        if not complete:
            missing.append("successful exit marker")
        if not epochs:
            missing.append("epoch summaries")
        if no_progress:
            failures.append("no per-interval engine progress")
        if routing_bad:
            failures.append("epoch routed legacy/sync or missing routing")
        status = (
            "FAIL"
            if failures or errors
            else (
                "N/A" if missing or len(stable) < 3 or duration < soak_min_s else "PASS"
            )
        )
        if status == "PASS" and any(
            e.get("engine") == "simulated" for e in epochs.values()
        ):
            status = "INFO"  # Tooling self-test, never G-7 acceptance.
        gate(
            "G-7",
            status,
            f"{key}: duration={duration}s, warmup={warmup_s}s, samples={len(stable)}, tolerance={tolerance}; instability={failures}, missing={missing}, errors={errors}, no_engine_progress={no_progress}",
            cases,
        )

    for row in [r for r in rows if r["step"] == "B-1"]:
        gate(
            "G-5",
            "N/A",
            "B-1 component costs recorded; missing role CPU, 10k ops/s and saturation mean residence/QD evidence",
            [row],
        )
    for name in ("G-1", "G-2", "G-3", "G-4", "G-5", "G-5a", "G-7"):
        if not any(g["gate"] == name for g in gates):
            gate(name, "N/A", "No applicable input rows", [])
    for name in ("G-6", "G-8"):
        gate(
            name,
            "N/A",
            "Requires external test/sanitizer and multi-filesystem approval; tooling cannot authorize a default flip",
            [],
        )
    comparisons = {
        "AM-5": [r for r in rows if r["step"] in ("B-0", "B-3", "B-4", "B-5", "B-6")],
        "AM-9": [r for r in rows if r["step"] == "B-0"],
        "AM-10": [r for r in rows if r["step"] in ("B-0", "B-9", "B-10")],
    }
    # Required coverage is independent of observed successes. In particular, writes
    # disabled in the runner must not silently remove B-6 from G-2.
    required = rows[0].get("expected", {}).get("required", []) if rows else []
    if not required:
        for mount, device in sorted(
            {
                (r["mount"], r["device"])
                for r in rows
                if r["step"].startswith("B-") or r["step"] == "G-7"
            }
        ):
            for name, steps in {
                "G-1": ["B-0"],
                "G-2": ["B-3", "B-4", "B-5", "B-6"],
                "G-3": ["B-3", "B-4", "B-5", "B-6"],
                "G-4": ["B-10"],
                "G-5": ["B-1", "B-3"],
                "G-7": ["G-7"],
            }.items():
                for step in steps:
                    cell = {
                        "gate": name,
                        "mount": mount,
                        "device": device,
                        "step": step,
                    }
                    if name in ("G-2", "G-3"):
                        cell["engine"] = "async"
                    if name == "G-4":
                        for arm in ("legacy", "async_auto"):
                            for threads in (1, 8):
                                for inode in ("shared", "distinct"):
                                    for workload in ("sustained", "single", "spill"):
                                        required.append(
                                            {
                                                **cell,
                                                "arm": arm,
                                                "writer_threads": str(threads),
                                                "inode_mode": inode,
                                                "workload": workload,
                                            }
                                        )
                    elif name == "G-7":
                        for workload in ("sustained", "restore", "spill"):
                            required.append({**cell, "workload": workload})
                    else:
                        required.append(cell)
    aggregates = []
    for (name, mount, device), cells in group(
        required, ("gate", "mount", "device")
    ).items():
        missing, statuses = [], []
        for cell in cells:
            matches = [
                g
                for g in gates
                if g["gate"] == name
                and any(
                    all(
                        (
                            num(r, k) == num({k: v}, k)
                            if k in ("qd", "op_mib", "threads", "writer_threads")
                            else str(r.get(k, "")) == str(v)
                        )
                        for k, v in cell.items()
                        if k != "gate"
                    )
                    for r in g["input_rows"]
                )
            ]
            if not matches:
                missing.append(cell)
            statuses.extend(g["status"] for g in matches)
        status = (
            "FAIL"
            if "FAIL" in statuses
            else (
                "N/A"
                if missing or not statuses or any(s != "PASS" for s in statuses)
                else "PASS"
            )
        )
        aggregates.append(
            {
                "gate": name,
                "mount": mount,
                "device": device,
                "status": status,
                "reason": f"missing cells={missing}; cell statuses={statuses}",
            }
        )
    failed_cases = [r for r in rows if r.get("exit_code", "0") != "0"]
    return {
        "gates": gates,
        "aggregates": aggregates,
        "failed_cases": failed_cases,
        "runner_failures": sorted(
            {r["runner_failures"] for r in rows if r.get("runner_failures")}
        ),
        "performance": performance,
        "comparisons": comparisons,
    }


def report(result):
    lines = [
        "# cuFile qualification report",
        "",
        "N/A is not a pass. Compat/simulator results are NOT GDS evidence.",
        "Raw nvidia-fs counters require isolated operator attribution (AM-11).",
        "",
        "## Failed commands (including partial CSVs)",
        "",
        *[f"- {r['source']}: rc={r.get('exit_code')}" for r in result["failed_cases"]],
        "",
        *result["runner_failures"],
        "",
        "## Required coverage (aggregate; N/A is incomplete, never PASS)",
        "",
        *[
            f"- {g['gate']} {g['mount']}/device={g['device']}: **{g['status']}** — {g['reason']}"
            for g in result["aggregates"]
        ],
        "",
        "## Gates (individual cells)",
        "",
        "| Gate | Result | Reason | Inputs |",
        "|---|---|---|---|",
    ]

    def esc(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for gate in result["gates"]:
        sources = sorted(
            {f"{r['source']}:{r.get('input_row', '?')}" for r in gate["input_rows"]}
        )
        lines.append(
            "| "
            + " | ".join(
                map(
                    esc,
                    (gate["gate"], gate["status"], gate["reason"], ", ".join(sources)),
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Matched baseline throughput",
        "",
        "| Mount | Workload | Variant | GiB/s | Baseline GiB/s | Ratio | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["performance"]:
        lines.append(
            "| "
            + " | ".join(
                esc(row[k])
                for k in (
                    "mount",
                    "workload",
                    "variant",
                    "gib_s",
                    "baseline_gib_s",
                    "ratio",
                    "source",
                )
            )
            + " |"
        )
    for name, rows in result["comparisons"].items():
        lines += [
            "",
            f"## {name} comparison",
            "",
            "| Mount/device | Workload/mode | Op MiB (effective) | Registration | QD | GiB/s | Content mismatches | Source |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            values = (
                f"{row['mount']}/{row['device']}",
                "/".join(
                    str(row[k])
                    for k in ("workload", "mode", "engine", "variant", "version")
                    if row.get(k)
                ),
                f"{row.get('op_mib', '?')} ({num(row, 'S_eff_max', 0) / 2**20})",
                row.get("registration", row.get("variant", "")),
                row.get("qd", "?"),
                row.get("gib_s", "?"),
                row.get("data_mismatches", "not reported"),
                row["source"],
            )
            lines.append("| " + " | ".join(map(esc, values)) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--expect",
        type=Path,
        help="Explicit required-cell JSON, including b10_repetitions",
    )
    parser.add_argument("--warmup-s", type=float, default=60)
    parser.add_argument("--soak-min-s", type=float, default=3600)
    parser.add_argument("--resource-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if (
        not args.results.is_dir()
        or min(args.warmup_s, args.soak_min_s, args.resource_tolerance) < 0
    ):
        parser.error("results must be a directory; thresholds must be nonnegative")
    rows = load_rows(args.results)
    if args.expect:
        expected = json.loads(args.expect.read_text())
        for row in rows:
            row["expected"] = expected
    result = analyze(rows, args.warmup_s, args.soak_min_s, args.resource_tolerance)
    (args.results / "gates.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.results / "report.md").write_text(report(result))
    print(args.results / "report.md")


if __name__ == "__main__":
    main()
