#!/usr/bin/env python3
"""Independent saved-record audit; never executes SQL, builds or CUDA."""
import csv
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

base = Path("/tmp/cucascade-pinned-pr-20261012")
reported = json.loads((base / "CORRECTED-RESULTS.json").read_text())
gate = json.loads((base / "sirius-build-corrected/INDEPENDENT-GATE.json").read_text())
build = json.loads((base / "sirius-build-corrected/build-metadata.json").read_text())
original = base / "sirius-benchmark"
schedule = [[f"noise-{i}", "B"] for i in range(3)] + [[f"ab-{i}", arm] for i, arm in enumerate("BCCBCBBC")]
all_digests = {}
all_sql_hashes = {}
all_runs = []
result = {"verdict": "PASS", "policies": {}, "endpoint_hashes": {}, "checks": {}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for policy in ["sirius", "none"]:
    root = base / ("corrected-input-" + policy)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert manifest["schedule"] == schedule
    assert [[r["name"], r["candidate"]] for r in manifest["runs"]] == schedule
    assert len(manifest["runs"]) == 11
    expected_config = (original / "config.yaml").read_text().replace("mode: sirius", "mode: " + policy)
    assert (root / "config.yaml").read_text() == expected_config
    assert sha(root / "config.yaml") == manifest["config_sha256"]
    assert sha(root / "validated_runner.py") == manifest["runner_sha256"] == sha(original / "validated_runner.py")
    summaries = []
    for record in manifest["runs"]:
        all_runs.append(record)
        bench = Path(record["benchmark_directory"])
        raw = list(csv.DictReader((bench / "csv/runtimes.csv").open()))
        rows = [r for r in raw if r["query"] != "TOTAL"]
        assert len(rows) == 88 and all(r["engine"] == "sirius" for r in rows)
        groups = {q: {} for q in range(1, 23)}
        for row in rows:
            query = int(row["query"][1:])
            iteration = int(row["iteration"])
            seconds = float(row["runtime_s"])
            assert query in groups and iteration not in groups[query]
            assert math.isfinite(seconds) and seconds > 0
            groups[query][iteration] = seconds * 1000
        assert all(sorted(g) == [0, 1, 2, 3] for g in groups.values())
        checks = [json.loads(line) for line in Path(record["iteration_validation"]).read_text().splitlines()]
        assert len(checks) == 88 and all(r["correct"] is True for r in checks)
        for row, check in zip(rows, checks):
            assert int(row["query"][1:]) == check["query"]
            assert abs(float(row["runtime_s"]) - check["seconds"]) <= 0.000000501
            all_digests.setdefault(check["query"], set()).add((check["rows"], check["digest"]))
        validation = list(csv.DictReader((bench / "validation.csv").open()))
        assert len(validation) == 22 and all(row["status"] == "success" for row in validation)
        for q in range(1, 23):
            query_file = bench / "queries" / f"q{q}.sql"
            all_sql_hashes.setdefault(q, set()).add(sha(query_file))
        assert (bench / "config.yml").read_text() == expected_config
        command = record["command"]
        assert record["exit_status"] == 0 and "--profile" not in command and "--mode" not in command
        assert "SIRIUS_PRE_SQL=SET enable_duckdb_fallback=false;" in command
        assert command[command.index("--iterations") + 1] == "4"
        assert command[command.index("--pin") + 1] == "none"
        assert command[command.index("--input") + 1] == "/tmp/sirius-pr1834-streams-20261007-1940/data/tpch_sf50"
        assert command[command.index("--queries") + 1] == "1-22"
        tag = record["candidate"]
        assert "SIRIUS_BUILD_PATH=" + str(base / "sirius-runtime-corrected" / tag) in command
        log = "\n".join(p.read_text(errors="replace") for p in (bench / "log_dir").rglob("*") if p.is_file())
        counts = {"starts": log.count("Transparent GPU execution: executing query"),
                  "completed": log.count("Transparent GPU execution: query completed"),
                  "engine_queries": len(re.findall(r"query \d+ telemetry_query=", log)),
                  "errors": re.findall(r"fallback to|Error executing query|Unknown error executing|sweep error", log, re.I)}
        assert counts == record["GPU_path"]
        assert all(counts[k] == 88 for k in ["starts", "completed", "engine_queries"]) and not counts["errors"]
        runner_log = (root / (record["name"] + ".log")).read_text()
        assert runner_log.count("Dropping OS page cache") == 1
        assert runner_log.count("posix_fadvise(DONTNEED) applied to 10 file(s)") == 1
        assert "Resetting Sirius prefetching cache" not in runner_log
        hot_per_query = {str(q): st.median(groups[q][i] for i in [1, 2, 3]) for q in groups}
        assert all(abs(hot_per_query[q] - record["per_query_hot_median_ms"][q]) < 1e-8 for q in hot_per_query)
        hot_sum = sum(hot_per_query.values())
        assert abs(hot_sum - record["FOM_ms"]) < 1e-8
        summaries.append({"name": record["name"], "arm": tag, "hot_ms": hot_sum,
                          "first_sum_ms": sum(g[0] for g in groups.values()),
                          "q1_first_ms": groups[1][0], "hot_per_query": hot_per_query})
    paired = [r for r in summaries if r["name"].startswith("ab-")]
    noise = [r["hot_ms"] for r in summaries if r["name"].startswith("noise-")]
    first_noise = [r["first_sum_ms"] for r in summaries if r["name"].startswith("noise-")]
    arms = {}
    for tag in ["B", "C"]:
        records = [r for r in paired if r["arm"] == tag]
        hot = [r["hot_ms"] for r in records]
        first = [r["first_sum_ms"] for r in records]
        arms[tag] = {"hot_process_ms": hot, "hot_median_ms": st.median(hot),
                     "first_process_ms": first, "first_median_ms": st.median(first),
                     "q1_first_median_ms": st.median(r["q1_first_ms"] for r in records)}
    pairs = []
    for i in range(0, 8, 2):
        block = paired[i:i + 2]
        b = next(r for r in block if r["arm"] == "B")
        c = next(r for r in block if r["arm"] == "C")
        pairs.append({"runs": [r["name"] for r in block],
                      "hot_reduction_percent": 100 * (1 - c["hot_ms"] / b["hot_ms"]),
                      "first_reduction_percent": 100 * (1 - c["first_sum_ms"] / b["first_sum_ms"])})
    med_noise = st.median(noise)
    metrics = {"arms": arms, "pairs": pairs,
               "hot_reduction_percent": 100 * (1 - arms["C"]["hot_median_ms"] / arms["B"]["hot_median_ms"]),
               "first_reduction_percent": 100 * (1 - arms["C"]["first_median_ms"] / arms["B"]["first_median_ms"]),
               "q1_first_reduction_percent": 100 * (1 - arms["C"]["q1_first_median_ms"] / arms["B"]["q1_first_median_ms"]),
               "noise_hot_ms": noise, "noise_hot_median_ms": med_noise,
               "noise_hot_range_ms": max(noise) - min(noise),
               "noise_hot_range_percent": 100 * (max(noise) - min(noise)) / med_noise,
               "noise_hot_MAD_ms": st.median(abs(n - med_noise) for n in noise),
               "noise_first_ms": first_noise,
               "noise_first_range_percent": 100 * (max(first_noise) - min(first_noise)) / st.median(first_noise),
               "per_query_hot_medians": {q: {tag: st.median(r["hot_per_query"][q] for r in paired if r["arm"] == tag)
                                            for tag in ["B", "C"]} for q in map(str, range(1, 23))}}
    expected = reported[policy]
    assert abs(metrics["hot_reduction_percent"] - expected["time_reduction_pct"]) < 1e-8
    assert abs(metrics["first_reduction_percent"] - expected["first_per_query_time_reduction_pct"]) < 1e-8
    assert abs(metrics["noise_hot_range_percent"] - expected["baseline_noise_range_pct"]) < 1e-8
    for tag in ["B", "C"]:
        assert all(abs(x-y) < 1e-8 for x,y in zip(arms[tag]["hot_process_ms"], expected["process_FOM_ms"][tag]))
        assert abs(arms[tag]["hot_median_ms"] - expected["median_FOM_ms"][tag]) < 1e-8
    for q, medians in metrics["per_query_hot_medians"].items():
        assert abs(medians["B"] - expected["per_query_medians"][q]["baseline_ms"]) < 1e-8
        assert abs(medians["C"] - expected["per_query_medians"][q]["candidate_ms"]) < 1e-8
    result["policies"][policy] = metrics
    for tag, item in manifest["extensions"].items():
        assert sha(item["path"]) == item["sha256"] == gate["hashes"]["baseline" if tag == "B" else "candidate"]
        runtime = base / "sirius-runtime-corrected" / tag / "extension/sirius/sirius.duckdb_extension"
        assert runtime.resolve() == Path(item["path"])
        result["endpoint_hashes"][item["path"]] = sha(item["path"])

expected_global = [(policy, f"noise-{i}", "B") for policy in ["sirius", "none"] for i in range(3)]
expected_global += [(policy, f"ab-{i}", tag) for i, tag in enumerate("BCCBCBBC") for policy in ["sirius", "none"]]
assert [(r["policy"], r["name"], r["candidate"]) for r in sorted(all_runs, key=lambda r: r["start_epoch"])] == expected_global
assert len(all_digests) == 22 and all(len(d) == 1 for d in all_digests.values())
assert len(all_sql_hashes) == 22 and all(len(s) == 1 for s in all_sql_hashes.values())
for path, digest in json.loads((original / "canonical-harness-hashes.json").read_text()).items():
    assert sha(path) == digest
    result["endpoint_hashes"][path] = digest
for path, digest in build["protected_before"].items():
    assert sha(path) == digest == build["protected_after"][path]
    result["endpoint_hashes"][path] = digest
assert sha(base / "sirius-build-corrected/build-metadata.json") == gate["metadata_sha256"]
assert sha(base / "sirius-build-corrected/build.py") == gate["build_script_sha256"]
result["checks"] = {"fixed_full_schedule": "PASS", "noise_plus_timing_iterations": 1936,
                    "paired_iterations": 1408, "hot_iterations_in_FOM": 1056, "first_iterations_excluded": 352,
                    "GPU_starts_completions_engine_entries": "88 each per accepted process",
                    "all_result_digests_and_row_counts": "equal across policies/arms/processes",
                    "all_rendered_SQL": "byte-hash equal across all 22 processes",
                    "adapter_and_canonical_source_hashes": "unchanged",
                    "corrected_artifact_and_production_original_hashes": "unchanged",
                    "configuration": "exact original except input-cache mode",
                    "cache_regime": "initial advisory eviction once; no inter-query/iteration reset; no uniformly cold-system claim",
                    "old_artifact_measurements": "excluded from normal-consumer inference and not pooled"}
(base / "CORRECTED-INDEPENDENT-RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
for policy, m in result["policies"].items():
    compact = {key:value for key,value in m.items() if key not in ["per_query_hot_medians"]}
    print(policy, json.dumps(compact, indent=2))
print(json.dumps(result["checks"], indent=2))
print("WROTE", base / "CORRECTED-INDEPENDENT-RESULTS.json")
