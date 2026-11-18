# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "analyze_gates.py"
FIXTURES = Path(__file__).parent / "fixtures"
spec = importlib.util.spec_from_file_location("analyze_gates", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GatesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for fixture in FIXTURES.glob("*.csv"):
            shutil.copy(fixture, self.root)
        (self.root / "expected.json").write_text(json.dumps({"b10_repetitions": 2}))
        (self.root / "soak.csv.ok").write_text("complete")
        for epoch in range(4):
            (self.root / f"soak.csv.epoch-{epoch}.csv").write_text(
                "engine,workload,gib_s,verify,engine_logical_bytes,legacy_bytes,sync_bytes\n"
                "async,sustained,1,1,1000,0,0\n"
            )

    def rows(self, direct=True):
        if direct:
            proof = {
                p.name: {
                    "qualified": True,
                    "reason": "isolated fixture",
                    "artifacts": ["fixture"],
                }
                for p in self.root.glob("*.csv")
            }
            (self.root / "direct-evidence.json").write_text(json.dumps(proof))
        return module.load_rows(self.root)

    def gates(self, name, rows=None, **kwargs):
        result = module.analyze(self.rows() if rows is None else rows, **kwargs)
        return [g for g in result["gates"] if g["gate"] == name]

    def test_matched_direction_mount_geometry(self):
        result = module.analyze(self.rows())
        self.assertEqual([p["ratio"] for p in result["performance"]], [0.8, 0.6])
        self.assertEqual(
            [g["status"] for g in result["gates"] if g["gate"] == "G-2"],
            ["PASS", "FAIL"],
        )
        for performance in result["performance"]:
            self.assertEqual(len(performance["matched_rows"]), 1)

    def test_no_direct_evidence_never_passes_gds(self):
        rows = self.rows(direct=False)
        self.assertTrue(all(g["status"] == "N/A" for g in self.gates("G-2", rows)))
        self.assertNotIn("PASS", [g["status"] for g in self.gates("G-1", rows)])

    def test_baseline_also_needs_direct_evidence(self):
        rows = self.rows()
        for row in rows:
            if row["step"] == "B-2":
                row["direct_qualified"] = False
        self.assertTrue(all(g["status"] == "N/A" for g in self.gates("G-2", rows)))

    def test_scaling_and_content_failure(self):
        self.assertEqual([g["status"] for g in self.gates("G-1")], ["PASS", "FAIL"])
        self.assertEqual(sum(g["status"] == "FAIL" for g in self.gates("AM-9")), 1)

    def test_qd_requires_backlog_time_and_uses_integral_mean(self):
        self.assertEqual([g["status"] for g in self.gates("G-3")], ["PASS", "FAIL"])
        rows = self.rows()
        for row in rows:
            row["backlog_ns"] = "0"
        self.assertTrue(all(g["status"] == "N/A" for g in self.gates("G-3", rows)))

    def test_query_ceiling_inclusive(self):
        self.assertEqual([g["status"] for g in self.gates("G-5a")], ["PASS", "FAIL"])
        self.assertIn("residence_p99_us=512", self.gates("G-5a")[0]["reason"])

    def test_compat_paired_boundary_and_incomplete(self):
        self.assertEqual([g["status"] for g in self.gates("G-4")], ["PASS", "FAIL"])
        rows = [
            r
            for r in self.rows()
            if not (r.get("iteration") == "2" and r.get("version") == "new")
        ]
        self.assertEqual(self.gates("G-4", rows)[0]["status"], "N/A")

    def test_soak_after_warmup_and_recreation(self):
        self.assertEqual(self.gates("G-7")[0]["status"], "PASS")
        rows = self.rows()
        rows = [
            r for r in rows if not (r["step"] == "G-7" and r["elapsed_s"] == "3600")
        ]
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "N/A")

    def test_soak_leak_is_failure_even_short(self):
        rows = self.rows()
        for row in rows:
            if row["step"] == "G-7" and row["elapsed_s"] == "120":
                row["pinned_host_bytes"] = "2048"
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "FAIL")

    def test_thread_growth_is_not_covered_by_byte_tolerance(self):
        rows = self.rows()
        for row in rows:
            if row["step"] == "G-7" and row["elapsed_s"] == "120":
                row["thread_count"] = "13"
        self.assertEqual(self.gates("G-7", rows, tolerance=0.5)[0]["status"], "FAIL")

    def test_standalone_epoch_not_counted_as_acceptance(self):
        (self.root / "bench.csv").write_text(
            "engine,workload,gib_s\nasync,sustained,1\n"
        )
        (self.root / "bench.csv.epoch-0.csv").write_text(
            "engine,workload,gib_s\nasync,sustained,1000\n"
        )
        rows = self.rows()
        self.assertTrue(
            any(r["source"] == "bench.csv" and r["step"] == "B-3" for r in rows)
        )
        self.assertFalse(any(".epoch-" in r["source"] for r in rows))

    def test_invalid_nan_and_missing_baseline(self):
        self.assertIsNone(module.num({"n": "nan"}, "n"))
        rows = self.rows()
        for row in rows:
            if row["step"] == "B-3":
                row["S_eff_max"] = str(2**20)
        self.assertEqual(self.gates("G-2", rows)[0]["status"], "N/A")

    def test_failed_command_not_hidden_by_partial_csv(self):
        rows = self.rows()
        for row in rows:
            if row["step"] == "B-3":
                row["exit_code"] = "1"
        self.assertEqual(self.gates("G-2", rows)[0]["status"], "FAIL")

    def test_manifest_gdsio_and_comments(self):
        shutil.copy(FIXTURES / "gdsio.txt", self.root / "baseline.log")
        (self.root / "manifest.tsv").write_text(
            "step\tmount\tfs_class\tdevice\tqd\top_mib\tdirection\tvariant\tpath\texit_code\nB-2\tnvme\tnvme\t0\t32\t4\tread\tgdsio\tbaseline.log\t0\n"
        )
        rows = self.rows()
        self.assertTrue(
            any(
                r["source"] == "baseline.log" and module.num(r, "gib_s") == 10
                for r in rows
            )
        )
        p0 = self.root / "comments.csv"
        p0.write_text("# version preamble\nkind,mode,qd,gib_per_s\nresult,async,1,2\n")
        self.assertEqual(module.csv_rows(p0)[0]["gib_per_s"], "2")

    def test_legacy_and_simulated_excluded_from_g2(self):
        rows = self.rows()
        for row in rows:
            if row["step"] in ("B-3", "B-6"):
                row["engine"] = "legacy"
        self.assertTrue(all(not g["input_rows"] for g in self.gates("G-2", rows)))

    def test_legacy_routing_cannot_pass_engine_gates(self):
        for legacy in ("1", "1000000"):
            rows = self.rows()
            for r in rows:
                if r.get("engine") == "async":
                    r["legacy_bytes"] = legacy
            self.assertNotIn("PASS", [g["status"] for g in self.gates("G-2", rows)])
            self.assertNotIn("PASS", [g["status"] for g in self.gates("G-5a", rows)])

    def test_demoted_engine_cell_fails(self):
        rows = self.rows()
        for r in rows:
            if r["step"] == "B-3":
                r.update(legacy_bytes="1", policy_demotions="1")
        self.assertEqual(self.gates("G-2", rows)[0]["status"], "FAIL")

    def test_compat_g3_informational(self):
        rows = self.rows(False)
        self.assertEqual(
            [g["status"] for g in self.gates("G-3", rows)], ["INFO", "INFO"]
        )

    def test_refill_gaps_checked(self):
        rows = self.rows()
        for r in rows:
            r["refill_gap_ns"] = r.get("backlog_ns", "0")
        self.assertEqual(self.gates("G-3", rows)[0]["status"], "FAIL")

    def test_g5_never_passes_partial_metrics(self):
        self.assertNotIn("PASS", [g["status"] for g in self.gates("G-5")])

    def test_missing_write_cell_prevents_aggregate_pass(self):
        result = module.analyze([r for r in self.rows() if r["step"] != "B-6"])
        self.assertTrue(
            all(
                g["status"] != "PASS"
                for g in result["aggregates"]
                if g["gate"] == "G-2"
            )
        )

    def test_b10_expected_pairs_and_failures(self):
        rows = self.rows()
        for r in rows:
            r["expected"] = {"b10_repetitions": 5}
        self.assertNotIn("PASS", [g["status"] for g in self.gates("G-4", rows)])
        for r in rows:
            r["runner_failures"] = "failed iteration"
            r["expected"] = {"b10_repetitions": 2}
        self.assertEqual(self.gates("G-4", rows)[0]["status"], "FAIL")

    def test_soak_idle_epoch_and_legacy_epoch_fail(self):
        rows = self.rows()
        for r in rows:
            if r["step"] == "G-7":
                r["completed_ops"] = "0"
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "FAIL")
        rows = self.rows()
        for r in rows:
            if r["step"] == "G-7":
                r["epoch_summary"]["legacy_bytes"] = "1000"
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "FAIL")

    def test_soak_requires_exit_marker_and_exact_arena_bytes(self):
        (self.root / "soak.csv.ok").unlink()
        self.assertEqual(self.gates("G-7")[0]["status"], "N/A")
        rows = self.rows()
        for r in rows:
            if r.get("elapsed_s") == "120":
                r["arena_bytes"] = "1025"
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "FAIL")

    def test_verify_required(self):
        rows = self.rows()
        for r in rows:
            r["verify"] = "0"
        self.assertNotIn("PASS", [g["status"] for g in self.gates("G-2", rows)])

    def test_arena_requires_direct_evidence(self):
        rows = self.rows(False)
        for r in rows:
            r["registration"] = "arena"
        self.assertNotIn("PASS", [g["status"] for g in self.gates("AM-9", rows)])

    def test_unknown_csv_rejected(self):
        (self.root / "stray.csv").write_text("gib_s,who\n1,unknown\n")
        with self.assertRaises(ValueError):
            self.rows()

    def test_short_or_unflushed_g2_is_not_acceptance(self):
        rows = self.rows()
        for r in rows:
            r["seconds"] = "3"
        self.assertNotIn("PASS", [g["status"] for g in self.gates("G-2", rows)])
        rows = self.rows()
        for r in rows:
            r["flush"] = "0"
        self.assertEqual(self.gates("G-2", rows)[1]["status"], "N/A")

    def test_simulator_soak_is_only_a_self_test(self):
        rows = self.rows()
        for r in rows:
            if r["step"] == "G-7":
                r["epoch_summary"]["engine"] = "simulated"
        self.assertEqual(self.gates("G-7", rows)[0]["status"], "INFO")

    def test_cli_reports_input_rows_and_amendment_tables(self):
        subprocess.run(
            [sys.executable, str(TOOL), str(self.root)], check=True, capture_output=True
        )
        data = json.loads((self.root / "gates.json").read_text())
        self.assertIn("input_rows", data["gates"][0])
        report = (self.root / "report.md").read_text()
        for text in (
            "AM-5",
            "AM-9",
            "AM-10",
            "Content mismatches",
            "N/A is not a pass",
        ):
            self.assertIn(text, report)


if __name__ == "__main__":
    unittest.main()
