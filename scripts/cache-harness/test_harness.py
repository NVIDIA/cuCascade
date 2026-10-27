#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regression controls for harness verdicts; no GPU or service binaries required."""
import argparse
import io
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.dont_write_bytecode = True
import harness as origin
import ais_harness as ais
import ais_test_lifecycle as faults


class HarnessControls(unittest.TestCase):
    def test_unrelated_failures_never_qualify(self):
        for kind in (
            "redirect",
            "timeout",
            "truncated",
            "malformed-range",
            "changed-validator",
            "wrong-block",
        ):
            for code in (-11, 1, 127, None):
                result = dict(status="FAIL", exit=code, result={"stage": "setup"})
                with self.subTest(kind=kind, code=code):
                    self.assertEqual(
                        faults.classify_fault(
                            kind, result, [], "unrelated startup crash", 1
                        ),
                        "FAIL",
                    )
                    calls = [dict(method="HEAD"), dict(method="GET", fault_sent=kind)]
                    self.assertEqual(
                        faults.classify_fault(
                            kind, result, calls, "unrelated startup crash", 1
                        ),
                        "FAIL",
                    )

    def test_rejection_requires_stage_delivery_and_error(self):
        good = dict(status="FAIL", exit=1, result={"stage": "read"})
        calls = [dict(method="HEAD"), dict(method="GET", fault_sent="malformed-range")]
        error = "rest_reactor: 206 Content-Range mismatch"
        self.assertEqual(
            faults.classify_fault("malformed-range", good, calls, error, 1), "PASS"
        )
        for result, trace, text, elapsed in (
            (dict(status="PASS", exit=0), calls, "", 1),
            (good, [dict(method="HEAD"), dict(method="GET")], error, 1),
            (good, calls, "unrelated startup crash", 1),
            (good, calls, error, 45),
            (dict(good, result={"stage": "open"}), calls, error, 1),
        ):
            self.assertEqual(
                faults.classify_fault("malformed-range", result, trace, text, elapsed),
                "FAIL",
            )

    def test_validator_limitation_requires_successful_faulted_read(self):
        good = dict(status="PASS", exit=0)
        calls = [
            dict(method="HEAD"),
            dict(method="GET", fault_sent="changed-validator"),
        ]
        self.assertEqual(
            faults.classify_fault("changed-validator", good, calls, "", 1),
            "UNSUPPORTED",
        )
        self.assertEqual(
            faults.classify_fault("changed-validator", good, calls[:1], "", 1), "FAIL"
        )

    def test_real_suite_rejects_mocked_startup_crashes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            origin, "probe", return_value=dict(status="FAIL", exit=-11)
        ):
            with self.assertRaisesRegex(AssertionError, "protocol regression failed"):
                faults.library_faults(
                    argparse.Namespace(binary=Path("/unused")), Path(tmp)
                )
            rows = origin.load(Path(tmp) / "library-faults.json")
            self.assertTrue(
                all(r["status"] == "FAIL" and not r["requests"] for r in rows)
            )

    def test_probe_startup_and_outer_timeout_are_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for error in (
                FileNotFoundError("missing binary"),
                subprocess.TimeoutExpired("probe", 45),
            ):
                with patch.object(origin.subprocess, "run", side_effect=error):
                    result = origin.probe(
                        Path("/missing"),
                        dict(bucket="b", endpoint="http://127.0.0.1:1"),
                        dict(key="k", file="expected", etag="v1"),
                        Path(tmp),
                        "failure",
                    )
                self.assertEqual(result["status"], "FAIL")
                self.assertNotIn("exit", result)
                self.assertIn("error", result)

    def test_cold_status(self):
        self.assertEqual(ais.cold_fill_status("PASS", 404, 200, 2, 3), "PASS")
        for before, after, a, b, expected in (
            (200, 200, 2, 2, "FAIL"),
            (404, 404, 2, 3, "FAIL"),
            (404, 200, 2, 2, "FAIL"),
            (404, 200, None, 3, "UNKNOWN"),
            (404, 200, 2, None, "UNKNOWN"),
            (404, 200, True, 2, "UNKNOWN"),
        ):
            self.assertEqual(
                ais.cold_fill_status("PASS", before, after, a, b), expected
            )
        self.assertEqual(ais.cold_fill_status("FAIL", 404, 200, 2, 3), "FAIL")

    def test_qualify_cannot_accept_warm_or_unobservable_cpp_objects(self):
        for before, counts, verdict in (
            (200, (2, 2), "FAIL"),
            (404, (None, None), "UNKNOWN"),
        ):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                (run / "origin").mkdir()
                data = b"known reference fixture bytes"
                file = run / "fixture.bin"
                file.write_bytes(data)
                fixtures = [
                    dict(key=key, file=str(file), etag='"v1"')
                    for key in (
                        "ordinary/65553.bin",
                        "ordinary/1048607.bin",
                        "qualification/footer-first.bin",
                    )
                ]
                origin.save(run / "origin/fixtures.json", fixtures)
                origin.save(
                    run / "ais-state.json",
                    dict(
                        schema=1,
                        run_dir=str(run),
                        bucket="b",
                        mode="reverse",
                        endpoint="http://127.0.0.1:1",
                    ),
                )
                origin.save(
                    run / "origin/state.json",
                    dict(schema=1, run_dir=str(run / "origin"), bucket="b"),
                )

                def snap(run, label, *args, **kwargs):
                    code, body = 200, data
                    if label in ("presence-before", "s3-missing"):
                        code = 404
                    if label.startswith("cucascade-") and label.endswith(
                        "presence-before"
                    ):
                        code = before
                    if label == "s3-invalid-range":
                        code = 416
                    if label == "s3-valid-range":
                        code, body = 206, data[7:39]
                    if label in ("fresh-v1", "fresh-ordinary"):
                        body = b"p0-version-one"
                    if label == "fresh-latest":
                        body = b"p0-version-two-longer"
                    if label == "target-stopped":
                        code = 400
                    return dict(status=code, chain=[]), body

                def stats(run, label):
                    if label.startswith("cucascade-"):
                        return counts[0 if label.endswith("before") else 1]
                    return 0 if label == "stats-before" else 1

                s3 = MagicMock()
                s3.put_object.side_effect = [{"ETag": "v1"}, {"ETag": "v2"}]
                with patch.object(ais, "snapshot", snap), patch.object(
                    ais, "stats", stats
                ), patch.object(
                    origin, "probe", return_value=dict(status="PASS", exit=0)
                ), patch.object(
                    origin, "client", return_value=s3
                ), patch.object(
                    origin, "down"
                ), patch.object(
                    ais, "stop_node"
                ):
                    rows = ais.qualify(
                        argparse.Namespace(run_dir=run, binary=Path("/unused"))
                    )
                self.assertEqual(
                    next(r for r in rows if r["case"] == "A2-cucascade")["status"],
                    "PASS",
                )
                for case in ("A2-cucascade-cold-fill", "A2-footer-first-cold-fill"):
                    self.assertEqual(
                        next(r for r in rows if r["case"] == case)["status"], verdict
                    )
                self.assertEqual(origin.load(run / "summary.json")["status"], "FAIL")

    def test_seed_distinguishes_offset_and_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "fixtures").mkdir()
            origin.save(
                run / "state.json", dict(schema=1, run_dir=str(run), bucket="b")
            )
            objects = {}
            s3 = MagicMock()

            def put(**kw):
                objects[kw["Key"]] = kw["Body"]
                return {"ETag": '"tag"'}

            s3.put_object.side_effect = put
            s3.get_object.side_effect = lambda **kw: {
                "Body": io.BytesIO(objects[kw["Key"]])
            }
            with patch.object(origin, "client", return_value=s3):
                origin.seed(run)
            data = objects["ordinary/1048607.bin"]
            self.assertNotEqual(data[256:], data[:-256])
            self.assertEqual(
                len({data[i : i + 65536] for i in range(0, 1048576, 65536)}), 16
            )
            self.assertEqual(
                data, origin.fixture_bytes("ordinary/1048607.bin", len(data))
            )
            self.assertNotEqual(data, objects["qualification/footer-first.bin"])

    def test_actual_origin_launch_interruption_and_state_failure(self):
        # Real child; inject precisely between Popen and persisted ownership.
        for failure in (
            "identity-interrupt",
            "save-error",
            "sigterm",
            "sigint",
            "sigalrm",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / "run"
                spawned = []
                real_popen = subprocess.Popen
                real_save = origin.save

                def spawn(*args, **kwargs):
                    proc = real_popen(
                        [sys.executable, "-c", "import time; time.sleep(120)"],
                        start_new_session=True,
                    )
                    spawned.append(proc)
                    return proc

                def identity(pid):
                    if failure == "identity-interrupt":
                        raise KeyboardInterrupt("injected")
                    if failure.startswith("sig"):
                        signal.raise_signal(getattr(signal, failure.upper()))
                    return "test-start"

                def save(path, value):
                    if (
                        failure == "save-error"
                        and path.name == "state.json"
                        and "pid" in value
                    ):
                        raise OSError("injected disk failure")
                    return real_save(path, value)

                def interrupted(signum, frame):
                    raise ais.RunInterrupted(str(signum))

                previous = {
                    sig: signal.signal(sig, interrupted)
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
                }
                try:
                    with patch.object(
                        origin, "preflight", return_value={"weed": "/unused"}
                    ), patch.object(origin.socket, "socket"), patch.object(
                        origin.subprocess, "Popen", spawn
                    ), patch.object(
                        origin, "process_identity", identity
                    ), patch.object(
                        origin, "save", save
                    ), patch.object(
                        origin, "down"
                    ):
                        with self.assertRaises(
                            (KeyboardInterrupt, OSError, ais.RunInterrupted)
                        ):
                            origin.up(argparse.Namespace(run_dir=run, port_base=20000))
                    self.assertEqual(len(spawned), 1)
                    self.assertIsNotNone(spawned[0].poll(), "launch leaked its child")
                    self.assertEqual(signal.getsignal(signal.SIGTERM), interrupted)
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
                    for child in spawned:
                        if child.poll() is None:
                            child.kill()
                            child.wait(timeout=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
