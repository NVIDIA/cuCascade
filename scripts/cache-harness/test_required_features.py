#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict-status tests plus semantic mutations of real H receipts via both CLIs."""
import copy
import json
import os
import signal
import time
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import required_features as required
from check_required_baseline import check

PACKET = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
if PACKET:
    sys.argv = sys.argv[:1]


class StrictTests(unittest.TestCase):
    def test_mandatory_statuses(self):
        rows = [dict(id=name, status="PASS") for name in required.REQUIRED]
        controls = [dict(id="healthy", status="PASS")]
        self.assertEqual(required.qualification_status(rows, controls), "PASS")
        for status in (
            "UNSUPPORTED",
            "UNKNOWN",
            "NOT_TESTED",
            "TIMEOUT",
            "FAIL",
            "PASS_WITH_LIMITATIONS",
            "",
            None,
        ):
            for index in range(len(rows)):
                with self.subTest(status=status, index=index):
                    changed = copy.deepcopy(rows)
                    changed[index]["status"] = status
                    self.assertEqual(
                        required.qualification_status(changed, controls), "FAIL"
                    )
        for altered in (
            rows[:-1],
            rows + [rows[0]],
            rows + [dict(id="extra", status="PASS")],
        ):
            self.assertEqual(required.qualification_status(altered, controls), "FAIL")
        self.assertEqual(required.qualification_status(rows, []), "FAIL")
        self.assertEqual(
            required.qualification_status(rows, [dict(id="healthy", status="FAIL")]),
            "FAIL",
        )

    def test_baseline_is_not_qualification(self):
        rows = [dict(id=name, status="UNSUPPORTED") for name in required.REQUIRED]
        for r in rows:
            if r["id"] == "list-etag":
                r["status"] = "PASS"
            elif r["id"].endswith("-cold"):
                r["status"] = "NOT_TESTED"
        good = dict(
            status="FAIL",
            rows=rows,
            controls=[dict(id=str(i), status="PASS") for i in range(31)],
        )
        self.assertEqual(check(good)["baseline_check"], "PASS")
        for key in ("status", "rows", "controls"):
            bad = copy.deepcopy(good)
            if key == "status":
                bad[key] = "PASS"
            else:
                bad[key][0]["status"] = "FAIL"
            self.assertEqual(check(bad)["baseline_check"], "FAIL")

    @unittest.skipUnless(sys.platform == "linux", "Linux child lifecycle check")
    def test_interrupted_probe_is_reaped(self):
        scripts = Path(__file__).parent.resolve()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fake = root / "probe"
                pidfile = root / "pid"
                fake.write_text(
                    "#!/usr/bin/env python3\nimport os,time,pathlib\npathlib.Path("
                    + repr(str(pidfile))
                    + ").write_text(str(os.getpid()))\ntime.sleep(60)\n"
                )
                fake.chmod(0o700)
                code = (
                    "import sys;sys.path.insert(0,"
                    + repr(str(scripts))
                    + ");from pathlib import Path;import required_fixture as f;f.install_signals();f.invoke(Path("
                    + repr(str(fake))
                    + "),'http://127.0.0.1:1','s3://h/o','-','-','head',1,Path("
                    + repr(str(root))
                    + "))"
                )
                parent = subprocess.Popen(
                    [sys.executable, "-B", "-c", code],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pidfile.exists())
                    child = int(pidfile.read_text())
                    parent.send_signal(sig)
                    self.assertNotEqual(parent.wait(timeout=8), 0)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child, 0)
                finally:
                    if parent.poll() is None:
                        parent.kill()
                        parent.wait()

    @unittest.skipUnless(PACKET, "pass packet.json to run real-receipt mutations")
    def test_real_receipts_and_cli_mutations(self):
        scripts = Path(__file__).parent
        base = required.evaluate(PACKET)
        self.assertEqual(check(base)["baseline_check"], "PASS")
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(scripts / "check_required_baseline.py"),
                str(PACKET),
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        strict = subprocess.run(
            [sys.executable, "-B", str(scripts / "required_features.py"), str(PACKET)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(strict.returncode, 1)
        self.assertEqual(json.loads(strict.stdout)["status"], "FAIL")
        # Synthetic success derived from a real two-range receipt exercises the
        # future delegated wire oracle; it is NOT a successful library run.
        metadata = required.read(PACKET)
        o = required.read(
            PACKET.parent / metadata["wire_dir"] / "rewrite-range/observation.json"
        )
        delegated = copy.deepcopy(o)
        targets = []
        for w in delegated["wire"]:
            if w["method"] == "GET":
                w["if_match"] = '"h-fixture-version-1"'
                t = copy.deepcopy(w)
                t.update(
                    role="target-0",
                    credential_headers=[],
                    response=206,
                    path="/s3/h/object?pid=fixture&utm=fixture",
                )
                targets.append(t)
        delegated["wire"].extend(targets)
        delegated["probe"]["exit"] = 0
        for e in delegated["probe"]["events"]:
            if "status" in e:
                e["status"] = "PASS"
                e.pop("error", None)
        self.assertEqual(
            required.wire_verdict("delegated-range", delegated)["status"], "PASS"
        )
        for field, wrong in (
            ("method", "POST"),
            ("range", "bytes=1-2"),
            ("if_match", None),
            ("credential_headers", ["authorization"]),
            ("path", "/wrong/object"),
        ):
            corrupt = copy.deepcopy(delegated)
            next(w for w in corrupt["wire"] if w["role"] == "target-0")[field] = wrong
            self.assertEqual(
                required.wire_verdict("delegated-range", corrupt)["status"],
                "FAIL",
                field,
            )
        original = required.read(PACKET)
        tests = (
            "missing",
            "bad-hash",
            "bad-base",
            "wrong-range",
            "wrong-validator",
            "missing-result",
            "serialized-queue",
            "early-response",
            "timeout",
            "crash",
            "wrong-binary",
            "cold-presence",
            "missing-ais-row",
            "duplicate-ais-row",
            "list-inventory",
            "list-raw-etag",
            "list-output",
        )
        for mutation in tests:
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                packet = copy.deepcopy(original)
                for relative in packet["files"]:
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(PACKET.parent / relative, target)
                path = root / "packet.json"

                def update(relative, content):
                    target = root / relative
                    target.write_text(json.dumps(content, indent=2) + "\n")
                    packet["files"][relative] = required.sha(target)

                prefix = packet["wire_dir"] + "/direct-q64/"
                o = required.read(root / prefix / "observation.json")
                if mutation == "missing":
                    (root / prefix / "stdout.jsonl").unlink()
                elif mutation == "bad-hash":
                    (root / prefix / "stdout.jsonl").write_text("tampered\n")
                elif mutation == "bad-base":
                    packet["production_base"] = "wrong"
                elif mutation in (
                    "wrong-range",
                    "wrong-validator",
                    "missing-result",
                    "serialized-queue",
                    "early-response",
                    "timeout",
                    "crash",
                    "wrong-binary",
                ):
                    if mutation == "wrong-range":
                        next(w for w in o["wire"] if w["method"] == "GET")[
                            "range"
                        ] = "bytes=99-100"
                    elif mutation == "wrong-validator":
                        next(w for w in o["wire"] if w["method"] == "GET")[
                            "if_match"
                        ] = '"other-version"'
                    elif mutation == "missing-result":
                        o["probe"]["events"].pop(1)
                    elif mutation == "serialized-queue":
                        e = o["probe"]["events"]
                        e[0], e[1] = e[1], e[0]
                    elif mutation == "early-response":
                        o["gate"]["responses_before_release"] = 1
                    elif mutation == "timeout":
                        o["probe"]["timeout"] = True
                    elif mutation == "crash":
                        o["probe"]["exit"] = -11
                    else:
                        o["probe"]["binary_sha256"] = "0" * 64
                    update(prefix + "observation.json", o)
                    (root / prefix / "stdout.jsonl").write_text(
                        "".join(json.dumps(e) + "\n" for e in o["probe"]["events"])
                    )
                    packet["files"][prefix + "stdout.jsonl"] = required.sha(
                        root / prefix / "stdout.jsonl"
                    )
                elif mutation in (
                    "cold-presence",
                    "missing-ais-row",
                    "duplicate-ais-row",
                ):
                    relative = packet["reverse_dir"] + "/results.json"
                    result = required.read(root / relative)
                    if mutation == "cold-presence":
                        next(
                            r for r in result if r["case"] == "A2-cucascade-cold-fill"
                        )["presence_before"] = 200
                    elif mutation == "missing-ais-row":
                        result.pop(0)
                    else:
                        result.append(result[0])
                    update(relative, result)
                else:
                    relative = packet["root_dir"] + "/h-list/observation.json"
                    listed = required.read(root / relative)
                    if mutation == "list-inventory":
                        listed["expected"].pop()
                    elif mutation == "list-raw-etag":
                        listed["pages"][0]["entries"][0]["etag"] = "invented"
                    else:
                        listed["probe"]["events"][0]["entries"][0]["size"] += 1
                    update(relative, listed)
                path.write_text(json.dumps(packet))
                for script in ("required_features.py", "check_required_baseline.py"):
                    result = subprocess.run(
                        [sys.executable, "-B", str(scripts / script), str(path)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode,
                        1,
                        (script, mutation, result.stdout, result.stderr),
                    )
                    report = json.loads(result.stdout)
                    self.assertEqual(
                        report.get("status", report.get("baseline_check")), "FAIL"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
