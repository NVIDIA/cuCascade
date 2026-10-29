#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DX-9 capture tests and mutations of a valid schema-2 H packet via both CLIs."""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import ais_harness as ais
from required_ais_evidence import bound_payload, decode_json_body

ARGS = argparse.ArgumentParser(description=__doc__)
ARGS.add_argument("packet", nargs="?", type=Path)
ARGS.add_argument("--results", type=Path)
OPTIONS = ARGS.parse_args()
sys.argv = sys.argv[:1]
SCRIPTS = Path(__file__).parent
RESULTS = []


class BindingTests(unittest.TestCase):
    def test_capture_actual_http_bytes(self):
        bodies = {
            "/v1/daemon?what=node_stats": b'{ "tracker" : {} }\n',
            "/v1/daemon?what=metrics": b'{ "aws.get.n" : "counter" }\n',
        }

        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                body = bodies[self.path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server:
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    state = dict(
                        schema=1,
                        run_dir=str(root),
                        endpoint=f"http://127.0.0.1:{server.server_port}",
                        target_endpoint=f"http://127.0.0.1:{server.server_port}",
                        ports=[server.server_port],
                    )
                    (root / "ais-state.json").write_text(json.dumps(state))
                    self.assertEqual(ais.stats(root, "stats-before"), 0)
                    load = lambda name: json.loads((root / name).read_bytes())
                    raw = lambda name: (root / name).read_bytes()
                    for label, path in (
                        ("stats-before", "/v1/daemon?what=node_stats"),
                        ("stats-before-names", "/v1/daemon?what=metrics"),
                    ):
                        self.assertEqual(raw(label + ".body"), bodies[path])
                        self.assertEqual(
                            bound_payload(load, raw, "", label, state, path),
                            json.loads(bodies[path]),
                        )
                    bodies["/v1/daemon?what=node_stats"] = (
                        b'{"tracker":{"aws.get.n":{"value":3}}}\n'
                    )
                    self.assertEqual(ais.stats(root, "stats-cold"), 3)
                    self.assertFalse((root / "stats-cold-names.body").exists())
            finally:
                server.shutdown()
                worker.join()

    def test_ambiguous_json_rejected(self):
        for body in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"{"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                decode_json_body(body)

    @unittest.skipUnless(OPTIONS.packet, "pass a real schema-2 packet for CLI checks")
    def test_packet_binding(self):
        source = OPTIONS.packet.resolve()
        original = json.loads(source.read_bytes())
        self.assertEqual(original["schema"], 2)
        cases = [("normal", None, None)]
        variants = (
            "bad_hash",
            "missing_hash",
            "bad_length",
            "missing_length",
            "bool_length",
            "raw_missing",
            "raw_unlisted",
            "raw_truncated",
            "payload_mismatch",
            "payload_missing",
            "raw_mismatch_bound",
            "invalid_json_bound",
            "duplicate_json_bound",
            "nonfinite_json_bound",
            "wrong_endpoint",
            "integrity",
            "equivalent_json",
        )
        for mode in ("root", "reverse"):
            for label in (
                "stats-cold",
                "stats-before-names",
                "cucascade-ais-stats-before",
            ):
                for case in variants:
                    cases.append((case, mode, label))
            for case in (
                "before_metadata",
                "before_body",
                "cross_mode_record",
                "payload_and_summary",
            ):
                cases.append((case, mode, "stats-cold"))
            cases.append(("registry_unregistered_bound", mode, "stats-before-names"))
            cases.append(("bool_counter_bound", mode, "stats-cold"))
        cases.append(("legacy_schema", None, None))

        for case, mode, label in cases:
            with self.subTest(
                case=case, mode=mode, label=label
            ), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                packet = copy.deepcopy(original)
                for name in packet["files"]:
                    dest = root / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source.parent / name, dest)

                def read(name):
                    return json.loads((root / name).read_bytes())

                def update(name, value, raw=False, rehash=True):
                    (root / name).write_bytes(
                        value if raw else (json.dumps(value) + "\n").encode()
                    )
                    if rehash:
                        packet["files"][name] = hashlib.sha256(
                            (root / name).read_bytes()
                        ).hexdigest()

                if mode:
                    prefix = packet[mode + "_dir"] + "/"
                    recname = prefix + label + ".json"
                    rawname = prefix + label + ".body"
                    payname = prefix + label + "-payload.json"
                    record = read(recname)
                    terminal = record["chain"][-1]
                    payload = read(payname)
                    if case in ("bad_hash", "integrity"):
                        terminal["sha256"] = "0" * 64
                    elif case in ("missing_hash", "missing_length"):
                        del terminal["sha256" if case == "missing_hash" else "length"]
                    elif case in ("bad_length", "bool_length"):
                        terminal["length"] = (
                            True if case == "bool_length" else terminal["length"] + 1
                        )
                    elif case in ("raw_missing", "raw_unlisted", "payload_missing"):
                        name = payname if case == "payload_missing" else rawname
                        if case != "raw_unlisted":
                            (root / name).unlink()
                        del packet["files"][name]
                    elif case == "raw_truncated":
                        update(rawname, (root / rawname).read_bytes()[:-1], raw=True)
                    elif case in ("payload_mismatch", "payload_and_summary"):
                        if "tracker" in payload:
                            payload["tracker"]["aws.get.n"] = {"value": 98765}
                        else:
                            payload["aws.get.n"] = "gauge"
                        update(payname, payload)
                        if case == "payload_and_summary":
                            summaries = read(prefix + "results.json")
                            next(
                                x
                                for x in summaries
                                if x["case"] == "A3-origin-get-count"
                            )["after_cold"] = 98765
                            update(prefix + "results.json", summaries)
                    elif case == "wrong_endpoint":
                        terminal["url"] += "&wrong=request"
                    elif case == "before_metadata":
                        before = read(prefix + "stats-before.json")["chain"][-1]
                        terminal.update(
                            length=before["length"], sha256=before["sha256"]
                        )
                    elif case == "before_body":
                        update(
                            rawname,
                            (root / prefix / "stats-before.body").read_bytes(),
                            raw=True,
                        )
                    elif case == "cross_mode_record":
                        other = packet[
                            ("reverse" if mode == "root" else "root") + "_dir"
                        ]
                        record = read(other + "/" + label + ".json")
                    else:
                        bodies = {
                            "raw_mismatch_bound": b'{"different":1}',
                            "invalid_json_bound": b"{",
                            "duplicate_json_bound": b'{"tracker":{},"tracker":{}}',
                            "nonfinite_json_bound": b'{"tracker":{"aws.get.n":NaN}}',
                        }
                        if case in (
                            "registry_unregistered_bound",
                            "bool_counter_bound",
                        ):
                            if case == "registry_unregistered_bound":
                                payload["aws.get.n"] = "gauge"
                            else:
                                payload["tracker"]["aws.get.n"] = {"value": True}
                            update(payname, payload)
                        body = bodies.get(
                            case,
                            json.dumps(payload, sort_keys=True, indent=3).encode()
                            + b"\n",
                        )
                        update(rawname, body, raw=True)
                        terminal.update(
                            length=len(body), sha256=hashlib.sha256(body).hexdigest()
                        )
                    update(recname, record, rehash=case != "integrity")
                if case == "legacy_schema":
                    packet["schema"] = 1
                path = root / "packet.json"
                path.write_text(json.dumps(packet))
                accepted = case in ("normal", "equivalent_json")
                entry = dict(
                    case=case, mode=mode, label=label, accepted=accepted, commands={}
                )
                for script in ("required_features.py", "check_required_baseline.py"):
                    run = subprocess.run(
                        [sys.executable, "-B", str(SCRIPTS / script), str(path)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    result = json.loads(run.stdout)
                    entry["commands"][script] = dict(exit=run.returncode, result=result)
                    self.assertEqual(
                        run.returncode,
                        1 if script == "required_features.py" or not accepted else 0,
                        run.stderr,
                    )
                    if accepted:
                        self.assertNotIn("error", result)
                        if script == "required_features.py":
                            self.assertEqual(len(result["controls"]), 31)
                            self.assertTrue(
                                all(x["status"] == "PASS" for x in result["controls"])
                            )
                    else:
                        self.assertIn("error", result)
                        self.assertRegex(
                            result["error"],
                            r"body|evidence|JSON|schema|counter|registry|endpoint|length|hash",
                        )
                RESULTS.append(entry)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if OPTIONS.results:
        OPTIONS.results.write_text(
            json.dumps(dict(success=result.wasSuccessful(), cases=RESULTS), indent=2)
            + "\n"
        )
    sys.exit(0 if result.wasSuccessful() else 1)
