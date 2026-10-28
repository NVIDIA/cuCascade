#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay independent H review counterexamples and adjacent semantic failures."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

PACKET = Path(sys.argv.pop(1)).resolve()
SCRIPTS = Path(__file__).parent


class ReviewReceipts(unittest.TestCase):
    def test_review_counterexamples(self):
        original = json.loads(PACKET.read_text())
        cases = [
            "canary_reached",
            "wrong_negative_location",
            "offline_500",
            "cold_no_increment",
            "missing_control_receipt",
            "list_truncation_disagrees",
            "target_reached",
            "one_operation_not_307",
            "userinfo_not_sent",
            "fragment_not_sent",
            "downgrade_not_sent",
            "wrong_fixture_authority",
            "native_bad_hash",
            "missing_counter",
            "unregistered_zero",
            "missing_stats_payload",
            "cold_both_summaries_false",
            "freshness_wrong_bytes",
            "target_stopped_success",
            "list_next_missing",
            "list_next_repeated",
            "list_token_input_wrong",
            "list_wire_query_wrong",
            "list_library_token_repeated",
            "list_library_boundary_wrong",
            "list_missing_page",
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                packet = copy.deepcopy(original)
                for name in packet["files"]:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(PACKET.parent / name, target)

                def read(name):
                    return json.loads((root / name).read_text())

                def update(name, data, raw=False):
                    (root / name).write_text(data if raw else json.dumps(data))
                    packet["files"][name] = hashlib.sha256(
                        (root / name).read_bytes()
                    ).hexdigest()

                if case in (
                    "canary_reached",
                    "wrong_negative_location",
                    "target_reached",
                    "one_operation_not_307",
                    "userinfo_not_sent",
                    "fragment_not_sent",
                    "downgrade_not_sent",
                    "wrong_fixture_authority",
                ):
                    chosen = {
                        "userinfo_not_sent": "userinfo",
                        "fragment_not_sent": "fragment",
                        "downgrade_not_sent": "downgrade",
                        "one_operation_not_307": "stress-q64",
                    }.get(case, "nonallowlisted")
                    name = packet["wire_dir"] + "/" + chosen + "/observation.json"
                    o = read(name)
                    w = next(w for w in o["wire"] if w["method"] == "GET")
                    if case in ("canary_reached", "target_reached"):
                        extra = dict(
                            w,
                            role="canary" if case == "canary_reached" else "target-0",
                            connection=999,
                            sequence=len(o["wire"]),
                            response=500,
                            response_headers={},
                        )
                        o["wire"].append(extra)
                    elif case == "one_operation_not_307":
                        w["response"] = 500
                    elif case == "wrong_fixture_authority":
                        o["allowed_targets"][0] = o["endpoints"][4]
                    elif case == "downgrade_not_sent":
                        w["response_headers"]["Location"] = w["response_headers"][
                            "Location"
                        ].replace("http://", "https://")
                    else:
                        w["response_headers"]["Location"] = (
                            o["allowed_targets"][0]
                            + "/s3/h/object?pid=fixture&utm=fixture"
                        )
                    update(name, o)
                elif case.startswith("list_"):
                    prefix = packet["root_dir"] + "/h-list/"
                    name = prefix + "observation.json"
                    o = read(name)
                    if case in (
                        "list_truncation_disagrees",
                        "list_next_missing",
                        "list_next_repeated",
                    ):
                        page = o["pages"][
                            -1 if case == "list_truncation_disagrees" else 1
                        ]
                        xml = ET.fromstring((root / prefix / page["body"]).read_bytes())
                        token = next(
                            (n for n in xml if n.tag.endswith("NextContinuationToken")),
                            None,
                        )
                        if case == "list_truncation_disagrees":
                            next(
                                n for n in xml if n.tag.endswith("IsTruncated")
                            ).text = "true"
                        elif case == "list_next_missing":
                            xml.remove(token)
                        else:
                            token.text = page["token_in"]
                        body = ET.tostring(xml, encoding="unicode")
                        update(prefix + page["body"], body, raw=True)
                        page["sha256"] = hashlib.sha256(body.encode()).hexdigest()
                        page["response"]["chain"][-1].update(
                            sha256=page["sha256"], length=len(body.encode())
                        )
                    elif case == "list_token_input_wrong":
                        o["pages"][1]["token_in"] = "wrong"
                    elif case == "list_wire_query_wrong":
                        o["pages"][1]["response"]["chain"][0][
                            "url"
                        ] += "&continuation-token=wrong"
                    elif case == "list_missing_page":
                        o["pages"].pop()
                    else:
                        pages = [
                            e for e in o["probe"]["events"] if e["event"] == "page"
                        ]
                        if case == "list_library_token_repeated":
                            pages[1]["next"] = pages[0]["next"]
                        else:
                            pages[0]["truncated"] = False
                            pages[0]["next"] = ""
                        update(
                            prefix + "stdout.jsonl",
                            "".join(json.dumps(e) + "\n" for e in o["probe"]["events"]),
                            raw=True,
                        )
                    update(name, o)
                else:
                    prefix = packet["reverse_dir"] + "/"
                    if case in (
                        "offline_500",
                        "native_bad_hash",
                        "freshness_wrong_bytes",
                        "target_stopped_success",
                    ):
                        name = (
                            prefix
                            + {
                                "offline_500": "origin-offline-warm.json",
                                "native_bad_hash": "native-cold.json",
                                "freshness_wrong_bytes": "fresh-latest.json",
                                "target_stopped_success": "target-stopped.json",
                            }[case]
                        )
                        o = read(name)
                        if case in ("offline_500", "target_stopped_success"):
                            o["status"] = o["chain"][-1]["status"] = (
                                500 if case == "offline_500" else 200
                            )
                        else:
                            o["chain"][-1]["sha256"] = "0" * 64
                        update(name, o)
                    elif case in ("missing_control_receipt", "missing_stats_payload"):
                        name = prefix + (
                            "origin-offline-warm.json"
                            if case == "missing_control_receipt"
                            else "cucascade-ais-stats-after-payload.json"
                        )
                        (root / name).unlink()
                        del packet["files"][name]
                    elif case in ("missing_counter", "unregistered_zero"):
                        name = prefix + (
                            "cucascade-ais-stats-after-payload.json"
                            if case == "missing_counter"
                            else "stats-before-names-payload.json"
                        )
                        o = read(name)
                        (o["tracker"] if case == "missing_counter" else o).pop(
                            "aws.get.n"
                        )
                        update(name, o)
                    else:
                        update(
                            prefix + "cucascade-ais-stats-after-payload.json",
                            read(prefix + "cucascade-ais-stats-before-payload.json"),
                        )
                        if case == "cold_both_summaries_false":
                            o = read(prefix + "results.json")
                            next(r for r in o if r["case"] == "A2-cucascade-cold-fill")[
                                "origin_get_after"
                            ] = 1
                            update(prefix + "results.json", o)
                path = root / "packet.json"
                path.write_text(json.dumps(packet))
                for script in ("required_features.py", "check_required_baseline.py"):
                    result = subprocess.run(
                        [sys.executable, "-B", str(SCRIPTS / script), str(path)],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    self.assertEqual(
                        result.returncode,
                        1,
                        (case, script, result.stdout, result.stderr),
                    )
                    value = json.loads(result.stdout)
                    self.assertEqual(
                        value.get("status", value.get("baseline_check")), "FAIL"
                    )
                print(case + ": both CLIs rejected", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
