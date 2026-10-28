#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict H qualification, separate from the permissive P0 characterization report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit, parse_qs
import xml.etree.ElementTree as ET
from required_ais_evidence import ais_controls, http_response

BASE = "1a9281093a0fe49634594d541d7ce327129f53d6"
WIRE_CASES = (
    "direct-head",
    "direct-footer",
    "direct-serial",
    "direct-q64",
    "direct-q256",
    "tls-direct",
    "list-fixture",
    "rewrite-head",
    "rewrite-footer",
    "rewrite-range",
    "nonallowlisted",
    "second-307",
    "userinfo",
    "fragment",
    "downgrade",
    "reuse-off",
    "retention-expanded",
    "retention-legacy",
    "stress-q64",
    "stress-q256",
)
REQUIRED = (
    "root-read",
    "root-full-cold",
    "root-footer",
    "root-footer-cold",
    "admission-nonallowlisted",
    "admission-second-307",
    "admission-userinfo",
    "admission-fragment",
    "admission-downgrade",
    "delegated-head",
    "delegated-footer",
    "delegated-range",
    "reuse-off",
    "retention-f0",
    "stress-q64",
    "stress-q256",
    "list-etag",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def row(name, status, reason, **evidence):
    return dict(id=name, status=status, reason=reason, **evidence)


def completed(o):
    p = o["probe"]
    events = p["events"]
    return (
        p["exit"] == 0
        and not p["timeout"]
        and not p["parse_errors"]
        and bool(events)
        and events[-1].get("event") == "done"
        and events[-1].get("status") == "PASS"
    )


def range_results(o, success):
    results = [e for e in o["probe"]["events"] if e["event"] == "result"]
    return len(results) == o["count"] and all(
        e.get("id") == i
        and e.get("offset") == i * 37
        and e.get("length") == 31
        and (
            e.get("status") == "PASS"
            if success
            else e.get("status") == "FAIL" and "HTTP 307" in e.get("error", "")
        )
        for i, e in enumerate(results)
    )


def origin_requests_valid(o):
    origin = [w for w in o["wire"] if w["role"] == "origin"]
    if o["mode"] == "list":
        return bool(origin) and all(w["method"] == "GET" for w in origin)
    if not all("authorization" in w["credential_headers"] for w in origin):
        return False
    if o["mode"] == "head":
        return (
            len(origin) == 1
            and origin[0]["method"] == "HEAD"
            and origin[0]["range"] is None
        )
    if o["mode"] == "footer":
        return (
            len(origin) == 1
            and origin[0]["method"] == "GET"
            and origin[0]["range"] == "bytes=-512"
        )
    head = [w for w in origin if w["method"] == "HEAD"]
    gets = [w for w in origin if w["method"] == "GET"]
    expected = {f"bytes={i * 37}-{i * 37 + 30}" for i in range(o["count"])}
    return (
        len(head) == 1
        and head[0]["response"] == 200
        and len(gets) == o["count"]
        and {w["range"] for w in gets} == expected
        and all(
            w["if_match"] is None or w["if_match"] == '"h-fixture-version-1"'
            for w in gets
        )
    )


def fixture_location_valid(o):
    endpoints = o["endpoints"]
    if (
        len(endpoints) != 5
        or o["allowed_targets"] != endpoints[1:4]
        or len(set(endpoints)) != 5
    ):
        return False
    for i, endpoint in enumerate(endpoints):
        url = urlsplit(endpoint)
        scheme = (
            "https" if i == 0 and o["name"] in ("tls-direct", "downgrade") else "http"
        )
        if (
            url.scheme != scheme
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
        ):
            return False
    name = o["name"]
    case = (
        name
        if name in ("nonallowlisted", "second-307", "userinfo", "fragment", "downgrade")
        else (
            "direct"
            if name.startswith("direct-") or name in ("list-fixture", "tls-direct")
            else "rewrite"
        )
    )
    if o["fixture_case"] != case or o["probe"]["command"][1] != endpoints[0]:
        return False
    redirects = [w for w in o["wire"] if w["role"] == "origin" and w["response"] == 307]
    allowed_roles = {"origin", "canary", "target-0", "target-1", "target-2"}
    if any(
        w["role"] not in allowed_roles or w["sequence"] != i
        for i, w in enumerate(o["wire"])
    ):
        return False
    for n, w in enumerate(redirects):
        target = endpoints[1 + (n % 3 if o["mode"] == "queue" else 0)]
        expected = target + "/s3/h/object?pid=fixture&utm=fixture"
        if case == "nonallowlisted":
            expected = endpoints[4] + "/h/object"
        elif case == "userinfo":
            expected = target.replace("://", "://fixture-user@") + "/h/object"
        elif case == "fragment":
            expected += "#fragment"
        if w["response_headers"].get("Location") != expected:
            return False
    return True


def delivered_307(o):
    p = o["probe"]
    failures = [e for e in p["events"] if e["event"] in ("error", "result")]
    # First-hop rejection means *every* failed operation stopped there. Even
    # one target/canary attempt invalidates this classification.
    if not fixture_location_valid(o) or any(w["role"] != "origin" for w in o["wire"]):
        return False
    reads = [
        w
        for w in o["wire"]
        if w["method"] == ("HEAD" if o["mode"] == "head" else "GET")
    ]
    expected_count = 1 if o["mode"] in ("head", "footer") else o["count"]
    return (
        p["exit"] == 1
        and not p["timeout"]
        and not p["parse_errors"]
        and len(reads) == len(failures) == expected_count
        and all(w["response"] == 307 for w in reads)
        and all("HTTP 307" in e.get("error", "") for e in failures)
        and (
            o["mode"] in ("head", "footer")
            and failures[0].get("stage") == "open"
            or o["mode"] in ("serial", "queue")
            and range_results(o, False)
        )
    )


def queue_valid(o):
    events = o["probe"]["events"]
    gates = [i for i, e in enumerate(events) if e["event"] == "submitted"]
    results = [i for i, e in enumerate(events) if e["event"] == "result"]
    gate = o.get("gate") or {}
    return (
        len(gates) == 1
        and len(results) == o["count"]
        and gates[0] < min(results)
        and events[gates[0]]["count"] == o["count"]
        and gate.get("submitted") == o["count"]
        and gate.get("responses_before_release") == 0
        and o["count"] in (64, 256)
    )


def wire_pair_valid(o):
    targets = [w for w in o["wire"] if w["role"].startswith("target-")]
    origins = [
        w for w in o["wire"] if w["role"] == "origin" and w.get("response") == 307
    ]
    if o["mode"] in ("serial", "queue") and any(
        w["if_match"] != '"h-fixture-version-1"' for w in origins
    ):
        return False
    if not targets or len(targets) != len(origins):
        return False
    # Serial cases preserve request order; concurrent cases match unique Range.
    for source in origins:
        pair = [
            w
            for w in targets
            if w["method"] == source["method"]
            and w["range"] == source["range"]
            and w["if_match"] == source["if_match"]
        ]
        if (
            len(pair) != 1
            or pair[0]["credential_headers"]
            or pair[0]["response"] not in (200, 206)
        ):
            return False
        target = pair[0]
        number = int(target["role"].split("-")[1])
        location = source["response_headers"].get("Location")
        if location != o["allowed_targets"][number] + target["path"] or not target[
            "path"
        ].startswith("/s3/h/object?pid=fixture&utm=fixture"):
            return False
    return True


def wire_verdict(name, o, *, needs_control=False):
    if not fixture_location_valid(o):
        return row(
            name, "FAIL", "fixture Location/authority/protocol does not match case"
        )
    if not origin_requests_valid(o):
        return row(
            name, "FAIL", "origin method/Range/If-Match/signing does not match workload"
        )
    if o["probe"]["timeout"]:
        return row(name, "TIMEOUT", "actual library probe exceeded its deadline")
    if o["probe"]["parse_errors"] or not o["cleanup"]:
        return row(name, "FAIL", "invalid probe output or incomplete cleanup")
    if delivered_307(o):
        if o["mode"] == "queue" and not queue_valid(o):
            return row(
                name,
                "FAIL",
                "307 observed but outstanding submission was not established",
            )
        return row(
            name,
            "UNSUPPORTED",
            "actual fixed library rejects first 307; delegated capability not reached",
            requests=len(o["wire"]),
        )
    ok = completed(o) and wire_pair_valid(o)
    if o["mode"] in ("serial", "queue"):
        ok = ok and range_results(o, True)
    if o["mode"] == "queue":
        ok = (
            ok
            and queue_valid(o)
            and len({w["role"] for w in o["wire"] if w["role"].startswith("target-")})
            == 3
        )
    if o["probe"]["exit"] != 0:
        return row(name, "FAIL", "failure is not a verified first-hop rejection")
    if needs_control and not o["policy_controls_available"]:
        return row(
            name,
            "NOT_TESTED",
            "required policy switch cannot be selected through this baseline API",
        )
    return row(
        name,
        "PASS" if ok else "FAIL",
        "actual method/Range/If-Match, credential absence, payload and gateway path checks",
    )


def xml_page(body):
    root = ET.fromstring(body)
    if root.tag.split("}")[-1] != "ListBucketResult":
        raise ValueError("not a LIST page")
    values = {}
    for node in root:
        key = node.tag.split("}")[-1]
        if key in ("IsTruncated", "NextContinuationToken"):
            if key in values:
                raise ValueError("duplicate LIST page metadata")
            values[key] = node.text or ""
    if values.get("IsTruncated") not in ("true", "false"):
        raise ValueError("missing/invalid LIST IsTruncated")
    truncated, token = values["IsTruncated"] == "true", values.get(
        "NextContinuationToken", ""
    )
    if bool(token) != truncated:
        raise ValueError("LIST truncation/token mismatch")
    return dict(truncated=truncated, next=token, entries=xml_entries(body))


def xml_entries(body):
    root = ET.fromstring(body)
    strip = lambda tag: tag.split("}")[-1]
    entries = []
    for node in root:
        if strip(node.tag) == "Contents":
            values = {strip(n.tag): n.text for n in node}
            entries.append(
                dict(
                    key=values["Key"],
                    size=int(values["Size"]),
                    etag_present="ETag" in values,
                    etag=(values.get("ETag") or "") if "ETag" in values else None,
                )
            )
    return entries


def list_valid(o, raw_pages, expected):
    pages = [e for e in o["probe"]["events"] if e["event"] == "page"]
    library = [e for p in pages for e in p["entries"]]
    raw = [dict(key=e["key"], size=e["size"]) for p in raw_pages for e in p]
    order = lambda entries: sorted(entries, key=lambda e: e["key"])
    return (
        completed(o)
        and bool(pages)
        and len(library) == len({e["key"] for e in library})
        and order(library) == order(raw) == order(expected)
        and all(
            p["index"] == i + 1
            and p["truncated"] == (i < len(pages) - 1)
            and bool(p["next"]) == p["truncated"]
            for i, p in enumerate(pages)
        )
    )


def assess_wire(observations):
    rows, controls = [], []
    for name in WIRE_CASES[:6]:
        o = observations[name]
        ok = (
            completed(o)
            and o["cleanup"]
            and origin_requests_valid(o)
            and fixture_location_valid(o)
        )
        if o["mode"] in ("serial", "queue"):
            ok = ok and range_results(o, True)
        if o["mode"] == "queue":
            ok = ok and queue_valid(o)
        controls.append(
            row(name, "PASS" if ok else "FAIL", "actual direct library control")
        )
    o = observations["list-fixture"]
    raw = [xml_entries(w["list_body"]) for w in o["wire"] if "list_body" in w]
    states = {e["key"]: e["etag"] for p in raw for e in p}
    valid = list_valid(
        o, raw, [{"key": k, "size": 3} for k in ("empty", "missing", "strong")]
    ) and states == {"empty": "", "missing": None, "strong": '"v1"'}
    controls.append(
        row(
            "list-fixture",
            "PASS" if valid else "FAIL",
            "empty/absent/nonempty ETag fixture, complete library pagination; API has no ETag field",
        )
    )
    for suffix in ("head", "footer", "range"):
        rows.append(
            wire_verdict("delegated-" + suffix, observations["rewrite-" + suffix])
        )
    positive = next(r for r in rows if r["id"] == "delegated-range")["status"] == "PASS"
    for case in ("nonallowlisted", "second-307", "userinfo", "fragment", "downgrade"):
        o = observations[case]
        base = wire_verdict("admission-" + case, o)
        if positive and o["policy_controls_available"]:
            # The old adapter cannot configure an allowlist. A future adapter must
            # prove intentional rejection in addition to the healthy positive path.
            forbidden = [
                w
                for w in o["wire"]
                if w["role"] == "canary"
                or (case != "second-307" and w["role"].startswith("target-"))
            ]
            rejected = (
                o["probe"]["exit"] == 1 and not o["probe"]["timeout"] and not forbidden
            )
            base = row(
                "admission-" + case,
                "PASS" if rejected else "FAIL",
                "policy rejection with legitimate delegation control",
            )
        elif base["status"] == "PASS":
            base = row(
                "admission-" + case,
                "NOT_TESTED",
                "no configured policy/positive admission control",
            )
        rows.append(base)
    rows.append(
        wire_verdict("reuse-off", observations["reuse-off"], needs_control=True)
    )
    a, b = (observations[n] for n in ("retention-expanded", "retention-legacy"))
    ar, br = (wire_verdict("retention-f0", o, needs_control=True) for o in (a, b))
    if ar["status"] == br["status"] == "PASS":
        connections = lambda o: len(
            {w["connection"] for w in o["wire"] if w["method"] == "GET"}
        )
        ok = (
            a["worker_count"]
            == b["worker_count"]
            == a["max_connections"]
            == b["max_connections"]
            == 1
            and a["count"] == b["count"] == 32
            and connections(a) <= 2
            and connections(b) >= 16
        )
        rows.append(
            row(
                "retention-f0",
                "PASS" if ok else "FAIL",
                "32 serial W1/C1; observed GET connections exclude separately counted HEAD warmup",
                expanded=connections(a),
                legacy=connections(b),
            )
        )
    else:
        rows.append(
            row(
                "retention-f0",
                ar["status"] if ar["status"] != "PASS" else br["status"],
                "expanded/legacy not selectable; both actual 32-read attempts retained; no retention proof",
                expanded=ar,
                legacy=br,
            )
        )
    for case in ("stress-q64", "stress-q256"):
        rows.append(wire_verdict(case, observations[case]))
    return rows, controls


def checked_file(root, path, manifest):
    relative = Path(path)
    full = (root / relative).resolve()
    if (
        relative.is_absolute()
        or not full.is_relative_to(root.resolve())
        or path not in manifest
        or not full.is_file()
        or sha(full) != manifest[path]
    ):
        raise ValueError("missing, outside, or changed evidence: " + path)
    return full


def evaluate(packet_path):
    root = packet_path.resolve().parent
    packet = read(packet_path)
    if packet.get("schema") != 1 or packet.get("production_base") != BASE:
        raise ValueError("unexpected profile schema/base")
    manifest = packet["files"]
    for path in manifest:
        checked_file(root, path, manifest)
    load = lambda path: read(checked_file(root, path, manifest))
    binding = load(packet["build_receipt"])
    if (
        binding["binary_sha256"] != packet["probe_sha256"]
        or binding["production_base"] != BASE
        or not binding["production_files"]
        or not binding["libraries"]
        or binding["compile_exit"] != 0
        or binding["link_exit"] != 0
    ):
        raise ValueError("build binding incomplete")
    wire = {}
    for name in WIRE_CASES:
        prefix = packet["wire_dir"] + "/" + name + "/"
        o = load(prefix + "observation.json")
        stdout = checked_file(root, prefix + "stdout.jsonl", manifest).read_text()
        if (
            o["name"] != name
            or o["probe"]["events"]
            != [json.loads(line) for line in stdout.splitlines()]
            or o["probe"]["binary_sha256"] != packet["probe_sha256"]
        ):
            raise ValueError("probe source/output binding mismatch: " + name)
        expected_mode = (
            "queue"
            if "q64" in name or "q256" in name
            else (
                "footer"
                if name.endswith("footer")
                else (
                    "head"
                    if name.endswith("head")
                    else "list" if name == "list-fixture" else "serial"
                )
            )
        )
        expected_count = (
            256
            if "q256" in name
            else (
                64
                if "q64" in name
                else (
                    32
                    if name
                    in ("direct-serial", "retention-expanded", "retention-legacy")
                    else 2 if name in ("rewrite-range", "reuse-off") else 1
                )
            )
        )
        if (
            o["mode"] != expected_mode
            or o["count"] != expected_count
            or len(o["allowed_targets"]) != 3
            or len(set(o["allowed_targets"])) != 3
        ):
            raise ValueError("workload configuration mismatch: " + name)
        wire[name] = o
    rows, controls = assess_wire(wire)
    for mode in ("root", "reverse"):
        prefix = packet[mode + "_dir"] + "/"
        results = load(prefix + "results.json")
        by_name = {r["case"]: r for r in results}
        expected_cases = {
            "A1-native-cold",
            "A3-native-presence-and-warm",
            "A3-origin-get-count",
            "A2-s3-reference",
            "A2-cucascade",
            "A2-cucascade-cold-fill",
            "A2-footer-first",
            "A2-footer-first-cold-fill",
            "F-s3-missing",
            "F-s3-invalid-range",
            "F-s3-stale-if-match",
            "F-s3-valid-range",
            "F-ais-client-auth",
            "A4-freshness",
            "F-origin-offline-warm",
            "F-target-stopped",
        }
        if set(by_name) != expected_cases:
            raise ValueError("AIS original row set incomplete")
        if len(by_name) != len(results):
            raise ValueError("duplicate AIS case")
        state = load(prefix + "ais-state.json")
        if (
            state["mode"] != mode
            or not state.get("stopped")
            or state.get("cleanup_errors")
        ):
            raise ValueError("AIS mode or cleanup mismatch")
        raw_statuses, raw_count, raw_response = ais_controls(
            load, prefix, by_name, state
        )
        for label, source, name in (
            ("cucascade-ais", "A2-cucascade", "read"),
            ("cucascade-footer-first", "A2-footer-first", "footer"),
        ):
            data = by_name[source]
            payload = json.loads(
                checked_file(root, prefix + label + ".stdout", manifest).read_text()
            )
            error = checked_file(root, prefix + label + ".stderr", manifest).read_text()
            if data["probe"]["result"] != payload:
                raise ValueError("AIS probe output mismatch")
            success = data["probe"]["exit"] == 0 and payload.get("status") == "PASS"
            unsupported = (
                mode == "root" and data["probe"]["exit"] == 1 and "HTTP 307" in error
            )
            status = "PASS" if success else "UNSUPPORTED" if unsupported else "FAIL"
            cold = by_name[source + "-cold-fill"]
            before = raw_response(label + "-presence-before")["status"]
            after = raw_response(label + "-presence-after")["status"]
            if before != cold["presence_before"] or after != cold["presence_after"]:
                raise ValueError("cold presence binding mismatch")
            counts = raw_count(label + "-stats-before"), raw_count(
                label + "-stats-after"
            )
            if counts != (cold["origin_get_before"], cold["origin_get_after"]):
                raise ValueError("library cold-fill raw counter/summary mismatch")
            cold_ok = (
                success
                and before == 404
                and after == 200
                and all(type(c) is int for c in counts)
                and counts[1] > counts[0]
            )
            target = rows if mode == "root" else controls
            target.append(
                row(
                    mode + "-" + name,
                    status,
                    "actual cuCascade smoke on independent untouched object",
                )
            )
            target.append(
                row(
                    mode + ("-full-cold" if name == "read" else "-footer-cold"),
                    (
                        "PASS"
                        if cold_ok
                        else (
                            "NOT_TESTED"
                            if unsupported and before == after == 404
                            else "FAIL"
                        )
                    ),
                    "successful library bytes + presence404→200 + backend GET increment required",
                    presence=[before, after],
                    origin_get=list(counts),
                )
            )
        # Preserve every existing characterization row, including explicit limits.
        for source, r in by_name.items():
            if source.startswith("A2-cucascade") or source.startswith(
                "A2-footer-first"
            ):
                continue
            if raw_statuses[source] == "PASS":
                controls.append(
                    row(
                        mode + ":" + source,
                        "PASS",
                        "rejudged from required raw HTTP/hash/presence/counter receipts",
                    )
                )
            elif not (
                source == "F-ais-client-auth" and r["status"] == "NOT_TESTED"
            ) and not (
                source == "F-s3-stale-if-match" and r["status"] == "UNSUPPORTED"
            ):
                controls.append(
                    row(
                        mode + ":" + source,
                        "FAIL",
                        "original characterization control regressed",
                    )
                )
    prefix = packet["root_dir"] + "/h-list/"
    listed = load(prefix + "observation.json")
    list_state = load(packet["root_dir"] + "/ais-state.json")
    raw_pages = []
    previous, seen = None, set()
    library_pages = [e for e in listed["probe"]["events"] if e["event"] == "page"]
    if not listed["pages"] or len(library_pages) != len(listed["pages"]):
        raise ValueError("LIST page count mismatch")
    library_tokens = set()
    for index, page in enumerate(listed["pages"]):
        body = checked_file(root, prefix + page["body"], manifest).read_bytes()
        terminal = http_response(page["response"], list_state)
        if terminal["sha256"] != hashlib.sha256(body).hexdigest() or terminal[
            "length"
        ] != len(body):
            raise ValueError("LIST response body binding mismatch")
        raw = xml_page(body)
        entries = raw["entries"]
        lib = library_pages[index]
        query = parse_qs(urlsplit(page["response"]["chain"][0]["url"]).query)
        if (
            page["truncated"] != raw["truncated"]
            or page["token_in"] != previous
            or query.get("continuation-token", [None]) != [previous]
            or raw["truncated"] != (index < len(listed["pages"]) - 1)
            or lib["truncated"] != raw["truncated"]
            or lib["entries"] != [dict(key=e["key"], size=e["size"]) for e in entries]
            or bool(lib["next"]) != lib["truncated"]
            or (lib["next"] and lib["next"] in library_tokens)
            or (raw["next"] and raw["next"] in seen)
        ):
            raise ValueError(
                "LIST raw page/token chain disagrees with observation/library"
            )
        # Tokens belong to independent AIS listing sessions; never compare their
        # opaque byte values across the reference and library calls.
        if raw["next"]:
            seen.add(raw["next"])
        if lib["next"]:
            library_tokens.add(lib["next"])
        previous = raw["next"] or None
        if (
            sha(root / prefix / page["body"]) != page["sha256"]
            or entries != page["entries"]
            or page["response"]["status"] != 200
        ):
            raise ValueError("LIST raw response mismatch")
        raw_pages.append(entries)
    list_stdout = [
        json.loads(line)
        for line in checked_file(root, prefix + "stdout.jsonl", manifest)
        .read_text()
        .splitlines()
    ]
    if (
        list_stdout != listed["probe"]["events"]
        or listed["probe"]["binary_sha256"] != packet["probe_sha256"]
    ):
        raise ValueError("LIST probe binding mismatch")
    backing = load(prefix + "backing-list.json")
    expected_inventory = [dict(key=e["key"], size=e["size"]) for e in backing]
    if listed["expected"] != expected_inventory:
        raise ValueError("LIST expected inventory differs from backing LIST receipt")
    ok = (
        list_valid(listed, raw_pages, expected_inventory)
        and not listed["pages"][-1]["truncated"]
    )
    observed = [e for p in raw_pages for e in p]
    rows.append(
        row(
            "list-etag",
            "PASS" if ok else "FAIL",
            "library key/size pagination and separately captured service ETag; empty is valid observation",
            objects=len(observed),
            empty=sum(e["etag"] == "" for e in observed),
            absent=sum(not e["etag_present"] for e in observed),
            api_exposes_etag=False,
            nonempty_validator_capability=False,
        )
    )
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(REQUIRED):
        raise ValueError("missing/duplicate/unknown mandatory row")
    rows.sort(key=lambda r: REQUIRED.index(r["id"]))
    return dict(
        schema=1,
        status=qualification_status(rows, controls),
        production_base=BASE,
        rows=rows,
        controls=controls,
        packet_sha256=sha(packet_path),
    )


def qualification_status(rows, controls):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(REQUIRED) or not controls:
        return "FAIL"
    return "PASS" if all(r["status"] == "PASS" for r in rows + controls) else "FAIL"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        result = evaluate(args.packet)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        ET.ParseError,
    ) as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}")
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            p.error("output already exists; keep prior evidence")
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
