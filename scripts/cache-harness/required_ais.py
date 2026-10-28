#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged AIS qualification plus actual library LIST; keep root/reverse separate."""
import argparse
import json
from pathlib import Path
import signal
import sys
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import ais_harness as ais
import harness
from required_fixture import digest, invoke, save


def observe_list(run, binary):
    state = ais.state_for(run)
    backing = harness.state_for(run / "origin")
    out = run / "h-list"
    out.mkdir()
    registered, _ = ais.snapshot(
        run,
        "h-list-bucket-register",
        "HEAD",
        f"/v1/buckets/{backing['bucket']}?provider=aws",
    )
    if registered["status"] != 200:
        raise RuntimeError("LIST backing bucket registration failed")
    probe = invoke(
        binary, state["endpoint"], backing["bucket"], "-", "-", "list", 1, out
    )
    pages, token, seen = [], None, set()
    for _ in range(64):
        query = {"list-type": "2", "max-keys": "2"}
        if token:
            query["continuation-token"] = token
        response, body = ais.request(
            state, "GET", "/" + backing["bucket"] + "?" + urlencode(query)
        )
        name = f"page-{len(pages) + 1}.xml"
        (out / name).write_bytes(body)
        page = dict(
            response=response, body=name, sha256=digest(out / name), token_in=token
        )
        pages.append(page)
        if response["status"] != 200:
            break
        xml = ET.fromstring(body)
        strip = lambda tag: tag.split("}")[-1]
        children = {strip(n.tag): n.text for n in xml}
        page["entries"] = []
        for node in xml:
            if strip(node.tag) == "Contents":
                values = {strip(n.tag): n.text for n in node}
                page["entries"].append(
                    dict(
                        key=values["Key"],
                        size=int(values["Size"]),
                        etag_present="ETag" in values,
                        etag=(values.get("ETag") or "") if "ETag" in values else None,
                    )
                )
        page["truncated"] = children.get("IsTruncated") == "true"
        token = children.get("NextContinuationToken")
        if not page["truncated"]:
            break
        if not token or token in seen:
            raise RuntimeError("LIST token missing or repeated")
        seen.add(token)
    else:
        raise RuntimeError("LIST exceeded 64 page bound")
    # Use the actual backing inventory. The existing fixture deliberately contains
    # duplicate-slash keys (a known service limitation) plus a readiness object.
    client = harness.client(backing)
    try:
        inventory = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=backing["bucket"]
        ):
            inventory.extend(
                dict(key=e["Key"], size=e["Size"], etag=e.get("ETag"))
                for e in page.get("Contents", [])
            )
    finally:
        client.close()
    save(out / "backing-list.json", inventory)
    expected = [{"key": e["key"], "size": e["size"]} for e in inventory]
    observation = dict(
        schema=1,
        kind="ais-list",
        mode=state["mode"],
        probe=probe,
        pages=pages,
        expected=expected,
        api_exposes_etag=False,
        validator_capability=False,
        note="Raw service ETag is a separate observation; baseline list_entry only exposes key/size.",
    )
    save(out / "observation.json", observation)


class RunInterrupted(BaseException):
    """Catchable whole-run termination; never SIGKILL the ownership registrar."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--mode", choices=("root", "reverse"), required=True)
    for name in ("ais-source", "aisnode", "weed", "binary"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--aisnode-sha256", required=True)
    p.add_argument("--weed-sha256", default=harness.WEED_SHA)
    p.add_argument("--port-base", type=int, default=19330)
    p.add_argument("--ais-port-base", type=int, default=19430)
    p.add_argument("--ready-timeout", type=float, default=75)
    p.add_argument("--deadline", type=int, default=300)
    args = p.parse_args()
    if args.deadline <= 0 or args.ready_timeout <= 0:
        p.error("deadlines must be positive")
    run = args.run_dir = args.run_dir.resolve()
    if run.exists():
        p.error("run-dir must be fresh")
    commands = []

    def execute(command, function):
        entry = dict(command=command, execution="in-process", exit=1)
        commands.append(entry)
        try:
            result = function(args)
            if (
                command == "qualify"
                and harness.load(run / "summary.json")["status"] == "FAIL"
            ):
                raise RuntimeError("AIS qualification control failed")
            entry["exit"] = 0
            return result
        except BaseException as exc:
            entry["error"] = type(exc).__name__ + ": " + str(exc)
            raise

    def interrupted(signum, frame):
        # Original launch_owned/defer_signals defers this handler until the
        # service handle is journaled. No subprocess.run exception can SIGKILL
        # the registrar in the middle of its ownership critical section.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        raise RunInterrupted(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(args.deadline)
    try:
        execute("up", ais.up)
        observe_list(run, args.probe.resolve())
        execute("qualify", ais.qualify)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        # Same original identity-checked TERM/KILL cleanup, with its own bounds.
        # Even a failed up records the error and attempts idempotent cleanup.
        if (run / "ais-state.json").exists():
            try:
                ais.down(run)
            finally:
                save(run / "h-commands.json", commands)
    save(
        run / "h-manifest.json",
        {
            str(f.relative_to(run)): digest(f)
            for f in sorted(run.rglob("*"))
            if f.is_file() and f.name != "h-manifest.json"
        },
    )
    print(json.dumps(dict(mode=args.mode, status="OBSERVED", run=str(run))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
