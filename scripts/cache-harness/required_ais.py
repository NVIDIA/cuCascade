#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged AIS qualification plus actual library LIST; keep root/reverse separate."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--mode", choices=("root", "reverse"), required=True)
    args, rest = p.parse_known_args()
    run = args.run_dir.resolve()
    if run.exists():
        p.error("run-dir must be fresh")
    script = Path(__file__).with_name("ais_harness.py")
    common = ["--run-dir", str(run), "--mode", args.mode, *rest]
    commands = []

    def execute(command):
        result = subprocess.run(
            [sys.executable, "-B", str(script), command, *common],
            capture_output=True,
            text=True,
            timeout=400,
        )
        commands.append(
            dict(
                command=command,
                exit=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            raise RuntimeError(f"AIS {command} failed; inspect commands.json")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        execute("up")
        observe_list(run, args.probe.resolve())
        execute("qualify")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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


if __name__ == "__main__":
    main()
