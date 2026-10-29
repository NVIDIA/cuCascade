#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a fixed-input H packet index. Existing runs and outputs are immutable."""
import argparse
import json
from pathlib import Path
from required_features import BASE, sha, WIRE_CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--wire", required=True)
    p.add_argument("--ais-root", required=True)
    p.add_argument("--ais-reverse", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--packet-name", default="packet.json")
    args = p.parse_args()
    root = args.root.resolve()
    if Path(args.packet_name).name != args.packet_name:
        p.error("packet-name must be a filename")
    output = root / args.packet_name
    if output.exists():
        p.error("packet already exists")
    files = {}
    # Include only review evidence, not server storage, secret configuration,
    # test binaries, or compiler objects. Raw stdout and protocol responses stay.
    selected = [root / args.build / "build.json"]
    for name in WIRE_CASES:
        selected += list((root / args.wire / name).glob("*"))
    for mode in (args.ais_root, args.ais_reverse):
        directory = root / mode
        selected += [
            f
            for f in directory.iterdir()
            if f.is_file()
            and f.suffix in (".json", ".stdout", ".stderr", ".body")
            and f.name != "h-manifest.json"
        ]
        selected += list((directory / "h-list").glob("*"))
        selected += [
            directory / "origin/fixtures.json",
            directory / "origin/state.json",
        ]
    for file in selected:
        if not file.resolve().is_relative_to(root):
            raise ValueError("evidence outside root")
        files[str(file.relative_to(root))] = sha(file)
    binding = json.loads((root / args.build / "build.json").read_text())
    packet = dict(
        schema=2,
        production_base=BASE,
        wire_dir=args.wire,
        root_dir=args.ais_root,
        reverse_dir=args.ais_reverse,
        build_receipt=args.build + "/build.json",
        probe_sha256=binding["binary_sha256"],
        files=files,
    )
    output.write_text(json.dumps(packet, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
