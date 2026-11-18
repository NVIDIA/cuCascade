#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only per-RDMA-port counter snapshots and deltas; never attribution proof."""

import argparse
import json
from pathlib import Path


def snapshot(root):
    counters, unavailable = {}, []
    for port in sorted((root / "class/infiniband").glob("*/ports/*")):
        for name in ("port_rcv_data", "port_xmit_data"):
            path = port / "counters" / name
            key = f"{port.parent.parent.name}/{port.name}/{name}"
            try:
                counters[key] = int(path.read_text().strip())
            except (OSError, ValueError) as error:
                unavailable.append(f"{key}: {error}")
    if not counters:
        unavailable.append("no readable RDMA port counters")
    return {"counters": counters, "unavailable": unavailable}


def delta(before, after):
    values = {}
    for key, value in after["counters"].items():
        old = before["counters"].get(key)
        values[key] = None if old is None or value < old else value - old
    return {
        "delta_4byte_units": values,
        "delta_bytes": {k: None if v is None else v * 4 for k, v in values.items()},
        "unavailable": before["unavailable"] + after["unavailable"],
        "note": "Host-global port traffic, not proof of cuFile attribution. Null means new/reset counter.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument(
        "--before", type=Path, help="emit delta against this earlier snapshot"
    )
    parser.add_argument(
        "--after", type=Path, help="use this saved after snapshot for exact deltas"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    current = (
        json.loads(args.after.read_text()) if args.after else snapshot(args.sysfs_root)
    )
    if args.before:
        current = delta(json.loads(args.before.read_text()), current)
    args.out.write_text(json.dumps(current, indent=2) + "\n")


if __name__ == "__main__":
    main()
