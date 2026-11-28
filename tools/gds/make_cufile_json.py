#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Select GPU-local RDMA addresses for cuFile-managed RDMA (stdlib only).

No driver is opened, affinity changed, or filesystem-client configuration modified.
A synthetic --sysfs-root disables live command discovery to keep fixtures isolated.
"""

import argparse
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BDF = re.compile(r"^(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


def bdf(value):
    if not BDF.fullmatch(value):
        raise ValueError(f"invalid PCI BDF: {value}")
    parts = value.lower().split(":")
    domain = int(parts[0], 16) if len(parts) == 3 else 0
    return f"{domain:04x}:{parts[-2]}:{parts[-1]}"


def command(args):
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def node(path):
    try:
        return int(read(path / "numa_node"))
    except ValueError:
        return -1


def uncomment(text):
    # Match complete strings first: // in URLs or escaped quotes is not a comment.
    return re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n\r]*',
        lambda m: "" if m[0].startswith("//") else m[0],
        text,
    )


def address(value):
    try:
        addr = ipaddress.ip_address(value.split("%", 1)[0])
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        # Link-local addresses need a scope and cannot safely be put in this list.
        if (
            addr.is_unspecified
            or addr.is_multicast
            or addr.is_link_local
            or addr.is_loopback
        ):
            return None
        return str(addr)
    except ValueError:
        return None


@dataclass
class Device:
    name: str
    bdf: str
    node: int
    path: Path
    addresses: tuple = ()
    netdevs: tuple = ()
    index: str = ""


def pci_path(root, pci):
    return (root / "bus/pci/devices" / pci).resolve()


def discover_gpus(requests, root, live):
    inventory = {}
    if live:
        for line in command(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines():
            fields = [v.strip() for v in line.split(",")]
            if len(fields) == 2:
                try:
                    inventory[fields[0]] = bdf(fields[1])
                except ValueError:
                    pass
    result = []
    for selected in requests:
        pci = inventory.get(selected) if selected.isdecimal() else bdf(selected)
        if not pci:
            raise ValueError(
                f"GPU index {selected} unavailable from nvidia-smi; use its PCI BDF "
                "(sysfs ordering is not a CUDA device index)"
            )
        path = pci_path(root, pci)
        if not path.exists():
            raise ValueError(f"GPU {selected}: PCI device {pci} missing under {root}")
        index = next((i for i, p in inventory.items() if p == pci), "")
        result.append(Device(selected, pci, node(path), path, index=index))
    return result


def discover_nics(root, live, allow_ipv6=False):
    ip_addrs = {}
    links = []
    if live:
        try:
            for interface in json.loads(command(["ip", "-j", "addr"]) or "[]"):
                ip_addrs[interface["ifname"]] = [
                    a["local"] for a in interface.get("addr_info", []) if "local" in a
                ]
            links = json.loads(command(["rdma", "-j", "link"]) or "[]")
        except (ValueError, KeyError, TypeError):
            pass
    result = []
    for ib in sorted((root / "class/infiniband").glob("*")):
        device = (ib / "device").resolve()
        if not device.exists():
            continue
        try:
            pci = bdf(device.name)
        except ValueError:
            print(f"Skipped RDMA device {ib.name}: non-PCI/auxiliary device {device}")
            continue
        netdevs = {p.name for p in (device / "net").glob("*")}
        netdevs.update(
            value for p in ib.glob("ports/*/gid_attrs/ndevs/*") if (value := read(p))
        )
        for link in links:
            if link.get("ifname", "").split("/")[0] == ib.name and link.get("netdev"):
                netdevs.add(link["netdev"])
        for netdev in (root / "class/net").glob("*"):
            if (netdev / "device").resolve() == device:
                netdevs.add(netdev.name)
        values = [v for net in netdevs for v in ip_addrs.get(net, [])]
        # RoCE GIDs with ndev associations supply IPv4-mapped addresses without iproute2.
        # IPoIB addresses come from ip -j addr, not unassociated IB GIDs.
        for gid in ib.glob("ports/*/gids/*"):
            net = read(gid.parent.parent / "gid_attrs/ndevs" / gid.name)
            if net and (root / "class/net" / net).exists():
                values.append(read(gid))
        candidates = {a for v in values if (a := address(v))}
        ipv6 = sorted(a for a in candidates if ":" in a)
        if ipv6:
            print(
                f"[verify] {ib.name} IPv6 candidates {'included' if allow_ipv6 else 'excluded'}: "
                f"{','.join(ipv6)}; cuFile documents IPv4 RDMA lists"
            )
        addresses = tuple(sorted(a for a in candidates if allow_ipv6 or ":" not in a))
        result.append(
            Device(
                ib.name, pci, node(device), device, addresses, tuple(sorted(netdevs))
            )
        )
    return result


def topo_links(path):
    """Optional nvidia-smi topo -m matrix + NIC legend, used only as fallback."""
    if path is None:
        return {}
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    names = dict(re.findall(r"^\s*(NIC\d+):\s*(\S+)", text, re.M))
    headers = []
    result = {}
    for line in text.splitlines():
        cols = line.split()
        if cols and cols[0] == "GPU0" and not headers:
            headers = cols
            continue
        if headers and cols and re.fullmatch(r"GPU\d+", cols[0]):
            for column, value in zip(headers, cols[1:]):
                if column in names:
                    result[(cols[0][3:], names[column])] = value
    return result


def rank(gpu, nic, topo):
    # Exclude endpoints and the PCI host root: shared upstream PCI bridges only.
    bridges = [p for p in gpu.path.parents if BDF.fullmatch(p.name)]
    nic_bridges = [p for p in nic.path.parents if BDF.fullmatch(p.name)]
    hops = [
        i + j
        for i, p in enumerate(bridges)
        for j, q in enumerate(nic_bridges)
        if p == q
    ]
    if hops:
        return (0, min(hops)), "shared upstream PCIe bridge (ranked by hop distance)"
    link = topo.get((gpu.index, nic.name))
    if link in ("PIX", "PXB"):
        return (
            0,
            0 if link == "PIX" else 2,
        ), f"[verify] topo {link}; sysfs bridge unavailable"
    if gpu.node >= 0 and gpu.node == nic.node:
        return (1, 0), "same NUMA node"
    return (2, 0), "[verify] remote or unknown proximity"


def select(gpus, nics, topo):
    selected = set()
    rows = []
    for gpu in gpus:
        usable = [(rank(gpu, nic, topo), nic) for nic in nics if nic.addresses]
        best = min((r[0] for r, _ in usable), default=None)
        for nic in nics:
            score, reason = rank(gpu, nic, topo)
            chosen = bool(nic.addresses) and score == best
            if chosen:
                selected.update(nic.addresses)
            rows.append((gpu, nic, chosen, reason))
    return sorted(selected), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="append", required=True, help="GPU index or PCI BDF; repeatable"
    )
    parser.add_argument(
        "--base", type=Path, help="optional base cufile.json (// comments accepted)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=("RoundRobin", "RoundRobinMaxMin"))
    parser.add_argument(
        "--allow-ipv6",
        action="store_true",
        help="[verify] include IPv6; libcufile documents IPv4 only",
    )
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--nvidia-smi-topo", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.base and args.out.resolve() == args.base.resolve():
            raise ValueError("--out must not overwrite --base")
        base = json.loads(uncomment(args.base.read_text())) if args.base else {}
        if not isinstance(base, dict) or not isinstance(
            base.get("properties", {}), dict
        ):
            raise ValueError(
                "base must be a JSON object with an object properties field"
            )
        live = args.sysfs_root.resolve() == Path("/sys")
        gpus = discover_gpus(args.gpu, args.sysfs_root, live)
        nics = discover_nics(args.sysfs_root, live, args.allow_ipv6)
        addresses, rows = select(gpus, nics, topo_links(args.nvidia_smi_topo))
        print(
            "GPU  GPU-BDF       NUMA  RDMA-NIC  NIC-BDF       NUMA  Selected  Addresses / reason"
        )
        if not nics:
            for gpu in gpus:
                print(f"{gpu.name}  {gpu.bdf}  {gpu.node}  (no RDMA NICs)")
        for gpu, nic, chosen, reason in rows:
            print(
                f"{gpu.name}  {gpu.bdf}  {gpu.node}  {nic.name}  {nic.bdf}  {nic.node}  "
                f"{'yes' if chosen else 'no'}  {','.join(nic.addresses) or '(no usable IP)'} / {reason}"
            )
        print(
            "[verify] libcufile multi-GPU list interpretation, policy support and rdma_dynamic_routing "
            "are version-specific; this union is not a per-GPU routing guarantee."
        )
        print(
            "Filesystem-client RDMA is NOT controlled here: GPFS: verify verbsPorts; "
            "WEKA: verify net=; VAST: verify localports= with the client administrator."
        )

        def warn_fs(value, path=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    location = f"{path}.{key}" if path else key
                    if (
                        (
                            path.startswith("fs.")
                            and key in ("rdma_dev_addr_list", "mount_table")
                        )
                        or key == "mount_table"
                        or (key == "rdma_dynamic_routing" and child is True)
                    ):
                        print(
                            f"WARNING: base {location} may steer independently; "
                            "only properties.rdma_dev_addr_list is updated"
                        )
                    warn_fs(child, location)
            elif isinstance(value, list):
                for child in value:
                    warn_fs(child, path)

        warn_fs(base)
        print(
            "properties.rdma_dev_addr_list: user-space filesystem clients per stock cuFile JSON; "
            "kernel NFS/Lustre/BeeGFS dynamic routing uses fs.<type> lists/mount_table."
        )
        if not addresses:
            print(
                "No RDMA NICs with usable IP addresses; no JSON written, no export generated."
            )
            return 0
        properties = base.setdefault("properties", {})
        properties["rdma_dev_addr_list"] = addresses
        if args.policy is not None:
            properties["rdma_load_balancing_policy"] = args.policy
        else:
            properties.setdefault("rdma_load_balancing_policy", "RoundRobin")
        if properties["rdma_load_balancing_policy"] not in (
            "RoundRobin",
            "RoundRobinMaxMin",
        ):
            print(
                "WARNING: [verify] undocumented base rdma_load_balancing_policy preserved"
            )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=args.out.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(json.dumps(base, indent=2) + "\n")
            os.replace(temporary, args.out)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Wrote {args.out}")
        print(f"export CUFILE_ENV_PATH_JSON={shlex.quote(str(args.out.resolve()))}")
        return 0
    except (OSError, ValueError) as error:
        print(f"make_cufile_json: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
