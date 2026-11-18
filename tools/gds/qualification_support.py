#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded, best-effort metadata collection for qualify.sh (stdlib only)."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess


def capture(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "exit_code": p.returncode,
            "output": p.stdout + p.stderr,
        }
    except (OSError, subprocess.SubprocessError) as e:
        return {"command": command, "error": str(e)}


def metadata(output, binaries, gdscheck):
    config = Path(os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json"))
    try:
        contents = config.read_text()
    except OSError as e:
        contents = f"unavailable: {e}"
    checksums = {}
    for binary in binaries:
        try:
            checksums[binary] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        except OSError as e:
            checksums[binary] = str(e)
    version = ctypes.c_int()
    try:
        lib = ctypes.CDLL("libcufile.so.0")
        rc = lib.cuFileGetVersion(ctypes.byref(version))
        cufile = {"version": version.value, "exit_code": rc}
    except (OSError, AttributeError) as e:
        cufile = {"error": str(e)}
    data = {
        "git": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "binary_sha256": checksums,
        "libcufile": cufile,
        "driver": capture(["nvidia-smi"]),
        "cuda": capture(["nvcc", "--version"]),
        "gdscheck": capture([gdscheck, "-p"]),
        "cufile_json_path": str(config),
        "cufile_json_contents": contents,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUFILE_", "CUDA_", "PIXI_"))
            or k in ("PATH", "LD_LIBRARY_PATH", "CONDA_PREFIX")
        },
        "note": "Allowlisted environment only; still review paths/config before sharing. No automatic direct-path attribution.",
    }
    Path(output).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--gdscheck", required=True)
    parser.add_argument("binaries", nargs="*")
    args = parser.parse_intermixed_args()
    metadata(args.output, args.binaries, args.gdscheck)
