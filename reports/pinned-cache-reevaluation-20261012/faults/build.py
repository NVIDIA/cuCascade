#!/usr/bin/env python3
"""Compile the independent harness; this script does not execute CUDA code."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path("/home/kkristensen/Code/cuCascade"))
parser.add_argument("--build", type=Path, default=Path("/home/kkristensen/Code/cuCascade/build/pr-reevaluation"))
parser.add_argument("--driver-only", action="store_true", help="Do not rewrite the loaded wrapper library")
arguments = parser.parse_args()
repo = arguments.repo.resolve()
build = arguments.build.resolve()
directory = Path(__file__).resolve().parent
cache = {}
for line in (build / "CMakeCache.txt").read_text().splitlines():
    if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
        continue
    key, value = line.split("=", 1)
    cache[key.split(":", 1)[0]] = value
compiler = cache["CMAKE_CXX_COMPILER"]
environment = Path(cache["rmm_DIR"]).parents[2]
flags = ["-std=c++20", "-O2", "-DNDEBUG", "-fPIC", "-pthread",
         "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
flags += shlex.split(cache.get("CMAKE_CXX_FLAGS", ""))
flags += ["-I", str(repo / "include"), "-I", str(environment / "include"),
          "-I", str(environment / "include/rapids")]
targets = list((environment / "targets").glob("*/include"))
for target in targets:
    flags += ["-I", str(target), "-I", str(target / "cccl")]
flags += ["-DCCCL_DISABLE_PDL", "-DCUB_DISABLE_NAMESPACE_MAGIC", "-DCUB_IGNORE_NAMESPACE_MAGIC_ERROR",
          "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA", "-DTHRUST_DISABLE_ABI_NAMESPACE",
          "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP", "-DTHRUST_IGNORE_ABI_NAMESPACE_ERROR"]
commands = [
    [compiler, *flags, "-shared", str(directory / "cuda_faults.cpp"), "-ldl",
     "-o", str(directory / "libpinned_faults.so")],
    [compiler, *flags, str(directory / "driver.cpp"),
     str(repo / "src/memory/small_pinned_host_memory_resource.cpp"),
     "-L", str(directory), "-lpinned_faults",
     "-L", str(build), "-lcucascade", "-lcucascade_topology_discovery",
     "-L", str(environment / "lib"), "-lrmm", "-lrapids_logger", "-lnuma", "-lcudart", "-ldl",
     "-Wl,--export-dynamic", "-Wl,-rpath," + str(directory), "-Wl,-rpath," + str(build),
     "-Wl,-rpath," + str(environment / "lib"),
     "-Wl,-rpath-link," + str(build), "-Wl,-rpath-link," + str(environment / "lib"),
     "-o", str(directory / "fault_driver.next")],
]
if arguments.driver_only:
    commands = commands[1:]
for command in commands:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)
(directory / "fault_driver.next").replace(directory / "fault_driver")
(directory / "build-metadata.json").write_text(json.dumps({
    "repo": str(repo), "build": str(build), "compiler": compiler,
    "environment": str(environment), "commands": commands,
}, indent=2) + "\n")
print("Built only; root agent must execute under its GPU lock.")
