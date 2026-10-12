#!/usr/bin/env python3
"""Compile the standalone TSAN driver and actual allocator TU; never run it."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

repo = Path('/home/kkristensen/Code/cuCascade')
build = repo / 'build/pr-reevaluation'
directory = Path(__file__).resolve().parent
cache = {}
for line in (build / 'CMakeCache.txt').read_text().splitlines():
    if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line:
        continue
    key, value = line.split('=', 1)
    cache[key.split(':', 1)[0]] = value
compiler = cache['CMAKE_CXX_COMPILER']
environment = Path(cache['rmm_DIR']).parents[2]
source = repo / 'src/memory/small_pinned_host_memory_resource.cpp'
flags = ['-std=c++20', '-O1', '-g', '-DNDEBUG', '-pthread', '-fno-omit-frame-pointer',
         '-fsanitize=thread', '-Wall', '-Wextra', '-Wpedantic', '-Werror']
flags += shlex.split(cache.get('CMAKE_CXX_FLAGS', ''))
flags += ['-I', str(repo / 'include'), '-isystem', str(environment / 'include'),
          '-isystem', str(environment / 'include/rapids')]
for target in (environment / 'targets').glob('*/include'):
    flags += ['-isystem', str(target), '-isystem', str(target / 'cccl')]
flags += ['-DCCCL_DISABLE_PDL', '-DCUB_DISABLE_NAMESPACE_MAGIC', '-DCUB_IGNORE_NAMESPACE_MAGIC_ERROR',
          '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA', '-DTHRUST_DISABLE_ABI_NAMESPACE',
          '-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP', '-DTHRUST_IGNORE_ABI_NAMESPACE_ERROR']
command = [compiler, *flags, str(directory / 'driver.cpp'), str(source),
           '-L', str(build), '-lcucascade', '-lcucascade_topology_discovery',
           '-L', str(environment / 'lib'), '-lrmm', '-lrapids_logger', '-lnuma', '-lcudart', '-ldl',
           '-Wl,--export-dynamic', '-Wl,-rpath,' + str(build),
           '-Wl,-rpath,' + str(environment / 'lib'),
           '-Wl,-rpath-link,' + str(build), '-Wl,-rpath-link,' + str(environment / 'lib'),
           '-o', str(directory / 'tsan_driver')]
print(shlex.join(command), flush=True)
subprocess.run(command, cwd=repo, check=True)
(directory / 'build-metadata.json').write_text(json.dumps({
    'repo': str(repo), 'compiler': compiler, 'command': command,
    'allocator_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'coverage': 'driver and exact allocator TU instrumented; fixed upstream and CUDA/RMM libraries uninstrumented',
}, indent=2) + '\n')
print('Compiled only; root agent must execute under its GPU lock.')
