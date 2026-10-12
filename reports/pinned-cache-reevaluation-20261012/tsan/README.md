# Allocator ThreadSanitizer stress harness

`build.py` compiles only; it never executes CUDA code. `tsan_driver` instruments both the driver and the exact final production `small_pinned_host_memory_resource.cpp` with GCC ThreadSanitizer. It links shared cudart so the instrumented allocator calls the same runtime as the driver. The candidate core library supplies only the fixed upstream implementation; core dependencies and CUDA/RMM internals are not instrumented.

The driver uses eight threads, each with a nonblocking stream, and 1000 allocate/write/deallocate cycles over six large request sizes (9000 through 300000 bytes). A 512 KiB retention cap forces eviction across buckets. Allocation streams synchronize before CPU access, a separately locked live-pointer tracker checks exclusive ownership, and each cycle verifies host data and the cap. Threads join before resource destruction.

Build:

```bash
source /tmp/cucascade-pinned-pr-20261012/env.sh
python3 /tmp/cucascade-pinned-pr-20261012/tsan/build.py
```

Root-owned execution under the existing GPU lock:

```bash
/tmp/cucascade-pinned-pr-20261012/gpu-run.sh env TSAN_OPTIONS=halt_on_error=1:exitcode=66 /tmp/cucascade-pinned-pr-20261012/tsan/tsan_driver
```

`build-metadata.json` includes the exact allocator source SHA256. The prepared binary has dynamic dependencies `libtsan.so.2` and `libcudart.so.13`. No developer-side execution occurred. An unsupported sanitizer address mapping or CUDA-library initialization failure is an environment limitation rather than a passed race check; preserve diagnostics and distinguish it from a reported allocator race.
