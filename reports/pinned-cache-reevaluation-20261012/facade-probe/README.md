# Observer-only NUMA facade dispatch probe

The observer extension compiled and linked successfully with zero warnings after the root agent finished all 11 fixed A/B timing blocks and authorized CPU compilation. No developer GPU execution occurred. The timing baseline/candidate extensions and production allocator header/source/library were checked by SHA256 before and after the build and remain unchanged.

The copied `numa_small_pinned_mr.hpp` adds four relaxed atomic entry counters: asynchronous allocation/deallocation and synchronous allocation/deallocation. Its destructor writes one `FACADE_PROBE` summary to stderr. Existing dispatch behavior, including synchronous methods' delegation into asynchronous methods, remains unchanged. Asynchronous totals include those delegated calls; `direct_async_*` subtracts the corresponding synchronous count. Entry counts prove reachability, including attempts that later throw; they do not imply a completed allocation.

`facade-observer.patch.gz` (gzip-compressed unified diff) is the complete minimal source patch. `prepare-metadata.json` records original wrapper, instrumented wrapper, and unchanged owner hashes. Only the private NUMA facade layout changes, and its constructing/deleting owner plus inline dispatch/resource-view callbacks will be freshly compiled in `SiriusContext`. The candidate allocator header/source/archive stay unchanged.

To reproduce the observer build after the root agent allows CPU load:

```bash
pixi run --as-is --manifest-path /home/kkristensen/Code/sirius/pixi.toml   python3 /tmp/cucascade-pinned-pr-20261012/facade-probe/build.py   > /tmp/cucascade-pinned-pr-20261012/facade-probe/build.log 2>&1
```

The script rebuilds only the owner against the copied observer header and the existing candidate allocator overlay. It patches a copy of the candidate core archive, verifies every other member remains identical, and reuses the candidate link/metadata commands with an independent output extension. It performs no runtime execution. `build-metadata.json` records the observer artifact and unchanged timing-candidate hash. The owner depfile confirms use of the instrumented NUMA facade header and unchanged candidate allocator header; the compiled observer marker is present. All 333 other core archive members were verified byte-identical.

The root agent should run the full 22-query diagnostic with the same configuration/pinning regime used for final timings, under its GPU lock, and capture stderr. Nonzero asynchronous counts establish that the facade was exercised; zero synchronous counts then bound the identified stale synchronous facade contract to an unused path in that diagnostic. Nonzero synchronous counts require a follow-up assessment; this observer does not fix or mask the behavior. The instrumented extension is for reachability evidence only and must never be used as a timing arm.


Compiled runtime artifact: `/tmp/cucascade-pinned-pr-20261012/facade-probe/sirius.duckdb_extension`, 112294030 bytes, SHA256 `8dd2d33209bb40b0c56ecdfce1df69c8ab4935e24c9626b26262f589e031036e`. Root-owned full-query diagnostic execution remains pending; no dispatch-reachability result is claimed until its summary is captured.
