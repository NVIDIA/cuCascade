# Isolated Sirius allocator A/B consumer build

Compilation and relinking completed without executing GPU code. Original Sirius and cuCascade sources and original build artifacts were read only. Outputs live exclusively in this directory.

The consumer engine source is current Sirius `45253bf5074eb9462bbe6ef0c1dc081a252adf03` with consumer cuCascade dependency `e1b67fe7d44229597af562d46bb8383828d77981`. The explorer independently compared every engine `src/` and `include/` file and every dependency `src/` and `include/` file against the newer Oct 9 worktree used for the Oct 11 object build and found them byte-identical. The donor worktree itself records an older merge commit plus dirty patches; its exact git status/diff hash and input artifact hashes are recorded in `build-metadata.json`. Existing original modifications to `pixi.lock` and the performance runner were preserved.

The original consumer dependency differs from current cuCascade main in reservation accounting. Therefore this experiment does not swap the whole latest core library into the consumer. It changes only `small_pinned_host_memory_resource.hpp` and its implementation, preserving all other consumer headers and archive members. The consumer's fixed-size upstream header and implementation already match current main byte-for-byte. The baseline allocator files come from cuCascade `origin/main` at `bd4aeedbee1201df68bd042374fd9ab6afb5b46d`; the candidate files come from merged commit `8b2d229c1f7b064aad6db930bf7a9c396531e1ba`.

Both arms freshly compile the actual allocator translation unit and `SiriusContext`, the allocating/deleting owner, against their corresponding narrow header overlay. This accounts for constructor and class-layout ABI changes. The inline NUMA wrapper is included only through that owner. A copy of the current 334-member `libsirius_core.a` replaces only `sirius_context.cpp.o`; a copy of the current 22-member `libcucascade.a` replaces only `small_pinned_host_memory_resource.cpp.o`. Every other archive member was extracted and compared byte-for-byte with its donor and between arms. Current CUDA device linking remains intact through the device-link object already present in the current core archive. Symbol inspection verifies a one-argument allocator constructor in the baseline owner/extension and the new two-argument constructor in the candidate owner/extension.

Current `sirius-duckdb` entry glue was freshly compiled once and shared by both arms. The exact known working extension link flags are reused, replacing every old Sirius object with the current core archive. Matching clean DuckDB revision `069cc9f9b5be802405797faecc284961b07c70ef` supplies existing DuckDB/core-functions/Parquet archives. Current donor cuCascade-cudf, topology, Simpatico and Rust telemetry archives are preserved. The normal DuckDB CPP metadata append step was executed on both isolated outputs (DuckDB v1.5.6). No whole latest-main header overlay, preload substitution, original artifact mutation, or GPU execution occurred.

## Outputs

- `baseline/sirius.duckdb_extension`: 112292862 bytes, SHA256 `5e4582186ce64df86d58a570377ee23bbc0aee7e01915ccb8191ba708b5fe6b8`.
- `candidate/sirius.duckdb_extension`: 112294014 bytes, SHA256 `00ed9a0b550907e7f2d49167e24bc127b921006b32cde8156b0c6e31dbc70ec5`.

`build-metadata.json` contains each compiler/linker argv, source/owner/object/archive/extension hashes, original consumer/donor commit and dirty-state provenance, and unchanged-member aggregate hashes. `build.log` contains compiler/linker output. `build.py` reconstructs both outputs with preserved compiler options and existing dependencies:

```bash
pixi run --as-is --manifest-path /home/kkristensen/Code/sirius/pixi.toml   python3 /tmp/cucascade-pinned-pr-20261012/sirius-build/build.py
```

The existing installed Sirius Pixi environment is activated with `--as-is`; no lock-file update or package installation occurs.

## Readings and execution ownership

The developer completed current and donor Sirius `AGENTS.md`, the frozen dependency's `AGENTS.md`, Sirius building/development/usage and wrapper documentation, Super Sirius's documentation index and pinned-memory/stream-ordering ownership contracts, and the TPCH primary-runner/profile/validation instructions before isolated mutations. Earlier assigned C++ software-design, C++20 and Core Guidelines readings remain complete. The user-authorized GSD bypass applies to this PR reevaluation. No operator implementation or original consumer source was modified.

The root agent owns all extension loading, GPU benchmarks, result validation and GPU locking. Available SF50 data and reusable runner:

- `/tmp/sirius-pr1834-streams-20261007-1940/data/tpch_sf50`
- `/tmp/sirius-pr1834-streams-20261007-1940/runner.py`

The earlier runner enables compressed pinning, so the root agent must select and record the intended hot-cache regime rather than silently equating that historical configuration with the original PR's full-query claim. Full 22-query CPU reference and GPU-path/result checks are separate runtime validation work; this build report makes no performance or correctness claim for the extensions before those runs.
