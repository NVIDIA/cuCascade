# Run inventory

All benchmark timings below are unprofiled; Nsight captures provide path evidence only.

| Run | Source | Timing | Validation |
|---|---|---|---|
| parquet-0-B | origin/main bd4aeed | [JSONL](raw/ab-parquet-0-B.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-1-C | PR 8b2d229 | [JSONL](raw/ab-parquet-1-C.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-2-C | PR 8b2d229 | [JSONL](raw/ab-parquet-2-C.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-3-B | origin/main bd4aeed | [JSONL](raw/ab-parquet-3-B.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-4-C | PR 8b2d229 | [JSONL](raw/ab-parquet-4-C.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-5-B | origin/main bd4aeed | [JSONL](raw/ab-parquet-5-B.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-6-B | origin/main bd4aeed | [JSONL](raw/ab-parquet-6-B.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| parquet-7-C | PR 8b2d229 | [JSONL](raw/ab-parquet-7-C.jsonl) | row hash, rows and columns; target allocation counts, exit 0 |
| churn-0-B | origin/main bd4aeed | [JSONL](raw/ab-churn-0-B.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-1-C | PR 8b2d229 | [JSONL](raw/ab-churn-1-C.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-2-C | PR 8b2d229 | [JSONL](raw/ab-churn-2-C.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-3-B | origin/main bd4aeed | [JSONL](raw/ab-churn-3-B.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-4-C | PR 8b2d229 | [JSONL](raw/ab-churn-4-C.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-5-B | origin/main bd4aeed | [JSONL](raw/ab-churn-5-B.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-6-B | origin/main bd4aeed | [JSONL](raw/ab-churn-6-B.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| churn-7-C | PR 8b2d229 | [JSONL](raw/ab-churn-7-C.jsonl) | H2D payload survives reuse; full requested allocation usable, exit 0 |
| noise-0-B | origin/main bd4aeed | [JSONL](raw/noise-baseline-0.jsonl) | same content/path checks, fixed baseline noise sample |
| noise-1-B | origin/main bd4aeed | [JSONL](raw/noise-baseline-1.jsonl) | same content/path checks, fixed baseline noise sample |
| noise-2-B | origin/main bd4aeed | [JSONL](raw/noise-baseline-2.jsonl) | same content/path checks, fixed baseline noise sample |
| noise-3-B | origin/main bd4aeed | [JSONL](raw/noise-baseline-3.jsonl) | same content/path checks, fixed baseline noise sample |

Profiles: `profiles/profile-baseline.nsys-rep`, `profiles/profile-candidate.nsys-rep`, and their SQLite exports; API/window/kernel/copy summary in `PROFILE-SUMMARY.json`. These capture four reads each with separate `read_parquet` NVTX windows. Initial setup and result hashing remain outside those read windows.

Prototype/pilot runs are retained in the original artifact directory but excluded from the verdict. Nsight reports one unavailable unified-memory tracing capability; runtime APIs, kernels, copies, NVTX and event data are present and are the only profiler evidence used.

Correctness logs: `raw/targeted-tests-final.log`, `raw/core-tests.log`, `raw/cudf-tests.log`, `raw/io-tests.log`, `raw/memcheck.log`, `raw/faults-results.log`, `raw/pending-dma-destructor.log`, and `raw/tsan-no-aslr.log`.
