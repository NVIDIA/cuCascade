# Pinned host cache validation and external handoff

This is the validation record for [ticket 11](https://github.com/kevkrist/cuCascade/issues/11) and [spec 1](https://github.com/kevkrist/cuCascade/issues/1). Two-GPU correctness is satisfied, and the user has accepted the measured deterministic eviction regression. Both issues remain open for the original four-GPU application comparison. No upstream review responses have been posted.

## Accepted performance tradeoff (2026-11-03)

The user explicitly revised the plan with "Let's accept this regression." This accepts the current implementation's measured `deterministic_full_cap` throughput losses of 7.98% on ARM GB10 and 7.72% on x86_64 Xeon/RTX PRO 4500. The exception covers the single-thread, single-stream, 64 KiB cache control with repeated 9,000-byte and 40,000-byte allocations. The tradeoff preserves the correctness guarantees and measured concurrent throughput gains without adding ownership or locking complexity solely for this stress case.

This is acceptance of a measured loss, not a passing result under the original 5% threshold. The exception is limited to this benchmark and validated implementation; it does not cover future worsening. The 5% gate remains unchanged for other cases and the original application comparison, and further regressions require investigation and resolution or another explicit plan revision. Original measurements and hashes are retained unchanged.

The original four-GPU application comparison remains the outstanding validation requirement. Acceptance of this microbenchmark regression does not waive that comparison, establish application performance, or authorize posting the draft upstream review responses.

## Local evidence

The local host has one NVIDIA GB10 GPU, Linux aarch64, driver 580.126.09, GNU C++ 15.3.0, and CUDA 13.2.86. The nightly environment provides RMM/cuDF 26.12 and CCCL 3.6; the installed stable environment provides RMM/cuDF 26.10 and CCCL 3.5. Both builds use Release, warnings as errors, and the normal production core, cuDF, and I/O libraries. Local builds target SM 120; external builds must select their own GPU architectures.

Final full-suite evidence at validation commit `10d9899` (reviewed feature cleanup `b2724f4`, two production controls, and the statistics test/documentation correction below), with fixed seed `218056058`:

| Environment | Production pinned | Private fault/event | Full cuDF |
| --- | --- | --- | --- |
| Nightly | 38 passed, 3 skipped; 868 assertions | 54 passed; 948 assertions | 197 passed, 2 skipped; 2,736 assertions |
| Stable | 38 passed, 3 skipped; 958 assertions | 54 passed; 948 assertions | 197 passed, 2 skipped; 2,737 assertions |

The concurrent statistics case also passed 20 separate process runs in each environment, seeds `218056058` through `218056077`, including the original failing seed. [Compact reports and summary](validation/pinned-host-cache/final-tests/summary.json) are retained. JUnit case counts include named sections and therefore exceed Catch's test-case totals above. Assertion totals in concurrent cases vary with scheduling. Every skip above requires physical multiple GPUs. Both production libraries also passed a dynamic-symbol check for absence of private test hooks. The single-GPU host cannot satisfy the production multi-GPU gate or reproduce the original four-L40S application measurements.

Owned C++ changes pass clang-format 20.1.4, the runner passes Black 25.1.0, the test source list passes cmake-format 0.6.13, and owned source/docs pass codespell and `git diff --check`. Both baseline and candidate benchmark executables compile/link with warnings as errors. Generated evidence is copied byte for byte from original reports; [SHA256SUMS](validation/pinned-host-cache/SHA256SUMS) covers the checked-in evidence files. Formatting these reports must not rewrite the original metadata. The generated JSON, JSONL, CSV, text evidence, and the exact `capture-perf/noquery.patch` diagnostic diff are excluded from formatting hooks and whitespace normalization to retain its recorded hashes; source, scripts, and manually authored Markdown remain subject to normal checks.

The fault executable compiles the real allocator and event implementation with private, scoped system-boundary hooks. It is separate from the production test executable and does not replace production-path validation. The two added `[small_pinned][production]` controls exercise over-aligned storage through a real mapped GPU kernel and zero/null synchronous and asynchronous wrappers during active capture. Existing production tests exercise real pending DMA and cross-stream reuse. Failure-path red/green evidence belongs to the implementation tickets; these additional controls verify already implemented production behavior.

### Statistics test correction

A final stable rerun exposed an intermittent test-assumption regression: seed `218056058` observed 58 arriving requests but only 57 classified hits/misses/uncacheable requests. The [original failure](validation/pinned-host-cache/statistics-before-fix.txt) is retained. Cache-hit wait submission now occurs outside the cache mutex, so a snapshot can observe an arriving hit before successful submission classifies it. This is a staged snapshot, not a lost request.

The concurrent nonfault test now bounds the temporary deficit by the number of workers, while retaining monotonicity, cache-cap/peak bounds, and exact conservation after workers finish. The gated progress control also checks literal counts while a hit wait is paused (2 requests, 0 hits, 1 miss), then checks 2 requests, 1 hit, 1 miss and exact conservation after joining. The statistics-scope documentation records this behavior alongside existing failure/retry both/neither semantics. This repair changes tests and documentation only.

### Paired production allocator results

Baseline is exactly `69ef173b10a75363d80aa3a8df833f2858159034`. The controlled measurement candidate is clean commit `5253670034dcb96cd14ebff4ba34a377977f911f`, containing reviewed feature cleanup `b2724f4` and the validation harness. The later statistics repair changes only tests and documentation. Both revision builds use the same compiler, Release flags, SM 120, dependencies, and CUDF/IO/benchmark build options; each executable uses that revision's own headers and production library. [Metadata](validation/pinned-host-cache/controlled/metadata.json) records exact commands, compiler version, implementation/header hashes, benchmark-source hash, and library hashes.

An initial five-pair run used the scheduler's broad inherited CPU affinity. It had bimodal 6/12 ms single-thread trials and large variance: hit/full-control throughput coefficients of variation were 17.6/11.3% and 37.5/38.3% for baseline/candidate. It reported median losses of 18.1% for mixed full-cap traffic, 9.6% for deterministic eviction, and 14.0% for half-cap rounding. These [raw results](validation/pinned-host-cache/exploratory/raw.jsonl) and [summary](validation/pinned-host-cache/exploratory/summary.json) remain available; they were not discarded or normalized by hit mix.

The follow-up uses seven alternating pairs, identical affinity on CPUs 15-18 (each reports capacity 1017), 10x longer mixed full-cap loops, and 100x longer deterministic control loops. CPU and GPU frequencies were not locked. [Hardware inventory](validation/pinned-host-cache/controlled/hardware.json), [raw results](validation/pinned-host-cache/controlled/raw.jsonl), and [summary](validation/pinned-host-cache/controlled/summary.json) are retained. Throughput below is allocation/deallocation pairs per second; CV is sample standard deviation divided by mean across the seven trials.

| Case | Baseline median | Candidate median | Throughput change | Baseline/candidate CV |
| --- | ---: | ---: | ---: | ---: |
| Concurrent hits | 943,111 | 2,383,439 | +152.72% | 1.4% / 9.8% |
| Concurrent full-cap | 196,629 | 230,953 | +17.46% | 0.9% / 2.1% |
| Deterministic full-cap | 1,024,324 | 942,635 | **-7.98%** | 0.3% / 0.2% |
| Half-cap rounding | 21.17 | 21.45 | +1.32% | 7.0% / 10.3% |

The deterministic regression is reproducible and exceeds the approved 5% threshold. Every controlled deterministic trial has exactly 300,000 hits, 300,000 misses, and 300,000 evictions in both revisions. Mixed concurrent hit counts differ and remain in the raw record; the mixed improvement does not erase the deterministic regression. Half-cap variability limits claims about small differences. At this measurement checkpoint the performance gate was open; the [2026-11-03 plan revision](#accepted-performance-tradeoff-2026-11-03) subsequently accepted this specific regression. The original application comparison remains pending.

Separate Nsight Systems diagnostics use 10,000 deterministic iterations and are excluded from acceptance timing. Both traces have 10,130 `cudaHostAlloc`, 10,130 `cudaFreeHost`, 20,257 event records, 10,128 stream waits, and 10,129 event synchronizations, including setup/warmup/teardown. The candidate additionally issues 40,514 capture-status queries, four per two-allocation loop plus warmup/setup, while the baseline issues none. [Baseline](validation/pinned-host-cache/api-diagnostics/baseline-api.csv) and [candidate](validation/pinned-host-cache/api-diagnostics/candidate-api.csv) API summaries identify the added capture checks as an overhead contributor; instrumented durations cannot prove that they account for the entire uninstrumented loss. No allocator algorithm or capture guarantee was changed in response to these timings.


### Bounded capture-check investigation

A separate uninstrumented diagnostic compared the two runtime capture-query APIs on CPU 15: seven alternating trials of 10 million queries gave median costs of 19.424 ns for `cudaStreamIsCapturing` and 21.555 ns for `cudaStreamGetCaptureInfo`. The alternative was slower and was not adopted. Both the installed CUDA 13.2 header and [NVIDIA stream documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html) describe capture-state queries, including the legacy-stream implicit-capture error. No driver-API substitution, cached capture state, or capture-policy change was attempted.

A seven-trial deterministic diagnostic then compared the baseline, an independently recompiled intact allocator, and a copied allocator whose capture-query helper always returned success and `cudaStreamCaptureStatusNone`. This last variant **violates the required capture-rejection contract, is unsafe for supported allocator behavior, and is not a candidate for integration**. Median throughput was 1,026,009, 944,244, and 980,962 allocation/deallocation pairs per second respectively; CV was 0.81%, 0.61%, and 0.87%. Each trial retained exactly 300,000 hits, misses, and evictions. Disabling the entire guard improved throughput by 3.89% relative to the intact diagnostic but still left a 4.39% loss against baseline. This isolates a contributor to the observed cost, not an acceptable fix or proof that all remaining cost is unavoidable.

The diagnostic reuses the candidate's other object files, headers, benchmark executable, flags, and static CUDA runtime; only the copied allocator translation unit is recompiled and relinked. It is not a clean-build acceptance comparison. An initial attempt failed to override the executable's `DT_RPATH` with `LD_LIBRARY_PATH`; that [invalid raw run](validation/pinned-host-cache/capture-perf/invalid-rpath-ablation-raw.jsonl) is retained and excluded. The final run uses explicit `LD_PRELOAD`, with loader selections and unique library hashes recorded in [experiment metadata](validation/pinned-host-cache/capture-perf/experiment-metadata.json). [Raw results](validation/pinned-host-cache/capture-perf/ablation-raw.jsonl), [summary](validation/pinned-host-cache/capture-perf/ablation-summary.json), [exact diagnostic patch](validation/pinned-host-cache/capture-perf/noquery.patch), and [reproduction notes](validation/pinned-host-cache/capture-perf/README.md) are retained. The bounded investigation found no simple safe reduction. At the end of that investigation, the 5% performance gate was still open. The [2026-11-03 plan revision](#accepted-performance-tradeoff-2026-11-03) subsequently accepted the measured deterministic eviction loss; production behavior remains unchanged.

## Two-GPU x86_64 validation evidence

A second host with two physical NVIDIA RTX PRO 4500 Blackwell Server Edition GPUs (`0000:37:00.0` and `0000:39:00.0`, single NUMA node, `NODE` PCIe interconnect), Linux x86_64 (`6.8.0-1064-aws`), Intel Xeon 6975P-C (48 vCPUs), driver `595.91.07`, GNU C++ `15.3.0`, and CUDA `13.2.86` completed the multi-GPU correctness gate and paired microbenchmark reproduction at commit `750df4f` (same allocator/event implementation hashes as `10d9899` and `5253670`). The `default` nightly environment built against RMM/cuDF `26.12` (`libcudf 26.12.00a674`, `librmm 26.12.00a38`, `libkvikio 26.12.00a42`, CCCL `3.6.0.0`); the `cuda-13-stable` environment built against RMM/cuDF `26.10` (`libcudf 26.10.00`, CCCL `3.5.0.0`). Both production `libcucascade.so` builds passed the `nm -D` check for absence of private test-hook symbols.

### Multi-GPU correctness suite (`CUDA_VISIBLE_DEVICES=0,1`, seed `218056058`)

| Environment | Production pinned (`[small_pinned]`) | Private fault/event | Full cuDF |
| --- | --- | --- | --- |
| Nightly (`default`) | 41 passed, 0 skipped; 921 assertions (50 JUnit sections) | 54 passed, 0 skipped; 948 assertions (60 JUnit sections) | 199 passed, 0 skipped; 2,781 assertions (276 JUnit sections) |
| Stable (`cuda-13-stable`) | 41 passed, 0 skipped; 931 assertions (50 JUnit sections) | 54 passed, 0 skipped; 948 assertions (60 JUnit sections) | 199 passed, 0 skipped; 2,763 assertions (276 JUnit sections) |

All three required multi-GPU `[small_pinned]` test cases (`Event ownership follows the freeing stream's device`, `Large cache reuse on another GPU waits for pending producer DMA`, and `Multiple GPUs concurrently reuse one large pinned cache`) and both multi-GPU `cucascade_cudf_tests` cases (`gpu_table_representation rejects a stream owned by another device` and `gpu_table_representation works on its own device whichever device is current`) executed and passed with zero skips, failures, or errors. Twenty additional concurrent statistics process runs per environment (seeds `218056058` through `218056077`) also passed. Complete [JUnit XML, compact reports, and summary](validation/pinned-host-cache/two-gpu/tests/summary.json) are retained. **The two-GPU correctness gate is satisfied.**

### Paired production allocator results on x86_64 (Xeon 6975P-C + 2x RTX PRO 4500)

Seven alternating AB/BA pairs were collected on CPUs 0-3 (`taskset --cpu-list 0-3`, `CUDA_VISIBLE_DEVICES=0,1`) comparing baseline `69ef173b10a75363d80aa3a8df833f2858159034` against candidate `750df4f84aa30cd9393913b2cacab6d30dba7491` with identical Release flags (`-O3 -DNDEBUG`), compiler (`x86_64-conda-linux-gnu-c++ 15.3.0`), CUDA architectures (`75-real;80-real;86-real;90a-real;100f-real;120a-real;120`), and dependencies. [Hardware inventory](validation/pinned-host-cache/two-gpu/controlled/hardware.json), [build metadata](validation/pinned-host-cache/two-gpu/controlled/metadata.json), [raw JSONL](validation/pinned-host-cache/two-gpu/controlled/raw.jsonl), and [summary](validation/pinned-host-cache/two-gpu/controlled/summary.json) are retained:

| Case | Baseline median | Candidate median | Throughput change | Paired ratio median | Baseline/candidate CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concurrent hits | 391,455 | 698,695 | +78.49% | 1.7816x (+78.16%) | 5.0% / 8.3% |
| Concurrent full-cap | 92,227 | 120,798 | +30.98% | 1.2729x (+27.29%) | 2.7% / 2.5% |
| Deterministic full-cap | 825,566 | 761,800 | **-7.72%** | 0.9230x (**-7.70%**) | 0.3% / 0.5% |
| Half-cap rounding | 24.58 | 24.51 | -0.25% | 0.9975x (-0.25%) | 0.3% / 1.2% |

The deterministic full-cap regression reproduces on x86_64 + Blackwell at **-7.72%** median throughput (`-7.70%` median paired ratio, CV 0.27% / 0.52%, with all 7 trials recording exactly 300,000 hits, 300,000 misses, and 300,000 evictions), closely matching the `-7.98%` result on the ARM GB10 host. The user accepted this measured deterministic eviction regression in the [2026-11-03 plan revision](#accepted-performance-tradeoff-2026-11-03). Because this machine has 2 physical GPUs rather than the original 4x L40S GPUs and SF1000 dataset, **the original four-GPU application comparison remains pending**, keeping [spec 1](https://github.com/kevkrist/cuCascade/issues/1) and [ticket 11](https://github.com/kevkrist/cuCascade/issues/11) open.

## Build and run on the external host

Start from the final integration commit and use separate build directories for each environment. Use Pixi activation for both configuration and compilation, so NVCC uses the matching host compiler. Do not reuse local ARM binaries or copy the local SM 120 override to L40S hardware.

```bash
set -euo pipefail
for CUCASCADE_CASE in default cuda-13-stable; do
  pixi run -e "$CUCASCADE_CASE" cmake --preset release \
    -B "build/pinned-spec-$CUCASCADE_CASE" \
    -DCUCASCADE_BUILD_TESTS=ON -DCUCASCADE_BUILD_CUDF=ON \
    -DCUCASCADE_BUILD_IO=ON -DCUCASCADE_BUILD_BENCHMARKS=ON \
    -DCUCASCADE_WARNINGS_AS_ERRORS=ON
  pixi run -e "$CUCASCADE_CASE" cmake --build "build/pinned-spec-$CUCASCADE_CASE" \
    --target cucascade_tests cucascade_pinned_host_fault_tests cucascade_cudf_tests \
    --parallel 4
done
```

Serialize these commands with other GPU work on that machine. For example, acquire a shared `flock` lease in the shell running the following loop. Select at least two actual physical GPUs; use four IDs for the four-device stress check. Preserve inventory, environment, exact commit, compiler versions, CMake caches, and complete test reports.

```bash
set -euo pipefail
exec 9>/tmp/cucascade-validation-gpu.lock
flock 9
export CUDA_VISIBLE_DEVICES=0,1
export CUCASCADE_TEST_GPU_POOL_BYTES=67108864
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
git rev-parse HEAD
for CUCASCADE_CASE in default cuda-13-stable; do
  CUCASCADE_OUTPUT="build/pinned-spec-$CUCASCADE_CASE"
  pixi run -e "$CUCASCADE_CASE" "$CUCASCADE_OUTPUT/test/cucascade_tests" \
    '[small_pinned]' --reporter junit --out "$CUCASCADE_OUTPUT/pinned.xml"
  pixi run -e "$CUCASCADE_CASE" "$CUCASCADE_OUTPUT/test/cucascade_pinned_host_fault_tests" \
    --reporter junit --out "$CUCASCADE_OUTPUT/faults.xml"
  pixi run -e "$CUCASCADE_CASE" "$CUCASCADE_OUTPUT/test/cucascade_cudf_tests" \
    --reporter junit --out "$CUCASCADE_OUTPUT/cudf.xml"
done
```

All three production pinned multi-GPU cases must be present and pass without skips:

1. `Event ownership follows the freeing stream's device`.
2. `Large cache reuse on another GPU waits for pending producer DMA`.
3. `Multiple GPUs concurrently reuse one large pinned cache`.

The full cuDF suite also includes two existing cases requiring two devices: `gpu_table_representation rejects a stream owned by another device` and `gpu_table_representation works on its own device whichever device is current`. A zero process exit code with skipped cases is not a completed multi-GPU gate. Check JUnit files for failures, errors, and skips, and verify case names rather than relying on an empty filtered run:

```bash
python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET

required = {
    "Event ownership follows the freeing stream's device",
    "Large cache reuse on another GPU waits for pending producer DMA",
    "Multiple GPUs concurrently reuse one large pinned cache",
}
for environment in ("default", "cuda-13-stable"):
    folder = Path("build") / ("pinned-spec-" + environment)
    for report in ("pinned.xml", "faults.xml", "cudf.xml"):
        root = ET.parse(folder / report).getroot()
        cases = list(root.iter("testcase"))
        assert cases, (environment, report, "no test cases")
        for tag in ("skipped", "failure", "error"):
            assert not list(root.iter(tag)), (environment, report, tag)
        if report == "pinned.xml":
            names = {case.attrib["name"] for case in cases}
            assert required <= names, (environment, required - names)
        print(environment, report, len(cases), "passed without skips")
PY
```

## Reproduce the paired microbenchmarks

The reusable runner compiles `benchmark/benchmark_pinned_host_cache.cpp` against each revision's own public headers and production library. It extracts dependencies from configured Ninja builds; it does not link the fault variant. Build baseline `69ef173b10a75363d80aa3a8df833f2858159034` and candidate with identical compiler, configuration, architecture, and dependencies. At minimum build `cucascade_shared` in each; configure tests so the runner can obtain the production link command. Run from the candidate checkout under the same GPU lease, without concurrent builds or workloads:

```bash
pixi run -e default python scripts/benchmark_pinned_host_cache.py \
  --baseline-build /path/to/baseline-build \
  --candidate-build /path/to/candidate-build \
  --output-dir /path/to/fresh-results --pairs 5
```

The baseline checkout does not need the benchmark source backported. The runner uses the same candidate benchmark source for both revisions, rejects differing compiler/configuration settings, alternates baseline/candidate order between pairs, and saves raw JSONL, metadata, and medians/minimum/maximum/sample coefficient of variation. Record hardware and active GPU selection separately. Keep the output directory; a second run must use a fresh directory.

On a heterogeneous CPU, choose the same homogeneous group for both revisions and record it. The local controlled command prefixed the runner with `taskset --cpu-list 15-18` and used `--pairs 7`. Those CPU IDs are specific to this machine. The runner records inherited CPU affinity; defaults are 500,000 hit iterations, 30,000 mixed full-cap iterations, 300,000 deterministic control iterations, and 40 half-cap iterations, each per worker. Override them with `--hit-iterations`, `--full-iterations`, `--control-iterations`, and `--half-iterations` when trials are too short or variable.

Each worker creates its own real nonblocking stream and completes untimed warmup before the timed gate. Timing includes the operation loops, final stream synchronization, and worker teardown. Counters subtract warmup. One operation is one allocation/deallocation pair; a two-allocation iteration counts twice. No buffer contents are initialized or copied inside the timed loops.

| Case | Threads | Cache geometry | Purpose |
| --- | --- | --- | --- |
| `concurrent_hits` | 4 | 1 MiB cap; 9,000-byte requests | Shared-resource hit contention |
| `concurrent_full_cap` | 4 | 64 KiB cap; 9,000 and 40,000 bytes live together | Mixed buckets, full-cap eviction and scheduling-dependent hit mix |
| `deterministic_full_cap` | 1 | Same 64 KiB geometry | Full-cap control with repeatable hit/miss/eviction counts |
| `half_cap_rounding` | 1 | Default 256 MiB cap; two 128 MiB + 1 byte requests live together | Both requests round to full-cap buckets; eviction of whole retained buffers |

The mixed concurrent workload remains part of the result even if hit rates differ. Do not divide away misses or report only the deterministic control. A reproducible median throughput loss above 5% requires investigation and resolution or explicit revision of the approved plan. The [2026-11-03 acceptance](#accepted-performance-tradeoff-2026-11-03) is limited to the recorded deterministic control and current implementation. If variability prevents classification, increase iteration counts and collect additional alternating pairs. These allocation-only microbenchmarks cannot establish application performance.

## Original four-GPU application gate

The original four-L40S runner, Sirius commit, configuration, and SF1000 data path have not been recovered. The adjacent single-GB300 `bench/sf1000-repro` recipe describes a different workload and is not a substitute. Reuse the user's original four-GPU setup, including only `lineitem` and `orders` compression, all other tables uncompressed, original NUMA placement, cache preparation, query selection, and host-pinning mode.

The existing Sirius runner accepts the following syntax; the paths/configuration below must be replaced with the original ones. This establishes a runnable interface, not equivalence to the previous measurement:

```bash
cd /path/to/sirius
SIRIUS_CONFIG_FILE=/path/to/original-four-gpu-config.yaml \
  pixi run ./test/tpch_performance/run_tpch_parquet.sh \
  --parquet-dir /path/to/original-tpch-parquet-sf1000 --iterations 6 \
  --pinning-mode none sirius 1000 $(seq 1 22)

SIRIUS_CONFIG_FILE=/path/to/original-four-gpu-config.yaml SIRIUS_PIN_TIER=host \
  pixi run ./test/tpch_performance/run_tpch_parquet.sh \
  --parquet-dir /path/to/original-tpch-parquet-sf1000 --iterations 6 \
  --pinning-mode pinned-hot sirius 1000 $(seq 1 22)
```

Sirius links cuCascade statically from its subdirectory: rebuild Sirius separately against the exact baseline and candidate cuCascade commits. Changing `LD_LIBRARY_PATH` is insufficient. Record both Sirius and cuCascade revisions and build commands. Preserve correctness validation against the existing CPU oracle or saved validated outputs.

Six iterations in one session mean one cold and five warm iterations, not five independent paired trials. Run at least five complete baseline/candidate session pairs for each original workload, alternating AB/BA order. Save every query timing and validation result, warm aggregates, median and variability across independent sessions, configuration/data identity, cache preparation and pinning details. Do not mix host-pin-all results with warm unpinned results. Apply the same >5% regression gate to elapsed time using `(candidate - baseline) / baseline`.

## Contract and compatibility audit

`allocate` provides stream order; CPU access requires synchronizing that stream or using `allocate_sync`. Contents remain uninitialized. Deallocation requires the original byte count and alignment; zero/null no-ops do not infer a live allocation's size. Exceptional aligned/exact allocations retain their true release base and bypass bucket caching. The cache limit measures retained large-buffer capacity, excluding live allocations, slabs, detached pending storage, and storage withheld after failure. Recovery deliberately skips partial reclamation when retained capacity is less than the complete exact backing size, including alignment padding. CUDA graph capture and host callback allocator use remain unsupported, with documented supplied-stream rejection behavior.

The public constructor and allocation/deallocation call shapes remain usable; invalid alignment is now rejected, and previously ignored larger alignment is honored. The new `cuda_event::reset_no_throw()` operation is additive. Private resource layout and synchronization state changed, so binary consumers must rebuild against matching headers and library. cuDF wiring remains `cudf::set_pinned_memory_resource(resource)` with `cudf::set_allocate_host_as_pinned_threshold(small_pinned_host_memory_resource::MAX_SLAB_SIZE)`. No `idisk_io_backend` API changes, new public fault controls, node pooling, or eviction-policy replacement are part of this work.
