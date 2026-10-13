# Corrected performance and Nsight follow-up — 2026-10-13

**The cache has a measured application benefit: 6.14% with Sirius input caching and 3.37% with input caching disabled, on this fixed GB10 configuration.** The earlier 1.60% consumer result is withdrawn because the isolated relink changed allocator sharing. These corrected results support retaining the cache; its value depends on workload and configuration. The original historical 13.9% claim remains unreproduced.

## Cold or hot?

The original reported comparison used **warm grouped repetitions**, not a uniformly cold suite. Each process executed each query four times, excluded iteration 0, and used the median of iterations 1–3. One connection served all query groups; input cache, footer metadata, CUDA context, and allocator state could survive between groups. Initial file-specific `fadvise` happened once. The explicit `--pin none` setting did not disable Sirius's separate cache of undecoded input.

The corrected experiment preserves that repetition protocol in two input policies. Mode `sirius` retains undecoded input with LRU eviction; mode `none` disables that cache policy and its derived automatic backend readahead. Compatible reads use `O_DIRECT`. The latter is a repeated uncached-input workload: metadata, CUDA state and allocation pools remain warm, and physical-media coldness is unverified. Neither result is a fully cold system price. First-per-query sums are descriptive checks, with shared preceding-query history; they are not a substitute for a prespecified cold benchmark.

## The actual build artifact

The first isolated build replaced normal direct Sirius object inputs with a whole archive but retained `--exclude-libs,ALL`. The archive exclusion localized RMM's device-resource map, guard and lock. Sirius's setter therefore could initialize a different registry from libcudf's default getter, so library temporaries could use raw CUDA allocation while explicitly passed output resources stayed pooled.

Both preserved normal extensions export those variables with GNU-unique/default visibility. The original isolated arms did not. Identical B/C sources and settings were insufficient to preserve normal consumer behavior: object-content equality does not establish linker/DSO equivalence. Thus the old 1.60% figure is only a paired result in that altered regime, and the old raw-device-malloc finding is not a normal-production optimization opportunity.

The repaired builds restore **all 334 core objects as direct link inputs**, retaining the original archive flags elsewhere. They preserve the duplicate protobuf objects and CUDA device-link object. Only the rebuilt allocator and its constructing SiriusContext owner differ between arms; all 333 other core objects and 21 other resource archive members are byte-identical. The reviewer independently compared content, link arguments, dynamic ELF visibility, and loader behavior. Each repaired extension resolves map, guard and lock to the same address as libcudf in separate GPU-hidden loader tests. GPU correctness and the new trace then validate the intended behavior.

This retains the documented source-equivalent frozen current-consumer build scope; it is not a fresh rebuild of every engine/generated/support component. [Build provenance](sirius-build-corrected/README.md) and [independent gate](sirius-build-corrected/INDEPENDENT-GATE.json) preserve the exact scope.

## Corrected matched measurements

| Input-cache policy | Main allocator | PR cache | Reduction | Baseline range |
|---|---:|---:|---:|---:|
| Sirius LRU | 7.611436 s | 7.144160 s | **6.139% / 467.276 ms** | 0.140% |
| Disabled (`none`) | 9.128435 s | 8.821221 s | **3.365% / 307.214 ms** | 0.750% |

For each policy, the fixed schedule was three baseline-only noise processes then **B,C,C,B,C,B,B,C**. Each process summed the 22 per-query medians of its three retained repetitions; the table reports the median of four process totals per arm. The paired schedules were interleaved between policies after both noise series. Every one of the four pairs improved in each policy: 5.995–6.447% with Sirius caching and 2.838–3.920% without. All candidate process totals were below all paired baseline totals in each policy.

All **1,936 query executions** across the 22 noise/timing processes matched CPU results, with identical per-query row counts/digests. Each process's combined logs show all 88 GPU starts, completions and engine entries without query errors; fallback was explicitly disabled. Four preliminary correctness smokes and two separate profiling processes also validated every result and were excluded from the timing verdict. Canonical sources, configs, validation adapter, binaries and production source identities remained frozen. The architect and reviewer independently recomputed the raw results.

Descriptive first-per-query sums improved 5.356% and 3.140% respectively, but share warm process history. They establish no uniformly cold-suite percentage.

The configurations differ only in `scan_manager.cache.mode`. Other settings remain GPU 48 GB, host 24 GB, pipeline/creator/scan threads 8/4/6, 512 MiB batches, explicit table pinning disabled, host-to-GPU memory prefetcher disabled, and Quent disabled. This is an explicit GB10 experiment configuration derived from a scan preset, not a universal production default. Cache mode affects retention, staging and derived readahead together, so the 2.774 percentage-point interaction cannot be attributed solely to LRU retention.

[Results](CORRECTED-RESULTS.json), [independent audit](CORRECTED-INDEPENDENT-RESULTS.json), [cached-input contract](corrected-input-sirius/CONTRACT.md), [uncached-input contract](corrected-input-none/CONTRACT.md), and [raw evidence](raw-evidence.tar.gz) preserve commands and all accepted records. The original private-registry sensitivity attempt was stopped/excluded; no samples from it enter these corrected prices.

## Which Nsight findings remain?

Both new traces capture the first hot Q18 iteration after the same preceding query groups. Profiling now actually stops collection: each export contains exactly one `sirius::query` NVTX range and the start marker; the stop callback is excluded. Counts below use that query range. All 88 queries in each profiling process still match CPU results; these durations do not enter the performance verdict.

| Hot Q18 observation | Main | PR | Interpretation |
|---|---:|---:|---|
| Pinned host allocations / frees | 186 / 186 | 0 / 0 | Cache mechanism confirmed |
| Raw device `cudaMalloc` / `cudaFree` | 0 / 0 | 0 / 0 | The old 1,741 pairs were an artifact-regime finding |
| Pooled async device allocations | 2,308 | 2,308 | Normal device pooling is active |
| Runtime stream synchronizations | 1,955 | 1,955 | Remain; necessity/removable price must be assessed per path |
| Driver stream synchronizations | 287 | 287 | Remain; same-thread check distinguishes these from nested runtime calls |
| Stream event waits | 901 | 1,087 | 186 additional reuse dependencies |
| GPU kernels | 1,608 | 1,608 | Logical work count retained |
| Copied bytes | 3,166,494,775 | 3,166,494,775 | Copy volume retained |

Pooled async frees count 2,306 inside the query interval; result cleanup outside the interval is excluded. Copy activity records differ slightly (2,381/2,382), while represented copy count sums and every copy group's bytes match. Transfer volume is dominated by **1.317 GB pinned H2D** and **1.801 GB D2D**. Pageable H2D is only 934 bytes and pageable D2H 16,916 bytes, so these traces do not identify a bulk pageable-transfer problem. Warm raw-input caching still repeats decoding, staging and GPU operations.

The remaining sync/copy counts are useful investigation leads. Source review identifies CPU-readiness waits in cuDF host-vector allocation and scalar retrieval, plus Sirius ownership/publication boundaries. Some preparation waits also support timing attribution, but their safe removal and end-to-end gain have not been measured. The trace cannot justify dropping barriers or copies by count alone, and there are no API stacks or allocation-size arguments to assign every call conclusively.

Inclusive API durations are not removable wall time. In the baseline Q18 window, host allocation/free calls sum to 497.093 ms across eight threads but occupy a 251.848 ms interval union; 227.576 ms of that union overlaps active GPU work. Total API duration is 3.942 s within a 0.793 s profiled query. Overlap, waiting and profiler overhead make additive speedup estimates invalid. The separately measured 6.14%/3.37% suite changes are the prices for this cache; other proposals still need controlled A/B validation.

[Scoped trace summary](CORRECTED-NSYS-SUMMARY.json), [independent profile audit](CORRECTED-INDEPENDENT-PROFILE.json), [bounded analyzer](corrected_trace_summary.py), and [source interpretation](FOLLOWUP-NSYS.md) preserve evidence and limits. Native reports and SQLite files are retained locally under `build/pr-reevaluation/evidence/corrected/`.

## Value and limits

The previous rejection case rested on an invalid application build. Corrected evidence supports a real, configuration-dependent benefit in both tested raw-Parquet policies. It supports retaining the bounded cache for those workloads; it does not prove the same benefit for fully cold systems, explicitly pinned decoded tables, other configurations/scales, or multiple GPUs. The original exact historical configuration and trace provenance are missing, so attributing its 13.9% to a specific configuration artifact would be speculation.

Core implementation and its prior 442-case correctness validation, focused sanitizer/failure checks, and standalone measurements are unchanged. No new production optimization was implemented in this follow-up. The preexisting external NUMA synchronous-wrapper defect remains documented; its earlier observer showed no synchronous calls in that diagnostic, not a general guarantee for every workload.

Plain-text evidence copies normalize extra final blank lines. Companion `.gz` files preserve byte-exact originals for the affected reading report/audit/loader scripts; recorded original source hashes refer to those original bytes.
