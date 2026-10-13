# Independent follow-up: measured cache regime and first iterations

Reviewer: /root/review. Read-only reanalysis of saved data; no GPU runs, builds, configuration changes or production edits. REVIEW.md remains unchanged.

## Verdict

The saved 1.60065% paired improvement measures repeated, warm-query execution in the fixed SF50 configuration using the isolated consumer binaries. It is not a cold-query result. Summing the first iteration of every query gives a separate 1.44226% candidate reduction, but that aggregate mixes different cache histories and must not be called a uniformly cold suite. The only first query in each fresh process, Q1, shows no positive reduction in its four-run point medians.

Subsequent independent ELF/link review identified a measurement-build artifact: the isolated binaries localize RMM's per-device resource registry, whereas the normal current consumer exports it for sharing with libcudf. These saved numbers therefore establish value only within that modified resource regime. They cannot currently establish normal Sirius application value or reproduce the historical 13.9% claim. The numerical recomputation and cache-regime classification below remain valid; the consumer-equivalence inference is withdrawn pending equivalent linking and new matched measurements.

## Link-visibility limitation discovered during this follow-up

The actual normal current extension at /home/kkristensen/Code/sirius/build/release/extension/sirius/sirius.duckdb_extension exports GNU-unique dynamic symbols for RMM get_ref_map()::device_id_to_resource (0x77ebc50), guard (0x77ebc48) and ref_map_lock (0x77ebc80). The older normal extension under /tmp/sirius-pr1834-streams-20261007-1940/build does likewise. Both isolated timing arms have no such dynamic exports; the candidate's map is a LOCAL b symbol at 0x5750de8. Installed libcudf separately exports its GNU-unique map at 0x52946f8.

The preserved original link command in sirius-build/build-metadata.json uses 323 direct .o inputs and --exclude-libs,ALL. The isolated command replaces those inputs with one direct wrapper plus whole-archive libsirius_core.a while retaining that flag. Thus byte-preserved object members undergo different export treatment. RMM per_device_resource.hpp lines41–49/72 explicitly require default visibility to share one registry across DSOs, and its getter lines110–120 initializes an unregistered device with the direct cuda_memory_resource. Sirius's setter is inline and operates on the extension registry. Library-side default allocations can consequently use a different resource from the extension's configured pool. ELF identity does not alone prove every runtime allocation's caller or live registry content, but the visibility divergence is a real consumer-equivalence defect.

The cheapest remedy must restore original direct-object symbol/link behavior while substituting only the required allocator and owning-context objects, then verify shared registry visibility and run new comparisons. Broad unhide-all-linker changes would alter more than the property requiring repair. None of that repair or GPU work was performed by this reviewer. The core allocator correctness and independent standalone results are unaffected by this consumer-build artifact.

## Source-backed cache and timer behavior

Re-read the fixed consumer CONTRACT.md, config.yaml, MANIFEST.json, validated_runner.py and canonical /home/kkristensen/Code/sirius/test/tpch_performance/performance_test.py before this analysis. The required reviewer/book/guidance readings from the initial review remain complete.

- Every accepted command omits --profile and --mode. Canonical DEFAULT_PROFILE (lines 140–150) uses grouped ordering with both drop_os_cache_between and reset_cache_between false. The copied configuration retains Sirius raw-chunk LRU caching, disabled prefetch and --pin none. No materialized pin does not disable the raw-chunk cache or this PR's pinned-allocation cache.
- Canonical main calls drop_os_cache once before opening the query connection (lines 2200–2201). The implementation (lines 669–699) calls os.sync(), then posix_fadvise(POSIX_FADV_DONTNEED) on each dataset file. This is advisory, file-scoped eviction; it does not flush dentries/inodes, unrelated files or device/controller caches, and filesystems may ignore the hint.
- All eight saved ab-*.log files show exactly one “Dropping OS page cache”, one “posix_fadvise(DONTNEED) applied to 10 file(s)”, one grouped-run announcement and zero Sirius cache reset announcements. There are no reported fadvise warnings. Successful advice is evidence the calls completed, not proof every later read went to physical storage.
- run_grouped (lines 929–1010) opens one connection and executes Q1 four times, then Q2 four times, through Q22. _flush_between_runs (lines 915–926) is invoked but its two false flags perform no eviction or reset. There is no reset at query boundaries either.
- open_connection (lines 756–815) loads the extension and registers all Parquet views before timing. Binding those views may read Parquet metadata after the initial eviction. time_query (lines 819–840) times execute(query).fetchall(); extension loading, connection setup, configuration and view registration are excluded.
- The custom pinned resource is created/installed by SiriusContext (current source lines 1032–1068), and a process does not destroy that owner between queries. Candidate allocation reuse can therefore benefit a later query's first iteration using allocations released by earlier queries. Shared-table raw chunks can likewise survive earlier queries; their exact residency was not measured here.
- The local managed backend intends O_DIRECT under cache.mode=sirius: io/cache/config.hpp lines128–130 select direct I/O for every mode except os, and scan_manager/config.hpp lines270–274 apply it. Therefore ordinary managed raw-input reads should not be explained as OS-page-cache hits without evidence of buffered fallback. Initial advisory file eviction still does not flush physical device/controller caches, and direct I/O does not erase Sirius raw-chunk or allocator cache history. The later cache.mode=none follow-up also intends O_DIRECT and is a separate configuration experiment.
- The separate canonical --profile cold (lines 60–73) does per-query advisory file eviction plus CALL reset_sirius_cache() in one connection. It was not selected. Even that mode preserves the process/GPU context and is a cold-data-cache mode, not full system startup or physical-media coldness.

Appropriate names: iterations 1–3 are repeated-query/hot measurements; iteration 0 is the first execution of that query within an already initialized process, with history from earlier query groups. Q1 iteration 0 is the first timed query in a fresh process after extension/config/catalog setup and advisory dataset eviction. None of these labels guarantees physical-disk coldness.

## Independent raw-CSV calculation

Inputs: every accepted MANIFEST.json run's benchmark_directory/csv/runtimes.csv. Checked 88 finite positive Sirius rows per process, exactly Q1–Q22 × iterations 0–3, no duplicate query/iteration. Recomputed the original hot FOM from raw rows as a cross-check; it matches all manifest values.

For each process, first_sum = sum(runtime_s for iteration == 0 and query != TOTAL). Four processes per arm are the independent comparison units, not the 88 individual query runs. The stored TOTAL rows are excluded and only used as a rounding cross-check (raw CSV query values have six decimals).

| Run | Arm | Sum of 22 first iterations, ms | Q1 first iteration, ms | Hot FOM, ms |
|---|---|---:|---:|---:|
| ab-0 | B | 14660.481 | 1352.782 | 13576.664 |
| ab-1 | C | 14548.759 | 1355.918 | 13366.717 |
| ab-2 | C | 14457.794 | 1343.580 | 13391.483 |
| ab-3 | B | 14672.407 | 1338.826 | 13578.534 |
| ab-4 | C | 14497.931 | 1336.451 | 13353.822 |
| ab-5 | B | 14707.046 | 1331.650 | 13525.791 |
| ab-6 | B | 14710.958 | 1330.155 | 13581.552 |
| ab-7 | C | 14448.078 | 1304.678 | 13287.594 |

| Statistic | Baseline B | Candidate C |
|---|---:|---:|
| Median first-iteration sum | 14,689.7265 ms | 14,477.8625 ms |
| Mean first-iteration sum | 14,687.7230 ms | 14,488.1405 ms |
| First-iteration sum range | 50.477 ms | 100.681 ms |
| First-iteration sum median absolute deviation | 19.2755 ms | 24.9265 ms |
| Median Q1 first iteration | 1,335.2380 ms | 1,340.0155 ms |
| Median original hot FOM | 13,577.5990 ms | 13,360.2695 ms |

Candidate reduction of the first-iteration sum: (1 − 14477.8625 / 14689.7265) × 100 = **1.4422596636%**, or 211.864 ms. The four adjacent, order-balanced pairs independently favor the candidate:

| Adjacent runs | First-sum reduction |
|---|---:|
| ab-0 / ab-1 | 0.762062% |
| ab-2 / ab-3 | 1.462698% |
| ab-4 / ab-5 | 1.421869% |
| ab-6 / ab-7 | 1.786967% |

The three accepted baseline-only first-iteration sums are 14,472.248, 14,622.280 and 14,609.070 ms: median 14,609.070 ms, range 150.032 ms (**1.0269784456%**) and median absolute deviation 13.210 ms. This is the relevant observed first-iteration variability; the previously reported 0.5255% range belongs to the hot FOM and must not be reused as a cold/first-iteration noise bound. A three-run observed range is not a formal confidence interval, and one first-iteration pair's 0.762% difference is smaller than that range.

Q1 first-iteration point medians favor baseline by 4.7775 ms (candidate +0.35780%). With only four measurements per arm and no Q1-specific preregistered uncertainty bound, this is **no demonstrated first-query benefit**, not a proven first-query regression. Q8's first-iteration point median also favors baseline; aggregate improvement does not imply every query improves.

The original hot reduction independently remains **1.6006475077%**, measured with the separately specified hot FOM. The first-iteration analysis is a post hoc diagnostic and does not change that contract.

## Per-query first-iteration point medians

These are descriptive; no per-query significance claim is made.

| Query | Baseline median, ms | Candidate median, ms |
|---|---:|---:|
| Q1 | 1335.2380 | 1340.0155 |
| Q2 | 255.5555 | 245.9240 |
| Q3 | 835.8750 | 817.8070 |
| Q4 | 434.0320 | 427.9675 |
| Q5 | 648.6450 | 629.9280 |
| Q6 | 393.8830 | 382.1920 |
| Q7 | 666.0965 | 659.8545 |
| Q8 | 690.3205 | 716.8560 |
| Q9 | 1278.6405 | 1255.5770 |
| Q10 | 874.5180 | 864.0590 |
| Q11 | 299.1825 | 292.7595 |
| Q12 | 575.5715 | 571.5515 |
| Q13 | 624.9800 | 619.8785 |
| Q14 | 435.3385 | 433.7220 |
| Q15 | 478.5725 | 465.1365 |
| Q16 | 208.1395 | 207.2295 |
| Q17 | 559.2170 | 549.2775 |
| Q18 | 1418.2915 | 1394.2120 |
| Q19 | 796.6830 | 780.9125 |
| Q20 | 533.6770 | 522.4235 |
| Q21 | 1165.9380 | 1149.9695 |
| Q22 | 181.3375 | 177.0630 |

## Inference limits and smallest useful next test

The matched identical configuration rules out a configuration difference between the two measured arms. It does not fix the shared link-visibility artifact, prove that the configuration is representative of every deployment, or prove that cache size, batch size and thread counts have no interaction with the allocator. Cache removal in the separate Q18 trace establishes the runtime mechanism in the measured artifact; it does not turn the hot measurement into a cold-data measurement or establish equivalent normal-consumer behavior.

If cold-data application value is required, run an explicitly contracted matched cold-data experiment with dataset-file eviction plus Sirius raw-chunk reset before each query, preserve identical allocator lifetime policy in both arms, and verify actual disk-read behavior if physical-storage coldness is part of the claim. Do not change configuration or reset the PR's allocation cache only in one arm. If true process-start value is required, test a fresh process per selected query and include or exclude setup explicitly. If value across configurations is required, choose a small, prespecified set of representative configurations and compare matched arms under each. Existing saved first iterations cannot substitute for those experiments.

No additional experiment was run or proposed as already authorized by this reviewer. This report is limited to independent saved-data recomputation and cache-regime classification.
