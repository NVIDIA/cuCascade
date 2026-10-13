# Corrected Q18 CUDA trace follow-up

The corrected consumer traces confirm the intended mechanism: the candidate avoids all 186 hot-Q18 `cudaHostAlloc` and 186 `cudaFreeHost` calls while preserving the same kernel count and transfer bytes. Ordinary device `cudaMalloc`/`cudaFree` calls are zero in both corrected arms. The previous 1,741 raw device pairs were caused by our measurement relink isolating RMM's registry; that device-allocation optimization lead and the earlier consumer timing inference are withdrawn.

Separate controlled, unprofiled full-22-query SF50 measurements now show 6.14% lower hot-suite time with Sirius input caching and 3.37% with `input-cache=none`. These are input-cache policies; the candidate pinned-allocation cache is enabled in both candidate regimes. `CORRECTED-RESULTS.json` and `CORRECTED-INDEPENDENT-RESULTS.json` own those conclusions. Their figure of merit is the median of four process sums of 22 per-query hot medians: 7,611.4355 -> 7,144.1595 ms and 9,128.4350 -> 8,821.2210 ms respectively. All results and rendered SQL were independently checked. These are warmed measurements, not uniformly cold-system claims, and they do not reproduce the old PR's 13.9% claim on this machine.

This report classifies remaining source paths and trace evidence. The developer performed only read-only analysis and the separately authorized compiler-only artifact repair; root owns all GPU execution. No production source changes were made for this follow-up.

## Corrected scope and reproducibility

Inputs are `corrected-input-sirius/profile-B-q18.sqlite` and `profile-C-q18.sqlite`, under `/tmp/cucascade-pinned-pr-20261012`. These are the corrected direct-object consumer traces. Exactly one hot Q18, iteration 1 after iteration 0, was captured in each process under `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop`. Root independently parity-gated the loaded artifacts and query validation.

The corrected capture records `cuProfilerStart` before the query, but the stop API ends collection and is not recorded. We therefore use the single complete `sirius::query` NVTX interval as the bound, rather than inventing an absent stop marker or using a whole-process export. All included runtime/activity intervals are fully contained within the query; no runtime API crosses its boundaries. `corrected_trace_summary.py` generates `CORRECTED-NSYS-SUMMARY.json`. The developer's bounded independent read matched every API count and saved additional source-context, copy aggregation, and overlap evidence in `FOLLOWUP-CORRECTED-DETAILS.json`.

| Scope | B | C |
|---|---:|---:|
| Query start, ns | 17,474,960 | 17,576,064 |
| Query end, ns | 810,717,840 | 775,898,240 |
| Profiled Q18 window, ms | 793.242880 | 758.322176 |
| Kernel activity records | 1,608 | 1,608 |
| Runtime API records | 17,660 | 17,945 |
| Sum of runtime API inclusive durations, ms | 3,942.354192 | 3,578.881264 |

All CUDA API `callchainId` values remain null. A smallest enclosing NVTX range on the same CPU thread provides function context, not an exact caller stack, resource pointer, or critical-path attribution. Allocation-size arguments and pointers are absent from the runtime schema, no allocation-activity table is exported, and every copy `virtualAddress` is null. Exact allocation sizes, retained pinned bytes, and bucket distribution cannot be recovered from these exports; transfer bytes are not allocation sizes.

## All observed allocation, synchronization, and event API types

Durations are inclusive CPU intervals summed across threads. They are neither exclusive work nor removable query time. The table omits API version suffixes; the JSON retains them.

| API | B count | C count | B inclusive ms | C inclusive ms |
|---|---:|---:|---:|---:|
| `cudaMalloc` | 0 | 0 | 0.000000 | 0.000000 |
| `cudaFree` | 0 | 0 | 0.000000 | 0.000000 |
| `cudaMallocFromPoolAsync` | 2,308 | 2,308 | 11.454160 | 11.346800 |
| `cudaFreeAsync` | 2,306 | 2,306 | 6.273040 | 4.327968 |
| `cudaHostAlloc` | 186 | 0 | 200.450368 | 0.000000 |
| `cudaFreeHost` | 186 | 0 | 296.642816 | 0.000000 |
| `cudaStreamSynchronize` | 1,955 | 1,955 | 1467.483520 | 2055.555856 |
| `cuStreamSynchronize` | 287 | 287 | 1241.799488 | 1460.143184 |
| `cudaEventSynchronize` | 6 | 7 | 0.156512 | 0.125792 |
| `cudaStreamWaitEvent` | 901 | 1,087 | 1.760240 | 1.819872 |
| `cudaEventQuery` | 252 | 350 | 0.257216 | 0.403280 |
| `cudaEventCreateWithFlags` | 107 | 107 | 62.137088 | 0.213040 |
| `cudaEventRecord` | 1,054 | 1,240 | 149.357424 | 2.680576 |
| `cudaEventDestroy` | 107 | 107 | 68.801072 | 0.548336 |
| `cudaStreamAddCallback` | 1,250 | 1,250 | 10.981600 | 6.268592 |
| `cudaStreamGetDevice` | 798 | 984 | 0.253152 | 0.270016 |

There are no observed `cudaDeviceSynchronize`, `cuCtxSynchronize`, `cudaMallocAsync`, `cudaMallocManaged`, `cudaMallocHost`, `cudaHostRegister`, `cudaHostUnregister`, or `cudaStreamQuery` records inside either corrected query bound. The 287 direct driver stream synchronizations do not nest inside the runtime stream-synchronize intervals on the same thread; they are separate records.

The candidate adds exactly 186 event records, waits, and stream-device lookups while removing exactly 186 host allocation/free pairs. Event create/destroy counts remain 107/107. Baseline host allocation calls are enclosed by `decompress_page_data` (92), `read_parquet` (69), and `sort_pages` (25); the candidate has none. This provides direct context for the cache's effect on cuDF Parquet host scratch allocations. It does not reveal their exact sizes.

The only nonzero API status is expected `cudaEventQuery` status 600 (`cudaErrorNotReady`): B has 160 pending and 92 successful queries, C has 259 pending and 91 successful queries. Pending status is ordinary completion polling, not a failed result. `cucascade/src/cuda/event.cpp:103` maps it to in-progress, `cucascade/src/data/data_batch.cpp:146` preserves pending reader events, and `src/io/uring/uring_reactor.cpp:779` retains copying slots. These source contracts do not identify each trace caller. Event query counts now increase in C, so the earlier artifact's reduction in polling is not retained as an optimization claim. The additional single event synchronization is likewise unassigned without caller evidence.

## Source-backed synchronization classification

Current `docs/super-sirius/memory-management.md:194` states that no device-memory owner may die, return storage to a pool, or rebind its free stream until every potential reader/writer stream is ordered before the free. Host locks alone do not establish that device work is complete.

- `src/pipeline/gpu_pipeline_task.cpp:306`: the normal `run_one_operator` call synchronizes after `execute`. The restored input and output owners remain alive through this call; the restored input dies when the helper returns, and `compute_task` replaces the previous operator data at lines 478-479. This is an existing lifetime boundary. Removing it requires another ownership/completion guarantee.
- `src/pipeline/gpu_pipeline_task.cpp:851`: the normal final task synchronization protects outputs and restored materialization declared at lines 759-760 through sink publication and subsequent finalization. Exception draining at lines 222-232 and its catch sites runs while those owners remain alive. Failure only logs; it cannot prove successful quiescence.
- `src/pipeline/gpu_pipeline_task.cpp:691`: preparation synchronization is explicitly justified at lines 689-690 by accurate Quent/log preparation timing. These sources do not establish that this particular barrier is required for lifetime, or that removing it would be safe across every auxiliary-stream preparation path. It needs its own audit before any change.
- The installed CCCL `include/rapids/cuda/__stream/stream_ref.h:171` implements `stream_ref::sync()` through the driver stream-synchronize operation. This explains why examining only runtime `cudaStreamSynchronize` misses relevant pipeline synchronization.
- Installed cuDF `include/cudf/detail/utilities/host_vector.hpp:143` calls `mr.allocate(stream, ...)`, then synchronizes the allocation stream before CPU initialization. A cached host allocation remains stream-ordered; its wait alone does not make it immediately CPU-writable. This cuDF synchronization remains meaningful after host allocation calls disappear.
- Installed cuDF `vector_factories.hpp:462` synchronizes after copying device data into a host vector. Synchronous device-vector factories also synchronize, for example at line 202. Thrust `include/rapids/thrust/system/cuda/detail/util.h:119` and line 134 copy scalars/data to or from the host and synchronize before the result/source lifetime is used. Such calls can be host-observation boundaries.
- `src/exec/cuda_event_completion_poll.hpp:388` publishes completed frontiers through `cudaStreamAddCallback`, not `cudaEventQuery`. Ordinary drain/wait paths poll that host state without stream synchronization. Callback-install failure drains at line 399; explicit stopped-producer lifecycle quiescence drains at line 302. Neither establishes that all 1,250 callback records or all event-query records came from this header. In the saved trace, all callback records are instead inside the cuDF `fetch_byte_ranges_to_device_async` NVTX range.
- The candidate allocator inserts checked stream waits before handing out cached slabs/large buffers (`small_pinned_host_memory_resource.cpp:182` and line 324). These establish stream reuse ordering. Event synchronization before releasing tracked storage, and stream synchronization after failed event publication, protect outstanding DMA. They are correctness operations, not automatically redundant overhead.

Most of the 1,955 runtime stream synchronizations are enclosed by cuDF ranges in both arms: `read_parquet` 341, `detect_malformed_pages` 275, `decode_page_headers` 225, `decode_page_data` 211, `decompress_page_data` 175, `aggregate` 127, `preprocess_subpass_pages` 106, `build_string_dict_indices` 75, and `minmax` 66. That distribution makes a blanket explanation of all synchronizations as Sirius task timing inaccurate. Installed headers show several valid CPU-readiness mechanisms, but the shipped cuDF implementation source and API call stacks are not present in these captures; each individual barrier is not proven necessary or removable here.


## Device allocation registration after repair

Sirius's device resource setup is explicit. `src/memory/sirius_memory_reservation_manager.cpp:41` obtains each GPU space's default allocator and selects its device; line 46 registers it through `cudf::set_current_device_resource`. The consumer's default upstream in `cucascade/src/memory/memory_space.cpp:127` is `rmm::mr::cuda_async_memory_resource`, wrapped by the reservation adaptor. Installed cuDF `utilities/memory_resource.hpp:118` forwards registration to RMM, and its getter at line 34 forwards to the same intended registry.

Both corrected arms now have 2,308 `cudaMallocFromPoolAsync` allocations and 2,306 `cudaFreeAsync` calls in the query bound, and zero ordinary `cudaMalloc`/`cudaFree`. Allocations include the former raw-call contexts: `detect_malformed_pages` 200, `aggregate` 197, `decode_page_headers` 175, `decompress_page_data` 154, `device_decompress` 150, plus `allocate_like` 121 and `allocate_columns` 116. Sirius passes explicit resources to device buffers and cuDF, for example `src/op/scan/parquet_materialize.cpp:148` and line 262. Library defaults can now see the shared configured registry, so ordinary output and intermediate allocations use the pooled API in these captures. The two-count allocation/free difference in a query bound does not establish a leak: owners may span the query boundary and deallocate later.

The older artifact's 1,741 raw pairs and 567 async allocations were not evidence of ordinary CUDA async-pool growth or a demonstrated production resource bug. The old relink changed direct Sirius `.o` inputs into a whole archive while preserving `--exclude-libs,ALL`, hiding RMM map/guard/lock symbols. Installed RMM `mr/per_device_resource.hpp:41` explicitly requires default visibility to share those static references across DSOs; its getter at line 110 seeds an unregistered device with a direct `cuda_memory_resource`. This explains why library defaults could stay uncached while explicit output allocations remained pooled in that artifact.

The corrected build directly links all 334 preserved Sirius objects, including distinct duplicate-name protobuf members and the existing CUDA device-link object, while retaining original flags for other archives. Independent review verified 333 unchanged Sirius members, 21 unchanged cuCascade members, and GNU-unique/default RMM exports matching both preserved normal consumer extensions. A loader-only check also resolved map/guard/lock to the same addresses from corrected extensions and `libcudf`. Root then validated results and timings with these corrected artifacts. See `sirius-build-corrected/build-metadata.json` and `CORRECTED-INDEPENDENT-RESULTS.json`.

The earlier report is archived as `FOLLOWUP-NSYS-original-link.md`; its original JSON/trace files remain diagnostic records of that rejected integration regime. They are excluded from corrected timing inference and are not pooled with corrected results. A separate device-temporary cache, a default-resource fix in production, and a payoff derived from 1,741 raw pairs are withdrawn as leads for this PR.

## Copies: bulk traffic is already pinned or device-to-device

Memory/copy kind labels come from each export's `ENUM_CUDA_MEM_KIND` and `ENUM_CUDA_MEMCPY_OPER`.

| Direction and memory kinds | B activity records | C activity records | Bytes, both arms | `SUM(copyCount)`, both arms |
|---|---:|---:|---:|---:|
| Pageable host -> device | 44 | 44 | 934 | 44 |
| Pinned host -> device | 1,595 | 1,596 | 1,316,896,253 | 2,740 |
| Device -> pageable host | 65 | 65 | 16,916 | 65 |
| Device -> pinned host | 325 | 325 | 9,217 | 333 |
| Device -> device | 202 | 202 | 1,801,246,399 | 217 |
| Pinned host -> pinned host | 150 | 150 | 48,325,056 | 150 |
| Total | 2,381 | 2,382 | 3,166,494,775 | 3,549 |

The one-record count difference is entirely pinned H2D; its bytes and `SUM(copyCount)` remain identical. Activity aggregation changed, not transferred volume. These traces show only 17,850 pageable bytes, so a claim of pageable bulk DMA dominating Q18 is unsupported. Pageable metadata/scalar observation can still introduce latency despite low byte volume; its payoff is unpriced here.

Both arms issue 2,130 `cudaMemcpyBatchAsync`, 251 `cudaMemcpyAsync`, and 296 `cudaMemsetAsync` calls. API counts, activity records, and batch copy counts measure different things. Source pathways include cuDF Parquet materialization at `src/op/scan/parquet_materialize.cpp:262`, explicit-resource buffers at line 148, and Sirius batched native uploads at `src/op/scan/duckdb_native_decoder.cpp:684`. GPU/host spill conversion batches copies at `src/data/spill_chunked_converters.cpp:230` and preserves source lifetime through completion as documented at line 256. The observed Parquet ranges establish the scan context, but do not prove every alternative pathway was exercised. Removing host allocation churn does not remove the data movement or its lifetime obligations.

## Inclusive timing, overlap, and unpriced work

The corrected B host allocation/free inclusive sum is 497.093184 ms across eight CPU threads. Its interval union is 251.847952 ms, and 227.575584 ms overlaps active kernels or copies. Neither the inclusive sum, union, nor union minus GPU overlap is a removable-time estimate: other CPU threads, dependencies, runtime serialization, and device work can hide or relocate waiting. The candidate's host API sum is zero, yet this single profiled query's wall time falls by 34.920704 ms, not 497 ms.

The runtime-plus-driver stream synchronization count stays unchanged. Their combined inclusive durations rise from 2,709.283008 to 3,515.699040 ms even though candidate query wall time falls. This is consistent with waiting moving from earlier blocking allocation/launch/copy calls to explicit synchronization and with concurrent threads waiting for the same work; it does not demonstrate added serialized computation or removable synchronization work. The GPU kernel/copy interval unions are 723.819744 and 718.545664 ms within the respective 793.242880 and 758.322176 ms query windows. These are timeline observations, not a new optimization price.

Recorded profiler overhead within the query totals 21.182672 ms for B and 20.129392 ms for C: TLS allocation, unknown overhead, and CUDA profiling memory allocation. These records need not be disjoint from API intervals and do not include all CUPTI callback/buffer or OS-runtime tracing costs. They cannot be subtracted to manufacture an unprofiled result. No paired tracing-on/off experiment quantifies profiler inflation here. Controlled unprofiled repeated suite measurements own the measured 6.14%/3.37% gains; one captured Q18 per arm only supports mechanism attribution.

Remaining work with no demonstrated payoff includes the preparation-timing barrier, cuDF CPU-observation synchronizations, callback/event polling, pooled device temporary allocation overhead, and tiny pageable metadata transfers. Kernel count and transfer bytes are unchanged, so the trace also supplies no evidence of reduced computational or transfer work. Any further change requires a narrow ownership/completion audit and controlled measurements; normal per-operator and final-task synchronization, cached-host reuse ordering, and DMA/source lifetime guarantees must be preserved.
