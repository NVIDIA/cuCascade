# TPC-H I/O patterns: what a scan reads, how big, how often, and what that means for the disk reactors

Synthesis of three evidence sources about how a column-pruning GPU engine reads TPC-H parquet from local NVMe. Use it to size requests, choose thresholds, and decide where I/O effort pays off.

| Source | What it gives | Detail document |
|---|---|---|
| Footers of the local SF10/SF100 files (DuckDB-written) | exact column-chunk offsets and sizes; simulated per-query scan | [tpch-parquet-layout-sf10-sf100.md](tpch-parquet-layout-sf10-sf100.md) |
| Sirius SF1000 range trace from the GB300 host (22 queries) | the real logical request stream: file, offset, size | [tpch-sf1000-gb300-trace.md](tpch-sf1000-gb300-trace.md) |
| GB300 host measurements (fio, gpuio, CTRACK, Sirius runs) | raw hardware ceiling, scan time split, current throughput | section 6 below |

Evidence quality: the local-layout numbers are measured from the files but the per-query scans are a model (column sets from the TPC-H SQL, perfect pruning). The SF1000 trace is real engine output but was summarised remotely and **not re-run by us**. The GB300 measurements were read from existing result files, not reproduced. None of this includes timing per request.

## 1. Key facts

1. **The suite is I/O-bound.** Scan I/O is 57-96% of CTRACK-tracked time (at least 79% in 17 of 22 queries). The suite requests about 950 GB; the 3-drive md RAID0 on the GB300 host delivers at most ~20.8 GB/s with O_DIRECT, so the I/O floor is ~45 s for those bytes. Sirius with one uring reactor measures 51.5-52.1 s for the 22 queries, about 88% of that floor.
2. **Request size is set by the parquet row-group size, not by TPC-H.** In the SF1000 files (about 1.06M rows per row group) lineitem column-chunk requests are 166 KB-4 MB, median 1.6 MB, and 87.8% of all requested bytes are in requests of at least 1 MiB (95.9% after merging adjacent chunks). In the local DuckDB-default files (122,880 rows per row group) lineitem chunks have a 91 KiB median, and only 3.6% of bytes (whole benchmark) are in requests of at least 1 MiB raw, 31.5% after merging.
3. **Merging touching ranges is free and halves the request count.** Adjacent projected columns are contiguous in the file. On SF1000, merging with gap 0 takes 534,988 requests to 284,293 (-47%) with no extra bytes. No gap between ranges is shorter than 64 KiB, so a merge gap anywhere from 0 to 64 KiB changes nothing.
4. **Almost nothing is aligned.** Chunk offsets are 4 KiB-aligned at the chance rate (118 of 473,760 lineitem ranges on SF1000; 0.026% locally). Direct I/O needs widening to page boundaries (+0.8% bytes locally).
5. **By request count, a large fraction is still small.** On SF1000, 47% of raw requests (253,848 of 534,988) and 30% of requests after gap-0 merging (85,422 of 284,293) are below 1 MiB, even though they carry only 12% and 4% of the bytes. A fixed 1 MiB cut-off therefore sends about 30% of requests to the fallback path.
6. **The suite reads each unique byte ~5.7 times.** 948 GB requested versus ~166 GB unique across all 22 queries (lineitem 800.6 vs 105.2 GB). About 92.6 GB (11.6% of lineitem) are duplicate reads inside a single query (Q17, Q18, Q21 scan lineitem two or three times).
7. **Row-group skipping does nothing here.** Min/max statistics skip zero row groups for all 22 queries on the local files; only the generation-order key columns are clustered. Column pruning is the only byte reduction available.
8. **Concurrency, destination and latency are unknown.** The trace has no timestamps, threads, streams or host/device flag.

## 2. Two layouts, two regimes

| | Local SF10/SF100 files | GB300 SF1000 files (trace) |
|---|---|---|
| Writer / layout | DuckDB v1.4.4, one file per table, Snappy, one page per column chunk, no page index | not inspected; lineitem split into 60 files of 94 row groups |
| Rows per row group | 122,880 | ~1.06M (derived from Q1 range count) |
| lineitem chunk size (median) | 90.8 KiB (p90 653 KiB, max 1.5 MiB) | 1.6 MB (p10 332 KB, p90 3.3 MB, max 4.0 MB) |
| Share of chunks below 1 MiB | 92.3% by count (56.6% by bytes), all tables | lineitem 48% by count; 12% of all bytes (raw) |
| Requests of at least 1 MiB, bytes (whole benchmark) | raw 3.6%, gap-0 merge 31.5%, gap-1 MiB merge 84.7% (+28.8% over-read) | raw 87.8%, gap-0 merge 95.9%, gap-1 MiB merge 97.1% |
| Gaps between chunks | exactly 0 everywhere | 0 (35% of lineitem ranges) or at least 64 KiB (median 7.5 MB) |

The GDS-versus-POSIX split is a property of the data layout. On the SF1000 layout nearly all bytes are in requests large enough for any cut-off up to a few MiB, so the threshold choice moves only a minority of bytes. On the DuckDB-default layout the threshold decides everything, and Q1's lineitem columns form runs of 1.044-1.047 MiB per row group, only about 4.4% above 1 MiB: a cut-off above ~1.05 MiB would flip all of Q1 to the fallback.

Published evidence agrees that small column chunks are the problem: a study of TPC-H lineitem scans through cuDF and GDS finds DuckDB-default chunks of about 100 KB "suboptimal" and needs row groups of about 4M rows or more for GDS to saturate one SSD (DaMoN'26, arXiv 2602.17335; see [gds-field-experience.md](gds-field-experience.md) section 6).

## 3. Request sizes and what they imply

SF1000 trace, share of **requests** (not bytes) by size, raw and after gap-0 merging (all 22 queries):

| size | raw requests | gap-0 requests |
|---|---:|---:|
| below 64 KiB | 21,307 (4.0%) | 7,916 (2.8%) |
| 64-256 KiB | 16,918 (3.2%) | 0 (0.0%) |
| 256-512 KiB | 94,020 (17.6%) | 59,038 (20.8%) |
| 512 KiB-1 MiB | 121,603 (22.7%) | 18,468 (6.5%) |
| 1-2 MiB | 105,900 (19.8%) | 42,252 (14.9%) |
| 2-4 MiB | 162,233 (30.3%) | 99,528 (35.0%) |
| 4-8 MiB | 10,612 (2.0%) | 55,188 (19.4%) |
| 8-16 MiB | 892 (0.2%) | 573 (0.2%) |
| at least 16 MiB | 1,503 (0.3%) | 1,330 (0.5%) |
| **total** | 534,988 | 284,293 |

For lineitem alone, merged requests range 0.32-7.3 MiB (median 3.2 MiB, p90 4.6 MiB). The few requests of 16 MiB or more come from the dimension tables and orders (largest 28 MiB).

Where to put thresholds (decision aid, not a measured result):

- A cut-off at **1 MiB** sends 30% of requests (4% of bytes) to the fallback path on this data.
- A cut-off at **256 KiB** would send only the 2.8% of requests below 64 KiB plus the ~0 that remain between 64 and 256 KiB, i.e. almost nothing, because lineitem has no merged requests below 256 KiB.
- NVIDIA documents no local-NVMe GDS-vs-POSIX threshold; its guidance is "64K and above" for the synchronous thread-pool API, below 1 MB bad for the stream API, and a kernel regression for IO up to 128 KB on kernels 6.11 and later ([gds-cufile-best-practices.md](gds-cufile-best-practices.md) section 3.4).

## 4. Contiguity, gaps and merge policy

- Within a row group the column chunks are laid out in schema order, back to back, with no padding. Across row groups the last chunk of row group *i* is directly followed by the first of row group *i+1* (local files: every one of ~95k chunk gaps is zero bytes).
- Projections of adjacent schema columns therefore form contiguous runs. In the SF1000 trace 35% of lineitem ranges start exactly where the previous one ended; 63% follow a forward gap whose median is 7.5 MB (a skipped column), p10 0.96 MB.
- **Merge gap policy.** Gap 0 is free. On SF1000 a 1 MiB gap removes a further 18% of requests for +4% bytes versus gap 0 (882 vs 847 GB); an 8 MiB gap over-reads 38-42% and is not worth it. Locally, a 1 MiB gap moves the GDS-eligible share from 31.5% to 84.7% but costs +28.8% bytes. Velox defaults to a 512 KiB coalesce distance; cuDF merges only exactly contiguous chunks.
- **Chaining hazard.** With a merge gap of 64 KiB or more a file-global merge can chain through a whole table (local Q10 customer scan becomes one 1.15 GiB request; Q2 supplier one 75 MiB request). Cap merged requests (16 MiB is NVIDIA's historical maximum direct-IO chunk) or merge per row group.

## 5. Alignment

- Parquet writers do not align column chunks. Only 118 of 473,760 SF1000 lineitem offsets (0.025%) and 25 of 94,914 local chunk offsets (0.026%) are 4 KiB-aligned, which is the chance rate. Sizes are aligned just as rarely.
- Widening each request to 4 KiB boundaries costs about 4 KiB per request (mean ~4,095 B added per chunk locally; +0.77% bytes over the benchmark) and does not change the request count meaningfully (250,852 vs 250,849 requests locally).
- cuFile's direct-DMA path also needs the **destination device pointer** 4 KiB-aligned (NVIDIA's definition of "unaligned" includes it). The trace says nothing about destination alignment; RMM allocations are 256-byte aligned by default. See [gds-reactor-recommendations.md](gds-reactor-recommendations.md) section 4.

## 6. How I/O-bound is it, and how close are we?

Scan I/O share per query (CTRACK `gpu_scan_operator[io]` sub-timer, SF1000, cold GPU runs; the I/O and decode sub-timers overlap, so they can sum above 100%). The CTRACK runs are separate runs whose wall times differ from the 52 s one-reactor runs used in the throughput discussion below, so only shares are used here, not wall times.

| Q | scan I/O share | requested GB | unique GB (gap 0) | requests (gap 0) | bytes in requests >= 1 MiB (gap 0) | shape |
|---|---:|---:|---:|---:|---:|---|
| 1 | 57% | 40.4 | 40.4 | 5,640 | 100.0% | lineitem 7 columns, one scan |
| 2 | 68% | 7.9 | 4.5 | 2,151 | 98.3% | small dimension tables |
| 3 | 92% | 40.8 | 40.9 | 20,648 | 95.2% | lineitem, orders, customer |
| 4 | 85% | 22.9 | 23.0 | 13,560 | 91.5% | lineitem, orders |
| 5 | 93% | 50.6 | 50.7 | 19,532 | 96.1% | six tables |
| 6 | 96% | 34.9 | 35.0 | 11,280 | 100.0% | lineitem 4 columns |
| 7 | 91% | 57.4 | 57.5 | 24,031 | 96.5% | lineitem, orders |
| 8 | 95% | 73.0 | 73.0 | 14,300 | 99.6% | seven tables |
| 9 | 88% | 76.8 | 76.9 | 15,323 | 99.9% | lineitem, orders, partsupp, part |
| 10 | 80% | 43.9 | 43.9 | 19,357 | 92.3% | customer-heavy (10.7 GB) |
| 11 | 79% | 9.6 | 4.8 | 791 | 99.9% | partsupp only, scanned twice |
| 12 | 89% | 32.0 | 32.1 | 19,200 | 85.0% | lineitem, orders |
| 13 | 91% | 33.2 | 33.2 | 1,309 | 100.0% | orders only, very large chunks |
| 14 | 95% | 52.8 | 52.9 | 17,328 | 99.6% | lineitem, part |
| 15 | 93% | 49.6 | 49.7 | 16,942 | 100.0% | lineitem 4 columns |
| 16 | 57% | 2.0 | 2.1 | 1,188 | 99.9% | tiny |
| 17 | 86% | 72.0 | 45.4 | 11,892 | 99.4% | lineitem scanned twice |
| 18 | 59% | 27.0 | 20.6 | 13,714 | 68.8% | lineitem twice, orders |
| 19 | 94% | 52.3 | 52.3 | 17,532 | 92.5% | lineitem, part |
| 20 | 92% | 60.1 | 60.2 | 17,904 | 92.4% | lineitem, partsupp, part |
| 21 | 66% | 98.7 | 39.4 | 19,223 | 94.1% | lineitem scanned three times |
| 22 | 80% | 10.4 | 8.3 | 1,448 | 100.0% | orders, customer |

(Unique GB is the per-query size after merging overlapping ranges, which removes within-query duplicates. The scan share is lowest where the engine spends time elsewhere: Q1 and Q16 in `grouped_aggregate`, Q18 in aggregation.)

Measured throughput ceilings on the GB300 host `/raid` (3 x PCIe Gen4 x4 NVMe, md RAID0, 512 KiB chunk, ext4; O_DIRECT; host destination unless noted):

| Measurement | Result |
|---|---:|
| fio, raw `/dev/md1`, sequential 512 KiB-8 MiB, QD32, 4 jobs | 20.9 GB/s |
| fio, random 1 MiB, 1 job, QD16 / QD64 / QD256 | 19.8 / 20.4 / 20.7 GB/s |
| fio, random 256 KiB, 4 jobs, QD4 / QD16 / QD64 | 19.0 / 20.5 / 20.8 GB/s |
| fio, single drive | 7.0 GB/s |
| Python `preadv` of the real parquet files, 16 MiB buffers, 1 / 4+ threads | 16.1 / 20.8-20.9 GB/s |
| gpuio (O_DIRECT to pinned host, then H2D) on the SF1000 lineitem trace, queue depth 1 / 4 / 16 per stream, host dest | 5.1 / 14.1 / 19.8-20.2 GB/s |
| same, device destination, queue depth 1 / 4 / 16 | 4.1 / 12.1 / 20.2 GB/s |
| gpuio, full 22-query trace (885 GiB), queue depth 16 | 20.4 GB/s |
| one uring slice per pass, 1 / 2 / 4 / 8 reactors (3 queries, 8 streams) | 13.4 / 17.4 / 19.8 / 20.1 GB/s |
| eight slices per pass, same | 20.3 / 20.5 / 20.6 / 20.6 GB/s |
| Sirius, 22 queries, total time, 1 reactor / 4 reactors / readahead off | 52.1 s / 51.5 s / 58.5 s |

Reading:

- A queue depth of about 16 requests of 1 MiB (~16 MiB in flight) is enough to reach 95% of the array; adding reactors beyond what keeps ~16-32 requests in flight adds CPU, not throughput.
- gpuio's **device-destination** path (no GDS, staged through pinned host) costs 2.3-3.6x the CPU per GB of its host path (27.8 vs 12.1 ms/GB at queue depth 16; 71.1 vs 19.8 at queue depth 1), because of the copies and CUDA calls per request. That CPU cost, not bandwidth, is the realistic prize for a true direct path.
- The ceiling is the drives. A GDS path cannot exceed ~20.8 GB/s here; the question is whether it is cheaper (CPU, pinned memory, copy bandwidth), not faster.
- Disabling readahead costs 12-16%; more readahead (8 to 16 scans) adds nothing, so the pipeline already saturates the array when it is allowed to prefetch.
- Those runs went through compat-mode or staged host paths only. **No measurement of true GDS exists on either host** (no `nvidia-fs`; `gdscheck` reports compat for every driver).

## 7. Repetition and caching potential

Requested versus unique bytes over the 22 queries (GB; unique = union of all ranges per file):

| table | requested | unique | on disk | requested / unique |
|---|---:|---:|---:|---:|
| lineitem | 800.6 | 105.2 | 177.2 | 7.6x |
| orders | 99.2 | 41.4 | 52.4 | 2.4x |
| partsupp | 23.8 | 4.8 | 37.9 | 5.0x |
| customer | 16.1 | 10.8 | 10.8 | 1.5x |
| part | 6.9 | 3.2 | 5.1 | 2.2x |
| supplier | 1.7 | 0.7 | 0.7 | 2.6x |
| **total** | **948** | **~166** | ~283 | **~5.7x** |

- **Within-query duplicates** (a table scanned more than once in one query): 92.6 GB, almost all in Q21 (98.3 GB requested of lineitem, 38.8 unique), Q17 (71.7 vs 45.0) and Q18 (12.8 vs 6.4). At 20.8 GB/s that is ~4.4 s of disk time out of ~52 s, and an engine-level scan-sharing change removes it regardless of the I/O backend.
- **Cross-query reuse**: 782 GB (82% of requested bytes) were already requested by an earlier query or earlier in the same query. If a cache held the 166 GB union, the I/O floor of the whole suite would fall from ~45 s to ~8 s. This is an upper bound: it assumes one process runs the queries back to back, the cache holds everything, and the engine then becomes compute-bound.
- The union is 59% of lineitem and 79% of orders on disk; partsupp is touched by only 12.7%.
- A direct-to-GPU path and a host-memory cache pull in different directions: GDS skips host memory, while cuCascade's prefetching cache (`fs_cache`) keeps pinned host chunks. A cache designed around GDS would live in GPU memory (166 GB is about 60% of the GB300's 256 GiB of HBM, so it would compete with the working set) or in host memory. See [gds-reactor-recommendations.md](gds-reactor-recommendations.md) section 6.

## 8. Layout levers when you control the files

- **Row-group size** is the main lever: the DaMoN'26 study found GDS needs row groups of about 4M rows or more to saturate one SSD (10M rows used) and recommends at least 100 pages per column chunk for decode parallelism; DuckDB's default of 122,880 rows gives ~100 KB chunks. The SF1000 files already sit at ~1.06M rows and 1.6 MB median chunks. Larger row groups raise the chunk size, but cost parallelism granularity and per-row-group GPU memory.
- **Column order** decides which projections merge for free. Columns that queries read together and that are adjacent in the schema merge at gap 0; the TPC-H queries read different column sets, so no single order helps everything.
- **Sorting/clustering** is what enables row-group skipping. The generated files are clustered only by key. If lineitem were sorted by `l_shipdate`, the local analysis estimates Q6 would read ~14.5% of its bytes, Q14 ~2.5%, Q15 ~4.3% (uniform-density estimate, not measured). Skipping removes requests but does not change their size.
- **Page index**: the local files have none, so skipping below the column chunk is impossible.

## 9. What we still do not know, and how to find out

| Unknown | Why it matters | Cheapest way to learn it |
|---|---|---|
| Per-request timing, in-flight depth, per-stream reuse | the cuFile stream API serialises per stream and has a limited internal thread pool | add an opt-in trace writer to the reactors (timestamp, reactor, stream, destination kind, size, priority, completion latency); the Sirius trace has none of these |
| Whether reads target host or device memory | decides which path (GDS vs staged) is even relevant | same trace; the engine already knows |
| Destination alignment actually used | cuFile bounces unaligned destinations | log `dst % 4096`, `offset % 4096`, `size % 4096` per slice |
| True GDS throughput and CPU cost on `/raid` | the central question | requires nvidia-fs/P2PDMA/C2C and `data=ordered`; see the bring-up checklist in [gds-reactor-recommendations.md](gds-reactor-recommendations.md) |
| SF1000 file internals (page sizes, encodings, row-group count per file) | derived above only from range counts | read the footers of one lineitem part file with the layout tooling |

## 10. Source index

- Local footers: `~/Documents/tpch/{sf10,sf100}/parquet/*.parquet` on the author's workstation; report in [tpch-parquet-layout-sf10-sf100.md](tpch-parquet-layout-sf10-sf100.md).
- Range trace: `/localhome/local-aaramoon/sf1000_parquet_range_trace/` on the GB300 host; statistics in [tpch-sf1000-gb300-trace.md](tpch-sf1000-gb300-trace.md).
- Throughput results: `workspace/uring_ultra/results/` (fio, reactors), `remote-mcp/workspace/NOTES.md`, `gpuio/bench/` on the GB300 host.
- CTRACK timings: `workspace/sirius/ctrack_tpch_sf1000_cold_{full,top5}.csv` and `_report.md`; Sirius one-reactor runs under `sirius_1reactor_runs/runs/tpch_20261007_*`.
- External: DaMoN'26 Parquet-on-GDS study (arXiv 2602.17335); PystachIO (arXiv 2512.02862); Velox, cuDF and Arrow coalescing defaults, all summarised in [gds-field-experience.md](gds-field-experience.md).
