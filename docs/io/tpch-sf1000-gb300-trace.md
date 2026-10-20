# TPC-H SF1000 parquet range trace: read-pattern statistics (GB300 host)

Recorded read requests of Sirius running TPC-H Q1-Q22 at scale factor 1000 over local parquet files, and the statistics computed from them. This is the closest thing we have to the real request stream the disk reactors see.

> **Provenance and trust.**
> - The trace and every number below live on the GB300 experiment host. They were read and summarised by a remote Claude Code agent reached through the `gb300-experiment` MCP (2026-10-15); the statistics scripts `trace_stats.py`, `union_stats.py` and `merge_stats.py` ran there, in the directory `remote-mcp/workspace/analysis/`. **We did not re-run them.**
> - I checked the figures for internal consistency (listed in [Consistency checks](#consistency-checks)); all passed. Interpretations of JSON key names (for example `aligned_4k`) are the relaying agent's.
> - The trace has **no timestamps, thread ids, CUDA streams, destination kind (host/device) or latencies**. It supports analysis of request sizes, offsets and repetition, not of concurrency or timing.

## 1. What the trace is

| | |
|---|---|
| Workload | TPC-H SF1000, queries 1-22, parquet, cold GPU runs |
| Engine | Sirius (DuckDB extension) with cuCascade, branch `exp/phase2-cucascade-decimal64` @ `0c070de` (2026-10-07), uring/reactor read path |
| Data | `/raid/tpch/sf1000/<table>/part.N.parquet` (269 GB total on the host's md RAID0) |
| Files | lineitem 60, orders 15, partsupp 8, part 2, customer 2, supplier 1, nation 1, region 1 (90 files) |
| Trace location (GB300 host) | `/localhome/local-aaramoon/sf1000_parquet_range_trace/query_N/<table>/part.K.json` (169 MB, 1,294 JSON files); a tarball copy is under `workspace/uring_ultra/traces/` |
| Record format | per file: `{query_id, table, file, ranges:[{seq, split, offset, size, row_groups[]}]}` |
| What a range is | the **logical** per-column-chunk request from the parquet reader (not post-merge, not post-alignment: only 118 of 473,760 lineitem offsets are 4 KiB-aligned) |
| `split` | a (file, split) pair is one scan unit; the replay harness issues it as one read call carrying all its column-chunk ranges widened to 4 KiB. What produced splits (scan task vs thread) is not documented |
| `seq` | looks like a global issue order (`trace_stats.py` sorts by it); no definition found |
| `row_groups` | the parquet row-group indices a range covers (the agent's inference) |

Derived layout facts for the SF1000 files (these differ from the local SF10/SF100 files, see [tpch-parquet-layout-sf10-sf100.md](tpch-parquet-layout-sf10-sf100.md)):

- Q1 reads 7 lineitem columns and issues 39,480 ranges, Q6 reads 4 columns and issues 22,560. Both are multiples of 5,640, so lineitem has **5,640 row groups = 94 per file x 60 files**, about **1.06M rows per row group** (6.0e9 rows / 5,640). The local SF100 files use 122,880-row groups.
- Request sizes are therefore **8-9x larger** than in the local layout: lineitem column-chunk requests are 166 KB-3.97 MB (median 1.6 MB), against a median of 91 KiB locally.
- Q17, Q18 and Q21 issue more ranges than distinct ranges because their self-joins scan lineitem more than once (Q21: 56,400 ranges = 10 column-instances x 5,640, only 22,560 distinct).

## 2. Request size distribution (all 22 queries, raw logical ranges)

Sizes in bytes. `files` is the number of files touched.

| table | files | ranges | GB | min | p10 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lineitem | 60 | 473,760 | 800.63 | 166,263 | 331,591 | 1,607,772 | 3,332,824 | 3,937,150 | 3,967,411 |
| orders | 15 | 36,480 | 99.16 | 1,938 | 43,418 | 1,987,946 | 5,402,661 | 23,699,596 | 23,785,368 |
| partsupp | 8 | 16,128 | 23.83 | 10,900 | 10,954 | 1,728,433 | 2,654,079 | 2,654,271 | 2,658,384 |
| customer | 2 | 3,542 | 16.07 | 3,955 | 4,005 | 612,318 | 11,314,965 | 25,128,525 | 25,242,933 |
| part | 2 | 4,692 | 6.91 | 3,969 | 4,019 | 616,510 | 987,548 | 12,076,455 | 12,250,306 |
| supplier | 1 | 352 | 1.75 | 3,700 | 3,747 | 571,511 | 21,015,462 | 23,458,348 | 23,466,902 |
| nation | 1 | 26 | ~0 | 32 | 32 | 70 | 227 | 227 | 227 |
| region | 1 | 8 | ~0 | 32 | 32 | 73 | 73 | 73 | 73 |
| **all** | 90 | **534,988** | **948.35** | 32 | 331,401 | 1,607,953 | 3,336,165 | 5,402,599 | 25,242,933 |

Count histogram by request size:

| table | <4 KiB | 4-64 KiB | 64 KiB-1 MiB | 1-16 MiB | >=16 MiB |
|---|---:|---:|---:|---:|---:|
| lineitem | 0 | 0 | 225,600 | 248,160 | 0 |
| orders | 1,140 | 12,540 | 3,420 | 18,240 | 1,140 |
| partsupp | 0 | 4,608 | 0 | 11,520 | 0 |
| part | 1,632 | 0 | 2,652 | 408 | 0 |
| customer | 1,232 | 0 | 770 | 1,232 | 308 |
| supplier | 121 | 0 | 99 | 77 | 55 |
| nation | 26 | 0 | 0 | 0 | 0 |
| region | 8 | 0 | 0 | 0 | 0 |
| **all** | 4,159 | 17,148 | 232,541 | 279,637 | 1,503 |

Reading: the benchmark is dominated by lineitem (84% of requested bytes) and its requests never fall below 166 KB nor exceed 4 MB. Only the dimension tables produce tiny (<4 KiB) or very large (>=16 MiB) requests.

## 3. Offsets, gaps and alignment (per file, ranges ordered by `seq`/offset)

How each range relates to the previous range of the same file: first range of the file; **contiguous** (starts exactly where the previous ended); **forward gap**; or **backward/overlap**.

| table | first | contiguous | forward gap | backward/overlap |
|---|---:|---:|---:|---:|
| lineitem | 1,020 | 163,560 | 297,636 | 11,544 |
| orders | 180 | 10,245 | 24,915 | 1,140 |
| partsupp | 40 | 8,448 | 7,623 | 17 |
| part | 16 | 1,224 | 3,452 | 0 |
| customer | 16 | 1,230 | 2,140 | 156 |
| supplier | 10 | 152 | 189 | 1 |
| nation | 9 | 12 | 2 | 3 |
| region | 3 | 4 | 0 | 1 |

Size of the forward gaps (bytes):

| table | gaps | p10 | p50 | p90 | p99 | max | <4 KiB | 4-64 KiB | 64 KiB-1 MiB | 1-16 MiB | >=16 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lineitem | 297,636 | 963,633 | 7,536,266 | 22,685,392 | 25,024,882 | 25,289,692 | 0 | 0 | 45,120 | 184,553 | 67,963 |
| orders | 24,915 | 4,677,574 | 16,780,091 | 40,149,887 | 40,527,012 | 40,606,384 | 0 | 0 | 1,140 | 10,533 | 13,242 |
| partsupp | 7,623 | 1,866,598 | 43,018,559 | 45,898,640 | 47,659,621 | 47,817,861 | 0 | 0 | 0 | 2,304 | 5,319 |

Findings:

- **No gap is below 64 KiB.** Every gap is either exactly zero (contiguous ranges) or at least 64 KiB, so a merge gap anywhere between 0 and 64 KiB changes nothing.
- 35% of lineitem ranges are contiguous with the previous one (adjacent projected columns), and 63% follow a forward gap whose median is 7.5 MB (a skipped column).
- **Alignment: effectively never.** Ranges whose offset is 4 KiB-aligned: lineitem 118 of 473,760, orders 21 of 36,480, part 3 of 4,692 (about 1 in 4,096, i.e. chance). Ranges whose size is a 4 KiB multiple: lineitem 279, partsupp 10. Everything upstream of the reactor must widen to page boundaries before any direct I/O.
- **No range crosses end of file.**

## 4. Per-query volume

Ranges and GB per query, and how they split by table (GB). `files` and `splits` count distinct files and (file, split) scan units.

| Q | ranges | GB | files | splits | lineitem | orders | partsupp | part | customer | supplier | nation/region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39,480 | 40.4 | 60 | 95 | 40.4 | | | | | | |
| 2 | 5,532 | 7.9 | 13 | 19 | | | 6.7 | 0.4 | | 0.7 | ~0 |
| 3 | 27,428 | 40.8 | 77 | 101 | 32.3 | 8.5 | | | 0.1 | | |
| 4 | 20,340 | 22.9 | 75 | 91 | 20.0 | 2.9 | | | | | |
| 5 | 26,315 | 50.6 | 80 | 104 | 42.0 | 8.5 | | | 0.1 | ~0 | ~0 |
| 6 | 22,560 | 34.9 | 60 | 81 | 34.9 | | | | | | |
| 7 | 30,814 | 57.4 | 79 | 105 | 51.1 | 6.2 | | | 0.1 | ~0 | ~0 |
| 8 | 32,364 | 73.0 | 82 | 109 | 64.2 | 8.5 | | 0.2 | 0.1 | ~0 | ~0 |
| 9 | 38,856 | 76.8 | 87 | 121 | 68.7 | 2.3 | 3.4 | 2.4 | | ~0 | ~0 |
| 10 | 27,060 | 43.9 | 78 | 106 | 24.7 | 8.5 | | | 10.7 | | ~0 |
| 11 | 5,400 | 9.6 | 10 | 16 | | | 9.6 | | | ~0 | ~0 |
| 12 | 30,480 | 32.0 | 75 | 99 | 31.4 | 0.6 | | | | | |
| 13 | 3,574 | 33.2 | 17 | 29 | | 33.2 | | | ~0 | | |
| 14 | 22,968 | 52.8 | 62 | 80 | 52.6 | | | 0.2 | | | |
| 15 | 22,604 | 49.6 | 61 | 79 | 49.2 | | | | | 0.4 | |
| 16 | 2,374 | 2.0 | 11 | 12 | | | 1.3 | 0.5 | | 0.2 | |
| 17 | 28,812 | 72.0 | 62 | 137 | 71.7 | | | 0.3 | | | |
| 18 | 27,428 | 27.0 | 77 | 129 | 12.8 | 13.4 | | | 0.7 | | |
| 19 | 34,656 | 52.3 | 62 | 82 | 51.8 | | | 0.4 | | | |
| 20 | 25,318 | 60.1 | 72 | 89 | 54.6 | | 2.8 | 2.4 | | 0.3 | ~0 |
| 21 | 58,715 | 98.7 | 77 | 231 | 98.3 | 0.4 | | | | 0.1 | ~0 |
| 22 | 1,910 | 10.4 | 17 | 17 | | 6.2 | | | 4.3 | | |
| **sum** | **534,988** | **948.3** | | | 800.7 | | | | | | |

Per-split scan units (the unit the replay harness issues as one read call):

| table | splits | p50 bytes | p90 bytes | max bytes | ranges per split (p50 / max) |
|---|---:|---:|---:|---:|---|
| lineitem | 1,597 | 465 MB | 877 MB | 1.20 GB | 296 / 658 |
| orders | 221 | 411 MB | 895 MB | 2.21 GB | 152 / 304 |
| partsupp | 53 | 422 MB | 844 MB | 1.20 GB | 288 / 672 |

## 5. Repetition: requested vs unique bytes

The statistic `repeat_cross_query_bytes` in the original `trace_stats.json` is **misnamed**: it equals requested minus unique, so it counts re-reads inside one query as well as across queries (checked: lineitem requested - unique = 695,469,214,903 B, byte-for-byte the stored value). The correct figures, from `union_stats.py` (unique = size of the union of all ranges per file across all 22 queries):

| table | requested GB | unique GB | on-disk GB | unique / disk | requested / unique |
|---|---:|---:|---:|---:|---:|
| lineitem | 800.63 | 105.16 | 177.21 | 59.3% | 7.61x |
| orders | 99.16 | 41.39 | 52.37 | 79.0% | 2.40x |
| partsupp | 23.83 | 4.81 | 37.88 | 12.7% | 4.96x |
| customer | 16.07 | 10.76 | 10.76 | 100% | 1.49x |
| part | 6.91 | 3.15 | 5.08 | 62.0% | 2.19x |
| supplier | 1.75 | 0.69 | 0.69 | 100% | 2.55x |
| nation, region | ~0 | ~0 | ~0 | | |
| **total** | **948.3** | **~166** | **~283** | | **~5.7x** |

- Widening every range to 4 KiB changes the unique total by only +0.05 GB.
- **First-touch bytes** (the new bytes each query adds to the union, running Q1 to Q22): lineitem Q1 40.4, Q3 1.9, Q4 18.2, Q5 18.8, Q8 22.2, Q12 2.3, Q19 1.5 GB, all others 0; orders Q13 27.0, Q3 8.5, Q18 5.0, Q4 0.6, Q21 0.4 GB. A cache that retains everything pays the I/O for ~166 GB once and serves the other ~780 GB.
- **Within-query repeats** (a table scanned more than once in a single query): the sum of per-query unique lineitem bytes is 708.08 GB against 800.63 GB requested, so 92.55 GB (11.6% of lineitem) are duplicate reads inside a query, almost all in Q21 (98.3 GB requested, 38.8 GB unique), Q17 (71.7 vs 45.0) and Q18 (12.8 vs 6.4). Smaller repeats occur in Q2, Q11 and Q22 (partsupp, supplier, customer).

## 6. What merging does (per query and file, no cross-query dedupe)

Simulation by `merge_stats.py`: for each (query, file) the ranges are sorted by offset, widened to 4 KiB boundaries, and merged when the gap is at most `g`. "Over-read" is merged bytes relative to the raw bytes; it is **negative at small `g`** because raw ranges overlap inside a (query, file), so summing raw sizes double-counts repeats.

| scope | g | requests | GB | vs raw | p10 | p50 | p90 | p99 | max (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | raw | 534,988 | 948.3 | | 0.316 | 1.533 | 3.182 | 5.152 | 24.1 |
| all | 0 (= 64 KiB) | 284,293 | 846.6 | -10.7% | 0.320 | 3.172 | 5.199 | 7.254 | 654.8 |
| all | 1 MiB | 231,820 | 882.1 | -7.0% | 0.320 | 3.184 | 6.836 | 11.898 | 5,130 |
| all | 8 MiB | 134,142 | 1,307.0 | +37.8% | 0.320 | 11.059 | 14.359 | 17.770 | 5,130 |
| lineitem | raw | 473,760 | 800.6 | | 0.316 | 1.533 | 3.178 | 3.755 | 3.78 |
| lineitem | 0 (= 64 KiB) | 248,160 | 709.1 | -11.4% | 0.320 | 3.176 | 4.609 | 7.250 | 7.31 |
| lineitem | 1 MiB | 197,400 | 743.8 | -7.1% | 0.320 | 3.609 | 6.836 | 11.891 | 12.0 |
| lineitem | 8 MiB | 107,162 | 1,138.0 | +42.1% | 0.320 | 11.875 | 17.391 | 17.750 | 17.9 |
| orders | raw | 36,480 | 99.2 | | 0.041 | 1.896 | 5.152 | 22.602 | 22.7 |
| orders | 0 (= 64 KiB) | 25,095 | 99.3 | +0.1% | 0.043 | 1.902 | 5.199 | 27.805 | 27.9 |
| orders | 1 MiB | 23,955 | 99.6 | +0.5% | 0.043 | 1.902 | 5.199 | 27.805 | 27.9 |
| orders | 8 MiB | 18,255 | 127.1 | +28.2% | 0.043 | 5.195 | 11.555 | 27.809 | 27.9 |

(The all-table maxima of 655 MiB and 5.1 GB come from whole-file reads of other tables; lineitem and orders top out at 28 MiB.)

Share of **bytes** by request size (what matters for the GDS-vs-POSIX decision):

| scope | g | <1 MiB | 1-4 MiB | 4-16 MiB | >=16 MiB | **>=1 MiB** |
|---|---|---:|---:|---:|---:|---:|
| all | raw | 12.2% | 77.0% | 7.0% | 3.8% | **87.8%** |
| all | 0 | 4.1% | 51.0% | 39.6% | 5.4% | **95.9%** |
| all | 1 MiB | 2.9% | 31.7% | 60.2% | 5.2% | **97.1%** |
| all | 8 MiB | 0.4% | 2.3% | 77.9% | 19.4% | **99.6%** |
| lineitem | raw | 13.9% | 86.1% | 0.0% | 0.0% | **86.1%** |
| lineitem | 0 | 4.5% | 57.1% | 38.4% | 0.0% | **95.5%** |
| lineitem | 1 MiB | 3.2% | 34.0% | 62.8% | 0.0% | **96.8%** |
| lineitem | 8 MiB | 0.3% | 1.6% | 79.8% | 18.3% | **99.7%** |
| orders | raw | 2.1% | 16.0% | 54.7% | 27.2% | **97.9%** |
| orders | 0 | 1.2% | 14.3% | 51.1% | 33.4% | **98.8%** |

Per query at gap 0: requests, GB (unique per query after merging) and the share of bytes in requests of at least 1 MiB.

| Q | requests | GB | >=1 MiB | Q | requests | GB | >=1 MiB |
|---|---:|---:|---:|---|---:|---:|---:|
| 1 | 5,640 | 40.41 | 100.0% | 12 | 19,200 | 32.07 | 85.0% |
| 2 | 2,151 | 4.50 | 98.3% | 13 | 1,309 | 33.22 | 100.0% |
| 3 | 20,648 | 40.91 | 95.2% | 14 | 17,328 | 52.88 | 99.6% |
| 4 | 13,560 | 22.97 | 91.5% | 15 | 16,942 | 49.69 | 100.0% |
| 5 | 19,532 | 50.65 | 96.1% | 16 | 1,188 | 2.05 | 99.9% |
| 6 | 11,280 | 34.99 | 100.0% | 17 | 11,892 | 45.35 | 99.4% |
| 7 | 24,031 | 57.48 | 96.5% | 18 | 13,714 | 20.62 | 68.8% |
| 8 | 14,300 | 73.01 | 99.6% | 19 | 17,532 | 52.34 | 92.5% |
| 9 | 15,323 | 76.90 | 99.9% | 20 | 17,904 | 60.16 | 92.4% |
| 10 | 19,357 | 43.94 | 92.3% | 21 | 19,223 | 39.37 | 94.1% |
| 11 | 791 | 4.82 | 99.9% | 22 | 1,448 | 8.30 | 100.0% |

Reading:

- **On this dataset nearly all bytes are in requests of at least 1 MiB** (87.8% raw, 95.9% after free gap-0 merging), so a 1 MiB GDS-vs-POSIX threshold routes almost everything to the GDS path. This is the opposite of the local SF100 layout (about 5% raw, 31% merged), where the threshold decides the outcome.
- **Merging contiguous ranges halves the request count** (534,988 to 284,293, -47%) at no extra bytes. Gap-0 and 64 KiB give identical results because no gaps fall between 0 and 64 KiB.
- A 1 MiB merge gap removes another 18% of requests for +4% bytes versus gap 0 (882 vs 847 GB); an 8 MiB gap over-reads +38-42% and is not worth it.

## 7. What the trace does not tell us

- **Concurrency**: no in-flight depth, no per-thread or per-stream pattern, no inter-arrival times. Whether reads of a (file, split) are issued back to back or paced is unknown.
- **Destination**: host vs device, and how many reads share a CUDA stream, are not recorded. That matters for the cuFile stream API (see [gds-reactor-recommendations.md](gds-reactor-recommendations.md)).
- **Latency and throughput per size bucket**: not recorded. Measured raw throughput on this host is in [tpch-io-patterns.md](tpch-io-patterns.md) section 6.
- **Cold or warm page cache**: runs used O_DIRECT; no cache state recorded.

## Consistency checks

Performed on the relayed figures (all passed):

- The per-query GB values sum to 948.3 GB, equal to the overall total (948.35 GB); the lineitem column of the per-query table sums to 800.7 GB (stated 800.63).
- Range counts per query are integer multiples of the row-group count for the columns each query reads (Q1: 39,480 = 7 x 5,640; Q6: 22,560 = 4 x 5,640; Q21: 56,400 = 10 x 5,640).
- The lineitem first-touch values (40.4 + 1.9 + 18.2 + 18.8 + 22.2 + 2.3 + 1.5) sum to 105.3 GB, the stated union of 105.16 GB.
- The order/contiguity counts sum to the range count for every table, and the gap-size histograms sum to the gap count.
- The share of 4 KiB-aligned offsets (118 of 473,760 = 0.025%) matches the expected chance rate and the 0.026% measured independently on the local SF100 files.
