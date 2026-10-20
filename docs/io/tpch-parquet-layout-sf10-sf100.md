# TPC-H Parquet physical layout (local SF10 / SF100) and the byte-range reads of a column-pruning GPU scan

> **Scope and caveats.**
> - This document analyses the **local DuckDB-written SF10 and SF100 files** (`~/Documents/tpch/{sf10,sf100}/parquet/`, one file per table, 122,880-row row groups). It is **not** the layout of the SF1000 files in the GB300 range trace (60 lineitem files, ~1.06M rows per row group, see [tpch-sf1000-gb300-trace.md](tpch-sf1000-gb300-trace.md)). Row-group size is the dominant factor in request sizes, so conclusions about "how much of the workload is large enough for GDS" differ sharply between the two layouts; [tpch-io-patterns.md](tpch-io-patterns.md) compares them.
> - Everything here is derived from Parquet footers and page headers (no data page was decoded). Per-query column sets are derived by hand from the standard TPC-H SQL and assume perfect column pruning, so request counts and byte shares are model outputs (tag **[A]**/**[D]**), not measurements of a real engine.
> - The per-query detail tables of the original run (one block per query and scale factor) are omitted here for size; section 7.1 keeps the per-query request counts and section 8.5 the per-query GDS-eligible shares. The analysis scripts were run from a scratch directory and are **not** committed (see section 11).
> - Generated 2026-10-14 by a read-only analysis; reconciliation and validation steps are listed in section 1.


## 0. Executive summary

**Provenance of every number:** `[M]` measured from the Parquet footers / page headers of the 16 local files; `[D]` derived arithmetic on measured values; `[A]` depends on an assumption (query column sets from the TPC-H spec, perfect column pruning, every row group read). Binary units (KiB = 1024 B) throughout.

1. **[M] What the files are.** One Parquet file per table per scale factor (sf10: 3.21 GiB, sf100: 33.20 GiB in total), written by `DuckDB version v1.4.4 (build dddafe8)`, format version 1. Every non-final row group has 122,880 rows (tables with fewer rows - nation, region, and supplier at sf10 - are one short row group): lineitem sf100 = 4,884 row groups of 4.4 MiB. All chunks are SNAPPY. **Exactly one data page per column chunk** (plus one dictionary page if dictionary-encoded), data pages are v1, **no page index / offset index**, no page CRCs; min/max/null_count stats on every chunk; bloom filters exist only for dictionary-encoded chunks and sit in a separate region after the last row group, not between chunks. Row-group size is rows-based (not bytes-based), so on-disk row groups are 3.7 MiB (part) ... 9.7 MiB (customer).

2. **[M] Chunk sizes are small relative to the 1 MiB GDS threshold.** sf100 has 94,914 column chunks: 92.3% of them (56.6% of all bytes) are < 1 MiB; none is >= 16 MiB; the largest is 4.99 MiB (partsupp.ps_comment). Median chunk per table: lineitem 90.8 KiB; orders 478.5 KiB; partsupp 480.7 KiB; part 121.7 KiB; customer 593.2 KiB; supplier 595.2 KiB. Only 10 columns have a median chunk >= 1 MiB (partsupp.ps_comment (5.0 MiB), customer.c_address (3.3 MiB), supplier.s_address (3.3 MiB), customer.c_comment (3.2 MiB), supplier.s_comment (2.8 MiB), orders.o_comment (2.3 MiB), part.p_name (1.6 MiB), lineitem.l_comment (1.5 MiB), customer.c_phone (1.4 MiB), supplier.s_phone (1.4 MiB)) - all free-text/long-string columns.

3. **[M] Layout is gap-free but unaligned.** Every gap between adjacent column chunks - inside a row group and across row-group boundaries - is exactly 0 B in all 16 files; the chunk region is one contiguous run from byte 4. Only 25 of 94,914 sf100 chunk offsets (0.026%) are 4 KiB-aligned (and 0.022% of sizes): coincidence-level. Consequently adjacent-column merging is free (0 over-read), while 4 KiB-aligning requests adds ~2 KiB per request edge.

4. **[A+D] Column-pruned scan of the 22 queries at sf100 (union model).** Per query: 1,590-34,188 ranges (Q22 ... Q1) and 588.5 MiB-12.1 GiB of useful bytes (Q16 ... Q9); sum over the 22 queries 124.8 GiB (reading whole tables instead would be 3.7x more). Uncoalesced, the median range of a query is 60.6 KiB-640.9 KiB (column chunks).

5. **[D] How much would use the GDS path (requests >= 1 MiB)? sf100, union model** - share of requested bytes (EW = every query weighted equally; whole = whole benchmark):

| request formation | requests (22 queries) | over-read | GDS-eligible bytes, EW, file-global | whole, file-global | EW, per-RG | whole, per-RG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 416,897 | 0.00% | 5.1% | 3.6% | 5.1% | 3.6% |
| gap 0 | 250,852 | 0.00% | 30.0% | 31.5% | 30.0% | 31.5% |
| 4K-aligned | 250,849 | 0.77% | 30.0% | 31.3% | 29.9% | 31.3% |
| gap 4 KiB | 250,849 | 0.00% | 30.0% | 31.5% | 30.0% | 31.5% |
| gap 64 KiB | 229,959 | 0.65% | 32.8% | 35.5% | 32.8% | 35.5% |
| gap 1 MiB | 137,796 | 28.81% | 73.9% | 84.7% | 73.9% | 84.7% |

   Reading it: raw column-chunk requests send only 5.1% (EW) / 3.6% (whole) of the bytes through GDS; merging *touching* ranges (free) lifts that to 30.0% / 31.5%; 4 KiB alignment changes the request count by 3 out of 250,852 but costs +0.77% extra bytes; a 64 KiB merge gap helps little because holes between projected columns are mostly whole column chunks: of the 148,880 non-contiguous within-row-group holes over the 22 queries, only 20,893 (14.0%) are <= 64 KiB and 35,824 are > 1 MiB; only a **1 MiB gap** moves most bytes to GDS (73.9% EW / 84.7% whole file-global; 73.9% / 84.7% per-RG) and it costs **+28.8% over-read** (36.0 GiB extra). The remaining POSIX share at gap 1 MiB comes from queries whose projected columns are all small or far apart (see 8.5). Reading unpruned row groups (3.7 MiB-9.7 MiB each) would be ~100% GDS-eligible but 3.7x the bytes.

6. **[M+A] Row-group skipping by min/max is worthless on this data**: 0 of the row groups of the predicate tables are skippable across all 22 queries at both scale factors (only generation-order key columns are clustered; dates, prices, quantities and flags span ~100% of their domain in every row group; the low-cardinality columns checked hold all their distinct values in every row group). Even if skipping were available it would change request *counts*, not request *sizes* (section 9).

7. **[M] String vs fixed width (sf100, median chunk).** Free-text comment columns are lineitem 34%, orders 46%, partsupp 78%, part 27%, customer 33%, supplier 30% of the on-disk bytes of their table; low-cardinality dictionary strings (flags, ship mode, priorities, brand/container/type, segment) are 18.8 KiB-121.7 KiB chunks; other PLAIN strings (names, addresses, phones, clerk) 596.0 KiB-3.3 MiB; fixed-width PLAIN columns (keys, prices, balances) 187.1 KiB-672.1 KiB; small dictionary-encoded fixed-width columns (dates, quantity, discount, tax, line number, nation keys, size) 53 B-271.1 KiB. Details in section 5.

## 1. Method, provenance and validation

**Data (read-only):** `~/Documents/tpch/sf10/parquet/` and `~/Documents/tpch/sf100/parquet/`, files `{lineitem,orders,partsupp,part,customer,supplier,nation,region}.parquet` (one per table; nothing else in the directories). Nothing in the data directories or in the git repository was modified; all scripts and outputs lived in a scratch directory and are not part of the repository (section 11).

**Parsing.** No pyarrow/duckdb/thrift package is installed (the pixi env has no pyarrow; no network use was needed), so the footers were decoded with a small, defensive, pure-Python Thrift *compact protocol* reader (`pq_thrift.py`; bounds-checked, depth/size-capped) implementing the relevant parts of `parquet.thrift` (FileMetaData, SchemaElement, RowGroup, ColumnChunk, ColumnMetaData, Statistics, PageHeader, OffsetIndex). Only the tail of each file (footer) plus, for the page walk, one tiny `pread` per page header was read.

**Validation of the parser / interpretation:** (1) per-file row counts equal the TPC-H cardinalities and each RG's rows sum to the footer's `num_rows`; (2) for every file `4 + sum(column chunk bytes) + sum(bloom lengths) + footer + 8 == file size` exactly (0 unexplained bytes); (3) RG-level `file_offset`/`total_compressed_size` agree with the chunk offsets/sizes of all row groups; (4) the page-header walk of **every** row group (104,446 chunks, 171,126 pages) tiles each chunk exactly (sum of header+compressed page sizes == `total_compressed_size`), the first data page sits at `data_page_offset` and a dictionary page exists iff `dictionary_page_offset` is set - 0 exceptions; (5) the scan simulator agrees with literal Python ports of the repo's `coalesce_ranges` and `uring_reactor::align_and_coalesce` on all 528 query/variant/scope checks (0 mismatches); (6) the row-group min/max evaluator passes unit tests including a case that must skip.

**Repo context used to choose the request model** (read-only): `include/cucascade/io/io_utils.hpp` (`coalesce_ranges`, `align_and_coalesce`), `src/io/uring/uring_reactor.cpp` (`align_and_coalesce`: 4096-byte `IO_BLOCK_SIZE` widening then merging of touching ranges), `benchmark/parquet_io_benchmark.cpp` (sorts all `all_column_chunks_byte_ranges` of a file and merges ranges with `offset <= previous_end`, i.e. file-global gap-0 merging), and `cufile_reactor::dispatch_slice` (GDS only if the slice's EOF-clamped size is `>= gds_threshold`, default 1 MiB, the slice is device-only and the file registered with cuFile; everything else is POSIX `pread`). I treat every merged range as one slice / one physical operation.

**Labels.** `[M]` measured from footers/page headers; `[D]` derived by arithmetic from measured values; `[A]` relies on an assumption (the per-query column sets come from the standard TPC-H SQL; perfect column pruning; all row groups read; one scan per table (union model) unless stated).

## 2. File-level facts

### 2.1 Files, row groups, sizes  [M]

One file per table per scale factor (no partitioning, no directory sharding). Sizes are bytes on disk (binary units: KiB = 1024 B).

| SF | table | file size | bytes | rows | row groups | rows per RG | RG on-disk size (median) | footer | bloom filters (#) | bloom bytes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sf10 | lineitem | 2.07 GiB | 2223320321 | 59,986,052 | 489 | 122,880 x 488 + 1 x 20,612 | 4.33 MiB | 771.3 KiB | 5,379 | 5.9 MiB |
| sf10 | orders | 557.08 MiB | 584139341 | 15,000,000 | 123 | 122,880 x 122 + 1 x 8,640 | 4.54 MiB | 125.4 KiB | 613 | 2.4 MiB |
| sf10 | partsupp | 419.54 MiB | 439923876 | 8,000,000 | 66 | 122,880 x 65 + 1 x 12,800 | 6.43 MiB | 63.1 KiB | 65 | 1.0 MiB |
| sf10 | part | 60.77 MiB | 63720671 | 2,000,000 | 17 | 122,880 x 16 + 1 x 33,920 | 3.71 MiB | 17.9 KiB | 102 | 417.0 KiB |
| sf10 | customer | 118.04 MiB | 123774648 | 1,500,000 | 13 | 122,880 x 12 + 1 x 25,440 | 9.67 MiB | 14.7 KiB | 26 | 1.2 KiB |
| sf10 | supplier | 7.54 MiB | 7904868 | 100,000 | 1 | 100,000 x 1 | 7.54 MiB | 1.1 KiB | 1 | 47 B |
| sf10 | nation | 2.24 KiB | 2294 | 25 | 1 | 25 x 1 | 1.49 KiB | 708 B | 1 | 47 B |
| sf10 | region | 1.03 KiB | 1057 | 5 | 1 | 5 x 1 | 441 B | 604 B | 0 | 0 B |
| sf100 | lineitem | 21.23 GiB | 22791208136 | 600,037,902 | 4,884 | 122,880 x 4,883 + 1 x 14,862 | 4.44 MiB | 7.6 MiB | 53,724 | 59.4 MiB |
| sf100 | orders | 6.06 GiB | 6506393573 | 150,000,000 | 1,221 | 122,880 x 1,220 + 1 x 86,400 | 5.08 MiB | 1.2 MiB | 4,884 | 5.0 MiB |
| sf100 | partsupp | 4.10 GiB | 4399128704 | 80,000,000 | 652 | 122,880 x 651 + 1 x 5,120 | 6.43 MiB | 624.6 KiB | 651 | 10.2 MiB |
| sf100 | part | 607.75 MiB | 637272481 | 20,000,000 | 163 | 122,880 x 162 + 1 x 93,440 | 3.71 MiB | 173.0 KiB | 978 | 4.2 MiB |
| sf100 | customer | 1.15 GiB | 1237876451 | 15,000,000 | 123 | 122,880 x 122 + 1 x 8,640 | 9.67 MiB | 139.5 KiB | 246 | 11.3 KiB |
| sf100 | supplier | 75.44 MiB | 79103880 | 1,000,000 | 9 | 122,880 x 8 + 1 x 16,960 | 9.27 MiB | 8.4 KiB | 9 | 423 B |
| sf100 | nation | 2.24 KiB | 2294 | 25 | 1 | 25 x 1 | 1.49 KiB | 708 B | 1 | 47 B |
| sf100 | region | 1.03 KiB | 1057 | 5 | 1 | 5 x 1 | 441 B | 604 B | 0 | 0 B |

Row counts equal the TPC-H cardinalities (e.g. lineitem sf100 = 600,037,902 rows), and `sum(RG rows) == footer num_rows` for every file: verified.

### 2.2 Format features (identical across all 16 files unless stated)  [M]

| feature | measured value |
| :--- | :--- |
| Writer (`created_by`) | `DuckDB version v1.4.4 (build dddafe8)` (all 16 files) |
| Parquet format `version` | 1 |
| Codec | SNAPPY on 100% of chunks (sf10: 9,532, sf100: 94,914 chunks; verified); no UNCOMPRESSED / ZSTD / LZ4 chunk anywhere |
| Data page type | only DATA_PAGE (v1); DATA_PAGE_V2 pages = 0 (page headers of **every** row group walked: yes) |
| Pages per column chunk | exactly 1 data page, plus 1 dictionary page iff the chunk is dictionary-encoded, for every chunk of every file (sf100 lineitem: 78,144 chunks = 78,144 data pages + 53,724 dictionary pages) |
| Page headers | data page header 17-25 B, dictionary page header 13-19 B; definition/repetition levels RLE; page CRC present on 0 pages; page-level statistics present on 0 pages |
| Chunk encodings (`ColumnMetaData.encodings`) | `[PLAIN]` or `[PLAIN_DICTIONARY]` only. Dictionary chunks: dictionary page = PLAIN, data page = PLAIN_DICTIONARY (legacy id 2, not RLE_DICTIONARY). No DELTA_*, BYTE_STREAM_SPLIT; no `encoding_stats` / `size_statistics` fields |
| Dictionary use | chosen per column chunk by the writer. sf100 lineitem: 11 of 16 columns dictionary-encoded in 100% of row groups (every column in 3.2); a few columns switch between PLAIN and dictionary across row groups (e.g. ps_availqty is PLAIN only in the last, 5,120-row group) |
| Offset index / column index (page index) | **none**: 0 chunks (of 104,446) carry `offset_index_offset` or `column_index_offset` |
| Bloom filters | present only for dictionary-encoded chunks (sf100 lineitem: 53,724 chunks, 59.4 MiB). They are **not** interleaved with column data: all bloom filters of a file sit back-to-back in one region between the last row group and the footer. Filter lengths seen: 47, 80, 272, 4,112, 8,209, 16,401, 32,785 B |
| Statistics | `min_value`/`max_value` (modern fields) + `null_count` (always 0) on 100% of chunks; `distinct_count` only on dictionary chunks. String min/max are complete values, not truncated prefixes (longest seen: 198 bytes), so string equality/prefix pruning is exact |
| Row-group metadata | `file_offset` and `total_compressed_size` set for every row group and consistent with the chunk offsets (RG offset == first chunk offset; RG size == sum of chunk sizes; verified for all row groups); `sorting_columns`/`ordinal` unset (verified); deprecated `ColumnChunk.file_offset` always 0 (verified); no key/value metadata (verified); no column_orders; no encryption |
| Row-group size | 122,880 rows (DuckDB default) for every non-final row group of every table; the final row group is short. There is no byte-size target: the on-disk RG size at sf100 ranges from 3.71 MiB (part) to 9.67 MiB (customer) |

File composition - `PAR1` (4 B) | all column chunks, back to back, no padding | bloom-filter region | footer | footer length (4 B) | `PAR1` (4 B). The bytes of every file reconcile exactly: `4 + sum(chunk sizes) + sum(bloom lengths) + footer + 8 == file size` (0 unexplained bytes in all 16 files; table below for sf100).

| sf100 table | column-chunk region | % of file | bloom region | % of file | footer | % of file | holes inside chunk region (B) | unexplained bytes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 21.16 GiB | 99.692% | 59.36 MiB | 0.273% | 7.61 MiB | 0.035% | 0 | 0 |
| orders | 6.05 GiB | 99.901% | 4.95 MiB | 0.080% | 1.20 MiB | 0.019% | 0 | 0 |
| partsupp | 4.09 GiB | 99.743% | 10.18 MiB | 0.243% | 624.61 KiB | 0.015% | 0 | 0 |
| part | 603.42 MiB | 99.287% | 4.16 MiB | 0.685% | 173.02 KiB | 0.028% | 0 | 0 |
| customer | 1.15 GiB | 99.988% | 11.29 KiB | 0.001% | 139.52 KiB | 0.012% | 0 | 0 |
| supplier | 75.43 MiB | 99.989% | 423 B | 0.001% | 8.40 KiB | 0.011% | 0 | 0 |
| nation | 1.49 KiB | 66.565% | 47 B | 2.049% | 708 B | 30.863% | 0 | 0 |
| region | 441 B | 41.722% | 0 B | 0.000% | 604 B | 57.143% | 0 | 0 |

### 2.3 Encoding / dictionary usage per table  [M]

Per table and scale factor: which columns are dictionary-encoded in 100% of their row groups, which switch between dictionary and PLAIN ('mixed'), and which are PLAIN everywhere; plus bloom-filter and page-index presence and the codecs seen. (`encodings` of each chunk are `[PLAIN]` or `[PLAIN_DICTIONARY]`; see 2.2.)

| SF | table | codec | dictionary in 100% of RGs | mixed (share of RGs with dictionary) | PLAIN in every RG | bloom-filter chunks / chunks | offset / column index chunks |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| sf10 | lineitem | SNAPPY | l_linenumber, l_quantity, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode | - | l_orderkey, l_partkey, l_suppkey, l_extendedprice, l_comment | 5,379 / 7,824 | 0 / 0 |
| sf10 | orders | SNAPPY | o_orderstatus, o_orderpriority, o_shippriority | o_orderdate (99.2%), o_clerk (99.2%) | o_orderkey, o_custkey, o_totalprice, o_comment | 613 / 1,107 | 0 / 0 |
| sf10 | partsupp | SNAPPY | - | ps_availqty (98.5%) | ps_partkey, ps_suppkey, ps_supplycost, ps_comment | 65 / 330 | 0 / 0 |
| sf10 | part | SNAPPY | p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice | - | p_partkey, p_name, p_comment | 102 / 153 | 0 / 0 |
| sf10 | customer | SNAPPY | c_nationkey, c_mktsegment | - | c_custkey, c_name, c_address, c_phone, c_acctbal, c_comment | 26 / 104 | 0 / 0 |
| sf10 | supplier | SNAPPY | s_nationkey | - | s_suppkey, s_name, s_address, s_phone, s_acctbal, s_comment | 1 / 7 | 0 / 0 |
| sf10 | nation | SNAPPY | n_regionkey | - | n_nationkey, n_name, n_comment | 1 / 4 | 0 / 0 |
| sf10 | region | SNAPPY | - | - | r_regionkey, r_name, r_comment | 0 / 3 | 0 / 0 |
| sf100 | lineitem | SNAPPY | l_linenumber, l_quantity, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode | - | l_orderkey, l_partkey, l_suppkey, l_extendedprice, l_comment | 53,724 / 78,144 | 0 / 0 |
| sf100 | orders | SNAPPY | o_orderstatus, o_orderdate, o_orderpriority, o_shippriority | - | o_orderkey, o_custkey, o_totalprice, o_clerk, o_comment | 4,884 / 10,989 | 0 / 0 |
| sf100 | partsupp | SNAPPY | - | ps_availqty (99.8%) | ps_partkey, ps_suppkey, ps_supplycost, ps_comment | 651 / 3,260 | 0 / 0 |
| sf100 | part | SNAPPY | p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice | - | p_partkey, p_name, p_comment | 978 / 1,467 | 0 / 0 |
| sf100 | customer | SNAPPY | c_nationkey, c_mktsegment | - | c_custkey, c_name, c_address, c_phone, c_acctbal, c_comment | 246 / 984 | 0 / 0 |
| sf100 | supplier | SNAPPY | s_nationkey | - | s_suppkey, s_name, s_address, s_phone, s_acctbal, s_comment | 9 / 63 | 0 / 0 |
| sf100 | nation | SNAPPY | n_regionkey | - | n_nationkey, n_name, n_comment | 1 / 4 | 0 / 0 |
| sf100 | region | SNAPPY | - | - | r_regionkey, r_name, r_comment | 0 / 3 | 0 / 0 |

## 3. Column-chunk size distributions  [M]

A *column chunk* is the byte range `[min(dictionary_page_offset, data_page_offset), +total_compressed_size)` of one column in one row group: dictionary page (if any) + the single data page, headers included. Sizes below are the **compressed on-disk** size unless labelled *uncompressed*. Percentiles use linear interpolation across row groups; the `min` of a column is almost always the short last row group.

### 3.1 Per table, all chunks pooled (sf100)  [M]

| table | chunks | min | p10 | p50 | p90 | p99 | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 78,144 | 2.5 KiB | 28.2 KiB | 90.8 KiB | 653.2 KiB | 1.5 MiB | 1.5 MiB |
| orders | 10,989 | 53 B | 53 B | 478.5 KiB | 2.3 MiB | 2.3 MiB | 2.3 MiB |
| partsupp | 3,260 | 8.8 KiB | 210.5 KiB | 480.7 KiB | 5.0 MiB | 5.0 MiB | 5.0 MiB |
| part | 1,467 | 34.7 KiB | 45.6 KiB | 121.7 KiB | 1.6 MiB | 1.6 MiB | 1.6 MiB |
| customer | 984 | 3.3 KiB | 45.6 KiB | 593.2 KiB | 3.3 MiB | 3.4 MiB | 3.4 MiB |
| supplier | 63 | 10.7 KiB | 75.6 KiB | 595.2 KiB | 3.3 MiB | 3.4 MiB | 3.4 MiB |
| nation | 4 | 80 B | 94 B | 199 B | 816 B | 1.0 KiB | 1.0 KiB |
| region | 3 | 44 B | 50 B | 75 B | 273 B | 317 B | 322 B |

Same chunks, **uncompressed** size (`total_uncompressed_size`: page headers + uncompressed payload) and the compression ratio (sf100):

| table | min | p10 | p50 | p90 | p99 | max | compression ratio | sum compressed | sum uncompressed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 3.7 KiB | 30.5 KiB | 90.9 KiB | 960.0 KiB | 3.6 MiB | 3.6 MiB | 1.87 | 21.16 GiB | 39.49 GiB |
| orders | 49 B | 49 B | 960.0 KiB | 6.1 MiB | 6.2 MiB | 6.2 MiB | 2.26 | 6.05 GiB | 13.65 GiB |
| partsupp | 40.0 KiB | 288.6 KiB | 960.0 KiB | 14.9 MiB | 15.0 MiB | 15.0 MiB | 2.81 | 4.09 GiB | 11.47 GiB |
| part | 34.7 KiB | 45.6 KiB | 124.1 KiB | 4.3 MiB | 4.3 MiB | 4.3 MiB | 2.16 | 603.42 MiB | 1.28 GiB |
| customer | 3.3 KiB | 45.6 KiB | 960.0 KiB | 9.0 MiB | 9.0 MiB | 9.0 MiB | 1.98 | 1.15 GiB | 2.28 GiB |
| supplier | 10.7 KiB | 75.6 KiB | 2.2 MiB | 7.8 MiB | 7.8 MiB | 7.8 MiB | 1.94 | 75.43 MiB | 146.03 MiB |
| nation | 125 B | 134 B | 228 B | 1.4 KiB | 1.9 KiB | 1.9 KiB | 1.68 | 1.49 KiB | 2.50 KiB |
| region | 43 B | 50 B | 77 B | 315 B | 369 B | 375 B | 1.12 | 441 B | 495 B |

Histogram of chunk sizes (sf100) - by **count** and by **bytes**:

| table | #<4 KiB | #4-64 KiB | #64 KiB-1 MiB | #1-16 MiB | #>=16 MiB | bytes <4 KiB | bytes 4-64 KiB | bytes 64 KiB-1 MiB | bytes 1-16 MiB | bytes >=16 MiB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 3 (0.0%) | 34,190 (43.8%) | 39,068 (50.0%) | 4,883 (6.2%) | 0 (0.0%) | 0.0% | 6.4% | 59.7% | 33.9% | 0.0% |
| orders | 1,221 (11.1%) | 2,442 (22.2%) | 6,105 (55.6%) | 1,221 (11.1%) | 0 (0.0%) | 0.0% | 1.5% | 52.8% | 45.7% | 0.0% |
| partsupp | 0 (0.0%) | 4 (0.1%) | 2,605 (79.9%) | 651 (20.0%) | 0 (0.0%) | 0.0% | 0.0% | 22.5% | 77.5% | 0.0% |
| part | 0 (0.0%) | 164 (11.2%) | 1,140 (77.7%) | 163 (11.1%) | 0 (0.0%) | 0.0% | 1.2% | 56.6% | 42.2% | 0.0% |
| customer | 1 (0.1%) | 126 (12.8%) | 491 (49.9%) | 366 (37.2%) | 0 (0.0%) | 0.0% | 0.5% | 17.7% | 81.9% | 0.0% |
| supplier | 0 (0.0%) | 1 (1.6%) | 38 (60.3%) | 24 (38.1%) | 0 (0.0%) | 0.0% | 0.0% | 19.8% | 80.2% | 0.0% |
| nation | 4 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| region | 3 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

All eight tables pooled (sf100): count <4 KiB 1,232 (1.3%), 4-64 KiB 36,927 (38.9%), 64 KiB-1 MiB 49,447 (52.1%), 1-16 MiB 7,308 (7.7%), >=16 MiB 0 (0.0%); bytes <4 KiB 0.0%, 4-64 KiB 4.4%, 64 KiB-1 MiB 52.3%, 1-16 MiB 43.4%, >=16 MiB 0.0%.

Row-group (all columns) on-disk size, i.e. the request size of an *unpruned* scan (sf100): lineitem p50 4.44 MiB (max 4.44 MiB); orders p50 5.08 MiB (max 5.08 MiB); partsupp p50 6.43 MiB (max 6.44 MiB); part p50 3.71 MiB (max 3.71 MiB); customer p50 9.67 MiB (max 9.69 MiB); supplier p50 9.27 MiB (max 9.28 MiB)

### 3.1b Per table, all chunks pooled (sf10)  [M]

| table | chunks | min | p10 | p50 | p90 | p99 | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 7,824 | 3.4 KiB | 28.2 KiB | 90.8 KiB | 619.8 KiB | 1.5 MiB | 1.5 MiB |
| orders | 1,107 | 53 B | 53 B | 263.6 KiB | 2.3 MiB | 2.3 MiB | 2.3 MiB |
| partsupp | 330 | 22.0 KiB | 210.5 KiB | 481.0 KiB | 5.0 MiB | 5.0 MiB | 5.0 MiB |
| part | 153 | 12.7 KiB | 45.6 KiB | 121.7 KiB | 1.5 MiB | 1.6 MiB | 1.6 MiB |
| customer | 104 | 9.6 KiB | 45.6 KiB | 593.1 KiB | 3.3 MiB | 3.4 MiB | 3.4 MiB |
| supplier | 7 | 61.6 KiB | 259.6 KiB | 485.8 KiB | 2.5 MiB | 2.7 MiB | 2.7 MiB |
| nation | 4 | 80 B | 94 B | 199 B | 816 B | 1.0 KiB | 1.0 KiB |
| region | 3 | 44 B | 50 B | 75 B | 273 B | 317 B | 322 B |

Same chunks, **uncompressed** size (`total_uncompressed_size`: page headers + uncompressed payload) and the compression ratio (sf10):

| table | min | p10 | p50 | p90 | p99 | max | compression ratio | sum compressed | sum uncompressed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 5.1 KiB | 30.5 KiB | 90.9 KiB | 960.0 KiB | 3.6 MiB | 3.6 MiB | 1.91 | 2.06 GiB | 3.95 GiB |
| orders | 49 B | 49 B | 396.1 KiB | 6.1 MiB | 6.2 MiB | 6.2 MiB | 2.12 | 554.55 MiB | 1.15 GiB |
| partsupp | 100.0 KiB | 288.6 KiB | 960.0 KiB | 14.9 MiB | 15.0 MiB | 15.0 MiB | 2.81 | 418.47 MiB | 1.15 GiB |
| part | 12.7 KiB | 45.6 KiB | 124.1 KiB | 3.9 MiB | 4.3 MiB | 4.3 MiB | 2.16 | 60.34 MiB | 130.58 MiB |
| customer | 9.6 KiB | 45.6 KiB | 960.0 KiB | 8.9 MiB | 9.0 MiB | 9.0 MiB | 1.98 | 118.03 MiB | 233.88 MiB |
| supplier | 61.6 KiB | 493.4 KiB | 1.8 MiB | 4.2 MiB | 6.1 MiB | 6.3 MiB | 1.94 | 7.54 MiB | 14.59 MiB |
| nation | 125 B | 134 B | 228 B | 1.4 KiB | 1.9 KiB | 1.9 KiB | 1.68 | 1.49 KiB | 2.50 KiB |
| region | 43 B | 50 B | 77 B | 315 B | 369 B | 375 B | 1.12 | 441 B | 495 B |

Histogram of chunk sizes (sf10) - by **count** and by **bytes**:

| table | #<4 KiB | #4-64 KiB | #64 KiB-1 MiB | #1-16 MiB | #>=16 MiB | bytes <4 KiB | bytes 4-64 KiB | bytes 64 KiB-1 MiB | bytes 1-16 MiB | bytes >=16 MiB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem | 1 (0.0%) | 3,427 (43.8%) | 3,908 (49.9%) | 488 (6.2%) | 0 (0.0%) | 0.0% | 6.5% | 58.7% | 34.7% | 0.0% |
| orders | 125 (11.3%) | 249 (22.5%) | 611 (55.2%) | 122 (11.0%) | 0 (0.0%) | 0.0% | 1.7% | 47.3% | 51.1% | 0.0% |
| partsupp | 0 (0.0%) | 4 (1.2%) | 261 (79.1%) | 65 (19.7%) | 0 (0.0%) | 0.0% | 0.0% | 22.6% | 77.4% | 0.0% |
| part | 0 (0.0%) | 21 (13.7%) | 116 (75.8%) | 16 (10.5%) | 0 (0.0%) | 0.0% | 1.4% | 57.1% | 41.5% | 0.0% |
| customer | 0 (0.0%) | 14 (13.5%) | 54 (51.9%) | 36 (34.6%) | 0 (0.0%) | 0.0% | 0.5% | 19.0% | 80.5% | 0.0% |
| supplier | 0 (0.0%) | 1 (14.3%) | 3 (42.9%) | 3 (42.9%) | 0 (0.0%) | 0.0% | 0.8% | 17.6% | 81.6% | 0.0% |
| nation | 4 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| region | 3 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

All eight tables pooled (sf10): count <4 KiB 133 (1.4%), 4-64 KiB 3,716 (39.0%), 64 KiB-1 MiB 4,953 (52.0%), 1-16 MiB 730 (7.7%), >=16 MiB 0 (0.0%); bytes <4 KiB 0.0%, 4-64 KiB 4.5%, 64 KiB-1 MiB 50.6%, 1-16 MiB 44.8%, >=16 MiB 0.0%.

Row-group (all columns) on-disk size, i.e. the request size of an *unpruned* scan (sf10): lineitem p50 4.33 MiB (max 4.33 MiB); orders p50 4.54 MiB (max 4.55 MiB); partsupp p50 6.43 MiB (max 6.44 MiB); part p50 3.71 MiB (max 3.71 MiB); customer p50 9.67 MiB (max 9.68 MiB); supplier p50 7.54 MiB (max 7.54 MiB)

### 3.2 Per column (sf100)  [M; the `Qs` column is A]

`dict` = fraction of row groups whose chunk has a dictionary page. `ratio` = sum(uncompressed)/sum(compressed). `share` = fraction of the table's on-disk bytes. `B/row` = compressed bytes per row. `Qs` = #queries (of 22) that reference the column [A].

**lineitem** (4,884 row groups, 600,037,902 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l_orderkey | INT64 / INT64 | plain | 22.7 KiB | 186.6 KiB | 187.1 KiB | 187.6 KiB | 187.9 KiB | 188.6 KiB | 960.0 KiB | 5.13 | 4.1% | 1.56 | 10 |
| l_partkey | INT64 / INT64 | plain | 79.1 KiB | 653.1 KiB | 653.2 KiB | 653.4 KiB | 653.5 KiB | 653.7 KiB | 960.0 KiB | 1.47 | 14.4% | 5.44 | 6 |
| l_suppkey | INT64 / INT64 | plain | 71.6 KiB | 592.0 KiB | 592.1 KiB | 592.2 KiB | 592.3 KiB | 592.4 KiB | 960.0 KiB | 1.62 | 13.0% | 4.93 | 7 |
| l_linenumber | INT64 / INT64 | dict 100% | 4.0 KiB | 29.3 KiB | 29.4 KiB | 29.6 KiB | 29.8 KiB | 30.7 KiB | 45.6 KiB | 1.55 | 0.6% | 0.25 | 0 |
| l_quantity | DECIMAL(15,2) / INT64 | dict 100% | 11.4 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.9 KiB | 1.00 | 2.0% | 0.76 | 7 |
| l_extendedprice | DECIMAL(15,2) / INT64 | plain | 75.1 KiB | 620.1 KiB | 620.3 KiB | 620.4 KiB | 620.5 KiB | 620.6 KiB | 960.0 KiB | 1.55 | 13.7% | 5.17 | 12 |
| l_discount | DECIMAL(15,2) / INT64 | dict 100% | 7.5 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 1.00 | 1.3% | 0.50 | 11 |
| l_tax | DECIMAL(15,2) / INT64 | dict 100% | 7.5 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 1.00 | 1.3% | 0.50 | 1 |
| l_returnflag | STRING / BYTE_ARRAY | dict 100% | 3.5 KiB | 28.1 KiB | 28.2 KiB | 28.3 KiB | 28.4 KiB | 30.5 KiB | 30.5 KiB | 1.08 | 0.6% | 0.24 | 2 |
| l_linestatus | STRING / BYTE_ARRAY | dict 100% | 2.5 KiB | 18.7 KiB | 18.8 KiB | 18.9 KiB | 19.0 KiB | 19.1 KiB | 30.1 KiB | 1.60 | 0.4% | 0.16 | 1 |
| l_shipdate | DATE / INT32 | dict 100% | 31.9 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 1.00 | 4.2% | 1.59 | 8 |
| l_commitdate | DATE / INT32 | dict 100% | 31.7 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.1 KiB | 1.00 | 4.2% | 1.58 | 3 |
| l_receiptdate | DATE / INT32 | dict 100% | 31.9 KiB | 190.4 KiB | 190.4 KiB | 190.5 KiB | 190.5 KiB | 190.5 KiB | 190.4 KiB | 1.00 | 4.2% | 1.59 | 3 |
| l_shipinstruct | STRING / BYTE_ARRAY | dict 100% | 5.7 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 1.00 | 1.0% | 0.38 | 1 |
| l_shipmode | STRING / BYTE_ARRAY | dict 100% | 5.7 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 1.00 | 1.0% | 0.38 | 2 |
| l_comment | STRING / BYTE_ARRAY | plain | 187.2 KiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 3.6 MiB | 2.38 | 33.9% | 12.84 | 0 |

**orders** (1,221 row groups, 150,000,000 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o_orderkey | INT64 / INT64 | plain | 336.4 KiB | 478.4 KiB | 478.5 KiB | 478.6 KiB | 478.6 KiB | 478.9 KiB | 960.0 KiB | 2.01 | 9.2% | 3.99 | 11 |
| o_custkey | INT64 / INT64 | plain | 450.7 KiB | 640.7 KiB | 640.9 KiB | 641.0 KiB | 641.1 KiB | 641.2 KiB | 960.0 KiB | 1.50 | 12.3% | 5.34 | 8 |
| o_orderstatus | STRING / BYTE_ARRAY | dict 100% | 21.5 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 1.00 | 0.6% | 0.25 | 1 |
| o_totalprice | DECIMAL(15,2) / INT64 | plain | 472.6 KiB | 671.9 KiB | 672.1 KiB | 672.2 KiB | 672.4 KiB | 672.6 KiB | 960.0 KiB | 1.43 | 12.9% | 5.60 | 1 |
| o_orderdate | DATE / INT32 | dict 100% | 136.5 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 1.00 | 3.7% | 1.58 | 7 |
| o_orderpriority | STRING / BYTE_ARRAY | dict 100% | 32.1 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 1.00 | 0.9% | 0.38 | 2 |
| o_clerk | STRING / BYTE_ARRAY | plain | 538.0 KiB | 765.4 KiB | 765.8 KiB | 766.1 KiB | 766.4 KiB | 766.7 KiB | 2.2 MiB | 2.98 | 14.7% | 6.38 | 0 |
| o_shippriority | INT32 / INT32 | dict 100% | 53 B | 53 B | 53 B | 53 B | 53 B | 53 B | 49 B | 0.92 | 0.0% | 0.00 | 1 |
| o_comment | STRING / BYTE_ARRAY | plain | 1.6 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 6.2 MiB | 2.65 | 45.7% | 19.81 | 1 |

**partsupp** (652 row groups, 80,000,000 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ps_partkey | INT64 / INT64 | plain | 8.8 KiB | 210.5 KiB | 210.5 KiB | 210.6 KiB | 211.0 KiB | 211.2 KiB | 960.0 KiB | 4.56 | 3.2% | 1.75 | 5 |
| ps_suppkey | INT64 / INT64 | plain | 20.1 KiB | 480.7 KiB | 480.7 KiB | 480.8 KiB | 480.8 KiB | 481.0 KiB | 960.0 KiB | 2.00 | 7.3% | 4.01 | 5 |
| ps_availqty | INT64 / INT64 | mixed (99.8% dict) | 19.0 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 288.6 KiB | 1.16 | 3.8% | 2.08 | 2 |
| ps_supplycost | DECIMAL(15,2) / INT64 | plain | 22.5 KiB | 538.3 KiB | 538.5 KiB | 538.7 KiB | 538.8 KiB | 538.9 KiB | 960.0 KiB | 1.78 | 8.2% | 4.49 | 3 |
| ps_comment | STRING / BYTE_ARRAY | plain | 215.2 KiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 14.9 MiB | 3.00 | 77.5% | 42.52 | 0 |

**part** (163 row groups, 20,000,000 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p_partkey | INT64 / INT64 | plain | 365.2 KiB | 480.3 KiB | 480.3 KiB | 480.3 KiB | 480.8 KiB | 481.0 KiB | 960.0 KiB | 2.00 | 12.7% | 4.00 | 8 |
| p_name | STRING / BYTE_ARRAY | plain | 1.2 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 4.3 MiB | 2.75 | 42.2% | 13.36 | 2 |
| p_mfgr | STRING / BYTE_ARRAY | dict 100% | 34.7 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 1.00 | 1.2% | 0.38 | 1 |
| p_brand | STRING / BYTE_ARRAY | dict 100% | 57.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.8 KiB | 1.00 | 2.0% | 0.63 | 3 |
| p_type | STRING / BYTE_ARRAY | dict 100% | 92.8 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 124.1 KiB | 1.02 | 3.2% | 1.01 | 4 |
| p_size | INT32 / INT32 | dict 100% | 69.0 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 1.00 | 2.4% | 0.76 | 3 |
| p_container | STRING / BYTE_ARRAY | dict 100% | 69.1 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 91.0 KiB | 1.00 | 2.4% | 0.76 | 2 |
| p_retailprice | DECIMAL(15,2) / INT64 | dict 100% | 204.2 KiB | 267.2 KiB | 271.1 KiB | 271.2 KiB | 271.2 KiB | 271.2 KiB | 320.6 KiB | 1.18 | 7.1% | 2.25 | 0 |
| p_comment | STRING / BYTE_ARRAY | plain | 774.7 KiB | 1017.9 KiB | 1019.1 KiB | 1020.0 KiB | 1020.8 KiB | 1021.1 KiB | 2.1 MiB | 2.06 | 26.8% | 8.49 | 0 |

**customer** (123 row groups, 15,000,000 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c_custkey | INT64 / INT64 | plain | 33.8 KiB | 480.3 KiB | 480.3 KiB | 480.3 KiB | 480.8 KiB | 481.0 KiB | 960.0 KiB | 2.00 | 4.9% | 4.00 | 8 |
| c_name | STRING / BYTE_ARRAY | plain | 42.3 KiB | 595.4 KiB | 596.0 KiB | 596.3 KiB | 596.7 KiB | 596.8 KiB | 2.6 MiB | 4.43 | 6.0% | 4.97 | 2 |
| c_address | STRING / BYTE_ARRAY | plain | 238.6 KiB | 3.3 MiB | 3.3 MiB | 3.4 MiB | 3.4 MiB | 3.4 MiB | 3.4 MiB | 1.01 | 34.6% | 28.59 | 1 |
| c_nationkey | INT32 / INT32 | dict 100% | 5.5 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 1.00 | 0.8% | 0.63 | 4 |
| c_phone | STRING / BYTE_ARRAY | plain | 100.7 KiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 2.2 MiB | 1.60 | 14.4% | 11.86 | 2 |
| c_acctbal | DECIMAL(15,2) / INT64 | plain | 41.8 KiB | 592.9 KiB | 593.0 KiB | 593.1 KiB | 593.2 KiB | 593.3 KiB | 960.0 KiB | 1.62 | 6.0% | 4.94 | 2 |
| c_mktsegment | STRING / BYTE_ARRAY | dict 100% | 3.3 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 1.00 | 0.5% | 0.38 | 1 |
| c_comment | STRING / BYTE_ARRAY | plain | 229.0 KiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 9.0 MiB | 2.82 | 32.9% | 27.15 | 1 |

**supplier** (9 row groups, 1,000,000 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s_suppkey | INT64 / INT64 | plain | 66.3 KiB | 397.5 KiB | 480.3 KiB | 480.4 KiB | 481.0 KiB | 481.0 KiB | 960.0 KiB | 2.00 | 5.1% | 4.00 | 10 |
| s_name | STRING / BYTE_ARRAY | plain | 81.9 KiB | 492.5 KiB | 596.0 KiB | 596.1 KiB | 596.2 KiB | 596.3 KiB | 2.6 MiB | 4.43 | 6.3% | 4.97 | 4 |
| s_address | STRING / BYTE_ARRAY | plain | 473.5 KiB | 2.8 MiB | 3.3 MiB | 3.4 MiB | 3.4 MiB | 3.4 MiB | 3.4 MiB | 1.01 | 36.1% | 28.58 | 3 |
| s_nationkey | INT32 / INT32 | dict 100% | 10.7 KiB | 62.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 1.00 | 0.8% | 0.63 | 8 |
| s_phone | STRING / BYTE_ARRAY | plain | 196.6 KiB | 1.1 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 2.2 MiB | 1.60 | 15.0% | 11.86 | 2 |
| s_acctbal | DECIMAL(15,2) / INT64 | plain | 81.9 KiB | 490.7 KiB | 593.0 KiB | 593.1 KiB | 593.2 KiB | 593.2 KiB | 960.0 KiB | 1.62 | 6.2% | 4.94 | 1 |
| s_comment | STRING / BYTE_ARRAY | plain | 398.0 KiB | 2.3 MiB | 2.8 MiB | 2.8 MiB | 2.8 MiB | 2.8 MiB | 7.8 MiB | 2.76 | 30.5% | 24.12 | 2 |

**nation** (1 row group, 25 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n_nationkey | INT32 / INT32 | plain | 125 B | 125 B | 125 B | 125 B | 125 B | 125 B | 125 B | 1.00 | 8.2% | 5.00 | 9 |
| n_name | STRING / BYTE_ARRAY | plain | 273 B | 273 B | 273 B | 273 B | 273 B | 273 B | 302 B | 1.11 | 17.9% | 10.92 | 9 |
| n_regionkey | INT32 / INT32 | dict 100% | 80 B | 80 B | 80 B | 80 B | 80 B | 80 B | 155 B | 1.94 | 5.2% | 3.20 | 3 |
| n_comment | STRING / BYTE_ARRAY | plain | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.9 KiB | 1.89 | 68.7% | 41.96 | 0 |

**region** (1 row group, 5 rows)

| column | type | encoding | min | p10 | p50 | p90 | p99 | max | uncompr. p50 | ratio | share | B/row | Qs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r_regionkey | INT32 / INT32 | plain | 44 B | 44 B | 44 B | 44 B | 44 B | 44 B | 43 B | 0.98 | 10.0% | 8.80 | 3 |
| r_name | STRING / BYTE_ARRAY | plain | 75 B | 75 B | 75 B | 75 B | 75 B | 75 B | 77 B | 1.03 | 17.0% | 15.00 | 3 |
| r_comment | STRING / BYTE_ARRAY | plain | 322 B | 322 B | 322 B | 322 B | 322 B | 322 B | 375 B | 1.16 | 73.0% | 64.40 | 0 |

### 3.2a Per column size histogram (sf100)  [M]

Chunk count per size class, with the share of the column's bytes in that class in brackets (a dash = empty class). Chunk sizes of a column are nearly constant across row groups, so each column sits in one class except for its short last row group.

| column | <4 KiB | 4-64 KiB | 64 KiB-1 MiB | 1-16 MiB | >=16 MiB |
| :--- | ---: | ---: | ---: | ---: | ---: |
| lineitem.l_orderkey | - | 1 [0.0%] | 4,883 [100.0%] | - | - |
| lineitem.l_partkey | - | - | 4,884 [100.0%] | - | - |
| lineitem.l_suppkey | - | - | 4,884 [100.0%] | - | - |
| lineitem.l_linenumber | 1 [0.0%] | 4,883 [100.0%] | - | - | - |
| lineitem.l_quantity | - | 1 [0.0%] | 4,883 [100.0%] | - | - |
| lineitem.l_extendedprice | - | - | 4,884 [100.0%] | - | - |
| lineitem.l_discount | - | 4,884 [100.0%] | - | - | - |
| lineitem.l_tax | - | 4,884 [100.0%] | - | - | - |
| lineitem.l_returnflag | 1 [0.0%] | 4,883 [100.0%] | - | - | - |
| lineitem.l_linestatus | 1 [0.0%] | 4,883 [100.0%] | - | - | - |
| lineitem.l_shipdate | - | 1 [0.0%] | 4,883 [100.0%] | - | - |
| lineitem.l_commitdate | - | 1 [0.0%] | 4,883 [100.0%] | - | - |
| lineitem.l_receiptdate | - | 1 [0.0%] | 4,883 [100.0%] | - | - |
| lineitem.l_shipinstruct | - | 4,884 [100.0%] | - | - | - |
| lineitem.l_shipmode | - | 4,884 [100.0%] | - | - | - |
| lineitem.l_comment | - | - | 1 [0.0%] | 4,883 [100.0%] | - |
| orders.o_orderkey | - | - | 1,221 [100.0%] | - | - |
| orders.o_custkey | - | - | 1,221 [100.0%] | - | - |
| orders.o_orderstatus | - | 1,221 [100.0%] | - | - | - |
| orders.o_totalprice | - | - | 1,221 [100.0%] | - | - |
| orders.o_orderdate | - | - | 1,221 [100.0%] | - | - |
| orders.o_orderpriority | - | 1,221 [100.0%] | - | - | - |
| orders.o_clerk | - | - | 1,221 [100.0%] | - | - |
| orders.o_shippriority | 1,221 [100.0%] | - | - | - | - |
| orders.o_comment | - | - | - | 1,221 [100.0%] | - |
| partsupp.ps_partkey | - | 1 [0.0%] | 651 [100.0%] | - | - |
| partsupp.ps_suppkey | - | 1 [0.0%] | 651 [100.0%] | - | - |
| partsupp.ps_availqty | - | 1 [0.0%] | 651 [100.0%] | - | - |
| partsupp.ps_supplycost | - | 1 [0.0%] | 651 [100.0%] | - | - |
| partsupp.ps_comment | - | - | 1 [0.0%] | 651 [100.0%] | - |
| part.p_partkey | - | - | 163 [100.0%] | - | - |
| part.p_name | - | - | - | 163 [100.0%] | - |
| part.p_mfgr | - | 163 [100.0%] | - | - | - |
| part.p_brand | - | 1 [0.5%] | 162 [99.5%] | - | - |
| part.p_type | - | - | 163 [100.0%] | - | - |
| part.p_size | - | - | 163 [100.0%] | - | - |
| part.p_container | - | - | 163 [100.0%] | - | - |
| part.p_retailprice | - | - | 163 [100.0%] | - | - |
| part.p_comment | - | - | 163 [100.0%] | - | - |
| customer.c_custkey | - | 1 [0.1%] | 122 [99.9%] | - | - |
| customer.c_name | - | 1 [0.1%] | 122 [99.9%] | - | - |
| customer.c_address | - | - | 1 [0.1%] | 122 [99.9%] | - |
| customer.c_nationkey | - | 1 [0.1%] | 122 [99.9%] | - | - |
| customer.c_phone | - | - | 1 [0.1%] | 122 [99.9%] | - |
| customer.c_acctbal | - | 1 [0.1%] | 122 [99.9%] | - | - |
| customer.c_mktsegment | 1 [0.1%] | 122 [99.9%] | - | - | - |
| customer.c_comment | - | - | 1 [0.1%] | 122 [99.9%] | - |
| supplier.s_suppkey | - | - | 9 [100.0%] | - | - |
| supplier.s_name | - | - | 9 [100.0%] | - | - |
| supplier.s_address | - | - | 1 [1.7%] | 8 [98.3%] | - |
| supplier.s_nationkey | - | 1 [1.7%] | 8 [98.3%] | - | - |
| supplier.s_phone | - | - | 1 [1.7%] | 8 [98.3%] | - |
| supplier.s_acctbal | - | - | 9 [100.0%] | - | - |
| supplier.s_comment | - | - | 1 [1.7%] | 8 [98.3%] | - |
| nation.n_nationkey | 1 [100.0%] | - | - | - | - |
| nation.n_name | 1 [100.0%] | - | - | - | - |
| nation.n_regionkey | 1 [100.0%] | - | - | - | - |
| nation.n_comment | 1 [100.0%] | - | - | - | - |
| region.r_regionkey | 1 [100.0%] | - | - | - | - |
| region.r_name | 1 [100.0%] | - | - | - | - |
| region.r_comment | 1 [100.0%] | - | - | - | - |

### 3.2b Per column (sf10; compressed chunk size)  [M]

Because the row-group size (122,880 rows) is the same at both scale factors, interior chunk sizes are nearly identical; sf10 simply has ~10x fewer row groups. Differences come from data-dependent compression (key magnitudes) and from the dictionary/plain decision (e.g. `o_clerk` is dictionary-encoded at sf10 - 10,000 distinct clerks - but PLAIN at sf100 - 100,000 distinct).

| column | chunks | encoding | min | p10 | p50 | p90 | p99 | max | sf100 p50 | sf10 vs sf100 p50 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem.l_orderkey | 489 | plain | 31.1 KiB | 186.6 KiB | 187.1 KiB | 187.6 KiB | 187.9 KiB | 188.2 KiB | 187.1 KiB | +0.0% |
| lineitem.l_partkey | 489 | plain | 100.2 KiB | 596.9 KiB | 597.0 KiB | 597.1 KiB | 597.2 KiB | 597.2 KiB | 653.2 KiB | -8.6% |
| lineitem.l_suppkey | 489 | plain | 90.5 KiB | 538.2 KiB | 538.4 KiB | 538.6 KiB | 538.8 KiB | 539.0 KiB | 592.1 KiB | -9.1% |
| lineitem.l_linenumber | 489 | dict 100% | 5.4 KiB | 29.3 KiB | 29.4 KiB | 29.6 KiB | 29.8 KiB | 29.9 KiB | 29.4 KiB | +0.0% |
| lineitem.l_quantity | 489 | dict 100% | 15.6 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | -0.0% |
| lineitem.l_extendedprice | 489 | plain | 104.0 KiB | 619.7 KiB | 619.8 KiB | 620.0 KiB | 620.1 KiB | 620.2 KiB | 620.3 KiB | -0.1% |
| lineitem.l_discount | 489 | dict 100% | 10.3 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | +0.0% |
| lineitem.l_tax | 489 | dict 100% | 10.3 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | 60.6 KiB | +0.0% |
| lineitem.l_returnflag | 489 | dict 100% | 4.8 KiB | 28.1 KiB | 28.2 KiB | 28.3 KiB | 28.4 KiB | 28.6 KiB | 28.2 KiB | +0.0% |
| lineitem.l_linestatus | 489 | dict 100% | 3.4 KiB | 18.7 KiB | 18.8 KiB | 18.9 KiB | 19.0 KiB | 19.1 KiB | 18.8 KiB | +0.0% |
| lineitem.l_shipdate | 489 | dict 100% | 40.3 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | 190.4 KiB | +0.0% |
| lineitem.l_commitdate | 489 | dict 100% | 40.1 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | 190.2 KiB | -0.0% |
| lineitem.l_receiptdate | 489 | dict 100% | 40.3 KiB | 190.4 KiB | 190.4 KiB | 190.5 KiB | 190.5 KiB | 190.5 KiB | 190.4 KiB | +0.0% |
| lineitem.l_shipinstruct | 489 | dict 100% | 7.8 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | +0.0% |
| lineitem.l_shipmode | 489 | dict 100% | 7.8 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | +0.0% |
| lineitem.l_comment | 489 | plain | 258.7 KiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | 1.5 MiB | -0.0% |
| orders.o_orderkey | 123 | plain | 33.7 KiB | 478.4 KiB | 478.5 KiB | 478.6 KiB | 478.6 KiB | 478.7 KiB | 478.5 KiB | +0.0% |
| orders.o_custkey | 123 | plain | 41.9 KiB | 594.7 KiB | 594.8 KiB | 594.9 KiB | 595.0 KiB | 595.0 KiB | 640.9 KiB | -7.2% |
| orders.o_orderstatus | 123 | dict 100% | 2.2 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | 30.6 KiB | -0.0% |
| orders.o_totalprice | 123 | plain | 47.3 KiB | 671.9 KiB | 672.1 KiB | 672.3 KiB | 672.4 KiB | 672.4 KiB | 672.1 KiB | +0.0% |
| orders.o_orderdate | 123 | mixed (99.2% dict) | 26.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | 189.9 KiB | +0.0% |
| orders.o_orderpriority | 123 | dict 100% | 3.3 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | +0.0% |
| orders.o_clerk | 123 | mixed (99.2% dict) | 44.0 KiB | 263.5 KiB | 263.6 KiB | 263.7 KiB | 263.7 KiB | 263.8 KiB | 765.8 KiB | -65.6% |
| orders.o_shippriority | 123 | dict 100% | 53 B | 53 B | 53 B | 53 B | 53 B | 53 B | 53 B | +0.0% |
| orders.o_comment | 123 | plain | 168.3 KiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | -0.0% |
| partsupp.ps_partkey | 66 | plain | 22.0 KiB | 210.5 KiB | 210.5 KiB | 210.6 KiB | 210.8 KiB | 211.2 KiB | 210.5 KiB | -0.0% |
| partsupp.ps_suppkey | 66 | plain | 50.1 KiB | 480.7 KiB | 481.0 KiB | 481.0 KiB | 481.1 KiB | 481.1 KiB | 480.7 KiB | +0.0% |
| partsupp.ps_availqty | 66 | mixed (98.5% dict) | 46.8 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | 249.7 KiB | -0.0% |
| partsupp.ps_supplycost | 66 | plain | 56.2 KiB | 538.3 KiB | 538.5 KiB | 538.6 KiB | 538.9 KiB | 538.9 KiB | 538.5 KiB | -0.0% |
| partsupp.ps_comment | 66 | plain | 531.7 KiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | -0.0% |
| part.p_partkey | 17 | plain | 132.6 KiB | 480.3 KiB | 480.3 KiB | 480.3 KiB | 480.9 KiB | 481.0 KiB | 480.3 KiB | -0.0% |
| part.p_name | 17 | plain | 442.7 KiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | 1.6 MiB | +0.0% |
| part.p_mfgr | 17 | dict 100% | 12.7 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | -0.0% |
| part.p_brand | 17 | dict 100% | 21.1 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | +0.0% |
| part.p_type | 17 | dict 100% | 34.6 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | 121.7 KiB | -0.0% |
| part.p_size | 17 | dict 100% | 25.3 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | 90.7 KiB | +0.0% |
| part.p_container | 17 | dict 100% | 25.4 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | 90.8 KiB | -0.0% |
| part.p_retailprice | 17 | dict 100% | 72.7 KiB | 267.2 KiB | 271.1 KiB | 271.1 KiB | 271.2 KiB | 271.2 KiB | 271.1 KiB | -0.0% |
| part.p_comment | 17 | plain | 281.4 KiB | 1018.4 KiB | 1019.2 KiB | 1019.9 KiB | 1020.9 KiB | 1021.1 KiB | 1019.1 KiB | +0.0% |
| customer.c_custkey | 13 | plain | 99.5 KiB | 480.3 KiB | 480.3 KiB | 480.3 KiB | 480.9 KiB | 481.0 KiB | 480.3 KiB | -0.0% |
| customer.c_name | 13 | plain | 124.0 KiB | 595.0 KiB | 596.0 KiB | 596.2 KiB | 596.3 KiB | 596.3 KiB | 596.0 KiB | +0.0% |
| customer.c_address | 13 | plain | 706.7 KiB | 3.3 MiB | 3.3 MiB | 3.4 MiB | 3.4 MiB | 3.4 MiB | 3.3 MiB | -0.0% |
| customer.c_nationkey | 13 | dict 100% | 15.9 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | 75.6 KiB | +0.0% |
| customer.c_phone | 13 | plain | 295.1 KiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | 1.4 MiB | -0.0% |
| customer.c_acctbal | 13 | plain | 122.8 KiB | 592.8 KiB | 593.0 KiB | 593.1 KiB | 593.2 KiB | 593.2 KiB | 593.0 KiB | +0.0% |
| customer.c_mktsegment | 13 | dict 100% | 9.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | 45.6 KiB | -0.0% |
| customer.c_comment | 13 | plain | 673.8 KiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | 3.2 MiB | +0.0% |
| supplier.s_suppkey | 1 | plain | 391.6 KiB | 391.6 KiB | 391.6 KiB | 391.6 KiB | 391.6 KiB | 391.6 KiB | 480.3 KiB | -18.5% |
| supplier.s_name | 1 | plain | 485.8 KiB | 485.8 KiB | 485.8 KiB | 485.8 KiB | 485.8 KiB | 485.8 KiB | 596.0 KiB | -18.5% |
| supplier.s_address | 1 | plain | 2.7 MiB | 2.7 MiB | 2.7 MiB | 2.7 MiB | 2.7 MiB | 2.7 MiB | 3.3 MiB | -18.8% |
| supplier.s_nationkey | 1 | dict 100% | 61.6 KiB | 61.6 KiB | 61.6 KiB | 61.6 KiB | 61.6 KiB | 61.6 KiB | 75.6 KiB | -18.5% |
| supplier.s_phone | 1 | plain | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.1 MiB | 1.4 MiB | -18.7% |
| supplier.s_acctbal | 1 | plain | 482.6 KiB | 482.6 KiB | 482.6 KiB | 482.6 KiB | 482.6 KiB | 482.6 KiB | 593.0 KiB | -18.6% |
| supplier.s_comment | 1 | plain | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.3 MiB | 2.8 MiB | -18.7% |
| nation.n_nationkey | 1 | plain | 125 B | 125 B | 125 B | 125 B | 125 B | 125 B | 125 B | +0.0% |
| nation.n_name | 1 | plain | 273 B | 273 B | 273 B | 273 B | 273 B | 273 B | 273 B | +0.0% |
| nation.n_regionkey | 1 | dict 100% | 80 B | 80 B | 80 B | 80 B | 80 B | 80 B | 80 B | +0.0% |
| nation.n_comment | 1 | plain | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | 1.0 KiB | +0.0% |
| region.r_regionkey | 1 | plain | 44 B | 44 B | 44 B | 44 B | 44 B | 44 B | 44 B | +0.0% |
| region.r_name | 1 | plain | 75 B | 75 B | 75 B | 75 B | 75 B | 75 B | 75 B | +0.0% |
| region.r_comment | 1 | plain | 322 B | 322 B | 322 B | 322 B | 322 B | 322 B | 322 B | +0.0% |

### 3.3 Dictionary-page vs data-page split, and page sizes (sf100)  [M]

Exact (every row group's page headers were walked: sf100 lineitem 131,868 pages in 78,144 chunks; all chunks tile exactly: sum of page header + compressed page sizes == `total_compressed_size`; first data page == `data_page_offset`; dictionary page present iff `dictionary_page_offset` is set; 0 exceptions in all 16 files). A dictionary page occupies `[dictionary_page_offset, data_page_offset)`; **there is exactly one data page per chunk, so 'data page size' == chunk size minus the dictionary page**; no offset index exists, none is needed to locate pages. Sizes include the page header.

| column | encoding | dictionary page p50 (min-max) | dict entries p50 | dict page / chunk | data page (compressed) p50 | data page (uncompressed) p50 | data hdr B | data pages / chunk |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lineitem.l_orderkey | plain | - | - | - | 187.0 KiB | 960.0 KiB | 23 | 1 |
| lineitem.l_partkey | plain | - | - | - | 653.2 KiB | 960.0 KiB | 23 | 1 |
| lineitem.l_suppkey | plain | - | - | - | 592.1 KiB | 960.0 KiB | 23 | 1 |
| lineitem.l_linenumber | dict 100% | 53 B (53 B-53 B) | 7 | 0.18% | 29.4 KiB | 45.5 KiB | 23 | 1 |
| lineitem.l_quantity | dict 100% | 252 B (246 B-257 B) | 50 | 0.27% | 90.5 KiB | 90.5 KiB | 23 | 1 |
| lineitem.l_extendedprice | plain | - | - | - | 620.2 KiB | 960.0 KiB | 23 | 1 |
| lineitem.l_discount | dict 100% | 68 B (64 B-70 B) | 11 | 0.11% | 60.5 KiB | 60.5 KiB | 23 | 1 |
| lineitem.l_tax | dict 100% | 60 B (56 B-62 B) | 9 | 0.10% | 60.5 KiB | 60.5 KiB | 23 | 1 |
| lineitem.l_returnflag | dict 100% | 30 B (30 B-30 B) | 3 | 0.10% | 28.1 KiB | 30.5 KiB | 23 | 1 |
| lineitem.l_linestatus | dict 100% | 25 B (25 B-25 B) | 2 | 0.13% | 18.8 KiB | 30.1 KiB | 23 | 1 |
| lineitem.l_shipdate | dict 100% | 9.9 KiB (9.7 KiB-9.9 KiB) | 2,522 | 5.19% | 180.5 KiB | 180.5 KiB | 23 | 1 |
| lineitem.l_commitdate | dict 100% | 9.7 KiB (9.5 KiB-9.7 KiB) | 2,464 | 5.07% | 180.5 KiB | 180.5 KiB | 23 | 1 |
| lineitem.l_receiptdate | dict 100% | 9.9 KiB (9.7 KiB-10.0 KiB) | 2,536 | 5.21% | 180.5 KiB | 180.5 KiB | 23 | 1 |
| lineitem.l_shipinstruct | dict 100% | 82 B (82 B-82 B) | 4 | 0.18% | 45.5 KiB | 45.5 KiB | 23 | 1 |
| lineitem.l_shipmode | dict 100% | 71 B (63 B-72 B) | 7 | 0.15% | 45.5 KiB | 45.5 KiB | 23 | 1 |
| lineitem.l_comment | plain | - | - | - | 1.5 MiB | 3.6 MiB | 25 | 1 |
| orders.o_orderkey | plain | - | - | - | 478.4 KiB | 960.0 KiB | 23 | 1 |
| orders.o_custkey | plain | - | - | - | 640.8 KiB | 960.0 KiB | 23 | 1 |
| orders.o_orderstatus | dict 100% | 30 B (30 B-30 B) | 3 | 0.10% | 30.5 KiB | 30.5 KiB | 23 | 1 |
| orders.o_totalprice | plain | - | - | - | 672.0 KiB | 960.0 KiB | 23 | 1 |
| orders.o_orderdate | dict 100% | 9.4 KiB (9.4 KiB-9.4 KiB) | 2,406 | 4.96% | 180.5 KiB | 180.5 KiB | 23 | 1 |
| orders.o_orderpriority | dict 100% | 79 B (76 B-79 B) | 5 | 0.17% | 45.5 KiB | 45.5 KiB | 23 | 1 |
| orders.o_clerk | plain | - | - | - | 765.7 KiB | 2.2 MiB | 24 | 1 |
| orders.o_shippriority | dict 100% | 19 B (19 B-19 B) | 1 | 35.85% | 15 B | 13 B | 19 | 1 |
| orders.o_comment | plain | - | - | - | 2.3 MiB | 6.2 MiB | 25 | 1 |
| partsupp.ps_partkey | plain | - | - | - | 210.5 KiB | 960.0 KiB | 23 | 1 |
| partsupp.ps_suppkey | plain | - | - | - | 480.7 KiB | 960.0 KiB | 23 | 1 |
| partsupp.ps_availqty | mixed (99.8% dict) | 39.2 KiB (39.1 KiB-39.2 KiB) | 9,999 | 15.69% | 210.5 KiB | 210.5 KiB | 23 | 1 |
| partsupp.ps_supplycost | plain | - | - | - | 538.5 KiB | 960.0 KiB | 23 | 1 |
| partsupp.ps_comment | plain | - | - | - | 5.0 MiB | 14.9 MiB | 25 | 1 |
| part.p_partkey | plain | - | - | - | 480.3 KiB | 960.0 KiB | 23 | 1 |
| part.p_name | plain | - | - | - | 1.6 MiB | 4.3 MiB | 25 | 1 |
| part.p_mfgr | dict 100% | 54 B (54 B-54 B) | 5 | 0.12% | 45.5 KiB | 45.5 KiB | 23 | 1 |
| part.p_brand | dict 100% | 135 B (131 B-139 B) | 25 | 0.17% | 75.5 KiB | 75.5 KiB | 23 | 1 |
| part.p_type | dict 100% | 1.2 KiB (1.1 KiB-1.2 KiB) | 150 | 0.96% | 120.5 KiB | 120.5 KiB | 23 | 1 |
| part.p_size | dict 100% | 219 B (219 B-219 B) | 50 | 0.24% | 90.5 KiB | 90.5 KiB | 23 | 1 |
| part.p_container | dict 100% | 280 B (255 B-305 B) | 40 | 0.30% | 90.5 KiB | 90.5 KiB | 23 | 1 |
| part.p_retailprice | dict 100% | 60.7 KiB (44.1 KiB-60.7 KiB) | 14,087 | 22.38% | 210.4 KiB | 210.5 KiB | 23 | 1 |
| part.p_comment | plain | - | - | - | 1019.1 KiB | 2.1 MiB | 24 | 1 |
| customer.c_custkey | plain | - | - | - | 480.3 KiB | 960.0 KiB | 23 | 1 |
| customer.c_name | plain | - | - | - | 595.9 KiB | 2.6 MiB | 24 | 1 |
| customer.c_address | plain | - | - | - | 3.3 MiB | 3.4 MiB | 25 | 1 |
| customer.c_nationkey | dict 100% | 115 B (115 B-118 B) | 25 | 0.15% | 75.5 KiB | 75.5 KiB | 23 | 1 |
| customer.c_phone | plain | - | - | - | 1.4 MiB | 2.2 MiB | 25 | 1 |
| customer.c_acctbal | plain | - | - | - | 593.0 KiB | 960.0 KiB | 23 | 1 |
| customer.c_mktsegment | dict 100% | 81 B (79 B-81 B) | 5 | 0.17% | 45.5 KiB | 45.5 KiB | 23 | 1 |
| customer.c_comment | plain | - | - | - | 3.2 MiB | 9.0 MiB | 25 | 1 |
| supplier.s_suppkey | plain | - | - | - | 480.3 KiB | 960.0 KiB | 23 | 1 |
| supplier.s_name | plain | - | - | - | 596.0 KiB | 2.6 MiB | 24 | 1 |
| supplier.s_address | plain | - | - | - | 3.3 MiB | 3.4 MiB | 25 | 1 |
| supplier.s_nationkey | dict 100% | 117 B (115 B-118 B) | 25 | 0.15% | 75.5 KiB | 75.5 KiB | 23 | 1 |
| supplier.s_phone | plain | - | - | - | 1.4 MiB | 2.2 MiB | 25 | 1 |
| supplier.s_acctbal | plain | - | - | - | 593.0 KiB | 960.0 KiB | 23 | 1 |
| supplier.s_comment | plain | - | - | - | 2.8 MiB | 7.8 MiB | 25 | 1 |
| nation.n_nationkey | plain | - | - | - | 106 B | 106 B | 19 | 1 |
| nation.n_name | plain | - | - | - | 254 B | 283 B | 19 | 1 |
| nation.n_regionkey | dict 100% | 35 B (35 B-35 B) | 5 | 43.75% | 27 B | 104 B | 18 | 1 |
| nation.n_comment | plain | - | - | - | 1.0 KiB | 1.9 KiB | 19 | 1 |
| region.r_regionkey | plain | - | - | - | 27 B | 26 B | 17 | 1 |
| region.r_name | plain | - | - | - | 58 B | 60 B | 17 | 1 |
| region.r_comment | plain | - | - | - | 303 B | 356 B | 19 | 1 |

Take-away: dictionary pages are small (median size from 19 B up to 60.7 KiB: largest are part.p_retailprice 60.7 KiB (14,087 entries); partsupp.ps_availqty 39.2 KiB (9,999 entries); lineitem.l_receiptdate 9.9 KiB (2,536 entries); lineitem.l_shipdate 9.9 KiB (2,522 entries); lineitem.l_commitdate 9.7 KiB (2,464 entries)), so a chunk read is effectively *one data page*. A reader that wants page-level pruning (fetch only the pages of a chunk that match a predicate) has nothing to prune: **one page per chunk and no page index**.

## 4. Alignment and contiguity  [M]

| SF | table | chunks | offset 4 KiB-aligned | size 4 KiB-multiple | end 4 KiB-aligned | offset 512 B-aligned | zero gaps / pairs, within RG | zero gaps / pairs, across RG boundary | bytes of holes in chunk region | mean bytes added by widening one chunk to 4 KiB | = % of chunk bytes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sf10 | lineitem | 7,824 | 0.04% | 0.01% | 0.04% | 0.20% | 7,335 / 7,335 | 488 / 488 | 0 | 4094 | 1.45% |
| sf10 | orders | 1,107 | 0.00% | 0.09% | 0.00% | 0.18% | 984 / 984 | 122 / 122 | 0 | 4094 | 0.78% |
| sf10 | partsupp | 330 | 0.00% | 0.00% | 0.00% | 0.00% | 264 / 264 | 65 / 65 | 0 | 4093 | 0.31% |
| sf10 | part | 153 | 0.00% | 0.00% | 0.00% | 0.00% | 136 / 136 | 16 / 16 | 0 | 4094 | 0.99% |
| sf10 | customer | 104 | 0.00% | 0.00% | 0.00% | 0.00% | 91 / 91 | 12 / 12 | 0 | 4078 | 0.34% |
| sf10 | supplier | 7 | 0.00% | 0.00% | 0.00% | 0.00% | 6 / 6 | 0 / 0 | 0 | 3742 | 0.33% |
| sf10 | nation | 4 | 0.00% | 0.00% | 0.00% | 0.00% | 3 / 3 | 0 / 0 | 0 | 3714 | 972.95% |
| sf10 | region | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 2 / 2 | 0 / 0 | 0 | 3949 | 2686.39% |
| sf100 | lineitem | 78,144 | 0.03% | 0.02% | 0.03% | 0.20% | 73,260 / 73,260 | 4,883 / 4,883 | 0 | 4095 | 1.41% |
| sf100 | orders | 10,989 | 0.00% | 0.03% | 0.00% | 0.12% | 9,768 / 9,768 | 1,220 / 1,220 | 0 | 4096 | 0.69% |
| sf100 | partsupp | 3,260 | 0.03% | 0.00% | 0.03% | 0.31% | 2,608 / 2,608 | 651 / 651 | 0 | 4095 | 0.30% |
| sf100 | part | 1,467 | 0.00% | 0.00% | 0.00% | 0.07% | 1,304 / 1,304 | 162 / 162 | 0 | 4095 | 0.95% |
| sf100 | customer | 984 | 0.00% | 0.00% | 0.00% | 0.10% | 861 / 861 | 122 / 122 | 0 | 4095 | 0.33% |
| sf100 | supplier | 63 | 0.00% | 0.00% | 0.00% | 1.59% | 54 / 54 | 8 / 8 | 0 | 4079 | 0.32% |
| sf100 | nation | 4 | 0.00% | 0.00% | 0.00% | 0.00% | 3 / 3 | 0 / 0 | 0 | 3714 | 972.95% |
| sf100 | region | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 2 / 2 | 0 / 0 | 0 | 3949 | 2686.39% |

Pooled over all tables: sf100 25 of 94,914 chunk offsets are 4 KiB-aligned (0.026%), 21 sizes are 4 KiB multiples (0.022%), offset *and* size both aligned: 0. sf10: 3 of 9,532 offsets aligned (0.031%), 2 sizes (0.021%), both: 0. This is exactly the ~1/4096 expected of unaligned data; the aligned ones are coincidences.

**Contiguity.** In every file, **every** gap between adjacent column chunks is exactly 0 bytes - inside a row group (columns appear in schema order) and across row-group boundaries (last column of row group *i* is immediately followed by the first column of row group *i+1*). The column-chunk region is one gap-free run starting at byte 4 (just after the magic) and ending where the bloom-filter region begins. Consequently:
- A multi-column projection whose columns are *adjacent in the schema* forms one contiguous byte run per row group (e.g. lineitem columns 4-10 = l_quantity ... l_shipdate: 1.04 MiB per row group).
- Chunks are never padded or aligned: the first chunk starts at offset 4, and every later offset is whatever the previous chunk's compressed size makes it. Aligned-down starts and aligned-up ends therefore overlap the neighbouring chunk's bytes (cheap over-read when the neighbour is also wanted, otherwise ~2 KiB per edge on average).
- The smallest chunks are tiny (smallest chunk of the six big tables at sf100: orders.o_shippriority = 53 B - all zeros, a 1-entry dictionary; nation/region columns 44-1,049 B; l_linestatus 2.5-19 KiB), so the only 'holes' that a 4 KiB merge rule could bridge are chunks smaller than 4 KiB (53 B o_shippriority, nation/region columns). See section 7 for the effect on requests.

## 5. Column character: string / dictionary-heavy vs small fixed width (sf100)  [M for sizes/encodings; Qs column is A]

| column | type | encoding | class | chunk p50 | B/row compressed | B/row uncompressed | share of table bytes | Qs |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| lineitem.l_orderkey | INT64 / INT64 | plain | fixed-width - PLAIN | 187.1 KiB | 1.56 | 8.00 | 4.1% | 10 |
| lineitem.l_partkey | INT64 / INT64 | plain | fixed-width - PLAIN | 653.2 KiB | 5.44 | 8.00 | 14.4% | 6 |
| lineitem.l_suppkey | INT64 / INT64 | plain | fixed-width - PLAIN | 592.1 KiB | 4.93 | 8.00 | 13.0% | 7 |
| lineitem.l_linenumber | INT64 / INT64 | dict 100% | fixed-width - dictionary-encoded (small) | 29.4 KiB | 0.25 | 0.38 | 0.6% | 0 |
| lineitem.l_quantity | DECIMAL(15,2) / INT64 | dict 100% | fixed-width - dictionary-encoded (small) | 90.8 KiB | 0.76 | 0.76 | 2.0% | 7 |
| lineitem.l_extendedprice | DECIMAL(15,2) / INT64 | plain | fixed-width - PLAIN | 620.3 KiB | 5.17 | 8.00 | 13.7% | 12 |
| lineitem.l_discount | DECIMAL(15,2) / INT64 | dict 100% | fixed-width - dictionary-encoded (small) | 60.6 KiB | 0.50 | 0.51 | 1.3% | 11 |
| lineitem.l_tax | DECIMAL(15,2) / INT64 | dict 100% | fixed-width - dictionary-encoded (small) | 60.6 KiB | 0.50 | 0.50 | 1.3% | 1 |
| lineitem.l_returnflag | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 28.2 KiB | 0.24 | 0.25 | 0.6% | 2 |
| lineitem.l_linestatus | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 18.8 KiB | 0.16 | 0.25 | 0.4% | 1 |
| lineitem.l_shipdate | DATE / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 190.4 KiB | 1.59 | 1.59 | 4.2% | 8 |
| lineitem.l_commitdate | DATE / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 190.2 KiB | 1.58 | 1.58 | 4.2% | 3 |
| lineitem.l_receiptdate | DATE / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 190.4 KiB | 1.59 | 1.59 | 4.2% | 3 |
| lineitem.l_shipinstruct | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 45.6 KiB | 0.38 | 0.38 | 1.0% | 1 |
| lineitem.l_shipmode | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 45.6 KiB | 0.38 | 0.38 | 1.0% | 2 |
| lineitem.l_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 1.5 MiB | 12.84 | 30.50 | 33.9% | 0 |
| orders.o_orderkey | INT64 / INT64 | plain | fixed-width - PLAIN | 478.5 KiB | 3.99 | 8.00 | 9.2% | 11 |
| orders.o_custkey | INT64 / INT64 | plain | fixed-width - PLAIN | 640.9 KiB | 5.34 | 8.00 | 12.3% | 8 |
| orders.o_orderstatus | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 30.6 KiB | 0.25 | 0.25 | 0.6% | 1 |
| orders.o_totalprice | DECIMAL(15,2) / INT64 | plain | fixed-width - PLAIN | 672.1 KiB | 5.60 | 8.00 | 12.9% | 1 |
| orders.o_orderdate | DATE / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 189.9 KiB | 1.58 | 1.58 | 3.7% | 7 |
| orders.o_orderpriority | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 45.6 KiB | 0.38 | 0.38 | 0.9% | 2 |
| orders.o_clerk | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 765.8 KiB | 6.38 | 19.00 | 14.7% | 0 |
| orders.o_shippriority | INT32 / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 53 B | 0.00 | 0.00 | 0.0% | 1 |
| orders.o_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 2.3 MiB | 19.81 | 52.50 | 45.7% | 1 |
| partsupp.ps_partkey | INT64 / INT64 | plain | fixed-width - PLAIN | 210.5 KiB | 1.75 | 8.00 | 3.2% | 5 |
| partsupp.ps_suppkey | INT64 / INT64 | plain | fixed-width - PLAIN | 480.7 KiB | 4.01 | 8.00 | 7.3% | 5 |
| partsupp.ps_availqty | INT64 / INT64 | mixed (99.8% dict) | fixed-width - dictionary-encoded (small) | 249.7 KiB | 2.08 | 2.41 | 3.8% | 2 |
| partsupp.ps_supplycost | DECIMAL(15,2) / INT64 | plain | fixed-width - PLAIN | 538.5 KiB | 4.49 | 8.00 | 8.2% | 3 |
| partsupp.ps_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 5.0 MiB | 42.52 | 127.50 | 77.5% | 0 |
| part.p_partkey | INT64 / INT64 | plain | fixed-width - PLAIN | 480.3 KiB | 4.00 | 8.00 | 12.7% | 8 |
| part.p_name | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 1.6 MiB | 13.36 | 36.75 | 42.2% | 2 |
| part.p_mfgr | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 45.6 KiB | 0.38 | 0.38 | 1.2% | 1 |
| part.p_brand | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 75.6 KiB | 0.63 | 0.63 | 2.0% | 3 |
| part.p_type | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 121.7 KiB | 1.01 | 1.03 | 3.2% | 4 |
| part.p_size | INT32 / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 90.7 KiB | 0.76 | 0.76 | 2.4% | 3 |
| part.p_container | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 90.8 KiB | 0.76 | 0.76 | 2.4% | 2 |
| part.p_retailprice | DECIMAL(15,2) / INT64 | dict 100% | fixed-width - dictionary-encoded (small) | 271.1 KiB | 2.25 | 2.65 | 7.1% | 0 |
| part.p_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 1019.1 KiB | 8.49 | 17.50 | 26.8% | 0 |
| customer.c_custkey | INT64 / INT64 | plain | fixed-width - PLAIN | 480.3 KiB | 4.00 | 8.00 | 4.9% | 8 |
| customer.c_name | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 596.0 KiB | 4.97 | 22.00 | 6.0% | 2 |
| customer.c_address | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 3.3 MiB | 28.59 | 29.00 | 34.6% | 1 |
| customer.c_nationkey | INT32 / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 75.6 KiB | 0.63 | 0.63 | 0.8% | 4 |
| customer.c_phone | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 1.4 MiB | 11.86 | 19.00 | 14.4% | 2 |
| customer.c_acctbal | DECIMAL(15,2) / INT64 | plain | fixed-width - PLAIN | 593.0 KiB | 4.94 | 8.00 | 6.0% | 2 |
| customer.c_mktsegment | STRING / BYTE_ARRAY | dict 100% | string - low-cardinality, dictionary | 45.6 KiB | 0.38 | 0.38 | 0.5% | 1 |
| customer.c_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 3.2 MiB | 27.15 | 76.49 | 32.9% | 1 |
| supplier.s_suppkey | INT64 / INT64 | plain | fixed-width - PLAIN | 480.3 KiB | 4.00 | 8.00 | 5.1% | 10 |
| supplier.s_name | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 596.0 KiB | 4.97 | 22.00 | 6.3% | 4 |
| supplier.s_address | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 3.3 MiB | 28.58 | 29.00 | 36.1% | 3 |
| supplier.s_nationkey | INT32 / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 75.6 KiB | 0.63 | 0.63 | 0.8% | 8 |
| supplier.s_phone | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 1.4 MiB | 11.86 | 19.00 | 15.0% | 2 |
| supplier.s_acctbal | DECIMAL(15,2) / INT64 | plain | fixed-width - PLAIN | 593.0 KiB | 4.94 | 8.00 | 6.2% | 1 |
| supplier.s_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 2.8 MiB | 24.12 | 66.50 | 30.5% | 2 |
| nation.n_nationkey | INT32 / INT32 | plain | fixed-width - PLAIN | 125 B | 5.00 | 5.00 | 8.2% | 9 |
| nation.n_name | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 273 B | 10.92 | 12.08 | 17.9% | 9 |
| nation.n_regionkey | INT32 / INT32 | dict 100% | fixed-width - dictionary-encoded (small) | 80 B | 3.20 | 6.20 | 5.2% | 3 |
| nation.n_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 1.0 KiB | 41.96 | 79.28 | 68.7% | 0 |
| region.r_regionkey | INT32 / INT32 | plain | fixed-width - PLAIN | 44 B | 8.80 | 8.60 | 10.0% | 3 |
| region.r_name | STRING / BYTE_ARRAY | plain | string - high-cardinality, PLAIN | 75 B | 15.00 | 15.40 | 17.0% | 3 |
| region.r_comment | STRING / BYTE_ARRAY | plain | string - free text (high entropy, largest chunks) | 322 B | 64.40 | 75.00 | 73.0% | 0 |

Share of each table's on-disk bytes by column class (sf100):

| table | free-text comment | other strings, PLAIN | strings, dictionary | fixed-width PLAIN (keys, prices, balances) | fixed-width dictionary (dates, qty, flags) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| lineitem | 33.9% | 0.0% | 3.0% | 45.2% | 17.9% |
| orders | 45.7% | 14.7% | 1.5% | 34.5% | 3.7% |
| partsupp | 77.5% | 0.0% | 0.0% | 18.7% | 3.8% |
| part | 26.8% | 42.2% | 8.8% | 12.7% | 9.5% |
| customer | 32.9% | 55.0% | 0.5% | 10.8% | 0.8% |
| supplier | 30.5% | 57.4% | 0.0% | 11.3% | 0.8% |
| nation | 68.7% | 17.9% | 0.0% | 8.2% | 5.2% |
| region | 73.0% | 17.0% | 0.0% | 10.0% | 0.0% |

**Reading guide (sf100, median chunk in brackets; the six big tables):**

- *String-heavy by bytes*: BYTE_ARRAY columns make up customer 88%, supplier 88%, part 78%, partsupp 78%, orders 62%, lineitem 37% of the on-disk bytes. Free-text comments and long PLAIN strings give the largest chunks of the dataset: part.p_comment (1019.1 KiB), lineitem.l_comment (1.5 MiB), orders.o_comment (2.3 MiB), supplier.s_comment (2.8 MiB), customer.c_comment (3.2 MiB), partsupp.ps_comment (5.0 MiB); other PLAIN strings: customer.c_name (596.0 KiB), supplier.s_name (596.0 KiB), orders.o_clerk (765.8 KiB), supplier.s_phone (1.4 MiB), customer.c_phone (1.4 MiB), part.p_name (1.6 MiB), supplier.s_address (3.3 MiB), customer.c_address (3.3 MiB).

- *Dictionary-heavy, low-cardinality strings* (small chunks, dictionary page about 1.2 KiB or less, dictionary-encoded in every row group): lineitem.l_linestatus (18.8 KiB), lineitem.l_returnflag (28.2 KiB), orders.o_orderstatus (30.6 KiB), part.p_mfgr (45.6 KiB), lineitem.l_shipmode (45.6 KiB), orders.o_orderpriority (45.6 KiB), customer.c_mktsegment (45.6 KiB), lineitem.l_shipinstruct (45.6 KiB), part.p_brand (75.6 KiB), part.p_container (90.8 KiB), part.p_type (121.7 KiB).

- *Small fixed-width columns, dictionary-encoded* (dates, quantities, rates, small ints, and also p_retailprice/ps_availqty which have 10-14k distinct values per row group): orders.o_shippriority (53 B), lineitem.l_linenumber (29.4 KiB), lineitem.l_tax (60.6 KiB), lineitem.l_discount (60.6 KiB), customer.c_nationkey (75.6 KiB), supplier.s_nationkey (75.6 KiB), part.p_size (90.7 KiB), lineitem.l_quantity (90.8 KiB), orders.o_orderdate (189.9 KiB), lineitem.l_commitdate (190.2 KiB), lineitem.l_shipdate (190.4 KiB), lineitem.l_receiptdate (190.4 KiB), partsupp.ps_availqty (249.7 KiB), part.p_retailprice (271.1 KiB). Dates carry ~2.4-2.5k distinct values per row group (dictionary page ~10 KiB).

- *Fixed-width PLAIN columns* (keys, prices, balances; Snappy ratio from 1.43x for orders.o_totalprice to 5.13x for lineitem.l_orderkey): lineitem.l_orderkey (187.1 KiB), partsupp.ps_partkey (210.5 KiB), orders.o_orderkey (478.5 KiB), supplier.s_suppkey (480.3 KiB), part.p_partkey (480.3 KiB), customer.c_custkey (480.3 KiB), partsupp.ps_suppkey (480.7 KiB), partsupp.ps_supplycost (538.5 KiB), lineitem.l_suppkey (592.1 KiB), customer.c_acctbal (593.0 KiB), supplier.s_acctbal (593.0 KiB), lineitem.l_extendedprice (620.3 KiB), orders.o_custkey (640.9 KiB), lineitem.l_partkey (653.2 KiB), orders.o_totalprice (672.1 KiB).

Columns referenced by **none** of the 22 queries [A] and their share of the table's bytes (these bytes are never read by a column-pruned TPC-H scan):

| table | unreferenced columns | share of table bytes |
| :--- | :--- | ---: |
| lineitem | l_linenumber, l_comment | 34.6% |
| orders | o_clerk | 14.7% |
| partsupp | ps_comment | 77.5% |
| part | p_retailprice, p_comment | 33.9% |
| customer | (none) | 0.0% |
| supplier | (none) | 0.0% |
| nation | n_comment | 68.7% |
| region | r_comment | 73.0% |

## 6. Query -> columns read  [A: derived from the standard TPC-H SQL, not measured]

Columns = every column the query references for that table instance (select list, predicates, join keys, group/order keys). Default validation parameters of the TPC-H specification. One row per *scan instance*; a table that appears several times (self-join aliases, subqueries, the Q15 view) has several rows. Two models are evaluated in sections 7-8: **union** (default; one shared scan per table with the union of its columns) and **multi** (every instance scanned separately; sensitivity).

| Q | name | table | scan instance | columns read | n/total |
| :--- | :--- | :--- | :--- | :--- | ---: |
| Q1 | Pricing Summary Report | lineitem |  | l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate | 7/16 |
| Q2 | Minimum Cost Supplier | part | outer | p_partkey, p_mfgr, p_type, p_size | 4/9 |
| Q2 | Minimum Cost Supplier | supplier | outer | s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment | 7/7 |
| Q2 | Minimum Cost Supplier | partsupp | outer | ps_partkey, ps_suppkey, ps_supplycost | 3/5 |
| Q2 | Minimum Cost Supplier | nation | outer | n_nationkey, n_name, n_regionkey | 3/4 |
| Q2 | Minimum Cost Supplier | region | outer | r_regionkey, r_name | 2/3 |
| Q2 | Minimum Cost Supplier | partsupp | min-cost subquery | ps_partkey, ps_suppkey, ps_supplycost | 3/5 |
| Q2 | Minimum Cost Supplier | supplier | min-cost subquery | s_suppkey, s_nationkey | 2/7 |
| Q2 | Minimum Cost Supplier | nation | min-cost subquery | n_nationkey, n_regionkey | 2/4 |
| Q2 | Minimum Cost Supplier | region | min-cost subquery | r_regionkey, r_name | 2/3 |
| Q3 | Shipping Priority | customer |  | c_custkey, c_mktsegment | 2/8 |
| Q3 | Shipping Priority | orders |  | o_orderkey, o_custkey, o_orderdate, o_shippriority | 4/9 |
| Q3 | Shipping Priority | lineitem |  | l_orderkey, l_extendedprice, l_discount, l_shipdate | 4/16 |
| Q4 | Order Priority Checking | orders |  | o_orderkey, o_orderdate, o_orderpriority | 3/9 |
| Q4 | Order Priority Checking | lineitem | EXISTS subquery | l_orderkey, l_commitdate, l_receiptdate | 3/16 |
| Q5 | Local Supplier Volume | customer |  | c_custkey, c_nationkey | 2/8 |
| Q5 | Local Supplier Volume | orders |  | o_orderkey, o_custkey, o_orderdate | 3/9 |
| Q5 | Local Supplier Volume | lineitem |  | l_orderkey, l_suppkey, l_extendedprice, l_discount | 4/16 |
| Q5 | Local Supplier Volume | supplier |  | s_suppkey, s_nationkey | 2/7 |
| Q5 | Local Supplier Volume | nation |  | n_nationkey, n_name, n_regionkey | 3/4 |
| Q5 | Local Supplier Volume | region |  | r_regionkey, r_name | 2/3 |
| Q6 | Forecasting Revenue Change | lineitem |  | l_quantity, l_extendedprice, l_discount, l_shipdate | 4/16 |
| Q7 | Volume Shipping | supplier |  | s_suppkey, s_nationkey | 2/7 |
| Q7 | Volume Shipping | lineitem |  | l_orderkey, l_suppkey, l_extendedprice, l_discount, l_shipdate | 5/16 |
| Q7 | Volume Shipping | orders |  | o_orderkey, o_custkey | 2/9 |
| Q7 | Volume Shipping | customer |  | c_custkey, c_nationkey | 2/8 |
| Q7 | Volume Shipping | nation | n1 | n_nationkey, n_name | 2/4 |
| Q7 | Volume Shipping | nation | n2 | n_nationkey, n_name | 2/4 |
| Q8 | National Market Share | part |  | p_partkey, p_type | 2/9 |
| Q8 | National Market Share | supplier |  | s_suppkey, s_nationkey | 2/7 |
| Q8 | National Market Share | lineitem |  | l_orderkey, l_partkey, l_suppkey, l_extendedprice, l_discount | 5/16 |
| Q8 | National Market Share | orders |  | o_orderkey, o_custkey, o_orderdate | 3/9 |
| Q8 | National Market Share | customer |  | c_custkey, c_nationkey | 2/8 |
| Q8 | National Market Share | nation | n1 | n_nationkey, n_regionkey | 2/4 |
| Q8 | National Market Share | nation | n2 | n_nationkey, n_name | 2/4 |
| Q8 | National Market Share | region |  | r_regionkey, r_name | 2/3 |
| Q9 | Product Type Profit Measure | part |  | p_partkey, p_name | 2/9 |
| Q9 | Product Type Profit Measure | supplier |  | s_suppkey, s_nationkey | 2/7 |
| Q9 | Product Type Profit Measure | lineitem |  | l_orderkey, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount | 6/16 |
| Q9 | Product Type Profit Measure | partsupp |  | ps_partkey, ps_suppkey, ps_supplycost | 3/5 |
| Q9 | Product Type Profit Measure | orders |  | o_orderkey, o_orderdate | 2/9 |
| Q9 | Product Type Profit Measure | nation |  | n_nationkey, n_name | 2/4 |
| Q10 | Returned Item Reporting | customer |  | c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_comment | 7/8 |
| Q10 | Returned Item Reporting | orders |  | o_orderkey, o_custkey, o_orderdate | 3/9 |
| Q10 | Returned Item Reporting | lineitem |  | l_orderkey, l_extendedprice, l_discount, l_returnflag | 4/16 |
| Q10 | Returned Item Reporting | nation |  | n_nationkey, n_name | 2/4 |
| Q11 | Important Stock Identification | partsupp | main | ps_partkey, ps_suppkey, ps_availqty, ps_supplycost | 4/5 |
| Q11 | Important Stock Identification | supplier | main | s_suppkey, s_nationkey | 2/7 |
| Q11 | Important Stock Identification | nation | main | n_nationkey, n_name | 2/4 |
| Q11 | Important Stock Identification | partsupp | HAVING subquery | ps_suppkey, ps_availqty, ps_supplycost | 3/5 |
| Q11 | Important Stock Identification | supplier | HAVING subquery | s_suppkey, s_nationkey | 2/7 |
| Q11 | Important Stock Identification | nation | HAVING subquery | n_nationkey, n_name | 2/4 |
| Q12 | Shipping Modes and Order Priority | orders |  | o_orderkey, o_orderpriority | 2/9 |
| Q12 | Shipping Modes and Order Priority | lineitem |  | l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode | 5/16 |
| Q13 | Customer Distribution | customer |  | c_custkey | 1/8 |
| Q13 | Customer Distribution | orders |  | o_orderkey, o_custkey, o_comment | 3/9 |
| Q14 | Promotion Effect | lineitem |  | l_partkey, l_extendedprice, l_discount, l_shipdate | 4/16 |
| Q14 | Promotion Effect | part |  | p_partkey, p_type | 2/9 |
| Q15 | Top Supplier | supplier |  | s_suppkey, s_name, s_address, s_phone | 4/7 |
| Q15 | Top Supplier | lineitem | revenue0 view (join side) | l_suppkey, l_extendedprice, l_discount, l_shipdate | 4/16 |
| Q15 | Top Supplier | lineitem | revenue0 view (max subquery) | l_suppkey, l_extendedprice, l_discount, l_shipdate | 4/16 |
| Q16 | Parts/Supplier Relationship | partsupp |  | ps_partkey, ps_suppkey | 2/5 |
| Q16 | Parts/Supplier Relationship | part |  | p_partkey, p_brand, p_type, p_size | 4/9 |
| Q16 | Parts/Supplier Relationship | supplier | NOT IN subquery | s_suppkey, s_comment | 2/7 |
| Q17 | Small-Quantity-Order Revenue | lineitem | outer | l_partkey, l_quantity, l_extendedprice | 3/16 |
| Q17 | Small-Quantity-Order Revenue | part |  | p_partkey, p_brand, p_container | 3/9 |
| Q17 | Small-Quantity-Order Revenue | lineitem | avg subquery | l_partkey, l_quantity | 2/16 |
| Q18 | Large Volume Customer | customer |  | c_custkey, c_name | 2/8 |
| Q18 | Large Volume Customer | orders |  | o_orderkey, o_custkey, o_totalprice, o_orderdate | 4/9 |
| Q18 | Large Volume Customer | lineitem | outer join side | l_orderkey, l_quantity | 2/16 |
| Q18 | Large Volume Customer | lineitem | GROUP BY/HAVING subquery | l_orderkey, l_quantity | 2/16 |
| Q19 | Discounted Revenue | lineitem |  | l_partkey, l_quantity, l_extendedprice, l_discount, l_shipinstruct, l_shipmode | 6/16 |
| Q19 | Discounted Revenue | part |  | p_partkey, p_brand, p_size, p_container | 4/9 |
| Q20 | Potential Part Promotion | supplier |  | s_suppkey, s_name, s_address, s_nationkey | 4/7 |
| Q20 | Potential Part Promotion | nation |  | n_nationkey, n_name | 2/4 |
| Q20 | Potential Part Promotion | partsupp | IN subquery | ps_partkey, ps_suppkey, ps_availqty | 3/5 |
| Q20 | Potential Part Promotion | part | IN subquery | p_partkey, p_name | 2/9 |
| Q20 | Potential Part Promotion | lineitem | correlated subquery | l_partkey, l_suppkey, l_quantity, l_shipdate | 4/16 |
| Q21 | Suppliers Who Kept Orders Waiting | supplier |  | s_suppkey, s_name, s_nationkey | 3/7 |
| Q21 | Suppliers Who Kept Orders Waiting | lineitem | l1 | l_orderkey, l_suppkey, l_commitdate, l_receiptdate | 4/16 |
| Q21 | Suppliers Who Kept Orders Waiting | orders |  | o_orderkey, o_orderstatus | 2/9 |
| Q21 | Suppliers Who Kept Orders Waiting | nation |  | n_nationkey, n_name | 2/4 |
| Q21 | Suppliers Who Kept Orders Waiting | lineitem | l2 (EXISTS) | l_orderkey, l_suppkey | 2/16 |
| Q21 | Suppliers Who Kept Orders Waiting | lineitem | l3 (NOT EXISTS) | l_orderkey, l_suppkey, l_commitdate, l_receiptdate | 4/16 |
| Q22 | Global Sales Opportunity | customer | outer | c_custkey, c_phone, c_acctbal | 3/8 |
| Q22 | Global Sales Opportunity | orders | NOT EXISTS subquery | o_custkey | 1/9 |
| Q22 | Global Sales Opportunity | customer | avg subquery | c_phone, c_acctbal | 2/8 |

Assumptions baked into every number that follows: perfect column pruning (a column is read iff referenced); every row group of every referenced table is read (no min/max row-group skipping - see section 9 - and no predicate pushdown / late materialisation); a column chunk is read as a single byte range `[min(dictionary_page_offset, data_page_offset), +total_compressed_size)` - the standard Parquet chunk extent, which is what a hybrid-scan reader's `all_column_chunks_byte_ranges` (documented as 'byte ranges of column chunks of all (or selected) columns') is expected to return; I did not inspect cuDF's implementation; footer, bloom filters and page indexes are not counted (footers are 0.6 KiB - 7.6 MiB per file and are read once).

## 7. Column-pruned scan of Q1..Q22  [A for column sets and 'every RG is read'; M for chunk sizes; D for all numbers below]

### 7.0 How requests are formed (variants)  [model]

For one table scan, the column-chunk ranges of the projected columns are sorted by file offset and turned into I/O requests in one of six ways. The merge rule is the repo's own (`coalesce_ranges`: merge when `next.start <= cur.end + max_gap`; `uring_reactor::align_and_coalesce`: widen each range to 4096-byte boundaries, sort, merge when `next.begin <= prev.end`). The simulator was checked against literal Python ports of those C++ functions on all 22 queries x 2 SF x 12 variant/scope combinations (528 checks, 0 mismatches).

| short | definition |
| :--- | :--- |
| raw | raw: 1 request per column chunk |
| gap 0 | g0: merge touching ranges (gap 0) |
| 4K-aligned | a4k: align to 4 KiB, merge touching (repo `align_and_coalesce`) |
| gap 4 KiB | g4k: merge if gap <= 4 KiB |
| gap 64 KiB | g64k: merge if gap <= 64 KiB |
| gap 1 MiB | g1m: merge if gap <= 1 MiB |

Two merge **scopes**: *file-global* (ranges of all row groups of a file are merged in one sorted pass - what `benchmark/parquet_io_benchmark.cpp` does with `all_column_chunks_byte_ranges`) and *per-RG* (a merged request never spans two row groups - a scan task per row group). Over-read % = (requested bytes - useful bytes) / useful bytes, where useful bytes = sum of the projected chunk sizes. Size classes: **<1 MiB** = below the cuFile reactor's `gds_threshold` (POSIX `pread` fallback), **1-16 MiB** and **>=16 MiB** = GDS-eligible.

### 7.1 sf100 summary: ranges and requests per query (union model, file-global merge scope)

`ranges (raw)` = number of column-chunk byte ranges the scan needs; `range p50/p90/max` describe their sizes; the `#req` columns give the number of I/O requests after merging (and the over-read as a % of useful bytes). Bytes requested equal the useful bytes except where noted by over-read.

| Q | tables: cols read/total | useful bytes | ranges (raw) | range p50 | range p90 | range max | #req gap 0 | #req 4K-aligned (over-read) | #req gap 4 KiB (over-read) | #req gap 64 KiB (over-read) | #req gap 1 MiB (over-read) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | lineitem 7/16 | 4.98 GiB | 34,188 | 60.6 KiB | 620.2 KiB | 620.6 KiB | 4,884 | 4,884 (+0.4%) | 4,884 (+0.0%) | 4,884 (+0.0%) | 4,884 (+0.0%) |
| Q2 | partsupp 3/5, part 4/9, supplier 7/7, nation 3/4, region 2/3 | 974.66 MiB | 2,676 | 480.7 KiB | 538.6 KiB | 3.4 MiB | 1,796 | 1,796 (+0.7%) | 1,796 (+0.0%) | 1,794 (+0.0%) | 981 (+17.5%) |
| Q3 | lineitem 4/16, orders 4/9, customer 2/8 | 6.51 GiB | 24,666 | 189.9 KiB | 620.3 KiB | 641.2 KiB | 18,561 | 18,561 (+1.1%) | 18,561 (+0.0%) | 18,560 (+0.0%) | 11,233 (+34.8%) |
| Q4 | lineitem 3/16, orders 3/9 | 3.47 GiB | 18,315 | 190.2 KiB | 190.5 KiB | 478.9 KiB | 12,210 | 12,210 (+1.3%) | 12,210 (+0.0%) | 12,210 (+0.0%) | 12,208 (+0.0%) |
| Q5 | lineitem 4/16, orders 3/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 8.39 GiB | 23,468 | 478.5 KiB | 620.3 KiB | 641.2 KiB | 17,360 | 17,360 (+0.8%) | 17,360 (+0.0%) | 17,359 (+0.0%) | 6,369 (+52.7%) |
| Q6 | lineitem 4/16 | 4.48 GiB | 19,536 | 90.8 KiB | 620.3 KiB | 620.6 KiB | 9,768 | 9,768 (+0.8%) | 9,768 (+0.0%) | 9,767 (+0.0%) | 4,884 (+11.2%) |
| Q7 | lineitem 5/16, orders 2/9, customer 2/8, supplier 2/7, nation 2/4 | 9.06 GiB | 27,128 | 190.4 KiB | 620.3 KiB | 641.2 KiB | 21,022 | 21,022 (+0.9%) | 21,022 (+0.0%) | 21,020 (+0.0%) | 6,368 (+45.3%) |
| Q8 | lineitem 5/16, orders 3/9, part 2/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 11.53 GiB | 28,678 | 592.1 KiB | 653.2 KiB | 653.7 KiB | 12,802 | 12,802 (+0.4%) | 12,802 (+0.0%) | 12,801 (+0.0%) | 6,695 (+12.0%) |
| Q9 | lineitem 6/16, orders 2/9, partsupp 3/5, part 2/9, supplier 2/7, nation 2/4 | 12.13 GiB | 34,048 | 478.4 KiB | 653.2 KiB | 1.6 MiB | 13,696 | 13,695 (+0.4%) | 13,695 (+0.0%) | 8,811 (+1.1%) | 8,158 (+2.4%) |
| Q10 | lineitem 4/16, orders 3/9, customer 7/8, nation 2/4 | 6.84 GiB | 24,062 | 187.0 KiB | 620.4 KiB | 3.4 MiB | 17,219 | 17,218 (+1.0%) | 17,218 (+0.0%) | 12,212 (+4.2%) | 10,990 (+16.2%) |
| Q11 | partsupp 4/5, supplier 2/7, nation 2/4 | 945.05 MiB | 2,628 | 249.7 KiB | 538.5 KiB | 538.9 KiB | 671 | 671 (+0.3%) | 671 (+0.0%) | 671 (+0.0%) | 670 (+0.1%) |
| Q12 | lineitem 5/16, orders 2/9 | 4.35 GiB | 26,862 | 190.2 KiB | 190.4 KiB | 478.9 KiB | 17,094 | 17,094 (+1.5%) | 17,094 (+0.0%) | 12,210 (+4.9%) | 12,209 (+4.9%) |
| Q13 | orders 3/9, customer 1/8 | 4.13 GiB | 3,786 | 640.9 KiB | 2.3 MiB | 2.3 MiB | 1,345 | 1,345 (+0.1%) | 1,345 (+0.0%) | 1,345 (+0.0%) | 1,345 (+0.0%) |
| Q14 | lineitem 4/16, part 2/9 | 7.19 GiB | 19,862 | 190.4 KiB | 653.3 KiB | 653.7 KiB | 14,978 | 14,978 (+0.8%) | 14,978 (+0.0%) | 14,977 (+0.0%) | 5,210 (+53.1%) |
| Q15 | lineitem 4/16, supplier 4/7 | 6.86 GiB | 19,572 | 591.8 KiB | 620.3 KiB | 3.4 MiB | 14,670 | 14,670 (+0.8%) | 14,670 (+0.0%) | 14,667 (+0.0%) | 4,893 (+15.5%) |
| Q16 | partsupp 2/5, part 4/9, supplier 2/7 | 588.46 MiB | 1,974 | 210.6 KiB | 480.8 KiB | 2.8 MiB | 988 | 988 (+0.7%) | 988 (+0.0%) | 988 (+0.0%) | 987 (+0.1%) |
| Q17 | lineitem 3/16, part 3/9 | 6.45 GiB | 15,141 | 620.2 KiB | 653.3 KiB | 653.7 KiB | 10,257 | 10,257 (+0.6%) | 10,257 (+0.0%) | 10,257 (+0.0%) | 5,210 (+45.4%) |
| Q18 | lineitem 2/16, orders 4/9, customer 2/8 | 3.73 GiB | 14,898 | 187.1 KiB | 641.0 KiB | 672.6 KiB | 12,333 | 12,333 (+1.3%) | 12,333 (+0.0%) | 11,112 (+1.0%) | 11,111 (+1.0%) |
| Q19 | lineitem 6/16, part 4/9 | 7.17 GiB | 29,956 | 90.7 KiB | 653.2 KiB | 653.7 KiB | 15,141 | 15,141 (+0.8%) | 15,141 (+0.0%) | 15,141 (+0.0%) | 5,210 (+84.7%) |
| Q20 | lineitem 4/16, partsupp 3/5, part 2/9, supplier 4/7, nation 2/4 | 8.05 GiB | 21,856 | 249.7 KiB | 653.3 KiB | 3.4 MiB | 15,477 | 15,476 (+0.7%) | 15,476 (+0.0%) | 10,593 (+1.7%) | 5,709 (+47.3%) |
| Q21 | lineitem 4/16, orders 2/9, supplier 3/7, nation 2/4 | 6.00 GiB | 22,007 | 190.2 KiB | 592.1 KiB | 596.3 KiB | 17,113 | 17,113 (+1.1%) | 17,113 (+0.0%) | 17,113 (+0.0%) | 11,006 (+63.1%) |
| Q22 | orders 1/9, customer 3/8 | 1.04 GiB | 1,590 | 640.9 KiB | 641.1 KiB | 1.4 MiB | 1,467 | 1,467 (+0.5%) | 1,467 (+0.0%) | 1,467 (+0.0%) | 1,466 (+0.0%) |
| **all 22** |  | **124.83 GiB** | **416,897** |  |  |  | **250,852** | **250,849 (+0.8%)** | **250,849 (+0.0%)** | **229,959 (+0.6%)** | **137,796 (+28.8%)** |

Same table with the per-row-group merge scope (a request never spans row groups):

| Q | tables: cols read/total | useful bytes | ranges (raw) | range p50 | range p90 | range max | #req gap 0 | #req 4K-aligned (over-read) | #req gap 4 KiB (over-read) | #req gap 64 KiB (over-read) | #req gap 1 MiB (over-read) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | lineitem 7/16 | 4.98 GiB | 34,188 | 60.6 KiB | 620.2 KiB | 620.6 KiB | 4,884 | 4,884 (+0.4%) | 4,884 (+0.0%) | 4,884 (+0.0%) | 4,884 (+0.0%) |
| Q2 | partsupp 3/5, part 4/9, supplier 7/7, nation 3/4, region 2/3 | 974.66 MiB | 2,676 | 480.7 KiB | 538.6 KiB | 3.4 MiB | 1,804 | 1,804 (+0.7%) | 1,804 (+0.0%) | 1,802 (+0.0%) | 989 (+17.5%) |
| Q3 | lineitem 4/16, orders 4/9, customer 2/8 | 6.51 GiB | 24,666 | 189.9 KiB | 620.3 KiB | 641.2 KiB | 18,561 | 18,561 (+1.1%) | 18,561 (+0.0%) | 18,560 (+0.0%) | 11,233 (+34.8%) |
| Q4 | lineitem 3/16, orders 3/9 | 3.47 GiB | 18,315 | 190.2 KiB | 190.5 KiB | 478.9 KiB | 12,210 | 12,210 (+1.3%) | 12,210 (+0.0%) | 12,210 (+0.0%) | 12,208 (+0.0%) |
| Q5 | lineitem 4/16, orders 3/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 8.39 GiB | 23,468 | 478.5 KiB | 620.3 KiB | 641.2 KiB | 17,360 | 17,360 (+0.8%) | 17,360 (+0.0%) | 17,359 (+0.0%) | 6,369 (+52.7%) |
| Q6 | lineitem 4/16 | 4.48 GiB | 19,536 | 90.8 KiB | 620.3 KiB | 620.6 KiB | 9,768 | 9,768 (+0.8%) | 9,768 (+0.0%) | 9,767 (+0.0%) | 4,884 (+11.2%) |
| Q7 | lineitem 5/16, orders 2/9, customer 2/8, supplier 2/7, nation 2/4 | 9.06 GiB | 27,128 | 190.4 KiB | 620.3 KiB | 641.2 KiB | 21,022 | 21,022 (+0.9%) | 21,022 (+0.0%) | 21,020 (+0.0%) | 6,368 (+45.3%) |
| Q8 | lineitem 5/16, orders 3/9, part 2/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 11.53 GiB | 28,678 | 592.1 KiB | 653.2 KiB | 653.7 KiB | 12,802 | 12,802 (+0.4%) | 12,802 (+0.0%) | 12,801 (+0.0%) | 6,695 (+12.0%) |
| Q9 | lineitem 6/16, orders 2/9, partsupp 3/5, part 2/9, supplier 2/7, nation 2/4 | 12.13 GiB | 34,048 | 478.4 KiB | 653.2 KiB | 1.6 MiB | 13,696 | 13,695 (+0.4%) | 13,695 (+0.0%) | 8,811 (+1.1%) | 8,158 (+2.4%) |
| Q10 | lineitem 4/16, orders 3/9, customer 7/8, nation 2/4 | 6.84 GiB | 24,062 | 187.0 KiB | 620.4 KiB | 3.4 MiB | 17,341 | 17,340 (+1.0%) | 17,340 (+0.0%) | 12,334 (+4.2%) | 11,112 (+16.2%) |
| Q11 | partsupp 4/5, supplier 2/7, nation 2/4 | 945.05 MiB | 2,628 | 249.7 KiB | 538.5 KiB | 538.9 KiB | 671 | 671 (+0.3%) | 671 (+0.0%) | 671 (+0.0%) | 670 (+0.1%) |
| Q12 | lineitem 5/16, orders 2/9 | 4.35 GiB | 26,862 | 190.2 KiB | 190.4 KiB | 478.9 KiB | 17,094 | 17,094 (+1.5%) | 17,094 (+0.0%) | 12,210 (+4.9%) | 12,209 (+4.9%) |
| Q13 | orders 3/9, customer 1/8 | 4.13 GiB | 3,786 | 640.9 KiB | 2.3 MiB | 2.3 MiB | 2,565 | 2,565 (+0.2%) | 2,565 (+0.0%) | 2,565 (+0.0%) | 2,565 (+0.0%) |
| Q14 | lineitem 4/16, part 2/9 | 7.19 GiB | 19,862 | 190.4 KiB | 653.3 KiB | 653.7 KiB | 14,978 | 14,978 (+0.8%) | 14,978 (+0.0%) | 14,977 (+0.0%) | 5,210 (+53.1%) |
| Q15 | lineitem 4/16, supplier 4/7 | 6.86 GiB | 19,572 | 591.8 KiB | 620.3 KiB | 3.4 MiB | 14,670 | 14,670 (+0.8%) | 14,670 (+0.0%) | 14,667 (+0.0%) | 4,893 (+15.5%) |
| Q16 | partsupp 2/5, part 4/9, supplier 2/7 | 588.46 MiB | 1,974 | 210.6 KiB | 480.8 KiB | 2.8 MiB | 996 | 996 (+0.7%) | 996 (+0.0%) | 996 (+0.0%) | 995 (+0.1%) |
| Q17 | lineitem 3/16, part 3/9 | 6.45 GiB | 15,141 | 620.2 KiB | 653.3 KiB | 653.7 KiB | 10,257 | 10,257 (+0.6%) | 10,257 (+0.0%) | 10,257 (+0.0%) | 5,210 (+45.4%) |
| Q18 | lineitem 2/16, orders 4/9, customer 2/8 | 3.73 GiB | 14,898 | 187.1 KiB | 641.0 KiB | 672.6 KiB | 12,333 | 12,333 (+1.3%) | 12,333 (+0.0%) | 11,112 (+1.0%) | 11,111 (+1.0%) |
| Q19 | lineitem 6/16, part 4/9 | 7.17 GiB | 29,956 | 90.7 KiB | 653.2 KiB | 653.7 KiB | 15,141 | 15,141 (+0.8%) | 15,141 (+0.0%) | 15,141 (+0.0%) | 5,210 (+84.7%) |
| Q20 | lineitem 4/16, partsupp 3/5, part 2/9, supplier 4/7, nation 2/4 | 8.05 GiB | 21,856 | 249.7 KiB | 653.3 KiB | 3.4 MiB | 15,477 | 15,476 (+0.7%) | 15,476 (+0.0%) | 10,593 (+1.7%) | 5,709 (+47.3%) |
| Q21 | lineitem 4/16, orders 2/9, supplier 3/7, nation 2/4 | 6.00 GiB | 22,007 | 190.2 KiB | 592.1 KiB | 596.3 KiB | 17,113 | 17,113 (+1.1%) | 17,113 (+0.0%) | 17,113 (+0.0%) | 11,006 (+63.1%) |
| Q22 | orders 1/9, customer 3/8 | 1.04 GiB | 1,590 | 640.9 KiB | 641.1 KiB | 1.4 MiB | 1,467 | 1,467 (+0.5%) | 1,467 (+0.0%) | 1,467 (+0.0%) | 1,466 (+0.0%) |
| **all 22** |  | **124.83 GiB** | **416,897** |  |  |  | **252,210** | **252,207 (+0.8%)** | **252,207 (+0.0%)** | **231,317 (+0.6%)** | **139,154 (+28.8%)** |


### 7.2 (omitted)

The per-query detail blocks of the original run (request-size classes, bytes-weighted medians and per-row-group request counts for each query at sf100) are not reproduced here; sections 7.1 and 8.5 carry the per-query numbers that matter for decisions.

### 7.3 sf10 compact table (union model, file-global scope)

At sf10 the chunk sizes are the same as at sf100 (same 122,880-row groups) but there are ~10x fewer of them, so request *counts* scale by ~10x while request *sizes* and over-read percentages stay the same.

| Q | tables: cols read/total | useful bytes | ranges (raw) | range p50 | range p90 | range max | #req gap 0 | #req 4K-aligned (over-read) | #req gap 4 KiB (over-read) | #req gap 64 KiB (over-read) | #req gap 1 MiB (over-read) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | lineitem 7/16 | 509.71 MiB | 3,423 | 60.6 KiB | 619.8 KiB | 620.2 KiB | 489 | 489 (+0.4%) | 489 (+0.0%) | 489 (+0.0%) | 489 (+0.0%) |
| Q2 | partsupp 3/5, part 4/9, supplier 7/7, nation 3/4, region 2/3 | 97.47 MiB | 278 | 480.3 KiB | 538.5 KiB | 2.7 MiB | 186 | 186 (+0.8%) | 186 (+0.0%) | 184 (+0.1%) | 102 (+18.0%) |
| Q3 | lineitem 4/16, orders 4/9, customer 2/8 | 661.21 MiB | 2,474 | 189.9 KiB | 619.8 KiB | 620.2 KiB | 1,862 | 1,862 (+1.1%) | 1,862 (+0.0%) | 1,859 (+0.0%) | 1,126 (+26.0%) |
| Q4 | lineitem 3/16, orders 3/9 | 355.78 MiB | 1,836 | 190.2 KiB | 190.5 KiB | 478.7 KiB | 1,224 | 1,224 (+1.3%) | 1,224 (+0.0%) | 1,224 (+0.0%) | 1,222 (+0.1%) |
| Q5 | lineitem 4/16, orders 3/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 827.90 MiB | 2,358 | 478.5 KiB | 619.8 KiB | 620.2 KiB | 1,743 | 1,743 (+0.8%) | 1,743 (+0.0%) | 1,741 (+0.0%) | 641 (+51.5%) |
| Q6 | lineitem 4/16 | 458.41 MiB | 1,956 | 90.8 KiB | 619.9 KiB | 620.2 KiB | 978 | 978 (+0.8%) | 978 (+0.0%) | 977 (+0.0%) | 489 (+11.2%) |
| Q7 | lineitem 5/16, orders 2/9, customer 2/8, supplier 2/7, nation 2/4 | 896.02 MiB | 2,721 | 190.4 KiB | 619.8 KiB | 620.2 KiB | 2,108 | 2,108 (+0.9%) | 2,108 (+0.0%) | 2,106 (+0.0%) | 640 (+44.0%) |
| Q8 | lineitem 5/16, orders 3/9, part 2/9, customer 2/8, supplier 2/7, nation 3/4, region 2/3 | 1.10 GiB | 2,881 | 538.3 KiB | 619.8 KiB | 620.2 KiB | 1,288 | 1,288 (+0.5%) | 1,288 (+0.0%) | 1,286 (+0.0%) | 674 (+12.7%) |
| Q9 | lineitem 6/16, orders 2/9, partsupp 3/5, part 2/9, supplier 2/7, nation 2/4 | 1.16 GiB | 3,416 | 478.5 KiB | 619.8 KiB | 1.6 MiB | 1,376 | 1,375 (+0.5%) | 1,376 (+0.0%) | 886 (+1.2%) | 820 (+2.5%) |
| Q10 | lineitem 4/16, orders 3/9, customer 7/8, nation 2/4 | 695.09 MiB | 2,418 | 187.0 KiB | 619.9 KiB | 3.4 MiB | 1,728 | 1,728 (+1.0%) | 1,728 (+0.0%) | 1,225 (+4.2%) | 1,102 (+16.3%) |
| Q11 | partsupp 4/5, supplier 2/7, nation 2/4 | 94.54 MiB | 268 | 249.7 KiB | 538.5 KiB | 538.9 KiB | 69 | 69 (+0.3%) | 69 (+0.0%) | 69 (+0.0%) | 69 (+0.0%) |
| Q12 | lineitem 5/16, orders 2/9 | 445.62 MiB | 2,691 | 190.2 KiB | 190.4 KiB | 478.7 KiB | 1,713 | 1,713 (+1.5%) | 1,713 (+0.0%) | 1,224 (+4.9%) | 1,222 (+5.0%) |
| Q13 | orders 3/9, customer 1/8 | 416.97 MiB | 382 | 594.8 KiB | 2.3 MiB | 2.3 MiB | 137 | 137 (+0.1%) | 137 (+0.0%) | 137 (+0.0%) | 136 (+0.0%) |
| Q14 | lineitem 4/16, part 2/9 | 709.31 MiB | 1,990 | 190.4 KiB | 619.9 KiB | 620.2 KiB | 1,501 | 1,501 (+0.8%) | 1,501 (+0.0%) | 1,500 (+0.0%) | 522 (+51.6%) |
| Q15 | lineitem 4/16, supplier 4/7 | 676.52 MiB | 1,960 | 291.0 KiB | 619.9 KiB | 2.7 MiB | 1,469 | 1,469 (+0.9%) | 1,469 (+0.0%) | 1,466 (+0.0%) | 490 (+16.1%) |
| Q16 | partsupp 2/5, part 4/9, supplier 2/7 | 58.86 MiB | 202 | 210.6 KiB | 481.0 KiB | 2.3 MiB | 102 | 102 (+0.7%) | 102 (+0.0%) | 102 (+0.0%) | 101 (+0.8%) |
| Q17 | lineitem 3/16, part 3/9 | 633.64 MiB | 1,518 | 597.0 KiB | 619.9 KiB | 620.2 KiB | 1,029 | 1,029 (+0.6%) | 1,029 (+0.0%) | 1,028 (+0.0%) | 522 (+43.3%) |
| Q18 | lineitem 2/16, orders 4/9, customer 2/8 | 376.00 MiB | 1,496 | 187.1 KiB | 594.9 KiB | 672.4 KiB | 1,237 | 1,236 (+1.3%) | 1,236 (+0.0%) | 1,114 (+1.0%) | 1,113 (+1.0%) |
| Q19 | lineitem 6/16, part 4/9 | 707.42 MiB | 3,002 | 90.7 KiB | 619.8 KiB | 620.2 KiB | 1,518 | 1,518 (+0.8%) | 1,518 (+0.0%) | 1,517 (+0.0%) | 522 (+84.3%) |
| Q20 | lineitem 4/16, partsupp 3/5, part 2/9, supplier 4/7, nation 2/4 | 771.91 MiB | 2,194 | 249.7 KiB | 597.0 KiB | 2.7 MiB | 1,552 | 1,551 (+0.8%) | 1,552 (+0.0%) | 1,063 (+1.8%) | 574 (+50.5%) |
| Q21 | lineitem 4/16, orders 2/9, supplier 3/7, nation 2/4 | 588.92 MiB | 2,207 | 190.2 KiB | 538.4 KiB | 539.0 KiB | 1,716 | 1,716 (+1.1%) | 1,716 (+0.0%) | 1,715 (+0.0%) | 1,103 (+60.4%) |
| Q22 | orders 1/9, customer 3/8 | 100.66 MiB | 162 | 594.8 KiB | 595.0 KiB | 1.4 MiB | 149 | 149 (+0.6%) | 149 (+0.0%) | 149 (+0.0%) | 148 (+0.8%) |
| **all 22** |  | **12.10 GiB** | **41,833** |  |  |  | **25,174** | **25,171 (+0.8%)** | **25,173 (+0.0%)** | **23,061 (+0.7%)** | **13,827 (+28.0%)** |

### 7.4 Multi-instance sensitivity (every table instance scanned separately)  [A]

Queries whose plan scans the same *large* table more than once read more bytes if the engine does not share the scan (rows with < 1% difference - Q7/Q8, which only re-scan the 25-row nation table - omitted):

| SF | Q | useful bytes (union) | useful bytes (multi) | delta | ranges (union) | ranges (multi) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| sf10 | Q2 | 97.47 MiB | 176.11 MiB | +81% | 278 | 482 |
| sf10 | Q11 | 94.54 MiB | 175.69 MiB | +86% | 268 | 470 |
| sf10 | Q15 | 676.52 MiB | 1.32 GiB | +99% | 1,960 | 3,916 |
| sf10 | Q17 | 633.64 MiB | 961.50 MiB | +52% | 1,518 | 2,496 |
| sf10 | Q18 | 376.00 MiB | 508.46 MiB | +35% | 1,496 | 2,474 |
| sf10 | Q21 | 588.92 MiB | 1.43 GiB | +148% | 2,207 | 5,141 |
| sf10 | Q22 | 100.66 MiB | 124.69 MiB | +24% | 162 | 188 |
| sf100 | Q2 | 974.66 MiB | 1.72 GiB | +81% | 2,676 | 4,654 |
| sf100 | Q11 | 945.05 MiB | 1.72 GiB | +86% | 2,628 | 4,604 |
| sf100 | Q15 | 6.86 GiB | 13.68 GiB | +99% | 19,572 | 39,108 |
| sf100 | Q17 | 6.45 GiB | 9.92 GiB | +54% | 15,141 | 24,909 |
| sf100 | Q18 | 3.73 GiB | 5.02 GiB | +35% | 14,898 | 24,666 |
| sf100 | Q21 | 6.00 GiB | 15.03 GiB | +150% | 22,007 | 51,311 |
| sf100 | Q22 | 1.04 GiB | 1.27 GiB | +23% | 1,590 | 1,836 |

## 8. Workload summary: how much of the scanned data can use the GDS path?  [D]

Definitions: for each request-formation variant, every request is classified by its own size. **Equal-weight** = mean over the 22 queries of (bytes in class / bytes requested by that query). **Whole benchmark** = sum of class bytes over all 22 queries / sum of requested bytes (so lineitem-heavy queries dominate). `<1 MiB` requests take the POSIX fallback of the cuFile reactor (threshold 1 MiB, `gds_threshold`); `>=1 MiB` are GDS-eligible. Union model unless stated.

### 8.1 sf100: share of requested bytes by request size

| variant | merge scope | EW <1 MiB | EW 1-16 MiB | EW >=16 MiB | whole <1 MiB | whole 1-16 MiB | whole >=16 MiB | total requests | requests >=1 MiB | over-read | bytes requested |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | n/a | 94.9% | 5.1% | 0.0% | 96.4% | 3.6% | 0.0% | 416,897 | 2,091 | 0.00% | 124.8 GiB |
| gap 0 | file-global | 70.0% | 29.7% | 0.4% | 68.5% | 31.4% | 0.1% | 250,852 | 29,459 | 0.00% | 124.8 GiB |
| gap 0 | per-RG | 70.0% | 30.0% | 0.0% | 68.5% | 31.5% | 0.0% | 252,210 | 30,807 | 0.00% | 124.8 GiB |
| 4K-aligned | file-global | 70.0% | 29.6% | 0.3% | 68.7% | 31.2% | 0.1% | 250,849 | 29,459 | 0.77% | 125.8 GiB |
| 4K-aligned | per-RG | 70.1% | 29.9% | 0.0% | 68.7% | 31.3% | 0.0% | 252,207 | 30,807 | 0.77% | 125.8 GiB |
| gap 4 KiB | file-global | 70.0% | 29.7% | 0.4% | 68.5% | 31.4% | 0.1% | 250,849 | 29,459 | 0.00% | 124.8 GiB |
| gap 4 KiB | per-RG | 70.0% | 30.0% | 0.0% | 68.5% | 31.5% | 0.0% | 252,207 | 30,807 | 0.00% | 124.8 GiB |
| gap 64 KiB | file-global | 67.2% | 31.7% | 1.1% | 64.5% | 34.5% | 1.0% | 229,959 | 29,338 | 0.65% | 125.6 GiB |
| gap 64 KiB | per-RG | 67.2% | 32.8% | 0.0% | 64.5% | 35.5% | 0.0% | 231,317 | 30,686 | 0.65% | 125.6 GiB |
| gap 1 MiB | file-global | 26.1% | 73.0% | 1.0% | 15.3% | 83.9% | 0.8% | 137,796 | 70,922 | 28.81% | 160.8 GiB |
| gap 1 MiB | per-RG | 26.1% | 73.9% | 0.0% | 15.3% | 84.7% | 0.0% | 139,154 | 72,271 | 28.81% | 160.8 GiB |

### 8.2 sf10: share of requested bytes by request size

| variant | merge scope | EW <1 MiB | EW 1-16 MiB | EW >=16 MiB | whole <1 MiB | whole 1-16 MiB | whole >=16 MiB | total requests | requests >=1 MiB | over-read | bytes requested |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | n/a | 94.8% | 5.2% | 0.0% | 96.3% | 3.7% | 0.0% | 41,833 | 209 | 0.00% | 12.1 GiB |
| gap 0 | file-global | 70.3% | 29.7% | 0.0% | 69.2% | 30.8% | 0.0% | 25,174 | 2,946 | 0.00% | 12.1 GiB |
| gap 0 | per-RG | 70.3% | 29.7% | 0.0% | 69.2% | 30.8% | 0.0% | 25,308 | 3,079 | 0.00% | 12.1 GiB |
| 4K-aligned | file-global | 70.4% | 29.6% | 0.0% | 69.3% | 30.7% | 0.0% | 25,171 | 2,946 | 0.80% | 12.2 GiB |
| 4K-aligned | per-RG | 70.4% | 29.6% | 0.0% | 69.3% | 30.7% | 0.0% | 25,305 | 3,079 | 0.80% | 12.2 GiB |
| gap 4 KiB | file-global | 70.3% | 29.7% | 0.0% | 69.2% | 30.8% | 0.0% | 25,173 | 2,946 | 0.00% | 12.1 GiB |
| gap 4 KiB | per-RG | 70.3% | 29.7% | 0.0% | 69.2% | 30.8% | 0.0% | 25,307 | 3,079 | 0.00% | 12.1 GiB |
| gap 64 KiB | file-global | 67.4% | 31.8% | 0.7% | 65.0% | 34.0% | 0.9% | 23,061 | 2,933 | 0.67% | 12.2 GiB |
| gap 64 KiB | per-RG | 67.4% | 32.6% | 0.0% | 65.0% | 35.0% | 0.0% | 23,195 | 3,066 | 0.67% | 12.2 GiB |
| gap 1 MiB | file-global | 26.5% | 72.9% | 0.7% | 15.9% | 83.3% | 0.7% | 13,827 | 7,090 | 27.97% | 15.5 GiB |
| gap 1 MiB | per-RG | 26.5% | 73.5% | 0.0% | 15.9% | 84.1% | 0.0% | 13,961 | 7,223 | 27.97% | 15.5 GiB |

`>=16 MiB` requests exist only in the file-global scope and are chain artifacts, not a property of the files. At sf100 they are (a) Q2 reading *every column* of supplier = the entire 75.4 MiB column-chunk region as one contiguous range (nothing can be pruned, so the whole table is one run), and (b) Q10's customer scan with a gap threshold >= 64 KiB: the 45.6 KiB c_mktsegment chunk is a hole smaller than 64 KiB, and because c_comment of row group *i* touches c_custkey of row group *i+1* the whole 1.15 GiB customer region chains into a single request. With per-RG merging the largest request is 9.7 MiB.

### 8.3 GDS-eligible share (bytes in requests >= 1 MiB), sensitivity: sf100

| variant | union/global EW | union/global whole | union/per-RG EW | union/per-RG whole | multi/global EW | multi/global whole | multi/per-RG EW | multi/per-RG whole |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 5.1% | 3.6% | = | = | 5.5% | 3.2% | = | = |
| gap 0 | 30.0% | 31.5% | 30.0% | 31.5% | 30.1% | 27.4% | 30.0% | 27.4% |
| 4K-aligned | 30.0% | 31.3% | 29.9% | 31.3% | 30.0% | 27.2% | 30.0% | 27.2% |
| gap 4 KiB | 30.0% | 31.5% | 30.0% | 31.5% | 30.1% | 27.4% | 30.0% | 27.4% |
| gap 64 KiB | 32.8% | 35.5% | 32.8% | 35.5% | 32.5% | 30.8% | 32.5% | 30.8% |
| gap 1 MiB | 73.9% | 84.7% | 73.9% | 84.7% | 74.2% | 85.7% | 74.2% | 85.7% |

### 8.4 GDS-eligible share (bytes in requests >= 1 MiB), sensitivity: sf10

| variant | union/global EW | union/global whole | union/per-RG EW | union/per-RG whole | multi/global EW | multi/global whole | multi/per-RG EW | multi/per-RG whole |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 5.2% | 3.7% | = | = | 5.5% | 3.3% | = | = |
| gap 0 | 29.7% | 30.8% | 29.7% | 30.8% | 29.7% | 26.9% | 29.7% | 26.9% |
| 4K-aligned | 29.6% | 30.7% | 29.6% | 30.7% | 29.7% | 26.8% | 29.7% | 26.8% |
| gap 4 KiB | 29.7% | 30.8% | 29.7% | 30.8% | 29.7% | 26.9% | 29.7% | 26.9% |
| gap 64 KiB | 32.6% | 35.0% | 32.6% | 35.0% | 32.3% | 30.5% | 32.3% | 30.5% |
| gap 1 MiB | 73.5% | 84.1% | 73.5% | 84.1% | 73.9% | 85.0% | 73.9% | 85.0% |

### 8.5 Per query (sf100, union): share of requested bytes in requests >= 1 MiB (GDS-eligible)

File-global merge scope; the per-RG scope gives the same values except where noted in the last column.

| Q | raw | gap 0 | 4K-aligned | gap 4 KiB | gap 64 KiB | gap 1 MiB | per-RG differs (>=0.5 pp) for |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Q1 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | - |
| Q2 | 6.2% | 7.7% | 7.7% | 7.7% | 7.7% | 88.7% | - |
| Q3 | 0.0% | 20.0% | 19.9% | 20.0% | 20.0% | 37.4% | - |
| Q4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | - |
| Q5 | 0.0% | 15.5% | 15.5% | 15.5% | 15.5% | 99.5% | - |
| Q6 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | - |
| Q7 | 0.0% | 14.4% | 14.3% | 14.4% | 14.4% | 99.5% | - |
| Q8 | 0.0% | 69.2% | 69.1% | 69.2% | 69.2% | 98.7% | - |
| Q9 | 2.1% | 57.6% | 57.6% | 57.6% | 87.4% | 93.7% | - |
| Q10 | 13.8% | 35.8% | 35.5% | 35.8% | 34.4% | 44.0% | - |
| Q11 | 0.0% | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% | - |
| Q12 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | - |
| Q13 | 67.1% | 98.6% | 98.6% | 98.6% | 98.6% | 98.6% | - |
| Q14 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 99.1% | - |
| Q15 | 0.5% | 0.7% | 0.7% | 0.7% | 0.7% | 100.0% | - |
| Q16 | 3.8% | 4.4% | 4.4% | 4.4% | 4.4% | 4.6% | gap 0, 4K-aligned, gap 4 KiB, gap 64 KiB, gap 1 MiB |
| Q17 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 98.6% | - |
| Q18 | 0.0% | 38.3% | 38.0% | 38.3% | 65.6% | 65.6% | - |
| Q19 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 99.0% | - |
| Q20 | 3.4% | 76.5% | 76.2% | 76.5% | 82.0% | 95.1% | - |
| Q21 | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | 81.9% | - |
| Q22 | 16.0% | 22.6% | 22.5% | 22.6% | 22.6% | 22.6% | - |

With gap-0 (free, over-read 0) merging, the GDS-eligible byte share per query is: >= 90%: Q1, Q11, Q13; 50-90%: Q8, Q9, Q20; 10-50%: Q3, Q5, Q7, Q10, Q18, Q22; < 10%: Q2, Q4, Q6, Q12, Q14, Q15, Q16, Q17, Q19, Q21.

### 8.6 Where do the benchmark's bytes come from (sf100, union)?  [D]

| table | useful bytes summed over 22 queries | share |
| :--- | ---: | ---: |
| lineitem | 100.7 GiB | 80.6% |
| orders | 17.3 GiB | 13.9% |
| partsupp | 3.5 GiB | 2.8% |
| part | 1.3 GiB | 1.0% |
| customer | 1.9 GiB | 1.5% |
| supplier | 217.0 MiB | 0.2% |
| nation | 3.7 KiB | 0.0% |
| region | 357 B | 0.0% |

## 9. Row-group skipping by min/max statistics  [M statistics; predicates are A; hypothetical rows are labelled]

**What was done.** Every query's literal predicates (standard validation parameters; only those that min/max can use, i.e. comparisons of a column with a constant, IN-lists and prefix LIKEs) were evaluated against the per-row-group `min_value`/`max_value` stored in the footers. A row group is skippable iff some conjunct is provably false for the whole [min, max] interval. Column-to-column predicates (`l_commitdate < l_receiptdate`), `LIKE '%x%'`, `<>`/`NOT LIKE` and HAVING aggregates cannot use min/max and were not evaluated (they can only make the true skip rate lower than reported). The evaluator was unit-tested (e.g. `o_orderkey <= 1,000,000` correctly keeps only RGs 0-2 of 1,221).

### 9.1 Measured result: skippable row groups per query  [M + A]

| Q | predicate columns evaluated | sf100 skippable RGs / RGs of predicate tables | sf10 skippable RGs / RGs | not evaluated (cannot use min/max) |
| :--- | :--- | :--- | :--- | :--- |
| Q1 | lineitem.l_shipdate | 0 / 4,884 | 0 / 489 |  |
| Q2 | part.p_size | 0 / 163 | 0 / 17 | p_type LIKE '%BRASS' (suffix), r_name = 'EUROPE' (1-RG table) not usable / irrelevant |
| Q3 | lineitem.l_shipdate; orders.o_orderdate; customer.c_mktsegment | 0 / 6,228 | 0 / 625 |  |
| Q4 | orders.o_orderdate | 0 / 1,221 | 0 / 123 | l_commitdate < l_receiptdate is column-to-column: not usable |
| Q5 | orders.o_orderdate | 0 / 1,221 | 0 / 123 |  |
| Q6 | lineitem.l_shipdate+l_discount+l_quantity | 0 / 4,884 | 0 / 489 |  |
| Q7 | lineitem.l_shipdate | 0 / 4,884 | 0 / 489 |  |
| Q8 | orders.o_orderdate; part.p_type | 0 / 1,384 | 0 / 140 |  |
| Q9 | (none usable) | - | - | p_name LIKE '%green%' not usable |
| Q10 | lineitem.l_returnflag; orders.o_orderdate | 0 / 6,105 | 0 / 612 |  |
| Q11 | (none usable) | - | - | n_name = 'GERMANY' on 1-RG table; no predicate on big tables |
| Q12 | lineitem.l_shipmode+l_receiptdate | 0 / 4,884 | 0 / 489 | l_commitdate < l_receiptdate, l_shipdate < l_commitdate are column-to-column: not usable |
| Q13 | (none usable) | - | - | o_comment NOT LIKE '%special%requests%' not usable |
| Q14 | lineitem.l_shipdate | 0 / 4,884 | 0 / 489 |  |
| Q15 | lineitem.l_shipdate | 0 / 4,884 | 0 / 489 |  |
| Q16 | part.p_size | 0 / 163 | 0 / 17 | p_brand <> 'Brand#45', p_type NOT LIKE 'MEDIUM POLISHED%' not usable; only p_size IN list evaluated |
| Q17 | part.p_brand+p_container | 0 / 163 | 0 / 17 |  |
| Q18 | (none usable) | - | - | HAVING sum(l_quantity) > 300 is an aggregate predicate: no min/max use |
| Q19 | lineitem.l_shipmode+l_shipinstruct+l_quantity; part.p_size+p_brand | 0 / 5,047 | 0 / 506 |  |
| Q20 | lineitem.l_shipdate; part.p_name | 0 / 5,047 | 0 / 506 |  |
| Q21 | orders.o_orderstatus | 0 / 1,221 | 0 / 123 |  |
| Q22 | (none usable) | - | - | substring(c_phone,1,2) IN (...) and c_acctbal > avg: not usable from RG stats |

**Result: 0 skippable row groups across all 22 queries at both scale factors** - min/max skipping removes nothing from this dataset; column pruning is the only byte reduction available.

### 9.2 Why: the files are clustered only by key, not by any filtered column  [M]

For each numeric/date column the table shows how much of the table-wide value range [global min, global max] a typical row group's own [min, max] covers (sf100). ~100% means every row group spans the whole domain, so any range predicate overlaps every row group.

| column | table-wide domain | RG span p50 | RG span p90 | RG span max | usable for RG pruning? |
| :--- | :--- | ---: | ---: | ---: | :--- |
| lineitem.l_orderkey | 1 .. 600,000,000 | 0.020% | 0.021% | 0.021% | yes (generation-order key) |
| lineitem.l_partkey | 1 .. 20,000,000 | 99.999% | 100.000% | 100.000% | no |
| lineitem.l_suppkey | 1 .. 1,000,000 | 99.999% | 100.000% | 100.000% | no |
| lineitem.l_linenumber | 1 .. 7 | 100.000% | 100.000% | 100.000% | no |
| lineitem.l_quantity | 1.00 .. 50.00 | 100.000% | 100.000% | 100.000% | no |
| lineitem.l_extendedprice | 900.05 .. 104948.50 | 99.498% | 99.811% | 99.997% | no |
| lineitem.l_discount | 0.00 .. 0.10 | 100.000% | 100.000% | 100.000% | no |
| lineitem.l_tax | 0.00 .. 0.08 | 100.000% | 100.000% | 100.000% | no |
| lineitem.l_shipdate | 1992-01-02 .. 1998-12-01 | 99.921% | 100.000% | 100.000% | no |
| lineitem.l_commitdate | 1992-01-31 .. 1998-10-31 | 99.959% | 100.000% | 100.000% | no |
| lineitem.l_receiptdate | 1992-01-03 .. 1998-12-31 | 99.608% | 99.765% | 100.000% | no |
| orders.o_orderkey | 1 .. 600,000,000 | 0.082% | 0.082% | 0.082% | yes (generation-order key) |
| orders.o_custkey | 1 .. 14,999,999 | 99.999% | 100.000% | 100.000% | no |
| orders.o_totalprice | 811.73 .. 591036.15 | 86.001% | 90.087% | 99.981% | no |
| orders.o_orderdate | 1992-01-01 .. 1998-08-02 | 100.000% | 100.000% | 100.000% | no |
| orders.o_shippriority | 0 .. 0 | 100.000% | 100.000% | 100.000% | no |
| partsupp.ps_partkey | 1 .. 20,000,000 | 0.154% | 0.154% | 0.154% | yes (generation-order key) |
| partsupp.ps_suppkey | 1 .. 1,000,000 | 78.072% | 100.000% | 100.000% | no |
| partsupp.ps_availqty | 1 .. 9,999 | 100.000% | 100.000% | 100.000% | no |
| partsupp.ps_supplycost | 1.00 .. 1000.00 | 99.999% | 100.000% | 100.000% | no |
| part.p_partkey | 1 .. 20,000,000 | 0.614% | 0.614% | 0.614% | yes (generation-order key) |
| part.p_size | 1 .. 50 | 100.000% | 100.000% | 100.000% | no |
| part.p_retailprice | 900.01 .. 2098.99 | 99.918% | 99.918% | 99.918% | no |
| customer.c_custkey | 1 .. 15,000,000 | 0.819% | 0.819% | 0.819% | yes (generation-order key) |
| customer.c_nationkey | 0 .. 24 | 100.000% | 100.000% | 100.000% | no |
| customer.c_acctbal | -999.99 .. 9999.99 | 99.999% | 100.000% | 100.000% | no |
| supplier.s_suppkey | 1 .. 1,000,000 | 12.288% | 12.288% | 12.288% | yes (generation-order key) |
| supplier.s_nationkey | 0 .. 24 | 100.000% | 100.000% | 100.000% | no |
| supplier.s_acctbal | -999.99 .. 9999.98 | 99.998% | 99.999% | 100.000% | no |

Only the generation-order key columns (`l_orderkey`, `o_orderkey`, `ps_partkey`, `p_partkey`, `c_custkey`, `s_suppkey`) are clustered (each RG covers 1/#RGs of the domain); no TPC-H query has a constant-range predicate on them (they appear only in joins, where min/max skipping would need a dynamic/semi-join filter). Every date, price, quantity, discount and low-cardinality column is uniformly scattered, so even `l_shipdate`/`o_orderdate` predicates (Q1, Q3-Q8, Q10, Q12, Q14, Q15, Q20) overlap every row group. Low-cardinality dictionary columns also carry every distinct value in every row group (checked below for the ten filter-relevant ones), so dictionary/bloom-filter based skipping cannot help either:

| column | row groups with distinct_count | max distinct_count seen in an RG | RGs containing ALL those distinct values |
| :--- | ---: | ---: | ---: |
| lineitem.l_shipmode | 4,884 | 7 | 4,884 of 4,884 (100.0%) |
| lineitem.l_returnflag | 4,884 | 3 | 4,884 of 4,884 (100.0%) |
| lineitem.l_shipinstruct | 4,884 | 4 | 4,884 of 4,884 (100.0%) |
| orders.o_orderstatus | 1,221 | 3 | 1,221 of 1,221 (100.0%) |
| orders.o_orderpriority | 1,221 | 5 | 1,221 of 1,221 (100.0%) |
| customer.c_mktsegment | 123 | 5 | 123 of 123 (100.0%) |
| part.p_brand | 163 | 25 | 163 of 163 (100.0%) |
| part.p_container | 163 | 40 | 163 of 163 (100.0%) |
| part.p_size | 163 | 50 | 163 of 163 (100.0%) |
| part.p_type | 163 | 150 | 163 of 163 (100.0%) |

### 9.3 Hypothetical (NOT measured): if a table were re-clustered by its filtered date column  [A: uniform-density approximation]

If lineitem were sorted by `l_shipdate` (or orders by `o_orderdate`), the fraction of row groups a date predicate keeps would be about (predicate interval / column domain), plus at most one boundary row group at each edge. The table applies that fraction to the bytes the query reads from that table and keeps every other table unchanged. Estimates only - assumes rows are uniformly dense in the date domain (TPC-H dates are close to uniform, with ~4-month ramps at the ends).

| Q | clustered column -> kept RGs | useful bytes now | useful bytes if clustered | remaining |
| :--- | :--- | ---: | ---: | ---: |
| Q1 | lineitem.l_shipdate keeps ~96.4% of RGs | 4.98 GiB | 4.80 GiB | 96.4% |
| Q3 | lineitem.l_shipdate keeps ~53.7% of RGs; orders.o_orderdate keeps ~48.6% of RGs | 6.51 GiB | 3.45 GiB | 53.0% |
| Q4 | orders.o_orderdate keeps ~3.8% of RGs | 3.47 GiB | 2.68 GiB | 77.0% |
| Q5 | orders.o_orderdate keeps ~15.2% of RGs | 8.39 GiB | 7.10 GiB | 84.6% |
| Q6 | lineitem.l_shipdate keeps ~14.5% of RGs | 4.48 GiB | 663.12 MiB | 14.5% |
| Q7 | lineitem.l_shipdate keeps ~28.9% of RGs | 9.06 GiB | 3.59 GiB | 39.7% |
| Q8 | orders.o_orderdate keeps ~30.4% of RGs | 11.53 GiB | 10.47 GiB | 90.8% |
| Q10 | orders.o_orderdate keeps ~3.8% of RGs | 6.84 GiB | 5.38 GiB | 78.6% |
| Q12 | lineitem.l_receiptdate keeps ~14.3% of RGs | 4.35 GiB | 1.14 GiB | 26.3% |
| Q14 | lineitem.l_shipdate keeps ~1.2% of RGs | 7.19 GiB | 182.06 MiB | 2.5% |
| Q15 | lineitem.l_shipdate keeps ~3.6% of RGs | 6.86 GiB | 298.60 MiB | 4.3% |
| Q20 | lineitem.l_shipdate keeps ~14.5% of RGs | 8.05 GiB | 1.97 GiB | 24.5% |

**Effect on the request pattern if skipping were possible.** Skipping removes whole row groups, so the *number* of requests and the bytes fall roughly in proportion to the kept fraction, but the *size* of each surviving request is unchanged (chunk sizes are fixed by the 122,880-row groups): the request-size classes in section 8 would stay the same and the GDS-eligible *share* of bytes would not improve. The only second-order effect is on file-global merging: skipped row groups break the cross-RG chains (matters only for the few chained cases noted above). Skipping at page granularity is impossible here (one data page per chunk, no page index).

## 10. Implications and caveats

### 10.1 What the numbers say about the GDS (>= 1 MiB) vs POSIX split  [D]

- **Uncoalesced column-chunk requests are almost entirely POSIX-sized**: 94.9% of requested bytes (EW) and 96.4% (whole) arrive in requests < 1 MiB. The few >= 1 MiB requests are the long-string chunks (o_comment in Q13, s_comment in Q2/Q16, c_comment/c_address/c_phone in Q10/Q22, p_name in Q9/Q20).

- **Gap-0 merging is the free lunch**: the files are gap-free, so adjacent projected columns merge with zero over-read; GDS-eligible bytes go from 3.6% to 31.5% (whole) / 5.1% to 30.0% (EW). Queries that already reach >= 90% GDS-eligible bytes at gap 0: Q1, Q11, Q13. Everything else stays mostly POSIX because their projected columns are separated by unprojected ones.

- **4 KiB alignment (`IO_BLOCK_SIZE`) / 'merge ranges closer than a page' is neutral for request counts here** (250,852 vs 250,849 requests over the benchmark: only chunks smaller than ~4 KiB can fall inside a page-sized hole - o_shippriority, the nation/region columns and a few columns of the short last row groups), **and costs +0.77% bytes** (widening every request edge to 4 KiB; 99.97% of chunk edges are unaligned).

- **A 64 KiB merge gap buys little** (+0.65% over-read, GDS-eligible whole 31.5% -> 35.5%) because holes between projected chunks are whole column chunks; **a 1 MiB gap is where the split changes**: GDS-eligible bytes 84.7% (whole, per-RG; 84.7% file-global) for **+28.8% over-read** (36.0 GiB of extra bytes at sf100). Queries still < 50% GDS-eligible even at 1 MiB (per-RG): Q3 (37%), Q4 (0%), Q10 (44%), Q12 (0%), Q16 (4%), Q22 (23%); 50-90%: Q2 (89%), Q18 (66%), Q21 (82%).

- **Knife edge for Q1-like projections**: lineitem columns 4-10 form one run of 1.044-1.047 MiB per row group (sf100 and sf10), only ~4.4% above the 1 MiB threshold (all 4,883 full row groups qualify; the short last one, 139.5 KiB, does not). A GDS threshold or request-size rule above ~1.05 MiB would flip all of Q1 to POSIX.

- **Request counts**: the benchmark needs 416,897 raw ranges, 250,852 after gap-0 merging, 137,796 after a 1 MiB gap (sf100, 22 queries). sf10 has ~10x fewer row groups, hence ~10x fewer requests of the same size.

- **Row-group skipping cannot reduce this workload** (section 9); and since each chunk is a single page with no page index, there is no finer-grained skipping either.

### 10.2 Caveats / limits of this analysis

- Column sets are derived by hand from the standard TPC-H SQL [A]; a real engine may read a somewhat different set (extra columns for late materialisation or row ids, fewer if a column is only needed by a predicate it can answer from statistics). The union model assumes shared scans; the multi model (7.4, 8.3-8.4) bounds the effect of re-scanning (Q2, Q11, Q15, Q17, Q18, Q21, Q22).
- 'Requests' are scan-level merged ranges. A reactor may split a very large range into several physical operations, and the GDS/POSIX decision is made on the *slice* size; ranges only slightly above 1 MiB (Q1) are sensitive to that.
- File-global merging at gap >= 64 KiB can chain through the whole file (Q10 customer, Q2 supplier at sf100); real implementations would cap/segment such requests, which is what the per-RG scope approximates.
- Footers, bloom filters and page indexes are excluded from scanned bytes (sf100 footers: 604 B for region up to 7.6 MiB for lineitem; the six big tables' footers are 8.4 KiB-7.6 MiB); they are separate reads, one footer per file.
- Both scale factors use identical 122,880-row groups, so *per-chunk* sizes do not depend on scale; this would change for other writers/row-group settings (e.g. byte-based parquet-mr row groups, ZSTD) - these files are a DuckDB-v1.4.4-specific layout.
- Page-cache state does not matter for this analysis (only offsets/sizes); nothing here measures throughput.


## 11. Reproduction

The analysis used a small pure-Python Thrift-compact-protocol reader for the Parquet footer and page headers (no `pyarrow`/`duckdb` was available in the pixi environment), followed by a column-pruned scan simulator whose merge rules were cross-checked against literal ports of the repo's own `coalesce_ranges` (`include/cucascade/io/io_utils.hpp`) and `uring_reactor::align_and_coalesce` on all 22 queries x 2 scale factors x 12 variant/scope combinations (528 checks, 0 mismatches). The stages were: (1) extract footers, (2) walk the page headers of every row group, (3) layout statistics, (4) scan simulation, (5) min/max row-group skipping with unit tests, (6) report assembly. The scripts are not part of the repository; they can be re-created or committed on request. The report's numbers reconcile to the byte (`4 + sum(chunk sizes) + bloom region + footer + 8 == file size` for all 16 files).
