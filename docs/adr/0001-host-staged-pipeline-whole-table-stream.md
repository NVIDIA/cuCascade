# Host-staged pipeline streams the whole table, sized up front

The host-staged route used to stage each leaf buffer in full, with a host sync after every buffer and an extra mid-copy sync per strings column to learn its chars size. We replaced it with a host-staged pipeline: all of a table's leaf buffers flow as one byte stream, cut into chunks that may span buffer boundaries. The source-to-host leg runs on a stream borrowed from the source GPU's stream pool, and the host-to-destination leg runs on the destination's target stream. Events on each staging slot let the two legs overlap. Strings chars sizes are learned before the pipeline starts, through one batched device-to-host read of every strings column's last offset, so the whole copy is planned in advance and there is a single sync at the end.

## Considered Options

- **Pipeline within each buffer only**: rejected. Small buffers (null masks, offsets) would get no overlap, and the sync per buffer would remain.
- **Stream offsets first, then plan chars**: rejected. It keeps a sync in the middle of the copy.
- **Start a chunk's host-to-destination copy before every destination buffer is allocated**: rejected. While any host-to-device copy is pending on the destination GPU, on whichever stream, an allocation from its memory pool takes 4 to 8 times longer (12-29 us instead of 3.4 us), which made tables up to 80 MiB slower. All destination buffers are allocated first; only source-to-host copies overlap the allocations.

## Consequences

- The old per-buffer host staging is deleted. The benchmark compares the pipeline with a bulk-copy proxy instead: depth 1 and one chunk covering the whole table, which moves the table in one batch per leg with one sync and overlaps nothing. The proxy is not the old code and is the harder baseline: the old code measured 3.6 times slower than the proxy for 320 KiB tables, 1.4 times for 320 MiB and 1.3 times for 2.5 GiB.
- A table smaller than one chunk is a single chunk, so there is no separate size threshold below which pipelining is skipped.
- Chunk size, pipeline depth and a copy-route override (auto / force peer / force host-staged) live in one config struct passed when the built-in converters are registered. The converter function signature does not change.
- Acceptance: for tables of 320 MiB and more (1M rows and up in the benchmark, which converts eight tables per batch), reach more than 90% of the raw overlapped source-to-host plus host-to-destination bound and be clearly faster than a bulk copy. Smaller tables are not the target; they must not get slower than a bulk copy. Choosing default chunk size and depth was a separate tuning phase.
- Inside a staging slot, every piece of a leaf buffer of 32 KiB or more starts on a host page boundary. A host-to-device copy from an unaligned host address ran about a quarter slower; the padding costs staging bytes, not copy time.
- Tuning outcome: the defaults stay at 8 MiB chunks and depth 4, the best or tied setting for tables of 320 MiB to 2.5 GiB, where the pipeline reaches 94% to 99% of the raw bound and runs 1.9 to 2.0 times faster than the bulk-copy proxy and 2.5 to 2.7 times faster than the old code. Tables that fit one chunk (5 MiB and less in the benchmark) take the proxy's path and tie with it; they are 2.9 to 3.7 times faster than the old code.
