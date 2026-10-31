# `IoBufFragmentedSink`: receiving one GET into several buffers with holes

Header: `include/cucascade/io/iobuf_fragmented_sink.hpp` (namespace `cucascade::io`, header-only).
Tests: `test/io/test_iobuf_fragmented_sink.cpp` (`cucascade_io_tests "[iobuf_sink]"`).
Benchmark: `benchmark/iobuf_fragmented_sink_bench.cpp` (`cucascade_iobuf_sink_benchmark`).

Status: standalone. No reader uses it yet; this document is the plan for wiring it in.

## Problem

A parquet scan asks for many byte ranges of one object. Two wanted ranges `[a,b)` and `[c,d)`
separated by a gap `[b,c)` are fetched today either

- as two GETs, paying one more request/response round trip, or
- as one GET of `[a,d)` into a staging buffer, paying gap-sized memory plus a memcpy to split the
  two ranges out of it.

For small gaps both are waste. The kernel can already scatter a stream into several buffers
(`recvmsg` with an iovec array); what was missing is a description of "these body bytes go here,
those are thrown away" that the receive loop can turn into iovecs cheaply.

## Model

The sink is a chain of *segments* in body order. A segment is either a borrowed destination
buffer (`data`, `length`) or a *hole* (`length` bytes that are received and discarded). The
vocabulary is folly::IOBuf's (`create`, `wrapBuffer`, `appendData` ~ `appendChain`, `length`,
`computeChainDataLength`, `countChainElements`, `isChained`, `front`, `back`), without a folly
dependency.

```
body:   [a ............. b)[b ..... c)[c ............. d)
chain:  [ data: slice A   ][  hole    ][ data: slice B   ]
iovecs:  A+off, len         scratch,..  B, len
```

Hole bytes land in one process-wide scratch block (64 KiB, page-aligned BSS by default;
`setHoleScratch` injects another). A hole longer than the scratch maps to several iovecs over the
same block; overlapping iovecs are legal for `recvmsg`/`readv`. A hole byte goes to scratch
position `hole offset mod scratch size`, so a hole that fits in the scratch leaves exactly its
bytes in `scratch[0, hole length)` no matter how the receive was split. Many sinks share the
scratch concurrently; nothing reads it, so the overwrites are harmless.

## Receive cursor (hot path)

| call | cost | notes |
|---|---|---|
| `fillIovecs(out, max_iov, max_bytes)` | O(iovecs written) | starts at the cursor; data segments map 1:1 (clipped); does not move the cursor |
| `commit(n)` | O(segments crossed) | `n` = bytes the kernel reported for the last fill |
| `scatter(src, n)` | O(n) | copy path for body bytes that arrived elsewhere (behind the headers); holes are skipped; advances the cursor |
| `cursor()`, `remaining()`, `done()` | O(1) | `cursor()` is the logical body offset |
| `reset()` / `clear()` | O(1) | rewind / drop the chain (spill capacity is kept) |

The cursor is `(segment index, offset within segment)` plus the logical offset, so neither call
walks the chain from the start. Nothing on this path allocates, throws or dispatches virtually.
`fillIovecs` returns 0 iovecs only when the sink is `done()` or a budget is 0. Construction
(`wrapBuffer`, `append*`, `prepend*`) throws `std::invalid_argument` for null data or zero
length; the receive calls only `assert` in debug builds.

Storage: up to 4 segments inline, a `std::vector` beyond that (allocated at construction, never
on the receive path). `sizeof` is 168 bytes on LP64.

## Cost

`cucascade_iobuf_sink_benchmark`, Release, one pinned core (aarch64 host), one emulated 256 KiB
receive per iteration (`fillIovecs(64, 256 KiB)` + `commit`) over an 8 MiB body, zero heap
allocations asserted by a counting `operator new`:

| body | ns per fill+commit | ns per MiB | iovecs per call |
|---|---|---|---|
| one 8 MiB segment | 5.7 | 22.7 | 1 |
| 3.5 MiB + 1 MiB hole + 3.5 MiB | 6.9 | 27.4 | 1.375 |
| 8 data segments, 7 holes of 64 KiB (spilled chain) | 7.5 | 30.0 | 1.44 |

The kernel copy of 1 MiB costs tens of microseconds, so the bookkeeping is noise.

## Integration plan

1. **Planner.** The read planner coalesces adjacent slices whose gap is at most `coalesce_gap`
   into one physical GET with a fragmented sink (`[slice A data][hole][slice B data]`), instead
   of one GET per slice or a gap-filled staging buffer. The first version emits at most one hole
   in the middle (two data segments); the sink already accepts N holes and holes at either end.
2. **Transport.** The receive path calls `fillIovecs` before each `recvmsg` (or
   `IORING_OP_RECVMSG`) and `commit(n)` on its completion. Body bytes that arrived behind the
   response headers in the header scratch buffer go through `scatter`.
3. **Short-read retry** resumes at `cursor()`: the retry GET starts at object offset
   `a + cursor()`, which may lie inside the hole; the sink keeps dropping those bytes.
4. **Validation.** Content-Range validation is unchanged; `computeChainDataLength()` is the
   expected body length.
5. **Device destinations.** The data segments are the host staging blocks; the H2D copy is
   unchanged.
6. **Later.** N holes per GET from the planner, and holes expressed as "don't care" receives
   (`IORING_OP_RECV` with `MSG_TRUNC`-style skipping) if the kernel ever offers that.
