/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file iobuf_fragmented_sink.hpp
 * @brief A folly::IOBuf-shaped receive sink whose chain may contain holes.
 *
 * ## Why
 *
 * Two wanted byte ranges `[a,b)` and `[c,d)` of one object separated by a gap `[b,c)` are fetched
 * today either as two GETs (one more round trip each) or as one GET whose gap bytes land in a
 * staging buffer (gap-sized memory plus a memcpy to split it). The sink describes the body of the
 * single GET `[a,d)` as a chain of destination segments and *holes*: one `recvmsg` scatters the
 * useful bytes straight into their final buffers and drops the hole bytes into a small, reusable,
 * process-wide scratch block. Larger GETs, no memory or memcpy penalty for the gap.
 *
 * ## Shape
 *
 * The vocabulary follows folly::IOBuf (`wrapBuffer`, `appendChain`-like `appendData`, `length`,
 * `computeChainDataLength`, `countChainElements`, `isChained`) without depending on folly. The
 * sink never owns the destination memory: data segments are borrowed `(pointer, length)` pairs
 * that must outlive the receive.
 *
 * ## Hot path
 *
 * The receive cursor is `(segment index, offset within segment)` plus the logical byte offset.
 * `fillIovecs` and `commit` start at the cursor and walk forward, so both are O(segments touched),
 * never O(chain). Neither allocates, throws or makes a virtual call. Up to
 * `kInlineSegments` segments live inline; only longer chains spill to a `std::vector`, and only at
 * construction time.
 *
 * A hole maps to iovecs over the shared scratch block. A hole longer than the scratch is covered by
 * several iovecs that point into the same block (they overlap); `recvmsg`/`readv` accept that, the
 * bytes are received and discarded. The scratch position of a hole byte is its offset within the
 * hole modulo the scratch size, so a hole no larger than the scratch leaves exactly its bytes in
 * `scratch[0, hole_length)` regardless of how the receive was split (useful for tests and
 * debugging; concurrent sinks overwrite one another's scratch bytes, which is harmless because
 * nothing reads them).
 *
 * ## Size
 *
 * `sizeof(IoBufFragmentedSink)` is 168 bytes on LP64: four inline 24-byte segments (96), one
 * `std::vector` (24), and six counters (count, total, data total, cursor, cursor segment, cursor
 * offset: 48). Moves are nothrow; a move is a flat copy of the inline part plus a pointer steal of
 * the spill vector. It is not trivially copyable because of that vector; it is a borrowed view of
 * the destinations, so copies are allowed and cheap (no allocation unless spilled).
 *
 * ## Integration plan (not wired into any reader yet)
 *
 * 1. The read planner coalesces adjacent slices whose gap is at most `coalesce_gap` into one
 *    physical GET with a fragmented sink (`[slice A data][hole][slice B data]`) instead of one GET
 *    per slice or a gap-filled staging buffer.
 * 2. The transport's receive path calls `fillIovecs` for each `recvmsg` and `commit` with the
 *    received byte count on completion; `scatter` handles the body bytes that arrived behind the
 *    response headers in the header scratch buffer.
 * 3. Short-read retry resumes at `cursor()`: the retry GET starts at logical object offset
 *    `a + cursor()`, which may lie inside the hole (fine: the sink keeps dropping those bytes).
 * 4. Content-Range validation is unchanged; `computeChainDataLength()` is the expected body length.
 * 5. Device destinations: the data segments are the host staging blocks; the H2D copy is unchanged.
 * 6. Later: N holes per GET (the API already allows it), and holes expressed as "don't care"
 *    receives (`IORING_OP_RECV` with `MSG_TRUNC`-style skipping) if the kernel ever offers that.
 *
 * See docs/io/iobuf-fragmented-sink.md for the long form.
 */

#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cucascade::io {

namespace detail {

/// Default hole scratch: 64 KiB, page-aligned, zero-initialised storage (BSS) so its pages are
/// only materialised by the kernel on the first hole receive. One definition per program.
inline constexpr std::size_t default_hole_scratch_size = std::size_t{64} * 1024;
alignas(4096) inline std::byte default_hole_scratch[default_hole_scratch_size];

inline std::atomic<std::byte*> hole_scratch_block{default_hole_scratch};
inline std::atomic<std::size_t> hole_scratch_size{default_hole_scratch_size};

}  // namespace detail

/**
 * @brief Receive sink describing one HTTP body as a chain of destination segments and holes.
 *
 * Not thread-safe: one sink is driven by one receiver at a time. The hole scratch is shared by all
 * sinks of the process.
 */
class IoBufFragmentedSink {
 public:
  /// One element of the chain. `data == nullptr` and `hole == true` for holes.
  struct Segment {
    std::byte* data;
    std::size_t length;
    bool hole;
  };

  /// Result of fillIovecs: iovecs written and the bytes they cover.
  struct Fill {
    std::size_t iov_count;
    std::size_t bytes;
  };

  /// Segments stored inline before the chain spills to the heap.
  static constexpr std::size_t kInlineSegments = 4;

  constexpr IoBufFragmentedSink() noexcept = default;

  // ---------------------------------------------------------------------------------------------
  // Construction (IOBuf-like factories). These may throw; the receive path never does.
  // ---------------------------------------------------------------------------------------------

  /// An empty chain.
  [[nodiscard]] static IoBufFragmentedSink create() noexcept { return IoBufFragmentedSink{}; }

  /// A chain of one data segment over `[buf, buf + len)`.
  /// @throws std::invalid_argument if `buf` is null or `len` is 0.
  [[nodiscard]] static IoBufFragmentedSink wrapBuffer(void* buf, std::size_t len)
  {
    IoBufFragmentedSink sink;
    sink.appendData(buf, len);
    return sink;
  }

  /// Append a data segment at the tail (IOBuf::appendChain analogue). Allowed mid-receive.
  /// @throws std::invalid_argument if `buf` is null or `len` is 0.
  void appendData(void* buf, std::size_t len)
  {
    if (buf == nullptr) { throw std::invalid_argument("IoBufFragmentedSink: null data segment"); }
    insert_at(_count, Segment{static_cast<std::byte*>(buf), len, false});
  }

  /// Append a hole of `len` bytes (received and discarded) at the tail. Allowed mid-receive.
  /// @throws std::invalid_argument if `len` is 0.
  void appendHole(std::size_t len) { insert_at(_count, Segment{nullptr, len, true}); }

  /// Prepend a data segment at the head.
  /// @throws std::invalid_argument if `buf` is null or `len` is 0.
  /// @throws std::logic_error once the receive has started (`cursor() != 0`).
  void prependData(void* buf, std::size_t len)
  {
    if (buf == nullptr) { throw std::invalid_argument("IoBufFragmentedSink: null data segment"); }
    require_unstarted();
    insert_at(0, Segment{static_cast<std::byte*>(buf), len, false});
  }

  /// Prepend a hole of `len` bytes at the head.
  /// @throws std::invalid_argument if `len` is 0.
  /// @throws std::logic_error once the receive has started (`cursor() != 0`).
  void prependHole(std::size_t len)
  {
    require_unstarted();
    insert_at(0, Segment{nullptr, len, true});
  }

  // ---------------------------------------------------------------------------------------------
  // IOBuf-like introspection
  // ---------------------------------------------------------------------------------------------

  /// Length of the segment the cursor is in (the head before the receive starts, like
  /// IOBuf::length on the head); 0 when the chain is empty or fully received.
  [[nodiscard]] std::size_t length() const noexcept
  {
    return _index < _count ? segs()[_index].length : 0;
  }

  /// Body length: all segments including holes.
  [[nodiscard]] constexpr std::size_t computeChainDataLength() const noexcept { return _total; }

  /// Bytes that land in data segments (holes excluded).
  [[nodiscard]] constexpr std::size_t computeChainDataLengthWithoutHoles() const noexcept
  {
    return _data_total;
  }

  [[nodiscard]] constexpr std::size_t countChainElements() const noexcept { return _count; }
  [[nodiscard]] constexpr bool isChained() const noexcept { return _count > 1; }
  [[nodiscard]] constexpr bool empty() const noexcept { return _count == 0; }

  /// First segment. Precondition: `!empty()`.
  [[nodiscard]] Segment const& front() const noexcept
  {
    assert(_count > 0);
    return segs()[0];
  }

  /// Last segment. Precondition: `!empty()`.
  [[nodiscard]] Segment const& back() const noexcept
  {
    assert(_count > 0);
    return segs()[_count - 1];
  }

  [[nodiscard]] std::span<Segment const> segments() const noexcept { return {segs(), _count}; }

  // ---------------------------------------------------------------------------------------------
  // Receive cursor (hot path: no allocation, no throw, no virtual call)
  // ---------------------------------------------------------------------------------------------

  /// Logical body offset received so far.
  [[nodiscard]] constexpr std::size_t cursor() const noexcept { return _cursor; }

  /// `computeChainDataLength() - cursor()`.
  [[nodiscard]] constexpr std::size_t remaining() const noexcept { return _total - _cursor; }

  /// True once every body byte has been received (also true for an empty chain).
  [[nodiscard]] constexpr bool done() const noexcept { return _cursor == _total; }

  /**
   * @brief Describe the next receive as iovecs starting at the cursor.
   *
   * Writes at most `max_iov` iovecs covering at most `max_bytes` bytes. Data segments map 1:1
   * (the first and last clipped); a hole maps to one or more iovecs over the shared scratch block.
   * Returns 0 iovecs only when `done()`, `max_iov == 0` or `max_bytes == 0`. Does not move the
   * cursor. O(iovecs written).
   */
  [[nodiscard]] Fill fillIovecs(iovec* out,
                                std::size_t max_iov,
                                std::size_t max_bytes) const noexcept
  {
    assert(out != nullptr || max_iov == 0);
    Segment const* const s = segs();
    std::size_t idx        = _index;
    std::size_t off        = _offset;
    std::size_t iov        = 0;
    std::size_t bytes      = 0;
    // A segment is only partially described when the iovec or byte budget runs out, which also
    // ends the loop, so the walk needs no early exits.
    while (idx < _count && iov < max_iov && bytes < max_bytes) {
      Segment const& seg     = s[idx];
      std::size_t const take = std::min(seg.length - off, max_bytes - bytes);
      if (!seg.hole) {
        out[iov++] = iovec{seg.data + off, take};
        bytes += take;
      } else {
        std::byte* const scratch = detail::hole_scratch_block.load(std::memory_order_relaxed);
        std::size_t const ssize  = detail::hole_scratch_size.load(std::memory_order_relaxed);
        std::size_t pos          = off < ssize ? off : off % ssize;
        std::size_t left         = take;
        while (left > 0 && iov < max_iov) {
          std::size_t const chunk = std::min(left, ssize - pos);
          out[iov++]              = iovec{scratch + pos, chunk};
          left -= chunk;
          pos = 0;
        }
        bytes += take - left;
      }
      ++idx;
      off = 0;
    }
    return Fill{iov, bytes};
  }

  /// Advance the cursor by `n` received bytes (`n` <= the last fill's bytes). O(segments crossed).
  void commit(std::size_t n) noexcept
  {
    assert(n <= remaining());
    _cursor += n;
    Segment const* const s = segs();
    while (n > 0) {
      std::size_t const avail = s[_index].length - _offset;
      if (n < avail) {
        _offset += n;
        return;
      }
      n -= avail;
      ++_index;
      _offset = 0;
    }
  }

  /**
   * @brief Copy path for body bytes that arrived in a foreign buffer (e.g. behind the response
   *        headers in the header scratch): copy the data parts into their segments, skip holes,
   *        and advance the cursor by `n`. Precondition: `n <= remaining()`. O(n).
   */
  void scatter(std::byte const* src, std::size_t n) noexcept
  {
    assert(n <= remaining());
    assert(src != nullptr || n == 0);
    _cursor += n;
    Segment const* const s = segs();
    while (n > 0) {
      Segment const& seg      = s[_index];
      std::size_t const avail = seg.length - _offset;
      std::size_t const take  = std::min(avail, n);
      if (!seg.hole) { std::memcpy(seg.data + _offset, src, take); }
      src += take;
      n -= take;
      if (take < avail) {
        _offset += take;
        return;
      }
      ++_index;
      _offset = 0;
    }
  }

  /// Rewind the cursor to the start of the body; the chain is kept.
  constexpr void reset() noexcept
  {
    _cursor = 0;
    _index  = 0;
    _offset = 0;
  }

  /// Drop the chain (spill capacity is kept for reuse) and rewind the cursor.
  void clear() noexcept
  {
    _spill.clear();
    _count      = 0;
    _total      = 0;
    _data_total = 0;
    reset();
  }

  // ---------------------------------------------------------------------------------------------
  // Hole scratch (process-wide)
  // ---------------------------------------------------------------------------------------------

  /**
   * @brief Inject the block hole bytes are received into; `(nullptr, 0)` restores the 64 KiB
   *        default. A configuration call: make it before receives start, not concurrently with
   *        fillIovecs (each fill reads the pair once; a concurrent change can pair an old block
   *        with a new size). The block must outlive every receive that uses it.
   */
  static void setHoleScratch(std::byte* block, std::size_t len) noexcept
  {
    assert((block == nullptr) == (len == 0));
    if (block == nullptr || len == 0) {
      block = detail::default_hole_scratch;
      len   = detail::default_hole_scratch_size;
    }
    detail::hole_scratch_size.store(len, std::memory_order_relaxed);
    detail::hole_scratch_block.store(block, std::memory_order_relaxed);
  }

  /// Size of the current hole scratch block.
  [[nodiscard]] static std::size_t holeScratchSize() noexcept
  {
    return detail::hole_scratch_size.load(std::memory_order_relaxed);
  }

  /// Start of the current hole scratch block.
  [[nodiscard]] static std::byte* holeScratch() noexcept
  {
    return detail::hole_scratch_block.load(std::memory_order_relaxed);
  }

 private:
  [[nodiscard]] Segment const* segs() const noexcept
  {
    return _count <= kInlineSegments ? _inline.data() : _spill.data();
  }

  void require_unstarted() const
  {
    if (_cursor != 0) {
      throw std::logic_error("IoBufFragmentedSink: cannot prepend after the receive started");
    }
  }

  void insert_at(std::size_t pos, Segment seg)
  {
    if (seg.length == 0) {
      throw std::invalid_argument("IoBufFragmentedSink: zero-length segment");
    }
    if (seg.length > std::numeric_limits<std::size_t>::max() - _total) {
      throw std::overflow_error("IoBufFragmentedSink: chain length overflows size_t");
    }
    if (_count < kInlineSegments) {
      std::copy_backward(_inline.begin() + static_cast<std::ptrdiff_t>(pos),
                         _inline.begin() + static_cast<std::ptrdiff_t>(_count),
                         _inline.begin() + static_cast<std::ptrdiff_t>(_count + 1));
      _inline[pos] = seg;
    } else {
      if (_count == kInlineSegments) {
        _spill.reserve(2 * kInlineSegments);
        _spill.assign(_inline.begin(), _inline.end());
      }
      _spill.insert(_spill.begin() + static_cast<std::ptrdiff_t>(pos), seg);
    }
    ++_count;
    _total += seg.length;
    if (!seg.hole) { _data_total += seg.length; }
    // Prepends only happen before the receive starts (cursor at (0, 0)); an append at the tail
    // leaves (_index, _offset) valid, including the "fully received" state _index == old count.
  }

  std::array<Segment, kInlineSegments> _inline {};
  std::vector<Segment> _spill{};  ///< all segments once _count > kInlineSegments
  std::size_t _count{0};
  std::size_t _total{0};
  std::size_t _data_total{0};
  std::size_t _cursor{0};
  std::size_t _index{0};   ///< cursor segment
  std::size_t _offset{0};  ///< cursor offset within segment _index
};

static_assert(std::is_nothrow_move_constructible_v<IoBufFragmentedSink>);
static_assert(std::is_nothrow_move_assignable_v<IoBufFragmentedSink>);

}  // namespace cucascade::io
