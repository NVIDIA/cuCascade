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

// Shared cache entry types used by both prefetching_cache and the IO context
// virtual interface (device_read_async_io_using).  Extracted here to break the
// circular include between io_context.hpp and prefetching_cache.hpp.

#include <cucascade/exec/spin_lock.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace cucascade::io::cache {

// ---------------------------------------------------------------------------
// Page-alignment helpers
// ---------------------------------------------------------------------------
//
// @c a must be a power of two (in practice @c io::IO_BLOCK_SIZE, the O_DIRECT
// page size).  @c align_down rounds @p x down to the nearest multiple of @p a;
// @c align_up rounds up.  Used to page-align cache sub-range reads so partial
// fills stay O_DIRECT-compatible — never hardcode 4096 at call sites.

[[nodiscard]] constexpr std::size_t align_down(std::size_t x, std::size_t a) noexcept
{
  return x & ~(a - 1);
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t x, std::size_t a) noexcept
{
  return (x + a - 1) & ~(a - 1);
}

/**
 * @brief How the prefetching layer should behave on top of a given backend.
 *
 * - @c none: no prefetching.  Either the backend does not support vector host
 *   reads (so the prefetcher cannot batch range requests cheaply) or the
 *   backend explicitly opted out.
 * - @c immediate: prefill the cache ahead of consumer demand.
 * - @c opportunistic: read-ahead on demand — issue extra IO only when triggered by a
 *   consumer read.
 * - @c disposable: prefetching is temporary and can be discarded when no longer needed.
 */
enum class prefetching_stage { none, opportunistic, immediate, just_in_time, disposable };

// ---------------------------------------------------------------------------
// buffer_pool — growable pool of pinned chunks
// ---------------------------------------------------------------------------
//
// Backed by a @c cucascade::memory::fixed_size_host_memory_resource.  Each
// grow step requests CHUNKS_PER_SLAB blocks from the upstream resource and
// appends the raw pointers to an internal free list.  Blocks are never
// returned to the upstream resource until the pool is destroyed; allocate()
// pops from the free list and deallocate() pushes back.
//
// The chunk size is taken from @c mr.get_block_size() — all cache layout
// arithmetic that needs the chunk size reads it from @c chunk_bytes().

class buffer_pool {
 public:
  /// @p initial_slabs slabs are allocated up-front from @p mr (clamped to
  /// @p max_slabs).  Default preserves the historical behaviour of warming
  /// the pool with up to 10 slabs at construction.
  buffer_pool(cucascade::memory::memory_reservation_manager& reservation_manager,
              double reservation_fraction_for_prefetching = 0.0,
              double max_prefetching_budget_fraction      = 0.0);

  ~buffer_pool();

  buffer_pool(buffer_pool const&)            = delete;
  buffer_pool& operator=(buffer_pool const&) = delete;

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if the pool is
  /// exhausted and cannot grow).
  std::vector<std::byte*> allocate_bulk(size_t n, int& numa_node);

  std::vector<std::byte*> allocate_bulk_from(size_t n, int numa_node);

  void deallocate_bulk(std::vector<std::byte*>&& out, int numa) noexcept;

  [[nodiscard]] size_t chunk_size() const noexcept { return _chunk_bytes; }

  [[nodiscard]] size_t total_allocated_bytes() const noexcept
  {
    return _n_allocated_chunks * _chunk_bytes;
  }

  [[nodiscard]] size_t total_allocated_chunks() const noexcept { return _n_allocated_chunks; }

  [[nodiscard]] size_t reservation_size_for_prefetching() const noexcept;

  [[nodiscard]] size_t max_allowed_budget_for_prefetching() const noexcept;

  [[nodiscard]] size_t max_system_wide_usage() const noexcept;

  [[nodiscard]] bool should_start_evicting() const noexcept;

 private:
  struct host_arena {
    int numa_id;
    std::unique_ptr<cucascade::memory::reservation> reservation;
    cucascade::memory::fixed_size_host_memory_resource* mr;
  };

  size_t _chunk_bytes{1};
  size_t _reserved_size{0};
  size_t _max_allowed_budget_for_prefetching{0};
  std::unordered_map<int, size_t> _numa_to_arena_index;
  std::vector<host_arena> _host_arenas;
  std::atomic<size_t> _n_allocated_chunks{0};
};

// ---------------------------------------------------------------------------
// entry_state — mutex-guarded state + pin_count
// ---------------------------------------------------------------------------
//
// Holds a state enum and a reader pin count guarded by a single lock
// (@c entry_lock).  Every transition takes the lock, verifies its precondition,
// and mutates plain members, which eliminates the TOCTOU race between checking
// state and modifying pin_count.  @c entry_lock is a type alias — currently a
// @c cucascade::exec::spin_lock (a folly-derived micro spinlock).  The critical
// sections are a handful of branch/assign instructions, so a spinlock avoids the
// syscall overhead of a blocking mutex under the fine-grained per-chunk locking
// here.  There is no blocking wait: a reader that observes an in-flight `loading`
// chunk does not park on it — @c acquire_read() / @c mark_loading() simply fail
// and the reader falls back to reading the bytes itself.
//
// State machine — each row is the complete set of valid outbound transitions
// for that state.  Any other transition is rejected by the corresponding
// method's precondition CAS (return value == false).
//
//   empty      ──mark_queued()──►       queued
//   queued     ──mark_allocated()──►    allocated
//   allocated  ──mark_loading()──►      loading
//   allocated  ──mark_evicting()──►     evicting
//   loading    ──mark_cached()──►       cached
//   loading    ──mark_loading_in_use()──►       in_use(pin = 1)
//   loading    ──mark_load_failed()──►    allocated        (IO failure)
//   cached     ──mark_evicting()──►     evicting
//   cached     ──acquire_read()──►      in_use(pin = 1)
//   in_use     ──acquire_read()──►      in_use(pin += 1)
//   in_use     ──release_read()──►      in_use(pin -= 1) | cached (when pin → 0)
//   evicting   ──mark_empty()──►        empty
//
// `empty` is the only state with no inbound transitions other than from
// `evicting` — once an entry leaves `empty`, it can only return through the
// `evicting` reclamation path.  `evicting` is a one-way transit state.

using entry_lock = cucascade::exec::spin_lock;

class entry_state {
 public:
  enum value : uint8_t {
    empty     = 0,
    queued    = 1,  ///< registered for prefetch, not yet given chunks
    allocated = 2,  ///< chunks assigned, IO not yet dispatched
    loading   = 3,
    cached    = 4,
    in_use    = 5,
    evicting  = 6,
  };

  entry_state() noexcept = default;

  [[nodiscard]] value get_state() const noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    return _state;
  }

  [[nodiscard]] uint32_t get_pin_count() const noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    return _pins;
  }

  /// empty → queued.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_queued() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != empty) return false;
    _state = queued;
    return true;
  }

  /// queued → allocated.  Returns false on precondition mismatch.  Called by
  /// the allocator when it attaches chunks to a previously-queued entry.
  [[nodiscard]] bool mark_allocated() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != queued) return false;
    _state = allocated;
    return true;
  }

  /// loading → allocated (IO-failure revert).  Returns false on precondition
  /// mismatch.  Used by io_dispatch_loop's failure paths to revert an entry
  /// whose IO did not complete: the entry's chunks stay attached so a
  /// subsequent allocated-steal read can retry the load with a fresh
  /// request_context, instead of discarding the entry to `empty` and forcing
  /// the next reader through a fresh queue/allocate roundtrip.
  [[nodiscard]] bool mark_load_failed() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != loading) return false;
    _state = allocated;
    return true;
  }

  /// allocated → loading.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_loading() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != allocated) return false;
    _state = loading;
    return true;
  }

  /// loading → cached.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_cached() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != loading) return false;
    _state = cached;
    return true;
  }

  /// loading → in_use(pin = 1).  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_loading_in_use() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != loading) return false;
    _state = in_use;
    _pins  = 1;
    return true;
  }

  /// allocated → evicting, or cached → evicting.  Returns false on
  /// precondition mismatch.  Both source states have pin_count == 0 by
  /// invariant (allocated is set with pin==0 by mark_allocated(); cached is
  /// only entered from in_use via release_read() when pin → 0), so the two
  /// accepted source states below cover every legal transition exactly.
  [[nodiscard]] bool mark_evicting() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if ((_state != allocated && _state != cached) || _pins != 0) return false;
    _state = evicting;
    return true;
  }

  /// evicting → empty.  Returns false on precondition mismatch.
  [[nodiscard]] bool mark_empty() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != evicting) return false;
    _state = empty;
    return true;
  }

  /// (cached | in_use) → in_use with pin_count += 1.
  /// Returns false if the entry is not in a readable state.
  [[nodiscard]] bool acquire_read() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    if (_state != cached && _state != in_use) return false;
    _state = in_use;
    ++_pins;
    return true;
  }

  /// Decrement pin_count.  If it reaches 0, transition in_use → cached.
  /// Returns true if this was the last reader.
  bool release_read() noexcept
  {
    std::lock_guard<entry_lock> lk(_mtx);
    assert(_state == in_use && _pins > 0);
    --_pins;
    if (_pins == 0) { _state = cached; }
    return _pins == 0;
  }

  /// Acquire the entry lock and hand it to the caller.  Lets a caller perform a
  /// multi-step read-modify-write against the entry (e.g. merging cache_from)
  /// atomically with respect to the state transitions above.
  [[nodiscard]] std::unique_lock<entry_lock> get_lock() noexcept
  {
    return std::unique_lock<entry_lock>(_mtx);
  }

  /// Read the current state WITHOUT locking.  Precondition: the caller must
  /// already hold this entry's lock (via @c get_lock()); otherwise the read
  /// races with concurrent transitions.
  [[nodiscard]] value state_locked() const noexcept { return _state; }

 private:
  value _state{empty};
  uint32_t _pins{0};
  // spin_lock is a POD with no constructor; value-initialize so it starts in the
  // FREE (all-bits-zero) state rather than indeterminate.
  mutable entry_lock _mtx{};
};

struct alignas(64) chunk_lifecycle {
  std::atomic<uint64_t> packed{0};

  static constexpr uint64_t INSERT_MASK = 0xFFFFull;
  static constexpr uint64_t READ_SHIFT  = 16;
  static constexpr uint64_t READ_MASK   = 0xFFFFull << READ_SHIFT;
  static constexpr uint64_t TICK_SHIFT  = 32;
  static constexpr uint64_t TICK_MASK   = 0xFFFFFFFFull << TICK_SHIFT;
  static constexpr uint64_t FRESH_SCORE = 4;

  static constexpr uint64_t pack(uint32_t tick, uint16_t reads, uint16_t inserts) noexcept
  {
    return (uint64_t(tick) << TICK_SHIFT) | (uint64_t(reads) << READ_SHIFT) | uint64_t(inserts);
  }

  void on_request(uint32_t query_tick) noexcept
  {
    uint64_t cur = packed.load(std::memory_order_relaxed);
    for (;;) {
      auto cur_tick    = uint32_t(cur >> TICK_SHIFT);
      auto cur_reads   = uint16_t((cur >> READ_SHIFT) & 0xFFFFu);
      auto cur_inserts = uint16_t(cur & INSERT_MASK);

      uint64_t next;
      if (query_tick > cur_tick) {
        // New query — reset counters to reflect just this insert.
        next = pack(query_tick, 0, 1);
      } else {
        // Same tick (or stale tick — treat as same). Increment inserts,
        // saturate at uint16 max to avoid wrap.
        uint16_t new_inserts =
          cur_inserts == 0xFFFFu ? uint16_t{0xFFFFu} : static_cast<uint16_t>(cur_inserts + 1);
        next = pack(cur_tick, cur_reads, new_inserts);
      }

      if (packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    }
  }

  void on_consume() noexcept
  {
    uint64_t cur = packed.load(std::memory_order_relaxed);
    for (;;) {
      auto cur_reads   = uint16_t((cur >> READ_SHIFT) & 0xFFFFu);
      auto cur_inserts = uint16_t(cur & INSERT_MASK);

      // Clamp: never read more than we promised.
      if (cur_reads >= cur_inserts) return;

      uint64_t next = (cur & ~READ_MASK) | (uint64_t(cur_reads + 1) << READ_SHIFT);

      if (packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    }
  }

  struct snapshot {
    uint32_t tick;
    uint16_t reads;
    uint16_t inserts;

    [[nodiscard]] uint16_t eviction_tier(uint32_t query_tick) const noexcept
    {
      return query_tick > tick
               ? 0
               : (inserts > reads ? std::min<uint16_t>(inserts - reads, FRESH_SCORE) : 0);
    }
  };

  [[nodiscard]] snapshot load() const noexcept
  {
    uint64_t v = packed.load(std::memory_order_acquire);
    return {
      uint32_t(v >> TICK_SHIFT),
      uint16_t((v >> READ_SHIFT) & 0xFFFFu),
      uint16_t(v & INSERT_MASK),
    };
  }
};

// ---------------------------------------------------------------------------
// cache_entry — per-range metadata
// ---------------------------------------------------------------------------
//
// State transitions are managed by the entry_state class above.
// See entry_state's state machine diagram for the full picture.

struct alignas(64) cached_chunk {
  explicit cached_chunk(size_t off) : offset(off) {}

  std::size_t offset;
  uint8_t* data;
  int numa_node{-1};
  entry_state state;
  // Signed, page-aligned extent (in bytes, relative to @c offset) that this
  // chunk's buffer is populated with:
  //   0    -> the whole chunk is populated (a "full" chunk);
  //   +n   -> only the left prefix [offset, offset + n) is populated;
  //   -n   -> only the right suffix [offset + chunk_size - n, offset + chunk_size)
  //           is populated.
  // See @c needed_cache_from / @c merge_cache_from / @c chunk_covers below.
  std::atomic<int32_t> cache_from{0};
  chunk_lifecycle lifecycle;
};

// The signed, page-aligned @c cache_from a single request over the byte range
// [@p req_lo, @p req_hi) (NOT yet clamped to the chunk) implies for the chunk at
// [@p chunk_off, @p chunk_off + @p chunk_size).  Returns 0 for a full (or
// non-overlapping) chunk, +n for a left prefix, -n for a right suffix.  The
// magnitudes are page-aligned (io::IO_BLOCK_SIZE) so the resulting sub-range
// reads stay O_DIRECT-compatible, matching the partial reads in prefetch_loop
// and device_read_async.
[[nodiscard]] inline int32_t needed_cache_from(size_t chunk_off,
                                               size_t chunk_size,
                                               size_t req_lo,
                                               size_t req_hi) noexcept
{
  const size_t page = io::IO_BLOCK_SIZE;
  const size_t lo   = std::max(req_lo, chunk_off);
  const size_t hi   = std::min(req_hi, chunk_off + chunk_size);
  if (lo >= hi) { return 0; }  // no overlap -- caller should not call in this case
  if (lo <= chunk_off && hi >= chunk_off + chunk_size) { return 0; }  // full chunk
  if (lo <= chunk_off) {
    // Touches the left edge -> populate a page-aligned left prefix.
    const size_t bytes = std::min(align_up(hi - chunk_off, page), chunk_size);
    return static_cast<int32_t>(bytes);
  }
  // Right side -> populate a page-aligned right suffix.
  const size_t right_bytes = (chunk_off + chunk_size) - align_down(lo, page);
  return -static_cast<int32_t>(right_bytes);
}

// Fold @p want into @p c.cache_from under the merge rule.  The caller MUST hold
// @c c.state.get_lock() so the read-modify-write is atomic with respect to the
// state transitions that read cache_from.
//   0 (either side) wins        -> the chunk is/becomes full;
//   same sign                   -> keep the larger magnitude (wider coverage);
//   opposite signs              -> the two sides together span the chunk -> full.
inline void merge_cache_from(cached_chunk& c, int32_t want) noexcept
{
  const int32_t cur = c.cache_from.load(std::memory_order_relaxed);
  if (cur == 0) { return; }  // already full
  if (want == 0) {
    c.cache_from.store(0, std::memory_order_relaxed);
    return;
  }
  const bool same_sign = (cur > 0) == (want > 0);
  if (!same_sign) {
    c.cache_from.store(0, std::memory_order_relaxed);  // opposite sides cover the whole chunk
    return;
  }
  const int32_t cur_mag  = cur < 0 ? -cur : cur;
  const int32_t want_mag = want < 0 ? -want : want;
  c.cache_from.store(want_mag > cur_mag ? want : cur, std::memory_order_relaxed);
}

// True iff the populated extent of @p c covers the request [@p req_lo, @p req_hi)
// (which the caller has clamped to the chunk's extent).
[[nodiscard]] inline bool chunk_covers(const cached_chunk& c,
                                       size_t chunk_size,
                                       size_t req_lo,
                                       size_t req_hi) noexcept
{
  const int32_t cf = c.cache_from.load(std::memory_order_relaxed);
  if (cf == 0) { return true; }  // full chunk
  if (cf > 0) { return req_hi <= c.offset + static_cast<size_t>(cf); }
  return req_lo >= c.offset + chunk_size - static_cast<size_t>(-cf);
}

// The half-open file byte span [seg_lo, seg_hi) that must be read to populate a
// chunk at @p offset (size @p chunk_size) to exactly the extent that its stored
// @p cache_from advertises.  cache_from is edge-anchored, so the fill span is
// too — deriving the span FROM cache_from (rather than from the request range)
// guarantees the bytes read match the bytes @c chunk_covers will later claim.
// This is the single source of truth shared by prefetch_loop and
// device_read_async's load branch.
[[nodiscard]] inline std::pair<size_t, size_t> chunk_fill_span(size_t offset,
                                                               size_t chunk_size,
                                                               int32_t cache_from,
                                                               size_t page) noexcept
{
  if (cache_from == 0) { return {offset, offset + chunk_size}; }  // full chunk
  if (cache_from > 0) {
    return {offset, offset + std::min(chunk_size, align_up(static_cast<size_t>(cache_from), page))};
  }
  return {offset + align_down(chunk_size - static_cast<size_t>(-cache_from), page),
          offset + chunk_size};
}

// Coverage requirement for find_entry.
enum class coverage_policy {
  full,     // return the chunks only when they fully cover [offset, offset + size); else none
  partial,  // return every chunk overlapping [offset, offset + size), even if coverage is partial
};

// A pointer-like handle to a cached_chunk: a raw pointer or a smart pointer
// (std::unique_ptr / std::shared_ptr), with or without const.  std::to_address
// yields the underlying cached_chunk* (const-qualified iff the handle is to a
// const cached_chunk).
template <class P>
concept cached_chunk_pointer = requires(const P& p) {
  { std::to_address(p) } -> std::convertible_to<const cached_chunk*>;
};

// Find the cached chunks of @p chunks (sorted by offset, non-overlapping,
// fixed-size) that serve the request [@p offset, @p offset + @p size).
//
// With coverage_policy::full the request must be wholly covered or an empty
// vector is returned; with coverage_policy::partial every overlapping chunk is
// returned regardless of gaps.  Pure lookup: it does not mutate the chunks
// (callers apply lifecycle side effects on the result).  Accepts any contiguous
// range (vector, span, array, …) of cached_chunk pointer handles (raw / shared /
// unique, const or not); the returned pointers preserve the handle's constness.
//
// @p chunk_size is the fixed chunk size the chunks were laid out with — pass
// the backing buffer_pool's chunk_bytes() so the alignment arithmetic matches.
template <std::ranges::contiguous_range Chunks>
  requires cached_chunk_pointer<std::ranges::range_value_t<Chunks>>
[[nodiscard]] std::vector<
  decltype(std::to_address(std::declval<const std::ranges::range_value_t<Chunks>&>()))>
find_entry(const Chunks& chunks,
           std::size_t offset,
           std::size_t size,
           coverage_policy policy,
           std::size_t chunk_size)
{
  using chunk_ptr_t =
    decltype(std::to_address(std::declval<const std::ranges::range_value_t<Chunks>&>()));
  if (size == 0) return {};

  auto const first        = std::ranges::begin(chunks);
  auto const last         = std::ranges::end(chunks);
  const std::size_t count = std::ranges::size(chunks);

  // Align the request to chunk boundaries.
  const std::size_t first_chunk_off = (offset / chunk_size) * chunk_size;
  const std::size_t end_off         = offset + size;
  const std::size_t last_chunk_off  = ((end_off - 1) / chunk_size) * chunk_size;
  const std::size_t expected_count  = (last_chunk_off - first_chunk_off) / chunk_size + 1;

  // Find the first chunk at/after the aligned start (chunks are sorted).
  auto first_it = std::lower_bound(first, last, first_chunk_off, [](const auto& c, std::size_t v) {
    return std::to_address(c)->offset < v;
  });

  if (policy == coverage_policy::full) {
    if (count < expected_count) return {};
    if (first_it == last || std::to_address(*first_it)->offset != first_chunk_off) {
      return {};  // first chunk missing
    }

    // Check the last chunk is at the expected position.
    const auto first_idx       = static_cast<std::size_t>(std::distance(first, first_it));
    const std::size_t last_idx = first_idx + expected_count - 1;

    if (last_idx >= count) return {};
    if (std::to_address(first[static_cast<std::ptrdiff_t>(last_idx)])->offset != last_chunk_off)
      return {};

    // Coverage confirmed by the invariant: sorted + non-overlapping + fixed-size
    // means consecutive chunks differ by exactly chunk_size, so the intermediates
    // are forced once the first and last are at the expected positions.
    //
    // NOTE: this confirms POSITIONAL coverage only (the requested byte span maps
    // onto existing chunks) — it does NOT confirm those chunks are populated over
    // the requested sub-range.  Populated-ness (cache_from) is enforced by the
    // pin-holding caller via chunk_covers() after acquire_read() (Steps 6/7).
    std::vector<chunk_ptr_t> result;
    result.reserve(expected_count);
    for (std::size_t i = 0; i < expected_count; ++i) {
      first[static_cast<std::ptrdiff_t>(first_idx + i)]
        ->lifecycle.on_consume();  // side effect: count this chunk as consumed for eviction scoring
      result.push_back(std::to_address(first[static_cast<std::ptrdiff_t>(first_idx + i)]));
    }
    return result;
  }

  // coverage_policy::partial: return every chunk overlapping the request range,
  // even when some are missing (the invariant means offsets in
  // [first_chunk_off, last_chunk_off] are exactly the overlapping chunks).
  std::vector<chunk_ptr_t> result;
  result.reserve(expected_count);
  for (auto it = first_it; it != last && std::to_address(*it)->offset <= last_chunk_off; ++it) {
    std::to_address(*it)->lifecycle.on_consume();
    result.push_back(std::to_address(*it));
  }
  return result;
}

}  // namespace cucascade::io::cache
