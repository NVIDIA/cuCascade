/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>
#include <cucascade/utils/atomics.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace cucascade {
namespace memory {
namespace detail {
namespace {

/// Cache-line size used to keep hot atomics apart (std::hardware_destructive_interference_size
/// would trigger -Winterference-size, an error under -Werror).
constexpr std::size_t cache_line_bytes = 64;

/// Granularity of the accounting: every request is charged align_up(bytes, 256), as in the legacy
/// adaptor. The alignment forwarded to the upstream is the caller's, verbatim.
constexpr std::size_t tracking_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

/// Largest byte count whose padded size is representable by the signed per-reservation counters.
constexpr std::size_t max_representable_bytes =
  static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) - (tracking_alignment - 1);

/// Largest accepted capacity (INT64_MAX / 2). Requests are bounded by the capacity, so the signed
/// sums of the accounting (a + padded request, R + growth) stay far from int64 overflow, assuming
/// frees mirror allocations (a free routed to a different tracker than its allocation can leave a
/// phantom balance in `a`; overflowing int64 that way would take ~2^63 / t mismatched cycles).
constexpr std::size_t max_capacity_bytes =
  static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / 2;

/// Padded (accounted) size of a request; callers guarantee bytes <= max_representable_bytes.
[[nodiscard]] std::int64_t tracking_size(std::size_t bytes) noexcept
{
  return static_cast<std::int64_t>(rmm::align_up(bytes, tracking_alignment));
}

// Partial-split accounting of the legacy adaptor, with f(a, R) = max(a, R): an allocation moving a
// reservation from `pre` to `post` changes the committed counter by f(post, R) - f(pre, R), i.e.
// only by the part that lies above the reservation; a free mirrors it.
[[nodiscard]] std::int64_t excess_of_alloc(std::int64_t pre, std::int64_t post, std::int64_t r)
{
  return std::max(r, post) - std::max(r, pre);
}

[[nodiscard]] std::int64_t reclaim_of_free(std::int64_t pre, std::int64_t post, std::int64_t r)
{
  return std::max(r, pre) - std::max(r, post);
}

/**
 * @brief Tiny test-and-test-and-set spin lock (BasicLockable, so std::lock_guard works).
 *
 * lock() is one acquire exchange when uncontended, unlock() one release store. Guards critical
 * sections of a few integer operations (plus, on overflow only, a short overflow policy), so it
 * spins briefly and then yields instead of sleeping.
 */
class ttas_spinlock {
 public:
  void lock() noexcept
  {
    for (;;) {
      if (!_locked.exchange(true, std::memory_order_acquire)) { return; }
      for (unsigned spins = 0; _locked.load(std::memory_order_relaxed); ++spins) {
        if (spins < spins_before_yield) {
          cpu_relax();
        } else {
          std::this_thread::yield();
          spins = 0;
        }
      }
    }
  }

  void unlock() noexcept { _locked.store(false, std::memory_order_release); }

 private:
  static constexpr unsigned spins_before_yield = 64;

  static void cpu_relax() noexcept
  {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  std::atomic<bool> _locked{false};
};

}  // namespace

//===----------------------------------------------------------------------===//
// Accounting core (shared by the resource and every reservation it created)
//===----------------------------------------------------------------------===//

struct reservation_accounting_core {
  reservation_accounting_core(std::size_t limit, std::size_t cap) noexcept
    : memory_limit(limit), capacity(cap)
  {
  }

  /// G: live reservations + bytes charged outside reservations. Never exceeds capacity.
  alignas(cache_line_bytes) utils::atomic_bounded_counter<std::size_t> committed{0};
  utils::atomic_peak_tracker<std::size_t> peak_committed{0};
  alignas(cache_line_bytes) std::atomic<std::size_t> reserved_bytes{0};  ///< statistic
  std::atomic<std::size_t> active_reservations{0};                       ///< statistic
  std::size_t const memory_limit;
  std::size_t const capacity;

  [[nodiscard]] std::size_t effective_cap(std::size_t commit_cap) const noexcept
  {
    return commit_cap == reservation_aware_memory_resource::use_memory_limit
             ? memory_limit
             : std::min(commit_cap, capacity);
  }

  /// All-or-nothing commit against @p cap; updates the peak on success. Returns {ok, post|current}.
  std::pair<bool, std::size_t> try_commit(std::size_t bytes, std::size_t cap) noexcept
  {
    auto const result = committed.try_add(bytes, cap);
    if (result.first) { peak_committed.update_peak(result.second); }
    return result;
  }

  /// Commits min(bytes, cap - committed) (clamps @p bytes in place); returns the post value.
  std::size_t commit_upto(std::size_t& bytes, std::size_t cap) noexcept
  {
    auto const post = committed.add_bounded(bytes, cap);
    if (bytes > 0) { peak_committed.update_peak(post); }
    return post;
  }

  void uncommit(std::size_t bytes) noexcept
  {
    if (bytes != 0) { committed.sub(bytes); }
  }
};

//===----------------------------------------------------------------------===//
// Per-reservation state
//===----------------------------------------------------------------------===//

/**
 * @brief State behind a reservation handle.
 *
 * Every mutation of and decision on (allocated, size) - tracked allocation/free accounting, the
 * overflow-policy call, rollback, grow, shrink and release - runs inside ONE critical section of
 * `lock`. Nothing is updated speculatively. Lock-free readers (getters) see snapshots.
 */
struct alignas(cache_line_bytes) reservation_state {
  reservation_state(std::shared_ptr<reservation_accounting_core> accounting_core,
                    std::size_t bytes,
                    std::unique_ptr<event_notifier> release_notifier)
    : size(static_cast<std::int64_t>(bytes)),
      core(std::move(accounting_core)),
      on_release(std::move(release_notifier))
  {
  }

  ~reservation_state() noexcept;

  reservation_state(reservation_state const&)            = delete;
  reservation_state& operator=(reservation_state const&) = delete;
  reservation_state(reservation_state&&)                 = delete;
  reservation_state& operator=(reservation_state&&)      = delete;

  // The *_locked variants require `lock` to be held by the caller (the overflow-policy bridge runs
  // inside the allocation's critical section; re-locking there would self-deadlock).
  bool grow_by_locked(std::size_t additional_bytes, std::size_t commit_cap) noexcept;
  void shrink_to_fit_locked() noexcept;

  bool grow_by(std::size_t additional_bytes, std::size_t commit_cap) noexcept
  {
    std::lock_guard<ttas_spinlock> guard(lock);
    return grow_by_locked(additional_bytes, commit_cap);
  }

  void shrink_to_fit() noexcept
  {
    std::lock_guard<ttas_spinlock> guard(lock);
    shrink_to_fit_locked();
  }

  // Hot line: touched together inside the critical section.
  ttas_spinlock lock;
  /// a: signed; may exceed `size` after an overflow and go negative when bytes allocated elsewhere
  /// are freed through this reservation (legacy semantics). Written only under `lock`.
  std::atomic<std::int64_t> allocated{0};
  std::atomic<std::int64_t> size;                    ///< R: written only under `lock`
  utils::atomic_peak_tracker<std::int64_t> peak{0};  ///< high-water mark of `allocated`

  // Cold.
  alignas(cache_line_bytes) std::shared_ptr<reservation_accounting_core> core;
  notify_on_exit on_release;  ///< posts after the destructor body returned the bytes
};

reservation_state::~reservation_state() noexcept
{
  std::lock_guard<ttas_spinlock> guard(lock);
  auto const a = allocated.load(std::memory_order_relaxed);
  auto const r = size.load(std::memory_order_relaxed);
  // The term f(a, R) = max(a, R) leaves the committed counter and the live bytes `a` stay there as
  // untracked bytes: net release max(0, R - a). For a < 0 (bytes allocated elsewhere were freed
  // through this reservation) this over-releases ON PURPOSE: the matching charge lives in another
  // reservation's term or in the untracked bytes. Clamping would leak those bytes forever.
  if (r > a) { core->uncommit(static_cast<std::size_t>(r - a)); }
  core->reserved_bytes.fetch_sub(static_cast<std::size_t>(r), std::memory_order_relaxed);
  core->active_reservations.fetch_sub(1, std::memory_order_relaxed);
}

bool reservation_state::grow_by_locked(std::size_t additional_bytes,
                                       std::size_t commit_cap) noexcept
{
  if (additional_bytes == 0) { return true; }
  if (additional_bytes > max_representable_bytes) { return false; }
  auto const a = allocated.load(std::memory_order_relaxed);
  auto const r = size.load(std::memory_order_relaxed);
  // f(a, R + x) - f(a, R): bytes already above the reservation are already charged, so growing over
  // them is free (the legacy adaptor charged x again and leaked the overlap).
  auto const already_charged = a > r ? static_cast<std::size_t>(a - r) : std::size_t{0};
  auto const charge          = additional_bytes - std::min(additional_bytes, already_charged);
  if (charge > 0 && !core->try_commit(charge, core->effective_cap(commit_cap)).first) {
    return false;
  }
  size.store(r + static_cast<std::int64_t>(additional_bytes), std::memory_order_release);
  core->reserved_bytes.fetch_add(additional_bytes, std::memory_order_relaxed);
  return true;
}

void reservation_state::shrink_to_fit_locked() noexcept
{
  auto const a = allocated.load(std::memory_order_relaxed);
  auto const r = size.load(std::memory_order_relaxed);
  if (a >= r) { return; }  // fully used or exceeded: nothing to give back
  auto const shrunk   = std::max<std::int64_t>(0, a);
  auto const released = static_cast<std::size_t>(r - shrunk);
  core->uncommit(released);
  core->reserved_bytes.fetch_sub(released, std::memory_order_relaxed);
  size.store(shrunk, std::memory_order_release);
}

namespace {

/**
 * @brief Stack-only bridge that lets legacy reservation_limit_policy (= overflow_policy)
 * implementations act on a reservation_state.
 *
 * Constructed only while the caller holds the state's lock (inside charge_alloc), hence the
 * *_locked forwards. The base-class size() is a snapshot taken at construction; policies are
 * one-shot, so that is sufficient. Growing through the policy is bounded by memory_limit, as in the
 * legacy adaptor.
 */
struct policy_arena_view final : reserved_arena {
  explicit policy_arena_view(reservation_state& state)
    : reserved_arena(state.size.load(std::memory_order_relaxed)), _state(state)
  {
  }

  bool grow_by(std::size_t additional_bytes) final
  {
    return _state.grow_by_locked(additional_bytes,
                                 reservation_aware_memory_resource::use_memory_limit);
  }

  void shrink_to_fit() final { _state.shrink_to_fit_locked(); }

 private:
  reservation_state& _state;
};

struct charge_result {
  bool committed;           ///< false: the excess did not fit the capacity; `allocated` unchanged
  std::int64_t excess;      ///< bytes charged to the committed counter
  std::int64_t post;        ///< reservation's allocated bytes after this allocation
  std::size_t global_post;  ///< committed bytes right after the charge (current value on failure)
};

/**
 * @brief Accounting step of a tracked allocation (does NOT call the upstream).
 *
 * The decision and every mutation happen in one critical section. `allocated` is stored only
 * after every check passed, so an overflow-policy exception or a capacity failure leaves it
 * unchanged. A growth the overflow policy already performed (through the bridge, i.e.
 * grow_by_locked) is NOT undone: it is a complete grow_by that moved R and the committed counter
 * consistently (charge f(a, R + x) - f(a, R)), so the accounting identity still holds. The shipped
 * `increase` policy with a padding factor >= 1 (default 1.25) grows to at least a + t, leaving no
 * excess to fail on; a custom policy (or a factor < 1) may grow less.
 */
charge_result charge_alloc(reservation_state& state,
                           std::int64_t padded_bytes,
                           ::cuda::stream_ref stream,
                           overflow_policy& overflow)
{
  std::lock_guard<ttas_spinlock> guard(state.lock);
  auto const a = state.allocated.load(std::memory_order_relaxed);
  auto r       = state.size.load(std::memory_order_relaxed);
  if (a + padded_bytes > r) {
    policy_arena_view view{state};
    // ignore: no-op; increase: grows through view (charging G); fail: throws (`allocated` not
    // stored; a growth done by a policy before it throws stays, fully accounted).
    overflow.handle_over_reservation(stream,
                                     static_cast<std::size_t>(padded_bytes),
                                     static_cast<std::size_t>(std::max<std::int64_t>(0, a)),
                                     &view);
    r = state.size.load(std::memory_order_relaxed);  // the policy may have grown the reservation
  }
  auto const post         = a + padded_bytes;
  auto const excess       = excess_of_alloc(a, post, r);  // against the current R
  std::size_t global_post = 0;
  if (excess > 0) {
    auto const [ok, value] =
      state.core->committed.try_add(static_cast<std::size_t>(excess), state.core->capacity);
    if (!ok) { return {false, 0, a, value}; }
    global_post = value;
  }
  state.allocated.store(post, std::memory_order_release);
  return {true, excess, post, global_post};
}

/// Accounting step of a tracked free (also the rollback of a failed allocation). Returns the bytes
/// to give back to the committed counter; the caller uncommits them.
[[nodiscard]] std::size_t reclaim_free(reservation_state& state, std::int64_t padded_bytes) noexcept
{
  std::lock_guard<ttas_spinlock> guard(state.lock);
  auto const a    = state.allocated.load(std::memory_order_relaxed);
  auto const r    = state.size.load(std::memory_order_relaxed);
  auto const post = a - padded_bytes;
  state.allocated.store(post, std::memory_order_release);
  return static_cast<std::size_t>(reclaim_of_free(a, post, r));
}

// A retry handed to an OOM policy is one self-contained attempt: it does not consult the overflow
// policy again (the first attempt already passed it; ignore == proceed) and does not recurse into
// the OOM policy (rethrow).
oom_handling_policy& rethrow_oom_policy() noexcept
{
  static throw_on_oom_policy policy;
  return policy;
}

overflow_policy& ignore_overflow_policy() noexcept
{
  static ignore_reservation_limit_policy policy;
  return policy;
}

// Recovers the CUDA memory pool backing an upstream resource for OOM diagnostics (exact-type
// downcast of the wrapped resource; any other upstream yields a null handle).
[[nodiscard]] cudaMemPool_t extract_pool_handle(rmm::device_async_resource_ref upstream) noexcept
{
  if (auto* mr = ::cuda::mr::resource_cast<rmm::mr::cuda_async_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  if (auto* mr = ::cuda::mr::resource_cast<rmm::mr::cuda_async_view_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  if (auto* mr =
        ::cuda::mr::resource_cast<rmm::mr::cuda_async_managed_memory_resource>(&upstream)) {
    return mr->pool_handle();
  }
  return nullptr;
}

std::shared_ptr<reservation_accounting_core> make_accounting_core(std::size_t memory_limit,
                                                                  std::size_t capacity)
{
  if (memory_limit > capacity) {
    throw std::invalid_argument(
      "reservation_aware_memory_resource: memory_limit must not exceed capacity");
  }
  if (capacity > max_capacity_bytes) {
    throw std::invalid_argument(
      "reservation_aware_memory_resource: capacity must not exceed INT64_MAX / 2 bytes");
  }
  return std::make_shared<reservation_accounting_core>(memory_limit, capacity);
}

}  // namespace
}  // namespace detail

//===----------------------------------------------------------------------===//
// reservation_aware_memory_resource::reservation
//===----------------------------------------------------------------------===//

reservation_aware_memory_resource::reservation::reservation() noexcept = default;

reservation_aware_memory_resource::reservation::reservation(
  std::unique_ptr<detail::reservation_state> state) noexcept
  : _state(std::move(state))
{
}

reservation_aware_memory_resource::reservation::reservation(reservation&& other) noexcept = default;

reservation_aware_memory_resource::reservation&
reservation_aware_memory_resource::reservation::operator=(reservation&& other) noexcept
{
  if (this != &other) {
    release();
    _state = std::move(other._state);
  }
  return *this;
}

reservation_aware_memory_resource::reservation::~reservation() = default;

bool reservation_aware_memory_resource::reservation::valid() const noexcept
{
  return _state != nullptr;
}

std::size_t reservation_aware_memory_resource::reservation::size() const noexcept
{
  if (!_state) { return 0; }
  return static_cast<std::size_t>(
    std::max<std::int64_t>(0, _state->size.load(std::memory_order_acquire)));
}

std::size_t reservation_aware_memory_resource::reservation::allocated_bytes() const noexcept
{
  if (!_state) { return 0; }
  return static_cast<std::size_t>(
    std::max<std::int64_t>(0, _state->allocated.load(std::memory_order_acquire)));
}

std::size_t reservation_aware_memory_resource::reservation::peak_allocated_bytes() const noexcept
{
  if (!_state) { return 0; }
  return static_cast<std::size_t>(std::max<std::int64_t>(0, _state->peak.peak()));
}

std::size_t reservation_aware_memory_resource::reservation::available_bytes() const noexcept
{
  if (!_state) { return 0; }
  auto const r = _state->size.load(std::memory_order_acquire);
  auto const a = _state->allocated.load(std::memory_order_acquire);
  return r > a ? static_cast<std::size_t>(r - a) : 0;
}

void reservation_aware_memory_resource::reservation::reset_peak_allocated_bytes() noexcept
{
  if (_state) { _state->peak.reset(0); }
}

bool reservation_aware_memory_resource::reservation::grow_by(std::size_t additional_bytes,
                                                             std::size_t commit_cap)
{
  if (!_state) { return false; }
  return _state->grow_by(additional_bytes, commit_cap);
}

void reservation_aware_memory_resource::reservation::shrink_to_fit()
{
  if (_state) { _state->shrink_to_fit(); }
}

void reservation_aware_memory_resource::reservation::release() noexcept { _state.reset(); }

//===----------------------------------------------------------------------===//
// reservation_aware_memory_resource
//===----------------------------------------------------------------------===//

reservation_aware_memory_resource::reservation_aware_memory_resource(
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  std::unique_ptr<overflow_policy> default_overflow_policy,
  cudaMemPool_t pool_handle)
  : _core(detail::make_accounting_core(memory_limit, capacity)),
    _upstream(upstream),
    _pool_handle(pool_handle != nullptr ? pool_handle : detail::extract_pool_handle(upstream)),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy()),
    _default_overflow_policy(default_overflow_policy ? std::move(default_overflow_policy)
                                                     : make_default_reservation_limit_policy())
{
}

reservation_aware_memory_resource::reservation_aware_memory_resource(
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  std::unique_ptr<overflow_policy> default_overflow_policy,
  cudaMemPool_t pool_handle)
  : reservation_aware_memory_resource(upstream,
                                      capacity,
                                      capacity,
                                      std::move(default_oom_policy),
                                      std::move(default_overflow_policy),
                                      pool_handle)
{
}

reservation_aware_memory_resource::~reservation_aware_memory_resource() = default;

//===----------------------------------------------------------------------===//
// Reservations
//===----------------------------------------------------------------------===//

reservation_aware_memory_resource::reservation reservation_aware_memory_resource::reserve(
  std::size_t bytes, std::size_t commit_cap, std::unique_ptr<event_notifier> release_notifier)
{
  CUCASCADE_FUNC_RANGE();
  auto const [ok, global] = _core->try_commit(bytes, _core->effective_cap(commit_cap));
  if (!ok) {
    throw cucascade_out_of_memory(
      "not enough memory to reserve", MemoryError::LIMIT_EXCEEDED, bytes, global, _pool_handle);
  }
  return make_reservation(bytes, std::move(release_notifier));
}

reservation_aware_memory_resource::reservation reservation_aware_memory_resource::try_reserve(
  std::size_t bytes, std::size_t commit_cap, std::unique_ptr<event_notifier> release_notifier)
{
  CUCASCADE_FUNC_RANGE();
  if (!_core->try_commit(bytes, _core->effective_cap(commit_cap)).first) { return reservation{}; }
  return make_reservation(bytes, std::move(release_notifier));
}

reservation_aware_memory_resource::reservation reservation_aware_memory_resource::reserve_upto(
  std::size_t bytes, std::size_t commit_cap, std::unique_ptr<event_notifier> release_notifier)
{
  CUCASCADE_FUNC_RANGE();
  auto granted = bytes;
  _core->commit_upto(granted, _core->effective_cap(commit_cap));
  return make_reservation(granted, std::move(release_notifier));
}

bool reservation_aware_memory_resource::owns(reservation const& res) const noexcept
{
  return res._state != nullptr && res._state->core.get() == _core.get();
}

reservation_aware_memory_resource::reservation reservation_aware_memory_resource::make_reservation(
  std::size_t committed_bytes, std::unique_ptr<event_notifier> release_notifier)
{
  std::unique_ptr<detail::reservation_state> state;
  try {
    state = std::make_unique<detail::reservation_state>(
      _core, committed_bytes, std::move(release_notifier));
  } catch (...) {
    _core->uncommit(committed_bytes);
    throw;
  }
  _core->reserved_bytes.fetch_add(committed_bytes, std::memory_order_relaxed);
  _core->active_reservations.fetch_add(1, std::memory_order_relaxed);
  return reservation{std::move(state)};
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

void* reservation_aware_memory_resource::upstream_allocate(::cuda::stream_ref stream,
                                                           std::size_t bytes,
                                                           std::size_t alignment)
{
  try {
    return _upstream.allocate(stream, bytes, alignment);
  } catch (std::bad_alloc const& e) {
    // Only the bad_alloc family (rmm::bad_alloc, rmm::out_of_memory, ...) is rewrapped; any other
    // exception propagates unchanged.
    throw cucascade_out_of_memory(
      e.what(), MemoryError::ALLOCATION_FAILED, bytes, _core->committed.load(), _pool_handle);
  }
}

void* reservation_aware_memory_resource::do_allocate_untracked(::cuda::stream_ref stream,
                                                               std::size_t bytes,
                                                               std::size_t alignment,
                                                               oom_handling_policy& oom_policy)
{
  CUCASCADE_FUNC_RANGE();
  auto& core = *_core;
  // A request larger than the capacity can never be satisfied (live bytes <= committed <=
  // capacity), so it is rejected up front without consulting the OOM policy (documented
  // deviation); this also keeps the padded size representable.
  if (bytes > core.capacity) {
    throw cucascade_out_of_memory("not enough capacity to allocate memory",
                                  MemoryError::LIMIT_EXCEEDED,
                                  bytes,
                                  core.committed.load(),
                                  _pool_handle);
  }
  auto const padded = static_cast<std::size_t>(detail::tracking_size(bytes));
  auto retry        = [this, alignment](std::size_t retry_bytes, ::cuda::stream_ref retry_stream) {
    return do_allocate_untracked(
      retry_stream, retry_bytes, alignment, detail::rethrow_oom_policy());
  };

  auto const [ok, post] = core.committed.try_add(padded, core.capacity);
  if (!ok) {
    return oom_policy.handle_oom(
      bytes,
      stream,
      std::make_exception_ptr(cucascade_out_of_memory("not enough capacity to allocate memory",
                                                      MemoryError::LIMIT_EXCEEDED,
                                                      bytes,
                                                      post,
                                                      _pool_handle)),
      retry);
  }
  void* ptr = nullptr;
  try {
    ptr = upstream_allocate(stream, bytes, alignment);
  } catch (cucascade_out_of_memory const&) {
    core.uncommit(padded);
    return oom_policy.handle_oom(bytes, stream, std::current_exception(), retry);
  } catch (...) {
    core.uncommit(padded);
    throw;
  }
  core.peak_committed.update_peak(post);  // only after success
  return ptr;
}

void reservation_aware_memory_resource::do_deallocate_untracked(::cuda::stream_ref stream,
                                                                void* ptr,
                                                                std::size_t bytes,
                                                                std::size_t alignment) noexcept
{
  CUCASCADE_FUNC_RANGE();
// Suppress false-positive null-dereference warnings from CCCL library code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
  _upstream.deallocate(stream, ptr, bytes, alignment);
#pragma GCC diagnostic pop
  _core->uncommit(static_cast<std::size_t>(detail::tracking_size(bytes)));
}

void* reservation_aware_memory_resource::do_allocate_tracked(::cuda::stream_ref stream,
                                                             std::size_t bytes,
                                                             std::size_t alignment,
                                                             detail::reservation_state& state,
                                                             oom_handling_policy& oom,
                                                             overflow_policy& overflow)
{
  CUCASCADE_FUNC_RANGE();
  assert(state.core.get() == _core.get());
  auto& core = *state.core;
  // Never satisfiable (see do_allocate_untracked): rejected before the overflow policy and without
  // the OOM policy (documented deviation from the legacy adaptor, which ran both).
  if (bytes > core.capacity) {
    throw cucascade_out_of_memory("not enough capacity to allocate memory",
                                  MemoryError::LIMIT_EXCEEDED,
                                  bytes,
                                  core.committed.load(),
                                  _pool_handle);
  }
  auto const padded = detail::tracking_size(bytes);
  auto retry = [this, &state, alignment](std::size_t retry_bytes, ::cuda::stream_ref retry_stream) {
    return do_allocate_tracked(retry_stream,
                               retry_bytes,
                               alignment,
                               state,
                               detail::rethrow_oom_policy(),
                               detail::ignore_overflow_policy());
  };

  auto const charge = detail::charge_alloc(state, padded, stream, overflow);
  if (!charge.committed) {
    return oom.handle_oom(
      bytes,
      stream,
      std::make_exception_ptr(cucascade_out_of_memory("not enough capacity to allocate memory",
                                                      MemoryError::LIMIT_EXCEEDED,
                                                      bytes,
                                                      charge.global_post,
                                                      _pool_handle)),
      retry);
  }
  void* ptr = nullptr;
  try {
    ptr = upstream_allocate(stream, bytes, alignment);
  } catch (cucascade_out_of_memory const&) {
    // Rollback == a free of `padded` under the lock: the identity is restored before any policy
    // runs, and it stays exact even if the reservation was grown/shrunk meanwhile.
    core.uncommit(detail::reclaim_free(state, padded));
    return oom.handle_oom(bytes, stream, std::current_exception(), retry);
  } catch (...) {
    core.uncommit(detail::reclaim_free(state, padded));
    throw;
  }
  // Peaks only after success.
  if (charge.excess > 0) { core.peak_committed.update_peak(charge.global_post); }
  state.peak.update_peak(charge.post);
  return ptr;
}

void reservation_aware_memory_resource::do_deallocate_tracked(
  ::cuda::stream_ref stream,
  void* ptr,
  std::size_t bytes,
  std::size_t alignment,
  detail::reservation_state& state) noexcept
{
  CUCASCADE_FUNC_RANGE();
  auto const reclaimed = detail::reclaim_free(state, detail::tracking_size(bytes));
// Suppress false-positive null-dereference warnings from CCCL library code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
  _upstream.deallocate(stream, ptr, bytes, alignment);
#pragma GCC diagnostic pop
  state.core->uncommit(reclaimed);
}

void* reservation_aware_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t alignment,
                                                  reservation& res)
{
  if (!owns(res)) {
    CUCASCADE_FAIL("reservation is empty or belongs to another reservation_aware_memory_resource");
  }
  return do_allocate_tracked(
    stream, bytes, alignment, *res._state, *_default_oom_policy, *_default_overflow_policy);
}

void reservation_aware_memory_resource::deallocate(::cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment,
                                                   reservation& res) noexcept
{
  assert(owns(res) && "deallocate: reservation is empty or belongs to another resource");
  if (owns(res)) {
    do_deallocate_tracked(stream, ptr, bytes, alignment, *res._state);
  } else {
    do_deallocate_untracked(stream, ptr, bytes, alignment);
  }
}

void* reservation_aware_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t alignment)
{
  return do_allocate_untracked(stream, bytes, alignment, *_default_oom_policy);
}

void reservation_aware_memory_resource::deallocate(::cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  do_deallocate_untracked(stream, ptr, bytes, alignment);
}

void* reservation_aware_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
  auto* ptr         = allocate(stream, bytes, alignment);
  try {
    stream.sync();
  } catch (...) {
    deallocate(stream, ptr, bytes, alignment);  // same (untracked) accounting; noexcept
    throw;
  }
  return ptr;
}

void reservation_aware_memory_resource::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamSynchronize(cudaStream_t{nullptr}));
}

bool reservation_aware_memory_resource::operator==(
  reservation_aware_memory_resource const& other) const noexcept
{
  return _core == other._core;
}

//===----------------------------------------------------------------------===//
// Introspection
//===----------------------------------------------------------------------===//

rmm::device_async_resource_ref reservation_aware_memory_resource::get_upstream_resource()
  const noexcept
{
  return _upstream;
}

cudaMemPool_t reservation_aware_memory_resource::get_pool_handle() const noexcept
{
  return _pool_handle;
}

std::size_t reservation_aware_memory_resource::get_capacity() const noexcept
{
  return _core->capacity;
}

std::size_t reservation_aware_memory_resource::get_memory_limit() const noexcept
{
  return _core->memory_limit;
}

std::size_t reservation_aware_memory_resource::get_total_allocated_bytes() const noexcept
{
  return _core->committed.load();
}

std::size_t reservation_aware_memory_resource::get_peak_total_allocated_bytes() const noexcept
{
  return _core->peak_committed.peak();
}

std::size_t reservation_aware_memory_resource::get_available_memory() const noexcept
{
  auto const committed = _core->committed.load();
  return _core->capacity > committed ? _core->capacity - committed : 0;
}

std::size_t reservation_aware_memory_resource::get_total_reserved_bytes() const noexcept
{
  return _core->reserved_bytes.load(std::memory_order_relaxed);
}

std::size_t reservation_aware_memory_resource::get_active_reservation_count() const noexcept
{
  return _core->active_reservations.load(std::memory_order_relaxed);
}

oom_handling_policy& reservation_aware_memory_resource::get_default_oom_policy() const noexcept
{
  return *_default_oom_policy;
}

overflow_policy& reservation_aware_memory_resource::get_default_overflow_policy() const noexcept
{
  return *_default_overflow_policy;
}

}  // namespace memory
}  // namespace cucascade
