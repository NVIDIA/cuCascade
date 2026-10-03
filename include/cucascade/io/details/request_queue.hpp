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

#include <cucascade/io/concurrent_queue.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <system_error>
#include <utility>

namespace cucascade::io::detail {

/**
 * @brief Shared, pull-based MPMC queue of grouped I/O requests, one lane per
 *        @ref request_class.
 *
 * Every submitter pushes; every runner of the context pulls the class chosen
 * by its @ref scheduling_policy.  Each lane is a lock-free moodycamel queue of
 * raw owning pointers (ownership moves into the queue only when the push
 * succeeds, so a failed push leaves the request with the caller), plus atomic
 * counters for the per-class request count, queued bytes, an approximate
 * "oldest enqueue time", and queue-wait statistics.
 *
 * Ordering: FIFO per producer thread within one lane; no ordering between
 * producers or between lanes (moodycamel semantics).
 *
 * Counters: the request count of a lane is incremented @em before the entry
 * becomes visible and decremented @em after it was dequeued, so it is never
 * below the number of dequeuable entries.  A runner that observes a non-zero
 * count may briefly fail to dequeue (the entry is still being published); it
 * must simply retry on its next loop iteration.  The counters use sequentially
 * consistent operations: the runner park protocol (@ref request_hub) relies on
 * "publish count, then read parked flags" vs. "publish parked flag, then read
 * count" never both missing each other.
 *
 * Thread-safety: all members may be called concurrently from any thread.
 */
class request_queue {
 public:
  using clock      = std::chrono::steady_clock;
  using time_point = clock::time_point;

  request_queue() = default;

  /// Remaining requests are cancelled (@c std::errc::operation_canceled) so no
  /// promise is broken.
  ~request_queue()
  {
    drain([](std::unique_ptr<grouped_io_request> request) noexcept {
      request->meta.state.store(request_state::cancelled, std::memory_order_release);
      request->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
    });
  }

  request_queue(request_queue const&)            = delete;
  request_queue& operator=(request_queue const&) = delete;

  /**
   * @brief Publish @p request in the lane of its (resolved) @c meta.cls.
   *
   * Sets @c meta.state = @c queued and, unless @p preserve_enqueue_time,
   * @c meta.enqueued_at = now.  On success ownership moves into the queue and
   * @p request becomes null; on failure (allocation) @p request is untouched.
   *
   * @param request Request to publish; must be non-null.
   * @param preserve_enqueue_time Keep the existing @c meta.enqueued_at (used
   *        when a retiring runner hands a partially processed request back).
   * @return Whether the request was published.
   */
  [[nodiscard]] bool push(std::unique_ptr<grouped_io_request>& request,
                          bool preserve_enqueue_time = false) noexcept
  {
    if (request == nullptr) return false;
    auto const lane  = request_class_index(request->meta.cls);
    auto const bytes = request->remaining_bytes();
    if (!preserve_enqueue_time) request->meta.enqueued_at = clock::now();
    request->meta.state.store(request_state::queued, std::memory_order_release);
    auto const enqueued_ns = to_ns(request->meta.enqueued_at);

    // Count first (see class comment), then publish.
    _bytes[lane].fetch_add(bytes, std::memory_order_relaxed);
    auto const previous = _count[lane].fetch_add(1, std::memory_order_seq_cst);
    _total.fetch_add(1, std::memory_order_seq_cst);
    if (previous == 0) {
      _oldest_ns[lane].store(enqueued_ns, std::memory_order_relaxed);
    } else {
      lower_oldest(lane, enqueued_ns);
    }

    bool published = false;
    try {
      published = _lanes[lane].enqueue(request.get());
    } catch (...) {
      published = false;
    }
    if (!published) {
      _total.fetch_sub(1, std::memory_order_seq_cst);
      _count[lane].fetch_sub(1, std::memory_order_seq_cst);
      _bytes[lane].fetch_sub(bytes, std::memory_order_relaxed);
      return false;
    }
    static_cast<void>(request.release());
    return true;
  }

  /**
   * @brief Try to take one request from lane @p cls.
   *
   * On success sets @c meta.state = @c assigned and @c meta.assigned_at = now
   * and records the queue wait.  May fail spuriously while a concurrent push
   * is still publishing (see class comment).
   *
   * @param cls Concrete class (never @c automatic).
   * @param out Receives the request on success.
   * @return Whether a request was taken.
   */
  [[nodiscard]] bool try_pull(request_class cls, std::unique_ptr<grouped_io_request>& out) noexcept
  {
    auto const lane         = request_class_index(cls);
    grouped_io_request* raw = nullptr;
    if (_count[lane].load(std::memory_order_seq_cst) == 0) return false;
    if (!_lanes[lane].try_dequeue(raw) || raw == nullptr) return false;
    out.reset(raw);

    auto const now = clock::now();
    _bytes[lane].fetch_sub(out->remaining_bytes(), std::memory_order_relaxed);
    _total.fetch_sub(1, std::memory_order_seq_cst);
    _count[lane].fetch_sub(1, std::memory_order_seq_cst);
    // Approximation: the next-oldest entry was enqueued no earlier than the one
    // just pulled (exact for a single producer, close otherwise).
    _oldest_ns[lane].store(to_ns(out->meta.enqueued_at), std::memory_order_relaxed);

    out->meta.assigned_at = now;
    out->meta.state.store(request_state::assigned, std::memory_order_release);
    auto const wait = std::max<std::int64_t>(0, to_ns(now) - to_ns(out->meta.enqueued_at));
    _last_wait_ns[lane].store(wait, std::memory_order_relaxed);
    auto max_wait = _max_wait_ns[lane].load(std::memory_order_relaxed);
    while (wait > max_wait &&
           !_max_wait_ns[lane].compare_exchange_weak(max_wait, wait, std::memory_order_relaxed)) {}
    return true;
  }

  /**
   * @brief Remove every queued request and hand each to @p fn.
   *
   * Requests pushed concurrently may or may not be drained.
   *
   * @param fn Callable invoked as @c fn(std::unique_ptr<grouped_io_request>).
   * @return Number of requests drained.
   */
  template <class Fn>
  std::size_t drain(Fn&& fn) noexcept
  {
    std::size_t drained = 0;
    for (std::size_t lane = 0; lane < request_class_count; ++lane) {
      grouped_io_request* raw = nullptr;
      while (_count[lane].load(std::memory_order_seq_cst) != 0 && _lanes[lane].try_dequeue(raw)) {
        std::unique_ptr<grouped_io_request> request(raw);
        if (request == nullptr) continue;
        _bytes[lane].fetch_sub(request->remaining_bytes(), std::memory_order_relaxed);
        _total.fetch_sub(1, std::memory_order_seq_cst);
        _count[lane].fetch_sub(1, std::memory_order_seq_cst);
        ++drained;
        fn(std::move(request));
      }
    }
    return drained;
  }

  /// Approximate number of requests queued in lane @p cls.
  [[nodiscard]] std::size_t approx_size(request_class cls) const noexcept
  {
    return _count[request_class_index(cls)].load(std::memory_order_seq_cst);
  }

  /// Approximate number of requests queued across all lanes.
  [[nodiscard]] std::size_t approx_total() const noexcept
  {
    return _total.load(std::memory_order_seq_cst);
  }

  /// Whether any lane appears non-empty (sequentially consistent load).
  [[nodiscard]] bool has_queued() const noexcept { return approx_total() != 0; }

  /// Bytes not yet taken of the requests queued in lane @p cls.
  [[nodiscard]] std::size_t queued_bytes(request_class cls) const noexcept
  {
    return _bytes[request_class_index(cls)].load(std::memory_order_relaxed);
  }

  /// Bytes not yet taken of all queued requests.
  [[nodiscard]] std::size_t total_queued_bytes() const noexcept
  {
    std::size_t total = 0;
    for (auto const& bytes : _bytes) {
      total += bytes.load(std::memory_order_relaxed);
    }
    return total;
  }

  /**
   * @brief Approximate enqueue time of the oldest request of lane @p cls.
   *
   * moodycamel cannot peek, so this is maintained heuristically: a push into
   * an empty lane sets it, a push of an older (re-queued) request lowers it,
   * and a pull sets it to the pulled request's enqueue time.  With several
   * producers the true oldest entry may be older than the hint by at most the
   * enqueue-time spread of concurrently published entries — accurate enough for
   * a starvation guard measured in milliseconds.
   *
   * @return The hint, or @c time_point::max() when the lane is empty.
   */
  [[nodiscard]] time_point oldest_hint(request_class cls) const noexcept
  {
    auto const lane = request_class_index(cls);
    if (_count[lane].load(std::memory_order_seq_cst) == 0) return time_point::max();
    return time_point{std::chrono::nanoseconds{_oldest_ns[lane].load(std::memory_order_relaxed)}};
  }

  /// Age of the oldest request of lane @p cls at @p now (0 when the lane is empty).
  [[nodiscard]] std::chrono::nanoseconds oldest_age(request_class cls,
                                                    time_point now) const noexcept
  {
    auto const oldest = oldest_hint(cls);
    if (oldest == time_point::max() || oldest >= now) return std::chrono::nanoseconds{0};
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - oldest);
  }

  /// Queue wait of the most recently pulled request of lane @p cls.
  [[nodiscard]] std::chrono::nanoseconds last_wait(request_class cls) const noexcept
  {
    return std::chrono::nanoseconds{
      _last_wait_ns[request_class_index(cls)].load(std::memory_order_relaxed)};
  }

  /// Longest queue wait observed in lane @p cls.
  [[nodiscard]] std::chrono::nanoseconds max_wait(request_class cls) const noexcept
  {
    return std::chrono::nanoseconds{
      _max_wait_ns[request_class_index(cls)].load(std::memory_order_relaxed)};
  }

 private:
  [[nodiscard]] static std::int64_t to_ns(time_point tp) noexcept
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
  }

  void lower_oldest(std::size_t lane, std::int64_t value) noexcept
  {
    auto current = _oldest_ns[lane].load(std::memory_order_relaxed);
    while (value < current &&
           !_oldest_ns[lane].compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
  }

  std::array<concurrent_queue<grouped_io_request*>, request_class_count> _lanes;
  std::array<std::atomic<std::size_t>, request_class_count> _count{};
  std::array<std::atomic<std::size_t>, request_class_count> _bytes{};
  std::array<std::atomic<std::int64_t>, request_class_count> _oldest_ns{};
  std::array<std::atomic<std::int64_t>, request_class_count> _last_wait_ns{};
  std::array<std::atomic<std::int64_t>, request_class_count> _max_wait_ns{};
  std::atomic<std::size_t> _total{0};
};

}  // namespace cucascade::io::detail
