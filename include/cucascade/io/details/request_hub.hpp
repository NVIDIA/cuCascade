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

#include <cucascade/io/details/request_queue.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>

namespace cucascade::io::detail {

/**
 * @brief What a runner about to block must do; see @ref request_hub::prepare_wait.
 */
enum class wait_action : std::uint8_t {
  /// Work the policy would pull is visible: loop again now, do not block.
  pull_now,
  /// Parked: block on the runner eventfd + own completion source with the
  /// normal (idle) timeout, then call @ref request_hub::unpark -- always.
  park,
  /// Queued work is visible but the policy refuses it now.  Do NOT park and
  /// do NOT spin: no wakeup will be sent for that work, so block on the
  /// eventfd + own completions with a bounded timeout
  /// (@ref request_hub::refused_work_retry) -- or, if own operations are in
  /// flight whose completions end the wait anyway, with the normal timeout.
  wait_bounded,
  /// The runner cannot take more work (group limit): do not park; block on
  /// its own completions (the eventfd stays armed so stop is seen) with the
  /// normal timeout.
  wait_unparked,
};

/**
 * @brief Everything submitters and runners of one context share: admission,
 *        the per-class @ref request_queue, the @ref runner_registry, and
 *        aggregate statistics.
 *
 * One instance is owned by each reactor (the reactor is the context-wide,
 * thread-safe dispatcher) and exposed through @c Reactor::hub(); the
 * @c templated_ioctx front end and every engine of the reactor use it.
 *
 * Admission: @ref enqueue and @ref requeue check @ref accepting and publish
 * under a shared lock; @ref set_accepting takes the lock exclusively.  So once
 * @c set_accepting(false) returned, no further request can be published and a
 * following @ref cancel_queued sees every request published before -- nothing
 * is lost or stranded between "flip" and "drain".  Requests rejected because
 * the hub is not accepting are settled with @c std::errc::operation_canceled
 * (outside the lock: settling runs user callbacks inline, which may re-enter).
 *
 * Bookkeeping of a grouped request (all done here, not by engines):
 *  - @ref enqueue / @ref requeue: @c meta.state = queued (+ @c enqueued_at);
 *  - @ref try_pull: @c state = assigned, @c assigned_at, @c runner_id; the
 *    runner's active-group count and the in-flight count go up;
 *  - @ref finish_group: @c state = completed / failed / cancelled,
 *    @c completed_at; counts go down; the first-I/O delay
 *    (@c first_io_at - @c enqueued_at) is recorded if an operation was submitted;
 *  - @ref requeue: counts go down (the request is queued again, keeping
 *    @c enqueued_at; nothing is recorded, so a request is counted once).
 * Engines set the intermediate states themselves (@c in_flight when the first
 * physical operation is submitted, plus @c first_io_at; @c copying while a
 * CUDA copy is outstanding).
 *
 * Thread-safety: all members may be called concurrently from any thread,
 * except that a @ref runner_slot is passed only by its own runner thread.
 */
class request_hub {
 public:
  using clock      = std::chrono::steady_clock;
  using time_point = clock::time_point;

  request_hub() = default;

  request_hub(request_hub const&)            = delete;
  request_hub& operator=(request_hub const&) = delete;

  // -- admission (submitters / lifecycle) ---------------------------------------

  /// Open or close admission.  Closing waits for in-progress publications.
  /// Either way clears a rejection reason set by @ref close_admission.
  void set_accepting(bool accepting) noexcept;

  /**
   * @brief Close admission because nothing is left to serve the queue, and
   *        remember @p reason for rejected submissions.
   *
   * Like @c set_accepting(false), but until the next @ref set_accepting call
   * @ref enqueue and @ref requeue settle rejected work with @p reason instead
   * of @c operation_canceled.  Does not touch queued requests: call
   * @ref cancel_queued (with the same reason) afterwards, outside any lock the
   * settled callbacks might need.
   *
   * Used by @c templated_ioctx when its last runner exited with a fatal engine
   * error, so submissions fail fast instead of waiting for a runner that will
   * never come.
   */
  void close_admission(grouped_coordinator::error_type reason) noexcept;

  /// The rejection reason set by @ref close_admission, if admission is closed by it.
  [[nodiscard]] std::optional<grouped_coordinator::error_type> rejection_reason() const;

  /// Whether new requests are admitted.
  [[nodiscard]] bool accepting() const noexcept
  {
    return _accepting.load(std::memory_order_acquire);
  }

  /**
   * @brief Publish @p request and wake one parked runner.
   *
   * Never throws and never drops a credit: when not accepting the request is
   * cancelled with @c operation_canceled (or the @ref close_admission reason), when the queue
   * cannot allocate it is failed with @c no_buffer_space.  Null requests are ignored.
   */
  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  /**
   * @brief Hand a pulled, partially processed request back (runner retirement).
   *
   * @p request must have been pulled by @p slot's runner and still have
   * untaken work (@c !request->empty()).  Keeps @c meta.enqueued_at (so the
   * request keeps its age), drops it from @p slot's bookkeeping, and publishes
   * it for other runners.  If the hub stopped accepting meanwhile, the
   * remaining work is cancelled with @c operation_canceled instead and the
   * request is finished as @c cancelled.
   */
  void requeue(std::unique_ptr<grouped_io_request> request, runner_slot& slot) noexcept;

  /**
   * @brief Remove every queued request and settle its untaken work with @p error.
   *
   * Used by @c templated_ioctx::shutdown after @c set_accepting(false).
   *
   * @return Number of requests cancelled.
   */
  std::size_t cancel_queued(grouped_coordinator::error_type const& error) noexcept;

  // -- runner side ----------------------------------------------------------------

  /**
   * @brief Pull one request of class @p cls for the runner of @p slot.
   *
   * @return The request (state @c assigned) or null when the lane is (or
   *         spuriously appears) empty.
   */
  [[nodiscard]] std::unique_ptr<grouped_io_request> try_pull(request_class cls,
                                                             runner_slot& slot) noexcept;

  /// Whether any request is queued (sequentially consistent).
  [[nodiscard]] bool has_queued() const noexcept { return _queue.has_queued(); }

  /**
   * @brief Announce that @p slot's runner is about to block, unless work is queued.
   *
   * Call right before blocking.  Returns @c true when parked: the runner may
   * block on its eventfd (and must call @ref unpark after waking, whatever the
   * reason).  Returns @c false when work became visible; the runner stays
   * unparked and must loop again without blocking.
   *
   * A runner may park while it still owns groups (e.g. waiting for
   * completions) as long as it has room to pull more work; it then gets woken
   * for new work too.  A runner that cannot take more work must not park; it
   * simply waits for its own completions.
   */
  [[nodiscard]] bool try_park(runner_slot& slot) noexcept;

  /// Clear @p slot's parked flag (idempotent).  Call after every wait that followed a park.
  void unpark(runner_slot& slot) noexcept { _registry.unpark(slot); }

  /// Longest block of a runner whose visible queued work the policy refused
  /// (@ref wait_action::wait_bounded) and that has no own completion pending.
  static constexpr std::chrono::milliseconds refused_work_retry{1};

  /**
   * @brief The waiting rule every engine follows right before it blocks.
   *
   * @ref try_park fails whenever *anything* is queued, but the scheduling
   * policy may refuse what is queued (class at its group limit, write /
   * background share exhausted, ...).  Looping immediately would then
   * busy-spin, and blocking with the idle timeout could stall for a long
   * time because publishers only wake *parked* runners.  So:
   *  - @p has_room false (the policy would not pull anything even if every
   *    lane had work, e.g. group limit reached) -> @ref wait_action::wait_unparked;
   *  - @ref try_park succeeds -> @ref wait_action::park (caller must
   *    @ref unpark after the wait, whatever ended it);
   *  - otherwise @p policy_would_pull() (typically
   *    @c policy.pick(view).has_value() on a fresh view) -> @ref wait_action::pull_now;
   *  - otherwise -> @ref wait_action::wait_bounded.
   *
   * @tparam WouldPull @c bool() callable; only invoked when parking failed.
   */
  template <class WouldPull>
  [[nodiscard]] wait_action prepare_wait(runner_slot& slot,
                                         bool has_room,
                                         WouldPull&& policy_would_pull) noexcept
  {
    if (!has_room) return wait_action::wait_unparked;
    if (try_park(slot)) return wait_action::park;
    return static_cast<bool>(policy_would_pull()) ? wait_action::pull_now
                                                  : wait_action::wait_bounded;
  }

  /**
   * @brief Retire a pulled request whose work is fully settled (or abandoned).
   *
   * Sets @c meta.completed_at and the terminal @c meta.state: @p state when
   * given, otherwise @c failed if the coordinator has an error and
   * @c completed if not.  If the engine stamped @c meta.first_io_at, records
   * the request's first-I/O delay in its class (see @ref class_stats).  The
   * request must not be touched afterwards (the caller usually destroys it
   * right away).
   */
  void finish_group(grouped_io_request& request,
                    runner_slot& slot,
                    std::optional<request_state> state = std::nullopt) noexcept;

  /// Fill the queue fields (@c queued, @c oldest_age) of @p view at @p now.
  void fill_queue_view(scheduling_view& view, time_point now) const noexcept;

  // -- observability --------------------------------------------------------------

  /**
   * @brief Aggregate statistics: per-class queue state and first-I/O delay,
   *        runner counts and per-runner gauges, in-flight groups.
   *
   * Takes the registry lock briefly and allocates @c queue_stats::runners; if
   * that allocation fails the vector is left empty (everything else is filled).
   */
  [[nodiscard]] queue_stats stats() const noexcept;

  /// Clear the peak statistics: per-class max queue wait and max first-I/O
  /// delay, per-runner max in-flight operations.  Monotonic counters are untouched.
  void reset_stats_peaks() noexcept
  {
    _queue.reset_peaks();
    _registry.reset_peaks();
  }

  /// Bytes not yet taken of all queued requests.
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return _queue.total_queued_bytes(); }

  /// Grouped requests currently owned by runners.
  [[nodiscard]] std::size_t in_flight_requests() const noexcept
  {
    return _in_flight.load(std::memory_order_relaxed);
  }

  [[nodiscard]] request_queue& queue() noexcept { return _queue; }
  [[nodiscard]] request_queue const& queue() const noexcept { return _queue; }
  [[nodiscard]] runner_registry& registry() noexcept { return _registry; }
  [[nodiscard]] runner_registry const& registry() const noexcept { return _registry; }

 private:
  void release_from_runner(runner_slot& slot) noexcept;

  /// Error for rejected work: the close_admission reason or operation_canceled.
  [[nodiscard]] grouped_coordinator::error_type rejection_error_locked() const;

  mutable std::shared_mutex _admission;
  std::atomic<bool> _accepting{false};
  std::optional<grouped_coordinator::error_type> _rejection;  // guarded by _admission
  runner_registry _registry;
  request_queue _queue;
  std::atomic<std::size_t> _in_flight{0};
};

}  // namespace cucascade::io::detail
