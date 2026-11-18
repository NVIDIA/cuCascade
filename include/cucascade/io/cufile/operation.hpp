/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/device_arena.hpp>
#include <cucascade/io/cufile/lane_pool.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/range_plan.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

inline constexpr ssize_t pending_sentinel = std::numeric_limits<ssize_t>::min();

/** @brief Pointer arguments retained at a stable address through the final observed event. */
struct async_args {
  std::size_t size{0};
  off_t file_offset{0};
  off_t buffer_offset{0};
  ssize_t bytes_done{pending_sentinel};
};

/**
 * @brief Move-only ownership of one coordinator credit, with exactly one terminal settlement.
 *
 * Construction adopts an existing credit. add_task adds one before it can be published.
 * Destruction cancels an unsettled credit; it never reports success implicitly. Concurrent
 * finish calls are allowed, but moving or destroying the wrapper requires exclusive access.
 */
class operation_credit {
 public:
  operation_credit() noexcept = default;
  explicit operation_credit(std::shared_ptr<grouped_coordinator> coordinator)
    : _coordinator(std::move(coordinator)), _settled(false)
  {
    if (_coordinator == nullptr)
      throw std::invalid_argument("operation_credit needs a coordinator");
  }

  [[nodiscard]] static operation_credit add_task(std::shared_ptr<grouped_coordinator> coordinator)
  {
    if (coordinator == nullptr) throw std::invalid_argument("operation_credit needs a coordinator");
    coordinator->add_tasks(1);
    return operation_credit{std::move(coordinator)};
  }

  operation_credit(operation_credit const&)            = delete;
  operation_credit& operator=(operation_credit const&) = delete;
  operation_credit(operation_credit&& other) noexcept
    : _coordinator(std::move(other._coordinator)), _settled(other._settled.exchange(true))
  {
  }
  operation_credit& operator=(operation_credit&& other) noexcept
  {
    if (this != &other) {
      cancel();
      _coordinator = std::move(other._coordinator);
      _settled.store(other._settled.exchange(true));
    }
    return *this;
  }
  ~operation_credit() noexcept { cancel(); }

  [[nodiscard]] std::shared_ptr<grouped_coordinator> coordinator() const noexcept
  {
    return _coordinator;
  }
  [[nodiscard]] bool is_settled() const noexcept { return _settled.load(); }
  [[nodiscard]] bool finish_success() noexcept
  {
    if (_settled.exchange(true)) return false;
    _coordinator->on_complete();
    return true;
  }
  [[nodiscard]] bool finish_error(grouped_coordinator::error_type const& error) noexcept
  {
    if (_settled.exchange(true)) return false;
    _coordinator->report_error(error);
    return true;
  }

 private:
  void cancel() noexcept
  {
    static_cast<void>(finish_error(std::make_error_code(std::errc::operation_canceled)));
  }

  std::shared_ptr<grouped_coordinator> _coordinator;
  std::atomic<bool> _settled{true};
};

enum class operation_phase : std::uint8_t {
  pending,
  input_wait,  ///< Writes only: staged reads fence only their final destination copy (AM-2).
  staging_copy,
  submission_queued,
  submitting,
  io_inflight,
  retry_pending,
  host_io,
  copy_inflight,
  draining_error,
  quarantined,
  terminal
};

enum class operation_kind : std::uint8_t {
  read_staged,
  read_direct,
  write_body_staged,
  write_body_direct,
  write_edge
};

/** @brief One exact caller interval and its credit; several may share a physical read (AM-6). */
struct operation_target {
  range logical;
  void* user_ptr{nullptr};
  operation_credit credit;
};

/** @brief Two inline targets, spilling only when coalescing needs more slices. */
class operation_targets {
 public:
  operation_targets()                                    = default;
  operation_targets(operation_targets const&)            = delete;
  operation_targets& operator=(operation_targets const&) = delete;
  operation_targets(operation_targets&&)                 = delete;
  operation_targets& operator=(operation_targets&&)      = delete;

  /// Allocate before adopting credits so expansion cannot strand a partially built group.
  void reserve(std::size_t count)
  {
    if (count > _inline.size()) _overflow.reserve(count - _inline.size());
  }

  operation_target& emplace_back(range logical, void* user_ptr, operation_credit credit)
  {
    if (_size < _inline.size()) {
      auto& value = _inline[_size].emplace(operation_target{logical, user_ptr, std::move(credit)});
      ++_size;
      return value;
    }
    auto& value = _overflow.emplace_back(operation_target{logical, user_ptr, std::move(credit)});
    ++_size;
    return value;
  }

  /// Reset credits while retaining control-thread-preallocated overflow capacity.
  void clear() noexcept
  {
    for (auto& value : _inline)
      value.reset();
    _overflow.clear();
    _size = 0;
  }
  void swap_storage(operation_targets& other) noexcept
  {
    assert(empty() && other.empty());
    _overflow.swap(other._overflow);
  }

  [[nodiscard]] std::size_t size() const noexcept { return _size; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }
  [[nodiscard]] operation_target& operator[](std::size_t index) noexcept
  {
    assert(index < _size);
    return index < _inline.size() ? *_inline[index] : _overflow[index - _inline.size()];
  }
  [[nodiscard]] operation_target const& operator[](std::size_t index) const noexcept
  {
    assert(index < _size);
    return index < _inline.size() ? *_inline[index] : _overflow[index - _inline.size()];
  }

 private:
  std::array<std::optional<operation_target>, 2> _inline;
  std::vector<operation_target> _overflow;
  std::size_t _size{0};
};

class operation;

struct recorded_event {
  operation* op{nullptr};
  cudaEvent_t event{nullptr};
  std::uint64_t generation{0};
  operation_phase phase{operation_phase::pending};
};

namespace detail {
struct operation_pool_state;
}

/**
 * @brief Nonmovable physical-operation state, retained independently of credit settlement.
 *
 * The scheduler/reaper exclusively owns state transitions; the submitter only uses published
 * arguments until it hands ownership back. Call mark_work_started BEFORE any command can
 * reference this record, even if enqueue later fails. The pool refuses release until a final
 * event was observed or recovery proved quiescence. An enqueued-and-joined read can settle
 * all targets early without releasing its file, input, args, staging, or lane (AM-1).
 */
class operation {
 public:
  operation()                            = default;
  operation(operation const&)            = delete;
  operation& operator=(operation const&) = delete;
  operation(operation&&)                 = delete;
  operation& operator=(operation&&)      = delete;

  [[nodiscard]] operation_kind kind() const noexcept { return _kind; }
  void set_kind(operation_kind kind) noexcept { _kind = kind; }
  [[nodiscard]] operation_phase phase() const noexcept { return _phase; }
  [[nodiscard]] bool is_read() const noexcept
  {
    return _kind == operation_kind::read_staged || _kind == operation_kind::read_direct;
  }

  void set_phase(operation_phase next)
  {
    if (is_read() &&
        (next == operation_phase::input_wait || next == operation_phase::staging_copy)) {
      throw std::logic_error("reads do not use write input-wait/staging-copy phases");
    }
    if (next == operation_phase::terminal && !can_retire()) {
      throw std::logic_error("operation is not quiescent");
    }
    _phase = next;
    if (next != operation_phase::pending && next != operation_phase::terminal) mark_work_started();
  }

  void mark_work_started() noexcept
  {
    _quiescent   = false;
    _final_event = false;
  }

  /** @brief Track an already-recorded event; this does not enqueue CUDA work. */
  [[nodiscard]] recorded_event track_event(cudaEvent_t event, operation_phase phase, bool final)
  {
    if (event == nullptr ||
        (phase != operation_phase::staging_copy && phase != operation_phase::io_inflight &&
         phase != operation_phase::copy_inflight)) {
      throw std::invalid_argument("operation event needs a handle and an event-bearing phase");
    }
    if (_event_generation == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("operation event generation exhausted");
    }
    set_phase(phase);
    _event       = event;
    _final_event = final;
    ++_event_generation;
    return {this, event, _event_generation, phase};
  }

  /**
   * @brief Accept only the current final event's successful host observation.
   * @param observed_status The result of cudaEventQuery, NOT cudaEventRecord/StreamWaitEvent.
   * @note The caller queries on the owning device. Not-ready, errors, stale generations and
   *       intermediate events never establish quiescence. Generations survive pool reuse.
   */
  [[nodiscard]] bool observe_final_event(recorded_event const& event,
                                         cudaError_t observed_status) noexcept
  {
    if (observed_status != cudaSuccess || !_final_event || event.op != this ||
        event.generation != _event_generation || event.event != _event || event.phase != _phase) {
      return false;
    }
    _quiescent = true;
    return true;
  }

  /** @brief Recovery-only proof that all submitter, CUDA and host accesses have stopped. */
  void prove_recovery_quiescence() noexcept
  {
    _quiescent   = true;
    _final_event = false;
  }

  void quarantine(bool lane_already_quarantined = false) noexcept
  {
    mark_work_started();
    _phase = operation_phase::quarantined;
    if (lane && !lane_already_quarantined) lane->quarantine();
  }

  [[nodiscard]] bool can_retire() const noexcept { return _quiescent && !holds_qd_token; }

  std::shared_ptr<const io_object> file;
  // These future scheduler/fence/direct/scratch types are not implemented in Phase 2a.
  // Their owners must include the complete fence/event/stream or buffer lifetime proof.
  std::shared_ptr<void> input;
  std::shared_ptr<void> request;
  std::shared_ptr<void> direct;
  std::shared_ptr<void> scratch;
  std::optional<staging_lease> staging;
  std::optional<lane_lease> lane;
  bool holds_qd_token{false};

  async_args args;
  range physical;
  operation_targets targets;
  std::size_t valid_end{0};
  std::uint32_t attempts{0};
  std::uint64_t write_seq{0};
  bool cancelled{false};

 private:
  friend class operation_pool;
  friend struct detail::operation_pool_state;
  operation_kind _kind{operation_kind::read_staged};
  operation_phase _phase{operation_phase::pending};
  cudaEvent_t _event{nullptr};
  std::uint64_t _event_generation{0};
  bool _quiescent{true};
  bool _final_event{false};
  bool _allocated{false};
  bool _releasing{false};
  std::size_t _index{0};
  std::weak_ptr<detail::operation_pool_state> _pool;
};

namespace detail {

struct operation_pool_state {
  explicit operation_pool_state(std::size_t capacity, std::size_t targets)
    : records(std::make_unique<operation[]>(capacity)), capacity(capacity)
  {
    free.reserve(capacity);
    for (std::size_t index = 0; index < capacity; ++index) {
      records[index].targets.reserve(targets);
      free.push_back(capacity - index - 1);
    }
  }

  std::mutex mutex;
  std::unique_ptr<operation[]> records;
  std::size_t const capacity;
  std::vector<std::size_t> free;
  std::size_t in_use{0};
  // Explicitly retained until the last allocated record is retired. Destroying the facade
  // cannot invalidate outstanding async pointers; recovery may still release them statically.
  std::shared_ptr<operation_pool_state> live_owner;
};

}  // namespace detail

/**
 * @brief Thread-safe fixed-capacity pool: four stable records per configured device lane.
 *
 * No automatic live return: try_release rejects unsettled device/host accesses, even after
 * credits have resolved. Unpublished/quiescent records may be released, cancelling any
 * leftover credits. Allocated records self-retain past pool-facade destruction; callers MUST
 * eventually retire them explicitly (or deliberately retain a permanent quarantine).
 */
class operation_pool {
 public:
  explicit operation_pool(async_engine_config const& config)
  {
    config.validate();
    auto const lanes = static_cast<std::size_t>(config.device_lane_cap);
    if (lanes > std::numeric_limits<std::size_t>::max() / 4) {
      throw std::overflow_error("operation_pool capacity overflow");
    }
    _state = std::make_shared<detail::operation_pool_state>(
      4 * lanes, config.coalesce ? config.coalesce_max_targets : 1);
  }
  operation_pool(operation_pool const&)            = delete;
  operation_pool& operator=(operation_pool const&) = delete;
  operation_pool(operation_pool&&)                 = delete;
  operation_pool& operator=(operation_pool&&)      = delete;
  ~operation_pool()                                = default;

  /// A contended try-lock is a transient miss: retry next scheduler tick, not capacity exhaustion.
  [[nodiscard]] operation* try_acquire(operation_kind kind = operation_kind::read_staged) noexcept
  {
    std::unique_lock lock(_state->mutex, std::try_to_lock);
    if (!lock.owns_lock() || _state->free.empty()) return nullptr;
    auto const index = _state->free.back();
    _state->free.pop_back();
    auto& op = _state->records[index];
    assert(!op._allocated);
    op._kind      = kind;
    op._index     = index;
    op._pool      = _state;
    op._allocated = true;
    ++_state->in_use;
    _state->live_owner = _state;
    return &op;
  }

  /// May drop final file/resource references. The Phase 3 reaper must hand those references
  /// to its blocking cleanup pool before retirement (FR-5); credit settlement stays separate.
  [[nodiscard]] static bool try_release(operation& op) noexcept
  {
    auto state = op._pool.lock();
    if (state == nullptr) return false;
    std::size_t index;
    {
      std::lock_guard lock(state->mutex);
      if (!op._allocated || op._releasing || !op.can_retire()) return false;
      op._releasing = true;
      index         = op._index;
    }
    // Settlement can run coordinator continuations; never hold the pool mutex across it.
    auto const generation = op._event_generation;
    op.targets.clear();
    operation_targets storage;
    op.targets.swap_storage(storage);
    std::destroy_at(&op);
    std::construct_at(&op);
    op.targets.swap_storage(storage);
    op._event_generation = generation;
    {
      std::lock_guard lock(state->mutex);
      state->free.push_back(index);
      if (--state->in_use == 0) state->live_owner.reset();
    }
    return true;
  }

  /// Nonblocking recycle after final external owners/settled credits were detached.
  /// A miss changes nothing. Pool facade must remain alive on the calling thread.
  [[nodiscard]] static bool try_release_empty(operation& op) noexcept
  {
    auto state = op._pool.lock();
    if (!state) return false;
    std::unique_lock lock(state->mutex, std::try_to_lock);
    if (!lock || !op._allocated || op._releasing || !op.can_retire() || op.file || op.input ||
        op.request || op.direct || op.scratch || (op.lane && *op.lane) ||
        (op.staging && *op.staging) || op.targets.size())
      return false;
    auto const index      = op._index;
    auto const generation = op._event_generation;
    operation_targets storage;
    op.targets.swap_storage(storage);
    std::destroy_at(&op);
    std::construct_at(&op);
    op.targets.swap_storage(storage);
    op._event_generation = generation;
    state->free.push_back(index);
    if (--state->in_use == 0) state->live_owner.reset();
    return true;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return _state->capacity; }
  [[nodiscard]] std::size_t available() const noexcept
  {
    std::lock_guard lock(_state->mutex);
    return _state->free.size();
  }

 private:
  std::shared_ptr<detail::operation_pool_state> _state;
};

}  // namespace cucascade::io::cufile
