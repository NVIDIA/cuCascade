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

#include <cucascade/cuda/device_copy_batch.hpp>
#include <cucascade/exec/invocable.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/types.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <sys/uio.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io {

/**
 * @brief Fan-in shared by every physical operation derived from one logical read.
 *
 * The initial task count is one credit per prepared slice. Before a reactor
 * expands a slice into N physical operations it adds N-1 credits, then every
 * success, failure, or cancellation settles exactly one credit. The first
 * error stops further dispatch immediately, but the future is not fulfilled
 * until all already-published operations have drained.
 *
 * An optional finalizer (@ref set_finalizer) runs once when the last credit is
 * about to settle without an error, letting a backend append trailing work
 * (a linked fsync, a multipart Complete, ...) before the future resolves.
 */
class grouped_coordinator final {
 public:
  using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;

  /**
   * @brief Callback run once when all credits but the finalizing one settled.
   *
   * Runs synchronously on the thread that settles the last credit, outside any
   * coordinator lock.  While it runs the coordinator still holds one credit,
   * so the finalizer may call @ref add_tasks to schedule extra work (and must
   * do so before publishing that work).  To fail the request from inside the
   * finalizer, call @c add_tasks(1) followed by @ref report_error.  If it
   * schedules nothing the request resolves right after it returns.
   */
  using finalize_fn = exec::invocable<void(grouped_coordinator&) noexcept>;

  grouped_coordinator(std::size_t bytes_requested, std::size_t initial_tasks)
    : _bytes_requested(bytes_requested), _tasks_remaining(initial_tasks)
  {
  }

  grouped_coordinator(grouped_coordinator const&)            = delete;
  grouped_coordinator& operator=(grouped_coordinator const&) = delete;

  [[nodiscard]] bool should_continue() const noexcept
  {
    return _continue.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool has_error() const noexcept { return !should_continue(); }

  [[nodiscard]] std::size_t bytes_requested() const noexcept { return _bytes_requested; }

  [[nodiscard]] std::size_t tasks_remaining() const noexcept
  {
    return _tasks_remaining.load(std::memory_order_acquire);
  }

  /**
   * @brief Add credits for a slice expansion.
   *
   * This must happen before any derived operation can become visible to a
   * worker. The original slice credit keeps the count non-zero while it is
   * being expanded.
   */
  void add_tasks(std::size_t count) noexcept
  {
    if (count == 0) return;

    auto current = _tasks_remaining.load(std::memory_order_acquire);
    while (current != 0) {
      assert(current <= std::numeric_limits<std::size_t>::max() - count);
      if (_tasks_remaining.compare_exchange_weak(
            current, current + count, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
    assert(false && "cannot expand a completed grouped I/O request");
  }

  void on_complete() noexcept { settle_one(); }

  /**
   * @brief Install the finalizer (see @ref finalize_fn).
   *
   * Must be called by a thread that holds an unsettled credit of this
   * coordinator (so the count cannot reach zero concurrently); it may be
   * called before or after @ref get_future.  The finalizer is skipped when the
   * request has already failed when the last credit settles.
   *
   * @param fn Finalizer to run; must hold a target.
   * @throws std::invalid_argument if @p fn is empty.
   * @throws std::logic_error if a finalizer was already installed.
   */
  void set_finalizer(finalize_fn fn)
  {
    if (!fn) throw std::invalid_argument("grouped_coordinator finalizer is empty");
    std::lock_guard lock(_state_mutex);
    if (_has_finalizer.load(std::memory_order_acquire)) {
      throw std::logic_error("grouped_coordinator finalizer may only be set once");
    }
    _finalizer = std::move(fn);
    _has_finalizer.store(true, std::memory_order_release);
  }

  /// Whether a finalizer was installed.
  [[nodiscard]] bool has_finalizer() const noexcept
  {
    return _has_finalizer.load(std::memory_order_acquire);
  }

  /// Whether the finalizer has started (it runs at most once).
  [[nodiscard]] bool finalizer_started() const noexcept
  {
    return _finalizer_started.load(std::memory_order_acquire);
  }

  void report_error(error_type const& error,
                    std::source_location loc = std::source_location::current()) noexcept
  {
    bool expected = true;
    if (_continue.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
      std::exception_ptr converted;
      try {
        converted = to_exception_ptr(error, loc);
        if (converted == nullptr) {
          converted = std::make_exception_ptr(std::runtime_error("unknown I/O error"));
        }
      } catch (...) {
        converted = std::current_exception();
      }

      std::lock_guard lock(_state_mutex);
      assert(_first_exception == nullptr);
      _first_exception = std::move(converted);
    }
    settle_one();
  }
  [[nodiscard]] exec::semi_future<std::size_t> get_future()
  {
    exec::semi_future<std::size_t> future;
    {
      std::lock_guard lock(_state_mutex);
      assert(!_future_taken && "grouped coordinator future may only be retrieved once");
      future        = _promise.get_semi_future();
      _future_taken = true;
    }
    resolve_if_ready();
    return future;
  }

 private:
  [[nodiscard]] static std::exception_ptr to_exception_ptr(error_type const& error,
                                                           std::source_location loc)
  {
    if (std::holds_alternative<std::exception_ptr>(error)) {
      return std::get<std::exception_ptr>(error);
    }
    if (std::holds_alternative<cudaError_t>(error)) {
      auto const value = std::get<cudaError_t>(error);
      return std::make_exception_ptr(
        std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(value)) + " at " +
                           loc.file_name() + ":" + std::to_string(loc.line())));
    }

    auto const value = std::get<std::error_code>(error);
    return std::make_exception_ptr(std::system_error(
      value, "System error at " + std::string(loc.file_name()) + ":" + std::to_string(loc.line())));
  }

  /// Claim the finalizer when the caller holds the last outstanding credit.
  /// With a single credit left no other thread can add or settle credits, so
  /// the claim cannot race with the count; the CAS guards against re-entry.
  [[nodiscard]] bool try_claim_finalizer() noexcept
  {
    if (!_has_finalizer.load(std::memory_order_acquire) || has_error()) return false;
    bool expected = false;
    return _finalizer_started.compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  void run_finalizer() noexcept
  {
    finalize_fn fn;
    {
      std::lock_guard lock(_state_mutex);
      fn = std::move(_finalizer);
    }
    // Invoked without _state_mutex: the finalizer calls add_tasks and may
    // complete work inline (which re-enters settle_one / resolve_if_ready).
    if (fn) fn(*this);
  }

  void settle_one() noexcept
  {
    auto current = _tasks_remaining.load(std::memory_order_acquire);
    while (current != 0) {
      if (current == 1 && try_claim_finalizer()) {
        // Keep the last credit while the finalizer runs so that the work it
        // schedules (add_tasks) cannot drive the count to zero underneath it,
        // then settle that credit normally.
        run_finalizer();
        current = _tasks_remaining.load(std::memory_order_acquire);
        continue;
      }
      if (_tasks_remaining.compare_exchange_weak(
            current, current - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        if (current == 1) resolve_if_ready();
        return;
      }
    }
    assert(false && "a grouped I/O task was completed more than once");
  }

  void resolve_if_ready() noexcept
  {
    std::exception_ptr error;
    {
      std::lock_guard lock(_state_mutex);
      if (_tasks_remaining.load(std::memory_order_acquire) != 0 || !_future_taken || _fulfilled) {
        return;
      }
      _fulfilled = true;
      error      = _first_exception;
    }

    if (error != nullptr) {
      _promise.set_exception(std::move(error));
    } else {
      _promise.set_value(_bytes_requested);
    }
  }

  std::size_t const _bytes_requested;
  std::atomic<std::size_t> _tasks_remaining;
  std::atomic<bool> _continue{true};

  mutable std::mutex _state_mutex;
  std::exception_ptr _first_exception;
  bool _future_taken{false};
  bool _fulfilled{false};
  exec::promise<std::size_t> _promise;
  finalize_fn _finalizer;  // guarded by _state_mutex
  std::atomic<bool> _has_finalizer{false};
  std::atomic<bool> _finalizer_started{false};
};

namespace detail {

/// Process-wide monotonically increasing request id (starts at 1).
[[nodiscard]] inline std::uint64_t next_request_id() noexcept
{
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

}  // namespace detail

/**
 * @brief Scheduling / observability metadata carried by every grouped request.
 *
 * @c state is atomic so statistics may read it from any thread; the
 * timestamps are written by the single thread that owns the request at that
 * point of its lifecycle (submitter, queue, runner) and must only be read by
 * the owner or after the request completed.
 */
struct request_meta {
  using clock      = std::chrono::steady_clock;
  using time_point = clock::time_point;

  request_meta() noexcept = default;
  request_meta(request_class request_cls, io_kind request_kind) noexcept
    : cls(request_cls), kind(request_kind)
  {
  }

  request_meta(request_meta const&)            = delete;
  request_meta& operator=(request_meta const&) = delete;

  std::uint64_t id{detail::next_request_id()};              ///< process-wide unique id
  request_class cls{request_class::read};                   ///< resolved scheduling class
  io_kind kind{io_kind::read};                              ///< carried operation
  std::atomic<request_state> state{request_state::queued};  ///< lifecycle state
  time_point created_at{clock::now()};                      ///< request construction
  time_point enqueued_at{};                                 ///< pushed to the shared queue
  time_point assigned_at{};                                 ///< pulled by a runner
  time_point first_io_at{};                                 ///< first physical op submitted
  time_point completed_at{};                                ///< last physical op settled
  std::uint64_t runner_id{0};                               ///< runner that pulled the request
};

/**
 * @brief A queue entry containing logical slices and their shared fan-in.
 *
 * The object and fragment-pointer arrays are owned for the full asynchronous
 * lifetime. Reactors keep an active request locally and consume its slices in
 * order; they do not explode the group into queue entries before slot capacity
 * is known.
 *
 * A request carries exactly one kind of work (@c meta.kind):
 *  - @c io_kind::read: @ref slices, one coordinator credit per slice;
 *  - @c io_kind::write: @ref write_segments, one credit per segment;
 *  - @c io_kind::flush / @c io_kind::commit: a single control operation, one
 *    credit, taken with @ref take_control.
 */
class grouped_io_request final {
 public:
  /**
   * @brief Build a read request.
   *
   * @param object Object to read; must be non-null.
   * @param slices Prepared slices (one coordinator credit each).
   * @param coordinator Shared fan-in; must be non-null.
   * @param opts Scheduling options; @c automatic is resolved from the byte count.
   * @throws std::invalid_argument if @p object or @p coordinator is null.
   */
  static std::unique_ptr<grouped_io_request> create(
    std::shared_ptr<const io_object> object,
    std::vector<prepared_io_slice> slices,
    std::shared_ptr<grouped_coordinator> coordinator,
    io_options opts = {})
  {
    check_args(object, coordinator);
    auto const cls = resolve_request_class(opts.cls, io_kind::read, coordinator->bytes_requested());
    auto request   = std::unique_ptr<grouped_io_request>(
      new grouped_io_request(std::move(object), std::move(coordinator), cls, io_kind::read));
    request->slices = std::move(slices);
    return request;
  }

  static std::unique_ptr<grouped_io_request> create(std::shared_ptr<const io_object> object,
                                                    std::vector<prepared_io_slice> slices,
                                                    io_options opts = {})
  {
    std::size_t bytes = 0;
    for (auto const& slice : slices) {
      if (slice.size() > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("grouped I/O byte count overflow");
      }
      bytes += slice.size();
    }
    auto coordinator = std::make_shared<grouped_coordinator>(bytes, slices.size());
    return create(std::move(object), std::move(slices), std::move(coordinator), opts);
  }

  /**
   * @brief Build a write request.
   *
   * The caller validates the segments (@ref validate_write_segments) and sizes
   * @p coordinator with one credit per segment.
   *
   * @param object Object to write; must be non-null.
   * @param segments Disjoint write segments.
   * @param opts Write options; @c automatic class resolves to @c write.
   * @param coordinator Shared fan-in; must be non-null.
   * @throws std::invalid_argument if @p object or @p coordinator is null.
   */
  static std::unique_ptr<grouped_io_request> create_write(
    std::shared_ptr<const io_object> object,
    std::vector<write_segment> segments,
    write_options opts,
    std::shared_ptr<grouped_coordinator> coordinator)
  {
    check_args(object, coordinator);
    auto const cls =
      resolve_request_class(opts.cls, io_kind::write, coordinator->bytes_requested());
    auto request = std::unique_ptr<grouped_io_request>(
      new grouped_io_request(std::move(object), std::move(coordinator), cls, io_kind::write));
    request->write_segments = std::move(segments);
    request->wopts          = opts;
    return request;
  }

  /// As above, building a coordinator with one credit per segment and the
  /// segments' total byte count.
  static std::unique_ptr<grouped_io_request> create_write(std::shared_ptr<const io_object> object,
                                                          std::vector<write_segment> segments,
                                                          write_options opts = {})
  {
    auto const bytes = validate_write_segments(segments);
    auto coordinator = std::make_shared<grouped_coordinator>(bytes, segments.size());
    return create_write(std::move(object), std::move(segments), opts, std::move(coordinator));
  }

  /**
   * @brief Build a control request (@c io_kind::flush or @c io_kind::commit).
   *
   * @param object Target object; must be non-null.
   * @param kind @c io_kind::flush or @c io_kind::commit.
   * @param opts Write options (durability for commit); class resolves to @c write.
   * @param coordinator Shared fan-in holding one credit; must be non-null.
   * @throws std::invalid_argument on null arguments or a non-control @p kind.
   */
  static std::unique_ptr<grouped_io_request> create_control(
    std::shared_ptr<const io_object> object,
    io_kind kind,
    write_options opts,
    std::shared_ptr<grouped_coordinator> coordinator)
  {
    check_args(object, coordinator);
    if (kind != io_kind::flush && kind != io_kind::commit) {
      throw std::invalid_argument("grouped_io_request::create_control requires flush or commit");
    }
    auto const cls = resolve_request_class(opts.cls, kind, 0);
    auto request   = std::unique_ptr<grouped_io_request>(
      new grouped_io_request(std::move(object), std::move(coordinator), cls, kind));
    request->wopts            = opts;
    request->_control_pending = true;
    return request;
  }

  [[nodiscard]] io_kind kind() const noexcept { return meta.kind; }

  [[nodiscard]] bool is_write() const noexcept { return meta.kind == io_kind::write; }

  [[nodiscard]] bool is_control() const noexcept
  {
    return meta.kind == io_kind::flush || meta.kind == io_kind::commit;
  }

  /// True when no slice, segment or control operation remains to be taken.
  [[nodiscard]] bool empty() const noexcept
  {
    return _next == slices.size() && _next_segment == write_segments.size() && !_control_pending;
  }

  [[nodiscard]] std::size_t remaining_slices() const noexcept { return slices.size() - _next; }

  [[nodiscard]] std::size_t remaining_write_segments() const noexcept
  {
    return write_segments.size() - _next_segment;
  }

  /// Bytes of the slices and write segments not yet taken.
  [[nodiscard]] std::size_t remaining_bytes() const noexcept
  {
    std::size_t bytes = 0;
    for (std::size_t i = _next; i < slices.size(); ++i) {
      bytes += slices[i].size();
    }
    for (std::size_t i = _next_segment; i < write_segments.size(); ++i) {
      bytes += write_segments[i].size();
    }
    return bytes;
  }

  [[nodiscard]] prepared_io_slice& front() noexcept
  {
    assert(_next < slices.size());
    return slices[_next];
  }

  [[nodiscard]] prepared_io_slice take_front() noexcept
  {
    assert(_next < slices.size());
    return std::move(slices[_next++]);
  }

  [[nodiscard]] write_segment& front_write_segment() noexcept
  {
    assert(_next_segment < write_segments.size());
    return write_segments[_next_segment];
  }

  [[nodiscard]] write_segment take_front_write_segment() noexcept
  {
    assert(_next_segment < write_segments.size());
    return write_segments[_next_segment++];
  }

  /// Whether the control operation of a flush / commit request is still untaken.
  [[nodiscard]] bool control_pending() const noexcept { return _control_pending; }

  /// Take the control operation (its credit now belongs to the caller).
  void take_control() noexcept
  {
    assert(_control_pending);
    _control_pending = false;
  }

  /// Settle every untaken slice / segment / control credit with @p error.
  void cancel_remaining(grouped_coordinator::error_type const& error) noexcept
  {
    while (_next < slices.size()) {
      auto slice = take_front();
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
      coordinator->report_error(error);
    }
    while (_next_segment < write_segments.size()) {
      ++_next_segment;
      coordinator->report_error(error);
    }
    if (_control_pending) {
      _control_pending = false;
      coordinator->report_error(error);
    }
  }

  std::shared_ptr<const io_object> obj;
  std::vector<prepared_io_slice> slices;  ///< io_kind::read only
  std::shared_ptr<grouped_coordinator> coordinator;
  request_meta meta;                          ///< scheduling / observability metadata
  std::vector<write_segment> write_segments;  ///< io_kind::write only
  write_options wopts{};                      ///< io_kind::write / flush / commit

 private:
  grouped_io_request(std::shared_ptr<const io_object> object,
                     std::shared_ptr<grouped_coordinator> group,
                     request_class cls,
                     io_kind kind)
    : obj(std::move(object)), coordinator(std::move(group)), meta(cls, kind)
  {
  }

  static void check_args(std::shared_ptr<const io_object> const& object,
                         std::shared_ptr<grouped_coordinator> const& coordinator)
  {
    if (object == nullptr || coordinator == nullptr) {
      throw std::invalid_argument("grouped_io_request requires an object and coordinator");
    }
  }

  std::size_t _next{0};
  std::size_t _next_segment{0};
  bool _control_pending{false};
};

struct device_cpy_request {
  range req_rng;
  device_buffer d_buffer;
  int device_id{-1};

  /**
   * @brief Copy the logical request window out of physical I/O buffers.
   *
   * @p host_buf represents all bytes in @p io_rng in order. Aligned physical
   * over-read is skipped, fragmented sources are batched, and the optional
   * event is recorded after the final copy on the destination stream.
   */
  [[nodiscard]] cudaError_t copy_async(range io_rng,
                                       std::span<iovec const> host_buf,
                                       cudaEvent_t event = nullptr) const noexcept
  {
    try {
      if (d_buffer.data == nullptr) return cudaErrorInvalidValue;

      auto const copy_rng = intersect(req_rng, io_rng);
      if (copy_rng.empty()) return req_rng.empty() ? cudaSuccess : cudaErrorInvalidValue;

      int target_device = device_id >= 0 ? device_id : d_buffer.device_id;
      if (target_device < 0) {
        auto const status = cudaGetDevice(&target_device);
        if (status != cudaSuccess) return status;
      }
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{target_device}};

      std::size_t skip      = copy_rng.offset - io_rng.offset;
      std::size_t copied    = 0;
      auto* device_dst      = d_buffer.data + (copy_rng.offset - req_rng.offset);
      std::size_t remaining = copy_rng.size;

      cucascade::cuda::device_copy_batch batch;
      batch.reserve(host_buf.size());
      for (auto const& entry : host_buf) {
        auto const length = entry.iov_len;
        if (skip >= length) {
          skip -= length;
          continue;
        }

        auto const available = length - skip;
        auto const bytes     = std::min(available, remaining);
        auto const* source   = static_cast<std::uint8_t const*>(entry.iov_base) + skip;
        batch.add(device_dst + copied, source, bytes);
        copied += bytes;
        remaining -= bytes;
        skip = 0;
        if (remaining == 0) break;
      }

      if (remaining != 0) return cudaErrorInvalidValue;
      auto const copy_status = batch.enqueue(d_buffer.stream);
      if (copy_status != cudaSuccess) return copy_status;
      return event == nullptr ? cudaSuccess : cudaEventRecord(event, d_buffer.stream.get());
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
};

/**
 * @brief Backend-neutral physical operation produced by a reactor worker.
 *
 * Reactor-specific transfer state may be retained through @ref staging_owner.
 * Every terminal path must call exactly one of finish_success/finish_error;
 * the cache callback runs before the final coordinator decrement.
 */
struct io_op_request {
  std::shared_ptr<const io_object> obj;
  range io_rng;
  std::vector<iovec> iovecs;
  std::shared_ptr<void> staging_owner;
  std::unique_ptr<device_cpy_request> device_copy;
  std::shared_ptr<grouped_coordinator> coordinator;
  std::shared_ptr<prepared_io_completion> on_complete;
  std::vector<cache::cached_chunk*> completion_chunks;

  void finish_success() noexcept
  {
    if (_terminal.exchange(true, std::memory_order_acq_rel)) return;
    if (on_complete != nullptr) { (*on_complete)(completion_chunks, true); }
    coordinator->on_complete();
  }

  void finish_error(grouped_coordinator::error_type const& error,
                    bool host_data_valid = false) noexcept
  {
    if (_terminal.exchange(true, std::memory_order_acq_rel)) return;
    if (on_complete != nullptr) { (*on_complete)(completion_chunks, host_data_valid); }
    coordinator->report_error(error);
  }

  [[nodiscard]] bool terminal() const noexcept { return _terminal.load(std::memory_order_acquire); }

 private:
  std::atomic<bool> _terminal{false};
};

}  // namespace cucascade::io
