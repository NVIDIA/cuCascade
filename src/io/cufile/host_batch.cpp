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

#include "io/cufile/host_batch.hpp"

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <cufile.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <limits>
#include <mutex>
#include <new>
#include <thread>

namespace cucascade::io::cufile::detail {

namespace {

[[nodiscard]] void* cookie_of(std::uint32_t index) noexcept
{
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(index) + 1U);
}

[[nodiscard]] std::uint32_t index_of(void* cookie) noexcept
{
  return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(cookie) - 1U);
}

/// Whether @p read is fully 4 KiB-aligned, non-empty and addressable by @c off_t.
[[nodiscard]] bool aligned_entry(batch_read const& read) noexcept
{
  constexpr auto off_max = static_cast<std::size_t>(std::numeric_limits<off_t>::max());
  auto const address     = reinterpret_cast<std::uintptr_t>(read.dst);
  return read.dst != nullptr && read.handle != nullptr && read.rng.size != 0 &&
         address % IO_BLOCK_SIZE == 0 && read.rng.offset % IO_BLOCK_SIZE == 0 &&
         read.rng.size % IO_BLOCK_SIZE == 0 && read.rng.offset <= off_max &&
         read.rng.size <= off_max - read.rng.offset;
}

[[nodiscard]] bool final_status(CUfileStatus_t status) noexcept
{
  return status != CUFILE_WAITING && status != CUFILE_PENDING;
}

/// The process-wide submit gate of batch_submit_policy::exclusive: the reader whose wave is in
/// flight (null: none), and how many readers wait for it.
std::atomic<host_batch_reader const*> gate_owner{nullptr};
std::atomic<unsigned> gate_waiters{0};

/// A reader whose wave just completed does not take the gate back for this long while another
/// reader waits for it.
constexpr std::chrono::microseconds gate_yield{1000};
/// Longest sleep of a reap() whose deferred wave waits for another reader's gate.
constexpr std::chrono::microseconds gate_retry_sleep{100};
/// One GetStatus call of a whole wave waits at most this long (then it is counted in
/// partial_status and repeated).
constexpr timespec wave_timeout{1, 0};
/// One GetStatus call under the continuous policy.
constexpr timespec continuous_timeout{0, 1'000'000};

/// The process submit policy: fixed by the readers alive first, released with the last one.
struct process_policy_state {
  std::mutex mutex;
  std::size_t live{0};
  batch_submit_policy policy{batch_submit_policy::exclusive};
  bool warned{false};
};

[[nodiscard]] process_policy_state& process_policy() noexcept
{
  static process_policy_state state;
  return state;
}

[[nodiscard]] char const* policy_name(batch_submit_policy policy) noexcept
{
  switch (policy) {
    case batch_submit_policy::exclusive: return "exclusive";
    case batch_submit_policy::per_reader: return "per_reader";
    case batch_submit_policy::continuous: return "continuous";
  }
  return "unknown";
}

template <typename T>
[[nodiscard]] std::unique_ptr<T[]> allocate(std::size_t count) noexcept
{
  return std::unique_ptr<T[]>(new (std::nothrow) T[count]());
}

}  // namespace

/**
 * State shared by the owner and the helper thread.  The fields under "mailbox" are guarded by
 * @c mutex; the "helper only" fields are touched by the helper alone (and by the owner after the
 * helper was joined).
 */
struct host_batch_reader::shared_state {
  // -- configuration (immutable while the helper runs) --
  host_batch_reader const* owner{nullptr};
  cufile_api const* api{nullptr};
  std::size_t capacity{0};
  unsigned failure_limit{1};
  batch_submit_policy policy{batch_submit_policy::exclusive};
  wake_fn wake{nullptr};
  void* wake_context{nullptr};

  // -- mailbox --
  std::mutex mutex;
  std::condition_variable to_helper;  ///< A wave to submit, or stop.
  std::condition_variable to_owner;   ///< Something was published.
  std::atomic<bool> published{false};
  bool stop{false};
  /// Owner -> helper: the entries to submit next.
  std::unique_ptr<CUfileIOParams_t[]> submit;
  std::size_t submit_count{0};
  /// Helper -> owner: completions, rejected entries (cookies) and verdicts.
  std::unique_ptr<CUfileIOEvents_t[]> events;
  std::size_t event_count{0};
  std::unique_ptr<void*[]> rejected;
  std::size_t rejected_count{0};
  int reject_error{0};
  bool aborted{false};  ///< GetStatus failed: the handle is gone; everything in flight failed.
  int abort_error{0};
  bool disabled{false};  ///< The helper gave up (rejections, reset failure, abort).
  char const* disable_why{nullptr};
  int disable_error{0};
  host_batch_counters deltas;  ///< Helper-side counter increments not yet collected.

  // -- helper only --
  CUfileBatchHandle_t handle{nullptr};
  std::unique_ptr<CUfileIOParams_t[]> wave;
  std::unique_ptr<CUfileIOEvents_t[]> wave_events;
  std::unique_ptr<void*[]> pending_rejected;  ///< Rejected, reported after the handle reset.
  std::size_t pending_rejected_count{0};
  int pending_reject_error{0};
  std::size_t in_flight{0};
  unsigned consecutive_rejections{0};

  /// Publish (caller holds @c mutex), then wake the owner after unlocking.
  void publish_locked() noexcept { published.store(true, std::memory_order_release); }
  void notify_owner() noexcept
  {
    to_owner.notify_all();
    if (wake != nullptr) wake(wake_context);
  }
  /// Release the process-wide gate when this reader holds it.
  void release_gate() const noexcept
  {
    auto const* expected = owner;
    gate_owner.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
  }
};

host_batch_reader::host_batch_reader(unsigned depth,
                                     unsigned failure_limit,
                                     batch_submit_policy policy,
                                     wake_fn wake,
                                     void* wake_context) noexcept
  : _failure_limit(failure_limit == 0 ? 1U : failure_limit), _policy(policy)
{
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_batch) {
    CUCASCADE_LOG_INFO("cufile: the cuFile batch API is unavailable; host batch reads are off");
    return;
  }
  if (depth == 0) { return; }
  if (!cufile_api::open_driver()) {
    CUCASCADE_LOG_INFO("cufile: the cuFile driver is not open; host batch reads are off");
    return;
  }
  unsigned capacity = depth;
  if (auto const limit = probe_platform().max_batch_io_size; limit != 0) {
    capacity = std::min(capacity, limit);
  }

  _params  = allocate<CUfileIOParams_t>(capacity);
  _events  = allocate<CUfileIOEvents_t>(capacity);
  _entries = allocate<entry>(capacity);
  _free    = allocate<std::uint32_t>(capacity);
  _failed  = allocate<std::uint32_t>(capacity);
  _shared.reset(new (std::nothrow) shared_state);
  if (_shared != nullptr) {
    _shared->submit           = allocate<CUfileIOParams_t>(capacity);
    _shared->events           = allocate<CUfileIOEvents_t>(capacity);
    _shared->rejected         = allocate<void*>(capacity);
    _shared->wave             = allocate<CUfileIOParams_t>(capacity);
    _shared->wave_events      = allocate<CUfileIOEvents_t>(capacity);
    _shared->pending_rejected = allocate<void*>(capacity);
  }
  if (!_params || !_events || !_entries || !_free || !_failed || !_shared || !_shared->submit ||
      !_shared->events || !_shared->rejected || !_shared->wave || !_shared->wave_events ||
      !_shared->pending_rejected) {
    CUCASCADE_LOG_WARN("cufile: allocating a host batch reader of {} entries failed", capacity);
    _shared.reset();
    return;
  }

  CUfileBatchHandle_t handle{nullptr};
  if (auto const status = api->batch_setup(&handle, capacity); status.err != CU_FILE_SUCCESS) {
    CUCASCADE_LOG_WARN(
      "cufile: cuFileBatchIOSetUp({}) failed (error {}); host batch reads are off",
      capacity,
      static_cast<int>(status.err));
    _shared.reset();
    return;
  }

  // Join the process policy (a mutex: construction and destruction only).
  auto& process = process_policy();
  bool mismatch = false;
  {
    std::lock_guard lock(process.mutex);
    if (process.live == 0) {
      process.policy = _policy;
    } else if (process.policy != _policy) {
      mismatch       = !process.warned;
      process.warned = true;
      _policy        = process.policy;
    }
    ++process.live;
    _registered = true;
  }
  if (mismatch) {
    CUCASCADE_LOG_WARN(
      "cufile: a host batch reader configured with submit policy '{}' uses the process policy "
      "'{}' of the readers already alive (mixing policies is unsupported; logged once)",
      policy_name(policy),
      policy_name(_policy));
  }

  auto& shared         = *_shared;
  shared.owner         = this;
  shared.api           = api;
  shared.capacity      = capacity;
  shared.failure_limit = _failure_limit;
  shared.policy        = _policy;
  shared.wake          = wake;
  shared.wake_context  = wake_context;
  shared.handle        = handle;
  try {
    _helper = std::thread([this] { helper_loop(); });
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile: starting a host batch helper thread failed; host batch reads are "
                       "off");
    api->batch_destroy(handle);
    shared.handle = nullptr;
    return;
  }
  std::ignore = cucascade::exec::thread_util::set_thread_name(_helper, "cufile-batch");

  _api      = api;
  _capacity = capacity;
  // Pop order 0, 1, 2, ...: the first entries reuse the same indices under light load.
  for (std::uint32_t i = 0; i < capacity; ++i) {
    _free[i] = capacity - 1U - i;
  }
  _free_top = capacity;
  _disabled = false;
}

host_batch_reader::~host_batch_reader()
{
  assert(_in_flight == 0 && "drain() a host_batch_reader before destroying it");
  if (_helper.joinable()) {
    {
      std::lock_guard lock(_shared->mutex);
      _shared->stop = true;
    }
    _shared->to_helper.notify_all();
    // The helper finishes whatever is in flight first: no destination is abandoned.
    _helper.join();
  }
  if (_shared != nullptr) {
    if (_shared->handle != nullptr) {
      _api->batch_destroy(_shared->handle);
      _shared->handle = nullptr;
    }
    _shared->release_gate();
  }
  stop_waiting();
  if (_registered) {
    auto& process = process_policy();
    std::lock_guard lock(process.mutex);
    --process.live;
  }
}

bool host_batch_reader::completions_pending() const noexcept
{
  return _shared != nullptr && _shared->published.load(std::memory_order_acquire);
}

void host_batch_reader::add(batch_read const& read) noexcept
{
  assert(free_entries() > 0 && "host_batch_reader::add() without a free entry");
  if (free_entries() == 0) {
    CUCASCADE_LOG_ERROR("cufile: host_batch_reader::add() without a free entry; read dropped");
    return;
  }
  auto const index = _free[--_free_top];
  auto& slot       = _entries[index];
  slot.read        = read;
  slot.error       = 0;

  bool const aligned = aligned_entry(read);
  assert(aligned && "host batch entries must be 4 KiB-aligned in address, offset and size");
  if (!aligned) {
    mark_failed(index, static_cast<int>(CU_FILE_INVALID_VALUE));
    return;
  }

  slot.state                   = entry_state::staged;
  auto& params                 = _params[_staged++];
  params                       = CUfileIOParams_t{};
  params.mode                  = CUFILE_BATCH;
  params.u.batch.devPtr_base   = read.dst;
  params.u.batch.devPtr_offset = 0;
  params.u.batch.file_offset   = static_cast<off_t>(read.rng.offset);
  params.u.batch.size          = read.rng.size;
  params.fh                    = read.handle;
  params.opcode                = CUFILE_READ;
  params.cookie                = cookie_of(index);
}

void host_batch_reader::flush() noexcept
{
  _deferred = false;
  if (_staged == 0) { return; }
  auto const count = _staged;
  // Staging needs a usable reader (free_entries() > 0); fail the entries rather than hand them
  // to a helper that gave up if that ever does not hold.
  if (_disabled || _shared == nullptr) {
    _staged = 0;
    stop_waiting();
    for (std::size_t k = 0; k < count; ++k) {
      mark_failed(index_of(_params[k].cookie), static_cast<int>(CU_FILE_INVALID_VALUE));
    }
    return;
  }
  if (!may_submit()) {
    _deferred = true;
    ++_counters.deferred_flushes;
    return;
  }
  {
    std::lock_guard lock(_shared->mutex);
    std::copy(_params.get(), _params.get() + count, _shared->submit.get() + _shared->submit_count);
    _shared->submit_count += count;
  }
  _shared->to_helper.notify_one();
  for (std::size_t k = 0; k < count; ++k) {
    _entries[index_of(_params[k].cookie)].state = entry_state::in_flight;
  }
  _in_flight += count;
  _staged = 0;
}

std::chrono::microseconds host_batch_reader::retry_deferred(std::chrono::microseconds wait) noexcept
{
  if (!_deferred) { return wait; }
  flush();
  if (!_deferred || _in_flight > 0 || wait.count() <= 0) { return wait; }
  // Another reader's wave holds the gate: nothing of ours can complete meanwhile.
  auto const nap = std::min(wait, gate_retry_sleep);
  std::this_thread::sleep_for(nap);
  flush();
  return wait - nap;
}

bool host_batch_reader::may_submit() noexcept
{
  if (_policy == batch_submit_policy::continuous) { return true; }
  if (_in_flight > 0) { return false; }  // this reader's wave is not fully reported yet
  if (_policy == batch_submit_policy::per_reader) { return true; }
  if (_yield_until != std::chrono::steady_clock::time_point{}) {
    // Waiters other than this reader.
    auto const others = gate_waiters.load(std::memory_order_acquire) - (_waiting ? 1U : 0U);
    if (others > 0 && std::chrono::steady_clock::now() < _yield_until) {
      return false;
    }
    _yield_until = {};
  }
  host_batch_reader const* expected = nullptr;
  if (gate_owner.compare_exchange_strong(expected, this, std::memory_order_acq_rel)) {
    stop_waiting();
    return true;
  }
  if (!_waiting) {
    _waiting = true;
    gate_waiters.fetch_add(1, std::memory_order_acq_rel);
  }
  return false;
}

void host_batch_reader::stop_waiting() noexcept
{
  if (!_waiting) { return; }
  _waiting = false;
  gate_waiters.fetch_sub(1, std::memory_order_acq_rel);
}

std::size_t host_batch_reader::collect(std::chrono::microseconds wait) noexcept
{
  if (_shared == nullptr) { return 0; }
  auto& shared = *_shared;
  std::size_t count{0};
  bool aborted{false};
  int abort_error{0};
  char const* disable_why{nullptr};
  int disable_error{0};
  {
    std::unique_lock lock(shared.mutex);
    auto const ready = [&] { return shared.published.load(std::memory_order_relaxed); };
    if (!ready() && wait.count() > 0 && _in_flight > 0) {
      shared.to_owner.wait_for(lock, wait, ready);
    }
    if (!ready()) { return 0; }
    shared.published.store(false, std::memory_order_relaxed);
    count = shared.event_count;
    std::copy(shared.events.get(), shared.events.get() + count, _events.get());
    shared.event_count = 0;
    for (std::size_t k = 0; k < shared.rejected_count; ++k) {
      mark_failed(index_of(shared.rejected[k]), shared.reject_error);
    }
    shared.rejected_count = 0;
    aborted               = shared.aborted;
    abort_error           = shared.abort_error;
    shared.aborted        = false;
    if (shared.disabled) {
      disable_why     = shared.disable_why;
      disable_error   = shared.disable_error;
      shared.disabled = false;
    }
    auto& d = shared.deltas;
    _counters.submits += d.submits;
    _counters.submitted_entries += d.submitted_entries;
    _counters.rejected_submits += d.rejected_submits;
    _counters.handle_resets += d.handle_resets;
    _counters.submit_ns += d.submit_ns;
    _counters.partial_status += d.partial_status;
    d = host_batch_counters{};
  }
  if (aborted) {
    // The helper cancelled and destroyed the handle: cuFile no longer writes any destination.
    // Entries completed before the error are among the events collected above.
    for (std::size_t i = 0; i < count; ++i) {
      auto const cookie = reinterpret_cast<std::uintptr_t>(_events[i].cookie);
      if (cookie == 0 || cookie > _capacity) continue;
      // Report those first: take_event() frees them below; mark the rest failed now.
      _entries[cookie - 1].error = -1;  // marker: has an event
    }
    for (std::uint32_t i = 0; i < _capacity; ++i) {
      if (_entries[i].state == entry_state::in_flight && _entries[i].error != -1) {
        mark_failed(i, abort_error);
      }
    }
    for (std::uint32_t i = 0; i < _capacity; ++i) {
      if (_entries[i].error == -1) _entries[i].error = 0;
    }
    // What is staged was never handed over: it fails too.
    for (std::size_t k = 0; k < _staged; ++k) {
      mark_failed(index_of(_params[k].cookie), abort_error);
    }
    _staged   = 0;
    _deferred = false;
    stop_waiting();
    disable("cuFileBatchIOGetStatus failed", abort_error);
  } else if (disable_why != nullptr) {
    disable(disable_why, disable_error);
  }
  if (_policy == batch_submit_policy::exclusive && _in_flight == count && count > 0 &&
      gate_waiters.load(std::memory_order_acquire) > 0) {
    // This reader's wave just ended and another reader waits: let it go first.
    _yield_until = std::chrono::steady_clock::now() + gate_yield;
  }
  return std::min(count, _capacity);
}

bool host_batch_reader::take_event(std::size_t i, batch_result& result) noexcept
{
  auto const& event = _events[i];
  auto const cookie = reinterpret_cast<std::uintptr_t>(event.cookie);
  if (cookie == 0 || cookie > _capacity) {
    // Never expected: cuFile returns the cookies it was given.
    CUCASCADE_LOG_ERROR("cufile: cuFileBatchIOGetStatus returned an unknown cookie {:#x}", cookie);
    return false;
  }
  auto const index = static_cast<std::uint32_t>(cookie - 1U);
  auto& slot       = _entries[index];
  if (slot.state != entry_state::in_flight) {
    CUCASCADE_LOG_ERROR("cufile: cuFileBatchIOGetStatus reported entry {} twice", index);
    return false;
  }

  auto const size = slot.read.rng.size;
  result          = batch_result{};
  result.token    = slot.read.token;
  if (event.status == CUFILE_COMPLETE && static_cast<std::int64_t>(event.ret) >= 0) {
    if (event.ret >= size) {
      result.outcome = batch_result::status::ok;
      result.bytes   = size;
      ++_counters.completed;
    } else {
      result.outcome = batch_result::status::short_read;
      result.bytes   = event.ret;
      ++_counters.short_reads;
    }
    _counters.bytes += result.bytes;
  } else {
    result.outcome = batch_result::status::failed;
    result.error   = event.status == CUFILE_COMPLETE
                       ? static_cast<int>(static_cast<std::int64_t>(event.ret))
                       : static_cast<int>(event.status);
    ++_counters.failed;
  }
  slot.state         = entry_state::free;
  _free[_free_top++] = index;
  --_in_flight;
  return true;
}

bool host_batch_reader::take_failure(batch_result& result) noexcept
{
  if (_failed_count == 0) { return false; }
  auto const index   = _failed[--_failed_count];
  auto& slot         = _entries[index];
  result             = batch_result{};
  result.token       = slot.read.token;
  result.outcome     = batch_result::status::failed;
  result.error       = slot.error;
  slot.state         = entry_state::free;
  _free[_free_top++] = index;
  ++_counters.failed;
  return true;
}

void host_batch_reader::mark_failed(std::uint32_t index, int error) noexcept
{
  auto& slot = _entries[index];
  if (slot.state == entry_state::in_flight) --_in_flight;
  slot.state               = entry_state::failed;
  slot.error               = error;
  _failed[_failed_count++] = index;
}

void host_batch_reader::disable(char const* why, int error) noexcept
{
  if (_disabled) { return; }
  _disabled = true;
  CUCASCADE_LOG_WARN(
    "cufile: host batch reads disabled for this reactor: {} (error {}); reads take the "
    "synchronous path",
    why,
    error);
}

void host_batch_reader::log_drain_stall(std::chrono::steady_clock::duration elapsed) const noexcept
{
  CUCASCADE_LOG_WARN(
    "cufile: draining host batch reads for {} s: {} entries in flight, {} waiting to be "
    "submitted",
    std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(),
    _in_flight,
    _staged);
}

void host_batch_reader::helper_loop() noexcept
{
  // No allocation, no logging on this thread: see the class comment.
  auto& s         = *_shared;
  auto const* api = s.api;
  bool const wave_policy = s.policy != batch_submit_policy::continuous;
  bool gave_up           = false;

  auto reset_and_report_rejected = [&]() noexcept {
    // Called with nothing in flight: the rejected entries may be reported once the handle that
    // rejected them is gone.
    if (s.handle != nullptr) {
      std::ignore = api->batch_cancel(s.handle);
      api->batch_destroy(s.handle);
      s.handle = nullptr;
    }
    char const* why = nullptr;
    int why_error   = s.pending_reject_error;
    bool reset      = false;
    if (s.consecutive_rejections >= s.failure_limit) {
      why = "consecutive submits were rejected";
    } else {
      CUfileBatchHandle_t handle{nullptr};
      if (auto const status = api->batch_setup(&handle, static_cast<unsigned>(s.capacity));
          status.err != CU_FILE_SUCCESS) {
        why       = "recreating the batch handle failed";
        why_error = static_cast<int>(status.err);
      } else {
        s.handle = handle;
        reset    = true;
      }
    }
    if (why != nullptr) gave_up = true;
    {
      std::lock_guard lock(s.mutex);
      std::copy(s.pending_rejected.get(),
                s.pending_rejected.get() + s.pending_rejected_count,
                s.rejected.get() + s.rejected_count);
      s.rejected_count += s.pending_rejected_count;
      s.reject_error = s.pending_reject_error;
      if (reset) ++s.deltas.handle_resets;
      if (why != nullptr) {
        s.disabled      = true;
        s.disable_why   = why;
        s.disable_error = why_error;
      }
      s.publish_locked();
    }
    s.pending_rejected_count = 0;
    s.release_gate();
    s.notify_owner();
  };

  std::unique_lock lock(s.mutex);
  for (;;) {
    s.to_helper.wait(lock, [&] {
      return s.stop || (s.submit_count > 0 && s.pending_rejected_count == 0) ||
             (!wave_policy && s.in_flight > 0);
    });
    std::size_t count = 0;
    if (s.submit_count > 0 && s.pending_rejected_count == 0) {
      count = s.submit_count;
      std::copy(s.submit.get(), s.submit.get() + count, s.wave.get());
      s.submit_count = 0;
    }
    if (count == 0 && s.in_flight == 0 && s.stop) break;
    lock.unlock();

    if (count > 0) {
      if (s.handle == nullptr || gave_up) {
        // The reader gave up meanwhile (the owner had not seen it yet): refuse the wave.
        for (std::size_t k = 0; k < count; ++k) {
          s.pending_rejected[s.pending_rejected_count++] = s.wave[k].cookie;
        }
        s.pending_reject_error = static_cast<int>(CU_FILE_INVALID_VALUE);
        std::lock_guard relock(s.mutex);
        std::copy(s.pending_rejected.get(),
                  s.pending_rejected.get() + s.pending_rejected_count,
                  s.rejected.get() + s.rejected_count);
        s.rejected_count += s.pending_rejected_count;
        s.reject_error           = s.pending_reject_error;
        s.pending_rejected_count = 0;
        s.publish_locked();
        s.release_gate();
        s.notify_owner();
      } else {
        auto const started = std::chrono::steady_clock::now();
        auto const status =
          api->batch_submit(s.handle, static_cast<unsigned>(count), s.wave.get(), 0);
        auto const elapsed = std::chrono::steady_clock::now() - started;
        bool const ok      = status.err == CU_FILE_SUCCESS;
        if (ok) {
          s.in_flight += count;
          s.consecutive_rejections = 0;
        } else {
          ++s.consecutive_rejections;
          for (std::size_t k = 0; k < count; ++k) {
            s.pending_rejected[s.pending_rejected_count++] = s.wave[k].cookie;
          }
          s.pending_reject_error = static_cast<int>(status.err);
        }
        std::lock_guard relock(s.mutex);
        s.deltas.submit_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        if (ok) {
          ++s.deltas.submits;
          s.deltas.submitted_entries += count;
        } else {
          ++s.deltas.rejected_submits;
        }
      }
    }

    // Reap: under a wave policy the whole wave in one call (see the class comment); under
    // continuous whatever completed, briefly, so new submissions are taken in between.
    while (s.in_flight > 0) {
      auto const min_nr  = wave_policy ? s.in_flight : std::size_t{1};
      auto events        = static_cast<unsigned>(s.capacity);
      auto timeout       = wave_policy ? wave_timeout : continuous_timeout;
      auto const status  = api->batch_get_status(
        s.handle, static_cast<unsigned>(min_nr), &events, s.wave_events.get(), &timeout);
      if (status.err != CU_FILE_SUCCESS) {
        // The handle is broken: cuFile must not write any destination afterwards.
        std::ignore = api->batch_cancel(s.handle);
        api->batch_destroy(s.handle);
        s.handle    = nullptr;
        s.in_flight = 0;
        gave_up     = true;
        {
          std::lock_guard relock(s.mutex);
          s.aborted      = true;
          s.abort_error  = static_cast<int>(status.err);
          s.submit_count = 0;  // handed over, never submitted: failed with the rest
          s.publish_locked();
        }
        s.release_gate();
        s.notify_owner();
        break;
      }
      auto const returned = std::min<std::size_t>(events, s.capacity);
      std::size_t finals  = 0;
      for (std::size_t k = 0; k < returned; ++k) {
        if (final_status(s.wave_events[k].status)) ++finals;
      }
      s.in_flight -= std::min(finals, s.in_flight);
      // The wave is over: the next reader may submit while the owner collects this one.
      if (s.in_flight == 0 && s.pending_rejected_count == 0) s.release_gate();
      {
        std::lock_guard relock(s.mutex);
        for (std::size_t k = 0; k < returned; ++k) {
          if (final_status(s.wave_events[k].status)) s.events[s.event_count++] = s.wave_events[k];
        }
        if (finals < min_nr && wave_policy) ++s.deltas.partial_status;
        if (finals > 0) s.publish_locked();
      }
      if (finals > 0) s.notify_owner();
      if (!wave_policy) break;  // take new submissions between polls
    }
    if (s.pending_rejected_count > 0 && s.in_flight == 0) reset_and_report_rejected();
    lock.lock();
  }
}

}  // namespace cucascade::io::cufile::detail
