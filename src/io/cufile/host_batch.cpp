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

/// Whether cuFile is done with an entry reported with @p status.  Anything else on a returned
/// event (@c CUFILE_TIMEOUT, @c CUFILE_INVALID, @c CUFILE_WAITING, @c CUFILE_PENDING) may leave
/// the destination cuFile's: the entry stays in flight.
[[nodiscard]] bool final_status(CUfileStatus_t status) noexcept
{
  return status == CUFILE_COMPLETE || status == CUFILE_FAILED || status == CUFILE_CANCELED;
}

/// One GetStatus call under the continuous policy.
constexpr timespec continuous_timeout{0, 1'000'000};

/// The process submit policy: fixed by the readers alive first, released with the last one.
struct process_policy_state {
  std::mutex mutex;
  std::size_t live{0};
  batch_submit_policy policy{batch_submit_policy::per_reader};
  bool warned{false};
};

[[nodiscard]] process_policy_state& process_policy() noexcept
{
  static process_policy_state state;
  return state;
}

template <typename T>
[[nodiscard]] std::unique_ptr<T[]> allocate(std::size_t count) noexcept
{
  return std::unique_ptr<T[]>(new (std::nothrow) T[count]());
}

}  // namespace

char const* batch_policy_name(batch_submit_policy policy) noexcept
{
  switch (policy) {
    case batch_submit_policy::exclusive: return "exclusive";
    case batch_submit_policy::per_reader: return "per_reader";
    case batch_submit_policy::continuous: return "continuous";
  }
  return "unknown";
}

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
  batch_submit_policy policy{batch_submit_policy::per_reader};
  wake_fn wake{nullptr};
  void* wake_context{nullptr};

  // -- gate waiters (guarded by gate_mutex() below) --
  shared_state* waiter_prev{nullptr};
  shared_state* waiter_next{nullptr};

  // -- mailbox --
  std::mutex mutex;
  std::condition_variable to_helper;  ///< A wave to submit, or stop.
  std::condition_variable to_owner;   ///< Something was published.
  std::atomic<bool> published{false};
  bool stop{false};
  /// The helper gave up (abort, poison, rejection limit, failed reset): nothing more may be
  /// handed over -- flush() fails staged entries itself.
  bool dead{false};
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
  bool poisoned{false};  ///< A wave outlived its GetStatus: its unreported entries stay.
  bool disabled{false};  ///< The helper gave up (rejections, reset failure, abort, poison).
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
  bool helper_poisoned{false};

  void publish_locked() noexcept { published.store(true, std::memory_order_release); }
  void notify_owner() noexcept
  {
    to_owner.notify_all();
    if (wake != nullptr) wake(wake_context);
  }
  /// Move everything handed over but not submitted to the rejected list (caller holds mutex).
  void reject_unsubmitted_locked(int error) noexcept
  {
    for (std::size_t k = 0; k < submit_count; ++k) {
      rejected[rejected_count++] = submit[k].cookie;
    }
    if (submit_count != 0) reject_error = error;
    submit_count = 0;
  }
  /// Release the process-wide gate when this reader holds it, waking the readers waiting for
  /// it (exclusive policy).
  void release_gate() const noexcept;
};

namespace {

/// The process-wide submit gate of batch_submit_policy::exclusive: the reader whose wave is in
/// flight (null: none), and the readers waiting for it (woken when it is released).
std::atomic<host_batch_reader const*> gate_owner{nullptr};

[[nodiscard]] std::mutex& gate_mutex() noexcept
{
  static std::mutex mutex;
  return mutex;
}

host_batch_reader::shared_state*& gate_waiters_head() noexcept
{
  static host_batch_reader::shared_state* head = nullptr;
  return head;
}

}  // namespace

void host_batch_reader::shared_state::release_gate() const noexcept
{
  auto const* expected = owner;
  if (!gate_owner.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) return;
  // A waiter links itself before retrying the gate (start_waiting), so either it sees the gate
  // free or it is in this list.
  std::lock_guard lock(gate_mutex());
  for (auto* waiter = gate_waiters_head(); waiter != nullptr; waiter = waiter->waiter_next) {
    if (waiter->wake != nullptr) waiter->wake(waiter->wake_context);
  }
}

host_batch_reader::host_batch_reader(unsigned depth,
                                     unsigned failure_limit,
                                     batch_submit_policy policy,
                                     wake_fn wake,
                                     void* wake_context) noexcept
  : host_batch_reader(host_batch_options{depth, failure_limit, policy, 0, wake, wake_context})
{
}

host_batch_reader::host_batch_reader(host_batch_options const& options) noexcept
  : _max_wave_bytes(options.max_wave_bytes),
    _failure_limit(options.failure_limit == 0 ? 1U : options.failure_limit),
    _policy(options.policy)
{
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_batch) {
    CUCASCADE_LOG_INFO("cufile: the cuFile batch API is unavailable; host batch reads are off");
    return;
  }
  if (options.depth == 0) { return; }
  if (!cufile_api::open_driver()) {
    CUCASCADE_LOG_INFO("cufile: the cuFile driver is not open; host batch reads are off");
    return;
  }
  unsigned capacity = options.depth;
  if (auto const limit = probe_platform().max_batch_io_size; limit != 0) {
    capacity = std::min(capacity, limit);
  }

  _params  = allocate<CUfileIOParams_t>(capacity);
  _events  = allocate<CUfileIOEvents_t>(capacity);
  _seen    = allocate<std::uint64_t>(capacity / 64 + 1);
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
  if (!_params || !_events || !_seen || !_entries || !_free || !_failed || !_shared ||
      !_shared->submit || !_shared->events || !_shared->rejected || !_shared->wave ||
      !_shared->wave_events || !_shared->pending_rejected) {
    CUCASCADE_LOG_WARN("cufile: allocating a host batch reader of {} entries failed", capacity);
    _shared.reset();
    return;
  }

  CUfileBatchHandle_t handle{nullptr};
  if (auto const status = api->batch_setup(&handle, capacity); status.err != CU_FILE_SUCCESS) {
    CUCASCADE_LOG_WARN(
      "cufile: cuFileBatchIOSetUp({}) failed (error {}); host batch reads are off for now",
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
      batch_policy_name(options.policy),
      batch_policy_name(_policy));
  }

  auto& shared         = *_shared;
  shared.owner         = this;
  shared.api           = api;
  shared.capacity      = capacity;
  shared.failure_limit = _failure_limit;
  shared.policy        = _policy;
  shared.wake          = options.wake;
  shared.wake_context  = options.wake_context;
  shared.handle        = handle;
  try {
    _helper = std::thread([this] { helper_loop(); });
  } catch (...) {
    CUCASCADE_LOG_WARN(
      "cufile: starting a host batch helper thread failed; host batch reads are off for now");
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
  assert((_in_flight == 0 || _poisoned) && "drain() a host_batch_reader before destroying it");
  if (_helper.joinable()) {
    {
      std::lock_guard lock(_shared->mutex);
      _shared->stop = true;
    }
    _shared->to_helper.notify_all();
    // The helper finishes whatever wave is in flight first: no destination is abandoned.
    _helper.join();
  }
  if (_shared != nullptr) {
    // A poisoned handle is leaked: cuFile may still dereference its state.
    if (_shared->handle != nullptr && !_shared->helper_poisoned) {
      _api->batch_destroy(_shared->handle);
    }
    _shared->handle = nullptr;
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
  slot.state       = entry_state::staged;

  bool const aligned = aligned_entry(read);
  assert(aligned && "host batch entries must be 4 KiB-aligned in address, offset and size");
  if (!aligned) {
    mark_failed(index, static_cast<int>(CU_FILE_INVALID_VALUE));
    return;
  }

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
  if (_disabled || _shared == nullptr) {
    stop_waiting();
    discard_staged();
    return;
  }
  if (!may_submit()) {
    _deferred = true;
    ++_counters.deferred_flushes;
    return;
  }
  // At most max_wave_bytes per wave (at least one entry).
  std::size_t count = 0;
  std::size_t bytes = 0;
  while (count < _staged) {
    auto const size = _params[count].u.batch.size;
    if (count != 0 && _max_wave_bytes != 0 && bytes + size > _max_wave_bytes) break;
    bytes += size;
    ++count;
  }
  bool handed = false;
  {
    std::lock_guard lock(_shared->mutex);
    // Checked under the mailbox lock: a helper that gave up never sees another wave, so every
    // entry is reported exactly once (here, or by the helper).
    if (!_shared->dead) {
      std::copy(
        _params.get(), _params.get() + count, _shared->submit.get() + _shared->submit_count);
      _shared->submit_count += count;
      handed = true;
    }
  }
  if (!handed) {
    // The helper's verdict reaches the owner with the next collect; fail what is staged now.
    _shared->release_gate();
    discard_staged();
    return;
  }
  _shared->to_helper.notify_one();
  for (std::size_t k = 0; k < count; ++k) {
    _entries[cookie_index(_params[k].cookie)].state = entry_state::in_flight;
  }
  _in_flight += count;
  std::copy(_params.get() + count, _params.get() + _staged, _params.get());
  _staged -= count;
  if (_staged != 0) {
    _deferred = true;
    ++_counters.deferred_flushes;
  }
}

bool host_batch_reader::may_submit() noexcept
{
  if (_policy == batch_submit_policy::continuous) { return true; }
  if (_in_flight > 0) { return false; }  // this reader's wave is not fully reported yet
  if (_policy == batch_submit_policy::per_reader) { return true; }
  host_batch_reader const* expected = nullptr;
  if (gate_owner.compare_exchange_strong(expected, this, std::memory_order_acq_rel)) {
    stop_waiting();
    return true;
  }
  // Link into the waiters first, then retry: a release either happened before (the retry sees
  // it) or will wake this reader.
  start_waiting();
  expected = nullptr;
  if (gate_owner.compare_exchange_strong(expected, this, std::memory_order_acq_rel)) {
    stop_waiting();
    return true;
  }
  return false;
}

void host_batch_reader::start_waiting() noexcept
{
  if (_waiting || _shared == nullptr) { return; }
  std::lock_guard lock(gate_mutex());
  auto& head                   = gate_waiters_head();
  _shared->waiter_prev = nullptr;
  _shared->waiter_next = head;
  if (head != nullptr) head->waiter_prev = _shared.get();
  head     = _shared.get();
  _waiting = true;
}

void host_batch_reader::stop_waiting() noexcept
{
  if (!_waiting) { return; }
  std::lock_guard lock(gate_mutex());
  auto& head = gate_waiters_head();
  if (_shared->waiter_prev != nullptr) {
    _shared->waiter_prev->waiter_next = _shared->waiter_next;
  } else {
    head = _shared->waiter_next;
  }
  if (_shared->waiter_next != nullptr) _shared->waiter_next->waiter_prev = _shared->waiter_prev;
  _shared->waiter_prev = nullptr;
  _shared->waiter_next = nullptr;
  _waiting             = false;
}

std::size_t host_batch_reader::collect(std::chrono::microseconds wait) noexcept
{
  if (_shared == nullptr) { return 0; }
  auto& shared = *_shared;
  std::size_t count{0};
  bool aborted{false};
  bool poisoned{false};
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
    count = std::min(shared.event_count, _capacity);
    std::copy(shared.events.get(), shared.events.get() + count, _events.get());
    shared.event_count = 0;
    for (std::size_t k = 0; k < shared.rejected_count; ++k) {
      mark_failed(cookie_index(shared.rejected[k]), shared.reject_error);
    }
    shared.rejected_count = 0;
    aborted               = shared.aborted;
    abort_error           = shared.abort_error;
    shared.aborted        = false;
    poisoned              = shared.poisoned;
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
    // The helper cancelled the handle and Destroy returned before it published this: fail every
    // entry still in flight except those with an event in this collect (take_event reports them).
    std::fill(_seen.get(), _seen.get() + (_capacity / 64 + 1), std::uint64_t{0});
    for (std::size_t i = 0; i < count; ++i) {
      auto const cookie = reinterpret_cast<std::uintptr_t>(_events[i].cookie);
      if (cookie == 0 || cookie > _capacity) continue;
      _seen[(cookie - 1) / 64] |= std::uint64_t{1} << ((cookie - 1) % 64);
    }
    for (std::uint32_t i = 0; i < _capacity; ++i) {
      bool const seen = (_seen[i / 64] >> (i % 64)) & 1U;
      if (_entries[i].state == entry_state::in_flight && !seen) mark_failed(i, abort_error);
    }
    stop_waiting();
    discard_staged();
    disable("cuFileBatchIOGetStatus failed", abort_error, false);
  } else if (disable_why != nullptr) {
    if (poisoned) _poisoned = true;
    disable(disable_why, disable_error, poisoned);
  }
  return count;
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
  assert(slot.state != entry_state::failed && slot.state != entry_state::free);
  if (slot.state == entry_state::failed || slot.state == entry_state::free) return;
  if (slot.state == entry_state::in_flight) --_in_flight;
  slot.state               = entry_state::failed;
  slot.error               = error;
  _failed[_failed_count++] = index;
}

void host_batch_reader::disable(char const* why, int error, bool as_error) noexcept
{
  if (_disabled) { return; }
  _disabled = true;
  if (as_error) {
    CUCASCADE_LOG_ERROR(
      "cufile: host batch reads disabled for this reactor: {} (error {}); its {} entries in "
      "flight stay in flight for good and the reactor cannot shut down cleanly",
      why,
      error,
      _in_flight);
  } else {
    CUCASCADE_LOG_WARN(
      "cufile: host batch reads disabled for this reactor: {} (error {}); reads take the "
      "synchronous path",
      why,
      error);
  }
}

void host_batch_reader::log_drain_stall(std::chrono::steady_clock::duration elapsed) const noexcept
{
  auto const seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
  if (seconds < drain_warning_interval.count()) {
    CUCASCADE_LOG_WARN(
      "cufile: draining host batch reads for {} s: {} entries in flight, {} waiting to be "
      "submitted",
      seconds,
      _in_flight,
      _staged);
  } else {
    CUCASCADE_LOG_ERROR(
      "cufile: draining host batch reads for {} s: {} entries in flight{}; waiting for the "
      "storage to complete them (a destination cuFile may still write is never abandoned)",
      seconds,
      _in_flight,
      _poisoned ? " on a poisoned reader (they will never be reported)" : "");
  }
}

void host_batch_reader::helper_loop() noexcept
{
  // No allocation, no logging on this thread: see the class comment.
  auto& s                = *_shared;
  auto const* api        = s.api;
  bool const wave_policy = s.policy != batch_submit_policy::continuous;

  // Give up for good (caller holds the mailbox lock): nothing more is handed over.
  auto die_locked = [&](char const* why, int error) noexcept {
    s.dead          = true;
    s.disabled      = true;
    s.disable_why   = why;
    s.disable_error = error;
    s.reject_unsubmitted_locked(error);
  };

  auto reset_and_report_rejected = [&]() noexcept {
    // Called with nothing in flight: the rejected entries are reported once the handle that
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
    {
      std::lock_guard lock(s.mutex);
      std::copy(s.pending_rejected.get(),
                s.pending_rejected.get() + s.pending_rejected_count,
                s.rejected.get() + s.rejected_count);
      s.rejected_count += s.pending_rejected_count;
      s.reject_error = s.pending_reject_error;
      if (reset) ++s.deltas.handle_resets;
      if (why != nullptr) die_locked(why, why_error);
      s.publish_locked();
    }
    s.pending_rejected_count = 0;
    s.release_gate();
    s.notify_owner();
  };

  std::unique_lock lock(s.mutex);
  for (;;) {
    s.to_helper.wait(lock, [&] {
      return s.stop || (!s.dead && s.submit_count > 0 && s.pending_rejected_count == 0) ||
             (!wave_policy && !s.dead && s.in_flight > 0);
    });
    std::size_t count = 0;
    if (!s.dead && s.submit_count > 0 && s.pending_rejected_count == 0) {
      count = s.submit_count;
      std::copy(s.submit.get(), s.submit.get() + count, s.wave.get());
      s.submit_count = 0;
    }
    // Stop only between waves: a wave in flight is always waited for (unless poisoned).
    if (count == 0 && s.stop && (s.in_flight == 0 || s.helper_poisoned || s.dead)) break;
    lock.unlock();

    if (count > 0) {
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

    // Reap.  Wave policies: the whole wave in ONE call (see the class comment).  Continuous:
    // whatever completed, briefly, so new submissions are taken in between.
    if (s.in_flight > 0 && !s.helper_poisoned) {
      auto const min_nr = wave_policy ? s.in_flight : std::size_t{1};
      auto events       = static_cast<unsigned>(s.capacity);
      timespec timeout  = continuous_timeout;
      if (wave_policy) {
        timeout = timespec{static_cast<std::time_t>(wave_timeout.count()), 0};
      }
      auto const status = api->batch_get_status(
        s.handle, static_cast<unsigned>(min_nr), &events, s.wave_events.get(), &timeout);
      if (status.err != CU_FILE_SUCCESS) {
        // The handle is broken: cuFile must not write any destination afterwards.  The entries
        // are failed by the owner only after Destroy returned (the publication follows it).
        std::ignore = api->batch_cancel(s.handle);
        api->batch_destroy(s.handle);
        s.handle    = nullptr;
        s.in_flight = 0;
        {
          std::lock_guard relock(s.mutex);
          s.aborted     = true;
          s.abort_error = static_cast<int>(status.err);
          s.submit_count = 0;  // handed over, never submitted: failed with the rest by the owner
          s.dead         = true;
          s.publish_locked();
        }
        s.release_gate();
        s.notify_owner();
      } else {
        auto const returned = std::min<std::size_t>(events, s.capacity);
        std::size_t finals  = 0;
        for (std::size_t k = 0; k < returned; ++k) {
          if (final_status(s.wave_events[k].status)) ++finals;
        }
        s.in_flight -= std::min(finals, s.in_flight);
        bool const partial = wave_policy && s.in_flight > 0;
        if (partial) s.helper_poisoned = true;
        // The wave is over (or the reader is poisoned): another reader may submit now.
        if ((s.in_flight == 0 && s.pending_rejected_count == 0) || partial) s.release_gate();
        {
          std::lock_guard relock(s.mutex);
          for (std::size_t k = 0; k < returned; ++k) {
            if (final_status(s.wave_events[k].status)) {
              s.events[s.event_count++] = s.wave_events[k];
            }
          }
          if (partial) {
            ++s.deltas.partial_status;
            s.poisoned = true;
            die_locked(
              "a wave did not complete within the cuFileBatchIOGetStatus timeout (or reported a "
              "non-final status); the reader is poisoned",
              0);
          }
          if (finals > 0 || partial) s.publish_locked();
        }
        if (finals > 0 || partial) s.notify_owner();
      }
    }
    if (s.pending_rejected_count > 0 && s.in_flight == 0 && !s.helper_poisoned) {
      reset_and_report_rejected();
    }
    lock.lock();
  }
}

}  // namespace cucascade::io::cufile::detail
