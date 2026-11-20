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
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <limits>
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

/// The process-wide submit gate of batch_submit_policy::exclusive: the reader whose wave is in
/// flight (null: none), and how many readers wait for it.
std::atomic<host_batch_reader const*> gate_owner{nullptr};
std::atomic<unsigned> gate_waiters{0};

/// A reader that hands the gate to a waiter does not take it back for this long, unless nobody
/// waits any more.
constexpr std::chrono::microseconds gate_yield{1000};
/// Longest sleep of a reap() whose deferred wave waits for another reader's gate.
constexpr std::chrono::microseconds gate_retry_sleep{100};

template <typename T>
[[nodiscard]] std::unique_ptr<T[]> allocate(std::size_t count) noexcept
{
  return std::unique_ptr<T[]>(new (std::nothrow) T[count]());
}

}  // namespace

host_batch_reader::host_batch_reader(unsigned depth,
                                     unsigned failure_limit,
                                     batch_submit_policy policy) noexcept
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
  if (!_params || !_events || !_entries || !_free || !_failed) {
    CUCASCADE_LOG_WARN("cufile: allocating a host batch reader of {} entries failed", capacity);
    return;
  }

  CUfileBatchHandle_t handle{nullptr};
  if (auto const status = api->batch_setup(&handle, capacity); status.err != CU_FILE_SUCCESS) {
    CUCASCADE_LOG_WARN(
      "cufile: cuFileBatchIOSetUp({}) failed (error {}); host batch reads are off",
      capacity,
      static_cast<int>(status.err));
    return;
  }

  _api      = api;
  _handle   = handle;
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
  if (_handle != nullptr) {
    // Release builds: never leave cuFile writing into destinations the owner may free.
    if (_in_flight > 0 || _needs_reset) { std::ignore = _api->batch_cancel(_handle); }
    _api->batch_destroy(_handle);
    _handle = nullptr;
  }
  _in_flight = 0;
  release_gate();
  stop_waiting();
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

  slot.state                  = entry_state::staged;
  auto& params                = _params[_staged++];
  params                      = CUfileIOParams_t{};
  params.mode                 = CUFILE_BATCH;
  params.u.batch.devPtr_base  = read.dst;
  params.u.batch.devPtr_offset = 0;
  params.u.batch.file_offset  = static_cast<off_t>(read.rng.offset);
  params.u.batch.size         = read.rng.size;
  params.fh                   = read.handle;
  params.opcode               = CUFILE_READ;
  params.cookie               = cookie_of(index);
}

void host_batch_reader::flush() noexcept
{
  _deferred = false;
  if (_staged == 0) { return; }
  auto const count    = _staged;
  auto const index_of = [this](std::size_t k) noexcept {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(_params[k].cookie) - 1U);
  };

  // Staging needs a usable handle (free_entries() > 0), and abort_handle() fails what is staged,
  // so this holds; fail the entries rather than submit through nothing if it ever does not.
  if (_handle == nullptr || _disabled || _needs_reset) {
    _staged = 0;
    stop_waiting();
    for (std::size_t k = 0; k < count; ++k) {
      mark_failed(index_of(k), static_cast<int>(CU_FILE_INVALID_VALUE));
    }
    return;
  }
  if (!may_submit()) {
    _deferred = true;
    ++_counters.deferred_flushes;
    return;
  }
  _staged = 0;

  auto const started = std::chrono::steady_clock::now();
  auto const status  = _api->batch_submit(_handle, static_cast<unsigned>(count), _params.get(), 0);
  _counters.submit_ns += static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
      .count());
  if (status.err == CU_FILE_SUCCESS) {
    for (std::size_t k = 0; k < count; ++k) {
      _entries[index_of(k)].state = entry_state::in_flight;
    }
    _in_flight += count;
    _consecutive_rejections = 0;
    ++_counters.submits;
    _counters.submitted_entries += count;
    return;
  }

  int const error = static_cast<int>(status.err);
  for (std::size_t k = 0; k < count; ++k) {
    mark_failed(index_of(k), error);
  }
  ++_counters.rejected_submits;
  ++_consecutive_rejections;
  _needs_reset = true;
  if (_consecutive_rejections >= _failure_limit) {
    disable("consecutive submits were rejected", error);
  } else {
    CUCASCADE_LOG_WARN(
      "cufile: cuFileBatchIOSubmit of {} entries failed (error {}, {} in flight); the batch "
      "handle will be recreated",
      count,
      error,
      _in_flight);
  }
  settle();
}

std::size_t host_batch_reader::poll_events(std::chrono::microseconds wait) noexcept
{
  if (_in_flight == 0 || _handle == nullptr) { return 0; }
  // Failures ready to report must not wait behind a blocking GetStatus.
  bool const poll = wait.count() <= 0 || (_failed_count > 0 && !_needs_reset);
  timespec timeout{};
  if (!poll) {
    auto const seconds = std::chrono::duration_cast<std::chrono::seconds>(wait);
    timeout.tv_sec     = static_cast<std::time_t>(seconds.count());
    timeout.tv_nsec =
      static_cast<long>(std::chrono::duration_cast<std::chrono::nanoseconds>(wait - seconds).count());
  }
  auto events = static_cast<unsigned>(_capacity);
  auto const status =
    _api->batch_get_status(_handle, poll ? 0U : 1U, &events, _events.get(), &timeout);
  if (status.err != CU_FILE_SUCCESS) {
    abort_handle(static_cast<int>(status.err));
    return 0;
  }
  return std::min(static_cast<std::size_t>(events), _capacity);
}

bool host_batch_reader::take_event(std::size_t i, batch_result& result) noexcept
{
  auto const& event  = _events[i];
  auto const cookie  = reinterpret_cast<std::uintptr_t>(event.cookie);
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
  if (event.status == CUFILE_WAITING || event.status == CUFILE_PENDING) { return false; }

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
    result.error   = event.status == CUFILE_COMPLETE ? static_cast<int>(static_cast<std::int64_t>(event.ret))
                                                     : static_cast<int>(event.status);
    ++_counters.failed;
  }
  slot.state           = entry_state::free;
  _free[_free_top++]   = index;
  --_in_flight;
  return true;
}

bool host_batch_reader::take_failure(batch_result& result) noexcept
{
  if (_needs_reset || _failed_count == 0) { return false; }
  auto const index = _failed[--_failed_count];
  auto& slot       = _entries[index];
  result           = batch_result{};
  result.token     = slot.read.token;
  result.outcome   = batch_result::status::failed;
  result.error     = slot.error;
  slot.state       = entry_state::free;
  _free[_free_top++] = index;
  ++_counters.failed;
  return true;
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
  if (_in_flight > 0) { return false; }  // this reader's wave is still running
  if (_owns_gate) {
    // Between waves (a done() callback refilling): let a waiting reader go first.
    if (gate_waiters.load(std::memory_order_acquire) == 0) { return true; }
    release_gate();
    return false;
  }
  if (_yield_until != std::chrono::steady_clock::time_point{}) {
    if (gate_waiters.load(std::memory_order_acquire) > 0 &&
        std::chrono::steady_clock::now() < _yield_until) {
      return false;
    }
    _yield_until = {};
  }
  host_batch_reader const* expected = nullptr;
  if (gate_owner.compare_exchange_strong(expected, this, std::memory_order_acq_rel)) {
    _owns_gate = true;
    stop_waiting();
    return true;
  }
  if (!_waiting) {
    _waiting = true;
    gate_waiters.fetch_add(1, std::memory_order_acq_rel);
  }
  return false;
}

void host_batch_reader::release_gate() noexcept
{
  if (!_owns_gate || _in_flight > 0) { return; }
  _owns_gate = false;
  gate_owner.store(nullptr, std::memory_order_release);
  if (gate_waiters.load(std::memory_order_acquire) > 0) {
    _yield_until = std::chrono::steady_clock::now() + gate_yield;
  }
}

void host_batch_reader::stop_waiting() noexcept
{
  if (!_waiting) { return; }
  _waiting = false;
  gate_waiters.fetch_sub(1, std::memory_order_acq_rel);
}

void host_batch_reader::settle() noexcept
{
  release_gate();
  if (!_needs_reset || _in_flight > 0) { return; }
  // The rejected submit may have left the handle in any state: cancel whatever it holds before
  // destroying it, so no rejected destination can still be written once it is reported.
  if (_handle != nullptr) {
    std::ignore = _api->batch_cancel(_handle);
    _api->batch_destroy(_handle);
    _handle = nullptr;
  }
  _needs_reset = false;
  if (_disabled) { return; }
  ++_counters.handle_resets;
  CUfileBatchHandle_t handle{nullptr};
  if (auto const status = _api->batch_setup(&handle, static_cast<unsigned>(_capacity));
      status.err != CU_FILE_SUCCESS) {
    disable("recreating the batch handle failed", static_cast<int>(status.err));
    return;
  }
  _handle = handle;
}

void host_batch_reader::abort_handle(int error) noexcept
{
  CUCASCADE_LOG_WARN(
    "cufile: cuFileBatchIOGetStatus failed (error {}); cancelling {} host batch entries in "
    "flight",
    error,
    _in_flight);
  if (_handle != nullptr) {
    std::ignore = _api->batch_cancel(_handle);
    _api->batch_destroy(_handle);
    _handle = nullptr;
  }
  for (std::uint32_t i = 0; i < _capacity; ++i) {
    if (_entries[i].state == entry_state::in_flight) { mark_failed(i, error); }
  }
  for (std::size_t k = 0; k < _staged; ++k) {
    mark_failed(static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(_params[k].cookie) - 1U),
                error);
  }
  _staged      = 0;
  _deferred    = false;
  _in_flight   = 0;
  _needs_reset = false;
  release_gate();
  stop_waiting();
  disable("cuFileBatchIOGetStatus failed", error);
}

void host_batch_reader::mark_failed(std::uint32_t index, int error) noexcept
{
  auto& slot                  = _entries[index];
  slot.state                  = entry_state::failed;
  slot.error                  = error;
  _failed[_failed_count++]    = index;
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

}  // namespace cucascade::io::cufile::detail
