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

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/lane_pool.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <rmm/cuda_device.hpp>

#include <cassert>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {
namespace {

cufile_api const* default_stream_api(int device, async_engine_config const& config)
{
  config.validate();
  if (config.mode == engine_mode::legacy) {
    throw std::invalid_argument("lane_pool requires an enabled async engine mode");
  }
  if (device < 0) throw std::invalid_argument("lane_pool device must be nonnegative");
  if (!config.stream_register) return nullptr;
  auto& process_runtime = runtime::instance();
  auto const report     = process_runtime.ensure_initialized();
  auto const* api       = process_runtime.api();
  if (!report.driver_opened || api == nullptr || !api->has_stream_register) return nullptr;
  return api;
}

}  // namespace

namespace detail {

struct lane_pool_state {
  struct entry {
    std::unique_ptr<lane> resource;
    bool leased{false};
    bool quarantined{false};
  };

  lane_pool_state(int device, async_engine_config const& config, cufile_api const* api)
    : device(device),
      register_stream(config.stream_register),
      api(api ? std::optional<cufile_api>(*api) : std::nullopt)
  {
    config.validate();
    if (config.mode == engine_mode::legacy) {
      throw std::invalid_argument("lane_pool requires an enabled async engine mode");
    }
    if (device < 0) throw std::invalid_argument("lane_pool device must be nonnegative");
    auto const count = static_cast<std::size_t>(config.device_lane_cap);
    lanes.reserve(count);
    free.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      lanes.push_back(
        entry{std::unique_ptr<lane>(new lane(device, index, config.stream_register, api))});
      free.push_back(count - index - 1);
    }
  }

  int device;
  bool register_stream;
  std::optional<cufile_api> api;
  std::mutex mutex;
  std::vector<entry> lanes;

  std::unique_ptr<lane> replacement(std::size_t index)
  {
    return std::unique_ptr<lane>(new lane(device, index, register_stream, api ? &*api : nullptr));
  }
  std::vector<std::size_t> free;
  std::size_t quarantine_count{0};
  // A returned quarantine must not destroy or deregister resources during facade teardown.
  // Recovery explicitly breaks this retention once all quarantines have proved quiescent.
  std::shared_ptr<lane_pool_state> quarantine_owner;
};

}  // namespace detail

lane::lane(int device, std::size_t index, bool register_stream, cufile_api const* api)
  : _device(device), _index(index)
{
  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&_io_event, cudaEventDisableTiming));
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&_copy_event, cudaEventDisableTiming));
    if (register_stream && api != nullptr && api->has_stream_register &&
        api->stream_register != nullptr && api->stream_deregister != nullptr) {
      _stream_deregister = api->stream_deregister;
      auto const status  = api->stream_register(_stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS);
      if (status.err != CU_FILE_SUCCESS) {
        // Registration errors may leave partially initialized cuFile stream state.
        static_cast<void>(_stream_deregister(_stream));
        throw std::runtime_error("cuFileStreamRegister failed: " +
                                 std::to_string(static_cast<int>(status.err)));
      }
      _stream_registered = true;
    }
  } catch (...) {
    destroy();
    throw;
  }
}

lane::~lane() noexcept { destroy(); }

void lane::destroy() noexcept
{
  if (_stream == nullptr) return;
  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    if (_stream_registered) {
      static_cast<void>(_stream_deregister(_stream));
      _stream_registered = false;
    }
    if (_copy_event != nullptr) CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(_copy_event));
    if (_io_event != nullptr) CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(_io_event));
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(_stream));
    _copy_event = nullptr;
    _io_event   = nullptr;
    _stream     = nullptr;
  } catch (...) {
    // Destruction is best effort if CUDA can no longer select the owning device.
  }
}

lane_lease::lane_lease(std::shared_ptr<detail::lane_pool_state> state, lane* resource) noexcept
  : _state(std::move(state)), _lane(resource)
{
}

lane_lease::lane_lease(lane_lease&& other) noexcept
  : _state(std::move(other._state)), _lane(std::exchange(other._lane, nullptr))
{
}

lane_lease& lane_lease::operator=(lane_lease&& other) noexcept
{
  if (this != &other) {
    reset();
    _state = std::move(other._state);
    _lane  = std::exchange(other._lane, nullptr);
  }
  return *this;
}

lane_lease::~lane_lease() noexcept { reset(); }

void lane_lease::quarantine() noexcept
{
  if (_lane == nullptr) return;
  std::lock_guard lock(_state->mutex);
  auto& entry = _state->lanes[_lane->index()];
  if (!entry.quarantined) {
    entry.quarantined = true;
    ++_state->quarantine_count;
    _state->quarantine_owner = _state;
  }
}

void lane_lease::reset() noexcept
{
  if (_lane == nullptr) return;
  auto state = std::move(_state);
  {
    std::lock_guard lock(state->mutex);
    auto& entry = state->lanes[_lane->index()];
    assert(entry.leased);
    entry.leased = false;
    if (!entry.quarantined) state->free.push_back(_lane->index());
    _lane = nullptr;
  }
}

bool lane_lease::try_reset() noexcept
{
  if (!_lane) return true;
  // Never destroy a final external owner on the reaper.
  if (_state.use_count() == 1) return false;
  std::unique_lock lock(_state->mutex, std::try_to_lock);
  if (!lock) return false;
  auto& entry = _state->lanes[_lane->index()];
  assert(entry.leased);
  entry.leased = false;
  if (!entry.quarantined) _state->free.push_back(_lane->index());
  _lane = nullptr;
  lock.unlock();
  _state.reset();
  return true;
}

bool lane_pool::replace_after_quiescence(lane_lease& lease) noexcept
{
  if (!lease || lease._state != _state) return false;
  try {
    auto replacement = _state->replacement(lease->index());
    {
      std::lock_guard lock(_state->mutex);
      auto& entry = _state->lanes[lease->index()];
      if (!entry.leased || !entry.quarantined) return false;
      entry.resource.swap(replacement);
      lease._lane       = entry.resource.get();
      entry.quarantined = false;
      if (--_state->quarantine_count == 0) _state->quarantine_owner.reset();
    }
    // cuFile deregistration and all CUDA destruction are outside the pool mutex/reaper.
    return true;
  } catch (...) {
    return false;
  }
}

lane_pool::lane_pool(int device, async_engine_config const& config)
  : lane_pool(device, config, default_stream_api(device, config))
{
}

lane_pool::lane_pool(int device, async_engine_config const& config, cufile_api const* api)
  : _state(std::make_shared<detail::lane_pool_state>(device, config, api))
{
}

std::optional<lane_lease> lane_pool::try_lease() noexcept
{
  std::unique_lock lock(_state->mutex, std::try_to_lock);
  if (!lock.owns_lock() || _state->free.empty()) return std::nullopt;
  auto const index = _state->free.back();
  _state->free.pop_back();
  auto& entry = _state->lanes[index];
  assert(!entry.leased && !entry.quarantined);
  entry.leased = true;
  return lane_lease{_state, entry.resource.get()};
}

std::size_t lane_pool::capacity() const noexcept { return _state->lanes.size(); }

std::size_t lane_pool::available() const noexcept
{
  std::lock_guard lock(_state->mutex);
  return _state->free.size();
}

std::size_t lane_pool::quarantined() const noexcept
{
  std::lock_guard lock(_state->mutex);
  return _state->quarantine_count;
}

bool lane_pool::release_quarantined_after_quiescence(std::size_t index) noexcept
{
  std::lock_guard lock(_state->mutex);
  if (index >= _state->lanes.size()) return false;
  auto& entry = _state->lanes[index];
  if (entry.leased || !entry.quarantined) return false;
  entry.quarantined = false;
  _state->free.push_back(index);
  if (--_state->quarantine_count == 0) _state->quarantine_owner.reset();
  return true;
}

}  // namespace cucascade::io::cufile
