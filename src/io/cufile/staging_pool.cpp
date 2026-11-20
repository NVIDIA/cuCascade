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

#include "io/cufile/staging_pool.hpp"

#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

namespace detail {

namespace {

/// Slots start on a GPU page: BAR1 maps device memory in 64 KiB pages, and a
/// 4 KiB-aligned base is what lets cuFile DMA into it directly.
constexpr std::size_t gpu_page = 64UL << 10;

}  // namespace

staging_pool& staging_pool::instance()
{
  static auto* const pool = new staging_pool;  // intentionally never destroyed
  return *pool;
}

staging_slot staging_pool::acquire(int device,
                                   std::size_t bytes,
                                   std::size_t cap_bytes,
                                   bool require_registered) noexcept
{
  if (device < 0 || bytes == 0) { return {}; }
  auto const footprint = bytes + gpu_page;
  try {
    {
      std::lock_guard lock(_mutex);
      auto& state = _devices[device];
      for (auto it = state.free.begin(); it != state.free.end(); ++it) {
        if (it->bytes != bytes || (require_registered && !it->registered)) { continue; }
        auto const slot = *it;
        state.free.erase(it);
        ++state.leased;
        return slot;
      }
      if (require_registered) {
        // Nothing to allocate when the slot could not be registered anyway.
        auto const* api = cufile_api::get();
        if (api == nullptr || api->buf_register == nullptr ||
            state.registration_failed_for == cufile_api::generation()) {
          return {};
        }
      }
      if (cap_bytes != 0 && state.allocated_bytes + footprint > cap_bytes) { return {}; }
      // Claim the bytes now and room to keep the slot, so concurrent leases see
      // the cap and keeping it later cannot fail; the slow part runs unlocked.
      state.free.reserve(state.free.size() + 1);
      state.allocated_bytes += footprint;
    }

    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    void* raw = nullptr;
    if (auto const status = cudaMalloc(&raw, footprint); status != cudaSuccess) {
      std::ignore = cudaGetLastError();
      {
        std::lock_guard lock(_mutex);
        _devices[device].allocated_bytes -= footprint;
      }
      CUCASCADE_LOG_WARN(
        "cufile: cannot allocate {} bytes of staging on device {}: {}; reads there go straight "
        "to cuFile",
        footprint,
        device,
        cudaGetErrorString(status));
      return {};
    }
    auto const address = reinterpret_cast<std::uintptr_t>(raw);
    staging_slot slot{
      reinterpret_cast<std::uint8_t*>((address + gpu_page - 1) / gpu_page * gpu_page),
      bytes,
      device,
      false,
      raw,
      0};

    auto const* api       = cufile_api::get();
    auto const generation = cufile_api::generation();
    bool try_register     = false;
    {
      std::lock_guard lock(_mutex);
      try_register = _devices[device].registration_failed_for != generation;
    }
    CUfileError_t status{};
    status.err     = CU_FILE_SUCCESS;
    bool attempted = false;
    if (try_register && api != nullptr && api->buf_register != nullptr &&
        cufile_api::open_driver()) {
      attempted       = true;
      status          = api->buf_register(slot.base, slot.bytes, 0);
      slot.registered = status.err == CU_FILE_SUCCESS;
      if (slot.registered) slot.registered_generation = generation;
    }

    bool keep_unregistered = false;
    {
      std::lock_guard lock(_mutex);
      auto& state = _devices[device];
      if (slot.registered) { state.registered_bytes += slot.bytes; }
      // A failed registration, or one that could not even be tried (no driver),
      // is not retried for this table.
      if (!slot.registered && (attempted || require_registered)) {
        state.registration_failed_for = generation;
      }
      if (require_registered && !slot.registered) {
        state.free.push_back(slot);  // capacity reserved above
        keep_unregistered = true;
      } else {
        ++state.leased;
      }
    }
    if (attempted && !slot.registered) {
      CUCASCADE_LOG_WARN(
        "cufile: cuFileBufRegister of {} bytes on device {} failed: {} ({}); staged reads there "
        "are not registered (BAR1 or cuFile's max_device_pinned_mem_size may be exhausted)",
        slot.bytes,
        device,
        cufileop_status_error(status.err),
        static_cast<int>(status.err));
    }
    return keep_unregistered ? staging_slot{} : slot;
  } catch (std::exception const& e) {
    CUCASCADE_LOG_WARN("cufile: staging allocation on device {} failed: {}", device, e.what());
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile: staging allocation on device {} failed", device);
  }
  return {};
}

void staging_pool::release(staging_slot slot) noexcept
{
  if (!slot) { return; }
  std::lock_guard lock(_mutex);
  try {
    auto& state = _devices[slot.device];
    state.free.push_back(slot);
    if (state.leased != 0) { --state.leased; }
  } catch (...) {
    // Cannot keep it: leak it rather than free memory a device sync would cost.
  }
}

std::size_t staging_pool::release_free(int device) noexcept
{
  std::vector<staging_slot> doomed;
  {
    std::lock_guard lock(_mutex);
    auto const it = _devices.find(device);
    if (it == _devices.end()) return 0;
    doomed.swap(it->second.free);
    // With the memory back, registering may succeed again.
    it->second.registration_failed_for.reset();
  }
  std::size_t freed = 0;
  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    for (auto const& slot : doomed) {
      // Deregister through the table that registered it: libcufile's own, or a
      // test's substitute while it is still installed (a removed one is gone).
      if (slot.registered) {
        cufile_api const* owner = nullptr;
        if (slot.registered_generation == 0) {
          owner = cufile_api::loaded();
        } else if (slot.registered_generation == cufile_api::generation()) {
          owner = cufile_api::get();
        }
        if (owner != nullptr && owner->buf_deregister != nullptr) {
          std::ignore = owner->buf_deregister(slot.base);
        }
      }
      if (cudaFree(slot.allocation) != cudaSuccess) {
        std::ignore = cudaGetLastError();
        continue;
      }
      freed += slot.bytes + gpu_page;
      std::lock_guard lock(_mutex);
      auto& state = _devices[device];
      state.allocated_bytes -= std::min(state.allocated_bytes, slot.bytes + gpu_page);
      if (slot.registered) {
        state.registered_bytes -= std::min(state.registered_bytes, slot.bytes);
      }
    }
  } catch (...) {
    // A failed device switch: whatever was not freed is leaked, not reused.
  }
  return freed;
}

staging_usage staging_pool::usage(int device) const noexcept
{
  staging_usage usage;
  std::lock_guard lock(_mutex);
  if (auto const it = _devices.find(device); it != _devices.end()) {
    usage.allocated_bytes  = it->second.allocated_bytes;
    usage.registered_bytes = it->second.registered_bytes;
    usage.free_slots       = it->second.free.size();
    usage.leased_slots     = it->second.leased;
  }
  return usage;
}

}  // namespace detail

std::size_t reserve_staging(int device, std::size_t slots, config const& cfg) noexcept
{
  return reserve_staging_slots(
    device, slots, cfg.staging_slot_bytes(), cfg.max_staging_bytes_per_device, true);
}

std::size_t reserve_staging_slots(int device,
                                  std::size_t slots,
                                  std::size_t slot_bytes,
                                  std::size_t cap_bytes,
                                  bool require_registered) noexcept
{
  auto& pool       = detail::staging_pool::instance();
  auto const bytes = slot_bytes;
  std::vector<detail::staging_slot> taken;
  try {
    taken.reserve(slots);
  } catch (...) {
    return 0;
  }
  // Leasing takes free slots first, so leasing `slots` and handing them back
  // leaves at least that many registered slots on the device.
  for (std::size_t i = 0; i < slots; ++i) {
    auto slot = pool.acquire(device, bytes, cap_bytes, require_registered);
    if (!slot) { break; }
    taken.push_back(slot);
  }
  auto const count = taken.size();
  for (auto const& slot : taken) {
    pool.release(slot);
  }
  return count;
}

staging_usage staging_usage_of(int device) noexcept
{
  return detail::staging_pool::instance().usage(device);
}

std::size_t release_staging(int device) noexcept
{
  return detail::staging_pool::instance().release_free(device);
}

}  // namespace cucascade::io::cufile
