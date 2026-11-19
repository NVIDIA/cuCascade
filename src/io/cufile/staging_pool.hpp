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

#include <cucascade/io/cufile/staging.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace cucascade::io::cufile::detail {

/**
 * @brief A device staging slot: @c bytes of GPU memory at a 64 KiB-aligned
 *        @c base, registered with cuFile when @c registered.
 *
 * Pass @c base as @c cuFileRead's @c bufPtr_base with a zero @c bufPtr_offset:
 * cuFile looks registered memory up by the base it was registered with.
 */
struct staging_slot {
  std::uint8_t* base{nullptr};
  std::size_t bytes{0};
  int device{-1};
  bool registered{false};
  /// What @c cudaMalloc returned (@c base is rounded up inside it).
  void* allocation{nullptr};
  /// The cuFile table that registered it (@c cufile_api::generation()).
  std::uint64_t registered_generation{0};

  [[nodiscard]] explicit operator bool() const noexcept { return base != nullptr; }
};

/**
 * @brief Process-wide free list of device staging slots, per device.
 *
 * Slots are allocated with @c cudaMalloc and registered with @c cuFileBufRegister
 * once, then reused by every reactor and ioctx of the process; they are never
 * freed or deregistered.  @c cudaFree synchronizes the whole device and
 * deregistration tears down the BAR1 mapping, so a reactor that gave its slots
 * back that way at shutdown would wait for other callers' GPU work.  Process exit
 * reclaims the memory.
 *
 * Thread-safe.  Never destroyed.
 */
class staging_pool {
 public:
  [[nodiscard]] static staging_pool& instance();

  /**
   * @brief Lease a slot of exactly @p bytes on @p device.
   *
   * Reuses a free slot when one fits, else allocates one -- unless that would
   * take the device's staging past @p cap_bytes (0 = no cap).  A new slot is
   * registered with cuFile when the library offers registration; when
   * @p require_registered and it cannot be registered, nothing is returned and
   * registration is not tried again on that device (the slot is kept, unregistered,
   * for leases that do not require it).
   *
   * Sets the calling thread's device to @p device for the call.  Never throws.
   *
   * @return The slot, or an empty one.
   */
  [[nodiscard]] staging_slot acquire(int device,
                                     std::size_t bytes,
                                     std::size_t cap_bytes,
                                     bool require_registered) noexcept;

  /// Return a slot leased from @ref acquire.  Its memory must no longer be in use
  /// by any read or copy.
  void release(staging_slot slot) noexcept;

  /// What @p device holds now.
  [[nodiscard]] staging_usage usage(int device) const noexcept;

  /// Deregister and free every slot of @p device no reactor holds; returns the
  /// bytes freed.  Synchronizes the device (@c cudaFree).
  std::size_t release_free(int device) noexcept;

 private:
  staging_pool() = default;

  struct device_state {
    std::vector<staging_slot> free;
    std::size_t allocated_bytes{0};
    std::size_t registered_bytes{0};
    std::size_t leased{0};
    /// The cuFile table (@c cufile_api::generation()) whose registration failed
    /// here: it is not asked again, but another table (a test's substitute) is.
    std::optional<std::uint64_t> registration_failed_for;
  };

  mutable std::mutex _mutex;
  std::map<int, device_state> _devices;
};

}  // namespace cucascade::io::cufile::detail
