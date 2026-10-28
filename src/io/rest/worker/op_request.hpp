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

#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cucascade::io::rest::detail {

/// Scheduling tier of a cuCascade request on a REST reactor (see @c io_priority): demand reads,
/// writes and flushes are @c high; fs_cache prefetch is @c low.
enum class tier : std::uint8_t { high = 0, low = 1 };

inline constexpr std::size_t n_tiers = 2;

[[nodiscard]] constexpr tier tier_of(io_priority p) noexcept
{
  return p == io_priority::low ? tier::low : tier::high;
}

/**
 * @brief One physical operation of a cuCascade request on a REST reactor.
 *
 * A ranged GET (read), one write slice (a part upload or an assembly into a part buffer) or a
 * commit (flush).  Wraps the backend-neutral @c io_op_request, which owns any staging allocation
 * so its iovecs stay stable across retries and until a device copy drained.
 */
struct rest_io_op_request {
  object_ref object;
  std::unique_ptr<io_op_request> op;
  /// Attempts so far (the first attempt is 0) and the separate budget of 403 retries (a
  /// presigned URL that expired while the operation waited).
  std::size_t attempt{0};
  std::size_t auth_attempt{0};
  /// A device read with no host destination: the worker allocates pinned staging blocks for
  /// @c op->io_rng before the GET starts.
  bool needs_staging{false};
  /// Logical (caller-visible) bytes of the operation, counted in the reactor's queued bytes until
  /// a connection picks it up.
  std::size_t logical_bytes{0};
  tier queue{tier::high};
  /// Writes: the S3 part this slice lands in, where in it, and what keeps its assembly buffer
  /// alive while a device copy fills it.
  unsigned part{0};
  std::uint64_t offset_in_part{0};
  std::shared_ptr<void const> hold;

  [[nodiscard]] io_op_kind kind() const noexcept
  {
    return op == nullptr ? io_op_kind::read : op->kind;
  }

  [[nodiscard]] bool is_device() const noexcept
  {
    return op != nullptr && op->device_copy != nullptr;
  }

  /// H2D copy of the received window into the device destination, on the caller's stream.
  [[nodiscard]] cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) const noexcept
  {
    if (!is_device()) return cudaSuccess;
    return op->device_copy->copy_async(op->io_rng, op->iovecs, event);
  }

  /// D2H copy of a device write source into @c op->iovecs (host staging), on the caller's stream.
  [[nodiscard]] cudaError_t copy_d2h_async(cudaEvent_t event = nullptr) const noexcept
  {
    if (!is_device()) return cudaSuccess;
    return op->device_copy->copy_to_host_async(op->io_rng, op->iovecs, event);
  }
};

}  // namespace cucascade::io::rest::detail
