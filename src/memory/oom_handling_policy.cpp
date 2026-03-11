/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/memory/error.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>

#include <exception>

namespace cucascade {
namespace memory {

void* throw_on_oom_policy::do_handle_oom([[maybe_unused]] std::size_t bytes,
                                         [[maybe_unused]] rmm::cuda_stream_view stream,
                                         std::exception_ptr eptr,
                                         [[maybe_unused]] RetryFunc retry_function)
{
  std::rethrow_exception(eptr);
}

std::string throw_on_oom_policy::get_policy_name() const noexcept { return "rethrow"; }

void* defragment_on_oom_policy::do_handle_oom(std::size_t bytes,
                                              rmm::cuda_stream_view stream,
                                              std::exception_ptr eptr,
                                              RetryFunc retry_function)
{
  try {
    std::rethrow_exception(eptr);
  } catch (const cucascade_out_of_memory& e) {
    // Only attempt defragmentation for upstream allocation failures with a known pool
    if (e.error_kind != MemoryError::ALLOCATION_FAILED || e.pool_handle == nullptr) { throw; }

    cuuint64_t pool_usage{0};
    cuuint64_t pool_capacity{0};
    cudaMemPoolGetAttribute(e.pool_handle, cudaMemPoolAttrUsedMemCurrent, &pool_usage);
    cudaMemPoolGetAttribute(e.pool_handle, cudaMemPoolAttrReservedMemCurrent, &pool_capacity);
    std::size_t const free_bytes =
      pool_capacity > pool_usage ? static_cast<std::size_t>(pool_capacity - pool_usage) : 0;

    // Two-regime fragmentation heuristic:
    //
    // Small allocations (< 64 MB): require a high ratio AND a meaningful absolute
    // free threshold. Without the absolute floor, even 4x of a 4 KB request (16 KB)
    // would trigger an expensive pool trim that is almost certainly not warranted.
    //
    // Large allocations (>= 64 MB): use a lower ratio (2x) since on a pool that holds
    // tens of GBs, having 2x the requested size free but still failing is already a
    // clear signal of fragmentation. A 4x ratio here would be too conservative —
    // e.g. requesting 20 GB with 40 GB free (2x) would not trigger a trim.
    // Using a fixed multiplier also avoids any overflow risk at this size.
    static constexpr std::size_t large_alloc_threshold = 64UL * 1024 * 1024;
    static constexpr std::size_t min_free_small        = 32UL * 1024 * 1024;
    static constexpr std::size_t ratio_small           = 8;
    static constexpr std::size_t ratio_large           = 2;

    bool const is_fragmented =
      (bytes < large_alloc_threshold)
        ? (free_bytes >= min_free_small && free_bytes >= bytes * ratio_small)
        : (free_bytes >= bytes * ratio_large);
    if (!is_fragmented) { throw; }

    // Release all free pool blocks back to the OS so the retry can get a fresh
    // contiguous region from the driver
    cudaMemPoolTrimTo(e.pool_handle, 0);
    stream.synchronize();

    try {
      return retry_function(bytes, stream);
    } catch (...) {
      // Retry failed — surface the original exception so the caller sees the
      // allocation failure that triggered defragmentation, not a secondary one
      std::rethrow_exception(eptr);
    }
  }
}

std::string defragment_on_oom_policy::get_policy_name() const noexcept
{
  return "defragment_on_oom";
}

std::unique_ptr<oom_handling_policy> make_default_oom_policy()
{
  return std::make_unique<throw_on_oom_policy>();
}

}  // namespace memory
}  // namespace cucascade
