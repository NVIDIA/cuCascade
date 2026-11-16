/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <stdexcept>

namespace cucascade {
namespace test {

/**
 * @brief Grant @p accessor's device read/write access to @p owner's GPU pool.
 *
 * A space without a pool yields pool_peer_access_status::CUDA_ERROR (cudaErrorInvalidValue).
 */
inline memory::pool_peer_access_result grant_pool_access(memory::memory_space const& owner,
                                                         memory::memory_space const& accessor)
{
  cudaMemPool_t const pool = owner.get_memory_resource_of<memory::Tier::GPU>()->pool_handle();
  return memory::grant_pool_peer_access(pool, owner.get_device_id(), accessor.get_device_id());
}

/**
 * @brief Grant each GPU space's pool read/write access to the other space's device. Without the
 * grant the converter may stage through host memory instead of peer DMA.
 *
 * @param first GPU memory space
 * @param second GPU memory space on another device
 * @return The first result that is not granted, otherwise the (granted) reverse result
 * @throws std::invalid_argument if both spaces are on the same device
 */
inline memory::pool_peer_access_result grant_two_way_pool_access(memory::memory_space const& first,
                                                                 memory::memory_space const& second)
{
  // Same-device "peer" access is a caller bug: the grant would test nothing between two GPUs.
  if (first.get_device_id() == second.get_device_id()) {
    CUCASCADE_FAIL("grant_two_way_pool_access needs spaces on different devices",
                   std::invalid_argument);
  }

  memory::pool_peer_access_result const forward = grant_pool_access(first, second);

  // e.g. UNSUPPORTED for a device pair without peer capability: skip the reverse grant.
  if (!forward.granted()) { return forward; }

  return grant_pool_access(second, first);
}

}  // namespace test
}  // namespace cucascade
