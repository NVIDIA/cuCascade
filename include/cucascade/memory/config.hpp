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

#pragma once

#include <cucascade/memory/common.hpp>

#include <variant>

namespace cucascade {
namespace memory {

static constexpr std::size_t default_block_size = 1 << 20;  ///< Default block size (1MB)
static constexpr std::size_t default_pool_size  = 128;      ///< Default number of blocks in pool
static constexpr std::size_t default_initial_number_pools =
  4;  ///< Default number of pools to pre-allocate

struct gpu_memory_space_config {
  int device_id{-1};
  double reservation_limit_fraction{0.9};
  double downgrade_trigger_fraction{0.75};
  double downgrade_stop_fraction{0.65};
  std::size_t memory_capacity{0};
  bool per_stream_reservation{true};
  DeviceMemoryResourceFactoryFn mr_factory_fn{nullptr};

  [[nodiscard]] Tier tier() const { return Tier::GPU; }

  [[nodiscard]] std::size_t reservation_limit() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) *
                                    reservation_limit_fraction);
  }

  [[nodiscard]] std::size_t downgrade_trigger_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) *
                                    downgrade_trigger_fraction);
  }

  [[nodiscard]] std::size_t downgrade_stop_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) * downgrade_stop_fraction);
  }
};

struct host_memory_space_config {
  int numa_id{-1};
  double reservation_limit_fraction{0.9};
  double downgrade_trigger_fraction{0.75};
  double downgrade_stop_fraction{0.65};
  std::size_t memory_capacity{0};
  std::size_t block_size           = default_block_size;
  std::size_t pool_size            = default_pool_size;
  std::size_t initial_number_pools = default_initial_number_pools;

  DeviceMemoryResourceFactoryFn mr_factory_fn{nullptr};

  [[nodiscard]] Tier tier() const { return Tier::HOST; }

  [[nodiscard]] std::size_t reservation_limit() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) *
                                    reservation_limit_fraction);
  }

  [[nodiscard]] std::size_t downgrade_trigger_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) *
                                    downgrade_trigger_fraction);
  }

  [[nodiscard]] std::size_t downgrade_stop_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) * downgrade_stop_fraction);
  }
};

struct disk_memory_space_config {
  int disk_id{-1};
  std::size_t memory_capacity{0};
  std::string mount_paths;

  [[nodiscard]] Tier tier() const { return Tier::DISK; }

  [[nodiscard]] std::size_t reservation_limit() const { return memory_capacity; }

  [[nodiscard]] std::size_t downgrade_trigger_threshold() const { return memory_capacity; }

  [[nodiscard]] std::size_t downgrade_stop_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(memory_capacity) * 0.99);
  }
};

/**
 * Configuration for a single memory_space.
 * Contains all parameters needed to create a memory_space instance.
 */
using memory_space_config = std::variant<std::monostate,
                                         gpu_memory_space_config,
                                         host_memory_space_config,
                                         disk_memory_space_config>;

}  // namespace memory
}  // namespace cucascade
