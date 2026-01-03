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

#include "memory/common.hpp"

namespace cucascade {
namespace memory {

/**
 * Configuration for a single memory_space.
 * Contains all parameters needed to create a memory_space instance.
 */
struct memory_space_config {
  Tier tier;
  int device_id;
  size_t memory_limit;
  float downgrade_tigger_threshold{0.75};
  float downgrade_stop_threshold{0.65f};
  std::size_t memory_capacity;  // Optional total capacity, defaults to device capacity
  DeviceMemoryResourceFactoryFn mr_factory_fn;

  // Constructor - allocators must be explicitly provided
  memory_space_config(Tier t,
                      int dev_id,
                      size_t mem_limit,
                      DeviceMemoryResourceFactoryFn mr_fn = nullptr)
    : memory_space_config(t, dev_id, mem_limit, mem_limit, std::move(mr_fn))
  {
  }

  // Constructor - allocators must be explicitly provided
  memory_space_config(Tier t,
                      int dev_id,
                      size_t mem_limit,
                      size_t mem_capacity,
                      DeviceMemoryResourceFactoryFn mr_fn = nullptr)
    : tier(t),
      device_id(dev_id),
      memory_limit(mem_limit),
      memory_capacity(mem_capacity),
      mr_factory_fn(std::move(mr_fn))
  {
    assert(memory_limit <= memory_capacity && "Memory limit cannot exceed device capacity");
    if (mr_factory_fn == nullptr) { mr_factory_fn = make_default_allocator_for_tier(t); }
  }
};

}  // namespace memory
}  // namespace cucascade
