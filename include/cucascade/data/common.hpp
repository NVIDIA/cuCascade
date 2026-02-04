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

#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>
#include <memory>

namespace cucascade {

/**
 * @brief Interface representing a data representation residing in a specific memory tier.
 *
 * The primary purpose is to allow to physically store data in different memory tiers differently
 * (allowing us to optimize the storage format to the tier) while providing a common representation
 * to the rest of the system to interact with.
 *
 * See representation_converter.hpp for utilities to convert between different underlying
 * representations.
 */
class idata_representation {
 public:
  /**
   * @brief Construct a new idata_representation object
   *
   * @param memory_space The memory space where the data resides
   */
  idata_representation(cucascade::memory::memory_space& memory_space) : _memory_space(memory_space)
  {
  }

  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes
   */
  virtual ~idata_representation() = default;

  /**
   * @brief Get the tier of memory that this representation resides in
   *
   * @return Tier The memory tier
   */
  memory::Tier get_current_tier() const { return _memory_space.get_tier(); }

  /**
   * @brief Get the device ID where the data resides
   *
   * @return device_id The device ID
   */
  int get_device_id() const { return _memory_space.get_device_id(); }

  /**
   * @brief Get the memory space where the data resides
   *
   * @return memory_space& Reference to the memory space
   */
  cucascade::memory::memory_space& get_memory_space() { return _memory_space; }

  /**
   * @brief Get the memory space where the data resides (const version)
   *
   * @return const memory_space& Const reference to the memory space
   */
  const cucascade::memory::memory_space& get_memory_space() const { return _memory_space; }

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  virtual std::size_t get_size_in_bytes() const = 0;

  /**
   * @brief Create a deep copy of this data representation.
   *
   * The cloned representation will have its own copy of the underlying data,
   * residing in the same memory space as the original.
   *
   * @return std::unique_ptr<idata_representation> A new data representation with copied data
   */
  virtual std::unique_ptr<idata_representation> clone() = 0;

  /**
   * @brief Safely casts this interface to a specific derived type
   *
   * @tparam TargetType The target type to cast to
   * @return TargetType& Reference to the casted object
   */
  template <class TargetType>
  TargetType& cast()
  {
    return reinterpret_cast<TargetType&>(*this);
  }

  /**
   * @brief Safely casts this interface to a specific derived type (const version)
   *
   * @tparam TargetType The target type to cast to
   * @return const TargetType& Const reference to the casted object
   */
  template <class TargetType>
  const TargetType& cast() const
  {
    return reinterpret_cast<const TargetType&>(*this);
  }

 private:
  cucascade::memory::memory_space& _memory_space;  ///< The memory space where the data resides
};

}  // namespace cucascade
