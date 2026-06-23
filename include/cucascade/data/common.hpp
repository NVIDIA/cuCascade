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

#include <cuda_runtime.h>

#include <concepts>
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
   * @brief Get the logical (uncompressed) data size in bytes
   *
   * For representations that store data in a compressed format (e.g. Parquet), this returns
   * the uncompressed data size. For uncompressed representations, this returns the same value
   * as get_size_in_bytes().
   *
   * @return std::size_t The logical data size in bytes
   */
  virtual std::size_t get_uncompressed_data_size_in_bytes() const = 0;

  /**
   * @brief Create a deep copy of this data representation.
   *
   * The cloned representation will have its own copy of the underlying data,
   * residing in the same memory space as the original.
   *
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<idata_representation> A new data representation with copied data
   */
  virtual std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Record a CUDA event marking completion of the most recent writes to this
   *        representation's memory, for cross-stream / cross-device synchronization.
   *
   * The default implementation is a no-op. Tier-specific representations whose memory is
   * produced asynchronously on a CUDA stream (e.g. GPU representations) override this to
   * record an event on @p writer_stream.
   *
   * @param writer_stream The stream on which the most recent writes were enqueued.
   */
  virtual void record_writer_event([[maybe_unused]] rmm::cuda_stream_view writer_stream) {}

  /**
   * @brief Get the writer event recorded by record_writer_event(), or nullptr if none.
   *
   * Readers that cross stream / device boundaries should call cudaStreamWaitEvent on the
   * returned event (when non-null) before reading this representation's memory. The default
   * implementation returns nullptr; representations that record writer events override this.
   *
   * @return cudaEvent_t The writer event, or nullptr if none has been recorded.
   */
  [[nodiscard]] virtual cudaEvent_t get_writer_event() const { return nullptr; }

  /**
   * @brief Rebind the representation's device buffers to use @p stream for future deallocation.
   *
   * The default implementation is a no-op. Representations that own stream-ordered device memory
   * (e.g. GPU representations) override this so their buffers free on the active stream rather
   * than the stream they were produced on, keeping RMM's per-stream free lists correct.
   *
   * @note Does NOT insert cross-stream ordering -- callers must enforce ordering separately.
   *
   * @param stream Stream used for future asynchronous deallocation of the data's buffers.
   */
  virtual void rebind_stream([[maybe_unused]] rmm::cuda_stream_view stream) {}

  /**
   * @brief Casts this interface to a specific derived type.
   *
   * @tparam TargetType The target derived type to cast to. Must be a subclass of
   *         `idata_representation` — enforced at compile time.
   * @return TargetType& Reference to the derived object.
   * @throws std::bad_cast if the runtime type of this object is not `TargetType`.
   */
  template <class TargetType>
    requires std::derived_from<TargetType, idata_representation>
  TargetType& cast()
  {
    return dynamic_cast<TargetType&>(*this);
  }

  /**
   * @brief Casts this interface to a specific derived type (const version).
   *
   * @tparam TargetType The target derived type to cast to. Must be a subclass of
   *         `idata_representation` — enforced at compile time.
   * @return const TargetType& Const reference to the derived object.
   * @throws std::bad_cast if the runtime type of this object is not `TargetType`.
   */
  template <class TargetType>
    requires std::derived_from<TargetType, idata_representation>
  const TargetType& cast() const
  {
    return dynamic_cast<const TargetType&>(*this);
  }

 private:
  cucascade::memory::memory_space& _memory_space;  ///< The memory space where the data resides
};

}  // namespace cucascade
