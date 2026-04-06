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

#include <cucascade/data/common.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>

#include <cudf/table/table.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace cucascade {
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace cucascade {

/**
 * @brief Synchronized wrapper for data_batch that separates data from synchronization.
 *
 * synchronized_data_batch owns an inner data_batch (pure data, no locking) and a
 * std::shared_mutex. Access to the inner data_batch is only possible through RAII accessor
 * types (read_only_data_batch, mutable_data_batch) that bundle lock lifetime with data access.
 *
 * Key characteristics:
 * - Compiler-enforced read/write separation via accessor types
 * - PtrType agnostic — works with shared_ptr, unique_ptr, or stack allocation
 * - No enable_shared_from_this — accessors borrow via raw pointer
 * - Independent atomic subscriber count for interest tracking
 * - Unique batch ID exposed on wrapper (lock-free) for repository lookups
 */
class synchronized_data_batch {
 public:
  // Forward declarations for accessor types
  class read_only_data_batch;
  class mutable_data_batch;

  /**
   * @brief Inner data type — pure data, no synchronization.
   *
   * Only accessible through read_only_data_batch (const) or mutable_data_batch (non-const).
   * Constructor is private — only synchronized_data_batch can create instances.
   */
  class data_batch {
   public:
    uint64_t get_batch_id() const;
    memory::Tier get_current_tier() const;
    idata_representation* get_data() const;
    cucascade::memory::memory_space* get_memory_space() const;

    void set_data(std::unique_ptr<idata_representation> data);

    template <typename TargetRepresentation>
    void convert_to(representation_converter_registry& registry,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream);

   private:
    friend class synchronized_data_batch;
    data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

    uint64_t _batch_id;
    std::unique_ptr<idata_representation> _data;
  };

  /**
   * @brief RAII read-only accessor. Borrows the parent, does not extend lifetime.
   *
   * operator-> returns const data_batch* — mutating methods are not callable.
   * Multiple read_only_data_batch instances can coexist for the same synchronized_data_batch.
   * Move-only. The shared lock is released when this object is destroyed or moved-from.
   */
  class read_only_data_batch {
   public:
    const data_batch* operator->() const { return &(parent_->_batch); }
    const data_batch& operator*() const { return parent_->_batch; }

    /**
     * @brief Downgrade from mutable → read-only.
     * Internally releases unique lock, then blocks until shared lock acquired.
     */
    static read_only_data_batch from_mutable(mutable_data_batch&& rw);

    read_only_data_batch(read_only_data_batch&&) noexcept            = default;
    read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;
    read_only_data_batch(const read_only_data_batch&)                = delete;
    read_only_data_batch& operator=(const read_only_data_batch&)     = delete;

   private:
    friend class synchronized_data_batch;
    friend class mutable_data_batch;
    read_only_data_batch(synchronized_data_batch* parent,
                         std::shared_lock<std::shared_mutex> lock)
      : parent_(parent), lock_(std::move(lock))
    {
    }

    synchronized_data_batch* parent_;
    std::shared_lock<std::shared_mutex> lock_;
  };

  /**
   * @brief RAII mutable accessor. Borrows the parent, does not extend lifetime.
   *
   * operator-> returns data_batch* — both read and write methods are callable.
   * Only one mutable_data_batch can exist at a time for a given synchronized_data_batch.
   * Move-only. The unique lock is released when this object is destroyed or moved-from.
   */
  class mutable_data_batch {
   public:
    data_batch* operator->() { return &(parent_->_batch); }
    data_batch& operator*() { return parent_->_batch; }

    /**
     * @brief Upgrade from read-only → mutable.
     * Internally releases shared lock, then blocks until unique lock acquired.
     */
    static mutable_data_batch from_read_only(read_only_data_batch&& ro);

    mutable_data_batch(mutable_data_batch&&) noexcept            = default;
    mutable_data_batch& operator=(mutable_data_batch&&) noexcept = default;
    mutable_data_batch(const mutable_data_batch&)                = delete;
    mutable_data_batch& operator=(const mutable_data_batch&)     = delete;

   private:
    friend class synchronized_data_batch;
    friend class read_only_data_batch;
    mutable_data_batch(synchronized_data_batch* parent,
                       std::unique_lock<std::shared_mutex> lock)
      : parent_(parent), lock_(std::move(lock))
    {
    }

    synchronized_data_batch* parent_;
    std::unique_lock<std::shared_mutex> lock_;
  };

  // -- Construction --

  synchronized_data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  synchronized_data_batch(synchronized_data_batch&& other);
  synchronized_data_batch& operator=(synchronized_data_batch&& other);

  synchronized_data_batch(const synchronized_data_batch&)            = delete;
  synchronized_data_batch& operator=(const synchronized_data_batch&) = delete;

  // -- Lock acquisition (blocking) --

  read_only_data_batch get_read_only();
  mutable_data_batch get_mutable();

  // -- Lock acquisition (non-blocking) --

  std::optional<read_only_data_batch> try_get_read_only();
  std::optional<mutable_data_batch> try_get_mutable();

  // -- Immutable field exposed on wrapper (lock-free, for repository lookups) --

  uint64_t get_batch_id() const;

  // -- Subscriber count (independent atomic, no lock needed) --

  bool subscribe();
  void unsubscribe();
  size_t get_subscriber_count() const;

  // -- Clone (acquires read lock internally) --

  std::shared_ptr<synchronized_data_batch> clone(uint64_t new_batch_id,
                                                 rmm::cuda_stream_view stream);

  template <typename TargetRepresentation>
  std::shared_ptr<synchronized_data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    const cucascade::memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream);

 private:
  data_batch _batch;
  mutable std::shared_mutex _rw_mutex;
  std::atomic<size_t> _subscriber_count{0};
};

// Template implementations

template <typename TargetRepresentation>
void synchronized_data_batch::data_batch::convert_to(
  representation_converter_registry& registry,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_data, target_memory_space, stream);
  _data = std::move(new_representation);
}

template <typename TargetRepresentation>
std::shared_ptr<synchronized_data_batch> synchronized_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);

  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch._data, target_memory_space, stream);
  return std::make_shared<synchronized_data_batch>(new_batch_id, std::move(new_representation));
}

}  // namespace cucascade
