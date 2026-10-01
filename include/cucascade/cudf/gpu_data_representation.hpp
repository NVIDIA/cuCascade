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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>

#include <any>
#include <cstddef>
#include <memory>
#include <variant>

namespace cucascade {

/**
 * @brief Data representation for a table being stored in GPU memory.
 *
 * This class currently represents a table just as a cuDF table along with the allocation where the
 * cudf's table data actually resides. The primary purpose for this is that the table can be
 * directly passed to cuDF APIs for processing without any additional copying while the underlying
 * memory is still owned/tracked by our memory allocator.
 *
 * The constructors and record_writer_event() record a writer event (see get_writer_event()) on the
 * stream that last wrote the table. A constructor given a null writer stream handle records none
 * because the writer is unknown, and until record_writer_event() is called, clone(),
 * release_table(), and the GPU-to-GPU converter registered by register_builtin_converters()
 * synchronize the whole device returned by get_device_id() before reading. That synchronization
 * waits for all work on the device, including unrelated work and work that itself waits on the
 * calling thread, so a caller holding up such work deadlocks. Callers should therefore pass the
 * actual writer stream, which avoids both the stall and the hang, and pass the legacy default
 * stream (`cudaStreamLegacy`) only when the writes really were enqueued on it: an event recorded
 * there does not cover work on non-blocking streams, and it makes every later reader wait for all
 * work on the blocking streams of that device.
 *
 * Every stream that a member function records on, waits on, copies on, or binds buffers to must
 * belong to the device returned by get_device_id(), and a null stream handle names the default
 * stream of the calling thread's current device. On a mismatch, the member function throws
 * cucascade::logic_error before it enqueues work or changes state. Member functions that allocate
 * do so on that device, whichever device is current.
 *
 * TODO: Once the GPU memory resource is implemented, replace the allocation type from
 * IAllocatedMemory to the concrete type returned by the GPU memory allocator.
 */
class gpu_table_representation : public idata_representation {
 public:
  /**
   * @brief Construct a new gpu_table_representation object.
   *
   * @pre Every allocation in @p table lives on the device of @p memory_space.
   *
   * @param table Unique pointer to the cuDF table with the data (ownership is transferred)
   * @param memory_space The memory space where the GPU table resides
   * @param writer_stream The stream on which @p table's data was last written, or a null handle if
   * the writer is unknown
   * @throws cucascade::logic_error if @p writer_stream is not null and belongs to a different CUDA
   * device
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space,
                           ::cuda::stream_ref writer_stream);

  /**
   * @brief Construct a new gpu_table_representation object from a cudf::table_view.
   *
   * @pre Every allocation viewed by @p table_view lives on the device of @p memory_space.
   *
   * @param table_view View of the cuDF table (data ownership lives in @p owner)
   * @tparam Owner The type of the owner of the cuDF table (e.g., a specific operator or component)
   * @param owner Owner of the underlying data (transferred via std::any storage)
   * @param alloc_size Allocation size in bytes for the data
   * @param memory_space The memory space where the GPU table resides
   * @param writer_stream The stream on which @p table_view's data was last written, or a null
   * handle if the writer is unknown
   * @throws cucascade::logic_error if @p writer_stream is not null and belongs to a different CUDA
   * device
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  template <typename Owner>
  gpu_table_representation(cudf::table_view table_view,
                           Owner&& owner,
                           std::size_t alloc_size,
                           cucascade::memory::memory_space& memory_space,
                           ::cuda::stream_ref writer_stream);

  /**
   * @brief Destructor — destroys the writer-event if one was recorded.
   *
   * STREAM-LINEAGE: events recorded on a writer stream via record_writer_event() are
   * owned by the representation and released on destruction.
   */
  ~gpu_table_representation() override;

  // Non-copyable / non-movable: the representation owns a CUDA event handle whose
  // lifetime must be unique. Move semantics could be added but are not needed by
  // any in-tree caller.
  gpu_table_representation(const gpu_table_representation&)            = delete;
  gpu_table_representation(gpu_table_representation&&)                 = delete;
  gpu_table_representation& operator=(const gpu_table_representation&) = delete;
  gpu_table_representation& operator=(gpu_table_representation&&)      = delete;

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @copydoc idata_representation::get_logical_data_size_in_bytes
   */
  std::size_t get_uncompressed_data_size_in_bytes() const override;

  /**
   * @brief Create a deep copy of this GPU table representation.
   *
   * The cloned representation will have its own copy of the underlying cuDF table,
   * residing in the same memory space as the original.
   *
   * The copy is enqueued on @p stream after the writer of this representation, and @p stream
   * becomes the writer of the clone, also when it is a null handle. This method does not wait for
   * the copy to complete.
   *
   * @pre This representation and its contents remain alive and unmodified until the work enqueued
   * on @p stream completes.
   *
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<idata_representation> A new gpu_table_representation with copied data
   * @throws cucascade::logic_error if @p stream belongs to a different CUDA device
   * @throws rmm::bad_alloc if allocating the copy fails
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  std::unique_ptr<idata_representation> clone(::cuda::stream_ref stream) override;

  /**
   * @brief Get the underlying cuDF table view
   *
   * @return cudf::table_view A view of the cuDF table
   */
  cudf::table_view get_table_view() const;

  /**
   * @brief Release ownership of the underlying cuDF table
   *
   * After calling this method, this representation no longer owns the table.
   *
   * Work enqueued on @p stream after this call is ordered after the writer of this representation.
   * An owned table is returned without synchronizing @p stream. A view-backed table is first copied
   * on @p stream, and this method then blocks until @p stream has finished all of its work,
   * including work enqueued before this call, because releasing the view destroys its owner and
   * possibly the viewed memory with it.
   *
   * @note Afterwards the representation holds no table. get_table_view(), get_size_in_bytes(),
   * get_uncompressed_data_size_in_bytes(), and clone() must not be called, and neither may the
   * built-in converters, because they would dereference the missing table. A second release_table()
   * returns nullptr, and destroying the representation remains valid.
   *
   * @pre Apart from the writer, no stream other than @p stream may have in-flight work touching the
   * table's device memory, because this method orders nothing else before that memory is bound to
   * @p stream (owned table) or released with its owner (view-backed table).
   *
   * @param stream Stream that will own deallocation ordering of the returned table's buffers
   *               (also used to materialize the table from a view path before release)
   * @return std::unique_ptr<cudf::table> The cuDF table
   * @throws cucascade::logic_error if @p stream belongs to a different CUDA device
   * @throws rmm::bad_alloc if allocating the copy of a view-backed table fails
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  std::unique_ptr<cudf::table> release_table(::cuda::stream_ref stream);

  /**
   * @brief Rebind the owned table's device buffers to use @p stream for future deallocation.
   *
   * Applies cudf::rebind_stream to every column (recursively rebinding data buffers, null
   * masks, and nested children) so that the buffers are freed on @p stream rather than on the
   * stream they were originally allocated on. No device memory is copied and no kernels are
   * launched.
   *
   * No-op when this representation holds an owning_table_view: that alternative references
   * memory owned by an external (type-erased) owner, which is responsible for its own
   * deallocation stream.
   *
   * @note This does NOT insert cross-stream ordering. The caller must ensure a happens-before
   * relationship from any stream with in-flight work touching this table's memory to @p stream
   * before the rebound memory is reused or freed. See cudf::rebind_stream.
   *
   * @p stream is checked only when the rebind takes effect (owned, non-empty table), because the
   * no-op cases above bind nothing.
   *
   * @param stream Stream used for future asynchronous deallocation of the table's buffers.
   * @throws cucascade::logic_error if @p stream belongs to a different CUDA device
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  void rebind_stream(::cuda::stream_ref stream) override;

  /**
   * @brief Record a CUDA event on @p writer_stream and store it as the writer event.
   *
   * Call this after enqueueing new writes to the table so that later readers order themselves after
   * them. The representation owns a single event, created on the device returned by get_device_id()
   * at the first call and re-recorded by later calls. If the event cannot be created or recorded,
   * the representation is left without one, so the writer becomes unknown. Unlike the constructors,
   * this method treats a null @p writer_stream as the default stream rather than as an unknown
   * writer.
   *
   * @param writer_stream The stream on which the most recent writes to this
   *                      representation's memory were enqueued.
   * @throws cucascade::logic_error if @p writer_stream belongs to a different CUDA device
   * @throws cucascade::cuda_error if a CUDA runtime call fails
   */
  void record_writer_event(::cuda::stream_ref writer_stream) override;

  /**
   * @brief Get the writer event, or nullptr if the writer is unknown.
   *
   * Readers handle both results as idata_representation::get_writer_event() describes. A returned
   * event stays valid until this representation is destroyed or a later record_writer_event()
   * fails, because that failure destroys the event and leaves the writer unknown.
   *
   * @return cudaEvent_t The writer event, or nullptr if the writer is unknown
   */
  [[nodiscard]] cudaEvent_t get_writer_event() const override;

 private:
  struct owning_table_view {
    std::any owner;  ///< The owner of the cuDF table
    std::size_t alloc_size{0};
    cudf::table_view view;  ///< A view of the owned table for easy access
  };

  std::variant<std::unique_ptr<cudf::table>, owning_table_view>
    _table;  ///< cudf::table is the underlying representation of the data

  /// Event recorded on the most recent writer stream and created on the device returned by
  /// get_device_id(); null while the writer is unknown.
  cudaEvent_t _writer_event{nullptr};
};

template <typename Owner>
gpu_table_representation::gpu_table_representation(cudf::table_view table_view,
                                                   Owner&& owner,
                                                   std::size_t alloc_size,
                                                   cucascade::memory::memory_space& memory_space,
                                                   ::cuda::stream_ref writer_stream)
  : idata_representation(memory_space),
    _table(
      owning_table_view{std::make_any<Owner>(std::forward<Owner>(owner)), alloc_size, table_view})
{
  // A null handle means the writer is unknown, so there is nothing to record.
  if (writer_stream.get() != nullptr) {
    gpu_table_representation::record_writer_event(writer_stream);
  }
}

}  // namespace cucascade
