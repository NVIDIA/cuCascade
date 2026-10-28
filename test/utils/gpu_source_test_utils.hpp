/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file gpu_source_test_utils.hpp
 * @brief GPU sources that the buffer-serializing converters (GPU -> host_data / DISK / REMOTE)
 *        must handle: view-backed slices of nested nullable tables, and tables whose writer is a
 *        non-blocking stream on another device that finishes late.
 */

#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade {
namespace test {

/// Host mask with bit i set unless @p is_null(i).
template <class IsNull>
std::vector<cudf::bitmask_type> host_validity(cudf::size_type rows,
                                              IsNull&& is_null,
                                              cudf::size_type& null_count)
{
  std::vector<cudf::bitmask_type> mask((static_cast<std::size_t>(rows) + 31) / 32, 0);
  null_count = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (is_null(i)) {
      ++null_count;
    } else {
      mask[static_cast<std::size_t>(i) / 32] |= cudf::bitmask_type{1} << (i % 32);
    }
  }
  return mask;
}

template <typename T>
std::unique_ptr<cudf::column> device_column(cudf::type_id type,
                                            std::vector<T> const& values,
                                            ::cuda::stream_ref stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{type},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  if (!values.empty()) {
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                       values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream.get()));
  }
  stream.sync();
  return col;
}

/**
 * @brief A nullable nested table of @p rows rows on the current device: INT64 (every 3rd row
 *        null), STRING (every 5th null, every 7th empty), LIST<INT32> (row i has i % 4 elements;
 *        empty lists at multiples of 8 are null) and STRUCT<INT32, FLOAT64> (every 4th null).
 */
inline std::unique_ptr<cudf::table> make_nested_nullable_table(cudf::size_type rows,
                                                               ::cuda::stream_ref stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::size_type nulls = 0;

  std::vector<std::int64_t> ints(static_cast<std::size_t>(rows));
  for (cudf::size_type i = 0; i < rows; ++i) {
    ints[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(i) * 1000003 - 17;
  }
  auto int_col = device_column(cudf::type_id::INT64, ints, stream);
  auto mask    = host_validity(rows, [](auto i) { return i % 3 == 0; }, nulls);
  int_col->set_null_mask(make_null_mask_from_host(rows, mask, stream), nulls);
  cols.push_back(std::move(int_col));

  std::vector<std::int32_t> offsets{0};
  std::vector<char> chars;
  for (cudf::size_type i = 0; i < rows; ++i) {
    auto const s = i % 7 == 0 ? std::string{} : "row-" + std::to_string(i * 31);
    chars.insert(chars.end(), s.begin(), s.end());
    offsets.push_back(static_cast<std::int32_t>(chars.size()));
  }
  rmm::device_buffer dev_chars(chars.data(), chars.size(), stream);
  mask = host_validity(rows, [](auto i) { return i % 5 == 0; }, nulls);
  cols.push_back(cudf::make_strings_column(rows,
                                           device_column(cudf::type_id::INT32, offsets, stream),
                                           std::move(dev_chars),
                                           nulls,
                                           make_null_mask_from_host(rows, mask, stream)));

  std::vector<std::int32_t> list_offsets{0};
  for (cudf::size_type i = 0; i < rows; ++i) {
    list_offsets.push_back(list_offsets.back() + i % 4);
  }
  std::vector<std::int32_t> list_values(static_cast<std::size_t>(list_offsets.back()));
  for (std::size_t i = 0; i < list_values.size(); ++i) {
    list_values[i] = static_cast<std::int32_t>(i * 7 + 1);
  }
  mask = host_validity(rows, [](auto i) { return i % 8 == 0; }, nulls);  // i % 4 == 0: empty
  cols.push_back(cudf::make_lists_column(rows,
                                         device_column(cudf::type_id::INT32, list_offsets, stream),
                                         device_column(cudf::type_id::INT32, list_values, stream),
                                         nulls,
                                         make_null_mask_from_host(rows, mask, stream)));

  std::vector<std::int32_t> field_a(static_cast<std::size_t>(rows));
  std::vector<double> field_b(static_cast<std::size_t>(rows));
  for (cudf::size_type i = 0; i < rows; ++i) {
    field_a[static_cast<std::size_t>(i)] = i * 3;
    field_b[static_cast<std::size_t>(i)] = static_cast<double>(i) / 4.0;
  }
  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(device_column(cudf::type_id::INT32, field_a, stream));
  fields.push_back(device_column(cudf::type_id::FLOAT64, field_b, stream));
  mask = host_validity(rows, [](auto i) { return i % 4 == 0; }, nulls);
  cols.push_back(cudf::make_structs_column(
    rows, std::move(fields), nulls, make_null_mask_from_host(rows, mask, stream), stream));
  stream.sync();
  return std::make_unique<cudf::table>(std::move(cols));
}

/// A view-backed source over rows [begin, end) of a nested nullable table, and the compacted
/// copy of the same rows to compare a round trip against.
struct sliced_source {
  std::unique_ptr<gpu_table_representation> representation;
  std::unique_ptr<cudf::table> expected;
};

inline sliced_source make_sliced_source(memory::memory_space& gpu_space,
                                        ::cuda::stream_ref stream,
                                        cudf::size_type rows  = 3000,
                                        cudf::size_type begin = 37,
                                        cudf::size_type end   = 2011)
{
  std::shared_ptr<cudf::table> full = make_nested_nullable_table(rows, stream);
  auto const slice                  = cudf::slice(full->view(), {begin, end}).front();
  sliced_source out;
  out.expected = std::make_unique<cudf::table>(slice, stream);
  stream.sync();
  auto const bytes = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, stream);
  return out;
}

/**
 * @brief Rows [begin, end) of a fixed-width table on the current device: a nullable INT64 column
 *        (every third row valid) and an INT32 column. No column has children, so only the offset
 *        and the null mask's start reveal the slice.
 */
inline sliced_source make_fixed_width_sliced_source(memory::memory_space& gpu_space,
                                                    ::cuda::stream_ref stream,
                                                    cudf::size_type rows  = 3000,
                                                    cudf::size_type begin = 37,
                                                    cudf::size_type end   = 2011)
{
  std::vector<std::int64_t> wide(static_cast<std::size_t>(rows));
  std::vector<std::int32_t> narrow(static_cast<std::size_t>(rows));
  for (cudf::size_type i = 0; i < rows; ++i) {
    wide[static_cast<std::size_t>(i)]   = static_cast<std::int64_t>(i) * 7919 + 3;
    narrow[static_cast<std::size_t>(i)] = i * 13 - 5;
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::size_type nulls = 0;
  auto wide_col         = device_column(cudf::type_id::INT64, wide, stream);
  auto const mask       = host_validity(rows, [](auto i) { return i % 3 != 0; }, nulls);
  wide_col->set_null_mask(make_null_mask_from_host(rows, mask, stream), nulls);
  cols.push_back(std::move(wide_col));
  cols.push_back(device_column(cudf::type_id::INT32, narrow, stream));
  std::shared_ptr<cudf::table> full = std::make_unique<cudf::table>(std::move(cols));
  auto const slice                  = cudf::slice(full->view(), {begin, end}).front();
  sliced_source out;
  out.expected = std::make_unique<cudf::table>(slice, stream);
  stream.sync();
  auto const bytes = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, stream);
  return out;
}

/**
 * @brief A one-column INT64 table on @p device whose final bytes are written late by a
 *        non-blocking producer stream: the stream first sleeps @p delay (a host function), then
 *        fills the column with 0x5A. The representation's writer is the producer stream, so only
 *        a reader that waits for the writer sees the 0x5A bytes.
 */
struct late_written_source {
  std::unique_ptr<gpu_table_representation> representation;
  cudaStream_t producer{nullptr};
  std::size_t bytes{0};

  late_written_source() = default;
  // The moved-from object must not destroy the producer stream the moved-to one still uses.
  late_written_source(late_written_source&& other) noexcept
    : representation(std::move(other.representation)),
      producer(std::exchange(other.producer, nullptr)),
      bytes(std::exchange(other.bytes, 0))
  {
  }
  late_written_source& operator=(late_written_source&& other) noexcept
  {
    if (this != &other) {
      release();
      representation = std::move(other.representation);
      producer       = std::exchange(other.producer, nullptr);
      bytes          = std::exchange(other.bytes, 0);
    }
    return *this;
  }
  late_written_source(late_written_source const&)            = delete;
  late_written_source& operator=(late_written_source const&) = delete;
  ~late_written_source() { release(); }

 private:
  void release() noexcept
  {
    representation.reset();
    if (producer != nullptr) {
      cudaStreamSynchronize(producer);
      cudaStreamDestroy(producer);
      producer = nullptr;
    }
  }
};

inline void CUDART_CB sleep_on_stream(void* delay_ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds{reinterpret_cast<std::intptr_t>(delay_ms)});
}

/// @param slice_begin When non-zero, the representation is a view-backed slice of rows
///        [slice_begin, rows) of the late-written table (so converters compact it first).
inline late_written_source make_late_written_source(memory::memory_space& gpu_space,
                                                    std::size_t rows,
                                                    std::chrono::milliseconds delay,
                                                    cudf::size_type slice_begin = 0)
{
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_space.get_device_id()}};
  late_written_source out;
  CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&out.producer, cudaStreamNonBlocking));
  ::cuda::stream_ref producer{out.producer};
  auto col  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       static_cast<cudf::size_type>(rows),
                                       cudf::mask_state::UNALLOCATED,
                                       producer,
                                       gpu_space.get_default_allocator());
  out.bytes = rows * sizeof(std::int64_t);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(col->mutable_view().head(), 0, out.bytes, out.producer));
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(out.producer));
  CUCASCADE_CUDA_TRY(
    cudaLaunchHostFunc(out.producer,
                       sleep_on_stream,
                       reinterpret_cast<void*>(static_cast<std::intptr_t>(delay.count()))));
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(col->mutable_view().head(), 0x5A, out.bytes, out.producer));
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  if (slice_begin == 0) {
    out.representation = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(cols)), gpu_space, producer);
    return out;
  }
  std::shared_ptr<cudf::table> full = std::make_unique<cudf::table>(std::move(cols));
  auto const slice =
    cudf::slice(full->view(), {slice_begin, static_cast<cudf::size_type>(rows)}).front();
  auto const bytes = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, producer);
  return out;
}

/**
 * @brief A table on @p gpu_space's device with a 2 MiB INT32 column and a 4 MiB INT64 column,
 *        whose writer is a non-blocking producer stream that sleeps @p delay and then writes
 *        column 0. A copy that allocates through a resource refusing the 4 MiB column (e.g. a
 *        3 MiB limit) fails after column 0's copy was enqueued behind the writer.
 *
 * The buffers come from the device's current resource, not from @p gpu_space's allocator, so the
 * table can live in a refusing space.
 *
 * @param slice_begin When non-zero, the representation is a view-backed slice of rows
 *        [slice_begin, rows) (so converters compact it first); otherwise it owns the table.
 */
inline late_written_source make_late_written_pair_source(memory::memory_space& gpu_space,
                                                         std::chrono::milliseconds delay,
                                                         cudf::size_type slice_begin = 0)
{
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_space.get_device_id()}};
  late_written_source out;
  CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&out.producer, cudaStreamNonBlocking));
  ::cuda::stream_ref producer{out.producer};
  constexpr cudf::size_type rows = 1 << 19;
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto type : {cudf::type_id::INT32, cudf::type_id::INT64}) {
    cols.push_back(cudf::make_numeric_column(
      cudf::data_type{type}, rows, cudf::mask_state::UNALLOCATED, producer));
    auto const bytes = static_cast<std::size_t>(rows) * cudf::size_of(cudf::data_type{type});
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(cols.back()->mutable_view().head(), 0, bytes, out.producer));
    out.bytes += bytes;
  }
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(out.producer));
  CUCASCADE_CUDA_TRY(
    cudaLaunchHostFunc(out.producer,
                       sleep_on_stream,
                       reinterpret_cast<void*>(static_cast<std::intptr_t>(delay.count()))));
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(cols.front()->mutable_view().head(),
                                     0x5A,
                                     static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                                     out.producer));
  if (slice_begin == 0) {
    out.representation = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(cols)), gpu_space, producer);
    return out;
  }
  std::shared_ptr<cudf::table> full = std::make_unique<cudf::table>(std::move(cols));
  auto const slice                  = cudf::slice(full->view(), {slice_begin, rows}).front();
  auto const bytes                  = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, producer);
  return out;
}

/// Every byte of @p table's single column (on @p device) is 0x5A.
inline bool all_bytes_are_5a(cudf::table_view const& table, int device)
{
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device}};
  CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  auto const col = table.column(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(col.size()) * sizeof(std::int64_t));
  CUCASCADE_CUDA_TRY(cudaMemcpy(bytes.data(), col.head(), bytes.size(), cudaMemcpyDefault));
  for (auto b : bytes) {
    if (b != 0x5A) { return false; }
  }
  return true;
}

/**
 * @brief A prefix slice [0, 10) of a 100-row table whose children keep the full extent: a
 *        DICTIONARY32 (its indices child has 100 rows), a STRING and a LIST (101 offsets) and a
 *        STRUCT (100-row fields). The slice has offset 0, so only the child extents reveal it.
 */
inline sliced_source make_prefix_sliced_source(memory::memory_space& gpu_space,
                                               ::cuda::stream_ref stream)
{
  constexpr cudf::size_type rows = 100;
  auto nested                    = make_nested_nullable_table(rows, stream);
  auto columns                   = nested->release();
  std::vector<std::int32_t> keys{10, 20, 30, 40, 50};
  std::vector<std::int32_t> indices(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    indices[static_cast<std::size_t>(i)] = (i * 3) % 5;
  }
  columns.push_back(
    cudf::make_dictionary_column(device_column(cudf::type_id::INT32, keys, stream),
                                 device_column(cudf::type_id::INT32, indices, stream)));
  std::shared_ptr<cudf::table> full = std::make_unique<cudf::table>(std::move(columns));
  auto const slice                  = cudf::slice(full->view(), {0, 10}).front();
  sliced_source out;
  out.expected = std::make_unique<cudf::table>(slice, stream);
  stream.sync();
  auto const bytes = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, stream);
  return out;
}

/**
 * @brief A device memory resource that refuses every allocation larger than a limit (and
 *        forwards the others to @p upstream, by default the current device's resource): a seam
 *        to make a converter's compaction or copy fail after it enqueued its first column copies.
 */
class refusing_large_allocations {
 public:
  explicit refusing_large_allocations(
    std::size_t limit,
    rmm::device_async_resource_ref upstream = rmm::mr::get_current_device_resource_ref())
    : _upstream(upstream), _limit(limit)
  {
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    if (bytes > _limit) { throw rmm::out_of_memory("test: allocation refused"); }
    return _upstream.allocate(stream, bytes, alignment);
  }
  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    _upstream.deallocate(stream, p, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }
  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }
  bool operator==(refusing_large_allocations const& other) const noexcept { return this == &other; }
  friend void get_property(refusing_large_allocations const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  rmm::device_async_resource_ref _upstream;
  std::size_t _limit;
};

/// A GPU memory space on @p device whose allocator refuses allocations above @p limit bytes and
/// forwards the others to that device's resource.
inline std::shared_ptr<memory::memory_space> make_refusing_gpu_space(std::size_t limit,
                                                                     int device = 0)
{
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device}};
  memory::gpu_memory_space_config config;
  config.device_id       = device;
  config.memory_capacity = std::size_t{1} << 30;
  config.mr_factory_fn   = [limit](int device_id, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{refusing_large_allocations{
      limit, rmm::mr::get_per_device_resource_ref(rmm::cuda_device_id{device_id})}};
  };
  return std::make_shared<memory::memory_space>(config);
}

/**
 * @brief A view-backed slice [1, 512 Ki) of a table with a 2 MiB INT32 column and a 4 MiB INT64
 *        column, on @p gpu_space (whose allocator should refuse the INT64 copy, e.g. with a 3 MiB
 *        limit), plus a non-blocking stream that sleeps @p delay before anything enqueued on it
 *        later runs.
 */
struct compaction_failure_source {
  std::unique_ptr<gpu_table_representation> representation;
  cudaStream_t stream{nullptr};

  compaction_failure_source() = default;
  // The moved-from object must not destroy the stream the moved-to one still uses.
  compaction_failure_source(compaction_failure_source&& other) noexcept
    : representation(std::move(other.representation)), stream(std::exchange(other.stream, nullptr))
  {
  }
  compaction_failure_source& operator=(compaction_failure_source&& other) noexcept
  {
    if (this != &other) {
      release();
      representation = std::move(other.representation);
      stream         = std::exchange(other.stream, nullptr);
    }
    return *this;
  }
  compaction_failure_source(compaction_failure_source const&)            = delete;
  compaction_failure_source& operator=(compaction_failure_source const&) = delete;
  ~compaction_failure_source() { release(); }

 private:
  void release() noexcept
  {
    representation.reset();  // its buffers are freed on `stream`
    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

inline compaction_failure_source make_compaction_failure_source(memory::memory_space& gpu_space,
                                                                std::chrono::milliseconds delay)
{
  compaction_failure_source out;
  CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&out.stream, cudaStreamNonBlocking));
  ::cuda::stream_ref stream{out.stream};
  std::vector<std::unique_ptr<cudf::column>> cols;
  constexpr cudf::size_type rows = 1 << 19;
  cols.push_back(device_column(cudf::type_id::INT32, std::vector<std::int32_t>(rows, 7), stream));
  cols.push_back(device_column(cudf::type_id::INT64, std::vector<std::int64_t>(rows, 9), stream));
  std::shared_ptr<cudf::table> full = std::make_unique<cudf::table>(std::move(cols));
  auto const slice                  = cudf::slice(full->view(), {1, rows}).front();
  auto const bytes                  = full->alloc_size();
  out.representation =
    std::make_unique<gpu_table_representation>(slice, std::move(full), bytes, gpu_space, stream);
  CUCASCADE_CUDA_TRY(
    cudaLaunchHostFunc(out.stream,
                       sleep_on_stream,
                       reinterpret_cast<void*>(static_cast<std::intptr_t>(delay.count()))));
  return out;
}

}  // namespace test
}  // namespace cucascade
