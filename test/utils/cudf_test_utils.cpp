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

#include <cucascade/cuda_utils.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

namespace cucascade {
namespace test {

// Forward declarations for helpers defined later in this file
static void install_rmm_logging_resource_once();
static inline bool host_mem_equal(const uint8_t* a, const uint8_t* b, size_t n);

static void dump_hex_context(const uint8_t* data,
                             size_t size,
                             size_t center,
                             size_t context_len = 64)
{
  size_t start = (center > context_len / 2) ? (center - context_len / 2) : 0;
  if (start + context_len > size) { context_len = (size > start) ? (size - start) : 0; }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < context_len; ++i) {
    size_t idx = start + i;
    if (i && (i % 16 == 0)) oss << " | ";
    if (idx < size) { oss << std::setw(2) << static_cast<unsigned int>(data[idx]) << ' '; }
  }
  std::cout << "  hex@" << std::dec << start << " (" << context_len << "B): " << oss.str()
            << std::endl
            << std::flush;
}

/**
 * @brief Copy device data to a host vector.
 */
static std::vector<uint8_t> device_to_host(const void* dev_ptr, std::size_t size)
{
  std::vector<uint8_t> host_data(size);
  if (size > 0 && dev_ptr != nullptr) {
    CUCASCADE_CUDA_TRY(cudaMemcpy(host_data.data(), dev_ptr, size, cudaMemcpyDeviceToHost));
  }
  return host_data;
}

/**
 * @brief Read an offsets column (INT32 or INT64) into a host vector of int64_t values.
 *
 * This normalizes INT32 and INT64 offsets to a common int64_t type so both can be compared
 * regardless of the internal offset width.
 */
static std::vector<int64_t> read_offsets_to_host(const cudf::column_view& col)
{
  auto const n = static_cast<std::size_t>(col.size());
  std::vector<int64_t> result(n);
  if (n == 0) { return result; }

  if (col.type().id() == cudf::type_id::INT64) {
    auto host_data = device_to_host(col.head(), n * sizeof(int64_t));
    std::memcpy(result.data(), host_data.data(), n * sizeof(int64_t));
  } else {
    // INT32 offsets -- widen to INT64
    auto host_data = device_to_host(col.head(), n * sizeof(int32_t));
    std::vector<int32_t> i32(n);
    std::memcpy(i32.data(), host_data.data(), n * sizeof(int32_t));
    for (std::size_t i = 0; i < n; ++i) {
      result[i] = static_cast<int64_t>(i32[i]);
    }
  }
  return result;
}

/**
 * @brief Recursively compare two column_views for logical equality.
 *
 * Handles all column types including compound types (STRING, LIST, STRUCT, DICTIONARY).
 * Allows INT32 vs INT64 offsets to differ (compares values after widening to INT64).
 */
static bool columns_equal_recursive(const cudf::column_view& left,
                                    const cudf::column_view& right,
                                    const std::string& path)
{
  // Type check: must match at the logical level
  if (left.type().id() != right.type().id()) {
    std::cout << "[cudf-equal] " << path
              << " type mismatch: left=" << static_cast<int>(left.type().id())
              << " right=" << static_cast<int>(right.type().id()) << std::endl
              << std::flush;
    return false;
  }

  // Size check
  if (left.size() != right.size()) {
    std::cout << "[cudf-equal] " << path << " size mismatch: left=" << left.size()
              << " right=" << right.size() << std::endl
              << std::flush;
    return false;
  }

  // Null count check
  if (left.null_count() != right.null_count()) {
    std::cout << "[cudf-equal] " << path << " null_count mismatch: left=" << left.null_count()
              << " right=" << right.null_count() << std::endl
              << std::flush;
    return false;
  }

  // Compare null masks (only significant bits, ignoring padding beyond num_rows)
  if (left.nullable() && right.nullable() && left.size() > 0) {
    auto const num_bits       = static_cast<std::size_t>(left.size());
    auto const num_full_bytes = num_bits / 8;
    auto const remaining_bits = num_bits % 8;
    auto const bytes_to_read  = num_full_bytes + (remaining_bits > 0 ? 1 : 0);
    auto const alloc_bytes    = cudf::bitmask_allocation_size_bytes(left.size());
    auto left_mask            = device_to_host(left.null_mask(), alloc_bytes);
    auto right_mask           = device_to_host(right.null_mask(), alloc_bytes);

    // Compare full bytes
    if (num_full_bytes > 0 &&
        !host_mem_equal(left_mask.data(), right_mask.data(), num_full_bytes)) {
      std::cout << "[cudf-equal] " << path << " null mask differs (full bytes)" << std::endl
                << std::flush;
      return false;
    }
    // Compare the last partial byte using a mask for significant bits only
    if (remaining_bits > 0) {
      auto const last_byte_mask =
        static_cast<uint8_t>((1u << remaining_bits) - 1u);  // e.g., 0b00011111 for 5 bits
      if ((left_mask[num_full_bytes] & last_byte_mask) !=
          (right_mask[num_full_bytes] & last_byte_mask)) {
        std::cout << "[cudf-equal] " << path << " null mask differs (last byte)" << std::endl
                  << std::flush;
        return false;
      }
    }
  }

  auto const type_id = left.type().id();

  // STRING: compare chars data + offsets (normalized to INT64)
  if (type_id == cudf::type_id::STRING) {
    // Compare chars data via strings_column_view
    if (left.size() > 0 && left.num_children() > 0) {
      auto left_scv  = cudf::strings_column_view(left);
      auto right_scv = cudf::strings_column_view(right);

      // Compare offsets (child 0), normalizing INT32/INT64
      auto left_offsets  = read_offsets_to_host(left_scv.offsets());
      auto right_offsets = read_offsets_to_host(right_scv.offsets());
      if (left_offsets != right_offsets) {
        std::cout << "[cudf-equal] " << path << " STRING offsets differ" << std::endl << std::flush;
        return false;
      }

      // Compare chars data
      rmm::cuda_stream stream;
      auto left_chars_size  = left_scv.chars_size(stream.view());
      auto right_chars_size = right_scv.chars_size(stream.view());
      if (left_chars_size != right_chars_size) {
        std::cout << "[cudf-equal] " << path
                  << " STRING chars_size mismatch: left=" << left_chars_size
                  << " right=" << right_chars_size << std::endl
                  << std::flush;
        return false;
      }
      if (left_chars_size > 0) {
        auto left_chars  = device_to_host(left.data<uint8_t>(), left_chars_size);
        auto right_chars = device_to_host(right.data<uint8_t>(), right_chars_size);
        if (!host_mem_equal(left_chars.data(), right_chars.data(), left_chars_size)) {
          std::cout << "[cudf-equal] " << path << " STRING chars data differs" << std::endl
                    << std::flush;
          return false;
        }
      }
    }
    return true;
  }

  // LIST: compare offsets (normalized) + recurse into values child
  if (type_id == cudf::type_id::LIST) {
    if (left.num_children() >= 2 && left.size() > 0) {
      auto left_offsets  = read_offsets_to_host(left.child(0));
      auto right_offsets = read_offsets_to_host(right.child(0));
      if (left_offsets != right_offsets) {
        std::cout << "[cudf-equal] " << path << " LIST offsets differ" << std::endl << std::flush;
        return false;
      }
      if (!columns_equal_recursive(left.child(1), right.child(1), path + ".values")) {
        return false;
      }
    }
    return true;
  }

  // STRUCT: recurse into each child field
  if (type_id == cudf::type_id::STRUCT) {
    if (left.num_children() != right.num_children()) {
      std::cout << "[cudf-equal] " << path
                << " STRUCT children count mismatch: left=" << left.num_children()
                << " right=" << right.num_children() << std::endl
                << std::flush;
      return false;
    }
    for (int i = 0; i < left.num_children(); ++i) {
      if (!columns_equal_recursive(
            left.child(i), right.child(i), path + ".field" + std::to_string(i))) {
        return false;
      }
    }
    return true;
  }

  // DICTIONARY: compare keys and indices children
  if (type_id == cudf::type_id::DICTIONARY32) {
    if (left.num_children() != right.num_children()) {
      std::cout << "[cudf-equal] " << path
                << " DICTIONARY children count mismatch: left=" << left.num_children()
                << " right=" << right.num_children() << std::endl
                << std::flush;
      return false;
    }
    for (int i = 0; i < left.num_children(); ++i) {
      if (!columns_equal_recursive(
            left.child(i), right.child(i), path + ".child" + std::to_string(i))) {
        return false;
      }
    }
    return true;
  }

  // Fixed-width types: compare data bytes directly
  if (left.size() > 0 && cudf::is_fixed_width(left.type())) {
    auto data_bytes = static_cast<std::size_t>(left.size()) * cudf::size_of(left.type());
    auto left_data  = device_to_host(left.head(), data_bytes);
    auto right_data = device_to_host(right.head(), data_bytes);
    if (!host_mem_equal(left_data.data(), right_data.data(), data_bytes)) {
      size_t diff_idx = 0;
      for (; diff_idx < data_bytes; ++diff_idx) {
        if (left_data[diff_idx] != right_data[diff_idx]) break;
      }
      std::cout << "[cudf-equal] " << path << " data differs at byte " << diff_idx
                << " left=" << static_cast<unsigned int>(left_data[diff_idx])
                << " right=" << static_cast<unsigned int>(right_data[diff_idx]) << std::endl
                << std::flush;
      dump_hex_context(left_data.data(), data_bytes, diff_idx);
      dump_hex_context(right_data.data(), data_bytes, diff_idx);
      return false;
    }
  }

  return true;
}

// Stream-aware comparison that recursively compares column content.
// Handles all column types including compound types (STRING, LIST, STRUCT, DICTIONARY)
// and tolerates INT32 vs INT64 offset differences in STRING/LIST columns.
bool cudf_tables_have_equal_contents_on_stream(const cudf::table& left,
                                               const cudf::table& right,
                                               rmm::cuda_stream_view stream_view)
{
  if (left.num_rows() != right.num_rows()) {
    std::cout << "[cudf-equal] row count mismatch: left=" << left.num_rows()
              << " right=" << right.num_rows() << std::endl
              << std::flush;
    return false;
  }
  if (left.num_columns() != right.num_columns()) {
    std::cout << "[cudf-equal] column count mismatch: left=" << left.num_columns()
              << " right=" << right.num_columns() << std::endl
              << std::flush;
    return false;
  }

  stream_view.synchronize();

  for (int col_idx = 0; col_idx < left.num_columns(); ++col_idx) {
    if (!columns_equal_recursive(left.view().column(col_idx),
                                 right.view().column(col_idx),
                                 "col[" + std::to_string(col_idx) + "]")) {
      return false;
    }
  }

  return true;
}

void expect_cudf_tables_equal_on_stream(const cudf::table& left,
                                        const cudf::table& right,
                                        rmm::cuda_stream_view stream_view)
{
  REQUIRE(cudf_tables_have_equal_contents_on_stream(left, right, stream_view));
}

// Simple logging adaptor to print all RMM device allocations/frees with pointers/sizes/stream/tid
class logging_device_resource : public rmm::mr::device_memory_resource {
 public:
  explicit logging_device_resource(rmm::mr::device_memory_resource* upstream) : _upstream(upstream)
  {
  }

  ~logging_device_resource() override = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    void* ptr = _upstream->allocate(stream, bytes);
    std::ostringstream oss;
    oss << "[rmm-alloc] ptr=" << ptr << " size=" << bytes << " stream=" << stream.value()
        << " tid=" << std::this_thread::get_id();
    std::cout << oss.str() << std::endl << std::flush;
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    std::ostringstream oss;
    oss << "[rmm-free ] ptr=" << ptr << " size=" << bytes << " stream=" << stream.value()
        << " tid=" << std::this_thread::get_id();
    std::cout << oss.str() << std::endl << std::flush;
    _upstream->deallocate(stream, ptr, bytes);
  }

  bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override
  {
    return this == &other;
  }

  rmm::mr::device_memory_resource* _upstream;
};

// Install the logging resource once per process (wraps whatever the current device resource is)
static void install_rmm_logging_resource_once()
{
  static bool installed = false;
  static std::unique_ptr<logging_device_resource> logging_resource;
  if (!installed) {
    auto* prev       = rmm::mr::get_current_device_resource();
    logging_resource = std::make_unique<logging_device_resource>(prev);
    rmm::mr::set_current_device_resource(logging_resource.get());
    installed = true;
    std::cout << "[rmm-log ] installed logging device resource adaptor" << std::endl << std::flush;
  }
}

static inline bool host_mem_equal(const uint8_t* a, const uint8_t* b, size_t n)
{
  if (a == b) return true;
  if ((a == nullptr) || (b == nullptr)) return false;
  return std::memcmp(a, b, n) == 0;
}

// Removed non-stream variants to enforce explicit stream usage in tests

}  // namespace test
}  // namespace cucascade
