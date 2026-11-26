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

// Randomized DISK round trips: tables of random bytes, random null masks, variable-length strings
// and nested types, written and read back through every disk io stack and along every path a
// batch can take (written by the device path and read by the host path, and vice versa). Unlike
// the per-type tests, every byte is distinct enough that a buffer read from the wrong offset or
// from a hole between buffers shows up. Adapted from the W3 review's experiments.

#include "utils/cudf_test_utils.hpp"
#include "utils/disk_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade;

namespace {

auto& shared_gpu_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::GPU, 0);
  return s;
}

auto& shared_host_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::HOST, 0);
  return s;
}

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

representation_converter_registry& registry()
{
  static auto r = [] {
    auto result = std::make_unique<representation_converter_registry>();
    register_builtin_converters(*result);
    return result;
  }();
  return *r;
}

rmm::device_buffer to_device(void const* data, std::size_t bytes)
{
  rmm::device_buffer buffer(data, bytes, shared_stream());
  shared_stream().sync();
  return buffer;
}

using mask_type = decltype(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

mask_type no_mask() { return cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED); }

mask_type mask_from_host(cudf::size_type rows, std::vector<std::uint8_t> const& host)
{
  auto mask = cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(mask.data(),
                                     host.data(),
                                     std::min(host.size(), mask.size()),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));
  shared_stream().sync();
  return mask;
}

std::pair<mask_type, cudf::size_type> random_mask(cudf::size_type rows, std::mt19937_64& rng)
{
  std::vector<std::uint8_t> host(cudf::bitmask_allocation_size_bytes(rows));
  for (auto& byte : host) {
    byte = static_cast<std::uint8_t>(rng());
  }
  cudf::size_type valid = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if ((host[static_cast<std::size_t>(i / 8)] & (1u << (i % 8))) != 0) { ++valid; }
  }
  return {mask_from_host(rows, host), rows - valid};
}

template <typename T>
std::unique_ptr<cudf::column> random_fixed(cudf::data_type type,
                                           cudf::size_type rows,
                                           std::mt19937_64& rng,
                                           bool nullable)
{
  std::vector<T> host(static_cast<std::size_t>(rows));
  for (auto& value : host) {
    auto const bits = rng();
    std::memcpy(&value, &bits, std::min(sizeof(T), sizeof(bits)));
    if constexpr (sizeof(T) > sizeof(bits)) {
      auto const more = rng();
      std::memcpy(reinterpret_cast<char*>(&value) + sizeof(bits), &more, sizeof(T) - sizeof(bits));
    }
    if constexpr (std::is_floating_point_v<T>) {
      if (value != value) { value = static_cast<T>(bits % 1000); }  // NaN != NaN
    }
  }
  auto data = to_device(host.data(), host.size() * sizeof(T));
  if (!nullable) {
    return std::make_unique<cudf::column>(type, rows, std::move(data), no_mask(), 0);
  }
  auto [mask, nulls] = random_mask(rows, rng);
  return std::make_unique<cudf::column>(type, rows, std::move(data), std::move(mask), nulls);
}

std::unique_ptr<cudf::column> strings_from_host(std::vector<std::int32_t> const& offsets,
                                                std::vector<char> const& chars,
                                                cudf::size_type rows,
                                                mask_type mask,
                                                cudf::size_type nulls)
{
  auto offsets_column =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   rows + 1,
                                   to_device(offsets.data(), offsets.size() * 4),
                                   no_mask(),
                                   0);
  return cudf::make_strings_column(
    rows, std::move(offsets_column), to_device(chars.data(), chars.size()), nulls, std::move(mask));
}

std::unique_ptr<cudf::column> random_strings(cudf::size_type rows,
                                             std::mt19937_64& rng,
                                             bool nullable)
{
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(rows) + 1, 0);
  std::vector<char> chars;
  for (cudf::size_type i = 0; i < rows; ++i) {
    auto const length = static_cast<int>(rng() % 23);
    for (int c = 0; c < length; ++c) {
      chars.push_back(static_cast<char>('a' + rng() % 26));
    }
    offsets[static_cast<std::size_t>(i) + 1] = static_cast<std::int32_t>(chars.size());
  }
  auto mask             = no_mask();
  cudf::size_type nulls = 0;
  if (nullable) { std::tie(mask, nulls) = random_mask(rows, rng); }
  return strings_from_host(offsets, chars, rows, std::move(mask), nulls);
}

std::unique_ptr<cudf::column> random_list(cudf::size_type rows, std::mt19937_64& rng)
{
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(rows) + 1, 0);
  for (cudf::size_type i = 0; i < rows; ++i) {
    offsets[static_cast<std::size_t>(i) + 1] =
      offsets[static_cast<std::size_t>(i)] + static_cast<std::int32_t>(rng() % 7);
  }
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    rows + 1,
                                                    to_device(offsets.data(), offsets.size() * 4),
                                                    no_mask(),
                                                    0));
  children.push_back(random_fixed<std::int32_t>(
    cudf::data_type{cudf::type_id::INT32}, offsets.back(), rng, /*nullable=*/true));
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                        rows,
                                        rmm::device_buffer{},
                                        no_mask(),
                                        0,
                                        std::move(children));
}

std::unique_ptr<cudf::column> random_dictionary(cudf::size_type rows, std::mt19937_64& rng)
{
  // Low cardinality: 50 distinct string keys.
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(rows) + 1, 0);
  std::vector<char> chars;
  for (cudf::size_type i = 0; i < rows; ++i) {
    auto const key = "key" + std::to_string(rng() % 50);
    chars.insert(chars.end(), key.begin(), key.end());
    offsets[static_cast<std::size_t>(i) + 1] = static_cast<std::int32_t>(chars.size());
  }
  auto strings = strings_from_host(offsets, chars, rows, no_mask(), 0);
  return cudf::dictionary::encode(strings->view());
}

/// Every column type the DISK converters handle, with random content and random null masks.
std::unique_ptr<cudf::table> random_mixed_table(cudf::size_type rows, std::uint64_t seed)
{
  using cudf::data_type;
  using cudf::type_id;
  std::mt19937_64 rng(seed);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(random_fixed<std::int8_t>(data_type{type_id::INT8}, rows, rng, true));
  columns.push_back(random_fixed<std::int16_t>(data_type{type_id::INT16}, rows, rng, false));
  columns.push_back(random_fixed<std::int32_t>(data_type{type_id::INT32}, rows, rng, true));
  columns.push_back(random_fixed<std::int64_t>(data_type{type_id::INT64}, rows, rng, true));
  columns.push_back(random_fixed<double>(data_type{type_id::FLOAT64}, rows, rng, true));
  columns.push_back(random_fixed<float>(data_type{type_id::FLOAT32}, rows, rng, false));
  columns.push_back(random_fixed<std::int32_t>(data_type{type_id::DECIMAL32, -3}, rows, rng, true));
  columns.push_back(random_fixed<std::int64_t>(data_type{type_id::DECIMAL64, 2}, rows, rng, true));
  columns.push_back(random_fixed<__int128_t>(data_type{type_id::DECIMAL128, -5}, rows, rng, true));
  columns.push_back(
    random_fixed<std::int64_t>(data_type{type_id::TIMESTAMP_NANOSECONDS}, rows, rng, true));
  columns.push_back(random_strings(rows, rng, true));
  columns.push_back(random_strings(rows, rng, false));
  columns.push_back(random_list(rows, rng));
  {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.push_back(random_fixed<std::int64_t>(data_type{type_id::INT64}, rows, rng, true));
    fields.push_back(random_strings(rows, rng, true));
    columns.push_back(cudf::make_structs_column(rows, std::move(fields), 0, no_mask()));
  }
  columns.push_back(random_dictionary(rows, rng));
  shared_stream().sync();
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<gpu_table_representation> make_gpu_rep(std::unique_ptr<cudf::table> table)
{
  return std::make_unique<gpu_table_representation>(
    std::move(table), *shared_gpu_space(), ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
}

}  // namespace

TEST_CASE("random mixed tables round-trip through every disk backend and path",
          "[disk][io][randomized]")
{
  // Row counts around the edges: empty, single row, unaligned tails, buffers spanning host blocks.
  auto const rows = GENERATE(as<cudf::size_type>{}, 0, 1, 7, 4099, 300'001);
  CAPTURE(rows);
  auto const gpu = make_gpu_rep(random_mixed_table(rows, 1234u + static_cast<std::uint64_t>(rows)));
  auto const& expected = gpu->get_table_view();
  auto& gpu_space      = *shared_gpu_space();
  auto& host_space     = *shared_host_space();
  auto const stream    = shared_stream();

  for (auto const backend : test::disk_test_backends) {
    CAPTURE(test::to_string(backend));
    auto* disk_space = test::shared_disk_space_for(backend);
    if (disk_space == nullptr) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: unavailable here");
      continue;
    }
    auto& disk = *disk_space;

    // Written by the device path ...
    auto const from_gpu = registry().convert<disk_data_representation>(*gpu, &disk, stream);
    {  // ... read by the device path
      auto back = registry().convert<gpu_table_representation>(*from_gpu, &gpu_space, stream);
      test::expect_cudf_tables_equal_on_stream(expected, back->get_table_view(), stream);
    }
    {  // ... read by the host path
      auto host = registry().convert<host_data_representation>(*from_gpu, &host_space, stream);
      auto back = registry().convert<gpu_table_representation>(*host, &gpu_space, stream);
      test::expect_cudf_tables_equal_on_stream(expected, back->get_table_view(), stream);
    }

    // Written by the host path ...
    auto const host     = registry().convert<host_data_representation>(*gpu, &host_space, stream);
    auto const from_cpu = registry().convert<disk_data_representation>(*host, &disk, stream);
    {  // ... read by the device path
      auto back = registry().convert<gpu_table_representation>(*from_cpu, &gpu_space, stream);
      test::expect_cudf_tables_equal_on_stream(expected, back->get_table_view(), stream);
    }
    {  // ... read by the host path
      auto host_back = registry().convert<host_data_representation>(*from_cpu, &host_space, stream);
      auto back      = registry().convert<gpu_table_representation>(*host_back, &gpu_space, stream);
      test::expect_cudf_tables_equal_on_stream(expected, back->get_table_view(), stream);
    }
    stream.sync();
  }
}

TEST_CASE("a zero-column table round-trips through every disk backend", "[disk][io][randomized]")
{
  auto const gpu =
    make_gpu_rep(std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}));
  for (auto const backend : test::disk_test_backends) {
    CAPTURE(test::to_string(backend));
    auto* disk_space = test::shared_disk_space_for(backend);
    if (disk_space == nullptr) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: unavailable here");
      continue;
    }
    auto& disk    = *disk_space;
    auto from_gpu = registry().convert<disk_data_representation>(*gpu, &disk, shared_stream());
    auto back     = registry().convert<gpu_table_representation>(
      *from_gpu, shared_gpu_space().get(), shared_stream());
    REQUIRE(back->get_table_view().num_columns() == 0);
    auto host = registry().convert<host_data_representation>(
      *gpu, shared_host_space().get(), shared_stream());
    auto from_cpu  = registry().convert<disk_data_representation>(*host, &disk, shared_stream());
    auto host_back = registry().convert<host_data_representation>(
      *from_cpu, shared_host_space().get(), shared_stream());
    REQUIRE(host_back->get_size_in_bytes() == 0);
  }
}
