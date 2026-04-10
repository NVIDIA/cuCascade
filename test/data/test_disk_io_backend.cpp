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

#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/io_backend_registry.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <catch2/catch.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

using namespace cucascade;

// =============================================================================
// I/O Backend Factory Tests
// =============================================================================

TEST_CASE("io_backend_registry creates pipeline backend", "[disk][io]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  auto backend = registry.create_backend("pipeline");
  REQUIRE(backend != nullptr);
}

// =============================================================================
// Pipeline Host I/O Round-Trip Tests
// =============================================================================

TEST_CASE("pipeline backend host write and read round-trip", "[disk][io][pipeline]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  auto backend = registry.create_backend("pipeline");

  auto tmp_dir   = std::filesystem::temp_directory_path();
  auto file_path = (tmp_dir / "test_pipeline_host.bin").string();

  // Write test data: 4096 bytes with sequential values
  std::vector<uint8_t> write_data(4096);
  std::iota(write_data.begin(), write_data.end(), static_cast<uint8_t>(0));

  backend->write_host(file_path, write_data.data(), write_data.size(), 0);

  // Read it back
  std::vector<uint8_t> read_data(4096, 0);
  backend->read_host(file_path, read_data.data(), read_data.size(), 0);

  REQUIRE(write_data == read_data);

  // Cleanup
  std::filesystem::remove(file_path);
}

// =============================================================================
// Disk File Path Generation Tests
// =============================================================================

TEST_CASE("generate_disk_file_path produces unique paths", "[disk][io]")
{
  auto tmp_dir = std::filesystem::temp_directory_path().string();
  auto path1   = cucascade::memory::generate_disk_file_path(tmp_dir);
  auto path2   = cucascade::memory::generate_disk_file_path(tmp_dir);
  REQUIRE(path1 != path2);
  REQUIRE(path1.find("batch_") != std::string::npos);
  REQUIRE(path1.find(".cucascade") != std::string::npos);
}

// =============================================================================
// Column Metadata Serialization Tests
// =============================================================================

TEST_CASE("column_metadata round-trip serialization", "[disk][format]")
{
  // Create column_metadata with nested children (simulates STRUCT<STRING, LIST<INT32>>)
  cucascade::memory::column_metadata string_child{};
  string_child.type_id          = cudf::type_id::STRING;
  string_child.num_rows         = 10;
  string_child.null_count       = 2;
  string_child.scale            = 0;
  string_child.has_null_mask    = true;
  string_child.null_mask_offset = 0;
  string_child.null_mask_size   = 64;
  string_child.has_data         = true;
  string_child.data_offset      = 0;
  string_child.data_size        = 128;

  cucascade::memory::column_metadata int_child{};
  int_child.type_id          = cudf::type_id::INT32;
  int_child.num_rows         = 10;
  int_child.null_count       = 0;
  int_child.scale            = 0;
  int_child.has_null_mask    = false;
  int_child.null_mask_offset = 0;
  int_child.null_mask_size   = 0;
  int_child.has_data         = true;
  int_child.data_offset      = 0;
  int_child.data_size        = 40;

  cucascade::memory::column_metadata list_child{};
  list_child.type_id          = cudf::type_id::LIST;
  list_child.num_rows         = 10;
  list_child.null_count       = 0;
  list_child.scale            = 0;
  list_child.has_null_mask    = false;
  list_child.null_mask_offset = 0;
  list_child.null_mask_size   = 0;
  list_child.has_data         = true;
  list_child.data_offset      = 0;
  list_child.data_size        = 44;  // offsets
  list_child.children.push_back(int_child);

  cucascade::memory::column_metadata struct_col{};
  struct_col.type_id          = cudf::type_id::STRUCT;
  struct_col.num_rows         = 10;
  struct_col.null_count       = 0;
  struct_col.scale            = 0;
  struct_col.has_null_mask    = false;
  struct_col.null_mask_offset = 0;
  struct_col.null_mask_size   = 0;
  struct_col.has_data         = false;
  struct_col.data_offset      = 0;
  struct_col.data_size        = 0;
  struct_col.children.push_back(string_child);
  struct_col.children.push_back(list_child);

  std::vector<cucascade::memory::column_metadata> columns = {struct_col};

  auto serialized = cucascade::serialize_column_metadata(columns);
  REQUIRE(serialized.size() > 0);

  auto deserialized = cucascade::deserialize_column_metadata(serialized.data(), serialized.size());
  REQUIRE(deserialized.size() == 1);

  // Verify struct column
  auto& result = deserialized[0];
  REQUIRE(result.type_id == cudf::type_id::STRUCT);
  REQUIRE(result.num_rows == 10);
  REQUIRE(result.children.size() == 2);

  // Verify string child
  REQUIRE(result.children[0].type_id == cudf::type_id::STRING);
  REQUIRE(result.children[0].null_count == 2);
  REQUIRE(result.children[0].has_null_mask == true);
  REQUIRE(result.children[0].null_mask_size == 64);

  // Verify list child with int grandchild
  REQUIRE(result.children[1].type_id == cudf::type_id::LIST);
  REQUIRE(result.children[1].children.size() == 1);
  REQUIRE(result.children[1].children[0].type_id == cudf::type_id::INT32);
  REQUIRE(result.children[1].children[0].data_size == 40);
}

// =============================================================================
// Alignment Utility Tests
// =============================================================================

TEST_CASE("align_up rounds to 4KB boundaries", "[disk][format]")
{
  REQUIRE(cucascade::align_up(0, 4096) == 0);
  REQUIRE(cucascade::align_up(1, 4096) == 4096);
  REQUIRE(cucascade::align_up(4095, 4096) == 4096);
  REQUIRE(cucascade::align_up(4096, 4096) == 4096);
  REQUIRE(cucascade::align_up(4097, 4096) == 8192);
  REQUIRE(cucascade::align_up(100, 8) == 104);
}
