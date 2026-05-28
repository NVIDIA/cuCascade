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

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/io_backend_registry.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/aligned.hpp>

#include <catch2/catch.hpp>

#include <cstdint>
#include <filesystem>
#include <numeric>
#include <string>
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
  auto file_path = tmp_dir / "test_pipeline_host.bin";

  // Write test data: 4096 bytes with sequential values
  std::vector<uint8_t> write_data(4096);
  std::iota(write_data.begin(), write_data.end(), static_cast<uint8_t>(0));

  backend->write(file_path, write_data.data(), write_data.size(), 0);

  // Read it back
  std::vector<uint8_t> read_data(4096, 0);
  backend->read(file_path, read_data.data(), read_data.size(), 0);

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
  // mkstemps creates the files — clean up
  std::filesystem::remove(path1);
  std::filesystem::remove(path2);
}

// =============================================================================
// Alignment Utility Tests
// =============================================================================

TEST_CASE("rmm::align_up rounds to 4KB boundaries", "[disk][format]")
{
  REQUIRE(rmm::align_up(0, 4096) == 0);
  REQUIRE(rmm::align_up(1, 4096) == 4096);
  REQUIRE(rmm::align_up(4095, 4096) == 4096);
  REQUIRE(rmm::align_up(4096, 4096) == 4096);
  REQUIRE(rmm::align_up(4097, 4096) == 8192);
  REQUIRE(rmm::align_up(100, 8) == 104);
}

// =============================================================================
// I/O Backend Registry Default Management Tests
// =============================================================================

TEST_CASE("io_backend_registry default name is pipeline", "[disk][io][registry]")
{
  io_backend_registry registry;
  REQUIRE(registry.get_default_name() == "pipeline");
}

TEST_CASE("io_backend_registry set_default changes default", "[disk][io][registry]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);

  // Register a second backend
  registry.register_backend("custom",
                            []() -> std::shared_ptr<idisk_io_backend> { return nullptr; });

  registry.set_default("custom");
  REQUIRE(registry.get_default_name() == "custom");

  registry.set_default("pipeline");
  REQUIRE(registry.get_default_name() == "pipeline");
}

TEST_CASE("io_backend_registry set_default throws for unregistered name", "[disk][io][registry]")
{
  io_backend_registry registry;
  REQUIRE_THROWS_AS(registry.set_default("nonexistent"), std::runtime_error);
}

TEST_CASE("io_backend_registry create_default_backend creates pipeline", "[disk][io][registry]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  auto backend = registry.create_default_backend();
  REQUIRE(backend != nullptr);
}

TEST_CASE("io_backend_registry clear resets default name", "[disk][io][registry]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  registry.register_backend("custom",
                            []() -> std::shared_ptr<idisk_io_backend> { return nullptr; });
  registry.set_default("custom");
  REQUIRE(registry.get_default_name() == "custom");

  registry.clear();
  REQUIRE(registry.get_default_name() == "pipeline");
}

// =============================================================================
// memory_space I/O Backend Tests
// =============================================================================

TEST_CASE("disk memory_space has default pipeline backend", "[disk][io][memory_space]")
{
  cucascade::memory::disk_memory_space_config config{0, 1024 * 1024 * 1024, "/tmp"};
  cucascade::memory::memory_space space(config);
  REQUIRE_NOTHROW(space.get_io_backend());
}

TEST_CASE("disk memory_space accepts custom backend", "[disk][io][memory_space]")
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  auto backend = registry.create_backend("pipeline");

  cucascade::memory::disk_memory_space_config config{0, 1024 * 1024 * 1024, "/tmp"};
  cucascade::memory::memory_space space(config, backend);
  REQUIRE(&space.get_io_backend() == backend.get());
}

TEST_CASE("disk memory_space rejects null backend", "[disk][io][memory_space]")
{
  cucascade::memory::disk_memory_space_config config{0, 1024 * 1024 * 1024, "/tmp"};
  REQUIRE_THROWS_AS(
    cucascade::memory::memory_space(config, std::shared_ptr<idisk_io_backend>(nullptr)),
    std::invalid_argument);
}
