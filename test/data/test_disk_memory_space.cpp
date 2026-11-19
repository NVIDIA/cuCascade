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

#include "utils/mock_test_utils.hpp"

#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/aligned.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade;

namespace {

constexpr std::size_t disk_capacity = 1024UL * 1024 * 1024;

[[nodiscard]] memory::disk_memory_space_config make_disk_config(int disk_id = 0)
{
  memory::disk_memory_space_config config;
  config.disk_id         = disk_id;
  config.memory_capacity = disk_capacity;
  config.mount_paths     = std::filesystem::temp_directory_path().string();
  return config;
}

/**
 * @brief A non-null io context handle the core library can store but never dereferences.
 *
 * cucascade::io::ioctx is incomplete here; the core only copies and destroys the shared_ptr, so an
 * aliasing pointer to some unrelated owned object stands in for a real context.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> make_placeholder_context()
{
  auto owner = std::make_shared<int>(0);
  return std::shared_ptr<io::ioctx>(owner, reinterpret_cast<io::ioctx*>(owner.get()));
}

}  // namespace

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
// memory_space io context Tests
//
// The core library only stores the io context (cucascade::io::ioctx is incomplete here); the
// tests that build real contexts and convert through them live in the cudf test executable.
// =============================================================================

TEST_CASE("disk memory_space has no io context until one is set or adopted",
          "[disk][io][memory_space]")
{
  memory::memory_space space(make_disk_config());
  REQUIRE(space.get_io_context() == nullptr);
}

TEST_CASE("disk memory_space rejects a null io context", "[disk][io][memory_space]")
{
  memory::memory_space space(make_disk_config());
  REQUIRE_THROWS_AS(space.set_io_context(nullptr), std::invalid_argument);
  REQUIRE(space.get_io_context() == nullptr);
}

TEST_CASE("disk memory_space installs nothing when the default factory fails",
          "[disk][io][memory_space]")
{
  memory::memory_space space(make_disk_config());
  int calls = 0;
  REQUIRE_THROWS_AS(space.get_or_init_io_context([&calls]() -> std::shared_ptr<io::ioctx> {
    ++calls;
    return nullptr;
  }),
                    std::runtime_error);
  REQUIRE(calls == 1);
  REQUIRE(space.get_io_context() == nullptr);

  REQUIRE_THROWS_AS(space.get_or_init_io_context([]() -> std::shared_ptr<io::ioctx> {
    throw std::runtime_error("factory failed");
  }),
                    std::runtime_error);
  REQUIRE(space.get_io_context() == nullptr);
}

TEST_CASE("disk memory_space stores the configured io context and replaces it",
          "[disk][io][memory_space]")
{
  auto configured   = make_placeholder_context();
  auto config       = make_disk_config();
  config.io_context = configured;
  memory::memory_space space(config);
  REQUIRE(space.get_io_context() == configured);

  int calls    = 0;
  auto adopted = space.get_or_init_io_context([&calls] {
    ++calls;
    return make_placeholder_context();
  });
  REQUIRE(adopted == configured);  // a configured context is never replaced by the default
  REQUIRE(calls == 0);

  auto replacement = make_placeholder_context();
  space.set_io_context(replacement);
  REQUIRE(space.get_io_context() == replacement);
}

TEST_CASE("concurrent first uses of a fresh disk memory_space adopt one default",
          "[disk][io][memory_space][concurrency]")
{
  memory::memory_space space(make_disk_config());
  constexpr std::size_t n_threads = 16;
  std::atomic<int> factory_calls{0};
  std::atomic<bool> go{false};
  std::vector<std::shared_ptr<io::ioctx>> seen(n_threads);
  {
    std::vector<std::jthread> threads;
    for (std::size_t i = 0; i < n_threads; ++i) {
      threads.emplace_back([&, i] {
        while (!go.load(std::memory_order_acquire)) {}
        seen[i] = space.get_or_init_io_context([&factory_calls] {
          factory_calls.fetch_add(1, std::memory_order_relaxed);
          return make_placeholder_context();
        });
      });
    }
    go.store(true, std::memory_order_release);
  }
  REQUIRE(factory_calls.load() == 1);
  REQUIRE(seen.front() != nullptr);
  for (auto const& context : seen) {
    REQUIRE(context == seen.front());
  }
  REQUIRE(space.get_io_context() == seen.front());
}

TEST_CASE("io context accessors reject non-DISK memory spaces", "[disk][io][memory_space]")
{
  auto gpu_space = test::make_mock_memory_space(memory::Tier::GPU, 0);
  REQUIRE_THROWS_AS(gpu_space->get_io_context(), std::logic_error);
  REQUIRE_THROWS_AS(gpu_space->set_io_context(nullptr), std::logic_error);
  REQUIRE_THROWS_AS(
    gpu_space->get_or_init_io_context([]() -> std::shared_ptr<io::ioctx> { return nullptr; }),
    std::logic_error);
}

TEST_CASE("disk memory_space rejects an empty mount path", "[disk][memory_space]")
{
  auto config        = make_disk_config();
  config.mount_paths = "";
  REQUIRE_THROWS_AS(memory::memory_space(config), std::invalid_argument);
}
