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
 * @file disk_test_utils.hpp
 * @brief DISK memory spaces served by each io stack, for the cudf-layer converter tests.
 */

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <string>

namespace cucascade {
namespace test {

/// The io stacks every DISK round-trip test runs on: the default a space adopts when it has no
/// context, and each backend forced explicitly. kvikio joins once its context supports writes.
inline constexpr std::array<disk_io_backend, 3> disk_test_backends{
  disk_io_backend::automatic, disk_io_backend::uring, disk_io_backend::cufile};

[[nodiscard]] inline char const* to_string(disk_io_backend backend) noexcept
{
  switch (backend) {
    case disk_io_backend::automatic: return "default";
    case disk_io_backend::uring: return "uring";
    case disk_io_backend::cufile: return "cufile";
    case disk_io_backend::kvikio: return "kvikio";
  }
  return "unknown";
}

/**
 * @brief A DISK memory space under @p mount_path served by @p backend.
 *
 * automatic leaves the space without a context, so the converters adopt the process-shared
 * default; every other backend gets its own context from make_disk_io_context().
 */
[[nodiscard]] inline std::shared_ptr<memory::memory_space> make_disk_space(
  disk_io_backend backend,
  int disk_id            = 0,
  std::string mount_path = std::filesystem::temp_directory_path().string())
{
  memory::disk_memory_space_config config;
  config.disk_id         = disk_id;
  config.memory_capacity = 1024UL * 1024 * 1024;
  config.mount_paths     = std::move(mount_path);
  if (backend != disk_io_backend::automatic) {
    disk_io_context_options options;
    options.backend   = backend;
    config.io_context = make_disk_io_context(options);
  }
  return std::make_shared<memory::memory_space>(config);
}

/// Process-lifetime DISK spaces, one per backend in disk_test_backends (shared by all tests to
/// avoid rebuilding reactor pools and pinned staging per test).
[[nodiscard]] inline memory::memory_space& shared_disk_space_for(disk_io_backend backend)
{
  static auto const spaces = [] {
    std::array<std::shared_ptr<memory::memory_space>, disk_test_backends.size()> result;
    for (std::size_t i = 0; i < disk_test_backends.size(); ++i) {
      result[i] = make_disk_space(disk_test_backends[i], static_cast<int>(i));
    }
    return result;
  }();
  for (std::size_t i = 0; i < disk_test_backends.size(); ++i) {
    if (disk_test_backends[i] == backend) { return *spaces[i]; }
  }
  return *spaces.front();
}

/**
 * @brief Whether the space's context (or the default it will adopt) can write what a DISK
 *        converter needs: device memory when @p device, host memory otherwise.
 */
[[nodiscard]] inline bool disk_space_can_write(memory::memory_space const& space, bool device)
{
  auto context = space.get_io_context();
  if (!context) { context = default_disk_io_context(); }
  return device ? context->supports_device_write() : context->supports_write();
}

}  // namespace test
}  // namespace cucascade
