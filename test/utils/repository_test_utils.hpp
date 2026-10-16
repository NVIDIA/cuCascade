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

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <stdexcept>

namespace cucascade {
namespace test {

/** @brief Reject I/O; repository storage tests need no device or pinned buffers. */
class repository_test_io_backend : public idisk_io_backend {
 public:
  void write(const std::filesystem::path&, const void*, size_t, size_t, ::cuda::stream_ref) override
  {
    throw std::logic_error("repository test attempted disk I/O");
  }
  void read(const std::filesystem::path&, void*, size_t, size_t, ::cuda::stream_ref) override
  {
    throw std::logic_error("repository test attempted disk I/O");
  }
  void write(const std::filesystem::path&, const void*, size_t, size_t) override
  {
    throw std::logic_error("repository test attempted disk I/O");
  }
  void read(const std::filesystem::path&, void*, size_t, size_t) override
  {
    throw std::logic_error("repository test attempted disk I/O");
  }
};

inline memory::memory_space& get_repository_test_memory_space()
{
  static memory::memory_space space(memory::disk_memory_space_config{0, 1ull << 30, "/tmp"},
                                    std::make_shared<repository_test_io_backend>());
  return space;
}

/** @brief Lightweight payload for testing repositories independently of CUDA allocation. */
class repository_test_representation : public idata_representation {
 public:
  explicit repository_test_representation(size_t size = 0)
    : idata_representation(get_repository_test_memory_space()), _size(size)
  {
  }

  size_t get_size_in_bytes() const override { return _size; }
  size_t get_uncompressed_data_size_in_bytes() const override { return _size; }
  std::unique_ptr<idata_representation> clone(::cuda::stream_ref) const override
  {
    return std::make_unique<repository_test_representation>(_size);
  }

 private:
  size_t _size;
};

inline std::shared_ptr<data_batch> make_repository_test_batch(uint64_t id)
{
  return data_batch::make(id, std::make_unique<repository_test_representation>());
}

}  // namespace test
}  // namespace cucascade
