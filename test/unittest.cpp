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

#define CATCH_CONFIG_RUNNER

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>

namespace {

class test_gpu_pool {
 public:
  test_gpu_pool()
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) { return; }

    auto [free_bytes, total_bytes] = rmm::available_device_memory();
    if (total_bytes == 0) { return; }

    auto max_bytes = read_pool_max_bytes(total_bytes);
    if (max_bytes == 0) { return; }

    auto initial_bytes = default_initial_bytes;
    if (initial_bytes > max_bytes) { initial_bytes = max_bytes; }

    pool_     = std::make_unique<rmm::mr::cuda_async_memory_resource>(initial_bytes, max_bytes);
    previous_ = rmm::mr::get_current_device_resource();
    rmm::mr::set_current_device_resource(pool_.get());
  }

  ~test_gpu_pool()
  {
    if (pool_ != nullptr && previous_ != nullptr) {
      rmm::mr::set_current_device_resource(previous_);
    }
  }

 private:
  static std::size_t read_pool_max_bytes(std::size_t total_bytes)
  {
    auto default_bytes = std::min(default_max_bytes, static_cast<std::size_t>(total_bytes * 0.9));
    const char* env    = std::getenv("CUCASCADE_TEST_GPU_POOL_BYTES");
    if (env == nullptr || *env == '\0') { return default_bytes; }
    char* end = nullptr;
    auto val  = std::strtoull(env, &end, 10);
    if (end == env) { return default_bytes; }
    return std::min(static_cast<std::size_t>(val), default_bytes);
  }

  static constexpr std::size_t default_initial_bytes = 2ULL * 1024 * 1024 * 1024;
  static constexpr std::size_t default_max_bytes     = 10ULL * 1024 * 1024 * 1024;

  std::unique_ptr<rmm::mr::cuda_async_memory_resource> pool_;
  rmm::mr::device_memory_resource* previous_{nullptr};
};

test_gpu_pool global_pool;

}  // namespace

namespace {
struct device_sync_listener : Catch::TestEventListenerBase {
  using Catch::TestEventListenerBase::TestEventListenerBase;

  void testCaseEnded(Catch::TestCaseStats const&) override { cudaDeviceSynchronize(); }
};
}  // namespace

CATCH_REGISTER_LISTENER(device_sync_listener)

int main(int argc, char* argv[])
{
  // Run tests
  int code = Catch::Session().run(argc, argv);
  std::fflush(stdout);
  std::fflush(stderr);
  return code;
}
