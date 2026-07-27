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

/**
 * Test Tags:
 * [reservation_aware] - reservation_aware_resource_adaptor behavior
 * [oom]               - out-of-memory diagnostics
 * [gpu]               - requires a CUDA device
 *
 * These tests drive the adaptor into an out-of-memory condition by giving it a
 * tiny capacity, then assert that the cudaMemPool_t reported in the resulting
 * cucascade_out_of_memory matches the pool of whatever upstream allocator was
 * supplied. The adaptor recovers the handle from the upstream itself (via
 * cuda::mr::resource_cast), so it must be present for pool-owning resources and
 * null for resources that do not expose a pool.
 */

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <cstddef>

using namespace cucascade::memory;

namespace {

// A capacity far smaller than the request below, so the adaptor's own tracked
// limit is exceeded before it ever touches the upstream allocator. This yields a
// deterministic OOM without depending on how much GPU memory is actually free.
constexpr std::size_t tiny_capacity   = 1024;
constexpr std::size_t oversized_bytes = 1ULL << 20;  // 1 MiB >> tiny_capacity

bool has_cuda_device()
{
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

// Builds an adaptor over `upstream` with a tiny capacity, forces an OOM, and
// returns the pool handle carried by the thrown cucascade_out_of_memory.
cudaMemPool_t oom_pool_handle_for(rmm::device_async_resource_ref upstream)
{
  reservation_aware_resource_adaptor adaptor{
    memory_space_id{Tier::GPU, 0}, upstream, tiny_capacity, tiny_capacity};

  try {
    adaptor.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256);
  } catch (const cucascade_out_of_memory& e) {
    // Compare underlying values rather than the enums directly: MemoryError has an
    // is_error_code_enum specialization, so a Catch2 comparison of MemoryError would
    // instantiate the error-code streaming path and pull in make_error_code(), which
    // error.hpp declares inline but never defines.
    REQUIRE(static_cast<int>(e.error_kind) == static_cast<int>(MemoryError::LIMIT_EXCEEDED));
    return e.pool_handle;
  }
  FAIL("expected the oversized allocation to throw cucascade_out_of_memory");
  return nullptr;
}

}  // namespace

TEST_CASE("OOM reports the pool of a cuda_async_memory_resource upstream",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::mr::cuda_async_memory_resource async_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{async_mr});

  CHECK(reported != nullptr);
  CHECK(reported == async_mr.pool_handle());
}

TEST_CASE("OOM reports the viewed pool of a cuda_async_view_memory_resource upstream",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  cudaMemPool_t default_pool = nullptr;
  REQUIRE(cudaDeviceGetDefaultMemPool(&default_pool, 0) == cudaSuccess);
  REQUIRE(default_pool != nullptr);

  rmm::mr::cuda_async_view_memory_resource view_mr{default_pool};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{view_mr});

  CHECK(reported == default_pool);
  CHECK(reported == view_mr.pool_handle());
}

TEST_CASE("OOM reports the default managed pool of a cuda_async_managed_memory_resource upstream",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::mr::cuda_async_managed_memory_resource managed_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{managed_mr});

  CHECK(reported != nullptr);
  CHECK(reported == managed_mr.pool_handle());
}

TEST_CASE("OOM reports a null pool for a non-pool upstream (cuda_memory_resource)",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::mr::cuda_memory_resource cuda_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{cuda_mr});

  CHECK(reported == nullptr);
}

TEST_CASE("An explicitly supplied pool handle overrides upstream introspection",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  // Upstream owns pool A, but we hand the adaptor a different handle B; the OOM
  // must report B, proving the explicit argument wins over recovery.
  rmm::mr::cuda_async_memory_resource async_mr{};
  cudaMemPool_t explicit_pool = nullptr;
  REQUIRE(cudaDeviceGetDefaultMemPool(&explicit_pool, 0) == cudaSuccess);
  REQUIRE(explicit_pool != async_mr.pool_handle());

  reservation_aware_resource_adaptor adaptor{
    memory_space_id{Tier::GPU, 0},
    rmm::device_async_resource_ref{async_mr},
    tiny_capacity,
    tiny_capacity,
    nullptr,
    nullptr,
    reservation_aware_resource_adaptor::AllocationTrackingScope::PER_STREAM,
    explicit_pool};

  try {
    adaptor.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256);
    FAIL("expected the oversized allocation to throw cucascade_out_of_memory");
  } catch (const cucascade_out_of_memory& e) {
    CHECK(e.pool_handle == explicit_pool);
    CHECK(e.pool_handle != async_mr.pool_handle());
  }
}
