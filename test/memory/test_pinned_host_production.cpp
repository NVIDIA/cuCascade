/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory/test_gpu_kernels.cuh"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

using namespace cucascade::memory;

TEST_CASE("Production over-aligned storage is mapped and usable by a GPU",
          "[small_pinned][production][alignment]")
{
  auto const alignment        = GENERATE(std::size_t{512}, std::size_t{8192}, std::size_t{65536});
  constexpr std::size_t bytes = 32768;
  rmm::mr::pinned_host_memory_resource pinned;
  fixed_size_host_memory_resource upstream{0, pinned, 1ULL << 20, 1ULL << 20, 65536, 4, 1};
  small_pinned_host_memory_resource resource{upstream};
  auto release = [&](void* ptr) { resource.deallocate_sync(ptr, bytes, alignment); };
  std::unique_ptr<void, decltype(release)> allocation{resource.allocate_sync(bytes, alignment),
                                                      release};
  REQUIRE(reinterpret_cast<std::uintptr_t>(allocation.get()) % alignment == 0);
  void* mapped = nullptr;
  CUCASCADE_CUDA_TRY(::cudaHostGetDevicePointer(&mapped, allocation.get(), 0));
  std::uint64_t checksum = 0;
  CUCASCADE_CUDA_TRY(performGpuMemoryWork(mapped, bytes, &checksum));
  REQUIRE(checksum != 0);
  REQUIRE(resource.get_statistics().large_uncacheable == 1);
  allocation.reset();
  REQUIRE(resource.large_cache_bytes() == 0);
  REQUIRE(::cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Production zero and null wrappers preserve an active capture",
          "[small_pinned][production][capture]")
{
  auto const alignment = GENERATE(std::size_t{0}, alignof(std::max_align_t));
  rmm::mr::pinned_host_memory_resource pinned;
  fixed_size_host_memory_resource upstream{0, pinned, 1ULL << 20, 1ULL << 20, 65536, 4, 1};
  small_pinned_host_memory_resource resource{upstream};
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  CUCASCADE_CUDA_TRY(::cudaStreamBeginCapture(stream.value(), cudaStreamCaptureModeThreadLocal));
  // Save outcomes until capture ends so an assertion cannot strand an active capture.
  void* async_zero = nullptr;
  void* sync_zero  = nullptr;
  std::exception_ptr error;
  try {
    async_zero = resource.allocate(stream, 0, alignment);
    sync_zero  = resource.allocate_sync(0, alignment);
    for (std::size_t const bytes : std::array<std::size_t, 3>{0, 42, 9000}) {
      resource.deallocate(stream, nullptr, bytes);
      resource.deallocate_sync(nullptr, bytes);
    }
  } catch (...) {
    error = std::current_exception();
  }
  cudaGraph_t graph = nullptr;
  auto const status = ::cudaStreamEndCapture(stream.value(), &graph);
  if (graph != nullptr) { CUCASCADE_CUDA_TRY(::cudaGraphDestroy(graph)); }
  if (error) { std::rethrow_exception(error); }
  REQUIRE(status == cudaSuccess);
  REQUIRE(async_zero == nullptr);
  REQUIRE(sync_zero == nullptr);
  REQUIRE(resource.get_statistics().large_requests == 0);
  REQUIRE(::cudaGetLastError() == cudaSuccess);
}
