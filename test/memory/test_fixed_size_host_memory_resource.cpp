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

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace cucascade::memory;

namespace {

constexpr std::size_t block_size = 1 << 20;
constexpr std::size_t pool_size  = 4;

/// Host upstream that counts live allocations and fails once @c fail_after have been made.
class counting_host_resource {
 public:
  struct state {
    std::atomic<int> allocations{0};
    std::atomic<int> live{0};
    int fail_after{-1};
  };

  explicit counting_host_resource(std::shared_ptr<state> s) : _s(std::move(s)) {}

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    if (_s->fail_after >= 0 && _s->allocations.fetch_add(1) >= _s->fail_after) {
      throw rmm::bad_alloc("counting_host_resource: injected failure");
    }
    if (_s->fail_after < 0) { _s->allocations.fetch_add(1); }
    void* p = std::malloc(bytes);
    if (p == nullptr) { throw rmm::bad_alloc("malloc"); }
    _s->live.fetch_add(1);
    return p;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::free(p);
    _s->live.fetch_sub(1);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(counting_host_resource const& other) const noexcept { return _s == other._s; }

  friend void get_property(counting_host_resource const&, ::cuda::mr::device_accessible) noexcept {}
  friend void get_property(counting_host_resource const&, ::cuda::mr::host_accessible) noexcept {}

 private:
  std::shared_ptr<state> _s;
};

}  // namespace

TEST_CASE("Initial pools are all pinned and usable", "[fixed_size_host]")
{
  numa_region_pinned_host_memory_resource upstream(0);
  constexpr std::size_t initial_pools = 40;  // more pools than allocator threads
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;
  fixed_size_host_memory_resource mr(
    0, upstream, capacity, capacity, block_size, pool_size, initial_pools);

  REQUIRE(mr.get_total_blocks() == initial_pools * pool_size);
  REQUIRE(mr.get_free_blocks() == initial_pools * pool_size);

  // Every block of every pool must be pinned: a copy from it to the device must succeed.
  void* d = nullptr;
  REQUIRE(cudaMalloc(&d, block_size) == cudaSuccess);
  auto blocks = mr.allocate_multiple_blocks(capacity);
  for (std::byte* b : blocks->get_blocks()) {
    cudaPointerAttributes attr{};
    REQUIRE(cudaPointerGetAttributes(&attr, b) == cudaSuccess);
    REQUIRE(attr.type == cudaMemoryTypeHost);
    REQUIRE(cudaMemcpy(d, b, block_size, cudaMemcpyHostToDevice) == cudaSuccess);
  }
  REQUIRE(cudaFree(d) == cudaSuccess);
}

TEST_CASE("A failed initial pool frees the pools already allocated", "[fixed_size_host]")
{
  auto state        = std::make_shared<counting_host_resource::state>();
  state->fail_after = 10;
  counting_host_resource upstream(state);
  constexpr std::size_t initial_pools = 40;
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;

  REQUIRE_THROWS_AS(fixed_size_host_memory_resource(
                      0, upstream, capacity, capacity, block_size, pool_size, initial_pools),
                    rmm::bad_alloc);
  REQUIRE(state->live.load() == 0);
}
