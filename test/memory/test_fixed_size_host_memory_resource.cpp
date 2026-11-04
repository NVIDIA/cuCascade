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

#include "utils/allocation_counter.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <sched.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <vector>

using namespace cucascade::memory;
using cucascade::test::scoped_allocation_counter;

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
    std::mutex threads_mutex;
    std::set<std::thread::id> threads;  ///< Threads that called allocate()
  };

  explicit counting_host_resource(std::shared_ptr<state> s) : _s(std::move(s)) {}

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    {
      std::lock_guard lock(_s->threads_mutex);
      _s->threads.insert(std::this_thread::get_id());
    }
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

/// One HOST space (id 0) over a counting upstream, so a test can reserve from it and count pools.
std::unique_ptr<memory_reservation_manager> make_counting_host_memory(
  std::shared_ptr<counting_host_resource::state> state,
  std::size_t capacity,
  std::size_t initial_pools)
{
  host_memory_space_config config;
  config.numa_id                    = 0;
  config.memory_capacity            = capacity;
  config.reservation_limit_fraction = 1.0;
  config.block_size                 = block_size;
  config.pool_size                  = pool_size;
  config.initial_number_pools       = initial_pools;
  config.mr_factory_fn              = [state](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{counting_host_resource{state}};
  };
  std::vector<memory_space_config> configs;
  configs.emplace_back(config);
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

}  // namespace

TEST_CASE("Initial pools are all pinned and usable", "[fixed_size_host]")
{
  auto const threads = GENERATE(std::size_t{1}, std::size_t{16});
  numa_region_pinned_host_memory_resource upstream(0);
  constexpr std::size_t initial_pools = 40;  // more pools than allocator threads
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;
  fixed_size_host_memory_resource mr(
    0, upstream, capacity, capacity, block_size, pool_size, initial_pools, threads);

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
  auto const threads = GENERATE(std::size_t{1}, std::size_t{16});
  auto state         = std::make_shared<counting_host_resource::state>();
  state->fail_after  = 10;
  counting_host_resource upstream(state);
  constexpr std::size_t initial_pools = 40;
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;

  REQUIRE_THROWS_AS(
    fixed_size_host_memory_resource(
      0, upstream, capacity, capacity, block_size, pool_size, initial_pools, threads),
    rmm::bad_alloc);
  REQUIRE(state->live.load() == 0);
}

TEST_CASE("Initial pools are allocated on the calling thread by default", "[fixed_size_host]")
{
  auto state = std::make_shared<counting_host_resource::state>();
  counting_host_resource upstream(state);
  constexpr std::size_t initial_pools = 40;
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;
  {
    fixed_size_host_memory_resource mr(
      0, upstream, capacity, capacity, block_size, pool_size, initial_pools);
    REQUIRE(mr.get_total_blocks() == initial_pools * pool_size);
    REQUIRE(state->live.load() == static_cast<int>(initial_pools));
  }
  REQUIRE(state->threads == std::set<std::thread::id>{std::this_thread::get_id()});
  REQUIRE(state->live.load() == 0);
}

TEST_CASE("NUMA pinned allocation restores the caller's CPU affinity", "[fixed_size_host]")
{
  cpu_set_t original;
  REQUIRE(sched_getaffinity(0, sizeof(original), &original) == 0);
  cpu_set_t one_cpu;
  CPU_ZERO(&one_cpu);
  CPU_SET(sched_getcpu(), &one_cpu);
  REQUIRE(sched_setaffinity(0, sizeof(one_cpu), &one_cpu) == 0);

  numa_region_pinned_host_memory_resource upstream(0);
  void* p = upstream.allocate_sync(block_size);
  cpu_set_t after;
  REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
  upstream.deallocate_sync(p, block_size);
  REQUIRE(sched_setaffinity(0, sizeof(original), &original) == 0);

  REQUIRE(CPU_EQUAL(&after, &one_cpu));
}

TEST_CASE("Returning every block after several pool expansions does not allocate",
          "[fixed_size_host]")
{
  constexpr std::size_t n_pools  = 5;  // the initial pool and four expansions
  constexpr std::size_t n_blocks = n_pools * pool_size;
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 2 * n_blocks * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  REQUIRE(res->size() == reserved);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // One pool per allocation, each taken while every earlier block is still out, so the free
  // list is empty at every expansion and only its reserved capacity can hold the returns.
  std::vector<std::byte*> blocks;
  for (std::size_t p = 0; p < n_pools; ++p) {
    auto taken = mr->allocate_multiple_blocks(pool_size * block_size, res.get())->release_blocks();
    blocks.insert(blocks.end(), taken.begin(), taken.end());
  }
  REQUIRE(state->live.load() == static_cast<int>(n_pools));
  REQUIRE(mr->get_total_blocks() == n_blocks);
  REQUIRE(mr->get_free_blocks() == 0);
  REQUIRE(mr->get_total_reserved_bytes() == reserved);
  REQUIRE(mr->get_total_allocated_bytes() == n_blocks * block_size);  // reservation included

  // The first half lies wholly above the reservation, the second half ends inside it.
  std::span<std::byte* const> const all{blocks};
  std::size_t allocations      = 0;
  std::size_t after_first_half = 0;
  {
    scoped_allocation_counter counter;
    mr->return_blocks(all.first(n_blocks / 2), res.get());
    after_first_half = mr->get_total_allocated_bytes();
    mr->return_blocks(all.subspan(n_blocks / 2), res.get());
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  CHECK(after_first_half == (n_blocks - n_blocks / 2) * block_size);
  CHECK(mr->get_free_blocks() == n_blocks);
  CHECK(mr->get_total_blocks() == n_blocks);
  CHECK(mr->get_total_reserved_bytes() == reserved);
  CHECK(mr->get_total_allocated_bytes() == reserved);  // the reservation stays charged

  // The returned blocks are reusable without growing the pool.
  auto again = mr->allocate_multiple_blocks(n_blocks * block_size);
  CHECK(again->size() == n_blocks);
  CHECK(state->live.load() == static_cast<int>(n_pools));
  again.reset();

  res.reset();
  CHECK(mr->get_free_blocks() == n_blocks);
  CHECK(mr->get_total_allocated_bytes() == 0);
}

TEST_CASE("trim_to returns the tail blocks without allocating", "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 2);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  auto allocation = mr->allocate_multiple_blocks(6 * block_size, res.get());
  REQUIRE(mr->get_free_blocks() == 2 * pool_size - 6);
  REQUIRE(mr->get_total_allocated_bytes() == 6 * block_size);

  std::size_t allocations = 0;
  {
    scoped_allocation_counter counter;
    allocation->trim_to(block_size + 1);  // still needs two blocks
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  CHECK(allocation->size() == 2);
  CHECK(mr->get_free_blocks() == 2 * pool_size - 2);
  CHECK(mr->get_total_allocated_bytes() == reserved);  // the two kept fill the reservation

  allocation.reset();
  CHECK(mr->get_free_blocks() == 2 * pool_size);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}

TEST_CASE("A failed pool expansion gives back the blocks the allocation already took",
          "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  state->fail_after              = 1;  // the initial pool succeeds, the first expansion fails
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // Takes the whole first pool, then needs a second one.
  REQUIRE_THROWS_AS(mr->allocate_multiple_blocks((pool_size + 2) * block_size, res.get()),
                    rmm::bad_alloc);
  CHECK(mr->get_free_blocks() == pool_size);
  CHECK(mr->get_total_blocks() == pool_size);
  CHECK(mr->get_total_allocated_bytes() == reserved);

  // The reservation's own count was rolled back too: this fits inside it again.
  auto allocation = mr->allocate_multiple_blocks(reserved, res.get());
  CHECK(allocation->size() == 2);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}
