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

#include <cucascade/io/cache/types.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace {

using cucascade::io::cache::buffer_pool;
using cucascade::memory::fixed_size_host_memory_resource;
using cucascade::memory::host_memory_space_config;
using cucascade::memory::memory_reservation_manager;
using cucascade::memory::memory_space_config;
using cucascade::memory::Tier;
using cucascade::test::scoped_allocation_counter;

constexpr std::size_t chunk           = 64UL << 10;
constexpr std::size_t blocks_per_pool = 4;
constexpr std::size_t arena_capacity  = 64 * chunk;
constexpr std::size_t arena_reserved  = 2 * chunk;  ///< Each arena's prefetch reservation

/// Plain malloc upstream: these tests need neither pinned memory nor a NUMA binding.
class malloc_host_resource {
 public:
  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    void* p = std::malloc(bytes);
    if (p == nullptr) { throw std::bad_alloc{}; }
    return p;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::free(p);
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

  bool operator==(malloc_host_resource const&) const noexcept { return true; }

  friend void get_property(malloc_host_resource const&, ::cuda::mr::device_accessible) noexcept {}
  friend void get_property(malloc_host_resource const&, ::cuda::mr::host_accessible) noexcept {}
};

/// Two HOST arenas, ids 0 and 1, each starting with one pool of @c blocks_per_pool chunks.
std::unique_ptr<memory_reservation_manager> two_arena_memory()
{
  std::vector<memory_space_config> configs;
  for (int const id : {0, 1}) {
    host_memory_space_config config;
    config.numa_id                    = id;
    config.memory_capacity            = arena_capacity;
    config.reservation_limit_fraction = 1.0;
    config.block_size                 = chunk;
    config.pool_size                  = blocks_per_pool;
    config.initial_number_pools       = 1;
    config.mr_factory_fn              = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{malloc_host_resource{}};
    };
    configs.emplace_back(config);
  }
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

fixed_size_host_memory_resource& arena_mr(memory_reservation_manager& memory, int id)
{
  auto* space = memory.get_memory_space(Tier::HOST, id);
  REQUIRE(space != nullptr);
  auto* mr = space->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);
  return *mr;
}

}  // namespace

TEST_CASE("deallocate_bulk returns chunks to their own arena without allocating",
          "[cache][buffer_pool]")
{
  auto memory = two_arena_memory();
  buffer_pool pool(*memory,
                   static_cast<double>(arena_reserved) / static_cast<double>(arena_capacity));
  auto& a = arena_mr(*memory, 0);
  auto& b = arena_mr(*memory, 1);
  REQUIRE(pool.chunk_size() == chunk);
  REQUIRE(pool.reservation_size_for_prefetching() == 2 * arena_reserved);
  REQUIRE(a.get_total_allocated_bytes() == arena_reserved);
  REQUIRE(b.get_total_allocated_bytes() == arena_reserved);

  // A grows pool by pool while every earlier chunk is still out; B stays inside its reservation.
  std::vector<std::byte*> a_chunks;
  for (int round = 0; round < 3; ++round) {
    auto taken = pool.allocate_bulk_from(blocks_per_pool, 0);
    REQUIRE(taken.size() == blocks_per_pool);
    a_chunks.insert(a_chunks.end(), taken.begin(), taken.end());
  }
  auto b_chunks = pool.allocate_bulk_from(1, 1);
  REQUIRE(b_chunks.size() == 1);

  auto const n_a    = a_chunks.size();
  auto const b_free = b.get_free_blocks();
  REQUIRE(a.get_total_blocks() == n_a);
  REQUIRE(a.get_free_blocks() == 0);
  REQUIRE(a.get_total_allocated_bytes() == n_a * chunk);
  REQUIRE(b.get_total_allocated_bytes() == arena_reserved);
  REQUIRE(pool.releasable_bytes(0) == n_a * chunk - arena_reserved);
  REQUIRE(pool.releasable_bytes(1) == 0);
  REQUIRE(pool.total_allocated_chunks() == n_a + 1);

  std::size_t allocations = 0;
  {
    scoped_allocation_counter counter;
    pool.deallocate_bulk(std::span<std::byte* const>(a_chunks), 0);
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  // Every chunk is back on A's free list; A keeps its reservation charged, nothing more.
  CHECK(a.get_free_blocks() == n_a);
  CHECK(a.get_total_blocks() == n_a);
  CHECK(a.get_total_allocated_bytes() == arena_reserved);
  CHECK(pool.releasable_bytes(0) == 0);
  // B is untouched.
  CHECK(b.get_free_blocks() == b_free);
  CHECK(b.get_total_allocated_bytes() == arena_reserved);
  CHECK(pool.total_allocated_chunks() == 1);

  // B's chunk sat inside B's reservation, so handing it back credits no HOST budget.  The
  // vector overload takes the same path and consumes its list.
  {
    scoped_allocation_counter counter;
    pool.deallocate_bulk(std::move(b_chunks), 1);
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  CHECK(b_chunks.empty());  // NOLINT(bugprone-use-after-move): documented to leave it empty
  CHECK(b.get_free_blocks() == b_free + 1);
  CHECK(b.get_total_allocated_bytes() == arena_reserved);
  CHECK(a.get_free_blocks() == n_a);
  CHECK(pool.total_allocated_chunks() == 0);
  CHECK(pool.releasable_bytes() == 0);
}
