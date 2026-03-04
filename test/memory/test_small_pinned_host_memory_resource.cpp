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

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <thread>
#include <vector>

using namespace cucascade::memory;

namespace {

// Helper to create a fixed_size_host_memory_resource backed by pinned host memory.
// Uses a small block size (64 KB) and pool to keep test memory footprint low.
struct test_fixture {
  static constexpr std::size_t block_size    = 64 * 1024;  // 64 KB
  static constexpr std::size_t pool_size     = 16;
  static constexpr std::size_t initial_pools = 1;
  static constexpr std::size_t mem_limit     = 16 * 1024 * 1024;  // 16 MB
  static constexpr std::size_t capacity      = 16 * 1024 * 1024;  // 16 MB

  rmm::mr::pinned_host_memory_resource pinned_mr;
  fixed_size_host_memory_resource upstream{
    0, pinned_mr, mem_limit, capacity, block_size, pool_size, initial_pools};
  small_pinned_host_memory_resource slab_mr{upstream};
};

}  // namespace

TEST_CASE("Zero-byte allocation returns nullptr", "[small_pinned]")
{
  test_fixture f;
  auto* ptr = f.slab_mr.allocate(rmm::cuda_stream_view{}, 0);
  REQUIRE(ptr == nullptr);
}

TEST_CASE("Allocate and deallocate each slab size", "[small_pinned]")
{
  test_fixture f;
  constexpr std::array<std::size_t, 5> sizes{512, 1024, 2048, 4096, 8192};

  for (auto sz : sizes) {
    SECTION("slab size " + std::to_string(sz))
    {
      auto* ptr = f.slab_mr.allocate(rmm::cuda_stream_view{}, sz);
      REQUIRE(ptr != nullptr);
      // Write to the memory to verify it is accessible
      std::memset(ptr, 0xAB, sz);
      f.slab_mr.deallocate(rmm::cuda_stream_view{}, ptr, sz);
    }
  }
}

TEST_CASE("Sub-slab sizes round up correctly", "[small_pinned]")
{
  test_fixture f;
  // Allocate 1 byte — should get a 512-byte slab
  auto* p1 = f.slab_mr.allocate(rmm::cuda_stream_view{}, 1);
  REQUIRE(p1 != nullptr);
  std::memset(p1, 0, 512);  // must be able to write 512 bytes
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, p1, 1);

  // Allocate 513 bytes — should get a 1024-byte slab
  auto* p2 = f.slab_mr.allocate(rmm::cuda_stream_view{}, 513);
  REQUIRE(p2 != nullptr);
  std::memset(p2, 0, 1024);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, p2, 513);

  // Allocate 4097 bytes — should get an 8192-byte slab
  auto* p3 = f.slab_mr.allocate(rmm::cuda_stream_view{}, 4097);
  REQUIRE(p3 != nullptr);
  std::memset(p3, 0, 8192);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, p3, 4097);
}

TEST_CASE("Large allocation falls back to malloc", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t big = small_pinned_host_memory_resource::MAX_SLAB_SIZE + 1;
  auto* ptr                 = f.slab_mr.allocate(rmm::cuda_stream_view{}, big);
  REQUIRE(ptr != nullptr);
  std::memset(ptr, 0xCD, big);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, ptr, big);
}

TEST_CASE("Deallocate nullptr is safe", "[small_pinned]")
{
  test_fixture f;
  // Should not crash
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, nullptr, 0);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, nullptr, 512);
}

TEST_CASE("Multiple allocations return distinct pointers", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t alloc_size = 512;
  constexpr int count              = 64;

  std::set<void*> ptrs;
  std::vector<void*> allocs;
  allocs.reserve(count);

  for (int i = 0; i < count; ++i) {
    auto* p = f.slab_mr.allocate(rmm::cuda_stream_view{}, alloc_size);
    REQUIRE(p != nullptr);
    REQUIRE(ptrs.insert(p).second);  // must be unique
    allocs.push_back(p);
  }

  for (auto* p : allocs) {
    f.slab_mr.deallocate(rmm::cuda_stream_view{}, p, alloc_size);
  }
}

TEST_CASE("Pool expansion provides more slabs", "[small_pinned]")
{
  test_fixture f;
  // The upstream block is 64 KB, so for 512-byte slabs we get 128 per block.
  // Allocate more than one block's worth to force pool expansion.
  constexpr std::size_t alloc_size = 512;
  constexpr int count              = 256;  // > 128 slabs per 64 KB block

  std::vector<void*> allocs;
  allocs.reserve(count);

  for (int i = 0; i < count; ++i) {
    auto* p = f.slab_mr.allocate(rmm::cuda_stream_view{}, alloc_size);
    REQUIRE(p != nullptr);
    allocs.push_back(p);
  }

  for (auto* p : allocs) {
    f.slab_mr.deallocate(rmm::cuda_stream_view{}, p, alloc_size);
  }
}

TEST_CASE("Freed slabs are reused", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t alloc_size = 1024;

  auto* p1 = f.slab_mr.allocate(rmm::cuda_stream_view{}, alloc_size);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, p1, alloc_size);

  // After deallocation the pointer should be reused (returned from the free list)
  auto* p2 = f.slab_mr.allocate(rmm::cuda_stream_view{}, alloc_size);
  REQUIRE(p2 == p1);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, p2, alloc_size);
}

TEST_CASE("do_is_equal identity check", "[small_pinned]")
{
  test_fixture f;
  REQUIRE(f.slab_mr.is_equal(f.slab_mr));

  // A second instance should not be equal
  small_pinned_host_memory_resource other{f.upstream};
  REQUIRE_FALSE(f.slab_mr.is_equal(other));
}

TEST_CASE("Concurrent allocations are thread-safe", "[small_pinned][threading]")
{
  test_fixture f;
  constexpr int num_threads        = 8;
  constexpr int allocs_per_thread  = 32;
  constexpr std::size_t alloc_size = 2048;

  std::vector<std::thread> threads;
  std::vector<std::vector<void*>> per_thread_allocs(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      per_thread_allocs[t].reserve(allocs_per_thread);
      for (int i = 0; i < allocs_per_thread; ++i) {
        auto* p = f.slab_mr.allocate(rmm::cuda_stream_view{}, alloc_size);
        REQUIRE(p != nullptr);
        // Touch the memory
        std::memset(p, static_cast<int>(t), alloc_size);
        per_thread_allocs[t].push_back(p);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  // Verify all pointers are unique across all threads
  std::set<void*> all_ptrs;
  for (auto& vec : per_thread_allocs) {
    for (auto* p : vec) {
      REQUIRE(all_ptrs.insert(p).second);
    }
  }

  // Deallocate everything
  for (auto& vec : per_thread_allocs) {
    for (auto* p : vec) {
      f.slab_mr.deallocate(rmm::cuda_stream_view{}, p, alloc_size);
    }
  }
}

TEST_CASE("Mixed slab sizes allocated and freed correctly", "[small_pinned]")
{
  test_fixture f;
  struct alloc_record {
    void* ptr;
    std::size_t size;
  };
  std::vector<alloc_record> allocs;

  // Allocate a mix of sizes
  constexpr std::array<std::size_t, 7> sizes{64, 256, 512, 1000, 2048, 4096, 8192};
  for (auto sz : sizes) {
    auto* p = f.slab_mr.allocate(rmm::cuda_stream_view{}, sz);
    REQUIRE(p != nullptr);
    std::memset(p, 0xFF, sz);
    allocs.push_back({p, sz});
  }

  // Free in reverse order
  for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
    f.slab_mr.deallocate(rmm::cuda_stream_view{}, it->ptr, it->size);
  }
}

TEST_CASE("Large allocations do not interfere with slab pool", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t big_size   = 16384;
  constexpr std::size_t small_size = 512;

  // Allocate a large chunk (goes to malloc)
  auto* big = f.slab_mr.allocate(rmm::cuda_stream_view{}, big_size);
  REQUIRE(big != nullptr);

  // Allocate a small chunk (goes to slab pool)
  auto* small1 = f.slab_mr.allocate(rmm::cuda_stream_view{}, small_size);
  REQUIRE(small1 != nullptr);

  // Free the large one
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, big, big_size);

  // Small allocations should still work
  auto* small2 = f.slab_mr.allocate(rmm::cuda_stream_view{}, small_size);
  REQUIRE(small2 != nullptr);

  f.slab_mr.deallocate(rmm::cuda_stream_view{}, small1, small_size);
  f.slab_mr.deallocate(rmm::cuda_stream_view{}, small2, small_size);
}
