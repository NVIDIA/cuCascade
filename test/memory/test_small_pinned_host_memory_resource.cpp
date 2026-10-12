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

#include "memory/test_gpu_kernels.cuh"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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

// Keeps queued work pending until the test releases it. Cleanup also releases the callback when
// an assertion aborts the test, so stream destruction cannot leave it blocked.
class pending_stream_work {
 public:
  explicit pending_stream_work(rmm::cuda_stream const& stream) : stream_(stream)
  {
    CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(
      stream_.value(),
      [](void* context) {
        auto& gate = *static_cast<pending_stream_work*>(context);
        gate.released_.wait(false);
      },
      this));
  }

  pending_stream_work(pending_stream_work const&)            = delete;
  pending_stream_work& operator=(pending_stream_work const&) = delete;
  pending_stream_work(pending_stream_work&&)                 = delete;
  pending_stream_work& operator=(pending_stream_work&&)      = delete;

  ~pending_stream_work()
  {
    release();
    stream_.synchronize_no_throw();
  }

  void release() noexcept
  {
    released_.store(true);
    released_.notify_all();
  }

 private:
  rmm::cuda_stream const& stream_;
  std::atomic<bool> released_{false};
};

}  // namespace

TEST_CASE("Zero-byte allocation returns nullptr", "[small_pinned]")
{
  test_fixture f;
  auto* ptr = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 0);
  REQUIRE(ptr == nullptr);
}

TEST_CASE("Allocate and deallocate each slab size", "[small_pinned]")
{
  test_fixture f;
  constexpr std::array<std::size_t, 5> sizes{512, 1024, 2048, 4096, 8192};

  for (auto sz : sizes) {
    SECTION("slab size " + std::to_string(sz))
    {
      auto* ptr = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, sz);
      REQUIRE(ptr != nullptr);
      // Write to the memory to verify it is accessible
      std::memset(ptr, 0xAB, sz);
      f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, ptr, sz);
    }
  }
}

TEST_CASE("Sub-slab sizes round up correctly", "[small_pinned]")
{
  test_fixture f;
  // Allocate 1 byte — should get a 512-byte slab
  auto* p1 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 1);
  REQUIRE(p1 != nullptr);
  std::memset(p1, 0, 512);  // must be able to write 512 bytes
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p1, 1);

  // Allocate 513 bytes — should get a 1024-byte slab
  auto* p2 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 513);
  REQUIRE(p2 != nullptr);
  std::memset(p2, 0, 1024);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p2, 513);

  // Allocate 4097 bytes — should get an 8192-byte slab
  auto* p3 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 4097);
  REQUIRE(p3 != nullptr);
  std::memset(p3, 0, 8192);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p3, 4097);
}

TEST_CASE("Large allocation is served by the bucketed pinned path", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t big = small_pinned_host_memory_resource::MAX_SLAB_SIZE + 1;
  auto* ptr = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, big);
  REQUIRE(ptr != nullptr);
  std::memset(ptr, 0xCD, big);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, ptr, big);
}

TEST_CASE("Deallocate nullptr is safe", "[small_pinned]")
{
  test_fixture f;
  // Should not crash
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, nullptr, 0);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, nullptr, 512);
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
    auto* p = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, alloc_size);
    REQUIRE(p != nullptr);
    REQUIRE(ptrs.insert(p).second);  // must be unique
    allocs.push_back(p);
  }

  for (auto* p : allocs) {
    f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, alloc_size);
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
    auto* p = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, alloc_size);
    REQUIRE(p != nullptr);
    allocs.push_back(p);
  }

  for (auto* p : allocs) {
    f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, alloc_size);
  }
}

TEST_CASE("Freed slabs are reused", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t alloc_size = 1024;

  auto* p1 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, alloc_size);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p1, alloc_size);

  // After deallocation the pointer should be reused (returned from the free list)
  auto* p2 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, alloc_size);
  REQUIRE(p2 == p1);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p2, alloc_size);
}

TEST_CASE("do_is_equal identity check", "[small_pinned]")
{
  test_fixture f;
  REQUIRE(f.slab_mr == f.slab_mr);

  // A second instance should not be equal
  small_pinned_host_memory_resource other{f.upstream};
  REQUIRE_FALSE(f.slab_mr == other);
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
        auto* p =
          f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, alloc_size);
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
      f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, alloc_size);
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
    auto* p = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, sz);
    REQUIRE(p != nullptr);
    std::memset(p, 0xFF, sz);
    allocs.push_back({p, sz});
  }

  // Free in reverse order
  for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
    f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, it->ptr, it->size);
  }
}

TEST_CASE("Large allocations do not interfere with slab pool", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t big_size   = 16384;
  constexpr std::size_t small_size = 512;

  auto* big = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, big_size);
  REQUIRE(big != nullptr);

  // Allocate a small chunk (goes to slab pool)
  auto* small1 =
    f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, small_size);
  REQUIRE(small1 != nullptr);

  // Free the large one
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, big, big_size);

  // Small allocations should still work
  auto* small2 =
    f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, small_size);
  REQUIRE(small2 != nullptr);

  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, small1, small_size);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, small2, small_size);
}

TEST_CASE("large_bucket_size_for rounds to power-of-two buckets", "[small_pinned]")
{
  using mr = small_pinned_host_memory_resource;
  REQUIRE(mr::large_bucket_size_for(mr::MAX_SLAB_SIZE + 1) == mr::MIN_LARGE_BUCKET);
  REQUIRE(mr::large_bucket_size_for(16384) == 16384);
  REQUIRE(mr::large_bucket_size_for(16385) == 32768);
  REQUIRE(mr::large_bucket_size_for(100000) == 131072);
  REQUIRE(mr::large_bucket_size_for(std::size_t{1} << 20) == std::size_t{1} << 20);
}

TEST_CASE("large_allocation_size rounds only cacheable requests", "[small_pinned]")
{
  // allocate sizes its cudaHostAlloc calls with this same helper, so these checks pin down the
  // physical size behavior: bucket rounding for cacheable requests, the exact request otherwise.
  test_fixture f;
  using mr_t = small_pinned_host_memory_resource;
  REQUIRE(f.slab_mr.large_allocation_size(100000) == 131072);
  REQUIRE(f.slab_mr.large_allocation_size(mr_t::MIN_LARGE_BUCKET) == mr_t::MIN_LARGE_BUCKET);
  REQUIRE(f.slab_mr.large_allocation_size(mr_t::DEFAULT_LARGE_CACHE_LIMIT) ==
          mr_t::DEFAULT_LARGE_CACHE_LIMIT);
  REQUIRE(f.slab_mr.large_allocation_size(mr_t::DEFAULT_LARGE_CACHE_LIMIT + 1) ==
          mr_t::DEFAULT_LARGE_CACHE_LIMIT + 1);

  small_pinned_host_memory_resource tiny{f.upstream, 64 * 1024};
  REQUIRE(tiny.large_allocation_size(9000) == 16384);
  REQUIRE(tiny.large_allocation_size(100000) == 100000);
}

TEST_CASE("Large allocations are cached and reused", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t big = 100000;  // rounds up to a 128 KB bucket

  auto* slab = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 512);
  auto* p1   = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, big);
  REQUIRE(p1 != nullptr);
  REQUIRE(p1 != slab);
  CUCASCADE_CUDA_TRY(::cudaStreamSynchronize(cudaStreamDefault));
  std::memset(p1, 0xEF, big);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p1, big);

  // The freed buffer is cached, so an allocation of the same size gets it back.
  auto* p2 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, big);
  REQUIRE(p2 == p1);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p2, big);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, slab, 512);
}

TEST_CASE("Large cache serves within a bucket but not across buckets", "[small_pinned]")
{
  test_fixture f;

  // 12 KB and 40 KB round to different buckets (16 KB and 64 KB): no cross-serving. The 12 KB
  // buffer stays cached (still allocated), so the 64 KB miss cannot alias it.
  auto* p16 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 12 * 1024);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p16, 12 * 1024);
  auto* p64 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 40 * 1024);
  REQUIRE(p64 != p16);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p64, 40 * 1024);

  // 33 KB and 40 KB round to the same 64 KB bucket: the cached buffer is reused.
  auto* p1 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 33 * 1024);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p1, 33 * 1024);
  auto* p2 = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 40 * 1024);
  REQUIRE(p2 == p1);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p2, 40 * 1024);
}

TEST_CASE("large_cache_bytes reflects bucket-size accounting", "[small_pinned]")
{
  test_fixture f;
  REQUIRE(f.slab_mr.large_cache_bytes() == 0);

  // 100000 bytes occupy a 128 KB bucket; the cache counts the bucket, not the request.
  constexpr std::size_t bytes = 100000;
  auto* p = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes);
  REQUIRE(f.slab_mr.large_cache_bytes() == 0);  // live buffers are not cached
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, bytes);
  REQUIRE(f.slab_mr.large_cache_bytes() == 128 * 1024);

  // A size one past a bucket boundary lands in the next bucket.
  auto* q = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 16 * 1024 + 1);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, q, 16 * 1024 + 1);
  REQUIRE(f.slab_mr.large_cache_bytes() == 128 * 1024 + 32 * 1024);

  // Cache hits remove the bucket from the total.
  auto* r = f.slab_mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes);
  REQUIRE(r == p);
  REQUIRE(f.slab_mr.large_cache_bytes() == 32 * 1024);
  f.slab_mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, r, bytes);
}

TEST_CASE("Large cache respects its cap and evicts oldest entries", "[small_pinned]")
{
  test_fixture f;
  constexpr std::size_t cap = 64 * 1024;
  small_pinned_host_memory_resource mr{f.upstream, cap};

  constexpr std::size_t bytes = 9000;  // bucket = 16 KB, so the cap holds four buffers
  std::array<void*, 5> ptrs{};
  for (auto& p : ptrs) {
    p = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes);
    REQUIRE(p != nullptr);
  }
  for (auto* p : ptrs) {
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, bytes);
    REQUIRE(mr.large_cache_bytes() <= cap);
  }
  // Four 16 KB buckets fill the cap; caching the fifth evicted the oldest entry (ptrs[0]).
  REQUIRE(mr.large_cache_bytes() == cap);

  std::set<void*> reused;
  for (int i = 0; i < 4; ++i) {
    reused.insert(mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes));
  }
  REQUIRE(mr.large_cache_bytes() == 0);
  REQUIRE(reused == std::set<void*>{ptrs[1], ptrs[2], ptrs[3], ptrs[4]});
  for (auto* p : reused) {
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, bytes);
  }
}

TEST_CASE("Buffers larger than the whole cap are freed, not cached", "[small_pinned]")
{
  test_fixture f;
  small_pinned_host_memory_resource mr{f.upstream, 64 * 1024};

  // Seed the cache so we can also verify the oversized free evicts nothing.
  auto* seeded = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 9000);
  mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, seeded, 9000);
  auto const cached_before = mr.large_cache_bytes();
  REQUIRE(cached_before == 16 * 1024);

  constexpr std::size_t huge = 128 * 1024;  // bucket exceeds the whole 64 KB cap
  auto* p = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, huge);
  REQUIRE(p != nullptr);
  mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, huge);
  REQUIRE(mr.large_cache_bytes() == cached_before);
}

TEST_CASE("Never-cacheable sizes round trip without polluting the cache", "[small_pinned]")
{
  // A non-power-of-two size whose bucket exceeds the whole cap is allocated at exactly the
  // requested size and freed on deallocate rather than cached.
  test_fixture f;
  small_pinned_host_memory_resource mr{f.upstream, 64 * 1024};

  constexpr std::size_t bytes = 100000;  // 128 KB bucket, beyond the 64 KB cap
  auto* p = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes);
  REQUIRE(p != nullptr);
  std::memset(p, 0x5A, bytes);  // the full requested size must be usable
  mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p, bytes);
  REQUIRE(mr.large_cache_bytes() == 0);

  auto* q = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes);
  REQUIRE(q != nullptr);
  mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, q, bytes);
}

TEST_CASE("Slab and large reuse work on a real stream", "[small_pinned]")
{
  // Exercises the deallocate-time event record and allocate-time wait on a genuine
  // (non-default) stream, driving the device-keyed event pool end to end.
  test_fixture f;
  rmm::cuda_stream stream;

  constexpr std::size_t slab_bytes = 2048;
  auto* s1                         = f.slab_mr.allocate(stream, slab_bytes);
  REQUIRE(s1 != nullptr);
  stream.synchronize();
  std::memset(s1, 0x11, slab_bytes);
  f.slab_mr.deallocate(stream, s1, slab_bytes);
  auto* s2 = f.slab_mr.allocate(stream, slab_bytes);
  REQUIRE(s2 == s1);
  f.slab_mr.deallocate(stream, s2, slab_bytes);

  constexpr std::size_t big_bytes = 100000;
  auto* p1                        = f.slab_mr.allocate(stream, big_bytes);
  REQUIRE(p1 != nullptr);
  stream.synchronize();
  std::memset(p1, 0x22, big_bytes);
  f.slab_mr.deallocate(stream, p1, big_bytes);
  auto* p2 = f.slab_mr.allocate(stream, big_bytes);
  REQUIRE(p2 == p1);
  f.slab_mr.deallocate(stream, p2, big_bytes);
  stream.synchronize();
}

TEST_CASE("Destruction with a populated large cache is safe", "[small_pinned]")
{
  test_fixture f;
  {
    // Deallocate -> allocate -> deallocate cycles on both paths recycle events through the idle
    // pool, so the events destroyed with this resource include recycled ones, not only fresh
    // ones, attached to a free slab as well as to cached large buffers.
    small_pinned_host_memory_resource mr{f.upstream};
    auto* s = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 512);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, s, 512);
    auto* s2 = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 512);
    REQUIRE(s2 == s);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, s2, 512);

    auto* p1 = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 10 * 1024);
    auto* p2 = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 100 * 1024);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p1, 10 * 1024);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p2, 100 * 1024);
    auto* p3 = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 10 * 1024);
    REQUIRE(p3 == p1);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p3, 10 * 1024);
    REQUIRE(mr.large_cache_bytes() == 16 * 1024 + 128 * 1024);
  }
  {
    // A 32 KB insertion into a cap-full cache of two 16 KB entries evicts both victims but
    // re-acquires only one event, so this resource is destroyed while its pool holds an idle
    // recycled event alongside the events still attached to a free slab and a cached buffer.
    small_pinned_host_memory_resource mr{f.upstream, 32 * 1024};
    auto* s = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 512);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, s, 512);
    auto* a = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 9000);
    auto* b = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 9000);
    auto* c = mr.allocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, 20 * 1024);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, a, 9000);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, b, 9000);
    mr.deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, c, 20 * 1024);
    REQUIRE(mr.large_cache_bytes() == 32 * 1024);
  }
  SUCCEED("resources destroyed with cached buffers, attached events, and pooled idle events");
}

TEST_CASE("Bucket rounding handles the largest representable sizes", "[small_pinned]")
{
  using mr                  = small_pinned_host_memory_resource;
  constexpr auto max_bucket = std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
  REQUIRE(mr::large_bucket_size_for(max_bucket) == max_bucket);
  REQUIRE(mr::large_bucket_size_for(max_bucket + 1) == max_bucket + 1);
  REQUIRE(mr::large_bucket_size_for(std::numeric_limits<std::size_t>::max()) ==
          std::numeric_limits<std::size_t>::max());
}

TEST_CASE("Disabled large cache preserves host and device accessibility", "[small_pinned]")
{
  test_fixture f;
  auto const limit =
    GENERATE(std::size_t{0}, small_pinned_host_memory_resource::MIN_LARGE_BUCKET - 1);
  small_pinned_host_memory_resource mr{f.upstream, limit};
  constexpr std::size_t bytes = 100000;
  REQUIRE(mr.large_allocation_size(bytes) == bytes);
  auto* ptr = mr.allocate_sync(bytes);
  std::memset(ptr, 0x5A, bytes);
  void* device_ptr = nullptr;
  CUCASCADE_CUDA_TRY(::cudaHostGetDevicePointer(&device_ptr, ptr, 0));
  REQUIRE(device_ptr != nullptr);
  std::uint64_t checksum = 0;
  CUCASCADE_CUDA_TRY(performGpuMemoryWork(device_ptr, bytes, &checksum));
  REQUIRE(checksum != 0);
  std::uint32_t first_word = 0;
  std::memcpy(&first_word, ptr, sizeof(first_word));
  REQUIRE(first_word == 0xDEADBEEF);
  mr.deallocate_sync(ptr, bytes);
  REQUIRE(mr.large_cache_bytes() == 0);
}

TEST_CASE("Large FIFO eviction orders entries across buckets", "[small_pinned]")
{
  test_fixture f;
  small_pinned_host_memory_resource mr{f.upstream, 64 * 1024};
  auto* oldest         = mr.allocate_sync(9000);
  auto* next           = mr.allocate_sync(20000);
  auto* retained_small = mr.allocate_sync(9000);
  auto* retained_large = mr.allocate_sync(20000);
  mr.deallocate_sync(oldest, 9000);
  mr.deallocate_sync(next, 20000);
  mr.deallocate_sync(retained_small, 9000);
  mr.deallocate_sync(retained_large, 20000);
  REQUIRE(mr.large_cache_bytes() == 48 * 1024);
  auto* small = mr.allocate_sync(9000);
  auto* large = mr.allocate_sync(20000);
  REQUIRE(small == retained_small);
  REQUIRE(large == retained_large);
  REQUIRE(mr.large_cache_bytes() == 0);
  mr.deallocate_sync(small, 9000);
  mr.deallocate_sync(large, 20000);
}

TEST_CASE("Reuse orders pending DMA across allocation streams", "[small_pinned][stream_ordering]")
{
  auto const bytes = GENERATE(std::size_t{2048}, std::size_t{100000});
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{bytes, producer.view()};
  auto* original = f.slab_mr.allocate_sync(bytes);
  std::memset(original, 0x31, bytes);
  pending_stream_work gate{producer};
  CUCASCADE_CUDA_TRY(
    ::cudaMemcpyAsync(device.data(), original, bytes, cudaMemcpyHostToDevice, producer.value()));
  f.slab_mr.deallocate(producer, original, bytes);
  auto* reused = f.slab_mr.allocate(consumer, bytes);
  REQUIRE(reused == original);
  REQUIRE(::cudaStreamQuery(consumer.value()) == cudaErrorNotReady);
  gate.release();
  consumer.synchronize();
  std::memset(reused, 0xA7, bytes);

  std::vector<std::byte> copied(bytes);
  CUCASCADE_CUDA_TRY(::cudaMemcpy(copied.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x31}; }));
  f.slab_mr.deallocate(consumer, reused, bytes);
}

TEST_CASE("Synchronous allocation waits for pending DMA before host reuse",
          "[small_pinned][stream_ordering]")
{
  auto const bytes = GENERATE(std::size_t{2048}, std::size_t{100000});
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{bytes, producer.view()};
  auto* original = f.slab_mr.allocate_sync(bytes);
  std::memset(original, 0x43, bytes);
  pending_stream_work gate{producer};
  CUCASCADE_CUDA_TRY(
    ::cudaMemcpyAsync(device.data(), original, bytes, cudaMemcpyHostToDevice, producer.value()));
  f.slab_mr.deallocate(producer, original, bytes);
  std::atomic<bool> observed_wait{false};
  auto const current_device = rmm::get_current_cuda_device();
  std::jthread release_thread{[&gate, &observed_wait, current_device] {
    rmm::cuda_set_device_raii device_guard{current_device};
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (std::chrono::steady_clock::now() < deadline) {
      if (::cudaStreamQuery(cudaStreamDefault) == cudaErrorNotReady) {
        observed_wait.store(true);
        break;
      }
      std::this_thread::yield();
    }
    gate.release();
  }};
  auto* reused = f.slab_mr.allocate_sync(bytes);
  release_thread.join();
  REQUIRE(observed_wait.load());
  REQUIRE(reused == original);
  std::memset(reused, 0xA7, bytes);
  std::vector<std::byte> copied(bytes);
  CUCASCADE_CUDA_TRY(::cudaMemcpy(copied.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x43}; }));
  f.slab_mr.deallocate_sync(reused, bytes);
}

TEST_CASE("Event ownership follows the freeing stream's device", "[small_pinned][multi_gpu]")
{
  int device_count = 0;
  CUCASCADE_CUDA_TRY(::cudaGetDeviceCount(&device_count));
  if (device_count < 2) { SKIP("requires two CUDA devices"); }
  auto const bytes = GENERATE(std::size_t{2048}, std::size_t{100000});
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{0}};
  test_fixture f;
  rmm::cuda_stream device_zero_stream{rmm::cuda_stream::flags::non_blocking};
  auto* original = f.slab_mr.allocate_sync(bytes);
  {
    rmm::cuda_set_device_raii other_device{rmm::cuda_device_id{1}};
    f.slab_mr.deallocate(device_zero_stream, original, bytes);
  }
  auto* reused = f.slab_mr.allocate(device_zero_stream, bytes);
  REQUIRE(reused == original);
  device_zero_stream.synchronize();
  f.slab_mr.deallocate(device_zero_stream, reused, bytes);
}
