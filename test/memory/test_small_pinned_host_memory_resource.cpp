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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
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
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <set>
#include <stop_token>
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
  explicit pending_stream_work(rmm::cuda_stream const& stream) : _stream(stream)
  {
    CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(
      _stream.value(),
      [](void* context) {
        auto& gate = *static_cast<pending_stream_work*>(context);
        gate._released.wait(false);
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
    _stream.synchronize_no_throw();
  }

  void release() noexcept
  {
    _released.store(true);
    _released.notify_all();
  }

 private:
  rmm::cuda_stream const& _stream;
  std::atomic<bool> _released{false};
};

// Takes every slab of `bytes` that a pool still has free after it handed out `taken` slabs and
// freed none: the unused rest of the last upstream block it carved. The next request of that size
// can then only be served by a slab freed afterwards.
[[nodiscard]] std::vector<void*> hold_remaining_slabs(small_pinned_host_memory_resource& mr,
                                                      std::size_t bytes,
                                                      std::size_t taken)
{
  auto const slabs_per_block = test_fixture::block_size / bytes;
  std::vector<void*> held;
  for (std::size_t i = taken % slabs_per_block; i != 0 && i < slabs_per_block; ++i) {
    held.push_back(mr.allocate_sync(bytes));
  }
  return held;
}

// Synchronizes `stream` and reports whether that finished while `gate` still held its work back.
// If the stream wrongly depends on the gated work, the gate opens at the deadline, so the caller's
// assertion fails instead of the test hanging.
[[nodiscard]] bool synchronizes_before_release(rmm::cuda_stream const& stream,
                                               pending_stream_work& gate)
{
  std::atomic<bool> deadline_reached{false};
  std::jthread watchdog{[&](std::stop_token stop) {
    auto const give_up_at = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < give_up_at) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    if (!stop.stop_requested()) {
      deadline_reached.store(true);
      gate.release();
    }
  }};
  stream.synchronize();
  watchdog.request_stop();
  watchdog.join();
  return !deadline_reached.load();
}

// Leaves a slab pool with two free slabs: an older one whose work has finished, and the most
// recently freed one, which a gated copy on `producer_device` still reads. Allocates twice on
// device 0 and checks that the first request takes the ready slab without depending on the gated
// work, and that the second falls back to the pending slab and waits for the copy.
void check_ready_slab_is_preferred(int producer_device)
{
  constexpr std::size_t bytes = 2048;
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{0}};
  test_fixture f;
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto producer = [producer_device] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{producer_device}};
    return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
  }();
  auto producer_data = [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{producer_device}};
    return rmm::device_buffer{bytes, ::cuda::stream_ref{producer.value()}};
  }();
  auto* older           = f.slab_mr.allocate_sync(bytes);
  auto* newest          = f.slab_mr.allocate_sync(bytes);
  auto const held_slabs = hold_remaining_slabs(f.slab_mr, bytes, 2);

  f.slab_mr.deallocate(consumer, older, bytes);
  consumer.synchronize();
  std::memset(newest, 0x31, bytes);
  // Declared after both devices' buffers and streams so a failed assertion releases the gate first.
  std::optional<pending_stream_work> gate;
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{producer_device}};
    gate.emplace(producer);
    CUCASCADE_CUDA_TRY(::cudaMemcpyAsync(
      producer_data.data(), newest, bytes, cudaMemcpyHostToDevice, producer.value()));
    f.slab_mr.deallocate(producer, newest, bytes);
  }

  auto* first = f.slab_mr.allocate(consumer, bytes);
  REQUIRE(first == older);
  // Finding a slab still pending is not a failure, so it leaves no CUDA error behind.
  REQUIRE(::cudaGetLastError() == cudaSuccess);
  REQUIRE(synchronizes_before_release(consumer, *gate));
  std::memset(first, 0xA7, bytes);

  auto* second = f.slab_mr.allocate(consumer, bytes);
  REQUIRE(second == newest);
  REQUIRE(::cudaStreamQuery(consumer.value()) == cudaErrorNotReady);
  gate->release();
  consumer.synchronize();
  std::memset(second, 0xA7, bytes);

  // The gated copy read the slab before its second owner could write to it.
  std::vector<std::byte> copied(bytes);
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{producer_device}};
    CUCASCADE_CUDA_TRY(
      ::cudaMemcpy(copied.data(), producer_data.data(), bytes, cudaMemcpyDeviceToHost));
  }
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x31}; }));
  REQUIRE(rmm::get_current_cuda_device() == rmm::cuda_device_id{0});
  f.slab_mr.deallocate(consumer, first, bytes);
  f.slab_mr.deallocate(consumer, second, bytes);
  for (auto* slab : held_slabs) {
    f.slab_mr.deallocate_sync(slab, bytes);
  }
}

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

TEST_CASE("Large allocation falls back to malloc", "[small_pinned]")
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
  // Synchronize as well: a slab is preferred for reuse once its freeing stream has caught up.
  f.slab_mr.deallocate_sync(p1, alloc_size);

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

  // Allocate a large chunk (goes to malloc)
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

TEST_CASE("Slab reuse prefers a ready slab and falls back to a pending one",
          "[small_pinned][stream_ordering]")
{
  check_ready_slab_is_preferred(0);
}

TEST_CASE("Slab reuse skips a slab pending on another GPU",
          "[small_pinned][multi_gpu][stream_ordering]")
{
  int device_count = 0;
  CUCASCADE_CUDA_TRY(::cudaGetDeviceCount(&device_count));
  if (device_count < 2) { SKIP("requires two CUDA devices"); }
  check_ready_slab_is_preferred(1);
}

TEST_CASE("Slab reuse takes an unused slab before a pending one", "[small_pinned][stream_ordering]")
{
  constexpr std::size_t bytes = 2048;
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{bytes, ::cuda::stream_ref{producer.value()}};
  auto* original = f.slab_mr.allocate_sync(bytes);
  std::memset(original, 0x31, bytes);
  pending_stream_work gate{producer};
  CUCASCADE_CUDA_TRY(
    ::cudaMemcpyAsync(device.data(), original, bytes, cudaMemcpyHostToDevice, producer.value()));
  f.slab_mr.deallocate(producer, original, bytes);

  auto* unused = f.slab_mr.allocate(consumer, bytes);
  REQUIRE(unused != original);
  REQUIRE(::cudaGetLastError() == cudaSuccess);
  REQUIRE(synchronizes_before_release(consumer, gate));
  std::memset(unused, 0xA7, bytes);

  gate.release();
  producer.synchronize();
  std::vector<std::byte> copied(bytes);
  CUCASCADE_CUDA_TRY(::cudaMemcpy(copied.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x31}; }));
  f.slab_mr.deallocate(consumer, unused, bytes);
}

TEST_CASE("Slab reuse checks a bounded number of pending slabs", "[small_pinned][stream_ordering]")
{
  constexpr std::size_t bytes = 2048;
  constexpr std::size_t limit = small_pinned_host_memory_resource::MAX_READINESS_QUERIES;
  // One ready slab sits behind `pending` slabs that gated work still uses.
  auto const pending = GENERATE(small_pinned_host_memory_resource::MAX_READINESS_QUERIES - 1,
                                small_pinned_host_memory_resource::MAX_READINESS_QUERIES);
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto* older = f.slab_mr.allocate_sync(bytes);
  std::vector<void*> pending_slabs;
  for (std::size_t i = 0; i < pending; ++i) {
    pending_slabs.push_back(f.slab_mr.allocate_sync(bytes));
  }
  auto const held_slabs = hold_remaining_slabs(f.slab_mr, bytes, pending + 1);
  f.slab_mr.deallocate(consumer, older, bytes);
  consumer.synchronize();
  pending_stream_work gate{producer};
  for (auto* slab : pending_slabs) {
    f.slab_mr.deallocate(producer, slab, bytes);
  }

  auto* taken = f.slab_mr.allocate(consumer, bytes);
  if (pending < limit) {
    // The last permitted query reaches the ready slab.
    REQUIRE(taken == older);
    REQUIRE(synchronizes_before_release(consumer, gate));
  } else {
    // The search stops one slab short of the ready one and settles for the newest, which the
    // allocation stream then waits on.
    REQUIRE(taken == pending_slabs.back());
    REQUIRE(::cudaStreamQuery(consumer.value()) == cudaErrorNotReady);
  }
  gate.release();
  producer.synchronize();
  consumer.synchronize();
  f.slab_mr.deallocate(consumer, taken, bytes);
  for (auto* slab : held_slabs) {
    f.slab_mr.deallocate_sync(slab, bytes);
  }
}

TEST_CASE("Concurrent slab reuse keeps each slab with one owner until its work finishes",
          "[small_pinned][threading][stream_ordering]")
{
  int device_count = 0;
  CUCASCADE_CUDA_TRY(::cudaGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);
  constexpr std::size_t bytes      = 2048;
  constexpr std::size_t iterations = 200;
  auto const devices               = static_cast<std::size_t>(std::min(device_count, 4));
  auto const worker_count          = std::max<std::size_t>(4, devices * 2);
  test_fixture f;
  std::vector<std::exception_ptr> errors(worker_count);
  std::atomic<bool> start{false};
  std::atomic<bool> valid{true};
  std::mutex owners_mutex;
  std::set<void*> owners;
  std::vector<std::jthread> workers;
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      start.wait(false);
      try {
        auto const device = rmm::cuda_device_id{static_cast<int>(worker % devices)};
        rmm::cuda_set_device_raii guard{device};
        rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
        // One region per iteration, so no copy is checked before every later reuse had its chance
        // to disturb it.
        rmm::device_buffer data{bytes * iterations, ::cuda::stream_ref{stream.value()}};
        auto pattern_of = [worker](std::size_t iteration) {
          return static_cast<unsigned char>(worker * 31 + iteration);
        };
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
          auto* host = f.slab_mr.allocate(stream, bytes);
          {
            std::scoped_lock lock(owners_mutex);
            if (!owners.insert(host).second) { valid.store(false); }
          }
          try {
            // Host writes require completion of any reuse dependency.
            stream.synchronize();
            std::memset(host, pattern_of(iteration), bytes);
            CUCASCADE_CUDA_TRY(
              ::cudaMemcpyAsync(static_cast<std::byte*>(data.data()) + iteration * bytes,
                                host,
                                bytes,
                                cudaMemcpyHostToDevice,
                                stream.value()));
          } catch (...) {
            std::scoped_lock lock(owners_mutex);
            owners.erase(host);
            f.slab_mr.deallocate(stream, host, bytes);
            throw;
          }
          {
            std::scoped_lock lock(owners_mutex);
            owners.erase(host);
          }
          // Freed with its copy still queued: the next owner must not write before the copy reads.
          f.slab_mr.deallocate(stream, host, bytes);
        }
        stream.synchronize();
        std::vector<unsigned char> copied(bytes * iterations);
        CUCASCADE_CUDA_TRY(
          ::cudaMemcpy(copied.data(), data.data(), copied.size(), cudaMemcpyDeviceToHost));
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
          auto const begin = copied.begin() + static_cast<std::ptrdiff_t>(iteration * bytes);
          if (!std::all_of(begin,
                           begin + static_cast<std::ptrdiff_t>(bytes),
                           [pattern = pattern_of(iteration)](auto x) { return x == pattern; })) {
            valid.store(false);
          }
        }
        if (rmm::get_current_cuda_device() != device) { valid.store(false); }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  start.store(true);
  start.notify_all();
  workers.clear();
  for (auto const& error : errors) {
    if (error) { std::rethrow_exception(error); }
  }
  REQUIRE(valid.load());
  REQUIRE(owners.empty());
}

TEST_CASE("Slab pool destruction waits for a pending slab's work",
          "[small_pinned][stream_ordering]")
{
  constexpr std::size_t bytes = 2048;
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{bytes, ::cuda::stream_ref{producer.value()}};
  std::optional<small_pinned_host_memory_resource> mr{std::in_place, f.upstream};
  auto* slab = mr->allocate_sync(bytes);
  std::memset(slab, 0x31, bytes);
  pending_stream_work gate{producer};
  CUCASCADE_CUDA_TRY(
    ::cudaMemcpyAsync(device.data(), slab, bytes, cudaMemcpyHostToDevice, producer.value()));
  mr->deallocate(producer, slab, bytes);

  std::atomic<bool> released{false};
  std::jthread releaser{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    released.store(true);
    gate.release();
  }};
  mr.reset();
  // The destructor returned only after the gated copy could run.
  REQUIRE(released.load());
  releaser.join();

  producer.synchronize();
  std::vector<std::byte> copied(bytes);
  CUCASCADE_CUDA_TRY(::cudaMemcpy(copied.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x31}; }));
}
