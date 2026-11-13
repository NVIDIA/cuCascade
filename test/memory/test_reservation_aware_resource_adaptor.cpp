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
 * [pool]              - borrowed CUDA memory-pool identity
 * [gpu]               - requires a CUDA device
 *
 * These tests cover the adaptor's borrowed pool accessor and its OOM diagnostics. The adaptor
 * recovers a handle from supported upstream resources via cuda::mr::resource_cast, accepts an
 * explicit handle for resources that cannot expose one, and otherwise reports a null handle.
 *
 * They also cover how growing a reservation is charged to the adaptor's allocated bytes.
 */

#include "utils/test_memory_resources.hpp"

#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::memory;

namespace {

// A capacity far smaller than the request below, so the adaptor's own tracked
// limit is exceeded before it ever touches the upstream allocator. This yields a
// deterministic OOM without depending on how much GPU memory is actually free.
constexpr std::size_t tiny_capacity   = 1024;
constexpr std::size_t oversized_bytes = 1ULL << 20;  // 1 MiB >> tiny_capacity

reservation_aware_resource_adaptor make_adaptor(rmm::device_async_resource_ref upstream,
                                                cudaMemPool_t pool = nullptr)
{
  return reservation_aware_resource_adaptor{
    memory_space_id{Tier::GPU, 0},
    upstream,
    tiny_capacity,
    tiny_capacity,
    nullptr,
    nullptr,
    reservation_aware_resource_adaptor::AllocationTrackingScope::PER_STREAM,
    pool};
}

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
    adaptor.allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256);
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

constexpr std::size_t chunk = 1 << 20;

/// One GPU space (device 0) over the current device resource, so a test can reserve from it and
/// attach the reservation to a stream.
std::unique_ptr<memory_reservation_manager> make_gpu_memory(std::size_t capacity)
{
  gpu_memory_space_config config;
  config.device_id                  = 0;
  config.memory_capacity            = capacity;
  config.reservation_limit_fraction = 1.0;
  config.mr_factory_fn              = cucascade::test::make_shared_current_device_resource;
  std::vector<memory_space_config> configs;
  configs.emplace_back(config);
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

reservation_aware_resource_adaptor* gpu_adaptor(memory_reservation_manager& memory)
{
  return memory.get_memory_space(Tier::GPU, 0)
    ->get_memory_resource_as<reservation_aware_resource_adaptor>();
}

/// Lets allocations run past the reservation, like the ignore policy, and hands the test the
/// reservation's arena, which only the stream's tracker holds once the reservation is attached.
class capture_arena_policy : public reservation_limit_policy {
 public:
  explicit capture_arena_policy(reserved_arena** arena) : _arena(arena) {}

  void handle_over_reservation([[maybe_unused]] ::cuda::stream_ref stream,
                               [[maybe_unused]] std::size_t requested_bytes,
                               [[maybe_unused]] std::size_t current_allocated,
                               reserved_arena* reserved_bytes) override
  {
    *_arena = reserved_bytes;
  }

  std::string get_policy_name() const override { return "capture_arena"; }

 private:
  reserved_arena** _arena;
};

}  // namespace

TEST_CASE("Pool accessor exposes pools discovered from supported RMM resources",
          "[reservation_aware][pool][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  SECTION("cuda_async_memory_resource")
  {
    rmm::mr::cuda_async_memory_resource upstream{};
    auto adaptor = make_adaptor(rmm::device_async_resource_ref{upstream});
    CHECK(adaptor.pool_handle() == upstream.pool_handle());
  }

  SECTION("cuda_async_view_memory_resource")
  {
    cudaMemPool_t default_pool = nullptr;
    REQUIRE(cudaDeviceGetDefaultMemPool(&default_pool, 0) == cudaSuccess);
    rmm::mr::cuda_async_view_memory_resource upstream{default_pool};
    auto adaptor = make_adaptor(rmm::device_async_resource_ref{upstream});
    CHECK(adaptor.pool_handle() == upstream.pool_handle());
  }

#if CUDART_VERSION >= 13000
  SECTION("cuda_async_managed_memory_resource")
  {
    rmm::mr::cuda_async_managed_memory_resource upstream{};
    auto adaptor = make_adaptor(rmm::device_async_resource_ref{upstream});
    CHECK(adaptor.pool_handle() == upstream.pool_handle());
  }
#endif
}

TEST_CASE("Pool accessor is null when the upstream pool is unknown",
          "[reservation_aware][pool][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_memory_resource upstream{};
  auto adaptor = make_adaptor(rmm::device_async_resource_ref{upstream});
  CHECK(adaptor.pool_handle() == nullptr);
}

TEST_CASE("Pool accessor preserves explicit handles through shared wrapper copies",
          "[reservation_aware][pool][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource upstream{};
  cudaMemPool_t explicit_pool = nullptr;
  REQUIRE(cudaDeviceGetDefaultMemPool(&explicit_pool, 0) == cudaSuccess);
  // Otherwise the copy could report the upstream pool and still match.
  REQUIRE(explicit_pool != upstream.pool_handle());

  auto copy = [&] {
    auto adaptor = make_adaptor(rmm::device_async_resource_ref{upstream}, explicit_pool);
    CHECK(adaptor.pool_handle() == explicit_pool);
    // Construct from an lvalue so the shared wrapper is copied rather than moved or elided.
    return reservation_aware_resource_adaptor{adaptor};
  }();

  CHECK(copy.pool_handle() == explicit_pool);
}

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

// cuda_async_managed_memory_resource requires CUDA 13.0 or higher (stream-ordered
// managed memory pools). Guarded at compile time so the CUDA 12 track skips it.
#if CUDART_VERSION >= 13000
TEST_CASE("OOM reports the default managed pool of a cuda_async_managed_memory_resource upstream",
          "[reservation_aware][oom][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::mr::cuda_async_managed_memory_resource managed_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{managed_mr});

  CHECK(reported != nullptr);
  CHECK(reported == managed_mr.pool_handle());
}
#endif

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

  CHECK(adaptor.pool_handle() == explicit_pool);

  try {
    adaptor.allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256);
    FAIL("expected the oversized allocation to throw cucascade_out_of_memory");
  } catch (const cucascade_out_of_memory& e) {
    CHECK(e.pool_handle == explicit_pool);
    CHECK(e.pool_handle != async_mr.pool_handle());
  }
}

TEST_CASE("Growing a device reservation over bytes allocated past it does not charge them again",
          "[reservation_aware][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t reserved = 2 * chunk;
  auto memory                    = make_gpu_memory(64 * chunk);
  auto* mr                       = gpu_adaptor(*memory);
  REQUIRE(mr != nullptr);
  auto const baseline = mr->get_total_allocated_bytes();

  rmm::cuda_stream stream;
  reserved_arena* arena = nullptr;
  REQUIRE(mr->attach_reservation_to_tracker(
    stream,
    memory->request_reservation(specific_memory_space{Tier::GPU, 0}, reserved),
    std::make_unique<capture_arena_policy>(&arena)));

  // Two chunks inside the reservation and two past it, already charged.
  void* ptr = mr->allocate(stream, 4 * chunk, 256);
  REQUIRE(arena != nullptr);
  REQUIRE(mr->get_total_allocated_bytes() == baseline + 4 * chunk);

  // The grown reservation covers exactly the four chunks in use: nothing new is charged.
  REQUIRE(arena->grow_by(2 * chunk));
  REQUIRE(arena->size() == static_cast<std::int64_t>(4 * chunk));
  CHECK(mr->get_total_allocated_bytes() == baseline + 4 * chunk);

  mr->deallocate(stream, ptr, 4 * chunk, 256);
  CHECK(mr->get_total_allocated_bytes() == baseline + 4 * chunk);  // the reservation stays charged

  mr->reset_stream_reservation(stream);  // releases the reservation
  CHECK(mr->get_total_allocated_bytes() == baseline);
  CHECK(mr->get_total_reserved_bytes() == 0);
}

TEST_CASE("Growing a device reservation partly over bytes allocated past it charges only the rest",
          "[reservation_aware][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t reserved = 2 * chunk;
  auto memory                    = make_gpu_memory(64 * chunk);
  auto* mr                       = gpu_adaptor(*memory);
  REQUIRE(mr != nullptr);
  auto const baseline = mr->get_total_allocated_bytes();

  rmm::cuda_stream stream;
  reserved_arena* arena = nullptr;
  REQUIRE(mr->attach_reservation_to_tracker(
    stream,
    memory->request_reservation(specific_memory_space{Tier::GPU, 0}, reserved),
    std::make_unique<capture_arena_policy>(&arena)));

  // One chunk past the reservation; growing by two covers it and reserves one new chunk.
  void* ptr = mr->allocate(stream, 3 * chunk, 256);
  REQUIRE(arena != nullptr);
  REQUIRE(mr->get_total_allocated_bytes() == baseline + 3 * chunk);
  REQUIRE(arena->grow_by(2 * chunk));
  CHECK(mr->get_total_allocated_bytes() == baseline + 4 * chunk);

  mr->deallocate(stream, ptr, 3 * chunk, 256);
  mr->reset_stream_reservation(stream);
  CHECK(mr->get_total_allocated_bytes() == baseline);
  CHECK(mr->get_total_reserved_bytes() == 0);
}

TEST_CASE("A policy that grows a device reservation charges the triggering allocation once",
          "[reservation_aware][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t reserved = 2 * chunk;
  auto memory                    = make_gpu_memory(64 * chunk);
  auto* mr                       = gpu_adaptor(*memory);
  REQUIRE(mr != nullptr);
  auto const baseline = mr->get_total_allocated_bytes();

  rmm::cuda_stream stream;
  REQUIRE(mr->attach_reservation_to_tracker(
    stream,
    memory->request_reservation(specific_memory_space{Tier::GPU, 0}, reserved),
    std::make_unique<increase_reservation_limit_policy>(1.0)));

  // The policy grows the reservation to exactly the allocation, so the allocation is covered by
  // the reservation and charges nothing on top of it.
  void* ptr = mr->allocate(stream, 4 * chunk, 256);
  REQUIRE(mr->get_total_reserved_bytes() == 4 * chunk);
  CHECK(mr->get_total_allocated_bytes() == baseline + 4 * chunk);

  mr->deallocate(stream, ptr, 4 * chunk, 256);
  CHECK(mr->get_total_allocated_bytes() == baseline + 4 * chunk);  // the reservation stays charged

  mr->reset_stream_reservation(stream);
  CHECK(mr->get_total_allocated_bytes() == baseline);
  CHECK(mr->get_total_reserved_bytes() == 0);
}

TEST_CASE("A failed allocation past a device reservation leaves the charged bytes unchanged",
          "[reservation_aware][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t reserved = 2 * chunk;
  auto memory                    = make_gpu_memory(64 * chunk);
  auto* mr                       = gpu_adaptor(*memory);
  REQUIRE(mr != nullptr);
  auto const baseline = mr->get_total_allocated_bytes();

  rmm::cuda_stream stream;  // the default policy lets allocations run past the reservation
  REQUIRE(mr->attach_reservation_to_tracker(
    stream, memory->request_reservation(specific_memory_space{Tier::GPU, 0}, reserved)));

  void* ptr = mr->allocate(stream, chunk, 256);  // inside the reservation

  // Crosses the end of the reservation and exceeds the capacity, so it is undone.
  REQUIRE_THROWS_AS(mr->allocate(stream, 128 * chunk, 256), cucascade_out_of_memory);
  CHECK(mr->get_total_allocated_bytes() == baseline + reserved);
  CHECK(mr->get_allocated_bytes(stream) == chunk);

  mr->deallocate(stream, ptr, chunk, 256);
  mr->reset_stream_reservation(stream);
  CHECK(mr->get_total_allocated_bytes() == baseline);
}

TEST_CASE("Growing a device reservation while another thread allocates past it keeps the charges",
          "[reservation_aware][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t reserved   = chunk;
  constexpr std::size_t alloc_size = 32 * chunk;
  constexpr std::size_t grow_step  = 4096;
  constexpr int max_grows          = 12288;  // 48 chunks, less than the two allocations
  constexpr int rounds             = 500;
  auto memory                      = make_gpu_memory(256 * chunk);
  auto* mr                         = gpu_adaptor(*memory);
  REQUIRE(mr != nullptr);
  auto const baseline = mr->get_total_allocated_bytes();

  rmm::cuda_stream stream;
  reserved_arena* arena = nullptr;
  REQUIRE(mr->attach_reservation_to_tracker(
    stream,
    memory->request_reservation(specific_memory_space{Tier::GPU, 0}, reserved),
    std::make_unique<capture_arena_policy>(&arena)));

  // Hand the test the arena before the threads start.
  mr->deallocate(stream, mr->allocate(stream, 2 * chunk, 256), 2 * chunk, 256);
  REQUIRE(arena != nullptr);

  // The two allocations together always end past the reservation, and the first one crosses its
  // end until the reservation has grown past it.
  std::atomic<bool> allocator_done{false};
  int grows = 0;
  // A copy, since the policy writes arena again on every allocation past the end.
  std::thread grower([&, target = arena]() {
    while (!allocator_done.load() && grows < max_grows) {
      if (target->grow_by(grow_step)) { ++grows; }
      std::this_thread::yield();
    }
  });
  std::thread allocator([&]() {
    for (int i = 0; i < rounds; ++i) {
      void* first  = mr->allocate(stream, alloc_size, 256);
      void* second = mr->allocate(stream, alloc_size, 256);
      mr->deallocate(stream, first, alloc_size, 256);
      mr->deallocate(stream, second, alloc_size, 256);
    }
    allocator_done.store(true);
  });
  allocator.join();
  grower.join();
  stream.synchronize();

  REQUIRE(grows > 0);
  auto const grown = reserved + static_cast<std::size_t>(grows) * grow_step;
  CHECK(arena->size() == static_cast<std::int64_t>(grown));
  CHECK(mr->get_total_allocated_bytes() == baseline + grown);  // the reservation stays charged

  mr->reset_stream_reservation(stream);
  CHECK(mr->get_total_allocated_bytes() == baseline);
  CHECK(mr->get_total_reserved_bytes() == 0);
}
