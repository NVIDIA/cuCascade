/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <optional>
#include <stop_token>
#include <thread>
#include <tuple>

namespace {

struct progress_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
  cucascade::memory::small_pinned_host_memory_resource resource{upstream, 64ULL << 10};
};

class call_gate {
 public:
  ~call_gate() { release(); }

  static std::optional<cudaError_t> wait(cucascade::test::pinned_host_call const&,
                                         void* context) noexcept
  {
    auto& gate = *static_cast<call_gate*>(context);
    gate.entered.store(true, std::memory_order_release);
    gate._released.wait(false, std::memory_order_acquire);
    return gate.status;
  }

  void release() noexcept
  {
    _released.store(true, std::memory_order_release);
    _released.notify_all();
  }

  std::atomic<bool> entered{false};
  std::optional<cudaError_t> status;

 private:
  std::atomic<bool> _released{false};
};

bool wait_for(std::atomic<bool> const& flag)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

}  // namespace

TEST_CASE("Pinned allocator traffic progresses while a reuse wait is submitted",
          "[small_pinned][fault_injection][cache_progress]")
{
  progress_fixture fixture;
  rmm::cuda_stream stream;
  auto const bytes = GENERATE(std::size_t{512}, std::size_t{9000});
  auto* seed       = fixture.resource.allocate_sync(bytes);
  fixture.resource.deallocate_sync(seed, bytes);

  call_gate gate;
  std::atomic<bool> observed{false};
  std::exception_ptr error;
  void* reused = nullptr;
  std::jthread borrower{[&](std::stop_token stop) {
    std::stop_callback release_on_stop{stop, [&] { gate.release(); }};
    try {
      cucascade::test::scoped_pinned_host_hook hook{
        cucascade::test::pinned_host_operation::event_wait, call_gate::wait, &gate};
      reused = fixture.resource.allocate(stream, bytes);
    } catch (...) {
      error = std::current_exception();
    }
  }};
  bool const entered = wait_for(gate.entered);
  std::jthread observer{[&] {
    std::ignore = fixture.resource.get_statistics();
    observed.store(true, std::memory_order_release);
  }};
  bool const progressed = wait_for(observed);
  gate.release();
  borrower.join();
  observer.join();

  if (reused != nullptr) { fixture.resource.deallocate(stream, reused, bytes); }

  REQUIRE(entered);
  REQUIRE(progressed);
  REQUIRE(error == nullptr);
  REQUIRE(reused == seed);
}

TEST_CASE("Pinned allocator traffic progresses while a cold event is created",
          "[small_pinned][fault_injection][cache_progress]")
{
  progress_fixture fixture;
  rmm::cuda_stream stream;
  auto const bytes = GENERATE(std::size_t{512}, std::size_t{9000});
  auto* pointer    = fixture.resource.allocate_sync(bytes);
  auto* other      = fixture.resource.allocate_sync(bytes);

  call_gate gate;
  std::atomic<bool> observed{false};
  std::jthread releasing{[&](std::stop_token stop) {
    std::stop_callback release_on_stop{stop, [&] { gate.release(); }};
    cucascade::test::scoped_pinned_host_hook hook{
      cucascade::test::pinned_host_operation::event_create, call_gate::wait, &gate};
    fixture.resource.deallocate(stream, pointer, bytes);
  }};
  bool const entered = wait_for(gate.entered);
  std::jthread observer{[&] {
    fixture.resource.deallocate(stream, other, bytes);
    std::ignore = fixture.resource.get_statistics();
    observed.store(true, std::memory_order_release);
  }};
  bool const progressed = wait_for(observed);
  gate.release();
  releasing.join();
  observer.join();

  REQUIRE(entered);
  REQUIRE(progressed);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Failed cold event creation keeps its reserved return slot reusable",
          "[small_pinned][fault_injection][cache_progress]")
{
  progress_fixture fixture;
  constexpr std::size_t bytes = 9000;
  auto* failed                = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_pinned_host_hook failure{
      cucascade::test::pinned_host_operation::event_create, cudaErrorInvalidValue};
    fixture.resource.deallocate_sync(failed, bytes);
  }
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, failed) == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeUnregistered);
  auto* next = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_pinned_host_hook reject_growth{
      cucascade::test::pinned_host_operation::event_pool_reserve, cudaErrorMemoryAllocation};
    fixture.resource.deallocate_sync(next, bytes);
    REQUIRE(fixture.resource.large_cache_bytes() == 16384);
    auto* reused = fixture.resource.allocate_sync(bytes);
    REQUIRE(reused == next);
    fixture.resource.deallocate_sync(reused, bytes);
  }
}

TEST_CASE("Failed cache hit cannot overfill a concurrently refilled cache",
          "[small_pinned][fault_injection][cache_progress]")
{
  progress_fixture fixture;
  rmm::cuda_stream stream;
  bool const partial_refill = GENERATE(false, true);
  auto const refill_bytes   = partial_refill ? std::size_t{20000} : std::size_t{40000};
  auto* seed                = fixture.resource.allocate_sync(9000);
  auto* newer               = partial_refill ? fixture.resource.allocate_sync(9000) : nullptr;
  auto* refill              = fixture.resource.allocate_sync(refill_bytes);
  fixture.resource.deallocate_sync(seed, 9000);
  if (newer != nullptr) { fixture.resource.deallocate_sync(newer, 9000); }

  call_gate gate;
  gate.status = cudaErrorInvalidResourceHandle;
  std::atomic<bool> filled{false};
  std::exception_ptr error;
  std::jthread borrower{[&](std::stop_token stop) {
    std::stop_callback release_on_stop{stop, [&] { gate.release(); }};
    try {
      cucascade::test::scoped_pinned_host_hook hook{
        cucascade::test::pinned_host_operation::event_wait, call_gate::wait, &gate};
      std::ignore = fixture.resource.allocate(stream, 9000);
    } catch (...) {
      error = std::current_exception();
    }
  }};
  bool const entered = wait_for(gate.entered);
  std::jthread filler{[&] {
    fixture.resource.deallocate_sync(refill, refill_bytes);
    filled.store(true, std::memory_order_release);
  }};
  bool const progressed = wait_for(filled);
  gate.release();
  borrower.join();
  filler.join();

  REQUIRE(entered);
  REQUIRE(progressed);
  REQUIRE(error != nullptr);
  REQUIRE_THROWS_AS(std::rethrow_exception(error), cucascade::cuda_error);
  auto const stats = fixture.resource.get_statistics();
  REQUIRE(stats.cached_large_bytes == 65536);
  REQUIRE(stats.peak_cached_large_bytes == 65536);
  REQUIRE(stats.large_requests == (partial_refill ? 4 : 3));
  REQUIRE(stats.large_hits == 0);
  REQUIRE(stats.large_misses == (partial_refill ? 3 : 2));
  REQUIRE(stats.large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(refill_bytes);
  REQUIRE(reused == refill);
  fixture.resource.deallocate_sync(reused, refill_bytes);
  if (partial_refill) {
    auto* first  = fixture.resource.allocate_sync(9000);
    auto* second = fixture.resource.allocate_sync(9000);
    REQUIRE(first == seed);
    REQUIRE(second == newer);
    fixture.resource.deallocate_sync(first, 9000);
    fixture.resource.deallocate_sync(second, 9000);
  } else {
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, seed) == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeHost);
    REQUIRE(cudaFreeHost(seed) == cudaSuccess);
  }
}

TEST_CASE("Failed cache hit restores oldest-first reuse when retained capacity permits",
          "[small_pinned][fault_injection][cache_progress]")
{
  progress_fixture fixture;
  auto* oldest = fixture.resource.allocate_sync(9000);
  auto* newer  = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(oldest, 9000);
  fixture.resource.deallocate_sync(newer, 9000);
  {
    cucascade::test::scoped_pinned_host_hook failure{
      cucascade::test::pinned_host_operation::event_wait, cudaErrorInvalidResourceHandle};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(9000), cucascade::cuda_error);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  auto* first  = fixture.resource.allocate_sync(9000);
  auto* second = fixture.resource.allocate_sync(9000);
  REQUIRE(first == oldest);
  REQUIRE(second == newer);
  fixture.resource.deallocate_sync(first, 9000);
  fixture.resource.deallocate_sync(second, 9000);
}
