/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fault-injection tests for small_pinned_host_memory_resource. Built into
// cucascade_pinned_host_fault_tests against the hooked core (see test/CMakeLists.txt).

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <tuple>
#include <utility>

namespace {

using cucascade::test::get_pointer_type;
using cucascade::test::pinned_host_test_fixture;
using cucascade::test::pinned_upstream_fixture;
using cucascade::test::require_pointer_type;

}  // namespace

//===----------------------------------------------------------------------===//
// Fault hook scoping
//===----------------------------------------------------------------------===//

TEST_CASE("Pinned allocation failures are scoped to one call and one thread",
          "[small_pinned][fault_injection][hook_scope]")
{
  pinned_host_test_fixture fixture{0};
  constexpr std::size_t bytes = 100000;

  SECTION("one failed call does not reject subsequent allocations")
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::host_alloc,
                                               cudaErrorInvalidValue};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::bad_alloc);
    auto* pointer = fixture.resource.allocate_sync(bytes);
    fixture.resource.deallocate_sync(pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("a hook does not reject another thread's allocation")
  {
    void* pointer = nullptr;
    std::exception_ptr error;
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::host_alloc,
                                                 cudaErrorInvalidValue};
      std::jthread worker{[&] {
        try {
          pointer = fixture.resource.allocate_sync(bytes);
        } catch (...) {
          error = std::current_exception();
        }
      }};
      worker.join();
      REQUIRE(error == nullptr);
      REQUIRE(pointer != nullptr);
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::bad_alloc);
    }
    fixture.resource.deallocate_sync(pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }
}

//===----------------------------------------------------------------------===//
// Release after confirmed completion
//===----------------------------------------------------------------------===//

TEST_CASE("Uncached pinned storage is released only after confirmed completion",
          "[small_pinned][fault_injection][release]")
{
  pinned_host_test_fixture fixture{0};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 100000;
  void* pointer               = fixture.resource.allocate_sync(bytes);
  require_pointer_type(pointer, cudaMemoryTypeHost);

  SECTION("completed work releases its storage")
  {
    fixture.resource.deallocate(stream, pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("unconfirmed completion withholds its storage")
  {
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::stream_synchronize,
                                                 cudaErrorInvalidResourceHandle};
      fixture.resource.deallocate(stream, pointer, bytes);
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);

    auto* next = fixture.resource.allocate_sync(bytes);
    fixture.resource.deallocate(stream, next, bytes);
    require_pointer_type(next, cudaMemoryTypeUnregistered);
  }
}

TEST_CASE("Cached pinned storage is released only after its event confirms completion",
          "[small_pinned][fault_injection][release]")
{
  pinned_upstream_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 64ULL << 10};
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource->allocate_sync(bytes);
  resource->deallocate_sync(pointer, bytes);

  SECTION("completed cached work releases its storage")
  {
    resource.reset();
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("unconfirmed cached work withholds its storage")
  {
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::event_synchronize,
                                                 cudaErrorInvalidResourceHandle};
      resource.reset();
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
}

TEST_CASE("Cache-entry metadata exhaustion releases incoming completed storage",
          "[small_pinned][fault_injection][release]")
{
  pinned_host_test_fixture fixture;
  auto& resource              = fixture.resource;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::large_cache_entry,
                                               cudaErrorMemoryAllocation};
    resource.deallocate_sync(pointer, bytes);
    REQUIRE(resource.large_cache_bytes() == 0);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);

    auto* next = resource.allocate_sync(bytes);
    resource.deallocate_sync(next, bytes);
    REQUIRE(resource.large_cache_bytes() == 16384);
    require_pointer_type(next, cudaMemoryTypeHost);
  }
}

//===----------------------------------------------------------------------===//
// Slab quarantine
//===----------------------------------------------------------------------===//

TEST_CASE("Slab destruction retains only backing with unconfirmed completion",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* uncertain = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(uncertain, 512);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> failure;
    if (fail) {
      failure.emplace(cucascade::test::fault_point::event_synchronize,
                      cudaErrorInvalidResourceHandle);
    }
    resource.reset();
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}

TEST_CASE("An unconfirmed slab deallocation retains only its backing owner",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  rmm::cuda_stream stream;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* uncertain = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> record_failure;
    std::optional<cucascade::test::scoped_fault_hook> wait_failure;
    if (fail) {
      record_failure.emplace(cucascade::test::fault_point::event_record,
                             cudaErrorInvalidResourceHandle);
      wait_failure.emplace(cucascade::test::fault_point::stream_synchronize,
                           cudaErrorInvalidResourceHandle);
    }
    resource->deallocate(stream, uncertain, 512);
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}

TEST_CASE("Slab return metadata failure retains only its backing owner",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  rmm::cuda_stream stream;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* withheld  = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> failure;
    if (fail) {
      failure.emplace(cucascade::test::fault_point::slab_return, cudaErrorMemoryAllocation);
    }
    resource->deallocate(stream, withheld, 512);
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}

//===----------------------------------------------------------------------===//
// Cache insertion
//===----------------------------------------------------------------------===//

TEST_CASE("Failed cache insertion preserves retained large-buffer capacity",
          "[small_pinned][fault_injection][cache_insertion]")
{
  pinned_host_test_fixture fixture{32ULL << 10};
  rmm::cuda_stream stream;
  constexpr std::size_t small_bytes = 9000;
  constexpr std::size_t large_bytes = 20000;
  auto* oldest                      = fixture.resource.allocate_sync(small_bytes);
  auto* next                        = fixture.resource.allocate_sync(small_bytes);
  auto* incoming                    = fixture.resource.allocate_sync(large_bytes);
  fixture.resource.deallocate(stream, oldest, small_bytes);
  fixture.resource.deallocate(stream, next, small_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);

  bool const reject_completion = GENERATE(false, true);
  {
    cucascade::test::scoped_fault_hook metadata_failure{
      cucascade::test::fault_point::large_cache_entry, cudaErrorMemoryAllocation};
    if (reject_completion) {
      cucascade::test::scoped_fault_hook completion_failure{
        cucascade::test::fault_point::event_synchronize, cudaErrorInvalidResourceHandle};
      fixture.resource.deallocate(stream, incoming, large_bytes);
    } else {
      fixture.resource.deallocate(stream, incoming, large_bytes);
    }
  }

  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  require_pointer_type(oldest, cudaMemoryTypeHost);
  require_pointer_type(next, cudaMemoryTypeHost);
  require_pointer_type(incoming,
                       reject_completion ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  if (reject_completion) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(incoming) == cudaSuccess);
  }

  auto* reused_oldest = fixture.resource.allocate_sync(small_bytes);
  auto* reused_next   = fixture.resource.allocate_sync(small_bytes);
  REQUIRE(reused_oldest == oldest);
  REQUIRE(reused_next == next);
  fixture.resource.deallocate_sync(reused_oldest, small_bytes);
  fixture.resource.deallocate_sync(reused_next, small_bytes);
}

TEST_CASE("Successful cache insertion releases every required completed victim",
          "[small_pinned][fault_injection][cache_insertion]")
{
  pinned_host_test_fixture fixture{32ULL << 10};
  rmm::cuda_stream stream;
  constexpr std::size_t small_bytes = 9000;
  constexpr std::size_t large_bytes = 20000;
  auto* oldest                      = fixture.resource.allocate_sync(small_bytes);
  auto* next                        = fixture.resource.allocate_sync(small_bytes);
  auto* incoming                    = fixture.resource.allocate_sync(large_bytes);
  fixture.resource.deallocate(stream, oldest, small_bytes);
  fixture.resource.deallocate(stream, next, small_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);

  fixture.resource.deallocate(stream, incoming, large_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 2);
  require_pointer_type(oldest, cudaMemoryTypeUnregistered);
  require_pointer_type(next, cudaMemoryTypeUnregistered);
  require_pointer_type(incoming, cudaMemoryTypeHost);

  auto* reused = fixture.resource.allocate_sync(large_bytes);
  REQUIRE(reused == incoming);
  fixture.resource.deallocate_sync(reused, large_bytes);
}

//===----------------------------------------------------------------------===//
// Cache progress
//===----------------------------------------------------------------------===//

namespace {

class call_gate {
 public:
  ~call_gate() { release(); }

  static std::optional<cudaError_t> wait(cucascade::test::fault_point_call const&,
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

[[nodiscard]] bool wait_for(std::atomic<bool> const& flag)
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
  pinned_host_test_fixture fixture;
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
      cucascade::test::scoped_fault_hook hook{
        cucascade::test::fault_point::event_wait, call_gate::wait, &gate};
      reused = fixture.resource.allocate(stream, bytes);
    } catch (...) {
      error = std::current_exception();
    }
  }};
  bool const entered = wait_for(gate.entered);
  cucascade::memory::small_pinned_host_memory_resource::statistics pending{};
  std::jthread observer{[&] {
    pending = fixture.resource.get_statistics();
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
  if (bytes == 9000) {
    REQUIRE(pending.large_requests == 2);
    REQUIRE(pending.large_hits == 0);
    REQUIRE(pending.large_misses == 1);
    REQUIRE(pending.large_uncacheable == 0);
    auto const complete = fixture.resource.get_statistics();
    REQUIRE(complete.large_requests == 2);
    REQUIRE(complete.large_hits == 1);
    REQUIRE(complete.large_misses == 1);
    REQUIRE(complete.large_uncacheable == 0);
    REQUIRE(complete.large_requests ==
            complete.large_hits + complete.large_misses + complete.large_uncacheable);
  }
}

TEST_CASE("Pinned allocator traffic progresses while a cold event is created",
          "[small_pinned][fault_injection][cache_progress]")
{
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  auto const bytes = GENERATE(std::size_t{512}, std::size_t{9000});
  auto* pointer    = fixture.resource.allocate_sync(bytes);
  auto* other      = fixture.resource.allocate_sync(bytes);

  call_gate gate;
  std::atomic<bool> observed{false};
  std::jthread releasing{[&](std::stop_token stop) {
    std::stop_callback release_on_stop{stop, [&] { gate.release(); }};
    cucascade::test::scoped_fault_hook hook{
      cucascade::test::fault_point::event_create, call_gate::wait, &gate};
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
  pinned_host_test_fixture fixture;
  constexpr std::size_t bytes = 9000;
  auto* failed                = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::event_create,
                                               cudaErrorInvalidValue};
    fixture.resource.deallocate_sync(failed, bytes);
  }
  require_pointer_type(failed, cudaMemoryTypeUnregistered);
  auto* next = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_fault_hook reject_growth{
      cucascade::test::fault_point::event_pool_reserve, cudaErrorMemoryAllocation};
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
  pinned_host_test_fixture fixture;
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
      cucascade::test::scoped_fault_hook hook{
        cucascade::test::fault_point::event_wait, call_gate::wait, &gate};
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
    require_pointer_type(seed, cudaMemoryTypeHost);
    REQUIRE(cudaFreeHost(seed) == cudaSuccess);
  }
}

TEST_CASE("Failed cache hit restores oldest-first reuse when retained capacity permits",
          "[small_pinned][fault_injection][cache_progress]")
{
  pinned_host_test_fixture fixture;
  auto* oldest = fixture.resource.allocate_sync(9000);
  auto* newer  = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(oldest, 9000);
  fixture.resource.deallocate_sync(newer, 9000);
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::event_wait,
                                               cudaErrorInvalidResourceHandle};
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

TEST_CASE("Failed event preparation still removes dedicated exact allocation ownership",
          "[small_pinned][fault_injection][cache_progress][recovery]")
{
  pinned_host_test_fixture fixture;
  auto const failed_operation  = GENERATE(cucascade::test::fault_point::event_create,
                                         cucascade::test::fault_point::event_record,
                                         cucascade::test::fault_point::large_cache_entry);
  bool const reject_completion = GENERATE(false, true);
  void* pointer                = nullptr;
  {
    cucascade::test::scoped_fault_hook rounded_oom{cucascade::test::fault_point::host_alloc,
                                                   cudaErrorMemoryAllocation};
    pointer = fixture.resource.allocate_sync(40000);
  }

  std::size_t free_calls = 0;
  {
    // Keep the real storage so the next allocation can deterministically return the same address.
    cucascade::test::scoped_fault_hook keep_storage{
      cucascade::test::fault_point::host_free,
      [](cucascade::test::fault_point_call const&,
         void* context) noexcept -> std::optional<cudaError_t> {
        ++*static_cast<std::size_t*>(context);
        return cudaSuccess;
      },
      &free_calls};
    cucascade::test::scoped_fault_hook preparation_failure{failed_operation,
                                                           cudaErrorMemoryAllocation};
    if (reject_completion) {
      auto const wait_operation =
        failed_operation == cucascade::test::fault_point::large_cache_entry
          ? cucascade::test::fault_point::event_synchronize
          : cucascade::test::fault_point::stream_synchronize;
      cucascade::test::scoped_fault_hook completion_failure{wait_operation,
                                                            cudaErrorInvalidResourceHandle};
      fixture.resource.deallocate(cucascade::test::legacy_default_stream, pointer, 40000);
    } else {
      fixture.resource.deallocate_sync(pointer, 40000);
    }
  }
  REQUIRE(free_calls == (reject_completion ? 0 : 1));
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  struct repeated_address {
    void* pointer;
    std::size_t attempts{};
  } reuse{pointer};
  void* next = nullptr;
  {
    cucascade::test::scoped_fault_hook same_address{
      cucascade::test::fault_point::host_alloc,
      [](cucascade::test::fault_point_call const& call,
         void* context) noexcept -> std::optional<cudaError_t> {
        auto& state = *static_cast<repeated_address*>(context);
        if (++state.attempts == 1) { return cudaErrorMemoryAllocation; }
        *call.buffer_out = state.pointer;
        return cudaSuccess;
      },
      &reuse};
    REQUIRE_NOTHROW(next = fixture.resource.allocate_sync(40000));
  }
  REQUIRE(next == pointer);
  REQUIRE(reuse.attempts == 2);
  fixture.resource.deallocate_sync(next, 40000);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  REQUIRE(fixture.resource.get_statistics().large_requests == 2);
}

//===----------------------------------------------------------------------===//
// cuda_event destruction failures
//===----------------------------------------------------------------------===//

namespace {

struct destruction_failure {
  int attempts{};
  cucascade::cuda::cuda_event* owner{};
  bool owner_was_empty{true};
};

std::optional<cudaError_t> destroy_then_fail(cucascade::test::fault_point_call const& call,
                                             void* context) noexcept
{
  auto& failure = *static_cast<destruction_failure*>(context);
  ++failure.attempts;
  if (failure.owner != nullptr && failure.owner->get() != nullptr) {
    failure.owner_was_empty = false;
  }
  auto const result = cudaEventDestroy(call.event);
  if (result != cudaSuccess) { return result; }
  return cudaSetDevice(-1);
}

std::optional<cudaError_t> count_destruction(cucascade::test::fault_point_call const&,
                                             void* context) noexcept
{
  ++*static_cast<int*>(context);
  return std::nullopt;
}

}  // namespace

TEST_CASE("Event destruction clears its own CUDA failure", "[cuda_event][fault_injection]")
{
  destruction_failure failure;
  cucascade::test::scoped_fault_hook hook{
    cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
  {
    cucascade::cuda::cuda_event event;
    REQUIRE(event.get() != nullptr);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(failure.attempts == 1);
}

TEST_CASE("Event reset relinquishes ownership even when destruction fails",
          "[cuda_event][fault_injection]")
{
  destruction_failure failure;
  cucascade::test::scoped_fault_hook hook{
    cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
  {
    cucascade::cuda::cuda_event event;
    failure.owner = &event;
    REQUIRE(event.reset_no_throw() == cudaErrorInvalidDevice);
    REQUIRE(event.get() == nullptr);
    REQUIRE(failure.owner_was_empty);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(event.reset_no_throw() == cudaSuccess);
  }
  REQUIRE(failure.attempts == 1);
}

TEST_CASE("An empty event reset preserves unrelated CUDA error state",
          "[cuda_event][fault_injection]")
{
  int destructions = 0;
  cucascade::test::scoped_fault_hook observed{
    cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
  {
    cucascade::cuda::cuda_event event;
    REQUIRE(event.reset_no_throw() == cudaSuccess);
    REQUIRE(event.get() == nullptr);
    REQUIRE(cudaSetDevice(-1) == cudaErrorInvalidDevice);
    REQUIRE(event.reset_no_throw() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaErrorInvalidDevice);
  }
  REQUIRE(destructions == 1);
}

TEST_CASE("Event move assignment safely replaces its previous ownership",
          "[cuda_event][fault_injection]")
{
  cucascade::cuda::cuda_event source;
  cucascade::cuda::cuda_event destination;
  auto const source_handle = source.get();
  destruction_failure failure{.owner = &destination};
  {
    cucascade::test::scoped_fault_hook hook{
      cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
    destination = std::move(source);
    REQUIRE(source.get() == nullptr);
    REQUIRE(destination.get() == source_handle);
    REQUIRE(failure.owner_was_empty);
    REQUIRE(failure.attempts == 1);
    REQUIRE(cudaGetLastError() == cudaSuccess);
  }
  REQUIRE(destination.reset_no_throw() == cudaSuccess);
}

//===----------------------------------------------------------------------===//
// Event retirement
//===----------------------------------------------------------------------===//

TEST_CASE("Eviction retires only events whose completion cannot be confirmed",
          "[small_pinned][fault_injection][event_retirement]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* victim                = resource.allocate_sync(bytes);
  auto* incoming              = resource.allocate_sync(bytes);
  resource.deallocate(stream, victim, bytes);
  int destructions = 0;
  bool const fail  = GENERATE(false, true);
  {
    cucascade::test::scoped_fault_hook observed{
      cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
    std::optional<cucascade::test::scoped_fault_hook> failed;
    if (fail) {
      failed.emplace(cucascade::test::fault_point::event_synchronize,
                     cudaErrorInvalidResourceHandle);
    }
    resource.deallocate(stream, incoming, bytes);
  }
  require_pointer_type(victim, fail ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (fail) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(victim) == cudaSuccess);
  }
  REQUIRE(destructions == (fail ? 1 : 0));
  REQUIRE(resource.large_cache_bytes() == 16384);
}

TEST_CASE("Unpublished buffers retire events after unconfirmed completion",
          "[small_pinned][fault_injection][event_retirement]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource.allocate_sync(bytes);
  int destructions            = 0;
  bool const fail             = GENERATE(false, true);
  {
    cucascade::test::scoped_fault_hook observed{
      cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
    cucascade::test::scoped_fault_hook metadata_failure{
      cucascade::test::fault_point::large_cache_entry, cudaErrorMemoryAllocation};
    std::optional<cucascade::test::scoped_fault_hook> failed;
    if (fail) {
      failed.emplace(cucascade::test::fault_point::event_synchronize,
                     cudaErrorInvalidResourceHandle);
    }
    resource.deallocate(stream, pointer, bytes);
  }
  require_pointer_type(pointer, fail ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (fail) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  REQUIRE(destructions == (fail ? 1 : 0));
  REQUIRE(resource.large_cache_bytes() == 0);
}

TEST_CASE("Repeated event retirement reuses reserved return capacity",
          "[small_pinned][fault_injection][event_retirement]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  int destructions            = 0;
  for (int iteration = 0; iteration != 8; ++iteration) {
    auto* pointer = resource.allocate_sync(bytes);
    {
      cucascade::test::scoped_fault_hook observed{
        cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
      cucascade::test::scoped_fault_hook metadata_failure{
        cucascade::test::fault_point::large_cache_entry, cudaErrorMemoryAllocation};
      cucascade::test::scoped_fault_hook completion_failure{
        cucascade::test::fault_point::event_synchronize, cudaErrorInvalidResourceHandle};
      std::optional<cucascade::test::scoped_fault_hook> growth_failure;
      if (iteration != 0) {
        growth_failure.emplace(cucascade::test::fault_point::event_pool_reserve,
                               cudaErrorMemoryAllocation);
      }
      resource.deallocate(stream, pointer, bytes);
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  REQUIRE(destructions == 8);

  auto* healthy = resource.allocate_sync(bytes);
  resource.deallocate(stream, healthy, bytes);
  REQUIRE(resource.large_cache_bytes() == 16384);
}

//===----------------------------------------------------------------------===//
// Out-of-memory recovery
//===----------------------------------------------------------------------===//

namespace {

struct allocation_trace {
  std::array<std::size_t, 8> sizes{};
  std::size_t count{};
  std::size_t failures{1};
  cudaError_t terminal_error{cudaSuccess};

  static std::optional<cudaError_t> call(cucascade::test::fault_point_call const& call,
                                         void* context) noexcept
  {
    auto& trace = *static_cast<allocation_trace*>(context);
    if (trace.count < trace.sizes.size()) { trace.sizes[trace.count] = call.bytes; }
    ++trace.count;
    if (trace.count <= trace.failures) { return cudaErrorMemoryAllocation; }
    if (trace.terminal_error != cudaSuccess) { return trace.terminal_error; }
    return std::nullopt;
  }
};

}  // namespace

TEST_CASE("Rounded pinned OOM uses uncached exact storage without reclaiming retained buffers",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture;
  auto* seed = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(seed, 9000);
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    pointer = fixture.resource.allocate_sync(40000);
  }
  REQUIRE(trace.count == 2);
  CHECK(trace.sizes[0] == 65536);
  CHECK(trace.sizes[1] == 40000);
  CHECK(fixture.resource.large_cache_bytes() == 16384);
  std::memset(pointer, 0x37, 40000);
  fixture.resource.deallocate_sync(pointer, 40000);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_misses == 2);
  REQUIRE(statistics.large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == seed);
  fixture.resource.deallocate_sync(reused, 9000);
}

TEST_CASE("Exact pinned OOM reclaims only enough oldest retained capacity and retries once",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{128ULL << 10};
  std::array<void*, 4> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(9000)};
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    fixture.resource.deallocate_sync(seeds[index], index == 3 ? 9000 : 20000);
  }
  allocation_trace trace;
  trace.failures = 2;
  void* pointer  = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(50000));
  }
  REQUIRE(pointer != nullptr);
  REQUIRE(trace.count == 3);
  REQUIRE(trace.sizes[0] == 65536);
  REQUIRE(trace.sizes[1] == 50000);
  REQUIRE(trace.sizes[2] == 50000);
  REQUIRE(fixture.resource.large_cache_bytes() == 49152);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 2);
  fixture.resource.deallocate_sync(pointer, 50000);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  auto* retained = fixture.resource.allocate_sync(20000);
  REQUIRE(retained == seeds[2]);
  fixture.resource.deallocate_sync(retained, 20000);
}

TEST_CASE(
  "Insufficient retained capacity preserves cached storage without duplicate exact attempts",
  "[small_pinned][fault_injection][recovery]")
{
  auto const [bytes, limit, first_size, attempts] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t, std::size_t, std::size_t>(
      {{50000, 32768, 50000, 1}, {65536, 65536, 65536, 1}, {50000, 65536, 65536, 2}}));
  pinned_host_test_fixture fixture{limit};
  auto* seed = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(seed, 9000);
  allocation_trace trace;
  trace.failures = std::numeric_limits<std::size_t>::max();
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::out_of_memory);
  }
  REQUIRE(trace.count == attempts);
  REQUIRE(trace.sizes[0] == first_size);
  if (attempts == 2) { REQUIRE(trace.sizes[1] == 50000); }
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == seed);
  fixture.resource.deallocate_sync(reused, 9000);
}

TEST_CASE("Recovery reuses a matching buffer returned during the exact attempt",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{128ULL << 10};
  auto* held = fixture.resource.allocate_sync(40000);
  struct returned_buffer {
    allocation_trace trace;
    cucascade::memory::small_pinned_host_memory_resource* resource;
    void* pointer;
    bool worker_failed{};
  } returned{{}, &fixture.resource, held};
  returned.trace.failures = 2;
  auto publish            = [](cucascade::test::fault_point_call const& call,
                    void* context) noexcept -> std::optional<cudaError_t> {
    auto& state = *static_cast<returned_buffer*>(context);
    auto status = allocation_trace::call(call, &state.trace);
    if (state.trace.count == 2) {
      try {
        std::jthread worker{[&] {
          state.resource->deallocate(cucascade::test::legacy_default_stream, state.pointer, 40000);
        }};
      } catch (...) {
        state.worker_failed = true;
      }
    }
    return status;
  };
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, publish, &returned};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(40000));
  }
  REQUIRE(pointer == held);
  REQUIRE_FALSE(returned.worker_failed);
  REQUIRE(returned.trace.count == 2);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_hits == 1);
  REQUIRE(statistics.large_misses == 2);
  REQUIRE(statistics.large_evictions == 0);
  fixture.resource.deallocate_sync(pointer, 40000);
  REQUIRE(fixture.resource.large_cache_bytes() == 65536);
}

TEST_CASE("Exact recovery distinguishes retained capacity below equal and above the request",
          "[small_pinned][fault_injection][recovery]")
{
  auto const [seed_count, retained_bytes, attempts, succeeds] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t, std::size_t, bool>(
      {{1, 32768, 1, false}, {2, 0, 2, true}, {3, 32768, 2, true}}));
  pinned_host_test_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{};
  for (std::size_t index = 0; index < seed_count; ++index) {
    seeds[index] = fixture.resource.allocate_sync(20000);
  }
  for (std::size_t index = 0; index < seed_count; ++index) {
    fixture.resource.deallocate_sync(seeds[index], 20000);
  }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    if (succeeds) {
      REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(65536));
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(65536), rmm::out_of_memory);
    }
  }
  REQUIRE(trace.count == attempts);
  REQUIRE(trace.sizes[0] == 65536);
  if (succeeds) { REQUIRE(trace.sizes[1] == 65536); }
  REQUIRE(fixture.resource.large_cache_bytes() == retained_bytes);
  REQUIRE(fixture.resource.get_statistics().large_evictions == (succeeds ? 2 : 0));
  if (pointer != nullptr) {
    fixture.resource.deallocate_sync(pointer, 65536);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }
}

TEST_CASE("Recovery does not purge on non-OOM failures and stops after its final exact attempt",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  auto const failures = GENERATE(
    std::size_t{0}, std::size_t{1}, std::size_t{2}, std::numeric_limits<std::size_t>::max());
  allocation_trace trace;
  trace.failures       = failures;
  trace.terminal_error = cudaErrorInvalidValue;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    if (failures == std::numeric_limits<std::size_t>::max()) {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(50000), rmm::out_of_memory);
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(50000), rmm::bad_alloc);
    }
  }
  REQUIRE(trace.count == (failures < 2 ? failures + 1 : 3));
  REQUIRE(fixture.resource.large_cache_bytes() == (failures < 2 ? 98304 : 32768));
  REQUIRE(fixture.resource.get_statistics().large_evictions == (failures < 2 ? 0 : 2));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Dedicated OOM reclamation uses the padded backing size",
          "[small_pinned][fault_injection][recovery]")
{
  bool const sufficient = GENERATE(false, true);
  pinned_host_test_fixture fixture;
  std::array<void*, 2> seeds{fixture.resource.allocate_sync(9000),
                             fixture.resource.allocate_sync(9000)};
  fixture.resource.deallocate_sync(seeds[0], 9000);
  if (sufficient) { fixture.resource.deallocate_sync(seeds[1], 9000); }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    if (sufficient) {
      REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(9000, 8192));
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(9000, 8192), rmm::out_of_memory);
    }
  }
  REQUIRE(trace.count == (sufficient ? 2 : 1));
  REQUIRE(trace.sizes[0] == 17191);
  if (sufficient) {
    REQUIRE(trace.sizes[1] == 17191);
    REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 8192 == 0);
    std::memset(pointer, 0x48, 9000);
    fixture.resource.deallocate_sync(pointer, 9000, 8192);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  } else {
    REQUIRE(fixture.resource.large_cache_bytes() == 16384);
    fixture.resource.deallocate_sync(seeds[1], 9000);
  }
}

TEST_CASE("Fallback metadata exhaustion preserves cached capacity before recovery",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{128ULL << 10};
  std::array<void*, 2> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_fault_hook metadata{cucascade::test::fault_point::dedicated_allocation,
                                                cudaErrorMemoryAllocation};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(65536), std::bad_alloc);
  }
  REQUIRE(trace.count == 1);
  REQUIRE(fixture.resource.large_cache_bytes() == 65536);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(20000);
  REQUIRE(reused == seeds[0]);
  fixture.resource.deallocate_sync(reused, 20000);
}

TEST_CASE("Unconfirmed reclamation withholds its victim without extending the retry budget",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  trace.failures = 2;
  void* pointer  = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_fault_hook completion{cucascade::test::fault_point::event_synchronize,
                                                  cudaErrorInvalidResourceHandle};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(50000));
  }
  REQUIRE(trace.count == 3);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  require_pointer_type(seeds[0], cudaMemoryTypeHost);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  fixture.resource.deallocate_sync(pointer, 50000);
  REQUIRE(cudaFreeHost(seeds[0]) == cudaSuccess);
}

TEST_CASE("Uncacheable exact recovery stays untracked and releases its original storage",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture{96ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_fault_hook metadata{cucascade::test::fault_point::dedicated_allocation,
                                                cudaErrorMemoryAllocation};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(70000));
  }
  REQUIRE(trace.count == 2);
  REQUIRE(trace.sizes[0] == 70000);
  REQUIRE(trace.sizes[1] == 70000);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 4);
  REQUIRE(statistics.large_uncacheable == 1);
  REQUIRE(statistics.large_evictions == 3);
  fixture.resource.deallocate_sync(pointer, 70000);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
}

TEST_CASE("Deferred OOM reporting preserves an unrelated CUDA error",
          "[small_pinned][fault_injection][recovery]")
{
  pinned_host_test_fixture fixture;
  cudaError_t unrelated_status = cudaSuccess;
  auto unrelated_error         = [](cucascade::test::fault_point_call const&,
                            void* context) noexcept -> std::optional<cudaError_t> {
    cudaDeviceProp properties{};
    *static_cast<cudaError_t*>(context) = cudaGetDeviceProperties(&properties, -1);
    return std::nullopt;
  };
  {
    cucascade::test::scoped_fault_hook allocation{cucascade::test::fault_point::host_alloc,
                                                  cudaErrorMemoryAllocation};
    cucascade::test::scoped_fault_hook metadata{
      cucascade::test::fault_point::dedicated_allocation, unrelated_error, &unrelated_status};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(65536), rmm::out_of_memory);
  }
  REQUIRE(unrelated_status == cudaErrorInvalidDevice);
  CHECK(cudaPeekAtLastError() == cudaErrorInvalidDevice);
  std::ignore = cudaGetLastError();
}

//===----------------------------------------------------------------------===//
// Alignment and dedicated storage
//===----------------------------------------------------------------------===//

namespace {

std::optional<cudaError_t> count_allocations(cucascade::test::fault_point_call const&,
                                             void* context) noexcept
{
  ++*static_cast<std::size_t*>(context);
  return std::nullopt;
}

struct dedicated_allocation_observation {
  void* base{};
  void* freed{};
  std::size_t bytes{};

  static std::optional<cudaError_t> allocate(cucascade::test::fault_point_call const& call,
                                             void* context) noexcept
  {
    auto& observation = *static_cast<dedicated_allocation_observation*>(context);
    observation.bytes = call.bytes;
    auto const status =
      cudaHostAlloc(&observation.base, call.bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { *call.buffer_out = observation.base; }
    return status;
  }

  static std::optional<cudaError_t> release(cucascade::test::fault_point_call const& call,
                                            void* context) noexcept
  {
    static_cast<dedicated_allocation_observation*>(context)->freed = call.buffer;
    return std::nullopt;
  }
};

}  // namespace

TEST_CASE("Nonzero pinned requests reject invalid alignment",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  auto const alignment = GENERATE(std::size_t{0}, std::size_t{3}, std::size_t{24});
  void* pointer        = nullptr;
  CHECK_THROWS_AS(pointer = fixture.resource.allocate_sync(128, alignment), std::invalid_argument);
  if (pointer != nullptr) { fixture.resource.deallocate_sync(pointer, 128, alignment); }
}

TEST_CASE("Alignment padding overflow fails before acquiring pinned storage",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  std::size_t allocations = 0;
  cucascade::test::scoped_fault_hook observe{
    cucascade::test::fault_point::host_alloc, count_allocations, &allocations};
  REQUIRE_THROWS_AS(fixture.resource.allocate_sync(std::numeric_limits<std::size_t>::max(), 64),
                    rmm::bad_alloc);
  REQUIRE(allocations == 0);
}

TEST_CASE("Dedicated metadata exhaustion acquires no pinned storage",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  std::size_t allocations = 0;
  cucascade::test::scoped_fault_hook observe{
    cucascade::test::fault_point::host_alloc, count_allocations, &allocations};
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::dedicated_allocation,
                                               cudaErrorMemoryAllocation};
    CHECK_THROWS_AS(pointer = fixture.resource.allocate_sync(9000, 64), std::bad_alloc);
  }
  if (pointer != nullptr) { fixture.resource.deallocate_sync(pointer, 9000, 64); }
  REQUIRE(allocations == 0);

  pointer = fixture.resource.allocate_sync(9000, 64);
  fixture.resource.deallocate_sync(pointer, 9000, 64);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
}

TEST_CASE("Dedicated storage uses alignment padding and frees its original allocation",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  auto const [alignment, allocation_bytes] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t>(
      {{64, 9063}, {4096, 13095}, {1048576, 1057575}}));
  dedicated_allocation_observation observation;
  cucascade::test::scoped_fault_hook allocate{cucascade::test::fault_point::host_alloc,
                                              dedicated_allocation_observation::allocate,
                                              &observation};
  cucascade::test::scoped_fault_hook release{cucascade::test::fault_point::host_free,
                                             dedicated_allocation_observation::release,
                                             &observation};

  auto* pointer = fixture.resource.allocate_sync(9000, alignment);
  REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0);
  REQUIRE(observation.bytes == allocation_bytes);
  require_pointer_type(pointer, cudaMemoryTypeHost);
  std::memset(pointer, 0x42, 9000);
  fixture.resource.deallocate_sync(pointer, 9000, alignment);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  REQUIRE(observation.freed == observation.base);
  require_pointer_type(observation.base, cudaMemoryTypeUnregistered);
}

TEST_CASE("Over-aligned small storage bypasses an awkward upstream block stride",
          "[small_pinned][fault_injection][alignment]")
{
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 8208, 8, 1};
  auto const initial_blocks = upstream.get_free_blocks();
  {
    cucascade::memory::small_pinned_host_memory_resource resource{upstream};
    for (std::size_t const bytes : std::array<std::size_t, 3>{1, 513, 8192}) {
      auto* pointer = resource.allocate_sync(bytes, 64);
      REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 64 == 0);
      std::memset(pointer, 0x53, bytes);
      resource.deallocate_sync(pointer, bytes, 64);
      require_pointer_type(pointer, cudaMemoryTypeUnregistered);
      REQUIRE(upstream.get_free_blocks() == initial_blocks);
    }

    auto* ordinary = resource.allocate_sync(513);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ordinary) % alignof(std::max_align_t) == 0);
    REQUIRE(upstream.get_free_blocks() < initial_blocks);
    resource.deallocate_sync(ordinary, 513);
    require_pointer_type(ordinary, cudaMemoryTypeHost);
  }
  REQUIRE(upstream.get_free_blocks() == initial_blocks);
}

TEST_CASE("Ordinary pooled storage does not require dedicated ownership metadata",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::dedicated_allocation,
                                             cudaErrorMemoryAllocation};
  auto* small = fixture.resource.allocate_sync(513);
  fixture.resource.deallocate_sync(small, 513);
  auto* large = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(large, 9000);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == large);
  fixture.resource.deallocate_sync(reused, 9000);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_hits == 1);
  REQUIRE(statistics.large_misses == 1);
  REQUIRE(statistics.large_uncacheable == 0);
}

TEST_CASE("Dedicated storage release withholds unconfirmed backing and recovers cleanly",
          "[small_pinned][fault_injection][alignment]")
{
  enum class release_failure { none, wait, free };
  auto const failure =
    GENERATE(release_failure::none, release_failure::wait, release_failure::free);
  auto const bytes = GENERATE(std::size_t{513}, std::size_t{9000});
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  dedicated_allocation_observation observation;
  cucascade::test::scoped_fault_hook allocate{cucascade::test::fault_point::host_alloc,
                                              dedicated_allocation_observation::allocate,
                                              &observation};
  auto* pointer = fixture.resource.allocate_sync(bytes, 64);
  {
    std::optional<cucascade::test::scoped_fault_hook> rejected;
    if (failure != release_failure::none) {
      rejected.emplace(failure == release_failure::wait
                         ? cucascade::test::fault_point::stream_synchronize
                         : cucascade::test::fault_point::host_free,
                       cudaErrorInvalidResourceHandle);
    }
    fixture.resource.deallocate(stream, pointer, bytes, 64);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  require_pointer_type(
    observation.base,
    failure == release_failure::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  if (failure != release_failure::none) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(observation.base) == cudaSuccess);
  }

  pointer = fixture.resource.allocate_sync(bytes, 64);
  fixture.resource.deallocate_sync(pointer, bytes, 64);
  require_pointer_type(observation.base, cudaMemoryTypeUnregistered);
}

TEST_CASE("Concurrent dedicated allocations preserve alignment and release ownership",
          "[small_pinned][fault_injection][alignment]")
{
  pinned_host_test_fixture fixture;
  constexpr std::size_t num_threads = 4;
  constexpr std::size_t iterations  = 12;
  std::array<std::array<void*, iterations>, num_threads> pointers{};
  std::array<std::exception_ptr, num_threads> errors{};
  {
    std::array<std::jthread, num_threads> workers;
    for (std::size_t thread = 0; thread < num_threads; ++thread) {
      workers[thread] = std::jthread{[&, thread] {
        try {
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto* pointer               = fixture.resource.allocate_sync(513, 64);
            pointers[thread][iteration] = pointer;
            std::memset(pointer, 0x64, 513);
            fixture.resource.deallocate_sync(pointer, 513, 64);
          }
        } catch (...) {
          errors[thread] = std::current_exception();
        }
      }};
    }
  }
  for (auto const& error : errors) {
    REQUIRE(error == nullptr);
  }
  for (auto const& row : pointers) {
    for (auto* pointer : row) {
      REQUIRE(pointer != nullptr);
      REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 64 == 0);
      require_pointer_type(pointer, cudaMemoryTypeUnregistered);
    }
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

namespace {

class captured_stream {
 public:
  explicit captured_stream(::cuda::stream_ref stream) : _stream(stream)
  {
    CUCASCADE_CUDA_TRY(cudaStreamBeginCapture(_stream.get(), cudaStreamCaptureModeThreadLocal));
  }

  captured_stream(captured_stream const&)            = delete;
  captured_stream& operator=(captured_stream const&) = delete;
  captured_stream(captured_stream&&)                 = delete;
  captured_stream& operator=(captured_stream&&)      = delete;

  ~captured_stream()
  {
    if (_active) { std::ignore = finish(); }
  }

  [[nodiscard]] cudaError_t finish() noexcept
  {
    cudaGraph_t graph = nullptr;
    auto const status = cudaStreamEndCapture(_stream.get(), &graph);
    _active           = false;
    if (status != cudaSuccess) { std::ignore = cudaGetLastError(); }
    if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) {
      std::ignore = cudaGetLastError();
    }
    return status;
  }

 private:
  ::cuda::stream_ref _stream;
  bool _active{true};
};

class observed_cuda_work {
 public:
  observed_cuda_work()
  {
    constexpr std::array operations{cucascade::test::fault_point::host_alloc,
                                    cucascade::test::fault_point::host_free,
                                    cucascade::test::fault_point::stream_synchronize,
                                    cucascade::test::fault_point::event_create,
                                    cucascade::test::fault_point::event_record,
                                    cucascade::test::fault_point::event_wait,
                                    cucascade::test::fault_point::event_synchronize,
                                    cucascade::test::fault_point::event_destroy};
    for (std::size_t i = 0; i < operations.size(); ++i) {
      _hooks[i].emplace(operations[i], count, &_calls);
    }
  }

  [[nodiscard]] std::size_t get_calls() const noexcept { return _calls; }

 private:
  static std::optional<cudaError_t> count(cucascade::test::fault_point_call const&,
                                          void* context) noexcept
  {
    ++*static_cast<std::size_t*>(context);
    return std::nullopt;
  }

  std::size_t _calls{};
  std::array<std::optional<cucascade::test::scoped_fault_hook>, 8> _hooks;
};

std::optional<cudaError_t> report_capture(cucascade::test::fault_point_call const& call,
                                          void*) noexcept
{
  *call.capture_status_out = cudaStreamCaptureStatusActive;
  return cudaSuccess;
}

enum class capture_rejection_mode { none, active, unknown };

class capture_rejection {
 public:
  capture_rejection(::cuda::stream_ref stream, capture_rejection_mode mode)
  {
    if (mode == capture_rejection_mode::active) {
      _capture.emplace(stream);
    } else if (mode == capture_rejection_mode::unknown) {
      _query_failure.emplace(
        cucascade::test::fault_point::stream_capture_status, fail_query, nullptr);
    }
  }

  [[nodiscard]] cudaError_t finish() noexcept
  {
    _query_failure.reset();
    return _capture ? _capture->finish() : cudaSuccess;
  }

 private:
  static std::optional<cudaError_t> fail_query(cucascade::test::fault_point_call const& call,
                                               void*) noexcept
  {
    *call.capture_status_out = cudaStreamCaptureStatusNone;
    return cudaSetDevice(-1);
  }

  std::optional<captured_stream> _capture;
  std::optional<cucascade::test::scoped_fault_hook> _query_failure;
};

struct dedicated_base {
  void* pointer{};

  static std::optional<cudaError_t> allocate(cucascade::test::fault_point_call const& call,
                                             void* context) noexcept
  {
    auto& base = static_cast<dedicated_base*>(context)->pointer;
    auto const status =
      cudaHostAlloc(&base, call.bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { *call.buffer_out = base; }
    return status;
  }
};

void release_registered_pointer(void* base)
{
  if (get_pointer_type(base) == cudaMemoryTypeHost) { REQUIRE(cudaFreeHost(base) == cudaSuccess); }
}

}  // namespace

TEST_CASE("Pinned allocation rejects a capturing stream without invalidating it",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 64ULL << 10};
  rmm::cuda_stream stream;
  auto const bytes     = GENERATE(std::size_t{512}, std::size_t{9000}, std::size_t{100000});
  auto const alignment = GENERATE(alignof(std::max_align_t), std::size_t{64});
  bool const seeded    = GENERATE(false, true);
  if (seeded) { resource.deallocate_sync(resource.allocate_sync(9000), 9000); }
  auto const before = resource.get_statistics();
  captured_stream capture{stream};
  void* pointer = nullptr;
  {
    observed_cuda_work observation;
    CHECK_THROWS_AS(pointer = resource.allocate(stream, bytes, alignment), cucascade::logic_error);
    CHECK(observation.get_calls() == 0);
  }
  auto const status = capture.finish();
  if (pointer != nullptr) { resource.deallocate_sync(pointer, bytes, alignment); }
  REQUIRE(status == cudaSuccess);
  auto const after = resource.get_statistics();
  REQUIRE(after.cached_large_bytes == before.cached_large_bytes);
  REQUIRE(after.large_requests == before.large_requests);
  REQUIRE(after.large_hits == before.large_hits);
  REQUIRE(after.large_misses == before.large_misses);
  REQUIRE(after.large_evictions == before.large_evictions);
}

TEST_CASE("Rejected pinned deallocation withholds storage without CUDA work",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 0};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  bool const reject      = mode != capture_rejection_mode::none;
  bool const synchronous = GENERATE(false, true);
  auto* pointer          = resource->allocate_sync(100000);
  std::optional<captured_stream> capture;
  std::optional<capture_rejection> query_failure;
  if (mode == capture_rejection_mode::active && !synchronous) { capture.emplace(stream); }
  if (mode == capture_rejection_mode::unknown) { query_failure.emplace(stream, mode); }
  {
    std::optional<cucascade::test::scoped_fault_hook> reported_capture;
    if (mode == capture_rejection_mode::active && synchronous) {
      reported_capture.emplace(
        cucascade::test::fault_point::stream_capture_status, report_capture, nullptr);
    }
    observed_cuda_work observation;
    if (synchronous) {
      resource->deallocate_sync(pointer, 100000);
    } else {
      resource->deallocate(stream, pointer, 100000);
    }
    CHECK((observation.get_calls() == 0) == reject);
  }
  if (capture) { CHECK(capture->finish() == cudaSuccess); }
  if (query_failure) { CHECK(query_failure->finish() == cudaSuccess); }
  CHECK(cudaGetLastError() == cudaSuccess);
  resource.reset();
  auto const type = get_pointer_type(pointer);
  CHECK(type == (reject ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered));
  if (type == cudaMemoryTypeHost) { REQUIRE(cudaFreeHost(pointer) == cudaSuccess); }
}

TEST_CASE("Rejected large deallocation preserves retained entries and withholds its buffer",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 16ULL << 10};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  auto* incoming = resource->allocate_sync(9000);
  auto* retained = resource->allocate_sync(9000);
  resource->deallocate_sync(retained, 9000);
  auto const before = resource->get_statistics();
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, incoming, 9000);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  auto const after = resource->get_statistics();
  CHECK(after.cached_large_bytes == before.cached_large_bytes);
  CHECK(after.large_evictions ==
        before.large_evictions + (mode == capture_rejection_mode::none ? 1 : 0));
  auto* reused = resource->allocate_sync(9000);
  CHECK(reused == (mode == capture_rejection_mode::none ? incoming : retained));
  resource->deallocate_sync(reused, 9000);
  resource.reset();
  CHECK(get_pointer_type(incoming) ==
        (mode == capture_rejection_mode::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost));
  CHECK(get_pointer_type(retained) == cudaMemoryTypeUnregistered);
  release_registered_pointer(incoming);
  release_registered_pointer(retained);
}

TEST_CASE("Rejected slab deallocation quarantines only its backing owner",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  auto const initial_free = fixture.upstream.get_free_blocks();
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  auto* incoming  = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  resource->deallocate_sync(completed, 8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, incoming, 512);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() ==
          initial_free - (mode == capture_rejection_mode::none ? 0 : 1));
}

TEST_CASE("Rejected dedicated deallocation never frees its original base",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  dedicated_base base;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook allocation{
      cucascade::test::fault_point::host_alloc, dedicated_base::allocate, &base};
    pointer = resource->allocate_sync(9000, 64);
  }
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, pointer, 9000, 64);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  resource.reset();
  CHECK(get_pointer_type(base.pointer) ==
        (mode == capture_rejection_mode::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost));
  release_registered_pointer(base.pointer);
}

TEST_CASE("Unknown capture status rejects allocation before changing retained capacity",
          "[small_pinned][fault_injection][capture]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream};
  rmm::cuda_stream stream;
  resource.deallocate_sync(resource.allocate_sync(9000), 9000);
  auto const before = resource.get_statistics();
  capture_rejection rejection{stream, capture_rejection_mode::unknown};
  {
    observed_cuda_work observation;
    CHECK_THROWS_AS(resource.allocate(stream, 9000), cucascade::cuda_error);
    REQUIRE(observation.get_calls() == 0);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  auto const after = resource.get_statistics();
  REQUIRE(after.cached_large_bytes == before.cached_large_bytes);
  REQUIRE(after.large_requests == before.large_requests);
}

//===----------------------------------------------------------------------===//
// Device selection
//===----------------------------------------------------------------------===//

namespace {

[[nodiscard]] int run_device_helper(char* mode)
{
  char executable[] = CUCASCADE_PINNED_HOST_DEVICE_HELPER;
  char* arguments[] = {executable, mode, nullptr};
  pid_t child{};
  REQUIRE(posix_spawn(&child, executable, nullptr, nullptr, arguments, environ) == 0);
  int status{};
  pid_t waited{};
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  REQUIRE(waited == child);
  REQUIRE(WIFEXITED(status));
  return WEXITSTATUS(status);
}

}  // namespace

TEST_CASE("Failed stream device discovery does not publish a cached buffer",
          "[small_pinned][fault_injection][device]")
{
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::stream_device,
                                               cudaErrorInvalidResourceHandle};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Failed device restoration terminates pinned cleanup",
          "[small_pinned][fault_injection][device]")
{
  char mode[] = "fail-restore";
  REQUIRE(run_device_helper(mode) == 86);
}

TEST_CASE("Successful event creation restores the previous device",
          "[small_pinned][fault_injection][device]")
{
  char mode[] = "restore";
  REQUIRE(run_device_helper(mode) == 0);
}

TEST_CASE("Ordinary device and event failures require confirmed stream completion",
          "[small_pinned][fault_injection][device]")
{
  using cucascade::test::fault_point;
  using cucascade::test::scoped_fault_hook;
  auto const operation         = GENERATE(fault_point::stream_device,
                                  fault_point::current_device,
                                  fault_point::event_create,
                                  fault_point::event_record);
  bool const reject_completion = GENERATE(false, true);
  CAPTURE(operation, reject_completion);
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  {
    std::optional<scoped_fault_hook> completion;
    if (reject_completion) {
      completion.emplace(fault_point::stream_synchronize, cudaErrorInvalidResourceHandle);
    }
    scoped_fault_hook failure{operation, cudaErrorInvalidValue};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer,
                       reject_completion ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (reject_completion) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  auto* next = fixture.resource.allocate_sync(bytes);
  fixture.resource.deallocate_sync(next, bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
}

TEST_CASE("Failed initial device switching does not change the caller's device",
          "[small_pinned][fault_injection][device]")
{
  using cucascade::test::fault_point;
  using cucascade::test::scoped_fault_hook;
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  int actual_device{};
  REQUIRE(cudaGetDevice(&actual_device) == cudaSuccess);
  int other_device = actual_device + 1;
  {
    scoped_fault_hook current{fault_point::current_device,
                              [](cucascade::test::fault_point_call const& call,
                                 void* context) noexcept -> std::optional<cudaError_t> {
                                *call.device_out = *static_cast<int*>(context);
                                return cudaSuccess;
                              },
                              &other_device};
    scoped_fault_hook failed_switch{fault_point::set_device, cudaErrorInvalidDevice};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  int restored_device{};
  REQUIRE(cudaGetDevice(&restored_device) == cudaSuccess);
  REQUIRE(restored_device == actual_device);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

int main(int argc, char** argv) { return Catch::Session().run(argc, argv); }
