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
 * [memory_space] - Basic memory space functionality tests
 * [threading] - Multi-threaded tests
 * [gpu] - GPU-specific tests requiring CUDA
 * [.multi-device] - Tests requiring multiple GPU devices (hidden by default)
 *
 * Running tests:
 * - Default (includes single GPU): ./test_executable
 * - Include multi-device tests: ./test_executable "[.multi-device]"
 * - Exclude multi-device tests: ./test_executable "~[.multi-device]"
 * - Run all tests: ./test_executable "[memory_space]"
 */

#include "utils/test_memory_resources.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <catch2/catch.hpp>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

using namespace cucascade::memory;

// Expected memory capacities
const size_t expected_gpu_capacity  = 2ull << 30;  // 2GB
const size_t expected_host_capacity = 4ull << 30;  // 4GB
const double limit_ratio            = 0.75;

std::unique_ptr<memory_reservation_manager> createSingleDeviceMemoryManager()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(expected_gpu_capacity);  // 2 GB
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(limit_ratio);
  builder.set_per_host_capacity(expected_host_capacity);  //  4 GB
  builder.use_host_per_gpu();
  builder.set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  return std::make_unique<memory_reservation_manager>(std::move(space_configs));
}

// PER_THREAD-mode manager: lookup uses calling thread's thread_local, ignoring stream.
// This is the mode sirius runs in (per_stream_reservation=false).
std::unique_ptr<memory_reservation_manager> createSingleDeviceMemoryManagerPerThread()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(expected_gpu_capacity);
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(limit_ratio);
  builder.set_per_host_capacity(expected_host_capacity);
  builder.use_host_per_gpu();
  builder.set_reservation_fraction_per_host(limit_ratio);
  builder.track_reservation_per_stream(false);

  auto space_configs = builder.build();
  return std::make_unique<memory_reservation_manager>(std::move(space_configs));
}

std::unique_ptr<memory_reservation_manager> createDualGpuMemoryManager()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(expected_gpu_capacity);  // 2 GB
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(limit_ratio);
  builder.set_per_host_capacity(expected_host_capacity);  //  4 GB
  builder.set_number_of_gpus(2);
  builder.use_host_per_gpu();
  builder.set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  return std::make_unique<memory_reservation_manager>(std::move(space_configs));
}

TEST_CASE("Single-Device Memory Space Access", "[memory_space]")
{
  auto manager = createSingleDeviceMemoryManager();

  // Test single GPU memory space
  auto gpu_device_0 = manager->get_memory_space(Tier::GPU, 0);

  REQUIRE(gpu_device_0 != nullptr);
  REQUIRE(gpu_device_0->get_tier() == Tier::GPU);
  REQUIRE(gpu_device_0->get_device_id() == 0);
  REQUIRE(gpu_device_0->get_max_memory() == expected_gpu_capacity * limit_ratio);
  REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity);

  // Test single HOST memory space (NUMA node)
  auto host_numa_0 = manager->get_memory_space(Tier::HOST, 0);

  REQUIRE(host_numa_0 != nullptr);
  REQUIRE(host_numa_0->get_tier() == Tier::HOST);
  REQUIRE(host_numa_0->get_device_id() == 0);
  REQUIRE(host_numa_0->get_max_memory() == expected_host_capacity * limit_ratio);
  REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity);

  // Test non-existent devices (only device 0 exists for each tier)
  REQUIRE(manager->get_memory_space(Tier::GPU, 1) == nullptr);
  REQUIRE(manager->get_memory_space(Tier::HOST, 1) == nullptr);

  // Verify all memory spaces are different objects
  REQUIRE(gpu_device_0 != host_numa_0);
}

TEST_CASE("Device-Specific Memory Reservations", "[memory_space]")
{
  auto manager = createSingleDeviceMemoryManager();

  // Memory size constants
  const size_t gpu_allocation_size  = 200ull * 1024 * 1024;   // 200MB
  const size_t host_allocation_size = 500ull * 1024 * 1024;   // 500MB
  const size_t disk_allocation_size = 1000ull * 1024 * 1024;  // 1GB

  auto gpu_device_0 = manager->get_memory_space(Tier::GPU, 0);
  auto host_numa_0  = manager->get_memory_space(Tier::HOST, 0);

  {
    // Test reservation on GPU device
    auto gpu_reservation =
      manager->request_reservation(specific_memory_space(Tier::GPU, 0), gpu_allocation_size);
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(gpu_reservation->tier() == Tier::GPU);
    REQUIRE(gpu_reservation->device_id() == 0);
    REQUIRE(gpu_reservation->size() == gpu_allocation_size);

    // Check memory accounting on GPU device
    REQUIRE(gpu_device_0->get_total_reserved_memory() == gpu_allocation_size);
    REQUIRE(gpu_device_0->get_active_reservation_count() == 1);
    REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity - gpu_allocation_size);

    // // Check that other devices are unaffected
    REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
    REQUIRE(host_numa_0->get_active_reservation_count() == 0);

    // // Test reservation on HOST NUMA node
    auto host_reservation =
      manager->request_reservation(any_memory_space_in_tier(Tier::HOST), host_allocation_size);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(host_reservation->tier() == Tier::HOST);
    REQUIRE(host_reservation->device_id() == 0);
    REQUIRE(host_reservation->size() == host_allocation_size);

    // // Check HOST memory accounting
    REQUIRE(host_numa_0->get_total_reserved_memory() == host_allocation_size);
    REQUIRE(host_numa_0->get_active_reservation_count() == 1);
    REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity - host_allocation_size);
  }

  // Verify cleanup
  REQUIRE(gpu_device_0->get_total_reserved_memory() == 0);
  REQUIRE(gpu_device_0->get_active_reservation_count() == 0);
  REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity);

  REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
  REQUIRE(host_numa_0->get_active_reservation_count() == 0);
  REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity);
}

TEST_CASE("Reservation Strategies with Single Device", "[memory_space]")
{
  auto manager = createSingleDeviceMemoryManager();

  // Test allocation sizes
  const size_t small_allocation  = 25ull * 1024 * 1024;   // 25MB
  const size_t medium_allocation = 50ull * 1024 * 1024;   // 50MB
  const size_t large_allocation  = 100ull * 1024 * 1024;  // 100MB

  // Test requesting reservation in any GPU
  auto gpu_any_reservation =
    manager->request_reservation(any_memory_space_in_tier(Tier::GPU), medium_allocation);
  REQUIRE(gpu_any_reservation != nullptr);
  REQUIRE(gpu_any_reservation->tier() == Tier::GPU);
  REQUIRE(gpu_any_reservation->size() == medium_allocation);

  // Should pick the single GPU device (device 0)
  REQUIRE(gpu_any_reservation->device_id() == 0);

  // Test requesting reservation across multiple tiers (simulates "anywhere")
  std::vector<Tier> any_tier_preferences = {Tier::GPU, Tier::HOST, Tier::DISK};
  auto anywhere_reservation =
    manager->request_reservation(any_memory_space_in_tiers(any_tier_preferences), small_allocation);
  REQUIRE(anywhere_reservation != nullptr);
  REQUIRE(anywhere_reservation->size() == small_allocation);

  // Should pick any available memory space
  Tier selected_tier = anywhere_reservation->tier();
  REQUIRE(
    (selected_tier == Tier::GPU || selected_tier == Tier::HOST || selected_tier == Tier::DISK));

  // Test specific memory space in tiers list with HOST preference
  std::vector<Tier> tier_preferences = {Tier::HOST, Tier::GPU, Tier::DISK};
  auto preference_reservation =
    manager->request_reservation(any_memory_space_in_tiers(tier_preferences), large_allocation);
  REQUIRE(preference_reservation != nullptr);
  REQUIRE(preference_reservation->size() == large_allocation);

  // Should prefer HOST first
  REQUIRE(preference_reservation->tier() == Tier::HOST);
}

SCENARIO("multi-reservation cross-stream dealloc preserves origin attribution", "[memory_space]")
{
  // Origin tracking debits the reservation that was active at *alloc time*, regardless
  // of which stream runs the dealloc.
  auto manager = createSingleDeviceMemoryManager();

  const size_t res_size         = 1ull * 1024 * 1024;  // 1MB
  const size_t small_alloc_size = res_size / 2;        // 512KB (fits in res arena)
  const size_t large_alloc_size = res_size * 2;        // 2MB  (over-reserves by 1MB)

  GIVEN("Two reservations of 1MB each attached to two different streams")
  {
    auto res1 = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    auto res2 = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);

    auto* mr = res1->get_memory_resource_of<Tier::GPU>();
    rmm::cuda_stream stream1, stream2;

    mr->attach_reservation_to_tracker(stream1, std::move(res1));
    mr->attach_reservation_to_tracker(stream2, std::move(res2));

    WHEN("buffers are allocated on each stream and then cross-deallocated")
    {
      auto upstream_leftover = mr->get_available_memory();
      REQUIRE(mr->get_available_memory(stream1) == upstream_leftover + res_size);
      REQUIRE(mr->get_available_memory(stream2) == upstream_leftover + res_size);
      auto* buff1 = mr->allocate(stream1, small_alloc_size, alignof(std::max_align_t));
      REQUIRE(mr->get_allocated_bytes(stream1) == small_alloc_size);
      REQUIRE(mr->get_available_memory(stream1) ==
              mr->get_available_memory() + res_size - small_alloc_size);

      auto* buff2 = mr->allocate(stream2, large_alloc_size, alignof(std::max_align_t));
      REQUIRE(mr->get_allocated_bytes(stream2) == large_alloc_size);
      REQUIRE(mr->get_available_memory(stream2) == mr->get_available_memory());

      THEN("each allocation is debited from its origin reservation, not the dealloc stream's")
      {
        // buff2 was allocated under res2; freeing it on stream1 still credits res2.
        mr->deallocate(stream1, buff2, large_alloc_size, alignof(std::max_align_t));
        CHECK(mr->get_allocated_bytes(stream1) == small_alloc_size);
        CHECK(mr->get_allocated_bytes(stream2) == 0);

        mr->deallocate(stream2, buff1, small_alloc_size, alignof(std::max_align_t));
        CHECK(mr->get_allocated_bytes(stream1) == 0);
        CHECK(mr->get_allocated_bytes(stream2) == 0);

        CHECK(mr->get_available_memory(stream1) == mr->get_available_memory() + res_size);
        CHECK(mr->get_available_memory(stream2) == mr->get_available_memory() + res_size);
      }
    }
  }
}

SCENARIO("PER_THREAD mode: cross-thread dealloc preserves origin attribution",
         "[memory_space][per_thread]")
{
  // In PER_THREAD mode (sirius's config), reservation lookup uses thread_local on the
  // calling CPU thread. Cross-thread buffer lifetimes must still debit the origin.
  auto manager = createSingleDeviceMemoryManagerPerThread();

  const size_t res_size   = 1ull * 1024 * 1024;  // 1MB
  const size_t alloc_size = res_size / 2;        // 512KB (fits in arena)

  GIVEN("Thread A allocates with no reservation; thread B holds reservation R_b")
  {
    auto res_b = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    auto* mr   = res_b->get_memory_resource_of<Tier::GPU>();
    rmm::cuda_stream stream;

    void* unowned_buff = nullptr;
    std::thread thread_a([&] {
      // No reservation attached on this thread; alloc charges global only.
      unowned_buff = mr->allocate(stream, alloc_size, alignof(std::max_align_t));
    });
    thread_a.join();
    REQUIRE(unowned_buff != nullptr);

    WHEN("thread B attaches R_b to its thread_local and frees thread A's buffer")
    {
      // Probe via get_available_memory; get_allocated_bytes would clamp a negative to 0.
      std::size_t r_b_avail_after_dealloc      = 0;
      std::size_t upstream_avail_after_dealloc = 0;
      std::thread thread_b([&] {
        mr->attach_reservation_to_tracker(stream, std::move(res_b));
        REQUIRE(mr->get_allocated_bytes(stream) == 0);

        mr->deallocate(stream, unowned_buff, alloc_size, alignof(std::max_align_t));

        upstream_avail_after_dealloc = mr->get_available_memory();
        r_b_avail_after_dealloc      = mr->get_available_memory(stream);

        mr->reset_stream_reservation(stream);
      });
      thread_b.join();

      THEN("R_b's arena reports available_memory == arena_size (no spurious credit)")
      {
        // Pre-fix: R_b.allocated_bytes -> -alloc_size, available = arena_size + alloc_size.
        CHECK(r_b_avail_after_dealloc == upstream_avail_after_dealloc + res_size);
      }
    }
  }

  GIVEN("Thread A holds R_a and allocates; thread B holds R_b and deallocates")
  {
    auto res_a = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    auto res_b = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    auto* mr   = res_a->get_memory_resource_of<Tier::GPU>();
    rmm::cuda_stream stream;

    void* a_buff = nullptr;

    std::thread thread_a([&, &res_a_ref = res_a] {
      mr->attach_reservation_to_tracker(stream, std::move(res_a_ref));
      a_buff = mr->allocate(stream, alloc_size, alignof(std::max_align_t));
      REQUIRE(mr->get_allocated_bytes(stream) == alloc_size);
      mr->reset_stream_reservation(stream);
    });
    thread_a.join();
    REQUIRE(a_buff != nullptr);

    WHEN("thread B attaches R_b and frees the buffer that R_a allocated")
    {
      std::size_t r_b_avail_after_dealloc      = 0;
      std::size_t upstream_avail_after_dealloc = 0;
      std::thread thread_b([&, &res_b_ref = res_b] {
        mr->attach_reservation_to_tracker(stream, std::move(res_b_ref));
        REQUIRE(mr->get_allocated_bytes(stream) == 0);

        mr->deallocate(stream, a_buff, alloc_size, alignof(std::max_align_t));

        upstream_avail_after_dealloc = mr->get_available_memory();
        r_b_avail_after_dealloc      = mr->get_available_memory(stream);

        mr->reset_stream_reservation(stream);
      });
      thread_b.join();

      THEN("R_b's arena is untouched; dealloc went to R_a (already released — global only)")
      {
        // Pre-fix: dealloc would debit R_b instead of R_a, leaking R_a's allocation into R_b.
        CHECK(r_b_avail_after_dealloc == upstream_avail_after_dealloc + res_size);
      }
    }
  }
}

SCENARIO("Peak Tracking On Streams with Reservation", "[memory_space][tracking]")
{
  auto manager                  = createSingleDeviceMemoryManager();
  const size_t reservation_size = 2048;
  const size_t chunk_size       = 1024;

  GIVEN("A reservation of specific size[= 2048] on GPU")
  {
    auto res = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
    REQUIRE(res->size() == reservation_size);
    REQUIRE(res->tier() == Tier::GPU);
    auto* mr = res->get_memory_resource_of<Tier::GPU>();
    REQUIRE(mr != nullptr);

    rmm::cuda_stream other_streams;
    rmm::cuda_stream reserved_stream;

    std::vector<rmm::device_buffer> ptrs;
    mr->attach_reservation_to_tracker(reserved_stream, std::move(res));

    THEN("reservation is reflected in peak allocated bytes in upstream")
    {
      REQUIRE(mr->get_peak_total_allocated_bytes() == reservation_size);
      REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == 0);
      REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
    }

    WHEN("allocation within reservation on reserved stream, upstream peak doesn't change")
    {
      ptrs.emplace_back(
        chunk_size, reserved_stream, rmm::device_async_resource_ref{*mr});  // within reservation

      THEN("upstream peak allocated bytes remain the same, only stream peak changes")
      {
        REQUIRE(mr->get_peak_total_allocated_bytes() == reservation_size);
        REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == chunk_size);
        REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
      }
    }

    WHEN("allocation exceeds reservation on reserved stream, upstream tracks overflow")
    {
      ptrs.emplace_back(
        chunk_size, reserved_stream, rmm::device_async_resource_ref{*mr});  // within reservation
      ptrs.emplace_back(
        chunk_size, reserved_stream, rmm::device_async_resource_ref{*mr});  // fills reservation
      ptrs.emplace_back(
        chunk_size, reserved_stream, rmm::device_async_resource_ref{*mr});  // exceeds reservation

      THEN("peak allocated bytes are tracked correctly")
      {
        // 3 x chunk_size on stream, but only chunk_size overflows beyond the reservation
        auto total_allocated_bytes = mr->get_total_allocated_bytes();
        REQUIRE(total_allocated_bytes == reservation_size + chunk_size);
        REQUIRE(mr->get_peak_total_allocated_bytes() == total_allocated_bytes);
        REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == 3 * chunk_size);
        REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
      }

      WHEN("allocations are freed")
      {
        std::size_t peak_stream_bytes = mr->get_peak_allocated_bytes(reserved_stream);

        REQUIRE(mr->get_total_allocated_bytes() == reservation_size + chunk_size);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == 3 * chunk_size);
        mr->reset_stream_reservation(reserved_stream);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size + chunk_size);
        ptrs.clear();
        REQUIRE(mr->get_total_allocated_bytes() == 0);

        THEN("peak allocated bytes are tracked correctly")
        {
          REQUIRE(mr->get_peak_total_allocated_bytes() == peak_stream_bytes);
          REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) ==
                  0);                                     // doesn't have a tracker attached
          REQUIRE(mr->get_total_allocated_bytes() == 0);  // doesn't have a tracker attached
        }
      }
    }
  }
}

SCENARIO("Reservation Concepts on Single Gpu Manager", "[memory_space]")
{
  auto manager                  = createSingleDeviceMemoryManager();
  const size_t reservation_size = 1024;

  GIVEN("A single gpu manager")
  {
    WHEN("a reservation is made with overflow policy to ignore")
    {
      auto res =
        manager->request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->size() == reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      auto* mr = res->get_memory_resource_as<reservation_aware_resource_adaptor>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream reserved_stream;
      rmm::cuda_stream other_streams;
      mr->attach_reservation_to_tracker(reserved_stream, std::move(res));

      THEN("upstream and others see it as allocated/unavailable")
      {
        REQUIRE(mr->get_total_allocated_bytes() == 1024);
        REQUIRE(mr->get_available_memory(other_streams) ==
                expected_gpu_capacity - reservation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
      }

      THEN("only reserved stream has access to it")
      {
        REQUIRE(mr->get_available_memory(reserved_stream) == expected_gpu_capacity);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == 0);
      }

      THEN("allocation within the reservations are seen by upstream/other stream")
      {
        std::size_t allocation_size = 512;
        void* ptr = mr->allocate(reserved_stream, allocation_size, alignof(std::max_align_t));
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
        REQUIRE(mr->get_available_memory(other_streams) ==
                expected_gpu_capacity - reservation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
        REQUIRE(mr->get_available_memory(reserved_stream) ==
                expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == allocation_size);
        mr->deallocate(reserved_stream, ptr, allocation_size, alignof(std::max_align_t));
      }

      THEN("allocation beyond the reservations are made from the upstream")
      {
        std::size_t allocation_size = reservation_size * 2;
        void* ptr = mr->allocate(reserved_stream, allocation_size, alignof(std::max_align_t));
        REQUIRE(mr->get_total_allocated_bytes() == allocation_size);
        REQUIRE(mr->get_available_memory(other_streams) == expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
        REQUIRE(mr->get_available_memory(reserved_stream) ==
                expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == allocation_size);
        mr->deallocate(reserved_stream, ptr, allocation_size, alignof(std::max_align_t));
      }
    }
  }
}

SCENARIO("Reservation Overflow Policy", "[memory_space][.overflow_policy]")
{
  auto manager                  = createSingleDeviceMemoryManager();
  const size_t reservation_size = 1024;

  GIVEN("A single gpu manager")
  {
    WHEN("allocation beyond reservation with ignore policy")
    {
      auto res =
        manager->request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);
      auto* mr = res->get_memory_resource_of<Tier::GPU>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<ignore_reservation_limit_policy>());

      THEN("total reservation doesn't change")
      {
        auto* buffer = mr->allocate(stream, reservation_size * 2, alignof(std::max_align_t));
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size * 2);
        mr->deallocate(stream, buffer, reservation_size * 2, alignof(std::max_align_t));
      }
    }

    WHEN("allocation beyond reservation with fail policy")
    {
      auto res =
        manager->request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);

      auto* mr = res->get_memory_resource_as<reservation_aware_resource_adaptor>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<fail_reservation_limit_policy>());

      THEN("oom on allocation")
      {
        REQUIRE_THROWS_AS(mr->allocate(stream, reservation_size * 2, alignof(std::max_align_t)),
                          rmm::bad_alloc);
      }
    }

    WHEN("allocation beyond reservation with increase policy")
    {
      auto res =
        manager->request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);

      auto* mr = res->get_memory_resource_of<Tier::GPU>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<increase_reservation_limit_policy>(2.0));

      THEN("increased reservation on allocation")
      {
        auto* buffer = mr->allocate(stream, reservation_size * 2, alignof(std::max_align_t));
        REQUIRE(mr->get_total_reserved_bytes() >= reservation_size * 2);
        REQUIRE(mr->get_total_allocated_bytes() >= reservation_size * 2);
        mr->deallocate(stream, buffer, reservation_size * 2, alignof(std::max_align_t));
      }
    }
  }
}

SCENARIO("Reservation On Multi Gpu System", "[memory_space][.multi-device]")
{
  auto manager = createDualGpuMemoryManager();

  auto gpu_device_0 = manager->get_memory_space(Tier::GPU, 0);
  auto gpu_device_1 = manager->get_memory_space(Tier::GPU, 1);
  auto host_numa_0  = manager->get_memory_space(Tier::HOST, 0);

  // Test that we can get default allocators from each device
  auto gpu_0_allocator  = gpu_device_0->get_default_allocator();
  auto gpu_1_allocator  = gpu_device_1->get_default_allocator();
  auto host_0_allocator = host_numa_0->get_default_allocator();

  // Test that allocators are valid (basic smoke test)
  (void)gpu_0_allocator;
  (void)gpu_1_allocator;
  (void)host_0_allocator;

  GIVEN("Dual gpu manager")
  {
    auto* gpu_space = manager->get_memory_space(Tier::GPU, 0);
    auto* mr        = gpu_space->get_memory_resource_as<reservation_aware_resource_adaptor>();
    REQUIRE(mr != nullptr);

    WHEN("a reservation doesn't fit on gpu 0 but fits on gpu 1")
    {
      size_t large_reservation = expected_gpu_capacity * limit_ratio - 1024;
      auto res =
        manager->request_reservation(specific_memory_space{Tier::GPU, 0}, large_reservation);
      REQUIRE(res->size() == large_reservation);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->device_id() == 0);

      THEN("reservation made on gpu 1")
      {
        auto other_res =
          manager->request_reservation(any_memory_space_in_tier{Tier::GPU}, large_reservation);
        REQUIRE(other_res->size() == large_reservation);
        REQUIRE(other_res->tier() == Tier::GPU);
        REQUIRE(other_res->device_id() == 1);
      }
    }
  }
}

SCENARIO("Host Reservation", "[memory_space][host_reservation]")
{
  auto manager                 = createSingleDeviceMemoryManager();
  std::size_t reservation_size = 2UL << 20;
  std::size_t small_allocation = 1UL << 20;
  std::size_t large_allocation = 4UL << 20;

  GIVEN("making a host reservation")
  {
    auto reservation =
      manager->request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
    REQUIRE(reservation->size() == reservation_size);
    REQUIRE(reservation->tier() == Tier::HOST);
    auto* mr = reservation->get_memory_resource_of<Tier::HOST>();
    REQUIRE(mr != nullptr);
    REQUIRE(mr->get_total_allocated_bytes() == reservation_size);

    WHEN("allocation made larger than reservation")
    {
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(large_allocation, reservation.get());

      THEN("upstream and others see it as allocated/unavailable")
      {
        REQUIRE(mr->get_available_memory() < free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == large_allocation);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held, extra is freed")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }
    }

    WHEN("allocation made fits inside reservation")
    {
      rmm::cuda_stream stream;
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(small_allocation, reservation.get());

      THEN("upstream and others see doesn't change")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }
    }

    WHEN("reservation is freed before allocation")
    {
      rmm::cuda_stream stream;
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(small_allocation, reservation.get());
      reservation.reset();

      THEN("upstream shrink the reservation to fit")
      {
        REQUIRE(mr->get_total_reserved_bytes() == 0);
        REQUIRE(mr->get_total_allocated_bytes() == small_allocation);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held")
      {
        REQUIRE(mr->get_total_reserved_bytes() == 0);
        REQUIRE(mr->get_total_allocated_bytes() == 0);
      }
    }
  }
}

TEST_CASE("Concurrent reset and deallocate on the same stream are safe",
          "[memory_space][threading]")
{
  // Reset and in-flight deallocates from the same stream race; per iteration the global
  // counter must return to baseline. Catches lifetime / accounting drift between
  // reset_stream_reservation and concurrent deallocate paths.
  auto manager = createSingleDeviceMemoryManager();

  const size_t res_size  = 4ull * 1024 * 1024;
  const size_t buf_size  = 4 * 1024;
  const size_t k_buffers = 32;
  const size_t n_iters   = 500;

  // Capture the resource adaptor pointer via a throwaway reservation; it is owned by the
  // manager and stays valid past seed_res's drop. baseline reads after the drop.
  reservation_aware_resource_adaptor* mr = nullptr;
  {
    auto seed_res = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    REQUIRE(seed_res != nullptr);
    mr = seed_res->get_memory_resource_of<Tier::GPU>();
  }
  const std::size_t baseline_global = mr->get_total_allocated_bytes();

  for (size_t i = 0; i < n_iters; ++i) {
    auto res = manager->request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    REQUIRE(res != nullptr);

    rmm::cuda_stream stream;
    REQUIRE(mr->attach_reservation_to_tracker(stream, std::move(res)));

    std::vector<void*> buffers(k_buffers, nullptr);
    for (size_t j = 0; j < k_buffers; ++j) {
      buffers[j] = mr->allocate(stream, buf_size, alignof(std::max_align_t));
      REQUIRE(buffers[j] != nullptr);
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(k_buffers + 1);

    for (size_t j = 0; j < k_buffers; ++j) {
      threads.emplace_back([&, j]() {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) {}
        mr->deallocate(stream, buffers[j], buf_size, alignof(std::max_align_t));
      });
    }
    threads.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!go.load(std::memory_order_acquire)) {}
      mr->reset_stream_reservation(stream);
    });

    while (ready.load(std::memory_order_acquire) < static_cast<int>(k_buffers + 1)) {}
    go.store(true, std::memory_order_release);

    for (auto& t : threads) { t.join(); }

    REQUIRE(mr->get_total_allocated_bytes() == baseline_global);
  }
}
