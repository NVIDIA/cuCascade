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

#include "utils/test_memory_resources.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <memory>

using namespace cucascade::memory;

namespace {

std::unique_ptr<memory_reservation_manager> make_manager()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(256ull << 20);
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(0.75);
  builder.set_per_numa_region_capacity(256ull << 20);
  builder.use_gpu_id_as_host_id();
  builder.set_reservation_fraction_per_numa_region(0.75);
  return std::make_unique<memory_reservation_manager>(builder.build());
}

}  // namespace

// A multiple_blocks_allocation keeps a pointer to the reservation it was made against. When that
// reservation is released before the allocation is freed, a later reservation can be allocated at
// the same address. Freeing the allocation must not credit the new reservation: its blocks have to
// go back to the resource, so available memory returns to capacity once everything is released.
TEST_CASE("Freeing an allocation after its reservation's address was reused returns its bytes",
          "[fixed_size_host][reservation]")
{
  auto manager = make_manager();
  auto* space  = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(space != nullptr);
  auto* mr = space->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  constexpr std::size_t bytes = 4ull << 20;
  auto const available_before = mr->get_available_memory();
  auto const allocated_before = mr->get_total_allocated_bytes();

  // R1, an allocation under it, then R1 released while the allocation lives on
  auto arena1             = mr->reserve(bytes);
  auto const* first_arena = arena1.get();
  auto r1                 = reservation::create(*space, std::move(arena1));
  auto allocation         = mr->allocate_multiple_blocks(bytes, r1.get());
  r1.reset();

  // R2 at R1's address (the allocator normally hands the freed object back at once)
  std::unique_ptr<reserved_arena> arena2;
  for (int attempt = 0; attempt < 64 && arena2.get() != first_arena; ++attempt) {
    arena2.reset();
    arena2 = mr->reserve(bytes);
  }
  if (arena2.get() != first_arena) {
    SKIP("a new reservation never reused the released one's address");
  }
  auto r2 = reservation::create(*space, std::move(arena2));

  // the allocation freed while R2 holds R1's address, then R2 released
  allocation.reset();
  r2.reset();

  REQUIRE(mr->get_active_reservation_count() == 0);
  REQUIRE(mr->get_total_allocated_bytes() == allocated_before);
  REQUIRE(mr->get_available_memory() == available_before);
}
