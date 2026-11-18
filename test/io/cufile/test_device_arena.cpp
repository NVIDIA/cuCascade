/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/io/cufile/capacity.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/device_arena.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace cucascade::io::cufile;

namespace {
constexpr std::size_t mib  = 1UL << 20;
constexpr std::size_t slab = device_arena::slab_bytes;

async_engine_config arena_config(std::size_t bytes = slab)
{
  async_engine_config config;
  config.mode        = engine_mode::async_simulated_for_testing;
  config.arena_bytes = bytes;
  return config;
}

struct fake_resources {
  std::size_t allocations{};
  std::size_t frees{};
  std::size_t registrations{};
  std::size_t deregistrations{};
  std::size_t fail_allocation{std::numeric_limits<std::size_t>::max()};
  std::size_t fail_registration{std::numeric_limits<std::size_t>::max()};
  std::size_t max_registration_bytes{std::numeric_limits<std::size_t>::max()};
  bool throw_allocate{};
  bool throw_register{};
  bool fail_deregister{};
  bool fail_free{};
  bool fail_release{};
  std::size_t charged{};
  std::size_t reserve_calls{};
  std::size_t grant_limit{std::numeric_limits<std::size_t>::max()};
  std::optional<arena_memory_info> memory{arena_memory_info{8UL << 30, 1UL << 30}};
  std::optional<std::size_t> bar1;
  std::optional<std::size_t> pin;
  std::unordered_map<void*, std::size_t> memory_blocks;
  std::unordered_map<void*, std::size_t> active_registrations;
  std::vector<std::pair<void*, std::size_t>> registration_log;

  ~fake_resources()
  {
    // The fake, not the arena, owns host stand-ins for intentionally persistent device memory.
    for (auto const& [ptr, bytes] : memory_blocks) {
      static_cast<void>(bytes);
      std::free(ptr);
    }
  }
};

device_arena_hooks fake_hooks(std::shared_ptr<fake_resources> const& fake)
{
  device_arena_hooks hooks;
  hooks.allocate = [fake](int, std::size_t bytes) -> void* {
    if (fake->allocations++ == fake->fail_allocation) {
      if (fake->throw_allocate) { throw std::bad_alloc(); }
      return nullptr;
    }
    auto* ptr = std::aligned_alloc(4096, bytes);
    if (ptr != nullptr) { fake->memory_blocks.emplace(ptr, bytes); }
    return ptr;
  };
  hooks.deallocate = [fake](int, void* ptr) {
    ++fake->frees;
    if (fake->fail_free) { throw std::runtime_error("injected free failure"); }
    auto const found = fake->memory_blocks.find(ptr);
    if (found == fake->memory_blocks.end()) { return false; }
    auto const start = reinterpret_cast<std::uintptr_t>(ptr);
    for (auto const& [base, bytes] : fake->active_registrations) {
      static_cast<void>(bytes);
      auto const address = reinterpret_cast<std::uintptr_t>(base);
      if (address >= start && address - start < found->second) { return false; }
    }
    fake->memory_blocks.erase(found);
    std::free(ptr);
    return true;
  };
  hooks.register_buffer = [fake](int, void* ptr, std::size_t bytes) {
    if (fake->registrations++ == fake->fail_registration || bytes > fake->max_registration_bytes) {
      if (fake->throw_register) { throw std::runtime_error("injected register failure"); }
      return false;
    }
    fake->active_registrations.emplace(ptr, bytes);
    fake->registration_log.emplace_back(ptr, bytes);
    return true;
  };
  hooks.deregister_buffer = [fake](int, void* ptr) {
    ++fake->deregistrations;
    if (fake->fail_deregister) { throw std::runtime_error("injected deregister failure"); }
    return fake->active_registrations.erase(ptr) == 1;
  };
  hooks.memory_info      = [fake](int) { return fake->memory; };
  hooks.free_bar1        = [fake](int) { return fake->bar1; };
  hooks.pinning_headroom = [fake](int) { return fake->pin; };
  hooks.try_reserve      = [fake](int, std::size_t bytes) {
    ++fake->reserve_calls;
    auto const granted = std::min(bytes, fake->grant_limit);
    fake->charged += granted;
    return granted;
  };
  hooks.release_reservation = [fake](int, std::size_t bytes) {
    if (fake->fail_release) { throw std::runtime_error("injected reservation release failure"); }
    fake->charged -= bytes;
  };
  return hooks;
}

struct arena_fixture {
  std::shared_ptr<fake_resources> fake{std::make_shared<fake_resources>()};
  device_arena arena{0, fake_hooks(fake)};
  ~arena_fixture()
  {
    try {
      arena.release_for_maintenance();
    } catch (...) {
    }
  }
};

std::vector<staging_lease> lease_all(device_arena& arena,
                                     arena_mount_id mount,
                                     std::size_t bytes,
                                     bool registered = true)
{
  std::vector<staging_lease> result;
  while (auto lease = arena.try_lease(mount, bytes, registered)) {
    result.push_back(std::move(*lease));
  }
  return result;
}
}  // namespace

TEST_CASE("device arena legacy mode has no resource side effects", "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(async_engine_config{}).allocated_bytes == 0);
  REQUIRE(f.fake->allocations == 0);
  REQUIRE(f.fake->reserve_calls == 0);
  REQUIRE(f.arena.gauges().slabs == 0);
}

TEST_CASE("device arena registry is persistent and per device", "[cufile][arena]")
{
  auto& first = device_arena::for_device(0);
  REQUIRE(&first == &device_arena::for_device(0));
  REQUIRE(&first != &device_arena::for_device(1));
  REQUIRE_THROWS_AS(device_arena::for_device(-1), std::invalid_argument);
}

TEST_CASE("device arena validates hooks and control options", "[cufile][arena]")
{
  REQUIRE_THROWS_AS(device_arena(0, device_arena_hooks{}), std::invalid_argument);
  REQUIRE_THROWS_AS(device_arena(-1), std::invalid_argument);
  arena_fixture f;
  device_arena_options options;
  options.bar1_fraction = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(f.arena.warm_up(arena_config(), options), std::invalid_argument);
  options.bar1_fraction = 1.01;
  REQUIRE_THROWS_AS(f.arena.warm_up(arena_config(), options), std::invalid_argument);
  options.bar1_fraction = 0.5;
  options.initial_class = static_cast<slot_class>(255);
  REQUIRE_THROWS_AS(f.arena.warm_up(arena_config(), options), std::invalid_argument);
  REQUIRE(f.fake->allocations == 0);
  REQUIRE_THROWS_AS(f.arena.set_quota(1, 1, 0, static_cast<slot_class>(255)),
                    std::invalid_argument);
}

TEST_CASE("device arena shared registration requires explicit opt-in before resource work",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.registration = GENERATE(registration_granularity::arena, registration_granularity::slab);
  REQUIRE_THROWS_AS(f.arena.warm_up(arena_config(), options), std::invalid_argument);
  REQUIRE(f.fake->allocations == 0);
  REQUIRE(f.fake->registrations == 0);
  REQUIRE(f.fake->reserve_calls == 0);
  options.allow_shared_registration = true;  // Fake resources qualify geometry, not real GDS.
  REQUIRE(f.arena.warm_up(arena_config(), options).registered_bytes == slab);
}

TEST_CASE("device arena default hooks provide optional BAR1 and pin-budget telemetry",
          "[cufile][arena]")
{
  auto const hooks = device_arena_hooks::defaults();
  REQUIRE(static_cast<bool>(hooks.free_bar1));
  REQUIRE(static_cast<bool>(hooks.pinning_headroom));
}

TEST_CASE("device arena auto sizing applies AM5 and floors complete slabs", "[cufile][arena]")
{
  arena_fixture f;
  auto const info     = GENERATE(arena_memory_info{8UL << 30, 1UL << 30},
                             arena_memory_info{64UL << 30, 2UL << 30},
                             arena_memory_info{8UL << 30, 37 * mib},
                             arena_memory_info{8UL << 30, slab - 1});
  f.fake->memory      = info;
  auto const report   = f.arena.warm_up(arena_config(0));
  auto const expected = std::min({1UL << 30, info.total_bytes / 20, info.reservable_bytes});
  REQUIRE(report.allocated_bytes == expected - expected % slab);
  REQUIRE(f.fake->charged == report.allocated_bytes);
  REQUIRE(f.arena.gauges().reserved_bytes == report.allocated_bytes);
}

TEST_CASE("device arena explicit targets and partial reservation are bounded", "[cufile][arena]")
{
  arena_fixture f;
  f.fake->grant_limit = 2 * slab + 4096;
  auto const report   = f.arena.warm_up(arena_config(3 * slab + 17));
  REQUIRE(report.allocation_target == 3 * slab);
  REQUIRE(report.allocated_bytes == 2 * slab);
  REQUIRE(f.fake->charged == 2 * slab);
  REQUIRE(f.arena.gauges().total_bytes == 2 * slab);
}

TEST_CASE("device arena missing or throwing telemetry is conservative", "[cufile][arena]")
{
  auto fake         = std::make_shared<fake_resources>();
  auto hooks        = fake_hooks(fake);
  hooks.memory_info = [](int) -> std::optional<arena_memory_info> {
    throw std::runtime_error("no telemetry");
  };
  hooks.free_bar1 = [](int) -> std::optional<std::size_t> { throw std::runtime_error("no BAR1"); };
  device_arena arena(0, std::move(hooks));
  REQUIRE(arena.warm_up(arena_config(0)).allocated_bytes == 0);
  REQUIRE(fake->allocations == 0);
  arena.release_for_maintenance();
  auto const report = arena.warm_up(arena_config());
  REQUIRE(report.allocated_bytes == slab);
  REQUIRE_FALSE(report.bar1_known);
  arena.release_for_maintenance();
}

TEST_CASE("device arena BAR1 and pinning caps bound registration not allocation", "[cufile][arena]")
{
  arena_fixture f;
  f.fake->bar1 = GENERATE(32 * mib, 64 * mib, 256 * mib);
  f.fake->pin  = 80 * mib;
  device_arena_options options;
  options.registration              = registration_granularity::slab;
  options.allow_shared_registration = true;
  auto const report                 = f.arena.warm_up(arena_config(128 * mib), options);
  auto const free                   = *f.fake->bar1 > 64 * mib ? *f.fake->bar1 - 64 * mib : 0;
  auto const target                 = std::min({128 * mib, free / 2, 80 * mib});
  REQUIRE(report.bar1_known);
  REQUIRE(report.registration_target == target);
  REQUIRE(report.allocated_bytes == 128 * mib);
  REQUIRE(report.registered_bytes == target - target % slab);
  REQUIRE(f.arena.gauges().unregistered_bytes == 128 * mib - report.registered_bytes);
}

TEST_CASE("device arena unknown BAR1 never registers beyond the conservative first step",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.registration              = GENERATE(registration_granularity::slot,
                                  registration_granularity::slab,
                                  registration_granularity::arena);
  options.allow_shared_registration = true;
  options.registration_first_step   = GENERATE(slab, 4 * slab);
  auto const report                 = f.arena.warm_up(arena_config(8 * slab), options);
  REQUIRE(report.first_registration_step == options.registration_first_step);
  REQUIRE(report.registration_target == options.registration_first_step);
  REQUIRE(report.registered_bytes == options.registration_first_step);
  REQUIRE(report.allocated_bytes == 8 * slab);
  REQUIRE(report.registration_failures == 0);
  REQUIRE(f.fake->deregistrations == 0);
  REQUIRE(f.arena.gauges().unregistered_bytes == 8 * slab - report.registered_bytes);
}

TEST_CASE("device arena bulk failure falls back to slab groups with correct bases",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.registration              = registration_granularity::arena;
  options.allow_shared_registration = true;
  options.initial_class             = slot_class::mib8;
  f.fake->bar1 = GENERATE(std::optional<std::size_t>{}, std::optional<std::size_t>{1UL << 30});
  f.fake->max_registration_bytes = slab;
  f.fake->throw_register         = GENERATE(false, true);
  auto const report              = f.arena.warm_up(arena_config(4 * slab), options);
  REQUIRE(report.registered_bytes == 4 * slab);
  REQUIRE(report.registration_failures == 1);
  REQUIRE(f.fake->registrations == 5);
  REQUIRE(f.fake->registration_log.size() == 4);
  REQUIRE(f.arena.set_quota(1, 4, 0, slot_class::mib8).pending_slabs == 0);
  auto leases = lease_all(f.arena, 1, 8 * mib);
  REQUIRE(leases.size() == 8);
  for (std::size_t i = 0; i < leases.size(); ++i) {
    REQUIRE(f.fake->registration_log[i / 2].second == slab);
    REQUIRE(leases[i].base_ptr_to_pass() == f.fake->registration_log[i / 2].first);
    REQUIRE(leases[i].buffer_offset_to_pass() == static_cast<off_t>(i % 2 * 8 * mib));
  }
}

TEST_CASE("device arena retains fallback groups when a later slab registration fails",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.registration              = registration_granularity::arena;
  options.allow_shared_registration = true;
  f.fake->bar1                      = 1UL << 30;
  f.fake->max_registration_bytes    = slab;
  f.fake->fail_registration         = 3;
  auto const report                 = f.arena.warm_up(arena_config(4 * slab), options);
  REQUIRE(report.registered_bytes == 2 * slab);
  REQUIRE(report.registration_failures == 2);
  REQUIRE(f.fake->registrations == 4);
  REQUIRE(f.fake->deregistrations == 0);
  REQUIRE(f.arena.set_quota(1, 2, 2, slot_class::mib1).pending_slabs == 0);
  REQUIRE(lease_all(f.arena, 1, mib).size() == 32);
  REQUIRE(lease_all(f.arena, 1, mib, false).size() == 32);
}

TEST_CASE("device arena allocation failure preserves slabs and returns unused charge",
          "[cufile][arena]")
{
  arena_fixture f;
  f.fake->fail_allocation = GENERATE(0U, 2U);
  f.fake->throw_allocate  = GENERATE(false, true);
  auto const report       = f.arena.warm_up(arena_config(4 * slab));
  REQUIRE(report.allocated_bytes == f.fake->fail_allocation * slab);
  REQUIRE(report.allocation_failures == 1);
  REQUIRE(f.fake->charged == report.allocated_bytes);
  REQUIRE(f.fake->frees == 0);
}

TEST_CASE("device arena retains successful slot registrations after later failure",
          "[cufile][arena]")
{
  arena_fixture f;
  f.fake->fail_registration = 3;
  f.fake->throw_register    = GENERATE(false, true);
  auto const report         = f.arena.warm_up(arena_config(2 * slab));
  REQUIRE(report.registered_bytes == 3 * mib);
  REQUIRE(report.registration_failures == 1);
  REQUIRE(f.fake->registrations == 4);
  REQUIRE(f.fake->active_registrations.size() == 3);
  REQUIRE(f.arena.gauges().partially_registered_slabs == 1);
  REQUIRE(f.arena.set_quota(1, 1, 1, slot_class::mib1).pending_slabs == 0);
  CHECK(f.arena.quota(1).registered_bytes == 3 * mib);
  auto registered   = lease_all(f.arena, 1, mib);
  auto unregistered = lease_all(f.arena, 1, mib, false);
  REQUIRE(registered.size() == 3);
  REQUIRE(unregistered.size() == 29);
  for (auto const& lease : unregistered) {
    REQUIRE_FALSE(lease.registered());
    REQUIRE(lease.buffer_offset_to_pass() == 0);
    REQUIRE(lease.base_ptr_to_pass() == lease.data());
  }
}

TEST_CASE("device arena total registration failure exposes only explicit unregistered tier",
          "[cufile][arena]")
{
  arena_fixture f;
  f.fake->fail_registration = 0;
  REQUIRE(f.arena.warm_up(arena_config()).registered_bytes == 0);
  REQUIRE(f.arena.set_quota(9, 0, 1, slot_class::mib1).pending_slabs == 0);
  REQUIRE_FALSE(f.arena.try_lease(9, mib, true));
  REQUIRE_FALSE(f.arena.try_lease(123, mib, false));
  auto lease = f.arena.try_lease(9, mib, false);
  REQUIRE(lease);
  REQUIRE_FALSE(lease->registered());
}

TEST_CASE("device arena registration granularity supplies the correct base and offset",
          "[cufile][arena]")
{
  arena_fixture f;
  auto const mode = GENERATE(registration_granularity::arena,
                             registration_granularity::slab,
                             registration_granularity::slot);
  device_arena_options options;
  options.registration              = mode;
  options.allow_shared_registration = true;
  options.initial_class             = slot_class::mib8;
  REQUIRE(f.arena.warm_up(arena_config(2 * slab), options).registered_bytes == 2 * slab);
  REQUIRE(f.arena.set_quota(1, 2, 0, slot_class::mib8).pending_slabs == 0);
  auto leases = lease_all(f.arena, 1, 8 * mib);
  REQUIRE(leases.size() == 4);
  REQUIRE(f.fake->registrations == (mode == registration_granularity::arena  ? 1
                                    : mode == registration_granularity::slab ? 2
                                                                             : 4));
  for (std::size_t i = 0; i < leases.size(); ++i) {
    auto const& lease = leases[i];
    auto const offset = mode == registration_granularity::arena  ? i * 8 * mib
                        : mode == registration_granularity::slab ? (i % 2) * 8 * mib
                                                                 : 0;
    REQUIRE(lease.buffer_offset_to_pass() == static_cast<off_t>(offset));
    REQUIRE(static_cast<std::byte*>(lease.base_ptr_to_pass()) + offset == lease.data());
    if (mode == registration_granularity::slot) {
      REQUIRE(f.fake->active_registrations.at(lease.base_ptr_to_pass()) == 8 * mib);
    }
  }
  REQUIRE_FALSE(f.arena.try_lease(1, 0, true));
  REQUIRE_FALSE(f.arena.try_lease(1, 8 * mib + 1, true));
}

TEST_CASE("device arena all size classes have disjoint bounded slots and recycle",
          "[cufile][arena]")
{
  arena_fixture f;
  auto const cls = GENERATE(slot_class::mib1, slot_class::mib2, slot_class::mib4, slot_class::mib8);
  device_arena_options options;
  options.initial_class = cls;
  REQUIRE(f.arena.warm_up(arena_config(), options).allocated_bytes == slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, cls).pending_slabs == 0);
  auto leases = lease_all(f.arena, 1, class_bytes(cls) - 1);
  REQUIRE(leases.size() == slab / class_bytes(cls));
  std::unordered_set<void*> pointers;
  for (auto const& lease : leases) {
    REQUIRE(lease.capacity() == class_bytes(cls));
    REQUIRE(pointers.insert(lease.data()).second);
  }
  auto* first = leases.front().data();
  leases.front().reset();
  auto recycled = f.arena.try_lease(1, class_bytes(cls), true);
  REQUIRE(recycled);
  REQUIRE(recycled->data() == first);
}

TEST_CASE("device arena profiles cap op size to the default slot geometry", "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.available_class_mask() == 0);
  REQUIRE(f.arena.warm_up(arena_config()).registered_bytes == slab);
  REQUIRE(f.arena.available_classes() == std::vector<slot_class>{slot_class::mib1});
  REQUIRE(f.arena.available_class_mask() == 1);
  REQUIRE(f.arena.available_classes(false).empty());
  auto const effective = fit({4, 4 * mib}, slab, 160, f.arena.available_class_mask());
  REQUIRE(effective.qd == 4);
  REQUIRE(effective.op_bytes == mib);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  for (unsigned i = 0; i != 10; ++i) {
    auto lease = f.arena.try_lease(1, effective.op_bytes, true);
    REQUIRE(lease);
    REQUIRE(lease->capacity() == effective.op_bytes);
  }
  f.arena.release_for_maintenance();
  REQUIRE(f.arena.available_classes().empty());
}

TEST_CASE("device arena class capabilities reflect prewarmed gaps and registration tiers",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.slab_classes      = {slot_class::mib1, slot_class::mib8};
  f.fake->fail_registration = 18;  // Two registered slabs, then two unregistered slabs.
  REQUIRE(f.arena.warm_up(arena_config(4 * slab), options).registered_bytes == 2 * slab);
  REQUIRE(f.arena.available_classes() ==
          std::vector<slot_class>{slot_class::mib1, slot_class::mib8});
  REQUIRE(f.arena.available_class_mask() == 0x09);
  REQUIRE(f.arena.available_class_mask(false) == 0x0f);  // Unregistered slabs can reformat.
  auto const effective = fit({1, 4 * mib}, slab, 160, f.arena.available_class_mask());
  REQUIRE(effective.op_bytes == mib);  // Do not select the absent 2 or 4 MiB class.
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib8).pending_slabs == 0);
  REQUIRE(f.arena.try_lease(1, 8 * mib, true));
}

TEST_CASE("device arena probe leases require no mount quota and retain maintenance safety",
          "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE_FALSE(f.arena.try_lease_probe(mib));
  REQUIRE(f.arena.warm_up(arena_config()).registered_bytes == slab);
  REQUIRE_FALSE(f.arena.try_lease_probe(0));
  REQUIRE_FALSE(f.arena.try_lease_probe(8 * mib + 1));
  REQUIRE_FALSE(f.arena.try_lease(99, mib, true));
  auto probe = f.arena.try_lease_probe(mib);
  REQUIRE(probe);
  REQUIRE(probe->registered());
  REQUIRE(probe->base_ptr_to_pass() == probe->data());
  REQUIRE(probe->buffer_offset_to_pass() == 0);
  REQUIRE(f.arena.gauges().live_leases == 1);
  REQUIRE(f.arena.gauges().live_probe_leases == 1);
  REQUIRE_THROWS_AS(f.arena.release_for_maintenance(), std::logic_error);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 1);
  REQUIRE_FALSE(f.arena.try_lease(1, mib, true));
  staging_lease moved = std::move(*probe);
  probe.reset();
  REQUIRE(f.arena.gauges().live_probe_leases == 1);
  moved.reset();
  REQUIRE(f.arena.gauges().live_probe_leases == 0);
  REQUIRE(f.arena.quota(1).pending_slabs == 0);
  REQUIRE(f.arena.try_lease(1, mib, true));
}

TEST_CASE("device arena probe borrows idle assigned capacity without sharing live slabs",
          "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(arena_config()).registered_bytes == slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto active = f.arena.try_lease(1, mib, true);
  REQUIRE(active);
  REQUIRE_FALSE(f.arena.try_lease_probe(mib));
  active.reset();
  auto probe = f.arena.try_lease_probe(mib);
  REQUIRE(probe);
  REQUIRE_FALSE(f.arena.try_lease(1, mib, true));
  REQUIRE(f.arena.set_quota(1, 0, 0, slot_class::mib1).draining_slabs == 1);
  REQUIRE(f.arena.gauges().live_probe_leases == 1);
  probe.reset();
  REQUIRE(f.arena.quota(1).pending_slabs == 0);
  REQUIRE(f.arena.gauges().draining_slabs == 0);
  REQUIRE(f.arena.try_lease_probe(mib));
}

TEST_CASE("device arena probes prefer unowned slabs and explicitly select registration tier",
          "[cufile][arena]")
{
  arena_fixture f;
  f.fake->fail_registration = 19;  // Three registrations in the second slab.
  REQUIRE(f.arena.warm_up(arena_config(2 * slab)).registered_bytes == 19 * mib);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto probe = f.arena.try_lease_probe(mib);
  REQUIRE(probe);
  auto const unowned_base = f.fake->registration_log[16].first;
  REQUIRE(probe->data() == unowned_base);
  REQUIRE(f.arena.try_lease(1, mib, true));  // Owned slab remains available.
  probe.reset();
  auto unregistered = f.arena.try_lease_probe(mib, false);
  REQUIRE(unregistered);
  REQUIRE_FALSE(unregistered->registered());
  REQUIRE(unregistered->data() == static_cast<std::byte*>(unowned_base) + 3 * mib);
  REQUIRE(unregistered->base_ptr_to_pass() == unregistered->data());
  REQUIRE(unregistered->buffer_offset_to_pass() == 0);
  REQUIRE(f.arena.available_class_mask(false) == 1);  // Partially registered geometry is fixed.
}

TEST_CASE("device arena lease moves and facade destruction retain valid metadata",
          "[cufile][arena]")
{
  auto fake = std::make_shared<fake_resources>();
  staging_lease survivor;
  {
    device_arena arena(0, fake_hooks(fake));
    REQUIRE(arena.warm_up(arena_config()).allocated_bytes == slab);
    REQUIRE(arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
    auto first  = arena.try_lease(1, mib, true);
    auto second = arena.try_lease(1, mib, true);
    REQUIRE(first);
    REQUIRE(second);
    *first = std::move(*second);
    REQUIRE_FALSE(*second);
    REQUIRE(arena.gauges().live_leases == 1);
    survivor = std::move(*first);
    survivor = std::move(survivor);  // Self-move must not retire the lease.
    REQUIRE(survivor);
  }
  REQUIRE(fake->frees == 0);
  REQUIRE(fake->deregistrations == 0);
  survivor.reset();
  survivor.reset();
  REQUIRE(fake->frees == 0);
  REQUIRE(fake->deregistrations == 0);
}

TEST_CASE("device arena production bootstrap splits mixed classes and caps assigned capacity",
          "[cufile][arena]")
{
  arena_fixture f;
  auto config               = arena_config(4 * slab);
  config.arena_slot_classes = {4, 8};
  REQUIRE(f.arena.bootstrap(config).registered_bytes == 4 * slab);
  CHECK(f.arena.available_class_mask() == 0x0c);
  auto const assigned = f.arena.set_quota(1, 3, 0, slot_class::mib8);
  CHECK(assigned.registered_slabs == 2);
  CHECK(assigned.pending_slabs == 1);
  CHECK(assigned.registered_bytes == 2 * slab);
  auto const effective = fit({4, 8 * mib}, assigned.registered_bytes, 160, 0x08);
  CHECK(effective.qd == 3);
  CHECK(effective.lanes * 8 * mib <= assigned.registered_bytes);
  config.arena_slot_classes = {1};
  // First warm-up wins (with a warning), until explicit maintenance.
  REQUIRE(f.arena.bootstrap(config).registered_bytes == 4 * slab);
  CHECK(f.arena.available_class_mask() == 0x0c);
}

TEST_CASE("device arena repeated warm-up and lease cycles do no resource work", "[cufile][arena]")
{
  arena_fixture f;
  auto const initial = f.arena.warm_up(arena_config());
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto const registrations = f.fake->registrations;
  for (unsigned i = 0; i < 100; ++i) {
    REQUIRE(f.arena.warm_up(arena_config(8 * slab)).allocated_bytes == initial.allocated_bytes);
    auto lease = f.arena.try_lease(1, mib, true);
    REQUIRE(lease);
  }
  REQUIRE(f.fake->allocations == 1);
  REQUIRE(f.fake->registrations == registrations);
  REQUIRE(f.fake->reserve_calls == 1);
  REQUIRE(f.fake->frees == 0);
  REQUIRE(f.fake->deregistrations == 0);
}

TEST_CASE("device arena quota reclassification drains at most one slab per mount",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.registration = GENERATE(registration_granularity::arena, registration_granularity::slab);
  options.allow_shared_registration = true;
  REQUIRE(f.arena.warm_up(arena_config(3 * slab), options).allocated_bytes == 3 * slab);
  REQUIRE(f.arena.set_quota(1, 3, 0, slot_class::mib1).pending_slabs == 0);
  auto leases = lease_all(f.arena, 1, mib);
  REQUIRE(leases.size() == 48);
  REQUIRE(f.arena.set_quota(1, 3, 0, slot_class::mib8).draining_slabs == 1);
  REQUIRE(f.arena.quota(1).pending_slabs == 3);
  CHECK(f.arena.quota(1).registered_bytes == 0);
  for (auto& lease : leases) {
    REQUIRE(lease.capacity() == mib);  // Never revoked or reformatted while live.
    lease.reset();
    REQUIRE(f.arena.quota(1).draining_slabs <= 1);
  }
  REQUIRE(f.arena.quota(1).pending_slabs == 0);
  REQUIRE(f.arena.gauges().reclassifications == 3);
  REQUIRE(f.fake->deregistrations == 0);
  REQUIRE(f.fake->frees == 0);
  auto larger = lease_all(f.arena, 1, 8 * mib);
  REQUIRE(larger.size() == 6);
}

TEST_CASE("device arena slot reclassification uses prewarmed replacement or reports pending",
          "[cufile][arena]")
{
  arena_fixture f;
  device_arena_options options;
  options.slab_classes = {slot_class::mib1, slot_class::mib8};
  REQUIRE(f.arena.warm_up(arena_config(2 * slab), options).allocated_bytes == 2 * slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto small = f.arena.try_lease(1, mib, true);
  REQUIRE(small);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib8).draining_slabs == 1);
  small.reset();
  REQUIRE(f.arena.quota(1).pending_slabs == 0);
  auto large = f.arena.try_lease(1, 8 * mib, true);
  REQUIRE(large);
  REQUIRE(large->buffer_offset_to_pass() == 0);
  large.reset();
  auto const pending = f.arena.set_quota(1, 1, 0, slot_class::mib4);
  REQUIRE(pending.pending_slabs == 1);
  CHECK(pending.registered_bytes == 0);
  REQUIRE(pending.draining_slabs == 0);
  auto still_usable = f.arena.try_lease(1, 8 * mib, true);
  REQUIRE(still_usable);
  REQUIRE(f.fake->deregistrations == 0);
  REQUIRE(f.fake->registrations == 18);
}

TEST_CASE("device arena idle capacity is borrowed without revoking leases", "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(arena_config(2 * slab)).allocated_bytes == 2 * slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  REQUIRE(f.arena.set_quota(2, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto borrowed = lease_all(f.arena, 1, mib);
  REQUIRE(borrowed.size() == 32);
  REQUIRE_FALSE(f.arena.try_lease(2, mib, true));
  REQUIRE(f.arena.set_quota(1, 0, 0, slot_class::mib1).draining_slabs == 1);
  REQUIRE(f.arena.quota(1).pending_slabs != 0);
  REQUIRE_FALSE(f.arena.try_lease(1, mib, true));
  borrowed.clear();
  REQUIRE(f.arena.quota(1).pending_slabs == 0);
  REQUIRE(f.arena.try_lease(2, mib, true));
}

TEST_CASE("device arena maintenance refuses live leases then releases exactly once",
          "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(arena_config()).allocated_bytes == slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto lease = f.arena.try_lease(1, mib, true);
  REQUIRE(lease);
  REQUIRE_THROWS_AS(f.arena.release_for_maintenance(), std::logic_error);
  REQUIRE(f.fake->frees == 0);
  REQUIRE(f.fake->deregistrations == 0);
  lease.reset();
  f.arena.release_for_maintenance();
  REQUIRE(f.fake->deregistrations == 16);
  REQUIRE(f.fake->frees == 1);
  REQUIRE(f.fake->charged == 0);
  REQUIRE(f.arena.gauges().total_bytes == 0);
  REQUIRE(f.arena.warm_up(arena_config()).registered_bytes == slab);
}

TEST_CASE("device arena maintenance failures retain resources and close admission for retry",
          "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(arena_config()).allocated_bytes == slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  auto const failure      = GENERATE(0, 1, 2);
  f.fake->fail_deregister = failure == 0;
  f.fake->fail_free       = failure == 1;
  f.fake->fail_release    = failure == 2;
  REQUIRE_THROWS_AS(f.arena.release_for_maintenance(), std::runtime_error);
  REQUIRE_FALSE(f.arena.try_lease(1, mib, true));
  REQUIRE_THROWS_AS(f.arena.warm_up(arena_config()), std::logic_error);
  if (failure == 0) {
    REQUIRE(f.fake->frees == 0);
    REQUIRE(f.fake->active_registrations.size() == 16);
  }
  if (failure != 2) {
    REQUIRE(f.fake->memory_blocks.size() == 1);
  } else {
    REQUIRE(f.arena.gauges().slabs == 0);
    REQUIRE(f.arena.gauges().registered_bytes == 0);
  }
  f.fake->fail_deregister = false;
  f.fake->fail_free       = false;
  f.fake->fail_release    = false;
  f.arena.release_for_maintenance();
  REQUIRE(f.fake->memory_blocks.empty());
  REQUIRE(f.fake->active_registrations.empty());
  REQUIRE(f.fake->charged == 0);
}

TEST_CASE("device arena preserves reservation charge when unused-grant rollback throws",
          "[cufile][arena]")
{
  arena_fixture f;
  f.fake->fail_allocation = 1;
  f.fake->fail_release    = true;
  REQUIRE(f.arena.warm_up(arena_config(2 * slab)).allocated_bytes == slab);
  REQUIRE(f.arena.gauges().reserved_bytes == 2 * slab);
  REQUIRE(f.fake->charged == 2 * slab);
  f.fake->fail_release = false;
  f.arena.release_for_maintenance();
  REQUIRE(f.fake->charged == 0);
}

TEST_CASE("device arena leasing never waits behind warm-up registration", "[cufile][arena]")
{
  auto fake         = std::make_shared<fake_resources>();
  auto hooks        = fake_hooks(fake);
  auto registration = hooks.register_buffer;
  std::promise<void> entered, release;
  auto wait             = release.get_future().share();
  hooks.register_buffer = [&](int device, void* ptr, std::size_t bytes) {
    if (fake->registrations == 0) {
      entered.set_value();
      wait.wait();
    }
    return registration(device, ptr, bytes);
  };
  device_arena arena(0, std::move(hooks));
  auto warm = std::async(std::launch::async, [&] { return arena.warm_up(arena_config()); });
  entered.get_future().wait();
  auto const start   = std::chrono::steady_clock::now();
  auto lease         = arena.try_lease(1, mib, true);
  auto probe         = arena.try_lease_probe(mib);
  auto const elapsed = std::chrono::steady_clock::now() - start;
  release.set_value();  // Always unblock before any assertion can throw.
  REQUIRE_FALSE(lease);
  REQUIRE_FALSE(probe);
  REQUIRE(elapsed < std::chrono::milliseconds(100));
  REQUIRE(warm.get().allocated_bytes == slab);
  arena.release_for_maintenance();
}

TEST_CASE("device arena concurrent leases never overlap or double retire", "[cufile][arena]")
{
  arena_fixture f;
  REQUIRE(f.arena.warm_up(arena_config()).allocated_bytes == slab);
  REQUIRE(f.arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);
  std::mutex mutex;
  std::unordered_set<void*> live;
  std::atomic<bool> failed{};
  std::vector<std::thread> workers;
  for (unsigned thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      for (unsigned i = 0; i < 100; ++i) {
        std::optional<staging_lease> lease;
        while (!(lease = f.arena.try_lease(1, mib, true))) {
          std::this_thread::yield();
        }
        {
          std::lock_guard lock(mutex);
          if (!live.insert(lease->data()).second) { failed = true; }
        }
        std::this_thread::yield();
        {
          std::lock_guard lock(mutex);
          if (live.erase(lease->data()) != 1) { failed = true; }
          lease.reset();
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  REQUIRE_FALSE(failed);
  REQUIRE(live.empty());
  REQUIRE(f.arena.gauges().live_leases == 0);
  REQUIRE(f.fake->registrations == 16);
}

namespace {
// Declaration order in the integration case keeps lanes/args alive until all streams drain,
// and keeps file handles and arena registrations alive until after the last accessor.
struct integration_file {
  std::string path;
  int fd{-1};
  CUfileHandle_t handle{};
  cufile_api const* api{};
  bool quarantined{};
  explicit integration_file(cufile_api const* table) : api(table)
  {
    std::string name = (std::filesystem::temp_directory_path() / "cucascade_arena_XXXXXX").string();
    std::vector<char> writable(name.begin(), name.end());
    writable.push_back('\0');
    fd = ::mkstemp(writable.data());
    if (fd < 0) { throw std::runtime_error("mkstemp failed"); }
    path = writable.data();
  }
  ~integration_file()
  {
    if (quarantined) { return; }  // A live DMA may still reference both handle and fd.
    if (handle != nullptr) { api->handle_deregister(handle); }
    if (fd >= 0) { static_cast<void>(::close(fd)); }
    if (!path.empty()) { static_cast<void>(::unlink(path.c_str())); }
  }
};

struct integration_args {
  std::size_t size{mib};
  off_t file_offset{};
  off_t buffer_offset{};
  ssize_t done{-1};
};

struct integration_lane {
  cudaStream_t stream{};
  cudaEvent_t event{};
  integration_args* args{};
  cufile_api const* api{};
  bool registered{};
  bool& quarantined;
  std::unique_ptr<staging_lease> lease;
  integration_lane(cufile_api const* table, bool& quarantine) : api(table), quarantined(quarantine)
  {
  }
  ~integration_lane()
  {
    if (stream != nullptr) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        // Do not release a slot, args, or registration without proof that DMA is quiescent.
        // The maintenance guard will refuse this intentionally quarantined live lease.
        quarantined = true;
        static_cast<void>(lease.release());
        return;
      }
      if (registered) { static_cast<void>(api->stream_deregister(stream)); }
    }
    if (event != nullptr) { static_cast<void>(cudaEventDestroy(event)); }
    if (stream != nullptr) { static_cast<void>(cudaStreamDestroy(stream)); }
    if (args != nullptr) { static_cast<void>(cudaFreeHost(args)); }
  }
};

struct arena_maintenance_guard {
  device_arena& arena;
  ~arena_maintenance_guard()
  {
    try {
      arena.release_for_maintenance();
    } catch (...) {
    }
  }
};

void compat_qd4(registration_granularity granularity)
{
  auto const* api = cufile_api::get();
  int devices{};
  if (api == nullptr || !api->has_async_read || !cufile_api::open_driver()) {
    SKIP("libcufile async reads unavailable");
  }
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { SKIP("CUDA unavailable"); }
  int device{};
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  device_arena arena(device);
  arena_maintenance_guard maintenance{arena};
  device_arena_options options;
  options.registration              = granularity;
  options.allow_shared_registration = granularity != registration_granularity::slot;
  auto const report                 = arena.warm_up(arena_config(), options);
  REQUIRE(report.registered_bytes == slab);
  REQUIRE(arena.set_quota(1, 1, 0, slot_class::mib1).pending_slabs == 0);

  integration_file file(api);
  std::vector<unsigned char> expected(4 * mib);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto const value = (static_cast<std::uint64_t>(i) * 1315423911ULL) ^
                       (static_cast<std::uint64_t>(i / mib) * 2654435761ULL) ^
                       static_cast<std::uint64_t>(i >> 8);
    expected[i] = static_cast<unsigned char>(value & 255U);
  }
  REQUIRE(::write(file.fd, expected.data(), expected.size()) ==
          static_cast<ssize_t>(expected.size()));
  REQUIRE(::close(file.fd) == 0);
  file.fd = ::open(file.path.c_str(), O_RDONLY | O_DIRECT);
  REQUIRE(file.fd >= 0);
  CUfileDescr_t descriptor{};
  descriptor.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  descriptor.handle.fd = file.fd;
  REQUIRE(api->handle_register(&file.handle, &descriptor).err == CU_FILE_SUCCESS);

  std::array<std::unique_ptr<integration_lane>, 4> lanes;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    lanes[i]   = std::make_unique<integration_lane>(api, file.quarantined);
    auto& lane = *lanes[i];
    auto lease = arena.try_lease(1, mib, true);
    REQUIRE(lease);
    lane.lease = std::make_unique<staging_lease>(std::move(*lease));
    REQUIRE(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaEventCreateWithFlags(&lane.event, cudaEventDisableTiming) == cudaSuccess);
    void* args{};
    REQUIRE(cudaMallocHost(&args, sizeof(integration_args)) == cudaSuccess);
    lane.args                = ::new (args) integration_args{};
    lane.args->file_offset   = static_cast<off_t>(i * mib);
    lane.args->buffer_offset = lane.lease->buffer_offset_to_pass();
    if (api->has_stream_register) {
      REQUIRE(api->stream_register(lane.stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS).err ==
              CU_FILE_SUCCESS);
      lane.registered = true;
    }
  }
  // An untimed, validated read per stream, then repeated disjoint QD4 reads. Arguments stay
  // pinned and untouched until the completion event has fired, matching the P0 harness.
  for (unsigned round = 0; round < 5; ++round) {
    for (std::size_t i = 0; i < lanes.size(); ++i) {
      auto& lane             = *lanes[i];
      lane.args->done        = -1;
      lane.args->file_offset = static_cast<off_t>((i + round) % lanes.size() * mib);
      REQUIRE(cudaMemsetAsync(lane.lease->data(), 0xa5, mib, lane.stream) == cudaSuccess);
      REQUIRE(api
                ->read_async(file.handle,
                             lane.lease->base_ptr_to_pass(),
                             &lane.args->size,
                             &lane.args->file_offset,
                             &lane.args->buffer_offset,
                             &lane.args->done,
                             lane.stream)
                .err == CU_FILE_SUCCESS);
      REQUIRE(cudaEventRecord(lane.event, lane.stream) == cudaSuccess);
      if (round == 0) { REQUIRE(cudaEventSynchronize(lane.event) == cudaSuccess); }
    }
    for (std::size_t i = 0; i < lanes.size(); ++i) {
      auto& lane = *lanes[i];
      INFO("registration=" << static_cast<unsigned>(granularity) << " round=" << round
                           << " lane=" << i << " file_offset=" << lane.args->file_offset
                           << " buffer_offset=" << lane.args->buffer_offset);
      REQUIRE(cudaEventSynchronize(lane.event) == cudaSuccess);
      REQUIRE(lane.args->done == static_cast<ssize_t>(mib));
      std::vector<unsigned char> actual(mib);
      REQUIRE(cudaMemcpy(actual.data(), lane.lease->data(), mib, cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      auto const expected_start =
        expected.begin() + static_cast<std::ptrdiff_t>(lane.args->file_offset);
      auto const mismatch = std::mismatch(actual.begin(), actual.end(), expected_start);
      INFO("first differing byte=" << std::distance(actual.begin(), mismatch.first));
      REQUIRE(mismatch.first == actual.end());
    }
  }
}
}  // namespace

TEST_CASE("device arena real libcufile concurrent QD4 per-slot reads",
          "[cufile][arena][integration]")
{
  compat_qd4(registration_granularity::slot);
}

// Deliberately opt-in: AM-9's shared-registration failure is unresolved, not a qualification
// of arena registration. Keep a runnable reproducer without making compat CI rely on it.
TEST_CASE("device arena real libcufile QD4 shared-registration reproducer", "[.][arena_reproducer]")
{
  compat_qd4(GENERATE(registration_granularity::arena, registration_granularity::slab));
}
