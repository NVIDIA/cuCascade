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

#include <catch2/catch_all.hpp>

#include <array>
#include <limits>
#include <numeric>
#include <vector>

using namespace cucascade::io::cufile;

TEST_CASE("cuFile capacity reproduces design section 4.7", "[io][cufile][capacity]")
{
  SECTION("256 MiB QD32 prefers 8 MiB")
  {
    auto const p = fit({32, 8 * capacity_mib}, 256 * capacity_mib, 160);
    REQUIRE(p.qd == 32);
    REQUIRE(p.op_bytes == 4 * capacity_mib);
    REQUIRE(p.lanes == 40);
    REQUIRE(p.lanes * p.op_bytes == 160 * capacity_mib);
  }
  SECTION("256 MiB QD64 prefers 8 MiB")
  {
    auto const p = fit({64, 8 * capacity_mib}, 256 * capacity_mib, 160);
    REQUIRE(p.qd == 64);
    REQUIRE(p.op_bytes == 2 * capacity_mib);
    REQUIRE(p.lanes == 80);
    REQUIRE(p.lanes * p.op_bytes == 160 * capacity_mib);
  }
  SECTION("two mounts receive 128 MiB each")
  {
    std::array<mount_capacity_demand, 2> const mounts{
      {{{64, 2 * capacity_mib}, 1}, {{64, 2 * capacity_mib}, 1}}};
    auto const grants = rebalance(mounts, 256 * capacity_mib);
    REQUIRE(grants.size() == 2);
    for (auto const& g : grants) {
      REQUIRE(g.slabs == 8);
      REQUIRE(g.lanes == 80);
      REQUIRE(g.effective.qd == 64);
      REQUIRE(g.effective.op_bytes == capacity_mib);
      REQUIRE(g.effective.lanes * g.effective.op_bytes == 80 * capacity_mib);
    }
  }
  SECTION("explicit 8 MiB QD32 and QD64 sweeps need 320 and 640 MiB")
  {
    for (auto q : {32U, 64U}) {
      auto const needed = static_cast<std::size_t>(lanes(q)) * 8 * capacity_mib;
      REQUIRE(needed == (q == 32 ? 320 : 640) * capacity_mib);
      auto const full = fit({q, 8 * capacity_mib}, needed, 160);
      REQUIRE(full.qd == q);
      REQUIRE(full.op_bytes == 8 * capacity_mib);
      REQUIRE(fit({q, 8 * capacity_mib}, needed - 1, 160).op_bytes == 4 * capacity_mib);
    }
  }
}

TEST_CASE("cuFile capacity fits lanes before operation classes", "[io][cufile][capacity]")
{
  REQUIRE(lanes(0) == 0);
  REQUIRE(lanes(1) == 2);
  REQUIRE(lanes(4) == 5);
  REQUIRE(lanes(5) == 7);
  REQUIRE(lanes(32) == 40);
  REQUIRE(lanes(64) == 80);
  auto const limited = fit({64, 8 * capacity_mib}, 16 * capacity_mib, 160);
  REQUIRE(limited.qd == 12);
  REQUIRE(limited.op_bytes == capacity_mib);
  REQUIRE(limited.lanes == 15);
  auto const lane_limited = fit({64, 8 * capacity_mib}, 640 * capacity_mib, 10);
  REQUIRE(lane_limited.qd == 8);
  REQUIRE(lane_limited.op_bytes == 8 * capacity_mib);
  REQUIRE(fit({0, capacity_mib}, 640 * capacity_mib, 160).is_fallback());
  REQUIRE(fit({1, capacity_mib}, capacity_mib, 160).is_fallback());
  REQUIRE(fit({64, capacity_mib}, 640 * capacity_mib, 1).is_fallback());
  REQUIRE(fit({64, capacity_mib}, 0, 160).is_fallback());
  auto const extreme = fit({std::numeric_limits<unsigned>::max(), 8 * capacity_mib},
                           std::numeric_limits<std::size_t>::max(),
                           160);
  REQUIRE(extreme.qd == 128);
  REQUIRE(extreme.lanes == 160);
  REQUIRE(extreme.op_bytes == 8 * capacity_mib);
}

TEST_CASE("cuFile backing slots charge physical bytes and retain larger logical pieces",
          "[io][cufile][capacity]")
{
  CHECK(backing_slot_bytes(capacity_mib, 0) == 0);
  CHECK(backing_slot_bytes(capacity_mib, 0x0c) == 4 * capacity_mib);
  CHECK(backing_slot_bytes(2 * capacity_mib, 0x05) == 4 * capacity_mib);
  CHECK(backing_slot_bytes(8 * capacity_mib, 0x04) == 0);
  CHECK(backing_slot_bytes(8 * capacity_mib, 0x08) == 8 * capacity_mib);
  auto const same_backing = fit({64, 2 * capacity_mib}, 64 * capacity_mib, 160, 0x04);
  CHECK(same_backing.op_bytes == 2 * capacity_mib);
  CHECK(same_backing.qd == 12);
  auto const exact = fit({64, 2 * capacity_mib}, 64 * capacity_mib, 160, 0x05);
  CHECK(exact.op_bytes == capacity_mib);
  CHECK(exact.qd == 51);
  // A mixed {4,8} arena's 8-MiB half owns 32 MiB, not the full 64 MiB.
  auto const assigned = fit({4, 8 * capacity_mib}, 32 * capacity_mib, 160, 0x08);
  CHECK(assigned.qd == 3);
  CHECK(assigned.lanes * 8 * capacity_mib <= 32 * capacity_mib);
}

TEST_CASE("cuFile rebalance demand uses the piece fit actually selects", "[io][cufile][capacity]")
{
  std::array<mount_capacity_demand, 1> mounts{{{{64, 8 * capacity_mib}, 1}}};
  auto const gpfs = rebalance(mounts, 1024 * capacity_mib, 160, 0x04).front();
  CHECK(gpfs.slabs == 20);
  CHECK(gpfs.effective.op_bytes == 4 * capacity_mib);
  CHECK(gpfs.effective.qd == 64);
  mounts[0].profile.preferred_op_bytes = 2 * capacity_mib;
  auto const weka                      = rebalance(mounts, 1024 * capacity_mib, 160, 0x05).front();
  CHECK(weka.slabs == 5);
  CHECK(weka.effective.op_bytes == capacity_mib);
  CHECK(weka.effective.qd == 64);
  CHECK(rebalance(mounts, 1024 * capacity_mib, 160, 0).front().slabs == 0);
}

TEST_CASE("cuFile water filling is weighted capped and deterministic", "[io][cufile][capacity]")
{
  std::array<capacity_demand, 3> demands{{{100, 1}, {100, 2}, {100, 1}}};
  REQUIRE(water_fill(demands, 16) == std::vector<unsigned>{4, 8, 4});
  demands[0].demand = 1;
  REQUIRE(water_fill(demands, 16) == std::vector<unsigned>{1, 10, 5});
  demands[1].demand = 0;
  REQUIRE(water_fill(demands, 16) == std::vector<unsigned>{1, 0, 15});
  REQUIRE(water_fill(demands, 1000) == std::vector<unsigned>{1, 0, 100});
  REQUIRE(water_fill(demands, 0) == std::vector<unsigned>{0, 0, 0});
  demands = {{{100, 1}, {100, 1}, {100, 1}}};
  REQUIRE(water_fill(demands, 5) == std::vector<unsigned>{2, 2, 1});
  demands[0].weight = 0;
  REQUIRE_THROWS_AS(water_fill(demands, 5), std::invalid_argument);
  REQUIRE(water_fill({}, 160).empty());
}

TEST_CASE("cuFile water filling conserves resource and demand bounds", "[io][cufile][capacity]")
{
  for (unsigned first = 0; first < 12; ++first) {
    for (unsigned second = 0; second < 12; ++second) {
      for (unsigned total = 0; total < 32; ++total) {
        std::array<capacity_demand, 3> const d{{{first, 1}, {second, 3}, {5, 2}}};
        auto const g = water_fill(d, total);
        REQUIRE(std::accumulate(g.begin(), g.end(), 0U) == std::min(total, first + second + 5));
        for (std::size_t i = 0; i < g.size(); ++i)
          REQUIRE(g[i] <= d[i].demand);
      }
    }
  }
  auto const max = std::numeric_limits<unsigned>::max();
  std::array<capacity_demand, 2> const d{{{max, max}, {max, max}}};
  auto const g = water_fill(d, max);
  REQUIRE(g[0] == max / 2 + 1);
  REQUIRE(g[1] == max / 2);
}

TEST_CASE("cuFile rebalance computes desired grants and leaves idle capacity borrowable",
          "[io][cufile][capacity]")
{
  std::array<mount_capacity_demand, 3> const mounts{
    {{{32, 4 * capacity_mib}, 1}, {{64, 2 * capacity_mib}, 2}, {{0, capacity_mib}, 1}}};
  auto const grants = rebalance(mounts, 256 * capacity_mib + 17, 100);
  REQUIRE(grants[0].lanes == 33);
  REQUIRE(grants[0].effective.qd == 26);
  REQUIRE(grants[0].effective.op_bytes == 2 * capacity_mib);
  REQUIRE(grants[1].lanes == 67);
  REQUIRE(grants[2].lanes == 0);
  REQUIRE(grants[0].slabs + grants[1].slabs == 16);
  REQUIRE(grants[2].slabs == 0);
  REQUIRE(grants[2].effective.is_fallback());
  REQUIRE(rebalance({}, 256 * capacity_mib).empty());
}

TEST_CASE("cuFile rebalance preserves operation size under the 160 lane cap",
          "[io][cufile][capacity]")
{
  auto const count = GENERATE(1U, 2U, 3U, 5U);
  std::vector<mount_capacity_demand> mounts(count, {{64, 4 * capacity_mib}, 1});
  auto const grants      = rebalance(mounts, 2UL << 30);
  unsigned granted_lanes = 0;
  unsigned used_lanes    = 0;
  for (auto const& grant : grants) {
    granted_lanes += grant.lanes;
    used_lanes += grant.effective.lanes;
    auto const expected_qd =
      count <= 2 ? 64U : (count == 3 ? (grant.lanes == 54 ? 43U : 42U) : 25U);
    REQUIRE(grant.effective.qd == expected_qd);
    REQUIRE(grant.effective.op_bytes == 4 * capacity_mib);
    REQUIRE(grant.effective.lanes <= grant.lanes);
    REQUIRE(grant.effective.lanes * grant.effective.op_bytes <= grant.slabs * capacity_slab_bytes);
  }
  REQUIRE(granted_lanes == (count == 1 ? 80 : 160));
  REQUIRE(used_lanes <= 160);
}

TEST_CASE("cuFile capacity only chooses prewarmed operation classes", "[io][cufile][capacity]")
{
  auto const slot_default = fit({32, 4 * capacity_mib}, 256 * capacity_mib, 160, 0x01);
  REQUIRE(slot_default.qd == 32);
  REQUIRE(slot_default.op_bytes == capacity_mib);
  REQUIRE(fit({32, 4 * capacity_mib}, 256 * capacity_mib, 160, 0).is_fallback());
  auto const larger_backing = fit({32, capacity_mib}, 64 * capacity_mib, 160, 0x04);
  REQUIRE(larger_backing.op_bytes == capacity_mib);
  REQUIRE(larger_backing.qd == 12);
  REQUIRE(larger_backing.lanes * 4 * capacity_mib <= 64 * capacity_mib);
  auto const no_small_slots = fit({32, 8 * capacity_mib}, 16 * capacity_mib, 160, 0x04);
  REQUIRE(no_small_slots.qd == 3);
  REQUIRE(no_small_slots.op_bytes == 4 * capacity_mib);
  std::array<mount_capacity_demand, 1> const mounts{{{{32, 4 * capacity_mib}, 1}}};
  REQUIRE(rebalance(mounts, 256 * capacity_mib, 160, 0x01)[0].effective.op_bytes == capacity_mib);
}
