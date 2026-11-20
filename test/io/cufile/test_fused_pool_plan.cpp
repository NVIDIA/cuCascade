/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using cucascade::io::cufile::config;
using cucascade::io::cufile::describe;
using cucascade::io::cufile::device_group_plan;
using cucascade::io::cufile::fused_lane_staging_bytes;
using cucascade::io::cufile::fused_read_config;
using cucascade::io::cufile::host_batch_config;
using cucascade::io::cufile::lane_completion;
using cucascade::io::cufile::local_reactors_per_device;
using cucascade::io::cufile::plan_cufile_pool;
using cucascade::io::cufile::platform_info;
using cucascade::io::cufile::pool_plan;
using cucascade::io::cufile::staging;

namespace {

constexpr std::size_t KiB = 1UL << 10;
constexpr std::size_t MiB = 1UL << 20;

[[nodiscard]] int device_count()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return 0;
  }
  return count;
}

/// A platform that reports nothing: no GDS, no pinned-memory or DMA limits, so the plan's
/// caps are exactly the configured ones.
[[nodiscard]] platform_info quiet_platform() { return platform_info{}; }

[[nodiscard]] bool any_contains(std::vector<std::string> const& lines, std::string const& text)
{
  return std::any_of(lines.begin(), lines.end(), [&](std::string const& line) {
    return line.find(text) != std::string::npos;
  });
}

/// A plan with fused reads on (staging never, so only the lanes use the cap), one GPU.
[[nodiscard]] config fused_config()
{
  config cfg;
  cfg.staging_mode        = staging::never;
  cfg.devices             = {0};
  cfg.fused_reads.enabled = true;
  return cfg;
}

constexpr std::size_t lane_footprint(std::size_t slot) { return slot + 64 * KiB; }

}  // namespace

TEST_CASE("fused_read_config validates only when enabled", "[io][cufile][fused_plan][placement]")
{
  fused_read_config cfg;
  CHECK_NOTHROW(cfg.validate());
  cfg.lanes_per_reactor = 0;  // ignored while disabled
  cfg.slot_bytes        = 1;
  CHECK_NOTHROW(cfg.validate());

  auto const rejects = [](auto mutate) {
    fused_read_config bad;
    bad.enabled = true;
    mutate(bad);
    CHECK_THROWS_AS(bad.validate(), std::invalid_argument);
  };
  fused_read_config enabled;
  enabled.enabled = true;
  CHECK_NOTHROW(enabled.validate());
  rejects([](fused_read_config& c) { c.reactors_per_device = 0; });
  rejects([](fused_read_config& c) { c.reactors_per_device = 65; });
  rejects([](fused_read_config& c) { c.lanes_per_reactor = 0; });
  rejects([](fused_read_config& c) { c.lanes_per_reactor = 65; });
  rejects([](fused_read_config& c) { c.max_lanes_per_device = 0; });
  rejects([](fused_read_config& c) { c.slot_bytes = 32 * KiB; });
  rejects([](fused_read_config& c) { c.slot_bytes = 8 * MiB + 4 * KiB; });
  rejects([](fused_read_config& c) { c.slot_bytes = 2UL << 30; });
  rejects([](fused_read_config& c) { c.fuse_gap_bytes = c.slot_bytes; });
  rejects([](fused_read_config& c) { c.max_gated_lanes = c.lanes_per_reactor + 1; });
  rejects([](fused_read_config& c) { c.failure_limit = 0; });
  rejects([](fused_read_config& c) { c.completion = static_cast<lane_completion>(7); });

  // Edges that are valid.
  auto edge                 = enabled;
  edge.reactors_per_device  = 64;
  edge.lanes_per_reactor    = 64;
  edge.slot_bytes           = 64 * KiB;
  edge.fuse_gap_bytes       = 64 * KiB - 1;
  edge.max_gated_lanes      = 64;
  edge.completion           = lane_completion::poll;
  CHECK_NOTHROW(edge.validate());

  // Gated lanes: half the lanes by default, at least one, or the configured number.
  CHECK(enabled.gated_lane_limit() == 16);
  auto one              = enabled;
  one.lanes_per_reactor = 1;
  CHECK(one.gated_lane_limit() == 1);
  auto three              = enabled;
  three.lanes_per_reactor = 3;
  CHECK(three.gated_lane_limit() == 1);
  auto explicit_limit            = enabled;
  explicit_limit.max_gated_lanes = 5;
  CHECK(explicit_limit.gated_lane_limit() == 5);
}

TEST_CASE("host_batch_config validates only when enabled", "[io][cufile][fused_plan][placement]")
{
  host_batch_config cfg;
  cfg.depth = 0;  // ignored while disabled
  CHECK_NOTHROW(cfg.validate());

  host_batch_config enabled;
  enabled.enabled = true;
  CHECK_NOTHROW(enabled.validate());
  auto const rejects = [&](auto mutate) {
    auto bad = enabled;
    mutate(bad);
    CHECK_THROWS_AS(bad.validate(), std::invalid_argument);
  };
  rejects([](host_batch_config& c) { c.reactors = 0; });
  rejects([](host_batch_config& c) { c.reactors = 65; });
  rejects([](host_batch_config& c) { c.depth = 0; });
  rejects([](host_batch_config& c) { c.depth = 1025; });
  rejects([](host_batch_config& c) { c.max_entry_bytes = 0; });
  rejects([](host_batch_config& c) { c.max_entry_bytes = 4 * KiB + 1; });
  rejects([](host_batch_config& c) { c.max_entry_bytes = 2UL << 30; });
  rejects([](host_batch_config& c) { c.poll_us = 0; });
  rejects([](host_batch_config& c) { c.poll_us = 1'000'001; });
  rejects([](host_batch_config& c) { c.failure_limit = 0; });
  auto edge            = enabled;
  edge.depth           = 1024;
  edge.max_entry_bytes = 4 * KiB;
  edge.poll_us         = 1'000'000;
  edge.min_bytes       = 0;
  CHECK_NOTHROW(edge.validate());
}

TEST_CASE("pool plan counts lane and batch reactors", "[io][cufile][fused_plan][placement]")
{
  pool_plan plan;
  plan.flat_reactors = 5;
  CHECK(plan.total_reactors() == 5);
  plan.host_batch_reactors = 2;  // a flat plan ignores it
  CHECK(plan.total_reactors() == 5);
  device_group_plan group;
  group.n_reactors    = 8;
  group.lane_reactors = 4;
  plan.groups         = {group, group};
  CHECK(plan.total_reactors() == 2 * (8 + 4) + 2);
}

TEST_CASE("fused_lane_staging_bytes is the registered memory the lanes want",
          "[io][cufile][fused_plan][placement]")
{
  config cfg;
  CHECK(fused_lane_staging_bytes(cfg) == 0);  // disabled
  cfg.fused_reads.enabled = true;
  CHECK(fused_lane_staging_bytes(cfg) == 128 * lane_footprint(8 * MiB));
  CHECK(fused_lane_staging_bytes(cfg) == 1032 * MiB);
  cfg.fused_reads.reactors_per_device = 2;
  cfg.fused_reads.lanes_per_reactor   = 4;
  cfg.fused_reads.slot_bytes          = 1 * MiB;
  CHECK(fused_lane_staging_bytes(cfg) == 8 * lane_footprint(1 * MiB));
  cfg.fused_reads.max_lanes_per_device = 5;
  CHECK(fused_lane_staging_bytes(cfg) == 5 * lane_footprint(1 * MiB));
}

TEST_CASE("pool plan budgets fused lanes on top of the staging cap",
          "[io][cufile][fused_plan][placement][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");

  SECTION("defaults: every lane, no warning")
  {
    auto const cfg  = fused_config();
    auto const plan = plan_cufile_pool(cfg, nullptr, quiet_platform());
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    auto const& group = plan.groups.front();
    CHECK(group.lane_reactors == 4);
    CHECK(group.lanes_per_reactor == 32);
    CHECK(group.lane_slots == 128);
    CHECK(group.staging_slots == 0);  // staging never
    // The staging pool's cap counts the lanes on top of the configured cap.
    CHECK(group.staging_cap_bytes == cfg.max_staging_bytes_per_device + 1032 * MiB);
    // The lanes carry the device reads: the synchronous reactors fall to the local count.
    CHECK(group.n_reactors == local_reactors_per_device);
    CHECK(plan.total_reactors() == local_reactors_per_device + 4);
    CHECK(plan.lane_slot_bytes == 8 * MiB);
    CHECK(plan.host_batch_reactors == 0);
    CHECK(plan.warnings.empty());
  }
  SECTION("no cap and no hardware limit: no cap at all")
  {
    auto cfg                         = fused_config();
    cfg.max_staging_bytes_per_device = 0;
    auto const plan                  = plan_cufile_pool(cfg, nullptr, quiet_platform());
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().staging_cap_bytes == 0);
    CHECK(plan.groups.front().lane_slots == 128);
  }
  SECTION("a small configured cap does not cut lanes")
  {
    auto cfg                         = fused_config();
    cfg.max_staging_bytes_per_device = 16 * MiB;
    auto const plan                  = plan_cufile_pool(cfg, nullptr, quiet_platform());
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 128);
    CHECK(plan.groups.front().staging_cap_bytes == 16 * MiB + 1032 * MiB);
    CHECK(plan.warnings.empty());
  }
  SECTION("max_lanes_per_device bounds the lanes")
  {
    auto cfg                             = fused_config();
    cfg.fused_reads.max_lanes_per_device = 40;
    cfg.fused_reads.reactors_per_device  = 2;
    cfg.fused_reads.lanes_per_reactor    = 64;
    cfg.fused_reads.slot_bytes           = 1 * MiB;
    auto const plan                      = plan_cufile_pool(cfg, nullptr, quiet_platform());
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_reactors == 2);
    CHECK(plan.groups.front().lanes_per_reactor == 64);
    CHECK(plan.groups.front().lane_slots == 40);
    CHECK(plan.groups.front().staging_cap_bytes ==
          cfg.max_staging_bytes_per_device + 40 * lane_footprint(1 * MiB));
    CHECK(plan.lane_slot_bytes == 1 * MiB);
    CHECK(plan.warnings.empty());
  }
  SECTION("an explicit synchronous reactor count is honoured")
  {
    auto cfg                = fused_config();
    cfg.reactors_per_device = 3;
    auto const plan         = plan_cufile_pool(cfg, nullptr, quiet_platform());
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().n_reactors == 3);
    CHECK(plan.total_reactors() == 3 + 4);
  }
  SECTION("cuFile's pinned limit cuts lanes: the warning names it")
  {
    auto platform                          = quiet_platform();
    platform.max_device_pinned_mem_size_kb = 64 * 1024;  // 64 MiB
    auto const plan = plan_cufile_pool(fused_config(), nullptr, platform);
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 7);
    // The rest of the hardware limit goes to the synchronous staging; the total stays in it.
    CHECK(plan.groups.front().staging_cap_bytes == 64 * MiB);
    CHECK(any_contains(plan.warnings, "max_device_pinned_mem_size"));
    CHECK(any_contains(plan.warnings, "holds only 7 of 128 fused-read lanes"));
    CHECK(any_contains(plan.warnings, "fewer lanes run"));
    CHECK_FALSE(any_contains(plan.warnings, "max_staging_bytes_per_device"));
  }
  SECTION("a roomy hardware limit: lanes first, the staging cap within what is left")
  {
    auto platform                          = quiet_platform();
    platform.max_device_pinned_mem_size_kb = 2048 * 1024;
    platform.max_device_cache_size_kb      = 128 * 1024;  // hard limit 1920 MiB
    auto cfg                               = fused_config();
    auto plan                              = plan_cufile_pool(cfg, nullptr, platform);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 128);
    // min(1024, 1920 - 1032) + 1032 = 1920.
    CHECK(plan.groups.front().staging_cap_bytes == 1920 * MiB);
    CHECK(plan.warnings.empty());
    // Without a configured cap the synchronous staging gets the rest of the hardware limit.
    cfg.max_staging_bytes_per_device = 0;
    plan                             = plan_cufile_pool(cfg, nullptr, platform);
    CHECK(plan.groups.front().staging_cap_bytes == 1920 * MiB);
    // A small configured cap stays below the hardware limit.
    cfg.max_staging_bytes_per_device = 100 * MiB;
    plan                             = plan_cufile_pool(cfg, nullptr, platform);
    CHECK(plan.groups.front().staging_cap_bytes == 1132 * MiB);
  }
}

TEST_CASE("pool plan sizes synchronous staging from the cap and what the lanes leave",
          "[io][cufile][fused_plan][placement][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  config cfg;
  cfg.devices                         = {0};
  cfg.staging_mode                    = staging::always;
  cfg.staging_piece_bytes             = 1 * MiB;
  cfg.reactors_per_device             = 3;
  cfg.fused_reads.enabled             = true;
  cfg.fused_reads.reactors_per_device = 1;
  cfg.fused_reads.lanes_per_reactor   = 4;
  auto const slot_footprint           = cfg.staging_slot_bytes() + 64 * KiB;
  auto const lanes                    = 4 * lane_footprint(8 * MiB);
  auto const pinned                   = [](std::size_t bytes) {
    auto platform                          = quiet_platform();
    platform.max_device_pinned_mem_size_kb = bytes / KiB;
    return platform;
  };

  SECTION("the configured cap holds every staging slot")
  {
    cfg.max_staging_bytes_per_device = 6 * slot_footprint;
    auto const plan                  = plan_cufile_pool(cfg, nullptr, quiet_platform());
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 4);
    CHECK(plan.groups.front().staging_slots == 6);
    CHECK(plan.groups.front().staging_cap_bytes == 6 * slot_footprint + lanes);
    CHECK(plan.warnings.empty());
  }
  SECTION("the configured cap binds the synchronous staging only, as without lanes")
  {
    cfg.max_staging_bytes_per_device = 3 * slot_footprint;
    auto const plan                  = plan_cufile_pool(cfg, nullptr, quiet_platform());
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 4);
    CHECK(plan.groups.front().staging_slots == 3);
    CHECK(plan.groups.front().staging_cap_bytes == 3 * slot_footprint + lanes);
    CHECK(plan.warnings.empty());
  }
  SECTION("the lanes leave the synchronous reactors three of six slots of the hardware limit")
  {
    auto const plan = plan_cufile_pool(cfg, nullptr, pinned(lanes + 3 * slot_footprint));
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 4);
    CHECK(plan.groups.front().staging_slots == 3);
    CHECK(plan.groups.front().staging_cap_bytes == lanes + 3 * slot_footprint);
    CHECK(any_contains(plan.warnings, "to its 3 synchronous reactors: 3 of 6 staging slots"));
    CHECK_FALSE(any_contains(plan.warnings, "holds only"));
  }
  SECTION("the hardware limit holds fewer lanes than wanted and no staging slot")
  {
    auto const plan = plan_cufile_pool(cfg, nullptr, pinned(2 * lane_footprint(8 * MiB)));
    INFO(describe(plan));
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().lane_slots == 2);
    CHECK(plan.groups.front().staging_slots == 0);
    CHECK(plan.groups.front().staging_cap_bytes == 2 * lane_footprint(8 * MiB));
    CHECK(any_contains(plan.warnings, "holds only 2 of 4 fused-read lanes"));
    CHECK(any_contains(plan.warnings, "0 of 6 staging slots"));
  }
}

TEST_CASE("pool plan with host batch adds shared batch reactors",
          "[io][cufile][fused_plan][placement][gpu]")
{
  auto const devices = device_count();
  if (devices == 0) SKIP("no CUDA device");
  config cfg;
  cfg.reactors_per_device = 2;
  cfg.staging_mode        = staging::never;
  cfg.host_batch.enabled  = true;
  cfg.host_batch.reactors = 3;
  auto const plan         = plan_cufile_pool(cfg, nullptr, quiet_platform());
  INFO(describe(plan));
  REQUIRE(plan.groups.size() == static_cast<std::size_t>(devices));
  CHECK(plan.host_batch_reactors == 3);
  CHECK(plan.total_reactors() == 2 * static_cast<std::size_t>(devices) + 3);
  for (auto const& group : plan.groups) {
    CHECK(group.lane_reactors == 0);
    CHECK(group.lane_slots == 0);
    CHECK(group.n_reactors == 2);
  }
  CHECK(plan.lane_slot_bytes == 0);
  CHECK(describe(plan).find("3 host-batch reactors") != std::string::npos);
}

TEST_CASE("pool plan plans an invalid feature block as disabled, with a warning",
          "[io][cufile][fused_plan][placement][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  auto cfg                          = fused_config();
  cfg.fused_reads.lanes_per_reactor = 0;
  cfg.host_batch.enabled            = true;
  cfg.host_batch.depth              = 0;
  auto const plan                   = plan_cufile_pool(cfg, nullptr, quiet_platform());
  INFO(describe(plan));
  REQUIRE(plan.groups.size() == 1);
  CHECK(plan.groups.front().lane_reactors == 0);
  CHECK(plan.groups.front().lane_slots == 0);
  CHECK(plan.groups.front().staging_cap_bytes == 0);
  CHECK(plan.lane_slot_bytes == 0);
  CHECK(plan.host_batch_reactors == 0);
  CHECK(any_contains(plan.warnings, "fused_reads is invalid"));
  CHECK(any_contains(plan.warnings, "host_batch is invalid"));
  // Planned as without the feature: the automatic synchronous count is the usual one.
  auto plain = cfg;
  plain.fused_reads = {};
  plain.host_batch  = {};
  CHECK(plan.groups.front().n_reactors ==
        plan_cufile_pool(plain, nullptr, quiet_platform()).groups.front().n_reactors);
}

TEST_CASE("pool plan with both features off is the legacy plan",
          "[io][cufile][fused_plan][placement][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  for (auto const mode : {staging::never, staging::always}) {
    config legacy;
    legacy.staging_mode = mode;
    // Disabled blocks are ignored, whatever they hold.
    auto disabled                            = legacy;
    disabled.fused_reads.reactors_per_device = 9;
    disabled.fused_reads.lanes_per_reactor   = 0;
    disabled.fused_reads.slot_bytes          = 1;
    disabled.host_batch.reactors             = 7;
    disabled.host_batch.depth                = 0;
    auto const a = plan_cufile_pool(legacy, nullptr, quiet_platform());
    auto const b = plan_cufile_pool(disabled, nullptr, quiet_platform());
    INFO(describe(a));
    CHECK(describe(a) == describe(b));
    CHECK(a.warnings == b.warnings);
    CHECK(a.host_batch_reactors == 0);
    CHECK(a.lane_slot_bytes == 0);
    REQUIRE(a.groups.size() == b.groups.size());
    for (std::size_t i = 0; i < a.groups.size(); ++i) {
      auto const& group = a.groups[i];
      CHECK(group.lane_reactors == 0);
      CHECK(group.lanes_per_reactor == 0);
      CHECK(group.lane_slots == 0);
      CHECK(group.n_reactors == b.groups[i].n_reactors);
      CHECK(group.staging_slots == b.groups[i].staging_slots);
      CHECK(group.staging_cap_bytes == b.groups[i].staging_cap_bytes);
      CHECK(group.reason == b.groups[i].reason);
      // Without staging the cap is not computed, as before.
      if (mode == staging::never) CHECK(group.staging_cap_bytes == 0);
      CHECK(group.reason.find("lane") == std::string::npos);
    }
    CHECK(a.total_reactors() == a.groups.size() * a.groups.front().n_reactors);
    CHECK(a.reason.find("lane") == std::string::npos);
  }
}

TEST_CASE("describe names the lanes, their slots and the batch reactors",
          "[io][cufile][fused_plan][placement][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  auto cfg               = fused_config();
  cfg.host_batch.enabled = true;
  auto const plan                  = plan_cufile_pool(cfg, nullptr, quiet_platform());
  auto const text                  = describe(plan);
  INFO(text);
  CHECK(text.find("4 lane reactors x 32 lanes") != std::string::npos);
  CHECK(text.find("128 lane slots of 8192 KiB (1032 MiB registered, on top of the staging cap)") !=
        std::string::npos);
  CHECK(text.find("registered staging cap 2056 MiB, lanes included") != std::string::npos);
  CHECK(text.find("lane slots of 8192 KiB") != std::string::npos);
  CHECK(text.find("2 host-batch reactors") != std::string::npos);
  CHECK(text.find("(4 lane, 8 synchronous, 2 host-batch)") != std::string::npos);
  CHECK(text.find("lanes carry the device reads") != std::string::npos);
}
