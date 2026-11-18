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

#include <catch2/catch_all.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace cucascade::io::cufile;

namespace {

void check_invalid(async_engine_config const& cfg, std::string const& field)
{
  CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
  CHECK_THROWS_WITH(cfg.validate(), Catch::Matchers::ContainsSubstring(field));
}

}  // namespace

TEST_CASE("production arena slot classes validate", "[io][cufile][config]")
{
  async_engine_config cfg;
  CHECK(cfg.arena_slot_classes == std::vector<unsigned>{4});
  for (auto const& bad :
       std::vector<std::vector<unsigned>>{{}, {0}, {3}, {16}, {4, 4}, {1, 8, 7}}) {
    cfg.arena_slot_classes = bad;
    check_invalid(cfg, "arena_slot_classes");
  }
  cfg.arena_slot_classes = {8, 1, 4, 2};
  CHECK_NOTHROW(cfg.validate());
  cfg.process_params.cufile_json_path = "";
  CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
  cfg.process_params.cufile_json_path = std::string("bad") + char(0) + "path";
  CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
}

TEST_CASE("async engine configuration defaults preserve legacy routing", "[io][cufile][config]")
{
  config const reactor;
  auto const& cfg = reactor.async_engine;
  CHECK(cfg.mode == engine_mode::legacy);
  CHECK(cfg.async_reads);
  CHECK_FALSE(cfg.async_writes);
  CHECK(cfg.unregistered_device_tier);
  CHECK_FALSE(cfg.direct_registered_io);
  CHECK(cfg.stream_register);
  CHECK_FALSE(cfg.affinity);
  CHECK(cfg.coalesce);
  CHECK(cfg.coalesce_max_targets == 64);
  CHECK(cfg.submitters_per_device == 1);
  CHECK(cfg.arena_bytes == 0);
  CHECK(cfg.device_lane_cap == 160);
  CHECK(cfg.mount_overrides.empty());

  CHECK(cfg.probe.submit_max_us == 100);
  CHECK(cfg.probe.require_direct_evidence);
  CHECK(cfg.probe.bytes == 16UL << 20);
  CHECK(cfg.demotion.slow_floor_us == 100);
  CHECK(cfg.demotion.p99_multiple == 4);
  CHECK(cfg.demotion.slow_of_last_eight == 3);
  CHECK(cfg.demotion.stuck_ms == 2);
  CHECK(cfg.demotion.cooldown_s == 30);
  CHECK(cfg.demotion.cooldown_cap_s == 960);
  CHECK(cfg.poll.io_cap_us == 100);
  CHECK(cfg.poll.copy_cap_us == 10);
  CHECK(cfg.poll.fence_hint_us == 100);
  CHECK(cfg.poll.due_events_per_round == 16);
  CHECK(cfg.poll.idle_health_ms == 20);
  CHECK_NOTHROW(cfg.validate());

  // Existing legacy knobs keep their defaults and are not rewritten by validation.
  CHECK(reactor.max_inflight == 32);
  CHECK(reactor.max_op_bytes == 64UL << 20);
  CHECK(reactor.staging_mode == cucascade::io::cufile::staging::automatic);
  CHECK(reactor.write_split_bytes == (64UL << 20));
  CHECK(reactor.gds_threshold == 256UL << 10);
  CHECK(reactor.use_odirect);
  CHECK_FALSE(reactor.compat_mode.has_value());
  CHECK(reactor.poll_interval_us == 200);
  CHECK(reactor.low_service_interval == 4);
}

TEST_CASE("async engine modes and feature switches validate independently", "[io][cufile][config]")
{
  auto const mode = GENERATE(
    engine_mode::legacy, engine_mode::async_auto, engine_mode::async_simulated_for_testing);
  for (unsigned flags = 0; flags < 128; ++flags) {
    CAPTURE(mode, flags);
    async_engine_config cfg;
    cfg.mode                          = mode;
    cfg.async_reads                   = (flags & 1U) != 0;
    cfg.async_writes                  = (flags & 2U) != 0;
    cfg.unregistered_device_tier      = (flags & 4U) != 0;
    cfg.direct_registered_io          = (flags & 8U) != 0;
    cfg.stream_register               = (flags & 16U) != 0;
    cfg.affinity                      = (flags & 32U) != 0;
    cfg.probe.require_direct_evidence = (flags & 64U) != 0;
    CHECK_NOTHROW(cfg.validate());
    CHECK(cfg.mode == mode);
  }
}

TEST_CASE("async write overrides require explicit operator qualification", "[io][cufile][config]")
{
  auto state = GENERATE(mount_override_state::operator_qualified,
                        mount_override_state::known_compat,
                        mount_override_state::legacy);
  async_engine_config cfg;
  cfg.mount_overrides.push_back({mount_match::mount_prefix, "/", state, {}, {}, true});
  if (state == mount_override_state::operator_qualified)
    CHECK_NOTHROW(cfg.validate());
  else
    check_invalid(cfg, "allow_writes");
}

TEST_CASE("async engine rejects unknown mode values", "[io][cufile][config]")
{
  auto const invalid = GENERATE(3U, 255U);
  async_engine_config cfg;
  cfg.mode = static_cast<engine_mode>(invalid);
  check_invalid(cfg, "mode");
}

TEST_CASE("async engine requires positive counts and timing thresholds", "[io][cufile][config]")
{
  async_engine_config cfg;
  std::array<std::pair<char const*, unsigned*>, 14> const fields{{
    {"submitters_per_device", &cfg.submitters_per_device},
    {"device_lane_cap", &cfg.device_lane_cap},
    {"probe.submit_max_us", &cfg.probe.submit_max_us},
    {"demotion.slow_floor_us", &cfg.demotion.slow_floor_us},
    {"demotion.p99_multiple", &cfg.demotion.p99_multiple},
    {"demotion.slow_of_last_eight", &cfg.demotion.slow_of_last_eight},
    {"demotion.stuck_ms", &cfg.demotion.stuck_ms},
    {"demotion.cooldown_s", &cfg.demotion.cooldown_s},
    {"demotion.cooldown_cap_s", &cfg.demotion.cooldown_cap_s},
    {"poll.io_cap_us", &cfg.poll.io_cap_us},
    {"poll.copy_cap_us", &cfg.poll.copy_cap_us},
    {"poll.fence_hint_us", &cfg.poll.fence_hint_us},
    {"poll.due_events_per_round", &cfg.poll.due_events_per_round},
    {"poll.idle_health_ms", &cfg.poll.idle_health_ms},
  }};
  for (auto const& [name, field] : fields) {
    CAPTURE(name);
    auto const previous = *field;
    *field              = 0;
    check_invalid(cfg, name);
    *field = previous;
    CHECK_NOTHROW(cfg.validate());
  }
}

TEST_CASE("async engine bounds submitter and rolling demotion counts", "[io][cufile][config]")
{
  async_engine_config cfg;
  auto const count = GENERATE(1U, 8U, 9U, std::numeric_limits<unsigned>::max());
  SECTION("submitter count")
  {
    cfg.submitters_per_device = count;
    if (count <= 8) {
      CHECK_NOTHROW(cfg.validate());
    } else {
      check_invalid(cfg, "submitters_per_device");
    }
  }
  SECTION("slow submissions among the last eight")
  {
    cfg.demotion.slow_of_last_eight = count;
    if (count <= 8) {
      CHECK_NOTHROW(cfg.validate());
    } else {
      check_invalid(cfg, "demotion.slow_of_last_eight");
    }
  }
}

TEST_CASE("async engine cooldown cap cannot precede its initial delay", "[io][cufile][config]")
{
  async_engine_config cfg;
  cfg.demotion.cooldown_s     = 31;
  auto const cap              = GENERATE(30U, 31U, 32U);
  cfg.demotion.cooldown_cap_s = cap;
  if (cap < cfg.demotion.cooldown_s) {
    check_invalid(cfg, "demotion.cooldown_cap_s");
  } else {
    CHECK_NOTHROW(cfg.validate());
  }
}

TEST_CASE("async engine accepts auto and explicit arena targets unchanged", "[io][cufile][config]")
{
  async_engine_config cfg;
  // AM-5 explicitly replaces the old multiple-of-16-MiB requirement and zero-disable meaning.
  auto const bytes = GENERATE(std::size_t{0},
                              std::size_t{1},
                              std::size_t{(16UL << 20) + 1},
                              std::numeric_limits<std::size_t>::max());
  cfg.arena_bytes  = bytes;
  CHECK_NOTHROW(cfg.validate());
  CHECK(cfg.arena_bytes == bytes);
}

TEST_CASE("async engine probe ranges must be positive and block aligned", "[io][cufile][config]")
{
  constexpr auto block = cucascade::io::IO_BLOCK_SIZE;
  async_engine_config cfg;
  auto const bytes = GENERATE(std::size_t{0},
                              std::size_t{1},
                              std::size_t{block - 1},
                              std::size_t{block},
                              std::size_t{block + 1},
                              std::size_t{3 * block});
  cfg.probe.bytes  = bytes;
  if (bytes == 0 || bytes % block != 0) {
    check_invalid(cfg, "probe.bytes");
  } else {
    CHECK_NOTHROW(cfg.validate());
  }
}

TEST_CASE("async engine mount overrides support every selector and routing state",
          "[io][cufile][config]")
{
  auto const match = GENERATE(mount_match::fsid, mount_match::mount_prefix, mount_match::fs_type);
  auto const state = GENERATE(mount_override_state::operator_qualified,
                              mount_override_state::known_compat,
                              mount_override_state::legacy);
  async_engine_config cfg;
  mount_override entry;
  entry.match = match;
  entry.state = state;
  switch (match) {
    case mount_match::fsid: entry.value = "0123:4567"; break;
    case mount_match::mount_prefix: entry.value = "/mnt/disk_2"; break;
    case mount_match::fs_type: entry.value = "ext4"; break;
  }
  CHECK_FALSE(entry.qd.has_value());
  CHECK_FALSE(entry.op_bytes.has_value());
  cfg.mount_overrides.push_back(entry);
  CHECK_NOTHROW(cfg.validate());

  // Preferences may exceed device capacity, which is fitted later, not rejected here.
  cfg.device_lane_cap                  = 1;
  cfg.mount_overrides.front().qd       = 128;
  cfg.mount_overrides.front().op_bytes = 3UL << 20;
  CHECK_NOTHROW(cfg.validate());
}

TEST_CASE("async engine checks every entry of a mount override list", "[io][cufile][config]")
{
  async_engine_config cfg;
  cfg.mount_overrides = {
    {mount_match::fsid, "1:2", mount_override_state::operator_qualified, 32, 8UL << 20},
    {mount_match::mount_prefix, "/", mount_override_state::legacy, std::nullopt, std::nullopt},
    {mount_match::fs_type, "ext4", mount_override_state::known_compat, std::nullopt, std::nullopt},
  };
  CHECK_NOTHROW(cfg.validate());
  cfg.mount_overrides.back().qd = 0;
  check_invalid(cfg, "mount_overrides.qd");
}

TEST_CASE("async engine rejects duplicate mount selectors regardless of state or profile",
          "[io][cufile][config]")
{
  auto const match = GENERATE(mount_match::fsid, mount_match::mount_prefix, mount_match::fs_type);
  auto const state = GENERATE(mount_override_state::operator_qualified,
                              mount_override_state::known_compat,
                              mount_override_state::legacy);
  auto const repeated_state = GENERATE(mount_override_state::operator_qualified,
                                       mount_override_state::known_compat,
                                       mount_override_state::legacy);
  async_engine_config cfg;
  mount_override entry;
  entry.match = match;
  entry.state = state;
  switch (match) {
    case mount_match::fsid: entry.value = "0123:4567"; break;
    case mount_match::mount_prefix: entry.value = "/mnt/disk_2"; break;
    case mount_match::fs_type: entry.value = "ext4"; break;
  }
  cfg.mount_overrides.push_back(entry);
  cfg.mount_overrides.push_back(
    {mount_match::fs_type, "tmpfs", mount_override_state::legacy, std::nullopt, std::nullopt});
  entry.state    = repeated_state;
  entry.qd       = 128;
  entry.op_bytes = 8UL << 20;
  cfg.mount_overrides.push_back(entry);
  check_invalid(cfg, "mount_overrides");
  CHECK_THROWS_WITH(cfg.validate(), Catch::Matchers::ContainsSubstring("duplicate"));
}

TEST_CASE("async engine accepts overlapping prefixes and distinct selector kinds",
          "[io][cufile][config]")
{
  async_engine_config cfg;
  cfg.mount_overrides = {
    {mount_match::fs_type, "xfs", mount_override_state::known_compat, std::nullopt, std::nullopt},
    {mount_match::mount_prefix, "/", mount_override_state::legacy, std::nullopt, std::nullopt},
    {mount_match::mount_prefix,
     "/mnt",
     mount_override_state::known_compat,
     std::nullopt,
     std::nullopt},
    {mount_match::mount_prefix,
     "/mnt/disk_2",
     mount_override_state::operator_qualified,
     std::nullopt,
     std::nullopt},
    {mount_match::fsid,
     "xfs",
     mount_override_state::operator_qualified,
     std::nullopt,
     std::nullopt},
  };
  CHECK_NOTHROW(cfg.validate());
  // Validation preserves list order; the future policy layer applies the documented precedence.
  CHECK(cfg.mount_overrides.front().match == mount_match::fs_type);
  CHECK(cfg.mount_overrides[2].value == "/mnt");
  CHECK(cfg.mount_overrides[3].value == "/mnt/disk_2");
  CHECK(cfg.mount_overrides.back().match == mount_match::fsid);
}

TEST_CASE("async engine mount override defaults to an explicit legacy decision",
          "[io][cufile][config]")
{
  mount_override const entry;
  CHECK(entry.match == mount_match::mount_prefix);
  CHECK(entry.value.empty());
  CHECK(entry.state == mount_override_state::legacy);
  CHECK_FALSE(entry.qd.has_value());
  CHECK_FALSE(entry.op_bytes.has_value());
  async_engine_config cfg;
  cfg.mount_overrides.push_back(entry);
  check_invalid(cfg, "mount_overrides.value");
}

TEST_CASE("async engine rejects unknown mount selector and state values", "[io][cufile][config]")
{
  async_engine_config cfg;
  cfg.mount_overrides.push_back({mount_match::mount_prefix,
                                 "/mnt/disk_2",
                                 mount_override_state::legacy,
                                 std::nullopt,
                                 std::nullopt});
  auto const invalid = GENERATE(3U, 255U);
  SECTION("selector")
  {
    cfg.mount_overrides.front().match = static_cast<mount_match>(invalid);
    check_invalid(cfg, "mount_overrides.match");
  }
  SECTION("state")
  {
    cfg.mount_overrides.front().state = static_cast<mount_override_state>(invalid);
    check_invalid(cfg, "mount_overrides.state");
  }
}

TEST_CASE("async engine mount selectors require nonempty NUL-free values", "[io][cufile][config]")
{
  auto const match = GENERATE(mount_match::fsid, mount_match::mount_prefix, mount_match::fs_type);
  async_engine_config cfg;
  mount_override entry;
  entry.match = match;
  entry.value = GENERATE(std::string{}, std::string{"/mnt\0suffix", 11});
  cfg.mount_overrides.push_back(entry);
  check_invalid(cfg, "mount_overrides.value");
}

TEST_CASE("async engine mount prefixes must be normalized absolute paths", "[io][cufile][config]")
{
  async_engine_config cfg;
  mount_override entry;
  entry.value = GENERATE("mnt/disk_2",
                         ".",
                         "//",
                         "//mnt",
                         "/mnt//disk_2",
                         "/mnt/disk_2/",
                         "/.",
                         "/mnt/.",
                         "/mnt/./disk_2",
                         "/..",
                         "/mnt/..",
                         "/mnt/../x",
                         "/./mnt",
                         "/../mnt");
  cfg.mount_overrides.push_back(entry);
  check_invalid(cfg, "mount_overrides.value");
}

TEST_CASE("async engine accepts root and normalized mount prefixes unchanged",
          "[io][cufile][config]")
{
  async_engine_config cfg;
  mount_override entry;
  entry.value = GENERATE("/", "/mnt", "/mnt/disk_2", "/mnt/.cache", "/mnt/..hidden", "/mnt/disk.");
  cfg.mount_overrides.push_back(entry);
  CHECK_NOTHROW(cfg.validate());
  CHECK(cfg.mount_overrides.front().value == entry.value);
}

TEST_CASE("async engine filesystem types reject whitespace in every position",
          "[io][cufile][config]")
{
  auto const whitespace = GENERATE(' ', '\t', '\n', '\r', '\f', '\v');
  std::string const space(1, whitespace);
  for (auto const& value : {space + "ext4", "ex" + space + "t4", "ext4" + space}) {
    CAPTURE(value);
    async_engine_config cfg;
    cfg.mount_overrides.push_back(
      {mount_match::fs_type, value, mount_override_state::legacy, std::nullopt, std::nullopt});
    check_invalid(cfg, "mount_overrides.value");
    CHECK_THROWS_WITH(cfg.validate(), Catch::Matchers::ContainsSubstring("whitespace"));
  }
}

TEST_CASE("async engine optional profile overrides are validated when present",
          "[io][cufile][config]")
{
  constexpr auto block = cucascade::io::IO_BLOCK_SIZE;
  async_engine_config cfg;
  cfg.mount_overrides.push_back({mount_match::fs_type,
                                 "ext4",
                                 mount_override_state::operator_qualified,
                                 std::nullopt,
                                 std::nullopt});
  auto& entry = cfg.mount_overrides.front();
  SECTION("queue depth must be positive")
  {
    entry.qd = 0;
    check_invalid(cfg, "mount_overrides.qd");
    entry.qd = 1;
    CHECK_NOTHROW(cfg.validate());
  }
  SECTION("operation bytes must be positive and block aligned")
  {
    auto const bytes = GENERATE(std::size_t{0},
                                std::size_t{1},
                                std::size_t{block - 1},
                                std::size_t{block},
                                std::size_t{block + 1},
                                std::size_t{3 * block});
    entry.op_bytes   = bytes;
    if (bytes == 0 || bytes % block != 0) {
      check_invalid(cfg, "mount_overrides.op_bytes");
    } else {
      CHECK_NOTHROW(cfg.validate());
    }
  }
}
