/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The REMOTE tier in the core library: configuration, memory_space, configurator, manager and
// object naming. No I/O (the converters are tested in the cudf test executable).

#include "utils/mock_test_utils.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace cucascade;
using memory::Tier;

namespace {

[[nodiscard]] memory::remote_memory_space_config make_remote_config(
  int remote_id = 0, std::size_t capacity = memory::unbounded_remote_capacity)
{
  memory::remote_memory_space_config config;
  config.remote_id       = remote_id;
  config.memory_capacity = capacity;
  config.uri_prefix      = "s3://spill-bucket/cucascade";
  return config;
}

/// A non-null io context handle the core library stores but never dereferences (see
/// test_disk_memory_space.cpp).
[[nodiscard]] std::shared_ptr<io::ioctx> make_placeholder_context()
{
  auto owner = std::make_shared<int>(0);
  return std::shared_ptr<io::ioctx>(owner, reinterpret_cast<io::ioctx*>(owner.get()));
}

}  // namespace

TEST_CASE("Tier::REMOTE is appended after DISK", "[memory][remote]")
{
  STATIC_REQUIRE(static_cast<int>(Tier::GPU) == 0);
  STATIC_REQUIRE(static_cast<int>(Tier::HOST) == 1);
  STATIC_REQUIRE(static_cast<int>(Tier::DISK) == 2);
  STATIC_REQUIRE(static_cast<int>(Tier::REMOTE) == 3);
  STATIC_REQUIRE(static_cast<int>(Tier::SIZE) == 4);
  // The variant alternatives keep their indices; REMOTE is the new last one.
  STATIC_REQUIRE(std::variant_size_v<memory::memory_space_config> == 5);
  STATIC_REQUIRE(std::is_same_v<std::variant_alternative_t<4, memory::memory_space_config>,
                                memory::remote_memory_space_config>);
}

TEST_CASE("remote_memory_space_config limits and thresholds", "[memory][remote]")
{
  memory::remote_memory_space_config const defaults;
  CHECK(defaults.tier() == Tier::REMOTE);
  CHECK(defaults.remote_id == 0);
  CHECK(defaults.memory_capacity == memory::unbounded_remote_capacity);
  CHECK(defaults.io_context == nullptr);

  auto const config = make_remote_config(3, 1000);
  CHECK(config.reservation_limit() == 1000);
  CHECK(config.downgrade_trigger_threshold() == 1000);
  CHECK(config.downgrade_stop_threshold() == 990);
}

TEST_CASE("validate_remote_uri_prefix accepts and normalizes s3 prefixes", "[memory][remote]")
{
  CHECK(memory::validate_remote_uri_prefix("s3://bucket") == "s3://bucket");
  CHECK(memory::validate_remote_uri_prefix("s3://bucket/") == "s3://bucket");
  CHECK(memory::validate_remote_uri_prefix("S3://bucket/a/b//") == "s3://bucket/a/b");
  CHECK(memory::validate_remote_uri_prefix("s3://my.bucket-1/spill/run") ==
        "s3://my.bucket-1/spill/run");
}

TEST_CASE("validate_remote_uri_prefix rejects malformed prefixes", "[memory][remote]")
{
  auto const bad = GENERATE(as<std::string>{},
                            "",
                            "bucket/prefix",
                            "/local/path",
                            "file:///tmp/x",
                            "https://bucket/prefix",
                            "s3:/bucket",
                            "s3://",
                            "s3:///prefix",
                            "s3://key:secret@bucket/prefix",
                            "s3://user@bucket",
                            "s3://bucket:9000/prefix",
                            "s3://bucket/prefix?versionId=1",
                            "s3://bucket/prefix#frag",
                            "s3://bucket/a b",
                            " s3://bucket",
                            "s3://bucket/a\tb",
                            "s3://bucket/a//b");
  CAPTURE(bad);
  CHECK_THROWS_AS(memory::validate_remote_uri_prefix(bad), std::invalid_argument);
}

TEST_CASE("REMOTE memory_space basics", "[memory][remote][memory_space]")
{
  auto config       = make_remote_config(2, 1UL << 30);
  config.uri_prefix = "S3://spill-bucket/cucascade/";
  memory::memory_space space(config);

  CHECK(space.get_tier() == Tier::REMOTE);
  CHECK(space.get_device_id() == 2);
  CHECK(space.get_id() == memory::memory_space_id(Tier::REMOTE, 2));
  CHECK(space.get_max_memory() == (1UL << 30));
  CHECK(space.get_available_memory() == (1UL << 30));
  CHECK(space.get_remote_uri_prefix() == "s3://spill-bucket/cucascade");
  CHECK(space.to_string() == "memory_space(tier=REMOTE, device_id=2, limit=1073741824)");
  CHECK_THROWS_AS(space.get_disk_mount_path(), std::logic_error);
  CHECK_THROWS_AS(space.acquire_stream(), std::runtime_error);
  CHECK(space.get_memory_resource_as<memory::disk_access_limiter>() != nullptr);
  CHECK_FALSE(space.should_downgrade_memory());
}

TEST_CASE("REMOTE memory_space rejects a malformed uri prefix", "[memory][remote][memory_space]")
{
  auto config       = make_remote_config();
  config.uri_prefix = "";
  CHECK_THROWS_AS(memory::memory_space(config), std::invalid_argument);
  config.uri_prefix = "s3:///no-bucket";
  CHECK_THROWS_AS(memory::memory_space(config), std::invalid_argument);
  config.uri_prefix = "gs://bucket";
  CHECK_THROWS_AS(memory::memory_space(config), std::invalid_argument);
}

TEST_CASE("REMOTE memory_space stores, replaces and rejects io contexts",
          "[memory][remote][memory_space]")
{
  auto configured   = make_placeholder_context();
  auto config       = make_remote_config();
  config.io_context = configured;
  memory::memory_space space(config);
  CHECK(space.get_io_context() == configured);

  auto replacement = make_placeholder_context();
  space.set_io_context(replacement);
  CHECK(space.get_io_context() == replacement);
  CHECK_THROWS_AS(space.set_io_context(nullptr), std::invalid_argument);
  CHECK(space.get_io_context() == replacement);

  memory::memory_space bare(make_remote_config(1));
  CHECK(bare.get_io_context() == nullptr);
}

TEST_CASE("io context accessors still reject GPU and HOST spaces", "[memory][remote][memory_space]")
{
  for (auto const tier : {Tier::GPU, Tier::HOST}) {
    auto space = test::make_mock_memory_space(tier, 0);
    CHECK_THROWS_AS(space->get_io_context(), std::logic_error);
    CHECK_THROWS_AS(space->set_io_context(make_placeholder_context()), std::logic_error);
    CHECK_THROWS_AS(space->get_remote_uri_prefix(), std::logic_error);
  }
  auto disk = test::make_mock_memory_space(Tier::DISK, 0);
  CHECK_THROWS_AS(disk->get_remote_uri_prefix(), std::logic_error);
}

TEST_CASE("REMOTE reservations are bounded by the capacity", "[memory][remote][memory_space]")
{
  memory::memory_space space(make_remote_config(0, 1000));
  auto first = space.make_reservation_or_null(600);
  REQUIRE(first != nullptr);
  CHECK(first->tier() == Tier::REMOTE);
  CHECK(space.get_total_reserved_memory() == 600);
  CHECK(space.make_reservation_or_null(600) == nullptr);
  auto second = space.make_reservation_or_null(400);
  REQUIRE(second != nullptr);
  CHECK(space.get_available_memory() == 0);
  CHECK(space.should_downgrade_memory());  // full: REMOTE only reports it when completely full
  first.reset();
  CHECK(space.get_available_memory() == 600);
  CHECK_FALSE(space.should_downgrade_memory());
}

TEST_CASE("memory_reservation_manager serves REMOTE spaces", "[memory][remote]")
{
  std::vector<memory::memory_space_config> configs;
  configs.emplace_back(make_remote_config(0, 1000));
  configs.emplace_back(make_remote_config(1, 2000));
  memory::memory_reservation_manager manager(std::move(configs));

  auto spaces = manager.get_memory_spaces_for_tier(Tier::REMOTE);
  REQUIRE(spaces.size() == 2);
  CHECK(manager.get_memory_space(Tier::REMOTE, 1)->get_max_memory() == 2000);
  CHECK(manager.get_available_memory_for_tier(Tier::REMOTE) == 3000);

  auto reservation =
    manager.request_reservation(memory::any_memory_space_in_tier(Tier::REMOTE), 1500);
  REQUIRE(reservation != nullptr);
  CHECK(reservation->tier() == Tier::REMOTE);
  CHECK(reservation->get_memory_space().get_device_id() == 1);
  CHECK(manager.get_total_reserved_memory_for_tier(Tier::REMOTE) == 1500);
}

TEST_CASE("configurator emits REMOTE spaces after DISK", "[memory][remote][configurator]")
{
  memory::system_topology_info topology;  // no GPUs, no NUMA nodes: only DISK and REMOTE
  memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(0);
  auto context = make_placeholder_context();
  builder.add_remote_space(7, "s3://bucket/spill/", 1UL << 40, context);
  builder.set_disk_mounting_point(0, 1UL << 30, "/tmp");
  builder.add_remote_space(8, "s3://other");

  auto const configs = builder.build(topology);
  REQUIRE(configs.size() >= 3);
  auto const n = configs.size();
  REQUIRE(std::holds_alternative<memory::disk_memory_space_config>(configs[n - 3]));
  auto const* first  = std::get_if<memory::remote_memory_space_config>(&configs[n - 2]);
  auto const* second = std::get_if<memory::remote_memory_space_config>(&configs[n - 1]);
  REQUIRE(first != nullptr);
  REQUIRE(second != nullptr);
  CHECK(first->remote_id == 7);
  CHECK(first->uri_prefix == "s3://bucket/spill");
  CHECK(first->memory_capacity == (1UL << 40));
  CHECK(first->io_context == context);
  CHECK(second->remote_id == 8);
  CHECK(second->memory_capacity == memory::unbounded_remote_capacity);
  CHECK(second->io_context == nullptr);

  CHECK_THROWS_AS(builder.add_remote_space(9, "not-a-uri"), std::invalid_argument);
}

TEST_CASE("generate_remote_object_uri names unique objects under the prefix", "[memory][remote]")
{
  auto const uri = memory::generate_remote_object_uri("s3://bucket/spill");
  CHECK(uri.rfind("s3://bucket/spill/", 0) == 0);
  CHECK(uri.find("/batch_") != std::string::npos);
  CHECK(uri.ends_with(".cucascade"));
  // {prefix}/{run_id}/batch_...: exactly one path level between the prefix and the object.
  auto const tail = uri.substr(std::string{"s3://bucket/spill/"}.size());
  auto const run  = tail.substr(0, tail.find('/'));
  CHECK_FALSE(run.empty());
  CHECK(tail.find('/', run.size() + 1) == std::string::npos);
  CHECK(memory::validate_remote_uri_prefix(uri) == uri);  // a valid key, no stray characters

  // The run id is fixed per process; a trailing '/' on the prefix is not doubled.
  auto const other = memory::generate_remote_object_uri("s3://bucket/spill/");
  CHECK(other.rfind("s3://bucket/spill/" + run + "/batch_", 0) == 0);

  constexpr std::size_t n_threads  = 16;
  constexpr std::size_t per_thread = 1000;
  std::vector<std::vector<std::string>> names(n_threads);
  {
    std::atomic<bool> go{false};
    std::vector<std::jthread> threads;
    for (std::size_t t = 0; t < n_threads; ++t) {
      threads.emplace_back([&, t] {
        while (!go.load(std::memory_order_acquire)) {}
        for (std::size_t i = 0; i < per_thread; ++i) {
          names[t].push_back(memory::generate_remote_object_uri("s3://bucket/spill"));
        }
      });
    }
    go.store(true, std::memory_order_release);
  }
  std::set<std::string> unique;
  for (auto const& list : names) {
    unique.insert(list.begin(), list.end());
  }
  CHECK(unique.size() == n_threads * per_thread);
}
