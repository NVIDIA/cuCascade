/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "engine_test_utils.hpp"

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

TEST_CASE("production arena configuration fits filesystem profiles in a simulated engine",
          "[cufile][engine][arena]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto cfg                          = configuration(4);
  auto const which                  = GENERATE(0, 1, 2, 3, 4);
  cfg.arena_bytes                   = 64 * mib;
  cfg.arena_slot_classes            = which == 1   ? std::vector<unsigned>{1}
                                      : which == 2 ? std::vector<unsigned>{4, 8}
                                                   : std::vector<unsigned>{4};
  cfg.mount_overrides.front().value = which == 2   ? "gpfs"
                                      : which == 3 ? "wekafs"
                                      : which == 4 ? "nfs"
                                                   : "ext4";
  cfg.mount_overrides.front().op_bytes.reset();
  fixture f(8 * mib, cfg);
  f.arena->release_for_maintenance();
  auto const report = f.arena->bootstrap(cfg);
  REQUIRE(report.registered_bytes == cfg.arena_bytes);
  auto identity        = mount_identity{};
  identity.fs_type     = which == 2 ? "gpfs" : which == 3 ? "wekafs" : which == 4 ? "nfs" : "ext4";
  identity.block_bytes = 8 * mib;
  if (which == 2) identity.geometry.geometry_bytes = 8 * mib;
  identity.valid = true;
  f.policy =
    std::make_shared<mount_policy>(mount_policy_key{1, 1, "", "bootstrap-0", 1}, identity, cfg);
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(8 * mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, 8 * mib}));
  REQUIRE(until([&] { return f.transport->snapshot().blocked_completions != 0; }));
  REQUIRE(until([&] {
    auto const snapshot = f.engine->gauges();
    return snapshot.mounts.size() == 1 && snapshot.mounts[0].effective.op_bytes != 0;
  }));
  auto const gauges = f.engine->gauges();
  REQUIRE(gauges.mounts.size() == 1);
  auto const expected = which == 1 || which == 4 ? mib
                        : which == 2             ? 8 * mib
                        : which == 3             ? 2 * mib
                                                 : 4 * mib;
  CHECK(gauges.mounts[0].effective.op_bytes == expected);
  CHECK(gauges.arena_registered_class_mask == (which == 1 ? 1 : which == 2 ? 12 : 4));
  auto const arena = f.arena->gauges();
  CHECK(arena.registered_class_mask == gauges.arena_registered_class_mask);
  CHECK(arena.registrations == (which == 1 ? 64 : which == 2 ? 12 : 16));
  if (which == 2) {
    auto const effective = gauges.mounts[0].effective;
    CHECK(effective.lanes * 8 * mib <= 32 * mib);
    CHECK(effective.qd == 3);
  }
  f.transport->unblock_all();
  REQUIRE(std::move(future).get(5s) == 8 * mib);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.contents() == f.image);
}

TEST_CASE("engine distinguishes inode refusals from mount driver and device failures",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const submit = GENERATE(false, true);
  auto const code   = GENERATE(-EOPNOTSUPP,
                             -EINVAL,
                             -EIO,
                             -static_cast<int>(CU_FILE_IO_NOT_SUPPORTED),
                             -static_cast<int>(CU_FILE_NVFS_DRIVER_ERROR),
                             -static_cast<int>(CU_FILE_CUDA_DRIVER_ERROR));
  auto const error  = submit && code < -4096 ? -code : code;
  bool const local  = code == -EOPNOTSUPP || code == -EINVAL || code == -CU_FILE_IO_NOT_SUPPORTED;
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.index = 0;
  if (submit)
    fault.submit_status.error = error;
  else
    fault.completion_result = error;
  f.transport->add_fault(fault);
  f.start();
  auto read = [&] {
    auto group    = std::make_shared<grouped_coordinator>(mib, 1);
    auto future   = group->get_future();
    auto s        = f.slice(group, {0, mib});
    s.file_policy = std::make_shared<file_async_policy>();
    f.engine->hand_off(std::move(s));
    CHECK(std::move(future).get(5s) == mib);
    REQUIRE(f.engine->drain(5s));
  };
  read();
  CHECK(f.policy->read_eligible() == local);
  read();  // A different live inode must not pay another failed submit for mount failures.
  CHECK(f.transport->snapshot().read_calls == (local ? 2 : 1));
  CHECK(f.contents() == f.image);
}

TEST_CASE("mount placement plans use selectors and ignore non-placement overrides",
          "[cufile][engine][hardening][topology]")
{
  mount_override qd;
  qd.value = "/qd";
  qd.qd    = 8;
  mount_override a;
  a.value = "/a";
  a.nic   = "nic-a";
  mount_override b;
  b.value = "/b";
  b.cpus  = {4};
  std::vector<mount_override> engine{qd, a, b};
  std::vector<mount_override> foreign{b, qd, a};
  CHECK(mount_affinity_plan_index(engine, &foreign[0]) == 2);
  CHECK(mount_affinity_plan_index(engine, &foreign[1]) == 0);
  CHECK(mount_affinity_plan_index(engine, &foreign[2]) == 1);
  CHECK(mount_affinity_plan_index(engine, nullptr) == 0);
  foreign[0].value = "/absent";
  CHECK(mount_affinity_plan_index(engine, &foreign[0]) == 0);
  auto cfg            = configuration(1);
  cfg.mount_overrides = {qd};
  mount_identity identity;
  identity.mount_point = "/qd";
  mount_policy policy(mount_policy_key{1, 99, "", "placement-test", 1}, identity, cfg);
  CHECK_FALSE(policy.affinity_override());
}

TEST_CASE("fence hint delay is bounded like other completion polling floors",
          "[cufile][engine][hardening]")
{
  auto cfg               = configuration(1);
  cfg.poll.fence_hint_us = GENERATE(0U, 1U, 1000U, 1001U);
  if (cfg.poll.fence_hint_us == 0 || cfg.poll.fence_hint_us > 1000)
    CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
  else
    CHECK_NOTHROW(cfg.validate());
}
