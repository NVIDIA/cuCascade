/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_test_utils.hpp"

#include <cucascade/io/cufile/capacity.hpp>
#include <cucascade/io/cufile/mount_policy.hpp>
#include <cucascade/io/cufile/topology.hpp>

#include <catch2/catch_all.hpp>
#include <sched.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <thread>

using namespace cucascade::io::cufile;
namespace {
struct sysfs_fixture {
  std::filesystem::path root;
  sysfs_fixture()
  {
    char path[]   = "/tmp/cufile-topology-XXXXXX";
    auto* created = ::mkdtemp(path);
    if (!created) throw std::runtime_error("mkdtemp");
    root = created;
    put("bus/pci/devices/0000:01:00.0/numa_node", "0");
    put("bus/pci/devices/0000:02:00.0/numa_node", "0");
    put("bus/pci/devices/0000:03:00.0/numa_node", "1");
    put("devices/system/node/node0/cpulist", "0-3,8-11");
    put("devices/system/node/node1/cpulist", "4-7,12-15");
    for (int cpu = 0; cpu < 16; ++cpu) {
      auto prefix = "devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      put(prefix + "physical_package_id", cpu % 8 < 4 ? "0" : "1");
      put(prefix + "core_id", std::to_string(cpu % 4));
    }
    std::filesystem::create_directories(root / "class/infiniband/remote");
    std::filesystem::create_directories(root / "class/infiniband/local");
    std::filesystem::create_symlink(root / "bus/pci/devices/0000:03:00.0",
                                    root / "class/infiniband/remote/device");
    std::filesystem::create_symlink(root / "bus/pci/devices/0000:02:00.0",
                                    root / "class/infiniband/local/device");
  }
  void put(std::string const& path, std::string const& value)
  {
    auto file = root / path;
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file) << value << '\n';
  }
  affinity_topology discover(std::vector<int> cpus = {
                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
  {
    return discover_affinity_topology("0000:01:00.0", root.string(), cpus);
  }
  ~sysfs_fixture()
  {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
};
}  // namespace

TEST_CASE("affinity discovers same-NUMA NIC and avoids SMT siblings", "[cufile][topology]")
{
  sysfs_fixture fs;
  auto topology = fs.discover();
  REQUIRE(topology.nics.size() == 2);
  CHECK(topology.gpu_bdf == "0000:01:00.0");
  CHECK(topology.gpu_node == 0);
  CHECK(topology.nics[0].name == "local");
  CHECK(topology.nics[0].bdf == "0000:02:00.0");
  auto plan = plan_affinity(topology, 4);
  CHECK(plan.reaper.cpu == 0);
  REQUIRE(plan.submitters.size() == 4);
  CHECK(plan.submitters[0].cpu == 1);
  CHECK(plan.submitters[1].cpu == 2);
  CHECK(plan.submitters[2].cpu == 3);
  CHECK(plan.submitters[3].cpu == 8);
  CHECK(plan.submitters[3].reason.find("exhausted") != std::string::npos);
}

TEST_CASE("affinity ignores a broken NIC symlink without losing healthy topology",
          "[cufile][topology]")
{
  sysfs_fixture fs;
  std::filesystem::create_directories(fs.root / "class/infiniband/broken");
  std::filesystem::create_symlink(fs.root / "missing-device",
                                  fs.root / "class/infiniband/broken/device");
  auto topology = fs.discover();
  CHECK(topology.nics.size() == 2);
  CHECK_FALSE(topology.cpus.empty());
  CHECK(topology.reason == "discovered");
}

TEST_CASE("affinity supports split NUMA, NIC BDF and explicit CPU overrides", "[cufile][topology]")
{
  sysfs_fixture fs;
  auto topology = fs.discover();
  auto plan     = plan_affinity(topology, 2, "remote");
  CHECK(plan.reaper.node == 0);
  CHECK(plan.reaper.cpu == 0);
  CHECK(plan.submitters[0].cpu == 4);
  CHECK(plan.submitters[1].cpu == 5);
  CHECK(plan.submitters[0].node == 1);
  CHECK(plan.submitters[0].nic == "remote");
  auto explicit_plan = plan_affinity(topology, 1, "0000:03:00.0", {6});
  CHECK(explicit_plan.submitters[0].cpu == 6);
  auto explicit_cpu = plan_affinity(topology, 1, {}, {7});
  CHECK(explicit_cpu.submitters[0].cpu == 7);
  CHECK(explicit_cpu.submitters[0].node == 1);
}

TEST_CASE("affinity intersects cpuset and reports missing topology without pinning",
          "[cufile][topology]")
{
  sysfs_fixture fs;
  auto topology = fs.discover({2, 3, 5});
  auto plan     = plan_affinity(topology, 1, "remote");
  CHECK(plan.reaper.cpu == 2);
  CHECK(plan.submitters[0].cpu == 5);
  CHECK(plan_affinity(topology, 1, {}, {0}).submitters[0].cpu == -1);
  CHECK(plan_affinity(topology, 1, "absent").submitters[0].cpu == -1);
  auto gpu_only = fs.discover({2, 3});
  auto fallback = plan_affinity(gpu_only, 1, "remote");
  CHECK(fallback.submitters[0].node == 0);
  CHECK(fallback.submitters[0].reason.find("GPU-node fallback") != std::string::npos);
  fs.put("bus/pci/devices/0000:01:00.0/numa_node", "-1");
  auto missing = plan_affinity(fs.discover(), 1);
  CHECK(missing.reaper.cpu == -1);
  CHECK(missing.submitters[0].cpu == -1);
  CHECK_FALSE(missing.reaper.reason.empty());
  fs.put("bus/pci/devices/0000:01:00.0/numa_node", "not-a-number");
  CHECK(plan_affinity(fs.discover(), 1).reaper.cpu == -1);
}

TEST_CASE("affinity falls back to GPU node without NICs and never changes caller affinity",
          "[cufile][topology]")
{
  sysfs_fixture fs;
  std::filesystem::remove_all(fs.root / "class/infiniband");
  auto plan = plan_affinity(fs.discover(), 1);
  CHECK(plan.submitters[0].node == 0);
  CHECK(plan.submitters[0].reason.find("fallback") != std::string::npos);
  cpu_set_t before{}, after{};
  REQUIRE(sched_getaffinity(0, sizeof(before), &before) == 0);
  int permitted = -1;
  for (int i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &before)) {
      permitted = i;
      break;
    }
  REQUIRE(permitted >= 0);
  affinity_decision decision;
  decision.cpu = permitted;
  std::thread worker([&] { apply_thread_affinity(decision); });
  worker.join();
  CHECK(decision.pinned);
  REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
  CHECK(CPU_EQUAL(&before, &after));
}

TEST_CASE("affinity core assignments can be shared across devices", "[cufile][topology]")
{
  sysfs_fixture fs;
  auto topology = fs.discover();
  std::vector<affinity_cpu> used;
  auto first  = plan_affinity(topology, 1, {}, {}, &used);
  auto second = plan_affinity(topology, 1, {}, {}, &used);
  CHECK(first.reaper.cpu == 0);
  CHECK(first.submitters[0].cpu == 1);
  CHECK(second.reaper.cpu == 2);
  CHECK(second.submitters[0].cpu == 3);
}

TEST_CASE("filesystem profile overrides include lanes without granting direct qualification",
          "[cufile][mount_policy]")
{
  async_engine_config cfg;
  mount_override entry;
  entry.value    = "/mount";
  entry.state    = mount_override_state::operator_qualified;
  entry.qd       = 64;
  entry.op_bytes = 4UL << 20;
  entry.lanes    = 10;
  entry.nic      = "remote";
  entry.cpus     = {4, 5};
  cfg.mount_overrides.push_back(entry);
  REQUIRE_NOTHROW(cfg.validate());
  mount_identity identity;
  identity.valid       = true;
  identity.mount_point = "/mount";
  identity.fs_type     = "nfs";
  auto key             = make_mount_policy_key(identity, "gpu", 1);
  mount_policy policy(key, identity, cfg);
  CHECK(policy.state() == mount_state::operator_qualified);
  CHECK(policy.base_profile().filesystem == filesystem_class::nfs_rdma);
  CHECK_FALSE(policy.base_profile().async_write_qualified);
  REQUIRE(policy.affinity_override());
  CHECK(policy.affinity_override()->match == entry.match);
  CHECK(policy.affinity_override()->value == entry.value);
  auto effective = fit(policy.base_profile().capacity, 256UL << 20, 160);
  CHECK(effective.qd == 8);
  CHECK(effective.lanes == 10);
  CHECK(effective.op_bytes == 4UL << 20);
}

TEST_CASE("validated Lustre and BeeGFS geometry selects the preferred operation size",
          "[cufile][mount_policy]")
{
  auto type = GENERATE("lustre", "beegfs");
  mount_identity identity;
  identity.valid                   = true;
  identity.fs_type                 = type;
  identity.geometry.geometry_bytes = 4UL << 20;
  mount_policy policy(make_mount_policy_key(identity, "geometry-gpu", 1), identity, {});
  CHECK(policy.base_profile().capacity.preferred_op_bytes == 4UL << 20);
  CHECK(policy.base_profile().geometry_bytes == 4UL << 20);
  CHECK_FALSE(policy.read_eligible());  // Geometry is not qualification.
}

TEST_CASE("placement override syntax validates without probing", "[cufile][config]")
{
  async_engine_config cfg;
  mount_override entry;
  entry.value = "/mount";
  cfg.mount_overrides.push_back(entry);
  auto bad = GENERATE(0, 1, 2);
  if (bad == 0) cfg.mount_overrides[0].cpus = {-1};
  if (bad == 1) cfg.mount_overrides[0].nic = std::string("nic\0hidden", 10);
  if (bad == 2) cfg.mount_overrides[0].lanes = 0;
  CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
}

TEST_CASE("engine pins only owned threads and reports mount NIC CPU overrides",
          "[cufile][engine][topology]")
{
  using namespace cucascade::io;
  using namespace cucascade::io::cufile::engine_test;
  if (!has_device()) SKIP("CUDA unavailable");
  sysfs_fixture fs;
  auto cfg                    = configuration(2);
  cfg.affinity                = true;
  cfg.submitters_per_device   = 2;
  cfg.device_lane_cap         = 8;
  cfg.arena_bytes             = 32 * mib;
  cfg.mount_overrides[0].nic  = "remote";
  cfg.mount_overrides[0].cpus = {4, 5, 6, 7};
  cpu_set_t before{}, after{};
  REQUIRE(sched_getaffinity(0, sizeof(before), &before) == 0);
  // This integration fixture actually pins threads; skip if none of its CPUs are allowed.
  bool eligible = true;
  for (int cpu : {0, 1, 2, 3, 4, 5, 6, 7})
    eligible = eligible && CPU_ISSET(cpu, &before);
  if (!eligible) SKIP("fixture CPUs are outside the process cpuset");
  fixture f(2 * mib, cfg);
  f.start([&](auto& options) {
    options.sysfs_root     = fs.root.string();
    options.gpu_bdf        = "0000:01:00.0";
    options.permitted_cpus = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  });
  auto group  = std::make_shared<grouped_coordinator>(2 * mib, 2);
  auto future = group->get_future();
  auto a      = f.slice(group, {0, mib});
  auto b      = f.slice(group, {mib, mib}, mib);
  b.policy    = f.make_policy(2);
  f.engine->hand_off(std::move(a));
  f.engine->hand_off(std::move(b));
  REQUIRE(std::move(future).get(5s) == 2 * mib);
  REQUIRE(f.engine->drain(5s));
  auto gauges = f.engine->gauges();
  CHECK(gauges.reaper_affinity.pinned);
  CHECK(gauges.reaper_affinity.node == 0);
  REQUIRE(gauges.submitter_affinity.size() == 2);
  for (auto const& value : gauges.submitter_affinity) {
    CHECK(value.pinned);
    CHECK(value.node == 1);
    CHECK(value.nic == "remote");
    CHECK(value.cpu >= 4);
    CHECK(value.cpu <= 7);
  }
  CHECK(gauges.submitter_affinity[0].cpu != gauges.submitter_affinity[1].cpu);
  REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
  CHECK(CPU_EQUAL(&before, &after));
}

TEST_CASE("retired engines release physical core reservations", "[cufile][engine][topology]")
{
  using namespace cucascade::io::cufile::engine_test;
  if (!has_device()) SKIP("CUDA unavailable");
  sysfs_fixture fs;
  auto cfg     = configuration(1);
  cfg.affinity = true;
  auto start   = [&] {
    auto f = std::make_unique<fixture>(mib, cfg);
    f->start([&](auto& options) {
      options.sysfs_root     = fs.root.string();
      options.gpu_bdf        = "0000:01:00.0";
      options.permitted_cpus = std::vector<int>{0, 1, 2, 3};
    });
    return f;
  };
  auto first  = start();
  auto second = start();
  CHECK(first->engine->gauges().reaper_affinity.cpu == 0);
  CHECK(second->engine->gauges().reaper_affinity.cpu == 2);
  first.reset();
  auto third = start();
  CHECK(third->engine->gauges().reaper_affinity.cpu == 0);
}

TEST_CASE("engine affinity is off by default and missing GPU NUMA is diagnostic",
          "[cufile][engine][topology]")
{
  using namespace cucascade::io::cufile::engine_test;
  if (!has_device()) SKIP("CUDA unavailable");
  auto enabled = GENERATE(false, true);
  auto cfg     = configuration(1);
  cfg.affinity = enabled;
  fixture f(mib, cfg);
  f.start([](auto& options) {
    options.gpu_bdf    = "0000:ff:00.0";
    options.sysfs_root = "/nonexistent-cufile-test-root";
  });
  REQUIRE(until([&] {
    auto value = f.engine->gauges();
    return !value.reaper_affinity.reason.empty();
  }));
  auto gauges = f.engine->gauges();
  CHECK_FALSE(gauges.reaper_affinity.pinned);
  CHECK(gauges.reaper_affinity.cpu == -1);
  if (!enabled)
    CHECK(gauges.reaper_affinity.reason == "affinity disabled");
  else
    CHECK(gauges.reaper_affinity.reason == "GPU NUMA node unavailable");
}
