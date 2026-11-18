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

#include "transport_simulator.hpp"

#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/mount_policy.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace cucascade::io::cufile;
using namespace std::chrono_literals;

namespace {
constexpr std::size_t mib = 1UL << 20;

class temporary_file {
 public:
  explicit temporary_file(std::size_t bytes = 2 * mib)
  {
    auto name = (std::filesystem::temp_directory_path() / "cucascade_policy_XXXXXX").string();
    std::vector<char> writable(name.begin(), name.end());
    writable.push_back('\0');
    fd = ::mkstemp(writable.data());
    if (fd < 0) { throw std::runtime_error("mkstemp failed"); }
    path = writable.data();
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      static_cast<void>(::close(fd));
      static_cast<void>(::unlink(path.c_str()));
      throw std::runtime_error("ftruncate failed");
    }
  }
  ~temporary_file()
  {
    static_cast<void>(::close(fd));
    static_cast<void>(::unlink(path.c_str()));
  }
  temporary_file(temporary_file const&)            = delete;
  temporary_file& operator=(temporary_file const&) = delete;
  int fd{-1};
  std::string path;
};

struct fake_clock {
  policy_clock::time_point now{};
};

class fake_executor final : public control_executor {
 public:
  explicit fake_executor(std::shared_ptr<fake_clock> clock) : _clock(std::move(clock)) {}
  bool accept{true};
  std::deque<std::function<void()>> ready;
  std::vector<std::pair<policy_clock::time_point, std::function<void()>>> timers;
  bool submit(std::function<void()> task) noexcept override
  {
    if (!accept) { return false; }
    try {
      ready.push_back(std::move(task));
      return true;
    } catch (...) {
      return false;
    }
  }
  bool schedule(policy_clock::time_point due, std::function<void()> task) noexcept override
  {
    if (!accept) { return false; }
    try {
      timers.emplace_back(due, std::move(task));
      return true;
    } catch (...) {
      return false;
    }
  }
  void run_ready()
  {
    for (;;) {
      auto timer = std::find_if(timers.begin(), timers.end(), [this](auto const& entry) {
        return entry.first <= _clock->now;
      });
      if (timer != timers.end()) {
        ready.push_back(std::move(timer->second));
        timers.erase(timer);
      }
      if (ready.empty()) { return; }
      auto task = std::move(ready.front());
      ready.pop_front();
      task();
    }
  }

 private:
  std::shared_ptr<fake_clock> _clock;
};

mount_identity synthetic_identity()
{
  return {17, 29, "01020304", "/mnt/policy/data", "ext4", 4096, true};
}

async_engine_config policy_config(bool qualified = false)
{
  async_engine_config config;
  config.mode = engine_mode::async_simulated_for_testing;
  if (qualified) {
    config.mount_overrides.push_back({mount_match::fsid,
                                      synthetic_identity().fsid,
                                      mount_override_state::operator_qualified,
                                      {},
                                      {}});
  }
  return config;
}

struct policy_fixture {
  std::shared_ptr<fake_clock> clock{std::make_shared<fake_clock>()};
  std::shared_ptr<fake_executor> executor{std::make_shared<fake_executor>(clock)};
  std::shared_ptr<mount_policy> policy;
  explicit policy_fixture(async_engine_config config = policy_config(true),
                          mount_identity identity    = synthetic_identity())
    : policy(
        std::make_shared<mount_policy>(make_mount_policy_key(identity, "policy-test-device", 91),
                                       identity,
                                       std::move(config),
                                       executor,
                                       [clock = clock] { return clock->now; }))
  {
    policy->set_effective({32, 4 * mib, 40});
  }
};

class simulator_evidence final : public evidence_source {
 public:
  explicit simulator_evidence(std::shared_ptr<transport_simulator> transport,
                              bool available    = true,
                              bool attributable = true)
    : _transport(std::move(transport)), _available(available), _attributable(attributable)
  {
  }
  evidence_snapshot sample() noexcept override
  {
    ++samples;
    auto const bytes = constant_counter ? 0 : _transport->snapshot().bytes_read;
    return {_available, _attributable, bytes + (samples > 1 ? unrelated_bytes : 0), 0};
  }
  unsigned samples{};
  std::uint64_t unrelated_bytes{};
  bool constant_counter{false};

 private:
  std::shared_ptr<transport_simulator> _transport;
  bool _available;
  bool _attributable;
};

void require_cuda()
{
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    SKIP("CUDA is needed for the transport simulator, not real direct GDS");
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
}

struct probe_fixture {
  temporary_file file;
  std::shared_ptr<cufile_io_object const> opened{
    cufile_reactor::create_io_object(file.path, false, compat::on)};
  std::shared_ptr<transport_simulator> transport{
    std::make_shared<transport_simulator>(std::vector<std::byte>(2 * mib, std::byte{0x5a}))};
  std::shared_ptr<device_arena> arena;
  std::shared_ptr<simulator_evidence> evidence{std::make_shared<simulator_evidence>(transport)};
  policy_fixture f;
  explicit probe_fixture(bool registered        = true,
                         bool require_evidence  = true,
                         unsigned submit_max_us = 100000)
    : f([require_evidence, submit_max_us] {
        auto config        = policy_config();
        config.probe.bytes = 2 * mib;
        // Simulator cold CUDA/pinned allocations are not production cuFile submit latency.
        config.probe.submit_max_us           = submit_max_us;
        config.probe.require_direct_evidence = require_evidence;
        return config;
      }())
  {
    device_arena_hooks hooks;
    hooks.allocate = [](int, std::size_t bytes) -> void* {
      void* ptr{};
      return cudaMalloc(&ptr, bytes) == cudaSuccess ? ptr : nullptr;
    };
    hooks.deallocate = [](int, void* ptr) { return cudaFree(ptr) == cudaSuccess; };
    // These are simulator-owned registrations: never call the process cuFile API table.
    hooks.register_buffer   = [registered](int, void*, std::size_t) { return registered; };
    hooks.deregister_buffer = [](int, void*) { return true; };
    arena                   = std::make_shared<device_arena>(0, std::move(hooks));
    auto config             = policy_config();
    config.arena_bytes      = device_arena::slab_bytes;
    REQUIRE(arena->warm_up(config).allocated_bytes == device_arena::slab_bytes);
    // Unknown mounts have no production quota. Qualification must not depend on rebalance().
  }
  ~probe_fixture()
  {
    transport->unblock_all();
    try {
      arena->release_for_maintenance();
    } catch (...) {
    }
  }
  mount_probe_context context() const
  {
    return {opened, arena.get(), transport, evidence, 0, 1, 4096, true, true};
  }
  void run()
  {
    // A fake queue controls admission deterministically; the probe still executes off-caller.
    std::thread worker([this] { f.executor->run_ready(); });
    worker.join();
  }
};
}  // namespace

TEST_CASE("verified mounts revoke async writes at the 16 MiB evidence boundary",
          "[cufile][mount_policy]")
{
  require_cuda();
  auto const direct = GENERATE(false, true);
  probe_fixture p;
  auto cfg                = policy_config();
  cfg.async_writes        = true;
  cfg.probe.bytes         = 2 * mib;
  cfg.probe.submit_max_us = 100000;
  auto identity           = synthetic_identity();
  p.f.policy =
    std::make_shared<mount_policy>(make_mount_policy_key(identity, "write-evidence-test", 91),
                                   identity,
                                   cfg,
                                   p.f.executor,
                                   [clock = p.f.clock] { return clock->now; });
  p.f.policy->ensure_probe(p.context());
  p.run();
  REQUIRE(p.f.policy->state() == mount_state::verified_direct);
  REQUIRE(p.f.policy->write_eligible());
  p.f.policy->observe_write_evidence(8 * mib, direct);
  CHECK(p.f.policy->write_eligible());
  p.f.policy->observe_write_evidence(8 * mib, false);
  CHECK(p.f.policy->write_eligible() == direct);
  CHECK(p.f.policy->read_eligible());
  p.f.policy->observe_write_evidence(16 * mib, true);  // No second chance or second diagnostic.
  CHECK(p.f.policy->write_eligible() == direct);
}

TEST_CASE("mount policy identity comes from a real descriptor", "[cufile][mount_policy]")
{
  temporary_file file;
  auto const identity = compute_mount_identity(file.fd);
  REQUIRE(identity.valid);
  struct stat namespace_stat{};
  struct statfs filesystem_stat{};
  REQUIRE(::stat("/proc/self/ns/mnt", &namespace_stat) == 0);
  REQUIRE(::fstatfs(file.fd, &filesystem_stat) == 0);
  CHECK(identity.namespace_inode == static_cast<std::uint64_t>(namespace_stat.st_ino));
  CHECK(identity.mount_id != 0);
  CHECK(identity.block_bytes == static_cast<std::size_t>(filesystem_stat.f_bsize));
  REQUIRE_FALSE(identity.mount_point.empty());
  CHECK(identity.mount_point.front() == '/');
  CHECK_FALSE(identity.fs_type.empty());
  std::string expected_fsid;
  auto const* bytes = reinterpret_cast<unsigned char const*>(&filesystem_stat.f_fsid);
  for (std::size_t i = 0; i < sizeof(filesystem_stat.f_fsid); ++i) {
    expected_fsid += "0123456789abcdef"[bytes[i] >> 4];
    expected_fsid += "0123456789abcdef"[bytes[i] & 15U];
  }
  CHECK(identity.fsid == expected_fsid);
  CHECK_FALSE(compute_mount_identity(-1).valid);
  auto const key = make_mount_policy_key(identity, "GPU-policy-0", 7);
  CHECK(key.namespace_inode == identity.namespace_inode);
  CHECK(key.mount_id == identity.mount_id);
  CHECK(key.fsid == expected_fsid);
  CHECK(key.device_uuid == "GPU-policy-0");
  CHECK(key.runtime_generation == 7);
  auto const equal_key = make_mount_policy_key(identity, "GPU-policy-0", 7);
  CHECK(key == equal_key);
  CHECK(mount_policy_key_hash{}(key) == mount_policy_key_hash{}(equal_key));
  CHECK_THROWS_AS(make_mount_policy_key(mount_identity{}, "GPU-policy-0", 7),
                  std::invalid_argument);
  CHECK_THROWS_AS(make_mount_policy_key(identity, "", 7), std::invalid_argument);
}

TEST_CASE("mount policy reactor accessor retains lazily cached identity", "[cufile][mount_policy]")
{
  temporary_file file;
  auto opened         = cufile_reactor::create_io_object(file.path, false, compat::on);
  auto const expected = compute_mount_identity(file.fd);
  auto const* cached  = &opened->mount_identity_info();
  REQUIRE(cached->valid);
  auto const key = opened->policy_key("GPU-cached", 123);
  CHECK(key == make_mount_policy_key(expected, "GPU-cached", 123));
  // The pathname need not remain resolvable; the passive accessor does no rediscovery.
  REQUIRE(::unlink(file.path.c_str()) == 0);
  CHECK(&opened->mount_identity_info() == cached);
  CHECK(opened->policy_key("GPU-cached", 123) == key);
  CHECK_FALSE(opened->cufile_registered());
  CHECK(opened->direct_fd() == -1);
}

TEST_CASE("mount policy identity discovery is lazy and concurrent", "[cufile][mount_policy]")
{
  temporary_file file;
  auto opened            = cufile_reactor::create_io_object(file.path, false, compat::on);
  auto const original    = compute_mount_identity(opened->buffered_fd());
  auto const replacement = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  REQUIRE(replacement >= 0);
  auto const expected = compute_mount_identity(replacement);
  auto const replaced = ::dup2(replacement, opened->buffered_fd());
  (void)::close(replacement);
  REQUIRE(replaced == opened->buffered_fd());
  REQUIRE(expected.valid);
  REQUIRE(expected.mount_id != original.mount_id);
  std::array<mount_identity const*, 16> addresses{};
  std::vector<std::thread> readers;
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    readers.emplace_back([&, i] { addresses[i] = &opened->mount_identity_info(); });
  }
  for (auto& reader : readers)
    reader.join();
  for (auto const* identity : addresses) {
    REQUIRE(identity == addresses.front());
    CHECK(identity->mount_id == expected.mount_id);
    CHECK(identity->fsid == expected.fsid);
  }
}

TEST_CASE("mount policy registry retains immutable state per full key", "[cufile][mount_policy]")
{
  mount_policy_registry registry;
  policy_fixture f;
  auto identity                     = synthetic_identity();
  auto key                          = make_mount_policy_key(identity, "GPU-registry-A", 101);
  auto config                       = policy_config(true);
  config.mount_overrides.front().qd = 7;
  auto first                        = registry.get_or_create(key, identity, config, f.executor);
  config.mount_overrides.front().qd = 9;
  CHECK(registry.get_or_create(key, identity, config, f.executor) == first);
  CHECK(first->base_profile().capacity.qd == 7);
  std::weak_ptr<mount_policy> retained = first;
  first.reset();
  REQUIRE_FALSE(retained.expired());
  CHECK(registry.get_or_create(key, identity, config, f.executor) == retained.lock());
  std::array<mount_policy_key, 5> separated{key, key, key, key, key};
  ++separated[0].namespace_inode;
  ++separated[1].mount_id;
  separated[2].fsid        = "different-fsid";
  separated[3].device_uuid = "GPU-registry-B";
  ++separated[4].runtime_generation;
  for (auto const& other_key : separated) {
    CAPTURE(other_key.namespace_inode,
            other_key.mount_id,
            other_key.fsid,
            other_key.device_uuid,
            other_key.runtime_generation);
    auto other = registry.get_or_create(other_key, identity, config, f.executor);
    CHECK(other != retained.lock());
    CHECK(other->key() == other_key);
    other->open_circuit();
    CHECK(retained.lock()->state() == mount_state::operator_qualified);
  }
}

TEST_CASE("mount policy override precedence is fsid prefix then filesystem",
          "[cufile][mount_policy]")
{
  auto identity = synthetic_identity();
  std::vector<mount_override> overrides{
    {mount_match::fs_type, "ext4", mount_override_state::legacy, {}, {}},
    {mount_match::mount_prefix, "/mnt", mount_override_state::known_compat, {}, {}},
    {mount_match::mount_prefix, "/mnt/policy", mount_override_state::operator_qualified, {}, {}},
    {mount_match::fsid, identity.fsid, mount_override_state::legacy, {}, {}}};
  CHECK(select_mount_override(overrides, identity) == &overrides[3]);
  overrides.pop_back();
  CHECK(select_mount_override(overrides, identity) == &overrides[2]);
  identity.mount_point = "/mnt/policy-other/data";
  CHECK(select_mount_override(overrides, identity) == &overrides[1]);
  identity.mount_point = "/other";
  CHECK(select_mount_override(overrides, identity) == &overrides[0]);
  identity.fs_type = "xfs";
  CHECK(select_mount_override(overrides, identity) == nullptr);
  overrides.push_back({mount_match::mount_prefix, "/", mount_override_state::known_compat, {}, {}});
  CHECK(select_mount_override(overrides, identity) == &overrides.back());
  identity.mount_point = "/mnt/policy";
  CHECK(select_mount_override(overrides, identity) == &overrides[2]);
  // Test the selector's tie contract directly; configuration validation rejects duplicates.
  overrides.push_back(overrides[2]);
  CHECK(select_mount_override(overrides, identity) == &overrides[2]);
}

TEST_CASE("mount policy overrides stay distinct from verified evidence", "[cufile][mount_policy]")
{
  auto const override_state = GENERATE(mount_override_state::operator_qualified,
                                       mount_override_state::known_compat,
                                       mount_override_state::legacy);
  auto config               = policy_config();
  config.mount_overrides.push_back({mount_match::fs_type, "ext4", override_state, 11, 2 * mib});
  policy_fixture f(config);
  CHECK(f.policy->base_profile().capacity.qd == 11);
  CHECK(f.policy->base_profile().capacity.preferred_op_bytes == 2 * mib);
  CHECK(f.policy->state() != mount_state::verified_direct);
  CHECK(f.policy->read_eligible() == (override_state == mount_override_state::operator_qualified));
  f.policy->ensure_probe({});
  CHECK(f.executor->ready.empty());
  CHECK(f.policy->gauges().probes == 0);
}

TEST_CASE("mount policy filesystem profiles respect geometry provenance", "[cufile][mount_policy]")
{
  for (auto const* type : {"ext4", "xfs"}) {
    auto const p = profile_for_filesystem(type);
    CHECK(p.filesystem == filesystem_class::local_nvme);
    CHECK(p.capacity.qd == 32);
    CHECK(p.capacity.preferred_op_bytes == 4 * mib);
  }
  for (auto block : {0UL, 4096UL, mib, 2 * mib, 4 * mib, 8 * mib, 16 * mib}) {
    auto const p = profile_for_filesystem("gpfs", block);
    CHECK(p.filesystem == filesystem_class::gpfs);
    CHECK(p.capacity.qd == 64);
    CHECK(p.capacity.preferred_op_bytes == std::clamp(block, mib, 8 * mib));
    CHECK(p.geometry_bytes == block);
    CHECK_FALSE(p.async_write_qualified);
  }
  struct profile_case {
    char const* type;
    filesystem_class cls;
    unsigned qd;
    std::size_t bytes;
  };
  for (auto const& [type, cls, qd, bytes] :
       {profile_case{"wekafs", filesystem_class::weka, 64, 2 * mib},
        profile_case{"nfs", filesystem_class::nfs_rdma, 64, mib},
        profile_case{"nfs4", filesystem_class::nfs_rdma, 64, mib},
        profile_case{"vastnfs", filesystem_class::nfs_rdma, 64, mib},
        profile_case{"lustre", filesystem_class::lustre, 64, mib},
        profile_case{"beegfs", filesystem_class::beegfs, 64, 2 * mib},
        profile_case{"compat", filesystem_class::compat, 4, mib},
        profile_case{"unrecognized", filesystem_class::unknown, 4, mib}}) {
    CAPTURE(type);
    auto const p = profile_for_filesystem(type, 8 * mib);
    CHECK(p.filesystem == cls);
    CHECK(p.capacity.qd == qd);
    CHECK(p.capacity.preferred_op_bytes == bytes);
    CHECK_FALSE(p.async_write_qualified);
    // statfs block size is not stripe/chunk evidence: use explicit conservative defaults.
    if (cls == filesystem_class::lustre || cls == filesystem_class::beegfs) {
      CHECK(p.geometry_bytes == bytes);
    }
  }
}

TEST_CASE("mount policy write qualification needs config reads and early evidence",
          "[cufile][mount_policy]")
{
  auto config         = policy_config(true);
  config.async_writes = true;
  policy_fixture reads_only(config);
  CHECK_FALSE(reads_only.policy->write_eligible());
  config.mount_overrides.front().allow_writes = true;
  policy_fixture f(config);
  CHECK(f.policy->write_eligible());  // Local profile is qualified; network profiles are not.
  f.policy->qualify_writes(false);
  CHECK_FALSE(f.policy->write_eligible());
  f.policy->qualify_writes(true);
  CHECK(f.policy->write_eligible());
  f.policy->observe_write_evidence(8 * mib, false);
  CHECK(f.policy->write_eligible());
  f.policy->observe_write_evidence(8 * mib, false);
  CHECK(f.policy->write_eligible());  // Standing write-enabled override is evidence-exempt.
  f.policy->qualify_writes(true);
  CHECK(f.policy->write_eligible());

  policy_fixture direct(config);
  direct.policy->qualify_writes(true);
  direct.policy->observe_write_evidence(8 * mib, true);
  direct.policy->observe_write_evidence(8 * mib, false);
  CHECK(direct.policy->write_eligible());
  direct.policy->qualify_writes(false);
  CHECK_FALSE(direct.policy->write_eligible());
  policy_fixture disabled;
  disabled.policy->qualify_writes(true);
  CHECK_FALSE(disabled.policy->write_eligible());
  config.mount_overrides.clear();
  policy_fixture unknown(config);
  unknown.policy->qualify_writes(true);
  CHECK_FALSE(unknown.policy->write_eligible());
  auto network_identity                               = synthetic_identity();
  network_identity.fs_type                            = "gpfs";
  auto network_config                                 = policy_config(true);
  network_config.async_writes                         = true;
  network_config.mount_overrides.front().allow_writes = false;
  policy_fixture network(network_config, network_identity);
  CHECK_FALSE(network.policy->base_profile().async_write_qualified);
  CHECK_FALSE(network.policy->write_eligible());
  network.policy->qualify_writes(true);
  CHECK_FALSE(network.policy->write_eligible());  // Cannot override a reads-only operator entry.
  auto reads_disabled_config        = policy_config(true);
  reads_disabled_config.async_reads = false;
  policy_fixture reads_disabled(reads_disabled_config);
  CHECK_FALSE(reads_disabled.policy->read_eligible());
}

TEST_CASE("mount policy facade and reaper never wait for the control mutex",
          "[cufile][mount_policy]")
{
  auto clock    = std::make_shared<fake_clock>();
  auto executor = std::make_shared<fake_executor>(clock);
  std::promise<void> locked, release;
  auto gate = release.get_future().share();
  std::atomic<bool> block{false};
  auto identity = synthetic_identity();
  auto policy =
    std::make_shared<mount_policy>(make_mount_policy_key(identity, "policy-test-nonblocking", 91),
                                   identity,
                                   policy_config(true),
                                   executor,
                                   [&, clock] {
                                     if (block.exchange(false)) {
                                       locked.set_value();
                                       gate.wait();
                                     }
                                     return clock->now;
                                   });
  policy->set_effective({32, 4 * mib, 40});
  executor->accept = false;
  block            = true;
  auto control     = std::async(std::launch::async, [&] { policy->ensure_probe({}); });
  locked.get_future().wait();
  auto hot_path       = std::async(std::launch::async, [&] {
    auto const readable = policy->read_eligible();
    (void)policy->write_eligible();
    auto const base      = policy->base_profile();
    auto const effective = policy->effective();
    policy->qualify_writes(true);
    policy->observe_write_evidence(4096, true);
    for (unsigned i = 0; i < 4096; ++i)
      policy->observe_submit(10us);
    policy->open_circuit();
    policy->begin_cooldown();  // A contended drain notification must not block either.
    return readable && base.capacity.qd == 32 && effective.qd == 32;
  });
  bool const returned = hot_path.wait_for(1s) == std::future_status::ready;
  release.set_value();
  control.get();
  CHECK(returned);
  CHECK(hot_path.get());
  CHECK(policy->state() == mount_state::circuit_open);
  CHECK(policy->gauges().healthy_p99 == 10us);
  policy->poll_timers();  // Retry queue was full: the drain proof still must not be lost.
  CHECK(policy->state() == mount_state::cooldown);
}

TEST_CASE("mount policy effective snapshot is coherent without waiting for control updates",
          "[cufile][mount_policy]")
{
  policy_fixture f;
  std::atomic<bool> done{};
  std::thread writer([&] {
    for (unsigned i = 0; i < 10000; ++i) {
      f.policy->set_effective({8, mib, 10});
      f.policy->set_effective({32, 4 * mib, 40});
    }
    done = true;
  });
  bool coherent = true;
  do {
    auto const p = f.policy->effective();
    coherent     = coherent && ((p.qd == 0 && p.op_bytes == 0 && p.lanes == 0) ||
                            (p.qd == 8 && p.op_bytes == mib && p.lanes == 10) ||
                            (p.qd == 32 && p.op_bytes == 4 * mib && p.lanes == 40));
  } while (!done);
  writer.join();
  CHECK(coherent);
}

TEST_CASE("mount policy healthy histogram rounds conservatively and ignores negative samples",
          "[cufile][mount_policy]")
{
  policy_fixture f;
  f.policy->observe_submit(-1ns);
  CHECK(f.policy->gauges().healthy_p99 == 0ns);
  for (unsigned i = 0; i < 1024; ++i)
    f.policy->observe_submit(20001ns);
  CHECK(f.policy->gauges().healthy_p99 == 21us);
  CHECK(f.policy->state() == mount_state::operator_qualified);
}

TEST_CASE("mount policy breaker counts only slow samples in the last eight",
          "[cufile][mount_policy]")
{
  policy_fixture f;
  for (unsigned i = 0; i < 1024; ++i) {
    f.policy->observe_submit(10us);
  }
  REQUIRE(f.policy->gauges().healthy_p99 == 10us);
  f.policy->observe_submit(101us);
  for (unsigned i = 0; i < 6; ++i) {
    f.policy->observe_submit(10us);
  }
  f.policy->observe_submit(101us);
  CHECK(f.policy->state() == mount_state::operator_qualified);
  f.policy->observe_submit(100us);  // Equal to the floor is not slow; evicts the first slow.
  f.policy->observe_submit(101us);
  CHECK(f.policy->state() == mount_state::operator_qualified);
  f.policy->observe_submit(101us);
  CHECK(f.policy->state() == mount_state::circuit_open);
  CHECK_FALSE(f.policy->read_eligible());
  CHECK(f.policy->gauges().demotions == 1);
  f.policy->open_circuit();
  CHECK(f.policy->gauges().demotions == 1);
  CHECK(f.policy->gauges().healthy_p99 == 10us);
}

TEST_CASE("mount policy p99 rolls over 1024 non-slow samples only", "[cufile][mount_policy]")
{
  policy_fixture f;
  for (unsigned i = 0; i < 1024; ++i) {
    f.policy->observe_submit(50us);
  }
  CHECK(f.policy->gauges().healthy_p99 == 50us);
  for (unsigned i = 0; i < 8; ++i) {
    f.policy->observe_submit(201us);  // Above 4 * p99, never incorporated into the baseline.
    for (unsigned j = 0; j < 8; ++j) {
      f.policy->observe_submit(50us);
    }
  }
  CHECK(f.policy->state() == mount_state::operator_qualified);
  CHECK(f.policy->gauges().healthy_p99 == 50us);
  for (unsigned i = 0; i < 1024; ++i) {
    f.policy->observe_submit(20us);
  }
  CHECK(f.policy->gauges().healthy_p99 == 20us);
  for (unsigned i = 0; i < 8; ++i) {
    f.policy->observe_submit(100us);
  }
  CHECK(f.policy->state() == mount_state::operator_qualified);
  f.policy->observe_submit(101us);
  f.policy->observe_submit(101us);
  f.policy->observe_submit(101us);
  CHECK(f.policy->state() == mount_state::circuit_open);
  CHECK(f.policy->gauges().healthy_p99 == 20us);
}

TEST_CASE("mount policy watchdog opens at the configured stuck boundary", "[cufile][mount_policy]")
{
  policy_fixture f;
  f.policy->observe_stuck_call(2ms - 1ns);
  CHECK(f.policy->state() == mount_state::operator_qualified);
  f.policy->observe_stuck_call(2ms);
  CHECK(f.policy->state() == mount_state::circuit_open);
  CHECK(f.policy->gauges().demotions == 1);
  f.policy->observe_stuck_call(1h);
  CHECK(f.policy->gauges().demotions == 1);
}

TEST_CASE("mount policy cooldown waits for drain and doubles to its cap", "[cufile][mount_policy]")
{
  policy_fixture f;
  for (auto const delay : {30s, 60s, 120s, 240s, 480s, 960s, 960s}) {
    CAPTURE(delay.count());
    f.policy->open_circuit();
    f.policy->poll_timers();
    CHECK(f.policy->state() == mount_state::circuit_open);
    auto const cooldowns = f.policy->gauges().cooldowns;
    f.policy->begin_cooldown();
    REQUIRE(f.policy->state() == mount_state::cooldown);
    CHECK(f.policy->gauges().cooldown == delay);
    f.policy->begin_cooldown();
    CHECK(f.policy->gauges().cooldowns == cooldowns + 1);
    f.clock->now += delay - 1ns;
    f.policy->poll_timers();
    CHECK(f.policy->state() == mount_state::cooldown);
    f.clock->now += 1ns;
    f.executor->run_ready();
    f.policy->poll_timers();
    CHECK(f.policy->state() == mount_state::operator_qualified);
    CHECK(f.policy->read_eligible());
  }
}

TEST_CASE("mount policy healthy override window resets cooldown backoff", "[cufile][mount_policy]")
{
  policy_fixture f;
  f.policy->observe_stuck_call(5ms);
  CHECK(f.policy->gauges().last_trigger == circuit_trigger::stuck);
  CHECK(f.policy->gauges().trigger_elapsed == 5ms);
  f.policy->begin_cooldown();
  f.clock->now += 30s;
  f.policy->poll_timers();
  for (unsigned i = 0; i < 1024; ++i)
    f.policy->observe_submit(10us);
  f.policy->open_circuit();
  f.policy->begin_cooldown();
  CHECK(f.policy->gauges().cooldown == 30s);
  f.clock->now += 30s;
  f.policy->poll_timers();
  f.policy->open_circuit();
  f.policy->begin_cooldown();
  CHECK(f.policy->gauges().cooldown == 60s);  // No healthy window: preserve escalation.
}

TEST_CASE("mount policy expired cooldown cannot bypass the drain gate", "[cufile][mount_policy]")
{
  policy_fixture f;
  f.policy->open_circuit();
  f.clock->now += 1h;
  f.policy->poll_timers();
  CHECK(f.policy->state() == mount_state::circuit_open);
  f.policy->begin_cooldown();
  f.policy->poll_timers();
  CHECK(f.policy->state() == mount_state::cooldown);
  f.clock->now += 30s;
  f.policy->poll_timers();
  CHECK(f.policy->state() == mount_state::operator_qualified);
}

TEST_CASE("mount policy timers can be polled when the control queue is full",
          "[cufile][mount_policy]")
{
  policy_fixture f;
  f.executor->accept = false;
  f.policy->open_circuit();
  f.policy->begin_cooldown();
  REQUIRE(f.policy->state() == mount_state::cooldown);
  f.clock->now += 30s;
  f.policy->poll_timers();
  CHECK(f.policy->state() == mount_state::operator_qualified);
}

TEST_CASE("mount policy contended one-shot cooldown timer schedules a retry",
          "[cufile][mount_policy]")
{
  auto clock    = std::make_shared<fake_clock>();
  auto executor = std::make_shared<fake_executor>(clock);
  std::promise<void> locked, release;
  auto gate = release.get_future().share();
  std::atomic<bool> block{false};
  auto identity = synthetic_identity();
  auto policy   = std::make_shared<mount_policy>(
    make_mount_policy_key(identity, "policy-test-timer-contention", 91),
    identity,
    policy_config(true),
    executor,
    [&, clock] {
      if (block.exchange(false)) {
        locked.set_value();
        gate.wait();
      }
      return clock->now;
    });
  policy->open_circuit();
  block = true;
  std::thread control([&] { policy->begin_cooldown(); });
  locked.get_future().wait();
  policy->poll_timers();  // Loses the control lock and must re-arm, not discard, the timer.
  release.set_value();
  control.join();
  REQUIRE(executor->timers.size() == 2);
  auto original = std::find_if(executor->timers.begin(),
                               executor->timers.end(),
                               [&](auto const& entry) { return entry.first == clock->now + 30s; });
  REQUIRE(original != executor->timers.end());
  executor->timers.erase(original);  // Only the contention retry remains.
  clock->now += 30s;
  executor->run_ready();
  CHECK(policy->state() == mount_state::operator_qualified);
}

TEST_CASE("mount policy executor bounds pending work and abandons future timers",
          "[cufile][mount_policy]")
{
  std::atomic<bool> timer_ran{};
  bool accepted{}, overflow{}, timer_accepted{};
  std::atomic<bool> ready_ran{};
  {
    std::promise<void> started, release;
    auto gate = release.get_future().share();
    single_thread_control_executor executor(1);
    while (!executor.submit([&] {
      started.set_value();
      gate.wait();
    })) {
      std::this_thread::yield();  // Non-blocking admission may lose the worker's mutex race.
    }
    started.get_future().wait();
    accepted       = executor.submit([&] { ready_ran = true; });
    overflow       = executor.submit([] {});
    timer_accepted = executor.schedule(policy_clock::now() + 1h, [&] { timer_ran = true; });
    release.set_value();
  }
  CHECK(accepted);
  CHECK(ready_ran);
  CHECK_FALSE(overflow);
  CHECK_FALSE(timer_accepted);
  {
    single_thread_control_executor executor(1);
    while (!executor.schedule(policy_clock::now() + 1h, [&] { timer_ran = true; })) {
      std::this_thread::yield();
    }
  }
  CHECK_FALSE(timer_ran);
}

TEST_CASE("mount policy probe scheduling rejects full queues without sticking probing",
          "[cufile][mount_policy]")
{
  policy_fixture f(policy_config());
  f.executor->accept = false;
  f.policy->ensure_probe({});
  CHECK(f.policy->state() == mount_state::unknown);
  CHECK(f.executor->ready.empty());
  f.executor->accept = true;
  f.policy->ensure_probe({});
  REQUIRE(f.executor->ready.size() == 1);
  f.policy->ensure_probe({});
  CHECK(f.executor->ready.size() == 1);
  f.executor->run_ready();
  CHECK(f.policy->state() != mount_state::probing);
}

TEST_CASE("mount policy dropped probe task releases state file and device budget",
          "[cufile][mount_policy]")
{
  policy_fixture first(policy_config());
  auto identity = synthetic_identity();
  ++identity.mount_id;
  policy_fixture second(policy_config(), identity);
  temporary_file file;
  std::shared_ptr<cufile_io_object const> opened =
    cufile_reactor::create_io_object(file.path, false, compat::on);
  std::weak_ptr<cufile_io_object const> weak_file = opened;
  mount_probe_context context;
  context.file = std::move(opened);
  first.policy->ensure_probe(std::move(context));
  REQUIRE(first.policy->state() == mount_state::probing);
  CHECK_FALSE(weak_file.expired());
  first.executor->ready.clear();  // Accepted but never run.
  CHECK(first.policy->state() == mount_state::unknown);
  CHECK(weak_file.expired());
  CHECK(first.policy->gauges().probes == 0);
  second.policy->ensure_probe({});
  CHECK(second.policy->state() == mount_state::probing);
  second.executor->ready.clear();
  first.policy->ensure_probe({});
  CHECK(first.policy->state() == mount_state::probing);
  first.executor->ready.clear();
}

TEST_CASE("mount policy watchdog cannot open inactive or probing circuits",
          "[cufile][mount_policy]")
{
  policy_fixture f(policy_config());
  f.policy->open_circuit();
  CHECK(f.policy->state() == mount_state::unknown);
  f.policy->ensure_probe({});
  REQUIRE(f.policy->state() == mount_state::probing);
  f.policy->observe_stuck_call(1h);
  CHECK(f.policy->state() == mount_state::probing);
  f.executor->run_ready();
  CHECK(f.policy->state() == mount_state::unknown);
  CHECK(f.policy->gauges().demotions == 0);
}

TEST_CASE("mount policy probe budgets serialize only a device", "[cufile][mount_policy]")
{
  policy_fixture first(policy_config());
  auto second_identity = synthetic_identity();
  ++second_identity.mount_id;
  policy_fixture second(policy_config(), second_identity);
  auto other_device = std::make_shared<mount_policy>(
    make_mount_policy_key(second_identity, "GPU-other-probe-budget", 91),
    second_identity,
    policy_config(),
    second.executor,
    [clock = second.clock] { return clock->now; });
  first.policy->ensure_probe({});
  REQUIRE(first.policy->state() == mount_state::probing);
  second.policy->ensure_probe({});
  CHECK(second.policy->state() == mount_state::unknown);
  CHECK(second.executor->ready.empty());
  other_device->ensure_probe({});
  CHECK(other_device->state() == mount_state::probing);
  first.executor->run_ready();
  second.executor->run_ready();
  second.policy->ensure_probe({});
  CHECK(second.policy->state() == mount_state::probing);
  second.executor->run_ready();
  CHECK(second.policy->state() == mount_state::unknown);
  second.policy->ensure_probe({});
  CHECK(second.executor->ready.empty());
  second.clock->now += 30s;
  second.policy->ensure_probe({});
  CHECK(second.policy->state() == mount_state::probing);
  second.executor->run_ready();
}

TEST_CASE("mount policy quota-free simulator probe requires attributable direct evidence",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  auto const caller = std::this_thread::get_id();
  p.f.policy->ensure_probe(p.context());
  p.f.policy->ensure_probe(p.context());
  REQUIRE(p.f.policy->state() == mount_state::probing);
  REQUIRE(p.f.executor->ready.size() == 1);
  CHECK(p.transport->snapshot().read_calls == 0);
  p.run();
  REQUIRE(p.f.policy->state() == mount_state::verified_direct);
  CHECK(p.f.policy->read_eligible());
  CHECK_FALSE(p.f.policy->write_eligible());
  auto const gauges = p.f.policy->gauges();
  CHECK(gauges.probes == 1);
  CHECK(gauges.probes_passed == 1);
  CHECK(gauges.last_probe == probe_reason::direct_evidence);
  CHECK(gauges.median_completion > 0ns);
  REQUIRE(p.transport->snapshot().bytes_read >= 2 * mib);
  CHECK(p.evidence->samples >= 2);
  for (auto const& submission : p.transport->submissions()) {
    CHECK(submission.thread != caller);
    CHECK(submission.file_offset % 4096 == 0);
    CHECK(submission.size % 4096 == 0);
    CHECK(submission.size <= mib);
  }
  CHECK(p.arena->gauges().live_leases == 0);
  p.f.policy->ensure_probe(p.context());
  CHECK(p.f.executor->ready.empty());
}

TEST_CASE("mount policy file-level probe rejection can retry a later qualifying file",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  auto bad_file      = p.context();
  bad_file.simulated = false;  // First descriptor has no registered cuFile handle/direct fd.
  REQUIRE_FALSE(bad_file.file->cufile_registered());
  p.f.policy->ensure_probe(bad_file);
  p.run();
  CHECK(p.f.policy->state() == mount_state::unknown);
  CHECK(p.f.policy->gauges().last_probe == probe_reason::unsupported_file);
  CHECK(p.transport->snapshot().read_calls == 0);
  p.f.policy->ensure_probe(p.context());
  CHECK(p.f.executor->ready.empty());
  p.f.clock->now += 30s;
  // Simulator represents the next descriptor having passed production registration gates.
  p.f.policy->ensure_probe(p.context());
  p.run();
  CHECK(p.f.policy->state() == mount_state::verified_direct);
  CHECK(p.f.policy->gauges().probes == 2);
}

TEST_CASE("mount policy no evidence or global counters cannot verify a probe",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  auto const failure = GENERATE(0, 1, 2, 3);
  CAPTURE(failure);
  probe_fixture p;
  p.evidence = std::make_shared<simulator_evidence>(p.transport, failure != 0, failure != 1);
  p.evidence->unrelated_bytes  = failure == 2 ? 4096 : 0;
  p.evidence->constant_counter = failure == 3;
  p.f.policy->ensure_probe(p.context());
  p.run();
  CHECK(p.transport->snapshot().bytes_read >= 2 * mib);
  CHECK(p.f.policy->state() == mount_state::unknown);
  CHECK_FALSE(p.f.policy->read_eligible());
  CHECK(p.f.policy->gauges().last_probe == probe_reason::no_evidence);
  p.f.policy->ensure_probe(p.context());
  CHECK(p.f.executor->ready.empty());  // Per-mount retry interval applies to failed probes too.
}

TEST_CASE("mount policy failed and short simulator probes never qualify",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  auto context  = p.context();
  auto expected = probe_reason::completion_error;
  temporary_file short_file(4096);
  SECTION("synchronous submit failure")
  {
    transport_simulator::fault rule;
    rule.submit_status = {-5, 0};
    p.transport->add_fault(rule);
    expected = probe_reason::submit_error;
  }
  SECTION("negative completion")
  {
    transport_simulator::fault rule;
    rule.completion_result = -5;
    p.transport->add_fault(rule);
  }
  SECTION("short completion")
  {
    transport_simulator::fault rule;
    rule.completion_result = 4096;
    p.transport->add_fault(rule);
  }
  SECTION("failed completion event record")
  {
    transport_simulator::fault rule;
    rule.record_error = cudaErrorUnknown;
    p.transport->add_fault(rule);
  }
  SECTION("file holds a granule but not the requested qualification interval")
  {
    context.file = cufile_reactor::create_io_object(short_file.path, false, compat::on);
    expected     = probe_reason::short_file;
  }
  p.f.policy->ensure_probe(context);
  p.run();
  CHECK(p.f.policy->state() == mount_state::unknown);
  CHECK_FALSE(p.f.policy->read_eligible());
  CHECK(p.f.policy->gauges().last_probe == expected);
  CHECK(p.f.policy->gauges().probes_failed == 1);
  CHECK(p.arena->gauges().live_leases == 0);
}

TEST_CASE("mount policy slow submit on GDS is not evidence of compat",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p(true, true, 1000);
  transport_simulator::fault rule;
  rule.submit_latency = 10ms;
  p.transport->add_fault(rule);
  auto context = p.context();
  context.evidence.reset();
  p.f.policy->ensure_probe(context);
  p.run();
  CHECK(p.f.policy->state() == mount_state::unknown);
  CHECK(p.f.policy->gauges().last_probe == probe_reason::slow_submit);
  CHECK(p.f.policy->gauges().median_submit >= 10ms);
}

TEST_CASE("mount policy retained probe context is reused after cooldown",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  p.f.policy->ensure_probe(p.context());
  p.run();
  REQUIRE(p.f.policy->state() == mount_state::verified_direct);
  p.f.policy->open_circuit();
  p.f.policy->begin_cooldown();
  p.f.clock->now += 30s;
  p.run();
  CHECK(p.f.policy->state() == mount_state::verified_direct);
  CHECK(p.f.policy->gauges().probes == 2);
  CHECK(p.f.policy->gauges().cooldown == 0s);
  CHECK(p.transport->snapshot().bytes_read >= 4 * mib);
  CHECK(p.arena->gauges().live_leases == 0);
}

TEST_CASE("mount policy failed reprobes double cooldown then successful reprobe resets it",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  p.f.policy->ensure_probe(p.context());
  p.run();
  REQUIRE(p.f.policy->state() == mount_state::verified_direct);
  p.evidence->constant_counter = true;
  p.f.policy->open_circuit();
  p.f.policy->begin_cooldown();
  for (auto const delay : {30s, 60s, 120s, 240s, 480s, 960s, 960s}) {
    CAPTURE(delay.count());
    REQUIRE(p.f.policy->state() == mount_state::cooldown);
    CHECK(p.f.policy->gauges().cooldown == delay);
    p.f.clock->now += delay;
    p.run();
    CHECK(p.f.policy->state() == mount_state::cooldown);
    CHECK(p.f.policy->gauges().cooldown == std::min(2 * delay, 960s));
    CHECK(p.f.policy->gauges().last_probe == probe_reason::no_evidence);
    CHECK(p.arena->gauges().live_leases == 0);
  }
  CHECK(p.f.policy->gauges().probes_failed == 7);
  p.evidence->constant_counter = false;
  p.f.clock->now += 960s;
  p.run();
  CHECK(p.f.policy->state() == mount_state::verified_direct);
  CHECK(p.f.policy->gauges().cooldown == 0s);
  CHECK(p.f.policy->gauges().probes_passed == 2);
}

TEST_CASE("mount policy completed probes do not keep file descriptors alive indefinitely",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p;
  p.f.policy->ensure_probe(p.context());
  p.run();
  REQUIRE(p.f.policy->state() == mount_state::verified_direct);
  std::weak_ptr<cufile_io_object const> file = p.opened;
  p.opened.reset();
  CHECK(file.expired());
  auto const read_calls = p.transport->snapshot().read_calls;
  p.f.policy->open_circuit();
  p.f.policy->begin_cooldown();
  p.f.clock->now += 30s;
  p.run();
  CHECK(p.f.policy->state() == mount_state::unknown);
  CHECK(p.transport->snapshot().read_calls == read_calls);
  CHECK(p.arena->gauges().live_leases == 0);
}

TEST_CASE("mount policy test-only evidence bypass is not reported verified",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p(true, false);
  auto context = p.context();
  context.evidence.reset();
  context.process_has_gds = false;  // This test gate also permits compat-only CI hosts.
  p.f.policy->ensure_probe(context);
  p.run();
  CHECK(p.f.policy->state() == mount_state::operator_qualified);
  CHECK(p.f.policy->read_eligible());
  CHECK(p.f.policy->gauges().last_probe == probe_reason::test_override);
}

TEST_CASE("mount policy unregistered simulator tier is separately degraded",
          "[cufile][mount_policy][simulator]")
{
  require_cuda();
  probe_fixture p(false);
  p.f.policy->ensure_probe(p.context());
  p.run();
  CHECK(p.f.policy->state() == mount_state::degraded_async);
  CHECK(p.f.policy->read_eligible());
  CHECK_FALSE(p.f.policy->write_eligible());
  CHECK(p.f.policy->gauges().last_probe == probe_reason::unregistered);
  CHECK(p.arena->gauges().registered_bytes == 0);
  CHECK(p.transport->snapshot().bytes_read >= 2 * mib);
  CHECK(p.arena->gauges().live_leases == 0);
}

TEST_CASE("mount policy compat short and unsupported files avoid transport I/O",
          "[cufile][mount_policy]")
{
  temporary_file short_file(4095);
  std::shared_ptr<cufile_io_object const> opened{
    cufile_reactor::create_io_object(short_file.path, false, compat::on)};
  device_arena_hooks hooks;
  hooks.allocate          = [](int, std::size_t) -> void* { return nullptr; };
  hooks.deallocate        = [](int, void*) { return true; };
  hooks.register_buffer   = [](int, void*, std::size_t) { return false; };
  hooks.deregister_buffer = [](int, void*) { return true; };
  mount_probe_context base;
  device_arena arena(0, std::move(hooks));
  base.file      = opened;
  base.arena     = &arena;
  base.transport = std::make_shared<transport_simulator>();
  SECTION("process compat is explicit even if async submission might be fast")
  {
    policy_fixture f(policy_config());
    auto context            = base;
    context.simulated       = true;
    context.process_has_gds = false;
    f.policy->ensure_probe(context);
    f.executor->run_ready();
    CHECK(f.policy->state() == mount_state::known_compat);
    CHECK(f.policy->gauges().last_probe == probe_reason::compat_platform);
  }
  SECTION("short file retains unknown")
  {
    policy_fixture f(policy_config());
    auto context            = base;
    context.simulated       = true;
    context.process_has_gds = true;
    f.policy->ensure_probe(context);
    f.executor->run_ready();
    CHECK(f.policy->state() == mount_state::unknown);
    CHECK(f.policy->gauges().last_probe == probe_reason::short_file);
  }
  SECTION("production descriptor gates are not bypassed by fake GDS availability")
  {
    temporary_file regular_file;
    policy_fixture f(policy_config());
    auto context = base;
    context.file = cufile_reactor::create_io_object(regular_file.path, false, compat::on);
    context.process_has_gds = true;
    f.policy->ensure_probe(context);
    f.executor->run_ready();
    CHECK(f.policy->state() == mount_state::unknown);
    CHECK(f.policy->gauges().last_probe == probe_reason::unsupported_file);
    f.policy->ensure_probe(context);
    CHECK(f.executor->ready.empty());  // File-level failures use the normal retry interval.
  }
}

TEST_CASE("mount policy production evidence explicitly stays unavailable until qualification",
          "[cufile][mount_policy]")
{
  // These are format sketches, not captured GDS-host fixtures. Neither real-looking
  // MiB text nor the old invented byte format may claim qualification support.
  auto const text = GENERATE(std::string{"Reads : n=1 ok=1 readMiB=2\n"
                                         "Writes : n=1 ok=1 writeMiB=1\n"},
                             std::string{"read_bytes: 4096\nwrite_bytes: 8192\n"});
  temporary_file file(0);
  {
    std::ofstream out(file.path);
    out << text;
    REQUIRE(out.good());
  }
  for (bool isolated : {false, true}) {
    auto const evidence = nvidia_fs_evidence(file.path, isolated).sample();
    CHECK_FALSE(evidence.available);
    CHECK_FALSE(evidence.attributable);
    CHECK(evidence.read_bytes == 0);
    CHECK(evidence.write_bytes == 0);
  }
  CHECK_FALSE(nvidia_fs_evidence{}.sample().available);
  CHECK_FALSE(cufile_stats_evidence{}.sample().available);
}
