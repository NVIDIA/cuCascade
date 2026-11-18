/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "engine_test_utils.hpp"
#include "utils/allocation_counter.hpp"

#include <cucascade/io/cufile/cufile_reactor.hpp>

#include <fcntl.h>
#include <unistd.h>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

TEST_CASE("allocation counter covers nothrow and array allocation families", "[io][allocation]")
{
  std::size_t count{};
  void* scalar{};
  void* array{};
  void* failed{};
  {
    cucascade::test::scoped_allocation_counter counter;
    scalar = ::operator new(32, std::nothrow);
    array  = ::operator new[](32, std::nothrow);
    counter.fail_at(3);
    failed = ::operator new(32, std::nothrow);
    count  = counter.count();
  }
  CHECK(count == 3);
  CHECK(scalar != nullptr);
  CHECK(array != nullptr);
  CHECK(failed == nullptr);
  ::operator delete(scalar, std::size_t{32});
  ::operator delete[](array);
}

TEST_CASE("engine polling deadlines survive unrelated notifications with a fake clock",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  CHECK(cucascade::io::cufile::detail::io_poll_cap_us(0ns, 100) == 100);
  CHECK(cucascade::io::cufile::detail::io_poll_cap_us(4ms, 100) == 500);
  CHECK(cucascade::io::cufile::detail::io_poll_cap_us(16ms, 100) == 1000);
  fixture f(mib, configuration(1));
  std::atomic<std::int64_t> ticks{};
  std::atomic<unsigned> queries{};
  std::atomic<bool> release{};
  auto group        = std::make_shared<grouped_coordinator>(mib, 1);
  auto future       = group->get_future();
  auto s            = f.slice(group, {0, mib});
  auto fence        = s.fence->event;
  auto const origin = policy_clock::now();
  f.start([&](auto& o) {
    o.now         = [&] { return origin + std::chrono::microseconds(ticks.load()); };
    o.query_event = [&](cudaEvent_t event) {
      if (event == fence) return cudaSuccess;
      ++queries;
      return release.load() ? cudaEventQuery(event) : cudaErrorNotReady;
    };
  });
  f.engine->hand_off(std::move(s));
  REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
  // The submit publication is processed without advancing the fake clock.
  for (int i = 0; i < 20; ++i) {
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  CHECK(queries.load() == 0);
  ticks = 10;
  f.engine->notify();
  REQUIRE(until([&] { return queries.load() == 1; }));
  ticks = 34;  // Not-ready deadline is 10 + 25 us, not the next ring wake.
  for (int i = 0; i < 20; ++i) {
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  CHECK(queries.load() == 1);
  ticks = 35;
  f.engine->notify();
  REQUIRE(until([&] { return queries.load() == 2; }));
  release = true;
  for (int i = 0; i < 100 && !future.is_ready(); ++i) {
    ticks.fetch_add(1000);
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  REQUIRE(std::move(future).get(5s) == mib);
  for (int i = 0; i < 100 && !f.engine->drain(0ms); ++i) {
    ticks.fetch_add(1000);
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  REQUIRE(f.engine->drain(5s));
}

TEST_CASE("engine fence hint queries obey fake-clock deadlines even at epoch zero",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  std::atomic<std::int64_t> ticks{};
  std::atomic<unsigned> hints{};
  std::atomic<bool> ready{};
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  auto s      = f.slice(group, {0, mib});
  auto fence  = s.fence->event;
  f.start([&](auto& opts) {
    opts.now = [&] { return policy_clock::time_point{} + std::chrono::microseconds(ticks.load()); };
    opts.query_event = [&](cudaEvent_t event) {
      if (event != fence) return cudaEventQuery(event);
      ++hints;
      return ready.load() ? cudaSuccess : cudaErrorNotReady;
    };
  });
  f.engine->hand_off(std::move(s));
  REQUIRE(until([&] { return hints.load() == 1; }));
  ticks = 99;
  for (int i = 0; i < 20; ++i) {
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  CHECK(hints.load() == 1);
  ready = true;
  for (int i = 0; i < 100 && !f.engine->drain(0ms); ++i) {
    ticks.fetch_add(1000);
    f.engine->notify();
    std::this_thread::sleep_for(100us);
  }
  REQUIRE(std::move(future).get(5s) == mib);
  REQUIRE(f.engine->drain(5s));
}

TEST_CASE("engine transient completed reads back off and bound attempts",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const bounded = GENERATE(false, true);
  auto const error   = GENERATE(ESTALE, EAGAIN, ETIMEDOUT, ENOTCONN);
  auto const submit  = GENERATE(false, true);
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  if (!bounded) fault.index = 0;
  if (submit)
    fault.submit_status.error = -error;
  else
    fault.completion_result = -error;
  f.transport->add_fault(fault);
  f.start();
  auto group       = std::make_shared<grouped_coordinator>(mib, 1);
  auto future      = group->get_future();
  auto const start = policy_clock::now();
  f.engine->hand_off(f.slice(group, {0, mib}));
  if (bounded)
    CHECK_THROWS(std::move(future).get(5s));
  else {
    CHECK(std::move(future).get(5s) == mib);
    CHECK(f.contents() == f.image);
  }
  REQUIRE(f.engine->drain(5s));
  CHECK(policy_clock::now() - start >= (bounded ? 30ms : 10ms));
  CHECK(f.transport->snapshot().read_calls == (bounded ? 3 : 2));
  CHECK(f.policy->read_eligible());
}

TEST_CASE("engine demotes only the failing file and preserves mount peers",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = -EOPNOTSUPP;
  f.transport->add_fault(fault);
  f.start();
  auto state = std::make_shared<file_async_policy>();
  auto read  = [&](std::shared_ptr<file_async_policy> file_policy) {
    auto group    = std::make_shared<grouped_coordinator>(mib, 1);
    auto future   = group->get_future();
    auto s        = f.slice(group, {0, mib});
    s.file_policy = std::move(file_policy);
    f.engine->hand_off(std::move(s));
    CHECK(std::move(future).get(5s) == mib);
    REQUIRE(f.engine->drain(5s));
  };
  read(state);
  CHECK(state->read_demoted.load());
  CHECK(f.policy->read_eligible());
  read(state);
  CHECK(f.transport->snapshot().read_calls == 1);
  read(std::make_shared<file_async_policy>());
  CHECK(f.transport->snapshot().read_calls == 2);
}

TEST_CASE("recovered CUDA lanes are replaced off the reaper before reuse",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  auto before = f.lanes->try_lease();
  REQUIRE(before);
  auto const old_stream = (*before)->stream();
  auto const old_event  = (*before)->copy_event();
  before.reset();
  auto const record = GENERATE(false, true);
  if (record)
    f.transport->fail_record(1, cudaErrorNotPermitted);
  else
    f.transport->fail_copy(0, cudaErrorNotPermitted);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  CHECK_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  auto after = f.lanes->try_lease();
  REQUIRE(after);
  CHECK((*after)->stream() != old_stream);
  CHECK((*after)->copy_event() != old_event);
  CHECK(f.lanes->quarantined() == 0);
}

TEST_CASE("file granules and geometry use statx with validated filesystem fallbacks",
          "[cufile][engine][hardening]")
{
  file_geometry_hooks hooks;
  hooks.page_size = [] { return 4096L; };
  hooks.statfs    = [](int, struct statfs* fs) {
    fs->f_bsize = 8L << 20;
    fs->f_type  = 0x47504653;
    return 0;
  };
  hooks.dio_alignment = [](int) { return 16384UL; };
  auto geometry       = discover_file_geometry(-1, "gpfs", hooks);
  CHECK(geometry.granule == 16384);
  CHECK(geometry.geometry_bytes == 8 * mib);
  mount_identity identity;
  identity.fs_type  = "gpfs";
  identity.geometry = geometry;
  mount_policy policy(mount_policy_key{1, 42, "", "geometry-test", 1}, identity, configuration(1));
  CHECK(policy.base_profile().granule == 16384);
  CHECK(policy.base_profile().geometry_bytes == 8 * mib);
  hooks.dio_alignment = [](int) { return std::nullopt; };
  geometry            = discover_file_geometry(-1, "gpfs", hooks);
  CHECK(geometry.granule == 4096);
  CHECK(geometry.geometry_bytes == 8 * mib);
  CHECK(geometry.direct_capable);
  hooks.statfs = [](int, struct statfs* fs) {
    fs->f_bsize = 1L << 20;  // NFS wsize is not an alignment either.
    fs->f_type  = 0x6969;
    return 0;
  };
  CHECK(discover_file_geometry(-1, "nfs", hooks).granule == 4096);
  CHECK(discover_file_geometry(-1, "nfs", hooks).geometry_bytes == 0);
  hooks.dio_alignment = [](int) { return 0UL; };
  CHECK_FALSE(discover_file_geometry(-1, "ext4", hooks).direct_capable);
  hooks.dio_alignment = [](int) { return std::nullopt; };
  hooks.statfs        = [](int, struct statfs* fs) {
    fs->f_bsize = 4096;
    fs->f_type  = 0;
    return 0;
  };
  CHECK(discover_file_geometry(-1, "gpfs", hooks).geometry_bytes == 0);
  CHECK(discover_file_geometry(-1, "lustre", hooks).geometry_bytes == mib);
  hooks.lustre_stripe = [](int) { return 4 * mib; };
  CHECK(discover_file_geometry(-1, "lustre", hooks).geometry_bytes == 4 * mib);
  hooks.lustre_stripe = [](int) { return 4097UL; };
  CHECK(discover_file_geometry(-1, "lustre", hooks).geometry_bytes == mib);
  CHECK(discover_file_geometry(-1, "beegfs", hooks).geometry_bytes == 2 * mib);
  CHECK(discover_file_geometry(-1, "ext4", hooks).geometry_bytes == 0);
  auto piece = range_plan::next_cut(3 * mib, 9 * mib, 8 * mib, 4096, {}, 9 * mib, 4 * mib);
  CHECK(piece.logical.end() == 4 * mib);
  CHECK(piece.physical.end() == 4 * mib);
}

TEST_CASE("engine physical reads honor per-file granules and repeating geometry",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(4 * mib, configuration(1));
  f.start();
  auto group       = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future      = group->get_future();
  auto s           = f.slice(group, {0, f.bytes});
  s.granule        = 16384;
  s.geometry_bytes = mib / 2;
  f.engine->hand_off(std::move(s));
  CHECK(std::move(future).get(5s) == f.bytes);
  REQUIRE(f.engine->drain(5s));
  auto submissions = f.transport->submissions();
  REQUIRE(submissions.size() == 8);
  for (auto const& item : submissions) {
    CHECK(item.file_offset % 16384 == 0);
    CHECK(item.size == mib / 2);
  }
  CHECK(f.contents() == f.image);
}

TEST_CASE("network EIO retry requires a positive transient classification",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const network    = GENERATE(false, true);
  auto const classified = GENERATE(false, true);
  auto cfg              = configuration(1);
  cfg.mount_overrides.push_back(
    {mount_match::fs_type, "nfs4", mount_override_state::operator_qualified, 1, mib});
  fixture f(mib, cfg);
  if (network) {
    mount_identity identity;
    identity.fs_type = "nfs4";
    identity.valid   = true;
    f.policy =
      std::make_shared<mount_policy>(mount_policy_key{1, 2, "", "eio-test", 1}, identity, cfg);
  }
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = -EIO;
  f.transport->add_fault(fault);
  f.start([&](auto& opts) {
    opts.transient_network_eio = [classified](auto const&) { return classified; };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  CHECK(std::move(future).get(5s) == mib);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == (network && classified ? 2 : 1));
  CHECK(f.contents() == f.image);
}

TEST_CASE("file demotion identity is shared across live handles but not other inodes",
          "[cufile][engine][hardening]")
{
  char name[] = "/tmp/cucs_file_policy_XXXXXX";
  file_descriptor fd{mkstemp(name)};
  REQUIRE(fd.get() >= 0);
  file_descriptor second{::open(name, O_RDONLY)};
  REQUIRE(second.get() >= 0);
  ::unlink(name);
  auto first = file_async_policy::get(fd.get());
  auto peer  = file_async_policy::get(second.get());
  CHECK(first == peer);
  first->read_demoted = true;
  CHECK(peer->read_demoted.load());
}

TEST_CASE("coordinator can fail after its real engine registration has retired",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 2);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(f.engine->drain(5s));
  REQUIRE(group->tasks_remaining() == 1);
  f.engine.reset();
  group->report_error(std::make_error_code(std::errc::io_error));
  CHECK_THROWS(std::move(future).get(500ms));
}

TEST_CASE("lease APIs return metadata without blocking or losing ownership",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  auto lane = f.lanes->try_lease();
  auto slot = f.arena->try_lease_probe(mib);
  REQUIRE(lane);
  REQUIRE(slot);
  operation_pool pool(f.config);
  auto* op = pool.try_acquire();
  REQUIRE(op);
  op->file = f.object;
  CHECK_FALSE(operation_pool::try_release_empty(*op));
  CHECK(op->file == f.object);
  op->file.reset();
  op->mark_work_started();
  CHECK_FALSE(operation_pool::try_release_empty(*op));
  op->prove_recovery_quiescence();
  CHECK(lane->try_reset());
  CHECK(slot->try_reset());
  CHECK_FALSE(bool(*lane));
  CHECK_FALSE(bool(*slot));
  CHECK(operation_pool::try_release_empty(*op));
  CHECK(pool.available() == pool.capacity());
  CHECK(f.lanes->available() == f.lanes->capacity());
  // A nonblocking release must refuse a possible final external-resource destructor.
  auto isolated = std::make_shared<lane_pool>(0, f.config, nullptr);
  auto final    = isolated->try_lease();
  REQUIRE(final);
  isolated.reset();
  CHECK_FALSE(final->try_reset());
  CHECK(bool(*final));
}

TEST_CASE("coordinator failure notifications are weak exactly once and safe after target death",
          "[cufile][engine][hardening]")
{
  auto group = std::make_shared<grouped_coordinator>(0, 3);
  unsigned calls{};
  auto token = std::make_shared<grouped_coordinator::failure_notification>([&] { ++calls; });
  group->register_failure_notification(token);
  group->register_failure_notification(token);
  auto expired =
    std::make_shared<grouped_coordinator::failure_notification>([] { FAIL("expired callback"); });
  group->register_failure_notification(expired);
  expired.reset();
  group->report_error(std::make_error_code(std::errc::io_error));
  CHECK(calls == 1);
  group->register_failure_notification(token);
  CHECK(calls == 1);
  group->on_complete();
  group->on_complete();
}

TEST_CASE("external sibling failure wakes a pending engine without its health timer",
          "[cufile][engine][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto cfg                = configuration(1);
  cfg.poll.idle_health_ms = 60000;
  fixture f(mib, cfg);
  f.start();
  auto group          = std::make_shared<grouped_coordinator>(mib, 2);
  auto future         = group->get_future();
  auto s              = f.slice(group, {0, mib});
  s.publication_ready = std::make_shared<std::atomic<bool>>(false);
  f.engine->hand_off(std::move(s));
  std::this_thread::sleep_for(10ms);
  group->report_error(std::make_error_code(std::errc::io_error));
  CHECK_THROWS(std::move(future).get(500ms));
  REQUIRE(f.engine->drain(5s));
  f.engine.reset();
  // A coordinator may outlive the engine; a late registration/failure is harmless.
  auto late               = std::make_shared<grouped_coordinator>(0, 1);
  auto dead               = std::make_shared<int>(1);
  std::weak_ptr<int> weak = dead;
  auto token              = std::make_shared<grouped_coordinator::failure_notification>(
    [weak] { static_cast<void>(weak.lock()); });
  late->register_failure_notification(token);
  dead.reset();
  late->report_error(std::make_error_code(std::errc::io_error));
}
