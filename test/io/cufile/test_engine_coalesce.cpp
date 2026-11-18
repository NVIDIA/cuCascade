/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "engine_test_utils.hpp"

#include <unistd.h>

#include <cstdio>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

namespace {
struct batch {
  fixture& f;
  std::shared_ptr<grouped_coordinator> group;
  std::vector<engine_slice> slices;
  std::shared_ptr<std::atomic<bool>> ready = std::make_shared<std::atomic<bool>>(false);

  batch(fixture& f, std::vector<range> const& ranges, bool ptds = false) : f(f)
  {
    std::size_t bytes{};
    for (auto rng : ranges)
      bytes += rng.size;
    group = std::make_shared<grouped_coordinator>(bytes, ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      auto s = f.slice(group, ranges[i], i * 8192, ptds);
      if (!slices.empty()) s.fence = slices.front().fence;
      s.publication_ready = ready;
      slices.push_back(std::move(s));
    }
  }
  void publish()
  {
    for (auto& s : slices)
      f.engine->hand_off(std::move(s));
    ready->store(true, std::memory_order_release);
    f.engine->notify();
  }
};
std::vector<range> columns(std::size_t count)
{
  std::vector<range> result;
  // Reverse order proves that selection sorts file ranges, not destinations.
  for (std::size_t i = count; i != 0; --i)
    result.push_back({(i - 1) * 4096 + 37, 1024});
  return result;
}
void check_columns(fixture& f, std::vector<range> const& ranges)
{
  auto actual   = f.contents();
  auto expected = std::vector<std::byte>(f.bytes, std::byte{0xcc});
  for (std::size_t i = 0; i < ranges.size(); ++i)
    std::copy_n(f.image.begin() + static_cast<std::ptrdiff_t>(ranges[i].offset),
                ranges[i].size,
                expected.begin() + static_cast<std::ptrdiff_t>(i * 8192));
  CHECK(actual == expected);  // Includes untouched gaps AND destination padding.
}
}  // namespace

TEST_CASE("engine tiny columns coalesce sorted offsets with exact destinations",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const enabled  = GENERATE(true, false);
  auto const geometry = GENERATE(0UL, 64UL << 10, 8UL << 20);
  auto cfg            = configuration();
  cfg.coalesce        = enabled;
  fixture f(mib, cfg);
  f.start();
  auto ranges = columns(100);
  batch b(f, ranges);
  for (auto& slice : b.slices)
    slice.geometry_bytes = geometry;
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == 100 * 1024);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  CHECK(b.group->tasks_remaining() == 0);
  auto const ops = enabled ? (geometry == 64UL << 10 ? 7 : 2) : 100;
  CHECK(f.transport->snapshot().read_calls == ops);
  if (geometry)
    for (auto const& call : f.transport->submissions())
      CHECK(call.file_offset / geometry == (call.file_offset + call.size - 1) / geometry);
  CHECK(f.transport->snapshot().copy_calls == 100);
  REQUIRE(until([&] { return f.engine->gauges().completed_ops == ops; }));
  auto stats = f.engine->gauges();
  CHECK(stats.coalesced_ops == (enabled ? ops : 0));
  CHECK(stats.coalesced_targets == (enabled ? 100 : 0));
  CHECK(stats.gap_bytes == (enabled ? (100 - ops) * (4096 - 1024) : 0));
}

TEST_CASE("engine coalescing respects target and physical window limits",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const limit         = GENERATE(1U, 3U, 64U);
  auto cfg                 = configuration();
  cfg.coalesce_max_targets = limit;
  fixture f(2 * mib, cfg);
  f.start();
  auto ranges = columns(12);
  batch b(f, ranges);
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == 12 * 1024);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == (12 + limit - 1) / limit);
  for (auto call : f.transport->submissions())
    CHECK(call.size <= mib);
}

TEST_CASE("engine coalescing separates stream fence file priority and request identities",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const different = GENERATE(0, 1, 2, 3, 4, 5);
  fixture f(mib);
  f.start();
  auto ranges = columns(2);
  batch b(f, ranges);
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  auto other_group  = std::make_shared<grouped_coordinator>(1024, 1);
  auto other_future = other_group->get_future();
  auto future       = b.group->get_future();
  if (different == 0) b.slices[1].user_stream = other;
  if (different == 1) b.slices[1].fence = f.slice(other_group, {0, 1}).fence;
  if (different == 2) b.slices[1].file = std::make_shared<file>(mib);
  if (different == 3) b.slices[1].priority = io_priority::low;
  if (different == 4) {
    // Transfer the original second credit out, then settle it; its replacement belongs
    // to another request and must not coalesce even with an identical stream and fence.
    CHECK(b.slices[1].credit.finish_success());
    b.slices[1].credit = operation_credit(other_group);
  }
  if (different == 5) b.slices[1].logical.offset = 16384;  // gap > merge_gap_size
  b.publish();
  CHECK(std::move(future).get(5s) == 2048);
  if (different == 4) CHECK(std::move(other_future).get(5s) == 1024);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == 2);
  CHECK(f.engine->gauges().coalesced_ops == 0);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
}

TEST_CASE("engine mixed sizes stripe large slices and coalesce small ones",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(6 * mib);
  f.start();
  auto ranges = columns(8);
  batch b(f, ranges);
  b.group->add_tasks(1);
  auto big              = f.slice(b.group, {mib, 3 * mib}, mib);
  big.publication_ready = b.ready;
  b.slices.push_back(std::move(big));
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == 8192);  // coordinator's byte result is fixed at creation
  REQUIRE(f.engine->drain(5s));
  auto actual = f.contents();
  CHECK(std::equal(actual.begin() + static_cast<std::ptrdiff_t>(mib),
                   actual.begin() + static_cast<std::ptrdiff_t>(4 * mib),
                   f.image.begin() + static_cast<std::ptrdiff_t>(mib)));
  CHECK(f.transport->snapshot().read_calls == 4);
  for (std::size_t i = 0; i < ranges.size(); ++i)
    CHECK(std::equal(actual.begin() + static_cast<std::ptrdiff_t>(i * 8192),
                     actual.begin() + static_cast<std::ptrdiff_t>(i * 8192 + ranges[i].size),
                     f.image.begin() + static_cast<std::ptrdiff_t>(ranges[i].offset)));
  CHECK(b.group->tasks_remaining() == 0);
  REQUIRE(until([&] { return f.engine->gauges().coalesced_targets == 8; }));
}

TEST_CASE("engine short coalesced reads retry until every target is covered",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib);
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = 5000;
  f.transport->add_fault(fault);
  f.start();
  auto ranges = columns(8);
  batch b(f, ranges);
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == 8192);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  auto calls = f.transport->submissions();
  REQUIRE(calls.size() == 2);
  CHECK(calls[1].file_offset == 4096);
  CHECK(calls[1].buffer_offset == 4096);
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine coalesced errors settle every credit and retain untouched destinations",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const copy_error = GENERATE(-1, 0, 3);
  fixture f(mib);
  if (copy_error >= 0)
    f.transport->fail_copy(static_cast<std::size_t>(copy_error));
  else {
    transport_simulator::fault fault;
    fault.index             = 0;
    fault.completion_result = -EACCES;
    f.transport->add_fault(fault);
  }
  f.start();
  batch b(f, columns(8));
  auto future = b.group->get_future();
  b.publish();
  CHECK_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(b.group->tasks_remaining() == 0);
  CHECK_FALSE(b.group->should_continue());
  auto expected = std::vector<std::byte>(mib, std::byte{0xcc});
  auto ranges   = columns(8);
  for (int i = 0; i < copy_error; ++i) {
    auto target = 7 - static_cast<std::size_t>(i);  // copies follow sorted file offsets
    std::copy_n(f.image.begin() + static_cast<std::ptrdiff_t>(ranges[target].offset),
                1024,
                expected.begin() + static_cast<std::ptrdiff_t>(target * 8192));
  }
  CHECK(f.contents() == expected);
  CHECK(f.transport->snapshot().read_calls == 1);
}

TEST_CASE("engine cancellation drains a coalesced physical read before settling all targets",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib);
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  f.transport->add_fault(fault);
  f.start();
  batch b(f, columns(8));
  auto cancel = std::make_shared<std::atomic<bool>>(false);
  for (auto& s : b.slices)
    s.cancel = cancel;
  auto future = b.group->get_future();
  b.publish();
  REQUIRE(until([&] { return f.transport->snapshot().submitted == 1; }));
  cancel->store(true);
  f.engine->notify();
  CHECK_FALSE(std::move(future).wait(10ms));
  f.transport->unblock_all();
  CHECK_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(b.group->tasks_remaining() == 0);
  CHECK(f.transport->snapshot().copy_calls == 0);
}

TEST_CASE("engine PTDS coalesced groups finish host complete", "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib);
  f.start();
  gate producer(f.stream);
  auto ranges = columns(8);
  batch b(f, ranges, true);
  auto future = b.group->get_future();
  b.publish();
  REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
  CHECK_FALSE(std::move(future).wait(10ms));
  producer.release();
  CHECK(std::move(future).get(5s) == 8192);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().wait_calls == 1);  // fence only, no stand-in join
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine physical coalescing windows fit one slot and exclude half-class slices",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto size = GENERATE(std::size_t{128 * 1024}, std::size_t{500 * 1024}, mib / 2);
  fixture f(8 * mib);
  f.start();
  std::vector<range> ranges;
  for (std::size_t i = 0; i < 10; ++i)
    ranges.push_back({i * (size + 4096), size});
  batch b(f, ranges);
  for (std::size_t i = 0; i < b.slices.size(); ++i)
    b.slices[i].user_ptr = static_cast<std::byte*>(f.destination) + i * (size + 8192);
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == size * 10);
  auto actual   = f.contents();
  auto expected = std::vector<std::byte>(f.bytes, std::byte{0xcc});
  for (std::size_t i = 0; i < ranges.size(); ++i)
    std::copy_n(f.image.begin() + static_cast<std::ptrdiff_t>(ranges[i].offset),
                size,
                expected.begin() + static_cast<std::ptrdiff_t>(i * (size + 8192)));
  CHECK(actual == expected);
  REQUIRE(f.engine->drain(5s));
  // Exactly-half-class slices are striped at the absolute grid, never coalesced.
  auto calls = f.transport->submissions();
  if (size < mib / 2)
    CHECK(calls.size() == (size == 128 * 1024 ? 2 : 5));
  else
    CHECK(f.engine->gauges().coalesced_ops == 0);
  for (auto call : calls)
    CHECK(call.size <= mib);
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine coalescing configuration bounds", "[cufile][engine][coalesce]")
{
  async_engine_config cfg;
  cfg.coalesce             = GENERATE(true, false);
  cfg.coalesce_max_targets = GENERATE(0U, 1U, 64U, 4096U, 4097U);
  if (cfg.coalesce_max_targets == 0 || cfg.coalesce_max_targets > 4096)
    CHECK_THROWS_AS(cfg.validate(), std::invalid_argument);
  else
    CHECK_NOTHROW(cfg.validate());
}

TEST_CASE("engine unpublished batches cannot submit before publication",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib);
  f.start();
  auto ranges = columns(8);
  batch b(f, ranges);
  auto future = b.group->get_future();
  for (auto& slice : b.slices)
    f.engine->hand_off(std::move(slice));
  REQUIRE(until([&] { return f.engine->gauges().outstanding_slices == 8; }));
  auto iterations = f.engine->gauges().reaper_iterations;
  f.engine->notify();
  REQUIRE(until([&] { return f.engine->gauges().reaper_iterations > iterations; }));
  CHECK(f.transport->snapshot().submitted == 0);
  CHECK_FALSE(future.is_ready());
  b.ready->store(true, std::memory_order_release);
  f.engine->notify();
  CHECK(std::move(future).get(5s) == 8192);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == 1);
}

TEST_CASE("engine coalesced submit and completion failures fall back exactly",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto submit = GENERATE(false, true);
  fixture f(mib);
  transport_simulator::fault fault;
  fault.index = 0;
  if (submit)
    fault.submit_status = {-EIO};
  else
    fault.completion_result = -EOPNOTSUPP;
  f.transport->add_fault(fault);
  f.start();
  auto ranges = columns(8);
  batch b(f, ranges);
  auto future = b.group->get_future();
  b.publish();
  CHECK(std::move(future).get(5s) == 8192);
  check_columns(f, ranges);
  REQUIRE(f.engine->drain(5s));
  CHECK(b.group->tasks_remaining() == 0);
  CHECK(f.engine->gauges().coalesced_targets == 8);
  CHECK(f.engine->gauges().sync_bytes > 0);
}

TEST_CASE("engine coalesced not-submitted groups fall back behind a stuck submitter",
          "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(2 * mib);
  transport_simulator::fault fault;
  fault.index        = 0;
  fault.stuck_submit = true;
  f.transport->add_fault(fault);
  f.start();
  auto first        = std::make_shared<grouped_coordinator>(mib, 1);
  auto first_future = first->get_future();
  f.engine->hand_off(f.slice(first, {mib, mib}, mib));
  REQUIRE(until([&] { return f.transport->snapshot().blocked_submissions == 1; }));
  batch b(f, columns(8));
  auto future = b.group->get_future();
  b.publish();
  REQUIRE(until([&] { return f.engine->gauges().coalesced_targets == 8; }));
  f.policy->open_circuit();
  f.engine->notify();
  CHECK(std::move(future).get(5s) == 8192);
  CHECK_FALSE(first_future.is_ready());
  f.transport->unblock_all();
  CHECK(std::move(first_future).get(5s) == mib);
  REQUIRE(f.engine->drain(5s));
  auto actual = f.contents();
  for (std::size_t i = 0; i < 8; ++i)
    CHECK(std::equal(actual.begin() + static_cast<std::ptrdiff_t>(i * 8192),
                     actual.begin() + static_cast<std::ptrdiff_t>(i * 8192 + 1024),
                     f.image.begin() + static_cast<std::ptrdiff_t>(columns(8)[i].offset)));
  CHECK(f.engine->gauges().not_submitted > 0);
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine zero-QD multi-target grants use sync fallback", "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  batch a(f, columns(8)), b(f, columns(8));
  auto other = f.make_policy(2);
  for (std::size_t i = 0; i < b.slices.size(); ++i) {
    b.slices[i].policy   = other;
    b.slices[i].user_ptr = static_cast<std::byte*>(f.destination) + 128 * 1024 + i * 8192;
  }
  auto af = a.group->get_future(), bf = b.group->get_future();
  a.publish();
  b.publish();
  REQUIRE(until([&] {
    control->run();
    return af.is_ready() && bf.is_ready();
  }));
  CHECK(std::move(af).get() == 8192);
  CHECK(std::move(bf).get() == 8192);
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
  auto actual   = f.contents();
  auto expected = std::vector<std::byte>(mib, std::byte{0xcc});
  auto ranges   = columns(8);
  for (auto base : {std::size_t{0}, std::size_t{128 * 1024}})
    for (std::size_t i = 0; i < ranges.size(); ++i)
      std::copy_n(f.image.begin() + static_cast<std::ptrdiff_t>(ranges[i].offset),
                  1024,
                  expected.begin() + static_cast<std::ptrdiff_t>(base + i * 8192));
  CHECK(actual == expected);
  CHECK(f.engine->gauges().sync_bytes > 0);
  CHECK(f.engine->gauges().coalesced_targets == 16);
  CHECK(a.group->tasks_remaining() == 0);
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine multi-target short reads check the real file size", "[cufile][engine][coalesce]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto shrink = GENERATE(false, true);
  fixture f(mib);
  std::unique_ptr<std::FILE, decltype(&std::fclose)> disk(std::tmpfile(), &std::fclose);
  REQUIRE(disk);
  auto fd = fileno(disk.get());
  REQUIRE(pwrite(fd, f.image.data(), f.image.size(), 0) == static_cast<ssize_t>(f.image.size()));
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = 5000;
  fault.stuck_completion  = true;
  f.transport->add_fault(fault);
  f.start([&](auto& opts) { opts.buffered_fd = [&](io_object const&) { return fd; }; });
  auto ranges = columns(8);
  batch b(f, ranges);
  auto future = b.group->get_future();
  b.publish();
  REQUIRE(until([&] { return f.transport->snapshot().submitted == 1; }));
  if (shrink) REQUIRE(ftruncate(fd, 5000) == 0);
  f.transport->unblock_all();
  if (shrink) {
    CHECK_THROWS(std::move(future).get(5s));
    CHECK(f.contents() == std::vector<std::byte>(mib, std::byte{0xcc}));
  } else {
    CHECK(std::move(future).get(5s) == 8192);
    check_columns(f, ranges);
  }
  REQUIRE(f.engine->drain(5s));
  CHECK(b.group->tasks_remaining() == 0);
}

TEST_CASE("engine operation pool retains preallocated coalescing storage",
          "[cufile][engine][coalesce]")
{
  auto cfg                 = configuration();
  cfg.coalesce_max_targets = 64;
  operation_pool pool(cfg);
  std::uintptr_t storage{};
  for (int attempt = 0; attempt < 10; ++attempt) {
    auto* op = pool.try_acquire();
    REQUIRE(op);
    auto group = std::make_shared<grouped_coordinator>(64, 64);
    for (int i = 0; i < 64; ++i)
      op->targets.emplace_back(range{0, 1}, nullptr, operation_credit(group));
    auto address = reinterpret_cast<std::uintptr_t>(&op->targets[2]);
    if (attempt) CHECK(address == storage);
    storage = address;
    CHECK(operation_pool::try_release(*op));
    CHECK(group->tasks_remaining() == 0);
  }
}
