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

// io_uring engine scheduling: uring::config::slices_per_pass (per-group, per-pass
// expansion cap), demand-before-background pass order, the background group
// sub-limit, first-I/O delay statistics, per-runner gauges and config validation.
//
// Determinism.  No assertion depends on wall-clock time.  The ordering tests
// run one runner and rely on two properties of the engine and the kernel:
//  - Requests published while the context has no runner stay queued, so the
//    first pull of a runner started afterwards sees all of them at once
//    (external runner, see open_admission / external_runner).
//  - Buffered reads of a file that is entirely in the page cache complete at
//    submission, and their CQEs are posted in submission order: the global
//    completion order of the slices mirrors the order the engine dispatched
//    them in.  These tests therefore set use_odirect = false.
// Gauge assertions only use values the engine publishes at the end of the
// first pass that submitted operations (before any of them can be reaped).

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::device_buffer;
using cucascade::io::host_buffer;
using cucascade::io::io_options;
using cucascade::io::prepared_io_completion;
using cucascade::io::prepared_io_slice;
using cucascade::io::queue_stats;
using cucascade::io::range;
using cucascade::io::request_class;
using cucascade::io::request_class_count;
using cucascade::io::request_class_index;
using cucascade::io::uring::uring_ioctx;
using cucascade::io::uring::uring_reactor;
using uring_config = cucascade::io::uring::config;
using chunk_span   = std::span<cucascade::io::cache::cached_chunk* const>;

constexpr std::size_t KiB = 1UL << 10;
constexpr std::size_t MiB = 1UL << 20;
/// Staging slots of one engine with this fixture: the engine's 64 MiB pinned
/// budget in 1 MiB blocks (every in-flight operation holds at least one).
constexpr std::size_t engine_slots = 64;
/// templated_ioctx::read_fanout: one grouped request per 64 MiB of a read, at most 4.
constexpr std::size_t fanout_bytes = 64UL << 20;
constexpr std::size_t max_fanout   = 4;
constexpr std::size_t no_index     = std::numeric_limits<std::size_t>::max();

constexpr std::array<request_class, request_class_count> all_classes{
  request_class::latency, request_class::read, request_class::write, request_class::background};

[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  return static_cast<std::uint8_t>((offset * 2654435761ULL) >> 11U);
}

[[nodiscard]] constexpr std::size_t expected_groups(std::size_t total_bytes) noexcept
{
  return std::clamp<std::size_t>(total_bytes / fanout_bytes, 1, max_fanout);
}

/// A temporary file filled with @ref pattern_byte, removed on destruction.
class temp_file {
 public:
  explicit temp_file(std::size_t size)
  {
    static std::atomic<int> counter{0};
    _path = std::filesystem::temp_directory_path() /
            ("cucascade_uring_sched_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1)) + ".bin");
    std::vector<char> data(size);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern_byte(i));
    }
    std::ofstream out(_path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }
  ~temp_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  temp_file(temp_file const&)            = delete;
  temp_file& operator=(temp_file const&) = delete;

  [[nodiscard]] std::string path() const { return _path.string(); }

 private:
  std::filesystem::path _path;
};

/// Pinned staging resource (1 MiB blocks) shared by the engines of a context.
/// One engine takes 64 MiB; the default capacity fits two at once.
class staging_resource {
 public:
  explicit staging_resource(std::size_t capacity = 128 * MiB)
    : _upstream(0, /*make_portable=*/true), _mr(0, _upstream, capacity, capacity, 1 * MiB)
  {
  }

  [[nodiscard]] std::shared_ptr<uring_reactor::reactor_context> context(uring_config cfg = {})
  {
    return std::make_shared<uring_reactor::reactor_context>(cfg, &_mr);
  }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream;
  cucascade::memory::fixed_size_host_memory_resource _mr;
};

/// Page-aligned host buffer (O_DIRECT-eligible destination).
class aligned_buffer {
 public:
  aligned_buffer(std::size_t size, std::uint8_t fill)
    : _data(static_cast<std::uint8_t*>(::operator new(size, std::align_val_t{alignment}))),
      _size(size)
  {
    std::memset(_data.get(), fill, size);
  }

  [[nodiscard]] std::uint8_t* data() const noexcept { return _data.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

 private:
  static constexpr std::size_t alignment = 4096;
  struct deleter {
    void operator()(std::uint8_t* ptr) const noexcept
    {
      ::operator delete(ptr, std::align_val_t{alignment});
    }
  };
  std::unique_ptr<std::uint8_t, deleter> _data;
  std::size_t _size;
};

/// Offset of the first byte of @p data that differs from the file content at
/// @p file_offset, or @c no_index.
[[nodiscard]] std::size_t first_mismatch(std::span<std::uint8_t const> data,
                                         std::size_t file_offset)
{
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i] != pattern_byte(file_offset + i)) return i;
  }
  return no_index;
}

template <class Predicate>
bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

void record_min(std::atomic<std::size_t>& target, std::size_t value) noexcept
{
  auto current = target.load();
  while (value < current && !target.compare_exchange_weak(current, value)) {}
}

[[nodiscard]] cucascade::io::class_stats const& of(queue_stats const& stats, request_class cls)
{
  return stats.per_class[request_class_index(cls)];
}

/// Opens admission of a context built without runner threads and leaves no
/// runner behind: requests published next stay queued until the test starts
/// an @ref external_runner, whose first pull then sees all of them at once.
void open_admission(uring_ioctx& ctx) { CHECK(ctx.run_for(0ms) == 0); }

/// One runner thread driving @p ctx (built with 0 runner threads) until destroyed.
class external_runner {
 public:
  explicit external_runner(uring_ioctx& ctx)
    : _thread([&ctx](std::stop_token stop) { static_cast<void>(ctx.run(std::move(stop))); })
  {
  }

 private:
  std::jthread _thread;
};

/// Polls (bounded) until the context's only runner published zero in-flight
/// operations: a future resolves on the runner thread before that pass
/// publishes its gauges.  Runners vanish from stats() once they exit, so this
/// must be called while the runner runs.
[[nodiscard]] bool wait_runner_drained(uring_ioctx const& ctx)
{
  return wait_until(
    [&] {
      auto const stats = ctx.stats();
      return stats.runners.size() == 1 && stats.runners.front().inflight_ops == 0;
    },
    1000ms);
}

/// Polls (bounded) until every grouped request was retired: retirement (which
/// records the first-I/O delay) follows the resolution of the future.
[[nodiscard]] bool wait_all_retired(uring_ioctx const& ctx)
{
  return wait_until([&] { return ctx.stats().in_flight_requests == 0; }, 5000ms);
}

/// Aborts the process unless destroyed within @p timeout: a hang in shutdown
/// must fail the run instead of blocking it (Catch2 assertions are not
/// thread-safe, so the watchdog cannot report through them).
class watchdog {
 public:
  watchdog(std::chrono::seconds timeout, char const* what)
    : _thread([timeout, what](std::stop_token stop) {
        std::mutex mutex;
        std::condition_variable_any cv;
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, stop, timeout, [&] { return stop.stop_requested(); })) {
          std::fprintf(stderr,
                       "watchdog: %s did not finish within %lld s\n",
                       what,
                       static_cast<long long>(timeout.count()));
          std::abort();
        }
      })
  {
  }

 private:
  std::jthread _thread;
};

}  // namespace

TEST_CASE("uring slices_per_pass never changes the bytes read", "[io][uring][scheduling]")
{
  constexpr std::size_t slice_bytes  = 64 * KiB;
  constexpr std::size_t n_slices     = 200;  // more than the 64 staging slots: slots are refilled
  constexpr std::size_t head_skip    = 7;    // odd slices start 7 bytes in and end 5 bytes early:
  constexpr std::size_t tail_skip    = 5;    // unaligned, so buffered; even ones O_DIRECT-eligible
  constexpr std::uint8_t poison      = 0xA5;
  constexpr std::size_t buffer_bytes = n_slices * slice_bytes;
  temp_file file(16 * MiB);
  staging_resource staging;

  for (auto const spp : {std::size_t{1}, std::size_t{8}, std::size_t{0}}) {
    CAPTURE(spp);
    aligned_buffer data(buffer_bytes, poison);  // outlives the context
    uring_config cfg{};
    cfg.slices_per_pass = spp;
    uring_ioctx ctx(1, staging.context(cfg));
    auto object = ctx.open_io_object(file.path());
    ctx.start();

    std::vector<prepared_io_slice> slices;
    std::size_t requested = 0;
    for (std::size_t i = 0; i < n_slices; ++i) {
      bool const ragged = i % 2 == 1;
      auto const offset = i * slice_bytes + (ragged ? head_skip : 0);
      auto const size   = slice_bytes - (ragged ? head_skip + tail_skip : 0);
      slices.emplace_back(range{offset, size}, host_buffer{data.data() + offset});
      requested += size;
    }
    CHECK(ctx.mixed_readv_async_io(*object, std::move(slices)).get(30s) == requested);
    ctx.shutdown();

    // Destination offset == file offset: requested bytes hold the file pattern,
    // the gaps of the ragged slices still hold the poison.
    std::size_t wrong       = 0;
    std::size_t first_wrong = no_index;
    for (std::size_t at = 0; at < buffer_bytes; ++at) {
      auto const within = at % slice_bytes;
      bool const ragged = (at / slice_bytes) % 2 == 1;
      bool const gap    = ragged && (within < head_skip || within >= slice_bytes - tail_skip);
      if (data.data()[at] != (gap ? poison : pattern_byte(at)) && wrong++ == 0) first_wrong = at;
    }
    INFO("first wrong byte at " << first_wrong);
    CHECK(wrong == 0);
  }
}

TEST_CASE("uring slices_per_pass interleaves the groups a runner holds", "[io][uring][scheduling]")
{
  constexpr std::size_t slice_bytes = 64 * KiB;
  constexpr std::size_t n_a         = 192;  // 12 MiB: one read-class group (fan-out 1)
  constexpr std::size_t n_b         = 8;
  constexpr std::size_t file_bytes  = 4 * MiB;
  constexpr std::size_t file_slices = file_bytes / slice_bytes;
  // Uncapped, A (pulled first) takes every freed slot while it has untaken
  // slices, so B's first operation is dispatched only once A dispatched its
  // last slice -- i.e. after n_a - engine_slots of A's completions freed the
  // slots for it.  Exact on one runner: slots are freed only by reaping.
  constexpr std::size_t fifo_bound = n_a - engine_slots;
  temp_file file(file_bytes);
  staging_resource staging;

  for (auto const spp : {std::size_t{8}, std::size_t{0}}) {
    CAPTURE(spp);
    // A cycles over the file; slice i of every cycle lands at the same
    // destination offset (identical bytes), B reads into its own buffer.
    std::vector<std::uint8_t> a_data(file_bytes);
    std::vector<std::uint8_t> b_data(n_b * slice_bytes);
    std::atomic<std::size_t> completions{0};
    std::atomic<std::size_t> b_first{no_index};
    auto const a_done = std::make_shared<prepared_io_completion>(
      [&completions](chunk_span, bool) noexcept { completions.fetch_add(1); });
    auto const b_done =
      std::make_shared<prepared_io_completion>([&completions, &b_first](chunk_span, bool) noexcept {
        record_min(b_first, completions.fetch_add(1));
      });

    uring_config cfg{};
    cfg.slices_per_pass = spp;
    cfg.use_odirect     = false;  // page-cache reads: completion order == dispatch order
    uring_ioctx ctx(0, staging.context(cfg));
    auto object = ctx.open_io_object(file.path());
    open_admission(ctx);

    std::vector<prepared_io_slice> a_slices;
    for (std::size_t i = 0; i < n_a; ++i) {
      auto const offset = (i % file_slices) * slice_bytes;
      a_slices.emplace_back(range{offset, slice_bytes}, host_buffer{a_data.data() + offset});
      a_slices.back().on_complete = a_done;
    }
    std::vector<prepared_io_slice> b_slices;
    for (std::size_t i = 0; i < n_b; ++i) {
      b_slices.emplace_back(range{i * slice_bytes, slice_bytes},
                            host_buffer{b_data.data() + i * slice_bytes});
      b_slices.back().on_complete = b_done;
    }
    // Both queued before the runner exists: its first pull takes A, then B.
    auto a =
      ctx.mixed_readv_async_io(*object, std::move(a_slices), io_options{request_class::read});
    auto b =
      ctx.mixed_readv_async_io(*object, std::move(b_slices), io_options{request_class::read});
    {
      external_runner runner(ctx);
      CHECK(std::move(a).get(30s) == n_a * slice_bytes);
      CHECK(std::move(b).get(30s) == n_b * slice_bytes);
    }
    CHECK(first_mismatch(a_data, 0) == no_index);
    CHECK(first_mismatch(b_data, 0) == no_index);
    CHECK(completions.load() == n_a + n_b);

    INFO("global completion index of B's first slice: " << b_first.load());
    if (spp == 0) {
      CHECK(b_first.load() >= fifo_bound);
    } else {
      // Capped, A and B each expand spp slices per pass: B is dispatched in the
      // runner's first pass, right after A's first spp slices (nominally index spp).
      CHECK(b_first.load() < 100);
    }
  }
}

TEST_CASE("uring slices_per_pass does not bound the queue depth of one request",
          "[io][uring][scheduling]")
{
  // One 256 MiB request: fanned out into 4 grouped requests of 256 slices.
  constexpr std::size_t slice_bytes = 256 * KiB;
  constexpr std::size_t n_slices    = 1024;
  constexpr std::size_t file_bytes  = 4 * MiB;
  constexpr std::size_t file_slices = file_bytes / slice_bytes;
  constexpr std::size_t total_bytes = n_slices * slice_bytes;
  constexpr std::size_t n_groups    = expected_groups(total_bytes);
  static_assert(n_groups == max_fanout);
  temp_file file(file_bytes);
  staging_resource staging;

  for (auto const spp : {std::size_t{1}, std::size_t{0}}) {
    CAPTURE(spp);
    std::vector<std::uint8_t> data(file_bytes);  // slices cycle over the file: identical bytes
    uring_config cfg{};
    cfg.slices_per_pass = spp;
    cfg.use_odirect     = false;
    uring_ioctx ctx(0, staging.context(cfg));
    auto object = ctx.open_io_object(file.path());
    open_admission(ctx);

    std::vector<prepared_io_slice> slices;
    for (std::size_t i = 0; i < n_slices; ++i) {
      auto const offset = (i % file_slices) * slice_bytes;
      slices.emplace_back(range{offset, slice_bytes}, host_buffer{data.data() + offset});
    }
    auto future = ctx.mixed_readv_async_io(*object, std::move(slices), io_options{});

    std::uint32_t depth = 0;
    {
      external_runner runner(ctx);
      CHECK(std::move(future).get(60s) == total_bytes);
      auto const stats = ctx.stats();  // read while the runner is registered
      REQUIRE(stats.runners.size() == 1);
      depth = stats.runners.front().max_inflight_ops;
    }
    CHECK(first_mismatch(data, 0) == no_index);

    // Only the part of the depth that does not depend on device latency is
    // asserted: page-cache reads complete at submission, so a capped runner
    // reaps them one pass later and its depth stays near spp per group, while
    // device reads accumulate up to the slot limit (the run loop never waits
    // while it makes progress).  Both bounds below are reached in the first
    // pass that submits operations, whose gauges are published before any
    // reap.
    INFO("max_inflight_ops " << depth);
    if (spp == 0) {
      // Uncapped: the first group takes every staging slot in one pass.
      CHECK(depth >= engine_slots / 2);
    } else {
      // The cap is per group: every fan-out group of the request expands
      // spp slices in the same pass, so one request runs deeper than spp.
      CHECK(depth >= n_groups * spp);
    }
  }
}

TEST_CASE("uring background work leaves slots and expansion room for demand reads",
          "[io][uring][scheduling]")
{
  constexpr std::size_t bg_slice     = 1 * MiB;
  constexpr std::size_t bg_slices    = 256;
  constexpr std::size_t bg_total     = bg_slices * bg_slice;
  constexpr std::size_t file_bytes   = 64 * MiB;
  constexpr std::size_t file_slices  = file_bytes / bg_slice;
  constexpr std::size_t demand_off   = 32 * MiB;
  constexpr std::size_t demand_bytes = 4 * MiB;
  // One grouped request per fan-out group (templated_ioctx::read_fanout): the
  // background read is 4 groups, more than max_background_groups (2).
  constexpr std::size_t bg_groups = expected_groups(bg_total);
  static_assert(bg_groups == 4);
  constexpr std::size_t bg_group_limit = 2;  // scheduling_config::max_background_groups default
  temp_file file(file_bytes);
  staging_resource staging;

  // Everything the runner touches outlives the context and the runner.
  std::vector<std::uint8_t> bg_data(file_bytes);  // slices cycle over the file: identical bytes
  std::vector<std::uint8_t> demand_data(demand_bytes);
  std::atomic<std::size_t> bg_done{0};
  std::atomic<std::size_t> bg_done_at_demand{no_index};
  std::atomic<bool> runner_held{false};
  std::atomic<bool> release{false};

  // The first background completion holds the runner (bounded) inside its
  // reap until the demand request is queued: the demand then arrives at a
  // known point, with background groups holding the runner.
  auto const bg_complete = std::make_shared<prepared_io_completion>([&](chunk_span, bool) noexcept {
    if (bg_done.fetch_add(1) != 0) return;
    runner_held.store(true);
    auto const deadline = std::chrono::steady_clock::now() + 10s;
    while (!release.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(100us);
    }
  });
  auto const demand_complete = std::make_shared<prepared_io_completion>(
    [&](chunk_span, bool) noexcept { bg_done_at_demand.store(bg_done.load()); });

  uring_config cfg{};
  cfg.use_odirect     = false;  // page-cache reads: completion order == dispatch order
  cfg.slices_per_pass = 8;      // the default, pinned: the bound below is derived for it
  uring_ioctx ctx(0, staging.context(cfg));
  auto object = ctx.open_io_object(file.path());
  open_admission(ctx);

  std::vector<prepared_io_slice> bg;
  for (std::size_t i = 0; i < bg_slices; ++i) {
    auto const offset = (i % file_slices) * bg_slice;
    bg.emplace_back(range{offset, bg_slice}, host_buffer{bg_data.data() + offset});
    bg.back().on_complete = bg_complete;
  }
  auto background =
    ctx.mixed_readv_async_io(*object, std::move(bg), io_options{request_class::background});

  queue_stats held;
  std::size_t bg_done_at_enqueue = 0;
  {
    external_runner runner(ctx);
    struct release_on_exit {
      std::atomic<bool>& flag;
      ~release_on_exit() { flag.store(true); }
    } const guard{release};  // destroyed (releasing the runner) before the runner joins

    REQUIRE(wait_until([&] { return runner_held.load(); }, 10000ms));
    held               = ctx.stats();
    bg_done_at_enqueue = bg_done.load();
    std::vector<prepared_io_slice> demand;
    demand.emplace_back(range{demand_off, demand_bytes}, host_buffer{demand_data.data()});
    demand.back().on_complete = demand_complete;
    auto demand_future =
      ctx.mixed_readv_async_io(*object, std::move(demand), io_options{request_class::read});
    release.store(true);

    CHECK(std::move(demand_future).get(30s) == demand_bytes);
    CHECK(std::move(background).get(60s) == bg_total);
    REQUIRE(wait_all_retired(ctx));
  }
  CHECK(first_mismatch(demand_data, demand_off) == no_index);
  CHECK(first_mismatch(bg_data, 0) == no_index);

  // While held, the runner had pulled only max_background_groups of the
  // background groups; the others were still queued (expansion room for reads).
  CHECK(bg_done_at_enqueue == 1);
  CHECK(held.in_flight_requests == bg_group_limit);
  CHECK(of(held, request_class::background).queued_requests == bg_groups - bg_group_limit);
  REQUIRE(held.runners.size() == 1);
  CHECK(held.runners.front().active_groups == bg_group_limit);

  // The demand is pulled at the end of the pass it arrived in and dispatched
  // first in the next pass (demand tier before background tier), so it
  // completes after 2 passes of background (2 groups x 8 slices each: 32;
  // 96 with slices_per_pass = 0).  Without the tiered pass order it would
  // complete after the background operations planned ahead of it (derived:
  // 128 for the pass order before this port, ~176 for the pre-port policy).
  INFO("background completions before the demand completed: " << bg_done_at_demand.load());
  CHECK(bg_done_at_demand.load() < 128);

  auto const stats = ctx.stats();
  // One first-I/O sample per grouped request.
  CHECK(of(stats, request_class::read).first_io_count == 1);
  CHECK(of(stats, request_class::background).first_io_count == bg_groups);
  // Background groups 3 and 4 were queued before the demand and could only be
  // pulled once group 1 or 2 finished expanding (8 passes), i.e. after the
  // demand's first operation: their first-I/O delay is the larger one.
  CHECK(of(stats, request_class::read).first_io_max <=
        of(stats, request_class::background).first_io_max);
}

TEST_CASE("uring first-I/O delay statistics count each request once per class",
          "[io][uring][scheduling]")
{
  struct traffic {
    request_class cls;
    std::size_t count;
    std::size_t first_offset;
    std::size_t bytes;
  };
  // Each request is a single grouped request (fan-out 1); disjoint ranges.
  constexpr std::array<traffic, 3> plan{traffic{request_class::latency, 6, 0, 4 * KiB},
                                        traffic{request_class::read, 3, 1 * MiB, 2 * MiB},
                                        traffic{request_class::background, 2, 8 * MiB, 4 * MiB}};
  constexpr std::size_t file_bytes = 16 * MiB;
  temp_file file(file_bytes);
  staging_resource staging;
  std::vector<std::uint8_t> data(file_bytes);  // destination offset == file offset

  uring_ioctx ctx(1, staging.context());
  auto object = ctx.open_io_object(file.path());
  ctx.start();

  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  std::vector<std::size_t> sizes;
  for (auto const& entry : plan) {
    for (std::size_t i = 0; i < entry.count; ++i) {
      auto const offset = entry.first_offset + i * entry.bytes;
      std::vector<prepared_io_slice> slices;
      slices.emplace_back(range{offset, entry.bytes}, host_buffer{data.data() + offset});
      futures.push_back(
        ctx.mixed_readv_async_io(*object, std::move(slices), io_options{entry.cls}));
      sizes.push_back(entry.bytes);
    }
  }
  for (std::size_t i = 0; i < futures.size(); ++i) {
    CHECK(std::move(futures[i]).get(30s) == sizes[i]);
  }
  REQUIRE(wait_all_retired(ctx));
  for (auto const& entry : plan) {
    for (std::size_t i = 0; i < entry.count; ++i) {
      auto const offset = entry.first_offset + i * entry.bytes;
      CHECK(first_mismatch(std::span{data}.subspan(offset, entry.bytes), offset) == no_index);
    }
  }

  auto const before = ctx.stats();
  for (auto const cls : all_classes) {
    CAPTURE(request_class_index(cls));
    auto const& entry          = of(before, cls);
    std::size_t expected_count = 0;
    for (auto const& item : plan) {
      if (item.cls == cls) expected_count = item.count;
    }
    CHECK(entry.first_io_count == expected_count);
    CHECK(std::accumulate(entry.first_io_histogram.begin(),
                          entry.first_io_histogram.end(),
                          std::uint64_t{0}) == entry.first_io_count);
    CHECK(entry.first_io_max <= entry.first_io_total);
    CHECK(entry.first_io_max * static_cast<std::int64_t>(entry.first_io_count) >=
          entry.first_io_total);
  }

  // reset_stats_peaks sets max_inflight_ops to the current gauge: wait until
  // the runner published the drained state first.
  REQUIRE(wait_runner_drained(ctx));
  ctx.reset_stats_peaks();
  auto const after = ctx.stats();
  for (auto const cls : all_classes) {
    CAPTURE(request_class_index(cls));
    auto const& was = of(before, cls);
    auto const& now = of(after, cls);
    CHECK(now.first_io_max == std::chrono::nanoseconds{0});
    CHECK(now.max_queue_wait == std::chrono::nanoseconds{0});
    CHECK(now.first_io_count == was.first_io_count);
    CHECK(now.first_io_total == was.first_io_total);
    CHECK(now.first_io_histogram == was.first_io_histogram);
  }
  REQUIRE(after.runners.size() == 1);
  CHECK(after.runners.front().max_inflight_ops == 0);
  ctx.shutdown();
}

TEST_CASE("uring runner gauges report in-flight depth and submitted bytes",
          "[io][uring][scheduling]")
{
  // Four 16 MiB read requests (one group each), queued before the runner
  // starts so its first pull takes all four (max_active_groups).
  constexpr std::size_t n_requests  = 4;
  constexpr std::size_t n_slices    = 16;
  constexpr std::size_t slice_bytes = 1 * MiB;
  constexpr std::size_t file_bytes  = n_slices * slice_bytes;
  temp_file file(file_bytes);
  staging_resource staging;
  std::vector<std::vector<std::uint8_t>> data(n_requests, std::vector<std::uint8_t>(file_bytes));

  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());
  open_admission(ctx);

  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t r = 0; r < n_requests; ++r) {
    std::vector<prepared_io_slice> slices;
    for (std::size_t i = 0; i < n_slices; ++i) {
      slices.emplace_back(range{i * slice_bytes, slice_bytes},
                          host_buffer{data[r].data() + i * slice_bytes});
    }
    futures.push_back(
      ctx.mixed_readv_async_io(*object, std::move(slices), io_options{request_class::read}));
  }

  {
    external_runner runner(ctx);
    for (auto& future : futures) {
      CHECK(std::move(future).get(30s) == file_bytes);
    }
    REQUIRE(wait_all_retired(ctx));
    REQUIRE(wait_runner_drained(ctx));
    CHECK(ctx.active_runners() == 1);

    auto const stats = ctx.stats();
    REQUIRE(stats.runners.size() == 1);
    auto const& gauges = stats.runners.front();
    CHECK(gauges.id > 0);
    CHECK(gauges.bytes_submitted >= n_requests * file_bytes);
    // The first pass expands all four groups (slices_per_pass = 8 each: 32
    // operations) before any completion is reaped.
    CHECK(gauges.max_inflight_ops >= 16);
    CHECK(gauges.inflight_ops == 0);
    CHECK(gauges.active_groups == 0);
    CHECK(gauges.retired_groups == n_requests);
    // Nothing left to do: the runner parks.
    CHECK(wait_until(
      [&] {
        auto const now = ctx.stats();
        return now.runners.size() == 1 && now.runners.front().parked;
      },
      5000ms));
  }
  for (std::size_t r = 0; r < n_requests; ++r) {
    CHECK(first_mismatch(data[r], 0) == no_index);
  }
}

TEST_CASE("uring shutdown settles a large request with an uncapped pass", "[io][uring][scheduling]")
{
  constexpr std::size_t slice_bytes = 4 * KiB;
  constexpr std::size_t n_slices    = 65536;  // 256 MiB: 4 fan-out groups of 16384 slices
  constexpr std::size_t file_bytes  = 4 * MiB;
  constexpr std::size_t file_slices = file_bytes / slice_bytes;
  temp_file file(file_bytes);
  staging_resource staging;
  std::vector<std::uint8_t> data(file_bytes);  // slices cycle over the file: identical bytes
  std::atomic<std::size_t> completions{0};
  auto const on_complete = std::make_shared<prepared_io_completion>(
    [&completions](chunk_span, bool) noexcept { completions.fetch_add(1); });

  uring_config cfg{};
  cfg.slices_per_pass = 0;
  uring_ioctx ctx(1, staging.context(cfg));
  auto object = ctx.open_io_object(file.path());
  ctx.start();

  std::vector<prepared_io_slice> slices;
  slices.reserve(n_slices);
  for (std::size_t i = 0; i < n_slices; ++i) {
    auto const offset = (i % file_slices) * slice_bytes;
    slices.emplace_back(range{offset, slice_bytes}, host_buffer{data.data() + offset});
    slices.back().on_complete = on_complete;
  }
  auto future = ctx.mixed_readv_async_io(*object, std::move(slices));
  REQUIRE(wait_until(
    [&] { return ctx.stats().in_flight_requests != 0 || completions.load() == n_slices; }, 5000ms));

  // The future settles with the full byte count (the read won the race) or
  // operation_canceled, never with another error.
  std::size_t bytes = 0;
  std::error_code code;
  std::string other_error;
  {
    watchdog const guard(30s, "uring shutdown with a large request in flight");
    ctx.shutdown();
    try {
      bytes = std::move(future).get(30s);
    } catch (std::system_error const& e) {
      code        = e.code();
      other_error = e.what();
    } catch (std::exception const& e) {
      other_error = e.what();
    }
  }
  INFO("error: " << other_error);
  if (code) {
    CHECK(code == std::errc::operation_canceled);
  } else {
    CHECK(other_error.empty());
    CHECK(bytes == n_slices * slice_bytes);
  }
  auto const stats = ctx.stats();
  for (auto const cls : all_classes) {
    CAPTURE(request_class_index(cls));
    CHECK(of(stats, cls).queued_requests == 0);
  }
}

TEST_CASE("uring config validation rejects out-of-range values", "[io][uring][scheduling]")
{
  staging_resource staging;
  auto const with = [](auto&& change) {
    uring_config cfg{};
    change(cfg);
    return cfg;
  };

  SECTION("rejected")
  {
    CHECK_THROWS_AS(
      uring_ioctx(1, staging.context(with([](uring_config& c) { c.slices_per_pass = 65; }))),
      std::invalid_argument);
    CHECK_THROWS_AS(
      uring_ioctx(
        1, staging.context(with([](uring_config& c) { c.scheduling.max_background_groups = 0; }))),
      std::invalid_argument);
    CHECK_THROWS_AS(
      uring_ioctx(
        1, staging.context(with([](uring_config& c) { c.scheduling.max_active_groups = 0; }))),
      std::invalid_argument);
    CHECK_THROWS_AS(uring_ioctx(1, staging.context(with([](uring_config& c) {
                      c.scheduling.background_slot_fraction = 1.5;
                    }))),
                    std::invalid_argument);
    CHECK_THROWS_AS(uring_ioctx(1, staging.context(with([](uring_config& c) {
                      c.scheduling.background_slot_fraction =
                        std::numeric_limits<double>::quiet_NaN();
                    }))),
                    std::invalid_argument);
  }

  SECTION("accepted")
  {
    CHECK_NOTHROW(
      uring_ioctx(1, staging.context(with([](uring_config& c) { c.slices_per_pass = 0; }))));
    CHECK_NOTHROW(uring_ioctx(1, staging.context(with([](uring_config& c) {
      c.slices_per_pass = cucascade::io::uring::max_slices_per_pass;
    }))));
    CHECK_NOTHROW(uring_ioctx(
      1, staging.context(with([](uring_config& c) { c.scheduling.max_background_groups = 4; }))));
    // Above max_active_groups (4): behaves as max_active_groups.
    CHECK_NOTHROW(uring_ioctx(
      1, staging.context(with([](uring_config& c) { c.scheduling.max_background_groups = 5; }))));
  }

  SECTION("max_active_groups = 1 alone is accepted and still serves background reads")
  {
    // The default max_background_groups (2) is clamped to 1.
    std::vector<std::uint8_t> bg_data(1 * MiB);
    std::vector<std::uint8_t> read_data(1 * MiB);
    temp_file file(2 * MiB);
    std::unique_ptr<uring_ioctx> ctx;
    REQUIRE_NOTHROW(
      ctx = std::make_unique<uring_ioctx>(
        1, staging.context(with([](uring_config& c) { c.scheduling.max_active_groups = 1; }))));
    auto object = ctx->open_io_object(file.path());
    ctx->start();
    std::vector<prepared_io_slice> bg;
    bg.emplace_back(range{0, 1 * MiB}, host_buffer{bg_data.data()});
    std::vector<prepared_io_slice> rd;
    rd.emplace_back(range{1 * MiB, 1 * MiB}, host_buffer{read_data.data()});
    auto bg_future =
      ctx->mixed_readv_async_io(*object, std::move(bg), io_options{request_class::background});
    auto rd_future =
      ctx->mixed_readv_async_io(*object, std::move(rd), io_options{request_class::read});
    CHECK(std::move(bg_future).get(10s) == 1 * MiB);
    CHECK(std::move(rd_future).get(10s) == 1 * MiB);
    ctx->shutdown();
    CHECK(first_mismatch(bg_data, 0) == no_index);
    CHECK(first_mismatch(read_data, 1 * MiB) == no_index);
  }
}

TEST_CASE("uring background staged device reads fit the background reservation",
          "[io][uring][scheduling][gpu]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    static_cast<void>(cudaGetLastError());
    SKIP("no CUDA device");
  }

  // One 64 MiB staged slice is planned at once (one plan_next) into operations
  // sized by dynamic_io_target over the free slots the group may use.
  constexpr std::size_t read_bytes = 64 * MiB;
  // Demand: up to max_dynamic_io_size (16 MiB = 16 slots) per operation; the
  // four operations fill the 64 slots in one pass.
  constexpr std::uint32_t read_peak = 4;
  // Background: operations are fitted to the background reservation
  // (min(reserved_background_slots 8, 64 / 4, share 48) = 8 slots = 8 MiB), so
  // they fit the slots demand may not take; the 48-slot share admits 6 at once.
  // Unfitted 16 MiB operations would peak at 3.
  constexpr std::uint32_t background_peak = 6;
  temp_file file(read_bytes);
  staging_resource staging;

  struct device_allocation {
    std::uint8_t* data{nullptr};
    cudaStream_t stream{nullptr};
    ~device_allocation()
    {
      if (stream != nullptr) static_cast<void>(cudaStreamDestroy(stream));
      if (data != nullptr) static_cast<void>(cudaFree(data));
    }
  } device;  // outlives the context
  REQUIRE(cudaMalloc(&device.data, read_bytes) == cudaSuccess);
  REQUIRE(cudaStreamCreate(&device.stream) == cudaSuccess);

  uring_ioctx ctx(1, staging.context());
  auto object = ctx.open_io_object(file.path());
  ctx.start();

  auto const peak_of = [&](request_class cls) {
    // Fresh peak: wait for the drained state the reset copies into the peak.
    REQUIRE(wait_runner_drained(ctx));
    ctx.reset_stats_peaks();
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(range{0, read_bytes},
                        device_buffer{device.data, ::cuda::stream_ref{device.stream}});
    CHECK(ctx.mixed_readv_async_io(*object, std::move(slices), io_options{cls}).get(30s) ==
          read_bytes);
    auto const stats = ctx.stats();
    REQUIRE(stats.runners.size() == 1);
    return stats.runners.front().max_inflight_ops;
  };
  CHECK(peak_of(request_class::read) == read_peak);
  CHECK(peak_of(request_class::background) == background_peak);
  ctx.shutdown();

  REQUIRE(cudaStreamSynchronize(device.stream) == cudaSuccess);
  std::vector<std::uint8_t> host(read_bytes);
  REQUIRE(cudaMemcpy(host.data(), device.data, read_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(first_mismatch(host, 0) == no_index);
}
