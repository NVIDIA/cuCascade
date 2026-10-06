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

#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <sys/uio.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

using cucascade::io::file_descriptor;
using cucascade::io::grouped_coordinator;
using cucascade::io::grouped_io_request;
using cucascade::io::host_buffer;
using cucascade::io::IO_BLOCK_SIZE;
using cucascade::io::prepared_io_completion;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::cache::cached_chunk;
using cucascade::io::uring::local_io_object;
using cucascade::io::uring::max_dynamic_io_size;
using cucascade::io::uring::min_dynamic_io_size;
using cucascade::io::uring::uring_io_op;
using cucascade::io::uring::uring_reactor;
using cucascade::io::uring::detail::dynamic_io_target;
using cucascade::io::uring::detail::fill_remaining_iovecs;
using cucascade::io::uring::detail::is_odirect_compatible;
using cucascade::io::uring::detail::is_odirect_runtime_error;
using cucascade::io::uring::detail::odirect_available;

namespace {

class aligned_bytes {
 public:
  explicit aligned_bytes(std::size_t bytes)
    : _data(static_cast<std::uint8_t*>(std::aligned_alloc(IO_BLOCK_SIZE, bytes)))
  {
    REQUIRE(_data != nullptr);
  }

  ~aligned_bytes() { std::free(_data); }

  aligned_bytes(aligned_bytes const&)            = delete;
  aligned_bytes& operator=(aligned_bytes const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return _data; }

 private:
  std::uint8_t* _data;
};

/** Consume @p future and report whether it carries the reactor's cancellation error. */
bool canceled(cucascade::exec::semi_future<std::size_t>&& future) noexcept
{
  auto result = std::move(future).get_try();
  try {
    std::move(result).get();
  } catch (std::system_error const& error) {
    return error.code() == std::errc::operation_canceled;
  } catch (...) {
  }
  return false;
}

/** The byte a @ref pattern_file holds at @p offset. */
[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  return static_cast<std::uint8_t>((offset * 2654435761ULL) >> 11);
}

/** A temporary file of @p bytes filled with @ref pattern_byte, removed on scope exit. */
class pattern_file {
 public:
  explicit pattern_file(std::size_t bytes)
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_uring_spp_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin"))
  {
    std::vector<char> data(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
      data[i] = static_cast<char>(pattern_byte(i));
    }
    std::ofstream out(_path, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    out.close();
    REQUIRE(out);
  }

  ~pattern_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }

  pattern_file(pattern_file const&)            = delete;
  pattern_file& operator=(pattern_file const&) = delete;

  [[nodiscard]] std::shared_ptr<cucascade::io::io_object const> open() const
  {
    return uring_reactor::create_io_object(_path.string());
  }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
};

/**
 * A started reactor whose staging is 64 one-MiB slots (the slot cap), reading
 * with @p slices_per_pass.
 */
class spp_reactor {
 public:
  static constexpr std::size_t block_size = 1UL << 20;
  static constexpr std::size_t capacity   = 128UL << 20;

  explicit spp_reactor(std::size_t slices_per_pass)
    : _mr{0, _upstream, capacity, capacity, block_size, 64, 0},
      _reactor{std::make_shared<uring_reactor::reactor_context>(config_for(slices_per_pass), &_mr),
               ""}
  {
    _reactor.start();
  }

  ~spp_reactor() { _reactor.shutdown(); }

  spp_reactor(spp_reactor const&)            = delete;
  spp_reactor& operator=(spp_reactor const&) = delete;

  [[nodiscard]] uring_reactor& get() noexcept { return _reactor; }

 private:
  [[nodiscard]] static cucascade::io::uring::config config_for(std::size_t slices_per_pass)
  {
    cucascade::io::uring::config cfg{};
    cfg.slices_per_pass = slices_per_pass;
    return cfg;
  }

  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
  uring_reactor _reactor;
};

/** One contiguous slice per element: file range and the destination offset it lands at. */
[[nodiscard]] std::vector<range> slice_ranges(std::size_t count, std::size_t stride, bool ragged)
{
  std::vector<range> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    // Odd slices of a ragged layout are unaligned at both ends, so they take the
    // buffered fallback while their aligned neighbours may use O_DIRECT.
    std::size_t const head = ragged && (i % 2 == 1) ? 7 : 0;
    std::size_t const tail = ragged && (i % 2 == 1) ? 5 : 0;
    out.push_back(range{i * stride + head, stride - head - tail});
  }
  return out;
}

/** Enqueue one grouped request reading @p ranges into @p destination at the same offsets. */
[[nodiscard]] cucascade::exec::semi_future<std::size_t> enqueue_ranges(
  uring_reactor& reactor,
  std::shared_ptr<cucascade::io::io_object const> object,
  std::vector<range> const& ranges,
  std::uint8_t* destination,
  cucascade::io::io_class cls = cucascade::io::io_class::demand)
{
  std::size_t bytes = 0;
  std::vector<prepared_io_slice> slices;
  slices.reserve(ranges.size());
  for (auto const& r : ranges) {
    slices.emplace_back(r, host_buffer{destination + r.offset});
    slices.back().cls = cls;
    bytes += r.size;
  }
  auto coordinator = std::make_shared<grouped_coordinator>(bytes, slices.size());
  auto future      = coordinator->get_future();
  reactor.enqueue(
    grouped_io_request::create(std::move(object), std::move(slices), std::move(coordinator)));
  return future;
}

}  // namespace

static_assert(!std::is_move_constructible_v<uring_io_op>);
static_assert(!std::is_move_assignable_v<uring_io_op>);

TEST_CASE("io_uring dynamic chunk target is backlog and slot aware", "[io][uring]")
{
  constexpr std::size_t block = 64UL << 10;

  CHECK(dynamic_io_target(1, 64, block) == min_dynamic_io_size);
  CHECK(dynamic_io_target(2UL << 20, 64, block) == 2UL << 20);
  CHECK(dynamic_io_target(64UL << 20, 64, block) == 4UL << 20);
  CHECK(dynamic_io_target(64UL << 20, 512, block) == max_dynamic_io_size);
  CHECK(dynamic_io_target(64UL << 20, 0, block) == 0);
  CHECK(dynamic_io_target(64UL << 20, 64, 0) == 0);
}

TEST_CASE("io_uring dynamic target honors fixed blocks and the physical cap", "[io][uring]")
{
  constexpr std::size_t block = 1UL << 20;

  CHECK(dynamic_io_target(1, 64, block) == block);
  CHECK(dynamic_io_target(3 * block + 1, 64, block) == 4 * block);
  CHECK(dynamic_io_target(max_dynamic_io_size, 3, block) == 3 * block);
  CHECK(dynamic_io_target(max_dynamic_io_size, 1, 2 * max_dynamic_io_size) == max_dynamic_io_size);
}

TEST_CASE("io_uring O_DIRECT validation covers the entire readv", "[io][uring]")
{
  aligned_bytes first{2 * IO_BLOCK_SIZE};
  aligned_bytes second{IO_BLOCK_SIZE};
  std::array<iovec, 2> buffers{iovec{first.get(), 2 * IO_BLOCK_SIZE},
                               iovec{second.get(), IO_BLOCK_SIZE}};

  CHECK(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE}, buffers));
  CHECK_FALSE(is_odirect_compatible(range{1, 3 * IO_BLOCK_SIZE}, buffers));
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE - 1}, buffers));

  buffers[1].iov_base = second.get() + 1;
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE}, buffers));
  buffers[1].iov_base = second.get();
  buffers[1].iov_len -= 1;
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE - 1}, buffers));
}

TEST_CASE("io_uring falls back when O_DIRECT is unavailable or rejected", "[io][uring]")
{
  CHECK(odirect_available(true, 3));
  CHECK_FALSE(odirect_available(false, 3));
  CHECK_FALSE(odirect_available(true, -1));
  CHECK(is_odirect_runtime_error(EINVAL));
  CHECK(is_odirect_runtime_error(EOPNOTSUPP));
  CHECK_FALSE(is_odirect_runtime_error(EIO));
}

TEST_CASE("io_uring readv resume preserves byte order after short reads", "[io][uring]")
{
  std::array<std::uint8_t, 600> storage{};
  std::array<iovec, 3> source{
    iovec{storage.data(), 100}, iovec{storage.data() + 100, 200}, iovec{storage.data() + 300, 300}};
  std::vector<iovec> remaining;

  SECTION("exact iovec boundary")
  {
    fill_remaining_iovecs(source, 100, remaining);
    REQUIRE(remaining.size() == 2);
    CHECK(remaining[0].iov_base == storage.data() + 100);
    CHECK(remaining[0].iov_len == 200);
  }

  SECTION("middle of an iovec")
  {
    fill_remaining_iovecs(source, 150, remaining);
    REQUIRE(remaining.size() == 2);
    CHECK(remaining[0].iov_base == storage.data() + 150);
    CHECK(remaining[0].iov_len == 150);
    CHECK(remaining[1].iov_base == storage.data() + 300);
    CHECK(remaining[1].iov_len == 300);
  }

  SECTION("all bytes consumed")
  {
    fill_remaining_iovecs(source, 600, remaining);
    CHECK(remaining.empty());
  }
}

TEST_CASE("io_uring cache callback precedes its coordinator credit", "[io][uring]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  cached_chunk chunk{0};

  std::size_t callback_count = 0;
  std::size_t credits_seen   = 0;
  bool callback_success      = false;
  cached_chunk* observed     = nullptr;
  auto completion            = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const> chunks, bool success) noexcept {
      ++callback_count;
      credits_seen     = coordinator->tasks_remaining();
      callback_success = success;
      if (chunks.size() == 1) observed = chunks.front();
    });

  auto op                 = std::make_unique<uring_io_op>();
  op->request.coordinator = coordinator;
  op->request.on_complete = completion;
  op->request.completion_chunks.push_back(&chunk);

  op->request.finish_success();
  op->request.finish_success();

  CHECK(callback_count == 1);
  CHECK(credits_seen == 1);
  CHECK(callback_success);
  CHECK(observed == &chunk);
  CHECK(std::move(future).get() == IO_BLOCK_SIZE);
}

TEST_CASE("io_uring physical expansion publishes cache fragments independently", "[io][uring]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(2 * IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  coordinator->add_tasks(1);

  cached_chunk first{0};
  cached_chunk second{IO_BLOCK_SIZE};
  std::array<cached_chunk*, 2> published{};
  std::size_t published_count = 0;
  bool callbacks_valid        = true;

  auto completion = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const> chunks, bool success) noexcept {
      callbacks_valid =
        callbacks_valid && success && chunks.size() == 1 && published_count < published.size();
      if (chunks.size() == 1 && published_count < published.size()) {
        published[published_count++] = chunks.front();
      }
    });

  auto make_op = [&](cached_chunk* chunk) {
    auto op                 = std::make_unique<uring_io_op>();
    op->request.coordinator = coordinator;
    op->request.on_complete = completion;
    op->request.completion_chunks.push_back(chunk);
    return op;
  };

  auto first_op  = make_op(&first);
  auto second_op = make_op(&second);
  first_op->request.finish_success();

  CHECK(coordinator->tasks_remaining() == 1);
  REQUIRE(published_count == 1);
  CHECK(published.front() == &first);

  second_op->request.finish_success();
  CHECK(callbacks_valid);
  REQUIRE(published_count == 2);
  CHECK(published.back() == &second);
  CHECK(std::move(future).get() == 2 * IO_BLOCK_SIZE);
}

TEST_CASE("io_uring CUDA-stage failure may publish valid host data before error", "[io][uring]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  bool host_valid  = false;

  auto completion = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const>, bool success) noexcept { host_valid = success; });

  auto op                 = std::make_unique<uring_io_op>();
  op->request.coordinator = coordinator;
  op->request.on_complete = completion;
  op->request.finish_error(cudaErrorInvalidValue, true);

  CHECK(host_valid);
  CHECK_THROWS(std::move(future).get());
}

TEST_CASE("io_uring alignment widens and coalesces safely", "[io][uring]")
{
  using range_info = cucascade::io::byte_range;

  auto physical = uring_reactor::align_to_physical(range_info{5000, 9000}, 20'000);
  CHECK(physical.offset() == 4096);
  CHECK(physical.size() == 12'288);

  std::array<range_info, 4> input{
    range_info{8193, 100}, range_info{1, 100}, range_info{4095, 2}, range_info{0, 0}};
  auto merged = uring_reactor::align_and_coalesce(input);

  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 12'288);
}

TEST_CASE("io_uring staging reservation follows the byte budget, not the slot cap", "[io][uring]")
{
  // 4 MiB blocks: the pre-budget sizing asked for 64 x 4 MiB = 256 MiB, which does not
  // fit in this resource; the 64 MiB budget asks for 16 blocks, which does.
  constexpr std::size_t block_size = 4UL << 20;
  constexpr std::size_t capacity   = 128UL << 20;

  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource mr{
    0, upstream, capacity, capacity, block_size, 4, 0};

  auto ctx = std::make_shared<uring_reactor::reactor_context>(cucascade::io::uring::config{}, &mr);
  uring_reactor reactor{ctx, ""};

  REQUIRE_NOTHROW(reactor.start());
  CHECK(mr.get_total_allocated_bytes() == (64UL << 20));
  reactor.shutdown();
}

TEST_CASE("io_uring staging failure names the reactor and the requested bytes", "[io][uring]")
{
  // A block larger than the budget clamps the slot count to one; the resource cannot
  // serve even that single block.
  constexpr std::size_t block_size = 128UL << 20;

  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource mr{
    0, upstream, 1UL << 20, 1UL << 20, block_size, 1, 0};

  auto ctx = std::make_shared<uring_reactor::reactor_context>(cucascade::io::uring::config{}, &mr);
  uring_reactor reactor{ctx, ""};

  REQUIRE_THROWS_WITH(
    reactor.start(),
    Catch::Matchers::ContainsSubstring(
      "uring_reactor: cannot reserve 134217728 bytes of pinned staging (1 x 134217728)"));
}

TEST_CASE("io_uring cancels rejected work outside the enqueue lock", "[io][uring]")
{
  // cancel_remaining() runs the slice callback and settles the coordinator inline, and a
  // continuation may submit another read on the same reactor: enqueue() must have released
  // _enqueue_mutex before it cancels, otherwise the re-entry relocks it on this thread.
  constexpr std::size_t block_size = 4UL << 20;
  constexpr std::size_t capacity   = 128UL << 20;

  auto destination    = std::make_shared<std::array<std::uint8_t, 1>>();
  auto outer_canceled = std::make_shared<bool>(false);
  auto inner_canceled = std::make_shared<bool>(false);
  auto finished       = std::make_shared<std::promise<void>>();
  auto watchdog       = finished->get_future();

  // Detached so that a re-entrant self-deadlock fails the watchdog instead of hanging the
  // whole test binary.
  std::thread([destination, outer_canceled, inner_canceled, finished] {
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, capacity, capacity, block_size, 4, 0};

    auto ctx =
      std::make_shared<uring_reactor::reactor_context>(cucascade::io::uring::config{}, &mr);
    uring_reactor reactor{ctx, ""};
    reactor.start();
    reactor.shutdown();

    auto submit = [&](std::shared_ptr<prepared_io_completion> on_complete) {
      auto object =
        std::make_shared<local_io_object>("/dev/null", file_descriptor{}, file_descriptor{}, 1);
      auto coordinator = std::make_shared<grouped_coordinator>(1, 1);
      auto future      = coordinator->get_future();
      std::vector<prepared_io_slice> slices;
      slices.emplace_back(range{0, 1}, host_buffer{destination->data()});
      slices.back().on_complete = std::move(on_complete);
      reactor.enqueue(
        grouped_io_request::create(std::move(object), std::move(slices), std::move(coordinator)));
      return future;
    };

    auto reenter =
      std::make_shared<prepared_io_completion>([&](std::span<cached_chunk* const>, bool) noexcept {
        *inner_canceled = canceled(submit(nullptr));
      });

    *outer_canceled = canceled(submit(std::move(reenter)));
    finished->set_value();
  }).detach();

  REQUIRE(watchdog.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  CHECK(*outer_canceled);
  CHECK(*inner_canceled);
}

TEST_CASE("io_uring slices_per_pass sets how deep a single request queues",
          "[io][uring][slices_per_pass]")
{
  constexpr std::size_t n_slices = 64;
  constexpr std::size_t stride   = 64UL << 10;
  pattern_file const file{n_slices * stride};
  auto const ranges = slice_ranges(n_slices, stride, false);

  auto max_inflight = [&](std::size_t slices_per_pass) {
    spp_reactor reactor{slices_per_pass};
    aligned_bytes destination{n_slices * stride};
    auto future = enqueue_ranges(reactor.get(), file.open(), ranges, destination.get());
    CHECK(std::move(future).get(std::chrono::seconds(30)) == n_slices * stride);
    // The worker publishes its depth after submitting and before reaping, so the
    // window cannot miss the deepest pass.
    return reactor.get().take_gauges().max_inflight_ops;
  };

  // One slice per pass: every pass submits one read and then waits for a completion.
  CHECK(max_inflight(1) <= 2);
  // A cap of 8 submits 8 reads in the very first pass.
  CHECK(max_inflight(8) >= 8);
  // No cap: the first pass fills the 64 free slots.
  CHECK(max_inflight(0) >= 32);
}

TEST_CASE("io_uring queue-delay gauge counts each request once, per io_class",
          "[io][uring][queue_delay]")
{
  using cucascade::io::io_class;
  constexpr std::size_t n_slices = 16;
  constexpr std::size_t stride   = 64UL << 10;
  pattern_file const file{n_slices * stride};
  auto const ranges = slice_ranges(n_slices, stride, false);

  spp_reactor reactor{8};
  aligned_bytes prefetch_dst{n_slices * stride};
  aligned_bytes demand_dst{n_slices * stride};
  auto prefetch =
    enqueue_ranges(reactor.get(), file.open(), ranges, prefetch_dst.get(), io_class::prefetch);
  auto demand_a = enqueue_ranges(reactor.get(), file.open(), ranges, demand_dst.get());
  auto demand_b = enqueue_ranges(reactor.get(), file.open(), ranges, demand_dst.get());
  CHECK(std::move(prefetch).get(std::chrono::seconds(30)) == n_slices * stride);
  CHECK(std::move(demand_a).get(std::chrono::seconds(30)) == n_slices * stride);
  CHECK(std::move(demand_b).get(std::chrono::seconds(30)) == n_slices * stride);

  auto const g    = reactor.get().take_gauges();
  auto const& d   = g.queue_delay[static_cast<std::size_t>(io_class::demand)];
  auto const& p   = g.queue_delay[static_cast<std::size_t>(io_class::prefetch)];
  auto hist_total = [](auto const& s) {
    std::uint64_t n = 0;
    for (auto c : s.histogram)
      n += c;
    return n;
  };
  CHECK(d.count == 2);
  CHECK(p.count == 1);
  CHECK(hist_total(d) == d.count);
  CHECK(hist_total(p) == p.count);
  CHECK(d.max_ns <= d.sum_ns);
  CHECK(d.max_ns * d.count >= d.sum_ns);

  // The window resets: nothing new was expanded since the previous take.
  auto const again = reactor.get().take_gauges();
  CHECK(again.queue_delay[0].count == 0);
  CHECK(again.queue_delay[1].count == 0);
  CHECK(again.queue_delay[0].max_ns == 0);
}

TEST_CASE("io_uring slices_per_pass never changes the bytes read", "[io][uring][slices_per_pass]")
{
  // More slices than slots, so the steady state (refilling slots freed by
  // completions) runs too; odd slices are unaligned and take the buffered path.
  constexpr std::size_t n_slices   = 200;
  constexpr std::size_t stride     = 64UL << 10;
  constexpr std::size_t total      = n_slices * stride;
  constexpr std::uint8_t untouched = 0xA5;
  pattern_file const file{total};
  auto const ranges    = slice_ranges(n_slices, stride, true);
  std::size_t expected = 0;
  for (auto const& r : ranges) {
    expected += r.size;
  }

  for (std::size_t const slices_per_pass : {std::size_t{1}, std::size_t{8}, std::size_t{0}}) {
    CAPTURE(slices_per_pass);
    spp_reactor reactor{slices_per_pass};
    aligned_bytes destination{total};
    std::memset(destination.get(), untouched, total);

    auto future = enqueue_ranges(reactor.get(), file.open(), ranges, destination.get());
    REQUIRE(std::move(future).get(std::chrono::seconds(30)) == expected);

    // Every requested byte matches the file; every byte between slices is untouched.
    std::size_t wrong = 0;
    std::size_t next  = 0;
    auto const* bytes = destination.get();
    for (auto const& r : ranges) {
      for (; next < r.offset; ++next) {
        wrong += bytes[next] != untouched ? 1 : 0;
      }
      for (; next < r.end(); ++next) {
        wrong += bytes[next] != pattern_byte(next) ? 1 : 0;
      }
    }
    for (; next < total; ++next) {
      wrong += bytes[next] != untouched ? 1 : 0;
    }
    CHECK(wrong == 0);
  }
}

TEST_CASE("io_uring shutdown settles a large active request with an uncapped pass",
          "[io][uring][slices_per_pass]")
{
  // 65536 x 4 KiB slices (256 MiB, cycling over a 4 MiB file) take far longer to
  // expand than it takes to call shutdown(), so the request is still active then.
  static constexpr std::size_t file_bytes  = 4UL << 20;
  static constexpr std::size_t slice_bytes = 4UL << 10;
  static constexpr std::size_t n_slices    = 65536;
  static constexpr std::size_t file_blocks = file_bytes / slice_bytes;

  enum class settled_as { value, canceled, other_error };
  struct outcome_t {
    settled_as settled{settled_as::other_error};
    bool started{false};
    std::size_t queued_bytes{0};
  };

  auto file        = std::make_shared<pattern_file>(file_bytes);
  auto destination = std::make_shared<aligned_bytes>(n_slices * slice_bytes);
  auto outcome     = std::make_shared<outcome_t>();
  auto finished    = std::make_shared<std::promise<void>>();
  auto watchdog    = finished->get_future();

  // Detached so a hung shutdown fails the watchdog instead of hanging the binary.
  std::thread([file, destination, outcome, finished] {
    spp_reactor reactor{0};

    std::vector<prepared_io_slice> slices;
    slices.reserve(n_slices);
    for (std::size_t i = 0; i < n_slices; ++i) {
      slices.emplace_back(range{(i % file_blocks) * slice_bytes, slice_bytes},
                          host_buffer{destination->get() + i * slice_bytes});
    }
    auto coordinator = std::make_shared<grouped_coordinator>(n_slices * slice_bytes, n_slices);
    auto future      = coordinator->get_future();
    reactor.get().enqueue(
      grouped_io_request::create(file->open(), std::move(slices), std::move(coordinator)));

    // Shut down only once the worker has taken the request, so it is active.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (reactor.get().take_gauges().requests_started != 0) {
        outcome->started = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    reactor.get().shutdown();

    // shutdown() joined the worker, which settles everything it owned on the way out.
    auto result = std::move(future).get_try();
    if (result.has_value()) {
      outcome->settled = settled_as::value;
    } else {
      try {
        std::move(result).get();
      } catch (std::system_error const& error) {
        if (error.code() == std::errc::operation_canceled) {
          outcome->settled = settled_as::canceled;
        }
      } catch (...) {
      }
    }
    outcome->queued_bytes = reactor.get().queued_bytes();
    finished->set_value();
  }).detach();

  REQUIRE(watchdog.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  CHECK(outcome->started);
  // Either the reads beat the shutdown or the remainder was canceled; never another error.
  CHECK(outcome->settled != settled_as::other_error);
  CHECK(outcome->queued_bytes == 0);
}
