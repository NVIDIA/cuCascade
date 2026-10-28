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
#include <cucascade/io/uring/uring_ioctx.hpp>
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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
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
using cucascade::io::uring::uring_ioctx;
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

  [[nodiscard]] std::string path() const { return _path.string(); }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
};

/**
 * A started reactor whose staging is 64 one-MiB slots (the slot cap), reading
 * with @p slices_per_pass and @p low_service_interval.
 */
class spp_reactor {
 public:
  static constexpr std::size_t block_size = 1UL << 20;
  static constexpr std::size_t capacity   = 128UL << 20;

  explicit spp_reactor(
    std::size_t slices_per_pass,
    std::size_t low_service_interval = cucascade::io::uring::config{}.low_service_interval)
    : _mr{0, _upstream, capacity, capacity, block_size, 64, 0},
      _reactor{std::make_shared<uring_reactor::reactor_context>(
                 config_for(slices_per_pass, low_service_interval), &_mr),
               ""}
  {
    _reactor.start();
  }

  ~spp_reactor() { _reactor.shutdown(); }

  spp_reactor(spp_reactor const&)            = delete;
  spp_reactor& operator=(spp_reactor const&) = delete;

  [[nodiscard]] uring_reactor& get() noexcept { return _reactor; }

 private:
  [[nodiscard]] static cucascade::io::uring::config config_for(std::size_t slices_per_pass,
                                                               std::size_t low_service_interval)
  {
    cucascade::io::uring::config cfg{};
    cfg.slices_per_pass      = slices_per_pass;
    cfg.low_service_interval = low_service_interval;
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
  cucascade::io::io_priority priority                   = cucascade::io::io_priority::high,
  std::shared_ptr<grouped_coordinator>* coordinator_out = nullptr)
{
  std::size_t bytes = 0;
  std::vector<prepared_io_slice> slices;
  slices.reserve(ranges.size());
  for (auto const& r : ranges) {
    slices.emplace_back(r, host_buffer{destination + r.offset});
    slices.back().priority = priority;
    bytes += r.size;
  }
  auto coordinator = std::make_shared<grouped_coordinator>(bytes, slices.size());
  auto future      = coordinator->get_future();
  if (coordinator_out != nullptr) *coordinator_out = coordinator;
  reactor.enqueue(
    grouped_io_request::create(std::move(object), std::move(slices), std::move(coordinator)));
  return future;
}

/**
 * Holds reactor workers inside a read's completion callback until released, so a
 * test can fill a reactor's queue while its worker provably takes nothing from it.
 * The hold gives up after 30 s so a failing test cannot wedge a worker forever;
 * destroying the plug releases it (declare it after the reactor it holds).
 */
class worker_plug {
 public:
  worker_plug() = default;
  ~worker_plug() { release(); }

  worker_plug(worker_plug const&)            = delete;
  worker_plug& operator=(worker_plug const&) = delete;

  /// A completion that holds the worker that runs it.
  [[nodiscard]] std::shared_ptr<prepared_io_completion> completion() const
  {
    return std::make_shared<prepared_io_completion>(
      [state = _state](std::span<cached_chunk* const>, bool) noexcept {
        state->entered.fetch_add(1);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!state->released.load() && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });
  }

  /// Wait until @p count workers are held; false on timeout.
  [[nodiscard]] bool wait_held(std::size_t count) const
  {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (_state->entered.load() < count) {
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
  }

  void release() noexcept { _state->released.store(true); }

 private:
  struct state {
    std::atomic<std::size_t> entered{0};
    std::atomic<bool> released{false};
  };
  std::shared_ptr<state> _state = std::make_shared<state>();
};

/// Enqueue a one-block high read whose completion is @p plug's hold.
[[nodiscard]] cucascade::exec::semi_future<std::size_t> enqueue_plug(
  uring_reactor& reactor,
  std::shared_ptr<cucascade::io::io_object const> object,
  std::uint8_t* destination,
  worker_plug const& plug)
{
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, IO_BLOCK_SIZE}, host_buffer{destination});
  slices.back().priority    = cucascade::io::io_priority::high;
  slices.back().on_complete = plug.completion();
  auto coordinator          = std::make_shared<grouped_coordinator>(IO_BLOCK_SIZE, 1);
  auto future               = coordinator->get_future();
  reactor.enqueue(
    grouped_io_request::create(std::move(object), std::move(slices), std::move(coordinator)));
  return future;
}

/// How a future settled within a deadline.
struct settle_outcome {
  bool ready{false};        ///< settled before the deadline
  std::size_t bytes{0};     ///< its value, when it succeeded
  std::error_code error{};  ///< its std::system_error code, when it failed with one
  bool other_error{false};  ///< failed with any other exception
};

[[nodiscard]] settle_outcome settle_within(cucascade::exec::semi_future<std::size_t>&& future,
                                           std::chrono::seconds timeout)
{
  settle_outcome out;
  try {
    out.bytes = std::move(future).get(timeout);
    out.ready = true;
  } catch (std::system_error const& error) {
    out.ready = true;
    out.error = error.code();
  } catch (std::runtime_error const& error) {
    // get() reports a timeout as a plain runtime_error.
    out.ready       = std::string_view{error.what()}.find("timed out") == std::string_view::npos;
    out.other_error = out.ready;
  } catch (...) {
    out.ready       = true;
    out.other_error = true;
  }
  return out;
}

/// Wait until @p reactor has parked a low request for a high one; false on timeout.
[[nodiscard]] bool wait_for_preemption(uring_reactor& reactor)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (reactor.take_gauges().preemptions == 0) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

/// Stop @p coordinator the way a caller cancelling its read does: the extra credit
/// keeps the cancellation from consuming a credit that belongs to a slice.
void cancel_coordinator(grouped_coordinator& coordinator)
{
  coordinator.add_tasks(1);
  coordinator.report_error(std::make_error_code(std::errc::interrupted));
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
  // The body reports any exception through the watchdog: one escaping a thread terminates.
  std::thread([destination, outer_canceled, inner_canceled, finished] {
    try {
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

      auto reenter = std::make_shared<prepared_io_completion>(
        [&](std::span<cached_chunk* const>, bool) noexcept {
          *inner_canceled = canceled(submit(nullptr));
        });

      *outer_canceled = canceled(submit(std::move(reenter)));
      finished->set_value();
    } catch (...) {
      finished->set_exception(std::current_exception());
    }
  }).detach();

  REQUIRE(watchdog.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  REQUIRE_NOTHROW(watchdog.get());
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
    // Destination first: it must outlive the reactor, which settles in-flight reads into
    // it on shutdown.
    aligned_bytes destination{n_slices * stride};
    spp_reactor reactor{slices_per_pass};
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

TEST_CASE("io_uring queue-delay gauge counts each request once, per priority tier",
          "[io][uring][queue_delay]")
{
  using cucascade::io::io_priority;
  constexpr std::size_t n_slices = 16;
  constexpr std::size_t stride   = 64UL << 10;
  pattern_file const file{n_slices * stride};
  auto const ranges = slice_ranges(n_slices, stride, false);

  // Destinations first: they must outlive the reactor (see the slices_per_pass test).
  aligned_bytes prefetch_dst{n_slices * stride};
  aligned_bytes demand_dst{n_slices * stride};
  spp_reactor reactor{8};
  auto prefetch =
    enqueue_ranges(reactor.get(), file.open(), ranges, prefetch_dst.get(), io_priority::low);
  auto demand_a = enqueue_ranges(reactor.get(), file.open(), ranges, demand_dst.get());
  auto demand_b = enqueue_ranges(reactor.get(), file.open(), ranges, demand_dst.get());
  CHECK(std::move(prefetch).get(std::chrono::seconds(30)) == n_slices * stride);
  CHECK(std::move(demand_a).get(std::chrono::seconds(30)) == n_slices * stride);
  CHECK(std::move(demand_b).get(std::chrono::seconds(30)) == n_slices * stride);

  // [0] = high tier, [1] = low tier.
  auto const g    = reactor.get().take_gauges();
  auto const& d   = g.queue_delay_for(io_priority::high);
  auto const& p   = g.queue_delay_for(io_priority::low);
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
  CHECK(again.queue_delay_high.count == 0);
  CHECK(again.queue_delay_low.count == 0);
  CHECK(again.queue_delay_high.max_ns == 0);
}

TEST_CASE("io_uring parks an active low request for a high one and resumes it",
          "[io][uring][priority]")
{
  using cucascade::io::io_priority;
  // A long low request at one slice per pass, so it is still active when the
  // high request lands behind it.
  constexpr std::size_t n_slices = 512;
  constexpr std::size_t stride   = 64UL << 10;
  constexpr std::size_t total    = n_slices * stride;
  pattern_file const file{total};
  auto const ranges = slice_ranges(n_slices, stride, false);

  // Destinations first: they must outlive the reactor.
  aligned_bytes low_dst{total};
  aligned_bytes high_dst{total};
  spp_reactor reactor{1};
  auto low = enqueue_ranges(reactor.get(), file.open(), ranges, low_dst.get(), io_priority::low);
  // Wait until the worker has taken the low request; otherwise the high one could
  // simply be dequeued first, with nothing to preempt.
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (reactor.get().take_gauges().requests_started == 0) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  std::vector<range> const one{ranges.back()};
  auto high = enqueue_ranges(reactor.get(), file.open(), one, high_dst.get(), io_priority::high);

  CHECK(std::move(high).get(std::chrono::seconds(30)) == stride);
  CHECK(std::move(low).get(std::chrono::seconds(30)) == total);
  CHECK(reactor.get().take_gauges().preemptions >= 1);

  // Parking loses nothing: both destinations hold the file's bytes.
  for (std::size_t i = 0; i < total; i += 4093) {
    REQUIRE(low_dst.get()[i] == pattern_byte(i));
  }
  for (std::size_t i = ranges.back().offset; i < total; ++i) {
    REQUIRE(high_dst.get()[i] == pattern_byte(i));
  }
}

namespace {

/// Outcome of one low request raced against a backlog of high ones.
struct low_vs_high_outcome {
  bool low_ok{false};
  /// High requests that had completed when the low one completed.
  std::size_t highs_before_low{0};
  std::uint64_t grants{0};
};

/**
 * Queue @p n_high high requests of @p high_slices slices each and then one
 * @p n_low_slices-slice low request on a reactor expanding one slice per pass,
 * all while its worker is held (see @ref worker_plug), so the worker first sees
 * the whole high backlog with the low request behind it.  Report how many high
 * requests finished before the low one.
 */
low_vs_high_outcome race_low_against_high_backlog(std::size_t low_service_interval,
                                                  std::size_t n_low_slices,
                                                  std::size_t n_high,
                                                  std::size_t high_slices)
{
  using cucascade::io::io_priority;
  constexpr std::size_t stride = 64UL << 10;
  std::size_t const n_slices   = n_low_slices + high_slices;
  pattern_file const file{n_slices * stride};
  auto const all = slice_ranges(n_slices, stride, false);
  std::vector<range> const low_ranges(all.begin(),
                                      all.begin() + static_cast<std::ptrdiff_t>(n_low_slices));
  std::vector<range> const high_ranges(all.begin() + static_cast<std::ptrdiff_t>(n_low_slices),
                                       all.end());

  // Open every object up front, so nothing slow sits between the enqueues.
  auto plug_object = file.open();
  auto low_object  = file.open();
  std::vector<std::shared_ptr<cucascade::io::io_object const>> high_objects;
  for (std::size_t i = 0; i < n_high; ++i) {
    high_objects.push_back(file.open());
  }

  // Destinations first: they must outlive the reactor.
  aligned_bytes low_dst{n_slices * stride};
  aligned_bytes high_dst{n_slices * stride};
  aligned_bytes plug_dst{IO_BLOCK_SIZE};
  spp_reactor reactor{1, low_service_interval};
  worker_plug plug;

  struct shared_state {
    std::atomic<std::size_t> highs_done{0};
    std::atomic<std::size_t> highs_before_low{0};
    std::atomic<bool> low_done{false};
    std::atomic<bool> low_ok{false};
    std::atomic<std::size_t> high_errors{0};
  };
  auto state = std::make_shared<shared_state>();

  auto plugged = enqueue_plug(reactor.get(), std::move(plug_object), plug_dst.get(), plug);
  REQUIRE(plug.wait_held(1));
  for (auto& object : high_objects) {
    enqueue_ranges(reactor.get(), std::move(object), high_ranges, high_dst.get(), io_priority::high)
      .install_callback([state](cucascade::exec::try_t<std::size_t>&& result) {
        if (!result.has_value()) state->high_errors.fetch_add(1);
        state->highs_done.fetch_add(1);
      });
  }
  enqueue_ranges(reactor.get(), std::move(low_object), low_ranges, low_dst.get(), io_priority::low)
    .install_callback([state](cucascade::exec::try_t<std::size_t>&& result) {
      state->highs_before_low.store(state->highs_done.load());
      state->low_ok.store(result.has_value());
      state->low_done.store(true);
    });
  REQUIRE(reactor.get().take_gauges().queued_requests == n_high + 1);
  plug.release();
  CHECK(std::move(plugged).get(std::chrono::seconds(30)) == IO_BLOCK_SIZE);

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!state->low_done.load() || state->highs_done.load() < n_high) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  CHECK(state->high_errors.load() == 0);
  for (auto const& r : low_ranges) {
    for (std::size_t i = r.offset; i < r.end(); i += 509) {
      REQUIRE(low_dst.get()[i] == pattern_byte(i));
    }
  }
  return {state->low_ok.load(),
          state->highs_before_low.load(),
          reactor.get().take_gauges().low_service_grants};
}

}  // namespace

TEST_CASE("io_uring low_service_interval defaults to bounded service", "[io][uring][priority]")
{
  CHECK(cucascade::io::uring::config{}.low_service_interval == 4);
}

TEST_CASE("io_uring grants a waiting low request bounded service under a high backlog",
          "[io][uring][priority][low_service]")
{
  // One slice per pass, so each grant's quantum is one slice: the low request's
  // n_low slices need at most n_low grants, one per `interval` high starts.
  constexpr std::size_t interval = 4;
  constexpr std::size_t n_low    = 4;
  constexpr std::size_t n_high   = 64;
  auto const outcome             = race_low_against_high_backlog(interval, n_low, n_high, 8);
  CHECK(outcome.low_ok);
  CHECK(outcome.grants >= 1);
  // At most n_low * interval high starts precede the low request's last grant
  // (plus the one that preempted it); allow a few more that were already in
  // flight while its last slice completed.
  CHECK(outcome.highs_before_low <= n_low * (interval + 1) + 4);
  CHECK(outcome.highs_before_low < n_high);
}

TEST_CASE("io_uring low_service_interval = 0 keeps priority strict",
          "[io][uring][priority][low_service]")
{
  constexpr std::size_t n_high = 64;
  auto const outcome           = race_low_against_high_backlog(0, 4, n_high, 8);
  CHECK(outcome.low_ok);
  CHECK(outcome.grants == 0);
  // The low request runs only once the high tier is empty, i.e. behind
  // (nearly) the whole backlog.
  CHECK(outcome.highs_before_low >= n_high / 2);
}

TEST_CASE("io_uring a read waiting on low work completes while high traffic continues",
          "[io][uring][priority][low_service]")
{
  // The fs_cache shape: a demand read blocks on an in-flight prefetch (a low
  // request) while other demand (high) reads keep arriving with no gap.  The
  // high stream only stops once the low request is done, so under strict
  // priority the wait would never end; bounded service must finish it.
  using cucascade::io::io_priority;
  constexpr std::size_t stride      = 64UL << 10;
  constexpr std::size_t n_low       = 32;
  constexpr std::size_t high_slices = 4;
  constexpr std::size_t max_backlog = 32;
  constexpr std::size_t n_slices    = n_low + high_slices;
  pattern_file const file{n_slices * stride};
  auto const all   = slice_ranges(n_slices, stride, false);
  auto const split = all.begin() + static_cast<std::ptrdiff_t>(n_low);
  std::vector<range> const low_ranges(all.begin(), split);
  std::vector<range> const high_ranges(split, all.end());

  // Destinations first: they must outlive the reactor.
  aligned_bytes low_dst{n_slices * stride};
  aligned_bytes high_dst{n_slices * stride};
  spp_reactor reactor{1};

  struct shared_state {
    std::atomic<std::size_t> highs_issued{0};
    std::atomic<std::size_t> highs_done{0};
    std::atomic<std::size_t> high_errors{0};
  };
  auto state = std::make_shared<shared_state>();

  auto prefetch =
    enqueue_ranges(reactor.get(), file.open(), low_ranges, low_dst.get(), io_priority::low);
  std::atomic<bool> stop{false};
  std::thread high_traffic([&] {
    while (!stop.load()) {
      if (state->highs_issued.load() - state->highs_done.load() >= max_backlog) {
        std::this_thread::yield();
        continue;
      }
      state->highs_issued.fetch_add(1);
      enqueue_ranges(reactor.get(), file.open(), high_ranges, high_dst.get(), io_priority::high)
        .install_callback([state](cucascade::exec::try_t<std::size_t>&& result) {
          if (!result.has_value()) state->high_errors.fetch_add(1);
          state->highs_done.fetch_add(1);
        });
    }
  });

  // The "demand read": blocked on the prefetch.
  std::size_t waited = 0;
  bool wait_ok       = false;
  try {
    waited  = std::move(prefetch).get(std::chrono::seconds(30));
    wait_ok = true;
  } catch (...) {
  }
  auto const highs_while_waiting = state->highs_done.load();
  stop.store(true);
  high_traffic.join();
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (state->highs_done.load() < state->highs_issued.load()) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  REQUIRE(wait_ok);
  CHECK(waited == n_low * stride);
  CHECK(state->high_errors.load() == 0);
  // High traffic really did run alongside, and the low tier got its turns by grant.
  CHECK(highs_while_waiting > 0);
  CHECK(reactor.get().take_gauges().low_service_grants >= 1);
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
    // Destination first: it must outlive the reactor.
    aligned_bytes destination{total};
    spp_reactor reactor{slices_per_pass};
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

  // Detached so a hung shutdown fails the watchdog instead of hanging the binary; the
  // body reports any exception through the watchdog, as one escaping a thread terminates.
  std::thread([file, destination, outcome, finished] {
    try {
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
    } catch (...) {
      finished->set_exception(std::current_exception());
    }
  }).detach();

  REQUIRE(watchdog.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  REQUIRE_NOTHROW(watchdog.get());
  CHECK(outcome->started);
  // Either the reads beat the shutdown or the remainder was canceled; never another error.
  CHECK(outcome->settled != settled_as::other_error);
  CHECK(outcome->queued_bytes == 0);
}

namespace {

/// Pinned staging for a pool of @p n_reactors uring reactors sharing one context.
class ioctx_staging {
 public:
  static constexpr std::size_t block_size = 1UL << 20;

  explicit ioctx_staging(std::size_t n_reactors, cucascade::io::uring::config cfg = {})
    : _mr{0, _upstream, n_reactors * (128UL << 20), n_reactors * (128UL << 20), block_size, 64, 0},
      _ctx{std::make_shared<uring_reactor::reactor_context>(cfg, &_mr)}
  {
  }

  [[nodiscard]] std::shared_ptr<uring_reactor::reactor_context> const& context() const noexcept
  {
    return _ctx;
  }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
  std::shared_ptr<uring_reactor::reactor_context> _ctx;
};

}  // namespace

TEST_CASE("io_uring ioctx reads a range-batched host read end to end",
          "[io][uring][uring_ioctx][range_batch_slices]")
{
  // 20 ranges is past the default range_batch_slices of 8, so each reactor's share
  // is split into several queue entries sharing one coordinator.  Odd ranges are
  // unaligned at both ends and take the buffered path.
  constexpr std::size_t n_ranges   = 20;
  constexpr std::size_t stride     = 64UL << 10;
  constexpr std::size_t total      = n_ranges * stride;
  constexpr std::uint8_t untouched = 0xA5;
  pattern_file const file{total};
  auto const ranges    = slice_ranges(n_ranges, stride, true);
  std::size_t expected = 0;
  for (auto const& r : ranges) {
    expected += r.size;
  }

  for (std::size_t const n_reactors : {std::size_t{1}, std::size_t{2}}) {
    CAPTURE(n_reactors);
    // Destination and staging first: both must outlive the ioctx's reactors.
    aligned_bytes destination{total};
    std::memset(destination.get(), untouched, total);
    ioctx_staging staging{n_reactors};
    uring_ioctx ioctx{n_reactors, staging.context()};
    ioctx.start();
    REQUIRE(ioctx.range_batch_slices() == 8);

    std::vector<cucascade::io::slice> slices;
    for (auto const& r : ranges) {
      slices.emplace_back(r.offset, r.size, destination.get() + r.offset);
    }
    auto const object = file.open();
    auto future       = ioctx.host_readv_async_io(*object, slices);
    REQUIRE(std::move(future).get(std::chrono::seconds(30)) == expected);

    // Every requested byte matches the file; every byte between ranges is untouched.
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

    // The read really was split: one reactor takes 8 + 8 + 4; two reactors take
    // their shares in at least three entries between them.
    auto const gauges = ioctx.reactor_gauges();
    REQUIRE(gauges.size() == n_reactors);
    std::uint64_t started = 0;
    for (auto const& g : gauges) {
      started += g.requests_started;
    }
    if (n_reactors == 1) {
      CHECK(started == 3);
    } else {
      CHECK(started >= 3);
    }
  }
}

TEST_CASE("io_uring ioctx reactor_gauges reports low requests queued behind a held request",
          "[io][uring][uring_ioctx][gauges]")
{
  using cucascade::io::io_priority;
  constexpr std::size_t n_reactors = 2;
  constexpr std::size_t n_ranges   = 40;
  constexpr std::size_t stride     = 64UL << 10;
  constexpr std::size_t total      = n_ranges * stride;
  pattern_file const file{total};
  auto const object = file.open();

  // Destinations and staging first: they must outlive the reactors.
  aligned_bytes plug_dst{2 * IO_BLOCK_SIZE};
  aligned_bytes destination{total};
  ioctx_staging staging{n_reactors};
  uring_ioctx ioctx{n_reactors, staging.context()};
  ioctx.start();
  worker_plug plug;

  // A two-slice read lands one slice on each reactor; both workers then hold
  // inside its completion, so nothing queued after it can be taken.
  std::vector<prepared_io_slice> plug_slices;
  for (std::size_t i = 0; i < n_reactors; ++i) {
    plug_slices.emplace_back(range{i * IO_BLOCK_SIZE, IO_BLOCK_SIZE},
                             host_buffer{plug_dst.get() + i * IO_BLOCK_SIZE});
    plug_slices.back().on_complete = plug.completion();
  }
  auto plugged = ioctx.mixed_readv_async_io(*object, std::move(plug_slices));
  REQUIRE(plug.wait_held(n_reactors));

  // 40 equal low ranges: 20 per reactor, each share split 8 + 8 + 4.
  std::vector<cucascade::io::slice> slices;
  for (std::size_t i = 0; i < n_ranges; ++i) {
    slices.emplace_back(i * stride, stride, destination.get() + i * stride);
  }
  auto low = ioctx.host_readv_async_io(*object, slices, io_priority::low);

  auto const gauges = ioctx.reactor_gauges();
  REQUIRE(gauges.size() == n_reactors);
  for (auto const& g : gauges) {
    CHECK(g.queued_low_requests == 3);
    CHECK(g.queued_requests == 3);
    CHECK(g.queued_bytes == (n_ranges / n_reactors) * stride);
  }

  plug.release();
  CHECK(std::move(plugged).get(std::chrono::seconds(30)) == n_reactors * IO_BLOCK_SIZE);
  CHECK(std::move(low).get(std::chrono::seconds(30)) == total);
  for (std::size_t i = 0; i < total; i += 4093) {
    REQUIRE(destination.get()[i] == pattern_byte(i));
  }
  for (auto const& g : ioctx.reactor_gauges()) {
    CHECK(g.queued_low_requests == 0);
  }
}

TEST_CASE("io_uring cancels a parked low request whose coordinator stopped",
          "[io][uring][priority][cancel]")
{
  // Strict priority and a long high request keep the low request parked; stopping
  // its coordinator must settle it without waiting for the high tier to drain,
  // and must not disturb the high request.
  using cucascade::io::io_priority;
  constexpr std::size_t n_slices = 512;
  constexpr std::size_t stride   = 64UL << 10;
  constexpr std::size_t total    = n_slices * stride;
  pattern_file const file{total};
  auto const ranges      = slice_ranges(n_slices, stride, false);
  auto const low_object  = file.open();
  auto const high_object = file.open();

  // Destinations first: they must outlive the reactor.
  aligned_bytes low_dst{total};
  aligned_bytes high_dst{total};
  spp_reactor reactor{1, 0};

  std::shared_ptr<grouped_coordinator> low_coordinator;
  auto low = enqueue_ranges(
    reactor.get(), low_object, ranges, low_dst.get(), io_priority::low, &low_coordinator);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (reactor.get().take_gauges().requests_started == 0) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  auto high = enqueue_ranges(reactor.get(), high_object, ranges, high_dst.get(), io_priority::high);
  REQUIRE(wait_for_preemption(reactor.get()));

  cancel_coordinator(*low_coordinator);
  auto const low_outcome = settle_within(std::move(low), std::chrono::seconds(10));
  CHECK(low_outcome.ready);
  CHECK(low_outcome.error == std::errc::interrupted);

  CHECK(std::move(high).get(std::chrono::seconds(30)) == total);
  for (std::size_t i = 0; i < total; i += 4093) {
    REQUIRE(high_dst.get()[i] == pattern_byte(i));
  }
  CHECK(reactor.get().queued_bytes() == 0);
}

TEST_CASE("io_uring settles a stopped parked low request under continuous high traffic",
          "[io][uring][priority][cancel]")
{
  // Strict priority (low_service_interval = 0) and a high stream that never lets
  // the high tier drain: the parked request is never resumed, so only the
  // per-pass check of its coordinator can settle it.
  using cucascade::io::io_priority;
  constexpr std::size_t stride      = 64UL << 10;
  constexpr std::size_t n_low       = 512;
  constexpr std::size_t high_slices = 4;
  constexpr std::size_t max_backlog = 32;
  constexpr std::size_t n_slices    = n_low + high_slices;
  pattern_file const file{n_slices * stride};
  auto const all   = slice_ranges(n_slices, stride, false);
  auto const split = all.begin() + static_cast<std::ptrdiff_t>(n_low);
  std::vector<range> const low_ranges(all.begin(), split);
  std::vector<range> const high_ranges(split, all.end());
  auto const object = file.open();

  // Destinations first: they must outlive the reactor.
  aligned_bytes low_dst{n_slices * stride};
  aligned_bytes high_dst{n_slices * stride};
  spp_reactor reactor{1, 0};

  struct shared_state {
    std::atomic<std::size_t> highs_issued{0};
    std::atomic<std::size_t> highs_done{0};
    std::atomic<std::size_t> high_errors{0};
  };
  auto state = std::make_shared<shared_state>();

  std::shared_ptr<grouped_coordinator> low_coordinator;
  auto low = enqueue_ranges(
    reactor.get(), object, low_ranges, low_dst.get(), io_priority::low, &low_coordinator);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (reactor.get().take_gauges().requests_started == 0) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  {
    // Joined (stop requested first) before the reactor and buffers go away, even
    // when an assertion below throws.
    std::jthread high_traffic([&](std::stop_token const& stop) {
      while (!stop.stop_requested()) {
        if (state->highs_issued.load() - state->highs_done.load() >= max_backlog) {
          std::this_thread::yield();
          continue;
        }
        state->highs_issued.fetch_add(1);
        enqueue_ranges(reactor.get(), object, high_ranges, high_dst.get(), io_priority::high)
          .install_callback([state](cucascade::exec::try_t<std::size_t>&& result) {
            if (!result.has_value()) state->high_errors.fetch_add(1);
            state->highs_done.fetch_add(1);
          });
      }
    });
    REQUIRE(wait_for_preemption(reactor.get()));

    cancel_coordinator(*low_coordinator);
    auto const low_outcome = settle_within(std::move(low), std::chrono::seconds(10));
    // The high stream is still running: the low request settled without a drain.
    auto const highs_after_low = state->highs_issued.load();
    CHECK(low_outcome.ready);
    CHECK(low_outcome.error == std::errc::interrupted);
    auto const traffic_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (state->highs_done.load() <= highs_after_low) {
      REQUIRE(std::chrono::steady_clock::now() < traffic_deadline);
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }

  auto const drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (state->highs_done.load() < state->highs_issued.load()) {
    REQUIRE(std::chrono::steady_clock::now() < drain_deadline);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  CHECK(state->high_errors.load() == 0);
  CHECK(reactor.get().queued_bytes() == 0);
}

TEST_CASE("io_uring shutdown cancels a parked low request", "[io][uring][priority][shutdown]")
{
  using cucascade::io::io_priority;
  constexpr std::size_t n_slices = 512;
  constexpr std::size_t stride   = 64UL << 10;
  constexpr std::size_t total    = n_slices * stride;

  struct outcome_t {
    bool parked{false};
    bool low_canceled{false};
    bool high_settled_cleanly{false};
    std::size_t queued_bytes{1};
  };

  auto file     = std::make_shared<pattern_file>(total);
  auto low_dst  = std::make_shared<aligned_bytes>(total);
  auto high_dst = std::make_shared<aligned_bytes>(total);
  auto outcome  = std::make_shared<outcome_t>();
  auto finished = std::make_shared<std::promise<void>>();
  auto watchdog = finished->get_future();

  // Detached so a hung shutdown fails the watchdog instead of hanging the binary; the
  // body reports any exception through the watchdog, as one escaping a thread terminates.
  std::thread([file, low_dst, high_dst, outcome, finished] {
    try {
      auto const ranges = slice_ranges(n_slices, stride, false);
      // Strict priority, so nothing resumes the parked request before shutdown.
      spp_reactor reactor{1, 0};
      auto low =
        enqueue_ranges(reactor.get(), file->open(), ranges, low_dst->get(), io_priority::low);
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (reactor.get().take_gauges().requests_started == 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
      auto high =
        enqueue_ranges(reactor.get(), file->open(), ranges, high_dst->get(), io_priority::high);
      outcome->parked = wait_for_preemption(reactor.get());
      reactor.get().shutdown();

      // shutdown() joined the worker, which settles everything it owned on the way out.
      outcome->low_canceled         = canceled(std::move(low));
      auto const high_outcome       = settle_within(std::move(high), std::chrono::seconds(0));
      outcome->high_settled_cleanly = high_outcome.ready && !high_outcome.other_error &&
                                      (high_outcome.error == std::error_code{}
                                         ? high_outcome.bytes == total
                                         : high_outcome.error == std::errc::operation_canceled);
      outcome->queued_bytes = reactor.get().queued_bytes();
      finished->set_value();
    } catch (...) {
      finished->set_exception(std::current_exception());
    }
  }).detach();

  REQUIRE(watchdog.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  REQUIRE_NOTHROW(watchdog.get());
  CHECK(outcome->parked);
  CHECK(outcome->low_canceled);
  // The high request either beat the shutdown or was canceled; never another error.
  CHECK(outcome->high_settled_cleanly);
  CHECK(outcome->queued_bytes == 0);
}

namespace {

/// The errc a failed read future carries (errc{} when it did not fail with a system_error).
[[nodiscard]] std::errc read_error(cucascade::exec::semi_future<std::size_t>&& future)
{
  try {
    std::ignore = std::move(future).get(std::chrono::seconds(30));
  } catch (std::system_error const& error) {
    return static_cast<std::errc>(error.code().value());
  } catch (...) {
  }
  return std::errc{};
}

/// Device memory of @p bytes filled with 0xAB, freed on scope exit.
class device_scratch {
 public:
  explicit device_scratch(std::size_t bytes) : _bytes(bytes)
  {
    REQUIRE(cudaMalloc(&_data, bytes) == cudaSuccess);
    REQUIRE(cudaMemset(_data, 0xAB, bytes) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }
  ~device_scratch() { std::ignore = cudaFree(_data); }
  device_scratch(device_scratch const&)            = delete;
  device_scratch& operator=(device_scratch const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return static_cast<std::uint8_t*>(_data); }
  [[nodiscard]] std::vector<std::uint8_t> bytes() const
  {
    std::vector<std::uint8_t> out(_bytes);
    REQUIRE(cudaMemcpy(out.data(), _data, _bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    return out;
  }

 private:
  void* _data{nullptr};
  std::size_t _bytes;
};

}  // namespace

TEST_CASE("io_uring device reads past EOF resolve with the bytes delivered", "[io][uring][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  constexpr std::size_t size = (1UL << 20) + 123;
  pattern_file const file{size};
  ioctx_staging staging{1};
  uring_ioctx ioctx{1, staging.context()};
  ioctx.start();
  auto const object = file.open();
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};

  device_scratch dst{2 * size};
  // A read running past EOF delivers the bytes up to it, and reports exactly those.
  CHECK(ioctx.device_read_async_io(*object, 0, 2 * size, dst.get(), stream)
          .get(std::chrono::seconds(30)) == size);
  CHECK(ioctx.device_read_async_io(*object, size - 100, 5000, dst.get(), stream)
          .get(std::chrono::seconds(30)) == 100);
  CHECK(ioctx.device_read_async_io(*object, size, 4096, dst.get(), stream)
          .get(std::chrono::seconds(30)) == 0);
  // The vector form reports the sum of what each slice delivered.
  using cucascade::io::slice;
  std::vector<slice> const slices{slice{0, 4096, dst.get()},
                                  slice{size - 10, 4096, dst.get() + 4096},
                                  slice{size + 4096, 4096, dst.get() + 8192}};
  CHECK(ioctx.device_readv_async_io(*object, slices, stream).get(std::chrono::seconds(30)) ==
        4096 + 10);
  auto const got = dst.bytes();
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK(got[4096 + i] == pattern_byte(size - 10 + i));
  }
  ioctx.shutdown();
}

TEST_CASE("io_uring reads of a file truncated after open never report unread bytes",
          "[io][uring][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  // The object keeps the size it was opened with.  A read the file can no longer
  // fill fails (EIO) instead of resolving with the requested count and an
  // untouched tail.
  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};
  ioctx_staging staging{1};
  uring_ioctx ioctx{1, staging.context()};
  ioctx.start();
  auto const object = file.open();
  REQUIRE(object->size() == size);
  std::filesystem::resize_file(file.path(), size / 2 + 100);
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};

  device_scratch dst{size};
  SECTION("aligned device read (O_DIRECT where available)")
  {
    CHECK(read_error(ioctx.device_read_async_io(*object, 0, size, dst.get(), stream)) ==
          std::errc::io_error);
  }
  SECTION("unaligned device read (buffered)")
  {
    CHECK(read_error(ioctx.device_read_async_io(*object, 1, size - 1, dst.get(), stream)) ==
          std::errc::io_error);
  }
  SECTION("device read entirely past the new end")
  {
    CHECK(read_error(ioctx.device_read_async_io(*object, size - 8192, 8192, dst.get(), stream)) ==
          std::errc::io_error);
  }
  SECTION("asynchronous host read")
  {
    std::vector<std::uint8_t> host(size);
    CHECK(read_error(ioctx.host_read_async_io(*object, 0, size, host.data())) ==
          std::errc::io_error);
  }
  SECTION("synchronous host read reports what pread delivered")
  {
    std::vector<std::uint8_t> host(size);
    CHECK(ioctx.host_read_io(*object, 0, size, host.data()) == size / 2 + 100);
  }
  SECTION("a part the file still holds reads in full")
  {
    CHECK(ioctx.device_read_async_io(*object, 0, size / 2, dst.get(), stream)
            .get(std::chrono::seconds(30)) == size / 2);
  }
  ioctx.shutdown();
}
namespace {

/** Makes the uring reactors report a failed event record after each H2D copy, for the guard's life.
 */
class copy_event_record_failure {
 public:
  copy_event_record_failure()
  {
    cucascade::io::uring::detail::set_copy_event_record_failure_for_testing(true);
  }
  ~copy_event_record_failure()
  {
    cucascade::io::uring::detail::set_copy_event_record_failure_for_testing(false);
  }
  copy_event_record_failure(copy_event_record_failure const&)            = delete;
  copy_event_record_failure& operator=(copy_event_record_failure const&) = delete;
};

/// Hold @p stream on the host for @p duration (a host function that sleeps).
void hold_stream(cudaStream_t stream, std::chrono::milliseconds duration)
{
  auto* const heap = new std::chrono::milliseconds{duration};
  REQUIRE(cudaLaunchHostFunc(
            stream,
            [](void* arg) {
              std::unique_ptr<std::chrono::milliseconds> const time{
                static_cast<std::chrono::milliseconds*>(arg)};
              std::this_thread::sleep_for(*time);
            },
            heap) == cudaSuccess);
}

/// Copy @p bytes of device memory back on a separate stream, so the read does
/// not wait for work still queued on the stream the device bytes were written on.
[[nodiscard]] std::vector<std::uint8_t> device_bytes_elsewhere(void const* device,
                                                               std::size_t bytes)
{
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  std::vector<std::uint8_t> out(bytes);
  REQUIRE(cudaMemcpyAsync(out.data(), device, bytes, cudaMemcpyDeviceToHost, other) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
  return out;
}

}  // namespace

TEST_CASE("io_uring device reads that fail after enqueuing a copy settle only once it landed",
          "[io][uring][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }

  // The caller's stream is held on the host, so the H2D copy queued behind the
  // hold cannot have run yet when its event record "fails".  The read reports
  // the error, but only after the queued copy finished: the destination holds
  // the file's bytes when the future settles, so the caller may free it (and
  // the reactor may reuse the staging block the copy reads).
  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};
  ioctx_staging staging{1};
  uring_ioctx ioctx{1, staging.context()};
  ioctx.start();
  auto const object = file.open();

  cudaStream_t stream{};
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  void* destination = nullptr;
  REQUIRE(cudaMalloc(&destination, size) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(destination, 0xAB, size, stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  {
    hold_stream(stream, std::chrono::milliseconds{300});
    copy_event_record_failure const failing;
    auto future = ioctx.device_read_async_io(
      *object, 0, size, static_cast<std::uint8_t*>(destination), ::cuda::stream_ref{stream});
    CHECK_THROWS(std::move(future).get(std::chrono::seconds(30)));
  }
  auto const bytes  = device_bytes_elsewhere(destination, size);
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < size; ++i) {
    wrong += bytes[i] != pattern_byte(i) ? 1 : 0;
  }
  CHECK(wrong == 0);

  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(cudaFree(destination) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  ioctx.shutdown();
}

TEST_CASE("io_uring device reads order after the caller's default streams", "[io][uring][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  // cudaStreamPerThread names the per-thread stream of whichever thread uses it:
  // the H2D copy must be ordered after the SUBMITTING thread's.  The legacy
  // default stream is the same from every thread.
  auto const [label, caller_stream] =
    GENERATE(std::pair{"per-thread default stream", cudaStreamPerThread},
             std::pair{"legacy default stream", cudaStreamLegacy});
  CAPTURE(label);
  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};
  ioctx_staging staging{1};
  uring_ioctx ioctx{1, staging.context()};
  ioctx.start();
  auto const object = file.open();

  void* destination = nullptr;
  REQUIRE(cudaMalloc(&destination, size) == cudaSuccess);
  // Hold the caller's stream, then queue a memset of the destination behind it:
  // a copy not ordered after it lands first and is then zeroed.
  hold_stream(caller_stream, std::chrono::milliseconds{300});
  REQUIRE(cudaMemsetAsync(destination, 0, size, caller_stream) == cudaSuccess);
  CHECK(ioctx
          .device_read_async_io(*object,
                                0,
                                size,
                                static_cast<std::uint8_t*>(destination),
                                ::cuda::stream_ref{caller_stream})
          .get(std::chrono::seconds(30)) == size);
  REQUIRE(cudaStreamSynchronize(caller_stream) == cudaSuccess);
  auto const bytes  = device_bytes_elsewhere(destination, size);
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < size; ++i) {
    wrong += bytes[i] != pattern_byte(i) ? 1 : 0;
  }
  CHECK(wrong == 0);
  REQUIRE(cudaFree(destination) == cudaSuccess);
  ioctx.shutdown();
}
