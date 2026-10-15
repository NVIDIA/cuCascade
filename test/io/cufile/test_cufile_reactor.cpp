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
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/shim/cufile.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using cucascade::io::byte_range;
using cucascade::io::file_descriptor;
using cucascade::io::grouped_coordinator;
using cucascade::io::grouped_io_request;
using cucascade::io::host_buffer;
using cucascade::io::io_priority;
using cucascade::io::prepared_io_completion;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;

namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  return static_cast<std::uint8_t>((offset * 2654435761ULL) >> 11);
}

/** A temporary file of @p bytes filled with @ref pattern_byte, removed on scope exit. */
class pattern_file {
 public:
  explicit pattern_file(std::size_t bytes)
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_cufile_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin")),
      _size(bytes)
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

  [[nodiscard]] std::string path() const { return _path.string(); }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
  std::size_t _size;
};

[[nodiscard]] bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/** Whether the first @p size bytes of @p got, which was read from @p offset, are the file's. */
[[nodiscard]] bool matches_file(std::vector<std::uint8_t> const& got,
                                std::size_t offset,
                                std::size_t size)
{
  if (got.size() < size) return false;
  for (std::size_t i = 0; i < size; ++i) {
    if (got[i] != pattern_byte(offset + i)) return false;
  }
  return true;
}

/** Copy @p bytes of @p device back to the host once @p stream has drained. */
[[nodiscard]] std::vector<std::uint8_t> device_bytes(rmm::device_buffer const& device,
                                                     std::size_t bytes,
                                                     rmm::cuda_stream const& stream)
{
  std::vector<std::uint8_t> out(bytes);
  REQUIRE(
    cudaMemcpyAsync(out.data(), device.data(), bytes, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  return out;
}

/** Consume @p future and report whether it carries the reactor's cancellation error. */
[[nodiscard]] bool canceled(cucascade::exec::semi_future<std::size_t>&& future) noexcept
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

/** A started cufile_ioctx over @p cfg, shut down on scope exit. */
class started_ioctx {
 public:
  explicit started_ioctx(config cfg, std::size_t n_reactors = 2)
    : _ioctx(n_reactors, std::make_shared<cufile_reactor::reactor_context>(cfg))
  {
    _ioctx.start();
  }

  ~started_ioctx() { _ioctx.shutdown(); }

  started_ioctx(started_ioctx const&)            = delete;
  started_ioctx& operator=(started_ioctx const&) = delete;

  [[nodiscard]] cufile_ioctx& get() noexcept { return _ioctx; }

 private:
  cufile_ioctx _ioctx;
};

}  // namespace

TEST_CASE("cufile reactor claims regular files only", "[io][cufile]")
{
  pattern_file const file{4096};
  CHECK(cufile_reactor::supports(file.path()));
  CHECK_FALSE(cufile_reactor::supports(std::filesystem::temp_directory_path().string()));
  CHECK_FALSE(cufile_reactor::supports("/nonexistent/cucascade_cufile_missing.bin"));
  CHECK_FALSE(cufile_reactor::supports("s3://bucket/key"));
}

TEST_CASE("cufile alignment widens and coalesces safely", "[io][cufile]")
{
  std::array<byte_range, 4> input{
    byte_range{8193, 100}, byte_range{1, 100}, byte_range{4095, 2}, byte_range{0, 0}};
  auto merged = cufile_reactor::align_and_coalesce(input);

  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 12'288);

  // A range far from the others stays separate; a sub-page alignment is ignored.
  std::array<byte_range, 2> apart{byte_range{10, 10}, byte_range{100'000, 1}};
  auto kept = cufile_reactor::align_and_coalesce(apart, std::size_t{1});
  REQUIRE(kept.size() == 2);
  CHECK(kept[0].offset() == 0);
  CHECK(kept[0].size() == 4096);
  CHECK(kept[1].offset() % 4096 == 0);
}

TEST_CASE("cufile create_io_object honours the compat mode", "[io][cufile]")
{
  pattern_file const file{1 << 20};

  SECTION("ON never registers with cuFile and opens no O_DIRECT fd")
  {
    auto object = cufile_reactor::create_io_object(file.path(), true, kvikio::CompatMode::ON);
    CHECK(object->size() == file.size());
    CHECK_FALSE(object->cufile_registered());
    CHECK(object->direct_fd() == -1);
    CHECK(object->buffered_fd() >= 0);
  }

  SECTION("without O_DIRECT nothing can be registered")
  {
    auto object = cufile_reactor::create_io_object(file.path(), false, kvikio::CompatMode::AUTO);
    CHECK_FALSE(object->cufile_registered());
    CHECK(object->direct_fd() == -1);
  }

  SECTION("OFF enforces cuFile, so it cannot be combined with use_odirect=false")
  {
    CHECK_THROWS_AS(cufile_reactor::create_io_object(file.path(), false, kvikio::CompatMode::OFF),
                    std::invalid_argument);
  }

  SECTION("OFF either registers the file or refuses it, and never silently degrades")
  {
    std::unique_ptr<cufile_io_object> object;
    try {
      object = cufile_reactor::create_io_object(file.path(), true, kvikio::CompatMode::OFF);
    } catch (std::exception const&) {
      return;  // cuFile cannot serve this host or filesystem: refusing is the contract
    }
    CHECK(object->cufile_registered());
  }

  SECTION("a missing path is rejected")
  {
    CHECK_THROWS(cufile_reactor::create_io_object("/nonexistent/cucascade_cufile_missing.bin"));
  }
}

TEST_CASE("cufile ioctx reads host destinations", "[io][cufile]")
{
  constexpr std::size_t file_bytes = 8UL << 20;
  pattern_file const file{file_bytes};
  started_ioctx ioctx{config{}};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(object->size() == file_bytes);

  SECTION("an aligned read")
  {
    std::vector<std::uint8_t> dst(1UL << 20);
    REQUIRE(ioctx.get().host_read_async_io(*object, 4096, dst.size(), dst.data()).get(10s) ==
            dst.size());
    CHECK(matches_file(dst, 4096, dst.size()));
  }

  SECTION("an unaligned read")
  {
    std::vector<std::uint8_t> dst(100'003);
    REQUIRE(ioctx.get().host_read_async_io(*object, 12'345, dst.size(), dst.data()).get(10s) ==
            dst.size());
    CHECK(matches_file(dst, 12'345, dst.size()));
  }

  SECTION("a synchronous read")
  {
    std::vector<std::uint8_t> dst(65'537);
    REQUIRE(ioctx.get().host_read_io(*object, 777, dst.size(), dst.data()) == dst.size());
    CHECK(matches_file(dst, 777, dst.size()));
  }

  SECTION("a vectored read lands each range at its own destination")
  {
    std::vector<std::uint8_t> a(8192);
    std::vector<std::uint8_t> b(5000);
    std::array<cucascade::io::slice, 2> slices{cucascade::io::slice{0, a.size(), a.data()},
                                               cucascade::io::slice{1'000'000, b.size(), b.data()}};
    REQUIRE(ioctx.get().host_readv_async_io(*object, slices).get(10s) == a.size() + b.size());
    CHECK(matches_file(a, 0, a.size()));
    CHECK(matches_file(b, 1'000'000, b.size()));
  }
}

TEST_CASE("cufile ioctx clamps a read that crosses the end of the file", "[io][cufile]")
{
  pattern_file const file{100'000};
  started_ioctx ioctx{config{}};
  auto object = ioctx.get().open_io_object(file.path());

  std::vector<std::uint8_t> dst(50'000, 0xEE);
  REQUIRE(ioctx.get().host_read_async_io(*object, 90'000, dst.size(), dst.data()).get(10s) ==
          10'000);
  CHECK(matches_file(dst, 90'000, 10'000));
  CHECK(dst[10'000] == 0xEE);  // nothing is written past the end of the file
}

TEST_CASE("cufile ioctx reads device destinations", "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t file_bytes = 16UL << 20;
  pattern_file const file{file_bytes};

  // 1 B forces every read onto cuFile when the file registered; the huge value
  // forces every read onto the POSIX fallback.  Either way the bytes must match.
  auto const [threshold, label] =
    GENERATE(std::pair<std::size_t, std::string>{1, "cuFile when registered"},
             std::pair<std::size_t, std::string>{1UL << 40, "POSIX fallback"});
  DYNAMIC_SECTION("gds_threshold selects " << label)
  {
    config cfg{};
    cfg.gds_threshold = threshold;
    started_ioctx ioctx{cfg};
    auto object       = ioctx.get().open_io_object(file.path());
    auto const& typed = dynamic_cast<cufile_io_object const&>(*object);
    if (threshold == 1 && !typed.cufile_registered()) {
      WARN("cuFile registration unavailable on this host: exercising the POSIX fallback only");
    }

    rmm::cuda_stream stream;

    SECTION("an aligned read")
    {
      constexpr std::size_t offset = 1UL << 20;
      constexpr std::size_t size   = 4UL << 20;
      rmm::device_buffer dst(size, stream);
      REQUIRE(ioctx.get()
                .device_read_async_io(
                  *object, offset, size, static_cast<std::uint8_t*>(dst.data()), stream)
                .get(30s) == size);
      CHECK(matches_file(device_bytes(dst, size, stream), offset, size));
    }

    SECTION("an unaligned read")
    {
      constexpr std::size_t offset = 123'457;
      constexpr std::size_t size   = 3'000'011;
      rmm::device_buffer dst(size, stream);
      REQUIRE(ioctx.get()
                .device_read_async_io(
                  *object, offset, size, static_cast<std::uint8_t*>(dst.data()), stream)
                .get(30s) == size);
      CHECK(matches_file(device_bytes(dst, size, stream), offset, size));
    }

    SECTION("a read crossing the end of the file is clamped")
    {
      constexpr std::size_t offset = file_bytes - 10'000;
      constexpr std::size_t size   = 50'000;
      rmm::device_buffer dst(size, stream);
      REQUIRE(ioctx.get()
                .device_read_async_io(
                  *object, offset, size, static_cast<std::uint8_t*>(dst.data()), stream)
                .get(30s) == 10'000);
      CHECK(matches_file(device_bytes(dst, 10'000, stream), offset, 10'000));
    }

    SECTION("a request mixing host and device destinations fills both")
    {
      constexpr std::size_t offset = 8192;
      constexpr std::size_t size   = 2UL << 20;
      rmm::device_buffer dst(size, stream);
      std::vector<std::uint8_t> host(size);
      std::vector<prepared_io_slice> slices;
      slices.emplace_back(
        range{offset, size},
        host_buffer{host.data()},
        cucascade::io::device_buffer{static_cast<std::uint8_t*>(dst.data()), stream});
      REQUIRE(ioctx.get().host_device_readv_async_io(*object, std::move(slices)).get(30s) == size);
      CHECK(matches_file(host, offset, size));
      CHECK(matches_file(device_bytes(dst, size, stream), offset, size));
    }

    SECTION("several device ranges are all filled")
    {
      constexpr std::size_t stride = 1UL << 20;
      constexpr std::size_t count  = 6;
      rmm::device_buffer dst(count * stride, stream);
      std::vector<cucascade::io::slice> slices;
      for (std::size_t i = 0; i < count; ++i) {
        slices.emplace_back(
          (2 * i) * stride + 17, stride - 17, static_cast<std::uint8_t*>(dst.data()) + i * stride);
      }
      REQUIRE(ioctx.get().device_readv_async_io(*object, slices, stream).get(30s) ==
              count * (stride - 17));
      auto const got = device_bytes(dst, count * stride, stream);
      for (std::size_t i = 0; i < count; ++i) {
        std::vector<std::uint8_t> part(
          got.begin() + static_cast<std::ptrdiff_t>(i * stride),
          got.begin() + static_cast<std::ptrdiff_t>((i + 1) * stride - 17));
        CHECK(matches_file(part, (2 * i) * stride + 17, stride - 17));
      }
    }
  }
}

TEST_CASE("cufile ioctx cancels reads submitted after shutdown", "[io][cufile]")
{
  pattern_file const file{64 * 1024};
  cufile_ioctx ioctx{1, std::make_shared<cufile_reactor::reactor_context>(config{})};
  ioctx.start();
  auto object = ioctx.open_io_object(file.path());
  ioctx.shutdown();

  std::vector<std::uint8_t> dst(4096);
  CHECK(canceled(ioctx.host_read_async_io(*object, 0, dst.size(), dst.data())));
}

TEST_CASE("cufile ioctx refuses the prefetching cache it cannot serve", "[io][cufile]")
{
  // With the cache armed, a device read that misses builds a slice whose host buffer is a
  // fragmented cache chunk.  The reactor stages nothing through host memory, so it rejects such
  // slices; the capability flags must therefore keep the ioctx from ever arming the cache,
  // otherwise those device reads would fail.
  started_ioctx ioctx{config{}};
  CHECK_FALSE(ioctx.get().supports_vector_host_read());
  CHECK_FALSE(ioctx.get().can_use_fs_cache());
  CHECK(ioctx.get().supports_device_read());
}

TEST_CASE("cufile reactor rejects a zero in-flight window", "[io][cufile]")
{
  config cfg{};
  cfg.max_inflight = 0;
  CHECK_THROWS_AS(cufile_reactor(std::make_shared<cufile_reactor::reactor_context>(cfg)),
                  std::invalid_argument);
  CHECK_THROWS_AS(cufile_reactor(nullptr), std::invalid_argument);
}

TEST_CASE("cufile reactor drops a slice from the wrong backend's object cleanly", "[io][cufile]")
{
  // A reactor handed an io_object it did not create must fail the read, not crash.
  struct foreign_object final : cucascade::io::io_object {
    std::string name{"foreign"};
    [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return name; }
    [[nodiscard]] const std::string& object_path() const noexcept override { return name; }
    [[nodiscard]] std::size_t size() const noexcept override { return 4096; }
  };

  cufile_reactor reactor{std::make_shared<cufile_reactor::reactor_context>(config{}), ""};
  reactor.start();

  std::vector<std::uint8_t> dst(4096);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, dst.size()}, host_buffer{dst.data()});
  auto coordinator = std::make_shared<grouped_coordinator>(dst.size(), 1);
  auto future      = coordinator->get_future();
  reactor.enqueue(grouped_io_request::create(
    std::make_shared<foreign_object>(), std::move(slices), std::move(coordinator)));

  CHECK_THROWS_AS(std::move(future).get(10s), std::invalid_argument);
  reactor.shutdown();
}

namespace {

/** Records the order in which tagged slices complete on the worker thread. */
class completion_log {
 public:
  void record(std::string tag)
  {
    std::lock_guard lock(_mutex);
    _tags.push_back(std::move(tag));
  }

  [[nodiscard]] std::vector<std::string> tags() const
  {
    std::lock_guard lock(_mutex);
    return _tags;
  }

 private:
  mutable std::mutex _mutex;
  std::vector<std::string> _tags;
};

/**
 * Enqueue a request of @p n_slices 4 KiB host slices, each logging "<tag><index>" as it
 * completes.  When @p first_slice_hook is set it runs on the worker inside slice 0's
 * completion, which parks the worker there until the hook returns.
 */
[[nodiscard]] cucascade::exec::semi_future<std::size_t> enqueue_tagged(
  cufile_reactor& reactor,
  std::shared_ptr<cucascade::io::io_object const> object,
  std::string const& tag,
  std::size_t n_slices,
  io_priority priority,
  completion_log& log,
  std::vector<std::uint8_t>& destination,
  std::function<void()> first_slice_hook = nullptr)
{
  constexpr std::size_t slice_bytes = 4096;
  std::vector<prepared_io_slice> slices;
  slices.reserve(n_slices);
  for (std::size_t i = 0; i < n_slices; ++i) {
    slices.emplace_back(range{i * slice_bytes, slice_bytes},
                        host_buffer{destination.data() + i * slice_bytes});
    slices.back().priority    = priority;
    slices.back().on_complete = std::make_shared<prepared_io_completion>(
      [&log, tag, i, first_slice_hook](std::span<cucascade::io::cache::cached_chunk* const>,
                                       bool) noexcept {
        if (i == 0 && first_slice_hook) first_slice_hook();
        log.record(tag + std::to_string(i));
      });
  }
  auto coordinator = std::make_shared<grouped_coordinator>(n_slices * slice_bytes, n_slices);
  auto future      = coordinator->get_future();
  reactor.enqueue(
    grouped_io_request::create(std::move(object), std::move(slices), std::move(coordinator)));
  return future;
}

/**
 * One low request of 12 slices is parked mid-way by a backlog of @p n_high high requests.  The
 * worker is held inside the low request's first slice while the backlog is queued, so the order
 * the worker then serves them in is a function of the scheduling policy alone.
 */
[[nodiscard]] std::vector<std::string> run_low_against_high_backlog(
  std::size_t low_service_interval, std::size_t n_high)
{
  pattern_file const file{1 << 20};
  config cfg{};
  cfg.max_inflight         = 2;  // a pass is two slices
  cfg.low_service_interval = low_service_interval;
  cufile_reactor reactor{std::make_shared<cufile_reactor::reactor_context>(cfg), ""};
  reactor.start();
  std::shared_ptr<cucascade::io::io_object const> object{
    cufile_reactor::create_io_object(file.path())};

  completion_log log;
  std::promise<void> entered;
  std::promise<void> release;
  auto const release_future = release.get_future().share();

  std::vector<std::uint8_t> low_dst(12 * 4096);
  std::vector<std::vector<std::uint8_t>> high_dst(n_high, std::vector<std::uint8_t>(4096));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;

  futures.push_back(enqueue_tagged(
    reactor, object, "L", 12, io_priority::low, log, low_dst, [&entered, release_future] {
      entered.set_value();
      release_future.wait();
    }));
  REQUIRE(entered.get_future().wait_for(10s) == std::future_status::ready);
  for (std::size_t i = 0; i < n_high; ++i) {
    futures.push_back(enqueue_tagged(
      reactor, object, "H" + std::to_string(i) + "_", 1, io_priority::high, log, high_dst[i]));
  }
  release.set_value();

  for (auto& future : futures) {
    REQUIRE(std::move(future).get(30s) > 0);
  }
  reactor.shutdown();
  for (std::size_t i = 0; i < low_dst.size(); i += 509) {
    REQUIRE(low_dst[i] == pattern_byte(i));  // parking and resuming lost no bytes
  }
  return log.tags();
}

}  // namespace

TEST_CASE("cufile reactor parks an active low request for a high one and resumes it",
          "[io][cufile][priority]")
{
  auto const order = run_low_against_high_backlog(/*low_service_interval=*/0, /*n_high=*/1);

  // The pass is two slices, so the low request is parked after L0, L1, served the high
  // request, then resumed where it left off.
  REQUIRE(order.size() == 13);
  CHECK(order[0] == "L0");
  CHECK(order[1] == "L1");
  CHECK(order[2] == "H0_0");
  for (std::size_t i = 2; i < 12; ++i) {
    CHECK(order[i + 1] == "L" + std::to_string(i));
  }
}

TEST_CASE("cufile reactor keeps priority strict when low_service_interval is 0",
          "[io][cufile][priority]")
{
  auto const order = run_low_against_high_backlog(/*low_service_interval=*/0, /*n_high=*/40);

  REQUIRE(order.size() == 52);
  // Every high request is served before the parked low request gets another turn.
  CHECK(order.back() == "L11");
}

TEST_CASE("cufile reactor grants a waiting low request bounded service under a high backlog",
          "[io][cufile][priority]")
{
  auto const order = run_low_against_high_backlog(/*low_service_interval=*/2, /*n_high=*/40);

  REQUIRE(order.size() == 52);
  // The low request finishes while most of the high backlog is still waiting.
  auto const low_done = std::find(order.begin(), order.end(), "L11") - order.begin();
  CHECK(low_done < 24);
  // ...and the grants interleave: two highs, then a two-slice quantum of low, repeating.
  CHECK(order[2] == "H0_0");
  CHECK(order[3] == "H1_0");
  CHECK(order[4] == "L2");
  CHECK(order[5] == "L3");
}
