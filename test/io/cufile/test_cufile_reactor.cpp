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

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/shim/cufile.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include <thread>
#include <tuple>
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

TEST_CASE("cufile ioctx reads a large device slice in aligned pieces up to the end of the file",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // An unaligned file size makes the last block a sub-block tail; a 64 KiB piece
  // size makes every read below span many cuFileRead calls.
  static constexpr std::size_t file_bytes = (1UL << 20) + 1234;
  pattern_file const file{file_bytes};
  config cfg{};
  cfg.gds_threshold = 1;
  cfg.max_op_bytes  = 64UL << 10;
  started_ioctx ioctx{cfg};
  auto object = ioctx.get().open_io_object(file.path());
  rmm::cuda_stream stream;

  auto const [offset, size] = GENERATE(std::pair<std::size_t, std::size_t>{0, file_bytes},
                                       std::pair<std::size_t, std::size_t>{4097, file_bytes},
                                       std::pair<std::size_t, std::size_t>{65'535, 300'001},
                                       std::pair<std::size_t, std::size_t>{file_bytes - 1000, 1000},
                                       std::pair<std::size_t, std::size_t>{1UL << 20, 1UL << 20});
  auto const expected       = std::min(size, file_bytes - offset);
  CAPTURE(offset, size);
  rmm::device_buffer dst(size, stream);
  REQUIRE(
    ioctx.get()
      .device_read_async_io(*object, offset, size, static_cast<std::uint8_t*>(dst.data()), stream)
      .get(30s) == expected);
  CHECK(matches_file(device_bytes(dst, expected, stream), offset, expected));
}

TEST_CASE("cufile ioctx device read starts after work already queued on its stream",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t size = 2UL << 20;
  pattern_file const file{size};
  config cfg{};
  cfg.gds_threshold = 1;
  started_ioctx ioctx{cfg};
  auto object = ioctx.get().open_io_object(file.path());
  rmm::cuda_stream stream;
  rmm::device_buffer dst(size, stream);

  // Hold the stream on the host, then queue a memset of the destination behind it.
  // A read that did not wait for that memset would land first and be overwritten.
  REQUIRE(cudaLaunchHostFunc(
            stream.value(),
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{200}); },
            nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(dst.data(), 0xAB, size, stream.value()) == cudaSuccess);
  REQUIRE(ioctx.get()
            .device_read_async_io(*object, 0, size, static_cast<std::uint8_t*>(dst.data()), stream)
            .get(30s) == size);
  CHECK(matches_file(device_bytes(dst, size, stream), 0, size));
}

TEST_CASE("cufile ioctx stages a large device read through POSIX when cuFile is not used",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // Larger than the two 8 MiB scratch slots together, so the slots are reused.
  static constexpr std::size_t file_bytes = (20UL << 20) + 123;
  pattern_file const file{file_bytes};

  auto const [label, cfg] = GENERATE(std::pair<std::string, config>{"below the threshold",
                                                                    [] {
                                                                      config c{};
                                                                      c.gds_threshold = 1UL << 40;
                                                                      return c;
                                                                    }()},
                                     std::pair<std::string, config>{"file not registered", [] {
                                                                      config c{};
                                                                      c.compat_mode =
                                                                        kvikio::CompatMode::ON;
                                                                      return c;
                                                                    }()});
  DYNAMIC_SECTION(label)
  {
    started_ioctx ioctx{cfg};
    auto object = ioctx.get().open_io_object(file.path());
    rmm::cuda_stream stream;
    constexpr std::size_t offset = 12'345;
    constexpr std::size_t size   = file_bytes - offset;
    rmm::device_buffer dst(size, stream);
    // The staged copies run on the caller's stream, after work already queued there.
    REQUIRE(cudaMemsetAsync(dst.data(), 0xAB, size, stream.value()) == cudaSuccess);
    REQUIRE(
      ioctx.get()
        .device_read_async_io(*object, offset, size, static_cast<std::uint8_t*>(dst.data()), stream)
        .get(30s) == size);
    CHECK(matches_file(device_bytes(dst, size, stream), offset, size));
  }
}

TEST_CASE("cufile platform probe is cached and self-consistent", "[io][cufile]")
{
  auto const& first  = cucascade::io::cufile::probe_platform();
  auto const& second = cucascade::io::cufile::probe_platform();
  CHECK(&first == &second);
  CHECK(cucascade::io::cufile::gds_available() ==
        (first.driver_opened && !first.compat_only && !first.forced_compat));
  if (!first.driver_opened) return;
  CHECK(first.library_available);
  CHECK(first.compat_only == (first.dstatusflags == 0));
  CHECK(first.max_direct_io_size_kb > 0);
  INFO("cuFile " << first.version << ", dstatusflags " << first.dstatusflags
                 << (first.compat_only ? " (compatibility mode)" : " (GDS)"));
  CHECK(first.version > 0);
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

TEST_CASE("cufile reactor rejects a cuFile piece smaller than a block", "[io][cufile]")
{
  config cfg{};
  cfg.max_op_bytes = cucascade::io::IO_BLOCK_SIZE - 1;
  CHECK_THROWS_AS(cufile_reactor(std::make_shared<cufile_reactor::reactor_context>(cfg)),
                  std::invalid_argument);
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

// ---------------------------------------------------------------------------
// Throughput probe (hidden; run explicitly with "[.cufile-bench]")
// ---------------------------------------------------------------------------

namespace {

/// Write @p bytes of a cheap pattern to @p path unless it already holds that many.
void ensure_bench_file(std::string const& path, std::size_t bytes)
{
  std::error_code ec;
  if (std::filesystem::file_size(path, ec) >= bytes && !ec) return;
  constexpr std::size_t block = 64UL << 20;
  std::vector<char> data(block);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  for (std::size_t done = 0; done < bytes; done += block) {
    for (std::size_t i = 0; i < block; i += 4096) {
      data[i] = static_cast<char>(pattern_byte(done + i));
    }
    out.write(data.data(), static_cast<std::streamsize>(std::min(block, bytes - done)));
  }
  out.close();
  REQUIRE(out);
}

/// Evict @p path from the page cache (no root needed for clean pages).
void drop_file_cache(std::string const& path)
{
  file_descriptor fd{::open(path.c_str(), O_RDONLY)};
  if (!fd) return;
  std::ignore = ::fdatasync(fd.get());
  std::ignore = ::posix_fadvise(fd.get(), 0, 0, POSIX_FADV_DONTNEED);
}

[[nodiscard]] std::size_t env_size(char const* name, std::size_t fallback)
{
  char const* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? std::stoull(value) : fallback;
}

/// GB/s of @p run over @p bytes, median of @p rounds, each after a page-cache drop.
template <class Run>
[[nodiscard]] double measure(std::string const& path, std::size_t bytes, int rounds, Run&& run)
{
  std::vector<double> rates;
  for (int r = 0; r < rounds; ++r) {
    drop_file_cache(path);
    auto const start = std::chrono::steady_clock::now();
    run();
    auto const secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    rates.push_back(static_cast<double>(bytes) / secs / 1e9);
  }
  std::sort(rates.begin(), rates.end());
  return rates[rates.size() / 2];
}

/// Run every probe scenario against @p ctx and print one line per scenario.
void run_probe(std::string const& label,
               cucascade::io::ioctx& ctx,
               std::string const& path,
               std::size_t file_bytes,
               int rounds)
{
  auto object                    = ctx.open_io_object(path);
  constexpr std::size_t window   = 1UL << 30;
  constexpr std::size_t n_stream = 8;
  rmm::cuda_stream alloc_stream;
  rmm::device_buffer dst(window, alloc_stream);
  alloc_stream.synchronize();
  auto* const base = static_cast<std::uint8_t*>(dst.data());
  std::vector<rmm::cuda_stream> streams(n_stream);
  auto const bytes = file_bytes / window * window;

  // CUCASCADE_IO_BENCH_ONLY=<substring>[,<substring>...] runs only the scenarios whose
  // name holds one of them.
  std::vector<std::string> only;
  if (char const* only_env = std::getenv("CUCASCADE_IO_BENCH_ONLY"); only_env != nullptr) {
    std::string list{only_env};
    for (std::size_t start = 0; start <= list.size();) {
      auto const comma = std::min(list.find(',', start), list.size());
      if (comma > start) only.push_back(list.substr(start, comma - start));
      start = comma + 1;
    }
  }
  auto selected = [&](std::string const& name) {
    return only.empty() || std::any_of(only.begin(), only.end(), [&](std::string const& part) {
             return name.find(part) != std::string::npos;
           });
  };
  auto scenario = [&](std::string const& name, std::size_t total, auto&& run) {
    if (!selected(name)) return;
    auto const gbps = measure(path, total, rounds, run);
    std::printf("%-8s %-28s %7.2f GB/s\n", label.c_str(), name.c_str(), gbps);
    std::fflush(stdout);
  };

  for (std::size_t const piece : {256UL << 20, 1UL << 30}) {
    scenario("dev 1 slice x " + std::to_string(piece >> 20) + " MiB, QD1", bytes, [&] {
      for (std::size_t off = 0; off < bytes; off += piece) {
        REQUIRE(ctx.device_read_async_io(*object, off, piece, base, streams[0]).get(120s) == piece);
      }
    });
  }

  for (std::size_t const piece : {4UL << 20, 16UL << 20}) {
    scenario("dev readv " + std::to_string(piece >> 20) + " MiB, 1 stream", bytes, [&] {
      for (std::size_t off = 0; off < bytes; off += window) {
        std::vector<cucascade::io::slice> slices;
        for (std::size_t p = 0; p < window; p += piece) {
          slices.emplace_back(off + p, piece, base + p);
        }
        REQUIRE(ctx.device_readv_async_io(*object, slices, streams[0]).get(120s) == window);
      }
    });
  }

  for (std::size_t const piece : {64UL << 10, 256UL << 10, 1UL << 20, 8UL << 20}) {
    auto const total = piece < (1UL << 20) ? std::min(bytes, std::size_t{1} << 30) : bytes;
    scenario("dev " + std::to_string(piece >> 10) + " KiB x 8 streams", total, [&] {
      for (std::size_t off = 0; off < total; off += window) {
        std::vector<cucascade::exec::semi_future<std::size_t>> futures;
        std::size_t i = 0;
        for (std::size_t p = 0; p < window; p += piece, ++i) {
          futures.push_back(
            ctx.device_read_async_io(*object, off + p, piece, base + p, streams[i % n_stream]));
        }
        for (auto& future : futures) {
          REQUIRE(std::move(future).get(120s) == piece);
        }
      }
    });
  }

  {
    constexpr std::size_t piece = 4UL << 20;
    std::vector<std::uint8_t> host(window);
    scenario("host 4 MiB, all queued", bytes, [&] {
      for (std::size_t off = 0; off < bytes; off += window) {
        std::vector<cucascade::exec::semi_future<std::size_t>> futures;
        for (std::size_t p = 0; p < window; p += piece) {
          futures.push_back(ctx.host_read_async_io(*object, off + p, piece, host.data() + p));
        }
        for (auto& future : futures) {
          REQUIRE(std::move(future).get(120s) == piece);
        }
      }
    });
  }
}

}  // namespace

TEST_CASE("cufile throughput probe", "[.cufile-bench]")
{
  char const* path_env = std::getenv("CUCASCADE_IO_BENCH_FILE");
  if (path_env == nullptr || !has_cuda_device()) {
    SKIP("set CUCASCADE_IO_BENCH_FILE (and have a GPU) to run the probe");
  }
  std::string const path  = path_env;
  auto const file_bytes   = env_size("CUCASCADE_IO_BENCH_BYTES", 4UL << 30);
  auto const rounds       = static_cast<int>(env_size("CUCASCADE_IO_BENCH_ROUNDS", 3));
  auto const reactors     = env_size("CUCASCADE_IO_BENCH_REACTORS", 4);
  char const* backend     = std::getenv("CUCASCADE_IO_BENCH_BACKEND");
  std::string const which = backend != nullptr ? backend : "both";
  ensure_bench_file(path, file_bytes);

  if (which == "both" || which == "cufile") {
    config cfg{};
    cfg.gds_threshold = env_size("CUCASCADE_IO_BENCH_GDS_THRESHOLD", cfg.gds_threshold);
    started_ioctx ioctx{cfg, reactors};
    run_probe("cufile", ioctx.get(), path, file_bytes, rounds);
  }
  if (which == "both" || which == "uring") {
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, 2UL << 30, 2UL << 30, 1UL << 20, 128, 8};
    cucascade::io::uring::uring_ioctx ioctx{
      reactors,
      std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
        cucascade::io::uring::config{}, &mr)};
    ioctx.start();
    run_probe("uring", ioctx, path, file_bytes, rounds);
    ioctx.shutdown();
  }
}

namespace {

struct cufile_stream_args_probe {
  std::size_t size{0};
  off_t file_offset{0};
  off_t devPtr_offset{0};
  ssize_t bytes_done{0};
};

/// Raw cuFile submission-mode experiment: sync cuFileRead from T threads vs cuFileReadAsync on K
/// lane streams from one thread, op size swept, registered vs unregistered destination.
void raw_cufile_probe(std::string const& path, std::size_t file_bytes, int rounds)
{
  auto object = cufile_reactor::create_io_object(path);
  if (!object->cufile_registered()) {
    std::printf("raw: file not registered with cuFile; skipping\n");
    return;
  }
  auto& api                    = kvikio::cuFileAPI::instance();
  constexpr std::size_t window = 1UL << 30;
  rmm::cuda_stream alloc_stream;
  rmm::device_buffer dst(window, alloc_stream);
  alloc_stream.synchronize();
  auto* const base = static_cast<std::uint8_t*>(dst.data());
  auto const bytes = file_bytes / window * window;
  // cuFile keeps per-stream state keyed by the handle: lanes live for the whole probe and are
  // deregistered before they are destroyed (a destroyed-then-reused handle breaks cuFile).
  std::vector<rmm::cuda_stream> lane(16);
  auto report = [](std::string const& what, double gbps) {
    std::printf("raw      %-40s %7.2f GB/s\n", what.c_str(), gbps);
    std::fflush(stdout);
  };

  for (bool const registered : {false, true}) {
    if (registered) {
      auto const t0 = std::chrono::steady_clock::now();
      auto const st = api.BufRegister(base, window, 0);
      auto const reg =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0);
      std::printf("raw      cuFileBufRegister 1 GiB: err=%d in %.1f ms\n",
                  static_cast<int>(st.err),
                  reg.count());
      if (st.err != CU_FILE_SUCCESS) return;
    }
    std::string const tag = registered ? " [reg]" : "";
    bool const with_async = env_size("CUCASCADE_IO_BENCH_ASYNC", 1) != 0;
    for (std::size_t const op : {256UL << 10, 1UL << 20, 4UL << 20, 16UL << 20, 64UL << 20}) {
      for (std::size_t const threads : {1UL, 4UL, 8UL}) {
        report(
          "sync " + std::to_string(op >> 10) + " KiB x " + std::to_string(threads) + " thr" + tag,
          measure(path, op < (4UL << 20) ? window : bytes, rounds, [&] {
            auto const bytes_here = op < (4UL << 20) ? window : bytes;
            std::atomic<std::size_t> next{0};
            auto worker = [&] {
              CUCASCADE_CUDA_TRY(cudaSetDevice(0));
              for (;;) {
                auto const off = next.fetch_add(op);
                if (off >= bytes_here) break;
                auto const in_window = off % window;
                ssize_t got =
                  registered
                    ? api.Read(object->cufile_handle(),
                               base,
                               op,
                               static_cast<off_t>(off),
                               static_cast<off_t>(in_window))
                    : api.Read(
                        object->cufile_handle(), base + in_window, op, static_cast<off_t>(off), 0);
                REQUIRE(got == static_cast<ssize_t>(op));
              }
            };
            std::vector<std::thread> pool;
            for (std::size_t t = 0; t < threads; ++t)
              pool.emplace_back(worker);
            for (auto& t : pool)
              t.join();
          }));
      }
      for (std::size_t const lanes : {1UL, 4UL, 8UL, 16UL}) {
        if (!with_async) break;
        report(
          "async " + std::to_string(op >> 20) + " MiB x " + std::to_string(lanes) + " lanes" + tag,
          measure(path, bytes, rounds, [&] {
            std::vector<cufile_stream_args_probe> args(bytes / op);
            for (std::size_t i = 0, off = 0; off < bytes; off += op, ++i) {
              auto& a         = args[i];
              a.size          = op;
              a.file_offset   = static_cast<off_t>(off);
              a.devPtr_offset = registered ? static_cast<off_t>(off % window) : 0;
              a.bytes_done    = 0;
              void* ptr       = registered ? base : base + off % window;
              auto const st   = api.ReadAsync(object->cufile_handle(),
                                            ptr,
                                            &a.size,
                                            &a.file_offset,
                                            &a.devPtr_offset,
                                            &a.bytes_done,
                                            lane[i % lanes].value());
              REQUIRE(st.err == CU_FILE_SUCCESS);
              // Bound the outstanding ops to 2 per lane, as a reactor's window would.
              if (i >= 2 * lanes) { lane[(i - 2 * lanes) % lanes].synchronize(); }
            }
            for (std::size_t l = 0; l < lanes; ++l)
              lane[l].synchronize();
            for (auto& a : args)
              REQUIRE(a.bytes_done == static_cast<ssize_t>(op));
          }));
      }
    }
    if (registered) { std::ignore = api.BufDeregister(base); }
  }
  for (auto& l : lane)
    std::ignore = api.StreamDeregister(l.value());
}

}  // namespace

TEST_CASE("cufile raw submission-mode probe", "[.cufile-probe-api]")
{
  char const* path_env = std::getenv("CUCASCADE_IO_BENCH_FILE");
  if (path_env == nullptr || !has_cuda_device()) { SKIP("set CUCASCADE_IO_BENCH_FILE"); }
  auto const file_bytes = env_size("CUCASCADE_IO_BENCH_BYTES", 4UL << 30);
  ensure_bench_file(path_env, file_bytes);
  kvikio::cuFileAPI::instance().driver_open();
  raw_cufile_probe(
    path_env, file_bytes, static_cast<int>(env_size("CUCASCADE_IO_BENCH_ROUNDS", 1)));
}

TEST_CASE("cufile async submit timing probe", "[.cufile-probe-submit]")
{
  char const* path_env = std::getenv("CUCASCADE_IO_BENCH_FILE");
  if (path_env == nullptr || !has_cuda_device()) { SKIP("set CUCASCADE_IO_BENCH_FILE"); }
  auto const op    = env_size("CUCASCADE_IO_BENCH_OP", 16UL << 20);
  auto const count = env_size("CUCASCADE_IO_BENCH_COUNT", 64);
  auto object      = cufile_reactor::create_io_object(path_env);
  REQUIRE(object->cufile_registered());
  auto& api = kvikio::cuFileAPI::instance();
  rmm::cuda_stream stream;
  rmm::device_buffer dst(op * count, stream);
  stream.synchronize();
  std::vector<cufile_stream_args_probe> args(count);
  double submit_ms = 0;
  auto const t0    = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < count; ++i) {
    auto& a       = args[i];
    a.size        = op;
    a.file_offset = static_cast<off_t>(i * op);
    auto const s0 = std::chrono::steady_clock::now();
    auto const st = api.ReadAsync(object->cufile_handle(),
                                  static_cast<std::uint8_t*>(dst.data()) + i * op,
                                  &a.size,
                                  &a.file_offset,
                                  &a.devPtr_offset,
                                  &a.bytes_done,
                                  stream.value());
    submit_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    REQUIRE(st.err == CU_FILE_SUCCESS);
  }
  auto const t1 = std::chrono::steady_clock::now();
  stream.synchronize();
  auto const t2   = std::chrono::steady_clock::now();
  std::size_t bad = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (args[i].bytes_done != static_cast<ssize_t>(op)) {
      if (bad++ < 3) std::printf("op %zu bytes_done %zd\n", i, args[i].bytes_done);
    }
  }
  std::printf(
    "op %zu MiB x %zu: submit total %.1f ms (%.2f ms/op), wait %.1f ms, %.2f GB/s, bad %zu\n",
    op >> 20,
    count,
    submit_ms,
    submit_ms / static_cast<double>(count),
    std::chrono::duration<double, std::milli>(t2 - t1).count(),
    static_cast<double>(op * count) / std::chrono::duration<double>(t2 - t0).count() / 1e9,
    bad);
  std::ignore = api.StreamDeregister(stream.value());
}

TEST_CASE("cufile raw write probe", "[.cufile-probe-write]")
{
  char const* path_env = std::getenv("CUCASCADE_IO_BENCH_WFILE");
  if (path_env == nullptr || !has_cuda_device()) { SKIP("set CUCASCADE_IO_BENCH_WFILE"); }
  std::string const path = path_env;
  auto const bytes       = env_size("CUCASCADE_IO_BENCH_BYTES", 4UL << 30);
  auto& api              = kvikio::cuFileAPI::instance();
  api.driver_open();
  constexpr std::size_t window = 1UL << 30;
  rmm::cuda_stream alloc_stream;
  rmm::device_buffer src(window, alloc_stream);
  REQUIRE(cudaMemsetAsync(src.data(), 0x5A, window, alloc_stream.value()) == cudaSuccess);
  alloc_stream.synchronize();
  auto* const base = static_cast<std::uint8_t*>(src.data());
  for (std::size_t const op : {1UL << 20, 4UL << 20, 16UL << 20, 64UL << 20}) {
    for (std::size_t const threads : {1UL, 4UL, 8UL}) {
      std::filesystem::remove(path);
      file_descriptor fd{::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644)};
      REQUIRE(fd);
      CUfileDescr_t descr{};
      descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      descr.handle.fd = fd.get();
      CUfileHandle_t handle{};
      REQUIRE(api.HandleRegister(&handle, &descr).err == CU_FILE_SUCCESS);
      std::atomic<std::size_t> next{0};
      auto const t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> pool;
      for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
          CUCASCADE_CUDA_TRY(cudaSetDevice(0));
          for (;;) {
            auto const off = next.fetch_add(op);
            if (off >= bytes) break;
            REQUIRE(api.Write(handle, base + off % window, op, static_cast<off_t>(off), 0) ==
                    static_cast<ssize_t>(op));
          }
        });
      }
      for (auto& t : pool)
        t.join();
      auto const t1 = std::chrono::steady_clock::now();
      REQUIRE(::fdatasync(fd.get()) == 0);
      auto const t2 = std::chrono::steady_clock::now();
      api.HandleDeregister(handle);
      std::printf(
        "raw      write %5zu KiB x %zu thr: %.2f GB/s (%.2f GB/s incl. fdatasync)\n",
        op >> 10,
        threads,
        static_cast<double>(bytes) / std::chrono::duration<double>(t1 - t0).count() / 1e9,
        static_cast<double>(bytes) / std::chrono::duration<double>(t2 - t0).count() / 1e9);
      std::fflush(stdout);
    }
  }
  std::filesystem::remove(path);
}
