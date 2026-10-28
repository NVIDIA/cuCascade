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

#include "io/churning_file.hpp"

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

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

/// splitmix64 of @p offset: never periodic, so bytes landing a whole number of
/// pieces / scratch slots away from their place cannot compare equal.
[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  std::uint64_t z = offset + 0x9E3779B97F4A7C15ULL;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<std::uint8_t>(z ^ (z >> 31));
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

/**
 * Copy @p bytes of @p device back on a NEW non-blocking stream, without
 * synchronizing the stream the read was issued on: a read is complete when its
 * future resolves, so the bytes must already be in place.
 */
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

/** Delays every staging copy of the cuFile reactors for the guard's lifetime. */
class staging_copy_delay {
 public:
  explicit staging_copy_delay(std::chrono::microseconds delay)
  {
    cucascade::io::cufile::detail::set_staging_copy_delay_for_testing(delay);
  }
  ~staging_copy_delay()
  {
    cucascade::io::cufile::detail::set_staging_copy_delay_for_testing(std::chrono::microseconds{0});
  }
  staging_copy_delay(staging_copy_delay const&)            = delete;
  staging_copy_delay& operator=(staging_copy_delay const&) = delete;
};

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
    auto object =
      cufile_reactor::create_io_object(file.path(), true, cucascade::io::cufile::compat::on);
    CHECK(object->size() == file.size());
    CHECK_FALSE(object->cufile_registered());
    CHECK(object->direct_fd() == -1);
    CHECK(object->buffered_fd() >= 0);
  }

  SECTION("without O_DIRECT nothing can be registered")
  {
    auto object = cufile_reactor::create_io_object(
      file.path(), false, cucascade::io::cufile::compat::automatic);
    CHECK_FALSE(object->cufile_registered());
    CHECK(object->direct_fd() == -1);
  }

  SECTION("OFF enforces cuFile, so it cannot be combined with use_odirect=false")
  {
    CHECK_THROWS_AS(
      cufile_reactor::create_io_object(file.path(), false, cucascade::io::cufile::compat::off),
      std::invalid_argument);
  }

  SECTION("OFF either registers the file or refuses it, and never silently degrades")
  {
    std::unique_ptr<cufile_io_object> object;
    try {
      object =
        cufile_reactor::create_io_object(file.path(), true, cucascade::io::cufile::compat::off);
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

TEST_CASE("cufile opens carry the stat identity of their descriptors",
          "[io][cufile][object_identity]")
{
  using cucascade::io::capture_local_identity;
  using cucascade::io::identity_kind;
  using cucascade::io::object_identity;
  pattern_file const file{1 << 20};

  for (auto const mode :
       {cucascade::io::cufile::compat::on, cucascade::io::cufile::compat::automatic}) {
    auto const first    = cufile_reactor::create_io_object(file.path(), true, mode);
    auto const identity = first->identity();
    CHECK(identity.kind == identity_kind::local_stat);
    CHECK(identity.size == file.size());
    CHECK(first->size() == file.size());
    // The O_DIRECT descriptor, when the filesystem has one, names the same file.
    if (first->direct_fd() >= 0) {
      object_identity direct;
      REQUIRE(capture_local_identity(first->direct_fd(), direct));
      CHECK(direct == identity);
    }
    // Two opens of the unchanged file share one version, whichever reactor
    // opened them.
    auto const second = cufile_reactor::create_io_object(file.path(), true, mode);
    CHECK(second->identity() == identity);
    CHECK(second->identity_cache_key() == first->identity_cache_key());
    CHECK(cucascade::io::uring::uring_reactor::create_io_object(file.path())->identity() ==
          identity);
  }
}

TEST_CASE("cufile opens of a path being replaced report the file they opened",
          "[io][cufile][object_identity]")
{
  // Every open races an atomic replacement of its path: the object's identity
  // and size describe the file its buffered descriptor holds, and its O_DIRECT
  // descriptor (reopened from that one) holds the same file, so no open sees a
  // mismatch to retry or throw on.
  cucascade::test::churning_file const file;
  for (auto const mode :
       {cucascade::io::cufile::compat::on, cucascade::io::cufile::compat::automatic}) {
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    auto const baseline = file.replacements();
    for (int opens = 0; opens < 100 || (file.replacements() < baseline + 100 &&
                                        std::chrono::steady_clock::now() < deadline);
         ++opens) {
      auto const object = cufile_reactor::create_io_object(file.path(), true, mode);
      cucascade::test::churning_file::check_opened(
        object->identity(), object->size(), object->buffered_fd(), object->direct_fd());
    }
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

TEST_CASE("cufile ioctx host reads land in place on every alignment", "[io][cufile]")
{
  // Host-only reads go O_DIRECT: straight into a block-aligned destination for a
  // block-aligned range, else whole blocks through the scratch (8 MiB slots).
  static constexpr std::size_t file_bytes = (24UL << 20) + 5;
  pattern_file const file{file_bytes};
  started_ioctx ioctx{config{}};
  auto object = ioctx.get().open_io_object(file.path());

  struct read_case {
    std::string label;
    std::size_t offset;
    std::size_t size;
    std::size_t misalignment;
  };
  auto const [label, offset, size, misalignment] =
    GENERATE(read_case{"aligned, direct", 4096, 8UL << 20, 0},
             read_case{"aligned range, misaligned buffer", 4096, 8UL << 20, 8},
             read_case{"unaligned, several slots", 4097, (20UL << 20) + 123, 8},
             read_case{"to the unaligned end of file", 3UL << 20, file_bytes - (3UL << 20), 0},
             read_case{"inside one block", 10, 100, 3});
  DYNAMIC_SECTION(label)
  {
    std::vector<std::uint8_t> storage(size + misalignment + 4096);
    auto* const base =
      storage.data() + (4096 - reinterpret_cast<std::uintptr_t>(storage.data()) % 4096) % 4096;
    auto* const dst = base + misalignment;
    REQUIRE(ioctx.get().host_read_async_io(*object, offset, size, dst).get(30s) == size);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < size; ++i) {
      if (dst[i] != pattern_byte(offset + i)) ++bad;
    }
    CHECK(bad == 0);
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

  auto const [label, cfg] =
    GENERATE(std::pair<std::string, config>{"below the threshold",
                                            [] {
                                              config c{};
                                              c.gds_threshold = 1UL << 40;
                                              return c;
                                            }()},
             std::pair<std::string, config>{"file not registered", [] {
                                              config c{};
                                              c.compat_mode = cucascade::io::cufile::compat::on;
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

TEST_CASE("cufile ioctx device reads are complete when their future resolves", "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // The destination's stream is held, so the read must wait for it; the bytes
  // are then checked on another stream.  Sizes above and below the two 8 MiB
  // scratch slots exercise slot reuse and the final drain of the staged path.
  struct read_case {
    std::string label;
    std::size_t threshold;
    std::size_t size;
  };
  auto const [label, threshold, size] =
    GENERATE(read_case{"staged, slots reused", 1UL << 40, (20UL << 20) + 123},
             read_case{"staged, within two slots", 1UL << 40, 12UL << 20},
             read_case{"cuFile", 1, (20UL << 20) + 123});
  DYNAMIC_SECTION(label)
  {
    pattern_file const file{size};
    config cfg{};
    cfg.gds_threshold = threshold;
    started_ioctx ioctx{cfg, 1};
    auto object = ioctx.get().open_io_object(file.path());
    rmm::cuda_stream stream;
    rmm::device_buffer dst(size, stream);
    stream.synchronize();
    hold_stream(stream.value(), std::chrono::milliseconds{300});
    // Slow every staging copy down, so a slot reused or a request settled before
    // its copy landed shows up as wrong bytes.
    staging_copy_delay const slow{std::chrono::milliseconds{50}};
    REQUIRE(
      ioctx.get()
        .device_read_async_io(*object, 0, size, static_cast<std::uint8_t*>(dst.data()), stream)
        .get(30s) == size);
    auto const got  = device_bytes_elsewhere(dst.data(), size);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < size; ++i) {
      if (got[i] != pattern_byte(i)) ++bad;
    }
    CHECK(bad == 0);
  }
}

TEST_CASE("cufile next_piece keeps cuFile reads aligned and splits the EOF tail", "[io][cufile]")
{
  using cucascade::io::cufile::detail::next_piece;
  constexpr std::size_t KiB = 1024;
  constexpr std::size_t MiB = 1024 * KiB;
  // Short ranges are read whole.
  CHECK(next_piece(0, 1 * MiB, 10 * MiB, 64 * MiB) == 1 * MiB);
  // A long range stops at max_op; an unaligned start stops at the last 4 KiB
  // boundary inside max_op, so the next piece starts aligned.
  CHECK(next_piece(0, 100 * MiB, 200 * MiB, 64 * MiB) == 64 * MiB);
  CHECK(next_piece(123, 100 * MiB, 200 * MiB, 64 * MiB) == 64 * MiB - 123);
  CHECK((123 + next_piece(123, 100 * MiB, 200 * MiB, 64 * MiB)) % (4 * KiB) == 0);
  // A piece reaching an unaligned end of file stops at align_down(file_size) ...
  constexpr std::size_t file_size = 10 * MiB + 1234;
  CHECK(next_piece(0, file_size, file_size, 64 * MiB) == 10 * MiB);
  // ... and the sub-block tail is its own piece.
  CHECK(next_piece(10 * MiB, file_size, file_size, 64 * MiB) == 1234);
  // A range that ends before EOF is not split there.
  CHECK(next_piece(0, 10 * MiB + 1000, file_size, 64 * MiB) == 10 * MiB + 1000);
  // A range wholly inside the last block is read whole.
  CHECK(next_piece(10 * MiB + 100, file_size, file_size, 64 * MiB) == 1134);
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

TEST_CASE("cufile ioctx vectored device reads fan out and settle as a whole", "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // More slices than the base selection's two reactors: the cuFile ioctx spreads
  // them over all four.  The destination stream is held and every staging copy
  // slowed, so a future that settled before its last slice landed shows up as
  // wrong bytes read back on another stream.
  static constexpr std::size_t file_bytes = 24UL << 20;
  static constexpr std::size_t count      = 12;
  static constexpr std::size_t stride     = 2UL << 20;
  static constexpr std::size_t length     = (1UL << 20) + 4097;
  pattern_file const file{file_bytes};
  auto const threshold = GENERATE(std::size_t{1}, std::size_t{1} << 40);
  CAPTURE(threshold);
  config cfg{};
  cfg.gds_threshold = threshold;
  started_ioctx ioctx{cfg, 4};
  auto object = ioctx.get().open_io_object(file.path());

  rmm::cuda_stream stream;
  rmm::device_buffer dst(count * stride, stream);
  stream.synchronize();
  auto* const base = static_cast<std::uint8_t*>(dst.data());
  std::vector<cucascade::io::slice> slices;
  for (std::size_t i = 0; i < count; ++i) {
    // Visit the file out of order, from unaligned offsets.
    auto const offset = ((i * 7) % count) * stride + 13;
    slices.emplace_back(offset, length, base + i * stride);
  }
  hold_stream(stream.value(), std::chrono::milliseconds{200});
  staging_copy_delay const slow{std::chrono::milliseconds{5}};
  REQUIRE(ioctx.get().device_readv_async_io(*object, slices, stream).get(60s) == count * length);

  auto const got = device_bytes_elsewhere(base, count * stride);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<std::uint8_t> part(got.begin() + static_cast<std::ptrdiff_t>(i * stride),
                                   got.begin() + static_cast<std::ptrdiff_t>(i * stride + length));
    CHECK(matches_file(part, slices[i].rng.offset, length));
  }
}

TEST_CASE("cufile ioctx vectored host reads fan out over every reactor", "[io][cufile]")
{
  static constexpr std::size_t file_bytes = 6UL << 20;
  pattern_file const file{file_bytes};
  started_ioctx ioctx{config{}, 4};
  auto object = ioctx.get().open_io_object(file.path());

  constexpr std::size_t count  = 16;
  constexpr std::size_t length = 100'003;
  std::vector<std::vector<std::uint8_t>> parts(count, std::vector<std::uint8_t>(length));
  std::vector<cucascade::io::slice> slices;
  for (std::size_t i = 0; i < count; ++i) {
    slices.emplace_back(i * 300'007, length, parts[i].data());
  }
  REQUIRE(ioctx.get().host_readv_async_io(*object, slices).get(30s) == count * length);
  for (std::size_t i = 0; i < count; ++i) {
    CHECK(matches_file(parts[i], slices[i].rng.offset, length));
  }
}

TEST_CASE("cufile reactor serves other requests while a device slice waits for its stream",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};
  auto const threshold = GENERATE(std::size_t{1}, std::size_t{1} << 40);
  CAPTURE(threshold);
  config cfg{};
  cfg.gds_threshold = threshold;
  started_ioctx ioctx{cfg, 1};  // one reactor: everything below shares its worker
  auto object = ioctx.get().open_io_object(file.path());

  rmm::cuda_stream held;
  rmm::cuda_stream idle;
  rmm::device_buffer held_dst(size, held);
  rmm::device_buffer idle_dst(size, idle);
  held.synchronize();
  idle.synchronize();
  hold_stream(held.value(), std::chrono::milliseconds{2000});

  auto waiting = ioctx.get().device_read_async_io(
    *object, 0, size, static_cast<std::uint8_t*>(held_dst.data()), held);
  auto served = ioctx.get().device_read_async_io(
    *object, 0, size, static_cast<std::uint8_t*>(idle_dst.data()), idle);
  std::vector<std::uint8_t> host(64UL << 10);
  auto host_read = ioctx.get().host_read_async_io(*object, 4096, host.size(), host.data());

  // Both later requests complete while the first still waits for its stream.
  REQUIRE(std::move(served).get(1500ms) == size);
  REQUIRE(std::move(host_read).get(1500ms) == host.size());
  CHECK_FALSE(waiting.is_ready());
  REQUIRE(std::move(waiting).get(30s) == size);
  CHECK(matches_file(device_bytes_elsewhere(held_dst.data(), size), 0, size));
  CHECK(matches_file(device_bytes_elsewhere(idle_dst.data(), size), 0, size));
}

TEST_CASE("cufile settles a parked device slice at once when a sibling slice fails",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // Slice 0 parks behind a held stream; slice 1 (a null host destination) fails
  // the shared request.  The request must settle with that error now, not when
  // the unrelated stream drains: nothing was issued for the parked slice.
  constexpr std::size_t size = 1UL << 20;
  pattern_file const file{size};
  cufile_reactor reactor{std::make_shared<cufile_reactor::reactor_context>(config{}), ""};
  reactor.start();
  std::shared_ptr<cucascade::io::io_object const> object{
    cufile_reactor::create_io_object(file.path())};

  rmm::cuda_stream held;
  rmm::device_buffer dst(size, held);
  held.synchronize();
  hold_stream(held.value(), std::chrono::milliseconds{3000});

  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, size},
                      cucascade::io::device_buffer{static_cast<std::uint8_t*>(dst.data()), held});
  host_buffer null_host;
  null_host.buffer = static_cast<std::uint8_t*>(nullptr);
  slices.emplace_back(range{0, 4096}, std::move(null_host));
  auto coordinator = std::make_shared<grouped_coordinator>(size + 4096, 2);
  auto future      = coordinator->get_future();
  auto const start = std::chrono::steady_clock::now();
  reactor.enqueue(grouped_io_request::create(object, std::move(slices), std::move(coordinator)));

  CHECK_THROWS(std::move(future).get(2s));
  CHECK(std::chrono::steady_clock::now() - start < 1500ms);
  held.synchronize();
  reactor.shutdown();
}

namespace {

/** Makes the reactors report a failed event record after each H2D copy, for the guard's life. */
class copy_event_record_failure {
 public:
  copy_event_record_failure()
  {
    cucascade::io::cufile::detail::set_copy_event_record_failure_for_testing(true);
  }
  ~copy_event_record_failure()
  {
    cucascade::io::cufile::detail::set_copy_event_record_failure_for_testing(false);
  }
  copy_event_record_failure(copy_event_record_failure const&)            = delete;
  copy_event_record_failure& operator=(copy_event_record_failure const&) = delete;
};

}  // namespace

TEST_CASE("cufile reads that fail after enqueuing a copy settle only once it landed",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  // The copy is held back (a delayed staging stream, or a held caller stream)
  // and its event record "fails".  The read reports the error, but only after
  // the queued copy finished: the bytes are in place when the future settles,
  // so the caller may free the destination and the scratch may be reused.
  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};

  SECTION("staged device read")
  {
    config cfg{};
    cfg.gds_threshold = 1UL << 40;
    started_ioctx ioctx{cfg, 1};
    auto object = ioctx.get().open_io_object(file.path());
    rmm::cuda_stream stream;
    rmm::device_buffer dst(size, stream);
    REQUIRE(cudaMemsetAsync(dst.data(), 0xAB, size, stream.value()) == cudaSuccess);
    stream.synchronize();
    staging_copy_delay const slow{std::chrono::milliseconds{300}};
    copy_event_record_failure const failing;
    CHECK_THROWS(
      ioctx.get()
        .device_read_async_io(*object, 0, size, static_cast<std::uint8_t*>(dst.data()), stream)
        .get(30s));
    CHECK(matches_file(device_bytes_elsewhere(dst.data(), size), 0, size));
  }

  SECTION("host + device pair")
  {
    started_ioctx ioctx{config{}, 1};
    auto object = ioctx.get().open_io_object(file.path());
    rmm::cuda_stream stream;
    rmm::device_buffer dst(size, stream);
    REQUIRE(cudaMemsetAsync(dst.data(), 0xAB, size, stream.value()) == cudaSuccess);
    stream.synchronize();
    // Pinned, so the H2D copy is truly asynchronous (a pageable source is
    // staged synchronously by the driver).
    void* pinned = nullptr;
    REQUIRE(cudaMallocHost(&pinned, size) == cudaSuccess);
    std::unique_ptr<void, void (*)(void*)> const host_owner{
      pinned, [](void* p) { std::ignore = cudaFreeHost(p); }};
    auto* const host = static_cast<std::uint8_t*>(pinned);
    hold_stream(stream.value(), std::chrono::milliseconds{300});
    copy_event_record_failure const failing;
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(
      range{0, size},
      host_buffer{host},
      cucascade::io::device_buffer{static_cast<std::uint8_t*>(dst.data()), stream});
    CHECK_THROWS(ioctx.get().host_device_readv_async_io(*object, std::move(slices)).get(30s));
    CHECK(matches_file(std::vector<std::uint8_t>(host, host + size), 0, size));
    CHECK(matches_file(device_bytes_elsewhere(dst.data(), size), 0, size));
  }
}

namespace {

/// The errc a failed read future carries (errc{} when it did not fail with a system_error).
[[nodiscard]] std::errc read_error(cucascade::exec::semi_future<std::size_t>&& future)
{
  try {
    std::ignore = std::move(future).get(30s);
  } catch (std::system_error const& error) {
    return static_cast<std::errc>(error.code().value());
  } catch (...) {
  }
  return std::errc{};
}

/// The cuFile read paths: cuFileRead for large device reads, or staged POSIX.
[[nodiscard]] std::vector<std::pair<char const*, config>> read_paths()
{
  config cufile_reads{};
  cufile_reads.gds_threshold = cucascade::io::IO_BLOCK_SIZE;
  config staged{};
  staged.gds_threshold = 1UL << 40;
  return {{"cuFileRead", cufile_reads}, {"staged POSIX", staged}};
}

}  // namespace

TEST_CASE("cufile device reads past EOF resolve with the bytes delivered", "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const [label, cfg] = GENERATE(from_range(read_paths()));
  CAPTURE(label);
  constexpr std::size_t size = (1UL << 20) + 123;
  pattern_file const file{size};
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  rmm::cuda_stream stream;
  rmm::device_buffer dst(2 * size, stream);
  auto* const out = static_cast<std::uint8_t*>(dst.data());

  CHECK(ioctx.get().device_read_async_io(*object, 0, 2 * size, out, stream).get(30s) == size);
  CHECK(matches_file(device_bytes_elsewhere(out, size), 0, size));
  CHECK(ioctx.get().device_read_async_io(*object, size - 100, 5000, out, stream).get(30s) == 100);
  CHECK(ioctx.get().device_read_async_io(*object, size, 4096, out, stream).get(30s) == 0);
  using cucascade::io::slice;
  std::vector<slice> const slices{
    slice{0, 4096, out}, slice{size - 10, 4096, out + 4096}, slice{size + 4096, 4096, out + 8192}};
  CHECK(ioctx.get().device_readv_async_io(*object, slices, stream).get(30s) == 4096 + 10);
}

TEST_CASE("cufile reads of a file truncated after open never report unread bytes",
          "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const [label, cfg] = GENERATE(from_range(read_paths()));
  CAPTURE(label);
  // The object keeps the size it was opened with.  A read the file can no longer
  // fill fails (EIO) instead of resolving with the requested count and an
  // untouched tail.
  constexpr std::size_t size = 4UL << 20;
  pattern_file const file{size};
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(object->size() == size);
  std::filesystem::resize_file(file.path(), size / 2 + 100);
  rmm::cuda_stream stream;
  rmm::device_buffer dst(size, stream);
  auto* const out = static_cast<std::uint8_t*>(dst.data());

  CHECK(read_error(ioctx.get().device_read_async_io(*object, 0, size, out, stream)) ==
        std::errc::io_error);
  CHECK(read_error(ioctx.get().device_read_async_io(*object, size - 8192, 8192, out, stream)) ==
        std::errc::io_error);
  std::vector<std::uint8_t> host(size);
  CHECK(read_error(ioctx.get().host_read_async_io(*object, 0, size, host.data())) ==
        std::errc::io_error);
  CHECK(ioctx.get().host_read_io(*object, 0, size, host.data()) == size / 2 + 100);
  CHECK(ioctx.get().device_read_async_io(*object, 0, size / 2, out, stream).get(30s) == size / 2);
}

TEST_CASE("cufile shutdown cancels a device read still waiting for its stream", "[io][cufile][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t size = 1UL << 20;
  pattern_file const file{size};
  // cuFile path, and the staged path after a first read gave the reactor its
  // pinned scratch (which shutdown must hand back without synchronizing the
  // device behind the held stream).
  auto const threshold = GENERATE(std::size_t{1}, std::size_t{1} << 40);
  CAPTURE(threshold);
  config cfg{};
  cfg.gds_threshold = threshold;
  cufile_ioctx ioctx{1, std::make_shared<cufile_reactor::reactor_context>(cfg)};
  ioctx.start();
  auto object = ioctx.open_io_object(file.path());
  rmm::cuda_stream held;
  rmm::device_buffer dst(size, held);
  held.synchronize();
  REQUIRE(ioctx.device_read_async_io(*object, 0, size, static_cast<std::uint8_t*>(dst.data()), held)
            .get(30s) == size);
  hold_stream(held.value(), std::chrono::milliseconds{3000});
  auto future =
    ioctx.device_read_async_io(*object, 0, size, static_cast<std::uint8_t*>(dst.data()), held);
  std::this_thread::sleep_for(50ms);  // let the worker park it

  auto const start = std::chrono::steady_clock::now();
  ioctx.shutdown();
  CHECK(canceled(std::move(future)));
  CHECK(std::chrono::steady_clock::now() - start < 1500ms);
  held.synchronize();
}

TEST_CASE("cufile ioctx cancels reads submitted before start", "[io][cufile]")
{
  pattern_file const file{64 * 1024};
  cufile_ioctx ioctx{1, std::make_shared<cufile_reactor::reactor_context>(config{})};
  auto object = ioctx.open_io_object(file.path());

  std::vector<std::uint8_t> dst(4096);
  CHECK(canceled(ioctx.host_read_async_io(*object, 0, dst.size(), dst.data())));
  ioctx.shutdown();
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
// Read throughput probe (hidden; run explicitly with "[.cufile-bench]").
// Reads an existing operator-supplied file; it never writes to it.
// ---------------------------------------------------------------------------

namespace {

/// Check that @p path is an existing regular file of at least @p bytes.  The
/// probe only reads it: it never creates, extends or truncates the input.
void require_bench_file(std::string const& path, std::size_t bytes)
{
  std::error_code ec;
  INFO("CUCASCADE_IO_BENCH_FILE=" << path << ", CUCASCADE_IO_BENCH_BYTES=" << bytes);
  REQUIRE(std::filesystem::is_regular_file(path, ec));
  REQUIRE(std::filesystem::file_size(path, ec) >= bytes);
  REQUIRE_FALSE(ec);
}

/// Evict @p path from the page cache (no root needed for clean pages).
void drop_file_cache(std::string const& path)
{
  file_descriptor fd{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
  if (!fd) return;
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
  require_bench_file(path, file_bytes);

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
