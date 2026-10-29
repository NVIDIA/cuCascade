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
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using cucascade::io::IO_BLOCK_SIZE;
using cucascade::io::write_mode;
using cucascade::io::write_open_options;
using cucascade::io::write_slice;
using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;

namespace {

using namespace std::chrono_literals;

constexpr std::size_t MiB = 1UL << 20;

[[nodiscard]] bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// splitmix64 of (@p seed, index): never periodic, so bytes landing a whole
/// number of pieces / scratch slots away from their place cannot compare equal.
[[nodiscard]] std::vector<std::uint8_t> pattern(std::size_t bytes, std::size_t seed)
{
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    std::uint64_t z = (static_cast<std::uint64_t>(seed) << 40) + i + 0x9E3779B97F4A7C15ULL;
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    out[i]          = static_cast<std::uint8_t>(z ^ (z >> 31));
  }
  return out;
}

/** A unique scratch path under the temp directory, removed on scope exit. */
class scratch_path {
 public:
  scratch_path()
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_cufile_write_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin"))
  {
  }

  ~scratch_path()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }

  scratch_path(scratch_path const&)            = delete;
  scratch_path& operator=(scratch_path const&) = delete;

  [[nodiscard]] std::string str() const { return _path.string(); }

  void fill(std::vector<std::uint8_t> const& bytes) const
  {
    std::ofstream out(_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<char const*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    REQUIRE(out);
  }

  [[nodiscard]] std::vector<std::uint8_t> contents() const
  {
    std::ifstream in(_path, std::ios::binary);
    REQUIRE(in);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
};

/** The expected file contents: writes applied to a byte vector. */
class shadow_file {
 public:
  void write(std::size_t offset, std::span<std::uint8_t const> bytes)
  {
    if (offset + bytes.size() > _bytes.size()) _bytes.resize(offset + bytes.size(), 0);
    std::copy(bytes.begin(), bytes.end(), _bytes.begin() + static_cast<std::ptrdiff_t>(offset));
  }

  [[nodiscard]] std::vector<std::uint8_t> const& bytes() const noexcept { return _bytes; }

 private:
  std::vector<std::uint8_t> _bytes;
};

/** Host memory at a chosen misalignment from a 4 KiB boundary. */
class host_bytes {
 public:
  host_bytes(std::vector<std::uint8_t> const& contents, std::size_t misalignment)
    : _storage(static_cast<std::uint8_t*>(std::aligned_alloc(
        IO_BLOCK_SIZE,
        std::max(
          IO_BLOCK_SIZE,
          (contents.size() + misalignment + IO_BLOCK_SIZE - 1) / IO_BLOCK_SIZE * IO_BLOCK_SIZE)))),
      _data(_storage + misalignment)
  {
    REQUIRE(_storage != nullptr);
    std::memcpy(_data, contents.data(), contents.size());
  }

  ~host_bytes() { std::free(_storage); }

  host_bytes(host_bytes const&)            = delete;
  host_bytes& operator=(host_bytes const&) = delete;

  [[nodiscard]] std::uint8_t const* get() const noexcept { return _data; }

 private:
  std::uint8_t* _storage;
  std::uint8_t* _data;
};

/** Device memory holding a copy of @p contents at a chosen offset from a 4 KiB boundary. */
class device_bytes {
 public:
  explicit device_bytes(std::vector<std::uint8_t> const& contents, std::size_t misalignment = 0)
    : _size(contents.size())
  {
    REQUIRE(cudaMalloc(&_storage, std::max<std::size_t>(1, _size + misalignment)) == cudaSuccess);
    _data = static_cast<std::uint8_t*>(_storage) + misalignment;
    if (_size != 0) {
      REQUIRE(cudaMemcpy(_data, contents.data(), _size, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    // A pageable cudaMemcpy may return before its DMA landed, and the tests read
    // the bytes on non-blocking streams that do not wait for the legacy stream.
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }

  ~device_bytes() { std::ignore = cudaFree(_storage); }

  device_bytes(device_bytes const&)            = delete;
  device_bytes& operator=(device_bytes const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return _data; }

 private:
  void* _storage{nullptr};
  std::uint8_t* _data{nullptr};
  std::size_t _size;
};

/** A non-default CUDA stream. */
class cuda_stream {
 public:
  cuda_stream()
  {
    REQUIRE(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking) == cudaSuccess);
  }
  ~cuda_stream() { std::ignore = cudaStreamDestroy(_stream); }
  cuda_stream(cuda_stream const&)            = delete;
  cuda_stream& operator=(cuda_stream const&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return _stream; }
  [[nodiscard]] ::cuda::stream_ref ref() const noexcept { return ::cuda::stream_ref{_stream}; }

 private:
  cudaStream_t _stream{nullptr};
};

/** A cufile ioctx over @p cfg, started unless asked otherwise, shut down on scope exit. */
class write_context {
 public:
  explicit write_context(config cfg = config{}, std::size_t n_reactors = 2, bool start = true)
    : _ctx{std::make_shared<cufile_ioctx>(n_reactors,
                                          std::make_shared<cufile_reactor::reactor_context>(cfg))}
  {
    if (start) _ctx->start();
  }

  ~write_context() { _ctx->shutdown(); }

  write_context(write_context const&)            = delete;
  write_context& operator=(write_context const&) = delete;

  [[nodiscard]] cufile_ioctx& get() noexcept { return *_ctx; }
  cufile_ioctx* operator->() noexcept { return _ctx.get(); }

 private:
  std::shared_ptr<cufile_ioctx> _ctx;
};

/**
 * Lets aligned device-write bodies reach cuFileWrite on a compat-only platform
 * (where the reactor would stage them) for the guard's lifetime.
 */
class force_cufile_writes {
 public:
  explicit force_cufile_writes(bool enabled = true)
  {
    cucascade::io::cufile::detail::set_cufile_writes_in_compat_for_testing(enabled);
  }
  ~force_cufile_writes()
  {
    cucascade::io::cufile::detail::set_cufile_writes_in_compat_for_testing(false);
  }
  force_cufile_writes(force_cufile_writes const&)            = delete;
  force_cufile_writes& operator=(force_cufile_writes const&) = delete;
};

/** The ways a device write can be served: cuFile for the body, or staged. */
struct write_path {
  std::string label;
  config cfg;
  bool force_cufile{false};  ///< cuFileWrite even on a compat-only platform
};

[[nodiscard]] std::vector<write_path> write_paths()
{
  config cufile_body{};
  cufile_body.gds_threshold = IO_BLOCK_SIZE;  // every aligned body goes to cuFile
  cufile_body.max_op_bytes  = 1 * MiB;        // ... in several cuFileWrite calls
  config staged{};
  staged.gds_threshold = 1UL << 40;  // nothing goes to cuFile
  config unregistered{};
  unregistered.compat_mode = cucascade::io::cufile::compat::on;  // no cuFile handle at all
  config buffered{};
  buffered.use_odirect   = false;  // POSIX through the page cache only
  buffered.compat_mode   = cucascade::io::cufile::compat::on;
  buffered.gds_threshold = IO_BLOCK_SIZE;
  return {{"cuFile body", cufile_body, true},
          {"default policy", cufile_body, false},
          {"staged", staged},
          {"unregistered file", unregistered},
          {"buffered only", buffered}};
}

/** Read the whole file through a NEW read-only io_object of @p ctx. */
[[nodiscard]] std::vector<std::uint8_t> read_back(cufile_ioctx& ctx, std::string const& path)
{
  auto object = ctx.open_io_object(path);
  std::vector<std::uint8_t> out(object->size());
  if (!out.empty()) {
    REQUIRE(ctx.host_read_async_io(*object, 0, out.size(), out.data()).get(30s) == out.size());
  }
  return out;
}

/**
 * Compare a file's bytes with the expected ones; on a mismatch report where, not
 * the whole (megabytes long) vectors.
 */
void check_same(std::vector<std::uint8_t> const& got, std::vector<std::uint8_t> const& want)
{
  CHECK(got.size() == want.size());
  auto const n        = std::min(got.size(), want.size());
  std::size_t first   = n;
  std::size_t differs = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      if (first == n) first = i;
      ++differs;
    }
  }
  INFO("first differing byte " << first << " of " << n << ", " << differs << " differ");
  CHECK(differs == 0);
}

[[nodiscard]] std::errc error_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return static_cast<std::errc>(e.code().value());
  } catch (...) {
  }
  return std::errc{};
}

template <class Future>
[[nodiscard]] std::exception_ptr failure_of(Future&& future)
{
  try {
    std::ignore = std::forward<Future>(future).get(60s);
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

}  // namespace

TEST_CASE("cufile open_io_object_for_write honours the open modes", "[io][cufile][write]")
{
  write_context ctx;
  scratch_path path;
  path.fill(pattern(10'000, 1));

  SECTION("create_or_truncate empties an existing file")
  {
    auto object = ctx->open_io_object_for_write(path.str());
    CHECK(object->writable());
    CHECK(object->size() == 0);
    CHECK(path.contents().empty());
  }

  SECTION("create_or_open keeps the content")
  {
    auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open});
    CHECK(object->size() == 10'000);
    CHECK(path.contents() == pattern(10'000, 1));
  }

  SECTION("open_existing fails on a missing file, create_new on an existing one")
  {
    scratch_path missing;
    try {
      std::ignore =
        ctx->open_io_object_for_write(missing.str(), {.mode = write_mode::open_existing});
      FAIL("open_existing of a missing file must throw");
    } catch (std::system_error const& e) {
      CHECK(e.code() == std::errc::no_such_file_or_directory);
    }
    try {
      std::ignore = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_new});
      FAIL("create_new of an existing file must throw");
    } catch (std::system_error const& e) {
      CHECK(e.code() == std::errc::file_exists);
    }
  }

  SECTION("a size hint never changes the size")
  {
    scratch_path fresh;
    auto object = ctx->open_io_object_for_write(fresh.str(), {.size_hint = 8 * MiB});
    CHECK(object->size() == 0);
  }

  SECTION("compat mode OFF cannot be combined with use_odirect=false")
  {
    CHECK_THROWS_AS(cufile_reactor::create_io_object_for_write(
                      path.str(), {}, false, cucascade::io::cufile::compat::off),
                    std::invalid_argument);
  }

  SECTION("the write object registers with cuFile like a read object would")
  {
    auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open});
    auto const& typed  = dynamic_cast<cufile_io_object const&>(*object);
    auto reader        = ctx->open_io_object(path.str());
    auto const& rtyped = dynamic_cast<cufile_io_object const&>(*reader);
    CHECK(typed.cufile_registered() == rtyped.cufile_registered());
  }

  CHECK(ctx->supports_write());
  CHECK(ctx->supports_device_write());
}

TEST_CASE("cufile host writes round-trip aligned and unaligned ranges", "[io][cufile][write]")
{
  auto const [label, cfg, force_cufile] = GENERATE(from_range(write_paths()));
  DYNAMIC_SECTION(label)
  {
    force_cufile_writes const force{force_cufile};
    write_context ctx{cfg};
    scratch_path path;
    auto object = ctx->open_io_object_for_write(path.str());
    shadow_file shadow;

    std::size_t seed = 0;
    for (std::size_t const offset : {0UL, 1UL, 4095UL, 4096UL, 4097UL, 3 * MiB + 5}) {
      for (std::size_t const size : {1UL, 4095UL, 4096UL, 4097UL, 3 * MiB + 7}) {
        auto const bytes = pattern(size, ++seed);
        host_bytes const src{bytes, seed % 2 == 0 ? std::size_t{0} : std::size_t{8}};
        if (seed % 3 == 0) {
          REQUIRE(ctx->host_write(*object, offset, size, src.get()) == size);
        } else {
          REQUIRE(ctx->host_write_async(*object, offset, size, src.get()).get(30s) == size);
        }
        shadow.write(offset, bytes);
      }
    }
    CHECK(object->size() == shadow.bytes().size());
    check_same(read_back(ctx.get(), path.str()), shadow.bytes());
    ctx->flush(*object);
    REQUIRE_NOTHROW(ctx->flush_async(*object).get(30s));
  }
}

TEST_CASE("cufile host writes from 8 B-aligned blocks land in place", "[io][cufile][write]")
{
  // The DISK tier's HOST -> DISK layout: 1 MiB pinned-pool blocks holding 8 B
  // aligned column buffers, written as one vectored request whose segments start
  // at file offsets that are not congruent with their source addresses (the
  // granule-aligned bodies are then staged through the scratch and written
  // O_DIRECT), across several columns at 4 KiB-aligned file offsets.
  auto const [label, cfg, force_cufile] = GENERATE(from_range(write_paths()));
  DYNAMIC_SECTION(label)
  {
    std::ignore = force_cufile;
    write_context ctx{cfg};
    scratch_path path;
    auto object = ctx->open_io_object_for_write(path.str());
    shadow_file shadow;

    constexpr std::size_t block = 1 * MiB;
    struct column {
      std::size_t host_offset;  // inside the block allocation, 8 B aligned
      std::size_t size;
      std::size_t file_offset;  // 4 KiB aligned
    };
    std::vector<column> const columns{{8, 3 * MiB + 77, 0},
                                      {3 * MiB + 88, 2 * MiB + 5, 8 * MiB},
                                      {5 * MiB + 96, 100'000, 12 * MiB},
                                      {5 * MiB + 100'104, 1 * MiB, 16 * MiB}};
    std::size_t const allocation = 8 * MiB;
    auto const host_data         = pattern(allocation, 1000);
    host_bytes const blocks{host_data, 0};  // block-aligned base, like the pool

    std::vector<write_slice> slices;
    for (auto const& col : columns) {
      for (std::size_t written = 0; written < col.size;) {
        auto const at    = col.host_offset + written;
        auto const chunk = std::min(col.size - written, block - at % block);
        slices.emplace_back(col.file_offset + written, chunk, blocks.get() + at);
        shadow.write(col.file_offset + written,
                     std::span<std::uint8_t const>(host_data.data() + at, chunk));
        written += chunk;
      }
    }
    std::size_t total = 0;
    for (auto const& sl : slices) {
      total += sl.size();
    }
    REQUIRE(ctx->host_writev_async(*object, slices).get(60s) == total);
    CHECK(object->size() == shadow.bytes().size());
    check_same(read_back(ctx.get(), path.str()), shadow.bytes());
  }
}

TEST_CASE("cufile device writes round-trip on every path", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const [label, cfg, force_cufile] = GENERATE(from_range(write_paths()));
  DYNAMIC_SECTION(label)
  {
    force_cufile_writes const force{force_cufile};
    write_context ctx{cfg};
    scratch_path path;
    auto object = ctx->open_io_object_for_write(path.str());
    cuda_stream stream;
    shadow_file shadow;

    struct shape {
      std::size_t offset;
      std::size_t size;
      std::size_t misalignment;
    };
    // Aligned, unaligned at both ends, larger than the staging scratch, tiny, and a
    // source pointer that is not congruent with its file offset.
    std::size_t seed = 100;
    for (auto const& [offset, size, misalignment] : {shape{4096, 8 * MiB, 0},
                                                     shape{123, 3 * MiB + 7, 0},
                                                     shape{12 * MiB, 20 * MiB + 5, 0},
                                                     shape{40 * MiB + 17, 100, 0},
                                                     shape{48 * MiB, 2 * MiB, 256}}) {
      auto const bytes = pattern(size, ++seed);
      device_bytes const src{bytes, misalignment};
      REQUIRE(ctx->device_write_async(*object, offset, size, src.get(), stream.ref()).get(60s) ==
              size);
      shadow.write(offset, bytes);
    }
    CHECK(object->size() == shadow.bytes().size());
    check_same(read_back(ctx.get(), path.str()), shadow.bytes());
  }
}

TEST_CASE("cufile staged device writes wait for their copies", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  config cfg{};
  cfg.gds_threshold = 1UL << 40;  // everything staged
  write_context ctx{cfg, 1};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  cuda_stream stream;
  // Slow every staging copy down: a chunk written before its D2H landed, or a
  // slot refilled while its chunk is still being copied, shows up as wrong bytes.
  cucascade::io::cufile::detail::set_staging_copy_delay_for_testing(std::chrono::milliseconds{50});
  auto const bytes = pattern(20 * MiB + 123, 950);
  device_bytes const src{bytes};
  auto const written =
    ctx->device_write_async(*object, 4096, bytes.size(), src.get(), stream.ref()).get(60s);
  cucascade::io::cufile::detail::set_staging_copy_delay_for_testing(std::chrono::microseconds{0});
  REQUIRE(written == bytes.size());
  shadow_file shadow;
  shadow.write(4096, bytes);
  check_same(read_back(ctx.get(), path.str()), shadow.bytes());
}

TEST_CASE("cufile device write observes the work queued on its stream", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const [label, cfg, force_cufile] = GENERATE(from_range(write_paths()));
  DYNAMIC_SECTION(label)
  {
    force_cufile_writes const force{force_cufile};
    write_context ctx{cfg};
    scratch_path path;
    auto object             = ctx->open_io_object_for_write(path.str());
    constexpr std::size_t n = 4 * MiB + 4096;
    device_bytes const src{pattern(n, 7)};
    cuda_stream stream;

    // Hold the stream on the host, then queue a memset of the source behind it: a
    // write that read the source early would store the pattern instead.
    REQUIRE(cudaLaunchHostFunc(
              stream.get(),
              [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{200}); },
              nullptr) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(src.get(), 0x5A, n, stream.get()) == cudaSuccess);
    REQUIRE(ctx->device_write_async(*object, 4096, n, src.get(), stream.ref()).get(60s) == n);

    auto const back = read_back(ctx.get(), path.str());
    REQUIRE(back.size() == 4096 + n);
    CHECK(std::all_of(back.begin() + 4096, back.end(), [](std::uint8_t b) { return b == 0x5A; }));
  }
}

TEST_CASE("cufile device writes order after the caller's default streams",
          "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  // cudaStreamPerThread names the per-thread stream of whichever thread uses it:
  // the stream gate must be taken on the SUBMITTING thread's.  The legacy
  // default stream is the same from every thread.
  auto const [label, caller_stream] =
    GENERATE(std::pair{"per-thread default stream", cudaStreamPerThread},
             std::pair{"legacy default stream", cudaStreamLegacy},
             std::pair{"null stream", cudaStream_t{nullptr}});
  auto const [path_label, cfg, force_cufile] = GENERATE(from_range(write_paths()));
  CAPTURE(label, path_label);
  force_cufile_writes const force{force_cufile};
  write_context ctx{cfg};
  scratch_path path;
  auto object             = ctx->open_io_object_for_write(path.str());
  constexpr std::size_t n = 4 * MiB + 4096;
  device_bytes const src{pattern(n, 7)};

  REQUIRE(cudaLaunchHostFunc(
            caller_stream,
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{300}); },
            nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(src.get(), 0x5A, n, caller_stream) == cudaSuccess);
  REQUIRE(
    ctx->device_write_async(*object, 0, n, src.get(), ::cuda::stream_ref{caller_stream}).get(60s) ==
    n);

  auto const back = read_back(ctx.get(), path.str());
  REQUIRE(back.size() == n);
  CHECK(std::all_of(back.begin(), back.end(), [](std::uint8_t b) { return b == 0x5A; }));
  REQUIRE(cudaStreamSynchronize(caller_stream) == cudaSuccess);
}

TEST_CASE("cufile vectored writes place every segment", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  config cfg{};
  cfg.gds_threshold     = IO_BLOCK_SIZE;
  cfg.write_split_bytes = 4 * MiB;  // also exercises templated_ioctx's cut
  write_context ctx{cfg};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  cuda_stream stream;
  shadow_file shadow;

  // The disk-tier layout: 4 KiB-aligned starts, unaligned sizes, holes, out of order.
  struct segment {
    std::size_t offset;
    std::size_t size;
  };
  std::vector<segment> const device_layout{
    {64 * 4096, 9 * MiB + 3}, {0, 4097}, {8192, 100'000}, {40 * 4096, 12345}};
  std::vector<std::vector<std::uint8_t>> device_data;
  std::vector<std::unique_ptr<device_bytes>> device_src;
  std::vector<write_slice> device_slices;
  for (std::size_t i = 0; i < device_layout.size(); ++i) {
    device_data.push_back(pattern(device_layout[i].size, 200 + i));
    device_src.push_back(std::make_unique<device_bytes>(device_data.back()));
    device_slices.emplace_back(
      device_layout[i].offset, device_layout[i].size, device_src[i]->get());
  }
  REQUIRE(ctx->device_writev_async(*object, device_slices, stream.ref()).get(60s) ==
          9 * MiB + 3 + 4097 + 100'000 + 12345);
  for (std::size_t i = 0; i < device_layout.size(); ++i) {
    shadow.write(device_layout[i].offset, device_data[i]);
  }

  // Host segments sharing blocks with each other, then a mixed request.
  auto const h0 = pattern(5000, 300);
  auto const h1 = pattern(70'000, 301);
  host_bytes const hs0{h0, 8};
  host_bytes const hs1{h1, 0};
  std::vector<write_slice> host_slices{{20 * MiB + 100, h0.size(), hs0.get()},
                                       {20 * MiB + 5100, h1.size(), hs1.get()}};
  REQUIRE(ctx->host_writev_async(*object, host_slices).get(30s) == h0.size() + h1.size());
  shadow.write(20 * MiB + 100, h0);
  shadow.write(20 * MiB + 5100, h1);

  auto const m0 = pattern(3 * MiB, 400);
  auto const m1 = pattern(6000, 401);
  device_bytes const ms0{m0};
  host_bytes const ms1{m1, 0};
  std::vector<write_slice> mixed_device{{24 * MiB, m0.size(), ms0.get()}};
  std::vector<write_slice> mixed_host{{24 * MiB - 6000, m1.size(), ms1.get()}};
  REQUIRE(ctx->mixed_writev_async(*object, mixed_host, mixed_device, stream.ref()).get(60s) ==
          m0.size() + m1.size());
  shadow.write(24 * MiB, m0);
  shadow.write(24 * MiB - 6000, m1);

  CHECK(object->size() == shadow.bytes().size());
  check_same(read_back(ctx.get(), path.str()), shadow.bytes());
}

TEST_CASE("cufile writes past EOF extend the file and reads see it", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  config cfg{};
  cfg.gds_threshold = IO_BLOCK_SIZE;
  write_context ctx{cfg};
  scratch_path path;
  path.fill(pattern(10'000, 1));
  auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open});
  cuda_stream stream;

  auto const tail = pattern(2 * MiB, 2);
  device_bytes const src{tail};
  constexpr std::size_t offset = 1 * MiB;
  REQUIRE(ctx->device_write_async(*object, offset, tail.size(), src.get(), stream.ref()).get(60s) ==
          tail.size());
  CHECK(object->size() == offset + tail.size());

  // A device read through the same object sees the extended size and the bytes.
  device_bytes const dst{std::vector<std::uint8_t>(tail.size())};
  REQUIRE(
    ctx->device_read_async_io(*object, offset, tail.size(), dst.get(), stream.ref()).get(60s) ==
    tail.size());
  std::vector<std::uint8_t> got(tail.size());
  REQUIRE(cudaMemcpy(got.data(), dst.get(), got.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
  check_same(got, tail);

  auto const back = read_back(ctx.get(), path.str());
  REQUIRE(back.size() == offset + tail.size());
  CHECK(std::all_of(back.begin() + 10'000,
                    back.begin() + static_cast<std::ptrdiff_t>(offset),
                    [](std::uint8_t b) { return b == 0; }));
}

TEST_CASE("cufile write errors reach the caller", "[io][cufile][write]")
{
  write_context ctx;
  scratch_path path;
  path.fill(pattern(4096, 1));
  auto reader = ctx->open_io_object(path.str());
  CHECK_FALSE(reader->writable());
  std::vector<std::uint8_t> src(4096, 1);

  CHECK(error_of(failure_of(ctx->host_write_async(*reader, 0, src.size(), src.data()))) ==
        std::errc::bad_file_descriptor);
  try {
    std::ignore = ctx->host_write(*reader, 0, src.size(), src.data());
    FAIL("a write through a read-only object must throw");
  } catch (std::system_error const& e) {
    CHECK(e.code() == std::errc::bad_file_descriptor);
  }
  CHECK_THROWS_AS(ctx->flush(*reader), std::system_error);
}

TEST_CASE("cufile concurrent writers of disjoint ranges sharing blocks", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  config cfg{};
  cfg.gds_threshold = IO_BLOCK_SIZE;
  write_context ctx{cfg, 4};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());

  // Thread t owns bytes [t * stride + i * 8 * stride ...): ranges of different
  // threads touch the same 4 KiB blocks at their edges.
  constexpr std::size_t threads = 8;
  constexpr std::size_t writes  = 16;
  constexpr std::size_t stride  = 64 * 1024 + 1000;
  shadow_file shadow;
  std::vector<std::vector<std::uint8_t>> data(threads * writes);
  for (std::size_t k = 0; k < data.size(); ++k) {
    data[k] = pattern(stride, 500 + k);
    shadow.write(k * stride, data[k]);
  }
  std::vector<std::thread> pool;
  std::atomic<int> failures{0};
  for (std::size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      try {
        cuda_stream stream;
        for (std::size_t i = 0; i < writes; ++i) {
          auto const k = i * threads + t;
          if (k % 2 == 0) {
            device_bytes const src{data[k]};
            if (ctx->device_write_async(*object, k * stride, stride, src.get(), stream.ref())
                  .get(60s) != stride) {
              ++failures;
            }
          } else if (ctx->host_write_async(*object, k * stride, stride, data[k].data()).get(60s) !=
                     stride) {
            ++failures;
          }
        }
      } catch (...) {
        ++failures;
      }
    });
  }
  for (auto& thread : pool) {
    thread.join();
  }
  CHECK(failures == 0);
  check_same(read_back(ctx.get(), path.str()), shadow.bytes());
}

TEST_CASE("cufile shutdown settles every in-flight write", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scratch_path path;
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  auto const bytes = pattern(16 * MiB, 9);
  device_bytes const src{bytes};
  cuda_stream stream;
  {
    config cfg{};
    cfg.gds_threshold = IO_BLOCK_SIZE;
    write_context ctx{cfg};
    auto object = ctx->open_io_object_for_write(path.str());
    for (std::size_t i = 0; i < 16; ++i) {
      futures.push_back(
        ctx->device_write_async(*object, i * bytes.size(), bytes.size(), src.get(), stream.ref()));
      futures.push_back(
        ctx->host_write_async(*object, (16 + i) * bytes.size(), 4096, bytes.data()));
    }
    ctx->shutdown();
  }
  for (auto& future : futures) {
    auto const error = failure_of(std::move(future));
    if (error != nullptr) CHECK(error_of(error) == std::errc::operation_canceled);
  }
}

TEST_CASE("cufile writes before start are cancelled", "[io][cufile][write]")
{
  write_context ctx{config{}, 1, /*start=*/false};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  std::vector<std::uint8_t> src(4096, 1);
  CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, src.size(), src.data()))) ==
        std::errc::operation_canceled);
  // The synchronous forms run on the caller thread and need no reactor.
  CHECK(ctx->host_write(*object, 0, src.size(), src.data()) == src.size());
  REQUIRE_NOTHROW(ctx->flush(*object));
}

TEST_CASE("cufile registers large aligned transfers when asked", "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  config cfg{};
  cfg.gds_threshold      = IO_BLOCK_SIZE;
  cfg.max_op_bytes       = 1 * MiB;  // several cuFile calls address one registration
  cfg.register_min_bytes = 2 * MiB;
  write_context ctx{cfg};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  cuda_stream stream;
  shadow_file shadow;

  // Registered (aligned, large), not registered (misaligned source; too small).
  struct shape {
    std::size_t offset;
    std::size_t size;
    std::size_t misalignment;
  };
  std::size_t seed = 600;
  for (auto const& [offset, size, misalignment] :
       {shape{0, 8 * MiB, 0}, shape{8 * MiB, 4 * MiB, 512}, shape{12 * MiB, 1 * MiB, 0}}) {
    auto const bytes = pattern(size, ++seed);
    device_bytes const src{bytes, misalignment};
    REQUIRE(ctx->device_write_async(*object, offset, size, src.get(), stream.ref()).get(60s) ==
            size);
    shadow.write(offset, bytes);

    device_bytes const dst{std::vector<std::uint8_t>(size), misalignment};
    REQUIRE(ctx->device_read_async_io(*object, offset, size, dst.get(), stream.ref()).get(60s) ==
            size);
    std::vector<std::uint8_t> got(size);
    REQUIRE(cudaMemcpy(got.data(), dst.get(), size, cudaMemcpyDeviceToHost) == cudaSuccess);
    check_same(got, bytes);

    // Deregistered before the futures resolved: the buffers register again.
    CHECK(cucascade::io::cufile::detail::live_registrations_for_testing() == 0);
    for (void const* base :
         {static_cast<void const*>(src.get()), static_cast<void const*>(dst.get())}) {
      REQUIRE(cucascade::io::cufile::cufile_api::get() != nullptr);
      auto const& api = *cucascade::io::cufile::cufile_api::get();
      REQUIRE(api.buf_register(base, size, 0).err == CU_FILE_SUCCESS);
      REQUIRE(api.buf_deregister(base).err == CU_FILE_SUCCESS);
    }
  }
  check_same(read_back(ctx.get(), path.str()), shadow.bytes());
}

TEST_CASE("cufile shares one registration between transfers of the same buffer",
          "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  config cfg{};
  cfg.gds_threshold      = IO_BLOCK_SIZE;
  cfg.max_op_bytes       = 1 * MiB;
  cfg.register_min_bytes = 1 * MiB;
  write_context ctx{cfg, 4};

  // One aligned source written to several files at once (replication): the
  // transfers share its registration, and none deregisters it under another.
  auto const bytes = pattern(16 * MiB, 800);
  device_bytes const src{bytes};
  constexpr std::size_t copies = 6;
  std::vector<std::unique_ptr<scratch_path>> paths;
  std::vector<std::shared_ptr<cucascade::io::io_object>> objects;
  std::vector<std::unique_ptr<cuda_stream>> streams;
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < copies; ++i) {
    paths.push_back(std::make_unique<scratch_path>());
    objects.push_back(ctx->open_io_object_for_write(paths.back()->str()));
    streams.push_back(std::make_unique<cuda_stream>());
  }
  for (std::size_t i = 0; i < copies; ++i) {
    futures.push_back(
      ctx->device_write_async(*objects[i], 0, bytes.size(), src.get(), streams[i]->ref()));
  }
  // A transfer overlapping the registered buffer at another base is served too.
  futures.push_back(ctx->device_write_async(
    *objects[0], 32 * MiB, 4 * MiB, src.get() + 4 * MiB, streams[0]->ref()));
  for (auto& future : futures) {
    REQUIRE(std::move(future).get(60s) > 0);
  }
  CHECK(cucascade::io::cufile::detail::live_registrations_for_testing() == 0);
  REQUIRE(cucascade::io::cufile::cufile_api::get() != nullptr);
  auto const& api = *cucascade::io::cufile::cufile_api::get();
  REQUIRE(api.buf_register(src.get(), bytes.size(), 0).err == CU_FILE_SUCCESS);
  REQUIRE(api.buf_deregister(src.get()).err == CU_FILE_SUCCESS);
  for (std::size_t i = 0; i < copies; ++i) {
    auto const back = read_back(ctx.get(), paths[i]->str());
    REQUIRE(back.size() >= bytes.size());
    check_same(std::vector<std::uint8_t>(back.begin(),
                                         back.begin() + static_cast<std::ptrdiff_t>(bytes.size())),
               bytes);
  }
}

namespace {

/// Runs a check on the worker during the first cuFile call after it is armed.
struct transfer_probe {
  std::function<void()> check;
  std::atomic<bool> armed{false};

  static void run(void* self) noexcept
  {
    auto& probe = *static_cast<transfer_probe*>(self);
    if (probe.armed.exchange(false)) { probe.check(); }
  }
};

/// Installs a transfer hook for the guard's lifetime.
class transfer_hook {
 public:
  explicit transfer_hook(transfer_probe& probe)
  {
    cucascade::io::cufile::detail::set_cufile_transfer_hook_for_testing(&transfer_probe::run,
                                                                        &probe);
  }
  ~transfer_hook()
  {
    cucascade::io::cufile::detail::set_cufile_transfer_hook_for_testing(nullptr, nullptr);
  }
  transfer_hook(transfer_hook const&)            = delete;
  transfer_hook& operator=(transfer_hook const&) = delete;
};

}  // namespace

TEST_CASE("cufile transfers of a context without registration still block others from registering",
          "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  force_cufile_writes const force;
  using cucascade::io::cufile::detail::registrations_created_for_testing;
  using cucascade::io::cufile::detail::transfer_tracked_for_testing;
  // Context A keeps the default (no registrations); context B opts in.  Both
  // hand every aligned body to cuFile.
  config plain{};
  plain.gds_threshold       = IO_BLOCK_SIZE;
  config opt_in             = plain;
  opt_in.register_min_bytes = 1 * MiB;
  write_context a{plain, 1};
  write_context b{opt_in, 1};
  scratch_path path_a;
  scratch_path path_b;
  auto object_a = a->open_io_object_for_write(path_a.str());
  auto object_b = b->open_io_object_for_write(path_b.str());
  if (!object_a->writable() ||
      !dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object_a)
         .cufile_registered()) {
    SKIP("cuFile cannot register files here");
  }
  constexpr std::size_t n = 8 * MiB;
  auto const bytes        = pattern(n, 900);
  device_bytes const src{bytes};
  cuda_stream stream;

  transfer_probe probe;
  transfer_hook const hook{probe};

  SECTION("a write of A is tracked, and B does not register its buffer meanwhile")
  {
    bool tracked                 = false;
    std::size_t created_during_b = 0;
    probe.check                  = [&] {
      tracked            = transfer_tracked_for_testing(src.get(), n);
      auto const before  = registrations_created_for_testing();
      auto const written = b->device_write_async(*object_b, 0, n, src.get(), stream.ref()).get(60s);
      created_during_b   = registrations_created_for_testing() - before;
      CHECK(written == n);
    };
    probe.armed = true;
    REQUIRE(a->device_write_async(*object_a, 0, n, src.get(), stream.ref()).get(60s) == n);
    CHECK_FALSE(probe.armed.load());
    CHECK(tracked);
    CHECK(created_during_b == 0);
    // Once A's transfer is over, B registers the buffer again.
    CHECK_FALSE(transfer_tracked_for_testing(src.get(), n));
    auto const before = registrations_created_for_testing();
    REQUIRE(b->device_write_async(*object_b, 0, n, src.get(), stream.ref()).get(60s) == n);
    CHECK(registrations_created_for_testing() == before + 1);
    CHECK(cucascade::io::cufile::detail::live_registrations_for_testing() == 0);
    check_same(read_back(a.get(), path_a.str()), bytes);
    check_same(read_back(b.get(), path_b.str()), bytes);
  }

  SECTION("a read of A is tracked too")
  {
    REQUIRE(a->device_write_async(*object_a, 0, n, src.get(), stream.ref()).get(60s) == n);
    auto reader = a->open_io_object(path_a.str());
    device_bytes const dst{std::vector<std::uint8_t>(n)};
    bool tracked = false;
    probe.check  = [&] { tracked = transfer_tracked_for_testing(dst.get(), n); };
    probe.armed  = true;
    REQUIRE(a->device_read_async_io(*reader, 0, n, dst.get(), stream.ref()).get(60s) == n);
    CHECK_FALSE(probe.armed.load());
    CHECK(tracked);
    CHECK_FALSE(transfer_tracked_for_testing(dst.get(), n));
  }
}

TEST_CASE("cufile host writes are not held behind a device write waiting for its stream",
          "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  write_context ctx{config{}, 1};  // one reactor: both requests share its worker
  scratch_path device_path;
  scratch_path host_path;
  auto device_object = ctx->open_io_object_for_write(device_path.str());
  auto host_object   = ctx->open_io_object_for_write(host_path.str());
  auto const bytes   = pattern(4 * MiB, 900);
  device_bytes const src{bytes};
  cuda_stream stream;
  REQUIRE(cudaLaunchHostFunc(
            stream.get(),
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{2000}); },
            nullptr) == cudaSuccess);
  auto waiting = ctx->device_write_async(*device_object, 0, bytes.size(), src.get(), stream.ref());
  auto const small = pattern(4096, 901);
  REQUIRE(ctx->host_write_async(*host_object, 0, small.size(), small.data()).get(1500ms) ==
          small.size());
  CHECK_FALSE(waiting.is_ready());
  REQUIRE(std::move(waiting).get(30s) == bytes.size());
  check_same(read_back(ctx.get(), device_path.str()), bytes);
}

TEST_CASE("cufile stages aligned device writes on a compat-only platform",
          "[io][cufile][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  config cfg{};
  cfg.gds_threshold = IO_BLOCK_SIZE;
  write_context ctx{cfg};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  if (!dynamic_cast<cufile_io_object const&>(*object).cufile_registered()) {
    SKIP("the file did not register with cuFile here");
  }
  cuda_stream stream;
  auto const bytes = pattern(4 * MiB, 700);
  device_bytes const src{bytes};

  auto calls_for_one_write = [&](std::size_t offset) {
    auto const before = cucascade::io::cufile::detail::cufile_write_calls_for_testing();
    REQUIRE(
      ctx->device_write_async(*object, offset, bytes.size(), src.get(), stream.ref()).get(60s) ==
      bytes.size());
    return cucascade::io::cufile::detail::cufile_write_calls_for_testing() - before;
  };

  bool const compat_only = cucascade::io::cufile::probe_platform().compat_only;
  CAPTURE(compat_only);
  // The default policy: cuFileWrite only with a real GDS mode.
  CHECK((calls_for_one_write(0) != 0) == !compat_only);
  {
    force_cufile_writes const force;
    CHECK(calls_for_one_write(4 * MiB) != 0);
  }
  shadow_file shadow;
  shadow.write(0, bytes);
  shadow.write(4 * MiB, bytes);
  check_same(read_back(ctx.get(), path.str()), shadow.bytes());
}

// ---------------------------------------------------------------------------
// Write throughput probe (hidden; run explicitly with "[.cufile-write-bench]")
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] std::size_t env_size(char const* name, std::size_t fallback)
{
  char const* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? std::stoull(value) : fallback;
}

/// Run the write scenarios against @p ctx, each into a fresh file under @p dir,
/// timing the writes plus the final flush; print one line per scenario.
void run_write_probe(std::string const& label,
                     cucascade::io::ioctx& ctx,
                     std::string const& dir,
                     std::size_t total,
                     int rounds)
{
  constexpr std::size_t window   = 1UL << 30;
  constexpr std::size_t n_stream = 8;
  auto const bytes               = std::max(window, total / window * window);
  device_bytes const src{pattern(window, 1)};
  host_bytes const host_src{pattern(window, 2), 0};
  std::vector<std::unique_ptr<cuda_stream>> streams;
  for (std::size_t i = 0; i < n_stream; ++i) {
    streams.push_back(std::make_unique<cuda_stream>());
  }
  auto const path = dir + "/cucascade_write_probe_" + std::to_string(::getpid()) + ".bin";

  // CUCASCADE_IO_BENCH_WMODE: "fresh" (default) truncates the file each round,
  // "hint" also preallocates it (size_hint), "overwrite" rewrites an existing one.
  char const* mode_env   = std::getenv("CUCASCADE_IO_BENCH_WMODE");
  std::string const mode = mode_env != nullptr ? mode_env : "fresh";
  write_open_options options{};
  if (mode == "hint") options.size_hint = bytes;
  if (mode == "overwrite") {
    options.mode = write_mode::create_or_open;
    auto object  = ctx.open_io_object_for_write(path);
    for (std::size_t off = 0; off < bytes; off += window) {
      REQUIRE(ctx.host_write(*object, off, window, host_src.get()) == window);
    }
    ctx.flush(*object);
  }

  auto scenario = [&](std::string const& name, auto&& run) {
    std::vector<double> rates;
    for (int r = 0; r < rounds; ++r) {
      auto object      = ctx.open_io_object_for_write(path, options);
      auto const start = std::chrono::steady_clock::now();
      run(*object);
      ctx.flush(*object);
      auto const secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      rates.push_back(static_cast<double>(bytes) / secs / 1e9);
    }
    std::sort(rates.begin(), rates.end());
    std::printf("%-14s %-9s %-30s %7.2f GB/s (incl. fdatasync)\n",
                label.c_str(),
                mode.c_str(),
                name.c_str(),
                rates[rates.size() / 2]);
    std::fflush(stdout);
  };

  for (std::size_t const piece : {256UL << 20, 1UL << 30}) {
    scenario("dev 1 slice x " + std::to_string(piece >> 20) + " MiB, QD1", [&](auto& object) {
      for (std::size_t off = 0; off < bytes; off += piece) {
        REQUIRE(
          ctx.device_write_async(object, off, piece, src.get(), streams[0]->ref()).get(300s) ==
          piece);
      }
    });
  }
  scenario("dev writev 4 MiB, 1 stream", [&](auto& object) {
    constexpr std::size_t piece = 4UL << 20;
    for (std::size_t off = 0; off < bytes; off += window) {
      std::vector<write_slice> slices;
      for (std::size_t p = 0; p < window; p += piece) {
        slices.emplace_back(off + p, piece, src.get() + p);
      }
      REQUIRE(ctx.device_writev_async(object, slices, streams[0]->ref()).get(300s) == window);
    }
  });
  scenario("dev 8 MiB x 8 streams", [&](auto& object) {
    constexpr std::size_t piece = 8UL << 20;
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    std::size_t i = 0;
    for (std::size_t off = 0; off < bytes; off += piece, ++i) {
      futures.push_back(ctx.device_write_async(
        object, off, piece, src.get() + off % window, streams[i % n_stream]->ref()));
    }
    for (auto& future : futures) {
      REQUIRE(std::move(future).get(300s) == piece);
    }
  });
  scenario("host writev 1 MiB, 8 B aligned", [&](auto& object) {
    // The DISK tier's HOST -> DISK shape: 1 MiB pool blocks whose bytes start 8 B
    // past a block boundary, so file offsets and sources are never congruent.
    constexpr std::size_t piece = 1UL << 20;
    for (std::size_t off = 0; off < bytes; off += window) {
      // Block 0 holds the first piece - 8 bytes at +8; every later block starts a
      // segment at a file offset 8 bytes short of a 4 KiB boundary.
      std::vector<write_slice> slices;
      slices.emplace_back(off, piece - 8, host_src.get() + 8);
      for (std::size_t p = piece; p < window; p += piece) {
        slices.emplace_back(off + p - 8, piece, host_src.get() + p);
      }
      slices.emplace_back(off + window - 8, 8, host_src.get());
      REQUIRE(ctx.host_writev_async(object, slices).get(300s) == window);
    }
  });
  scenario("host 4 MiB, all queued", [&](auto& object) {
    constexpr std::size_t piece = 4UL << 20;
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    for (std::size_t off = 0; off < bytes; off += piece) {
      futures.push_back(ctx.host_write_async(object, off, piece, host_src.get() + off % window));
    }
    for (auto& future : futures) {
      REQUIRE(std::move(future).get(300s) == piece);
    }
  });
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

}  // namespace

TEST_CASE("cufile write throughput probe", "[.cufile-write-bench]")
{
  char const* dir_env = std::getenv("CUCASCADE_IO_BENCH_WDIR");
  if (dir_env == nullptr || !has_cuda_device()) {
    SKIP("set CUCASCADE_IO_BENCH_WDIR (a directory on the device under test) to run the probe");
  }
  std::string const dir   = dir_env;
  auto const total        = env_size("CUCASCADE_IO_BENCH_BYTES", 4UL << 30);
  auto const rounds       = static_cast<int>(env_size("CUCASCADE_IO_BENCH_ROUNDS", 3));
  auto const reactors     = env_size("CUCASCADE_IO_BENCH_REACTORS", 4);
  char const* backend     = std::getenv("CUCASCADE_IO_BENCH_BACKEND");
  std::string const which = backend != nullptr ? backend : "all";

  if (which == "all" || which == "cufile") {
    write_context ctx{config{}, reactors};
    run_write_probe("cufile", ctx.get(), dir, total, rounds);
  }
  if (which == "all" || which == "cufile-split") {
    config cfg{};
    cfg.write_split_bytes = env_size("CUCASCADE_IO_BENCH_SPLIT", 64UL << 20);
    write_context ctx{cfg, reactors};
    run_write_probe("cufile-split", ctx.get(), dir, total, rounds);
  }
  if (which == "all" || which == "cufile-buffered") {
    config cfg{};
    cfg.use_odirect = false;
    cfg.compat_mode = cucascade::io::cufile::compat::on;
    write_context ctx{cfg, reactors};
    run_write_probe("cufile-buffer", ctx.get(), dir, total, rounds);
  }
  if (which == "all" || which == "cufile-staged") {
    config cfg{};
    cfg.gds_threshold = 1UL << 40;
    write_context ctx{cfg, reactors};
    run_write_probe("cufile-staged", ctx.get(), dir, total, rounds);
  }
  if (which == "all" || which == "uring") {
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, 2UL << 30, 2UL << 30, 1UL << 20, 128, 8};
    cucascade::io::uring::uring_ioctx ctx{
      reactors,
      std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
        cucascade::io::uring::config{}, &mr)};
    ctx.start();
    run_write_probe("uring", ctx, dir, total, rounds);
    ctx.shutdown();
  }
}
