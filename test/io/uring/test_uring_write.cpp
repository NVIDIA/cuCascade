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

#include "io/detail/local_descriptor.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_errors.hpp>
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
#include <fcntl.h>
#include <linux/filter.h>
#include <linux/magic.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using cucascade::io::IO_BLOCK_SIZE;
using cucascade::io::io_object;
using cucascade::io::io_priority;
using cucascade::io::range;
using cucascade::io::write_mode;
using cucascade::io::write_open_options;
using cucascade::io::write_slice;
using cucascade::io::uring::local_io_object;
using cucascade::io::uring::uring_ioctx;
using cucascade::io::uring::uring_reactor;
using cucascade::io::uring::detail::split_write_range;

namespace {

constexpr std::size_t MiB = 1UL << 20;

/** The byte a pattern of @p seed holds at @p index. */
[[nodiscard]] std::uint8_t pattern_byte(std::size_t index, std::size_t seed) noexcept
{
  return static_cast<std::uint8_t>(((index + seed * 7919U) * 2654435761ULL) >> 13);
}

[[nodiscard]] std::vector<std::uint8_t> pattern(std::size_t bytes, std::size_t seed)
{
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = pattern_byte(i, seed);
  }
  return out;
}

/** A unique scratch path under the temp directory, removed on scope exit. */
class scratch_path {
 public:
  scratch_path()
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_uring_write_" + std::to_string(::getpid()) + "_" +
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
    : _storage(static_cast<std::uint8_t*>(
        std::aligned_alloc(IO_BLOCK_SIZE, round_up(contents.size() + misalignment)))),
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
  static std::size_t round_up(std::size_t bytes)
  {
    return std::max(IO_BLOCK_SIZE, (bytes + IO_BLOCK_SIZE - 1) / IO_BLOCK_SIZE * IO_BLOCK_SIZE);
  }

  std::uint8_t* _storage;
  std::uint8_t* _data;
};

/** Device memory holding a copy of @p contents. */
class device_bytes {
 public:
  explicit device_bytes(std::vector<std::uint8_t> const& contents) : _size(contents.size())
  {
    REQUIRE(cudaMalloc(&_data, std::max<std::size_t>(1, _size)) == cudaSuccess);
    if (_size != 0) {
      REQUIRE(cudaMemcpy(_data, contents.data(), _size, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    // A pageable cudaMemcpy may return before its DMA lands, and the writes
    // read the buffer on non-blocking streams that do not wait for it.
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }

  ~device_bytes() { std::ignore = cudaFree(_data); }

  device_bytes(device_bytes const&)            = delete;
  device_bytes& operator=(device_bytes const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return static_cast<std::uint8_t*>(_data); }

 private:
  void* _data{nullptr};
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

/** A started uring ioctx over 1 MiB pinned staging blocks. */
class write_context {
 public:
  static constexpr std::size_t block_size = MiB;
  static constexpr std::size_t capacity   = 512 * MiB;

  explicit write_context(std::size_t n_reactors           = 2,
                         cucascade::io::uring::config cfg = cucascade::io::uring::config{},
                         bool start                       = true)
    : _mr{0, _upstream, capacity, capacity, block_size, 64, 0},
      _ctx{std::make_shared<uring_ioctx>(
        n_reactors, std::make_shared<uring_reactor::reactor_context>(cfg, &_mr))}
  {
    if (start) _ctx->start();
  }

  ~write_context() { _ctx->shutdown(); }

  write_context(write_context const&)            = delete;
  write_context& operator=(write_context const&) = delete;

  [[nodiscard]] uring_ioctx& get() noexcept { return *_ctx; }
  uring_ioctx* operator->() noexcept { return _ctx.get(); }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
  std::shared_ptr<uring_ioctx> _ctx;
};

/** Read the whole object through a NEW read-only io_object of @p ctx. */
[[nodiscard]] std::vector<std::uint8_t> read_back(uring_ioctx& ctx, std::string const& path)
{
  auto object = ctx.open_io_object(path);
  std::vector<std::uint8_t> out(object->size());
  if (!out.empty()) {
    REQUIRE(ctx.host_read_async_io(*object, 0, out.size(), out.data()).get() == out.size());
  }
  return out;
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
    std::ignore = std::forward<Future>(future).get();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

[[nodiscard]] std::exception_ptr flush_failure_of(cucascade::exec::semi_future<void>&& future)
{
  try {
    std::move(future).get();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

[[nodiscard]] bool is_invalid_argument(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::invalid_argument const&) {
    return true;
  } catch (...) {
  }
  return false;
}

/** Holds a CUDA stream at a host function until released (or for 10 s at most). */
class stream_gate {
 public:
  stream_gate()                              = default;
  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;
  ~stream_gate() { release(); }

  void block(cudaStream_t stream)
  {
    REQUIRE(cudaLaunchHostFunc(stream, &stream_gate::wait, this) == cudaSuccess);
  }

  void release() noexcept { _open.store(true, std::memory_order_release); }

 private:
  static void CUDART_CB wait(void* self)
  {
    auto const* gate    = static_cast<stream_gate const*>(self);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gate->_open.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  std::atomic<bool> _open{false};
};

/**
 * Make io_uring_setup fail with EPERM for this thread and the threads it
 * creates from now on, as a container's default seccomp profile does.
 */
void block_io_uring_setup()
{
  auto statement = [](std::uint32_t code, std::uint32_t k) {
    return sock_filter{static_cast<std::uint16_t>(code), 0, 0, k};
  };
  std::array<sock_filter, 4> filter{
    statement(BPF_LD | BPF_W | BPF_ABS, static_cast<std::uint32_t>(offsetof(seccomp_data, nr))),
    sock_filter{static_cast<std::uint16_t>(BPF_JMP | BPF_JEQ | BPF_K),
                0,
                1,
                static_cast<std::uint32_t>(__NR_io_uring_setup)},
    statement(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
    statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)};
  sock_fprog program{static_cast<unsigned short>(filter.size()), filter.data()};
  REQUIRE(::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0);
  REQUIRE(::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0);
}

/** An io_object of no backend at all. */
class foreign_object final : public io_object {
 public:
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 0; }
  [[nodiscard]] bool writable() const noexcept override { return true; }

 private:
  std::string _path{"foreign"};
};

}  // namespace

TEST_CASE("uring write splits a range into head, body and tail", "[io][uring][write]")
{
  for (std::size_t const granule : {std::size_t{4096}, std::size_t{65536}}) {
    auto check = [granule](range input) {
      auto const pieces  = split_write_range(input, granule);
      std::size_t cursor = input.offset;
      for (auto const& piece : pieces) {
        if (piece.empty()) continue;
        CHECK(piece.offset == cursor);
        cursor += piece.size;
      }
      CHECK(cursor == input.end());
      if (!pieces[1].empty()) {
        CHECK(pieces[1].offset % granule == 0);
        CHECK(pieces[1].size % granule == 0);
        CHECK(pieces[0].size < granule);
        CHECK(pieces[2].size < granule);
      }
      return pieces;
    };

    auto const inside = check(range{100, 50});
    CHECK(inside[0].size == 50);
    CHECK(inside[1].empty());
    CHECK(inside[2].empty());

    auto const aligned = check(range{granule, 3 * granule});
    CHECK(aligned[0].empty());
    CHECK(aligned[1].size == 3 * granule);
    CHECK(aligned[2].empty());

    auto const ragged = check(range{7, 5 * granule});
    CHECK(ragged[0].size == granule - 7);
    CHECK(ragged[1].offset == granule);
    CHECK(ragged[1].size == 4 * granule);
    CHECK(ragged[2].size == 7);

    std::ignore = check(range{granule - 1, 2});
    std::ignore = check(range{0, granule + 1});
    std::ignore = check(range{3, granule - 3});
  }
}

TEST_CASE("uring open_io_object_for_write honours the open modes", "[io][uring][write]")
{
  write_context ctx;
  scratch_path path;
  auto const old_bytes = pattern(10000, 1);

  CHECK(ctx->supports_write());
  CHECK(ctx->supports_device_write());

  SECTION("create_or_truncate empties an existing file")
  {
    path.fill(old_bytes);
    auto object = ctx->open_io_object_for_write(path.str());
    CHECK(object->writable());
    CHECK(object->size() == 0);
    CHECK(path.contents().empty());
  }

  SECTION("create_or_open keeps the content")
  {
    path.fill(old_bytes);
    auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open});
    CHECK(object->size() == old_bytes.size());
    CHECK(path.contents() == old_bytes);
  }

  SECTION("open_existing keeps the content and fails on a missing file")
  {
    try {
      std::ignore = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
      FAIL("open_existing created a file");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::no_such_file_or_directory);
    }
    path.fill(old_bytes);
    auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
    CHECK(object->size() == old_bytes.size());
  }

  SECTION("create_new refuses an existing file")
  {
    {
      auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_new});
      CHECK(object->size() == 0);
    }
    try {
      std::ignore = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_new});
      FAIL("create_new opened an existing file");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::file_exists);
    }
  }

  SECTION("a file: scheme is stripped and a size hint does not change the size")
  {
    auto object = ctx->open_io_object_for_write("file:" + path.str(), {.size_hint = 8 * MiB});
    CHECK(object->object_path() == path.str());
    CHECK(object->size() == 0);
    CHECK(std::filesystem::file_size(path.str()) == 0);
  }

  SECTION("a directory cannot be opened for writing")
  {
    CHECK_THROWS_AS(ctx->open_io_object_for_write(std::filesystem::temp_directory_path().string()),
                    std::system_error);
  }

  SECTION("objects opened for reading are read-only")
  {
    path.fill(old_bytes);
    CHECK_FALSE(ctx->open_io_object(path.str())->writable());
  }
}

namespace {

/// Renames @c from onto @c to the first time the reopen hook runs: the moment a
/// write open has created / truncated its buffered fd and is about to open the
/// O_DIRECT one.
struct rename_on_reopen {
  std::string from;
  std::string to;
  int calls{0};

  static void run(void* self) noexcept
  {
    auto& state = *static_cast<rename_on_reopen*>(self);
    if (state.calls++ == 0) { std::ignore = ::rename(state.from.c_str(), state.to.c_str()); }
  }
};

/// Installs a reopen hook for the guard's lifetime.
class reopen_hook {
 public:
  reopen_hook(rename_on_reopen& state, bool bypass_procfs)
  {
    cucascade::io::detail::set_reopen_hook_for_testing(
      {&rename_on_reopen::run, &state, bypass_procfs});
  }
  ~reopen_hook() { cucascade::io::detail::set_reopen_hook_for_testing({}); }
  reopen_hook(reopen_hook const&)            = delete;
  reopen_hook& operator=(reopen_hook const&) = delete;
};

/// Whether @p directory's file system accepts O_DIRECT opens.
[[nodiscard]] bool odirect_supported(std::string const& probe)
{
  int const fd = ::open(probe.c_str(), O_RDWR | O_CREAT | O_DIRECT | O_CLOEXEC, 0644);
  if (fd < 0) return false;
  ::close(fd);
  std::error_code ignored;
  std::filesystem::remove(probe, ignored);
  return true;
}

}  // namespace

TEST_CASE("uring write opens anchor the O_DIRECT fd to the opened file", "[io][uring][write]")
{
  // A concurrent rename of the path between the (destructive, never retried)
  // create / truncate and the O_DIRECT open must not pair the two descriptors
  // with different files: aligned bytes would land in the other one.
  scratch_path path;
  scratch_path replacement;
  scratch_path probe;
  if (!odirect_supported(probe.str())) { SKIP("the temp file system has no O_DIRECT"); }
  write_context ctx;
  auto const original_bytes    = pattern(2 * IO_BLOCK_SIZE, 3);
  auto const replacement_bytes = pattern(2 * IO_BLOCK_SIZE, 4);
  path.fill(original_bytes);
  replacement.fill(replacement_bytes);
  // Keeps the original file reachable once the rename replaced it at `path`.
  cucascade::io::file_descriptor original{::open(path.str().c_str(), O_RDONLY | O_CLOEXEC)};
  REQUIRE(original);
  rename_on_reopen state{replacement.str(), path.str()};

  SECTION("through /proc/self/fd the rename cannot redirect the O_DIRECT fd")
  {
    std::shared_ptr<io_object> object;
    {
      reopen_hook const hook{state, /*bypass_procfs=*/false};
      object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open});
    }
    REQUIRE(state.calls == 1);
    auto const* local = dynamic_cast<local_io_object const*>(object.get());
    REQUIRE(local != nullptr);
    REQUIRE(local->odirect_handle() >= 0);

    // An aligned write from an aligned buffer takes the O_DIRECT fd.
    auto const update = pattern(IO_BLOCK_SIZE, 5);
    host_bytes const source{update, 0};
    CHECK(ctx->host_write(*object, 0, update.size(), source.get()) == update.size());
    std::vector<std::uint8_t> written(update.size());
    REQUIRE(::pread(original.get(), written.data(), written.size(), 0) ==
            static_cast<ssize_t>(written.size()));
    CHECK(written == update);
    // The file now at the path (the replacement) is untouched.
    CHECK(path.contents() == replacement_bytes);
  }

  SECTION("a pathname fallback that sees another file fails closed")
  {
    {
      reopen_hook const hook{state, /*bypass_procfs=*/true};
      CHECK_THROWS_AS(
        ctx->open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open}),
        cucascade::io::descriptor_mismatch_error);
    }
    // The open was not retried (it could repeat a destructive create / truncate),
    // and neither file was written.
    CHECK(state.calls == 1);
    CHECK(path.contents() == replacement_bytes);
    std::vector<std::uint8_t> kept(original_bytes.size());
    REQUIRE(::pread(original.get(), kept.data(), kept.size(), 0) ==
            static_cast<ssize_t>(kept.size()));
    CHECK(kept == original_bytes);
  }
}

TEST_CASE("uring host_write round-trips aligned and unaligned ranges", "[io][uring][write]")
{
  auto const use_odirect = GENERATE(true, false);
  cucascade::io::uring::config cfg{};
  cfg.use_odirect = use_odirect;
  write_context ctx{1, cfg};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  shadow_file shadow;

  std::size_t seed = 0;
  for (std::size_t const offset :
       {std::size_t{0}, std::size_t{1}, std::size_t{4095}, std::size_t{4096}, std::size_t{4097}}) {
    for (std::size_t const size :
         {std::size_t{1}, std::size_t{4095}, std::size_t{4096}, std::size_t{4097}, 3 * MiB + 7}) {
      for (std::size_t const misalignment : {std::size_t{0}, std::size_t{8}}) {
        auto const bytes = pattern(size, ++seed);
        host_bytes source{bytes, misalignment};
        // Spread the writes so they land on fresh ranges and on earlier ones.
        auto const at = offset + (seed % 5) * 2 * MiB;
        CHECK(ctx->host_write(*object, at, size, source.get()) == size);
        shadow.write(at, bytes);
        CHECK(object->size() == shadow.bytes().size());
      }
    }
  }
  CHECK(path.contents() == shadow.bytes());
  CHECK(read_back(ctx.get(), path.str()) == shadow.bytes());
}

TEST_CASE("uring asynchronous host and device writes round-trip", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  shadow_file shadow;

  std::size_t seed = 100;
  for (std::size_t const offset :
       {std::size_t{0}, std::size_t{3}, std::size_t{4096}, std::size_t{65536 + 17}}) {
    for (std::size_t const size :
         {std::size_t{1}, std::size_t{4096}, std::size_t{12345}, 5 * MiB, 40 * MiB + 4096}) {
      auto const host_data   = pattern(size, ++seed);
      auto const device_data = pattern(size, ++seed);
      host_bytes host{host_data, offset % 2 == 0 ? std::size_t{0} : std::size_t{8}};
      device_bytes device{device_data};
      auto const host_at   = offset + (seed % 3) * 64 * MiB;
      auto const device_at = host_at + size + 8191;

      auto host_future = ctx->host_write_async(*object, host_at, size, host.get());
      auto device_future =
        ctx->device_write_async(*object, device_at, size, device.get(), stream.ref());
      CHECK(std::move(host_future).get() == size);
      CHECK(std::move(device_future).get() == size);
      shadow.write(host_at, host_data);
      shadow.write(device_at, device_data);
      CHECK(object->size() == shadow.bytes().size());
    }
  }
  CHECK(read_back(ctx.get(), path.str()) == shadow.bytes());
  CHECK(path.contents() == shadow.bytes());
}

TEST_CASE("uring device write observes the work queued on its stream", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());

  constexpr std::size_t bytes = 48 * MiB + 5;
  device_bytes device{std::vector<std::uint8_t>(bytes, 0)};
  // Queue the producer, then the write, without synchronizing in between.
  REQUIRE(cudaMemsetAsync(device.get(), 0x11, bytes, stream.get()) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(device.get(), 0xAB, bytes, stream.get()) == cudaSuccess);
  CHECK(ctx->device_write_async(*object, 0, bytes, device.get(), stream.ref()).get() == bytes);

  auto const contents = path.contents();
  REQUIRE(contents.size() == bytes);
  CHECK(std::all_of(contents.begin(), contents.end(), [](std::uint8_t b) { return b == 0xAB; }));
}

TEST_CASE("uring device writes order after the caller's default streams", "[io][uring][write]")
{
  // cudaStreamPerThread names the per-thread stream of whichever thread uses it:
  // the write must be ordered after the SUBMITTING thread's, not the worker's.
  // The legacy default stream is the same from every thread.
  auto const [label, caller_stream] =
    GENERATE(std::pair{"per-thread default stream", cudaStreamPerThread},
             std::pair{"legacy default stream", cudaStreamLegacy},
             std::pair{"null stream", cudaStream_t{nullptr}});
  CAPTURE(label);
  write_context ctx;
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());

  constexpr std::size_t bytes = 4 * MiB + 5;
  device_bytes device{std::vector<std::uint8_t>(bytes, 0x11)};
  // Hold the caller's stream on the host, then queue the producer behind it: a
  // write ordered after any other stream would store 0x11.
  REQUIRE(cudaLaunchHostFunc(
            caller_stream,
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{300}); },
            nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(device.get(), 0xAB, bytes, caller_stream) == cudaSuccess);
  CHECK(ctx->device_write_async(*object, 0, bytes, device.get(), ::cuda::stream_ref{caller_stream})
          .get() == bytes);

  auto const contents = path.contents();
  REQUIRE(contents.size() == bytes);
  CHECK(std::all_of(contents.begin(), contents.end(), [](std::uint8_t b) { return b == 0xAB; }));
  REQUIRE(cudaStreamSynchronize(caller_stream) == cudaSuccess);
}

TEST_CASE("uring vectored writes place every segment", "[io][uring][write]")
{
  auto const split = GENERATE(std::size_t{0}, MiB, 32 * MiB);
  cucascade::io::uring::config cfg{};
  cfg.write_split_bytes = split;
  write_context ctx{2, cfg};
  cuda_stream stream;
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());
  shadow_file shadow;

  SECTION("scattered host segments, out of order, sharing blocks")
  {
    auto const bytes = pattern(3 * MiB, 7);
    host_bytes source{bytes, 8};
    // Adjacent segments meet inside a block; listed out of order; one empty.
    std::vector<write_slice> slices{{2 * MiB + 100, MiB - 100, source.get() + 2 * MiB + 100},
                                    {0, 10, source.get()},
                                    {10, 4086, source.get() + 10},
                                    {4096, 2 * MiB + 100 - 4096, source.get() + 4096},
                                    {5 * MiB, 0, nullptr}};
    CHECK(ctx->host_writev_async(*object, slices).get() == 3 * MiB);
    shadow.write(0, bytes);
  }

  SECTION("the disk-tier layout: aligned starts, ragged sizes, holes, mixed sources")
  {
    // Column buffers start at 4 KiB-aligned offsets; sizes are not page
    // multiples; the gap up to the next start is a hole.  Host sources are only
    // 8-byte aligned (pinned blocks), device sources come from cudaMalloc.
    std::vector<std::size_t> const sizes{32, 5000, 1, 64 * MiB + 3, 4096, 3 * MiB + 1};
    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<std::unique_ptr<host_bytes>> hosts;
    std::vector<std::unique_ptr<device_bytes>> devices;
    std::vector<write_slice> host_slices;
    std::vector<write_slice> device_slices;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      payloads.push_back(pattern(sizes[i], 1000 + i));
      if (i % 2 == 0) {
        hosts.push_back(std::make_unique<host_bytes>(payloads.back(), 8));
        host_slices.emplace_back(offset, sizes[i], hosts.back()->get());
      } else {
        devices.push_back(std::make_unique<device_bytes>(payloads.back()));
        device_slices.emplace_back(offset, sizes[i], devices.back()->get());
      }
      shadow.write(offset, payloads.back());
      offset =
        (offset + sizes[i] + IO_BLOCK_SIZE - 1) / IO_BLOCK_SIZE * IO_BLOCK_SIZE + IO_BLOCK_SIZE;
    }
    std::size_t total = 0;
    for (auto const size : sizes) {
      total += size;
    }
    CHECK(ctx->mixed_writev_async(*object, host_slices, device_slices, stream.ref()).get() ==
          total);
  }

  SECTION("device segments only")
  {
    auto const bytes = pattern(9 * MiB + 11, 9);
    device_bytes source{bytes};
    std::vector<write_slice> slices{{6 * MiB, 3 * MiB + 11, source.get() + 6 * MiB},
                                    {0, 6 * MiB, source.get()}};
    CHECK(ctx->device_writev_async(*object, slices, stream.ref()).get() == bytes.size());
    shadow.write(0, bytes);
  }

  SECTION("only empty segments write nothing")
  {
    std::vector<write_slice> slices{{0, 0, nullptr}, {4096, 0, nullptr}};
    CHECK(ctx->host_writev_async(*object, slices).get() == 0);
    CHECK(ctx->device_writev_async(*object, slices, stream.ref()).get() == 0);
  }

  CHECK(object->size() == shadow.bytes().size());
  CHECK(read_back(ctx.get(), path.str()) == shadow.bytes());
}

TEST_CASE("uring writes past EOF extend the file and leave holes", "[io][uring][write]")
{
  write_context ctx;
  scratch_path path;
  path.fill(pattern(1000, 3));
  auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
  REQUIRE(object->size() == 1000);

  auto const tail = pattern(10, 4);
  CHECK(ctx->host_write_async(*object, MiB + 3, tail.size(), tail.data()).get() == tail.size());
  CHECK(object->size() == MiB + 13);

  // Reads through the writable object clamp to the grown size.
  std::vector<std::uint8_t> through(MiB + 100, 0xFF);
  CHECK(ctx->host_read_async_io(*object, 0, through.size(), through.data()).get() == MiB + 13);
  CHECK(std::equal(tail.begin(), tail.end(), through.begin() + MiB + 3));
  CHECK(std::all_of(
    through.begin() + 1000, through.begin() + MiB + 3, [](std::uint8_t b) { return b == 0; }));

  auto const contents = read_back(ctx.get(), path.str());
  REQUIRE(contents.size() == MiB + 13);
  CHECK(std::all_of(
    contents.begin() + 1000, contents.begin() + MiB + 3, [](std::uint8_t b) { return b == 0; }));
}

TEST_CASE("uring zero-length writes do nothing", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  path.fill(pattern(100, 5));
  auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});

  CHECK(ctx->host_write(*object, 5, 0, nullptr) == 0);
  CHECK(ctx->host_write_async(*object, 5000, 0, nullptr).get() == 0);
  CHECK(ctx->device_write_async(*object, 5000, 0, nullptr, stream.ref()).get() == 0);
  CHECK(object->size() == 100);
  CHECK(path.contents() == pattern(100, 5));

  CHECK_THROWS_AS(ctx->host_write(*object, 0, 1, nullptr), std::invalid_argument);
  CHECK(is_invalid_argument(failure_of(ctx->host_write_async(*object, 0, 1, nullptr))));
  CHECK(
    is_invalid_argument(failure_of(ctx->device_write_async(*object, 0, 1, nullptr, stream.ref()))));
}

TEST_CASE("uring write errors reach the caller", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  auto const bytes = pattern(8192, 6);
  path.fill(bytes);

  SECTION("a read-only object is a bad file descriptor")
  {
    auto object = ctx->open_io_object(path.str());
    try {
      std::ignore = ctx->host_write(*object, 0, 10, bytes.data());
      FAIL("wrote through a read-only object");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::bad_file_descriptor);
    }
    CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, 10, bytes.data()))) ==
          std::errc::bad_file_descriptor);
    device_bytes device{bytes};
    CHECK(error_of(failure_of(ctx->device_write_async(
            *object, 0, 10, device.get(), stream.ref()))) == std::errc::bad_file_descriptor);
    CHECK(error_of(flush_failure_of(ctx->flush_async(*object))) == std::errc::bad_file_descriptor);
    CHECK_THROWS_AS(ctx->flush(*object), std::system_error);
    // The hook rejects it too, when called directly.
    std::vector<cucascade::io::prepared_io_slice> slices;
    slices.push_back(cucascade::io::make_host_write_slice(range{0, 10}, bytes.data()));
    CHECK(error_of(failure_of(ctx->mixed_writev_async_io(*object, std::move(slices)))) ==
          std::errc::bad_file_descriptor);
    CHECK(path.contents() == bytes);
  }

  SECTION("overlapping segments are rejected before any byte is written")
  {
    auto object = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
    auto const other = pattern(8192, 7);
    std::vector<write_slice> slices{{0, 4096, other.data()}, {4000, 100, other.data() + 4000}};
    CHECK(is_invalid_argument(failure_of(ctx->host_writev_async(*object, slices))));
    CHECK(path.contents() == bytes);
  }

  SECTION("an object of another backend is rejected")
  {
    foreign_object object;
    CHECK(is_invalid_argument(failure_of(ctx->host_write_async(object, 0, 10, bytes.data()))));
    CHECK_THROWS_AS(ctx->host_write(object, 0, 10, bytes.data()), std::invalid_argument);
  }

  SECTION("a full device fails with ENOSPC")
  {
    if (!std::filesystem::exists("/dev/full")) { SKIP("no /dev/full"); }
    auto object = ctx->open_io_object_for_write("/dev/full", {.mode = write_mode::open_existing});
    auto const big = pattern(3 * MiB, 8);
    CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, big.size(), big.data()))) ==
          std::errc::no_space_on_device);
    device_bytes device{big};
    CHECK(error_of(failure_of(ctx->device_write_async(
            *object, 0, big.size(), device.get(), stream.ref()))) == std::errc::no_space_on_device);
    try {
      std::ignore = ctx->host_write(*object, 0, 10, big.data());
      FAIL("wrote to /dev/full");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::no_space_on_device);
    }
  }

  SECTION("an invalid descriptor fails with EBADF through the reactor")
  {
    auto const object = std::make_shared<local_io_object>(path.str(),
                                                          cucascade::io::file_descriptor{},
                                                          cucascade::io::file_descriptor{},
                                                          0,
                                                          cucascade::io::io_access::read_write);
    CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, 100, bytes.data()))) ==
          std::errc::bad_file_descriptor);
    CHECK(error_of(flush_failure_of(ctx->flush_async(*object))) == std::errc::bad_file_descriptor);
    CHECK(object->size() == 0);
  }
}

TEST_CASE("uring flush makes completed writes durable", "[io][uring][write]")
{
  write_context ctx;
  scratch_path path;
  auto object      = ctx->open_io_object_for_write(path.str());
  auto const bytes = pattern(MiB + 3, 9);
  CHECK(ctx->host_write_async(*object, 0, bytes.size(), bytes.data()).get() == bytes.size());
  CHECK_NOTHROW(ctx->flush(*object));
  CHECK_NOTHROW(ctx->flush_async(*object).get());
  CHECK_NOTHROW(ctx->flush_async(*object, io_priority::low).get());
  CHECK(path.contents() == bytes);
}

TEST_CASE("uring low-priority writes complete", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  auto object       = ctx->open_io_object_for_write(path.str());
  auto const first  = pattern(5 * MiB, 10);
  auto const second = pattern(5 * MiB, 11);
  device_bytes device{second};
  auto low  = ctx->host_write_async(*object, 0, first.size(), first.data(), io_priority::low);
  auto high = ctx->device_write_async(
    *object, first.size(), second.size(), device.get(), stream.ref(), io_priority::high);
  CHECK(std::move(low).get() == first.size());
  CHECK(std::move(high).get() == second.size());
  auto contents = path.contents();
  REQUIRE(contents.size() == first.size() + second.size());
  CHECK(std::equal(first.begin(), first.end(), contents.begin()));
  CHECK(std::equal(second.begin(), second.end(), contents.begin() + 5 * MiB));
}

TEST_CASE("uring concurrent writers of disjoint ranges sharing blocks", "[io][uring][write]")
{
  write_context ctx{4};
  scratch_path path;
  auto object = ctx->open_io_object_for_write(path.str());

  constexpr std::size_t threads    = 8;
  constexpr std::size_t per_thread = 64;
  constexpr std::size_t segments   = threads * per_thread;
  // Ragged boundaries: most segments share a 4 KiB block with a neighbour.
  std::vector<std::size_t> bounds{0};
  for (std::size_t i = 1; i <= segments; ++i) {
    bounds.push_back(bounds.back() + 3001 + (i % 11) * 977 + (i % 5 == 0 ? 3 * MiB / 2 : 0));
  }
  auto const expected = pattern(bounds.back(), 12);

  // Catch2 assertions are not thread-safe: workers report through exceptions.
  auto cuda_check = [](cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  };
  std::vector<std::exception_ptr> errors(threads);
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      cudaStream_t stream = nullptr;
      void* device        = nullptr;
      try {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cuda_check(cudaMalloc(&device, expected.size()));
        cuda_check(cudaMemcpy(device, expected.data(), expected.size(), cudaMemcpyHostToDevice));
        cuda_check(cudaDeviceSynchronize());
        for (std::size_t k = 0; k < per_thread; ++k) {
          auto const i        = k * threads + t;
          auto const offset   = bounds[i];
          auto const size     = bounds[i + 1] - bounds[i];
          auto const* src     = expected.data() + offset;
          std::size_t written = 0;
          switch (i % 4) {
            case 0: written = ctx->host_write(*object, offset, size, src); break;
            case 1: written = ctx->host_write_async(*object, offset, size, src).get(); break;
            default:
              written = ctx
                          ->device_write_async(*object,
                                               offset,
                                               size,
                                               static_cast<std::uint8_t*>(device) + offset,
                                               ::cuda::stream_ref{stream})
                          .get();
          }
          if (written != size) throw std::runtime_error("short write");
        }
      } catch (...) {
        errors[t] = std::current_exception();
      }
      std::ignore = cudaFree(device);
      if (stream != nullptr) std::ignore = cudaStreamDestroy(stream);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (auto const& error : errors) {
    CHECK(error == nullptr);
  }
  CHECK(object->size() == expected.size());
  CHECK(path.contents() == expected);
}

TEST_CASE("uring shutdown settles every in-flight write", "[io][uring][write]")
{
  scratch_path path;
  constexpr std::size_t count = 24;
  constexpr std::size_t bytes = 16 * MiB;
  auto const data             = pattern(count * bytes, 13);
  device_bytes device{data};
  cuda_stream stream;

  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  std::shared_ptr<io_object> object;
  {
    write_context ctx;
    object = ctx->open_io_object_for_write(path.str());
    for (std::size_t i = 0; i < count; ++i) {
      auto const offset = i * bytes;
      futures.push_back(
        i % 2 == 0
          ? ctx->host_write_async(*object, offset, bytes, data.data() + offset)
          : ctx->device_write_async(*object, offset, bytes, device.get() + offset, stream.ref()));
    }
    ctx->shutdown();
  }

  auto const contents   = path.contents();
  std::size_t succeeded = 0;
  for (std::size_t i = 0; i < count; ++i) {
    auto const error = failure_of(std::move(futures[i]));
    if (error == nullptr) {
      ++succeeded;
      auto const offset = i * bytes;
      REQUIRE(contents.size() >= offset + bytes);
      CHECK(std::equal(data.begin() + static_cast<std::ptrdiff_t>(offset),
                       data.begin() + static_cast<std::ptrdiff_t>(offset + bytes),
                       contents.begin() + static_cast<std::ptrdiff_t>(offset)));
    } else {
      CHECK(error_of(error) == std::errc::operation_canceled);
    }
  }
  INFO("succeeded=" << succeeded);
  SUCCEED();
}

TEST_CASE("uring writes before start are cancelled", "[io][uring][write]")
{
  write_context ctx{1, cucascade::io::uring::config{}, false};
  scratch_path path;
  auto object      = ctx->open_io_object_for_write(path.str());
  auto const bytes = pattern(100, 14);
  CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, bytes.size(), bytes.data()))) ==
        std::errc::operation_canceled);
  // The synchronous path needs no reactor thread.
  CHECK(ctx->host_write(*object, 0, bytes.size(), bytes.data()) == bytes.size());
  CHECK(path.contents() == bytes);
}

TEST_CASE("uring aligned writes bypass the page cache", "[io][uring][write]")
{
  write_context ctx;
  cuda_stream stream;
  scratch_path path;
  auto object       = ctx->open_io_object_for_write(path.str());
  auto const* local = dynamic_cast<local_io_object const*>(object.get());
  REQUIRE(local != nullptr);
  if (local->fd_direct() < 0) { SKIP("the temp filesystem refuses O_DIRECT"); }
  // tmpfs accepts O_DIRECT opens, but its data lives in the page cache.
  struct statfs filesystem{};
  REQUIRE(::fstatfs(local->fd(), &filesystem) == 0);
  if (filesystem.f_type == TMPFS_MAGIC) { SKIP("the temp filesystem is tmpfs (always cached)"); }

  constexpr std::size_t body = 8 * MiB;
  auto const host_data       = pattern(body, 20);
  auto const device_data     = pattern(body + 100, 21);
  host_bytes host{host_data, 0};
  device_bytes device{device_data};
  // Host body straight from an aligned buffer, device body staged; the device
  // write ends 100 bytes into a page, which alone goes through the page cache.
  CHECK(ctx->host_write_async(*object, 0, body, host.get()).get() == body);
  CHECK(ctx->device_write_async(*object, body, body + 100, device.get(), stream.ref()).get() ==
        body + 100);

  auto const file_bytes = object->size();
  REQUIRE(file_bytes == 2 * body + 100);
  auto const page  = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  auto const pages = (file_bytes + page - 1) / page;
  int const fd     = ::open(path.str().c_str(), O_RDONLY);
  REQUIRE(fd >= 0);
  void* mapping = ::mmap(nullptr, file_bytes, PROT_READ, MAP_SHARED, fd, 0);
  REQUIRE(mapping != MAP_FAILED);
  std::vector<unsigned char> resident(pages);
  REQUIRE(::mincore(mapping, file_bytes, resident.data()) == 0);
  std::size_t cached = 0;
  for (auto const flag : resident) {
    cached += static_cast<std::size_t>(flag & 1U);
  }
  ::munmap(mapping, file_bytes);
  ::close(fd);
  CHECK(cached <= 1);
}

TEST_CASE("uring write throughput probe", "[.][io][uring][write_bench]")
{
  // Hidden: writes CUCASCADE_WRITE_BENCH_GIB GiB (default 4) of device and of
  // pinned host memory to CUCASCADE_WRITE_BENCH_DIR (default: the temp dir), as
  // 4 KiB-aligned 256 MiB segments in one vectored call, and prints GB/s.
  auto const* dir_env  = std::getenv("CUCASCADE_WRITE_BENCH_DIR");
  auto const* size_env = std::getenv("CUCASCADE_WRITE_BENCH_GIB");
  std::filesystem::path const dir =
    dir_env != nullptr ? std::filesystem::path{dir_env} : std::filesystem::temp_directory_path();
  std::size_t const gib     = size_env != nullptr ? std::stoul(size_env) : 4;
  std::size_t const total   = gib << 30;
  constexpr std::size_t seg = 256 * MiB;

  write_context ctx{4};
  cuda_stream stream;
  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, seg) == cudaSuccess);
  REQUIRE(cudaMemset(device, 0x5A, seg) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  void* pinned = nullptr;
  REQUIRE(cudaMallocHost(&pinned, seg + 8) == cudaSuccess);
  std::memset(pinned, 0x5B, seg + 8);

  auto run = [&](char const* label, bool on_device, std::size_t misalignment) {
    auto const path =
      (dir / ("cucascade_write_bench_" + std::to_string(::getpid()) + ".bin")).string();
    auto object = ctx->open_io_object_for_write(path);
    std::vector<write_slice> slices;
    for (std::size_t offset = 0; offset < total; offset += seg) {
      auto const* src = on_device ? static_cast<std::uint8_t const*>(device)
                                  : static_cast<std::uint8_t const*>(pinned) + misalignment;
      slices.emplace_back(offset, seg, src);
    }
    auto const start   = std::chrono::steady_clock::now();
    auto const written = on_device ? ctx->device_writev_async(*object, slices, stream.ref()).get()
                                   : ctx->host_writev_async(*object, slices).get();
    auto const seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ctx->flush(*object);
    auto const synced =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(written == total);
    std::printf("%-28s %6.2f GB/s (%.2f GB/s incl. fdatasync) to %s\n",
                label,
                static_cast<double>(total) / seconds / 1e9,
                static_cast<double>(total) / synced / 1e9,
                dir.c_str());
    std::error_code ec;
    std::filesystem::remove(path, ec);
  };
  run("device -> file", true, 0);
  run("pinned host -> file", false, 0);
  run("pinned host+8B -> file", false, 8);

  std::ignore = cudaFreeHost(pinned);
  std::ignore = cudaFree(device);
}

TEST_CASE("uring a request stalled on its caller's stream does not hold up other requests",
          "[io][uring][write]")
{
  // One reactor, so the stalled request and the probes share it.
  write_context ctx{1};
  scratch_path read_path;
  read_path.fill(pattern(MiB, 30));
  scratch_path write_path;
  auto reader = ctx->open_io_object(read_path.str());
  auto writer = ctx->open_io_object_for_write(write_path.str());
  cuda_stream stream;
  auto const device_data = pattern(4 * MiB, 31);
  device_bytes device{device_data};
  stream_gate gate;
  gate.block(stream.get());

  cucascade::exec::semi_future<std::size_t> stalled;
  std::size_t stalled_bytes = 0;
  SECTION("a staged device write")
  {
    stalled_bytes = 4 * MiB;
    stalled       = ctx->device_write_async(*writer, 0, stalled_bytes, device.get(), stream.ref());
  }
  SECTION("a device read's host-to-device copy")
  {
    stalled_bytes = MiB;
    stalled = ctx->device_read_async_io(*reader, 0, stalled_bytes, device.get(), stream.ref());
  }
  // Let the reactor reach the stalled request's copy.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto const start = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> read_back_bytes(4096);
  CHECK(
    ctx->host_read_async_io(*reader, 4096, read_back_bytes.size(), read_back_bytes.data()).get() ==
    read_back_bytes.size());
  auto const other = pattern(8192, 32);
  CHECK(ctx->host_write_async(*writer, 8 * MiB, other.size(), other.data()).get() == other.size());
  auto const elapsed = std::chrono::steady_clock::now() - start;
  CHECK(elapsed < std::chrono::seconds(1));
  CHECK_FALSE(stalled.is_ready());

  gate.release();
  CHECK(std::move(stalled).get() == stalled_bytes);
  CHECK(
    std::equal(read_back_bytes.begin(), read_back_bytes.end(), pattern(MiB, 30).begin() + 4096));
}

TEST_CASE("uring short writes are resubmitted and a file size limit fails cleanly",
          "[io][uring][write]")
{
  // RLIMIT_FSIZE is process-wide, so the body runs in a child process.
  constexpr char const* marker_name = "CUCASCADE_URING_WRITE_FSIZE_CHILD";
  if (auto const* marker = std::getenv(marker_name);
      marker == nullptr || std::string_view{marker} != "1") {
    cucascade::test::child_test_runner child{
      "uring short writes are resubmitted and a file size limit fails cleanly",
      {{marker_name, std::string{"1"}}}};
    bool const completed = child.wait_for(std::chrono::seconds(120));
    if (!completed) child.stop();
    INFO(child.output());
    REQUIRE(completed);
    REQUIRE(WIFEXITED(child.status()));
    REQUIRE(WEXITSTATUS(child.status()) == 0);
    return;
  }

  // Past the limit a write fails with EFBIG instead of raising SIGXFSZ.  A
  // write crossing it is cut short at the limit first: the reactor resubmits the
  // remainder (falling back from O_DIRECT, whose shortened length is no longer
  // aligned) and only then reports EFBIG.
  std::signal(SIGXFSZ, SIG_IGN);
  constexpr std::size_t limit = MiB + 1000;
  rlimit fsize{};
  REQUIRE(::getrlimit(RLIMIT_FSIZE, &fsize) == 0);
  fsize.rlim_cur = limit;
  REQUIRE(::setrlimit(RLIMIT_FSIZE, &fsize) == 0);

  for (bool const use_odirect : {true, false}) {
    INFO("use_odirect=" << use_odirect);
    cucascade::io::uring::config cfg{};
    cfg.use_odirect = use_odirect;
    write_context ctx{1, cfg};
    cuda_stream stream;
    scratch_path path;
    auto object      = ctx->open_io_object_for_write(path.str());
    auto const bytes = pattern(4 * MiB, 33);
    host_bytes host{bytes, 0};
    device_bytes device{bytes};

    CHECK(error_of(failure_of(ctx->host_write_async(*object, 0, bytes.size(), host.get()))) ==
          std::errc::file_too_large);
    CHECK(object->size() == 0);
    try {
      std::ignore = ctx->host_write(*object, 0, bytes.size(), host.get());
      FAIL("wrote past RLIMIT_FSIZE");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::file_too_large);
    }
    CHECK(error_of(failure_of(ctx->device_write_async(
            *object, 0, bytes.size(), device.get(), stream.ref()))) == std::errc::file_too_large);
    CHECK(object->size() <= limit);
    CHECK(std::filesystem::file_size(path.str()) <= limit);

    // The reactor is still usable, and a write below the limit succeeds.
    CHECK(ctx->host_write_async(*object, 0, 1000, host.get()).get() == 1000);
    CHECK(object->size() >= 1000);
  }
}

TEST_CASE("uring the first error of many sibling operations fails the write once",
          "[io][uring][write]")
{
  if (!std::filesystem::exists("/dev/full")) { SKIP("no /dev/full"); }
  cucascade::io::uring::config cfg{};
  cfg.write_split_bytes = 64UL << 10;  // hundreds of slices over both reactors
  write_context ctx{2, cfg};
  cuda_stream stream;
  auto object = ctx->open_io_object_for_write("/dev/full", {.mode = write_mode::open_existing});
  auto const bytes = pattern(32 * MiB, 34);
  device_bytes device{bytes};
  for (int round = 0; round < 5; ++round) {
    CHECK(error_of(failure_of(
            ctx->device_write_async(*object, 0, bytes.size(), device.get(), stream.ref()))) ==
          std::errc::no_space_on_device);
    CHECK(error_of(failure_of(ctx->host_write_async(*object, 3, bytes.size() - 3, bytes.data()))) ==
          std::errc::no_space_on_device);
  }
}

TEST_CASE("uring writes and truncating opens invalidate the path's metadata", "[io][uring][write]")
{
  write_context ctx;
  scratch_path path;
  path.fill(pattern(8192, 40));
  auto& store         = ctx->metadata_store();
  auto const metadata = std::make_shared<cucascade::io::io_object_metadata>();
  auto const before   = ctx->open_io_object(path.str());
  store.register_metadata(*before, metadata);
  REQUIRE(store.has_path(path.str()));

  SECTION("a write")
  {
    auto writer = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
    CHECK(store.has_path(path.str()));
    auto const bytes = pattern(100, 41);
    CHECK(ctx->host_write_async(*writer, 0, bytes.size(), bytes.data()).get() == bytes.size());
    CHECK_FALSE(store.has_path(path.str()));
  }
  SECTION("a truncating open")
  {
    auto writer = ctx->open_io_object_for_write(path.str());
    CHECK_FALSE(store.has_path(path.str()));
  }

  // Metadata parsed through an object opened before the change is refused; an
  // object opened afterwards registers again.
  store.register_metadata(*before, metadata);
  CHECK_FALSE(store.has_path(path.str()));
  auto const after = ctx->open_io_object(path.str());
  store.register_metadata(*after, metadata);
  CHECK(store.has_path(path.str()));
}

TEST_CASE("uring a reactor that cannot set up io_uring fails start and every request with why",
          "[io][uring][write]")
{
  // The seccomp filter is process state, so the body runs in a child process.
  constexpr char const* marker_name = "CUCASCADE_URING_SETUP_FAILURE_CHILD";
  if (auto const* marker = std::getenv(marker_name);
      marker == nullptr || std::string_view{marker} != "1") {
    cucascade::test::child_test_runner child{
      "uring a reactor that cannot set up io_uring fails start and every request with why",
      {{marker_name, std::string{"1"}}}};
    bool const completed = child.wait_for(std::chrono::seconds(120));
    if (!completed) child.stop();
    INFO(child.output());
    REQUIRE(completed);
    REQUIRE(WIFEXITED(child.status()));
    REQUIRE(WEXITSTATUS(child.status()) == 0);
    return;
  }

  block_io_uring_setup();
  write_context ctx{1, cucascade::io::uring::config{}, false};
  scratch_path path;
  path.fill(pattern(4096, 50));
  try {
    ctx->start();
    FAIL("start() succeeded without io_uring");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::operation_not_permitted);
  }

  // Asynchronous requests fail with the setup error, not operation_canceled.
  auto reader = ctx->open_io_object(path.str());
  std::vector<std::uint8_t> out(4096);
  CHECK(error_of(failure_of(ctx->host_read_async_io(*reader, 0, out.size(), out.data()))) ==
        std::errc::operation_not_permitted);
  auto writer      = ctx->open_io_object_for_write(path.str(), {.mode = write_mode::open_existing});
  auto const bytes = pattern(100, 51);
  CHECK(error_of(failure_of(ctx->host_write_async(*writer, 0, bytes.size(), bytes.data()))) ==
        std::errc::operation_not_permitted);
  CHECK(error_of(flush_failure_of(ctx->flush_async(*writer))) ==
        std::errc::operation_not_permitted);

  // The synchronous paths run on the caller and need no ring.
  CHECK(ctx->host_write(*writer, 0, bytes.size(), bytes.data()) == bytes.size());
  CHECK(ctx->host_read_io(*reader, 0, out.size(), out.data()) == out.size());
  CHECK(std::equal(bytes.begin(), bytes.end(), out.begin()));
}
