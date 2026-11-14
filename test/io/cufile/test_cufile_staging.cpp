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

/**
 * Test Tags:
 * [cufile]  - cuFile backend
 * [staging] - staged reads through registered device slots, stream gates,
 *             request splitting and the per-GPU pool plan
 * [gpu]     - needs a CUDA device
 *
 * Most cases run against a scripted cuFile (detail::set_cufile_api_for_testing)
 * that records every call, so they check what the reactor asks cuFile for --
 * aligned, never past the end of the file, into registered slots -- on hosts
 * without a GDS stack too.
 */

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;
using cucascade::io::cufile::staging;

namespace {

using namespace std::chrono_literals;

constexpr std::size_t block = cucascade::io::IO_BLOCK_SIZE;

[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  std::uint64_t z = offset + 0x9E3779B97F4A7C15ULL;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<std::uint8_t>(z ^ (z >> 31));
}

/// A file of @ref pattern_byte bytes next to the test binary's working directory
/// (not /tmp, which may be tmpfs and refuse O_DIRECT), removed on scope exit.
class pattern_file {
 public:
  explicit pattern_file(std::size_t bytes)
    : _path(std::filesystem::current_path() /
            ("cucascade_cufile_staging_" + std::to_string(::getpid()) + "_" +
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

  /// Whether the filesystem holding the file accepts O_DIRECT, which cuFile
  /// registration (scripted or real) needs.
  [[nodiscard]] bool odirect() const
  {
    int const fd = ::open(_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) return false;
    ::close(fd);
    return true;
  }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
  std::size_t _size;
};

/// Skip a test whose file cannot be opened O_DIRECT (the cuFile paths need it).
#define REQUIRE_ODIRECT(file)                                                           \
  do {                                                                                  \
    if (!(file).odirect()) SKIP("the working directory's filesystem refuses O_DIRECT"); \
  } while (false)

[[nodiscard]] int device_count()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return 0;
  }
  return count;
}

/// The bytes at @p device, copied on a fresh stream without synchronizing the
/// read's own stream: a settled read must already be in place.
[[nodiscard]] std::vector<std::uint8_t> bytes_at(void const* device, std::size_t bytes)
{
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  std::vector<std::uint8_t> out(bytes);
  REQUIRE(cudaMemcpyAsync(out.data(), device, bytes, cudaMemcpyDeviceToHost, other) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
  return out;
}

[[nodiscard]] bool matches_file(std::vector<std::uint8_t> const& got, std::size_t offset)
{
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != pattern_byte(offset + i)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// A scripted cuFile.  Plain functions over one process-wide state, since the
// table holds C function pointers.
// ---------------------------------------------------------------------------

struct fake_call {
  std::size_t file_offset{0};
  std::size_t size{0};
  void const* base{nullptr};
  std::size_t buffer_offset{0};
  std::size_t file_size{0};
};

struct fake_handle {
  int fd{-1};  // a buffered descriptor of the registered file
  std::size_t size{0};
};

struct fake_state {
  std::mutex mutex;
  std::vector<fake_call> calls;
  std::vector<void const*> registered;
  /// 1-based index of the read call to fail with EIO (0 = none).
  std::size_t fail_call{0};
  /// Most bytes one read call returns (0 = all asked).
  std::size_t max_return{0};
  bool register_buffers{true};
};

fake_state& fake() noexcept
{
  static fake_state state;
  return state;
}

CUfileError_t fake_ok() noexcept
{
  CUfileError_t status{};
  status.err = CU_FILE_SUCCESS;
  return status;
}

CUfileError_t fake_driver_open() { return fake_ok(); }

CUfileError_t fake_get_properties(CUfileDrvProps_t* props)
{
  *props                         = CUfileDrvProps_t{};
  props->nvfs.max_direct_io_size = 16384;
  return fake_ok();
}

CUfileError_t fake_handle_register(CUfileHandle_t* handle, CUfileDescr_t* descr)
{
  auto* const owned = new fake_handle{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  auto const fd = descr->handle.fd;
  owned->fd     = ::open(("/proc/self/fd/" + std::to_string(fd)).c_str(), O_RDONLY | O_CLOEXEC);
  owned->size   = static_cast<std::size_t>(::lseek(owned->fd, 0, SEEK_END));
  *handle       = owned;
  return fake_ok();
}

void fake_handle_deregister(CUfileHandle_t handle)
{
  auto* const owned = static_cast<fake_handle*>(handle);
  ::close(owned->fd);
  delete owned;
}

ssize_t fake_read(
  CUfileHandle_t handle, void* base, std::size_t size, off_t file_offset, off_t buffer_offset)
{
  auto const* owned = static_cast<fake_handle*>(handle);
  std::size_t index = 0;
  std::size_t limit = size;
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    state.calls.push_back(fake_call{static_cast<std::size_t>(file_offset),
                                    size,
                                    base,
                                    static_cast<std::size_t>(buffer_offset),
                                    owned->size});
    index = state.calls.size();
    if (state.fail_call == index) {
      errno = EIO;
      return -1;
    }
    if (state.max_return != 0) limit = std::min(limit, state.max_return);
  }
  std::vector<std::uint8_t> host(limit);
  auto const got = ::pread(owned->fd, host.data(), limit, file_offset);
  // Like cuFileRead, return only once the bytes are in device memory: a plain
  // cudaMemcpy from pageable memory may return before its DMA has landed.
  if (got > 0 && (cudaMemcpyAsync(static_cast<std::uint8_t*>(base) + buffer_offset,
                                  host.data(),
                                  static_cast<std::size_t>(got),
                                  cudaMemcpyHostToDevice,
                                  cudaStreamPerThread) != cudaSuccess ||
                  cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)) {
    errno = EIO;
    return -1;
  }
  return got;
}

CUfileError_t fake_buf_register(void const* base, std::size_t, int)
{
  auto& state = fake();
  std::lock_guard lock(state.mutex);
  if (!state.register_buffers) {
    CUfileError_t status{};
    status.err = CU_FILE_GPU_MEMORY_PINNING_FAILED;
    return status;
  }
  state.registered.push_back(base);
  return fake_ok();
}

CUfileError_t fake_buf_deregister(void const*) { return fake_ok(); }

/// Installs the scripted cuFile for the guard's lifetime.  Declare it before the
/// ioctx and objects it serves, so it outlives them.
class scripted_cufile {
 public:
  scripted_cufile()
  {
    {
      auto& state = fake();
      std::lock_guard lock(state.mutex);
      state.calls.clear();
      state.fail_call        = 0;
      state.max_return       = 0;
      state.register_buffers = true;
    }
    _api.driver_open           = &fake_driver_open;
    _api.driver_get_properties = &fake_get_properties;
    _api.handle_register       = &fake_handle_register;
    _api.handle_deregister     = &fake_handle_deregister;
    _api.read                  = &fake_read;
    _api.buf_register          = &fake_buf_register;
    _api.buf_deregister        = &fake_buf_deregister;
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&_api);
  }
  ~scripted_cufile() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  scripted_cufile(scripted_cufile const&)            = delete;
  scripted_cufile& operator=(scripted_cufile const&) = delete;

  [[nodiscard]] std::vector<fake_call> calls() const
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    return state.calls;
  }
  [[nodiscard]] bool registered(void const* base) const
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    return std::find(state.registered.begin(), state.registered.end(), base) !=
           state.registered.end();
  }
  void fail_call(std::size_t index)
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    state.fail_call = index;
  }
  void max_return(std::size_t bytes)
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    state.max_return = bytes;
  }
  /// Make cuFileBufRegister fail, as an exhausted BAR1 would.
  void refuse_registration()
  {
    auto& state = fake();
    std::lock_guard lock(state.mutex);
    state.register_buffers = false;
  }

 private:
  cufile_api _api{};
};

/// Makes gds_available() report GDS for the guard's lifetime, so automatic
/// staging runs on hosts without a GDS stack.
class gds_reported {
 public:
  gds_reported() { cucascade::io::cufile::detail::set_gds_available_for_testing(true); }
  ~gds_reported() { cucascade::io::cufile::detail::set_gds_available_for_testing(std::nullopt); }
  gds_reported(gds_reported const&)            = delete;
  gds_reported& operator=(gds_reported const&) = delete;
};

/// A staging piece size whose slot (piece + two blocks, rounded up to 64 KiB) is
/// exactly @p k x 64 KiB: tests use distinct @p k so the process-wide staging pool
/// never hands one test another's slots.
[[nodiscard]] constexpr std::size_t slot_class(std::size_t k) noexcept
{
  return k * (64UL << 10) - 2 * block;
}

/// A forced-staging config with pieces of @p piece_bytes.
[[nodiscard]] config staging_config(std::size_t piece_bytes)
{
  config cfg;
  cfg.gds_threshold             = 1;
  cfg.staging_mode              = staging::always;
  cfg.staging_piece_bytes       = piece_bytes;
  cfg.staging_slots_per_reactor = 2;
  return cfg;
}

/** A started flat cufile_ioctx over @p cfg, shut down on scope exit. */
class started_ioctx {
 public:
  explicit started_ioctx(config cfg, std::size_t n_reactors = 2)
    : _ioctx(n_reactors, std::make_shared<cufile_reactor::reactor_context>(std::move(cfg)))
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

/// Releases a host-held stream when it goes out of scope, whatever happened.
class stream_latch {
 public:
  stream_latch() : _released(_promise.get_future().share()) {}
  ~stream_latch() { release(); }
  stream_latch(stream_latch const&)            = delete;
  stream_latch& operator=(stream_latch const&) = delete;

  /// Hold @p stream until @ref release (or 30 s, so a failing test cannot hang).
  void hold(cudaStream_t stream)
  {
    auto* const waiter = new std::shared_future<void>(_released);
    REQUIRE(cudaLaunchHostFunc(
              stream,
              [](void* arg) {
                std::unique_ptr<std::shared_future<void>> const released{
                  static_cast<std::shared_future<void>*>(arg)};
                std::ignore = released->wait_for(30s);
              },
              waiter) == cudaSuccess);
  }

  void release()
  {
    if (!_done.exchange(true)) _promise.set_value();
  }

 private:
  std::promise<void> _promise;
  std::shared_future<void> _released;
  std::atomic<bool> _done{false};
};

/// Every call the scripted cuFile saw is what a direct DMA can serve: a 4 KiB
/// aligned file offset, a whole number of blocks unless it ends exactly at the end
/// of the file, never past it, into the base of a registered slot.
void check_calls_are_dma_ready(scripted_cufile const& cufile)
{
  auto const calls = cufile.calls();
  REQUIRE_FALSE(calls.empty());
  for (auto const& call : calls) {
    INFO("cuFileRead(offset " << call.file_offset << ", size " << call.size << ") of a "
                              << call.file_size << "-byte file");
    CHECK(call.file_offset % block == 0);
    CHECK(call.file_offset + call.size <= call.file_size);
    CHECK((call.size % block == 0 || call.file_offset + call.size == call.file_size));
    CHECK(call.buffer_offset == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(call.base) % block == 0);
    CHECK(cufile.registered(call.base));
  }
}

}  // namespace

TEST_CASE("cufile staged_calls widens to blocks and splits the end-of-file tail",
          "[io][cufile][staging]")
{
  using cucascade::io::cufile::detail::staged_calls;
  constexpr std::size_t slot = 64 * block;
  // Inside one slot: one call, wherever the range starts.
  CHECK(staged_calls(0, block, 1UL << 30, slot) == 1);
  CHECK(staged_calls(13, 13 + slot - 2 * block, 1UL << 30, slot) == 1);
  // A slot's worth starting unaligned spills into a second call.
  CHECK(staged_calls(13, 13 + slot, 1UL << 30, slot) == 2);
  // Reaching into a partial last block: the aligned body, then the tail.
  CHECK(staged_calls(0, 10 * block + 7, 10 * block + 7, slot) == 2);
  // Only the tail: one call.
  CHECK(staged_calls(10 * block + 1, 10 * block + 7, 10 * block + 7, slot) == 1);
  // A file that ends on a block boundary needs no tail call.
  CHECK(staged_calls(5, 10 * block, 10 * block, slot) == 1);
  CHECK(staged_calls(7, 7, 100, slot) == 0);
}

TEST_CASE("cufile staged reads ask cuFile only for aligned ranges within the file",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  // A file ending in a partial block; pieces of 64 KiB so ranges span several.
  pattern_file file{(3UL << 20) + 1234};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);

  struct sample {
    std::size_t offset;
    std::size_t size;
    std::size_t misalign;  // of the destination
  };
  auto const s = GENERATE(sample{0, block, 0},
                          sample{1, 1, 13},
                          sample{block - 1, 2, 255},
                          sample{12345, (1UL << 20) + 3, 13},
                          sample{block, 64UL << 10, 0},
                          sample{(3UL << 20) + 1234 - 5000, 5000, 7},  // the last bytes
                          sample{(3UL << 20) + 1000, 234, 1},          // inside the tail block
                          sample{(3UL << 20) - 3 * block + 9, 9000, 4093});  // body and tail
  INFO("offset " << s.offset << " size " << s.size << " misalign " << s.misalign);

  rmm::cuda_stream stream;
  constexpr std::size_t guard = 64;
  rmm::device_buffer dst(s.size + s.misalign + 2 * guard, ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{staging_config(slot_class(3))};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(cudaMemsetAsync(dst.data(), 0xEE, dst.size(), stream.value()) == cudaSuccess);
  auto* const target = static_cast<std::uint8_t*>(dst.data()) + guard + s.misalign;
  REQUIRE(
    ioctx.get()
      .device_read_async_io(*object, s.offset, s.size, target, ::cuda::stream_ref{stream.value()})
      .get(60s) == s.size);

  auto const all = bytes_at(dst.data(), dst.size());
  std::vector<std::uint8_t> const got(
    all.begin() + static_cast<std::ptrdiff_t>(guard + s.misalign),
    all.begin() + static_cast<std::ptrdiff_t>(guard + s.misalign + s.size));
  CHECK(matches_file(got, s.offset));
  for (std::size_t i = 0; i < guard + s.misalign; ++i) {
    REQUIRE(all[i] == 0xEE);  // nothing before the destination
  }
  for (std::size_t i = guard + s.misalign + s.size; i < all.size(); ++i) {
    REQUIRE(all[i] == 0xEE);  // nothing after it
  }
  check_calls_are_dma_ready(cufile);

  auto const stats = ioctx.get().stats();
  CHECK(stats.staged_bytes == s.size);
  CHECK(stats.direct_reads == 0);
  CHECK(stats.posix_device_reads == 0);
  CHECK(stats.cufile_failures == 0);
}

TEST_CASE("cufile splits a large device read into pieces several reactors stage",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{4UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  constexpr std::size_t piece = slot_class(4);

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 777;
  constexpr std::size_t size   = (2UL << 20) + 99;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{staging_config(piece), 4};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(ioctx.get()
            .device_read_async_io(*object,
                                  offset,
                                  size,
                                  static_cast<std::uint8_t*>(dst.data()),
                                  ::cuda::stream_ref{stream.value()})
            .get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));

  // Pieces end at align_down(offset) + k * piece, so each is one aligned call.
  auto const expected = (offset + size - (offset & ~(block - 1)) + piece - 1) / piece;
  auto const stats    = ioctx.get().stats();
  CHECK(stats.staged_reads == expected);
  CHECK(stats.staged_bytes == size);
  CHECK(cufile.calls().size() == expected);
  check_calls_are_dma_ready(cufile);
}

TEST_CASE("cufile staged read that cuFile fails is served by POSIX instead",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  cufile.fail_call(1);

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 4100;
  constexpr std::size_t size   = 100000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{staging_config(slot_class(5)), 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(ioctx.get()
            .device_read_async_io(*object,
                                  offset,
                                  size,
                                  static_cast<std::uint8_t*>(dst.data()),
                                  ::cuda::stream_ref{stream.value()})
            .get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));
  auto const stats = ioctx.get().stats();
  CHECK(stats.cufile_failures == 1);
  CHECK(stats.posix_device_reads == 1);
  CHECK(stats.staged_reads == 0);
}

TEST_CASE("cufile staged reads continue after short cuFile returns", "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{(1UL << 20) + 77};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  cufile.max_return(block + 100);  // never a whole piece, and never block-aligned

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 3;
  std::size_t const size       = file.size() - offset;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{staging_config(slot_class(6)), 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(ioctx.get()
            .device_read_async_io(*object,
                                  offset,
                                  size,
                                  static_cast<std::uint8_t*>(dst.data()),
                                  ::cuda::stream_ref{stream.value()})
            .get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));
  CHECK(ioctx.get().stats().cufile_failures == 0);
  for (auto const& call : cufile.calls()) {
    CHECK(call.file_offset + call.size <= call.file_size);
  }
}

TEST_CASE("cufile reads a staged piece before its busy stream is ready; only the copy waits",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 8192 + 5;
  constexpr std::size_t size   = 100000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{staging_config(slot_class(7)), 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(cudaMemsetAsync(dst.data(), 0, size, stream.value()) == cudaSuccess);
  stream_latch latch;
  latch.hold(stream.value());

  auto future = ioctx.get().device_read_async_io(*object,
                                                 offset,
                                                 size,
                                                 static_cast<std::uint8_t*>(dst.data()),
                                                 ::cuda::stream_ref{stream.value()});
  // The storage read happens while the stream is held...
  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (cufile.calls().empty() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  CHECK(cufile.calls().size() == 1);
  // ...but the read does not settle, nor touch the destination, before it.
  std::this_thread::sleep_for(100ms);
  CHECK_FALSE(future.is_ready());
  latch.release();
  REQUIRE(std::move(future).get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));
  auto const stats = ioctx.get().stats();
  CHECK(stats.early_staged_reads == 1);
  CHECK(stats.staged_reads == 1);
}

TEST_CASE("cufile device read on a per-thread default stream waits for that thread's work",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  // The real cuFile (or POSIX): the gate must hold whatever path serves the read.
  config cfg;
  cfg.gds_threshold = 1;

  constexpr std::size_t size = 300000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{cudaStream_t{nullptr}});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  stream_latch latch;
  std::optional<cucascade::exec::semi_future<std::size_t>> future;
  std::promise<void> submitted;
  auto submitted_future = submitted.get_future();
  std::thread submitter([&] {
    // This thread's per-thread default stream is busy: the read must wait for it,
    // although on the reactor's worker the same handle names an idle stream.
    latch.hold(cudaStreamPerThread);
    future.emplace(ioctx.get().device_read_async_io(*object,
                                                    0,
                                                    size,
                                                    static_cast<std::uint8_t*>(dst.data()),
                                                    ::cuda::stream_ref{cudaStreamPerThread}));
    submitted.set_value();
    // Keep the thread, and so its per-thread stream, alive until the work drains.
    std::ignore = cudaStreamSynchronize(cudaStreamPerThread);
  });
  struct joiner {
    stream_latch& latch;
    std::thread& thread;
    ~joiner()
    {
      latch.release();
      if (thread.joinable()) thread.join();
    }
  } const join_on_exit{latch, submitter};
  REQUIRE(submitted_future.wait_for(30s) == std::future_status::ready);
  REQUIRE(future.has_value());
  std::this_thread::sleep_for(200ms);
  CHECK_FALSE(future->is_ready());
  latch.release();
  REQUIRE(std::move(*future).get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), 0));
}

TEST_CASE("cufile refuses a device read on a stream being captured, leaving the capture intact",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{64UL << 10};

  rmm::cuda_stream stream;
  rmm::device_buffer dst(file.size(), ::cuda::stream_ref{stream.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{config{}, 1};
  auto object = ioctx.get().open_io_object(file.path());
  stream.synchronize();
  REQUIRE(cudaStreamBeginCapture(stream.value(), cudaStreamCaptureModeRelaxed) == cudaSuccess);
  auto future = ioctx.get().device_read_async_io(*object,
                                                 0,
                                                 file.size(),
                                                 static_cast<std::uint8_t*>(dst.data()),
                                                 ::cuda::stream_ref{stream.value()});
  cudaGraph_t graph{};
  // The capture was not invalidated by the refused read.
  CHECK(cudaStreamEndCapture(stream.value(), &graph) == cudaSuccess);
  if (graph != nullptr) CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK_THROWS_AS(std::move(future).get(10s), std::invalid_argument);
}

TEST_CASE("cufile parked device slices do not hold up other requests on the reactor",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  config cfg;
  cfg.gds_threshold = 1;
  cfg.max_inflight  = 2;
  cfg.staging_mode  = staging::never;

  rmm::cuda_stream busy;
  constexpr std::size_t slices = 32;
  constexpr std::size_t piece  = 4096;
  rmm::device_buffer dst(slices * piece, ::cuda::stream_ref{busy.value()});
  // Built after the destination, so a failing test shuts it down (waiting for
  // any read in flight) before the destination is freed.
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  busy.synchronize();
  stream_latch latch;
  latch.hold(busy.value());
  std::vector<cucascade::io::slice> ranges;
  for (std::size_t i = 0; i < slices; ++i) {
    ranges.emplace_back(i * piece, piece, static_cast<std::uint8_t*>(dst.data()) + i * piece);
  }
  auto parked =
    ioctx.get().device_readv_async_io(*object, ranges, ::cuda::stream_ref{busy.value()});

  // More slices are parked than the in-flight window holds, yet a host read on the
  // same (only) reactor is served while the stream stays busy.
  std::vector<std::uint8_t> host(piece);
  auto const start = std::chrono::steady_clock::now();
  REQUIRE(ioctx.get().host_read_async_io(*object, 0, piece, host.data()).get(10s) == piece);
  CHECK(std::chrono::steady_clock::now() - start < 5s);
  CHECK_FALSE(parked.is_ready());
  latch.release();
  REQUIRE(std::move(parked).get(60s) == slices * piece);
  CHECK(matches_file(bytes_at(dst.data(), slices * piece), 0));
  CHECK(ioctx.get().stats().parked_slices >= slices);
}

TEST_CASE("cufile reads a destination on another GPU than the calling thread's",
          "[io][cufile][staging][gpu]")
{
  if (device_count() < 2) SKIP("needs two CUDA devices");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);

  constexpr std::size_t offset = 100;
  constexpr std::size_t size   = 200000;
  std::optional<rmm::cuda_stream> stream;
  std::optional<rmm::device_buffer> dst;
  {
    rmm::cuda_set_device_raii const on_one{rmm::cuda_device_id{1}};
    stream.emplace();
    dst.emplace(size, ::cuda::stream_ref{stream->value()});
    stream->synchronize();
  }
  started_ioctx ioctx{staging_config(slot_class(8)), 1};
  auto object = ioctx.get().open_io_object(file.path());
  rmm::cuda_set_device_raii const on_zero{rmm::cuda_device_id{0}};
  stream_latch latch;
  latch.hold(stream->value());  // busy, so the gate is a real event on GPU 1
  auto future = ioctx.get().device_read_async_io(*object,
                                                 offset,
                                                 size,
                                                 static_cast<std::uint8_t*>(dst->data()),
                                                 ::cuda::stream_ref{stream->value()});
  latch.release();
  REQUIRE(std::move(future).get(60s) == size);
  rmm::cuda_set_device_raii const back_on_one{rmm::cuda_device_id{1}};
  CHECK(matches_file(bytes_at(dst->data(), size), offset));
}

TEST_CASE("cufile reserves staging once per device and reuses it", "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  scripted_cufile cufile;
  auto cfg          = staging_config(slot_class(9));
  auto const before = cucascade::io::cufile::staging_usage_of(0);
  CHECK(cucascade::io::cufile::reserve_staging(0, 3, cfg) == 3);
  auto const after = cucascade::io::cufile::staging_usage_of(0);
  CHECK(after.allocated_bytes >= before.allocated_bytes + 3 * cfg.staging_slot_bytes());
  CHECK(after.registered_bytes >= before.registered_bytes + 3 * cfg.staging_slot_bytes());
  // Reserving again takes the same slots: nothing more is allocated.
  CHECK(cucascade::io::cufile::reserve_staging(0, 3, cfg) == 3);
  CHECK(cucascade::io::cufile::staging_usage_of(0).allocated_bytes == after.allocated_bytes);
  // The cap stops a device's staging from growing past it.
  auto capped                         = staging_config(slot_class(10));
  capped.max_staging_bytes_per_device = after.allocated_bytes;
  CHECK(cucascade::io::cufile::reserve_staging(0, 1, capped) == 0);
}

TEST_CASE("cufile pool plan gives each GPU its own pinned reactor group",
          "[io][cufile][staging][gpu]")
{
  auto const devices = device_count();
  if (devices == 0) SKIP("no CUDA device");
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS));
  auto const& topology = discovery.get_topology();

  config cfg;
  cfg.reactors_per_device = 3;
  cfg.staging_mode        = staging::never;
  auto const plan         = cucascade::io::cufile::plan_cufile_pool(
    cfg, &topology, cucascade::io::cufile::probe_platform());
  INFO(cucascade::io::cufile::describe(plan));
  REQUIRE(plan.groups.size() == static_cast<std::size_t>(devices));
  CHECK(plan.total_reactors() == 3 * static_cast<std::size_t>(devices));
  for (auto const& group : plan.groups) {
    CHECK(group.n_reactors == 3);
    CHECK(group.staging_slots == 0);
    CHECK_FALSE(group.pci_bus_id.empty());
  }

  // Built from the plan, a read is routed to its destination GPU's group.
  cufile_ioctx ioctx{plan, std::make_shared<cufile_reactor::reactor_context>(cfg)};
  pattern_file file{64UL << 10};
  auto object       = ioctx.open_io_object(file.path());
  auto const& typed = dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object);
  for (int device = 0; device < devices; ++device) {
    for (auto const type :
         {cufile_ioctx::io_op_type::host_vector_async, cufile_ioctx::io_op_type::device_async}) {
      auto const picked = ioctx.next_reactor(typed, 8, type, device);
      REQUIRE_FALSE(picked.empty());
      for (auto const* reactor : picked) {
        CHECK(reactor->placement().device == device);
      }
    }
  }
  // Host-only reads may go anywhere.
  CHECK_FALSE(ioctx.next_reactor(typed, 1, cufile_ioctx::io_op_type::host, -1).empty());
}

TEST_CASE("cufile pool plan sizes staging within the configured cap", "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  config cfg;
  cfg.reactors_per_device          = 4;
  cfg.staging_mode                 = staging::always;
  cfg.staging_piece_bytes          = 1UL << 20;
  cfg.max_staging_bytes_per_device = 3 * (cfg.staging_slot_bytes() + (64UL << 10));
  auto const plan =
    cucascade::io::cufile::plan_cufile_pool(cfg, nullptr, cucascade::io::cufile::probe_platform());
  INFO(cucascade::io::cufile::describe(plan));
  REQUIRE_FALSE(plan.groups.empty());
  for (auto const& group : plan.groups) {
    CHECK(group.staging_slots == 3);  // 4 reactors x 2 slots wanted, the cap allows 3
    CHECK(group.staging_cap_bytes <= cfg.max_staging_bytes_per_device);
    CHECK(group.cpus.empty());  // no topology: unpinned
  }
}

TEST_CASE("cufile shutdown waits for an early staged copy before settling it",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);

  rmm::cuda_stream stream;
  constexpr std::size_t size = 100000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  stream.synchronize();
  cufile_ioctx ioctx{
    1, std::make_shared<cufile_reactor::reactor_context>(staging_config(slot_class(11)))};
  ioctx.start();
  auto object = ioctx.open_io_object(file.path());
  stream_latch latch;
  latch.hold(stream.value());
  auto future = ioctx.device_read_async_io(
    *object, 5, size, static_cast<std::uint8_t*>(dst.data()), ::cuda::stream_ref{stream.value()});
  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (cufile.calls().empty() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(cufile.calls().size() == 1);  // read early, its copy waiting for the stream

  // Shutdown must not give the request back while its copy may still write the
  // destination: it returns only once the stream let the copy run.
  auto stopped = std::async(std::launch::async, [&ioctx] { ioctx.shutdown(); });
  CHECK(stopped.wait_for(300ms) == std::future_status::timeout);
  CHECK_FALSE(future.is_ready());
  latch.release();
  REQUIRE(stopped.wait_for(30s) == std::future_status::ready);
  // Settled (as cancelled, since shutdown took it), and only after the copy landed.
  REQUIRE(std::move(future).wait(5s));
  CHECK(matches_file(bytes_at(dst.data(), size), 5));
}

TEST_CASE("cufile early staged read that cuFile fails is served by POSIX after its stream",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  cufile.fail_call(1);

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 777;
  constexpr std::size_t size   = 300000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  stream.synchronize();
  started_ioctx ioctx{staging_config(slot_class(12)), 1};
  auto object = ioctx.get().open_io_object(file.path());
  stream_latch latch;
  latch.hold(stream.value());
  auto future = ioctx.get().device_read_async_io(*object,
                                                 offset,
                                                 size,
                                                 static_cast<std::uint8_t*>(dst.data()),
                                                 ::cuda::stream_ref{stream.value()});
  // The failed cuFile call happened early; the POSIX retry writes the destination
  // itself, so it must wait for the stream.
  std::this_thread::sleep_for(200ms);
  CHECK_FALSE(future.is_ready());
  latch.release();
  REQUIRE(std::move(future).get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));
  auto const stats = ioctx.get().stats();
  CHECK(stats.cufile_failures == 1);
  CHECK(stats.posix_device_reads == 1);
  CHECK(stats.early_staged_reads == 0);
}

TEST_CASE("cufile routes a read to its destination GPU's group, whatever the caller's device",
          "[io][cufile][staging][gpu]")
{
  if (device_count() < 2) SKIP("needs two CUDA devices");
  pattern_file file{1UL << 20};
  config cfg;
  cfg.reactors_per_device    = 2;
  cfg.staging_mode           = staging::never;
  cfg.gds_threshold          = 1;
  constexpr std::size_t size = 200000;
  std::optional<rmm::cuda_stream> stream;
  std::optional<rmm::device_buffer> dst;
  {
    rmm::cuda_set_device_raii const on_one{rmm::cuda_device_id{1}};
    stream.emplace();
    dst.emplace(size, ::cuda::stream_ref{stream->value()});
    stream->synchronize();
  }
  auto const plan =
    cucascade::io::cufile::plan_cufile_pool(cfg, nullptr, cucascade::io::cufile::probe_platform());
  cufile_ioctx ioctx{plan, std::make_shared<cufile_reactor::reactor_context>(cfg)};
  ioctx.start();
  auto object = ioctx.open_io_object(file.path());
  {
    rmm::cuda_set_device_raii const on_zero{rmm::cuda_device_id{0}};
    REQUIRE(ioctx
              .device_read_async_io(*object,
                                    0,
                                    size,
                                    static_cast<std::uint8_t*>(dst->data()),
                                    ::cuda::stream_ref{stream->value()})
              .get(60s) == size);
  }
  auto const on_one  = ioctx.stats(1);
  auto const on_zero = ioctx.stats(0);
  CHECK(on_one.direct_bytes + on_one.posix_device_bytes == size);
  CHECK(on_zero.direct_bytes + on_zero.posix_device_bytes == 0);
  ioctx.shutdown();
}

TEST_CASE("cufile automatic staging reads into registered slots when cuFile reports GDS",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  gds_reported gds;

  rmm::cuda_stream stream;
  constexpr std::size_t offset = 12345;
  constexpr std::size_t size   = 400000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  stream.synchronize();
  auto cfg         = staging_config(slot_class(13));
  cfg.staging_mode = staging::automatic;
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  REQUIRE(ioctx.get()
            .device_read_async_io(*object,
                                  offset,
                                  size,
                                  static_cast<std::uint8_t*>(dst.data()),
                                  ::cuda::stream_ref{stream.value()})
            .get(60s) == size);
  CHECK(matches_file(bytes_at(dst.data(), size), offset));
  check_calls_are_dma_ready(cufile);
  CHECK(ioctx.get().stats().staged_bytes == size);
}

TEST_CASE("cufile staging that cuFile will not register falls back to direct reads",
          "[io][cufile][staging][gpu]")
{
  if (device_count() == 0) SKIP("no CUDA device");
  pattern_file file{1UL << 20};
  scripted_cufile cufile;
  REQUIRE_ODIRECT(file);
  cufile.refuse_registration();
  gds_reported gds;

  auto cfg         = staging_config(slot_class(14));
  cfg.staging_mode = staging::automatic;
  // Registration is refused: nothing can be reserved, and nothing is allocated
  // over and over trying.
  CHECK(cucascade::io::cufile::reserve_staging(0, 2, cfg) == 0);
  auto const allocated = cucascade::io::cufile::staging_usage_of(0).allocated_bytes;
  CHECK(cucascade::io::cufile::reserve_staging(0, 2, cfg) == 0);
  CHECK(cucascade::io::cufile::staging_usage_of(0).allocated_bytes == allocated);

  rmm::cuda_stream stream;
  constexpr std::size_t size = 300000;
  rmm::device_buffer dst(size, ::cuda::stream_ref{stream.value()});
  stream.synchronize();
  {
    started_ioctx ioctx{cfg, 1};
    auto object = ioctx.get().open_io_object(file.path());
    REQUIRE(ioctx.get()
              .device_read_async_io(*object,
                                    9,
                                    size,
                                    static_cast<std::uint8_t*>(dst.data()),
                                    ::cuda::stream_ref{stream.value()})
              .get(60s) == size);
    CHECK(matches_file(bytes_at(dst.data(), size), 9));
    auto const stats = ioctx.get().stats();
    CHECK(stats.staged_reads == 0);
    CHECK(stats.direct_bytes == size);
  }
  // Forced staging still stages, into the unregistered slot (cuFile bounces it).
  {
    auto forced         = cfg;
    forced.staging_mode = staging::always;
    started_ioctx ioctx{forced, 1};
    auto object = ioctx.get().open_io_object(file.path());
    REQUIRE(ioctx.get()
              .device_read_async_io(*object,
                                    9,
                                    size,
                                    static_cast<std::uint8_t*>(dst.data()),
                                    ::cuda::stream_ref{stream.value()})
              .get(60s) == size);
    CHECK(matches_file(bytes_at(dst.data(), size), 9));
    CHECK(ioctx.get().stats().staged_bytes == size);
  }
  // Freeing the device's spare staging lets a later registration try again.
  std::ignore = cucascade::io::cufile::release_staging(0);
}
