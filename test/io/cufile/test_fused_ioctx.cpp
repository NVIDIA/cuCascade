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
 * [cufile]      - cuFile backend
 * [fused_ioctx] - fused stream-lane reads end to end through cufile_ioctx
 * [gpu]         - needs a CUDA device
 *
 * Every destination byte is checked against the file, and the guard bytes around every
 * destination against their fill.  Cases run against libcufile (compatibility mode on hosts
 * without GDS) unless they install a scripted cuFile table to inject failures.
 */

#include "io/cufile/fused_lanes.hpp"
#include "io/cufile/staging_pool.hpp"
#include "transport_simulator_kernels.hpp"

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using cucascade::io::io_priority;
using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;
using cucascade::io::cufile::engine_mode;
using cucascade::io::cufile::lane_completion;
using cucascade::io::cufile::make_cufile_ioctx;
using cucascade::io::cufile::reactor_stats;
using io_slice = cucascade::io::slice;

namespace {

using namespace std::chrono_literals;

constexpr std::size_t block = 4096;
constexpr std::size_t KiB   = 1024;
constexpr std::size_t MiB   = 1024 * KiB;
constexpr std::uint8_t guard_fill = 0xA5;

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
            ("cucascade_fused_ioctx_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin")),
      _data(bytes)
  {
    for (std::size_t i = 0; i < bytes; ++i) {
      _data[i] = pattern_byte(i);
    }
    std::ofstream out(_path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(_data.data()), static_cast<std::streamsize>(bytes));
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
  [[nodiscard]] std::size_t size() const noexcept { return _data.size(); }
  [[nodiscard]] std::uint8_t const* data() const noexcept { return _data.data(); }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
  std::vector<std::uint8_t> _data;
};

[[nodiscard]] bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void require_async_cufile()
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_async_read || !cufile_api::open_driver()) {
    SKIP("libcufile with cuFileReadAsync is unavailable");
  }
}

/// Device memory (outside any pool) filled with @ref guard_fill, freed on scope exit.
class device_memory {
 public:
  explicit device_memory(std::size_t bytes) : _bytes(bytes)
  {
    REQUIRE(cudaMalloc(&_ptr, bytes) == cudaSuccess);
    fill();
  }
  ~device_memory() { std::ignore = cudaFree(_ptr); }
  device_memory(device_memory const&)            = delete;
  device_memory& operator=(device_memory const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return static_cast<std::uint8_t*>(_ptr); }
  [[nodiscard]] std::size_t size() const noexcept { return _bytes; }

  void fill() const
  {
    REQUIRE(cudaMemset(_ptr, guard_fill, _bytes) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }

  /// The whole buffer, copied on a fresh stream (the read's stream is not synchronized:
  /// a settled read must already be in place).
  [[nodiscard]] std::vector<std::uint8_t> host_copy() const
  {
    cudaStream_t other{};
    REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
    std::vector<std::uint8_t> out(_bytes);
    REQUIRE(cudaMemcpyAsync(out.data(), _ptr, _bytes, cudaMemcpyDeviceToHost, other) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
    return out;
  }

 private:
  void* _ptr{nullptr};
  std::size_t _bytes;
};

/// A non-blocking stream destroyed on scope exit.
class stream_holder {
 public:
  stream_holder() { REQUIRE(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking) == cudaSuccess); }
  ~stream_holder()
  {
    std::ignore = cudaStreamSynchronize(_stream);
    std::ignore = cudaStreamDestroy(_stream);
  }
  stream_holder(stream_holder const&)            = delete;
  stream_holder& operator=(stream_holder const&) = delete;
  [[nodiscard]] cudaStream_t get() const noexcept { return _stream; }
  [[nodiscard]] ::cuda::stream_ref ref() const noexcept { return ::cuda::stream_ref{_stream}; }

 private:
  cudaStream_t _stream{};
};

/**
 * Destinations of one vectored read laid out in a device arena with guard bytes around
 * each; @ref verify checks every destination against the file and every guard against its
 * fill.
 */
class read_plan {
 public:
  /// Add a slice of @p size at file offset @p offset; returns its destination offset.
  void add(std::size_t offset, std::size_t size, std::size_t guard = 61)
  {
    _cursor += guard;
    _entries.push_back(entry{offset, size, _cursor});
    _cursor += size;
  }
  [[nodiscard]] std::size_t arena_bytes() const noexcept { return _cursor + 4096; }

  [[nodiscard]] std::vector<io_slice> slices(std::uint8_t* arena) const
  {
    std::vector<io_slice> out;
    out.reserve(_entries.size());
    for (auto const& e : _entries) {
      out.emplace_back(e.offset, e.size, arena + e.at);
    }
    return out;
  }

  /// Bytes the read returns: every slice clamped to the file.
  [[nodiscard]] std::size_t expected_bytes(std::size_t file_size) const noexcept
  {
    std::size_t total = 0;
    for (auto const& e : _entries) {
      total += e.offset >= file_size ? 0 : std::min(e.size, file_size - e.offset);
    }
    return total;
  }

  /// Whether every destination holds the file's bytes (clamped to EOF; past EOF the fill
  /// stays) and every byte outside them still holds the fill.
  [[nodiscard]] bool verify(std::vector<std::uint8_t> const& arena,
                            pattern_file const& file,
                            std::string* why = nullptr) const
  {
    std::vector<bool> covered(arena.size(), false);
    for (std::size_t i = 0; i < _entries.size(); ++i) {
      auto const& e     = _entries[i];
      auto const inside = e.offset >= file.size() ? 0 : std::min(e.size, file.size() - e.offset);
      if (std::memcmp(arena.data() + e.at, file.data() + e.offset, inside) != 0) {
        if (why) *why = "slice " + std::to_string(i) + " content";
        return false;
      }
      for (std::size_t b = e.at; b < e.at + inside; ++b) {
        covered[b] = true;
      }
    }
    for (std::size_t b = 0; b < arena.size(); ++b) {
      if (!covered[b] && arena[b] != guard_fill) {
        if (why) *why = "guard byte " + std::to_string(b) + " overwritten";
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::size_t count() const noexcept { return _entries.size(); }

 private:
  struct entry {
    std::size_t offset;
    std::size_t size;
    std::size_t at;
  };
  std::vector<entry> _entries;
  std::size_t _cursor{0};
};

/// A random vectored read of @p file: contiguous runs, gaps, overlaps and duplicates,
/// unaligned offsets, sizes from 4 KiB to @p max_size, about @p budget bytes in total.
[[nodiscard]] read_plan random_plan(std::mt19937_64& rng,
                                    std::size_t file_size,
                                    std::size_t max_slices,
                                    std::size_t max_size,
                                    std::size_t budget)
{
  auto uniform = [&](std::size_t lo, std::size_t hi) {
    return std::uniform_int_distribution<std::size_t>{lo, hi}(rng);
  };
  read_plan plan;
  auto const n         = uniform(1, max_slices);
  std::size_t total    = 0;
  std::size_t previous = uniform(0, file_size - 1);
  std::size_t last_end = previous;
  for (std::size_t i = 0; i < n && total < budget; ++i) {
    auto size = uniform(0, 9) == 0 ? uniform(block, max_size) : uniform(block, 256 * KiB);
    size      = std::min(size, budget - total + block);
    std::size_t offset = 0;
    switch (uniform(0, 4)) {
      case 0: offset = last_end; break;                                  // contiguous
      case 1: offset = last_end + uniform(1, 64 * KiB); break;           // gap
      case 2: offset = previous + uniform(0, 8 * KiB); break;            // overlap
      case 3: offset = previous; break;                                  // duplicate
      default: offset = uniform(0, file_size - 1); break;                // anywhere
    }
    offset = std::min(offset, file_size - 1);
    size   = std::min(size, file_size - offset);
    if (size == 0) continue;
    plan.add(offset, size, uniform(0, 300));
    previous = offset;
    last_end = offset + size;
    total += size;
  }
  if (plan.count() == 0) plan.add(0, block);
  return plan;
}

/// A config with fused lanes on: @p lanes lanes per reactor of @p slot_bytes.
[[nodiscard]] config fused_config(std::size_t lanes = 4, std::size_t slot_bytes = 1 * MiB)
{
  config cfg;
  cfg.gds_threshold                    = block;
  cfg.max_staging_bytes_per_device     = 0;  // no cap: other tests' slots must not starve lanes
  cfg.reactors_per_device              = 2;
  cfg.fused_reads.enabled              = true;
  cfg.fused_reads.reactors_per_device  = 2;
  cfg.fused_reads.lanes_per_reactor    = lanes;
  cfg.fused_reads.max_lanes_per_device = 64;
  cfg.fused_reads.slot_bytes           = slot_bytes;
  return cfg;
}

/// A started ioctx, shut down on scope exit.
class started_ioctx {
 public:
  /// A flat pool of @p n_reactors, or a planned pool when it is 0.
  started_ioctx(config const& cfg, std::size_t n_reactors)
    : _ioctx(n_reactors == 0
               ? make_cufile_ioctx(0, cfg)
               : std::make_shared<cufile_ioctx>(
                   n_reactors, std::make_shared<cufile_reactor::reactor_context>(cfg)))
  {
    _ioctx->start();
  }
  ~started_ioctx()
  {
    if (_ioctx) _ioctx->shutdown();
  }
  started_ioctx(started_ioctx const&)            = delete;
  started_ioctx& operator=(started_ioctx const&) = delete;

  [[nodiscard]] cufile_ioctx& get() const noexcept { return *_ioctx; }
  [[nodiscard]] std::shared_ptr<cufile_ioctx>& owner() noexcept { return _ioctx; }

 private:
  std::shared_ptr<cufile_ioctx> _ioctx;
};

/// Sum of every reactor's queued bytes.
[[nodiscard]] std::size_t queued_bytes(cufile_ioctx const& ioctx)
{
  std::size_t total = 0;
  for (std::size_t i = 0; i < ioctx.reactor_count(); ++i) {
    total += ioctx.reactor(i).queued_bytes();
  }
  return total;
}

/// Wait (bounded) until every reactor's backlog drained.
void require_queued_bytes_zero(cufile_ioctx const& ioctx)
{
  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (queued_bytes(ioctx) != 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(queued_bytes(ioctx) == 0);
}

/**
 * Holds a stream on the GPU (a one-thread kernel polling mapped host memory) until released
 * or destroyed.  Unlike a blocking host function it does not occupy CUDA's host-callback
 * thread, which the lanes' completion callbacks need.
 */
class stream_latch {
 public:
  stream_latch()
  {
    void* raw = nullptr;
    REQUIRE(cudaHostAlloc(&raw, sizeof(cucascade::io::cufile::simulator_copy_args),
                          cudaHostAllocMapped) == cudaSuccess);
    _args = static_cast<cucascade::io::cufile::simulator_copy_args*>(raw);
    *_args = cucascade::io::cufile::simulator_copy_args{};
  }
  ~stream_latch()
  {
    release();
    if (_stream != nullptr) std::ignore = cudaStreamSynchronize(_stream);
    std::ignore = cudaFreeHost(_args);
  }
  stream_latch(stream_latch const&)            = delete;
  stream_latch& operator=(stream_latch const&) = delete;

  void hold(cudaStream_t stream)
  {
    _stream = stream;
    REQUIRE(cucascade::io::cufile::launch_simulator_gate(_args, stream) == cudaSuccess);
  }
  void release() noexcept
  {
    std::atomic_ref<unsigned>{_args->released}.store(1, std::memory_order_release);
  }
  [[nodiscard]] cucascade::io::cufile::simulator_copy_args* args() const noexcept { return _args; }

 private:
  cucascade::io::cufile::simulator_copy_args* _args{nullptr};
  cudaStream_t _stream{nullptr};
};

/// Run one vectored read of @p plan and check content, guards and the byte count.
void run_and_verify(cufile_ioctx& ioctx,
                    cucascade::io::io_object const& object,
                    pattern_file const& file,
                    read_plan const& plan,
                    ::cuda::stream_ref stream,
                    io_priority priority = io_priority::automatic)
{
  device_memory arena{plan.arena_bytes()};
  auto const slices = plan.slices(arena.get());
  auto const got    = ioctx.device_readv_async_io(object, slices, stream, priority).get(60s);
  CHECK(got == plan.expected_bytes(file.size()));
  std::string why;
  CHECK(plan.verify(arena.host_copy(), file, &why));
  INFO(why);
}

/// One lane read before a test holds a stream on the GPU: libcufile (compatibility mode)
/// synchronizes the device in its first cuFileReadAsync of the process, which would then wait
/// for the held stream.
void warm_up(cufile_ioctx& ioctx, cucascade::io::io_object const& object, pattern_file const& file)
{
  read_plan plan;
  plan.add(0, 64 * KiB);
  cudaStream_t stream{};
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  run_and_verify(ioctx, object, file, plan, ::cuda::stream_ref{stream});
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

// ---------------------------------------------------------------------------
// A scripted cuFile table (process-wide state: the table holds C function pointers).
// ---------------------------------------------------------------------------

struct fake_action {
  /// latched: the read waits on the GPU for the installed latch (the lane stays busy).
  enum class kind : std::uint8_t { normal, refuse, negative, short_read, latched };
  kind what{kind::normal};
  std::size_t bytes{0};  ///< short_read: bytes returned
};

struct fake_handle {
  int fd{-1};
};

struct fake_state {
  std::mutex mutex;
  std::size_t async_calls{0};
  std::vector<fake_action> script;  ///< per async call; later calls use `every`
  fake_action every{};
  std::deque<std::vector<std::uint8_t>> keep_alive;
  cucascade::io::cufile::simulator_copy_args* latch{nullptr};
  std::vector<void*> pinned;  ///< reads copy from these, round robin
  std::size_t next_buffer{0};
};

/// Largest read the scripted cuFileReadAsync serves (the scripted tests use 1 MiB slots).
constexpr std::size_t fake_buffer_bytes = 1 * MiB;

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
  *handle       = owned;
  return fake_ok();
}

void fake_handle_deregister(CUfileHandle_t handle)
{
  auto* const owned = static_cast<fake_handle*>(handle);
  ::close(owned->fd);
  delete owned;
}

/// Synchronous cuFileRead: pread, then H2D, returning once the bytes are in device memory.
ssize_t fake_read(
  CUfileHandle_t handle, void* base, std::size_t size, off_t file_offset, off_t buffer_offset)
{
  std::vector<std::uint8_t> host(size);
  auto const got = ::pread(static_cast<fake_handle*>(handle)->fd, host.data(), size, file_offset);
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

CUfileError_t fake_buf_register(void const*, std::size_t, int) { return fake_ok(); }
CUfileError_t fake_buf_deregister(void const*) { return fake_ok(); }
CUfileError_t fake_stream_register(CUstream, unsigned) { return fake_ok(); }
CUfileError_t fake_stream_deregister(CUstream) { return fake_ok(); }

CUfileError_t fake_read_async(CUfileHandle_t handle,
                              void* base,
                              std::size_t* size_p,
                              off_t* file_offset_p,
                              off_t* buffer_offset_p,
                              ssize_t* bytes_read_p,
                              CUstream stream)
{
  auto& state = fake();
  std::lock_guard lock(state.mutex);
  auto const index  = state.async_calls++;
  auto const action = index < state.script.size() ? state.script[index] : state.every;
  if (action.what == fake_action::kind::refuse) {
    CUfileError_t status{};
    status.err = CU_FILE_IO_ERROR;
    return status;
  }
  if (action.what == fake_action::kind::negative) {
    *bytes_read_p = -5;
    return fake_ok();
  }
  if (action.what == fake_action::kind::latched && state.latch != nullptr &&
      cucascade::io::cufile::launch_simulator_gate(state.latch, static_cast<cudaStream_t>(stream)) !=
        cudaSuccess) {
    CUfileError_t status{};
    status.err = CU_FILE_IO_ERROR;
    return status;
  }
  auto limit = *size_p;
  if (action.what == fake_action::kind::short_read) limit = std::min(limit, action.bytes);
  std::uint8_t* host = nullptr;
  // Pinned: a pageable copy queued on a stream that cannot run yet would hold the driver's
  // pageable staging and stall every other pageable copy of the process meanwhile.
  // From pinned buffers allocated up front: allocating pinned memory, or copying from
  // pageable memory, can synchronize the device, which would wait for a held stream.
  if (limit > fake_buffer_bytes || state.pinned.empty()) {
    *bytes_read_p = -5;
    return fake_ok();
  }
  host = static_cast<std::uint8_t*>(state.pinned[state.next_buffer++ % state.pinned.size()]);
  auto const got = ::pread(static_cast<fake_handle*>(handle)->fd, host, limit, *file_offset_p);
  if (got > 0 && cudaMemcpyAsync(static_cast<std::uint8_t*>(base) + *buffer_offset_p,
                                 host,
                                 static_cast<std::size_t>(got),
                                 cudaMemcpyHostToDevice,
                                 static_cast<cudaStream_t>(stream)) != cudaSuccess) {
    *bytes_read_p = -5;
    return fake_ok();
  }
  *bytes_read_p = got;
  return fake_ok();
}

/// Installs the scripted cuFile for its lifetime; declare it before the ioctx it serves.
class scripted_cufile {
 public:
  scripted_cufile()
  {
    {
      std::lock_guard lock(fake().mutex);
      fake().async_calls = 0;
      fake().script.clear();
      fake().every = {};
      fake().keep_alive.clear();
      fake().latch = nullptr;
    }
    _api.driver_open           = &fake_driver_open;
    _api.driver_get_properties = &fake_get_properties;
    _api.handle_register       = &fake_handle_register;
    _api.handle_deregister     = &fake_handle_deregister;
    _api.read                  = &fake_read;
    _api.buf_register          = &fake_buf_register;
    _api.buf_deregister        = &fake_buf_deregister;
    _api.read_async            = &fake_read_async;
    _api.has_async_read        = true;
    _api.stream_register       = &fake_stream_register;
    _api.stream_deregister     = &fake_stream_deregister;
    _api.has_stream_register   = true;
    {
      std::lock_guard lock(fake().mutex);
      for (std::size_t i = 0; i < 32; ++i) {
        void* raw = nullptr;
        REQUIRE(cudaMallocHost(&raw, fake_buffer_bytes) == cudaSuccess);
        fake().pinned.push_back(raw);
      }
    }
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&_api);
  }
  ~scripted_cufile()
  {
    cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr);
    std::lock_guard lock(fake().mutex);
    fake().keep_alive.clear();
    fake().latch = nullptr;
    for (auto* memory : fake().pinned) {
      std::ignore = cudaFreeHost(memory);
    }
    fake().pinned.clear();
  }
  scripted_cufile(scripted_cufile const&)            = delete;
  scripted_cufile& operator=(scripted_cufile const&) = delete;

  void script(std::vector<fake_action> actions)
  {
    std::lock_guard lock(fake().mutex);
    fake().script      = std::move(actions);
    fake().async_calls = 0;
  }
  void every(fake_action action)
  {
    std::lock_guard lock(fake().mutex);
    fake().every = action;
  }
  /// Reads scripted @c latched wait on the GPU for @p latch.
  void latch_with(stream_latch const& latch)
  {
    std::lock_guard lock(fake().mutex);
    fake().latch = latch.args();
  }
  [[nodiscard]] std::size_t async_calls() const
  {
    std::lock_guard lock(fake().mutex);
    return fake().async_calls;
  }

 private:
  cufile_api _api{};
};

/// A plan of @p count reads of @p size bytes at @p stride apart from @p first.
[[nodiscard]] read_plan strided_plan(std::size_t first,
                                     std::size_t count,
                                     std::size_t size,
                                     std::size_t stride)
{
  read_plan plan;
  for (std::size_t i = 0; i < count; ++i) {
    plan.add(first + i * stride, size);
  }
  return plan;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1 / 8. Content through flat and planned pools, callback and poll completion.
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: random vectored device reads land the file's bytes",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  auto const pool       = GENERATE(as<std::size_t>{}, 1, 4, 0);  // 0 = planned pool
  auto const completion = GENERATE(lane_completion::callback, lane_completion::poll);
  CAPTURE(pool, static_cast<int>(completion));
  pattern_file const file{(48 * MiB) + 1234};
  auto cfg                   = fused_config(4, 1 * MiB);
  cfg.fused_reads.completion = completion;
  started_ioctx ioctx{cfg, pool};
  auto object = ioctx.get().open_io_object(file.path());
  stream_holder stream;

  std::mt19937_64 rng{1234 + pool};
  auto const before = ioctx.get().stats();
  for (int round = 0; round < 12; ++round) {
    auto const plan = random_plan(rng, file.size(), 40, 24 * MiB, 48 * MiB);
    run_and_verify(ioctx.get(), *object, file, plan, stream.ref());
  }
  // A single-slice read longer than a slot.
  {
    read_plan single;
    single.add(777, 5 * MiB + 3);
    device_memory arena{single.arena_bytes()};
    auto const slices = single.slices(arena.get());
    REQUIRE(ioctx.get()
              .device_read_async_io(*object, 777, 5 * MiB + 3, slices[0].dst, stream.ref())
              .get(60s) == 5 * MiB + 3);
    CHECK(single.verify(arena.host_copy(), file));
  }
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_reads > 0);
  CHECK(delta.lane_parts >= delta.lane_reads);
  CHECK(delta.lane_issued_bytes >= delta.lane_bytes);
  CHECK(delta.lane_failures == 0);
  CHECK(delta.batch_reads == 0);

  // Contiguous chunks fuse: 64 x 100 KiB is about 7 reads of 1 MiB, not 64.
  auto const fused_before = ioctx.get().stats();
  run_and_verify(ioctx.get(), *object, file, strided_plan(3 * MiB + 5, 64, 100 * KiB, 100 * KiB),
                 stream.ref());
  auto const fused = ioctx.get().stats() - fused_before;
  CHECK(fused.lane_parts >= 64);
  CHECK(fused.lane_reads <= 10);
  CHECK(fused.lane_bytes == 64 * 100 * KiB);
  require_queued_bytes_zero(ioctx.get());
}

TEST_CASE("fused ioctx: default 8 MiB slots", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{(40 * MiB) + 7};
  auto cfg = fused_config(4, 8 * MiB);
  started_ioctx ioctx{cfg, 2};
  auto object = ioctx.get().open_io_object(file.path());
  stream_holder stream;
  std::mt19937_64 rng{88};
  auto const before = ioctx.get().stats();
  for (int round = 0; round < 6; ++round) {
    run_and_verify(
      ioctx.get(), *object, file, random_plan(rng, file.size(), 30, 24 * MiB, 40 * MiB), stream.ref());
  }
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_reads > 0);
  CHECK(delta.lane_failures == 0);
}

// ---------------------------------------------------------------------------
// 2. EOF
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: reads at the end of a file", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  constexpr std::size_t size = (3 * MiB) + 1000;  // partial last block
  pattern_file const file{size};
  started_ioctx ioctx{fused_config(4, 1 * MiB), 2};
  auto object = ioctx.get().open_io_object(file.path());
  stream_holder stream;

  read_plan plan;
  plan.add(size - 200 * KiB, 300 * KiB);   // crosses EOF: clamped
  plan.add(size - 500, 400);               // inside the last partial block
  plan.add(size / block * block, 2000);    // from the partial block's start, crossing EOF
  plan.add(0, size);                       // the whole file
  plan.add(2 * MiB + 1, 1 * MiB);          // ends inside the partial block
  run_and_verify(ioctx.get(), *object, file, plan, stream.ref());

  // Entirely past EOF: zero bytes, destination untouched.
  device_memory dst{block};
  CHECK(ioctx.get().device_read_async_io(*object, size + 10, 100, dst.get(), stream.ref()).get(30s) ==
        0);
  auto const after = dst.host_copy();
  CHECK(std::all_of(after.begin(), after.end(), [](std::uint8_t b) { return b == guard_fill; }));
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// 3. Busy caller streams
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: a busy caller stream gates the copies", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  auto cfg                        = fused_config(4, 1 * MiB);
  cfg.fused_reads.max_gated_lanes = 2;
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  warm_up(ioctx.get(), *object, file);
  stream_holder busy;
  stream_holder idle;

  SECTION("a gated lane read completes after the stream is released")
  {
    auto const before = ioctx.get().stats();
    read_plan plan;
    plan.add(12345, 600 * KiB);
    device_memory arena{plan.arena_bytes()};
    stream_latch latch;
    latch.hold(busy.get());
    auto future = ioctx.get().device_readv_async_io(*object, plan.slices(arena.get()), busy.ref());
    std::this_thread::sleep_for(300ms);
    CHECK_FALSE(future.is_ready());
    latch.release();
    CHECK(std::move(future).get(30s) == 600 * KiB);
    CHECK(plan.verify(arena.host_copy(), file));
    auto const delta = ioctx.get().stats() - before;
    CHECK(delta.lane_gated_reads == 1);
    CHECK(delta.lane_reads == 1);
  }

  SECTION("gated reads beyond the limit park; idle streams are served meanwhile")
  {
    auto const before = ioctx.get().stats();
    constexpr std::size_t reads = 5;
    std::vector<read_plan> plans(reads);
    std::vector<std::unique_ptr<device_memory>> arenas;
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    // Allocated before the stream is held: filling them synchronizes the device.
    for (std::size_t i = 0; i < reads; ++i) {
      plans[i].add(i * MiB + 99, 300 * KiB);
      arenas.push_back(std::make_unique<device_memory>(plans[i].arena_bytes()));
    }
    read_plan other;
    other.add(6 * MiB + 1, 200 * KiB);
    device_memory other_arena{other.arena_bytes()};
    stream_latch latch;
    latch.hold(busy.get());
    for (std::size_t i = 0; i < reads; ++i) {
      futures.push_back(
        ioctx.get().device_readv_async_io(*object, plans[i].slices(arenas[i]->get()), busy.ref()));
    }
    // An idle stream's read is served while the busy ones wait.
    CHECK(ioctx.get()
            .device_readv_async_io(*object, other.slices(other_arena.get()), idle.ref())
            .get(20s) == 200 * KiB);
    CHECK(other.verify(other_arena.host_copy(), file));
    for (auto& future : futures) {
      CHECK_FALSE(future.is_ready());
    }
    // Five pending-gate reads, limit 2: two take lanes (two-phase), three park; the idle
    // read got one of the two other lanes.
    auto const held = ioctx.get().stats() - before;
    CHECK(held.parked_slices == reads - 2);
    CHECK(held.lane_reads == 1);
    latch.release();
    for (std::size_t i = 0; i < reads; ++i) {
      CHECK(std::move(futures[i]).get(30s) == 300 * KiB);
      CHECK(plans[i].verify(arenas[i]->host_copy(), file));
    }
    auto const delta = ioctx.get().stats() - before;
    CHECK(delta.lane_gated_reads == 2);
    CHECK(delta.lane_reads == reads + 1);  // the parked ones took lanes once their gate fired
  }

  SECTION("at most the limit of lanes ever wait for gates, whatever the burst")
  {
    // Limit 2 of 4 lanes: a burst of pending-gate reads, submitted back to back before any
    // read completes, never takes more than two lanes; idle streams keep the other two.
    auto const before = ioctx.get().stats();
    constexpr std::size_t reads = 8;
    std::vector<read_plan> plans(reads);
    std::vector<std::unique_ptr<device_memory>> arenas;
    for (std::size_t i = 0; i < reads; ++i) {
      plans[i].add(i * MiB / 2 + 7, 100 * KiB);
      arenas.push_back(std::make_unique<device_memory>(plans[i].arena_bytes()));
    }
    std::vector<read_plan> others(4);
    std::vector<std::unique_ptr<device_memory>> other_arenas;
    for (std::size_t i = 0; i < others.size(); ++i) {
      others[i].add(5 * MiB + i * 300 * KiB + 1, 200 * KiB);
      other_arenas.push_back(std::make_unique<device_memory>(others[i].arena_bytes()));
    }
    stream_latch latch;
    latch.hold(busy.get());
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    for (std::size_t i = 0; i < reads; ++i) {
      futures.push_back(
        ioctx.get().device_readv_async_io(*object, plans[i].slices(arenas[i]->get()), busy.ref()));
    }
    for (std::size_t i = 0; i < others.size(); ++i) {
      CHECK(ioctx.get()
              .device_readv_async_io(*object, others[i].slices(other_arenas[i]->get()), idle.ref())
              .get(20s) == 200 * KiB);
      CHECK(others[i].verify(other_arenas[i]->host_copy(), file));
    }
    auto const held = ioctx.get().stats() - before;
    CHECK(held.parked_slices == reads - 2);
    CHECK(held.lane_reads == others.size());
    for (auto& future : futures) {
      CHECK_FALSE(future.is_ready());
    }
    latch.release();
    for (std::size_t i = 0; i < reads; ++i) {
      CHECK(std::move(futures[i]).get(30s) == 100 * KiB);
      CHECK(plans[i].verify(arenas[i]->host_copy(), file));
    }
    auto const delta = ioctx.get().stats() - before;
    CHECK(delta.lane_gated_reads == 2);
    CHECK(delta.lane_reads == reads + others.size());
  }
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// 4. Failures through a scripted cuFile
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: failed lane reads fall back; the cooldown trips and ends",
          "[io][cufile][fused_ioctx][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile;
  pattern_file const file{8 * MiB + 100};
  auto cfg                      = fused_config(2, 1 * MiB);
  cfg.fused_reads.failure_limit = 2;
  cfg.fused_reads.cooldown_s    = 1;
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  stream_holder stream;
  auto one_read = [&](std::size_t offset) {
    read_plan plan;
    plan.add(offset, 100 * KiB);
    plan.add(offset + 100 * KiB, 50 * KiB + 3);  // fused with the first
    run_and_verify(ioctx.get(), *object, file, plan, stream.ref());
  };

  SECTION("refused, failed and short-without-progress reads still succeed")
  {
    auto const kind = GENERATE(fake_action::kind::refuse, fake_action::kind::negative,
                               fake_action::kind::short_read);
    cufile.script({fake_action{kind, 100}});
    auto const before = ioctx.get().stats();
    one_read(5000);
    auto const delta = ioctx.get().stats() - before;
    CHECK(delta.lane_failures == 1);
    CHECK(delta.cufile_failures >= 1);
    CHECK(delta.lane_reads == 0);
  }

  SECTION("a short read with progress is continued on its lane")
  {
    cufile.script({fake_action{fake_action::kind::short_read, 3 * block + 5}});
    auto const before = ioctx.get().stats();
    one_read(4 * MiB + 17);
    auto const delta = ioctx.get().stats() - before;
    CHECK(delta.lane_failures == 0);
    CHECK(delta.lane_reads == 1);
    CHECK(delta.lane_resubmits == 1);
  }

  SECTION("consecutive failures trip the cooldown; lanes resume after it")
  {
    cufile.every(fake_action{fake_action::kind::refuse, 0});
    cufile.script({});
    one_read(0);
    one_read(1 * MiB);
    auto const tripped = ioctx.get().stats();
    CHECK(tripped.lane_failures == 2);
    auto const calls = cufile.async_calls();
    // Cooling down: served synchronously, cuFileReadAsync not called.
    one_read(2 * MiB);
    CHECK(cufile.async_calls() == calls);
    CHECK(ioctx.get().stats().lane_failures == 2);
    cufile.every(fake_action{});
    std::this_thread::sleep_for(1200ms);
    auto const before = ioctx.get().stats();
    one_read(3 * MiB);
    auto const delta = ioctx.get().stats() - before;
    CHECK(cufile.async_calls() > calls);
    CHECK(delta.lane_reads == 1);
  }
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// 5. Cancellation and shutdown
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: shutdown with reads queued and in flight settles every future",
          "[io][cufile][fused_ioctx][gpu][shutdown]")
{
  require_async_cufile();
  pattern_file const file{32 * MiB};
  stream_holder stream;
  for (int repetition = 0; repetition < 3; ++repetition) {
    auto const destroy = repetition == 2;  // last round: destroy instead of shutdown
    std::optional<started_ioctx> ioctx;
    ioctx.emplace(fused_config(2, 1 * MiB), 2);
    auto object = ioctx->get().open_io_object(file.path());
    constexpr std::size_t requests = 60;
    std::vector<read_plan> plans(requests);
    std::vector<std::unique_ptr<device_memory>> arenas;
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    for (std::size_t i = 0; i < requests; ++i) {
      for (std::size_t s = 0; s < 4; ++s) {
        plans[i].add(((i * 4 + s) * 129 * KiB) % (file.size() - MiB), 300 * KiB + s);
      }
      arenas.push_back(std::make_unique<device_memory>(plans[i].arena_bytes()));
      futures.push_back(ioctx->get().device_readv_async_io(
        *object, plans[i].slices(arenas[i]->get()), stream.ref()));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{5 * repetition});
    if (destroy) {
      ioctx.reset();
    } else {
      ioctx->get().shutdown();
    }
    std::size_t succeeded = 0;
    std::size_t failed    = 0;
    for (std::size_t i = 0; i < requests; ++i) {
      REQUIRE(std::move(futures[i]).wait(30s));
      std::optional<std::size_t> got;
      try {
        got = std::move(futures[i]).get();
      } catch (std::exception const&) {
        ++failed;
      }
      if (got) {
        CHECK(*got == plans[i].expected_bytes(file.size()));
        CHECK(plans[i].verify(arenas[i]->host_copy(), file));
        ++succeeded;
      }
    }
    CHECK(succeeded + failed == requests);
    // Every future settled: the destinations may go now.
    arenas.clear();
    if (!destroy) {
      // Reads after shutdown are cancelled.
      device_memory dst{MiB};
      auto late = ioctx->get().device_read_async_io(*object, 0, 100 * KiB, dst.get(), stream.ref());
      REQUIRE(std::move(late).wait(10s));
      CHECK_THROWS(std::move(late).get());
    }
  }
}

// ---------------------------------------------------------------------------
// 6. Backpressure
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: far more lane reads than lanes", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{64 * MiB};
  auto cfg                             = fused_config(2, 1 * MiB);
  cfg.fused_reads.max_lanes_per_device = 2;
  started_ioctx ioctx{cfg, 1};
  auto object = ioctx.get().open_io_object(file.path());
  stream_holder stream;
  // 500 reads with gaps between them: none fuse.
  auto const plan = strided_plan(17, 500, 40 * KiB, 128 * KiB);
  device_memory arena{plan.arena_bytes()};
  auto const before = ioctx.get().stats();
  std::atomic<bool> done{false};
  std::atomic<std::size_t> max_lanes{0};
  std::thread watcher{[&] {
    while (!done.load()) {
      auto const lanes = cucascade::io::cufile::detail::lanes_in_use_for_testing(0);
      auto seen        = max_lanes.load();
      while (lanes > seen && !max_lanes.compare_exchange_weak(seen, lanes)) {}
      std::this_thread::sleep_for(200us);
    }
  }};
  auto const got =
    ioctx.get().device_readv_async_io(*object, plan.slices(arena.get()), stream.ref()).get(60s);
  done = true;
  watcher.join();
  CHECK(got == plan.expected_bytes(file.size()));
  CHECK(plan.verify(arena.host_copy(), file));
  CHECK(max_lanes.load() <= 2);
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_reads == 500);
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// 7. Routing by role
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: a planned pool routes by role", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{16 * MiB};
  started_ioctx ioctx{fused_config(4, 1 * MiB), 0};
  auto& pool = ioctx.get();
  REQUIRE_FALSE(pool.plan().groups.empty());
  auto const& group = pool.plan().groups.front();
  CHECK(group.lane_reactors == 2);
  auto object = pool.open_io_object(file.path());
  auto const& typed = dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object);
  stream_holder stream;

  // Lane reads on GPU 0, host reads, small device reads below the threshold.
  std::mt19937_64 rng{7};
  for (int round = 0; round < 4; ++round) {
    run_and_verify(pool, *object, file, random_plan(rng, file.size(), 20, 4 * MiB, 16 * MiB),
                   stream.ref());
  }
  std::vector<std::uint8_t> host(256 * KiB);
  CHECK(pool.host_read_async_io(*object, 4096, host.size(), host.data()).get(30s) == host.size());
  CHECK(std::memcmp(host.data(), file.data() + 4096, host.size()) == 0);
  std::vector<io_slice> host_slices;
  std::vector<std::uint8_t> host_many(4 * 64 * KiB);
  for (std::size_t i = 0; i < 4; ++i) {
    host_slices.emplace_back(i * MiB + 3, 64 * KiB, host_many.data() + i * 64 * KiB);
  }
  CHECK(pool.host_readv_async_io(*object, host_slices).get(30s) == 4 * 64 * KiB);
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(std::memcmp(host_many.data() + i * 64 * KiB, file.data() + i * MiB + 3, 64 * KiB) == 0);
  }

  std::size_t lane_reads_on_lanes = 0;
  for (std::size_t i = 0; i < pool.reactor_count(); ++i) {
    auto const& reactor = pool.reactor(i);
    auto const stats    = reactor.stats();
    INFO("reactor " << i << " lanes " << reactor.placement().fused_lanes);
    if (reactor.placement().fused_lanes) {
      lane_reads_on_lanes += stats.lane_reads;
      CHECK(stats.host_reads == 0);
    } else {
      CHECK(stats.lane_reads == 0);
    }
  }
  CHECK(lane_reads_on_lanes > 0);

  // Writes and flushes go to the synchronous reactors.
  for (auto const type : {cufile_ioctx::io_op_type::write_vector_async,
                          cufile_ioctx::io_op_type::flush_async,
                          cufile_ioctx::io_op_type::host_write}) {
    for (int device : {-1, group.device}) {
      auto const chosen = pool.next_reactor(typed, 1, type, device);
      REQUIRE_FALSE(chosen.empty());
      for (auto const* reactor : chosen) {
        CHECK_FALSE(reactor->placement().fused_lanes);
      }
    }
  }
  // Host-only reads never go to lane reactors.
  for (auto const* reactor :
       pool.next_reactor(typed, 8, cufile_ioctx::io_op_type::host_vector_async, -1)) {
    CHECK_FALSE(reactor->placement().fused_lanes);
  }
  require_queued_bytes_zero(pool);
}

// ---------------------------------------------------------------------------
// 9. Configuration
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: configuration", "[io][cufile][fused_ioctx][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  SECTION("fused reads and the async engine are mutually exclusive")
  {
    auto cfg              = fused_config();
    cfg.async_engine.mode = engine_mode::async_auto;
    CHECK_THROWS_AS(
      cufile_ioctx(1, std::make_shared<cufile_reactor::reactor_context>(cfg)),
      std::invalid_argument);
  }

  SECTION("an invalid fused_reads block is refused")
  {
    auto cfg                          = fused_config();
    cfg.fused_reads.lanes_per_reactor = 0;
    CHECK_THROWS_AS(
      cufile_ioctx(1, std::make_shared<cufile_reactor::reactor_context>(cfg)),
      std::invalid_argument);
    cfg                        = fused_config();
    cfg.fused_reads.slot_bytes = 100 * KiB;
    CHECK_THROWS_AS(
      cufile_ioctx(1, std::make_shared<cufile_reactor::reactor_context>(cfg)),
      std::invalid_argument);
  }

  SECTION("fused reads off: legacy behaviour, no lane counters")
  {
    pattern_file const file{8 * MiB + 5};
    auto cfg                = fused_config();
    cfg.fused_reads.enabled = false;
    started_ioctx ioctx{cfg, 2};
    auto object = ioctx.get().open_io_object(file.path());
    stream_holder stream;
    std::mt19937_64 rng{3};
    for (int round = 0; round < 3; ++round) {
      run_and_verify(ioctx.get(), *object, file, random_plan(rng, file.size(), 20, 4 * MiB, 8 * MiB),
                     stream.ref());
    }
    auto const stats = ioctx.get().stats();
    CHECK(stats.lane_reads == 0);
    CHECK(stats.lane_bytes == 0);
    CHECK(stats.lane_failures == 0);
    CHECK(stats.lane_parts == 0);
    CHECK(stats.direct_reads + stats.staged_reads + stats.posix_device_reads > 0);
    for (std::size_t i = 0; i < ioctx.get().reactor_count(); ++i) {
      CHECK(ioctx.get().reactor(i).placement().fused_lanes);  // flat: capability, unused
    }
  }

  SECTION("stats arithmetic covers every counter")
  {
    reactor_stats a;
    for (auto const field : reactor_stats::fields()) {
      a.*field = 5;
    }
    reactor_stats b;
    b.lane_reads  = 7;
    b.batch_bytes = 2;
    auto const sum = a + b;
    CHECK(sum.lane_reads == 12);
    CHECK(sum.batch_bytes == 7);
    auto const diff = a - b;
    CHECK(diff.lane_reads == 0);  // saturates
    CHECK(diff.batch_bytes == 3);
    CHECK(diff.staged_reads == 5);
    CHECK((sum - b) == a);
  }
}

// ---------------------------------------------------------------------------
// 10. Multi-threaded stress
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: multi-threaded stress", "[io][cufile][fused_ioctx][gpu][stress]")
{
  require_async_cufile();
  pattern_file const file{(64 * MiB) + 4321};
  auto cfg                   = fused_config(6, 1 * MiB);
  cfg.fused_reads.completion = GENERATE(lane_completion::poll, lane_completion::callback);
  started_ioctx ioctx{cfg, 4};
  auto object = ioctx.get().open_io_object(file.path());
  constexpr int threads        = 4;
  constexpr int reads_per_thread = 150;
  std::atomic<std::size_t> failures{0};
  std::atomic<std::size_t> completed{0};
  auto const deadline = std::chrono::steady_clock::now() + 240s;
  auto worker         = [&](int index) {
    try {
      std::mt19937_64 rng{static_cast<std::uint64_t>(1000 + index)};
      cudaStream_t stream{};
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        ++failures;
        return;
      }
      for (int i = 0; i < reads_per_thread && std::chrono::steady_clock::now() < deadline; ++i) {
        auto const plan = random_plan(rng, file.size(), 24, 6 * MiB, 12 * MiB);
        void* raw       = nullptr;
        if (cudaMalloc(&raw, plan.arena_bytes()) != cudaSuccess) {
          ++failures;
          break;
        }
        auto* arena = static_cast<std::uint8_t*>(raw);
        std::ignore = cudaMemset(arena, guard_fill, plan.arena_bytes());
        std::ignore = cudaDeviceSynchronize();
        auto const priority = (i % 3 == 0) ? io_priority::low : io_priority::high;
        auto future         = ioctx.get().device_readv_async_io(
          *object, plan.slices(arena), ::cuda::stream_ref{stream}, priority);
        if (!std::move(future).wait(60s)) {
          ++failures;  // leak the arena rather than free memory a read may still write
          break;
        }
        auto const got = std::move(future).get();
        std::vector<std::uint8_t> host(plan.arena_bytes());
        std::ignore = cudaMemcpy(host.data(), arena, host.size(), cudaMemcpyDeviceToHost);
        if (got != plan.expected_bytes(file.size()) || !plan.verify(host, file)) ++failures;
        std::ignore = cudaFree(arena);
        ++completed;
      }
      std::ignore = cudaStreamDestroy(stream);
    } catch (...) {
      ++failures;
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back(worker, t);
  }
  for (auto& thread : pool) {
    thread.join();
  }
  CHECK(failures.load() == 0);
  CHECK(completed.load() == static_cast<std::size_t>(threads * reads_per_thread));
  auto const stats = ioctx.get().stats();
  CHECK(stats.lane_reads > 0);
  CHECK(stats.lane_failures == 0);
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// 11. Default-stream callers
// ---------------------------------------------------------------------------

TEST_CASE("fused ioctx: per-thread and legacy default stream callers",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{16 * MiB + 9};
  started_ioctx ioctx{fused_config(4, 1 * MiB), 2};
  auto object = ioctx.get().open_io_object(file.path());
  auto const stream = GENERATE(cudaStreamPerThread, static_cast<cudaStream_t>(nullptr));
  CAPTURE(stream == nullptr);
  std::mt19937_64 rng{11};
  auto const before = ioctx.get().stats();
  for (int round = 0; round < 4; ++round) {
    run_and_verify(ioctx.get(), *object, file, random_plan(rng, file.size(), 20, 4 * MiB, 16 * MiB),
                   ::cuda::stream_ref{stream});
  }
  CHECK((ioctx.get().stats() - before).lane_reads > 0);
  require_queued_bytes_zero(ioctx.get());
}

// ---------------------------------------------------------------------------
// Review round: two-phase gates, backlog, quotas, routing by usability, mixed requests.
// ---------------------------------------------------------------------------

namespace {

/// A hand-built pool for GPU 0: @p lane_reactors lane reactors then @p sync_reactors
/// synchronous ones, with a lane quota of @p lane_slots.  Started; shut down on scope exit.
class manual_pool {
 public:
  manual_pool(config const& cfg,
              std::size_t lane_reactors,
              std::size_t sync_reactors,
              std::size_t lane_slots)
  {
    cucascade::io::cufile::pool_plan plan;
    cucascade::io::cufile::device_group_plan group;
    group.device            = 0;
    group.n_reactors        = sync_reactors;
    group.lane_reactors     = lane_reactors;
    group.lanes_per_reactor = cfg.fused_reads.lanes_per_reactor;
    group.lane_slots        = lane_slots;
    plan.groups.push_back(group);
    plan.lane_slot_bytes = cfg.fused_reads.slot_bytes;
    auto ctx             = std::make_shared<cufile_reactor::reactor_context>(
      cfg, std::vector<std::size_t>{0}, std::vector<std::size_t>{lane_slots});
    _ioctx = std::make_shared<cufile_ioctx>(std::move(plan), std::move(ctx));
    _ioctx->start();
  }
  ~manual_pool() { _ioctx->shutdown(); }
  manual_pool(manual_pool const&)            = delete;
  manual_pool& operator=(manual_pool const&) = delete;
  [[nodiscard]] cufile_ioctx& get() const noexcept { return *_ioctx; }

 private:
  std::shared_ptr<cufile_ioctx> _ioctx;
};

[[nodiscard]] cucascade::io::prepared_io_slice device_piece(std::size_t offset,
                                                            std::size_t size,
                                                            std::uint8_t* dst,
                                                            cudaStream_t stream)
{
  return cucascade::io::prepared_io_slice{
    cucascade::io::range{offset, size},
    cucascade::io::device_buffer{dst, ::cuda::stream_ref{stream}}};
}

[[nodiscard]] bool all_fill(std::vector<std::uint8_t> const& bytes)
{
  return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == guard_fill; });
}

/// @p stats without the worker's timing and pass counters, which move even when idle.
[[nodiscard]] reactor_stats traffic(reactor_stats stats)
{
  stats.lane_submit_ns     = 0;
  stats.lane_read_async_ns = 0;
  stats.lane_flight_ns     = 0;
  stats.lane_notify_ns     = 0;
  stats.lane_idle_ns       = 0;
  stats.worker_wait_ns     = 0;
  stats.worker_passes      = 0;
  return stats;
}

/// Disarms the lane failure seams on scope exit.
struct seam_reset {
  seam_reset()                             = default;
  seam_reset(seam_reset const&)            = delete;
  seam_reset& operator=(seam_reset const&) = delete;
  ~seam_reset()
  {
    cucascade::io::cufile::detail::set_lane_callback_failure_for_testing(0);
    cucascade::io::cufile::detail::set_lane_copy_failure_for_testing(0);
    cucascade::io::cufile::detail::set_lane_stream_create_failure_for_testing(0);
  }
};

}  // namespace

TEST_CASE("fused ioctx: shutdown with a gated read on a held stream settles promptly",
          "[io][cufile][fused_ioctx][gpu][shutdown]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  stream_holder busy;
  started_ioctx ioctx{fused_config(4, 1 * MiB), 1};
  auto object = ioctx.get().open_io_object(file.path());
  warm_up(ioctx.get(), *object, file);
  read_plan plan;
  plan.add(4097, 300 * KiB);
  device_memory arena{plan.arena_bytes()};
  stream_latch latch;
  latch.hold(busy.get());
  auto future = ioctx.get().device_readv_async_io(*object, plan.slices(arena.get()), busy.ref());
  std::this_thread::sleep_for(300ms);  // its read is done: held for the gate
  auto const started = std::chrono::steady_clock::now();
  ioctx.get().shutdown();
  REQUIRE(std::move(future).wait(5s));
  CHECK(std::chrono::steady_clock::now() - started < 5s);
  CHECK_THROWS(std::move(future).get());
  CHECK(all_fill(arena.host_copy()));  // its copies were never queued
  latch.release();
}

TEST_CASE("fused ioctx: a sibling failure cancels a held read",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  stream_holder lane_stream;
  stream_holder sync_stream;
  auto cfg          = fused_config(4, 1 * MiB);
  cfg.gds_threshold = 64 * KiB;
  manual_pool pool{cfg, 1, 1, 4};
  auto object = pool.get().open_io_object(file.path());
  warm_up(pool.get(), *object, file);
  std::filesystem::resize_file(file.path(), 2 * MiB);  // the small read below now fails
  device_memory lane_dst{300 * KiB};
  device_memory small_dst{8 * KiB};
  stream_latch lane_latch;
  stream_latch sync_latch;
  lane_latch.hold(lane_stream.get());
  sync_latch.hold(sync_stream.get());
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(device_piece(4096 + 3, 200 * KiB, lane_dst.get(), lane_stream.get()));
  slices.push_back(device_piece(6 * MiB + 5, 8 * KiB, small_dst.get(), sync_stream.get()));
  auto future = pool.get().mixed_readv_async_io(*object, std::move(slices));
  std::this_thread::sleep_for(300ms);  // the lane read is held, the small one parked
  CHECK_FALSE(future.is_ready());
  sync_latch.release();  // the small read runs and fails: the request stops
  REQUIRE(std::move(future).wait(10s));  // while the lane read's stream is still held
  CHECK_THROWS(std::move(future).get());
  CHECK(all_fill(lane_dst.host_copy()));
  lane_latch.release();
  require_queued_bytes_zero(pool.get());
}

TEST_CASE("fused ioctx: a refused gated read falls back without blocking the reactor",
          "[io][cufile][fused_ioctx][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile;
  pattern_file const file{8 * MiB};
  stream_holder busy;
  stream_holder idle;
  started_ioctx ioctx{fused_config(4, 1 * MiB), 1};
  auto object = ioctx.get().open_io_object(file.path());
  read_plan gated;
  gated.add(12345, 300 * KiB);
  read_plan other;
  other.add(5 * MiB + 1, 200 * KiB);
  device_memory gated_arena{gated.arena_bytes()};
  device_memory other_arena{other.arena_bytes()};
  // The lane set exists before the stream is held: creating it may allocate device memory,
  // which can synchronize the device.
  warm_up(ioctx.get(), *object, file);
  cufile.script({fake_action{fake_action::kind::refuse, 0}});
  auto const before = ioctx.get().stats();
  stream_latch latch;
  latch.hold(busy.get());
  auto future =
    ioctx.get().device_readv_async_io(*object, gated.slices(gated_arena.get()), busy.ref());
  // The same reactor serves another read while the failed one waits for its gate.
  CHECK(ioctx.get()
          .device_readv_async_io(*object, other.slices(other_arena.get()), idle.ref())
          .get(10s) == 200 * KiB);
  CHECK(other.verify(other_arena.host_copy(), file));
  CHECK_FALSE(future.is_ready());
  CHECK(all_fill(gated_arena.host_copy()));
  latch.release();
  CHECK(std::move(future).get(30s) == 300 * KiB);
  CHECK(gated.verify(gated_arena.host_copy(), file));
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_failures == 1);
  CHECK(delta.lane_reads == 1);
  require_queued_bytes_zero(ioctx.get());
}

TEST_CASE("fused ioctx: cancel while a read sits in the lane backlog",
          "[io][cufile][fused_ioctx][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile;
  pattern_file const file{8 * MiB};
  stream_holder idle;
  stream_holder held;
  auto cfg                          = fused_config(1, 1 * MiB);
  cfg.gds_threshold                 = 64 * KiB;
  manual_pool pool{cfg, 1, 1, 1};
  auto object = pool.get().open_io_object(file.path());
  read_plan first;
  first.add(5, 200 * KiB);
  device_memory first_arena{first.arena_bytes()};
  device_memory backlog_dst{200 * KiB};
  device_memory small_dst{8 * KiB};
  stream_latch lane_latch;  // the one lane's read waits on it
  stream_latch sync_latch;
  cufile.latch_with(lane_latch);
  cufile.script({fake_action{fake_action::kind::latched, 0}});
  auto busy_lane =
    pool.get().device_readv_async_io(*object, first.slices(first_arena.get()), idle.ref());
  std::this_thread::sleep_for(100ms);
  sync_latch.hold(held.get());
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(device_piece(1 * MiB + 5, 200 * KiB, backlog_dst.get(), idle.get()));
  slices.push_back(device_piece(6 * MiB + 5, 8 * KiB, small_dst.get(), held.get()));
  auto future = pool.get().mixed_readv_async_io(*object, std::move(slices));
  std::this_thread::sleep_for(200ms);  // the lane read waits in the backlog
  std::filesystem::resize_file(file.path(), 2 * MiB);
  sync_latch.release();  // the small read fails: the request stops
  REQUIRE(std::move(future).wait(10s));  // the lane is still busy: cancelled from the backlog
  CHECK_THROWS(std::move(future).get());
  CHECK_FALSE(busy_lane.is_ready());
  CHECK(all_fill(backlog_dst.host_copy()));
  lane_latch.release();
  CHECK(std::move(busy_lane).get(30s) == 200 * KiB);
  CHECK(first.verify(first_arena.host_copy(), file));
  require_queued_bytes_zero(pool.get());
}

TEST_CASE("fused ioctx: a backlog whose lanes are all lost is served synchronously",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  seam_reset const reset;
  pattern_file const file{8 * MiB};
  stream_holder stream;
  started_ioctx ioctx{fused_config(1, 1 * MiB), 1};
  auto object = ioctx.get().open_io_object(file.path());
  // Warm up: the lane set exists before the seams are armed.
  run_and_verify(ioctx.get(), *object, file, strided_plan(0, 1, 100 * KiB, 0), stream.ref());
  CHECK(ioctx.get().reactor(0).lanes_usable(0));
  // The next read's completion cannot be installed and its lane's new stream cannot be
  // created: the set loses its only lane while the second read waits in the backlog.
  cucascade::io::cufile::detail::set_lane_callback_failure_for_testing(1);
  cucascade::io::cufile::detail::set_lane_stream_create_failure_for_testing(1);
  auto const before = ioctx.get().stats();
  run_and_verify(
    ioctx.get(), *object, file, strided_plan(1 * MiB + 3, 2, 100 * KiB, 2 * MiB), stream.ref());
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_failures == 1);
  CHECK(delta.lane_reads == 0);
  CHECK_FALSE(ioctx.get().reactor(0).lanes_usable(0));
  // Later reads take the synchronous paths.
  run_and_verify(ioctx.get(), *object, file, strided_plan(7, 3, 50 * KiB, MiB), stream.ref());
  require_queued_bytes_zero(ioctx.get());
}

TEST_CASE("fused ioctx: device_failed through the ioctx falls back",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  seam_reset const reset;
  pattern_file const file{8 * MiB};
  stream_holder stream;
  started_ioctx ioctx{fused_config(2, 1 * MiB), 1};
  auto object = ioctx.get().open_io_object(file.path());
  cucascade::io::cufile::detail::set_lane_callback_failure_for_testing(1);
  auto const before = ioctx.get().stats();
  run_and_verify(ioctx.get(), *object, file, strided_plan(99, 1, 300 * KiB, 0), stream.ref());
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_failures == 1);
  CHECK(delta.cufile_failures >= 1);
  require_queued_bytes_zero(ioctx.get());
}

TEST_CASE("fused ioctx: the plan's lane quota bounds the lanes", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{32 * MiB};
  stream_holder stream;
  auto cfg = fused_config(4, 1 * MiB);
  REQUIRE(cucascade::io::cufile::detail::lanes_in_use_for_testing(0) == 0);
  manual_pool pool{cfg, 2, 1, 3};  // two lane reactors of 4 lanes, but 3 lanes planned
  auto object = pool.get().open_io_object(file.path());
  std::atomic<bool> done{false};
  std::atomic<std::size_t> max_lanes{0};
  std::thread watcher{[&] {
    while (!done.load()) {
      auto const lanes = cucascade::io::cufile::detail::lanes_in_use_for_testing(0);
      auto seen        = max_lanes.load();
      while (lanes > seen && !max_lanes.compare_exchange_weak(seen, lanes)) {}
      std::this_thread::sleep_for(200us);
    }
  }};
  std::mt19937_64 rng{41};
  for (int round = 0; round < 6; ++round) {
    run_and_verify(
      pool.get(), *object, file, random_plan(rng, file.size(), 24, 4 * MiB, 16 * MiB), stream.ref());
  }
  done = true;
  watcher.join();
  CHECK(max_lanes.load() >= 1);
  CHECK(max_lanes.load() <= 3);
  CHECK(pool.get().stats().lane_reads > 0);
  // The synchronous staging still has room: a slot of its size can be leased.
  auto& staging   = cucascade::io::cufile::detail::staging_pool::instance();
  auto const slot = staging.acquire(0, cfg.staging_slot_bytes(), 0, false);
  CHECK(static_cast<bool>(slot));
  if (slot) staging.release(slot);
}

TEST_CASE("fused ioctx: a cooling lane reactor is routed around",
          "[io][cufile][fused_ioctx][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile;
  pattern_file const file{16 * MiB};
  stream_holder stream;
  auto cfg                      = fused_config(4, 1 * MiB);
  cfg.fused_reads.failure_limit = 1;
  cfg.fused_reads.cooldown_s    = 60;
  manual_pool pool{cfg, 2, 2, 8};
  auto object = pool.get().open_io_object(file.path());
  cufile.every(fake_action{fake_action::kind::refuse, 0});
  // Eight separate lane reads fan out over both lane reactors; each fails once and cools.
  run_and_verify(pool.get(), *object, file, strided_plan(5, 8, 100 * KiB, MiB), stream.ref());
  CHECK_FALSE(pool.get().reactor(0).lanes_usable(0));
  CHECK_FALSE(pool.get().reactor(1).lanes_usable(0));

  std::vector<reactor_stats> before;
  for (std::size_t i = 0; i < pool.get().reactor_count(); ++i) {
    before.push_back(pool.get().reactor(i).stats());
  }
  auto const calls = cufile.async_calls();
  std::mt19937_64 rng{5};
  for (int round = 0; round < 3; ++round) {
    run_and_verify(
      pool.get(), *object, file, random_plan(rng, file.size(), 16, 4 * MiB, 8 * MiB), stream.ref());
  }
  CHECK(cufile.async_calls() == calls);
  for (std::size_t i = 0; i < 2; ++i) {
    // Lane reactors served nothing (their workers still pass and wait).
    CHECK(traffic(pool.get().reactor(i).stats()) == traffic(before[i]));
  }
  reactor_stats sync_delta;
  for (std::size_t i = 2; i < pool.get().reactor_count(); ++i) {
    sync_delta += pool.get().reactor(i).stats() - before[i];
  }
  CHECK(sync_delta.direct_reads + sync_delta.staged_reads + sync_delta.posix_device_reads > 0);
  require_queued_bytes_zero(pool.get());
}

TEST_CASE("fused ioctx: an ioctx whose GPU lane budget is spent serves through the legacy path",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{16 * MiB};
  stream_holder stream;
  auto cfg                             = fused_config(4, 1 * MiB);
  cfg.fused_reads.max_lanes_per_device = 4;
  started_ioctx first{cfg, 1};
  auto object = first.get().open_io_object(file.path());
  run_and_verify(first.get(), *object, file, strided_plan(1, 1, 100 * KiB, 0), stream.ref());
  REQUIRE(cucascade::io::cufile::detail::lanes_in_use_for_testing(0) == 4);

  started_ioctx second{cfg, 2};
  auto second_object = second.get().open_io_object(file.path());
  // The first request finds no lanes on either reactor; afterwards neither is a lane
  // candidate and requests take the legacy path.
  run_and_verify(second.get(), *second_object, file, strided_plan(3, 6, 100 * KiB, MiB),
                 stream.ref());
  CHECK_FALSE(second.get().reactor(0).lanes_usable(0));
  CHECK_FALSE(second.get().reactor(1).lanes_usable(0));
  std::mt19937_64 rng{9};
  for (int round = 0; round < 3; ++round) {
    run_and_verify(second.get(), *second_object, file,
                   random_plan(rng, file.size(), 16, 4 * MiB, 8 * MiB), stream.ref());
  }
  CHECK(second.get().stats().lane_reads == 0);
  require_queued_bytes_zero(second.get());
}

TEST_CASE("fused ioctx: a flat pool with more reactors than the lane budget",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{32 * MiB};
  stream_holder stream;
  auto cfg                             = fused_config(2, 1 * MiB);
  cfg.fused_reads.max_lanes_per_device = 4;
  started_ioctx ioctx{cfg, 4};
  auto object = ioctx.get().open_io_object(file.path());
  // Every reactor gets a lane read of the first request; two of them find lanes.
  run_and_verify(ioctx.get(), *object, file, strided_plan(9, 8, 100 * KiB, MiB), stream.ref());
  std::size_t usable = 0;
  for (std::size_t i = 0; i < ioctx.get().reactor_count(); ++i) {
    if (ioctx.get().reactor(i).lanes_usable(0)) ++usable;
  }
  CHECK(usable == 2);
  CHECK(cucascade::io::cufile::detail::lanes_in_use_for_testing(0) == 4);
  auto const before = ioctx.get().stats();
  std::mt19937_64 rng{13};
  for (int round = 0; round < 4; ++round) {
    run_and_verify(
      ioctx.get(), *object, file, random_plan(rng, file.size(), 16, 4 * MiB, 8 * MiB), stream.ref());
  }
  CHECK((ioctx.get().stats() - before).lane_reads > 0);
  require_queued_bytes_zero(ioctx.get());
}

TEST_CASE("fused ioctx: a mixed request keeps its synchronous slices off the lane reactors",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  constexpr std::size_t size = (3 * MiB) + 1000;
  pattern_file const file{size};
  stream_holder stream;
  auto cfg          = fused_config(4, 1 * MiB);
  cfg.gds_threshold = 64 * KiB;
  manual_pool pool{cfg, 1, 1, 4};
  auto object = pool.get().open_io_object(file.path());
  read_plan plan;
  plan.add(4096 + 1, 300 * KiB);      // lane read
  plan.add(1 * MiB + 5, 400 * KiB);   // lane read
  plan.add(3 * MiB + 10, 500);        // EOF tail: partial last block
  plan.add(2 * MiB + 3, 8 * KiB);     // below gds_threshold
  run_and_verify(pool.get(), *object, file, plan, stream.ref());
  auto const lane = pool.get().reactor(0).stats();
  auto const sync = pool.get().reactor(1).stats();
  REQUIRE(pool.get().reactor(0).placement().fused_lanes);
  CHECK(lane.lane_reads == 2);
  CHECK(lane.posix_device_reads + lane.direct_reads + lane.staged_reads == 0);
  CHECK(sync.lane_reads == 0);
  CHECK(sync.posix_device_reads + sync.direct_reads + sync.staged_reads == 2);
  require_queued_bytes_zero(pool.get());
}

TEST_CASE("fused ioctx: contiguous slices into one buffer cost one copy per window",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{16 * MiB};
  stream_holder stream;
  started_ioctx ioctx{fused_config(4, 1 * MiB), 1};
  auto object = ioctx.get().open_io_object(file.path());
  read_plan plan;
  for (std::size_t i = 0; i < 48; ++i) {
    plan.add(5 + i * 64 * KiB, 64 * KiB, /*guard=*/0);
  }
  auto const before        = ioctx.get().stats();
  auto const copies_before = cucascade::io::cufile::detail::lane_copy_calls_for_testing();
  run_and_verify(ioctx.get(), *object, file, plan, stream.ref());
  auto const delta  = ioctx.get().stats() - before;
  auto const copies = cucascade::io::cufile::detail::lane_copy_calls_for_testing() - copies_before;
  CHECK(delta.lane_reads >= 3);
  CHECK(copies == delta.lane_reads);
  CHECK(delta.lane_parts >= 48);
}



TEST_CASE("fused ioctx: lane timing counters", "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{16 * MiB};
  stream_holder stream;
  auto const completion      = GENERATE(lane_completion::callback, lane_completion::poll);
  auto cfg                   = fused_config(4, 1 * MiB);
  cfg.fused_reads.completion = completion;
  started_ioctx ioctx{cfg, 1};
  auto object       = ioctx.get().open_io_object(file.path());
  auto const before = ioctx.get().stats();
  run_and_verify(ioctx.get(), *object, file, strided_plan(7, 12, 300 * KiB, MiB), stream.ref());
  auto const delta = ioctx.get().stats() - before;
  CHECK(delta.lane_reads == 12);
  CHECK(delta.lane_submit_ns > 0);
  CHECK(delta.lane_read_async_ns > 0);
  CHECK(delta.lane_read_async_ns <= delta.lane_submit_ns);
  CHECK(delta.lane_flight_ns > 0);
  if (completion == lane_completion::callback) {
    CHECK(delta.lane_notify_ns > 0);
  } else {
    CHECK(delta.lane_notify_ns == 0);  // no callback time to measure from
  }
  CHECK(delta.lane_notify_ns <= delta.lane_flight_ns);
  CHECK(delta.lane_idle_ns <= delta.lane_notify_ns);
  CHECK(delta.worker_passes > 0);
  CHECK(delta.worker_wait_ns > 0);
}

TEST_CASE("fused ioctx: a GPU the plan gave no lanes is never planned for lanes",
          "[io][cufile][fused_ioctx][gpu]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  stream_holder stream;
  manual_pool pool{fused_config(4, 1 * MiB), 1, 1, 0};  // a lane reactor, but 0 lane slots
  CHECK_FALSE(pool.get().reactor(0).lanes_usable(0));    // from the start
  auto object = pool.get().open_io_object(file.path());
  std::mt19937_64 rng{17};
  for (int round = 0; round < 3; ++round) {
    run_and_verify(
      pool.get(), *object, file, random_plan(rng, file.size(), 16, 2 * MiB, 6 * MiB), stream.ref());
  }
  auto const stats = pool.get().stats();
  CHECK(stats.lane_reads == 0);
  CHECK(stats.fused_slices == 0);  // the legacy planner did not fuse for lanes either
  require_queued_bytes_zero(pool.get());
}
