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
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::io_context_type;
using cucascade::io::io_object;
using cucascade::io::io_priority;
using cucascade::io::slice;
using cucascade::io::write_mode;
using cucascade::io::write_open_options;
using cucascade::io::write_slice;
using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;

constexpr std::size_t KiB = 1UL << 10;
constexpr std::size_t MiB = 1UL << 20;

/// Two read and two write lanes of 1 MiB on GPU 0: a few MiB of I/O crosses lanes and slots.
[[nodiscard]] config small_config()
{
  config cfg{};
  cfg.lanes      = 2;
  cfg.lane_bytes = MiB;
  cfg.devices    = {0};
  return cfg;
}

[[nodiscard]] bool cufile_usable()
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::ignore = cudaGetLastError();
    return false;
  }
  return cufile_api::get() != nullptr && cufile_api::open_driver();
}

[[nodiscard]] std::vector<std::uint8_t> pattern(std::size_t bytes, std::size_t seed)
{
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::uint8_t>(((i + seed * 7919U) * 2654435761ULL) >> 13);
  }
  return out;
}

[[nodiscard]] std::vector<std::uint8_t> sub(std::vector<std::uint8_t> const& bytes,
                                            std::size_t offset,
                                            std::size_t size)
{
  auto const first = bytes.begin() + static_cast<std::ptrdiff_t>(offset);
  return {first, first + static_cast<std::ptrdiff_t>(size)};
}

/// A scratch file holding @p contents, removed on scope exit.
class scratch_file {
 public:
  explicit scratch_file(std::vector<std::uint8_t> const& contents = {})
    : path((std::filesystem::temp_directory_path() /
            ("cucascade_cufile_" + std::to_string(::getpid()) + "_" +
             std::to_string(next.fetch_add(1)) + ".bin"))
             .string())
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(contents.data()),
              static_cast<std::streamsize>(contents.size()));
    REQUIRE(out);
  }
  ~scratch_file()
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  [[nodiscard]] std::vector<std::uint8_t> contents() const
  {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }

  std::string path;

 private:
  static inline std::atomic<int> next{0};
};

/// Device memory of the current GPU and the stream its work is queued on.
class device_bytes {
 public:
  explicit device_bytes(std::size_t bytes) : size(bytes)
  {
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&data), bytes) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(data, 0xEE, bytes, raw_stream) == cudaSuccess);
  }
  explicit device_bytes(std::vector<std::uint8_t> const& contents) : device_bytes(contents.size())
  {
    // Left queued: a write must order itself after it.
    REQUIRE(
      cudaMemcpyAsync(data, contents.data(), contents.size(), cudaMemcpyHostToDevice, raw_stream) ==
      cudaSuccess);
  }
  ~device_bytes()
  {
    std::ignore = cudaStreamSynchronize(raw_stream);
    std::ignore = cudaStreamDestroy(raw_stream);
    std::ignore = cudaFree(data);
  }
  device_bytes(device_bytes const&)            = delete;
  device_bytes& operator=(device_bytes const&) = delete;

  [[nodiscard]] ::cuda::stream_ref stream() const { return ::cuda::stream_ref{raw_stream}; }

  [[nodiscard]] std::vector<std::uint8_t> download() const
  {
    std::vector<std::uint8_t> out(size);
    REQUIRE(cudaMemcpy(out.data(), data, size, cudaMemcpyDeviceToHost) == cudaSuccess);
    return out;
  }

  std::size_t size;
  std::uint8_t* data{};
  cudaStream_t raw_stream{};
};

/// A started context; shutdown drains even on assertion failure.
class started_ioctx {
 public:
  explicit started_ioctx(config const& cfg = small_config()) : context(cfg) { context.start(); }
  ~started_ioctx() { context.shutdown(); }
  cufile_ioctx context;
};

[[nodiscard]] bool registered(io_object const& object)
{
  return dynamic_cast<cufile_io_object const&>(object).cufile_registered();
}

[[nodiscard]] CUfileError_t cufile_status(CUfileOpError error)
{
  CUfileError_t status{};
  status.err = error;
  return status;
}

// What the substituted cuFile calls below do, and how often they ran.
std::atomic<int> read_calls{0};
std::atomic<int> write_calls{0};
std::atomic<int> buffer_registrations{0};  // cuFileBufRegister calls, failed ones included
std::atomic<int> reads_to_fail{0};
std::atomic<std::size_t> transfer_cap{0};  // 0 = uncapped
std::atomic<bool> hold_reads{false};       // a read waits inside cuFileRead while set
std::mutex read_order_mutex;
std::vector<std::size_t> read_order;  // the file offset of every cuFileRead, in call order

/// Which cuFileBufRegister calls fail.
enum class failing_registrations { none, every_second, all };
std::atomic<failing_registrations> fail_registrations{failing_registrations::none};
/// The buffers registered (base, bytes) and not deregistered since.
std::mutex registered_mutex;
std::vector<std::pair<void const*, std::size_t>> registered_buffers;
/// cuFile reads and writes into a buffer no registration starts at...
std::atomic<int> unregistered_transfers{0};
/// ...and those with an offset cuFile does not take: a nonzero one into an unregistered buffer,
/// or one past the end of the registration.
std::atomic<int> misaddressed_transfers{0};
/// A cuFileRead reaching past this many bytes of the file fails (0 = no limit), as some
/// nvidia-fs builds would crash.
std::atomic<std::size_t> file_limit{0};
std::atomic<int> reads_past_eof{0};
/// cuFileBufRegister fails once the buffers registered would exceed this many bytes (0 = no
/// limit), as cuFile's pinned-memory limit makes it.
std::atomic<std::size_t> pin_budget{0};
/// cuFileBufRegister calls made while the buffer's GPU was not the current device.
std::atomic<int> registrations_off_device{0};

void check_address(void const* base, std::size_t size, off_t base_offset)
{
  std::lock_guard const lock(registered_mutex);
  auto const it = std::ranges::find_if(registered_buffers,
                                       [base](auto const& buffer) { return buffer.first == base; });
  if (it == registered_buffers.end()) {
    ++unregistered_transfers;
    if (base_offset != 0) { ++misaddressed_transfers; }
  } else if (static_cast<std::size_t>(base_offset) + size > it->second) {
    ++misaddressed_transfers;
  }
}

/// libcufile's table with counting, fault-injecting reads and writes, installed for a scope.
/// Declare it before the context and the objects that use it.
class substituted_api {
 public:
  substituted_api()
  {
    table             = *cufile_api::loaded();
    table.driver_open = []() -> CUfileError_t { return cufile_status(CU_FILE_SUCCESS); };
    table.read        = [](CUfileHandle_t handle,
                    void* base,
                    std::size_t size,
                    off_t file_offset,
                    off_t base_offset) -> ssize_t {
      ++read_calls;
      {
        std::lock_guard const lock(read_order_mutex);
        read_order.push_back(static_cast<std::size_t>(file_offset));
      }
      check_address(base, size, base_offset);
      while (hold_reads.load()) {
        std::this_thread::sleep_for(1ms);
      }
      if (auto const limit = file_limit.load();
          limit != 0 && static_cast<std::size_t>(file_offset) + size > limit) {
        ++reads_past_eof;
        errno = EINVAL;
        return -1;
      }
      if (reads_to_fail.load() > 0) {
        --reads_to_fail;
        errno = EIO;
        return -1;
      }
      auto const cap = transfer_cap.load();
      return cufile_api::loaded()->read(
        handle, base, cap == 0 ? size : std::min(size, cap), file_offset, base_offset);
    };
    table.write = [](CUfileHandle_t handle,
                     void const* base,
                     std::size_t size,
                     off_t file_offset,
                     off_t base_offset) -> ssize_t {
      ++write_calls;
      check_address(base, size, base_offset);
      auto const cap = transfer_cap.load();
      return cufile_api::loaded()->write(
        handle, base, cap == 0 ? size : std::min(size, cap), file_offset, base_offset);
    };
    table.buf_register = [](void const* base, std::size_t size, int flags) -> CUfileError_t {
      auto const call = ++buffer_registrations;
      {
        int current = -1;
        cudaPointerAttributes attributes{};
        if (cudaGetDevice(&current) != cudaSuccess ||
            cudaPointerGetAttributes(&attributes, base) != cudaSuccess ||
            attributes.device != current) {
          ++registrations_off_device;
        }
      }
      auto const fail = fail_registrations.load();
      if (fail == failing_registrations::all ||
          (fail == failing_registrations::every_second && call % 2 == 0)) {
        return cufile_status(CU_FILE_GPU_MEMORY_PINNING_FAILED);
      }
      std::lock_guard const lock(registered_mutex);
      if (auto const budget = pin_budget.load(); budget != 0) {
        std::size_t pinned = 0;
        for (auto const& buffer : registered_buffers) {
          pinned += buffer.second;
        }
        if (pinned + size > budget) { return cufile_status(CU_FILE_GPU_MEMORY_PINNING_FAILED); }
      }
      auto const status = cufile_api::loaded()->buf_register(base, size, flags);
      if (status.err == CU_FILE_SUCCESS) { registered_buffers.emplace_back(base, size); }
      return status;
    };
    table.buf_deregister = [](void const* base) -> CUfileError_t {
      {
        std::lock_guard const lock(registered_mutex);
        std::erase_if(registered_buffers,
                      [base](auto const& buffer) { return buffer.first == base; });
      }
      return cufile_api::loaded()->buf_deregister(base);
    };
    read_calls               = 0;
    write_calls              = 0;
    buffer_registrations     = 0;
    reads_to_fail            = 0;
    transfer_cap             = 0;
    hold_reads               = false;
    fail_registrations       = failing_registrations::none;
    unregistered_transfers   = 0;
    misaddressed_transfers   = 0;
    file_limit               = 0;
    reads_past_eof           = 0;
    pin_budget               = 0;
    registrations_off_device = 0;
    read_order.clear();
    {
      // Arenas an earlier scope registered were released through another table.
      std::lock_guard const lock(registered_mutex);
      registered_buffers.clear();
    }
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&table);
  }
  ~substituted_api() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  substituted_api(substituted_api const&)            = delete;
  substituted_api& operator=(substituted_api const&) = delete;

  cufile_api table;
};

/// No arena is kept for any GPU during the scope, nor after it: the case's reactors allocate and
/// register their own, and one an earlier case left behind -- with more slots, or registered
/// under another substituted table -- cannot stand in for it.  Declare it before the context.
class fresh_arenas {
 public:
  fresh_arenas() { cucascade::io::cufile::detail::release_arenas_for_testing(); }
  ~fresh_arenas() { cucascade::io::cufile::detail::release_arenas_for_testing(); }
  fresh_arenas(fresh_arenas const&)            = delete;
  fresh_arenas& operator=(fresh_arenas const&) = delete;
};

/// Whether @p future resolves within @p limit.
[[nodiscard]] bool ready_within(cucascade::exec::semi_future<std::size_t> const& future,
                                std::chrono::milliseconds limit)
{
  auto const deadline = std::chrono::steady_clock::now() + limit;
  while (!future.is_ready() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return future.is_ready();
}

std::atomic<bool> stream_held{false};

/// Holds @p stream: work queued on it after the hold waits until release() or the end of the
/// scope.  Declare it after everything that synchronizes the stream on destruction.
class stream_hold {
 public:
  explicit stream_hold(cudaStream_t stream)
  {
    stream_held = true;
    REQUIRE(cudaLaunchHostFunc(
              stream,
              [](void*) {
                while (stream_held.load()) {
                  std::this_thread::sleep_for(1ms);
                }
              },
              nullptr) == cudaSuccess);
  }
  ~stream_hold() { release(); }
  stream_hold(stream_hold const&)            = delete;
  stream_hold& operator=(stream_hold const&) = delete;

  void release() { stream_held = false; }
};

/// Every query of a copy-done event of the scope reports @p event_error, and every query or
/// synchronization of a copy stream @p stream_error.
class failing_copy_queries {
 public:
  explicit failing_copy_queries(cudaError_t event_error, cudaError_t stream_error = cudaSuccess)
  {
    cucascade::io::cufile::detail::fail_copy_queries_for_testing(event_error, stream_error);
  }
  ~failing_copy_queries() { cucascade::io::cufile::detail::fail_copy_queries_for_testing(0); }
  failing_copy_queries(failing_copy_queries const&)            = delete;
  failing_copy_queries& operator=(failing_copy_queries const&) = delete;
};

/// Every reactor worker started in the scope fails to select its GPU with @p status.
class failing_worker_setup {
 public:
  explicit failing_worker_setup(cudaError_t status)
  {
    cucascade::io::cufile::detail::fail_worker_setup_for_testing(status);
  }
  ~failing_worker_setup() { cucascade::io::cufile::detail::fail_worker_setup_for_testing(0); }
  failing_worker_setup(failing_worker_setup const&)            = delete;
  failing_worker_setup& operator=(failing_worker_setup const&) = delete;
};

}  // namespace

TEST_CASE("GDS serves a filesystem only when the driver reports its storage class", "[io][cufile]")
{
  using cucascade::io::cufile::gds_supports_filesystem;
  auto const flag = [](CUfileDriverStatusFlags_t storage_class) { return 1U << storage_class; };

  // Local filesystems count through the NVMe class, and only on NVMe drives.
  CHECK(gds_supports_filesystem(flag(CU_FILE_NVME_SUPPORTED), "ext4", /*on_nvme=*/true));
  CHECK(gds_supports_filesystem(flag(CU_FILE_NVME_SUPPORTED), "xfs", /*on_nvme=*/true));
  CHECK_FALSE(gds_supports_filesystem(flag(CU_FILE_NVME_SUPPORTED), "ext4", /*on_nvme=*/false));
  CHECK_FALSE(gds_supports_filesystem(flag(CU_FILE_GPFS_SUPPORTED), "ext4", /*on_nvme=*/true));

  // Distributed filesystems each have a class of their own.
  CHECK(gds_supports_filesystem(flag(CU_FILE_GPFS_SUPPORTED), "gpfs", /*on_nvme=*/false));
  CHECK(gds_supports_filesystem(flag(CU_FILE_LUSTRE_SUPPORTED), "lustre", /*on_nvme=*/false));
  CHECK(gds_supports_filesystem(flag(CU_FILE_WEKAFS_SUPPORTED), "wekafs", /*on_nvme=*/false));
  CHECK(gds_supports_filesystem(flag(CU_FILE_BEEGFS_SUPPORTED), "beegfs", /*on_nvme=*/false));
  CHECK(gds_supports_filesystem(flag(CU_FILE_NFS_SUPPORTED), "nfs", /*on_nvme=*/false));
  CHECK(gds_supports_filesystem(flag(CU_FILE_NFS_SUPPORTED), "nfs4", /*on_nvme=*/false));
  CHECK_FALSE(gds_supports_filesystem(flag(CU_FILE_NVME_SUPPORTED), "gpfs", /*on_nvme=*/true));

  // No class at all, or a filesystem cuFile has no GDS mode for.
  auto const every_class = ~0U;
  CHECK_FALSE(gds_supports_filesystem(0, "gpfs", /*on_nvme=*/true));
  CHECK_FALSE(gds_supports_filesystem(every_class, "tmpfs", /*on_nvme=*/false));
  CHECK_FALSE(gds_supports_filesystem(every_class, "overlay", /*on_nvme=*/true));
  CHECK_FALSE(gds_supports_filesystem(every_class, "", /*on_nvme=*/true));
}

TEST_CASE("gds_available agrees with the driver's GDS status flags", "[io][cufile]")
{
  CHECK(cucascade::io::cufile::gds_available() == (cucascade::io::cufile::gds_status_flags() != 0));
}

TEST_CASE("a cufile context rejects a configuration its reactors cannot run", "[io][cufile]")
{
  auto no_lanes  = small_config();
  no_lanes.lanes = 0;
  CHECK_THROWS_AS(cufile_ioctx{no_lanes}, std::invalid_argument);

  auto odd_lane       = small_config();
  odd_lane.lane_bytes = MiB + 1;
  CHECK_THROWS_AS(cufile_ioctx{odd_lane}, std::invalid_argument);
}

TEST_CASE("a cufile context reads and writes host and device memory", "[io][cufile]")
{
  cufile_ioctx context{small_config()};
  CHECK(context.type() == io_context_type::cufile);
  CHECK(context.supports_device_read());
  CHECK(context.supports_write());
  CHECK(context.supports_device_write());
  CHECK(context.min_alignment_requirement() == cucascade::io::IO_BLOCK_SIZE);
}

TEST_CASE("cufile device reads deliver the file's bytes", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  // A file that ends on a block boundary, and one that does not.
  auto const file_bytes = GENERATE(4 * MiB, 3 * MiB + 1234);
  auto const expected   = pattern(file_bytes, 1);
  scratch_file file{expected};
  started_ioctx started;
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  auto const [offset, size] = GENERATE_COPY(table<std::size_t, std::size_t>({
    {0, file_bytes},              // more than every read lane's slot
    {0, 4 * KiB},                 // one block
    {1, 1},                       // inside a block
    {4 * KiB - 1, 2},             // across a block boundary
    {5000, 2 * MiB + 77},         // unaligned, several slots
    {file_bytes - 100, 100},      // up to the end of the file
    {MiB - 7, file_bytes - MiB},  // unaligned, up to the end of the file
  }));
  CAPTURE(file_bytes, offset, size);
  device_bytes destination(size);
  CHECK(started.context
          .device_read_async_io(*object, offset, size, destination.data, destination.stream())
          .get(30s) == size);
  CHECK(destination.download() == sub(expected, offset, size));
}

TEST_CASE("cufile device reads that lie close together share one cuFile read", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  // 16 destinations of 60000 bytes, 5536 bytes apart: 1 MiB of file, one slot.
  constexpr std::size_t count  = 16;
  constexpr std::size_t stride = 64 * KiB;
  constexpr std::size_t length = 60000;
  auto const expected          = pattern(count * stride, 2);
  scratch_file file{expected};

  auto const coalesce = GENERATE(true, false);
  CAPTURE(coalesce);
  auto cfg = small_config();
  if (!coalesce) { cfg.coalesce_gap_bytes = 0; }
  substituted_api api;
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  device_bytes destination(count * length);
  std::vector<slice> slices;
  for (std::size_t i = 0; i < count; ++i) {
    slices.emplace_back(i * stride, length, destination.data + i * length);
  }
  CHECK(started.context.device_readv_async_io(*object, slices, destination.stream()).get(30s) ==
        count * length);
  CHECK(read_calls == (coalesce ? 1 : static_cast<int>(count)));

  auto const read = destination.download();
  for (std::size_t i = 0; i < count; ++i) {
    CAPTURE(i);
    CHECK(sub(read, i * length, length) == sub(expected, i * stride, length));
  }
}

TEST_CASE("cufile device writes land in the file", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto expected = pattern(3 * MiB, 3);
  scratch_file file{expected};
  started_ioctx started;
  auto object = started.context.open_io_object_for_write(
    file.path, write_open_options{.mode = write_mode::open_existing});
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  auto const [offset, size] = GENERATE(table<std::size_t, std::size_t>({
    {0, 3 * MiB},              // whole granules, more than every write lane's slot
    {4 * KiB, 8 * KiB},        // whole granules
    {100, 3000},               // inside a granule: no body
    {4000, 200},               // across a granule boundary: no body
    {1234, 2 * MiB + 4321},    // head, body over several slots, tail
    {3 * MiB - 5, 10},         // across the end of the file
    {4 * MiB + 5, MiB + 100},  // past the end of the file: the gap reads back as zeros
  }));
  CAPTURE(offset, size);
  auto const written = pattern(size, 4);
  device_bytes source(written);
  CHECK(started.context.device_write_async(*object, offset, size, source.data, source.stream())
          .get(30s) == size);

  if (expected.size() < offset + size) { expected.resize(offset + size, 0); }
  std::copy(written.begin(), written.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset));
  CHECK(object->size() == expected.size());
  CHECK(file.contents() == expected);
}

TEST_CASE("a cufile context serves host I/O and mixed writes", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  scratch_file file;
  started_ioctx started;
  auto& context = started.context;
  auto object   = context.open_io_object_for_write(file.path);

  // Host writes, synchronous and queued, at unaligned offsets.
  auto const first  = pattern(100000, 5);
  auto const second = pattern(50000, 6);
  CHECK(context.host_write(*object, 0, first.size(), first.data()) == first.size());
  CHECK(context.host_write_async(*object, first.size(), second.size(), second.data()).get(30s) ==
        second.size());

  // One request with a host and a device source: each goes to the reactor that serves it.
  auto const third  = pattern(30000, 7);
  auto const fourth = pattern(MiB + 123, 8);
  device_bytes source(fourth);
  auto const third_offset  = first.size() + second.size();
  auto const fourth_offset = third_offset + third.size();
  std::vector<write_slice> host_slices{{third_offset, third.size(), third.data()}};
  std::vector<write_slice> device_slices{{fourth_offset, fourth.size(), source.data}};
  auto mixed = context.mixed_writev_async(*object, host_slices, device_slices, source.stream());
  if (registered(*object)) {
    CHECK(std::move(mixed).get(30s) == third.size() + fourth.size());
  } else {
    CHECK_THROWS_AS(std::move(mixed).get(30s), std::system_error);
  }
  context.flush(*object);
  context.flush_async(*object).get(30s);

  auto expected = first;
  expected.insert(expected.end(), second.begin(), second.end());
  expected.insert(expected.end(), third.begin(), third.end());
  if (registered(*object)) { expected.insert(expected.end(), fourth.begin(), fourth.end()); }
  CHECK(file.contents() == expected);

  // Host reads, synchronous and queued.
  std::vector<std::uint8_t> read(expected.size());
  CHECK(context.host_read(*object, 0, read.size(), read.data()) == read.size());
  CHECK(read == expected);
  std::ranges::fill(read, std::uint8_t{0});
  CHECK(context.host_read_async(*object, 0, read.size(), read.data()).get(30s) == read.size());
  CHECK(read == expected);
}

TEST_CASE("a failing cuFile read is retried, then fails the request", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(MiB, 9);
  scratch_file file{expected};
  substituted_api api;
  started_ioctx started;
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  device_bytes destination(expected.size());
  auto read = [&] {
    return started.context.device_read_async_io(
      *object, 0, expected.size(), destination.data, destination.stream());
  };

  SECTION("a read that fails twice succeeds on its third attempt")
  {
    reads_to_fail = 2;
    CHECK(read().get(30s) == expected.size());
    CHECK(read_calls == 3);
    CHECK(destination.download() == expected);
  }
  SECTION("a read that keeps failing reports the error")
  {
    reads_to_fail = 100;
    try {
      read().get(30s);
      FAIL("the read resolved although cuFileRead failed");
    } catch (std::system_error const& error) {
      CHECK(error.code().value() == EIO);
    }
    CHECK(read_calls == 3);
    // The context keeps serving.
    reads_to_fail = 0;
    CHECK(read().get(30s) == expected.size());
    CHECK(destination.download() == expected);
  }
  SECTION("a read that stops before the end of the file is an error")
  {
    transfer_cap = 64 * KiB;
    try {
      read().get(30s);
      FAIL("the read resolved although cuFileRead returned short");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::io_error);
    }
  }
}

TEST_CASE("a short cuFile write continues where it stopped", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  scratch_file file;
  substituted_api api;
  started_ioctx started;
  auto object = started.context.open_io_object_for_write(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  // A cap that ends on a granule boundary and one that does not: the write
  // continues from the last whole granule either way.
  transfer_cap        = GENERATE(64 * KiB, 64 * KiB + 100);
  auto const expected = pattern(MiB, 10);
  device_bytes source(expected);
  CHECK(
    started.context.device_write_async(*object, 0, expected.size(), source.data, source.stream())
      .get(30s) == expected.size());
  CHECK(write_calls == 16);
  CHECK(file.contents() == expected);
}

TEST_CASE("a file cuFile cannot register moves its device I/O through host memory",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  // Unaligned, and over several bounce buffers of 1 MiB.
  auto expected = pattern(3 * MiB + 1234, 11);
  scratch_file file{expected};
  substituted_api api;
  api.table.handle_register = [](CUfileHandle_t*, CUfileDescr_t*) -> CUfileError_t {
    return cufile_status(CU_FILE_IO_NOT_SUPPORTED);
  };
  started_ioctx started;
  auto object = started.context.open_io_object(file.path);
  REQUIRE_FALSE(registered(*object));

  std::vector<std::uint8_t> host(expected.size());
  CHECK(started.context.host_read(*object, 0, host.size(), host.data()) == host.size());
  CHECK(host == expected);

  // Reads: one slot or several, aligned or not, up to the end of the file.
  for (auto const [offset, size] : std::vector<std::pair<std::size_t, std::size_t>>{
         {0, expected.size()}, {1, 1}, {5000, 2 * MiB + 77}, {expected.size() - 100, 100}}) {
    CAPTURE(offset, size);
    device_bytes destination(size);
    CHECK(started.context
            .device_read_async_io(*object, offset, size, destination.data, destination.stream())
            .get(30s) == size);
    CHECK(destination.download() == sub(expected, offset, size));
  }

  // A write with an unaligned head, a body over several slots and a tail, past the old end.
  auto writable = started.context.open_io_object_for_write(
    file.path, write_open_options{.mode = write_mode::open_existing});
  REQUIRE_FALSE(registered(*writable));
  auto const written = pattern(2 * MiB + 4321, 12);
  device_bytes source(written);
  constexpr std::size_t offset = 2 * MiB + 1234;
  CHECK(started.context
          .device_write_async(*writable, offset, written.size(), source.data, source.stream())
          .get(30s) == written.size());
  expected.resize(offset + written.size());
  std::ranges::copy(written, expected.begin() + static_cast<std::ptrdiff_t>(offset));
  CHECK(file.contents() == expected);

  // cuFile itself was never asked to move a byte.
  CHECK(read_calls == 0);
  CHECK(write_calls == 0);
}

TEST_CASE("cufile device reads never ask cuFile for bytes past the end of the file",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  // Files ending inside a block: a large one, and one smaller than a block.
  auto const file_bytes = GENERATE(3 * MiB + 1234, std::size_t{1000});
  auto const expected   = pattern(file_bytes, 17);
  scratch_file file{expected};
  substituted_api api;
  file_limit = file_bytes;  // a read reaching past it fails, as some nvidia-fs builds crash
  started_ioctx started;
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  auto const ranges = file_bytes > MiB ? std::vector<std::pair<std::size_t, std::size_t>>{
                                           {0, file_bytes},
                                           {file_bytes - 100, 100},
                                           {MiB - 7, file_bytes - MiB},
                                           {file_bytes - 5000, 4000},
                                         }
                                       : std::vector<std::pair<std::size_t, std::size_t>>{
                                           {0, file_bytes},
                                           {10, 20},
                                         };
  for (auto const [offset, size] : ranges) {
    CAPTURE(file_bytes, offset, size);
    device_bytes destination(size);
    CHECK(started.context
            .device_read_async_io(*object, offset, size, destination.data, destination.stream())
            .get(30s) == size);
    CHECK(destination.download() == sub(expected, offset, size));
  }
  CHECK(reads_past_eof == 0);
}

TEST_CASE("cuFile is passed an offset only into a registered buffer", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const per_slot = GENERATE(true, false);
  CAPTURE(per_slot);
  auto const expected = pattern(3 * MiB + 99, 18);
  scratch_file file{expected};
  scratch_file sink;
  fresh_arenas fresh;
  substituted_api api;
  // Per slot every second registration fails; as a whole, the one registration does.
  fail_registrations = per_slot ? failing_registrations::every_second : failing_registrations::all;
  auto cfg           = small_config();
  cfg.register_per_slot = per_slot;
  started_ioctx started{cfg};
  auto readable = started.context.open_io_object(file.path);
  auto writable = started.context.open_io_object_for_write(sink.path);
  if (!registered(*readable) || !registered(*writable)) {
    SKIP("file cannot register with cuFile");
  }

  // One segment at a time: a registered slot is always free.
  for (std::size_t offset = 0; offset < expected.size(); offset += MiB) {
    auto const size = std::min(MiB, expected.size() - offset);
    device_bytes destination(size);
    CHECK(started.context
            .device_read_async_io(*readable, offset, size, destination.data, destination.stream())
            .get(30s) == size);
    CHECK(destination.download() == sub(expected, offset, size));
  }
  // A write cuFile takes in pieces continues from an offset into its slot.
  transfer_cap       = 64 * KiB + 100;
  auto const written = pattern(MiB, 19);
  device_bytes source(written);
  CHECK(
    started.context.device_write_async(*writable, 0, written.size(), source.data, source.stream())
      .get(30s) == written.size());
  CHECK(sink.contents() == written);

  CHECK(misaddressed_transfers == 0);
  if (per_slot) {
    CHECK(unregistered_transfers == 0);  // the registered slots were taken first
  } else {
    CHECK(unregistered_transfers == read_calls + write_calls);
  }
}

TEST_CASE("a cufile reactor's arena is registered per slot or as a whole, and kept",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const per_slot = GENERATE(true, false);
  CAPTURE(per_slot);
  fresh_arenas fresh;
  substituted_api api;
  auto cfg                    = small_config();
  cfg.lanes                   = 3;
  cfg.register_per_slot       = per_slot;
  auto const registrations_of = [per_slot](config const& shape) {
    return per_slot ? static_cast<int>((shape.read_slots_per_lane + 1) * shape.lanes) : 1;
  };
  auto const first = registrations_of(cfg);
  CHECK(first == (per_slot ? 9 : 1));
  cucascade::io::cufile::reserve_arena(0, cfg);
  CHECK(buffer_registrations == first);
  // The reactors take the reserved arena, one after the other...
  {
    started_ioctx one{cfg};
  }
  {
    started_ioctx other{cfg};
  }
  // ...and so do reactors it covers -- fewer lanes or read slots, smaller slots: their lanes
  // share its slots.
  auto fewer  = cfg;
  fewer.lanes = 1;
  {
    started_ioctx one{fewer};
  }
  auto smaller       = cfg;
  smaller.lane_bytes = 512 * KiB;
  {
    started_ioctx one{smaller};
  }
  CHECK(buffer_registrations == first);
  // Larger slots need an arena that covers both shapes.
  auto larger       = fewer;
  larger.lane_bytes = 2 * MiB;
  {
    started_ioctx one{larger};
  }
  auto grown              = cfg;
  grown.lane_bytes        = larger.lane_bytes;
  auto const after_larger = first + registrations_of(grown);
  CHECK(buffer_registrations == after_larger);
  // So does the other registration mode.
  auto other_mode              = cfg;
  other_mode.register_per_slot = !per_slot;
  {
    started_ioctx one{other_mode};
  }
  CHECK(buffer_registrations > after_larger);
}

TEST_CASE("a later cufile context is not held to an earlier context's arena depth",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 22);
  scratch_file file{expected};
  fresh_arenas fresh;
  substituted_api api;
  auto shallow                = small_config();
  shallow.lanes               = 1;
  shallow.read_slots_per_lane = 1;  // an arena of one read slot
  started_ioctx idle{shallow};
  auto deep = small_config();  // two lanes, four read slots
  started_ioctx started{deep};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  struct release_reads {
    ~release_reads() { hold_reads = false; }
  } const release;

  // Both reads are inside cuFileRead at once: the deeper context got an arena that covers it.
  hold_reads = true;
  device_bytes first(length);
  device_bytes second(length);
  auto first_read =
    started.context.device_read_async_io(*object, 0, length, first.data, first.stream());
  auto second_read =
    started.context.device_read_async_io(*object, length, length, second.data, second.stream());
  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (read_calls < 2 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  CHECK(read_calls == 2);
  hold_reads = false;
  CHECK(std::move(first_read).get(30s) == length);
  CHECK(std::move(second_read).get(30s) == length);
  CHECK(first.download() == sub(expected, 0, length));
  CHECK(second.download() == sub(expected, length, length));
}

TEST_CASE("a GPU keeps its arena when a larger one cannot be allocated", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  fresh_arenas fresh;
  substituted_api api;
  auto const cfg = small_config();
  cucascade::io::cufile::reserve_arena(0, cfg);
  auto const reserved = buffer_registrations.load();

  {
    // Leave the GPU no memory for a larger arena.
    struct hog {
      std::vector<void*> chunks;
      ~hog()
      {
        for (auto* chunk : chunks) {
          std::ignore = cudaFree(chunk);
        }
      }
    } taken;
    for (void* chunk = nullptr; cudaMalloc(&chunk, 16 * MiB) == cudaSuccess; chunk = nullptr) {
      taken.chunks.push_back(chunk);
    }
    std::ignore       = cudaGetLastError();
    auto larger       = cfg;
    larger.lane_bytes = 64 * MiB;
    CHECK_THROWS(cucascade::io::cufile::reserve_arena(0, larger));
  }

  // The arena it had still serves a context of its shape: nothing new is registered.
  {
    started_ioctx one{cfg};
  }
  CHECK(buffer_registrations == reserved);
}

TEST_CASE("an arena grown for two reactors has the larger of their slot counts",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  fresh_arenas fresh;
  substituted_api api;
  auto deep                = small_config();
  deep.lanes               = 1;
  deep.read_slots_per_lane = 64;  // 64 read slots, 1 write slot
  cucascade::io::cufile::reserve_arena(0, deep);
  CHECK(buffer_registrations == 65);
  auto wide  = small_config();
  wide.lanes = 32;  // 64 read slots, 32 write slots
  cucascade::io::cufile::reserve_arena(0, wide);
  // 64 read + 32 write slots cover both, not 32 lanes x 64 read slots.
  CHECK(buffer_registrations == 65 + 96);
}

TEST_CASE("a grown arena registers what its released predecessor had pinned", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  fresh_arenas fresh;
  substituted_api api;
  // Pinning room for either arena, not both: 6 and 9 slots of 1 MiB under a 10 MiB limit.
  pin_budget       = 10 * MiB;
  auto const small = small_config();  // 4 read + 2 write slots
  cucascade::io::cufile::reserve_arena(0, small);
  auto larger  = small;
  larger.lanes = 3;  // 6 read + 3 write slots
  cucascade::io::cufile::reserve_arena(0, larger);
  // No reactor held the old arena: once it let go of its pins, every slot of the new one
  // registered.
  std::lock_guard const lock(registered_mutex);
  CHECK(registered_buffers.size() == 9);
}

TEST_CASE("an arena registers on its own GPU whichever device is current",
          "[io][cufile][gpu][multi-device]")
{
  int devices = 0;
  if (!cufile_usable() || cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    SKIP("requires two CUDA devices and a cuFile driver");
  }
  fresh_arenas fresh;
  substituted_api api;
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  // GPU 1's arena, reserved and then grown -- the retry included -- from a thread on GPU 0.
  pin_budget    = 10 * MiB;
  auto small    = small_config();
  small.devices = {1};
  cucascade::io::cufile::reserve_arena(1, small);
  auto larger  = small;
  larger.lanes = 3;
  cucascade::io::cufile::reserve_arena(1, larger);
  CHECK(registrations_off_device == 0);
  int current = -1;
  REQUIRE(cudaGetDevice(&current) == cudaSuccess);
  CHECK(current == 0);
  std::lock_guard const lock(registered_mutex);
  CHECK(registered_buffers.size() == 9);
}

TEST_CASE("an arena covering two valid shapes but too large to address is refused",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  fresh_arenas fresh;
  substituted_api api;
  auto large                = small_config();
  large.lanes               = 1;
  large.read_slots_per_lane = 1;
  large.lane_bytes          = 64 * MiB;  // 2 slots of 64 MiB
  cucascade::io::cufile::reserve_arena(0, large);
  auto const reserved      = buffer_registrations.load();
  auto many                = small_config();
  many.lanes               = 1;
  many.read_slots_per_lane = std::size_t{1} << 50;  // valid alone: 2^62 bytes of 4 KiB slots
  many.lane_bytes          = 4 * KiB;
  CHECK_THROWS_AS(cucascade::io::cufile::reserve_arena(0, many), std::invalid_argument);
  // The GPU keeps its arena.
  {
    started_ioctx one{large};
  }
  CHECK(buffer_registrations == reserved);
}

TEST_CASE("an arena larger than the address space is refused", "[io][cufile]")
{
  auto cfg                = small_config();
  cfg.read_slots_per_lane = std::size_t{1} << 62;
  CHECK_THROWS_AS(cufile_ioctx{cfg}, std::invalid_argument);
  CHECK_THROWS_AS(cucascade::io::cufile::reserve_arena(0, cfg), std::invalid_argument);
}

TEST_CASE("the cufile contexts of a GPU share its arena", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(6 * MiB + 321, 13);
  scratch_file file{expected};
  fresh_arenas fresh;
  substituted_api api;
  auto cfg  = small_config();
  cfg.lanes = 5;
  started_ioctx first{cfg};
  // Another lane count on the same GPU: the first context's arena serves it too.
  auto other  = cfg;
  other.lanes = 2;
  started_ioctx second{other};
  CHECK(buffer_registrations == static_cast<int>((cfg.read_slots_per_lane + 1) * cfg.lanes));

  // Both at once: seven read lanes take turns on the arena's ten read slots.
  auto first_object  = first.context.open_io_object(file.path);
  auto second_object = second.context.open_io_object(file.path);
  if (!registered(*first_object)) { SKIP("file cannot register with cuFile"); }
  device_bytes first_destination(expected.size());
  device_bytes second_destination(expected.size());
  auto first_read = first.context.device_read_async_io(
    *first_object, 0, expected.size(), first_destination.data, first_destination.stream());
  auto second_read = second.context.device_read_async_io(
    *second_object, 0, expected.size(), second_destination.data, second_destination.stream());
  CHECK(std::move(first_read).get(30s) == expected.size());
  CHECK(std::move(second_read).get(30s) == expected.size());
  CHECK(first_destination.download() == expected);
  CHECK(second_destination.download() == expected);
}

TEST_CASE("a read segment keeps its slot until its copies are done, whatever the query says",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 20);
  scratch_file file{expected};
  fresh_arenas fresh;
  substituted_api api;
  auto cfg                = small_config();
  cfg.lanes               = 1;
  cfg.read_slots_per_lane = 1;  // one read slot: the second read needs the first one's
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  device_bytes first(length);
  device_bytes second(length);
  // The first read's copy waits for its caller's stream, and every query of a copy reports an
  // error, as cudaEventQuery does for errors of earlier asynchronous work.
  failing_copy_queries failing{cudaErrorLaunchFailure};
  stream_hold hold{first.raw_stream};

  auto first_read =
    started.context.device_read_async_io(*object, 0, length, first.data, first.stream());
  auto const deadline = std::chrono::steady_clock::now() + 30s;
  while (read_calls == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(read_calls == 1);
  auto second_read =
    started.context.device_read_async_io(*object, length, length, second.data, second.stream());
  std::this_thread::sleep_for(300ms);
  // Neither the first caller nor the next read may take over what the copy still uses.
  CHECK_FALSE(first_read.is_ready());
  CHECK(read_calls == 1);

  hold.release();
  CHECK(std::move(first_read).get(30s) == length);
  CHECK(std::move(second_read).get(30s) == length);
  CHECK(first.download() == sub(expected, 0, length));
  CHECK(second.download() == sub(expected, length, length));
  CHECK(read_calls == 2);
}

TEST_CASE("a copy query error holds up no other read", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 23);
  scratch_file file{expected};
  fresh_arenas fresh;
  auto cfg  = small_config();
  cfg.lanes = 1;  // one lane, two read slots
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  device_bytes slow(length);
  device_bytes ready(length);
  failing_copy_queries failing{cudaErrorLaunchFailure};
  stream_hold hold{slow.raw_stream};

  auto slow_read =
    started.context.device_read_async_io(*object, 0, length, slow.data, slow.stream());
  auto ready_read =
    started.context.device_read_async_io(*object, length, length, ready.data, ready.stream());
  // The completion thread does not wait for the held copy to settle the ready one.
  CHECK(ready_within(ready_read, 10s));
  CHECK_FALSE(slow_read.is_ready());

  hold.release();
  CHECK(std::move(slow_read).get(30s) == length);
  CHECK(std::move(ready_read).get(30s) == length);
  CHECK(slow.download() == sub(expected, 0, length));
  CHECK(ready.download() == sub(expected, length, length));
}

TEST_CASE("a read whose copy stream reports an error stays pending while the context is in service",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 26);
  scratch_file file{expected};
  fresh_arenas fresh;
  auto cfg  = small_config();
  cfg.lanes = 1;
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  // An error the CUDA runtime does not document as terminal -- an allocation failure, say -- may
  // be earlier work's: the copies may still write the destination, so the read does not resolve
  // (and its buffer stays the reactor's) until they are shown done.
  auto const stream_error = GENERATE(cudaErrorMemoryAllocation, cudaErrorInvalidValue);
  CAPTURE(stream_error);
  device_bytes destination(length);
  auto read = [&] {
    failing_copy_queries failing{cudaErrorLaunchFailure, stream_error};
    auto pending = started.context.device_read_async_io(
      *object, 0, length, destination.data, destination.stream());
    std::this_thread::sleep_for(300ms);
    CHECK_FALSE(pending.is_ready());
    return pending;
  }();
  CHECK(std::move(read).get(30s) == length);
  CHECK(destination.download() == sub(expected, 0, length));
  // Its slot came back.
  device_bytes next(length);
  CHECK(started.context.device_read_async_io(*object, length, length, next.data, next.stream())
          .get(30s) == length);
}

TEST_CASE("once the context is out of service a read fails and retires its slot",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(3 * length, 24);
  scratch_file file{expected};
  fresh_arenas fresh;
  substituted_api api;
  auto cfg                = small_config();  // two lanes
  cfg.read_slots_per_lane = 1;               // two read slots
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }

  device_bytes failed(length);
  {
    // A status the CUDA runtime documents as leaving the context unusable: nothing queued on it
    // runs any more, so the read fails -- with that status, for the caller to see.
    failing_copy_queries failing{cudaErrorLaunchFailure, cudaErrorIllegalAddress};
    try {
      std::ignore =
        started.context.device_read_async_io(*object, 0, length, failed.data, failed.stream())
          .get(30s);
      FAIL("the read resolved although its context is out of service");
    } catch (cucascade::cuda_status_error const& error) {
      CHECK(error.error_code() == cudaErrorIllegalAddress);
    }
  }

  // One read slot is left: of two reads held inside cuFileRead, only one gets that far.
  struct release_reads {
    ~release_reads() { hold_reads = false; }
  } const release;
  read_calls = 0;
  hold_reads = true;
  device_bytes first(length);
  device_bytes second(length);
  auto first_read =
    started.context.device_read_async_io(*object, length, length, first.data, first.stream());
  auto second_read =
    started.context.device_read_async_io(*object, 2 * length, length, second.data, second.stream());
  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (read_calls < 1 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  std::this_thread::sleep_for(300ms);
  CHECK(read_calls == 1);
  hold_reads = false;
  CHECK(std::move(first_read).get(30s) == length);
  CHECK(std::move(second_read).get(30s) == length);
  CHECK(first.download() == sub(expected, length, length));
  CHECK(second.download() == sub(expected, 2 * length, length));
}

TEST_CASE("a GPU whose read slots are all retired fails reads, and a new context gets a new arena",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 27);
  scratch_file file{expected};
  fresh_arenas fresh;
  substituted_api api;
  auto cfg                = small_config();
  cfg.lanes               = 1;
  cfg.read_slots_per_lane = 1;  // one read slot
  device_bytes destination(length);
  {
    started_ioctx started{cfg};
    auto object = started.context.open_io_object(file.path);
    if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
    {
      failing_copy_queries failing{cudaErrorLaunchFailure, cudaErrorIllegalAddress};
      CHECK_THROWS(
        started.context
          .device_read_async_io(*object, 0, length, destination.data, destination.stream())
          .get(30s));
    }
    // No slot is left: the next read fails at once instead of waiting for one.
    CHECK_THROWS_AS(
      started.context
        .device_read_async_io(*object, length, length, destination.data, destination.stream())
        .get(10s),
      std::system_error);
  }
  // The depleted arena does not cover a new context: it gets an arena of its own.
  auto const before = buffer_registrations.load();
  started_ioctx started{cfg};
  CHECK(buffer_registrations > before);
  auto object = started.context.open_io_object(file.path);
  CHECK(started.context
          .device_read_async_io(*object, length, length, destination.data, destination.stream())
          .get(30s) == length);
  CHECK(destination.download() == sub(expected, length, length));
}

TEST_CASE("a cufile context shuts down while another context of its GPU holds every slot",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(2 * length, 25);
  scratch_file file{expected};
  fresh_arenas fresh;
  auto cfg                = small_config();
  cfg.lanes               = 1;
  cfg.read_slots_per_lane = 1;  // one read slot, shared by both contexts
  started_ioctx holder{cfg};
  cufile_ioctx waiter{cfg};
  waiter.start();
  auto holder_object = holder.context.open_io_object(file.path);
  auto waiter_object = waiter.open_io_object(file.path);
  if (!registered(*holder_object)) { SKIP("file cannot register with cuFile"); }
  device_bytes held(length);
  device_bytes waiting(length);
  stream_hold hold{held.raw_stream};

  // The holder's copy keeps the only slot until its caller's stream is released.
  auto held_read =
    holder.context.device_read_async_io(*holder_object, 0, length, held.data, held.stream());
  std::this_thread::sleep_for(200ms);
  auto waiting_read =
    waiter.device_read_async_io(*waiter_object, length, length, waiting.data, waiting.stream());
  std::this_thread::sleep_for(200ms);

  // The waiter's lane gives up its wait for the slot, so shutting the waiter down returns.
  auto stopped        = std::async(std::launch::async, [&waiter] { waiter.shutdown(); });
  bool const returned = stopped.wait_for(10s) == std::future_status::ready;
  CHECK(returned);
  hold.release();
  stopped.get();
  CHECK_THROWS(std::move(waiting_read).get(30s));
  CHECK(std::move(held_read).get(30s) == length);
  CHECK(held.download() == sub(expected, 0, length));
}

TEST_CASE("a copy waiting for its caller does not hold up another caller's read",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t length = 64 * KiB;
  auto const expected          = pattern(3 * length, 21);
  scratch_file file{expected};
  fresh_arenas fresh;
  auto cfg  = small_config();
  cfg.lanes = 1;  // one lane, two read slots
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  device_bytes slow(length);
  device_bytes ready(length);
  device_bytes next(length);
  stream_hold hold{slow.raw_stream};

  auto slow_read =
    started.context.device_read_async_io(*object, 0, length, slow.data, slow.stream());
  auto ready_read =
    started.context.device_read_async_io(*object, length, length, ready.data, ready.stream());
  auto next_read =
    started.context.device_read_async_io(*object, 2 * length, length, next.data, next.stream());
  // The second read's copy does not queue behind the first one's, so its slot comes back for
  // the third read while the first caller's stream is still busy.
  CHECK(ready_within(ready_read, 10s));
  CHECK(ready_within(next_read, 10s));
  CHECK_FALSE(slow_read.is_ready());

  hold.release();
  CHECK(std::move(slow_read).get(30s) == length);
  CHECK(std::move(ready_read).get(30s) == length);
  CHECK(std::move(next_read).get(30s) == length);
  CHECK(slow.download() == sub(expected, 0, length));
  CHECK(ready.download() == sub(expected, length, length));
  CHECK(next.download() == sub(expected, 2 * length, length));
}

TEST_CASE("a waiting low-priority cufile read is served after four high-priority ones",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  constexpr std::size_t stride = 64 * KiB;
  constexpr std::size_t length = 4 * KiB;
  auto const expected          = pattern(8 * stride, 14);
  scratch_file file{expected};
  auto cfg  = small_config();
  cfg.lanes = 1;  // one read at a time, in the order the lane takes them
  substituted_api api;
  started_ioctx started{cfg};
  auto object = started.context.open_io_object(file.path);
  if (!registered(*object)) { SKIP("file cannot register with cuFile"); }
  // Declared after the context: its lane leaves cuFileRead before the context shuts down.
  struct release_reads {
    ~release_reads() { hold_reads = false; }
  } const release;

  device_bytes destination(8 * length);
  std::vector<cucascade::exec::semi_future<std::size_t>> reads;
  auto read = [&](std::size_t i, io_priority priority) {
    reads.push_back(started.context.device_read_async_io(
      *object, i * stride, length, destination.data + i * length, destination.stream(), priority));
  };

  // The lane stays inside its first read while the others queue up behind it.
  hold_reads = true;
  read(0, io_priority::high);
  auto const deadline = std::chrono::steady_clock::now() + 30s;
  while (read_calls == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(read_calls == 1);
  read(7, io_priority::low);
  for (std::size_t i = 1; i <= 6; ++i) {
    read(i, io_priority::high);
  }
  hold_reads = false;
  for (auto& pending : reads) {
    CHECK(std::move(pending).get(30s) == length);
  }

  std::lock_guard const lock(read_order_mutex);
  CHECK(read_order ==
        std::vector<std::size_t>{
          0, 1 * stride, 2 * stride, 3 * stride, 4 * stride, 7 * stride, 5 * stride, 6 * stride});
}

TEST_CASE("a cufile context reports what its reactors moved", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(2 * MiB, 15);
  scratch_file source{expected};
  scratch_file sink;
  started_ioctx started;
  auto readable = started.context.open_io_object(source.path);
  auto writable = started.context.open_io_object_for_write(sink.path);
  if (!registered(*readable) || !registered(*writable)) {
    SKIP("file cannot register with cuFile");
  }

  // Two lane slots of file in, then out.
  device_bytes buffer(expected.size());
  CHECK(started.context
          .device_read_async_io(*readable, 0, expected.size(), buffer.data, buffer.stream())
          .get(30s) == expected.size());
  CHECK(
    started.context.device_write_async(*writable, 0, expected.size(), buffer.data, buffer.stream())
      .get(30s) == expected.size());

  auto const gauges = started.context.reactor_gauges();
  REQUIRE(gauges.size() == 1);
  CHECK(gauges[0].device == 0);
  CHECK(gauges[0].queued_bytes == 0);
  CHECK(gauges[0].reads == 2);
  CHECK(gauges[0].bytes_read == expected.size());
  CHECK(gauges[0].writes == 2);
  CHECK(gauges[0].bytes_written == expected.size());
}

TEST_CASE("a cufile reactor that was shut down can be started again", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(MiB + 77, 16);
  scratch_file file{expected};
  cufile_reactor reactor{small_config(), 0};
  std::shared_ptr<cufile_io_object const> const object =
    cufile_reactor::create_io_object(file.path);
  if (!object->cufile_registered()) { SKIP("file cannot register with cuFile"); }

  for (int round = 0; round < 2; ++round) {
    CAPTURE(round);
    reactor.start();
    device_bytes destination(expected.size());
    std::vector<cucascade::io::prepared_io_slice> slices;
    slices.emplace_back(cucascade::io::range{0, expected.size()},
                        cucascade::io::device_buffer{destination.data, destination.stream(), 0});
    auto request = cucascade::io::grouped_io_request::create(object, std::move(slices));
    auto read    = request->coordinator->get_future();
    reactor.enqueue(std::move(request));
    CHECK(std::move(read).get(30s) == expected.size());
    CHECK(destination.download() == expected);
    reactor.shutdown();
  }
}

TEST_CASE("a cufile worker that cannot select its GPU fails start instead of the process",
          "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(MiB + 77, 17);
  scratch_file file{expected};
  cufile_reactor reactor{small_config(), 0};
  {
    failing_worker_setup const failing{cudaErrorLaunchFailure};
    CHECK_THROWS_AS(reactor.start(), cucascade::cuda_error);
  }

  // The failed start left nothing behind: the reactor starts again and reads.
  reactor.start();
  std::shared_ptr<cufile_io_object const> const object =
    cufile_reactor::create_io_object(file.path);
  device_bytes destination(expected.size());
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(cucascade::io::range{0, expected.size()},
                      cucascade::io::device_buffer{destination.data, destination.stream(), 0});
  auto request = cucascade::io::grouped_io_request::create(object, std::move(slices));
  auto read    = request->coordinator->get_future();
  reactor.enqueue(std::move(request));
  CHECK(std::move(read).get(30s) == expected.size());
  CHECK(destination.download() == expected);
  reactor.shutdown();
}

TEST_CASE("cufile device I/O runs on the reactor of the buffer's GPU",
          "[io][cufile][gpu][.multi-device]")
{
  int devices = 0;
  if (!cufile_usable() || cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    SKIP("requires two CUDA devices and a cuFile driver");
  }
  auto const expected = pattern(2 * MiB + 99, 12);
  scratch_file file{expected};

  // The submitting thread's current device names the GPU; its own thread keeps device 0 current
  // for the rest of the test binary.
  auto read_on = [&](cufile_ioctx& context, int device, bool served) {
    std::thread([&] {
      REQUIRE(cudaSetDevice(device) == cudaSuccess);
      auto object = context.open_io_object(file.path);
      if (!registered(*object)) { return; }
      device_bytes destination(expected.size());
      auto read = context.device_read_async_io(
        *object, 0, expected.size(), destination.data, destination.stream());
      if (served) {
        CHECK(std::move(read).get(30s) == expected.size());
        CHECK(destination.download() == expected);
      } else {
        CHECK_THROWS(std::move(read).get(30s));
      }
    }).join();
  };

  auto cfg = small_config();
  SECTION("a context of every GPU serves each of them")
  {
    cfg.devices = {};
    started_ioctx started{cfg};
    read_on(started.context, 0, true);
    read_on(started.context, 1, true);
  }
  SECTION("a context of one GPU refuses another GPU's buffers")
  {
    cfg.devices = {1};
    started_ioctx started{cfg};
    read_on(started.context, 1, true);
    read_on(started.context, 0, false);
  }
}
