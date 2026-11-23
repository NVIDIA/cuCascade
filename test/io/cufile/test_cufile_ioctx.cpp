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
std::atomic<int> buffer_registrations{0};
std::atomic<int> reads_to_fail{0};
std::atomic<std::size_t> transfer_cap{0};  // 0 = uncapped
std::atomic<bool> hold_reads{false};       // a read waits inside cuFileRead while set
std::mutex read_order_mutex;
std::vector<std::size_t> read_order;  // the file offset of every cuFileRead, in call order

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
      while (hold_reads.load()) {
        std::this_thread::sleep_for(1ms);
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
      auto const cap = transfer_cap.load();
      return cufile_api::loaded()->write(
        handle, base, cap == 0 ? size : std::min(size, cap), file_offset, base_offset);
    };
    table.buf_register = [](void const* base, std::size_t size, int flags) -> CUfileError_t {
      ++buffer_registrations;
      return cufile_api::loaded()->buf_register(base, size, flags);
    };
    read_calls           = 0;
    write_calls          = 0;
    buffer_registrations = 0;
    reads_to_fail        = 0;
    transfer_cap         = 0;
    hold_reads           = false;
    read_order.clear();
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&table);
  }
  ~substituted_api() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  substituted_api(substituted_api const&)            = delete;
  substituted_api& operator=(substituted_api const&) = delete;

  cufile_api table;
};

}  // namespace

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

TEST_CASE("a file cuFile cannot register serves host I/O only", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(64 * KiB, 11);
  scratch_file file{expected};
  substituted_api api;
  api.table.handle_register = [](CUfileHandle_t*, CUfileDescr_t*) -> CUfileError_t {
    return cufile_status(CU_FILE_IO_NOT_SUPPORTED);
  };
  started_ioctx started;
  auto object = started.context.open_io_object(file.path);
  REQUIRE_FALSE(registered(*object));

  std::vector<std::uint8_t> read(expected.size());
  CHECK(started.context.host_read(*object, 0, read.size(), read.data()) == read.size());
  CHECK(read == expected);

  device_bytes destination(expected.size());
  try {
    started.context
      .device_read_async_io(*object, 0, expected.size(), destination.data, destination.stream())
      .get(30s);
    FAIL("a device read resolved on a file without a cuFile handle");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::operation_not_supported);
  }
}

TEST_CASE("a cufile reactor's arena is registered once and kept", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  substituted_api api;
  // A size no other test uses: the reservation has to allocate it.
  auto cfg  = small_config();
  cfg.lanes = 3;
  cucascade::io::cufile::reserve_arena(0, cfg);
  CHECK(buffer_registrations == 1);
  // The reactors take the reserved arena, one after the other.
  {
    started_ioctx first{cfg};
  }
  {
    started_ioctx second{cfg};
  }
  CHECK(buffer_registrations == 1);
  // Another size needs another arena.
  cfg.lanes = 1;
  {
    started_ioctx third{cfg};
  }
  CHECK(buffer_registrations == 2);
}

TEST_CASE("the cufile contexts of a GPU share its arena", "[io][cufile][gpu]")
{
  if (!cufile_usable()) { SKIP("requires a CUDA device and a cuFile driver"); }
  auto const expected = pattern(6 * MiB + 321, 13);
  scratch_file file{expected};
  substituted_api api;
  // A size no other test uses: the first context has to allocate it.
  auto cfg  = small_config();
  cfg.lanes = 5;
  started_ioctx first{cfg};
  started_ioctx second{cfg};
  CHECK(buffer_registrations == 1);

  // Both at once: ten read lanes take turns on the arena's five read slots.
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
