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
 * [fused_lanes] - the fused lane set (cuFileReadAsync on per-lane slots and streams)
 * [gpu]         - needs a CUDA device
 *
 * Content is checked against the file for every read.  Cases run against libcufile (in
 * compatibility mode on hosts without GDS) unless they install a scripted cuFile table
 * (detail::set_cufile_api_for_testing) to inject failures and short reads.
 */

#include "io/cufile/fused_lanes.hpp"
#include "transport_simulator_kernels.hpp"

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using cucascade::io::range;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_reactor;
using cucascade::io::cufile::lane_completion;
using cucascade::io::cufile::detail::fused_lane_set;
using cucascade::io::cufile::detail::lane_copy;
using cucascade::io::cufile::detail::lane_read;
using cucascade::io::cufile::detail::lane_result;
using cucascade::io::cufile::detail::lane_set_options;
using cucascade::io::cufile::detail::lanes_in_use_for_testing;
using cucascade::io::cufile::detail::set_lane_callback_failure_for_testing;
using cucascade::io::cufile::detail::set_lane_copy_failure_for_testing;

namespace {

using namespace std::chrono_literals;

constexpr std::size_t block = 4096;
constexpr std::size_t KiB   = 1024;
constexpr std::size_t MiB   = 1024 * KiB;

/// splitmix64 of @p offset: never periodic, so misplaced bytes cannot compare equal.
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
            ("cucascade_fused_lanes_" + std::to_string(::getpid()) + "_" +
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

/// Device memory freed on scope exit.
class device_memory {
 public:
  explicit device_memory(std::size_t bytes) : _bytes(bytes)
  {
    REQUIRE(cudaMalloc(&_ptr, bytes) == cudaSuccess);
    fill(0);
  }
  ~device_memory() { std::ignore = cudaFree(_ptr); }
  device_memory(device_memory const&)            = delete;
  device_memory& operator=(device_memory const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return static_cast<std::uint8_t*>(_ptr); }

  void fill(int value) const
  {
    REQUIRE(cudaMemset(_ptr, value, _bytes) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }

 private:
  void* _ptr{nullptr};
  std::size_t _bytes;
};

/// @p bytes at @p device, copied on a fresh stream (the lane streams are not synchronized).
[[nodiscard]] std::vector<std::uint8_t> bytes_at(void const* device, std::size_t bytes)
{
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  std::vector<std::uint8_t> out(bytes);
  REQUIRE(cudaMemcpyAsync(out.data(), device, bytes, cudaMemcpyDeviceToHost, other) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
  return out;
}

/// Whether every copy of @p read landed the file's bytes.
[[nodiscard]] bool copies_match(std::vector<lane_copy> const& copies, pattern_file const& file)
{
  for (auto const& copy : copies) {
    auto const got = bytes_at(copy.dst, copy.bytes);
    if (std::memcmp(got.data(), file.data() + copy.offset, copy.bytes) != 0) return false;
  }
  return true;
}

/// Counts wakes and lets a test wait for one (bounded).
struct waker {
  std::atomic<std::size_t> count{0};
  std::mutex mutex;
  std::condition_variable changed;
  bool pending{false};

  static void wake(void* context) noexcept
  {
    auto* const self = static_cast<waker*>(context);
    self->count.fetch_add(1);
    {
      std::lock_guard lock(self->mutex);
      self->pending = true;
    }
    self->changed.notify_all();
  }

  void wait(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(mutex);
    changed.wait_for(lock, timeout, [this] { return pending; });
    pending = false;
  }
};

[[nodiscard]] lane_set_options make_options(std::size_t lanes,
                                            std::size_t slot_bytes = 1 * MiB,
                                            waker* wake            = nullptr)
{
  lane_set_options options;
  options.device           = 0;
  options.lanes            = lanes;
  options.slot_bytes       = slot_bytes;
  options.max_device_lanes = 64;
  if (wake != nullptr) {
    options.wake         = &waker::wake;
    options.wake_context = wake;
  }
  return options;
}

/// Reap until @p count results arrived; fails (does not hang) after @p timeout.
std::vector<lane_result> reap_n(fused_lane_set& set,
                                std::size_t count,
                                std::chrono::seconds timeout = 30s)
{
  std::vector<lane_result> results;
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (results.size() < count) {
    std::ignore = set.reap([&](lane_result const& r) { results.push_back(r); });
    if (results.size() >= count) break;
    if (std::chrono::steady_clock::now() > deadline) {
      FAIL("timed out waiting for lane completions (" << results.size() << " of " << count
                                                      << ")");
    }
    std::this_thread::sleep_for(100us);
  }
  return results;
}

/// A read of @p window with @p copies (file offsets) packed into @p dst at @p gap apart.
[[nodiscard]] lane_read make_read(void* handle,
                                  range window,
                                  std::vector<std::pair<std::size_t, std::size_t>> const& parts,
                                  std::uint8_t* dst,
                                  std::size_t gap      = 17,
                                  std::uint64_t token  = 0)
{
  lane_read read;
  read.handle = handle;
  read.window = window;
  read.token  = token;
  std::size_t pos = 3;
  for (auto const& [offset, bytes] : parts) {
    read.copies.push_back(lane_copy{dst + pos, offset, bytes});
    pos += bytes + gap;
  }
  return read;
}

/// Opens @p file with cuFile, or skips the test when cuFile did not register it.
[[nodiscard]] std::unique_ptr<cufile_io_object> open_registered(pattern_file const& file)
{
  auto object = cufile_reactor::create_io_object(file.path());
  if (object->cufile_handle() == nullptr) { SKIP("cuFile did not register the file"); }
  return object;
}

/// Skips unless libcufile offers cuFileReadAsync.
void require_async_cufile()
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_async_read || !cufile_api::open_driver()) {
    SKIP("libcufile with cuFileReadAsync is unavailable");
  }
}


// ---------------------------------------------------------------------------
// A scripted cuFile: plain functions over one process-wide state.
// ---------------------------------------------------------------------------

struct fake_call {
  std::size_t size{0};
  std::size_t file_offset{0};
  std::size_t buffer_offset{0};
  void const* base{nullptr};
  CUstream stream{nullptr};
};

/// What one cuFileReadAsync call does.
struct fake_action {
  enum class kind : std::uint8_t { normal, refuse, negative, short_read };
  kind what{kind::normal};
  std::size_t bytes{0};  ///< short_read: bytes it returns
};

struct fake_handle {
  int fd{-1};
};

struct fake_state {
  std::mutex mutex;
  std::vector<fake_call> calls;
  std::vector<fake_action> script;  ///< per call; later calls are normal
  std::optional<fake_action> every;  ///< overrides the script for every call
  std::deque<std::vector<std::uint8_t>> keep_alive;
  std::size_t stream_registers{0};
  std::size_t stream_deregisters{0};
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
  *handle       = owned;
  return fake_ok();
}

void fake_handle_deregister(CUfileHandle_t handle)
{
  auto* const owned = static_cast<fake_handle*>(handle);
  ::close(owned->fd);
  delete owned;
}

ssize_t fake_read(CUfileHandle_t, void*, std::size_t, off_t, off_t)
{
  errno = EIO;
  return -1;
}

CUfileError_t fake_buf_register(void const*, std::size_t, int) { return fake_ok(); }
CUfileError_t fake_buf_deregister(void const*) { return fake_ok(); }

CUfileError_t fake_stream_register(CUstream, unsigned)
{
  std::lock_guard lock(fake().mutex);
  ++fake().stream_registers;
  return fake_ok();
}

CUfileError_t fake_stream_deregister(CUstream)
{
  std::lock_guard lock(fake().mutex);
  ++fake().stream_deregisters;
  return fake_ok();
}

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
  state.calls.push_back(fake_call{*size_p,
                                  static_cast<std::size_t>(*file_offset_p),
                                  static_cast<std::size_t>(*buffer_offset_p),
                                  base,
                                  stream});
  auto const index  = state.calls.size() - 1;
  auto const action = state.every ? *state.every
                      : index < state.script.size() ? state.script[index]
                                                    : fake_action{};
  if (action.what == fake_action::kind::refuse) {
    CUfileError_t status{};
    status.err = CU_FILE_IO_ERROR;
    return status;
  }
  if (action.what == fake_action::kind::negative) {
    *bytes_read_p = -5;
    return fake_ok();
  }
  auto limit = *size_p;
  if (action.what == fake_action::kind::short_read) limit = std::min(limit, action.bytes);
  auto& host     = state.keep_alive.emplace_back(limit);
  auto const got = ::pread(static_cast<fake_handle*>(handle)->fd, host.data(), limit, *file_offset_p);
  if (got > 0 &&
      cudaMemcpyAsync(static_cast<std::uint8_t*>(base) + *buffer_offset_p,
                      host.data(),
                      static_cast<std::size_t>(got),
                      cudaMemcpyHostToDevice,
                      static_cast<cudaStream_t>(stream)) != cudaSuccess) {
    *bytes_read_p = -5;
    return fake_ok();
  }
  *bytes_read_p = got;
  return fake_ok();
}

/// Installs the scripted cuFile for the guard's lifetime; declare it before the objects and
/// lane sets it serves.
class scripted_cufile {
 public:
  explicit scripted_cufile(bool async_reads = true)
  {
    {
      auto& state = fake();
      std::lock_guard lock(state.mutex);
      state.calls.clear();
      state.script.clear();
      state.every.reset();
      state.keep_alive.clear();
      state.stream_registers   = 0;
      state.stream_deregisters = 0;
    }
    _api.driver_open           = &fake_driver_open;
    _api.driver_get_properties = &fake_get_properties;
    _api.handle_register       = &fake_handle_register;
    _api.handle_deregister     = &fake_handle_deregister;
    _api.read                  = &fake_read;
    _api.buf_register          = &fake_buf_register;
    _api.buf_deregister        = &fake_buf_deregister;
    if (async_reads) {
      _api.read_async          = &fake_read_async;
      _api.has_async_read      = true;
      _api.stream_register     = &fake_stream_register;
      _api.stream_deregister   = &fake_stream_deregister;
      _api.has_stream_register = true;
    }
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&_api);
  }
  ~scripted_cufile()
  {
    cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr);
    std::lock_guard lock(fake().mutex);
    fake().keep_alive.clear();
  }
  scripted_cufile(scripted_cufile const&)            = delete;
  scripted_cufile& operator=(scripted_cufile const&) = delete;

  void script(std::vector<fake_action> actions)
  {
    std::lock_guard lock(fake().mutex);
    fake().script = std::move(actions);
  }
  void every(fake_action action)
  {
    std::lock_guard lock(fake().mutex);
    fake().every = action;
  }
  [[nodiscard]] std::vector<fake_call> calls() const
  {
    std::lock_guard lock(fake().mutex);
    return fake().calls;
  }
  [[nodiscard]] std::size_t stream_registers() const
  {
    std::lock_guard lock(fake().mutex);
    return fake().stream_registers;
  }
  [[nodiscard]] std::size_t stream_deregisters() const
  {
    std::lock_guard lock(fake().mutex);
    return fake().stream_deregisters;
  }

 private:
  cufile_api _api{};
};

/// Disarms the lane failure seams on scope exit.
struct seam_reset {
  seam_reset()                             = default;
  seam_reset(seam_reset const&)            = delete;
  seam_reset& operator=(seam_reset const&) = delete;
  ~seam_reset()
  {
    set_lane_callback_failure_for_testing(0);
    set_lane_copy_failure_for_testing(0);
    cucascade::io::cufile::detail::set_lane_stream_create_failure_for_testing(0);
  }
};

void check_calls_aligned(std::vector<fake_call> const& calls)
{
  for (auto const& call : calls) {
    CHECK(call.buffer_offset == 0);
    CHECK(call.size % block == 0);
    CHECK(call.file_offset % block == 0);
  }
}

}  // namespace

TEST_CASE("fused lanes: one read with several copies lands the file's bytes",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  pattern_file const file{4 * MiB};
  auto const object = open_registered(file);
  waker wake;
  fused_lane_set set{make_options(4, 1 * MiB, &wake)};
  REQUIRE(set.capacity() == 4);
  CHECK(set.free_lanes() == 4);
  CHECK(set.in_flight() == 0);
  CHECK(set.slot_bytes() == 1 * MiB);
  CHECK_FALSE(set.completions_pending());
  CHECK(set.oldest_in_flight() == std::chrono::steady_clock::duration::zero());
  if (cufile_api::get()->has_stream_register && set.registered_streams() != set.capacity()) {
    WARN("cuFileStreamRegister failed: lanes run on unregistered streams");
  }

  device_memory dst{2 * MiB};
  range const window{64 * KiB, 512 * KiB};
  // Unaligned parts inside the aligned window, the last ending at the window's end.
  auto read = make_read(object->cufile_handle(),
                        window,
                        {{window.offset + 5, 1000},
                         {window.offset + 100'000, 77'777},
                         {window.end() - 3000, 3000}},
                        dst.get(),
                        4099,
                        42);
  auto const copies = read.copies;
  REQUIRE(set.submit(std::move(read)));
  CHECK(set.in_flight() == 1);
  CHECK(set.free_lanes() == 3);

  auto const results = reap_n(set, 1);
  REQUIRE(results.size() == 1);
  CHECK(results[0].token == 42);
  CHECK(results[0].outcome == lane_result::status::ok);
  CHECK(results[0].io_result >= static_cast<ssize_t>(window.size));
  CHECK(results[0].resubmits == 0);
  CHECK(results[0].cuda_status == cudaSuccess);
  CHECK(set.in_flight() == 0);
  CHECK(set.free_lanes() == 4);
  CHECK(wake.count.load() >= 1);
  CHECK(copies_match(copies, file));
  // The gaps between the packed copies stay untouched.
  auto const gap = bytes_at(dst.get() + 3 + 1000, 4099);
  CHECK(std::all_of(gap.begin(), gap.end(), [](std::uint8_t b) { return b == 0; }));
}

TEST_CASE("fused lanes: more reads than lanes, round-robin lane use",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  auto const object = open_registered(file);
  fused_lane_set set{make_options(4, 256 * KiB)};
  REQUIRE(set.capacity() == 4);

  SECTION("submit refuses when full; reap frees lanes")
  {
    constexpr std::size_t reads = 12;
    device_memory dst{reads * 256 * KiB};
    std::vector<std::vector<lane_copy>> expected(reads);
    std::size_t next = 0;
    auto make        = [&](std::size_t i) {
      range const window{i * 512 * KiB, 256 * KiB};
      auto read   = make_read(object->cufile_handle(),
                            window,
                            {{window.offset + 1, 100 * KiB}, {window.offset + 150 * KiB, 7}},
                            dst.get() + i * 256 * KiB,
                            3,
                            i);
      expected[i] = read.copies;
      return read;
    };
    while (next < 4) {
      REQUIRE(set.submit(make(next++)));
    }
    CHECK(set.free_lanes() == 0);
    CHECK(set.in_flight() == 4);
    auto refused = make(next);
    CHECK_FALSE(set.submit(std::move(refused)));
    CHECK(refused.copies.size() == 2);  // untouched
    CHECK(refused.token == next);

    std::size_t completed = 0;
    auto const deadline   = std::chrono::steady_clock::now() + 60s;
    while (completed < reads) {
      completed += set.reap([&](lane_result const& r) {
        CHECK(r.outcome == lane_result::status::ok);
        CHECK(copies_match(expected[r.token], file));
        if (next < reads) {
          REQUIRE(set.submit(make(next)));  // refill from inside done
          ++next;
        }
      });
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::sleep_for(100us);
    }
    CHECK(set.in_flight() == 0);
    CHECK(set.free_lanes() == 4);
  }

  SECTION("sequential reads rotate over the lanes")
  {
    device_memory dst{256 * KiB};
    std::vector<unsigned> lanes;
    for (std::size_t i = 0; i < 9; ++i) {
      range const window{i * 256 * KiB, 64 * KiB};
      auto read = make_read(object->cufile_handle(),
                            window,
                            {{window.offset + 10, 5000}},
                            dst.get(),
                            1,
                            i);
      auto const copies = read.copies;
      REQUIRE(set.submit(std::move(read)));
      auto const results = reap_n(set, 1);
      REQUIRE(results.size() == 1);
      CHECK(results[0].outcome == lane_result::status::ok);
      CHECK(copies_match(copies, file));
      lanes.push_back(results[0].lane);
    }
    for (std::size_t i = 1; i < lanes.size(); ++i) {
      CHECK(lanes[i] == (lanes[i - 1] + 1) % 4);
    }
  }
}

/// @p read turned into a held read: its copies are returned for copy_out.
[[nodiscard]] std::vector<lane_copy> make_held(lane_read& read)
{
  auto copies = std::move(read.copies);
  read.copies.clear();
  read.hold         = true;
  read.needed_bytes = 0;
  for (auto const& copy : copies) {
    read.needed_bytes = std::max(read.needed_bytes, copy.offset + copy.bytes - read.window.offset);
  }
  return copies;
}

TEST_CASE("fused lanes: held reads", "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  pattern_file const file{4 * MiB};
  auto const object = open_registered(file);
  waker wake;
  std::optional<fused_lane_set> set;
  set.emplace(make_options(2, 1 * MiB, &wake));
  REQUIRE(set->capacity() == 2);
  device_memory dst{1 * MiB};
  dst.fill(0xAB);
  range const window{256 * KiB, 512 * KiB};
  auto read = make_read(object->cufile_handle(),
                        window,
                        {{window.offset + 3, 200 * KiB}, {window.offset + 300 * KiB, 1000}},
                        dst.get(),
                        5,
                        7);
  auto copies       = make_held(read);
  auto const wanted = copies;
  REQUIRE(set->submit(std::move(read)));

  // Phase one: the read only.  Nothing touches the destination.
  auto results = reap_n(*set, 1);
  REQUIRE(results.size() == 1);
  CHECK(results[0].held);
  CHECK(results[0].outcome == lane_result::status::ok);
  CHECK(results[0].token == 7);
  CHECK(set->held() == 1);
  CHECK(set->active() == 0);
  CHECK(set->in_flight() == 1);
  CHECK(set->free_lanes() == 1);
  CHECK(set->oldest_in_flight() == std::chrono::steady_clock::duration::zero());
  auto const untouched = bytes_at(dst.get(), 1 * MiB);
  CHECK(std::all_of(untouched.begin(), untouched.end(), [](std::uint8_t b) { return b == 0xAB; }));
  // drain returns at once: a held lane has nothing queued.
  std::size_t drained = 0;
  set->drain([&](lane_result const&) { ++drained; });
  CHECK(drained == 0);
  CHECK(set->held() == 1);

  SECTION("copy_out lands the bytes")
  {
    auto const lane = results[0].lane;
    CHECK_FALSE(set->copy_out(lane == 0 ? 1U : 0U, std::vector<lane_copy>{}));  // not held
    REQUIRE(set->copy_out(lane, std::move(copies)));
    CHECK(set->held() == 0);
    results = reap_n(*set, 1);
    CHECK_FALSE(results[0].held);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(results[0].lane == lane);
    CHECK(copies_match(wanted, file));
    CHECK(set->in_flight() == 0);
  }

  SECTION("release drops the data and frees the lane")
  {
    set->release_held(results[0].lane);
    CHECK(set->held() == 0);
    CHECK(set->in_flight() == 0);
    CHECK(set->free_lanes() == 2);
    auto const after = bytes_at(dst.get(), 1 * MiB);
    CHECK(std::all_of(after.begin(), after.end(), [](std::uint8_t b) { return b == 0xAB; }));
  }

  SECTION("the set is destroyed once its held lane is released")
  {
    set->release_held(results[0].lane);
    set.reset();
  }
  CHECK(wake.count.load() >= 1);
}

TEST_CASE("fused lanes: scripted cuFile failures and short reads",
          "[io][cufile][fused_lanes][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile;
  pattern_file const file{4 * MiB};
  auto const object = cufile_reactor::create_io_object(file.path());
  REQUIRE(object->cufile_handle() != nullptr);
  auto options          = make_options(2, 1 * MiB);
  options.max_resubmits = 2;
  waker wake;
  options.wake         = &waker::wake;
  options.wake_context = &wake;
  std::optional<fused_lane_set> set;
  set.emplace(options);
  REQUIRE(set->capacity() == 2);
  CHECK(cufile.stream_registers() == 2);
  device_memory dst{2 * MiB};

  range const window{128 * KiB, 64 * KiB};
  auto const parts = std::vector<std::pair<std::size_t, std::size_t>>{
    {window.offset + 7, 5000}, {window.offset + 9000, 20'000}, {window.offset + 40'000, 25'000}};
  auto read = [&] { return make_read(object->cufile_handle(), window, parts, dst.get(), 11, 9); };

  SECTION("a refused submission is io_failed, reported by reap")
  {
    cufile.script({{fake_action::kind::refuse, 0}});
    auto const before = wake.count.load();
    REQUIRE(set->submit(read()));
    CHECK(set->completions_pending());
    CHECK(wake.count.load() == before + 1);
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(results[0].io_result < 0);
    CHECK(results[0].token == 9);
    CHECK(set->in_flight() == 0);
  }

  SECTION("negative bytes_done is io_failed")
  {
    cufile.script({{fake_action::kind::negative, 0}});
    REQUIRE(set->submit(read()));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(results[0].io_result == -5);
  }

  SECTION("a short read with progress is resubmitted on its lane")
  {
    cufile.script({{fake_action::kind::short_read, 2 * block + 100}});
    auto r            = read();
    auto const copies = r.copies;
    REQUIRE(set->submit(std::move(r)));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(results[0].resubmits == 1);
    CHECK(copies_match(copies, file));
    auto const calls = cufile.calls();
    REQUIRE(calls.size() == 2);
    CHECK(calls[1].file_offset == window.offset + 2 * block);
    CHECK(calls[1].size == window.size - 2 * block);
    CHECK(calls[1].base == calls[0].base);
    CHECK(calls[1].stream == calls[0].stream);
  }

  SECTION("a short read without progress is io_failed")
  {
    cufile.script({{fake_action::kind::short_read, 100}});
    REQUIRE(set->submit(read()));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(results[0].io_result == 100);
    CHECK(cufile.calls().size() == 1);
  }

  SECTION("a short read past the copies' end is ok")
  {
    // Only the copies matter: the window's last bytes are not needed.
    cufile.script({{fake_action::kind::short_read, 40'000 + 25'000}});
    auto r            = read();
    auto const copies = r.copies;
    REQUIRE(set->submit(std::move(r)));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(results[0].resubmits == 0);
    CHECK(copies_match(copies, file));
  }

  SECTION("the resubmit budget runs out")
  {
    cufile.every({fake_action::kind::short_read, block});
    REQUIRE(set->submit(read()));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(results[0].resubmits == 2);
    CHECK(cufile.calls().size() == 3);
  }

  SECTION("a malformed read is io_failed without reaching cuFile")
  {
    auto r          = read();
    r.window.offset = 100;
    REQUIRE(set->submit(std::move(r)));
    auto results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(results[0].io_result == -EINVAL);
    auto big        = read();
    big.window.size = 2 * MiB;
    REQUIRE(set->submit(std::move(big)));
    results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(cufile.calls().empty());
  }

  SECTION("a refused continuation is reported only after the queued copies ran")
  {
    // Short first part, then the continuation is refused.  copy_out queues the first part's
    // copies before the continuation: the lane must not be reported while they may run.
    cufile.script({{fake_action::kind::short_read, 2 * block + 100}, {fake_action::kind::refuse, 0}});
    auto r            = read();
    auto copies       = make_held(r);
    auto const wanted = copies;
    REQUIRE(set->submit(std::move(r)));
    auto results = reap_n(*set, 1);
    REQUIRE(results[0].held);
    // Keep the lane's stream busy for 300 ms on the GPU, so the copies run late.
    void* raw = nullptr;
    REQUIRE(cudaHostAlloc(&raw, sizeof(cucascade::io::cufile::simulator_copy_args),
                          cudaHostAllocMapped) == cudaSuccess);
    auto* delay = static_cast<cucascade::io::cufile::simulator_copy_args*>(raw);
    *delay      = cucascade::io::cufile::simulator_copy_args{};
    delay->released              = 1;
    delay->completion_latency_ns = 300'000'000;
    auto const lane_stream       = static_cast<cudaStream_t>(cufile.calls()[0].stream);
    REQUIRE(cucascade::io::cufile::launch_simulator_gate(delay, lane_stream) == cudaSuccess);
    auto const started = std::chrono::steady_clock::now();
    REQUIRE(set->copy_out(results[0].lane, std::move(copies)));
    results = reap_n(*set, 1);
    CHECK(std::chrono::steady_clock::now() - started >= 250ms);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    // The first part's bytes landed before the lane was reported.
    for (auto const& copy : wanted) {
      auto const end = std::min(copy.offset + copy.bytes, window.offset + 2 * block);
      if (copy.offset >= end) continue;
      auto const got = bytes_at(copy.dst, end - copy.offset);
      CHECK(std::memcmp(got.data(), file.data() + copy.offset, got.size()) == 0);
    }
    REQUIRE(cudaStreamSynchronize(lane_stream) == cudaSuccess);
    REQUIRE(cudaFreeHost(raw) == cudaSuccess);
    CHECK(set->in_flight() == 0);
  }

  SECTION("a held short read continues after copy_out")
  {
    cufile.script({{fake_action::kind::short_read, 2 * block + 100}});
    auto r      = read();
    auto copies = make_held(r);
    auto const wanted = copies;
    REQUIRE(set->submit(std::move(r)));
    auto results = reap_n(*set, 1);
    REQUIRE(results[0].held);
    CHECK(cufile.calls().size() == 1);  // the continuation waits for copy_out
    REQUIRE(set->copy_out(results[0].lane, std::move(copies)));
    results = reap_n(*set, 1);
    CHECK_FALSE(results[0].held);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(results[0].resubmits == 1);
    CHECK(copies_match(wanted, file));
    auto const calls = cufile.calls();
    REQUIRE(calls.size() == 2);
    CHECK(calls[1].file_offset == window.offset + 2 * block);
    CHECK(calls[1].base == calls[0].base);
  }

  SECTION("a held read without progress fails without being held")
  {
    cufile.script({{fake_action::kind::short_read, 100}});
    auto r = read();
    std::ignore = make_held(r);
    REQUIRE(set->submit(std::move(r)));
    auto const results = reap_n(*set, 1);
    CHECK_FALSE(results[0].held);
    CHECK(results[0].outcome == lane_result::status::io_failed);
    CHECK(set->held() == 0);
  }

  SECTION("many copies of one window go out in one call")
  {
    std::vector<std::pair<std::size_t, std::size_t>> many;
    for (std::size_t i = 0; i < 8; ++i) {
      many.emplace_back(window.offset + i * 8000 + 1, 3000);
    }
    auto r            = make_read(object->cufile_handle(), window, many, dst.get(), 9, 9);
    auto const copies = r.copies;
    auto const before = cucascade::io::cufile::detail::lane_copy_calls_for_testing();
    REQUIRE(set->submit(std::move(r)));
    auto const results = reap_n(*set, 1);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(cucascade::io::cufile::detail::lane_copy_calls_for_testing() - before == 1);
    CHECK(copies_match(copies, file));
  }

  // Whatever happened, the set works afterwards.
  cufile.every({fake_action::kind::normal, 0});
  auto r            = read();
  auto const copies = r.copies;
  REQUIRE(set->submit(std::move(r)));
  auto const results = reap_n(*set, 1);
  CHECK(results[0].outcome == lane_result::status::ok);
  CHECK(copies_match(copies, file));
  check_calls_aligned(cufile.calls());
  auto const retired = cucascade::io::cufile::detail::retired_lane_streams_for_testing();
  set.reset();
  // Retired, never deregistered: cuFileStreamDeregister would wait for the whole device.
  CHECK(cufile.stream_deregisters() == 0);
  CHECK(cucascade::io::cufile::detail::retired_lane_streams_for_testing() - retired == 2);
}

TEST_CASE("fused lanes: callback and copy enqueue failures are device_failed",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  seam_reset const reset;
  pattern_file const file{2 * MiB};
  auto const object = open_registered(file);
  auto const mode   = GENERATE(lane_completion::callback, lane_completion::poll);
  auto options      = make_options(1, 512 * KiB);
  options.completion = mode;
  fused_lane_set set{options};
  REQUIRE(set.capacity() == 1);
  device_memory dst{1 * MiB};
  range const window{512 * KiB, 512 * KiB};
  auto read = [&] {
    return make_read(object->cufile_handle(),
                     window,
                     {{window.offset + 1, 1000}, {window.offset + 2000, 3000}, {window.end() - 1, 1}},
                     dst.get());
  };

  SECTION("callback installation") { set_lane_callback_failure_for_testing(1); }
  SECTION("copy enqueue") { set_lane_copy_failure_for_testing(2); }

  REQUIRE(set.submit(read()));
  auto results = reap_n(set, 1);
  CHECK(results[0].outcome == lane_result::status::device_failed);
  CHECK(results[0].cuda_status != cudaSuccess);
  CHECK(set.capacity() == 1);
  CHECK(set.in_flight() == 0);

  dst.fill(0);
  auto again        = read();
  auto const copies = again.copies;
  REQUIRE(set.submit(std::move(again)));
  results = reap_n(set, 1);
  CHECK(results[0].outcome == lane_result::status::ok);
  CHECK(copies_match(copies, file));
}

TEST_CASE("fused lanes: a caller on another device", "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  int devices = 0;
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  if (devices < 2) { SKIP("requires two CUDA devices"); }
  pattern_file const file{2 * MiB};
  auto const object = open_registered(file);
  device_memory dst{1 * MiB};  // on device 0
  REQUIRE(cudaSetDevice(1) == cudaSuccess);
  struct restore_device {
    restore_device()                                 = default;
    restore_device(restore_device const&)            = delete;
    restore_device& operator=(restore_device const&) = delete;
    ~restore_device() { std::ignore = cudaSetDevice(0); }
  } const restore;
  auto const current = [] {
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    return device;
  };
  std::vector<lane_copy> copies;
  {
    fused_lane_set set{make_options(2, 1 * MiB)};
    CHECK(current() == 1);
    REQUIRE(set.capacity() == 2);
    range const window{256 * KiB, 1 * MiB};
    auto read = make_read(object->cufile_handle(), window, {{window.offset + 9, 600 * KiB}}, dst.get());
    copies    = read.copies;
    REQUIRE(set.submit(std::move(read)));
    CHECK(current() == 1);
    auto const results = reap_n(set, 1);
    CHECK(current() == 1);
    CHECK(results[0].outcome == lane_result::status::ok);
    set.drain([](lane_result const&) {});
  }
  CHECK(current() == 1);
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  CHECK(copies_match(copies, file));
}

TEST_CASE("fused lanes: poll mode", "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  pattern_file const file{4 * MiB};
  auto const object  = open_registered(file);
  auto options       = make_options(3, 1 * MiB);
  options.completion = lane_completion::poll;
  fused_lane_set set{options};
  REQUIRE(set.capacity() == 3);
  device_memory dst{3 * MiB};
  std::vector<std::vector<lane_copy>> expected;
  for (std::size_t i = 0; i < 3; ++i) {
    range const window{i * MiB, 1 * MiB};
    auto read = make_read(object->cufile_handle(),
                          window,
                          {{window.offset + 333, 300 * KiB}, {window.offset + 600 * KiB, 123}},
                          dst.get() + i * MiB,
                          5,
                          i);
    expected.push_back(read.copies);
    REQUIRE(set.submit(std::move(read)));
  }
  CHECK(set.completions_pending());
  auto const results = reap_n(set, 3);
  for (auto const& r : results) {
    CHECK(r.outcome == lane_result::status::ok);
    CHECK(copies_match(expected[r.token], file));
  }
  CHECK_FALSE(set.completions_pending());
}

TEST_CASE("fused lanes: the per-device lane budget is shared", "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  auto const base = lanes_in_use_for_testing(0);
  REQUIRE(base == 0);
  auto options             = make_options(3, 64 * KiB);
  options.max_device_lanes = 5;
  std::optional<fused_lane_set> first;
  first.emplace(options);
  CHECK(first->capacity() == 3);
  CHECK(lanes_in_use_for_testing(0) == 3);
  options.lanes = 4;
  {
    fused_lane_set second{options};
    CHECK(second.capacity() == 2);
    CHECK(lanes_in_use_for_testing(0) == 5);
    fused_lane_set third{options};
    CHECK(third.capacity() == 0);
    lane_read read;
    CHECK_FALSE(third.submit(std::move(read)));
  }
  CHECK(lanes_in_use_for_testing(0) == 3);
  first.reset();
  CHECK(lanes_in_use_for_testing(0) == 0);
  fused_lane_set fourth{options};
  CHECK(fourth.capacity() == 4);
  CHECK(lanes_in_use_for_testing(0) == 4);
}

TEST_CASE("fused lanes: drain completes reads in flight", "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  pattern_file const file{8 * MiB};
  auto const object  = open_registered(file);
  auto const mode    = GENERATE(lane_completion::callback, lane_completion::poll);
  auto options       = make_options(8, 1 * MiB);
  options.completion = mode;
  device_memory dst{9 * MiB};
  std::vector<std::vector<lane_copy>> expected;
  std::vector<lane_result> results;
  {
    fused_lane_set set{options};
    REQUIRE(set.capacity() == 8);
    for (std::size_t i = 0; i < 8; ++i) {
      range const window{i * MiB, 1 * MiB};
      auto read = make_read(
        object->cufile_handle(), window, {{window.offset + 1, MiB - 1}}, dst.get() + i * MiB, 0, i);
      expected.push_back(read.copies);
      REQUIRE(set.submit(std::move(read)));
    }
    CHECK(set.in_flight() == 8);
    set.drain([&](lane_result const& r) { results.push_back(r); });
    CHECK(set.in_flight() == 0);
    CHECK(set.free_lanes() == 8);
  }
  REQUIRE(results.size() == 8);
  for (auto const& r : results) {
    CHECK(r.outcome == lane_result::status::ok);
    CHECK(copies_match(expected[r.token], file));
  }
}

TEST_CASE("fused lanes: no lanes without cuFileReadAsync", "[io][cufile][fused_lanes][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  scripted_cufile cufile{/*async_reads=*/false};
  fused_lane_set set{make_options(4, 64 * KiB)};
  CHECK(set.capacity() == 0);
  CHECK(set.free_lanes() == 0);
  CHECK(lanes_in_use_for_testing(0) == 0);
  lane_read read;
  read.window = range{0, block};
  CHECK_FALSE(set.submit(std::move(read)));
  std::size_t reported = 0;
  CHECK(set.reap([&](lane_result const&) { ++reported; }) == 0);
  set.drain([&](lane_result const&) { ++reported; });
  CHECK(reported == 0);
}

TEST_CASE("fused lanes: stress with random windows and copies",
          "[io][cufile][fused_lanes][gpu][stress]")
{
  require_async_cufile();
  constexpr std::size_t file_bytes = 48 * MiB;
  constexpr std::size_t slot       = 1 * MiB;
  constexpr std::size_t lanes      = 16;
  constexpr std::size_t total      = 3000;
  constexpr std::size_t region     = slot + 2 * block;
  pattern_file const file{file_bytes};
  auto const object = open_registered(file);
  waker wake;
  fused_lane_set set{make_options(lanes, slot, &wake)};
  REQUIRE(set.capacity() == lanes);
  device_memory dst{lanes * region};

  std::mt19937_64 rng{20261120};
  auto uniform = [&](std::size_t lo, std::size_t hi) {  // [lo, hi]
    return std::uniform_int_distribution<std::size_t>{lo, hi}(rng);
  };
  std::vector<std::size_t> free_regions;
  for (std::size_t i = 0; i < lanes; ++i) {
    free_regions.push_back(i);
  }
  struct pending {
    std::vector<lane_copy> copies;
    std::size_t region{0};
    std::size_t used{0};
  };
  std::vector<pending> reads(total);
  std::size_t next = 0;
  std::size_t completed = 0;
  std::size_t bytes     = 0;
  std::size_t failures  = 0;

  auto make = [&](std::size_t id) {
    auto const blocks = uniform(1, slot / block);
    auto const start  = uniform(0, file_bytes / block - blocks);
    range const window{start * block, blocks * block};
    auto const n = uniform(1, 4);
    std::vector<std::size_t> cuts;
    for (std::size_t i = 0; i < 2 * n; ++i) {
      cuts.push_back(uniform(window.offset, window.end()));
    }
    std::sort(cuts.begin(), cuts.end());
    auto const index = free_regions.back();
    free_regions.pop_back();
    lane_read read;
    read.handle    = object->cufile_handle();
    read.window    = window;
    read.token     = id;
    std::size_t at = uniform(0, 63);
    for (std::size_t i = 0; i < n; ++i) {
      auto const size = cuts[2 * i + 1] - cuts[2 * i];
      if (size == 0) continue;
      read.copies.push_back(lane_copy{dst.get() + index * region + at, cuts[2 * i], size});
      at += size + uniform(0, 100);
    }
    reads[id] = pending{read.copies, index, at};
    return read;
  };

  auto on_done = [&](lane_result const& r) {
    auto& p = reads[r.token];
    if (r.outcome != lane_result::status::ok) ++failures;
    auto const got = bytes_at(dst.get() + p.region * region, p.used);
    for (auto const& copy : p.copies) {
      auto const* at = got.data() + (copy.dst - (dst.get() + p.region * region));
      if (std::memcmp(at, file.data() + copy.offset, copy.bytes) != 0) ++failures;
      bytes += copy.bytes;
    }
    free_regions.push_back(p.region);
    ++completed;
    // Refill from inside done, as the reactor does.
    if (next < total && set.free_lanes() > 0) { REQUIRE(set.submit(make(next++))); }
  };

  auto const started  = std::chrono::steady_clock::now();
  auto const deadline = started + 120s;
  while (completed < total) {
    while (next < total && set.free_lanes() > 0) {
      REQUIRE(set.submit(make(next++)));
    }
    if (!set.completions_pending()) wake.wait(20ms);
    std::ignore = set.reap(on_done);
    if (std::chrono::steady_clock::now() > deadline) {
      set.drain([](lane_result const&) {});
      FAIL("stress timed out after " << completed << " of " << total << " reads");
    }
  }
  auto const seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  CHECK(failures == 0);
  CHECK(set.in_flight() == 0);
  WARN("fused lanes stress: " << total << " reads, " << bytes / MiB << " MiB copied in "
                              << seconds << " s (" << static_cast<double>(total) / seconds
                              << " reads/s, includes per-read verification)");
}

TEST_CASE("fused lanes: a lane whose stream cannot be re-created leaves the set",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  seam_reset const reset;
  pattern_file const file{2 * MiB};
  auto const object = open_registered(file);
  fused_lane_set set{make_options(2, 512 * KiB)};
  REQUIRE(set.capacity() == 2);
  device_memory dst{1 * MiB};
  range const window{0, 512 * KiB};
  set_lane_callback_failure_for_testing(1);
  cucascade::io::cufile::detail::set_lane_stream_create_failure_for_testing(1);
  REQUIRE(set.submit(make_read(object->cufile_handle(), window, {{5, 1000}}, dst.get())));
  auto const results = reap_n(set, 1);
  CHECK(results[0].outcome == lane_result::status::device_failed);
  CHECK(set.capacity() == 1);
  CHECK(set.in_flight() == 0);
  CHECK(set.free_lanes() == 1);
  // The other lane still works.
  auto read         = make_read(object->cufile_handle(), window, {{7, 3000}}, dst.get());
  auto const copies = read.copies;
  REQUIRE(set.submit(std::move(read)));
  CHECK(reap_n(set, 1)[0].outcome == lane_result::status::ok);
  CHECK(copies_match(copies, file));
}

TEST_CASE("fused lanes: a lane keeps working across files opened and closed",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  fused_lane_set set{make_options(1, 1 * MiB)};
  REQUIRE(set.capacity() == 1);
  device_memory dst{1 * MiB};
  for (int round = 0; round < 4; ++round) {
    INFO("round " << round);
    pattern_file const file{2 * MiB};
    auto object = open_registered(file);
    range const window{256 * KiB, 512 * KiB};
    auto read = make_read(object->cufile_handle(), window, {{window.offset + 3, 100 * KiB}}, dst.get());
    auto const copies = read.copies;
    REQUIRE(set.submit(std::move(read)));
    auto const results = reap_n(set, 1);
    CHECK(results[0].outcome == lane_result::status::ok);
    CHECK(results[0].io_result >= 0);
    CHECK(copies_match(copies, file));
    object.reset();  // cuFileHandleDeregister
  }
}

TEST_CASE("fused lanes: lane sets on threads of their own, one after another",
          "[io][cufile][fused_lanes][gpu]")
{
  require_async_cufile();
  device_memory dst{1 * MiB};
  for (int round = 0; round < 6; ++round) {
    INFO("round " << round);
    pattern_file const file{2 * MiB};
    auto object = open_registered(file);
    std::size_t failures = 0;
    std::size_t mismatches = 0;
    // Each round's set lives on a thread of its own, as each reactor's does.
    std::thread worker{[&] {
      fused_lane_set set{make_options(2, 1 * MiB)};
      for (int i = 0; i < 2; ++i) {
        range const window{256 * KiB, 512 * KiB};
        auto read = make_read(object->cufile_handle(), window, {{window.offset + 3, 100 * KiB}}, dst.get());
        auto const copies = read.copies;
        if (!set.submit(std::move(read))) { ++failures; continue; }
        std::vector<lane_result> results;
        auto const deadline = std::chrono::steady_clock::now() + 30s;
        while (results.empty() && std::chrono::steady_clock::now() < deadline) {
          std::ignore = set.reap([&](lane_result const& r) { results.push_back(r); });
          std::this_thread::sleep_for(100us);
        }
        if (results.empty() || results[0].outcome != lane_result::status::ok) {
          ++failures;
          set.drain([](lane_result const&) {});
          continue;
        }
        for (auto const& copy : copies) {
          std::vector<std::uint8_t> got(copy.bytes);
          std::ignore = cudaMemcpy(got.data(), copy.dst, copy.bytes, cudaMemcpyDeviceToHost);
          if (std::memcmp(got.data(), file.data() + copy.offset, copy.bytes) != 0) ++mismatches;
        }
      }
    }};
    worker.join();
    CHECK(failures == 0);
    CHECK(mismatches == 0);
  }
}
