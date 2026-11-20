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
 * @file
 * Tests of the cuFile batch host reader (src/io/cufile/host_batch.hpp).
 *
 * Content tests run against the real libcufile (compatibility mode is enough) and verify every
 * byte against the file.  Failure paths run against a scripted batch group installed with
 * detail::set_cufile_api_for_testing: either a fake batch (no I/O; the test completes entries)
 * or libcufile with the batch calls intercepted to reject submits, alter events or fail
 * GetStatus.
 */

#include "io/cufile/host_batch.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <cufile.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

using cucascade::io::range;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_reactor;
using cucascade::io::cufile::detail::batch_read;
using cucascade::io::cufile::detail::batch_result;
using cucascade::io::cufile::detail::batch_submit_policy;
using cucascade::io::cufile::detail::host_batch_reader;

namespace {

using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

constexpr std::size_t block = cucascade::io::IO_BLOCK_SIZE;

[[nodiscard]] std::uint8_t pattern_byte(std::size_t offset) noexcept
{
  std::uint64_t z = offset + 0x9E3779B97F4A7C15ULL;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<std::uint8_t>(z ^ (z >> 31));
}

/** A temporary pattern-filled file, kept in memory too for content checks. */
class pattern_file {
 public:
  explicit pattern_file(std::size_t bytes)
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_host_batch_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin")),
      _data(bytes)
  {
    for (std::size_t i = 0; i < bytes; ++i) {
      _data[i] = pattern_byte(i);
    }
    std::ofstream out(_path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(_data.data()),
              static_cast<std::streamsize>(_data.size()));
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
  /// Whether @p size bytes at @p got are the file's bytes at @p offset.
  [[nodiscard]] bool matches(void const* got, std::size_t offset, std::size_t size) const
  {
    return offset + size <= _data.size() && std::memcmp(got, _data.data() + offset, size) == 0;
  }

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

/** 4 KiB-aligned host memory: std::aligned_alloc, or pinned (cudaMallocHost) when asked. */
class host_memory {
 public:
  host_memory(std::size_t bytes, bool pinned) : _bytes(bytes), _pinned(pinned)
  {
    if (pinned) {
      void* p = nullptr;
      REQUIRE(cudaMallocHost(&p, bytes) == cudaSuccess);
      _data = static_cast<std::uint8_t*>(p);
    } else {
      _data = static_cast<std::uint8_t*>(std::aligned_alloc(block, (bytes + block - 1) / block * block));
    }
    REQUIRE(_data != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(_data) % block == 0);
    std::memset(_data, 0xEE, bytes);
  }
  ~host_memory()
  {
    if (_pinned) {
      std::ignore = cudaFreeHost(_data);
    } else {
      std::free(_data);
    }
  }
  host_memory(host_memory const&)            = delete;
  host_memory& operator=(host_memory const&) = delete;

  [[nodiscard]] std::uint8_t* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _bytes; }

 private:
  std::uint8_t* _data{nullptr};
  std::size_t _bytes{0};
  bool _pinned{false};
};

/** The real table with a batch group, or null (the caller skips). */
[[nodiscard]] cufile_api const* real_batch_api()
{
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_batch || !cufile_api::open_driver()) { return nullptr; }
  return api;
}

/** Open @p file through cuFile; null when cuFile cannot register it. */
[[nodiscard]] std::unique_ptr<cufile_io_object> open_registered(pattern_file const& file)
{
  auto object = cufile_reactor::create_io_object(file.path());
  if (!object || object->cufile_handle() == nullptr) { return nullptr; }
  return object;
}

/**
 * Reap until nothing is in flight or pending, collecting results; false at @p deadline.
 * Never leaves this function with cuFile still writing unless the deadline passed.
 */
template <typename Done>
[[nodiscard]] bool reap_all(host_batch_reader& reader, Done&& done, clock_type::duration limit = 20s)
{
  auto const deadline = clock_type::now() + limit;
  while (reader.in_flight() > 0 || reader.pending_failures() > 0) {
    if (clock_type::now() > deadline) { return false; }
    std::ignore = reader.reap(10ms, done);
  }
  return true;
}

// -- Scripted batch group ------------------------------------------------------------------

[[nodiscard]] CUfileError_t status_of(int code) noexcept
{
  CUfileError_t e{};
  e.err = static_cast<CUfileOpError>(code);
  return e;
}

/** Process-wide script state of the intercepted batch group (one scripted reader at a time). */
struct batch_script {
  std::mutex mutex;
  std::condition_variable cv;
  cufile_api real{};
  bool fake{false};

  // Scripted failures.
  int setup_error{0};                     ///< Non-zero: SetUp fails with it.
  int setup_fail_after{0};                ///< SetUp calls that succeed before setup_error applies.
  int reject_submits{0};                  ///< Next submits rejected.
  int reject_code{static_cast<int>(CU_FILE_BATCH_FULL)};
  bool get_status_error{false};           ///< GetStatus fails.
  int fail_events{0};                     ///< Next final real events turned CUFILE_FAILED.
  int short_events{0};                    ///< Next final real events cut to short_ret bytes.
  std::size_t short_ret{block};

  // Observations.
  int setups{0}, submits{0}, rejected{0}, cancels{0}, destroys{0}, get_status_calls{0};
  int submits_while_pending{0};
  std::size_t outstanding{0}, max_outstanding{0}, capacity{0};
  bool params_ok{true};
  bool null_timeout{false};
  bool over_capacity{false};
  bool force_partial{false};       ///< Fake: the next GetStatus returns what is ready at once.
  int partial_returns{0};
  std::vector<std::size_t> submit_sizes;  ///< Entries per accepted submit.

  // Fake batch: submitted and completed entries.
  struct fake_entry {
    void* cookie;
    std::size_t size;
    CUfileBatchHandle_t handle;
    std::chrono::steady_clock::time_point submitted;
  };
  std::deque<fake_entry> pending;
  std::deque<std::pair<CUfileBatchHandle_t, CUfileIOEvents_t>> ready;
  int next_handle{1};

  /// Back to a clean script serving @p table's batch calls (fake: none).
  void reset(bool fake_mode, cufile_api const& table)
  {
    real                  = table;
    fake                  = fake_mode;
    setup_error           = 0;
    setup_fail_after      = 0;
    reject_submits        = 0;
    reject_code           = static_cast<int>(CU_FILE_BATCH_FULL);
    get_status_error      = false;
    fail_events           = 0;
    short_events          = 0;
    short_ret             = block;
    setups                = 0;
    submits               = 0;
    rejected              = 0;
    cancels               = 0;
    destroys              = 0;
    get_status_calls      = 0;
    submits_while_pending = 0;
    outstanding           = 0;
    max_outstanding       = 0;
    capacity              = 0;
    params_ok             = true;
    null_timeout          = false;
    over_capacity         = false;
    force_partial         = false;
    partial_returns       = 0;
    submit_sizes.clear();
    pending.clear();
    ready.clear();
    next_handle = 1;
  }
};

batch_script& script()
{
  static batch_script state;
  return state;
}

CUfileError_t scripted_driver_open() { return status_of(CU_FILE_SUCCESS); }

CUfileError_t scripted_setup(CUfileBatchHandle_t* handle, unsigned nr)
{
  auto& s = script();
  {
    std::lock_guard lock(s.mutex);
    ++s.setups;
    if (s.setup_error != 0) {
      if (s.setup_fail_after > 0) {
        --s.setup_fail_after;
      } else {
        return status_of(s.setup_error);
      }
    }
    s.capacity = nr;
    if (s.fake) {
      *handle = reinterpret_cast<CUfileBatchHandle_t>(static_cast<std::uintptr_t>(s.next_handle++));
      return status_of(CU_FILE_SUCCESS);
    }
  }
  return s.real.batch_setup(handle, nr);
}

CUfileError_t scripted_submit(CUfileBatchHandle_t handle,
                              unsigned nr,
                              CUfileIOParams_t* params,
                              unsigned int flags)
{
  auto& s = script();
  {
    std::lock_guard lock(s.mutex);
    ++s.submits;
    for (unsigned i = 0; i < nr; ++i) {
      auto const& p = params[i].u.batch;
      if (params[i].mode != CUFILE_BATCH || params[i].opcode != CUFILE_READ ||
          p.devPtr_offset != 0 || reinterpret_cast<std::uintptr_t>(p.devPtr_base) % block != 0 ||
          static_cast<std::size_t>(p.file_offset) % block != 0 || p.size % block != 0 ||
          p.size == 0 || params[i].cookie == nullptr) {
        s.params_ok = false;
      }
    }
    if (s.reject_submits > 0) {
      --s.reject_submits;
      ++s.rejected;
      return status_of(s.reject_code);
    }
    if (s.outstanding > 0) { ++s.submits_while_pending; }
    if (s.outstanding + nr > s.capacity) { s.over_capacity = true; }
    s.submit_sizes.push_back(nr);
    if (s.fake) {
      for (unsigned i = 0; i < nr; ++i) {
        s.pending.push_back(
          {params[i].cookie, params[i].u.batch.size, handle, std::chrono::steady_clock::now()});
      }
      s.outstanding += nr;
      s.max_outstanding = std::max(s.max_outstanding, s.outstanding);
      s.cv.notify_all();
      return status_of(CU_FILE_SUCCESS);
    }
  }
  auto const status = s.real.batch_submit(handle, nr, params, flags);
  if (status.err == CU_FILE_SUCCESS) {
    std::lock_guard lock(s.mutex);
    s.outstanding += nr;
    s.max_outstanding = std::max(s.max_outstanding, s.outstanding);
  }
  return status;
}

[[nodiscard]] bool final_status(CUfileStatus_t status) noexcept
{
  return status != CUFILE_WAITING && status != CUFILE_PENDING;
}

CUfileError_t scripted_get_status(CUfileBatchHandle_t handle,
                                  unsigned min_nr,
                                  unsigned* nr,
                                  CUfileIOEvents_t* events,
                                  timespec* timeout)
{
  auto& s = script();
  {
    std::unique_lock lock(s.mutex);
    ++s.get_status_calls;
    if (timeout == nullptr) { s.null_timeout = true; }
    if (s.get_status_error) { return status_of(CU_FILE_INTERNAL_ERROR); }
    if (s.fake) {
      auto const wait = timeout == nullptr
                          ? std::chrono::nanoseconds{5s}
                          : std::chrono::seconds{timeout->tv_sec} +
                              std::chrono::nanoseconds{timeout->tv_nsec};
      // A test that failed before completing its entries must not hang in the reader's
      // destructor: entries pending for 10 s complete as cancelled (checked while waiting).
      auto expire = [&] {
        auto const now = std::chrono::steady_clock::now();
        for (auto it = s.pending.begin(); it != s.pending.end();) {
          if (it->handle == handle && now - it->submitted > 10s) {
            CUfileIOEvents_t event{};
            event.cookie = it->cookie;
            event.status = CUFILE_CANCELED;
            s.ready.emplace_back(handle, event);
            it = s.pending.erase(it);
          } else {
            ++it;
          }
        }
      };
      auto ready_for_handle = [&] {
        return static_cast<std::size_t>(std::count_if(
          s.ready.begin(), s.ready.end(), [&](auto const& e) { return e.first == handle; }));
      };
      auto const deadline = std::chrono::steady_clock::now() + wait;
      for (;;) {
        expire();
        if (ready_for_handle() >= min_nr || s.force_partial) break;
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        s.cv.wait_for(lock, std::min<std::chrono::nanoseconds>(deadline - now, 100ms));
      }
      if (s.force_partial) {
        s.force_partial = false;
        ++s.partial_returns;
      }
      unsigned count = 0;
      for (auto it = s.ready.begin(); it != s.ready.end() && count < *nr;) {
        if (it->first != handle) {
          ++it;
          continue;
        }
        events[count++] = it->second;
        it              = s.ready.erase(it);
      }
      s.outstanding -= count;
      *nr = count;
      return status_of(CU_FILE_SUCCESS);
    }
  }
  auto const status = s.real.batch_get_status(handle, min_nr, nr, events, timeout);
  if (status.err == CU_FILE_SUCCESS) {
    std::lock_guard lock(s.mutex);
    for (unsigned i = 0; i < *nr; ++i) {
      if (!final_status(events[i].status)) { continue; }
      --s.outstanding;
      if (events[i].status != CUFILE_COMPLETE) { continue; }
      if (s.fail_events > 0) {
        --s.fail_events;
        events[i].status = CUFILE_FAILED;
      } else if (s.short_events > 0) {
        --s.short_events;
        events[i].ret = std::min(events[i].ret, s.short_ret);
      }
    }
  }
  return status;
}

CUfileError_t scripted_cancel(CUfileBatchHandle_t handle)
{
  auto& s = script();
  {
    std::lock_guard lock(s.mutex);
    ++s.cancels;
    if (s.fake) {
      auto const before = s.pending.size();
      std::erase_if(s.pending, [&](auto const& e) { return e.handle == handle; });
      s.outstanding -= before - s.pending.size();
      return status_of(CU_FILE_SUCCESS);
    }
  }
  return s.real.batch_cancel(handle);
}

void scripted_destroy(CUfileBatchHandle_t handle)
{
  auto& s = script();
  {
    std::lock_guard lock(s.mutex);
    ++s.destroys;
    if (s.fake) {
      auto const before = s.pending.size() + s.ready.size();
      std::erase_if(s.pending, [&](auto const& e) { return e.handle == handle; });
      std::erase_if(s.ready, [&](auto const& e) { return e.first == handle; });
      s.outstanding -= before - s.pending.size() - s.ready.size();
      return;
    }
  }
  s.real.batch_destroy(handle);
}

/**
 * Installs a cuFile table whose batch group is scripted, for its lifetime.  @c fake: every
 * batch call is served without I/O (tests complete entries with @ref complete); otherwise the
 * batch calls go to libcufile through the interceptors.
 */
class scripted_batch {
 public:
  explicit scripted_batch(bool fake, bool has_batch = true)
  {
    auto& s = script();
    if (fake) {
      _api.driver_open = &scripted_driver_open;
    } else {
      auto const* real = cufile_api::get();
      REQUIRE(real != nullptr);
      _api = *real;
    }
    {
      std::lock_guard lock(s.mutex);
      s.reset(fake, _api);
    }
    _api.batch_setup      = &scripted_setup;
    _api.batch_submit     = &scripted_submit;
    _api.batch_get_status = &scripted_get_status;
    _api.batch_cancel     = &scripted_cancel;
    _api.batch_destroy    = &scripted_destroy;
    _api.has_batch        = has_batch;
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&_api);
  }
  ~scripted_batch() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  scripted_batch(scripted_batch const&)            = delete;
  scripted_batch& operator=(scripted_batch const&) = delete;

  template <typename F>
  auto with(F&& f) const
  {
    auto& s = script();
    std::lock_guard lock(s.mutex);
    return f(s);
  }

  /// Fake mode: complete the @p n oldest pending entries with @p status (ret: the size, or
  /// @p ret when given).
  static constexpr std::size_t all = std::numeric_limits<std::size_t>::max();

  /// Fake mode: wait (bounded) until @p n entries were submitted and are pending.
  [[nodiscard]] bool wait_pending(std::size_t n) const
  {
    auto& s = script();
    std::unique_lock lock(s.mutex);
    return s.cv.wait_for(lock, 5s, [&] { return s.pending.size() >= n; });
  }

  void complete(std::size_t n, CUfileStatus_t status = CUFILE_COMPLETE, std::optional<std::size_t> ret = {})
  {
    auto& s = script();
    {
      std::unique_lock lock(s.mutex);
      // The reader's helper thread submits asynchronously: wait (bounded) for the entries.
      if (n != all) s.cv.wait_for(lock, 5s, [&] { return s.pending.size() >= n; });
      for (std::size_t i = 0; i < n && !s.pending.empty(); ++i) {
        auto const entry = s.pending.front();
        s.pending.pop_front();
        CUfileIOEvents_t event{};
        event.cookie = entry.cookie;
        event.status = status;
        event.ret    = ret.value_or(entry.size);
        s.ready.emplace_back(entry.handle, event);
      }
    }
    s.cv.notify_all();
  }

 private:
  cufile_api _api{};
};

/// A fully aligned fake read (fake mode does no I/O; the destination is never written).
[[nodiscard]] batch_read fake_read(std::uint8_t* dst, std::size_t index, std::size_t size = block)
{
  return batch_read{reinterpret_cast<void*>(std::uintptr_t{0x1000}), dst, range{index * size, size}, index};
}

struct result_log {
  std::vector<batch_result> results;
  void operator()(batch_result const& r) { results.push_back(r); }
  [[nodiscard]] std::size_t count(batch_result::status s) const
  {
    return static_cast<std::size_t>(
      std::count_if(results.begin(), results.end(), [&](auto const& r) { return r.outcome == s; }));
  }
};

}  // namespace

// -- Real libcufile ---------------------------------------------------------------------------

TEST_CASE("host batch reader reads several entries into distinct destinations",
          "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  bool const pinned = GENERATE(false, true);
  if (pinned && !has_cuda_device()) { SKIP("pinned memory requires a CUDA device"); }

  constexpr std::size_t entry_bytes = 256UL << 10;
  constexpr std::size_t entries     = 12;
  pattern_file const file{entries * entry_bytes + 3 * block};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  host_memory memory(entries * entry_bytes, pinned);

  host_batch_reader reader(16, 4);
  REQUIRE(reader.available());
  REQUIRE(reader.capacity() > 0);
  auto const capacity = reader.capacity();
  REQUIRE(capacity >= entries);
  CHECK(reader.free_entries() == capacity);

  // Reverse file order into the destinations, so a mix-up shows.
  for (std::size_t i = 0; i < entries; ++i) {
    auto const offset = (entries - 1 - i) * entry_bytes;
    reader.add(batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{offset, entry_bytes}, i});
  }
  CHECK(reader.staged() == entries);
  CHECK(reader.in_flight() == 0);
  CHECK(reader.free_entries() == capacity - entries);
  reader.flush();
  CHECK(reader.staged() == 0);
  CHECK(reader.in_flight() == entries);
  CHECK(reader.free_entries() == capacity - entries);

  result_log log;
  REQUIRE(reap_all(reader, log));
  REQUIRE(log.results.size() == entries);
  std::vector<bool> seen(entries, false);
  for (auto const& r : log.results) {
    REQUIRE(r.token < entries);
    CHECK_FALSE(seen[r.token]);
    seen[r.token] = true;
    CHECK(r.outcome == batch_result::status::ok);
    CHECK(r.bytes == entry_bytes);
    CHECK(r.error == 0);
  }
  for (std::size_t i = 0; i < entries; ++i) {
    INFO("entry " << i);
    CHECK(file.matches(memory.data() + i * entry_bytes, (entries - 1 - i) * entry_bytes, entry_bytes));
  }
  CHECK(reader.in_flight() == 0);
  CHECK(reader.free_entries() == capacity);
  CHECK(reader.counters().submits == 1);
  CHECK(reader.counters().completed == entries);
  CHECK(reader.counters().bytes == entries * entry_bytes);

  // A read ending exactly at the file's last whole block.
  auto const body_end = file.size() / block * block;
  reader.add(batch_read{object->cufile_handle(), memory.data(), range{body_end - 2 * block, 2 * block}, 99});
  reader.flush();
  log.results.clear();
  REQUIRE(reap_all(reader, log));
  REQUIRE(log.results.size() == 1);
  CHECK(log.results[0].outcome == batch_result::status::ok);
  CHECK(file.matches(memory.data(), body_end - 2 * block, 2 * block));
}

TEST_CASE("host batch reader refills entries as they complete", "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{16UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }

  constexpr unsigned depth          = 8;
  constexpr std::size_t reads       = 400;
  constexpr std::size_t entry_bytes = 64UL << 10;
  host_memory memory(depth * entry_bytes, false);
  scripted_batch const intercept(false);
  {
    host_batch_reader reader(depth, 4);
    REQUIRE(reader.available());
    REQUIRE(reader.capacity() == depth);
    REQUIRE(reader.policy() == batch_submit_policy::per_reader);  // the default

    // Each read goes into a destination slot a completed read freed.
    std::vector<std::size_t> free_slots;
    for (std::size_t i = 0; i < depth; ++i) {
      free_slots.push_back(i);
    }
    std::vector<std::size_t> slot_of(reads);
    std::vector<std::size_t> offset_of(reads);
    std::size_t issued = 0, done_count = 0, bad = 0, max_used = 0;
    std::mt19937_64 rng(7);
    auto issue = [&] {
      while (issued < reads && reader.free_entries() > 0 && !free_slots.empty()) {
        auto const slot = free_slots.back();
        free_slots.pop_back();
        auto const offset = (rng() % (file.size() / block - entry_bytes / block)) * block;
        slot_of[issued]   = slot;
        offset_of[issued] = offset;
        reader.add(batch_read{object->cufile_handle(), memory.data() + slot * entry_bytes, range{offset, entry_bytes}, issued});
        ++issued;
      }
      max_used = std::max(max_used, reader.in_flight() + reader.staged());
    };
    auto done = [&](batch_result const& r) {
      ++done_count;
      auto const slot = slot_of[r.token];
      if (r.outcome != batch_result::status::ok ||
          !file.matches(memory.data() + slot * entry_bytes, offset_of[r.token], entry_bytes)) {
        ++bad;
      }
      std::memset(memory.data() + slot * entry_bytes, 0xEE, entry_bytes);
      free_slots.push_back(slot);
    };

    auto const deadline = clock_type::now() + 60s;
    issue();
    reader.flush();
    while (done_count < reads && clock_type::now() < deadline) {
      std::ignore = reader.reap(5ms, done);
      issue();  // refilled once the wave was reported whole
      reader.flush();
    }
    reader.drain(done);
    CHECK(done_count == reads);
    CHECK(bad == 0);
    CHECK(max_used <= depth);
    CHECK(reader.counters().completed == reads);
    CHECK(reader.counters().submits > 1);
  }
  intercept.with([&](batch_script& s) {
    CHECK(s.max_outstanding <= depth);
    CHECK_FALSE(s.over_capacity);
    CHECK(s.submits_while_pending == 0);  // exclusive: whole waves only
    CHECK(s.params_ok);
    CHECK_FALSE(s.null_timeout);
    return 0;
  });
}

TEST_CASE("host batch reader submits while entries are pending under the continuous policy",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  constexpr unsigned depth    = 4;
  constexpr std::size_t reads = 200;
  host_memory memory(depth * block, false);
  host_batch_reader reader(depth, 4, batch_submit_policy::continuous);
  REQUIRE(reader.available());
  REQUIRE(reader.capacity() == depth);
  std::size_t issued = 0, done_count = 0, ok = 0;
  std::mt19937_64 rng(3);
  auto const deadline = clock_type::now() + 20s;
  while (done_count < reads && clock_type::now() < deadline) {
    while (issued < reads && reader.free_entries() > 0) {
      reader.add(fake_read(memory.data(), issued));
      ++issued;
    }
    CHECK(reader.in_flight() + reader.staged() <= depth);
    reader.flush();
    CHECK(reader.staged() == 0);  // never deferred
    fake.complete(1 + rng() % 2);
    done_count += reader.reap(1ms, [&](batch_result const& r) {
      if (r.outcome == batch_result::status::ok) { ++ok; }
    });
  }
  CHECK(done_count == reads);
  CHECK(ok == reads);
  CHECK(reader.counters().deferred_flushes == 0);
  fake.with([&](batch_script& s) {
    CHECK(s.submits_while_pending > 0);
    CHECK(s.max_outstanding <= depth);
    CHECK_FALSE(s.over_capacity);
    CHECK(s.params_ok);
    return 0;
  });
}

TEST_CASE("host batch readers take turns under the exclusive policy", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  std::atomic<int> b_woken{0};
  host_batch_reader a(4, 4, batch_submit_policy::exclusive);
  host_batch_reader b(
    4,
    4,
    batch_submit_policy::exclusive,
    [](void* context) noexcept { static_cast<std::atomic<int>*>(context)->fetch_add(1); },
    &b_woken);
  REQUIRE(a.available());
  REQUIRE(b.available());
  result_log log_a, log_b;

  a.add(fake_read(memory.data(), 0));
  a.add(fake_read(memory.data() + block, 1));
  a.flush();
  CHECK(a.in_flight() == 2);
  REQUIRE(fake.wait_pending(2));

  // Another reader's wave is in flight: b's flush is deferred.
  b.add(fake_read(memory.data() + 4 * block, 10));
  b.flush();
  CHECK(b.in_flight() == 0);
  CHECK(b.staged() == 1);
  CHECK(b.deferred());
  CHECK(b.counters().deferred_flushes == 1);
  auto t0 = clock_type::now();
  CHECK(b.reap(50ms, log_b) == 0);  // nothing of b in flight: returns at once
  CHECK(clock_type::now() - t0 < 20ms);
  CHECK(b.staged() == 1);

  // a's own refill waits for its wave too.
  a.add(fake_read(memory.data() + 2 * block, 2));
  a.flush();
  CHECK(a.in_flight() == 2);
  CHECK(a.staged() == 1);

  auto const woken_before = b_woken.load();
  fake.complete(2);
  CHECK(a.reap(5s, log_a) == 2);
  CHECK(a.staged() == 1);  // reap never hands anything over
  CHECK(a.in_flight() == 0);
  // a's helper released the gate and woke the waiting b.
  auto const deadline = clock_type::now() + 5s;
  while (b_woken.load() == woken_before && clock_type::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  CHECK(b_woken.load() > woken_before);
  b.flush();
  CHECK(b.in_flight() == 1);
  CHECK(b.staged() == 0);
  a.flush();
  CHECK(a.staged() == 1);  // b's wave is in flight
  REQUIRE(fake.wait_pending(1));
  fake.complete(1);
  REQUIRE(reap_all(b, log_b, 5s));
  CHECK(log_b.results.size() == 1);
  a.flush();  // b's wave completed: a takes the gate
  CHECK(a.in_flight() == 1);
  fake.complete(1);
  REQUIRE(reap_all(a, log_a, 5s));
  fake.with([](batch_script& s) {
    CHECK(s.submits == 3);
    CHECK(s.submits_while_pending == 0);
    return 0;
  });
  CHECK(log_a.results.size() == 3);
}

TEST_CASE("host batch reader drains entries in flight before destruction",
          "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{32UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }

  constexpr std::size_t entry_bytes = 1UL << 20;
  host_memory memory(32 * entry_bytes, false);
  result_log log;
  {
    host_batch_reader reader(32, 4);
    REQUIRE(reader.available());
    auto const n = std::min<std::size_t>(reader.capacity(), 32);
    for (std::size_t i = 0; i < n; ++i) {
      reader.add(batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{i * entry_bytes, entry_bytes}, i});
    }
    reader.flush();
    // One more staged, never flushed by the test: drain() submits and drains it too.
    REQUIRE(n >= 2);
    auto const start = clock_type::now();
    reader.drain(log);
    CHECK(clock_type::now() - start < 30s);
    CHECK(reader.in_flight() == 0);
    CHECK(reader.staged() == 0);
    REQUIRE(log.results.size() == n);
    CHECK(log.count(batch_result::status::ok) == n);
    for (std::size_t i = 0; i < n; ++i) {
      CHECK(file.matches(memory.data() + i * entry_bytes, i * entry_bytes, entry_bytes));
    }
    // Staged-only drain.
    reader.add(batch_read{object->cufile_handle(), memory.data(), range{0, block}, 77});
    log.results.clear();
    reader.drain(log);
    REQUIRE(log.results.size() == 1);
    CHECK(log.results[0].token == 77);
  }  // destroyed idle
}

TEST_CASE("host batch reader polls without blocking on real entries", "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{64UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  constexpr std::size_t entry_bytes = 4UL << 20;
  host_memory memory(16 * entry_bytes, false);
  host_batch_reader reader(16, 4);
  REQUIRE(reader.available());
  auto const n = std::min<std::size_t>(reader.capacity(), 16);
  for (std::size_t i = 0; i < n; ++i) {
    reader.add(batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{i * entry_bytes, entry_bytes}, i});
  }
  reader.flush();
  result_log log;
  auto const start = clock_type::now();
  std::ignore      = reader.reap(0us, log);
  auto const poll  = clock_type::now() - start;
  // Generous: a poll must not wait for the reads (they take tens of ms), only for cuFile's
  // own bookkeeping.
  CHECK(poll < 20ms);
  REQUIRE(reap_all(reader, log));
  CHECK(log.count(batch_result::status::ok) == n);
  // Nothing in flight: a waiting reap returns at once.
  auto const idle_start = clock_type::now();
  CHECK(reader.reap(500ms, log) == 0);
  CHECK(clock_type::now() - idle_start < 50ms);
}

namespace {

struct stress_outcome {
  bool available{false};
  std::size_t reads{0}, bad{0}, failed{0}, max_used{0}, capacity{0}, bytes{0}, submits{0};
  bool timed_out{false};
  double seconds{0};
};

/**
 * @p reads reads of log-uniform aligned sizes in [4 KiB, @p max_entry] at random aligned offsets
 * of @p file through one reader, refilled as entries free, each verified against the file.  No
 * Catch2 assertions (runs on helper threads too).
 */
[[nodiscard]] stress_outcome run_stress(pattern_file const& file,
                                        void* handle,
                                        std::size_t reads,
                                        unsigned depth,
                                        std::size_t max_entry,
                                        std::uint64_t seed,
                                        batch_submit_policy policy,
                                        bool pinned)
{
  stress_outcome out;
  host_memory memory(depth * max_entry, pinned);
  host_batch_reader reader(depth, 4, policy);
  out.available = reader.available();
  if (!out.available) { return out; }
  out.capacity = reader.capacity();

  std::mt19937_64 rng(seed);
  auto next_size = [&] {
    auto const exp    = rng() % 11;  // 2^0 .. 2^10 blocks
    auto const blocks = std::size_t{1} << exp;
    auto const extra  = blocks > 1 ? rng() % blocks : 0;
    return std::min(max_entry, (blocks + extra) * block);
  };
  struct pending_read {
    std::size_t slot, offset, size;
  };
  std::vector<pending_read> info(reads);
  std::vector<std::size_t> free_slots;
  for (std::size_t i = 0; i < out.capacity; ++i) {
    free_slots.push_back(i);
  }
  std::size_t issued = 0;
  auto done          = [&](batch_result const& r) {
    ++out.reads;
    auto const& p = info[r.token];
    if (r.outcome != batch_result::status::ok) {
      ++out.failed;
    } else if (r.bytes != p.size ||
               !file.matches(memory.data() + p.slot * max_entry, p.offset, p.size)) {
      ++out.bad;
    }
    out.bytes += p.size;
    // Sentinels: a stale buffer would not match the next read.
    memory.data()[p.slot * max_entry]              = 0xEE;
    memory.data()[p.slot * max_entry + p.size - 1] = 0xEE;
    free_slots.push_back(p.slot);
  };
  auto const start    = clock_type::now();
  auto const deadline = start + 120s;
  while (out.reads < reads) {
    if (clock_type::now() > deadline) {
      out.timed_out = true;
      break;
    }
    while (issued < reads && reader.free_entries() > 0 && !free_slots.empty()) {
      auto const slot = free_slots.back();
      free_slots.pop_back();
      auto const size   = next_size();
      auto const offset = (rng() % ((file.size() - size) / block + 1)) * block;
      info[issued]      = {slot, offset, size};
      reader.add(batch_read{handle, memory.data() + slot * max_entry, range{offset, size}, issued});
      ++issued;
    }
    reader.flush();
    out.max_used = std::max(out.max_used, reader.in_flight() + reader.staged());
    std::ignore  = reader.reap(2ms, done);
  }
  reader.drain(done);  // never leaves cuFile writing into `memory`
  out.seconds = std::chrono::duration<double>(clock_type::now() - start).count();
  out.submits = reader.counters().submits;
  return out;
}

}  // namespace

TEST_CASE("host batch reader stress: mixed aligned sizes with content checks",
          "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{64UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  constexpr std::size_t reads = 3000;
  auto const out = run_stress(file, object->cufile_handle(), reads, 64, 4UL << 20, 20260423,
                              batch_submit_policy::exclusive, has_cuda_device());
  REQUIRE(out.available);
  CHECK_FALSE(out.timed_out);
  CHECK(out.reads == reads);
  CHECK(out.bad == 0);
  CHECK(out.failed == 0);
  CHECK(out.max_used <= out.capacity);
  WARN("host batch stress: " << out.reads << " reads, " << (out.bytes >> 20) << " MiB in "
                             << out.seconds << " s ("
                             << static_cast<double>(out.bytes) / out.seconds / 1e9
                             << " GB/s), " << out.submits << " submits");
}

TEST_CASE("host batch readers on several threads stay exclusive", "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{64UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  scripted_batch const intercept(false);
  constexpr std::size_t threads = 3;
  constexpr std::size_t reads   = 800;
  std::vector<stress_outcome> outcomes(threads);
  {
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < threads; ++t) {
      workers.emplace_back([&, t] {
        outcomes[t] = run_stress(file, object->cufile_handle(), reads, 32, 1UL << 20, 100 + t,
                                 batch_submit_policy::exclusive, false);
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }
  for (auto const& out : outcomes) {
    REQUIRE(out.available);
    CHECK_FALSE(out.timed_out);
    CHECK(out.reads == reads);
    CHECK(out.bad == 0);
    CHECK(out.failed == 0);
  }
  intercept.with([](batch_script& s) {
    CHECK(s.submits_while_pending == 0);
    CHECK(s.params_ok);
    return 0;
  });
}

/// The workload that crashes libcufile 1.17.1 (compatibility mode) when submits overlap entries
/// in flight: three readers on three threads, each doing 1500 mixed reads of up to 4 MiB at
/// depth 32, with the policy named by CUCASCADE_HOST_BATCH_CRASH_CHILD.  Run in child processes
/// by the test below.
TEST_CASE("host batch submit-policy crash probe (child)", "[.][host_batch_crash_child]")
{
  auto const* name = std::getenv("CUCASCADE_HOST_BATCH_CRASH_CHILD");
  if (name == nullptr) { SKIP("child process only"); }
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  if (std::string{name} == "raw_partial") {
    // The library bug itself, through the raw API: a GetStatus that returns fewer events than
    // are in flight, then any allocation on this thread reusing the chunk cuFile freed, then the
    // next GetStatus.
    auto const* api   = cufile_api::get();
    pattern_file const file{64UL << 20};
    auto const object = open_registered(file);
    if (!object) { SKIP("cuFile cannot register the test file"); }
    host_memory memory(32UL << 20, false);
    CUfileBatchHandle_t handle{nullptr};
    REQUIRE(api->batch_setup(&handle, 8).err == CU_FILE_SUCCESS);
    std::vector<void*> spray;
    for (int round = 0; round < 50; ++round) {
      std::array<CUfileIOParams_t, 4> params{};
      std::array<std::size_t, 4> const sizes{block, 4UL << 20, 8UL << 20, 1UL << 20};
      for (std::size_t i = 0; i < 4; ++i) {
        params[i].mode                  = CUFILE_BATCH;
        params[i].opcode                = CUFILE_READ;
        params[i].fh                    = object->cufile_handle();
        params[i].u.batch.devPtr_base   = memory.data() + i * (8UL << 20);
        params[i].u.batch.file_offset   = static_cast<off_t>(((static_cast<std::size_t>(round) * 4 + i) % 6) * (8UL << 20));
        params[i].u.batch.size          = sizes[i];
        params[i].cookie                = reinterpret_cast<void*>(i + 1);
      }
      REQUIRE(api->batch_submit(handle, 4, params.data(), 0).err == CU_FILE_SUCCESS);
      unsigned done = 0;
      std::array<CUfileIOEvents_t, 8> events{};
      while (done < 4) {
        unsigned nr = 8;
        timespec timeout{2, 0};
        REQUIRE(api->batch_get_status(handle, 1, &nr, events.data(), &timeout).err == CU_FILE_SUCCESS);
        done += nr;
        if (done < 4) {
          for (std::size_t size = 16; size <= 2048; size += 16) {
            auto* chunk = static_cast<std::uint64_t*>(std::malloc(size));
            std::fill(chunk, chunk + size / 8, std::uint64_t{0x40000});
            spray.push_back(chunk);
          }
        }
      }
    }
    api->batch_destroy(handle);
    for (auto* chunk : spray) std::free(chunk);
    return;
  }
  auto const policy = std::string{name} == "continuous"   ? batch_submit_policy::continuous
                      : std::string{name} == "per_reader" ? batch_submit_policy::per_reader
                                                          : batch_submit_policy::exclusive;
  pattern_file const file{64UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  constexpr std::size_t threads = 3;
  std::vector<stress_outcome> outcomes(threads);
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      outcomes[t] = run_stress(
        file, object->cufile_handle(), 1500, 32, 4UL << 20, 1 + t, policy, has_cuda_device());
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (auto const& out : outcomes) {
    CHECK(out.bad == 0);
  }
}

TEST_CASE("cuFile batch crashes when submits overlap entries in flight (why exclusive)",
          "[.][host_batch_quirk]")
{
  // Qualifies the submit policies on this stack (see batch_submit_policy): runs the probe above
  // in child processes per policy, plus the raw-API reproducer of the libcufile bug the wave
  // policies avoid ("raw_partial": GetStatus returning part of what is in flight, then heap
  // reuse), and reports how many died.  Hidden: a child may dump core.  Never fails on a crash
  // count; fails only when a child hangs.
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  auto const policy =
    GENERATE(as<std::string>{}, "exclusive", "per_reader", "continuous", "raw_partial");
  constexpr int runs = 5;
  int crashed = 0, passed = 0, failed = 0;
  for (int run = 0; run < runs; ++run) {
    cucascade::test::child_test_runner child("host batch submit-policy crash probe (child)",
                                             {{"CUCASCADE_HOST_BATCH_CRASH_CHILD", policy}});
    if (!child.wait_for(std::chrono::milliseconds{180'000})) {
      child.stop();
      FAIL("crash probe child hung (policy " << policy << ")");
    }
    if (WIFSIGNALED(child.status())) {
      ++crashed;
    } else if (WIFEXITED(child.status()) && WEXITSTATUS(child.status()) == 0) {
      ++passed;
    } else {
      ++failed;
    }
  }
  WARN("libcufile " << cufile_api::version() << ", policy " << policy << ": " << crashed << " of "
                    << runs << " runs crashed, " << failed << " failed, " << passed << " passed");
}

TEST_CASE("host batch readers under per_reader submit waves independently",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  host_batch_reader a(4, 4, batch_submit_policy::per_reader);
  host_batch_reader b(4, 4, batch_submit_policy::per_reader);
  REQUIRE(a.policy() == batch_submit_policy::per_reader);
  REQUIRE(b.policy() == batch_submit_policy::per_reader);
  result_log log;
  a.add(fake_read(memory.data(), 0));
  a.flush();
  REQUIRE(fake.wait_pending(1));
  b.add(fake_read(memory.data() + block, 1));
  b.flush();
  REQUIRE(fake.wait_pending(2));  // no gate: b's wave overlaps a's
  a.add(fake_read(memory.data() + 2 * block, 2));
  a.flush();
  CHECK(a.staged() == 1);  // a's own wave is still in flight
  CHECK(a.in_flight() == 1);
  fake.complete(2);
  CHECK(a.reap(5s, log) == 1);
  CHECK(a.staged() == 1);  // reap never hands anything over
  a.flush();
  CHECK(a.staged() == 0);
  CHECK(a.in_flight() == 1);
  REQUIRE(reap_all(b, log, 5s));
  fake.complete(1);
  REQUIRE(reap_all(a, log, 5s));
  CHECK(log.results.size() == 3);
  fake.with([](batch_script& s) {
    CHECK(s.submits_while_pending > 0);
    return 0;
  });
}

TEST_CASE("host batch readers alive together share the first reader's policy",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  {
    host_batch_reader first(4, 4, batch_submit_policy::per_reader);
    host_batch_reader second(4, 4, batch_submit_policy::continuous);
    CHECK(first.policy() == batch_submit_policy::per_reader);
    CHECK(second.policy() == batch_submit_policy::per_reader);
  }
  // All gone: the next reader fixes the policy again.
  host_batch_reader later(4, 4, batch_submit_policy::continuous);
  CHECK(later.policy() == batch_submit_policy::continuous);
}

TEST_CASE("host batch config validates the submit policy", "[io][cufile][host_batch]")
{
  cucascade::io::cufile::host_batch_config config;
  config.enabled = true;
  CHECK(config.submit_policy == batch_submit_policy::per_reader);
  CHECK(config.max_wave_bytes == (256UL << 20));
  CHECK_NOTHROW(config.validate());
  config.submit_policy = batch_submit_policy::per_reader;
  CHECK_NOTHROW(config.validate());
  config.submit_policy = static_cast<batch_submit_policy>(7);
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  config.submit_policy  = batch_submit_policy::exclusive;
  config.max_wave_bytes = 0;
  CHECK_NOTHROW(config.validate());
  config.max_wave_bytes = block + 1;
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
}

TEST_CASE("cuFile batch ignores devPtr_offset on host memory (why entries use offset 0)",
          "[io][cufile][host_batch]")
{
  // Pins a libcufile quirk the reader works around: with a shared devPtr_base and per-entry
  // devPtr_offset, libcufile 1.17 (compatibility mode, host memory) lands every aligned entry
  // at the base.  Skips when the library honours the offset.
  auto const* api = real_batch_api();
  if (api == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  constexpr unsigned entries        = 16;
  constexpr std::size_t entry_bytes = 64UL << 10;
  pattern_file const file{entries * entry_bytes};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  host_memory memory(entries * entry_bytes, false);

  CUfileBatchHandle_t handle{nullptr};
  REQUIRE(api->batch_setup(&handle, entries).err == CU_FILE_SUCCESS);
  auto run = [&](bool shared_base) {
    std::memset(memory.data(), 0xEE, memory.size());
    std::vector<CUfileIOParams_t> params(entries);
    for (unsigned i = 0; i < entries; ++i) {
      auto& p                 = params[i];
      p                       = CUfileIOParams_t{};
      p.mode                  = CUFILE_BATCH;
      p.opcode                = CUFILE_READ;
      p.fh                    = object->cufile_handle();
      p.u.batch.file_offset   = static_cast<off_t>(i * entry_bytes);
      p.u.batch.size          = entry_bytes;
      p.u.batch.devPtr_base   = shared_base ? memory.data() : memory.data() + i * entry_bytes;
      p.u.batch.devPtr_offset = shared_base ? static_cast<off_t>(i * entry_bytes) : 0;
      p.cookie                = reinterpret_cast<void*>(std::uintptr_t{i} + 1);
    }
    REQUIRE(api->batch_submit(handle, entries, params.data(), 0).err == CU_FILE_SUCCESS);
    std::vector<CUfileIOEvents_t> events(entries);
    unsigned reported   = 0;
    auto const deadline = clock_type::now() + 20s;
    while (reported < entries) {
      REQUIRE(clock_type::now() < deadline);
      unsigned nr = entries;
      timespec timeout{0, 10'000'000};
      REQUIRE(api->batch_get_status(handle, 1, &nr, events.data(), &timeout).err == CU_FILE_SUCCESS);
      for (unsigned k = 0; k < nr; ++k) {
        if (final_status(events[k].status)) { ++reported; }
      }
    }
    unsigned wrong = 0;
    for (unsigned i = 1; i < entries; ++i) {
      if (!file.matches(memory.data() + i * entry_bytes, i * entry_bytes, entry_bytes)) { ++wrong; }
    }
    return wrong;
  };
  auto const distinct_wrong = run(false);
  auto const shared_wrong   = run(true);
  api->batch_destroy(handle);

  CHECK(distinct_wrong == 0);  // the reader's layout always works
  if (shared_wrong == 0) {
    SKIP("this libcufile honours devPtr_offset on host memory; offset 0 is merely a choice");
  }
  WARN("libcufile " << cufile_api::version() << ": " << shared_wrong << " of " << (entries - 1)
                    << " shared-base entries landed elsewhere (devPtr_offset ignored)");
  CHECK(shared_wrong > 0);
}

// -- Scripted batch ---------------------------------------------------------------------------

TEST_CASE("host batch reader is unavailable without a batch group or a handle",
          "[io][cufile][host_batch]")
{
  SECTION("no batch group")
  {
    scripted_batch const fake(true, false);
    host_batch_reader reader(8, 4);
    CHECK_FALSE(reader.available());
    CHECK(reader.free_entries() == 0);
    CHECK(fake.with([](batch_script& s) { return s.setups; }) == 0);
  }
  SECTION("SetUp fails")
  {
    scripted_batch fake(true);
    fake.with([](batch_script& s) { return s.setup_error = static_cast<int>(CU_FILE_INTERNAL_ERROR); });
    host_batch_reader reader(8, 4);
    CHECK_FALSE(reader.available());
    CHECK(reader.free_entries() == 0);
    CHECK(reader.capacity() == 0);
    CHECK(fake.with([](batch_script& s) { return s.setups; }) == 1);
    result_log log;
    CHECK(reader.reap(10ms, log) == 0);
    reader.drain(log);
    CHECK(log.results.empty());
  }
  SECTION("zero depth")
  {
    scripted_batch const fake(true);
    host_batch_reader reader(0, 4);
    CHECK_FALSE(reader.available());
    CHECK(reader.free_entries() == 0);
  }
}

TEST_CASE("host batch reader reap honours the wait", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  host_batch_reader reader(4, 4);
  REQUIRE(reader.available());
  CHECK(reader.capacity() == 4);
  reader.add(fake_read(memory.data(), 0));
  reader.add(fake_read(memory.data() + block, 1));
  reader.flush();
  REQUIRE(reader.in_flight() == 2);
  result_log log;

  // Poll: returns at once.
  auto t0 = clock_type::now();
  CHECK(reader.reap(0us, log) == 0);
  CHECK(clock_type::now() - t0 < 20ms);

  // Nothing completes: returns after about the wait.
  t0 = clock_type::now();
  CHECK(reader.reap(50ms, log) == 0);
  auto const waited = clock_type::now() - t0;
  CHECK(waited >= 45ms);
  CHECK(waited < 1s);

  // The wave completes meanwhile: returns early (waves are reaped whole).
  std::thread completer([&] {
    std::this_thread::sleep_for(20ms);
    fake.complete(2);
  });
  t0 = clock_type::now();
  auto const got = reader.reap(5s, log);
  auto const early = clock_type::now() - t0;
  completer.join();
  CHECK(got == 2);
  CHECK(early < 2s);
  CHECK(reader.in_flight() == 0);
  CHECK(reader.free_entries() == 4);
  REQUIRE(log.results.size() == 2);
  CHECK(log.count(batch_result::status::ok) == 2);
  fake.with([](batch_script& s) {
    CHECK_FALSE(s.null_timeout);
    return 0;
  });
}

TEST_CASE("host batch reader classifies failed and short entries", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  host_batch_reader reader(8, 4);
  REQUIRE(reader.available());
  for (std::size_t i = 0; i < 4; ++i) {
    reader.add(fake_read(memory.data() + i * 2 * block, i, 2 * block));
  }
  reader.flush();
  fake.complete(1);                                      // token 0: ok
  fake.complete(1, CUFILE_FAILED);                       // token 1: failed
  fake.complete(1, CUFILE_COMPLETE, block);              // token 2: short, 4 KiB of 8
  fake.complete(1, CUFILE_CANCELED);                     // token 3: failed
  result_log log;
  REQUIRE(reap_all(reader, log, 5s));
  REQUIRE(log.results.size() == 4);
  std::sort(log.results.begin(), log.results.end(), [](auto const& a, auto const& b) { return a.token < b.token; });
  CHECK(log.results[0].outcome == batch_result::status::ok);
  CHECK(log.results[0].bytes == 2 * block);
  CHECK(log.results[1].outcome == batch_result::status::failed);
  CHECK(log.results[1].bytes == 0);
  CHECK(log.results[1].error == static_cast<int>(CUFILE_FAILED));
  CHECK(log.results[2].outcome == batch_result::status::short_read);
  CHECK(log.results[2].bytes == block);
  CHECK(log.results[3].outcome == batch_result::status::failed);
  CHECK(log.results[3].error == static_cast<int>(CUFILE_CANCELED));
  CHECK(reader.counters().completed == 1);
  CHECK(reader.counters().short_reads == 1);
  CHECK(reader.counters().failed == 2);
  CHECK(reader.available());
  CHECK(reader.free_entries() == 8);
}

TEST_CASE("host batch reader frees an entry before reporting it", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(2 * block, false);
  host_batch_reader reader(1, 4);
  REQUIRE(reader.capacity() == 1);
  reader.add(fake_read(memory.data(), 0));
  reader.flush();
  CHECK(reader.free_entries() == 0);
  fake.complete(1);
  std::size_t reported = 0;
  // done re-adds into the entry the result freed.
  std::ignore = reader.reap(1s, [&](batch_result const& r) {
    ++reported;
    CHECK(reader.free_entries() == 1);
    reader.add(fake_read(memory.data(), r.token + 1));
    reader.flush();
  });
  CHECK(reported == 1);
  CHECK(reader.in_flight() == 1);
  fake.complete(1);
  result_log log;
  REQUIRE(reap_all(reader, log, 5s));
  REQUIRE(log.results.size() == 1);
  CHECK(log.results[0].token == 1);
}

TEST_CASE("host batch reader recovers from a rejected submit", "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{8UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  constexpr std::size_t entry_bytes = 256UL << 10;
  host_memory memory(8 * entry_bytes, false);
  scripted_batch intercept(false);
  host_batch_reader reader(8, 3);
  REQUIRE(reader.available());
  auto read_at = [&](std::size_t i) {
    return batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{i * entry_bytes, entry_bytes}, i};
  };
  result_log log;

  SECTION("nothing else in flight: reset at once")
  {
    intercept.with([](batch_script& s) { return s.reject_submits = 1; });
    for (std::size_t i = 0; i < 4; ++i) {
      reader.add(read_at(i));
    }
    reader.flush();
    // The helper submits, sees the rejection, recreates the handle, then reports.
    REQUIRE(reap_all(reader, log));
    CHECK(reader.in_flight() == 0);
    CHECK(reader.available());
    CHECK(intercept.with([](batch_script& s) { return s.cancels; }) == 1);
    CHECK(intercept.with([](batch_script& s) { return s.destroys; }) == 1);
    CHECK(intercept.with([](batch_script& s) { return s.setups; }) == 2);
    CHECK(log.results.size() == 4);
    CHECK(log.count(batch_result::status::failed) == 4);
    for (auto const& r : log.results) {
      CHECK(r.error == static_cast<int>(CU_FILE_BATCH_FULL));
      CHECK(r.bytes == 0);
    }
    CHECK(reader.free_entries() == 8);
    CHECK(reader.counters().rejected_submits == 1);
    CHECK(reader.counters().handle_resets == 1);
  }
  // Later reads succeed on the new handle.
  log.results.clear();
  std::memset(memory.data(), 0xEE, memory.size());
  for (std::size_t i = 0; i < 8; ++i) {
    reader.add(read_at(i));
  }
  reader.flush();
  REQUIRE(reap_all(reader, log));
  CHECK(log.count(batch_result::status::ok) == 8);
  CHECK(file.matches(memory.data(), 0, 8 * entry_bytes));
  intercept.with([](batch_script& s) {
    CHECK(s.params_ok);
    return 0;
  });
}

TEST_CASE("host batch reader reports a rejected submit after the entries in flight",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  host_batch_reader reader(8, 3, batch_submit_policy::continuous);
  REQUIRE(reader.available());
  reader.add(fake_read(memory.data(), 0));
  reader.add(fake_read(memory.data() + block, 1));
  reader.flush();
  REQUIRE(reader.in_flight() == 2);
  REQUIRE(fake.wait_pending(2));  // submitted by the helper
  fake.with([](batch_script& s) { return s.reject_submits = 1; });
  reader.add(fake_read(memory.data() + 2 * block, 2));
  reader.add(fake_read(memory.data() + 3 * block, 3));
  reader.flush();
  CHECK(reader.available());
  result_log log;
  // Not reported while the handle that rejected them still has entries in flight.
  CHECK(reader.reap(50ms, log) == 0);
  CHECK(fake.with([](batch_script& s) { return s.rejected; }) == 1);
  CHECK(fake.with([](batch_script& s) { return s.destroys; }) == 0);
  fake.complete(2);
  REQUIRE(reap_all(reader, log, 5s));
  REQUIRE(log.results.size() == 4);
  CHECK(log.results[0].outcome == batch_result::status::ok);
  CHECK(log.results[1].outcome == batch_result::status::ok);
  CHECK(log.results[2].outcome == batch_result::status::failed);
  CHECK(log.results[3].outcome == batch_result::status::failed);
  CHECK(fake.with([](batch_script& s) { return s.cancels; }) == 1);
  CHECK(fake.with([](batch_script& s) { return s.destroys; }) == 1);
  CHECK(fake.with([](batch_script& s) { return s.setups; }) == 2);
  CHECK(reader.free_entries() == 8);
  reader.add(fake_read(memory.data(), 4));
  reader.flush();
  fake.complete(1);
  log.results.clear();
  REQUIRE(reap_all(reader, log, 5s));
  REQUIRE(log.results.size() == 1);
  CHECK(log.results[0].outcome == batch_result::status::ok);
}

TEST_CASE("host batch reader gives up after consecutive rejected submits",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  host_batch_reader reader(4, 3);
  REQUIRE(reader.available());
  result_log log;
  auto cycle = [&](std::size_t token) {
    reader.add(fake_read(memory.data(), token));
    reader.flush();
    fake.complete(1);
    REQUIRE(reap_all(reader, log, 5s));
  };

  // Two rejections, a success (resets the count), two more: still available.
  fake.with([](batch_script& s) { return s.reject_submits = 2; });
  cycle(0);
  cycle(1);
  cycle(2);
  fake.with([](batch_script& s) { return s.reject_submits = 2; });
  cycle(3);
  cycle(4);
  CHECK(reader.available());
  CHECK(log.count(batch_result::status::failed) == 4);
  CHECK(log.count(batch_result::status::ok) == 1);

  // The third consecutive rejection disables it for good; its entry is still reported.
  fake.with([](batch_script& s) { return s.reject_submits = 1; });
  cycle(5);
  CHECK_FALSE(reader.available());
  CHECK(reader.free_entries() == 0);
  CHECK(log.results.size() == 6);
  CHECK(log.results.back().outcome == batch_result::status::failed);
  CHECK(reader.counters().rejected_submits == 5);
  auto const setups = fake.with([](batch_script& s) { return s.setups; });
  CHECK(setups == 5);  // initial + four resets; none after giving up
  CHECK(fake.with([](batch_script& s) { return s.destroys; }) == 5);
}

TEST_CASE("host batch reader gives up when the handle cannot be recreated",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(2 * block, false);
  fake.with([](batch_script& s) {
    s.setup_error      = static_cast<int>(CU_FILE_INTERNAL_ERROR);
    s.setup_fail_after = 1;  // the constructor's SetUp succeeds
    s.reject_submits   = 1;
    return 0;
  });
  host_batch_reader reader(2, 4);
  REQUIRE(reader.available());
  reader.add(fake_read(memory.data(), 0));
  reader.flush();
  result_log log;
  reader.drain(log);
  CHECK_FALSE(reader.available());
  REQUIRE(log.results.size() == 1);
  CHECK(log.results[0].outcome == batch_result::status::failed);
}

TEST_CASE("host batch reader aborts the handle on a GetStatus error", "[io][cufile][host_batch]")
{
  SECTION("fake")
  {
    scripted_batch fake(true);
    host_memory memory(8 * block, false);
    host_batch_reader reader(8, 4);
    for (std::size_t i = 0; i < 5; ++i) {
      reader.add(fake_read(memory.data() + i * block, i));
    }
    fake.with([](batch_script& s) { return s.get_status_error = true; });
    reader.flush();
    reader.add(fake_read(memory.data() + 5 * block, 5));  // staged, not flushed
    result_log log;
    auto const reported = reader.reap(5s, log);
    CHECK(reported == 6);
    CHECK(log.count(batch_result::status::failed) == 6);
    for (auto const& r : log.results) {
      // The staged entry was never handed over: it is dropped, not aborted.
      CHECK(r.error == (r.token == 5 ? static_cast<int>(CUFILE_CANCELED) : static_cast<int>(CU_FILE_INTERNAL_ERROR)));
    }
    CHECK_FALSE(reader.available());
    CHECK(reader.in_flight() == 0);
    CHECK(reader.free_entries() == 0);
    CHECK(fake.with([](batch_script& s) { return s.cancels; }) == 1);
    CHECK(fake.with([](batch_script& s) { return s.destroys; }) == 1);
    reader.drain(log);
  }
  SECTION("libcufile entries in flight")
  {
    if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
    pattern_file const file{32UL << 20};
    auto const object = open_registered(file);
    if (!object) { SKIP("cuFile cannot register the test file"); }
    constexpr std::size_t entry_bytes = 4UL << 20;
    host_memory memory(8 * entry_bytes, false);
    scripted_batch intercept(false);
    {
      host_batch_reader reader(8, 4);
      REQUIRE(reader.available());
      for (std::size_t i = 0; i < 8; ++i) {
        reader.add(batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{i * entry_bytes, entry_bytes}, i});
      }
      intercept.with([](batch_script& s) { return s.get_status_error = true; });
      reader.flush();
      result_log log;
      std::ignore = reader.reap(5s, log);
      CHECK(log.results.size() == 8);
      CHECK(log.count(batch_result::status::failed) == 8);
      CHECK_FALSE(reader.available());
      CHECK(intercept.with([](batch_script& s) { return s.cancels; }) == 1);
      CHECK(intercept.with([](batch_script& s) { return s.destroys; }) == 1);
      // Cancelled and destroyed: cuFile no longer writes the destinations.
      std::memset(memory.data(), 0xEE, memory.size());
      std::this_thread::sleep_for(200ms);
      CHECK(std::all_of(memory.data(), memory.data() + memory.size(), [](std::uint8_t b) {
        return b == 0xEE;
      }));
    }
    CHECK(intercept.with([](batch_script& s) { return s.destroys; }) == 1);  // not again
  }
}

TEST_CASE("host batch reader maps real event failures", "[io][cufile][host_batch]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  pattern_file const file{4UL << 20};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  constexpr std::size_t entry_bytes = 64UL << 10;
  host_memory memory(4 * entry_bytes, false);
  scripted_batch intercept(false);
  host_batch_reader reader(4, 4);
  REQUIRE(reader.available());
  intercept.with([](batch_script& s) {
    s.fail_events  = 1;
    s.short_events = 1;
    s.short_ret    = 2 * block;
    return 0;
  });
  for (std::size_t i = 0; i < 4; ++i) {
    reader.add(batch_read{object->cufile_handle(), memory.data() + i * entry_bytes, range{i * entry_bytes, entry_bytes}, i});
  }
  reader.flush();
  result_log log;
  REQUIRE(reap_all(reader, log));
  REQUIRE(log.results.size() == 4);
  CHECK(log.count(batch_result::status::failed) == 1);
  CHECK(log.count(batch_result::status::short_read) == 1);
  CHECK(log.count(batch_result::status::ok) == 2);
  for (auto const& r : log.results) {
    if (r.outcome == batch_result::status::short_read) {
      CHECK(r.bytes == 2 * block);
      CHECK(file.matches(memory.data() + r.token * entry_bytes, r.token * entry_bytes, r.bytes));
    }
    if (r.outcome == batch_result::status::ok) {
      CHECK(file.matches(memory.data() + r.token * entry_bytes, r.token * entry_bytes, entry_bytes));
    }
  }
}

TEST_CASE("host batch reader reaps a slow wave in one GetStatus call", "[io][cufile][host_batch]")
{
  // A wave slower than any short timeout: the helper must not return early (a partial return is
  // the libcufile use-after-free trigger), so exactly one GetStatus per wave.
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  host_batch_reader reader(4, 4);
  REQUIRE(reader.policy() == batch_submit_policy::per_reader);
  result_log log;
  for (int wave = 0; wave < 2; ++wave) {
    for (std::size_t i = 0; i < 3; ++i) {
      reader.add(fake_read(memory.data() + i * block, i));
    }
    reader.flush();
    REQUIRE(fake.wait_pending(3));
    std::thread completer([&] {
      std::this_thread::sleep_for(1500ms);
      fake.complete(3);
    });
    REQUIRE(reap_all(reader, log, 10s));
    completer.join();
  }
  CHECK(log.count(batch_result::status::ok) == 6);
  CHECK(fake.with([](batch_script& s) { return s.get_status_calls; }) == 2);
  CHECK(reader.counters().partial_status == 0);
  CHECK(reader.available());
}

TEST_CASE("host batch reader is poisoned by a partial wave return", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  result_log log;
  {
    host_batch_reader reader(4, 4);
    reader.add(fake_read(memory.data(), 0));
    reader.add(fake_read(memory.data() + block, 1));
    fake.with([](batch_script& s) { return s.force_partial = true; });
    reader.flush();
    REQUIRE(fake.wait_pending(2));
    // Nothing completed when the call returns: the whole wave stays cuFile's.
    REQUIRE(reap_all(reader, log, 0s) == false);  // in flight for good
    auto const deadline = clock_type::now() + 5s;
    while (reader.available() && clock_type::now() < deadline) {
      std::ignore = reader.reap(10ms, log);
    }
    CHECK_FALSE(reader.available());
    CHECK(reader.poisoned());
    CHECK(reader.in_flight() == 2);  // never reported, never freed
    CHECK(log.results.empty());
    CHECK(reader.counters().partial_status == 1);
    auto const calls = fake.with([](batch_script& s) { return s.get_status_calls; });
    std::this_thread::sleep_for(100ms);
    CHECK(fake.with([](batch_script& s) { return s.get_status_calls; }) == calls);  // no more
    reader.add(fake_read(memory.data() + 2 * block, 2));  // free_entries() == 0: dropped
    CHECK(reader.staged() == 0);
  }
  // The poisoned handle was never destroyed.
  CHECK(fake.with([](batch_script& s) { return s.destroys; }) == 0);
}

TEST_CASE("host batch reader reports every entry once when a wave follows an abort",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  host_batch_reader reader(8, 4, batch_submit_policy::continuous);
  reader.add(fake_read(memory.data(), 0));
  reader.add(fake_read(memory.data() + block, 1));
  reader.flush();
  REQUIRE(fake.wait_pending(2));
  fake.with([](batch_script& s) { return s.get_status_error = true; });
  // The helper aborts and publishes; before the owner collects, another wave is flushed.
  auto const deadline = clock_type::now() + 5s;
  while (!reader.completions_pending() && clock_type::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(reader.completions_pending());
  reader.add(fake_read(memory.data() + 2 * block, 2));
  reader.add(fake_read(memory.data() + 3 * block, 3));
  reader.flush();
  result_log log;
  CHECK(reader.reap(1s, log) == 4);
  CHECK(reader.reap(10ms, log) == 0);
  REQUIRE(log.results.size() == 4);
  std::vector<int> seen(4, 0);
  for (auto const& r : log.results) {
    REQUIRE(r.token < 4);
    ++seen[r.token];
    CHECK(r.outcome == batch_result::status::failed);
  }
  CHECK(seen == std::vector<int>{1, 1, 1, 1});
  CHECK_FALSE(reader.completions_pending());  // the wait point does not spin
  CHECK_FALSE(reader.available());
  CHECK(reader.in_flight() == 0);
  CHECK(fake.with([](batch_script& s) { return s.submits; }) == 1);
}

TEST_CASE("host batch reader tops waves up instead of ping-ponging", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  constexpr unsigned depth = 16;
  host_memory memory(depth * block, false);
  host_batch_reader reader(depth, 4);
  result_log log;
  // A lone entry first, then a sustained queue: the owner reaps, stages what it can, then
  // flushes once, as the reactor's pass does.
  reader.add(fake_read(memory.data(), 0));
  reader.flush();
  std::size_t issued = 1;
  auto stage = [&] {
    while (reader.free_entries() > 0 && issued < 400) {
      reader.add(fake_read(memory.data() + (issued % depth) * block, issued));
      ++issued;
    }
  };
  stage();      // staged behind the lone entry's wave
  reader.flush();  // deferred: the wave is in flight
  auto const deadline = clock_type::now() + 30s;
  while (log.results.size() < issued && clock_type::now() < deadline) {
    fake.complete(scripted_batch::all);
    std::ignore = reader.reap(1s, log);
    stage();
    reader.flush();
    if (reader.in_flight() > 0) REQUIRE(fake.wait_pending(reader.in_flight()));
  }
  REQUIRE(log.results.size() == issued);
  auto const sizes = fake.with([](batch_script& s) { return s.submit_sizes; });
  REQUIRE(sizes.size() > 4);
  double total = 0;
  for (std::size_t i = 2; i + 1 < sizes.size(); ++i) {  // after warm-up, before the tail
    total += static_cast<double>(sizes[i]);
  }
  auto const mean = total / static_cast<double>(sizes.size() - 3);
  INFO("mean wave " << mean);
  CHECK(mean >= 0.9 * depth);
}

TEST_CASE("host batch reader caps a wave at max_wave_bytes", "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(8 * block, false);
  cucascade::io::cufile::detail::host_batch_options options;
  options.depth          = 8;
  options.max_wave_bytes = 2 * block;
  host_batch_reader reader(options);
  for (std::size_t i = 0; i < 5; ++i) {
    reader.add(fake_read(memory.data() + i * block, i));
  }
  result_log log;
  reader.flush();
  CHECK(reader.in_flight() == 2);
  CHECK(reader.staged() == 3);
  CHECK(reader.deferred());
  for (int wave = 0; wave < 3; ++wave) {
    REQUIRE(fake.wait_pending(reader.in_flight()));
    fake.complete(scripted_batch::all);
    REQUIRE(reap_all(reader, log, 5s));
    reader.flush();
  }
  CHECK(log.results.size() == 5);
  CHECK(fake.with([](batch_script& s) { return s.submit_sizes; }) ==
        std::vector<std::size_t>{2, 2, 1});
}

TEST_CASE("host batch reader discards staged entries without submitting them",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  host_batch_reader reader(4, 4);
  for (std::size_t i = 0; i < 3; ++i) {
    reader.add(fake_read(memory.data() + i * block, i));
  }
  reader.discard_staged_if([](std::uint64_t token) noexcept { return token == 1; });
  CHECK(reader.staged() == 2);
  reader.discard_staged();
  CHECK(reader.staged() == 0);
  reader.flush();
  result_log log;
  CHECK(reader.reap(0us, log) == 3);
  CHECK(log.count(batch_result::status::failed) == 3);
  for (auto const& r : log.results) {
    CHECK(r.error == static_cast<int>(CUFILE_CANCELED));
  }
  CHECK(fake.with([](batch_script& s) { return s.submits; }) == 0);
  CHECK(reader.free_entries() == 4);
}

#ifdef NDEBUG
TEST_CASE("host batch reader fails misaligned entries without submitting them",
          "[io][cufile][host_batch]")
{
  scripted_batch fake(true);
  host_memory memory(4 * block, false);
  host_batch_reader reader(4, 4);
  reader.add(batch_read{reinterpret_cast<void*>(std::uintptr_t{0x1000}), memory.data() + 512, range{0, block}, 1});
  reader.add(batch_read{reinterpret_cast<void*>(std::uintptr_t{0x1000}), memory.data(), range{512, block}, 2});
  reader.add(batch_read{reinterpret_cast<void*>(std::uintptr_t{0x1000}), memory.data(), range{0, 100}, 3});
  reader.flush();
  CHECK(fake.with([](batch_script& s) { return s.submits; }) == 0);
  result_log log;
  CHECK(reader.reap(0us, log) == 3);
  CHECK(log.count(batch_result::status::failed) == 3);
  CHECK(reader.free_entries() == 4);
}
#endif

// -- Measurement (hidden) -----------------------------------------------------------------------

TEST_CASE("host batch reader throughput vs pread", "[.][host_batch_bench]")
{
  if (real_batch_api() == nullptr) { SKIP("libcufile with the batch API is unavailable"); }
  constexpr std::size_t file_bytes  = 1UL << 30;
  constexpr std::size_t entry_bytes = 1UL << 20;
  pattern_file const file{file_bytes};
  auto const object = open_registered(file);
  if (!object) { SKIP("cuFile cannot register the test file"); }
  for (unsigned depth : {8U, 32U, 64U, 128U}) {
    host_memory memory(depth * entry_bytes, has_cuda_device());
    host_batch_reader reader(depth, 4);
    REQUIRE(reader.available());
    std::vector<std::size_t> free_slots;
    for (std::size_t i = 0; i < reader.capacity(); ++i) {
      free_slots.push_back(i);
    }
    std::size_t next = 0, done_count = 0, bad = 0;
    std::vector<std::size_t> slot_of(file_bytes / entry_bytes);
    auto done = [&](batch_result const& r) {
      ++done_count;
      if (r.outcome != batch_result::status::ok) { ++bad; }
      free_slots.push_back(slot_of[r.token]);
    };
    auto const start = clock_type::now();
    while (done_count < file_bytes / entry_bytes) {
      while (next < file_bytes / entry_bytes && reader.free_entries() > 0) {
        auto const slot = free_slots.back();
        free_slots.pop_back();
        slot_of[next] = slot;
        reader.add(batch_read{object->cufile_handle(), memory.data() + slot * entry_bytes, range{next * entry_bytes, entry_bytes}, next});
        ++next;
      }
      reader.flush();
      std::ignore = reader.reap(200us, done);
    }
    reader.drain(done);
    auto const s = std::chrono::duration<double>(clock_type::now() - start).count();
    CHECK(bad == 0);
    auto const& c = reader.counters();
    WARN("batch depth " << depth << ": " << static_cast<double>(file_bytes) / s / 1e9 << " GB/s, "
                        << c.submits << " submits of " << c.submitted_entries << " entries, mean submit "
                        << static_cast<double>(c.submit_ns) / 1e3 / static_cast<double>(c.submits)
                        << " us, " << static_cast<double>(c.submit_ns) / 1e9 / s * 100
                        << "% of the time in Submit");
  }
  for (bool direct : {false, true}) {
    int const fd = ::open(file.path().c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    REQUIRE(fd >= 0);
    host_memory memory(entry_bytes, false);
    auto const start = clock_type::now();
    for (std::size_t off = 0; off < file_bytes; off += entry_bytes) {
      REQUIRE(::pread(fd, memory.data(), entry_bytes, static_cast<off_t>(off)) == static_cast<ssize_t>(entry_bytes));
    }
    auto const s = std::chrono::duration<double>(clock_type::now() - start).count();
    ::close(fd);
    WARN("pread" << (direct ? " O_DIRECT" : "") << " 1 thread: " << static_cast<double>(file_bytes) / s / 1e9
                 << " GB/s");
  }
}

