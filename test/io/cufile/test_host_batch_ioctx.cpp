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
 * [cufile]           - cuFile backend
 * [host_batch_ioctx] - host reads through the cuFile batch API, end to end through cufile_ioctx
 *
 * Every destination byte is checked against the file and the guard bytes around every
 * destination against their fill.  Cases run against libcufile (compatibility mode on hosts
 * without GDS); failure cases intercept its batch calls through a scripted cuFile table that
 * forwards everything else.
 */

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <cufile.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

using cucascade::io::io_priority;
using cucascade::io::cufile::batch_submit_policy;
using cucascade::io::cufile::config;
using cucascade::io::cufile::cufile_api;
using cucascade::io::cufile::cufile_io_object;
using cucascade::io::cufile::cufile_ioctx;
using cucascade::io::cufile::cufile_reactor;
using cucascade::io::cufile::make_cufile_ioctx;
using cucascade::io::cufile::reactor_stats;
using io_slice = cucascade::io::slice;

namespace {

using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

constexpr std::size_t block           = 4096;
constexpr std::size_t KiB             = 1024;
constexpr std::size_t MiB             = 1024 * KiB;
constexpr std::uint8_t guard_fill     = 0xA5;
constexpr std::size_t default_min     = 64 * KiB;  // host_batch_config::min_bytes below

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
            ("cucascade_host_batch_ioctx_" + std::to_string(::getpid()) + "_" +
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

/// Skip unless libcufile with the batch group is usable.
void require_batch_cufile()
{
  auto const* api = cufile_api::get();
  if (api == nullptr || !api->has_batch || !cufile_api::open_driver()) {
    SKIP("libcufile with the batch API is unavailable");
  }
}

/** 4 KiB-aligned host memory filled with @ref guard_fill: pageable or pinned. */
class host_memory {
 public:
  host_memory(std::size_t bytes, bool pinned) : _bytes(bytes), _pinned(pinned)
  {
    if (pinned) {
      void* p = nullptr;
      REQUIRE(cudaMallocHost(&p, bytes) == cudaSuccess);
      _data = static_cast<std::uint8_t*>(p);
    } else {
      _data = static_cast<std::uint8_t*>(
        std::aligned_alloc(block, (bytes + block - 1) / block * block));
    }
    REQUIRE(_data != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(_data) % block == 0);
    std::memset(_data, guard_fill, bytes);
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

/**
 * Destinations of host reads laid out in one 4 KiB-aligned arena with guard bytes around
 * each.  A congruent destination sits where `(dst - file offset) % 4096 == 0`, so its aligned
 * body can be read straight into it; a non-congruent one is 64 bytes off that.
 */
class host_plan {
 public:
  void add(std::size_t offset, std::size_t size, bool congruent)
  {
    _cursor += 64;  // guard
    auto at = _cursor;
    // Place the destination congruent (or 64 bytes off) with the file offset.
    auto const want = (offset + (congruent ? 0 : 64)) % block;
    at              = at + ((want + block - at % block) % block);
    _entries.push_back(entry{offset, size, at, congruent});
    _cursor = at + size;
  }
  [[nodiscard]] std::size_t arena_bytes() const noexcept { return _cursor + block; }
  [[nodiscard]] std::size_t offset_of(std::size_t i) const noexcept { return _entries[i].offset; }
  [[nodiscard]] std::size_t size_of(std::size_t i) const noexcept { return _entries[i].size; }
  /// Where slice @p i's destination sits within a 4 KiB block.
  [[nodiscard]] std::size_t misalign(std::size_t i) const noexcept { return _entries[i].at % block; }
  [[nodiscard]] std::size_t count() const noexcept { return _entries.size(); }

  [[nodiscard]] std::vector<io_slice> slices(std::uint8_t* arena) const
  {
    std::vector<io_slice> out;
    for (auto const& e : _entries) {
      out.emplace_back(e.offset, e.size, arena + e.at);
    }
    return out;
  }

  [[nodiscard]] std::size_t clamped(std::size_t i, std::size_t file_size) const noexcept
  {
    auto const& e = _entries[i];
    return e.offset >= file_size ? 0 : std::min(e.size, file_size - e.offset);
  }

  [[nodiscard]] std::size_t expected_bytes(std::size_t file_size) const noexcept
  {
    std::size_t total = 0;
    for (std::size_t i = 0; i < _entries.size(); ++i) {
      total += clamped(i, file_size);
    }
    return total;
  }

  /// Whether slice @p i is served by the batch (see cufile_reactor::try_batch_read).
  [[nodiscard]] bool eligible(std::size_t i, std::size_t file_size, std::size_t min_bytes) const
  {
    auto const& e   = _entries[i];
    auto const size = clamped(i, file_size);
    if (size == 0) return false;
    auto const begin = (e.offset + block - 1) / block * block;
    auto const end   = (e.offset + size) / block * block;
    return end > begin && end - begin >= min_bytes && e.congruent;
  }

  [[nodiscard]] std::size_t eligible_count(std::size_t file_size, std::size_t min_bytes) const
  {
    std::size_t n = 0;
    for (std::size_t i = 0; i < _entries.size(); ++i) {
      n += eligible(i, file_size, min_bytes) ? 1U : 0U;
    }
    return n;
  }

  [[nodiscard]] bool verify(std::uint8_t const* arena,
                            std::size_t arena_size,
                            pattern_file const& file,
                            std::string* why = nullptr) const
  {
    std::vector<bool> covered(arena_size, false);
    for (std::size_t i = 0; i < _entries.size(); ++i) {
      auto const& e     = _entries[i];
      auto const inside = clamped(i, file.size());
      if (std::memcmp(arena + e.at, file.data() + e.offset, inside) != 0) {
        if (why) *why = "slice " + std::to_string(i) + " content";
        return false;
      }
      std::fill(covered.begin() + static_cast<std::ptrdiff_t>(e.at),
                covered.begin() + static_cast<std::ptrdiff_t>(e.at + inside),
                true);
    }
    for (std::size_t b = 0; b < arena_size; ++b) {
      if (!covered[b] && arena[b] != guard_fill) {
        if (why) *why = "guard byte " + std::to_string(b) + " overwritten";
        return false;
      }
    }
    return true;
  }

 private:
  struct entry {
    std::size_t offset;
    std::size_t size;
    std::size_t at;
    bool congruent;
  };
  std::vector<entry> _entries;
  std::size_t _cursor{0};
};

/// A config with the host batch on.
[[nodiscard]] config batch_config(unsigned depth              = 8,
                                  std::size_t max_entry_bytes = 1 * MiB,
                                  std::size_t reactors        = 2)
{
  config cfg;
  cfg.host_batch.enabled         = true;
  cfg.host_batch.depth           = depth;
  cfg.host_batch.max_entry_bytes = max_entry_bytes;
  cfg.host_batch.min_bytes       = default_min;
  cfg.host_batch.reactors        = reactors;
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

[[nodiscard]] std::size_t queued_bytes(cufile_ioctx const& ioctx)
{
  std::size_t total = 0;
  for (std::size_t i = 0; i < ioctx.reactor_count(); ++i) {
    total += ioctx.reactor(i).queued_bytes();
  }
  return total;
}

void require_queued_bytes_zero(cufile_ioctx const& ioctx)
{
  auto const deadline = clock_type::now() + 10s;
  while (queued_bytes(ioctx) != 0 && clock_type::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(queued_bytes(ioctx) == 0);
}

/// Open @p file through @p ioctx; skips when cuFile cannot register it with an O_DIRECT fd.
[[nodiscard]] std::shared_ptr<cucascade::io::io_object> open_registered(cufile_ioctx& ioctx,
                                                                        pattern_file const& file)
{
  auto object       = ioctx.open_io_object(file.path());
  auto const& typed = dynamic_cast<cufile_io_object const&>(*object);
  if (!typed.cufile_registered() || typed.direct_fd() < 0) {
    SKIP("cuFile cannot register the test file with an O_DIRECT descriptor");
  }
  return object;
}

/// Run @p plan as one vectored host read and verify content, guards and the byte count.
void run_vectored(cufile_ioctx& ioctx,
                  cucascade::io::io_object const& object,
                  pattern_file const& file,
                  host_plan const& plan,
                  bool pinned,
                  io_priority priority = io_priority::automatic)
{
  host_memory arena{plan.arena_bytes(), pinned};
  auto const slices = plan.slices(arena.data());
  auto const got    = ioctx.host_readv_async_io(object, slices, priority).get(60s);
  CHECK(got == plan.expected_bytes(file.size()));
  std::string why;
  CHECK(plan.verify(arena.data(), arena.size(), file, &why));
  INFO(why);
}

// ---------------------------------------------------------------------------
// A cuFile table forwarding to libcufile with the batch calls intercepted.
// ---------------------------------------------------------------------------

struct intercept_state {
  std::mutex mutex;
  cufile_api real{};
  int setup_error{0};
  int reject_submits{0};
  int fail_events{0};
  int short_events{0};
  bool get_status_error{false};
  std::chrono::milliseconds submit_delay{0};  ///< Slows every wave down (on the helper thread).
  int submits{0};
  int setups{0};
  int cancels{0};
  int destroys{0};
  std::size_t entries{0};
  std::size_t max_entry_end{0};  ///< Largest file_offset + size submitted.
  bool params_ok{true};
};

intercept_state& intercept() noexcept
{
  static intercept_state state;
  return state;
}

[[nodiscard]] CUfileError_t status_of(int code) noexcept
{
  CUfileError_t e{};
  e.err = static_cast<CUfileOpError>(code);
  return e;
}

CUfileError_t intercepted_setup(CUfileBatchHandle_t* handle, unsigned nr)
{
  auto& s = intercept();
  {
    std::lock_guard lock(s.mutex);
    ++s.setups;
    if (s.setup_error != 0) return status_of(s.setup_error);
  }
  return s.real.batch_setup(handle, nr);
}

CUfileError_t intercepted_submit(CUfileBatchHandle_t handle,
                                 unsigned nr,
                                 CUfileIOParams_t* params,
                                 unsigned flags)
{
  auto& s = intercept();
  {
    std::lock_guard lock(s.mutex);
    ++s.submits;
    for (unsigned i = 0; i < nr; ++i) {
      auto const& p = params[i].u.batch;
      if (p.devPtr_offset != 0 || reinterpret_cast<std::uintptr_t>(p.devPtr_base) % block != 0 ||
          static_cast<std::size_t>(p.file_offset) % block != 0 || p.size % block != 0) {
        s.params_ok = false;
      }
      s.max_entry_end =
        std::max(s.max_entry_end, static_cast<std::size_t>(p.file_offset) + p.size);

    }
    if (s.reject_submits > 0) {
      --s.reject_submits;
      return status_of(CU_FILE_BATCH_FULL);
    }
    s.entries += nr;
  }
  if (auto const delay = intercept().submit_delay; delay.count() > 0) {
    std::this_thread::sleep_for(delay);
  }
  return s.real.batch_submit(handle, nr, params, flags);
}

CUfileError_t intercepted_get_status(CUfileBatchHandle_t handle,
                                     unsigned min_nr,
                                     unsigned* nr,
                                     CUfileIOEvents_t* events,
                                     timespec* timeout)
{
  auto& s = intercept();
  {
    std::lock_guard lock(s.mutex);
    if (s.get_status_error) {
      s.get_status_error = false;  // once: the reader gives up on its first error
      return status_of(CU_FILE_INTERNAL_ERROR);
    }
  }
  auto const status = s.real.batch_get_status(handle, min_nr, nr, events, timeout);
  if (status.err == CU_FILE_SUCCESS) {
    std::lock_guard lock(s.mutex);
    for (unsigned i = 0; i < *nr; ++i) {
      if (events[i].status != CUFILE_COMPLETE) continue;
      if (s.fail_events > 0) {
        --s.fail_events;
        events[i].status = CUFILE_FAILED;
      } else if (s.short_events > 0 && events[i].ret > block) {
        --s.short_events;
        events[i].ret = block + 100;  // unaligned on purpose
      }
    }
  }
  return status;
}

CUfileError_t intercepted_cancel(CUfileBatchHandle_t handle)
{
  {
    std::lock_guard lock(intercept().mutex);
    ++intercept().cancels;
  }
  return intercept().real.batch_cancel(handle);
}

void intercepted_destroy(CUfileBatchHandle_t handle)
{
  {
    std::lock_guard lock(intercept().mutex);
    ++intercept().destroys;
  }
  intercept().real.batch_destroy(handle);
}

/// Installs the intercepting table for its lifetime; declare it before the ioctx it serves.
class batch_intercept {
 public:
  batch_intercept()
  {
    auto const* real = cufile_api::get();
    REQUIRE(real != nullptr);
    auto& s = intercept();
    {
      std::lock_guard lock(s.mutex);
      s.real             = *real;
      s.setup_error      = 0;
      s.reject_submits   = 0;
      s.fail_events      = 0;
      s.short_events     = 0;
      s.get_status_error = false;
      s.submit_delay     = std::chrono::milliseconds{0};
      s.submits = s.setups = s.cancels = s.destroys = 0;
      s.entries                                     = 0;
      s.max_entry_end                               = 0;
      s.params_ok                                   = true;
    }
    _api                  = *real;
    _api.batch_setup      = &intercepted_setup;
    _api.batch_submit     = &intercepted_submit;
    _api.batch_get_status = &intercepted_get_status;
    _api.batch_cancel     = &intercepted_cancel;
    _api.batch_destroy    = &intercepted_destroy;
    cucascade::io::cufile::detail::set_cufile_api_for_testing(&_api);
  }
  ~batch_intercept() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  batch_intercept(batch_intercept const&)            = delete;
  batch_intercept& operator=(batch_intercept const&) = delete;

  template <typename F>
  auto with(F&& f) const
  {
    std::lock_guard lock(intercept().mutex);
    return f(intercept());
  }

 private:
  cufile_api _api{};
};

/// The alignment matrix: offsets, sizes and congruence, sized for @p file_size.
[[nodiscard]] host_plan alignment_plan(std::size_t file_size, bool congruent_only = false)
{
  host_plan plan;
  std::vector<std::size_t> const offsets{0, 2 * block, 777, file_size - 300};
  std::vector<std::size_t> const sizes{
    1, 4095, block, 64 * KiB, 64 * KiB + 2 * block, 300 * KiB + 77, 2 * MiB + 4096};
  for (auto const offset : offsets) {
    for (auto const size : sizes) {
      plan.add(offset, size, true);
      if (!congruent_only) plan.add(offset, size, false);
    }
  }
  return plan;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Every alignment, single and vectored, flat and planned pools.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: every alignment lands the file's bytes; stats tell the paths apart",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  auto const pool   = GENERATE(as<std::size_t>{}, 1, 4, 0);  // 0 = planned pool
  auto const pinned = GENERATE(false, true);
  if ((pinned || pool == 0) && !has_cuda_device()) SKIP("requires a CUDA device");
  CAPTURE(pool, pinned);
  pattern_file const file{(9 * MiB) + 1234};
  started_ioctx ioctx{batch_config(), pool};
  auto& ctx   = ioctx.get();
  auto object = open_registered(ctx, file);

  auto const plan = alignment_plan(file.size());
  // Single reads, one per matrix entry.
  auto const before = ctx.stats();
  std::size_t eligible = 0, others = 0, eligible_bytes = 0, other_bytes = 0;
  for (std::size_t i = 0; i < plan.count(); ++i) {
    auto const size = plan.clamped(i, file.size());
    if (plan.eligible(i, file.size(), default_min)) {
      ++eligible;
      eligible_bytes += size;
    } else {
      ++others;
      other_bytes += size;
    }
    // A destination of the same congruence as in the plan, with guards around it.
    host_memory memory{plan.size_of(i) + 3 * block, pinned};
    auto* const dst = memory.data() + block + plan.misalign(i);
    auto const got =
      ctx.host_read_async_io(*object, plan.offset_of(i), plan.size_of(i), dst).get(30s);
    CHECK(got == size);
    CHECK(std::memcmp(dst, file.data() + plan.offset_of(i), size) == 0);
    bool const guards =
      std::all_of(memory.data(), dst, [](auto b) { return b == guard_fill; }) &&
      std::all_of(dst + size, memory.data() + memory.size(), [](auto b) { return b == guard_fill; });
    CHECK(guards);
  }
  auto const single = ctx.stats() - before;
  CHECK(single.batch_reads == eligible);
  CHECK(single.batch_bytes == eligible_bytes);
  CHECK(single.host_reads == others);
  CHECK(single.host_bytes == other_bytes);
  CHECK(eligible > 0);

  // The whole matrix as one vectored read.
  auto const mid = ctx.stats();
  run_vectored(ctx, *object, file, plan, pinned);
  auto const vectored = ctx.stats() - mid;
  CHECK(vectored.batch_reads == plan.eligible_count(file.size(), default_min));
  CHECK(vectored.batch_reads + vectored.host_reads == plan.count());
  require_queued_bytes_zero(ctx);

  if (pool == 0) {
    // A planned pool: host reads go to the batch reactors only.
    for (std::size_t i = 0; i < ctx.reactor_count(); ++i) {
      auto const& reactor = ctx.reactor(i);
      auto const stats    = reactor.stats();
      INFO("reactor " << i);
      if (!reactor.placement().host_batch) {
        CHECK(stats.batch_reads == 0);
        CHECK(stats.host_reads == 0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2. End of file.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: reads at and past the end of a file", "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  batch_intercept const watch;
  auto const tail = GENERATE(as<std::size_t>{}, 0, 1, 1234, block - 1);
  pattern_file const file{(4 * MiB) + tail};
  CAPTURE(tail);
  {
    started_ioctx ioctx{batch_config(4, 256 * KiB), 2};
    auto& ctx   = ioctx.get();
    auto object = open_registered(ctx, file);
    host_plan plan;
    auto const size = file.size();
    plan.add(0, size, true);                      // the whole file
    plan.add(0, size + 10 * block, true);         // past EOF
    plan.add(size - 200 * KiB, 300 * KiB, true);  // crossing EOF
    plan.add(block, size - block, true);          // ending exactly at EOF
    plan.add(777, size, true);                    // unaligned start, crossing EOF
    run_vectored(ctx, *object, file, plan, false);
    auto const stats = ctx.stats();
    CHECK(stats.batch_reads == plan.eligible_count(size, default_min));
    require_queued_bytes_zero(ctx);
  }
  watch.with([&](intercept_state& s) {
    CHECK(s.max_entry_end <= file.size() / block * block);
    CHECK(s.params_ok);
    CHECK(s.entries > 0);
    return 0;
  });
}

// ---------------------------------------------------------------------------
// 3. Many waves for one large read.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: a large read takes many waves", "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  pattern_file const file{(256 * MiB) + 1234};
  batch_intercept const watch;
  {
    started_ioctx ioctx{batch_config(8, 1 * MiB), 1};
    auto& ctx   = ioctx.get();
    auto object = open_registered(ctx, file);
    host_plan plan;
    plan.add(777, file.size(), true);  // unaligned head, sub-block tail, 256 entries
    run_vectored(ctx, *object, file, plan, has_cuda_device());
    auto const stats = ctx.stats();
    CHECK(stats.batch_reads == 1);
    CHECK(stats.batch_bytes == file.size() - 777);
    CHECK(stats.batch_failures == 0);
    CHECK(stats.host_reads == 0);
    // The helper's counters are exported through reactor_stats.
    CHECK(stats.batch_entries == 256);
    CHECK(stats.batch_waves >= 32);
    CHECK(stats.batch_waves <= 256);
    CHECK(stats.batch_rejected_submits == 0);
    CHECK(stats.batch_partial_status == 0);
    CHECK(stats.batch_submit_ns > 0);
    require_queued_bytes_zero(ctx);
  }
  watch.with([](intercept_state& s) {
    CHECK(s.entries == 256);
    CHECK(s.submits >= 32);  // waves of at most 8 entries
    CHECK(s.params_ok);
    return 0;
  });
}

// ---------------------------------------------------------------------------
// 4. Failures injected under the batch: the reads still succeed.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: batch failures are finished synchronously",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  pattern_file const file{(8 * MiB) + 1234};
  batch_intercept watch;
  enum class fault { reject, failed_entry, short_entry, get_status, setup };
  auto const what = GENERATE(
    fault::reject, fault::failed_entry, fault::short_entry, fault::get_status, fault::setup);
  CAPTURE(static_cast<int>(what));
  watch.with([&](intercept_state& s) {
    switch (what) {
      case fault::reject: s.reject_submits = 1; break;
      case fault::failed_entry: s.fail_events = 2; break;
      case fault::short_entry: s.short_events = 2; break;
      case fault::get_status: s.get_status_error = true; break;
      case fault::setup: s.setup_error = static_cast<int>(CU_FILE_INTERNAL_ERROR); break;
    }
    return 0;
  });
  started_ioctx ioctx{batch_config(4, 256 * KiB), 1};
  auto& ctx   = ioctx.get();
  auto object = open_registered(ctx, file);
  host_plan plan;
  plan.add(0, 2 * MiB, true);
  plan.add(3 * MiB + 5, 1 * MiB, true);
  for (int round = 0; round < 3; ++round) {
    run_vectored(ctx, *object, file, plan, false);
  }
  auto const stats = ctx.stats();
  switch (what) {
    case fault::reject:
    case fault::failed_entry:
    case fault::short_entry:
      CHECK(stats.batch_failures >= 1);
      CHECK(stats.batch_reads == 6);  // the reader stays in service
      break;
    case fault::get_status:
      // The reader gave up for good; the first read was finished synchronously, the later
      // ones took the synchronous path.
      CHECK(stats.batch_failures >= 1);
      CHECK(stats.host_reads >= 2);
      CHECK(watch.with([](intercept_state& s) { return s.cancels; }) >= 1);
      break;
    case fault::setup:
      CHECK(stats.batch_reads == 0);
      CHECK(stats.host_reads == 6);
      break;
  }
  require_queued_bytes_zero(ctx);
}

// ---------------------------------------------------------------------------
// 5. Shutdown with waves in flight.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: shutdown with reads in flight settles every future",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  pattern_file const file{(64 * MiB) + 1234};
  auto const destroy = GENERATE(false, true);
  CAPTURE(destroy);
  for (int round = 0; round < 3; ++round) {
    started_ioctx ioctx{batch_config(8, 1 * MiB), 2};
    auto& ctx   = ioctx.get();
    auto object = open_registered(ctx, file);
    constexpr std::size_t reads = 24;
    host_memory memory{reads * 4 * MiB, false};
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    for (std::size_t i = 0; i < reads; ++i) {
      futures.push_back(ctx.host_read_async_io(
        *object, (i * 2 * MiB) % (60 * MiB), 4 * MiB, memory.data() + i * 4 * MiB));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{round * 3});
    auto const start = clock_type::now();
    if (destroy) {
      object.reset();
      ioctx.owner().reset();  // destroying the ioctx shuts it down
    } else {
      ctx.shutdown();
    }
    std::size_t ok = 0, failed = 0;
    for (std::size_t i = 0; i < reads; ++i) {
      try {
        auto const got = std::move(futures[i]).get(30s);
        CHECK(got == 4 * MiB);
        CHECK(std::memcmp(memory.data() + i * 4 * MiB,
                          file.data() + (i * 2 * MiB) % (60 * MiB),
                          4 * MiB) == 0);
        ++ok;
      } catch (std::system_error const&) {
        ++failed;
      }
    }
    CHECK(ok + failed == reads);
    CHECK(clock_type::now() - start < 30s);
  }
}

// ---------------------------------------------------------------------------
// 6. Several batch reactors (exclusive gate) and the flat-pool mix.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: several batch reactors take turns", "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  auto const pool = GENERATE(as<std::size_t>{}, 3, 0);
  if (pool == 0 && !has_cuda_device()) SKIP("requires a CUDA device");
  CAPTURE(pool);
  pattern_file const file{(32 * MiB) + 1234};
  started_ioctx ioctx{batch_config(8, 512 * KiB, 3), pool};
  auto& ctx   = ioctx.get();
  auto object = open_registered(ctx, file);
  constexpr int threads = 4;
  std::atomic<int> bad{0};
  std::vector<std::thread> workers;
  auto const start = clock_type::now();
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      std::mt19937_64 rng(static_cast<std::uint64_t>(t) + 11);
      for (int round = 0; round < 6; ++round) {
        host_plan plan;
        for (int i = 0; i < 6; ++i) {
          auto const size   = (rng() % (2 * MiB)) + 1;
          auto const offset = rng() % (file.size() - 1);
          plan.add(offset, size, rng() % 4 != 0);
        }
        host_memory arena{plan.arena_bytes(), false};
        try {
          auto const got = ctx.host_readv_async_io(*object, plan.slices(arena.data())).get(60s);
          if (got != plan.expected_bytes(file.size()) ||
              !plan.verify(arena.data(), arena.size(), file)) {
            ++bad;
          }
        } catch (...) {
          ++bad;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(bad.load() == 0);
  CHECK(clock_type::now() - start < 120s);
  std::size_t serving = 0;
  for (std::size_t i = 0; i < ctx.reactor_count(); ++i) {
    serving += ctx.reactor(i).stats().batch_reads != 0 ? 1U : 0U;
  }
  CHECK(serving >= 2);
  require_queued_bytes_zero(ctx);
}

TEST_CASE("host batch ioctx: a flat pool mixes batch reads, large writes and device reads",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  if (!has_cuda_device()) SKIP("requires a CUDA device");
  pattern_file const file{(32 * MiB) + 1234};
  auto const out_path = std::filesystem::temp_directory_path() /
                        ("cucascade_host_batch_ioctx_out_" + std::to_string(::getpid()) + ".bin");
  started_ioctx ioctx{batch_config(8, 1 * MiB), 2};
  auto& ctx    = ioctx.get();
  auto object  = open_registered(ctx, file);
  auto written = ctx.open_io_object_for_write(out_path.string());
  std::atomic<int> bad{0};
  auto const start = clock_type::now();
  std::vector<std::thread> workers;
  // Host batch reads.
  for (int t = 0; t < 2; ++t) {
    workers.emplace_back([&, t] {
      host_memory memory{8 * MiB, false};
      for (int i = 0; i < 12; ++i) {
        auto const offset = static_cast<std::size_t>((t * 12 + i) % 24) * MiB;
        try {
          if (ctx.host_read_async_io(*object, offset, 8 * MiB, memory.data()).get(60s) != 8 * MiB ||
              std::memcmp(memory.data(), file.data() + offset, 8 * MiB) != 0) {
            ++bad;
          }
        } catch (...) {
          ++bad;
        }
      }
    });
  }
  // Large host writes (synchronous on the same reactors).
  workers.emplace_back([&] {
    std::vector<std::uint8_t> source(16 * MiB, 0x3C);
    for (int i = 0; i < 6; ++i) {
      try {
        auto const got = ctx.host_write_async(*written, static_cast<std::size_t>(i) * 16 * MiB,
                                              source.size(), source.data())
                           .get(60s);
        if (got != source.size()) ++bad;
      } catch (...) {
        ++bad;
      }
    }
  });
  // Device reads (synchronous cuFile / POSIX paths).
  workers.emplace_back([&] {
    void* device = nullptr;
    if (cudaMalloc(&device, 4 * MiB) != cudaSuccess) {
      ++bad;
      return;
    }
    std::vector<std::uint8_t> host(4 * MiB);
    for (int i = 0; i < 12; ++i) {
      auto const offset = static_cast<std::size_t>(i) * 2 * MiB + 333;
      try {
        auto const got = ctx.device_read_async_io(*object, offset, 4 * MiB,
                                                  static_cast<std::uint8_t*>(device),
                                                  ::cuda::stream_ref{cudaStreamPerThread})
                           .get(60s);
        if (got != 4 * MiB ||
            cudaMemcpy(host.data(), device, 4 * MiB, cudaMemcpyDeviceToHost) != cudaSuccess ||
            std::memcmp(host.data(), file.data() + offset, 4 * MiB) != 0) {
          ++bad;
        }
      } catch (...) {
        ++bad;
      }
    }
    std::ignore = cudaFree(device);
  });
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(bad.load() == 0);
  CHECK(clock_type::now() - start < 120s);
  auto const stats = ctx.stats();
  CHECK(stats.batch_reads > 0);
  require_queued_bytes_zero(ctx);
  written.reset();
  object.reset();
  ctx.shutdown();
  std::error_code ec;
  std::filesystem::remove(out_path, ec);
}

TEST_CASE("host batch ioctx: a batch backlog does not hold back other traffic",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  if (!has_cuda_device()) SKIP("requires a CUDA device");
  pattern_file const file{(64 * MiB) + 1234};
  auto const out_path = std::filesystem::temp_directory_path() /
                        ("cucascade_host_batch_backlog_" + std::to_string(::getpid()) + ".bin");
  batch_intercept watch;
  // Slow waves of 4 x 256 KiB: a 32 MiB read keeps the batch backlog occupied for ~1 s.
  watch.with([](intercept_state& s) { return s.submit_delay = std::chrono::milliseconds{30}; });
  auto cfg                       = batch_config(4, 256 * KiB);
  cfg.host_batch.submit_policy   = batch_submit_policy::exclusive;
  started_ioctx ioctx{cfg, 1};  // one reactor: everything below lands on it
  auto& ctx    = ioctx.get();
  auto object  = open_registered(ctx, file);
  auto written = ctx.open_io_object_for_write(out_path.string());
  host_memory big{32 * MiB, false};
  auto const start = clock_type::now();
  auto long_read   = ctx.host_read_async_io(*object, 0, 32 * MiB, big.data(), io_priority::low);
  std::this_thread::sleep_for(50ms);  // the long read now sits in the batch backlog

  using ms = std::chrono::duration<double, std::milli>;
  // A write, a device read and a small high-priority host read on the same reactor.
  std::vector<std::uint8_t> source(4 * MiB, 0x5A);
  auto t0         = clock_type::now();
  auto const wrote = ctx.host_write_async(*written, 0, source.size(), source.data()).get(30s);
  auto const write_ms = ms(clock_type::now() - t0).count();
  CHECK(wrote == source.size());

  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, 4 * MiB) == cudaSuccess);
  t0              = clock_type::now();
  auto const read = ctx.device_read_async_io(*object, 333, 4 * MiB,
                                             static_cast<std::uint8_t*>(device),
                                             ::cuda::stream_ref{cudaStreamPerThread})
                      .get(30s);
  auto const device_ms = ms(clock_type::now() - t0).count();
  CHECK(read == 4 * MiB);
  std::vector<std::uint8_t> host(4 * MiB);
  REQUIRE(cudaMemcpy(host.data(), device, 4 * MiB, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(std::memcmp(host.data(), file.data() + 333, 4 * MiB) == 0);
  std::ignore = cudaFree(device);

  host_memory small{8 * KiB, false};
  t0 = clock_type::now();
  auto const small_read =
    ctx.host_read_async_io(*object, 5 * MiB + 7, 8 * KiB, small.data(), io_priority::high).get(30s);
  auto const small_ms = ms(clock_type::now() - t0).count();
  CHECK(small_read == 8 * KiB);
  CHECK(std::memcmp(small.data(), file.data() + 5 * MiB + 7, 8 * KiB) == 0);

  CHECK(std::move(long_read).get(60s) == 32 * MiB);
  auto const long_ms = ms(clock_type::now() - start).count();
  CHECK(std::memcmp(big.data(), file.data(), 32 * MiB) == 0);
  INFO("write " << write_ms << " ms, device read " << device_ms << " ms, small host read "
                << small_ms << " ms, long batch read " << long_ms << " ms");
  CHECK(long_ms > 600);  // the backlog really was there
  CHECK(write_ms < 400);
  CHECK(device_ms < 400);
  CHECK(small_ms < 400);
  require_queued_bytes_zero(ctx);
  written.reset();
  object.reset();
  ctx.shutdown();
  std::error_code ec;
  std::filesystem::remove(out_path, ec);
}

// ---------------------------------------------------------------------------
// 7. Both features: fused lanes for device reads, batch for host reads.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: lanes and batch together", "[io][cufile][host_batch_ioctx][gpu]")
{
  require_batch_cufile();
  if (!has_cuda_device()) SKIP("requires a CUDA device");
  auto const* api = cufile_api::get();
  if (!api->has_async_read) SKIP("libcufile without cuFileReadAsync");
  auto const pool = GENERATE(as<std::size_t>{}, 2, 0);
  CAPTURE(pool);
  auto cfg                             = batch_config(8, 1 * MiB, 2);
  cfg.gds_threshold                    = block;
  cfg.max_staging_bytes_per_device     = 0;
  cfg.reactors_per_device              = 2;
  cfg.fused_reads.enabled              = true;
  cfg.fused_reads.reactors_per_device  = 2;
  cfg.fused_reads.lanes_per_reactor    = 4;
  cfg.fused_reads.max_lanes_per_device = 64;
  cfg.fused_reads.slot_bytes           = 1 * MiB;
  pattern_file const file{(32 * MiB) + 1234};
  started_ioctx ioctx{cfg, pool};
  auto& ctx   = ioctx.get();
  auto object = open_registered(ctx, file);
  std::atomic<int> bad{0};
  std::vector<std::thread> workers;
  auto const start = clock_type::now();
  for (int t = 0; t < 2; ++t) {
    // Host reads.
    workers.emplace_back([&, t] {
      std::mt19937_64 rng(static_cast<std::uint64_t>(t) + 100);
      for (int round = 0; round < 8; ++round) {
        host_plan plan;
        for (int i = 0; i < 4; ++i) {
          plan.add((rng() % (file.size() - 1)), (rng() % (3 * MiB)) + 1, true);
        }
        host_memory arena{plan.arena_bytes(), true};
        try {
          auto const got = ctx.host_readv_async_io(*object, plan.slices(arena.data())).get(60s);
          if (got != plan.expected_bytes(file.size()) ||
              !plan.verify(arena.data(), arena.size(), file)) {
            ++bad;
          }
        } catch (...) {
          ++bad;
        }
      }
    });
    // Device reads.
    workers.emplace_back([&, t] {
      std::mt19937_64 rng(static_cast<std::uint64_t>(t) + 200);
      cudaStream_t stream{};
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        ++bad;
        return;
      }
      void* device = nullptr;
      if (cudaMalloc(&device, 8 * MiB) != cudaSuccess) {
        ++bad;
        return;
      }
      std::vector<std::uint8_t> host(8 * MiB);
      for (int round = 0; round < 8; ++round) {
        std::vector<io_slice> slices;
        std::vector<std::pair<std::size_t, std::size_t>> where;  // (offset, size)
        std::size_t at = 0;
        for (int i = 0; i < 6; ++i) {
          auto const size   = (rng() % (1 * MiB)) + block;
          auto const offset = rng() % (file.size() - size);
          slices.emplace_back(offset, size, static_cast<std::uint8_t*>(device) + at);
          where.emplace_back(offset, size);
          at += size + 64;
        }
        try {
          auto const got =
            ctx.device_readv_async_io(*object, slices, ::cuda::stream_ref{stream}).get(60s);
          if (cudaMemcpy(host.data(), device, at, cudaMemcpyDeviceToHost) != cudaSuccess) ++bad;
          std::size_t pos = 0, total = 0;
          for (auto const& [offset, size] : where) {
            if (std::memcmp(host.data() + pos, file.data() + offset, size) != 0) ++bad;
            pos += size + 64;
            total += size;
          }
          if (got != total) ++bad;
        } catch (...) {
          ++bad;
        }
      }
      std::ignore = cudaFree(device);
      std::ignore = cudaStreamDestroy(stream);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(bad.load() == 0);
  CHECK(clock_type::now() - start < 120s);
  auto const stats = ctx.stats();
  CHECK(stats.lane_reads > 0);
  CHECK(stats.batch_reads > 0);
  require_queued_bytes_zero(ctx);
}

// ---------------------------------------------------------------------------
// 8. Priorities.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: a high read is not starved behind a long low one",
          "[io][cufile][host_batch_ioctx]")
{
  require_batch_cufile();
  pattern_file const file{(128 * MiB) + 1234};
  started_ioctx ioctx{batch_config(8, 1 * MiB), 1};
  auto& ctx   = ioctx.get();
  auto object = open_registered(ctx, file);
  host_memory low{4 * 32 * MiB, false};
  host_memory high{256 * KiB, false};
  std::vector<cucascade::exec::semi_future<std::size_t>> lows;
  for (std::size_t i = 0; i < 4; ++i) {
    lows.push_back(ctx.host_read_async_io(
      *object, i * 32 * MiB, 32 * MiB, low.data() + i * 32 * MiB, io_priority::low));
  }
  auto const start = clock_type::now();
  auto const got =
    ctx.host_read_async_io(*object, 5 * MiB, high.size(), high.data(), io_priority::high).get(30s);
  auto const high_done = clock_type::now() - start;
  CHECK(got == high.size());
  CHECK(std::memcmp(high.data(), file.data() + 5 * MiB, high.size()) == 0);
  std::size_t total = 0;
  for (auto& future : lows) {
    total += std::move(future).get(60s);
  }
  auto const all_done = clock_type::now() - start;
  CHECK(total == 4 * 32 * MiB);
  CHECK(std::memcmp(low.data(), file.data(), 4 * 32 * MiB) == 0);
  // The high read waits at most for the low slice being staged, not for every low read.
  CHECK(high_done < 10s);
  using ms           = std::chrono::duration<double, std::milli>;
  auto const high_ms = ms(high_done).count();
  auto const all_ms  = ms(all_done).count();
  WARN("high read after " << high_ms << " ms; all low reads after " << all_ms << " ms");
}

// ---------------------------------------------------------------------------
// 9. Host batch off: unchanged behaviour.
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: off leaves the synchronous host path unchanged",
          "[io][cufile][host_batch_ioctx]")
{
  pattern_file const file{(9 * MiB) + 1234};
  auto const pool = GENERATE(as<std::size_t>{}, 2, 0);
  if (pool == 0 && !has_cuda_device()) SKIP("requires a CUDA device");
  config cfg;  // host_batch.enabled stays false
  started_ioctx ioctx{cfg, pool};
  auto& ctx       = ioctx.get();
  auto object     = ctx.open_io_object(file.path());
  auto const plan = alignment_plan(file.size());
  run_vectored(ctx, *object, file, plan, false);
  auto const stats = ctx.stats();
  CHECK(stats.batch_reads == 0);
  CHECK(stats.batch_bytes == 0);
  CHECK(stats.batch_failures == 0);
  CHECK(stats.host_reads == plan.count());
  CHECK(stats.host_bytes == plan.expected_bytes(file.size()));
  CHECK(ctx.plan().host_batch_reactors == 0);
  require_queued_bytes_zero(ctx);
}

// ---------------------------------------------------------------------------
// Measurement (hidden).
// ---------------------------------------------------------------------------

TEST_CASE("host batch ioctx: host read throughput, batch on and off", "[.][host_batch_ioctx_bench]")
{
  require_batch_cufile();
  if (!has_cuda_device()) SKIP("requires a CUDA device");
  constexpr std::size_t file_bytes = 1UL << 30;
  constexpr std::size_t read_bytes = 16 * MiB;
  constexpr std::size_t in_flight  = 16;
  pattern_file const file{file_bytes};
  struct variant {
    char const* name;
    bool batch;
    std::size_t batch_reactors;
    std::size_t pool;
    batch_submit_policy policy{batch_submit_policy::exclusive};
  };
  for (auto const& v :
       {variant{"sync, planned", false, 0, 0},
        variant{"batch, 1 batch reactor", true, 1, 0},
        variant{"batch, 2 batch reactors", true, 2, 0},
        variant{"batch, 2 batch reactors, per_reader", true, 2, 0, batch_submit_policy::per_reader},
        variant{"sync, flat 4", false, 0, 4},
        variant{"batch, flat 4", true, 0, 4},
        variant{"batch, flat 4, per_reader", true, 0, 4, batch_submit_policy::per_reader}}) {
    auto cfg = batch_config(64, 8 * MiB, std::max<std::size_t>(1, v.batch_reactors));
    cfg.host_batch.enabled       = v.batch;
    cfg.host_batch.submit_policy = v.policy;
    started_ioctx ioctx{cfg, v.pool};
    auto& ctx   = ioctx.get();
    auto object = ctx.open_io_object(file.path());
    host_memory memory{in_flight * read_bytes, true};
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    auto const start = clock_type::now();
    std::size_t next = 0, total = 0;
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t offset = 0; offset < file_bytes; offset += read_bytes) {
        if (futures.size() == in_flight) {
          total += std::move(futures.front()).get(60s);
          futures.erase(futures.begin());
        }
        futures.push_back(ctx.host_read_async_io(
          *object, offset, read_bytes, memory.data() + (next++ % in_flight) * read_bytes));
      }
    }
    for (auto& future : futures) {
      total += std::move(future).get(60s);
    }
    auto const seconds = std::chrono::duration<double>(clock_type::now() - start).count();
    auto const stats   = ctx.stats();
    CHECK(total == 2 * file_bytes);
    WARN(v.name << ": " << static_cast<double>(total) / seconds / 1e9 << " GB/s (batch_reads "
                << stats.batch_reads << ", host_reads " << stats.host_reads << ")");
  }
}

