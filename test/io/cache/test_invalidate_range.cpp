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

// Write invalidation of the prefetching cache, driven through the ioctx write
// wrappers on a small synchronous fake backend (pread/pwrite) whose reads and
// writes can be held back to stage the races deterministically.

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cache {

struct fs_cache_test_access {
  static cache_handle insert(fs_cache& cache,
                             io_object const& obj,
                             std::span<byte_range const> ranges)
  {
    return cache.initiate_prefetching_request(obj, ranges);
  }

  static prepare_result prepare(fs_cache& cache, cache_handle& handle)
  {
    return cache.prepare(handle, /*wait_for_eviction=*/true);
  }
};

}  // namespace cucascade::io::cache

namespace {

using cucascade::io::byte_range;
using cucascade::io::host_source;
using cucascade::io::io_context_type;
using cucascade::io::io_object;
using cucascade::io::ioctx;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::request_class;
using cucascade::io::write_options;
using cucascade::io::write_segment;
using cucascade::io::cache::cache_handle;
using cucascade::io::cache::chunk_state;
using cucascade::io::cache::fs_cache;
using cucascade::io::cache::fs_cache_test_access;
using cucascade::io::cache::prepare_result;

constexpr std::size_t chunk_bytes = 64 * 1024;
constexpr std::size_t n_chunks    = 4;
constexpr std::size_t file_bytes  = n_chunks * chunk_bytes;

void pread_all(int fd, std::uint8_t* dst, std::size_t size, std::size_t offset)
{
  while (size > 0) {
    auto const n = ::pread(fd, dst, size, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) { continue; }
    if (n <= 0) { throw std::system_error(errno, std::generic_category(), "pread"); }
    auto const got = static_cast<std::size_t>(n);
    dst += got;
    offset += got;
    size -= got;
  }
}

void pwrite_all(int fd, std::uint8_t const* src, std::size_t size, std::size_t offset)
{
  while (size > 0) {
    auto const n = ::pwrite(fd, src, size, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) { continue; }
    if (n <= 0) { throw std::system_error(errno, std::generic_category(), "pwrite"); }
    auto const put = static_cast<std::size_t>(n);
    src += put;
    offset += put;
    size -= put;
  }
}

class fake_file final : public io_object {
 public:
  fake_file(std::string path, int fd, std::size_t size)
    : _path(std::move(path)), _fd(fd), _size(size)
  {
  }
  ~fake_file() override { ::close(_fd); }
  fake_file(fake_file const&)            = delete;
  fake_file& operator=(fake_file const&) = delete;

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }
  [[nodiscard]] int fd() const noexcept { return _fd; }

 private:
  std::string _path;
  int _fd;
  std::size_t _size;
};

/// Synchronous pread/pwrite backend.  Reads and writes complete inline unless
/// held, in which case their work is parked until release_*().  Records the
/// request class of every read that reaches it.
class fake_ioctx final : public ioctx {
 public:
  ~fake_ioctx() override { pre_destroy(); }

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::uring; }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_write() const noexcept override { return true; }

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges, std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }

  std::size_t host_read_io(io_object const& obj,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override
  {
    pread_all(as_file(obj).fd(), dst, size, offset);
    return size;
  }

  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    io_object const& obj,
    std::vector<prepared_io_slice>&& slices,
    cucascade::io::io_options opts) noexcept override
  {
    auto work = [owner = obj.shared_from_this(), slices = std::move(slices), this](
                  std::function<void()> const& between_read_and_publish) mutable -> std::size_t {
      auto const& file  = as_file(*owner);
      std::size_t total = 0;
      bool ok           = true;
      try {
        for (auto& slice : slices) {
          total += slice.size();
          if (slice.is_fragmented()) {
            for (auto* chunk : slice.h_buffer.fragments()) {
              auto [lo, hi] = cucascade::io::cache::fill_span(
                chunk->state.get_fill(), chunk->offset, chunk_bytes);
              hi = std::min(hi, file.size());
              pread_all(file.fd(), chunk->data + (lo - chunk->offset), hi - lo, lo);
            }
          } else {
            pread_all(file.fd(),
                      std::get<std::uint8_t*>(slice.h_buffer.buffer),
                      slice.size(),
                      slice.offset());
          }
        }
      } catch (...) {
        ok = false;
      }
      if (between_read_and_publish) { between_read_and_publish(); }
      if (_jitter.load(std::memory_order_relaxed)) { std::this_thread::yield(); }
      for (auto& slice : slices) {
        if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), ok); }
      }
      if (!ok) { throw std::runtime_error("fake_ioctx: read failed"); }
      return total;
    };

    try {
      {
        std::lock_guard lock(_mutex);
        _seen_classes.push_back(opts.cls);
      }
      if (!_hold_reads.load()) {
        return cucascade::exec::make_semi_future<std::size_t>(work(std::function<void()>{}));
      }
      auto promise = std::make_shared<cucascade::exec::promise<std::size_t>>();
      auto future  = promise->get_semi_future();
      std::lock_guard lock(_mutex);
      _held_reads.emplace_back(
        [work = std::move(work), promise](std::function<void()> const& hook) mutable {
          try {
            promise->set_value(work(hook));
          } catch (...) {
            promise->set_exception(std::current_exception());
          }
        });
      return future;
    } catch (...) {
      return cucascade::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }

  void hold_reads(bool hold) { _hold_reads = hold; }
  void hold_writes(bool hold) { _hold_writes = hold; }
  void set_jitter(bool jitter) { _jitter = jitter; }

  /// Run every held read; @p hook runs after each read's pread and before its
  /// completion publishes the chunks.
  void release_reads(std::function<void()> const& hook = {})
  {
    std::vector<std::function<void(std::function<void()> const&)>> held;
    {
      std::lock_guard lock(_mutex);
      held.swap(_held_reads);
    }
    for (auto& work : held) {
      work(hook);
    }
  }

  void release_writes()
  {
    std::vector<std::function<void()>> held;
    {
      std::lock_guard lock(_mutex);
      held.swap(_held_writes);
    }
    for (auto& work : held) {
      work();
    }
  }

  [[nodiscard]] std::size_t held_writes() const
  {
    std::lock_guard lock(_mutex);
    return _held_writes.size();
  }

  /// Request class of every read that reached the backend, in arrival order.
  [[nodiscard]] std::vector<request_class> seen_classes() const
  {
    std::lock_guard lock(_mutex);
    return _seen_classes;
  }

  /// Request class of the most recent read, if any reached the backend.
  [[nodiscard]] std::optional<request_class> last_class() const
  {
    std::lock_guard lock(_mutex);
    if (_seen_classes.empty()) { return std::nullopt; }
    return _seen_classes.back();
  }

  /// The next commit stays pending until finish_commit().
  void hold_commit()
  {
    std::lock_guard lock(_mutex);
    _held_commit.emplace();
  }

  void finish_commit(bool ok)
  {
    std::optional<cucascade::exec::promise<void>> held;
    {
      std::lock_guard lock(_mutex);
      held.swap(_held_commit);
    }
    REQUIRE(held.has_value());
    if (ok) {
      held->set_value();
    } else {
      held->set_exception(std::make_exception_ptr(std::runtime_error("commit failed")));
    }
  }

 protected:
  std::shared_ptr<io_object> create_io_object(std::string path) override
  {
    int const fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open"); }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::system_error(errno, std::generic_category(), "fstat");
    }
    return std::make_shared<fake_file>(std::move(path), fd, static_cast<std::size_t>(st.st_size));
  }

  std::size_t host_write_io(io_object const& obj,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t const* src,
                            write_options) override
  {
    pwrite_all(as_file(obj).fd(), src, size, offset);
    return size;
  }

  cucascade::exec::semi_future<std::size_t> mixed_writev_async_io(
    io_object const& obj, std::vector<write_segment>&& segments, write_options) noexcept override
  {
    try {
      auto work = [owner = obj.shared_from_this(), segments = std::move(segments)] {
        std::size_t total = 0;
        for (auto const& segment : segments) {
          auto const& src = std::get<host_source>(segment.src);
          pwrite_all(as_file(*owner).fd(), src.data, segment.size(), segment.offset());
          total += segment.size();
        }
        return total;
      };
      if (!_hold_writes.load()) { return cucascade::exec::make_semi_future<std::size_t>(work()); }
      auto promise = std::make_shared<cucascade::exec::promise<std::size_t>>();
      auto future  = promise->get_semi_future();
      std::lock_guard lock(_mutex);
      _held_writes.emplace_back([work = std::move(work), promise]() mutable {
        try {
          promise->set_value(work());
        } catch (...) {
          promise->set_exception(std::current_exception());
        }
      });
      return future;
    } catch (...) {
      return cucascade::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }

  cucascade::exec::semi_future<void> commit_async_io(
    io_object const&, cucascade::io::write_durability) noexcept override
  {
    std::lock_guard lock(_mutex);
    if (_held_commit.has_value()) { return _held_commit->get_semi_future(); }
    return cucascade::exec::make_semi_future<void>(cucascade::exec::try_t<void>::make_value());
  }

 private:
  static fake_file const& as_file(io_object const& obj)
  {
    return dynamic_cast<fake_file const&>(obj);
  }

  std::atomic<bool> _hold_reads{false};
  std::atomic<bool> _hold_writes{false};
  std::atomic<bool> _jitter{false};
  mutable std::mutex _mutex;
  std::vector<std::function<void(std::function<void()> const&)>> _held_reads;
  std::vector<std::function<void()>> _held_writes;
  std::optional<cucascade::exec::promise<void>> _held_commit;
  std::vector<request_class> _seen_classes;
};

std::vector<cucascade::memory::memory_space_config> host_space_configs()
{
  cucascade::memory::host_memory_space_config host;
  host.numa_id              = 0;
  host.memory_capacity      = 64UL << 20;
  host.block_size           = chunk_bytes;
  host.pool_size            = 64;
  host.initial_number_pools = 1;
  return {host};
}

std::uint8_t pattern(std::size_t offset, std::uint8_t seed) noexcept
{
  return static_cast<std::uint8_t>((offset * 131U + seed) & 0xFFU);
}

struct cache_fixture {
  cache_fixture()
    : manager(host_space_configs()), ctx(std::make_shared<fake_ioctx>()), path(make_file())
  {
    cucascade::io::cache::config cfg;
    cfg.mode                            = cucascade::io::cache::cache_mode::cucs;
    cfg.min_prefetching_budget_fraction = 0.5;
    ctx->initialize_cache(manager, cfg, nullptr);
    REQUIRE(ctx->cache() != nullptr);
    REQUIRE(ctx->cache()->chunk_size() == chunk_bytes);
    obj = ctx->open_io_object(path);
    REQUIRE(obj->size() == file_bytes);
  }

  ~cache_fixture()
  {
    ctx->shutdown_cache();
    obj.reset();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  cache_fixture(cache_fixture const&)            = delete;
  cache_fixture& operator=(cache_fixture const&) = delete;

  static std::string make_file()
  {
    auto tmpl = (std::filesystem::temp_directory_path() / "cucascade_invalidate_XXXXXX").string();
    int const fd = ::mkstemp(tmpl.data());
    REQUIRE(fd >= 0);
    std::vector<std::uint8_t> bytes(file_bytes);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = pattern(i, 0);
    }
    pwrite_all(fd, bytes.data(), bytes.size(), 0);
    ::close(fd);
    return tmpl;
  }

  [[nodiscard]] fs_cache& cache() { return *ctx->cache(); }

  /// Register the whole file with the cache and attach buffers to its chunks.
  cache_handle insert_and_prepare()
  {
    byte_range const whole{0, static_cast<std::int64_t>(file_bytes)};
    auto handle =
      fs_cache_test_access::insert(cache(), *obj, std::span<byte_range const>{&whole, 1});
    REQUIRE(handle);
    REQUIRE(fs_cache_test_access::prepare(cache(), handle) == prepare_result::prepared);
    REQUIRE(handle.chunks()->size() == n_chunks);
    return handle;
  }

  /// Prefetch every chunk of @p handle and wait for the outcome.
  bool prefetch(cache_handle& handle)
  {
    std::atomic<int> result{-1};
    std::ignore = cache().prefetch(handle, [&result](bool ok) noexcept { result = ok ? 1 : 0; });
    while (result.load() < 0) {
      std::this_thread::yield();
    }
    return result.load() == 1;
  }

  [[nodiscard]] std::vector<std::uint8_t> read(std::size_t offset, std::size_t size)
  {
    std::vector<std::uint8_t> out(size);
    REQUIRE(ctx->host_read(*obj, offset, size, out.data()) == size);
    return out;
  }

  cucascade::memory::memory_reservation_manager manager;
  std::shared_ptr<fake_ioctx> ctx;
  std::string path;
  std::shared_ptr<io_object> obj;
};

chunk_state::value state_of(cache_handle const& handle, std::size_t index)
{
  return (*handle.chunks())[index]->state.get_state();
}

bool matches(std::vector<std::uint8_t> const& bytes,
             std::size_t base,
             std::size_t lo,
             std::size_t hi,
             std::uint8_t seed)
{
  for (auto off = lo; off < hi; ++off) {
    if (bytes[off - base] != pattern(off, seed)) { return false; }
  }
  return true;
}

}  // namespace

TEST_CASE("writes invalidate exactly the cached chunks they overlap", "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();
  REQUIRE(fx.prefetch(handle));
  for (std::size_t i = 0; i < n_chunks; ++i) {
    REQUIRE(state_of(handle, i) == chunk_state::cached);
  }
  REQUIRE(matches(fx.read(0, file_bytes), 0, 0, file_bytes, 0));

  // Synchronous write inside chunk 1.
  std::size_t const lo = chunk_bytes + 1000;
  std::size_t const hi = lo + 300;
  std::vector<std::uint8_t> payload(hi - lo);
  for (std::size_t off = lo; off < hi; ++off) {
    payload[off - lo] = pattern(off, 7);
  }
  REQUIRE(fx.ctx->host_write(*fx.obj, lo, payload.size(), payload.data()) == payload.size());
  CHECK(state_of(handle, 0) == chunk_state::cached);
  CHECK(state_of(handle, 1) == chunk_state::allocated);
  CHECK(state_of(handle, 2) == chunk_state::cached);
  CHECK(state_of(handle, 3) == chunk_state::cached);

  auto bytes = fx.read(0, file_bytes);
  CHECK(matches(bytes, 0, 0, lo, 0));
  CHECK(matches(bytes, 0, lo, hi, 7));
  CHECK(matches(bytes, 0, hi, file_bytes, 0));
  CHECK(state_of(handle, 1) == chunk_state::cached);  // the read reloaded it

  // Vectored async write straddling chunks 2 and 3.
  std::size_t const lo2 = 3 * chunk_bytes - 10;
  std::size_t const hi2 = 3 * chunk_bytes + 10;
  std::vector<std::uint8_t> payload2(hi2 - lo2);
  for (std::size_t off = lo2; off < hi2; ++off) {
    payload2[off - lo2] = pattern(off, 9);
  }
  std::vector<write_segment> segments;
  segments.push_back(write_segment{range{lo2, payload2.size()}, host_source{payload2.data()}});
  REQUIRE(std::move(fx.ctx->writev_async(*fx.obj, std::move(segments))).get() == payload2.size());
  CHECK(state_of(handle, 1) == chunk_state::cached);
  CHECK(state_of(handle, 2) == chunk_state::allocated);
  CHECK(state_of(handle, 3) == chunk_state::allocated);
  bytes = fx.read(lo2 - 100, 200);
  CHECK(matches(bytes, lo2 - 100, lo2 - 100, lo2, 0));
  CHECK(matches(bytes, lo2 - 100, lo2, hi2, 9));
  CHECK(matches(bytes, lo2 - 100, hi2, lo2 + 100, 0));

  // Direct API: past-EOF and unknown files are no-ops.
  CHECK(fx.cache().invalidate_range(*fx.obj, file_bytes, 4096) == 0);
  CHECK(fx.cache().invalidate_range(*fx.obj, 0, 0) == 0);
}

TEST_CASE("a prefetch in flight across a write is not published", "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();

  fx.ctx->hold_reads(true);
  std::atomic<int> outcome{-1};
  REQUIRE(fx.cache().prefetch(handle, [&outcome](bool ok) noexcept { outcome = ok ? 1 : 0; }));
  for (std::size_t i = 0; i < n_chunks; ++i) {
    REQUIRE(state_of(handle, i) == chunk_state::loading);
  }
  fx.ctx->hold_reads(false);

  // The prefetch read the old bytes; the write lands (both invalidation passes
  // run) before the prefetch's completion tries to publish them.
  std::size_t const lo = 2 * chunk_bytes + 4096;
  std::vector<std::uint8_t> payload(8192);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = pattern(lo + i, 42);
  }
  fx.ctx->release_reads([&] {
    REQUIRE(
      std::move(fx.ctx->host_write_async(*fx.obj, lo, payload.size(), payload.data())).get() ==
      payload.size());
  });
  REQUIRE(outcome.load() == 1);

  CHECK(state_of(handle, 0) == chunk_state::cached);
  CHECK(state_of(handle, 1) == chunk_state::cached);
  CHECK(state_of(handle, 2) == chunk_state::allocated);  // stale load refused
  CHECK_FALSE((*handle.chunks())[2]->state.load().is_stale());
  CHECK(state_of(handle, 3) == chunk_state::cached);

  auto const bytes = fx.read(2 * chunk_bytes, chunk_bytes);
  CHECK(matches(bytes, 2 * chunk_bytes, 2 * chunk_bytes, lo, 0));
  CHECK(matches(bytes, 2 * chunk_bytes, lo, lo + payload.size(), 42));
  CHECK(matches(bytes, 2 * chunk_bytes, lo + payload.size(), 3 * chunk_bytes, 0));
}

TEST_CASE("a pinned chunk invalidated by a write is never served again", "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();
  REQUIRE(fx.prefetch(handle));
  auto* chunk = (*handle.chunks())[0];
  REQUIRE(chunk->state.acquire_read());  // a reader still copying out of chunk 0

  std::vector<std::uint8_t> payload(100);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = pattern(500 + i, 3);
  }
  REQUIRE(fx.ctx->host_write(*fx.obj, 500, payload.size(), payload.data()) == payload.size());
  CHECK(chunk->state.get_state() == chunk_state::in_use);
  CHECK(chunk->state.load().is_stale());

  // New reads bypass the stale chunk.
  auto bytes = fx.read(0, 1024);
  CHECK(matches(bytes, 0, 0, 500, 0));
  CHECK(matches(bytes, 0, 500, 600, 3));
  CHECK(matches(bytes, 0, 600, 1024, 0));

  CHECK(chunk->state.release_read());
  CHECK(chunk->state.get_state() == chunk_state::allocated);
  bytes = fx.read(0, 1024);  // reloads in place
  CHECK(matches(bytes, 0, 500, 600, 3));
  CHECK(chunk->state.get_state() == chunk_state::cached);
}

TEST_CASE("a write completing after shutdown_cache never touches the destroyed cache",
          "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();
  REQUIRE(fx.prefetch(handle));
  auto gate = fx.cache().invalidation_gate();
  REQUIRE(gate->is_open());
  handle = cache_handle{};

  fx.ctx->hold_writes(true);
  std::vector<std::uint8_t> payload(4096, 0xAB);
  auto future = fx.ctx->host_write_async(*fx.obj, 0, payload.size(), payload.data());
  REQUIRE(fx.ctx->held_writes() == 1);

  fx.ctx->shutdown_cache();
  REQUIRE(fx.ctx->cache() == nullptr);
  CHECK_FALSE(gate->is_open());
  CHECK_FALSE(gate->invalidate_range(*fx.obj, 0, 4096));

  fx.ctx->release_writes();  // the completion's second pass must be a no-op
  CHECK(std::move(future).get() == payload.size());
}

TEST_CASE("shutdown_cache racing write completions is safe", "[cache][invalidate][stress]")
{
  cache_fixture fx;
  fx.ctx->shutdown_cache();

  cucascade::io::cache::config cfg;
  cfg.mode                            = cucascade::io::cache::cache_mode::cucs;
  cfg.min_prefetching_budget_fraction = 0.5;
  constexpr int rounds                = 50;
  constexpr int writes                = 64;
  std::vector<std::uint8_t> payload(512, 0x5A);

  for (int round = 0; round < rounds; ++round) {
    fx.ctx->initialize_cache(fx.manager, cfg, nullptr);
    REQUIRE(fx.ctx->cache() != nullptr);
    {
      auto handle = fx.insert_and_prepare();
      REQUIRE(fx.prefetch(handle));
    }

    fx.ctx->hold_writes(true);
    std::vector<cucascade::exec::semi_future<std::size_t>> futures;
    for (int i = 0; i < writes; ++i) {
      auto const offset = static_cast<std::size_t>(i) * 1024;
      futures.push_back(fx.ctx->host_write_async(*fx.obj, offset, payload.size(), payload.data()));
    }
    fx.ctx->hold_writes(false);

    std::atomic<bool> go{false};
    std::thread completer([&] {
      while (!go.load()) {
        std::this_thread::yield();
      }
      fx.ctx->release_writes();
    });
    go = true;
    fx.ctx->shutdown_cache();
    completer.join();
    for (auto& f : futures) {
      CHECK(std::move(f).get() == payload.size());
    }
  }
}

TEST_CASE("read-after-write through the cache sees the write under concurrent reads",
          "[cache][invalidate][stress]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();
  REQUIRE(fx.prefetch(handle));
  fx.ctx->set_jitter(true);

  constexpr std::size_t n_writers = 4;
  constexpr std::size_t n_readers = 4;
  constexpr int writes_per_writer = 4000;
  constexpr std::size_t region    = 512;
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> failures{0};
  std::atomic<std::size_t> reads_done{0};
  std::vector<std::thread> threads;

  for (std::size_t r = 0; r < n_readers; ++r) {
    threads.emplace_back([&, r] {
      std::mt19937_64 rng(r + 1);
      std::vector<std::uint8_t> buffer(file_bytes);
      while (!stop.load(std::memory_order_relaxed)) {
        auto const offset = static_cast<std::size_t>(rng() % file_bytes);
        auto const size = std::min<std::size_t>(1 + rng() % (2 * chunk_bytes), file_bytes - offset);
        try {
          std::ignore = fx.ctx->host_read(*fx.obj, offset, size, buffer.data());
        } catch (...) {
          failures.fetch_add(1);
        }
        reads_done.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  std::vector<std::thread> writers;
  for (std::size_t w = 0; w < n_writers; ++w) {
    writers.emplace_back([&, w] {
      // Each writer owns one region per chunk; regions of different writers
      // share chunks, so every write invalidates chunks other writers and the
      // readers keep reloading.
      std::vector<std::uint8_t> payload(region);
      std::vector<std::uint8_t> back(region);
      for (int i = 0; i < writes_per_writer; ++i) {
        auto const chunk  = static_cast<std::size_t>(i) % n_chunks;
        auto const offset = chunk * chunk_bytes + w * 4 * region + region / 2;
        auto const seed   = static_cast<std::uint8_t>(i + 1);
        for (std::size_t b = 0; b < region; ++b) {
          payload[b] = pattern(offset + b, seed);
        }
        bool const vectored = (i % 2) == 0;
        if (vectored) {
          std::vector<write_segment> segments;
          segments.push_back(write_segment{range{offset, region}, host_source{payload.data()}});
          std::ignore = std::move(fx.ctx->writev_async(*fx.obj, std::move(segments))).get();
        } else {
          std::ignore = fx.ctx->host_write(*fx.obj, offset, region, payload.data());
        }
        std::ignore = fx.ctx->host_read(*fx.obj, offset, region, back.data());
        if (back != payload) { failures.fetch_add(1); }
      }
    });
  }
  for (auto& thread : writers) {
    thread.join();
  }
  stop = true;
  for (auto& thread : threads) {
    thread.join();
  }

  CHECK(failures.load() == 0);
  CHECK(reads_done.load() > 0);
}

TEST_CASE("commit invalidates every cached chunk of the object once it settles",
          "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle    = fx.insert_and_prepare();
  auto all_state = [&](chunk_state::value expected) {
    for (std::size_t i = 0; i < n_chunks; ++i) {
      if (state_of(handle, i) != expected) { return false; }
    }
    return true;
  };

  // Successful commit: nothing is dropped while it is pending (object-store
  // bytes are not visible yet), everything once it resolved.
  REQUIRE(fx.prefetch(handle));
  REQUIRE(all_state(chunk_state::cached));
  fx.ctx->hold_commit();
  auto pending = fx.ctx->commit_async(*fx.obj);
  CHECK(all_state(chunk_state::cached));
  fx.ctx->finish_commit(true);
  CHECK_NOTHROW(std::move(pending).get());
  CHECK(all_state(chunk_state::allocated));

  // A failed commit invalidates as well (it may have published bytes).
  REQUIRE(matches(fx.read(0, file_bytes), 0, 0, file_bytes, 0));  // reloads every chunk
  REQUIRE(all_state(chunk_state::cached));
  fx.ctx->hold_commit();
  auto failing = fx.ctx->commit_async(*fx.obj);
  fx.ctx->finish_commit(false);
  CHECK_THROWS_WITH(std::move(failing).get(), "commit failed");
  CHECK(all_state(chunk_state::allocated));

  // Synchronously resolved commit.
  REQUIRE(matches(fx.read(0, file_bytes), 0, 0, file_bytes, 0));
  REQUIRE(all_state(chunk_state::cached));
  CHECK_NOTHROW(fx.ctx->commit_async(*fx.obj).get());
  CHECK(all_state(chunk_state::allocated));
  CHECK(matches(fx.read(0, file_bytes), 0, 0, file_bytes, 0));
}

TEST_CASE("a commit settling after shutdown_cache never touches the destroyed cache",
          "[cache][invalidate]")
{
  cache_fixture fx;
  auto handle = fx.insert_and_prepare();
  REQUIRE(fx.prefetch(handle));
  fx.ctx->hold_commit();
  auto pending = fx.ctx->commit_async(*fx.obj);
  handle       = cache_handle{};
  fx.ctx->shutdown_cache();
  fx.ctx->finish_commit(true);
  CHECK_NOTHROW(std::move(pending).get());
}

TEST_CASE("prefetch reads are background class, demand reads are not", "[io][cache]")
{
  using namespace std::chrono_literals;

  cache_fixture fx;
  // A request naming chunks 0 and 1 only: chunks 2 and 3 stay uncovered.
  byte_range const head{0, static_cast<std::int64_t>(2 * chunk_bytes)};
  auto handle =
    fs_cache_test_access::insert(fx.cache(), *fx.obj, std::span<byte_range const>{&head, 1});
  REQUIRE(handle);
  REQUIRE(fs_cache_test_access::prepare(fx.cache(), handle) == prepare_result::prepared);
  REQUIRE(handle.chunks()->size() == 2);
  CHECK(handle.demand_wait_ns() == 0);

  // The prefetch reaches the backend as background class; hold it there so a
  // demand read through the handle has to wait for it.
  fx.ctx->hold_reads(true);
  std::atomic<int> outcome{-1};
  REQUIRE(fx.cache().prefetch(handle, [&outcome](bool ok) noexcept { outcome = ok ? 1 : 0; }));
  fx.ctx->hold_reads(false);
  REQUIRE(fx.ctx->seen_classes() == std::vector<request_class>{request_class::background});
  REQUIRE(handle.is_prefetch_in_flight());

  // A demand read inside chunk 0 blocks on the in-flight prefetch, then is
  // served from the chunk it published: no backend read of its own.
  std::thread releaser([&fx] {
    std::this_thread::sleep_for(50ms);
    fx.ctx->release_reads();
  });
  std::size_t const head_offset = 100;
  std::vector<std::uint8_t> head_bytes(1024);
  auto const head_got =
    std::move(
      fx.ctx->host_read_async(*fx.obj, head_offset, head_bytes.size(), head_bytes.data(), &handle))
      .get();
  releaser.join();
  REQUIRE(head_got == head_bytes.size());
  REQUIRE(outcome.load() == 1);
  CHECK(matches(head_bytes, head_offset, head_offset, head_offset + head_bytes.size(), 0));
  CHECK(fx.ctx->seen_classes().size() == 1);
  auto const waited = handle.demand_wait_ns();
  CHECK(waited > 0);

  // A demand read of a range the request does not cover goes to the backend
  // with a demand class, never background.
  std::size_t const tail_offset = 3 * chunk_bytes;
  std::vector<std::uint8_t> tail_bytes(chunk_bytes);
  REQUIRE(std::move(fx.ctx->host_read_async(
                      *fx.obj, tail_offset, tail_bytes.size(), tail_bytes.data(), &handle))
            .get() == tail_bytes.size());
  CHECK(matches(tail_bytes, tail_offset, tail_offset, file_bytes, 0));
  REQUIRE(fx.ctx->seen_classes().size() == 2);
  auto const demand_class = fx.ctx->last_class();
  REQUIRE(demand_class.has_value());
  CHECK(*demand_class != request_class::background);
  CHECK((*demand_class == request_class::latency || *demand_class == request_class::read));
  CHECK(handle.demand_wait_ns() == waited);  // no prefetch in flight: no further wait

  // Moves transfer the accumulated wait; the moved-from handle reports 0.
  cache_handle moved{std::move(handle)};
  CHECK(handle.demand_wait_ns() == 0);  // NOLINT(bugprone-use-after-move)
  CHECK(moved.demand_wait_ns() == waited);
  cache_handle assigned;
  assigned = std::move(moved);
  CHECK(moved.demand_wait_ns() == 0);  // NOLINT(bugprone-use-after-move)
  CHECK(assigned.demand_wait_ns() == waited);
}
