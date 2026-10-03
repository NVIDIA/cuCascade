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

// io_uring write path: host / device sources, single / vectored, sync / async,
// O_DIRECT body + buffered head/tail, durability (data_sync / flush / commit),
// write-vs-read scheduling, cache coherence and shutdown with writes in flight.

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/prefetching_cache.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cache {

// Same definition as in test/io/cache/test_invalidate_range.cpp (ODR: identical).
struct prefetching_cache_test_access {
  static prefetching_handle insert(prefetching_cache& cache,
                                   io_object const& obj,
                                   std::span<byte_range const> ranges)
  {
    return cache.initiate_prefetching_request(obj, ranges);
  }

  static prepare_result prepare(prefetching_cache& cache, prefetching_handle& handle)
  {
    return cache.prepare(handle, /*wait_for_eviction=*/true);
  }
};

}  // namespace cucascade::io::cache

namespace {

using namespace std::chrono_literals;
using cucascade::io::device_source;
using cucascade::io::host_source;
using cucascade::io::io_object;
using cucascade::io::range;
using cucascade::io::write_durability;
using cucascade::io::write_mode;
using cucascade::io::write_open_options;
using cucascade::io::write_options;
using cucascade::io::write_segment;
using cucascade::io::uring::uring_ioctx;
using cucascade::io::uring::uring_reactor;

constexpr std::size_t KiB  = 1UL << 10;
constexpr std::size_t MiB  = 1UL << 20;
constexpr std::size_t PAGE = 4096;

[[nodiscard]] std::uint8_t pattern_at(std::size_t offset, std::uint8_t seed) noexcept
{
  return static_cast<std::uint8_t>((offset * 131U + (offset >> 12U) * 7U + seed) & 0xFFU);
}

/// A temporary path (file not created), removed on destruction.
class temp_path {
 public:
  temp_path()
  {
    static std::atomic<int> counter{0};
    _path = std::filesystem::temp_directory_path() /
            ("cucascade_uring_write_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1)) + ".bin");
  }
  ~temp_path()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  temp_path(temp_path const&)            = delete;
  temp_path& operator=(temp_path const&) = delete;

  [[nodiscard]] std::string str() const { return _path.string(); }

 private:
  std::filesystem::path _path;
};

/// Pinned staging resource (1 MiB blocks) shared by the engines of a context.
class staging_resource {
 public:
  explicit staging_resource(std::size_t capacity = 320 * MiB)
    : _upstream(0, /*make_portable=*/true), _mr(0, _upstream, capacity, capacity, 1 * MiB)
  {
  }

  [[nodiscard]] std::shared_ptr<uring_reactor::reactor_context> context(bool use_odirect = true)
  {
    cucascade::io::uring::config cfg{};
    cfg.use_odirect = use_odirect;
    return std::make_shared<uring_reactor::reactor_context>(cfg, &_mr);
  }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream;
  cucascade::memory::fixed_size_host_memory_resource _mr;
};

/// Page-aligned host buffer.
class aligned_buffer {
 public:
  explicit aligned_buffer(std::size_t size)
    : _size(size),
      _data(static_cast<std::uint8_t*>(std::aligned_alloc(PAGE, (size + PAGE - 1) / PAGE * PAGE)))
  {
    REQUIRE(_data != nullptr);
  }
  ~aligned_buffer() { std::free(_data); }
  aligned_buffer(aligned_buffer const&)            = delete;
  aligned_buffer& operator=(aligned_buffer const&) = delete;

  [[nodiscard]] std::uint8_t* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

  void fill(std::size_t file_offset, std::uint8_t seed)
  {
    for (std::size_t i = 0; i < _size; ++i) {
      _data[i] = pattern_at(file_offset + i, seed);
    }
  }

 private:
  std::size_t _size;
  std::uint8_t* _data;
};

[[nodiscard]] std::vector<std::uint8_t> read_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// First index where @p actual and @p expected differ (or the shorter size).
[[nodiscard]] std::size_t first_mismatch(std::vector<std::uint8_t> const& actual,
                                         std::vector<std::uint8_t> const& expected)
{
  auto const n = std::min(actual.size(), expected.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (actual[i] != expected[i]) return i;
  }
  return actual.size() == expected.size() ? std::size_t(-1) : n;
}

void check_file(std::string const& path, std::vector<std::uint8_t> const& expected)
{
  auto const actual = read_file(path);
  CHECK(actual.size() == expected.size());
  CHECK(first_mismatch(actual, expected) == std::size_t(-1));
}

/// Number of pages of [offset, offset + size) resident in the page cache.
[[nodiscard]] std::size_t resident_pages(std::string const& path,
                                         std::size_t offset,
                                         std::size_t size)
{
  int const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  REQUIRE(fd >= 0);
  auto const length = static_cast<std::size_t>(std::filesystem::file_size(path));
  void* map         = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  REQUIRE(map != MAP_FAILED);
  std::vector<unsigned char> vec((length + PAGE - 1) / PAGE);
  REQUIRE(::mincore(map, length, vec.data()) == 0);
  ::munmap(map, length);
  std::size_t count = 0;
  for (auto page = offset / PAGE; page < (offset + size + PAGE - 1) / PAGE; ++page) {
    count += (vec[page] & 1U) != 0 ? 1 : 0;
  }
  return count;
}

/// Error code carried by a std::system_error thrown from @p fn (empty if none / other).
template <class Fn>
[[nodiscard]] std::error_code system_error_of(Fn&& fn)
{
  try {
    fn();
  } catch (std::system_error const& error) {
    return error.code();
  } catch (...) {
    return std::make_error_code(std::errc::interrupted);  // "some other exception"
  }
  return {};
}

[[nodiscard]] bool have_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// RAII device buffer + non-blocking stream.
struct device_area {
  explicit device_area(std::size_t size) : bytes(size)
  {
    if (cudaMalloc(&data, size) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      data = nullptr;
      return;
    }
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  }
  /// False when the device memory could not be allocated (GPU full).
  [[nodiscard]] bool ok() const noexcept { return data != nullptr; }
  ~device_area()
  {
    if (data == nullptr) return;
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaStreamDestroy(stream));
    static_cast<void>(cudaFree(data));
  }
  device_area(device_area const&)            = delete;
  device_area& operator=(device_area const&) = delete;

  [[nodiscard]] std::uint8_t* ptr(std::size_t offset = 0) const
  {
    return static_cast<std::uint8_t*>(data) + offset;
  }

  void* data{nullptr};
  cudaStream_t stream{nullptr};
  std::size_t bytes;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Host sources
//===----------------------------------------------------------------------===//

TEST_CASE("uring host writes round-trip", "[io][uring][write]")
{
  bool const odirect = GENERATE(true, false);
  CAPTURE(odirect);
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(1, staging.context(odirect));
  CHECK(ctx.supports_write());
  CHECK(ctx.supports_device_write());
  ctx.start();

  auto object = ctx.open_io_object_for_write(path.str());
  REQUIRE(object != nullptr);
  CHECK(object->size() == 0);

  std::vector<std::uint8_t> expected;
  auto apply = [&](std::size_t offset, std::span<std::uint8_t const> bytes) {
    if (expected.size() < offset + bytes.size()) expected.resize(offset + bytes.size(), 0);
    std::copy(bytes.begin(), bytes.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset));
  };

  SECTION("single unaligned write, then an aligned overwrite")
  {
    std::vector<std::uint8_t> first(3 * MiB + 777);
    for (std::size_t i = 0; i < first.size(); ++i) {
      first[i] = pattern_at(i, 1);
    }
    CHECK(ctx.host_write_async(*object, 0, first.size(), first.data()).get(10s) == first.size());
    apply(0, first);
    CHECK(object->size() == first.size());
    check_file(path.str(), expected);

    aligned_buffer second(8 * MiB);
    second.fill(MiB, 2);
    CHECK(ctx.host_write_async(*object, MiB, second.size(), second.data()).get(10s) ==
          second.size());
    apply(MiB, {second.data(), second.size()});
    CHECK(object->size() == MiB + second.size());
    check_file(path.str(), expected);

    // Reads through the same (writable) object see the written bytes.
    std::vector<std::uint8_t> back(expected.size());
    CHECK(ctx.host_read_async(*object, 0, back.size(), back.data()).get(10s) == back.size());
    CHECK(first_mismatch(back, expected) == std::size_t(-1));
  }

  SECTION("vectored writes at unaligned offsets and sizes")
  {
    std::mt19937_64 rng(odirect ? 7 : 11);
    // Disjoint segments laid out left to right with random gaps (0 => contiguous,
    // exercising coalescing), then shuffled.
    std::vector<std::vector<std::uint8_t>> owned;
    aligned_buffer big(4 * MiB);
    std::vector<write_segment> segments;
    std::size_t cursor = 13;
    for (int i = 0; i < 96; ++i) {
      std::size_t const gap = (rng() % 3 == 0) ? 0 : rng() % (3 * PAGE);
      cursor += gap;
      std::size_t size = 0;
      switch (rng() % 4) {
        case 0: size = 1 + rng() % 64; break;
        case 1: size = PAGE * (1 + rng() % 4) + (rng() % 2) * (rng() % PAGE); break;
        case 2: size = 1 + rng() % (300 * KiB); break;
        default: size = PAGE * (1 + rng() % 64); break;
      }
      owned.emplace_back(size);
      for (std::size_t j = 0; j < size; ++j) {
        owned.back()[j] = pattern_at(cursor + j, 3);
      }
      segments.push_back(write_segment{range{cursor, size}, host_source{owned.back().data()}});
      cursor += size;
    }
    // One page-aligned segment from an aligned buffer (O_DIRECT body candidate).
    cursor = (cursor + 5 * PAGE) / PAGE * PAGE;
    big.fill(cursor, 3);
    segments.push_back(write_segment{range{cursor, big.size()}, host_source{big.data()}});

    for (auto const& segment : segments) {
      apply(segment.offset(), {segment.data(), segment.size()});
    }
    std::shuffle(segments.begin(), segments.end() - 1, rng);
    auto const total = std::accumulate(
      segments.begin(), segments.end(), std::size_t{0}, [](std::size_t acc, auto const& s) {
        return acc + s.size();
      });
    CHECK(ctx.writev_async(*object, std::move(segments)).get(20s) == total);
    CHECK(object->size() == expected.size());
    check_file(path.str(), expected);
  }

  SECTION("contiguous small segments coalesce into one run")
  {
    std::vector<std::uint8_t> source(64 * 1000);
    for (std::size_t i = 0; i < source.size(); ++i) {
      source[i] = pattern_at(5000 + i, 4);
    }
    std::vector<write_segment> segments;
    for (std::size_t i = 0; i < 64; ++i) {
      segments.push_back(
        write_segment{range{5000 + i * 1000, 1000}, host_source{source.data() + i * 1000}});
    }
    CHECK(ctx.writev_async(*object, std::move(segments)).get(10s) == source.size());
    apply(5000, source);
    check_file(path.str(), expected);
  }

  SECTION("overlapping segments are rejected")
  {
    std::vector<std::uint8_t> source(8192, 1);
    std::vector<write_segment> segments{
      write_segment{range{0, 4096}, host_source{source.data()}},
      write_segment{range{4000, 4096}, host_source{source.data() + 4096}}};
    CHECK_THROWS_AS(ctx.writev_async(*object, std::move(segments)).get(10s), std::invalid_argument);
  }
}

TEST_CASE("uring O_DIRECT writes the aligned body and buffers the unaligned ends",
          "[io][uring][write]")
{
  bool const odirect = GENERATE(true, false);
  CAPTURE(odirect);
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(1, staging.context(odirect));
  ctx.start();
  auto object = ctx.open_io_object_for_write(path.str());
  {
    // A direct handle must exist for this test to mean anything.
    auto const* local = dynamic_cast<cucascade::io::uring::local_io_object const*>(object.get());
    REQUIRE(local != nullptr);
    if (local->fd_direct() < 0) SKIP("filesystem without O_DIRECT");
  }

  // File range [100, 100 + 8 MiB + 50): head [100, 4096), body [4096, ...),
  // tail after the last page boundary.  src + (4096 - 100) is page aligned.
  constexpr std::size_t offset = 100;
  constexpr std::size_t size   = 8 * MiB + 50;
  aligned_buffer storage(size + PAGE);
  auto* const source = storage.data() + offset;
  for (std::size_t i = 0; i < size; ++i) {
    source[i] = pattern_at(offset + i, 9);
  }
  CHECK(ctx.host_write_async(*object, offset, size, source).get(10s) == size);

  std::vector<std::uint8_t> expected(offset + size, 0);
  std::copy(source, source + size, expected.begin() + offset);
  check_file(path.str(), expected);  // exact size: never padded

  auto const body_begin = PAGE;
  auto const body_end   = (offset + size) / PAGE * PAGE;
  // check_file read the file through the page cache; measure residency on a
  // fresh write instead.
  CHECK(ctx.host_write_async(*object, offset, size, source).get(10s) == size);
  auto const body_resident = resident_pages(path.str(), body_begin, body_end - body_begin);
  auto const body_pages    = (body_end - body_begin) / PAGE;
  if (odirect) {
    // O_DIRECT invalidates the written range of the page cache.
    CHECK(body_resident == 0);
  } else {
    CHECK(body_resident == body_pages);
  }
  CHECK(resident_pages(path.str(), 0, PAGE) == 1);                             // head: buffered
  CHECK(resident_pages(path.str(), body_end, offset + size - body_end) == 1);  // tail: buffered
}

TEST_CASE("uring writes extend files, open modes and read-only objects", "[io][uring][write]")
{
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(1, staging.context());
  ctx.start();

  SECTION("write past EOF leaves a zero hole")
  {
    auto object = ctx.open_io_object_for_write(path.str());
    std::vector<std::uint8_t> data(10'000, 0x5A);
    CHECK(ctx.host_write_async(*object, 3 * MiB + 3, data.size(), data.data()).get(10s) ==
          data.size());
    CHECK(object->size() == 3 * MiB + 3 + data.size());
    std::vector<std::uint8_t> expected(3 * MiB + 3 + data.size(), 0);
    std::copy(data.begin(), data.end(), expected.begin() + 3 * MiB + 3);
    check_file(path.str(), expected);
  }

  SECTION("create_or_open keeps content, create_or_truncate truncates")
  {
    {
      std::ofstream out(path.str(), std::ios::binary);
      std::vector<char> init(PAGE * 3, 'a');
      out.write(init.data(), static_cast<std::streamsize>(init.size()));
    }
    write_open_options keep;
    keep.mode   = write_mode::create_or_open;
    auto object = ctx.open_io_object_for_write(path.str(), keep);
    CHECK(object->size() == PAGE * 3);
    std::vector<std::uint8_t> data(10, 'b');
    CHECK(ctx.host_write_async(*object, PAGE, data.size(), data.data()).get(10s) == data.size());
    std::vector<std::uint8_t> expected(PAGE * 3, 'a');
    std::fill_n(expected.begin() + PAGE, 10, 'b');
    check_file(path.str(), expected);
    CHECK(object->size() == PAGE * 3);  // overwrite inside: size unchanged

    auto truncated = ctx.open_io_object_for_write(path.str());
    CHECK(truncated->size() == 0);
    CHECK(std::filesystem::file_size(path.str()) == 0);
  }

  SECTION("open_existing requires the file")
  {
    write_open_options existing;
    existing.mode = write_mode::open_existing;
    CHECK(system_error_of([&] {
            static_cast<void>(ctx.open_io_object_for_write(path.str(), existing));
          }) == std::errc::no_such_file_or_directory);
    std::ofstream(path.str()).put('x');
    auto object = ctx.open_io_object_for_write(path.str(), existing);
    CHECK(object->size() == 1);
  }

  SECTION("size_hint does not change the visible size")
  {
    write_open_options hinted;
    hinted.size_hint = 64 * MiB;
    auto object      = ctx.open_io_object_for_write(path.str(), hinted);
    CHECK(object->size() == 0);
    CHECK(std::filesystem::file_size(path.str()) == 0);
  }

  SECTION("objects opened for reading reject writes")
  {
    std::ofstream(path.str()).put('x');
    auto object = ctx.open_io_object(path.str());
    std::uint8_t byte{1};
    CHECK_THROWS_AS(ctx.host_write(*object, 0, 1, &byte), std::invalid_argument);
    CHECK_THROWS_AS(ctx.host_write_async(*object, 0, 1, &byte).get(10s), std::invalid_argument);
    CHECK_THROWS_AS(ctx.commit_async(*object).get(10s), std::invalid_argument);
    ctx.flush_async(*object).get(10s);  // flushing a read-only file is harmless
  }
}

TEST_CASE("uring synchronous host_write works without runners", "[io][uring][write]")
{
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(0, staging.context());  // never started
  auto object = ctx.open_io_object_for_write(path.str());

  std::vector<std::uint8_t> data(1 * MiB + 17);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = pattern_at(i + 33, 5);
  }
  CHECK(ctx.host_write(*object, 33, data.size(), data.data()) == data.size());
  write_options sync;
  sync.durability = write_durability::data_sync;
  CHECK(ctx.host_write(*object, 33, 100, data.data(), sync) == 100);
  CHECK(object->size() == 33 + data.size());

  std::vector<std::uint8_t> expected(33 + data.size(), 0);
  std::copy(data.begin(), data.end(), expected.begin() + 33);
  check_file(path.str(), expected);

  // Async work needs a runner: before start() admission is closed.
  CHECK(system_error_of([&] {
          static_cast<void>(ctx.host_write_async(*object, 0, 10, data.data()).get(10s));
        }) == std::errc::operation_canceled);
}

TEST_CASE("uring durability: data_sync, flush and commit", "[io][uring][write]")
{
  staging_resource staging;
  temp_path path;
  temp_path path2;
  uring_ioctx ctx(2, staging.context());
  ctx.start();
  auto object = ctx.open_io_object_for_write(path.str());

  std::vector<std::uint8_t> data(5 * MiB + 1);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = pattern_at(i, 6);
  }
  write_options sync;
  sync.durability = write_durability::data_sync;

  // data_sync on a multi-segment request (the sync runs once, after every write).
  std::vector<write_segment> segments{
    write_segment{range{0, 2 * MiB}, host_source{data.data()}},
    write_segment{range{2 * MiB, data.size() - 2 * MiB}, host_source{data.data() + 2 * MiB}}};
  CHECK(ctx.writev_async(*object, std::move(segments), sync).get(10s) == data.size());
  // Many concurrent data_sync requests all resolve.
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < 16; ++i) {
    futures.push_back(ctx.host_write_async(*object, i * 1000, 1000, data.data() + i * 1000, sync));
  }
  for (auto& future : futures) {
    CHECK(std::move(future).get(10s) == 1000);
  }
  ctx.flush_async(*object).get(10s);
  check_file(path.str(), data);

  SECTION("commit without sync") { ctx.commit_async(*object).get(10s); }
  SECTION("commit with data_sync")
  {
    ctx.commit_async(*object, write_durability::data_sync).get(10s);
  }

  // Committed: read-only now.
  std::uint8_t byte{0};
  CHECK_THROWS_AS(ctx.host_write(*object, 0, 1, &byte), std::invalid_argument);
  CHECK_THROWS_AS(ctx.host_write_async(*object, 0, 1, &byte).get(10s), std::invalid_argument);
  CHECK_THROWS_AS(ctx.commit_async(*object).get(10s), std::invalid_argument);
  CHECK_THROWS_AS(ctx.commit_async(*object, write_durability::data_sync).get(10s),
                  std::invalid_argument);
  ctx.flush_async(*object).get(10s);
  std::vector<std::uint8_t> back(data.size());
  CHECK(ctx.host_read_async(*object, 0, back.size(), back.data()).get(10s) == back.size());
  CHECK(back == data);

  // Other objects stay writable.
  auto other = ctx.open_io_object_for_write(path2.str());
  CHECK(ctx.host_write_async(*other, 0, 10, data.data()).get(10s) == 10);
}

//===----------------------------------------------------------------------===//
// Device sources
//===----------------------------------------------------------------------===//

TEST_CASE("uring device-source writes", "[io][uring][write][gpu]")
{
  if (!have_gpu()) SKIP("no CUDA device");
  bool const odirect = GENERATE(true, false);
  CAPTURE(odirect);
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(1, staging.context(odirect));
  ctx.start();
  auto object = ctx.open_io_object_for_write(path.str());

  constexpr std::size_t area_size = 40 * MiB;
  device_area device(area_size);
  if (!device.ok()) SKIP("cannot allocate device memory (GPU full)");
  std::vector<std::uint8_t> host(area_size);
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = pattern_at(i, 8);
  }
  REQUIRE(cudaMemcpy(device.data, host.data(), host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  ::cuda::stream_ref const stream{device.stream};

  std::vector<std::uint8_t> expected;
  auto apply = [&](std::size_t offset, std::uint8_t const* bytes, std::size_t size) {
    if (expected.size() < offset + size) expected.resize(offset + size, 0);
    std::copy(bytes, bytes + size, expected.begin() + static_cast<std::ptrdiff_t>(offset));
  };

  // Aligned and large (several staged operations, O_DIRECT body).
  CHECK(ctx.device_write_async(*object, 0, 20 * MiB, device.ptr(), stream).get(20s) == 20 * MiB);
  apply(0, host.data(), 20 * MiB);
  // Unaligned offset and size (buffered head / tail).
  CHECK(ctx.device_write_async(*object, 20 * MiB + 777, 3 * MiB + 5, device.ptr(20 * MiB), stream)
          .get(20s) == 3 * MiB + 5);
  apply(20 * MiB + 777, host.data() + 20 * MiB, 3 * MiB + 5);
  CHECK(object->size() == expected.size());
  check_file(path.str(), expected);

  // Stream ordering: the D2H copy runs after work already enqueued on the stream.
  REQUIRE(cudaMemsetAsync(device.ptr(30 * MiB), 0xAB, 2 * MiB, device.stream) == cudaSuccess);
  CHECK(ctx.device_write_async(*object, 4 * MiB, 2 * MiB, device.ptr(30 * MiB), stream).get(20s) ==
        2 * MiB);
  std::fill_n(expected.begin() + static_cast<std::ptrdiff_t>(4 * MiB), 2 * MiB, 0xAB);
  check_file(path.str(), expected);

  // Mixed host + device segments in one request, with data_sync.
  std::vector<std::uint8_t> host_part(100'000);
  for (std::size_t i = 0; i < host_part.size(); ++i) {
    host_part[i] = pattern_at(i, 12);
  }
  std::vector<write_segment> segments{
    write_segment{range{25 * MiB, host_part.size()}, host_source{host_part.data()}},
    write_segment{range{26 * MiB + 1, 5 * MiB}, device_source{device.ptr(MiB), stream, -1}},
    write_segment{range{25 * MiB + host_part.size(), 4096},
                  device_source{device.ptr(), stream, -1}}};
  write_options sync;
  sync.durability = write_durability::data_sync;
  CHECK(ctx.writev_async(*object, std::move(segments), sync).get(20s) ==
        host_part.size() + 5 * MiB + 4096);
  apply(25 * MiB, host_part.data(), host_part.size());
  apply(26 * MiB + 1, host.data() + MiB, 5 * MiB);
  apply(25 * MiB + host_part.size(), host.data(), 4096);
  check_file(path.str(), expected);
}

//===----------------------------------------------------------------------===//
// Scheduling, coherence, lifecycle
//===----------------------------------------------------------------------===//

TEST_CASE("uring large writes do not starve small reads", "[io][uring][write]")
{
  staging_resource staging;
  temp_path write_path;
  temp_path read_path;
  {
    std::vector<char> init(8 * MiB);
    for (std::size_t i = 0; i < init.size(); ++i) {
      init[i] = static_cast<char>(pattern_at(i, 0));
    }
    std::ofstream(read_path.str(), std::ios::binary)
      .write(init.data(), static_cast<std::streamsize>(init.size()));
  }
  uring_ioctx ctx(1, staging.context());
  ctx.start();
  auto writable = ctx.open_io_object_for_write(write_path.str());
  auto readable = ctx.open_io_object(read_path.str());

  // 1.5 GiB of writes queued at once (one 64 MiB source reused).
  aligned_buffer source(64 * MiB);
  source.fill(0, 1);
  constexpr std::size_t n_writes = 24;
  std::vector<cucascade::exec::semi_future<std::size_t>> writes;
  std::atomic<std::size_t> writes_done{0};
  std::atomic<std::size_t> write_failures{0};
  auto const start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < n_writes; ++i) {
    writes.push_back(
      ctx.host_write_async(*writable, i * source.size(), source.size(), source.data()));
  }
  std::jthread waiter([&] {
    // No Catch assertions off the main thread.
    for (auto& write : writes) {
      try {
        if (std::move(write).get(120s) != source.size()) write_failures.fetch_add(1);
      } catch (...) {
        write_failures.fetch_add(1);
      }
      writes_done.fetch_add(1);
    }
  });

  // Small reads while the writes are in flight.
  std::vector<std::chrono::microseconds> latencies;
  std::mt19937_64 rng(3);
  std::vector<std::uint8_t> buffer(4 * KiB);
  while (writes_done.load() < n_writes && latencies.size() < 2000) {
    auto const offset = (rng() % (8 * MiB - buffer.size()));
    auto const t0     = std::chrono::steady_clock::now();
    CHECK(ctx.host_read_async(*readable, offset, buffer.size(), buffer.data()).get(10s) ==
          buffer.size());
    latencies.push_back(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0));
    CHECK(buffer[0] == pattern_at(offset, 0));
    std::this_thread::sleep_for(500us);
  }
  auto const reads_during_writes = latencies.size();
  waiter.join();
  CHECK(write_failures.load() == 0);
  auto const write_time = std::chrono::steady_clock::now() - start;

  std::sort(latencies.begin(), latencies.end());
  auto const p50 = latencies.empty() ? 0us : latencies[latencies.size() / 2];
  auto const max = latencies.empty() ? 0us : latencies.back();
  UNSCOPED_INFO(
    "writes: " << n_writes * source.size() / MiB << " MiB in "
               << std::chrono::duration_cast<std::chrono::milliseconds>(write_time).count()
               << " ms; reads during writes: " << reads_during_writes << " p50=" << p50.count()
               << "us max=" << max.count() << "us");
  CHECK(reads_during_writes >= 3);
  // Without a write budget a read would wait for the whole write backlog.
  CHECK(max < std::max<std::chrono::microseconds>(
                100ms, std::chrono::duration_cast<std::chrono::microseconds>(write_time / 4)));
  CHECK(p50 < 20ms);
}

TEST_CASE("uring writes are visible to subsequent reads", "[io][uring][write]")
{
  bool const odirect = GENERATE(true, false);
  CAPTURE(odirect);
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(2, staging.context(odirect));
  ctx.start();
  auto object = ctx.open_io_object_for_write(path.str());

  aligned_buffer base(4 * MiB);
  base.fill(0, 0);
  CHECK(ctx.host_write_async(*object, 0, base.size(), base.data()).get(10s) == base.size());

  std::mt19937_64 rng(odirect ? 21 : 22);
  aligned_buffer update(256 * KiB);
  aligned_buffer readback(256 * KiB);
  for (int round = 1; round <= 64; ++round) {
    bool const aligned = rng() % 2 == 0;
    auto const size    = aligned ? PAGE * (1 + rng() % 64) : 1 + rng() % update.size();
    auto offset        = rng() % (base.size() - size);
    if (aligned) offset = offset / PAGE * PAGE;
    auto const seed = static_cast<std::uint8_t>(round);
    for (std::size_t i = 0; i < size; ++i) {
      update.data()[i] = pattern_at(offset + i, seed);
    }
    CHECK(ctx.host_write_async(*object, offset, size, update.data()).get(10s) == size);
    // Read back through both read paths right after the future resolved.
    CHECK(ctx.host_read_async(*object, offset, size, readback.data()).get(10s) == size);
    CHECK(std::equal(update.data(), update.data() + size, readback.data()));
    std::fill_n(readback.data(), size, 0);
    CHECK(ctx.host_read(*object, offset, size, readback.data()) == size);
    CHECK(std::equal(update.data(), update.data() + size, readback.data()));
  }
}

TEST_CASE("uring writes invalidate the prefetching cache", "[io][uring][write][cache]")
{
  constexpr std::size_t chunk_bytes = 1 * MiB;
  constexpr std::size_t n_chunks    = 4;
  constexpr std::size_t file_bytes  = n_chunks * chunk_bytes;

  cucascade::memory::host_memory_space_config host;
  host.numa_id              = 0;
  host.memory_capacity      = 64 * MiB;
  host.block_size           = chunk_bytes;
  host.pool_size            = 64;
  host.initial_number_pools = 1;
  cucascade::memory::memory_reservation_manager manager(
    std::vector<cucascade::memory::memory_space_config>{host});

  staging_resource staging;  // 1 MiB blocks == chunk size
  temp_path path;
  auto ctx = std::make_shared<uring_ioctx>(1, staging.context());
  ctx->start();
  cucascade::io::cache::config cfg;
  cfg.mode                            = cucascade::io::cache::cache_mode::cucs;
  cfg.min_prefetching_budget_fraction = 0.5;
  ctx->initialize_cache(manager, cfg, nullptr);
  REQUIRE(ctx->cache() != nullptr);
  REQUIRE(ctx->cache()->chunk_size() == chunk_bytes);

  auto object = ctx->open_io_object_for_write(path.str());
  aligned_buffer initial(file_bytes);
  initial.fill(0, 0);
  CHECK(ctx->host_write_async(*object, 0, file_bytes, initial.data()).get(10s) == file_bytes);

  // Cache the whole file.
  {
    auto& cache = *ctx->cache();
    cucascade::io::byte_range const whole{0, static_cast<std::int64_t>(file_bytes)};
    auto handle = cucascade::io::cache::prefetching_cache_test_access::insert(
      cache, *object, std::span<cucascade::io::byte_range const>{&whole, 1});
    REQUIRE(handle);
    REQUIRE(cucascade::io::cache::prefetching_cache_test_access::prepare(cache, handle) ==
            cucascade::io::cache::prepare_result::prepared);
    std::atomic<int> prefetched{-1};
    std::ignore = cache.prefetch(handle, [&](bool ok) noexcept { prefetched = ok ? 1 : 0; });
    while (prefetched.load() < 0) {
      std::this_thread::sleep_for(1ms);
    }
    REQUIRE(prefetched.load() == 1);
    for (auto* chunk : *handle.chunks()) {
      REQUIRE(chunk->state.get_state() == cucascade::io::cache::chunk_state::cached);
    }

    std::vector<std::uint8_t> back(file_bytes);
    CHECK(ctx->host_read(*object, 0, file_bytes, back.data()) == file_bytes);
    CHECK(std::equal(back.begin(), back.end(), initial.data()));

    // Overwrite inside chunk 1 (async) and across chunks 2/3 (sync).
    std::vector<std::uint8_t> update(300 * KiB + 3);
    for (std::size_t i = 0; i < update.size(); ++i) {
      update[i] = pattern_at(chunk_bytes + 1000 + i, 77);
    }
    CHECK(
      ctx->host_write_async(*object, chunk_bytes + 1000, update.size(), update.data()).get(10s) ==
      update.size());
    std::vector<std::uint8_t> across(chunk_bytes);
    for (std::size_t i = 0; i < across.size(); ++i) {
      across[i] = pattern_at(2 * chunk_bytes + chunk_bytes / 2 + i, 78);
    }
    CHECK(
      ctx->host_write(*object, 2 * chunk_bytes + chunk_bytes / 2, across.size(), across.data()) ==
      across.size());

    std::vector<std::uint8_t> expected(initial.data(), initial.data() + file_bytes);
    std::copy(update.begin(), update.end(), expected.begin() + chunk_bytes + 1000);
    std::copy(across.begin(), across.end(), expected.begin() + 2 * chunk_bytes + chunk_bytes / 2);
    std::fill(back.begin(), back.end(), 0);
    CHECK(ctx->host_read(*object, 0, file_bytes, back.data()) == file_bytes);
    CHECK(first_mismatch(back, expected) == std::size_t(-1));
    CHECK((*handle.chunks())[0]->state.get_state() == cucascade::io::cache::chunk_state::cached);
  }
  ctx->shutdown_cache();
}

TEST_CASE("uring shutdown with writes in flight settles every request", "[io][uring][write]")
{
  staging_resource staging;
  temp_path path;
  auto ctx = std::make_unique<uring_ioctx>(2, staging.context());
  ctx->start();
  auto object = ctx->open_io_object_for_write(path.str());

  aligned_buffer source(4 * MiB);
  source.fill(0, 4);
  write_options sync;
  sync.durability = write_durability::data_sync;
  std::vector<std::pair<std::size_t, cucascade::exec::semi_future<std::size_t>>> futures;
  for (std::size_t i = 0; i < 64; ++i) {
    auto const offset = i * source.size();
    futures.emplace_back(
      offset,
      ctx->host_write_async(
        *object, offset, source.size(), source.data(), i % 4 == 0 ? sync : write_options{}));
  }
  auto flush = ctx->flush_async(*object);
  ctx->shutdown();

  std::size_t succeeded = 0;
  std::size_t cancelled = 0;
  std::vector<bool> completed(futures.size(), false);
  for (std::size_t i = 0; i < futures.size(); ++i) {
    std::size_t bytes = 0;
    auto const code   = system_error_of([&] { bytes = std::move(futures[i].second).get(30s); });
    if (!code) {
      CHECK(bytes == source.size());
      completed[i] = true;
      ++succeeded;
    } else {
      CHECK(code == std::errc::operation_canceled);
      ++cancelled;
    }
  }
  CHECK(system_error_of([&] { std::move(flush).get(30s); }) !=
        std::make_error_code(std::errc::interrupted));
  CHECK(succeeded + cancelled == futures.size());
  UNSCOPED_INFO("shutdown: " << succeeded << " writes completed, " << cancelled << " cancelled");

  // Completed writes landed intact.
  auto const bytes     = read_file(path.str());
  std::size_t verified = 0;
  for (std::size_t i = 0; i < futures.size(); ++i) {
    auto const offset = futures[i].first;
    if (completed[i] && bytes.size() >= offset + source.size() &&
        std::equal(source.data(), source.data() + source.size(), bytes.begin() + offset)) {
      ++verified;
    }
  }
  CHECK(verified == succeeded);

  // The context restarts and writes again.
  ctx->start();
  CHECK(ctx->host_write_async(*object, 0, 10, source.data()).get(10s) == 10);
  ctx.reset();
}

TEST_CASE("uring writes driven by external runners", "[io][uring][write]")
{
  staging_resource staging;
  temp_path path;
  uring_ioctx ctx(0, staging.context());
  std::stop_source stop;
  std::atomic<std::size_t> retired{0};
  std::vector<std::jthread> runners;
  for (int i = 0; i < 2; ++i) {
    runners.emplace_back([&] { retired.fetch_add(ctx.run(stop.get_token())); });
  }
  while (ctx.active_runners() < 2) {
    std::this_thread::sleep_for(1ms);
  }
  auto object = ctx.open_io_object_for_write(path.str());

  aligned_buffer source(32 * MiB);
  source.fill(0, 2);
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < 32; ++i) {
    futures.push_back(
      ctx.host_write_async(*object, i * MiB, MiB, source.data() + i * MiB, write_options{}));
  }
  for (auto& future : futures) {
    CHECK(std::move(future).get(20s) == MiB);
  }
  ctx.flush_async(*object).get(10s);
  ctx.commit_async(*object).get(10s);
  stop.request_stop();
  runners.clear();
  CHECK(retired.load() == 34);
  check_file(path.str(), std::vector<std::uint8_t>(source.data(), source.data() + source.size()));
}

//===----------------------------------------------------------------------===//
// Throughput (manual: build/.../cucascade_io_tests "[uring-write-bench]")
//===----------------------------------------------------------------------===//

TEST_CASE("uring write throughput", "[.][uring-write-bench]")
{
  auto const* dir_env = std::getenv("CUCASCADE_WRITE_BENCH_DIR");
  std::filesystem::path const dir =
    dir_env != nullptr ? std::filesystem::path{dir_env} : std::filesystem::temp_directory_path();
  auto const* gib_env       = std::getenv("CUCASCADE_WRITE_BENCH_GIB");
  std::size_t const total   = (gib_env != nullptr ? std::stoul(gib_env) : 4) << 30;
  constexpr std::size_t op  = 64 * MiB;
  auto const* runners_env   = std::getenv("CUCASCADE_WRITE_BENCH_RUNNERS");
  std::size_t const runners = runners_env != nullptr ? std::stoul(runners_env) : 2;

  staging_resource staging;
  aligned_buffer source(op);
  source.fill(0, 1);
  std::unique_ptr<device_area> device;
  if (have_gpu()) {
    device = std::make_unique<device_area>(op);
    if (!device->ok()) device.reset();
  }
  if (device != nullptr) {
    REQUIRE(cudaMemcpy(device->data, source.data(), op, cudaMemcpyHostToDevice) == cudaSuccess);
  }

  // dd-equivalent baseline: one thread, sequential 16 MiB pwrite (dd bs=16M [oflag=direct]).
  for (bool const odirect : {true, false}) {
    auto const file = dir / ("cucascade_write_bench_dd_" + std::to_string(::getpid()) + ".bin");
    int const fd =
      ::open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC | (odirect ? O_DIRECT : 0), 0644);
    REQUIRE(fd >= 0);
    constexpr std::size_t block = 16 * MiB;
    auto const t0               = std::chrono::steady_clock::now();
    for (std::size_t offset = 0; offset < total; offset += block) {
      REQUIRE(::pwrite(fd, source.data(), block, static_cast<off_t>(offset)) ==
              static_cast<ssize_t>(block));
    }
    auto const t1 = std::chrono::steady_clock::now();
    REQUIRE(::fdatasync(fd) == 0);
    auto const t2 = std::chrono::steady_clock::now();
    ::close(fd);
    WARN("baseline pwrite bs=16M" << (odirect ? " O_DIRECT" : " buffered") << ": "
                                  << static_cast<double>(total) /
                                       std::chrono::duration<double>(t1 - t0).count() / 1e9
                                  << " GB/s (incl. fdatasync: "
                                  << static_cast<double>(total) /
                                       std::chrono::duration<double>(t2 - t0).count() / 1e9
                                  << " GB/s)");
    std::error_code ec;
    std::filesystem::remove(file, ec);
  }

  for (bool const odirect : {true, false}) {
    for (bool const from_device : {false, true}) {
      if (from_device && device == nullptr) continue;
      auto const file = dir / ("cucascade_write_bench_" + std::to_string(::getpid()) + ".bin");
      uring_ioctx ctx(runners, staging.context(odirect));
      ctx.start();
      auto object   = ctx.open_io_object_for_write(file.string());
      auto const t0 = std::chrono::steady_clock::now();
      std::vector<cucascade::exec::semi_future<std::size_t>> futures;
      for (std::size_t offset = 0; offset < total; offset += op) {
        futures.push_back(
          from_device ? ctx.device_write_async(
                          *object, offset, op, device->ptr(), ::cuda::stream_ref{device->stream})
                      : ctx.host_write_async(*object, offset, op, source.data()));
      }
      for (auto& future : futures) {
        CHECK(std::move(future).get(600s) == op);
      }
      auto const t1 = std::chrono::steady_clock::now();
      ctx.flush_async(*object).get(600s);
      auto const t2   = std::chrono::steady_clock::now();
      auto const secs = std::chrono::duration<double>(t1 - t0).count();
      auto const sync = std::chrono::duration<double>(t2 - t0).count();
      WARN((from_device ? "device" : "host")
           << (odirect ? " odirect" : " buffered") << " runners=" << runners << ": "
           << static_cast<double>(total) / secs / 1e9
           << " GB/s (incl. fdatasync: " << static_cast<double>(total) / sync / 1e9 << " GB/s)");
      object.reset();
      std::error_code ec;
      std::filesystem::remove(file, ec);
    }
  }
}
