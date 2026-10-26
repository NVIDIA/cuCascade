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

// A write through an ioctx drops what its prefetching cache and metadata store
// hold for the written path; a file replaced by someone else is told apart by
// its object identity instead.  Generations are only ever built through the
// cudf datasource's fadvise, so this lives with the cudf cache tests.

#include "cache_test_fixture.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/config.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::datasource;
using cucascade::test::advise;
using cucascade::test::identity_cache_config;
using cucascade::test::identity_memory;
using cucascade::test::wait_for_cache;

[[nodiscard]] std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint8_t seed)
{
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::uint8_t>(i * 31U + seed);
  }
  return out;
}

[[nodiscard]] std::vector<std::uint8_t> read_all(datasource& ds)
{
  std::vector<std::uint8_t> bytes(ds.size());
  REQUIRE(ds.host_read(0, bytes.size(), bytes.data()) == bytes.size());
  return bytes;
}

/** A uring ioctx with an armed fs_cache, and a scratch file of known bytes. */
struct write_cache_fixture {
  write_cache_fixture() : memory(identity_memory())
  {
    cucascade::io::io_config config;
    config.uring_reactors_per_gpu = 1;
    context                       = cucascade::io::make_uring_ioctx_factory(*memory)(config);
    REQUIRE(context != nullptr);
    context->start();
    context->initialize_cache(*memory, identity_cache_config(), nullptr);
    REQUIRE(context->uses_fs_cache());
    bytes = 2 * context->cache()->chunk_size();
    path  = (std::filesystem::temp_directory_path() /
            ("cucascade_write_cache_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin"))
             .string();
    auto const old_bytes = pattern(bytes, 1);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(old_bytes.data()),
              static_cast<std::streamsize>(old_bytes.size()));
  }

  ~write_cache_fixture()
  {
    context->shutdown();
    context->shutdown_cache();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  write_cache_fixture(write_cache_fixture const&)            = delete;
  write_cache_fixture& operator=(write_cache_fixture const&) = delete;

  /// A datasource over a fresh read-only object, its whole range prefetched
  /// and read through the cache.
  std::unique_ptr<datasource> cached_reader()
  {
    auto reader = std::make_unique<datasource>(context, context->open_io_object(path));
    advise(*reader, bytes);
    std::ignore = read_all(*reader);
    return reader;
  }

  static inline std::atomic<int> next_id{0};
  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::shared_ptr<cucascade::io::ioctx> context;
  std::size_t bytes{0};
  std::string path;
};

}  // namespace

TEST_CASE("a write retires the cached generations of its path", "[cache][write][uring]")
{
  write_cache_fixture fixture;
  auto& cache = *fixture.context->cache();
  auto& ctx   = *fixture.context;

  auto reader = fixture.cached_reader();
  CHECK(read_all(*reader) == pattern(fixture.bytes, 1));
  REQUIRE(cache.generation_count(fixture.path) == 1);
  CHECK(cache.retired_generation_count() == 0);
  ctx.metadata_store().register_metadata(reader->get_io_object(),
                                         std::make_shared<cucascade::io::io_object_metadata>());
  REQUIRE(ctx.metadata_store().has_path(fixture.path));

  auto const new_bytes = pattern(fixture.bytes, 7);
  auto writer =
    ctx.open_io_object_for_write(fixture.path, {.mode = cucascade::io::write_mode::open_existing});

  SECTION("asynchronous write")
  {
    CHECK(ctx.host_write_async(*writer, 0, new_bytes.size(), new_bytes.data()).get() ==
          new_bytes.size());
  }
  SECTION("synchronous write")
  {
    CHECK(ctx.host_write(*writer, 0, new_bytes.size(), new_bytes.data()) == new_bytes.size());
  }

  // The open reader still owns its generation, now retired; its pinned bytes
  // stay valid for it.  The parsed metadata of the path is gone.
  CHECK(cache.generation_count(fixture.path) == 1);
  CHECK(cache.retired_generation_count() == 1);
  CHECK_FALSE(ctx.metadata_store().has_path(fixture.path));
  CHECK(read_all(*reader).size() == fixture.bytes);
  reader.reset();
  CHECK(wait_for_cache([&] { return cache.generation_count(fixture.path) == 0; }, 5s));
  CHECK(cache.retired_generation_count() == 0);

  // A new reader builds a fresh generation from the written bytes.
  auto fresh = fixture.cached_reader();
  CHECK(read_all(*fresh) == new_bytes);
  CHECK(cache.generation_count(fixture.path) == 1);
}

TEST_CASE("a pending invalidation drops a generation built while the write ran",
          "[cache][write][uring]")
{
  write_cache_fixture fixture;
  auto& cache = *fixture.context->cache();

  // What the write front end does: one pass before the write starts, and one
  // held until the write settles.  A prefetch that raced the write builds a
  // generation in between; firing the pending pass drops it.  The race is
  // simulated with the cache calls the front end makes, not with a real write
  // racing a real prefetch.
  cache.invalidate(fixture.path);
  auto pending = cache.invalidate_later(fixture.path);
  REQUIRE(pending);
  auto racing = fixture.cached_reader();
  REQUIRE(cache.generation_count(fixture.path) == 1);
  CHECK(cache.retired_generation_count() == 0);

  pending.fire();
  CHECK_FALSE(pending);
  CHECK(cache.retired_generation_count() == 1);

  // An invalidated generation is never handed out again, even for the same key
  // and while its old reader keeps it alive: a new reader gets a new one.
  auto again = fixture.cached_reader();
  CHECK(cache.generation_count(fixture.path) == 2);
  CHECK(cache.retired_generation_count() == 1);
  racing.reset();
  CHECK(wait_for_cache([&] { return cache.generation_count(fixture.path) == 1; }, 5s));
  CHECK(cache.retired_generation_count() == 0);
}

TEST_CASE("shutting the cache down waits for a pending invalidation", "[cache][write][uring]")
{
  write_cache_fixture fixture;
  auto pending = fixture.context->cache()->invalidate_later(fixture.path);
  REQUIRE(pending);

  // A pending invalidation stands for a write still in flight: the cache must
  // outlive it, so its destructor waits until it fires.
  std::atomic<bool> closed{false};
  std::thread closer([&] {
    fixture.context->shutdown_cache();
    closed.store(true);
  });
  std::this_thread::sleep_for(200ms);
  CHECK_FALSE(closed.load());
  pending.fire();
  closer.join();
  CHECK(closed.load());
  CHECK(fixture.context->cache() == nullptr);
}

TEST_CASE("local opens share a generation until the file is replaced",
          "[cache][uring][object_identity]")
{
  write_cache_fixture fixture;
  auto& cache = *fixture.context->cache();
  auto first  = fixture.cached_reader();
  REQUIRE(cache.generation_count(fixture.path) == 1);
  auto const resident = cache.claimed_bytes();

  // A second open of the unchanged file has the same stat identity: it reuses
  // the first open's generation and its resident chunks.
  auto second = fixture.cached_reader();
  CHECK(cache.generation_count(fixture.path) == 1);
  CHECK(cache.file_entry_count_for_testing() == 1);
  CHECK(cache.claimed_bytes() == resident);
  CHECK(read_all(*second) == pattern(fixture.bytes, 1));

  // The file is replaced atomically by someone else: nothing written through
  // this ioctx invalidates the path, but the next open has a new identity, so
  // it builds a new generation and retires the old one.
  auto const new_bytes = pattern(fixture.bytes, 9);
  auto const staged    = fixture.path + ".next";
  {
    std::ofstream out(staged, std::ios::binary);
    out.write(reinterpret_cast<char const*>(new_bytes.data()),
              static_cast<std::streamsize>(new_bytes.size()));
  }
  std::filesystem::rename(staged, fixture.path);
  auto third = fixture.cached_reader();
  CHECK(read_all(*third) == new_bytes);
  CHECK(cache.generation_count(fixture.path) == 2);
  CHECK(cache.retired_generation_count() == 1);
  CHECK(cache.file_entry_count_for_testing() == 1);

  // The earlier opens keep the bytes of the file they opened.
  CHECK(read_all(*first) == pattern(fixture.bytes, 1));
  first.reset();
  second.reset();
  CHECK(wait_for_cache([&] { return cache.generation_count(fixture.path) == 1; }, 5s));
  CHECK(cache.retired_generation_count() == 0);
}

TEST_CASE("local reopens reuse parsed metadata until the file is replaced",
          "[cache][uring][object_identity][metadata_store]")
{
  write_cache_fixture fixture;
  auto& store       = fixture.context->metadata_store();
  auto const parsed = std::make_shared<cucascade::io::io_object_metadata>();

  datasource first(fixture.context, fixture.context->open_io_object(fixture.path));
  REQUIRE(first.store_metadata(parsed));
  CHECK(store.has_path(fixture.path));
  // A separate open of the unchanged file has the same stat identity.
  datasource const reopened(fixture.context, fixture.context->open_io_object(fixture.path));
  CHECK(reopened.metadata() == parsed);

  // An atomic replacement by someone else: the next open has a new identity, so
  // it neither sees the old metadata nor lets the old open publish more.
  auto const staged = fixture.path + ".next";
  {
    auto const new_bytes = pattern(fixture.bytes, 9);
    std::ofstream out(staged, std::ios::binary);
    out.write(reinterpret_cast<char const*>(new_bytes.data()),
              static_cast<std::streamsize>(new_bytes.size()));
  }
  std::filesystem::rename(staged, fixture.path);
  datasource const replaced(fixture.context, fixture.context->open_io_object(fixture.path));
  CHECK_FALSE(replaced.metadata());
  CHECK_FALSE(store.has_path(fixture.path));
  REQUIRE(first.store_metadata(std::make_shared<cucascade::io::io_object_metadata>()));
  CHECK_FALSE(replaced.metadata());
  CHECK_FALSE(first.metadata());
}
