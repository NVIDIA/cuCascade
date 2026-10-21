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

#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct fake_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
  std::size_t range_batch_slices{0};
};

class fake_object final : public cucascade::io::io_object {
 public:
  fake_object(std::string path, std::size_t size) : _path(std::move(path)), _size(size) {}

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  std::size_t _size;
};

class fake_reactor {
 public:
  using io_object_type                  = fake_object;
  using reactor_config_type             = fake_config;
  static constexpr bool prefers_bulk_io = false;

  explicit fake_reactor(std::size_t backlog = 0) : backlog(backlog) {}

  [[nodiscard]] fake_config const& get_config() const noexcept { return config; }

  void enqueue(std::unique_ptr<cucascade::io::grouped_io_request> request) noexcept
  {
    requests.push_back(std::move(request));
  }

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return backlog; }

  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  std::size_t host_read(fake_object const&, std::size_t, std::size_t size, std::uint8_t*) const
  {
    return size;
  }

  void start() {}
  void shutdown() {}
  void interrupt() {}

  [[nodiscard]] static std::unique_ptr<fake_object> create_io_object(std::string path)
  {
    return std::make_unique<fake_object>(std::move(path), 4096);
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

  fake_config config;
  std::size_t backlog;
  std::vector<std::unique_ptr<cucascade::io::grouped_io_request>> requests;
};

static_assert(cucascade::io::io_reactor_c<fake_reactor>);

class fake_context final : public cucascade::io::templated_ioctx<fake_reactor> {
 public:
  using templated_ioctx::templated_ioctx;

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::uring;
  }
};

void complete_request(cucascade::io::grouped_io_request& request)
{
  while (!request.empty()) {
    static_cast<void>(request.take_front());
    request.coordinator->on_complete();
  }
}

}  // namespace

TEST_CASE("mixed dispatch selects two least-busy reactors and shares one coordinator",
          "[io][templated_ioctx]")
{
  std::vector<std::unique_ptr<fake_reactor>> reactors;
  std::vector<fake_reactor*> raw;
  for (auto const backlog : {1000U, 10U, 20U, 500U}) {
    reactors.push_back(std::make_unique<fake_reactor>(backlog));
    raw.push_back(reactors.back().get());
  }
  fake_context context{std::move(reactors)};
  auto object = std::make_shared<fake_object>("fake", 4096);

  std::uint8_t byte{};
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(cucascade::io::range{0, 60}, cucascade::io::host_buffer{&byte});
  slices.emplace_back(cucascade::io::range{100, 40}, cucascade::io::host_buffer{&byte});
  slices.emplace_back(cucascade::io::range{200, 20}, cucascade::io::host_buffer{&byte});

  auto future = context.mixed_readv_async_io(*object, std::move(slices));

  CHECK(raw[0]->requests.empty());
  REQUIRE(raw[1]->requests.size() == 1);
  REQUIRE(raw[2]->requests.size() == 1);
  CHECK(raw[3]->requests.empty());

  auto& first  = *raw[1]->requests.front();
  auto& second = *raw[2]->requests.front();
  CHECK(first.coordinator == second.coordinator);
  CHECK(first.remaining_bytes() == 60);
  CHECK(second.remaining_bytes() == 60);
  CHECK_FALSE(future.is_ready());

  complete_request(first);
  CHECK_FALSE(future.is_ready());
  complete_request(second);

  CHECK(future.is_ready());
  CHECK(std::move(future).get() == 120);
}

TEST_CASE("mixed dispatch with no reactors fails and releases prepared cache slices",
          "[io][templated_ioctx]")
{
  fake_context context{std::vector<std::unique_ptr<fake_reactor>>{}};
  auto object = std::make_shared<fake_object>("fake", 4096);
  std::atomic<bool> callback_success{true};

  auto completion = std::make_shared<cucascade::io::prepared_io_completion>(
    [&callback_success](std::span<cucascade::io::cache::cached_chunk* const>,
                        bool success) noexcept {
      callback_success.store(success, std::memory_order_release);
    });

  std::uint8_t byte{};
  cucascade::io::prepared_io_slice slice{cucascade::io::range{0, 64},
                                         cucascade::io::host_buffer{&byte}};
  slice.on_complete = completion;
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(std::move(slice));

  auto future = context.mixed_readv_async_io(*object, std::move(slices));

  CHECK_FALSE(callback_success.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(), "mixed_readv_async_io: no available reactors");
}

TEST_CASE("mixed dispatch reports failure for a dropped fragmented slice past EOF",
          "[io][templated_ioctx]")
{
  std::vector<std::unique_ptr<fake_reactor>> reactors;
  reactors.push_back(std::make_unique<fake_reactor>());
  fake_context context{std::move(reactors)};
  auto object = std::make_shared<fake_object>("fake", 4096);

  std::atomic<bool> callback_success{true};
  std::atomic<int> callback_count{0};
  auto completion = std::make_shared<cucascade::io::prepared_io_completion>(
    [&](std::span<cucascade::io::cache::cached_chunk* const>, bool success) noexcept {
      callback_success.store(success, std::memory_order_release);
      callback_count.fetch_add(1, std::memory_order_acq_rel);
    });

  cucascade::io::cache::cached_chunk chunk{object->size()};
  cucascade::io::prepared_io_slice slice{
    cucascade::io::range{object->size(), 64},
    cucascade::io::host_buffer{std::vector<cucascade::io::cache::cached_chunk*>{&chunk}}};
  slice.on_complete = completion;
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(std::move(slice));

  auto future = context.mixed_readv_async_io(*object, std::move(slices));

  CHECK_FALSE(callback_success.load(std::memory_order_acquire));
  CHECK(callback_count.load(std::memory_order_acquire) == 1);
  REQUIRE(future.is_ready());
  CHECK(std::move(future).get() == 0);
}

namespace {

/// A pool of fake reactors with the given backlogs; the pool reads
/// range_batch_slices from the first reactor's config, as uring_ioctx does.
struct reactor_pool {
  explicit reactor_pool(std::initializer_list<std::size_t> backlogs,
                        std::size_t range_batch_slices = 0)
  {
    std::vector<std::unique_ptr<fake_reactor>> reactors;
    for (auto const backlog : backlogs) {
      reactors.push_back(std::make_unique<fake_reactor>(backlog));
      reactors.back()->config.range_batch_slices = range_batch_slices;
      raw.push_back(reactors.back().get());
    }
    context = std::make_unique<fake_context>(std::move(reactors));
  }

  /// Dispatch one request of @p n_slices slices stamped @p priority; return how
  /// many requests each reactor received from it.
  std::vector<std::size_t> dispatch(
    std::size_t n_slices                = 3,
    cucascade::io::io_priority priority = cucascade::io::io_priority::automatic)
  {
    std::vector<std::size_t> before;
    for (auto* reactor : raw) {
      before.push_back(reactor->requests.size());
    }
    std::vector<cucascade::io::prepared_io_slice> slices;
    for (std::size_t i = 0; i < n_slices; ++i) {
      slices.emplace_back(cucascade::io::range{i * 100, 40}, cucascade::io::host_buffer{&byte});
      slices.back().priority = priority;
    }
    static_cast<void>(context->mixed_readv_async_io(*object, std::move(slices)));
    std::vector<std::size_t> received;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      received.push_back(raw[i]->requests.size() - before[i]);
    }
    return received;
  }

  std::vector<fake_reactor*> raw;
  std::unique_ptr<fake_context> context;
  std::shared_ptr<fake_object> object = std::make_shared<fake_object>("fake", 4096);
  std::uint8_t byte{};
};

}  // namespace

TEST_CASE("next_reactor sends a read to the two least-backlogged reactors",
          "[io][templated_ioctx][next_reactor]")
{
  reactor_pool pool{{1000, 10, 20, 500}};
  for (int round = 0; round < 4; ++round) {
    CHECK(pool.dispatch() == std::vector<std::size_t>{0, 1, 1, 0});
    // A single-slice request goes to the least busy reactor.
    CHECK(pool.dispatch(1) == std::vector<std::size_t>{0, 1, 0, 0});
  }
}

namespace {

constexpr std::size_t MiB = 1UL << 20;

/// Four reactors: 0 and 1 homed on GPU 0 (NUMA 0), 2 and 3 on GPU 1 (NUMA 1).
struct placed_pool {
  explicit placed_pool(std::initializer_list<std::size_t> backlogs,
                       std::initializer_list<int> numa_nodes = {0, 0, 1, 1})
  {
    cucascade::io::reactor_layout layout;
    std::vector<std::unique_ptr<fake_reactor>> reactors;
    auto numa = numa_nodes.begin();
    for (auto const backlog : backlogs) {
      reactors.push_back(std::make_unique<fake_reactor>(backlog));
      raw.push_back(reactors.back().get());
      cucascade::io::reactor_placement placement;
      placement.gpu_id    = static_cast<int>(layout.placements.size() / 2);
      placement.numa_node = *numa++;
      layout.placements.push_back(placement);
    }
    context = std::make_unique<fake_context>(std::move(reactors), layout);
  }

  /// Indices of the reactors next_reactor picks for a request on @p device_id.
  std::vector<std::size_t> pick(int device_id)
  {
    std::vector<std::size_t> picked;
    for (auto* reactor :
         context->next_reactor(*object, 3, fake_context::io_op_type::device_async, device_id)) {
      picked.push_back(
        static_cast<std::size_t>(std::find(raw.begin(), raw.end(), reactor) - raw.begin()));
    }
    return picked;
  }

  std::vector<fake_reactor*> raw;
  std::unique_ptr<fake_context> context;
  std::shared_ptr<fake_object> object = std::make_shared<fake_object>("fake", 4096);
};

}  // namespace

TEST_CASE("next_reactor keeps a device request on its GPU's NUMA node",
          "[io][templated_ioctx][next_reactor][numa]")
{
  placed_pool pool{{1000, 10, 0, 5}};
  // The idle remote reactors do not draw a request away from a local node that keeps up.
  CHECK(pool.pick(0) == std::vector<std::size_t>{1, 0});
  CHECK(pool.pick(1) == std::vector<std::size_t>{2, 3});
}

TEST_CASE("next_reactor uses the whole pool for host-only requests and unknown GPUs",
          "[io][templated_ioctx][next_reactor][numa]")
{
  placed_pool pool{{1000, 10, 20, 500}};
  CHECK(pool.pick(-1) == std::vector<std::size_t>{1, 2});
  CHECK(pool.pick(7) == std::vector<std::size_t>{1, 2});
}

TEST_CASE("next_reactor spills a device request once its NUMA node is much busier",
          "[io][templated_ioctx][next_reactor][numa]")
{
  SECTION("over the floor and over the ratio: spill to the least backlogged of the pool")
  {
    placed_pool pool{{200 * MiB, 100 * MiB, 1 * MiB, 0}};
    CHECK(pool.pick(0) == std::vector<std::size_t>{3, 2});
  }
  SECTION("over the floor but within the ratio: stay local")
  {
    placed_pool pool{{100 * MiB, 100 * MiB, 60 * MiB, 60 * MiB}};
    CHECK(pool.pick(0) == std::vector<std::size_t>{0, 1});
  }
  SECTION("under the floor: stay local however idle the rest is")
  {
    placed_pool pool{{64 * MiB, 64 * MiB, 0, 0}};
    CHECK(pool.pick(0) == std::vector<std::size_t>{0, 1});
  }
}

TEST_CASE("next_reactor falls back to the GPU's own reactors when its NUMA node is unknown",
          "[io][templated_ioctx][next_reactor][numa]")
{
  placed_pool pool{{0, 0, 0, 0}, {-1, -1, -1, -1}};
  for (int round = 0; round < 4; ++round) {
    auto picked = pool.pick(1);
    std::ranges::sort(picked);
    CHECK(picked == std::vector<std::size_t>{2, 3});
  }
}

TEST_CASE("a device read is routed by the GPU of its first device slice",
          "[io][templated_ioctx][next_reactor][numa]")
{
  placed_pool pool{{0, 0, 0, 0}};
  std::uint8_t byte{};
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(cucascade::io::range{0, 40}, cucascade::io::host_buffer{&byte});
  slices.emplace_back(
    cucascade::io::range{100, 40},
    cucascade::io::device_buffer{&byte, ::cuda::stream_ref{cudaStream_t{nullptr}}, 1});
  static_cast<void>(pool.context->mixed_readv_async_io(*pool.object, std::move(slices)));

  CHECK(pool.raw[0]->requests.empty());
  CHECK(pool.raw[1]->requests.empty());
  CHECK(pool.raw[2]->requests.size() + pool.raw[3]->requests.size() == 2);
  for (auto* reactor : pool.raw) {
    for (auto& request : reactor->requests) {
      complete_request(*request);
    }
  }
}

TEST_CASE("a layout must place every reactor", "[io][templated_ioctx][numa]")
{
  std::vector<std::unique_ptr<fake_reactor>> reactors;
  reactors.push_back(std::make_unique<fake_reactor>());
  reactors.push_back(std::make_unique<fake_reactor>());
  auto const layout = cucascade::io::reactor_layout::unplaced(1);
  CHECK_THROWS_AS(fake_context(std::move(reactors), layout), std::invalid_argument);
}

TEST_CASE("automatic priority always resolves high; only an explicit low is low",
          "[io][templated_ioctx][priority]")
{
  using cucascade::io::io_priority;
  reactor_pool pool{{0}};
  auto& requests = pool.raw[0]->requests;
  std::uint8_t buffer[64]{};
  std::array<cucascade::io::slice, 2> const ranges{cucascade::io::slice{0, 16, buffer},
                                                   cucascade::io::slice{100, 16, buffer + 16}};

  static_cast<void>(pool.context->host_read_async_io(*pool.object, 0, 16, buffer));
  CHECK(requests.back()->priority == io_priority::high);

  // A multi-range host read is demand work too (a native decoder blocks on it).
  static_cast<void>(pool.context->host_readv_async_io(*pool.object, ranges));
  CHECK(requests.back()->priority == io_priority::high);

  static_cast<void>(pool.context->host_readv_async_io(*pool.object, ranges, io_priority::low));
  CHECK(requests.back()->priority == io_priority::low);

  static_cast<void>(pool.context->host_readv_async_io(*pool.object, ranges, io_priority::high));
  CHECK(requests.back()->priority == io_priority::high);

  static_cast<void>(
    pool.context->host_read_async_io(*pool.object, 0, 16, buffer, io_priority::low));
  CHECK(requests.back()->priority == io_priority::low);

  // Slices handed straight to mixed_readv_async_io resolve the same way,
  // whatever the call shape.
  static_cast<void>(pool.dispatch(1));
  CHECK(requests.back()->priority == io_priority::high);
  static_cast<void>(pool.dispatch(3));
  CHECK(requests.back()->priority == io_priority::high);
  // A cache prefetch stamps its slices low explicitly, and that wins.
  static_cast<void>(pool.dispatch(1, io_priority::low));
  CHECK(requests.back()->priority == io_priority::low);
  static_cast<void>(pool.dispatch(3, io_priority::low));
  CHECK(requests.back()->priority == io_priority::low);
}

TEST_CASE("a host-only range read is split into range_batch_slices-sized requests",
          "[io][templated_ioctx][range_batch_slices]")
{
  auto slice_counts = [](fake_reactor const& reactor) {
    std::vector<std::size_t> counts;
    for (auto const& request : reactor.requests) {
      counts.push_back(request->remaining_slices());
    }
    return counts;
  };

  SECTION("batch 3: each reactor's 5-slice partition becomes a 3 + 2 pair")
  {
    reactor_pool pool{{0, 0}, 3};
    CHECK(pool.context->range_batch_slices() == 3);
    CHECK(pool.dispatch(10) == std::vector<std::size_t>{2, 2});
    CHECK(slice_counts(*pool.raw[0]) == std::vector<std::size_t>{3, 2});
    CHECK(slice_counts(*pool.raw[1]) == std::vector<std::size_t>{3, 2});
  }

  SECTION("batch 0 keeps one request per selected reactor")
  {
    reactor_pool pool{{0, 0}};
    CHECK(pool.context->range_batch_slices() == 0);
    CHECK(pool.dispatch(10) == std::vector<std::size_t>{1, 1});
  }
}

// ===========================================================================
// Writes
// ===========================================================================

namespace {

class fake_writable_object final : public cucascade::io::io_object {
 public:
  fake_writable_object(std::string path, std::size_t size, bool writable)
    : _path(std::move(path)), _size(size), _writable(writable)
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override
  {
    return _size.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool writable() const noexcept override { return _writable; }
  void note_written(std::size_t end) const noexcept override
  {
    if (_writable) cucascade::io::atomic_store_max(_size, end);
  }

 private:
  std::string _path;
  mutable std::atomic<std::size_t> _size;
  bool _writable;
};

struct fake_write_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
  std::size_t range_batch_slices{0};
  std::size_t write_split_bytes{0};
};

/** A reactor modelling io_write_reactor_c that only records what it is given. */
class fake_write_reactor {
 public:
  using io_object_type      = fake_writable_object;
  using reactor_config_type = fake_write_config;

  explicit fake_write_reactor(fake_write_config config = {}, std::size_t backlog = 0)
    : config(config), backlog(backlog)
  {
  }

  [[nodiscard]] fake_write_config const& get_config() const noexcept { return config; }

  void enqueue(std::unique_ptr<cucascade::io::grouped_io_request> request) noexcept
  {
    requests.push_back(std::move(request));
  }

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return backlog; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  std::size_t host_read(fake_writable_object const&, std::size_t, std::size_t size, std::uint8_t*)
  {
    return size;
  }

  std::size_t host_write(fake_writable_object const& object,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const*)
  {
    object.note_written(offset + size);
    return size;
  }

  static void flush(fake_writable_object const&) { ++flushes; }

  void start() {}
  void shutdown() {}
  void interrupt() {}

  [[nodiscard]] static std::unique_ptr<fake_writable_object> create_io_object(std::string path)
  {
    return std::make_unique<fake_writable_object>(std::move(path), 4096, false);
  }

  [[nodiscard]] static std::unique_ptr<fake_writable_object> create_io_object_for_write(
    std::string path, cucascade::io::write_open_options)
  {
    return std::make_unique<fake_writable_object>(std::move(path), 0, true);
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

  static inline std::atomic<int> flushes{0};

  fake_write_config config;
  std::size_t backlog;
  std::vector<std::unique_ptr<cucascade::io::grouped_io_request>> requests;
};

static_assert(cucascade::io::io_reactor_c<fake_write_reactor>);
static_assert(cucascade::io::io_write_reactor_c<fake_write_reactor>);
static_assert(!cucascade::io::io_write_reactor_c<fake_reactor>);
static_assert(cucascade::io::reactor_write_traits<fake_write_reactor>::supports_device_write);
static_assert(!cucascade::io::reactor_write_traits<fake_reactor>::supports_device_write);

class fake_write_context final : public cucascade::io::templated_ioctx<fake_write_reactor> {
 public:
  using templated_ioctx::_reactors;
  using templated_ioctx::templated_ioctx;

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::uring;
  }
};

[[nodiscard]] std::unique_ptr<fake_write_context> make_write_context(
  std::vector<std::size_t> const& backlogs, fake_write_config config = {})
{
  std::vector<std::unique_ptr<fake_write_reactor>> reactors;
  for (auto const backlog : backlogs) {
    reactors.push_back(std::make_unique<fake_write_reactor>(config, backlog));
  }
  return std::make_unique<fake_write_context>(std::move(reactors));
}

template <class Future>
[[nodiscard]] std::errc error_code_of(Future&& future)
{
  try {
    std::forward<Future>(future).get();
  } catch (std::system_error const& error) {
    return static_cast<std::errc>(error.code().value());
  } catch (...) {
  }
  return std::errc{};
}

}  // namespace

TEST_CASE("a reactor without writes reports not_supported for every write API",
          "[io][templated_ioctx][write]")
{
  std::vector<std::unique_ptr<fake_reactor>> reactors;
  reactors.push_back(std::make_unique<fake_reactor>());
  fake_context context{std::move(reactors)};
  CHECK_FALSE(context.supports_write());
  CHECK_FALSE(context.supports_device_write());

  try {
    std::ignore = context.open_io_object_for_write("file:/tmp/x");
    FAIL("opened for writing without write support");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::not_supported);
  }

  fake_object object{"fake", 4096};
  std::uint8_t byte{};
  CHECK_THROWS_AS(context.host_write(object, 0, 1, &byte), std::system_error);
  CHECK(error_code_of(context.host_write_async(object, 0, 1, &byte)) == std::errc::not_supported);
  CHECK(error_code_of(context.flush_async(object)) == std::errc::not_supported);
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(cucascade::io::make_host_write_slice(cucascade::io::range{0, 1}, &byte));
  CHECK(error_code_of(context.mixed_writev_async_io(object, std::move(slices))) ==
        std::errc::not_supported);
}

TEST_CASE("write dispatch spreads unclamped write slices over two reactors",
          "[io][templated_ioctx][write]")
{
  auto context = make_write_context({1000, 10, 20, 500});
  CHECK(context->supports_write());
  CHECK(context->supports_device_write());
  auto object = context->open_io_object_for_write("fake");
  REQUIRE(object->writable());

  std::array<std::uint8_t, 64> host{};
  // A device pointer the fake reactor never dereferences.
  auto* device = reinterpret_cast<std::uint8_t const*>(std::uintptr_t{0x10000});
  std::vector<cucascade::io::write_slice> host_slices{{10000, 64, host.data()}};
  std::vector<cucascade::io::write_slice> device_slices{{20000, 300, device}, {50, 0, nullptr}};
  auto future = context->mixed_writev_async(
    *object, host_slices, device_slices, ::cuda::stream_ref{cudaStream_t{nullptr}});

  auto& raw = context->_reactors;
  CHECK(raw[0]->requests.empty());
  REQUIRE(raw[1]->requests.size() == 1);
  REQUIRE(raw[2]->requests.size() == 1);
  CHECK(raw[3]->requests.empty());

  for (auto* reactor : {raw[1].get(), raw[2].get()}) {
    auto& request = *reactor->requests.front();
    CHECK(request.priority == cucascade::io::io_priority::high);
    for (auto const& slice : request.slices) {
      CHECK(slice.is_write());
      CHECK(slice.on_complete == nullptr);
      if (slice.has_device_request()) {
        CHECK(slice.rng.offset == 20000);
        CHECK(slice.rng.size == 300);
        CHECK(slice.d_buffer.device_id >= 0);
      } else {
        // Past the object's size, and still not clamped.
        CHECK(slice.rng.offset == 10000);
        CHECK(slice.rng.size == 64);
      }
    }
  }
  CHECK_FALSE(future.is_ready());
  complete_request(*raw[1]->requests.front());
  complete_request(*raw[2]->requests.front());
  CHECK(std::move(future).get() == 364);
}

TEST_CASE("write_split_bytes cuts large write slices at aligned boundaries",
          "[io][templated_ioctx][write]")
{
  auto const granule = cucascade::io::write_granule();
  fake_write_config config;
  config.write_split_bytes = 2 * granule + 100;  // rounded down to 2 granules
  auto context             = make_write_context({0, 0}, config);
  CHECK(context->write_split_bytes() == 2 * granule + 100);
  auto object = context->open_io_object_for_write("fake");

  std::vector<std::uint8_t> source(5 * granule);
  auto future = context->host_write_async(*object, 100, 5 * granule, source.data());

  std::vector<cucascade::io::prepared_io_slice> pieces;
  for (auto& reactor : context->_reactors) {
    for (auto& request : reactor->requests) {
      for (auto const& slice : request->slices) {
        pieces.push_back(slice);
      }
    }
  }
  std::sort(pieces.begin(), pieces.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.offset() < rhs.offset();
  });
  REQUIRE(pieces.size() == 3);
  CHECK(pieces[0].rng.offset == 100);
  CHECK(pieces[0].rng.end() == 2 * granule);
  CHECK(pieces[1].rng.offset == 2 * granule);
  CHECK(pieces[1].rng.size == 2 * granule);
  CHECK(pieces[2].rng.offset == 4 * granule);
  CHECK(pieces[2].rng.end() == 5 * granule + 100);
  for (auto const& piece : pieces) {
    CHECK(std::get<std::uint8_t*>(piece.h_buffer.buffer) == source.data() + (piece.offset() - 100));
  }

  for (auto& reactor : context->_reactors) {
    for (auto& request : reactor->requests) {
      complete_request(*request);
    }
  }
  CHECK(std::move(future).get() == 5 * granule);
}

TEST_CASE("flush_async queues one flush slice on one reactor", "[io][templated_ioctx][write]")
{
  auto context = make_write_context({50, 5, 7});
  auto object  = context->open_io_object_for_write("fake");
  auto future  = context->flush_async(*object, cucascade::io::io_priority::low);

  std::size_t requests = 0;
  for (auto& reactor : context->_reactors) {
    for (auto& request : reactor->requests) {
      ++requests;
      REQUIRE(request->slices.size() == 1);
      CHECK(request->slices.front().is_flush());
      CHECK(request->slices.front().rng.empty());
      CHECK(request->priority == cucascade::io::io_priority::low);
      CHECK(request->coordinator->bytes_requested() == 0);
      complete_request(*request);
    }
  }
  CHECK(requests == 1);
  CHECK_NOTHROW(std::move(future).get());

  auto const before = fake_write_reactor::flushes.load();
  context->flush(*object);
  CHECK(fake_write_reactor::flushes.load() == before + 1);
}

TEST_CASE("synchronous writes run on the caller and grow the size", "[io][templated_ioctx][write]")
{
  auto context = make_write_context({0});
  auto object  = context->open_io_object_for_write("fake");
  std::array<std::uint8_t, 16> bytes{};
  CHECK(context->host_write(*object, 1000, bytes.size(), bytes.data()) == bytes.size());
  CHECK(object->size() == 1016);
  CHECK(context->_reactors.front()->requests.empty());
}

TEST_CASE("writes to a read-only object are bad_file_descriptor", "[io][templated_ioctx][write]")
{
  auto context = make_write_context({0});
  auto object  = context->open_io_object("fake");
  REQUIRE_FALSE(object->writable());
  std::array<std::uint8_t, 4> bytes{};
  CHECK(error_code_of(context->host_write_async(*object, 0, bytes.size(), bytes.data())) ==
        std::errc::bad_file_descriptor);
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.push_back(cucascade::io::make_host_write_slice(cucascade::io::range{0, 4}, bytes.data()));
  CHECK(error_code_of(context->mixed_writev_async_io(*object, std::move(slices))) ==
        std::errc::bad_file_descriptor);
  CHECK(error_code_of(context->flush_async(*object)) == std::errc::bad_file_descriptor);
  CHECK(context->_reactors.front()->requests.empty());
}

TEST_CASE("validate_write_slices enforces the write slice invariants",
          "[io][templated_ioctx][write]")
{
  using cucascade::io::make_device_write_slice;
  using cucascade::io::make_host_write_slice;
  using cucascade::io::prepared_io_slice;
  using cucascade::io::range;
  using cucascade::io::validate_write_slices;

  std::array<std::uint8_t, 64> host{};
  auto* device      = reinterpret_cast<std::uint8_t const*>(std::uintptr_t{0x10000});
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};

  std::vector<prepared_io_slice> good{make_host_write_slice(range{100, 10}, host.data()),
                                      make_device_write_slice(range{0, 100}, device, stream)};
  CHECK(validate_write_slices(good) == 110);

  auto rejects = [](std::vector<prepared_io_slice> const& slices) {
    CHECK_THROWS_AS(validate_write_slices(slices), std::invalid_argument);
  };
  rejects({make_host_write_slice(range{0, 10}, host.data()),
           make_host_write_slice(range{9, 10}, host.data())});
  rejects({make_host_write_slice(range{0, 0}, host.data())});
  rejects({prepared_io_slice{range{0, 10}, cucascade::io::host_buffer{host.data()}}});
  rejects({cucascade::io::make_flush_slice()});
  auto both     = make_host_write_slice(range{0, 10}, host.data());
  both.d_buffer = cucascade::io::device_buffer{host.data(), stream};
  rejects({both});
  rejects(
    {make_host_write_slice(range{std::numeric_limits<std::size_t>::max() - 4, 10}, host.data())});
}

TEST_CASE("io_op_request raises the object size for writes only", "[io][templated_ioctx][write]")
{
  auto object      = std::make_shared<fake_writable_object>("fake", 100, true);
  auto coordinator = std::make_shared<cucascade::io::grouped_coordinator>(0, 2);

  cucascade::io::io_op_request read;
  read.obj         = object;
  read.coordinator = coordinator;
  read.io_rng      = cucascade::io::range{0, 5000};
  read.finish_success();
  CHECK(object->size() == 100);

  cucascade::io::io_op_request write;
  write.obj         = object;
  write.coordinator = coordinator;
  write.kind        = cucascade::io::io_op_kind::write;
  write.io_rng      = cucascade::io::range{4000, 1000};
  write.finish_success();
  CHECK(object->size() == 5000);
}
