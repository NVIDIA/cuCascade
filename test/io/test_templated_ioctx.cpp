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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct fake_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
  std::size_t prefetch_reactors{0};
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
/// prefetch_reactors from the first reactor's config, as uring_ioctx does.
struct reactor_pool {
  reactor_pool(std::initializer_list<std::size_t> backlogs, std::size_t prefetch_reactors)
  {
    std::vector<std::unique_ptr<fake_reactor>> reactors;
    for (auto const backlog : backlogs) {
      reactors.push_back(std::make_unique<fake_reactor>(backlog));
      reactors.back()->config.prefetch_reactors = prefetch_reactors;
      raw.push_back(reactors.back().get());
    }
    context = std::make_unique<fake_context>(std::move(reactors));
  }

  /// Dispatch one request of @p n_slices slices of class @p cls; return how many
  /// requests each reactor received from it.
  std::vector<std::size_t> dispatch(cucascade::io::io_class cls, std::size_t n_slices = 3)
  {
    std::vector<std::size_t> before;
    for (auto* reactor : raw) {
      before.push_back(reactor->requests.size());
    }
    std::vector<cucascade::io::prepared_io_slice> slices;
    for (std::size_t i = 0; i < n_slices; ++i) {
      slices.emplace_back(cucascade::io::range{i * 100, 40}, cucascade::io::host_buffer{&byte});
      slices.back().cls = cls;
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

TEST_CASE("prefetch_reactors routes prefetch reads to the last K reactors and demand to the rest",
          "[io][templated_ioctx][prefetch_reactors]")
{
  using cucascade::io::io_class;

  SECTION("K = 1: prefetch goes to the last reactor even when it is the busiest")
  {
    reactor_pool pool{{1000, 10, 20, 5000}, 1};
    for (int round = 0; round < 8; ++round) {
      CHECK(pool.dispatch(io_class::prefetch) == std::vector<std::size_t>{0, 0, 0, 1});
      // Demand ranks only among reactors 0-2: the two least busy of those.
      CHECK(pool.dispatch(io_class::demand) == std::vector<std::size_t>{0, 1, 1, 0});
    }
    // A request carries the class of its slices.
    CHECK(pool.raw[3]->requests.front()->cls == io_class::prefetch);
    CHECK(pool.raw[1]->requests.front()->cls == io_class::demand);
  }

  SECTION("K = 2: prefetch fans out over the last two, demand over the first two")
  {
    reactor_pool pool{{30, 20, 1000, 2000}, 2};
    for (int round = 0; round < 8; ++round) {
      CHECK(pool.dispatch(io_class::prefetch) == std::vector<std::size_t>{0, 0, 1, 1});
      CHECK(pool.dispatch(io_class::demand) == std::vector<std::size_t>{1, 1, 0, 0});
      // A single-slice request goes to the least busy reactor of its partition.
      CHECK(pool.dispatch(io_class::prefetch, 1) == std::vector<std::size_t>{0, 0, 1, 0});
      CHECK(pool.dispatch(io_class::demand, 1) == std::vector<std::size_t>{0, 1, 0, 0});
    }
  }

  SECTION("K at or above the reactor count leaves no demand partition: all reactors rank")
  {
    reactor_pool pool{{1000, 10, 20, 500}, 4};
    CHECK(pool.dispatch(io_class::prefetch) == std::vector<std::size_t>{0, 1, 1, 0});
    CHECK(pool.dispatch(io_class::demand) == std::vector<std::size_t>{0, 1, 1, 0});
  }
}

TEST_CASE("prefetch_reactors = 0 ranks every class among all reactors",
          "[io][templated_ioctx][prefetch_reactors]")
{
  using cucascade::io::io_class;
  reactor_pool pool{{1000, 10, 20, 500}, 0};
  for (int round = 0; round < 4; ++round) {
    CHECK(pool.dispatch(io_class::prefetch) == std::vector<std::size_t>{0, 1, 1, 0});
    CHECK(pool.dispatch(io_class::demand) == std::vector<std::size_t>{0, 1, 1, 0});
  }
  CHECK(pool.context->prefetch_reactor_count() == 0);
}
