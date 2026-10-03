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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::grouped_coordinator;
using cucascade::io::grouped_io_request;
using cucascade::io::host_source;
using cucascade::io::io_kind;
using cucascade::io::range;
using cucascade::io::request_class;
using cucascade::io::write_options;
using cucascade::io::write_segment;

class fake_object final : public cucascade::io::io_object {
 public:
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 1024; }

 private:
  std::string _path{"fake://object"};
};

/// Minimal read-only backend: exercises the base-class write / runner defaults.
class read_only_ioctx final : public cucascade::io::ioctx {
 public:
  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::uring;
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }
  [[nodiscard]] std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<const cucascade::io::byte_range> ranges,
    std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }
  std::size_t host_read_io(const cucascade::io::io_object&,
                           std::size_t,
                           std::size_t size,
                           std::uint8_t*) override
  {
    return size;
  }
  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    const cucascade::io::io_object&,
    std::vector<cucascade::io::prepared_io_slice>&& slices,
    cucascade::io::io_options opts = {}) noexcept override
  {
    last_read_class   = opts.cls;
    std::size_t bytes = 0;
    for (auto const& slice : slices) {
      bytes += slice.size();
    }
    return cucascade::exec::make_semi_future<std::size_t>(bytes);
  }

  request_class last_read_class{request_class::automatic};

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string) override
  {
    return std::make_shared<fake_object>();
  }
};

template <class Exception, class Future>
void require_throws(Future&& future)
{
  REQUIRE(future.is_ready());
  REQUIRE_THROWS_AS(std::move(future).get(), Exception);
}

void require_not_supported(cucascade::exec::semi_future<std::size_t> future)
{
  try {
    static_cast<void>(std::move(future).get());
    FAIL("expected not_supported");
  } catch (std::system_error const& error) {
    REQUIRE(error.code() == std::errc::not_supported);
  }
}

}  // namespace

TEST_CASE("finalizer runs once and may add tasks", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(100, 2);
  auto future      = coordinator->get_future();
  int runs         = 0;
  coordinator->set_finalizer([&runs](grouped_coordinator& self) noexcept {
    ++runs;
    self.add_tasks(1);
  });
  REQUIRE(coordinator->has_finalizer());

  coordinator->on_complete();
  REQUIRE(runs == 0);
  coordinator->on_complete();
  REQUIRE(runs == 1);
  REQUIRE(coordinator->finalizer_started());
  REQUIRE_FALSE(future.is_ready());
  REQUIRE(coordinator->tasks_remaining() == 1);

  coordinator->on_complete();
  REQUIRE(runs == 1);
  REQUIRE(future.is_ready());
  REQUIRE(std::move(future).get() == 100);
}

TEST_CASE("finalizer that schedules nothing resolves the request", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(7, 1);
  int runs         = 0;
  coordinator->set_finalizer([&runs](grouped_coordinator&) noexcept { ++runs; });
  auto future = coordinator->get_future();
  coordinator->on_complete();
  REQUIRE(runs == 1);
  REQUIRE(std::move(future).get() == 7);
}

TEST_CASE("finalizer may be set after get_future and complete work inline", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(5, 1);
  auto future      = coordinator->get_future();
  int runs         = 0;
  coordinator->set_finalizer([&runs](grouped_coordinator& self) noexcept {
    ++runs;
    self.add_tasks(2);
    self.on_complete();  // inline completion must not re-run the finalizer
    self.on_complete();
  });
  coordinator->on_complete();
  REQUIRE(runs == 1);
  REQUIRE(std::move(future).get() == 5);
}

TEST_CASE("error before the last credit skips the finalizer", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(10, 2);
  auto future      = coordinator->get_future();
  int runs         = 0;
  coordinator->set_finalizer([&runs](grouped_coordinator&) noexcept { ++runs; });
  coordinator->report_error(std::make_error_code(std::errc::io_error));
  coordinator->on_complete();
  REQUIRE(runs == 0);
  REQUIRE_FALSE(coordinator->finalizer_started());
  require_throws<std::system_error>(std::move(future));
}

TEST_CASE("finalizer can fail the request", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(10, 1);
  auto future      = coordinator->get_future();
  coordinator->set_finalizer([](grouped_coordinator& self) noexcept {
    self.add_tasks(1);
    self.report_error(std::make_error_code(std::errc::no_space_on_device));
  });
  coordinator->on_complete();
  try {
    static_cast<void>(std::move(future).get());
    FAIL("expected an error");
  } catch (std::system_error const& error) {
    REQUIRE(error.code() == std::errc::no_space_on_device);
  }
}

TEST_CASE("finalizer can only be set once and must be non-empty", "[io][coordinator]")
{
  grouped_coordinator coordinator(1, 1);
  REQUIRE_THROWS_AS(coordinator.set_finalizer(grouped_coordinator::finalize_fn{}),
                    std::invalid_argument);
  coordinator.set_finalizer([](grouped_coordinator&) noexcept {});
  REQUIRE_THROWS_AS(coordinator.set_finalizer([](grouped_coordinator&) noexcept {}),
                    std::logic_error);
  coordinator.on_complete();
}

TEST_CASE("concurrent settles run the finalizer exactly once", "[io][coordinator]")
{
  constexpr std::size_t n_threads = 8;
  constexpr std::size_t per_task  = 1000;
  for (int iteration = 0; iteration < 20; ++iteration) {
    auto coordinator = std::make_shared<grouped_coordinator>(42, n_threads * per_task);
    auto future      = coordinator->get_future();
    std::atomic<int> runs{0};
    coordinator->set_finalizer([&runs](grouped_coordinator& self) noexcept {
      runs.fetch_add(1);
      self.add_tasks(1);
      std::thread([&self] { self.on_complete(); }).join();
    });
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < n_threads; ++t) {
      threads.emplace_back([&coordinator] {
        for (std::size_t i = 0; i < per_task; ++i) {
          coordinator->on_complete();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    REQUIRE(runs.load() == 1);
    REQUIRE(std::move(future).get() == 42);
  }
}

TEST_CASE("request classification", "[io][coordinator]")
{
  using cucascade::io::resolve_request_class;
  REQUIRE(resolve_request_class(request_class::automatic, io_kind::read, 4096) ==
          request_class::latency);
  REQUIRE(resolve_request_class(request_class::automatic, io_kind::read, 1UL << 20) ==
          request_class::read);
  REQUIRE(resolve_request_class(request_class::automatic, io_kind::read, 16, true) ==
          request_class::background);
  REQUIRE(resolve_request_class(request_class::automatic, io_kind::write, 16) ==
          request_class::write);
  REQUIRE(resolve_request_class(request_class::automatic, io_kind::commit, 0) ==
          request_class::write);
  REQUIRE(resolve_request_class(request_class::background, io_kind::write, 16) ==
          request_class::background);
}

TEST_CASE("write segment validation", "[io][coordinator]")
{
  std::vector<std::uint8_t> buffer(64);
  std::vector<write_segment> disjoint{{range{32, 16}, host_source{buffer.data()}},
                                      {range{0, 32}, host_source{buffer.data()}}};
  REQUIRE(cucascade::io::validate_write_segments(disjoint) == 48);

  std::vector<write_segment> overlap{{range{0, 32}, host_source{buffer.data()}},
                                     {range{31, 4}, host_source{buffer.data()}}};
  REQUIRE_THROWS_AS(cucascade::io::validate_write_segments(overlap), std::invalid_argument);

  std::vector<write_segment> null_source{{range{0, 1}, host_source{nullptr}}};
  REQUIRE_THROWS_AS(cucascade::io::validate_write_segments(null_source), std::invalid_argument);
}

TEST_CASE("write and control grouped requests", "[io][coordinator]")
{
  auto object = std::make_shared<fake_object>();
  std::vector<std::uint8_t> buffer(64);
  std::vector<write_segment> segments{{range{0, 16}, host_source{buffer.data()}},
                                      {range{16, 8}, host_source{buffer.data() + 16}}};
  auto write = grouped_io_request::create_write(object, segments);
  REQUIRE(write->is_write());
  REQUIRE(write->meta.cls == request_class::write);
  REQUIRE(write->meta.kind == io_kind::write);
  REQUIRE(write->remaining_bytes() == 24);
  REQUIRE(write->remaining_write_segments() == 2);
  auto future = write->coordinator->get_future();
  static_cast<void>(write->take_front_write_segment());
  write->coordinator->on_complete();
  REQUIRE(write->remaining_bytes() == 8);
  write->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
  REQUIRE(write->empty());
  require_throws<std::system_error>(std::move(future));

  auto control = grouped_io_request::create_control(
    object, io_kind::commit, write_options{}, std::make_shared<grouped_coordinator>(0, 1));
  REQUIRE(control->is_control());
  REQUIRE(control->control_pending());
  REQUIRE_FALSE(control->empty());
  auto control_future = control->coordinator->get_future();
  control->take_control();
  REQUIRE(control->empty());
  control->coordinator->on_complete();
  REQUIRE(std::move(control_future).get() == 0);

  REQUIRE_THROWS_AS(
    grouped_io_request::create_control(
      object, io_kind::read, write_options{}, std::make_shared<grouped_coordinator>(0, 1)),
    std::invalid_argument);

  auto read = grouped_io_request::create(object, {});
  REQUIRE(read->meta.kind == io_kind::read);
  REQUIRE(read->meta.cls == request_class::latency);
  REQUIRE(read->meta.id != write->meta.id);
}

TEST_CASE("ioctx write and runner defaults", "[io][coordinator]")
{
  read_only_ioctx ctx;
  auto object = std::make_shared<fake_object>();
  std::vector<std::uint8_t> buffer(64);

  REQUIRE_FALSE(ctx.supports_write());
  REQUIRE_FALSE(ctx.supports_device_write());
  REQUIRE(ctx.active_runners() == 0);
  REQUIRE(ctx.stats().active_runners == 0);

  REQUIRE_THROWS_AS(ctx.open_io_object_for_write("file:///tmp/x"), std::system_error);
  REQUIRE_THROWS_AS(ctx.host_write(*object, 0, 8, buffer.data()), std::system_error);
  REQUIRE(ctx.host_write(*object, 0, 0, nullptr) == 0);
  REQUIRE_THROWS_AS(ctx.host_write(*object, 0, 8, nullptr), std::invalid_argument);

  require_not_supported(ctx.host_write_async(*object, 0, 8, buffer.data()));
  REQUIRE(ctx.host_write_async(*object, 0, 0, buffer.data()).is_ready());

  std::vector<write_segment> overlap{{range{0, 32}, host_source{buffer.data()}},
                                     {range{8, 4}, host_source{buffer.data()}}};
  require_throws<std::invalid_argument>(ctx.writev_async(*object, std::move(overlap)));

  REQUIRE_THROWS_AS(std::move(ctx.flush_async(*object)).get(), std::system_error);
  REQUIRE_THROWS_AS(std::move(ctx.commit_async(*object)).get(), std::system_error);

  REQUIRE_THROWS_AS(ctx.run(std::stop_token{}), std::logic_error);
  REQUIRE_THROWS_AS(ctx.run_for(std::chrono::milliseconds{1}), std::logic_error);
  REQUIRE_THROWS_AS(ctx.run_until(std::chrono::steady_clock::now()), std::logic_error);

  // Read classification on the uncached path.
  REQUIRE(std::move(ctx.host_read_async(*object, 0, 64, buffer.data())).get() == 64);
  REQUIRE(ctx.last_read_class == request_class::latency);
  cucascade::io::io_options opts{request_class::background};
  REQUIRE(std::move(ctx.host_read_async(*object, 0, 64, buffer.data(), nullptr, opts)).get() == 64);
  REQUIRE(ctx.last_read_class == request_class::background);
}
