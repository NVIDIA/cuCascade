/*
 * Copyright 2026, Sirius Contributors.
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

#include "rest_contract_test_utils.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_request.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using cucascade::io::byte_range;
using cucascade::io::grouped_coordinator;
using cucascade::io::grouped_io_request;
using cucascade::io::host_buffer;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::rest::rest_io_object;
using cucascade::io::rest::rest_reactor;
using cucascade::test::child_test_runner;
using cucascade::test::loopback_range_server;
using cucascade::test::make_rest_contract_reactor;
using cucascade::test::rest_contract_config;
using namespace std::chrono_literals;

std::vector<byte_range> coalesce(std::vector<byte_range> ranges,
                                 std::optional<std::size_t> alignment = std::nullopt)
{
  return rest_reactor::align_and_coalesce(ranges, alignment);
}

cucascade::exec::semi_future<std::size_t> enqueue_one(rest_reactor& reactor,
                                                      std::uint8_t* destination)
{
  auto object      = std::make_shared<rest_io_object>("s3://bucket/object", "bucket", "object", 1);
  auto coordinator = std::make_shared<grouped_coordinator>(1, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 1}, host_buffer{destination});
  reactor.enqueue(grouped_io_request::create(std::move(object), std::move(slices), coordinator));
  return future;
}

}  // namespace

TEST_CASE("rest_reactor::supports only accepts s3 URLs", "[rest][reactor]")
{
  CHECK(rest_reactor::supports("s3://bucket/key"));
  CHECK(rest_reactor::supports("s3://bucket/path/to/obj.parquet"));
  CHECK_FALSE(rest_reactor::supports("file:///tmp/x"));
  CHECK_FALSE(rest_reactor::supports("https://host/obj"));
  CHECK_FALSE(rest_reactor::supports("/local/abs/path"));
  CHECK_FALSE(rest_reactor::supports("not a uri"));
}

TEST_CASE("align_and_coalesce coalesces without alignment by default", "[rest][reactor]")
{
  SECTION("empty input") { CHECK(coalesce({}).empty()); }
  SECTION("zero-size ranges dropped") { CHECK(coalesce({byte_range{100, 0}}).empty()); }
  SECTION("disjoint ranges stay separate and sorted")
  {
    auto out = coalesce({byte_range{200, 50}, byte_range{0, 50}});
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
    CHECK(out[1].offset() == 200);
    CHECK(out[1].size() == 50);
  }
  SECTION("overlapping ranges merge")
  {
    auto out = coalesce({byte_range{0, 100}, byte_range{50, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 150);
  }
  SECTION("adjacent ranges merge")
  {
    auto out = coalesce({byte_range{0, 100}, byte_range{100, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 200);
  }
}

TEST_CASE("align_and_coalesce honors a caller alignment as a lower bound", "[rest][reactor]")
{
  auto out = coalesce({byte_range{100, 100}, byte_range{9000, 100}}, 4096);
  REQUIRE(out.size() == 2);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 8192);
  CHECK(out[1].size() == 4096);
  auto merged = coalesce({byte_range{100, 100}, byte_range{3000, 100}}, 4096);
  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 4096);
}

TEST_CASE("rest_reactor rejects work outside its running lifetime", "[rest][reactor]")
{
  std::array<std::uint8_t, 1> destination{};
  loopback_range_server server({0x5a});
  auto config               = rest_contract_config();
  config.max_connections    = 1;
  config.max_retry_attempts = 10;
  SECTION("before start")
  {
    auto reactor = make_rest_contract_reactor(server, config);
    auto future  = enqueue_one(*reactor, destination.data());
    CHECK(future.is_ready());
    CHECK_THROWS(std::move(future).get());
    CHECK(reactor->queued_bytes() == 0);
  }
  SECTION("after shutdown")
  {
    auto reactor = make_rest_contract_reactor(server, config);
    reactor->start();
    reactor->shutdown();
    auto future = enqueue_one(*reactor, destination.data());
    CHECK(future.is_ready());
    CHECK_THROWS(std::move(future).get());
    CHECK(reactor->queued_bytes() == 0);
  }
}

TEST_CASE("rest_reactor drops an empty grouped request without live-locking", "[rest][reactor]")
{
  auto empty_coordinator = std::make_shared<grouped_coordinator>(0, 0);
  auto empty_future      = empty_coordinator->get_future();
  CHECK(empty_future.is_ready());
  if (auto const* child = std::getenv("CUCASCADE_REST_EMPTY_GROUP_CHILD");
      child == nullptr || std::string_view{child} != "1") {
    child_test_runner process{"rest_reactor drops an empty grouped request without live-locking",
                              {{"CUCASCADE_REST_EMPTY_GROUP_CHILD", std::string{"1"}}}};
    bool const completed = process.wait_for(5s);
    if (!completed) { process.stop(); }
    INFO(process.output());
    REQUIRE(completed);
    REQUIRE(WIFEXITED(process.status()));
    CHECK(WEXITSTATUS(process.status()) == 0);
    return;
  }

  std::array<std::uint8_t, 1> destination{};
  loopback_range_server server({0x5a});
  auto config               = rest_contract_config();
  config.max_connections    = 1;
  config.max_retry_attempts = 1;
  auto reactor              = make_rest_contract_reactor(server, config);
  reactor->start();
  auto object = std::make_shared<rest_io_object>("s3://bucket/object", "bucket", "object", 1);
  reactor->enqueue(grouped_io_request::create(std::move(object), {}, empty_coordinator));
  auto future = enqueue_one(*reactor, destination.data());
  static_cast<void>(std::move(future).get_try());
  reactor->shutdown();
}
