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

// REST writes (S3 PUT / multipart upload) are not implemented yet: every write
// and flush entry point of the REST context and reactor must fail with
// io::not_implemented_error, without any request reaching the server.

#include "rest_contract_test_utils.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using cucascade::io::grouped_coordinator;
using cucascade::io::grouped_io_request;
using cucascade::io::io_object;
using cucascade::io::ioctx;
using cucascade::io::not_implemented_error;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::write_slice;
using cucascade::io::rest::rest_io_object;
using cucascade::io::rest::rest_reactor;
using cucascade::test::loopback_range_server;
using namespace std::chrono_literals;

// The stubs give the reactor the structural shape of a writing reactor; only
// the explicit traits keep the context from advertising writes.
static_assert(cucascade::io::io_write_reactor_c<rest_reactor>);
static_assert(!cucascade::io::reactor_write_traits<rest_reactor>::supports_write);
static_assert(!cucascade::io::reactor_write_traits<rest_reactor>::supports_device_write);

/// Run @p call and require a not_implemented_error naming @p operation.
template <class Call>
void require_not_implemented(Call&& call, std::string_view operation)
{
  CAPTURE(operation);
  try {
    std::forward<Call>(call)();
    FAIL("expected not_implemented_error");
  } catch (not_implemented_error const& error) {
    // Still a std::system_error with errc::not_supported for existing callers.
    std::system_error const& base = error;
    CHECK(base.code() == std::errc::not_supported);
    std::string_view const what = error.what();
    CHECK(what.find("not yet implemented") != std::string_view::npos);
    CHECK(what.find("rest") != std::string_view::npos);
    CHECK(what.find(operation) != std::string_view::npos);
  }
}

std::shared_ptr<rest_io_object> remote_object()
{
  return std::make_shared<rest_io_object>("s3://bucket/object", "bucket", "object", 4096);
}

void require_no_traffic(loopback_range_server const& server)
{
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 0);
  CHECK(server.list_count() == 0);
}

}  // namespace

TEST_CASE("the REST context advertises no writes", "[io][rest][write][unimplemented]")
{
  loopback_range_server server({0x5a});
  auto context = cucascade::test::make_rest_contract_ioctx(server);
  cucascade::test::rest_shutdown_guard guard{*context};
  ioctx& base = *context;
  CHECK_FALSE(base.supports_write());
  CHECK_FALSE(base.supports_device_write());
}

TEST_CASE("every REST write entry point is not yet implemented", "[io][rest][write][unimplemented]")
{
  loopback_range_server server({0x5a});
  auto context = cucascade::test::make_rest_contract_ioctx(server);
  cucascade::test::rest_shutdown_guard guard{*context};
  ioctx& base = *context;
  auto object = remote_object();

  std::array<std::uint8_t, 16> host{};
  // A device pointer nothing may dereference: the call must fail first.
  auto* device       = reinterpret_cast<std::uint8_t const*>(std::uintptr_t{0x10000});
  auto const stream  = ::cuda::stream_ref{cudaStream_t{nullptr}};
  auto const segment = std::vector<write_slice>{{0, host.size(), host.data()}};
  auto const dev_seg = std::vector<write_slice>{{64, 16, device}};

  SECTION("front wrappers through ioctx&")
  {
    require_not_implemented([&] { std::ignore = base.open_io_object_for_write("s3://bucket/new"); },
                            "open_io_object_for_write");
    require_not_implemented([&] { base.host_write(*object, 0, host.size(), host.data()); },
                            "host_write");
    require_not_implemented(
      [&] { std::ignore = base.host_write_async(*object, 0, host.size(), host.data()).get(); },
      "write");
    require_not_implemented(
      [&] { std::ignore = base.device_write_async(*object, 0, 16, device, stream).get(); },
      "write");
    require_not_implemented([&] { std::ignore = base.host_writev_async(*object, segment).get(); },
                            "write");
    require_not_implemented(
      [&] { std::ignore = base.device_writev_async(*object, dev_seg, stream).get(); }, "write");
    require_not_implemented(
      [&] { std::ignore = base.mixed_writev_async(*object, segment, dev_seg, stream).get(); },
      "write");
    require_not_implemented([&] { base.flush(*object); }, "flush");
    require_not_implemented([&] { base.flush_async(*object).get(); }, "flush_async");
  }

  SECTION("direct backend hooks")
  {
    require_not_implemented([&] { context->host_write_io(*object, 0, host.size(), host.data()); },
                            "host_write");
    std::vector<prepared_io_slice> slices;
    slices.push_back(cucascade::io::make_host_write_slice(range{0, host.size()}, host.data()));
    require_not_implemented(
      [&] { std::ignore = context->mixed_writev_async_io(*object, std::move(slices)).get(); },
      "mixed_writev_async");
    require_not_implemented([&] { context->flush_io(*object); }, "flush");
    require_not_implemented(
      [&] { context->flush_async_io(*object, cucascade::io::io_priority::high).get(); },
      "flush_async");
  }

  SECTION("zero-size writes stay no-ops")
  {
    CHECK(base.host_write(*object, 0, 0, nullptr) == 0);
    CHECK(base.host_write_async(*object, 0, 0, nullptr).get() == 0);
    CHECK(base.device_write_async(*object, 0, 0, nullptr, stream).get() == 0);
    CHECK(base.host_writev_async(*object, {}).get() == 0);
    CHECK(base.mixed_writev_async(*object, {}, {}, stream).get() == 0);
  }

  require_no_traffic(server);
}

TEST_CASE("the REST reactor's write stubs and enqueue reject writes",
          "[io][rest][write][unimplemented]")
{
  loopback_range_server server({0x5a});
  auto reactor = cucascade::test::make_rest_contract_reactor(server);
  reactor->start();
  auto object = remote_object();
  std::array<std::uint8_t, 16> host{};

  require_not_implemented(
    [&] { std::ignore = rest_reactor::create_io_object_for_write("s3://bucket/new", {}); },
    "open_io_object_for_write");
  require_not_implemented([&] { reactor->host_write(*object, 0, host.size(), host.data()); },
                          "host_write");
  require_not_implemented([&] { rest_reactor::flush(*object); }, "flush");

  SECTION("a write request is failed, not read as a GET")
  {
    std::vector<prepared_io_slice> slices;
    slices.push_back(cucascade::io::make_host_write_slice(range{0, 8}, host.data()));
    slices.push_back(cucascade::io::make_host_write_slice(range{8, 8}, host.data() + 8));
    auto coordinator = std::make_shared<grouped_coordinator>(16, slices.size());
    auto future      = coordinator->get_future();
    reactor->enqueue(grouped_io_request::create(object, std::move(slices), coordinator));
    require_not_implemented([&] { std::ignore = std::move(future).get(5s); }, "write");
  }

  SECTION("a flush request is failed")
  {
    std::vector<prepared_io_slice> slices;
    slices.push_back(cucascade::io::make_flush_slice());
    auto coordinator = std::make_shared<grouped_coordinator>(0, 1);
    auto future      = coordinator->get_future();
    reactor->enqueue(grouped_io_request::create(object, std::move(slices), coordinator));
    require_not_implemented([&] { std::ignore = std::move(future).get(5s); }, "write");
  }

  CHECK(reactor->queued_bytes() == 0);
  reactor->shutdown();
  require_no_traffic(server);
}
