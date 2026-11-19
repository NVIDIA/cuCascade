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

// ioctx::remove_io_object[_async] and the REST DeleteObject behind it, against the loopback
// server's in-memory object store.

#include "io/rest/http/classify.hpp"
#include "io/rest/http/message.hpp"
#include "rest_contract_test_utils.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::listed_object;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_payload;
using cucascade::test::s3_write_policy;
using cucascade::test::write_fault;
using cucascade::test::write_kind;
using namespace std::chrono_literals;

enum class signing : std::uint8_t { presigned, header };

cucascade::io::rest::config delete_config()
{
  auto cfg               = rest_contract_config();
  cfg.max_retry_attempts = 3;
  return cfg;
}

std::shared_ptr<rest_ioctx> make_ioctx(loopback_range_server const& server,
                                       signing how = signing::presigned,
                                       bool start  = true)
{
  cucascade::io::rest::s3::static_credentials creds;
  creds.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  std::shared_ptr<cucascade::io::rest::request_authorizer> authorizer;
  if (how == signing::header) {
    authorizer = std::make_shared<cucascade::io::rest::s3::sigv4_header_authorizer>(
      creds, "us-east-1", server.endpoint());
  } else {
    authorizer = std::make_shared<cucascade::io::rest::s3::sigv4_presigned_authorizer>(
      creds, "us-east-1", server.endpoint());
  }
  auto context = std::make_shared<rest_reactor::reactor_context>(
    delete_config(), std::move(authorizer), nullptr);
  auto ioctx = std::make_shared<rest_ioctx>(1, std::move(context));
  if (start) { ioctx->start(); }
  return ioctx;
}

std::unique_ptr<loopback_range_server> make_store(s3_write_policy policy = {})
{
  policy.missing_key_404 = true;
  return std::make_unique<loopback_range_server>(
    rest_test_payload(4096),
    range_fault_policy{},
    std::vector<listed_object>{},
    std::unordered_map<std::string, cucascade::test::key_response_script>{},
    std::move(policy));
}

std::vector<loopback_range_server::write_record> deletes_of(loopback_range_server const& server)
{
  auto records = server.write_requests();
  std::erase_if(records, [](auto const& r) { return r.kind != write_kind::delete_object; });
  return records;
}

template <class Done>
bool eventually(Done&& done, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() >= deadline) { return done(); }
    std::this_thread::sleep_for(2ms);
  }
  return true;
}

template <class Fn>
std::exception_ptr failure_of(Fn&& fn)
{
  try {
    fn();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

std::string message_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return {};
  }
}

std::error_code code_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return e.code();
  } catch (...) {
    return {};
  }
}

}  // namespace

TEST_CASE("rest_ioctx removes an object with DeleteObject", "[rest][delete]")
{
  auto const how = GENERATE(signing::presigned, signing::header);
  auto server    = make_store();
  server->put_stored_object("dir/a.bin", rest_test_payload(1000));
  server->put_stored_object("dir/b.bin", rest_test_payload(10));
  auto ioctx = make_ioctx(*server, how);
  rest_shutdown_guard cleanup{*ioctx};

  ioctx->remove_io_object("s3://bkt/dir/a.bin");

  CHECK(server->delete_object_count() == 1);
  CHECK_FALSE(server->stored_object("dir/a.bin").has_value());
  CHECK(server->stored_object("dir/b.bin").has_value());
  auto const records = deletes_of(*server);
  REQUIRE(records.size() == 1);
  CHECK(records.front().bucket == "bkt");
  CHECK(records.front().key == "dir/a.bin");
  CHECK(records.front().upload_id.empty());  // DeleteObject, not AbortMultipartUpload
  CHECK(records.front().status == 204);
  if (how == signing::header) {
    CHECK(records.front().header_authorized);
  } else {
    CHECK(records.front().presigned);
  }
  CHECK(eventually([&] { return ioctx->stats().deletes == 1; }));
  // Readers see it gone (the HEAD answers 404).
  CHECK_THROWS(ioctx->open_io_object("s3://bkt/dir/a.bin"));
}

TEST_CASE("rest_ioctx treats DeleteObject 404 as success", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .what   = write_fault::action::status,
                                      .status = 404,
                                      .code   = "NoSuchKey"});
  auto server = make_store(policy);
  auto ioctx  = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  REQUIRE_NOTHROW(ioctx->remove_io_object("s3://bkt/never-written.bin"));
  CHECK(server->delete_object_count() == 1);  // not retried

  // A key that does not exist on the server (its DELETE answers 204 too) is also fine.
  REQUIRE_NOTHROW(ioctx->remove_io_object("s3://bkt/never-written.bin"));
  CHECK(server->delete_object_count() == 2);
}

// Regression: every 404 counted as "already gone", so a misconfigured bucket looked like a
// successful removal.
TEST_CASE("rest_ioctx reports a DeleteObject 404 NoSuchBucket", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .what   = write_fault::action::status,
                                      .status = 404,
                                      .code   = "NoSuchBucket"});
  auto server = make_store(policy);
  server->put_stored_object("still-there.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  auto const failure = failure_of([&] { ioctx->remove_io_object("s3://bkt/still-there.bin"); });
  REQUIRE(failure != nullptr);
  CHECK_THAT(message_of(failure), Catch::Matchers::ContainsSubstring("NoSuchBucket"));
  CHECK(server->delete_object_count() == 1);  // a 404 is final: not retried
  CHECK(server->stored_object("still-there.bin").has_value());
}

TEST_CASE("rest_ioctx treats a DeleteObject 404 without an error code as success", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .what   = write_fault::action::status,
                                      .status = 404,
                                      .code   = ""});
  auto server = make_store(policy);
  auto ioctx  = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  REQUIRE_NOTHROW(ioctx->remove_io_object("s3://bkt/never-written.bin"));
  CHECK(server->delete_object_count() == 1);
  // The 404 was really answered, and its body carries no code the client could read.
  auto const records = deletes_of(*server);
  REQUIRE(records.size() == 1);
  CHECK(records.front().status == 404);
  cucascade::io::http::response_meta meta;
  CHECK_FALSE(cucascade::io::http::extract_error_code(
    loopback_range_server::s3_error_body("", "injected fault"), meta));
  CHECK(meta.amz_error_code.view().empty());
}

// A 404 whose error body the client could not read to its end (here: larger than the transport
// drains) carries no code: it may be a NoSuchBucket as well as a NoSuchKey, so it is retried like
// a transport error instead of being taken for "already gone".
TEST_CASE("rest_ioctx retries a DeleteObject 404 whose error body was not read", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(
    write_fault{.kind   = write_kind::delete_object,
                .what   = write_fault::action::status,
                .status = 404,
                .code   = "NoSuchBucket",
                .body   = loopback_range_server::s3_error_body(
                  "NoSuchBucket", std::string(cucascade::io::http::max_error_body, 'x'))});
  auto server = make_store(policy);
  server->put_stored_object("big-404.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  ioctx->remove_io_object("s3://bkt/big-404.bin");
  auto const records = deletes_of(*server);
  REQUIRE(records.size() == 2);
  CHECK(records[0].status == 404);
  CHECK(records[1].status == 204);
  CHECK_FALSE(server->stored_object("big-404.bin").has_value());
}

TEST_CASE("rest_ioctx reports a DeleteObject 403 without retrying it", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .what   = write_fault::action::status,
                                      .status = 403,
                                      .code   = "AccessDenied"});
  auto server = make_store(policy);
  server->put_stored_object("locked.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  auto const failure = failure_of([&] { ioctx->remove_io_object("s3://bkt/locked.bin"); });
  REQUIRE(failure != nullptr);
  CHECK_THAT(message_of(failure), Catch::Matchers::ContainsSubstring("delete_object"));
  CHECK(server->delete_object_count() == 1);
  CHECK(server->stored_object("locked.bin").has_value());
}

TEST_CASE("rest_ioctx retries a throttled DeleteObject", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .what   = write_fault::action::status,
                                      .status = 503,
                                      .code   = "SlowDown"});
  auto server = make_store(policy);
  server->put_stored_object("busy.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  ioctx->remove_io_object("s3://bkt/busy.bin");
  CHECK(server->delete_object_count() == 2);
  auto const records = deletes_of(*server);
  REQUIRE(records.size() == 2);
  CHECK(records[0].status == 503);
  CHECK(records[1].status == 204);
  CHECK_FALSE(server->stored_object("busy.bin").has_value());
}

TEST_CASE("rest_ioctx gives up on a persistently failing DeleteObject", "[rest][delete]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind   = write_kind::delete_object,
                                      .count  = 100,
                                      .what   = write_fault::action::status,
                                      .status = 500,
                                      .code   = "InternalError"});
  auto server = make_store(policy);
  auto ioctx  = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  auto future        = ioctx->remove_io_object_async("s3://bkt/x.bin");
  auto const failure = failure_of([&] { std::move(future).get(30s); });
  REQUIRE(failure != nullptr);
  CHECK_THAT(message_of(failure), Catch::Matchers::ContainsSubstring("exhausted retries"));
  CHECK(server->delete_object_count() == delete_config().max_retry_attempts);
}

TEST_CASE("rest_ioctx removal runs on a private ring before start", "[rest][delete]")
{
  auto server = make_store();
  server->put_stored_object("cold.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server, signing::presigned, /*start=*/false);

  auto future = ioctx->remove_io_object_async("s3://bkt/cold.bin");
  REQUIRE_NOTHROW(std::move(future).get(10s));
  CHECK(server->delete_object_count() == 1);
  CHECK_FALSE(server->stored_object("cold.bin").has_value());
  ioctx->shutdown();
}

TEST_CASE("rest_ioctx removal survives a dropped future", "[rest][delete]")
{
  auto control = std::make_shared<cucascade::test::range_response_control>();
  s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::delete_object, .what = write_fault::action::hold, .control = control});
  auto server = make_store(policy);
  server->put_stored_object("detached.bin", rest_test_payload(64));
  auto ioctx = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx, control};

  // Drop the future while the request is held at the server: the request keeps running and
  // completes on its own, with nobody waiting.
  {
    auto dropped = ioctx->remove_io_object_async("s3://bkt/detached.bin");
  }
  REQUIRE(control->wait_for_holds(1, 5s));
  CHECK(server->stored_object("detached.bin").has_value());
  control->release_all();
  REQUIRE(eventually([&] { return !server->stored_object("detached.bin").has_value(); }));
  CHECK(server->delete_object_count() == 1);

  // Many detached low-priority removals in a row (each job owns its outcome: nothing is kept).
  for (int i = 0; i < 32; ++i) {
    auto const key = "many/" + std::to_string(i);
    server->put_stored_object(key, rest_test_payload(8));
    std::ignore = ioctx->remove_io_object_async("s3://bkt/" + key, cucascade::io::io_priority::low);
  }
  REQUIRE(eventually([&] { return server->delete_object_count() == 33; }));
  REQUIRE(eventually([&] { return server->stored_keys().empty(); }));
}

TEST_CASE("rest_ioctx shutdown settles a removal in flight", "[rest][delete]")
{
  auto control = std::make_shared<cucascade::test::range_response_control>();
  s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::delete_object, .what = write_fault::action::hold, .control = control});
  auto server = make_store(policy);
  auto ioctx  = make_ioctx(*server);

  auto kept = ioctx->remove_io_object_async("s3://bkt/kept.bin");
  {
    auto dropped = ioctx->remove_io_object_async("s3://bkt/dropped.bin");
  }
  REQUIRE(control->wait_for_holds(1, 5s));
  // Destroy the context with both requests outstanding: the kept one fails, the dropped one is
  // settled with nobody listening, and nothing touches freed state.
  ioctx->shutdown();
  ioctx.reset();
  control->release_all();
  auto const failure = failure_of([&] { std::move(kept).get(10s); });
  // The held request may have completed just before the shutdown cancelled it.
  if (failure != nullptr) { CHECK(code_of(failure) == std::errc::operation_canceled); }
}

TEST_CASE("rest_ioctx rejects removing a non-s3 path through the future", "[rest][delete]")
{
  auto server = make_store();
  auto ioctx  = make_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};

  auto future        = ioctx->remove_io_object_async("/tmp/local.bin");
  auto const failure = failure_of([&] { std::move(future).get(5s); });
  REQUIRE(failure != nullptr);
  CHECK_THAT(message_of(failure), Catch::Matchers::ContainsSubstring("unsupported scheme"));
  CHECK(server->delete_object_count() == 0);
  CHECK_THROWS_AS(ioctx->remove_io_object(""), std::invalid_argument);
}

TEST_CASE("local backends report remove_io_object as not supported", "[rest][delete]")
{
  constexpr std::size_t MiB = std::size_t{1} << 20;
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource staging{
    0, upstream, 4 * MiB, 4 * MiB, MiB, 4, 0};
  auto ctx = std::make_shared<cucascade::io::uring::uring_ioctx>(
    1,
    std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
      cucascade::io::uring::config{}, &staging));

  auto const failure = failure_of([&] { ctx->remove_io_object("/tmp/whatever.bin"); });
  REQUIRE(failure != nullptr);
  CHECK(code_of(failure) == std::errc::not_supported);
  bool is_not_implemented = false;
  try {
    std::rethrow_exception(failure);
  } catch (cucascade::io::not_implemented_error const&) {
    is_not_implemented = true;
  } catch (...) {
  }
  CHECK(is_not_implemented);
  ctx->shutdown();
}
