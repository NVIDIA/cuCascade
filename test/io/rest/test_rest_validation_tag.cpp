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

#include "loopback_range_server.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/mock_authorizer.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using cucascade::io::open_hint;
using cucascade::io::rest::config;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using namespace std::chrono_literals;

constexpr std::size_t object_size{4096};
constexpr std::size_t probe_size{512};
constexpr std::string_view object_uri{"s3://bucket/object.bin"};

std::vector<std::uint8_t> test_payload()
{
  std::vector<std::uint8_t> bytes(object_size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xffU);
  }
  return bytes;
}

config test_config()
{
  config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.tls_verify              = false;
  cfg.max_retry_attempts      = 2;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.footer_probe_bytes      = probe_size;
  return cfg;
}

std::unique_ptr<rest_reactor> make_reactor(loopback_range_server const& server, config cfg)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context =
    std::make_shared<rest_reactor::reactor_context>(cfg, std::move(authorizer), nullptr);
  return std::make_unique<rest_reactor>(std::move(context), "validation-tag-test");
}

std::shared_ptr<rest_ioctx> make_ioctx(loopback_range_server const& server, config cfg)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context =
    std::make_shared<rest_reactor::reactor_context>(cfg, std::move(authorizer), nullptr);
  return std::make_shared<rest_ioctx>(1, std::move(context));
}

}  // namespace

TEST_CASE("footer probes capture only the verified response validation tag",
          "[rest][validation_tag]")
{
  SECTION("quoted ETag is preserved")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"footer-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());

    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);

    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag == "\"footer-v1\"");
    CHECK(server.get_count() == 1);
  }

  SECTION("missing ETag stays empty on the probe and io object")
  {
    loopback_range_server server(test_payload());
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());

    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag().empty());
  }

  SECTION("weak ETag is preserved verbatim")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "W/\"weak\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "W/\"weak\"");
  }

  SECTION("empty entity-tag is distinct from a missing header")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"\"");
  }
}

TEST_CASE("HEAD returns size and validation tag without changing the size wrapper",
          "[rest][validation_tag]")
{
  SECTION("quoted ETag is preserved")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());

    auto const result = reactor->head_object("bucket", "object.bin");

    CHECK(result.object_size == object_size);
    CHECK(result.etag == "\"head-v1\"");
    CHECK(reactor->head_object_size("bucket", "object.bin") == object_size);
    CHECK(server.head_count() == 2);
  }

  SECTION("missing ETag does not affect size discovery")
  {
    loopback_range_server server(test_payload());
    auto reactor      = make_reactor(server, test_config());
    auto const result = reactor->head_object("bucket", "object.bin");
    CHECK(result.object_size == object_size);
    CHECK(result.etag.empty());
    CHECK(server.head_count() == 1);
  }
}

TEST_CASE("validation tags are isolated between retry attempts", "[rest][validation_tag]")
{
  SECTION("HEAD uses the successful attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_heads     = 1;
    fault.failed_head_etag     = "\"stale-head\"";
    fault.successful_head_etag = "\"fresh-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag == "\"fresh-head\"");
    CHECK(server.head_count() == 2);
  }

  SECTION("HEAD does not leak a failed attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_heads = 1;
    fault.failed_head_etag = "\"stale-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag.empty());
    CHECK(server.head_count() == 2);
  }

  SECTION("footer probe uses the successful attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_gets     = 1;
    fault.failed_get_etag     = "\"stale-probe\"";
    fault.successful_get_etag = "\"fresh-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag == "\"fresh-probe\"");
    CHECK(server.get_count() == 2);
  }

  SECTION("footer probe does not leak a failed attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 1;
    fault.failed_get_etag = "\"stale-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());
    CHECK(server.get_count() == 2);
  }
}

TEST_CASE("unusable footer responses discard their tag before HEAD fallback",
          "[rest][validation_tag]")
{
  auto check_fallback = [](range_fault_policy fault) {
    fault.failed_get_etag      = "\"discarded-probe\"";
    fault.successful_head_etag = "\"fallback-head\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);

    CHECK(object->size() == object_size);
    CHECK(object->validation_tag() == "\"fallback-head\"");
    CHECK(server.get_count() == 1);
    CHECK(server.head_count() == 1);
  };

  SECTION("200 full-body response")
  {
    range_fault_policy fault{};
    fault.ignore_range_with_200 = true;
    check_fallback(fault);
  }

  SECTION("416 response")
  {
    range_fault_policy fault{};
    fault.fail_range_with_416 = true;
    check_fallback(fault);
  }

  SECTION("malformed 206 response")
  {
    range_fault_policy fault{};
    fault.malformed_content_range = true;
    check_fallback(fault);
  }
}

TEST_CASE("interim response tags do not leak into the final response", "[rest][validation_tag]")
{
  SECTION("HEAD")
  {
    range_fault_policy fault{};
    fault.interim_head_etag = "\"interim-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag.empty());
    CHECK(server.head_count() == 1);
  }

  SECTION("footer probe")
  {
    range_fault_policy fault{};
    fault.interim_get_etag = "\"interim-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("footer probes do not reuse Content-Range from an interim response",
          "[rest][validation_tag]")
{
  range_fault_policy fault{};
  fault.interim_get_content_range = "bytes " + std::to_string(object_size - probe_size) + "-" +
                                    std::to_string(object_size - 1) + "/" +
                                    std::to_string(object_size);
  fault.omit_successful_content_range = true;
  fault.successful_get_etag           = "\"discarded-probe\"";
  fault.successful_head_etag          = "\"fallback-head\"";
  loopback_range_server server(test_payload(), fault);
  auto ioctx  = make_ioctx(server, test_config());
  auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);

  CHECK(object->size() == object_size);
  CHECK(object->validation_tag() == "\"fallback-head\"");
  CHECK(server.get_count() == 1);
  CHECK(server.head_count() == 1);
}

TEST_CASE("retry backoff uses only the final response block", "[rest][validation_tag]")
{
  auto retry_config = [](std::chrono::milliseconds fallback) {
    auto cfg               = test_config();
    cfg.honor_retry_after  = true;
    cfg.retry_backoff_base = fallback;
    return cfg;
  };

  SECTION("HEAD ignores interim Retry-After when the final block omits it")
  {
    range_fault_policy fault{};
    fault.fail_first_heads         = 1;
    fault.interim_head_retry_after = "2";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const result  = reactor->head_object("bucket", "object.bin");
    auto const elapsed = std::chrono::steady_clock::now() - start;

    CHECK(result.object_size == object_size);
    CHECK(elapsed < 1s);
    CHECK(server.head_count() == 2);
  }

  SECTION("HEAD uses Retry-After from the final block")
  {
    range_fault_policy fault{};
    fault.fail_first_heads         = 1;
    fault.interim_head_retry_after = "2";
    fault.failed_head_retry_after  = "0";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1500ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const result  = reactor->head_object("bucket", "object.bin");
    auto const elapsed = std::chrono::steady_clock::now() - start;

    CHECK(result.object_size == object_size);
    CHECK(elapsed < 1s);
    CHECK(server.head_count() == 2);
  }

  SECTION("footer probe ignores interim Retry-After when the final block omits it")
  {
    range_fault_policy fault{};
    fault.fail_first_gets         = 1;
    fault.interim_get_retry_after = "2";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const probe   = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(probe.bytes != nullptr);
    CHECK(elapsed < 1s);
    CHECK(server.get_count() == 2);
  }

  SECTION("footer probe uses Retry-After from the final block")
  {
    range_fault_policy fault{};
    fault.fail_first_gets         = 1;
    fault.interim_get_retry_after = "2";
    fault.failed_get_retry_after  = "0";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1500ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const probe   = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(probe.bytes != nullptr);
    CHECK(elapsed < 1s);
    CHECK(server.get_count() == 2);
  }
}

TEST_CASE("rest ioctx threads validation tags through every network open path",
          "[rest][validation_tag]")
{
  SECTION("HEAD open")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri});
    CHECK(object->validation_tag() == "\"head-open\"");
  }

  SECTION("footer probe open")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"probe-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"probe-open\"");
  }

  SECTION("footer fallback open")
  {
    range_fault_policy fault{};
    fault.fail_range_with_416  = true;
    fault.failed_get_etag      = "\"discarded-probe\"";
    fault.successful_head_etag = "\"fallback-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"fallback-open\"");
  }
}

TEST_CASE("validation tag capture adds no network requests", "[rest][validation_tag]")
{
  SECTION("HEAD open remains one HEAD")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx = make_ioctx(server, test_config());
    (void)ioctx->open_io_object(std::string{object_uri});
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
  }

  SECTION("footer probe open remains one GET")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"probe-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx = make_ioctx(server, test_config());
    (void)ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("known-size open stays tagless and performs no requests", "[rest][validation_tag]")
{
  range_fault_policy fault{};
  fault.successful_get_etag  = "\"unused-get\"";
  fault.successful_head_etag = "\"unused-head\"";
  loopback_range_server server(test_payload(), fault);
  auto ioctx  = make_ioctx(server, test_config());
  auto object = ioctx->open_io_object(std::string{object_uri}, object_size);

  CHECK(object->size() == object_size);
  CHECK(object->validation_tag().empty());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 0);
}
