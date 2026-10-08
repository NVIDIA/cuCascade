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

// Object identity and conditional reads for the REST backend, against the
// loopback server: which opens get a shareable cache id, which data GETs carry
// If-Match, and which responses fail the read with object_changed_error.  The
// byte cache is not involved; the cache-level behaviour that rests on these ids
// is covered by test/cudf/test_cache_object_identity.cpp.

#include "loopback_range_server.hpp"
#include "mock_authorizer.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::object_changed_error;
using cucascade::io::open_hint;
using cucascade::io::rest::config;
using cucascade::io::rest::footer_resolve_result;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::rest_io_object;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;

constexpr char const* object_path = "s3://bucket/object.bin";

std::vector<std::uint8_t> payload(std::size_t n)
{
  std::vector<std::uint8_t> bytes(n);
  for (std::size_t i = 0; i < n; ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 7U) & 0xffU);
  }
  return bytes;
}

config test_config()
{
  config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.tls_verify              = false;
  cfg.max_retry_attempts      = 3;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.max_connections         = 1;
  return cfg;
}

/// How the context authorizes: a canned URL (no signing), SigV4 query-string
/// presigning, or SigV4 Authorization headers.
enum class signing { canned, presigned, header };

std::shared_ptr<rest_ioctx> make_ctx(loopback_range_server const& server,
                                     signing mode,
                                     config cfg = test_config())
{
  namespace s3 = cucascade::io::rest::s3;
  std::shared_ptr<cucascade::io::rest::request_authorizer> auth;
  s3::static_credentials creds{"rest-access-key", "rest-secret-key", {}, std::nullopt};
  switch (mode) {
    case signing::canned:
      auth = std::make_shared<mock_authorizer>(
        cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
      break;
    case signing::presigned:
      auth =
        std::make_shared<s3::sigv4_presigned_authorizer>(creds, "us-east-1", server.endpoint());
      break;
    case signing::header:
      auth = std::make_shared<s3::sigv4_header_authorizer>(creds, "us-east-1", server.endpoint());
      break;
  }
  auto ctx = std::make_shared<rest_ioctx>(
    1, std::make_shared<rest_reactor::reactor_context>(cfg, auth, nullptr));
  ctx->start();
  return ctx;
}

/// Open @p object_path against a server whose HEAD reports the strong tag
/// "opened-generation", read one range, and check what the data GET(s) looked
/// like on the wire (in both SigV4 modes) and whether the read failed with
/// object_changed_error carrying @p observed.
void verify_conditional_get(range_fault_policy fault,
                            bool expect_error,
                            std::string const& observed = {})
{
  std::string const expected = "\"opened-generation\"";
  fault.successful_head_etag = expected;
  auto const bytes_in        = payload(64U << 10);
  auto const offset          = fault.full_object_with_200 ? std::size_t{0} : std::size_t{17};
  auto const length          = fault.full_object_with_200 ? bytes_in.size() : std::size_t{4096};
  auto const attempts        = fault.fail_all_gets ? std::size_t{1} : fault.fail_first_gets + 1;
  for (auto mode : {signing::presigned, signing::header}) {
    DYNAMIC_SECTION("signing=" << (mode == signing::header ? "header" : "presigned"))
    {
      loopback_range_server server(bytes_in, fault);
      auto ctx = make_ctx(server, mode);
      auto obj = ctx->open_io_object(object_path);
      REQUIRE(obj->validation_tag() == expected);
      REQUIRE(server.head_count() == 1);
      REQUIRE(server.get_count() == 0);

      std::vector<std::uint8_t> out(length);
      if (expect_error) {
        bool caught = false;
        try {
          static_cast<void>(ctx->host_read(*obj, offset, length, out.data()));
        } catch (object_changed_error const& e) {
          caught = true;
          CHECK(e.object_path() == object_path);
          CHECK(e.expected_tag() == expected);
          CHECK(e.observed_tag() == observed);
        }
        REQUIRE(caught);
      } else {
        REQUIRE(ctx->host_read(*obj, offset, length, out.data()) == length);
        CHECK(std::equal(out.begin(), out.end(), bytes_in.begin() + static_cast<long>(offset)));
      }

      // A conditional failure is terminal; only a transient one is retried, and
      // the retry keeps the condition.
      CHECK(server.get_count() == attempts);
      auto const records = server.get_requests();
      REQUIRE(records.size() == attempts);
      auto const range =
        "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + length - 1);
      for (auto const& r : records) {
        CHECK(r.if_match == std::vector<std::string>{expected});
        CHECK(r.ranges == std::vector<std::string>{range});
        CHECK(r.header_authorized == (mode == signing::header));
        CHECK(r.presigned == (mode == signing::presigned));
      }
      ctx->shutdown();
    }
  }
}

}  // namespace

TEST_CASE("cache identity accepts only single strong entity tags", "[rest][cache_identity]")
{
  for (auto const* tag : {"\"abc\"", "\"\"", "\"multipart-5\"", "\"a,b\""}) {
    INFO(tag);
    CHECK(rest_io_object::is_strong_tag(tag));
  }
  for (auto const* tag :
       {"", "*", "W/\"x\"", "w/\"x\"", "open#1", "\"a\",\"b\"", "\"bad\tvalue\"", "\"unclosed"}) {
    INFO(tag);
    CHECK_FALSE(rest_io_object::is_strong_tag(tag));
  }
  std::string const path = "s3://cache-identity/key";
  CHECK(rest_io_object::generation_key(path, "\"abc\"") == path + '\x1f' + "\"abc\"");
}

TEST_CASE("cache identity isolates opens without a usable validator", "[rest][cache_identity]")
{
  // No ETag, a weak one, a wildcard, an unquoted token: none identifies a version.
  for (std::string const tag :
       {std::string{}, std::string{"W/\"x\""}, std::string{"*"}, std::string{"open#1"}}) {
    DYNAMIC_SECTION("tag=" << (tag.empty() ? "<none>" : tag))
    {
      range_fault_policy fault;
      fault.successful_head_etag = tag;
      // The GET answers with a different, strong tag: with nothing to hold it to,
      // the read must still succeed (such an open has no snapshot consistency).
      fault.successful_get_etag = "\"unrelated\"";
      loopback_range_server server(payload(8192), fault);
      auto ctx = make_ctx(server, signing::canned);

      auto a = ctx->open_io_object(object_path);
      auto b = ctx->open_io_object(object_path);
      CHECK(a->validation_tag() == tag);
      CHECK(a->raw_file_cache_id() != b->raw_file_cache_id());
      CHECK(a->object_path() == b->object_path());
      CHECK(a->raw_file_cache_id() != a->object_path());
      CHECK(a->raw_file_cache_id().starts_with(std::string{object_path} + '\x1f'));
      CHECK(server.head_count() == 2);

      std::vector<std::uint8_t> out(100);
      REQUIRE(ctx->host_read(*a, 10, 100, out.data()) == 100);
      REQUIRE(ctx->host_read(*b, 10, 100, out.data()) == 100);
      auto const records = server.get_requests();
      REQUIRE(records.size() == 2);
      for (auto const& r : records) {
        CHECK(r.if_match.empty());
        CHECK(r.ranges == std::vector<std::string>{"bytes=10-109"});
      }

      // An open with a known size does no HEAD and has no tag at all.
      auto c = ctx->open_io_object(object_path, std::uint64_t{8192});
      CHECK(server.head_count() == 2);
      CHECK(c->validation_tag().empty());
      CHECK(c->raw_file_cache_id() != a->raw_file_cache_id());
      CHECK(c->raw_file_cache_id() != b->raw_file_cache_id());
      REQUIRE(ctx->host_read(*c, 10, 100, out.data()) == 100);
      REQUIRE(server.get_requests().size() == 3);
      CHECK(server.get_requests().back().if_match.empty());
      ctx->shutdown();
    }
  }
}

TEST_CASE("cache identity reuses a strong generation across opens", "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.successful_head_etag = "\"same\"";
  fault.successful_get_etag  = "\"same\"";
  loopback_range_server server(payload(8192), fault);
  auto ctx = make_ctx(server, signing::canned);

  auto a = ctx->open_io_object(object_path);
  auto b = ctx->open_io_object(object_path);
  CHECK(a->raw_file_cache_id() == b->raw_file_cache_id());
  CHECK(a->raw_file_cache_id() == rest_io_object::generation_key(object_path, "\"same\""));
  CHECK(a->object_path() == object_path);
  // Sharing is keyed by path and tag, not by object identity or by context.
  CHECK(a.get() != b.get());
  ctx->shutdown();
}

TEST_CASE("cache identity gives a new strong tag a new generation", "[rest][cache_identity]")
{
  // The key is overwritten between the two opens: same path, different ETag.
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = "\"v1\""},
                                 scripted_response{.etag = "\"v1\""},
                                 scripted_response{.etag = "\"v2\""}};
  loopback_range_server server(payload(8192), {}, {}, std::move(scripts));
  auto ctx = make_ctx(server, signing::canned);

  auto first  = ctx->open_io_object(object_path);
  auto again  = ctx->open_io_object(object_path);
  auto second = ctx->open_io_object(object_path);

  CHECK(first->validation_tag() == "\"v1\"");
  CHECK(again->raw_file_cache_id() == first->raw_file_cache_id());
  CHECK(second->validation_tag() == "\"v2\"");
  CHECK(second->raw_file_cache_id() != first->raw_file_cache_id());
  CHECK(second->object_path() == first->object_path());
  CHECK(second->raw_file_cache_id() == rest_io_object::generation_key(object_path, "\"v2\""));
  ctx->shutdown();
}

TEST_CASE("cache identity sends If-Match and does not retry 412", "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.fail_all_gets   = true;
  fault.fail_status     = 412;
  fault.failed_get_etag = "\"untrusted-412-tag\"";
  verify_conditional_get(fault, true);
}

TEST_CASE("cache identity does not retry a 412 with a truncated error body",
          "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.fail_all_gets       = true;
  fault.fail_status         = 412;
  fault.failed_get_etag     = "\"untrusted-412-tag\"";
  fault.truncate_error_body = true;
  verify_conditional_get(fault, true);
}

TEST_CASE("cache identity rejects a different ETag on full-object 200", "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.full_object_with_200 = true;
  fault.successful_get_etag  = "\"replacement\"";
  verify_conditional_get(fault, true, fault.successful_get_etag);
}

TEST_CASE("cache identity accepts a matching ETag on full-object 200", "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.full_object_with_200 = true;
  fault.successful_get_etag  = "\"opened-generation\"";
  verify_conditional_get(fault, false);
}

TEST_CASE("cache identity rejects a missing ETag on 206", "[rest][cache_identity]")
{
  verify_conditional_get({}, true);
}

TEST_CASE("cache identity rejects weak and different ETags on 206", "[rest][cache_identity]")
{
  for (std::string const observed : {"W/\"opened-generation\"", "\"replacement\""}) {
    DYNAMIC_SECTION("observed=" << observed)
    {
      range_fault_policy fault;
      fault.successful_get_etag = observed;
      verify_conditional_get(fault, true, observed);
    }
  }
}

TEST_CASE("cache identity retries 503 with the original If-Match", "[rest][cache_identity]")
{
  range_fault_policy fault;
  fault.fail_first_gets     = 1;
  fault.fail_status         = 503;
  fault.failed_get_etag     = "\"transient-response\"";
  fault.successful_get_etag = "\"opened-generation\"";
  verify_conditional_get(fault, false);
}

TEST_CASE("cache identity discards validators from earlier responses", "[rest][cache_identity]")
{
  SECTION("interim response does not supply the final validator")
  {
    range_fault_policy fault;
    fault.interim_get_etag = "\"opened-generation\"";
    verify_conditional_get(fault, true);
  }
  SECTION("retry response does not inherit the failed attempt validator")
  {
    range_fault_policy fault;
    fault.fail_first_gets = 1;
    fault.fail_status     = 503;
    fault.failed_get_etag = "\"opened-generation\"";
    verify_conditional_get(fault, true);
  }
}

TEST_CASE("cache identity conditions reads of a footer-probe object on the probe ETag",
          "[rest][cache_identity][footer_resolve]")
{
  range_fault_policy fault;
  fault.successful_get_etag = "\"probe-tag\"";
  loopback_range_server server(payload(64U << 10), fault);
  auto cfg               = test_config();
  cfg.footer_probe_bytes = 512;
  auto ctx               = make_ctx(server, signing::canned, cfg);

  auto obj = ctx->open_io_object(object_path, open_hint::parquet_footer_probe);
  CHECK(obj->validation_tag() == "\"probe-tag\"");
  CHECK(obj->raw_file_cache_id() == rest_io_object::generation_key(object_path, "\"probe-tag\""));

  std::vector<std::uint8_t> out(100);
  REQUIRE(ctx->host_read(*obj, 0, 100, out.data()) == 100);
  auto const records = server.get_requests();
  REQUIRE(records.size() == 2);  // the suffix probe, then the data GET
  CHECK(records[0].if_match.empty());
  CHECK(records[1].if_match == std::vector<std::string>{"\"probe-tag\""});
  ctx->shutdown();
}

TEST_CASE("cache identity gives a footer-probe object with a weak ETag its own generation",
          "[rest][cache_identity][footer_resolve]")
{
  range_fault_policy fault;
  fault.successful_get_etag = "W/\"probe-tag\"";
  loopback_range_server server(payload(64U << 10), fault);
  auto cfg               = test_config();
  cfg.footer_probe_bytes = 512;
  auto ctx               = make_ctx(server, signing::canned, cfg);

  auto a = ctx->open_io_object(object_path, open_hint::parquet_footer_probe);
  auto b = ctx->open_io_object(object_path, open_hint::parquet_footer_probe);
  CHECK(a->validation_tag() == "W/\"probe-tag\"");
  CHECK(a->raw_file_cache_id() != b->raw_file_cache_id());

  std::vector<std::uint8_t> out(100);
  REQUIRE(ctx->host_read(*a, 0, 100, out.data()) == 100);
  auto const records = server.get_requests();
  REQUIRE(records.size() == 3);  // two suffix probes, then one data GET
  CHECK(records.back().if_match.empty());
  ctx->shutdown();
}

TEST_CASE("cache identity conditions reads of a batch-resolved object on its footer ETag",
          "[rest][cache_identity][footer_resolve]")
{
  range_fault_policy fault;
  fault.successful_get_etag = "\"batch-tag\"";
  loopback_range_server server(payload(64U << 10), fault);
  auto cfg                        = test_config();
  cfg.footer_probe_bytes          = 512;
  cfg.footer_resolve_max_inflight = 2;
  cfg.footer_resolve_stash_budget = 4 * 512;
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto ctx        = std::make_shared<rest_ioctx>(
    1, std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr));
  ctx->start();

  std::vector<std::string> const paths{"s3://bucket/a.parquet", "s3://bucket/b.parquet"};
  std::vector<footer_resolve_result> results;
  ctx->resolve_footer_objects(
    paths, [&](footer_resolve_result result) { results.push_back(std::move(result)); });
  REQUIRE(results.size() == paths.size());
  std::sort(
    results.begin(), results.end(), [](auto const& l, auto const& r) { return l.index < r.index; });

  for (auto const& result : results) {
    REQUIRE_FALSE(result.error);
    REQUIRE(result.object != nullptr);
    CHECK(result.object->validation_tag() == "\"batch-tag\"");
    CHECK(result.object->raw_file_cache_id() ==
          rest_io_object::generation_key(paths[result.index], "\"batch-tag\""));
  }
  auto const probes = server.get_requests().size();
  CHECK(probes == paths.size());

  std::vector<std::uint8_t> out(100);
  for (auto const& result : results) {
    // The batch object is stashless: this read goes over the network.
    REQUIRE(ctx->host_read(*result.object, 0, 100, out.data()) == 100);
  }
  auto const records = server.get_requests();
  REQUIRE(records.size() == probes + results.size());
  for (std::size_t i = 0; i < probes; ++i) {
    CHECK(records[i].if_match.empty());
  }
  for (std::size_t i = probes; i < records.size(); ++i) {
    CHECK(records[i].if_match == std::vector<std::string>{"\"batch-tag\""});
  }
  ctx->shutdown();
}

TEST_CASE("cache identity conditions reads of a batch object resolved by HEAD fallback",
          "[rest][cache_identity][footer_resolve]")
{
  range_fault_policy fault;
  fault.successful_head_etag = "\"head-tag\"";
  fault.successful_get_etag  = "\"head-tag\"";
  fault.failed_get_etag      = "\"discarded-probe-tag\"";
  // The probe (the first GET) answers with a malformed Content-Range, so the
  // batch falls back to HEAD; the data GET that follows is well-formed.
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["fallback.parquet"].gets = {scripted_response{.malformed_content_range = true},
                                      scripted_response{}};
  loopback_range_server server(payload(64U << 10), fault, {}, std::move(scripts));
  auto cfg                        = test_config();
  cfg.footer_probe_bytes          = 512;
  cfg.footer_resolve_max_inflight = 1;
  cfg.footer_resolve_stash_budget = 4 * 512;
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto ctx        = std::make_shared<rest_ioctx>(
    1, std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr));
  ctx->start();

  std::vector<std::string> const paths{"s3://bucket/fallback.parquet"};
  std::vector<footer_resolve_result> results;
  ctx->resolve_footer_objects(
    paths, [&](footer_resolve_result result) { results.push_back(std::move(result)); });
  REQUIRE(results.size() == 1);
  REQUIRE_FALSE(results.front().error);
  REQUIRE(results.front().object != nullptr);
  CHECK_FALSE(results.front().footer);
  CHECK(server.head_count() == 1);
  CHECK(results.front().object->validation_tag() == "\"head-tag\"");
  CHECK(results.front().object->raw_file_cache_id() ==
        rest_io_object::generation_key(paths.front(), "\"head-tag\""));

  std::vector<std::uint8_t> out(100);
  REQUIRE(ctx->host_read(*results.front().object, 0, 100, out.data()) == 100);
  auto const records = server.get_requests();
  REQUIRE(records.size() == 2);  // the unusable probe, then the data GET
  CHECK(records[0].if_match.empty());
  CHECK(records[1].if_match == std::vector<std::string>{"\"head-tag\""});
  ctx->shutdown();
}
