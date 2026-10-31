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

#include "http_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <span>

namespace http_test {

using namespace cucascade::io::http;

head_result parse_head(std::string_view msg,
                       std::vector<std::size_t> const& cuts,
                       parse_options opts)
{
  head_result r;
  response_parser p{opts};
  std::string pending;
  std::size_t base = 0;  // message offset of pending[0]
  std::size_t pos  = 0;  // next message byte to feed
  std::size_t ci   = 0;
  auto on_field    = [&r](std::string_view n, std::string_view v) noexcept {
    r.fields.emplace_back(std::string{n}, std::string{v});
  };
  for (;;) {
    std::size_t next = msg.size();
    while (ci < cuts.size()) {
      std::size_t const c = cuts[ci++];
      if (c > pos && c < msg.size()) {
        next = c;
        break;
      }
    }
    pending.append(msg.substr(pos, next - pos));
    pos                  = next;
    std::size_t consumed = 0;
    r.status             = p.parse_header(
      std::span<char const>{pending.data(), pending.size()}, consumed, r.meta, r.ec, on_field);
    REQUIRE(consumed <= pending.size());
    base += consumed;
    pending.erase(0, consumed);
    if (r.status != parse_status::need_more || pos == msg.size()) break;
  }
  r.body_offset    = base;
  r.detail         = p.last_error();
  r.body_remaining = p.body_remaining();
  r.done           = p.done();
  r.interim        = p.interim_responses();
  return r;
}

chunked_result decode_chunked(std::string_view body,
                              std::vector<std::size_t> const& cuts,
                              std::uint64_t limit)
{
  chunked_result r;
  chunked_decoder d{limit};
  std::string pending;
  std::size_t pos = 0;
  std::size_t ci  = 0;
  auto sink       = [&r](std::span<char const> s) noexcept {
    r.body.append(s.data(), s.size());
    ++r.sink_calls;
  };
  for (;;) {
    std::size_t next = body.size();
    while (ci < cuts.size()) {
      std::size_t const c = cuts[ci++];
      if (c > pos && c < body.size()) {
        next = c;
        break;
      }
    }
    pending.append(body.substr(pos, next - pos));
    pos                  = next;
    std::size_t consumed = 0;
    r.status = d.feed(std::span<char const>{pending.data(), pending.size()}, consumed, sink, r.ec);
    REQUIRE(consumed <= pending.size());
    r.consumed += consumed;
    pending.erase(0, consumed);
    if (r.status != parse_status::need_more || pos == body.size()) break;
  }
  r.detail = d.last_error();
  if (r.status == parse_status::ok) REQUIRE(d.done());
  return r;
}

message_result parse_message(std::string_view msg,
                             std::vector<std::size_t> const& cuts,
                             parse_options opts)
{
  message_result r;
  r.head = parse_head(msg, cuts, opts);
  if (r.head.status != parse_status::ok) return r;
  std::string_view const rest = msg.substr(r.head.body_offset);
  switch (r.head.meta.body) {
    case framing::none: r.body_complete = true; break;
    case framing::length: {
      auto const cl   = r.head.meta.content_length.value_or(0);
      r.body_complete = rest.size() >= cl;
      r.body          = std::string{rest.substr(0, static_cast<std::size_t>(cl))};
      break;
    }
    case framing::chunked: {
      std::vector<std::size_t> shifted;
      for (std::size_t c : cuts) {
        if (c > r.head.body_offset) shifted.push_back(c - r.head.body_offset);
      }
      r.chunked       = decode_chunked(rest, shifted);
      r.body          = r.chunked.body;
      r.body_complete = r.chunked.status == parse_status::ok;
      break;
    }
    case framing::until_eof:
      r.body          = std::string{rest};
      r.body_complete = true;
      break;
  }
  return r;
}

std::vector<std::vector<std::size_t>> split_plans(std::size_t size)
{
  std::vector<std::vector<std::size_t>> plans;
  plans.emplace_back();
  for (std::size_t i = 1; i < size; ++i)
    plans.push_back({i});
  std::vector<std::size_t> drip;
  for (std::size_t i = 1; i < size; ++i)
    drip.push_back(i);
  plans.push_back(std::move(drip));
  return plans;
}

namespace {

void check_same_meta(response_meta const& a, response_meta const& b)
{
  CHECK(a.status == b.status);
  CHECK(a.version == b.version);
  CHECK(a.keep_alive == b.keep_alive);
  CHECK(a.body == b.body);
  CHECK(a.content_length == b.content_length);
  CHECK(a.range.has_value() == b.range.has_value());
  if (a.range && b.range) {
    CHECK(a.range->first == b.range->first);
    CHECK(a.range->last == b.range->last);
    CHECK(a.range->complete == b.range->complete);
    CHECK(a.range->unsatisfied == b.range->unsatisfied);
  }
  CHECK(a.etag.view() == b.etag.view());
  CHECK(a.request_id.view() == b.request_id.view());
  CHECK(a.amz_id_2.view() == b.amz_id_2.view());
  CHECK(a.date.view() == b.date.view());
  CHECK(a.retry_after_ms == b.retry_after_ms);
  CHECK(a.truncated_fields == b.truncated_fields);
}

void check_same(message_result const& a, message_result const& b)
{
  CHECK(a.head.status == b.head.status);
  CHECK(a.head.ec == b.head.ec);
  CHECK(a.head.detail == b.head.detail);
  if (a.head.status != parse_status::error) {
    CHECK(a.head.body_offset == b.head.body_offset);
    CHECK(a.head.fields == b.head.fields);
    CHECK(a.head.body_remaining == b.head.body_remaining);
    CHECK(a.head.done == b.head.done);
    CHECK(a.head.interim == b.head.interim);
    check_same_meta(a.head.meta, b.head.meta);
  }
  CHECK(a.chunked.status == b.chunked.status);
  CHECK(a.chunked.detail == b.chunked.detail);
  CHECK(a.chunked.consumed == b.chunked.consumed);
  CHECK(a.body == b.body);
  CHECK(a.body_complete == b.body_complete);
}

}  // namespace

message_result grind(std::string_view msg, parse_options opts)
{
  INFO("message: " << std::string{msg});
  message_result const whole = parse_message(msg, {}, opts);
  auto const plans           = split_plans(msg.size());
  for (std::size_t i = 1; i < plans.size(); ++i) {
    INFO("split plan " << i << " of " << plans.size() - 1);
    check_same(parse_message(msg, plans[i], opts), whole);
  }
  return whole;
}

parse_error final_error(message_result const& r)
{
  if (r.head.status == parse_status::error) return r.head.detail;
  if (r.chunked.status == parse_status::error) return r.chunked.detail;
  return parse_error::none;
}

void failgrind(std::string_view msg, parse_error expected, parse_options opts)
{
  INFO("message: " << std::string{msg});
  message_result const r = grind(msg, opts);
  REQUIRE(final_error(r) == expected);
}

chunked_result grind_chunked(std::string_view body, std::uint64_t limit)
{
  INFO("chunked body: " << std::string{body});
  chunked_result const whole = decode_chunked(body, {}, limit);
  auto const plans           = split_plans(body.size());
  for (std::size_t i = 1; i < plans.size(); ++i) {
    INFO("split plan " << i);
    chunked_result const r = decode_chunked(body, plans[i], limit);
    CHECK(r.status == whole.status);
    CHECK(r.ec == whole.ec);
    CHECK(r.detail == whole.detail);
    CHECK(r.consumed == whole.consumed);
    CHECK(r.body == whole.body);
  }
  return whole;
}

}  // namespace http_test
