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

// Test drivers for the HTTP codec: incremental feeding at arbitrary split points (the way the
// reactor feeds receive completions), whole-message parsing (head + body by framing), and
// "grind" helpers equivalent to Beast's `parsegrind` / `failgrind`: every message is parsed in
// one piece, split at every byte boundary, and byte by byte, and all results must be identical.

#pragma once

#include "io/rest/errc.hpp"
#include "io/rest/http/codec.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace http_test {

using namespace std::string_view_literals;

struct head_result {
  cucascade::io::http::parse_status status{cucascade::io::http::parse_status::need_more};
  std::error_code ec;
  cucascade::io::http::parse_error detail{cucascade::io::http::parse_error::none};
  std::size_t body_offset{0};  ///< message offset of the first body byte (on ok)
  cucascade::io::http::response_meta meta;
  std::vector<std::pair<std::string, std::string>> fields;  ///< on_unknown_field calls, in order
  std::uint64_t body_remaining{0};
  bool done{false};
  std::uint16_t interim{0};
};

struct chunked_result {
  cucascade::io::http::parse_status status{cucascade::io::http::parse_status::need_more};
  std::error_code ec;
  cucascade::io::http::parse_error detail{cucascade::io::http::parse_error::none};
  std::size_t consumed{0};  ///< total bytes consumed
  std::string body;
  std::size_t sink_calls{0};
};

struct message_result {
  head_result head;
  chunked_result chunked;  ///< only for framing::chunked
  std::string body;
  bool body_complete{false};  ///< length framing: all Content-Length bytes present
};

/// Feed `msg` to a response_parser in pieces ending at `cuts` (ascending offsets), re-presenting
/// unconsumed bytes each time like a receive loop.
head_result parse_head(std::string_view msg,
                       std::vector<std::size_t> const& cuts,
                       cucascade::io::http::parse_options opts = {});

/// Feed a chunked body to a chunked_decoder in pieces ending at `cuts`.
chunked_result decode_chunked(std::string_view body,
                              std::vector<std::size_t> const& cuts,
                              std::uint64_t limit = cucascade::io::http::unknown_length);

/// Head, then body according to the parsed framing (chunked bodies through the decoder, fed with
/// the same cut points shifted to the body).
message_result parse_message(std::string_view msg,
                             std::vector<std::size_t> const& cuts,
                             cucascade::io::http::parse_options opts = {});

/// Every cut list worth trying: none, each single split point, and byte-by-byte.
std::vector<std::vector<std::size_t>> split_plans(std::size_t size);

/// Parse `msg` under every split plan; CHECK all results equal the one-piece result; return it.
message_result grind(std::string_view msg, cucascade::io::http::parse_options opts = {});

/// Like grind, and REQUIRE the final result to be an error with `expected` detail.
void failgrind(std::string_view msg,
               cucascade::io::http::parse_error expected,
               cucascade::io::http::parse_options opts = {});

/// The error that ended a message parse (head or chunked body), or none.
cucascade::io::http::parse_error final_error(message_result const& r);

/// Grind a chunked body alone (every split + drip); CHECK identical results; return the whole one.
chunked_result grind_chunked(std::string_view body,
                             std::uint64_t limit = cucascade::io::http::unknown_length);

}  // namespace http_test
