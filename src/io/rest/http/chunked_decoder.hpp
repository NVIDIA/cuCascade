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

/// @file chunked_decoder.hpp
/// @brief `Transfer-Encoding: chunked` body decoder (cold path: error bodies, listings).
/// Contract: gpuio docs/arch/API.md §A.12. Port of Beast `parse_chunk_header` / `parse_chunk_body`
/// / trailer handling (gpuio docs/research/beast.md §2.2.5): resume offset for partial chunk lines,
/// 8 KiB chunk-line cap, full chunk-extension grammar, trailers parsed and ignored (RFC 9112
/// §7.1.2: framing fields in a trailer must not affect the message).

#pragma once

#include "io/rest/http/message.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <system_error>
#include <type_traits>

namespace cucascade::io::http {

class chunked_decoder {  // cold path: error bodies, listings
 public:
  chunked_decoder() noexcept = default;
  /// @param body_limit maximum decoded body bytes; a chunk that would exceed it fails with
  /// `errc::http_length_mismatch` (`parse_error::body_limit`) before any of its bytes are emitted.
  explicit chunked_decoder(std::uint64_t body_limit) noexcept : _limit(body_limit) {}

  /// Decode chunked body bytes.
  /// @param in contiguous bytes starting where the previous call stopped consuming.
  /// @param consumed [out] bytes of `in` consumed by this call; on `need_more` keep
  ///        `in[consumed, size)` and present it again followed by new bytes.
  /// @param sink called as `sink(std::span<char const>)` with decoded data, in order.
  /// @param ec [out] `errc::http_bad_message`, `errc::http_header_too_large` (trailers > 16 KiB)
  ///        or `errc::http_length_mismatch` (body limit) on `error`.
  /// @return `ok` after the last chunk and the trailer section (`in[consumed, size)` is the next
  /// message), `need_more`, or `error` (sticky).
  template <class Sink>  // void(std::span<char const>)
  [[nodiscard]] parse_status feed(std::span<char const> in,
                                  std::size_t& consumed,
                                  Sink&& sink,
                                  std::error_code& ec) noexcept
  {
    using sink_t = std::remove_reference_t<Sink>;
    data_sink const s{
      const_cast<void*>(static_cast<void const*>(std::addressof(sink))),
      [](void* ctx, std::span<char const> data) noexcept { (*static_cast<sink_t*>(ctx))(data); }};
    return feed_impl(in, consumed, s, ec);
  }
  [[nodiscard]] bool done() const noexcept { return _state == state::done; }
  /// Prepare for the next body (keeps the body limit).
  void reset() noexcept { *this = chunked_decoder(_limit); }

  /// @return decoded body bytes emitted so far.
  [[nodiscard]] std::uint64_t body_bytes() const noexcept { return _total; }
  /// @return the rule that failed when `feed` returned `error`.
  [[nodiscard]] parse_error last_error() const noexcept { return _error; }

 private:
  struct data_sink {
    void* ctx;
    void (*fn)(void*, std::span<char const>) noexcept;
  };
  enum class state : std::uint8_t { size_line, data, trailers, done, failed };

  parse_status feed_impl(std::span<char const> in,
                         std::size_t& consumed,
                         data_sink sink,
                         std::error_code& ec) noexcept;

  std::uint64_t _limit{unknown_length};
  std::uint64_t _total{0};
  std::uint64_t _remaining{0};      // bytes left in the current chunk
  std::size_t _skip{0};             // resume offset into a partial chunk line
  std::uint32_t _trailer_bytes{0};  // trailer section size so far
  bool _expect_crlf{false};         // the previous chunk's data CRLF is still unread
  state _state{state::size_line};
  parse_error _error{parse_error::none};
};

}  // namespace cucascade::io::http
