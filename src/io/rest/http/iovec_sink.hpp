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

/// @file iovec_sink.hpp
/// @brief Cursor over destination iovecs: the body goes straight into caller memory.
/// Contract: gpuio docs/arch/API.md §A.12. Design: gpuio docs/research/beast.md §3.1 #5 (Beast
/// `buffer_body` window + `span_body::reader::init` capacity check, minus the parser-side copy) and
/// gpuio ARCHITECTURE §3.5 (leftover copy once per GET, then direct RECVs into the destination).

#pragma once

#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <system_error>

namespace cucascade::io::http {

/// Cursor over destination iovecs: window() for direct receive, commit() after the CQE.
/// Zero-length iovecs are skipped. Never allocates; trivially copyable.
class iovec_sink {
 public:
  /// An empty sink (capacity 0); assign a real one per exchange (`conn_slot` holds one).
  iovec_sink() noexcept = default;
  explicit iovec_sink(std::span<iovec const> dst) noexcept;

  /// Validate the response length against the space left before any body byte arrives.
  /// @return `errc::http_length_mismatch` when `content_length > remaining()`, else success.
  [[nodiscard]] std::error_code init(std::optional<std::uint64_t> content_length) const noexcept;
  /// Copy body bytes that arrived with the header (the only copy per GET).
  /// @note Bytes beyond `remaining()` are dropped (precondition: `leftover.size() <= remaining()`).
  void copy_in(std::span<char const> leftover) noexcept;
  /// Describe the next receive window without advancing.
  /// @param scratch caller iovec array the window is written into (e.g. UIO_MAXIOV entries).
  /// @param max_bytes byte cap for this receive.
  /// @return the filled prefix of `scratch` (empty when the sink is full).
  [[nodiscard]] std::span<iovec> window(std::span<iovec> scratch, std::uint64_t max_bytes) noexcept;
  /// Single-iovec fast path: true when the next `min(max_bytes, remaining())` bytes lie in one
  /// destination iovec (`ptr`, `len` describe them) so a plain RECV suffices; false when the
  /// window spans iovecs (use `window()` + RECVMSG) or the sink is full.
  [[nodiscard]] bool contiguous_window(std::byte*& ptr,
                                       std::uint64_t& len,
                                       std::uint64_t max_bytes) const noexcept;
  /// Advance the cursor by `n` bytes received directly into the window (clamped to remaining()).
  void commit(std::uint64_t n) noexcept;
  [[nodiscard]] std::uint64_t written() const noexcept { return _written; }
  [[nodiscard]] std::uint64_t remaining() const noexcept { return _capacity - _written; }
  [[nodiscard]] std::uint64_t capacity() const noexcept { return _capacity; }

 private:
  void advance(std::uint64_t n) noexcept;

  std::span<iovec const> _dst{};
  std::size_t _idx{0};    // current iovec
  std::uint64_t _off{0};  // offset inside the current iovec
  std::uint64_t _written{0};
  std::uint64_t _capacity{0};
};

}  // namespace cucascade::io::http
