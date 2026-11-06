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

/// @file cipher_ring.hpp
/// @brief Fixed-capacity byte FIFO for TLS ciphertext of the user-space record layer
/// (`tls_stream`, API.md §B.6; gpuio ARCHITECTURE §4.4 "User-space record layer"; tls.md §2.9 (b),
/// §2.10). Tier 1 (gpuio T16) receives into it with `IORING_OP_RECV` and lets OpenSSL pull records
/// out through a custom read BIO; the same type queues the ciphertext `SSL_write_ex` produces until
/// a `SEND` completes. Tier 2 (gpuio T19, own RX record layer) is meant to keep this type and
/// replace only the decrypt step: `readable()` exposes the buffered ciphertext as one span for a
/// streaming `EVP_DecryptUpdate` straight into the destination, `consume()` drops what was
/// decrypted.
///
/// Shape: a LINEAR FIFO `[head, tail)` over inline storage, not a wrapping ring. Every receive
/// targets the contiguous tail (`write_span`), so a record never straddles the end of the
/// storage and no receive is ever split in two. When the FIFO empties both cursors return to 0
/// (the common case: OpenSSL drains it completely before asking for more); `compact()` moves a
/// short remainder (at most one partial TLS record, < 18.5 KiB) to the front before a receive.
/// Nothing here allocates, locks or knows about OpenSSL.
///
/// Wrap mode (gpuio T19, tier 2): `recv_iov()` is the receive target of the own record layer: ALL
/// free space in ring order, `[tail, N)` then `[0, head)` (one RECVMSG with two iovecs), so a
/// receive is as large as with compaction but nothing is ever moved. Bytes that land at the front
/// start a second segment: the buffered bytes are `segments().a` followed by `segments().b` and a
/// record may straddle the two. The record layer decrypts a straddling record from both pieces (two
/// `EVP_DecryptUpdate` calls; only the 5-byte header, TLS 1.2's 8-byte explicit nonce and the
/// 16-byte tag are ever reassembled into locals), so no payload byte is copied. When the first
/// segment drains, the second becomes the first. `size`, `empty`, `peek`, `consume`, `produce`,
/// `segments` and `clear` work in both modes; `write_span`, `tail_space`, `append`, `readable`,
/// `read` and `compact` are linear-mode only (tier 1 and the send FIFO never call `recv_iov`, and
/// their `produce` never exceeds `write_span`, so they never see a wrapped ring).

#pragma once

#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace cucascade::io::net {

template <std::size_t N>
// _buf (N bytes, >= 32 KiB) is deliberately not zeroed; every byte is written before it is read.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init): buffer written before read
class cipher_ring {
  static_assert(N >= std::size_t{32} * 1024 && N <= (1u << 30),
                "one TLS record (<= 18.5 KiB) must fit twice");

 public:
  static constexpr std::size_t capacity = N;

  /// Buffered bytes, oldest first: `a`, then `b` (non-empty only when the ring wrapped).
  struct view {
    std::span<std::byte const> a;
    std::span<std::byte const> b;
    [[nodiscard]] std::size_t size() const noexcept { return a.size() + b.size(); }
  };

  [[nodiscard]] std::size_t size() const noexcept
  {
    return (_tail - _head) + (_wrap ? _wtail : 0u);
  }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }
  /// @return true when the buffered bytes continue at offset 0 (wrap mode, `recv_iov`).
  [[nodiscard]] bool wrapped() const noexcept { return _wrap; }
  /// @return bytes that can still be appended at the tail (without compaction). Linear mode.
  [[nodiscard]] std::size_t tail_space() const noexcept { return N - _tail; }

  /// @return the contiguous free region at the tail (the target of the next receive).
  [[nodiscard]] std::span<std::byte> write_span() noexcept
  {
    return {_buf.data() + _tail, N - _tail};
  }
  /// @brief Commit @p n bytes written into `write_span()` or the `recv_iov()` regions (in ring
  /// order: what does not fit the tail continues at offset 0 and starts the second segment).
  void produce(std::size_t n) noexcept
  {
    if (!_wrap) {
      std::size_t const k = n < N - _tail ? n : N - _tail;
      _tail += static_cast<std::uint32_t>(k);
      n -= k;
      if (n == 0) return;
      _wrap  = true;
      _wtail = 0;
    }
    _wtail += static_cast<std::uint32_t>(n);
  }

  /// @brief Wrap mode (tier 2): the receive target without moving buffered bytes: all free space
  /// in ring order. Empty ring: the whole storage (cursors back to 0). Unwrapped: `[tail, N)`
  /// then `[0, head)`; wrapped: `[b_end, head)`. Commit with `produce(n)`.
  /// @return the number of regions written to @p out (0 only when the ring is full).
  [[nodiscard]] std::size_t recv_iov(std::span<iovec, 2> out) noexcept
  {
    if (size() == 0) {
      _head = _tail = _wtail = 0;
      _wrap                  = false;
      out[0]                 = iovec{_buf.data(), N};
      return 1;
    }
    if (_wrap) {
      if (_head == _wtail) return 0;
      out[0] = iovec{_buf.data() + _wtail, static_cast<std::size_t>(_head - _wtail)};
      return 1;
    }
    std::size_t k = 0;
    if (_tail < N) out[k++] = iovec{_buf.data() + _tail, N - _tail};
    if (_head > 0) out[k++] = iovec{_buf.data(), static_cast<std::size_t>(_head)};
    return k;
  }

  /// @return the buffered bytes as (at most) two contiguous pieces, oldest first.
  [[nodiscard]] view segments() const noexcept
  {
    return {{_buf.data() + _head, static_cast<std::size_t>(_tail - _head)},
            {_buf.data(), _wrap ? static_cast<std::size_t>(_wtail) : std::size_t{0}}};
  }

  /// @brief Append up to @p n bytes at the tail. @return bytes appended.
  std::size_t append(void const* src, std::size_t n) noexcept
  {
    std::size_t const k = n < tail_space() ? n : tail_space();
    if (k != 0) std::memcpy(_buf.data() + _tail, src, k);
    _tail += static_cast<std::uint32_t>(k);
    return k;
  }

  /// @return the buffered bytes, head first. Linear mode.
  [[nodiscard]] std::span<std::byte const> readable() const noexcept
  {
    return {_buf.data() + _head, size()};
  }
  /// @return a mutable view of the buffered bytes (in-place processing). Linear mode.
  [[nodiscard]] std::span<std::byte> readable() noexcept { return {_buf.data() + _head, size()}; }

  /// @brief Copy up to @p n bytes from the head into @p dst and drop them. @return bytes copied.
  /// Linear mode.
  std::size_t read(void* dst, std::size_t n) noexcept
  {
    std::size_t const k = n < size() ? n : size();
    if (k != 0) std::memcpy(dst, _buf.data() + _head, k);
    consume(k);
    return k;
  }

  /// @brief Copy @p n bytes at offset @p off from the head without consuming. @return false when
  /// fewer than off + n bytes are buffered.
  bool peek(std::size_t off, void* dst, std::size_t n) const noexcept
  {
    if (off + n > size()) return false;
    std::size_t const a = _tail - _head;
    auto* out           = static_cast<std::byte*>(dst);
    if (off < a) {
      std::size_t const k = n < a - off ? n : a - off;
      std::memcpy(out, _buf.data() + _head + off, k);
      out += k;
      n -= k;
      off = a;
    }
    if (n != 0) std::memcpy(out, _buf.data() + (off - a), n);
    return true;
  }

  /// @brief Drop @p n bytes from the head (n <= size()); an emptied FIFO restarts at offset 0.
  /// In wrap mode a drained first segment hands over to the second.
  void consume(std::size_t n) noexcept
  {
    _head += static_cast<std::uint32_t>(n);
    if (_head < _tail) return;
    if (_wrap) {
      _head  = _head - _tail;  // bytes consumed past the end of the first segment
      _tail  = _wtail;
      _wtail = 0;
      _wrap  = false;
      if (_head < _tail) return;
    }
    _head = _tail = 0;
  }

  /// @brief Move the buffered bytes to the front so the tail is as long as possible.
  /// @pre no I/O targets the buffered region (the caller owns the timing). Linear mode (a no-op
  /// on a wrapped ring).
  /// @return bytes moved.
  std::size_t compact() noexcept
  {
    if (_head == 0 || _wrap) return 0;
    std::size_t const n = size();
    if (n != 0) std::memmove(_buf.data(), _buf.data() + _head, n);
    _head = 0;
    _tail = static_cast<std::uint32_t>(n);
    return n;
  }

  void clear() noexcept
  {
    _head = _tail = _wtail = 0;
    _wrap                  = false;
  }

 private:
  std::uint32_t _head{0};
  std::uint32_t _tail{0};   ///< end of the first segment
  std::uint32_t _wtail{0};  ///< wrap mode: end of the second segment, which starts at 0
  bool _wrap{false};
  alignas(64) std::array<std::byte, N> _buf;  // not value-initialised: written before read
};

}  // namespace cucascade::io::net
