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

#pragma once

#include <memory>
#include <system_error>
#include <type_traits>
#include <utility>

/**
 * @file errc.hpp
 * @brief Error vocabulary of the io_uring REST transport (ported from gpuio's `error.hpp`).
 *
 * Errors travel as `std::error_code` inside the transport: no exception crosses the reactor
 * thread and nothing allocates on an error path.  `rest_reactor` maps them to
 * `std::system_error` and cuCascade's exceptions at its boundary.  `result<T>` is the
 * value-or-error of the ported code.
 */

namespace cucascade::io::rest {

/// @brief Library error codes (category error_category(), name "cucascade.rest").
enum class errc : int {
  ok = 0,
  operation_canceled,   ///< stop_token triggered or context stopped before completion
  context_stopped,      ///< submitted after request_stop()
  invalid_argument,     ///< null pointer, overlap, misaligned registered buffer, size 0 with ptr
  eof,                  ///< range starts at/after end (per-range hook only; never a request error)
  short_read,           ///< device returned fewer bytes than the file size allows
  io_error,             ///< generic -EIO or malformed short read
  unaligned,            ///< O_DIRECT alignment violated where required
  not_registered,       ///< registered_buffer handle invalid
  memlock_exceeded,     ///< io_uring registration failed with ENOMEM
  ring_error,           ///< fatal io_uring failure (setup, enter with non-retriable error)
  staging_quarantined,  ///< staging run quarantined after a CUDA enqueue failure
  dns_failed,
  connect_failed,
  connect_timeout,
  tls_handshake_failed,
  tls_verify_failed,
  tls_protocol_error,  ///< unexpected control record, EKEYEXPIRED, pending bytes after handshake
  tls_bad_record_mac,  ///< user-space record layer AEAD failure (destination undefined)
  ktls_unavailable,    ///< only with tls_config::ktls == require
  stale_connection,    ///< reused connection died before any response byte (internal; retried)
  http_bad_message,
  http_header_too_large,
  http_length_mismatch,    ///< Content-Length vs received bytes
  content_range_mismatch,  ///< 206 with wrong start/end/total
  range_ignored,           ///< 200 to a ranged GET that cannot be sliced
  object_changed,          ///< 412 Precondition Failed on If-Match
  request_timeout,         ///< stall / attempt deadline
  retries_exhausted,
  throttled,  ///< terminal 503/429 after budget exhaustion
  credentials_unavailable,
  not_supported,
  unhandled_exception,       ///< a then(ex, f) continuation threw something that is not a
                             ///< std::system_error (kept) or std::bad_alloc (not_enough_memory)
  http_unsupported_framing,  ///< 2xx GET body that is chunked or close-delimited (S3 stores
                             ///< always send Content-Length)
};

/// @return the "cucascade.rest" category (values are `errc`).
[[nodiscard]] std::error_category const& error_category() noexcept;
/// @return the "http_status" category (values are HTTP status codes; ARCHITECTURE.md §8).
[[nodiscard]] std::error_category const& http_status_category() noexcept;

/// @brief Found by ADL; makes `std::error_code ec = errc::x;` work.
[[nodiscard]] inline std::error_code make_error_code(errc e) noexcept
{
  return {static_cast<int>(e), error_category()};
}

/// @param status HTTP status code (e.g. 403, 503).
[[nodiscard]] inline std::error_code make_http_error(unsigned status) noexcept
{
  return {static_cast<int>(status), http_status_category()};
}

/// @brief Value-or-error. `value()` throws std::system_error(error()) when !has_value().
/// Trivially copyable when T is. sizeof(result<std::size_t>) == 24.
///
/// The error code is the discriminator: has_value() == !error(). The value lives in a union, so
/// T need not be default constructible (e.g. `result<pinned_pool>`). When T is nothrow default
/// constructible the slot is always alive (value-initialised on the error path); copies and
/// assignments are then unconditional, which keeps trivially copyable T trivial and avoids reads
/// of indeterminate bytes. Otherwise the slot is alive exactly when has_value().
///
/// @note Constructing from a *success* error code (value 0, e.g. `std::error_code{}` or
/// `errc::ok`) is a caller bug: it is normalised to the error `errc::invalid_argument` for every
/// T, so has_value() is false. `result<void>` differs: a success code is a success.
template <class T>
class result {
  static_assert(std::is_object_v<T> && !std::is_array_v<T>,
                "result<T>: T must be a non-array object type");
  static_assert(!std::is_same_v<std::remove_cv_t<T>, std::error_code>,
                "result<std::error_code> is ambiguous; use std::error_code directly");

  static constexpr bool always_alive = std::is_nothrow_default_constructible_v<T>;
  static constexpr bool trivial_copy = std::is_trivially_copy_constructible_v<T>;
  static constexpr bool trivial_move = std::is_trivially_move_constructible_v<T>;
  static constexpr bool trivial_dtor = std::is_trivially_destructible_v<T>;
  static constexpr bool trivial_copy_assign =
    trivial_copy && trivial_dtor && std::is_trivially_copy_assignable_v<T>;
  static constexpr bool trivial_move_assign =
    trivial_move && trivial_dtor && std::is_trivially_move_assignable_v<T>;

 public:
  using value_type = T;

  // Implicit by design (`return value;`), like std::expected.
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design
  result(T value) noexcept(std::is_nothrow_move_constructible_v<T>)
    : _ec{}, _value(std::move(value))
  {
  }
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design (`return ec;`)
  result(std::error_code ec) noexcept : _ec(ec)
  {
    // A success code is not a valid error for result<T> (uniform rule for every T):
    // it is normalised to invalid_argument so that has_value() never lies about the slot.
    if (!_ec) [[unlikely]] { _ec = make_error_code(errc::invalid_argument); }
    if constexpr (always_alive) { std::construct_at(std::addressof(_value)); }
  }
  /// Convenience: `return errc::x;` from a function returning result<T>.
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design (`return errc::x;`)
  result(errc e) noexcept : result(make_error_code(e)) {}

  result(result const&)
    requires trivial_copy
  = default;
  result(result const& other) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires(!trivial_copy && std::is_copy_constructible_v<T>)
    : _ec(other._ec)
  {
    if (alive()) { std::construct_at(std::addressof(_value), other._value); }
  }

  result(result&&)
    requires trivial_move
  = default;
  result(result&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    requires(!trivial_move && std::is_move_constructible_v<T>)
    : _ec(other._ec)
  {
    if (alive()) { std::construct_at(std::addressof(_value), std::move(other._value)); }
  }

  result& operator=(result const&)
    requires trivial_copy_assign
  = default;
  result& operator=(result const& other) noexcept(std::is_nothrow_copy_constructible_v<T> &&
                                                  std::is_nothrow_copy_assignable_v<T>)
    requires(!trivial_copy_assign && std::is_copy_constructible_v<T> &&
             std::is_copy_assignable_v<T>)
  {
    if (this == &other) { return *this; }
    assign(other);
    return *this;
  }

  result& operator=(result&&)
    requires trivial_move_assign
  = default;
  result& operator=(result&& other) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                             std::is_nothrow_move_assignable_v<T>)
    requires(!trivial_move_assign && std::is_move_constructible_v<T> &&
             std::is_move_assignable_v<T>)
  {
    assign(std::move(other));
    return *this;
  }

  ~result()
    requires trivial_dtor
  = default;
  ~result()
    requires(!trivial_dtor)
  {
    if (alive()) { _value.~T(); }
  }

  [[nodiscard]] bool has_value() const noexcept { return !_ec; }
  explicit operator bool() const noexcept { return has_value(); }
  /// @return the error; a default (success) code when has_value().
  [[nodiscard]] std::error_code error() const noexcept { return _ec; }

  /// @throws std::system_error(error()) when !has_value().
  [[nodiscard]] T& value() &
  {
    throw_if_error();
    return _value;
  }
  /// @throws std::system_error(error()) when !has_value().
  [[nodiscard]] T const& value() const&
  {
    throw_if_error();
    return _value;
  }
  /// @throws std::system_error(error()) when !has_value().
  [[nodiscard]] T&& value() &&
  {
    throw_if_error();
    return std::move(_value);
  }

  /// @return the value when has_value(), else @p fallback.
  [[nodiscard]] T value_or(T fallback) const& noexcept(std::is_nothrow_copy_constructible_v<T> &&
                                                       std::is_nothrow_move_constructible_v<T>)
  {
    if (has_value()) { return _value; }
    return fallback;
  }
  /// @return the value (moved out) when has_value(), else @p fallback.
  [[nodiscard]] T value_or(T fallback) && noexcept(std::is_nothrow_move_constructible_v<T>)
  {
    if (has_value()) { return std::move(_value); }
    return fallback;
  }

  /// @pre has_value()
  [[nodiscard]] T* operator->() noexcept { return std::addressof(_value); }
  /// @pre has_value()
  [[nodiscard]] T const* operator->() const noexcept { return std::addressof(_value); }
  /// @pre has_value()
  [[nodiscard]] T& operator*() noexcept { return _value; }
  /// @pre has_value()
  [[nodiscard]] T const& operator*() const noexcept { return _value; }

 private:
  [[nodiscard]] bool alive() const noexcept { return always_alive || !_ec; }

  void throw_if_error() const
  {
    if (_ec) { throw std::system_error(_ec); }
  }

  template <class R>
  void assign(R&& other)
  {
    // Read before `other._value` is forwarded (only that member is moved from; this keeps the
    // order of the reads independent of it).
    std::error_code const other_ec = other._ec;
    if constexpr (always_alive) {
      // Both slots are alive; an error-state slot holds a value-initialised T. If the assignment
      // throws, _ec is unchanged and *this stays valid.
      _value = std::forward<R>(other)._value;
      _ec    = other_ec;
    } else if (!_ec) {
      if (!other_ec) {
        _value = std::forward<R>(other)._value;
      } else {
        _value.~T();
        _ec = other_ec;
      }
    } else if (!other_ec) {
      // If the construction throws, *this keeps its error (still a valid state).
      std::construct_at(std::addressof(_value), std::forward<R>(other)._value);
      _ec = other_ec;
    } else {
      _ec = other_ec;
    }
  }

  std::error_code _ec;
  union {
    T _value;
  };
};

/// @brief Success-or-error for operations without a value. Default constructed = success.
template <>
class result<void> {
 public:
  using value_type = void;

  result() noexcept = default;
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design (`return ec;`)
  result(std::error_code ec) noexcept : _ec(ec) {}
  /// Convenience: `return errc::x;` from a function returning result<void>.
  // NOLINTNEXTLINE(google-explicit-constructor): implicit by design (`return errc::x;`)
  result(errc e) noexcept : _ec(make_error_code(e)) {}

  [[nodiscard]] bool has_value() const noexcept { return !_ec; }
  explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] std::error_code error() const noexcept { return _ec; }
  /// @throws std::system_error(error()) when !has_value().
  void value() const
  {
    if (_ec) { throw std::system_error(_ec); }
  }

 private:
  std::error_code _ec;
};

}  // namespace cucascade::io::rest

namespace std {
template <>
struct is_error_code_enum<cucascade::io::rest::errc> : true_type {};
}  // namespace std
