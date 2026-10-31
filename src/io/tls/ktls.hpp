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

#include <cucascade/io/tls/tls.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <system_error>

/**
 * @file ktls.hpp
 * @brief Kernel TLS: cuCascade installs the keys itself and handles the control records.
 *
 * gpuio let OpenSSL install kTLS keys through its socket BIO and fed drained control records back
 * into OpenSSL (replay BIO).  That needs an OpenSSL built with kTLS, which conda-forge's is not
 * (`OPENSSL_NO_KTLS`), and ties kTLS to one library.  Here the keys come from
 * `tls::session::export_keys`, and this module installs them and follows the connection's
 * control records itself (docs/io/rest-uring-design.md section 4.3).
 *
 * Socket I/O here is limited to `setsockopt` and, for a KeyUpdate reply, one non-blocking
 * `sendmsg` of a 5-byte handshake record.  Data receives and sends stay with the transport.
 * Everything is reactor-thread confined and allocation-free except @ref make_rekey_state.
 */
namespace cucascade::io::tls::ktls {

/// What the running kernel offers (probed once per process: `uname`, then the first install).
struct kernel_caps {
  bool tls13{false};        ///< TLS 1.3 kTLS (Linux >= 6.0 for RX)
  bool tls13_rekey{false};  ///< TLS 1.3 KeyUpdate re-key of kTLS (Linux >= 6.14)
};

/// @return the cached capabilities of the running kernel.
[[nodiscard]] kernel_caps probe() noexcept;

/// Test hook: pretend the kernel is @p major.@p minor (0.0 restores the real one).
void override_kernel_version_for_testing(unsigned major, unsigned minor) noexcept;

/// Outcome of @ref install.
struct install_result {
  bool rx{false};      ///< TLS_RX installed
  bool tx{false};      ///< TLS_TX installed
  bool no_pad{false};  ///< TLS_RX_EXPECT_NO_PAD accepted (TLS 1.3 only)
  /// Set when neither direction was installed (the socket's record protection is unchanged and
  /// user-space TLS may take over; the errno, or @c not_supported), or when TX was installed and
  /// RX refused (kernel TX state cannot be undone: the connection is unusable,
  /// @c errc::ktls_unavailable).
  std::error_code error{};
};

/**
 * @brief Move a connection's record protection into the kernel.
 *
 * Order: `TCP_ULP "tls"`, `TLS_TX`, `TLS_RX`, then `TLS_RX_EXPECT_NO_PAD` (TLS 1.3, when
 * @p expect_no_pad).  TX goes first so that a refusal leaves the socket usable for user-space
 * TLS.  Only AES-128-GCM and AES-256-GCM are installed; anything else returns
 * @c errc::not_supported without touching the socket.
 *
 * @param fd   connected socket after a finished handshake with nothing buffered by the library
 * @param rx   receive keys (server → client)
 * @param tx   send keys (client → server)
 */
[[nodiscard]] install_result install(int fd,
                                     traffic_keys const& rx,
                                     traffic_keys const& tx,
                                     bool expect_no_pad) noexcept;

/**
 * @brief TLS 1.3 state needed to follow KeyUpdates on a kTLS connection.
 *
 * Holds both application traffic secrets (wiped on destruction).  Allocated separately so the
 * transport's kTLS stream stays small; only TLS 1.3 connections on a re-key-capable kernel have
 * one.
 */
class rekey_state;

struct rekey_state_deleter {
  void operator()(rekey_state* state) const noexcept;
};
using rekey_state_ptr = std::unique_ptr<rekey_state, rekey_state_deleter>;

/**
 * @brief Keep what a later KeyUpdate needs.
 *
 * @param kdf  the provider's KDF; must outlive the state
 * @return null for TLS 1.2 (no KeyUpdate), or when the kernel cannot re-key (a KeyUpdate is then
 *         a protocol error and the connection is replaced)
 */
[[nodiscard]] rekey_state_ptr make_rekey_state(traffic_keys const& rx,
                                               traffic_keys const& tx,
                                               tls::kdf const& kdf) noexcept;

/// TLS record content types the kernel reports through `TLS_GET_RECORD_TYPE`.
inline constexpr std::uint8_t record_alert     = 21;
inline constexpr std::uint8_t record_handshake = 22;
inline constexpr std::uint8_t record_data      = 23;

/// What the transport does after @ref on_control_record.
enum class control_action : std::uint8_t {
  consumed,    ///< nothing to do (ticket skipped, KeyUpdate applied with no reply owed)
  reply_owed,  ///< a KeyUpdate reply must be sent: call @ref flush_reply (no send in flight)
  closed,      ///< close_notify: end of stream; the connection is not reusable
  failed,      ///< protocol error (@c errc::protocol_error); close the connection
};

/**
 * @brief Handle one record the kernel did not deliver as data.
 *
 * TLS 1.3 handshake records carry NewSessionTicket (skipped) and KeyUpdate (RX re-keyed with
 * `HKDF-Expand-Label(secret, "traffic upd")`; a reply is owed when the peer requested one).  A
 * TLS 1.2 HelloRequest is ignored.  Alerts: close_notify is @c closed; any other alert fails.
 * A malformed record, application data, a KeyUpdate without @p state, or a KeyUpdate that does
 * not end its record fail.  The transport caps control records in a row (32) itself: TLS 1.2
 * connections have no state to count in.
 *
 * @param fd         the kTLS socket (for the RX re-key)
 * @param state      the connection's re-key state, or null
 * @param type       content type from `TLS_GET_RECORD_TYPE`
 * @param plaintext  the record's plaintext
 */
[[nodiscard]] control_action on_control_record(int fd,
                                               rekey_state* state,
                                               std::uint8_t type,
                                               std::span<std::byte const> plaintext,
                                               std::error_code& ec) noexcept;

/**
 * @brief Send the owed KeyUpdate(update_not_requested) and re-key TX.
 *
 * @return {} when sent and re-keyed (or nothing was owed); @c std::errc::operation_would_block
 *         when the socket is full (arm POLLOUT and call again); any other error is fatal for the
 *         connection.
 */
[[nodiscard]] std::error_code flush_reply(int fd, rekey_state* state) noexcept;

}  // namespace cucascade::io::tls::ktls
