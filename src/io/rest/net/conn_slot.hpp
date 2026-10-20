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

/// @file conn_slot.hpp
/// @brief One pooled connection (API.md §B.6 `conn_slot`): socket, TLS session, stream kind, the
/// 16 KiB header buffer (response head, control-record drain target), the request-head buffer,
/// the HTTP parser state the exchange driver uses, the destination cursor, and the pool's
/// private bookkeeping (op tokens, cursors, idle generation). Fixed size, allocated once per
/// endpoint table (`connection_pool`); nothing in it allocates.
///
/// Lifecycle (`conn_state`): free -> [resolving] -> connecting -> [handshaking] -> idle <-> busy
/// -> closing -> free; idle -> draining (TLS 1.3 tickets on an idle kTLS socket) -> idle;
/// idle -> probing (keep-warm HEAD) -> idle. `kind` is the B.6 byte: the TLS setup's result
/// (plain 0, ktls 1, user-space tls 2) and `kind_connecting` 3 / `kind_handshaking` 4 while the
/// connection is being set up (gpuio CHANGE_REQUESTS "gpuio T7 — slot kind numbering").
///
/// TLS objects (docs/io/rest-uring-design.md section 4.2): the `tls::session` (control plane:
/// handshake, key export, and on user-space connections the library's sealer) and, on TLS 1.3
/// kTLS connections, the `tls::ktls::rekey_state` live here, outside the <= 64 B `ktls_stream`;
/// the user-space stream (`tls`) owns its record opener. All are freed before the socket closes.
///
/// Memory lifetime (gpuio ARCHITECTURE §5): a slot is reused only after every CQE of every op that
/// targets its buffers has been reaped (`closing` waits for them); the idle watch (POLL_ADD) does
/// not target slot memory and is never cancelled (its stale CQE is dropped by generation).

#pragma once

#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/response_parser.hpp"
#include "io/rest/net/common.hpp"
#include "io/rest/net/deadline_heap.hpp"
#include "io/rest/net/streams.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace cucascade::io::tls::ktls {
class rekey_state;  // src/io/tls/ktls.hpp
}  // namespace cucascade::io::tls::ktls

namespace cucascade::io::net {

// ------------------------------------------------------------------------------------------------
// Kinds and states
// ------------------------------------------------------------------------------------------------

/// B.6 kind byte. 0..2 are the TLS setup's outcome (or plain TCP).
inline constexpr std::uint8_t kind_plain       = 0;
inline constexpr std::uint8_t kind_ktls        = 1;
inline constexpr std::uint8_t kind_tls         = 2;  ///< user-space TLS (`tls_stream`)
inline constexpr std::uint8_t kind_connecting  = 3;
inline constexpr std::uint8_t kind_handshaking = 4;
inline constexpr std::uint8_t kind_fail        = 0xff;
static_assert(kind_plain == 0 && kind_ktls == 1 && kind_tls == 2 && kind_fail == 0xff);

enum class conn_state : std::uint8_t {
  free,         ///< no socket; on the endpoint's free list
  resolving,    ///< waiting for the DNS resolver's first answer
  connecting,   ///< IORING_OP_SOCKET / IORING_OP_CONNECT in flight
  handshaking,  ///< TLS handshake (POLL_ADD readiness between `session::handshake` steps)
  idle,         ///< pooled: on the idle LIFO, idle watch armed
  busy,         ///< checked out by an exchange (send / recv_header / recv_body)
  draining,     ///< idle kTLS socket readable: draining a control record (tickets)
  probing,      ///< keep-warm probe exchange (HEAD), background ops
  closing,      ///< socket shut down; waiting for the last CQE before the slot is freed
};

/// Op token value meaning "none".
inline constexpr std::uint32_t no_op = ~0u;
/// Intrusive list terminator.
inline constexpr std::uint32_t no_conn = ~0u;

/// 16 KiB: one TLS record of plaintext; S3 heads are ~0.5 KiB, x-amz-meta-* can reach ~2 KiB
/// (tls.md §2.5.3; gpuio ARCHITECTURE §11 `header_buffer_bytes`). Also the control-drain target.
inline constexpr std::size_t conn_header_bytes = 16 * KiB;
/// Request-head storage (GET head with SigV4 and a ~1 KiB session token is ~1.8 KiB). Separate
/// from the header buffer so the head can never be overwritten by the response while a short
/// send is being continued.
inline constexpr std::size_t conn_tx_bytes = 4 * KiB;
/// Destination iovecs per scattered body receive (RECVMSG window, gpuio ARCHITECTURE §4.2): a wider
/// destination is received in several WAITALL receives (each still capped by remaining).
inline constexpr std::size_t conn_rx_iovecs = 64;
/// Request iovecs per SENDMSG window (head + PUT body pieces).
inline constexpr std::size_t conn_tx_iovecs = 16;

/// What the slot's `io` op is doing (pool-private).
enum class io_kind : std::uint8_t {
  none,
  socket,        ///< IORING_OP_SOCKET
  connect,       ///< IORING_OP_CONNECT
  hs_poll,       ///< POLL_ADD between handshake steps
  header,        ///< one-shot header RECV
  body,          ///< RECV|WAITALL / RECVMSG|WAITALL into the sink window
  drain_header,  ///< control drain in the header phase (target: header_buf + hdr_len)
  drain_body,    ///< control drain in the body phase (target: header_buf)
  drain_idle,    ///< control drain on an idle socket (tickets after the handshake)
  flush_poll,    ///< POLLOUT before flush_key_update (KeyUpdate reply blocked)
};

// ------------------------------------------------------------------------------------------------
// The slot
// ------------------------------------------------------------------------------------------------

struct conn_slot {
  // ---- identity and lifecycle (pool-maintained; read-only for the exchange driver) ----
  std::uint8_t kind{kind_fail};  ///< B.6 kind byte (see above)
  conn_state state{conn_state::free};
  priority prio{priority::normal};  ///< priority of the current checkout
  bool background{false};           ///< current setup/probe ops do not keep run() alive
  int fd{-1};
  /// TLS control plane (TLS slots, from the handshake on; owned, freed before the socket).
  std::unique_ptr<tls::session> session{};
  /// TLS 1.3 kTLS on a re-key-capable kernel: what a KeyUpdate needs (owned; released through
  /// the pool's TLS parameter before the socket closes).
  tls::ktls::rekey_state* rekey{nullptr};
  plain_stream plain{};       ///< valid when kind == kind_plain
  ktls_stream ktls{};         ///< valid when kind == kind_ktls
  void* tls{nullptr};         ///< kind == kind_tls: the user-space `tls_stream`, owned by the pool
  std::uint32_t endpoint{0};  ///< endpoint id (remote_context)
  std::uint32_t index{0};     ///< position in the endpoint's slot table (constant)
  std::uint32_t generation{0};  ///< bumped per idle period, checkout and close (watch tags)
  std::uint64_t created_ns{0};  ///< connected (reactor monotonic clock)
  std::uint64_t idle_since_ns{0};
  std::uint64_t bytes_on_connection{0};  ///< received + sent on this connection
  std::uint16_t requests_on_connection{0};

  // ---- current exchange (pool-maintained, read by the exchange driver) ----
  bool reused{false};      ///< the connection served a request before this checkout
  bool first_byte{false};  ///< a response byte arrived in this exchange (stale rule)
  bool last_request{
    false};  ///< byte cap / age reached: send `Connection: close`, closed at check_in
  bool close_after{false};   ///< set by the driver (Connection: close, error): close at check_in
  std::uint32_t hdr_len{0};  ///< bytes of `header_buf` filled by recv_header (the driver may
                             ///< compact: set it after moving the unconsumed tail to the front)
  std::uint64_t exchange_rx{0}, exchange_tx{0};  ///< payload bytes of this exchange

  // ---- exchange owner (the REST worker): opaque to the pool ----
  void* owner{nullptr};
  std::uint64_t owner_tag{0};

  // ---- HTTP state for the exchange driver (B.6) ----
  http::response_parser parser{};
  http::response_meta meta{};
  http::iovec_sink sink{};  ///< body destination cursor (recv_body receives into it)

  // ---- pool-private bookkeeping ----
  struct private_state {
    std::uint32_t io_op{no_op};     ///< connect / handshake / receive token (busy: held)
    std::uint32_t tx_op{no_op};     ///< send token (busy: held)
    std::uint32_t watch_op{no_op};  ///< the current idle watch's token (in flight)
    std::uint32_t next{no_conn};    ///< free / idle list link
    void* pool{nullptr};            ///< owning pool (deadline callbacks)
    io_kind io{io_kind::none};
    bool io_inflight{false}, tx_inflight{false};
    bool cancel_requested{false};
    bool reply_pending{false};    ///< KeyUpdate reply owed while a send was in flight
    bool tx_poll_first{false};    ///< header receive after a send: POLL_FIRST
    bool rx_active{false};        ///< recv_header/recv_body called, completion not delivered yet
    bool rx_waitall{false};       ///< the body receive in flight carries MSG_WAITALL
    bool watchdog{false};         ///< the WAITALL watchdog cancelled the receive in flight
    bool wd_armed{false};         ///< the watchdog sampled this receive (gpuio T13b baseline below)
    std::uint8_t wd_quiet{0};     ///< ticks since the baseline with both counters unchanged
    bool oneshot_next{false};     ///< the next body receive is one-shot (after the watchdog)
    bool bound_dns{false};        ///< peer came from the resolver (bench on failure)
    std::uint8_t drain_type{0};   ///< type of the record being accumulated
    std::uint32_t drain_len{0};   ///< control-record bytes accumulated so far
    std::uint32_t replay_off{0};  ///< offset of the drained record in header_buf
    std::uint8_t control_run{0};  ///< kTLS control records in a row (cap 32)
    std::uint32_t parse_off{0};   ///< keep-warm probe: header bytes consumed by the parser
    std::uint32_t rx_iovcnt{0};   ///< iovecs of the body window in flight
    std::uint64_t rx_len{0};      ///< length requested by the receive in flight
    std::uint64_t rx_remaining{0};  ///< HTTP body bytes outstanding when recv_body was called
    http::iovec_sink tx_cursor{};
    std::uint64_t tx_total{0}, tx_done{0};
    deadline_handle deadline{};  ///< connect / TLS deadline
    std::uint64_t deadline_ns{0};
    std::uint64_t op_started_ns{0};
    std::uint64_t wd_since_ns{0};  ///< watchdog baseline: tick time
    std::uint64_t wd_rx_bytes{0};  ///< watchdog baseline: TCP_INFO tcpi_bytes_received
    int wd_inq{0};                 ///< watchdog baseline: FIONREAD
    int last_errno{0};
  } pv{};

  // ---- kernel-visible storage ----
  stream_msg msg{};
  sockaddr_storage peer{};
  socklen_t peer_len{0};
  std::array<iovec, conn_rx_iovecs> rx_window{};
  std::array<iovec, conn_tx_iovecs> tx_window{};
  /// Response head (recv_header appends at hdr_len) and control-record drain target.
  alignas(64) std::array<char, conn_header_bytes> header_buf{};
  /// Request head storage (the driver writes the head here, then `send`s it).
  alignas(64) std::array<char, conn_tx_bytes> tx_buf{};

  /// @return the header bytes received so far.
  [[nodiscard]] std::span<char const> header() const noexcept
  {
    return {header_buf.data(), hdr_len};
  }
  [[nodiscard]] bool is_tls() const noexcept { return kind == kind_ktls || kind == kind_tls; }
};

}  // namespace cucascade::io::net
