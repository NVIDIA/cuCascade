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

/// @file streams.hpp
/// @brief Receive-result vocabulary shared by the transport and the user-space TLS stream
/// (`tls_stream`), the stream options, and the kTLS receive helpers.  The transport
/// (`uring_transport`) prepares plain-TCP and kTLS SQEs itself; only the user-space record layer
/// needs a stream object (it owns a ciphertext ring and a send FIFO).

#pragma once

#include "io/rest/net/common.hpp"

#include <liburing.h>
#include <linux/tls.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <span>

#ifndef SOL_TLS
#define SOL_TLS 282  // include/linux/socket.h
#endif

namespace cucascade::io::net {

/// What one receive CQE means for the caller.
enum class rx_kind : std::uint8_t {
  data,     ///< `bytes` of payload were placed into the caller's window (may be short)
  eof,      ///< orderly end of stream (res == 0, or a close_notify)
  control,  ///< kTLS: a non-data record arrived (see @ref ktls_record_type)
  again,    ///< nothing placed, re-issue the receive (EAGAIN/EINTR, or more ciphertext needed)
  error,    ///< `err` holds the positive errno
};

struct rx_result {
  std::size_t bytes{0};
  rx_kind kind{rx_kind::error};
  int err{0};
};

/// Per-connection receive options.
struct stream_options {
  bool waitall{false};  ///< bounded MSG_WAITALL body receives (config::recv_waitall)
  /// tls::config::fallback_rx == record_layer: user-space TLS connections receive through the
  /// built-in record layer (`tls_stream` tier 2).
  bool rx_record_layer{true};
  /// TLS records sealed per ciphertext SEND (1..`tls_stream::max_tx_batch_records`).  A request
  /// body goes out in batches of this many 16 KiB records: 16 cuts a 16 MiB part to ~64 SEND
  /// round trips instead of ~340 with 3 (each SEND waits for the previous one: record order).
  unsigned tx_batch_records{16};
};

/// MSG_NOSIGNAL on every send (SIGPIPE is never ignored process-wide).
inline constexpr int send_msg_flags = MSG_NOSIGNAL;

/// The control-message storage of a kTLS receive: one `TLS_GET_RECORD_TYPE` byte.
struct ktls_cmsg_buffer {
  alignas(alignof(cmsghdr)) unsigned char bytes[CMSG_SPACE(sizeof(unsigned char))]{};
};

/// Result of interpreting a kTLS `recvmsg` with a `TLS_GET_RECORD_TYPE` control buffer.
struct ktls_recv {
  rx_kind kind{rx_kind::error};
  std::size_t bytes{0};
  std::uint8_t type{0};  ///< record content type (23 = application data); 0 = no cmsg
  int err{0};
};

/**
 * @brief Interpret the CQE of a kTLS `recvmsg` issued with a `TLS_GET_RECORD_TYPE` cmsg buffer.
 *
 * The kernel never mixes record types in one such receive: either data records (type 23, or
 * no cmsg at all when nothing was decrypted) or exactly one non-data record (handshake: a
 * NewSessionTicket or KeyUpdate; alert), whose plaintext was written into the receive iovecs.
 *
 * @param res   the CQE result
 * @param msg   the msghdr the kernel wrote back (msg_flags, msg_controllen, cmsg)
 * @return data (`bytes`), control (`type`, `bytes` of record plaintext in the iovecs), eof,
 *         again, or error (`EPROTO` for a truncated cmsg, else the errno)
 */
[[nodiscard]] inline ktls_recv interpret_ktls_recv(int res, msghdr const& msg) noexcept
{
  ktls_recv out{};
  if (res < 0) {
    if (res == -EAGAIN || res == -EINTR) {
      out.kind = rx_kind::again;
      return out;
    }
    out.kind = rx_kind::error;
    out.err  = -res;
    return out;
  }
  if ((msg.msg_flags & MSG_CTRUNC) != 0) {
    out.kind = rx_kind::error;
    out.err  = EPROTO;
    return out;
  }
  for (cmsghdr const* c = CMSG_FIRSTHDR(&msg); c != nullptr;
       c                = CMSG_NXTHDR(const_cast<msghdr*>(&msg), const_cast<cmsghdr*>(c))) {
    if (c->cmsg_level == SOL_TLS && c->cmsg_type == TLS_GET_RECORD_TYPE &&
        c->cmsg_len >= CMSG_LEN(sizeof(unsigned char))) {
      out.type = *CMSG_DATA(c);
      break;
    }
  }
  out.bytes = static_cast<std::size_t>(res);
  if (out.type != 0 && out.type != 23) {
    out.kind = rx_kind::control;
    return out;
  }
  out.kind = res == 0 ? rx_kind::eof : rx_kind::data;
  return out;
}

}  // namespace cucascade::io::net
