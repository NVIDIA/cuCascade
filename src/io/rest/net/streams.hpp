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
/// @brief Connection streams (API.md §B.6): the `completion_stream` concept, `plain_stream`
/// (TCP) and `ktls_stream` (kernel TLS RX+TX, zero user-space TLS state on the data path), the
/// kTLS control-record drain, and the extension points the user-space fallback (`tls_stream`,
/// gpuio T16) plugs into. Streams only PREPARE SQEs and INTERPRET results; the connection pool owns
/// the op slots, the cursors and the state machine (`connection_pool.hpp`), and the HTTP driver
/// (`http1_exchange`) sits on top of the pool's whole-buffer operations.
///
/// Design: gpuio ARCHITECTURE §4.1 (layering, no virtual dispatch: the pool switches once per CQE
/// on the slot's kind byte, then calls the stream statically), §4.2 (receive strategy and the three
/// MSG_WAITALL rules), §4.3 (decryption placement), §3.5 (GET flow). Evidence:
/// gpuio docs/research/experiments/E-KTLS-1_io_uring_recv_waitall.md, io_uring_knobs §2.5.5/§2.5.6,
/// tls.md §2.5, §2.12.
///
/// ## The three WAITALL rules (E-KTLS-1 §3, §6), enforced here and in the pool
///  1. Every body receive is `min(remaining Content-Length, body_recv_cap)`; never posted before
///     the length is known (an over-long WAITALL receive completes only when the peer closes:
///     75 s on nginx, ~5 s on S3). The pool computes the length; `prep_recv` asserts nothing larger
///     than the caller's window is requested.
///  2. Never `IOSQE_IO_LINK` an SQE after a WAITALL receive: kTLS sets MSG_CTRUNC on every
///     cmsg-less receive, io_uring then `req_set_fail`s the receive even with the full result and
///     a linked SQE completes with -ECANCELED (64/64 measured). `body_recv_sqe_flags` is the only
///     flag set a body receive ever carries (static_assert below); deadlines live in the reactor
///     heap instead of linked timeouts.
///  3. A short positive WAITALL CQE is normal progress (a control record follows): the cursor
///     advances by `res`; the next receive returns -EIO, which is the cmsg drain path.

#pragma once

#include "io/rest/net/common.hpp"

#include <liburing.h>
#include <linux/tls.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <cerrno>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <system_error>
#include <type_traits>

#ifndef SOL_TLS
#define SOL_TLS 282  // include/linux/socket.h
#endif

namespace cucascade::io::net {

// ------------------------------------------------------------------------------------------------
// Results
// ------------------------------------------------------------------------------------------------

/// What one receive CQE means for the caller (API.md §B.6).
enum class rx_kind : std::uint8_t {
  data,     ///< `bytes` of payload were placed into the caller's window (may be short: rule 3)
  eof,      ///< orderly end of stream (res == 0, or a replayed close_notify)
  control,  ///< kTLS: a non-data record is queued (-EIO); drain it with `prep_drain`
  again,    ///< nothing placed, re-issue the same receive (EAGAIN/EINTR, or gpuio T16: need more
            ///< input)
  error,    ///< `err` holds the positive errno
};

struct rx_result {
  std::size_t bytes{0};
  rx_kind kind{rx_kind::error};
  int err{0};
};

/// Result of one control-record drain (`RECVMSG` with a `TLS_GET_RECORD_TYPE` cmsg).
struct drain_result {
  std::size_t bytes{0};   ///< plaintext bytes written into the drain buffer
  std::uint8_t type{0};   ///< record content type (21 alert, 22 handshake, 23 data); 0 = none
  bool eor{false};        ///< MSG_EOR: the record is complete
  bool truncated{false};  ///< MSG_CTRUNC: no room for the cmsg (protocol error)
  int err{0};             ///< positive errno when the drain failed; 0 otherwise
};

// ------------------------------------------------------------------------------------------------
// Kernel-visible message storage
// ------------------------------------------------------------------------------------------------

/// `msghdr`s and the cmsg buffer a stream prepares into. One per connection slot (never on a
/// stream: `ktls_stream` stays <= 64 B). The kernel imports the iovecs at submission
/// (IORING_FEAT_SUBMIT_STABLE) but writes `msg_flags` / `msg_controllen` / the cmsg back at
/// completion (`____sys_recvmsg`), so this storage must live until the CQE.
struct stream_msg {
  msghdr rx{};        ///< RECVMSG: scattered body window, or the control drain
  msghdr tx{};        ///< SENDMSG: request head + body iovecs
  iovec drain_iov{};  ///< the drain target (a region of the slot's header buffer)
  alignas(alignof(cmsghdr)) unsigned char cmsg[CMSG_SPACE(sizeof(unsigned char))]{};
};

/// Per-endpoint receive options a stream is constructed with.
struct stream_options {
  bool waitall{true};  ///< connection_config::recv_waitall (E-KTLS-1: default true)
  bool iowq{false};    ///< connection_config::tls_recv_placement == iowq: IOSQE_ASYNC on bodies
  /// tls_config::fallback_rx == record_layer: user-space TLS connections receive through the
  /// own record layer (`tls_stream` tier 2, gpuio T19). Ignored by plain and kTLS streams.
  bool rx_record_layer{false};
};

/// The only SQE flags a body receive carries (rule 2: never IOSQE_IO_LINK / IO_HARDLINK).
[[nodiscard]] constexpr std::uint8_t body_recv_sqe_flags(bool iowq) noexcept
{
  return iowq ? static_cast<std::uint8_t>(IOSQE_ASYNC) : std::uint8_t{0};
}
static_assert((body_recv_sqe_flags(false) & (IOSQE_IO_LINK | IOSQE_IO_HARDLINK)) == 0);
static_assert((body_recv_sqe_flags(true) & (IOSQE_IO_LINK | IOSQE_IO_HARDLINK)) == 0);

/// recv/send flags: MSG_NOSIGNAL on every SEND/SENDMSG (SIGPIPE is never ignored process-wide,
/// gpuio ARCHITECTURE §4.4); MSG_WAITALL only on body receives.
inline constexpr int send_msg_flags = MSG_NOSIGNAL;
[[nodiscard]] constexpr int body_recv_msg_flags(bool waitall) noexcept
{
  return waitall ? MSG_WAITALL : 0;
}

// ------------------------------------------------------------------------------------------------
// Concepts
// ------------------------------------------------------------------------------------------------

/// The stream concept (API.md §B.6): layers prepare SQEs; the HTTP driver is written once.
/// `prep_recv` is the BODY receive (WAITALL per options, capped by the caller, never linked);
/// `on_recv` interprets any receive CQE of the stream (header or body) for the window `rx`.
template <class S>
concept completion_stream = requires(S& s,
                                     io_uring_sqe* sqe,
                                     std::span<iovec const> rx,
                                     std::span<iovec const> tx,
                                     int res,
                                     unsigned flags) {
  { s.prep_send(sqe, tx) } -> std::same_as<void>;
  { s.prep_recv(sqe, rx) } -> std::same_as<void>;
  { s.on_recv(res, flags, rx) } -> std::same_as<rx_result>;
  { s.fd() } -> std::same_as<int>;
};

/// What the connection pool needs beyond B.6's minimum: a one-shot header receive (no WAITALL,
/// optional IORING_RECVSEND_POLL_FIRST right after a send) and the bytes-received counter.
/// Satisfied by `plain_stream`, `ktls_stream` and (gpuio T16) `tls_stream`.
template <class S>
concept pool_stream = completion_stream<S> &&
                      requires(S& s, S const& cs, io_uring_sqe* sqe, std::span<char> hb, bool pf) {
                        { s.prep_recv_header(sqe, hb, pf) } -> std::same_as<void>;
                        { cs.bytes_rx() } -> std::convertible_to<std::uint64_t>;
                      };

/// kTLS only: drain one queued control record into a buffer the caller owns.
template <class S>
concept control_drain_stream =
  pool_stream<S> && requires(S& s, S const& cs, io_uring_sqe* sqe, std::span<char> buf, int res) {
    { s.prep_drain(sqe, buf) } -> std::same_as<void>;
    { cs.on_drain(res) } -> std::same_as<drain_result>;
  };

// ------------------------------------------------------------------------------------------------
// socket_stream: the common CRTP base of plain_stream and ktls_stream
// ------------------------------------------------------------------------------------------------

/// CRTP base: identical SEND/SENDMSG and RECV/RECVMSG preparation for plain TCP and kTLS (the
/// kernel encrypts and decrypts transparently). `Derived::on_recv` differs only in -EIO.
template <class Derived>
class socket_stream {
 public:
  socket_stream() noexcept = default;
  socket_stream(int fd, stream_msg* msg, stream_options opts) noexcept
    : _fd(fd), _waitall(opts.waitall), _iowq(opts.iowq), _msg(msg)
  {
  }

  [[nodiscard]] int fd() const noexcept { return _fd; }
  [[nodiscard]] std::uint64_t bytes_rx() const noexcept { return _bytes_rx; }
  [[nodiscard]] bool waitall() const noexcept { return _waitall; }
  [[nodiscard]] bool iowq() const noexcept { return _iowq; }

  /// @brief Whole-buffer send of @p tx (head, or head + body iovecs). One iovec: SEND, else
  /// SENDMSG over `stream_msg::tx`. Always MSG_NOSIGNAL; never SEND_ZC (kTLS rejects it,
  /// io_uring_knobs §2.5.6). @p tx must stay valid until the CQE; a short send is continued by
  /// the pool with the remainder.
  void prep_send(io_uring_sqe* sqe, std::span<iovec const> tx) noexcept
  {
    if (tx.size() == 1) {
      io_uring_prep_send(sqe, _fd, tx[0].iov_base, tx[0].iov_len, send_msg_flags);
    } else {
      _msg->tx            = msghdr{};
      _msg->tx.msg_iov    = const_cast<iovec*>(tx.data());
      _msg->tx.msg_iovlen = tx.size();
      io_uring_prep_sendmsg(sqe, _fd, &_msg->tx, static_cast<unsigned>(send_msg_flags));
    }
  }

  /// @brief Body receive into the window @p rx: RECV for one iovec (the fastest inline op,
  /// io_uring_knobs §2.5.5), else RECVMSG with the iovecs and NO msg_control (with a cmsg buffer
  /// WAITALL is silently ignored, net.c:1064). MSG_WAITALL per options; IOSQE_ASYNC when the
  /// endpoint decrypts in io-wq. The caller caps the window to min(remaining, cap) (rule 1).
  void prep_recv(io_uring_sqe* sqe, std::span<iovec const> rx) noexcept
  {
    prep_body(sqe, rx, _waitall);
  }

  /// @brief Body receive WITHOUT MSG_WAITALL (one-shot), whatever the options: used once after
  /// the WAITALL watchdog cancelled a receive (a pending TLS 1.3 KeyUpdate masks POLLIN on
  /// kernels >= 6.14, so io_uring's internal WAITALL retry would never run; the first attempt of
  /// a one-shot receive is not poll-gated and returns -EIO for the queued record).
  void prep_recv_once(io_uring_sqe* sqe, std::span<iovec const> rx) noexcept
  {
    prep_body(sqe, rx, false);
  }

 private:
  void prep_body(io_uring_sqe* sqe, std::span<iovec const> rx, bool waitall) noexcept
  {
    CUCASCADE_NET_ASSERT(!rx.empty());
    int const fl = body_recv_msg_flags(waitall);
    if (rx.size() == 1) {
      io_uring_prep_recv(sqe, _fd, rx[0].iov_base, rx[0].iov_len, fl);
    } else {
      _msg->rx            = msghdr{};
      _msg->rx.msg_iov    = const_cast<iovec*>(rx.data());
      _msg->rx.msg_iovlen = rx.size();
      io_uring_prep_recvmsg(sqe, _fd, &_msg->rx, static_cast<unsigned>(fl));
    }
    sqe->flags = body_recv_sqe_flags(_iowq);  // rule 2: never linked (static check above)
    CUCASCADE_NET_ASSERT((sqe->flags & (IOSQE_IO_LINK | IOSQE_IO_HARDLINK)) == 0);
  }

 public:
  /// @brief One-shot header receive into @p buf (no WAITALL: the head length is unknown).
  /// @param poll_first IORING_RECVSEND_POLL_FIRST: the receive follows a send on a keep-alive
  /// connection, so the first non-blocking attempt would certainly return -EAGAIN (§4.2 table).
  void prep_recv_header(io_uring_sqe* sqe, std::span<char> buf, bool poll_first) noexcept
  {
    io_uring_prep_recv(sqe, _fd, buf.data(), buf.size(), 0);
    if (poll_first) sqe->ioprio |= IORING_RECVSEND_POLL_FIRST;
  }

 protected:
  /// Shared CQE interpretation; `eio_is_control` = kTLS.
  rx_result interpret(int res, bool eio_is_control) noexcept
  {
    if (res > 0) {
      _bytes_rx += static_cast<std::uint64_t>(res);
      return {static_cast<std::size_t>(res), rx_kind::data, 0};
    }
    if (res == 0) return {0, rx_kind::eof, 0};
    if (res == -EIO && eio_is_control) return {0, rx_kind::control, 0};
    if (res == -EAGAIN || res == -EINTR) return {0, rx_kind::again, 0};
    return {0, rx_kind::error, -res};
  }

  int _fd{-1};
  bool _waitall{true};
  bool _iowq{false};
  stream_msg* _msg{nullptr};
  std::uint64_t _bytes_rx{0};
};

/// Plain TCP (http://). -EIO is an ordinary error.
class plain_stream : public socket_stream<plain_stream> {
 public:
  using socket_stream::socket_stream;
  /// @return the meaning of a receive CQE (header or body).
  rx_result on_recv(int res, unsigned flags, std::span<iovec const> rx) noexcept
  {
    (void)flags;
    (void)rx;
    return interpret(res, false);
  }
};

/// Kernel TLS, RX and TX engaged (both directions installed from the session's exported keys by
/// `tls::ktls::install`). Holds no ring buffer, no AEAD state, no TLS library object and no
/// scratch: <= 64 B (gpuio ARCHITECTURE §4.1). The TLS 1.3 re-key state lives in the slot
/// (`conn_slot::rekey`), never here.
class ktls_stream : public socket_stream<ktls_stream> {
 public:
  using socket_stream::socket_stream;

  /// @return data (`res` > 0, possibly short: rule 3), eof (0), control (-EIO: a non-data
  /// record is queued; `tls_sw.c:1849-1858`), again, or error (e.g. EKEYEXPIRED, ECONNRESET).
  rx_result on_recv(int res, unsigned flags, std::span<iovec const> rx) noexcept
  {
    (void)flags;
    (void)rx;
    return interpret(res, true);
  }

  /// @brief Control-record drain: RECVMSG into @p buf with a TLS_GET_RECORD_TYPE cmsg and NO
  /// WAITALL (ignored with a cmsg anyway). The record stays queued until drained; a record larger
  /// than @p buf arrives in pieces (MSG_EOR on the last). @p buf must stay valid until the CQE.
  void prep_drain(io_uring_sqe* sqe, std::span<char> buf) noexcept
  {
    std::memset(_msg->cmsg, 0, sizeof _msg->cmsg);
    _msg->drain_iov         = iovec{buf.data(), buf.size()};
    _msg->rx                = msghdr{};
    _msg->rx.msg_iov        = &_msg->drain_iov;
    _msg->rx.msg_iovlen     = 1;
    _msg->rx.msg_control    = _msg->cmsg;
    _msg->rx.msg_controllen = sizeof _msg->cmsg;
    io_uring_prep_recvmsg(sqe, _fd, &_msg->rx, 0);
  }

  /// @return the record type, size and completeness of a drain CQE.
  [[nodiscard]] drain_result on_drain(int res) const noexcept
  {
    drain_result d{};
    if (res < 0) {
      d.err = -res;
      return d;
    }
    d.bytes     = static_cast<std::size_t>(res);
    d.eor       = (_msg->rx.msg_flags & MSG_EOR) != 0;
    d.truncated = (_msg->rx.msg_flags & MSG_CTRUNC) != 0;
    for (cmsghdr* c = CMSG_FIRSTHDR(&_msg->rx); c != nullptr; c = CMSG_NXTHDR(&_msg->rx, c)) {
      if (c->cmsg_level == SOL_TLS && c->cmsg_type == TLS_GET_RECORD_TYPE) {
        d.type = *CMSG_DATA(c);
        break;
      }
    }
    return d;
  }
};

// gpuio ARCHITECTURE §4.1: the kTLS path carries no user-space TLS state.
static_assert(sizeof(ktls_stream) <= 64, "ktls_stream must stay <= 64 B");
static_assert(sizeof(plain_stream) <= 64);
static_assert(pool_stream<plain_stream>);
static_assert(control_drain_stream<ktls_stream>);
static_assert(std::is_trivially_copyable_v<ktls_stream>);
static_assert(std::is_trivially_copyable_v<plain_stream>);

// ------------------------------------------------------------------------------------------------
// TLS extension point (the user-space `tls_stream` and the kTLS key installation)
// ------------------------------------------------------------------------------------------------

/// The pool's TLS parameter for builds without TLS: `https://` endpoints fail their connects with
/// errc::not_supported, and no TLS symbol is referenced.
struct no_tls {};

/// What `connection_pool<Env, Handler, Tls>` needs from its user-space TLS stream (heap state per
/// user-space connection, allocated at handshake time only, never per exchange): `pool_stream`
/// ops (its `on_recv` decrypts into `rx` and returns `again` when it needs more ciphertext;
/// control records are consumed internally; an optional `int on_send(int res)` maps a SEND CQE of
/// ciphertext to the plaintext bytes of `tx` it completed, or -errno) plus
/// `static void destroy(Fallback*) noexcept`.
template <class F>
concept fallback_stream = pool_stream<F> && requires(F* p) {
  { F::destroy(p) } -> std::same_as<void>;
};

}  // namespace cucascade::io::net
