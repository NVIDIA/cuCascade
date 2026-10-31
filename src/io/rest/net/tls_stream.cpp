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

/// @file tls_stream.cpp
/// @brief The user-space TLS I/O shell (gpuio src/net/tls_stream.cpp minus OpenSSL: the ring,
/// the FIFO, the NOP rule, bounded WAITALL, check-in and idle drains, plaintext mapping of the
/// ciphertext SEND) over `tls::record_opener` / `tls::record_sealer`, and the pool's TLS seam
/// (`tls_support`: kTLS key installation and control records). Design: tls_stream.hpp;
/// docs/io/rest-uring-design.md sections 4.2 and 4.3.

#include "io/rest/net/tls_stream.hpp"

#include <cucascade/log/logging.hpp>

#include <pthread.h>
#include <signal.h>  // NOLINT(modernize-deprecated-headers): POSIX sigset_t / pthread_sigmask

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <new>
#include <string_view>
#include <utility>

namespace cucascade::io::net {

namespace {

/// SIGPIPE blocked on this thread while the TLS library may write to the socket itself (the
/// handshake). Never touch the process-wide disposition (gpuio T14).
class sigpipe_block {
 public:
  sigpipe_block() noexcept
  {
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGPIPE);
    _active = pthread_sigmask(SIG_BLOCK, &block, &_old) == 0 && sigismember(&_old, SIGPIPE) == 0;
  }
  ~sigpipe_block()
  {
    if (!_active) return;
    sigset_t pending;
    if (sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1) {
      sigset_t only;
      sigemptyset(&only);
      sigaddset(&only, SIGPIPE);
      timespec const zero{0, 0};
      (void)sigtimedwait(&only, nullptr, &zero);
    }
    pthread_sigmask(SIG_SETMASK, &_old, nullptr);
  }
  sigpipe_block(sigpipe_block const&)            = delete;
  sigpipe_block& operator=(sigpipe_block const&) = delete;

 private:
  sigset_t _old{};
  bool _active{false};
};

/// tls_stream::force_rx_tier_for_testing (-1: follow stream_options).
std::atomic<int> g_force_rx_tier{-1};

/// tls_support::set_ktls_install_for_testing (nullptr: the kernel).
std::atomic<tls_support::install_fn> g_ktls_install{nullptr};

/// Endpoints already logged (FNV-1a of host and port; 0 = free). Lock-free, bounded: after 64
/// distinct endpoints the log goes quiet (the pool's counter keeps counting).
std::array<std::atomic<std::uint64_t>, 64> g_logged{};

bool first_fallback_for(std::uint64_t key) noexcept
{
  key |= 1;  // never 0 (the free marker)
  for (auto& slot : g_logged) {
    std::uint64_t v = slot.load(std::memory_order_acquire);
    if (v == key) return false;
    if (v == 0) {
      if (slot.compare_exchange_strong(v, key, std::memory_order_acq_rel)) return true;
      if (v == key) return false;
    }
  }
  return false;
}

std::uint64_t fnv1a(std::string_view s, std::uint64_t h = 0xcbf29ce484222325ULL) noexcept
{
  for (char const c : s) {
    h ^= static_cast<unsigned char>(c);
    h *= 0x100000001b3ULL;
  }
  return h;
}

/// The once-per-endpoint fallback log (gpuio ARCHITECTURE §4.4 "logged once per endpoint"; the
/// per-connection count is the pool's `ktls_fallbacks`). Cold path (once per connection).
void log_fallback_once(std::string_view host,
                       std::uint16_t port,
                       tls::session_info const& info,
                       bool ktls_disabled,
                       std::string_view rx) noexcept
{
  if (!first_fallback_for(fnv1a(host) ^ (static_cast<std::uint64_t>(port) << 48))) return;
  // gpuio T24b: an explicit `ktls = disable` is the configured mode, not a degradation.
  if (ktls_disabled) {
    CUCASCADE_LOG_INFO(
      "TLS for {}:{} uses the user-space record layer (receive: {}): kTLS disabled by policy",
      host,
      port,
      rx);
  } else {
    CUCASCADE_LOG_WARN(
      "kTLS not engaged for {}:{} (TLS version 0x{:04x}, suite 0x{:04x}): using the user-space "
      "record layer (receive: {}); counted in ktls_fallbacks",
      host,
      port,
      static_cast<unsigned>(info.version),
      static_cast<unsigned>(info.cipher_suite),
      rx);
  }
}

}  // namespace

// ================================================================================================
// tls_stream: lifetime
// ================================================================================================

tls_stream::tls_stream() noexcept = default;

void tls_stream::force_rx_tier_for_testing(int tier) noexcept
{
  g_force_rx_tier.store(tier, std::memory_order_relaxed);
}

int tls_stream::forced_rx_tier() noexcept
{
  return g_force_rx_tier.load(std::memory_order_relaxed);
}

tls_stream* tls_stream::create(int fd,
                               tls::session& session,
                               std::unique_ptr<tls::record_opener> opener,
                               stream_options opts,
                               std::error_code& ec) noexcept
{
  ec.clear();
  tls::record_sealer* const sealer = session.sealer();
  tls::record_opener* const rx     = opener ? opener.get() : session.opener();
  if (fd < 0 || sealer == nullptr || rx == nullptr) {
    ec = make_error_code(errc::invalid_argument);
    return nullptr;
  }
  auto* s = new (std::nothrow) tls_stream;
  if (s == nullptr) {
    ec = make_error_code(std::errc::not_enough_memory);
    return nullptr;
  }
  s->_fd         = fd;
  s->_sealer     = sealer;
  s->_opener     = rx;
  s->_tier2      = static_cast<bool>(opener);
  s->_own_opener = std::move(opener);
  s->_waitall    = opts.waitall;
  s->_tx_batch   = std::clamp(opts.tx_batch_records, 1U, max_tx_batch_records);
  return s;
}

void tls_stream::destroy(tls_stream* s) noexcept { delete s; }

// ================================================================================================
// Send path
// ================================================================================================

void tls_stream::prep_send(io_uring_sqe* sqe, std::span<iovec const> tx) noexcept
{
  if (_broken || _tx_inflight != 0) {
    // Broken stream, or a ciphertext SEND still in flight (unreachable with the pool's
    // protocol). One SEND at a time keeps the records in order, so fail this send instead of
    // issuing a second one.
    io_uring_prep_nop(sqe);
    _tx_err = _broken ? EPROTO : EBUSY;
    return;
  }
  if (_tx_owed == 0) seal(tx);  // else: a short CQE left ciphertext of the owed batch queued
  if (_broken || _tx_owed == 0 || _tx.empty()) {
    io_uring_prep_nop(sqe);
    _tx_err = _broken ? EPROTO : ENOBUFS;  // FIFO full of control records: pathological
    return;
  }
  auto const q = _tx.readable();
  io_uring_prep_send(sqe, _fd, q.data(), q.size(), send_msg_flags | MSG_WAITALL);
  _tx_inflight = static_cast<std::uint32_t>(q.size());
}

void tls_stream::seal(std::span<iovec const> tx) noexcept
{
  _tx.compact();  // no SEND in flight here
  // At most `_tx_batch` records per batch; an owed KeyUpdate reply goes first (the
  // sealer emits it before the next record, RFC 8446 §4.6.3). The sealer stops on its own when
  // the FIFO has no room for another record (its `record_overhead()` reserve).
  std::span<std::byte> const out = _tx.write_span();
  if (out.size() < _sealer->record_overhead()) {
    _broken = true;  // unreachable: the FIFO is empty after a completed SEND
    return;
  }
  tls::seal_result const r = _sealer->seal(tx, 0, out, _tx_batch);
  if (r.error) {
    _broken = true;
    return;
  }
  _tx.produce(r.ciphertext);
  _tx_enq += r.ciphertext;
  _tx_owed += static_cast<std::uint32_t>(r.plaintext);
  _tx_mark = _tx_enq;  // the owed plaintext is complete once everything queued so far is sent
}

void tls_stream::seal_control() noexcept
{
  // No plaintext: the sealer emits what the library owes (a KeyUpdate reply, an alert) only.
  _tx.compact();  // no SEND in flight here
  tls::seal_result const r = _sealer->seal({}, 0, _tx.write_span(), _tx_batch);
  if (r.error) {
    _broken = true;
    return;
  }
  _tx.produce(r.ciphertext);
  _tx_enq += r.ciphertext;
}

void tls_stream::tx_sent(std::size_t n) noexcept
{
  _tx.consume(n);
  _tx_sent_total += n;
  _ctr.send_bytes += n;
}

int tls_stream::on_send(int res) noexcept
{
  if (_tx_err != 0) {  // NOP placeholder from prep_send
    int const e = _tx_err;
    _tx_err     = 0;
    return -e;
  }
  std::uint32_t const n = _tx_inflight;
  _tx_inflight          = 0;
  ++_ctr.sends;
  if (res < 0) {
    _broken = true;  // ciphertext partially on the wire: the TLS stream cannot continue
    return res;
  }
  std::size_t const sent = std::min<std::size_t>(static_cast<std::size_t>(res), n);
  tx_sent(sent);
  if (sent < n) ++_ctr.short_sends;
  if (_tx_sent_total >= _tx_mark) {
    auto const r = static_cast<int>(_tx_owed);
    _tx_owed     = 0;
    if (r == 0) return -EPROTO;  // unreachable: prep_send only sends with plaintext owed
    _ctr.plaintext_tx += static_cast<std::uint64_t>(r);
    return r;
  }
  // Short ciphertext CQE (with MSG_WAITALL: an error or a cancel cut the send). Report part of
  // the batch, never all of it: the pool continues with prep_send, which sends the rest of the
  // FIFO without sealing anything new.
  if (_tx_owed < 2) {
    _broken = true;
    return -EPIPE;
  }
  auto const r = static_cast<int>(_tx_owed / 2);
  _tx_owed -= static_cast<std::uint32_t>(r);
  _ctr.plaintext_tx += static_cast<std::uint64_t>(r);
  return r;
}

// ================================================================================================
// Receive path
// ================================================================================================

tls::ciphertext_view tls_stream::buffered() const noexcept
{
  auto const v = _rx.segments();
  return tls::ciphertext_view{v.a, v.b};
}

bool tls_stream::progress_possible() const noexcept
{
  // Plaintext pending in the opener, or a complete record at the ring head (tier 1: the head is
  // a record boundary, the ring is linear; tier 2: a record may straddle the wrap).
  return _opener->progress_possible(buffered());
}

void tls_stream::prep_recv_header(io_uring_sqe* sqe, std::span<char> hb, bool poll_first) noexcept
{
  (void)hb;  // decrypted into in on_recv (the pool passes the same window there)
  prep_rx(sqe, true, poll_first);
}

void tls_stream::prep_recv(io_uring_sqe* sqe, std::span<iovec const> rx) noexcept
{
  std::uint64_t w = 0;
  for (auto const& v : rx)
    w += v.iov_len;
  _rx_window = w;
  prep_rx(sqe, false, false);
  _rx_window = 0;
}

void tls_stream::prep_rx(io_uring_sqe* sqe, bool header, bool poll_first) noexcept
{
  _header_phase = header;
  if (_flush_pending && !_broken) {
    if (_tx_inflight != 0) {  // the pool's SEND drains the FIFO first: wait for its CQE
      io_uring_prep_nop(sqe);
      _rx_op = rx_op::nop;
      ++_ctr.flush_waits;
      return;
    }
    if (_tx.empty()) seal_control();
    if (!_broken && !_tx.empty()) {  // the library's output goes out before it reads on
      auto const q = _tx.readable();
      io_uring_prep_send(sqe, _fd, q.data(), q.size(), send_msg_flags | MSG_WAITALL);
      _tx_inflight = static_cast<std::uint32_t>(q.size());
      _rx_op       = rx_op::flush;
      ++_ctr.flushes;
      return;
    }
    _flush_pending = false;  // nothing owed: the library can read again
  }
  if (_broken || _rx_err != 0 || _peer_eof || _closed || progress_possible()) {
    io_uring_prep_nop(sqe);  // NOP rule: progress without touching the socket
    _rx_op = rx_op::nop;
    return;
  }
  prep_ring_recv(sqe);
  if (poll_first && _rx_op == rx_op::recv) sqe->ioprio |= IORING_RECVSEND_POLL_FIRST;
}

void tls_stream::prep_ring_recv(io_uring_sqe* sqe) noexcept
{
  if (!_tier2) {
    // Tier 1: the ring holds at most one partial record here: move it to the front (<= 18.5
    // KiB, at most once per window) so the receive gets the whole tail.
    if (_rx.compact() != 0) ++_ctr.compactions;
    auto const w = _rx.write_span();
    io_uring_prep_recv(sqe, _fd, w.data(), w.size(), 0);
    _rx_op = rx_op::recv;
    return;
  }
  // Tier 2: wrap mode, nothing moves; all free space in ring order (a record may straddle the
  // end of the storage). Two regions => one RECVMSG (the msghdr lives until the CQE).
  std::size_t n = _rx.recv_iov(_rx_iov);
  int flags     = 0;
  // Batching (stream_options::waitall = connection_config::recv_waitall): a body window still
  // owes at least `owed` ciphertext bytes (its plaintext minus what the ring and the record
  // layer already hold: ciphertext is never shorter than its plaintext), so a MSG_WAITALL
  // receive of at most that much can never wait for bytes the server will not send (gpuio §4.2
  // rule 1). Without it a consumer faster than the sender wakes for every TCP segment (measured:
  // 38 vs 240 KiB per receive, +60% kernel CPU per GB). Header and idle receives stay one-shot.
  std::uint64_t const have = _rx.size() + _opener->pending();
  std::uint64_t const owed = _rx_window > have ? _rx_window - have : 0;
  if (_waitall && owed >= waitall_min && n != 0) {
    std::uint64_t left = owed;
    for (std::size_t i = 0; i < n; ++i) {
      if (left <= _rx_iov[i].iov_len) {
        _rx_iov[i].iov_len = static_cast<std::size_t>(left);
        n                  = i + 1;
        break;
      }
      left -= _rx_iov[i].iov_len;
    }
    flags = MSG_WAITALL;
    ++_ctr.waitall_recvs;
  }
  if (n == 1) {
    io_uring_prep_recv(sqe, _fd, _rx_iov[0].iov_base, _rx_iov[0].iov_len, flags);
  } else {
    _rx_msg            = msghdr{};
    _rx_msg.msg_iov    = _rx_iov.data();
    _rx_msg.msg_iovlen = n;
    io_uring_prep_recvmsg(sqe, _fd, &_rx_msg, static_cast<unsigned>(flags));
  }
  _rx_op = rx_op::recv;
}

rx_result tls_stream::on_recv(int res, unsigned flags, std::span<iovec const> rx) noexcept
{
  (void)flags;
  rx_op const op = _rx_op;
  _rx_op         = rx_op::none;
  switch (op) {
    case rx_op::recv:
      ++_ctr.recvs;
      if (res > 0) {
        _rx.produce(static_cast<std::size_t>(res));
        _ctr.recv_bytes += static_cast<std::uint64_t>(res);
      } else if (res == 0) {
        _peer_eof = true;  // decrypt what is buffered, then eof
      } else if (res == -EAGAIN || res == -EINTR) {
        ++_ctr.again;
        return {0, rx_kind::again, 0};
      } else {
        return {0, rx_kind::error, -res};
      }
      break;
    case rx_op::nop:
      ++_ctr.nops;
      if (res < 0) return {0, rx_kind::error, -res};
      break;
    case rx_op::flush: {
      std::uint32_t const n = _tx_inflight;
      _tx_inflight          = 0;
      ++_ctr.sends;
      if (res < 0) {
        _broken = true;
        return {0, rx_kind::error, -res};
      }
      std::size_t const sent = std::min<std::size_t>(static_cast<std::size_t>(res), n);
      tx_sent(sent);
      if (sent < n) ++_ctr.short_sends;
      _flush_pending = false;  // room freed; the library retries its pending write next
      break;
    }
    case rx_op::none: return {0, rx_kind::error, EPROTO};  // a CQE this stream did not prepare
  }
  if (_rx_err != 0) return {0, rx_kind::error, _rx_err};
  if (_broken) return {0, rx_kind::error, EPROTO};
  if (_flush_pending) {  // still waiting for the pool's SEND to drain the FIFO
    ++_ctr.again;
    return {0, rx_kind::again, 0};
  }
  return decrypt(rx);
}

bool tls_stream::request_key_update_reply() noexcept
{
  // RFC 8446 §4.6.3: answer update_requested with update_not_requested before the next record we
  // send; the sealer coalesces several requests into one reply (gpuio security review F3), so a
  // KeyUpdate storm never makes the shell write while reading.
  ++_ctr.key_updates;
  if (_sealer->request_key_update_reply()) {
    _broken = true;
    _rx_err = EPROTO;
    return false;
  }
  ++_ctr.key_update_requests;
  return true;
}

rx_result tls_stream::decrypt(std::span<iovec const> rx) noexcept
{
  std::size_t total = 0;
  bool starved      = false;
  // gpuio F7: the pool never passes an empty window (a data record would stay pending while
  // progress_possible() keeps NOP-ing); fail instead of spinning if it ever did.
  std::size_t window = 0;
  for (auto const& v : rx)
    window += v.iov_len;
  CUCASCADE_NET_ASSERT(window != 0);
  if (window == 0) [[unlikely]] {
    _broken = true;
    _rx_err = EPROTO;
    return {0, rx_kind::error, EPROTO};
  }
  for (;;) {
    tls::open_result const st = _opener->open(buffered(), rx, total, _header_phase);
    _rx.consume(st.consumed);
    total += st.plaintext;
    if (st.stop == tls::open_stop::key_update) {  // RX re-keyed; the reply is owed
      if (!request_key_update_reply()) break;
      if (st.consumed != 0 || st.plaintext != 0) continue;
      _flush_pending = true;  // the library must write before it reads on
      break;
    }
    if (st.stop == tls::open_stop::closed) {
      _closed = true;
    } else if (st.stop == tls::open_stop::failed) {
      // Malformed record, tag failure, alert: destination contents undefined, connection dead.
      _broken = true;
      _rx_err = EBADMSG;
    }
    starved = st.stop == tls::open_stop::starved;
    break;
  }
  if (_broken) return {0, rx_kind::error, _rx_err != 0 ? _rx_err : EPROTO};
  if (total != 0) {
    _ctr.plaintext_rx += total;
    return {total, rx_kind::data, 0};
  }
  if (_closed) return {0, rx_kind::eof, 0};
  if (_peer_eof && starved) {
    // gpuio F7: FIN in the middle of a record is a truncated record (decode_error), not an EOF.
    if (!_rx.empty()) {
      _broken = true;
      _rx_err = EBADMSG;
      return {0, rx_kind::error, EBADMSG};
    }
    return {0, rx_kind::eof, 0};
  }
  ++_ctr.again;
  return {0, rx_kind::again, 0};
}

bool tls_stream::consume_control_records() noexcept
{
  // An empty window: control records are consumed, application data stays buffered (= desync
  // at check-in, dead on an idle connection).
  for (;;) {
    tls::open_result const st = _opener->open(buffered(), {}, 0, false);
    _rx.consume(st.consumed);
    if (st.stop == tls::open_stop::key_update) {
      // The reply goes out with the next request; without progress the library must write
      // first, which an idle / checked-in connection cannot do: not reusable.
      if (!request_key_update_reply() || (st.consumed == 0 && st.plaintext == 0)) return false;
      continue;
    }
    if (st.stop == tls::open_stop::starved)
      return _opener->pending() == 0;  // tickets / KeyUpdate consumed, nothing else complete
    if (st.stop == tls::open_stop::closed) {
      _closed = true;
    } else if (st.stop == tls::open_stop::failed) {
      _broken = true;
      _rx_err = EBADMSG;
    }
    return false;  // application data, close_notify, protocol error
  }
}

// ================================================================================================
// Check-in (security review F5)
// ================================================================================================

bool tls_stream::clean_for_reuse() const noexcept
{
  if (_broken || _closed || _peer_eof || _flush_pending || _rx_err != 0 || !_rx.empty())
    return false;
  return _opener->pending() == 0;
}

bool tls_stream::drain_for_reuse() noexcept
{
  if (_broken || _closed || _peer_eof || _flush_pending || _rx_err != 0) return false;
  if (_opener->pending() != 0) return false;  // plaintext past the response: desync
  if (!consume_control_records()) return false;
  return _rx.empty();  // a partial record left: not reusable
}

// ================================================================================================
// Idle connection
// ================================================================================================

bool tls_stream::drain_idle_socket() noexcept
{
  _header_phase = false;
  if (_broken || _closed || _peer_eof || _rx_err != 0 || _flush_pending) return false;
  if (_rx.size() == ring_bytes) return false;  // unreachable: records are consumed as they come
  std::size_t const n = _rx.recv_iov(_rx_iov);
  if (n == 0) return false;
  msghdr msg{};
  msg.msg_iov     = _rx_iov.data();
  msg.msg_iovlen  = n;
  ssize_t const r = ::recvmsg(_fd, &msg, MSG_DONTWAIT);
  if (r < 0) return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;  // spurious
  if (r == 0) {
    _peer_eof = true;
    return false;
  }
  ++_ctr.idle_drains;
  ++_ctr.recvs;
  _rx.produce(static_cast<std::size_t>(r));
  _ctr.recv_bytes += static_cast<std::uint64_t>(r);
  // A partial record (half a ticket) stays buffered for the next receive: the connection is
  // healthy, only application data, close_notify, EOF or an error retire it.
  return consume_control_records();
}

// ================================================================================================
// tls_support: the pool's TLS seam
// ================================================================================================

void tls_support::set_ktls_install_for_testing(install_fn fn) noexcept
{
  g_ktls_install.store(fn, std::memory_order_relaxed);
}

tls::handshake_status tls_support::handshake(tls::session& s, std::error_code& ec) noexcept
{
  sigpipe_block const no_sigpipe;
  return s.handshake(ec);
}

std::error_code tls_support::handshake_error(std::error_code ec) noexcept
{
  if (!ec) return make_error_code(errc::tls_handshake_failed);
  if (ec.category() == tls::error_category()) {
    switch (static_cast<tls::errc>(ec.value())) {
      case tls::errc::verify_failed: return make_error_code(errc::tls_verify_failed);
      case tls::errc::ktls_unavailable: return make_error_code(errc::ktls_unavailable);
      case tls::errc::protocol_error: return make_error_code(errc::tls_protocol_error);
      default: return make_error_code(errc::tls_handshake_failed);
    }
  }
  if (ec.category() == std::system_category() || ec.category() == std::generic_category())
    return ec;  // the socket failed under the handshake (ECONNRESET, ...)
  return make_error_code(errc::tls_handshake_failed);
}

tls_support::setup_result tls_support::finish(int fd,
                                              tls::session& s,
                                              tls::provider const& p,
                                              stream_options const& opts,
                                              std::string_view host,
                                              std::uint16_t port) noexcept
{
  setup_result r{};
  r.info                = s.info();
  tls::config const& cf = p.policy();
  // The library must hold nothing past the handshake before the record protection moves (read
  // ahead off, gpuio T5); otherwise the kernel / built-in layer would miss those bytes and the
  // library's own record layer (tier 1) keeps the connection.
  bool const clean = r.info.buffered_bytes == 0;
  tls::traffic_keys rx;
  tls::traffic_keys tx;
  bool have_rx = false;
  if (cf.ktls != tls::ktls_mode::disable) {
    bool have_tx = false;
    if (clean) {
      have_rx = !s.export_keys(tls::direction::receive, rx);
      have_tx = have_rx && !s.export_keys(tls::direction::send, tx);
    }
    if (have_rx && have_tx) {
      bool const tls13         = r.info.version == tls::protocol_version::tls1_3;
      install_fn const install = g_ktls_install.load(std::memory_order_relaxed);
      tls::ktls::install_result const ir =
        install != nullptr ? install(fd, rx, tx, cf.expect_no_pad && tls13)
                           : tls::ktls::install(fd, rx, tx, cf.expect_no_pad && tls13);
      r.ktls_rx = ir.rx;
      r.ktls_tx = ir.tx;
      r.no_pad  = ir.no_pad;
      if (ir.rx && ir.tx) {  // gpuio T1: kTLS RX and TX, both or nothing
        r.path  = tls_path::ktls;
        r.rekey = tls::ktls::make_rekey_state(rx, tx, p.kdf()).release();
        return r;  // the key copies are wiped by their destructors
      }
      if (ir.rx || ir.tx) {
        // One direction in the kernel: that state cannot be undone, and neither the kTLS stream
        // nor the user-space shell can run on it (gpuio T1: RX-only fails).
        r.ec = ir.error ? ir.error : make_error_code(errc::ktls_unavailable);
        if (r.ec.category() == tls::error_category())
          r.ec = make_error_code(errc::ktls_unavailable);
        return r;
      }
    }
    if (cf.ktls == tls::ktls_mode::require) {
      r.ec = make_error_code(errc::ktls_unavailable);
      return r;
    }
  }
  // User space: tier 2 (the built-in record layer over the receive keys) when asked for and the
  // suite allows it, else tier 1 (the library's own opener).
  r.user_space     = true;
  int const forced = tls_stream::forced_rx_tier();
  bool const want2 = forced >= 0 ? forced == 1 : opts.rx_record_layer;
  std::unique_ptr<tls::record_opener> opener;
  std::string_view rx_name = "library record layer (tier 1)";
  if (want2 && clean) {
    if (!have_rx) have_rx = !s.export_keys(tls::direction::receive, rx);
    if (have_rx && rx.cipher != tls::aead_cipher::other) {
      std::error_code oe;
      opener = tls::make_record_opener(std::move(rx), p, oe);
    }
    rx_name = opener ? std::string_view{"built-in record layer (tier 2)"}
                     : std::string_view{"library record layer (tier 1; suite not served)"};
  }
  rx.wipe();
  tx.wipe();
  if (s.detach_from_socket()) {
    r.ec = make_error_code(errc::tls_handshake_failed);
    return r;
  }
  std::error_code ec;
  r.stream = tls_stream::create(fd, s, std::move(opener), opts, ec);
  if (r.stream == nullptr) {
    r.ec = ec;
    return r;
  }
  r.path         = tls_path::user_space;
  r.fallback     = cf.ktls != tls::ktls_mode::disable;
  r.record_layer = r.stream->record_layer();
  log_fallback_once(host, port, r.info, cf.ktls == tls::ktls_mode::disable, rx_name);
  return r;
}

void tls_support::release_rekey(tls::ktls::rekey_state* state) noexcept
{
  if (state != nullptr) tls::ktls::rekey_state_deleter{}(state);
}

}  // namespace cucascade::io::net
