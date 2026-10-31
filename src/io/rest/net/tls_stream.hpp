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

#include "io/rest/net/cipher_ring.hpp"
#include "io/rest/net/common.hpp"
#include "io/rest/net/streams.hpp"
#include "io/tls/ktls.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <liburing.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <system_error>

/**
 * @file tls_stream.hpp
 * @brief The user-space TLS I/O shell (`tls_stream`, gpuio `tls_stream` minus OpenSSL) and the
 *        connection pool's TLS seam (`tls_support`): what a connection does after its handshake.
 *
 * ## TLS setup (`tls_support::finish`, docs/io/rest-uring-design.md section 4.3)
 * After `tls::session::handshake` returned `done`, with `ktls != disable` and nothing buffered by
 * the library: both directions' keys are exported and installed by `tls::ktls::install`
 * (TCP_ULP, TLS_TX, TLS_RX, TLS_RX_EXPECT_NO_PAD for TLS 1.3, gpuio T1/T4). Both installed: a
 * kTLS connection (`ktls_stream`, <= 64 B) plus, for TLS 1.3 on a re-key-capable kernel, the
 * `rekey_state` the slot keeps for KeyUpdates. Nothing installed: user space (counted in
 * `ktls_fallbacks` unless kTLS is disabled; `require` fails with errc::ktls_unavailable). One
 * direction only: errc::ktls_unavailable (kernel state cannot be undone).
 *
 * ## The user-space stream (`tls_stream`; gpuio T16 tier 1, gpuio T19 tier 2)
 * The shell owns the socket I/O; the TLS objects never submit I/O for data (sans-I/O):
 *  - Receive: ciphertext lands in a 256 KiB `cipher_ring` (one-shot `IORING_OP_RECV`); a
 *    `tls::record_opener` decrypts from it straight into the pool's window (the free tail of the
 *    header buffer for the header phase, one record per header receive; the body iovecs, which
 *    are min(remaining, body_recv_cap): nothing is ever delivered past what the caller asked
 *    for, WAITALL rule 1). Tier 2 (default, gpuio T10): the built-in record layer
 *    (`tls::make_record_opener` over the receive keys), ring in wrap mode (`recv_iov`: all free
 *    space in ring order, one RECV or a two-iovec RECVMSG, nothing ever moved), body receives
 *    with MSG_WAITALL bounded by the ciphertext the window still owes when that bound is
 *    >= 32 KiB (38 vs 240 KiB per receive without it, +60% kernel CPU/GB). Tier 1 (escape hatch,
 *    or a suite the record layer does not serve): the library's own opener after
 *    `session::detach_from_socket`, ring in linear mode (compaction before a receive), one-shot
 *    receives (the ciphertext owed is not known exactly).
 *  - NOP rule: when buffered state already allows progress without the socket
 *    (`record_opener::progress_possible`: plaintext pending, or a complete record at the ring
 *    head) the receive SQE is an `IORING_OP_NOP`, and the decrypt runs when its CQE comes back
 *    through the pool's normal CQE path. A RECV there could wait forever: the server may have
 *    sent everything already.
 *  - Control records are consumed by the opener: NewSessionTicket skipped; KeyUpdate re-keys RX
 *    and makes the shell call `record_sealer::request_key_update_reply` (coalesced, emitted before
 *    the next record, RFC 8446 §4.6.3); close_notify => eof; a tag failure, an alert, a malformed
 *    record or a FIN inside a record => error EBADMSG (errc::tls_protocol_error; the connection
 *    is not reused). A `key_update` stop without progress means the library must write before
 *    it reads on (tier 1: a reply OpenSSL writes while reading): the next receive SQE becomes a
 *    SEND of a control-only seal (or a NOP while the pool's own SEND is in flight).
 *  - Send: `prep_send` seals at most `stream_options::tx_batch_records` records (default 16:
 *    256 KiB of plaintext) per batch with the session's
 *    `tls::record_sealer` into a ciphertext FIFO of 17 records and SENDs the FIFO (`MSG_NOSIGNAL |
 *    MSG_WAITALL`); `on_send` maps the ciphertext CQE back to plaintext bytes of the pool's tx
 *    window (the whole batch once all of its ciphertext is out; on a short CQE part of the batch,
 *    never all of it). One ciphertext SEND in flight at a time (record order).
 *  - Check-in (gpuio security review F5): `clean_for_reuse() || drain_for_reuse()`: bytes past
 *    the response close the slot instead of desynchronising the next response; control records
 *    that arrived with the response end are consumed.
 *  - Idle: a readable idle connection (TLS 1.3 tickets) is received into the ring and its control
 *    records consumed (`prep_idle_recv` / `on_idle_recv`).
 *  - Memory: one allocation per connection, in `create()` (ring 256 KiB + FIFO 64 KiB + state);
 *    nothing per exchange. `io-wq` placement does not apply: decryption runs on the reactor
 *    thread (gpuio T11).
 */

namespace cucascade::io::net {

/// Per-connection counters of the user-space stream (tests, stats aggregation).
struct tls_stream_counters {
  std::uint64_t recvs{0};          ///< ciphertext RECV CQEs
  std::uint64_t recv_bytes{0};     ///< ciphertext bytes received
  std::uint64_t waitall_recvs{0};  ///< tier 2: bounded MSG_WAITALL receives issued
  std::uint64_t nops{0};           ///< receives served from buffered state (NOP rule)
  std::uint64_t again{0};          ///< receive CQEs that placed nothing (more needed)
  std::uint64_t sends{0};          ///< ciphertext SEND CQEs
  std::uint64_t send_bytes{0};     ///< ciphertext bytes sent
  std::uint64_t short_sends{0};    ///< SEND CQEs shorter than the ciphertext queued
  std::uint64_t flushes{0};        ///< receive SQEs turned into a FIFO SEND (library output owed)
  std::uint64_t flush_waits{0};    ///< NOPs waiting for the pool's SEND before a flush
  std::uint64_t compactions{0};    ///< tier 1: partial records moved to the ring's front
  std::uint64_t idle_drains{0};    ///< idle-connection receives (tickets)
  std::uint64_t key_updates{0};    ///< KeyUpdates received that requested a reply
  std::uint64_t key_update_requests{0};  ///< replies requested from the sealer (coalesced there)
  std::uint64_t plaintext_rx{0};         ///< == bytes_rx()
  std::uint64_t plaintext_tx{0};         ///< plaintext bytes reported to the pool as sent
};

/// User-space TLS I/O shell, tier 1 / tier 2 (see the file comment). Satisfies
/// `fallback_stream` and the pool's idle-drain and reuse-check hooks.
class tls_stream {
 public:
  /// Ciphertext receive ring (gpuio ARCHITECTURE §4.4: 256 KiB): 15 full records per RECV.
  static constexpr std::size_t ring_bytes = 256 * KiB;
  /// Largest `stream_options::tx_batch_records`.
  static constexpr unsigned max_tx_batch_records = 16;
  /// Ciphertext send FIFO: one batch of up to 16 full records plus control replies (KeyUpdate
  /// ~30 B).  Not value-initialised: a connection that never sends a body touches one page of it.
  static constexpr std::size_t tx_bytes =
    (max_tx_batch_records + 1) * tls::max_record_ciphertext + 4 * KiB;
  /// Tier 2: smallest bounded MSG_WAITALL body receive (below it the receive is one-shot).
  static constexpr std::size_t waitall_min = 32 * KiB;

  /**
   * @brief Create the stream of a user-space connection (after `detach_from_socket`).
   *
   * @param fd       the connected socket (not owned)
   * @param session  the finished, detached session (not owned: the slot frees it after
   *                 `destroy`); its `sealer()` sends, and its `opener()` receives on tier 1
   * @param opener   the built-in record layer (tier 2, owned from now on), or null (tier 1)
   * @param opts     endpoint receive options (`waitall` bounds tier-2 body receives)
   * @param ec       not_enough_memory, or invalid_argument (no sealer / no opener)
   * @return the stream, or null with @p ec
   */
  [[nodiscard]] static tls_stream* create(int fd,
                                          tls::session& session,
                                          std::unique_ptr<tls::record_opener> opener,
                                          stream_options opts,
                                          std::error_code& ec) noexcept;
  /// @brief Free the stream and its opener (the pool calls it before freeing the session, after
  /// every CQE was reaped).
  static void destroy(tls_stream* s) noexcept;

  tls_stream(tls_stream const&)            = delete;
  tls_stream& operator=(tls_stream const&) = delete;

  /// @brief TESTING ONLY: force the receive tier of streams set up from now on, process-wide
  /// (-1: follow `stream_options::rx_record_layer`, 0: tier 1, 1: tier 2 when possible).
  static void force_rx_tier_for_testing(int tier) noexcept;
  /// @return the forced tier (-1 when not forced).
  [[nodiscard]] static int forced_rx_tier() noexcept;

  [[nodiscard]] int fd() const noexcept { return _fd; }
  /// @return plaintext bytes delivered into receive windows.
  [[nodiscard]] std::uint64_t bytes_rx() const noexcept { return _ctr.plaintext_rx; }
  [[nodiscard]] tls_stream_counters const& counters() const noexcept { return _ctr; }
  /// @return ciphertext bytes buffered (received, not yet decrypted).
  [[nodiscard]] std::size_t buffered_ciphertext() const noexcept { return _rx.size(); }
  /// @return ciphertext bytes queued for sending.
  [[nodiscard]] std::size_t queued_ciphertext() const noexcept { return _tx.size(); }
  /// @return true when this connection receives through the built-in record layer (tier 2).
  [[nodiscard]] bool record_layer() const noexcept { return _tier2; }

  // ---- completion_stream / pool_stream ----

  /// @brief Seal (a batch of) @p tx and SEND the ciphertext FIFO (`MSG_NOSIGNAL|MSG_WAITALL`).
  /// After a short ciphertext CQE the remaining FIFO is sent and nothing new is sealed.
  void prep_send(io_uring_sqe* sqe, std::span<iovec const> tx) noexcept;
  /// @brief Map a ciphertext SEND CQE to plaintext bytes of the pool's tx window, or -errno.
  [[nodiscard]] int on_send(int res) noexcept;
  /// @brief Header receive: NOP when buffered state allows progress, else a one-shot RECV into
  /// the ring (`IORING_RECVSEND_POLL_FIRST` when @p poll_first). @p hb is the window `on_recv`
  /// decrypts into (it is passed again there).
  void prep_recv_header(io_uring_sqe* sqe, std::span<char> hb, bool poll_first) noexcept;
  /// @brief Body receive into the window @p rx (same rule; @p rx is passed again to `on_recv`).
  void prep_recv(io_uring_sqe* sqe, std::span<iovec const> rx) noexcept;
  /// @brief Interpret the receive CQE and decrypt into @p rx.
  /// @return data (bytes placed, <= window), again (nothing placed: re-issue), eof, error.
  [[nodiscard]] rx_result on_recv(int res, unsigned flags, std::span<iovec const> rx) noexcept;

  // ---- check-in (security review F5) ----

  /// @return true when nothing past the finished response is buffered: no pending plaintext, an
  /// empty ciphertext ring, and the stream neither closed, at EOF nor broken.
  [[nodiscard]] bool clean_for_reuse() const noexcept;
  /// @brief Synchronously consume the complete control records already in the ring.
  /// @return true when the stream is then clean; false on any application byte, a close_notify,
  /// a partial record or an error (the pool closes the slot).
  [[nodiscard]] bool drain_for_reuse() noexcept;
  /// @return true when the peer ended the stream cleanly: close_notify consumed or TCP FIN seen.
  [[nodiscard]] bool peer_closed() const noexcept { return _closed || _peer_eof; }

  // ---- idle connection (the transport's reuse check) ----

  /// @brief Synchronously receive what is queued on an idle connection (TLS 1.3
  /// NewSessionTickets, a KeyUpdate) with one non-blocking `recv` into the ring and consume its
  /// control records.
  /// @return true: nothing but control records, possibly a partial one left buffered for the
  /// next receive (the connection stays reusable); false: EOF, close_notify, application data on
  /// an idle connection or an error (close it).
  [[nodiscard]] bool drain_idle_socket() noexcept;

 private:
  enum class rx_op : std::uint8_t { none, recv, nop, flush };

  tls_stream() noexcept;  // ring storage left uninitialised (written before read)
  ~tls_stream() = default;

  [[nodiscard]] tls::ciphertext_view buffered() const noexcept;
  void prep_rx(io_uring_sqe* sqe, bool header, bool poll_first) noexcept;
  [[nodiscard]] bool progress_possible() const noexcept;
  [[nodiscard]] rx_result decrypt(std::span<iovec const> rx) noexcept;
  [[nodiscard]] bool request_key_update_reply() noexcept;
  [[nodiscard]] bool consume_control_records() noexcept;
  void prep_ring_recv(io_uring_sqe* sqe) noexcept;
  void seal(std::span<iovec const> tx) noexcept;
  void seal_control() noexcept;
  void tx_sent(std::size_t n) noexcept;

  int _fd{-1};
  tls::record_opener* _opener{nullptr};  ///< tier 2: `_own_opener`; tier 1: the session's
  tls::record_sealer* _sealer{nullptr};  ///< the session's
  std::unique_ptr<tls::record_opener> _own_opener{};
  bool _tier2{false};   ///< receive through the built-in record layer
  bool _waitall{true};  ///< tier 2: bounded MSG_WAITALL body receives (recv_waitall)
  unsigned _tx_batch{max_tx_batch_records};  ///< records sealed per SEND
  bool _header_phase{false};
  bool _peer_eof{false};  ///< RECV returned 0 (TCP FIN)
  bool _closed{false};    ///< close_notify consumed
  bool _broken{false};    ///< fatal TLS/TX error: every later op fails
  /// The opener stopped on `key_update` without progress: the library must write (a KeyUpdate
  /// reply, tier 1) before it reads on; the next receive SQE becomes a SEND of the sealer's output
  /// (gpuio tier 1 WANT_WRITE flush).
  bool _flush_pending{false};
  rx_op _rx_op{rx_op::none};
  int _rx_err{0};                   ///< positive errno reported by the next on_recv
  int _tx_err{0};                   ///< positive errno reported by the next on_send (NOP)
  std::uint32_t _tx_inflight{0};    ///< FIFO bytes in the SEND in flight (0: none)
  std::uint32_t _tx_owed{0};        ///< plaintext sealed for the pool, not yet reported
  std::uint64_t _tx_enq{0};         ///< ciphertext bytes ever appended to the FIFO
  std::uint64_t _tx_sent_total{0};  ///< ciphertext bytes ever sent
  std::uint64_t _tx_mark{0};        ///< _tx_enq after the last owed batch: owed done when sent
  tls_stream_counters _ctr{};
  std::uint64_t _rx_window{0};  ///< tier 2: plaintext bytes of the body window being prepared
  msghdr _rx_msg{};             ///< tier 2: RECVMSG into the ring's free tail + front
  std::array<iovec, 2> _rx_iov{};
  cipher_ring<ring_bytes> _rx;
  cipher_ring<tx_bytes> _tx;
};

/// How a finished handshake continues (`tls_support::finish`).
enum class tls_path : std::uint8_t {
  fail,        ///< see `setup_result::ec`
  ktls,        ///< kTLS RX and TX installed: plain socket I/O from now on
  user_space,  ///< `tls_stream` (built-in record layer or the library's)
};

/**
 * @brief The transport's TLS seam: the non-blocking handshake step and what follows it.
 *
 * Reactor thread; noexcept.  `finish` decides between kTLS and the user-space stream from the
 * policy (`ktls` prefer / require / disable) and what the kernel accepted; the kernel install is
 * replaceable for tests (`set_ktls_install_for_testing`), so the decision logic is covered on
 * hosts without `tls.ko`.
 */
struct tls_support {
  using stream = tls_stream;

  /// What the TLS setup produced.
  struct setup_result {
    tls_path path{tls_path::fail};
    tls::session_info info{};
    bool ktls_rx{false};
    bool ktls_tx{false};
    bool no_pad{false};                      ///< TLS_RX_EXPECT_NO_PAD accepted
    bool user_space{false};                  ///< kTLS not engaged
    bool fallback{false};                    ///< user space although kTLS was allowed (prefer)
    bool record_layer{false};                ///< user space through the built-in record layer
    tls_stream* stream{nullptr};             ///< path == user_space (owned by the caller)
    tls::ktls::rekey_state* rekey{nullptr};  ///< path == ktls, TLS 1.3 on a re-key kernel
    std::error_code ec{};
  };

  /// The kTLS install step (`tls::ktls::install` unless replaced for tests).
  using install_fn = tls::ktls::install_result (*)(int fd,
                                                   tls::traffic_keys const& rx,
                                                   tls::traffic_keys const& tx,
                                                   bool expect_no_pad) noexcept;

  /// @brief TESTING ONLY: replace the kernel install (nullptr restores `tls::ktls::install`).
  static void set_ktls_install_for_testing(install_fn fn) noexcept;

  /// @brief One non-blocking handshake step, with SIGPIPE blocked on this thread while the
  /// library writes the socket (never ignore it process-wide).
  [[nodiscard]] static tls::handshake_status handshake(tls::session& s,
                                                       std::error_code& ec) noexcept;
  /// @return the transport error of a failed handshake (tls_verify_failed,
  /// tls_handshake_failed, ktls_unavailable, tls_protocol_error, or the socket errno).
  [[nodiscard]] static std::error_code handshake_error(std::error_code ec) noexcept;
  /// @brief After `done`: kTLS or the user-space stream (file comment).
  [[nodiscard]] static setup_result finish(int fd,
                                           tls::session& s,
                                           tls::provider const& p,
                                           stream_options const& opts,
                                           std::string_view host,
                                           std::uint16_t port) noexcept;
  static void release_rekey(tls::ktls::rekey_state* state) noexcept;
};

}  // namespace cucascade::io::net
