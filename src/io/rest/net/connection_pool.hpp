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

/// @file connection_pool.hpp
/// @brief The per-reactor connection pool (API.md §B.6 `connection_pool`): per-endpoint slot
/// tables of fixed capacity (`max_connections_per_reactor`, allocated once per endpoint, nothing
/// per exchange), checkout with the idle cutoff and the urgent reserve, connect-ahead / warm-up
/// (IORING_OP_SOCKET + IORING_OP_CONNECT, socket options, `connect_timeout` from the reactor
/// heap), the TLS handshake driven through `tls::session::handshake` with POLL_ADD readiness,
/// the kTLS/plain/user-space slot kind from the TLS setup (`Tls::finish`: cuCascade installs the
/// kTLS keys exported by the session, docs/io/rest-uring-design.md section 4.3), whole-buffer
/// stream operations for the HTTP driver (send, one-shot header receive, capped WAITALL body
/// receive with transparent kTLS control-record drain + `tls::ktls::on_control_record`), the
/// idle watch (FIN detection by generation, never cancelled), the stale-reuse rule, byte cap /
/// age, keep-warm probes, io-wq placement and the shutdown protocol (cancel, reap every CQE,
/// then free).
///
/// Design: gpuio ARCHITECTURE §3.5 (GET flow), §4.1–§4.5, §5 (lifetimes), §7.3 (retry classes: the
/// pool reports errc::stale_connection; the worker retries it (gpuio decide_retry)), §9.4 (costs),
/// §11; curl §4.1/§9d (pool rules); uring_ultra §2.3 (the slot machine replaced: no blocking DNS on
/// the reactor (S11), no per-GET strings (S5), no idle blindness).
///
/// ## Template parameters
/// - `Env`: reactor services (the contract in uring_loop.hpp; `loop_env` over a `uring_loop` in
///   production and in tests). No virtual calls.
/// - `Handler`: the exchange driver (the REST worker; tests' GET client). Required, noexcept:
///   ```
///   void on_send(conn_slot&, std::error_code, std::size_t bytes);   // whole tx sent, or error
///   void on_header(conn_slot&, std::error_code, std::size_t n);      // n bytes appended at
///                                                                     // header_buf[hdr_len - n]
///   void on_body(conn_slot&, std::error_code, std::size_t n);        // n bytes committed to sink
///   ```
///   Optional (detected): `void on_ready(std::uint32_t endpoint)` (a new connection became idle;
///   called from CQE context only), `void on_connect_failed(std::uint32_t endpoint,
///   std::error_code)`, `std::size_t build_probe(std::uint32_t endpoint, conn_slot&,
///   std::span<char> out)` (keep-warm HEAD head; 0 = skip).
///   Callbacks run on the reactor thread from `on_op_cqe`; the handler may call any pool function
///   from them (they are always the last thing the pool does with the slot).
/// - `Tls`: the TLS seam (`tls_support`, tls_stream.hpp: handshake error mapping, key
///   installation, kTLS control records, and the user-space `tls_stream` as `Tls::stream`), or
///   `no_tls`: `https://` endpoints then fail their connects with errc::not_supported and no TLS
///   symbol is referenced. Optional on the stream (detected): `void prep_idle_recv(io_uring_sqe*)`
///   + `bool on_idle_recv(int)`: a readable idle kind_tls connection (TLS 1.3 tickets) is drained
///   through them instead of being closed.
///
/// ## Exchange protocol for the driver (http1_exchange)
/// `checkout` -> write the head into `c.tx_buf` -> `send(c, iov)` and `recv_header(c, true)` in the
/// same pass -> parse `c.header()` on each `on_header` (re-issue `recv_header` while need_more)
/// -> copy the leftover body bytes with `c.sink.copy_in` (the header buffer is the control-drain
/// target during the body) -> `recv_body(c, parser.body_remaining())` per `on_body` until 0 ->
/// `check_in(c)` (keep-alive) or `close(c)` (error, `Connection: close`, EOF framing). Set
/// `c.close_after` instead of `close` to finish normally and close at check_in. Deadlines (stall,
/// attempt) are the driver's: arm them on the reactor heap and call `cancel(c)`.

#pragma once

#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/response_parser.hpp"
#include "io/rest/net/common.hpp"
#include "io/rest/net/config.hpp"
#include "io/rest/net/conn_slot.hpp"
#include "io/rest/net/dns_resolver.hpp"
#include "io/rest/net/endpoint.hpp"
#include "io/rest/net/idle_watch.hpp"
#include "io/rest/net/streams.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "io/tls/ktls.hpp"

#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/tls/tls.hpp>

#include <liburing.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>

namespace cucascade::io::net {

// ------------------------------------------------------------------------------------------------
// Counters (gpuio ARCHITECTURE §12 "remote (per endpoint)"); plain reactor-local integers
// ------------------------------------------------------------------------------------------------

struct endpoint_counters {
  // connections
  std::uint64_t connects_started{0}, connects_ok{0}, connect_failures{0}, connect_timeouts{0};
  std::uint64_t dns_failures{0}, dns_waits{0}, socket_fallbacks{0}, bind_failures{0};
  std::uint64_t handshakes_full{0}, handshakes_resumed{0}, handshake_failures{0},
    handshake_timeouts{0};
  std::uint64_t ktls_rx{0}, ktls_tx{0}, ktls_fallbacks{0}, no_pad_set{0}, no_pad_failed{0};
  std::uint64_t tls_version{0}, cipher_id{0};  // last negotiated
  std::uint64_t connect_ahead{0}, warmup_requested{0}, warmup_started{0};
  // checkout / reuse
  std::uint64_t checkouts{0}, checkout_misses{0}, reused_checkouts{0}, reserve_denials{0},
    op_waits{0};
  std::uint64_t stale_detected{0};  ///< reused slot died before any response byte
  // liveness
  std::uint64_t watch_armed{0}, watch_stale{0}, watch_skipped{0}, idle_fin_evictions{0},
    idle_readable_drains{0}, idle_cutoff_closes{0}, byte_cap_closes{0}, age_closes{0}, closes{0};
  /// gpuio T20c: closes of a slot whose response head said `Connection: close` (S3 ends a
  /// connection after ~100 requests this way).
  std::uint64_t connection_close_closes{0};
  std::uint64_t replacements{0};  ///< gpuio T20c (g): connections pre-opened by connect_replacement
  /// gpuio T20c (g): complete responses that ended with `Connection: close`, and the requests those
  /// connections had served; `server_request_limit` = round(requests / closes) after two closes
  /// (S3: 100; 0 = not learned yet). The limit is max-merged by `merge_counters`.
  std::uint64_t peer_closes{0}, peer_close_requests{0}, server_request_limit{0};
  /// kind_tls slots closed at check-in because bytes past the response were buffered (gpuio T19b,
  /// F5).
  std::uint64_t reuse_desync_closes{0};
  std::uint64_t keep_warm_probes{0}, keep_warm_failures{0};
  // kTLS control records
  std::uint64_t control_records_ticket{0}, control_records_key_update{0}, control_records_alert{0},
    control_records_other{0}, close_notify{0}, eio_drains{0}, drain_cqes{0},
    key_update_flush_polls{0}, replays_deferred{0};
  /// More than `max_control_records` kTLS control records in a row (the connection is failed).
  std::uint64_t control_record_storms{0};
  // data path
  std::uint64_t sends{0}, send_cqes{0}, short_sends{0}, bytes_sent{0};
  std::uint64_t header_cqes{0}, header_bytes{0};
  std::uint64_t body_receives{0}, body_cqes{0}, body_bytes{0}, body_bytes_requested{0};
  std::uint64_t waitall_short_cqes{0};  ///< short WAITALL CQE before a control record (rule 3)
  std::uint64_t overread_guarded{0};    ///< receives the min(remaining, cap) rule shortened
  std::uint64_t overread_issued{0};     ///< receives longer than remaining (test knob only)
  std::uint64_t cancels{0};
  std::uint64_t waitall_watchdog_cancels{0};  ///< stuck WAITALL receives recovered (KeyUpdate)
  /// WAITALL receives the watchdog started sampling (first tick that saw no POLLIN with bytes
  /// queued; gpuio T13b): `waitall_watchdog_cancels / waitall_watchdog_armed` is the stall ratio.
  std::uint64_t waitall_watchdog_armed{0};
  std::uint64_t cqes{0};
};

/// Instantaneous slot counts of one endpoint on one reactor.
struct pool_gauges {
  std::uint32_t capacity{0}, free{0}, resolving{0}, connecting{0}, handshaking{0}, idle{0}, busy{0},
    draining{0}, probing{0}, closing{0};
  [[nodiscard]] std::uint32_t live() const noexcept
  {
    return connecting + handshaking + idle + busy + draining + probing;
  }
};

/// @brief Add @p from into @p into (aggregation across reactors and endpoints); `tls_version` and
/// `cipher_id` are the last negotiated values, not sums.
void merge_counters(endpoint_counters& into, endpoint_counters const& from) noexcept;

// ------------------------------------------------------------------------------------------------
// Non-template helpers (src/net/pool_support.cpp)
// ------------------------------------------------------------------------------------------------

/// @brief TCP_NODELAY, SO_KEEPALIVE + TCP_KEEPIDLE/INTVL/CNT, TCP_USER_TIMEOUT (ARCHITECTURE
/// §4.4/§4.5), optional SO_BINDTODEVICE @p bind_iface. No SO_RCVBUF, no SO_BUSY_POLL (§11).
/// @return {} or the errno of the first failing option (SO_BINDTODEVICE failures are reported
/// separately through @p bind_failed and do not fail the socket).
[[nodiscard]] std::error_code apply_socket_options(int fd,
                                                   connection_config const& cfg,
                                                   std::string_view bind_iface,
                                                   bool& bind_failed) noexcept;

/// @brief Cap (bound 0, unbound = cfg.iowq_max_unbound, 0 = auto = 16) and affine (cfg.iowq_cpus,
/// else the calling reactor thread's NUMA-local allowed CPUs minus the reactor's own CPU when it
/// is pinned to one) the ring's io-wq (uring_loop.hpp `apply_iowq_placement`). Called once per
/// pool before the first receive of an io-wq endpoint.
[[nodiscard]] iowq_placement apply_iowq_placement(io_uring& ring,
                                                  connection_config const& cfg) noexcept;

/// @brief Map a negative CQE errno of a stream op to the error the driver sees.
[[nodiscard]] std::error_code map_socket_errno(int err) noexcept;

/// One WAITALL-watchdog probe of a socket (gpuio ARCHITECTURE §4.2 rule 4, gpuio T13b amendment).
struct rx_stall_sample {
  std::uint64_t bytes_received{0};  ///< TCP_INFO `tcpi_bytes_received` (Linux >= 4.1)
  int inq{0};                       ///< FIONREAD: bytes queued in the TCP receive queue
};
/// @brief Probe @p fd for the KeyUpdate-under-WAITALL hang signature: no POLLIN while bytes are
/// queued (kTLS masks POLLIN for a pending KeyUpdate and for an incomplete record alike), plus
/// the TCP receive counter that tells a stalled socket from a slow-but-alive one. Three syscalls
/// (poll, FIONREAD, TCP_INFO); called from the pool tick only.
/// @return true and @p out filled when the socket is a candidate (no POLLIN, FIONREAD > 0);
/// false when it is readable, has nothing queued, or a probe failed (not the hang).
[[nodiscard]] bool sample_rx_stall(int fd, rx_stall_sample& out) noexcept;

/// Test-only switches (never set in production).
struct pool_debug_options {
  /// Body receives ask for min(destination space, cap) instead of min(remaining, cap): proves the
  /// over-read guard matters (the receive then completes only when the peer closes).
  bool disable_overread_guard{false};
};

// ------------------------------------------------------------------------------------------------
// The pool
// ------------------------------------------------------------------------------------------------

/// The user-space stream type of a TLS seam (`Tls::stream`), or `no_tls`.
template <class Tls>
struct tls_stream_of {
  using type = no_tls;
};
template <class Tls>
  requires requires { typename Tls::stream; }
struct tls_stream_of<Tls> {
  using type = typename Tls::stream;
};

/// kTLS control records accepted in a row before the connection is failed (tickets, KeyUpdates;
/// gpuio tls_record_rx's 32 cap, here on the kTLS path, where the kernel does not count them).
inline constexpr std::uint8_t max_control_records = 32;

template <class Env, class Handler, class Tls = no_tls>
class connection_pool {
  static constexpr bool has_tls      = !std::is_same_v<Tls, no_tls>;
  using Fallback                     = typename tls_stream_of<Tls>::type;
  static constexpr bool has_fallback = has_tls && fallback_stream<Fallback>;
  /// gpuio T16: the fallback can consume control records on an idle connection (optional hooks).
  static constexpr bool fallback_idle_drain =
    has_fallback && requires(Fallback& f, io_uring_sqe* sqe, int res) {
      { f.prep_idle_recv(sqe) } -> std::same_as<void>;
      { f.on_idle_recv(res) } -> std::same_as<bool>;
    };
  /// gpuio T19b (security review F5): the fallback can tell whether bytes past the response remain
  /// (optional hooks; checked at check-in, the slot is closed otherwise).
  static constexpr bool fallback_reuse_check =
    has_fallback && requires(Fallback& f, Fallback const& cf) {
      { cf.clean_for_reuse() } -> std::same_as<bool>;
      { f.drain_for_reuse() } -> std::same_as<bool>;
    };
  /// G5: the fallback can tell a clean peer close (close_notify / FIN) from a desync (optional).
  static constexpr bool fallback_peer_closed = has_fallback && requires(Fallback const& cf) {
    { cf.peer_closed() } -> std::same_as<bool>;
  };

 public:
  /// @param rc context-wide endpoints, DNS and keep-warm policy (outlives the pool).
  connection_pool(Env env, Handler& h, remote_context& rc) noexcept
    : _env(std::move(env)), _h(&h), _rc(&rc)
  {
    _env.bind(this);
    _op_cap = _env.op_capacity();
    _ops.reset(new (std::nothrow) op_rec[_op_cap]());
    _eps.reset(new (std::nothrow) ep_table[remote_context::max_endpoints]());
    if (!_ops || !_eps) _op_cap = 0;
  }
  ~connection_pool()
  {
    // No env calls here: the ring may already be gone (gpuio T9 destroys backends after closing
    // it).
    if (_eps)
      for (std::uint32_t e = 0; e < remote_context::max_endpoints; ++e) {
        ep_table& ep = _eps[e];
        for (std::uint32_t i = 0; i < ep.capacity; ++i)
          release_socket(ep.slots[i]);
      }
  }
  connection_pool(connection_pool const&)            = delete;
  connection_pool& operator=(connection_pool const&) = delete;

  [[nodiscard]] bool valid() const noexcept { return _op_cap != 0; }
  [[nodiscard]] Env& env() noexcept { return _env; }
  [[nodiscard]] remote_context& context() const noexcept { return *_rc; }
  [[nodiscard]] pool_debug_options& debug() noexcept { return _debug; }

  // ------------------------------------------------------------------------------- endpoints --

  /// @brief Create the per-endpoint slot table (the endpoint's only allocation on this reactor:
  /// `max_connections_per_reactor` slots). Called implicitly by checkout/connect_ahead/warmup.
  /// @return false for an unknown endpoint or out of memory.
  [[nodiscard]] bool prepare(std::uint32_t endpoint) noexcept { return table(endpoint) != nullptr; }

  /// @brief Take an idle, fresh connection of @p endpoint for one exchange (B.6). Idle slots are
  /// LIFO (the warmest first, curl §9d rule 6); one idle longer than `reuse_idle_max` (4 s: S3
  /// closes at ~5 s, dev_box_probes §7) is closed instead (§4.5 mechanism 2). Non-urgent callers
  /// cannot use the last `reserved` connections (latency class, s3_line_rate §3.2). The slot's
  /// two op tokens (receive, send) are taken with @p p and held until check_in/close.
  /// @return the slot (state busy), or nullptr: no idle connection (call connect_ahead), the
  /// reserve, or no free op slot.
  [[nodiscard]] conn_slot* checkout(std::uint32_t endpoint, priority p) noexcept
  {
    ep_table* ep = table(endpoint);
    if (ep == nullptr || _stopping) return nullptr;
    if (p != priority::urgent && ep->busy_nonurgent >= ep->limit - ep->desc->reserved) {
      ++ep->ctr.reserve_denials;
      return nullptr;
    }
    std::uint64_t const now      = _env.now_ns();
    std::uint64_t const max_idle = ms_ns(ep->desc->cfg.conn.reuse_idle_max);
    while (ep->idle_head != no_conn) {
      conn_slot& c = ep->slots[ep->idle_head];
      if (now - c.idle_since_ns > max_idle) {
        // Everything below the top is older still: the whole list is past the cutoff.
        unlink_idle(*ep, c);
        ++ep->ctr.idle_cutoff_closes;
        close_conn(*ep, c);
        continue;
      }
      std::uint32_t io = no_op, tx = no_op;
      if (!_env.take_op(p, false, io)) {
        ++ep->ctr.op_waits;
        return nullptr;
      }
      if (!_env.take_op(p, false, tx)) {
        _env.release_op(io);
        ++ep->ctr.op_waits;
        return nullptr;
      }
      unlink_idle(*ep, c);
      ++c.generation;  // the idle period's watch becomes stale (never cancelled)
      c.state    = conn_state::busy;
      c.prio     = p;
      c.pv.io_op = io;
      c.pv.tx_op = tx;
      _ops[io]   = op_rec{&c, c.generation, op_role::io};
      _ops[tx]   = op_rec{&c, c.generation, op_role::tx};
      _held += 2;
      begin_exchange(*ep, c, now);
      ++ep->n_busy;
      if (p != priority::urgent) ++ep->busy_nonurgent;
      ++ep->ctr.checkouts;
      if (c.reused) ++ep->ctr.reused_checkouts;
      maybe_replace(*ep, c, p);
      return &c;
    }
    ++ep->ctr.checkout_misses;
    return nullptr;
  }

  /// @brief Open up to @p n more connections for waiting requests (§4.5 mechanism 4; B.6
  /// `connect_ahead`), bounded by free slots and `max_concurrent_connects` (the rest queue and
  /// start as earlier connects finish). Ops use @p p and keep run() alive. @return accepted.
  unsigned connect_ahead(std::uint32_t endpoint, unsigned n, priority p = priority::normal) noexcept
  {
    ep_table* ep = table(endpoint);
    if (ep == nullptr || _stopping || n == 0) return 0;
    std::uint32_t const live = ep->live_or_pending();
    unsigned const room      = live >= ep->limit ? 0u : ep->limit - live;
    unsigned const take      = std::min(n, room);
    ep->backlog_demand += take;
    if (p == priority::urgent || ep->backlog_prio == priority::bulk) ep->backlog_prio = p;
    ep->ctr.connect_ahead += take;
    start_backlog(*ep);
    return take;
  }

  /// @brief Background warm-up (API.md A.6 `warmup`): open @p n connections (DNS, TCP, TLS,
  /// kTLS) that do not count as outstanding work, rate-limited per endpoint.
  void warmup(std::uint32_t endpoint, unsigned n) noexcept
  {
    ep_table* ep = table(endpoint);
    if (ep == nullptr || _stopping || n == 0) return;
    std::uint32_t const live = ep->live_or_pending();
    unsigned const room      = live >= ep->limit ? 0u : ep->limit - live;
    unsigned const take      = std::min(n, room);
    ep->backlog_background += take;
    ep->ctr.warmup_requested += take;
    start_backlog(*ep);
  }

  /// @brief gpuio T20c (g): open ONE replacement for a connection about to reach the server's
  /// per-connection request limit (S3 closes after 100 requests), so the reconnect is off the
  /// critical path. It may use the spare slots beyond `max_connections_per_reactor` (the table
  /// holds max(1, max / 32) more); the old connection serves until the server closes it.
  /// @return 1 when a connection was started, 0 (a slot table without a free slot, stopping).
  unsigned connect_replacement(std::uint32_t endpoint, priority p = priority::normal) noexcept
  {
    ep_table* ep = table(endpoint);
    if (ep == nullptr || _stopping) return 0;
    if (ep->live_or_pending() >= ep->capacity) return 0;
    ep->backlog_demand += 1;
    if (p == priority::urgent || ep->backlog_prio == priority::bulk) ep->backlog_prio = p;
    ++ep->ctr.connect_ahead;
    ++ep->ctr.replacements;
    start_backlog(*ep);
    return 1;
  }

  /// @brief Warm @p endpoint up TO @p floor connections (API.md A.6 `set_keep_warm`, gpuio T15b):
  /// opens max(0, floor - live_or_pending), bounded by the slot table; idempotent (nothing is
  /// opened on an endpoint that already has @p floor connections live or being set up).
  void warmup_to(std::uint32_t endpoint, unsigned floor) noexcept
  {
    ep_table* ep = table(endpoint);
    if (ep == nullptr || _stopping) return;
    std::uint32_t const live = ep->live_or_pending();
    if (live < floor) warmup(endpoint, floor - live);
  }

  /// @return this pool's share of a per-context keep-warm floor: ceil(min_connections / max(1,
  /// live reactors)) (gpuio T13's warm-up share rule; Envs without `live_reactors()` count one).
  [[nodiscard]] unsigned keep_warm_share(unsigned min_connections) const noexcept
  {
    unsigned live = 1;
    if constexpr (requires(Env const& e) {
                    { e.live_reactors() } -> std::convertible_to<unsigned>;
                  })
      live = std::max(1u, _env.live_reactors());
    return (min_connections + live - 1) / live;
  }

  /// @brief End an exchange on a reusable connection (B.6 `on_idle`): release the op tokens;
  /// close instead when the driver set `close_after`, the byte cap / age was reached, or the pool
  /// is stopping; else timestamp, generation++, arm the idle watch, push on the idle LIFO.
  /// @pre no send/receive of @p c in flight.
  void check_in(conn_slot& c) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    CUCASCADE_NET_ASSERT(c.state == conn_state::busy && !c.pv.io_inflight && !c.pv.tx_inflight);
    if (c.state != conn_state::busy) return;  // misuse: not checked out
    if (c.pv.io_inflight || c.pv.tx_inflight) [[unlikely]] {
      close(c);  // misuse: ops still in flight; never pool such a slot
      return;
    }
    end_busy(ep, c);
    if (c.parser.header_done() && !c.meta.keep_alive) learn_peer_close(ep, c);
    ++c.requests_on_connection;
    std::uint64_t const now = _env.now_ns();
    bool const cap =
      ep.desc->max_connection_bytes != 0 && c.bytes_on_connection >= ep.desc->max_connection_bytes;
    bool const old = aged(ep, c, now);
    if (c.close_after || c.last_request || cap || old || _stopping) {
      if (cap) ++ep.ctr.byte_cap_closes;
      if (old) ++ep.ctr.age_closes;
      close_conn(ep, c);
      return;
    }
    if (!stream_drained(ep, c)) {
      close_conn(ep, c);
      return;
    }
    enter_idle(ep, c, now, now);
  }
  void on_idle(conn_slot& c) noexcept { check_in(c); }  ///< B.6 name

 private:
  /// gpuio T19b (security review F5): bytes past a response (plaintext pending in a user-space TLS
  /// stream, a record in its ring) would become the next response's head, and the idle watch
  /// cannot see them (the socket is empty). Complete control records are consumed; anything else
  /// makes the slot unusable (counted; the caller closes it). kTLS/plain: the idle watch covers it.
  [[nodiscard]] bool stream_drained(auto& ep, conn_slot& c) noexcept
  {
    if constexpr (fallback_reuse_check) {
      if (c.kind == kind_tls && c.tls != nullptr) {
        auto& f = *static_cast<Fallback*>(c.tls);
        if (!f.clean_for_reuse() && !f.drain_for_reuse()) {
          // A close_notify (or FIN) right behind the response is the peer's close, which kTLS
          // counts through the idle watch: the same counter here (G5 parity), not a desync.
          if constexpr (fallback_peer_closed) {
            if (f.peer_closed()) {
              ++ep.ctr.idle_fin_evictions;
              return false;
            }
          }
          ++ep.ctr.reuse_desync_closes;
          return false;
        }
      }
    }
    return true;
  }

 public:
  /// @brief Discard a checked-out connection (error, `Connection: close`, EOF framing, or an
  /// abandoned exchange). In-flight ops are cut short (shutdown(2) + cancel); the slot is freed
  /// once their CQEs arrive. No further callbacks for @p c.
  void close(conn_slot& c) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    if (c.state != conn_state::busy) return;  // already closing / not checked out
    if (c.parser.header_done() && !c.meta.keep_alive) {
      ++ep.ctr.connection_close_closes;              // gpuio T20c
      if (c.parser.done()) learn_peer_close(ep, c);  // complete response, then the server's close
    }
    end_busy_keep_tokens(ep, c);
    close_conn(ep, c);
  }

  // ------------------------------------------------------------------------------ stream I/O --

  /// @brief Send @p tx (head, or head + PUT body iovecs) completely: SEND/SENDMSG with
  /// MSG_NOSIGNAL, short sends continued. @p tx must stay valid until `on_send`.
  void send(conn_slot& c, std::span<iovec const> tx) noexcept
  {
    CUCASCADE_NET_ASSERT(!c.pv.tx_inflight);
    c.pv.tx_cursor = http::iovec_sink(tx);
    c.pv.tx_total  = c.pv.tx_cursor.capacity();
    c.pv.tx_done   = 0;
    ++_eps[c.endpoint].ctr.sends;
    issue_send(c);
  }

  /// @brief One-shot receive into `header_buf[hdr_len, header_buffer_bytes)`; `on_header`
  /// reports the bytes appended (control records are drained and replayed transparently).
  /// @param after_send IORING_RECVSEND_POLL_FIRST (the response cannot be there yet).
  /// @return errc::http_header_too_large when the buffer is full (nothing issued).
  [[nodiscard]] std::error_code recv_header(conn_slot& c, bool after_send = false) noexcept
  {
    CUCASCADE_NET_ASSERT(!c.pv.io_inflight && !c.pv.rx_active);
    if (c.pv.rx_active || c.pv.io_inflight) return make_error_code(errc::invalid_argument);
    if (header_space(c) == 0) return make_error_code(errc::http_header_too_large);
    c.pv.rx_active     = true;
    c.pv.tx_poll_first = after_send;
    c.pv.drain_len     = 0;
    issue_header(c);
    return {};
  }

  /// @brief Receive the next body piece into `c.sink`: ONE receive of
  /// len = min(@p http_remaining, body_recv_cap) (rule 1), RECV|WAITALL for a contiguous window,
  /// RECVMSG|WAITALL without cmsg for a scattered one, never linked (rule 2). `on_body` reports
  /// the bytes placed (a short CQE is progress, rule 3; the following control record is drained,
  /// replayed and the receive re-issued transparently).
  /// @param http_remaining body bytes still due (parser.body_remaining()); must be > 0 and known.
  /// @return errc::http_length_mismatch when the destination has no room (nothing issued).
  [[nodiscard]] std::error_code recv_body(conn_slot& c, std::uint64_t http_remaining) noexcept
  {
    CUCASCADE_NET_ASSERT(!c.pv.io_inflight && http_remaining != 0 &&
                         http_remaining != http::unknown_length);
    if (c.pv.rx_active || c.pv.io_inflight) return make_error_code(errc::invalid_argument);
    if (http_remaining == 0 || http_remaining == http::unknown_length || c.sink.remaining() == 0)
      return make_error_code(errc::http_length_mismatch);
    c.pv.rx_remaining = http_remaining;
    c.pv.drain_len    = 0;
    auto ec           = issue_body(c);
    c.pv.rx_active    = !ec;
    return ec;
  }

  /// @brief Cut the slot's in-flight send/receive short (stall / attempt deadline, request
  /// cancel): ASYNC_CANCEL; the CQEs still arrive (a WAITALL receive cancelled after progress
  /// reports the bytes already placed, net.c:1600) with errc::operation_canceled.
  void cancel(conn_slot& c) noexcept
  {
    c.pv.cancel_requested = true;
    ++_eps[c.endpoint].ctr.cancels;
    if (c.pv.io_inflight) _env.cancel_op(c.pv.io_op);
    if (c.pv.tx_inflight) _env.cancel_op(c.pv.tx_op);
  }

  // ---------------------------------------------------------------------------- reactor hooks --

  /// @brief CQE of an op taken by this pool (from `Env`). The only entry point of completions.
  void on_op_cqe(std::uint32_t token, int res, unsigned flags) noexcept
  {
    ++_cqes;
    op_rec const r = _ops[token];
    conn_slot* c   = r.c;
    if (c == nullptr) [[unlikely]] {  // not ours any more (cannot happen; keep the slot sane)
      _env.release_op(token);
      return;
    }
    ++_eps[c->endpoint].ctr.cqes;
    switch (r.role) {
      case op_role::watch: on_watch_cqe(token, *c, r.gen, res); return;
      case op_role::tx:
        c->pv.tx_inflight = false;
        on_tx_cqe(*c, res);
        return;
      case op_role::io:
        c->pv.io_inflight = false;
        on_io_cqe(*c, res, flags);
        return;
    }
  }
  /// B.6 shape (the env routes by token; kind/slot/gen are carried in user_data by the reactor).
  void on_cqe(ud_kind, std::uint32_t token, std::uint32_t, int res, unsigned flags) noexcept
  {
    on_op_cqe(token, res, flags);
  }

  /// @brief Periodic upkeep (armed on the reactor heap while connections exist, every
  /// `tick_period`; B.6 `tick`): idle cutoff, keep-warm probes and floor, queued connects.
  void tick() noexcept
  {
    std::uint64_t const now = _env.now_ns();
    // Endpoints with a keep-warm floor get their table even before the first request.
    for (std::uint32_t e = 0, n = _rc->endpoint_count(); e < n && !_stopping; ++e)
      if (_eps[e].capacity == 0 && _rc->keep_warm(e).min_connections != 0) std::ignore = table(e);
    for (std::uint32_t e = 0; e < remote_context::max_endpoints; ++e) {
      ep_table& ep = _eps[e];
      if (ep.capacity == 0) continue;
      // Idle cutoff (§4.5 mechanism 2): close idle > reuse_idle_max.
      std::uint64_t const max_idle = ms_ns(ep.desc->cfg.conn.reuse_idle_max);
      for (std::uint32_t i = ep.idle_head; i != no_conn;) {
        conn_slot& c = ep.slots[i];
        i            = c.pv.next;
        if (now - c.idle_since_ns > max_idle) {
          unlink_idle(ep, c);
          ++ep.ctr.idle_cutoff_closes;
          close_conn(ep, c);
        }
      }
      watchdog(ep, now);
      keep_warm(ep, now);
      start_backlog(ep);
    }
    _tick_armed = false;
    if (need_tick()) ensure_tick();
  }
  void tick(std::uint64_t) noexcept { tick(); }  ///< B.6 shape

  /// @brief Stop: no new checkouts/connects, queued connects dropped, idle connections closed,
  /// every in-flight op cancelled (and its socket shut down so it completes promptly). Busy
  /// exchanges with an op in flight still get their (failing) callbacks; busy slots with nothing
  /// in flight are closed at once without a callback. The owner then reaps CQEs (the reactor's
  /// sync cancel + drain, or a test loop) until `quiescent()`; only then may the pool be
  /// destroyed while the ring lives.
  void shutdown() noexcept
  {
    if (_stopping) return;
    _stopping = true;
    if (_tick_armed) _tick_armed = !_env.cancel_deadline(_tick);
    if (_dns_armed) _dns_armed = !_env.cancel_deadline(_dns_poll);
    for (std::uint32_t e = 0; e < remote_context::max_endpoints; ++e) {
      ep_table& ep      = _eps[e];
      ep.backlog_demand = ep.backlog_background = 0;
      for (std::uint32_t i = 0; i < ep.capacity; ++i) {
        conn_slot& c = ep.slots[i];
        switch (c.state) {
          case conn_state::idle:
            unlink_idle(ep, c);
            close_conn(ep, c);
            break;
          case conn_state::resolving: fail_connect(ep, c, errc::context_stopped); break;
          case conn_state::busy:
            if (!c.pv.io_inflight && !c.pv.tx_inflight) {
              // Checked out but between operations: nothing will call back, close it now (the
              // driver must not use slots it held across shutdown()).
              close(c);
              break;
            }
            [[fallthrough]];
          case conn_state::connecting:
          case conn_state::handshaking:
          case conn_state::draining:
          case conn_state::probing:
            c.pv.cancel_requested = true;
            if (c.fd >= 0) ::shutdown(c.fd, SHUT_RDWR);
            if (c.pv.io_inflight) _env.cancel_op(c.pv.io_op);
            if (c.pv.tx_inflight) _env.cancel_op(c.pv.tx_op);
            break;
          case conn_state::free:
          case conn_state::closing: break;
        }
      }
    }
  }
  /// @return true once every op token this pool took has been returned (every CQE reaped).
  [[nodiscard]] bool quiescent() const noexcept { return _held == 0; }
  [[nodiscard]] bool stopping() const noexcept { return _stopping; }
  /// @return op tokens currently held (in flight or reserved by busy slots).
  [[nodiscard]] std::uint32_t ops_held() const noexcept { return _held; }
  [[nodiscard]] std::uint64_t cqes() const noexcept { return _cqes; }

  // ---------------------------------------------------------------------------- introspection --

  [[nodiscard]] endpoint_counters const& counters(std::uint32_t endpoint) const noexcept
  {
    static endpoint_counters const none{};
    if (endpoint >= remote_context::max_endpoints || _eps[endpoint].capacity == 0) return none;
    return _eps[endpoint].ctr;
  }
  [[nodiscard]] pool_gauges gauges(std::uint32_t endpoint) const noexcept
  {
    pool_gauges g{};
    if (endpoint >= remote_context::max_endpoints) return g;
    ep_table const& ep = _eps[endpoint];
    g.capacity         = ep.capacity;  // slots: max_connections_per_reactor + replacement spares
    for (std::uint32_t i = 0; i < ep.capacity; ++i) {
      switch (ep.slots[i].state) {
        case conn_state::free: ++g.free; break;
        case conn_state::resolving: ++g.resolving; break;
        case conn_state::connecting: ++g.connecting; break;
        case conn_state::handshaking: ++g.handshaking; break;
        case conn_state::idle: ++g.idle; break;
        case conn_state::busy: ++g.busy; break;
        case conn_state::draining: ++g.draining; break;
        case conn_state::probing: ++g.probing; break;
        case conn_state::closing: ++g.closing; break;
      }
    }
    return g;
  }
  /// @return idle connections ready for checkout.
  [[nodiscard]] std::uint32_t idle_count(std::uint32_t endpoint) const noexcept
  {
    return endpoint < remote_context::max_endpoints ? _eps[endpoint].n_idle : 0;
  }
  /// @return connections being set up plus queued connects.
  /// @return the server's per-connection request limit learned on this pool (0 = unknown).
  [[nodiscard]] std::uint32_t server_request_limit(std::uint32_t endpoint) const noexcept
  {
    return endpoint < remote_context::max_endpoints ? _eps[endpoint].server_limit : 0;
  }
  [[nodiscard]] std::uint32_t pending_connects(std::uint32_t endpoint) const noexcept
  {
    if (endpoint >= remote_context::max_endpoints) return 0;
    ep_table const& ep = _eps[endpoint];
    return ep.n_setup + ep.backlog_demand + ep.backlog_background;
  }
  /// @return the slot table of @p endpoint (tests, stats).
  [[nodiscard]] std::span<conn_slot> slots(std::uint32_t endpoint) noexcept
  {
    if (endpoint >= remote_context::max_endpoints) return {};
    return {_eps[endpoint].slots.get(), _eps[endpoint].capacity};
  }
  [[nodiscard]] iowq_placement const& iowq() const noexcept { return _iowq; }
  [[nodiscard]] bool iowq_applied() const noexcept { return _iowq_applied; }
  /// @brief Add the counters of @p endpoint into @p out (aggregation across reactors).
  void accumulate(endpoint_counters& out, std::uint32_t endpoint) const noexcept
  {
    if (endpoint >= remote_context::max_endpoints || _eps[endpoint].capacity == 0) return;
    merge_counters(out, _eps[endpoint].ctr);
  }

  /// Tick period: the idle cutoff and keep-warm need ~1 s resolution (§4.5).
  static constexpr std::uint64_t tick_period_ns = 250'000'000ULL;
  /// DNS wait poll while a slot waits for its host's first answer.
  static constexpr std::uint64_t dns_poll_ns = 1'000'000ULL;

 private:
  enum class op_role : std::uint8_t { io, tx, watch };
  struct op_rec {
    conn_slot* c{nullptr};
    std::uint32_t gen{0};
    op_role role{op_role::io};
  };

  struct ep_table {
    endpoint_desc const* desc{nullptr};
    std::unique_ptr<conn_slot[]> slots;
    std::uint32_t capacity{0};  ///< slots: limit + spare replacement slots (gpuio T20c (g))
    std::uint32_t limit{0};     ///< max_connections_per_reactor: the live limit of every opener
    std::uint32_t free_head{no_conn}, idle_head{no_conn};
    std::uint32_t n_idle{0}, n_busy{0}, busy_nonurgent{0}, n_setup{0}, n_resolving{0};
    std::uint32_t n_aux{0};  ///< draining + probing (live, not checkout-able)
    std::uint32_t backlog_demand{0}, backlog_background{0};
    priority backlog_prio{priority::bulk};
    std::uint32_t bind_rr{0};
    std::uint32_t server_limit{0};  ///< gpuio T20c (g): learned per-connection request limit
    endpoint_counters ctr{};
    [[nodiscard]] std::uint32_t live_or_pending() const noexcept
    {
      return n_idle + n_busy + n_setup + n_aux + backlog_demand + backlog_background;
    }
  };

  static constexpr std::uint64_t ms_ns(std::chrono::milliseconds d) noexcept
  {
    return d.count() <= 0 ? 0 : static_cast<std::uint64_t>(d.count()) * 1'000'000ULL;
  }

  // ------------------------------------------------------------------------------- tables ----

  ep_table* table(std::uint32_t endpoint) noexcept
  {
    if (_op_cap == 0 || endpoint >= remote_context::max_endpoints) return nullptr;
    ep_table& ep = _eps[endpoint];
    if (ep.capacity != 0) return &ep;
    endpoint_desc const* d = _rc->endpoint(endpoint);
    if (d == nullptr) return nullptr;
    std::uint32_t const lim = std::max(1u, d->cfg.conn.max_connections_per_reactor);
    std::uint32_t const cap = lim + std::max(1u, lim / 32);  // gpuio T20c (g): replacement spares
    ep.slots.reset(new (std::nothrow) conn_slot[cap]);
    if (!ep.slots) return nullptr;
    ep.desc     = d;
    ep.capacity = cap;
    ep.limit    = lim;
    for (std::uint32_t i = cap; i-- > 0;) {
      conn_slot& c = ep.slots[i];
      c.index      = i;
      c.endpoint   = endpoint;
      c.pv.pool    = this;
      c.pv.next    = ep.free_head;
      ep.free_head = i;
    }
    return &ep;
  }

  conn_slot* pop_free(ep_table& ep) noexcept
  {
    if (ep.free_head == no_conn) return nullptr;
    conn_slot& c = ep.slots[ep.free_head];
    ep.free_head = c.pv.next;
    c.pv.next    = no_conn;
    return &c;
  }
  void push_free(ep_table& ep, conn_slot& c) noexcept
  {
    c.pv.next    = ep.free_head;
    ep.free_head = c.index;
  }
  void push_idle(ep_table& ep, conn_slot& c) noexcept
  {
    c.pv.next    = ep.idle_head;
    ep.idle_head = c.index;
    ++ep.n_idle;
  }
  void unlink_idle(ep_table& ep, conn_slot& c) noexcept
  {
    std::uint32_t* link = &ep.idle_head;
    while (*link != no_conn && *link != c.index)
      link = &ep.slots[*link].pv.next;
    if (*link == c.index) {
      *link     = c.pv.next;
      c.pv.next = no_conn;
      --ep.n_idle;
    }
  }

  bool take(conn_slot& c, std::uint32_t& tok, priority p, bool background, op_role role) noexcept
  {
    if (!_env.take_op(p, background, tok)) {
      tok = no_op;
      return false;
    }
    _ops[tok] = op_rec{&c, c.generation, role};
    ++_held;
    return true;
  }
  void put(std::uint32_t& tok) noexcept
  {
    if (tok == no_op) return;
    _ops[tok] = op_rec{};
    _env.release_op(tok);
    --_held;
    tok = no_op;
  }

  [[nodiscard]] bool aged(ep_table const& ep, conn_slot const& c, std::uint64_t now) const noexcept
  {
    auto const age = ep.desc->cfg.max_connection_age;
    return age.count() > 0 &&
           now - c.created_ns >= static_cast<std::uint64_t>(age.count()) * 1'000'000'000ULL;
  }

  void begin_exchange(ep_table& ep, conn_slot& c, std::uint64_t now) noexcept
  {
    c.reused       = c.requests_on_connection != 0;
    c.first_byte   = false;
    c.close_after  = false;
    c.last_request = (ep.desc->max_connection_bytes != 0 &&
                      c.bytes_on_connection >= ep.desc->max_connection_bytes) ||
                     aged(ep, c, now);
    c.hdr_len     = 0;
    c.exchange_rx = 0;
    c.exchange_tx = 0;
    c.owner       = nullptr;
    c.owner_tag   = 0;
    c.parser.reset();
    c.meta = http::response_meta{};
    c.sink = http::iovec_sink{};
    reset_exchange_flags(c);
  }

  static void reset_exchange_flags(conn_slot& c) noexcept
  {
    c.pv.cancel_requested = false;
    c.pv.reply_pending    = false;
    c.pv.rx_active        = false;
    c.pv.rx_waitall       = false;
    c.pv.watchdog         = false;
    c.pv.wd_armed         = false;
    c.pv.oneshot_next     = false;
    c.pv.tx_poll_first    = false;
    c.pv.drain_len        = 0;
    c.pv.io               = io_kind::none;
  }

  void end_busy(ep_table& ep, conn_slot& c) noexcept
  {
    end_busy_keep_tokens(ep, c);
    put(c.pv.io_op);
    put(c.pv.tx_op);
  }
  void end_busy_keep_tokens(ep_table& ep, conn_slot& c) noexcept
  {
    --ep.n_busy;
    if (c.prio != priority::urgent) --ep.busy_nonurgent;
  }

  // ------------------------------------------------------------- server request limit -----

  /// gpuio T20c (g): a complete response that ended with `Connection: close` (S3 closes a
  /// connection after exactly 100 requests): learn the server's per-connection request limit after
  /// two.
  void learn_peer_close(ep_table& ep, conn_slot const& c) noexcept
  {
    ep.ctr.peer_close_requests += std::uint64_t{c.requests_on_connection} + 1;
    if (++ep.ctr.peer_closes >= 2) {
      std::uint64_t const avg =
        (ep.ctr.peer_close_requests + ep.ctr.peer_closes / 2) / ep.ctr.peer_closes;
      ep.server_limit             = static_cast<std::uint32_t>(std::min<std::uint64_t>(avg, 65535));
      ep.ctr.server_request_limit = ep.server_limit;
    }
  }

  /// gpuio T20c (g): a connection that starts request min(ceil(0.95 x limit), limit - 1) gets its
  /// replacement pre-opened unless an idle spare (an idle connection not itself near the limit)
  /// or a pending connect already covers it -- counted over the slot table, once per connection
  /// crossing the threshold -- so the reconnect after the server's `Connection: close` is off the
  /// critical path (I3 Step 18: admission_waits 63k -> 896). The old connection is never closed
  /// early. (gpuio ran this in the backend at attempt start, right after the checkout.)
  void maybe_replace(ep_table& ep, conn_slot const& c, priority p) noexcept
  {
    std::uint32_t const lim = ep.server_limit;
    if (lim < 2) return;
    std::uint32_t const thr = std::min((lim * 95 + 99) / 100, lim - 1);
    if (std::uint32_t{c.requests_on_connection} + 1 != thr) return;
    std::uint32_t expiring = 0, spares = 0;
    for (std::uint32_t i = 0; i < ep.capacity; ++i) {
      conn_slot const& sl = ep.slots[i];
      bool const near     = std::uint32_t{sl.requests_on_connection} + 1 >= thr;
      if (sl.state == conn_state::busy && near) ++expiring;
      if (sl.state == conn_state::idle) (near ? expiring : spares) += 1;
    }
    if (spares + pending_connects(c.endpoint) < expiring)
      std::ignore = connect_replacement(c.endpoint, p);
  }

  // -------------------------------------------------------------------------------- idle -----

  /// @param idle_since keep the original timestamp after an idle drain (the server's idle timer
  /// did not restart because it sent us a ticket).
  void enter_idle(ep_table& ep, conn_slot& c, std::uint64_t now, std::uint64_t idle_since) noexcept
  {
    (void)now;
    c.state         = conn_state::idle;
    c.idle_since_ns = idle_since;
    ++c.generation;
    c.pv.io        = io_kind::none;
    c.pv.drain_len = 0;
    push_idle(ep, c);
    if (ep.desc->cfg.conn.watch_idle_fin) arm_watch(ep, c);
    ensure_tick();
  }

  void arm_watch(ep_table& ep, conn_slot& c) noexcept
  {
    std::uint32_t tok = no_op;
    if (!take(c, tok, priority::normal, true, op_role::watch)) {
      ++ep.ctr.watch_skipped;  // no op slot: the stale-reuse retry still covers this connection
      return;
    }
    c.pv.watch_op = tok;
    prep_idle_watch(_env.get_sqe(tok, ud_kind::idle_watch), c.fd);
    ++ep.ctr.watch_armed;
  }

  void on_watch_cqe(std::uint32_t tok, conn_slot& c, std::uint32_t gen, int res) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    if (c.pv.watch_op == tok) c.pv.watch_op = no_op;
    if (gen != c.generation || c.state != conn_state::idle) {
      ++ep.ctr.watch_stale;  // the slot was reused / closed since: response byte or our shutdown
      put(tok);
      return;
    }
    switch (classify_idle_watch(res)) {
      case watch_verdict::fin:
        put(tok);
        ++ep.ctr.idle_fin_evictions;
        unlink_idle(ep, c);
        close_conn(ep, c);
        return;
      case watch_verdict::failed:
        put(tok);
        unlink_idle(ep, c);
        close_conn(ep, c);
        return;
      case watch_verdict::readable:
        if constexpr (fallback_idle_drain) {
          if (c.kind == kind_tls && !_stopping) {
            // gpuio T16: the same tickets on a user-space TLS connection: the stream receives them
            // into its ciphertext ring and lets its record opener consume them (gpuio
            // CHANGE_REQUESTS "gpuio T16 — idle").
            ++ep.ctr.idle_readable_drains;
            unlink_idle(ep, c);
            c.state = conn_state::draining;
            ++ep.n_aux;
            _ops[tok]      = op_rec{&c, c.generation, op_role::io};
            c.pv.io_op     = tok;
            c.pv.drain_len = 0;
            c.pv.io        = io_kind::drain_idle;
            static_cast<Fallback*>(c.tls)->prep_idle_recv(_env.get_sqe(tok, ud_kind::op));
            c.pv.io_inflight = true;
            return;
          }
        }
        if (c.kind == kind_ktls && !_stopping) {
          // TLS 1.3 servers send NewSessionTickets right after the handshake (nginx, MinIO): the
          // idle socket turns readable. Drain + replay them with the watch's own token, then
          // watch again. Anything else readable on an idle connection means it is dead.
          ++ep.ctr.idle_readable_drains;
          unlink_idle(ep, c);
          c.state = conn_state::draining;
          ++ep.n_aux;
          _ops[tok]      = op_rec{&c, c.generation, op_role::io};
          c.pv.io_op     = tok;
          c.pv.drain_len = 0;
          issue_drain(c, io_kind::drain_idle);
          return;
        }
        put(tok);
        unlink_idle(ep, c);
        close_conn(ep, c);
        return;
    }
  }

  // ----------------------------------------------------------------------------- connecting --

  void start_backlog(ep_table& ep) noexcept
  {
    unsigned const limit = std::max(1u, ep.desc->cfg.max_concurrent_connects);
    while (!_stopping && ep.n_setup < limit && (ep.backlog_demand + ep.backlog_background) != 0) {
      bool const demand = ep.backlog_demand != 0;
      priority const p  = demand ? ep.backlog_prio : priority::normal;
      if (!start_connect(ep, p, !demand)) break;
      if (demand) {
        --ep.backlog_demand;
        if (ep.backlog_demand == 0) ep.backlog_prio = priority::bulk;
      } else {
        --ep.backlog_background;
        ++ep.ctr.warmup_started;
      }
    }
  }

  bool start_connect(ep_table& ep, priority p, bool background) noexcept
  {
    conn_slot* cp = pop_free(ep);
    if (cp == nullptr) return false;
    conn_slot& c = *cp;
    ++c.generation;
    if (!take(c, c.pv.io_op, p, background, op_role::io)) {
      ++ep.ctr.op_waits;
      push_free(ep, c);
      return false;
    }
    c.kind       = kind_connecting;
    c.state      = conn_state::connecting;
    c.prio       = p;
    c.background = background;
    c.fd         = -1;
    c.session.reset();
    c.rekey                  = nullptr;
    c.tls                    = nullptr;
    c.requests_on_connection = 0;
    c.bytes_on_connection    = 0;
    reset_exchange_flags(c);
    c.pv.io_inflight = c.pv.tx_inflight = false;
    c.pv.tx_op                          = no_op;
    c.pv.bound_dns                      = false;
    c.pv.last_errno                     = 0;
    c.pv.control_run                    = 0;
    ++ep.n_setup;
    ++ep.ctr.connects_started;
    std::uint64_t const now = _env.now_ns();
    arm_conn_deadline(c, now + ms_ns(ep.desc->cfg.conn.connect_timeout));
    if (!resolve_peer(ep, c, now, false)) return true;  // waiting for DNS (or its failure)
    issue_socket(c);
    return true;
  }

  /// @return true when the peer address is known (else the slot waits or has failed).
  /// @param from_poll failures are reported only from the DNS poll deadline, never synchronously
  /// inside an API call (connect_ahead / warmup / checkout must not re-enter the handler).
  bool resolve_peer(ep_table& ep, conn_slot& c, std::uint64_t now, bool from_poll) noexcept
  {
    endpoint_desc const& d = *ep.desc;
    if (d.literal) {
      c.peer     = d.literal_addr;
      c.peer_len = d.literal_len;
      return true;
    }
    switch (_rc->dns().pick(d.dns, c.peer, c.peer_len, now)) {
      case dns_resolver::status::ok: c.pv.bound_dns = true; return true;
      case dns_resolver::status::failed:
        if (from_poll) break;
        [[fallthrough]];
      case dns_resolver::status::pending:
        if (c.state != conn_state::resolving) {
          c.state = conn_state::resolving;
          ++ep.n_resolving;
          ++ep.ctr.dns_waits;
        }
        arm_dns_poll();
        return false;
    }
    ++ep.ctr.dns_failures;
    fail_connect(ep, c, errc::dns_failed);
    return false;
  }

  void issue_socket(conn_slot& c) noexcept
  {
    c.state           = conn_state::connecting;
    c.pv.io           = io_kind::socket;
    io_uring_sqe* sqe = _env.get_sqe(c.pv.io_op, ud_kind::op);
    io_uring_prep_socket(
      sqe, c.peer.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, IPPROTO_TCP, 0);
    c.pv.io_inflight = true;
  }

  void on_socket(ep_table& ep, conn_slot& c, int res) noexcept
  {
    if (c.pv.cancel_requested) {
      if (res >= 0) ::close(res);
      fail_connect(ep, c, timeout_or_stop(errc::connect_timeout));
      return;
    }
    if (res == -EINVAL || res == -EOPNOTSUPP) {  // kernel without IORING_OP_SOCKET (< 5.19)
      res = ::socket(c.peer.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, IPPROTO_TCP);
      if (res < 0) res = -errno;
      ++ep.ctr.socket_fallbacks;
    }
    if (res < 0) {
      c.pv.last_errno = -res;
      fail_connect(ep, c, errc::connect_failed);
      return;
    }
    c.fd            = res;
    auto const& ifs = ep.desc->cfg.conn.bind_interfaces;
    std::string_view iface{};
    if (!ifs.empty()) iface = ifs[ep.bind_rr++ % ifs.size()];  // round-robin (multi-NIC)
    bool bind_failed = false;
    if (auto ec = apply_socket_options(c.fd, ep.desc->cfg.conn, iface, bind_failed)) {
      c.pv.last_errno = ec.value();
      fail_connect(ep, c, errc::connect_failed);
      return;
    }
    if (bind_failed) ++ep.ctr.bind_failures;
    c.pv.io           = io_kind::connect;
    io_uring_sqe* sqe = _env.get_sqe(c.pv.io_op, ud_kind::op);
    io_uring_prep_connect(sqe, c.fd, reinterpret_cast<sockaddr const*>(&c.peer), c.peer_len);
    c.pv.io_inflight = true;
  }

  void on_connect(ep_table& ep, conn_slot& c, int res) noexcept
  {
    if (res < 0 || c.pv.cancel_requested) {
      bool const timed_out = c.pv.cancel_requested;
      if (c.pv
            .bound_dns)  // failed / slow front end: prefer another one for 30 s (s3_line_rate §2.7)
        _rc->dns().bench(ep.desc->dns, c.peer, _rc->dns().config().bench_time, _env.now_ns());
      c.pv.last_errno = res < 0 ? -res : 0;
      fail_connect(
        ep, c, timed_out ? timeout_or_stop(errc::connect_timeout) : errc::connect_failed);
      return;
    }
    if (!ep.desc->cfg.tls) {
      c.plain = plain_stream(c.fd, &c.msg, ep.desc->stream);
      c.kind  = kind_plain;
      become_ready(ep, c);
      return;
    }
    if constexpr (!has_tls) {
      fail_connect(ep, c, errc::not_supported);  // built without TLS (no_tls)
    } else {
      // TLS: the connect deadline becomes the handshake deadline (tls_timeout).
      cancel_conn_deadline(c);
      std::uint64_t const now = _env.now_ns();
      arm_conn_deadline(c, now + ms_ns(ep.desc->cfg.conn.tls_timeout));
      c.kind  = kind_handshaking;
      c.state = conn_state::handshaking;
      std::error_code ec;
      c.session = ep.desc->cfg.tls_provider->new_session(
        c.fd, tls::server_identity{ep.desc->cfg.host, ep.desc->cfg.port}, ec);
      if (!c.session) {
        ++ep.ctr.handshake_failures;
        fail_connect(ep, c, Tls::handshake_error(ec));
        return;
      }
      drive_handshake(ep, c);
    }
  }

  /// The handshake runs on the slot's reactor with a real fd: the library reads and writes the
  /// non-blocking socket itself, POLL_ADD readiness between steps (gpuio T14).
  void drive_handshake(ep_table& ep, conn_slot& c) noexcept
  {
    if constexpr (has_tls) {
      std::error_code ec;
      switch (Tls::handshake(*c.session, ec)) {
        case tls::handshake_status::want_read: issue_hs_poll(c, POLLIN); return;
        case tls::handshake_status::want_write: issue_hs_poll(c, POLLOUT); return;
        case tls::handshake_status::done: finish_handshake(ep, c); return;
        case tls::handshake_status::failed:
          ++ep.ctr.handshake_failures;
          fail_connect(ep, c, Tls::handshake_error(ec));
          return;
      }
    }
  }

  void issue_hs_poll(conn_slot& c, unsigned events) noexcept
  {
    c.pv.io           = io_kind::hs_poll;
    io_uring_sqe* sqe = _env.get_sqe(c.pv.io_op, ud_kind::handshake_poll);
    io_uring_prep_poll_add(sqe, c.fd, events);
    c.pv.io_inflight = true;
  }

  void on_hs_poll(ep_table& ep, conn_slot& c, int res) noexcept
  {
    if (c.pv.cancel_requested) {
      ++ep.ctr.handshake_timeouts;
      fail_connect(ep, c, timeout_or_stop(errc::connect_timeout));
      return;
    }
    if (res < 0) {
      c.pv.last_errno = -res;
      ++ep.ctr.handshake_failures;
      fail_connect(ep, c, errc::tls_handshake_failed);
      return;
    }
    drive_handshake(ep, c);
  }

  /// After `done`: kTLS RX+TX (both or neither, gpuio T1), else the user-space stream (tier 2 or
  /// tier 1), decided by `Tls::finish` (docs/io/rest-uring-design.md section 4.3).
  void finish_handshake(ep_table& ep, conn_slot& c) noexcept
  {
    if constexpr (has_tls) {
      tls::provider const& prov = *ep.desc->cfg.tls_provider;
      tls::config const& tcfg   = prov.policy();
      auto r =
        Tls::finish(c.fd, *c.session, prov, ep.desc->stream, ep.desc->cfg.host, ep.desc->cfg.port);
      if (r.info.resumed)
        ++ep.ctr.handshakes_resumed;
      else
        ++ep.ctr.handshakes_full;
      if (r.ktls_rx) ++ep.ctr.ktls_rx;
      if (r.ktls_tx) ++ep.ctr.ktls_tx;
      if (r.kind == kind_ktls && r.info.version == tls::protocol_version::tls1_3 &&
          tcfg.expect_no_pad) {
        if (r.no_pad)
          ++ep.ctr.no_pad_set;
        else
          ++ep.ctr.no_pad_failed;
      }
      ep.ctr.tls_version = static_cast<std::uint64_t>(r.info.version);
      ep.ctr.cipher_id   = r.info.cipher_suite;
      // Counted (and logged once per endpoint by the stream) only when kTLS was asked for: an
      // explicit `ktls = disable` is the configured mode (gpuio T24b).
      if (r.user_space && tcfg.ktls != tls::ktls_mode::disable) ++ep.ctr.ktls_fallbacks;
      if (r.kind == kind_ktls) {
        c.ktls  = ktls_stream(c.fd, &c.msg, ep.desc->stream);
        c.rekey = r.rekey;
        c.kind  = kind_ktls;
        become_ready(ep, c);
        return;
      }
      if (r.kind == kind_tls && r.stream != nullptr) {
        c.tls  = r.stream;
        c.kind = kind_tls;
        become_ready(ep, c);
        return;
      }
      ++ep.ctr.handshake_failures;
      fail_connect(ep, c, r.ec ? r.ec : make_error_code(errc::tls_handshake_failed));
    }
  }

  void become_ready(ep_table& ep, conn_slot& c) noexcept
  {
    cancel_conn_deadline(c);
    put(c.pv.io_op);
    --ep.n_setup;
    std::uint64_t const now  = _env.now_ns();
    c.created_ns             = now;
    c.bytes_on_connection    = 0;
    c.requests_on_connection = 0;
    c.background             = false;
    ++ep.ctr.connects_ok;
    enter_idle(ep, c, now, now);
    start_backlog(ep);
    notify_ready(c.endpoint);
  }
  void notify_ready(std::uint32_t endpoint) noexcept
  {
    if constexpr (requires(Handler& h) { h.on_ready(std::uint32_t{}); }) _h->on_ready(endpoint);
  }

  void fail_connect(ep_table& ep, conn_slot& c, std::error_code ec) noexcept
  {
    cancel_conn_deadline(c);
    CUCASCADE_NET_ASSERT(!c.pv.io_inflight);
    if (c.state == conn_state::resolving) --ep.n_resolving;
    put(c.pv.io_op);
    --ep.n_setup;
    if (ec == errc::connect_timeout)
      ++ep.ctr.connect_timeouts;
    else
      ++ep.ctr.connect_failures;
    release_socket(c);
    c.state = conn_state::free;
    c.kind  = kind_fail;
    ++c.generation;
    push_free(ep, c);
    if (ep.backlog_demand + ep.backlog_background != 0) ensure_tick();  // retried by the tick
    std::uint32_t const endpoint = c.endpoint;
    if constexpr (requires(Handler& h, std::error_code e) { h.on_connect_failed(endpoint, e); })
      _h->on_connect_failed(endpoint, ec);
  }
  void fail_connect(ep_table& ep, conn_slot& c, errc e) noexcept
  {
    fail_connect(ep, c, make_error_code(e));
  }
  [[nodiscard]] errc timeout_or_stop(errc timeout) const noexcept
  {
    return _stopping ? errc::context_stopped : timeout;
  }

  // ---- per-slot connect / TLS deadline (reactor heap; never a linked timeout) ----
  void arm_conn_deadline(conn_slot& c, std::uint64_t when) noexcept
  {
    c.pv.deadline_ns = when;
    c.pv.deadline = _env.template arm_deadline<&connection_pool::on_conn_deadline_thunk>(when, &c);
  }
  void cancel_conn_deadline(conn_slot& c) noexcept
  {
    if (c.pv.deadline.valid()) _env.cancel_deadline(c.pv.deadline);
    c.pv.deadline = {};
  }
  static void on_conn_deadline_thunk(void* arg) noexcept
  {
    auto* c        = static_cast<conn_slot*>(arg);
    c->pv.deadline = {};
    if (c->pv.io_op == no_op) return;  // armed only while the setup token is held
    static_cast<connection_pool*>(c->pv.pool)->on_conn_deadline(*c);
  }
  void on_conn_deadline(conn_slot& c) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    switch (c.state) {
      case conn_state::resolving:
        ++ep.ctr.dns_failures;
        fail_connect(ep, c, errc::connect_timeout);
        return;
      case conn_state::connecting:
      case conn_state::handshaking:
        c.pv.cancel_requested = true;
        if (c.fd >= 0) ::shutdown(c.fd, SHUT_RDWR);
        if (c.pv.io_inflight) _env.cancel_op(c.pv.io_op);
        return;
      default: return;
    }
  }

  void arm_dns_poll() noexcept
  {
    if (_dns_armed || _stopping) return;
    _dns_poll = _env.template arm_deadline<&connection_pool::dns_poll_thunk>(
      _env.now_ns() + dns_poll_ns, this);
    _dns_armed = _dns_poll.valid();
  }
  static void dns_poll_thunk(void* arg) noexcept
  {
    static_cast<connection_pool*>(arg)->on_dns_poll();
  }
  void on_dns_poll() noexcept
  {
    _dns_armed              = false;
    std::uint64_t const now = _env.now_ns();
    for (std::uint32_t e = 0; e < remote_context::max_endpoints; ++e) {
      ep_table& ep = _eps[e];
      if (ep.n_resolving == 0) continue;
      for (std::uint32_t i = 0; i < ep.capacity; ++i) {
        conn_slot& c = ep.slots[i];
        if (c.state != conn_state::resolving) continue;
        if (resolve_peer(ep, c, now, true)) {
          --ep.n_resolving;
          issue_socket(c);
        }
      }
    }
  }

  /// Upkeep is only needed while connections exist, connects are queued or keep-warm is on (an
  /// unused pool never wakes its reactor).
  [[nodiscard]] bool need_tick() const noexcept
  {
    for (std::uint32_t e = 0; e < remote_context::max_endpoints; ++e) {
      ep_table const& ep = _eps[e];
      if (ep.capacity == 0) continue;
      if (ep.n_idle + ep.n_busy + ep.n_setup + ep.backlog_demand + ep.backlog_background != 0)
        return true;
    }
    for (std::uint32_t e = 0, n = _rc->endpoint_count(); e < n; ++e)
      if (_rc->keep_warm(e).min_connections != 0) return true;
    return false;
  }
  void ensure_tick() noexcept
  {
    if (_tick_armed || _stopping) return;
    _tick = _env.template arm_deadline<&connection_pool::tick_thunk>(_env.now_ns() + tick_period_ns,
                                                                     this);
    _tick_armed = _tick.valid();
  }
  static void tick_thunk(void* arg) noexcept { static_cast<connection_pool*>(arg)->tick(); }

  // ----------------------------------------------------------------------- WAITALL watchdog --

  /// A WAITALL body receive stuck behind a pending TLS 1.3 KeyUpdate shows bytes queued in the
  /// socket and no POLLIN (kTLS masks it while key_update_pending, 6.14+), so io_uring's
  /// poll-driven partial retry never runs. Cancel it (the CQE reports the bytes it placed) and
  /// continue with one one-shot receive. Only receives older than `waitall_watchdog` are probed
  /// (3 syscalls per stuck-looking receive per tick; nothing on the per-receive path).
  ///
  /// gpuio T13b (gpuio ARCHITECTURE §4.2 rule 4 amendment): "no CQE, bytes queued, no POLLIN" alone
  /// also matches a slow-but-alive connection (WAITALL hides partial progress, and an incomplete
  /// record masks POLLIN too), which fired 29-40 times per EC2 run with no KeyUpdate. A candidate
  /// is therefore only sampled on its first tick (`waitall_watchdog_armed`) and cancelled when
  /// `tcpi_bytes_received` AND FIONREAD stayed unchanged (FIONREAD > 0) at every tick for at
  /// least `waitall_watchdog` and two ticks; any movement re-samples, a readable or empty socket
  /// drops the candidate. With the 500 ms default: cancel 1.0-1.25 s after the receive started.
  void watchdog(ep_table& ep, std::uint64_t now) noexcept
  {
    std::uint64_t const wd = ms_ns(ep.desc->cfg.waitall_watchdog);
    if (wd == 0 || !ep.desc->stream.waitall || ep.n_busy == 0) return;
    // The hang needs the 6.14 kTLS pause on KeyUpdate (key_update_pending); older kernels fail
    // the receive with EKEYEXPIRED instead (reconnect, pool_support map_socket_errno).
    if constexpr (!has_tls) {
      return;  // kTLS slots exist only with TLS
    } else {
      if (!Tls::kernel_rekey()) return;
    }
    for (std::uint32_t i = 0; i < ep.capacity; ++i) {
      conn_slot& c = ep.slots[i];
      if (c.state != conn_state::busy || c.pv.io != io_kind::body || !c.pv.io_inflight ||
          !c.pv.rx_waitall || c.kind != kind_ktls || c.pv.cancel_requested || c.pv.watchdog) {
        c.pv.wd_armed = false;
        continue;
      }
      if (now - c.pv.op_started_ns < wd) continue;
      rx_stall_sample s;
      if (!sample_rx_stall(c.fd, s)) {  // readable, nothing queued: not the hang signature
        c.pv.wd_armed = false;
        continue;
      }
      if (!c.pv.wd_armed) {  // first tick that saw the signature: baseline
        c.pv.wd_armed    = true;
        c.pv.wd_since_ns = now;
        c.pv.wd_rx_bytes = s.bytes_received;
        c.pv.wd_inq      = s.inq;
        c.pv.wd_quiet    = 0;
        ++ep.ctr.waitall_watchdog_armed;
        continue;
      }
      if (s.bytes_received != c.pv.wd_rx_bytes || s.inq != c.pv.wd_inq) {  // alive: re-sample
        c.pv.wd_since_ns = now;
        c.pv.wd_rx_bytes = s.bytes_received;
        c.pv.wd_inq      = s.inq;
        c.pv.wd_quiet    = 0;
        continue;
      }
      if (c.pv.wd_quiet < 255) ++c.pv.wd_quiet;
      if (c.pv.wd_quiet < 2 || now - c.pv.wd_since_ns < wd) continue;
      c.pv.wd_armed = false;
      c.pv.watchdog = true;
      ++ep.ctr.waitall_watchdog_cancels;
      _env.cancel_op(c.pv.io_op);
    }
  }

  // ------------------------------------------------------------------------------ keep-warm --

  void keep_warm(ep_table& ep, std::uint64_t now) noexcept
  {
    keep_warm_config const& kw = _rc->keep_warm(ep.desc->id);
    if (kw.min_connections == 0 || _stopping) return;
    std::uint64_t const interval = ms_ns(kw.interval);
    if constexpr (requires(Handler& h, conn_slot& c, std::span<char> out) {
                    { h.build_probe(std::uint32_t{}, c, out) } -> std::convertible_to<std::size_t>;
                  }) {
      for (std::uint32_t i = ep.idle_head; i != no_conn;) {
        conn_slot& c = ep.slots[i];
        i            = c.pv.next;
        if (now - c.idle_since_ns < interval) continue;
        std::size_t const n = _h->build_probe(ep.desc->id, c, std::span<char>(c.tx_buf));
        if (n == 0 || n > c.tx_buf.size()) continue;
        start_probe(ep, c, n);
      }
    }
    // Keep this pool's share of the per-context floor: reconnect in the background (never
    // outstanding work, §4.5; ceil(min / live reactors), gpuio T15b).
    std::uint32_t const live = ep.live_or_pending();
    unsigned const floor     = keep_warm_share(kw.min_connections);
    if (live < floor) {
      ep.backlog_background += floor - live;
      ep.ctr.warmup_requested += floor - live;
    }
  }

  void start_probe(ep_table& ep, conn_slot& c, std::size_t n) noexcept
  {
    std::uint32_t io = no_op, tx = no_op;
    if (!_env.take_op(priority::normal, true, io)) return;
    if (!_env.take_op(priority::normal, true, tx)) {
      _env.release_op(io);
      return;
    }
    unlink_idle(ep, c);
    ++c.generation;
    c.state = conn_state::probing;
    ++ep.n_aux;
    c.pv.io_op = io;
    c.pv.tx_op = tx;
    _ops[io]   = op_rec{&c, c.generation, op_role::io};
    _ops[tx]   = op_rec{&c, c.generation, op_role::tx};
    _held += 2;
    begin_exchange(ep, c, _env.now_ns());
    c.reused = true;
    c.parser.reset(http::parse_options{
      .header_limit = static_cast<std::uint32_t>(conn_header_bytes), .head_request = true});
    c.pv.parse_off = 0;
    c.rx_window[0] = iovec{c.tx_buf.data(), n};
    ++ep.ctr.keep_warm_probes;
    send(c, std::span<iovec const>(c.rx_window.data(), 1));
    std::ignore = recv_header(c, true);
  }

  void probe_done(ep_table& ep, conn_slot& c, bool ok) noexcept
  {
    if (!ok) {
      ++ep.ctr.keep_warm_failures;
      close_probe(ep, c);
      return;
    }
    if (c.pv.io_inflight || c.pv.tx_inflight) return;  // wait for the send CQE too
    put(c.pv.io_op);
    put(c.pv.tx_op);
    ++c.requests_on_connection;
    if (!c.meta.keep_alive || _stopping || !stream_drained(ep, c)) {
      close_conn(ep, c);
      return;
    }
    --ep.n_aux;
    std::uint64_t const now = _env.now_ns();
    enter_idle(ep, c, now, now);
    notify_ready(c.endpoint);
  }
  void close_probe(ep_table& ep, conn_slot& c) noexcept { close_conn(ep, c); }

  /// Probe response head: complete, keep-alive, no body (HEAD) => the connection is warm.
  void probe_on_header(ep_table& ep, conn_slot& c, std::error_code ec) noexcept
  {
    if (ec) {
      probe_done(ep, c, false);
      return;
    }
    std::size_t consumed = 0;
    std::error_code pe;
    auto const st = c.parser.parse_header(
      std::span<char const>(c.header_buf.data() + c.pv.parse_off, c.hdr_len - c.pv.parse_off),
      consumed,
      c.meta,
      pe);
    c.pv.parse_off += static_cast<std::uint32_t>(consumed);
    if (st == http::parse_status::need_more) {
      if (recv_header(c, false)) probe_done(ep, c, false);
      return;
    }
    probe_done(ep, c, st == http::parse_status::ok && c.parser.done());
  }

  // --------------------------------------------------------------------------- stream I/O ----

  template <class F>
  decltype(auto) with_stream(conn_slot& c, F&& f) noexcept
  {
    if constexpr (has_fallback) {
      if (c.kind == kind_tls) [[unlikely]]
        return f(*static_cast<Fallback*>(c.tls));
    }
    if (c.kind == kind_ktls) [[likely]]
      return f(c.ktls);
    return f(c.plain);
  }

  [[nodiscard]] std::size_t header_space(conn_slot const& c) const noexcept
  {
    std::size_t const cap =
      std::min<std::size_t>(conn_header_bytes, _eps[c.endpoint].desc->cfg.conn.header_buffer_bytes);
    return c.hdr_len >= cap ? 0 : cap - c.hdr_len;
  }

  void issue_send(conn_slot& c) noexcept
  {
    std::span<iovec> w;
    std::byte* ptr           = nullptr;
    std::uint64_t len        = 0;
    std::uint64_t const left = c.pv.tx_total - c.pv.tx_done;
    if (c.pv.tx_cursor.contiguous_window(ptr, len, left)) {
      c.tx_window[0] = iovec{ptr, static_cast<std::size_t>(len)};
      w              = std::span<iovec>(c.tx_window.data(), 1);
    } else {
      w = c.pv.tx_cursor.window(std::span<iovec>(c.tx_window), left);
    }
    io_uring_sqe* sqe = _env.get_sqe(c.pv.tx_op, ud_kind::op);
    with_stream(c, [&](auto& s) noexcept { s.prep_send(sqe, std::span<iovec const>(w)); });
    c.pv.tx_inflight = true;
  }

  void on_tx_cqe(conn_slot& c, int res) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    ++ep.ctr.send_cqes;
    if (c.state == conn_state::closing) {
      maybe_finalize(ep, c);
      return;
    }
    // A user-space TLS stream (gpuio T16) sends ciphertext: it maps the CQE to plaintext bytes of
    // tx.
    res = with_stream(c, [&](auto& s) noexcept -> int {
      if constexpr (requires {
                      { s.on_send(res) } -> std::convertible_to<int>;
                    })
        return s.on_send(res);
      else
        return res;
    });
    if (res > 0) {
      auto const n = static_cast<std::uint64_t>(res);
      c.pv.tx_cursor.commit(n);
      c.pv.tx_done += n;
      c.exchange_tx += n;
      c.bytes_on_connection += n;
      ep.ctr.bytes_sent += n;
      if (c.pv.tx_done < c.pv.tx_total && !c.pv.cancel_requested) {
        ++ep.ctr.short_sends;
        issue_send(c);
        return;
      }
    }
    std::error_code ec;
    if (c.pv.cancel_requested && c.pv.tx_done < c.pv.tx_total)
      ec = make_error_code(_stopping ? errc::context_stopped : errc::operation_canceled);
    else if (res < 0)
      ec = stale_or(ep, c, -res);
    else if (res == 0 && c.pv.tx_done < c.pv.tx_total)
      ec = stale_or(ep, c, EPIPE);
    // A KeyUpdate reply may be owed behind this send (the reply is a kTLS record written to the
    // socket: never while another send is in flight). It is flushed after the callback, if the
    // slot still runs the same exchange and the callback did not start another send.
    bool const replay_after = c.pv.reply_pending;
    c.pv.reply_pending      = false;
    std::uint32_t const gen = c.generation;
    if (c.state == conn_state::probing) {
      if (ec)
        probe_done(ep, c, false);
      else if (c.parser.done() && !c.pv.io_inflight && !replay_after)
        probe_done(ep, c, true);
    } else {
      _h->on_send(c, ec, static_cast<std::size_t>(c.pv.tx_done));
    }
    if (!replay_after || c.generation != gen ||
        (c.state != conn_state::busy && c.state != conn_state::probing))
      return;
    if (c.pv.tx_inflight) {
      c.pv.reply_pending = true;
      return;
    }
    flush_owed_reply(
      ep, c, c.pv.io == io_kind::drain_body ? io_kind::drain_body : io_kind::drain_header);
  }

  void issue_header(conn_slot& c) noexcept
  {
    c.pv.io                 = io_kind::header;
    std::size_t const space = header_space(c);
    c.pv.rx_len             = space;
    io_uring_sqe* sqe       = _env.get_sqe(c.pv.io_op, ud_kind::op);
    bool const pf           = c.pv.tx_poll_first;
    c.pv.tx_poll_first      = false;
    with_stream(c, [&](auto& s) noexcept {
      s.prep_recv_header(sqe, std::span<char>(c.header_buf.data() + c.hdr_len, space), pf);
    });
    c.pv.io_inflight = true;
  }

  [[nodiscard]] std::error_code issue_body(conn_slot& c) noexcept
  {
    ep_table& ep            = _eps[c.endpoint];
    std::uint64_t const cap = std::max<std::uint64_t>(ep.desc->cfg.conn.body_recv_cap, 1);
    // Rule 1: never ask for more than the response still owes (an over-long WAITALL receive
    // completes only when the peer closes: 75 s nginx, ~5 s S3; E-KTLS-1 §3).
    std::uint64_t want = std::min(c.pv.rx_remaining, cap);
    if (_debug.disable_overread_guard) want = std::min(c.sink.remaining(), cap);
    std::byte* ptr    = nullptr;
    std::uint64_t len = 0;
    std::span<iovec> w;
    if (c.sink.contiguous_window(ptr, len, want)) {
      c.rx_window[0] = iovec{ptr, static_cast<std::size_t>(len)};
      w              = std::span<iovec>(c.rx_window.data(), 1);
    } else {
      w   = c.sink.window(std::span<iovec>(c.rx_window), want);
      len = 0;
      for (auto const& v : w)
        len += v.iov_len;
    }
    if (w.empty() || len == 0) return make_error_code(errc::http_length_mismatch);
    if (len > c.pv.rx_remaining)
      ++ep.ctr.overread_issued;  // only reachable with the test knob
    else if (std::min(c.sink.remaining(), cap) > len)
      ++ep.ctr.overread_guarded;
    CUCASCADE_NET_ASSERT(_debug.disable_overread_guard || len <= c.pv.rx_remaining);
    c.pv.rx_len    = len;
    c.pv.rx_iovcnt = static_cast<std::uint32_t>(w.size());
    c.pv.io        = io_kind::body;
    if (ep.desc->stream.iowq && !_iowq_applied) {
      // gpuio ARCHITECTURE §4.3: cap + affinity applied once per ring BEFORE the first receive.
      _iowq         = apply_iowq_placement(_env.ring(), ep.desc->cfg.conn);
      _iowq_applied = true;
    }
    ++ep.ctr.body_receives;
    ep.ctr.body_bytes_requested += len;
    c.pv.op_started_ns = _env.now_ns();
    c.pv.wd_armed      = false;  // a new receive: the watchdog samples it afresh
    bool const once    = c.pv.oneshot_next;
    c.pv.oneshot_next  = false;
    c.pv.rx_waitall    = ep.desc->stream.waitall && !once;
    io_uring_sqe* sqe  = _env.get_sqe(c.pv.io_op, ud_kind::op);
    with_stream(c, [&](auto& s) noexcept {
      if constexpr (requires { s.prep_recv_once(sqe, std::span<iovec const>(w)); }) {
        if (once) {
          s.prep_recv_once(sqe, std::span<iovec const>(w));
          return;
        }
      }
      s.prep_recv(sqe, std::span<iovec const>(w));
    });
    c.pv.io_inflight = true;
    return {};
  }

  /// Control-record drain: header phase into the free tail of the header buffer, body/idle phase
  /// into the whole buffer (the driver copied the leftover body bytes out before recv_body).
  void issue_drain(conn_slot& c, io_kind k) noexcept
  {
    std::size_t const base = k == io_kind::drain_header ? c.hdr_len : 0;
    std::size_t const off  = base + c.pv.drain_len;
    c.pv.io                = k;
    c.pv.replay_off        = static_cast<std::uint32_t>(base);
    if (off >= conn_header_bytes) {  // a record that does not fit: cannot handle it
      c.pv.last_errno = EMSGSIZE;
      on_drain_failed(_eps[c.endpoint], c, make_error_code(errc::tls_protocol_error));
      return;
    }
    io_uring_sqe* sqe = _env.get_sqe(c.pv.io_op, ud_kind::op);
    c.ktls.prep_drain(sqe, std::span<char>(c.header_buf.data() + off, conn_header_bytes - off));
    c.pv.io_inflight = true;
  }

  void on_io_cqe(conn_slot& c, int res, unsigned flags) noexcept
  {
    ep_table& ep = _eps[c.endpoint];
    if (c.state == conn_state::closing) {
      maybe_finalize(ep, c);
      return;
    }
    switch (c.pv.io) {
      case io_kind::socket: on_socket(ep, c, res); return;
      case io_kind::connect: on_connect(ep, c, res); return;
      case io_kind::hs_poll: on_hs_poll(ep, c, res); return;
      case io_kind::header: on_header_cqe(ep, c, res, flags); return;
      case io_kind::body: on_body_cqe(ep, c, res, flags); return;
      case io_kind::drain_header:
      case io_kind::drain_body:
      case io_kind::drain_idle: on_drain_cqe(ep, c, res); return;
      case io_kind::flush_poll: on_flush_poll(ep, c, res); return;
      case io_kind::none: return;
    }
  }

  void on_header_cqe(ep_table& ep, conn_slot& c, int res, unsigned flags) noexcept
  {
    ++ep.ctr.header_cqes;
    iovec const win{c.header_buf.data() + c.hdr_len, static_cast<std::size_t>(c.pv.rx_len)};
    rx_result const r = with_stream(
      c, [&](auto& s) noexcept { return s.on_recv(res, flags, std::span<iovec const>(&win, 1)); });
    switch (r.kind) {
      case rx_kind::data:
        c.first_byte     = true;
        c.pv.control_run = 0;
        c.hdr_len += static_cast<std::uint32_t>(r.bytes);
        c.exchange_rx += r.bytes;
        c.bytes_on_connection += r.bytes;
        ep.ctr.header_bytes += r.bytes;
        deliver_header(ep, c, cancel_error(c), r.bytes);
        return;
      case rx_kind::control:
        ++ep.ctr.eio_drains;
        c.pv.drain_len = 0;
        issue_drain(c, io_kind::drain_header);
        return;
      case rx_kind::again:
        if (!c.pv.cancel_requested) {
          issue_header(c);
          return;
        }
        deliver_header(ep, c, cancel_error(c), 0);
        return;
      case rx_kind::eof: deliver_header(ep, c, stale_or(ep, c, ECONNRESET), 0); return;
      case rx_kind::error:
        deliver_header(ep, c, c.pv.cancel_requested ? cancel_error(c) : stale_or(ep, c, r.err), 0);
        return;
    }
  }

  void deliver_header(ep_table& ep, conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    c.pv.rx_active = false;
    if (c.state == conn_state::probing) {
      probe_on_header(ep, c, ec);
      return;
    }
    _h->on_header(c, ec, n);
  }
  void deliver_body(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    c.pv.rx_active = false;
    _h->on_body(c, ec, n);
  }

  void on_body_cqe(ep_table& ep, conn_slot& c, int res, unsigned flags) noexcept
  {
    ++ep.ctr.body_cqes;
    if (c.pv.watchdog && !c.pv.cancel_requested) {
      // Our own cancel: keep what was placed (done_io), continue with a one-shot receive.
      c.pv.watchdog     = false;
      c.pv.oneshot_next = true;
      if (res == -ECANCELED || res == -EINTR) {
        if (auto ec = issue_body(c)) deliver_body(c, ec, 0);
        return;
      }
    }
    c.pv.watchdog     = false;
    rx_result const r = with_stream(c, [&](auto& s) noexcept {
      return s.on_recv(res, flags, std::span<iovec const>(c.rx_window.data(), c.pv.rx_iovcnt));
    });
    switch (r.kind) {
      case rx_kind::data: {
        if (r.bytes < c.pv.rx_len && c.pv.rx_waitall && !c.pv.cancel_requested &&
            !c.pv.oneshot_next)
          ++ep.ctr.waitall_short_cqes;  // rule 3: a control record follows
        c.sink.commit(r.bytes);
        c.first_byte     = true;
        c.pv.control_run = 0;
        c.exchange_rx += r.bytes;
        c.bytes_on_connection += r.bytes;
        ep.ctr.body_bytes += r.bytes;
        deliver_body(c, cancel_error(c), r.bytes);
        return;
      }
      case rx_kind::control:
        ++ep.ctr.eio_drains;
        c.pv.drain_len = 0;
        issue_drain(c, io_kind::drain_body);
        return;
      case rx_kind::again:
        if (!c.pv.cancel_requested) {
          if (auto ec = issue_body(c)) deliver_body(c, ec, 0);
          return;
        }
        deliver_body(c, cancel_error(c), 0);
        return;
      case rx_kind::eof:
        deliver_body(c, make_error_code(errc::http_length_mismatch), 0);  // truncated body
        return;
      case rx_kind::error:
        deliver_body(c, c.pv.cancel_requested ? cancel_error(c) : map_socket_errno(r.err), 0);
        return;
    }
  }

  void on_drain_cqe(ep_table& ep, conn_slot& c, int res) noexcept
  {
    ++ep.ctr.drain_cqes;
    if constexpr (fallback_idle_drain) {
      if (c.kind == kind_tls) {  // gpuio T16: only the idle drain exists (no rx_kind::control)
        if (!c.pv.cancel_requested && static_cast<Fallback*>(c.tls)->on_idle_recv(res)) {
          resume_after_control(ep, c, io_kind::drain_idle);  // back to idle, watch re-armed
          return;
        }
        bool peer_closed = res == 0;
        if constexpr (fallback_peer_closed)  // close_notify on an idle slot: kTLS counts it too
          peer_closed = peer_closed || static_cast<Fallback*>(c.tls)->peer_closed();
        if (peer_closed) ++ep.ctr.idle_fin_evictions;
        on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));  // closes the slot
        return;
      }
    }
    io_kind const k      = c.pv.io;
    drain_result const d = c.ktls.on_drain(res);
    if (c.pv.cancel_requested) {
      on_drain_failed(ep, c, cancel_error(c));
      return;
    }
    if (d.err != 0) {
      c.pv.last_errno = d.err;
      on_drain_failed(
        ep, c, k == io_kind::drain_body ? map_socket_errno(d.err) : stale_or(ep, c, d.err));
      return;
    }
    if (d.bytes == 0 && c.pv.drain_len == 0) {  // EOF instead of a record
      if (k == io_kind::drain_idle) ++ep.ctr.idle_fin_evictions;
      on_drain_failed(ep,
                      c,
                      k == io_kind::drain_body ? make_error_code(errc::http_length_mismatch)
                                               : stale_or(ep, c, ECONNRESET));
      return;
    }
    if (d.truncated) {
      on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
      return;
    }
    if (d.type == tls::ktls::record_data) {
      // The queued record turned out to be data (a race with the -EIO): deliver it as such.
      if (c.pv.drain_len != 0 || k == io_kind::drain_idle) {
        on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
        return;
      }
      if (k == io_kind::drain_header) {
        c.first_byte = true;
        c.hdr_len += static_cast<std::uint32_t>(d.bytes);
        c.exchange_rx += d.bytes;
        c.bytes_on_connection += d.bytes;
        ep.ctr.header_bytes += d.bytes;
        c.pv.io = io_kind::header;
        deliver_header(ep, c, {}, d.bytes);
        return;
      }
      if (d.bytes > c.pv.rx_remaining) {  // more than the response owes: never pipelined
        on_drain_failed(ep, c, make_error_code(errc::http_length_mismatch));
        return;
      }
      c.sink.copy_in(std::span<char const>(c.header_buf.data(), d.bytes));
      c.first_byte = true;
      c.exchange_rx += d.bytes;
      c.bytes_on_connection += d.bytes;
      ep.ctr.body_bytes += d.bytes;
      c.pv.io = io_kind::body;
      deliver_body(c, {}, d.bytes);
      return;
    }
    // A control record: accumulate it (MSG_EOR on its last piece), then handle it.
    c.pv.drain_type = d.type;
    c.pv.drain_len += static_cast<std::uint32_t>(d.bytes);
    if (!d.eor) {
      issue_drain(c, k);
      return;
    }
    replay(ep, c, k);
  }

  /// Handle one drained control record: `tls::ktls::on_control_record` (NewSessionTicket skipped,
  /// KeyUpdate re-keys RX, close_notify is EOF, anything else fails), then resume what it
  /// interrupted; an owed KeyUpdate reply is flushed first (gpuio T6, replaced: section 4.3).
  void replay(ep_table& ep, conn_slot& c, io_kind k) noexcept
  {
    std::span<std::byte const> const rec(
      reinterpret_cast<std::byte const*>(c.header_buf.data() + c.pv.replay_off), c.pv.drain_len);
    std::uint8_t const type = c.pv.drain_type;
    c.pv.drain_len          = 0;
    c.pv.io                 = k;  // remembered for flush_poll / resume
    if constexpr (!has_tls) {
      (void)rec;
      (void)type;
      on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
    } else {
      if (++c.pv.control_run > max_control_records) {  // a storm: never loop on control records
        ++ep.ctr.control_record_storms;
        if (k == io_kind::drain_idle) ++ep.ctr.idle_fin_evictions;
        on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
        return;
      }
      count_control_record(ep, type, rec);
      std::error_code ec;
      tls::ktls::control_action const act = Tls::on_control_record(c.fd, c.rekey, type, rec, ec);
      switch (act) {
        case tls::ktls::control_action::consumed: resume_after_control(ep, c, k); return;
        case tls::ktls::control_action::reply_owed: flush_owed_reply(ep, c, k); return;
        case tls::ktls::control_action::closed:
          // close_notify: EOF, slot not reusable. On an IDLE connection it is the TLS form of
          // the peer's FIN.
          ++ep.ctr.close_notify;
          if (k == io_kind::drain_idle) ++ep.ctr.idle_fin_evictions;
          c.close_after = true;
          on_drain_failed(ep,
                          c,
                          k == io_kind::drain_body ? make_error_code(errc::http_length_mismatch)
                                                   : stale_or(ep, c, ECONNRESET));
          return;
        case tls::ktls::control_action::failed:
          if (k == io_kind::drain_idle) ++ep.ctr.idle_fin_evictions;
          on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));  // reconnect
          return;
      }
    }
  }

  /// Counters by record: handshake messages by their type byte (4 NewSessionTicket, 24
  /// KeyUpdate), alerts apart from close_notify (counted by the action).
  static void count_control_record(ep_table& ep,
                                   std::uint8_t type,
                                   std::span<std::byte const> rec) noexcept
  {
    if (type == tls::ktls::record_handshake && !rec.empty()) {
      auto const msg = static_cast<std::uint8_t>(rec[0]);
      if (msg == 4)
        ++ep.ctr.control_records_ticket;
      else if (msg == 24)
        ++ep.ctr.control_records_key_update;
      else
        ++ep.ctr.control_records_other;
    } else if (type == tls::ktls::record_alert) {
      ++ep.ctr.control_records_alert;
    } else {
      ++ep.ctr.control_records_other;
    }
  }

  /// Send the owed KeyUpdate reply (`tls::ktls::flush_reply`), only with no send in flight (the
  /// reply is a record of its own on the kTLS socket), then resume @p k.
  void flush_owed_reply(ep_table& ep, conn_slot& c, io_kind k) noexcept
  {
    c.pv.io = k;
    if (c.pv.tx_inflight) {  // flushed from on_tx_cqe once the send completed
      ++ep.ctr.replays_deferred;
      c.pv.reply_pending = true;
      return;
    }
    if constexpr (has_tls) {
      std::error_code const ec = Tls::flush_reply(c.fd, c.rekey);
      if (!ec) {
        resume_after_control(ep, c, k);
        return;
      }
      if (ec == std::errc::operation_would_block) {  // socket full: POLLOUT, then flush again
        issue_flush_poll(ep, c, k);
        return;
      }
    }
    on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
  }

  void issue_flush_poll(ep_table& ep, conn_slot& c, io_kind resume) noexcept
  {
    ++ep.ctr.key_update_flush_polls;
    c.pv.drain_type   = static_cast<std::uint8_t>(resume);  // reuse: phase to resume
    c.pv.io           = io_kind::flush_poll;
    io_uring_sqe* sqe = _env.get_sqe(c.pv.io_op, ud_kind::op);
    io_uring_prep_poll_add(sqe, c.fd, POLLOUT);
    c.pv.io_inflight = true;
  }

  void on_flush_poll(ep_table& ep, conn_slot& c, int res) noexcept
  {
    auto const resume = static_cast<io_kind>(c.pv.drain_type);
    if (res < 0 || c.pv.cancel_requested) {
      c.pv.io = resume;
      on_drain_failed(
        ep, c, c.pv.cancel_requested ? cancel_error(c) : make_error_code(errc::tls_protocol_error));
      return;
    }
    std::error_code ec = make_error_code(errc::tls_protocol_error);
    if constexpr (has_tls) ec = Tls::flush_reply(c.fd, c.rekey);
    if (!ec) {
      resume_after_control(ep, c, resume);
      return;
    }
    if (ec == std::errc::operation_would_block) {
      issue_flush_poll(ep, c, resume);
      return;
    }
    c.pv.io = resume;
    on_drain_failed(ep, c, make_error_code(errc::tls_protocol_error));
  }

  /// The control record is consumed: re-issue what was interrupted. Control-record plaintext never
  /// advanced any cursor (header bytes stay at hdr_len, the body sink was not committed).
  void resume_after_control(ep_table& ep, conn_slot& c, io_kind k) noexcept
  {
    switch (k) {
      case io_kind::drain_header:
        if (c.state == conn_state::busy || c.state == conn_state::probing) issue_header(c);
        return;
      case io_kind::drain_body:
        if (auto ec = issue_body(c)) deliver_body(c, ec, 0);
        return;
      case io_kind::drain_idle: {
        // Tickets consumed on an idle connection: watch it again (keep the idle timestamp).
        put(c.pv.io_op);
        --ep.n_aux;
        std::uint64_t const now = _env.now_ns();
        enter_idle(ep, c, now, c.idle_since_ns);
        notify_ready(c.endpoint);
        return;
      }
      default: return;
    }
  }

  void on_drain_failed(ep_table& ep, conn_slot& c, std::error_code ec) noexcept
  {
    switch (c.pv.io) {
      case io_kind::drain_idle:
      case io_kind::flush_poll:
        if (c.state == conn_state::draining) {
          put(c.pv.io_op);
          close_conn(ep, c);
          return;
        }
        if (static_cast<io_kind>(c.pv.drain_type) == io_kind::drain_body) {
          deliver_body(c, ec, 0);
          return;
        }
        deliver_header(ep, c, ec, 0);
        return;
      case io_kind::drain_body: deliver_body(c, ec, 0); return;
      default: deliver_header(ep, c, ec, 0); return;
    }
  }

  [[nodiscard]] std::error_code cancel_error(conn_slot const& c) const noexcept
  {
    if (!c.pv.cancel_requested) return {};
    return make_error_code(_stopping ? errc::context_stopped : errc::operation_canceled);
  }

  /// Stale-reuse rule (§4.5 mechanism 3, curl transfer.c:600-660): a REUSED connection that dies
  /// (EOF, ECONNRESET, EPIPE) before any response byte is errc::stale_connection; the worker
  /// retries it at once on a fresh connection, free of backoff and attempt charge (gpuio T14
  /// decide_retry).
  [[nodiscard]] std::error_code stale_or(ep_table& ep, conn_slot const& c, int err) noexcept
  {
    bool const dead = err == ECONNRESET || err == EPIPE || err == ECONNABORTED;
    if (dead && c.reused && !c.first_byte) {
      ++ep.ctr.stale_detected;
      return make_error_code(errc::stale_connection);
    }
    return map_socket_errno(err);
  }

  // ------------------------------------------------------------------------------ teardown ----

  /// Shut the socket down (wakes every op on it: pending receives return 0, the idle watch fires
  /// with POLLHUP) and free the slot once no op that targets its memory is in flight.
  void close_conn(ep_table& ep, conn_slot& c) noexcept
  {
    ++ep.ctr.closes;
    if (c.state == conn_state::draining || c.state == conn_state::probing) {
      --ep.n_aux;
      // Background ops: release whatever is not in flight.
      if (!c.pv.io_inflight) put(c.pv.io_op);
      if (!c.pv.tx_inflight) put(c.pv.tx_op);
    }
    c.state = conn_state::closing;
    ++c.generation;
    if (c.fd >= 0) ::shutdown(c.fd, SHUT_RDWR);
    if (c.pv.io_inflight) _env.cancel_op(c.pv.io_op);
    if (c.pv.tx_inflight) _env.cancel_op(c.pv.tx_op);
    maybe_finalize(ep, c);
  }

  void maybe_finalize(ep_table& ep, conn_slot& c) noexcept
  {
    if (c.pv.io_inflight || c.pv.tx_inflight) return;
    put(c.pv.io_op);
    put(c.pv.tx_op);
    release_socket(c);
    c.state = conn_state::free;
    c.kind  = kind_fail;
    push_free(ep, c);
    start_backlog(ep);
  }

  /// Free the TLS objects (user-space stream with its opener, re-key state, session: no
  /// close_notify, as gpuio) before the socket is closed.
  void release_socket(conn_slot& c) noexcept
  {
    if constexpr (has_tls) {
      if constexpr (has_fallback) {
        if (c.tls != nullptr) Fallback::destroy(static_cast<Fallback*>(c.tls));
      }
      if (c.rekey != nullptr) Tls::release_rekey(c.rekey);
    }
    c.tls   = nullptr;
    c.rekey = nullptr;
    c.session.reset();
    if (c.fd >= 0) ::close(c.fd);
    c.fd = -1;
  }

  Env _env;
  Handler* _h;
  remote_context* _rc;
  pool_debug_options _debug{};
  std::uint32_t _op_cap{0};
  std::unique_ptr<op_rec[]> _ops;
  std::unique_ptr<ep_table[]> _eps;
  std::uint32_t _held{0};
  std::uint64_t _cqes{0};
  bool _stopping{false};
  bool _tick_armed{false}, _dns_armed{false};
  deadline_handle _tick{}, _dns_poll{};
  bool _iowq_applied{false};
  iowq_placement _iowq{};
};

}  // namespace cucascade::io::net
