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

#include "io/rest/http/chunked_decoder.hpp"
#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/response_parser.hpp"
#include "io/rest/net/common.hpp"
#include "io/rest/net/dns_resolver.hpp"
#include "io/rest/net/streams.hpp"
#include "io/tls/ktls.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <liburing.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/uio.h>

// <liburing.h> includes <linux/fs.h>, whose BLOCK_SIZE macro breaks the moodycamel queue the
// public REST headers include after this one.  Nothing here uses it.
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

/**
 * @file uring_transport.hpp
 * @brief The REST reactor's io_uring HTTP/1.1 transport (uring_ultra's `uring_https_transport`,
 *        wired to cuCascade's TLS layer and HTTP codec).
 *
 * One instance per reactor thread (worker-confined).  `slots[max_connections]`, one keep-alive
 * connection per slot, at most one primary SQE plus one cancel SQE in flight per slot; a slot is
 * never completed or reused before both of its CQEs are reaped (`pending` / `cancel_pending` /
 * `canceling`).
 *
 * Connections: non-blocking `IORING_OP_CONNECT` over the resolved addresses (IP literals bypass
 * the resolver; names come from the context's resolver thread, round robin, so the loop never
 * blocks on DNS), socket options (TCP_NODELAY, SO_KEEPALIVE with idle / interval / count,
 * optional buffers, busy poll, user timeout), then for https:// the TLS layer's non-blocking
 * handshake (readiness through
 * `IORING_OP_POLL_ADD`) followed by kTLS (plain SEND / RECVMSG with the `TLS_GET_RECORD_TYPE`
 * cmsg; control records follow the TLS layer's rules) or the user-space stream (ciphertext ring
 * and record layer, `tls_stream`).
 *
 * Reuse (curl parity): an idle connection is reused only while idle for less than
 * `conn_max_age` and when a zero-timeout poll finds it quiet (pending TLS control records are
 * consumed first); otherwise it is closed and replaced in its slot when a request needs it.
 *
 * Exchanges: `submit(slot, exchange)` sends the request head (written by the codec's
 * `request_head`) and an optional body (fixed iovecs or a generator), optionally after
 * `100 Continue`, then parses the response head (`response_parser`) through the slot's scratch
 * buffer.  A body goes straight into the destination iovecs (`sink`), into a bounded buffer
 * (`buffer`) or nowhere (`discard`).  Bodies of statuses the exchange does not accept are drained
 * when at most 64 KiB long (keeping the connection; how such a body ends does not matter, the
 * status is the answer), else the connection is closed.  An accepted body must be
 * Content-Length framed when the exchange requires it (data GETs, footer probes: chunked fails
 * with `errc::http_unsupported_framing`); other exchanges also decode chunked bodies.  A
 * close-delimited accepted body always fails with that error: it cannot be told from a cut one.
 * Bytes past a response (a HEAD with a body, data after the last chunk) retire the connection.
 *
 * Time bounds: a connect deadline (TCP + TLS), then either a whole-exchange deadline or the stall
 * detector (fewer than limit x time bytes in the last `time`): on expiry the in-flight SQE is
 * cancelled and the exchange completes with `errc::request_timeout` (`connect_timeout` while
 * connecting).
 */

namespace cucascade::io::net {

/// Thrown when the ring itself fails: fatal to the reactor.
class transport_fatal_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// An absolute http:// or https:// URL, split for the transport.
struct url_parts {
  bool tls{true};
  std::string host;  ///< without IPv6 brackets
  std::uint16_t port{443};
  std::string authority;  ///< Host header value (port only when not the scheme default)
  std::string target;     ///< path and query, never empty
  /// Pool key: scheme, host and port.
  [[nodiscard]] std::string origin() const;
};

/// @throws std::invalid_argument for anything but an absolute http(s) URL without userinfo.
[[nodiscard]] url_parts parse_url(std::string_view url);

/// DMI / hypervisor hint of an EC2 instance (`<sysfs>/devices/virtual/dmi/id/{sys,board}_vendor`
/// = "Amazon EC2", or `<sysfs>/hypervisor/uuid` starting with "ec2").
[[nodiscard]] bool ec2_platform(std::string_view sysfs_root = "/sys") noexcept;

/// @return whether kTLS body receives carry IOSQE_ASYNC (decryption on io-wq workers):
/// `iowq` always, `automatic` on EC2, `inline_` never.
[[nodiscard]] bool ktls_recv_async(tls::recv_placement placement, bool on_ec2) noexcept;

/**
 * @brief The KeyUpdate reply a kTLS connection owes (RFC 8446 section 4.6.3).
 *
 * `tls::ktls::flush_reply` sends it with one non-blocking `sendmsg`; a full socket buffer
 * (`operation_would_block`) is not an error: the reply stays owed, receiving continues, and the
 * transport waits for POLLOUT and flushes it before the connection sends anything else.
 */
struct ktls_reply {
  using flush_fn = std::error_code (*)(int fd, tls::ktls::rekey_state* state) noexcept;
  enum class step : std::uint8_t {
    done,           ///< nothing owed (any more)
    wait_writable,  ///< still owed: wait for POLLOUT, then flush again
    failed,         ///< the connection is unusable
  };

  bool owed{false};

  /// Try to send the owed reply with @p flush (`tls::ktls::flush_reply` in production).
  step flush(int fd, tls::ktls::rekey_state* state, flush_fn flush) noexcept
  {
    owed          = true;
    auto const ec = flush(fd, state);
    if (!ec) {
      owed = false;
      return step::done;
    }
    if (ec == std::errc::operation_would_block || ec == std::errc::resource_unavailable_try_again) {
      return step::wait_writable;
    }
    return step::failed;
  }
};

/// Static options of one transport (from `rest::config`).
struct transport_options {
  std::size_t max_connections{64};
  std::size_t queue_depth{1024};
  std::size_t receive_buffer_bytes{256 * KiB};
  std::size_t max_recv_iovecs{64};
  bool recv_waitall{false};
  bool sqpoll{false};
  bool single_issuer{false};
  bool defer_taskrun{false};
  bool register_ring_fd{true};
  unsigned iowq_max_unbound{0};
  unsigned iowq_max_bound{0};
  /// uring wait mode: submit() leaves SQEs for wait() to flush.
  bool defer_flush{false};

  int keepalive_idle_s{60};
  int keepalive_interval_s{60};
  int keepalive_count{0};
  int so_rcvbuf{0};
  int so_sndbuf{0};
  int busy_poll_us{0};
  unsigned tcp_user_timeout_ms{0};
  std::chrono::milliseconds connect_timeout{3000};
  std::chrono::milliseconds conn_max_age{20'000};  ///< idle cutoff; 0 = none
  std::uint64_t max_connection_bytes{0};           ///< retire after this many bytes; 0 = off
  bool ktls_recv_async{false};
  stream_options user_space{};
};

/// Context-wide services (shared by every reactor of a pool).
struct transport_services {
  dns_resolver* dns{nullptr};
  /// The TLS provider of https:// endpoints (built on first use; may throw std::system_error).
  std::function<std::shared_ptr<tls::provider const>()> tls;
};

/// Where a response body goes.
enum class body_target : std::uint8_t {
  sink,     ///< the destination iovecs (Content-Length bodies only)
  buffer,   ///< `exchange_result::body`, at most `buffer_limit` bytes
  discard,  ///< received and dropped (keeps the connection)
};

/// One HTTP exchange.
struct exchange {
  http::method method{http::method::get};
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;  ///< authorizer headers, verbatim
  /// `Range: bytes=first-last` when set.
  std::optional<std::pair<std::uint64_t, std::uint64_t>> range;
  /// `Range: bytes=-N` when > 0 (footer probe).
  std::uint64_t suffix_bytes{0};
  std::string if_match;  ///< `If-Match` when non-empty

  /// Request body (B's PUT / POST): fixed iovecs, or a generator called whenever everything it
  /// returned before was sent (an empty span ends the body).  `body_length` is sent as
  /// Content-Length and must equal the bytes produced.  The memory must stay valid until the
  /// exchange completes.
  bool has_body{false};
  std::uint64_t body_length{0};
  std::vector<iovec> body;
  std::function<std::span<iovec const>()> body_next;
  /// Send the body only after `100 Continue` (a final response first ends the exchange).
  bool expect_continue{false};

  /// Response body handling.
  body_target target{body_target::discard};
  std::span<iovec const> sink;  ///< target == sink
  std::size_t buffer_limit{0};  ///< target == buffer: bytes kept (a longer body is truncated)
  /// Status whose body goes to `target`: 0 = any 2xx.  Other statuses' bodies are drained
  /// (<= 64 KiB) or the connection is closed.
  std::uint16_t accept_status{0};
  /// An accepted body must be Content-Length framed (else errc::http_unsupported_framing).
  bool require_length{false};

  /// Close an idle connection in the slot first (curl's FRESH_CONNECT).
  bool fresh_connection{false};
  /// Whole-exchange bound (0 = none).
  std::chrono::milliseconds timeout{0};
  /// Stall detector: fail when fewer than `stall_bytes_per_s * stall_time` bytes arrived during
  /// the last `stall_time` (either 0 = off).
  std::uint64_t stall_bytes_per_s{0};
  std::chrono::seconds stall_time{0};
};

/// The outcome of one exchange.
struct exchange_result {
  std::error_code ec;            ///< transport failure, or success
  http::response_meta meta{};    ///< status 0 when no head arrived
  std::uint64_t body_bytes{0};   ///< body bytes placed into the sink / buffer
  bool body_complete{false};     ///< the whole body of an accepted status was received
  bool truncated{false};         ///< buffer target: the body exceeded `buffer_limit`
  bool reused{false};            ///< ran on a pooled keep-alive connection
  bool response_started{false};  ///< at least one response byte arrived
  std::vector<char> body;        ///< target == buffer
};

/// Cumulative counters of one transport.
struct transport_counters {
  std::uint64_t requests{0};
  std::uint64_t connects{0};
  std::uint64_t connect_failures{0};
  std::uint64_t handshakes{0};
  std::uint64_t ktls_sessions{0};
  std::uint64_t userspace_tls_sessions{0};
  std::uint64_t ktls_fallbacks{0};  ///< user space although kTLS was allowed
  std::uint64_t reused{0};
  std::uint64_t closed{0};
  std::uint64_t idle_closes{0};     ///< reuse refused: idle longer than conn_max_age
  std::uint64_t dead_closes{0};     ///< reuse refused: the health poll found input / HUP / error
  std::uint64_t stale_failures{0};  ///< reused connection died before any response byte
  std::uint64_t sends{0};
  std::uint64_t recvs{0};
  std::uint64_t bytes_sent{0};
  std::uint64_t recv_direct_bytes{0};
  std::uint64_t recv_scratch_bytes{0};
  std::uint64_t timeouts{0};
  std::uint64_t stalls{0};
  std::uint64_t errors{0};
  std::uint64_t control_records{0};
};

class uring_transport {
 public:
  struct completion {
    std::size_t slot;
    exchange_result result;
  };

  uring_transport(transport_options opts, transport_services services);
  ~uring_transport();

  uring_transport(uring_transport const&)            = delete;
  uring_transport& operator=(uring_transport const&) = delete;

  [[nodiscard]] std::size_t slots() const noexcept { return _slots.size(); }
  /// Pollable fd that becomes readable when CQEs are available (epoll wait mode).
  [[nodiscard]] int completion_fd() const noexcept { return _ring.ring_fd; }
  /// Milliseconds until the earliest deadline / stall check (-1 = none, 0 = act now).
  [[nodiscard]] int timeout_ms() const noexcept;

  /// Start @p x on idle slot @p slot.
  void submit(std::size_t slot, exchange x);
  /// True when @p slot holds an exchange (or a parked connect).
  [[nodiscard]] bool busy(std::size_t slot) const noexcept;
  /// True when @p slot holds an established connection (to any origin).
  [[nodiscard]] bool established(std::size_t slot) const noexcept;
  /// Connect (TCP + TLS) every idle, unconnected slot to the origin of @p url without a request
  /// (uring_ultra's warm-up; such slots are busy until connected).
  void warmup(std::string const& url);
  /// Close the idle connection of @p slot now (no-op while busy).
  void close_idle(std::size_t slot) noexcept;
  /// When @p slot's idle connection was last used (time_point{} = never).
  [[nodiscard]] std::chrono::steady_clock::time_point idle_since(std::size_t slot) const noexcept;

  /// Submit queued SQEs.
  void flush();
  /// Reap CQEs, advance connections, cancel expired exchanges.  The completions (and their
  /// results) are valid until the next call.
  std::vector<completion>& poll();

  /// uring wait mode: submit and wait up to @p timeout_ms (-1 = forever) for one CQE.
  void wait(int timeout_ms);
  /// uring wait mode: report readability of @p fd through the ring (multishot poll).
  void watch_fd(int fd);
  /// Whether a watched fd fired since the last call.
  bool take_watch_events() noexcept;

  /// Cancel and reap every kernel operation; close every connection.  Idempotent.
  void shutdown() noexcept;

  [[nodiscard]] std::size_t established_count() const noexcept;
  [[nodiscard]] transport_counters const& counters() const noexcept { return _ctr; }

 private:
  struct connection;
  enum class op : std::uint8_t;

  io_uring_sqe* sqe(std::uint64_t id);
  void cancel(std::size_t index);
  void begin(std::size_t index);
  void start_connect(std::size_t index);
  void open_socket(std::size_t index);
  void on_connected(std::size_t index);
  void handshake(std::size_t index);
  void established_now(std::size_t index);
  void start_exchange(std::size_t index);
  void send(std::size_t index);
  void on_sent(std::size_t index, std::size_t bytes);
  void receive(std::size_t index);
  void on_receive(std::size_t index, int res);
  void on_drain(std::size_t index, int res);
  void on_payload(std::size_t index, std::size_t bytes);
  void on_eof(std::size_t index);
  void on_recv_error(std::size_t index, int err);
  void parse_head(std::size_t index);
  void deliver_body(std::size_t index);
  void continue_body(std::size_t index);
  void advance(std::size_t index, int res, unsigned flags);
  bool reusable(std::size_t index);
  bool healthy_idle(std::size_t index);
  void close(std::size_t index) noexcept;
  void finish(std::size_t index, std::error_code ec);
  void check_deadlines();

  transport_options _opts;
  transport_services _services;
  std::shared_ptr<tls::provider const> _tls;
  io_uring _ring{};
  bool _ring_live{false};
  std::vector<std::unique_ptr<connection>> _slots;
  std::vector<completion> _completed;
  std::vector<completion> _returned;
  bool _watch_fired{false};
  transport_counters _ctr;
};

}  // namespace cucascade::io::net
