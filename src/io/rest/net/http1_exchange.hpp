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

/// @file http1_exchange.hpp
/// @brief One HTTP/1.1 request/response exchange on a pooled connection (API.md §B.6
/// `http1_exchange`), written once over the `connection_pool` whole-buffer operations (which in
/// turn are written once over `completion_stream`): send -> head -> body straight into the
/// destination -> finish.
///
/// Design: gpuio ARCHITECTURE §3.5 (GET flow: SEND, one-shot header RECV with POLL_FIRST,
/// incremental `parse_header`, leftover memcpy once per GET, capped `RECV|WAITALL` body receives
/// straight into the destination), §3.6 (PUT with Content-Length: head + body iovecs in one
/// SENDMSG), §7.3 (Content-Range validation, error-body drain, retry classification), §4.2 (the
/// header buffer is the control-record drain target during the body: every header byte the
/// driver needs is consumed before the first `recv_body`). Research: s3_line_rate §2.1 "Range
/// semantics", §2.8 (error policy); beast.md §2.2 (parser), §3.1 #5 (direct body).
///
/// Division of labour (connection_pool.hpp "Exchange protocol"): the POOL owns streams, op slots,
/// short-send continuation, the -EIO control drain, the over-read cap and the stale-reuse signal
/// (`errc::stale_connection`); THIS class owns the HTTP state of one exchange: request framing
/// (`Expect: 100-continue`), incremental head parsing across receives, response validation
/// (`http::classify` with the requested range and the known object size: a 206 whose
/// Content-Range start/end/total disagree fails hard), the body cursor (`conn_slot::sink`), and
/// the error-body drain: a non-2xx body of at most `error_body_max` bytes (Content-Length, or
/// chunked through `http::chunked_decoder(error_body_max)`) is read with one-shot header-phase
/// receives (never a WAITALL receive of unknown length, rule 1), its first `error_capture_bytes`
/// kept, `http::extract_error_code` run on them, and only then `http::classify` (so `SlowDown`,
/// `ExpiredToken`, `NoSuchKey` ... decide the retry class). Larger or EOF-delimited error bodies
/// close the connection. The OWNER (the REST worker, a test client) owns the policy:
/// connection checkout, request head, deadlines, retries, check-in/close after `done`.
///
/// Owner interface (static, no virtual calls; all noexcept, reactor thread):
/// ```
///   // Destination of the 2xx body bytes [object_offset, object_offset + length): iovecs that
///   // stay valid until the exchange ends. `skip` = do not read the body (the connection is then
///   // closed, the exchange still succeeds), e.g. a footer probe answered with a whole object.
///   exchange_window body_window(conn_slot&, std::uint64_t object_offset, std::uint64_t length);
///   // Before each further body receive: false aborts the exchange (operation_canceled; the
///   // owner then closes the connection). Used for cancel_abort_bytes (gpuio ARCHITECTURE §3.1).
///   bool keep_receiving(conn_slot&, std::uint64_t remaining);
/// ```
/// Driving: the owner calls `start` after checkout, then forwards the pool's Handler callbacks
/// (`on_send` / `on_header` / `on_body`) and, for `Expect: 100-continue`, `continue_timeout`
/// when its 1 s fallback deadline fires. When a call returns `step::done`, `result()` is final:
/// `reusable` => `pool.check_in(c)` (nothing of this exchange is in flight), else `pool.close(c)`
/// (which also cuts a still-running send short). No allocation anywhere.

#pragma once

#include "io/rest/http/chunked_decoder.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/response_parser.hpp"
#include "io/rest/net/common.hpp"
#include "io/rest/net/conn_slot.hpp"

#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <system_error>

namespace cucascade::io::net {

/// Bytes of a drained error body kept for `extract_error_code` (S3 puts `<Code>` in the first
/// ~100 bytes of a ~300-500 byte body; the rest is drained and dropped).
inline constexpr std::size_t error_capture_bytes = 1024;

/// What the owner sends and how the response is validated.
struct exchange_spec {
  http::method method{http::method::get};
  /// The whole request: `tx[0, head_iovs)` is the head (in `conn_slot::tx_buf`), the rest the
  /// PUT body (caller memory). Must stay valid until the exchange ends.
  std::span<iovec const> tx{};
  std::size_t head_iovs{1};
  /// Send the head alone and the body after `100 Continue` (or `continue_timeout`).
  bool expect_continue{false};
  /// GET with `Range: bytes=first-last` (inclusive). A 206 is validated against it (start, end,
  /// total = `object_size` when non-zero) and a 200 is accepted only when sliceable (§7.3).
  bool ranged{false};
  std::uint64_t first{0}, last{0};
  /// Suffix probe (`Range: bytes=-N`): any 206 / 200 is accepted (the owner reads the
  /// Content-Range); a 416 with `bytes */size` is a normal answer for an empty object.
  bool suffix{false};
  std::uint64_t object_size{0};  ///< 0 = unknown
  /// connection_config::error_body_drain_max (64 KiB): larger error bodies close the connection.
  std::uint64_t error_body_max{64 * KiB};
  /// gpuio T18b: 2xx response-body window of a `remote_call` (any method). When non-empty, a 2xx
  /// body (Content-Length or chunked) is drained into it with header-phase receives instead of the
  /// 1 KiB error capture; a Content-Length above its size closes the connection after the head
  /// (the owner sees `body_overflow()`). Error bodies still go to the error capture.
  std::span<char> body_capture{};
};

/// The body destination for a 2xx response.
struct exchange_window {
  std::span<iovec const> iov{};
  bool skip{false};
};

/// Final state of an exchange.
struct exchange_result {
  /// Socket / TLS / codec / cancel error of the exchange; empty when a complete response was read.
  std::error_code transport{};
  /// `http::classify` of the response (or of the failure): `outcome.ec` empty => success.
  http::outcome outcome{};
  std::uint16_t status{0};
  std::uint64_t body_bytes{0};  ///< 2xx body bytes delivered into the window
  bool complete{false};         ///< the whole response (head + body / drained error body) was read
  bool reusable{false};         ///< the connection can be checked in
  bool body_sent{true};         ///< false: Expect: 100-continue and the body was never sent
  bool head_received{false};
  std::uint16_t interim{0};  ///< 1xx heads seen (100 Continue)
};

/// One exchange (see file comment). Lives in the owner's per-operation record; reusable.
template <class Pool, class Owner>
class http1_exchange {
 public:
  enum class step : std::uint8_t { in_progress, done };

  /// @brief Begin the exchange on a checked-out connection: send the head (and the body unless
  /// `expect_continue`) and post the header receive (POLL_FIRST) in the same pass.
  /// @return `done` when the receive could not be posted (result() says why).
  step start(Pool& pool, Owner& owner, conn_slot& c, exchange_spec const& spec) noexcept
  {
    _pool         = &pool;
    _owner        = &owner;
    _spec         = spec;
    _res          = exchange_result{};
    _parse_off    = 0;
    _err_len      = 0;
    _cap_len      = 0;
    _cap_2xx      = false;
    _cap_overflow = false;
    _drain_left   = 0;
    _phase        = phase::head;
    _tx_inflight  = false;
    _body_sent    = !spec.expect_continue || spec.tx.size() <= spec.head_iovs;
    _body_wanted  = false;
    _pending_done = false;
    _finished     = false;
    _dec          = http::chunked_decoder(spec.error_body_max);
    if (spec.method == http::method::head) {
      c.parser.reset(http::parse_options{
        .header_limit = static_cast<std::uint32_t>(conn_header_bytes), .head_request = true});
    }
    std::span<iovec const> const first =
      _body_sent ? spec.tx : spec.tx.first(std::min(spec.head_iovs, spec.tx.size()));
    _tx_inflight = true;
    pool.send(c, first);
    if (auto ec = pool.recv_header(c, true)) return fail(c, ec);
    return step::in_progress;
  }

  // ---- connection_pool Handler callbacks, forwarded by the owner ----

  step on_send(conn_slot& c, std::error_code ec, std::size_t) noexcept
  {
    _tx_inflight = false;
    if (_finished) return step::done;
    if (ec) {
      if (_res.head_received && c.meta.status >= 300) {
        // The server answered (e.g. 403 to a PUT) and dropped the rest of the body: the status
        // explains the failure; the connection cannot be reused.
        return finish_status(c, ec);
      }
      return fail(c, ec);
    }
    if (_body_wanted && !_body_sent) {
      send_body(c);
      return step::in_progress;
    }
    return maybe_done();
  }

  step on_header(conn_slot& c, std::error_code ec, std::size_t) noexcept
  {
    if (_finished) return step::done;
    if (_phase == phase::drain) return on_drain_bytes(c, ec);
    if (ec) return fail(c, ec);
    for (;;) {
      std::size_t consumed = 0;
      std::error_code pe;
      auto const st = c.parser.parse_header(
        std::span<char const>(c.header_buf.data() + _parse_off, c.hdr_len - _parse_off),
        consumed,
        c.meta,
        pe);
      _parse_off += static_cast<std::uint32_t>(consumed);
      _res.interim = c.parser.interim_responses();
      if (st == http::parse_status::need_more) {
        // Expect: 100-continue: the interim head was consumed inside parse_header.
        if (!_body_sent && _res.interim > 0) request_body(c);
        compact(c);
        if (auto e = _pool->recv_header(c, false)) return fail(c, e);
        return step::in_progress;
      }
      if (st == http::parse_status::error)
        return fail(c, pe ? pe : make_error_code(errc::http_bad_message));
      break;
    }
    _res.head_received = true;
    _res.status        = c.meta.status;
    if (!_body_sent) {
      // A final status before 100 Continue: the body is never sent (B.6); the server may still
      // expect it, so the connection is not reused.
      _res.body_sent = false;
    }
    if (c.meta.status >= 200 && c.meta.status < 300) return on_success_head(c);
    return start_drain(c);  // error status (or a probe's 416): drain, extract the code, classify
  }

  step on_body(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    if (_finished) return step::done;
    if (_phase != phase::body) return fail(c, make_error_code(errc::http_bad_message));
    c.parser.consume_direct(n);
    _res.body_bytes += n;
    if (ec) return fail(c, ec);
    return next_body(c);
  }

  /// @brief `Expect: 100-continue` fallback (1 s without an interim or final head): send the body.
  step continue_timeout(conn_slot& c) noexcept
  {
    if (_finished) return step::done;
    if (!_body_sent && !_res.head_received) request_body(c);
    return step::in_progress;
  }

  [[nodiscard]] exchange_result const& result() const noexcept { return _res; }
  [[nodiscard]] bool finished() const noexcept { return _finished; }
  /// @return true while the head was sent alone and the body waits for 100 Continue.
  [[nodiscard]] bool awaiting_continue() const noexcept { return !_body_sent && !_finished; }
  /// @return the captured prefix of a drained error body.
  [[nodiscard]] std::string_view error_body() const noexcept { return {_err.data(), _err_len}; }
  /// @return the 2xx body drained into `exchange_spec::body_capture` (gpuio T18b).
  [[nodiscard]] std::string_view body_captured() const noexcept
  {
    return {_spec.body_capture.data(), _cap_len};
  }
  /// @return true when that 2xx body did not fit `body_capture` (gpuio T18b).
  [[nodiscard]] bool body_overflow() const noexcept { return _cap_overflow; }

 private:
  enum class phase : std::uint8_t { head, body, drain };

  step on_success_head(conn_slot& c) noexcept
  {
    auto const m = _spec.method;
    if (m == http::method::head) return finish_complete(c, http::outcome{});
    if (m != http::method::get || !_spec.body_capture.empty()) {
      // PUT / POST / DELETE: drain the (usually empty) body like an error body; a remote_call
      // with a body window (gpuio T18b) drains its 2xx body into that window.
      return start_drain(c);
    }
    std::uint64_t obj_off = 0;
    std::uint64_t len     = 0;
    if (_spec.suffix) {
      if (c.meta.body != http::framing::length || !c.meta.content_length)
        return fail_outcome(c, make_error_code(errc::http_bad_message));
      len = *c.meta.content_length;
      if (c.meta.status == 206) {
        if (!c.meta.range || c.meta.range->unsatisfied) {
          // No usable Content-Range (MinIO answers an empty object with "bytes 0--1/0"): the
          // owner learns nothing from the body and falls back to a HEAD.
          if (len == 0) return finish_complete(c, http::outcome{});
          _res.reusable = false;
          return finish_body_skipped(c);
        }
        obj_off = c.meta.range->first;
        if (c.meta.range->last - c.meta.range->first + 1 != len)
          return fail_outcome(c, make_error_code(errc::http_length_mismatch));
      }
    } else {
      // Validate before any body byte lands (§7.3: a wrong Content-Range fails hard).
      http::outcome const o = http::classify(c.meta,
                                             {},
                                             m,
                                             _spec.ranged ? _spec.first : 0,
                                             _spec.ranged ? _spec.last : http::unknown_length,
                                             _spec.object_size,
                                             0,
                                             c.reused);
      if (o.ec) return finish_failed_head(c, o);
      if (c.meta.body != http::framing::length || !c.meta.content_length) {
        // Chunked / EOF-delimited 2xx GET bodies are not used by S3-compatible stores.
        return fail_outcome(c, make_error_code(errc::http_bad_message));
      }
      len     = *c.meta.content_length;
      obj_off = c.meta.status == 206 && c.meta.range ? c.meta.range->first : 0;
    }
    exchange_window const w = _owner->body_window(c, obj_off, len);
    if (w.skip) {
      _res.reusable = false;
      return finish_body_skipped(c);
    }
    c.sink = http::iovec_sink(w.iov);
    if (auto e = c.sink.init(c.meta.content_length)) return fail_outcome(c, e);
    _phase = phase::body;
    // The only copy per GET: body bytes that shared the header receive (§3.5). The header buffer
    // is the control-record drain target from here on (§4.2), so it is consumed now.
    std::size_t const left = c.hdr_len - _parse_off;
    std::size_t const take =
      static_cast<std::size_t>(std::min<std::uint64_t>(left, c.parser.body_remaining()));
    if (take != 0) {
      c.sink.copy_in(std::span<char const>(c.header_buf.data() + _parse_off, take));
      c.parser.consume_direct(take);
      _res.body_bytes += take;
    }
    return next_body(c);
  }

  step next_body(conn_slot& c) noexcept
  {
    std::uint64_t const rem = c.parser.body_remaining();
    if (rem == 0) {
      _res.complete = true;
      _res.reusable = c.meta.keep_alive && _body_sent;
      _res.outcome =
        http::outcome{{}, false, http::outcome::retry_class::none, c.meta.keep_alive, false};
      _phase = phase::head;
      return finish();
    }
    if (!_owner->keep_receiving(c, rem)) return fail(c, make_error_code(errc::operation_canceled));
    if (auto e = _pool->recv_body(c, rem)) return fail(c, e);
    return step::in_progress;
  }

  // ---- error-body drain (and PUT/POST response bodies) ----

  step start_drain(conn_slot& c) noexcept
  {
    _phase   = phase::drain;
    _cap_2xx = !_spec.body_capture.empty() && c.meta.status >= 200 && c.meta.status < 300;
    std::size_t const left = c.hdr_len - _parse_off;
    char const* const p    = c.header_buf.data() + _parse_off;
    switch (c.meta.body) {
      case http::framing::none: return finish_status(c, {});
      case http::framing::until_eof:
        capture(p, left);
        _res.reusable = false;
        return finish_status_close(c);
      case http::framing::length: {
        std::uint64_t const len = c.meta.content_length.value_or(0);
        auto const take         = static_cast<std::size_t>(std::min<std::uint64_t>(left, len));
        capture(p, take);
        _drain_left = len - take;
        if (_drain_left == 0) return finish_status(c, {});
        if (len > (_cap_2xx ? _spec.body_capture.size() : _spec.error_body_max)) {
          _cap_overflow = _cap_overflow || _cap_2xx;
          return finish_status_close(c);
        }
        break;
      }
      case http::framing::chunked: {
        std::size_t consumed = 0;
        std::error_code ec;
        auto const st = feed_chunked(std::span<char const>(p, left), consumed, ec);
        if (st == http::parse_status::ok) return finish_status(c, {});
        if (st == http::parse_status::error) return finish_status_close(c);
        // keep the unconsumed partial chunk line for the next receive
        std::memmove(c.header_buf.data(), p + consumed, left - consumed);
        c.hdr_len  = static_cast<std::uint32_t>(left - consumed);
        _parse_off = 0;
        if (auto e = _pool->recv_header(c, false)) return finish_status(c, e);
        return step::in_progress;
      }
    }
    c.hdr_len  = 0;
    _parse_off = 0;
    if (auto e = _pool->recv_header(c, false)) return finish_status(c, e);
    return step::in_progress;
  }

  step on_drain_bytes(conn_slot& c, std::error_code ec) noexcept
  {
    if (ec) return finish_status(c, ec);
    char const* const p = c.header_buf.data();
    std::size_t const n = c.hdr_len;
    if (c.meta.body == http::framing::chunked) {
      std::size_t consumed = 0;
      std::error_code de;
      auto const st = feed_chunked(std::span<char const>(p, n), consumed, de);
      if (st == http::parse_status::ok) return finish_status(c, {});
      if (st == http::parse_status::error) return finish_status_close(c);
      std::memmove(c.header_buf.data(), p + consumed, n - consumed);
      c.hdr_len = static_cast<std::uint32_t>(n - consumed);
    } else {
      auto const take = static_cast<std::size_t>(std::min<std::uint64_t>(n, _drain_left));
      capture(p, take);
      _drain_left -= take;
      c.hdr_len = 0;
      if (_drain_left == 0) return finish_status(c, {});
    }
    if (auto e = _pool->recv_header(c, false)) return finish_status(c, e);
    return step::in_progress;
  }

  http::parse_status feed_chunked(std::span<char const> in,
                                  std::size_t& consumed,
                                  std::error_code& ec) noexcept
  {
    return _dec.feed(
      in, consumed, [this](std::span<char const> d) noexcept { capture(d.data(), d.size()); }, ec);
  }

  void capture(char const* p, std::size_t n) noexcept
  {
    if (_cap_2xx) {  // gpuio T18b: 2xx body of a remote_call into its window
      std::size_t const room = _spec.body_capture.size() - _cap_len;
      std::size_t const k    = std::min(n, room);
      if (k != 0) std::memcpy(_spec.body_capture.data() + _cap_len, p, k);
      _cap_len += k;
      if (k < n) _cap_overflow = true;
      return;
    }
    std::size_t const room = _err.size() - _err_len;
    std::size_t const k    = std::min(n, room);
    if (k != 0) std::memcpy(_err.data() + _err_len, p, k);
    _err_len += static_cast<std::uint32_t>(k);
  }

  // ---- endings ----

  /// The response head (and its body, drained or not) is final: extract the S3 code, classify.
  step finish_status(conn_slot& c, std::error_code transport) noexcept
  {
    if (!transport && _phase == phase::drain) {
      // A drained body: the connection is intact.
      _res.complete = true;
      _res.reusable = c.meta.keep_alive && _body_sent && !_tx_inflight;
    }
    if (_err_len != 0) (void)http::extract_error_code(error_body(), c.meta);
    _res.transport = transport;
    _res.outcome   = http::classify(c.meta,
                                  transport,
                                  _spec.method,
                                  _spec.ranged ? _spec.first : 0,
                                  _spec.ranged ? _spec.last : http::unknown_length,
                                  _spec.object_size,
                                  _res.body_bytes,
                                  c.reused);
    if (transport) _res.reusable = false;
    if (_spec.suffix && c.meta.status == 416) _res.outcome.ec = {};  // empty object: owner decides
    if (c.meta.status >= 200 && c.meta.status < 300 && !transport) _res.outcome.ec = {};
    if (!_res.outcome.reuse_connection) _res.reusable = false;
    return _res.reusable ? finish() : finish_now();
  }

  step finish_status_close(conn_slot& c) noexcept
  {
    _res.reusable = false;
    if (_err_len != 0) (void)http::extract_error_code(error_body(), c.meta);
    _res.outcome = http::classify(c.meta,
                                  {},
                                  _spec.method,
                                  _spec.ranged ? _spec.first : 0,
                                  _spec.ranged ? _spec.last : http::unknown_length,
                                  _spec.object_size,
                                  _res.body_bytes,
                                  c.reused);
    if (_spec.suffix && c.meta.status == 416) _res.outcome.ec = {};
    if (c.meta.status >= 200 && c.meta.status < 300) _res.outcome.ec = {};
    _res.outcome.reuse_connection = false;
    return finish_now();
  }

  /// 2xx head rejected by classify (Content-Range mismatch, unsliceable 200): close.
  step finish_failed_head(conn_slot&, http::outcome const& o) noexcept
  {
    _res.outcome                  = o;
    _res.outcome.reuse_connection = false;
    _res.reusable                 = false;
    return finish_now();
  }

  step finish_complete(conn_slot& c, http::outcome o) noexcept
  {
    _res.complete = true;
    _res.reusable = c.meta.keep_alive && _body_sent;
    _res.outcome  = o;
    return finish();
  }

  step finish_body_skipped(conn_slot&) noexcept
  {
    _res.complete = false;
    _res.reusable = false;
    _res.outcome  = http::outcome{};
    return finish_now();
  }

  /// A failed exchange with the head already parsed (bad framing / length): close.
  step fail_outcome(conn_slot&, std::error_code ec) noexcept
  {
    _res.outcome  = http::outcome{ec, false, http::outcome::retry_class::none, false, false};
    _res.reusable = false;
    return finish_now();
  }

  step fail(conn_slot& c, std::error_code ec) noexcept
  {
    _res.transport = ec;
    _res.reusable  = false;
    _res.complete  = false;
    _res.outcome   = http::classify(c.meta,
                                  ec,
                                  _spec.method,
                                  _spec.ranged ? _spec.first : 0,
                                  _spec.ranged ? _spec.last : http::unknown_length,
                                  _spec.object_size,
                                  _res.body_bytes,
                                  c.reused);
    return finish_now();
  }

  /// Success path: wait for the send CQE before declaring the exchange done (check_in requires
  /// nothing in flight).
  step finish() noexcept
  {
    _pending_done = true;
    return maybe_done();
  }
  step maybe_done() noexcept
  {
    if (!_pending_done || _tx_inflight) return step::in_progress;
    _finished = true;
    return step::done;
  }
  /// Failure path: the owner closes the connection, which cuts a running send short.
  step finish_now() noexcept
  {
    _finished = true;
    return step::done;
  }

  void request_body(conn_slot& c) noexcept
  {
    if (_body_sent) return;
    if (_tx_inflight) {
      _body_wanted = true;  // sent from on_send (one send in flight per connection)
      return;
    }
    send_body(c);
  }
  void send_body(conn_slot& c) noexcept
  {
    _body_wanted = false;
    _body_sent   = true;
    _tx_inflight = true;
    _pool->send(c, _spec.tx.subspan(std::min(_spec.head_iovs, _spec.tx.size())));
  }

  /// Keep the unconsumed tail of the head at the front of the header buffer.
  void compact(conn_slot& c) noexcept
  {
    if (_parse_off == 0) return;
    std::size_t const tail = c.hdr_len - _parse_off;
    if (tail != 0) std::memmove(c.header_buf.data(), c.header_buf.data() + _parse_off, tail);
    c.hdr_len  = static_cast<std::uint32_t>(tail);
    _parse_off = 0;
  }

  Pool* _pool{nullptr};
  Owner* _owner{nullptr};
  exchange_spec _spec{};
  exchange_result _res{};
  http::chunked_decoder _dec{};
  std::uint64_t _drain_left{0};
  std::uint32_t _parse_off{0};
  std::uint32_t _err_len{0};
  std::size_t _cap_len{0};  ///< gpuio T18b: bytes in exchange_spec::body_capture
  bool _cap_2xx{false}, _cap_overflow{false};
  phase _phase{phase::head};
  bool _tx_inflight{false};
  bool _body_sent{true};
  bool _body_wanted{false};
  bool _pending_done{false};
  bool _finished{false};
  std::array<char, error_capture_bytes> _err{};
};

}  // namespace cucascade::io::net
