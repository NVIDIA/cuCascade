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

#include "io/rest/net/connection_pool.hpp"
#include "net_test_support.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @file get_client.hpp
 * @brief A minimal HTTP/1.1 GET driver over the pool (gpuio tests/unit/net/get_client.hpp),
 *        written exactly as the REST worker uses it (connection_pool.hpp "Exchange protocol"):
 *        checkout, head in `tx_buf`, `send` + `recv_header(after_send)`, incremental
 *        `parse_header`, leftover `copy_in`, `recv_body(parser.body_remaining())` until 0,
 *        check_in / close; stale-reuse free retry on errc::stale_connection (<= 5). Used as the
 *        pool's Handler.
 */

namespace net_test {

using cucascade::io::net::conn_slot;

template <class Env, class Tls = cucascade::io::net::no_tls>
class get_client {
 public:
  using pool_t = cucascade::io::net::connection_pool<Env, get_client<Env, Tls>, Tls>;

  struct job {
    std::uint32_t endpoint{0};
    std::string path{};
    std::optional<std::pair<std::uint64_t, std::uint64_t>> range{};
    std::vector<iovec> dst{};
    bool head{false};
    cucascade::io::rest::priority prio{cucascade::io::rest::priority::normal};
    // ---- results ----
    bool done{false};
    std::error_code ec{};
    std::uint16_t status{0};
    std::uint64_t body{0};
    std::uint64_t content_length{0};
    unsigned attempts{0};
    unsigned stale_retries{0};
    bool reused{false};
    std::uint32_t slot{~0u};  ///< slot index of the last attempt
    std::uint64_t t_start{0}, t_header{0}, t_end{0};
    // ---- internal ----
    conn_slot* c{nullptr};
    bool sent{false}, body_done{false};
    std::uint32_t parse_off{0};
    iovec tx{};
  };

  get_client(cucascade::io::net::remote_context& rc, Env env) : _pool(std::move(env), *this, rc)
  {
    _waiting.reserve(1024);
  }
  get_client(get_client const&)            = delete;
  get_client& operator=(get_client const&) = delete;

  [[nodiscard]] pool_t& pool() noexcept { return _pool; }

  /// Start (or restart) @p j: an idle connection now, else connect-ahead and wait.
  void start(job& j) noexcept
  {
    j.done   = false;
    j.ec     = {};
    j.c      = nullptr;
    j.status = 0;
    j.body   = 0;
    if (j.t_start == 0) j.t_start = now_ns();
    if (conn_slot* c = _pool.checkout(j.endpoint, j.prio)) {
      begin(j, *c);
      return;
    }
    (void)_pool.connect_ahead(j.endpoint, 1, j.prio);
    _waiting.push_back(&j);
  }
  /// Retry waiting jobs (call after connections became ready).
  void pump() noexcept
  {
    for (std::size_t i = 0; i < _waiting.size();) {
      job& j = *_waiting[i];
      if (conn_slot* c = _pool.checkout(j.endpoint, j.prio)) {
        _waiting.erase(_waiting.begin() + static_cast<std::ptrdiff_t>(i));
        begin(j, *c);
        continue;
      }
      ++i;
    }
  }

  // ---- Handler (connection_pool.hpp) ----
  void on_send(conn_slot& c, std::error_code ec, std::size_t) noexcept
  {
    job& j = *static_cast<job*>(c.owner);
    if (ec) {
      fail(j, c, ec);
      return;
    }
    j.sent = true;
    maybe_complete(j, c);
  }
  void on_header(conn_slot& c, std::error_code ec, std::size_t) noexcept
  {
    job& j = *static_cast<job*>(c.owner);
    if (ec) {
      fail(j, c, ec);
      return;
    }
    std::size_t consumed = 0;
    std::error_code pe;
    auto const st = c.parser.parse_header(
      std::span<char const>(c.header_buf.data() + j.parse_off, c.hdr_len - j.parse_off),
      consumed,
      c.meta,
      pe);
    j.parse_off += static_cast<std::uint32_t>(consumed);
    if (st == cucascade::io::http::parse_status::need_more) {
      if (auto e = _pool.recv_header(c, false)) fail(j, c, e);
      return;
    }
    if (st == cucascade::io::http::parse_status::error) {
      fail(j, c, pe ? pe : make_error_code(cucascade::io::rest::errc::http_bad_message));
      return;
    }
    j.t_header       = now_ns();
    j.status         = c.meta.status;
    j.content_length = c.meta.content_length.value_or(0);
    c.sink           = cucascade::io::http::iovec_sink(std::span<iovec const>(j.dst));
    if (!j.head) {
      if (auto e = c.sink.init(c.meta.content_length)) {
        fail(j, c, e);
        return;
      }
    }
    std::span<char const> const left(c.header_buf.data() + j.parse_off, c.hdr_len - j.parse_off);
    std::size_t const take =
      std::min<std::size_t>(left.size(), static_cast<std::size_t>(c.parser.body_remaining()));
    if (take != 0) {
      c.sink.copy_in(left.first(take));
      c.parser.consume_direct(take);
      j.body += take;
    }
    std::uint64_t const rem = c.parser.body_remaining();
    if (rem == 0 || j.head) {
      j.body_done = true;
      maybe_complete(j, c);
      return;
    }
    if (auto e = _pool.recv_body(c, rem)) fail(j, c, e);
  }
  void on_body(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    job& j = *static_cast<job*>(c.owner);
    c.parser.consume_direct(n);
    j.body += n;
    if (ec) {
      fail(j, c, ec);
      return;
    }
    std::uint64_t const rem = c.parser.body_remaining();
    if (rem != 0) {
      if (auto e = _pool.recv_body(c, rem)) fail(j, c, e);
      return;
    }
    j.body_done = true;
    maybe_complete(j, c);
  }
  void on_ready(std::uint32_t) noexcept
  {
    ++ready_events;
    pump();
  }
  void on_connect_failed(std::uint32_t ep, std::error_code ec) noexcept
  {
    ++connect_failures;
    last_connect_error = ec;
    for (std::size_t i = 0; i < _waiting.size(); ++i)
      if (_waiting[i]->endpoint == ep) {
        job& j = *_waiting[i];
        _waiting.erase(_waiting.begin() + static_cast<std::ptrdiff_t>(i));
        j.done  = true;
        j.ec    = ec;
        j.t_end = now_ns();
        ++failed;
        finished.fetch_add(1, std::memory_order_release);
        return;
      }
  }
  std::size_t build_probe(std::uint32_t, conn_slot&, std::span<char> out) noexcept
  {
    if (probe_path.empty()) return 0;
    int const n = std::snprintf(out.data(),
                                out.size(),
                                "HEAD %s HTTP/1.1\r\nHost: %s\r\n\r\n",
                                probe_path.c_str(),
                                host.c_str());
    return n > 0 ? static_cast<std::size_t>(n) : 0;
  }

  [[nodiscard]] std::size_t waiting() const noexcept { return _waiting.size(); }

  // ---- knobs and results ----
  std::string host{"localhost"};
  std::string probe_path{};
  bool free_retry{true};
  bool keep_alive{true};
  unsigned completed{0}, failed{0}, stale_retries{0}, ready_events{0}, connect_failures{0};
  std::atomic<unsigned> finished{0};  ///< jobs done (success or failure)
  std::error_code last_connect_error{};

 private:
  void begin(job& j, conn_slot& c) noexcept
  {
    j.c         = &c;
    c.owner     = &j;
    j.sent      = false;
    j.body_done = false;
    j.parse_off = 0;
    j.reused    = c.reused;
    j.slot      = c.index;
    ++j.attempts;
    if (j.head) c.parser.reset(cucascade::io::http::parse_options{.head_request = true});
    char range[96] = "";
    if (j.range)
      std::snprintf(range,
                    sizeof range,
                    "Range: bytes=%llu-%llu\r\n",
                    static_cast<unsigned long long>(j.range->first),
                    static_cast<unsigned long long>(j.range->second));
    int const n = std::snprintf(c.tx_buf.data(),
                                c.tx_buf.size(),
                                "%s %s HTTP/1.1\r\nHost: %s\r\n%sConnection: %s\r\n\r\n",
                                j.head ? "HEAD" : "GET",
                                j.path.c_str(),
                                host.c_str(),
                                range,
                                (keep_alive && !c.last_request) ? "keep-alive" : "close");
    j.tx        = iovec{c.tx_buf.data(), static_cast<std::size_t>(n)};
    _pool.send(c, std::span<iovec const>(&j.tx, 1));
    if (auto e = _pool.recv_header(c, true)) fail(j, c, e);
  }
  void maybe_complete(job& j, conn_slot& c) noexcept
  {
    if (!j.sent || !j.body_done) return;
    j.done  = true;
    j.ec    = {};
    j.t_end = now_ns();
    j.c     = nullptr;
    ++completed;
    finished.fetch_add(1, std::memory_order_release);
    if (!c.meta.keep_alive || !keep_alive) c.close_after = true;
    _pool.check_in(c);
    pump();  // a connection came back: the worker retries held work every pass
  }
  void fail(job& j, conn_slot& c, std::error_code ec) noexcept
  {
    _pool.close(c);
    j.c = nullptr;
    if (ec == cucascade::io::rest::errc::stale_connection && free_retry && j.stale_retries < 5) {
      // free_stale: immediate retry on a fresh connection, no backoff, no attempt charge.
      ++j.stale_retries;
      ++stale_retries;
      start(j);
      return;
    }
    j.done  = true;
    j.ec    = ec;
    j.t_end = now_ns();
    ++failed;
    finished.fetch_add(1, std::memory_order_release);
  }

  pool_t _pool;
  std::vector<job*> _waiting;
};

}  // namespace net_test
