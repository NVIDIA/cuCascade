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

/// @file uring_transport.cpp
/// @brief The REST reactor's io_uring HTTP/1.1 transport.  Structure ported from uring_ultra's
/// `uring_https_transport` (slots, the pending / cancel discipline, connect over the resolved
/// addresses, the POLL_ADD-driven handshake, direct body receives); TLS, HTTP parsing and request
/// heads come from cuCascade's TLS layer and codec (see uring_transport.hpp).

#include "io/rest/net/uring_transport.hpp"

#include "io/rest/errc.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/http/request_head.hpp"
#include "io/rest/net/tls_stream.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace cucascade::io::net {

namespace {

using clock_type = std::chrono::steady_clock;

constexpr std::uint64_t cancel_bit   = std::uint64_t{1} << 63;
constexpr std::uint64_t external_bit = std::uint64_t{1} << 62;
/// Error bodies up to this size are drained to keep the connection (http::max_error_body).
constexpr std::size_t max_drained_body = http::max_error_body;
/// Request head storage per connection (presigned URLs and SigV4 headers fit easily).
constexpr std::size_t head_bytes = 16 * KiB;
/// Control records in a row before the connection is declared broken.
constexpr unsigned max_control_run = 32;
/// kTLS control-record drain buffer: one full record.
constexpr std::size_t drain_bytes = tls::max_record_ciphertext + 256;
/// Sends gather at most this many iovecs.
constexpr std::size_t max_send_iovecs = 64;

std::atomic<mtu_probe_fn> g_mtu_probe{nullptr};

std::error_code sys_error(int err) noexcept { return {err, std::system_category()}; }

std::string read_small(std::string const& path)
{
  std::ifstream in(path);
  std::string line;
  if (in) { std::getline(in, line); }
  return line;
}

}  // namespace

// ================================================================================================
// helpers
// ================================================================================================

std::string url_parts::origin() const
{
  return std::string(tls ? "https://" : "http://") + host + ":" + std::to_string(port);
}

url_parts parse_url(std::string_view url)
{
  url_parts out;
  std::string_view v = url;
  if (v.starts_with("https://")) {
    out.tls = true;
    v.remove_prefix(8);
  } else if (v.starts_with("http://")) {
    out.tls = false;
    v.remove_prefix(7);
  } else {
    throw std::invalid_argument("rest transport: only http:// and https:// URLs are supported");
  }
  auto const slash     = v.find_first_of("/?");
  auto const authority = v.substr(0, slash);
  std::string target   = slash == std::string_view::npos ? "/" : std::string(v.substr(slash));
  if (!target.empty() && target.front() == '?') { target.insert(target.begin(), '/'); }
  if (auto const hash = target.find('#'); hash != std::string::npos) { target.resize(hash); }
  if (authority.find('@') != std::string_view::npos) {
    throw std::invalid_argument("rest transport: URL userinfo is unsupported");
  }
  std::string_view host = authority;
  std::string_view port;
  if (authority.starts_with('[')) {
    auto const close = authority.find(']');
    if (close == std::string_view::npos) {
      throw std::invalid_argument("rest transport: bad IPv6 host");
    }
    host = authority.substr(1, close - 1);
    if (close + 1 < authority.size()) {
      if (authority[close + 1] != ':') { throw std::invalid_argument("rest transport: bad host"); }
      port = authority.substr(close + 2);
    }
  } else if (auto const colon = authority.rfind(':'); colon != std::string_view::npos) {
    host = authority.substr(0, colon);
    port = authority.substr(colon + 1);
  }
  std::uint16_t const default_port = out.tls ? 443 : 80;
  out.port                         = default_port;
  if (!port.empty()) {
    unsigned value  = 0;
    auto const conv = std::from_chars(port.data(), port.data() + port.size(), value);
    if (conv.ec != std::errc{} || conv.ptr != port.data() + port.size() || value == 0 ||
        value > 65535) {
      throw std::invalid_argument("rest transport: bad port in URL");
    }
    out.port = static_cast<std::uint16_t>(value);
  }
  out.host = std::string(host);
  if (out.host.empty() || target.find_first_of("\r\n ") != std::string::npos) {
    throw std::invalid_argument("rest transport: invalid HTTP authority or target");
  }
  bool const v6 = out.host.find(':') != std::string::npos;
  out.authority = v6 ? "[" + out.host + "]" : out.host;
  if (out.port != default_port) { out.authority += ":" + std::to_string(out.port); }
  out.target = std::move(target);
  return out;
}

mtu_sample mtu_probe(int fd) noexcept
{
  mtu_sample s{};
  int mss       = 0;
  socklen_t len = sizeof(mss);
  if (::getsockopt(fd, IPPROTO_TCP, TCP_MAXSEG, &mss, &len) != 0) { return s; }
  tcp_info info{};
  socklen_t info_len = sizeof(info);
  if (::getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &info_len) == 0) {
    s.timestamps = (info.tcpi_options & TCPI_OPT_TIMESTAMPS) != 0;
  }
  sockaddr_storage local{};
  socklen_t local_len = sizeof(local);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &local_len) == 0) {
    s.ipv6 = local.ss_family == AF_INET6;
  }
  s.mss = mss;
  s.ok  = true;
  return s;
}

void set_mtu_probe_for_testing(mtu_probe_fn fn) noexcept
{
  g_mtu_probe.store(fn, std::memory_order_relaxed);
}

bool ec2_platform(std::string_view sysfs_root) noexcept
{
  try {
    std::string const root(sysfs_root);
    for (char const* f :
         {"/devices/virtual/dmi/id/sys_vendor", "/devices/virtual/dmi/id/board_vendor"}) {
      if (read_small(root + f).find("Amazon EC2") != std::string::npos) { return true; }
    }
    auto const uuid = read_small(root + "/hypervisor/uuid");
    return uuid.starts_with("ec2") || uuid.starts_with("EC2");
  } catch (...) {
    return false;
  }
}

bool ktls_recv_async(tls::recv_placement placement, bool on_ec2) noexcept
{
  switch (placement) {
    case tls::recv_placement::iowq: return true;
    case tls::recv_placement::inline_: return false;
    case tls::recv_placement::automatic: return on_ec2;
  }
  return false;
}

// ================================================================================================
// connection state
// ================================================================================================

enum class uring_transport::op : std::uint8_t {
  none,
  connect,
  handshake,
  send,
  recv,
  drain,
  writable,       ///< POLLOUT before flushing an owed kTLS KeyUpdate reply
  continue_poll,  ///< POLLIN while a held body waits for `100 Continue` (continue_timeout)
};

namespace {

enum class conn_kind : std::uint8_t { none, plain, ktls, user };
enum class phase : std::uint8_t { idle, dns, connecting, handshaking, sending, head, body };
enum class body_mode : std::uint8_t { none, sink, buffer, discard, error_body };

}  // namespace

struct uring_transport::connection {
  // ---- socket / TLS ----
  int fd{-1};
  conn_kind kind{conn_kind::none};
  std::unique_ptr<tls::session> session;
  tls_stream* stream{nullptr};
  tls::ktls::rekey_state* rekey{nullptr};
  url_parts remote;
  std::string origin;
  bool established{false};
  std::uint64_t bytes_carried{0};
  ktls_reply reply;  ///< kTLS: a KeyUpdate reply owed to the peer

  // ---- flight discipline ----
  bool pending{false};
  bool cancel_pending{false};
  bool canceling{false};
  op cur{op::none};
  phase ph{phase::idle};
  std::error_code timeout_ec{};

  // ---- time ----
  clock_type::time_point idle_since{};
  std::optional<clock_type::time_point> connect_deadline;
  std::optional<clock_type::time_point> exchange_deadline;
  bool stall_on{false};
  clock_type::time_point stall_start{};
  std::uint64_t stall_base{0};
  std::uint64_t progress{0};

  // ---- connect ----
  std::uint32_t dns_handle{dns_resolver::invalid};
  bool literal{false};
  sockaddr_storage addr{};
  socklen_t addr_len{0};
  std::size_t connect_tries{0};
  unsigned mtu_rejects{0};

  // ---- exchange ----
  bool has_request{false};
  exchange x;
  std::vector<char> head;
  std::vector<iovec> tx;
  std::size_t tx_idx{0};
  bool body_held{false};      ///< expect-continue: body not sent yet
  bool continue_wait{false};  ///< head received while waiting for 100 Continue
  /// expect-continue with a timeout: when the held body goes out without an answer.
  std::optional<clock_type::time_point> continue_deadline;
  bool continue_cancel{false};  ///< the continue poll is being cancelled (deadline)
  bool continue_ready{false};   ///< the continue poll fired: receive, do not poll again
  int continue_res{0};          ///< the continue poll's result, held until its cancel is reaped
  bool generator_done{false};
  msghdr txmsg{};
  http::response_parser parser;
  http::response_meta meta;
  bool head_done{false};
  std::vector<char> scratch;
  std::size_t sbeg{0};
  std::size_t slen{0};
  std::vector<iovec> rx_iov;
  bool direct{false};  ///< the receive in flight targets the sink
  msghdr rxmsg{};
  ktls_cmsg_buffer cmsg;
  std::vector<std::byte> drain_buf;
  http::iovec_sink sink;
  http::chunked_decoder chunked;
  body_mode bmode{body_mode::none};
  bool body_done{false};
  bool no_reuse{false};
  bool reused{false};
  unsigned control_run{0};
  exchange_result res;

  [[nodiscard]] std::size_t window_bytes() const noexcept
  {
    std::size_t n = 0;
    for (auto const& v : rx_iov) {
      n += v.iov_len;
    }
    return n;
  }
};

// ================================================================================================
// construction / teardown
// ================================================================================================

uring_transport::uring_transport(transport_options opts, transport_services services)
  : _opts(std::move(opts)), _services(std::move(services))
{
  if (_opts.max_connections == 0) {
    throw std::invalid_argument("rest transport: max_connections must be > 0");
  }
  _opts.queue_depth = std::max(_opts.queue_depth, 2 * _opts.max_connections);
  _slots.reserve(_opts.max_connections);
  for (std::size_t i = 0; i < _opts.max_connections; ++i) {
    _slots.push_back(std::make_unique<connection>());
  }
  io_uring_params params{};
  if (_opts.sqpoll) {
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 10;
  }
  if (_opts.single_issuer || _opts.defer_taskrun) { params.flags |= IORING_SETUP_SINGLE_ISSUER; }
  if (_opts.single_issuer && !_opts.sqpoll && !_opts.defer_taskrun) {
    params.flags |= IORING_SETUP_COOP_TASKRUN;
  }
  if (_opts.defer_taskrun) { params.flags |= IORING_SETUP_DEFER_TASKRUN; }
  auto const rc =
    io_uring_queue_init_params(static_cast<unsigned>(_opts.queue_depth), &_ring, &params);
  if (rc < 0) {
    throw std::system_error(-rc, std::system_category(), "rest transport: io_uring_queue_init");
  }
  _ring_live = true;
  if (_opts.register_ring_fd) { (void)io_uring_register_ring_fd(&_ring); }
  if (_opts.iowq_max_bound != 0 || _opts.iowq_max_unbound != 0) {
    unsigned values[2] = {_opts.iowq_max_bound, _opts.iowq_max_unbound};
    (void)io_uring_register_iowq_max_workers(&_ring, values);
  }
}

uring_transport::~uring_transport() { shutdown(); }

void uring_transport::shutdown() noexcept
{
  if (!_ring_live) { return; }
  // Never release sockets, msghdrs or caller memory while the kernel may still reference them:
  // cancel every primary SQE and reap until both CQEs of every slot are in.
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    auto& c = *_slots[i];
    if (c.pending && !c.canceling && !c.cancel_pending) {
      try {
        cancel(i);
      } catch (...) {
        if (c.fd >= 0) { ::shutdown(c.fd, SHUT_RDWR); }
      }
    }
  }
  auto any_pending = [&] {
    return std::any_of(
      _slots.begin(), _slots.end(), [](auto const& c) { return c->pending || c->cancel_pending; });
  };
  while (any_pending()) {
    auto const submitted = io_uring_submit(&_ring);
    if (submitted < 0 && submitted != -EINTR) {
      for (auto& c : _slots) {
        if (c->fd >= 0) { ::shutdown(c->fd, SHUT_RDWR); }
      }
    }
    io_uring_cqe* cqe = nullptr;
    if (io_uring_wait_cqe(&_ring, &cqe) < 0) { continue; }
    auto const id = io_uring_cqe_get_data64(cqe);
    io_uring_cqe_seen(&_ring, cqe);
    if ((id & external_bit) != 0) { continue; }
    auto& c = *_slots[static_cast<std::size_t>((id & ~cancel_bit) - 1)];
    if ((id & cancel_bit) != 0) {
      c.cancel_pending = false;
    } else {
      c.pending = false;
    }
  }
  io_uring_queue_exit(&_ring);
  _ring_live = false;
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    close(i);
  }
}

// ================================================================================================
// public entry points
// ================================================================================================

int uring_transport::timeout_ms() const noexcept
{
  if (!_completed.empty()) { return 0; }
  std::optional<clock_type::time_point> earliest;
  auto consider = [&](clock_type::time_point t) {
    if (!earliest || t < *earliest) { earliest = t; }
  };
  auto const now = clock_type::now();
  for (auto const& p : _slots) {
    auto const& c = *p;
    if (c.ph == phase::dns && !c.pending) {
      consider(now + std::chrono::milliseconds{1});  // the first answer is on its way
      continue;
    }
    if (!c.pending || c.canceling) { continue; }
    if (c.connect_deadline) { consider(*c.connect_deadline); }
    if (c.exchange_deadline) { consider(*c.exchange_deadline); }
    if (c.stall_on) { consider(c.stall_start + c.x.stall_time); }
    if (c.cur == op::continue_poll && c.continue_deadline && !c.cancel_pending) {
      consider(*c.continue_deadline);
    }
  }
  if (!earliest) { return -1; }
  auto const remaining =
    std::chrono::duration_cast<std::chrono::milliseconds>(*earliest - now).count() + 1;
  return static_cast<int>(std::clamp<std::int64_t>(remaining, 0, INT_MAX));
}

bool uring_transport::busy(std::size_t slot) const noexcept
{
  auto const& c = *_slots.at(slot);
  return c.has_request || c.pending || c.cancel_pending;
}

bool uring_transport::established(std::size_t slot) const noexcept
{
  return _slots.at(slot)->established;
}

void uring_transport::close_idle(std::size_t slot) noexcept
{
  if (slot < _slots.size() && !busy(slot)) { close(slot); }
}

clock_type::time_point uring_transport::idle_since(std::size_t slot) const noexcept
{
  return _slots.at(slot)->idle_since;
}

std::size_t uring_transport::established_count() const noexcept
{
  return static_cast<std::size_t>(
    std::count_if(_slots.begin(), _slots.end(), [](auto const& c) { return c->established; }));
}

void uring_transport::submit(std::size_t index, exchange x)
{
  auto& c = *_slots.at(index);
  if (c.has_request || c.pending || c.cancel_pending) {
    throw std::logic_error("rest transport: occupied slot");
  }
  ++_ctr.requests;
  c.has_request  = true;
  c.x            = std::move(x);
  c.res          = exchange_result{};
  c.reused       = false;
  c.no_reuse     = false;
  auto const now = clock_type::now();
  c.exchange_deadline.reset();
  if (c.x.timeout.count() > 0) { c.exchange_deadline = now + c.x.timeout; }
  try {
    c.remote = parse_url(c.x.url);
  } catch (std::exception const&) {
    finish(index, make_error_code(errc::invalid_argument));
    return;
  }
  try {
    begin(index);
  } catch (transport_fatal_error const&) {
    throw;
  } catch (std::bad_alloc const&) {
    if (!c.pending && !c.cancel_pending && c.has_request) {
      finish(index, make_error_code(std::errc::not_enough_memory));
    }
  }
}

void uring_transport::warmup(std::string const& url)
{
  auto const remote = parse_url(url);
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    auto& c = *_slots[i];
    if (busy(i) || c.established) { continue; }
    c.remote = remote;
    c.x      = exchange{};
    c.exchange_deadline.reset();
    try {
      begin(i);
    } catch (transport_fatal_error const&) {
      throw;
    } catch (...) {
      if (!c.pending && !c.cancel_pending) { close(i); }
    }
  }
  if (!_opts.defer_flush) { flush(); }
}

void uring_transport::flush()
{
  while (io_uring_sq_ready(&_ring) != 0) {
    auto const rc = io_uring_submit(&_ring);
    if (rc == -EINTR || rc == -EAGAIN || rc == -EBUSY) { continue; }
    if (rc < 0) {
      throw transport_fatal_error(std::string("rest transport: io_uring_submit: ") +
                                  std::strerror(-rc));
    }
    if (rc == 0 && !_opts.sqpoll) {
      throw transport_fatal_error("rest transport: io_uring_submit made no progress");
    }
    if (rc == 0) { break; }
  }
}

void uring_transport::wait(int timeout_ms)
{
  io_uring_cqe* cqe = nullptr;
  __kernel_timespec ts{};
  __kernel_timespec* tsp = nullptr;
  if (timeout_ms >= 0) {
    ts.tv_sec  = timeout_ms / 1000;
    ts.tv_nsec = static_cast<long long>(timeout_ms % 1000) * 1'000'000LL;
    tsp        = &ts;
  }
  auto const rc = io_uring_submit_and_wait_timeout(&_ring, &cqe, 1, tsp, nullptr);
  if (rc < 0 && rc != -ETIME && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY) {
    throw transport_fatal_error(std::string("rest transport: submit_and_wait: ") +
                                std::strerror(-rc));
  }
}

void uring_transport::watch_fd(int fd)
{
  auto* entry = sqe(external_bit | static_cast<std::uint32_t>(fd));
  io_uring_prep_poll_multishot(entry, fd, POLLIN);
}

bool uring_transport::take_watch_events() noexcept { return std::exchange(_watch_fired, false); }

void uring_transport::check_deadlines()
{
  auto const now = clock_type::now();
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    auto& c = *_slots[i];
    if (c.ph == phase::dns && !c.pending) {
      if ((c.connect_deadline && now >= *c.connect_deadline) ||
          (c.exchange_deadline && now >= *c.exchange_deadline)) {
        ++_ctr.timeouts;
        finish(i, make_error_code(errc::connect_timeout));
      } else {
        start_connect(i);
      }
      continue;
    }
    if (!c.pending || c.canceling || c.cancel_pending) { continue; }
    if (c.cur == op::continue_poll && c.continue_deadline && now >= *c.continue_deadline) {
      // The server ignored `Expect: 100-continue`: stop waiting and send the body.  A soft
      // cancel: the poll's CQE then resumes the exchange (continue_body) instead of failing it.
      auto* entry = sqe(cancel_bit | (i + 1));
      io_uring_prep_cancel64(entry, i + 1, 0);
      c.cancel_pending  = true;
      c.continue_cancel = true;
      continue;
    }
    if (c.connect_deadline && now >= *c.connect_deadline) {
      ++_ctr.timeouts;
      c.timeout_ec = make_error_code(errc::connect_timeout);
      cancel(i);
      continue;
    }
    if (c.exchange_deadline && now >= *c.exchange_deadline) {
      ++_ctr.timeouts;
      c.timeout_ec = make_error_code(errc::request_timeout);
      cancel(i);
      continue;
    }
    if (c.stall_on && now - c.stall_start >= c.x.stall_time) {
      auto const need = c.x.stall_bytes_per_s * static_cast<std::uint64_t>(c.x.stall_time.count());
      if (c.progress - c.stall_base < need) {
        ++_ctr.timeouts;
        ++_ctr.stalls;
        c.timeout_ec = make_error_code(errc::request_timeout);
        cancel(i);
      } else {
        c.stall_start = now;
        c.stall_base  = c.progress;
      }
    }
  }
}

std::vector<uring_transport::completion>& uring_transport::poll()
{
  io_uring_cqe* cqe = nullptr;
  // Bound one pass: a hot connection must not starve retries or copy polling.
  for (std::size_t count = 0;
       count < _opts.queue_depth && io_uring_peek_cqe(&_ring, &cqe) == 0 && cqe != nullptr;
       ++count) {
    auto const id     = io_uring_cqe_get_data64(cqe);
    auto const result = cqe->res;
    auto const flags  = cqe->flags;
    io_uring_cqe_seen(&_ring, cqe);
    if ((id & external_bit) != 0) {
      _watch_fired = true;
      if ((flags & IORING_CQE_F_MORE) == 0) {  // multishot terminated: re-arm
        auto const fd = static_cast<int>(id & 0xffffffffU);
        auto* entry   = sqe(id);
        io_uring_prep_poll_multishot(entry, fd, POLLIN);
      }
      continue;
    }
    auto const index = static_cast<std::size_t>((id & ~cancel_bit) - 1);
    auto& c          = *_slots.at(index);
    if ((id & cancel_bit) != 0) {
      c.cancel_pending = false;
    } else {
      c.pending = false;
    }
    if (c.continue_cancel) {
      // Both CQEs of the soft cancel first; then the poll's result decides (cancelled: send
      // the body; readable: the answer raced the deadline, receive it).
      if ((id & cancel_bit) == 0) { c.continue_res = result; }
      if (c.pending || c.cancel_pending) { continue; }
      c.continue_cancel = false;
      try {
        advance(index, c.continue_res, 0);
      } catch (transport_fatal_error const&) {
        throw;
      } catch (std::exception const&) {
        if (!c.pending && !c.cancel_pending && c.has_request) {
          finish(index, make_error_code(errc::http_bad_message));
        }
      }
      continue;
    }
    if (c.canceling) {
      if (!c.pending && !c.cancel_pending) {
        // A canceled user-space stream may hold state for an operation that never completed.
        c.no_reuse = true;
        finish(index, c.timeout_ec ? c.timeout_ec : make_error_code(errc::request_timeout));
      }
      continue;
    }
    if ((id & cancel_bit) != 0) { continue; }
    try {
      advance(index, result, flags);
    } catch (transport_fatal_error const&) {
      throw;
    } catch (std::exception const&) {
      if (!c.pending && !c.cancel_pending && c.has_request) {
        finish(index, make_error_code(errc::http_bad_message));
      }
    }
  }
  // Deadlines after the reap: a response that landed by the deadline completes; only an
  // exchange still waiting for the network is cancelled.
  check_deadlines();
  if (!_opts.defer_flush) { flush(); }
  _returned.swap(_completed);
  _completed.clear();
  return _returned;
}

// ================================================================================================
// connection state machine
// ================================================================================================

io_uring_sqe* uring_transport::sqe(std::uint64_t id)
{
  auto* entry = io_uring_get_sqe(&_ring);
  if (entry == nullptr) {
    flush();
    entry = io_uring_get_sqe(&_ring);
  }
  if (entry == nullptr) {
    throw transport_fatal_error("rest transport: submission queue exhausted");
  }
  io_uring_sqe_set_data64(entry, id);
  return entry;
}

void uring_transport::cancel(std::size_t index)
{
  auto& c     = *_slots[index];
  auto* entry = sqe(cancel_bit | (index + 1));
  io_uring_prep_cancel64(entry, index + 1, 0);
  c.canceling      = true;
  c.cancel_pending = true;
}

void uring_transport::close(std::size_t index) noexcept
{
  auto& c = *_slots[index];
  if (c.stream != nullptr) {
    tls_stream::destroy(c.stream);
    c.stream = nullptr;
  }
  if (c.rekey != nullptr) {
    tls_support::release_rekey(c.rekey);
    c.rekey = nullptr;
  }
  c.session.reset();
  if (c.fd >= 0) {
    ++_ctr.closed;
    if (_opts.close_with_rst) {
      // An RST instead of a FIN: neither end lingers in TIME_WAIT / FIN_WAIT.
      linger const lg{1, 0};
      (void)::setsockopt(c.fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    }
    ::close(c.fd);
    c.fd = -1;
  }
  c.kind          = conn_kind::none;
  c.established   = false;
  c.reply         = ktls_reply{};
  c.bytes_carried = 0;
  c.drain_buf.clear();
  c.drain_buf.shrink_to_fit();
}

bool uring_transport::healthy_idle(std::size_t index)
{
  auto& c        = *_slots[index];
  auto const now = clock_type::now();
  if (_opts.conn_max_age.count() > 0 && now - c.idle_since > _opts.conn_max_age) {
    ++_ctr.idle_closes;
    return false;
  }
  // curl's conn_is_alive: an idle HTTP/1.1 connection with input pending is not reusable
  // (RFC 9112 section 9.2), HUP / error mean dead.  TLS connections first consume control
  // records that arrived while idle (TLS 1.3 tickets), as long as the socket stays readable;
  // only a closed or failed stream (or a socket that never stops delivering) retires them.
  constexpr int max_drain_rounds = 64;
  for (int round = 0; round < max_drain_rounds; ++round) {
    pollfd pfd{c.fd, POLLIN | POLLPRI | POLLRDHUP, 0};
    int const r = ::poll(&pfd, 1, 0);
    if (r == 0) { return true; }
    if (r < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL | POLLPRI | POLLRDHUP)) != 0) {
      ++_ctr.dead_closes;
      return false;
    }
    if (c.kind == conn_kind::user && c.stream != nullptr) {
      if (!c.stream->drain_idle_socket()) {
        ++_ctr.dead_closes;
        return false;
      }
      continue;
    }
    if (c.kind == conn_kind::ktls) {
      if (c.drain_buf.empty()) { c.drain_buf.resize(drain_bytes); }
      iovec iov{c.drain_buf.data(), c.drain_buf.size()};
      msghdr msg{};
      ktls_cmsg_buffer cm;
      msg.msg_iov        = &iov;
      msg.msg_iovlen     = 1;
      msg.msg_control    = cm.bytes;
      msg.msg_controllen = sizeof(cm.bytes);
      auto const n       = ::recvmsg(c.fd, &msg, MSG_DONTWAIT);
      auto const got     = interpret_ktls_recv(n < 0 ? -errno : static_cast<int>(n), msg);
      if (got.kind == rx_kind::again) { continue; }
      if (got.kind != rx_kind::control) {
        ++_ctr.dead_closes;
        return false;
      }
      std::error_code ec;
      auto const action =
        tls::ktls::on_control_record(c.fd, c.rekey, got.type, {c.drain_buf.data(), got.bytes}, ec);
      ++_ctr.control_records;
      if (action == tls::ktls::control_action::reply_owed) {
        // A full send buffer keeps the reply owed: the next exchange flushes it first.
        if (c.reply.flush(c.fd, c.rekey, &tls::ktls::flush_reply) == ktls_reply::step::failed) {
          ++_ctr.dead_closes;
          return false;
        }
      } else if (action != tls::ktls::control_action::consumed) {
        ++_ctr.dead_closes;
        return false;
      }
      continue;
    }
    ++_ctr.dead_closes;
    return false;
  }
  ++_ctr.dead_closes;
  return false;
}

void uring_transport::begin(std::size_t index)
{
  auto& c           = *_slots[index];
  auto const origin = c.remote.origin();
  if (c.established && (c.origin != origin || c.x.fresh_connection)) { close(index); }
  if (c.established && !healthy_idle(index)) { close(index); }
  if (c.established) {
    c.reused = true;
    ++_ctr.reused;
    start_exchange(index);
    return;
  }
  close(index);
  c.origin           = origin;
  c.connect_tries    = 0;
  c.mtu_rejects      = 0;
  c.dns_handle       = dns_resolver::invalid;
  c.connect_deadline = clock_type::now() + _opts.connect_timeout;
  c.literal          = parse_ip_literal(c.remote.host, c.remote.port, c.addr, c.addr_len);
  start_connect(index);
}

void uring_transport::start_connect(std::size_t index)
{
  auto& c = *_slots[index];
  if (!c.literal) {
    if (_services.dns == nullptr) {
      finish(index, make_error_code(errc::dns_failed));
      return;
    }
    if (c.dns_handle == dns_resolver::invalid) {
      c.dns_handle = _services.dns->intern(c.remote.host, c.remote.port);
      if (c.dns_handle == dns_resolver::invalid) {
        finish(index, make_error_code(errc::dns_failed));
        return;
      }
    }
    switch (_services.dns->pick(c.dns_handle, c.addr, c.addr_len, mono_ns())) {
      case dns_resolver::status::ok: break;
      case dns_resolver::status::pending: c.ph = phase::dns; return;
      case dns_resolver::status::failed: finish(index, make_error_code(errc::dns_failed)); return;
    }
  }
  open_socket(index);
}

void uring_transport::open_socket(std::size_t index)
{
  auto& c = *_slots[index];
  c.fd    = ::socket(c.addr.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (c.fd < 0) {
    ++_ctr.connect_failures;
    finish(index, make_error_code(errc::connect_failed));
    return;
  }
  int one = 1;
  (void)::setsockopt(c.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  (void)::setsockopt(c.fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
  (void)::setsockopt(c.fd, IPPROTO_TCP, TCP_KEEPIDLE, &_opts.keepalive_idle_s, sizeof(int));
  (void)::setsockopt(c.fd, IPPROTO_TCP, TCP_KEEPINTVL, &_opts.keepalive_interval_s, sizeof(int));
  if (_opts.keepalive_count > 0) {
    (void)::setsockopt(c.fd, IPPROTO_TCP, TCP_KEEPCNT, &_opts.keepalive_count, sizeof(int));
  }
  if (_opts.so_rcvbuf > 0) {
    (void)::setsockopt(c.fd, SOL_SOCKET, SO_RCVBUF, &_opts.so_rcvbuf, sizeof(int));
  }
  if (_opts.so_sndbuf > 0) {
    (void)::setsockopt(c.fd, SOL_SOCKET, SO_SNDBUF, &_opts.so_sndbuf, sizeof(int));
  }
  if (_opts.busy_poll_us > 0) {
    (void)::setsockopt(c.fd, SOL_SOCKET, SO_BUSY_POLL, &_opts.busy_poll_us, sizeof(int));
  }
  if (_opts.tcp_user_timeout_ms > 0) {
    (void)::setsockopt(
      c.fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &_opts.tcp_user_timeout_ms, sizeof(unsigned));
  }
  ++_ctr.connects;
  c.kind      = conn_kind::plain;
  auto* entry = sqe(index + 1);
  io_uring_prep_connect(entry, c.fd, reinterpret_cast<sockaddr*>(&c.addr), c.addr_len);
  c.cur     = op::connect;
  c.ph      = phase::connecting;
  c.pending = true;
}

void uring_transport::on_connected(std::size_t index)
{
  auto& c            = *_slots[index];
  mtu_probe_fn fn    = g_mtu_probe.load(std::memory_order_relaxed);
  mtu_sample const s = fn != nullptr ? fn(c.fd) : mtu_probe(c.fd);
  if (s.ok && s.mss > 0) {
    auto const mss = static_cast<std::uint64_t>(s.mss);
    _ctr.mss_min   = _ctr.mss_min == 0 ? mss : std::min(_ctr.mss_min, mss);
    _ctr.mss_max   = std::max(_ctr.mss_max, mss);
    if (_opts.min_mtu > 0 && effective_mtu(s) < _opts.min_mtu) {
      if (c.mtu_rejects < _opts.mtu_reject_max_attempts) {
        // Reset and reconnect on the next address; the budget is per request.
        ++c.mtu_rejects;
        ++_ctr.mtu_rejects;
        close(index);
        start_connect(index);
        return;
      }
      ++_ctr.mtu_fallbacks;
    }
  }
  if (!c.remote.tls) {
    established_now(index);
    return;
  }
  if (!_tls) {
    try {
      if (!_services.tls) { throw std::system_error(make_error_code(errc::not_supported)); }
      _tls = _services.tls();
    } catch (std::system_error const& e) {
      finish(index, e.code());
      return;
    } catch (...) {
      finish(index, make_error_code(errc::tls_handshake_failed));
      return;
    }
  }
  std::error_code ec;
  c.session = _tls->new_session(c.fd, tls::server_identity{c.remote.host, c.remote.port}, ec);
  if (!c.session) {
    finish(index,
           ec ? tls_support::handshake_error(ec) : make_error_code(errc::tls_handshake_failed));
    return;
  }
  c.ph = phase::handshaking;
  handshake(index);
}

void uring_transport::handshake(std::size_t index)
{
  auto& c = *_slots[index];
  std::error_code ec;
  auto const st = tls_support::handshake(*c.session, ec);
  if (st == tls::handshake_status::done) {
    ++_ctr.handshakes;
    auto r =
      tls_support::finish(c.fd, *c.session, *_tls, _opts.user_space, c.remote.host, c.remote.port);
    if (r.path == tls_path::ktls) {
      c.kind  = conn_kind::ktls;
      c.rekey = r.rekey;
      ++_ctr.ktls_sessions;
    } else if (r.path == tls_path::user_space) {
      c.kind   = conn_kind::user;
      c.stream = r.stream;
      ++_ctr.userspace_tls_sessions;
      if (r.fallback) { ++_ctr.ktls_fallbacks; }
    } else {
      if (r.stream != nullptr) { tls_stream::destroy(r.stream); }
      finish(index, r.ec ? r.ec : make_error_code(errc::tls_handshake_failed));
      return;
    }
    established_now(index);
    return;
  }
  if (st == tls::handshake_status::failed) {
    finish(index, tls_support::handshake_error(ec));
    return;
  }
  auto* entry = sqe(index + 1);
  io_uring_prep_poll_add(entry, c.fd, st == tls::handshake_status::want_read ? POLLIN : POLLOUT);
  c.cur     = op::handshake;
  c.pending = true;
}

void uring_transport::established_now(std::size_t index)
{
  auto& c       = *_slots[index];
  c.established = true;
  c.idle_since  = clock_type::now();
  c.connect_deadline.reset();
  if (c.has_request) {
    start_exchange(index);
  } else {
    c.ph = phase::idle;
  }
}

void uring_transport::start_exchange(std::size_t index)
{
  auto& c = *_slots[index];
  c.connect_deadline.reset();
  c.ph            = phase::sending;
  c.progress      = 0;
  c.stall_on      = c.x.stall_bytes_per_s > 0 && c.x.stall_time.count() > 0;
  c.stall_start   = clock_type::now();
  c.stall_base    = 0;
  c.head_done     = false;
  c.body_done     = false;
  c.continue_wait = false;
  c.continue_deadline.reset();
  c.continue_cancel = false;
  c.continue_ready  = false;
  c.control_run     = 0;
  c.bmode           = body_mode::none;
  c.meta            = http::response_meta{};
  c.parser.reset(http::parse_options{
    static_cast<std::uint32_t>(std::min<std::size_t>(64 * KiB, _opts.receive_buffer_bytes)),
    c.x.method == http::method::head});
  c.chunked.reset();
  c.sbeg = c.slen = 0;
  if (c.scratch.size() != _opts.receive_buffer_bytes) {
    c.scratch.resize(_opts.receive_buffer_bytes);
  }

  // Request head.
  if (c.head.size() != head_bytes) { c.head.resize(head_bytes); }
  http::request_head h{std::span<char>{c.head.data(), c.head.size()}};
  h.start(c.x.method, c.remote.target).host(c.remote.authority);
  for (auto const& [name, value] : c.x.headers) {
    if (name.size() == 4 && (name[0] == 'H' || name[0] == 'h') &&
        (name == "Host" || name == "host" || name == "HOST")) {
      continue;
    }
    h.field(name, value);
  }
  if (c.x.range) {
    h.range(c.x.range->first, c.x.range->second);
  } else if (c.x.suffix_bytes > 0) {
    h.field("Range", "bytes=-" + std::to_string(c.x.suffix_bytes));
  }
  if (!c.x.if_match.empty()) { h.field("If-Match", c.x.if_match); }
  if (c.x.has_body) {
    h.content_length(c.x.body_length);
    if (c.x.expect_continue) { h.expect_100_continue(); }
  }
  auto const text = h.finish();
  if (h.overflow() || text.empty()) {
    finish(index, make_error_code(errc::invalid_argument));
    return;
  }
  c.tx.clear();
  c.tx_idx = 0;
  c.tx.push_back(iovec{const_cast<char*>(text.data()), text.size()});
  c.generator_done = !c.x.body_next;
  c.body_held      = c.x.has_body && c.x.expect_continue;
  if (c.x.has_body && !c.body_held) { c.tx.insert(c.tx.end(), c.x.body.begin(), c.x.body.end()); }
  send(index);
}

void uring_transport::send(std::size_t index)
{
  auto& c = *_slots[index];
  if (c.kind == conn_kind::ktls && c.reply.owed) {
    // RFC 8446 4.6.3: the owed KeyUpdate reply precedes anything else this side sends.
    switch (c.reply.flush(c.fd, c.rekey, &tls::ktls::flush_reply)) {
      case ktls_reply::step::done: break;
      case ktls_reply::step::wait_writable: {
        auto* entry = sqe(index + 1);
        io_uring_prep_poll_add(entry, c.fd, POLLOUT);
        c.cur     = op::writable;
        c.pending = true;
        return;
      }
      case ktls_reply::step::failed:
        finish(index, make_error_code(errc::tls_protocol_error));
        return;
    }
  }
  // Skip exhausted iovecs; refill from the body generator once everything was sent.
  while (c.tx_idx < c.tx.size() && c.tx[c.tx_idx].iov_len == 0) {
    ++c.tx_idx;
  }
  if (c.tx_idx == c.tx.size() && !c.body_held && c.x.has_body && !c.generator_done) {
    c.tx.clear();
    c.tx_idx        = 0;
    auto const next = c.x.body_next();
    if (next.empty()) {
      c.generator_done = true;
    } else {
      c.tx.assign(next.begin(), next.end());
    }
    send(index);
    return;
  }
  if (c.tx_idx == c.tx.size()) {
    // Everything owed is on the wire: the response (or 100 Continue) is next.
    if (!c.body_held) {
      c.res.request_sent = true;
    } else if (c.x.continue_timeout.count() > 0 && !c.continue_deadline) {
      c.continue_deadline = clock_type::now() + c.x.continue_timeout;
    }
    c.ph = phase::head;
    receive(index);
    return;
  }
  auto const window = std::span<iovec const>{c.tx.data() + c.tx_idx,
                                             std::min(c.tx.size() - c.tx_idx, max_send_iovecs)};
  auto* entry       = sqe(index + 1);
  if (c.kind == conn_kind::user) {
    c.stream->prep_send(entry, window);
  } else if (window.size() == 1) {
    io_uring_prep_send(entry, c.fd, window[0].iov_base, window[0].iov_len, send_msg_flags);
  } else {
    c.txmsg            = msghdr{};
    c.txmsg.msg_iov    = const_cast<iovec*>(window.data());
    c.txmsg.msg_iovlen = window.size();
    io_uring_prep_sendmsg(entry, c.fd, &c.txmsg, static_cast<unsigned>(send_msg_flags));
  }
  c.cur     = op::send;
  c.pending = true;
  ++_ctr.sends;
}

void uring_transport::on_sent(std::size_t index, std::size_t bytes)
{
  auto& c = *_slots[index];
  _ctr.bytes_sent += bytes;
  c.bytes_carried += bytes;
  c.progress += bytes;  // the stall detector counts both directions
  while (bytes > 0 && c.tx_idx < c.tx.size()) {
    auto& v      = c.tx[c.tx_idx];
    auto const n = std::min(bytes, v.iov_len);
    v.iov_base   = static_cast<char*>(v.iov_base) + n;
    v.iov_len -= n;
    bytes -= n;
    if (v.iov_len == 0) { ++c.tx_idx; }
  }
  send(index);
}

void uring_transport::continue_body(std::size_t index)
{
  // 100 Continue arrived: the body goes out, then the final head is received.
  auto& c     = *_slots[index];
  c.body_held = false;
  c.continue_deadline.reset();
  c.continue_ready = false;
  c.tx.clear();
  c.tx_idx = 0;
  c.tx.insert(c.tx.end(), c.x.body.begin(), c.x.body.end());
  c.ph = phase::sending;
  send(index);
}

void uring_transport::receive(std::size_t index)
{
  auto& c = *_slots[index];
  if (c.body_held && c.continue_deadline && c.ph == phase::head && c.slen == c.sbeg) {
    // Waiting for `100 Continue` with a deadline: wait for readability instead of receiving, so
    // the deadline can cancel the wait without disturbing a TLS stream's receive state.
    if (clock_type::now() >= *c.continue_deadline) {
      continue_body(index);
      return;
    }
    // (A reused user-space stream holds no read-ahead: bytes past a response retire it.)
    if (!std::exchange(c.continue_ready, false)) {
      auto* entry = sqe(index + 1);
      io_uring_prep_poll_add(entry, c.fd, POLLIN);
      c.cur     = op::continue_poll;
      c.pending = true;
      return;
    }
  }
  c.rx_iov.clear();
  bool const direct =
    c.ph == phase::body && c.bmode == body_mode::sink && c.parser.body_remaining() > 0;
  if (direct) {
    auto const cap = std::min<std::uint64_t>(c.parser.body_remaining(), _opts.receive_buffer_bytes);
    c.rx_iov.resize(_opts.max_recv_iovecs);
    auto const w = c.sink.window(c.rx_iov, cap);
    c.rx_iov.resize(w.size());
  }
  c.direct = !c.rx_iov.empty();
  if (!c.direct) {
    // Scratch receive (head, buffered bodies): keep the unconsumed tail (a partial head line or
    // chunk line) at the front.
    if (c.sbeg == c.slen) {
      c.sbeg = c.slen = 0;
    } else if (c.sbeg > 0) {
      std::memmove(c.scratch.data(), c.scratch.data() + c.sbeg, c.slen - c.sbeg);
      c.slen -= c.sbeg;
      c.sbeg = 0;
    }
    c.rx_iov.push_back(iovec{c.scratch.data() + c.slen, c.scratch.size() - c.slen});
  }
  auto* entry = sqe(index + 1);
  c.cur       = op::recv;
  c.pending   = true;
  switch (c.kind) {
    case conn_kind::user:
      if (direct) {
        c.stream->prep_recv(entry, c.rx_iov);
      } else {
        c.stream->prep_recv_header(
          entry,
          std::span<char>{static_cast<char*>(c.rx_iov[0].iov_base), c.rx_iov[0].iov_len},
          c.ph == phase::head && c.slen == 0);
      }
      return;
    case conn_kind::ktls:
    case conn_kind::plain:
    case conn_kind::none: break;
  }
  int const flags = direct && _opts.recv_waitall ? MSG_WAITALL : 0;
  if (c.rx_iov.size() == 1) {
    io_uring_prep_recv(entry, c.fd, c.rx_iov[0].iov_base, c.rx_iov[0].iov_len, flags);
  } else {
    c.rxmsg            = msghdr{};
    c.rxmsg.msg_iov    = c.rx_iov.data();
    c.rxmsg.msg_iovlen = c.rx_iov.size();
    io_uring_prep_recvmsg(entry, c.fd, &c.rxmsg, static_cast<unsigned>(flags));
  }
  // Software kTLS decrypts in the receiving task: IOSQE_ASYNC moves that work to io-wq.
  if (c.kind == conn_kind::ktls && _opts.ktls_recv_async) { entry->flags |= IOSQE_ASYNC; }
}

void uring_transport::advance(std::size_t index, int res, unsigned flags)
{
  auto& c = *_slots[index];
  switch (c.cur) {
    case op::connect:
      if (res < 0) {
        ++_ctr.connect_failures;
        if (!c.literal && _services.dns != nullptr) {
          _services.dns->bench(c.dns_handle, c.addr, std::chrono::seconds{30}, mono_ns());
        }
        close(index);
        ++c.connect_tries;
        std::size_t const tries =
          c.literal || _services.dns == nullptr
            ? 1
            : std::max<std::size_t>(1, _services.dns->address_count(c.dns_handle));
        if (c.connect_tries < tries) {
          start_connect(index);
        } else {
          finish(index, make_error_code(errc::connect_failed));
        }
        return;
      }
      on_connected(index);
      return;
    case op::handshake:
      if (res < 0) {
        finish(index, sys_error(-res));
        return;
      }
      handshake(index);
      return;
    case op::send: {
      int n = res;
      if (c.kind == conn_kind::user) { n = c.stream->on_send(res); }
      if (n == -EAGAIN || n == -EINTR) {
        send(index);
        return;
      }
      if (n <= 0) {
        finish(index, n == 0 ? make_error_code(errc::io_error) : sys_error(-n));
        return;
      }
      on_sent(index, static_cast<std::size_t>(n));
      return;
    }
    case op::recv:
      (void)flags;
      on_receive(index, res);
      return;
    case op::drain: on_drain(index, res); return;
    case op::writable:
      if (res < 0) {
        finish(index, sys_error(-res));
        return;
      }
      send(index);
      return;
    case op::continue_poll:
      if (res == -ECANCELED) {
        continue_body(index);  // the continue deadline passed without an answer
        return;
      }
      if (res < 0) {
        finish(index, sys_error(-res));
        return;
      }
      c.continue_ready = true;  // something arrived: receive it
      receive(index);
      return;
    case op::none: break;
  }
  finish(index, make_error_code(errc::io_error));
}

void uring_transport::on_receive(std::size_t index, int res)
{
  auto& c = *_slots[index];
  ++_ctr.recvs;
  if (c.kind == conn_kind::user) {
    auto const r = c.stream->on_recv(res, 0, c.rx_iov);
    switch (r.kind) {
      case rx_kind::data: on_payload(index, r.bytes); return;
      case rx_kind::again: receive(index); return;
      case rx_kind::eof: on_eof(index); return;
      case rx_kind::control:
      case rx_kind::error: on_recv_error(index, r.err != 0 ? r.err : EPROTO); return;
    }
    return;
  }
  if (res > 0) {
    on_payload(index, static_cast<std::size_t>(res));
    return;
  }
  if (res == 0) {
    on_eof(index);
    return;
  }
  if (res == -EAGAIN || res == -EINTR) {
    receive(index);
    return;
  }
  if (res == -EIO && c.kind == conn_kind::ktls) {
    // A control record is next in the stream: drain it with a TLS_GET_RECORD_TYPE cmsg.
    if (c.drain_buf.empty()) { c.drain_buf.resize(drain_bytes); }
    c.rx_iov.assign(1, iovec{c.drain_buf.data(), c.drain_buf.size()});
    c.cmsg                 = ktls_cmsg_buffer{};
    c.rxmsg                = msghdr{};
    c.rxmsg.msg_iov        = c.rx_iov.data();
    c.rxmsg.msg_iovlen     = 1;
    c.rxmsg.msg_control    = c.cmsg.bytes;
    c.rxmsg.msg_controllen = sizeof(c.cmsg.bytes);
    auto* entry            = sqe(index + 1);
    io_uring_prep_recvmsg(entry, c.fd, &c.rxmsg, 0);
    c.cur     = op::drain;
    c.pending = true;
    return;
  }
  on_recv_error(index, -res);
}

void uring_transport::on_drain(std::size_t index, int res)
{
  auto& c        = *_slots[index];
  auto const got = interpret_ktls_recv(res, c.rxmsg);
  if (got.kind == rx_kind::again) {
    receive(index);
    return;
  }
  if (got.kind != rx_kind::control || (c.rxmsg.msg_flags & MSG_EOR) == 0) {
    on_recv_error(index, got.kind == rx_kind::error ? got.err : EPROTO);
    return;
  }
  ++_ctr.control_records;
  if (++c.control_run > max_control_run) {
    on_recv_error(index, EPROTO);
    return;
  }
  std::error_code ec;
  auto const action =
    tls::ktls::on_control_record(c.fd, c.rekey, got.type, {c.drain_buf.data(), got.bytes}, ec);
  switch (action) {
    case tls::ktls::control_action::consumed: receive(index); return;
    case tls::ktls::control_action::reply_owed:
      // Owed before the next send; a full socket buffer is not an error (send() waits for
      // POLLOUT and flushes first), so receiving simply continues.
      if (c.reply.flush(c.fd, c.rekey, &tls::ktls::flush_reply) == ktls_reply::step::failed) {
        on_recv_error(index, EPROTO);
      } else {
        receive(index);
      }
      return;
    case tls::ktls::control_action::closed: on_eof(index); return;
    case tls::ktls::control_action::failed: on_recv_error(index, EPROTO); return;
  }
}

void uring_transport::on_recv_error(std::size_t index, int err)
{
  auto& c    = *_slots[index];
  c.no_reuse = true;
  if (c.ph == phase::body && c.bmode == body_mode::error_body) {
    // The status line already says what happened; how the error body ended does not matter.
    finish(index, {});
    return;
  }
  if (err == EBADMSG || err == EPROTO || err == EKEYEXPIRED) {
    finish(index,
           make_error_code(c.kind == conn_kind::user && err == EBADMSG ? errc::tls_bad_record_mac
                                                                       : errc::tls_protocol_error));
    return;
  }
  finish(index, sys_error(err));
}

void uring_transport::on_eof(std::size_t index)
{
  auto& c    = *_slots[index];
  c.no_reuse = true;
  if (c.ph == phase::body && c.bmode != body_mode::none &&
      c.meta.body == http::framing::until_eof) {
    c.body_done = true;
    if (c.bmode != body_mode::error_body) { c.res.body_complete = true; }
    finish(index, {});
    return;
  }
  if (c.ph == phase::body) {
    // A cut error body still carries its status; a cut accepted body is a short read.
    finish(index,
           c.bmode == body_mode::error_body ? std::error_code{}
                                            : make_error_code(errc::http_length_mismatch));
    return;
  }
  finish(index, make_error_code(errc::io_error));
}

void uring_transport::on_payload(std::size_t index, std::size_t bytes)
{
  auto& c       = *_slots[index];
  c.control_run = 0;
  c.progress += bytes;
  c.bytes_carried += bytes;
  c.res.response_started = true;
  if (c.ph == phase::head) {
    _ctr.recv_scratch_bytes += bytes;
    c.slen += bytes;
    parse_head(index);
    return;
  }
  if (c.direct) {
    _ctr.recv_direct_bytes += bytes;
    c.sink.commit(bytes);
    c.parser.consume_direct(bytes);
    c.res.body_bytes += bytes;
    if (c.parser.done()) {
      c.body_done         = true;
      c.res.body_complete = true;
      finish(index, {});
      return;
    }
    receive(index);
    return;
  }
  _ctr.recv_scratch_bytes += bytes;
  c.slen += bytes;
  deliver_body(index);
}

void uring_transport::parse_head(std::size_t index)
{
  auto& c              = *_slots[index];
  std::size_t consumed = 0;
  std::error_code ec;
  auto const st = c.parser.parse_header(
    std::span<char const>{c.scratch.data() + c.sbeg, c.slen - c.sbeg}, consumed, c.meta, ec);
  c.sbeg += consumed;
  if (st == http::parse_status::error) {
    c.no_reuse = true;
    finish(index, ec ? ec : make_error_code(errc::http_bad_message));
    return;
  }
  if (st == http::parse_status::need_more) {
    if (c.body_held && c.parser.interim_responses() > 0) {
      continue_body(index);
      return;
    }
    if (c.sbeg == 0 && c.slen == c.scratch.size()) {
      c.no_reuse = true;
      finish(index, make_error_code(errc::http_header_too_large));
      return;
    }
    receive(index);
    return;
  }
  c.head_done = true;
  if (c.body_held) {
    // A final response before 100 Continue: the body is never sent, the connection is unusable.
    c.body_held = false;
    c.no_reuse  = true;
  }
  auto const status  = c.meta.status;
  bool const accept  = c.x.accept_status == 0 ? status / 100 == 2 : status == c.x.accept_status;
  auto const framing = c.meta.body;
  if (framing == http::framing::none || c.parser.done()) {
    c.bmode     = body_mode::none;
    c.body_done = true;
    if (accept) {
      c.res.body_complete = true;
    } else {
      c.res.error_body_complete = true;
    }
    if (c.sbeg != c.slen) { c.no_reuse = true; }  // bytes past the response
    finish(index, {});
    return;
  }
  if (accept) {
    // A close-delimited body cannot be told from a cut one: an accepted body must be
    // length-framed, or chunked where the exchange allows it.
    if (framing == http::framing::until_eof ||
        (framing != http::framing::length && c.x.require_length)) {
      c.no_reuse = true;
      finish(index, make_error_code(errc::http_unsupported_framing));
      return;
    }
    c.bmode = c.x.target == body_target::sink     ? body_mode::sink
              : c.x.target == body_target::buffer ? body_mode::buffer
                                                  : body_mode::discard;
    if (c.bmode == body_mode::sink) {
      if (framing != http::framing::length) {
        c.no_reuse = true;
        finish(index, make_error_code(errc::http_unsupported_framing));
        return;
      }
      c.sink = http::iovec_sink{c.x.sink};
      if (c.sink.init(c.meta.content_length)) {
        // More than the destination holds (e.g. a 200 with the whole object for a range): leave
        // the body unread; the caller classifies the head.
        c.no_reuse = true;
        finish(index, {});
        return;
      }
    }
  } else {
    if (framing != http::framing::length || c.meta.content_length.value_or(0) > max_drained_body) {
      c.no_reuse = true;  // an unbounded error body is not drained
      finish(index, {});
      return;
    }
    c.bmode = body_mode::error_body;
  }
  c.ph = phase::body;
  // Bytes after the head are body.
  deliver_body(index);
}

void uring_transport::deliver_body(std::size_t index)
{
  auto& c                 = *_slots[index];
  char const* data        = c.scratch.data() + c.sbeg;
  std::size_t const bytes = c.slen - c.sbeg;
  auto keep               = [&](char const* p, std::size_t n) {
    if (c.bmode == body_mode::sink) {
      c.sink.copy_in({p, n});
      c.res.body_bytes += n;
      return;
    }
    if (c.bmode == body_mode::discard) { return; }
    std::size_t const limit =
      c.bmode == body_mode::error_body ? max_drained_body : c.x.buffer_limit;
    auto const room = limit > c.res.body.size() ? limit - c.res.body.size() : 0;
    auto const take = std::min(room, n);
    c.res.body.insert(c.res.body.end(), p, p + take);
    if (c.bmode != body_mode::error_body) {
      c.res.body_bytes += take;
      if (take < n) { c.res.truncated = true; }
    }
  };
  std::size_t used = bytes;
  switch (c.meta.body) {
    case http::framing::length: {
      auto const take =
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes, c.parser.body_remaining()));
      keep(data, take);
      c.parser.consume_direct(take);
      if (take < bytes) { c.no_reuse = true; }  // bytes past the response
      if (c.parser.done()) { c.body_done = true; }
      used = bytes;
      break;
    }
    case http::framing::chunked: {
      std::error_code ec;
      auto const st = c.chunked.feed(
        std::span<char const>{data, bytes},
        used,
        [&](std::span<char const> d) { keep(d.data(), d.size()); },
        ec);
      if (st == http::parse_status::error) {
        c.no_reuse = true;
        finish(index, ec ? ec : make_error_code(errc::http_bad_message));
        return;
      }
      if (st == http::parse_status::ok) {
        c.body_done = true;
        if (used < bytes) { c.no_reuse = true; }  // bytes past the response
        used = bytes;
      }
      break;
    }
    case http::framing::until_eof: keep(data, bytes); break;
    case http::framing::none: c.body_done = true; break;
  }
  c.sbeg += used;
  if (c.res.truncated) {
    // A buffered body larger than the caller wants: stop reading it.
    c.no_reuse = true;
    finish(index, {});
    return;
  }
  if (c.body_done) {
    if (c.bmode != body_mode::error_body) {
      c.res.body_complete = true;
    } else {
      c.res.error_body_complete = true;
    }
    finish(index, {});
    return;
  }
  receive(index);
}

bool uring_transport::reusable(std::size_t index)
{
  auto& c = *_slots[index];
  if (!c.established || c.no_reuse || !c.head_done || !c.body_done || !c.meta.keep_alive) {
    return false;
  }
  if (_opts.max_connection_bytes != 0 && c.bytes_carried >= _opts.max_connection_bytes) {
    return false;
  }
  if (c.kind == conn_kind::user) {
    return c.stream != nullptr && (c.stream->clean_for_reuse() || c.stream->drain_for_reuse());
  }
  return true;
}

void uring_transport::finish(std::size_t index, std::error_code ec)
{
  auto& c = *_slots[index];
  if (c.pending || c.cancel_pending) {
    throw std::logic_error("rest transport: completing pending I/O");
  }
  bool const socket_failure =
    ec == make_error_code(errc::io_error) || ec.category() == std::system_category();
  if (socket_failure && c.reused && !c.res.response_started) {
    // The pooled connection died before any response byte (curl's reuse retry).
    ++_ctr.stale_failures;
    ec = make_error_code(errc::stale_connection);
  }
  if (ec) { ++_ctr.errors; }
  if (ec || !reusable(index)) { close(index); }
  c.idle_since = clock_type::now();
  c.connect_deadline.reset();
  c.exchange_deadline.reset();
  c.continue_deadline.reset();
  c.continue_cancel = false;
  c.continue_ready  = false;
  c.stall_on        = false;
  c.canceling       = false;
  c.timeout_ec.clear();
  c.ph  = phase::idle;
  c.cur = op::none;
  if (c.has_request) {
    if (c.bmode == body_mode::error_body && !c.res.body.empty()) {
      // The S3 <Code> of a drained error body (NoSuchUpload, NoSuchKey, SlowDown, ...): the
      // status alone does not tell a caller which 404 or 503 it got.
      (void)http::extract_error_code(std::string_view{c.res.body.data(), c.res.body.size()},
                                     c.meta);
    }
    c.res.ec     = ec;
    c.res.meta   = c.meta;
    c.res.reused = c.reused;
    _completed.push_back(completion{index, std::move(c.res)});
  }
  c.has_request = false;
  c.res         = exchange_result{};
  c.x           = exchange{};
  c.tx.clear();
  c.rx_iov.clear();
  c.sink = http::iovec_sink{};
}

}  // namespace cucascade::io::net
