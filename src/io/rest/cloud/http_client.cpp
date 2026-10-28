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

// Minimal blocking HTTP/1.1 client for IMDSv2 (ported from gpuio's src/auth/http_client.cpp,
// plus `env` from src/auth/util.cpp). Cold path, helper thread only. One request per connection
// (`Connection: close`), non-blocking connect + poll() deadline, Content-Length, chunked or
// read-until-close framing, 1xx skipped, 16 KiB head / 64 KiB body caps.

#include "io/rest/cloud/http_client.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstdlib>
#include <memory>
#include <utility>

namespace cucascade::io::rest::cloud {
namespace {

using steady = std::chrono::steady_clock;

constexpr std::size_t max_head_bytes = std::size_t{16} * 1024;
constexpr std::size_t max_body_bytes = std::size_t{64} * 1024;

class fd_guard {
 public:
  explicit fd_guard(int fd = -1) noexcept : _fd(fd) {}
  fd_guard(fd_guard const&)            = delete;
  fd_guard& operator=(fd_guard const&) = delete;
  fd_guard(fd_guard&&)                 = delete;
  fd_guard& operator=(fd_guard&&)      = delete;
  ~fd_guard()
  {
    if (_fd >= 0) { ::close(_fd); }
  }
  [[nodiscard]] int get() const noexcept { return _fd; }
  int release() noexcept { return std::exchange(_fd, -1); }

 private:
  int _fd;
};

[[nodiscard]] int remaining_ms(steady::time_point deadline) noexcept
{
  auto const ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - steady::now()).count();
  return ms <= 0 ? 0 : static_cast<int>(std::min<long long>(ms, INT_MAX));
}

// @return > 0 ready, 0 timed out, < 0 error.
[[nodiscard]] int wait_fd(int fd, short events, steady::time_point deadline) noexcept
{
  for (;;) {
    pollfd p{fd, events, 0};
    int const r = ::poll(&p, 1, remaining_ms(deadline));
    if (r < 0 && errno == EINTR) { continue; }
    return r;
  }
}

[[nodiscard]] result<int> connect_to(std::string const& host,
                                     std::uint16_t port,
                                     steady::time_point deadline) noexcept
{
  addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags    = AI_NUMERICSERV;
  std::array<char, 8> port_str{};
  std::to_chars(port_str.data(), port_str.data() + port_str.size() - 1, port);
  addrinfo* raw = nullptr;
  if (::getaddrinfo(host.c_str(), port_str.data(), &hints, &raw) != 0 || raw == nullptr) {
    return errc::dns_failed;
  }
  std::unique_ptr<addrinfo, void (*)(addrinfo*)> list{raw, &::freeaddrinfo};
  bool timed_out = false;
  for (addrinfo const* ai = list.get(); ai != nullptr; ai = ai->ai_next) {
    fd_guard fd{
      ::socket(ai->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, ai->ai_protocol)};
    if (fd.get() < 0) { continue; }
    int r = ::connect(fd.get(), ai->ai_addr, ai->ai_addrlen);
    if (r != 0 && errno == EINPROGRESS) {
      int const w = wait_fd(fd.get(), POLLOUT, deadline);
      if (w == 0) {
        timed_out = true;
        break;
      }
      int so_error  = 0;
      socklen_t len = sizeof so_error;
      r = (w > 0 && ::getsockopt(fd.get(), SOL_SOCKET, SO_ERROR, &so_error, &len) == 0 &&
           so_error == 0)
            ? 0
            : -1;
    }
    if (r == 0) {
      int one = 1;
      static_cast<void>(::setsockopt(fd.get(), IPPROTO_TCP, TCP_NODELAY, &one, sizeof one));
      return fd.release();
    }
  }
  return timed_out ? errc::connect_timeout : errc::connect_failed;
}

[[nodiscard]] std::error_code send_all(int fd,
                                       std::string_view data,
                                       steady::time_point deadline) noexcept
{
  while (!data.empty()) {
    auto const n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
    if (n > 0) {
      data.remove_prefix(static_cast<std::size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) { continue; }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      int const w = wait_fd(fd, POLLOUT, deadline);
      if (w == 0) { return make_error_code(errc::request_timeout); }
      if (w < 0) { return make_error_code(errc::io_error); }
      continue;
    }
    return make_error_code(errc::io_error);
  }
  return {};
}

[[nodiscard]] char lower(char c) noexcept
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] bool iequals(std::string_view a, std::string_view b) noexcept
{
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return lower(x) == lower(y);
         });
}

[[nodiscard]] std::string_view trim(std::string_view s) noexcept
{
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
    s.remove_suffix(1);
  }
  return s;
}

struct head_info {
  unsigned status{0};
  std::size_t body_offset{0};
  std::optional<std::uint64_t> content_length;
  bool chunked{false};
};

// Parse one response head starting at `buf[0]`; nullopt when incomplete, status 0 when malformed.
[[nodiscard]] std::optional<head_info> parse_head(std::string_view buf)
{
  auto const end = buf.find("\r\n\r\n");
  if (end == std::string_view::npos) { return std::nullopt; }
  head_info h;
  h.body_offset = end + 4;
  auto lines    = buf.substr(0, end);
  auto const nl = lines.find("\r\n");
  auto status   = lines.substr(0, nl);
  lines         = nl == std::string_view::npos ? std::string_view{} : lines.substr(nl + 2);
  if (status.size() < 12 || !status.starts_with("HTTP/1.") || status[8] != ' ') { return h; }
  unsigned code = 0;
  auto const r  = std::from_chars(status.data() + 9, status.data() + 12, code);
  if (r.ec != std::errc{} || r.ptr != status.data() + 12 || code < 100 || code > 599) { return h; }
  while (!lines.empty()) {
    auto const e     = lines.find("\r\n");
    auto const line  = lines.substr(0, e);
    lines            = e == std::string_view::npos ? std::string_view{} : lines.substr(e + 2);
    auto const colon = line.find(':');
    if (colon == std::string_view::npos) { return h; }  // status 0: malformed
    auto const name  = line.substr(0, colon);
    auto const value = trim(line.substr(colon + 1));
    if (iequals(name, "content-length")) {
      std::uint64_t v  = 0;
      auto const* last = value.data() + value.size();
      auto const cr    = std::from_chars(value.data(), value.data() + value.size(), v);
      if (cr.ec != std::errc{} || cr.ptr != last) { return h; }
      if (h.content_length && *h.content_length != v) { return h; }
      h.content_length = v;
    } else if (iequals(name, "transfer-encoding")) {
      std::string lowered{value};
      std::transform(lowered.begin(), lowered.end(), lowered.begin(), lower);
      h.chunked = lowered.find("chunked") != std::string::npos;
    }
  }
  if (h.chunked) {
    h.content_length.reset();
  }  // RFC 9112 §6.3: Transfer-Encoding overrides Content-Length
  h.status = code;
  return h;
}

// Decode a complete chunked body; false when malformed or incomplete.
[[nodiscard]] bool dechunk(std::string_view in, std::string& out)
{
  out.clear();
  for (;;) {
    auto const e = in.find("\r\n");
    if (e == std::string_view::npos) { return false; }
    auto size_str = in.substr(0, e);
    if (auto semi = size_str.find(';'); semi != std::string_view::npos) {
      size_str = size_str.substr(0, semi);
    }
    size_str      = trim(size_str);
    std::size_t n = 0;
    auto const r  = std::from_chars(size_str.data(), size_str.data() + size_str.size(), n, 16);
    if (r.ec != std::errc{} || r.ptr != size_str.data() + size_str.size()) { return false; }
    in.remove_prefix(e + 2);
    if (n == 0) { return true; }  // trailers ignored
    if (in.size() < n + 2 || out.size() + n > max_body_bytes) { return false; }
    out.append(in.substr(0, n));
    in.remove_prefix(n + 2);
  }
}

}  // namespace

std::optional<std::string> env(char const* name)
{
  char const* v = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): helper thread, cold path
  if (v == nullptr || *v == '\0') { return std::nullopt; }
  return std::string{v};
}

bool env_true(char const* name)
{
  auto const v = env(name);
  if (!v) { return false; }
  std::string_view s{*v};
  while (!s.empty() &&
         (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
    s.remove_prefix(1);
  }
  while (!s.empty() &&
         (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
    s.remove_suffix(1);
  }
  return iequals(s, "true");
}

result<http_reply> http_exchange(std::string const& host,
                                 std::uint16_t port,
                                 std::string_view request,
                                 std::chrono::milliseconds timeout) noexcept
{
  try {
    auto const deadline = steady::now() + timeout;
    auto c              = connect_to(host, port, deadline);
    if (!c) { return c.error(); }
    fd_guard fd{*c};
    if (auto ec = send_all(fd.get(), request, deadline)) { return ec; }

    std::string buf;
    std::optional<head_info> head;
    bool eof = false;
    for (;;) {
      // Skip interim 1xx heads (RFC 9110 §15.2).
      while (!head) {
        auto h = parse_head(buf);
        if (!h) { break; }
        if (h->status == 0) { return errc::http_bad_message; }
        if (h->status >= 200) {
          head = h;
          break;
        }
        buf.erase(0, h->body_offset);
      }
      if (!head && buf.size() > max_head_bytes) { return errc::http_header_too_large; }
      if (head) {
        auto const have = buf.size() - head->body_offset;
        if (head->content_length && have >= *head->content_length) { break; }
        if (have > max_body_bytes + 64) { return errc::http_bad_message; }
      }
      if (eof) { break; }
      int const w = wait_fd(fd.get(), POLLIN, deadline);
      if (w == 0) { return errc::request_timeout; }
      if (w < 0) { return errc::io_error; }
      std::array<char, 4096> tmp{};
      auto const n = ::recv(fd.get(), tmp.data(), tmp.size(), 0);
      if (n < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) { continue; }
        return errc::io_error;
      }
      if (n == 0) {
        eof = true;
        continue;
      }
      buf.append(tmp.data(), static_cast<std::size_t>(n));
    }
    if (!head) { return errc::http_bad_message; }
    http_reply reply;
    reply.status = head->status;
    auto body    = std::string_view{buf}.substr(head->body_offset);
    if (head->chunked) {
      if (!dechunk(body, reply.body)) { return errc::http_bad_message; }
    } else if (head->content_length) {
      if (body.size() < *head->content_length) { return errc::http_length_mismatch; }
      reply.body.assign(body.substr(0, static_cast<std::size_t>(*head->content_length)));
    } else {
      reply.body.assign(body);
    }
    if (reply.body.size() > max_body_bytes) { return errc::http_bad_message; }
    return reply;
  } catch (...) {
    return std::make_error_code(std::errc::not_enough_memory);
  }
}

}  // namespace cucascade::io::rest::cloud
