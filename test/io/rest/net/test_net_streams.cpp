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

/// @file test_streams.cpp
/// @brief Stream layer (streams.hpp, idle_watch.hpp): concepts, sizes, the exact SQEs each
/// operation prepares (opcode, MSG_WAITALL, MSG_NOSIGNAL, POLL_FIRST, never IOSQE_IO_LINK), CQE
/// classification, control-drain parsing; plain_stream end to end over loopback TCP.

#include "io/rest/net/conn_slot.hpp"
#include "io/rest/net/connection_pool.hpp"
#include "io/rest/net/idle_watch.hpp"
#include "io/rest/net/streams.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "net_test_support.hpp"

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <vector>

using namespace cucascade::io::net;

namespace {

io_uring_sqe blank_sqe()
{
  io_uring_sqe s;
  std::memset(&s, 0xA5, sizeof s);  // garbage, then what io_uring_get_sqe guarantees (2.14)
  io_uring_initialize_sqe(&s);
  return s;
}

}  // namespace

TEST_CASE("net: stream concepts, kinds and sizes", "[rest][net][streams]")
{
  STATIC_REQUIRE(completion_stream<plain_stream>);
  STATIC_REQUIRE(completion_stream<ktls_stream>);
  STATIC_REQUIRE(pool_stream<ktls_stream>);
  STATIC_REQUIRE(control_drain_stream<ktls_stream>);
  STATIC_REQUIRE_FALSE(control_drain_stream<plain_stream>);
  STATIC_REQUIRE(sizeof(ktls_stream) <= 64);
  STATIC_REQUIRE(kind_plain == 0);
  STATIC_REQUIRE(kind_ktls == 1);
  STATIC_REQUIRE(kind_tls == 2);
  STATIC_REQUIRE(kind_connecting == 3);
  STATIC_REQUIRE(kind_handshaking == 4);
  STATIC_REQUIRE((body_recv_sqe_flags(false) & IOSQE_IO_LINK) == 0);
  STATIC_REQUIRE((body_recv_sqe_flags(true) & (IOSQE_IO_LINK | IOSQE_IO_HARDLINK)) == 0);
  CHECK(sizeof(ktls_stream) <= 64);
  CHECK(sizeof(conn_slot) < 32 * 1024);
}

TEST_CASE("net: ktls_stream prepares RECV|WAITALL / RECVMSG|WAITALL bodies, never linked",
          "[rest][net][streams]")
{
  stream_msg msg{};
  std::array<char, 4096> buf{};
  ktls_stream s(7, &msg, stream_options{true, false});

  SECTION("contiguous body: IORING_OP_RECV with MSG_WAITALL, no flags")
  {
    io_uring_sqe sqe = blank_sqe();
    iovec v{buf.data(), 1000};
    s.prep_recv(&sqe, std::span<iovec const>(&v, 1));
    CHECK(sqe.opcode == IORING_OP_RECV);
    CHECK(sqe.fd == 7);
    CHECK(sqe.len == 1000);
    CHECK(sqe.addr == reinterpret_cast<std::uint64_t>(buf.data()));
    CHECK((sqe.msg_flags & MSG_WAITALL) != 0);
    CHECK(sqe.flags == 0);  // rule 2: no IOSQE_IO_LINK, no ASYNC (inline placement)
  }
  SECTION("scattered body: IORING_OP_RECVMSG with iovecs, WAITALL, NO msg_control")
  {
    io_uring_sqe sqe = blank_sqe();
    iovec v[2]{{buf.data(), 100}, {buf.data() + 200, 300}};
    s.prep_recv(&sqe, std::span<iovec const>(v, 2));
    CHECK(sqe.opcode == IORING_OP_RECVMSG);
    CHECK((sqe.msg_flags & MSG_WAITALL) != 0);
    CHECK(sqe.addr == reinterpret_cast<std::uint64_t>(&msg.rx));
    CHECK(msg.rx.msg_iov == v);
    CHECK(msg.rx.msg_iovlen == 2);
    CHECK(msg.rx.msg_control == nullptr);  // with a cmsg WAITALL is ignored (net.c:1064)
    CHECK(msg.rx.msg_controllen == 0);
    CHECK(sqe.flags == 0);
  }
  SECTION("iowq placement: IOSQE_ASYNC only")
  {
    ktls_stream a(7, &msg, stream_options{true, true});
    io_uring_sqe sqe = blank_sqe();
    iovec v{buf.data(), 64};
    a.prep_recv(&sqe, std::span<iovec const>(&v, 1));
    CHECK(sqe.flags == IOSQE_ASYNC);
  }
  SECTION("recv_waitall = false: one-shot receives (experiments)")
  {
    ktls_stream o(7, &msg, stream_options{false, false});
    io_uring_sqe sqe = blank_sqe();
    iovec v{buf.data(), 64};
    o.prep_recv(&sqe, std::span<iovec const>(&v, 1));
    CHECK((sqe.msg_flags & MSG_WAITALL) == 0);
  }
  SECTION("header receive: one-shot RECV, no WAITALL, optional POLL_FIRST")
  {
    io_uring_sqe sqe = blank_sqe();
    s.prep_recv_header(&sqe, std::span<char>(buf.data(), 512), true);
    CHECK(sqe.opcode == IORING_OP_RECV);
    CHECK((sqe.msg_flags & MSG_WAITALL) == 0);
    CHECK((sqe.ioprio & IORING_RECVSEND_POLL_FIRST) != 0);
    io_uring_sqe sqe2 = blank_sqe();
    s.prep_recv_header(&sqe2, std::span<char>(buf.data(), 512), false);
    CHECK((sqe2.ioprio & IORING_RECVSEND_POLL_FIRST) == 0);
  }
  SECTION("send: SEND / SENDMSG with MSG_NOSIGNAL")
  {
    io_uring_sqe sqe = blank_sqe();
    iovec v{buf.data(), 10};
    s.prep_send(&sqe, std::span<iovec const>(&v, 1));
    CHECK(sqe.opcode == IORING_OP_SEND);
    CHECK((sqe.msg_flags & MSG_NOSIGNAL) != 0);
    io_uring_sqe sqe2 = blank_sqe();
    iovec w[3]{{buf.data(), 10}, {buf.data() + 10, 10}, {buf.data() + 20, 10}};
    s.prep_send(&sqe2, std::span<iovec const>(w, 3));
    CHECK(sqe2.opcode == IORING_OP_SENDMSG);
    CHECK((sqe2.msg_flags & MSG_NOSIGNAL) != 0);
    CHECK(msg.tx.msg_iovlen == 3);
  }
  SECTION("control drain: RECVMSG with the TLS_GET_RECORD_TYPE cmsg buffer, no WAITALL")
  {
    io_uring_sqe sqe = blank_sqe();
    s.prep_drain(&sqe, std::span<char>(buf.data(), buf.size()));
    CHECK(sqe.opcode == IORING_OP_RECVMSG);
    CHECK((sqe.msg_flags & MSG_WAITALL) == 0);
    CHECK(msg.rx.msg_control == msg.cmsg);
    CHECK(msg.rx.msg_controllen == sizeof msg.cmsg);
    CHECK(msg.drain_iov.iov_len == buf.size());
  }
}

TEST_CASE("net: receive CQE classification (kTLS -EIO is a control record)", "[rest][net][streams]")
{
  stream_msg msg{};
  ktls_stream k(3, &msg, {});
  plain_stream p(3, &msg, {});
  std::span<iovec const> none{};
  CHECK(k.on_recv(4096, 0, none).kind == rx_kind::data);
  CHECK(k.on_recv(4096, 0, none).bytes == 4096);
  CHECK(k.bytes_rx() == 8192);
  CHECK(k.on_recv(0, 0, none).kind == rx_kind::eof);
  CHECK(k.on_recv(-EIO, 0, none).kind == rx_kind::control);
  CHECK(p.on_recv(-EIO, 0, none).kind == rx_kind::error);
  CHECK(p.on_recv(-EIO, 0, none).err == EIO);
  CHECK(k.on_recv(-EAGAIN, 0, none).kind == rx_kind::again);
  CHECK(k.on_recv(-ECONNRESET, 0, none).kind == rx_kind::error);
  CHECK(k.on_recv(-EKEYEXPIRED, 0, none).err == EKEYEXPIRED);
}

TEST_CASE("net: control drain parsing (record type, MSG_EOR, MSG_CTRUNC)", "[rest][net][streams]")
{
  stream_msg msg{};
  ktls_stream k(3, &msg, {});
  std::array<char, 64> buf{};
  io_uring_sqe sqe = blank_sqe();
  k.prep_drain(&sqe, std::span<char>(buf));
  // What ____sys_recvmsg writes back for a 217-byte NewSessionTicket record.
  cmsghdr* c            = CMSG_FIRSTHDR(&msg.rx);
  c->cmsg_level         = SOL_TLS;
  c->cmsg_type          = TLS_GET_RECORD_TYPE;
  c->cmsg_len           = CMSG_LEN(sizeof(unsigned char));
  *CMSG_DATA(c)         = 22;
  msg.rx.msg_controllen = CMSG_SPACE(sizeof(unsigned char));
  msg.rx.msg_flags      = MSG_EOR;
  drain_result const d  = k.on_drain(217);
  CHECK(d.bytes == 217);
  CHECK(d.type == 22);  // handshake (tls::ktls::record_handshake)
  CHECK(d.eor);
  CHECK_FALSE(d.truncated);
  msg.rx.msg_flags = 0;
  CHECK_FALSE(k.on_drain(100).eor);
  msg.rx.msg_flags = MSG_CTRUNC;
  CHECK(k.on_drain(1).truncated);
  CHECK(k.on_drain(-ECONNRESET).err == ECONNRESET);
}

TEST_CASE("net: idle watch verdicts", "[rest][net][streams]")
{
  CHECK(idle_watch_events == (POLLIN | POLLRDHUP));
  CHECK(classify_idle_watch(POLLIN) == watch_verdict::readable);
  CHECK(classify_idle_watch(POLLIN | POLLRDHUP) == watch_verdict::fin);
  CHECK(classify_idle_watch(POLLHUP) == watch_verdict::fin);
  CHECK(classify_idle_watch(POLLERR | POLLIN) == watch_verdict::fin);
  CHECK(classify_idle_watch(-ECANCELED) == watch_verdict::failed);
  io_uring_sqe sqe = blank_sqe();
  prep_idle_watch(&sqe, 9);
  CHECK(sqe.opcode == IORING_OP_POLL_ADD);
  CHECK(sqe.fd == 9);
  CHECK(sqe.flags == 0);
}

TEST_CASE("net: plain_stream send and WAITALL receive end to end on loopback TCP",
          "[rest][net][streams]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  // A connected TCP pair on loopback.
  int const l = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  sockaddr_in a{};
  a.sin_family      = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  REQUIRE(::bind(l, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
  REQUIRE(::listen(l, 4) == 0);
  socklen_t al = sizeof a;
  ::getsockname(l, reinterpret_cast<sockaddr*>(&a), &al);
  int const c = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(::connect(c, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
  int const s = ::accept4(l, nullptr, nullptr, SOCK_CLOEXEC);
  REQUIRE(s >= 0);

  net_test::test_loop tl(uring_loop_config{.sq_entries = 64});
  loop_env env = tl.env();
  struct sink_t {
    int last_res{0};
    unsigned cqes{0};
    void on_op_cqe(std::uint32_t, int res, unsigned) noexcept
    {
      last_res = res;
      ++cqes;
    }
  } sink;
  env.bind(&sink);
  stream_msg msg{};
  plain_stream ps(c, &msg, stream_options{true, false});

  std::vector<std::byte> out(1 << 20), in(1 << 20);
  net_test::fill_pattern(out, 7, 0);
  std::uint32_t tok = 0;
  REQUIRE(env.take_op(cucascade::io::rest::priority::normal, false, tok));
  // Receive first (WAITALL for the whole 1 MiB), then let the peer send it in pieces.
  iovec rv{in.data(), in.size()};
  ps.prep_recv(env.get_sqe(tok, ud_kind::op), std::span<iovec const>(&rv, 1));
  tl.run_once(0);
  for (std::size_t off = 0; off < out.size(); off += 64 * 1024)
    REQUIRE(::send(s, out.data() + off, 64 * 1024, 0) == 64 * 1024);
  REQUIRE(tl.run_until([&] { return sink.cqes == 1; }, std::chrono::seconds(5)));
  CHECK(sink.last_res == static_cast<int>(in.size()));  // ONE CQE for the whole length
  CHECK(net_test::check_pattern(in, 7, 0) == static_cast<std::size_t>(-1));
  // Send 3 iovecs back through SENDMSG.
  iovec tv[3]{{out.data(), 100}, {out.data() + 100, 1000}, {out.data() + 1100, 10}};
  ps.prep_send(env.get_sqe(tok, ud_kind::op), std::span<iovec const>(tv, 3));
  REQUIRE(tl.run_until([&] { return sink.cqes == 2; }, std::chrono::seconds(5)));
  CHECK(sink.last_res == 1110);
  std::vector<std::byte> got(1110);
  std::size_t have = 0;
  while (have < got.size()) {
    ssize_t const n = ::recv(s, got.data() + have, got.size() - have, 0);
    REQUIRE(n > 0);
    have += static_cast<std::size_t>(n);
  }
  CHECK(std::memcmp(got.data(), out.data(), got.size()) == 0);
  env.release_op(tok);
  ::close(c);
  ::close(s);
  ::close(l);
}

TEST_CASE("net: socket options: NODELAY, keepalive 60/10/3, TCP_USER_TIMEOUT, BINDTODEVICE",
          "[rest][net][streams]")
{
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  REQUIRE(fd >= 0);
  cucascade::io::net::connection_config cfg;
  bool bind_failed = false;
  REQUIRE_FALSE(apply_socket_options(fd, cfg, "lo", bind_failed));
  auto get = [&](int level, int opt) {
    int v         = -1;
    socklen_t len = sizeof v;
    ::getsockopt(fd, level, opt, &v, &len);
    return v;
  };
  CHECK(get(IPPROTO_TCP, TCP_NODELAY) == 1);
  CHECK(get(SOL_SOCKET, SO_KEEPALIVE) == 1);
  CHECK(get(IPPROTO_TCP, TCP_KEEPIDLE) == 60);
  CHECK(get(IPPROTO_TCP, TCP_KEEPINTVL) == 10);
  CHECK(get(IPPROTO_TCP, TCP_KEEPCNT) == 3);
  CHECK(get(IPPROTO_TCP, TCP_USER_TIMEOUT) == 30000);
  char dev[32]{};
  socklen_t dl = sizeof dev;
  if (!bind_failed) {
    REQUIRE(::getsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, dev, &dl) == 0);
    CHECK(std::string(dev) == "lo");
  } else {
    WARN("SO_BINDTODEVICE refused (no CAP_NET_RAW): counted as bind_failures, socket unbound");
  }
  ::close(fd);
  CHECK(map_socket_errno(ECANCELED) == cucascade::io::rest::errc::operation_canceled);
  CHECK(map_socket_errno(EKEYEXPIRED) == cucascade::io::rest::errc::tls_protocol_error);
  CHECK(map_socket_errno(ECONNRESET) == std::errc::connection_reset);
  CHECK(reserved_connections_for(0.05, 64) == 4);
  CHECK(reserved_connections_for(0.05, 8) == 2);
  CHECK(reserved_connections_for(0.05, 2) == 1);
  CHECK(reserved_connections_for(0.05, 1) == 0);
}
