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

/// @file test_net_uring_loop.cpp
/// @brief The REST ring (uring_loop.hpp): user_data encoding and the deadline heap (gpuio
/// tests/unit/reactor/test_primitives.cpp), op slots (LIFO, generations, urgent reserve,
/// background, deferred release behind a cancel, stale CQEs), get_sqe never null, deadlines that
/// bound the wait, the cross-thread wake (sync MSG_RING and the eventfd fallback) with the
/// Dekker park, and the shutdown protocol (sync cancel, reap every CQE).

#include "io/rest/net/deadline_heap.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "net_test_support.hpp"

#include <catch2/catch_all.hpp>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace cucascade::io::net;
using cucascade::io::rest::priority;
using namespace std::chrono_literals;

namespace {

/// Records the CQEs of the ops it owns.
struct recorder {
  std::vector<std::pair<std::uint32_t, int>> cqes;
  void on_op_cqe(std::uint32_t token, int res, unsigned) noexcept { cqes.emplace_back(token, res); }
};

void bind(loop_env& env, recorder& r) { env.bind(&r); }

}  // namespace

TEST_CASE("net: user_data encodes kind, generation and slot index", "[rest][net][loop]")
{
  std::uint64_t const ud = make_user_data(ud_kind::handshake_poll, 0x00ABCDEFu, 1234567u);
  CHECK(ud_kind_of(ud) == ud_kind::handshake_poll);
  CHECK(ud_generation_of(ud) == 0x00ABCDEFu);
  CHECK(ud_index_of(ud) == 1234567u);
  // generation wraps at 24 bits and never touches the kind byte
  std::uint64_t const w = make_user_data(ud_kind::op, 0xFFFF'FFFFu, 7);
  CHECK(ud_kind_of(w) == ud_kind::op);
  CHECK(ud_generation_of(w) == ud_generation_mask);
  CHECK(ud_index_of(make_user_data(ud_kind::wake, 0, 0)) == 0);
  CHECK(ud_kind_of(make_user_data(ud_kind::wake, 0, 0)) == ud_kind::wake);
}

TEST_CASE("net: deadline heap pops in deadline order, FIFO among equals", "[rest][net][loop]")
{
  deadline_heap<deadline_fn> h;
  REQUIRE(h.init(1024));
  std::mt19937_64 rng(42);
  std::multimap<std::uint64_t, std::uintptr_t> model;  // multimap keeps insertion order for ties
  std::vector<deadline_handle> handles;
  for (std::uintptr_t i = 1; i <= 1000; ++i) {
    std::uint64_t const when = rng() % 200;  // many ties
    auto const hd            = h.arm(when, nullptr, reinterpret_cast<void*>(i));
    REQUIRE(hd.valid());
    handles.push_back(hd);
    model.emplace(when, i);
  }
  // cancel a third of them
  for (std::size_t i = 0; i < handles.size(); i += 3) {
    auto const arg = static_cast<std::uintptr_t>(i + 1);
    REQUIRE(h.cancel(handles[i]));
    CHECK_FALSE(h.cancel(handles[i]));  // stale handle rejected
    for (auto it = model.begin(); it != model.end(); ++it) {
      if (it->second == arg) {
        model.erase(it);
        break;
      }
    }
  }
  CHECK(h.size() == model.size());
  deadline_heap<deadline_fn>::entry e{};
  if (model.begin()->first > 0) CHECK_FALSE(h.pop_due(model.begin()->first - 1, e));
  std::vector<std::pair<std::uint64_t, std::uintptr_t>> popped;
  while (h.pop_due(~std::uint64_t{0}, e))
    popped.emplace_back(e.when, reinterpret_cast<std::uintptr_t>(e.arg));
  std::vector<std::pair<std::uint64_t, std::uintptr_t>> expect(model.begin(), model.end());
  CHECK(popped == expect);
  CHECK(h.empty());
  CHECK(h.next() == ~std::uint64_t{0});
}

TEST_CASE("net: deadline heap capacity, next() and stale handles after reuse", "[rest][net][loop]")
{
  deadline_heap<deadline_fn> h;
  REQUIRE(h.init(4));
  auto a = h.arm(30, nullptr, nullptr);
  auto b = h.arm(10, nullptr, nullptr);
  auto c = h.arm(20, nullptr, nullptr);
  auto d = h.arm(40, nullptr, nullptr);
  CHECK(h.arm(50, nullptr, nullptr).valid() == false);  // full: invalid handle, nothing lost
  CHECK(h.next() == 10);
  deadline_heap<deadline_fn>::entry e{};
  CHECK_FALSE(h.pop_due(9, e));
  REQUIRE(h.pop_due(10, e));
  CHECK(e.when == 10);
  CHECK_FALSE(h.live(b));
  CHECK(h.cancel(d));
  auto f = h.arm(5, nullptr, nullptr);  // reuses a freed node
  CHECK(f.valid());
  CHECK_FALSE(h.cancel(b));  // old handle of a reused node is stale
  CHECK(h.next() == 5);
  CHECK(h.live(a));
  CHECK(h.live(c));
  CHECK(h.size() == 3);
}

TEST_CASE("net: loop deadlines fire in order and bound the ring wait (no timeout SQEs)",
          "[rest][net][loop]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  // A long safety tick: only the heap head can end the waits below.
  net_test::test_loop tl(uring_loop_config{.max_park_ns = 10'000'000'000ULL});
  uring_loop& l = tl.loop();
  struct fired_t {
    std::vector<int> order;
    std::vector<std::uint64_t> at;
  } fired;
  struct arg_t {
    fired_t* f;
    int id;
    uring_loop* l;
  };
  auto const fn = [](void* p) noexcept {
    auto* a = static_cast<arg_t*>(p);
    a->f->order.push_back(a->id);
    a->f->at.push_back(a->l->now_ns());
  };
  l.update_now();
  std::uint64_t const t0 = l.now_ns();
  arg_t a1{&fired, 1, &l}, a2{&fired, 2, &l}, a3{&fired, 3, &l}, a4{&fired, 4, &l};
  REQUIRE(l.arm_deadline(t0 + 60'000'000ULL, fn, &a3).valid());
  REQUIRE(l.arm_deadline(t0 + 20'000'000ULL, fn, &a1).valid());
  REQUIRE(l.arm_deadline(t0 + 40'000'000ULL, fn, &a2).valid());
  auto const h4 = l.arm_deadline(t0 + 30'000'000ULL, fn, &a4);
  REQUIRE(l.cancel_deadline(h4));
  CHECK(l.next_deadline_ns() == t0 + 20'000'000ULL);
  auto const start                = std::chrono::steady_clock::now();
  std::uint64_t const sqes_before = l.stats().ops_submitted;
  for (;;) {  // no bound of its own: each wait ends at the next deadline
    l.update_now();
    l.reap();
    l.run_due_deadlines();
    if (fired.order.size() == 3) break;
    l.submit_and_wait(uring_loop::no_bound);
  }
  auto const took = std::chrono::steady_clock::now() - start;
  CHECK(fired.order == std::vector<int>{1, 2, 3});
  CHECK(took < 2s);  // not the 10 s safety tick
  for (std::size_t i = 0; i < fired.at.size(); ++i)
    CHECK(fired.at[i] >= t0 + (i + 1) * 20'000'000ULL);
  CHECK(l.stats().ops_submitted == sqes_before);  // deadlines never become SQEs
  CHECK(l.stats().deadlines_fired == 3);
}

TEST_CASE("net: op slots: LIFO reuse, generations, urgent reserve, background, never-null SQEs",
          "[rest][net][loop]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  // SQ 8 (urgent reserve 2 = 8 / 4): 6 non-urgent ops at most.
  net_test::test_loop tl(uring_loop_config{.sq_entries = 8, .urgent_reserve_slots = 4});
  uring_loop& l = tl.loop();
  loop_env env  = tl.env();
  recorder r;
  bind(env, r);
  CHECK(env.op_capacity() == 8);
  std::uint32_t t = 0;
  REQUIRE(env.take_op(priority::normal, false, t));
  std::uint32_t const first = t;
  env.release_op(t);
  REQUIRE(env.take_op(priority::normal, false, t));
  CHECK(t == first);  // LIFO: the hot slot again

  std::vector<std::uint32_t> held{t};
  std::uint32_t x = 0;
  while (env.take_op(priority::bulk, false, x))
    held.push_back(x);
  CHECK(held.size() == 6);
  CHECK(l.free_ops() == 2);
  REQUIRE(env.take_op(priority::urgent, true, x));  // the reserve is the urgent class's
  held.push_back(x);
  CHECK(l.background_ops() == 1);
  CHECK(l.work_ops() == 6);

  // Fill the SQ with raw doorbell NOPs: the next get_sqe finds no room, submits inline and
  // never returns null.
  for (unsigned i = 0; i < 8; ++i) {
    io_uring_sqe* raw = io_uring_get_sqe(&l.ring());
    REQUIRE(raw != nullptr);
    io_uring_prep_nop(raw);
    io_uring_sqe_set_data64(raw, make_user_data(ud_kind::control, 0, 0));
  }
  CHECK(io_uring_get_sqe(&l.ring()) == nullptr);  // full
  for (std::uint32_t tok : held)
    io_uring_prep_nop(env.get_sqe(tok, ud_kind::op));  // one SQE per token at a time (contract)
  CHECK(l.stats().sq_full_submits > 0);
  CHECK(l.inflight_ops() == held.size());
  REQUIRE(tl.run_until([&] { return r.cqes.size() == held.size(); }, 5s));
  for (auto const& c : r.cqes)
    CHECK(c.second == 0);
  CHECK(l.inflight_ops() == 0);
  CHECK(l.stats().stale_cqes == 0);
  for (std::uint32_t tok : held)
    env.release_op(tok);
  CHECK(l.free_ops() == 8);
  CHECK(l.work_ops() == 0);
  CHECK(l.background_ops() == 0);
}

TEST_CASE("net: cancel_op defers the release to the cancel CQE; stale CQEs are dropped",
          "[rest][net][loop]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  net_test::test_loop tl(uring_loop_config{.sq_entries = 16});
  uring_loop& l = tl.loop();
  loop_env env  = tl.env();
  recorder r;
  bind(env, r);
  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) == 0);
  char buf[64];
  std::uint32_t t = 0;
  REQUIRE(env.take_op(priority::normal, false, t));
  io_uring_prep_recv(env.get_sqe(t, ud_kind::op), sv[0], buf, sizeof buf, 0);
  tl.run_once(0);  // submitted; nothing to read
  CHECK(l.inflight_ops() == 1);
  env.cancel_op(t);
  env.cancel_op(t);  // idempotent while the cancel is pending
  CHECK(l.stats().cancel_ops == 1);
  REQUIRE(tl.run_until([&] { return !r.cqes.empty(); }, 5s));
  CHECK(r.cqes.front().second == -ECANCELED);
  // Released from the CQE handler's point of view; the slot comes back only once the cancel
  // CQE has been seen too (both arrive in the same or the next pass).
  env.release_op(t);
  REQUIRE(tl.run_until([&] { return l.free_ops() == 16; }, 5s));
  // A stale CQE (an older generation of a live slot) is counted and never dispatched.
  std::uint64_t const stale_before = l.stats().stale_cqes;
  std::uint32_t t2                 = 0;
  REQUIRE(env.take_op(priority::normal, false, t2));
  io_uring_prep_recv(env.get_sqe(t2, ud_kind::op), sv[0], buf, sizeof buf, 0);  // in flight
  io_uring_sqe* raw = io_uring_get_sqe(&l.ring());
  REQUIRE(raw != nullptr);
  io_uring_prep_nop(raw);
  io_uring_sqe_set_data64(raw, make_user_data(ud_kind::op, 0x123456u, t2));  // wrong generation
  std::size_t const seen = r.cqes.size();
  REQUIRE(tl.run_until([&] { return l.stats().stale_cqes == stale_before + 1; }, 5s));
  CHECK(r.cqes.size() == seen);
  CHECK(l.inflight_ops() == 1);  // the real receive is untouched
  REQUIRE(::send(sv[1], "x", 1, 0) == 1);
  REQUIRE(tl.run_until([&] { return r.cqes.size() == seen + 1; }, 5s));
  CHECK(r.cqes.back().second == 1);
  env.release_op(t2);
  ::close(sv[0]);
  ::close(sv[1]);
}

namespace {

/// Blocks the loop in park_and_wait (no bound) and wakes it from another thread with notify().
void wake_round_trip(uring_loop_config::wake_mode mode)
{
  net_test::test_loop tl(uring_loop_config{.wake = mode, .max_park_ns = 10'000'000'000ULL});
  uring_loop& l = tl.loop();
  if (mode == uring_loop_config::wake_mode::eventfd) CHECK_FALSE(l.uses_sync_msg_ring());
  std::atomic<bool> work{false};
  std::atomic<std::uint64_t> notified_at{0};
  std::thread producer([&] {
    std::this_thread::sleep_for(100ms);  // let the loop park first
    work.store(true, std::memory_order_relaxed);
    notified_at.store(net_test::now_ns());
    l.notify();
  });
  std::uint64_t const parks_before = l.stats().parks;
  auto const start                 = std::chrono::steady_clock::now();
  while (!work.load(std::memory_order_relaxed)) {
    l.update_now();
    l.reap();
    l.run_due_deadlines();
    l.park_and_wait(uring_loop::no_bound,
                    [&]() noexcept { return work.load(std::memory_order_relaxed); });
  }
  std::uint64_t const woke = net_test::now_ns();
  producer.join();
  auto const took = std::chrono::steady_clock::now() - start;
  CHECK(took < 5s);  // the wake, not the 10 s safety tick
  CHECK(l.stats().parks >= parks_before);
  CHECK(woke >= notified_at.load());
  CHECK(l.wake_send_errors() == 0);
  // The doorbell CQE is reaped by the next pass.
  REQUIRE(tl.run_until([&] { return l.stats().wakes_received >= 1; }, 5s));

  // A busy loop costs producers nothing: notify() without a park sends no wake.
  std::uint64_t const wakes = l.stats().wakes_received;
  for (int i = 0; i < 1000; ++i)
    l.notify();
  for (int i = 0; i < 3; ++i)
    l.run_pass(1'000'000ULL);
  CHECK(l.stats().wakes_received == wakes);
  // wake() always rings.
  l.wake();
  REQUIRE(tl.run_until([&] { return l.stats().wakes_received == wakes + 1; }, 5s));
}

}  // namespace

TEST_CASE("net: cross-thread wake through sync MSG_RING", "[rest][net][loop][wake]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  if (!cached_kernel_features().has_sync_msg_ring) SKIP("kernel < 6.13: no sync MSG_RING");
  {
    net_test::test_loop tl;
    CHECK(tl.loop().uses_sync_msg_ring());
  }
  wake_round_trip(uring_loop_config::wake_mode::automatic);
}

TEST_CASE("net: cross-thread wake through the eventfd fallback", "[rest][net][loop][wake]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  wake_round_trip(uring_loop_config::wake_mode::eventfd);
}

TEST_CASE("net: Dekker park: work published before the park is never slept on",
          "[rest][net][loop][wake]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  net_test::test_loop tl(uring_loop_config{.max_park_ns = 10'000'000'000ULL});
  uring_loop& l = tl.loop();
  // Many rounds of producer/consumer hand-offs: each round the producer publishes one item and
  // notifies; the loop must see every item within the round (a lost wake would cost 10 s).
  std::atomic<unsigned> published{0};
  std::atomic<unsigned> consumed_a{0};
  unsigned consumed         = 0;
  constexpr unsigned rounds = 200;
  std::thread producer([&] {
    for (unsigned i = 0; i < rounds; ++i) {
      while (consumed_a.load(std::memory_order_acquire) != i)
        std::this_thread::yield();
      published.fetch_add(1, std::memory_order_release);
      l.notify();
    }
  });
  auto const start = std::chrono::steady_clock::now();
  while (consumed < rounds && std::chrono::steady_clock::now() - start < 20s) {
    l.update_now();
    l.reap();
    if (published.load(std::memory_order_acquire) > consumed) {
      ++consumed;
      consumed_a.store(consumed, std::memory_order_release);
      continue;
    }
    l.park_and_wait(uring_loop::no_bound, [&]() noexcept {
      return published.load(std::memory_order_acquire) > consumed;
    });
  }
  producer.join();
  CHECK(consumed == rounds);
  CHECK(std::chrono::steady_clock::now() - start < 10s);
}

TEST_CASE("net: shutdown with ops in flight: sync cancel, every CQE reaped", "[rest][net][loop]")
{
  if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available");
  net_test::test_loop tl(uring_loop_config{.sq_entries = 64});
  uring_loop& l = tl.loop();
  loop_env env  = tl.env();
  recorder r;
  bind(env, r);
  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) == 0);
  std::vector<std::uint32_t> toks;
  std::vector<char> buf(16 * 64);
  for (int i = 0; i < 16; ++i) {
    std::uint32_t t = 0;
    REQUIRE(env.take_op(priority::normal, i % 2 == 0, t));
    io_uring_sqe* sqe = env.get_sqe(t, i % 2 == 0 ? ud_kind::idle_watch : ud_kind::op);
    if (i % 2 == 0)
      io_uring_prep_poll_add(sqe, sv[0], POLLIN);
    else
      io_uring_prep_recv(sqe, sv[0], buf.data() + i * 64, 64, 0);
    toks.push_back(t);
  }
  tl.run_once(0);
  CHECK(l.inflight_ops() == 16);
  l.set_stopping(true);
  l.cancel_all();
  CHECK(l.closing());
  REQUIRE(l.drain([&]() noexcept { return r.cqes.size() == 16; }, 5'000'000'000ULL));
  CHECK(l.inflight_ops() == 0);
  for (auto const& c : r.cqes)
    CHECK(c.second == -ECANCELED);
  for (std::uint32_t t : toks)
    env.release_op(t);
  CHECK(l.free_ops() == 64);
  ::close(sv[0]);
  ::close(sv[1]);
}

TEST_CASE("net: kernel feature probe and release parsing", "[rest][net][loop]")
{
  auto const v = parse_kernel_release("7.0.0-38-generic");
  CHECK(v.major == 7);
  CHECK(v.minor == 0);
  CHECK(v.patch == 0);
  CHECK(parse_kernel_release("6.13.2").at_least(6, 13));
  CHECK_FALSE(parse_kernel_release("6.12.9").at_least(6, 13));
  CHECK(parse_kernel_release(nullptr).major == 0);
  kernel_features const& k = cached_kernel_features();
  if (k.has_io_uring) {
    CHECK(k.version.major >= 5);
    if (k.version.at_least(6, 13)) CHECK(k.has_sync_msg_ring);
    if (k.version.at_least(5, 19)) CHECK(k.has_op_socket);
  }
  CHECK(&cached_kernel_features() == &k);  // probed once per process
}
