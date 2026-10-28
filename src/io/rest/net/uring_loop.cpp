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

/**
 * @file uring_loop.cpp
 * @brief The REST ring (uring_loop.hpp): setup (gpuio reactor.cpp `setup_ring`), op slots, reap
 *        and dispatch, the single enter (gpuio `do_enter` / `enter_blocking` /
 *        `park_and_wait`), the doorbell (gpuio wake.cpp), shutdown (`cancel_sockets`), the
 *        kernel feature probe (gpuio kernel_features.cpp) and the io-wq placement (gpuio
 *        pool_support.cpp `apply_iowq_placement`).
 */

#include "io/rest/net/uring_loop.hpp"

#include "io/rest/net/uring_enter.hpp"

#include <cucascade/log/logging.hpp>

#include <sys/eventfd.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <utility>

namespace cucascade::io::net {

namespace {

[[nodiscard]] std::uint64_t mono_now() noexcept
{
  timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

[[nodiscard]] std::error_code sys_error(int negative_errno) noexcept
{
  return {-negative_errno, std::system_category()};
}

/// gpuio sync.hpp `dekker_fence`: the "fence" of `publish; fence; load parked` on the producer
/// and `set parked; fence; re-check` on the loop (store-buffer litmus, gpuio ARCHITECTURE §2.2).
inline void dekker_fence() noexcept { std::atomic_thread_fence(std::memory_order_seq_cst); }

// ---- kernel feature probe (gpuio kernel_features.cpp) ----

/// Throw-away ring created and destroyed on the probing thread.
struct scoped_ring {
  io_uring ring{};
  unsigned features{0};
  bool ok{false};

  explicit scoped_ring(unsigned flags) noexcept
  {
    io_uring_params p{};
    p.flags  = flags;
    ok       = io_uring_queue_init_params(4, &ring, &p) == 0;
    features = ok ? p.features : 0;
  }
  ~scoped_ring()
  {
    if (ok) io_uring_queue_exit(&ring);
  }
  scoped_ring(scoped_ring const&)            = delete;
  scoped_ring& operator=(scoped_ring const&) = delete;
};

/// A real ring-less send into a plain (non-DEFER) ring we own: the CQE is posted synchronously.
bool trial_sync_msg(io_uring& target) noexcept
{
  constexpr std::uint64_t magic = 0x6375'6373'2d72'6e67ULL;
  io_uring_sqe sqe{};
  io_uring_prep_msg_ring(&sqe, target.ring_fd, 0, magic, 0);
  if (io_uring_register_sync_msg(&sqe) < 0) return false;
  io_uring_cqe* cqe = nullptr;
  if (io_uring_peek_cqe(&target, &cqe) != 0 || cqe == nullptr) return false;
  bool const delivered = cqe->user_data == magic;
  io_uring_cqe_seen(&target, cqe);
  return delivered;
}

// ---- io-wq placement (gpuio pool_support.cpp) ----

/// Parse a sysfs cpulist ("0-3,8,10-11") into @p out. @return false when unreadable.
bool read_cpulist(char const* path, cpu_set_t& out) noexcept
{
  CPU_ZERO(&out);
  std::FILE* f = std::fopen(path, "re");
  if (f == nullptr) return false;
  char buf[4096]{};
  std::size_t const n = std::fread(buf, 1, sizeof buf - 1, f);
  std::fclose(f);
  buf[n]   = '\0';
  char* p  = buf;
  bool any = false;
  while (*p != '\0' && *p != '\n') {
    char* end    = nullptr;
    long const a = std::strtol(p, &end, 10);
    if (end == p) break;
    long b = a;
    p      = end;
    if (*p == '-') {
      b = std::strtol(p + 1, &end, 10);
      p = end;
    }
    for (long c = a; c <= b && c < CPU_SETSIZE; ++c) {
      CPU_SET(static_cast<std::size_t>(c), &out);
      any = true;
    }
    if (*p == ',') ++p;
  }
  return any;
}

}  // namespace

// ================================================================================================
// Kernel features
// ================================================================================================

kernel_version parse_kernel_release(char const* release) noexcept
{
  kernel_version v{};
  if (release == nullptr) return v;
  unsigned* fields[] = {&v.major, &v.minor, &v.patch};
  char const* p      = release;
  for (unsigned* f : fields) {
    if (*p < '0' || *p > '9') break;
    unsigned x = 0;
    while (*p >= '0' && *p <= '9')
      x = x * 10 + static_cast<unsigned>(*p++ - '0');
    *f = x;
    if (*p != '.') break;
    ++p;
  }
  return v;
}

kernel_features probe_kernel_features() noexcept
{
  kernel_features k{};
  utsname u{};
  if (::uname(&u) == 0) k.version = parse_kernel_release(u.release);

  scoped_ring a(0);
  if (!a.ok) return k;
  k.has_io_uring  = true;
  k.has_no_iowait = (a.features & IORING_FEAT_NO_IOWAIT) != 0;

  if (io_uring_probe* probe = io_uring_get_probe_ring(&a.ring)) {
    k.has_msg_ring  = io_uring_opcode_supported(probe, IORING_OP_MSG_RING) != 0;
    k.has_op_socket = io_uring_opcode_supported(probe, IORING_OP_SOCKET) != 0;
    io_uring_free_probe(probe);
  }
  {
    scoped_ring d(IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
    k.has_defer_taskrun = d.ok;
  }
  k.has_sync_msg_ring = k.has_msg_ring && trial_sync_msg(a.ring);
  return k;
}

kernel_features const& cached_kernel_features() noexcept
{
  static kernel_features const features = probe_kernel_features();
  return features;
}

// ================================================================================================
// io-wq placement (gpuio ARCHITECTURE §4.3)
// ================================================================================================

iowq_placement apply_iowq_placement(io_uring& ring,
                                    unsigned max_unbound,
                                    std::span<int const> cpus) noexcept
{
  iowq_placement p{};
  // Cap: bound workers untouched (0), unbound = iowq_max_unbound; 0 = auto = 16 (gpuio T20c: the
  // EC2 sweep: 4 -> 8 gave +37%, 16 reached NIC parity; uncapped costs +55% CPU/GB and halved EC2
  // throughput). Never "unlimited".
  unsigned vals[2]   = {0, max_unbound == 0 ? 16u : max_unbound};
  p.unbound_cap      = vals[1];
  p.max_workers_rc   = io_uring_register_iowq_max_workers(&ring, vals);
  p.previous_unbound = vals[1];

  // Affinity: explicit list, else the NUMA-local CPUs minus the reactor's own CPU when the
  // reactor is pinned to one (a pinned reactor without IOWQ_AFF collapses io-wq onto its core).
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (!cpus.empty()) {
    for (int c : cpus)
      if (c >= 0 && c < CPU_SETSIZE) CPU_SET(static_cast<std::size_t>(c), &mask);
  } else {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (::sched_getaffinity(0, sizeof allowed, &allowed) != 0) CPU_ZERO(&allowed);
    cpu_set_t node;
    CPU_ZERO(&node);
    int const cpu  = ::sched_getcpu();
    bool have_node = false;
    if (cpu >= 0) {
      char path[128];
      for (int n = 0; n < 64 && !have_node; ++n) {
        std::snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", n);
        cpu_set_t s;
        if (!read_cpulist(path, s)) continue;
        if (CPU_ISSET(static_cast<std::size_t>(cpu), &s)) {
          node      = s;
          have_node = true;
        }
      }
    }
    if (!have_node) {
      CPU_ZERO(&node);
      for (long c = 0, n = ::sysconf(_SC_NPROCESSORS_ONLN); c < n && c < CPU_SETSIZE; ++c)
        CPU_SET(static_cast<std::size_t>(c), &node);
    }
    if (CPU_COUNT(&allowed) == 1) {
      // Pinned reactor: everything NUMA-local except its core.
      for (std::size_t c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &node) && !CPU_ISSET(c, &allowed)) CPU_SET(c, &mask);
    } else {
      CPU_AND(&mask, &allowed, &node);
    }
    if (CPU_COUNT(&mask) == 0) mask = allowed;
  }
  p.cpus        = static_cast<unsigned>(CPU_COUNT(&mask));
  p.mask        = mask;
  p.affinity_rc = p.cpus == 0 ? -EINVAL : io_uring_register_iowq_aff(&ring, sizeof mask, &mask);
  return p;
}

// ================================================================================================
// Construction / setup
// ================================================================================================

uring_loop::uring_loop(uring_loop_config cfg) noexcept : _cfg(std::move(cfg))
{
  _cfg.sq_entries         = std::clamp(_cfg.sq_entries, 8u, 32768u);
  std::uint32_t const cap = _cfg.sq_entries;
  _slots.reset(new (std::nothrow) op_slot[cap]());
  _free.reset(new (std::nothrow) std::uint32_t[cap]);
  if (!_slots || !_free || !_deadlines.init(cap + 256)) {
    _slots.reset();
    _free.reset();
    return;
  }
  for (std::uint32_t i = cap; i-- > 0;) {
    _slots[i].index      = i;
    _free[_free_count++] = i;  // slot 0 on top (LIFO: hot slots stay hot)
  }
  _capacity = cap;
  _reserve  = std::min<std::uint32_t>(_cfg.urgent_reserve_slots, cap / 4);
  _now      = mono_now();
}

uring_loop::~uring_loop() { close(); }

std::error_code uring_loop::start() noexcept
{
  if (_ring_ok) return {};
  if (_capacity == 0) return make_error_code(std::errc::not_enough_memory);
  kernel_features const& kf = cached_kernel_features();
  io_uring_params p{};
  // SINGLE_ISSUER | DEFER_TASKRUN | TASKRUN_FLAG | SUBMIT_ALL | CQSIZE | CLAMP; NO_SQARRAY is
  // liburing's default. Created R_DISABLED and enabled here, on the owning thread, which becomes
  // the issuer (gpuio io_uring_knobs §2.3; every later io_uring_register_* comes from this thread
  // or fails with -EEXIST). DEFER_TASKRUN: kTLS receive inline 6.69 vs 5.98 GB/s (+12%, gpuio U1).
  p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_TASKRUN_FLAG |
            IORING_SETUP_SUBMIT_ALL | IORING_SETUP_CQSIZE | IORING_SETUP_CLAMP |
            IORING_SETUP_R_DISABLED;
  p.cq_entries = 4 * _cfg.sq_entries;  // CQ = 4 x SQ (gpuio ARCHITECTURE §11, U5)
  int rc       = io_uring_queue_init_params(_cfg.sq_entries, &_ring, &p);
  if (rc < 0) return sys_error(rc);
  rc = io_uring_enable_rings(&_ring);
  if (rc < 0) {
    io_uring_queue_exit(&_ring);
    return sys_error(rc);
  }
  _ring_ok       = true;
  _ring_fd       = _ring.ring_fd;
  _has_no_iowait = (p.features & IORING_FEAT_NO_IOWAIT) != 0;
  if (_cfg.register_ring_fd) {
    // Removes the fdget/fdput of every enter (gpuio io_uring_knobs §2.4); best effort (16 per
    // thread).
    _ring_fd_registered = io_uring_register_ring_fd(&_ring) == 1;
  }
  bool const want_sync =
    _cfg.wake == uring_loop_config::wake_mode::msg_ring ||
    (_cfg.wake == uring_loop_config::wake_mode::automatic && kf.has_sync_msg_ring);
  _sync_wake = want_sync && kf.has_msg_ring;
  if (!_sync_wake) {
    // Blocking eventfd: io_uring issues the READ non-blocking internally and arms its own poll
    // when the counter is 0, so the READ completes on the next write (gpuio io_uring_knobs
    // §2.5.8).
    _event_fd = ::eventfd(0, EFD_CLOEXEC);
    if (_event_fd < 0) {
      close();
      return make_error_code(errc::ring_error);
    }
    post_wake_read();
  }
  update_now();
  return {};
}

void uring_loop::close() noexcept
{
  if (_ring_ok) {
    if (_ring_fd_registered) (void)io_uring_unregister_ring_fd(&_ring);
    _ring_fd_registered = false;
    io_uring_queue_exit(&_ring);
    _ring_ok = false;
  }
  // After the ring: a posted eventfd READ targets `_wake_read_buf` until the ring is gone.
  if (_event_fd >= 0) ::close(_event_fd);
  _event_fd         = -1;
  _wake_read_posted = false;
}

// ================================================================================================
// Op slots and SQEs
// ================================================================================================

bool uring_loop::take_op(
  void* owner, op_handler handler, priority p, bool background, std::uint32_t& token) noexcept
{
  std::uint32_t const keep = p == priority::urgent ? 0 : _reserve;
  if (_free_count <= keep) return false;
  op_slot& s         = _slots[_free[--_free_count]];
  s.generation       = (s.generation + 1) & ud_generation_mask;
  s.on_cqe           = handler;
  s.owner            = owner;
  s.kind             = ud_kind::op;
  s.background       = background;
  s.release_deferred = false;
  if (background)
    ++_background_slots;
  else
    ++_work_slots;
  token = s.index;
  return true;
}

void uring_loop::release_op(std::uint32_t token) noexcept
{
  if (token >= _capacity) return;
  release_slot(_slots[token]);
}

void uring_loop::release_slot(op_slot& s) noexcept
{
  if (s.in_flight) {  // misuse: an SQE of this slot is still in flight; never recycle it
    ++_stats.misuse;
    return;
  }
  if (s.cancel_pending) {
    s.release_deferred = true;  // recycled when the ASYNC_CANCEL CQE arrives
    return;
  }
  s.on_cqe = nullptr;
  s.owner  = nullptr;
  if (s.background)
    --_background_slots;
  else
    --_work_slots;
  _free[_free_count++] = s.index;
}

io_uring_sqe* uring_loop::get_raw_sqe() noexcept
{
  io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
  return sqe != nullptr ? sqe : get_sqe_slow();
}

io_uring_sqe* uring_loop::get_sqe_slow() noexcept
{
  // SQ full: hand what is prepared to the kernel (one extra enter, counted) and retry.
  for (unsigned attempt = 0;; ++attempt) {
    ++_stats.sq_full_submits;
    (void)do_enter(uring::flush_sq(_ring), 0, 0);
    if (io_uring_sqe* sqe = io_uring_get_sqe(&_ring)) return sqe;
    if (attempt > 16) {
      timespec const ts{0, 10'000};
      ::nanosleep(&ts, nullptr);
    }
  }
}

io_uring_sqe* uring_loop::get_sqe(std::uint32_t token, ud_kind kind) noexcept
{
  op_slot& s        = _slots[token];
  s.kind            = kind;
  io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
  if (sqe == nullptr) [[unlikely]]
    sqe = get_sqe_slow();
  io_uring_sqe_set_data64(sqe, make_user_data(s.kind, s.generation, s.index));
  if (!s.in_flight) {
    s.in_flight = true;
    ++_inflight;
  }
  ++_stats.ops_submitted;
  return sqe;
}

void uring_loop::cancel_op(std::uint32_t token) noexcept
{
  if (token >= _capacity) return;
  op_slot& s = _slots[token];
  if (!s.in_flight || s.cancel_pending) return;
  io_uring_sqe* sqe = get_raw_sqe();
  io_uring_prep_cancel64(sqe, make_user_data(s.kind, s.generation, s.index), 0);
  io_uring_sqe_set_data64(sqe, make_user_data(ud_kind::cancel, s.generation, s.index));
  s.cancel_pending = true;
  ++_stats.cancel_ops;
}

// ================================================================================================
// Time
// ================================================================================================

void uring_loop::update_now() noexcept
{
  _now = mono_now();
  ++_stats.passes;
}

std::size_t uring_loop::run_due_deadlines() noexcept
{
  deadline_heap<deadline_fn>::entry e{};
  std::size_t fired = 0;
  // Bounded: callbacks may re-arm at or before `now`; those run next pass.
  for (std::uint32_t n = _deadlines.size(); n != 0 && _deadlines.pop_due(_now, e); --n) {
    ++_stats.deadlines_fired;
    ++fired;
    e.fn(e.arg);
  }
  return fired;
}

// ================================================================================================
// Reap and dispatch
// ================================================================================================

std::size_t uring_loop::reap() noexcept
{
  if (!_ring_ok) return 0;
  std::size_t total   = 0;
  unsigned const mask = _ring.cq.ring_mask;
  unsigned const sh   = io_uring_cqe_shift(&_ring);
  // Manual CQ walk + one cq_advance per round (folly; gpuio R9); never enters the kernel (CQEs
  // were posted by the previous pass's enter: DEFER_TASKRUN runs task_work only in GETEVENTS).
  for (unsigned round = 0; round < 8; ++round) {
    unsigned head = std::atomic_ref<unsigned>(*_ring.cq.khead).load(std::memory_order_relaxed);
    unsigned const tail =
      std::atomic_ref<unsigned>(*_ring.cq.ktail).load(std::memory_order_acquire);
    unsigned const n = tail - head;
    if (n == 0) break;
    for (unsigned i = 0; i < n; ++i, ++head) {
      io_uring_cqe const* cqe = &_ring.cq.cqes[(head & mask) << sh];
      dispatch_cqe(cqe->user_data, cqe->res, cqe->flags);
    }
    io_uring_cq_advance(&_ring, n);
    total += n;
    if (n == 1)
      ++_stats.cqe_batches_1;
    else if (n <= 8)
      ++_stats.cqe_batches_2_8;
    else
      ++_stats.cqe_batches_9plus;
  }
  _stats.cqes += total;
  return total;
}

void uring_loop::dispatch_cqe(std::uint64_t ud, int res, unsigned flags) noexcept
{
  ud_kind const k = ud_kind_of(ud);
  switch (k) {
    case ud_kind::wake:
    case ud_kind::control: on_doorbell(); return;
    case ud_kind::cancel: on_cancel_cqe(ud); return;
    default: break;
  }
  std::uint32_t const idx = ud_index_of(ud);
  if (idx >= _capacity) [[unlikely]] {
    ++_stats.stale_cqes;
    return;
  }
  op_slot& s = _slots[idx];
  if (!s.in_flight || s.generation != ud_generation_of(ud)) [[unlikely]] {
    ++_stats.stale_cqes;  // stale CQE: never dispatch to a reused slot
    return;
  }
  if ((flags & IORING_CQE_F_MORE) == 0) {
    s.in_flight = false;
    --_inflight;
    ++_stats.ops_completed;
  }
  if (s.on_cqe != nullptr) [[likely]]
    s.on_cqe(s.owner, s.index, res, flags);
  else if (!s.in_flight)
    release_slot(s);
}

void uring_loop::on_doorbell() noexcept
{
  ++_stats.wakes_received;
  // eventfd fallback: every doorbell CQE is the READ completing (the sender's kind is not
  // carried): re-post it, unless the loop is shutting down (sync cancel). MSG_RING mode never
  // needs this.
  if (!_sync_wake) {
    _wake_read_posted = false;
    if (!_cancel_requested) post_wake_read();
  }
}

void uring_loop::on_cancel_cqe(std::uint64_t ud) noexcept
{
  std::uint32_t const idx = ud_index_of(ud);
  if (idx >= _capacity) return;
  op_slot& s = _slots[idx];
  if (s.generation != ud_generation_of(ud) || !s.cancel_pending) return;
  s.cancel_pending = false;
  if (s.release_deferred) {
    s.release_deferred = false;
    release_slot(s);
  }
}

// ================================================================================================
// Wake (any thread except post_wake_read)
// ================================================================================================

void uring_loop::post_wake_read() noexcept
{
  if (_wake_read_posted || _event_fd < 0) return;
  io_uring_sqe* sqe = get_raw_sqe();
  io_uring_prep_read(sqe, _event_fd, &_wake_read_buf, sizeof _wake_read_buf, 0);
  io_uring_sqe_set_data64(sqe, make_user_data(ud_kind::wake, 0, 0));
  _wake_read_posted = true;
}

void uring_loop::send_wake() noexcept
{
  if (_sync_wake) {
    io_uring_sqe sqe{};  // zeroed: the kernel rejects non-zero reserved fields
    io_uring_prep_msg_ring(&sqe, _ring_fd, 0, make_user_data(ud_kind::wake, 0, 0), 0);
    // -ENOMEM (task_work request allocation into a DEFER target) and -EINTR are transient.
    for (int attempt = 0; attempt < 8; ++attempt) {
      int const rc = io_uring_register_sync_msg(&sqe);
      if (rc >= 0) return;
      if (rc != -EINTR && rc != -EAGAIN && rc != -ENOMEM) break;
    }
  } else if (_event_fd >= 0) {
    std::uint64_t const one = 1;
    for (;;) {
      auto const w = ::write(_event_fd, &one, sizeof one);
      if (w == static_cast<ssize_t>(sizeof one)) return;
      if (w < 0 && errno == EINTR) continue;
      break;
    }
  }
  _wake_errors.fetch_add(1, std::memory_order_relaxed);
}

void uring_loop::wake() noexcept { send_wake(); }

void uring_loop::notify() noexcept
{
  // Producer side of the Dekker protocol: the caller published its work; fence; only a loop that
  // published its parked flag needs a wake, and only the thread whose exchange observed the flag
  // set sends it (one wake per park; a busy loop costs no RMW and no syscall).
  dekker_fence();
  if (!_parked.load(std::memory_order_relaxed)) return;
  if (_parked.exchange(false, std::memory_order_acq_rel)) send_wake();
}

bool uring_loop::park_prepare() noexcept
{
  if (!_ring_ok) return false;
  _parked.store(true, std::memory_order_seq_cst);
  dekker_fence();
  return true;
}

void uring_loop::park_cancel() noexcept
{
  // Work became visible: withdraw. If a producer already took the flag it owes us one wake,
  // which arrives later as a spurious doorbell (correct either way).
  (void)_parked.exchange(false, std::memory_order_acq_rel);
  ++_stats.park_cancels;
}

void uring_loop::park_finish() noexcept
{
  if (_parked.load(std::memory_order_relaxed)) (void)_parked.exchange(false);
}

// ================================================================================================
// The single enter (gpuio ARCHITECTURE §9.2: one io_uring_enter per pass)
// ================================================================================================

int uring_loop::do_enter(unsigned to_submit, unsigned wait_nr, std::uint64_t timeout_ns) noexcept
{
  _enter_skips = 0;
  uring::enter_args a{};
  a.wait_nr     = wait_nr;
  a.getevents   = true;
  a.has_timeout = wait_nr != 0;
  a.timeout_ns  = timeout_ns;
  a.registered  = _ring_fd_registered;
  // R7: every op on this ring is a socket op: iowait never applies (IORING_ENTER_NO_IOWAIT when
  // the kernel advertises it; else iowait stays on, as before 6.15).
  a.no_iowait = _has_no_iowait;
  ++_stats.enters;
  int const r = uring::enter(_ring, to_submit, a);
  if (r < 0 && r != -ETIME && r != -EINTR && r != -EAGAIN && r != -EBUSY) [[unlikely]] {
    // -EAGAIN/-EBUSY: CQ overflow backpressure; the unsubmitted SQEs stay queued and are retried
    // next pass (uring_ultra L14 treated this as fatal).
    if (_stats.enter_errors++ == 0)
      CUCASCADE_LOG_ERROR("rest ring: io_uring_enter: {}", sys_error(r).message());
  }
  return r;
}

void uring_loop::submit() noexcept
{
  if (!_ring_ok) return;
  unsigned const flags = uring::sq_flags(_ring);
  if ((flags & IORING_SQ_CQ_OVERFLOW) != 0) ++_stats.cq_overflows;
  unsigned const to_submit = uring::flush_sq(_ring);
  // Nothing to submit, no deferred task_work, no overflow: skip the syscall.
  if (to_submit == 0 && (flags & (IORING_SQ_TASKRUN | IORING_SQ_CQ_OVERFLOW)) == 0) {
    _enter_skips = 0;
    return;
  }
  (void)do_enter(to_submit, 0, 0);
}

void uring_loop::enter_blocking(std::uint64_t timeout_ns) noexcept
{
  unsigned const flags = uring::sq_flags(_ring);
  if ((flags & IORING_SQ_CQ_OVERFLOW) != 0) ++_stats.cq_overflows;
  // R5/R6: socket ops only: wait for 1 CQE (min-wait batching is for storage-dense waits only;
  // under DEFER, MSG_RING and socket task_work only wake a wait_nr = 1 waiter promptly).
  ++_stats.waits;
  (void)do_enter(uring::flush_sq(_ring), 1, timeout_ns);
}

std::uint64_t uring_loop::wait_timeout_ns(std::uint64_t bound) const noexcept
{
  std::uint64_t t          = _cfg.max_park_ns;
  std::uint64_t const next = _deadlines.next();
  if (next != ~std::uint64_t{0}) t = std::min(t, next > _now ? next - _now : 0);
  return std::min(t, bound);
}

void uring_loop::skip_or_submit(bool runnable) noexcept
{
  // gpuio T9b: more work is runnable now: go straight to the next pass and let its enter submit
  // this pass's SQEs, up to `max_enter_skips` passes in a row (CQEs keep being reaped from the
  // ring; deferred task_work waits at most that many passes). split_w1: 1.168 -> 1.085 enters/op.
  if (runnable && _enter_skips < max_enter_skips) {
    ++_enter_skips;
    ++_stats.enter_skips;
    return;
  }
  submit();
}

void uring_loop::submit_and_wait(std::uint64_t bound) noexcept
{
  if (!_ring_ok) return;
  std::uint64_t const t = wait_timeout_ns(bound);
  if (t == 0) {
    submit();
    return;
  }
  enter_blocking(t);
}

void uring_loop::run_pass(std::uint64_t bound) noexcept
{
  update_now();
  reap();
  run_due_deadlines();
  submit_and_wait(bound);
}

// ================================================================================================
// Shutdown (U14)
// ================================================================================================

void uring_loop::cancel_all() noexcept
{
  _cancel_requested = true;
  if (!_ring_ok) return;
  // Sync cancel ANY|ALL (gpuio io_uring_knobs §2.5.8): sockets and polls get -ECANCELED CQEs.
  // The CQEs are reaped by `drain` (uring_ultra L13: never zero the in-flight count without
  // reaping). Prepared but unsubmitted SQEs are submitted first so they are cancelled too.
  submit();
  io_uring_sync_cancel_reg reg{};
  reg.flags          = IORING_ASYNC_CANCEL_ANY | IORING_ASYNC_CANCEL_ALL;
  reg.fd             = -1;
  reg.timeout.tv_sec = 1;
  ++_stats.sync_cancels;
  _submitted_at_cancel = _stats.ops_submitted;
  (void)io_uring_register_sync_cancel(&_ring, &reg);
}

}  // namespace cucascade::io::net
