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

#include "io/rest/net/common.hpp"
#include "io/rest/net/deadline_heap.hpp"

#include <liburing.h>
#include <sched.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <system_error>
#include <type_traits>

/**
 * @file uring_loop.hpp
 * @brief The host reactor of the REST transport: one io_uring owned by one thread, op tokens,
 *        the single enter per pass, the deadline heap, the cross-thread wake and the shutdown
 *        protocol. These are the services gpuio's `reactor` gave its remote backend (gpuio
 *        `reactor.hpp` §user_data / op slots, `reactor.cpp` pass / reap / enter / drain,
 *        `wake.hpp`), reduced to what a socket-only ring needs.
 *
 * ## How the REST worker drives it (one thread; everything below on that thread unless marked
 * "any thread")
 * ```
 *   uring_loop loop{cfg};
 *   loop.start();                                   // on the worker thread: ring, fd, wake
 *   loop_env env{loop};                             // the pool's Env (one per pool)
 *   connection_pool<loop_env, H, T> pool{env, h, rc};
 *   while (!stop) {
 *     loop.update_now();                            // cache now_ns for the pass
 *     loop.reap();                                  // CQEs -> pool.on_op_cqe (manual CQ walk)
 *     loop.run_due_deadlines();                     // pool ticks, connect / stall deadlines
 *     ... retire device copies, intake, pump ...    // may get_sqe / arm_deadline freely
 *     loop.park_and_wait(copy_eta_or_no_bound,      // the ONE enter of the pass (R5)
 *                        [&]() noexcept { return intake_nonempty() || stop; });
 *   }
 *   pool.shutdown(); loop.cancel_all();             // U14
 *   loop.drain([&]() noexcept { return pool.quiescent(); }, timeout);
 *   loop.close();                                   // on the worker thread
 * ```
 * `run_pass(work, has_work)` bundles one such pass. Producers (`enqueue`, internal posts,
 * `interrupt`, `shutdown`) publish their work (queue push, flag) and then call `notify()` (any
 * thread): it costs one fence and one relaxed load while the worker is busy and wakes it (one
 * syscall) only when it announced it is parking (Dekker, gpuio D2/D4). `wake()` (any thread)
 * rings the doorbell unconditionally.
 *
 * ## The pass's wait (gpuio R5, R7, R8, R9)
 * `park_and_wait(bound, has_work)` is the single `io_uring_enter` of a pass (raw syscall:
 * `GETEVENTS | EXT_ARG | NO_IOWAIT | REGISTERED_RING`), `wait_nr = 1`, bounded by the deadline
 * heap's head, by @p bound (the caller's external bound: the H2D copy ETA; `no_bound` for none)
 * and by a 1 s safety tick. `bound == 0` means more work is runnable now: the enter is skipped
 * for up to 8 passes (CQEs keep being reaped from the ring), then a non-blocking enter submits.
 * There are no timeout SQEs and no linked SQEs. CQ overflow (-EBUSY/-EAGAIN) is backpressure:
 * the SQEs stay queued for the next pass.
 *
 * ## Op tokens (gpuio map §3.1-§3.2, Env contract §3.4)
 * A fixed table of `sq_entries` op slots with a LIFO free list. `user_data` is
 * `[63:56] ud_kind | [55:32] generation (24 bit) | [31:0] index`. `get_sqe` never returns null
 * (it submits inline when the SQ is full); `cancel_op` issues `ASYNC_CANCEL` and the slot's
 * release is deferred to the cancel CQE; a CQE whose generation does not match, or whose slot is
 * not in flight, is stale: counted and dropped.
 *
 * ## Wake (gpuio D4)
 * Ring-less `MSG_RING` through `io_uring_register_sync_msg` (kernel >= 6.13), else an eventfd
 * with one `IORING_OP_READ` kept posted (no POLL_ADD, no read syscall).
 */

namespace cucascade::io::net {

// ------------------------------------------------------------------------------------------------
// user_data encoding (gpuio reactor.hpp)
// ------------------------------------------------------------------------------------------------

/// user_data layout: [63:56] kind, [55:32] generation, [31:0] slot index. `op`, `idle_watch`,
/// `handshake_poll` and `timeout` route to the slot's owner (the kind is only a tag); `cancel`
/// completes a pending ASYNC_CANCEL; `wake` and `control` are doorbells.
enum class ud_kind : std::uint8_t {
  op,
  cancel,
  wake,
  control,
  idle_watch,
  handshake_poll,
  timeout
};

inline constexpr std::uint32_t ud_generation_mask = 0x00FF'FFFFu;

[[nodiscard]] constexpr std::uint64_t make_user_data(ud_kind k,
                                                     std::uint32_t generation,
                                                     std::uint32_t index) noexcept
{
  return (static_cast<std::uint64_t>(k) << 56) |
         (static_cast<std::uint64_t>(generation & ud_generation_mask) << 32) | index;
}
[[nodiscard]] constexpr ud_kind ud_kind_of(std::uint64_t ud) noexcept
{
  return static_cast<ud_kind>(ud >> 56);
}
[[nodiscard]] constexpr std::uint32_t ud_generation_of(std::uint64_t ud) noexcept
{
  return static_cast<std::uint32_t>(ud >> 32) & ud_generation_mask;
}
[[nodiscard]] constexpr std::uint32_t ud_index_of(std::uint64_t ud) noexcept
{
  return static_cast<std::uint32_t>(ud);
}

/// Deadline callback (`arm_deadline`): runs on the loop thread once `now_ns() >= when`.
using deadline_fn = void (*)(void* arg) noexcept;
/// CQE callback of an op slot's owner (`take_op`): the only indirect call per CQE.
using op_handler = void (*)(void* owner, std::uint32_t token, int res, unsigned flags) noexcept;

// ------------------------------------------------------------------------------------------------
// Kernel features (gpuio kernel_features.hpp, reduced to what this ring needs: K1-K3)
// ------------------------------------------------------------------------------------------------

struct kernel_version {
  unsigned major{0};
  unsigned minor{0};
  unsigned patch{0};

  [[nodiscard]] constexpr bool at_least(unsigned ma, unsigned mi) const noexcept
  {
    return major > ma || (major == ma && minor >= mi);
  }
};

/// @brief What the running kernel supports. All flags are false when io_uring is unavailable
/// (`kernel.io_uring_disabled`, seccomp).
struct kernel_features {
  kernel_version version{};       ///< uname(2) release, e.g. 7.0.0
  bool has_io_uring{false};       ///< io_uring_setup succeeded
  bool has_defer_taskrun{false};  ///< SINGLE_ISSUER | DEFER_TASKRUN ring created (6.1)
  bool has_msg_ring{false};       ///< IORING_OP_MSG_RING opcode (5.18)
  bool has_sync_msg_ring{false};  ///< IORING_REGISTER_SEND_MSG_RING trial send delivered (6.13)
  bool has_op_socket{false};      ///< IORING_OP_SOCKET (5.19; else socket(2))
  bool has_no_iowait{false};      ///< IORING_FEAT_NO_IOWAIT (6.15; else iowait stays on)

  /// @return true when the REST ring can run: io_uring allowed, DEFER_TASKRUN | SINGLE_ISSUER
  /// (floor 6.1; gpuio's 6.12 floor is for local-file features this ring never uses).
  [[nodiscard]] bool meets_floor() const noexcept
  {
    return has_io_uring && has_defer_taskrun && version.at_least(6, 1);
  }
};

/// @brief Run every probe now (a few throw-away rings on the calling thread; ~100 us).
[[nodiscard]] kernel_features probe_kernel_features() noexcept;
/// @brief `probe_kernel_features()` once per process (thread-safe), then cached.
[[nodiscard]] kernel_features const& cached_kernel_features() noexcept;
/// @brief Parse a uname(2) release string ("7.0.0-38-generic"); missing fields are 0.
[[nodiscard]] kernel_version parse_kernel_release(char const* release) noexcept;

// ------------------------------------------------------------------------------------------------
// io-wq placement (gpuio U15 / gpuio T9)
// ------------------------------------------------------------------------------------------------

/// What `apply_iowq_placement` did (gpuio ARCHITECTURE §4.3).
struct iowq_placement {
  int max_workers_rc{0};    ///< io_uring_register_iowq_max_workers result
  int affinity_rc{0};       ///< io_uring_register_iowq_aff result
  unsigned unbound_cap{0};  ///< cap applied
  unsigned previous_unbound{0};
  unsigned cpus{0};  ///< CPUs in the affinity mask applied
  cpu_set_t mask{};
};

/// @brief Cap (bound 0 = unchanged, unbound = @p max_unbound, 0 = auto = 16, never unlimited)
/// and affine (@p cpus, else the calling thread's NUMA-local allowed CPUs minus its own CPU when
/// it is pinned to exactly one) the ring's io-wq. Owner thread; once per ring, before the first
/// receive that carries IOSQE_ASYNC (EC2: uncapped 5.45 GB/s vs capped 11.56 GB/s; a pinned
/// reactor without IOWQ_AFF collapses decryption onto its own core).
[[nodiscard]] iowq_placement apply_iowq_placement(io_uring& ring,
                                                  unsigned max_unbound,
                                                  std::span<int const> cpus) noexcept;

// ------------------------------------------------------------------------------------------------
// The loop
// ------------------------------------------------------------------------------------------------

/// @brief Construction settings of a `uring_loop`.
struct uring_loop_config {
  /// SQ entries and op slots (U5: 1024; clamped to [8, 32768]); CQ = 4 x SQ.
  unsigned sq_entries{1024};
  /// Op slots non-urgent `take_op` callers cannot take (gpuio `urgent_reserve_slots`; capped at
  /// sq_entries / 4).
  unsigned urgent_reserve_slots{4};
  /// Doorbell mechanism: `automatic` = sync MSG_RING when the kernel has it (>= 6.13), else
  /// eventfd. `eventfd` forces the fallback (tests, field fallbacks).
  enum class wake_mode : std::uint8_t { automatic, msg_ring, eventfd } wake{wake_mode::automatic};
  /// Register the ring fd (U6): removes fdget/fdput from every enter.
  bool register_ring_fd{true};
  /// Safety tick: no wait is longer (lost-wake insurance; gpuio `max_park_ns`).
  std::uint64_t max_park_ns{1'000'000'000ULL};
};

/// @brief Counters of one loop (owner thread writes; read them on that thread).
struct uring_loop_stats {
  std::uint64_t passes{0};           ///< update_now() calls
  std::uint64_t enters{0};           ///< io_uring_enter syscalls (every one is counted)
  std::uint64_t waits{0};            ///< blocking enters
  std::uint64_t enter_skips{0};      ///< passes that skipped the enter (more work runnable)
  std::uint64_t enter_errors{0};     ///< enters failing with something else than ETIME/EINTR/...
  std::uint64_t sq_full_submits{0};  ///< inline submits because the SQ was full (get_sqe)
  std::uint64_t cq_overflows{0};     ///< passes that saw IORING_SQ_CQ_OVERFLOW
  std::uint64_t cqes{0};             ///< CQEs reaped (doorbells included)
  std::uint64_t cqe_batches_1{0}, cqe_batches_2_8{0}, cqe_batches_9plus{0};
  std::uint64_t stale_cqes{0};     ///< generation mismatch / slot not in flight: dropped
  std::uint64_t ops_submitted{0};  ///< SQEs handed out by get_sqe
  std::uint64_t ops_completed{0};  ///< final CQEs of op slots
  std::uint64_t cancel_ops{0};     ///< ASYNC_CANCEL SQEs
  std::uint64_t misuse{0};         ///< release of a slot with an SQE in flight (never recycled)
  std::uint64_t deadlines_fired{0};
  std::uint64_t parks{0};           ///< blocking waits with the parked flag published
  std::uint64_t park_cancels{0};    ///< parks withdrawn by the Dekker re-check
  std::uint64_t wakes_received{0};  ///< doorbell CQEs
  std::uint64_t sync_cancels{0};    ///< io_uring_register_sync_cancel calls
};

/**
 * @brief One io_uring owned by one thread, with gpuio's reactor services for a socket-only ring.
 *
 * Construction allocates the op-slot table and the deadline heap (the only allocations). `start`
 * creates the ring on the calling thread, which becomes its owner: `IORING_SETUP_SINGLE_ISSUER |
 * DEFER_TASKRUN | TASKRUN_FLAG | SUBMIT_ALL | CQSIZE (4 x SQ) | CLAMP | R_DISABLED`, enabled
 * there (U1); no SQPOLL, no COOP_TASKRUN, no IOPOLL (U2-U4). Every member is owner-thread only
 * except `wake`, `notify`, `uses_sync_msg_ring` and `wake_send_errors`.
 *
 * Neither copyable nor movable (op slots and the eventfd READ target are addressed by the
 * kernel). Destroy it on the owner thread, or call `close` there first.
 */
class uring_loop {
 public:
  /// Passes as the caller's wait bound: no external bound.
  static constexpr std::uint64_t no_bound = ~std::uint64_t{0};
  /// Consecutive passes that may skip the enter while work is runnable (gpuio T9b: 8).
  static constexpr unsigned max_enter_skips = 8;

  explicit uring_loop(uring_loop_config cfg = {}) noexcept;
  ~uring_loop();
  uring_loop(uring_loop const&)            = delete;
  uring_loop& operator=(uring_loop const&) = delete;
  uring_loop(uring_loop&&)                 = delete;
  uring_loop& operator=(uring_loop&&)      = delete;

  /// @brief Create the ring on the calling thread (the owner) and set up the wake.
  /// @return {} or the setup errno (io_uring forbidden: EPERM / ENOSYS; ENOMEM: memlock), or
  /// `errc::ring_error` when the eventfd fallback could not be created.
  [[nodiscard]] std::error_code start() noexcept;
  /// @brief Owner thread: unregister the ring fd, tear the ring down (no CQE is reaped: call
  /// `cancel_all` + `drain` first). Idempotent; the destructor calls it.
  ///
  /// @note Tearing a ring down makes the kernel's exit worker queue TWA_SIGNAL task work to every
  /// thread that created or used it, asynchronously; on a thread that lives on, the next blocking
  /// call that the kernel does not restart (an SO_RCVTIMEO recv) fails with EINTR. Own rings on
  /// threads that end with them (the reactor workers, `rest_reactor`'s per-call helpers).
  void close() noexcept;
  [[nodiscard]] bool started() const noexcept { return _ring_ok; }

  // ---- op slots (the Env contract; `loop_env` binds an owner) ----

  [[nodiscard]] std::uint32_t op_capacity() const noexcept { return _capacity; }
  [[nodiscard]] std::uint32_t free_ops() const noexcept { return _free_count; }
  /// @brief Take a free op slot for @p owner (CQEs go to @p handler(owner, token, res, flags)).
  /// Non-urgent callers cannot take the last `urgent_reserve_slots` slots. Background ops (idle
  /// watches, keep-warm, warm-up) are counted apart (`work_ops` excludes them).
  [[nodiscard]] bool take_op(
    void* owner, op_handler handler, priority p, bool background, std::uint32_t& token) noexcept;
  /// @brief Return a slot after its final CQE (deferred while a cancel CQE is outstanding).
  void release_op(std::uint32_t token) noexcept;
  /// @brief The next SQE, tagged `[kind | generation | token]`; marks the slot in flight. Never
  /// null: submits inline (one extra enter, counted) when the SQ is full.
  [[nodiscard]] io_uring_sqe* get_sqe(std::uint32_t token, ud_kind kind) noexcept;
  /// @brief IORING_OP_ASYNC_CANCEL for the slot's in-flight SQE; the slot stays reserved until
  /// the cancel CQE arrives too. No-op when nothing is in flight or a cancel is pending.
  void cancel_op(std::uint32_t token) noexcept;
  [[nodiscard]] std::uint32_t inflight_ops() const noexcept { return _inflight; }
  [[nodiscard]] std::uint32_t work_ops() const noexcept { return _work_slots; }
  [[nodiscard]] std::uint32_t background_ops() const noexcept { return _background_slots; }

  // ---- time and deadlines (R8) ----

  /// @brief Cache CLOCK_MONOTONIC for this pass (start of every pass).
  void update_now() noexcept;
  /// @return CLOCK_MONOTONIC ns cached at the start of the pass.
  [[nodiscard]] std::uint64_t now_ns() const noexcept { return _now; }
  /// @brief Run @p fn(arg) once `now_ns() >= when_ns` (checked every pass; bounds the wait).
  /// @return an invalid handle when the heap is full.
  [[nodiscard]] deadline_handle arm_deadline(std::uint64_t when_ns,
                                             deadline_fn fn,
                                             void* arg) noexcept
  {
    return _deadlines.arm(when_ns, fn, arg);
  }
  bool cancel_deadline(deadline_handle h) noexcept { return _deadlines.cancel(h); }
  /// @return the earliest armed deadline, or UINT64_MAX.
  [[nodiscard]] std::uint64_t next_deadline_ns() const noexcept { return _deadlines.next(); }

  // ---- the pass ----

  /// @brief Dispatch every CQE already in the CQ (manual walk, one `cq_advance` per round, at
  /// most 8 rounds; never enters the kernel: with DEFER_TASKRUN, CQEs are posted inside the
  /// previous pass's enter). @return CQEs reaped.
  std::size_t reap() noexcept;
  /// @brief Run the deadlines due at `now_ns()` (bounded: callbacks that re-arm at or before
  /// `now` run next pass). @return callbacks run.
  std::size_t run_due_deadlines() noexcept;
  /// @brief Non-blocking enter: submit what this pass prepared and run deferred task_work; the
  /// syscall is skipped when there is nothing to submit, no task_work and no CQ overflow.
  void submit() noexcept;
  /// @brief Blocking enter: submit and wait for one CQE or the wait bound (deadline head,
  /// @p bound, safety tick). @p bound == 0 is `submit()`. No Dekker protocol: a wake still
  /// breaks the wait, but `notify` only wakes a parked loop (use `park_and_wait`).
  void submit_and_wait(std::uint64_t bound) noexcept;
  /**
   * @brief The pass's single enter, publishing the parked flag (Dekker, gpuio D2).
   *
   * Sequence: parked = true; fence; `has_work()`. Work visible: withdraw and enter non-blocking.
   * Else wait for one CQE or the wait bound, then withdraw. A producer that published work
   * before the fence is seen by `has_work`; one that publishes after it sees the flag and wakes.
   *
   * @param bound    the caller's external bound (copy ETA), `no_bound`, or 0 (more work is
   *                 runnable now: skip the enter for up to `max_enter_skips` passes)
   * @param has_work noexcept re-check of everything producers signal: intake queues, stop flags
   */
  template <class HasWork>
  void park_and_wait(std::uint64_t bound, HasWork&& has_work) noexcept
  {
    static_assert(std::is_nothrow_invocable_r_v<bool, HasWork&>, "has_work must be noexcept");
    std::uint64_t const t = wait_timeout_ns(bound);
    if (t == 0) {
      skip_or_submit(bound == 0);
      return;
    }
    if (!park_prepare()) {
      submit();
      return;
    }
    if (has_work()) {
      park_cancel();
      submit();
      return;
    }
    enter_blocking(t);
    park_finish();
  }
  /**
   * @brief One loop pass: `update_now`, `reap`, `run_due_deadlines`, @p work, `park_and_wait`.
   * @param work     `std::uint64_t() noexcept`: intake and pump; returns the wait bound
   *                 (`no_bound`, a copy ETA, or 0 when more work is runnable now)
   * @param has_work as for `park_and_wait`
   */
  template <class Work, class HasWork>
  void run_pass(Work&& work, HasWork&& has_work) noexcept
  {
    static_assert(std::is_nothrow_invocable_r_v<std::uint64_t, Work&>, "work must be noexcept");
    update_now();
    reap();
    run_due_deadlines();
    std::uint64_t const bound = work();
    park_and_wait(bound, has_work);
  }
  /// @brief One pass without intake: `update_now`, `reap`, `run_due_deadlines`,
  /// `submit_and_wait(bound)` (tests, drains).
  void run_pass(std::uint64_t bound = no_bound) noexcept;

  // ---- cross-thread (any thread) ----

  /// @brief "Work arrived": wake the loop only if it published its parked flag (one fence and
  /// one relaxed load when it is busy; one syscall otherwise). Call after publishing the work.
  void notify() noexcept;
  /// @brief Ring the doorbell unconditionally (one syscall).
  void wake() noexcept;
  [[nodiscard]] bool uses_sync_msg_ring() const noexcept { return _sync_wake; }
  /// @return doorbell sends that failed (0 in a healthy run).
  [[nodiscard]] std::uint64_t wake_send_errors() const noexcept
  {
    return _wake_errors.load(std::memory_order_relaxed);
  }

  // ---- shutdown (U14) ----

  /// @brief `io_uring_register_sync_cancel(ANY | ALL)`, 1 s: every socket and poll op on the ring
  /// completes with -ECANCELED. Marks the loop closing (the eventfd READ is no longer re-posted).
  /// Owner thread.
  void cancel_all() noexcept;
  /**
   * @brief Reap until no op slot is in flight and @p done() holds, re-cancelling ops issued
   * meanwhile, or until @p timeout_ns passed. @return true when drained.
   */
  template <class Done>
  bool drain(Done&& done, std::uint64_t timeout_ns = 5'000'000'000ULL) noexcept
  {
    static_assert(std::is_nothrow_invocable_r_v<bool, Done&>, "done must be noexcept");
    update_now();
    std::uint64_t const end = _now + timeout_ns;
    for (;;) {
      update_now();
      reap();
      run_due_deadlines();
      if (_inflight == 0 && done()) return true;
      if (_now >= end) return false;
      // Ops issued after the last sync cancel (a callback started one): cancel them too.
      if (_inflight != 0 && _cancel_requested && _stats.ops_submitted != _submitted_at_cancel)
        cancel_all();
      submit_and_wait(_inflight == 0 ? 1'000'000ULL : 10'000'000ULL);
    }
  }
  /// @return true once `cancel_all` ran.
  [[nodiscard]] bool closing() const noexcept { return _cancel_requested; }

  // ---- misc ----

  /// @brief Reactors of the ioctx serving the same endpoints (keep-warm floor share,
  /// `ceil(min_connections / live)`; gpuio T15b). Set by the owner; default 1.
  [[nodiscard]] unsigned live_reactors() const noexcept { return _live_reactors; }
  void set_live_reactors(unsigned n) noexcept { _live_reactors = n == 0 ? 1 : n; }
  /// @brief The owner's stop flag (Env `stopping()`): pools stop starting new work.
  [[nodiscard]] bool stopping() const noexcept { return _stopping; }
  void set_stopping(bool v) noexcept { _stopping = v; }
  /// @return the ring (io-wq registration only).
  [[nodiscard]] io_uring& ring() noexcept { return _ring; }
  [[nodiscard]] uring_loop_stats const& stats() const noexcept { return _stats; }
  [[nodiscard]] uring_loop_config const& config() const noexcept { return _cfg; }
  /// @return true when enters carry IORING_ENTER_NO_IOWAIT (FEAT_NO_IOWAIT, kernel >= 6.15).
  [[nodiscard]] bool no_iowait() const noexcept { return _has_no_iowait; }
  /// @return true when the ring fd is registered (enters use IORING_ENTER_REGISTERED_RING).
  [[nodiscard]] bool ring_fd_registered() const noexcept { return _ring_fd_registered; }

 private:
  struct op_slot {
    op_handler on_cqe{nullptr};
    void* owner{nullptr};
    std::uint32_t index{0};
    std::uint32_t generation{0};  ///< bumped by take_op; low 24 bits travel in user_data
    ud_kind kind{ud_kind::op};    ///< tag of the next SQE prepared by get_sqe
    bool in_flight{false};        ///< an SQE of this slot awaits its final CQE
    bool cancel_pending{false};   ///< an ASYNC_CANCEL CQE is outstanding
    bool release_deferred{false};
    bool background{false};
  };

  [[nodiscard]] io_uring_sqe* get_raw_sqe() noexcept;
  [[nodiscard]] io_uring_sqe* get_sqe_slow() noexcept;
  void release_slot(op_slot& s) noexcept;
  void dispatch_cqe(std::uint64_t ud, int res, unsigned flags) noexcept;
  void on_doorbell() noexcept;
  void on_cancel_cqe(std::uint64_t ud) noexcept;
  void post_wake_read() noexcept;
  void send_wake() noexcept;
  [[nodiscard]] std::uint64_t wait_timeout_ns(std::uint64_t bound) const noexcept;
  void skip_or_submit(bool runnable) noexcept;
  int do_enter(unsigned to_submit, unsigned wait_nr, std::uint64_t timeout_ns) noexcept;
  void enter_blocking(std::uint64_t timeout_ns) noexcept;
  /// @return false when the loop cannot park (not started).
  [[nodiscard]] bool park_prepare() noexcept;
  void park_cancel() noexcept;
  void park_finish() noexcept;

  // ---- hot state ----
  io_uring _ring{};
  std::unique_ptr<op_slot[]> _slots;
  std::unique_ptr<std::uint32_t[]> _free;
  std::uint32_t _free_count{0};
  std::uint32_t _capacity{0};
  std::uint32_t _reserve{0};
  std::uint32_t _inflight{0};
  std::uint32_t _work_slots{0};
  std::uint32_t _background_slots{0};
  std::uint64_t _now{0};
  unsigned _enter_skips{0};
  bool _ring_ok{false};
  bool _ring_fd_registered{false};
  bool _has_no_iowait{false};
  bool _sync_wake{false};
  bool _wake_read_posted{false};
  bool _cancel_requested{false};
  bool _stopping{false};
  std::uint64_t _submitted_at_cancel{0};
  unsigned _live_reactors{1};
  deadline_heap<deadline_fn> _deadlines;
  uring_loop_config _cfg{};
  uring_loop_stats _stats{};
  int _ring_fd{-1};   ///< the real ring fd (MSG_RING target; never the registered index)
  int _event_fd{-1};  ///< eventfd fallback
  // Producer-written state on its own lines: the parked flag (Dekker) and the error counter. The
  // eventfd READ target is written by the kernel.
  alignas(64) std::atomic<bool> _parked{false};
  alignas(64) std::atomic<std::uint64_t> _wake_errors{0};
  alignas(64) std::uint64_t _wake_read_buf{0};
};

/**
 * @brief The pool's Env (gpuio `remote_env.hpp` contract) over a `uring_loop`: one per owner.
 *
 * ```
 *   uint32_t      op_capacity() const;   uint32_t free_ops() const;
 *   bool          take_op(priority, bool background, uint32_t& token);
 *   void          release_op(uint32_t token);                // after the op's final CQE
 *   io_uring_sqe* get_sqe(uint32_t token, ud_kind kind);     // never null; CQE -> owner
 *   void          cancel_op(uint32_t token);                 // ASYNC_CANCEL; release deferred
 *   uint64_t      now_ns() const;                            // monotonic, cached per pass
 *   template <void (*Fn)(void*) noexcept> deadline_handle arm_deadline(uint64_t when, void* arg);
 *   bool          cancel_deadline(deadline_handle);
 *   io_uring&     ring();                                    // io-wq registration only
 *   bool          stopping() const;
 *   template <class Owner> void bind(Owner*);                // CQEs -> Owner::on_op_cqe
 * ```
 * Several envs (pools, the worker's own ops) share one loop; tokens are loop-wide slot indices.
 */
class loop_env {
 public:
  explicit loop_env(uring_loop& l) noexcept : _l(&l) {}

  /// @brief Route every CQE of ops taken through this env to `o->on_op_cqe(token, res, flags)`.
  template <class Owner>
  void bind(Owner* o) noexcept
  {
    _owner   = o;
    _handler = [](void* p, std::uint32_t t, int r, unsigned f) noexcept {
      static_cast<Owner*>(p)->on_op_cqe(t, r, f);
    };
  }

  [[nodiscard]] std::uint32_t op_capacity() const noexcept { return _l->op_capacity(); }
  [[nodiscard]] std::uint32_t free_ops() const noexcept { return _l->free_ops(); }
  [[nodiscard]] bool take_op(priority p, bool background, std::uint32_t& token) noexcept
  {
    return _l->take_op(_owner, _handler, p, background, token);
  }
  void release_op(std::uint32_t token) noexcept { _l->release_op(token); }
  [[nodiscard]] io_uring_sqe* get_sqe(std::uint32_t token, ud_kind kind) noexcept
  {
    return _l->get_sqe(token, kind);
  }
  void cancel_op(std::uint32_t token) noexcept { _l->cancel_op(token); }
  [[nodiscard]] std::uint64_t now_ns() const noexcept { return _l->now_ns(); }
  template <void (*Fn)(void*) noexcept>
  [[nodiscard]] deadline_handle arm_deadline(std::uint64_t when_ns, void* arg) noexcept
  {
    return _l->arm_deadline(when_ns, Fn, arg);
  }
  bool cancel_deadline(deadline_handle h) noexcept { return _l->cancel_deadline(h); }
  [[nodiscard]] io_uring& ring() noexcept { return _l->ring(); }
  [[nodiscard]] bool stopping() const noexcept { return _l->stopping(); }
  /// Optional Env member: the pool's keep-warm share is ceil(min_connections / live reactors).
  [[nodiscard]] unsigned live_reactors() const noexcept { return _l->live_reactors(); }
  [[nodiscard]] uring_loop& loop() const noexcept { return *_l; }

 private:
  uring_loop* _l;
  void* _owner{nullptr};
  op_handler _handler{nullptr};
};

}  // namespace cucascade::io::net
