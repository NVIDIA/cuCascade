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

/// @file uring_enter.hpp
/// @brief Private: the reactor's single `io_uring_enter` function (the only place a reactor
/// enters the kernel for submission or waiting) and the SQ flush it builds on.
///
/// Why not liburing's `io_uring_submit_and_wait_min_timeout`: its loop may enter the kernel
/// twice per call (peek, enter, re-enter when no CQE was returned) and `io_uring_peek_batch_cqe`
/// silently calls `io_uring_get_events` when IORING_SQ_TASKRUN is set. Owning the syscall makes
/// "one enter per pass" (gpuio ARCHITECTURE §9.2) exact and measurable: every call of `enter` is
/// one syscall and is counted in `reactor_stats::enters`. Semantics are those of
/// `submit_and_wait_min_timeout` (io_uring_knobs §2.6.2): EXT_ARG with `min_wait_usec` and a
/// relative timeout; NO_IOWAIT per call (§2.6.4); IORING_ENTER_REGISTERED_RING when the ring fd
/// is registered (§2.4).

#pragma once

#include <liburing.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>

namespace cucascade::io::net::uring {

struct enter_args {
  unsigned wait_nr{0};    ///< CQEs to wait for (0 = do not block)
  bool getevents{false};  ///< run deferred task_work / flush overflow without waiting
  bool has_timeout{false};
  std::uint64_t timeout_ns{0};  ///< relative, with has_timeout
  unsigned min_wait_us{0};      ///< IORING_FEAT_MIN_TIMEOUT
  bool no_iowait{false};        ///< IORING_ENTER_NO_IOWAIT (IORING_FEAT_NO_IOWAIT)
  bool registered{false};       ///< ring fd registered: IORING_ENTER_REGISTERED_RING
};

/// @brief Publish locally prepared SQEs to the kernel (non-SQPOLL ring).
/// @return the number of SQEs the kernel has not consumed yet.
inline unsigned flush_sq(io_uring& ring) noexcept
{
  io_uring_sq& sq     = ring.sq;
  unsigned const tail = sq.sqe_tail;
  if (sq.sqe_head != tail) {
    sq.sqe_head = tail;
    std::atomic_ref<unsigned>(*sq.ktail).store(tail, std::memory_order_release);
  }
  return tail - std::atomic_ref<unsigned>(*sq.khead).load(std::memory_order_acquire);
}

/// @return the shared SQ flags word (IORING_SQ_TASKRUN, IORING_SQ_CQ_OVERFLOW).
inline unsigned sq_flags(io_uring const& ring) noexcept
{
  return std::atomic_ref<unsigned>(*ring.sq.kflags).load(std::memory_order_relaxed);
}

/// @brief One io_uring_enter(2). @return submitted count (>= 0) or -errno (-ETIME on timeout,
/// -EINTR on a signal, -EBUSY/-EAGAIN under CQ overflow backpressure: all non-fatal).
inline int enter(io_uring& ring, unsigned to_submit, enter_args const& a) noexcept
{
  unsigned flags = 0;
  io_uring_getevents_arg arg{};
  __kernel_timespec ts{};
  void* argp        = nullptr;
  std::size_t argsz = 0;
  if (a.wait_nr > 0 || a.getevents) flags |= IORING_ENTER_GETEVENTS;
  if (a.wait_nr > 0 && (a.has_timeout || a.min_wait_us != 0)) {
    arg.sigmask       = 0;
    arg.sigmask_sz    = _NSIG / 8;
    arg.min_wait_usec = a.min_wait_us;
    if (a.has_timeout) {
      ts.tv_sec  = static_cast<long long>(a.timeout_ns / 1'000'000'000ULL);
      ts.tv_nsec = static_cast<long long>(a.timeout_ns % 1'000'000'000ULL);
      arg.ts     = reinterpret_cast<std::uint64_t>(&ts);
    }
    flags |= IORING_ENTER_EXT_ARG;
    argp  = &arg;
    argsz = sizeof(arg);
  }
  if (a.no_iowait) flags |= IORING_ENTER_NO_IOWAIT;
  int fd = ring.ring_fd;
  if (a.registered) {
    flags |= IORING_ENTER_REGISTERED_RING;
    fd = ring.enter_ring_fd;
  }
  long const r = ::syscall(__NR_io_uring_enter, fd, to_submit, a.wait_nr, flags, argp, argsz);
  return r < 0 ? -errno : static_cast<int>(r);
}

}  // namespace cucascade::io::net::uring
