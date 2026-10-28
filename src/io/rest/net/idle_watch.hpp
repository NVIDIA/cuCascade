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

/// @file idle_watch.hpp
/// @brief Idle-connection liveness watch (gpuio ARCHITECTURE §4.5 mechanism 1): one background
/// `IORING_OP_POLL_ADD(POLLIN | POLLRDHUP)` per idle period, tagged with the slot's generation,
/// NEVER cancelled. Its CQE on a still-idle slot with a matching generation means the peer sent a
/// FIN (or an RST, or a close_notify / ticket record is readable): the pool closes the slot before
/// it can be checked out (or, for a readable kTLS socket, drains the control record first). When
/// the slot is reused the watch fires on the response's first byte and is dropped because the
/// generation no longer matches. Cost: 1 SQE + 1 CQE per exchange, no cancel, no extra pass.
///
/// Why not a receive: a RECV armed on an idle kTLS socket would consume the first response record
/// into the wrong buffer; a poll consumes nothing (curl's "readable byte on an idle HTTP/1.1
/// connection = dead", conncache.c:1066, detected by event instead of by poll(2) at checkout).

#pragma once

#include <liburing.h>
#include <poll.h>

#include <cstdint>

namespace cucascade::io::net {

/// Events the idle watch waits for.
inline constexpr unsigned idle_watch_events = POLLIN | POLLRDHUP;

/// @brief Prepare the idle watch on @p fd (the caller tags user_data with the generation).
inline void prep_idle_watch(io_uring_sqe* sqe, int fd) noexcept
{
  io_uring_prep_poll_add(sqe, fd, idle_watch_events);
}

/// What an idle-watch CQE says about the connection.
enum class watch_verdict : std::uint8_t {
  fin,       ///< POLLRDHUP / POLLHUP / POLLERR: peer closed or reset; close the slot
  readable,  ///< POLLIN only: a record is queued (kTLS: ticket / close_notify: drain it);
             ///< plain HTTP: unsolicited bytes on an idle connection = dead
  failed,    ///< res < 0 (-ECANCELED at shutdown, -EBADF after a local close)
};

/// @brief Classify an idle-watch CQE result (`res` is the poll mask or -errno).
[[nodiscard]] constexpr watch_verdict classify_idle_watch(int res) noexcept
{
  if (res < 0) return watch_verdict::failed;
  auto const ev = static_cast<unsigned>(res);
  if ((ev & (POLLRDHUP | POLLHUP | POLLERR)) != 0) return watch_verdict::fin;
  return watch_verdict::readable;
}

}  // namespace cucascade::io::net
