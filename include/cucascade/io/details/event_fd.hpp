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

#include <cucascade/io/types.hpp>  // file_descriptor

#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <system_error>

namespace cucascade::io::detail {

/**
 * @brief Create a non-blocking eventfd (@c EFD_NONBLOCK | @c EFD_CLOEXEC), initial count 0.
 *
 * Used as the cross-thread wakeup of an I/O runner: submitters (and stop
 * callbacks, CUDA host callbacks, ...) @ref signal_event_fd it, the runner
 * waits on it (epoll / an io_uring read SQE / poll) and @ref drain_event_fd
 * resets it.  Counter semantics (not semaphore): one read returns and clears
 * the accumulated count, so any number of signals collapse into one wakeup.
 *
 * @return The owned eventfd.
 * @throws std::system_error if @c eventfd() fails.
 */
[[nodiscard]] inline file_descriptor make_event_fd()
{
  int const fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) { throw std::system_error(errno, std::generic_category(), "io: eventfd failed"); }
  return file_descriptor{fd};
}

/**
 * @brief Add 1 to the counter of eventfd @p fd, waking any waiter.
 *
 * Retries on @c EINTR.  @c EAGAIN (counter saturated) still leaves the fd
 * readable, so it counts as success.
 *
 * @return @c false only when the write failed for another reason (e.g. a
 *         closed / invalid fd).
 */
inline bool signal_event_fd(int fd) noexcept
{
  std::uint64_t const one = 1;
  for (;;) {
    auto const written = ::write(fd, &one, sizeof(one));
    if (written == static_cast<ssize_t>(sizeof(one))) return true;
    if (written < 0 && errno == EINTR) continue;
    return written < 0 && errno == EAGAIN;
  }
}

/**
 * @brief Read (and thereby reset) the counter of non-blocking eventfd @p fd.
 *
 * @return The accumulated count, 0 when the fd was not signalled (or on error).
 */
inline std::uint64_t drain_event_fd(int fd) noexcept
{
  std::uint64_t value = 0;
  for (;;) {
    auto const got = ::read(fd, &value, sizeof(value));
    if (got == static_cast<ssize_t>(sizeof(value))) return value;
    if (got < 0 && errno == EINTR) continue;
    return 0;
  }
}

}  // namespace cucascade::io::detail
