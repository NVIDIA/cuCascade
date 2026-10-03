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

#include <cucascade/io/io_request.hpp>

#include <cuda/stream_ref>

#include <sys/uio.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cucascade::io::uring {

inline constexpr std::size_t min_dynamic_io_size = 256UL << 10;
inline constexpr std::size_t max_dynamic_io_size = 16UL << 20;

/// What one physical io_uring operation does.
enum class uring_op_kind : std::uint8_t {
  read,   ///< file -> host buffer / staging (+ optional H2D copy)
  write,  ///< host buffer / staging (after an optional D2H copy) -> file
  fsync,  ///< fdatasync() of the file (flush, data_sync durability, commit)
};

/**
 * @brief Device source of a staged write operation.
 *
 * The operation's bytes are copied device -> pinned staging on @c stream
 * (ordered after the caller's work on it) before the write SQE is prepared.
 */
struct device_write_source {
  const std::uint8_t* data{nullptr};                 ///< first byte of this operation
  ::cuda::stream_ref stream{cudaStream_t{nullptr}};  ///< caller's stream
  int device_id{-1};                                 ///< device owning @c data
};

/**
 * @brief Backend-specific state for one physical io_uring operation.
 *
 * The generic portion owns the object, coordinator, copy request and cache
 * completion. The uring portion records the selected fd and how many of the
 * engine's pinned blocks must be claimed before the SQE can be published.
 *
 * Writes: @c request.iovecs are the caller's host buffers (zero-copy) or, for
 * a staged write (@c device_src set), rebuilt over the claimed staging blocks
 * at dispatch.  @c request.io_rng is the exact file range written (never
 * padded).  An fsync operation carries no buffers.
 */
struct uring_io_op {
  io_op_request request;
  int fd{-1};
  std::size_t file_size{0};
  std::size_t staging_blocks{0};
  bool use_odirect{false};
  uring_op_kind kind{uring_op_kind::read};
  /// Staged write: the device bytes to copy into the staging blocks first.
  std::optional<device_write_source> device_src;
  /// fsync of a commit: mark the object committed once the sync succeeded.
  bool commit_after{false};
  /// Engine bookkeeping (owning group, per-class resource accounting); released
  /// -- and thereby accounted as terminal -- when the operation is destroyed.
  /// Declared last so it is released before @c request (and its staging lease).
  std::shared_ptr<void> engine_ticket;

  [[nodiscard]] bool needs_staging() const noexcept { return staging_blocks != 0; }
  [[nodiscard]] bool is_vectored() const noexcept { return request.iovecs.size() > 1; }
  [[nodiscard]] bool is_write() const noexcept { return kind == uring_op_kind::write; }
  [[nodiscard]] bool is_fsync() const noexcept { return kind == uring_op_kind::fsync; }
};

namespace detail {

[[nodiscard]] constexpr bool odirect_available(bool enabled, int handle) noexcept
{
  return enabled && handle >= 0;
}

[[nodiscard]] constexpr bool is_odirect_runtime_error(int errc) noexcept
{
  return errc == EINVAL || errc == EOPNOTSUPP;
}

/** Pick the staged operation size from queued pressure and currently free blocks. */
[[nodiscard]] inline std::size_t dynamic_io_target(std::size_t backlog_bytes,
                                                   std::size_t free_slots,
                                                   std::size_t block_size) noexcept
{
  if (block_size == 0 || free_slots == 0) return 0;

  auto const desired = std::clamp(backlog_bytes, min_dynamic_io_size, max_dynamic_io_size);
  auto const desired_blocks =
    std::max<std::size_t>(1, desired / block_size + (desired % block_size != 0));
  auto const max_blocks = std::max<std::size_t>(1, max_dynamic_io_size / block_size);
  auto const blocks     = std::min({desired_blocks, free_slots, max_blocks});
  return std::min(blocks * block_size, max_dynamic_io_size);
}

[[nodiscard]] inline bool is_odirect_compatible(range io_range,
                                                std::span<iovec const> iovecs) noexcept
{
  if (io_range.offset % IO_BLOCK_SIZE != 0 || io_range.size % IO_BLOCK_SIZE != 0) return false;
  if (iovecs.empty()) return false;
  std::size_t total = 0;
  for (auto const& iov : iovecs) {
    if (iov.iov_base == nullptr || iov.iov_len == 0 || iov.iov_len % IO_BLOCK_SIZE != 0 ||
        reinterpret_cast<std::uintptr_t>(iov.iov_base) % IO_BLOCK_SIZE != 0) {
      return false;
    }
    total += iov.iov_len;
  }
  return total == io_range.size;
}

/**
 * @brief The sub-list of @p source covering bytes [@p skip, @p skip + @p length).
 *
 * Used to cut a (coalesced) write run into head / O_DIRECT body / tail and
 * size-capped operations without copying any data.
 */
[[nodiscard]] inline std::vector<iovec> slice_iovecs(std::span<iovec const> source,
                                                     std::size_t skip,
                                                     std::size_t length)
{
  std::vector<iovec> output;
  for (auto const& iov : source) {
    if (length == 0) break;
    if (skip >= iov.iov_len) {
      skip -= iov.iov_len;
      continue;
    }
    auto const bytes = std::min(iov.iov_len - skip, length);
    output.push_back(iovec{static_cast<std::uint8_t*>(iov.iov_base) + skip, bytes});
    length -= bytes;
    skip = 0;
  }
  return output;
}

/** Rebuild an iovec list after @p skip bytes have already completed. */
inline void fill_remaining_iovecs(std::span<iovec const> source,
                                  std::size_t skip,
                                  std::vector<iovec>& output)
{
  output.clear();
  output.reserve(source.size());
  for (auto const& iov : source) {
    if (skip >= iov.iov_len) {
      skip -= iov.iov_len;
      continue;
    }
    output.push_back(iovec{static_cast<std::uint8_t*>(iov.iov_base) + skip, iov.iov_len - skip});
    skip = 0;
  }
}

}  // namespace detail
}  // namespace cucascade::io::uring
