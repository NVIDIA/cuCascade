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

#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cucascade::io::uring {

inline constexpr std::size_t min_dynamic_io_size = 256UL << 10;
inline constexpr std::size_t max_dynamic_io_size = 16UL << 20;

/**
 * @brief Backend-specific state for one physical io_uring operation.
 *
 * The generic portion owns the object, coordinator, copy request and cache
 * completion. The uring portion records the selected fd and how many of the
 * reactor's pinned blocks must be claimed before the SQE can be published.
 *
 * A staged write fills its blocks before the SQE: by a D2H copy from
 * @c request.device_copy, or (host bounce) by a memcpy from @ref host_source.
 */
struct uring_io_op {
  io_op_request request;
  int fd{-1};
  std::size_t file_size{0};
  std::size_t staging_blocks{0};
  bool use_odirect{false};
  /// Host bytes of a staged host write, at @c request.io_rng.offset; null otherwise.
  std::uint8_t const* host_source{nullptr};

  [[nodiscard]] bool needs_staging() const noexcept { return staging_blocks != 0; }
  [[nodiscard]] bool is_vectored() const noexcept { return request.iovecs.size() > 1; }
  [[nodiscard]] bool is_write() const noexcept { return request.kind == io_op_kind::write; }
  [[nodiscard]] bool is_flush() const noexcept { return request.kind == io_op_kind::flush; }
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
 * @brief Split a write range at @p granule boundaries.
 *
 * @return {head, body, tail}: @c head is the part before the first granule
 *         boundary inside @p rng, @c body the granule-aligned middle, @c tail
 *         the part after the last boundary.  Empty pieces have size 0; a range
 *         inside a single granule is all head.  Concatenated in order they are
 *         exactly @p rng.
 */
[[nodiscard]] inline std::array<range, 3> split_write_range(range rng, std::size_t granule) noexcept
{
  auto const begin = rng.offset;
  auto const end   = rng.end();
  if (granule == 0 || rng.empty()) return {rng, range{end, 0}, range{end, 0}};
  auto const first = begin % granule == 0 ? begin : begin - begin % granule + granule;
  auto const last  = end - end % granule;
  if (first >= end || last <= first) { return {rng, range{end, 0}, range{end, 0}}; }
  return {range{begin, first - begin}, range{first, last - first}, range{last, end - last}};
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
