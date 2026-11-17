/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file chunk_plan.hpp
 * @brief Where a host-staged pipeline puts its bytes in pinned host memory: staging_slot_layout
 * cuts the staging slots out of HOST pool blocks, and chunk_plan cuts a table's byte stream into
 * chunks laid out in those slots.
 *
 * Used by copy_table_to_gpu(). Everything here computes with sizes and host addresses only: it
 * reads and writes no memory behind those addresses and makes no CUDA call.
 */

#include <rmm/aligned.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cucascade {
namespace detail {

/**
 * @brief Source bytes of one leaf buffer, before its destination buffer exists.
 */
struct source_span {
  std::byte const* src;
  std::size_t size;
};

/**
 * @brief Host page size. Measured: a host-to-destination copy runs about a quarter slower when its
 * piece does not start on a host page boundary; the source-to-host copy does not care.
 */
inline constexpr std::size_t host_page_bytes = 4096;

/**
 * @brief Smallest piece that is moved to a page boundary, so that the padding (4KiB) is at most an
 * eighth of the bytes it aligns.
 */
inline constexpr std::size_t aligned_piece_min_bytes = 32 * 1024;

/**
 * @brief Staging bytes that page alignment can add to @p spans staged in one slot of page-aligned
 * runs.
 */
[[nodiscard]] inline std::size_t alignment_padding_bound(
  std::span<source_span const> spans) noexcept
{
  auto const aligned = std::ranges::count_if(
    spans, [](source_span const& span) { return span.size >= aligned_piece_min_bytes; });
  return static_cast<std::size_t>(aligned) * (host_page_bytes - 1);
}

/**
 * @brief Host runs of each staging slot of one host-staged pipeline.
 *
 * The slots are consecutive `slot_bytes` ranges of the blocks taken as one byte sequence, so a slot
 * may span blocks; it is described by its host runs, one per block it touches. Blocks that happen
 * to be adjacent in memory are not joined into one run: the pool pins its memory in separate
 * regions, and a copy that crosses from one region into the next fails.
 */
class staging_slot_layout {
 public:
  /**
   * @brief A layout without slots.
   */
  staging_slot_layout() = default;

  /**
   * @brief Cut @p depth slots of @p slot_bytes out of @p blocks taken as one byte sequence.
   *
   * @param blocks Start of each block; together the blocks must hold `depth * slot_bytes` bytes
   * @param block_size Bytes per block
   * @param slot_bytes Bytes per staging slot
   * @param depth Number of staging slots
   */
  staging_slot_layout(std::span<std::byte* const> blocks,
                      std::size_t block_size,
                      std::size_t slot_bytes,
                      std::size_t depth)
    : _depth{depth}
  {
    std::size_t block = 0;
    std::size_t used  = 0;  // bytes of blocks[block] already handed to slots
    for (std::size_t slot = 0; slot < depth; ++slot) {
      _slot_first_run.push_back(_runs.size());
      for (std::size_t remaining = slot_bytes; remaining > 0;) {
        std::size_t const take = std::min(remaining, block_size - used);
        _runs.emplace_back(blocks[block] + used, take);
        remaining -= take;
        used += take;
        if (used == block_size) {
          ++block;
          used = 0;
        }
      }
    }
    _slot_first_run.push_back(_runs.size());
  }

  [[nodiscard]] std::size_t get_depth() const noexcept { return _depth; }

  /**
   * @brief Contiguous host runs that make up slot @p slot, in byte order.
   */
  [[nodiscard]] std::span<std::span<std::byte> const> get_runs(std::size_t slot) const noexcept
  {
    return std::span{_runs}.subspan(_slot_first_run[slot],
                                    _slot_first_run[slot + 1] - _slot_first_run[slot]);
  }

 private:
  std::size_t _depth{0};
  std::vector<std::span<std::byte>> _runs;
  std::vector<std::size_t> _slot_first_run;  ///< Per slot, plus one past the last slot
};

/**
 * @brief Part of a chunk: `size` bytes of source span `span` from `offset`, staged at `host`.
 */
struct piece {
  std::size_t span;
  std::size_t offset;
  std::byte* host;
  std::size_t size;
};

/**
 * @brief A table's byte stream cut into chunks, each laid out in the staging slot it passes
 * through.
 *
 * Chunk c uses slot c % depth. A chunk takes source bytes in stream order until its slot is full; a
 * piece ends where its source span or its host run ends. Every piece of at least
 * `aligned_piece_min_bytes` starts on a host page boundary, which leaves small gaps in the slot.
 *
 * Every source byte is staged exactly once, the pieces of a chunk do not overlap, and every chunk
 * stages at least one byte.
 */
class chunk_plan {
 public:
  /**
   * @param spans Leaf buffers of the table in byte-stream order, none of them empty
   * @param slots Staging slots to lay the chunks out in; at least one, none of them empty
   */
  chunk_plan(std::span<source_span const> spans, staging_slot_layout const& slots)
  {
    std::size_t span   = 0;
    std::size_t offset = 0;  // bytes of spans[span] already planned
    while (span < spans.size()) {
      std::size_t const slot = _chunk_first_piece.size() % slots.get_depth();
      _chunk_first_piece.push_back(_pieces.size());
      for (std::span<std::byte> run : slots.get_runs(slot)) {
        while (span < spans.size()) {
          std::size_t const remaining = spans[span].size - offset;
          if (remaining >= aligned_piece_min_bytes) { run = skip_to_page_boundary(run); }
          if (run.empty()) { break; }
          std::size_t const take = std::min(remaining, run.size());
          _pieces.push_back({span, offset, run.data(), take});
          run = run.subspan(take);
          offset += take;
          if (offset == spans[span].size) {
            ++span;
            offset = 0;
          }
        }
      }
    }
    _chunk_first_piece.push_back(_pieces.size());
  }

  [[nodiscard]] std::size_t get_chunk_count() const noexcept
  {
    return _chunk_first_piece.size() - 1;
  }

  [[nodiscard]] std::span<piece const> get_pieces(std::size_t chunk) const noexcept
  {
    return std::span{_pieces}.subspan(_chunk_first_piece[chunk],
                                      _chunk_first_piece[chunk + 1] - _chunk_first_piece[chunk]);
  }

 private:
  /**
   * @brief @p run from its first page boundary on, or @p run itself when the boundary is its end or
   * beyond, so that a chunk always stages at least one byte.
   */
  [[nodiscard]] static std::span<std::byte> skip_to_page_boundary(std::span<std::byte> run) noexcept
  {
    auto const address        = reinterpret_cast<std::uintptr_t>(run.data());
    std::size_t const padding = rmm::align_up(address, host_page_bytes) - address;
    return padding < run.size() ? run.subspan(padding) : run;
  }

  std::vector<piece> _pieces;
  std::vector<std::size_t> _chunk_first_piece;  ///< Per chunk, plus one past the last chunk
};

}  // namespace detail
}  // namespace cucascade
