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

/**
 * @file test_chunk_plan.cpp
 * @brief Tests of the host-staged pipeline's layout in host memory. They need no GPU: the layout is
 * computed from sizes and addresses, and the memory behind the addresses is never touched.
 */

#include "chunk_plan.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

using cucascade::detail::aligned_piece_min_bytes;
using cucascade::detail::alignment_padding_bound;
using cucascade::detail::chunk_plan;
using cucascade::detail::host_page_bytes;
using cucascade::detail::piece;
using cucascade::detail::source_span;
using cucascade::detail::staging_slot_layout;

constexpr std::size_t kKiB = 1024;
constexpr std::size_t kMiB = 1024 * kKiB;

enum class block_order { ascending, descending, shuffled };

/**
 * @brief Equal blocks of one host allocation, standing in for the HOST pool blocks that back the
 * staging slots.
 */
class block_arena {
 public:
  /**
   * @param block_size Bytes per block
   * @param min_bytes Bytes the blocks must hold together
   * @param order Order in which the blocks are handed out; ascending blocks are adjacent in memory
   * @param misalignment Distance of the first block from a host page boundary
   */
  block_arena(std::size_t block_size,
              std::size_t min_bytes,
              block_order order,
              std::size_t misalignment = 0)
    : _block_size{block_size}
  {
    std::size_t const count = (min_bytes + block_size - 1) / block_size;
    // Whole pages, with one to spare for the misalignment.
    std::size_t const pages = (count * block_size + host_page_bytes - 1) / host_page_bytes + 1;
    _memory.reset(
      static_cast<std::byte*>(std::aligned_alloc(host_page_bytes, pages * host_page_bytes)));
    if (_memory == nullptr) { throw std::bad_alloc{}; }
    for (std::size_t i = 0; i < count; ++i) {
      _blocks.push_back(_memory.get() + misalignment + i * block_size);
    }
    if (order == block_order::descending) { std::ranges::reverse(_blocks); }
    if (order == block_order::shuffled) { std::ranges::shuffle(_blocks, std::mt19937{42}); }
  }

  [[nodiscard]] std::span<std::byte* const> get_blocks() const noexcept { return _blocks; }
  [[nodiscard]] std::size_t get_block_size() const noexcept { return _block_size; }

 private:
  struct free_deleter {
    void operator()(std::byte* p) const noexcept { std::free(p); }
  };

  std::size_t _block_size;
  std::unique_ptr<std::byte, free_deleter> _memory;
  std::vector<std::byte*> _blocks;
};

/**
 * @brief Leaf buffers of the given sizes; their source addresses play no part in the layout.
 */
std::vector<source_span> spans_of(std::span<std::size_t const> sizes)
{
  std::vector<source_span> spans;
  for (std::size_t size : sizes) {
    spans.push_back({nullptr, size});
  }
  return spans;
}

std::size_t total_bytes(std::span<source_span const> spans)
{
  return std::accumulate(
    spans.begin(), spans.end(), std::size_t{0}, [](std::size_t sum, source_span const& span) {
      return sum + span.size;
    });
}

bool on_page_boundary(std::byte const* address)
{
  return reinterpret_cast<std::uintptr_t>(address) % host_page_bytes == 0;
}

/**
 * @brief Require that @p slots are consecutive @p slot_bytes ranges of @p arena's blocks taken as
 * one byte sequence, with every run inside one block.
 */
void require_sound_layout(block_arena const& arena,
                          std::size_t slot_bytes,
                          std::size_t depth,
                          staging_slot_layout const& slots)
{
  std::size_t const block_size = arena.get_block_size();
  REQUIRE(slots.get_depth() == depth);
  std::size_t position = 0;  // bytes of the block sequence that earlier runs cover
  for (std::size_t slot = 0; slot < depth; ++slot) {
    std::size_t slot_total = 0;
    for (std::span<std::byte> const run : slots.get_runs(slot)) {
      std::size_t const in_block = position % block_size;
      REQUIRE_FALSE(run.empty());
      REQUIRE(run.data() == arena.get_blocks()[position / block_size] + in_block);
      REQUIRE(run.size() <= block_size - in_block);
      position += run.size();
      slot_total += run.size();
    }
    REQUIRE(slot_total == slot_bytes);
  }
}

/**
 * @brief Require what a correct copy rests on: @p plan stages every byte of @p spans exactly once
 * and in stream order, every chunk stages something, every piece lies inside one host run of its
 * chunk's slot behind the chunk's earlier pieces, and every piece of aligned_piece_min_bytes or
 * more starts on a host page boundary.
 *
 * @return Bytes each chunk stages
 */
std::vector<std::size_t> require_sound_plan(std::span<source_span const> spans,
                                            staging_slot_layout const& slots,
                                            chunk_plan const& plan)
{
  std::vector<std::size_t> chunk_bytes;
  std::size_t span   = 0;  // next source byte the plan must stage
  std::size_t offset = 0;
  for (std::size_t chunk = 0; chunk < plan.get_chunk_count(); ++chunk) {
    auto const runs   = slots.get_runs(chunk % slots.get_depth());
    auto const pieces = plan.get_pieces(chunk);
    REQUIRE_FALSE(pieces.empty());

    std::size_t run            = 0;
    std::byte const* run_floor = runs.front().data();  // end of the chunk's last piece in runs[run]
    chunk_bytes.push_back(0);
    for (piece const& p : pieces) {
      REQUIRE(p.size > 0);
      REQUIRE(span < spans.size());
      REQUIRE(p.span == span);
      REQUIRE(p.offset == offset);
      REQUIRE(p.size <= spans[span].size - offset);
      offset += p.size;
      if (offset == spans[span].size) {
        ++span;
        offset = 0;
      }

      auto const run_end = [&] { return runs[run].data() + runs[run].size(); };
      while (p.host < runs[run].data() || p.host >= run_end()) {
        ++run;
        REQUIRE(run < runs.size());
        run_floor = runs[run].data();
      }
      REQUIRE(p.host >= run_floor);
      REQUIRE(p.size <= static_cast<std::size_t>(run_end() - p.host));
      run_floor = p.host + p.size;

      if (p.size >= aligned_piece_min_bytes) { REQUIRE(on_page_boundary(p.host)); }
      chunk_bytes.back() += p.size;
    }
  }
  REQUIRE(span == spans.size());
  REQUIRE(offset == 0);
  return chunk_bytes;
}

}  // namespace

TEST_CASE("staging_slot_layout cuts consecutive slots out of the blocks, one run per block",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  // Slots smaller than a block, equal to one, spanning several, and not a multiple of anything.
  auto const [block_size, slot_bytes, depth] =
    GENERATE(table<std::size_t, std::size_t, std::size_t>({{64 * kKiB, 509, 1},
                                                           {64 * kKiB, 4093, 7},
                                                           {64 * kKiB, 64 * kKiB, 4},
                                                           {64 * kKiB, 100 * kKiB + 13, 3},
                                                           {1 * kMiB, 3 * kMiB, 4},
                                                           {1 * kMiB, 5 * kMiB + 1, 2}}));
  auto const order =
    GENERATE(block_order::ascending, block_order::descending, block_order::shuffled);
  CAPTURE(block_size, slot_bytes, depth, static_cast<int>(order));

  block_arena const arena{block_size, slot_bytes * depth, order};
  staging_slot_layout const slots{arena.get_blocks(), block_size, slot_bytes, depth};

  require_sound_layout(arena, slot_bytes, depth, slots);
}

TEST_CASE("staging_slot_layout keeps blocks that are adjacent in memory in separate runs",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  // The pool pins its memory in separate regions; a copy must not cross from one into the next.
  block_arena const arena{64 * kKiB, 192 * kKiB, block_order::ascending};
  staging_slot_layout const slots{arena.get_blocks(), 64 * kKiB, 192 * kKiB, 1};

  auto const runs = slots.get_runs(0);
  REQUIRE(runs.size() == 3);
  for (std::size_t i = 0; i < runs.size(); ++i) {
    CHECK(runs[i].data() == arena.get_blocks()[i]);
    CHECK(runs[i].size() == 64 * kKiB);
  }
}

TEST_CASE("chunk_plan stages every source byte once, inside single host runs",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  // Leaf buffers below, at and above the size that is page aligned, larger than a slot and larger
  // than a block.
  std::vector<std::size_t> const sizes{128,
                                       32767,
                                       32768,
                                       32769,
                                       1,
                                       5 * kKiB,
                                       2 * kMiB,
                                       8,
                                       300 * kKiB,
                                       4096,
                                       40960,
                                       3 * kMiB + 7,
                                       64,
                                       65536,
                                       12345};
  auto const spans = spans_of(sizes);

  // Slots: far below a leaf buffer, barely above the smallest page-aligned piece, unaligned to
  // pages and blocks, several blocks, and the whole table. Depths: 1, a few, and more than the
  // chunks.
  auto const [block_size, slot_bytes, depth] =
    GENERATE(table<std::size_t, std::size_t, std::size_t>({{64 * kKiB, 509, 1},
                                                           {64 * kKiB, 4093, 3},
                                                           {64 * kKiB, 40 * kKiB, 2},
                                                           {1 * kMiB, 40 * kKiB, 64},
                                                           {1 * kMiB, 300 * kKiB + 13, 3},
                                                           {1 * kMiB, 3 * kMiB, 4},
                                                           {1 * kMiB, 8 * kMiB, 1},
                                                           {1 * kMiB, 8 * kMiB, 4}}));
  auto const order =
    GENERATE(block_order::ascending, block_order::descending, block_order::shuffled);
  // Blocks on page boundaries, as the pinned HOST pool provides them, and off them.
  auto const misalignment = GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{2048});
  CAPTURE(block_size, slot_bytes, depth, static_cast<int>(order), misalignment);

  block_arena const arena{block_size, slot_bytes * depth, order, misalignment};
  staging_slot_layout const slots{arena.get_blocks(), block_size, slot_bytes, depth};
  chunk_plan const plan{spans, slots};

  require_sound_plan(spans, slots, plan);
}

TEST_CASE("chunk_plan stages every source byte once for random tables and staging slots",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  std::mt19937_64 random{20261117};
  auto const pick = [&random](std::size_t low, std::size_t high) {
    return std::uniform_int_distribution<std::size_t>{low, high}(random);
  };
  constexpr std::array<std::size_t, 3> kBlockSizes{4 * kKiB, 64 * kKiB, 1 * kMiB};
  constexpr std::array<std::size_t, 3> kMisalignments{0, 0, 1000};
  constexpr std::array<block_order, 3> kOrders{
    block_order::ascending, block_order::descending, block_order::shuffled};

  for (int trial = 0; trial < 300; ++trial) {
    std::vector<std::size_t> sizes(pick(1, 40));
    for (std::size_t& size : sizes) {
      // Mostly small buffers, some around the alignment threshold, a few large ones.
      switch (pick(0, 9)) {
        case 0: size = pick(aligned_piece_min_bytes - 2, aligned_piece_min_bytes + 2); break;
        case 1: size = pick(1, 512 * kKiB); break;
        default: size = pick(1, 16 * kKiB); break;
      }
    }
    auto const spans             = spans_of(sizes);
    std::size_t const block_size = kBlockSizes[pick(0, 2)];
    std::size_t const slot_bytes = pick(1, 2) == 1 ? pick(256, 8 * kKiB) : pick(256, 3 * kMiB);
    std::size_t const depth      = pick(1, 6);
    CAPTURE(trial, sizes, block_size, slot_bytes, depth);

    block_arena const arena{
      block_size, slot_bytes * depth, kOrders[pick(0, 2)], kMisalignments[pick(0, 2)]};
    staging_slot_layout const slots{arena.get_blocks(), block_size, slot_bytes, depth};
    require_sound_layout(arena, slot_bytes, depth, slots);
    chunk_plan const plan{spans, slots};
    require_sound_plan(spans, slots, plan);
  }
}

TEST_CASE("chunk_plan fills each staging slot before it starts the next chunk",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  // Leaf buffers below the page-aligned size leave no gaps, so every chunk but the last is full.
  std::vector<std::size_t> const sizes{1000, 31 * kKiB, 7, 20 * kKiB, 4096, 30 * kKiB, 12 * kKiB};
  auto const spans        = spans_of(sizes);
  std::size_t const total = total_bytes(spans);

  auto const [slot_bytes, depth] =
    GENERATE(table<std::size_t, std::size_t>({{509, 1}, {4093, 3}, {10 * kKiB, 2}}));
  CAPTURE(slot_bytes, depth);

  block_arena const arena{64 * kKiB, slot_bytes * depth, block_order::shuffled};
  staging_slot_layout const slots{arena.get_blocks(), 64 * kKiB, slot_bytes, depth};
  chunk_plan const plan{spans, slots};

  auto const chunk_bytes = require_sound_plan(spans, slots, plan);
  REQUIRE(chunk_bytes.size() == (total + slot_bytes - 1) / slot_bytes);
  for (std::size_t chunk = 0; chunk + 1 < chunk_bytes.size(); ++chunk) {
    CHECK(chunk_bytes[chunk] == slot_bytes);
  }
}

TEST_CASE("chunk_plan keeps a table in one chunk when its slot has room for the padding",
          "[gpu-to-gpu-converter][chunk-plan]")
{
  // The pipeline sizes the slot of a table smaller than a chunk as its bytes plus
  // alignment_padding_bound(); on page-aligned blocks that must hold the table in a single chunk.
  std::mt19937_64 random{7};
  auto const pick = [&random](std::size_t low, std::size_t high) {
    return std::uniform_int_distribution<std::size_t>{low, high}(random);
  };
  constexpr std::array<std::size_t, 2> kBlockSizes{64 * kKiB, 1 * kMiB};

  for (int trial = 0; trial < 200; ++trial) {
    std::vector<std::size_t> sizes(pick(1, 60));
    for (std::size_t& size : sizes) {
      size = pick(0, 3) == 0 ? pick(aligned_piece_min_bytes - 1, 600 * kKiB) : pick(1, 40 * kKiB);
    }
    auto const spans             = spans_of(sizes);
    std::size_t const block_size = kBlockSizes[pick(0, 1)];
    std::size_t const slot_bytes = total_bytes(spans) + alignment_padding_bound(spans);
    CAPTURE(trial, sizes, block_size);

    block_arena const arena{block_size, slot_bytes, block_order::shuffled};
    staging_slot_layout const slots{arena.get_blocks(), block_size, slot_bytes, 1};
    chunk_plan const plan{spans, slots};

    require_sound_plan(spans, slots, plan);
    REQUIRE(plan.get_chunk_count() == 1);
  }
}
