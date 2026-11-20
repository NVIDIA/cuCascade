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

#include "utils/allocation_counter.hpp"

#include <cucascade/cuda/stream.hpp>
#include <cucascade/io/cufile/fused_read_plan.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using cucascade::io::device_buffer;
using cucascade::io::fused_part;
using cucascade::io::host_buffer;
using cucascade::io::IO_BLOCK_SIZE;
using cucascade::io::io_op_kind;
using cucascade::io::io_priority;
using cucascade::io::prepared_io_completion;
using cucascade::io::prepared_io_slice;
using cucascade::io::range;
using cucascade::io::cufile::fused_plan_options;
using cucascade::io::cufile::fused_plan_stats;
using cucascade::io::cufile::is_lane_read;
using cucascade::io::cufile::lane_window;
using cucascade::io::cufile::plan_fused_reads;

namespace {

constexpr std::size_t KiB = 1UL << 10;
constexpr std::size_t MiB = 1UL << 20;

// Nothing is dereferenced: destinations, streams and gates are fake non-null addresses.
[[nodiscard]] std::uint8_t* address(std::uintptr_t value)
{
  return reinterpret_cast<std::uint8_t*>(value);
}

[[nodiscard]] std::shared_ptr<cucascade::cuda::cuda_event> fake_gate(std::uintptr_t value)
{
  if (value == 0) return nullptr;
  // Aliasing an empty owner: a distinct non-null pointer that is never deleted.
  return {std::shared_ptr<void>{}, reinterpret_cast<cucascade::cuda::cuda_event*>(value)};
}

struct key {
  std::uintptr_t stream{0x100};
  std::uintptr_t gate{0};
  int device{0};
  io_priority priority{io_priority::high};
};

/// The destination of input slice @p index: its own 4 GiB region, so a destination address
/// names the slice and the byte inside it.
[[nodiscard]] std::uintptr_t base_of(std::size_t index)
{
  return static_cast<std::uintptr_t>(index + 1) << 32;
}

[[nodiscard]] prepared_io_slice device_slice(std::size_t offset,
                                             std::size_t size,
                                             std::uintptr_t dst,
                                             key const& k = {})
{
  prepared_io_slice slice{
    range{offset, size},
    device_buffer{address(dst), ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(k.stream)},
                  k.device}};
  slice.d_buffer.ready_gate     = fake_gate(k.gate);
  slice.d_buffer.ready_gate_set = true;
  slice.priority                = k.priority;
  return slice;
}

/// Device slices at @p offsets/@p sizes, slice i writing to @ref base_of(i).
[[nodiscard]] std::vector<prepared_io_slice> device_slices(
  std::vector<std::pair<std::size_t, std::size_t>> const& ranges, key const& k = {})
{
  std::vector<prepared_io_slice> out;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    out.push_back(device_slice(ranges[i].first, ranges[i].second, base_of(i), k));
  }
  return out;
}

[[nodiscard]] fused_plan_options window_of(std::size_t bytes, std::size_t gap = 0)
{
  fused_plan_options options;
  options.window_bytes = bytes;
  options.gap_bytes    = gap;
  return options;
}

/// One destination piece of an emitted slice.
struct piece {
  range rng;
  std::uint8_t* data;
};

[[nodiscard]] std::vector<piece> pieces_of(prepared_io_slice const& slice)
{
  std::vector<piece> out;
  if (slice.is_fused()) {
    for (auto const& part : slice.fused) {
      out.push_back(piece{part.rng, part.data});
    }
  } else {
    out.push_back(piece{slice.rng, slice.d_buffer.data});
  }
  return out;
}

[[nodiscard]] bool same_range(range a, range b) { return a.offset == b.offset && a.size == b.size; }

void check_part(fused_part const& part, std::size_t offset, std::size_t size, std::uintptr_t dst)
{
  CHECK(part.rng.offset == offset);
  CHECK(part.rng.size == size);
  CHECK(part.data == address(dst));
}

}  // namespace

TEST_CASE("fused plan packs contiguous 1/2/5 MiB chunks into 8 MiB windows",
          "[io][cufile][fused_plan]")
{
  // 1, 2, 5, 1, 2, 5, 1, 2 MiB back to back: two exact 8 MiB windows and a 3 MiB one.
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    auto const size = std::array<std::size_t, 3>{1 * MiB, 2 * MiB, 5 * MiB}[i % 3];
    ranges.emplace_back(offset, size);
    offset += size;
  }
  auto slices      = device_slices(ranges);
  auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
  REQUIRE(slices.size() == 3);
  CHECK(stats.reads == 3);
  CHECK(stats.parts == 8);
  CHECK(stats.split_slices == 0);
  CHECK(stats.tail_slices == 0);
  CHECK(stats.payload_bytes == 19 * MiB);
  CHECK(stats.issued_bytes == 19 * MiB);

  CHECK(same_range(slices[0].rng, range{0, 8 * MiB}));
  CHECK(same_range(slices[1].rng, range{8 * MiB, 8 * MiB}));
  CHECK(same_range(slices[2].rng, range{16 * MiB, 3 * MiB}));
  REQUIRE(slices[0].fused.size() == 3);
  check_part(slices[0].fused[0], 0, 1 * MiB, base_of(0));
  check_part(slices[0].fused[1], 1 * MiB, 2 * MiB, base_of(1));
  check_part(slices[0].fused[2], 3 * MiB, 5 * MiB, base_of(2));
  REQUIRE(slices[2].fused.size() == 2);
  check_part(slices[2].fused[1], 17 * MiB, 2 * MiB, base_of(7));
  for (auto const& slice : slices) {
    CHECK(slice.d_buffer.data == slice.fused.front().data);
    CHECK(slice.d_buffer.ready_gate_set);
    CHECK(slice.priority == io_priority::high);
    CHECK(slice.kind == io_op_kind::read);
    CHECK(is_lane_read(slice, 64 * MiB, 8 * MiB));
  }
}

TEST_CASE("fused plan splits a chunk at the window and fuses its rest with what follows",
          "[io][cufile][fused_plan]")
{
  SECTION("5 + 5 MiB")
  {
    auto slices      = device_slices({{0, 5 * MiB}, {5 * MiB, 5 * MiB}});
    auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
    REQUIRE(slices.size() == 2);
    CHECK(stats.reads == 2);
    CHECK(stats.parts == 3);
    CHECK(stats.split_slices == 1);
    REQUIRE(slices[0].fused.size() == 2);
    check_part(slices[0].fused[1], 5 * MiB, 3 * MiB, base_of(1));
    // The rest alone in its window: a plain slice of just that piece.
    CHECK_FALSE(slices[1].is_fused());
    CHECK(same_range(slices[1].rng, range{8 * MiB, 2 * MiB}));
    CHECK(slices[1].d_buffer.data == address(base_of(1) + 3 * MiB));
  }
  SECTION("a slice longer than the window")
  {
    auto slices      = device_slices({{0, 20 * MiB}, {20 * MiB, 1 * MiB}});
    auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
    REQUIRE(slices.size() == 3);
    CHECK(stats.reads == 3);
    CHECK(stats.parts == 4);
    CHECK(stats.split_slices == 1);
    CHECK(stats.payload_bytes == 21 * MiB);
    CHECK(same_range(slices[0].rng, range{0, 8 * MiB}));
    CHECK(slices[0].d_buffer.data == address(base_of(0)));
    CHECK(same_range(slices[1].rng, range{8 * MiB, 8 * MiB}));
    CHECK(slices[1].d_buffer.data == address(base_of(0) + 8 * MiB));
    REQUIRE(slices[2].fused.size() == 2);
    check_part(slices[2].fused[0], 16 * MiB, 4 * MiB, base_of(0) + 16 * MiB);
    check_part(slices[2].fused[1], 20 * MiB, 1 * MiB, base_of(1));
  }
}

TEST_CASE("fused plan bridges gaps up to gap_bytes only", "[io][cufile][fused_plan]")
{
  auto const ranges = std::vector<std::pair<std::size_t, std::size_t>>{
    {0, 1 * MiB}, {1 * MiB + 4 * KiB, 1 * MiB - 4 * KiB}};
  SECTION("gap 0: contiguous only")
  {
    auto slices      = device_slices(ranges);
    auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
    REQUIRE(slices.size() == 2);
    CHECK(stats.reads == 2);
    // Each window holds one whole input slice: the slice itself.
    CHECK_FALSE(slices[0].is_fused());
    CHECK_FALSE(slices[1].is_fused());
    CHECK(slices[1].d_buffer.data == address(base_of(1)));
  }
  SECTION("gap one byte short")
  {
    auto slices = device_slices(ranges);
    CHECK(plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 4 * KiB - 1)).reads == 2);
  }
  SECTION("gap bridged")
  {
    auto slices      = device_slices(ranges);
    auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 4 * KiB));
    REQUIRE(slices.size() == 1);
    CHECK(stats.reads == 1);
    CHECK(stats.parts == 2);
    CHECK(stats.payload_bytes == 2 * MiB - 4 * KiB);
    CHECK(stats.issued_bytes == 2 * MiB);
    CHECK(same_range(slices[0].rng, range{0, 2 * MiB}));
    CHECK(slices[0].payload_bytes() == 2 * MiB - 4 * KiB);
  }
  SECTION("an unaligned gap is bridged by bytes, not blocks")
  {
    auto slices = device_slices({{100, 1000}, {1100 + 3000, 500}});
    CHECK(plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 2999)).reads == 2);
    slices = device_slices({{100, 1000}, {1100 + 3000, 500}});
    CHECK(plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 3000)).reads == 1);
    CHECK(same_range(slices[0].rng, range{100, 4500}));
  }
}

TEST_CASE("fused plan keeps unaligned windows within the window size",
          "[io][cufile][fused_plan]")
{
  // Starts 100 bytes into a block: the window is the aligned span, never wider than 8 MiB.
  auto slices      = device_slices({{100, 8 * MiB}, {8 * MiB + 100, 300}});
  auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
  REQUIRE(slices.size() == 2);
  CHECK(stats.split_slices == 1);
  CHECK(same_range(slices[0].rng, range{100, 8 * MiB - 100}));
  CHECK(same_range(lane_window(slices[0]), range{0, 8 * MiB}));
  REQUIRE(slices[1].fused.size() == 2);
  check_part(slices[1].fused[0], 8 * MiB, 100, base_of(0) + 8 * MiB - 100);
  check_part(slices[1].fused[1], 8 * MiB + 100, 300, base_of(1));
  CHECK(same_range(lane_window(slices[1]), range{8 * MiB, 4 * KiB}));
  CHECK(stats.issued_bytes == 8 * MiB + 4 * KiB);
  CHECK(stats.payload_bytes == 8 * MiB + 300);
  for (auto const& slice : slices) {
    CHECK(is_lane_read(slice, 64 * MiB, 8 * MiB));
  }
}

TEST_CASE("fused plan never puts overlapping or duplicate ranges in one window",
          "[io][cufile][fused_plan]")
{
  auto slices      = device_slices({{0, 1 * MiB}, {0, 1 * MiB}, {512 * KiB, 1 * MiB}});
  auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 64 * KiB));
  REQUIRE(slices.size() == 3);
  CHECK(stats.reads == 3);
  for (std::size_t i = 0; i < slices.size(); ++i) {
    CHECK_FALSE(slices[i].is_fused());
    CHECK(slices[i].d_buffer.data == address(base_of(i)));  // sorted by offset, then input order
  }
}

TEST_CASE("fused plan with fuse off only splits", "[io][cufile][fused_plan]")
{
  auto options = window_of(8 * MiB);
  options.fuse = false;
  auto slices  = device_slices({{0, 1 * MiB}, {1 * MiB, 2 * MiB}, {3 * MiB, 20 * MiB}});
  auto const stats = plan_fused_reads(slices, 64 * MiB, options);
  REQUIRE(slices.size() == 5);
  CHECK(stats.reads == 5);
  CHECK(stats.parts == 5);
  CHECK(stats.split_slices == 1);
  for (auto const& slice : slices) {
    CHECK_FALSE(slice.is_fused());
    CHECK(is_lane_read(slice, 64 * MiB, 8 * MiB));
  }
  // A window starts at its first piece's block, so the long slice splits at 11 and 19 MiB.
  CHECK(same_range(slices[2].rng, range{3 * MiB, 8 * MiB}));
  CHECK(same_range(slices[3].rng, range{11 * MiB, 8 * MiB}));
  CHECK(same_range(slices[4].rng, range{19 * MiB, 4 * MiB}));
}

TEST_CASE("fused plan peels the sub-block EOF tail", "[io][cufile][fused_plan]")
{
  auto const file_size = 10 * MiB + 100;  // last whole block ends at 10 MiB
  SECTION("a slice reaching EOF")
  {
    auto slices      = device_slices({{9 * MiB, 1 * MiB + 100}});
    auto const stats = plan_fused_reads(slices, file_size, window_of(8 * MiB));
    REQUIRE(slices.size() == 2);
    CHECK(stats.reads == 1);
    CHECK(stats.tail_slices == 1);
    CHECK(stats.split_slices == 1);
    CHECK(stats.payload_bytes == 1 * MiB);
    CHECK(same_range(slices[0].rng, range{9 * MiB, 1 * MiB}));
    CHECK(is_lane_read(slices[0], file_size, 8 * MiB));
    CHECK(same_range(slices[1].rng, range{10 * MiB, 100}));
    CHECK(slices[1].d_buffer.data == address(base_of(0) + 1 * MiB));
    CHECK_FALSE(is_lane_read(slices[1], file_size, 8 * MiB));
  }
  SECTION("a slice past EOF is clamped")
  {
    auto slices      = device_slices({{9 * MiB, 2 * MiB}});
    auto const stats = plan_fused_reads(slices, file_size, window_of(8 * MiB));
    REQUIRE(slices.size() == 2);
    CHECK(stats.tail_slices == 1);
    CHECK(same_range(slices[1].rng, range{10 * MiB, 100}));
  }
  SECTION("a slice inside the tail block is moved as is")
  {
    auto slices      = device_slices({{10 * MiB + 10, 50}});
    auto const stats = plan_fused_reads(slices, file_size, window_of(8 * MiB));
    REQUIRE(slices.size() == 1);
    CHECK(stats.reads == 0);
    CHECK(stats.tail_slices == 1);
    CHECK(same_range(slices[0].rng, range{10 * MiB + 10, 50}));
  }
  SECTION("a slice starting at or after EOF passes through")
  {
    auto slices = device_slices({{file_size, 10}, {file_size + 4 * KiB, 4 * KiB}});
    CHECK(plan_fused_reads(slices, file_size, window_of(8 * MiB)) == fused_plan_stats{});
    REQUIRE(slices.size() == 2);
    CHECK(slices[0].d_buffer.data == address(base_of(0)));
    CHECK(slices[1].d_buffer.data == address(base_of(1)));
  }
  SECTION("windows stop at the last whole block")
  {
    auto slices = device_slices({{0, 6 * MiB}, {6 * MiB, 6 * MiB}});
    plan_fused_reads(slices, file_size, window_of(8 * MiB));
    REQUIRE(slices.size() == 3);
    CHECK(same_range(slices[0].rng, range{0, 8 * MiB}));
    CHECK(same_range(slices[1].rng, range{8 * MiB, 2 * MiB}));  // limit = 10 MiB, not 16
    CHECK(same_range(slices[2].rng, range{10 * MiB, 100}));
    for (auto const& slice : slices) {
      CHECK(lane_window(slice).end() <= 10 * MiB + (slice.rng.offset >= 10 * MiB ? 4 * KiB : 0));
    }
  }
  SECTION("a file smaller than one block is all tail")
  {
    auto slices      = device_slices({{0, 100}, {100, 200}});
    auto const stats = plan_fused_reads(slices, 300, window_of(8 * MiB));
    CHECK(stats.reads == 0);
    CHECK(stats.tail_slices == 2);
    REQUIRE(slices.size() == 2);
  }
}

TEST_CASE("fused plan never fuses across stream, gate, device or priority",
          "[io][cufile][fused_plan]")
{
  key const base{};
  auto other_stream   = base;
  other_stream.stream = 0x200;
  auto other_gate     = base;
  other_gate.gate     = 0x300;
  auto other_device   = base;
  other_device.device = 1;
  auto other_priority = base;
  other_priority.priority = io_priority::low;

  for (auto const& other : {other_stream, other_gate, other_device, other_priority}) {
    std::vector<prepared_io_slice> slices;
    slices.push_back(device_slice(0, 1 * MiB, base_of(0), base));
    slices.push_back(device_slice(1 * MiB, 1 * MiB, base_of(1), other));
    slices.push_back(device_slice(2 * MiB, 1 * MiB, base_of(2), base));
    auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB, 1 * MiB));
    REQUIRE(slices.size() == 2);
    CHECK(stats.reads == 2);
    // The two of the same key fuse across the gap the other left.
    auto const fused = std::find_if(
      slices.begin(), slices.end(), [](auto const& slice) { return slice.is_fused(); });
    REQUIRE(fused != slices.end());
    REQUIRE(fused->fused.size() == 2);
    CHECK(fused->fused[1].data == address(base_of(2)));
  }
  // The same gate pointer does fuse.
  auto gated   = base;
  gated.gate   = 0x300;
  auto slices  = device_slices({{0, 1 * MiB}, {1 * MiB, 1 * MiB}}, gated);
  CHECK(plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB)).reads == 1);
  REQUIRE(slices.size() == 1);
  CHECK(slices[0].d_buffer.ready_gate.get() ==
        reinterpret_cast<cucascade::cuda::cuda_event*>(0x300));
}

TEST_CASE("fused plan passes ineligible slices through in their order",
          "[io][cufile][fused_plan]")
{
  std::vector<prepared_io_slice> slices;
  slices.push_back(device_slice(0, 1 * MiB, base_of(0)));                    // planned
  slices.emplace_back(range{1 * MiB, 1 * MiB}, host_buffer{address(0x1000)});  // host only
  slices.push_back(device_slice(2 * MiB, 1 * MiB, base_of(2)));
  slices.back().h_buffer = host_buffer{address(0x2000)};  // host + device
  slices.push_back(device_slice(3 * MiB, 1 * MiB, base_of(3)));
  auto const completion = std::make_shared<prepared_io_completion>(
    [](std::span<cucascade::io::cache::cached_chunk* const>, bool) noexcept {});
  slices.back().on_complete = completion;
  slices.push_back(device_slice(4 * MiB, 1 * MiB, base_of(4)));
  slices.back().kind = io_op_kind::write;
  slices.push_back(device_slice(5 * MiB, 2 * MiB, base_of(5)));
  slices.back().fused = {fused_part{range{5 * MiB, 1 * MiB}, address(base_of(5))},
                         fused_part{range{6 * MiB, 1 * MiB}, address(base_of(6))}};
  slices.push_back(device_slice(7 * MiB, 1 * MiB, base_of(7)));
  slices.back().d_buffer.ready_gate_set = false;  // gate not resolved
  slices.push_back(device_slice(8 * MiB, 0, base_of(8)));  // empty
  slices.push_back(device_slice(8 * MiB, 1 * MiB, base_of(9)));  // planned

  auto const stats = plan_fused_reads(slices, 64 * MiB, window_of(16 * MiB, 7 * MiB));
  CHECK(stats.reads == 1);
  CHECK(stats.parts == 2);
  REQUIRE(slices.size() == 8);
  CHECK(slices[0].h_buffer.is_contiguous());
  CHECK(same_range(slices[0].rng, range{1 * MiB, 1 * MiB}));
  CHECK(slices[1].d_buffer.data == address(base_of(2)));
  CHECK(slices[1].h_buffer.is_contiguous());
  CHECK(slices[2].on_complete == completion);
  CHECK(slices[3].kind == io_op_kind::write);
  CHECK(slices[4].fused.size() == 2);
  CHECK_FALSE(slices[5].d_buffer.ready_gate_set);
  CHECK(slices[6].rng.size == 0);
  REQUIRE(slices[7].fused.size() == 2);
  CHECK(same_range(slices[7].rng, range{0, 9 * MiB}));  // the 7 MiB gap is bridged
}

TEST_CASE("fused plan keeps windows below min_read_bytes as ordinary slices",
          "[io][cufile][fused_plan]")
{
  auto options           = window_of(8 * MiB);
  options.min_read_bytes = 1 * MiB;
  SECTION("small neighbours")
  {
    auto slices      = device_slices({{0, 64 * KiB}, {64 * KiB, 64 * KiB}});
    auto const stats = plan_fused_reads(slices, 64 * MiB, options);
    CHECK(stats == fused_plan_stats{});
    REQUIRE(slices.size() == 2);
    CHECK_FALSE(slices[0].is_fused());
    CHECK(same_range(slices[1].rng, range{64 * KiB, 64 * KiB}));
    CHECK(slices[1].d_buffer.data == address(base_of(1)));
  }
  SECTION("the small rest of a split slice")
  {
    auto slices      = device_slices({{0, 8 * MiB + 64 * KiB}});
    auto const stats = plan_fused_reads(slices, 64 * MiB, options);
    CHECK(stats.reads == 1);
    CHECK(stats.split_slices == 1);
    CHECK(stats.payload_bytes == 8 * MiB);
    REQUIRE(slices.size() == 2);
    CHECK(same_range(slices[1].rng, range{8 * MiB, 64 * KiB}));
    CHECK(slices[1].d_buffer.data == address(base_of(0) + 8 * MiB));
  }
  SECTION("a window reaching the minimum is a read")
  {
    auto slices = device_slices({{0, 512 * KiB}, {512 * KiB, 512 * KiB}});
    CHECK(plan_fused_reads(slices, 64 * MiB, options).reads == 1);
  }
}

TEST_CASE("fused plan validates its options and leaves the slices alone",
          "[io][cufile][fused_plan]")
{
  auto slices      = device_slices({{0, 1 * MiB}, {1 * MiB, 1 * MiB}});
  auto const check = [&](fused_plan_options const& options) {
    CHECK_THROWS_AS(plan_fused_reads(slices, 64 * MiB, options), std::invalid_argument);
    REQUIRE(slices.size() == 2);
    CHECK(slices[1].d_buffer.data == address(base_of(1)));
  };
  check(window_of(0));
  check(window_of(IO_BLOCK_SIZE + 1));
  check(window_of(8 * MiB, 8 * MiB));
  check(window_of(8 * MiB, 9 * MiB));
  CHECK_NOTHROW(plan_fused_reads(slices, 64 * MiB, window_of(IO_BLOCK_SIZE)));
  // Nothing eligible: nothing touched.
  std::vector<prepared_io_slice> none;
  CHECK(plan_fused_reads(none, 64 * MiB, window_of(8 * MiB)) == fused_plan_stats{});
}

TEST_CASE("fused plan leaves the slices unchanged when an allocation fails",
          "[io][cufile][fused_plan]")
{
  auto const make = [] {
    auto slices = device_slices({{0, 20 * MiB}, {20 * MiB, 1 * MiB}, {21 * MiB, 3 * MiB}});
    slices.emplace_back(range{0, 4 * KiB}, host_buffer{address(0x1000)});
    return slices;
  };
  auto const reference = make();
  bool succeeded       = false;
  for (std::size_t n = 1; n < 64 && !succeeded; ++n) {
    auto slices = make();
    cucascade::test::scoped_allocation_counter counter;
    counter.fail_at(n);
    try {
      std::ignore = plan_fused_reads(slices, 64 * MiB, window_of(8 * MiB));
      succeeded   = true;
    } catch (std::bad_alloc const&) {
      REQUIRE(slices.size() == reference.size());
      for (std::size_t i = 0; i < slices.size(); ++i) {
        CHECK(same_range(slices[i].rng, reference[i].rng));
        CHECK(slices[i].d_buffer.data == reference[i].d_buffer.data);
        CHECK(slices[i].h_buffer.is_contiguous() == reference[i].h_buffer.is_contiguous());
        CHECK_FALSE(slices[i].is_fused());
      }
    }
  }
  CHECK(succeeded);
}

TEST_CASE("lane_window and is_lane_read edge cases", "[io][cufile][fused_plan]")
{
  auto const file_size = 64 * MiB;
  auto slice           = device_slice(100, 200, base_of(0));
  CHECK(same_range(lane_window(slice), range{0, 4 * KiB}));
  CHECK(is_lane_read(slice, file_size, 4 * KiB));
  CHECK_FALSE(is_lane_read(slice, file_size, 0));

  slice = device_slice(4 * KiB, 4 * KiB, base_of(0));
  CHECK(same_range(lane_window(slice), range{4 * KiB, 4 * KiB}));
  slice = device_slice(4 * KiB - 1, 2, base_of(0));
  CHECK(same_range(lane_window(slice), range{0, 8 * KiB}));
  CHECK_FALSE(is_lane_read(slice, file_size, 4 * KiB));
  CHECK(is_lane_read(slice, file_size, 8 * KiB));

  // The window must end inside the file's whole blocks.
  slice = device_slice(0, 4 * KiB + 1, base_of(0));
  CHECK(is_lane_read(slice, 8 * KiB, 8 * MiB));
  CHECK_FALSE(is_lane_read(slice, 8 * KiB - 1, 8 * MiB));
  CHECK_FALSE(is_lane_read(slice, 0, 8 * MiB));

  auto const lane = [&] { return device_slice(0, 1 * MiB, base_of(0)); };
  slice           = lane();
  slice.rng.size  = 0;
  CHECK_FALSE(is_lane_read(slice, file_size, 8 * MiB));
  slice          = lane();
  slice.h_buffer = host_buffer{address(0x1000)};
  CHECK_FALSE(is_lane_read(slice, file_size, 8 * MiB));
  slice      = lane();
  slice.kind = io_op_kind::write;
  CHECK_FALSE(is_lane_read(slice, file_size, 8 * MiB));
  slice             = lane();
  slice.on_complete = std::make_shared<prepared_io_completion>(
    [](std::span<cucascade::io::cache::cached_chunk* const>, bool) noexcept {});
  CHECK_FALSE(is_lane_read(slice, file_size, 8 * MiB));
  prepared_io_slice host_only{range{0, 4 * KiB}, host_buffer{address(0x1000)}};
  CHECK_FALSE(is_lane_read(host_only, file_size, 8 * MiB));

  // Saturating end: no overflow, not a lane read.
  slice = device_slice(std::numeric_limits<std::size_t>::max() - 10, 5, base_of(0));
  CHECK(lane_window(slice).offset ==
        (std::numeric_limits<std::size_t>::max() & ~(IO_BLOCK_SIZE - 1)));
  CHECK_FALSE(is_lane_read(slice, std::numeric_limits<std::size_t>::max(), 8 * MiB));

  // Fused parts: sorted, non-overlapping, inside rng, non-empty, with a destination.
  auto const fused = [&](std::vector<fused_part> parts) {
    auto out  = device_slice(0, 2 * MiB, base_of(0));
    out.fused = std::move(parts);
    return out;
  };
  auto const part = [](std::size_t offset, std::size_t size, std::uintptr_t dst = 0x1000) {
    return fused_part{range{offset, size}, address(dst)};
  };
  CHECK(is_lane_read(fused({part(0, 1 * MiB), part(1 * MiB, 1 * MiB)}), file_size, 8 * MiB));
  CHECK(is_lane_read(fused({part(4 * KiB, 4 * KiB), part(1 * MiB, 4 * KiB)}), file_size, 8 * MiB));
  CHECK_FALSE(
    is_lane_read(fused({part(1 * MiB, 1 * MiB), part(0, 1 * MiB)}), file_size, 8 * MiB));
  CHECK_FALSE(
    is_lane_read(fused({part(0, 1 * MiB), part(1 * MiB - 1, 4 * KiB)}), file_size, 8 * MiB));
  CHECK_FALSE(is_lane_read(fused({part(0, 1 * MiB), part(2 * MiB, 1)}), file_size, 8 * MiB));
  CHECK_FALSE(
    is_lane_read(fused({part(0, 1 * MiB), part(1 * MiB, 1 * MiB + 1)}), file_size, 8 * MiB));
  CHECK_FALSE(is_lane_read(fused({part(0, 0), part(1 * MiB, 1)}), file_size, 8 * MiB));
  CHECK_FALSE(is_lane_read(fused({part(0, 1, 0), part(1 * MiB, 1)}), file_size, 8 * MiB));
  auto shifted       = fused({part(0, 1 * MiB)});
  shifted.rng.offset = 4 * KiB;  // the part starts before rng
  CHECK_FALSE(is_lane_read(shifted, file_size, 8 * MiB));
}

TEST_CASE("fused plan post-conditions hold on random inputs", "[io][cufile][fused_plan]")
{
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    CAPTURE(seed);
    std::mt19937_64 rng{seed};
    auto const pick = [&rng](std::size_t lo, std::size_t hi) {
      return std::uniform_int_distribution<std::size_t>{lo, hi}(rng);
    };
    fused_plan_options options;
    options.window_bytes   = std::array<std::size_t, 3>{4 * KiB, 64 * KiB, 1 * MiB}[pick(0, 2)];
    options.gap_bytes      = pick(0, 2) == 0 ? 0 : pick(0, options.window_bytes - 1);
    options.fuse           = pick(0, 4) != 0;
    options.min_read_bytes = pick(0, 3) == 0 ? pick(0, options.window_bytes) : 0;
    auto const window      = options.window_bytes;
    auto const span        = window * pick(1, 8);
    std::size_t const file_size =
      pick(0, 1) == 0 ? IO_BLOCK_SIZE * pick(0, span / IO_BLOCK_SIZE) : pick(0, span);
    CAPTURE(window, options.gap_bytes, options.fuse, options.min_read_bytes, file_size);

    // Inputs: mostly device slices of a few keys; some ineligible host slices.
    auto const n = pick(1, 40);
    std::vector<prepared_io_slice> slices;
    std::vector<bool> host;
    std::size_t cursor = pick(0, window);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t offset = 0;
      switch (pick(0, 3)) {
        case 0: offset = cursor; break;                                  // contiguous
        case 1: offset = cursor + pick(0, options.gap_bytes + 8 * KiB); break;  // gap
        case 2: offset = cursor > 0 ? cursor - pick(0, std::min(cursor, window)) : 0; break;
        default: offset = pick(0, span); break;
      }
      std::size_t const size =
        pick(0, 3) == 0 ? IO_BLOCK_SIZE * pick(1, 3 * window / IO_BLOCK_SIZE)
                        : pick(1, pick(0, 1) == 0 ? 16 * KiB : 3 * window);
      cursor = offset + size;
      if (pick(0, 6) == 0) {
        slices.emplace_back(range{offset, size}, host_buffer{address(base_of(i))});
        host.push_back(true);
        continue;
      }
      key k;
      k.stream   = pick(0, 3) == 0 ? 0x200 : 0x100;
      k.gate     = pick(0, 3) == 0 ? 0x300 : 0;
      k.device   = pick(0, 5) == 0 ? 1 : 0;
      k.priority = pick(0, 5) == 0 ? io_priority::low : io_priority::high;
      slices.push_back(device_slice(offset, size, base_of(i), k));
      host.push_back(false);
    }
    auto const input = [&] {
      std::vector<std::pair<range, std::uintptr_t>> out;
      for (auto const& slice : slices) {
        out.emplace_back(slice.rng,
                         slice.has_device_request()
                           ? reinterpret_cast<std::uintptr_t>(slice.d_buffer.data)
                           : 0);
      }
      return out;
    }();
    std::vector<std::size_t> passed;  // inputs expected to pass through, in order
    std::vector<std::size_t> planned_inputs;
    for (std::size_t i = 0; i < n; ++i) {
      if (host[i] || input[i].first.offset >= file_size) {
        passed.push_back(i);
      } else {
        planned_inputs.push_back(i);
      }
    }
    std::vector<key> keys;
    for (auto const& slice : slices) {
      keys.push_back(key{reinterpret_cast<std::uintptr_t>(slice.d_buffer.stream.get()),
                         reinterpret_cast<std::uintptr_t>(slice.d_buffer.ready_gate.get()),
                         slice.d_buffer.device_id,
                         slice.priority});
    }

    auto const stats = plan_fused_reads(slices, file_size, options);

    // Pass-through first, in the original order.
    REQUIRE(slices.size() >= passed.size());
    for (std::size_t j = 0; j < passed.size(); ++j) {
      auto const i = passed[j];
      CHECK(same_range(slices[j].rng, input[i].first));
      if (host[i]) {
        CHECK(slices[j].h_buffer.is_contiguous());
      } else {
        CHECK(reinterpret_cast<std::uintptr_t>(slices[j].d_buffer.data) == input[i].second);
      }
    }

    // Every planned slice, and the pieces it serves per input slice.
    auto const body_end = file_size & ~(IO_BLOCK_SIZE - 1);
    std::vector<std::vector<range>> covered(n);  // per input: covered ranges inside the slice
    std::size_t reads = 0, parts = 0, payload = 0, issued = 0, tails = 0, total = 0;
    for (std::size_t j = passed.size(); j < slices.size(); ++j) {
      auto const& slice = slices[j];
      CAPTURE(j, slice.rng.offset, slice.rng.size, slice.fused.size());
      REQUIRE(slice.kind == io_op_kind::read);
      REQUIRE(slice.has_device_request());
      REQUIRE_FALSE(slice.has_host_request());
      REQUIRE(slice.on_complete == nullptr);
      CHECK(slice.d_buffer.ready_gate_set);
      bool const tail = slice.rng.offset >= body_end;
      if (tail) {
        ++tails;
        CHECK_FALSE(slice.is_fused());
        CHECK(slice.rng.end() <= file_size);
      } else {
        CHECK(is_lane_read(slice, file_size, window));
        CHECK(lane_window(slice).end() <= body_end);
        ++reads;
        parts += slice.is_fused() ? slice.fused.size() : 1;
        payload += slice.payload_bytes();
        issued += lane_window(slice).size;
      }
      if (slice.is_fused()) {
        CHECK(options.fuse);
        CHECK(slice.d_buffer.data == slice.fused.front().data);
        CHECK(slice.rng.offset == slice.fused.front().rng.offset);
        CHECK(slice.rng.end() == slice.fused.back().rng.end());
        CHECK(lane_window(slice).size >= options.min_read_bytes);
        for (std::size_t p = 1; p < slice.fused.size(); ++p) {
          CHECK(slice.fused[p].rng.offset - slice.fused[p - 1].rng.end() <= options.gap_bytes);
        }
      }
      for (auto const& pc : pieces_of(slice)) {
        auto const addr  = reinterpret_cast<std::uintptr_t>(pc.data);
        auto const index = static_cast<std::size_t>(addr >> 32) - 1;
        auto const inner = static_cast<std::size_t>(addr & 0xFFFFFFFFUL);
        REQUIRE(index < n);
        REQUIRE_FALSE(host[index]);
        // The same key as the slice it came from.
        CHECK(keys[index].stream ==
              reinterpret_cast<std::uintptr_t>(slice.d_buffer.stream.get()));
        CHECK(keys[index].gate ==
              reinterpret_cast<std::uintptr_t>(slice.d_buffer.ready_gate.get()));
        CHECK(keys[index].device == slice.d_buffer.device_id);
        CHECK(keys[index].priority == slice.priority);
        // At the right file offset.
        CHECK(pc.rng.offset == input[index].first.offset + inner);
        covered[index].push_back(range{inner, pc.rng.size});
        total += pc.rng.size;
      }
    }

    // Each destination byte of each eligible input slice is covered exactly once.
    std::size_t expected_total = 0;
    std::size_t split          = 0;
    for (auto const i : planned_inputs) {
      auto const& [r, dst] = input[i];
      auto const clamped   = std::min(r.end(), file_size) - r.offset;
      expected_total += clamped;
      auto& cover = covered[i];
      std::sort(cover.begin(), cover.end(), [](range a, range b) { return a.offset < b.offset; });
      std::size_t at = 0;
      for (auto const& c : cover) {
        CHECK(c.offset == at);
        at = c.offset + c.size;
      }
      CHECK(at == clamped);
      if (cover.size() > 1) ++split;
    }
    CHECK(total == expected_total);
    CHECK(stats.tail_slices == tails);
    CHECK(stats.split_slices == split);
    if (options.min_read_bytes == 0) {
      CHECK(stats.reads == reads);
      CHECK(stats.parts == parts);
      CHECK(stats.payload_bytes == payload);
      CHECK(stats.issued_bytes == issued);
    } else {
      CHECK(stats.reads <= reads);
      CHECK(stats.payload_bytes <= payload);
    }
  }
}

TEST_CASE("fused plan: a window closes at max_parts", "[io][cufile][fused_plan]")
{
  // 40 contiguous 4 KiB slices fit one 1 MiB window; with a cap of 16 parts they make 3.
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (std::size_t i = 0; i < 40; ++i) {
    ranges.emplace_back(i * 4 * KiB, 4 * KiB);
  }
  auto slices       = device_slices(ranges);
  auto options      = window_of(1 * MiB);
  options.max_parts = 16;
  auto const stats  = plan_fused_reads(slices, 64 * MiB, options);
  CHECK(stats.reads == 3);
  CHECK(stats.parts == 40);
  REQUIRE(slices.size() == 3);
  CHECK(slices[0].fused.size() == 16);
  CHECK(slices[1].fused.size() == 16);
  CHECK(slices[2].fused.size() == 8);
  for (auto const& slice : slices) {
    CHECK(is_lane_read(slice, 64 * MiB, 1 * MiB));
  }

  // No cap: one window.
  auto uncapped = device_slices(ranges);
  CHECK(plan_fused_reads(uncapped, 64 * MiB, window_of(1 * MiB)).reads == 1);
}

TEST_CASE("fused plan: a slice whose end overflows is clamped to the file",
          "[io][cufile][fused_plan]")
{
  constexpr std::size_t file_size = (3 * MiB) + 100;
  auto slices = device_slices({{5000, std::numeric_limits<std::size_t>::max() - 1000}});
  auto const stats = plan_fused_reads(slices, file_size, window_of(1 * MiB));
  std::size_t payload = 0;
  for (auto const& slice : slices) {
    for (auto const& p : pieces_of(slice)) {
      CHECK(p.rng.end() <= file_size);
      payload += p.rng.size;
    }
  }
  CHECK(payload == file_size - 5000);
  CHECK(stats.tail_slices == 1);
  CHECK(stats.reads >= 3);
}
