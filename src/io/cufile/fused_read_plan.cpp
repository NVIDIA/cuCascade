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

#include <cucascade/io/cufile/fused_read_plan.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

namespace {

constexpr auto npos = std::numeric_limits<std::size_t>::max();

[[nodiscard]] constexpr std::size_t align_down_block(std::size_t value) noexcept
{
  return value & ~(IO_BLOCK_SIZE - 1);
}

/// @p value rounded up to a block, saturating at the largest block boundary.
[[nodiscard]] constexpr std::size_t align_up_block(std::size_t value) noexcept
{
  constexpr auto last = std::numeric_limits<std::size_t>::max() & ~(IO_BLOCK_SIZE - 1);
  return value > last ? last : (value + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1);
}

[[nodiscard]] bool eligible(prepared_io_slice const& slice, std::size_t file_size) noexcept
{
  return slice.kind == io_op_kind::read && slice.has_device_request() &&
         !slice.has_host_request() && slice.on_complete == nullptr &&
         slice.d_buffer.ready_gate_set && !slice.is_fused() && slice.rng.size != 0 &&
         slice.rng.offset < file_size;
}

/// The grouping key, then the file offset, then the input position (a total order, so the
/// plan does not depend on the sort's stability).
[[nodiscard]] auto sort_key(std::vector<prepared_io_slice> const& slices, std::size_t i) noexcept
{
  auto const& slice = slices[i];
  return std::tuple{slice.d_buffer.device_id,
                    reinterpret_cast<std::uintptr_t>(slice.d_buffer.stream.get()),
                    reinterpret_cast<std::uintptr_t>(slice.d_buffer.ready_gate.get()),
                    slice.priority,
                    slice.rng.offset,
                    i};
}

[[nodiscard]] bool same_group(prepared_io_slice const& a, prepared_io_slice const& b) noexcept
{
  return a.d_buffer.device_id == b.d_buffer.device_id &&
         a.d_buffer.stream.get() == b.d_buffer.stream.get() &&
         a.d_buffer.ready_gate == b.d_buffer.ready_gate && a.priority == b.priority;
}

/// One piece of an input slice placed in a window.
struct part_ref {
  range rng;
  std::uint8_t* data{nullptr};
  std::size_t source{npos};  ///< Index of the input slice.
  bool whole{false};         ///< The piece is the whole input slice.
};

/// Builds the planned slices of one call without touching the input.  A slice to be moved
/// from the input as is is recorded as a placeholder plus its index in @c move_from.
class planner {
 public:
  planner(std::vector<prepared_io_slice> const& slices,
          std::size_t file_size,
          fused_plan_options const& options,
          std::size_t expected)
    : _slices(slices),
      _options(options),
      _body_end(align_down_block(file_size)),
      _file_size(file_size)
  {
    _planned.reserve(expected);
    _move_from.reserve(expected);
  }

  void add(std::size_t index)
  {
    auto const& slice = _slices[index];
    auto pos          = slice.rng.offset;
    // offset < file_size (eligible): no overflow, whatever the size.
    auto const end     = pos + std::min(slice.rng.size, _file_size - pos);
    bool const whole   = slice.rng.size <= _file_size - pos;
    std::size_t pieces = 0;
    while (pos < end) {
      ++pieces;
      if (pos >= _body_end) {  // the file's partial last block
        close();
        ++_stats.tail_slices;
        if (pos == slice.rng.offset && whole) {
          emit_original(index);
        } else {
          emit_plain(part_ref{
            range{pos, end - pos}, slice.d_buffer.data + (pos - slice.rng.offset), index, false});
        }
        break;
      }
      if (_open && (!_options.fuse || pos < _data_end || pos - _data_end > _options.gap_bytes ||
                    pos >= _limit)) {
        close();
      }
      if (!_open) {
        _open  = true;
        _start = align_down_block(pos);
        _limit = std::min(_start + _options.window_bytes, _body_end);
      }
      auto const take = std::min(end, _limit) - pos;
      _parts.push_back(part_ref{range{pos, take},
                                slice.d_buffer.data + (pos - slice.rng.offset),
                                index,
                                whole && pos == slice.rng.offset && take == slice.rng.size});
      _data_end = pos + take;
      pos += take;
      if (_data_end == _limit || (_options.max_parts != 0 && _parts.size() >= _options.max_parts)) {
        close();
      }
    }
    if (pieces > 1) ++_stats.split_slices;
  }

  /// Ends the current group.
  void finish_group() { close(); }

  [[nodiscard]] fused_plan_stats const& stats() const noexcept { return _stats; }
  [[nodiscard]] std::vector<prepared_io_slice>& planned() noexcept { return _planned; }
  [[nodiscard]] std::vector<std::size_t> const& move_from() const noexcept { return _move_from; }

 private:
  void close()
  {
    if (!_open) return;
    _open             = false;
    auto const window = align_up_block(_data_end) - _start;
    if (window < _options.min_read_bytes) {
      for (auto const& part : _parts) {
        if (part.whole) {
          emit_original(part.source);
        } else {
          emit_plain(part);
        }
      }
    } else {
      if (_parts.size() == 1) {
        if (_parts.front().whole) {
          emit_original(_parts.front().source);
        } else {
          emit_plain(_parts.front());
        }
      } else {
        emit_fused();
      }
      ++_stats.reads;
      _stats.parts += _parts.size();
      for (auto const& part : _parts) {
        _stats.payload_bytes += part.rng.size;
      }
      _stats.issued_bytes += window;
    }
    _parts.clear();
  }

  void emit_original(std::size_t index)
  {
    _planned.emplace_back();
    try {
      _move_from.push_back(index);
    } catch (...) {
      _planned.pop_back();
      throw;
    }
  }

  void push(prepared_io_slice&& slice)
  {
    _planned.push_back(std::move(slice));
    try {
      _move_from.push_back(npos);
    } catch (...) {
      _planned.pop_back();
      throw;
    }
  }

  void emit_plain(part_ref const& part)
  {
    auto const& source = _slices[part.source];
    prepared_io_slice slice;
    slice.rng           = part.rng;
    slice.d_buffer      = source.d_buffer;
    slice.d_buffer.data = part.data;
    slice.priority      = source.priority;
    slice.kind          = io_op_kind::read;
    push(std::move(slice));
  }

  void emit_fused()
  {
    auto const& first  = _parts.front();
    auto const& last   = _parts.back();
    auto const& source = _slices[first.source];
    prepared_io_slice slice;
    slice.rng           = range{first.rng.offset, last.rng.end() - first.rng.offset};
    slice.d_buffer      = source.d_buffer;
    slice.d_buffer.data = first.data;
    slice.priority      = source.priority;
    slice.kind          = io_op_kind::read;
    slice.fused.reserve(_parts.size());
    for (auto const& part : _parts) {
      slice.fused.push_back(fused_part{part.rng, part.data});
    }
    push(std::move(slice));
  }

  std::vector<prepared_io_slice> const& _slices;
  fused_plan_options const& _options;
  std::size_t const _body_end;
  std::size_t const _file_size;

  // The open window.
  bool _open{false};
  std::size_t _start{0};
  std::size_t _limit{0};
  std::size_t _data_end{0};
  std::vector<part_ref> _parts;

  std::vector<prepared_io_slice> _planned;
  std::vector<std::size_t> _move_from;  ///< Per planned slice: input index, or npos.
  fused_plan_stats _stats;
};

}  // namespace

fused_plan_stats plan_fused_reads(std::vector<prepared_io_slice>& slices,
                                  std::size_t file_size,
                                  fused_plan_options const& options)
{
  if (options.window_bytes == 0 || options.window_bytes % IO_BLOCK_SIZE != 0) {
    throw std::invalid_argument(
      "plan_fused_reads: window_bytes must be a non-zero multiple of IO_BLOCK_SIZE");
  }
  if (options.gap_bytes >= options.window_bytes) {
    throw std::invalid_argument("plan_fused_reads: gap_bytes must be smaller than window_bytes");
  }

  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < slices.size(); ++i) {
    if (eligible(slices[i], file_size)) {
      if (order.empty()) order.reserve(slices.size() - i);
      order.push_back(i);
    }
  }
  if (order.empty()) return {};
  std::sort(order.begin(), order.end(), [&slices](std::size_t a, std::size_t b) {
    return sort_key(slices, a) < sort_key(slices, b);
  });

  planner plan{slices, file_size, options, order.size()};
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (i != 0 && !same_group(slices[order[i - 1]], slices[order[i]])) plan.finish_group();
    plan.add(order[i]);
  }
  plan.finish_group();

  // Build the result before touching `slices`, so a failed allocation leaves every slice
  // with its caller.
  auto& planned         = plan.planned();
  auto const& move_from = plan.move_from();
  std::vector<prepared_io_slice> out;
  out.reserve(slices.size() - order.size() + planned.size());

  // Nothing below allocates or throws.  Eligibility is re-evaluated: a pass-through slice is
  // moved only after its own check, and an eligible one is only moved by the second loop.
  for (auto& slice : slices) {
    if (!eligible(slice, file_size)) out.push_back(std::move(slice));
  }
  for (std::size_t i = 0; i < planned.size(); ++i) {
    out.push_back(move_from[i] == npos ? std::move(planned[i]) : std::move(slices[move_from[i]]));
  }
  slices = std::move(out);
  return plan.stats();
}

range lane_window(prepared_io_slice const& slice) noexcept
{
  auto const begin = align_down_block(slice.rng.offset);
  auto const end   = align_up_block(slice.rng.end());
  return range{begin, end - begin};
}

bool is_lane_read(prepared_io_slice const& slice,
                  std::size_t file_size,
                  std::size_t window_bytes) noexcept
{
  if (slice.kind != io_op_kind::read || !slice.has_device_request() || slice.has_host_request() ||
      slice.on_complete != nullptr || slice.rng.size == 0) {
    return false;
  }
  auto const body_end = align_down_block(file_size);
  if (slice.rng.offset >= body_end || slice.rng.size > body_end - slice.rng.offset) return false;
  auto const window = lane_window(slice);
  if (window.size > window_bytes || window.end() > body_end) return false;
  if (slice.fused.empty()) return true;
  auto previous_end = slice.rng.offset;
  for (auto const& part : slice.fused) {
    if (part.rng.size == 0 || part.data == nullptr || part.rng.offset < previous_end ||
        part.rng.offset >= slice.rng.end() || part.rng.size > slice.rng.end() - part.rng.offset) {
      return false;
    }
    previous_end = part.rng.offset + part.rng.size;
  }
  return true;
}

}  // namespace cucascade::io::cufile
