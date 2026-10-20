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

#include "io/rest/worker/intake.hpp"

#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cucascade::io::rest::detail {

namespace {

/// The physical ranges of a fragmented cache fill: one range per run of contiguous chunk fills,
/// each chunk's extent being its advertised fill span; a run grows up to @p target, and one fill
/// may reach max(@p max_segment, block size) on its own (a fill is published atomically at chunk
/// granularity, so it is never split).
std::vector<range> fill_ranges(prepared_io_slice const& slice,
                               std::size_t target,
                               std::size_t max_segment,
                               std::size_t block_size)
{
  if (block_size == 0) {
    throw std::runtime_error("rest_reactor: fragmented read requires a cache block size");
  }
  auto const max_fill = std::max(max_segment, block_size);

  std::vector<range> result;
  range current{};
  for (auto* chunk : slice.h_buffer.fragments()) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::runtime_error("rest_reactor: fragmented read has no cache buffer");
    }
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    auto const fill = range{fill_lo, fill_hi - fill_lo};
    if (fill.empty()) continue;
    if (fill.size > max_fill) {
      throw std::runtime_error("rest_reactor: one cache fill exceeds the REST segment maximum");
    }

    bool const contiguous = !current.empty() && current.end() == fill.offset;
    if (current.empty()) {
      current = fill;
    } else if (contiguous && current.size < target && current.size + fill.size <= max_fill) {
      current.size += fill.size;
    } else {
      result.push_back(current);
      current = fill;
    }
  }
  if (!current.empty()) result.push_back(current);
  if (result.empty()) {
    throw std::runtime_error("rest_reactor: fragmented read has no physical fill ranges");
  }
  return result;
}

/// The destination iovecs of the physical range @p io_rng of @p slice: the caller's buffer at
/// the op's offset, or the cache chunks' fill extents it covers; empty for a staged device read
/// (the worker binds staging blocks before the GET starts).
std::vector<iovec> operation_iovecs(prepared_io_slice const& slice,
                                    range io_rng,
                                    std::size_t block_size)
{
  std::vector<iovec> result;
  if (slice.is_contiguous()) {
    auto* base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    result.push_back(iovec{base + (io_rng.offset - slice.rng.offset), io_rng.size});
    return result;
  }
  if (slice.needs_staging()) return result;

  std::size_t covered = 0;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    auto const overlap = intersect(io_rng, range{fill_lo, fill_hi - fill_lo});
    if (overlap.empty()) continue;
    result.push_back(iovec{chunk->data + (overlap.offset - chunk->offset), overlap.size});
    covered += overlap.size;
  }
  if (covered != io_rng.size) {
    throw std::runtime_error("rest_reactor: cache fragments do not cover the physical range");
  }
  return result;
}

/// The cache chunks the physical range @p io_rng fills (handed to the slice's completion).
std::vector<cache::cached_chunk*> operation_chunks(prepared_io_slice const& slice,
                                                   range io_rng,
                                                   std::size_t block_size)
{
  std::vector<cache::cached_chunk*> result;
  if (!slice.is_fragmented()) return result;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    if (!intersect(io_rng, range{fill_lo, fill_hi - fill_lo}).empty()) { result.push_back(chunk); }
  }
  return result;
}

}  // namespace

void split_even(range slice, std::size_t target, std::size_t max_segment, std::vector<range>& out)
{
  if (slice.empty()) return;
  max_segment                = std::max<std::size_t>(max_segment, 1);
  target                     = std::clamp<std::size_t>(target, 1, max_segment);
  auto const preferred_count = std::max<std::size_t>(1, slice.size / target);
  auto const max_bound_count = 1 + (slice.size - 1) / max_segment;
  auto const count           = std::max(preferred_count, max_bound_count);
  auto const base            = slice.size / count;
  auto const rem             = slice.size % count;
  auto offset                = slice.offset;
  for (std::size_t i = 0; i < count; ++i) {
    auto const bytes = base + (i < rem ? 1 : 0);
    out.push_back(range{offset, bytes});
    offset += bytes;
  }
}

std::size_t expand_slice(grouped_io_request& group,
                         tier t,
                         cut_fn const& cut,
                         std::size_t fill_target,
                         std::size_t block_size,
                         std::size_t max_segment,
                         std::atomic<std::size_t>& queued_bytes,
                         std::deque<std::unique_ptr<rest_io_op_request>>& out) noexcept
{
  auto slice            = group.take_front();
  auto const slice_size = slice.size();
  try {
    auto const* file = dynamic_cast<rest_io_object const*>(group.obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("rest_reactor: logical request belongs to another backend");
    }

    std::vector<range> ranges;
    if (slice.is_write() || slice.is_flush()) {
      // A write slice is never widened or clamped; templated_ioctx cut it at part boundaries.
      ranges.push_back(slice.rng);
    } else {
      if (slice.needs_staging() && !slice.has_device_request()) {
        throw std::invalid_argument("rest_reactor: a staged read must have a device destination");
      }
      // Footer-probe opens already fetched this suffix: a contiguous host-only read inside the
      // stash completes here.  Fragmented cache fills and device reads keep the normal physical
      // lifecycle (their completion publishes chunks / records copies).
      if (slice.is_host_request() && slice.is_contiguous()) {
        auto const& stash = file->stash();
        auto const lo     = file->stash_window_lo();
        auto const hi     = stash == nullptr ? lo : lo + stash->size();
        if (stash != nullptr && slice.rng.offset >= lo && slice.rng.offset <= hi &&
            slice.rng.size <= hi - slice.rng.offset) {
          auto* dst = std::get<std::uint8_t*>(slice.h_buffer.buffer);
          std::memcpy(dst, stash->data() + (slice.rng.offset - lo), slice.rng.size);
          queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
          if (slice.on_complete != nullptr) { (*slice.on_complete)({}, true); }
          group.coordinator->on_complete();
          return 0;
        }
      }

      if (slice.is_fragmented()) {
        ranges = fill_ranges(slice, fill_target, max_segment, block_size);
      } else {
        cut(slice.rng, ranges);
      }
      for (auto& io_rng : ranges) {
        io_rng.size =
          io_rng.offset >= file->size() ? 0 : std::min(io_rng.size, file->size() - io_rng.offset);
      }
      std::erase_if(ranges, [](range const& io_rng) { return io_rng.empty(); });
      if (ranges.empty()) throw std::runtime_error("rest_reactor: empty physical plan");
    }

    std::vector<std::unique_ptr<rest_io_op_request>> expanded;
    expanded.reserve(ranges.size());
    std::size_t logical_bytes = 0;
    for (auto const io_rng : ranges) {
      auto op         = std::make_unique<io_op_request>();
      op->obj         = group.obj;
      op->kind        = slice.kind;
      op->io_rng      = io_rng;
      op->coordinator = group.coordinator;
      op->on_complete = slice.on_complete;
      if (slice.is_write()) {
        if (!slice.has_device_request()) {
          auto* src = std::get<std::uint8_t*>(slice.h_buffer.buffer);
          op->iovecs.push_back(iovec{src, io_rng.size});
        }
      } else if (!slice.is_flush()) {
        op->iovecs            = operation_iovecs(slice, io_rng, block_size);
        op->completion_chunks = operation_chunks(slice, io_rng, block_size);
      }
      if (slice.has_device_request()) {
        op->device_copy = std::make_unique<device_cpy_request>(
          device_cpy_request{slice.rng, slice.d_buffer, slice.d_buffer.device_id});
      }

      auto request           = std::make_unique<rest_io_op_request>();
      request->object        = file->get_object_ref();
      request->needs_staging = !slice.is_write() && !slice.is_flush() && slice.needs_staging();
      request->logical_bytes = slice.is_flush() ? 0 : intersect(slice.rng, io_rng).size;
      request->queue         = t;
      logical_bytes += request->logical_bytes;
      request->op = std::move(op);
      expanded.push_back(std::move(request));
    }
    if (logical_bytes != slice_size) {
      throw std::runtime_error("rest_reactor: physical plan does not cover logical slice");
    }

    auto const before = out.size();
    try {
      for (auto& request : expanded) {
        out.push_back(std::move(request));
      }
    } catch (...) {
      while (out.size() != before) {
        out.pop_back();
      }
      throw;
    }
    group.coordinator->add_tasks(expanded.size() - 1);
    return expanded.size();
  } catch (...) {
    queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
    if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    group.coordinator->report_error(std::current_exception());
    return 0;
  }
}

}  // namespace cucascade::io::rest::detail
