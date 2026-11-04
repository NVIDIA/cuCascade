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

#include <cucascade/cudf/column_admission.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cucascade::io {

namespace {

/// Eviction priority of a column's chunks: 1 + 2 x rank, capped. 0 stays reserved for chunks with
/// no ranked column.
constexpr double max_eviction_priority = 250.0;

}  // namespace

column_admission::column_admission(column_admission_config config) : _config(config)
{
  if (_config.chunk_bytes == 0) {
    throw std::invalid_argument("column_admission: chunk_bytes must be positive");
  }
}

column_admission::piece_key column_admission::key_of(std::uint32_t column,
                                                     std::uint32_t file) noexcept
{
  return (static_cast<piece_key>(column) << 32) | file;
}

double column_admission::rank(std::uint32_t column) const
{
  auto const& c = _columns[column];
  return c.registered == 0 ? 0.0
                           : static_cast<double>(c.requested) / static_cast<double>(c.registered);
}

std::vector<std::uint64_t> column_admission::piece_chunks(layout const& file,
                                                          std::uint32_t column) const
{
  auto const chunk = static_cast<std::uint64_t>(_config.chunk_bytes);
  std::vector<std::uint64_t> chunks;
  for (auto const& seg : file.segments) {
    if (seg.column != column || seg.end == seg.begin) { continue; }
    auto first = seg.begin / chunk;
    if (!chunks.empty() && chunks.back() == first) { ++first; }  // shared with the previous segment
    for (auto c = first; c <= (seg.end - 1) / chunk; ++c) {
      chunks.push_back(c);
    }
  }
  return chunks;
}

std::uint64_t column_admission::uncovered_bytes(layout const& file,
                                                std::vector<std::uint64_t> const& chunks) const
{
  auto const uncovered =
    std::ranges::count_if(chunks, [&](auto c) { return file.chunk_refs[c] == 0; });
  return static_cast<std::uint64_t>(uncovered) * _config.chunk_bytes;
}

void column_admission::release(std::uint32_t column)
{
  auto& state = _columns[column];
  for (auto const piece : state.admitted_pieces) {
    auto& file = *_layout_of_file[piece & 0xffffffffU];
    for (auto const c : piece_chunks(file, column)) {
      if (--file.chunk_refs[c] == 0) { _admitted_total -= _config.chunk_bytes; }
    }
    _admitted.erase(piece);
  }
  state.admitted_footprint = 0;
  state.admitted_pieces.clear();
}

void column_admission::register_layout(std::string const& path,
                                       std::string const& table,
                                       std::vector<std::string> const& columns,
                                       std::span<std::uint32_t const> column_of_chunk,
                                       std::span<std::uint64_t const> offsets,
                                       std::span<std::uint64_t const> sizes)
{
  if (column_of_chunk.size() != offsets.size() || offsets.size() != sizes.size()) {
    throw std::invalid_argument("column_admission: chunk arrays must have the same length");
  }
  if (std::ranges::any_of(column_of_chunk, [&](auto c) { return c >= columns.size(); })) {
    throw std::invalid_argument("column_admission: column index out of range");
  }
  std::lock_guard lock(_mutex);
  if (_layouts.contains(path)) { return; }

  auto const file = static_cast<std::uint32_t>(_layouts.size());
  std::vector<std::uint32_t> ids(columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const name = table + "/" + columns[i];
    auto const [it, inserted] =
      _column_ids.try_emplace(name, static_cast<std::uint32_t>(_columns.size()));
    if (inserted) { _columns.emplace_back().name = name; }
    ids[i] = it->second;
  }

  layout entry{.file = file, .segments = {}, .chunk_refs = {}};
  entry.segments.reserve(offsets.size());
  for (std::size_t k = 0; k < offsets.size(); ++k) {
    auto const id = ids[column_of_chunk[k]];
    entry.segments.push_back({offsets[k], offsets[k] + sizes[k], id});
    _columns[id].registered += sizes[k];
  }
  std::ranges::sort(entry.segments,
                    [](segment const& a, segment const& b) { return a.begin < b.begin; });
  std::uint64_t last_end = 0;
  for (auto const& seg : entry.segments) {
    last_end = std::max(last_end, seg.end);
  }
  entry.chunk_refs.assign(last_end == 0 ? 0 : (last_end - 1) / _config.chunk_bytes + 1, 0);
  auto const [placed, added] = _layouts.emplace(path, std::move(entry));
  _layout_of_file.push_back(&placed->second);
}

bool column_admission::has_layout(std::string const& path) const
{
  std::lock_guard lock(_mutex);
  return _layouts.contains(path);
}

bool column_admission::admit(std::uint32_t column, layout& file)
{
  auto const key = key_of(column, file.file);
  if (_admitted.contains(key)) { return true; }
  // A piece is all of one column in one file, however little of it this read touches.
  auto chunks = piece_chunks(file, column);
  auto bytes  = uncovered_bytes(file, chunks);
  if (_admitted_total + bytes > _config.budget_bytes) {
    // Replace the lowest-ranked admitted column, but only if this column ranks clearly higher.
    auto lowest      = column;
    auto lowest_rank = std::numeric_limits<double>::max();
    for (std::uint32_t c = 0; c < _columns.size(); ++c) {
      if (c == column || _columns[c].admitted_footprint == 0) { continue; }
      if (auto const r = rank(c); r < lowest_rank) {
        lowest_rank = r;
        lowest      = c;
      }
    }
    if (lowest == column || rank(column) <= lowest_rank * _config.rank_margin) { return false; }
    release(lowest);
    bytes = uncovered_bytes(file, chunks);
    if (_admitted_total + bytes > _config.budget_bytes) { return false; }
  }
  for (auto const c : chunks) {
    ++file.chunk_refs[c];
  }
  _admitted.insert(key);
  _columns[column].admitted_pieces.push_back(key);
  _columns[column].admitted_footprint += bytes;
  _admitted_total += bytes;
  return true;
}

admission_advice column_admission::advise(std::string const& path,
                                          std::uint64_t offset,
                                          std::uint64_t size,
                                          std::uint64_t object_bytes)
{
  admission_advice advice;
  std::vector<cudf::io::text::byte_range_info> wanted;

  {
    std::lock_guard lock(_mutex);
    auto const it = _layouts.find(path);
    if (it == _layouts.end()) {
      wanted.emplace_back(static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size));
    } else {
      auto& file           = it->second;
      auto const& segments = file.segments;
      auto const end       = offset + size;
      auto seg             = std::ranges::upper_bound(
        segments, offset, std::less{}, [](segment const& s) { return s.end; });
      for (; seg != segments.end() && seg->begin < end; ++seg) {
        auto const begin = std::max(offset, seg->begin);
        auto const stop  = std::min(end, seg->end);
        auto& col        = _columns[seg->column];
        col.requested += stop - begin;
        if (admit(seg->column, file)) {
          wanted.emplace_back(static_cast<std::int64_t>(begin),
                              static_cast<std::int64_t>(stop - begin));
          col.admitted += stop - begin;
          auto const priority = std::min(max_eviction_priority, 1.0 + 2.0 * rank(seg->column));
          advice.eviction_priority =
            std::max(advice.eviction_priority, static_cast<std::uint8_t>(priority));
        } else {
          col.refused += stop - begin;
        }
      }
    }
  }

  // Widen to whole cache chunks (clipped to the object) and merge ranges that now touch.
  auto const chunk  = static_cast<std::int64_t>(_config.chunk_bytes);
  auto const object = static_cast<std::int64_t>(object_bytes);
  for (auto const& range : wanted) {
    auto begin = range.offset() / chunk * chunk;
    auto end   = std::min((range.offset() + range.size() + chunk - 1) / chunk * chunk, object);
    if (!advice.ranges.empty()) {
      auto const& last = advice.ranges.back();
      if (begin <= last.offset() + last.size()) {
        end   = std::max(end, last.offset() + last.size());
        begin = last.offset();
        advice.ranges.pop_back();
      }
    }
    advice.ranges.emplace_back(begin, end - begin);
  }
  return advice;
}

std::vector<column_stats> column_admission::stats() const
{
  std::lock_guard lock(_mutex);
  std::vector<column_stats> out;
  out.reserve(_columns.size());
  for (std::uint32_t c = 0; c < _columns.size(); ++c) {
    auto const& col = _columns[c];
    out.push_back({.name               = col.name,
                   .registered         = col.registered,
                   .requested          = col.requested,
                   .admitted           = col.admitted,
                   .refused            = col.refused,
                   .admitted_footprint = col.admitted_footprint,
                   .rank               = rank(c)});
  }
  return out;
}

}  // namespace cucascade::io
