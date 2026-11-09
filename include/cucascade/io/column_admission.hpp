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

#include <cucascade/io/byte_range.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cucascade::io {

/// Default size of one cache chunk, in bytes. 256 KiB measured best on S3: smaller chunks cost more
/// in per-chunk bookkeeping than they save in padding.
inline constexpr std::size_t default_cache_chunk_bytes = 256UL << 10;

/// Settings of a @ref column_admission policy.
struct column_admission_config {
  /// Cache footprint, in bytes, that admitted columns may fill. Footprints are counted in whole
  /// cache chunks, because a column chunk that ends mid-chunk still occupies the whole chunk. A
  /// chunk that two admitted columns share is counted once.
  std::size_t budget_bytes{0};

  /// Size of one cache chunk (see @c cache::config), used to count footprints and to widen advised
  /// ranges to whole chunks: a cached chunk cannot be widened later, so a partly filled one makes
  /// the neighbouring range miss on every repeat.
  std::size_t chunk_bytes{default_cache_chunk_bytes};

  /// A column replaces the lowest-ranked admitted column only if its rank exceeds that column's
  /// rank by this factor, so near-equal columns do not displace each other.
  double rank_margin{1.5};
};

/// What the cache should keep of one read.
struct admission_advice {
  /// Chunk-aligned ranges to advise to the cache. Empty if the read should bypass the cache.
  std::vector<byte_range> ranges;

  /// Eviction priority of the chunks in @ref ranges (see @c
  /// cache::cached_chunk::eviction_priority).
  std::uint8_t eviction_priority{0};
};

/// Per-column accounting, in bytes.
struct column_stats {
  std::string name;                     ///< "<table>/<column>"
  std::uint64_t registered{0};          ///< size of the column in every registered file
  std::uint64_t requested{0};           ///< bytes of the column that reads asked for
  std::uint64_t admitted{0};            ///< requested bytes the policy asked the cache to keep
  std::uint64_t refused{0};             ///< requested bytes the policy left uncached
  std::uint64_t admitted_footprint{0};  ///< cache footprint of the column's admitted pieces
  double rank{0.0};  ///< requested / registered: how often the column is read in full
};

/**
 * @brief Column-aware admission for the parquet reads of a cache that cannot hold everything.
 *
 * The cache keeps a chunk only if some request advised it, and evicts oldest-first. A scan that
 * reads many columns, only a few of which every query shares, would fill the cache with data that
 * is read once. This policy decides which column chunks of a read to advise, and at which
 * eviction priority, from how often each column has been read so far.
 *
 * Callers register the byte ranges of every column chunk of a file with @ref register_layout (from
 * the parquet footer), then pass every read to @ref advise. A column's rank is the bytes of it
 * that reads asked for divided by the bytes registered, that is how often it is read in full.
 *
 * Admission is sticky: a (file, column) piece that was admitted stays admitted, because the cache's
 * oldest-first eviction would otherwise evict hot chunks when the policy changed its mind. A new
 * piece is admitted while the admitted footprint fits @ref column_admission_config::budget_bytes.
 * When it does not, a column that ranks clearly higher (by @ref
 * column_admission_config::rank_margin) than the lowest-ranked admitted column replaces that column
 * entirely, and its chunks age out.
 *
 * Thread-safe.
 */
class column_admission {
 public:
  explicit column_admission(column_admission_config config);

  /**
   * @brief Register the column chunks of one file. Registering a path again has no effect.
   *
   * @param path Object path of the file.
   * @param table Name shared by the files of one table; columns are ranked across a table.
   * @param columns Column names.
   * @param column_of_chunk For each column chunk, its index into @p columns.
   * @param offsets For each column chunk, its byte offset in the file.
   * @param sizes For each column chunk, its size in bytes.
   */
  void register_layout(std::string const& path,
                       std::string const& table,
                       std::vector<std::string> const& columns,
                       std::span<std::uint32_t const> column_of_chunk,
                       std::span<std::uint64_t const> offsets,
                       std::span<std::uint64_t const> sizes);

  /// Whether @ref register_layout has seen @p path.
  [[nodiscard]] bool has_layout(std::string const& path) const;

  /**
   * @brief Decide what the cache should keep of the read [@p offset, @p offset + @p size) of @p
   * path.
   *
   * For a registered file the read is split by column, each column is admitted or refused, and the
   * admitted parts are returned. For an unregistered file the whole read is advised.
   *
   * @param object_bytes Size of the object, to clip the chunk-aligned ranges.
   */
  [[nodiscard]] admission_advice advise(std::string const& path,
                                        std::uint64_t offset,
                                        std::uint64_t size,
                                        std::uint64_t object_bytes);

  /// Per-column accounting, one entry per column seen.
  [[nodiscard]] std::vector<column_stats> stats() const;

 private:
  /// A piece is all of one column in one file: (column << 32) | file.
  using piece_key = std::uint64_t;

  struct segment {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint32_t column;  // index into _columns
  };
  struct layout {
    std::uint32_t file;             // index of the file in registration order
    std::vector<segment> segments;  // sorted by offset
    /// For each cache chunk of the file, how many admitted pieces cover it.
    std::vector<std::uint16_t> chunk_refs;
  };
  struct column_state {
    std::string name;
    std::uint64_t registered{0};
    std::uint64_t requested{0};
    std::uint64_t admitted{0};
    std::uint64_t refused{0};
    std::uint64_t admitted_footprint{0};
    std::vector<piece_key> admitted_pieces;
  };

  [[nodiscard]] static piece_key key_of(std::uint32_t column, std::uint32_t file) noexcept;
  [[nodiscard]] double rank(std::uint32_t column) const;
  /// The cache chunks of @p file that hold @p column, in ascending order.
  [[nodiscard]] std::vector<std::uint64_t> piece_chunks(layout const& file,
                                                        std::uint32_t column) const;
  /// Bytes of @p chunks that no admitted piece of @p file covers yet.
  [[nodiscard]] std::uint64_t uncovered_bytes(layout const& file,
                                              std::vector<std::uint64_t> const& chunks) const;
  /// Stop counting the pieces of @p column against the budget.
  void release(std::uint32_t column);
  /// Admit the piece (@p column, @p file), evicting a lower-ranked column if the budget needs it.
  bool admit(std::uint32_t column, layout& file);

  column_admission_config _config;
  mutable std::mutex _mutex;
  std::unordered_map<std::string, layout> _layouts;
  std::unordered_map<std::string, std::uint32_t> _column_ids;
  std::vector<column_state> _columns;
  std::vector<layout*> _layout_of_file;  // by file index; the map's nodes do not move
  std::unordered_set<piece_key> _admitted;
  std::uint64_t _admitted_total{0};
};

}  // namespace cucascade::io
