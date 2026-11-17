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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @file row_range_read.hpp
 * @brief Row-range read: copy a range of rows of a representation into caller-owned device buffers.
 *
 * A row-range reader appends rows `[row_begin, row_begin + k)` of the requested columns of a source
 * idata_representation to caller-owned device buffers (one column_destination per requested
 * column), without materializing the whole source on the GPU. This header is cudf-free; the
 * cudf-backed readers are registered by register_builtin_row_range_readers()
 * (cucascade/cudf/builtin_row_range_readers.hpp).
 *
 * @par Reader contract
 * Every registered row_range_read_fn implements these rules.
 *
 * 1. **Byte budget = slot capacity.** Deliver the largest `k <= max_rows` such that, for every
 *    requested column, all of these fit:
 *    - fixed-width: `(dst_row_offset + k) * width <= data.capacity_bytes`;
 *    - STRING: `dst_byte_offset + (off[row_begin+k] - off[row_begin]) <= data.capacity_bytes` and
 *      `(dst_row_offset + k + 1) * offset_width <= offsets.capacity_bytes`;
 *    - null mask (when present): `ceil((dst_row_offset + k) / 32) * 4 <= null_mask.capacity_bytes`;
 *    - and `row_begin + k <= num_rows`.
 * 2. `k == 0` only when `row_begin == num_rows` (then `source_exhausted = true`) or a single row
 *    does not fit (then `source_exhausted = false`; the caller grows the slot and retries).
 *    `row_begin > num_rows` → `std::out_of_range`.
 * 3. **Offsets are rebased:**
 *    `dst.offsets[dst_row_offset + i] =
 *    dst_byte_offset + (src_off[row_begin+i] - src_off[row_begin])`
 *    for `i in [0, k]` — i.e. `k + 1` offsets are written, converting width if needed. When
 *    `k == 0` nothing is written.
 * 4. **Null bits** go at bit `dst_row_offset + i`. Bits below `dst_row_offset` and at/after
 *    `dst_row_offset + k` are left untouched (read-modify-write the partial words). Because a
 *    partial word is read back, successive reads that append to the same destination must be
 *    ordered on the device: use one stream, or make the later stream wait for the earlier. An
 *    all-valid
 *    source (no source mask, or null_count == 0) writes 1s without reading a source mask. If the
 *    null_mask slot is absent and the source range may contain nulls (source null_count > 0) →
 *    `row_range_unsupported_error`.
 * 5. Types: all fixed-width cudf types and STRING (INT32/INT64 offsets). LIST/STRUCT/DICTIONARY32
 *    in the requested columns → `row_range_unsupported_error`. `dst.type_id` must equal the source
 *    column's type id, else `std::invalid_argument`.
 * 6. **Stream ordering:** all device work is enqueued on `stream`. The registry has already
 *    ordered `stream` after the source's writer: it waits on the representation's writer event
 *    when there is one, and synchronizes the source's device when a GPU-tier source has none
 *    (unknown writer). The caller records the reader event (`read_rows_into` does). Readers may
 *    synchronize `stream` only where the spec of that reader allows it (GPU reader: offsets D2H,
 *    once per offsets chunk; disk reader: host-side offset read). The GPU reader requires the
 *    source to live on the device of `stream` and throws `row_range_unsupported_error` otherwise.
 *    The host reader never synchronizes; it keeps the source's pinned blocks alive until `stream`
 *    has run its copies, so the host memory resource must outlive the work on `stream`.
 * 7. `scratch`: optional reservation on the destination GPU memory space for temporaries; allocate
 *    through `detail::allocate_scratch`. It may be nullptr.
 */

namespace cucascade {

/**
 * @brief One caller-owned device buffer a reader writes into.
 */
struct device_buffer_slot {
  /// Device pointer to the start of the buffer; nullptr means the buffer is absent.
  void* data{nullptr};
  /// Bytes available from `data`. Counted from `data`, not from the append offset, so it is the
  /// byte budget of reader-contract rule 1.
  std::size_t capacity_bytes{0};
};

/**
 * @brief Destination of one requested column: the buffers to append to and the append position.
 *
 * The reader appends at `dst_row_offset` (and `dst_byte_offset` for variable-width data) and never
 * writes before those positions, except for the read-modify-write of a partial null-mask word
 * (reader-contract rule 4).
 */
struct column_destination {
  /// Generic type tag (the numeric cudf::type_id in the cudf layer), as
  /// memory::column_metadata::type_id. Must equal the source column's type id, else the reader
  /// throws std::invalid_argument (rule 5).
  int32_t type_id{0};
  /// Fixed-width values, or STRING chars.
  device_buffer_slot data;
  /// Validity bitmask of 32-bit words. Absent means the source range must be all-valid; a source
  /// with null_count > 0 then makes the reader throw row_range_unsupported_error (rule 4).
  device_buffer_slot null_mask;
  /// STRING offsets; absent for fixed-width columns. `k + 1` rebased offsets are written (rule 3).
  device_buffer_slot offsets;
  /// Width in bytes of one destination offset: 4 (INT32) or 8 (INT64). The reader converts when
  /// the source offsets have a different width.
  uint8_t offset_width{4};
  /// Append position in rows: row `i` of the range lands at destination row `dst_row_offset + i`.
  std::size_t dst_row_offset{0};
  /// Append position in bytes within `data`. Used by variable-width columns only.
  std::size_t dst_byte_offset{0};
  /// Child destinations for LIST/STRUCT. Readers of this version refuse nested types with
  /// row_range_unsupported_error (rule 5).
  std::vector<column_destination> children;
};

/**
 * @brief Which columns and which rows of the source to read.
 */
struct row_range_request {
  /// Top-level source column indices, in output order. `dst[i]` receives column `columns[i]`.
  std::span<const std::size_t> columns;
  /// First source row to read. `row_begin > num_rows` makes the reader throw std::out_of_range;
  /// `row_begin == num_rows` reads nothing and reports the source exhausted (rule 2).
  std::size_t row_begin{0};
  /// Upper bound on the rows delivered. The reader delivers the largest `k <= max_rows` that fits
  /// every destination slot (rule 1).
  std::size_t max_rows{0};
};

/**
 * @brief Outcome of one row-range read.
 */
struct row_range_result {
  /// Rows delivered (`k`). 0 only when the source is exhausted or a single row does not fit
  /// (rule 2).
  std::size_t rows_written{0};
  /// Per requested column: bytes written to `data` (fixed-width: rows * width; STRING: chars
  /// bytes). Has one entry per requested column.
  std::vector<std::size_t> data_bytes_written;
  /// True when `row_begin + rows_written == source num_rows`.
  bool source_exhausted{false};
};

/**
 * @brief Thrown when a row-range read cannot be served for this source.
 *
 * Raised when no reader is registered for the source's representation type, when a requested
 * column has an unsupported type (LIST/STRUCT/DICTIONARY32), or when the destination has no
 * null-mask slot but the source range may contain nulls.
 */
class row_range_unsupported_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Opaque per-(caller, source batch) state a reader may keep between calls.
 *
 * Uncompressed readers ignore it.
 */
struct row_range_reader_state {
  virtual ~row_range_reader_state() = default;
};

/**
 * @brief Type alias for a row-range reader function.
 *
 * Registered for one concrete idata_representation type and bound by the reader contract in this
 * file's header comment.
 *
 * @param source The representation to read from; its dynamic type is the registered type.
 * @param request The columns and row range to read.
 * @param dst One destination per requested column (`dst.size() == request.columns.size()`).
 * @param stream CUDA stream all device work is enqueued on. The registry has already ordered it
 *        after the source's writer (rule 6).
 * @param scratch Optional reservation on the destination GPU memory space for temporaries
 *        (rule 7). May be nullptr.
 * @param state Optional slot for reader state kept between calls. May be nullptr.
 * @return The rows delivered, the bytes written per column, and whether the source is exhausted.
 */
using row_range_read_fn =
  std::function<row_range_result(const idata_representation& source,
                                 const row_range_request& request,
                                 std::span<column_destination> dst,
                                 ::cuda::stream_ref stream,
                                 memory::reservation* scratch,
                                 std::unique_ptr<row_range_reader_state>* state)>;

/**
 * @brief Registry of row-range readers, keyed by the source representation type.
 *
 * Type-indexed, thread-safe for concurrent registration and lookup, non-copyable and non-movable;
 * mirrors representation_converter_registry.
 *
 * @note cuCascade core ships no built-in readers. The cucascade-cudf layer registers them via
 * register_builtin_row_range_readers() (cucascade/cudf/builtin_row_range_readers.hpp).
 */
class row_range_reader_registry {
 public:
  /**
   * @brief Construct an empty reader registry.
   */
  row_range_reader_registry() = default;

  // Non-copyable, non-movable
  row_range_reader_registry(const row_range_reader_registry&)            = delete;
  row_range_reader_registry& operator=(const row_range_reader_registry&) = delete;
  row_range_reader_registry(row_range_reader_registry&&)                 = delete;
  row_range_reader_registry& operator=(row_range_reader_registry&&)      = delete;

  /**
   * @brief Register the reader for a representation type.
   *
   * @tparam Rep The source representation type (must derive from idata_representation)
   * @param fn The reader function to register
   * @throws std::invalid_argument if a reader for this type is already registered
   */
  template <typename Rep>
    requires std::derived_from<Rep, idata_representation>
  void register_reader(row_range_read_fn fn)
  {
    register_reader_impl(std::type_index(typeid(Rep)), std::move(fn));
  }

  /**
   * @brief Register the reader for a representation type given as a runtime type index.
   *
   * @param type The source representation type
   * @param fn The reader function to register
   * @throws std::invalid_argument if a reader for this type is already registered
   */
  void register_reader_impl(std::type_index type, row_range_read_fn fn);

  /**
   * @brief Check if a reader is registered for the dynamic type of @p source.
   *
   * @param source The source data representation
   * @return true if a reader is registered, false otherwise
   */
  [[nodiscard]] bool has_reader(const idata_representation& source) const;

  /**
   * @brief Check if a reader is registered for a representation type.
   *
   * @tparam Rep The source representation type
   * @return true if a reader is registered, false otherwise
   */
  template <typename Rep>
  [[nodiscard]] bool has_reader() const
  {
    return has_reader_impl(std::type_index(typeid(Rep)));
  }

  /**
   * @brief Check if a reader is registered for a runtime type index.
   *
   * @param type The source representation type
   * @return true if a reader is registered, false otherwise
   */
  [[nodiscard]] bool has_reader_impl(std::type_index type) const;

  /**
   * @brief Read a row range of @p source into @p dst.
   *
   * Dispatches on typeid(source). Before dispatch it validates
   * `dst.size() == request.columns.size()` and orders @p stream after the source's writer: if
   * `source.get_writer_event() != nullptr` it makes @p stream wait on that event; otherwise, for a
   * GPU-tier source (unknown writer), it calls cudaDeviceSynchronize on the source's device. That
   * call blocks the calling thread until every stream on the device is idle, so a caller that
   * holds up work on the device (for example a stream parked behind a host callback it controls)
   * deadlocks. Host and disk sources without a writer event are read without a wait. The reader
   * then follows the reader contract in this file's header comment. The caller records the reader
   * event (read_rows_into() does).
   *
   * @param source The representation to read from
   * @param request The columns and row range to read
   * @param dst One destination per requested column
   * @param stream CUDA stream all device work is enqueued on
   * @param scratch Optional reservation on the destination GPU memory space for temporaries; may
   *        be nullptr
   * @param state Optional slot for reader state kept between calls; may be nullptr
   * @return The rows delivered, the bytes written per column, and whether the source is exhausted
   * @throws std::invalid_argument if `dst.size() != request.columns.size()`
   * @throws row_range_unsupported_error if no reader is registered for the type of @p source
   */
  row_range_result read(const idata_representation& source,
                        const row_range_request& request,
                        std::span<column_destination> dst,
                        ::cuda::stream_ref stream,
                        memory::reservation* scratch                   = nullptr,
                        std::unique_ptr<row_range_reader_state>* state = nullptr) const;

 private:
  mutable std::shared_mutex _mutex;
  std::unordered_map<std::type_index, row_range_read_fn> _readers;
};

/**
 * @brief Read a row range of a locked batch into @p dst.
 *
 * Reads through a read lock the CALLER holds (@p batch), then calls
 * `batch.record_reader_event(stream)`, also when the read throws, so a later writer of a GPU-tier
 * batch waits for whatever this read enqueued. The reader event does not cover HOST sources
 * (record_reader_event is a no-op for them): there the host reader keeps the source's pinned
 * blocks alive until @p stream has run the copies.
 *
 * @param registry The reader registry to dispatch through
 * @param batch Read lock on the source batch, held by the caller for the duration of the call
 * @param request The columns and row range to read
 * @param dst One destination per requested column
 * @param stream CUDA stream all device work is enqueued on
 * @param scratch Optional reservation on the destination GPU memory space for temporaries; may be
 *        nullptr
 * @param state Optional slot for reader state kept between calls; may be nullptr
 * @return The rows delivered, the bytes written per column, and whether the source is exhausted
 * @throws std::logic_error if `batch.get_data()` is nullptr
 * @throws std::invalid_argument if `dst.size() != request.columns.size()`
 * @throws row_range_unsupported_error if no reader is registered for the batch's representation
 */
row_range_result read_rows_into(const row_range_reader_registry& registry,
                                const read_only_data_batch& batch,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch                   = nullptr,
                                std::unique_ptr<row_range_reader_state>* state = nullptr);

}  // namespace cucascade
