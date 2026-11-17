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

#include "cross_gpu_copy.hpp"

#include "batch_copy.hpp"
#include "chunk_plan.hpp"
#include "nvtx_scope.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade {
namespace {

using detail::source_span;

enum class copy_route { peer, host_staged };

/**
 * @brief One leaf buffer: `size` bytes at `src` on the source GPU go to `dst` on the destination
 * GPU.
 */
struct copy_segment {
  std::byte const* src;
  std::byte* dst;
  std::size_t size;
};

/**
 * @brief Receives each segment as soon as its destination buffer is allocated, in byte-stream
 * order.
 */
template <typename Sink>
concept segment_sink = requires(Sink& sink, copy_segment const& segment) { sink(segment); };

/**
 * @brief NUMA node of a GPU's PCI device, or -1 when it is unknown.
 */
[[nodiscard]] int resolve_gpu_numa_node(int device_id) noexcept
{
  if (device_id < 0) { return -1; }
  char pci_buf[32] = {0};
  if (cudaDeviceGetPCIBusId(pci_buf, sizeof(pci_buf), device_id) != cudaSuccess) { return -1; }
  std::string pci(pci_buf);
  for (auto& c : pci) {
    c = static_cast<char>(std::tolower(c));
  }
  std::ifstream f("/sys/bus/pci/devices/" + pci + "/numa_node");
  if (!f.is_open()) { return -1; }
  std::string content;
  std::getline(f, content);
  if (content.empty()) { return -1; }
  try {
    return std::stoi(content);
  } catch (...) {
    return -1;
  }
}

/**
 * @brief resolve_gpu_numa_node() of @p device_id, resolved once per process for every visible GPU.
 */
[[nodiscard]] int gpu_numa_node(int device_id)
{
  static std::vector<int> const nodes = [] {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      count       = 0;
    }
    std::vector<int> resolved(static_cast<std::size_t>(count));
    for (int device = 0; device < count; ++device) {
      resolved[static_cast<std::size_t>(device)] = resolve_gpu_numa_node(device);
    }
    return resolved;
  }();
  return device_id >= 0 && static_cast<std::size_t>(device_id) < nodes.size()
           ? nodes[static_cast<std::size_t>(device_id)]
           : -1;
}

[[nodiscard]] copy_route select_route(copy_route_selection selection,
                                      int src_device,
                                      int dst_device)
{
  switch (selection) {
    case copy_route_selection::host_staged: return copy_route::host_staged;
    case copy_route_selection::peer:
      if (!memory::probe_peer_dma_works(src_device, dst_device)) {
        CUCASCADE_FAIL("peer route forced, but peer DMA from GPU " + std::to_string(src_device) +
                       " to GPU " + std::to_string(dst_device) + " failed verification");
      }
      return copy_route::peer;
    case copy_route_selection::automatic: break;
  }
  return memory::probe_peer_dma_works(src_device, dst_device) ? copy_route::peer
                                                              : copy_route::host_staged;
}

/**
 * @brief Best-effort drain on an error path; a second failure must not replace the original
 * exception or stay pending.
 */
void drain_after_failure(::cuda::stream_ref stream) noexcept
{
  if (cudaStreamSynchronize(stream.get()) != cudaSuccess) { std::ignore = cudaGetLastError(); }
}

/**
 * @brief Whether @p col is a strings column whose chars buffer must be copied, which needs its
 * size.
 */
[[nodiscard]] bool has_chars(cudf::column_view const& col) noexcept
{
  return col.type().id() == cudf::type_id::STRING && col.num_children() >= 1 && col.size() > 0 &&
         col.head() != nullptr;
}

/**
 * @brief Whether @p col is a STRING column without an offsets child. cudf produces these for empty
 * results only, and validate_source_column() rejects one that has rows, so it is rebuilt as an
 * empty column and copies nothing.
 */
[[nodiscard]] bool is_degenerate_string(cudf::column_view const& col) noexcept
{
  return col.type().id() == cudf::type_id::STRING && col.num_children() < 1;
}

/**
 * @brief Throw std::invalid_argument if a column in @p col's tree cannot be copied buffer by
 * buffer: a sliced view, whose rows start inside its buffers, or a strings column that has rows but
 * no offsets child.
 */
void validate_source_column(cudf::column_view const& col)
{
  if (col.offset() != 0) {
    CUCASCADE_FAIL("copy_table_to_gpu: a sliced column view (non-zero offset) is not supported",
                   std::invalid_argument);
  }
  if (is_degenerate_string(col) && col.size() > 0) {
    CUCASCADE_FAIL("copy_table_to_gpu: STRING column with rows must have an offsets child",
                   std::invalid_argument);
  }
  for (auto child = col.child_begin(); child != col.child_end(); ++child) {
    validate_source_column(*child);
  }
}

/**
 * @brief Bytes of @p col's null mask that are copied: the words that hold its bits.
 */
[[nodiscard]] std::size_t null_mask_bytes(cudf::column_view const& col) noexcept
{
  return col.nullable() ? static_cast<std::size_t>(cudf::num_bitmask_words(col.size())) *
                            sizeof(cudf::bitmask_type)
                        : 0;
}

/**
 * @brief Bytes of the data buffer of a fixed-width (or fixed-point) column.
 */
[[nodiscard]] std::size_t fixed_width_bytes(cudf::column_view const& col)
{
  return col.head() == nullptr ? 0
                               : static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
}

/**
 * @brief Whether the copy of @p col descends into its children; fixed-width columns have none to
 * copy.
 */
[[nodiscard]] bool copies_children(cudf::column_view const& col) noexcept
{
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32: return true;
    default: return false;
  }
}

/**
 * @brief Address and width of the last offset of a strings column, which equals its chars size.
 */
struct last_offset {
  std::byte const* address;
  std::size_t width;
};

/**
 * @brief Collect the last offset of every strings column in @p col's tree, in the order
 * table_builder::build() and collect_source_spans() consume them.
 */
void collect_last_offsets(cudf::column_view const& col, std::vector<last_offset>& out)
{
  if (has_chars(col)) {
    cudf::column_view const offsets = col.child(0);
    std::size_t const width         = cudf::size_of(offsets.type());
    out.push_back({static_cast<std::byte const*>(offsets.head()) +
                     static_cast<std::size_t>(offsets.size() - 1) * width,
                   width});
  }
  if (!copies_children(col)) { return; }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_last_offsets(col.child(i), out);
  }
}

/**
 * @brief The source table and a stream on the source GPU that reads it, ordered after the source's
 * writer once, on first use.
 *
 * The stream comes from the source memory_space's stream pool. Both the planning read and the
 * source-to-host leg of the host-staged route read through it, so whichever runs first pays for the
 * ordering.
 */
class source_reader {
 public:
  explicit source_reader(gpu_table_representation const& source)
    : _source{source}, _stream{source.get_memory_space().acquire_stream()}
  {
  }

  source_reader(source_reader const&)            = delete;
  source_reader& operator=(source_reader const&) = delete;

  [[nodiscard]] gpu_table_representation const& get_source() const noexcept { return _source; }

  /**
   * @brief The reading stream, not necessarily ordered yet; only for draining it after a failure.
   */
  [[nodiscard]] ::cuda::stream_ref get_stream() const noexcept { return _stream; }

  /**
   * @brief The reading stream, ordered after the source's writer.
   */
  [[nodiscard]] ::cuda::stream_ref get_ordered_stream()
  {
    if (!_ordered) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_source.get_device_id()}};
      _source.wait_for_writer(_stream);
      _ordered = true;
    }
    return _stream;
  }

 private:
  gpu_table_representation const& _source;
  ::cuda::stream_ref _stream;
  bool _ordered{false};
};

/**
 * @brief Read the chars size of every strings column in @p view, the source table, with one batched
 * device-to-host copy and one sync.
 *
 * @return The chars sizes in the order collect_last_offsets() visits the strings columns
 */
[[nodiscard]] std::vector<std::size_t> read_chars_sizes(source_reader& reader,
                                                        cudf::table_view const& view)
{
  nvtx_scope const range{"convert:gpu_to_gpu:plan"};
  std::vector<last_offset> offsets;
  for (cudf::column_view const& col : view) {
    collect_last_offsets(col, offsets);
  }
  if (offsets.empty()) { return {}; }

  constexpr std::size_t slot_bytes              = sizeof(std::int64_t);
  std::size_t const bytes                       = offsets.size() * slot_bytes;
  rmm::host_device_async_resource_ref pinned_mr = cudf::get_pinned_memory_resource();
  auto release                                  = [pinned_mr, bytes](void* p) mutable noexcept {
    pinned_mr.deallocate_sync(p, bytes, slot_bytes);
  };
  std::unique_ptr<void, decltype(release)> scratch{pinned_mr.allocate_sync(bytes, slot_bytes),
                                                   release};
  auto* const host = static_cast<std::byte*>(scratch.get());

  ::cuda::stream_ref const stream = reader.get_ordered_stream();
  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{reader.get_source().get_device_id()}};
    batch_copy_accumulator batch;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      batch.add(host + i * slot_bytes, offsets[i].address, offsets[i].width);
    }
    batch.flush(stream, cudaMemcpySrcAccessOrderStream);
    stream.sync();
  } catch (...) {
    // Drain before the pinned scratch the copies write into is freed.
    drain_after_failure(stream);
    throw;
  }

  std::vector<std::size_t> sizes;
  sizes.reserve(offsets.size());
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    std::int64_t value = 0;
    if (offsets[i].width == sizeof(std::int32_t)) {
      std::int32_t narrow = 0;
      std::memcpy(&narrow, host + i * slot_bytes, sizeof(narrow));
      value = narrow;
    } else {
      std::memcpy(&value, host + i * slot_bytes, sizeof(value));
    }
    sizes.push_back(static_cast<std::size_t>(std::max<std::int64_t>(value, 0)));
  }
  return sizes;
}

/**
 * @brief Hands out the chars sizes of strings columns in the order they were collected.
 */
class chars_size_cursor {
 public:
  explicit chars_size_cursor(std::span<std::size_t const> sizes) noexcept : _sizes{sizes} {}

  [[nodiscard]] std::size_t next()
  {
    if (_next >= _sizes.size()) {
      CUCASCADE_FAIL("copy_table_to_gpu: strings column without a planned chars size");
    }
    return _sizes[_next++];
  }

 private:
  std::span<std::size_t const> _sizes;
  std::size_t _next{0};
};

/**
 * @brief Append the leaf buffers of @p col's tree in the order table_builder::build() hands their
 * segments to its sink.
 */
void collect_source_spans(cudf::column_view const& col,
                          chars_size_cursor& chars,
                          std::vector<source_span>& out)
{
  auto add = [&out](void const* src, std::size_t size) {
    if (size > 0 && src != nullptr) { out.push_back({static_cast<std::byte const*>(src), size}); }
  };
  if (is_degenerate_string(col)) { return; }
  add(col.null_mask(), null_mask_bytes(col));
  if (has_chars(col)) { add(col.head(), chars.next()); }
  if (!copies_children(col)) {
    add(col.head(), fixed_width_bytes(col));
    return;
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_source_spans(col.child(i), chars, out);
  }
}

/**
 * @brief Builds the destination column tree, allocating every buffer on the destination GPU and
 * handing each segment that fills one to a sink as soon as the buffer exists, so copies overlap the
 * remaining allocations.
 *
 * Only cudf factories that launch no work are used, so the tree may be assembled before its buffers
 * are written.
 */
template <segment_sink Sink>
class table_builder {
 public:
  table_builder(std::span<std::size_t const> chars_sizes,
                ::cuda::stream_ref stream,
                rmm::device_async_resource_ref mr,
                Sink& sink)
    : _chars{chars_sizes}, _stream{stream}, _mr{mr}, _sink{sink}
  {
  }

  [[nodiscard]] std::unique_ptr<cudf::table> build(cudf::table_view const& view)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(static_cast<std::size_t>(view.num_columns()));
    for (cudf::column_view const& col : view) {
      columns.push_back(build(col));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

 private:
  [[nodiscard]] std::unique_ptr<cudf::column> build(cudf::column_view const& src)
  {
    if (is_degenerate_string(src)) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }

    auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, _stream, _mr);
    if (src.nullable()) {
      null_mask = cudf::create_null_mask(src.size(), cudf::mask_state::UNINITIALIZED, _stream, _mr);
      add_segment(src.null_mask(), null_mask.data(), null_mask_bytes(src));
    }
    cudf::size_type const null_count = src.nullable() ? src.null_count() : 0;

    switch (src.type().id()) {
      case cudf::type_id::STRING: {
        std::size_t const chars_size = has_chars(src) ? _chars.next() : 0;
        rmm::device_buffer chars     = allocate_and_add_segment(src.head(), chars_size);
        auto offsets                 = build(src.child(0));
        return cudf::make_strings_column(
          src.size(), std::move(offsets), std::move(chars), null_count, std::move(null_mask));
      }
      case cudf::type_id::LIST: {
        if (src.num_children() < 2) {
          CUCASCADE_FAIL("copy_table_to_gpu: LIST column must have two children (offsets, values)",
                         std::invalid_argument);
        }
        auto offsets = build(src.child(0));
        auto values  = build(src.child(1));
        return cudf::make_lists_column(
          src.size(), std::move(offsets), std::move(values), null_count, std::move(null_mask));
      }
      case cudf::type_id::STRUCT: {
        std::vector<std::unique_ptr<cudf::column>> fields;
        fields.reserve(static_cast<std::size_t>(src.num_children()));
        for (cudf::size_type i = 0; i < src.num_children(); ++i) {
          fields.push_back(build(src.child(i)));
        }
        // Construct directly to skip make_structs_column's null superimposition: the copied
        // null masks are already consistent.
        return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                              src.size(),
                                              rmm::device_buffer{},
                                              std::move(null_mask),
                                              null_count,
                                              std::move(fields));
      }
      case cudf::type_id::DICTIONARY32: {
        if (src.num_children() < 2) {
          CUCASCADE_FAIL(
            "copy_table_to_gpu: DICTIONARY32 column must have two children (indices, keys)",
            std::invalid_argument);
        }
        // Child order is [0] = indices, [1] = keys; make_dictionary_column takes keys first.
        auto indices = build(src.child(0));
        auto keys    = build(src.child(1));
        return cudf::make_dictionary_column(
          std::move(keys), std::move(indices), std::move(null_mask), null_count);
      }
      default: {
        rmm::device_buffer data = allocate_and_add_segment(src.head(), fixed_width_bytes(src));
        return std::make_unique<cudf::column>(
          src.type(), src.size(), std::move(data), std::move(null_mask), null_count);
      }
    }
  }

  void add_segment(void const* src, void* dst, std::size_t size)
  {
    if (size == 0 || src == nullptr) { return; }
    _sink(copy_segment{static_cast<std::byte const*>(src), static_cast<std::byte*>(dst), size});
  }

  [[nodiscard]] rmm::device_buffer allocate_and_add_segment(void const* src, std::size_t size)
  {
    rmm::device_buffer buffer{size, _stream, _mr};
    add_segment(src, buffer.data(), size);
    return buffer;
  }

  chars_size_cursor _chars;
  ::cuda::stream_ref _stream;
  rmm::device_async_resource_ref _mr;
  Sink& _sink;
};

/**
 * @brief Peer route: copies each segment directly between the GPUs on the target stream.
 */
class peer_copier {
 public:
  peer_copier(int src_device, int dst_device, ::cuda::stream_ref target_stream) noexcept
    : _src_device{src_device}, _dst_device{dst_device}, _target_stream{target_stream}
  {
  }

  void operator()(copy_segment const& segment)
  {
    CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(
      segment.dst, _dst_device, segment.src, _src_device, segment.size, _target_stream.get()));
  }

 private:
  int _src_device;
  int _dst_device;
  ::cuda::stream_ref _target_stream;
};

/**
 * @brief Staging slots of one host-staged pipeline: pinned HOST pool blocks held under a HOST
 * reservation that the pipeline took for itself, cut into slots as detail::staging_slot_layout
 * describes.
 */
class staging_slots {
 public:
  /**
   * @brief Reserve and allocate up to @p max_depth slots of @p slot_bytes from @p host_space,
   * falling back to fewer slots as cross_gpu_copy_config describes.
   */
  staging_slots(memory::memory_space& host_space, std::size_t slot_bytes, std::size_t max_depth)
  {
    nvtx_scope const range{"convert:gpu_to_gpu:acquire_staging"};
    auto* const pool = host_space.get_memory_resource_of<memory::Tier::HOST>();
    if (pool == nullptr) { CUCASCADE_FAIL("host-staged route needs a HOST-tier memory_space"); }
    std::size_t const block_size = pool->get_block_size();

    std::size_t held_depth = 0;
    for (std::size_t depth = max_depth; depth > 0 && !_reservation; --depth) {
      std::size_t const bytes = rmm::align_up(depth * slot_bytes, block_size);
      if (bytes > host_space.get_max_memory()) { continue; }
      _reservation = host_space.make_reservation_or_null(bytes);
      if (!_reservation) { continue; }
      try {
        _blocks    = pool->allocate_multiple_blocks(bytes, _reservation.get());
        held_depth = depth;
      } catch (std::bad_alloc const&) {
        // The pool could not grow to back the reservation; treat it like a failed reservation.
        _reservation.reset();
      }
    }
    if (!_reservation) {
      throw memory::cucascade_out_of_memory(
        "host-staged route: HOST memory_space cannot reserve and allocate one staging slot",
        memory::MemoryError::LIMIT_EXCEEDED,
        rmm::align_up(slot_bytes, block_size),
        pool->get_total_allocated_bytes(),
        nullptr);
    }
    _layout =
      detail::staging_slot_layout{_blocks->get_blocks(), block_size, slot_bytes, held_depth};
  }

  [[nodiscard]] detail::staging_slot_layout const& get_layout() const noexcept { return _layout; }

 private:
  // Declared before _blocks so the blocks return to the reservation before it is released.
  std::unique_ptr<memory::reservation> _reservation;
  memory::fixed_multiple_blocks_allocation _blocks;
  detail::staging_slot_layout _layout;
};

/**
 * @brief Host-staged pipeline over a table's leaf buffers taken as one byte stream.
 *
 * Chunk c of the detail::chunk_plan goes through slot c % depth. Its source-to-host leg runs on the
 * source stream after the slot's previous "drained" event and records "filled"; its
 * host-to-destination leg runs on the target stream after "filled" and records "drained". Each leg
 * is one batched copy with one piece per leaf buffer and host run the chunk touches.
 *
 * The source-to-host leg needs only the source buffers, so prefill() can start the first chunks
 * before the destination buffers are allocated, hiding that allocation behind the copies. finish()
 * starts any chunk that prefill() did not.
 */
class host_staged_pipeline {
 public:
  host_staged_pipeline(std::span<source_span const> spans,
                       detail::staging_slot_layout const& slots,
                       int src_device,
                       int dst_device,
                       ::cuda::stream_ref source_stream,
                       ::cuda::stream_ref target_stream)
    : _spans{spans},
      _plan{spans, slots},
      _depth{slots.get_depth()},
      _src_device{src_device},
      _source_stream{source_stream},
      _target_stream{target_stream}
  {
    _filled.reserve(_depth);
    _drained.reserve(_depth);
    {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{src_device}};
      for (std::size_t i = 0; i < _depth; ++i) {
        _filled.emplace_back();
      }
    }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{dst_device}};
    for (std::size_t i = 0; i < _depth; ++i) {
      _drained.emplace_back();
    }
  }

  [[nodiscard]] std::size_t get_chunk_count() const noexcept { return _plan.get_chunk_count(); }

  /**
   * @brief Start the source-to-host leg of every chunk that has a free staging slot.
   */
  void prefill()
  {
    while (_filled_chunks < std::min(_depth, get_chunk_count())) {
      fill_next();
    }
  }

  /**
   * @brief Run every remaining leg and wait until the last chunk has drained.
   *
   * @param destinations Destination address of each source span, in the same order
   */
  void finish(std::span<std::byte* const> destinations)
  {
    std::size_t const chunks = get_chunk_count();
    while (_drained_chunks < chunks) {
      if (_filled_chunks == _drained_chunks) { fill_next(); }
      drain_next(destinations);
      // The slot just drained is the one the next unfilled chunk uses.
      if (_filled_chunks < chunks) { fill_next(); }
    }
    // The last chunk drains after every earlier chunk on the target stream, and each chunk drains
    // after it was filled, so this covers every copy in both legs.
    if (chunks > 0) { _drained[(chunks - 1) % _depth].synchronize(); }
  }

 private:
  void fill_next()
  {
    std::size_t const slot = _filled_chunks % _depth;
    for (detail::piece const& p : _plan.get_pieces(_filled_chunks)) {
      _batch.add(p.host, _spans[p.span].src + p.offset, p.size);
    }

    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_src_device}};
    if (_filled_chunks >= _depth) { _drained[slot].wait(_source_stream); }
    _batch.flush(_source_stream, cudaMemcpySrcAccessOrderStream);
    _filled[slot].record(_source_stream);
    ++_filled_chunks;
  }

  void drain_next(std::span<std::byte* const> destinations)
  {
    std::size_t const slot = _drained_chunks % _depth;
    for (detail::piece const& p : _plan.get_pieces(_drained_chunks)) {
      _batch.add(destinations[p.span] + p.offset, p.host, p.size);
    }
    _filled[slot].wait(_target_stream);
    _batch.flush(_target_stream, cudaMemcpySrcAccessOrderStream);
    _drained[slot].record(_target_stream);
    ++_drained_chunks;
  }

  std::span<source_span const> _spans;
  detail::chunk_plan _plan;
  std::size_t _depth;
  int _src_device;
  ::cuda::stream_ref _source_stream;
  ::cuda::stream_ref _target_stream;
  std::vector<cuda::cuda_event> _filled;   ///< Per slot, on the source GPU
  std::vector<cuda::cuda_event> _drained;  ///< Per slot, on the destination GPU
  batch_copy_accumulator _batch;
  std::size_t _filled_chunks{0};
  std::size_t _drained_chunks{0};
};

/**
 * @brief HOST memory_space to stage through, chosen as cross_gpu_copy_config describes.
 */
[[nodiscard]] memory::memory_space& find_staging_space(memory::memory_space const& target)
{
  memory::memory_reservation_manager* manager = target.get_reservation_manager();
  if (manager != nullptr) {
    int const numa_node = gpu_numa_node(target.get_device_id());
    if (memory::memory_space* local = manager->get_memory_space(memory::Tier::HOST, numa_node)) {
      return *local;
    }
    std::span<memory::memory_space const*> const hosts =
      manager->get_memory_spaces_for_tier(memory::Tier::HOST);
    if (!hosts.empty()) {
      return *manager->get_memory_space(memory::Tier::HOST, hosts.front()->get_device_id());
    }
  }
  CUCASCADE_FAIL("host-staged route needs a HOST memory_space in the target's reservation manager");
}

[[nodiscard]] std::unique_ptr<cudf::table> copy_over_peer_route(
  gpu_table_representation const& source,
  cudf::table_view const& view,
  std::span<std::size_t const> chars_sizes,
  memory::memory_space const& target,
  ::cuda::stream_ref target_stream)
{
  nvtx_scope const range{"convert:gpu_to_gpu:peer_route"};
  peer_copier copier{source.get_device_id(), target.get_device_id(), target_stream};
  table_builder builder{chars_sizes, target_stream, target.get_default_allocator(), copier};
  try {
    source.wait_for_writer(target_stream);
    auto table = builder.build(view);
    // Sync for the source's lifetime: a caller may free the source right after the conversion, on
    // a stream not ordered with these reads.
    target_stream.sync();
    return table;
  } catch (...) {
    // The destination buffers were freed on the target stream while the stack unwound, behind the
    // copies into them. Drain so that no copy still reads the source when the exception leaves.
    drain_after_failure(target_stream);
    throw;
  }
}

[[nodiscard]] std::unique_ptr<cudf::table> copy_over_host_staged_route(
  source_reader& reader,
  cudf::table_view const& view,
  std::span<std::size_t const> chars_sizes,
  memory::memory_space const& target,
  ::cuda::stream_ref target_stream,
  cross_gpu_copy_config const& config)
{
  gpu_table_representation const& source = reader.get_source();
  nvtx_scope const range{"convert:gpu_to_gpu:host_staged_route"};
  rmm::device_async_resource_ref const mr = target.get_default_allocator();

  std::vector<source_span> spans;
  {
    chars_size_cursor chars{chars_sizes};
    for (cudf::column_view const& col : view) {
      collect_source_spans(col, chars, spans);
    }
  }
  std::size_t total_bytes = 0;
  for (source_span const& span : spans) {
    total_bytes += span.size;
  }

  std::vector<std::byte*> destinations;
  destinations.reserve(spans.size());
  auto record_destination = [&](copy_segment const& segment) {
    std::size_t const i = destinations.size();
    if (i >= spans.size() || spans[i].src != segment.src || spans[i].size != segment.size) {
      CUCASCADE_FAIL("copy_table_to_gpu: destination buffers do not match the planned byte stream");
    }
    destinations.push_back(segment.dst);
  };
  if (total_bytes == 0) {
    return table_builder{chars_sizes, target_stream, mr, record_destination}.build(view);
  }

  int const src_device             = source.get_device_id();
  int const dst_device             = target.get_device_id();
  memory::memory_space& host_space = find_staging_space(target);

  // A table that fits one staging slot is a single chunk, and no more slots are held than the
  // table has chunks. Padding is counted at its upper bound, so a table that only just fits its
  // slots can hold one slot more than it uses.
  std::size_t const staged_bytes = total_bytes + detail::alignment_padding_bound(spans);
  std::size_t const slot_bytes   = std::min(config.chunk_size_bytes, staged_bytes);
  std::size_t const max_chunks   = (staged_bytes + slot_bytes - 1) / slot_bytes;
  staging_slots const slots{host_space, slot_bytes, std::min(config.pipeline_depth, max_chunks)};

  try {
    nvtx_scope const pipeline_range{"convert:gpu_to_gpu:host_staged_pipeline"};
    ::cuda::stream_ref const source_stream = reader.get_ordered_stream();
    host_staged_pipeline pipeline{
      spans, slots.get_layout(), src_device, dst_device, source_stream, target_stream};
    // A single chunk is filled and drained back to back instead: its drain is one batch with a
    // piece per leaf buffer, and such a batch runs piece by piece, several times slower, when it
    // is enqueued on a target stream that has gone idle because the fill finished long before.
    if (pipeline.get_chunk_count() > 1) { pipeline.prefill(); }
    auto table = table_builder{chars_sizes, target_stream, mr, record_destination}.build(view);
    if (destinations.size() != spans.size()) {
      CUCASCADE_FAIL("copy_table_to_gpu: destination buffers do not match the planned byte stream");
    }
    pipeline.finish(destinations);
    return table;
  } catch (...) {
    // Drain both legs before `slots` returns the staging memory they use to the HOST pool. The
    // destination buffers were already freed on the target stream while the stack unwound.
    drain_after_failure(reader.get_stream());
    drain_after_failure(target_stream);
    throw;
  }
}

}  // namespace

void validate_cross_gpu_copy_config(cross_gpu_copy_config const& config)
{
  if (config.chunk_size_bytes == 0) {
    CUCASCADE_FAIL("cross_gpu_copy_config: chunk_size_bytes must be positive",
                   std::invalid_argument);
  }
  if (config.pipeline_depth == 0) {
    CUCASCADE_FAIL("cross_gpu_copy_config: pipeline_depth must be at least 1",
                   std::invalid_argument);
  }
  if (config.pipeline_depth > std::numeric_limits<std::size_t>::max() / config.chunk_size_bytes) {
    CUCASCADE_FAIL("cross_gpu_copy_config: chunk_size_bytes x pipeline_depth overflows",
                   std::invalid_argument);
  }
}

std::unique_ptr<cudf::table> copy_table_to_gpu(gpu_table_representation const& source,
                                               memory::memory_space const& target,
                                               ::cuda::stream_ref target_stream,
                                               cross_gpu_copy_config const& config)
{
  cudf::table_view const view = source.get_table_view();
  for (cudf::column_view const& col : view) {
    validate_source_column(col);
  }

  int const src_device   = source.get_device_id();
  int const dst_device   = target.get_device_id();
  copy_route const route = select_route(config.route, src_device, dst_device);

  source_reader reader{source};
  std::vector<std::size_t> const chars_sizes = read_chars_sizes(reader, view);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{dst_device}};
  if (route == copy_route::peer) {
    return copy_over_peer_route(source, view, chars_sizes, target, target_stream);
  }
  return copy_over_host_staged_route(reader, view, chars_sizes, target, target_stream, config);
}

}  // namespace cucascade
