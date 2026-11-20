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

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/device_engine.hpp>
#include <cucascade/io/cufile/fused_read_plan.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

// Lets the context be destroyed on one of its reactors' workers (see ~templated_ioctx).
static_assert(reactor_reports_worker_thread<cufile_reactor>);

namespace {

/// Round @p value down to a multiple of @c IO_BLOCK_SIZE.
[[nodiscard]] constexpr std::size_t align_down_block(std::size_t value) noexcept
{
  return value & ~(IO_BLOCK_SIZE - 1);
}

/// Round @p value up to a multiple of @c IO_BLOCK_SIZE.
[[nodiscard]] constexpr std::size_t align_up_block(std::size_t value) noexcept
{
  return align_down_block(value + IO_BLOCK_SIZE - 1);
}

/**
 * The distance between the piece boundaries of [@p begin, @p end) split into
 * pieces of at most @p split bytes (a multiple of 4 KiB): boundaries fall at
 * align_down(begin) + k * step, and the pieces come out near-equal, so none is a
 * sliver.
 */
[[nodiscard]] std::size_t split_step(std::size_t begin, std::size_t end, std::size_t split) noexcept
{
  auto const span  = end - align_down_block(begin);
  auto const count = (span + split - 1) / split;
  return std::min(split, align_up_block((span + count - 1) / count));
}

/// How many pieces the cuFile ioctx splits @p slice into (1 = not split): only
/// device-only slices without a cache completion, larger than @p split bytes.
[[nodiscard]] std::size_t split_pieces(prepared_io_slice const& slice,
                                       std::size_t file_size,
                                       std::size_t split) noexcept
{
  if (split == 0 || !slice.has_device_request() || slice.has_host_request() ||
      slice.on_complete != nullptr || slice.rng.offset >= file_size) {
    return 1;
  }
  auto const begin = slice.rng.offset;
  auto const end   = begin + std::min(slice.rng.size, file_size - begin);
  if (end - begin <= split) return 1;
  auto const step = split_step(begin, end, split);
  return (end - align_down_block(begin) + step - 1) / step;
}

/// The reactors of a planned pool, each placed in its GPU's group.
[[nodiscard]] std::vector<std::unique_ptr<cufile_reactor>> make_reactors(
  pool_plan const& plan, std::shared_ptr<cufile_reactor::reactor_context> const& ctx)
{
  std::vector<std::unique_ptr<cufile_reactor>> reactors;
  reactors.reserve(plan.total_reactors());
  std::size_t index = 0;
  if (plan.groups.empty()) {
    for (std::size_t i = 0; i < plan.flat_reactors; ++i) {
      reactors.push_back(
        std::make_unique<cufile_reactor>(ctx, "cufile-" + std::to_string(index++)));
    }
    return reactors;
  }
  // Per GPU its lane reactors first, then its synchronous ones; the batch reactors, shared
  // by every GPU, after all groups.  Roles are exclusive in a planned pool.
  for (auto const& group : plan.groups) {
    for (std::size_t i = 0; i < group.lane_reactors; ++i) {
      reactor_placement placement{group.device, group.cpus, group.numa_node, true, false};
      reactors.push_back(std::make_unique<cufile_reactor>(
        ctx,
        "cufile-g" + std::to_string(group.device) + "-lane" + std::to_string(i),
        std::move(placement)));
      ++index;
    }
    for (std::size_t i = 0; i < group.n_reactors; ++i) {
      reactor_placement placement{group.device, group.cpus, group.numa_node, false, false};
      reactors.push_back(std::make_unique<cufile_reactor>(
        ctx,
        "cufile-g" + std::to_string(group.device) + "-" + std::to_string(i),
        std::move(placement)));
      ++index;
    }
  }
  for (std::size_t i = 0; i < plan.host_batch_reactors; ++i) {
    reactor_placement placement{-1, {}, -1, false, true};
    reactors.push_back(std::make_unique<cufile_reactor>(
      ctx, "cufile-batch-" + std::to_string(i), std::move(placement)));
    ++index;
  }
  return reactors;
}

/// The GPU that owns @p pointer, else the calling thread's current device.
[[nodiscard]] int device_of(void const* pointer)
{
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
      (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) &&
      attributes.device >= 0) {
    return attributes.device;
  }
  std::ignore = cudaGetLastError();
  int current = 0;
  CUCASCADE_CUDA_TRY(cudaGetDevice(&current));
  return current;
}

/**
 * @brief The gates of one submission, one per stream: an event recorded on the
 *        stream, on the submitting thread, when work queued on it before the
 *        call has not finished yet; null when it has.
 *
 * Reactors wait on the gate instead of resolving the stream on their worker,
 * where a per-thread default stream names a different stream.  A stream being
 * captured is refused (querying it would invalidate the capture, and the
 * transfer could not be part of the graph anyway).
 */
class stream_gates {
 public:
  /// @p operation ("read" / "write") names the transfer in errors.
  explicit stream_gates(std::string_view operation) : _operation(operation) {}

  [[nodiscard]] std::shared_ptr<cucascade::cuda::cuda_event> const& of(cudaStream_t stream)
  {
    for (auto const& entry : _gates) {
      if (entry.stream == stream) return entry.gate;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUCASCADE_CUDA_TRY(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
      throw std::invalid_argument(std::string{"cufile_ioctx: a device "} + std::string{_operation} +
                                  " cannot be issued on a stream that is being captured");
    }
    std::shared_ptr<cucascade::cuda::cuda_event> gate;
    // The legacy default stream (also the null handle) orders after prior
    // work on every blocking stream, which a query of it does not report:
    // always record its event and query that instead.
    bool const legacy = stream == nullptr || stream == cudaStreamLegacy;
    auto const state  = legacy ? cudaErrorNotReady : cudaStreamQuery(stream);
    if (state == cudaErrorNotReady) {
      std::ignore       = cudaGetLastError();
      int stream_device = 0;
      CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream, &stream_device));
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{stream_device}};
      gate = std::make_shared<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
      CUCASCADE_CUDA_TRY(cudaEventRecord(gate->get(), stream));
      if (legacy) {
        auto const recorded = cudaEventQuery(gate->get());
        if (recorded == cudaSuccess) {
          gate.reset();
        } else if (recorded != cudaErrorNotReady) {
          std::ignore = cudaGetLastError();
          throw cucascade::cuda_error(std::string{"cufile_ioctx: cudaEventQuery failed: "} +
                                      cudaGetErrorString(recorded));
        } else {
          std::ignore = cudaGetLastError();
        }
      }
    } else if (state != cudaSuccess) {
      std::ignore = cudaGetLastError();
      throw cucascade::cuda_error(std::string{"cufile_ioctx: cudaStreamQuery failed: "} +
                                  cudaGetErrorString(state));
    }
    _gates.push_back(entry{stream, std::move(gate)});
    return _gates.back().gate;
  }

 private:
  struct entry {
    cudaStream_t stream;
    std::shared_ptr<cucascade::cuda::cuda_event> gate;
  };
  std::string_view _operation;
  std::vector<entry> _gates;
};

/// Settle every slice of a request that could not be dispatched: release any
/// cache chunks and report the slices as not read.
void fail_slices(std::vector<prepared_io_slice>& slices) noexcept
{
  for (auto& slice : slices) {
    if (slice.on_complete != nullptr) {
      (*slice.on_complete)(slice.h_buffer.fragments(), false);
      slice.on_complete.reset();
    }
  }
}

}  // namespace

cufile_ioctx::cufile_ioctx(std::size_t n_reactors,
                           std::shared_ptr<cufile_reactor::reactor_context> ctx)
  : templated_ioctx<cufile_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<cufile_reactor>(ctx, "cufile-" + std::to_string(i++));
    })
{
  _plan.flat_reactors = n_reactors;
  _plan.reason        = "flat pool of " + std::to_string(n_reactors) + " reactors (configured)";
  _groups.push_back(group_range{-1, 0, _reactors.size()});
}

cufile_ioctx::cufile_ioctx(pool_plan plan, std::shared_ptr<cufile_reactor::reactor_context> ctx)
  : templated_ioctx<cufile_reactor>(make_reactors(plan, ctx)), _plan(std::move(plan))
{
  if (_plan.groups.empty()) {
    _groups.push_back(group_range{-1, 0, _reactors.size()});
    return;
  }
  std::size_t first = 0;
  for (auto const& group : _plan.groups) {
    auto const count = group.lane_reactors + group.n_reactors;
    _groups.push_back(group_range{group.device, first, count, group.lane_reactors});
    first += count;
  }
  _batch_first = first;
  _batch_count = _plan.host_batch_reactors;
}

void cufile_ioctx::start()
{
  if (_started) return;
  // Lane slots first: the plan budgets them before the synchronous staging.
  for (auto const& group : _plan.groups) {
    if (group.lane_slots == 0 || _plan.lane_slot_bytes == 0) continue;
    auto const reserved = reserve_staging_slots(group.device,
                                                group.lane_slots,
                                                _plan.lane_slot_bytes,
                                                group.staging_cap_bytes,
                                                /*require_registered=*/gds_available());
    if (reserved < group.lane_slots) {
      CUCASCADE_LOG_WARN(
        "cufile: reserved {} of {} fused-read lane slots on GPU {}; its lane reactors run "
        "with fewer lanes",
        reserved,
        group.lane_slots,
        group.device);
    } else {
      CUCASCADE_LOG_INFO("cufile: reserved {} fused-read lane slots ({} MiB) on GPU {}",
                         reserved,
                         (reserved * (_plan.lane_slot_bytes + (64UL << 10))) >> 20,
                         group.device);
    }
  }
  for (auto const& group : _plan.groups) {
    if (group.staging_slots == 0) continue;
    auto cfg                         = _config;
    cfg.max_staging_bytes_per_device = group.staging_cap_bytes;
    auto const reserved              = reserve_staging(group.device, group.staging_slots, cfg);
    if (reserved < group.staging_slots) {
      CUCASCADE_LOG_WARN(
        "cufile: reserved {} of {} registered staging slots on GPU {}; its reactors without a "
        "slot read directly through cuFile's bounce buffers",
        reserved,
        group.staging_slots,
        group.device);
    } else {
      CUCASCADE_LOG_INFO("cufile: reserved {} registered staging slots ({} MiB) on GPU {}",
                         reserved,
                         (reserved * _config.staging_slot_bytes()) >> 20,
                         group.device);
    }
  }
  templated_ioctx<cufile_reactor>::start();
}

std::shared_ptr<io_object> cufile_ioctx::create_io_object(std::string path)
{
  return std::shared_ptr<io_object>(
    cufile_reactor::create_io_object(std::move(path), _config.use_odirect, _config.compat_mode));
}

std::shared_ptr<io_object> cufile_ioctx::create_io_object_for_write(std::string path,
                                                                    write_open_options options)
{
  return std::shared_ptr<io_object>(cufile_reactor::create_io_object_for_write(
    std::move(path), options, _config.use_odirect, _config.compat_mode));
}

path_info cufile_ioctx::path_capability(io_object const& object) const
{
  auto const* typed = dynamic_cast<cufile_io_object const*>(&object);
  if (typed == nullptr) {
    throw std::invalid_argument(
      "cufile_ioctx::path_capability: I/O object belongs to another backend");
  }
  return probe_object(*typed, _config);
}

reactor_stats cufile_ioctx::stats() const noexcept
{
  reactor_stats total;
  for (auto const& reactor : _reactors) {
    total += reactor->stats();
  }
  return total;
}

reactor_stats cufile_ioctx::stats(int device) const noexcept
{
  reactor_stats total;
  for (auto const& group : _groups) {
    if (group.device != device || device < 0) continue;
    for (std::size_t i = group.first; i < group.first + group.count; ++i) {
      total += _reactors[i]->stats();
    }
  }
  return total;
}

bool cufile_ioctx::fused_reads_usable() const noexcept
{
  if (!_config.fused_reads.enabled) return false;
  auto const* api = cufile_api::get();
  return api != nullptr && api->has_async_read;
}

template <typename Visit>
void cufile_ioctx::for_each_lane_reactor(int device_id, Visit&& visit) const
{
  if (_plan.groups.empty()) {
    // A flat pool: every reactor may own lanes.
    for (std::size_t i = 0; i < _reactors.size(); ++i) {
      if (!visit(i)) return;
    }
    return;
  }
  for (auto const& group : _groups) {
    if (group.device != device_id) continue;
    for (std::size_t i = group.first; i < group.first + group.lanes; ++i) {
      if (!visit(i)) return;
    }
    return;
  }
}

bool cufile_ioctx::lanes_routable(int device_id) const noexcept
{
  if (device_id < 0 || !fused_reads_usable()) return false;
  bool usable = false;
  for_each_lane_reactor(device_id, [&](std::size_t index) {
    usable = _reactors[index]->lanes_usable(device_id);
    return !usable;
  });
  return usable;
}

bool cufile_ioctx::is_routable_lane_read(cufile_io_object const& object,
                                         prepared_io_slice const& slice) const noexcept
{
  return is_lane_read(slice, object.size(), _config.fused_reads.slot_bytes) &&
         lane_window(slice).size >= _config.gds_threshold && object.cufile_registered() &&
         lanes_routable(slice.d_buffer.device_id);
}

bool cufile_ioctx::has_lane_read(cufile_io_object const& object,
                                 std::span<prepared_io_slice const> slices) const noexcept
{
  if (!object.cufile_registered() || !fused_reads_usable()) return false;
  return std::any_of(slices.begin(), slices.end(), [&](prepared_io_slice const& slice) {
    return is_routable_lane_read(object, slice);
  });
}

std::size_t cufile_ioctx::slice_class(cufile_io_object const& object,
                                      prepared_io_slice const& slice) const
{
  // Lane reads go to lane reactors; everything else of the same request to the reactors
  // its own role picks (see select_reactors).
  return _config.fused_reads.enabled && is_routable_lane_read(object, slice) ? 1 : 0;
}

std::vector<std::size_t> cufile_ioctx::candidates(reactor_role role, int device_id) const
{
  std::vector<std::size_t> out;
  auto add = [&](std::size_t first, std::size_t count) {
    for (std::size_t i = first; i < first + count; ++i) {
      out.push_back(i);
    }
  };
  group_range const* group = nullptr;
  if (device_id >= 0) {
    for (auto const& entry : _groups) {
      if (entry.device == device_id && entry.count != 0) {
        group = &entry;
        break;
      }
    }
  }
  // Every reactor without lanes and without a batch role: the synchronous ones.
  auto add_all_sync = [&] {
    for (auto const& entry : _groups) {
      add(entry.first + entry.lanes, entry.count - entry.lanes);
    }
  };
  switch (role) {
    case reactor_role::lane:
      // Only reactors that can use lanes on the GPU now (not cooling down, lanes there).
      for_each_lane_reactor(device_id, [&](std::size_t index) {
        if (_reactors[index]->lanes_usable(device_id)) out.push_back(index);
        return true;
      });
      return out;  // empty: no lane reactor is usable; the caller routes synchronously
    case reactor_role::host:
      if (_batch_count != 0) {
        add(_batch_first, _batch_count);
      } else {
        add_all_sync();
      }
      break;
    case reactor_role::sync:
      if (group != nullptr) {
        if (group->count > group->lanes) {
          add(group->first + group->lanes, group->count - group->lanes);
        } else {
          add(group->first, group->count);
        }
      } else {
        add_all_sync();
      }
      break;
  }
  if (out.empty()) add(0, _reactors.size());
  return out;
}

std::vector<cufile_reactor*> cufile_ioctx::rank(std::vector<std::size_t> const& candidates,
                                                std::size_t n_slices,
                                                io_op_type type,
                                                int device_id)
{
  auto const count = candidates.size();
  if (count == 0) return {};

  // Rotation breaks backlog ties and keeps synchronous reads off reactor zero.
  auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  ranked.reserve(count);
  for (std::size_t distance = 0; distance < count; ++distance) {
    ranked.emplace_back(_reactors[candidates[(start + distance) % count]]->queued_bytes(),
                        distance);
  }
  // A vectored read with more slices than the default fan-out gets every
  // candidate; anything else the two least backlogged.  A host-only vectored
  // write (device_id stays unresolved without device slices) gets one: its
  // page-cache writes with early writeback already keep the device busy, and
  // interleaving writers of one file only contends on its inode.
  bool const fan_out_all = type == io_op_type::host_vector_async && n_slices > 2 && count > 2;
  bool const host_write  = type == io_op_type::write_vector_async && device_id < 0;
  auto const selected =
    fan_out_all ? count : std::min<std::size_t>(count, host_write ? 1 : 2);
  std::partial_sort(
    ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected), ranked.end());
  std::vector<cufile_reactor*> result;
  result.reserve(selected);
  for (std::size_t i = 0; i < selected; ++i) {
    result.push_back(_reactors[candidates[(start + ranked[i].second) % count]].get());
  }
  return result;
}

std::vector<cufile_reactor*> cufile_ioctx::next_reactor(
  [[maybe_unused]] cufile_io_object const& object,
  std::size_t n_slices,
  io_op_type type,
  int device_id)
{
  // A read without a device destination is host-only; everything else is served by the
  // synchronous reactors of its GPU (every reactor of a flat pool).
  bool const read = type == io_op_type::host || type == io_op_type::host_async ||
                    type == io_op_type::host_vector_async;
  auto const role = read && device_id < 0 ? reactor_role::host : reactor_role::sync;
  return rank(candidates(role, device_id), n_slices, type, device_id);
}

std::vector<cufile_reactor*> cufile_ioctx::select_reactors(
  cufile_io_object const& object,
  std::span<prepared_io_slice const> slices,
  io_op_type operation,
  int device_id)
{
  std::vector<cufile_reactor*> selected;
  if (operation == io_op_type::host_vector_async && device_id >= 0 &&
      has_lane_read(object, slices)) {
    if (auto const lane_reactors = candidates(reactor_role::lane, device_id);
        !lane_reactors.empty()) {
      selected = rank(lane_reactors, slices.size(), operation, device_id);
    }
  }
  if (selected.empty()) selected = next_reactor(object, slices.size(), operation, device_id);
  if (operation == io_op_type::write_vector_async && !selected.empty()) {
    for (auto const& slice : slices)
      if (slice.backend_owner &&
          selected.front()->engine_write_candidate(object.shared_from_this(), slice))
        object.write_state()->activate();
  }
  if (operation == io_op_type::host_vector_async && slices.size() > 1 && !selected.empty() &&
      selected.front()->coalescing_candidate(object, slices))
    selected.resize(1);
  return selected;
}

std::size_t cufile_ioctx::host_write_io(io_object const& object,
                                        std::size_t offset,
                                        std::size_t size,
                                        std::uint8_t const* source)
{
  auto const* file = dynamic_cast<cufile_io_object const*>(&object);
  if (_config.async_engine.mode == engine_mode::legacy || !_config.async_engine.async_writes ||
      !file || !file->writable() || !size)
    return templated_ioctx<cufile_reactor>::host_write_io(object, offset, size, source);
  auto state       = file->write_state();
  auto ticket      = state->begin();
  auto const parts = range_plan::split_write_range({offset, size}, write_granule());
  auto const mode  = parts[1].empty() ? write_io_mode::buffered : write_io_mode::mixed;
  auto interval    = state->acquire_legacy({offset, size}, write_granule(), mode);
  return templated_ioctx<cufile_reactor>::host_write_io(object, offset, size, source);
}


exec::semi_future<void> cufile_ioctx::flush_async_io(io_object const& object,
                                                     io_priority priority) noexcept
{
  try {
    auto const* file = dynamic_cast<cufile_io_object const*>(&object);
    if (_config.async_engine.mode != engine_mode::legacy && _config.async_engine.async_writes &&
        file && file->writable()) {
      auto state = file->write_state();
      if (state->has_engine_work())
        return device_engine::flush_async(object.shared_from_this(), std::move(state));
    }
    return templated_ioctx<cufile_reactor>::flush_async_io(object, priority);
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}
void cufile_ioctx::prepare_device_slices(io_object const& object,
                                         std::vector<prepared_io_slice>& slices)
{
  stream_gates gates{"read"};
  // Pieces no smaller than cuFile still serves: a piece below gds_threshold would
  // take the POSIX path.
  auto const split_bytes = std::max(
    _config.staging_piece_bytes, align_up_block(std::max<std::size_t>(_config.gds_threshold, 1)));
  auto const file_size = object.size();

  for (auto& slice : slices) {
    if (!slice.has_device_request() || slice.d_buffer.ready_gate_set) continue;
    if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_of(slice.d_buffer.data);

    slice.d_buffer.ready_gate     = gates.of(slice.d_buffer.stream.get());
    slice.d_buffer.ready_gate_set = true;
  }
  // Fused reads replace the split and the legacy fusion: the planner packs and splits the
  // device slices into lane reads of at most a slot.  Only when a lane reactor can take them
  // on every GPU the request reads into; otherwise the request takes the legacy path end to
  // end (its windows would only be served synchronously, by fewer reactors).
  auto const* file = dynamic_cast<cufile_io_object const*>(&object);
  if (file != nullptr && file->cufile_registered() && fused_reads_usable()) {
    bool routable  = false;
    bool any_other = false;
    int checked    = -1;
    for (auto const& slice : slices) {
      if (!slice.has_device_request()) continue;
      auto const device = slice.d_buffer.device_id;
      if (device == checked) continue;
      checked = device;
      if (lanes_routable(device)) {
        routable = true;
      } else {
        any_other = true;
      }
    }
    if (routable && !any_other) {
      fused_plan_options options;
      options.window_bytes   = _config.fused_reads.slot_bytes;
      options.gap_bytes      = _config.fused_reads.fuse_gap_bytes;
      options.fuse           = _config.fused_reads.fuse;
      options.min_read_bytes = align_up_block(_config.gds_threshold);
      options.max_parts      = _config.fused_reads.max_parts_per_read;
      std::ignore            = plan_fused_reads(slices, file_size, options);
      return;
    }
  }
  std::size_t piece_count = 0;  // slices after splitting
  for (auto const& slice : slices) {
    if (auto const pieces = split_pieces(slice, file_size, split_bytes); pieces > 1) {
      piece_count += pieces - 1;
    }
  }
  if (piece_count == 0) {
    fuse_device_slices(slices, file_size);
    return;
  }
  piece_count += slices.size();

  // Split on 4 KiB boundaries into near-equal pieces (see split_step): every
  // piece but the first starts aligned, each fits one staging slot, and none is
  // a sliver that would fall below gds_threshold.
  std::vector<prepared_io_slice> pieces;
  // Reserved up front: once slices start moving, nothing may throw.
  pieces.reserve(piece_count);
  for (auto& slice : slices) {
    if (split_pieces(slice, file_size, split_bytes) <= 1) {
      pieces.push_back(std::move(slice));
      continue;
    }
    auto const begin = slice.rng.offset;
    auto const end   = begin + std::min(slice.rng.size, file_size - begin);
    auto const step  = split_step(begin, end, split_bytes);
    auto boundary    = align_down_block(begin) + step;
    for (auto position = begin; position < end;) {
      auto const stop = std::min(boundary, end);
      prepared_io_slice piece;
      piece.rng      = range{position, stop - position};
      piece.d_buffer = slice.d_buffer;
      piece.d_buffer.data += position - begin;
      piece.priority = slice.priority;
      pieces.push_back(std::move(piece));
      position = stop;
      boundary += step;
    }
  }
  slices = std::move(pieces);
  fuse_device_slices(slices, file_size);
}

void cufile_ioctx::fuse_device_slices(std::vector<prepared_io_slice>& slices,
                                      std::size_t file_size) const
{
  if (_config.fuse_gap_bytes == 0 || slices.size() < 2) return;
  if (_config.staging_mode == staging::never) return;
  if (_config.staging_mode == staging::automatic && !gds_available()) return;

  // One staged call covers a whole fused window, and a part is small enough that
  // fusing it is worth more than reading it by itself (the halves of a split
  // slice are not: they were split to be read side by side).
  auto const window   = align_down_block(_config.staging_slot_bytes());
  auto const max_part = _config.staging_piece_bytes / 2;
  auto const eligible = [&](prepared_io_slice const& slice) {
    return slice.has_device_request() && !slice.has_host_request() && !slice.is_fused() &&
           slice.on_complete == nullptr && slice.d_buffer.ready_gate_set && slice.rng.size != 0 &&
           slice.rng.size <= max_part && slice.rng.offset < file_size &&
           slice.rng.size <= file_size - slice.rng.offset;
  };

  std::vector<std::size_t> order;
  order.reserve(slices.size());
  for (std::size_t i = 0; i < slices.size(); ++i) {
    if (eligible(slices[i])) order.push_back(i);
  }
  if (order.size() < 2) return;
  auto const key = [&](std::size_t i) {
    auto const& slice = slices[i];
    return std::tuple{slice.d_buffer.device_id,
                      static_cast<void const*>(slice.d_buffer.stream.get()),
                      static_cast<void const*>(slice.d_buffer.ready_gate.get()),
                      slice.priority,
                      slice.rng.offset};
  };
  std::sort(
    order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key(a) < key(b); });

  // group_of[i]: the run slice i belongs to, or npos when it stays as it is.
  constexpr auto npos = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> group_of(slices.size(), npos);
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t first = 0; first < order.size();) {
    auto const& head = slices[order[first]];
    auto const begin = align_down_block(head.rng.offset);
    auto run_end     = head.rng.offset + head.rng.size;
    std::size_t last = first + 1;
    for (; last < order.size(); ++last) {
      auto const& next = slices[order[last]];
      if (next.d_buffer.device_id != head.d_buffer.device_id ||
          next.d_buffer.stream.get() != head.d_buffer.stream.get() ||
          next.d_buffer.ready_gate != head.d_buffer.ready_gate || next.priority != head.priority) {
        break;
      }
      auto const next_end = next.rng.offset + next.rng.size;
      if (next.rng.offset < run_end || next.rng.offset - run_end > _config.fuse_gap_bytes ||
          align_up_block(next_end) - begin > window) {
        break;
      }
      run_end = next_end;
    }
    if (last - first >= 2) {
      for (std::size_t i = first; i < last; ++i) {
        group_of[order[i]] = groups.size();
      }
      groups.emplace_back(order.begin() + static_cast<std::ptrdiff_t>(first),
                          order.begin() + static_cast<std::ptrdiff_t>(last));
    }
    first = last;
  }
  if (groups.empty()) return;

  // Build the result before touching `slices`, so a failed allocation leaves
  // every slice with its caller.
  std::vector<prepared_io_slice> fused;
  fused.reserve(groups.size());
  for (auto const& group : groups) {
    auto const& head = slices[group.front()];
    auto const& tail = slices[group.back()];
    prepared_io_slice slice;
    slice.rng      = range{head.rng.offset, tail.rng.offset + tail.rng.size - head.rng.offset};
    slice.d_buffer = head.d_buffer;
    slice.priority = head.priority;
    slice.fused.reserve(group.size());
    for (auto const index : group) {
      slice.fused.push_back(fused_part{slices[index].rng, slices[index].d_buffer.data});
    }
    fused.push_back(std::move(slice));
  }
  std::vector<prepared_io_slice> out;
  out.reserve(slices.size() - (order.size() - groups.size()));
  for (std::size_t i = 0; i < slices.size(); ++i) {
    if (group_of[i] == npos) out.push_back(std::move(slices[i]));
  }
  for (auto& slice : fused) {
    out.push_back(std::move(slice));
  }
  slices = std::move(out);
}

std::shared_ptr<cufile_ioctx> make_cufile_ioctx(std::size_t n_reactors, config const& cfg)
{
  std::optional<memory::system_topology_info> topology;
  try {
    memory::topology_discovery discovery;
    if (discovery.discover(memory::NetworkDeviceVerification::EXISTS_ACTIVE)) {
      topology = discovery.get_topology();
    }
  } catch (std::exception const& e) {
    CUCASCADE_LOG_WARN("cufile: topology discovery failed ({}); planning without it", e.what());
  }
  auto plan      = plan_cufile_pool(cfg, topology ? &*topology : nullptr, probe_platform());
  auto effective = cfg;
  if (plan.staging_piece_bytes != 0) effective.staging_piece_bytes = plan.staging_piece_bytes;
  // The plan's per-device staging caps (BAR1, cuFile's pinned limit) hold for a
  // flat pool too; only the groups are dropped.
  std::vector<std::size_t> caps;
  for (auto const& group : plan.groups) {
    auto const index = static_cast<std::size_t>(group.device);
    // Devices the plan does not cover get what reactor_context::staging_cap gives them.
    auto const unplanned = effective.max_staging_bytes_per_device == 0
                             ? std::size_t{0}
                             : effective.max_staging_bytes_per_device +
                                 fused_lane_staging_bytes(effective);
    if (caps.size() <= index) caps.resize(index + 1, unplanned);
    caps[index] = group.staging_cap_bytes;
  }
  // The lanes each planned GPU's plan reserved bound its lane sets (unplanned: the
  // default), in a flat pool too, like the staging caps.
  std::vector<std::size_t> lane_quotas;
  for (auto const& group : plan.groups) {
    auto const index = static_cast<std::size_t>(group.device);
    if (lane_quotas.size() <= index) {
      lane_quotas.resize(index + 1, std::numeric_limits<std::size_t>::max());
    }
    lane_quotas[index] = group.lane_slots;
  }
  if (n_reactors != 0) {
    // A flat pool: no groups, so no lane or batch roles (every reactor has every enabled
    // capability) and no lane slots reserved up front.
    plan.groups.clear();
    plan.host_batch_reactors = 0;
    plan.lane_slot_bytes     = 0;
    plan.flat_reactors       = n_reactors;
    plan.reason        = "flat pool of " + std::to_string(n_reactors) + " reactors (configured)";
  }
  CUCASCADE_LOG_INFO("{}", describe(plan));
  for (auto const& warning : plan.warnings) {
    CUCASCADE_LOG_WARN("cufile: {}", warning);
  }
  auto ctx = std::make_shared<cufile_reactor::reactor_context>(
    std::move(effective), std::move(caps), std::move(lane_quotas));
  return std::make_shared<cufile_ioctx>(std::move(plan), std::move(ctx));
}

exec::semi_future<std::size_t> cufile_ioctx::mixed_readv_async_io(
  io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept
{
  try {
    prepare_device_slices(object, input_slices);
  } catch (...) {
    auto const error = std::current_exception();
    fail_slices(input_slices);
    return exec::make_semi_future<std::size_t>(error);
  }
  return templated_ioctx<cufile_reactor>::mixed_readv_async_io(object, std::move(input_slices));
}

void cufile_ioctx::prepare_device_writes(std::vector<prepared_io_slice>& slices)
{
  stream_gates gates{"write"};
  for (auto& slice : slices) {
    if (!slice.is_write() || !slice.has_device_request() || slice.d_buffer.ready_gate_set) {
      continue;
    }
    if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_of(slice.d_buffer.data);
    slice.d_buffer.ready_gate     = gates.of(slice.d_buffer.stream.get());
    slice.d_buffer.ready_gate_set = true;
  }
}

exec::semi_future<std::size_t> cufile_ioctx::mixed_writev_async_io(
  io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept
{
  try {
    prepare_device_writes(input_slices);
    if (_config.async_engine.mode != engine_mode::legacy && _config.async_engine.async_writes) {
      auto const* file = dynamic_cast<cufile_io_object const*>(&object);
      if (file && file->writable() && !input_slices.empty()) {
        auto ticket = file->write_state()->begin();
        for (auto& slice : input_slices) slice.backend_owner = ticket;
      }
    }
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
  return templated_ioctx<cufile_reactor>::mixed_writev_async_io(object, std::move(input_slices));
}

}  // namespace cucascade::io::cufile
