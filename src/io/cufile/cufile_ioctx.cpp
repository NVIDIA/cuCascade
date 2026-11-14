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
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

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
  for (auto const& group : plan.groups) {
    for (std::size_t i = 0; i < group.n_reactors; ++i) {
      reactor_placement placement{group.device, group.cpus};
      reactors.push_back(std::make_unique<cufile_reactor>(
        ctx,
        "cufile-g" + std::to_string(group.device) + "-" + std::to_string(i),
        std::move(placement)));
      ++index;
    }
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
    _groups.push_back(group_range{group.device, first, group.n_reactors});
    first += group.n_reactors;
  }
}

void cufile_ioctx::start()
{
  if (_started) return;
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

std::vector<cufile_reactor*> cufile_ioctx::next_reactor(
  [[maybe_unused]] cufile_io_object const& object,
  std::size_t n_slices,
  io_op_type type,
  int device_id)
{
  // The candidates: the reactors of the destination's GPU, or every reactor.
  std::size_t first = 0;
  std::size_t count = _reactors.size();
  if (device_id >= 0) {
    for (auto const& group : _groups) {
      if (group.device == device_id && group.count != 0) {
        first = group.first;
        count = group.count;
        break;
      }
    }
  }
  if (count == 0) return {};

  // Rotation breaks backlog ties and keeps synchronous reads off reactor zero.
  auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  ranked.reserve(count);
  for (std::size_t distance = 0; distance < count; ++distance) {
    ranked.emplace_back(_reactors[first + (start + distance) % count]->queued_bytes(), distance);
  }
  // A vectored read with more slices than the default fan-out gets every
  // candidate; anything else the two least backlogged.
  bool const fan_out_all = type == io_op_type::host_vector_async && n_slices > 2 && count > 2;
  auto const selected    = fan_out_all ? count : std::min<std::size_t>(count, 2);
  std::partial_sort(
    ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected), ranked.end());
  std::vector<cufile_reactor*> result;
  result.reserve(selected);
  for (std::size_t i = 0; i < selected; ++i) {
    result.push_back(_reactors[first + (start + ranked[i].second) % count].get());
  }
  return result;
}

void cufile_ioctx::prepare_device_slices(io_object const& object,
                                         std::vector<prepared_io_slice>& slices)
{
  // One gate per stream: the work queued on it before this call.
  struct stream_gate {
    cudaStream_t stream;
    std::shared_ptr<cucascade::cuda::cuda_event> gate;
  };
  std::vector<stream_gate> gates;
  // Pieces no smaller than cuFile still serves: a piece below gds_threshold would
  // take the POSIX path.
  auto const split_bytes = std::max(
    _config.staging_piece_bytes, align_up_block(std::max<std::size_t>(_config.gds_threshold, 1)));
  auto const file_size    = object.size();
  std::size_t piece_count = 0;  // slices after splitting

  for (auto& slice : slices) {
    if (!slice.has_device_request() || slice.d_buffer.ready_gate_set) continue;
    if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_of(slice.d_buffer.data);

    auto const stream = slice.d_buffer.stream.get();
    auto found        = std::find_if(
      gates.begin(), gates.end(), [stream](stream_gate const& g) { return g.stream == stream; });
    if (found == gates.end()) {
      // Never touch a stream being captured: querying it would invalidate the
      // capture, and the read could not be part of the graph anyway.
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      CUCASCADE_CUDA_TRY(cudaStreamIsCapturing(stream, &capture));
      if (capture != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument(
          "cufile_ioctx: a device read cannot be issued on a stream that is being captured");
      }
      std::shared_ptr<cucascade::cuda::cuda_event> gate;
      auto const state = cudaStreamQuery(stream);
      if (state == cudaErrorNotReady) {
        std::ignore       = cudaGetLastError();
        int stream_device = 0;
        CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream, &stream_device));
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{stream_device}};
        gate = std::make_shared<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
        CUCASCADE_CUDA_TRY(cudaEventRecord(gate->get(), stream));
      } else if (state != cudaSuccess) {
        std::ignore = cudaGetLastError();
        throw cucascade::cuda_error(std::string{"cufile_ioctx: cudaStreamQuery failed: "} +
                                    cudaGetErrorString(state));
      }
      gates.push_back(stream_gate{stream, std::move(gate)});
      found = std::prev(gates.end());
    }
    slice.d_buffer.ready_gate     = found->gate;
    slice.d_buffer.ready_gate_set = true;

    if (auto const pieces = split_pieces(slice, file_size, split_bytes); pieces > 1) {
      piece_count += pieces - 1;
    }
  }
  if (piece_count == 0) return;
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
    if (caps.size() <= index) caps.resize(index + 1, effective.max_staging_bytes_per_device);
    caps[index] = group.staging_cap_bytes;
  }
  if (n_reactors != 0) {
    plan.groups.clear();
    plan.flat_reactors = n_reactors;
    plan.reason        = "flat pool of " + std::to_string(n_reactors) + " reactors (configured)";
  }
  CUCASCADE_LOG_INFO("{}", describe(plan));
  for (auto const& warning : plan.warnings) {
    CUCASCADE_LOG_WARN("cufile: {}", warning);
  }
  auto ctx =
    std::make_shared<cufile_reactor::reactor_context>(std::move(effective), std::move(caps));
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

}  // namespace cucascade::io::cufile
