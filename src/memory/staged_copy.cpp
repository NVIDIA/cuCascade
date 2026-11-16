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

#include "staged_copy.hpp"

#include <cucascade/cuda/device_copy_batch.hpp>
#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/nvtx.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cucascade {
namespace memory {
namespace detail {

namespace {

using nvtx_range = nvtx3::scoped_range_in<libcucascade_domain>;

[[nodiscard]] constexpr std::size_t ceil_div(std::size_t value, std::size_t divisor) noexcept
{
  return value / divisor + (value % divisor != 0 ? 1U : 0U);
}

void require_stream_of_device(staged_copy_endpoint const& endpoint, char const* role)
{
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(cudaStreamGetDevice(endpoint.stream.get(), &stream_device));
  if (stream_device != endpoint.device_id) {
    throw std::invalid_argument(std::string{"staged copy: the "} + role +
                                " stream belongs to device " + std::to_string(stream_device) +
                                ", not to device " + std::to_string(endpoint.device_id));
  }
}

/**
 * @brief Wait for all work enqueued so far on @p endpoint's stream, on a path that must not throw.
 *
 * @param endpoint The stream and its device
 * @param marker An event of that device, recorded here and then waited on
 * @return false if the work could not be confirmed complete
 */
[[nodiscard]] bool wait_for_enqueued_work(staged_copy_endpoint const& endpoint,
                                          cuda::cuda_event const& marker) noexcept
{
  int previous_device = -1;
  if (cudaGetDevice(&previous_device) != cudaSuccess ||
      cudaSetDevice(endpoint.device_id) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return false;
  }
  bool complete = cudaEventRecord(marker.get(), endpoint.stream.get()) == cudaSuccess &&
                  cudaEventSynchronize(marker.get()) == cudaSuccess;
  if (cudaSetDevice(previous_device) != cudaSuccess) { complete = false; }
  if (!complete) { std::ignore = cudaGetLastError(); }
  return complete;
}

/// Pinned host memory for one copy, used when no HOST pool is registered.
class temporary_pinned_memory {
 public:
  temporary_pinned_memory(int numa_node, std::size_t bytes)
    : _mr(numa_node, /*make_portable=*/true),
      _bytes(bytes),
      _base(static_cast<std::byte*>(_mr.allocate_sync(bytes)))
  {
  }

  ~temporary_pinned_memory()
  {
    if (_base != nullptr) { _mr.deallocate_sync(_base, _bytes); }
  }

  temporary_pinned_memory(temporary_pinned_memory const&)            = delete;
  temporary_pinned_memory& operator=(temporary_pinned_memory const&) = delete;

  [[nodiscard]] std::byte* data() const noexcept { return _base; }

  /// Keep the memory pinned and mapped for the rest of the process.
  void abandon() noexcept { _base = nullptr; }

 private:
  numa_region_pinned_host_memory_resource _mr;
  std::size_t _bytes;
  std::byte* _base;
};

/// The host slots of one staged copy. Gives them back on destruction unless abandoned.
class staging_ring {
 public:
  staging_ring(std::size_t total_bytes,
               int destination_device,
               staged_copy_ring_settings const& settings)
  {
    int const numa_node = gpu_numa_node(destination_device);
    if (fixed_size_host_memory_resource* pool = find_host_pool(numa_node); pool != nullptr) {
      _block_bytes = pool->get_block_size();
      _geometry    = plan_staged_copy_ring(total_bytes, _block_bytes, settings);
      for (;;) {
        try {
          _pool_blocks = pool->allocate_multiple_blocks(bytes());
          break;
        } catch (rmm::out_of_memory const&) {
          if (!shrink_staged_copy_ring(_geometry)) { throw; }
        }
      }
    } else {
      _block_bytes = fixed_size_host_memory_resource::default_block_size;
      _geometry    = plan_staged_copy_ring(total_bytes, _block_bytes, settings);
      _temporary   = std::make_unique<temporary_pinned_memory>(numa_node, bytes());
    }
  }

  staging_ring(staging_ring const&)            = delete;
  staging_ring& operator=(staging_ring const&) = delete;

  [[nodiscard]] std::size_t slots() const noexcept { return _geometry.slots; }
  [[nodiscard]] std::size_t block_bytes() const noexcept { return _block_bytes; }
  [[nodiscard]] std::size_t slot_bytes() const noexcept
  {
    return _geometry.blocks_per_slot * _block_bytes;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return _geometry.slots * slot_bytes(); }

  /// Block @p index of slot @p slot; a slot's blocks need not be adjacent in memory.
  [[nodiscard]] std::byte* block(std::size_t slot, std::size_t index) const noexcept
  {
    std::size_t const position = slot * _geometry.blocks_per_slot + index;
    if (_pool_blocks) { return _pool_blocks->get_blocks()[position]; }
    return _temporary->data() + position * _block_bytes;
  }

  /// Never give the memory back: device work that could not be waited for may still write to it.
  void abandon() noexcept
  {
    if (_pool_blocks) {
      // The blocks stay allocated in the pool's accounting, which is what keeps them unused.
      std::ignore = _pool_blocks->release_blocks();
    }
    if (_temporary) { _temporary->abandon(); }
  }

 private:
  staged_copy_ring_geometry _geometry{};
  std::size_t _block_bytes{0};
  fixed_size_host_memory_resource::fixed_multiple_blocks_allocation _pool_blocks;
  std::unique_ptr<temporary_pinned_memory> _temporary;
};

/**
 * @brief The events of one leg: one per slot, recorded when the leg's copy into or out of that
 * slot is enqueued, and one to wait on after a failure.
 *
 * An event can only be recorded on a stream of the device that was current when it was created.
 */
class leg_events {
 public:
  leg_events(int device_id, std::size_t slots)
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
    _events.reserve(slots + 1);
    for (std::size_t i = 0; i < slots + 1; ++i) {
      _events.emplace_back();
    }
  }

  [[nodiscard]] cuda::cuda_event& slot(std::size_t index) noexcept { return _events[index]; }
  [[nodiscard]] cuda::cuda_event const& after_failure() const noexcept { return _events.back(); }

 private:
  std::vector<cuda::cuda_event> _events;
};

void call_before_leg(staged_copy_instrumentation* instrumentation,
                     staged_copy_leg leg,
                     std::size_t chunk_index,
                     ::cuda::stream_ref stream)
{
  if (instrumentation != nullptr && instrumentation->before_leg) {
    instrumentation->before_leg(leg, chunk_index, stream);
  }
}

}  // namespace

int gpu_numa_node(int device_id) noexcept
{
  if (device_id < 0) { return -1; }
  char pci_buf[32] = {0};
  if (cudaDeviceGetPCIBusId(pci_buf, static_cast<int>(sizeof(pci_buf)), device_id) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return -1;
  }
  try {
    // CUDA reports the bus id in upper case; sysfs names the device in lower case.
    std::string pci(pci_buf);
    for (auto& c : pci) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::ifstream file("/sys/bus/pci/devices/" + pci + "/numa_node");
    std::string content;
    if (!std::getline(file, content) || content.empty()) { return -1; }
    return std::stoi(content);
  } catch (...) {
    return -1;
  }
}

staged_copy_ring_geometry plan_staged_copy_ring(std::size_t total_bytes,
                                                std::size_t block_bytes,
                                                staged_copy_ring_settings const& settings)
{
  if (total_bytes == 0 || block_bytes == 0) {
    throw std::invalid_argument("staged copy: the payload and the block size must not be zero");
  }
  if (settings.chunk_bytes == 0 ||
      settings.chunk_bytes > staged_copy_ring_settings::max_chunk_bytes || settings.depth == 0 ||
      settings.depth > staged_copy_ring_settings::max_depth) {
    throw std::invalid_argument(
      "staged copy: chunk capacity must be 1 B to 1 GiB and ring depth 1 to 64");
  }

  std::size_t const chunk_blocks   = ceil_div(settings.chunk_bytes, block_bytes);
  std::size_t const payload_blocks = ceil_div(total_bytes, block_bytes);
  std::size_t const slots          = std::min(settings.depth, payload_blocks);
  std::size_t const blocks_per_slot =
    std::min(chunk_blocks, ceil_div(ceil_div(total_bytes, slots), block_bytes));

  if (blocks_per_slot > std::numeric_limits<std::size_t>::max() / slots / block_bytes) {
    throw std::invalid_argument("staged copy: the ring size overflows");
  }
  return staged_copy_ring_geometry{slots, blocks_per_slot};
}

bool shrink_staged_copy_ring(staged_copy_ring_geometry& geometry) noexcept
{
  if (geometry.blocks_per_slot > 1) {
    geometry.blocks_per_slot /= 2;
    return true;
  }
  if (geometry.slots > 2) {
    geometry.slots = 2;
    return true;
  }
  if (geometry.slots > 1) {
    geometry.slots = 1;
    return true;
  }
  return false;
}

void staged_copy(std::span<staged_copy_segment const> segments,
                 staged_copy_endpoint source,
                 staged_copy_endpoint destination,
                 staged_copy_ring_settings const& settings,
                 staged_copy_instrumentation* instrumentation)
{
  CUCASCADE_FUNC_RANGE();

  std::size_t total_bytes = 0;
  for (auto const& segment : segments) {
    if (segment.bytes == 0) { continue; }
    if (segment.source == nullptr || segment.destination == nullptr) {
      throw std::invalid_argument("staged copy: a segment with bytes has a null address");
    }
    if (segment.bytes > std::numeric_limits<std::size_t>::max() - total_bytes) {
      throw std::invalid_argument("staged copy: the payload size overflows");
    }
    total_bytes += segment.bytes;
  }
  if (total_bytes == 0) { return; }

  require_stream_of_device(source, "source");
  require_stream_of_device(destination, "destination");

  staging_ring ring{total_bytes, destination.device_id, settings};
  if (instrumentation != nullptr) {
    ++instrumentation->rings;
    instrumentation->last_ring_bytes = ring.bytes();
  }

  // Everything the loop needs is created before the first copy is enqueued.
  leg_events source_done{source.device_id, ring.slots()};
  leg_events destination_done{destination.device_id, ring.slots()};
  cuda::device_copy_batch to_host;
  cuda::device_copy_batch to_device;

  try {
    std::size_t segment_index  = 0;
    std::size_t segment_offset = 0;
    std::size_t chunk_index    = 0;
    std::size_t remaining      = total_bytes;

    while (remaining > 0) {
      std::size_t const slot        = chunk_index % ring.slots();
      std::size_t const chunk_bytes = std::min(ring.slot_bytes(), remaining);

      // Lay the next chunk_bytes of the segments out over the slot's blocks. A copy may cross
      // neither a segment boundary nor a block boundary.
      to_host.clear();
      to_device.clear();
      std::size_t filled = 0;
      while (filled < chunk_bytes) {
        auto const& segment = segments[segment_index];
        if (segment_offset == segment.bytes) {
          ++segment_index;
          segment_offset = 0;
          continue;
        }
        std::size_t const block_offset = filled % ring.block_bytes();
        std::size_t const piece        = std::min({segment.bytes - segment_offset,
                                                   ring.block_bytes() - block_offset,
                                                   chunk_bytes - filled});
        std::byte* const host = ring.block(slot, filled / ring.block_bytes()) + block_offset;
        to_host.add(host, static_cast<std::byte const*>(segment.source) + segment_offset, piece);
        to_device.add(static_cast<std::byte*>(segment.destination) + segment_offset, host, piece);
        filled += piece;
        segment_offset += piece;
      }

      {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source.device_id}};
        call_before_leg(instrumentation, staged_copy_leg::SOURCE, chunk_index, source.stream);
        // The slot may only be overwritten once the chunk that used it last has left it.
        if (chunk_index >= ring.slots()) { destination_done.slot(slot).wait(source.stream); }
        CUCASCADE_CUDA_TRY(to_host.enqueue(source.stream));
        source_done.slot(slot).record(source.stream);
      }
      {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{destination.device_id}};
        call_before_leg(
          instrumentation, staged_copy_leg::DESTINATION, chunk_index, destination.stream);
        source_done.slot(slot).wait(destination.stream);
        CUCASCADE_CUDA_TRY(to_device.enqueue(destination.stream));
        destination_done.slot(slot).record(destination.stream);
      }

      remaining -= chunk_bytes;
      ++chunk_index;
    }

    // The destination stream runs its legs in order and each waited for its source leg, so the
    // last destination leg finishing means every byte has arrived.
    {
      nvtx_range const range{"staged:final"};
      destination_done.slot((chunk_index - 1) % ring.slots()).synchronize();
    }
    if (instrumentation != nullptr) { ++instrumentation->final_syncs; }
  } catch (...) {
    // Copies already enqueued still read the source and write the ring and the destination.
    bool const source_idle = wait_for_enqueued_work(source, source_done.after_failure());
    bool const destination_idle =
      wait_for_enqueued_work(destination, destination_done.after_failure());
    bool const report_failure = instrumentation != nullptr && instrumentation->fail_cleanup.load();
    if (!source_idle || !destination_idle || report_failure) {
      ring.abandon();
      if (instrumentation != nullptr) { ++instrumentation->abandoned_rings; }
      CUCASCADE_LOG_ERROR(
        "staged copy from device {} to device {} failed and its pending work could not be "
        "waited for; {} bytes of pinned host memory are not returned",
        source.device_id,
        destination.device_id,
        ring.bytes());
    }
    throw;
  }
}

std::vector<std::int64_t> read_device_integers(std::span<device_integer_probe const> probes,
                                               staged_copy_endpoint source,
                                               rmm::device_async_resource_ref scratch_mr,
                                               staged_copy_instrumentation* instrumentation)
{
  if (probes.empty()) { return {}; }
  for (auto const& probe : probes) {
    if (probe.address == nullptr ||
        (probe.width != sizeof(std::int32_t) && probe.width != sizeof(std::int64_t))) {
      throw std::invalid_argument(
        "device integer read: a probe needs an address and a width of 4 or 8 bytes");
    }
  }
  require_stream_of_device(source, "source");

  constexpr std::size_t slot_bytes = sizeof(std::int64_t);
  std::vector<std::int64_t> values(probes.size());
  std::vector<std::byte> host(probes.size() * slot_bytes);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source.device_id}};
  cuda::cuda_event gathered;
  rmm::device_buffer scratch(host.size(), source.stream, scratch_mr);
  cuda::device_copy_batch gather;
  gather.reserve(probes.size());
  for (std::size_t i = 0; i < probes.size(); ++i) {
    gather.add(
      static_cast<std::byte*>(scratch.data()) + i * slot_bytes, probes[i].address, probes[i].width);
  }

  try {
    CUCASCADE_CUDA_TRY(gather.enqueue(source.stream));
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      host.data(), scratch.data(), host.size(), cudaMemcpyDeviceToHost, source.stream.get()));
    gathered.record(source.stream);
    nvtx_range const range{"staged:metadata"};
    gathered.synchronize();
  } catch (...) {
    // The scratch buffer and the host bytes are freed when this frame unwinds.
    if (!wait_for_enqueued_work(source, gathered)) {
      CUCASCADE_LOG_ERROR(
        "reading {} integers from device {} failed and the pending copies could not be waited for",
        probes.size(),
        source.device_id);
    }
    throw;
  }
  if (instrumentation != nullptr) { ++instrumentation->metadata_syncs; }

  for (std::size_t i = 0; i < probes.size(); ++i) {
    std::byte const* const bytes = host.data() + i * slot_bytes;
    if (probes[i].width == sizeof(std::int32_t)) {
      std::int32_t narrow = 0;
      std::memcpy(&narrow, bytes, sizeof(narrow));
      values[i] = narrow;
    } else {
      std::memcpy(&values[i], bytes, sizeof(std::int64_t));
    }
  }
  return values;
}

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
