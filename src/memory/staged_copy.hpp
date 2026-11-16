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

/**
 * @file
 * Staged copies: moving bytes from one GPU to another through pinned host memory, for device pairs
 * that cannot copy to each other directly.
 *
 * A staged copy has two legs. The source leg copies device memory to host memory on a stream of
 * the source device; the destination leg copies that host memory to device memory on a stream of
 * the destination device. The bytes travel in chunks through a ring of host slots, so the source
 * leg of one chunk runs while the destination leg of an earlier chunk is still in flight.
 *
 * The two streams are ordered against each other only by events: a destination leg waits for the
 * event its source leg recorded, and a source leg that reuses a slot waits for the event of the
 * destination leg that last read the slot. Every wait names an event recorded earlier, so staged
 * copies in any mix of directions cannot deadlock, and none of them needs a stream to itself.
 */

#pragma once

#include <cucascade/cuda/stream.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace cucascade {
namespace memory {
namespace detail {

/** @brief One run of bytes to move from the source device to the destination device. */
struct staged_copy_segment {
  void const* source{nullptr};  ///< First byte, in the source device's memory
  void* destination{nullptr};   ///< First byte, in the destination device's memory
  std::size_t bytes{0};         ///< Number of bytes; an empty segment is skipped
};

/** @brief One side of a staged copy: a device and the stream of that device its leg runs on. */
struct staged_copy_endpoint {
  int device_id{-1};
  ::cuda::stream_ref stream{cudaStream_t{nullptr}};
};

/** @brief Upper bounds on the host memory one staged copy borrows. */
struct staged_copy_ring_settings {
  static constexpr std::size_t default_chunk_bytes = std::size_t{4} << 20;
  static constexpr std::size_t default_depth       = 4;
  static constexpr std::size_t max_chunk_bytes     = std::size_t{1} << 30;
  static constexpr std::size_t max_depth           = 64;

  std::size_t chunk_bytes{default_chunk_bytes};  ///< Most bytes in one chunk
  std::size_t depth{default_depth};              ///< Most chunks in flight at once
};

/** @brief Shape of a ring: @p slots slots of @p blocks_per_slot host blocks each. */
struct staged_copy_ring_geometry {
  std::size_t slots{0};
  std::size_t blocks_per_slot{0};

  [[nodiscard]] bool operator==(staged_copy_ring_geometry const&) const = default;
};

/**
 * @brief The ring a copy of @p total_bytes asks for first.
 *
 * With block size `b`, chunk capacity `s` (rounded up to whole blocks) and depth `d`, the ring has
 * `n = min(d, ceil(total / b))` slots of `min(s, round_up(ceil(total / n), b))` bytes. A ring is
 * therefore never larger than `d * s`, nor than the payload rounded up to blocks plus `n - 1`
 * blocks, so a small payload borrows about its own size.
 *
 * @param total_bytes Payload size; must not be zero
 * @param block_bytes Size of one host block; must not be zero
 * @param settings Chunk capacity and depth
 * @throws std::invalid_argument if an argument is zero, a setting exceeds its maximum, or the ring
 *         size overflows
 */
[[nodiscard]] staged_copy_ring_geometry plan_staged_copy_ring(
  std::size_t total_bytes, std::size_t block_bytes, staged_copy_ring_settings const& settings);

/**
 * @brief Step @p geometry to the next smaller ring, for a copy whose ring could not be allocated.
 *
 * Slots are halved (in whole blocks) down to one block each, then the ring drops to two slots, then
 * to one: 4x4 -> 4x2 -> 4x1 -> 2x1 -> 1x1 for the default settings and 1 MiB blocks.
 *
 * @return false if @p geometry is already one slot of one block, and is left unchanged
 */
[[nodiscard]] bool shrink_staged_copy_ring(staged_copy_ring_geometry& geometry) noexcept;

/** @brief Which half of a chunk's journey a hook is called for. */
enum class staged_copy_leg {
  SOURCE,      ///< Device to host, on the source stream
  DESTINATION  ///< Host to device, on the destination stream
};

/**
 * @brief Hooks and counters for tests; production callers pass none.
 *
 * One object may be shared by concurrent copies. @ref before_leg must be thread-safe, must not
 * start a staged copy and must not block on another one.
 */
struct staged_copy_instrumentation {
  /// Called before each leg of each chunk is enqueued, with that leg's device current. An
  /// exception it throws fails the copy as an enqueue failure would.
  std::function<void(staged_copy_leg leg, std::size_t chunk_index, ::cuda::stream_ref stream)>
    before_leg;

  /// Report the wait after a failed copy as failed, after really waiting, so the ring is abandoned
  /// as it would be if the wait had failed.
  std::atomic<bool> fail_cleanup{false};

  std::atomic<std::size_t> readiness_syncs{0};  ///< Host waits for a source with an unknown writer
  std::atomic<std::size_t> metadata_syncs{0};   ///< Host waits for sizes read from the device
  std::atomic<std::size_t> final_syncs{0};      ///< Host waits for the last chunk of a copy
  std::atomic<std::size_t> rings{0};            ///< Rings acquired
  std::atomic<std::size_t> last_ring_bytes{0};  ///< Size of the most recently acquired ring
  std::atomic<std::size_t> abandoned_rings{0};  ///< Rings never given back
};

/**
 * @brief Copy @p segments from @p source to @p destination through pinned host memory, and wait
 * until every byte has arrived.
 *
 * The ring comes from the HOST pool registered for the destination device's NUMA node (any
 * registered pool otherwise), and from a temporary pinned allocation when no pool is registered.
 * If the pool cannot supply the ring, smaller rings are tried, down to one block; only then does
 * the copy fail. Nothing is copied and no host memory is taken when the segments hold no bytes.
 *
 * The calling thread waits once, for the last chunk. On return, the ring has been given back and
 * no work enqueued by this call is pending on either stream. The caller's current device is
 * restored on every exit.
 *
 * If the copy fails after enqueueing work, it waits for that work before giving the ring back, and
 * rethrows the original error. If that wait fails too, the ring is abandoned instead of given back
 * (its blocks stay allocated) because device work may still write to it; the same work may still
 * read the source and write the destination.
 *
 * @pre The source bytes are final in the order of @p source's stream: whatever wrote them is
 *      enqueued on that stream or the stream already waits for it.
 * @pre The destination buffers were allocated on @p destination's stream.
 * @pre The source and destination buffers, and the HOST pool, outlive the call.
 *
 * @param segments The byte runs to move, in the order they are sent
 * @param source Source device and the stream its legs run on
 * @param destination Destination device and the stream its legs run on
 * @param settings Bounds for the ring
 * @param instrumentation Test hooks and counters, or null
 * @throws std::invalid_argument if a non-empty segment has a null address, a stream does not
 *         belong to its endpoint's device, or @p settings is invalid
 * @throws rmm::out_of_memory if not even one host block is available
 * @throws cucascade::cuda_error if a CUDA runtime call fails
 */
void staged_copy(std::span<staged_copy_segment const> segments,
                 staged_copy_endpoint source,
                 staged_copy_endpoint destination,
                 staged_copy_ring_settings const& settings    = {},
                 staged_copy_instrumentation* instrumentation = nullptr);

/** @brief Where to read one signed integer of @p width bytes (4 or 8) from device memory. */
struct device_integer_probe {
  void const* address{nullptr};
  std::size_t width{0};
};

/**
 * @brief Read one integer per probe from @p source's device with a single wait of the calling
 * thread, however many probes there are.
 *
 * The values are gathered on the device into a scratch buffer, which is then copied to the host in
 * one piece. An empty @p probes returns at once without touching the device.
 *
 * @pre The probed values are final in the order of @p source's stream, as for staged_copy().
 *
 * @param probes Addresses in the source device's memory
 * @param source Device to read from and the stream to read on
 * @param scratch_mr Resource of that device for the scratch buffer
 * @param instrumentation Test counters, or null
 * @return The values, sign-extended, in the order of @p probes
 * @throws std::invalid_argument if a probe has a null address or a width other than 4 or 8, or the
 *         stream does not belong to the device
 * @throws cucascade::cuda_error if a CUDA runtime call fails
 */
[[nodiscard]] std::vector<std::int64_t> read_device_integers(
  std::span<device_integer_probe const> probes,
  staged_copy_endpoint source,
  rmm::device_async_resource_ref scratch_mr,
  staged_copy_instrumentation* instrumentation = nullptr);

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
