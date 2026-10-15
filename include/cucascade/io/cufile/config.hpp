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

#include <cucascade/exec/config.hpp>
#include <cucascade/io/types.hpp>

#include <kvikio/compat_mode.hpp>

#include <cstddef>
#include <optional>

namespace cucascade::io::cufile {

/**
 * @brief Tunables for the cuFile / GPUDirect Storage local-file reactor.
 *
 * This backend uses the same mechanism kvikIO uses internally
 * (@c cuFileHandleRegister / @c cuFileReadAsync) but drives it directly from a
 * cuCascade reactor rather than through @c kvikio::FileHandle.  The pure-device
 * path is stream-ordered against the caller's CUDA stream: cuFile DMAs from NVMe
 * straight into GPU memory with no host-side bounce of its own.
 *
 * Host-only reads and mixed host+device requests fall back to POSIX @c pread on
 * the buffered fd; the reactor exists for the device fast path and treats
 * everything else as a correctness-preserving fallback.
 *
 * @note This backend is available only in a build with kvikIO
 *       (@c CUCASCADE_HAS_KVIKIO, i.e. @c CUCASCADE_BUILD_CUDF), because it
 *       reaches libcufile through kvikIO's lazily-loading @c cuFileAPI shim.
 */
struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.  Local NVMe /
  /// GDS saturates at modest queue depth, so one scan per pipeline thread is
  /// enough to keep the decoders fed without over-committing the device.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  /// Whether the config named @c n_max_concurrent_scans explicitly.  Same role as
  /// in the uring config: distinguish an explicit value equal to the struct
  /// default from a defaulted one, so a derived default (which follows the
  /// pipeline width) can override the latter but not the former.
  bool n_max_concurrent_scans_explicit{false};

  /// Maximum physical operations (cuFile submissions plus fallback host-to-device
  /// copies) in flight per reactor.  The reactor stops dispatching new slices
  /// once this many are awaiting their CUDA event.
  std::size_t max_inflight{32};

  /// Physical operations below this size use POSIX pread on the buffered fd
  /// instead of cuFile.  Matches kvikIO's @c gds_threshold (env
  /// @c KVIKIO_GDS_THRESHOLD, default 1 MiB): small reads pay more in cuFile
  /// bookkeeping than they save in DMA.
  std::size_t gds_threshold{1UL << 20};

  /// Whether the reactor opens the O_DIRECT fd needed to register with cuFile.
  /// If false, cuFile registration is skipped and every read falls back to
  /// POSIX -- effectively disabling the reactor's fast path.  Kept as a knob so
  /// a filesystem that rejects O_DIRECT can still exercise the reactor
  /// end-to-end via the fallback.
  bool use_odirect{true};

  /// Per-file cuFile compat mode.  @c OFF enforces cuFile/GDS (opening a file
  /// whose handle cannot be registered throws), @c ON forces POSIX at handle
  /// level, @c AUTO tries cuFile and falls back to POSIX at registration time.
  /// Unset leaves it at kvikIO's @c defaults::compat_mode() (which honours
  /// @c KVIKIO_COMPAT_MODE).
  std::optional<kvikio::CompatMode> compat_mode;

  /// How long the worker sleeps between completion polls while operations are
  /// in flight and it has nothing to submit.  A completion is only noticed at a
  /// poll, so this bounds the latency a finished read adds; it is not a wait the
  /// reactor takes when it has work to dispatch.
  std::size_t poll_interval_us{200};

  /// Bounded service for the low-priority tier.  Priority is otherwise strict (a
  /// waiting high request is taken first and parks an active low one), so a
  /// demand read that waits on in-flight low work could starve behind a
  /// continuous stream of high requests.  After this many high requests have
  /// started while low work was waiting, the reactor grants the low tier one
  /// turn (at least @c max_inflight slices) even though high work is waiting.
  /// 0 restores strict priority.
  std::size_t low_service_interval{4};

  /// cuFile can only DMA whole pages, so a read is widened to a page boundary
  /// either way -- reporting it lets the caller align once, up front, instead of
  /// every layer rediscovering it.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// A local read is a syscall against NVMe, so bridging is only worth it when
  /// the bridged bytes cost less than the extra request -- one page.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }
};

}  // namespace cucascade::io::cufile
