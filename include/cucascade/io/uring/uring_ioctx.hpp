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

#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace cucascade::io::uring {

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief io_uring-backed ioctx. Thin specialisation of
 *        @c templated_ioctx<uring_reactor>.
 */
class uring_ioctx : public templated_ioctx<uring_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context
  /// per pool: it carries the per-reactor @c config and the pinned bounce-staging
  /// resource, which must outlive this ioctx).  The ioctx config is sourced from
  /// the reactors themselves — see @c templated_ioctx.
  uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx);

  /// The reactors of one GPU in a per-GPU pool.
  struct device_group {
    int device{-1};  ///< CUDA device id
    /// CPUs of the GPU's NUMA node, which the workers are pinned to; empty leaves
    /// them unpinned.
    std::vector<int> cpus;
    /// Config and the staging resource of the GPU's NUMA node (must outlive this
    /// ioctx).  Every group's resource must have the same block size.
    std::shared_ptr<uring_reactor::reactor_context> ctx;
  };

  /// Build a per-GPU pool: every group gets @p readers_per_group read reactors
  /// and one write reactor, so reads never queue behind writes.  With
  /// `stride = readers_per_group + 1`, group i owns reactors
  /// `[i * stride, (i + 1) * stride)`: its readers, then its writer.  A device
  /// request goes to the least-backlogged reactors of its GPU; host I/O, which
  /// has no NUMA affinity, to the least-backlogged reactors of its kind.
  ///
  /// @ref start starts the readers only.  A writer starts -- and reserves its
  /// pinned staging -- with the first write routed to it, so a pool that only
  /// reads pins no write staging.
  /// @throws std::invalid_argument for no groups, no readers, a negative device
  ///         or staging block sizes that differ between groups.
  explicit uring_ioctx(std::vector<device_group> const& groups, std::size_t readers_per_group = 1);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::uring; }

  void start() override;
  void shutdown() noexcept override;

  /// One gauge snapshot per reactor, in reactor order.  Each call restarts the
  /// reactors' max-inflight and queue-delay windows.  Nothing samples them in
  /// the background; callers that want them poll this.
  [[nodiscard]] std::vector<uring_reactor::gauges> reactor_gauges() noexcept;

  std::vector<uring_reactor*> next_reactor(io_object_type const& object,
                                           std::size_t n_slices,
                                           io_op_type type,
                                           int device_id = -1) override;

 protected:
  /// In a per-GPU pool the slices of one request are routed per GPU, host
  /// slices apart from device ones.
  [[nodiscard]] std::size_t slice_class(io_object_type const& object,
                                        prepared_io_slice const& slice) const override;

 private:
  /// The GPUs of a per-GPU pool, in group order; empty for a flat pool.
  std::vector<int> _devices;
  std::size_t _readers{1};  ///< Read reactors per GPU of a per-GPU pool.
  /// Guards starting the writers of a per-GPU pool against @ref start and @ref shutdown.
  std::mutex _writers_mutex;
  bool _accepts_writes{false};  ///< Started and not shut down.
};

}  // namespace cucascade::io::uring
