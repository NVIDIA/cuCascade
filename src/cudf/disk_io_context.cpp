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

#include "disk_io.hpp"

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace cucascade {
namespace {

/// Pinned staging each uring reactor claims at start(): the reactor sizes its bounce slots from
/// the resource's block size and takes a fixed 64 MiB budget of them.
constexpr std::size_t uring_staging_bytes_per_reactor = 64UL << 20;

/// Staging block size: the uring reactor's bounce-slot size and its physical I/O granule.
constexpr std::size_t uring_staging_block_size = 1UL << 20;

/**
 * @brief The pinned staging a private uring context owns.
 *
 * A base class of owning_uring_reactor_context so it is constructed before, and destroyed after,
 * the reactor_context that points at it (base-from-member).
 */
class owned_uring_staging {
 protected:
  owned_uring_staging(std::size_t n_reactors, int numa_node)
    : _upstream(numa_node, /*make_portable=*/true),
      _staging(numa_node,
               _upstream,
               std::max<std::size_t>(n_reactors, 1) * uring_staging_bytes_per_reactor,
               std::max<std::size_t>(n_reactors, 1) * uring_staging_bytes_per_reactor,
               uring_staging_block_size,
               uring_staging_bytes_per_reactor / uring_staging_block_size,
               /*initial_pools=*/0)
  {
  }

  memory::numa_region_pinned_host_memory_resource _upstream;
  memory::fixed_size_host_memory_resource _staging;
};

/**
 * @brief A uring reactor_context that owns its pinned staging resource.
 *
 * Every reactor of the pool holds the shared reactor_context and releases its bounce blocks
 * before dropping it, so the staging resource outlives every block taken from it without the
 * ioctx having to know about it.
 */
class owning_uring_reactor_context final : private owned_uring_staging,
                                           public io::uring::uring_reactor::reactor_context {
 public:
  owning_uring_reactor_context(io::uring::config const& cfg, std::size_t n_reactors, int numa_node)
    : owned_uring_staging(n_reactors, numa_node), reactor_context(cfg, &_staging)
  {
  }
};

[[nodiscard]] disk_io_backend resolve_backend(disk_io_backend requested) noexcept
{
  if (requested != disk_io_backend::automatic) { return requested; }
  return detail::gds_is_available() ? disk_io_backend::cufile : disk_io_backend::uring;
}

[[nodiscard]] std::shared_ptr<io::ioctx> build_context(disk_io_context_options const& options)
{
  auto const& cfg = options.io;
  switch (resolve_backend(options.backend)) {
    case disk_io_backend::cufile: {
      auto reactor_ctx = std::make_shared<io::cufile::cufile_reactor::reactor_context>(cfg.cufile);
      return std::make_shared<io::cufile::cufile_ioctx>(cfg.cufile_n_reactors,
                                                        std::move(reactor_ctx));
    }
    case disk_io_backend::kvikio:
      return std::make_shared<io::kvikio_context>(cfg.kvikio, cfg.object_store);
    case disk_io_backend::uring:
    case disk_io_backend::automatic: break;
  }
  auto reactor_ctx = std::make_shared<owning_uring_reactor_context>(
    cfg.uring, cfg.uring_n_reactors, options.staging_numa_node);
  return std::make_shared<io::uring::uring_ioctx>(cfg.uring_n_reactors, std::move(reactor_ctx));
}

}  // namespace

namespace detail {

bool gds_is_available() noexcept
{
  // Cached process-wide by the cuFile layer; the first call opens the cuFile driver once, which
  // only happens when an automatic context is built (the shared default, or an explicit request).
  return io::cufile::gds_available();
}

}  // namespace detail

std::shared_ptr<io::ioctx> make_disk_io_context(disk_io_context_options const& options)
{
  auto context = build_context(options);
  if (!context) {
    throw std::runtime_error("make_disk_io_context: failed to build the io context");
  }
  context->start();
  return context;
}

std::shared_ptr<io::ioctx> default_disk_io_context()
{
  // Function-local so the first use builds it; a weak reference so the disk spaces that adopted
  // it own it, and it is shut down with the last of them rather than during static destruction.
  static std::mutex mutex;
  static std::weak_ptr<io::ioctx> shared_default;

  std::lock_guard<std::mutex> lock(mutex);
  if (auto context = shared_default.lock()) { return context; }
  auto context   = make_disk_io_context();
  shared_default = context;
  return context;
}

}  // namespace cucascade
