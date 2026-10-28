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
#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/platform.hpp>
#endif
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <liburing.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
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
      // The first argument only tags the resource's HOST memory_space_id (a HOST space's id is
      // its NUMA node); this resource belongs to no memory space.
      _staging(numa_node,
               _upstream,
               n_reactors * uring_staging_bytes_per_reactor,
               n_reactors * uring_staging_bytes_per_reactor,
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

/**
 * @brief 0 when this process can set up io_uring rings, otherwise the (positive) errno of the
 *        failed setup. Probed once with a one-entry ring.
 *
 * io_uring can be forbidden even on a kernel that has it: container seccomp profiles (Docker 25+),
 * gVisor and kernel.io_uring_disabled make io_uring_setup fail with EPERM / ENOSYS. A uring
 * reactor whose ring cannot be set up dies in its worker and fails every request, so the DISK tier
 * checks up front instead.
 */
[[nodiscard]] int uring_setup_errno() noexcept
{
  static int const error = [] {
    io_uring ring{};
    int const rc = io_uring_queue_init(1, &ring, 0);
    if (rc == 0) {
      io_uring_queue_exit(&ring);
      return 0;
    }
    return -rc;
  }();
  return error;
}

[[nodiscard]] disk_io_backend resolve_backend(disk_io_backend requested)
{
  if (requested != disk_io_backend::automatic) { return requested; }
  if (detail::gds_is_available()) { return disk_io_backend::cufile; }
  if (uring_setup_errno() == 0) { return disk_io_backend::uring; }
#ifdef CUCASCADE_HAS_CUFILE
  // The cuFile reactor needs no io_uring ring: without GDS it serves every transfer with POSIX
  // I/O (staged through its own pinned scratch for device data).
  CUCASCADE_LOG_WARN(
    "disk tier: io_uring is unavailable in this process ({}); DISK spaces without an explicit io "
    "context use the cuFile reactor's POSIX path",
    std::strerror(uring_setup_errno()));
  return disk_io_backend::cufile;
#else
  throw std::system_error(uring_setup_errno(),
                          std::generic_category(),
                          "make_disk_io_context: io_uring is unavailable in this process and "
                          "this build has no cuFile backend to fall back to");
#endif
}

void validate(disk_io_context_options const& options, disk_io_backend backend)
{
  auto const& cfg = options.io;
  if (backend == disk_io_backend::uring && cfg.uring_n_reactors == 0) {
    throw std::invalid_argument("make_disk_io_context: io.uring_n_reactors must be at least 1");
  }
#ifdef CUCASCADE_HAS_CUFILE
  if (backend == disk_io_backend::cufile && cfg.cufile_n_reactors == 0) {
    throw std::invalid_argument("make_disk_io_context: io.cufile_n_reactors must be at least 1");
  }
#endif
  if (backend == disk_io_backend::uring && uring_setup_errno() != 0) {
    throw std::system_error(uring_setup_errno(),
                            std::generic_category(),
                            "make_disk_io_context: io_uring is unavailable in this process "
                            "(io_uring_queue_init)");
  }
}

[[nodiscard]] std::shared_ptr<io::ioctx> build_context(disk_io_context_options const& options,
                                                       disk_io_backend backend)
{
  auto const& cfg = options.io;
  validate(options, backend);
  switch (backend) {
    case disk_io_backend::cufile: {
#ifdef CUCASCADE_HAS_CUFILE
      auto reactor_ctx = std::make_shared<io::cufile::cufile_reactor::reactor_context>(cfg.cufile);
      return std::make_shared<io::cufile::cufile_ioctx>(cfg.cufile_n_reactors,
                                                        std::move(reactor_ctx));
#else
      throw std::invalid_argument(
        "make_disk_io_context: the cufile backend was requested but this build has no cufile.h");
#endif
    }
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
#ifdef CUCASCADE_HAS_CUFILE
  // Cached process-wide by the cuFile layer; the first call opens the cuFile driver once, which
  // only happens when an automatic context is built (the shared default, or an explicit request).
  return io::cufile::gds_available();
#else
  return false;
#endif
}

}  // namespace detail

std::shared_ptr<io::ioctx> make_disk_io_context(disk_io_context_options const& options)
{
  auto const backend = resolve_backend(options.backend);
  auto context       = build_context(options, backend);
  if (!context) {
    throw std::runtime_error("make_disk_io_context: failed to build the io context");
  }
  try {
    context->start();
  } catch (std::system_error const& error) {
#ifdef CUCASCADE_HAS_CUFILE
    // The one-entry probe passed, but a reactor could not set up its own ring
    // (larger depth, setup flags, memlock limits): for automatic that is still
    // "io_uring unavailable" -- fall back to the cuFile reactor's POSIX path.
    if (options.backend == disk_io_backend::automatic && backend == disk_io_backend::uring) {
      CUCASCADE_LOG_WARN(
        "disk tier: io_uring reactors could not start ({}); DISK spaces without an explicit io "
        "context use the cuFile reactor's POSIX path",
        error.what());
      context.reset();
      context = build_context(options, disk_io_backend::cufile);
      context->start();
      return context;
    }
#endif
    throw;
  }
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
