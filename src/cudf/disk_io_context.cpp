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
#endif
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <liburing.h>

#include <array>
#include <cstddef>
#include <filesystem>
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
  if (detail::gds_status_flags() != 0) { return disk_io_backend::cufile; }
  return disk_io_backend::uring;
}

void validate(disk_io_context_options const& options, disk_io_backend backend)
{
  auto const& cfg = options.io;
  if (backend == disk_io_backend::uring && cfg.uring_n_reactors == 0) {
    throw std::invalid_argument("make_disk_io_context: io.uring_n_reactors must be at least 1");
  }
  // The cuFile reactors serve host I/O through an io_uring reactor of their own.
  if (uring_setup_errno() != 0) {
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
      // As the scan factory builds it: a reactor per GPU.
      return std::make_shared<io::cufile::cufile_ioctx>(cfg.cufile);
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

unsigned int gds_status_flags() noexcept
{
#ifdef CUCASCADE_HAS_CUFILE
  // Cached process-wide by the cuFile layer; the first call opens the cuFile driver once, which
  // only happens when a backend is chosen automatically (a DISK space that allows GDS adopting
  // its default, default_disk_io_context(), or an explicit automatic request).
  return io::cufile::gds_status_flags();
#else
  return 0;
#endif
}

bool mount_supports_gds([[maybe_unused]] std::string const& mount_path, unsigned int status_flags)
{
  if (status_flags == 0) { return false; }
#ifdef CUCASCADE_HAS_CUFILE
  // Storage discovery describes regular files only, so look at a file created in the mount.
  auto const probe   = memory::generate_disk_file_path(mount_path);
  auto const storage = memory::topology_discovery::discover_storage_for_path(probe);
  std::filesystem::remove(probe);
  return storage.status == memory::path_storage_status::ok &&
         io::cufile::gds_supports_filesystem(
           status_flags, storage.filesystem_type, storage.all_devices_nvme());
#else
  return false;
#endif
}

}  // namespace detail

namespace {

/// The process-shared context of @p backend (uring or cufile), built on first use and cached
/// through a weak reference: see default_disk_io_context().
[[nodiscard]] std::shared_ptr<io::ioctx> shared_disk_io_context(disk_io_backend backend)
{
  // Function-local so the first use builds it; weak references so the disk spaces that adopted
  // a context own it, and it is shut down with the last of them rather than during static
  // destruction.
  static std::mutex mutex;
  static std::array<std::weak_ptr<io::ioctx>, 2> shared;  // [0] uring, [1] cufile

  std::lock_guard<std::mutex> lock(mutex);
  auto& slot = shared[backend == disk_io_backend::cufile ? 1 : 0];
  if (auto context = slot.lock()) { return context; }
  disk_io_context_options options;
  options.backend = backend;
  auto context    = make_disk_io_context(options);
  slot            = context;
  return context;
}

}  // namespace

std::shared_ptr<io::ioctx> make_disk_io_context(disk_io_context_options const& options)
{
  auto const backend = resolve_backend(options.backend);
  auto context       = build_context(options, backend);
  if (!context) {
    throw std::runtime_error("make_disk_io_context: failed to build the io context");
  }
  context->start();
  return context;
}

std::shared_ptr<io::ioctx> default_disk_io_context()
{
  return shared_disk_io_context(resolve_backend(disk_io_backend::automatic));
}

std::shared_ptr<io::ioctx> resolve_disk_io_context(memory::memory_space const& disk_space)
{
  return disk_space.get_or_init_io_context([&disk_space] {
    bool const gds = disk_space.allow_gds() &&
                     detail::mount_supports_gds(std::string(disk_space.get_disk_mount_path()),
                                                detail::gds_status_flags());
    return shared_disk_io_context(gds ? disk_io_backend::cufile : disk_io_backend::uring);
  });
}

}  // namespace cucascade
