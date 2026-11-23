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

#pragma once

#include <cucascade/io/config.hpp>

#include <cstdint>
#include <memory>

namespace cucascade {

namespace io {
class ioctx;
}  // namespace io

namespace memory {
class memory_space;
}  // namespace memory

/**
 * @brief Which io stack serves a DISK memory space's files.
 */
enum class disk_io_backend : std::uint8_t {
  automatic,  ///< cufile when GPUDirect Storage is really available (not compat mode), else uring
  uring,      ///< io_uring reactors; device data is staged through private pinned host memory
  cufile,     ///< cuFile reactors (GDS), one per GPU; forced even when cuFile only runs in
              ///< compatibility mode (it fails to start when libcufile is absent)
};

/**
 * @brief Options for make_disk_io_context().
 *
 * Configuration is static at the io-context level: the DISK converters pass no per-request knobs,
 * so everything that tunes the disk tier's I/O lives here (and inside the io stack's own config).
 */
struct disk_io_context_options {
  /// Which backend to build (see disk_io_backend::automatic for how automatic resolves).
  disk_io_backend backend{disk_io_backend::automatic};

  /// The io stack's own static configuration. Used: uring_n_reactors / uring for uring,
  /// cufile for cufile. Ignored: backend (see @ref backend) and cache (the
  /// DISK tier never arms a prefetching cache).
  io::io_config io{};

  /// NUMA node the uring context's private pinned staging is bound to; -1 leaves it unbound.
  /// The staging is always allocated portable, so it serves every GPU.
  int staging_numa_node{-1};
};

/**
 * @brief Build and start an io context suitable for DISK memory spaces.
 *
 * The returned context owns everything it needs and is already started: the uring backend owns a
 * private pinned staging resource (uring_n_reactors x 64 MiB) that lives exactly as long as its
 * reactors. Pass it to a disk space through disk_memory_space_config::io_context,
 * reservation_manager_configurator::set_disk_mounting_point() or memory_space::set_io_context().
 * One context may serve any number of disk spaces.
 *
 * Resolving disk_io_backend::automatic probes the platform once per process: it opens the cuFile
 * driver to ask whether GPUDirect Storage is really available (cuFile then writes its
 * `cufile.log` into the working directory unless cufile.json / CUFILE_LOGFILE_PATH points it
 * elsewhere). Every backend needs io_uring (the cuFile reactors serve host I/O through it), which
 * is checked with a one-entry ring.
 *
 * @param options Backend selection and its static configuration.
 * @return The started context; reactors are shut down and joined when the last owner releases it.
 * @throws std::invalid_argument if the selected backend is configured with zero reactors.
 * @throws std::system_error if this process may not use io_uring (e.g. EPERM under a container
 *         seccomp profile).
 * @throws std::runtime_error (or the backend's exception) if the context cannot be built.
 *
 * @note Prefer to release every context (and every memory space or reservation manager holding
 *       one) before the process tears down the CUDA runtime and the cuFile driver: shutting a
 *       context down synchronizes CUDA events and frees pinned staging. Objects with static storage
 *       duration that were constructed before CUDA was first used (e.g. a namespace-scope
 *       `std::shared_ptr<memory_reservation_manager>`) are destroyed after that teardown. Release
 *       builds tolerate it (the failing CUDA calls' errors are discarded); debug builds assert on
 *       them, as they already do for HOST memory spaces. Reset such objects explicitly (or from an
 *       `std::atexit` handler registered after the context was built) to stay clean in both.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> make_disk_io_context(
  disk_io_context_options const& options = {});

/**
 * @brief The process-shared default context for callers that have no DISK memory space at hand.
 *
 * One of the two shared contexts DISK spaces adopt (see resolve_disk_io_context()): the cuFile
 * one when GPUDirect Storage is really available on this host, else the uring one
 * (disk_io_backend::automatic). It does not look at any mount, so prefer
 * resolve_disk_io_context() for a space.
 *
 * Each shared context is built on first use with make_disk_io_context() and the io_config
 * defaults, serving both the host and the device paths. It is cached through a weak reference, so
 * every space that adopts it shares one reactor set; it is shut down when the last space holding
 * it is destroyed and rebuilt on a later use. The same teardown-order note as for
 * make_disk_io_context() applies to the spaces holding it. The cache itself (a function-local
 * mutex and weak references) is destroyed during static destruction, so do not start a DISK
 * conversion from a static destructor.
 *
 * @return The shared default context (started).
 */
[[nodiscard]] std::shared_ptr<io::ioctx> default_disk_io_context();

/**
 * @brief The io context of a DISK memory space, adopting a process-shared default first when the
 *        space has none yet. What the DISK converters use.
 *
 * A space without a context adopts the shared cuFile context when it allows GDS
 * (disk_memory_space_config::allow_gds, the default) and its mount is on a filesystem the cuFile
 * driver serves through GPUDirect Storage; otherwise the shared uring context. The mount check
 * runs once per space, on this first use: it opens the cuFile driver, and when the driver has a
 * GDS mode it creates and removes one probe file in the mount to find its filesystem and drives.
 * A space with allow_gds off never opens the cuFile driver. The check is a prediction (see
 * io::cufile::gds_supports_filesystem()): on a mount that passes it cuFile can still serve a file
 * in compatibility mode, which works but is slower than uring.
 *
 * Hold the returned reference for the whole operation, so a concurrent
 * memory_space::set_io_context() cannot tear the context down underneath it.
 *
 * @param disk_space A DISK-tier memory space.
 * @return The space's context (started); never null.
 * @throws std::logic_error if @p disk_space is not a DISK-tier memory space.
 * @throws whatever make_disk_io_context() throws when a shared context has to be built.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> resolve_disk_io_context(
  memory::memory_space const& disk_space);

}  // namespace cucascade
