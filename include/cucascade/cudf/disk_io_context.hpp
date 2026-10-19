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

/**
 * @brief Which io stack serves a DISK memory space's files.
 */
enum class disk_io_backend : std::uint8_t {
  automatic,  ///< cufile when GPUDirect Storage is really available (not compat mode), else uring
  uring,      ///< io_uring reactors; device data is staged through private pinned host memory
  cufile,     ///< cuFile reactors (GDS); forced even when cuFile only runs in compatibility mode
  kvikio,     ///< kvikIO's FileHandle (GDS or its own bounce buffers)
};

/**
 * @brief Options for make_disk_io_context().
 *
 * Configuration is static at the io-context level: the DISK converters pass no per-request knobs,
 * so everything that tunes the disk tier's I/O lives here (and inside the io stack's own config).
 */
struct disk_io_context_options {
  /// Which backend to build; automatic picks cufile iff GDS is available, else uring.
  disk_io_backend backend{disk_io_backend::automatic};

  /// The io stack's own static configuration. Used: uring_n_reactors / uring for uring,
  /// cufile_n_reactors / cufile for cufile, kvikio / object_store for kvikio. Ignored: backend
  /// (see @ref backend) and cache (the DISK tier never arms a prefetching cache).
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
 * @param options Backend selection and its static configuration.
 * @return The started context; reactors are shut down and joined when the last owner releases it.
 * @throws std::runtime_error (or the backend's exception) if the context cannot be built.
 *
 * @note Release every context (and every memory space or reservation manager holding one) before
 *       the process tears down the CUDA runtime and the cuFile driver: shutting a context down
 *       synchronizes CUDA events, frees pinned staging and closes cuFile handles. Objects with
 *       static storage duration that were constructed before CUDA was first used (e.g. a
 *       namespace-scope `std::shared_ptr<memory_reservation_manager>`) are destroyed after that
 *       teardown; reset them explicitly (or from an `std::atexit` handler registered after the
 *       context was built) instead.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> make_disk_io_context(
  disk_io_context_options const& options = {});

/**
 * @brief The process-shared default context of DISK memory spaces configured without one.
 *
 * Built on first use with make_disk_io_context() and the io_config defaults (automatic backend:
 * cuFile when GPUDirect Storage is available, otherwise uring), serving both the host and the
 * device paths. It is cached through a weak reference, so every space that adopts it shares one
 * reactor set; it is shut down when the last space holding it is destroyed and rebuilt on a later
 * use. The same teardown-order note as for make_disk_io_context() applies to the spaces holding it.
 *
 * @return The shared default context (started).
 */
[[nodiscard]] std::shared_ptr<io::ioctx> default_disk_io_context();

}  // namespace cucascade
