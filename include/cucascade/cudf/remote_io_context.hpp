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

#include <cucascade/io/object_store_config.hpp>
#include <cucascade/io/rest/config.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <span>

namespace cucascade {

namespace io {
class ioctx;
}  // namespace io

/**
 * @brief Options for make_remote_io_context().
 *
 * The bucket and prefix belong to the memory space (remote_memory_space_config::uri_prefix);
 * everything about reaching the store belongs here.
 */
struct remote_io_context_options {
  /// Endpoint ("http(s)://host[:port]", nothing else), region, static credentials (access key,
  /// secret key, optional session token), signing mode, and TLS trust (ca_bundle_path,
  /// tls_verify; applied on top of @ref rest, as the io_context registry does). Endpoint, region,
  /// access key and secret key are required.
  io::object_store_config object_store;

  /// The REST transport's static configuration (TLS policy, timeouts, retries, part size
  /// @c write_split_bytes, connection limits, ...).
  io::rest::config rest{};

  /// Reactors (io_uring threads, each with its own connection pool).
  std::size_t n_reactors{2};

  /// Pinned host staging per reactor, in bytes (1 MiB blocks; shared by all reactors of the
  /// context). It holds the S3 parts being assembled for upload and the staging of device reads.
  /// The context's REMOTE converters write and read in windows that fit it, so one batch never
  /// needs more than it, and concurrent conversions are admitted against it. Requirements:
  /// at least 1 MiB, and n_reactors x staging_bytes_per_reactor >= 2 x the part size
  /// (rest.write_split_bytes rounded up to 1 MiB).
  std::size_t staging_bytes_per_reactor{256UL << 20};

  /// NUMA node the staging is bound to; -1 leaves it unbound. The staging is always allocated
  /// portable, so it serves every GPU.
  int staging_numa_node{-1};
};

/**
 * @brief Build and start an io context for REMOTE memory spaces (S3 or an S3-compatible store).
 *
 * The returned context is a started REST io context (io_context_type::restful, supports host and
 * device writes) that owns its pinned staging, which lives exactly as long as its reactors. It
 * arms no prefetching cache. Pass it to a REMOTE space through
 * remote_memory_space_config::io_context, reservation_manager_configurator::add_remote_space() or
 * memory_space::set_io_context(); one context may serve any number of REMOTE spaces (of buckets
 * the credentials can reach). Unlike DISK there is no default context: credentials are required.
 *
 * @param options Endpoint, credentials, transport configuration and staging.
 * @return The started context; reactors are shut down and joined when the last owner (spaces,
 *         REMOTE representations, conversions in flight) releases it.
 * @throws std::invalid_argument if the endpoint, region, access key or secret key is empty, the
 *         endpoint is not exactly `http(s)://host[:port]` (userinfo, path, query, fragment and
 *         whitespace are rejected), n_reactors is 0, or the staging is too small (see
 *         remote_io_context_options::staging_bytes_per_reactor).
 * @throws std::system_error if the reactors cannot start (e.g. io_uring is forbidden in this
 *         process, EPERM under a container seccomp profile).
 *
 * @note The same teardown-order note as for make_disk_io_context() applies: release the context
 *       and everything holding it (memory spaces, reservation managers, REMOTE representations)
 *       before the process tears down the CUDA runtime.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> make_remote_io_context(
  remote_io_context_options const& options);

/**
 * @brief Wait until every REMOTE object removal issued so far has settled.
 *
 * Destroying a remote_data_representation removes its object asynchronously: a process-wide
 * removal queue (one thread, never a reactor thread) issues the DeleteObject and keeps the io
 * context that issues it alive until it settled, then releases it on its own thread. So removals
 * survive the memory spaces and reservation managers that held their contexts, and a
 * representation may be released anywhere, a reactor completion callback included, without
 * blocking. The queue is created by the first make_remote_io_context() (or REMOTE write).
 *
 * Call this (or the scoped, bounded overload below) before tearing the CUDA runtime down. The
 * queue also drains itself from a std::atexit handler, but a representation released after that
 * handler ran (e.g. by a late static destructor) is not guaranteed to be removed, and its context
 * may outlive the CUDA runtime: embedders must drain explicitly. Also useful to observe the
 * removals in a test.
 *
 * Limits: this zero-argument form has no deadline (each removal is bounded by the transport's
 * request timeout and retries; a throttled retry honours Retry-After, capped at 30 s), and it
 * waits for the removals of every embedder in the process; embedders should tear down with the
 * scoped, bounded form. A removal whose context was already shut down runs as a blocking call on
 * the queue's thread; the queue is unbounded; forking after the queue exists is not supported.
 *
 * @warning Must not be called on a reactor thread of an io context with removals outstanding:
 *          the drain would block the very thread that has to settle them.
 */
void drain_remote_object_removals();

/// The outcome of drain_remote_object_removals(contexts, timeout).
struct remote_removal_drain_result {
  std::size_t abandoned{0};  ///< removals of the contexts cancelled or never issued
  bool timed_out{false};     ///< the deadline expired and the contexts were shut down
};

/**
 * @brief Wait, for at most @p timeout, until every REMOTE object removal submitted through one of
 *        @p contexts has settled and released its context; at the deadline, cancel them.
 *
 * Only the removals of @p contexts are waited for: other embedders' removals are neither waited
 * for nor cancelled. If they all settle in time, nothing is cancelled and the contexts stay
 * usable. At the deadline:
 *  - removals still queued for these contexts are never issued;
 *  - each context is shut down with io::ioctx::shutdown(), which settles its in-flight, retrying
 *    and queued calls (a removal whose 204 or 404 already arrived still counts as done);
 *  - all their entries are released on the calling thread.
 * The objects of the cancelled removals are left to the bucket's lifecycle rule. The deadline
 * never depends on the progress of the queue's own thread.
 *
 * @param contexts The io contexts whose removals to drain (null and duplicate entries are
 *        skipped). The caller's references keep them alive through the call.
 * @param timeout How long to wait before cancelling; 0 means do not wait.
 * @return How many removals were cancelled or never issued, and whether the deadline expired
 *         (then the contexts were shut down). Without a removal queue (nothing was ever
 *         submitted) the result is {0, false}. If the call cannot even allocate its bookkeeping,
 *         it logs an error and returns {0, true} without cancelling anything.
 *
 * @pre Nothing submits further removals through @p contexts (release their spaces and REMOTE
 *      representations first); the contexts of concurrent scoped drains are disjoint; never call
 *      it on a reactor thread of one of the contexts, or from a completion callback of one.
 * @note If one of the contexts was already shut down by its owner, a removal of it that the
 *       queue's thread is issuing at that moment runs as a blocking private-ring call, and the
 *       drain waits for it.
 */
[[nodiscard]] remote_removal_drain_result drain_remote_object_removals(
  std::span<std::shared_ptr<io::ioctx> const> contexts, std::chrono::milliseconds timeout) noexcept;

}  // namespace cucascade
