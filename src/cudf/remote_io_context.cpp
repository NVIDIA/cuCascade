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

#include "remote_io.hpp"

#include <cucascade/cudf/remote_io_context.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <liburing.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace cucascade {
namespace {

/// Staging block size: the pinned blocks part buffers and device-read staging are cut from.
constexpr std::size_t staging_block_size = std::size_t{1} << 20;

[[nodiscard]] std::size_t round_up(std::size_t value, std::size_t unit) noexcept
{
  return (value + unit - 1) / unit * unit;
}

/**
 * @brief The pinned staging a REMOTE context owns.
 *
 * A base class of owning_rest_reactor_context so it is constructed before, and destroyed after,
 * the reactor_context that points at it (base-from-member, as for the DISK uring context).
 */
class owned_rest_staging {
 protected:
  owned_rest_staging(std::size_t staging_bytes, int numa_node)
    : _upstream(numa_node, /*make_portable=*/true),
      // The first argument only tags the resource's HOST memory_space_id; this resource belongs
      // to no memory space.
      _staging(numa_node,
               _upstream,
               staging_bytes,
               staging_bytes,
               staging_block_size,
               staging_bytes / staging_block_size,
               /*initial_pools=*/0)
  {
  }

  memory::numa_region_pinned_host_memory_resource _upstream;
  memory::fixed_size_host_memory_resource _staging;
};

/// A REST reactor_context that owns its pinned staging (every reactor holds the shared context
/// and releases its staging blocks before dropping it).
class owning_rest_reactor_context final : private owned_rest_staging,
                                          public io::rest::rest_reactor::reactor_context {
 public:
  owning_rest_reactor_context(io::rest::config cfg,
                              std::shared_ptr<io::rest::request_authorizer> authorizer,
                              std::size_t staging_bytes,
                              int numa_node)
    : owned_rest_staging(staging_bytes, numa_node),
      reactor_context(std::move(cfg), std::move(authorizer), &_staging)
  {
  }
};

[[noreturn]] void reject(std::string const& why)
{
  throw std::invalid_argument("make_remote_io_context: " + why);
}

/// The endpoint must be exactly `http(s)://host[:port]`: no userinfo (credentials never travel
/// in it), path, query, fragment or whitespace.
void validate_endpoint(std::string_view endpoint)
{
  auto const sep = endpoint.find("://");
  if (sep == std::string_view::npos) {
    reject("object_store.endpoint must be http(s)://host[:port]");
  }
  std::string scheme{endpoint.substr(0, sep)};
  for (auto& c : scheme) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (scheme != "http" && scheme != "https") {
    reject("object_store.endpoint scheme must be http or https");
  }
  auto const authority = endpoint.substr(sep + 3);
  if (authority.empty()) { reject("object_store.endpoint has no host"); }
  for (char const c : authority) {
    auto const u = static_cast<unsigned char>(c);
    if (std::isspace(u) != 0 || std::iscntrl(u) != 0) {
      reject("object_store.endpoint contains whitespace or control characters");
    }
    if (c == '@') { reject("object_store.endpoint must not carry userinfo"); }
    if (c == '/' || c == '?' || c == '#') {
      reject("object_store.endpoint must be http(s)://host[:port] (no path, query or fragment)");
    }
  }
  if (authority.front() == ':') { reject("object_store.endpoint has no host"); }
}

void validate(remote_io_context_options const& options)
{
  auto const& os = options.object_store;
  if (os.endpoint.empty()) { reject("object_store.endpoint is required"); }
  validate_endpoint(os.endpoint);
  if (os.region.empty()) { reject("object_store.region is required"); }
  if (os.access_key.empty() || os.secret_key.empty()) {
    reject("object_store.access_key and object_store.secret_key are required");
  }
  if (options.n_reactors == 0) { reject("n_reactors must be at least 1"); }
  if (options.staging_bytes_per_reactor < staging_block_size) {
    reject("staging_bytes_per_reactor must be at least 1 MiB");
  }
  auto const part =
    round_up(std::max<std::size_t>(options.rest.write_split_bytes, 1), staging_block_size);
  auto const total = options.n_reactors * (options.staging_bytes_per_reactor / staging_block_size) *
                     staging_block_size;
  if (total < 2 * part) {
    reject("the staging (n_reactors x staging_bytes_per_reactor = " + std::to_string(total) +
           " bytes) must hold at least two parts of rest.write_split_bytes (" +
           std::to_string(part) + " bytes each)");
  }
}

/**
 * @brief Fail now when this process cannot set up io_uring rings (container seccomp profiles,
 *        gVisor, kernel.io_uring_disabled): a REST reactor whose ring cannot be set up dies in its
 *        worker and fails every request instead. Probed once with a one-entry ring.
 */
void require_io_uring()
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
  if (error != 0) {
    throw std::system_error(error,
                            std::generic_category(),
                            "make_remote_io_context: io_uring is unavailable in this process");
  }
}

}  // namespace

std::shared_ptr<io::ioctx> make_remote_io_context(remote_io_context_options const& options)
{
  validate(options);
  require_io_uring();
  // The queue that removes REMOTE objects (and owns contexts until their removals settled) is
  // created here, on a caller's thread, rather than lazily from a representation's destructor.
  static_cast<void>(detail::removal_queue::instance());
  auto authorizer = io::rest::s3::make_authorizer(options.object_store);
  if (!authorizer) { reject("the object store is not configured"); }

  auto cfg = options.rest;
  // The object store owns the endpoint's TLS trust (as in the io_context registry): the
  // connections verify against the CA bundle / policy the authorizer signs for.
  cfg.ca_bundle_path = options.object_store.ca_bundle_path;
  cfg.tls_verify     = options.object_store.tls_verify;

  auto const staging_bytes = options.n_reactors *
                             (options.staging_bytes_per_reactor / staging_block_size) *
                             staging_block_size;
  auto reactor_ctx = std::make_shared<owning_rest_reactor_context>(
    std::move(cfg), std::move(authorizer), staging_bytes, options.staging_numa_node);
  auto context = std::make_shared<detail::remote_rest_ioctx>(
    options.n_reactors, std::move(reactor_ctx), staging_bytes, staging_block_size);
  context->start();
  return context;
}

void drain_remote_object_removals() { detail::removal_queue::instance().drain(); }

remote_removal_drain_result drain_remote_object_removals(
  std::span<std::shared_ptr<io::ioctx> const> contexts, std::chrono::milliseconds timeout) noexcept
{
  try {
    // No queue: nothing was ever submitted (and none is created here).
    auto* const queue = detail::removal_queue::existing();
    if (queue == nullptr) { return {}; }
    auto const outcome = queue->drain(contexts, timeout);
    return {outcome.abandoned, outcome.timed_out};
  } catch (...) {
    return {0, true};
  }
}

}  // namespace cucascade
