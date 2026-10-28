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

#include <cucascade/io/config.hpp>
#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#endif
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/object_store_config.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cctype>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace cucascade::io {

namespace {

/// First HOST-tier pinned staging resource, or nullptr when none exists.  The
/// uring / rest reactors stage device reads through it and size their bounce
/// slots from its block size.
cucascade::memory::fixed_size_host_memory_resource* first_host_resource(
  cucascade::memory::memory_reservation_manager& rm)
{
  auto host_spaces = rm.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_spaces.empty()) { return nullptr; }
  return host_spaces.front()->get_memory_resource_of<cucascade::memory::Tier::HOST>();
}

}  // namespace

using scheme_checker_type = io_context_registry::scheme_checker_type;
using factory_type        = io_context_registry::factory_type;

#ifdef CUCASCADE_HAS_CUFILE
factory_type make_cufile_ioctx_factory()
{
  return [](const io_config& config) -> std::shared_ptr<ioctx> {
    try {
      auto ctx = std::make_shared<cufile::cufile_reactor::reactor_context>(config.cufile);
      return std::make_shared<cufile::cufile_ioctx>(config.cufile_n_reactors, std::move(ctx));
    } catch (const std::exception& e) {
      CUCASCADE_LOG_ERROR("make_cufile_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}
#endif

factory_type make_uring_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager)
{
  return [&reservation_manager](const io_config& config) -> std::shared_ptr<ioctx> {
    try {
      auto* host_mr = first_host_resource(reservation_manager);
      if (host_mr == nullptr) {
        CUCASCADE_LOG_ERROR(
          "make_uring_ioctx_factory: no HOST-tier memory resource for the reactor staging");
        return nullptr;
      }
      // One reactor_context shares config and the pinned staging resource.
      auto ctx = std::make_shared<uring::uring_reactor::reactor_context>(config.uring, host_mr);
      return std::make_shared<uring::uring_ioctx>(config.uring_n_reactors, std::move(ctx));
    } catch (const std::exception& e) {
      CUCASCADE_LOG_ERROR("make_uring_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}

factory_type make_rest_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager)
{
  return [&reservation_manager](const io_config& config) -> std::shared_ptr<ioctx> {
    try {
      auto authorizer = rest::s3::make_authorizer(config.object_store);
      if (!authorizer) {
        CUCASCADE_LOG_WARN(
          "make_rest_ioctx_factory: object store not configured (endpoint / credentials / "
          "region missing); REST backend disabled");
        return nullptr;
      }
      auto* host_mr = first_host_resource(reservation_manager);
      auto rest_cfg = config.rest;
      // The object store owns the endpoint and its TLS trust; the reactor's
      // connections must verify against the same CA bundle / policy the
      // authorizer signs for, so source these from object_store rather than rest
      // config (they are applied on top of rest_cfg.tls).
      rest_cfg.ca_bundle_path = config.object_store.ca_bundle_path;
      rest_cfg.tls_verify     = config.object_store.tls_verify;
      auto ctx                = std::make_shared<rest::rest_reactor::reactor_context>(
        std::move(rest_cfg), std::move(authorizer), host_mr);
      return std::make_shared<rest::rest_ioctx>(config.rest_n_reactors, std::move(ctx));
    } catch (const std::exception& e) {
      CUCASCADE_LOG_ERROR("make_rest_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

io_context_registry::io_context_registry(
  config_type config, cucascade::memory::memory_reservation_manager& reservation_manager)
  : _config(std::move(config)),
    _reservation_manager(reservation_manager),
    _prefer_cufile(_config.backend == io_backend::cufile)
{
#ifndef CUCASCADE_HAS_CUFILE
  if (_prefer_cufile) {
    throw std::invalid_argument(
      "io_context_registry: io_backend::cufile requested, but this cuCascade build has no cuFile "
      "backend (cufile.h was not found at configure time)");
  }
#endif
  // uring / rest claim paths via their reactor's static supports() (local
  // files and s3:// URLs respectively).  There is no catch-all: a path no
  // backend claims resolves to nothing.  cufile claims local regular files like
  // uring; lookup_path decides between them by the configured backend, never by
  // iteration order.
#ifdef CUCASCADE_HAS_CUFILE
  _entries.emplace(
    io_context_type::cufile,
    entry{io_context_type::cufile, &cufile::cufile_reactor::supports, make_cufile_ioctx_factory()});
#endif
  _entries.emplace(io_context_type::uring,
                   entry{io_context_type::uring,
                         &uring::uring_reactor::supports,
                         make_uring_ioctx_factory(_reservation_manager)});
  _entries.emplace(io_context_type::restful,
                   entry{io_context_type::restful,
                         &rest::rest_reactor::supports,
                         make_rest_ioctx_factory(_reservation_manager)});
}

void io_context_registry::register_ioctx(io_context_type type,
                                         scheme_checker_type checker,
                                         factory_type factory)
{
  if (!checker) throw std::invalid_argument("datasource_registry: null scheme checker");
  if (!factory) throw std::invalid_argument("datasource_registry: null factory");
  std::lock_guard lk{_mtx};
  _entries[type] = {type, std::move(checker), std::move(factory)};
}

void io_context_registry::replace_ioctx(io_context_type old_type,
                                        io_context_type new_type,
                                        scheme_checker_type checker,
                                        factory_type factory)
{
  if (!checker) {
    throw std::invalid_argument("datasource_registry: replace_ioctx: null scheme checker");
  }
  if (!factory) { throw std::invalid_argument("datasource_registry: replace_ioctx: null factory"); }
  std::lock_guard lk{_mtx};
  if (_lookup_latched.load(std::memory_order_acquire)) {
    throw std::logic_error(
      "datasource_registry: replace_ioctx after the first lookup_path (bootstrap-only)");
  }
  if (!_entries.contains(old_type)) {
    throw std::invalid_argument("datasource_registry: replace_ioctx: old type not registered");
  }
  if (_entries.contains(new_type)) {
    throw std::invalid_argument("datasource_registry: replace_ioctx: new type already registered");
  }
  // Strong guarantee: the emplace is the only throwing step and precedes the
  // erase; erase by KEY, not by a pre-emplace iterator (emplace may rehash).
  _entries.emplace(new_type, entry{new_type, std::move(checker), std::move(factory)});
  _entries.erase(old_type);
}

std::optional<io_context_type> io_context_registry::lookup_path(
  std::string_view path) const noexcept
{
  std::shared_lock lk{_mtx};
  _lookup_latched.store(true, std::memory_order_release);
  // Both cufile and uring answer true for a local file, and _entries iterates in
  // unspecified order, so iteration order must not decide between them: cufile
  // is considered only when the config asked for it, and then it displaces
  // uring.  The rules, per configured backend:
  //   * cufile -- skip uring for local files (cufile wins); rest keeps s3://.
  //   * native -- skip cufile for local files (uring wins); rest keeps s3://.
  for (const auto& [type, entry] : _entries) {
    if (!entry.checker(path)) continue;
    if (_prefer_cufile && type == io_context_type::uring) { continue; }
    if (!_prefer_cufile && type == io_context_type::cufile) { continue; }
    return type;
  }
  return std::nullopt;
}

std::shared_ptr<ioctx> io_context_registry::make_ioctx(io_context_type type) const noexcept
{
  std::shared_lock lk{_mtx};
  auto it = _entries.find(type);
  if (it == _entries.end()) return nullptr;
  return it->second.factory(_config);
}

void io_context_registry::clear()
{
  std::unique_lock lk{_mtx};
  _entries.clear();
}

}  // namespace cucascade::io
