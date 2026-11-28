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

#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::io::uring {

// Lets the context be destroyed on one of its reactors' workers (see ~templated_ioctx).
static_assert(reactor_reports_worker_thread<uring_reactor>);

uring_ioctx::uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx)
  : templated_ioctx<uring_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<uring_reactor>(ctx, "reactor-" + std::to_string(i++));
    })
{
}

namespace {

std::vector<std::unique_ptr<uring_reactor>> make_device_reactors(
  std::vector<uring_ioctx::device_group> const& groups, std::size_t readers)
{
  if (groups.empty()) { throw std::invalid_argument("uring_ioctx: no device groups"); }
  if (readers == 0) { throw std::invalid_argument("uring_ioctx: no read reactors per group"); }
  std::vector<std::unique_ptr<uring_reactor>> reactors;
  reactors.reserve((readers + 1) * groups.size());
  for (auto const& group : groups) {
    if (group.device < 0) { throw std::invalid_argument("uring_ioctx: negative device id"); }
    auto const name = "ur-g" + std::to_string(group.device);
    reactor_placement const where{group.device, group.cpus};
    for (std::size_t reader = 0; reader < readers; ++reader) {
      reactors.push_back(
        std::make_unique<uring_reactor>(group.ctx, name + "-r" + std::to_string(reader), where));
    }
    reactors.push_back(std::make_unique<uring_reactor>(group.ctx, name + "-wr", where));
    // The pool reports one staging block size (see templated_ioctx::staging_block_size).
    if (reactors.back()->staging_block_size() != reactors.front()->staging_block_size()) {
      throw std::invalid_argument("uring_ioctx: device groups differ in staging block size");
    }
  }
  return reactors;
}

}  // namespace

uring_ioctx::uring_ioctx(std::vector<device_group> const& groups, std::size_t readers_per_group)
  : templated_ioctx<uring_reactor>(make_device_reactors(groups, readers_per_group)),
    _readers(readers_per_group)
{
  _devices.reserve(groups.size());
  for (auto const& group : groups) {
    _devices.push_back(group.device);
  }
}

void uring_ioctx::start()
{
  if (_devices.empty()) {
    templated_ioctx<uring_reactor>::start();
    return;
  }
  std::lock_guard const lock(_writers_mutex);
  if (_started) return;
  for (std::size_t i = 0; i < _reactors.size(); ++i) {
    if (i % (_readers + 1) != _readers) _reactors[i]->start();  // the writers start on demand
  }
  _started        = true;
  _accepts_writes = true;
}

void uring_ioctx::shutdown() noexcept
{
  {
    std::lock_guard const lock(_writers_mutex);
    _accepts_writes = false;
  }
  templated_ioctx<uring_reactor>::shutdown();
}

std::vector<uring_reactor*> uring_ioctx::next_reactor(io_object_type const& object,
                                                      std::size_t n_slices,
                                                      io_op_type type,
                                                      int device_id)
{
  if (_devices.empty()) {
    return templated_ioctx<uring_reactor>::next_reactor(object, n_slices, type, device_id);
  }

  bool const writing = type == io_op_type::host_write || type == io_op_type::write_vector_async ||
                       type == io_op_type::flush_async;
  // A device request stays on its GPU; host I/O (or a GPU without a group) may
  // use every group.
  auto const it      = std::find(_devices.begin(), _devices.end(), device_id);
  auto const routed  = it != _devices.end();
  auto const first   = routed ? static_cast<std::size_t>(it - _devices.begin()) : 0;
  auto const last    = routed ? first + 1 : _devices.size();
  auto const stride  = _readers + 1;  // a group's readers, then its writer
  auto const members = writing ? 1 : _readers;
  auto const count   = (last - first) * members;

  // The least-backlogged of the role's reactors there, as the flat pool picks
  // them: the rotating start breaks ties.
  auto const start      = _next.fetch_add(1, std::memory_order_relaxed) % count;
  auto const reactor_at = [&](std::size_t distance) {
    auto const nth = (start + distance) % count;
    return _reactors[(first + nth / members) * stride + (writing ? _readers : nth % members)].get();
  };
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  ranked.reserve(count);
  for (std::size_t distance = 0; distance < count; ++distance) {
    ranked.emplace_back(reactor_at(distance)->queued_bytes(), distance);
  }
  auto const selected = std::min(count, dispatch_fanout());
  std::partial_sort(
    ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected), ranked.end());

  std::vector<uring_reactor*> result;
  result.reserve(selected);
  for (std::size_t i = 0; i < selected; ++i) {
    result.push_back(reactor_at(ranked[i].second));
  }
  if (writing) {
    std::lock_guard const lock(_writers_mutex);
    if (_accepts_writes) {
      for (auto* const writer : result) {
        writer->start();  // returns at once when its worker already runs
      }
    }
  }
  return result;
}

std::size_t uring_ioctx::slice_class([[maybe_unused]] io_object_type const& object,
                                     prepared_io_slice const& slice) const
{
  if (_devices.empty() || !slice.has_device_request()) return 0;
  return static_cast<std::size_t>(slice.d_buffer.device_id + 1);
}

std::vector<uring_reactor::gauges> uring_ioctx::reactor_gauges() noexcept
{
  std::vector<uring_reactor::gauges> out;
  try {
    out.reserve(_reactors.size());
    for (auto& reactor : _reactors) {
      out.push_back(reactor->take_gauges());
    }
  } catch (...) {
    out.clear();
  }
  return out;
}

}  // namespace cucascade::io::uring
