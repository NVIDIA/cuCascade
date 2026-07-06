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

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/prefetching_cache.hpp>
#include <cucascade/io/io_context.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

namespace cucascade::io {

ioctx::ioctx()  = default;
ioctx::~ioctx() = default;

void ioctx::initialize_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  io::cache::config const& cache_config,
  std::shared_ptr<const cucascade::memory::topology_index> topology_index) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) { return; }
  if (!can_use_prefetching_cache()) { return; }
  try {
    _cache = std::make_unique<cache::prefetching_cache>(
      reservation_manager, this, cache_config, std::move(topology_index));
  } catch (const std::exception& e) {
    _cache.reset();
  } catch (...) {
    _cache.reset();
  }
}

void ioctx::shutdown_cache() noexcept { _cache.reset(); }

size_t ioctx::host_read(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, cache::prefetching_handle* handle)
{
  if (uses_prefetching_cache()) { return _cache->host_read(obj, offset, size, dst, handle); }
  return host_read_io(obj, offset, size, dst);
}

exec::semi_future<size_t> ioctx::host_read_async(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, cache::prefetching_handle* handle)
{
  if (uses_prefetching_cache()) { return _cache->host_read_async(obj, offset, size, dst, handle); }
  return host_read_async_io(obj, offset, size, dst);
}

exec::semi_future<size_t> ioctx::device_read_async(const io_object& obj,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst,
                                                   rmm::cuda_stream_view stream,
                                                   cache::prefetching_handle* handle)
{
  if (uses_prefetching_cache()) {
    return _cache->device_read_async(obj, offset, size, dst, stream, handle);
  }
  return device_read_async_io(obj, offset, size, dst, stream);
}

}  // namespace cucascade::io
