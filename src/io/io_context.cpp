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

#include <cucascade/error.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::io {
namespace {

using nvtx_range = nvtx3::scoped_range_in<libcucascade_domain>;

struct io_read_to_host_message {
  static constexpr char const* message{"io:read_to_host"};
};

}  // namespace

ioctx::ioctx()  = default;
ioctx::~ioctx() = default;

void ioctx::initialize_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  io::cache::config const& cache_config,
  std::shared_ptr<const cucascade::memory::topology_index> topology_index) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) {
    CUCASCADE_LOG_WARN("ioctx::initialize_cache() called but fs_cache already present");
    return;
  }
  if (!can_use_fs_cache()) {
    CUCASCADE_LOG_WARN(
      "ioctx::initialize_cache() called but backend does not support vector host read");
    return;
  }
  try {
    _cache = std::make_unique<cache::fs_cache>(
      reservation_manager, this, cache_config, std::move(topology_index));
  } catch (const std::exception& e) {
    CUCASCADE_LOG_ERROR("fs_cache construction failed: {}", e.what());
    _cache.reset();
  } catch (...) {
    CUCASCADE_LOG_ERROR("fs_cache construction failed: unknown error");
    _cache.reset();
  }
  // The reactors plan a fragmented fill's extent with
  // cache::fill_span(fill, chunk->offset, their own staging block size).  A
  // cache whose chunks are a different size makes every partial fill compute
  // the wrong extent -- with the larger staging block that is an out-of-bounds
  // write past the end of a pinned chunk.  The two are equal today only because
  // both read the same front HOST arena; refuse the cache rather than let a
  // future split of those resources corrupt the heap silently.
  if (_cache && staging_block_size() != 0 && staging_block_size() != _cache->chunk_size()) {
    CUCASCADE_LOG_ERROR(
      "ioctx::initialize_cache: backend {} stages in {}-byte blocks but the prefetching cache "
      "chunk is {} bytes; the two must match because fragmented fills are planned with the "
      "staging block size -- running without a cache",
      static_cast<int>(type()),
      staging_block_size(),
      _cache->chunk_size());
    _cache.reset();
  }
}

void ioctx::shutdown_cache() noexcept { _cache.reset(); }

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, open_hint /*hint*/)
{
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, std::uint64_t /*known_size*/)
{
  return create_io_object(std::move(path));
}

size_t ioctx::host_read(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, cache::cache_handle* handle)
{
  auto const& message =
    nvtx3::registered_string_in<libcucascade_domain>::get<io_read_to_host_message>();
  nvtx_range const read_range{message, nvtx3::payload{static_cast<std::uint64_t>(size)}};
  if (uses_fs_cache()) { return _cache->host_read(obj, offset, size, dst, handle); }
  return host_read_io(obj, offset, size, dst);
}

exec::semi_future<size_t> ioctx::host_read_async(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, cache::cache_handle* handle)
{
  if (uses_fs_cache()) { return _cache->host_read_async(obj, offset, size, dst, handle); }
  return host_read_async_io(obj, offset, size, dst);
}

exec::semi_future<size_t> ioctx::device_read_async(const io_object& obj,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst,
                                                   ::cuda::stream_ref stream,
                                                   cache::cache_handle* handle)
{
  if (uses_fs_cache()) { return _cache->device_read_async(obj, offset, size, dst, stream, handle); }
  return device_read_async_io(obj, offset, size, dst, stream);
}

exec::semi_future<size_t> ioctx::host_read_async_io(const io_object& obj,
                                                    size_t offset,
                                                    size_t size,
                                                    uint8_t* dst) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("host read destination is null");
    std::vector<prepared_io_slice> slices{prepared_io_slice{range{offset, size}, host_buffer{dst}}};
    return host_device_readv_async_io(obj, std::move(slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_read_async_io(const io_object& obj,
                                                      size_t offset,
                                                      size_t size,
                                                      uint8_t* dst,
                                                      ::cuda::stream_ref stream) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("device read destination is null");
    std::vector<prepared_io_slice> slices{
      prepared_io_slice{range{offset, size}, device_buffer{dst, stream}}};
    return host_device_readv_async_io(obj, std::move(slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::host_readv_async_io(const io_object& obj,
                                                     std::span<const slice> slices) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("host readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   host_buffer{current.dst});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_readv_async_io(const io_object& obj,
                                                       std::span<const slice> slices,
                                                       ::cuda::stream_ref stream) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("device readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   device_buffer{current.dst, stream});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

}  // namespace cucascade::io
