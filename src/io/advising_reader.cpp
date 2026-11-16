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

#include <cucascade/io/advising_reader.hpp>
#include <cucascade/io/details/std_future.hpp>
#include <cucascade/log/logging.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace cucascade::io {

advising_reader::advising_reader(std::shared_ptr<ioctx> context,
                                 std::shared_ptr<io_object> object,
                                 std::shared_ptr<column_admission> admission,
                                 std::size_t min_advise_bytes)
  : _context(std::move(context)),
    _object(std::move(object)),
    _admission(std::move(admission)),
    _min_advise_bytes(min_advise_bytes)
{
}

advising_reader::~advising_reader() = default;

std::size_t advising_reader::size() const { return _object->size(); }

bool advising_reader::supports_device_read() const { return _context->supports_device_read(); }

std::unique_ptr<cache::cache_handle> advising_reader::route(std::size_t offset, std::size_t size)
{
  if (_min_advise_bytes == 0 || size < _min_advise_bytes) { return nullptr; }
  auto* cache = _context->cache();
  if (cache == nullptr || !_context->can_use_fs_cache()) { return nullptr; }
  auto const& path  = _object->object_path();
  auto const advice = _admission->advise(path, offset, size, _object->size());
  if (advice.ranges.empty()) { return nullptr; }

  // An empty handle means the cache had no new work to enqueue (every range coalesced with an
  // existing request): the read then goes through the cache without one.
  auto handle = cache->initiate_prefetching_request(
    *_object, advice.ranges, std::nullopt, advice.eviction_priority);
  if (!handle) { return nullptr; }
  if (auto const prepared = cache->prepare(handle, true);
      prepared != cache::prepare_result::prepared) {
    CUCASCADE_LOG_DEBUG("advising_reader: {} [{}, +{}) advised but not staged ({})",
                        path,
                        offset,
                        size,
                        static_cast<int>(prepared));
  }
  return std::make_unique<cache::cache_handle>(std::move(handle));
}

std::size_t advising_reader::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst)
{
  auto const handle = route(offset, size);
  return std::move(_context->host_read_async(*_object, offset, size, dst, handle.get())).get();
}

std::future<std::size_t> advising_reader::host_read_async(std::size_t offset,
                                                          std::size_t size,
                                                          std::uint8_t* dst)
{
  auto const handle = route(offset, size);
  return bridge_semi_to_std(
    [&] { return _context->host_read_async(*_object, offset, size, dst, handle.get()); });
}

std::future<std::size_t> advising_reader::device_read_async(std::size_t offset,
                                                            std::size_t size,
                                                            std::uint8_t* dst,
                                                            ::cuda::stream_ref stream)
{
  auto const handle = route(offset, size);
  return bridge_semi_to_std(
    [&] { return _context->device_read_async(*_object, offset, size, dst, stream, handle.get()); });
}

}  // namespace cucascade::io
