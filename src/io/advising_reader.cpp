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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/try.hpp>
#include <cucascade/io/advising_reader.hpp>

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace cucascade::io {

namespace {

// Bridge a semi_future into a real (promise-backed) std::future. Promise, std::future, and the
// type-erased terminal are all built before producer() may publish IO, so callback installation is
// a move-only, non-allocating handoff.
template <typename Producer>
std::future<std::size_t> bridge_semi_to_std(Producer&& producer)
{
  auto p              = std::make_shared<std::promise<std::size_t>>();
  auto fut            = p->get_future();
  using terminal_type = exec::invocable<void(exec::try_t<std::size_t>&&) &&>;
  terminal_type terminal{[p = std::move(p)](exec::try_t<std::size_t>&& t) mutable {
    if (t.has_exception()) {
      p->set_exception(std::move(t).exception());
    } else {
      p->set_value(std::move(t).value());
    }
  }};
  auto sf = std::forward<Producer>(producer)();
  std::move(sf).install_callback(std::move(terminal));
  return fut;
}

}  // namespace

advising_reader::advising_reader(std::shared_ptr<ioctx> context,
                                 std::shared_ptr<io_object> object,
                                 std::shared_ptr<column_admission> admission,
                                 std::string path,
                                 std::size_t min_advise_bytes)
  : _context(std::move(context)),
    _object(std::move(object)),
    _admission(std::move(admission)),
    _path(std::move(path)),
    _min_advise_bytes(min_advise_bytes)
{
}

advising_reader::~advising_reader() = default;

std::size_t advising_reader::size() const { return _object->size(); }

bool advising_reader::supports_device_read() const { return _context->supports_device_read(); }

cache::cache_handle* advising_reader::route(std::size_t offset, std::size_t size)
{
  if (_min_advise_bytes == 0 || size < _min_advise_bytes) { return nullptr; }
  auto* cache = _context->cache();
  if (cache == nullptr || !_context->can_use_fs_cache()) { return nullptr; }
  auto const advice = _admission->advise(_path, offset, size, _object->size());
  if (advice.ranges.empty()) { return nullptr; }

  // An empty handle means the cache had no new work to enqueue (every range coalesced with an
  // existing request): the read then goes through the cache without one.
  auto handle = cache->initiate_prefetching_request(
    *_object, advice.ranges, std::nullopt, advice.eviction_priority);
  if (!handle) { return nullptr; }
  std::ignore = cache->prepare(handle, true);

  auto owned = std::make_unique<cache::cache_handle>(std::move(handle));
  std::lock_guard lock(_mutex);
  _handles.push_back(std::move(owned));
  return _handles.back().get();
}

std::size_t advising_reader::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst)
{
  auto* handle = route(offset, size);
  if (_context->uses_fs_cache()) {
    return _context->cache()->host_read(*_object, offset, size, dst, handle);
  }
  return std::move(_context->host_read_async_io(*_object, offset, size, dst)).get();
}

std::future<std::size_t> advising_reader::host_read_async(std::size_t offset,
                                                          std::size_t size,
                                                          std::uint8_t* dst)
{
  auto* handle = route(offset, size);
  return bridge_semi_to_std([&] {
    if (_context->uses_fs_cache()) {
      return _context->cache()->host_read_async(*_object, offset, size, dst, handle);
    }
    return _context->host_read_async_io(*_object, offset, size, dst);
  });
}

std::future<std::size_t> advising_reader::device_read_async(std::size_t offset,
                                                            std::size_t size,
                                                            std::uint8_t* dst,
                                                            ::cuda::stream_ref stream)
{
  auto* handle = route(offset, size);
  return bridge_semi_to_std([&] {
    if (_context->uses_fs_cache()) {
      return _context->cache()->device_read_async(*_object, offset, size, dst, stream, handle);
    }
    return _context->device_read_async_io(*_object, offset, size, dst, stream);
  });
}

}  // namespace cucascade::io
