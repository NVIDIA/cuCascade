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

#include <cucascade/cudf/advising_datasource.hpp>

#include <optional>
#include <tuple>
#include <utility>

namespace cucascade::io {

advising_datasource::advising_datasource(std::unique_ptr<cucascade::io::datasource> base,
                                         std::shared_ptr<column_admission> admission,
                                         std::string path,
                                         std::size_t min_advise_bytes)
  : _base(std::move(base)),
    _admission(std::move(admission)),
    _path(std::move(path)),
    _min_advise_bytes(min_advise_bytes)
{
}

size_t advising_datasource::size() const { return _base->size(); }

bool advising_datasource::supports_device_read() const { return _base->supports_device_read(); }

bool advising_datasource::is_device_read_preferred(size_t size) const
{
  return _base->is_device_read_preferred(size);
}

cucascade::io::datasource& advising_datasource::route(size_t offset, size_t size)
{
  if (_min_advise_bytes == 0 || size < _min_advise_bytes) { return *_base; }
  auto const advice = _admission->advise(_path, offset, size, _base->size());
  if (advice.ranges.empty()) { return *_base; }

  auto reader = _base->duplicate();
  reader->fadvise(advice.ranges, std::nullopt, advice.eviction_priority);
  std::ignore = reader->prepare_prefetch(true);
  std::lock_guard lock(_mutex);
  _readers.push_back(std::move(reader));
  return *_readers.back();
}

size_t advising_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return route(offset, size).host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> advising_datasource::host_read(size_t offset,
                                                                             size_t size)
{
  return route(offset, size).host_read(offset, size);
}

std::future<size_t> advising_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return route(offset, size).host_read_async(offset, size, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> advising_datasource::host_read_async(
  size_t offset, size_t size)
{
  return route(offset, size).host_read_async(offset, size);
}

std::unique_ptr<cudf::io::datasource::buffer> advising_datasource::device_read(
  size_t offset, size_t size, cudf_stream_type stream)
{
  return route(offset, size).device_read(offset, size, stream);
}

size_t advising_datasource::device_read(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cudf_stream_type stream)
{
  return route(offset, size).device_read(offset, size, dst, stream);
}

std::future<size_t> advising_datasource::device_read_async(size_t offset,
                                                           size_t size,
                                                           uint8_t* dst,
                                                           cudf_stream_type stream)
{
  return route(offset, size).device_read_async(offset, size, dst, stream);
}

}  // namespace cucascade::io
