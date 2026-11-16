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

#pragma once

#include <cucascade/io/types.hpp>

#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace cucascade::io {

/**
 * @brief Remembers opened @c io_object handles by path, so opening an object a second time needs
 * no request.
 *
 * Opening an object-store object resolves its size, its version tag and, for a parquet file, its
 * footer, with one request. A scan opens the same files for every query, so without this the
 * request is paid again by each one. A kept handle also keeps its version tag, which the
 * prefetching cache needs to share cached chunks between opens: an object opened with only a known
 * size has no tag, and its chunks are cached for that open alone.
 *
 * Objects are assumed not to change while the cache exists, as the prefetching cache itself
 * assumes for the data it holds.
 *
 * Thread-safe.
 */
class object_cache {
 public:
  /**
   * @brief The handle kept for @p path, opening it with @p open on the first call.
   *
   * Concurrent first calls for one path share one call of @p open. If it throws, every one of them
   * rethrows its exception and nothing is kept, so a later call opens again.
   */
  template <typename Open>
  std::shared_ptr<io_object> find_or_open(std::string const& path, Open&& open)
  {
    std::promise<std::shared_ptr<io_object>> opening;
    std::shared_future<std::shared_ptr<io_object>> opened;
    bool first = false;
    {
      std::lock_guard lock(_mutex);
      auto const [it, inserted] = _objects.try_emplace(path);
      if (inserted) { it->second = opening.get_future().share(); }
      opened = it->second;
      first  = inserted;
    }
    if (!first) { return opened.get(); }
    try {
      opening.set_value(std::forward<Open>(open)());
    } catch (...) {
      {
        std::lock_guard lock(_mutex);
        _objects.erase(path);
      }
      opening.set_exception(std::current_exception());
    }
    return opened.get();
  }

  /// Number of handles kept or being opened.
  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard lock(_mutex);
    return _objects.size();
  }

 private:
  mutable std::mutex _mutex;
  std::unordered_map<std::string, std::shared_future<std::shared_ptr<io_object>>> _objects;
};

}  // namespace cucascade::io
