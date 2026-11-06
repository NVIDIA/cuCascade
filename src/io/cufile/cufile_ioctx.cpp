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

#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {

cufile_ioctx::cufile_ioctx(std::size_t n_reactors,
                           std::shared_ptr<cufile_reactor::reactor_context> ctx)
  : templated_ioctx<cufile_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<cufile_reactor>(ctx, "cufile-" + std::to_string(i++));
    })
{
}

std::shared_ptr<io_object> cufile_ioctx::create_io_object(std::string path)
{
  return std::shared_ptr<io_object>(
    cufile_reactor::create_io_object(std::move(path), _config.use_odirect, _config.compat_mode));
}

path_info cufile_ioctx::path_capability(io_object const& object) const
{
  auto const* typed = dynamic_cast<cufile_io_object const*>(&object);
  if (typed == nullptr) {
    throw std::invalid_argument(
      "cufile_ioctx::path_capability: I/O object belongs to another backend");
  }
  return probe_object(*typed, _config);
}

std::vector<cufile_reactor*> cufile_ioctx::next_reactor(cufile_io_object const& object,
                                                        std::size_t n_slices,
                                                        io_op_type type,
                                                        int device_id)
{
  auto const count = _reactors.size();
  if (type != io_op_type::host_vector_async || n_slices <= 2 || count <= 2) {
    return templated_ioctx<cufile_reactor>::next_reactor(object, n_slices, type, device_id);
  }
  // A vectored read: every reactor, least backlogged first.
  auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  ranked.reserve(count);
  for (std::size_t distance = 0; distance < count; ++distance) {
    ranked.emplace_back(_reactors[(start + distance) % count]->queued_bytes(), distance);
  }
  std::sort(ranked.begin(), ranked.end());
  std::vector<cufile_reactor*> result;
  result.reserve(count);
  for (auto const& [bytes, distance] : ranked) {
    result.push_back(_reactors[(start + distance) % count].get());
  }
  return result;
}

}  // namespace cucascade::io::cufile
