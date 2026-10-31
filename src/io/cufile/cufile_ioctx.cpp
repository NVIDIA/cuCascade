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
  : cufile_ioctx(reactor_layout::unplaced(n_reactors), std::move(ctx))
{
}

cufile_ioctx::cufile_ioctx(reactor_layout const& layout,
                           std::shared_ptr<cufile_reactor::reactor_context> ctx)
  : templated_ioctx<cufile_reactor>(
      layout, [ctx = std::move(ctx), i = 0](reactor_placement const& placement) mutable {
        return std::make_unique<cufile_reactor>(ctx, "cufile-" + std::to_string(i++), placement);
      })
{
}

std::shared_ptr<io_object> cufile_ioctx::create_io_object(std::string path)
{
  return std::shared_ptr<io_object>(
    cufile_reactor::create_io_object(std::move(path), _config.use_odirect, _config.compat_mode));
}

std::shared_ptr<io_object> cufile_ioctx::create_io_object_for_write(std::string path,
                                                                    write_open_options options)
{
  return std::shared_ptr<io_object>(cufile_reactor::create_io_object_for_write(
    std::move(path), options, _config.use_odirect, _config.compat_mode));
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
  if (type == io_op_type::write_vector_async && device_id < 0 && _reactors.size() > 1) {
    // A host-only vectored write (device_id stays unresolved without device
    // slices) goes to one reactor: its page-cache writes with early writeback
    // already keep the device busy, and interleaving writers of one file only
    // contends on its inode.
    auto selected =
      templated_ioctx<cufile_reactor>::next_reactor(object, n_slices, type, device_id);
    if (selected.size() > 1) selected.resize(1);
    return selected;
  }
  auto const candidates = routable_reactors(device_id);
  if (type != io_op_type::host_vector_async || n_slices <= 2 || candidates.size() <= 2) {
    return templated_ioctx<cufile_reactor>::next_reactor(object, n_slices, type, device_id);
  }
  // A vectored read: every reactor it may use, least backlogged first.
  return least_backlogged(candidates, candidates.size());
}

}  // namespace cucascade::io::cufile
