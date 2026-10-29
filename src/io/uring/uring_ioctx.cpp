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

#include <memory>
#include <string>
#include <vector>

namespace cucascade::io::uring {

uring_ioctx::uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx)
  : uring_ioctx(reactor_layout::unplaced(n_reactors), std::move(ctx))
{
}

uring_ioctx::uring_ioctx(reactor_layout const& layout,
                         std::shared_ptr<uring_reactor::reactor_context> ctx)
  : templated_ioctx<uring_reactor>(
      layout, [ctx = std::move(ctx), i = 0](reactor_placement const& placement) mutable {
        return std::make_unique<uring_reactor>(ctx, "reactor-" + std::to_string(i++), placement);
      })
{
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
