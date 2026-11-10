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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/try.hpp>

#include <cstddef>
#include <future>
#include <memory>
#include <utility>

namespace cucascade::io {

/**
 * @brief Bridge a semi_future into a real (promise-backed) std::future.
 *
 * Promise, std::future, and the exact type-erased terminal are all built before @p producer may
 * publish IO, so callback installation is a move-only, non-allocating handoff.
 *
 * @param producer Callable that starts the IO and returns its @c exec::semi_future<std::size_t>.
 */
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

}  // namespace cucascade::io
