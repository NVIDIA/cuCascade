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

#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

namespace cucascade::io::uring {

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief io_uring-backed ioctx. Thin specialisation of
 *        @c templated_ioctx<uring_reactor>.
 */
class uring_ioctx : public templated_ioctx<uring_reactor> {
 public:
  /// Build the context over one @c uring_reactor sharing @p ctx (it carries the
  /// @c config and the pinned bounce-staging resource, which must outlive this
  /// ioctx).  @p n_runner_threads is the number of runner threads @c start()
  /// spawns (each owns its own ring and 64 MiB of pinned staging); callers may
  /// additionally drive the context with @c run / @c run_for / @c run_until.
  uring_ioctx(size_t n_runner_threads, std::shared_ptr<uring_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::uring; }
};

}  // namespace cucascade::io::uring
