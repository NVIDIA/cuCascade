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

#include "io/rest/worker/op_request.hpp"

#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace cucascade::io::rest::detail {

/// Cut a contiguous read [offset, offset + size) into GET ranges (appended to @p out).
using cut_fn = std::function<void(range slice, std::vector<range>& out)>;

/**
 * @brief Expand the next slice of @p group into physical operations (appended to @p out).
 *
 * The cuCascade side of a REST worker: a slice is only cut once the transport is about to use
 * it, so a large request never pins credits or memory it cannot use yet.  Rules (unchanged from
 * the libcurl reactor, plus gpuio's GET sizing through @p cut):
 *  - contiguous host / staged device read: @p cut (gpuio P2-P8);
 *  - fragmented cache fill: one range per run of contiguous chunk fills, each chunk's extent
 *    being `cache::fill_span(fill, chunk->offset, block_size)`; one fill is never split; runs
 *    merge up to @p fill_target and never beyond max(@p max_segment, @p block_size);
 *  - read ranges are clamped to EOF;
 *  - a contiguous host read fully inside the object's footer stash completes on the spot;
 *  - a write slice is one operation (templated_ioctx already cut it at part boundaries);
 *  - a flush slice is one operation.
 * `add_tasks(N - 1)` is called before the operations are visible.  A slice that fails to expand
 * reports its error on the coordinator and runs its completion with `false`.  @p queued_bytes
 * loses the bytes of a slice completed or failed here; an operation keeps its logical bytes
 * until the transport takes it.
 *
 * @return the number of operations appended (0: the slice completed or failed here).
 */
std::size_t expand_slice(grouped_io_request& group,
                         tier t,
                         cut_fn const& cut,
                         std::size_t fill_target,
                         std::size_t block_size,
                         std::size_t max_segment,
                         std::atomic<std::size_t>& queued_bytes,
                         std::deque<std::unique_ptr<rest_io_op_request>>& out) noexcept;

/// Equal split of a contiguous read with no runt: `max(size / target, ceil(size / max))` pieces
/// of near-equal size, every piece at most @p max_segment (gpuio P7).
void split_even(range slice, std::size_t target, std::size_t max_segment, std::vector<range>& out);

}  // namespace cucascade::io::rest::detail
