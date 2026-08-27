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

#include <cstddef>

namespace cucascade::io::cache {

struct config {
  // Maximum number of in-flight prefetch IO *tasks* (not chunks): the prefetch
  // loop reserves one unit per dispatched read task and releases it on
  // completion, so this caps how many prefetch reads are outstanding at once.
  size_t inflight_io_chunk_budget = 16;
  double min_prefetching_budget_fraction{0.05};
  double eviction_threshold_fraction{0.6};
  bool dispose_after_use = false;
};

}  // namespace cucascade::io::cache
