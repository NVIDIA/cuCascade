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

/**
 * @file concurrent_queue.hpp
 * @brief Single include point for moodycamel's lock-free MPMC queues.
 *
 * The io reactors and prefetching cache use moodycamel's `ConcurrentQueue` /
 * `BlockingConcurrentQueue`. This header decouples that code from *which*
 * moodycamel it links against: standalone builds fetch cameron314's upstream
 * (namespace `moodycamel`), while an embedding host may provide its own vendored
 * copy under a different namespace (e.g. duckdb's fork is `duckdb_moodycamel`,
 * used by sirius).
 *
 * `CUCASCADE_MOODYCAMEL_NAMESPACE` selects the namespace and defaults to
 * `moodycamel`; the CMake build propagates it as a public compile definition so
 * the library and every consumer of these headers agree on the same value. Use
 * the `cucascade::io::concurrent_queue` / `cucascade::io::blocking_concurrent_queue`
 * aliases below rather than naming the moodycamel namespace directly.
 */

#ifndef CUCASCADE_MOODYCAMEL_NAMESPACE
#define CUCASCADE_MOODYCAMEL_NAMESPACE moodycamel
#endif

#include <blockingconcurrentqueue.h>
#include <concurrentqueue.h>

namespace cucascade::io {

namespace detail {
namespace moodycamel_ns = ::CUCASCADE_MOODYCAMEL_NAMESPACE;
}  // namespace detail

/// Lock-free MPMC queue (alias for moodycamel::ConcurrentQueue).
template <typename T, typename Traits = detail::moodycamel_ns::ConcurrentQueueDefaultTraits>
using concurrent_queue = detail::moodycamel_ns::ConcurrentQueue<T, Traits>;

/// Blocking lock-free MPMC queue (alias for moodycamel::BlockingConcurrentQueue).
template <typename T, typename Traits = detail::moodycamel_ns::ConcurrentQueueDefaultTraits>
using blocking_concurrent_queue = detail::moodycamel_ns::BlockingConcurrentQueue<T, Traits>;

}  // namespace cucascade::io
