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

#include "io/rest/common.hpp"
#include "io/rest/errc.hpp"

#include <cstdio>
#include <cstdlib>

/**
 * @file common.hpp
 * @brief Vocabulary of the net layer (`cucascade::io::net`): the REST transport's error codes,
 *        `result<T>`, priorities and byte units under their gpuio names, and the debug-only
 *        invariant check of the remote path.
 */

/// Debug-only invariant check of the remote path (release builds compile it out; the pool keeps
/// counters for the same invariants so tests can assert them in any build type).
#ifndef CUCASCADE_NET_ASSERT
#ifdef NDEBUG
#define CUCASCADE_NET_ASSERT(cond) ((void)0)
#else
#define CUCASCADE_NET_ASSERT(cond)                                                             \
  ((cond)                                                                                      \
     ? (void)0                                                                                 \
     : (std::fprintf(                                                                          \
          stderr, "cucascade net: %s:%d: assertion `%s' failed\n", __FILE__, __LINE__, #cond), \
        std::abort()))
#endif
#endif

namespace cucascade::io::net {

using rest::errc;
using rest::GiB;
using rest::KiB;
using rest::MiB;
using rest::priority;
using rest::result;

}  // namespace cucascade::io::net
