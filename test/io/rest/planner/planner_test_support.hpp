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

// Shared helpers for the planner tests (ported from gpuio's tests/unit/planner/
// remote_test_support.hpp): a deterministic RNG. gpuio's allocation counter (a replacement of the
// global operator new for the whole test binary) is not ported: cucascade_io_tests is shared by
// every module, so the "allocates nothing" cases are not part of this port.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace planner_test {

/// splitmix64: tiny, fast, statistically good enough for jitter and property tests.
struct splitmix64 {
  std::uint64_t state;
  std::uint64_t operator()() noexcept
  {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  /// Uniform in [lo, hi] (inclusive).
  std::uint64_t between(std::uint64_t lo, std::uint64_t hi) noexcept
  {
    return lo + (hi == lo ? 0 : (*this)() % (hi - lo + 1));
  }
  /// Log-uniform in [lo, hi].
  std::uint64_t log_between(std::uint64_t lo, std::uint64_t hi) noexcept
  {
    double const a = std::log(static_cast<double>(lo));
    double const b = std::log(static_cast<double>(hi));
    double const v = std::exp(a + (b - a) * unit());
    auto const r   = static_cast<std::uint64_t>(v);
    return r < lo ? lo : r > hi ? hi : r;
  }
  /// Uniform in [0, 1).
  double unit() noexcept { return static_cast<double>((*this)() >> 11) * 0x1.0p-53; }
};

}  // namespace planner_test
